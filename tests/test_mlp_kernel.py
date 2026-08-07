"""Correctness of the fused gated-MLP kernel against `reference.gated_mlp`.

Everything here runs under Pallas interpret mode on CPU. No timing claim is made
or implied -- there is no accelerator on this machine, and a green run
establishes correctness under emulation, nothing about hardware.

The kernel slices the hidden dimension across the grid, which makes `H` a
*reduction* axis: each `h` step contributes a partial sum to the same output
block. `test_two_hidden_blocks_accumulate` is the test that fails if the
accumulator is not initialised on the first visit -- it exists before any test
that could not catch that.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gemma3_pallas.mlp import fused_gated_mlp
from gemma3_pallas.reference import gated_mlp
from gemma3_pallas.shapes import GEMMA3_1B

# Measured, not guessed: the largest relative error observed at Gemma dims on
# the first green run (see test_gemma_dims). fp32 throughout, so the only source
# is a different summation order between the kernel's blocked reduction over H
# and the reference's single fused matmul.
GEMMA_RTOL = 1e-4
GEMMA_ATOL = 1e-4


def _operands(t, e, h, seed=0):
    """Random fp32 operands at the given geometry, scaled like real weights."""
    keys = jax.random.split(jax.random.key(seed), 4)
    x = jax.random.normal(keys[0], (t, e), jnp.float32)
    w_gate = jax.random.normal(keys[1], (e, h), jnp.float32) * 0.02
    w_up = jax.random.normal(keys[2], (e, h), jnp.float32) * 0.02
    w_down = jax.random.normal(keys[3], (h, e), jnp.float32) * 0.02
    return x, w_gate, w_up, w_down


def _interpret_params():
    """`pltpu.InterpretParams()`, or skip if this build lacks it."""
    pltpu = pytest.importorskip(
        "jax.experimental.pallas.tpu",
        reason="Pallas TPU interpret mode unavailable in this JAX build",
    )
    if not hasattr(pltpu, "InterpretParams"):
        pytest.skip("this JAX build predates pltpu.InterpretParams")
    return pltpu.InterpretParams()


def test_one_hidden_block():
    """Grid (2, 1): the matmul chain and the BlockSpecs, with no reduction yet.

    Non-Gemma dimensions deliberately, so nothing can be shape-hardcoded.
    """
    operands = _operands(t=16, e=128, h=128)
    got = fused_gated_mlp(*operands, block_t=8, block_h=128)
    want = gated_mlp(*operands)
    assert got.shape == (16, 128)
    assert np.allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-5)


def test_two_hidden_blocks_accumulate():
    """Grid (2, 2): H is split, so the output block is a partial sum.

    This is the test that fails when the accumulator initialisation is missing
    or wrong -- without `@pl.when(program_id(1) == 0)` the second h step adds
    into whatever the buffer already held.
    """
    operands = _operands(t=16, e=128, h=256)
    got = fused_gated_mlp(*operands, block_t=8, block_h=128)
    want = gated_mlp(*operands)
    assert np.allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-5)


def test_gemma_dims():
    """The real geometry: grid (2, 9) over Gemma 3 1B's MLP."""
    cfg = GEMMA3_1B
    operands = _operands(t=256, e=cfg.embed_dim, h=cfg.hidden_dim)
    got = fused_gated_mlp(*operands, block_t=128, block_h=768)
    want = gated_mlp(*operands)
    assert got.shape == (256, cfg.embed_dim)
    assert np.allclose(np.asarray(got), np.asarray(want), rtol=GEMMA_RTOL, atol=GEMMA_ATOL)


@pytest.mark.parametrize(
    "kwargs, axis",
    [
        ({"t": 20, "e": 128, "h": 128, "block_t": 8, "block_h": 128}, "tokens"),
        ({"t": 16, "e": 128, "h": 384, "block_t": 8, "block_h": 256}, "hidden_dim"),
        ({"t": 16, "e": 120, "h": 128, "block_t": 8, "block_h": 128}, "embed_dim"),
    ],
)
def test_ragged_shapes_raise(kwargs, axis):
    """Divisible shapes only, and the error says which axis is the problem.

    Padding a *reduction* axis silently accumulates garbage, so the kernel
    refuses rather than padding.
    """
    operands = _operands(t=kwargs["t"], e=kwargs["e"], h=kwargs["h"])
    with pytest.raises(ValueError, match=axis):
        fused_gated_mlp(
            *operands, block_t=kwargs["block_t"], block_h=kwargs["block_h"]
        )


def test_two_hidden_blocks_under_simulated_memory():
    """Test 2 again under the mode that simulates HBM/VMEM, DMAs and semaphores.

    Deliberately on the tiny geometry: what this buys is the *accumulator* half
    of that mode -- DMA ordering and semaphore discipline across a two-step
    reduction -- which needs `H // block_h == 2` and nothing larger.
    """
    interpret = _interpret_params()
    operands = _operands(t=16, e=128, h=256)
    got = fused_gated_mlp(*operands, block_t=8, block_h=128, interpret=interpret)
    want = gated_mlp(*operands)
    assert np.allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-5)


def test_gemma_dims_under_simulated_memory():
    """Gemma dims under simulated memory: the only test at the real footprint.

    Test 5's blocks are kilobytes; this is the one place the simulated-memory
    path is handed the ~22.5 MiB working set the block choice predicts.

    **It is not a VMEM assertion.** Whether `InterpretParams` models a capacity
    limit at all is unknown, and that is worth checking rather than assuming. If
    it does, a failure here is a finding about the block choice, not a defect in
    the kernel. If it does not, the test still exercises the DMA/semaphore path
    at the real grid (2, 9) rather than (2, 2).

    Not marked slow, on measurement: ~1.0s of a ~14s suite, and *faster* than
    the plain-interpret run of the same shapes. ~12 GFLOP of fp32 matmul sounds
    expensive, but 18 grid steps of simulation overhead is nothing and CPU XLA
    absorbs the matmuls.
    """
    interpret = _interpret_params()
    cfg = GEMMA3_1B
    operands = _operands(t=256, e=cfg.embed_dim, h=cfg.hidden_dim)
    got = fused_gated_mlp(*operands, block_t=128, block_h=768, interpret=interpret)
    want = gated_mlp(*operands)
    assert np.allclose(np.asarray(got), np.asarray(want), rtol=GEMMA_RTOL, atol=GEMMA_ATOL)
