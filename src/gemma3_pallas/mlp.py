"""Fused gated-MLP (GeGLU) Pallas kernel.

One kernel for what `reference.gated_mlp` writes as three matmuls and a
multiply: `(gelu(x @ w_gate) * (x @ w_up)) @ w_down`. The two intermediates,
`[tokens, hidden_dim]` each, never reach HBM -- they are produced and consumed
inside a single grid step, which is the whole point of fusing.

fp32 throughout, matching the reference exactly so a mismatch is a bug rather
than rounding. RMSNorm is applied *outside*, per `reference.py`'s stated kernel
boundary.

The grid is `(tokens // block_t, hidden_dim // block_h)` with the hidden axis
**minormost**, and that ordering is load-bearing rather than stylistic -- see
`_fused_gated_mlp_kernel` below.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from .reference import gelu_tanh

# TPU block-shape rule: the last two dimensions of a block must equal the array
# dimension or be divisible by 8 and 128 respectively.
_SUBLANE = 8
_LANE = 128


def _fused_gated_mlp_kernel(x_ref, wg_ref, wu_ref, wd_ref, o_ref):
    """One grid step: a `block_h`-wide slice of the hidden dimension.

    `hidden_dim` is a *reduction* axis -- `w_down` contracts it away -- so each
    `h` step produces a partial sum of the same output block and accumulates
    into it.
    """
    # Initialise on the first visit only. Zeroing unconditionally would pass a
    # grid with one h step and discard every partial sum but the last on a grid
    # with more; not zeroing at all accumulates into an uninitialised buffer.
    @pl.when(pl.program_id(1) == 0)
    def _():
        o_ref[...] = jnp.zeros_like(o_ref)

    gate = gelu_tanh(x_ref[...] @ wg_ref[...])  # [block_t, block_h]
    up = x_ref[...] @ wu_ref[...]  # [block_t, block_h]
    o_ref[...] += (gate * up) @ wd_ref[...]  # [block_t, embed_dim]


def fused_gated_mlp(
    x: jax.Array,
    w_gate: jax.Array,
    w_up: jax.Array,
    w_down: jax.Array,
    *,
    block_t: int,
    block_h: int,
    interpret=True,
) -> jax.Array:
    """Gemma's GeGLU feed-forward block as a single Pallas kernel.

    x       [tokens, embed_dim]
    w_gate  [embed_dim, hidden_dim]   up-projection through `gelu_tanh` -- the gate
    w_up    [embed_dim, hidden_dim]   up-projection with no activation
    w_down  [hidden_dim, embed_dim]   down-projection back to the residual stream

    Returns [tokens, embed_dim].

    `block_t` rows of `tokens` and `block_h` columns of `hidden_dim` are resident
    per grid step; `embed_dim` is never blocked. Divisible shapes only -- a
    ragged block raises `ValueError` naming the axis, because padding a
    reduction axis would silently accumulate garbage into the result.

    `interpret` is passed through to `pallas_call`: `True` for plain functional
    emulation, `pltpu.InterpretParams()` for simulated HBM/VMEM, DMAs and
    semaphores, `False` on real hardware.
    """
    x, w_gate, w_up, w_down = (a.astype(jnp.float32) for a in (x, w_gate, w_up, w_down))
    tokens, embed_dim, hidden_dim = _validate(x, w_gate, w_up, w_down, block_t, block_h)

    return pl.pallas_call(
        _fused_gated_mlp_kernel,
        # `h` last: the output block's index_map is invariant in `h`, so with `h`
        # innermost its visits are consecutive -- the block stays resident and is
        # written back once. Swapping the axes would revisit output block 0 at
        # (h=0, t=0) and again at (h=1, t=0), which is the undocumented
        # `RevisitMode.ANY` territory that lowering rejects when buffer_count > 1.
        grid=(tokens // block_t, hidden_dim // block_h),
        in_specs=[
            pl.BlockSpec((block_t, embed_dim), lambda i, j: (i, 0)),  # x
            pl.BlockSpec((embed_dim, block_h), lambda i, j: (0, j)),  # w_gate
            pl.BlockSpec((embed_dim, block_h), lambda i, j: (0, j)),  # w_up
            pl.BlockSpec((block_h, embed_dim), lambda i, j: (j, 0)),  # w_down
        ],
        out_specs=pl.BlockSpec((block_t, embed_dim), lambda i, j: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((tokens, embed_dim), jnp.float32),
        interpret=interpret,
    )(x, w_gate, w_up, w_down)


def _validate(x, w_gate, w_up, w_down, block_t, block_h) -> tuple[int, int, int]:
    """Reject anything the kernel would otherwise get wrong quietly."""
    if x.ndim != 2 or w_gate.ndim != 2 or w_up.ndim != 2 or w_down.ndim != 2:
        raise ValueError("all operands must be rank 2")

    tokens, embed_dim = x.shape
    if w_gate.shape != w_up.shape:
        raise ValueError(f"w_gate {w_gate.shape} and w_up {w_up.shape} must have the same shape")
    if w_gate.shape[0] != embed_dim:
        raise ValueError(
            f"w_gate {w_gate.shape} does not start at x's embed_dim {embed_dim}"
        )
    hidden_dim = w_gate.shape[1]
    if w_down.shape != (hidden_dim, embed_dim):
        raise ValueError(
            f"w_down {w_down.shape} must be (hidden_dim, embed_dim) = {(hidden_dim, embed_dim)}"
        )

    if tokens % block_t:
        raise ValueError(f"tokens {tokens} is not divisible by block_t {block_t}")
    if hidden_dim % block_h:
        raise ValueError(f"hidden_dim {hidden_dim} is not divisible by block_h {block_h}")

    # embed_dim is never blocked, so it has no divisor of its own -- but it is
    # the lane dimension of every block, so it still owes the (8, 128) rule.
    if embed_dim % _LANE:
        raise ValueError(f"embed_dim {embed_dim} is not a multiple of {_LANE}")
    if block_t % _SUBLANE:
        raise ValueError(f"block_t {block_t} is not a multiple of {_SUBLANE}")
    if block_h % _LANE:
        raise ValueError(f"block_h {block_h} is not a multiple of {_LANE}")

    return tokens, embed_dim, hidden_dim
