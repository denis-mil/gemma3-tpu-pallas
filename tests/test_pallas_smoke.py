"""Proves the local development loop works: Pallas kernels on a CPU-only box.

There is no TPU on the development machine. Pallas kernels are exercised via
interpret mode, which runs pallas_call as a jitted scan over the grid with the
kernel body lowered to ordinary JAX.

Two fidelity levels are covered:

  interpret=True          plain functional emulation
  interpret=InterpretParams()   simulates HBM/VMEM, DMAs and semaphores

Neither is a substitute for a real TPU run -- see jax-ml/jax#36287 for a kernel
that is correct under interpret mode and wrong on hardware. Every kernel still
gets a correctness run on Colab before any benchmark number is published.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl


def _add_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]


def _add(x, y, *, interpret):
    return pl.pallas_call(
        _add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=interpret,
    )(x, y)


def _blocked_double_kernel(x_ref, o_ref):
    o_ref[...] = x_ref[...] * 2.0


def _blocked_double(x, *, block: int, interpret):
    """Exercises a grid and BlockSpecs -- the machinery every real kernel uses."""
    rows = x.shape[0]
    return pl.pallas_call(
        _blocked_double_kernel,
        grid=(rows // block,),
        in_specs=[pl.BlockSpec((block, x.shape[1]), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((block, x.shape[1]), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=interpret,
    )(x)


def test_cpu_is_the_only_backend_here():
    """Documents the environment: if this ever fails, we grew a TPU."""
    assert jax.devices()[0].platform == "cpu"


def test_elementwise_kernel_under_interpret_mode():
    x = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)
    y = jnp.ones_like(x)
    got = _add(x, y, interpret=True)
    assert np.allclose(np.asarray(got), np.asarray(x + y))


def test_grid_and_blockspec_under_interpret_mode():
    x = jnp.arange(16 * 128, dtype=jnp.float32).reshape(16, 128)
    got = _blocked_double(x, block=4, interpret=True)
    assert np.allclose(np.asarray(got), np.asarray(x * 2.0))


def test_tpu_interpret_params_simulates_the_memory_hierarchy():
    """The higher-fidelity mode: simulated HBM/VMEM, DMAs and semaphores."""
    pltpu = pytest.importorskip(
        "jax.experimental.pallas.tpu",
        reason="Pallas TPU interpret mode unavailable in this JAX build",
    )
    if not hasattr(pltpu, "InterpretParams"):
        pytest.skip("this JAX build predates pltpu.InterpretParams")

    x = jnp.arange(16 * 128, dtype=jnp.float32).reshape(16, 128)
    got = _blocked_double(x, block=4, interpret=pltpu.InterpretParams())
    assert np.allclose(np.asarray(got), np.asarray(x * 2.0))
