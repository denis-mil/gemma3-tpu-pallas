# Correctness is developed on CPU; the TPU is reserved for performance

Kernels are written and correctness-tested on a CPU-only Windows machine using
Pallas interpret mode. The Colab TPU is used only for performance measurement and a
final correctness confirmation.

This exists because of a constraint invisible in the code: the development machine
has no accelerator, and the only TPU available is a free Colab runtime that is
capped at 12 hours per session, disconnects after 90 minutes idle, and may refuse to
allocate a TPU at all during Pacific business hours. Treating that runtime as the
place where kernels are debugged would make the scarcest resource in the project the
one absorbing every typo and off-by-one.

Interpret mode removes that coupling almost entirely. `interpret=True` runs a
`pallas_call` as a jitted scan over the grid with the kernel lowered to ordinary JAX,
and `jax.experimental.pallas.tpu.InterpretParams` goes further, simulating HBM and
VMEM, local and remote DMAs, and semaphores. Both were verified to work on this
machine under JAX 0.11.0 on CPU. The local Python is pinned to 3.12 to match Colab's
3.12.13 runtime, so the two environments do not diverge.

## Consequences

Interpret mode is not evidence of hardware correctness — jax-ml/jax#36287 documents a
kernel that passes under interpretation and produces wrong results on TPU. Every
kernel therefore gets a correctness run on the TPU before any benchmark number
derived from it is published, and no result is reported from interpret mode alone.

Practical note: after any exception under TPU interpret mode,
`reset_tpu_interpret_mode_state()` must be called before using it again in the same
process.
