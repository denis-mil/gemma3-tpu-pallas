# Mission: Writing TPU kernels in Pallas

## Why

Denis wants to become someone who can author accelerator kernels — able to look at a
model's arithmetic, find the structural win, and write the Pallas kernel that takes it.
The Gemma 3 1B windowed-prefill kernel in this repo is the vehicle, not the destination:
the goal is that the *next* kernel, on a different model and a different shape, can be
written without a tutorial.

## Success looks like

- Deriving, unaided, which tiles of a masked attention computation are load-bearing —
  and therefore which the kernel may refuse to load.
- Writing a Pallas TPU kernel from a blank file: grid, `BlockSpec`s, `index_map`,
  VMEM budget, without copying a template.
- Deriving online softmax from scratch, and explaining why the running max is required
  rather than merely convenient.
- Reading a roofline and stating *which* roof binds a given kernel, with the arithmetic
  to back it.
- Explaining a kernel's remaining gap to `splash_attention` in mechanistic terms.

## Constraints

- No local accelerator. Correctness is developed on CPU under Pallas interpret mode;
  TPU time is a scarce free-Colab resource (12h cap, 90min idle timeout, no guarantee
  of getting a chip). Lessons must not assume hardware is available.
- Target is one TPU v5e (128×128 MXU). Tuned block sizes do not transfer to v6e.
- Attention internals are known; flash-style tiling and online softmax are not yet.
- Pallas experience is limited to running interpret mode.

## Out of scope

- Serving, checkpoints, tokenizers, samplers — see [ADR-0002](docs/adr/0002-kernel-only-scope.md).
- GPU kernels (Triton, CUDA). The mission is TPU.
- Training-time concerns: backward passes, gradient checkpointing.
- Model quality effects of windowing. This project optimises arithmetic, not accuracy.
