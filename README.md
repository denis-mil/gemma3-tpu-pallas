# gemma3-tpu-pallas

Custom TPU kernels written in [Pallas](https://docs.jax.dev/en/latest/pallas/), benchmarked at the exact
shapes [Gemma 3 1B](https://huggingface.co/google/gemma-3-1b-it) runs, on a single TPU v5e.

**Status: scaffolding and references complete. No kernel is written yet.** The roadmap below
is a plan, not a claim.

📖 **[Lessons and reference sheets →](https://denis-mil.github.io/gemma3-tpu-pallas/)** — six
lessons deriving the windowed-attention kernel from first principles, plus five reference sheets.
Interactive; read them in a browser rather than as source.

## The idea

Gemma 3 interleaves **local** and **global** attention layers. In the 1B model, 22 of 26 layers are
local with a **512-token window**; only layers 5, 11, 17 and 23 attend to everything.

A dense implementation computes the full score matrix and then masks it. At a 32K prompt, that means
a local layer computes ~64× more score pairs than it keeps — 1.6% of the work is load-bearing.
The kernel here skips those tiles instead of masking them, so it wins by **doing less work**
rather than by out-tuning anyone.

The same structure dominates memory:

| KV cache @ 32K context | batch 1 | batch 32 |
|---|---|---|
| Dense | 872 MB | **27.9 GB — exceeds 16 GB HBM** |
| Windowed | 146 MB | 4.66 GB — fits |

Windowing is not an optimisation at batch 32; it is the difference between running and not running.

## Architecture, verified

26 layers · embed 1152 · MLP hidden 6912 · **4 query heads, 1 KV head (MQA)** · head_dim 256 ·
window 512, pattern 6 · 32K context · RoPE 1e6 global / 1e4 local · GeGLU (`gelu_pytorch_tanh`).

Target hardware is one **TPU v5e** chip: 16 GB HBM, 197 bf16 TFLOP/s, 819 GB/s, 128×128 MXU.
Block sizes tuned here do not transfer to v6e, whose MXU is 256×256.

Everything above lives in [`shapes.py`](src/gemma3_pallas/shapes.py) and is asserted in
[`test_shapes.py`](tests/test_shapes.py) — no benchmark hardcodes a dimension.

## Roadmap

- [x] **Phase 0** — local CPU environment, Pallas interpret mode verified working
- [x] **Phase 1** — repo scaffolding, shape constants, fp32 references
- [ ] **Phase 2** — fused gated-MLP kernel *(warm-up; goal is understanding, parity with XLA is fine)*
- [ ] **Phase 3** — flash attention, stage 1: online softmax, no mask
- [ ] **Phase 4** — stage 2: causal masking
- [ ] **Phase 5** — stage 3: window block skipping via a shrunk grid and custom `index_map`,
      so out-of-window tiles are never DMA'd
- [ ] **Phase 6** — benchmarks, roofline, profiler traces
- [ ] **Phase 7** — writeup

## How correctness and speed are separated

The development machine has no accelerator. Kernels are written and tested locally on CPU under
Pallas interpret mode; the TPU is used only for performance and a final correctness confirmation.
See [ADR-0003](docs/adr/0003-cpu-interpret-mode-for-correctness.md).

Interpret mode is *not* proof of hardware correctness
([jax#36287](https://github.com/jax-ml/jax/issues/36287)), so no number is published from it alone.

**Numerics:** bf16 storage, fp32 accumulation for QK<sup>T</sup>, the running softmax max/sum, and PV —
matching what flash and splash attention do, and free on an MXU that accumulates in fp32 anyway.
Kernels are asserted against pure-fp32 references at `rtol=atol=2e-2`, justified by bf16's 8-bit mantissa.

**Baselines:** two of them. The *naive baseline* (dense scores, then mask) is what the kernel should beat.
`splash_attention` is the *ceiling* — it already does sliding-window block skipping, and the goal is to
size the remaining gap and explain it with a roofline, not to beat it.

## Setup

```bash
conda create -n tpu-pallas python=3.12 -y
conda activate tpu-pallas
pip install -r requirements.txt
pip install -e . --no-deps
pytest
```

Python 3.12 matches the Colab runtime (3.12.13). Verified on JAX 0.11.0, CPU backend.

> On a Colab TPU runtime, do **not** `pip install jax` — the runtime ships its own build and
> reinstalling it breaks `libtpu`.

## Layout

```
src/gemma3_pallas/
  shapes.py       Gemma 3 1B + TPU v5e constants, KV-cache and roofline arithmetic
  reference.py    pure-JAX fp32 ground truth (GeGLU MLP, MQA attention, masks)
tests/            reference correctness + Pallas interpret-mode smoke tests
docs/adr/         why the target, scope and workflow are what they are
CONTEXT.md        glossary — the project's canonical vocabulary
```

## Reading

- [Ragged Paged Attention](https://arxiv.org/abs/2604.15464) — the current state of the art for TPU serving
- [`splash_attention`](https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py) — the ceiling
- [Writing TPU kernels with Pallas](https://docs.jax.dev/en/latest/pallas/tpu/details.html)
