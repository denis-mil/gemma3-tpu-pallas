# Pallas TPU Kernels — Resources

Trusted sources for this workspace. Lessons cite from here rather than from
parametric recall.

## Knowledge

### Architecture

- [Gemma 3 Technical Report (arXiv:2503.19786)](https://arxiv.org/abs/2503.19786)
  ([DeepMind PDF mirror](https://storage.googleapis.com/deepmind-media/gemma/Gemma3Report.pdf))
  Primary source for the interleaved local/global design and the 5:1 ratio. **Caveat:**
  its headline "sliding window reduced from 4096 to 1024" describes the larger models —
  the 1B config ships `sliding_window: 512`. Always prefer the config over the report
  for per-model numbers.
- [`gemma-3-1b-it` config.json](https://huggingface.co/google/gemma-3-1b-it/blob/main/config.json)
  The authority for every shape in [`shapes.py`](src/gemma3_pallas/shapes.py). Gated on
  HF; the [unsloth mirror](https://huggingface.co/unsloth/gemma-3-1b-it/raw/main/config.json)
  is readable without auth and matches. Use for: settling any dimension dispute.

### Attention algorithms

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
  (arXiv:2205.14135)](https://arxiv.org/abs/2205.14135) — Dao, Fu, Ermon, Rudra, Ré.
  Use for: tiling, the online-softmax recurrence, and the IO-complexity argument that
  justifies the whole approach. Read §3.1 before Phase 3.
- [`splash_attention_kernel.py`](https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py)
  The ceiling. Use for: how block skipping is *actually* expressed on TPU — read its
  mask-processing code when Phase 5 starts, not before.
- [Ragged Paged Attention (arXiv:2604.15464)](https://arxiv.org/abs/2604.15464)
  Current state of the art for TPU serving. Use for: context on where decode kernels
  are heading. Not needed for prefill.

### Pallas

- [Pallas: a JAX kernel language](https://docs.jax.dev/en/latest/pallas/index.html)
  Entry point. Use for: `pallas_call`, grid semantics, `BlockSpec` basics.
- [Writing TPU kernels with Pallas](https://docs.jax.dev/en/latest/pallas/tpu/details.html)
  The important one. Use for: VMEM budgeting, the pipelining model, why `index_map`
  exists and what the hardware does with it.

## Wisdom (Communities)

- [jax-ml/jax GitHub Discussions](https://github.com/jax-ml/jax/discussions)
  Where Pallas questions actually get answered, sometimes by the JAX team. Prior art
  worth reading: [#31841 on TPU interpret mode](https://github.com/jax-ml/jax/discussions/31841),
  [#20509 on accumulation in Pallas](https://github.com/jax-ml/jax/discussions/20509).
  Use for: "is this supposed to be this slow", "does interpret mode model X".
- [jax-ml/jax Issues](https://github.com/jax-ml/jax/issues)
  Use for: confirming a suspected interpret-mode/hardware divergence before burning
  TPU quota on it — e.g. [#36287](https://github.com/jax-ml/jax/issues/36287),
  already cited in the README.

*No community preference recorded yet — ask before leaning on this section.*

## Gaps

- **No trusted source found yet** for TPU v5e microarchitecture beyond the public
  spec sheet — MXU issue latency, VMEM banking, DMA granularity. Needed for Phase 6's
  roofline to explain gaps rather than just measure them.
- **No worked Pallas tutorial** at the level of "here is why this `index_map` and not
  that one". The `splash_attention` source is the closest thing, and it is production
  code, not pedagogy.
