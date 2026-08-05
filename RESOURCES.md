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

- [Online normalizer calculation for softmax (arXiv:1805.02867)](https://arxiv.org/abs/1805.02867)
  — Milakov & Gimelshein, NVIDIA, 2018. **The origin of the online-softmax recurrence**,
  four pages, with an induction proof. Algorithm 2 is safe softmax (three passes);
  Algorithm 3 is the single-pass version,
  `d_j ← d_{j−1}·e^{m_{j−1}−m_j} + e^{x_j−m_j}`. Use for: the recurrence itself, and the
  invariant that makes it testable. Prefer it over FlashAttention for *why the recurrence
  is correct*; FlashAttention inherits it without re-deriving it.
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
  (arXiv:2205.14135)](https://arxiv.org/abs/2205.14135) — Dao, Fu, Ermon, Rudra, Ré.
  **Section map, verified against the ar5iv HTML** — the numbering is easy to get
  backwards:
  - §3.1 "An Efficient Attention Algorithm With Tiling and Recomputation" — tiling *and*
    the online-softmax recurrence (Algorithm 1, with `m` and `ℓ`). Lesson 02's reading.
  - §3.2 "Analysis: IO Complexity of FlashAttention" — the memory-traffic argument,
    Theorem 2. Lesson 01's reading.
  - §3.3 "Extension: Block-Sparse FlashAttention" — the block-skipping payoff. Phase 5.
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
  The important one. Use for: lexicographic grid order, **the copy-skip statement**
  ("when two consecutive grid indices use the same slice of an input, the HBM transfer
  for the second iteration is skipped"), the reduction-on-last-axis rule, the `(8, 128)`
  block rule, and the VMEM OOM failure mode. **Caveat:** carries no reviewed-date and
  its "VMEM is 16MB+" is the v4 number — see the disagreement in
  [research 0001](docs/research/0001-pallas-grid-blockspec-index-map.md).
- [Software Pipelining](https://docs.jax.dev/en/latest/pallas/pipelining.html)
  Platform-neutral. Use for: the derivation of the double-buffered loop, buffer-revisiting
  hazards, and the accumulator pattern with a worked correct/incorrect pair.
- [TPU Pipelining](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html)
  Use for: the memory-space table, "pipelining is not allowed unless the `memory_space`
  is marked as `VMEM`", default buffer count 2, and Megacore availability (v4/v5p only —
  a no-op on our v5e). Reviewed 2024-04-08.
- [Grids and BlockSpecs](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html)
  Use for: `index_map` returning *block* indices, the executable `slices_for_invocation`
  definition, padding of non-dividing blocks, `None`/`Squeezed`. Reviewed 2024-06-01.
- [TPU Hardware Reference](https://docs.jax.dev/en/latest/pallas/tpu/hardware.html)
  Per-TensorCore VMEM/SMEM/HBM/peak-FLOPs by generation. Generated from
  `jax/_src/pallas/mosaic/tpu_info.py`, so it is queryable offline via
  `pltpu.get_tpu_info_for_chip` — the only primary source found that states v5e VMEM.
- [Scalar Prefetch and Block-Sparse Computation](https://docs.jax.dev/en/latest/pallas/tpu/sparse.html)
  `PrefetchScalarGridSpec`. Phase 5 — how a data-dependent `index_map` expresses block
  skipping. Do not read before Phase 5.
- [Cloud TPU v5e](https://docs.cloud.google.com/tpu/docs/v5e)
  The chip spec: 1 TensorCore, 16 GB HBM, 800 GiBps, 197 TFLOPs bf16. **Does not state
  VMEM** — do not go looking there for it.

### Workspace research

- [research 0001 · Pallas grid, `BlockSpec`, `index_map`](docs/research/0001-pallas-grid-blockspec-index-map.md)
  Primary-source findings behind lessons 03 and 04, with a verified/inferred split and five
  documented disagreements inside the JAX tree. Use for: the copy-elision rule (§3), the
  VMEM budget (§4), the block-shape rule (§5), and the interpret-mode experiments that
  demonstrate them. **§4.3 was re-derived and corrected in session 6** — the parts still
  marked *inferred* are inferred.

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

**Chosen community: [jax-ml/jax Discussions](https://github.com/jax-ml/jax/discussions)** — confirmed
by Denis, session 3. Use it as the fallback authority when a question has no written
answer in the docs or the source, and as the place to test a derivation against
practitioners. Nothing has been posted yet.

## Gaps

- **No trusted source found yet** for TPU v5e microarchitecture beyond the public
  spec sheet — MXU issue latency, VMEM banking, DMA granularity. Needed for Phase 6's
  roofline to explain gaps rather than just measure them.
- ~~**No worked Pallas tutorial** at the level of "here is why this `index_map` and not
  that one"~~ — closed by
  [research 0001](docs/research/0001-pallas-grid-blockspec-index-map.md), which pins the
  copy-elision rule to `tpu/details.html` and demonstrates it by execution.
- **The default value of `--xla_tpu_scoped_vmem_limit_kib`** — not the physical 128 MiB,
  is what a kernel actually gets. `pltpu.CompilerParams` proves the flag governs the
  budget but never states its default. Not in the JAX Python tree.
- **Whether copy elision is contractual on the hardware path.** The docs state it as a
  property and two JAX-authored pipeline implementations honour it, but the `pallas_call`
  pipeline on TPU is emitted by the Mosaic compiler, out of reach from Python. An xprof
  DMA count on Colab would settle it — the highest-value single use of TPU time on this
  topic. **First candidate for a jax-ml/jax Discussions post.**
- **`RevisitMode.ANY`** — documented only in `jax/_src/pallas/core.py`, rejected by
  lowering when `buffer_count > 1`, zero doc-page coverage. Possibly the escape hatch for
  a shrunk grid whose output blocks are not visited consecutively; possibly a dead end.
- **Which reading of the `(8, 128)` block-shape rule Mosaic implements.** "The last two
  dimensions … must be equal to the respective dimension of the overall array, or be
  divisible by 8 and 128 respectively" is ambiguous between a per-axis and a paired
  disjunction; they disagree on an `(8, 7)` block over a `(16, 7)` array. Interpret mode
  cannot adjudicate — it accepts illegal shapes in both modes. See
  [research 0001](docs/research/0001-pallas-grid-blockspec-index-map.md) §5.1.
  **The best first candidate for a jax-ml/jax Discussions post**: one sentence, one
  counter-example, no experiment to defend.
