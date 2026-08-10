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
- [Roofline: An Insightful Visual Performance Model for Floating-Point Programs and
  Multicore Architectures](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html)
  — Williams, Waterman & Patterson, UC Berkeley EECS-2008-134, 2008. **The origin of the
  roofline**, and lesson 06's recommended reading. Use for: the model itself
  (`min(peak, I × β)`), the ridge point, and — the part most summaries drop — the
  *ceilings* below the roof, which are how the model explains a gap rather than merely
  bounding it. Fetch the `.html` landing page; the PDF comes back as unparseable binary.
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
  `jax/_src/pallas/mosaic/tpu_info.py`, so it is queryable offline — the only primary
  source found that states v5e VMEM. **The working call (session 8):**
  ```python
  from jax._src.pallas.mosaic import tpu_info as ti
  ti.get_tpu_info_for_chip(ti.ChipVersion.TPU_V5E, 1)
  ```
  The re-export `pltpu.get_tpu_info_for_chip` takes the same two arguments, and the first
  must be the `ti.ChipVersion` **enum member**, not the string `'v5e'` — a string fails
  late and unhelpfully on `.is_lite`. Members are `TPU_V2 … TPU_V6E`, plus `TPU_7`,
  `TPU_7X`, `TPU_8I`; the second argument is TensorCores per logical device, 1 for v5e.
  Fields: `bf16_ops_per_second`, `mem_bw_bytes_per_second`, `vmem_capacity_bytes`,
  `smem_capacity_bytes`, `hbm_capacity_bytes`, `num_mxus`, `mxu_column_size`,
  `num_lanes`, `num_sublanes`.
- [Scalar Prefetch and Block-Sparse Computation](https://docs.jax.dev/en/latest/pallas/tpu/sparse.html)
  `PrefetchScalarGridSpec`. Phase 5 — how a data-dependent `index_map` expresses block
  skipping. **Gate opened session 7**; it is lesson 05's recommended reading. Use for: the
  fixed argument order of `index_map` and `kernel` under scalar prefetch, and the
  block-sparse matmul worked example.
- [`splash_attention_mask_info.py`](https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask_info.py)
  Where the prefetch tables are actually built. Use for: `MaskInfo`'s docstring (the
  `block_mask` 0/1/2 encoding, which is lesson 01's dead/partial/full), and
  `_shrink_mask_info` (grid shrinking, and the fact that padding entries are index **0**
  rather than a repeat). Importable and runnable on CPU: `process_mask(mask, (bq, bkv),
  shrink_grid=...)` returns the tables directly, so it can be checked rather than read.
- [Cloud TPU v5e](https://docs.cloud.google.com/tpu/docs/v5e)
  The chip spec: 1 TensorCore, 16 GB HBM, 800 GiBps, 197 TFLOPs bf16. **Does not state
  VMEM** — do not go looking there for it. **Its bandwidth disagrees with JAX's** (session
  8): 800 GiBps is 858.99e9 B/s against `tpu_info`'s 820e9, and the same table writes
  "16 GB" for what `tpu_info` records as 17.2e9 bytes = 16 *GiB*, so its GB/GiB labelling
  is demonstrably loose. Prefer `tpu_info`; `shapes.py`'s `819e9` is a third value and
  agrees with neither. The resulting ridge point is 229–241 FLOPs/byte — carry the range,
  then check whether it changes the answer.

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

- **TPU v5e microarchitecture — closed as an errand, deferred until there is a measurement
  to explain.** `tpu_info.py` already supplies both roofs (197 TFLOP/s, 820e9 B/s), VMEM
  128 MiB, SMEM 1 MiB, HBM 16 GiB, and **4 MXUs of 128×128** — enough to place a kernel
  against the roofs and to state what the compute roof assumes. What was still being
  chased — **MXU issue latency, VMEM banking, DMA granularity** — is the material for a
  *ceiling below the roof*, and a ceiling exists to explain a **shortfall**. With no chip
  there is no shortfall, so a ceiling fitted now would be a mechanism fitted to zero data
  points. Two of the three are not published for v5e at all; the third, DMA granularity,
  was already searched in [research 0001](docs/research/0001-pallas-grid-blockspec-index-map.md)
  and yielded only a passing "4KBi" in the SMEM section. **Reopen when a Colab run
  produces a gap that the roofs alone do not account for**, and not before. The two items
  that do cash in on a first Colab session are the next two entries: the
  `--xla_tpu_scoped_vmem_limit_kib` default and the copy-elision xprof count.
  Do not try to back out a clock from `peak / (num_mxus · dim² · 2)`: it gives a clean
  1.5 GHz on v5e and an impossible 3.5 GHz on v6e, so it is a mnemonic, not a derivation.
- ~~**No worked Pallas tutorial** at the level of "here is why this `index_map` and not
  that one"~~ — closed by
  [research 0001](docs/research/0001-pallas-grid-blockspec-index-map.md), which pins the
  copy-elision rule to `tpu/details.html` and demonstrates it by execution.
- **The default value of `--xla_tpu_scoped_vmem_limit_kib`** — not the physical 128 MiB,
  is what a kernel actually gets. `pltpu.CompilerParams` proves the flag governs the
  budget but never states its default. Not in the JAX Python tree. **Session 10 made this
  strictly a hardware question:** `pltpu.InterpretParams` models **no VMEM capacity at
  all**. Its twelve fields are `detect_races`, `out_of_bounds_reads`,
  `skip_floating_point_ops`, `uninitialized_memory`, `num_cores_or_threads`,
  `vector_clock_size`, `logging_mode`, `dma_execution_mode`, `random_seed`,
  `grid_point_recorder`, `allow_hbm_allocation_in_run_scoped`, `buffer_bounds` — no
  capacity among them — and `jax._src.pallas.mosaic.interpret` contains zero occurrences
  of "vmem", "capacity" or "OOM". So the higher-fidelity interpreter simulates HBM/VMEM
  *movement* (DMAs, semaphores) but not VMEM *size*: **no local test can ever falsify a
  block-size choice.** `tests/test_mlp_kernel.py::test_gemma_dims_under_simulated_memory`
  passes on a 22.5 MiB predicted working set and that fact carries no information about
  the budget. Useful side effect: `uninitialized_memory` defaults to `nan`, which is why
  a missing accumulator init fails loudly rather than plausibly.
  **Session 11 built the errand rather than answering it:** `bench.bench_kernel` records a
  Mosaic VMEM failure as `status="failed"` with the compiler's message instead of raising,
  so a `block_h` sweep at the default limit brackets the budget between the largest working
  set that compiles and the smallest that does not — cell 4 of
  [`notebooks/phase2_v5e.ipynb`](notebooks/phase2_v5e.ipynb). Two things that sharpen the
  question: the headline working set is **23.25 MiB, not 22.5** (the 22.5 figure omits the
  two `[block_t, block_h]` fp32 intermediates, 0.375 MiB each), and `pltpu.CompilerParams`
  **does** carry `vmem_limit_bytes` on JAX 0.11.0, verified locally — so the limit is
  sweepable per `pallas_call`, with no `LIBTPU_INIT_ARGS` and no runtime restart.
- **Whether copy elision is contractual on the hardware path.** The docs state it as a
  property and two JAX-authored pipeline implementations honour it, but the `pallas_call`
  pipeline on TPU is emitted by the Mosaic compiler, out of reach from Python. An xprof
  DMA count on Colab would settle it — the highest-value single use of TPU time on this
  topic. **First candidate for a jax-ml/jax Discussions post.**
  **Session 11 made the question quantitative.** For `fused_gated_mlp` the two answers are
  not "elision or not" in the abstract: the `w_*` `index_map`s vary in the *innermost* grid
  index, so elision cannot bridge the reset of that index between `t` steps, and the weight
  traffic is `tokens // block_t` reads rather than one. At the headline geometry that is
  2 versus 1 transfer per weight per call, 193.5 MB versus 97.9 MB, intensity 63 versus
  125 — a difference large enough to move the kernel across `DEFAULT`'s ridge. Both counts
  are computable (`shapes.mlp_bytes(..., elide_weights=)`), the prediction is registered in
  [measurement 0001](docs/measurements/0001-fused-gated-mlp-on-v5e.md), and cell 6 of the
  notebook is the DMA count that decides it.
- **`RevisitMode.ANY`** — documented only in `jax/_src/pallas/core.py`, rejected by
  lowering when `buffer_count > 1`, zero doc-page coverage. **Session 7 removed its most
  likely motivation without finding a replacement:** a shrunk grid does *not* need it,
  because shrinking removes steps from inside a row and never permutes the outer index, so
  output revisits stay consecutive — confirmed by an accumulating shrunk-grid kernel
  matching a dense reference to `8.7e-7` under both interpreters. So the question is now
  sharper rather than closed: *what grid shape is it for?* That well-posedness makes it the
  best jax-ml/jax Discussions candidate after the block-shape ambiguity.
- **Which reading of the `(8, 128)` block-shape rule Mosaic implements.** "The last two
  dimensions … must be equal to the respective dimension of the overall array, or be
  divisible by 8 and 128 respectively" is ambiguous between a per-axis and a paired
  disjunction; they disagree on an `(8, 7)` block over a `(16, 7)` array. Interpret mode
  cannot adjudicate — it accepts illegal shapes in both modes. See
  [research 0001](docs/research/0001-pallas-grid-blockspec-index-map.md) §5.1.
  **The best first candidate for a jax-ml/jax Discussions post**: one sentence, one
  counter-example, no experiment to defend.
