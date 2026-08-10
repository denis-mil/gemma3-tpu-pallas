# Measurement 0001 — `fused_gated_mlp` on a TPU v5e

**Status: re-run on 2026-08-10 00:31 UTC on a fresh v5e session, verdicts filled from that
run.** The predictions scored below were registered before any hardware time and are quoted
unaltered from `git show b40bee8^:docs/measurements/0001-fused-gated-mlp-on-v5e.md`; only
the verdict cells and the measured numbers come from this session. A prediction adjusted
after seeing the number is not a prediction.

Two things about this document that the earlier version of it did not have to say:

- **It scores a re-run, not the original run.** The first v5e session was 2026-08-07; this
  one repeats the same notebook end to end on a fresh runtime. Every verdict came out the
  same way. The timings moved by a few percent in both directions (`block_t=512` at
  `T=2048`: 0.766 ms then, 0.802 ms now; XLA at the same point 0.817 then, 0.795 now), and
  the device time per call at the headline geometry is identical to within 0.1 us.
- **FLOPs are counted the way ADR-0004 counts them.** The earlier version carried a
  "three roofs" table with effective peaks of 197 / 65.7 / 32.8 TFLOP/s. There is one roof;
  the bf16 pass count multiplies the FLOPs instead of dividing it
  ([ADR-0004](../adr/0004-flop-counts-are-hardware-work.md)). No bound verdict moves —
  `passes · I` against `P` is the same inequality as `I` against `P / passes` — so the
  registered predictions are scored unchanged, in the new vocabulary.

- Kernel: [`src/gemma3_pallas/mlp.py`](../../src/gemma3_pallas/mlp.py), `fused_gated_mlp`
- Harness: [`src/gemma3_pallas/bench.py`](../../src/gemma3_pallas/bench.py)
- Driver: [`notebooks/phase2_v5e.ipynb`](../../notebooks/phase2_v5e.ipynb)
- Hardware: one TPU v5e chip, free Colab runtime, jax 0.11.0
- Constants: 197e12 bf16 FLOP/s, 819e9 B/s, 128 MiB VMEM — see the note on bandwidth below

Three conditions of the run that no verdict cell can carry:

- **The headline `block_h` was 384, not the 768 the original prediction assumed.** 768 asks
  22.5 MiB of scoped VMEM and does not compile at the default limit, which is prediction 3's
  own finding. The grid was therefore `(2, 18)`, not `(2, 9)`. Nothing else moves with it:
  `mlp_bytes` does not depend on `block_h`, so the byte model and its intensity of 63.2 are
  the same at either block.
- **`HIGH` never ran.** `jax.lax.Precision.HIGH` does not lower on jax 0.11.0 — Mosaic's
  `_dot_general_lowering_rule` raises `NotImplementedError: Unsupported dot precision: HIGH`
  — so the precision probe was cut to `DEFAULT` and `HIGHEST`. Every registered `HIGH` claim
  is unscoreable, including the 21%-margin crossover that prediction 1 named its better test.
- **The intensity ladder was traversed by `block_t`, not by `T`.** With `block_t` pinned,
  `I = 6EH·T / (T/block_t · 3EH·4 + 8TE)` and `T` cancels exactly, so the token sweep sits at
  I = 63.2 at every size. Sweeping `block_t` at a fixed `T = 2048` walks 63.2, 124.9, 244.0,
  466.0, 855.1 — prediction 1's ladder exactly — and that is where its claims are scored.

## Geometry

Gemma 3 1B's MLP: `embed_dim = 1152`, `hidden_dim = 6912`, fp32 operands. The grid is
`(tokens // block_t, hidden_dim // block_h)` with the hidden axis **innermost**. The
headline configuration as run is `T = 256, block_t = 128, block_h = 384`, i.e. grid
`(2, 18)`.

Working set per grid step, at the headline configuration:

| | bytes |
|---|---|
| five blocked operands (`x`, `w_gate`, `w_up`, `w_down`, output), double-buffered | 12.375 MiB |
| two `[block_t, block_h]` fp32 intermediates (`gate`, `up`), not buffered | 0.375 MiB |
| **total** | **12.75 MiB** |

At `block_h = 768` the same table reads 22.5 MiB + 0.75 MiB = 23.25 MiB, and the compiler
counts only the first of those two rows — it reported "size 22.50M" against a 23.25 MiB
working set. The intermediates are produced and consumed inside the kernel body, which is
what makes the fusion worth doing, but they are still resident; they are simply not part of
the allocation the scoped-VMEM limit governs.

---

## The byte model, corrected

The `w_*` `index_map`s vary in the innermost grid index. So one `t` step walks every hidden
block; when that index resets for the next `t` step, block 0 is no longer the previously
fetched slice, and copy elision — which skips a transfer only between two *consecutive*
identical slices — cannot apply. The weights are therefore re-read **once per `t` step**:

```
bytes = (tokens // block_t) · 3·E·H·dtype  +  2·T·E·dtype
```

At the headline geometry that is **193.5 MB and intensity 63.2**, not 97.9 MB and 124.9.
Both counts are available: `shapes.mlp_bytes(..., elide_weights=True)` gives the one-pass
counterfactual, so the two models are printed side by side and the trace in section 6
adjudicates between them.

This is an extension of a rule the workspace already wrote down — *an intensity argument is
only valid if every operand is in the byte count* — to **every visit**, not merely every
operand.

## The one roof, and where each precision sits under it

`tpu_info` publishes no fp32 peak for v5e, and there is none to publish: TPU emulates an
fp32 matmul with several bf16 passes, which `jax.lax.DotAlgorithmPreset` names
`BF16_BF16_F32_X3` and `_X6`. Those passes are work the chip does, so they are counted in
the FLOPs and the roof stays at one number (ADR-0004). A precision does not get its own
ceiling; it slides right along the same one.

| `precision` | bf16 passes | roof | I at `block_t = tokens = 128` | against the ridge, 240.5 |
|---|---|---|---|---|
| `DEFAULT` | 1 | 197 TFLOP/s | 63.2 | 74% below — memory-bound |
| `HIGH` | 3 | 197 TFLOP/s | 189.7 | 21% below — memory-bound |
| `HIGHEST` | 6 | 197 TFLOP/s | 379.3 | 58% above — compute-bound |

The margins are the same ones the old per-precision-ridge table quoted, because the two
formulations are the same inequality.

## Registered prediction 1 — where the crossovers are

At `block_t = tokens` (one `t` step, weights read once) the `DEFAULT` intensities are:

| T | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|
| I (FLOP/byte) | 63.2 | 124.9 | 244.0 | 466.0 | 855.1 |

Multiply by the pass count for the other precisions. Against the one ridge at 240.5:

| `precision` | HBM-bound at | at the ridge (unresolvable) | compute-bound at |
|---|---|---|---|
| `DEFAULT` | 128, 256 | **512** (244.0 — 1.4% above) | 1024, 2048 |
| `HIGH` | **128** (189.7 — 21% below) | — | 256, 512, 1024, 2048 |
| `HIGHEST` | — | — | every T (128 is 58% above) |

Two corrections to the obvious reading, both registered in advance:

1. **`HIGH` is not compute-bound at every T.** At T=128 the kernel sits 21% *below* the
   ridge, so `HIGH` has a crossover of its own, between T=128 and T=256 — and its 21%
   margin makes it a **better** test of the byte model than `DEFAULT`'s.
2. **`DEFAULT`'s crossover cannot be located by measurement.** T=512 lands 1.4% above the
   ridge, inside the noise of any timing this notebook collects. So it is registered *now*
   as unresolvable rather than scored later, and `bench.summarise` prints any point within
   2% of its ridge as **at ridge** rather than as a verdict.

The two resolvable claims are therefore: `HIGH` bends between T=128 and T=256, and
`DEFAULT` bends somewhere in 256…1024. **If the timings do not bend there, the byte model
is wrong — which is the interesting result, not a failure.**

| Claim | Verdict |
|---|---|
| `HIGH` crosses between T=128 and T=256 | **unscoreable** — `HIGH` does not lower on jax 0.11.0 |
| `DEFAULT` is HBM-bound at I=124.9 | **held** — 1.256 ms against 2.320 ms at I=63.2, a 1.85x gain for a 1.98x cut in bytes |
| `DEFAULT` is compute-bound at I=466.0 | **held** — 0.764 ms against 0.802 ms at I=244.0, and halving the bytes again to I=855.1 makes it *slower* at 0.776 ms |
| `DEFAULT` at I=244.0 is unresolvable | *registered as unscoreable* — measured 0.802 ms, 62.0% of a 497 us floor, which fits either roof |
| `HIGHEST` is compute-bound at every T | **held** — kernel and XLA land within 6% of each other at every T (1.04–1.06x from T=256 up) even though XLA reads the weights once and the kernel re-reads them `T/block_t` times; under a memory roof the lighter traffic would win |

The token sweep itself, `block_t` pinned at 128, `block_h = 384`, wall-clock medians of 20
repeats:

| precision | T | kernel | XLA | kernel/XLA | I | hardware TFLOP/s | % of roof | binds |
|---|---|---|---|---|---|---|---|---|
| `DEFAULT` | 128 | 0.342 ms | 0.305 ms | 0.89x | 63.2 | 17.9 | 34.6 | memory |
| `DEFAULT` | 256 | 0.492 ms | 0.312 ms | 0.64x | 63.2 | 24.9 | 48.0 | memory |
| `DEFAULT` | 512 | 0.735 ms | 0.364 ms | 0.50x | 63.2 | 33.3 | 64.3 | memory |
| `DEFAULT` | 1024 | 1.286 ms | 0.502 ms | 0.39x | 63.2 | 38.1 | 73.5 | memory |
| `DEFAULT` | 2048 | 2.320 ms | 0.783 ms | 0.34x | 63.2 | 42.2 | 81.4 | memory |
| `HIGHEST` | 128 | 0.423 ms | 0.421 ms | 1.00x | 379.3 | 86.8 | 44.0 | compute |
| `HIGHEST` | 256 | 0.610 ms | 0.639 ms | 1.05x | 379.3 | 120.3 | 61.1 | compute |
| `HIGHEST` | 512 | 1.043 ms | 1.097 ms | 1.05x | 379.3 | 140.7 | 71.4 | compute |
| `HIGHEST` | 1024 | 1.779 ms | 1.892 ms | 1.06x | 379.3 | 165.0 | 83.7 | compute |
| `HIGHEST` | 2048 | 3.349 ms | 3.495 ms | 1.04x | 379.3 | 175.3 | 89.0 | compute |

The TFLOP/s column is **hardware** work, comparable to 197 and to nothing else. The
`HIGHEST` row at T=2048 is 175.3 TFLOP/s of bf16 passes and 29.2 TFLOP/s of the matmul
Gemma asked for; divide by the pass count to move between them.

The `DEFAULT` column is the honest bad news of the pinned geometry, and the next section is
where it stops being the kernel's number.

## The pinned block is not the kernel's best geometry

The token sweep above holds `block_t` at 128 while `T` grows, which is what isolates the cost
of re-streaming the weights — and also why it cannot see a crossover: with `block_t` pinned,
`T` cancels out of the intensity exactly and every row sits at I = 63.2. The byte model has
one term under our control and only one: `mlp_bytes` does not depend on `block_h` at all, as
the Geometry section above records. So hold `T = 2048` and `DEFAULT` and sweep `block_t`
instead. The registration for that sweep is the table committed under `### 5b` in `d177fe5`,
2026-08-07 12:25 PDT, about ninety minutes before the first run that scored it. The notebook's
5b prose was rewritten afterwards and no longer carries that table, so the predicted columns
below are quoted from `git show d177fe5:notebooks/phase2_v5e.ipynb` rather than from the
notebook's current text — the same rule this document applies to itself with `b40bee8^`.

Fixed for every row: `T = 2048`, `block_h = 384`, `DEFAULT`, `vmem_limit_bytes = 64 MiB`.

| `block_t` | bytes | I | roof | binds | measured | % of roof | gain over previous |
|---|---|---|---|---|---|---|---|
| 128 | 1547.7 MB | 63.2 | 1890 us | memory | 2.320 ms | 81.5 | — |
| 256 | 783.3 MB | 124.9 | 956 us | memory | 1.256 ms | 76.2 | 1.85x |
| 512 | 401.1 MB | 244.0 | 497 us | compute | 0.802 ms | 62.0 | 1.57x |
| 1024 | 210.0 MB | 466.0 | 497 us | compute | 0.764 ms | 65.0 | 1.05x |
| 2048 | 114.4 MB | 855.1 | 497 us | compute | 0.776 ms | 64.0 | 0.98x |
| XLA | — | — | — | — | 0.795 ms | — | — |

The registration named both ways it could fail, and they point in opposite directions: a
kernel that keeps improving past `block_t=512` falsifies the byte model, and one that never
improves falsifies the diagnosis. **Neither fired.** The kernel gains 1.85x and 1.57x across
the two memory-bound halvings and then stops — 1.05x into 1024 and a 1.6% regression at 2048
— against byte counts that keep halving throughout. The curve flattens where the roofs cross
and not before, and the minimum is interior: 1024, not the largest block available. The last
row has an explanation the sweep does not isolate, so it is offered rather than measured: at
`block_t = 2048` the `t` axis is one step long, so by the single-buffering rule prediction 3
confirms, `x` and the output get one buffer each and there is nothing left to overlap the
first fetch against.

**The sign of the comparison against XLA depends on the block, not on the kernel.** Same `T`,
same precision, same `block_h`:

| | ms | vs XLA |
|---|---|---|
| kernel, `block_t=128` (the token sweep's pinned geometry) | 2.320 | 0.34x |
| kernel, `block_t=1024` | 0.764 | **1.04x** |
| XLA | 0.795 | — |

A factor of 3.0 between two rows of the same kernel. The `DEFAULT` rows above are therefore
one geometry measured at five sizes, not the kernel's result, and section 7 of the notebook
prints the best `block_t` row beside them for exactly that reason. The comparison is not
quite like for like either, and the difference favours XLA: XLA materialises both
`[T, hidden_dim]` intermediates in HBM, so it moves bytes this model does not describe, and
it reads the weights once regardless — the kernel at `block_t=1024` still reads them twice.

Two caveats on the table:

- **Every time in it is wall clock**, from `bench.time_call`, and prediction 4 below finds
  roughly 233 us per call of host dispatch inside it at T=256. If that overhead is constant
  in the geometry — which was not measured — the flat region sits nearer 90% of its roof than
  the 65% printed. The verdict does not turn on it: this is a claim about where the curve
  bends, and a constant offset does not move a bend.
- **`vmem_limit_bytes = 64 MiB` on every row, including the ones that fit in the default 16.**
  The largest geometry asks 28.12 MiB of scoped VMEM and would not otherwise compile. Raising
  the limit only where it is needed would put a second variable in the sweep.

## Registered prediction 2 — the on-device error

The CPU interpret run's largest absolute error at Gemma dims was 3.28e-6, against an output
scale of about 2.15. On hardware at `DEFAULT` precision the multiplies are bf16, so the
error is expected to be **far larger** and `GEMMA_RTOL = 1e-4` is expected to fail. That
tolerance is a CPU measurement and does not transfer, so the notebook **reports** the error
rather than asserting an inherited number.

The precision whose error approaches the CPU figure is the one doing full-mantissa work, and
its pass count is what multiplies its FLOPs against the 197 TFLOP/s roof. That is how the
pass count gets chosen by measurement rather than by assumption.

| Claim | Verdict |
|---|---|
| `DEFAULT` error ≫ 3.28e-6 | **held** — 9.636e-3 absolute against an output scale of 2.152, about 2900x the CPU figure, and a max *relative* error of 3.03 |
| error falls monotonically DEFAULT → HIGH → HIGHEST | **partly scoreable** — the one measurable leg falls, 9.636e-3 to 2.050e-5; the `HIGH` point does not exist |
| some precision reproduces the CPU error | **not held** — `HIGHEST` reaches 2.050e-5, still 6.3x the CPU's 3.28e-6, and it is the closest the toolchain offers. Its 6 passes were used as the FLOP multiplier anyway, which is what the prediction wanted the error for |

## Registered prediction 3 — the VMEM budget

`pltpu.InterpretParams` models no VMEM capacity at all, so this cannot be asked anywhere but
on hardware. The physical capacity is 128 MiB; what a `pallas_call` gets is the
`--xla_tpu_scoped_vmem_limit_kib` default, which is not published.

No prediction is registered for its value — there is no basis for one. What is registered is
the **method**: sweep `block_h` at the default limit until compilation fails, and the largest
working set that compiles and the smallest that does not bracket the budget. Then re-run the
first failing config with explicit `vmem_limit_bytes` values to confirm the flag moves that
boundary.

| Claim | Verdict |
|---|---|
| the default limit lies strictly between two swept working sets | **held** — 12.375 MiB compiles, 22.50 MiB does not, and no admissible `block_h` lies between them (`block_h` must divide 6912 and be a multiple of 128, which excludes 512 and 1728). The bracket did not need to be narrowed: the compiler names its own limit, "Scoped allocation with size 22.50M and limit 16.00M exceeded scoped vmem limit by 6.50M" |
| `vmem_limit_bytes` moves the boundary, without a runtime restart | **held** — `block_h=768` fails at 16 MiB and runs at 32 (0.532 ms), 64 (0.480 ms), 100 (0.483 ms) and 128 (0.482 ms); `block_h=6912` fails through 64 MiB and runs at 100 (0.454 ms) and 128 (0.428 ms). The second row also confirms the single-buffering rule: its 91.12 MiB of weights fit under 100 MiB only because the `h` axis is one step long and gets one buffer, where two would ask 184.50 MiB and fit nowhere on the chip |

## Registered prediction 4 — copy elision

The docs state copy elision as a property of the TPU pipeline, but that pipeline is emitted
by Mosaic and is out of reach from Python. The two byte models above differ by exactly a
factor of `tokens // block_t` in the weight traffic, so a trace settles it. At the headline
geometry: **2 transfers per weight per call** under the corrected model, 1 under elision.

| Claim | Verdict |
|---|---|
| `w_*` DMAs per call = `tokens // block_t` × 3, not 3 | **held** — see below |

The count did not come from where the prediction expected it. `Pallas Primitives` is empty
in the capture even under `tpu_trace_mode="TRACE_COMPUTE_AND_SYNC"`, so there is no row of
named DMA events to tally. What that mode does add is the `Tensor Core Sync Flag` line, and
the DMAs are visible there as the waits that retire them: four flags — `SyncWait:55, 56, 57,
58` — 18 waits each, **72 per call**, identically in all five traced calls, in a rigid
13.98 us cycle. The grid is 36 steps and every one of them must execute, so 18 cycles cover
36 steps at two waits per step: two streams, double-buffered.

The decisive evidence is positional rather than arithmetic. Under elision the second `t`
pass — steps 18 through 35 — fetches no weights at all, so the waits would stop at the
halfway mark. They do not: exactly 36 of the 72 waits fall in each half of the call, the
first at 3.6 us and the last at 253.2 us of a 258.5 us call, on the same cycle end to end.
Both `t` passes fetch, which is the corrected model.

Three supporting numbers, all from the same trace:

- 180.6 us of the 258.5 us call — **69.9%** — is spent blocked on those flags. A kernel
  moving only the elided model's 97.9 MB needs 119.5 us of transfer in total and cannot
  spend 181 us waiting for it.
- Per grid step the kernel moves 5.38 MB in 7.18 us, **748 GB/s**, which is 91% of the
  819 GB/s roof — the corrected model's traffic running at very nearly the speed the
  hardware can supply it. Scored against the elided model the same call reads 379 GB/s,
  46% of the roof, which is not a number a kernel blocked 70% of the time produces.
- Device time is 258.5 us against a memory roof of 236.2 us for the corrected model
  (1.09x) and 119.5 us for the elided one (2.16x). The compute floor at `DEFAULT` is
  62.1 us, so compute is not what this call is doing.

One instrumentation finding falls out of the same trace, and it applies to every wall-clock
number this notebook reports. Device time at the headline geometry is 258.5 us per call
(258.42, 258.56, 258.51, 258.43, 258.55 across five calls) against a wall-clock median of
492 us, so roughly **233 us per call is host dispatch** that no roof accounts for.
`bench.time_call` measures wall clock, so the `%roof` column understates on-chip efficiency
badly at small `T` — the kernel reaches 91% of its memory roof at T=256, not the 48.0%
printed. It does not rescue the kernel against XLA: subtracting a constant from both makes
the ratio worse, not better.

## Notes on the constants

`shapes.py` carries 819e9 B/s. `tpu_info` reports 820e9 — it is printed in section 0 of the
notebook, so both numbers are visible in the same run — and the Cloud TPU v5e page states
800 GiBps (858.99e9). The three disagree, and the resulting ridge point spans roughly
229–241 FLOP/byte. None of the verdicts above turns on the difference: the two resolvable
crossovers have margins of 21% and larger, and the one point that would be sensitive to it
(I = 244.0 at `DEFAULT`) is already registered as unresolvable. If a later measurement does
turn on it, prefer `tpu_info` and say so.

## Artifacts

`results/correctness.json`, `results/token_sweep.json`, `results/vmem_sweep.json`,
`results/block_t_sweep.json`, `results/roofline.png` and the xplane trace, bundled by
section 8 of the notebook as `phase2_v5e-20260810-003146.zip`. They are gitignored; the
numbers reach the repo as this document.
