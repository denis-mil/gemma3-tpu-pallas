# Measurement 0001 — `fused_gated_mlp` on a TPU v5e

**Status: run on 2026-08-07, verdicts filled.** Everything below the horizontal rule was
written *before* any hardware time and none of it has been altered — the verdict columns
are the only cells this session touched. A prediction adjusted after seeing the number is
not a prediction.

Three conditions of the run that no verdict cell can carry:

- **The headline `block_h` was 384, not the 768 quoted under Geometry.** 768 asks 22.5 MiB
  of scoped VMEM and does not compile at the default limit, which is prediction 3's own
  finding. The grid was therefore `(2, 18)`, not `(2, 9)`. Nothing else moves with it:
  `mlp_bytes` does not depend on `block_h`, so the byte model and its intensity of 63.2 are
  the same at either block.
- **`HIGH` never ran.** `jax.lax.Precision.HIGH` does not lower on jax 0.11.0, so the
  precision probe was cut to `DEFAULT` and `HIGHEST`. Every registered `HIGH` claim is
  unscoreable, including the 21%-margin crossover that prediction 1 named its better test.
- **The intensity ladder was traversed by `block_t`, not by `T`.** With `block_t` pinned,
  `I = 6EH·T / (T/block_t · 3EH·4 + 8TE)` and `T` cancels exactly, so the token sweep sits
  at I = 63.2 at every size. Sweeping `block_t` at a fixed `T = 2048` walks 63.2, 124.9,
  244.0, 466.0, 855.1 — prediction 1's ladder exactly — and that is where its claims are
  scored.

- Kernel: [`src/gemma3_pallas/mlp.py`](../../src/gemma3_pallas/mlp.py), `fused_gated_mlp`
- Harness: [`src/gemma3_pallas/bench.py`](../../src/gemma3_pallas/bench.py)
- Driver: [`notebooks/phase2_v5e.ipynb`](../../notebooks/phase2_v5e.ipynb)
- Hardware: one TPU v5e chip, free Colab runtime
- Constants: 197e12 bf16 FLOP/s, 819e9 B/s, 128 MiB VMEM — see the note on bandwidth below

## Geometry

Gemma 3 1B's MLP: `embed_dim = 1152`, `hidden_dim = 6912`, fp32 operands. The grid is
`(tokens // block_t, hidden_dim // block_h)` with the hidden axis **innermost**. The
headline configuration is `T = 256, block_t = 128, block_h = 768`, i.e. grid `(2, 9)`.

Working set per grid step, at the headline configuration:

| | bytes |
|---|---|
| five blocked operands (`x`, `w_gate`, `w_up`, `w_down`, output), double-buffered | 22.5 MiB |
| two `[block_t, block_h]` fp32 intermediates (`gate`, `up`), not buffered | 0.75 MiB |
| **total** | **23.25 MiB** |

The 22.5 MiB figure quoted in earlier sessions omits the intermediates. They are produced
and consumed inside the kernel body, which is what makes the fusion worth doing, but they
are still resident.

---

## The byte model, corrected

The `w_*` `index_map`s vary in the innermost grid index. So one `t` step walks every
hidden block; when that index resets for the next `t` step, block 0 is no longer the
previously fetched slice, and copy elision — which skips a transfer only between two
*consecutive* identical slices — cannot apply. The weights are therefore re-read **once
per `t` step**:

```
bytes = (tokens // block_t) · 3·E·H·dtype  +  2·T·E·dtype
```

At the headline geometry that is **193.5 MB and intensity 63**, not 97.9 MB and 125. Both
counts are available: `shapes.mlp_bytes(..., elide_weights=True)` gives the one-pass
counterfactual, so the two models can be printed side by side and the xprof DMA count in
cell 6 adjudicates between them.

This is an extension of a rule the workspace already wrote down — *an intensity argument is
only valid if every operand is in the byte count* — to **every visit**, not merely every
operand.

## The three roofs

`tpu_info` publishes no fp32 peak for v5e, and there is none to publish: TPU emulates an
fp32 matmul with several bf16 passes, which `jax.lax.DotAlgorithmPreset` names
`BF16_BF16_F32_X3` and `_X6`. So the compute roof is a function of the requested precision,
and `roofline_bound` takes `peak_flops` as a **required** keyword for that reason. No
`peak_fp32_flops` constant was added; inventing one is the failure mode the required
keyword exists to prevent.

| `precision` | bf16 passes | effective peak | ridge (FLOP/byte) |
|---|---|---|---|
| `DEFAULT` | 1 | 197 TFLOP/s | 240.5 |
| `HIGH` | 3 | 65.7 TFLOP/s | 80.2 |
| `HIGHEST` | 6 | 32.8 TFLOP/s | 40.1 |

## Registered prediction 1 — where the crossovers are

At `block_t = tokens` (one `t` step, weights read once) the intensities are:

| T | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|
| I (FLOP/byte) | 63.2 | 124.9 | 244.0 | 466.0 | 855.1 |

Against each precision's ridge:

| `precision` | ridge | HBM-bound at | at the ridge (unresolvable) | compute-bound at |
|---|---|---|---|---|
| `DEFAULT` | 240.5 | 128, 256 | **512** (244.0 — 1.4% above) | 1024, 2048 |
| `HIGH` | 80.2 | **128** (63.2 — 21% below) | — | 256, 512, 1024, 2048 |
| `HIGHEST` | 40.1 | — | — | every T (128 is 58% above) |

Two corrections to the obvious reading, both registered in advance:

1. **`HIGH` is not compute-bound at every T.** At T=128 the kernel sits 21% *below* its
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
| `DEFAULT` is HBM-bound at T=256 | **held** — at I=124.9 the kernel takes 1.260 ms against 2.393 ms at I=63.2, a 1.90x gain for a 1.98x cut in bytes |
| `DEFAULT` is compute-bound at T=1024 | **held** — I=466.0 gives 0.747 ms against 0.766 ms at I=244.0, and halving the bytes again to I=855.1 makes it *slower* at 0.785 ms |
| `DEFAULT` at T=512 is unresolvable | *registered as unscoreable* — measured 0.766 ms, 2.5% off the floor, which fits either roof |
| `HIGHEST` is compute-bound at every T | **held** — kernel and XLA land within 4% of each other at every T (1.02–1.04x from T=256 up) even though XLA moves half the bytes; under a memory roof the lighter traffic would win |

## Registered prediction 2 — the on-device error

The CPU interpret run's largest absolute error at Gemma dims was 3.28e-6, against an
output scale of about 2.15. On hardware at `DEFAULT` precision the multiplies are bf16, so
the error is expected to be **far larger** and `GEMMA_RTOL = 1e-4` is expected to fail.
That tolerance is a CPU measurement and does not transfer, so the notebook **reports** the
error rather than asserting an inherited number.

The precision whose error approaches the CPU figure is the one doing full-mantissa work,
and its pass count is the divisor to use on the 197 TFLOP/s roof. That is how the roof gets
chosen by measurement rather than by assumption.

| Claim | Verdict |
|---|---|
| `DEFAULT` error ≫ 3.28e-6 | **held** — 9.64e-3 absolute against an output scale of 2.15, about 2900x the CPU figure, and a max *relative* error of 3.03 |
| error falls monotonically DEFAULT → HIGH → HIGHEST | **partly scoreable** — the one measurable leg falls, 9.64e-3 to 2.05e-5; the `HIGH` point does not exist |
| some precision reproduces the CPU error | **not held** — `HIGHEST` reaches 2.05e-5, still 6.3x the CPU's 3.28e-6, and it is the closest the toolchain offers. Its 6 passes were used as the divisor anyway, which is what the prediction wanted the error for |

## Registered prediction 3 — the VMEM budget

`pltpu.InterpretParams` models no VMEM capacity at all, so this cannot be asked anywhere
but on hardware. The physical capacity is 128 MiB; what a `pallas_call` gets is the
`--xla_tpu_scoped_vmem_limit_kib` default, which is not published.

No prediction is registered for its value — there is no basis for one. What is registered
is the **method**: sweep `block_h` at the default limit until compilation fails, and the
largest working set that compiles and the smallest that does not bracket the budget. Then
re-run the first failing config with explicit `vmem_limit_bytes` values to confirm the flag
moves that boundary.

| Claim | Verdict |
|---|---|
| the default limit lies strictly between two swept working sets | **held** — 12.375 MiB compiles, 22.5 MiB does not, and no admissible `block_h` lies between them. The bracket did not need to be narrowed: the compiler names its own limit, "size 22.50M and limit 16.00M" |
| `vmem_limit_bytes` moves the boundary, without a runtime restart | **held** — `block_h=768` fails at 16 MiB and runs at 32 (0.499 ms); `block_h=6912` fails through 64 MiB and runs at 100 (0.399 ms). The second row also confirms the single-buffering rule: its 91.1 MiB of weights fit under 100 MiB only because the `h` axis is one step long, and two buffers could not have |

## Registered prediction 4 — copy elision

The docs state copy elision as a property of the TPU pipeline, but that pipeline is emitted
by Mosaic and is out of reach from Python. The two byte models above differ by exactly a
factor of `tokens // block_t` in the weight traffic, so an xprof DMA count settles it. At
the headline geometry: **2 transfers per weight per call** under the corrected model, 1
under elision.

| Claim | Verdict |
|---|---|
| `w_*` DMAs per call = `tokens // block_t` × 3, not 3 | **held** — see below |

The count did not come from where the prediction expected it. `Pallas Primitives` is empty
in the capture even under `tpu_trace_mode="TRACE_COMPUTE_AND_SYNC"`, so there is no row of
named DMA events to tally. What that mode does add is the `Tensor Core Sync Flag` line, and
the DMAs are visible there as the waits that retire them: four flags, 18 waits each, 72 per
call, in a rigid 14.03 us cycle of `SyncWait:55, 57, 56, 58` — two streams, double-buffered.
The grid is 36 steps and every one of them must execute, so 18 cycles cover 36 steps at two
waits per step.

The decisive evidence is positional rather than arithmetic. Under elision the second `t`
pass — steps 18 through 35 — fetches no weights at all, so the waits stop at the halfway
mark. They do not: the spacing is uniform end to end, the last `SyncWait:58` landing at
255.5 us of a 258.4 us call on the same 14.03 us rhythm as the first. Both `t` passes fetch,
which is the corrected model.

Two supporting numbers. 187.2 us of the 258.4 us call, 72.4%, is spent blocked on those
flags; a kernel moving only the elided model's 95.5 MB needs 117 us of transfer in total and
cannot spend 187 us waiting for it. And per grid step the kernel moves 5.31 MB in 7.02 us,
756 GB/s, which is 92% of the 819 GB/s roof — the corrected model's traffic running at very
nearly the speed the hardware can supply it.

One instrumentation finding falls out of the same trace, and it applies to every wall-clock
number this notebook reports. Device time at the headline geometry is 258.4 us per call
(258.40, 258.53, 258.43, 258.56, 258.43 across five calls) against a wall-clock median of
486 us, so roughly 228 us per call is host dispatch that no roof accounts for.
`bench.time_call` measures wall clock, so the `%roof` column understates on-chip efficiency
badly at small `T` — the kernel reaches 91% of its memory roof at T=256, not the 48.6%
printed. It does not rescue the kernel against XLA: subtracting a constant from both makes
the ratio worse, not better.

## Notes on the constants

`shapes.py` carries 819e9 B/s. `tpu_info` reports 820e9 and the Cloud TPU v5e page states
800 GiBps (858.99e9). The three disagree, and the resulting ridge point spans roughly
229–241 FLOP/byte. None of the verdicts above turns on the difference: the two resolvable
crossovers have margins of 21% and larger, and the one point that would be sensitive to it
(T=512 at `DEFAULT`) is already registered as unresolvable. If a later measurement does
turn on it, prefer `tpu_info` and say so.
