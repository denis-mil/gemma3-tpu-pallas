# Measurement 0001 — `fused_gated_mlp` on a TPU v5e

**Status: predictions registered, not yet run.** Everything below the horizontal rule is
written *before* any hardware time, and the verdict column is deliberately empty. Filling
it in is the only edit this document should receive from the Colab session; a prediction
adjusted after seeing the number is not a prediction.

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
| `HIGH` crosses between T=128 and T=256 | *not yet run* |
| `DEFAULT` is HBM-bound at T=256 | *not yet run* |
| `DEFAULT` is compute-bound at T=1024 | *not yet run* |
| `DEFAULT` at T=512 is unresolvable | *registered as unscoreable* |
| `HIGHEST` is compute-bound at every T | *not yet run* |

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
| `DEFAULT` error ≫ 3.28e-6 | *not yet run* |
| error falls monotonically DEFAULT → HIGH → HIGHEST | *not yet run* |
| some precision reproduces the CPU error | *not yet run* |

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
| the default limit lies strictly between two swept working sets | *not yet run* |
| `vmem_limit_bytes` moves the boundary, without a runtime restart | *not yet run* |

## Registered prediction 4 — copy elision

The docs state copy elision as a property of the TPU pipeline, but that pipeline is emitted
by Mosaic and is out of reach from Python. The two byte models above differ by exactly a
factor of `tokens // block_t` in the weight traffic, so an xprof DMA count settles it. At
the headline geometry: **2 transfers per weight per call** under the corrected model, 1
under elision.

| Claim | Verdict |
|---|---|
| `w_*` DMAs per call = `tokens // block_t` × 3, not 3 | *not yet run* |

## Notes on the constants

`shapes.py` carries 819e9 B/s. `tpu_info` reports 820e9 and the Cloud TPU v5e page states
800 GiBps (858.99e9). The three disagree, and the resulting ridge point spans roughly
229–241 FLOP/byte. None of the verdicts above turns on the difference: the two resolvable
crossovers have margins of 21% and larger, and the one point that would be sensitive to it
(T=512 at `DEFAULT`) is already registered as unresolvable. If a later measurement does
turn on it, prefer `tpu_info` and say so.
