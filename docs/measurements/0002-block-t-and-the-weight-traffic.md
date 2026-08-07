# Measurement 0002 — `block_t` and the weight traffic

**Status: registered 2026-08-07 19:26 UTC, run 2026-08-07 20:55 UTC, verdict filled.**

The prediction was registered in the notebook rather than here: it is the table under
`### 5b · block_t — the term that actually moves bytes` in
[`notebooks/phase2_v5e.ipynb`](../../notebooks/phase2_v5e.ipynb), committed as `d177fe5`
about ninety minutes before the run that scored it. This document quotes that registration
rather than restating it, so the same rule that governs
[0001](0001-fused-gated-mlp-on-v5e.md) applies: nothing in the prediction was adjusted after
the numbers came back, and the commit is what makes that checkable.

- Kernel: [`src/gemma3_pallas/mlp.py`](../../src/gemma3_pallas/mlp.py), `fused_gated_mlp`
- Hardware: one TPU v5e chip, free Colab runtime
- Fixed for every row: `T = 2048`, `block_h = 384`, `DEFAULT` precision,
  `vmem_limit_bytes = 64 MiB`

## Why the sweep exists

0001 established the corrected byte model,

```
bytes = (tokens // block_t) · 3·E·H·dtype  +  2·T·E·dtype
```

and its token sweep holds `block_t` at 128 while `T` grows. That pinning is what isolates
the cost of re-streaming the weights, and it is also why the token sweep cannot see a
crossover: with `block_t` fixed, `T` cancels out of the intensity exactly, and every row
sits at I = 63.2.

So the model has one term under our control, and only one — 0001's prediction 3 established
that `block_h` is not it, since `block_h` does not appear in the byte count at all. Holding
`T` and sweeping `block_t` therefore turns the model into a *shape* rather than a direction,
which is a claim that can fail in two distinguishable ways.

## The registered prediction

| `block_t` | bytes | roof | binds |
|---|---|---|---|
| 128 | 1548 MB | 1890 us | memory |
| 256 | 783 MB | 956 us | memory |
| 512 | 401 MB | 497 us | compute |
| 1024 | 210 MB | 497 us | compute |
| 2048 | 114 MB | 497 us | compute |

> Halving the traffic halves the time until `block_t=512`, where the memory roof crosses
> under the compute roof at 497 us and the curve goes flat. A kernel that keeps improving
> past 512 falsifies the byte model; one that never improves falsifies the diagnosis.

Both failure modes are named in advance, and they point in opposite directions. Continued
improvement past the crossover would mean the weight traffic was never what bounded the
kernel. No improvement at all would mean the traffic is real but not what makes the kernel
slow.

## Result

| `block_t` | bytes | I | roof | binds | measured | % of roof | gain over previous |
|---|---|---|---|---|---|---|---|
| 128 | 1547.7 MB | 63.2 | 1890 us | memory | 2.393 ms | 79.0 | — |
| 256 | 783.3 MB | 124.9 | 956 us | memory | 1.260 ms | 75.9 | 1.90x |
| 512 | 401.1 MB | 244.0 | 497 us | compute | 0.766 ms | 64.9 | 1.65x |
| 1024 | 210.0 MB | 466.0 | 497 us | compute | 0.747 ms | 66.5 | 1.03x |
| 2048 | 114.4 MB | 855.1 | 497 us | compute | 0.785 ms | 63.3 | 0.95x |
| XLA | — | — | — | — | 0.817 ms | — | — |

**The prediction holds, and neither falsification fired.** The kernel improves 1.90x and
1.65x across the two memory-bound halvings, then stops: 1.03x into `block_t=1024` and a 5%
*regression* at 2048, against byte counts that keep halving throughout. The curve flattens
where the roofs cross and not before.

The last row is worth naming rather than smoothing over. At `block_t=2048` the `t` axis is
one step long, so by the single-buffering rule 0001's prediction 3 confirmed, `x` and the
output get one buffer each instead of two — there is no second `t` step to prefetch for, and
nothing left to overlap the first fetch against. The sweep does not isolate that, so it is
an explanation offered, not a thing measured. What it does establish is that the minimum is
interior: 1024, not the largest block available.

## What this does to 0001's headline

0001's token sweep reports the kernel at 0.36x XLA at T=2048. That is true of `block_t=128`
and false of the kernel. At `block_t=1024`, same `T`, same precision, same `block_h`:

| | ms | vs XLA |
|---|---|---|
| kernel, `block_t=128` (0001's pinned geometry) | 2.296 | 0.36x |
| kernel, `block_t=1024` | 0.747 | **1.09x** |
| XLA | 0.817 | — |

A factor of 3.1 between two rows of the same kernel, and the sign of the comparison against
XLA flips. Section 7 of the notebook now prints the second row alongside the first for
exactly this reason; a summary that carries only the pinned block misreports the result.

The comparison is not quite like for like, and the difference favours XLA: XLA materialises
both `[T, hidden_dim]` intermediates in HBM, so it moves bytes this byte model does not
describe, and it reads the weights once regardless. The kernel at `block_t=1024` still reads
them twice. Beating it by 9% while doing that is the fusion paying for the extra pass.

## Caveats

- **Every time here is wall clock**, from `bench.time_call`, and 0001's prediction 4 found
  roughly 228 us per call of host dispatch inside it at T=256. If that overhead is constant
  in `T` — which was not measured — the flat region of this table sits nearer 90% of its
  roof than the 65% printed, and the shape of the curve is unchanged. The verdict does not
  turn on it: it is a claim about where the curve bends, and a constant offset does not move
  a bend.
- **`vmem_limit_bytes = 64 MiB` on every row, including the ones that fit in the default 16.**
  The largest geometry asks 28.13 MiB of scoped VMEM and would not otherwise compile. Raising
  the limit only where it is needed would put a second variable in the sweep.
- **`block_h = 384` throughout**, which is not the 768 quoted under 0001's Geometry section,
  for the reason 0001's prediction 3 records: 768 does not compile at the default scoped-VMEM
  limit. `block_h` does not enter the byte model, so the predicted column is the same at
  either block.
