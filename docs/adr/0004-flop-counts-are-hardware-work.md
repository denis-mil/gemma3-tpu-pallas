# FLOP counts are hardware work; the bf16 pass count is in the numerator

`shapes.mlp_flops` returns the FLOPs the **MXU issues**, not the FLOPs the algebra
specifies, and it takes a required `passes` argument to say how many bf16 passes the
requested precision costs. `TpuV5e.peak_flops` is one published constant, 197 TFLOP/s,
and `ridge_point` is one, 240.5 FLOP/byte. Nothing divides either.

This replaces the earlier convention, in which `mlp_flops` counted algebraic FLOPs and
`peak_flops(passes)` / `ridge_point(passes)` divided the roof.

## Why

v5e has no fp32 matmul unit. `tpu_info` publishes no fp32 peak and there is none to
publish: an fp32-precision matmul is *emulated* with three or six bf16 passes, which
`jax.lax.DotAlgorithmPreset` names `BF16_BF16_F32_X3` and `_X6`. That physics is not in
dispute and neither convention changes it. What was wrong was where the pass count was
booked.

Dividing the roof puts a property of the *work* onto the *hardware*. It produced two
numbers, 65.7 and 32.8 TFLOP/s, that describe nothing physical — the chip does not
acquire a lower ceiling because a kernel asked for more mantissa, it does six times as
much work at the same ceiling. And it split the roofline plot into three stacked lines,
one of which (`HIGH`, 3 passes) was not lowerable by Mosaic on jax 0.11.0 and had to be
drawn dashed and disclaimed: a roof no measurement could ever belong to.

## The identity that makes this safe

    passes · I  >  P/β        is the same inequality as        I  >  (P/passes)/β

So **no predicted time and no bound verdict moves.** Every crossover registered in
`docs/measurements/0001` reads true under either convention, at the same token count for
the same precision. `tests/test_bench.py::test_moving_the_pass_count_changes_no_prediction`
pins this directly: it writes the old convention out inline, as a fixture, and asserts
agreement in both seconds and which roof binds across the full tokens × passes grid.

## Consequences

**Arithmetic intensity now scales with precision.** The same geometry has I = 63.2 at
`DEFAULT` and 379.3 at `HIGHEST`, on a flag that moves not one byte. This contradicts a
sentence already published in lesson 06 — that intensity belongs to the algorithm and its
schedule — and the honest repair is to widen the claim rather than patch it: intensity is
a fact about a kernel *as executed*, precision included, because the same source at two
precisions is two different amounts of work for the machine. The gain is that the
roofline can now explain why `HIGHEST` is slower, which under the old convention required
moving a roof that plainly does not move.

**The achieved-throughput axis is a hardware rate.** This is the real cost. At T = 2048,
`HIGHEST`, the kernel plots at 177 TFLOP/s on a 197 TFLOP/s chip — 90% of peak — while
doing 29.5 TFLOP/s of the matmul the model asked for. The old convention read that row as
29.5 against a 32.8 roof, which was *both* the model rate and 90% of the applicable
ceiling; it gave both readings for free, and that is what is being given up. Four
mitigations, all of them load-bearing:

- `bench.summarise` prints a `p` column and a legend line: *divide by the p column for
  model FLOPs*.
- The notebook's y axis reads `achieved hardware TFLOP/s (bf16 passes counted)`.
- `CONTEXT.md` defines **bf16 pass**, **Hardware FLOPs** and **model FLOPs** under
  Measurement.
- Section 7 of the notebook states both numbers side by side rather than only the larger.

**`passes` is required on `mlp_flops` and defaulted to 1 on `attention_flops`.** The
required keyword did not disappear when `roofline_bound`'s `peak_flops` gained a default —
it moved to the parameter that actually cannot be inherited. "How many FLOPs is this
matmul" has no dtype-free answer here, and a call site inheriting 1 reports six times the
throughput an fp32-precision kernel achieved. `attention_flops` is different in kind:
every consumer of it is a windowed-against-dense *ratio*, where a common factor cancels,
and the material it feeds is bf16 throughout.

**`summarise` derives the count per row** from `r.precision`, via `bench.passes_for`. It
used to take a table-level `passes`, a second copy of a fact every `BenchResult` already
carried; `summarise(highest_rows, passes=1)` was a silently six-times-wrong table. One
table can now hold both precisions against the one roof — which is the story the
single-roof plot tells — so the kernel-vs-XLA ratio block is keyed on (precision, tokens)
rather than on tokens alone.

**"Pass" is now overloaded**, three ways: a bf16 emulation pass, a weight re-read, and a
softmax pass. `mlp_bytes`'s local is renamed `weight_reads`, the docs take the same word
swap, and `CONTEXT.md` lists "weight pass" as a phrase to avoid.
