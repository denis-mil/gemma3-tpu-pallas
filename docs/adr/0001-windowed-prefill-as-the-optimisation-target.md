# Windowed prefill attention is the optimisation target

Four kernels were candidates for the project's headline result: a fused gated MLP,
an int8 weight-only matmul, windowed decode attention, and windowed prefill
attention. We chose windowed prefill.

The deciding argument is *how* each one would win. The MLP and int8 kernels both
have to beat XLA at work XLA already does well — fusing gated feed-forward blocks
and lowering quantised matmuls — so the likely outcome is parity after considerable
effort. Windowed prefill wins structurally instead: twenty-two of Gemma 3 1B's
twenty-six layers attend to at most 512 positions, so at a 32K prompt a local layer
needs roughly 1.6% of dense attention's score computations. The naive path computes
those pairs and then masks them away. Skipping them is guaranteed by arithmetic, not
by out-tuning anyone.

Windowed decode was the closest alternative and remains the more valuable kernel —
decode is where real serving cost lives. It was rejected on risk. MQA gives decode a
`[4, 256]` query against a 128×128 MXU, so most of the work is recovering
utilisation from a shape that cannot fill a tile; and because weights (2.0 GB) cost
13× more bandwidth than the windowed KV cache (145.8 MB) at batch 1, the kernel does
not affect end-to-end latency until batch 8 or so. On a free Colab runtime that is a
plausible way to spend the entire quota on a kernel that is correct and slow.

A fused gated MLP is still built first, deliberately, as a warm-up. Its purpose is
to teach grids, `BlockSpec`s and VMEM budgeting against a problem with trivially
checkable correctness, so that online softmax is the only unfamiliar thing in the
attention kernel. Parity with XLA is an acceptable result for it.

## Consequences

Benchmarks compare against two things, not one: the naive baseline, which the kernel
is expected to beat, and `splash_attention`, which already implements sliding-window
block skipping and which the kernel is *not* expected to beat. Reporting the gap to
the ceiling, with a roofline analysis explaining it, is treated as a result rather
than a failure.
