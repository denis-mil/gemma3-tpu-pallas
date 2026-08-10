# Gemma 3 TPU Pallas

Custom TPU kernels written in Pallas and benchmarked at the exact shapes Gemma 3 1B
runs. The project optimises inference arithmetic; it does not serve the model.

## Language

### Inference phases

**Prefill**:
Processing an entire prompt in one pass, where every position is known up front.
Compute-bound and MXU-friendly.
_Avoid_: prompt processing, encoding, the forward pass

**Decode**:
Generating one token at a time, each step attending over all previous positions.
Bandwidth-bound.
_Avoid_: generation, sampling, autoregressive step

### Attention structure

**Local layer**:
An attention layer whose queries may only attend to the most recent `window`
positions. Twenty-two of Gemma 3 1B's twenty-six layers are local.
_Avoid_: sliding layer, SWA layer, banded layer, short-range layer

**Global layer**:
An attention layer whose queries may attend to every earlier position. Four of
Gemma 3 1B's layers are global, at indices 5, 11, 17 and 23.
_Avoid_: full layer, dense layer, long-range layer

**Window**:
The number of most-recent positions a local layer may attend to. 512 in Gemma 3 1B.
_Avoid_: context window (which means the model's whole 32K limit), span, block size

**MQA**:
An attention layout where all query heads share a single key/value head. Gemma 3 1B
has four query heads and one KV head.
_Avoid_: GQA (which permits more than one KV head), multi-head attention

**KV cache**:
The stored keys and values from earlier positions that decode attends over.
_Avoid_: attention cache, past, memory

### Kernels

**Kernel**:
A single unit of work handed to the accelerator and executed without host
interaction. Here, always one written in Pallas.
_Avoid_: op, custom op, fused op

**Block skipping**:
Declining to load or compute a tile of keys and values that the mask would have
discarded anyway. The mechanism by which a local layer does less work rather than
merely masking more.
_Avoid_: sparsity, pruning, early exit, masking out

**Interpret mode**:
Running a Pallas kernel on CPU as ordinary JAX, so correctness can be checked
without accelerator hardware. Not evidence of correctness on real hardware.
_Avoid_: simulation, emulation, CPU mode, dry run

### Measurement

**Reference**:
The pure-JAX fp32 implementation a kernel is asserted against. Ground truth for
correctness, never for speed.
_Avoid_: golden, oracle, expected, ground truth

**Naive baseline**:
Dense attention that materialises the full score matrix and then applies the mask.
The thing a kernel is expected to beat.
_Avoid_: baseline (unqualified), vanilla, unoptimised

**Ceiling**:
`splash_attention`, Google's production kernel. A kernel is measured against it to
size the remaining gap, not to beat it.
_Avoid_: competitor, target, SOTA

**Roofline**:
The lower bound on a kernel's runtime implied by peak compute and peak bandwidth,
used to say which of the two is binding.
_Avoid_: theoretical max, peak, limit

**bf16 pass**:
One traversal of an operand pair through the MXU at bf16. v5e has no fp32 matmul
unit, so it emulates an fp32-precision multiply with 3 or 6 of them
(`DotAlgorithmPreset` `BF16_BF16_F32_X3` and `_X6`); `DEFAULT` is one.
_Avoid_: iteration, round, repeat, weight pass — a weight re-*read* is not a pass

**Hardware FLOPs**:
What the MXU issues, pass count included. What this project's counters return and
what both axes of a roofline here carry. Distinguish from **model FLOPs**, which is
hardware FLOPs divided by the pass count and is what a published throughput figure
quotes.
_Avoid_: unqualified "FLOPs" wherever a precision above `DEFAULT` is in play
