# Pallas TPU: the grid, `BlockSpec`, and `index_map` — what gets DMA'd, and when

Research findings, not a lesson. Every claim below carries the source that owns it.
Gathered against JAX 0.11.0 (the version pinned in this repo) and the JAX docs tree at
`main`. Experiments were run on CPU with
`C:\Users\milov\.conda\envs\gemma3-tpu-pallas\python.exe`.

Closes the `RESOURCES.md` gap: *"No worked Pallas tutorial at the level of 'here is why
this `index_map` and not that one'."*

---

## What is verified vs what is inferred

**Verified by execution** (run on this machine under Pallas interpret mode; scripts
reproduced inline below):

- Grid traversal is lexicographic with the **last** grid axis varying fastest.
- `index_map` returns block indices; element offset = `block_index × block_size`.
- **The HBM→VMEM input copy is skipped when the block index is unchanged from the
  previous step** — demonstrated directly, by scribbling a sentinel into the input
  buffer and observing that it survives into the next step.
- Which operand stays resident flips when you swap the grid axis order, with a
  4× difference in DMA count over the same 12 grid steps.
- Output revisiting acts as an accumulator; putting the reduction on a leading grid
  axis instead of the trailing one produces a wrong answer.
- `None` in `block_shape` squeezes the axis out of the ref.
- Out-of-bounds block elements are padded, with NaN under interpret mode.
- `dimension_semantics=("parallel", ...)` visibly reorders grid traversal under
  interpret mode.
- Interpret mode does **not** enforce the TPU `(8, 128)` block-shape rule.
- v5e per-TensorCore hardware numbers as JAX reports them (`pltpu.get_tpu_info_for_chip`).

**Verified by reading a primary source** (documentation prose or JAX source, not run):

- The per-step prologue/steady-state/epilogue schedule of the double-buffered pipeline.
- That the skip-on-unchanged-index behaviour is *documented*, not merely observed.
- The `(8, 128)` block-shape divisibility rule and the rank-1 rule.
- The VMEM out-of-memory failure mode.
- `memory_space` semantics (`ANY`/`VMEM`/`SMEM`/`SEMAPHORE`) and the scalar-prefetch path.
- `pallas_call` on TPU accepts only buffer counts 1 and 2.

**Inferred — reasoning I did, not a claim any source makes:**

- The VMEM budget arithmetic for a Gemma 3 1B windowed-prefill kernel (§4.3). The
  *rules* are cited; multiplying them out for our shapes is mine.
- That interpret mode's copy-elision logic faithfully mirrors what the Mosaic compiler
  emits on hardware. The interpreter clearly implements the *same rule* (§3.4), but the
  hardware pipeline is emitted by the Mosaic/XLA compiler, which is not in the Python
  source I can read. **See jax-ml/jax#36287 — interpret mode is not evidence of hardware
  behaviour.** Treat §3.3's experiments as evidence that the rule is real and that JAX's
  own simulator believes it, not as a measurement of DMA traffic on a v5e.

**Not verified at all** — see [Still open](#still-open).

---

## 1. Grid semantics on TPU

### 1.1 The grid is a loop nest

> `pl.pallas_call(some_kernel, grid=(n, m))(...)` is equivalent to
> ```python
> for i in range(n):
>   for j in range(m):
>     some_kernel(...)
> ```
> This generalizes to any tuple of integers (a length `d` grid will correspond to `d`
> nested loops). The kernel is executed as many times as `prod(grid)`.

— [Grids and BlockSpecs](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html#grid-a-k-a-kernels-in-a-loop)

Dimension 0 is the outermost loop. **The last grid axis varies fastest.** The docs state
this only implicitly, via the `for` nest; I confirmed it by execution (§1.3).

### 1.2 Sequential is a guarantee, not an accident

> What's more, compared to GPUs, TPUs are actually highly sequential machines. Ergo, the
> grid is generally not processed in parallel, but sequentially, in lexicographic order
> (though see the *Multicore TPU configurations* section for exceptions).

— [Writing TPU kernels with Pallas §Noteworthy properties](https://docs.jax.dev/en/latest/pallas/tpu/details.html#noteworthy-properties-and-restrictions)

The escape hatch is `dimension_semantics`:

> `dimension_semantics` should be a tuple of same length as `grid` where each entry is
> either `"parallel"` or `"arbitrary"`. `"parallel"` indicates to Pallas that the
> iterations of the for loop corresponding to that dimension can be executed
> independently without affecting the correctness of the program. `"arbitrary"`
> indicates to Pallas that there can be no assumptions made about this grid dimension
> and it therefore cannot be parallelized.

— [TPU Pipelining §Megacore](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html#tpus-in-megacore-configuration)

> Note that Megacore is only currently available on TPU `v4` and TPU `v5p`. Supplying
> `dimension_semantics` annotations is a no-op on other platforms, but *not* specifying
> it will result in only one TensorCore being used (even if there are more than one
> available).

— same page.

**Consequence for this project.** v5e has **one** TensorCore per chip
(`ChipVersion.TPU_V5E.num_physical_tensor_cores_per_chip == 1`,
[`tpu_info.py`](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/tpu_info.py)),
and `supports_megacore` returns `False` for it. So on our target, `dimension_semantics`
buys nothing at run time — but it still *documents intent*, and the same kernel run on a
v5p would change behaviour. Note also that JAX's TPU interpret mode randomises the
traversal of `"parallel"` axes (§1.3), so annotating an axis `"parallel"` when it isn't
will show up as a correctness failure on CPU even though v5e would never reorder it.
That makes `"parallel"` a *portability assertion you can test locally*, which is worth
more than the zero speedup.

The two docs are not phrased identically — see [Disagreements](#disagreements-found).

### 1.3 Execution evidence

`pltpu.InterpretParams` exposes a `grid_point_recorder` callback, documented as invoked
"for each grid point in the order in which the grid points are traversed"
([`interpret/params.py`](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/interpret/params.py)).
With `grid=(2, 3, 4)` and all axes `"arbitrary"`:

```
first 10: [(0,0,0), (0,0,1), (0,0,2), (0,0,3), (0,1,0), (0,1,1), (0,1,2), (0,1,3), (0,2,0), (0,2,1)]
len: 24
lexicographic, last axis fastest: True
```

With `grid=(3, 2)` and `dimension_semantics=("parallel", "arbitrary")`, the recorded
order depends on `random_seed`:

```
seed=     0 -> [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)]
seed=     1 -> [(2,0), (2,1), (0,0), (0,1), (1,0), (1,1)]
seed=     7 -> [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)]
seed= 12345 -> [(2,0), (2,1), (0,0), (0,1), (1,0), (1,1)]
```

Declaring an axis `"parallel"` is a promise you are making; the interpreter will try to
break you on it.

---

## 2. The pipeline: what copies happen, and when

### 2.1 Where the operands live

> Pipelining on TPUs is typically done between HBM (DRAM) to VMEM (Vector SRAM). The
> default behavior for `pallas_call` on TPU is that arguments to `pallas_call` are
> assumed to live in HBM, and inputs to the user kernel body are stored in VMEM.

— [TPU Pipelining §TPU Memory Spaces](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html#tpu-memory-spaces)

> While the inputs to `pallas_call` will often reside in HBM (the main TPU memory), the
> references passed in to the kernel body will point to buffers in lower levels of
> memory hierarchy (VMEM or SMEM). This enables the kernel body to write and read them
> at very high speeds, while all the communication with HBM (which has very high
> latency) is handled by the compiler and overlapped with compute.

— [Writing TPU kernels with Pallas](https://docs.jax.dev/en/latest/pallas/tpu/details.html#noteworthy-properties-and-restrictions)

### 2.2 The per-step schedule

The [Software Pipelining](https://docs.jax.dev/en/latest/pallas/pipelining.html) guide
derives the schedule from first principles and lands on this pseudocode — the single
most useful thing in the whole doc tree for this question, quoted verbatim:

```python
def double_buffered_pipeline(
    grid: tuple[int, ...],
    kernel: Callable,
    in_slices: Callable,
    out_slices: Callable):
  # Prologue
  copy_in_start(in_hbm[in_slices(0)], in_sram[0])

  # Main loop
  grid_size = prod(grid)
  for i in range(grid_size):
    cur_slot = i % 2
    next_slot = (i + 1) % 2
    if (i + 1) < grid_size:
      copy_in_start(in_hbm[in_slices(i+1)], in_sram[next_slot])
    copy_in_wait(in_sram[cur_slot])

    kernel(in_sram[cur_slot], out_sram[cur_slot])

    copy_out_start(out_sram[cur_slot], out_hbm[out_slices(i)])
    if i > 0:
      copy_out_wait(out_sram[next_slot])

  # Epilogue
  last_slot = (grid_size - 1) % 2
  copy_out_wait(out_sram[last_slot])
```

— [Software Pipelining §Deriving a Double-Buffered Pipeline](https://docs.jax.dev/en/latest/pallas/pipelining.html#deriving-a-double-buffered-pipeline)

So the sequence for grid step `i` is: **start the fetch for `i+1` → wait on the fetch for
`i` → run the kernel body → start the store for `i` → wait on the store for `i-1`.** The
`copy_in_start` for step `i+1` is issued *before* the kernel body for step `i` runs; that
is the whole trick. Copies are asynchronous DMAs, so the compute for step `i` overlaps
the fetch for step `i+1` and the store for step `i-1`.

The reason two buffers are needed at all is stated plainly:

> in the current state of the loop there is a fake data dependency through X — we cannot
> simultaneously perform an async copy into X while using it for computation or else we
> may have a race condition. Therefore, we can use a **multiple-buffering** technique
> where we keep 2 buffers for each input X and each output Y.

— same page.

### 2.3 How many buffers

> The default buffer count is 2 for all inputs and outputs.

— [TPU Pipelining §Multiple Buffering](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html#multiple-buffering)

For `pallas_call` specifically, more than 2 is rejected at lowering time:

```
"Only single (1) and double (2) buffering are supported. Got {buffer_count}."
```

— [`jax/_src/pallas/mosaic/lowering.py`](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/lowering.py),
in the `BlockSpec` → Mosaic `window_params` lowering (JAX 0.11.0). The same block also
raises `NotImplementedError("Lookahead is not supported for XLA pipeline emitter
lowering.")`. Buffer counts > 2 and lookahead are `pltpu.emit_pipeline` features only.
This resolves an apparent doc conflict — see [Disagreements](#disagreements-found).

There is one automatic exception. If a `BlockSpec`'s block covers the whole array and its
`index_map` returns literal zeros, lowering forces single buffering:

```python
      # Force single-buffering pipelining for trivial windowing in VMEM.
      pipeline_mode = bm.pipeline_mode
      if (
          tpu_memory_space == tpu_core.MemorySpace.VMEM
          and bm.has_trivial_window()
      ):
        pipeline_mode = pallas_core.Buffered(1)
```

— `lowering.py`, with `BlockMapping.has_trivial_window` documented in
[`jax/_src/pallas/core.py`](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/core.py)
as *"If block shape is same as the array shape and index_map returns 0s."* A whole-array
operand costs one buffer, not two.

### 2.4 What the pipeline costs

> In a **compute-bound** regime, a pipeline running $N$ iterations would take
> $(\alpha + X/\beta) + N (Y/F)$ seconds, where the first term represents the cost of
> the initial bubble (multiply by a factor of 2 if there is also a bubble at the end),
> and the second term represents the total time of the steady-state of the pipeline.

— [Software Pipelining §Analyzing the performance](https://docs.jax.dev/en/latest/pallas/pipelining.html#analyzing-the-performance)

with $\alpha$ memory latency, $\beta$ bandwidth, $F$ FLOP/s, $X$ bytes moved per
iteration and $Y$ FLOPs per iteration. This is the hook for Phase 6's roofline.

---

## 3. `index_map` mechanics — the crux

### 3.1 It returns block indices

> Informally, the `index_map` of the `BlockSpec` takes as arguments the invocation
> indices (as many as the length of the `grid` tuple), and returns **block indices** (one
> block index for each axis of the overall array). Each block index is then multiplied by
> the corresponding axis size from `block_shape` to get the actual element index on the
> corresponding array axis.

— [Grids and BlockSpecs §BlockSpec](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html#blockspec-a-k-a-how-to-chunk-up-inputs)

The docs give an executable definition, `slices_for_invocation`, on the same page. Its
worked example:

```python
>>> slices_for_invocation(x_shape=(100, 100),
...                       x_spec = pl.BlockSpec((10, 20), lambda i, j: (i, j)),
...                       grid = (10, 5),
...                       invocation_indices = (2, 4))
[slice(20, 30, None), slice(80, 100, None)]
```

Verified by execution: with `x = arange(32*128).reshape(32,128)`, `grid=(4,)`,
`BlockSpec((8, 128), lambda i: (i, 0))`, the first element of each block was
`[0., 1024., 2048., 3072.]` — exactly `8*128*i`.

The default when `index_map=None` is
`lambda *invocation_indices: (0,) * len(block_shape)`, and when `block_shape=None` the
array's own shape is used ([same page](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html#blockspec-a-k-a-how-to-chunk-up-inputs)).

### 3.2 Consecutive steps mapping to the same block: the copy is skipped

**This is documented behaviour, not an implementation detail.** Two independent places in
the docs say so.

> * When two (lexicographically) consecutive grid indices use the same slice of an input,
>   the HBM transfer for the second iteration is skipped, as the data is already
>   available.
>
> * Multiple invocations of the kernel body can write to the same slice of the output,
>   without any risk of race conditions. However, we do require that all invocations that
>   write to a particular slice are consecutive.

— [Writing TPU kernels with Pallas](https://docs.jax.dev/en/latest/pallas/tpu/details.html#noteworthy-properties-and-restrictions).
Note the framing: the doc presents this as a *capability that sequential execution
unlocks* ("This unlocks some interesting capabilities"), which is the strongest available
signal that it is contractual rather than incidental.

And, in the general pipelining guide, describing the mechanism by name:

> The Pallas pipeline emitter performs an optimization where if the data slices between
> two consecutive iterations are the same, the pipeline will not issue a
> `copy_in`/`copy_out` on that buffer. This means the same SRAM buffer used in a previous
> iteration will be passed into the kernel again on the following iteration, and thus any
> writes that were issued to the output buffer will become visible on the next iteration.
> Once the data slice changes, the final accumulated SRAM buffer will be written out to
> HBM.

— [Software Pipelining §Reductions and accumulation](https://docs.jax.dev/en/latest/pallas/pipelining.html#reductions-and-accumulation)

Two things follow immediately, both stated by the docs:

1. **Residency is a property of the fastest-varying axis.** The operand whose `index_map`
   ignores the last grid axis is fetched once per *outer* step, not once per grid step.
2. **Accumulation must live on the last grid axis.** From the same section:
   *"Reduction/accumulation should only be performed over the last (innermost) dimensions
   of the grid, and the buffer should be initialized manually first."* And from
   `details.rst`: *"The grid axis corresponding to the reduction dimension has to be the
   last one, since the output window does not vary along this axis. The output reference
   can be then used as an accumulator for partial results."*

`details.rst` also names the general shape of the constraint:

> The "consecutive" restriction on the output usually means that some prefix of the grid
> dimensions always varies the slice of the output an invocation needs to access, while
> the output window remains constant for the remaining suffix.

### 3.3 Execution evidence for the skip

The cleanest probe is to violate the read-only rule on purpose. The kernel reads
`x_ref[0,0]`, records what it saw, then writes a sentinel into `x_ref[0,0]`. If the
pipeline re-copies from HBM next step, the sentinel is clobbered; if the copy is elided,
the *same VMEM buffer* comes back and the sentinel survives.

```python
def k3(x_ref, o_ref):
    o_ref[0, 0] = x_ref[0, 0]      # what did this step actually see?
    x_ref[0, 0] = 99.0             # scribble on the input buffer
```

With `grid=(4,)` over a zero-filled array:

```
invariant index_map  (lambda i: (0, 0)) -> [ 0. 99. 99. 99.]   # 1 DMA,  3 reuses
varying   index_map  (lambda i: (i, 0)) -> [ 0.  0.  0.  0.]   # 4 DMAs, 0 reuses
```

Identical under `interpret=True` and under `pltpu.InterpretParams()`.

**Pitfall in this technique — added on independent re-verification.** A *constant*
sentinel is only safe when each block is visited at most once, as above. Under interpret
mode a write to an input ref **persists into the backing array**, so when a grid revisits
a block, the copy is re-issued and brings the sentinel *back* — indistinguishable from
elision. Re-running this probe with a constant `99.0` on the 12-step grid below reports
`K: 4 DMAs`, which is wrong; the correct answer is 12. Use a **per-step** sentinel
(`1000 + step`) and classify a read as `reuse` only when it equals the sentinel written
at the *immediately preceding* step. Anything else is a copy, whether the source was
clean or polluted.

Scaling this up to an attention-shaped grid is where it earns its keep. Two operands,
`grid=(3, 4)`, Q's `index_map` ignoring the fast axis and K's following it:

```python
in_specs=[
    pl.BlockSpec((8, 128), lambda i, j: (i, 0)),   # Q: ignores j
    pl.BlockSpec((8, 128), lambda i, j: (j, 0)),   # K: follows j
]
```

```
Q buffer, rows = i, cols = j:        K buffer, rows = i, cols = j:
[['DMA  ' 'reuse' 'reuse' 'reuse']   [['DMA  ' 'DMA  ' 'DMA  ' 'DMA  ']
 ['DMA  ' 'reuse' 'reuse' 'reuse']    ['DMA  ' 'DMA  ' 'DMA  ' 'DMA  ']
 ['DMA  ' 'reuse' 'reuse' 'reuse']]   ['DMA  ' 'DMA  ' 'DMA  ' 'DMA  ']]

Q DMAs: 3   K DMAs: 12   grid steps: 12
```

Swap the grid axes so the Q index varies fastest, keeping the same `index_map`s, and the
residency flips:

```
grid=(3,4)=(i_q, j_kv)  -> (Q DMAs, K DMAs) = (3, 12)
grid=(4,3)=(j_kv, i_q)  -> (Q DMAs, K DMAs) = (12, 4)
```

Same twelve grid steps, same block shapes, same arithmetic — **4× the Q traffic**, decided
entirely by which axis you put last. That is the answer to "why this `index_map` and not
that one".

**Independently re-verified** with the per-step-sentinel probe. The raw reads are more
instructive than the counts, so they are recorded here verbatim (`0` = clean first copy
of a block, `1000+n` = the sentinel step *n* wrote):

```
grid=(3,4)  i_q outer, j_kv fast
  Q raw: [   0 1000 1001 1002    0 1004 1005 1006    0 1008 1009 1010]   ->  3 DMAs /  9 reuses
  K raw: [   0    0    0    0 1000 1001 1002 1003 1004 1005 1006 1007]   -> 12 DMAs /  0 reuses

grid=(4,3)  j_kv outer, i_q fast
  Q raw: [   0    0    0 1000 1001 1002 1003 1004 1005 1006 1007 1008]   -> 12 DMAs /  0 reuses
  K raw: [   0 1000 1001    0 1003 1004    0 1006 1007    0 1009 1010]   ->  4 DMAs /  8 reuses
```

Read the Q row of the first block: `0` at steps 0, 4, 8 — a genuinely new Q block being
fetched once per outer step — and an unbroken sentinel chain in between, the same VMEM
buffer surviving three more steps. The K row of the first block is the opposite: four
clean copies while the blocks are new, then from step 4 onward every read returns the
sentinel from that block's *previous* visit, which is positive proof the copy was
re-issued rather than elided.

### 3.4 The rule as JAX implements it

Two independent implementations, both readable, both agreeing:

**The interpreter.** In
[`jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py`](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py)
the input store into the kernel buffer is guarded by

```python
          token = jax.lax.cond(
              (iteration_idx == initial_iteration_idx)
              | jax.lax.reduce_or(
                  cur_start_indices[j] != prev_start_indices[j], axes=(0,)
              ),
              ...
```

and the output store back to HBM by

```python
          token = jax.lax.cond(
              (iteration_idx + 1 == loop_bound)
              | jax.lax.reduce_or(
                  cur_start_indices[num_inputs + j]
                  != next_start_indices[num_inputs + j], axes=(0,),
              ),
              ...
```

Read that as: *copy in on the first step or when the start index changed since the
previous step; copy out on the last step or when the start index will change on the next
step.* Note the asymmetry — inputs look **backwards**, outputs look **forwards**. That is
exactly the accumulator semantics of §3.2.

**The in-kernel pipeline emitter.** `pltpu.emit_pipeline` expresses the same rule in
[`jax/_src/pallas/mosaic/pipeline.py`](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/pipeline.py):

```python
  def has_changed(self, buffered_ref):
    ...
    indices = buffered_ref.compute_index(*self.indices)
    prev_indices = buffered_ref.compute_index(*self.prev_indices)
    return _tuples_differ(indices, prev_indices)
```

used as `wait_in` predicate `has_changed(...) | first_step`, `wait_out` predicate
`has_changed(...) & ~first_step`, and `copy_out` predicate
`will_change_current(...) | last_step`.

**Caveat, and it matters.** Neither of these is the code that runs on a v5e. For
`pallas_call`, the pipeline is emitted by the Mosaic/XLA compiler behind
`tpu_custom_call`, which is not Python and not in this tree. The Python evidence shows
that two JAX-authored implementations of the documented rule both do the same thing; the
documented rule is what binds the hardware path. jax-ml/jax#36287 is the standing reminder
that interpret mode and hardware can diverge.

### 3.5 The corollary the docs are blunt about

> In general, a good rule-of-thumb to follow is that **the input buffers passed into the
> kernel function should be interpreted as read-only, and output buffers are write only**.
>
> Writing to inputs and reading from outputs will in most cases result in incorrectness.

— [Software Pipelining §Buffer Revisiting](https://docs.jax.dev/en/latest/pallas/pipelining.html#buffer-revisiting)

The §3.3 probe is a deliberate violation used as an instrument. Do not ship it. The two
sanctioned exceptions are named on the same page: accumulation, and `input_output_aliases`.

And when several invocations write the same output elements *non*-consecutively:

> When multiple invocations write to the same elements of the output array the result is
> platform dependent.

— [Grids and BlockSpecs](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html#blockspec-a-k-a-how-to-chunk-up-inputs)

### 3.6 Verified: the accumulator, right way and wrong way

The docs' `correct_sum` / `incorrect_sum` pair reduces an `(8, 256, 256)` array along axis
0. Re-run here on CPU with `interpret=True`:

```
reduction on LEADING  grid axis -> out[0,0] = nan   matches jnp.sum: False
reduction on TRAILING grid axis -> out[0,0] = 8.0   matches jnp.sum: True
```

The `nan` is informative: the output buffer starts uninitialised, interpret mode fills
uninitialised memory with NaN, and `o_ref[...] += x_ref[...]` never zeroes it. Both the
errors the docs call out (wrong axis, missing init) show up in one number. The correct
version needs both fixes:

```python
def correct_sum_kernel(x_ref, o_ref):
  @pl.when(pl.program_id(2) == 0)
  def _():
    o_ref[...] = jnp.zeros_like(o_ref)
  o_ref[...] += x_ref[...]
```

— [Software Pipelining §Reductions and accumulation](https://docs.jax.dev/en/latest/pallas/pipelining.html#reductions-and-accumulation)

---

## 4. VMEM budgeting

### 4.1 How much there is

The published JAX **TPU Hardware Reference** page is generated at doc-build time by
calling `pltpu.get_tpu_info_for_chip` — the notebook source
([`docs/pallas/tpu/hardware.ipynb`](https://github.com/jax-ml/jax/blob/main/docs/pallas/tpu/hardware.ipynb))
literally loops over `pltpu.ChipVersion` and formats
`info.vmem_capacity_bytes // (1024*1024)`. So the doc page and
[`jax/_src/pallas/mosaic/tpu_info.py`](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/tpu_info.py)
are the same source, and it can be queried locally. Executed on this machine:

```
v4    gen=4 cores/chip=2  VMEM=16 MiB   SMEM=1024 KiB  HBM=17.2 GB  BW=615 GB/s   bf16=138 TFLOP/s  mxu=128  num_mxus=4
v5e   gen=5 cores/chip=1  VMEM=128 MiB  SMEM=1024 KiB  HBM=17.2 GB  BW=820 GB/s   bf16=197 TFLOP/s  mxu=128  num_mxus=4
v6e   gen=6 cores/chip=1  VMEM=128 MiB  SMEM=1024 KiB  HBM=34.4 GB  BW=1640 GB/s  bf16=920 TFLOP/s  mxu=256  num_mxus=2
```

All values are **per TensorCore**; the notebook states this explicitly ("all specs listed
in the table below are **per TensorCore**"). v5e has one TensorCore per chip, so
per-TensorCore == per-chip for us.

The Cloud TPU v5e page confirms the chip-level figures — one TensorCore per chip, 16 GB
HBM, 800 GiBps HBM bandwidth, 197 TFLOPs bf16, 393 TOPs int8, 400 GBps interchip, 256
chips per pod — and **does not state VMEM or on-chip SRAM size anywhere**
([cloud.google.com/tpu/docs/v5e](https://docs.cloud.google.com/tpu/docs/v5e)). For VMEM,
the JAX table is the only primary source I found. The `details.rst` prose says something
different — see [Disagreements](#disagreements-found).

`shapes.py` currently records `peak_hbm_bandwidth = 819e9` and `hbm_bytes = 16 GiB`.
Cloud says "800 GiBps" (= 859e9 B/s); JAX says 820e9 B/s (= 764 GiBps). These are the same
number quoted in three unit conventions and rounded differently. Not worth changing, but
worth knowing which one a roofline is standing on.

### 4.2 How the budget multiplies

Per grid step, VMEM must simultaneously hold, for every pipelined operand:

```
bytes(operand) = prod(block_shape) × itemsize × buffer_count
```

with `buffer_count = 2` by default for every input and output
([TPU Pipelining §Multiple Buffering](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html#multiple-buffering)),
dropping to 1 for whole-array trivial-window operands (§2.3). Scratch buffers declared via
`scratch_shapes` are *not* multiply-buffered — they "are persistent across kernel
iterations" ([same page](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html#tpu-memory-spaces)) —
so they count once. On top of that sit spilled vector registers, which `details.rst` names
explicitly as part of the budget.

### 4.3 Worked, for this repo's shapes — *inferred*

Windowed prefill, Gemma 3 1B (`src/gemma3_pallas/shapes.py`): `head_dim = 256`,
`num_kv_heads = 1` (MQA), `sliding_window = 512`. Take `bq = bkv = 512`, inputs bf16,
accumulators f32.

| Operand | Block shape | dtype | Bytes | ×buffers | VMEM |
|---|---|---|---|---|---|
| Q | (512, 256) | bf16 | 256 KiB | 2 | 512 KiB |
| K | (512, 256) | bf16 | 256 KiB | 2 | 512 KiB |
| V | (512, 256) | bf16 | 256 KiB | 2 | 512 KiB |
| O | (512, 256) | f32  | 512 KiB | 2 | 1024 KiB |
| `m`, `l` scratch | (512, 128) | f32 | 256 KiB each | 1 | 512 KiB |
| **Total** | | | | | **≈ 3 MiB** |

Plus the `(512, 512)` f32 logits tile — 1 MiB — which lives in vector registers and
spills to VMEM if it does not fit. Against a 128 MiB v5e VMEM this is not close to
binding; against the 16 MiB that `details.rst` quotes it is still comfortable. **The
arithmetic is mine; only the rules it applies are cited.** No number here has been
measured on hardware.

### 4.4 The failure mode

> VMEM is fairly large for such a low-level memory hierarchy (16MB+), making it possible
> to use large window sizes. And, oftentimes, the larger the window size, the better the
> eventual hardware utilization will be. However, it is possible to specify a window size
> that (together with space necessary to hold spilled vector registers) exceeds the size
> of VMEM. In this case, you will likely see a low-level compiler error message
> complaining about an out-of-memory error.

— [Writing TPU kernels with Pallas](https://docs.jax.dev/en/latest/pallas/tpu/details.html#noteworthy-properties-and-restrictions)

It is a **compile-time** failure from the Mosaic compiler, not a runtime OOM, and the
message is low-level. There is a knob:

> `vmem_limit_bytes`: Overrides the default VMEM limit for a kernel. Note that this must
> be used in conjunction with the `--xla_tpu_scoped_vmem_limit_kib=N` flag with
> `N*1kib > vmem_limit_bytes`.

— `pltpu.CompilerParams` docstring,
[`jax/_src/pallas/mosaic/core.py`](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/core.py)

The existence of `xla_tpu_scoped_vmem_limit_kib` means the *effective* budget a kernel is
allowed is an XLA scoped limit, not the physical 128 MiB. I could not find the default
value of that flag in a primary source — see [Still open](#still-open).

None of this is reachable from CPU: interpret mode allocates ordinary host arrays and
will happily "fit" a block that no TPU could.

---

## 5. Block shape constraints on TPU

### 5.1 The rule

The two docs state it with slightly different coverage; `grid_blockspec.md` is the more
complete of the two:

> * On TPU, only blocks with rank at least 1 are supported. Furthermore, the last two
>   dimensions of your block shape must be equal to the respective dimension of the
>   overall array, or be divisible by 8 and 128 respectively. For blocks of rank 1, the
>   block dimension must be equal to the array dimension, or be a multiple of 1024, or be
>   a power of 2 and at least `128 * (32 / bitwidth(dtype))`.

— [Grids and BlockSpecs](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html#blockspec-a-k-a-how-to-chunk-up-inputs)

`details.rst` gives the same rule minus the rank-1 clause. Note it is a **disjunction**:
`8 | dim[-2]` and `128 | dim[-1]`, *or* the block simply equals the array along that axis.
A `(7, 5)` array with a `(7, 5)` block is legal; a `(7, 5)` array with a `(2, 3)` block is
not.

### 5.2 Why 8 and 128

> TPUs perform the bulk of the computation on 2D vector registers, which are typically of
> size 8x128 for 32-bit values (as of TPU v6). When a vector value is loaded from VMEM
> into registers (e.g. `x = x_ref[...]`), the last two dimensions of the array will be
> tiled into the registers. Pallas will only ever consider mapping the last two
> dimensions of intermediate arrays to the 8x128 vector register dimensions (sublanes and
> lanes respectively).

— [Writing TPU kernels with Pallas §Array Layouts](https://docs.jax.dev/en/latest/pallas/tpu/details.html#array-layouts)

Confirmed in source: `tpu_info.py` sets `NUM_LANES = 128`, `NUM_SUBLANES = 8` as "Common
parameters for all TensorCores", and `Tiling.COMPACT.shape == (8, 128)`. Executed locally:
v5e reports `lanes=128, sublanes=8`.

The cost of getting it wrong is stated as arithmetic, not vibes:

> all vector computation is padded up to the tile size. Adding a two 1x1 arrays costs as
> much as adding two 8x128 arrays, and adding two 8x128x1x1 arrays will be 1024 times as
> expensive as adding two 8x128 arrays, since the 8x128x1x1 array will be padded to
> 8x128x8x128.

— same section. This is why singleton trailing dims are a design smell, and why
`splash_attention` broadcasts `segment_ids` out to `NUM_LANES` / `NUM_SUBLANES` rather
than carrying a rank-1 array (§7).

### 5.3 Non-dividing blocks: padded in, discarded out

> If the block shape does not divide evenly the overall shape then the last iteration on
> each axis will still receive references to blocks of `block_shape` but the elements
> that are out-of-bounds are padded on input and discarded on output. The values of the
> padding are unspecified, and you should assume they are garbage. In the `interpret=True`
> mode, we pad with NaN for floating-point values, to give users a chance to spot
> accessing out-of-bounds elements, but this behavior should not be depended upon. Note
> that at least one of the elements in each block must be within bounds.

— [Grids and BlockSpecs](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html#blockspec-a-k-a-how-to-chunk-up-inputs)

Verified by execution. Reading `x_ref[-1, -1]` from every block of a `(7, 5)` array of
ones with `block_shape=(2, 3)`, `grid=(4, 2)`:

```
[[ 1. nan]
 [ 1. nan]
 [ 1. nan]
 [nan nan]]
```

`1.0` where the block's last element is in bounds, `nan` where it is padding. **The grid
still runs the full 4×2 steps** — the padded blocks are not skipped, they are computed and
their out-of-range outputs thrown away. If your kernel needs a mask, this is why: Pallas
pads the *shape*, it does not neutralise the *values*.

### 5.4 Interpret mode will not catch a bad block shape

```
block_shape=(3,7) on a (6,7) array ACCEPTED under interpret=True; shape (6, 7)
```

`(3, 7)` violates §5.1 on both trailing dims (3 ∤ 8 and ≠ 6; 7 ∤ 128 but *does* equal the
array dim, so only the second-minor is actually illegal). Interpret mode runs it happily.
**Block-shape legality is a hardware-only check.** For a workflow that develops on CPU and
validates on scarce Colab TPU time, this is a landmine: the shape rule has to be applied
by hand, at authoring time.

---

## 6. `BlockSpec` variants and when each is right

`pl.BlockSpec` has four fields
([`jax/_src/pallas/core.py`](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/core.py)):

```python
class BlockSpec:
  block_shape: Sequence[BlockDim | int | None] | None = None
  index_map: Callable[..., Any] | None = None
  memory_space: Any | None = ...
  pipeline_mode: Buffered | None = None
```

### 6.1 Block dimension kinds

`BlockDim = Element | Squeezed | Blocked | BoundedSlice | Indirect` (`core.py`). An `int`
canonicalises to `Blocked(n)`; `None` canonicalises to `Squeezed()`.

| Kind | `index_map` returns | Use when |
|---|---|---|
| `Blocked(n)` / plain `int` | a **block** index (scaled by `n`) | the default; regular tiling |
| `Squeezed()` / `None` | a block index into a size-1 slice | you want the axis gone from the ref's shape |
| `Element(n, padding)` | an **element** index, unscaled | non-block-aligned starts. *"currently supported only on TPUs"* |
| `BoundedSlice(max)` | a `pl.ds(start, size)`, both may be dynamic | dynamic block sizes — `emit_pipeline` only |
| `Indirect(n)` | *"A dimension indexed by an array of indices. Implies a gather for inputs and a scatter for outputs."* | gather/scatter; rejected by `pallas_call` lowering |

Sources: [Grids and BlockSpecs §The "element" indexing mode](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html#the-element-indexing-mode)
for `Blocked`/`Element`; `core.py` docstrings for `Squeezed`, `BoundedSlice`, `Indirect`;
[TPU Pipelining §Dynamic Block Shapes](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html#dynamic-block-shapes)
for `BoundedSlice`. `lowering.py` raises
`NotImplementedError("Unsupported block dimension type: ...")` for anything that is not
`Element`, `Squeezed` or `Blocked`, which is how you know `BoundedSlice`/`Indirect` are
`emit_pipeline`-only.

Squeezing, verified: `BlockSpec((None, 2), lambda i, j: (i, j))` gives
`o_ref.shape == (2,)`. Matching doc example
[here](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html#blockspec-a-k-a-how-to-chunk-up-inputs).

### 6.2 Memory spaces

| Pallas enum | TPU memory space | Type |
|---|---|---|
| `pl.ANY` | HBM (usually) or VMEM | DRAM |
| `pltpu.VMEM` | VMEM | SRAM |
| `pltpu.SMEM` | SMEM | SRAM |
| `pltpu.SEMAPHORE` | Semaphore | SRAM |

— [TPU Pipelining §TPU Memory Spaces](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html#tpu-memory-spaces), verbatim.

The decisive sentences from that section:

- `VMEM` "is the default memory space if nothing is specified."
- `SMEM`: "Only scalar loads and stores can be performed to/from SMEM."
- `ANY`: "a hint to the compiler that the memory space is unconstrained. In most cases,
  XLA will place this buffer in HBM. A buffer assigned to the `ANY` memory space cannot be
  dereferenced normally using array indexing syntax (e.g. `x[...]`). Instead, we must
  first copy the values into a VMEM or SMEM buffer using `pltpu.sync_copy` or
  `pltpu.async_copy`."
- **"pipelining is not allowed unless the `memory_space` is marked as `VMEM`."**

That last one is the decision rule: `memory_space=pl.ANY` means *"I am opting out of the
automatic pipeline for this operand and will move it myself."* You take the ref in HBM and
do your own `sync_copy`/`async_copy`. The doc's worked example:

```python
def hbm_vmem_kernel(x_hbm_ref, out_vmem_ref, scratch_vmem_ref):
  pltpu.sync_copy(x_hbm_ref.at[0:1], scratch_vmem_ref)
  out_vmem_ref[...] = scratch_vmem_ref[...] + 1

out = pl.pallas_call(hbm_vmem_kernel,
  in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
  out_shape=jax.ShapeDtypeStruct((1, 128), jnp.float32),
  scratch_shapes=(pltpu.VMEM(shape=(1, 128), dtype=jnp.float32),)
)(x)
```

For SMEM, `details.rst` gives the rule of thumb: *"any data used to perform control-flow
decisions should be placed in SMEM"*, and notes SMEM "lets you only read and write 32-bit
values with a single instruction (very small compared to the 4KBi granularity of VMEM
transactions, but much more flexible due to lack of alignment requirements!)". SMEM is
1 MiB per core on v5e (§4.1).

The full `pltpu.MemorySpace` enum in JAX 0.11.0 is wider than the doc table —
`VMEM, VMEM_SHARED, SMEM, CMEM, SEMAPHORE, HBM`
([`jax/_src/pallas/mosaic/core.py`](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/core.py)) —
and `pl.MemorySpace` adds `ANY, DEFAULT, ERROR, INDEX, KEY`. Only the four in the table
are documented.

### 6.3 Scalar prefetch — gathered for later, not for now

This is how data-dependent `index_map`s, and therefore block *skipping*, get expressed.
Phase 5 material; recorded here so it need not be re-found.

> Scalar prefetch allows you to pass in a small amount of data into SMEM ("scalar memory")
> that is loaded before the start of the pipeline ("prefetch"). Because this data is
> loaded before the pipeline, it is available for use in the `index_map` for each
> BlockSpec, allowing you to perform data-dependent indexing calculations.

— [Scalar Prefetch and Block-Sparse Computation](https://docs.jax.dev/en/latest/pallas/tpu/sparse.html)

Mechanics, from the same page and from `details.rst`:

- Replace `grid=` with `grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=n, ...)`.
- "If `num_scalar_prefetch` is `n`, then the first `n` arguments to `pallas_call` will be
  placed in SMEM. No `BlockSpec`s should be specified for those arguments."
- Argument order is fixed: `index_map(*grid_indices, *prefetch_refs)` and
  `kernel(*prefetch_refs, *input_refs, *output_refs, *scratch_refs)`; the caller passes
  `kernel(*prefetch_args, *input_args)`.

The point for a windowed kernel: an `index_map` that reads a prefetched SMEM table can
return *the same block index twice in a row on purpose*, and §3.2 then elides the fetch.
Block skipping is not a separate mechanism — it is copy elision, driven by a table.

---

## 7. Worked example: how `splash_attention` uses all of the above

[`splash_attention_kernel.py`](https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py),
`_splash_attention_forward`. The whole §3 argument is visible in about thirty lines.

```python
grid = (num_q_heads, q_seq_len // bq, grid_width)     # (h, i, j) — j is FASTEST
```

```python
  def q_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None):
    del j, data_next_ref, mask_next_ref, block_mask_ref     # <-- j deleted
    return from_head_minor((h, i, 0), q_layout)

  def out_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None):
    del j, data_next_ref, mask_next_ref, block_mask_ref     # <-- j deleted
    return h, i, 0

  def k_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None):
    next_j, *_ = _next_nonzero(h, i, j, data_next_ref, block_mask_ref, mask_next_ref)
    prefix = () if is_mqa else (_div(h, q_heads_per_kv_head),)
    return from_head_minor((*prefix, next_j, 0), k_layout)   # <-- follows j
```

Read against §3.2:

- **Q ignores `j`.** Its block index is constant across the entire inner KV loop, so the
  Q tile is fetched once per `(h, i)` and stays in VMEM for all `grid_width` steps.
- **The output ignores `j` too.** Consecutive invocations write the same output slice, so
  the output VMEM buffer *is* the flash-attention accumulator, flushed to HBM only when
  `i` advances. Exactly the pattern `details.rst` prescribes for reductions.
- **`m_scratch`, `l_scratch`, `o_scratch` use `lambda h, i, j, *_: (0, 0)`** — constant in
  *every* grid index. Whole-array trivial windows (§2.3), single-buffered, permanently
  resident. Online-softmax running state, held in VMEM for the kernel's whole life.
- **K and V follow `j`, through `_next_nonzero`.** That helper reads the prefetched SMEM
  tables `data_next_ref` / `block_mask_ref` and returns `next_j`, the index of the next
  *non-empty* KV block. When two consecutive `j` steps resolve to the same `next_j`, §3.2
  elides the fetch, and the kernel body's `should_run` flag skips the compute. That is
  block-sparse attention: `num_scalar_prefetch = 3`, three SMEM tables, and an `index_map`
  that lies about `j`.
- **`dimension_semantics=("parallel", "arbitrary", "arbitrary")`** — heads are
  independent; `i` and `j` are not, because the output window is constant along `j`.
  Matches the `details.rst` rule that `dimension_semantics` is "always a number of
  `parallel` axes followed by a number of `arbitrary` axes".
- **Tile-shape hygiene**: `pl.BlockSpec((bq, NUM_LANES), ...)` for `q_segment_ids` and
  `(NUM_SUBLANES, bkv)` for `kv_segment_ids`, with the rank-1 inputs broadcast out via
  `jax.lax.broadcast_in_dim` before the call. §5.2 is why.
- **`memory_space=pltpu.SMEM`** on the `sinks` spec — a small per-head scalar table, read
  for control purposes, exactly the §6.2 rule of thumb.

The one thing the grid does *not* do: `bkv` is decoupled from `bkv_compute`
(`if bkv % bkv_compute: raise`, and `bkv_compute % NUM_LANES` must be 0). The pipeline
granularity and the compute granularity are separate knobs. Worth remembering when block
sizes get tuned.

---

## Disagreements found

Flagged rather than silently resolved, per `NOTES.md`.

**1. VMEM size on v5e: 16 MB vs 128 MiB. Unresolved.**

- `details.rst` (reviewed date not stated on the page): *"VMEM is fairly large for such a
  low-level memory hierarchy (16MB+)"* — generation-agnostic, and 16 MiB is exactly the
  v4 figure.
- `pallas/pipelining.md`: *"TPU v5p contains 96MB of VMEM"*.
- `tpu_info.py` / the generated TPU Hardware Reference table: v4 = 16 MiB, **v5e = 128
  MiB**, v5p = **64 MiB**, v6e = 128 MiB, all per TensorCore.

The v5p figure alone is quoted as 96 MB in one JAX doc and 64 MiB in another JAX source,
which is a straight contradiction inside the same repository. My reading: `details.rst`'s
"16MB+" is deliberately vague and stale, the "96MB" is an error or a
different-configuration number, and `tpu_info.py` is the number to use because it is
machine-readable, versioned, and is what the docs render. But this is a judgement, not a
resolution. Cloud's v5e page does not adjudicate — it never mentions VMEM.

**2. "sequentially, in lexicographic order" vs "a combination of parallel and sequential".**

- `details.rst`: *"the grid is generally not processed in parallel, but sequentially, in
  lexicographic order (though see the Multicore TPU configurations section for
  exceptions)"*.
- `grid_blockspec.md`: *"On TPUs, programs are executed in a combination of parallel and
  sequential."*

These are reconcilable — `grid_blockspec.md` is the platform-neutral page and is gesturing
at Megacore, which `details.rst` names as the exception. But the flat phrasing in
`grid_blockspec.md` would mislead someone reading only that page. For a v5e (one
TensorCore, `supports_megacore == False`) the `details.rst` statement is the operative one.

**3. Buffer count: "only double-buffering" vs `pl.Buffered(buffer_count=N)`. Resolved.**

- `pallas/pipelining.md`: *"Pallas on TPU only supports double-buffering, as TPU programs
  can operate on larger block sizes and double-buffering is typically enough to cover the
  latency."*
- `tpu/pipelining.md`: presents `pl.BlockSpec(pipeline_mode=pl.Buffered(buffer_count=N))`
  with no stated ceiling, plus `use_lookahead`.

Resolved in source: `lowering.py` rejects anything outside `{1, 2}` for `pallas_call`
(*"Only single (1) and double (2) buffering are supported"*) and rejects `use_lookahead`
outright (*"Lookahead is not supported for XLA pipeline emitter lowering"*).
`pipeline.py`'s `Scheduler` handles arbitrary `buffer_count`, but that is
`pltpu.emit_pipeline`. Both docs are right about their own scope; neither says which scope
it is talking about.

**4. HBM bandwidth on v5e: 800 GiBps (Cloud) vs 820 GB/s (JAX) vs 819e9 (`shapes.py`).**
Minor, and mostly a unit-convention artefact, but a roofline that quotes a utilisation
percentage should say which it used.

---

## Still open

Feeds back into `RESOURCES.md` `## Gaps`.

- **The default value of `--xla_tpu_scoped_vmem_limit_kib`.** This, not the physical 128
  MiB, is the number a kernel actually gets. `pltpu.CompilerParams`'s docstring proves the
  flag governs the budget but never states its default. Not present anywhere in the JAX
  Python tree. Needs an OpenXLA source read or an on-hardware experiment.
- **Whether the copy elision is contractual on the hardware path.** `details.rst` states
  it as a property, and two JAX-authored pipeline implementations honour it, but the
  `pallas_call` pipeline on TPU is emitted by the Mosaic compiler, which is out of reach
  from this tree. Nothing found that says "guaranteed". Everything found says "is". A
  Colab session with `xprof` DMA counts would settle it — that is the single highest-value
  thing to do with TPU time on this topic.
- **DMA granularity.** `details.rst` mentions "the 4KBi granularity of VMEM transactions"
  in passing, in the SMEM section, and `SparseCoreInfo` carries a
  `dma_granule_size_bytes` field (32 B on v5p/v6e) — but there is no TensorCore-side
  figure and no statement of what a sub-granule copy costs. This is the same gap
  `RESOURCES.md` already records for v5e microarchitecture.
- **What the VMEM-overflow error actually looks like.** "a low-level compiler error
  message complaining about an out-of-memory error" is all the docs offer. Not
  reproducible on CPU. Worth capturing verbatim the first time it happens on Colab, so it
  is recognisable.
- **Whether `interpret=True` and `pltpu.InterpretParams()` ever diverge on copy elision.**
  They agreed on every probe here, but they are separate implementations
  (`hlo_interpreter.py` vs `interpret/interpret_pallas_call.py`) and only one of them was
  read.
- **`RevisitMode.ANY`.** `core.py` documents it as inserting "additional DMAs as needed to
  restore the buffer state", enabling non-consecutive output revisiting — and
  `lowering.py` rejects it when `buffer_count > 1`. No doc-page coverage at all. Possibly
  the escape hatch for a shrunk grid whose output blocks are not visited consecutively;
  possibly a dead end.
- **`freshness` metadata.** `tpu/pipelining.md` carries `reviewed: '2024-04-08'` and
  `grid_blockspec.md` carries `reviewed: '2024-06-01'`; `details.rst` carries none. Two
  of the three primary pages here have been unreviewed for over two years. Weigh source
  code over prose when they disagree.

---

## Sources

Primary only. Nothing below is a blog post, a Medium article, or a tutorial of unknown
provenance.

**JAX documentation**

- [Grids and BlockSpecs](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html) —
  authoritative for: `index_map` returning block indices, the `slices_for_invocation`
  definition, padding of non-dividing blocks, `None`/`Squeezed`, `pl.Element` mode, and
  the platform-dependence of non-consecutive output writes. Marked `reviewed: 2024-06-01`.
- [Writing TPU kernels with Pallas](https://docs.jax.dev/en/latest/pallas/tpu/details.html) —
  authoritative for: lexicographic sequential grid execution, **the copy-skip statement**,
  the consecutive-write requirement on outputs, the reduction-on-last-axis rule, the
  `(8, 128)` block rule, `8×128` vector-register tiling and its padding cost, the VMEM
  OOM failure mode, SMEM guidance, and `PrefetchScalarGridSpec`. Source is `details.rst`,
  no freshness metadata. **The single most load-bearing page for this topic.**
- [Software Pipelining](https://docs.jax.dev/en/latest/pallas/pipelining.html) —
  authoritative for: the derivation of the double-buffered loop and its pseudocode, the
  named copy-elision optimisation, buffer-revisiting hazards, the accumulator pattern with
  a worked correct/incorrect pair, and the compute/memory-bound cost model.
- [TPU Pipelining](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html) —
  authoritative for: the memory-space table, "pipelining is not allowed unless the
  `memory_space` is marked as `VMEM`", scratch-buffer persistence, default buffer count 2,
  `pl.Buffered`, lookahead prefetch, dynamic block shapes, and `dimension_semantics` /
  Megacore availability. Marked `reviewed: 2024-04-08`.
- [Scalar Prefetch and Block-Sparse Computation](https://docs.jax.dev/en/latest/pallas/tpu/sparse.html) —
  authoritative for: `PrefetchScalarGridSpec` semantics and the fixed argument order for
  prefetched refs. Phase 5 reading.
- [TPU Hardware Reference](https://docs.jax.dev/en/latest/pallas/tpu/hardware.html) —
  authoritative for: per-TensorCore VMEM/SMEM/HBM/peak-FLOPs by generation. Note it is
  *generated* from `tpu_info.py`, so the two cannot disagree.

**JAX source (jax 0.11.0 locally; links to `main`)**

- [`jax/_src/pallas/mosaic/tpu_info.py`](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/tpu_info.py) —
  the v5e numbers: 1 TensorCore/chip, 128 MiB VMEM, 1 MiB SMEM, 128 lanes × 8 sublanes,
  128-wide MXU, `supports_megacore == False`. Queryable from CPU via
  `pltpu.get_tpu_info_for_chip`.
- [`jax/_src/pallas/core.py`](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/core.py) —
  `BlockSpec` fields, the `BlockDim` union and each variant's docstring, `Buffered`,
  `RevisitMode`, `BlockMapping.has_trivial_window`, and the default `index_map`.
- [`jax/_src/pallas/mosaic/core.py`](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/core.py) —
  `CompilerParams` (`dimension_semantics`, `vmem_limit_bytes` and its coupling to
  `--xla_tpu_scoped_vmem_limit_kib`), the full `pltpu.MemorySpace` enum,
  `PrefetchScalarGridSpec`.
- [`jax/_src/pallas/mosaic/lowering.py`](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/lowering.py) —
  the buffer-count ceiling for `pallas_call`, forced single-buffering for trivial windows,
  and which `BlockDim` kinds `pallas_call` actually accepts. Settles the buffer-count
  doc conflict.
- [`jax/_src/pallas/mosaic/pipeline.py`](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/pipeline.py) —
  `Scheduler.has_changed` / `will_change_current` / `will_change_fetch` and the
  copy/wait predicates. The clearest statement of the elision rule as code.
- [`jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py`](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py) —
  the interpreter's own `lax.cond` guards on input and output copies. Proof that the
  simulator models the elision.
- [`jax/_src/pallas/mosaic/interpret/params.py`](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/interpret/params.py) —
  `InterpretParams`, including `grid_point_recorder` and `random_seed`'s randomisation of
  `parallel` axes. The instrument used for §1.3.
- [`splash_attention_kernel.py`](https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py) —
  the reference for how a production kernel arranges grid, `index_map`s, scratch specs and
  scalar prefetch. `_splash_attention_forward` and `_next_nonzero` are the two functions
  that matter for §7.

**Google Cloud**

- [Cloud TPU v5e](https://docs.cloud.google.com/tpu/docs/v5e) — authoritative for the
  chip-level spec: one TensorCore per chip, 16 GB HBM, 800 GiBps, 197 TFLOPs bf16,
  393 TOPs int8, 400 GBps interchip, 256 chips per pod. **Does not state VMEM.**

**Prior art already in `RESOURCES.md`**

- [jax-ml/jax#36287](https://github.com/jax-ml/jax/issues/36287) — the standing reason
  every execution-verified claim above is labelled interpret-mode, not hardware.

---

## Appendix: reproducing the experiments

Scripts live in the session scratchpad, not in the repo. Each is ~40 lines and rebuildable
from the snippets above. Run with
`C:\Users\milov\.conda\envs\gemma3-tpu-pallas\python.exe` (JAX 0.11.0, CPU, one device).

| § | What it shows | Mode |
|---|---|---|
| 1.3 | lexicographic order, last axis fastest | `pltpu.InterpretParams(grid_point_recorder=...)` |
| 1.3 | `"parallel"` reorders traversal | `InterpretParams(random_seed=...)` |
| 3.1 | `index_map` returns block indices | `interpret=True` |
| 3.3 | input copy elided iff block index unchanged | both modes, agreeing |
| 3.3 | residency flips with grid axis order (3/12 vs 12/4 DMAs) | `InterpretParams()` |
| 3.6 | accumulator correct only on the last grid axis | `interpret=True` |
| 5.3 | padding on non-dividing blocks, NaN under interpret | `interpret=True` |
| 5.4 | illegal `(3, 7)` block accepted on CPU | `interpret=True` |
| 6.1 | `None` squeezes the axis | `interpret=True` |

Standing caveat for all of them: **interpret mode may not model real DMA behaviour**
(jax-ml/jax#36287). What these establish is that the documented rule is real and that
JAX's own simulator implements it — not what a v5e's DMA engine does.
