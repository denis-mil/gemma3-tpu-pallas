# Phases 3–5: the windowed-prefill attention kernel

## Context

Phase 2 is done and measured: `fused_gated_mlp` is correct on a v5e and scored in
[measurement 0001](../measurements/0001-fused-gated-mlp-on-v5e.md). The README roadmap's
next three unchecked items — Phase 3 (online softmax, no mask), Phase 4 (causal masking),
Phase 5 (window block skipping via a shrunk grid and a custom `index_map`) — are the actual
artifact the mission is built around; six lessons of predicates exist for them and no
`pl.pallas_call` does.

`NOTES.md` gap 7 records the workspace as kernel-led from session 10 on: kernels get written
first, lessons get written when writing one produces friction. So this plan is the kernel, all
three phases, built test-first on CPU under interpret mode against the fp32 oracle that is
already committed in [reference.py](../../src/gemma3_pallas/reference.py).

Intended outcome: `src/gemma3_pallas/attention.py` holding one kernel that lands in three
stages, each stage a commit with its own tests green before the next starts. No timing claim
comes out of this — hardware is a later errand (Phase 6), with predictions registered before
it as measurement 0002.

## Stage 0 — the oracle gains a no-mask path

`reference.attention` is unconditionally causal, so stage 1 has nothing to assert against.

- [reference.py](../../src/gemma3_pallas/reference.py): add `causal: bool = True` to
  `attention_mask` and thread it through `attention`. This mirrors
  `shapes.attention_flops(causal=...)`, which already carries exactly that flag, and keeps one
  attention oracle in the repo rather than a throwaway second one in the test file.
- [test_reference.py](../../tests/test_reference.py): one case — `causal=False` attends to every
  key, so every row of the mask is all-True and the output of a uniform `v` is that `v`.

## Design decisions, fixed before any code

**Layout.** The kernel takes head-major operands: `q [num_heads, T, head_dim]`,
`k [S, head_dim]`, `v [S, head_dim]` with the single MQA head squeezed out — the interface
`splash_attention` exposes, and the one that makes the grid's head axis a real axis.
`reference.attention`'s `[T, H, D]` stays the oracle's layout; tests transpose. Record the
transpose as a layout boundary in the module docstring, next to the existing RoPE/QK-norm
boundary note in `reference.py`.

**Grid `(h, i, j)`, kv axis `j` minormost.** Same argument as
[mlp.py:119-123](../../src/gemma3_pallas/mlp.py#L119-L123): the output/accumulator `index_map` is
invariant in `j`, so consecutive `j` steps keep the block resident and write back once, while
K and V stream. Because MQA's K and V do not vary with `h`, they are re-read once per
`(h, i)` — a per-visit cost, not a per-operand one, exactly the correction `mlp_bytes` had to
take. Say so in the docstring rather than discovering it in a trace.

**Carry state in scratch.** `pltpu.VMEM` scratch shapes on `pallas_call`:
`o_acc [block_q, head_dim]` fp32, plus `m` and `l` shaped `[block_q, 128]` (lane-padded, per
`splash_attention_kernel`'s own scratch — read it and copy the shape rather than guessing what
a 1-lane scratch costs). Initialise at `j == 0` under `@pl.when`, finalise (`o_acc / l` into
`o_ref`) at the last `j` under `@pl.when` — two guards, two distinguishable failure modes, per
the `NOTES.md` accumulator rule.

**Precision and dtype.** fp32 operands and a threaded `precision=` argument, identical in
shape and rationale to `fused_gated_mlp`; `interpret=False` remains the default so a forgotten
argument raises on this machine. bf16 storage is a Phase 6 question, not this one.

**Validation.** `_validate` in the shape of [mlp.py:138-170](../../src/gemma3_pallas/mlp.py#L138-L170):
divisible shapes only, `ValueError` naming the axis, `block_q % 8`, `block_k % 8`,
`head_dim % 128`.

## Stage 1 — online softmax, no mask (Phase 3)

`src/gemma3_pallas/attention.py`: `flash_attention(q, k, v, *, block_q, block_k, interpret,
precision)`.

Per `j` step: `s = dot(q, k.T) * cfg.attn_scale`; `m_new = max(m, rowmax(s))`;
`alpha = exp(m - m_new)`; `l = l * alpha + rowsum(exp(s - m_new))`;
`o_acc = o_acc * alpha + dot(exp(s - m_new), v)`. Divide by `l` at the last step only.

`tests/test_attention_kernel.py`, structured like
[test_mlp_kernel.py](../../tests/test_mlp_kernel.py) and with the same explicit `interpret=` on
every call:

1. one kv block, one q block, one head;
2. **two or more kv blocks — written before the rescale guard exists**, because a one-step
   reduction axis passes a kernel that never rescales at all;
3. two heads, so the head axis is exercised as an axis;
4. Gemma dims (`head_dim=256`, `num_heads=4`) at a small `T`/`S`;
5. ragged `block_q` / `block_k` raise, naming the axis;
6. the same geometry under `pltpu.InterpretParams()`, using the existing
   `_interpret_params()` skip helper.

Tolerance is **measured on the first green run and written into the file with the number that
produced it**, as `GEMMA_RTOL`'s comment does — and with the same caveat that it is a CPU
number that does not transfer.

## Stage 2 — causal masking (Phase 4)

Positions from the grid: `q_pos = i * block_q + arange(block_q)`,
`k_pos = j * block_k + arange(block_k)`. Three block classes, the lesson-0001 predicate in
code:

- **dead** (`k_start > q_end`): skip the compute under `@pl.when`. The grid step still runs —
  the shrink is Phase 5's job, not this one.
- **partial**: `jnp.where(allowed, s, -inf-ish)` before the running max. Use a large finite
  negative, not `-jnp.inf`, so `exp` of a fully-masked row cannot produce `nan` through
  `0/0` — no such row exists in causal prefill, but the windowed stage can create one.
- **full**: no masking work at all.

Tests: assert against `reference.attention(causal=True, window=None)` at geometries where the
diagonal falls inside a block and where it falls on a block boundary.

## Stage 3 — window block skipping via a shrunk grid (Phase 5)

Confirmed present in this environment (jax 0.11.0):
`pltpu.PrefetchScalarGridSpec`, and
`jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_mask_info.process_mask(mask,
block_shape, *, shrink_grid=True)`.

- Build the index table with **`process_mask` itself**, not a reimplementation — the session-7
  rule in `NOTES.md`, whose counter-example is precisely this function. Feed it a
  `MultiHeadMask` of local masks at `window=512`; take `data_next`, `mask_next` and
  `block_mask`.
- Pass those through `PrefetchScalarGridSpec(num_scalar_prefetch=...)`; the K and V
  `BlockSpec.index_map`s read `data_next` instead of returning `j`. The kv grid axis shrinks to
  the live-block count.
- **`_shrink_mask_info` pads with block index 0, not with the previous index** (`NOTES.md`
  records this as the case that killed the hand-built table). So the in-kernel mask from stage
  2 must still run, and `block_mask == 0` steps must be skipped under `@pl.when` — otherwise a
  padded step re-accumulates block 0's keys into the running sum.

Tests: against `reference.attention(window=512)` and at least one `window` smaller than
`block_k`, plus a case where a q block's live count differs from its neighbour's so the padding
path is actually taken.

## Optional last item — the byte model

`shapes.attention_bytes(...)`, counting **every visit**: K and V once per `(h, i)`, Q and O
once each. This is what lets Phase 6 state what the shrink buys, and it is the same
per-visit correction `mlp_bytes(elide_weights=)` already carries. Skip it if stage 3 runs
long; it is not needed for correctness.

## Verification

```
"$HOME/.conda/envs/gemma3-tpu-pallas/python.exe" -m pytest -q
"$HOME/.conda/envs/gemma3-tpu-pallas/python.exe" -m pytest tests/test_attention_kernel.py -q
```

Green means correct under emulation and nothing more — ADR-0003, and
[jax#36287](https://github.com/jax-ml/jax/issues/36287). No number from these runs is
published, and no timing is taken on this machine.

Per stage: run the full suite, tick the README roadmap line, and append to `NOTES.md` gap 7
what the stage settled and what it could not. Hardware (a v5e session with predictions
registered first, as measurement 0002) is Phase 6 and out of scope here.
