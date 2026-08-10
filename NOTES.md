# Working Notes

## Workspace conventions

- **The glossary is [`CONTEXT.md`](CONTEXT.md), not `GLOSSARY.md`.** It predates this
  teaching workspace and is the project's canonical vocabulary, with an `_Avoid_` line
  per term. Lessons must adhere to it and must not duplicate it. New terms the user has
  demonstrably mastered get added *there*.
- Lessons link the shared stylesheet at `assets/lesson.css`. Use classic `<script src>`
  tags, never `type="module"` — ES modules and `fetch` are blocked under `file://`,
  and these pages are opened from disk.
- **The lessons are published over GitHub Pages** (from `main` at the repo root, with
  `.nojekyll`; commit `2d2c3e7`). Two consequences for every new page: links to `.md` and
  `.py` files must be **absolute `github.com/.../blob/main/...` URLs** — a relative one
  downloads raw text under Pages — while links to `.html` and `../assets/` stay relative so
  the pages still open from disk. And **`index.html` at the root is the landing page**; a
  new lesson or reference sheet is not shipped until it is listed there with a standfirst.
- **Vary the quiz answer position.** Lesson 03 shipped with all six answers at index 0.
  `quiz.js` deliberately does not shuffle, on the grounds that a deliberate order carries
  no signal — but a *constant* one does, and it is the kind of thing he would spot and
  then stop reading the distractors. Order the options on some principle (numeric,
  descending strength of claim) and check the resulting positions are mixed.
- **Exercise a finished lesson headlessly with Node before shipping it.** Load the asset
  scripts and the inline `<script>` into one `vm` context with a stub `document`; that
  fires `quiz.js`'s option-hygiene warnings, catches a widget spec that throws, and can
  walk every relative `href`. A shim that silently fails to mount reports a clean pass —
  assert that the mount actually happened. Session 7 hardened it further: assert each
  mount point actually produced content, not merely that it exists.
- **Build a new widget out of the old widgets' exports.** `skip-table.js` takes the
  live/partial/full predicate from `MaskGrid.classify` and the copy-elision rule from
  `PipelineGrid.trace`, so each rule has exactly one implementation in the workspace and
  a lesson cannot quietly contradict an earlier one. Both were already exported; check
  for that before reimplementing anything.
- **When a library builds the artefact you are teaching, run *its* builder, not your
  reimplementation of it.** Session 7's hand-built `data_next` matched
  `process_mask(..., shrink_grid=True)` byte for byte at `W=512, B=512` — and diverged at
  `W=1024`, because `_shrink_mask_info` pads with block index **0** while the obvious
  guess is to repeat the previous index. The agreeing case was the one the lesson led
  with; the diverging case was the one the quiz turned on. A reimplementation that matches
  on your headline geometry is not validated.

- **When there is no library to defer to, write the derivation twice and diff it.** Session
  8's numbers exist only in this workspace, so `assets/roofline.js` (which counts DMAs by
  running `PipelineGrid.trace`) and a scratch `roofline.py` (which imports `shapes.py` and
  counts closed-form) were written independently and compared on FLOPs, bytes, steps and
  computed-steps across 9 geometries × 3 kernels. The rule from session 7 — do not trust a
  reimplementation that agrees on your headline geometry — applies to *your own* instrument
  first. Sweep the controls too: every reachable combination should be finite, positive,
  and fast enough to be interactive.

- **Check what the simulator models before designing a test around it.** Session 10 built
  a test to hand `InterpretParams` the real 22.5 MiB working set, on the open question of
  whether it enforces a VMEM capacity. Reading its twelve fields and grepping
  `jax._src.pallas.mosaic.interpret` for `vmem|capacity|OOM` — under a minute — answered
  it: **it models no capacity at all**, so the test could never have failed for the reason
  it was written. It was kept for the DMA/semaphore path at the real grid, with the
  docstring rewritten to say what it cannot do. Generalise: *a test justified by a
  behaviour you have not confirmed the tool has is a test you cannot read the result of* —
  and passing is the outcome that hides the problem. The workspace's session-7 rule (run
  the library's builder, not your model of it) has a cheaper sibling: **read the library's
  field list before predicting its behaviour.**
- **Two ways to get an accumulator wrong, and only one test tells them apart.** The
  reduction axis is minormost, so the output block is a running partial sum. Omitting the
  init and zeroing on *every* step both look like "initialise the accumulator", and a grid
  with one reduction step (`H // block_h == 1`) passes the second. Session 10 ran all
  three variants before keeping `@pl.when(program_id(1) == 0)`. Generalise: *a
  reduction-axis kernel needs a test with at least two steps on that axis, and it must be
  written before the guard, not after.*

## Teaching notes

- Denis reasons from primary sources and writes ADRs for his own decisions; the repo's
  README explicitly separates "plan" from "claim". Match that register — no lesson
  should assert a number it has not cited.
- Corollary: when a source conflicts with the repo, resolve it out loud rather than
  quietly picking one. The 1024-vs-512 window discrepancy in the Gemma 3 report was the
  first instance and made a good teaching beat.
- Mission is *transferable* kernel-authoring skill, not shipping this one kernel. When
  choosing between "here is the answer for Gemma" and "here is the predicate, now
  evaluate it for Gemma", always choose the second.
- **Concrete instance first, generalisation second.** Session 2 established this the hard
  way: the abstract "why is it only two comparisons" failed twice, a tile with real
  integers landed on the first try. Lesson 0001 did it the other way round. See
  [`0002`](learning-records/0002-tile-predicate-derives-under-concrete-instance.md).
- **Question with instances the reference sheet cannot answer.** Given an abstract prompt he
  will produce the formula verbatim from `reference/` — fluency, not storage. Concrete
  numbers close that door. Now a rule for quiz stems, since the quiz is the only instrument
  left — see the standing instruction below.
- **Prefer break-it questions to recall-it questions.** Session 4: asked to swap the
  running max for a running minimum and report the damage, he produced a sharper statement
  of the invariant than lesson 0002 contains. A reference sheet lists what the algorithm
  *is* and cannot say what a specific wrong choice *does* — so a deliberate-break question
  is sheet-proof by construction. Cheapest instrument found so far.
- **Spacing works on him.** A question skipped twice in session 2 (`full` vs `live`) came
  back clean in session 4 after one intervening lesson, and lesson 02's range residual came
  back clean in session 5 the same way. Two for two. Seed prior-lesson probes into later
  quizzes rather than scheduling reteaches.
- **His failure mode is a dropped discriminator, not weak recall.** Three instances now: the
  session-2 `d`-sign inversion, the `grid=(4,3)` axis-swap trap, and session 5's
  mode-dependent trace reported without naming the interpreter — the last with both traces
  in front of him. Structure is reliably right; the qualifier that makes it true goes
  missing. Probe for it directly; do not wait for it to surface.
- **He polices boundaries when asked and crosses them when not.** Same session, he bounded
  interpret mode against a v5e correctly on request, then asserted an unverified claim about
  v5e memory paths unprompted. Ask for the caveat and it is there; the discipline is not yet
  automatic.
- **No post-lesson assessments. Settled, session 8 — do not offer one, do not open a
  session with one, do not restate the cost.** Skipped in sessions 7 and 8 and then ruled
  out explicitly. Consequences, recorded once and not to be repeated at him: lessons 04,
  05 and 06 are unassessed, gaps 1 and 4 stay formally open, and lesson 05's three spaced
  probes are unspent. **The in-lesson quiz is the only instrument left**, so it now carries
  the whole diagnostic load — keep seeding prior-lesson probes into it (spacing is the one
  technique that has worked on him twice) and keep preferring break-it questions, because
  those are what a reference sheet cannot answer. Learning records go on documenting what a
  lesson *taught*, not what he demonstrated.
- **The project is kernel-led from session 10 on. Decided session 9.** Lessons are no
  longer scheduled off the gap list; they get written when writing a kernel produces real
  friction, and are named after the thing that bit him. Learning records continue as before.
  Two costs he took on knowingly: fewer lessons, and therefore fewer firings of the
  in-lesson quiz, which the no-assessment rule has already left as the only diagnostic
  instrument. Consequence for lesson design — when a lesson *is* earned it should carry
  **more** spaced probes than before, not fewer, because the next one may be further away.
- **`CONTEXT.md` gains *arithmetic intensity* and *ridge point* only after he demonstrates
  them.** The glossary is the record of mastered vocabulary, not of taught vocabulary; it
  already carries *Roofline* and the prefill/decode regime claims, which lesson 06 now
  supplies the arithmetic for.
- **Community: jax-ml/jax Discussions**, chosen session 3. Recorded in `RESOURCES.md`.
  The wisdom leg is still at zero — nothing posted. Two live candidates now: the standing
  copy-elision-on-hardware question, and (session 6, easier and sharper) the **block-shape
  rule's per-axis-versus-paired ambiguity** — one sentence, one counter-example, answerable
  by anyone with a chip. The second is the better *first* post: low stakes, obviously
  well-posed, and it does not require him to defend an experiment.
- **Re-derive an agent-written number before teaching it, and expect the error to be in
  the derivation rather than the total.** Session 6 re-derived research 0001 §4.3: the
  3 MiB total was right, the inventory that produced it was not (no `o_scratch`, output
  mistyped f32, two errors cancelling). Second instance of a research-doc defect surviving
  multiple readings — see [`0004`](learning-records/0004-residency-verified-and-the-unnamed-interpreter.md)
  for the first. A right total is not evidence of a right derivation.

## Gaps to close, roughly in order

1. ~~Tile-level mask arithmetic~~ — lesson 0001, assessed in session 2 (record 0002).
   Residual: `full` is not held as a relation distinct from `live`. Re-probe by spacing
   inside a later lesson; do not repeat lesson 0001.
2. ~~Online softmax: the recurrence, and why the running max is required~~ — lesson 0002,
   assessed session 4 (record [`0003`](learning-records/0003-online-softmax-executes-and-generalises.md)).
   Held. Residual: he does not spontaneously close the range argument into *"therefore no
   overflow, by construction"*. One sentence, not a lesson — spaced into lesson 0003's quiz.
3. ~~Pallas grid + `BlockSpec`: what actually gets DMA'd, and when~~ — lesson 0003,
   assessed session 5 (record [`0004`](learning-records/0004-residency-verified-and-the-unnamed-interpreter.md)).
   Held. Lesson 0005 may lean on residency. Residual is not about Pallas — see the
   discriminator note below.
4. ~~Block shapes and the VMEM budget: why `(8, 128)`, and what double buffering costs~~
   — lesson [`0004`](lessons/0004-two-rules-your-cpu-wont-enforce.html), **delivered
   session 6, unassessed and staying that way.** Reference sheet:
   [`block-shapes-and-vmem.html`](reference/block-shapes-and-vmem.html). Two things still
   worth spacing into a later quiz: whether he reaches for the *regime* (linear vs
   quadratic) rather than the 4 MiB total, and whether the discriminator question (Q2,
   "what has a clean interpret-mode run established") holds up when re-asked.
5. ~~`index_map` and shrunk grids — where lesson 0001's predicate becomes code (Phase 5)~~
   — lesson [`0005`](lessons/0005-the-index-map-that-lies.html), **delivered session 7,
   unassessed and staying that way.** Reference sheet:
   [`scalar-prefetch-and-shrunk-grids.html`](reference/scalar-prefetch-and-shrunk-grids.html).
   Its quiz carries three spaced probes: gap 4's *regime* question (Q5), gap 4's
   *discriminator* question (Q4), and gap 1's residual — `full` as a relation distinct from
   `live` (Q3). Those three are unspent unless he worked the quiz.
   Lesson 05's own load-bearing question is Q1: the table buys the traffic, the shrink buys
   the steps.
6. ~~Roofline: which roof binds, and the arithmetic intensity that decides it (Phase 6)~~
   — lesson [`0006`](lessons/0006-the-roof-that-didnt-move.html), **delivered session 8,
   unassessed and staying that way.** Reference sheet:
   [`roofline-and-arithmetic-intensity.html`](reference/roofline-and-arithmetic-intensity.html).
   The lesson exists to *falsify* the prediction this entry used to carry (see the note
   below). Its load-bearing question is Q1 — why skipping cannot change the regime — and
   Q6 spaces lesson 03's which-axis-goes-last onto the head axis.
7. Write the kernel. Six lessons of predicates have no `pl.pallas_call` behind them — the
   only one in the repo is in [`tests/test_pallas_smoke.py`](tests/test_pallas_smoke.py),
   and `src/` has none. But `src/` is not bare: alongside `shapes.py` it holds
   [`reference.py`](src/gemma3_pallas/reference.py), the fp32 ground truth every kernel is
   asserted against (`gated_mlp`, `attention_mask`, `attention`). This is a blank kernel
   file, not a blank slate — the oracle is already committed, which is what makes TDD the
   right shape for it.

   **Decided session 9: start at README Phase 2 — the fused gated-MLP, test-first against
   `reference.gated_mlp` under interpret mode.** Phase 2 before Phase 3 for two reasons.
   It is the first blank-file kernel, so it should exercise grid, `BlockSpec`, `index_map`
   and the VMEM budget with no online-softmax carry state to debug at the same time. And
   its operands are *weights*, not activations: `1152 × 6912` in bf16 is 15.2 MiB per
   matrix, 45.6 MiB for all three, ~91 MiB double-buffered — close enough to the contested
   VMEM figure in [`block-shapes-and-vmem.html`](reference/block-shapes-and-vmem.html) that
   the block decomposition is a decision he has to make rather than an exercise he can skip.
   The workspace's own rule — run the library's builder, not your model of it — now applies
   to the kernel itself.

   **Done session 10.** [`src/gemma3_pallas/mlp.py`](src/gemma3_pallas/mlp.py) —
   `fused_gated_mlp`, grid `(tokens // block_t, hidden_dim // block_h)` with the hidden
   axis minormost, `@pl.when` accumulator init, fp32, divisible shapes only.
   [`tests/test_mlp_kernel.py`](tests/test_mlp_kernel.py) has 8 items; README Phase 2 is
   ticked. Two things the session settled and one it could not: the accumulator has two
   distinguishable failure modes (rule above), `InterpretParams` models no VMEM capacity
   (rule above, and `RESOURCES.md`), and therefore **the 22.5 MiB block-size prediction
   remains unverified** — it is now a Colab errand, not a local one. No lesson was written:
   the friction was real but it produced rules for this file, not a predicate he lacks.

## Corrections this workspace has made to itself

- **Gap 6's own premise was wrong, and that became lesson 06.** The entry above used to
  read: *block skipping cuts loads 4096 → 64 and compute 4096 → 127, by different factors,
  so it moves the arithmetic intensity.* Those two numbers are K DMAs and live blocks —
  a ratio between one operand's traffic and all of the compute. Counting **all four**
  operands (Q and O do not shrink with the mask), bytes fall 32.50× against FLOPs' 32.25×:
  intensity moves 504.1 → 508.0, **1.008×**, and the kernel stays compute-bound on both
  sides. Block skipping is regime-preserving by construction — a dead block takes its load
  and its compute away together. Generalise: *an intensity argument is only valid if every
  operand is in the byte count.*
- **`AI = W` was also wrong.** The plausible guess — that intensity tracks the sliding
  window — survives a single geometry and dies under a sweep: intensity is flat in `W`
  whenever `W ≤ B`. The closed form is `I = L·B/(R+k)`, which reduces to **≈ B**. A
  hypothesis confirmed at one point is confirmed at one point.
- **The MLP's weights are read once per `t` step, not once — and the first byte count said
  once.** The grid is `(tokens // block_t, hidden_dim // block_h)` with the hidden axis
  innermost, and the `w_*` `index_map`s vary in the *inner* index. So one `t` step walks
  every hidden block, and when that index resets for the next `t` step, block 0 is no
  longer the previously fetched slice — copy elision skips only *consecutive* identical
  slices, so it cannot bridge the reset. At `T=256, block_t=128` that is two reads of
  the weights: **193.5 MB and intensity 63**, not 97.9 MB and 125, which moves the kernel
  from one side of `DEFAULT`'s ridge to the other. Generalise: the session-8 rule *an
  intensity argument is only valid if every operand is in the byte count* extends to
  **every visit**, not merely every operand — a byte count is over the grid, not over the
  operand list. Both models are now in `shapes.mlp_bytes` (`elide_weights=` picks), because
  the difference is a factor of `tokens // block_t` in the DMA count and therefore
  something an xprof trace can decide; see
  [measurement 0001](docs/measurements/0001-fused-gated-mlp-on-v5e.md).
- **An fp32-typed kernel is not an fp32-precision kernel.** `mlp.py` casts every operand to
  fp32 and the docstring called it "fp32 throughout", which is true of the storage and
  false of the arithmetic: on TPU, `precision=DEFAULT` truncates the multiplies to bf16 and
  accumulates in fp32. So the same source is 1, 3 or 6 bf16 passes of work depending on the
  `precision` asked for, since fp32 matmul is emulated rather than executed. `mlp_flops` now
  takes `passes` as a **required** keyword and multiplies its count by it, so a FLOP count in
  this repo is the work the MXU issues; no `peak_fp32_flops` constant was added, because
  there is no published one and there is no second roof to publish. The CPU-derived
  `GEMMA_RTOL = 1e-4` inherits the same defect and is expected to fail on hardware, which is
  why the notebook reports the error instead of asserting it.
- **The required keyword was on the wrong parameter.** The fix above originally put it on
  `roofline_bound(peak_flops=...)` and divided the roof by the pass count — three roofs, 197,
  65.7 and 32.8 TFLOP/s. But the roof was never the un-inheritable fact: v5e publishes
  exactly one peak, and dividing it invented two ceilings the chip does not have, one of them
  (`HIGH`) for a precision Mosaic cannot even lower. The un-inheritable fact is the **pass
  count**, which is a property of the work, so it belongs in the numerator. Inverting it cost
  nothing to verify, because `passes · I > P/β` is the same inequality as `I > (P/passes)/β`:
  not one predicted time and not one verdict in either measurement document moved, and a test
  writes the old convention out inline to keep proving it. The price is that achieved FLOP/s
  and intensity became *hardware* rates — at T=2048 `HIGHEST` the kernel reads 177 TFLOP/s
  against the 197 roof while doing 29.5 TFLOP/s of the matmul Gemma asked for — so the pass
  count has to be printed beside them. Generalise: *when a required keyword is protecting
  against an invented number, check which of the two operands the number actually belongs
  to.* See [ADR-0004](docs/adr/0004-flop-counts-are-hardware-work.md).


