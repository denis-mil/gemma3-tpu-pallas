# Working Notes

## Workspace conventions

- **The glossary is [`CONTEXT.md`](CONTEXT.md), not `GLOSSARY.md`.** It predates this
  teaching workspace and is the project's canonical vocabulary, with an `_Avoid_` line
  per term. Lessons must adhere to it and must not duplicate it. New terms the user has
  demonstrably mastered get added *there*.
- Lessons link the shared stylesheet at `assets/lesson.css`. Use classic `<script src>`
  tags, never `type="module"` — ES modules and `fetch` are blocked under `file://`,
  and these pages are opened from disk.
- **Vary the quiz answer position.** Lesson 03 shipped with all six answers at index 0.
  `quiz.js` deliberately does not shuffle, on the grounds that a deliberate order carries
  no signal — but a *constant* one does, and it is the kind of thing he would spot and
  then stop reading the distractors. Order the options on some principle (numeric,
  descending strength of claim) and check the resulting positions are mixed.
- **Exercise a finished lesson headlessly with Node before shipping it.** Load the asset
  scripts and the inline `<script>` into one `vm` context with a stub `document`; that
  fires `quiz.js`'s option-hygiene warnings, catches a widget spec that throws, and can
  walk every relative `href`. A shim that silently fails to mount reports a clean pass —
  assert that the mount actually happened.

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
- **Assess with instances the reference sheet cannot answer.** Given an abstract prompt he
  will produce the formula verbatim from `reference/` — fluency, not storage. Concrete
  numbers close that door.
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
   session 6, not yet assessed.** Reference sheet:
   [`block-shapes-and-vmem.html`](reference/block-shapes-and-vmem.html). Assess before
   opening gap 5. Two things to probe: whether he reaches for the *regime* (linear vs
   quadratic) rather than the 4 MiB total, and whether the discriminator question (Q2,
   "what has a clean interpret-mode run established") sticks outside the quiz.
5. `index_map` and shrunk grids — where lesson 0001's predicate becomes code (Phase 5).
6. Roofline: which roof binds, and the arithmetic intensity that decides it (Phase 6).


