# Working Notes

## Workspace conventions

- **The glossary is [`CONTEXT.md`](CONTEXT.md), not `GLOSSARY.md`.** It predates this
  teaching workspace and is the project's canonical vocabulary, with an `_Avoid_` line
  per term. Lessons must adhere to it and must not duplicate it. New terms the user has
  demonstrably mastered get added *there*.
- Lessons link the shared stylesheet at `assets/lesson.css`. Use classic `<script src>`
  tags, never `type="module"` — ES modules and `fetch` are blocked under `file://`,
  and these pages are opened from disk.

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
  back clean in session 4 after one intervening lesson. Seed prior-lesson probes into later
  quizzes rather than scheduling reteaches.
- **Community: jax-ml/jax Discussions**, chosen session 3. Recorded in `RESOURCES.md`.
  The wisdom leg is still at zero — nothing posted. First real candidate is the
  `index_map` question, if research leaves it open.

## Gaps to close, roughly in order

1. ~~Tile-level mask arithmetic~~ — lesson 0001, assessed in session 2 (record 0002).
   Residual: `full` is not held as a relation distinct from `live`. Re-probe by spacing
   inside a later lesson; do not repeat lesson 0001.
2. ~~Online softmax: the recurrence, and why the running max is required~~ — lesson 0002,
   assessed session 4 (record [`0003`](learning-records/0003-online-softmax-executes-and-generalises.md)).
   Held. Residual: he does not spontaneously close the range argument into *"therefore no
   overflow, by construction"*. One sentence, not a lesson — spaced into lesson 0003's quiz.
3. ~~Pallas grid + `BlockSpec`: what actually gets DMA'd, and when~~ — lesson 0003.
   **Not yet assessed.** Assess before lesson 0005 leans on residency.
4. Block shapes and the VMEM budget: why `(8, 128)`, and what double buffering costs
   (lesson 0004, signposted at the foot of 0003). Sources already in
   `docs/research/0001` §4–5 — do not re-research.
5. `index_map` and shrunk grids — where lesson 0001's predicate becomes code (Phase 5).
6. Roofline: which roof binds, and the arithmetic intensity that decides it (Phase 6).

## Small things to fix in passing

- ~~Denis flagged not remembering **PV**~~ — retained, checked by free recall in session 2.
  Closed.
- Watch sign discipline on `d = q − k`. He inverted the direction once (session 2) while
  having the structure right. Cheap to catch in conversation, expensive to catch in a
  kernel.
