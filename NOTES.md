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

## Gaps to close, roughly in order

1. ~~Tile-level mask arithmetic~~ — lesson 0001, assessed in session 2 (record 0002).
   Residual: `full` is not held as a relation distinct from `live`. Re-probe by spacing
   inside a later lesson; do not repeat lesson 0001.
2. ~~Online softmax: the recurrence, and why the running max is required~~ — lesson 0002.
   **Not yet assessed.** Check before building Phase 3 on it; the split that matters is
   *"running" comes from the algorithm, "max" comes from the hardware*.
3. Pallas grid + `BlockSpec`: what actually gets DMA'd, and when (Phase 2 warm-up).
4. `index_map` and shrunk grids — where lesson 0001's predicate becomes code (Phase 5).
5. Roofline: which roof binds, and the arithmetic intensity that decides it (Phase 6).

## Small things to fix in passing

- ~~Denis flagged not remembering **PV**~~ — retained, checked by free recall in session 2.
  Closed.
- Watch sign discipline on `d = q − k`. He inverted the direction once (session 2) while
  having the structure right. Cheap to catch in conversation, expensive to catch in a
  kernel.
