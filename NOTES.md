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

## Gaps to close, roughly in order

1. ~~Tile-level mask arithmetic~~ — lesson 0001.
2. Online softmax: the recurrence, and why the running max is required (Phase 3).
3. Pallas grid + `BlockSpec`: what actually gets DMA'd, and when (Phase 2 warm-up).
4. `index_map` and shrunk grids — where lesson 0001's predicate becomes code (Phase 5).
5. Roofline: which roof binds, and the arithmetic intensity that decides it (Phase 6).

## Small things to fix in passing

- Denis flagged not remembering **PV** — the second attention matmul, `probs @ V`.
  Covered as an aside in lesson 0001 and in the reference sheet. Check retention next
  session rather than re-explaining unprompted.
