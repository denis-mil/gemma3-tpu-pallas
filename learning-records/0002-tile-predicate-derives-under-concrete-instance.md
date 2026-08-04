# Tile predicate: derives on a concrete instance, not from the abstract argument

Denis can derive the live/full tile predicate when handed a specific tile and asked for
specific quantities, but could not reconstruct it from the abstract prompt and reached for
the reference sheet instead. Lesson 0001's content is therefore **assessed and largely
held** — with one real hole (`full` as a distinct relation) and one pedagogy finding that
should change how later lessons are built.

**Held, unaided:**

- Which denominator a "64×" claim is against — `N²` dense vs `N²/2` causal (32×). Kept
  straight without prompting.
- **PV** — retained. `probs @ V`, the second attention matmul. The
  [[0001-starting-floor]] flag is closed; strike it from `NOTES.md`.
- Block-size cost: `B = W = 512` loads 2.00× the pair-level floor, and he named the floor
  correctly as pair-level rather than as any tile-based number.
- On a concrete tile (`q ∈ [8,11]`, `k ∈ [4,7]`, `W = 4`): computed `d_min = q_lo − k_hi`,
  `d_max = q_hi − k_lo`, identified the mask as the band `0 ≤ d ≤ W−1`, and classified the
  tile by interval intersection. This is the derivation, done rather than recalled.
- Justified contiguity of the `d` set unprompted and well: *a rectangular tile is a dense
  product of two contiguous ranges*. Better than the descent argument that was being
  fished for.

**Not held:**

- **`full` is not stored as a separate relation.** Asked twice, skipped twice. He has
  `live` = intervals intersect; he does not have `full` = mask band *contains* the tile's
  `d` interval, nor `partial` = live ∧ ¬full. Right answer given for the concrete tile,
  but justified only by the intersection half.
- **Sign discipline on `d`.** Asked how to reach `d − 1` while staying in the tile, he
  answered "increase `q` or decrease `k`" — both of which reach `d + 1`. The structural
  insight (a non-minimal pair has slack in at least one coordinate) was correct; the
  direction was inverted. Worth watching: this is the error class that yields a kernel
  that runs and is wrong.
- Unprompted recall of the abstract derivation. Twice he produced the formula verbatim
  and neither time the argument beneath it.

**Evidence:** free-recall assessment, session 2, five questions with the lesson and
reference sheet closed, followed by a concrete-instance reteach after the abstract form
failed.

**Implications:**

- **Lead with the concrete instance, then generalise.** The abstract framing ("why is it
  only two comparisons") failed twice; a four-by-four tile with real integers landed
  immediately. Lesson 0001 taught the general form and let the numbers follow. Invert that
  for lessons 02–05 — this matters most for online softmax, where the recurrence is
  exactly the kind of object that reads as obvious and stores as nothing.
- **A formula on a reference sheet is a retrieval hazard.** `reference/` is doing its job
  as reference and quietly undermining assessment. Future checks must use instances the
  sheet cannot answer.
- Re-probe `full` vs `live` as *two relations between the same two intervals* — by
  spacing, inside a later lesson, not by repeating lesson 0001.
- Not a blocker for lesson 02. Online softmax does not depend on the `full` case.

**Also established this session** (by the teacher, not the learner — folded into
`reference/block-skipping-arithmetic.html` rather than taught): hull-relaxing a mask is
always *sound*, never exact. Replacing the achieved `d` set by `[d_min, d_max]` only
grows it, so `live` can false-positive (a wasted DMA) and `full` can false-negative (a
wasted elementwise mask). Both errors cost work, neither costs correctness. Contiguity
buys exactness, not validity — the framing originally put to Denis, that contiguity is
what makes the predicate legal, was wrong and was corrected in session.
