# Online softmax: executes on numbers, and separates correctness from range

Denis ran the recurrence to five decimals on an unseen two-tile instance, and — unprompted
— justified *why the running max is not required for correctness* more sharply than lesson
02 states it. Lesson 0002 is **assessed and held**. Phase 3 may be built on it. The
residual `full`-vs-`live` hole from [[0002-tile-predicate-derives-under-concrete-instance]]
is **closed** by the same assessment.

**Held, unaided:**

- **The recurrence, executed.** `m₂ = 5`, `α₂ = 0.135335`, `ℓ₂ = 1.203438`,
  `O₂ = (1.117890, 1.135335)`, answer `(0.928913, 0.943410)` — every figure correct, on a
  concrete instance the reference sheet cannot answer.
- **Both accumulators rescaled, in step.** This was the trap the question was built around:
  rescaling `ℓ` but not `O` returns `(1.026151, 1.661905)`, a visibly wrong vector rather
  than a near miss. It did not land. Trap 1 is not his failure mode.
- **The max is not load-bearing for correctness.** Asked what breaks if the running max is
  replaced by the running *minimum*, he said it still works, because — his words — nothing
  in the correctness argument mentions the max; it only cares that both accumulators were
  built under the same `m_old` and receive the same `α`. That is the §4 claim stated better
  than §4 states it, and it is the generalisation, not the instance.
- **Both range consequences of the min-variant:** `α ∈ [1,∞)` so the rescale amplifies once
  per tile, and exponents `≥ 0` so every term inverts from `(0,1]` to `[1,∞)`.
- **`full` as containment, not intersection** — see below.

**Not held:**

- **The consequence of the inverted range was never named.** He described the mechanism
  completely and stopped one step before *"therefore the no-overflow guarantee is gone"*.
  The split `NOTES.md` cares about — *"running" comes from the algorithm, "max" comes from
  the hardware* — is therefore two-thirds present: he owns the algorithm half outright, and
  the mechanism of the hardware half, but does not spontaneously close it into a statement
  about float range. Reteach cost: one sentence, not a lesson. Fold it into a later lesson
  by spacing.

**Closed from [[0002-tile-predicate-derives-under-concrete-instance]]:**

- **`full` is now held as a relation distinct from `live`.** Given `N=512, W=256, B=128`,
  he named `q[256,383] × k[128,255]` full (`d ∈ [1,255] ⊆ [0,255]`) and
  `q[256,383] × k[0,127]` live-not-full (`d ∈ [129,383]`, intersects, not contained), and
  justified each by the *relation* rather than by intersection alone. Asked twice in
  session 2 and skipped twice; answered cleanly here after one lesson of spacing.
- **Sign discipline on `d` held.** `d_min = q_lo − k_hi` in the correct direction. The
  session-2 inversion did not recur.

**Evidence:** three-question closed-book assessment, session 4. Written by the teacher in
session 3 and verified against NumPy and against `reference.attention_mask(512, 512,
window=256)` before being put to him; answers recorded in the session-3 handoff. Q1 an
unseen numeric instance, Q2 a deliberate-break generalisation, Q3 spaced retrieval back to
lesson 0001.

**Implications:**

- **Phase 3 is unblocked.** Gap 2 in `NOTES.md` closes. Nothing downstream needs to
  re-teach the recurrence.
- **Deliberate-break questions are the strongest instrument found so far.** Q2 asked him to
  damage the algorithm and report the damage, and it extracted a *better* statement of the
  invariant than free recall did in session 2 — because a reference sheet lists what the
  algorithm is and cannot say what a specific wrong choice does. Record 0002 established
  that a formula on a reference sheet is a retrieval hazard; this establishes the cheapest
  way around it. **Prefer break-it questions over recall-it questions from here on.**
- The concrete-instance-first ordering from
  [[0002-tile-predicate-derives-under-concrete-instance]] is confirmed a second time, and
  should stand for lesson 03 — lead with the `grid=(3,4)` vs `grid=(4,3)` DMA counts, not
  with what `index_map` is.
- Spacing works on this learner: one intervening lesson turned a twice-skipped question
  into a clean answer. Keep seeding prior-lesson probes into later lessons rather than
  scheduling reteaches.
