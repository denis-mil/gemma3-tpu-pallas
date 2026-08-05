# Copy elision verified; the boundary held; the instrument's configuration went unnamed

Denis reproduced lesson 03's central claim — `grid=(3,4)` costs `(Q 3, K 12)` DMAs and
`grid=(4,3)` costs `(Q 12, K 4)`, same twelve steps, same block shapes — and stated the
measurement's boundary correctly and unprompted. **Lesson 0003 is assessed and held;** gap 3
in `NOTES.md` closes and lesson 05 may lean on residency. The session also exposed a
recurring failure mode that is *not* about Pallas: he reports mode-dependent results without
naming the mode.

**Held, unaided:**

- **The boundary, stated before the run rather than after.** Asked to finish both halves in
  advance: what a clean interpret-mode run establishes is that the documented
  consecutive-index elision is real behaviour in the one implementation exercised and that
  the chain *`index_map` ignores the fast axis → slice unchanged between consecutive steps →
  transfer skipped* holds end to end; what it does not establish is anything about a v5e,
  where the pipeline is emitted by Mosaic. This is the mission's "which roof binds, with the
  arithmetic to back it" discipline applied to his own instrument, and it is the single most
  valuable thing in the session.
- **The pollution mechanism, explained rather than merely observed.** Under `interpret=True`
  a store to an input ref reaches the backing array, so a revisited block's re-fetch carries
  the sentinel back — "refilled with the evidence", his phrase. Under
  `pltpu.InterpretParams()` HBM and VMEM are modelled as separate memories joined only by
  simulated DMAs, so the store has nowhere to go and the re-fetch arrives clean. A run
  reports *that* the two interpreters differ; it does not say *why*. The why is his.
- **`(0,1]`, so `exp` cannot overflow.** The one sentence
  [[0003-online-softmax-executes-and-generalises]] recorded him stopping short of, delivered
  cold as lesson 03's spaced Q5. Gap 2's residual is closed. **Spacing has now worked twice.**

**Not held:**

- **Mode discrimination.** Asked for K's twelve raw reads at `grid=(3,4)`, he gave the
  all-zeros trace — which is exactly right under `InterpretParams()` and wrong under
  `interpret=True`, where the answer is `[0 0 0 0 1000 1001 … 1007]`. He named no mode. He
  had **both traces on screen at the time** (see Evidence), and one answer later
  demonstrated he knew they diverged. So this is not recall failure. It is the same error
  class as the session-2 `d`-sign inversion and the `grid=(4,3)` axis-swap trap written up
  in lesson 03: **structure correct, discriminator dropped.** Three instances now. Treat it
  as his standing failure mode and probe for it directly rather than hoping it surfaces.
- **One unbounded hardware claim.** "On hardware there is no path for that write to reach
  HBM, so the pollution mechanism should not exist" — a plausible inference (input buffers
  have no `copy_out`) asserted as fact, in the same answer set whose other half correctly
  refuses to assert anything about a v5e. He accepted the correction. Watch for the
  asymmetry: he polices the boundary when asked about it and crosses it when not.

**Evidence:** three prediction questions issued before the probe was run, plus lesson 03's
quiz Q5. Q3 and Q5 are clean — neither is answerable by running anything. **Q1 and Q2 are
compromised as assessment:** he ran the probe before answering. The teacher said
"closed-book" and "do not open the research doc" but never said "do not run it", while
volunteering that the script existed and was unrun — an invitation, not a gate. Q2's
predicted traces therefore record observation, not prediction; its mechanism prose still
counts, on the reasoning above. Verified counts and both modes' raw traces are in
`docs/research/0001` §3.3, corrected this session.

**Implications:**

- **A prediction question must gate the run explicitly, or be unanswerable by running.** The
  second is better. [[0003-online-softmax-executes-and-generalises]] established break-it
  questions as the cheapest instrument on him; this establishes their precondition. Q3 is the
  model to copy — *state the boundary of a measurement you have not taken yet* has no
  executable answer, which is why it survived a session in which the gate failed.
- **The verification errand produced a repo correction, and that is the transferable win.**
  `docs/research/0001` §3.3 recorded the `interpret=True` traces unlabelled, generalised the
  write-through to "interpret mode", and the Appendix credited the experiment to
  `InterpretParams()` — which does not produce those digits. A research document written by an
  agent and read twice still carried a mode error until someone ran it two ways. That is the
  lesson about primary sources that the mission actually cares about, and it landed by
  accident.
- **Gap 3 closes; do not re-teach residency.** Lesson 04 (`(8,128)` and the VMEM budget) is
  next. Re-derive §4.3's arithmetic before teaching it — it is marked *inferred*.
