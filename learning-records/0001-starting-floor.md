# Starting floor: attention yes, flash no, Pallas barely

Denis stated his starting point at the opening of session 1: comfortable with attention
itself (QK<sup>T</sup>, softmax, masking) but new to flash-style tiling and online
softmax; Pallas experience limited to getting interpret mode running, with grid,
`BlockSpec` and `index_map` still fuzzy. He also flagged not recalling what **PV** meant.

This sets the sequencing for everything downstream. Attention fundamentals must not be
re-taught — but nothing may assume tiling, the online-softmax recurrence, or the Pallas
pipelining model until each is explicitly built. In particular, lesson 0001 was
deliberately written to require *neither* flash nor Pallas knowledge: tile-level mask
arithmetic is reachable from plain attention alone, which is what made it the right first
step rather than a compromise.

The mission is transferable kernel-authoring skill, not shipping this one kernel
(see [[MISSION.md]]) — so lessons should teach predicates and derivations that Denis
evaluates himself, rather than the Gemma-specific answers.

**Evidence:** self-reported, session 1, in response to a direct question about floor.
No demonstrated understanding recorded yet — lesson 0001 has been delivered but not
assessed. Assess before writing a record about it.

**Implications:**
- PV was defined in passing rather than made a topic. Check retention next session; do
  not re-explain unprompted.
- Online softmax (lesson 02) is the single largest gap blocking Phase 3.
- The Phase 2 gated-MLP warm-up exists to teach grids and `BlockSpec`s against trivially
  checkable correctness — per [ADR-0001](../docs/adr/0001-windowed-prefill-as-the-optimisation-target.md),
  that ordering is deliberate and lessons should respect it.
