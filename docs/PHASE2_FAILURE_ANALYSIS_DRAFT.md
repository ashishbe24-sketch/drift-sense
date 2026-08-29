# Phase 2 Failure Analysis — working draft

Not the final submission artifact. This captures honest, **evidence-based** failure modes
discovered while building and testing the Phase 2 pipeline (28-29 Aug), so the real
`failure_analysis.pdf` (max 2 pages, required in the submission zip per the addendum) can be
written from real numbers instead of reconstructed from memory near the deadline. TBDs mark what
depends on decisions not yet finalized (net-vs-classical architecture, rotation).

---

## 1. Scale mismatch was the dominant new failure mode, and it's fixed

Before scale search, running the (Phase-1, fixed-scale-10) solver on Phase 2's [8,12] zoom range
produced **3 catastrophic failures in 12 pairs** (570px, 390px, 88px errors) — the reference was
shrunk by the wrong factor, so the template didn't even match its true footprint. This was the
single highest-impact issue found, because it silently wrecks localization, not just the pose
score. Fixed via a coarse-to-fine scale search + golden-section refinement (median scale error
0.62% after the fix). **Residual risk:** the search grid is [8,12] in 0.5 steps — if the
organizers' actual distribution has structure our even coverage doesn't anticipate, refinement
quality could vary; not observed in testing, but not exhaustively ruled out either.

## 2. A large fraction of "failures" are unsolvable by construction, not bugs

Diagnosing 9 localization misses (of 44 present pairs) on a 60-pair set: 4/9 were `below_floor`
difficulty — pairs our generator deliberately makes unsolvable (landmark below the visibility
floor, ~8% by design policy, matching what Applied Materials' problem statement implies about real
tool limits). On solvable pairs only, classical localization is closer to 88% than the 80%
headline. **Honest framing for the write-up:** report both numbers — the raw score the organizers
will compute (which includes intrinsically hard cases) and the solvable-only number (which
isolates the method's actual weakness) — rather than letting one number overstate or understate
the method.

## 3. Raw correlation confidence is NOT a reliable rejection signal — a real near-miss

First attempt at the `found` (Set C) rejection used peak NCC directly and looked perfect on a
small (18-pair) calibration set (F1 1.0). On a larger, harder 60-pair set it dropped to F1 0.86,
because absent pairs on **periodic structure** can score a high raw peak (up to 0.91) even with no
true landmark — the same periodicity that causes multi-match localization ambiguity also produces
false-positive-prone rejection signals. This was caught by testing on a bigger, harder set before
shipping, not by the small set alone — worth stating plainly in the write-up as a methodology
point (calibrate rejection thresholds on hard, periodic-heavy data, not easy data), since it
reflects understanding of *why* Set C is designed the way it is ("a different die region of the
same architecture" is a deliberate hard-negative, not a random one).

## 4. Geometric warps: classical breaks, and we can now show exactly where and why

Added scan distortion + astigmatism + barrel/pincushion + vignette + gamma to the generator (none
were previously implemented, despite being named in the generator spec). On a 20-pair set with all
aberrations enabled, 17/19 solvable pairs still localized to sub-pixel accuracy (ground truth
verified exact under compounded warps), but **2/19 failed classical badly** (229px, 270px) —
aberrations strong enough to make a confident, wrong correlation peak win over the true, warped
site. This is a genuine, reproducible failure mode of pure correlation matching, and it's the
concrete evidence for *why* a learned component is included in the submission rather than
classical-only: [TBD — depends on final net-vs-classical/router decision, not yet made as of this
draft; see PHASE2_RESEARCH_NOTES.md for the live comparison numbers].

## 5. Rotation — implemented; the sign convention was wrong on the first attempt, caught by testing

Implementing relative rotation exposed a genuine bug rather than just a missing feature: the first
`THETA_SIGN` guess (`-1.0`, chosen by pattern-matching the geometry without a labelled example) was
backwards. Generating 30 pairs with our own generator's new signed ground truth and comparing
against the recovered angle gave correlation **-0.95** with the original sign (median error 3.35°)
and **+0.95** after flipping it (median error 0.26°) — a clean, unambiguous signal, not a close
call. This is the same lesson the barrel-distortion sign already taught earlier in the project:
**a geometric sign/pivot convention should never ship on a pattern-matched guess; build the
synthetic or self-consistent test first.** Once fixed, on the well-localized subset of a 30-pair
self-generated test set: median absolute theta error 0.286°, with 37% / 74% / 89% of pairs at the
≤0.25° / ≤0.5° / ≤1.0° pose-recovery tiers respectively. Scale accuracy on the same pairs (median
0.98% error) is unchanged from the rotation-free numbers, i.e. the new rotation axis does not
degrade the existing scale search.

**Honest limitation, unresolved as of this draft:** the above validates that our recovery pipeline
is *self-consistent* with our own generator's convention — it does not yet confirm that convention
matches the organizers' own "CCW positive, about the match centre" definition, since their sample
ground truth has not been released. If their convention differs (a flipped sign, or a different
rotation pivot), every reported `theta` would be systematically wrong until re-verified against
their sample — a known, tracked risk, not a silent one.

## 6. A single peak-NCC threshold has a real ceiling on periodic absent pairs, not just a calibration gap

Measuring the `score` column's AUC for the first time (previously assumed monotonic, never tested)
gave 0.657 on a fresh 150-pair mixed set — mediocre, and worth explaining rather than just
reporting. The cause: on "resolved"-regime pairs (the most common pitch regime, 60% of pairs by
design), an **absent** pair's peak correlation against its own periodic decoys can reach 0.97 —
statistically indistinguishable from a present pair's true-landmark peak (median 0.94). Sweeping
every threshold in [0.50, 0.95] on this set still only correctly rejects 6 of 28 absent pairs at
best. This is the same underlying fact that motivates the localization centre tie-break — periodic
structure produces near-identical candidate peaks — now shown to also defeat a single-value
rejection rule, not just localization. **Framing for the write-up:** this is not "the threshold
needs recalibrating" (more calibration data narrows an estimate; it cannot create separability a
signal doesn't have) — it is a legitimate open limitation of a single-scalar rejection signal, and
the honest fix is a richer signal (e.g. combining peak with the existing second-peak-ratio
diagnostic, already computed for the multi-match case, in a regime-aware way), not a threshold
sweep. Not implemented as of this draft — flagged as the clearest concrete next step for `found`.

## 7. Cross-generator risk remains real (carried forward from Phase 1, still applies)

Phase 1's honest limitation — "a third, unseen generator is the real open risk" — still applies
and arguably more so in Phase 2: our own generator's aberrations were calibrated by us, not
measured against the organizers' actual Set B parameter ranges (deliberately undisclosed, per the
addendum). We hedge with breadth (independently-gated, randomized aberration parameters covering a
wide plausible range) rather than a best guess at their exact distribution, but the gap is
unmeasurable until scored.

---

**For the final 2-page PDF:** trim to the highest-signal 3-4 items (likely #1 (scale), #5
(rotation), #6 (rejection ceiling), and whichever of #3/#4/#7 is most developed by submission
time), each with the concrete number, not just the description — the organizers explicitly asked
for a version of the Phase 1 failure analysis
"re-judged under Phase 2 conditions," which reads as wanting genuine new failure modes, not a
copy-paste of the old ones.
