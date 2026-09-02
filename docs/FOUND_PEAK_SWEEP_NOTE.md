# FOUND_PEAK sweep — quantifying the severity-4 rejection cliff

**Date:** 2 Sep 2026 · **Author:** DevaanshGupta8 · **Status:** measurement only, NO code change.
For Aryan's decision. Nothing here is fitted to the organizers' 20 validation pairs.

The `score` column the pipeline already outputs **is** the peak-NCC that `found` thresholds
on (`route.predict_full`: `score = info["score"]; found = int(score >= FOUND_PEAK)`), so the
threshold can be swept offline from the existing per-pair CSVs — no re-run, no retuning.
Pooled data: 549 pairs across the three draws (425 present / 124 absent).

---

## The finding that outranks the threshold question: our reported localization may be OPTIMISTIC

`scripts/score_phase2.py` computes localization error from the pipeline's **internal**
predicted `res.x, res.y` — even for pairs it rejects (`found=0`). But the shipped
`register.py` (via `PairResult.as_row`) writes **`x=0, y=0`** whenever `found=0`, as the
addendum mandates ("when found==0, write 0 in the pose columns"). If the organizers score
the submitted file — which contains 0,0 for those pairs — then every present pair we
false-reject contributes **0 localization credit**, not the ~0.9 credit `score_phase2` gives it.

The severity-4 pairs localize *accurately* (credit ~0.9) but are ~90% false-rejected, so this
gap is large. On the real-composition mixed 200-set at FOUND_PEAK=0.53:

| Localization accounting | /40 |
|---|---|
| `score_phase2` as-is (credits internal x,y regardless of found) | **35.42** |
| Real submission (found=0 → 0,0 → 0 credit) | **31.85** |

**~3.5 points of localization we have been reporting may not exist in the real submission**,
because the sev-4 cliff forfeits those pairs' x,y as well as marking them absent.

**Important caveat — this hinges on an unresolved rubric detail.** How localization is scored
for a present pair we declare `found=0` has three plausible readings:
1. **Zeroed-and-scored** (most literal, given the mandated 0,0): rejected present → 0 credit.
   Our real localization is ~31.9/40; lowering FOUND_PEAK recovers credit.
2. **Independent** (what `score_phase2` currently assumes): localization scored on x,y
   regardless of found. 35.4/40; threshold barely matters for localization.
3. **Conditional-exclude**: localization averaged only over found=1 present pairs. Rejecting
   hard pairs would *raise* the mean; lowering the threshold would hurt.
We cannot resolve this without the organizers' scorer or a clarification. Reading 1 is the
most literal and is assumed for the "pessimistic" column below.

---

## The sweep (mixed 200, real composition A70/B70/C40/D20)

`core_opt` = localization independent of found (reading 2). `core_pes` = found gates
localization+pose (reading 1, ≈ real submission).

| thr | FN (present rejected) | FP (absent accepted) | F1(present+) | loc_pes /40 | core_pes /85 | core_opt /85 |
|----:|----:|----:|----:|----:|----:|----:|
| 0.41 | 2 | 18 | 0.940 | 34.97 | 73.02 | 73.47 |
| 0.43 | 4 | 13 | 0.948 | 34.77 | 72.65 | 73.25 |
| **0.45** | **4** | **9** | **0.960** | **34.77** | **72.45** | 73.05 |
| 0.47 | 10 | 9 | 0.940 | 33.61 | 71.36 | 73.21 |
| 0.49 | 14 | 7 | 0.932 | 32.81 | 70.60 | 73.20 |
| 0.51 | 17 | 3 | 0.934 | 32.10 | 69.92 | 73.17 |
| **0.53 (shipped)** | **18** | **2** | **0.934** | **31.85** | **69.67** | 73.17 |
| 0.57 | 23 | 0 | 0.922 | 30.94 | 68.61 | 73.04 |
| 0.61 | 26 | 0 | 0.911 | 30.29 | 67.78 | 72.88 |

Reading it:
- **FOUND_PEAK = 0.45 dominates 0.53 on our data on nearly every axis:** rejection
  F1(present+) peaks at **0.960** (vs 0.934), false-rejects fall **18 → 4**, at the cost of
  false-accepts **2 → 9**. Under the realistic pessimistic accounting that is **+2.8 core**;
  under the optimistic accounting it is roughly neutral.
- Going **below ~0.45** keeps raising `core` on our data only because F1(present+) is
  present-weighted (present:absent ≈ 3.4:1) and stops penalizing accepted decoys — an
  **artifact that will not transfer to the organizers' real Set C**. So 0.35 "winning" is not
  a real signal; ~0.45 is the honest candidate.

---

## Why this is NOT a unilateral change — the risks

1. **Overfit risk (the same one that sank the net).** This optimizes the threshold against
   **our generator's** absent-score distribution. If the organizers' real Set C decoys score
   differently, a lower threshold could over-accept on the real set. `FOUND_PEAK=0.53` was
   chosen on a cost argument and **validated on their 20-sample (F1 0.963, recovered p014 at
   0.557)**; 0.45 has not been.
2. **The gain depends on the unresolved reading 1 vs 2 vs 3 above.** Under reading 2 or 3 the
   change is neutral-to-harmful.

## Recommendation

- **Surface the localization-accounting caveat regardless** — it makes our own numbers honest
  and belongs in `failure_analysis.pdf` (the sev-4 cliff costs localization too, not just
  rejection F1). Optionally add a `--strict` mode to `score_phase2.py` that scores the zeroed
  output, so we can see the real-submission number directly.
- **Do NOT change FOUND_PEAK unilaterally.** If Aryan wants to act, the disciplined path:
  (a) adopt the pessimistic loc accounting; (b) re-derive the cost-optimal threshold on our
  generator; (c) **sanity-check, not fit,** on the 20-sample that F1 stays ≥0.90 and
  p008/p014 don't regress. **0.45 is the evidence-based candidate.** Given the overfit risk and
  that 0.53 is already sample-validated, "document and leave it" is also fully defensible.

---

## Reproduction

```bash
.venv/Scripts/python sweep_found.py perpair150.csv perpair200.csv perpair_mixed.csv   # pooled
.venv/Scripts/python sweep_found.py perpair_mixed.csv                                  # real-composition
```

(`sweep_found.py` and the per-pair CSVs are in the session scratchpad; the CSVs come from
`scripts/score_phase2.py --out`. No repo code was modified.)
