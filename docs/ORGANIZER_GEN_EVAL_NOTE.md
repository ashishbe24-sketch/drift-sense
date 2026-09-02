# Organizer-generator eval: 150- and 200-pair score distributions

**Date:** 2 Sep 2026 · **Author:** DevaanshGupta8 (measurement only — no core change)

Purpose: turn the handoff's hand-wave ("expect the real score below 89%") into a
**measured** score distribution, by scoring the shipped classical pipeline on many
pairs drawn from **Applied Materials' own Phase 2 generator** with new seeds and
randomised poses. Feeds `failure_analysis.pdf`.

---

## Method

- **Generator:** `AMP_Phase 2 material/generator` (the organizers' source), called
  via its own `generate_phase2_sample(arch, params, rng)`.
- **Compliance (the no-appeal DQ line):** NEW seeds (base 990000 for the 150-set,
  3000000 for the 200-set — both disjoint from the 770000 training range and the
  20260827 sample seed) and NEW randomised architecture / zoom / theta / severity.
  The 20 provided sample pairs (p001–p020) are never read — they remain a pure
  validation fold. This is exactly what their spec sanctions ("keep the sampling
  path for scaling to 200 pairs later").
- **Composition** mirrors the real blind 200-set's present/absent balance
  (A : B : C = 70 : 70 : 40, Set D held out for now):
  - **Set A** — nominal (severity 0), reference present.
  - **Set B** — degraded, present, **severity skewed to 3–4** (weights
    1:0.15, 2:0.20, 3:0.35, 4:0.30) because the handoff records the real Set B as
    skewing to the heavier severities. This makes the estimate honest / if anything
    pessimistic, not flattering.
  - **Set C** — absent (no true instance), mild severity 0–2.
- **Pipeline:** `route.predict_full(..., use_net_xy=False)` — the shipped classical
  path supplies all six fields (net OFF). `FOUND_PEAK = 0.53`, unchanged.
- **Scorer:** `scripts/score_phase2.py` — the exact Phase 2 rubric.
- **Env:** generation in a scratch venv with opencv 5.0 (the generator needs cv2);
  scoring in `.venv`. Both dataset + scripts live outside the repo (scratchpad),
  regeneratable, nothing staged.

Two **independent** draws (150 and 200, disjoint seeds) so the spread between them
measures sampling noise, not just a single lucky/unlucky set.

---

## Results — rubric, both draws

| Component | **200 pairs** | **150 pairs** | Organizer 20-sample |
|---|---|---|---|
| Localization /40 (A/B-weighted) | **36.67** | **34.76** | 35.60 |
| ↳ Set A mean credit | 0.915 | 0.898 | 1.000 |
| ↳ Set B mean credit | 0.918 | 0.840 | 0.800 |
| Pose — scale /10 | 7.43 | 7.25 | ~8.9 |
| Pose — rotation /10 | 8.41 | 8.49 | 10.0 |
| Rejection F1 /15 | 13.60 (F1 0.907) | 13.13 (F1 0.876) | ~14.5 (F1 0.963) |
| Calibration AUC /10 | 7.40 (0.740) | 7.40 (0.740) | ~0.73 |
| **Core /85 (auto-scorable)** | **73.51** | **71.03** | ~74 |
| Runtime median | 1.27 s/pair | 1.30 s/pair | ~1.5 s/pair |

### Accuracy, three senses

| Meaning | 200 | 150 |
|---|---|---|
| Localization hit rate (present ≤5 px) | **97.4%** | 92.7% |
| End-to-end correct (right spot AND right present/absent call) | **84.5%** (169/200) | 80.7% (121/150) |
| Rubric core | 86.5% (73.51/85) | 83.6% (71.03/85) |

---

## Localization by Set B severity

| Severity | 200-set: n / mean credit / %@5px | 150-set: n / mean credit / %@5px |
|---|---|---|
| B sev1 | 16 / 0.912 / 100% | 9 / 1.000 / 100% |
| B sev2 | 15 / 0.973 / 100% | 5 / 0.680 / 80% |
| B sev3 | 28 / 0.886 / 100% | 23 / 0.835 / 91% |
| B sev4 | 19 / 0.926 / 100% | 18 / 0.811 / 89% |

Localization stays accurate even at severity 4 (median error <1 px). The credit that
is lost there is lost to the **rejection flag**, not to a mislocalization — see below.

---

## Key finding: the severity-4 rejection cliff (reproduces on both draws)

| False-rejects (present pair flagged `found=0`) | 200-set | 150-set |
|---|---|---|
| B sev3 | 8 / 28 (29%) | 4 / 23 (17%) |
| **B sev4** | **17 / 19 (89%)** | **17 / 18 (94%)** |
| All present | 25 / 156 (16%) | 22 / 110 (20%) |
| False-accepts (absent flagged `found=1`) | 2 / 44 (4.5%) | 3 / 40 (7.5%) |

At severity 4, dose+noise+aberration crush the true-match correlation peak below
`FOUND_PEAK = 0.53`, so ~90% of present sev-4 pairs are declared **absent**. This is
the single dominant driver of the rejection-F1 drop from 0.963 (mild 20-sample) to
**0.88–0.91** on a realistically-skewed Set B.

Root tension (already noted in the draft as open-item #6, now quantified): a **single
scalar** peak threshold cannot simultaneously reject periodic absent decoys (wants a
high threshold) and accept crushed present peaks (wants a low one) — they overlap.
The honest fix is a richer / severity-aware `found` signal, not a different scalar.
**Not changed here** — `FOUND_PEAK` stays 0.53 (settled; must not be tuned on the 20
validation pairs). Flagged as the clearest post-deadline lever for the lead's call.

---

## Honest caveats

- **Sampling noise ≈ ±2–3 pts.** Two independent draws gave core 71.0 and 73.5,
  localization 34.6 and 36.7. Quote a **range (~84–86% core)**, not a point estimate.
  The 200-set drew an easier Set B (localized 100% @5px), which is why it reads higher.
- **Still our-vs-their-generator gap is unmeasurable.** These pairs come from their
  generator SOURCE but their real 200-set parameters (Set B ranges) are undisclosed;
  this is the closest honest proxy, not the real thing.
- The **calibration AUC is stable at 0.740** across both — consistent with the
  ~0.72–0.79 seen elsewhere; the weakest scored bucket, unchanged.

---

## Reproduction

```bash
# scratch env with opencv (generator needs cv2); .venv is torch-only, no cv2
python -m venv genenv && genenv/Scripts/pip install opencv-python-headless

# 150-set (base 990000) / 200-set (base 3000000)
genenv/Scripts/python gen_rubric_eval.py \
    --gen-dir "AMP_Phase 2 material/generator" --out organizer_gen_eval200 \
    --n-a 78 --n-b 78 --n-c 44 --n-d 0 --seed-base 3000000 --workers 3

# score with the shipped classical pipeline
.venv/Scripts/python scripts/score_phase2.py organizer_gen_eval200 \
    --ckpt driftmatch/checkpoints/best_phase2_speckle.pt --out perpair200.csv
```

(`gen_rubric_eval.py` and both datasets are in the session scratchpad — regeneratable,
deliberately not committed.)
