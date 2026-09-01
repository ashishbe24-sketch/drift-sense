# Start here — Phase 2 DriftSense, continuing from a prior session

**If you are picking this up fresh: read the three files below, in order,
before doing anything else.** They are the full compiled understanding, research, and
decision log from the work so far. Do not re-derive this from scratch or re-read the raw
PDF/transcript unless the user asks — it's already distilled here.

1. [`docs/PHASE2_UNDERSTANDING.md`](PHASE2_UNDERSTANDING.md) — the Phase 2 spec itself: what
   changed from Phase 1, dataset structure, I/O contract, scoring, allowed/disqualifying rules,
   timeline. Ground truth for "what are we being graded on."
2. [`docs/PHASE2_RESEARCH_NOTES.md`](PHASE2_RESEARCH_NOTES.md) — the full engineering log:
   every decision made, every number measured, every mistake made and fixed, in chronological
   order. This is the important one — it explains *why* the code looks the way it does.
3. [`docs/PHASE2_FAILURE_ANALYSIS_DRAFT.md`](PHASE2_FAILURE_ANALYSIS_DRAFT.md) — draft failure
   analysis for the submission's 10-point generator/citations/failure-analysis criterion.

## Current status (brutally honest, as of the end of the last session)

Roughly **65-70% of total submission readiness** (up from 55-60% as of 28 Aug -- rotation moved
from ~5% to ~70% overnight on 29 Aug, closing the single biggest gap). Breakdown:

| Piece | Status |
|---|---|
| Localization (classical scale-search + net hybrid router) | Done, validated. 81-83% @5px on fresh sets that now include real relative rotation (down from the earlier 89% figure, which had no rotation in the data yet -- confirmed via a rigorous A/B that this is not a code regression, see research notes 29 Aug). **Only tested through .venv's classical-only fallback until 29 Aug** -- .venv has no torch; re-verify future numbers through `C:\Users\ARYAN\AppData\Local\Programs\Python\Python312\python.exe` (has torch+CUDA) to get real router numbers, not the degraded fallback. |
| Scale recovery | Done. ~0.6-1% error. |
| **Rotation / `theta`** | **~70% done (29 Aug).** Relative rotation now exists in the generator (`raster.make_pair`'s `relative_theta_deg`), `solve.py` has a golden-section angle refinement (`PHASE2_ANGLES`/`_refine_angle`) mirroring the scale search, and `THETA_SIGN` was empirically fixed (the original guess was backwards -- see research notes). Self-consistent validation on our own generator: median abs theta error 0.286° on well-localized pairs (37%/74%/89% at the ≤0.25°/0.5°/1.0° tiers). **Still open:** the sign convention is only verified against OUR OWN generator, not the organizers' -- re-check the moment their sample ground truth lands (still not released as of 29 Aug). |
| Rejection (`found`) | Done, validated. F1 = 0.925, threshold cost-calibrated (not just raw F1 — see research notes for why that distinction mattered). |
| Confidence calibration (`score`) | **Tested (29 Aug): AUC = 0.657, mediocre.** Root cause found, not just measured: raw peak-NCC rejection has a real separability ceiling on "resolved"-regime (periodic) absent pairs -- their scores overlap present-pair scores almost completely (up to 0.97). Not a threshold-calibration problem; see research notes. Relevant to the teammate's recalibration task -- more data will likely help some but won't fully close this. |
| Efficiency | Done. ~1.6s/pair, well under the 5s budget. |
| Generator + citations + failure analysis (10 pts) | Generator extended (scale, absent pairs, full aberration suite). Citations/writeup not polished to submission standard yet. |
| **PPT for Phase 2** | **Not started.** |
| **Demo video for Phase 2** | **Not started** (Phase 1's video is stale). |
| **Final zip packaging** (requirements.txt regen, `failure_analysis.pdf` ≤2 pages as an actual file) | **Not started.** |

**The biggest unknown, not a percentage:** everything has been validated only on our own
generator. None of it has touched the organizers' actual data or sample pairs yet.

## Two open questions for the organizers -- ANSWERED (1 Sep)

1. Does "materially different from Phase 1 approach" tolerate extending the router with scale
   search + absent-pair rejection + a net retrained on the wider distribution?
   **Mentor answer: "Yes, they can definitely continue with the Phase 1 architecture and extend
   it. It's expected that they'll need to make adjustments to address the requirements for
   Phase 2."** Confirms our whole approach is compliant, including the 1 Sep change to which
   component (classical vs net) leads on `x,y` -- that is a threshold/policy adjustment within
   the same two declared components, not a new architecture.
2. Is the Rejection F1 computed with `found=1` or `found=0` as the positive class, or
   macro-averaged?
   **Mentor answer: standard binary-classification F1, no further disambiguation given.** Does
   not resolve which convention is used -- `scripts/score_phase2.py` already reports all three
   (`present+`, `absent+`, macro) for exactly this reason; no change needed, keep reporting all
   three.

## Priority order for continuing

1. **Rotation sign re-check against the organizers' sample** — the moment sample pairs/ground-truth
   theta are available, re-run the same comparison already done on our own generator
   (see PHASE2_RESEARCH_NOTES.md, 29 Aug entry) against their labelled theta. Everything else about
   rotation is implemented; this is a verification step, not new engineering.
2. ~~Validate the confidence-calibration AUC properly.~~ **Done (29 Aug): AUC 0.657, and a real
   rejection-separability weakness found on resolved-regime absent pairs (see research notes) --
   worth fixing eventually (better rejection signal than a single peak value), but deliberately not
   attempted yet: overlaps the teammate's in-flight recalibration task on the same file (`route.py`).
3. Polish the generator citations + failure analysis to submission quality (2-page PDF) -- rotation
   and the AUC/rejection findings both need folding into `PHASE2_FAILURE_ANALYSIS_DRAFT.md` with
   real numbers (rotation done; AUC finding still needs its own section).
4. Phase 2 PPT.
5. Phase 2 demo video.
6. Final zip assembly and a full dry-run of `register.py` against the organizers' actual
   `pairs.csv` once it's released.
7. **A better rejection signal than a single peak-NCC threshold -- investigated, deferred with a
   reason, not abandoned.** Tried two literature-grounded candidates (`second_ratio`/`distinct`,
   then proper Peak-to-Sidelobe Ratio at 4 radii) -- both lose to plain peak NCC on the periodic-
   absent case (see research notes, 29 Aug). The likely real fix (a small classifier combining
   multiple diagnostics, or using the net's own heatmap confidence) needs more labeled absent-pair
   data than currently exists (only 28 in the largest set so far) to fit and validate honestly.
   **Deferred until the teammate's 300-pair calibration set lands** -- pick this up then, not
   before, to avoid overfitting a "fix" to too little data.

Note: a teammate is separately working `docs/TEAMMATE_TASK_LARGER_TRAINING.md` (bigger training
set, net retrain, rejection-threshold recalibration on more data) -- don't duplicate that work in
this session; pick up from whatever they report back.

## Practical notes for whoever picks this up

- **Submission deadline: 3 September 2026, 23:59** (T+7 from the 27 Aug Phase 1 results — this
  is the PS-02-specific date, not the general hackathon site's date).
- The submission entry point is `python register.py --input pairs.csv --output predictions.csv`
  — this filename/signature is literally mandated by the addendum, not a choice.
- The active checkpoint is `driftmatch/checkpoints/best_phase2.pt` (trained on the full
  aberration suite: scale + astigmatism + barrel distortion + vignette + gamma + scan
  distortion). The original Phase 1 `best.pt` is untouched and still used by `infer.py` (the
  Phase 1 entry point) — don't confuse the two.
- `torch` is required for the net path (`requirements.txt` already lists it); without it,
  everything degrades gracefully to classical-only (verified, not just assumed). **This repo's
  `.venv` does NOT have torch installed as of 29 Aug** -- any test run through `.venv` silently
  falls back to classical-only, which is meaningfully weaker than the shipped router (see 29 Aug
  research notes for the size of the gap: ~63-68% vs ~81-83% on the same pairs). For numbers meant
  to represent the shipped system, use
  `C:\Users\ARYAN\AppData\Local\Programs\Python\Python312\python.exe` instead (torch 2.5.1+cu121,
  CUDA available -- the environment Phase 1 training already used), and load
  `driftmatch/checkpoints/best_phase2.pt` via `route.load_net()` explicitly (matching what
  `register.py` does) rather than the shared `route.DEFAULT_CKPT` (which points at the Phase 1
  checkpoint).
- Nothing has been git-committed from the Phase 2 work as of this handoff — check `git status`
  before assuming anything is saved beyond the working tree.
- If regenerating training/eval data from the research notes' documented commands, note it can
  take 30-40 minutes for a 4000-pair set — plan around that, don't assume it's instant.
- **A GPU-cleanup mistake was made and fixed once already** (see research notes) — a launcher
  script blindly killed all python processes and took out a concurrent legitimate job. If
  writing any new training/generation launcher script, do NOT blindly kill processes by name;
  target specific stale PIDs if cleanup is genuinely needed.
