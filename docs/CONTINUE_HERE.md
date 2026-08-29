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

Roughly **55-60% of total submission readiness.** Breakdown:

| Piece | Status |
|---|---|
| Localization (classical scale-search + net hybrid router) | Done, validated. 89% @5px on our hardest test set. |
| Scale recovery | Done. ~0.6-1% error. |
| **Rotation / `theta`** | **~5% done — only sampled in the generator, no relative geometry, no recovery algorithm, zero validation.** Deliberately deferred: needs the organizers' sample ground-truth theta to verify the sign/pivot convention before implementing blind. **This is the single biggest remaining gap.** |
| Rejection (`found`) | Done, validated. F1 = 0.925, threshold cost-calibrated (not just raw F1 — see research notes for why that distinction mattered). |
| Confidence calibration (`score`) | Signal exists (peak NCC) but its actual AUC has never been tested. |
| Efficiency | Done. ~1.6s/pair, well under the 5s budget. |
| Generator + citations + failure analysis (10 pts) | Generator extended (scale, absent pairs, full aberration suite). Citations/writeup not polished to submission standard yet. |
| **PPT for Phase 2** | **Not started.** |
| **Demo video for Phase 2** | **Not started** (Phase 1's video is stale). |
| **Final zip packaging** (requirements.txt regen, `failure_analysis.pdf` ≤2 pages as an actual file) | **Not started.** |

**The biggest unknown, not a percentage:** everything has been validated only on our own
generator. None of it has touched the organizers' actual data or sample pairs yet.

## Two open questions for the organizers (not yet answered as of last session)

1. Does "materially different from Phase 1 approach" tolerate extending the router with scale
   search + absent-pair rejection + a net retrained on the wider distribution? (High confidence
   the answer is yes — the addendum's own language says "extend, don't rewrite" — but not
   confirmed.)
2. Is the Rejection F1 computed with `found=1` or `found=0` as the positive class, or
   macro-averaged? Affects how aggressively the reject threshold should be tuned.

## Priority order for continuing

1. **Rotation** — the moment sample pairs/ground-truth theta are available, this goes first.
   Biggest scored gap (10 of the 20 pose points).
2. Validate the confidence-calibration AUC properly.
3. Polish the generator citations + failure analysis to submission quality (2-page PDF).
4. Phase 2 PPT.
5. Phase 2 demo video.
6. Final zip assembly and a full dry-run of `register.py` against the organizers' actual
   `pairs.csv` once it's released.

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
  everything degrades gracefully to classical-only (verified, not just assumed).
- Nothing has been git-committed from the Phase 2 work as of this handoff — check `git status`
  before assuming anything is saved beyond the working tree.
- If regenerating training/eval data from the research notes' documented commands, note it can
  take 30-40 minutes for a 4000-pair set — plan around that, don't assume it's instant.
- **A GPU-cleanup mistake was made and fixed once already** (see research notes) — a launcher
  script blindly killed all python processes and took out a concurrent legitimate job. If
  writing any new training/generation launcher script, do NOT blindly kill processes by name;
  target specific stale PIDs if cleanup is genuinely needed.
