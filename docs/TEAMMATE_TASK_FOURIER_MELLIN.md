# Task: Fourier-Mellin invariant scale+rotation estimator (Phase 2, PS-02 Drift-Sense)

Read [`docs/CONTINUE_HERE.md`](CONTINUE_HERE.md) first for the overall project state, then
[`docs/PHASE2_UNDERSTANDING.md`](PHASE2_UNDERSTANDING.md) for the exact Phase 2 spec (what we're
graded on), then this file for your specific piece.

You are working from a fresh clone of the repo. Set up the environment first:
`python -m venv .venv`, then `.venv\Scripts\python.exe -m pip install -r requirements.txt`. This
task is **pure classical numpy/scipy/PIL — you do NOT need torch or a GPU for it.** (torch is only
for the neural path, which this task does not touch.)

## Why this task

On the Phase 2 orientation call, the Applied Materials mentor (Gokul Ganesan) explicitly said the
*best* way to handle the new unknown scale + rotation is **"moving to an invariant formulation. If
you can do that, well and good... Both are acceptable."** (verbatim from the session transcript).
An invariant formulation means recovering scale and rotation directly, in one shot, instead of
searching a grid for them.

The **Fourier-Mellin Transform (FMT) / log-polar phase correlation** is the textbook method for
exactly this: it recovers unknown rotation + scale + translation between two images by turning
rotation and scale into pure translations in log-polar space, solvable by a single phase
correlation. Right now our pipeline recovers scale and rotation by a coarse-to-fine **grid search**
(`solve.py`: `PHASE2_SCALES`, `PHASE2_ANGLES`, `_refine_scale`, `_refine_angle`). That works, but:
- it costs runtime (grid + golden-section refinement per pair), and
- a genuinely independent second estimator would let us **cross-check** scale/rotation and improve
  the 20 pose-recovery points (and possibly localization robustness).

Your job is to build that independent estimator as a **standalone, well-tested module** and measure
honestly whether it beats our current grid search on scale/rotation accuracy. If it does, we wire
it into the router afterward. If it doesn't, that's a real, useful negative result — report it
either way, with numbers, not vibes.

Reference implementations to study (do NOT copy blindly — understand them, then write our own clean
version that fits our code style):
- https://github.com/Smorodov/LogPolarFFTTemplateMatcher (OpenCV Fourier-Mellin)
- OpenCV's own `cv2.logPolar` / `cv2.warpPolar` + `cv2.phaseCorrelate`
- Background: the Fourier transform in log-polar coordinates is the Fourier-Mellin transform;
  polar transform of the 2-D FFT magnitude gives rotation invariance, log scaling of the radial
  axis gives scale invariance.

## Scope — build this, and ONLY this (isolation matters)

**Create ONE new file: `fmt_pose.py`** in the repo root, with a clean public function:

```python
def estimate_scale_rotation(reference, wide):
    """Estimate (scale, theta_deg) of `wide` relative to `reference` via
    Fourier-Mellin / log-polar phase correlation.

    reference, wide : 1000x1000 uint8 or float numpy arrays (same as solve.locate).
    Returns (scale, theta_deg, confidence) where:
      scale      : recovered downscaling factor, nominally in [8, 12]
      theta_deg  : recovered rotation, degrees, CCW-positive about the image
                   centre (MUST match our convention -- see 'Sign convention' below)
      confidence : the phase-correlation peak value, a [0,1]-ish quality signal
    """
```

Notes on the geometry you'll have to get right:
- The reference is 1 nm/px and the wide is `scale` nm/px, so the reference's footprint inside the
  wide view is `1000/scale` px (~83-125 px). You'll likely want to work from the wide image and the
  reference's frequency-magnitude spectra; the log-polar of the FFT magnitude is
  translation-invariant already (magnitude discards the shift), which is what lets FMT separate
  rotation+scale from translation.
- Windowing (Hann/Hanning) before the FFT matters a lot for FMT — spectral leakage from image
  borders wrecks the log-polar peak. Study why the reference implementations apply a window; apply
  one.
- A high-pass filter on the magnitude spectrum before log-polar is standard in FMT (suppresses the
  DC blob that otherwise dominates). Include it.

## Sign / scale conventions you MUST match (do not guess — verify against the generator)

Our generator produces ground-truth `scale` and `rotation_deg` you can validate against directly.
The contract (from `docs/PHASE2_UNDERSTANDING.md`): **theta is degrees, CCW-positive, about the
match centre.** Our own recovery pipeline was already validated to agree with the generator's
CCW-positive convention (see the rotation entry in
[`docs/PHASE2_RESEARCH_NOTES.md`](PHASE2_RESEARCH_NOTES.md) — the sign was empirically fixed there,
`THETA_SIGN` in `solve.py`, after it turned out the first guess was backwards). **Your FMT estimator
must produce theta with the SAME sign as `solve.locate`'s recovered theta and the generator's
`rotation_deg`.** Verify this empirically the same way — do not assume; generate a few pairs with
known rotation and check the sign of your output against the manifest's `rotation_deg`. Document
what you found in a comment, like `solve.py` does for `THETA_SIGN`.

## How to generate test data (you have the generator — use it)

```bash
python generate_dataset.py --phase2 --pairs 200 --seed 810000 --workers 5 --out data/fmt_test200
```

`--phase2` now produces the full Phase 2 distribution: unknown scale in [8,12], **relative rotation
±5° between reference and wide** (this was added recently — the wide is genuinely rotated relative
to the reference now, not just a shared tilt), plus the full aberration suite. The manifest
(`data/fmt_test200/labels.csv`) has ground-truth `scale`, `rotation_deg`, `gt_x`, `gt_y`, and
`present` columns per pair.

## What to measure (be rigorous, report numbers)

Write a validation script (`scripts/eval_fmt.py`) that, on the 200-pair set above, for each
**present** pair compares:

1. **Your FMT estimator** `estimate_scale_rotation(ref, wide)` vs. ground truth `scale` /
   `rotation_deg`.
2. **The current grid search** `solve.locate(ref, wide, return_info=True,
   scales=solve.PHASE2_SCALES, angles=solve.PHASE2_ANGLES)` -> `info['scale']`, `info['theta']`,
   vs. the same ground truth.

Report, for both methods, on present pairs where localization succeeded (so you're comparing pose
accuracy on the same solvable pairs):
- **scale**: median % error, and fraction within the scoring tiers (≤1% → full, ≤2%, ≤5%).
- **rotation**: median absolute error in degrees, and fraction within tiers (≤0.25° → full, ≤0.5°,
  ≤1.0°).
- **runtime**: seconds per pair for each method (FMT should be much faster than the grid — that's
  part of its appeal; confirm it).

Then answer plainly: **does FMT beat the grid search on scale/rotation accuracy, tie it, or lose?**
Include a per-pair CSV so the result is auditable. A tie or a loss is a perfectly good outcome to
report — we need to know either way before deciding whether to integrate it.

## Stretch (only if the above is done and FMT looks promising)

- Try FMT as a **coarse initializer** feeding our existing golden-section refinement (FMT gets you
  close in one shot, refinement polishes to the tight tiers) — often the best of both: FMT's speed
  + the grid's precision. Measure whether that hybrid beats either alone.

## What NOT to touch (so our work doesn't collide — this is important)

Aryan is separately retraining the neural net on the same branch's generator. To avoid merge
conflicts, your task is **isolated to new files**:
- **DO** create `fmt_pose.py` and `scripts/eval_fmt.py` (new files).
- **DO NOT** edit `solve.py`, `route.py`, `driftmatch/` (anything), `generate_dataset.py`,
  `driftsense/`, or any checkpoint. You may *import and call* `solve` for the head-to-head
  comparison, but do not modify it. Integration into `route.py` will be done later, once both your
  result and the retrain have landed — that keeps us from editing the same lines at the same time.
- **DO NOT** commit generated datasets (`data/fmt_test200/` etc.) — they're large and
  regeneratable from the seed. Only commit `fmt_pose.py`, `scripts/eval_fmt.py`, and your findings
  written into the docs.

## Report back

Add a dated section to [`docs/PHASE2_RESEARCH_NOTES.md`](PHASE2_RESEARCH_NOTES.md) in the same style
as the existing entries — the measured numbers (scale/rotation accuracy vs the grid, runtime), the
sign convention you found, and your plain verdict on whether FMT is worth integrating. Then commit
your new files + the doc update and push.

## Commit / push rules (strict, non-negotiable)

- Commit as **Aryan Chourasia <achourasia_be24@thapar.edu>** (should already be the git identity on
  the clone; verify with `git config user.email`). Do not add yourself or anyone else as a
  co-author or contributor.
- **No co-author or trailer lines of any kind.** Everything committed — commit messages, code
  comments, docs — must read as your own hand-written work. Scan your changes and strip any
  auto-generated attribution or trailer lines before pushing.
- Check `git status` before committing and stage files explicitly (`git add fmt_pose.py
  scripts/eval_fmt.py docs/PHASE2_RESEARCH_NOTES.md`) — do NOT `git add -A` (it would sweep in the
  large generated datasets).
