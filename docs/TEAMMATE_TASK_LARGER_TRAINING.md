# Task: bigger training set + retrain + recalibrate on it

Read [`docs/CONTINUE_HERE.md`](CONTINUE_HERE.md) first for full context, then this file for your
specific piece. Everything below assumes you're in the repo root with the venv set up
(`python -m venv .venv`, then `pip install -r requirements.txt` — needs `torch` for this task,
CUDA build recommended if you have a GPU, CPU build also works, just slower).

> **IMPORTANT — pull the latest first (`git pull`) before you generate any data.** The generator
> was upgraded after this doc was first written: **`--phase2` now renders the wide image rotated
> relative to the reference (±5°), not just a shared tilt.** This is the single most important
> reason this retrain matters now. The current shipped net (`best_phase2.pt`) was trained *before*
> that change, so **it has never seen a rotated pair** — which is why accuracy drops on Phase 2's
> ±5° rotation. Because `--phase2` now bakes rotation into the training data automatically, this
> retrain will produce the first genuinely rotation-aware net. That is the accuracy win we're after,
> so treat it as the headline goal of this task, not a side effect.
>
> Two consequences for the steps below:
> 1. Your eval set (step 3) **must also be `--phase2`** so it contains rotation — otherwise you'll
>    be measuring on rotation-free data and won't see the actual improvement. The commands below
>    already use `--phase2` for eval; keep it that way.
> 2. When you report back (step 6), **measure rotation (`theta`) recovery accuracy too**, not just
>    localization — compare recovered `theta` against the manifest's `rotation_deg` on well-localized
>    pairs (median abs error in degrees, fraction within ≤0.25°/0.5°/1.0° tiers). `theta` still
>    comes from the classical path, so this checks the whole pipeline, not just the net.

## Why this task

The current model (`driftmatch/checkpoints/best_phase2.pt`) was trained overnight on a
RAM-capped, time-constrained run: 4000 training pairs, only 2500 of them actually cached per run
due to a 16GB RAM limit, 12 epochs. It works (89% @5px on our hardest 116-pair test), but two
things are under-powered and would benefit from more scale:

1. **The training set is smaller than it should be** — more data, more epochs, should improve the
   net further, especially if you have more RAM/GPU memory than the box this was trained on.
2. **The rejection threshold (`FOUND_PEAK` in `route.py`) was calibrated on only 60 pairs** — the
   research notes flag this as a real risk ("C09 clears the threshold by only 0.006 margin, not a
   robust one"). A much larger calibration set would firm this up significantly.

## Steps

### 1. Generate a bigger training set (present pairs, full aberration suite)

```bash
python generate_dataset.py --phase2 --pairs 8000 --seed 900000 --workers 5 --out data/p2train8k
```

Expect ~35-70 minutes depending on your CPU core count (`--workers` = your core count minus 2 is
a good default if you don't set it explicitly). This is deterministic — same seed always gives the
same 8000 pairs, so it's reproducible.

### 2. Generate a much larger calibration set (present + absent mixed)

```bash
python generate_dataset.py --phase2 --absent-fraction 0.22 --pairs 300 --seed 950000 --workers 5 --out data/p2calib300
```

~22% absent (matches the real Set C proportion). This is the set to recalibrate `FOUND_PEAK`
against — 300 pairs instead of 60 gives a far more statistically solid threshold.

### 3. Fine-tune the net on the bigger set, resuming from the current checkpoint

```bash
python -m driftmatch.train \
    --data data/p2train8k --resume driftmatch/checkpoints/best_phase2.pt \
    --epochs 15 --batch 4 --lr 2e-4 --workers 0 \
    --eval1 data/p2train8k --eval2 data/p2train8k \
    --out driftmatch/checkpoints_new
```

Notes:
- `--batch 4` if you have >4GB VRAM (the box this ran on had a 4GB card and had to use batch 2 —
  use a bigger batch if you can, it'll train faster and possibly better).
- If you have a lot of RAM (16GB+), you can drop the `--limit` cap entirely (it's not used above,
  which is intentional — only add `--limit N` if you hit an out-of-memory crash, see
  `docs/PHASE2_RESEARCH_NOTES.md` for what happened when this box did).
- `--eval1`/`--eval2` here point at the training set itself as a placeholder — better if you split
  off ~100 held-out pairs from `p2train8k` first, or generate a small separate eval set the same
  way as step 1 with a different seed (e.g. seed 910000, 100 pairs) and point `--eval2` at that.
- This saves to `driftmatch/checkpoints_new/`, NOT overwriting `best_phase2.pt` — keep both so
  there's a fallback if the new one is somehow worse.

### 4. Recalibrate the rejection threshold on the 300-pair calibration set

Sweep peak-NCC threshold for cost-optimal F1 (see `route.py`'s `FOUND_PEAK` comment for the exact
methodology — false-rejects cost more than false-accepts because they also zero the localization
score, so don't just optimize plain F1). Rough script shape:

```python
import csv, numpy as np, pathlib
from PIL import Image
import solve   # from repo root

root = pathlib.Path('data/p2calib300')
rows = list(csv.DictReader((root/'labels.csv').open()))
peaks, labs = [], []
for r in rows:
    ref = np.asarray(Image.open(root/r['ref_path']).convert('L'))
    wide = np.asarray(Image.open(root/r['wide_path']).convert('L'))
    x, y, info = solve.locate(ref, wide, return_info=True, scales=solve.PHASE2_SCALES)
    peaks.append(info['score']); labs.append(1 if r['present']=='1' else 0)
peaks, labs = np.array(peaks), np.array(labs)
# sweep threshold minimizing weighted cost (false-reject weight 2x, matching route.py's reasoning)
best = None
for t in np.round(np.arange(0.40, 0.85, 0.01), 2):
    pred = (peaks >= t).astype(int)
    fn = ((pred==0)&(labs==1)).sum(); fp = ((pred==1)&(labs==0)).sum()
    cost = fn*2 + fp
    if best is None or cost < best[0]: best = (cost, t, fn, fp)
print('recalibrated threshold:', best)
```

Update `FOUND_PEAK` in `route.py` with the result if it differs meaningfully from the current
`0.68`.

### 5. Re-run the decisive comparison (classical vs net vs hybrid) on held-out data

Confirm the new checkpoint actually beats (or at least matches) the current 89% @5px /
1.6s-per-pair numbers before treating it as the new default. If it's worse, keep
`best_phase2.pt` as-is — bigger isn't automatically better, verify it.

### 6. Report back

Update `docs/PHASE2_RESEARCH_NOTES.md` with what you found (new numbers, whether the bigger
checkpoint replaced the old one, the recalibrated threshold) in the same style as the existing
entries — dated, with the actual measured numbers, not just "it worked."

## What NOT to do

- Don't touch rotation/`theta` — that's still blocked on the organizers' sample ground-truth data
  (see `docs/CONTINUE_HERE.md`), not part of this task.
- Don't commit/push without checking `git status` first for anything unintended (stray scratch
  files, huge datasets — the `data/p2train8k` etc. directories you create should probably stay
  local/gitignored rather than pushed, since they're regeneratable from the seed).
