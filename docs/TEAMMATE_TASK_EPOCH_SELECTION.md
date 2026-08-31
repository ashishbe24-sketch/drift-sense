# Task: verify the retrain's epoch selection (DriftSense Phase 2) — do this next

Read [`docs/CONTINUE_HERE.md`](CONTINUE_HERE.md) first, then this file. **Do exactly these two
things and nothing else** — do NOT start the Fourier-Mellin task (Aryan is handling that
separately), and do NOT change the rejection threshold. Scope discipline matters here.

Start by syncing: `git pull` (you should already be at commit `7d9089e`, but pull anyway).

---

## Background — why this task exists

Your retrain kept the checkpoint from **epoch 2** because held-out accuracy "peaked" there and then
declined while training accuracy kept climbing (a mild-overfit signature). That reasoning is sound
**in principle**, but the "peak" was measured on the training loop's **quick-eval of only ~60
pairs** — small enough that "epoch 2 was best" could be a noise artifact rather than a real peak.
Loss fell monotonically all the way to epoch 15, so it is entirely possible a later epoch is
actually better and the 60-pair curve just wobbled.

The job: **decide, on a proper sample size, whether epoch 2 really is the best checkpoint, or
whether a later epoch beats it.** This is a *measurement correctness* fix — we are not tuning
anything, we are checking that the checkpoint we shipped was selected on a trustworthy signal. No
new training data is needed (this is NOT a "generate more data" task — the symptom is overfitting,
not data starvation, so more data is the wrong lever).

Checkpoint name map (so the commands below are unambiguous):
- **old net** (pre-rotation) = `driftmatch/checkpoints/best_phase2.pt`
- **new net, epoch 2** (currently shipped) = `driftmatch/checkpoints/best_phase2_rot8k.pt`
- **new net, epoch 15** (final) = `driftmatch/checkpoints_new/last.pt` *(if your run saved it —
  verify in Step 1)*

---

## Step 1 — inventory what the training run left on disk (cheap)

```
ls driftmatch/checkpoints_new/
```

Note which of these exist: `best.pt` (= epoch 2, same as `best_phase2_rot8k.pt`), `last.pt`
(= epoch 15), and whether any per-epoch files (`epoch_3.pt`, etc.) were saved. Most likely you only
have **best (epoch 2)** and **last (epoch 15)** — that is enough for the cheap primary check below.

---

## Step 2 — proper head-to-head on a REAL sample size (no retrain, do this first)

Evaluate all the checkpoints you have on the **full 100-pair** `data/p2eval100` (not the 60-pair
quick-eval subset), through the shipping pipeline, using the exact-rubric scorer that is now in the
repo (`scripts/score_phase2.py`, it takes `--ckpt`):

```
python scripts/score_phase2.py data/p2eval100 --ckpt driftmatch/checkpoints/best_phase2.pt
python scripts/score_phase2.py data/p2eval100 --ckpt driftmatch/checkpoints/best_phase2_rot8k.pt
python scripts/score_phase2.py data/p2eval100 --ckpt driftmatch/checkpoints_new/last.pt
```

Record the **Localization line (@5px and the /40 credit)** for each — that is the number the net
affects. (Pose/rejection come from the classical path and are checkpoint-independent, so ignore
them for this comparison.)

If `data/p2eval100` was deleted, regenerate it first — it is deterministic:
```
python generate_dataset.py --phase2 --pairs 100 --seed 910000 --workers 8 --out data/p2eval100
```

---

## Step 3 — get an HONEST final number on a FRESH set (this is important)

Whichever checkpoint looks best in Step 2, you **selected it on `p2eval100`**, so `p2eval100` is now
a *selection* set — reporting its number would be optimistic. Generate a **fresh, unused** test set
and report the chosen checkpoint's number there:

```
python generate_dataset.py --phase2 --pairs 150 --seed 920000 --workers 8 --out data/p2test150
python scripts/score_phase2.py data/p2test150 --ckpt <the checkpoint you chose>
```

(Seed 920000 does not overlap the training seeds 900000–908000, `p2eval100` at 910000, or
`p2calib300` at 950000 — so it is genuinely held out.)

---

## Step 4 (OPTIONAL, only if Step 2 is inconclusive AND you want the full curve)

If epoch 2 and epoch 15 are within noise of each other and you want to know whether some *middle*
epoch (3–14) is genuinely better, re-run the fine-tune and save every epoch, then rank them all on
`p2eval100`.

**Do it as ONE continuous 15-epoch run** (so the learning-rate schedule and optimizer state are
correct) with per-epoch checkpoints — do **not** approximate it by fifteen separate 1-epoch
`--resume` calls, which would reset the schedule and give different, misleading results. The exact
retrain command was:

```
python -m driftmatch.train --data data/p2train8k --resume driftmatch/checkpoints/best_phase2.pt \
  --epochs 15 --batch 4 --lr 2e-4 --workers 0 --eval1 data/p2train8k --eval2 data/p2eval100 \
  --out driftmatch/checkpoints_epochs
```

If `train.py` does not already write a checkpoint per epoch, add the smallest possible change to
make it do so (save `epoch_{n}.pt` each epoch) — do not change anything else in `train.py`. Then
score each with `scripts/score_phase2.py data/p2eval100 --ckpt driftmatch/checkpoints_epochs/epoch_N.pt`
and pick the real best, then confirm it on `p2test150` per Step 3.

This step costs ~3 GPU-hours (~11.5 min/epoch). **Skip it** if Step 2 already gives a clear answer —
it usually will.

---

## Decision rule (explicit — so there is no judgment-call inconsistency)

On `p2test150` (n=150, so binomial noise is roughly ±4 pp), a difference counts as **real only if it
is ≥ 5 percentage points @5px**. Otherwise treat it as a tie.

- **A later epoch beats epoch 2 by ≥5 pp on the fresh set** → switch `register.py` to that
  checkpoint (one-line change, the same line that currently selects `best_phase2_rot8k.pt`).
- **Everything is within 5 pp (a tie)** → **keep epoch 2 as-is.** The current shipped checkpoint
  stands; the retrain is marginal-but-principled and we move on. Do not manufacture a winner.
- **The OLD `best_phase2.pt` somehow beats both by ≥5 pp** → flag it loudly in your report (this
  would be surprising, since the new net is trained on the correct rotation distribution) and leave
  `register.py` on whatever actually scores best.

Whatever you decide, the point is that it is now decided on 100–150 pairs, not 60.

---

## The other item: leave `FOUND_PEAK` alone

**Do not touch the rejection threshold (`FOUND_PEAK = 0.68` in `route.py`).** Your own 300-pair
recalibration already settled this: F1 sits within 0.006 across the whole plausible threshold range,
and the absent-pair max (0.967) exceeds the present-pair median (0.933), so a single cutoff cannot
separate the classes any better — it is a signal-separability ceiling, not a tuning problem. Re-
sweeping it would be chasing noise. It stays at 0.68. (The real fix would be a multi-signal
rejection rule, which is a separate, later piece of work and not part of this task.)

---

## Report back

Append a dated section to [`docs/PHASE2_RESEARCH_NOTES.md`](PHASE2_RESEARCH_NOTES.md), same style as
the existing entries, with:
- the Step 2 table (old / epoch 2 / epoch 15 localization @5px on the full `p2eval100`),
- the Step 3 fresh-set number for the chosen checkpoint,
- your decision and which `register.py` line it corresponds to,
- (if you did Step 4) the per-epoch curve and the real best epoch.

## What NOT to do

- **Do NOT start Fourier-Mellin** — Aryan is handling that separately and has not released that task
  to you yet.
- **Do NOT change `FOUND_PEAK`** or any rejection logic.
- **Do NOT generate more training data or retrain from scratch** — this task is about selecting the
  right epoch from the existing run, not producing a new one (Step 4's optional re-run reuses the
  same `p2train8k`).
- **Do NOT commit the generated datasets** (`p2eval100`, `p2test150`, etc.) — they are large and
  regeneratable from their seeds; they should stay gitignored.

## Commit / push rules (strict — keep the history clean and consistent)

- Commit under your own git identity, same as your previous commit `7d9089e` — no change needed
  there.
- **No co-author or trailer lines of any kind.** Everything committed — commit message, code
  comments, docs — must read as your own hand-written work. Scan your diff and strip any
  auto-generated attribution or trailer lines before pushing.
- `git status` before committing; stage files explicitly (`git add register.py
  docs/PHASE2_RESEARCH_NOTES.md` and, only if Step 4 required it, `driftmatch/train.py` and the new
  checkpoint) — do NOT `git add -A` (it would sweep in the large datasets).
- `git pull --rebase` right before you push, in case anything else landed on `main` meanwhile.
