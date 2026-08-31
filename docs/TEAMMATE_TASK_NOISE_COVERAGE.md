# Task: add missing degradation categories (speckle, salt-and-pepper), then one retrain

Read [`docs/CONTINUE_HERE.md`](CONTINUE_HERE.md) first. `git pull` before starting.

## Why

Three independent sources all name **multiplicative speckle** and **salt-and-pepper (impulse)
noise** as standard SEM/detector degradation categories our generator does not model — grepped
`driftsense/physics.py`, confirmed absent:
- Applied Materials' own Phase 2 generator spec lists `add_speckle_noise`, `add_salt_and_pepper_noise`
  as reusable functions in their reference pipeline (`src/sem_imaging.py`).
- Two independent competitor repos (`TharunBabu-05/I4C_Drift_sense_D_RAM_submission`,
  `DK-A/Techtonics_Drift-Sense_Wafer_Inspection_PS2`) both explicitly model multiplicative speckle
  as a named degradation factor.

This is not "more data" (already proven to plateau — 8k pairs of the existing distribution gave
identical accuracy to the smaller set). This is **new physics categories the net has never seen**,
which is the one thing that measurably worked before (29 Aug: adding barrel/astigmatism/scan
distortion is what made the net actually win on the harder domain — see
`docs/PHASE2_RESEARCH_NOTES.md`). Same move, two new categories.

## Step 1 — add both to `driftsense/physics.py`

- **Speckle**: multiplicative noise, `I_out = I_in * (1 + n)`, `n ~ N(0, sigma^2)`, applied to the
  wide capture only (a detector/coherent-imaging effect, same spirit as the existing charging/scan
  functions which are wide-only). Gate behind a new parameter, default off, so Phase 1 stays
  byte-identical (verify: seed 7000 unaffected).
- **Salt-and-pepper**: isolated pixel-level impulse noise (dead/saturated detector pixels) —
  randomly flip a small fraction of pixels to 0 or 255. Also wide-only, gated off by default.
- **Citation framing** — match the existing honest style already used for astigmatism/barrel/vignette
  in `docs/GENERATOR_SPEC.md` section 8: these are *standard, textbook noise models* (cite Goldstein
  et al., *Scanning Electron Microscopy and X-Ray Microanalysis*, for the general SEM noise-source
  taxonomy, same reference already used elsewhere in the ledger), not SEM-specific measured
  parameters — say so plainly, don't oversell. Add a numbered entry to `docs/GENERATOR_SPEC.md`
  section 8 in the same format as entries 25-30.
- Wire into `generate_dataset.py`'s `--phase2` path (new CLI flags, e.g. `--speckle`,
  `--salt-pepper`, both included by default under `--phase2` like the existing aberrations) and
  `sampling.py`/manifest columns, following the exact pattern the barrel/vignette/gamma additions
  already used (gated params, new manifest columns, Phase 1 regression check).
- **Regression gate (mandatory, same as every prior generator change):** seed 7000, no `--phase2`
  → byte-identical output to before. curated30 C00 → `solve.locate` still returns `559.904, 470.001`.

## Step 2 — generate an enriched training set (moderate size, not another 8k)

```
python generate_dataset.py --phase2 --pairs 4000 --seed 970000 --workers 8 --out data/p2train_speckle4k
python generate_dataset.py --phase2 --pairs 100 --seed 980000 --workers 8 --out data/p2eval_speckle100
```

(Seeds 970000/980000 don't overlap any existing train/eval/calib/test seed.) Keep this smaller than
the 8k set — the goal is coverage of the two new categories, not more volume of what's already
plateaued; ~30-45 min is the expected budget, not 1.5 hours.

## Step 3 — one retrain, warm-started, same pattern as the rotation-aware retrain

```
python -m driftmatch.train --data data/p2train_speckle4k \
  --resume driftmatch/checkpoints/best_phase2_rot8k.pt \
  --epochs 10 --batch 4 --lr 1.5e-4 --workers 0 \
  --eval1 data/p2train_speckle4k --eval2 data/p2eval_speckle100 \
  --out driftmatch/checkpoints_speckle
```

## Step 4 — decisive comparison (reuse the existing script, do not write a new one)

```
python scripts/compare_checkpoints.py data/p2eval_speckle100 \
  driftmatch/checkpoints/best_phase2_rot8k.pt driftmatch/checkpoints_speckle/best.pt
```

**Decision rule, same discipline as every prior task — do not manufacture a winner:**
- New checkpoint wins by **≥5pp @5px** on the fresh eval → adopt it (`register.py`'s one-line
  checkpoint pointer), keep the old as fallback.
- Tie (within 5pp) → **keep the current checkpoint.** The generator enrichment (Step 1) still has
  standalone value for the generator/citations bucket even if the retrain doesn't move accuracy —
  report that plainly, it is not a failed task.
- New checkpoint worse by ≥5pp → keep the current checkpoint, report why if you can tell (e.g. too
  little speckle/S&P data to learn from at this pair count).

## What NOT to do

- Do NOT touch theta/scale/rotation code, `solve.py`'s search logic, or `route.py`'s `FOUND_PEAK` —
  unrelated to this task.
- Do NOT regenerate the 8k/300/100/150/300 sets from the other tasks — this uses its own new seeds.
- Do NOT attempt a new model architecture (e.g. a cross-encoder re-ranker, an idea from one
  competitor's video) — that is a materially different approach from our declared Phase 1 method and
  risks the no-appeal disqualification rule. Out of scope for this task and for now.
- Do NOT commit the generated datasets — gitignored, regeneratable from seeds (add the two new
  dirs to `.gitignore` if not already covered by the existing `data/p2*` pattern).

## Report back

Append a dated section to `docs/PHASE2_RESEARCH_NOTES.md`: what was added and why (with the
citation), the regression-gate result, the decisive comparison table, and your adopt/keep decision.
Commit under your own identity, no AI trailers, stage explicitly (no `git add -A`).
