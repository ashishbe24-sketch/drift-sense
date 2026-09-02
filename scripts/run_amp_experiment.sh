#!/usr/bin/env bash
# Overnight orchestration for the AMP-generator retrain experiment.
#
# Waits for generation to finish, merges the batches into one training view,
# fine-tunes from the shipped checkpoint, then evaluates every configuration.
# ISOLATED: writes only to data/amp_* and driftmatch/checkpoints_amp/. Never
# touches route.py / register.py / solve.py / the shipped checkpoints, and
# never pushes.
#
#   bash scripts/run_amp_experiment.sh
set -u

PY312="C:/Users/ARYAN/AppData/Local/Programs/Python/Python312/python.exe"
VENV=".venv/Scripts/python.exe"
LOG="amp_experiment_log.txt"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== AMP experiment start ==="

# ---------------------------------------------------------------- 1. wait
say "waiting for generation (amp_train2 -> 2000, amp_holdout -> 250)"
# Primary signal: the holdout set (generated last) reaching its target.
# Fallback: a wall-clock cap, so a stalled generator cannot hang the night.
# Deliberately NOT using pgrep -- it is absent in some Git Bash installs, and a
# missing command would make the "no generator running" test fire immediately.
WAITED=0
MAX_WAIT=5400          # 90 min
while true; do
  n2=$(( $(wc -l < data/amp_train2/labels.csv 2>/dev/null || echo 1) - 1 ))
  nh=$(( $(wc -l < data/amp_holdout/labels.csv 2>/dev/null || echo 1) - 1 ))
  if [ "$nh" -ge 240 ]; then say "generation complete (train2=$n2, holdout=$nh)"; break; fi
  if [ "$WAITED" -ge "$MAX_WAIT" ]; then
    say "wait cap reached (train2=$n2, holdout=$nh) -- proceeding with what exists"
    break
  fi
  sleep 60
  WAITED=$((WAITED + 60))
done

n1=$(( $(wc -l < data/amp_train/labels.csv 2>/dev/null || echo 1) - 1 ))
n2=$(( $(wc -l < data/amp_train2/labels.csv 2>/dev/null || echo 1) - 1 ))
nh=$(( $(wc -l < data/amp_holdout/labels.csv 2>/dev/null || echo 1) - 1 ))
say "pairs: train=$n1 train2=$n2 holdout=$nh"

# ------------------------------------------------------- 2. merged view
# driftmatch/data.py resolves ref_path/wide_path relative to the dataset root,
# so a merged labels.csv using ../ paths avoids duplicating ~3 GB of PNGs.
say "building merged training view data/amp_all"
mkdir -p data/amp_all
"$VENV" - <<'PYEOF'
import csv, pathlib
out = pathlib.Path("data/amp_all"); out.mkdir(exist_ok=True)
rows, cols = [], None
for src in ("amp_train", "amp_train2"):
    p = pathlib.Path("data") / src / "labels.csv"
    if not p.exists():
        continue
    for r in csv.DictReader(p.open()):
        cols = cols or list(r.keys())
        r = dict(r)
        r["ref_path"] = f"../{src}/{r['ref_path']}"
        r["wide_path"] = f"../{src}/{r['wide_path']}"
        r["pair_id"] = f"{src}_{r['pair_id']}"
        rows.append(r)
with (out / "labels.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
print(f"merged {len(rows)} pairs -> data/amp_all/labels.csv")
PYEOF

# ------------------------------------------------------------- 3. train
say "fine-tuning from best_phase2_speckle.pt (checkpoint selected on amp_holdout)"
"$PY312" -m driftmatch.train \
  --data data/amp_all \
  --resume driftmatch/checkpoints/best_phase2_speckle.pt \
  --epochs 12 --batch 4 --lr 1.5e-4 --workers 0 \
  --eval1 data/amp_holdout --eval2 data/amp_holdout \
  --out driftmatch/checkpoints_amp >> "$LOG" 2>&1
say "training exit=$?"

NEW="driftmatch/checkpoints_amp/best.pt"
if [ ! -f "$NEW" ]; then
  say "NO new checkpoint produced -- stopping before evaluation"
  exit 1
fi

# ---------------------------------------------------------- 4. evaluate
# Order matters: the held-out AMP set and our own p2eval100 come FIRST (those
# are legitimate selection/forgetting checks); the organizers' 20 pairs are
# scored LAST, once, with the checkpoint already fixed.
say "--- our own p2eval100 (forgetting check) ---"
"$PY312" scripts/compare_checkpoints.py data/p2eval100 \
  driftmatch/checkpoints/best_phase2_speckle.pt "$NEW" >> "$LOG" 2>&1

say "--- organizer 20 pairs: shipped classical (baseline) ---"
"$PY312" scripts/eval_organizer.py --quiet >> "$LOG" 2>&1

say "--- organizer 20 pairs: OLD net for x,y ---"
"$PY312" scripts/eval_organizer.py --quiet \
  --ckpt driftmatch/checkpoints/best_phase2_speckle.pt --use-net-xy >> "$LOG" 2>&1

say "--- organizer 20 pairs: NEW AMP-trained net for x,y ---"
"$PY312" scripts/eval_organizer.py --quiet --ckpt "$NEW" --use-net-xy >> "$LOG" 2>&1

say "=== AMP experiment done -- see $LOG ==="
