#!/bin/bash
# Union-of-domains training: our physics generator + their reference generator,
# so the net learns generator-INVARIANT features (robust to an unseen 3rd domain),
# not one generator's fingerprint. Preserves the current specialist checkpoint.
set +e
SEM="${SEM:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# Checkout of the organizers' reference generator (HuggingFace Space
# aayushraina21/drift-sense-synthetic-data). Override with REF_ROOT=...
REF_ROOT="${REF_ROOT:-$SEM/ds_ref}"
VENV="$SEM/.venv/Scripts/python.exe"
# Interpreter with torch installed. Override with PY312=... (or just PY312=python
# if torch lives in the default interpreter).
PY312="${PY312:-C:/Users/ARYAN/AppData/Local/Programs/Python/Python312/python.exe}"
DS="$REF_ROOT"
XEVAL="$SEM/scripts/eval_manifest.py"
THEIR="$DS/rrdata/train"
OURS="$DS/ourdata"
UNION="$DS/uniondata"
LOG="$SEM/union_log.txt"

echo "=== UNION START $(date) ===" > "$LOG"
powershell -c "Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force" 2>/dev/null; sleep 3

# Preserve the their-domain specialist net (95%/84%) before overwriting best.pt
cp -f "$SEM/driftmatch/checkpoints/best.pt" "$SEM/driftmatch/checkpoints/best_theirs_domain.pt" 2>/dev/null
echo "[preserve] saved specialist -> best_theirs_domain.pt" >> "$LOG"

echo "[1/4 GEN] our-domain pairs (2000, our physics renderer)  $(date)" >> "$LOG"
cd "$SEM" || exit 1
"$VENV" generate_dataset.py --seed 900000 --pairs 2000 --style mixed --out "$OURS" >> "$LOG" 2>&1

echo "[2/4 MERGE] their(4000) + ours(2000) -> union  $(date)" >> "$LOG"
"$VENV" - "$THEIR" "$OURS" "$UNION" >> "$LOG" 2>&1 <<'PYEOF'
import csv, sys, os
their, ours, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
rows = []
for root in (their, ours):
    with open(os.path.join(root, "labels.csv")) as f:
        for r in csv.DictReader(f):
            r["ref_path"] = os.path.abspath(os.path.join(root, r["ref_path"]))
            r["wide_path"] = os.path.abspath(os.path.join(root, r["wide_path"]))
            rows.append(r)
with open(os.path.join(out, "labels.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["ref_path", "wide_path", "gt_x", "gt_y"])
    w.writeheader()
    for r in rows:
        w.writerow({k: r[k] for k in ("ref_path", "wide_path", "gt_x", "gt_y")})
print("union rows:", len(rows))
PYEOF
rm -f "$UNION/_wide_cache.mmap" "$UNION/_ref_cache.mmap"

echo "[3/4 TRAIN] 30 epochs on UNION (memmap, best.pt on their-noisy)  $(date)" >> "$LOG"
PYTHONUNBUFFERED=1 "$PY312" -u -m driftmatch.train --data "$UNION" \
    --epochs 30 --batch 4 --workers 0 \
    --eval1 "$DS/refdata/test" --eval2 "$DS/rrdata/heldout" >> "$LOG" 2>&1

echo "[4/4 FINAL EVAL across domains]  $(date)" >> "$LOG"
echo "-- THEIR default --" >> "$LOG"; "$PY312" "$XEVAL" "$DS/refdata/test/manifest.csv" >> "$LOG" 2>&1
echo "-- THEIR noisy --"   >> "$LOG"; "$PY312" "$XEVAL" "$DS/rrdata/heldout/manifest.csv" >> "$LOG" 2>&1
echo "-- OUR held-out (val_resize60), net --" >> "$LOG"
"$PY312" "$SEM/scripts/eval_net.py" "$SEM/data/val_resize60" "$SEM/driftmatch/checkpoints/best.pt" >> "$LOG" 2>&1
echo "=== UNION DONE $(date) ===" >> "$LOG"
echo "best.pt = generalist (union); best_theirs_domain.pt = specialist. Compare in log." >> "$LOG"
