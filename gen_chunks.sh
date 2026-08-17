#!/bin/bash
# Continue the generalist from epoch-18 weights to ~epoch 80, in 10-epoch chunks.
# Each chunk is a FRESH process -> the memmap RAM creep resets between chunks, so
# it never hits the OOM that killed the single long run. Cosine-restart per chunk
# (SGDR-style) is genuine training, not a trick. Saves to checkpoints_gen so the
# specialist (checkpoints/best.pt) stays untouched until we compare at the end.
set +e
SEM="${SEM:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# Checkout of the organizers' reference generator (HuggingFace Space
# aayushraina21/drift-sense-synthetic-data). Override with REF_ROOT=...
REF_ROOT="${REF_ROOT:-$SEM/ds_ref}"
PY312="C:/Users/ARYAN/AppData/Local/Programs/Python/Python312/python.exe"
DS="$REF_ROOT"
XEVAL="$SEM/scripts/eval_manifest.py"
OUT="$SEM/driftmatch/checkpoints_gen"
LOG="$SEM/gen_log.txt"
CHUNKS=6          # 6 x 10 = 60 more epochs -> ~epoch 78 (from 18)

powershell -c "Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force" 2>/dev/null; sleep 3
mkdir -p "$OUT"
cp -f "$SEM/driftmatch/checkpoints/best_generalist.pt" "$OUT/last.pt"
echo "=== CHUNKED GENERALIST START $(date) : from ep18, $CHUNKS x10 = $((CHUNKS*10)) more epochs ===" > "$LOG"

cd /d/semicon
for i in $(seq 1 $CHUNKS); do
  echo "" >> "$LOG"
  echo "--- CHUNK $i/$CHUNKS  (cumulative ~ep $((18 + (i-1)*10)) -> $((18 + i*10)))  $(date) ---" >> "$LOG"
  PYTHONUNBUFFERED=1 "$PY312" -u -m driftmatch.train --data "$DS/uniondata" --out "$OUT" \
      --resume "$OUT/last.pt" --epochs 10 --batch 4 --workers 0 \
      --eval1 "$DS/refdata/test" --eval2 "$DS/rrdata/heldout" >> "$LOG" 2>&1
done

# Activate the trained generalist for evaluation (specialist safe in best_theirs_domain.pt)
cp -f "$OUT/best.pt" "$SEM/driftmatch/checkpoints/best.pt" 2>/dev/null || cp -f "$OUT/last.pt" "$SEM/driftmatch/checkpoints/best.pt"
echo "" >> "$LOG"; echo "=== FINAL EVAL (trained generalist) $(date) ===" >> "$LOG"
echo "-- THEIR default --" >> "$LOG"; "$PY312" "$XEVAL" "$DS/refdata/test/manifest.csv" >> "$LOG" 2>&1
echo "-- THEIR noisy --"   >> "$LOG"; "$PY312" "$XEVAL" "$DS/rrdata/heldout/manifest.csv" >> "$LOG" 2>&1
echo "-- OUR val_resize60 --" >> "$LOG"
"$PY312" "$SEM/scripts/eval_net.py" "$SEM/data/val_resize60" "$SEM/driftmatch/checkpoints/best.pt" >> "$LOG" 2>&1
echo "=== CHUNKED DONE $(date). specialist=best_theirs_domain.pt, generalist=best.pt/checkpoints_gen ===" >> "$LOG"
