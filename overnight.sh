#!/bin/bash
# Autonomous overnight: bigger dataset + long training + eval on their generator.
# Everything logged to overnight_log.txt. Runs unattended.
set +e
VENV="D:/semicon/.venv/Scripts/python.exe"
PY312="C:/Users/ARYAN/AppData/Local/Programs/Python/Python312/python.exe"
DS="D:/Temp/claude/D--semicon/554e4c23-4b58-4c48-b4b1-2846205aea1a/scratchpad/ds_ref"
XEVAL="D:/Temp/claude/D--semicon/554e4c23-4b58-4c48-b4b1-2846205aea1a/scratchpad/xdomain_eval.py"
CONV="D:/Temp/claude/D--semicon/554e4c23-4b58-4c48-b4b1-2846205aea1a/scratchpad/convert_manifest.py"
LOG="D:/semicon/overnight_log.txt"

echo "=== OVERNIGHT START $(date) ===" > "$LOG"
powershell -c "Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force" 2>/dev/null
sleep 3
"$PY312" -m pip install --quiet opencv-python-headless >> "$LOG" 2>&1
echo "[env] cv2 ready for py312" >> "$LOG"

echo "[1/4 GEN] 4000 randomized pairs (heavy-noise coverage)  $(date)" >> "$LOG"
cd "$DS" || exit 1
"$VENV" gen_random.py --num-samples 4000 --split train --seed 500 >> "$LOG" 2>&1
"$VENV" "$CONV" "$DS/rrdata/train" >> "$LOG" 2>&1
rm -f "$DS/rrdata/train/_wide_cache.mmap" "$DS/rrdata/train/_ref_cache.mmap"
echo "[GEN] train pairs = $(($(wc -l < "$DS/rrdata/train/labels.csv")-1))  $(date)" >> "$LOG"

echo "[2/4 TRAIN] 35 epochs, memmap (crash-proof), best.pt checkpointed on the NOISY held-out  $(date)" >> "$LOG"
cd /d/semicon || exit 1
PYTHONUNBUFFERED=1 "$PY312" -u -m driftmatch.train --data "$DS/rrdata/train" \
    --epochs 35 --batch 4 --workers 0 \
    --eval1 "$DS/refdata/test" --eval2 "$DS/rrdata/heldout" >> "$LOG" 2>&1
echo "[TRAIN] done  $(date)" >> "$LOG"

echo "[3/4 EVAL DEFAULT] refdata/test (DriftFind vs DriftRoute vs ZNCC, acc@5px)  $(date)" >> "$LOG"
"$PY312" "$XEVAL" "$DS/refdata/test/manifest.csv" >> "$LOG" 2>&1
echo "[4/4 EVAL NOISY] rrdata/heldout  $(date)" >> "$LOG"
"$PY312" "$XEVAL" "$DS/rrdata/heldout/manifest.csv" >> "$LOG" 2>&1

echo "=== OVERNIGHT DONE $(date) ===" >> "$LOG"
echo "best.pt is the trained net; read this log top-to-bottom for the trajectory." >> "$LOG"
