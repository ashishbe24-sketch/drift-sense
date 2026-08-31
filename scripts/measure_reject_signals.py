"""Measure present-vs-absent separation of the rejection signals on a mixed set.

Reports, for the classical peak NCC and the two auxiliary signals `solve.locate`
already returns (`second_ratio`, `distinct`):
  - the present/absent value distributions (min / median / max),
  - the headline Set-C test: does the absent-peak MAX drop below the present-peak
    MEDIAN? (Before the generator fix it did not: absent max 0.967 > present
    median 0.933.)
  - a rank-sum AUC (Mann-Whitney U) of each signal against the present/absent
    label, oriented so higher = more present-like.

    <venv-python> scripts/measure_reject_signals.py data/p2calib300

Matches route.predict_full's call exactly (scales=PHASE2_SCALES,
angles=PHASE2_ANGLES). Reads-only.
"""
from __future__ import annotations

import csv
import pathlib
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import solve


def load(p):
    return np.asarray(Image.open(p).convert("L"))


def auc(scores, labels):
    """Rank-sum AUC that `scores` ranks label==1 above label==0 (higher=present)."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks within ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_pos = ranks[labels == 1].sum()
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def dist(name, vals):
    v = np.asarray(vals, float)
    return f"{name:8s} n={len(v):3d}  min {v.min():.4f}  median {np.median(v):.4f}  max {v.max():.4f}"


def main():
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data/p2calib300")
    rows = list(csv.DictReader((root / "labels.csv").open()))
    peak, sratio, distinct, lab = [], [], [], []
    t0 = time.perf_counter()
    for i, r in enumerate(rows):
        ref = load(root / r["ref_path"]); wide = load(root / r["wide_path"])
        _x, _y, info = solve.locate(ref, wide, return_info=True,
                                    scales=solve.PHASE2_SCALES,
                                    angles=solve.PHASE2_ANGLES)
        peak.append(info["score"]); sratio.append(info["second_ratio"])
        distinct.append(info["distinct"]); lab.append(1 if r["present"] == "1" else 0)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(rows)}  ({(time.perf_counter()-t0)/(i+1):.2f}s/pair)",
                  file=sys.stderr)
    peak, sratio, distinct, lab = map(np.array, (peak, sratio, distinct, lab))
    pres, absent = lab == 1, lab == 0
    print(f"\n=== {root}  ({pres.sum()} present, {absent.sum()} absent, "
          f"{(time.perf_counter()-t0)/len(rows):.2f}s/pair) ===")
    print("peak NCC")
    print("  " + dist("present", peak[pres]))
    print("  " + dist("absent ", peak[absent]))
    clean = peak[absent].max() < np.median(peak[pres])
    print(f"  >>> absent max ({peak[absent].max():.4f}) "
          f"{'<' if clean else '>='} present median ({np.median(peak[pres]):.4f})"
          f"  -> separation is {'CLEAN' if clean else 'NOT clean'}")
    print("\nAUC vs present/absent label (higher = more present-like; 1.0 = perfect):")
    print(f"  peak NCC        {auc(peak, lab):.4f}")
    print(f"  distinct        {auc(distinct, lab):.4f}")
    print(f"  1 - second_ratio {auc(-sratio, lab):.4f}")


if __name__ == "__main__":
    main()
