r"""Evaluate DriftFind (solve.locate) on a dataset and report the score.

    .\.venv\Scripts\python.exe scripts\eval_solver.py data\eval200

Prints, over all pairs and split by plain vs multi-match:
  - median / mean pixel error (predicted centre vs ground truth)
  - hit-rate at 1, 2, 5 px thresholds  (the scored accuracy)
  - time per pair
It also lists the worst few misses so failures can be looked at, not guessed.
"""
from __future__ import annotations

import csv
import pathlib
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from solve import locate


def load(path):
    return np.asarray(Image.open(path).convert("L"))


def main():
    root = pathlib.Path(sys.argv[1])
    rows = list(csv.DictReader((root / "labels.csv").open()))

    errs, times, multi = [], [], []
    worst = []
    for row in rows:
        ref = load(root / row["ref_path"])
        wide = load(root / row["wide_path"])
        gx, gy = float(row["gt_x"]), float(row["gt_y"])

        t0 = time.perf_counter()
        px, py = locate(ref, wide)
        times.append(time.perf_counter() - t0)

        e = float(np.hypot(px - gx, py - gy))
        errs.append(e)
        # a pair is "multi-match" if the landmark was repeated on a lattice
        is_multi = float(row.get("coarse_period_nm", 0) or 0) > 0
        multi.append(is_multi)
        worst.append((e, row["pair_id"], is_multi, round(px, 1), round(py, 1),
                      round(gx, 1), round(gy, 1)))

    errs = np.array(errs)
    multi = np.array(multi, bool)

    def report(name, mask):
        if mask.sum() == 0:
            return
        e = errs[mask]
        print(f"\n{name}  (n={mask.sum()})")
        print(f"  median err {np.median(e):6.1f} px   mean {e.mean():6.1f} px")
        for thr in (1, 2, 5, 10, 25):
            print(f"    within {thr:2d} px : {100*np.mean(e <= thr):5.1f} %")

    print(f"=== DriftFind on {root}  ({len(errs)} pairs) ===")
    report("ALL", np.ones(len(errs), bool))
    report("plain (single match)", ~multi)
    report("multi-match (repeats)", multi)
    print(f"\ntime/pair: {1000*np.mean(times):.1f} ms")

    print("\nworst 8 misses (err, id, multi?, pred_x, pred_y, gt_x, gt_y):")
    for w in sorted(worst, reverse=True)[:8]:
        print("  ", w)


if __name__ == "__main__":
    main()
