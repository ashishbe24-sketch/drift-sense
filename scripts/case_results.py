r"""Per-case DriftFind results for curated30, in case-id order.

Joins predictions to labels and prints one row per curated case with the
pixel error and a pass/fail at 5 px, so the results can be written back into
CASES.md next to each rationale.

    .\.venv\Scripts\python.exe scripts\case_results.py
"""
from __future__ import annotations

import csv
import pathlib
import re
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from solve import locate

ROOT = pathlib.Path("data/curated30")


def cid(path):
    return re.search(r"(C\d\d)_", path).group(1)


def main():
    labels = {cid(r["wide_path"]): r
              for r in csv.DictReader((ROOT / "labels.csv").open())}
    print(f"{'case':>4} {'gt_x':>8} {'gt_y':>8} {'pred_x':>8} {'pred_y':>8} "
          f"{'err_px':>7}  verdict")
    for c in sorted(labels):
        row = labels[c]
        ref = np.asarray(Image.open(ROOT / row["ref_path"]).convert("L"))
        wide = np.asarray(Image.open(ROOT / row["wide_path"]).convert("L"))
        gx, gy = float(row["gt_x"]), float(row["gt_y"])
        px, py = locate(ref, wide)
        e = float(np.hypot(px - gx, py - gy))
        verdict = "hit" if e <= 5 else ("MISS" if e > 25 else "near")
        print(f"{c:>4} {gx:>8.1f} {gy:>8.1f} {px:>8.1f} {py:>8.1f} "
              f"{e:>7.2f}  {verdict}")


if __name__ == "__main__":
    main()
