#!/usr/bin/env python
r"""DriftFind -- submission entry point for PS-02 (Solution 1).

The algorithm is the pure function `solve.locate(reference, wide) -> (x, y)`.
This file only wraps it for the evaluator: it can be called on a single pair or
over the organisers' CSV of (reference_path, wide_path[, gt_x, gt_y]).

Single pair -- prints "x y" (wide-image pixel centre of the reference's site):
    .\.venv\Scripts\python.exe predict.py data\eval200\images\01000000_ref.png \
                                           data\eval200\images\01000000_wide.png

Batch over a CSV (writes predictions and reports timing):
    .\.venv\Scripts\python.exe predict.py --csv data\eval200\labels.csv \
                                           --root data\eval200 --out preds.csv

The evaluator said they "will throw the function name" -- whatever they name it,
it forwards to solve.locate, so only this thin wrapper changes, never the core.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from solve import locate


def _load(path) -> np.ndarray:
    """Read a grayscale PNG as a 2-D uint8 array, asserting the stated size."""
    img = np.asarray(Image.open(path).convert("L"))
    assert img.shape == (1000, 1000), f"{path}: expected 1000x1000, got {img.shape}"
    return img


def locate_paths(ref_path, wide_path) -> tuple[float, float]:
    """Convenience: locate from two file paths. Returns (x, y)."""
    return locate(_load(ref_path), _load(wide_path))


def _run_batch(csv_path: pathlib.Path, root: pathlib.Path, out: pathlib.Path):
    rows = list(csv.DictReader(csv_path.open()))
    preds, times = [], []
    for row in rows:
        ref = root / row["ref_path"]
        wide = root / row["wide_path"]
        t0 = time.perf_counter()
        x, y = locate_paths(ref, wide)
        times.append(time.perf_counter() - t0)
        preds.append(dict(ref_path=row["ref_path"], wide_path=row["wide_path"],
                          pred_x=round(x, 3), pred_y=round(y, 3)))

    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ref_path", "wide_path", "pred_x", "pred_y"])
        w.writeheader()
        w.writerows(preds)

    t = np.array(times)
    print(f"{len(preds)} pairs -> {out}")
    print(f"time/pair: mean {1000*t.mean():.0f} ms  median {1000*np.median(t):.0f} ms  "
          f"max {1000*t.max():.0f} ms")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref", nargs="?", help="reference PNG (single-pair mode)")
    ap.add_argument("wide", nargs="?", help="wide PNG (single-pair mode)")
    ap.add_argument("--csv", type=pathlib.Path, help="CSV with ref_path,wide_path")
    ap.add_argument("--root", type=pathlib.Path, default=pathlib.Path("."),
                    help="directory the CSV paths are relative to")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("predictions.csv"),
                    help="where to write predictions (batch mode)")
    a = ap.parse_args()

    if a.csv:
        _run_batch(a.csv, a.root, a.out)
    elif a.ref and a.wide:
        x, y = locate_paths(a.ref, a.wide)
        print(f"{x:.3f} {y:.3f}")
    else:
        ap.error("give a reference and wide image, or --csv for batch mode")


if __name__ == "__main__":
    main()
