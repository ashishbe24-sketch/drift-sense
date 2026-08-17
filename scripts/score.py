r"""Score a predictions CSV against ground truth -- the organisers' contract.

The webinar described their scoring utility as taking a CSV of reference path,
wide path and ground truth, and publishing the accuracy metric over pixel-error
thresholds. This is our stand-in with that same interface: it joins predictions
(from predict.py) to the labels on the wide-image path and reports the
hit-rate at 1..25 px, plus median/mean error. When their utility lands, it
drops in here unchanged upstream.

    .\.venv\Scripts\python.exe scripts\score.py preds.csv data\eval200\labels.csv
"""
from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np


def _index(rows, xk, yk):
    return {r["wide_path"]: (float(r[xk]), float(r[yk])) for r in rows}


def main():
    preds = list(csv.DictReader(pathlib.Path(sys.argv[1]).open()))
    labels = list(csv.DictReader(pathlib.Path(sys.argv[2]).open()))

    pred = _index(preds, "pred_x", "pred_y")
    truth = _index(labels, "gt_x", "gt_y")
    # optional multi-match flag from the labels, for a split report
    multi = {r["wide_path"]: float(r.get("coarse_period_nm", 0) or 0) > 0
             for r in labels}

    keys = [k for k in pred if k in truth]
    errs = np.array([np.hypot(pred[k][0] - truth[k][0],
                              pred[k][1] - truth[k][1]) for k in keys])
    is_multi = np.array([multi.get(k, False) for k in keys], bool)

    def report(name, mask):
        if mask.sum() == 0:
            return
        e = errs[mask]
        print(f"\n{name}  (n={int(mask.sum())})")
        print(f"  median {np.median(e):.2f} px   mean {e.mean():.2f} px")
        for thr in (1, 2, 5, 10, 25):
            print(f"    <= {thr:2d} px : {100*np.mean(e <= thr):5.1f} %")

    print(f"=== scored {len(keys)} pairs "
          f"({sys.argv[1]} vs {sys.argv[2]}) ===")
    report("ALL", np.ones(len(errs), bool))
    if is_multi.any() and (~is_multi).any():
        report("plain", ~is_multi)
        report("multi-match", is_multi)


if __name__ == "__main__":
    main()
