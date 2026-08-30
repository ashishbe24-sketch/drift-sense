"""Recalibrate route.FOUND_PEAK on a large present/absent calibration set.

The rejection flag `found` thresholds the classical peak NCC (`info['score']`
from `solve.locate`, exactly as `route.predict_full` does). The current 0.68 was
tuned on only 60 pairs, where one absent pair (C09) cleared the threshold by a
0.006 margin -- a thin, provisional operating point (see route.py's FOUND_PEAK
comment and docs/PHASE2_RESEARCH_NOTES.md). This sweeps the threshold on a much
larger set for a statistically firmer value.

Cost, not raw F1, is minimized: a false-reject on a present pair also zeros that
pair's localization + pose credit, so it is weighed 2x a false-accept -- the same
reasoning route.py's comment records. Plain F1 and the per-class counts are
printed alongside so the tradeoff is visible.

    <venv-python> scripts/recalibrate_found.py data/p2calib300

Matches route.predict_full's call exactly: scales=PHASE2_SCALES,
angles=PHASE2_ANGLES. Reads-only -- it prints the recommended threshold and does
NOT edit route.py.
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
import route


def load(p):
    return np.asarray(Image.open(p).convert("L"))


def f1(pred, labs):
    # positive class = present (found==1), matching the localization-credit logic
    tp = int(((pred == 1) & (labs == 1)).sum())
    fp = int(((pred == 1) & (labs == 0)).sum())
    fn = int(((pred == 0) & (labs == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return (2 * prec * rec / (prec + rec)) if prec + rec else 0.0, prec, rec


def main():
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data/p2calib300")
    rows = list(csv.DictReader((root / "labels.csv").open()))
    peaks, labs = [], []
    t0 = time.perf_counter()
    for i, r in enumerate(rows):
        ref = load(root / r["ref_path"])
        wide = load(root / r["wide_path"])
        _x, _y, info = solve.locate(ref, wide, return_info=True,
                                    scales=solve.PHASE2_SCALES,
                                    angles=solve.PHASE2_ANGLES)
        peaks.append(info["score"])
        labs.append(1 if r["present"] == "1" else 0)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(rows)} pairs "
                  f"({(time.perf_counter()-t0)/(i+1):.2f}s/pair)", file=sys.stderr)
    peaks, labs = np.array(peaks), np.array(labs)
    n_pres, n_abs = int((labs == 1).sum()), int((labs == 0).sum())
    print(f"\n{len(rows)} pairs: {n_pres} present, {n_abs} absent  "
          f"({(time.perf_counter()-t0)/len(rows):.2f}s/pair)")
    print(f"peak NCC  present: min {peaks[labs==1].min():.4f} "
          f"med {np.median(peaks[labs==1]):.4f}   "
          f"absent: max {peaks[labs==0].max():.4f} "
          f"med {np.median(peaks[labs==0]):.4f}")

    # cost sweep: false-reject weight 2x false-accept (route.py's reasoning)
    best_cost = None
    best_f1 = None
    for t in np.round(np.arange(0.40, 0.851, 0.01), 2):
        pred = (peaks >= t).astype(int)
        fn = int(((pred == 0) & (labs == 1)).sum())
        fp = int(((pred == 1) & (labs == 0)).sum())
        cost = fn * 2 + fp
        f, prec, rec = f1(pred, labs)
        if best_cost is None or cost < best_cost[0]:
            best_cost = (cost, float(t), fn, fp, f)
        if best_f1 is None or f > best_f1[0]:
            best_f1 = (f, float(t), fn, fp, cost)

    print(f"\ncurrent route.FOUND_PEAK = {route.FOUND_PEAK}")
    cpred = (peaks >= route.FOUND_PEAK).astype(int)
    cf, cprec, crec = f1(cpred, labs)
    cfn = int(((cpred == 0) & (labs == 1)).sum())
    cfp = int(((cpred == 1) & (labs == 0)).sum())
    print(f"  at {route.FOUND_PEAK}:  F1 {cf:.4f}  prec {cprec:.3f}  rec {crec:.3f}  "
          f"FN {cfn}  FP {cfp}  cost {cfn*2+cfp}")

    cost, t, fn, fp, f = best_cost
    print(f"\ncost-optimal threshold (2x FN weight): {t:.2f}  "
          f"cost {cost}  FN {fn}  FP {fp}  F1 {f:.4f}")
    f, t2, fn2, fp2, cost2 = best_f1
    print(f"plain-F1-optimal threshold:            {t2:.2f}  "
          f"F1 {f:.4f}  FN {fn2}  FP {fp2}  cost {cost2}")


if __name__ == "__main__":
    main()
