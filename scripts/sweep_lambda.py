r"""Sweep the centre-prior strength LAMBDA in a single pass over the images.

The expensive part (coarse+fine correlation) is computed once per pair via
fine_score(); only the cheap peak selection is repeated per LAMBDA. This maps
the plain-vs-multi trade-off directly so the operating point is chosen from
evidence, not guessed.

    .\.venv\Scripts\python.exe scripts\sweep_lambda.py data\eval200
"""
from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from solve import fine_score, _select_peak

LAMBDAS = [0.0, 0.03, 0.05, 0.08, 0.12, 0.20, 0.35]


def load(p):
    return np.asarray(Image.open(p).convert("L"))


def main():
    root = pathlib.Path(sys.argv[1])
    rows = list(csv.DictReader((root / "labels.csv").open()))

    # err[lam] -> list, split by group
    errs = {lam: {"all": [], "plain": [], "multi": []} for lam in LAMBDAS}
    for row in rows:
        ref = load(root / row["ref_path"])
        wide = load(root / row["wide_path"])
        gx, gy = float(row["gt_x"]), float(row["gt_y"])
        is_multi = float(row.get("coarse_period_nm", 0) or 0) > 0

        score, h, w, n_px = fine_score(ref, wide)
        for lam in LAMBDAS:
            r, c = _select_peak(score, h, w, n_px, center_rule=True, lam=lam)
            e = float(np.hypot((c + w / 2.0) - gx, (r + h / 2.0) - gy))
            errs[lam]["all"].append(e)
            errs[lam]["plain" if not is_multi else "multi"].append(e)

    def hit(lst):
        a = np.array(lst)
        return 100.0 * np.mean(a <= 5) if len(a) else float("nan")

    print(f"=== LAMBDA sweep on {root} (within 5 px %) ===")
    print(f"{'LAMBDA':>7} {'ALL':>7} {'plain':>7} {'multi':>7}")
    for lam in LAMBDAS:
        print(f"{lam:>7.2f} {hit(errs[lam]['all']):>7.1f} "
              f"{hit(errs[lam]['plain']):>7.1f} {hit(errs[lam]['multi']):>7.1f}")


if __name__ == "__main__":
    main()
