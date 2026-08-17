r"""Verify the rotation-search branch actually helps.

Our own pairs have no tilt between reference and wide, so the eval200 score
cannot show whether rotation search works -- it is insurance for the
evaluator's "turn one image 3 deg" case. Here we manufacture that case: take
plain pairs, rotate the WIDE image by a few degrees about its centre, and check
that locate() with the angle search recovers the site far better than with the
search switched off.

    .\.venv\Scripts\python.exe scripts\verify_rotation.py
"""
from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import affine_transform

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from solve import locate


def load(p):
    return np.asarray(Image.open(p).convert("L"))


def rotate_about_site(img, deg, cx, cy):
    """Rotate the image about the site (cx, cy), keeping that point fixed.

    Rotating about the image centre would move an off-centre site, invalidating
    the ground-truth label; rotating about the site itself leaves the label
    exactly where it was, which is the only fair way to test the angle search.
    """
    th = np.radians(deg)
    c, s = np.cos(th), np.sin(th)
    M = np.array([[c, s], [-s, c]])          # output->input, in (row, col)
    centre = np.array([cy, cx])
    return affine_transform(img, M, offset=centre - M @ centre,
                            order=1, mode="nearest")


def main():
    root = pathlib.Path("data/eval200")
    rows = [r for r in csv.DictReader((root / "labels.csv").open())
            if float(r.get("coarse_period_nm", 0) or 0) == 0]   # plain only
    rows = rows[:20]

    for tilt in (2.0, 3.0, 5.0):
        with_err, without_err = [], []
        for row in rows:
            ref = load(root / row["ref_path"])
            wide = load(root / row["wide_path"])
            gx, gy = float(row["gt_x"]), float(row["gt_y"])
            # rotate the wide about the true site so the label stays put
            wr = rotate_about_site(wide, tilt, gx, gy)

            # with the angle search
            px, py = locate(ref, wr)
            with_err.append(np.hypot(px - gx, py - gy))
            # with rotation switched off (angles = {0}) -- the Phase-1 behaviour
            px0, py0 = locate(ref, wr, angles=(0.0,))
            without_err.append(np.hypot(px0 - gx, py0 - gy))

        with_err = np.array(with_err)
        without_err = np.array(without_err)
        print(f"wide tilted {tilt:.0f} deg  (n={len(rows)} plain pairs)")
        print(f"  angle search ON : median {np.median(with_err):6.1f} px   "
              f"within 5px {100*np.mean(with_err <= 5):5.1f} %")
        print(f"  angle search OFF: median {np.median(without_err):6.1f} px   "
              f"within 5px {100*np.mean(without_err <= 5):5.1f} %\n")


if __name__ == "__main__":
    main()
