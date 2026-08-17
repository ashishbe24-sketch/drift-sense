"""Measure the resize shortcut.

In the organisers' pipeline the reference is a crop of the same canvas the wide
view is shrunk from, so simply down-scaling the reference by 10 should match the
wide view's target region almost exactly. In the physical pipeline the two views
are separate acquisitions, so that match is much weaker.

The size of this difference is the whole argument for training on the physical
path: a network trained on shared-canvas data can learn "find the patch whose
downsample matches" and will lose that crutch on anything else.
"""
import csv, pathlib
import numpy as np
from PIL import Image

N, RATIO, FOOT = 1000, 10, 100


def score(root: pathlib.Path):
    rows = list(csv.DictReader((root / "labels.csv").open()))
    best, at_gt, margins = [], [], []
    for r in rows:
        ref = np.asarray(Image.open(root / r["ref_path"])).astype(np.float32)
        wide = np.asarray(Image.open(root / r["wide_path"])).astype(np.float32)
        small = np.asarray(Image.fromarray(ref).resize((FOOT, FOOT), Image.BOX)
                           ).astype(np.float32)
        gx, gy = float(r["gt_x"]), float(r["gt_y"])

        t = small - small.mean()
        tn = np.sqrt((t * t).sum()) + 1e-9

        # normalised cross-correlation of the down-scaled reference against the
        # wide view, evaluated on a coarse grid plus the true location
        sc = []
        for yy in range(0, N - FOOT, 12):
            for xx in range(0, N - FOOT, 12):
                w = wide[yy:yy + FOOT, xx:xx + FOOT]
                w = w - w.mean()
                sc.append(float((t * w).sum() / (tn * (np.sqrt((w * w).sum()) + 1e-9))))
        sc = np.array(sc)

        y0, x0 = int(round(gy)) - FOOT // 2, int(round(gx)) - FOOT // 2
        y0 = min(max(y0, 0), N - FOOT); x0 = min(max(x0, 0), N - FOOT)
        w = wide[y0:y0 + FOOT, x0:x0 + FOOT]; w = w - w.mean()
        true_score = float((t * w).sum() / (tn * (np.sqrt((w * w).sum()) + 1e-9)))

        best.append(sc.max()); at_gt.append(true_score)
        margins.append(true_score - np.percentile(sc, 99.5))
    return np.array(best), np.array(at_gt), np.array(margins)


for name, root in (("physical", pathlib.Path("out/phys24")),
                   ("resize (shared canvas)", pathlib.Path("out/resz24"))):
    b, g, m = score(root)
    print(f"{name:24s} NCC at true location: median {np.median(g):.3f} "
          f"(p10 {np.percentile(g,10):.3f}, p90 {np.percentile(g,90):.3f})")
    print(f"{'':24s} margin over 99.5th pct of decoys: median {np.median(m):+.3f}")
    print(f"{'':24s} fraction where truth is the global best: "
          f"{100*np.mean(g >= b - 1e-6):.0f}%\n")
