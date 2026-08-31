"""Head-to-head: fmt_pose.estimate_scale_rotation vs. solve.py's grid search.

Usage: python scripts/eval_fmt.py <dataset_dir> [--out per_pair.csv]

For each present pair, on pairs where the classical grid-search localizer
already lands within 5px of ground truth (so both methods are being compared
on the same solvable pairs, per docs/TEAMMATE_TASK_FOURIER_MELLIN.md), reports
scale/rotation accuracy and runtime for both:
  1. fmt_pose.estimate_scale_rotation(ref, wide) -- the Fourier-Mellin estimator.
  2. solve.locate(..., scales=PHASE2_SCALES, angles=PHASE2_ANGLES) -- the
     existing coarse-to-fine grid search + golden-section refinement.

Does NOT modify solve.py/route.py -- read-only comparison, per the task's
isolation rule.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import solve
import fmt_pose


def _tier_fracs(errs, tiers):
    errs = np.asarray(errs)
    return [float(np.mean(errs <= t)) for t in tiers]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = pathlib.Path(args.dataset)
    rows = list(csv.DictReader((root / "labels.csv").open()))
    present = [r for r in rows if r.get("present", "1") == "1"]
    print(f"{len(present)} present pairs in {root}")

    per_pair = []
    for r in present:
        ref = np.asarray(Image.open(root / r["ref_path"]).convert("L"))
        wide = np.asarray(Image.open(root / r["wide_path"]).convert("L"))
        gt_x, gt_y = float(r["gt_x"]), float(r["gt_y"])
        gt_scale, gt_rot = float(r["scale"]), float(r["rotation_deg"])

        t0 = time.perf_counter()
        x, y, info = solve.locate(ref, wide, return_info=True,
                                   scales=solve.PHASE2_SCALES, angles=solve.PHASE2_ANGLES)
        t_grid = time.perf_counter() - t0
        loc_err = float(np.hypot(x - gt_x, y - gt_y))
        solved = loc_err <= 5.0

        t0 = time.perf_counter()
        fmt_scale, fmt_theta, fmt_conf = fmt_pose.estimate_scale_rotation(ref, wide)
        t_fmt = time.perf_counter() - t0

        per_pair.append(dict(
            pair_id=r["pair_id"], regime=r.get("regime", "?"), solved=solved,
            loc_err=loc_err, gt_scale=gt_scale, gt_rot=gt_rot,
            grid_scale=info["scale"], grid_theta=info["theta"], t_grid=t_grid,
            fmt_scale=fmt_scale, fmt_theta=fmt_theta, fmt_conf=fmt_conf, t_fmt=t_fmt,
        ))

    solvable = [p for p in per_pair if p["solved"]]
    print(f"\n{len(solvable)}/{len(per_pair)} present pairs localized <=5px (comparison base)")

    for name, skey, tkey, tcol in (("grid search", "grid_scale", "grid_theta", "t_grid"),
                                    ("FMT", "fmt_scale", "fmt_theta", "t_fmt")):
        scale_err_pct = [abs(p[skey] - p["gt_scale"]) / p["gt_scale"] * 100.0 for p in solvable]
        rot_err_deg = [abs(p[tkey] - p["gt_rot"]) for p in solvable]
        s_tiers = _tier_fracs(scale_err_pct, [1.0, 2.0, 5.0])
        r_tiers = _tier_fracs(rot_err_deg, [0.25, 0.5, 1.0])
        runtime = [p[tcol] for p in per_pair]
        print(f"\n--- {name} ---")
        print(f"  scale: median {np.median(scale_err_pct):.3f}% err | "
              f"<=1%:{s_tiers[0]:.0%} <=2%:{s_tiers[1]:.0%} <=5%:{s_tiers[2]:.0%}")
        print(f"  theta: median {np.median(rot_err_deg):.3f} deg err | "
              f"<=0.25:{r_tiers[0]:.0%} <=0.5:{r_tiers[1]:.0%} <=1.0:{r_tiers[2]:.0%}")
        print(f"  runtime: median {np.median(runtime)*1000:.1f} ms/pair")

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_pair[0].keys()))
            w.writeheader()
            w.writerows(per_pair)
        print(f"\nper-pair CSV written to {args.out}")


if __name__ == "__main__":
    main()
