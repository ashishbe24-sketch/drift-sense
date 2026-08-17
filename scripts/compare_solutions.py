"""Head-to-head: DriftFind (classical) vs DriftMatchNet (learned) on one dataset.

Runs both solutions over the same pairs and prints a side-by-side table of
within-1/2/5 px hit-rate (overall and split by plain vs multi-match) plus time
per pair. The held-out set (val_resize60) is the one that decides which ships:
beating our own eval200 is easy, generalising to a different generator is the
real test.

    <py312> scripts/compare_solutions.py data/val_resize60 driftmatch/checkpoints/best.pt

Uses Python312 (has numpy+scipy for the classical path and torch for the net).
Do not run while training is in progress.
"""
from __future__ import annotations

import csv
import pathlib
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from solve import locate as locate_classical
from driftmatch.model import DriftMatchNet
from driftmatch.infer import locate_net, load_checkpoint


def load(p):
    return np.asarray(Image.open(p).convert("L"))


def hits(errs, mask, thr=5):
    e = errs[mask]
    return 100.0 * np.mean(e <= thr) if len(e) else float("nan")


def main():
    root = pathlib.Path(sys.argv[1])
    ckpt_path = sys.argv[2] if len(sys.argv) > 2 else "driftmatch/checkpoints/best.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = load_checkpoint(ckpt_path, map_location=device)
    net = DriftMatchNet(C=64).to(device).eval()
    net.load_state_dict(ckpt["model"])

    rows = list(csv.DictReader((root / "labels.csv").open()))
    ec, en, tc, tn, multi = [], [], [], [], []
    for r in rows:
        ref = load(root / r["ref_path"]); wide = load(root / r["wide_path"])
        gx, gy = float(r["gt_x"]), float(r["gt_y"])

        t0 = time.perf_counter(); xc, yc = locate_classical(ref, wide); tc.append(time.perf_counter()-t0)
        t0 = time.perf_counter(); xn, yn = locate_net(net, ref, wide, device); tn.append(time.perf_counter()-t0)
        ec.append(np.hypot(xc-gx, yc-gy)); en.append(np.hypot(xn-gx, yn-gy))
        multi.append(float(r.get("coarse_period_nm", 0) or 0) > 0)

    ec, en, multi = np.array(ec), np.array(en), np.array(multi, bool)
    allm = np.ones(len(ec), bool)

    print(f"=== {root}  ({len(ec)} pairs)   within 5 px ===")
    print(f"{'group':14} {'DriftFind':>12} {'DriftMatchNet':>14}")
    for name, m in [("ALL", allm), ("plain", ~multi), ("multi-match", multi)]:
        print(f"{name:14} {hits(ec,m):>11.1f}% {hits(en,m):>13.1f}%")
    print(f"\nwithin 1 px    ALL   DriftFind {hits(ec,allm,1):.1f}%   "
          f"DriftMatchNet {hits(en,allm,1):.1f}%")
    print(f"time/pair      DriftFind {1000*np.mean(tc):.0f} ms   "
          f"DriftMatchNet {1000*np.mean(tn):.0f} ms")


if __name__ == "__main__":
    main()
