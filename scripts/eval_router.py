"""Verify and tune DriftRoute (the combined router).

For each pair it computes the net response and the classical prediction once,
then sweeps the multi-match detection threshold, reporting for each:
  - overall within-5 px (the routed result)
  - how many pairs were sent to the classical path
  - routing quality vs the true multi-match label (precision / recall)
so the operating point is chosen from evidence and the router is checked against
each solution alone.

    <py312> scripts/eval_router.py data/eval200 driftmatch/checkpoints/best.pt
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
from driftmatch.infer import (net_response, predict_from_response,
                              load_checkpoint)
from route import is_multimatch

RATIOS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.85]


def load(p):
    return np.asarray(Image.open(p).convert("L"))


def main():
    root = pathlib.Path(sys.argv[1])
    ckpt = sys.argv[2] if len(sys.argv) > 2 else "driftmatch/checkpoints/best.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = DriftMatchNet(C=64).to(device).eval()
    net.load_state_dict(load_checkpoint(ckpt, map_location=device)["model"])

    rows = list(csv.DictReader((root / "labels.csv").open()))
    en, ec, hms, multi = [], [], [], []
    tn = tc = 0.0
    for r in rows:
        ref = load(root / r["ref_path"]); wide = load(root / r["wide_path"])
        gx, gy = float(r["gt_x"]), float(r["gt_y"])
        t0 = time.perf_counter(); hm, off = net_response(net, ref, wide, device); tn += time.perf_counter()-t0
        xn, yn = predict_from_response(hm, off)
        t0 = time.perf_counter(); xc, yc = locate_classical(ref, wide); tc += time.perf_counter()-t0
        en.append(np.hypot(xn-gx, yn-gy)); ec.append(np.hypot(xc-gx, yc-gy))
        hms.append(hm); multi.append(float(r.get("coarse_period_nm", 0) or 0) > 0)

    en, ec, multi = np.array(en), np.array(ec), np.array(multi, bool)
    h5 = lambda e: 100.0*np.mean(e <= 5)
    print(f"=== {root}  ({len(en)} pairs) ===")
    print(f"net alone       within5 {h5(en):.1f}%   ({1000*tn/len(en):.0f} ms/pair)")
    print(f"classical alone within5 {h5(ec):.1f}%   ({1000*tc/len(ec):.0f} ms/pair)")
    print(f"\n{'ratio':>6} {'routed->cls':>12} {'within5':>9} "
          f"{'route-precision':>16} {'route-recall':>13}")
    for ratio in RATIOS:
        to_cls = np.array([is_multimatch(hm, ratio) for hm in hms], bool)
        routed = np.where(to_cls, ec, en)
        # routing quality: did "send to classical" match the true multi label?
        tp = (to_cls & multi).sum(); fp = (to_cls & ~multi).sum(); fn = (~to_cls & multi).sum()
        prec = 100.0*tp/max(tp+fp, 1); rec = 100.0*tp/max(tp+fn, 1)
        print(f"{ratio:>6.2f} {int(to_cls.sum()):>12} {h5(routed):>8.1f}% "
              f"{prec:>15.0f}% {rec:>12.0f}%")


if __name__ == "__main__":
    main()
