"""Full evaluation of a trained DriftMatchNet checkpoint on a dataset.

Mirrors scripts/eval_solver.py (the classical matcher's harness) exactly, so the
two solutions are scored the same way and are directly comparable: within
1/2/5 px hit-rate, median/mean error, split by plain vs multi-match, and time
per pair. Run with the training interpreter (Python312 has torch+CUDA):

    <py312> scripts/eval_net.py data/eval200 driftmatch/checkpoints/best.pt

Runs on GPU if available, else CPU. Do not run while training is in progress --
it would contend for the 4 GB card.
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
from driftmatch.model import DriftMatchNet
from driftmatch.infer import locate_net, load_checkpoint


def load(path):
    return np.asarray(Image.open(path).convert("L"))


def main():
    root = pathlib.Path(sys.argv[1])
    ckpt_path = sys.argv[2] if len(sys.argv) > 2 else "driftmatch/checkpoints/best.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = load_checkpoint(ckpt_path, map_location=device)
    net = DriftMatchNet(C=64).to(device).eval()
    net.load_state_dict(ckpt["model"])
    print(f"=== DriftMatchNet ({ckpt_path}, epoch {ckpt.get('epoch','?')}) "
          f"on {root}  device={device} ===")

    rows = list(csv.DictReader((root / "labels.csv").open()))
    errs, times, multi = [], [], []
    for r in rows:
        ref = load(root / r["ref_path"])
        wide = load(root / r["wide_path"])
        t0 = time.perf_counter()
        x, y = locate_net(net, ref, wide, device)
        times.append(time.perf_counter() - t0)
        errs.append(float(np.hypot(x - float(r["gt_x"]), y - float(r["gt_y"]))))
        multi.append(float(r.get("coarse_period_nm", 0) or 0) > 0)

    errs = np.array(errs); multi = np.array(multi, bool)

    def report(name, mask):
        if mask.sum() == 0:
            return
        e = errs[mask]
        print(f"\n{name}  (n={int(mask.sum())})")
        print(f"  median {np.median(e):.2f} px   mean {e.mean():.2f} px")
        for thr in (1, 2, 5, 10, 25):
            print(f"    <= {thr:2d} px : {100*np.mean(e <= thr):5.1f} %")

    report("ALL", np.ones(len(errs), bool))
    report("plain", ~multi)
    report("multi-match", multi)
    print(f"\ntime/pair: mean {1000*np.mean(times):.0f} ms  "
          f"median {1000*np.median(times):.0f} ms")


if __name__ == "__main__":
    main()
