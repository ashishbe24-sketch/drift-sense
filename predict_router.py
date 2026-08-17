#!/usr/bin/env python
r"""DriftRoute -- combined submission entry point for PS-02.

One function, `route.locate(reference, wide)`, that dispatches each pair to the
solution measured best at it: the learned matcher for single-target pairs, the
classical matcher for the multi-match case. Falls back to pure-classical if no
GPU / torch / checkpoint is present, so it always runs.

Single pair -- prints "x y":
    <py312> predict_router.py ref.png wide.png

Batch over a CSV:
    <py312> predict_router.py --csv data/eval200/labels.csv --root data/eval200 \
                              --out preds_router.csv
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
import route


def _load(path):
    img = np.asarray(Image.open(path).convert("L"))
    assert img.shape == (1000, 1000), f"{path}: expected 1000x1000, got {img.shape}"
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref", nargs="?")
    ap.add_argument("wide", nargs="?")
    ap.add_argument("--ckpt", default="driftmatch/checkpoints/best.pt")
    ap.add_argument("--csv", type=pathlib.Path)
    ap.add_argument("--root", type=pathlib.Path, default=pathlib.Path("."))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("predictions_router.csv"))
    a = ap.parse_args()

    net, device = route.load_net(a.ckpt)   # None -> classical-only, still runs

    if a.csv:
        rows = list(csv.DictReader(a.csv.open()))
        preds, times = [], []
        for r in rows:
            t0 = time.perf_counter()
            x, y = route.locate(_load(a.root / r["ref_path"]),
                                _load(a.root / r["wide_path"]), net=net, device=device)
            times.append(time.perf_counter() - t0)
            preds.append(dict(ref_path=r["ref_path"], wide_path=r["wide_path"],
                              pred_x=round(x, 3), pred_y=round(y, 3)))
        with a.out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["ref_path", "wide_path", "pred_x", "pred_y"])
            w.writeheader(); w.writerows(preds)
        t = np.array(times)
        print(f"{len(preds)} pairs -> {a.out}")
        print(f"time/pair: mean {1000*t.mean():.0f} ms  median {1000*np.median(t):.0f} ms")
    elif a.ref and a.wide:
        x, y = route.locate(_load(a.ref), _load(a.wide), net=net, device=device)
        print(f"{x:.3f} {y:.3f}")
    else:
        ap.error("give a reference and wide image, or --csv for batch mode")


if __name__ == "__main__":
    main()
