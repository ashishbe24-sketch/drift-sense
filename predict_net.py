#!/usr/bin/env python
r"""DriftMatchNet -- learned submission entry point for PS-02 (Solution 2).

The algorithm is the network in driftmatch/, wrapped so the evaluator can call
it on a single pair or over a CSV, exactly like predict.py wraps the classical
solver. Needs the training interpreter (torch) and a trained checkpoint.

Single pair -- prints "x y":
    <py312> predict_net.py ref.png wide.png --ckpt driftmatch/checkpoints/best.pt

Batch over a CSV:
    <py312> predict_net.py --csv data/eval200/labels.csv --root data/eval200 \
                           --ckpt driftmatch/checkpoints/best.pt --out preds_net.csv

Solution 1 (predict.py, classical) is the domain-safe default; this learned
variant is submitted alongside it and used where it measurably wins.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from driftmatch.model import DriftMatchNet
from driftmatch.infer import locate_net, load_checkpoint


def _load(path):
    img = np.asarray(Image.open(path).convert("L"))
    assert img.shape == (1000, 1000), f"{path}: expected 1000x1000, got {img.shape}"
    return img


def _make_net(ckpt_path, device):
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    net = DriftMatchNet(C=64).to(device).eval()
    net.load_state_dict(ckpt["model"])
    return net


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref", nargs="?")
    ap.add_argument("wide", nargs="?")
    ap.add_argument("--ckpt", default="driftmatch/checkpoints/best.pt")
    ap.add_argument("--csv", type=pathlib.Path)
    ap.add_argument("--root", type=pathlib.Path, default=pathlib.Path("."))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("predictions_net.csv"))
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = _make_net(a.ckpt, device)

    if a.csv:
        rows = list(csv.DictReader(a.csv.open()))
        preds, times = [], []
        for r in rows:
            t0 = time.perf_counter()
            x, y = locate_net(net, _load(a.root / r["ref_path"]),
                              _load(a.root / r["wide_path"]), device)
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
        x, y = locate_net(net, _load(a.ref), _load(a.wide), device)
        print(f"{x:.3f} {y:.3f}")
    else:
        ap.error("give a reference and wide image, or --csv for batch mode")


if __name__ == "__main__":
    main()
