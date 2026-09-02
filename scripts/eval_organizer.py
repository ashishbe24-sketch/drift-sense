"""Score a checkpoint on the organizers' real 20-pair sample, under the exact rubric.

    python scripts/eval_organizer.py --data data/organizer_sample \
        [--ckpt driftmatch/checkpoints/X.pt] [--use-net-xy]

Without --use-net-xy this measures the shipped classical path (the net is not
consulted for x,y). With it, the net supplies x,y -- the configuration whose
adoption is being decided.

The organizer sample is their withheld validation fold: it is used ONLY to score
a checkpoint that has already been selected elsewhere. Nothing here tunes
against it.
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
import route


def credit(err_px: float) -> float:
    """Localization credit tiers from the addendum."""
    if err_px <= 1.0:
        return 1.00
    if err_px <= 2.0:
        return 0.80
    if err_px <= 3.0:
        return 0.60
    if err_px <= 5.0:
        return 0.40
    return 0.00


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/organizer_sample")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--use-net-xy", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    root = pathlib.Path(args.data)
    gt = {r["pair_id"]: r for r in csv.DictReader((root / "ground_truth.csv").open())}
    pairs = list(csv.DictReader((root / "pairs.csv").open()))

    net = device = None
    if args.ckpt:
        net, device = route.load_net(pathlib.Path(args.ckpt))
        if net is None:
            print(f"WARNING: checkpoint {args.ckpt} did not load; classical only")

    rows, times = [], []
    for p in pairs:
        pid = p["pair_id"]
        ref = np.asarray(Image.open(root / p["reference_path"]).convert("L"))
        wide = np.asarray(Image.open(root / p["search_path"]).convert("L"))
        t0 = time.perf_counter()
        try:
            res = route.predict_full(ref, wide, net=net, device=device,
                                     use_net_xy=args.use_net_xy)
        except TypeError:      # older signature without use_net_xy
            res = route.predict_full(ref, wide, net=net, device=device)
        times.append(time.perf_counter() - t0)

        g = gt[pid]
        present = int(g["present"])
        err = (float(np.hypot(res.x - float(g["x"]), res.y - float(g["y"])))
               if present else float("nan"))
        rows.append(dict(pid=pid, present=present, found=res.found, err=err,
                         score=res.score,
                         scale=res.scale, gt_scale=float(g["scale"]),
                         theta=res.theta, gt_theta=float(g["theta"])))

    # --- localization: credit only on present pairs we did not falsely reject ---
    present_rows = [r for r in rows if r["present"] == 1]
    creds = [credit(r["err"]) if r["found"] == 1 else 0.0 for r in present_rows]
    # Set A = p001-p008, Set B = p009-p014 + p019/p020 are Set D (bonus, excluded here)
    setA = [c for c, r in zip(creds, present_rows) if r["pid"] <= "p008"]
    setB = [c for c, r in zip(creds, present_rows)
            if "p009" <= r["pid"] <= "p014"]
    mA = float(np.mean(setA)) if setA else 0.0
    mB = float(np.mean(setB)) if setB else 0.0
    loc40 = 40.0 * (0.45 * mA + 0.55 * mB)

    within5 = sum(1 for r in present_rows if r["found"] == 1 and r["err"] <= 5.0)

    # --- rejection F1 (present-positive) ---
    tp = sum(1 for r in rows if r["present"] == 1 and r["found"] == 1)
    fp = sum(1 for r in rows if r["present"] == 0 and r["found"] == 1)
    fn = sum(1 for r in rows if r["present"] == 1 and r["found"] == 0)
    f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0

    if not args.quiet:
        print(f"{'id':6} {'gtP':3} {'fnd':3} {'err_px':>8} {'credit':>6} "
              f"{'scale(rec/gt)':>16} {'theta(rec/gt)':>16}")
        for r in rows:
            e = f"{r['err']:8.2f}" if r["present"] else "     n/a"
            c = f"{credit(r['err']) if r['found'] and r['present'] else 0.0:6.2f}" if r["present"] else "     -"
            sc = f"{r['scale']:6.2f}/{r['gt_scale']:5.2f}" if r["present"] else "        -"
            th = f"{r['theta']:+6.2f}/{r['gt_theta']:+5.2f}" if r["present"] else "        -"
            print(f"{r['pid']:6} {r['present']:3} {r['found']:3} {e} {c} {sc:>16} {th:>16}")

    label = f"ckpt={pathlib.Path(args.ckpt).name if args.ckpt else 'none'} use_net_xy={args.use_net_xy}"
    print(f"\n=== {label} ===")
    print(f"Localization /40      : {loc40:.2f}   (SetA mean {mA:.3f}, SetB mean {mB:.3f})")
    print(f"Present within 5px    : {within5}/{len(present_rows)}")
    print(f"Rejection F1(present+): {f1:.3f}   (TP {tp} FP {fp} FN {fn})")
    print(f"Median time/pair      : {np.median(times)*1000:.0f} ms")


if __name__ == "__main__":
    main()
