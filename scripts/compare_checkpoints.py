"""Decisive head-to-head on held-out Phase 2 data: classical vs net vs hybrid,
for the OLD and NEW checkpoints, plus scale/theta pose recovery.

Answers task step 5 ("does the new checkpoint beat 89% @5px / 1.6s?") and the
step-6 requirement to measure rotation recovery, not just localization.

    <venv-python> scripts/compare_checkpoints.py data/p2eval100 \
        driftmatch/checkpoints/best_phase2.pt driftmatch/checkpoints_new/best.pt

Localization (present pairs): @5px within-tolerance, median error, ms/pair for
  - classical alone            (solve.locate, PHASE2 scales+angles)
  - net alone (old / new)      (net x,y at fixed-10, no scale correction)
  - hybrid (old / new)         (classical scale/theta/found + net x,y == route.predict_full)
Pose (classical, on well-localized present pairs -- theta/scale always come from
the classical path, so this is checkpoint-independent): scale % error + tiers,
theta abs error (deg) + tiers, matching the Phase 2 scoring ladder.
Reads-only.
"""
from __future__ import annotations

import csv
import pathlib
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import solve

try:
    import torch
    from driftmatch.model import DriftMatchNet
    from driftmatch.infer import net_response, predict_from_response, load_checkpoint
    HAVE_TORCH = True
except Exception as e:  # noqa: BLE001
    print(f"[warn] torch/driftmatch unavailable ({e}); net paths skipped", file=sys.stderr)
    HAVE_TORCH = False


def load(p):
    return np.asarray(Image.open(p).convert("L"))


def load_net(path, device):
    net = DriftMatchNet(C=64).to(device).eval()
    net.load_state_dict(load_checkpoint(path, map_location=device)["model"])
    return net


def tiers(vals, edges):
    """fraction of `vals` (abs errors) within each cumulative edge."""
    v = np.asarray(vals)
    return [100.0 * np.mean(v <= e) for e in edges]


def main():
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data/p2eval100")
    ckpts = sys.argv[2:] if len(sys.argv) > 2 else ["driftmatch/checkpoints/best_phase2.pt"]
    device = "cuda" if (HAVE_TORCH and torch.cuda.is_available()) else "cpu"

    rows = [r for r in csv.DictReader((root / "labels.csv").open())
            if r["present"] == "1"]
    print(f"=== {root}  ({len(rows)} present pairs)  device={device} ===\n")

    nets = {}
    if HAVE_TORCH:
        for c in ckpts:
            if pathlib.Path(c).exists():
                nets[c] = load_net(c, device)
            else:
                print(f"[warn] checkpoint missing, skipped: {c}", file=sys.stderr)

    # accumulators
    e_cls, t_cls = [], 0.0
    e_net = {c: [] for c in nets}
    t_net = {c: 0.0 for c in nets}
    e_hyb = {c: [] for c in nets}
    scale_err, theta_err = [], []   # classical pose, on well-localized pairs

    for r in rows:
        ref = load(root / r["ref_path"]); wide = load(root / r["wide_path"])
        gx, gy = float(r["gt_x"]), float(r["gt_y"])
        gscale, gtheta = float(r["scale"]), float(r["rotation_deg"])

        t0 = time.perf_counter()
        xc, yc, info = solve.locate(ref, wide, return_info=True,
                                    scales=solve.PHASE2_SCALES,
                                    angles=solve.PHASE2_ANGLES)
        t_cls += time.perf_counter() - t0
        d_cls = np.hypot(xc - gx, yc - gy)
        e_cls.append(d_cls)

        # pose (classical) scored only where classical localized (credit>0, i.e. <=5px)
        if d_cls <= 5.0:
            scale_err.append(100.0 * abs(info["scale"] - gscale) / gscale)
            theta_err.append(abs(info["theta"] - gtheta))

        for c, net in nets.items():
            t0 = time.perf_counter()
            hm, off = net_response(net, ref, wide, device)
            xn, yn = predict_from_response(hm, off, center_rule=True)
            t_net[c] += time.perf_counter() - t0
            e_net[c].append(np.hypot(xn - gx, yn - gy))
            # hybrid = net x,y with classical scale/theta/found (route.predict_full)
            e_hyb[c].append(np.hypot(xn - gx, yn - gy))  # x,y identical to net; pose differs only in output cols

    n = len(rows)
    h5 = lambda e: 100.0 * np.mean(np.array(e) <= 5)
    med = lambda e: float(np.median(e))
    print("LOCALIZATION (present pairs)")
    print(f"  classical alone     {h5(e_cls):5.1f}% @5px   med {med(e_cls):.3f}px   "
          f"{1000*t_cls/n:.0f} ms/pair")
    for c in nets:
        print(f"  net alone  [{pathlib.Path(c).name}]  {h5(e_net[c]):5.1f}% @5px   "
              f"med {med(e_net[c]):.3f}px   {1000*t_net[c]/n:.0f} ms/pair")
    for c in nets:
        # hybrid x,y == net x,y; total time = classical (scale/theta/found) + net (x,y)
        thyb = (t_cls + t_net[c]) / n
        print(f"  hybrid     [{pathlib.Path(c).name}]  {h5(e_hyb[c]):5.1f}% @5px   "
              f"med {med(e_hyb[c]):.3f}px   {1000*thyb:.0f} ms/pair "
              f"(classical pose + net x,y == route.predict_full)")

    print(f"\nPOSE (classical path, {len(scale_err)} well-localized present pairs)")
    if scale_err:
        s1, s2, s5 = tiers(scale_err, [1.0, 2.0, 5.0])
        print(f"  scale : median {med(scale_err):.3f}%   "
              f"<=1% {s1:.0f}%  <=2% {s2:.0f}%  <=5% {s5:.0f}%")
        r025, r05, r10 = tiers(theta_err, [0.25, 0.5, 1.0])
        print(f"  theta : median {med(theta_err):.4f}deg   "
              f"<=0.25 {r025:.0f}%  <=0.5 {r05:.0f}%  <=1.0 {r10:.0f}%")
    else:
        print("  (no well-localized pairs)")


if __name__ == "__main__":
    main()
