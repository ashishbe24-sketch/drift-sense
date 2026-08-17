"""Measure how detectable the landmark actually is in each rendered wide view.

The intended policy is that ~8% of pairs are unsolvable by construction (the
landmark falls below the visibility floor) and the rest are solvable. That floor
was set from pixel size alone, so this checks it against the rendered result,
where defocus also matters.

Detectability is measured directly: render the wide view with and without the
landmark, take the RMS difference over the 100x100 footprint, and divide by the
local noise sigma. Below ~1 the landmark carries less signal than the noise and
no algorithm can find it.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys, pathlib, argparse
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from driftsense.physics import render
from driftsense.raster import make_pair
from driftsense.sampling import build_layout, sample_spec

TARGET_NM = (6000.0, 6000.0)
EXTENT_NM = 14000.0
N = 1000

ap = argparse.ArgumentParser()
ap.add_argument("--seed0", type=int, default=90000)
ap.add_argument("--n", type=int, default=80)
a = ap.parse_args()

rows = []
for seed in range(a.seed0, a.seed0 + a.n):
    spec = sample_spec(seed, n_px=N)
    off = (spec.gt_x - N / 2.0, spec.gt_y - N / 2.0)

    lay = build_layout(spec, TARGET_NM, EXTENT_NM)
    _, mat_with, _ = make_pair(lay, TARGET_NM, off, n_px=N)

    # identical layout, landmark removed
    lay0 = build_layout(spec, TARGET_NM, EXTENT_NM)
    n_lm = 3 if spec.landmark == "plus" else 1
    lay0.shapes = lay0.shapes[:-n_lm]
    _, mat_without, _ = make_pair(lay0, TARGET_NM, off, n_px=N)

    rng = np.random.default_rng(seed ^ 0xA11CE)
    w1 = render(mat_with, spec.wide, np.random.default_rng(seed ^ 0xA11CE)).astype(np.float32)
    w0 = render(mat_without, spec.wide, np.random.default_rng(seed ^ 0xA11CE)).astype(np.float32)

    x0, y0 = int(spec.gt_x) - 50, int(spec.gt_y) - 50
    sl = (slice(max(0, y0), min(N, y0 + 100)), slice(max(0, x0), min(N, x0 + 100)))
    signal = float(np.sqrt(np.mean((w1[sl] - w0[sl]) ** 2)))

    # local noise: high-frequency residual away from the landmark
    from scipy.signal import convolve2d
    L = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    c = convolve2d(w1[100:300, 100:300], L, mode="valid")
    noise = float(np.sqrt(np.pi / 2) * np.abs(c).mean() / 6.0)

    rows.append(dict(seed=seed, snr=signal / max(noise, 1e-6),
                     size=spec.landmark_size_nm, blur=spec.wide.total_blur_nm(),
                     regime=spec.regime, style=spec.style,
                     res=float(np.hypot(spec.wide.px_nm,
                                        2.355 * spec.wide.total_blur_nm()))))

snr = np.array([r["snr"] for r in rows])
size = np.array([r["size"] for r in rows])
blur = np.array([r["blur"] for r in rows])
res = np.array([r["res"] for r in rows])

print(f"n = {len(rows)}\n")
print("landmark SNR in the wide view (RMS signal / noise sigma):")
for p in (5, 10, 25, 50, 75, 95):
    print(f"   p{p:<3d} {np.percentile(snr, p):7.2f}")
for thr in (0.5, 1.0, 2.0):
    k = (snr < thr).sum()
    print(f"   below {thr:>4.1f}: {k:3d}/{len(rows)} ({100*k/len(rows):.0f}%)")
print("\n   intended: ~8% unsolvable by construction")

print("\nby landmark size vs effective wide resolution (size / res):")
ratio = size / res
for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 8), (8, 99)):
    m = (ratio >= lo) & (ratio < hi)
    if m.any():
        print(f"   size/res {lo}-{hi}: n={m.sum():3d}  median SNR {np.median(snr[m]):6.2f}"
              f"  frac SNR<1 {100*(snr[m] < 1).mean():5.0f}%")

print("\nby regime:")
for r in ("resolved", "coarse", "aliased"):
    m = np.array([x["regime"] == r for x in rows])
    if m.any():
        print(f"   {r:9s} n={m.sum():3d}  median SNR {np.median(snr[m]):6.2f}"
              f"  frac SNR<1 {100*(snr[m] < 1).mean():5.0f}%")
