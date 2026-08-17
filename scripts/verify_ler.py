"""Verify line-edge roughness.

Three checks:
  1. the rendered roughness matches the cited 3-sigma LWR and correlation length
  2. both captures see the *identical* physical wobble -- it is a property of
     the wafer, not of the capture
  3. the cost is acceptable
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys, pathlib, time
import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from driftsense.layout import build_dram, make_edge_roughness
from driftsense.raster import capture, make_pair
from driftsense.physics import CaptureParams, render
from driftsense.sampling import build_layout, sample_spec

N = 1000
TARGET = (6000.0, 6000.0)


def measure_edges(img, px_nm, thresh=None):
    """Sub-pixel edge positions of horizontal lines, per column."""
    a = img.astype(np.float32)
    if thresh is None:
        thresh = 0.5 * (np.percentile(a, 15) + np.percentile(a, 85))
    prof = a.mean(1)
    rising = [i for i in range(2, len(prof) - 2)
              if prof[i] < thresh <= prof[i + 1]]
    pos = []
    for r in rising:
        col = a[r - 1:r + 3, :]
        # linear interpolation of the threshold crossing in each column
        y = []
        for c in range(a.shape[1]):
            seg = col[:, c]
            k = np.where((seg[:-1] < thresh) & (seg[1:] >= thresh))[0]
            if len(k):
                k = k[0]
                t = (thresh - seg[k]) / max(seg[k + 1] - seg[k], 1e-6)
                y.append((r - 1 + k + t) * px_nm)
            else:
                y.append(np.nan)
        pos.append(np.array(y))
    return pos


# --- 1. magnitude and correlation length ------------------------------------
print("=== rendered roughness vs the cited model ===")
for sigma, xi in ((0.6, 8.0), (1.0, 10.0), (1.3, 13.0)):
    rough = make_edge_roughness(np.random.default_rng(5), sigma_nm=sigma, xi_nm=xi)
    lay = build_dram(extent_nm=14000.0, wl_pitch_nm=160.0, bl_pitch_nm=1e9,
                     bl_width_nm=0.0, contact_r_nm=0.0, contact_core_r_nm=0.0,
                     angle_deg=0.0)
    for arr in lay.arrays:
        if hasattr(arr, "roughness"):
            arr.roughness = rough
    mat = capture(lay, TARGET, 1.0, N)
    edges = measure_edges(mat, 1.0)
    devs = [e - np.nanmean(e) for e in edges if np.isfinite(e).mean() > 0.9]
    if not devs:
        print(f"  sigma {sigma}: no edges found"); continue
    d = np.concatenate(devs)
    d = d[np.isfinite(d)]
    # correlation length from the lag at which autocorrelation falls to 1/e
    x = devs[len(devs) // 2]; x = x[np.isfinite(x)]; x = x - x.mean()
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    ac /= ac[0]
    lag = int(np.argmax(ac < 1 / np.e)) if (ac < 1 / np.e).any() else -1
    print(f"  target 3sigma {3*sigma:.2f} nm, xi {xi:.0f} nm  ->  "
          f"measured 3sigma {3*d.std():.2f} nm, xi ~{lag} nm  ({len(devs)} edges)")

# --- 2. the two captures must see identical roughness -----------------------
print("\n=== is the wobble identical in both views? ===")
spec = sample_spec(4242, n_px=N)
lay = build_layout(spec, TARGET, 14000.0)
mref, mwide, gt = make_pair(lay, TARGET, (0.0, 0.0), n_px=N)
# the reference covers 1 um; the corresponding 100x100 region of the wide view
# is the same physical area, so a 10x downscale of the reference must agree
small = np.asarray(Image.fromarray(np.clip(mref, 0, 255).astype(np.uint8))
                   .resize((100, 100), Image.BOX)).astype(np.float32)
crop = mwide[450:550, 450:550]
a, b = small - small.mean(), crop - crop.mean()
print(f"  correlation of down-scaled reference vs wide target region: "
      f"{float((a*b).mean()/(a.std()*b.std()+1e-9)):.4f}")
print("  (a low value here would mean the two views disagree about where the "
      "edges physically are)")

# --- 3. cost ----------------------------------------------------------------
print("\n=== cost ===")
for tag, use in (("without LER", False), ("with LER", True)):
    s = sample_spec(4243, n_px=N)
    lay = build_layout(s, TARGET, 14000.0)
    if not use:
        for arr in lay.arrays:
            if hasattr(arr, "roughness"):
                arr.roughness = None
    t0 = time.perf_counter()
    make_pair(lay, TARGET, (60.0, -40.0), n_px=N)
    print(f"  {tag:12s} {time.perf_counter()-t0:.2f} s/pair")

