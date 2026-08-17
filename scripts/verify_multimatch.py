"""Verify the coarse-periodic case.

Three things must hold for these pairs to be usable:
  1. decoys actually appear in the wide view (not culled or off-field)
  2. the true site is the candidate closest to the image centre, so our label
     agrees with the stated "closest to centre" rule
  3. rendering cost has not exploded now that a lattice carries many shapes
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys, pathlib, time
import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from driftsense.physics import render
from driftsense.raster import make_pair
from driftsense.sampling import build_layout, sample_spec

TARGET, EXTENT, N = (6000.0, 6000.0), 14000.0, 1000
WIDE_PX_NM = 10.0

periodic, plain, times = [], [], []
for seed in range(3000, 3120):
    s = sample_spec(seed, n_px=N)
    (periodic if s.coarse_period_nm else plain).append(s)

print(f"120 seeds: {len(periodic)} periodic ({100*len(periodic)/120:.0f}%, "
      f"policy 40%), {len(plain)} plain")

# --- rule consistency: the true site must be the centre-most instance --------
bad = 0
for s in periodic:
    p_px = s.coarse_period_nm / WIDE_PX_NM
    dx, dy = s.gt_x - N / 2.0, s.gt_y - N / 2.0
    # nearest lattice instance to the image centre, in units of the period
    if abs(dx) > p_px / 2 or abs(dy) > p_px / 2:
        bad += 1
print(f"true site is the centre-most instance: {len(periodic)-bad}/{len(periodic)}"
      f"  ({'OK' if bad == 0 else 'VIOLATIONS'})")

inst = np.array([s.n_instances for s in periodic])
vis = np.array([(10000.0 / s.coarse_period_nm) ** 2 for s in periodic])
print(f"instances built  : {inst.min()}..{inst.max()}")
print(f"expected visible in a 10 um field: {vis.min():.0f}..{vis.max():.0f} "
      f"(median {np.median(vis):.0f})")

# --- render a few and count decoys that actually appear ----------------------
out = pathlib.Path("out/multimatch"); out.mkdir(parents=True, exist_ok=True)
sheet = []
for s in periodic[:4]:
    lay = build_layout(s, TARGET, EXTENT)
    t0 = time.perf_counter()
    mref, mwide, gt = make_pair(lay, TARGET, (s.gt_x - N/2, s.gt_y - N/2), n_px=N)
    rng = np.random.default_rng(s.seed ^ 0xA11CE)
    ref, wide = render(mref, s.ref, rng), render(mwide, s.wide, rng)
    times.append(time.perf_counter() - t0)

    # count instances by correlating the down-scaled reference and peak-picking
    small = np.asarray(Image.fromarray(ref).resize((100, 100), Image.BOX)).astype(np.float32)
    t = small - small.mean(); tn = np.sqrt((t*t).sum()) + 1e-9
    step = 6
    sc = np.full(((N-100)//step + 1, (N-100)//step + 1), -1.0, np.float32)
    for iy, yy in enumerate(range(0, N-100+1, step)):
        for ix, xx in enumerate(range(0, N-100+1, step)):
            w = wide[yy:yy+100, xx:xx+100].astype(np.float32); w = w - w.mean()
            sc[iy, ix] = (t*w).sum() / (tn * (np.sqrt((w*w).sum()) + 1e-9))
    peaks = int((sc > 0.7 * sc.max()).sum())
    ys, xs = np.unravel_index(np.argmax(sc), sc.shape)
    best = (xs*step + 50, ys*step + 50)
    err = float(np.hypot(best[0]-s.gt_x, best[1]-s.gt_y))
    print(f"  seed {s.seed}: period {s.coarse_period_nm:6.0f} nm  "
          f"strong peaks {peaks:4d}  best-NCC {sc.max():.3f}  "
          f"best-peak error {err:6.1f} px  {'(decoy won)' if err > 25 else ''}")

    w2 = wide.copy()
    for (cx, cy), val in (((s.gt_x, s.gt_y), 255),):
        x0, y0 = int(cx)-50, int(cy)-50
        w2[max(0,y0):min(N,y0+100), max(0,x0)] = val
        w2[max(0,y0):min(N,y0+100), min(N-1,x0+100)] = val
        w2[max(0,y0), max(0,x0):min(N,x0+100)] = val
        w2[min(N-1,y0+100), max(0,x0):min(N,x0+100)] = val
    sheet.append(np.array(Image.fromarray(w2).resize((420, 420), Image.BOX)))

Image.fromarray(np.hstack(sheet)).save(out / "periodic_wide.png")
print(f"\nrender time with lattice: {np.mean(times):.2f} s/pair "
      f"(plain pairs were ~0.28 s)")
print(f"wrote {out/'periodic_wide.png'} (true site boxed)")
