"""Where does the time actually go in one pair? Warm timings, repeated."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys, pathlib, time
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from driftsense.layout import make_edge_roughness, LineArray
from driftsense.physics import render
from driftsense.raster import capture, make_pair
from driftsense.sampling import build_layout, sample_spec

TARGET, EXTENT, N = (6000.0, 6000.0), 14000.0, 1000


def t(fn, n=5):
    fn()                                   # warm
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


print("roughness table build : %6.0f ms" %
      (1000 * t(lambda: make_edge_roughness(np.random.default_rng(1), 1.0, 10.0))))

for tag, seed in (("plain pair", 90001), ("periodic pair", 90000)):
    s = sample_spec(seed, n_px=N)
    per = "periodic" in s.placement
    print(f"\n--- {tag} (seed {seed}, periodic={per}, "
          f"instances={s.n_instances}) ---")
    print("  build_layout        : %6.0f ms" %
          (1000 * t(lambda: build_layout(s, TARGET, EXTENT))))
    lay = build_layout(s, TARGET, EXTENT)
    off = (s.gt_x - N / 2, s.gt_y - N / 2)
    print("  reference raster    : %6.0f ms" %
          (1000 * t(lambda: capture(lay, TARGET, 1.0, N))))
    lay_nr = build_layout(s, TARGET, EXTENT)
    for a in lay_nr.arrays:
        if isinstance(a, LineArray):
            a.roughness = None
    print("    (same, no LER)    : %6.0f ms" %
          (1000 * t(lambda: capture(lay_nr, TARGET, 1.0, N))))
    print("  full make_pair      : %6.0f ms" %
          (1000 * t(lambda: make_pair(lay, TARGET, off, n_px=N))))
    m1, m2, _ = make_pair(lay, TARGET, off, n_px=N)
    rng = np.random.default_rng(7)
    print("  physics x2          : %6.0f ms" %
          (1000 * t(lambda: (render(m1, s.ref, rng), render(m2, s.wide, rng)))))
