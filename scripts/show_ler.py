"""Side-by-side zoom with and without line-edge roughness, so the wobble can
be judged by eye against the real SEM imagery it is meant to imitate."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys, pathlib
import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from driftsense.layout import LineArray, build_dram, make_edge_roughness
from driftsense.physics import CaptureParams, render
from driftsense.raster import capture

N, TARGET = 1000, (6000.0, 6000.0)
P = CaptureParams(px_nm=1.0, probe_sigma_nm=1.0, drift_nm=1.0,
                  vibration_nm=0.4, dose_e_per_grey=45.0, read_sigma=0.9)

tiles = []
for sigma in (0.0, 0.7, 1.0, 1.3):
    lay = build_dram(extent_nm=14000.0, wl_pitch_nm=150.0, wl_width_nm=52.0,
                     bl_pitch_nm=1e9, bl_width_nm=0.0, contact_r_nm=0.0,
                     contact_core_r_nm=0.0, angle_deg=0.0)
    if sigma > 0:
        r = make_edge_roughness(np.random.default_rng(3), sigma_nm=sigma, xi_nm=10.0)
        for a in lay.arrays:
            if isinstance(a, LineArray):
                a.roughness = r
    img = render(capture(lay, TARGET, 1.0, N), P, np.random.default_rng(11))
    crop = img[380:580, 380:580]
    tiles.append(np.array(Image.fromarray(crop).resize((400, 400), Image.NEAREST)))

out = pathlib.Path("out/ler"); out.mkdir(parents=True, exist_ok=True)
Image.fromarray(np.hstack(tiles)).save(out / "ler_zoom.png")
print("3-sigma LWR left to right: 0.00, 2.10, 3.00, 3.90 nm")
print("wrote", out / "ler_zoom.png")
