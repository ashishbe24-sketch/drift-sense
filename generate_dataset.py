#!/usr/bin/env python
"""Drift-Sense synthetic dataset generator (SEMICON India Hackathon 2026, PS-02).

Produces (reference, wide-search) SEM image pairs with exact ground-truth
coordinates for the navigation-error recovery problem.

    python generate_dataset.py --style dram --pairs 30 --out data/eval

Design
------
A single vector layout is defined in nanometres and each view is rendered by
sampling that layout independently at its own pixel size, dose, focus and noise
realisation. The wide view is therefore a genuinely coarser *acquisition*, not
a resized copy of the reference, and the ground-truth coordinate is analytic --
it comes from where the landmark was placed, never from matching pixels.

No generative image model is used anywhere: every pixel is computed from
geometry and a documented SEM image-formation model. See docs/GENERATOR_SPEC.md
for the physical parameters and their sources.

Outputs
-------
    images/<id>_ref.png     1000x1000 uint8 grayscale, lossless
    images/<id>_wide.png    1000x1000 uint8 grayscale, lossless
    labels.csv              one row per pair; ref_path, wide_path, gt_x, gt_y
                            plus every sampled parameter
    pairs/<id>.json         full per-pair record, including the seed
    dataset_meta.json       generator version and global configuration
"""
from __future__ import annotations

import os

# Must precede the first numpy import. Each worker renders one image at a time,
# so BLAS threading buys nothing here -- but left unpinned, every worker
# reserves its own thread pool and a modest machine runs out of commit charge
# before it runs out of cores.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import json
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from driftsense.physics import render, render_downsampled
from driftsense.raster import make_pair, make_pair_shared_canvas
from driftsense.sampling import build_layout, sample_spec

VERSION = "0.1.0"
N_PX = 1000
TARGET_NM = (6000.0, 6000.0)
EXTENT_NM = 14000.0
FINE_PX_NM = 2.0   # intermediate resolution used by --render-mode resize


def generate_one(seed: int, style: str | None, out_dir: pathlib.Path,
                 n_px: int = N_PX, render_mode: str = "physical"):
    """Render and write one pair. Returns the manifest row."""
    spec = sample_spec(seed, style=style, n_px=n_px)
    layout = build_layout(spec, target_nm=TARGET_NM, extent_nm=EXTENT_NM)

    offset = (spec.gt_x - n_px / 2.0, spec.gt_y - n_px / 2.0)
    rng = np.random.default_rng(seed ^ 0xA11CE)

    if render_mode == "resize":
        mat_ref, fine_wide, gt, factor = make_pair_shared_canvas(
            layout, TARGET_NM, offset, n_px=n_px)
        ref = render(mat_ref, spec.ref, rng)
        wide = render_downsampled(fine_wide, factor, 1.0, spec.wide, rng)
    else:
        mat_ref, mat_wide, gt = make_pair(layout, TARGET_NM, offset, n_px=n_px)
        ref = render(mat_ref, spec.ref, rng)
        wide = render(mat_wide, spec.wide, rng)

    # the placement is the ground truth; this only guards against a regression
    assert abs(gt[0] - spec.gt_x) < 1e-6 and abs(gt[1] - spec.gt_y) < 1e-6

    # the organisers' automated harness assumes exactly this size
    assert ref.shape == (n_px, n_px) and wide.shape == (n_px, n_px)

    pid = f"{seed:08d}"
    img_dir = out_dir / "images"
    Image.fromarray(ref).save(img_dir / f"{pid}_ref.png", optimize=True)
    Image.fromarray(wide).save(img_dir / f"{pid}_wide.png", optimize=True)

    row = spec.manifest_row()
    row.update(pair_id=pid, render_mode=render_mode,
               ref_path=f"images/{pid}_ref.png",
               wide_path=f"images/{pid}_wide.png")

    rec = dict(row)
    rec["ref_capture"] = spec.ref.as_dict()
    rec["wide_capture"] = spec.wide.as_dict()
    rec["layout_meta"] = {k: v for k, v in layout.meta.items()}
    (out_dir / "pairs" / f"{pid}.json").write_text(json.dumps(rec, indent=2))
    return row


def _worker(args):
    return generate_one(*args)


COLUMNS = ["pair_id", "ref_path", "wide_path", "gt_x", "gt_y", "style",
           "regime", "pitch_nm", "rotation_deg", "landmark",
           "landmark_size_nm", "landmark_wide_px", "difficulty",
           "size_over_res", "coarse_period_nm", "n_instances",
           "ler_3sigma_nm", "ler_xi_nm", "wide_res_nm", "placement",
           "ref_blur_nm", "wide_blur_nm", "ref_dose", "wide_dose",
           "wide_charging", "ref_drift_nm", "wide_drift_nm",
           "ref_vib_nm", "wide_vib_nm", "render_mode", "seed"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--style", choices=["dram", "finfet", "mixed"], default="mixed",
                    help="die architecture to generate (default: mixed)")
    ap.add_argument("--pairs", type=int, default=30,
                    help="number of image pairs (default: 30, the stated minimum)")
    ap.add_argument("--out", type=pathlib.Path, required=True,
                    help="output directory")
    ap.add_argument("--seed", type=int, default=1000,
                    help="first seed; pairs use seed..seed+pairs-1")
    ap.add_argument("--seeds-file", type=pathlib.Path,
                    help="explicit seed list, one per line; reproduces a fixed set")
    ap.add_argument("--render-mode", choices=["physical", "resize"], default="physical",
                    help="physical: render each view independently from the nm "
                         "layout (default, used for training). resize: build the "
                         "wide view by downsampling a fine render, mimicking the "
                         "organisers' starter prompt -- held-out validation only")
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel processes (0 = cpu_count-2)")
    a = ap.parse_args()

    if a.seeds_file:
        seeds = [int(s) for s in a.seeds_file.read_text().split() if s.strip()]
    else:
        seeds = list(range(a.seed, a.seed + a.pairs))

    style = None if a.style == "mixed" else a.style
    for sub in ("images", "pairs"):
        (a.out / sub).mkdir(parents=True, exist_ok=True)

    # The resize path holds a 10000x10000 float32 field plus filter
    # temporaries, roughly 1 GB per worker, so it needs a much smaller pool
    # than the physical path.
    default_workers = (3 if a.render_mode == "resize"
                       else max(1, (os.cpu_count() or 2) - 2))
    workers = a.workers or default_workers

    t0 = time.perf_counter()
    jobs = [(s, style, a.out, N_PX, a.render_mode) for s in seeds]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(_worker, jobs, chunksize=1))
    else:
        rows = [_worker(j) for j in jobs]
    dt = time.perf_counter() - t0

    rows.sort(key=lambda r: r["pair_id"])
    with (a.out / "labels.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    (a.out / "dataset_meta.json").write_text(json.dumps(dict(
        generator="driftsense", version=VERSION, n_px=N_PX,
        ref_px_nm=1.0, wide_px_nm=10.0,
        ref_fov_nm=N_PX * 1.0, wide_fov_nm=N_PX * 10.0,
        footprint_px=int(N_PX * 1.0 / 10.0),
        style=a.style, render_mode=a.render_mode, pairs=len(rows), seeds=seeds,
        layout_extent_nm=EXTENT_NM, target_nm=TARGET_NM,
    ), indent=2))
    (a.out / "seeds.txt").write_text("\n".join(str(s) for s in seeds) + "\n")

    n_alias = sum(r["regime"] == "aliased" for r in rows)
    n_unsolv = sum(r["landmark_size_nm"] < 30.0 for r in rows)
    n_unif = sum(r["placement"] == "uniform" for r in rows)
    print(f"{len(rows)} pairs in {dt:.1f}s  ({dt / len(rows):.2f}s/pair, "
          f"{workers} workers)  ->  {a.out}")
    print(f"  regimes: aliased {n_alias}, uniform placement {n_unif}, "
          f"landmark below the ~30 nm visibility floor {n_unsolv}")


if __name__ == "__main__":
    main()



