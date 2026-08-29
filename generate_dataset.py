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
from driftsense.sampling import build_layout, sample_spec, SCALE_RANGE

VERSION = "0.1.0"
N_PX = 1000
TARGET_NM = (6000.0, 6000.0)
EXTENT_NM = 14000.0
FINE_PX_NM = 2.0   # intermediate resolution used by --render-mode resize


def generate_one(seed: int, style: str | None, out_dir: pathlib.Path,
                 n_px: int = N_PX, render_mode: str = "physical",
                 scale_range: tuple | None = None, signed_rotation: bool = False,
                 absent: bool = False, scan_distortion_max: float = 0.0,
                 optical_aberrations: bool = False):
    """Render and write one pair. Returns the manifest row."""
    spec = sample_spec(seed, style=style, n_px=n_px, scale_range=scale_range,
                       signed_rotation=signed_rotation, absent=absent,
                       scan_distortion_max=scan_distortion_max,
                       optical_aberrations=optical_aberrations)
    layout = build_layout(spec, target_nm=TARGET_NM, extent_nm=EXTENT_NM)

    offset = (spec.gt_x - n_px / 2.0, spec.gt_y - n_px / 2.0)
    rng = np.random.default_rng(seed ^ 0xA11CE)

    # The wide view is rendered at the pair's actual zoom (spec.wide.px_nm). In
    # Phase 1 this is the fixed 10.0; in Phase 2 it is the sampled [8,12] value,
    # so the rendered pixels match the recorded ground-truth scale.
    if render_mode == "resize":
        if scale_range is not None or absent or signed_rotation:
            raise ValueError("resize render-mode supports neither a scale range, "
                             "absent pairs, nor signed (relative) rotation; use "
                             "physical mode")
        mat_ref, fine_wide, gt, factor = make_pair_shared_canvas(
            layout, TARGET_NM, offset, n_px=n_px)
        ref = render(mat_ref, spec.ref, rng)
        wide = render_downsampled(fine_wide, factor, 1.0, spec.wide, rng)
    else:
        # Phase 2 `theta`: reuse the existing signed-rotation draw (spec.rotation_deg,
        # already CCW-signed when signed_rotation=True) as the WIDE-relative rotation
        # to recover, rather than a rotation shared identically by both captures. When
        # signed_rotation=False (Phase 1), relative_theta_deg is 0.0 and make_pair()'s
        # behaviour is untouched -- ref and wide still share one layout.angle_deg.
        rel_theta = spec.rotation_deg if signed_rotation else 0.0
        mat_ref, mat_wide, gt = make_pair(layout, TARGET_NM, offset,
                                          wide_px_nm=spec.wide.px_nm, n_px=n_px,
                                          absent=absent, relative_theta_deg=rel_theta)
        ref = render(mat_ref, spec.ref, rng)
        wide = render(mat_wide, spec.wide, rng)

    # the placement is the ground truth; this only guards against a regression
    # (for absent pairs both sides carry the -1 sentinel, so it still holds)
    assert abs(gt[0] - spec.gt_x) < 1e-6 and abs(gt[1] - spec.gt_y) < 1e-6

    # Barrel distortion (Phase 2 Set B) is applied INSIDE render() (physics.py),
    # so `wide` already carries the warp by this point -- the ground truth must
    # be shifted to match. Sign convention verified empirically (not assumed):
    # new_gt = gt + displacement_at(gt) for the forward map this function uses.
    # Applied here, BEFORE the scan-distortion block below, because that is the
    # actual pixel order (barrel happens first, inside render()) -- warps
    # compose in sequence, so scan-distortion's field must be evaluated at the
    # already-barrel-shifted position, not the original one.
    if spec.wide.barrel_k1 != 0 and spec.present:
        from driftsense.physics import barrel_displacement_at
        bdx, bdy = barrel_displacement_at(spec.gt_x, spec.gt_y, wide.shape,
                                          spec.wide.barrel_k1)
        spec.gt_x = float(spec.gt_x + bdx)
        spec.gt_y = float(spec.gt_y + bdy)

    # Scan distortion (Phase 2 Set B): warp the wide by a smooth field and shift
    # the ground truth by the field's value at the landmark, so the label stays
    # exact. Applied after the placement check, wide-only (a relative warp).
    if spec.wide.scan_distortion_px > 0:
        from driftsense.physics import scan_distortion_field, apply_scan_distortion
        dx, dy = scan_distortion_field(wide.shape, spec.wide.scan_distortion_px, rng)
        wide = np.clip(np.rint(apply_scan_distortion(wide, dx, dy)), 0, 255).astype(np.uint8)
        if spec.present:
            gyi = min(max(int(round(spec.gt_y)), 0), wide.shape[0] - 1)
            gxi = min(max(int(round(spec.gt_x)), 0), wide.shape[1] - 1)
            spec.gt_x = float(spec.gt_x - float(dx[gyi, gxi]))
            spec.gt_y = float(spec.gt_y - float(dy[gyi, gxi]))

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


COLUMNS = ["pair_id", "ref_path", "wide_path", "gt_x", "gt_y", "scale", "style",
           "regime", "pitch_nm", "rotation_deg", "landmark",
           "landmark_size_nm", "landmark_wide_px", "difficulty",
           "size_over_res", "coarse_period_nm", "n_instances",
           "ler_3sigma_nm", "ler_xi_nm", "wide_res_nm", "placement", "present",
           "ref_blur_nm", "wide_blur_nm", "ref_dose", "wide_dose",
           "wide_charging", "wide_scan_distortion_px", "wide_astig_sigma_nm",
           "wide_astig_angle_deg", "wide_barrel_k1", "wide_vignette", "wide_gamma",
           "ref_drift_nm", "wide_drift_nm",
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
    ap.add_argument("--phase2", action="store_true",
                    help="Phase 2 mode: sample the zoom ratio in [8,12] and use "
                         "signed rotation (+/-5 deg). Physical render-mode only.")
    ap.add_argument("--scale-range", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="explicit zoom range (overrides --phase2's [8,12])")
    ap.add_argument("--absent-fraction", type=float, default=0.0,
                    help="fraction of pairs with no true instance (Set C); "
                         "decided per-seed, deterministically. Phase 2 mix ~0.22")
    ap.add_argument("--scan-distortion", type=float, default=0.0,
                    help="max scan-distortion warp amplitude in wide px (Set B); "
                         "--phase2 defaults it to 6.0")
    ap.add_argument("--optical-aberrations", action="store_true",
                    help="enable astigmatism/barrel/vignette/gamma (Set B "
                         "realism); --phase2 enables this by default")
    a = ap.parse_args()

    scale_range = None
    signed_rotation = False
    scan_distortion_max = a.scan_distortion
    optical_aberrations = a.optical_aberrations
    if a.phase2:
        scale_range, signed_rotation = SCALE_RANGE, True
        if scan_distortion_max == 0.0:
            scan_distortion_max = 6.0
        optical_aberrations = True
    if a.scale_range is not None:
        scale_range = tuple(a.scale_range)

    def _is_absent(seed):
        if a.absent_fraction <= 0.0:
            return False
        return float(np.random.default_rng(seed ^ 0xAB5E17).random()) < a.absent_fraction

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
    jobs = [(s, style, a.out, N_PX, a.render_mode, scale_range, signed_rotation,
             _is_absent(s), scan_distortion_max, optical_aberrations) for s in seeds]
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

    # In Phase 2 the zoom is per-pair (see the `scale` column); the fixed
    # wide_px_nm/fov/footprint below describe the Phase 1 nominal only.
    (a.out / "dataset_meta.json").write_text(json.dumps(dict(
        generator="driftsense", version=VERSION, n_px=N_PX,
        ref_px_nm=1.0, wide_px_nm=10.0,
        ref_fov_nm=N_PX * 1.0, wide_fov_nm=N_PX * 10.0,
        footprint_px=int(N_PX * 1.0 / 10.0),
        phase2=bool(scale_range is not None or signed_rotation),
        scale_range=list(scale_range) if scale_range else None,
        signed_rotation=signed_rotation,
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



