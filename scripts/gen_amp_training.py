"""Generate training pairs from Applied Materials' OWN Phase 2 generator.

Why this exists: our net is trained on our generator and overfits it -- on the
organizers' real 20-pair sample the classical path localises 13/14 present pairs
while the net misses ~6. Phase 1 hit the same cross-generator gap and closed it
by training on THEIR distribution (71% -> 95.5%). This regenerates that fix for
Phase 2 using the generator source they shared.

COMPLIANCE (the no-appeal DQ line, stated here so it cannot be missed):
  * ALLOWED and what this script does -- run their generator SOURCE with NEW
    seeds and NEW randomised poses to synthesise NEW pairs.
  * FORBIDDEN -- training on the 20 provided sample pairs (p001-p020). This
    script never reads them. They stay a validation fold, used only to score.
  * Our own generator (driftsense/, generate_dataset.py) remains the shipped
    deliverable for the generator/citations rubric bucket; this changes nothing
    there.

Their `generate_phase2_samples.py` hard-codes a 20-row PLAN and exposes only
--seed, so re-seeding reuses the same 20 (zoom, theta) combos -- noise diversity
but no pose diversity, useless for training a pose-robust net. So this calls
their `generate_phase2_sample(arch, params, rng)` directly with randomised
architecture / zoom / theta / severity, which their own spec sanctions ("keep
the sampling path for scaling to 200 pairs later").

Writes directly in the labels.csv schema driftmatch/data.py reads
(ref_path, wide_path, gt_x, gt_y), so no conversion pass is needed.

    python scripts/gen_amp_training.py --gen-dir <their generator dir> \
        --out data/amp_train --pairs 900 --minutes 180 --workers 3
"""
from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys
import time
from multiprocessing import Pool

import numpy as np

# Their pipeline holds one very large canvas at a time (a z=12 canvas is
# >13000x13000 px at 1 nm/px -- their spec warns about peak memory), so keep the
# worker count low and let each worker release its canvas before the next.
DEFAULT_WORKERS = 3

# Severity mix. Their SEVERITY dict is the disclosed-category / undisclosed-
# parameter ladder used for Set B. Weighted toward the low end so most pairs
# stay learnable -- a training set dominated by severity 4 teaches the net
# mostly noise.
SEV_WEIGHTS = {0: 0.34, 1: 0.26, 2: 0.20, 3: 0.13, 4: 0.07}

_GEN = {}


def _init(gen_dir: str):
    """Import their generator once per worker."""
    if str(gen_dir) not in sys.path:
        sys.path.insert(0, str(gen_dir))
    from src.phase2_pipeline import (Phase2Params, generate_phase2_sample,
                                     ZOOM_MIN, ZOOM_MAX, THETA_MIN, THETA_MAX)
    from src.presets import PRESETS
    sys.path.insert(0, str(pathlib.Path(gen_dir)))
    # SEVERITY lives in their top-level sample script, not in src/.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "amp_samples", str(pathlib.Path(gen_dir) / "generate_phase2_samples.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _GEN.update(dict(Phase2Params=Phase2Params,
                     generate_phase2_sample=generate_phase2_sample,
                     SEVERITY=mod.SEVERITY, PRESETS=PRESETS,
                     ZOOM_MIN=ZOOM_MIN, ZOOM_MAX=ZOOM_MAX,
                     THETA_MIN=THETA_MIN, THETA_MAX=THETA_MAX))


def _one(job):
    """Generate a single pair. Returns (idx, arrays, meta) or (idx, None, reason)."""
    idx, seed = job
    Phase2Params = _GEN["Phase2Params"]
    generate_phase2_sample = _GEN["generate_phase2_sample"]
    SEVERITY = _GEN["SEVERITY"]
    archs = sorted(_GEN["PRESETS"].keys())

    rng = np.random.default_rng(seed)
    arch = archs[int(rng.integers(len(archs)))]
    zoom = float(rng.uniform(_GEN["ZOOM_MIN"], _GEN["ZOOM_MAX"]))
    theta = float(rng.uniform(_GEN["THETA_MIN"], _GEN["THETA_MAX"]))
    levels = sorted(SEV_WEIGHTS)
    sev = int(rng.choice(levels, p=[SEV_WEIGHTS[k] for k in levels]))

    params = Phase2Params(zoom=zoom, theta_deg=theta, present=True,
                          boundary_bias=0.70, **SEVERITY[sev])
    try:
        s = generate_phase2_sample(arch, params, rng)
    except Exception as exc:                      # a failed crop search etc.
        return idx, None, f"exception: {type(exc).__name__}: {exc}"

    v = s.get("verify") or {}
    # Their own label-verification gate: the rendered template must correlate
    # back onto the label. Dropping failures means every pair we train on has a
    # provably hittable label -- free training-data quality.
    if not v.get("ok", False):
        return idx, None, f"verify_failed err={v.get('err_px')} margin={v.get('margin')}"

    gt = s["gt"]
    gx = float(gt["x"] if isinstance(gt, dict) else gt[0])
    gy = float(gt["y"] if isinstance(gt, dict) else gt[1])
    meta = dict(arch=arch, zoom=round(zoom, 4), theta=round(theta, 4), sev=sev,
                gt_x=gx, gt_y=gy, seed=seed,
                verify_err_px=v.get("err_px"), verify_margin=v.get("margin"))
    return idx, (s["reference_img"], s["search_img"]), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True,
                    help="their generator dir (contains src/ and generate_phase2_samples.py)")
    ap.add_argument("--out", default="data/amp_train")
    ap.add_argument("--pairs", type=int, default=900)
    ap.add_argument("--minutes", type=float, default=180.0, help="hard time box")
    ap.add_argument("--seed-base", type=int, default=770000)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = ap.parse_args()

    import cv2                                     # their generator's writer dep

    out = pathlib.Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    labels = out / "labels.csv"

    cols = ["pair_id", "ref_path", "wide_path", "gt_x", "gt_y", "present",
            "scale", "rotation_deg", "architecture", "severity",
            "verify_err_px", "verify_margin", "seed"]
    fh = labels.open("w", newline="")
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    fh.flush()

    jobs = [(i, args.seed_base + i * 7919) for i in range(args.pairs)]
    t0 = time.time()
    kept = dropped = 0
    drop_reasons = {}

    print(f"generating up to {args.pairs} pairs, time box {args.minutes:.0f} min, "
          f"{args.workers} workers -> {out}", flush=True)

    with Pool(args.workers, initializer=_init, initargs=(args.gen_dir,)) as pool:
        for idx, arrays, meta in pool.imap_unordered(_one, jobs, chunksize=1):
            if arrays is None:
                dropped += 1
                key = str(meta).split(" ")[0]
                drop_reasons[key] = drop_reasons.get(key, 0) + 1
            else:
                ref_img, srch_img = arrays
                pid = f"a{idx:05d}"
                rp = f"images/{pid}_ref.png"
                sp = f"images/{pid}_wide.png"
                cv2.imwrite(str(out / rp), ref_img)
                cv2.imwrite(str(out / sp), srch_img)
                w.writerow(dict(pair_id=pid, ref_path=rp, wide_path=sp,
                                gt_x=meta["gt_x"], gt_y=meta["gt_y"], present=1,
                                scale=meta["zoom"], rotation_deg=meta["theta"],
                                architecture=meta["arch"], severity=meta["sev"],
                                verify_err_px=meta["verify_err_px"],
                                verify_margin=meta["verify_margin"],
                                seed=meta["seed"]))
                fh.flush()                          # incremental: a timeout still leaves usable data
                kept += 1

            el = (time.time() - t0) / 60.0
            done = kept + dropped
            if done % 10 == 0 or el >= args.minutes:
                rate = done / max(el, 1e-6)
                print(f"  {done} done ({kept} kept, {dropped} dropped) "
                      f"{el:.1f} min, {rate:.1f} pairs/min", flush=True)
            if el >= args.minutes:
                print("time box reached -- stopping early (data written so far is usable)",
                      flush=True)
                pool.terminate()
                break

    fh.close()
    el = (time.time() - t0) / 60.0
    print(f"\nDONE: kept {kept}, dropped {dropped}, {el:.1f} min -> {out}")
    print(f"drop reasons: {drop_reasons}")
    print(f"seed range: {args.seed_base} .. {args.seed_base + args.pairs * 7919}")


if __name__ == "__main__":
    main()
