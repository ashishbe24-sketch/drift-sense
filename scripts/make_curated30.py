#!/usr/bin/env python
"""Build the 30 curated evaluation cases, each with a written rationale.

The evaluator was explicit that the *quality of the test cases* is graded, not
just the score on them: "give us top 30 samples justifying what is unique about
this... we don't care that the results are average or bad". He also confirmed a
reference may be reused across several wide views, which lets us build ladders
that isolate one variable at a time -- far more informative than 30 unrelated
pairs.

Layout: three 6-step ladders (18 pairs) plus 12 individually chosen cases.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse, csv, json, pathlib, sys
from dataclasses import replace

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from driftsense.layout import (LineArray, add_landmark_lattice, build_dram,
                               build_finfet, make_edge_roughness)
from driftsense.physics import CaptureParams, render
from driftsense.raster import make_pair
from driftsense.sampling import build_layout, effective_resolution_nm, sample_spec


def with_roughness(lay, seed, sigma_nm=1.0, xi_nm=10.0):
    """Attach line-edge roughness, as the sampled pairs get automatically."""
    r = make_edge_roughness(np.random.default_rng(seed), sigma_nm=sigma_nm,
                            xi_nm=xi_nm)
    for a in lay.arrays:
        if isinstance(a, LineArray):
            a.roughness = r
    return lay

N = 1000
TARGET = (6000.0, 6000.0)
EXTENT = 14000.0

BASE_REF = CaptureParams(px_nm=1.0, probe_sigma_nm=1.0, drift_nm=2.0,
                         vibration_nm=0.7, dose_e_per_grey=40.0, read_sigma=1.0)
BASE_WIDE = CaptureParams(px_nm=10.0, probe_sigma_nm=1.3, defocus_sigma_nm=10.0,
                          drift_nm=5.0, vibration_nm=2.0,
                          dose_e_per_grey=6.0, read_sigma=2.0)


def emit(out, cases):
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, c in enumerate(cases):
        pid = f"C{i:02d}"
        lay = c["layout"]
        off = (c["gt"][0] - N / 2.0, c["gt"][1] - N / 2.0)
        mref, mwide, gt = make_pair(lay, TARGET, off, n_px=N)
        rng = np.random.default_rng(c.get("noise_seed", 4242 + i))
        ref = render(mref, c["ref"], rng)
        wide = render(mwide, c["wide"], rng)
        assert ref.shape == (N, N) and wide.shape == (N, N)
        Image.fromarray(ref).save(out / "images" / f"{pid}_ref.png", optimize=True)
        Image.fromarray(wide).save(out / "images" / f"{pid}_wide.png", optimize=True)
        rows.append(dict(
            pair_id=pid,
            ref_path=f"images/{pid}_ref.png",
            wide_path=f"images/{pid}_wide.png",
            gt_x=round(gt[0], 4), gt_y=round(gt[1], 4),
            group=c["group"], variable=c["variable"], step=c["step"],
            style=lay.meta.get("style"),
            pitch_nm=round(lay.meta.get("wl_pitch_nm") or lay.meta.get("fin_pitch_nm"), 2),
            rotation_deg=round(lay.meta.get("angle_deg", 0.0), 3),
            landmark_nm=round(lay.meta.get("landmark", {}).get("size_nm", 0), 1),
            wide_res_nm=round(effective_resolution_nm(c["wide"]), 2),
            wide_blur_nm=round(c["wide"].total_blur_nm(), 2),
            wide_dose=round(c["wide"].dose_e_per_grey, 3),
            rationale=c["rationale"],
        ))
    with (out / "labels.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    md = ["# The 30 curated cases\n",
          "Each case exists to test one specific thing. Ladders hold everything",
          "fixed except the named variable, so a failure can be attributed.\n"]
    for r in rows:
        md.append(f"### {r['pair_id']} - {r['group']} (step {r['step']})\n")
        md.append(f"**Variable:** {r['variable']}  \n"
                  f"**Truth:** ({r['gt_x']}, {r['gt_y']}) px  \n"
                  f"**Layout:** {r['style']}, pitch {r['pitch_nm']} nm, "
                  f"rotation {r['rotation_deg']} deg, landmark {r['landmark_nm']} nm  \n"
                  f"**Wide view:** blur {r['wide_blur_nm']} nm -> effective "
                  f"resolution {r['wide_res_nm']} nm, dose {r['wide_dose']}\n")
        md.append(f"{r['rationale']}\n")
    (out / "CASES.md").write_text("\n".join(md), encoding="utf-8")
    print(f"{len(rows)} curated cases -> {out}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/curated30"))
    a = ap.parse_args()
    cases = []

    # ---- Ladder 1: dose. Everything fixed but the wide-view electron dose. --
    lay1 = with_roughness(build_dram(extent_nm=EXTENT, wl_pitch_nm=130.0, bl_pitch_nm=104.0,
                      angle_deg=1.8, landmark_xy_nm=TARGET, landmark="plus"), 101)
    lay1.meta["landmark"]["size_nm"] = 300.0
    for k, dose in enumerate([24.0, 12.0, 6.0, 3.0, 1.5, 0.8]):
        cases.append(dict(
            group="L1 dose ladder", variable="wide-view dose (electrons/grey)",
            step=k + 1, layout=lay1, gt=(560.0, 470.0), ref=BASE_REF,
            wide=replace(BASE_WIDE, dose_e_per_grey=dose),
            rationale=(
                "Identical layout, identical reference, identical placement; only the "
                f"survey dose changes ({dose} e/grey, i.e. a noise ratio of "
                f"{np.sqrt(40.0/dose):.1f}x versus the reference). This isolates pure "
                "photon-starvation: where the method breaks along this ladder is its "
                "noise limit, with no other variable to confound it.")))

    # ---- Ladder 2: periodicity. Pitch shrinks toward and past Nyquist. ------
    for k, pitch in enumerate([300.0, 190.0, 120.0, 70.0, 42.0, 26.0]):
        lay = with_roughness(build_dram(extent_nm=EXTENT, wl_pitch_nm=pitch,
                         bl_pitch_nm=pitch * 0.8, wl_width_nm=pitch * 0.34,
                         bl_width_nm=pitch * 0.28, contact_r_nm=pitch * 0.16,
                         contact_core_r_nm=pitch * 0.062, angle_deg=1.2,
                         landmark_xy_nm=TARGET, landmark="plus"), 102)
        lay.meta["landmark"]["size_nm"] = 300.0
        wide_px = pitch / 10.0
        cases.append(dict(
            group="L2 periodicity ladder", variable="array pitch (nm)", step=k + 1,
            layout=lay, gt=(430.0, 605.0), ref=BASE_REF, wide=BASE_WIDE,
            rationale=(
                f"Pitch {pitch:.0f} nm gives {wide_px:.1f} px per period in the wide "
                "view. " + ("Above Nyquist, the array resolves normally."
                            if wide_px >= 3 else
                            "At or below the two-pixel Nyquist limit the array folds "
                            "into moire rather than resolving, so the periodic "
                            "background stops being usable evidence and the landmark "
                            "carries the entire signal.") +
                " Walking the pitch down while holding the landmark fixed separates "
                "'cannot resolve the array' from 'cannot find the landmark'.")))

    # ---- Ladder 3: landmark saliency against a fixed blur. ------------------
    for k, size in enumerate([340.0, 220.0, 140.0, 90.0, 55.0, 32.0]):
        lay = with_roughness(build_dram(extent_nm=EXTENT, wl_pitch_nm=120.0, bl_pitch_nm=96.0,
                         angle_deg=2.6, landmark_xy_nm=TARGET, landmark="plus"), 103)
        lay.shapes[-3].w_nm = lay.shapes[-3].h_nm = min(1.22 * size, 470.0)
        lay.shapes[-2].w_nm, lay.shapes[-2].h_nm = size, size * 0.32
        lay.shapes[-1].h_nm, lay.shapes[-1].w_nm = size, size * 0.32
        lay.meta["landmark"]["size_nm"] = size
        res = effective_resolution_nm(BASE_WIDE)
        cases.append(dict(
            group="L3 saliency ladder", variable="landmark size (nm)", step=k + 1,
            layout=lay, gt=(500.0, 500.0), ref=BASE_REF, wide=BASE_WIDE,
            rationale=(
                f"Landmark {size:.0f} nm against an effective wide resolution of "
                f"{res:.0f} nm, a ratio of {size/res:.1f}. " +
                ("Comfortably resolved." if size / res >= 8 else
                 "Marginal: the aperiodic feature is approaching the resolution limit."
                 if size / res >= 3 else
                 "Below the floor - the disambiguating feature is not recorded in the "
                 "wide view at all, so this pair is unsolvable by construction and is "
                 "included deliberately as the endpoint of the ladder.") +
                " Every other parameter is held fixed, so this measures exactly how "
                "much aperiodic signal the method needs.")))

    # ---- 12 individually chosen cases ---------------------------------------
    singles = [
        (91004, "S1 off-centre placement",
         "Target placed far from centre, well outside the stage-accuracy prior. "
         "Tests whether the method genuinely searches or has learned to bias its "
         "prediction toward the image centre, which the stage prior would reward "
         "on most pairs."),
        (91011, "S2 heavy rotation",
         "Rotation near the top of the range the problem statement allows. Over a "
         "100 px footprint this displaces corners by several pixels, so a "
         "translation-only matcher degrades here while a rotation-aware one does not."),
        (91025, "S3 charging artefacts",
         "Wide view carries charge-induced bands and fast-scan streaks. These are "
         "large, smooth, non-structural intensity changes that defeat any matcher "
         "keying on absolute grey level rather than local structure."),
        (91033, "S4 FinFET one-dimensional periodicity",
         "Parallel fins are periodic along one axis only, so correlation is sharply "
         "peaked across the fins and nearly flat along them. Error should be "
         "strongly anisotropic - a single scalar accuracy figure hides this."),
        (91042, "S5 aliased array, salient landmark",
         "Array folds into moire while the landmark stays well resolved. The correct "
         "behaviour is to ignore the background entirely; a method that weights all "
         "pixels equally is actively misled by the aliased texture."),
        (91050, "S6 low dose and heavy defocus together",
         "Both degradations at once. Included because failure modes are not "
         "independent: blur removes the high-frequency content that would otherwise "
         "survive the noise."),
        (91066, "S7 large landmark, little context",
         "The aperiodic feature dominates the reference, leaving little periodic "
         "surround. Easy to localise coarsely but the sub-pixel estimate rests on a "
         "few long edges rather than a dense lattice."),
        (91078, "S8 minimal landmark, dense context",
         "The mirror of S7: a small landmark inside a strongly periodic field, which "
         "is the classic many-near-identical-peaks case the centre tie-break exists "
         "to resolve."),
        (91085, "S9 strong drift shear",
         "Slow stage creep during the raster shears the wide field. A rigid matcher "
         "accumulates a systematic offset that grows down the image."),
        (91097, "S10 vibration serration",
         "Per-scanline displacement serrates edges that should be straight, adding "
         "high-frequency error exactly where sub-pixel refinement takes its signal."),
        (91103, "S11 coarse pitch, near-uniform background",
         "Very relaxed pitch leaves large flat areas with little texture to lock "
         "onto away from the landmark."),
        (91119, "S12 below-floor failure case",
         "Deliberately unsolvable: the landmark falls under the visibility floor for "
         "this blur. Included as the honest failure case - the useful output here is "
         "a low-confidence flag, not a coordinate."),
    ]
    for seed, title, why in singles:
        spec = sample_spec(seed, n_px=N)
        cases.append(dict(
            group=title, variable="selected case", step=1,
            layout=build_layout(spec, TARGET, EXTENT),
            gt=(spec.gt_x, spec.gt_y), ref=spec.ref, wide=spec.wide,
            noise_seed=seed ^ 0xA11CE, rationale=why))

    rows = emit(a.out, cases)
    (a.out / "seeds.txt").write_text(
        "\n".join(str(s) for s, _, _ in singles) + "\n")
    print(f"groups: {len(set(r['group'] for r in rows))}, "
          f"ladder steps: {sum(1 for r in rows if r['group'].startswith('L'))}, "
          f"singles: {sum(1 for r in rows if r['group'].startswith('S'))}")


if __name__ == "__main__":
    main()

