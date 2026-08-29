"""Randomisation policy: one seed -> one fully specified pair.

Ranges are deliberately wider than any single plausible tool setting. The
evaluation set comes from Applied Materials' own generator, so our distribution
has to contain theirs; breadth costs nothing and protects the inference score.

Everything sampled here is recorded in the manifest, so any pair can be
reproduced exactly and any accuracy figure can be re-derived.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .layout import (BUILDERS, CLEAR_FACTOR, CLEAR_MAX_NM, LANDMARKS, LineArray,
                     add_landmark_lattice, make_edge_roughness, _lattice_offsets)
from .physics import CaptureParams

# --- pitch regimes ---------------------------------------------------------
# Nyquist in the 10 nm/px wide view needs >= 2 px per period. A modern DRAM
# cell array (24-42 nm) is at or below that and aliases into moire; upper metal
# (70-320 nm) resolves comfortably. Applied Materials' own example figures sit
# near 110 nm. Both regimes are generated: the aliased one is physically real
# and is where honest failure cases come from.
REGIMES = {
    "resolved": dict(weight=0.60, pitch_nm=(70.0, 320.0)),
    "coarse":   dict(weight=0.25, pitch_nm=(42.0, 70.0)),
    "aliased":  dict(weight=0.15, pitch_nm=(24.0, 42.0)),
}

# Landmark saliency is drawn relative to the wide view's *effective* resolution,
# not as an absolute size. Pixel size alone is the wrong floor: with defocus up
# to 28 nm a 60 nm landmark smears below detectability even though it spans six
# pixels. Measuring rendered landmark SNR against a landmark-free render showed
# 34% of pairs unsolvable under absolute sizing, against an intended 8%.
#
# Effective resolution combines the pixel and the blur FWHM in quadrature, and
# the size/resolution ratio predicts detectability well:
#     ratio < 2   -> 100% unsolvable      ratio 4-8  ->  20% unsolvable
#     ratio 2-4   ->  61% unsolvable      ratio > 8  ->   0% unsolvable
# Bands are set from the measured curve above, not guessed: the below_floor
# class alone very nearly fills the 8% unsolvable budget, so "hard" has to sit
# above ratio 5 rather than straddle it.
LANDMARK_ABS_NM = (30.0, 400.0)     # must span a few px, must fit the reference
DIFFICULTY = (
    ("below_floor", 0.08, (0.8, 2.5)),   # unsolvable by construction, labelled
    ("hard",        0.25, (5.0, 8.0)),
    ("normal",      0.67, (8.0, 20.0)),
)


def effective_resolution_nm(cap) -> float:
    """Smallest feature the capture can resolve: pixel and blur FWHM in
    quadrature."""
    return float(np.hypot(cap.px_nm, 2.355 * cap.total_blur_nm()))

# Published inspection-stage accuracy is <1.5 um against a 10 um field, so the
# target lands near the centre. That is what makes "closest to centre" the
# correct tie-break, and it is modelled rather than assumed. A uniform minority
# guards against a model degenerating to "always predict centre" -- and against
# the possibility that their generator places uniformly.
STAGE_SIGMA_UM = (0.7, 2.0)
UNIFORM_PLACEMENT_FRACTION = 0.20

ROTATION_DEG = (0.0, 5.0)       # webinar: "only 1 to 3 degrees", up to 5
SCALE_JITTER = 0.20             # webinar: shrink or grow a polygon by 20%

# Phase 2: the zoom ratio is no longer fixed at 10x. The addendum draws it
# uniformly in [8, 12], different per pair, and asks it to be recovered. The
# reference stays 1 nm/px, so the wide pixel size in nm *is* the scale.
SCALE_RANGE = (8.0, 12.0)

# Fraction of pairs where the landmark itself recurs on a coarse lattice,
# producing several near-identical candidates in one wide view. The evaluator
# confirmed their test data contains these: "we will also give repeated areas
# in the image". 1-4 um puts 6 to 100 instances inside a 10 um field.
REPEAT_FRACTION = 0.40
COARSE_PERIOD_NM = (1000.0, 4000.0)
DECOY_JITTER_NM = 12.0          # decoys are near-identical, not bit-identical

# Line-edge roughness. Reported for modern lithography as 3-sigma LWR of
# 2.6-3.5 nm with correlation length 7-13 nm (Cutler et al., J. Micro/Nanopattern.
# Mater. Metrol. 20(1) 010901, 2021). 3-sigma 2.6-3.5 nm is 1-sigma 0.87-1.17 nm.
LER_SIGMA_NM = (0.6, 1.3)
LER_XI_NM = (7.0, 13.0)


@dataclass
class PairSpec:
    seed: int
    style: str
    regime: str
    pitch_nm: float
    rotation_deg: float
    landmark: str
    landmark_size_nm: float
    difficulty: str
    size_over_res: float
    coarse_period_nm: float
    n_instances: int
    ler_sigma_nm: float
    ler_xi_nm: float
    placement: str
    gt_x: float
    gt_y: float
    ref: CaptureParams
    wide: CaptureParams
    present: int = 1          # Phase 2 Set C: 0 = the reference is absent (found=0)

    def manifest_row(self):
        r, w = self.ref, self.wide
        return dict(
            seed=self.seed, style=self.style, regime=self.regime,
            pitch_nm=round(self.pitch_nm, 3),
            rotation_deg=round(self.rotation_deg, 4),
            scale=round(self.wide.px_nm / self.ref.px_nm, 4),
            landmark=self.landmark,
            landmark_size_nm=round(self.landmark_size_nm, 2),
            landmark_wide_px=round(self.landmark_size_nm / w.px_nm, 2),
            difficulty=self.difficulty,
            size_over_res=round(self.size_over_res, 3),
            coarse_period_nm=round(self.coarse_period_nm, 1),
            n_instances=self.n_instances,
            ler_3sigma_nm=round(3 * self.ler_sigma_nm, 2),
            ler_xi_nm=round(self.ler_xi_nm, 2),
            wide_res_nm=round(effective_resolution_nm(w), 2),
            placement=self.placement, present=self.present,
            gt_x=round(self.gt_x, 4), gt_y=round(self.gt_y, 4),
            ref_blur_nm=round(r.total_blur_nm(), 3),
            wide_blur_nm=round(w.total_blur_nm(), 3),
            ref_dose=r.dose_e_per_grey, wide_dose=w.dose_e_per_grey,
            wide_charging=round(w.charging, 3),
            wide_scan_distortion_px=round(w.scan_distortion_px, 3),
            wide_astig_sigma_nm=round(w.astig_extra_sigma_nm, 3),
            wide_astig_angle_deg=round(w.astig_angle_deg, 2),
            wide_barrel_k1=round(w.barrel_k1, 4),
            wide_vignette=round(w.vignette_strength, 3),
            wide_gamma=round(w.gamma, 3),
            ref_drift_nm=round(r.drift_nm, 2), wide_drift_nm=round(w.drift_nm, 2),
            ref_vib_nm=round(r.vibration_nm, 2), wide_vib_nm=round(w.vibration_nm, 2),
        )


def _pick_regime(rng):
    names = list(REGIMES)
    w = np.array([REGIMES[n]["weight"] for n in names], float)
    return names[int(rng.choice(len(names), p=w / w.sum()))]


def sample_spec(seed: int, style: str | None = None, n_px: int = 1000,
                wide_px_nm: float = 10.0, scale_range: tuple | None = None,
                signed_rotation: bool = False, absent: bool = False,
                scan_distortion_max: float = 0.0,
                optical_aberrations: bool = False) -> PairSpec:
    """Draw every parameter of one pair from `seed` alone.

    Phase 2 options, both gated so the default (Phase 1) call is byte-identical:
      scale_range     : if given (e.g. (8, 12)), the wide pixel size -- i.e. the
                        zoom ratio -- is drawn from it per pair instead of the
                        fixed `wide_px_nm`. This draw is only taken when the flag
                        is set, so Phase 1 seeds are undisturbed.
      signed_rotation : if True, rotation is drawn in +/-ROTATION_DEG[1] instead
                        of [0, ROTATION_DEG[1]]. One uniform() call either way,
                        so the rng stream position is identical to Phase 1 -- only
                        the value changes.
    """
    rng = np.random.default_rng(seed)

    # Phase 2 zoom: drawn first, before any other parameter, so the [8,12] scale
    # feeds the placement geometry below. Gated: no draw when scale_range is None.
    if scale_range is not None:
        wide_px_nm = float(rng.uniform(*scale_range))

    style = style or ("dram" if rng.random() < 0.5 else "finfet")
    regime = _pick_regime(rng)
    lo, hi = REGIMES[regime]["pitch_nm"]
    pitch = float(rng.uniform(lo, hi))
    rot = float(rng.uniform(-ROTATION_DEG[1], ROTATION_DEG[1]) if signed_rotation
                else rng.uniform(*ROTATION_DEG))
    ler_sigma = float(rng.uniform(*LER_SIGMA_NM))
    ler_xi = float(rng.uniform(*LER_XI_NM))
    landmark = "plus" if rng.random() < 0.6 else "pad"

    # Coarse-periodic repeats of the landmark. Straps and stitch regions recur
    # every 32-128 cells, i.e. every 1-4 um at DRAM pitches, so a 10 um view
    # holds several near-identical candidates and the centre tie-break becomes
    # the only thing that resolves them.
    repeats = rng.random() < REPEAT_FRACTION
    coarse_period_nm = (float(rng.uniform(*COARSE_PERIOD_NM)) if repeats else 0.0)

    half = n_px / 2.0
    margin = 60.0
    if repeats:
        # The stated rule is "return the match closest to the search image's
        # centre". For our label to agree with that rule, the true site must be
        # the centre-most instance -- so it is kept within half a period of the
        # centre. Without this the rule would select a decoy and our ground
        # truth would be inconsistent with the problem statement.
        placement = "stage_prior_periodic"
        p_px = coarse_period_nm / wide_px_nm
        lim = min(0.45 * p_px, n_px / 2.0 - margin)
        sig_px = min(float(rng.uniform(*STAGE_SIGMA_UM)) * 1000.0 / wide_px_nm,
                     lim / 2.0)
        while True:
            gx = half + float(rng.normal(0, sig_px))
            gy = half + float(rng.normal(0, sig_px))
            if abs(gx - half) <= lim and abs(gy - half) <= lim:
                break
    elif rng.random() < UNIFORM_PLACEMENT_FRACTION:
        placement = "uniform"
        gx = float(rng.uniform(margin, n_px - margin))
        gy = float(rng.uniform(margin, n_px - margin))
    else:
        placement = "stage_prior"
        sig_px = float(rng.uniform(*STAGE_SIGMA_UM)) * 1000.0 / wide_px_nm
        while True:
            gx = half + float(rng.normal(0, sig_px))
            gy = half + float(rng.normal(0, sig_px))
            if margin <= gx <= n_px - margin and margin <= gy <= n_px - margin:
                break

    # Reference: slow, careful characterisation capture taken earlier.
    ref = CaptureParams(
        px_nm=1.0,
        probe_sigma_nm=float(rng.uniform(0.8, 1.6)),
        drift_nm=float(rng.uniform(0.0, 4.0)),
        drift_angle_deg=float(rng.uniform(0, 360)),
        vibration_nm=float(rng.uniform(0.0, 1.5)),
        dose_e_per_grey=float(rng.uniform(25.0, 60.0)),
        read_sigma=float(rng.uniform(0.6, 1.6)),
    )

    # Wide: fast survey grab. Short dwell gives 1.5-4x the reference noise;
    # no time to refocus gives a large defocus term; and raising beam current
    # to compensate for the short dwell enlarges the probe -- so blur and dose
    # are coupled rather than drawn independently.
    dose_ratio = float(rng.uniform(2.5, 16.0))
    wide_dose = ref.dose_e_per_grey / dose_ratio
    current_boost = float(np.clip(np.sqrt(dose_ratio) / 2.5, 1.0, 2.2))
    # Phase 2 Set B scan distortion, on the wide (survey) capture only, so it is
    # a *relative* warp between ref and wide -- the thing a rigid template cannot
    # absorb. Gated: no draw unless scan_distortion_max is set, so Phase 1 seeds
    # are byte-identical. ~60% of enabled pairs carry some distortion.
    if scan_distortion_max > 0:
        scan_dist = (float(rng.uniform(0.0, scan_distortion_max))
                     if rng.random() < 0.6 else 0.0)
    else:
        scan_dist = 0.0
    # Optical aberrations (astigmatism, barrel, vignette, gamma) -- Set B
    # realism, previously spec'd (README) but never implemented. Each is
    # independently gated at ~40-50% so training sees a mix of clean and
    # aberrated pairs, not aberrations-always. All zero unless enabled, so
    # Phase 1 seeds are unaffected.
    if optical_aberrations:
        astig_sigma = float(rng.uniform(1.0, 5.0)) if rng.random() < 0.4 else 0.0
        astig_angle = float(rng.uniform(0, 180)) if astig_sigma > 0 else 0.0
        barrel_k1 = (float(rng.uniform(-0.06, 0.06)) if rng.random() < 0.4 else 0.0)
        vignette = float(rng.uniform(0.1, 0.35)) if rng.random() < 0.5 else 0.0
        gamma = float(rng.uniform(0.85, 1.2)) if rng.random() < 0.5 else 1.0
    else:
        astig_sigma = astig_angle = barrel_k1 = vignette = 0.0
        gamma = 1.0
    wide = CaptureParams(
        px_nm=wide_px_nm,
        probe_sigma_nm=ref.probe_sigma_nm * current_boost,
        # Capped at 20 nm: beyond that the effective resolution exceeds what a
        # landmark can compensate for without filling the reference field, and
        # the size clamp would silently fight the difficulty policy.
        defocus_sigma_nm=float(rng.uniform(4.0, 20.0)),
        drift_nm=float(rng.uniform(0.0, 14.0)),
        drift_angle_deg=float(rng.uniform(0, 360)),
        vibration_nm=float(rng.uniform(0.0, 8.0)),
        charging=float(rng.uniform(0.0, 4.0)) if rng.random() < 0.5 else 0.0,
        streak_rate=float(rng.uniform(0.0, 3.0)) if rng.random() < 0.3 else 0.0,
        scan_distortion_px=scan_dist,
        astig_extra_sigma_nm=astig_sigma, astig_angle_deg=astig_angle,
        barrel_k1=barrel_k1, vignette_strength=vignette, gamma=gamma,
        dose_e_per_grey=wide_dose,
        read_sigma=float(rng.uniform(1.0, 3.5)),
    )

    # Landmark saliency, drawn against what the wide view can actually resolve.
    # Difficulty is chosen first, so the fraction of unsolvable pairs is set by
    # policy instead of falling out of unrelated draws.
    res_nm = effective_resolution_nm(wide)
    names = [d[0] for d in DIFFICULTY]
    w = np.array([d[1] for d in DIFFICULTY], float)
    k = int(rng.choice(len(names), p=w / w.sum()))
    difficulty, ratio_lo, ratio_hi = names[k], *DIFFICULTY[k][2]
    ratio = float(rng.uniform(ratio_lo, ratio_hi))
    size = float(np.clip(ratio * res_nm, *LANDMARK_ABS_NM))

    # Absent (Set C): the reference has no true instance in the wide view. The
    # placement draws above are kept (so the rng stream is unchanged) but the
    # ground truth is marked absent and the coordinates are set to a -1 sentinel;
    # generate_one renders the wide from the same architecture with the landmark
    # removed. present=0 is the rejection ground truth.
    if absent:
        placement = "absent"
        gx = gy = -1.0

    return PairSpec(seed=seed, style=style, regime=regime, pitch_nm=pitch,
                    rotation_deg=rot, landmark=landmark, landmark_size_nm=size,
                    difficulty=difficulty, size_over_res=size / res_nm,
                    coarse_period_nm=coarse_period_nm,
                    ler_sigma_nm=ler_sigma, ler_xi_nm=ler_xi,
                    n_instances=len(_lattice_offsets(coarse_period_nm or None, 7000.0)),
                    placement=placement, present=(0 if absent else 1),
                    gt_x=gx, gt_y=gy, ref=ref, wide=wide)


def build_layout(spec: PairSpec, target_nm=(6000.0, 6000.0), extent_nm=14000.0):
    """Instantiate the layout described by a spec."""
    rng = np.random.default_rng(spec.seed ^ 0x5EED)
    kw = dict(extent_nm=extent_nm, angle_deg=spec.rotation_deg,
              landmark_xy_nm=target_nm, landmark=spec.landmark)

    j = lambda: 1.0 + float(rng.uniform(-SCALE_JITTER, SCALE_JITTER))
    if spec.style == "dram":
        kw.update(wl_pitch_nm=spec.pitch_nm,
                  bl_pitch_nm=spec.pitch_nm * float(rng.uniform(0.72, 1.0)),
                  wl_width_nm=spec.pitch_nm * 0.34 * j(),
                  bl_width_nm=spec.pitch_nm * 0.28 * j(),
                  contact_r_nm=spec.pitch_nm * 0.16 * j(),
                  contact_core_r_nm=spec.pitch_nm * 0.062 * j())
    else:
        kw.update(fin_pitch_nm=spec.pitch_nm,
                  fin_width_nm=spec.pitch_nm * 0.38 * j(),
                  gate_pitch_nm=spec.pitch_nm * float(rng.uniform(4.0, 9.0)),
                  gate_width_nm=spec.pitch_nm * float(rng.uniform(1.0, 1.8)))

    # The builder lays down the periodic arrays only; the landmark is placed
    # afterwards so it can be stamped at every lattice position at the sampled
    # size, rather than placed once and resized in-place.
    kw["landmark_xy_nm"] = None
    lay = BUILDERS[spec.style](**kw)

    # Line-edge roughness. Built from the layout seed and attached to the
    # gratings, so both captures see the identical physical wobble.
    rough = make_edge_roughness(np.random.default_rng(spec.seed ^ 0x10E5),
                                sigma_nm=spec.ler_sigma_nm,
                                xi_nm=spec.ler_xi_nm)
    for arr in lay.arrays:
        if isinstance(arr, LineArray):
            arr.roughness = rough

    # The keep-out is capped so a large landmark cannot blank out most of the
    # reference: a reference that is mostly empty square carries almost no
    # periodic context, which both makes the pair unrealistically easy and stops
    # it testing what it is meant to.
    s = spec.landmark_size_nm
    if spec.landmark == "plus":
        lkw = dict(arm_len_nm=s, arm_w_nm=s * 0.32,
                   clear_nm=min(CLEAR_FACTOR * s, CLEAR_MAX_NM))
    else:
        lkw = dict(size_nm=s)

    n = add_landmark_lattice(
        lay, target_nm[0], target_nm[1], spec.landmark,
        spec.coarse_period_nm or None, reach_nm=extent_nm / 2.0,
        jitter_nm=DECOY_JITTER_NM, rng=rng, **lkw)
    lay.meta["landmark"]["size_nm"] = s
    lay.meta["landmark"]["n_instances"] = n
    return lay



