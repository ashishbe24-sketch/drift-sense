"""DriftFind -- classical coarse-to-fine matcher for PS-02 (Solution 1).

The public entry point is `locate(reference, wide) -> (x, y)`: given the
1000x1000 reference and the 1000x1000 wide image, return the pixel centre of
the reference's site inside the wide image. The CLI (added later) just wraps
this function, so the core stays a pure function of two arrays.

Phase 1: raw engine -- downsample the reference 10x to a 100x100 stamp and
         slide it over the wide image with FFT normalised cross-correlation.
Phase 2 (this version): before matching, also try the stamp at a few
         Gaussian blurs and a few small rotations, and keep whichever fits
         best.
    - Blur: the wide view is defocused while the shrunk reference is sharp, so
      a sharp stamp under-matches. Trying a few blur levels lets the stamp meet
      the wide view on its own terms.
    - Rotation: the evaluator may turn one image 1-5 deg relative to the other
      ("move either one of them by 3 2 5 degrees"). A small angle search is
      cheap insurance so a slight tilt does not break the match.
Phase 3: centre tie-break for repeated patterns, applied as a soft prior so
         plain pairs are unaffected while decoy ties resolve toward the centre.
Phase 4 (this version): subpixel refinement -- fit a parabola to the three
         score samples around the winning peak in each axis so the returned
         coordinate is fractional, not snapped to the nearest pixel. The ground
         truth is fractional and scored at 1-5 px, so the sub-pixel a whole-
         number peak leaves on the table is worth recovering. Refinement is
         local to the chosen peak and cannot move the answer to another site.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter, rotate as ndrotate, maximum_filter

SCALE = 10          # wide is 10 nm/px, reference 1 nm/px -> known 10x gap (Phase 1)

# Phase 2: the zoom ratio is unknown in [8, 12]. When `locate(..., scales=...)`
# is given a grid, the reference is resampled to each candidate scale and the
# best-correlating one is kept and reported. This coarse grid is refined locally
# afterwards (golden section) to reach the tight pose-recovery tiers. Default
# (scales=None) keeps the exact fixed-SCALE Phase 1 path, byte-for-byte.
PHASE2_SCALES = tuple(float(s) for s in np.arange(8.0, 12.0001, 0.5))

# Phase 2: the relative rotation is unknown in +/-5 deg (see PHASE2_UNDERSTANDING.md).
# ANGLES below (+/-4 deg, Phase 1's tilt-noise bracket) is too narrow and too coarse
# for the pose-recovery tiers (<=0.25/0.5 deg); this grid brackets the full +/-5 deg
# range at coarse steps, then _refine_angle() golden-sections locally around the
# winner. Used only by the scales-given (Phase 2) branch of _fine_score_full --
# the scales=None Phase 1 path keeps its original ANGLES grid, untouched.
PHASE2_ANGLES = tuple(float(a) for a in np.arange(-5.0, 5.0001, 2.5))

# Multi-match tie-break, as a soft centre prior rather than a hard threshold.
# Repeated landmarks sit >=100 px apart, so peaks are enumerated with an 80 px
# non-max window. Among them the winner maximises (NCC - LAMBDA * dist/n_px):
# match quality is primary, so a strong peak far from centre still wins, but
# genuine near-ties (the near-identical decoys) are resolved toward the centre,
# where the stage prior says the true site lands. This is the posterior of the
# stage accuracy, not an arbitrary pull to the middle -- a hard "nearest centre
# among near-ties" rule instead dragged correct off-centre plain matches inward.
NMS_SIZE = 80
# Chosen from a sweep on eval200 (scripts/sweep_lambda.py): 0.08 sits at the
# overall accuracy peak while keeping both plain (83%) and multi-match (77%)
# strong, so it does not depend on the plain/multi ratio of the test set.
LAMBDA = 0.08

# What the stamp is tried at. Blur in stamp pixels (= wide pixels = 10 nm);
# defocus in the data reaches ~2 px, so 0-3 px brackets it. Angles span the
# stated 1-5 deg tilt in both directions.
BLURS = (0.0, 1.0, 2.0, 3.0)
ANGLES = (-4.0, -2.0, 0.0, 2.0, 4.0)

# During the scale search, scale is ranked with a single cheap variant rather
# than the full blur x angle grid: mismatched blur/angle penalise every scale
# candidate about equally, so the correct scale still wins the ranking, and the
# real blur/angle are recovered once at the chosen scale. This keeps the scale
# search inside the CPU runtime budget (measured ~5x faster, same accuracy).
SCAN_BLURS = (0.0,)
SCAN_ANGLES = (0.0,)


def downsample(img: np.ndarray, factor: int = SCALE) -> np.ndarray:
    """Shrink a square image by an integer factor by block-averaging.

    The reference's 1 um field occupies exactly 100 px inside the wide view, so
    averaging each 10x10 block turns the 1000x1000 reference into the 100x100
    stamp we search for -- now at the wide view's pixel scale.
    """
    n = img.shape[0] // factor
    img = img[: n * factor, : n * factor].astype(np.float64)
    return img.reshape(n, factor, n, factor).mean(axis=(1, 3))


def _resize_box(img: np.ndarray, tgt_px: int) -> np.ndarray:
    """Area-average (BOX) resample of a square image to tgt_px x tgt_px.

    BOX averaging matches the wide detector's block integration -- the same
    physical operation the integer `downsample` performs, but at a fractional
    ratio. Used for the Phase 2 scale search, where the zoom is not an integer.
    """
    im = Image.fromarray(np.asarray(img, dtype=np.float32), mode="F")
    im = im.resize((int(tgt_px), int(tgt_px)), Image.BOX)
    return np.asarray(im, dtype=np.float64)


def _stamp_at_scale(reference: np.ndarray, scale: float) -> np.ndarray:
    """The reference as it appears at the wide view's pixel size.

    The reference is 1 nm/px; the wide is `scale` nm/px, so the reference's
    field occupies reference_size / scale pixels in the wide view.
    """
    return _resize_box(reference, int(round(reference.shape[0] / scale)))


def ncc_map(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Normalised cross-correlation score of `template` at every position.

    NCC measures shape agreement independent of brightness/contrast, so it does
    not care that the wide view is darker or noisier. Computed the fast way
    (Lewis 1995): the numerator by FFT, the local image statistics by FFT with a
    box kernel, so a 100x100 stamp over a 1000x1000 image costs milliseconds.
    """
    image = image.astype(np.float64)
    t = template.astype(np.float64)
    t = t - t.mean()
    t_energy = float(np.sum(t * t))
    out_shape = (image.shape[0] - t.shape[0] + 1, image.shape[1] - t.shape[1] + 1)
    if t_energy <= 0:
        return np.zeros(out_shape)
    win = t.size
    ones = np.ones_like(t)

    num = fftconvolve(image, t[::-1, ::-1], mode="valid")     # correlation
    s1 = fftconvolve(image, ones[::-1, ::-1], mode="valid")   # local sum
    s2 = fftconvolve(image * image, ones[::-1, ::-1], mode="valid")  # local sum sq

    local_var = s2 - (s1 * s1) / win
    denom = np.sqrt(np.clip(local_var, 0.0, None) * t_energy)
    out = np.zeros_like(num)
    np.divide(num, denom, out=out, where=denom > 0)
    return out


def _make_variant(stamp, sig, ang):
    """The stamp under one (blur, angle). Rotation keeps the 100x100 size with
    nearest-edge fill; at <=4 deg the corner sliver does not disturb the central
    landmark."""
    v = stamp if ang == 0 else ndrotate(stamp, ang, reshape=False, order=1,
                                        mode="nearest")
    return gaussian_filter(v, sig) if sig > 0 else v


def _denom_image(img, tshape):
    """Image side of the NCC denominator, sqrt(local variance) under the window.

    This depends only on the image and the template *shape*, not the template
    values, so it is identical for every (blur, angle) variant and is computed
    once instead of inside the loop. Same maths as ncc_map -- only hoisted out.
    """
    th, tw = tshape
    win = th * tw
    ones = np.ones((th, tw))
    s1 = fftconvolve(img, ones[::-1, ::-1], mode="valid")
    s2 = fftconvolve(img * img, ones[::-1, ::-1], mode="valid")
    local_var = s2 - (s1 * s1) / win
    return np.sqrt(np.clip(local_var, 0.0, None))


def _best_variant(image, stamp, blurs, angles):
    """Search all (blur, angle) on `image`; return (blur, angle, peak_value).

    Only the *setting* and its peak NCC are taken from here -- the precise
    location is redone at full resolution afterwards. All variants share the
    image and the template shape, so the denominator's image side (two of the
    three FFTs per variant) is computed once up front; each variant then costs a
    single correlation. The peak value is returned so a scale search can rank
    stamps of different sizes against each other.
    """
    img = image.astype(np.float64)
    denom_img = _denom_image(img, stamp.shape)      # shared across all variants

    best_val, best = -np.inf, (0.0, 0.0)
    for ang in angles:
        for sig in blurs:
            t = _make_variant(stamp, sig, ang).astype(np.float64)
            t -= t.mean()
            t_energy = float(np.sum(t * t))
            num = fftconvolve(img, t[::-1, ::-1], mode="valid")
            denom = denom_img * np.sqrt(t_energy)
            out = np.zeros_like(num)
            np.divide(num, denom, out=out, where=denom > 0)
            val = float(out.max()) if out.size else -np.inf
            if val > best_val:
                best_val, best = val, (sig, ang)
    return best[0], best[1], best_val


def _coarse_peak(cw, stamp_full, blurs, angles):
    """Peak NCC of a full-res stamp against the half-res wide, plus its best
    (blur, angle). The stamp is resized to half so its pixel size matches cw
    (which is the wide block-averaged by 2). Used to rank scale candidates."""
    cs = _resize_box(stamp_full, max(4, stamp_full.shape[0] // 2))
    sig_c, ang, val = _best_variant(cw, cs, tuple(b / 2 for b in blurs), angles)
    return val, sig_c * 2.0, ang


def _select_peak(score, h, w, n_px, center_rule, nms=NMS_SIZE, lam=LAMBDA):
    """Pick the winning top-left (r, c) from a correlation score map.

    Without the centre rule this is a plain argmax. With it, local maxima are
    enumerated and the one maximising (NCC - lam * dist_to_centre / n_px) wins:
    quality-first, with a gentle pull toward the centre that only decides
    genuine ties. On a plain pair one peak dominates on NCC, so it wins
    regardless of where it sits; among near-identical decoys the centre-most
    edges ahead.
    """
    if not center_rule:
        return np.unravel_index(int(np.argmax(score)), score.shape)

    local_max = score == maximum_filter(score, size=nms)
    ys, xs = np.nonzero(local_max)
    if len(ys) <= 1:
        return np.unravel_index(int(np.argmax(score)), score.shape)

    cx, cy = xs + w / 2.0, ys + h / 2.0           # match centres
    d = np.hypot(cx - n_px / 2.0, cy - n_px / 2.0)  # distance to image centre
    combined = score[ys, xs] - lam * d / n_px
    k = int(np.argmax(combined))
    return ys[k], xs[k]


def _refine_scale(cw, reference, s0, step, blurs, angles, iters=4):
    """Golden-section refine of the zoom around a coarse winner s0 +/- step.

    Maximises the half-res peak NCC as a function of scale. A handful of
    iterations narrows the bracket enough to reach the tight scale tiers without
    a dense full-res grid. Returns (scale, blur, angle) at the refined scale.
    """
    invphi = (np.sqrt(5.0) - 1.0) / 2.0            # 0.618...
    a, b = max(1.0, s0 - step), s0 + step
    fc = fd = None
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc = _coarse_peak(cw, _stamp_at_scale(reference, c), blurs, angles)
    fd = _coarse_peak(cw, _stamp_at_scale(reference, d), blurs, angles)
    for _ in range(iters):
        if fc[0] >= fd[0]:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = _coarse_peak(cw, _stamp_at_scale(reference, c), blurs, angles)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = _coarse_peak(cw, _stamp_at_scale(reference, d), blurs, angles)
    if fc[0] >= fd[0]:
        s = c; _, blur, ang = fc
    else:
        s = d; _, blur, ang = fd
    return float(s), float(blur), float(ang)


def _refine_angle(cw, stamp_full, a0, step, blurs, iters=4):
    """Golden-section refine of the rotation angle around a coarse winner a0 +/- step.

    Same structure as _refine_scale, but the objective is the angle instead of
    the zoom: _coarse_peak already accepts a one-element `angles` tuple and
    returns the half-res peak NCC at that exact angle (best over blur), so no
    new evaluator is needed -- only the golden-section bracket around it.
    Returns (angle, blur) at the refined angle.
    """
    invphi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = a0 - step, a0 + step
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc = _coarse_peak(cw, stamp_full, blurs, (c,))
    fd = _coarse_peak(cw, stamp_full, blurs, (d,))
    for _ in range(iters):
        if fc[0] >= fd[0]:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = _coarse_peak(cw, stamp_full, blurs, (c,))
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = _coarse_peak(cw, stamp_full, blurs, (d,))
    if fc[0] >= fd[0]:
        ang = c; _, blur, _ = fc
    else:
        ang = d; _, blur, _ = fd
    return float(ang), float(blur)


def _fine_score_full(reference: np.ndarray, wide: np.ndarray,
                     blurs=BLURS, angles=ANGLES, scales=None):
    """The full-resolution correlation map, stamp geometry, and the recovered
    (angle, blur, scale).

    Single source of truth; `fine_score` slices it to the historical 4-tuple so
    existing callers are unchanged, while `locate(return_info=True)` reads the
    recovered pose for the Phase 2 columns without recomputing the correlation.

    scales=None (default): the fixed-SCALE Phase 1 path, byte-for-byte as before.
    scales=<iterable>: Phase 2 -- rank each candidate zoom by its half-res peak
    NCC, refine the winner by golden section, then correlate once at full
    resolution with the recovered (scale, blur, angle).
    """
    n_px = wide.shape[0]
    cw = downsample(wide, 2)                       # half-res wide, shared

    if scales is None:
        stamp = downsample(reference, SCALE)       # 1000x1000 -> 100x100
        cs = downsample(stamp, 2)
        sig_c, ang, _ = _best_variant(cw, cs, tuple(b / 2 for b in blurs), angles)
        blur = sig_c * 2
        scale = float(SCALE)
    else:
        scales = list(scales)
        # coarse: rank each candidate zoom by its half-res peak NCC, using one
        # cheap variant per scale (see SCAN_BLURS/SCAN_ANGLES)
        best = None                                # (val, scale)
        for s in scales:
            val, _, _ = _coarse_peak(cw, _stamp_at_scale(reference, s),
                                     SCAN_BLURS, SCAN_ANGLES)
            if best is None or val > best[0]:
                best = (val, s)
        # refine the winning scale locally (bracket = one coarse grid step)
        gstep = (scales[1] - scales[0]) if len(scales) > 1 else 0.5
        scale, _, _ = _refine_scale(cw, reference, best[1], gstep,
                                    SCAN_BLURS, SCAN_ANGLES)
        # recover the real blur/angle once, at the chosen scale, full grid
        stamp = _stamp_at_scale(reference, scale)
        _, blur, ang = _coarse_peak(cw, stamp, blurs, angles)
        # refine the coarse-grid angle locally (golden section), the same
        # treatment scale already gets above -- needed to reach the tight
        # <=0.25/0.5 deg pose-recovery tiers, which the raw grid step cannot.
        gstep_ang = (angles[1] - angles[0]) if len(angles) > 1 else 2.0
        ang, blur = _refine_angle(cw, stamp, ang, gstep_ang, blurs)

    # fine pass -- full resolution, recovered setting
    score = ncc_map(wide, _make_variant(stamp, blur, ang))
    h, w = stamp.shape
    return score, h, w, n_px, float(ang), float(blur), float(scale)


def fine_score(reference: np.ndarray, wide: np.ndarray,
               blurs=BLURS, angles=ANGLES):
    """The full-resolution correlation map plus stamp geometry.

    Split out from locate() so peak selection can be studied (e.g. sweeping the
    centre-prior strength) without recomputing the expensive correlation. Thin
    wrapper over `_fine_score_full`; returns the historical 4-tuple.
    """
    score, h, w, n_px, _ang, _blur, _sc = _fine_score_full(reference, wide,
                                                           blurs, angles)
    return score, h, w, n_px


def _subpixel(score, r, c):
    """Refine an integer peak (r, c) to fractional position by a 1-D parabola
    fit in each axis.

    Around a correlation maximum the score is locally quadratic, so the vertex
    of the parabola through the peak and its two neighbours estimates the true
    peak to a fraction of a pixel. Done independently per axis, which is exact
    for a separable peak and a good approximation otherwise. The offset is
    bounded to one pixel: a fit that wants to move further is degenerate (a flat
    or double peak) and is not trusted.
    """
    r, c = int(r), int(c)
    H, W = score.shape
    dy = dx = 0.0
    if 0 < c < W - 1:
        a, b, d = score[r, c - 1], score[r, c], score[r, c + 1]
        den = a - 2.0 * b + d
        if den != 0:
            dx = 0.5 * (a - d) / den
    if 0 < r < H - 1:
        a, b, d = score[r - 1, c], score[r, c], score[r + 1, c]
        den = a - 2.0 * b + d
        if den != 0:
            dy = 0.5 * (a - d) / den
    return r + float(np.clip(dy, -1.0, 1.0)), c + float(np.clip(dx, -1.0, 1.0))


# Sign convention for the recovered rotation reported in Phase 2's `theta`
# column (contract: degrees, CCW positive, about the match centre). `ANGLES`
# are the angles the stamp is rotated by (via scipy ndrotate) to best match the
# wide view.
#
# Fixed empirically, not assumed (29 Aug): with our own generator's new
# relative-rotation ground truth (driftsense.raster.make_pair's
# relative_theta_deg -- see PHASE2_RESEARCH_NOTES.md), generating 30 pairs with
# a known signed rotation and comparing against solve.locate's recovered angle
# gave correlation -0.95 with THETA_SIGN=-1.0 (median abs error 3.35 deg) and
# +0.95 with THETA_SIGN=+1.0 (median abs error 0.26 deg) -- i.e. the original
# guess had the wrong sign. Flipped here.
#
# CAVEAT, still open: this validates that our own recovery pipeline agrees with
# our own generator's CCW-positive convention (a real, useful check -- it rules
# out a whole class of bugs). It does NOT yet confirm our convention matches
# the organizers' -- that needs their sample pairs' ground-truth theta
# (expected ~29 Aug per the addendum timeline; not released as of this fix).
# Re-verify against their sample the moment it lands.
THETA_SIGN = 1.0


def locate(reference: np.ndarray, wide: np.ndarray,
           blurs=BLURS, angles=ANGLES, center_rule=True,
           lam=LAMBDA, subpixel=True, return_info=False, scales=None):
    """Return (x, y) pixel centre of the reference's site in the wide image.

    Coarse-to-fine: pick the best (blur, angle) on a half-size image, then
    correlate once at full resolution with that setting. Among tied peaks the
    centre-most is chosen (the multi-match tie-break); set center_rule=False to
    fall back to plain argmax.

    scales=None (default): the fixed-SCALE Phase 1 path, unchanged. Pass a grid
    (e.g. PHASE2_SCALES) to also search the unknown [8,12] zoom -- the stamp size
    then tracks the recovered scale, so x/y stay correct off the nominal 10x.

    With return_info=True, returns (x, y, info) where info carries the Phase 2
    diagnostics: `score` (peak NCC in [-1, 1], the confidence signal), `theta`
    (recovered rotation, degrees, CCW-positive per THETA_SIGN), and `scale`
    (the recovered downscaling factor). The default 2-tuple return is unchanged.
    """
    score, h, w, n_px, ang, blur, scale = _fine_score_full(
        reference, wide, blurs, angles, scales=scales)
    r, c = _select_peak(score, h, w, n_px, center_rule, lam=lam)   # top-left (ints)
    peak = float(score[int(r), int(c)])                            # confidence at the peak
    if subpixel:
        r, c = _subpixel(score, r, c)
    x, y = float(c + w / 2.0), float(r + h / 2.0)                  # x = column, y = row
    if return_info:
        # Rejection signal: how much the winning peak stands out from the next
        # distinct peak. On an absent pair the wide is periodic with no unique
        # landmark, so many peaks tie (ratio -> 1); a present pair's landmark
        # makes one peak stand proud (ratio < 1). Validated to separate
        # present/absent far better than the absolute peak (F1 0.96 vs 0.86).
        lm = score == maximum_filter(score, size=NMS_SIZE)
        vals = np.sort(score[lm])[::-1]
        second_ratio = float(vals[1] / vals[0]) if len(vals) > 1 and vals[0] > 0 else 1.0
        distinct = float(peak * (1.0 - second_ratio))   # gap, scaled by peak height
        return x, y, {"score": peak, "theta": THETA_SIGN * ang, "scale": scale,
                      "blur": blur, "second_ratio": second_ratio,
                      "distinct": distinct}
    return x, y
