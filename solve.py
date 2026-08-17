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
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter, rotate as ndrotate, maximum_filter

SCALE = 10          # wide is 10 nm/px, reference 1 nm/px -> known 10x gap

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


def downsample(img: np.ndarray, factor: int = SCALE) -> np.ndarray:
    """Shrink a square image by an integer factor by block-averaging.

    The reference's 1 um field occupies exactly 100 px inside the wide view, so
    averaging each 10x10 block turns the 1000x1000 reference into the 100x100
    stamp we search for -- now at the wide view's pixel scale.
    """
    n = img.shape[0] // factor
    img = img[: n * factor, : n * factor].astype(np.float64)
    return img.reshape(n, factor, n, factor).mean(axis=(1, 3))


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
    """Search all (blur, angle) on `image`; return the winning (blur, angle).

    Only the *setting* is taken from here -- the precise location is redone at
    full resolution afterwards. All variants share the image and the template
    shape, so the denominator's image side (two of the three FFTs per variant)
    is computed once up front; each variant then costs a single correlation.
    Numerically identical to calling ncc_map per variant, ~2-3x fewer FFTs.
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
    return best


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


def fine_score(reference: np.ndarray, wide: np.ndarray,
               blurs=BLURS, angles=ANGLES):
    """The full-resolution correlation map plus stamp geometry.

    Split out from locate() so peak selection can be studied (e.g. sweeping the
    centre-prior strength) without recomputing the expensive correlation.
    """
    stamp = downsample(reference, SCALE)          # 1000x1000 -> 100x100
    h, w = stamp.shape
    n_px = wide.shape[0]

    # coarse pass -- half resolution; blur scales with resolution, angle doesn't
    cw = downsample(wide, 2)
    cs = downsample(stamp, 2)
    sig_c, ang = _best_variant(cw, cs, tuple(b / 2 for b in blurs), angles)

    # fine pass -- full resolution, chosen setting
    score = ncc_map(wide, _make_variant(stamp, sig_c * 2, ang))
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


def locate(reference: np.ndarray, wide: np.ndarray,
           blurs=BLURS, angles=ANGLES, center_rule=True,
           lam=LAMBDA, subpixel=True) -> tuple[float, float]:
    """Return (x, y) pixel centre of the reference's site in the wide image.

    Coarse-to-fine: pick the best (blur, angle) on a half-size image, then
    correlate once at full resolution with that setting. Among tied peaks the
    centre-most is chosen (the multi-match tie-break); set center_rule=False to
    fall back to plain argmax.
    """
    score, h, w, n_px = fine_score(reference, wide, blurs, angles)
    r, c = _select_peak(score, h, w, n_px, center_rule, lam=lam)   # top-left
    if subpixel:
        r, c = _subpixel(score, r, c)
    return float(c + w / 2.0), float(r + h / 2.0)   # x = column, y = row
