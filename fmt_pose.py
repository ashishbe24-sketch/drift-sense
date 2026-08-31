"""Fourier-Mellin invariant scale+rotation estimator for PS-02 Phase 2.

Standalone, independent of solve.py's coarse-to-fine grid search
(PHASE2_SCALES / PHASE2_ANGLES / _refine_scale / _refine_angle). Recovers the
relative scale (wide's nm/px, nominally in [8,12]) and rotation (theta,
degrees, CCW-positive about the image centre, matching solve.py's THETA_SIGN
convention) between a reference and a wide image in one shot via classical
Fourier-Mellin / log-polar phase correlation, instead of the grid's iterative
search -- the "invariant formulation" the orientation call named as the other
acceptable approach (see docs/TEAMMATE_TASK_FOURIER_MELLIN.md).

Why this can work despite reference and wide NOT sharing a field of view
(reference: 1 um FOV at 1 nm/px; wide: ~8-12 um FOV at ~scale nm/px) -- unlike
textbook FMT registration, which assumes near-identical content differing only
by rotation/scale/translation: DriftSense's die layouts are periodic (DRAM /
FinFET arrays). The periodic lattice's own spatial frequency is a
translation-invariant signature present in BOTH images. A physical pitch of
P nm appears at P px in the reference and P/scale px in the wide view, and the
whole layout -- lattice included -- is rotated by the same `relative_theta_deg`
in the wide capture as the landmark is (driftsense/raster.py::make_pair only
re-renders the wide capture at a shifted layout angle). So the lattice's
frequency-domain peak moves radially (in log-polar space) by log(scale) and
angularly by theta between the two images' magnitude spectra -- exactly what
Fourier-Mellin measures, just driven by the periodic background rather than a
shared foreground.

Expected failure mode, stated up front: in the "aliased" pitch regime (pitch
below the wide view's Nyquist limit -- docs/GENERATOR_SPEC.md section 2.3),
the lattice folds into moire and its apparent frequency is no longer simply
related to the true one by the scale ratio. This estimator is expected to
degrade there -- measured, not just predicted; see the dated entry in
docs/PHASE2_RESEARCH_NOTES.md for the actual numbers.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

N_THETA = 720          # 0.25 deg/bin over the 0..180 deg span (spectra are
                       # symmetric under 180 deg rotation; our true rotation is
                       # always well inside +/-5 deg so this never wraps).
N_RHO = 720
R_MIN = 2.0            # px (FFT bins); excludes the DC neighbourhood.
R_MAX_FRAC = 0.48      # fraction of Nyquist (N/2); leaves headroom at the edge.
HIGHPASS_SIGMA_FRAC = 0.003  # Gaussian high-pass radius, fraction of image size
                             # -- deliberately tight (a few px): the reference's
                             # periodic-lattice frequency can sit as low as
                             # radius~3-4 bins (a 300nm pitch at 1 nm/px on a
                             # 1000px frame), so a wide rolloff here would
                             # suppress exactly the signal this module depends
                             # on, not just the DC spike it targets.


def _hann2d(n: int) -> np.ndarray:
    w = np.hanning(n)
    return np.outer(w, w)


def _highpass_mask(n: int, sigma_frac: float) -> np.ndarray:
    c = n / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    r2 = (yy - c) ** 2 + (xx - c) ** 2
    sigma = sigma_frac * n
    return 1.0 - np.exp(-r2 / (2.0 * sigma * sigma))


def _magnitude_spectrum(img: np.ndarray, window: np.ndarray, hp_mask: np.ndarray) -> np.ndarray:
    f = np.fft.fftshift(np.fft.fft2(img.astype(np.float64) * window))
    mag = np.abs(f) * hp_mask
    # log-magnitude compresses the dynamic range so a handful of very strong
    # low-frequency bins (still leaking past the high-pass) don't dominate the
    # log-polar cross-correlation the way raw linear magnitude would.
    return np.log1p(mag)


def _logpolar_grid(n: int):
    r_max = R_MAX_FRAC * (n / 2.0)
    log_r = np.linspace(np.log(R_MIN), np.log(r_max), N_RHO)
    thetas = np.linspace(0.0, np.pi, N_THETA, endpoint=False)
    return thetas, log_r


def _logpolar_remap(mag: np.ndarray, thetas: np.ndarray, log_r: np.ndarray) -> np.ndarray:
    n = mag.shape[0]
    c = n / 2.0
    rs = np.exp(log_r)
    th_grid, r_grid = np.meshgrid(thetas, rs, indexing="ij")   # (N_THETA, N_RHO)
    xs = c + r_grid * np.cos(th_grid)
    ys = c + r_grid * np.sin(th_grid)
    coords = np.stack([ys.ravel(), xs.ravel()])
    sampled = map_coordinates(mag, coords, order=1, mode="constant", cval=0.0)
    lp = sampled.reshape(th_grid.shape)
    # Whiten per radial ring (each column): both a reference and a wide image
    # share a similar broadband 1/f-ish radial envelope (any textured image
    # does), and that shared envelope -- not the lattice's angular
    # concentration -- otherwise dominates a naive cross-correlation of the
    # two log-polar images (confirmed empirically: without this, the
    # recovered scale collapsed to ~1.0 on every pair, i.e. the correlation
    # peak sat at zero shift regardless of the true scale). Subtracting each
    # column's mean and dividing by its std removes the common radial shape
    # and leaves only the azimuthal structure at each radius -- which is
    # exactly where the periodic lattice's signature lives.
    col_mean = lp.mean(axis=0, keepdims=True)
    col_std = lp.std(axis=0, keepdims=True)
    lp = (lp - col_mean) / (col_std + 1e-6)
    return lp


def _phase_correlate(a: np.ndarray, b: np.ndarray):
    """Return (dy, dx, peak, surface) -- the integer shift that best aligns
    `a` onto `b` (circular, standard normalised cross-power spectrum)."""
    Fa = np.fft.fft2(a)
    Fb = np.fft.fft2(b)
    R = Fa * np.conj(Fb)
    R /= (np.abs(R) + 1e-12)
    surf = np.fft.ifft2(R).real
    idx = np.unravel_index(int(np.argmax(surf)), surf.shape)
    h, w = surf.shape
    dy = idx[0] if idx[0] <= h // 2 else idx[0] - h
    dx = idx[1] if idx[1] <= w // 2 else idx[1] - w
    return dy, dx, float(surf[idx]), surf, idx


def _parabolic_subbin(surf: np.ndarray, idx: tuple) -> tuple:
    """1-D parabola fit through the peak and its neighbours, per axis
    (same technique as solve.py's `_subpixel`), with circular wraparound."""
    h, w = surf.shape
    r, c = idx
    r0, r1, r2 = surf[(r - 1) % h, c], surf[r, c], surf[(r + 1) % h, c]
    c0, c1, c2 = surf[r, (c - 1) % w], surf[r, c], surf[r, (c + 1) % w]
    dr = 0.0
    den = (r0 - 2 * r1 + r2)
    if den != 0:
        dr = float(np.clip(0.5 * (r0 - r2) / den, -1.0, 1.0))
    dc = 0.0
    den = (c0 - 2 * c1 + c2)
    if den != 0:
        dc = float(np.clip(0.5 * (c0 - c2) / den, -1.0, 1.0))
    return dr, dc


# Sign / direction conventions, fixed empirically (see
# docs/PHASE2_RESEARCH_NOTES.md, dated entry, for the calibration numbers) --
# not assumed, same standard this codebase holds itself to for solve.py's
# THETA_SIGN and driftsense/raster.py's barrel-distortion sign.
THETA_SIGN = -1.0
SCALE_DIRECTION = -1.0   # sign applied to the recovered delta-log-radius before
                         # exponentiating into a scale ratio (see calibration).


def estimate_scale_rotation(reference: np.ndarray, wide: np.ndarray):
    """Estimate (scale, theta_deg, confidence) of `wide` relative to
    `reference` via Fourier-Mellin / log-polar phase correlation.

    reference, wide : 1000x1000 uint8 or float numpy arrays (same as
        solve.locate). Both are resampled to a common square size internally
        if their shapes differ (not expected in this project, but kept
        general).
    Returns (scale, theta_deg, confidence):
      scale      : recovered wide-image nm/px, nominally in [8, 12].
      theta_deg  : recovered rotation, degrees, CCW-positive, matching
                   solve.py's THETA_SIGN convention.
      confidence : the log-polar phase-correlation peak height, roughly in
                   [0, 1] (not calibrated against solve.py's peak-NCC scale --
                   a different signal, do not mix the two directly).
    """
    ref = np.asarray(reference, dtype=np.float64)
    wid = np.asarray(wide, dtype=np.float64)
    if ref.shape != wid.shape:
        # Log-polar bin geometry assumes a shared frame; resample the smaller
        # up to the larger's size (bilinear) rather than special-case a ratio
        # this project never actually produces (both images are always 1000^2).
        n = max(ref.shape[0], wid.shape[0])
        if ref.shape[0] != n:
            zoom = n / ref.shape[0]
            from scipy.ndimage import zoom as _zoom
            ref = _zoom(ref, zoom, order=1)
        if wid.shape[0] != n:
            zoom = n / wid.shape[0]
            from scipy.ndimage import zoom as _zoom
            wid = _zoom(wid, zoom, order=1)

    n = ref.shape[0]
    window = _hann2d(n)
    hp = _highpass_mask(n, HIGHPASS_SIGMA_FRAC)
    mag_ref = _magnitude_spectrum(ref, window, hp)
    mag_wid = _magnitude_spectrum(wid, window, hp)

    thetas, log_r = _logpolar_grid(n)
    lp_ref = _logpolar_remap(mag_ref, thetas, log_r)
    lp_wid = _logpolar_remap(mag_wid, thetas, log_r)

    dth_bin, dr_bin, peak, surf, idx = _phase_correlate(lp_wid, lp_ref)
    sub_dth, sub_dr = _parabolic_subbin(surf, idx)
    dth_bin_f = dth_bin + sub_dth
    dr_bin_f = dr_bin + sub_dr

    dtheta_rad = dth_bin_f * (np.pi / N_THETA)
    dtheta_deg = np.degrees(dtheta_rad)
    # Fold the 180 deg ambiguity (magnitude spectra are point-symmetric) into
    # the nearest representative in (-90, 90] -- our true rotation is always
    # well inside +/-5 deg, so this never discards the right branch.
    if dtheta_deg > 90.0:
        dtheta_deg -= 180.0
    elif dtheta_deg <= -90.0:
        dtheta_deg += 180.0

    d_log_r = dr_bin_f * (log_r[1] - log_r[0])
    scale_ratio = float(np.exp(SCALE_DIRECTION * d_log_r))
    scale = scale_ratio * 1.0   # reference is always 1.0 nm/px

    theta_deg = THETA_SIGN * dtheta_deg
    confidence = float(np.clip(peak, 0.0, 1.0))
    return scale, theta_deg, confidence
