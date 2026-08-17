"""SEM image-formation model applied to a rasterised material map.

Every parameter here is specified in **nanometres or physical units**, never in
pixels. That is what makes the two captures physically consistent: an edge
effect with a 5 nm escape band is 5 px wide in the 1 nm/px reference and 0.5 px
wide in the 10 nm/px view, and it falls below resolution in the wide view on its
own, exactly as it does on a real tool.

Pipeline order follows the NIST ARTIMAGEN paradigm -- define the sample, then
render it with defined, known amounts of each artefact:

    material map
      -> secondary-electron edge response (overshoot + dark halo)
      -> charging
      -> probe PSF, widened by defocus and by beam current
      -> drift shear and per-scanline vibration
      -> over-dispersed Poisson shot noise
      -> Gaussian read noise
      -> 8-bit quantisation

Sources for the numbers are listed in docs/GENERATOR_SPEC.md; the edge
overshoot (+16%), dark halo (-19%) and probe sigma are measured values from
real IC SEM imagery, and the Fano factor of 1.9 is from published measurements
of secondary-electron emission from silicon at 500 eV.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace

import numpy as np
from scipy import ndimage as ndi


# --- measured / cited defaults -------------------------------------------
EDGE_BAND_NM = 5.0        # SE escape depth; literature 3-7 nm, < 10 nm
EDGE_OVERSHOOT = 0.16     # measured on real IC SEM, n=749 edges
EDGE_UNDERSHOOT = 0.19    # measured dark halo outside the edge
PROBE_SIGMA_NM = 1.0      # tuned so the rendered 10-90% rise matches 3.6 nm
FANO = 1.9                # SE-count variance / Poisson variance, Si at 500 eV

# Lobe gains that make the rendered profile reproduce the measured overshoot
# and halo. The lobes are coupled -- raising the inner gain lifts the whole
# profile -- so these are solved numerically, not assumed; see
# scripts/calibrate_edge.py, which also checks the image-domain implementation
# against the analytic step response.
EDGE_G_OVER = 0.8342
EDGE_G_UNDER = 0.9698


@dataclass
class CaptureParams:
    """Physics of one capture. Two of these describe a pair."""
    px_nm: float
    # optics
    probe_sigma_nm: float = PROBE_SIGMA_NM
    defocus_sigma_nm: float = 0.0
    edge_band_nm: float = EDGE_BAND_NM
    edge_overshoot: float = EDGE_G_OVER     # calibrated lobe gains, not the
    edge_undershoot: float = EDGE_G_UNDER   # measured percentages themselves
    # scan artefacts
    drift_nm: float = 0.0          # total stage creep across one frame
    drift_angle_deg: float = 0.0
    vibration_nm: float = 0.0      # rms per-scanline displacement
    # charging
    charging: float = 0.0          # 0 = none; grey-level scale of the field
    charging_scale_nm: float = 400.0
    streak_rate: float = 0.0       # expected number of bright streaks
    # detection
    dose_e_per_grey: float = 40.0  # electrons per grey level; higher = cleaner
    fano: float = FANO
    read_sigma: float = 1.5        # grey levels

    def total_blur_nm(self) -> float:
        """Probe and defocus add in quadrature."""
        return float(np.hypot(self.probe_sigma_nm, self.defocus_sigma_nm))

    def as_dict(self):
        d = asdict(self)
        d["total_blur_nm"] = round(self.total_blur_nm(), 3)
        return d


# --------------------------------------------------------------------------

EDGE_INNER_RATIO = 0.30   # narrow lobe of the band-pass, as a fraction of band


def edge_profile_1d(x_nm, band_nm, g_over, g_under, inner=EDGE_INNER_RATIO):
    """Analytic response to a unit step at x=0 (0 outside, 1 inside).

    Used to calibrate the gains against the measured profile without having to
    render an image, and as the reference the image-domain implementation is
    tested against.
    """
    from scipy.stats import norm
    s1 = max(inner * band_nm, 1e-6)
    a = norm.cdf(x_nm / s1)          # step seen through the narrow lobe
    b = norm.cdf(x_nm / band_nm)     # step seen through the wide lobe
    dog = a - b
    return a + np.where(dog > 0, dog * g_over, dog * g_under)


def calibrate_edge_gains(params=None, target_over=EDGE_OVERSHOOT,
                         target_under=EDGE_UNDERSHOOT):
    """Solve for the lobe gains that make the *rendered* profile match the
    measured one.

    The measured +16% / -19% comes from real SEM images, so it already includes
    that instrument's optics. Calibrating the edge stage in isolation to those
    numbers therefore overshoots: the probe PSF then attenuates the result to
    roughly +12% / -14%. The gains have to be solved through the deterministic
    optics chain instead.

    Stochastic stages are excluded on purpose -- drift and vibration do blur
    the edge, but by an amount that legitimately varies per capture.
    """
    from scipy.optimize import brentq
    p = params or CaptureParams(px_nm=1.0)

    n = int(round(120.0 / p.px_nm))
    step = np.zeros((8, 2 * n), np.float32)
    step[:, n:] = 1.0

    def rendered(g_over, g_under):
        q = replace(p, edge_overshoot=g_over, edge_undershoot=g_under)
        return apply_optics(edge_response(step, q), q).mean(0)

    def over_err(g):
        prof = rendered(g, 0.0)
        return prof[n:].max() - prof[-n // 3:].mean() - target_over

    def under_err(g):
        return -rendered(0.0, g)[:n].min() - target_under

    return brentq(over_err, 0.0, 60.0), brentq(under_err, 0.0, 60.0)


def edge_response(img: np.ndarray, p: CaptureParams) -> np.ndarray:
    """Secondary-electron edge contrast.

    SE yield rises with local surface tilt -- delta(phi) = delta(0)/cos(phi) --
    so feature edges emit more than flat tops, and the region just outside an
    edge is depleted. The measured profile on real IC SEM is a dark halo about
    19% below background, a sharp rise, then a 16% overshoot peaking a couple of
    nanometres *inside* the feature before settling to the plateau.

    That shape needs a band-pass, not a high-pass. A plain unsharp mask
    (img - blur) peaks exactly at the edge, where the underlying step is only
    half its plateau value, so it can never lift the profile above the plateau
    and produces no overshoot at all. A difference of two Gaussians places its
    positive lobe a short distance inside the edge and its negative lobe
    outside, which is the measured shape.

    The two lobes are gained separately because the measured response is
    asymmetric: +16% inside against -19% outside.
    """
    band_px = p.edge_band_nm / p.px_nm
    if band_px < 0.05:
        return img                      # band is far below this pixel size
    s1 = max(EDGE_INNER_RATIO * band_px, 1e-3)
    inner = ndi.gaussian_filter(img, s1, mode="nearest")
    outer = ndi.gaussian_filter(img, band_px, mode="nearest")
    dog = inner - outer
    return inner + np.where(dog > 0, dog * p.edge_overshoot,
                            dog * p.edge_undershoot)


def apply_charging(img: np.ndarray, p: CaptureParams, rng) -> np.ndarray:
    """Charge accumulation on a poorly grounded surface.

    Non-zero net charge deflects the primary beam and shifts local yield,
    which appears as slowly varying bright and dark bands, plus occasional
    bright streaks along the fast-scan direction.
    """
    if p.charging <= 0:
        return img
    h, w = img.shape
    scale_px = max(1.0, p.charging_scale_nm / p.px_nm)
    field = rng.standard_normal((h, w)).astype(np.float32)
    field = ndi.gaussian_filter(field, scale_px, mode="reflect")
    field /= (field.std() + 1e-6)
    out = img + field * p.charging

    if p.streak_rate > 0:
        n = rng.poisson(p.streak_rate)
        for _ in range(int(n)):
            r = rng.integers(0, h)
            length = int(rng.integers(w // 8, w))
            c0 = int(rng.integers(0, max(1, w - length)))
            amp = p.charging * rng.uniform(2.0, 5.0)
            decay = np.exp(-np.arange(length) / (length / 3.0)).astype(np.float32)
            out[r, c0:c0 + length] += amp * decay
    return out


def apply_optics(img: np.ndarray, p: CaptureParams) -> np.ndarray:
    """Probe point-spread function, widened by defocus.

    The wide survey view is blurrier not because it is wide but because the
    tool grabs it quickly without refocusing on the plane of interest, and
    because raising beam current to compensate for the short dwell enlarges the
    virtual source. Both enter through this one sigma.
    """
    sigma_px = p.total_blur_nm() / p.px_nm
    if sigma_px < 0.05:
        return img
    return ndi.gaussian_filter(img, sigma_px, mode="nearest")


def apply_scan_artefacts(img: np.ndarray, p: CaptureParams, rng) -> np.ndarray:
    """Drift shear and vibration.

    A raster image is built line by line, so slow stage creep during the frame
    elongates and shears the field, while vibration displaces individual scan
    lines and serrates what should be straight edges.
    """
    out = img
    if p.drift_nm > 0:
        d_px = p.drift_nm / p.px_nm
        th = np.radians(p.drift_angle_deg)
        h = out.shape[0]
        # displacement grows linearly with scan-line index
        shift_x = np.cos(th) * d_px * np.arange(h) / h
        shift_y = np.sin(th) * d_px * np.arange(h) / h
        rows, cols = np.indices(out.shape, dtype=np.float32)
        out = ndi.map_coordinates(
            out, [rows + shift_y[:, None], cols + shift_x[:, None]],
            order=1, mode="nearest")
    if p.vibration_nm > 0:
        amp_px = p.vibration_nm / p.px_nm
        jitter = rng.standard_normal(out.shape[0]).astype(np.float32) * amp_px
        # vibration is not white: neighbouring lines are correlated in time
        jitter = ndi.gaussian_filter1d(jitter, 1.5, mode="nearest")
        rows, cols = np.indices(out.shape, dtype=np.float32)
        out = ndi.map_coordinates(out, [rows, cols + jitter[:, None]],
                                  order=1, mode="nearest")
    return out


def add_shot_noise(img: np.ndarray, p: CaptureParams, rng) -> np.ndarray:
    """Over-dispersed Poisson shot noise.

    Electron counting is Poisson, but measured secondary-electron emission from
    silicon at 500 eV carries about 1.9x the Poisson variance, so plain Poisson
    understates real SEM noise by nearly a factor of two. A negative binomial
    with mean m and variance F*m reproduces the measured over-dispersion
    exactly, and reduces to Poisson as F -> 1.
    """
    lam = np.clip(img, 1e-3, None) * p.dose_e_per_grey
    if p.fano <= 1.0 + 1e-6:
        counts = rng.poisson(lam)
    else:
        r = lam / (p.fano - 1.0)                 # NB shape parameter
        counts = rng.poisson(rng.gamma(r, (p.fano - 1.0), size=lam.shape))
    return (counts / p.dose_e_per_grey).astype(np.float32)


def add_read_noise(img: np.ndarray, p: CaptureParams, rng) -> np.ndarray:
    if p.read_sigma <= 0:
        return img
    return img + rng.standard_normal(img.shape).astype(np.float32) * p.read_sigma


def quantize(img: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def render(material: np.ndarray, p: CaptureParams, rng) -> np.ndarray:
    """Full image-formation chain for one capture."""
    x = material.astype(np.float32)
    x = edge_response(x, p)
    x = apply_charging(x, p, rng)
    x = apply_optics(x, p)
    x = apply_scan_artefacts(x, p, rng)
    x = add_shot_noise(x, p, rng)
    x = add_read_noise(x, p, rng)
    return quantize(x)


def render_downsampled(fine: np.ndarray, factor: int, fine_px_nm: float,
                       p: CaptureParams, rng) -> np.ndarray:
    """Image formation the way the organisers' starter prompt does it.

    Their wide view is a *resize* of a finely rendered field, so the fine
    scale's edge brightening and probe blur are computed where they are
    resolved and then averaged down. The result keeps a residue of structure
    that a genuine 10 nm/px capture could never record -- in the physical path
    the 5 nm edge band is sub-pixel and vanishes (we measure +0.001 overshoot),
    while here it survives the averaging.

    That residue is precisely the deterministic fine-to-coarse relationship a
    network will latch onto, which is why this path is validation-only.
    """
    fine_p = replace(p, px_nm=fine_px_nm, defocus_sigma_nm=0.0, drift_nm=0.0,
                     vibration_nm=0.0, charging=0.0, streak_rate=0.0)
    x = edge_response(fine.astype(np.float32), fine_p)
    x = apply_optics(x, fine_p)

    n = x.shape[0] // factor
    x = x.reshape(n, factor, n, factor).mean(axis=(1, 3)).astype(np.float32)

    # everything that genuinely happens at the coarse capture
    x = apply_charging(x, p, rng)
    sigma_px = p.defocus_sigma_nm / p.px_nm
    if sigma_px >= 0.05:
        x = ndi.gaussian_filter(x, sigma_px, mode="nearest")
    x = apply_scan_artefacts(x, p, rng)
    x = add_shot_noise(x, p, rng)
    x = add_read_noise(x, p, rng)
    return quantize(x)
