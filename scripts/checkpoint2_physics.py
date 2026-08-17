"""Checkpoint 2: the SEM image-formation model.

Renders a pair with full physics and then measures the result with exactly the
same estimator used on real IC SEM imagery, so the synthetic edge profile can
be compared against the measured one in nanometres rather than asserted.
"""
import sys, time, pathlib
import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from driftsense.layout import build_dram, build_finfet
from driftsense.raster import make_pair, to_uint8
from driftsense.physics import CaptureParams, render

OUT = pathlib.Path(__file__).resolve().parents[1] / "out" / "checkpoint2"
OUT.mkdir(parents=True, exist_ok=True)
N = 1000

# Measured on real IC SEM imagery (MIIC, n=749 edges, ~2.5 nm/px):
#   10-90% rise 1.44 px  -> ~3.6 nm
#   overshoot +16% peaking ~2 px inside -> ~5 nm
#   dark halo -19% below background
TARGET_RISE_NM = 3.6
TARGET_OVERSHOOT = 0.16
TARGET_UNDERSHOOT = -0.19


def edge_profile(img, px_nm, w=10):
    """Same estimator that was run on the real SEM data."""
    a = img.astype(np.float32)
    profs = []
    for prof in (a.mean(1), a.mean(0)):
        d = np.diff(prof)
        if d.std() < 1:
            continue
        for i in range(w + 1, len(prof) - w - 1):
            if d[i] <= 0 or d[i] < 3 * d.std():
                continue
            if d[i] < d[i - 1] or d[i] < d[i + 1]:
                continue
            seg = prof[i - w:i + w + 1]
            base, top = seg[:3].mean(), seg[-4:].mean()
            if top - base < 15:
                continue
            profs.append((seg - base) / (top - base))
    if not profs:
        return None
    m = np.median(np.array(profs), 0)
    x = np.arange(len(m))
    rise = m[:w + 4]
    lo = np.interp(0.1, rise, x[:w + 4])
    hi = np.interp(0.9, rise, x[:w + 4])
    return dict(n=len(profs), rise_px=hi - lo, rise_nm=(hi - lo) * px_nm,
                overshoot=m.max() - 1.0, undershoot=m[:w].min(), profile=m)


def noise_sigma(img):
    """Immerkaer robust noise estimate, as used on the real data."""
    from scipy.signal import convolve2d
    L = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    c = convolve2d(img.astype(np.float32), L, mode="valid")
    return float(np.sqrt(np.pi / 2) * np.abs(c).mean() / 6.0)


def report(name, img, px_nm):
    a = img.astype(np.float32)
    ep = edge_profile(img, px_nm)
    print(f"  {name:9s} grey {a.mean():6.1f} +- {a.std():5.1f} "
          f"| p20/p85 {np.percentile(a,20):5.1f}/{np.percentile(a,85):5.1f} "
          f"| noise sigma {noise_sigma(img):5.2f}")
    if ep:
        print(f"            edges n={ep['n']:4d}  10-90% rise {ep['rise_nm']:5.2f} nm "
              f"({ep['rise_px']:4.2f} px)  overshoot {ep['overshoot']:+.3f}  "
              f"halo {ep['undershoot']:+.3f}")
    return ep


if __name__ == "__main__":
    rng = np.random.default_rng(20260802)
    tgt = (6000.0, 6000.0)

    # Reference: slow, careful characterisation capture taken earlier.
    ref_p = CaptureParams(px_nm=1.0, probe_sigma_nm=1.0, defocus_sigma_nm=0.0,
                          drift_nm=2.0, vibration_nm=0.8,
                          dose_e_per_grey=40.0, read_sigma=1.0)
    # Wide: fast survey grab. Short dwell -> ~3x the noise; no time to refocus
    # -> large defocus term; higher current -> slightly larger probe.
    wide_p = CaptureParams(px_nm=10.0, probe_sigma_nm=1.4, defocus_sigma_nm=17.0,
                           drift_nm=6.0, vibration_nm=3.0,
                           charging=1.8, streak_rate=1.5,
                           dose_e_per_grey=4.0, read_sigma=2.2)

    # --- optics validation -------------------------------------------------
    # The edge estimator collapses rows to find steps, so it needs an
    # unrotated field. Rotation does not affect the optics being measured.
    print("=== optics validation (unrotated field) ===")
    lay0 = build_dram(landmark_xy_nm=tgt, landmark="plus", angle_deg=0.0)
    m0, mw0, _ = make_pair(lay0, tgt, (0.0, 0.0), n_px=N)
    for tag, mat, prm, px in (("reference", m0, ref_p, 1.0),
                              ("wide", mw0, wide_p, 10.0)):
        ep = report(tag, render(mat, prm, rng), px)
    print(f"    real SEM      10-90% rise {TARGET_RISE_NM:.2f} nm | "
          f"overshoot {TARGET_OVERSHOOT:+.3f} | halo {TARGET_UNDERSHOOT:+.3f}")

    # geometry-only reference, to separate optics from noise
    ep_raw = report("no-physics", to_uint8(m0), 1.0)

    for style, builder, lm in (("dram", build_dram, "plus"),
                               ("finfet", build_finfet, "pad")):
        lay = builder(landmark_xy_nm=tgt, landmark=lm, angle_deg=2.2)
        t0 = time.perf_counter()
        mref, mwide, gt = make_pair(lay, tgt, (+120.0, -85.0), n_px=N)
        t_raster = time.perf_counter() - t0

        t0 = time.perf_counter()
        ref = render(mref, ref_p, rng)
        wide = render(mwide, wide_p, rng)
        t_phys = time.perf_counter() - t0

        print(f"\n=== {style} ===  raster {t_raster:.2f}s + physics {t_phys:.2f}s "
              f"= {t_raster + t_phys:.2f}s   gt=({gt[0]:.2f}, {gt[1]:.2f})")
        ep = report("reference", ref, 1.0)
        report("wide", wide, 10.0)

        if ep:
            print(f"    vs real SEM   rise {TARGET_RISE_NM:.2f} nm | "
                  f"overshoot {TARGET_OVERSHOOT:+.3f} | halo {TARGET_UNDERSHOOT:+.3f}")

        Image.fromarray(ref).save(OUT / f"{style}_ref.png")
        Image.fromarray(wide).save(OUT / f"{style}_wide.png")

        # check sheet: ref | wide(boxed) over  ref-zoom | wide-crop-zoom
        marked = wide.copy()
        x0, x1 = int(gt[0]) - 50, int(gt[0]) + 50
        y0, y1 = int(gt[1]) - 50, int(gt[1]) + 50
        marked[y0:y1, x0] = 255; marked[y0:y1, x1] = 255
        marked[y0, x0:x1] = 255; marked[y1, x0:x1] = 255
        crop = np.array(Image.fromarray(wide[y0:y1, x0:x1]).resize((500, 500), Image.NEAREST))
        refz = np.array(Image.fromarray(ref[300:500, 300:500]).resize((500, 500), Image.NEAREST))
        top = np.hstack([np.array(Image.fromarray(ref).resize((500, 500), Image.BOX)),
                         np.array(Image.fromarray(marked).resize((500, 500), Image.BOX))])
        sheet = np.vstack([top, np.hstack([refz, crop])])
        Image.fromarray(sheet).save(OUT / f"{style}_check.png")

    print(f"\nwrote to {OUT}")
