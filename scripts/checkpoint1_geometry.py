"""Checkpoint 1: geometry and two-scale sampling only. No physics yet.

Renders one DRAM pair and one FinFET pair and writes a visual check sheet:
the reference, the wide view, the true 100x100 footprint marked on the wide
view, and a zoom of that footprint next to a 10x-downscaled reference so the
two scales can be compared by eye.
"""
import sys, time, pathlib
import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from driftsense.layout import build_dram, build_finfet
from driftsense.raster import make_pair, to_uint8

OUT = pathlib.Path(__file__).resolve().parents[1] / "out" / "checkpoint1"
OUT.mkdir(parents=True, exist_ok=True)
N = 1000


def box(img, cx, cy, half, val=255, w=2):
    """Draw a hollow box on a uint8 image (visual aid only)."""
    im = img.copy()
    x0, x1 = int(round(cx - half)), int(round(cx + half))
    y0, y1 = int(round(cy - half)), int(round(cy + half))
    for t in range(w):
        for (a, b) in ((y0 + t, None), (y1 - t, None)):
            if 0 <= a < im.shape[0]:
                im[a, max(0, x0):min(im.shape[1], x1)] = val
        for (a, b) in ((x0 + t, None), (x1 - t, None)):
            if 0 <= a < im.shape[1]:
                im[max(0, y0):min(im.shape[0], y1), a] = val
    return im


def run(name, layout, target, offset):
    t0 = time.perf_counter()
    ref, wide, gt = make_pair(layout, target, offset, n_px=N, supersample=1)
    dt = time.perf_counter() - t0

    ref8, wide8 = to_uint8(ref), to_uint8(wide)
    Image.fromarray(ref8).save(OUT / f"{name}_ref.png")
    Image.fromarray(wide8).save(OUT / f"{name}_wide.png")

    print(f"\n=== {name} ===")
    print(f"  render time      : {dt:.2f} s for the pair")
    print(f"  ground truth     : ({gt[0]:.2f}, {gt[1]:.2f}) px in the wide image")
    print(f"  ref   grey min/mean/max : {ref.min():6.1f} {ref.mean():6.1f} {ref.max():6.1f}")
    print(f"  wide  grey min/mean/max : {wide.min():6.1f} {wide.mean():6.1f} {wide.max():6.1f}")

    # visual check sheet
    marked = box(wide8, gt[0], gt[1], 50)
    crop = wide8[int(gt[1]) - 50:int(gt[1]) + 50, int(gt[0]) - 50:int(gt[0]) + 50]
    crop_big = np.array(Image.fromarray(crop).resize((500, 500), Image.NEAREST))
    ref_small = np.array(Image.fromarray(ref8).resize((100, 100), Image.BOX)
                         .resize((500, 500), Image.NEAREST))

    top = np.hstack([np.array(Image.fromarray(ref8).resize((500, 500), Image.BOX)),
                     np.array(Image.fromarray(marked).resize((500, 500), Image.BOX))])
    bot = np.hstack([ref_small, crop_big])
    sheet = np.vstack([top, bot])
    Image.fromarray(sheet).save(OUT / f"{name}_check.png")
    return gt, dt


def verify_analytic_coverage():
    """The analytic distance-field coverage should agree with brute-force
    supersampling. Disagreement would mean every label we produce is subtly
    wrong, so this is checked rather than assumed."""
    from driftsense.raster import capture
    tgt = (6000.0, 6000.0)
    lay = build_dram(landmark_xy_nm=tgt, landmark="plus", angle_deg=2.0)
    for px_nm, name in ((1.0, "reference 1 nm/px"), (10.0, "wide 10 nm/px")):
        a = capture(lay, tgt, px_nm, N, supersample=1)
        b = capture(lay, tgt, px_nm, N, supersample=6)
        err = np.abs(a - b)
        print(f"  {name}: mean |diff| {err.mean():.3f} grey levels, "
              f"max {err.max():.2f}, 99.9th pct {np.percentile(err, 99.9):.2f}")


if __name__ == "__main__":
    # Landmark near the middle of a 12 um layout; the site lands off-centre in
    # the wide view by a plausible stage error (~1.5 um = 150 wide px).
    tgt = (6000.0, 6000.0)

    run("dram", build_dram(landmark_xy_nm=tgt, landmark="plus"), tgt, (+150.0, -95.0))
    run("dram_rot3", build_dram(landmark_xy_nm=tgt, landmark="plus", angle_deg=3.0),
        tgt, (-60.0, +130.0))
    run("finfet", build_finfet(landmark_xy_nm=tgt, landmark="pad"), tgt, (+40.0, +70.0))

    print("\n=== analytic coverage vs 6x supersampled ground truth ===")
    verify_analytic_coverage()

    print(f"\nwrote check sheets to {OUT}")
