"""Confirm the resize domain is genuinely different from the physical one.

If the two render modes produced near-identical wide views, the held-out
validation set would prove nothing. This measures the gap directly on matched
seeds and writes a side-by-side sheet.
"""
import sys, csv, pathlib
import numpy as np
from PIL import Image

A = pathlib.Path("out/phys24")
B = pathlib.Path("out/resz24")
ra = {r["pair_id"]: r for r in csv.DictReader((A / "labels.csv").open())}
rb = {r["pair_id"]: r for r in csv.DictReader((B / "labels.csv").open())}
ids = sorted(set(ra) & set(rb))

diffs, corrs, sharp_a, sharp_b = [], [], [], []
for pid in ids:
    wa = np.asarray(Image.open(A / ra[pid]["wide_path"])).astype(np.float32)
    wb = np.asarray(Image.open(B / rb[pid]["wide_path"])).astype(np.float32)
    diffs.append(float(np.abs(wa - wb).mean()))
    za, zb = wa - wa.mean(), wb - wb.mean()
    corrs.append(float((za * zb).mean() / (za.std() * zb.std() + 1e-9)))
    gx = lambda z: float(np.abs(np.diff(z, axis=1)).mean())
    sharp_a.append(gx(wa)); sharp_b.append(gx(wb))

    # ground truth must be identical -- only the rendering path changed
    assert abs(float(ra[pid]["gt_x"]) - float(rb[pid]["gt_x"])) < 1e-6
    assert abs(float(ra[pid]["gt_y"]) - float(rb[pid]["gt_y"])) < 1e-6

d, c = np.array(diffs), np.array(corrs)
print(f"matched pairs: {len(ids)}   (ground truth identical in both, asserted)")
print(f"mean |physical - resize| : {d.mean():6.2f} grey levels  "
      f"(min {d.min():.2f}, max {d.max():.2f})")
print(f"correlation between them : {c.mean():6.3f}  "
      f"(min {c.min():.3f}, max {c.max():.3f})")
print(f"mean |dI/dx| physical    : {np.mean(sharp_a):6.2f}")
print(f"mean |dI/dx| resize      : {np.mean(sharp_b):6.2f}  "
      f"-> resize is {np.mean(sharp_b)/np.mean(sharp_a):.2f}x as sharp")

k = 4
tiles = []
for pid in ids[:k]:
    wa = np.asarray(Image.open(A / ra[pid]["wide_path"]))
    wb = np.asarray(Image.open(B / rb[pid]["wide_path"]))
    gx, gy = int(float(ra[pid]["gt_x"])), int(float(ra[pid]["gt_y"]))
    x0, y0 = max(0, gx - 60), max(0, gy - 60)
    ca = np.array(Image.fromarray(wa[y0:y0+120, x0:x0+120]).resize((240, 240), Image.NEAREST))
    cb = np.array(Image.fromarray(wb[y0:y0+120, x0:x0+120]).resize((240, 240), Image.NEAREST))
    tiles.append(np.vstack([ca, cb]))
Image.fromarray(np.hstack(tiles)).save("out/render_mode_compare.png")
print("\nwrote out/render_mode_compare.png  (top row physical, bottom row resize,"
      " same seeds, zoomed on the target)")
