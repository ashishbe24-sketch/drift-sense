"""Verify a generated dataset: parameter distributions against the sampling
policy, and a contact sheet so the variety can be checked by eye."""
import sys, csv, json, pathlib, argparse
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("data", type=pathlib.Path)
ap.add_argument("--tiles", type=int, default=16)
a = ap.parse_args()

rows = list(csv.DictReader((a.data / "labels.csv").open()))
n = len(rows)
f = lambda k: np.array([float(r[k]) for r in rows])
print(f"{n} pairs in {a.data}\n")

print("style     :", {s: sum(r["style"] == s for r in rows) for s in ("dram", "finfet")})
print("regime    :", {s: sum(r["regime"] == s for r in rows)
                      for s in ("resolved", "coarse", "aliased")},
      "   policy 60/25/15 %")
print("placement :", {s: sum(r["placement"] == s for r in rows)
                      for s in ("stage_prior", "uniform")}, "   policy 80/20 %")
print("landmark  :", {s: sum(r["landmark"] == s for r in rows)
                      for s in ("plus", "pad")})

print("difficulty:", {s: sum(r["difficulty"] == s for r in rows)
                      for s in ("below_floor", "hard", "normal")},
      "   policy 8/25/67 %")

for k, unit in (("pitch_nm", "nm"), ("rotation_deg", "deg"),
                ("landmark_size_nm", "nm"), ("landmark_wide_px", "px"),
                ("ref_blur_nm", "nm"), ("wide_blur_nm", "nm"),
                ("wide_charging", ""), ("gt_x", "px"), ("gt_y", "px")):
    v = f(k)
    print(f"  {k:18s} {v.min():8.2f} .. {v.max():8.2f}  "
          f"median {np.median(v):8.2f} {unit}")

ratio = f("ref_dose") / f("wide_dose")
print(f"  {'dose ratio':18s} {ratio.min():8.2f} .. {ratio.max():8.2f}  "
      f"median {np.median(ratio):8.2f} x   -> noise ratio "
      f"{np.sqrt(np.median(ratio)):.2f}x   spec 1.5-4x")

d = np.hypot(f("gt_x") - 500, f("gt_y") - 500)
sp = np.array([r["placement"] == "stage_prior" for r in rows])
if sp.any():
    print(f"\nstage-prior offset from centre: median {np.median(d[sp]):.0f} px "
          f"= {np.median(d[sp])*10/1000:.2f} um   (stage accuracy < 1.5 um)")

# contact sheet
k = min(a.tiles, n)
side = int(np.ceil(np.sqrt(k)))
T = 260
sheet = np.full((side * T * 2, side * T), 30, np.uint8)
for i, r in enumerate(rows[:k]):
    ry, rx = divmod(i, side)
    ref = Image.open(a.data / r["ref_path"]).resize((T, T), Image.BOX)
    wide = Image.open(a.data / r["wide_path"])
    gx, gy = float(r["gt_x"]), float(r["gt_y"])
    w = np.array(wide)
    x0, y0 = int(gx) - 50, int(gy) - 50
    x1, y1 = x0 + 100, y0 + 100
    w[max(0,y0):min(999,y1), max(0,x0)] = 255
    w[max(0,y0):min(999,y1), min(999,x1)] = 255
    w[max(0,y0), max(0,x0):min(999,x1)] = 255
    w[min(999,y1), max(0,x0):min(999,x1)] = 255
    sheet[ry*2*T:ry*2*T+T, rx*T:(rx+1)*T] = np.array(ref)
    sheet[ry*2*T+T:(ry+1)*2*T, rx*T:(rx+1)*T] = np.array(
        Image.fromarray(w).resize((T, T), Image.BOX))
out = a.data / "_contact_sheet.png"
Image.fromarray(sheet).save(out)
print(f"\ncontact sheet -> {out}  (each column pair: reference above, wide below)")
