"""Contact sheet for the curated set: one row per ladder, so the variable being
swept is visible left to right."""
import csv, pathlib
import numpy as np
from PIL import Image

root = pathlib.Path("data/curated30")
rows = list(csv.DictReader((root / "labels.csv").open()))
groups = {}
for r in rows:
    groups.setdefault(r["group"], []).append(r)

T = 200
ladders = [g for g in groups if g.startswith("L")]
singles = [g for g in groups if g.startswith("S")]

lines = []
for g in ladders:
    tiles = []
    for r in sorted(groups[g], key=lambda x: int(x["step"])):
        w = np.asarray(Image.open(root / r["wide_path"]))
        gx, gy = int(float(r["gt_x"])), int(float(r["gt_y"]))
        y0, x0 = max(0, gy - 60), max(0, gx - 60)
        tiles.append(np.array(Image.fromarray(w[y0:y0 + 120, x0:x0 + 120])
                              .resize((T, T), Image.NEAREST)))
    lines.append(np.hstack(tiles))

srow = []
for g in sorted(singles, key=lambda s: int(s.split()[0][1:])):
    r = groups[g][0]
    w = np.asarray(Image.open(root / r["wide_path"]))
    gx, gy = int(float(r["gt_x"])), int(float(r["gt_y"]))
    y0, x0 = max(0, min(gy - 60, 880)), max(0, min(gx - 60, 880))
    srow.append(np.array(Image.fromarray(w[y0:y0 + 120, x0:x0 + 120])
                         .resize((T, T), Image.NEAREST)))
lines.append(np.hstack(srow[:6]))
lines.append(np.hstack(srow[6:12]))

W = max(l.shape[1] for l in lines)
sheet = np.full((sum(l.shape[0] for l in lines), W), 25, np.uint8)
y = 0
for l in lines:
    sheet[y:y + l.shape[0], :l.shape[1]] = l
    y += l.shape[0]
Image.fromarray(sheet).save(root / "_ladders.png")
print("rows:", [g for g in ladders] + ["singles S1-S6", "singles S7-S12"])
print("wrote", root / "_ladders.png")
