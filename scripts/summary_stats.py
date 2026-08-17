"""Compact distribution report for a generated set, including the new
coarse-periodic and roughness axes."""
import sys, csv, pathlib
import numpy as np

root = pathlib.Path(sys.argv[1])
rows = list(csv.DictReader((root / "labels.csv").open()))
n = len(rows)
f = lambda k: np.array([float(r[k]) for r in rows])
cnt = lambda k: {v: sum(r[k] == v for r in rows) for v in sorted({r[k] for r in rows})}

print(f"{n} pairs in {root}\n")
print("style      :", cnt("style"))
print("regime     :", cnt("regime"), "  policy 60/25/15 %")
print("difficulty :", cnt("difficulty"), "  policy 8/25/67 %")
print("placement  :", cnt("placement"))

per = np.array([r["placement"] == "stage_prior_periodic" for r in rows])
print(f"\nperiodic (multi-match) pairs : {per.sum()}/{n} "
      f"({100*per.mean():.0f}%)   policy 40%")
if per.any():
    cp, ni = f("coarse_period_nm")[per], f("n_instances")[per]
    vis = (10000.0 / cp) ** 2
    d = np.hypot(f("gt_x") - 500, f("gt_y") - 500)[per]
    lim = 0.45 * cp / 10.0
    print(f"  coarse period      : {cp.min():.0f}..{cp.max():.0f} nm")
    print(f"  lattice instances  : {ni.min():.0f}..{ni.max():.0f}")
    print(f"  decoys in the field: {vis.min():.0f}..{vis.max():.0f} "
          f"(median {np.median(vis):.0f})")
    ok = int((np.abs(f('gt_x')[per] - 500) <= lim).sum() +
             (np.abs(f('gt_y')[per] - 500) <= lim).sum())
    print(f"  true site is centre-most instance: {ok}/{2*int(per.sum())} axis checks")
    print(f"  offset from centre : median {np.median(d):.0f} px, max {d.max():.0f} px")

print(f"\nLER 3-sigma : {f('ler_3sigma_nm').min():.2f}..{f('ler_3sigma_nm').max():.2f} nm"
      f"   cited 2.6-3.5 nm")
print(f"LER xi      : {f('ler_xi_nm').min():.1f}..{f('ler_xi_nm').max():.1f} nm"
      f"   cited 7-13 nm")

for k, unit in (("pitch_nm", "nm"), ("rotation_deg", "deg"),
                ("landmark_size_nm", "nm"), ("wide_blur_nm", "nm")):
    v = f(k)
    print(f"{k:18s} {v.min():8.2f} .. {v.max():8.2f}  median {np.median(v):8.2f} {unit}")
r = f("ref_dose") / f("wide_dose")
print(f"{'dose ratio':18s} {r.min():8.2f} .. {r.max():8.2f}  median {np.median(r):8.2f} x"
      f"  -> noise {np.sqrt(np.median(r)):.2f}x   spec 1.5-4x")
