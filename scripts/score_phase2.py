r"""Score our pipeline against the exact Phase 2 rubric (PS-02 addendum, slide 6-7).

Runs route.predict_full over a generated dataset and reports the true submission
score component-by-component, instead of the proxy metrics (%@5px etc.) the
older scripts print. "What would this actually score" -- the number that matters.

    python scripts/score_phase2.py data/p2eval_rot200
    python scripts/score_phase2.py data/p2eval_rot200 --ckpt driftmatch/checkpoints/best_phase2.pt

Needs a dataset dir with a labels.csv carrying: ref_path, wide_path, gt_x, gt_y,
scale, rotation_deg, present. (Our --phase2 generator writes exactly these.)

The rubric it reproduces (100 pts, excluding the 10 generator/citations/failure
pts which aren't auto-scorable, and the +10 RGB bonus which is out of scope):

  Localization  40   present pairs, tiered on Euclidean error:
                       <=1px 1.00, <=2px 0.80, <=3px 0.60, <=5px 0.40, >5px 0.
                     Organizers weight 0.45*A + 0.55*B (nominal vs degraded).
                     We usually can't split A/B on our own mixed data, so the
                     headline is the unweighted mean credit; if the manifest has
                     a `set` column (A/B), the weighted number is also shown.
  Pose          20   scale 10 + rotation 10, scored ONLY on present pairs that
                     localized (loc credit > 0) -- "a pose on the wrong tile is
                     noise" (addendum). Scale %-err tiers: <=1% 1.00, <=2% 0.60,
                     <=5% 0.30. Rotation deg tiers: <=0.25 1.00, <=0.5 0.60,
                     <=1.0 0.30.
  Rejection     15   F1 on the `found` flag across ALL pairs (present + absent).
  Calibration   10   AUC of `score` vs per-pair correctness.
  Efficiency     5   relative quartile vs other teams -- not computable here;
                     we report median s/pair against the 5s budget instead.

Assumptions that aren't 100% pinned by the addendum are labelled inline and in
the printout, so nobody reads a number as more certain than it is.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import route


# --- rubric tier tables (exact, from the addendum) ------------------------

def loc_credit(err_px: float) -> float:
    for thr, cr in ((1, 1.0), (2, 0.8), (3, 0.6), (5, 0.4)):
        if err_px <= thr:
            return cr
    return 0.0


def scale_credit(pct_err: float) -> float:
    for thr, cr in ((1.0, 1.0), (2.0, 0.6), (5.0, 0.3)):
        if pct_err <= thr:
            return cr
    return 0.0


def rot_credit(deg_err: float) -> float:
    for thr, cr in ((0.25, 1.0), (0.5, 0.6), (1.0, 0.3)):
        if deg_err <= thr:
            return cr
    return 0.0


def auc(scores, labels) -> float:
    """AUROC via the rank-sum (Mann-Whitney U) identity, no sklearn dependency."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    ss = scores[order]
    i = 0                                   # average ranks within tie groups
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def f1(pred, truth, positive=1):
    pred = np.asarray(pred, int)
    truth = np.asarray(truth, int)
    tp = int(((pred == positive) & (truth == positive)).sum())
    fp = int(((pred == positive) & (truth != positive)).sum())
    fn = int(((pred != positive) & (truth == positive)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return (2 * prec * rec / (prec + rec) if prec + rec else 0.0), tp, fp, fn


def _f(row, *keys, default=None):
    for k in keys:
        if k in row and row[k] not in ("", None):
            return row[k]
    return default


def main():
    ap = argparse.ArgumentParser(description="Score against the exact Phase 2 rubric.")
    ap.add_argument("dataset", type=pathlib.Path, help="dir with labels.csv + images/")
    ap.add_argument("--ckpt", type=pathlib.Path, default=None,
                    help="net checkpoint (default: driftmatch/checkpoints/best_phase2.pt); "
                         "falls back to classical-only if torch/ckpt unavailable")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N pairs")
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="optional per-pair CSV dump")
    a = ap.parse_args()

    root = a.dataset
    rows = list(csv.DictReader((root / "labels.csv").open()))
    if a.limit:
        rows = rows[:a.limit]
    if not rows:
        raise SystemExit(f"no rows in {root/'labels.csv'}")

    ckpt = a.ckpt or (ROOT / "driftmatch" / "checkpoints" / "best_phase2.pt")
    net, device = route.load_net(ckpt)
    print(f"[score] net {'loaded ('+device+')' if net is not None else 'UNAVAILABLE -> classical only'}"
          f"  |  {len(rows)} pairs from {root}")

    per_pair, times = [], []
    for r in rows:
        ref = np.asarray(Image.open(root / _f(r, "ref_path")).convert("L"))
        wide = np.asarray(Image.open(root / _f(r, "wide_path")).convert("L"))
        t0 = time.perf_counter()
        res = route.predict_full(ref, wide, net=net, device=device)
        times.append(time.perf_counter() - t0)

        present = int(float(_f(r, "present", default=1)))
        rec = dict(pair_id=_f(r, "pair_id", "seed", default=""), present=present,
                   found=res.found, score=res.score)
        if present:
            gx, gy = float(_f(r, "gt_x")), float(_f(r, "gt_y"))
            err = float(np.hypot(res.x - gx, res.y - gy))
            lc = loc_credit(err)
            rec.update(loc_err=err, loc_credit=lc)
            if lc > 0:                        # pose scored only where localized
                gsc = float(_f(r, "scale", default="nan"))
                grot = float(_f(r, "rotation_deg", default="nan"))
                if not np.isnan(gsc):
                    rec["scale_credit"] = scale_credit(abs(res.scale - gsc) / gsc * 100.0)
                    rec["scale_pcterr"] = abs(res.scale - gsc) / gsc * 100.0
                if not np.isnan(grot):
                    rec["rot_credit"] = rot_credit(abs(res.theta - grot))
                    rec["rot_err_deg"] = abs(res.theta - grot)
            correct = int(res.found == 1 and err <= 5.0)
        else:
            correct = int(res.found == 0)
        rec["correct"] = correct
        per_pair.append(rec)

    pres = [p for p in per_pair if p["present"]]
    loc = [p for p in per_pair if p["present"] and "loc_credit" in p]
    localized = [p for p in loc if p["loc_credit"] > 0]

    # --- localization (40) ---
    loc_creds = np.array([p["loc_credit"] for p in loc]) if loc else np.array([0.0])
    loc_mean = float(loc_creds.mean())
    loc_pts = loc_mean * 40.0

    # optional A/B weighting if a `set` column exists
    sets = {(_f(r, "set", default="") or "").upper() for r in rows}
    weighted_note = ""
    if {"A", "B"} & sets:
        def setmean(letter):
            v = [p["loc_credit"] for p, r in zip(per_pair, rows)
                 if p["present"] and "loc_credit" in p and (_f(r, "set", default="").upper() == letter)]
            return float(np.mean(v)) if v else None
        ca, cb = setmean("A"), setmean("B")
        if ca is not None and cb is not None:
            w = 0.45 * ca + 0.55 * cb
            weighted_note = f"   [A/B-weighted 0.45*{ca:.3f}+0.55*{cb:.3f} = {w:.3f} -> {w*40:.1f} pts]"

    # --- pose (20): scale 10 + rotation 10, only over localized present pairs ---
    sc = np.array([p["scale_credit"] for p in localized if "scale_credit" in p])
    rc = np.array([p["rot_credit"] for p in localized if "rot_credit" in p])
    scale_pts = float(sc.mean()) * 10.0 if len(sc) else 0.0
    rot_pts = float(rc.mean()) * 10.0 if len(rc) else 0.0

    # --- rejection (15): F1 on found across all pairs ---
    found_pred = [p["found"] for p in per_pair]
    present_truth = [p["present"] for p in per_pair]
    f1_present, tp, fp, fn = f1(found_pred, present_truth, positive=1)
    f1_absent, _, _, _ = f1([1 - x for x in found_pred], [1 - x for x in present_truth], positive=1)
    f1_macro = 0.5 * (f1_present + f1_absent)
    rej_pts = f1_present * 15.0

    # --- calibration (10): AUC of score vs correctness ---
    a_uc = auc([p["score"] for p in per_pair], [p["correct"] for p in per_pair])
    cal_pts = (a_uc if not np.isnan(a_uc) else 0.0) * 10.0

    med_t = float(np.median(times))

    # --- report ---
    def line(label, pts, maxp, extra=""):
        print(f"  {label:<26} {pts:6.2f} / {maxp:<3}  {extra}")

    print("\n" + "=" * 62)
    print("  PHASE 2 RUBRIC SCORE (self-scored on our own generator)")
    print("=" * 62)
    line("Localization (40)", loc_pts, 40,
         f"mean credit {loc_mean:.3f} over {len(loc)} present{weighted_note}")
    line("  scale (10)", scale_pts, 10, f"mean credit {float(sc.mean()) if len(sc) else 0:.3f} over {len(sc)} localized")
    line("  rotation (10)", rot_pts, 10, f"mean credit {float(rc.mean()) if len(rc) else 0:.3f} over {len(rc)} localized")
    line("Rejection F1 (15)", rej_pts, 15,
         f"F1(present+)={f1_present:.3f}  TP{tp} FP{fp} FN{fn}  [macro {f1_macro:.3f}, absent+ {f1_absent:.3f}]")
    line("Calibration AUC (10)", cal_pts, 10, f"AUC={a_uc:.3f}")
    core = loc_pts + scale_pts + rot_pts + rej_pts + cal_pts
    print("-" * 62)
    print(f"  CORE TOTAL (of 85 auto-scorable)   {core:6.2f} / 85")
    print(f"  Efficiency (5): median {med_t:.2f}s/pair vs 5s budget "
          f"({'OK' if med_t <= 5 else 'OVER'}); relative quartile is cross-team, not scorable here")
    print(f"  Generator/citations/failure (10): manual, not auto-scored")
    print("=" * 62)
    # honest diagnostics alongside the rubric
    if loc:
        errs = np.array([p["loc_err"] for p in loc])
        print(f"  diag: localization %@5px {100*np.mean(errs<=5):.1f}%  median {np.median(errs):.2f}px"
              f"   (misses count 0 in localization credit above)")
    if len(sc):
        print(f"  diag: scale median %err {np.median([p['scale_pcterr'] for p in localized if 'scale_pcterr' in p]):.2f}%"
              f"   theta median err {np.median([p['rot_err_deg'] for p in localized if 'rot_err_deg' in p]):.3f} deg")

    if a.out:
        keys = sorted({k for p in per_pair for k in p})
        with a.out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(per_pair)
        print(f"  per-pair -> {a.out}")


if __name__ == "__main__":
    main()
