#!/usr/bin/env python
"""Evaluate the matchers against a reference-generator manifest.

Scores accuracy@5px (the competition metric) for DriftFind and DriftRoute on
any manifest that carries `reference_path, search_path, gt_x, gt_y` columns --
in particular the reference generator's own `manifest.csv`, which is the
cross-generator test that matters.

    python scripts/eval_manifest.py path/to/manifest.csv
    python scripts/eval_manifest.py path/to/manifest.csv --limit 50

Image paths inside the manifest are resolved relative to the manifest's own
directory, then relative to its parent directories, so a manifest written with
repo-root-relative paths works without editing.

If the reference generator's ZNCC baseline is importable (pass --baseline-root
pointing at that checkout), it is scored alongside for direct comparison.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solve import locate as driftfind          # noqa: E402
import route                                    # noqa: E402


def resolve(path, manifest_dir):
    """Find an image referenced by the manifest.

    Manifests in the wild store paths relative to the set directory or to the
    checkout root, and on Windows they may carry backslashes. Try the plausible
    bases in order rather than demanding one convention.
    """
    p = path.replace("\\", "/")
    if os.path.isabs(p) and os.path.exists(p):
        return p
    base = manifest_dir
    for _ in range(4):
        cand = os.path.normpath(os.path.join(base, p))
        if os.path.exists(cand):
            return cand
        base = os.path.dirname(base)
    raise FileNotFoundError(f"could not resolve {path!r} from {manifest_dir!r}")


def load_gray(path):
    return np.asarray(Image.open(path).convert("L"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N pairs")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint to load (default: the shipped best.pt)")
    ap.add_argument("--baseline-root", default=None,
                    help="checkout of the reference generator, to also score its ZNCC baseline")
    args = ap.parse_args()

    manifest_dir = os.path.dirname(os.path.abspath(args.manifest))
    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    net, dev = route.load_net(args.ckpt)

    zncc = None
    if args.baseline_root:
        sys.path.insert(0, os.path.abspath(args.baseline_root))
        try:
            from baseline_solution.zncc import zncc_match as zncc
        except Exception as e:                       # noqa: BLE001
            print(f"[eval] ZNCC baseline unavailable ({type(e).__name__}: {e})")

    methods = [("DriftFind", lambda r, s: driftfind(r, s)),
               ("DriftRoute", lambda r, s: route.locate(r, s, net=net, device=dev))]
    if zncc is not None:
        methods.insert(0, ("ZNCC (reference)",
                           lambda r, s: (lambda m: (m["x"], m["y"]))(zncc(r, s))))

    errs = {name: [] for name, _ in methods}
    secs = {name: 0.0 for name, _ in methods}

    for i, r in enumerate(rows, 1):
        ref = load_gray(resolve(r["reference_path"], manifest_dir))
        srch = load_gray(resolve(r["search_path"], manifest_dir))
        gx, gy = float(r["gt_x"]), float(r["gt_y"])
        for name, fn in methods:
            t0 = time.perf_counter()
            x, y = fn(ref, srch)
            secs[name] += time.perf_counter() - t0
            errs[name].append(np.hypot(x - gx, y - gy))
        if i % 25 == 0:
            print(f"  ...{i}/{len(rows)}", file=sys.stderr)

    n = len(rows)
    print(f"\n=== {args.manifest}  ({n} pairs) ===")
    print(f"{'method':18} {'acc@5px':>9} {'acc@10px':>9} {'median':>8} {'ms/pair':>9}")
    for name, _ in methods:
        e = np.array(errs[name])
        print(f"{name:18} {100 * np.mean(e <= 5):>8.1f}% {100 * np.mean(e <= 10):>8.1f}% "
              f"{np.median(e):>7.2f} {1000 * secs[name] / n:>8.0f}")


if __name__ == "__main__":
    main()
