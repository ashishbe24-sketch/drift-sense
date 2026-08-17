"""Convert the reference generator's manifest.csv into our labels.csv schema.

Lets the existing PairSet loader and evaluation scripts read the organizers'
reference-generator data with no code changes. `coarse_period_nm` is set to 0,
since their manifest carries no equivalent field.

    python scripts/convert_manifest.py <dir-with-manifest.csv>
    python scripts/convert_manifest.py <dir> --root <generator-checkout>

Image paths in their manifest are stored relative to the generator checkout, so
--root says where that checkout lives; it defaults to the manifest directory's
nearest ancestor that resolves. Output paths are absolute, which is what the
training cache expects.
"""
from __future__ import annotations

import argparse
import csv
import os


def resolve_base(d, explicit):
    """Find the directory that the manifest's relative paths hang off."""
    if explicit:
        return os.path.abspath(explicit)
    base = os.path.abspath(d)
    for _ in range(4):
        if os.path.isdir(os.path.join(base, "refdata")) or \
           os.path.isdir(os.path.join(base, "reference")):
            return base
        base = os.path.dirname(base)
    return os.path.abspath(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="directory containing manifest.csv")
    ap.add_argument("--root", default=None,
                    help="reference-generator checkout the paths are relative to")
    args = ap.parse_args()

    base = resolve_base(args.dir, args.root)
    with open(os.path.join(args.dir, "manifest.csv"), newline="") as f:
        rows = list(csv.DictReader(f))

    out = os.path.join(args.dir, "labels.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair_id", "ref_path", "wide_path",
                                          "gt_x", "gt_y", "coarse_period_nm",
                                          "architecture"])
        w.writeheader()
        for r in rows:
            def abspath(p):
                p = p.replace("\\", "/")
                return p if os.path.isabs(p) else os.path.abspath(os.path.join(base, p))
            w.writerow({"pair_id": r["id"],
                        "ref_path": abspath(r["reference_path"]),
                        "wide_path": abspath(r["search_path"]),
                        "gt_x": r["gt_x"], "gt_y": r["gt_y"],
                        "coarse_period_nm": 0,
                        "architecture": r.get("architecture", "")})
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
