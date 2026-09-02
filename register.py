#!/usr/bin/env python
r"""Phase 2 submission entry point for PS-02 (Drift-Sense).

Exact signature mandated by the Phase 2 addendum (slide 5, "ONE ENTRY POINT,
EXACT SIGNATURE"):

    python register.py --input pairs.csv --output predictions.csv

Reads `pairs.csv` (one pair per row: an id plus a reference-image path and a
wide/search-image path), locates each reference inside its wide image, and
writes `predictions.csv` with the six mandated columns, one row per input
pair_id, in input order:

    pair_id, x, y, theta, scale, found, score

Contract details honoured here:
  - Every pair_id appears exactly once. A missing row scores zero, so a pair
    that raises is NOT dropped -- it is written as found=0 with zeroed pose,
    which is strictly safer than omitting it.
  - When found == 0, the pose columns (x, y, theta, scale) are written as 0.
  - No network access, no reading outside the supplied paths: image paths come
    only from pairs.csv (resolved against the CSV's own directory, or used
    as-is), nothing is fetched or discovered elsewhere.

The actual six-field prediction is route.predict_full(); this file is only the
CSV I/O shell around it, so improving the matcher never changes this contract.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import route

# Column-name candidates, tried in order. The organizers' sample pairs.csv has
# not been released yet (~29 Aug per the timeline), so the reader is tolerant of
# a few plausible spellings; lock these to their exact names once the sample
# lands. pair_id falls back to the row index if absent.
ID_KEYS = ("pair_id", "id", "pairid", "pair")
REF_KEYS = ("ref_path", "reference_path", "reference", "ref", "ref_image", "template")
WIDE_KEYS = ("wide_path", "search_path", "search", "wide", "wide_image", "scene")

OUT_FIELDS = ("pair_id", "x", "y", "theta", "scale", "found", "score")
EXPECTED_SHAPE = (1000, 1000)


def _first_key(row: dict, candidates) -> str | None:
    """Return the first candidate column name actually present in the row."""
    lower = {k.lower(): k for k in row}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _resolve(path_str: str, root: pathlib.Path) -> pathlib.Path:
    """Resolve an image path from the CSV against the CSV's directory.

    Tries root/path first (the usual layout, where paths are relative to the
    dataset dir), then the path as-given (absolute, or relative to CWD). Only
    these two -- nothing is searched for elsewhere.
    """
    p = pathlib.Path(path_str)
    joined = root / p
    if joined.exists():
        return joined
    if p.exists():
        return p
    raise FileNotFoundError(f"image not found: {path_str} (looked in {root} and CWD)")


def _load_gray(path: pathlib.Path) -> np.ndarray:
    """Load an image as a single-channel grayscale array.

    Sets A/B/C ship single-channel ('L') SEM images; Set D (the optical-RGB
    bonus track) ships 3-channel RGB. Both feed the same classical matcher: any
    non-grayscale mode (RGB/RGBA/palette) is converted to luminance -- PIL's
    convert('L') applies ITU-R 601 luma (0.299R + 0.587G + 0.114B). An image
    already in 'L' mode is returned untouched, so the A/B/C grayscale output is
    byte-identical to loading it directly -- Set D support costs the grayscale
    path nothing.
    """
    im = Image.open(path)
    if im.mode != "L":
        # Set D optical-RGB (or any multi-channel input) -> luminance.
        print(f"[register] {path.name}: {im.mode} -> L (luma), Set D optical path",
              file=sys.stderr)
        im = im.convert("L")
    img = np.asarray(im)
    if img.ndim != 2:
        raise ValueError(f"{path}: expected 2-D grayscale, got shape {img.shape}")
    if img.shape != EXPECTED_SHAPE:
        print(f"[register] warning: {path} is {img.shape}, expected {EXPECTED_SHAPE}; "
              f"continuing", file=sys.stderr)
    return img


def _safe_row(pair_id) -> dict:
    """A zeroed found=0 row -- used when a pair cannot be processed, so its
    pair_id is still present (a missing row scores zero; this at least keeps the
    rejection F1 in play if the pair happened to be a true absent)."""
    return dict(pair_id=pair_id, x=0.0, y=0.0, theta=0.0, scale=0.0, found=0, score=0.0)


def run(input_csv: pathlib.Path, output_csv: pathlib.Path,
        root: pathlib.Path | None = None) -> int:
    root = root if root is not None else input_csv.resolve().parent
    with input_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"[register] {input_csv} has no data rows")

    id_key = _first_key(rows[0], ID_KEYS)
    ref_key = _first_key(rows[0], REF_KEYS)
    wide_key = _first_key(rows[0], WIDE_KEYS)
    if ref_key is None or wide_key is None:
        raise SystemExit(
            f"[register] could not find reference/wide path columns in "
            f"{input_csv}; saw columns {list(rows[0])}. "
            f"Expected one of {REF_KEYS} and one of {WIDE_KEYS}.")

    # Phase 2 checkpoint (trained on scale + all optical aberrations), NOT the
    # Phase 1 checkpoint route.DEFAULT_CKPT points at -- infer.py (the Phase 1
    # entry point) must stay untouched, so this is loaded explicitly here rather
    # than by changing the shared default. route.load_net() degrades to
    # (None, "cpu") on any failure (no torch, no CUDA, missing file), and
    # route.predict_full() falls back to classical x,y when net is None, so this
    # always runs even on the no-GPU reference machine.
    #
    # best_phase2_speckle.pt is the noise-coverage retrain (1 Sep): warm-started
    # from best_phase2_rot8k.pt on 4000 --phase2 pairs that now also carry
    # multiplicative speckle + salt-and-pepper impulse noise (standard SEM
    # detector degradations the generator previously did not model). On the noisy
    # eval it beats the rot8k net by +5 pp @5px (82% vs 77%), and is a within-
    # noise wash on the clean eval (82% vs 84%) -- i.e. more uniform across
    # distributions. Same architecture, identical CPU latency. Its lineage is
    # best_phase2.pt -> best_phase2_rot8k.pt -> this; both earlier checkpoints are
    # kept as fallbacks. Revert this one line to roll back to rot8k. See
    # PHASE2_RESEARCH_NOTES.md for the full comparison.
    _phase2_ckpt = (pathlib.Path(__file__).resolve().parent / "driftmatch" /
                    "checkpoints" / "best_phase2_speckle.pt")
    net, device = route.load_net(_phase2_ckpt)

    out_rows, n_found = [], 0
    for i, r in enumerate(rows):
        pair_id = r[id_key] if id_key else i
        try:
            ref = _load_gray(_resolve(r[ref_key], root))
            wide = _load_gray(_resolve(r[wide_key], root))
            res = route.predict_full(ref, wide, net=net, device=device)
            out_rows.append(dict(pair_id=pair_id, **res.as_row()))
            n_found += res.found
        except Exception as e:  # noqa: BLE001 -- never drop a pair_id
            print(f"[register] pair {pair_id}: {type(e).__name__}: {e}; "
                  f"writing found=0", file=sys.stderr)
            out_rows.append(_safe_row(pair_id))

    with output_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        w.writeheader()
        w.writerows(out_rows)

    print(f"[register] {len(out_rows)} pairs -> {output_csv}  "
          f"({n_found} found, {len(out_rows) - n_found} rejected)")
    return len(out_rows)


def main():
    ap = argparse.ArgumentParser(
        description="PS-02 Phase 2 batch localisation entry point.")
    ap.add_argument("--input", required=True, type=pathlib.Path,
                    help="input CSV of pairs (pair_id + reference/wide image paths)")
    ap.add_argument("--output", required=True, type=pathlib.Path,
                    help="output CSV of predictions (the 6-column contract)")
    ap.add_argument("--root", type=pathlib.Path, default=None,
                    help="directory image paths are relative to "
                         "(default: the input CSV's own directory)")
    a = ap.parse_args()
    run(a.input, a.output, a.root)


if __name__ == "__main__":
    main()
