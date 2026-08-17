#!/usr/bin/env python
"""Submission entry point for PS-02 (Drift-Sense).

Matches the evaluator's expected interface (cf. their baseline_solution/infer.py):

    predict(reference_path, search_path) -> (x, y)

Returns the predicted centre of the reference site inside the search image, in
search-image pixel coordinates. Run as a script it prints "x,y" (comma), e.g.

    python infer.py reference.png search.png     ->     558.72,409.06

Under the hood this is DriftRoute: the learned matcher (DriftMatchNet) with a
classical fallback (DriftFind). If torch / a GPU / the checkpoint is unavailable
it degrades to the pure-classical path, so the function always runs.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import route

# The network is loaded once and reused across calls (the evaluator calls
# predict() many times; reloading per call would be needlessly slow).
_NET = None
_DEVICE = "cpu"
_LOADED = False


def _load_gray(path):
    return np.asarray(Image.open(path).convert("L"))


def predict(reference_path, search_path):
    """Locate the reference inside the search image. Returns (x, y) as floats."""
    global _NET, _DEVICE, _LOADED
    if not _LOADED:
        _NET, _DEVICE = route.load_net()      # None -> classical-only, still runs
        _LOADED = True
    x, y = route.locate(_load_gray(reference_path), _load_gray(search_path),
                        net=_NET, device=_DEVICE)
    return float(x), float(y)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python infer.py <reference_path> <search_path>", file=sys.stderr)
        sys.exit(1)
    x, y = predict(sys.argv[1], sys.argv[2])
    print(f"{x:.2f},{y:.2f}")
