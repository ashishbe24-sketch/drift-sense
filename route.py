"""DriftRoute -- the combined submission: one function that dispatches each pair
to the solution that is best at it.

Rationale (measured, not assumed): DriftMatchNet (the learned matcher) is
near-perfect on single-target pairs and an order of magnitude faster, but weaker
on the multi-match case; DriftFind (classical normalised cross-correlation +
centre prior) is the multi-match specialist. So:

    run the net once (fast)  ->  read its heatmap
        heatmap has ONE dominant peak      -> trust the net        (plain case)
        heatmap has SEVERAL strong peaks   -> hand to the classical (multi-match)

This is one algorithm exposed as one function `locate(reference, wide) -> (x, y)`,
using both a classical and a learned component -- exactly the "you can add
classical methods, you can add deep learning networks also" the evaluator
described. It also degrades gracefully: if torch / a GPU / the checkpoint is not
available, it falls back to the pure-classical path so the function still runs
on any machine.
"""
from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
from scipy.ndimage import maximum_filter

from solve import locate as locate_classical
import solve

# driftmatch pulls in torch, so it is imported lazily, inside the branches that
# actually need the network. That is what makes the classical fallback real:
# on a machine with no torch at all this module still imports and locate()
# still returns an answer.

# The shipped checkpoint, resolved against THIS file rather than the working
# directory -- the evaluator may run infer.py from anywhere, and a relative
# path would silently miss the net and quietly drop the submission to the
# classical-only accuracy.
DEFAULT_CKPT = pathlib.Path(__file__).resolve().parent / "driftmatch" / "checkpoints" / "best.pt"

# A pair is treated as multi-match when a second distinct heatmap peak reaches
# this fraction of the top peak -- i.e. the net sees more than one strong
# candidate and cannot, on its own, tell the true site from a near-identical
# decoy. Tuned on eval200 (scripts/eval_router.py sweeps it).
MULTI_RATIO = 0.60


def is_multimatch(hm, ratio=MULTI_RATIO, nms=None):
    """True if the heatmap holds >=2 strong, well-separated peaks."""
    if nms is None:
        from driftmatch.infer import NMS_FEAT
        nms = NMS_FEAT
    local_max = hm == maximum_filter(hm, size=nms)
    vals = np.sort(hm[local_max])[::-1]
    if len(vals) < 2 or vals[0] <= 0:
        return False
    return vals[1] >= ratio * vals[0]


def locate(reference, wide, net=None, device="cpu", ratio=MULTI_RATIO):
    """Route one pair. With no usable net, this is just the classical matcher.

    net : a loaded DriftMatchNet in eval mode, or None to force classical.
    """
    if net is None:
        return locate_classical(reference, wide)

    from driftmatch.infer import net_response, predict_from_response

    hm, off = net_response(net, reference, wide, device)
    if is_multimatch(hm, ratio):
        return locate_classical(reference, wide)          # classical wins here
    return predict_from_response(hm, off, center_rule=True)  # net wins here


# Rejection threshold for the Phase 2 `found` flag, on the peak NCC.
#
# This value and the *choice of signal* were settled by calibration, and the
# choice reversed once on more data -- worth recording so it is not re-litigated:
# on a small (18-pair) set the distinctiveness signal peak*(1-second_ratio) beat
# raw peak (F1 1.0 vs 0.86), but on a larger, harder 60-pair set with more
# aliased/multi-match present pairs, raw peak clearly won (F1 0.925 vs 0.854).
# The reason: a present-but-periodic pair still has a strong landmark peak (high
# raw peak) but low distinctiveness (its decoys tie it), so `distinct` conflates
# "absent" with "present-and-periodic" -- exactly the Set C trap. Raw peak does
# not. Peak is also the natural confidence for the `score` column, so one signal
# serves both.
#
# 0.70 -> 0.68: a regression check on curated30 (Phase 1 data, always present)
# found ONE pair (C09, a hard noise-ladder case) falsely rejected at 0.70 (peak
# 0.6857). This mattered more than a plain F1 check suggests: a false reject on
# a genuinely-present pair zeros BOTH its localization credit (found=0 forces
# the pose columns to 0) AND counts as a rejection-F1 false-negative -- a false
# accept on an absent pair only costs the second. Re-swept the threshold on the
# 60-pair calibration set optimizing cost under false-reject weights of 1x/2x/3x
# (not just raw F1): **0.68 is cost-optimal under all three weightings**
# (FN=1, FP=6, F1=0.925 -- identical to 0.70's F1, so this is not a tradeoff,
# it is a clean improvement) and it clears C09's 0.6857. STILL PROVISIONAL:
# recalibrate on the organizers' data / a larger set, and re-check once Q2 (the
# rejection-F1 positive class) is answered. Residual risk: C09 clears by only
# 0.006 -- a thin margin, not a robust one; a better long-term fix is a
# multi-signal rejection score, not a single-threshold peak cutoff.
# `solve.locate` still returns `distinct`/`second_ratio` for that future work.
#
# 31 Aug -- RECALIBRATED on a 5x-larger set (300 pairs, 233 present / 67 absent,
# 22% absent, seed 950000) via scripts/recalibrate_found.py. Finding: the
# threshold is NOT meaningfully improvable, confirming the separability-ceiling
# note in PHASE2_RESEARCH_NOTES.md. On this set the absent peak-NCC MAX (0.967)
# exceeds the present MEDIAN (0.933), so no single cutoff cleanly separates the
# classes. F1 sits within 0.006 across the whole plausible range: cost-optimal
# (2x FN weight, this comment's own methodology) lands at 0.53 (F1 0.878, but it
# rejects only 10/67 absent -- too permissive), plain-F1-optimal at 0.73 (F1
# 0.882, FN 13), and the current 0.68 (F1 0.876) sits between them. Because the
# "optimum" swings 0.53-0.73 depending only on the weighting and the F1 gain is
# noise, 0.68 is KEPT as the validated middle operating point rather than chasing
# a 0.006 improvement. The real fix remains a multi-signal rejection rule (peak +
# second_ratio/distinct, or the net's own heatmap), which 300 labeled pairs still
# do not comfortably support fitting -- see the research notes.
FOUND_PEAK = 0.68


@dataclasses.dataclass
class PairResult:
    """The six Phase 2 output fields for one pair (slide 5 contract).

    When found == 0, the pose columns (x, y, theta, scale) are zeroed by
    as_row(), per "When 0, write 0 in the pose columns".
    """
    x: float
    y: float
    theta: float
    scale: float
    found: int
    score: float

    def as_row(self):
        if not self.found:
            return dict(x=0.0, y=0.0, theta=0.0, scale=0.0, found=0,
                        score=round(self.score, 6))
        # `+ 0.0` normalises a negative zero (from the theta sign flip) to 0.0.
        return dict(x=round(self.x, 3), y=round(self.y, 3),
                    theta=round(self.theta, 4) + 0.0, scale=round(self.scale, 4),
                    found=1, score=round(self.score, 6))


def predict_full(reference, wide, net=None, device="cpu",
                 found_threshold=FOUND_PEAK) -> PairResult:
    """Produce all six Phase 2 fields for one pair.

    v2 -- a validated hybrid, not a guess: classical scale-search supplies
    `theta`, `scale`, `found`, `score`; the net supplies `x, y` when available.

    Why split it this way (measured, not assumed, on a 116-pair full-aberration
    set, 29 Aug overnight): feeding the net a reference resampled to the
    classical-estimated scale ("helping" it) actually HURT it (86% @5px) versus
    feeding it the reference at the fixed 10x downsample it was trained on, with
    no scale correction at all (89% @5px) -- the net's encoder was trained on a
    fixed 100x100 footprint, so a variable-sized stamp is out-of-distribution for
    it, and it has separately learned to be tolerant of the resulting scale
    mismatch. So the net's OWN (x, y) at fixed-10 beats both classical alone
    (88%) and a "corrected" net. But the net has no scale/pose head, so
    `scale`/`theta`/`found`/`score` still come from classical -- needed anyway,
    not just as a hedge. Combined: 89% @5px, 1.6s/pair (well inside the 5s CPU
    budget). If no net is available (no torch, no checkpoint), falls back to
    classical's own (x, y) -- degrades, does not fail. register.py calls only
    this function, so this design can be revised without touching the I/O layer.
    """
    # Phase 2 runs on unknown zoom in [8,12], so the scale search is always on.
    # It recovers ~10 on nominal pairs and the true value off-nominal.
    x, y, info = solve.locate(reference, wide, return_info=True,
                              scales=solve.PHASE2_SCALES, angles=solve.PHASE2_ANGLES)
    if net is not None:
        try:
            from driftmatch.infer import net_response, predict_from_response
            hm, off = net_response(net, reference, wide, device)
            x, y = predict_from_response(hm, off, center_rule=True)
        except Exception as e:                # noqa: BLE001 -- degrade, don't crash
            print(f"[route] net inference failed ({type(e).__name__}: {e}); "
                  f"using classical x,y for this pair", file=__import__("sys").stderr)
    # `found` thresholds the peak NCC (the signal that separated present/absent
    # best on the larger validation set); `score` is that same peak, a natural
    # confidence for the calibration column. Its optimal definition for the
    # present/absent-mixed AUC may be refined once Q3 is answered.
    score = info["score"]
    found = int(score >= found_threshold)
    return PairResult(x=x, y=y, theta=info["theta"], scale=info["scale"],
                      found=found, score=score)


def load_net(ckpt_path=None):
    """Best-effort net loader; returns (net, device) or (None, 'cpu').

    Any failure -- no torch, no CUDA, no checkpoint -- yields None so the router
    silently falls back to the classical path.
    """
    try:
        import torch
        from driftmatch.model import DriftMatchNet
        from driftmatch.infer import load_checkpoint
        ckpt_path = DEFAULT_CKPT if ckpt_path is None else ckpt_path
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = load_checkpoint(ckpt_path, map_location=device)
        net = DriftMatchNet(C=64).to(device).eval()
        net.load_state_dict(ckpt["model"])
        return net, device
    except Exception as e:                    # noqa: BLE001 -- degrade, don't crash
        print(f"[route] net unavailable ({type(e).__name__}: {e}); "
              f"using classical only")
        return None, "cpu"
