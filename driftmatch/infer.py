"""Inference for DriftMatchNet: (reference, wide) -> (x, y).

The network gives a 250x250 heatmap and a 2-channel offset. The prediction is
the heatmap peak (mapped back through stride 4) plus the offset at that cell.
The same centre tie-break as the classical matcher is applied on the heatmap
peaks -- the multi-match physics prior does not depend on how the response was
produced, so it carries over unchanged to the learned features.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import maximum_filter

from .data import downsample10, STRIDE, FEAT

# heatmap is a probability in [0,1]; decoys sit >=100 wide-px = 25 feature cells
# apart, so a 20-cell non-max window separates them. LAMBDA plays the same role
# as in the classical matcher: quality-first, gentle pull to centre for ties.
NMS_FEAT = 20
LAMBDA = 0.10


def load_checkpoint(path, map_location="cpu"):
    """Load a checkpoint with weights_only=True, allowlisting the one non-tensor
    type our own train.py stores (numpy scalars in the accuracy metadata).

    Keeping weights_only=True means an unexpected pickle payload is still
    rejected; we only widen the allowlist by a single, known-safe numpy scalar.
    """
    try:
        import numpy.core.multiarray as _ma
        torch.serialization.add_safe_globals([_ma.scalar])
    except Exception:
        pass
    return torch.load(path, map_location=map_location, weights_only=True)


def _select(hm, center_rule, lam=LAMBDA, nms=NMS_FEAT):
    if not center_rule:
        return np.unravel_index(int(np.argmax(hm)), hm.shape)
    local_max = hm == maximum_filter(hm, size=nms)
    ys, xs = np.nonzero(local_max)
    if len(ys) <= 1:
        return np.unravel_index(int(np.argmax(hm)), hm.shape)
    d = np.hypot(xs - FEAT / 2.0, ys - FEAT / 2.0)
    combined = hm[ys, xs] - lam * d / FEAT
    k = int(np.argmax(combined))
    return ys[k], xs[k]


@torch.no_grad()
def net_response(net, ref_img, wide_img, device):
    """Raw network output: (heatmap 250x250, offset (2,250,250)) as numpy.

    Split out from locate_net so the router can read the heatmap to decide
    whether this is a repeated-pattern case, reusing the same forward pass.
    """
    ref = downsample10(ref_img.astype(np.float32)) / 255.0
    wide = wide_img.astype(np.float32) / 255.0
    rt = torch.from_numpy(ref[None, None]).float().to(device)
    wt = torch.from_numpy(wide[None, None]).float().to(device)
    with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
        hm_logits, off = net(rt, wt)
    hm = torch.sigmoid(hm_logits)[0, 0].float().cpu().numpy()
    off = off[0].float().cpu().numpy()               # (2, 250, 250)
    return hm, off


def predict_from_response(hm, off, center_rule=True):
    """Heatmap + offset -> (x, y) in wide-image pixels."""
    cy, cx = _select(hm, center_rule)
    ox, oy = float(off[0, cy, cx]), float(off[1, cy, cx])
    return float((cx + ox) * STRIDE), float((cy + oy) * STRIDE)


def locate_net(net, ref_img, wide_img, device, center_rule=True):
    """ref_img, wide_img: 1000x1000 numpy arrays. Returns (x, y) float."""
    hm, off = net_response(net, ref_img, wide_img, device)
    return predict_from_response(hm, off, center_rule)
