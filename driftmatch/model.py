"""DriftMatchNet -- learned matcher for PS-02 (Solution 2).

A fully-convolutional Siamese cross-correlation network with a centre-point
head. The design is deliberately *not* a coordinate-regression CNN: regression
throws away spatial structure and is brittle on repeated patterns. Instead both
images pass through the same shallow, high-resolution encoder; the reference
becomes a small feature filter; that filter is slid over the wide feature map by
correlation; and a small head reads the resulting response into a location
heatmap plus a sub-cell offset.

Shapes (stride 4, so position survives to a few pixels):
    reference (B,1,100,100) --enc--> (B,C,25,25)     the feature filter
    wide      (B,1,1000,1000)--enc--> (B,C,250,250)   the search field
    depthwise x-correlation  --------> (B,C,250,250)   response
    head --> heatmap (B,1,250,250) + offset (B,2,250,250)

Why shallow and high-resolution: classification backbones (ResNet-50, VGG,
EfficientNet) downsample 32x, which destroys the "where" we are scored on at
1-5 px, and are too heavy for a 4 GB GPU. A stride-4 encoder with residual
blocks keeps the resolution and the trainability without the depth.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.short = (nn.Identity() if stride == 1 and cin == cout else
                      nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                    nn.BatchNorm2d(cout)))

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        return F.relu(y + self.short(x), inplace=True)


class Encoder(nn.Module):
    """Shared Siamese encoder. Total stride 4, output C channels."""

    def __init__(self, C=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),  # /2
            nn.Conv2d(32, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.layer1 = ResBlock(32, 64, stride=2)     # /4
        self.layer2 = ResBlock(64, C, stride=1)
        self.proj = nn.Conv2d(C, C, 3, 1, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return self.proj(x)


def xcorr_depthwise(search, kernel):
    """Per-sample, per-channel ("depthwise") cross-correlation.

    conv2d shares one kernel across the batch, but here every sample has its own
    reference filter, so the batch is folded into the channel/group dimension:
    B*C groups, one 25x25 kernel each. `padding = Hk//2` keeps the response the
    same size as the search field, so heatmap position maps straight back to a
    wide-feature cell.
    """
    B, C, Hs, Ws = search.shape
    Hk, Wk = kernel.shape[2], kernel.shape[3]
    search = search.reshape(1, B * C, Hs, Ws)
    kernel = kernel.reshape(B * C, 1, Hk, Wk)
    out = F.conv2d(search, kernel, groups=B * C, padding=(Hk // 2, Wk // 2))
    return out.reshape(B, C, out.shape[2], out.shape[3])


def adaptive_shrink(x, eps=1e-3, clamp=20.0):
    """Parameter-free clutter suppression (the RS-2024 idea).

    Per sample and channel, standardise the response and keep only the part
    above the mean. Background and periodic decoys give diffuse, near-mean
    correlation and are pushed toward zero; a true match stands proud. No extra
    parameters, so it cannot overfit.

    Done in float32 with a non-trivial eps and a clamp: under fp16 autocast a
    channel whose response nearly collapses gives a tiny std, and 1/std then
    overflows to inf -> NaN. Standardising in float32, flooring the divisor, and
    bounding the output keeps it finite without changing the intended effect.
    """
    xf = x.float()
    m = xf.mean(dim=(2, 3), keepdim=True)
    s = xf.std(dim=(2, 3), keepdim=True).clamp_min(eps)
    return F.relu((xf - m) / s).clamp_max(clamp).to(x.dtype)


class Head(nn.Module):
    def __init__(self, C=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(C, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.hm = nn.Conv2d(64, 1, 1)
        self.off = nn.Conv2d(64, 2, 1)
        self.hm.bias.data.fill_(-2.19)      # focal-loss prior: rare positives

    def forward(self, x):
        x = self.body(x)
        return self.hm(x), self.off(x)


class DriftMatchNet(nn.Module):
    STRIDE = 4

    def __init__(self, C=64, shrink=True):
        super().__init__()
        self.enc = Encoder(C)
        self.head = Head(C)
        self.use_shrink = shrink

    def forward(self, ref, wide):
        fz = self.enc(ref)                 # (B,C,25,25)  the feature filter
        fx = self.enc(wide)                # (B,C,250,250) the search field
        # Correlation and shrink in float32: the encoder runs under fp16
        # autocast for memory, but the cross-correlation of unbounded features
        # overflows fp16 (>65504 -> inf -> NaN). Normalising each channel of the
        # reference filter to unit L2 makes this a *learned* normalised
        # cross-correlation -- bounded, and the same idea the classical matcher
        # uses -- so the response scale no longer depends on feature magnitude.
        with torch.autocast("cuda", enabled=False):
            fz = fz.float()
            fx = fx.float()
            fz = fz / (fz.flatten(2).norm(dim=2).clamp_min(1e-4)[:, :, None, None])
            r = xcorr_depthwise(fx, fz)    # (B,C,250,250)
            if self.use_shrink:
                r = adaptive_shrink(r)
        return self.head(r)                # heatmap logits, offset
