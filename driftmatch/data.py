"""Data pipeline for DriftMatchNet.

Loads the pre-rendered training pool and turns each pair into the tensors the
centre-point network is trained against:

    ref_s  (1,100,100)   reference block-averaged 10x to the wide scale
    wide   (1,1000,1000) the search image
    hm     (1,250,250)   Gaussian heatmap target, peak at the landmark cell
    off    (2,)          sub-cell offset (feature units) at that cell
    cell   (2,)          (row, col) of the landmark cell = where off/hm apply
    gt     (2,)          the raw (x, y) label, for evaluation

The heatmap grid is the wide image at stride 4 (1000 -> 250). A landmark at
wide pixel (gx, gy) lands in feature cell (round(gy/4), round(gx/4)); the
fractional remainder is the offset the head learns, which recovers the sub-cell
position lost to the stride. Exact fractional labels -- the reason option (b)
was chosen -- make this offset target exact.
"""
from __future__ import annotations

import csv
import pathlib

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

STRIDE = 4
FEAT = 250            # 1000 / STRIDE


def load_gray(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def downsample10(img: np.ndarray) -> np.ndarray:
    """1000x1000 -> 100x100 block mean (matches solve.downsample)."""
    return img.reshape(100, 10, 100, 10).mean(axis=(1, 3))


def gaussian_peak(size, cy, cx, sigma):
    ys = np.arange(size, dtype=np.float32)[:, None]
    xs = np.arange(size, dtype=np.float32)[None, :]
    return np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2.0 * sigma * sigma))


def make_targets(gx, gy, size=FEAT, stride=STRIDE, sigma=2.0):
    """Heatmap + offset target for a landmark at wide pixel (gx, gy)."""
    fx, fy = gx / stride, gy / stride
    cx = min(max(int(round(fx)), 0), size - 1)
    cy = min(max(int(round(fy)), 0), size - 1)
    hm = gaussian_peak(size, cy, cx, sigma).astype(np.float32)
    hm[cy, cx] = 1.0                                   # exact positive
    off = np.array([fx - cx, fy - cy], dtype=np.float32)   # (off_x, off_y)
    return hm, off, (cy, cx)


class PairSet(Dataset):
    """Training pairs, cached in RAM.

    On this 4 GB Windows GPU, DataLoader worker processes destabilise CUDA, so
    training runs single-process (num_workers=0). Decoding PNGs in that one
    process then starves the GPU, so the whole pool is decoded once into RAM as
    uint8 (wide ~1 MB each, reference block-averaged to 100x100). __getitem__ is
    then pure array indexing, keeping the GPU fed without any worker processes.
    `limit` caps the pool (and the RAM) when needed.
    """

    def __init__(self, root, cache=True, limit=None):
        self.root = pathlib.Path(root)
        self.rows = list(csv.DictReader((self.root / "labels.csv").open()))
        if limit:
            self.rows = self.rows[:limit]
        self.cache = cache
        if cache:
            import os
            n = len(self.rows)
            cw = str(self.root / "_wide_cache.mmap")
            cr = str(self.root / "_ref_cache.mmap")
            # Reuse an existing memmap of the right size (skips the slow rebuild
            # and its memory churn on re-runs); otherwise build it once.
            if (os.path.exists(cw) and os.path.getsize(cw) == n * 1000 * 1000 and
                    os.path.exists(cr) and os.path.getsize(cr) == n * 100 * 100 * 4):
                self.wide_c = np.memmap(cw, dtype=np.uint8, mode="r", shape=(n, 1000, 1000))
                self.ref_c = np.memmap(cr, dtype=np.float32, mode="r", shape=(n, 100, 100))
            else:
                self.wide_c = np.memmap(cw, dtype=np.uint8, mode="w+", shape=(n, 1000, 1000))
                self.ref_c = np.memmap(cr, dtype=np.float32, mode="w+", shape=(n, 100, 100))
                for i, r in enumerate(self.rows):
                    self.wide_c[i] = np.asarray(
                        Image.open(self.root / r["wide_path"]).convert("L"), dtype=np.uint8)
                    self.ref_c[i] = downsample10(
                        np.asarray(Image.open(self.root / r["ref_path"]).convert("L"),
                                   dtype=np.float32))
                self.wide_c.flush(); self.ref_c.flush()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        if self.cache:
            ref = self.ref_c[i] / 255.0
            wide = self.wide_c[i].astype(np.float32) / 255.0
        else:
            ref = downsample10(load_gray(self.root / r["ref_path"])) / 255.0
            wide = load_gray(self.root / r["wide_path"]) / 255.0
        gx, gy = float(r["gt_x"]), float(r["gt_y"])
        hm, off, (cy, cx) = make_targets(gx, gy)
        return (torch.from_numpy(ref[None]).float(),
                torch.from_numpy(wide[None]).float(),
                torch.from_numpy(hm[None]).float(),
                torch.from_numpy(off).float(),
                torch.tensor([cy, cx], dtype=torch.long),
                torch.tensor([gx, gy], dtype=torch.float32))
