"""Train DriftMatchNet (Solution 2, PS-02).

    <py312> -m driftmatch.train --data data/train --epochs 30 --batch 8

Loss = CenterNet penalty-reduced focal loss on the heatmap + L1 on the offset
at the landmark cell. Mixed precision keeps it inside 4 GB. After each epoch a
quick accuracy check runs on subsets of eval200 and the held-out val_resize60;
the checkpoint is kept on the held-out score, because beating our own data is
not the goal -- generalising to a different generator is.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

# Keep the 4 GB card stable: fixed cuDNN algos, no autotuner workspace spikes.
# (Fragmentation is capped via PYTORCH_CUDA_ALLOC_CONF, set on the launch line
# since it must precede the torch import to take effect.)
torch.backends.cudnn.benchmark = False

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from driftmatch.model import DriftMatchNet
from driftmatch.data import PairSet
from driftmatch.infer import locate_net


def focal_loss(logits, gt_hm):
    """CenterNet penalty-reduced focal loss. Positives are the exact peaks."""
    pred = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = gt_hm.eq(1.0).float()
    neg = 1.0 - pos
    neg_w = torch.pow(1.0 - gt_hm, 4)
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_w * neg
    n = pos.sum().clamp(min=1.0)
    return -(pos_loss.sum() + neg_loss.sum()) / n


def offset_loss(pred_off, off_t, cell):
    b = torch.arange(pred_off.shape[0], device=pred_off.device)
    picked = pred_off[b, :, cell[:, 0], cell[:, 1]]     # (B,2)
    return F.l1_loss(picked, off_t)


def load_gray(p):
    return np.asarray(Image.open(p).convert("L"))


def quick_eval(net, root, device, n=60, center_rule=True):
    """within-5px hit-rate on the first n pairs of a dataset."""
    root = pathlib.Path(root)
    rows = list(csv.DictReader((root / "labels.csv").open()))[:n]
    hit = 0
    for r in rows:
        ref = load_gray(root / r["ref_path"])
        wide = load_gray(root / r["wide_path"])
        x, y = locate_net(net, ref, wide, device, center_rule=center_rule)
        e = np.hypot(x - float(r["gt_x"]), y - float(r["gt_y"]))
        hit += (e <= 5)
    return 100.0 * hit / len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="driftmatch/checkpoints")
    ap.add_argument("--lambda-off", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the cached training pool (RAM safety on 16 GB)")
    ap.add_argument("--eval1", default="data/eval200")
    ap.add_argument("--eval2", default="data/val_resize60",
                    help="held-out set the best checkpoint is kept on")
    ap.add_argument("--resume", default=None,
                    help="warm-start the model weights from a checkpoint")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = pathlib.Path(a.out); outdir.mkdir(parents=True, exist_ok=True)

    ds = PairSet(a.data, limit=a.limit)
    # pin_memory stays off: on this 4 GB Windows card the pinned staging buffers
    # tipped training into OOM, while parallel workers alone are fine.
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                    pin_memory=False, drop_last=True, persistent_workers=a.workers > 0)
    print(f"train pairs: {len(ds)}  batches/epoch: {len(dl)}  device: {device}")

    net = DriftMatchNet(C=64).to(device)
    best = -1.0
    if a.resume:
        from driftmatch.infer import load_checkpoint
        ck = load_checkpoint(a.resume, map_location=device)
        net.load_state_dict(ck["model"])
        best = float(ck.get("acc_val", -1.0))
        print(f"resumed {a.resume}: prev epoch {ck.get('epoch','?')}, held-out {best:.1f}%")
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    for ep in range(a.epochs):
        net.train(); t0 = time.perf_counter(); run = 0.0
        for ref, wide, hm, off, cell, _gt in dl:
            ref, wide = ref.to(device), wide.to(device)
            hm, off, cell = hm.to(device), off.to(device), cell.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                pred_hm, pred_off = net(ref, wide)
                l_hm = focal_loss(pred_hm, hm)
                l_off = offset_loss(pred_off, off, cell)
                loss = l_hm + a.lambda_off * l_off
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)  # stability
            scaler.step(opt); scaler.update()
            run += float(loss)
        sched.step()

        net.eval()
        import gc
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        acc_e = quick_eval(net, a.eval1, device, n=50)
        acc_v = quick_eval(net, a.eval2, device, n=60)
        gc.collect()
        dt = time.perf_counter() - t0
        print(f"ep {ep+1:02d}/{a.epochs}  loss {run/len(dl):.3f}  "
              f"eval1 {acc_e:5.1f}%  held-out {acc_v:5.1f}%  ({dt:.0f}s)")

        # store metadata as plain Python floats/ints so the checkpoint stays a
        # pure-tensor file loadable with weights_only=True (no numpy globals)
        torch.save({"model": net.state_dict(), "epoch": int(ep + 1),
                    "acc_eval": float(acc_e), "acc_val": float(acc_v)}, outdir / "last.pt")
        if acc_v > best:
            best = acc_v
            torch.save({"model": net.state_dict(), "epoch": int(ep + 1),
                        "acc_eval": float(acc_e), "acc_val": float(acc_v)}, outdir / "best.pt")
            print(f"    -> new best held-out {best:.1f}%  (saved best.pt)")

    print(f"done. best held-out {best:.1f}%")


if __name__ == "__main__":
    main()
