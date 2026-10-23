"""Train the xView2 damage-segmentation UNet.

Example:
    python train.py --images-dir data/train/images \
                     --labels-dir data/train/labels \
                     --epochs 30 --batch-size 4 --img-size 512
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import XView2Dataset, make_splits
from mask_utils import IGNORE_INDEX, NUM_CLASSES
from model import UNet

CLASS_NAMES = ["no-damage", "minor", "major", "destroyed", "background"]


def pick_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    total_loss, n_batches = 0.0, 0

    for images, masks in tqdm(loader, desc="val", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        total_loss += criterion(logits, masks).item()
        n_batches += 1

        preds = logits.argmax(1).cpu().numpy().ravel()
        targets = masks.cpu().numpy().ravel()
        valid = targets != IGNORE_INDEX
        conf += np.bincount(
            NUM_CLASSES * targets[valid] + preds[valid],
            minlength=NUM_CLASSES**2,
        ).reshape(NUM_CLASSES, NUM_CLASSES)

    tp = np.diag(conf)
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    denom = tp + fp + fn
    iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
    present = denom > 0
    mean_iou = float(np.nanmean(iou[present])) if present.any() else 0.0
    pixel_acc = float(tp.sum() / max(conf.sum(), 1))

    return {
        "loss": total_loss / max(n_batches, 1),
        "mean_iou": mean_iou,
        "pixel_acc": pixel_acc,
        "per_class_iou": iou,
    }


def train_one_epoch(model, loader, device, criterion, optimizer):
    model.train()
    running, n = 0.0, 0
    for images, masks in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running += loss.item()
        n += 1
    return running / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", default="data/train/images")
    parser.add_argument("--labels-dir", default="data/train/labels")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--base", type=int, default=32, help="UNet base channels")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="checkpoints/unet_best.pt")
    args = parser.parse_args()

    if not os.path.isdir(args.images_dir):
        raise SystemExit(f"images dir not found: {args.images_dir}")
    if not os.path.isdir(args.labels_dir):
        raise SystemExit(f"labels dir not found: {args.labels_dir}")

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    print(f"device: {device}")

    train_ids, val_ids = make_splits(
        args.images_dir, args.labels_dir, args.val_frac, args.seed
    )
    if not train_ids:
        raise SystemExit("no training samples found (need pre+post png and post json)")
    print(f"train={len(train_ids)}  val={len(val_ids)}")

    train_ds = XView2Dataset(
        args.images_dir, args.labels_dir, train_ids, args.img_size, augment=True
    )
    val_ds = XView2Dataset(
        args.images_dir, args.labels_dir, val_ids, args.img_size, augment=False
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    model = UNet(in_channels=6, num_classes=NUM_CLASSES, base=args.base).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best_iou = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, device, criterion, optimizer
        )
        metrics = (
            evaluate(model, val_loader, device, criterion)
            if len(val_ds) > 0
            else None
        )

        msg = f"epoch {epoch:3d}  train_loss={train_loss:.4f}"
        if metrics:
            per_class = "  ".join(
                f"{name}={iou:.3f}"
                for name, iou in zip(CLASS_NAMES, metrics["per_class_iou"])
            )
            msg += (
                f"  val_loss={metrics['loss']:.4f}"
                f"  mIoU={metrics['mean_iou']:.4f}"
                f"  pixAcc={metrics['pixel_acc']:.4f}\n    {per_class}"
            )
        print(msg)

        score = metrics["mean_iou"] if metrics else -train_loss
        if score > best_iou:
            best_iou = score
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "score": score},
                args.out,
            )
            print(f"    saved checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
