"""Evaluate a trained checkpoint on the held-out validation split.

Rebuilds the exact train/val split used in training (same images dir, events,
val fraction and seed, read from the checkpoint), then reports metrics for:

  * the trained UNet
  * baseline "all background": predict background for every pixel
  * baseline "oracle footprints + no-damage": copy the ground-truth building
    footprints and label every building pixel no-damage (the majority damage
    class). This gets localization for free, so it is a strong baseline for
    the background / no-damage classes and isolates damage classification.

Metrics (pixels with the ignore label are excluded everywhere):
  * per-class IoU over the 5 classes, mean IoU, pixel accuracy
  * on ground-truth building pixels only: damage accuracy and macro F1 over
    the 4 damage classes (a building pixel predicted as background counts as
    wrong)

Example:
    python evaluate.py --checkpoint checkpoints/unet_best.pt \\
        --save-examples docs --n-examples 4
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from dataset import XView2Dataset, make_splits  # noqa: E402
from mask_utils import BACKGROUND_CLASS, IGNORE_INDEX, NUM_CLASSES  # noqa: E402
from model import UNet  # noqa: E402
from train import CLASS_NAMES, pick_device  # noqa: E402

DAMAGE_CLASSES = [0, 1, 2, 3]
# no-damage, minor, major, destroyed, background ; ignore drawn grey
COLORS = ["#2ca02c", "#ffdd57", "#ff7f0e", "#d62728", "#000000"]


def confusion(targets, preds):
    valid = targets != IGNORE_INDEX
    return np.bincount(
        NUM_CLASSES * targets[valid].astype(np.int64) + preds[valid],
        minlength=NUM_CLASSES**2,
    ).reshape(NUM_CLASSES, NUM_CLASSES)


def summarize(conf):
    tp = np.diag(conf).astype(np.float64)
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    denom = tp + fp + fn
    iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)

    # Damage metrics restricted to ground-truth building pixels (rows 0..3).
    rows = conf[DAMAGE_CLASSES]  # (4, 5): GT damage class x predicted class
    d_tp = rows[:, DAMAGE_CLASSES].diagonal().astype(np.float64)
    d_support = rows.sum(1)
    d_pred = conf[:, DAMAGE_CLASSES][DAMAGE_CLASSES].sum(0)  # predicted among GT buildings
    prec = np.where(d_pred > 0, d_tp / np.maximum(d_pred, 1), 0.0)
    rec = np.where(d_support > 0, d_tp / np.maximum(d_support, 1), 0.0)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)

    return {
        "per_class_iou": {n: float(v) for n, v in zip(CLASS_NAMES, iou)},
        "mean_iou": float(np.nanmean(iou)),
        "pixel_acc": float(tp.sum() / max(conf.sum(), 1)),
        "building_damage_acc": float(d_tp.sum() / max(d_support.sum(), 1)),
        "building_damage_f1": {n: float(v) for n, v in zip(CLASS_NAMES[:4], f1)},
        "building_damage_macro_f1": float(f1.mean()),
    }


def colorize(mask):
    rgb = np.full(mask.shape + (3,), 0.5)
    for c, hexcol in enumerate(COLORS):
        rgb[mask == c] = matplotlib.colors.to_rgb(hexcol)
    return rgb


def save_panel(path, image, gt, pred, title):
    pre = image[:3].transpose(1, 2, 0)
    post = image[3:].transpose(1, 2, 0)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    for ax, img, name in zip(
        axes,
        [pre, post, colorize(gt), colorize(pred)],
        ["pre-disaster", "post-disaster", "ground truth", "prediction"],
    ):
        ax.imshow(img)
        ax.set_title(name)
        ax.axis("off")
    handles = [Patch(color=c, label=n) for c, n in zip(COLORS, CLASS_NAMES)]
    handles.append(Patch(color="#808080", label="ignored (un-classified)"))
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=80)
    plt.close(fig)


def fmt_table(results):
    names = list(results)
    lines = ["| metric | " + " | ".join(names) + " |",
             "|---|" + "---|" * len(names)]

    def row(label, get):
        lines.append(f"| {label} | " + " | ".join(f"{get(results[n]):.3f}" for n in names) + " |")

    for c in CLASS_NAMES:
        row(f"IoU {c}", lambda r, c=c: r["per_class_iou"][c])
    row("**mean IoU**", lambda r: r["mean_iou"])
    row("pixel accuracy", lambda r: r["pixel_acc"])
    row("damage acc. (building px)", lambda r: r["building_damage_acc"])
    for c in CLASS_NAMES[:4]:
        row(f"F1 {c} (building px)", lambda r, c=c: r["building_damage_f1"][c])
    row("**damage macro F1 (building px)**", lambda r: r["building_damage_macro_f1"])
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/unet_best.pt")
    parser.add_argument("--images-dir", default=None, help="override checkpoint value")
    parser.add_argument("--labels-dir", default=None, help="override checkpoint value")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-examples", default=None, help="dir for sample PNGs")
    parser.add_argument("--n-examples", type=int, default=4)
    parser.add_argument("--metrics-out", default=None, help="write metrics JSON here")
    return parser


@torch.no_grad()
def main():
    args = build_parser().parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    targs = ckpt["args"]
    images_dir = args.images_dir or targs["images_dir"]
    labels_dir = args.labels_dir or targs["labels_dir"]
    device = pick_device(args.device)

    _, val_ids = make_splits(
        images_dir, labels_dir, targs["val_frac"], targs["seed"], targs.get("events")
    )
    print(f"checkpoint epoch {ckpt['epoch']}  val images={len(val_ids)}  device={device}")

    model = UNet(in_channels=6, num_classes=NUM_CLASSES, base=targs["base"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    ds = XView2Dataset(images_dir, labels_dir, val_ids, targs["img_size"], augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    confs = {k: np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
             for k in ["UNet (ours)", "all background", "oracle footprints + no-damage"]}
    # (damaged-pixel count, index) to pick informative examples
    candidates = []
    idx = 0
    for images, masks in tqdm(loader, desc="eval"):
        preds = model(images.to(device)).argmax(1).cpu().numpy()
        masks = masks.numpy()
        for m, p in zip(masks, preds):
            t = m.ravel()
            confs["UNet (ours)"] += confusion(t, p.ravel())
            confs["all background"] += confusion(t, np.full_like(t, BACKGROUND_CLASS))
            oracle = np.where((t >= 0) & (t <= 3), 0, BACKGROUND_CLASS)
            confs["oracle footprints + no-damage"] += confusion(t, oracle)
            candidates.append((int(((m >= 1) & (m <= 3)).sum()), int((m <= 3).sum()), idx))
            idx += 1

    results = {k: summarize(c) for k, c in confs.items()}
    table = fmt_table(results)
    print(table)

    gt_pixels = confs["UNet (ours)"].sum(1)
    print("val GT pixel share: " + "  ".join(
        f"{n}={v / gt_pixels.sum():.4f}" for n, v in zip(CLASS_NAMES, gt_pixels)))

    if args.metrics_out:
        os.makedirs(os.path.dirname(args.metrics_out) or ".", exist_ok=True)
        with open(args.metrics_out, "w") as f:
            json.dump({"checkpoint_epoch": ckpt["epoch"], "train_args": targs,
                       "n_val_images": len(val_ids), "results": results,
                       "val_gt_pixel_counts": dict(zip(CLASS_NAMES, gt_pixels.tolist()))},
                      f, indent=2)

    if args.save_examples:
        os.makedirs(args.save_examples, exist_ok=True)
        # Most-damaged val tiles first, then the most building-dense ones.
        picked = sorted(candidates, reverse=True)[: args.n_examples]
        for k, (_, _, i) in enumerate(picked):
            image, mask = ds[i]
            pred = model(image[None].to(device)).argmax(1)[0].cpu().numpy()
            path = os.path.join(args.save_examples, f"example_{k + 1}.png")
            save_panel(path, image.numpy(), mask.numpy(), pred, ds.ids[i])
            print(f"saved {path}")


if __name__ == "__main__":
    main()
