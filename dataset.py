"""xView2 pre+post image-pair dataset for damage segmentation.

Each sample is a 6-channel tensor (pre RGB stacked with post RGB) and an
integer label mask produced by ``mask_utils.polygons_to_mask``.
"""

import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from mask_utils import polygons_to_mask

PRE_SUFFIX = "_pre_disaster"
POST_SUFFIX = "_post_disaster"


def _sample_paths(image_id, images_dir, labels_dir):
    pre = os.path.join(images_dir, f"{image_id}{PRE_SUFFIX}.png")
    post = os.path.join(images_dir, f"{image_id}{POST_SUFFIX}.png")
    label = os.path.join(labels_dir, f"{image_id}{POST_SUFFIX}.json")
    return pre, post, label


def list_ids(images_dir, labels_dir):
    """Return sorted base ids that have a pre image, post image, and post JSON."""
    ids = []
    for fname in os.listdir(images_dir):
        if not fname.endswith(f"{PRE_SUFFIX}.png"):
            continue
        image_id = fname[: -len(f"{PRE_SUFFIX}.png")]
        pre, post, label = _sample_paths(image_id, images_dir, labels_dir)
        if os.path.exists(pre) and os.path.exists(post) and os.path.exists(label):
            ids.append(image_id)
    return sorted(ids)


def make_splits(images_dir, labels_dir, val_frac=0.15, seed=42):
    """Deterministically split ids into (train_ids, val_ids)."""
    ids = list_ids(images_dir, labels_dir)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = int(round(len(ids) * val_frac))
    return ids[n_val:], ids[:n_val]


def _read_rgb(path, img_size):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (img_size, img_size):
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return img


class XView2Dataset(Dataset):
    def __init__(self, images_dir, labels_dir, ids, img_size=512, augment=False):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.ids = list(ids)
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.ids)

    def _maybe_augment(self, image, mask):
        # Consistent flips applied to both the 6-channel image and the mask.
        if random.random() < 0.5:
            image = image[:, :, ::-1]
            mask = mask[:, ::-1]
        if random.random() < 0.5:
            image = image[:, ::-1, :]
            mask = mask[::-1, :]
        return np.ascontiguousarray(image), np.ascontiguousarray(mask)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        pre_path, post_path, label_path = _sample_paths(
            image_id, self.images_dir, self.labels_dir
        )

        pre = _read_rgb(pre_path, self.img_size)
        post = _read_rgb(post_path, self.img_size)

        mask = polygons_to_mask(label_path, image_size=(1024, 1024))
        if mask.shape != (self.img_size, self.img_size):
            mask = cv2.resize(
                mask,
                (self.img_size, self.img_size),
                interpolation=cv2.INTER_NEAREST,
            )

        # H, W, 6  (pre RGB then post RGB)
        image = np.concatenate([pre, post], axis=2)

        if self.augment:
            image, mask = self._maybe_augment(image, mask)

        image = torch.from_numpy(
            image.transpose(2, 0, 1).astype(np.float32) / 255.0
        )
        mask = torch.from_numpy(mask.astype(np.int64))
        return image, mask
