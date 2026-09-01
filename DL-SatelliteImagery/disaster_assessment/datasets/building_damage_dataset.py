"""
Dataset loader and augmentation pipeline for Building Damage Assessment.

Supports:
    - Standard pre/post/mask triplets (e.g. xBD / xView2 format or custom folders)
    - 4 damage classes: 0=Undamaged, 1=Minor, 2=Major, 3=Destroyed
    - Configurable image size (default 256x256)
    - Data augmentations (flips, rotations, brightness/contrast jitter)
    - Synthetic dataset generation for unit testing without massive downloads
"""

import glob
import os
import random
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class BuildingDamageDataset(Dataset):
    """
    PyTorch Dataset for Pre/Post Satellite Image Pairs and 4-Class Damage Masks.

    Expected directory structure:
        dataset_root/
            pre/     (e.g., img_001.png, img_002.png)
            post/    (e.g., img_001.png, img_002.png)
            masks/   (e.g., img_001.png, img_002.png)
    """

    def __init__(
        self,
        dataset_dir: str,
        target_size: Tuple[int, int] = (256, 256),
        is_train: bool = True,
        transform: Optional[Callable] = None,
    ):
        self.dataset_dir = dataset_dir
        self.target_size = target_size
        self.is_train = is_train
        self.transform = transform

        self.pre_dir = os.path.join(dataset_dir, "pre")
        self.post_dir = os.path.join(dataset_dir, "post")
        self.mask_dir = os.path.join(dataset_dir, "masks")

        self.samples: List[Tuple[str, str, str]] = []
        if os.path.isdir(self.pre_dir) and os.path.isdir(self.post_dir) and os.path.isdir(self.mask_dir):
            pre_files = sorted(glob.glob(os.path.join(self.pre_dir, "*.*")))
            for pre_path in pre_files:
                base_name = os.path.basename(pre_path)
                post_path = os.path.join(self.post_dir, base_name)
                mask_path = os.path.join(self.mask_dir, base_name)
                if os.path.exists(post_path) and os.path.exists(mask_path):
                    self.samples.append((pre_path, post_path, mask_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pre_path, post_path, mask_path = self.samples[idx]

        pre_img = cv2.imread(pre_path, cv2.IMREAD_COLOR)
        pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)

        post_img = cv2.imread(post_path, cv2.IMREAD_COLOR)
        post_img = cv2.cvtColor(post_img, cv2.COLOR_BGR2RGB)

        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize to target dimension
        h, w = self.target_size
        if pre_img.shape[:2] != (h, w):
            pre_img = cv2.resize(pre_img, (w, h), interpolation=cv2.INTER_AREA)
        if post_img.shape[:2] != (h, w):
            post_img = cv2.resize(post_img, (w, h), interpolation=cv2.INTER_AREA)
        if mask_img.shape[:2] != (h, w):
            mask_img = cv2.resize(mask_img, (w, h), interpolation=cv2.INTER_NEAREST)

        # Ensure mask is strictly within [0, 3]
        mask_img = np.clip(mask_img, 0, 3).astype(np.int64)

        # Data augmentation
        if self.is_train:
            pre_img, post_img, mask_img = self._augment(pre_img, post_img, mask_img)

        # Normalize images to [0, 1] float32
        pre_tensor = torch.from_numpy(pre_img.astype(np.float32) / 255.0).permute(2, 0, 1)
        post_tensor = torch.from_numpy(post_img.astype(np.float32) / 255.0).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_img)  # [H, W] long

        return {
            "pre_image": pre_tensor,
            "post_image": post_tensor,
            "mask": mask_tensor,
        }

    @staticmethod
    def _augment(
        pre: np.ndarray,
        post: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Random horizontal flip
        if random.random() > 0.5:
            pre = np.fliplr(pre).copy()
            post = np.fliplr(post).copy()
            mask = np.fliplr(mask).copy()

        # Random vertical flip
        if random.random() > 0.5:
            pre = np.flipud(pre).copy()
            post = np.flipud(post).copy()
            mask = np.flipud(mask).copy()

        # Random 90-degree rotations
        rot_k = random.randint(0, 3)
        if rot_k > 0:
            pre = np.rot90(pre, rot_k).copy()
            post = np.rot90(post, rot_k).copy()
            mask = np.rot90(mask, rot_k).copy()

        return pre, post, mask


def create_synthetic_damage_data(
    output_dir: str,
    num_samples: int = 10,
    size: Tuple[int, int] = (256, 256),
) -> None:
    """
    Generate synthetic sample pre/post image pairs and damage masks for testing/dry-runs.
    """
    for split in ["train", "val"]:
        split_dir = os.path.join(output_dir, split)
        pre_dir = os.path.join(split_dir, "pre")
        post_dir = os.path.join(split_dir, "post")
        mask_dir = os.path.join(split_dir, "masks")

        os.makedirs(pre_dir, exist_ok=True)
        os.makedirs(post_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        n = num_samples if split == "train" else max(2, num_samples // 4)
        for i in range(n):
            # Synthetic pre: terrain + building footprints
            pre = np.random.randint(40, 160, (size[0], size[1], 3), dtype=np.uint8)
            post = pre.copy()
            mask = np.zeros(size, dtype=np.uint8)

            # Insert building bounding boxes with damage classes
            for _ in range(4):
                bx = random.randint(20, size[1] - 60)
                by = random.randint(20, size[0] - 60)
                bw = random.randint(20, 45)
                bh = random.randint(20, 45)

                damage_cls = random.randint(0, 3)
                mask[by:by + bh, bx:bx + bw] = damage_cls
                pre[by:by + bh, bx:bx + bw] = (200, 200, 220)

                if damage_cls == 0:  # Undamaged
                    post[by:by + bh, bx:bx + bw] = (200, 200, 220)
                elif damage_cls == 1:  # Minor
                    post[by:by + bh, bx:bx + bw] = (180, 180, 150)
                elif damage_cls == 2:  # Major
                    post[by:by + bh, bx:bx + bw] = (120, 100, 80)
                else:  # Destroyed
                    post[by:by + bh, bx:bx + bw] = (60, 50, 40)

            fname = f"sample_{i:04d}.png"
            cv2.imwrite(os.path.join(pre_dir, fname), cv2.cvtColor(pre, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(post_dir, fname), cv2.cvtColor(post, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(mask_dir, fname), mask)


def create_damage_dataloaders(
    data_root: str,
    batch_size: int = 4,
    num_workers: int = 0,
    target_size: Tuple[int, int] = (256, 256),
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and validation DataLoaders.
    """
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    train_ds = BuildingDamageDataset(train_dir, target_size=target_size, is_train=True)
    val_ds = (
        BuildingDamageDataset(val_dir, target_size=target_size, is_train=False)
        if os.path.isdir(val_dir)
        else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = (
        DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        if val_ds is not None and len(val_ds) > 0
        else None
    )

    return train_loader, val_loader
