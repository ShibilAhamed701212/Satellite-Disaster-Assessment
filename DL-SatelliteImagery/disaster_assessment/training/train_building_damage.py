"""
Lightweight Training Pipeline for Building Damage Assessment Model.

Supports:
    - 4-class Building Damage Segmentation (Undamaged, Minor, Major, Destroyed)
    - Mixed precision training on CUDA (NVIDIA GTX 1650 4GB VRAM)
    - Full CPU fallback
    - Checkpointing (best model + latest state)
    - Multi-class IoU, F1 score, precision, and accuracy metrics
    - Resume training support
"""

import argparse
import os
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add parent directory for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from disaster_assessment.datasets.building_damage_dataset import (
    create_damage_dataloaders,
    create_synthetic_damage_data,
)
from disaster_assessment.models.base_unet import get_device
from disaster_assessment.models.building_damage_model import (
    DAMAGE_CLASSES,
    NUM_DAMAGE_CLASSES,
    BuildingDamageUNet,
)


class MultiClassDiceLoss(nn.Module):
    """Multi-class Dice Loss for imbalanced segmentation."""

    def __init__(self, num_classes: int = 4, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        dice_total = 0.0
        for c in range(self.num_classes):
            p = probs[:, c, :, :]
            t = targets_one_hot[:, c, :, :]
            intersection = torch.sum(p * t)
            dice_c = (2.0 * intersection + self.smooth) / (torch.sum(p) + torch.sum(t) + self.smooth)
            dice_total += (1.0 - dice_c)

        return dice_total / self.num_classes


class CombinedDamageLoss(nn.Module):
    """Cross-Entropy + Dice Loss combination."""

    def __init__(self, num_classes: int = 4, ce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = MultiClassDiceLoss(num_classes=num_classes)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(logits, targets) + self.dice_weight * self.dice(logits, targets)


def compute_metrics(preds: torch.Tensor, targets: torch.Tensor, num_classes: int = 4) -> Dict[str, float]:
    """Compute overall accuracy, mean IoU, and macro F1."""
    preds = preds.view(-1)
    targets = targets.view(-1)

    # Pixel accuracy
    correct = (preds == targets).sum().item()
    total = targets.numel()
    accuracy = correct / max(total, 1)

    # Per-class IoU and F1
    ious = []
    f1s = []
    for c in range(num_classes):
        p_c = preds == c
        t_c = targets == c

        tp = (p_c & t_c).sum().item()
        fp = (p_c & (~t_c)).sum().item()
        fn = ((~p_c) & t_c).sum().item()

        union = tp + fp + fn
        iou = tp / max(union, 1)
        ious.append(iou)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-6)
        f1s.append(f1)

    return {
        "accuracy": accuracy * 100.0,
        "mean_iou": (sum(ious) / len(ious)) * 100.0,
        "macro_f1": (sum(f1s) / len(f1s)) * 100.0,
    }


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    use_cuda = device.type == "cuda" and scaler is not None

    for batch in loader:
        pre = batch["pre_image"].to(device)
        post = batch["post_image"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()

        if use_cuda:
            with torch.amp.autocast("cuda"):
                logits = model(pre, post)
                loss = criterion(logits, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(pre, post)
            loss = criterion(logits, mask)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.detach().cpu())
        all_targets.append(mask.detach().cpu())

    avg_loss = total_loss / max(len(loader), 1)
    metrics = compute_metrics(torch.cat(all_preds), torch.cat(all_targets))
    return avg_loss, metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in loader:
        pre = batch["pre_image"].to(device)
        post = batch["post_image"].to(device)
        mask = batch["mask"].to(device)

        logits = model(pre, post)
        loss = criterion(logits, mask)

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(mask.cpu())

    avg_loss = total_loss / max(len(loader), 1)
    metrics = compute_metrics(torch.cat(all_preds), torch.cat(all_targets))
    return avg_loss, metrics


def train_building_damage(
    data_dir: str,
    output_dir: str = "DL-SatelliteImagery/disaster_assessment/weights/building_damage",
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 1e-3,
    resume_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> str:
    """
    Main training function. Returns path to best saved model weights.
    """
    if device is None:
        device = get_device()

    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, "best_damage_model.pth")
    latest_ckpt_path = os.path.join(output_dir, "latest_checkpoint.pth")

    train_loader, val_loader = create_damage_dataloaders(data_dir, batch_size=batch_size)
    if len(train_loader.dataset) == 0:
        raise ValueError(f"No training data found in {data_dir}/train")

    model = BuildingDamageUNet(num_classes=NUM_DAMAGE_CLASSES).to(device)
    criterion = CombinedDamageLoss(num_classes=NUM_DAMAGE_CLASSES)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    start_epoch = 1
    best_val_iou = 0.0

    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_iou = ckpt.get("best_val_iou", 0.0)
        print(f"[Training] Resumed from {resume_path} (epoch {start_epoch})")

    print(f"\n{'='*60}")
    print(f"  BUILDING DAMAGE ASSESSMENT TRAINING PIPELINE")
    print(f"  Device: {device} | Epochs: {epochs} | Batch Size: {batch_size}")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        scheduler.step()

        val_msg = ""
        val_iou = train_metrics["mean_iou"]
        if val_loader is not None and len(val_loader.dataset) > 0:
            val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
            val_iou = val_metrics["mean_iou"]
            val_msg = f" | Val Loss: {val_loss:.4f} | Val mIoU: {val_iou:.1f}% | Val Acc: {val_metrics['accuracy']:.1f}%"

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train mIoU: {train_metrics['mean_iou']:.1f}% | Train Acc: {train_metrics['accuracy']:.1f}%"
            f"{val_msg}"
        )

        # Checkpoint saving
        is_best = val_iou >= best_val_iou
        if is_best:
            best_val_iou = val_iou
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_iou": best_val_iou,
                },
                best_model_path,
            )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_iou": best_val_iou,
            },
            latest_ckpt_path,
        )

    print(f"\n✓ Training Complete. Best model saved to: {best_model_path}")
    return best_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Building Damage Model")
    parser.add_argument("--data_dir", type=str, default="data/damage_dataset")
    parser.add_argument("--output_dir", type=str, default="DL-SatelliteImagery/disaster_assessment/weights/building_damage")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dry_run", action="store_true", help="Generate synthetic data and train 1 epoch")
    args = parser.parse_args()

    if args.dry_run:
        print("[Dry Run] Generating synthetic training data...")
        create_synthetic_damage_data(args.data_dir, num_samples=8)
        args.epochs = 1

    train_building_damage(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
