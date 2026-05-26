from __future__ import annotations

import argparse

import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits).view(logits.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersection = (probs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        return bce + (1.0 - dice.mean())


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).view(logits.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersection = (probs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (alpha_t * (1.0 - pt).pow(self.gamma) * bce).mean()


class ComboSegmentationLoss(nn.Module):
    def __init__(
        self,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        focal_weight: float = 0.5,
        smooth: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.dice = DiceLoss(smooth=smooth)
        self.bce = nn.BCEWithLogitsLoss()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total = logits.new_tensor(0.0)
        if self.dice_weight:
            total = total + self.dice_weight * self.dice(logits, targets)
        if self.bce_weight:
            total = total + self.bce_weight * self.bce(logits, targets)
        if self.focal_weight:
            total = total + self.focal_weight * self.focal(logits, targets)
        return total


def build_criterion(args: argparse.Namespace) -> nn.Module:
    if args.loss == "dice_bce":
        return DiceBCELoss(smooth=args.dice_smooth)
    if args.loss == "combo":
        return ComboSegmentationLoss(
            dice_weight=args.loss_dice_weight,
            bce_weight=args.loss_bce_weight,
            focal_weight=args.loss_focal_weight,
            smooth=args.dice_smooth,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
        )
    if args.loss == "bce":
        return nn.BCEWithLogitsLoss()
    if args.loss == "dice":
        return DiceLoss(smooth=args.dice_smooth)
    if args.loss == "focal":
        return FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    raise ValueError(f"Unsupported loss: {args.loss}")
