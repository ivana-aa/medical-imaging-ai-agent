from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SegmentationMetrics:
    loss: float
    dice: float
    iou: float
    precision: float
    recall: float


def binary_segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> dict[str, float]:
    preds = (torch.sigmoid(logits) >= threshold).float()
    targets = (targets >= 0.5).float()
    dims = tuple(range(1, preds.ndim))
    tp = torch.sum(preds * targets, dim=dims)
    fp = torch.sum(preds * (1.0 - targets), dim=dims)
    fn = torch.sum((1.0 - preds) * targets, dim=dims)
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return {
        "dice": float(dice.mean().item()),
        "iou": float(iou.mean().item()),
        "precision": float(precision.mean().item()),
        "recall": float(recall.mean().item()),
    }
