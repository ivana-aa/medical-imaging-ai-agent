from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .postprocess import PostProcessConfig, apply_postprocess_tensor


@dataclass
class Metrics:
    loss: float
    dice: float
    iou: float
    precision: float
    recall: float


@dataclass
class ThresholdMetrics:
    threshold: float
    dice: float
    iou: float
    precision: float
    recall: float
    postprocess: Optional[dict]


def binary_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
    postprocess: Optional[PostProcessConfig] = None,
) -> Metrics:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    preds = apply_postprocess_tensor(preds, postprocess)
    targets = (targets > 0.5).float()

    dims = (1, 2, 3)
    intersection = (preds * targets).sum(dim=dims)
    pred_sum = preds.sum(dim=dims)
    target_sum = targets.sum(dim=dims)
    union = pred_sum + target_sum - intersection

    dice = ((2.0 * intersection + 1.0) / (pred_sum + target_sum + 1.0)).mean().item()
    iou = ((intersection + 1.0) / (union + 1.0)).mean().item()
    precision = ((intersection + 1.0) / (pred_sum + 1.0)).mean().item()
    recall = ((intersection + 1.0) / (target_sum + 1.0)).mean().item()
    return Metrics(loss=0.0, dice=dice, iou=iou, precision=precision, recall=recall)


@torch.no_grad()
def predict_logits(model: nn.Module, images: torch.Tensor, tta: str = "off") -> torch.Tensor:
    if tta == "off":
        return model(images)
    if tta != "flip":
        raise ValueError(f"Unsupported TTA mode: {tta}")

    logits = model(images)
    horizontal = torch.flip(model(torch.flip(images, dims=[3])), dims=[3])
    vertical = torch.flip(model(torch.flip(images, dims=[2])), dims=[2])
    return (logits + horizontal + vertical) / 3.0


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split_name: str,
    threshold: float,
    postprocess: Optional[PostProcessConfig] = None,
    tta: str = "off",
) -> Metrics:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_samples = 0

    for images, masks, _ in tqdm(loader, desc=split_name, leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, masks)
        metric_logits = logits if tta == "off" else predict_logits(model, images, tta=tta)
        metrics = binary_metrics(metric_logits, masks, threshold, postprocess)
        batch_size = images.size(0)

        total_loss += float(loss.item()) * batch_size
        total_dice += metrics.dice * batch_size
        total_iou += metrics.iou * batch_size
        total_precision += metrics.precision * batch_size
        total_recall += metrics.recall * batch_size
        total_samples += batch_size

    return Metrics(
        loss=total_loss / total_samples,
        dice=total_dice / total_samples,
        iou=total_iou / total_samples,
        precision=total_precision / total_samples,
        recall=total_recall / total_samples,
    )


@torch.no_grad()
def find_best_threshold(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    thresholds: list[float],
    postprocess_configs: list[PostProcessConfig],
    split_name: str = "threshold",
    search_output_path: Optional[Path] = None,
    tta: str = "off",
) -> ThresholdMetrics:
    model.eval()
    sums = {
        (threshold, index): {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0, "samples": 0}
        for threshold in thresholds
        for index in range(len(postprocess_configs))
    }

    for images, masks, _ in tqdm(loader, desc=split_name, leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = predict_logits(model, images, tta=tta)
        batch_size = images.size(0)
        for threshold in thresholds:
            for index, config in enumerate(postprocess_configs):
                metrics = binary_metrics(logits, masks, threshold, config)
                item = sums[(threshold, index)]
                item["dice"] += metrics.dice * batch_size
                item["iou"] += metrics.iou * batch_size
                item["precision"] += metrics.precision * batch_size
                item["recall"] += metrics.recall * batch_size
                item["samples"] += batch_size

    rows = []
    best: Optional[ThresholdMetrics] = None
    for threshold in thresholds:
        for index, config in enumerate(postprocess_configs):
            item = sums[(threshold, index)]
            samples = item["samples"]
            row = {
                "threshold": threshold,
                "dice": item["dice"] / samples,
                "iou": item["iou"] / samples,
                "precision": item["precision"] / samples,
                "recall": item["recall"] / samples,
                "postprocess": config.to_dict(),
            }
            rows.append(row)
            candidate = ThresholdMetrics(
                threshold=threshold,
                dice=row["dice"],
                iou=row["iou"],
                precision=row["precision"],
                recall=row["recall"],
                postprocess=config.to_dict(),
            )
            if best is None or candidate.dice > best.dice:
                best = candidate

    if search_output_path is not None:
        search_output_path.parent.mkdir(parents=True, exist_ok=True)
        with search_output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["threshold", "dice", "iou", "precision", "recall", "postprocess"],
            )
            writer.writeheader()
            writer.writerows(rows)

    assert best is not None
    return best
