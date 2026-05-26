from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ssl_project.augmentations import build_eval_image_transform, build_mask_transform
from ssl_project.config import load_config, resolve_path, section
from ssl_project.data import SegmentationPairDataset
from ssl_project.models import ResNetEncoder, ResNetUNet
from ssl_project.utils import ensure_dir, get_device, seed_everything, write_json


def build_loader(
    data_root: Path,
    split: str,
    data_config: dict[str, Any],
    image_transform: Any,
    mask_transform: Any,
) -> DataLoader:
    dataset = SegmentationPairDataset(
        root=data_root,
        split=split,
        image_transform=image_transform,
        mask_transform=mask_transform,
        image_dir_name=str(data_config.get("image_dir_name", "images")),
        label_dir_name=str(data_config.get("label_dir_name", "labels")),
        image_channels=int(data_config.get("image_channels", 1)),
    )
    return DataLoader(
        dataset,
        batch_size=int(data_config.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(data_config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )


def collect_probabilities(
    model: ResNetUNet,
    loader: DataLoader,
    device: torch.device,
    desc: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    probabilities: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    progress = tqdm(loader, desc=desc, leave=False, disable=not sys.stderr.isatty())
    with torch.no_grad():
        for images, masks, _ in progress:
            logits = model(images.to(device, non_blocking=True))
            probabilities.append(torch.sigmoid(logits).cpu())
            targets.append((masks >= 0.5).float().cpu())
    return torch.cat(probabilities, dim=0), torch.cat(targets, dim=0)


def metrics_from_probs(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
    eps: float = 1e-7,
) -> dict[str, float]:
    preds = (probabilities >= threshold).float()
    dims = tuple(range(1, preds.ndim))
    tp = torch.sum(preds * targets, dim=dims)
    fp = torch.sum(preds * (1.0 - targets), dim=dims)
    fn = torch.sum((1.0 - preds) * targets, dim=dims)
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return {
        "threshold": float(threshold),
        "dice": float(dice.mean().item()),
        "iou": float(iou.mean().item()),
        "precision": float(precision.mean().item()),
        "recall": float(recall.mean().item()),
    }


def build_thresholds(start: float, end: float, step: float) -> list[float]:
    thresholds: list[float] = []
    value = start
    while value <= end + step * 0.5:
        thresholds.append(round(value, 6))
        value += step
    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune a segmentation probability threshold on validation data.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=str, default=None, help="Override project.output_dir.")
    parser.add_argument("--start", type=float, default=0.30)
    parser.add_argument("--end", type=float, default=0.85)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument("--output-name", type=str, default="threshold_sweep.json")
    args = parser.parse_args()

    config = load_config(args.config)
    project_config = section(config, "project")
    data_config = section(config, "data")
    model_config = section(config, "model")
    if args.output_dir is not None:
        project_config["output_dir"] = args.output_dir

    seed_everything(int(project_config.get("seed", 42)))
    output_dir = ensure_dir(resolve_path(project_config.get("output_dir", "runs/decoder"), PROJECT_ROOT))
    data_root = resolve_path(data_config.get("root", "../Dataset"), PROJECT_ROOT)
    image_channels = int(data_config.get("image_channels", 1))
    image_size = int(data_config.get("image_size", 128))
    image_transform = build_eval_image_transform(image_size, image_channels)
    mask_transform = build_mask_transform(image_size)

    device = get_device()
    encoder = ResNetEncoder(
        backbone=str(model_config.get("backbone", "resnet18")),
        in_channels=image_channels,
    )
    model = ResNetUNet(
        encoder=encoder,
        freeze_encoder=bool(model_config.get("freeze_encoder", False)),
    ).to(device)
    checkpoint_path = resolve_path(args.checkpoint, PROJECT_ROOT)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    val_loader = build_loader(data_root, args.val_split, data_config, image_transform, mask_transform)
    test_loader = build_loader(data_root, args.test_split, data_config, image_transform, mask_transform)
    val_probs, val_targets = collect_probabilities(model, val_loader, device, desc="collect val")
    test_probs, test_targets = collect_probabilities(model, test_loader, device, desc="collect test")

    thresholds = build_thresholds(args.start, args.end, args.step)
    val_results = [metrics_from_probs(val_probs, val_targets, threshold) for threshold in thresholds]
    best_val = max(val_results, key=lambda item: (item["dice"], item["iou"]))
    test_at_best = metrics_from_probs(test_probs, test_targets, best_val["threshold"])
    test_at_default = metrics_from_probs(test_probs, test_targets, 0.5)

    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "threshold_range": {
            "start": args.start,
            "end": args.end,
            "step": args.step,
        },
        "selection_metric": "val_dice",
        "best_val": best_val,
        "test_at_best_val_threshold": test_at_best,
        "test_at_default_threshold": test_at_default,
        "val_results": val_results,
    }
    write_json(output_dir / args.output_name, result)
    print(
        "best_threshold={threshold:.3f} val_dice={val_dice:.5f} val_iou={val_iou:.5f} "
        "test_dice={test_dice:.5f} test_iou={test_iou:.5f} precision={precision:.5f} recall={recall:.5f}".format(
            threshold=best_val["threshold"],
            val_dice=best_val["dice"],
            val_iou=best_val["iou"],
            test_dice=test_at_best["dice"],
            test_iou=test_at_best["iou"],
            precision=test_at_best["precision"],
            recall=test_at_best["recall"],
        )
    )


if __name__ == "__main__":
    main()
