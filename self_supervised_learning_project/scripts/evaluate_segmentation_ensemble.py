from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
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


def build_model(
    model_config: dict[str, Any],
    image_channels: int,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[ResNetUNet, dict[str, Any]]:
    encoder = ResNetEncoder(
        backbone=str(model_config.get("backbone", "resnet18")),
        in_channels=image_channels,
    )
    model = ResNetUNet(
        encoder=encoder,
        freeze_encoder=bool(model_config.get("freeze_encoder", False)),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def collect_ensemble_probabilities(
    models: list[ResNetUNet],
    loader: DataLoader,
    device: torch.device,
    desc: str,
    tta_modes: tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    progress = tqdm(loader, desc=desc, leave=False, disable=not sys.stderr.isatty())
    with torch.no_grad():
        for images, masks, _ in progress:
            images = images.to(device, non_blocking=True)
            batch_probability = None
            for model in models:
                probability = predict_probability(model, images, tta_modes=tta_modes)
                if batch_probability is None:
                    batch_probability = probability
                else:
                    batch_probability = batch_probability + probability
            if batch_probability is None:
                raise RuntimeError("No models were provided for ensemble evaluation.")
            batch_probability = batch_probability / float(len(models))
            probabilities.append(batch_probability.cpu())
            targets.append((masks >= 0.5).float().cpu())
    return torch.cat(probabilities, dim=0), torch.cat(targets, dim=0)


def predict_probability(
    model: ResNetUNet,
    images: torch.Tensor,
    tta_modes: tuple[str, ...],
) -> torch.Tensor:
    if not tta_modes:
        return torch.sigmoid(model(images))

    probability = torch.sigmoid(model(images))
    count = 1

    if "h" in tta_modes:
        flipped = torch.flip(images, dims=(-1,))
        probability = probability + torch.flip(torch.sigmoid(model(flipped)), dims=(-1,))
        count += 1

    if "v" in tta_modes:
        flipped = torch.flip(images, dims=(-2,))
        probability = probability + torch.flip(torch.sigmoid(model(flipped)), dims=(-2,))
        count += 1

    if "hv" in tta_modes:
        flipped = torch.flip(images, dims=(-2, -1))
        probability = probability + torch.flip(torch.sigmoid(model(flipped)), dims=(-2, -1))
        count += 1

    return probability / float(count)


def metrics_from_probs(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
    min_area: int = 0,
    eps: float = 1e-7,
) -> dict[str, float]:
    preds = (probabilities >= threshold).float()
    if min_area > 0:
        preds = remove_small_components_batch(preds, min_area)
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
        "min_area": int(min_area),
        "dice": float(dice.mean().item()),
        "iou": float(iou.mean().item()),
        "precision": float(precision.mean().item()),
        "recall": float(recall.mean().item()),
    }


def remove_small_components_batch(preds: torch.Tensor, min_area: int) -> torch.Tensor:
    processed = [
        torch.from_numpy(remove_small_components_single(preds[index, 0].numpy(), min_area)).unsqueeze(0)
        for index in range(preds.shape[0])
    ]
    return torch.stack(processed, dim=0).to(dtype=preds.dtype)


def remove_small_components_single(mask: np.ndarray, min_area: int) -> np.ndarray:
    mask_bool = mask.astype(bool, copy=False)
    height, width = mask_bool.shape
    visited = np.zeros((height, width), dtype=bool)
    output = np.zeros((height, width), dtype=np.float32)
    neighbors = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )

    for start_y in range(height):
        for start_x in range(width):
            if visited[start_y, start_x] or not mask_bool[start_y, start_x]:
                continue
            component: list[tuple[int, int]] = []
            queue: deque[tuple[int, int]] = deque([(start_y, start_x)])
            visited[start_y, start_x] = True
            while queue:
                y, x = queue.popleft()
                component.append((y, x))
                for offset_y, offset_x in neighbors:
                    next_y = y + offset_y
                    next_x = x + offset_x
                    if not (0 <= next_y < height and 0 <= next_x < width):
                        continue
                    if visited[next_y, next_x] or not mask_bool[next_y, next_x]:
                        continue
                    visited[next_y, next_x] = True
                    queue.append((next_y, next_x))
            if len(component) >= min_area:
                ys, xs = zip(*component)
                output[ys, xs] = 1.0
    return output


def build_thresholds(start: float, end: float, step: float) -> list[float]:
    thresholds: list[float] = []
    value = start
    while value <= end + step * 0.5:
        thresholds.append(round(value, 6))
        value += step
    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an averaged probability ensemble for segmentation.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=str, default="runs/ensemble_hard_mim100_label5")
    parser.add_argument("--start", type=float, default=0.30)
    parser.add_argument("--end", type=float, default=0.85)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument("--output-name", type=str, default="ensemble_threshold_sweep.json")
    parser.add_argument("--tta-flips", action="store_true", help="Average original, H-flip, V-flip, and HV-flip predictions.")
    parser.add_argument(
        "--tta-modes",
        nargs="*",
        choices=("h", "v", "hv"),
        default=None,
        help="Additional flip modes to average with the original prediction.",
    )
    parser.add_argument(
        "--min-area-values",
        type=int,
        nargs="*",
        default=[0],
        help="Foreground connected components smaller than these areas are removed during threshold search.",
    )
    args = parser.parse_args()
    tta_modes = ("h", "v", "hv") if args.tta_flips else tuple(args.tta_modes or ())

    config = load_config(args.config)
    project_config = section(config, "project")
    data_config = section(config, "data")
    model_config = section(config, "model")
    project_config["output_dir"] = args.output_dir

    seed_everything(int(project_config.get("seed", 42)))
    output_dir = ensure_dir(resolve_path(project_config["output_dir"], PROJECT_ROOT))
    data_root = resolve_path(data_config.get("root", "../Dataset"), PROJECT_ROOT)
    image_channels = int(data_config.get("image_channels", 1))
    image_size = int(data_config.get("image_size", 128))
    image_transform = build_eval_image_transform(image_size, image_channels)
    mask_transform = build_mask_transform(image_size)

    device = get_device()
    checkpoint_paths = [resolve_path(checkpoint, PROJECT_ROOT) for checkpoint in args.checkpoints]
    loaded_models: list[ResNetUNet] = []
    checkpoint_epochs: list[int] = []
    for checkpoint_path in checkpoint_paths:
        model, checkpoint = build_model(model_config, image_channels, checkpoint_path, device)
        loaded_models.append(model)
        checkpoint_epochs.append(int(checkpoint.get("epoch", -1)))

    val_loader = build_loader(data_root, args.val_split, data_config, image_transform, mask_transform)
    test_loader = build_loader(data_root, args.test_split, data_config, image_transform, mask_transform)
    val_probs, val_targets = collect_ensemble_probabilities(
        loaded_models,
        val_loader,
        device,
        desc="collect ensemble val",
        tta_modes=tta_modes,
    )
    test_probs, test_targets = collect_ensemble_probabilities(
        loaded_models,
        test_loader,
        device,
        desc="collect ensemble test",
        tta_modes=tta_modes,
    )

    thresholds = build_thresholds(args.start, args.end, args.step)
    min_area_values = sorted(set(args.min_area_values or [0]))
    val_results = [
        metrics_from_probs(val_probs, val_targets, threshold, min_area=min_area)
        for min_area in min_area_values
        for threshold in thresholds
    ]
    best_val = max(val_results, key=lambda item: (item["dice"], item["iou"]))
    test_at_best = metrics_from_probs(
        test_probs,
        test_targets,
        best_val["threshold"],
        min_area=int(best_val["min_area"]),
    )
    test_at_default = metrics_from_probs(test_probs, test_targets, 0.5)

    result = {
        "checkpoints": [str(path) for path in checkpoint_paths],
        "checkpoint_epochs": checkpoint_epochs,
        "ensemble": "mean_probability",
        "tta_modes": list(tta_modes),
        "threshold_range": {
            "start": args.start,
            "end": args.end,
            "step": args.step,
        },
        "min_area_values": min_area_values,
        "selection_metric": "val_dice",
        "best_val": best_val,
        "test_at_best_val_threshold": test_at_best,
        "test_at_default_threshold": test_at_default,
        "val_results": val_results,
    }
    write_json(output_dir / args.output_name, result)
    print(
        "models={models} best_threshold={threshold:.3f} min_area={min_area} val_dice={val_dice:.5f} val_iou={val_iou:.5f} "
        "test_dice={test_dice:.5f} test_iou={test_iou:.5f} precision={precision:.5f} recall={recall:.5f}".format(
            models=len(loaded_models),
            threshold=best_val["threshold"],
            min_area=best_val["min_area"],
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
