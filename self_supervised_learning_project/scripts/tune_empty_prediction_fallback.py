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
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_segmentation_ensemble import (  # noqa: E402
    build_loader,
    build_model,
    predict_probability,
    remove_small_components_batch,
)
from ssl_project.augmentations import build_eval_image_transform, build_mask_transform  # noqa: E402
from ssl_project.config import load_config, resolve_path, section  # noqa: E402
from ssl_project.utils import ensure_dir, get_device, seed_everything, write_json  # noqa: E402


def collect_ensemble_probabilities(
    models: list[torch.nn.Module],
    loader: DataLoader,
    device: torch.device,
    desc: str,
    tta_modes: tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    probabilities: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    paths: list[str] = []

    progress = tqdm(loader, desc=desc, leave=False, disable=not sys.stderr.isatty())
    with torch.no_grad():
        for images, masks, batch_paths in progress:
            images = images.to(device, non_blocking=True)
            batch_probability = None
            for model in models:
                probability = predict_probability(model, images, tta_modes=tta_modes)
                batch_probability = probability if batch_probability is None else batch_probability + probability
            if batch_probability is None:
                raise RuntimeError("No models were provided.")
            probabilities.append((batch_probability / float(len(models))).cpu())
            targets.append((masks >= 0.5).float().cpu())
            paths.extend(str(path) for path in batch_paths)
    return torch.cat(probabilities, dim=0), torch.cat(targets, dim=0), paths


def metrics_from_preds(
    preds: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-7,
) -> dict[str, float]:
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


def apply_empty_prediction_fallback(
    probabilities: torch.Tensor,
    base_threshold: float,
    fallback_threshold: float,
    min_area: int,
    fallback_base_area_max: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    base_preds = (probabilities >= base_threshold).float()
    dims = tuple(range(1, base_preds.ndim))
    base_area = torch.sum(base_preds, dim=dims)
    max_prob = torch.amax(probabilities, dim=dims)
    fallback_cases = (base_area <= fallback_base_area_max) & (max_prob >= fallback_threshold)

    preds = base_preds.clone()
    fallback_preds = (probabilities >= fallback_threshold).float()
    if bool(fallback_cases.any()):
        preds[fallback_cases] = fallback_preds[fallback_cases]
    if min_area > 0:
        preds = remove_small_components_batch(preds, min_area)
    return preds, fallback_cases


def summarize_rule(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    base_threshold: float,
    fallback_threshold: float,
    min_area: int,
    fallback_base_area_max: float = 0.0,
) -> dict[str, float | int]:
    preds, fallback_cases = apply_empty_prediction_fallback(
        probabilities,
        base_threshold=base_threshold,
        fallback_threshold=fallback_threshold,
        min_area=min_area,
        fallback_base_area_max=fallback_base_area_max,
    )
    metrics = metrics_from_preds(preds, targets)
    dims = tuple(range(1, targets.ndim))
    target_area = torch.sum(targets, dim=dims)
    pred_area = torch.sum(preds, dim=dims)
    positive_targets = target_area > 0
    empty_targets = target_area <= 0
    rescued_positive = fallback_cases & positive_targets & (pred_area > 0)
    fallback_false_positive = fallback_cases & empty_targets & (pred_area > 0)
    result: dict[str, float | int] = {
        "base_threshold": float(base_threshold),
        "fallback_threshold": float(fallback_threshold),
        "min_area": int(min_area),
        "fallback_base_area_max": float(fallback_base_area_max),
        **metrics,
        "fallback_cases": int(fallback_cases.sum().item()),
        "rescued_positive_cases": int(rescued_positive.sum().item()),
        "fallback_false_positive_cases": int(fallback_false_positive.sum().item()),
    }
    return result


def build_thresholds(start: float, end: float, step: float) -> list[float]:
    values: list[float] = []
    value = start
    while value <= end + step * 0.5:
        values.append(round(value, 6))
        value += step
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune an inference fallback for empty ensemble predictions.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=str, default="runs/empty_prediction_fallback")
    parser.add_argument("--base-threshold", type=float, default=0.458)
    parser.add_argument("--fallback-start", type=float, default=0.10)
    parser.add_argument("--fallback-end", type=float, default=0.45)
    parser.add_argument("--fallback-step", type=float, default=0.01)
    parser.add_argument("--min-area-values", type=int, nargs="*", default=[0, 25, 50, 100, 200, 500])
    parser.add_argument(
        "--fallback-base-area-max-values",
        type=float,
        nargs="*",
        default=[0.0],
        help="Apply the fallback when the base-threshold prediction area is at or below these values.",
    )
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument("--output-name", type=str, default="empty_prediction_fallback_sweep.json")
    parser.add_argument(
        "--tta-modes",
        nargs="*",
        choices=("h", "v", "hv"),
        default=None,
        help="Additional flip modes to average with the original prediction.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    project_config = section(config, "project")
    data_config = section(config, "data")
    model_config = section(config, "model")

    seed_everything(int(project_config.get("seed", 42)))
    output_dir = ensure_dir(resolve_path(args.output_dir, PROJECT_ROOT))
    data_root = resolve_path(data_config.get("root", "../Dataset"), PROJECT_ROOT)
    image_channels = int(data_config.get("image_channels", 1))
    image_size = int(data_config.get("image_size", 128))
    image_transform = build_eval_image_transform(image_size, image_channels)
    mask_transform = build_mask_transform(image_size)
    tta_modes = tuple(args.tta_modes or ())

    device = get_device()
    checkpoint_paths = [resolve_path(checkpoint, PROJECT_ROOT) for checkpoint in args.checkpoints]
    models: list[torch.nn.Module] = []
    checkpoint_epochs: list[int] = []
    for checkpoint_path in checkpoint_paths:
        model, checkpoint = build_model(model_config, image_channels, checkpoint_path, device)
        models.append(model)
        checkpoint_epochs.append(int(checkpoint.get("epoch", -1)))

    val_loader = build_loader(data_root, args.val_split, data_config, image_transform, mask_transform)
    test_loader = build_loader(data_root, args.test_split, data_config, image_transform, mask_transform)
    val_probs, val_targets, _ = collect_ensemble_probabilities(models, val_loader, device, "collect val", tta_modes)
    test_probs, test_targets, _ = collect_ensemble_probabilities(models, test_loader, device, "collect test", tta_modes)

    fallback_thresholds = build_thresholds(args.fallback_start, args.fallback_end, args.fallback_step)
    min_area_values = sorted(set(args.min_area_values or [0]))
    val_default = summarize_rule(
        val_probs,
        val_targets,
        base_threshold=args.base_threshold,
        fallback_threshold=args.base_threshold,
        min_area=0,
        fallback_base_area_max=0.0,
    )
    test_default = summarize_rule(
        test_probs,
        test_targets,
        base_threshold=args.base_threshold,
        fallback_threshold=args.base_threshold,
        min_area=0,
        fallback_base_area_max=0.0,
    )
    val_results = [
        summarize_rule(
            val_probs,
            val_targets,
            base_threshold=args.base_threshold,
            fallback_threshold=fallback_threshold,
            min_area=min_area,
            fallback_base_area_max=fallback_base_area_max,
        )
        for fallback_base_area_max in sorted(set(args.fallback_base_area_max_values or [0.0]))
        for min_area in min_area_values
        for fallback_threshold in fallback_thresholds
    ]
    best_val = max(val_results, key=lambda item: (float(item["dice"]), float(item["iou"])))
    test_at_best = summarize_rule(
        test_probs,
        test_targets,
        base_threshold=args.base_threshold,
        fallback_threshold=float(best_val["fallback_threshold"]),
        min_area=int(best_val["min_area"]),
        fallback_base_area_max=float(best_val["fallback_base_area_max"]),
    )

    result = {
        "checkpoints": [str(path) for path in checkpoint_paths],
        "checkpoint_epochs": checkpoint_epochs,
        "base_threshold": args.base_threshold,
        "fallback_range": {
            "start": args.fallback_start,
            "end": args.fallback_end,
            "step": args.fallback_step,
        },
        "min_area_values": min_area_values,
        "fallback_base_area_max_values": sorted(set(args.fallback_base_area_max_values or [0.0])),
        "selection_metric": "val_dice",
        "val_default": val_default,
        "test_default": test_default,
        "best_val": best_val,
        "test_at_best_val_rule": test_at_best,
        "val_results": val_results,
    }
    write_json(output_dir / args.output_name, result)
    print(
        "best_fallback={fallback:.3f} min_area={min_area} val_dice={val_dice:.5f} "
        "test_dice={test_dice:.5f} test_iou={test_iou:.5f} precision={precision:.5f} recall={recall:.5f} "
        "rescued_val={rescued_val} rescued_test={rescued_test}".format(
            fallback=best_val["fallback_threshold"],
            min_area=best_val["min_area"],
            val_dice=best_val["dice"],
            test_dice=test_at_best["dice"],
            test_iou=test_at_best["iou"],
            precision=test_at_best["precision"],
            recall=test_at_best["recall"],
            rescued_val=best_val["rescued_positive_cases"],
            rescued_test=test_at_best["rescued_positive_cases"],
        )
    )


if __name__ == "__main__":
    main()
