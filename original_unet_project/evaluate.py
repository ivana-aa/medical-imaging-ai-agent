"""Evaluation entry point for trained U-Net checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import torch

from train_unet import (
    DEFAULT_DATA_DIR,
    DEFAULT_RUNS_DIR,
    PostProcessConfig,
    UNet,
    apply_postprocess_tensor,
    build_arg_parser,
    build_criterion,
    build_postprocess_configs,
    build_threshold_grid,
    find_best_threshold,
    loader_sample_count,
    make_loader,
    parse_size,
    postprocess_from_dict,
    predict_logits,
    save_predictions,
)


DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "models" / "weights" / "original_unet" / "best_unet.pt"


def _image_size_from_checkpoint(value: Any) -> Optional[tuple[int, int]]:
    if value in (None, "", []):
        return None
    return int(value[0]), int(value[1])


def _load_model(
    checkpoint_path: Path,
    device: torch.device,
    base_channels: Optional[int],
    image_size: Optional[tuple[int, int]],
):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_args = checkpoint.get("args", {})

    resolved_base_channels = int(base_channels or checkpoint_args.get("base_channels", 16))
    resolved_image_size = image_size or _image_size_from_checkpoint(checkpoint_args.get("image_size"))

    model = UNet(in_channels=1, out_channels=1, base_channels=resolved_base_channels).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, checkpoint, checkpoint_args, resolved_base_channels, resolved_image_size


def _loss_args_from_checkpoint(checkpoint_args: dict[str, Any]) -> argparse.Namespace:
    loss_args = build_arg_parser().parse_args([])
    for key in (
        "loss",
        "loss_dice_weight",
        "loss_bce_weight",
        "loss_focal_weight",
        "dice_smooth",
        "focal_alpha",
        "focal_gamma",
    ):
        if key in checkpoint_args:
            setattr(loss_args, key, checkpoint_args[key])
    return loss_args


def _postprocess_search_args(args: argparse.Namespace) -> argparse.Namespace:
    search_args = build_arg_parser().parse_args([])
    search_args.postprocess = args.postprocess
    search_args.postprocess_search = args.search_postprocess
    search_args.post_close_iters = args.post_close_iters
    search_args.post_open_iters = args.post_open_iters
    search_args.post_fill_holes = args.post_fill_holes
    search_args.post_min_area = args.post_min_area
    search_args.post_min_area_ratio = args.post_min_area_ratio
    return search_args


def _stats(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float32)
    variance = tensor.var(unbiased=False).item() if tensor.numel() > 0 else 0.0
    return {
        "mean": tensor.mean().item() if tensor.numel() > 0 else 0.0,
        "std": variance**0.5,
        "variance": variance,
        "min": tensor.min().item() if tensor.numel() > 0 else 0.0,
        "max": tensor.max().item() if tensor.numel() > 0 else 0.0,
    }


@torch.no_grad()
def evaluate_with_stats(
    model,
    loader,
    criterion,
    device: torch.device,
    threshold: float,
    postprocess: Optional[PostProcessConfig],
    tta: str,
) -> dict[str, dict[str, float]]:
    model.eval()
    values = {
        "loss": [],
        "dice": [],
        "iou": [],
        "precision": [],
        "recall": [],
    }

    for images, masks, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        batch_loss = float(criterion(logits, masks).item())
        metric_logits = logits if tta == "off" else predict_logits(model, images, tta=tta)

        probs = torch.sigmoid(metric_logits)
        preds = (probs > threshold).float()
        preds = apply_postprocess_tensor(preds, postprocess)
        targets = (masks > 0.5).float()

        dims = (1, 2, 3)
        intersection = (preds * targets).sum(dim=dims)
        pred_sum = preds.sum(dim=dims)
        target_sum = targets.sum(dim=dims)
        union = pred_sum + target_sum - intersection
        denom = pred_sum + target_sum

        dice = (2.0 * intersection + 1.0) / (denom + 1.0)
        iou = (intersection + 1.0) / (union + 1.0)
        precision = (intersection + 1.0) / (pred_sum + 1.0)
        recall = (intersection + 1.0) / (target_sum + 1.0)

        batch_size = images.size(0)
        values["loss"].extend([batch_loss] * batch_size)
        values["dice"].extend(dice.detach().cpu().tolist())
        values["iou"].extend(iou.detach().cpu().tolist())
        values["precision"].extend(precision.detach().cpu().tolist())
        values["recall"].extend(recall.detach().cpu().tolist())

    return {name: _stats(metric_values) for name, metric_values in values.items()}


def _format_mean_std(stats: dict[str, float]) -> str:
    return f"{stats['mean']:.4f} +/- {stats['std']:.4f}"


def print_metric_table(stats: dict[str, dict[str, float]]) -> None:
    rows = [
        ("Loss", stats["loss"]),
        ("Dice", stats["dice"]),
        ("IoU", stats["iou"]),
        ("Precision", stats["precision"]),
        ("Recall", stats["recall"]),
    ]
    headers = ("Metric", "Mean +/- Std", "Variance", "Min", "Max")
    table_rows = [
        (
            name,
            _format_mean_std(item),
            f"{item['variance']:.6f}",
            f"{item['min']:.4f}",
            f"{item['max']:.4f}",
        )
        for name, item in rows
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in table_rows))
        for index in range(len(headers))
    ]

    def line() -> str:
        return "+-" + "-+-".join("-" * width for width in widths) + "-+"

    print(line())
    print("| " + " | ".join(headers[index].ljust(widths[index]) for index in range(len(headers))) + " |")
    print(line())
    for row in table_rows:
        print("| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(row))) + " |")
    print(line())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained U-Net checkpoint.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--image-size", type=parse_size, default=None)
    parser.add_argument("--threshold", type=float, default=None, help="Override checkpoint threshold.")
    parser.add_argument("--search-threshold", action="store_true", help="Search the best threshold on this split.")
    parser.add_argument("--threshold-min", type=float, default=0.20)
    parser.add_argument("--threshold-max", type=float, default=0.60)
    parser.add_argument("--threshold-steps", type=int, default=17)
    parser.add_argument("--tta", choices=["off", "flip"], default="off")
    parser.add_argument("--disable-postprocess", action="store_true")
    parser.add_argument("--postprocess", action="store_true", help="Use fixed morphology arguments below.")
    parser.add_argument("--search-postprocess", action="store_true", help="Search morphology configs too.")
    parser.add_argument("--post-close-iters", type=int, default=0)
    parser.add_argument("--post-open-iters", type=int, default=0)
    parser.add_argument("--post-fill-holes", action="store_true")
    parser.add_argument("--post-min-area", type=int, default=0)
    parser.add_argument("--post-min-area-ratio", type=float, default=0.0)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--pred-dir", type=Path, default=None)
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint, checkpoint_args, base_channels, image_size = _load_model(
        args.checkpoint,
        device,
        args.base_channels,
        args.image_size,
    )

    loader = make_loader(
        args.data_dir,
        args.split,
        image_size,
        args.batch_size,
        args.num_workers,
        augment="off",
        shuffle=False,
    )
    criterion = build_criterion(_loss_args_from_checkpoint(checkpoint_args))

    checkpoint_postprocess = None if args.disable_postprocess else postprocess_from_dict(checkpoint.get("best_postprocess"))
    selected_threshold = float(args.threshold if args.threshold is not None else checkpoint.get("best_threshold", 0.5))
    selected_postprocess = checkpoint_postprocess
    threshold_metrics = None

    if args.search_threshold:
        threshold_grid = build_threshold_grid(args.threshold_min, args.threshold_max, args.threshold_steps)
        postprocess_configs: Optional[list[PostProcessConfig]]
        if args.disable_postprocess:
            postprocess_configs = [PostProcessConfig(enabled=False)]
        elif args.search_postprocess or args.postprocess:
            postprocess_configs = build_postprocess_configs(_postprocess_search_args(args))
        elif checkpoint_postprocess is not None:
            postprocess_configs = [checkpoint_postprocess]
        else:
            postprocess_configs = [PostProcessConfig(enabled=False)]

        threshold_metrics = find_best_threshold(
            model,
            loader,
            device,
            threshold_grid,
            split_name=f"{args.split}-threshold",
            postprocess_configs=postprocess_configs,
            tta=args.tta,
        )
        selected_threshold = threshold_metrics.threshold
        selected_postprocess = postprocess_from_dict(threshold_metrics.postprocess)

    metric_stats = evaluate_with_stats(
        model,
        loader,
        criterion,
        device,
        threshold=selected_threshold,
        postprocess=selected_postprocess,
        tta=args.tta,
    )

    result = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "samples": loader_sample_count(loader),
        "device": str(device),
        "base_channels": base_channels,
        "image_size": list(image_size) if image_size else None,
        "threshold": selected_threshold,
        "postprocess": selected_postprocess.to_dict() if selected_postprocess else None,
        "loss": metric_stats["loss"]["mean"],
        "dice": metric_stats["dice"]["mean"],
        "iou": metric_stats["iou"]["mean"],
        "precision": metric_stats["precision"]["mean"],
        "recall": metric_stats["recall"]["mean"],
        "metrics": metric_stats,
    }
    if threshold_metrics is not None:
        result["threshold_search"] = {
            "dice": threshold_metrics.dice,
            "iou": threshold_metrics.iou,
            "precision": threshold_metrics.precision,
            "recall": threshold_metrics.recall,
        }

    print(f"\nEvaluation summary: split={args.split}, samples={result['samples']}, device={device}")
    print(f"checkpoint={args.checkpoint}")
    print(f"threshold={selected_threshold:.2f}, postprocess={result['postprocess']}")
    print_metric_table(metric_stats)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Evaluation JSON saved to: {args.output_json}")

    if args.save_predictions:
        pred_dir = args.pred_dir or DEFAULT_RUNS_DIR / "evaluate_predictions" / args.split
        save_predictions(model, loader, device, pred_dir, selected_threshold, selected_postprocess, args.tta)
        print(f"Predictions saved to: {pred_dir}")


if __name__ == "__main__":
    main()
