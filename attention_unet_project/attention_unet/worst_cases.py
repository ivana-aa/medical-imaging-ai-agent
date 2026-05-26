from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw

from .dataset import SegmentationDataset, parse_size
from .metrics import predict_logits
from .model import AttentionUNet
from .postprocess import apply_postprocess_mask, postprocess_from_dict


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = REPOSITORY_ROOT / "models" / "weights" / "attention_unet" / "best_attention_unet.pt"


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

    model = AttentionUNet(in_channels=1, out_channels=1, base_channels=resolved_base_channels).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint, resolved_base_channels, resolved_image_size


def _metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred_bool = pred.astype(bool)
    target_bool = target.astype(bool)
    intersection = float(np.logical_and(pred_bool, target_bool).sum())
    pred_sum = float(pred_bool.sum())
    target_sum = float(target_bool.sum())
    union = float(np.logical_or(pred_bool, target_bool).sum())
    return {
        "dice": (2.0 * intersection + 1.0) / (pred_sum + target_sum + 1.0),
        "iou": (intersection + 1.0) / (union + 1.0),
        "precision": (intersection + 1.0) / (pred_sum + 1.0),
        "recall": (intersection + 1.0) / (target_sum + 1.0),
        "pred_pixels": pred_sum,
        "target_pixels": target_sum,
        "fp_pixels": float(np.logical_and(pred_bool, ~target_bool).sum()),
        "fn_pixels": float(np.logical_and(~pred_bool, target_bool).sum()),
    }


def _to_uint8_gray(array: np.ndarray) -> np.ndarray:
    return np.clip(array * 255.0, 0, 255).astype(np.uint8)


def _mask_overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> Image.Image:
    rgb = np.repeat(_to_uint8_gray(image)[..., None], 3, axis=2).astype(np.float32)
    color_array = np.zeros_like(rgb)
    color_array[..., 0] = color[0]
    color_array[..., 1] = color[1]
    color_array[..., 2] = color[2]
    mask_bool = mask.astype(bool)
    rgb[mask_bool] = rgb[mask_bool] * (1.0 - alpha) + color_array[mask_bool] * alpha
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def _error_overlay(image: np.ndarray, pred: np.ndarray, target: np.ndarray, alpha: float) -> Image.Image:
    rgb = np.repeat(_to_uint8_gray(image)[..., None], 3, axis=2).astype(np.float32)
    pred_bool = pred.astype(bool)
    target_bool = target.astype(bool)
    colors = [
        (np.logical_and(pred_bool, target_bool), (50, 220, 90)),
        (np.logical_and(pred_bool, ~target_bool), (255, 70, 70)),
        (np.logical_and(~pred_bool, target_bool), (60, 140, 255)),
    ]
    for mask, color in colors:
        color_array = np.zeros_like(rgb)
        color_array[..., 0] = color[0]
        color_array[..., 1] = color[1]
        color_array[..., 2] = color[2]
        rgb[mask] = rgb[mask] * (1.0 - alpha) + color_array[mask] * alpha
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def _labelled_panel(panel: Image.Image, title: str, line: str) -> Image.Image:
    header_height = 34
    labelled = Image.new("RGB", (panel.width, panel.height + header_height), (20, 20, 20))
    labelled.paste(panel.convert("RGB"), (0, header_height))
    draw = ImageDraw.Draw(labelled)
    draw.text((8, 4), title, fill=(245, 245, 245))
    draw.text((8, 18), line, fill=(190, 190, 190))
    return labelled


def _make_case_panel(
    image: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
    probability: np.ndarray,
    sample_name: str,
    metrics: dict[str, float],
    alpha: float,
) -> Image.Image:
    gray = Image.fromarray(_to_uint8_gray(image), mode="L").convert("RGB")
    prob = Image.fromarray(_to_uint8_gray(probability), mode="L").convert("RGB")
    gt = _mask_overlay(image, target, (50, 220, 90), alpha)
    pred_overlay = _mask_overlay(image, pred, (255, 70, 70), alpha)
    errors = _error_overlay(image, pred, target, alpha)

    metric_line = (
        f"Dice {metrics['dice']:.4f} | IoU {metrics['iou']:.4f} | "
        f"P {metrics['precision']:.4f} | R {metrics['recall']:.4f}"
    )
    panels = [
        _labelled_panel(gray, "Image", sample_name),
        _labelled_panel(gt, "Ground truth", "green"),
        _labelled_panel(pred_overlay, "Prediction", "red"),
        _labelled_panel(prob, "Probability", f"threshold selected separately"),
        _labelled_panel(errors, "Errors", "TP green | FP red | FN blue"),
    ]

    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height + 24), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), metric_line, fill=(20, 20, 20))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 24))
        x += panel.width
    return canvas


def _case_filename(rank: int, sample_name: str, dice: float) -> str:
    safe_name = Path(sample_name).stem.replace(" ", "_")
    return f"{rank:02d}_dice_{dice:.4f}_{safe_name}.png"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate worst-case overlays for Attention U-Net.")
    parser.add_argument("--data-dir", type=Path, default=REPOSITORY_ROOT / "Dataset")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--image-size", type=parse_size, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--tta", choices=["off", "flip"], default="off")
    parser.add_argument("--disable-postprocess", action="store_true")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    return parser


@torch.no_grad()
def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint, base_channels, image_size = _load_model(
        args.checkpoint,
        device,
        args.base_channels,
        args.image_size,
    )
    threshold = float(args.threshold if args.threshold is not None else checkpoint.get("best_threshold", 0.5))
    postprocess = None if args.disable_postprocess else postprocess_from_dict(checkpoint.get("best_postprocess"))

    dataset = SegmentationDataset(args.data_dir, args.split, image_size=image_size, augment="off")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    cases = []
    for index in range(len(dataset)):
        image_tensor, mask_tensor, sample_name = dataset[index]
        image_batch = image_tensor.unsqueeze(0).to(device)
        logits = predict_logits(model, image_batch, tta=args.tta)
        probability = torch.sigmoid(logits).squeeze().detach().cpu().numpy().astype(np.float32)
        pred = probability > threshold
        if postprocess is not None:
            pred = apply_postprocess_mask(pred, postprocess)

        image = image_tensor.squeeze().numpy().astype(np.float32)
        target = mask_tensor.squeeze().numpy() > 0.5
        item_metrics = _metrics(pred, target)
        row = {
            "rank": 0,
            "sample": sample_name,
            "dice": item_metrics["dice"],
            "iou": item_metrics["iou"],
            "precision": item_metrics["precision"],
            "recall": item_metrics["recall"],
            "pred_pixels": int(item_metrics["pred_pixels"]),
            "target_pixels": int(item_metrics["target_pixels"]),
            "fp_pixels": int(item_metrics["fp_pixels"]),
            "fn_pixels": int(item_metrics["fn_pixels"]),
            "overlay": "",
        }
        rows.append(row)
        cases.append((row, image, target, pred, probability))

    cases.sort(key=lambda item: item[0]["dice"])
    rows_sorted = [item[0] for item in cases]

    for rank, (row, image, target, pred, probability) in enumerate(cases[: args.top_k], start=1):
        row["rank"] = rank
        panel = _make_case_panel(
            image,
            target,
            pred,
            probability,
            row["sample"],
            row,
            args.overlay_alpha,
        )
        overlay_path = args.output_dir / _case_filename(rank, row["sample"], row["dice"])
        panel.save(overlay_path)
        row["overlay"] = str(overlay_path)

    all_cases_csv = args.output_dir / "all_cases_sorted.csv"
    with all_cases_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
        writer.writeheader()
        writer.writerows(rows_sorted)

    worst_csv = args.output_dir / "worst_cases.csv"
    with worst_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
        writer.writeheader()
        writer.writerows(rows_sorted[: args.top_k])

    summary = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "samples": len(dataset),
        "device": str(device),
        "base_channels": base_channels,
        "image_size": list(image_size) if image_size else None,
        "threshold": threshold,
        "tta": args.tta,
        "postprocess": postprocess.to_dict() if postprocess else None,
        "top_k": args.top_k,
        "all_cases_csv": str(all_cases_csv),
        "worst_cases_csv": str(worst_csv),
        "worst_cases": rows_sorted[: args.top_k],
    }
    summary_path = args.output_dir / "worst_cases_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Scored {len(dataset)} {args.split} sample(s).")
    print(f"Worst Dice: {rows_sorted[0]['dice']:.4f} ({rows_sorted[0]['sample']})")
    print(f"Saved worst-case overlays to: {args.output_dir}")


if __name__ == "__main__":
    main()
