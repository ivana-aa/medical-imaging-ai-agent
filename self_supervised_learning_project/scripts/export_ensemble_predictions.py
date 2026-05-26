from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
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


def per_case_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
    min_area: int = 0,
    empty_fallback_threshold: float | None = None,
    fallback_base_area_max: float = 0.0,
    eps: float = 1e-7,
) -> tuple[list[dict[str, float]], torch.Tensor, torch.Tensor]:
    preds = (probabilities >= threshold).float()
    fallback_cases = torch.zeros(preds.shape[0], dtype=torch.bool)
    if empty_fallback_threshold is not None:
        dims = tuple(range(1, preds.ndim))
        base_area = torch.sum(preds, dim=dims)
        max_prob = torch.amax(probabilities, dim=dims)
        fallback_cases = (base_area <= fallback_base_area_max) & (max_prob >= empty_fallback_threshold)
        fallback_preds = (probabilities >= empty_fallback_threshold).float()
        if bool(fallback_cases.any()):
            preds[fallback_cases] = fallback_preds[fallback_cases]
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
    target_area = torch.sum(targets, dim=dims)
    pred_area = torch.sum(preds, dim=dims)

    rows: list[dict[str, float]] = []
    for index in range(preds.shape[0]):
        rows.append(
            {
                "dice": float(dice[index].item()),
                "iou": float(iou[index].item()),
                "precision": float(precision[index].item()),
                "recall": float(recall[index].item()),
                "target_area": float(target_area[index].item()),
                "pred_area": float(pred_area[index].item()),
                "prob_mean": float(probabilities[index].mean().item()),
                "prob_max": float(probabilities[index].max().item()),
                "fallback_case": float(fallback_cases[index].item()),
            }
        )
    return rows, preds, fallback_cases


def collect_predictions(
    models: list[torch.nn.Module],
    loader: DataLoader,
    device: torch.device,
    tta_modes: tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    image_batches: list[torch.Tensor] = []
    probability_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    paths: list[str] = []

    progress = tqdm(loader, desc="export ensemble predictions", leave=False, disable=not sys.stderr.isatty())
    with torch.no_grad():
        for images, masks, batch_paths in progress:
            images = images.to(device, non_blocking=True)
            batch_probability = None
            for model in models:
                probability = predict_probability(model, images, tta_modes=tta_modes)
                batch_probability = probability if batch_probability is None else batch_probability + probability
            if batch_probability is None:
                raise RuntimeError("No models were provided for ensemble export.")

            probability_batches.append((batch_probability / float(len(models))).cpu())
            target_batches.append((masks >= 0.5).float().cpu())
            image_batches.append(images.cpu())
            paths.extend(str(path) for path in batch_paths)

    return (
        torch.cat(image_batches, dim=0),
        torch.cat(probability_batches, dim=0),
        torch.cat(target_batches, dim=0),
        paths,
    )


def image_tensor_to_rgb(tensor: torch.Tensor) -> Image.Image:
    data = (tensor.detach().cpu() * 0.5 + 0.5).clamp(0.0, 1.0)
    if data.shape[0] == 1:
        array = (data.squeeze(0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(array, mode="L").convert("RGB")
    array = (data.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def mask_to_rgb(mask: np.ndarray) -> Image.Image:
    array = (mask.astype(np.float32) * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="L").convert("RGB")


def probability_to_heatmap(probability: np.ndarray) -> Image.Image:
    p = probability.astype(np.float32).clip(0.0, 1.0)
    red = np.clip((p - 0.35) / 0.65, 0.0, 1.0)
    green = np.clip(1.0 - np.abs(p - 0.5) * 2.0, 0.0, 1.0) * 0.9
    blue = np.clip((0.65 - p) / 0.65, 0.0, 1.0)
    rgb = np.stack([red, green, blue], axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")


def overlay_errors(image: Image.Image, target: np.ndarray, pred: np.ndarray) -> Image.Image:
    base = np.asarray(image.convert("RGB")).astype(np.float32)
    target_bool = target.astype(bool)
    pred_bool = pred.astype(bool)
    colors = np.zeros_like(base)
    alpha = np.zeros(target_bool.shape, dtype=np.float32)

    true_positive = target_bool & pred_bool
    false_positive = ~target_bool & pred_bool
    false_negative = target_bool & ~pred_bool

    colors[true_positive] = np.array([34, 197, 94], dtype=np.float32)
    colors[false_positive] = np.array([239, 68, 68], dtype=np.float32)
    colors[false_negative] = np.array([59, 130, 246], dtype=np.float32)
    alpha[true_positive | false_positive | false_negative] = 0.7
    alpha_3d = alpha[..., None]
    blended = base * (1.0 - alpha_3d) + colors * alpha_3d
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8), mode="RGB")


def add_header(image: Image.Image, title: str) -> Image.Image:
    header_height = 28
    width, height = image.size
    output = Image.new("RGB", (width, height + header_height), color=(18, 24, 38))
    output.paste(image, (0, header_height))
    draw = ImageDraw.Draw(output)
    draw.text((8, 7), title, fill=(245, 247, 250))
    return output


def save_panel(
    output_path: Path,
    image: torch.Tensor,
    target: torch.Tensor,
    probability: torch.Tensor,
    pred: torch.Tensor,
    title: str,
) -> None:
    image_rgb = image_tensor_to_rgb(image)
    target_array = target.squeeze(0).numpy()
    probability_array = probability.squeeze(0).numpy()
    pred_array = pred.squeeze(0).numpy()

    tiles = [
        add_header(image_rgb, "image"),
        add_header(mask_to_rgb(target_array), "ground truth"),
        add_header(probability_to_heatmap(probability_array), "probability"),
        add_header(mask_to_rgb(pred_array), "prediction"),
        add_header(overlay_errors(image_rgb, target_array, pred_array), "TP green / FP red / FN blue"),
    ]
    tile_width, tile_height = tiles[0].size
    title_height = 30
    gap = 4
    panel_width = tile_width * len(tiles) + gap * (len(tiles) - 1)
    panel_height = title_height + tile_height
    panel = Image.new("RGB", (panel_width, panel_height), color=(11, 16, 27))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 8), title, fill=(245, 247, 250))
    x = 0
    for tile in tiles:
        panel.paste(tile, (x, title_height))
        x += tile_width + gap
    panel.save(output_path)


def save_contact_sheet(panel_paths: list[Path], output_path: Path, columns: int = 1) -> None:
    if not panel_paths:
        return
    panels = [Image.open(path).convert("RGB") for path in panel_paths]
    panel_width, panel_height = panels[0].size
    gap = 8
    rows = int(np.ceil(len(panels) / columns))
    width = panel_width * columns + gap * (columns - 1)
    height = panel_height * rows + gap * (rows - 1)
    sheet = Image.new("RGB", (width, height), color=(11, 16, 27))
    for index, panel in enumerate(panels):
        row = index // columns
        column = index % columns
        sheet.paste(panel, (column * (panel_width + gap), row * (panel_height + gap)))
    sheet.save(output_path)


def save_ranked_panels(
    output_dir: Path,
    prefix: str,
    indices: list[int],
    rows: list[dict[str, Any]],
    images: torch.Tensor,
    targets: torch.Tensor,
    probabilities: torch.Tensor,
    preds: torch.Tensor,
) -> list[Path]:
    ensure_dir(output_dir)
    panel_paths: list[Path] = []
    for rank, index in enumerate(indices, start=1):
        row = rows[index]
        stem = Path(str(row["image_path"])).stem
        filename = f"{rank:02d}_{stem}_dice_{row['dice']:.4f}.png"
        panel_path = output_dir / filename
        title = (
            f"{prefix} #{rank} | idx={index} | dice={row['dice']:.4f} | "
            f"iou={row['iou']:.4f} | {Path(str(row['image_path'])).name}"
        )
        save_panel(panel_path, images[index], targets[index], probabilities[index], preds[index], title)
        panel_paths.append(panel_path)
    save_contact_sheet(panel_paths, output_dir / f"{prefix}_contact.png")
    return panel_paths


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "index",
        "split",
        "image_path",
        "panel_path",
        "threshold",
        "min_area",
        "empty_fallback_threshold",
        "fallback_base_area_max",
        "fallback_case",
        "dice",
        "iou",
        "precision",
        "recall",
        "target_area",
        "pred_area",
        "prob_mean",
        "prob_max",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ensemble segmentation predictions and per-case panels.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output-dir", type=str, default="runs/ensemble_prediction_exports")
    parser.add_argument("--first-count", type=int, default=20)
    parser.add_argument("--rank-count", type=int, default=10)
    parser.add_argument("--min-area", type=int, default=0)
    parser.add_argument(
        "--empty-fallback-threshold",
        type=float,
        default=None,
        help="For images with no prediction at --threshold, retry with this lower threshold.",
    )
    parser.add_argument(
        "--fallback-base-area-max",
        type=float,
        default=0.0,
        help="Apply the fallback when the base-threshold prediction area is at or below this value.",
    )
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

    loader = build_loader(data_root, args.split, data_config, image_transform, mask_transform)
    images, probabilities, targets, paths = collect_predictions(models, loader, device, tta_modes=tta_modes)
    metric_rows, preds, fallback_cases = per_case_metrics(
        probabilities,
        targets,
        threshold=args.threshold,
        min_area=args.min_area,
        empty_fallback_threshold=args.empty_fallback_threshold,
        fallback_base_area_max=args.fallback_base_area_max,
    )

    rows: list[dict[str, Any]] = []
    for index, metric_row in enumerate(metric_rows):
        rows.append(
            {
                "index": index,
                "split": args.split,
                "image_path": paths[index],
                "panel_path": "",
                "threshold": args.threshold,
                "min_area": args.min_area,
                "empty_fallback_threshold": args.empty_fallback_threshold,
                "fallback_base_area_max": args.fallback_base_area_max,
                **metric_row,
            }
        )

    first_indices = list(range(min(args.first_count, len(rows))))
    sorted_indices = sorted(range(len(rows)), key=lambda index: float(rows[index]["dice"]))
    worst_indices = sorted_indices[: min(args.rank_count, len(rows))]
    best_indices = list(reversed(sorted_indices[-min(args.rank_count, len(rows)) :]))

    panel_root = ensure_dir(output_dir / "panels")
    first_paths = save_ranked_panels(
        panel_root / "first",
        "first",
        first_indices,
        rows,
        images,
        targets,
        probabilities,
        preds,
    )
    worst_paths = save_ranked_panels(
        panel_root / "worst",
        "worst",
        worst_indices,
        rows,
        images,
        targets,
        probabilities,
        preds,
    )
    best_paths = save_ranked_panels(
        panel_root / "best",
        "best",
        best_indices,
        rows,
        images,
        targets,
        probabilities,
        preds,
    )

    for index, panel_path in zip(first_indices, first_paths):
        rows[index]["panel_path"] = str(panel_path)
    for index, panel_path in zip(worst_indices, worst_paths):
        rows[index]["panel_path"] = str(panel_path)
    for index, panel_path in zip(best_indices, best_paths):
        rows[index]["panel_path"] = str(panel_path)

    summary = {
        "split": args.split,
        "sample_count": len(rows),
        "threshold": args.threshold,
        "min_area": args.min_area,
        "empty_fallback_threshold": args.empty_fallback_threshold,
        "fallback_base_area_max": args.fallback_base_area_max,
        "tta_modes": list(tta_modes),
        "checkpoints": [str(path) for path in checkpoint_paths],
        "checkpoint_epochs": checkpoint_epochs,
        "mean_metrics": {
            "dice": mean_metric(rows, "dice"),
            "iou": mean_metric(rows, "iou"),
            "precision": mean_metric(rows, "precision"),
            "recall": mean_metric(rows, "recall"),
        },
        "fallback_cases": int(fallback_cases.sum().item()),
        "rescued_positive_cases": int(
            sum(
                1
                for row in rows
                if int(row["fallback_case"]) == 1
                and float(row["target_area"]) > 0
                and float(row["pred_area"]) > 0
            )
        ),
        "fallback_false_positive_cases": int(
            sum(
                1
                for row in rows
                if int(row["fallback_case"]) == 1
                and float(row["target_area"]) <= 0
                and float(row["pred_area"]) > 0
            )
        ),
        "best_cases": [rows[index] for index in best_indices],
        "worst_cases": [rows[index] for index in worst_indices],
        "outputs": {
            "per_case_csv": str(output_dir / "per_case_metrics.csv"),
            "per_case_json": str(output_dir / "per_case_metrics.json"),
            "summary_json": str(output_dir / "summary.json"),
            "panels": str(panel_root),
        },
    }

    write_metrics_csv(output_dir / "per_case_metrics.csv", rows)
    write_json(output_dir / "per_case_metrics.json", {"cases": rows})
    write_json(output_dir / "summary.json", summary)

    print(
        "split={split} samples={samples} threshold={threshold:.3f} dice={dice:.5f} "
        "iou={iou:.5f} precision={precision:.5f} recall={recall:.5f} "
        "fallback_cases={fallback_cases} output={output}".format(
            split=args.split,
            samples=len(rows),
            threshold=args.threshold,
            dice=summary["mean_metrics"]["dice"],
            iou=summary["mean_metrics"]["iou"],
            precision=summary["mean_metrics"]["precision"],
            recall=summary["mean_metrics"]["recall"],
            fallback_cases=summary["fallback_cases"],
            output=output_dir,
        )
    )


if __name__ == "__main__":
    main()
