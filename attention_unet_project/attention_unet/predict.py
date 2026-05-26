from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

from .dataset import IMAGE_EXTENSIONS, parse_size
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


def _collect_images(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path.suffix}")
        return [path]
    iterator = path.rglob("*") if recursive else path.iterdir()
    return sorted(item for item in iterator if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)


def _to_tensor(image: Image.Image, image_size: Optional[tuple[int, int]], device: torch.device) -> torch.Tensor:
    if image_size is not None:
        height, width = image_size
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0).to(device)


def _probability_to_rgb(array: np.ndarray) -> Image.Image:
    array = np.clip(array, 0.0, 1.0)
    rgb = np.zeros((*array.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip((1.0 - np.abs(array - 0.5) * 2.0) * 255.0, 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip((1.0 - array) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _overlay(image: Image.Image, mask: np.ndarray, alpha: float) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    red = np.zeros_like(base)
    red[..., 0] = 255.0
    mask_bool = mask.astype(bool)
    base[mask_bool] = base[mask_bool] * (1.0 - alpha) + red[mask_bool] * alpha
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict masks with an independent Attention U-Net checkpoint.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "runs" / "predict")
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--image-size", type=parse_size, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--tta", choices=["off", "flip"], default="off")
    parser.add_argument("--disable-postprocess", action="store_true")
    parser.add_argument("--recursive", action="store_true")
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
    image_paths = _collect_images(args.input, args.recursive)
    if not image_paths:
        raise RuntimeError(f"No supported images found under: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for image_path in image_paths:
        original = Image.open(image_path).convert("L")
        original_size = original.size
        tensor = _to_tensor(original, image_size, device)
        logits = predict_logits(model, tensor, tta=args.tta)
        probability = torch.sigmoid(logits).squeeze().detach().cpu().numpy().astype(np.float32)
        mask = probability > threshold
        if postprocess is not None:
            mask = apply_postprocess_mask(mask, postprocess)

        prob_scalar_image = Image.fromarray((np.clip(probability, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
        prob_image = _probability_to_rgb(probability)
        mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        if prob_image.size != original_size:
            prob_image = prob_image.resize(original_size, Image.Resampling.BILINEAR)
            prob_scalar_image = prob_scalar_image.resize(original_size, Image.Resampling.BILINEAR)
            mask_image = mask_image.resize(original_size, Image.Resampling.NEAREST)

        final_mask = np.asarray(mask_image, dtype=np.uint8) > 127
        final_probability = np.asarray(prob_scalar_image, dtype=np.float32) / 255.0
        overlay_image = _overlay(original, final_mask, args.overlay_alpha)

        stem = image_path.stem
        mask_path = args.output_dir / f"{stem}_mask.png"
        prob_path = args.output_dir / f"{stem}_probability.png"
        overlay_path = args.output_dir / f"{stem}_overlay.png"
        mask_image.save(mask_path)
        prob_image.save(prob_path)
        overlay_image.save(overlay_path)

        rows.append(
            {
                "image": str(image_path),
                "mask": str(mask_path),
                "probability": str(prob_path),
                "overlay": str(overlay_path),
                "threshold": threshold,
                "mask_pixels": int(final_mask.sum()),
                "mask_ratio": float(final_mask.mean()),
                "mean_probability_in_mask": float(final_probability[final_mask].mean()) if final_mask.any() else 0.0,
            }
        )
        print(f"predicted {image_path.name}: mask_ratio={rows[-1]['mask_ratio']:.4f}")

    csv_path = args.output_dir / "prediction_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = args.output_dir / "prediction_summary.json"
    json_path.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "device": str(device),
                "base_channels": base_channels,
                "image_size": list(image_size) if image_size else None,
                "threshold": threshold,
                "postprocess": postprocess.to_dict() if postprocess else None,
                "items": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {len(rows)} prediction(s) to: {args.output_dir}")


if __name__ == "__main__":
    main()
