"""
Local U-Net segmentation agent.

This module loads the U-Net checkpoint trained in D:\\hz and exposes a small
medical-image segmentation analysis layer for the existing FastAPI app.
"""

import base64
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: "torch.Tensor", skip: "torch.Tensor") -> "torch.Tensor":
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)
        self.up1 = Up(base_channels * 16, base_channels * 8, base_channels * 8)
        self.up2 = Up(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up4 = Up(base_channels * 2, base_channels, base_channels)
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


def _image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _default_model_path() -> Path:
    env_path = os.getenv("UNET_AGENT_MODEL")
    if env_path:
        return Path(env_path)

    candidates = [
        Path(r"D:\hz\runs\unet_agent_seed_stability\seed_011\best_unet.pt"),
        Path(r"D:\hz\runs\unet_agent_combo_focal_5ep_wide_thr\best_unet.pt"),
        Path(r"D:\hz\runs\unet_agent_finetune_lr1e4\best_unet.pt"),
        Path(r"D:\hz\runs\unet_agent_es\best_unet.pt"),
        Path(r"D:\hz\runs\unet_agent\best_unet.pt"),
        Path(r"D:\hz\runs\unet\best_unet.pt"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _binary_dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    result = mask.astype(bool, copy=True)
    for _ in range(max(iterations, 0)):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        neighbors = [
            padded[dy : dy + result.shape[0], dx : dx + result.shape[1]]
            for dy in range(3)
            for dx in range(3)
        ]
        result = np.logical_or.reduce(neighbors)
    return result


def _binary_erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    result = mask.astype(bool, copy=True)
    for _ in range(max(iterations, 0)):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        neighbors = [
            padded[dy : dy + result.shape[0], dx : dx + result.shape[1]]
            for dy in range(3)
            for dx in range(3)
        ]
        result = np.logical_and.reduce(neighbors)
    return result


def _fill_binary_holes(mask: np.ndarray) -> np.ndarray:
    if mask.all():
        return mask

    background = ~mask.astype(bool)
    height, width = background.shape
    visited = np.zeros_like(background, dtype=bool)
    stack = []

    for x in range(width):
        if background[0, x]:
            stack.append((0, x))
            visited[0, x] = True
        if background[height - 1, x] and not visited[height - 1, x]:
            stack.append((height - 1, x))
            visited[height - 1, x] = True
    for y in range(height):
        if background[y, 0] and not visited[y, 0]:
            stack.append((y, 0))
            visited[y, 0] = True
        if background[y, width - 1] and not visited[y, width - 1]:
            stack.append((y, width - 1))
            visited[y, width - 1] = True

    while stack:
        y, x = stack.pop()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < height and 0 <= nx < width and background[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                stack.append((ny, nx))

    return mask | (background & ~visited)


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1 or not mask.any():
        return mask

    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    keep = np.zeros_like(mask, dtype=bool)
    ys, xs = np.where(mask)

    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if visited[start_y, start_x]:
            continue
        stack = [(start_y, start_x)]
        visited[start_y, start_x] = True
        pixels = []
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for ny in (y - 1, y, y + 1):
                for nx in (x - 1, x, x + 1):
                    if ny == y and nx == x:
                        continue
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        if len(pixels) >= min_area:
            py, px = zip(*pixels)
            keep[np.array(py), np.array(px)] = True
    return keep


def apply_postprocess_mask(mask: np.ndarray, config: Optional[Dict[str, Any]]) -> np.ndarray:
    if not config or not config.get("enabled"):
        return mask.astype(bool, copy=False)

    result = mask.astype(bool, copy=True)
    close_iters = int(config.get("close_iters", 0) or 0)
    open_iters = int(config.get("open_iters", 0) or 0)
    if close_iters > 0:
        result = _binary_erode(_binary_dilate(result, close_iters), close_iters)
    if open_iters > 0:
        result = _binary_dilate(_binary_erode(result, open_iters), open_iters)
    if bool(config.get("fill_holes", False)):
        result = _fill_binary_holes(result)

    min_area = max(
        int(config.get("min_area", 0) or 0),
        int(round(result.size * float(config.get("min_area_ratio", 0.0) or 0.0))),
    )
    if min_area > 1:
        result = _remove_small_components(result, min_area)
    return result


class UNetSegmentationAgent:
    def __init__(
        self,
        model_path: Optional[Path] = None,
        threshold: float = 0.5,
        input_size: Tuple[int, int] = (512, 512),
    ) -> None:
        self.model_path = model_path or _default_model_path()
        self.threshold = threshold
        self.input_size = input_size
        self.device = None
        self.model = None
        self.base_channels = 32
        self.checkpoint_epoch = None
        self.checkpoint_best_dice = None
        self.postprocess = {"enabled": False}
        self.postprocess_source = ""
        self.postprocess_val_dice = None
        self.loaded = False
        self.last_error = ""
        self.load()

    def load(self) -> bool:
        if not TORCH_AVAILABLE:
            self.last_error = "PyTorch is not installed in the backend environment."
            return False
        if not self.model_path.exists():
            self.last_error = f"Checkpoint not found: {self.model_path}"
            return False

        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            args = checkpoint.get("args", {})
            self.base_channels = int(args.get("base_channels", 32))
            self.threshold = float(checkpoint.get("best_threshold", args.get("threshold", self.threshold)))
            self.checkpoint_epoch = checkpoint.get("epoch")
            self.checkpoint_best_dice = checkpoint.get("best_dice")
            self.postprocess = {"enabled": False}
            self.postprocess_source = ""
            self.postprocess_val_dice = None
            checkpoint_postprocess = checkpoint.get("best_postprocess") or {}
            if checkpoint_postprocess.get("enabled"):
                self.postprocess = checkpoint_postprocess
                self.postprocess_source = "checkpoint"
                self.postprocess_val_dice = self.checkpoint_best_dice
            self._load_postprocess_config()
            image_size = args.get("image_size")
            if image_size:
                self.input_size = (int(image_size[0]), int(image_size[1]))
            self.model = UNet(in_channels=1, out_channels=1, base_channels=self.base_channels).to(self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.eval()
            self.loaded = True
            self.last_error = ""
            return True
        except Exception as exc:
            self.loaded = False
            self.last_error = str(exc)
            return False

    def _load_postprocess_config(self) -> None:
        config_path = os.getenv("UNET_AGENT_POSTPROCESS")
        path = Path(config_path) if config_path else self.model_path.parent / "postprocess_config.json"
        if not path.exists():
            return

        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            postprocess = data.get("postprocess") or {}
            if postprocess.get("enabled"):
                self.postprocess = postprocess
                self.postprocess_source = str(path)
                self.postprocess_val_dice = data.get("val_dice")
                if data.get("best_threshold") is not None:
                    self.threshold = float(data["best_threshold"])
        except Exception as exc:
            self.postprocess = {"enabled": False}
            self.postprocess_source = str(path)
            self.last_error = f"Postprocess config failed: {exc}"

    def status(self) -> Dict[str, Any]:
        return {
            "available": TORCH_AVAILABLE,
            "loaded": self.loaded,
            "model_path": str(self.model_path),
            "base_channels": self.base_channels,
            "threshold": self.threshold,
            "postprocess": self.postprocess,
            "postprocess_source": self.postprocess_source,
            "postprocess_val_dice": self.postprocess_val_dice,
            "checkpoint_epoch": self.checkpoint_epoch,
            "checkpoint_best_dice": self.checkpoint_best_dice,
            "device": str(self.device) if self.device is not None else "unavailable",
            "last_error": self.last_error,
        }

    def analyze_file(self, file_path: Path, threshold: Optional[float] = None) -> Dict[str, Any]:
        if not self.loaded:
            self.load()
        if not self.loaded or self.model is None or self.device is None:
            return {
                "success": False,
                "error": self.last_error or "U-Net model is not ready.",
                "status": self.status(),
            }

        threshold_value = self.threshold if threshold is None else threshold
        original = Image.open(file_path).convert("L")
        original_size = original.size
        model_input = original.resize((self.input_size[1], self.input_size[0]), Image.Resampling.BILINEAR)
        image_array = np.asarray(model_input, dtype=np.float32) / 255.0

        tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probability = torch.sigmoid(logits)[0, 0].cpu().numpy()

        probability_image = Image.fromarray((probability * 255).astype(np.uint8), mode="L")
        probability_original = probability_image.resize(original_size, Image.Resampling.BILINEAR)
        prob_array = np.asarray(probability_original, dtype=np.float32) / 255.0
        mask = prob_array > threshold_value
        mask = apply_postprocess_mask(mask, self.postprocess)

        mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        overlay = self._make_overlay(original, mask, prob_array)
        metrics = self._mask_metrics(mask, prob_array)
        metrics["postprocess"] = self.postprocess
        report = self._make_report(metrics, threshold_value)

        return {
            "success": True,
            "model_path": str(self.model_path),
            "threshold": threshold_value,
            "metrics": metrics,
            "mask": _image_to_base64(mask_image),
            "overlay": _image_to_base64(overlay),
            "probability": _image_to_base64(probability_original.convert("L")),
            "report": report,
            "status": self.status(),
        }

    def _mask_metrics(self, mask: np.ndarray, probability: np.ndarray) -> Dict[str, Any]:
        total_pixels = int(mask.size)
        positive_pixels = int(mask.sum())
        area_ratio = positive_pixels / max(total_pixels, 1)

        bbox = None
        if positive_pixels > 0:
            ys, xs = np.where(mask)
            bbox = {
                "x_min": int(xs.min()),
                "y_min": int(ys.min()),
                "x_max": int(xs.max()),
                "y_max": int(ys.max()),
                "width": int(xs.max() - xs.min() + 1),
                "height": int(ys.max() - ys.min() + 1),
            }

        mean_probability = float(probability[mask].mean()) if positive_pixels > 0 else 0.0
        max_probability = float(probability.max()) if probability.size else 0.0
        risk_level = "low"
        if area_ratio >= 0.15 or mean_probability >= 0.8:
            risk_level = "high"
        elif area_ratio >= 0.03 or mean_probability >= 0.65:
            risk_level = "medium"

        return {
            "positive_pixels": positive_pixels,
            "total_pixels": total_pixels,
            "area_ratio": area_ratio,
            "area_percent": round(area_ratio * 100, 2),
            "mean_probability": round(mean_probability, 4),
            "max_probability": round(max_probability, 4),
            "bbox": bbox,
            "risk_level": risk_level,
            "has_candidate_region": positive_pixels > 0,
        }

    def _make_overlay(self, original: Image.Image, mask: np.ndarray, probability: np.ndarray) -> Image.Image:
        base = original.convert("RGB")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        mask_uint8 = mask.astype(np.uint8)
        ys, xs = np.where(mask_uint8)
        for x, y in zip(xs.tolist(), ys.tolist()):
            alpha = int(80 + 130 * min(float(probability[y, x]), 1.0))
            draw.point((x, y), fill=(255, 64, 64, alpha))
        return Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")

    def _make_report(self, metrics: Dict[str, Any], threshold: float) -> str:
        post = metrics.get("postprocess") or {}
        post_line = ""
        if post.get("enabled"):
            post_line = (
                f"- 后处理: close={int(post.get('close_iters', 0))}, "
                f"open={int(post.get('open_iters', 0))}, "
                f"fill_holes={bool(post.get('fill_holes', False))}, "
                f"min_area={int(post.get('min_area', 0))}\n"
            )
        if metrics["has_candidate_region"]:
            bbox = metrics["bbox"] or {}
            return (
                "本地 U-Net 分割初筛已完成。\n"
                f"- 候选区域面积占比: {metrics['area_percent']}%\n"
                f"- 平均预测概率: {metrics['mean_probability']:.4f}\n"
                f"- 最大预测概率: {metrics['max_probability']:.4f}\n"
                f"- 候选框: x={bbox.get('x_min', '-')}-{bbox.get('x_max', '-')}, "
                f"y={bbox.get('y_min', '-')}-{bbox.get('y_max', '-')}\n"
                f"- 阈值: {threshold:.2f}\n"
                f"{post_line}"
                "该结果来自本地分割模型，仅用于辅助定位可疑区域，不能替代医生诊断。"
            )
        return (
            "本地 U-Net 分割初筛已完成，当前阈值下未检出明确候选区域。\n"
            f"- 最大预测概率: {metrics['max_probability']:.4f}\n"
            f"- 阈值: {threshold:.2f}\n"
            f"{post_line}"
            "该结果仅用于辅助筛查，仍需结合原始影像和专业判断。"
        )


def create_unet_agent(model_path: Optional[str] = None) -> UNetSegmentationAgent:
    return UNetSegmentationAgent(Path(model_path) if model_path else None)
