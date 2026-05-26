"""
Unified local segmentation model registry.

The FastAPI app uses this module to expose the current self-supervised
ensemble, Attention U-Net, and the original U-Net through one response shape.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

try:
    from .unet_segmenter import UNet, UNetSegmentationAgent, apply_postprocess_mask
except ImportError:  # pragma: no cover - supports direct script-style imports
    from unet_segmenter import UNet, UNetSegmentationAgent, apply_postprocess_mask


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(os.getenv("SEGMENTATION_WORKSPACE_ROOT", str(REPOSITORY_ROOT)))
WEIGHTS_ROOT = WORKSPACE_ROOT / "models" / "weights"
ORIGINAL_UNET_CHECKPOINT = WEIGHTS_ROOT / "original_unet" / "best_unet.pt"
ATTENTION_UNET_CHECKPOINT = WEIGHTS_ROOT / "attention_unet" / "best_attention_unet.pt"
ATTENTION_UNET_CONFIG = WEIGHTS_ROOT / "attention_unet" / "best_inference_config.json"
SSL_PROJECT_ROOT = WORKSPACE_ROOT / "self_supervised_learning_project"
SSL_PROJECT_SRC = SSL_PROJECT_ROOT / "src"
SSL_CHECKPOINTS = [
    WEIGHTS_ROOT / "ssl_current" / "seed7_best_decoder.pt",
    WEIGHTS_ROOT / "ssl_current" / "seed42_best_decoder.pt",
    WEIGHTS_ROOT / "ssl_current" / "seed123_best_decoder.pt",
]


def _image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _as_path_list(value: str) -> List[Path]:
    return [Path(item.strip()) for item in value.split(";") if item.strip()]


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _probability_to_rgb(probability: np.ndarray) -> Image.Image:
    array = np.clip(probability, 0.0, 1.0)
    rgb = np.zeros((*array.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip((1.0 - np.abs(array - 0.5) * 2.0) * 255.0, 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip((1.0 - array) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _make_overlay(original: Image.Image, mask: np.ndarray, probability: np.ndarray, color: Tuple[int, int, int]) -> Image.Image:
    base = np.asarray(original.convert("RGB"), dtype=np.float32)
    mask_bool = mask.astype(bool, copy=False)
    if mask_bool.any():
        tint = np.zeros_like(base)
        tint[..., 0] = color[0]
        tint[..., 1] = color[1]
        tint[..., 2] = color[2]
        alpha = np.zeros(mask_bool.shape, dtype=np.float32)
        alpha[mask_bool] = 0.35 + 0.35 * np.clip(probability[mask_bool], 0.0, 1.0)
        base = base * (1.0 - alpha[..., None]) + tint * alpha[..., None]
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1 or not mask.any():
        return mask.astype(bool, copy=False)

    mask_bool = mask.astype(bool, copy=False)
    height, width = mask_bool.shape
    visited = np.zeros_like(mask_bool, dtype=bool)
    output = np.zeros_like(mask_bool, dtype=bool)
    neighbors = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )

    for start_y in range(height):
        for start_x in range(width):
            if visited[start_y, start_x] or not mask_bool[start_y, start_x]:
                continue
            stack = [(start_y, start_x)]
            pixels = []
            visited[start_y, start_x] = True
            while stack:
                y, x = stack.pop()
                pixels.append((y, x))
                for dy, dx in neighbors:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and mask_bool[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if len(pixels) >= min_area:
                ys, xs = zip(*pixels)
                output[np.asarray(ys), np.asarray(xs)] = True
    return output


def _mask_metrics(mask: np.ndarray, probability: np.ndarray) -> Dict[str, Any]:
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


def _report(model_name: str, metrics: Dict[str, Any], threshold: float, extra: str = "") -> str:
    if metrics.get("has_candidate_region"):
        bbox = metrics.get("bbox") or {}
        return (
            f"{model_name} 分割已完成。\n"
            f"- 候选区域面积占比：{metrics.get('area_percent', 0):.2f}%\n"
            f"- 阳性像素：{metrics.get('positive_pixels', 0)} / {metrics.get('total_pixels', 0)}\n"
            f"- 掩膜内平均概率：{metrics.get('mean_probability', 0):.4f}\n"
            f"- 最大预测概率：{metrics.get('max_probability', 0):.4f}\n"
            f"- 候选框：x={bbox.get('x_min', '-')}-{bbox.get('x_max', '-')}, "
            f"y={bbox.get('y_min', '-')}-{bbox.get('y_max', '-')}\n"
            f"- 阈值：{threshold:.4f}\n"
            f"{extra}"
            "该区域是模型生成的候选分割结果，仅用于辅助医生复核，不代表诊断结论。"
        )
    return (
        f"{model_name} 分割已完成。在阈值 {threshold:.4f} 下未发现明确候选区域。\n"
        f"- 最大预测概率：{metrics.get('max_probability', 0):.4f}\n"
        f"{extra}"
        "该结果仅用于辅助筛查，仍需结合原始影像进行复核。"
    )


class SegmentationModelAdapter:
    model_id = ""
    model_name = ""
    architecture = ""
    training_type = ""
    default_threshold = 0.5
    color = (255, 64, 64)
    test_metrics: Dict[str, float] = {}

    def __init__(self) -> None:
        self.loaded = False
        self.last_error = ""
        self.device: Optional[torch.device] = None
        self.model: Any = None

    def load(self) -> bool:
        raise NotImplementedError

    def analyze_file(self, file_path: Path, threshold: Optional[float] = None) -> Dict[str, Any]:
        raise NotImplementedError

    def _base_status(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "architecture": self.architecture,
            "training_type": self.training_type,
            "available": self.available(),
            "loaded": self.loaded,
            "threshold": self.default_threshold,
            "device": str(self.device) if self.device is not None else "unloaded",
            "test_metrics": self.test_metrics,
            "last_error": self.last_error,
        }

    def available(self) -> bool:
        return True

    def status(self) -> Dict[str, Any]:
        return self._base_status()

    def _finalize(
        self,
        original: Image.Image,
        probability: np.ndarray,
        mask: np.ndarray,
        threshold: float,
        started: float,
        extra_report: str = "",
        extra_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        probability = np.clip(probability.astype(np.float32, copy=False), 0.0, 1.0)
        mask = mask.astype(bool, copy=False)
        metrics = _mask_metrics(mask, probability)
        if extra_metrics:
            metrics.update(extra_metrics)
        mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        probability_image = _probability_to_rgb(probability)
        overlay = _make_overlay(original, mask, probability, self.color)
        inference_time_ms = round((time.time() - started) * 1000.0, 1)
        return {
            "success": True,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "architecture": self.architecture,
            "training_type": self.training_type,
            "threshold": threshold,
            "metrics": metrics,
            "mask": _image_to_base64(mask_image),
            "overlay": _image_to_base64(overlay),
            "probability": _image_to_base64(probability_image),
            "report": _report(self.model_name, metrics, threshold, extra_report),
            "inference_time_ms": inference_time_ms,
            "status": self.status(),
        }


class OriginalUNetAdapter(SegmentationModelAdapter):
    model_id = "original_unet"
    model_name = "初始 U-Net"
    architecture = "U-Net"
    training_type = "监督学习基线模型"
    default_threshold = 0.36
    color = (255, 80, 80)
    test_metrics = {"dice": 0.94581, "iou": 0.89933, "precision": 0.94949, "recall": 0.94365}

    def __init__(self) -> None:
        super().__init__()
        env_path = os.getenv("ORIGINAL_UNET_MODEL")
        self.model_path = Path(env_path) if env_path else ORIGINAL_UNET_CHECKPOINT
        self.agent: Optional[UNetSegmentationAgent] = None

    def available(self) -> bool:
        return self.model_path.exists()

    def load(self) -> bool:
        if self.loaded and self.agent is not None:
            return True
        self.agent = UNetSegmentationAgent(self.model_path, threshold=self.default_threshold)
        self.loaded = bool(self.agent.loaded)
        self.last_error = self.agent.last_error
        self.device = self.agent.device
        if self.loaded:
            self.default_threshold = float(self.agent.threshold)
        return self.loaded

    def status(self) -> Dict[str, Any]:
        status = self._base_status()
        status["checkpoint"] = str(self.model_path)
        if self.agent is not None:
            status["agent_status"] = self.agent.status()
        return status

    def analyze_file(self, file_path: Path, threshold: Optional[float] = None) -> Dict[str, Any]:
        started = time.time()
        if not self.load() or self.agent is None:
            return {"success": False, "model_id": self.model_id, "model_name": self.model_name, "error": self.last_error, "status": self.status()}
        result = self.agent.analyze_file(file_path, threshold=threshold)
        result.update(
            {
                "model_id": self.model_id,
                "model_name": self.model_name,
                "architecture": self.architecture,
                "training_type": self.training_type,
                "inference_time_ms": round((time.time() - started) * 1000.0, 1),
            }
        )
        return result


class AttentionUNetAdapter(SegmentationModelAdapter):
    model_id = "attention_unet"
    model_name = "注意力 U-Net"
    architecture = "Attention U-Net"
    training_type = "监督学习注意力模型"
    default_threshold = 0.23
    color = (255, 176, 64)
    test_metrics = {"dice": 0.95580, "iou": 0.91630, "precision": 0.94370, "recall": 0.96860}

    def __init__(self) -> None:
        super().__init__()
        env_path = os.getenv("ATTENTION_UNET_MODEL")
        self.model_path = Path(env_path) if env_path else ATTENTION_UNET_CHECKPOINT
        self.tta = "flip"
        self.base_channels = 16
        self.input_size: Optional[Tuple[int, int]] = (256, 256)
        self.postprocess: Optional[Dict[str, Any]] = None
        self._load_config()

    def _load_config(self) -> None:
        config = _read_json(ATTENTION_UNET_CONFIG)
        self.default_threshold = float(config.get("threshold", self.default_threshold))
        self.tta = str(config.get("tta", self.tta))
        self.postprocess = config.get("postprocess") if config.get("postprocess", {}).get("enabled") else None

    def available(self) -> bool:
        return self.model_path.exists()

    def load(self) -> bool:
        if self.loaded and self.model is not None:
            return True
        try:
            attention_root = WORKSPACE_ROOT / "attention_unet_project"
            if str(attention_root) not in sys.path:
                sys.path.insert(0, str(attention_root))
            from attention_unet.model import AttentionUNet

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            args = checkpoint.get("args", {})
            self.base_channels = int(args.get("base_channels", self.base_channels))
            image_size = args.get("image_size")
            if image_size:
                self.input_size = (int(image_size[0]), int(image_size[1]))
            self.model = AttentionUNet(in_channels=1, out_channels=1, base_channels=self.base_channels).to(self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.eval()
            self.loaded = True
            self.last_error = ""
            return True
        except Exception as exc:
            self.loaded = False
            self.last_error = str(exc)
            return False

    def status(self) -> Dict[str, Any]:
        status = self._base_status()
        status.update(
            {
                "checkpoint": str(self.model_path),
                "tta": self.tta,
                "base_channels": self.base_channels,
                "input_size": list(self.input_size) if self.input_size else None,
                "postprocess": self.postprocess or {"enabled": False},
            }
        )
        return status

    def _predict_logits(self, tensor: torch.Tensor) -> torch.Tensor:
        assert self.model is not None
        logits = self.model(tensor)
        if self.tta != "flip":
            return logits
        horizontal = torch.flip(self.model(torch.flip(tensor, dims=[3])), dims=[3])
        vertical = torch.flip(self.model(torch.flip(tensor, dims=[2])), dims=[2])
        return (logits + horizontal + vertical) / 3.0

    def analyze_file(self, file_path: Path, threshold: Optional[float] = None) -> Dict[str, Any]:
        started = time.time()
        if not self.load() or self.model is None or self.device is None:
            return {"success": False, "model_id": self.model_id, "model_name": self.model_name, "error": self.last_error, "status": self.status()}
        threshold_value = self.default_threshold if threshold is None else float(threshold)
        original = Image.open(file_path).convert("L")
        original_size = original.size
        model_input = original
        if self.input_size:
            height, width = self.input_size
            model_input = original.resize((width, height), Image.Resampling.BILINEAR)
        image_array = np.asarray(model_input, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probability = torch.sigmoid(self._predict_logits(tensor))[0, 0].cpu().numpy()
        prob_image = Image.fromarray((np.clip(probability, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
        if prob_image.size != original_size:
            prob_image = prob_image.resize(original_size, Image.Resampling.BILINEAR)
        prob_array = np.asarray(prob_image, dtype=np.float32) / 255.0
        mask = prob_array >= threshold_value
        if self.postprocess:
            mask = apply_postprocess_mask(mask, self.postprocess)
        return self._finalize(
            original,
            prob_array,
            mask,
            threshold_value,
            started,
            extra_report=f"- 测试时增强：{self.tta}\n",
            extra_metrics={"postprocess": self.postprocess or {"enabled": False}, "tta": self.tta},
        )


class SelfSupervisedEnsembleAdapter(SegmentationModelAdapter):
    model_id = "ssl_current"
    model_name = "当前自监督集成模型"
    architecture = "ResNet-UNet ensemble"
    training_type = "自监督 MIM 预训练 + 10% 标签 hard-mining 微调"
    default_threshold = 0.335
    color = (64, 220, 160)
    test_metrics = {"dice": 0.92074, "iou": 0.86022, "precision": 0.89812, "recall": 0.95396}

    def __init__(self) -> None:
        super().__init__()
        env_paths = os.getenv("SSL_SEGMENTATION_CHECKPOINTS")
        self.checkpoints = _as_path_list(env_paths) if env_paths else SSL_CHECKPOINTS
        self.fallback_threshold = float(os.getenv("SSL_EMPTY_FALLBACK_THRESHOLD", "0.0004"))
        self.min_area = int(os.getenv("SSL_MIN_AREA", "50"))
        self.image_size = 256
        self.models: List[Any] = []

    def available(self) -> bool:
        return all(path.exists() for path in self.checkpoints)

    def load(self) -> bool:
        if self.loaded and self.models:
            return True
        try:
            if str(SSL_PROJECT_SRC) not in sys.path:
                sys.path.insert(0, str(SSL_PROJECT_SRC))
            from ssl_project.models import ResNetEncoder, ResNetUNet

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.models = []
            for checkpoint_path in self.checkpoints:
                encoder = ResNetEncoder(backbone="resnet18", in_channels=1)
                model = ResNetUNet(encoder=encoder, freeze_encoder=False).to(self.device)
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                model.load_state_dict(checkpoint["model"])
                model.eval()
                self.models.append(model)
            self.loaded = True
            self.last_error = ""
            return True
        except Exception as exc:
            self.loaded = False
            self.last_error = str(exc)
            return False

    def status(self) -> Dict[str, Any]:
        status = self._base_status()
        status.update(
            {
                "checkpoints": [str(path) for path in self.checkpoints],
                "ensemble": "mean_probability",
                "image_size": self.image_size,
                "empty_fallback_threshold": self.fallback_threshold,
                "min_area": self.min_area,
                "parameter_count": {
                    "single_model": 14370401,
                    "ensemble_total": 43111203,
                    "encoder_single": 11170240,
                    "decoder_single": 3200161,
                },
            }
        )
        return status

    def analyze_file(self, file_path: Path, threshold: Optional[float] = None) -> Dict[str, Any]:
        started = time.time()
        if not self.load() or not self.models or self.device is None:
            return {"success": False, "model_id": self.model_id, "model_name": self.model_name, "error": self.last_error, "status": self.status()}
        threshold_value = self.default_threshold if threshold is None else float(threshold)
        original = Image.open(file_path).convert("L")
        original_size = original.size
        model_input = original.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        image_array = np.asarray(model_input, dtype=np.float32) / 255.0
        image_array = (image_array - 0.5) / 0.5
        tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probability_sum = None
            for model in self.models:
                probability = torch.sigmoid(model(tensor))
                probability_sum = probability if probability_sum is None else probability_sum + probability
            probability_tensor = probability_sum / float(len(self.models))
            probability = probability_tensor[0, 0].cpu().numpy()

        prob_image = Image.fromarray((np.clip(probability, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
        if prob_image.size != original_size:
            prob_image = prob_image.resize(original_size, Image.Resampling.BILINEAR)
        prob_array = np.asarray(prob_image, dtype=np.float32) / 255.0
        base_mask = prob_array >= threshold_value
        fallback_applied = False
        mask = base_mask
        if int(base_mask.sum()) <= 0 and float(prob_array.max()) >= self.fallback_threshold:
            mask = prob_array >= self.fallback_threshold
            fallback_applied = True
        if self.min_area > 0:
            mask = _remove_small_components(mask, self.min_area)
        extra = ""
        if fallback_applied:
            extra = f"- 已启用空掩膜回退阈值：{self.fallback_threshold:.4f}\n"
        return self._finalize(
            original,
            prob_array,
            mask,
            threshold_value,
            started,
            extra_report=extra,
            extra_metrics={
                "ensemble_size": len(self.models),
                "empty_fallback_threshold": self.fallback_threshold,
                "fallback_applied": fallback_applied,
                "min_area": self.min_area,
            },
        )


class SegmentationModelRegistry:
    def __init__(self) -> None:
        self.models: Dict[str, SegmentationModelAdapter] = {
            "ssl_current": SelfSupervisedEnsembleAdapter(),
            "attention_unet": AttentionUNetAdapter(),
            "original_unet": OriginalUNetAdapter(),
        }

    def list_models(self) -> List[Dict[str, Any]]:
        return [model.status() for model in self.models.values()]

    def get(self, model_id: str) -> SegmentationModelAdapter:
        if model_id not in self.models:
            raise KeyError(f"Unknown segmentation model: {model_id}")
        return self.models[model_id]

    def analyze(self, model_id: str, file_path: Path, threshold: Optional[float] = None) -> Dict[str, Any]:
        return self.get(model_id).analyze_file(file_path, threshold=threshold)


def create_segmentation_registry() -> SegmentationModelRegistry:
    return SegmentationModelRegistry()
