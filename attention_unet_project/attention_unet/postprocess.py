from __future__ import annotations

import argparse
import itertools
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class PostProcessConfig:
    enabled: bool = False
    close_iters: int = 0
    open_iters: int = 0
    fill_holes: bool = False
    min_area: int = 0
    min_area_ratio: float = 0.0

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "close_iters": self.close_iters,
            "open_iters": self.open_iters,
            "fill_holes": self.fill_holes,
            "min_area": self.min_area,
            "min_area_ratio": self.min_area_ratio,
        }


def parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("values must be >= 0")
    return values


def parse_float_list(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    if any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("values must be >= 0")
    return values


def parse_bool_list(value: str) -> list[bool]:
    mapping = {
        "1": True,
        "true": True,
        "yes": True,
        "y": True,
        "on": True,
        "0": False,
        "false": False,
        "no": False,
        "n": False,
        "off": False,
    }
    values = []
    for part in value.split(","):
        text = part.strip().lower()
        if not text:
            continue
        if text not in mapping:
            raise argparse.ArgumentTypeError(f"invalid bool value: {part}")
        values.append(mapping[text])
    if not values:
        raise argparse.ArgumentTypeError("expected at least one bool")
    return values


def build_threshold_grid(min_value: float, max_value: float, steps: int) -> list[float]:
    if steps < 2:
        raise argparse.ArgumentTypeError("threshold steps must be at least 2")
    if not 0.0 < min_value < max_value < 1.0:
        raise argparse.ArgumentTypeError("threshold range must satisfy 0 < min < max < 1")
    return [round(float(value), 4) for value in np.linspace(min_value, max_value, steps)]


def postprocess_from_dict(value: Optional[dict]) -> Optional[PostProcessConfig]:
    if value is None:
        return None
    return PostProcessConfig(
        enabled=bool(value.get("enabled", False)),
        close_iters=int(value.get("close_iters", 0)),
        open_iters=int(value.get("open_iters", 0)),
        fill_holes=bool(value.get("fill_holes", False)),
        min_area=int(value.get("min_area", 0)),
        min_area_ratio=float(value.get("min_area_ratio", 0.0)),
    )


def build_postprocess_configs(args: argparse.Namespace) -> list[PostProcessConfig]:
    if args.postprocess_search:
        configs = [
            PostProcessConfig(
                enabled=True,
                close_iters=close_iters,
                open_iters=open_iters,
                fill_holes=fill_holes,
                min_area=min_area,
                min_area_ratio=min_area_ratio,
            )
            for close_iters, open_iters, fill_holes, min_area, min_area_ratio in itertools.product(
                args.post_search_close_iters,
                args.post_search_open_iters,
                args.post_search_fill_holes,
                args.post_search_min_areas,
                args.post_search_min_area_ratios,
            )
        ]
        configs.insert(0, PostProcessConfig(enabled=False))
        return configs

    if args.postprocess:
        return [
            PostProcessConfig(
                enabled=True,
                close_iters=args.post_close_iters,
                open_iters=args.post_open_iters,
                fill_holes=args.post_fill_holes,
                min_area=args.post_min_area,
                min_area_ratio=args.post_min_area_ratio,
            )
        ]

    return [PostProcessConfig(enabled=False)]


def _binary_dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    result = mask.astype(bool)
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = np.zeros_like(result, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                result |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return result


def _binary_erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    result = mask.astype(bool)
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = np.ones_like(result, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                result &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return result


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask.astype(bool)
    result = np.zeros_like(mask, dtype=bool)
    visited = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if visited[y, x] or not mask[y, x]:
                continue
            queue = deque([(y, x)])
            visited[y, x] = True
            component = []
            while queue:
                cy, cx = queue.popleft()
                component.append((cy, cx))
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if not visited[ny, nx] and mask[ny, nx]:
                            visited[ny, nx] = True
                            queue.append((ny, nx))
            if len(component) >= min_area:
                for cy, cx in component:
                    result[cy, cx] = True
    return result


def _fill_binary_holes(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    inverse = ~mask
    visited = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    queue: deque[tuple[int, int]] = deque()

    for x in range(width):
        if inverse[0, x]:
            queue.append((0, x))
            visited[0, x] = True
        if inverse[height - 1, x] and not visited[height - 1, x]:
            queue.append((height - 1, x))
            visited[height - 1, x] = True
    for y in range(height):
        if inverse[y, 0] and not visited[y, 0]:
            queue.append((y, 0))
            visited[y, 0] = True
        if inverse[y, width - 1] and not visited[y, width - 1]:
            queue.append((y, width - 1))
            visited[y, width - 1] = True

    while queue:
        cy, cx = queue.popleft()
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if 0 <= ny < height and 0 <= nx < width and inverse[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                queue.append((ny, nx))

    holes = inverse & ~visited
    return mask | holes


def apply_postprocess_mask(mask: np.ndarray, config: Optional[PostProcessConfig]) -> np.ndarray:
    if config is None or not config.enabled:
        return mask.astype(bool)
    result = mask.astype(bool)
    if config.close_iters > 0:
        result = _binary_erode(_binary_dilate(result, config.close_iters), config.close_iters)
    if config.open_iters > 0:
        result = _binary_dilate(_binary_erode(result, config.open_iters), config.open_iters)
    if config.fill_holes:
        result = _fill_binary_holes(result)
    min_area = max(config.min_area, int(round(config.min_area_ratio * result.size)))
    if min_area > 0:
        result = _remove_small_components(result, min_area)
    return result


def _torch_dilate(mask: torch.Tensor, iterations: int) -> torch.Tensor:
    result = mask.float()
    for _ in range(iterations):
        result = F.max_pool2d(result, kernel_size=3, stride=1, padding=1)
    return result


def _torch_erode(mask: torch.Tensor, iterations: int) -> torch.Tensor:
    result = mask.float()
    for _ in range(iterations):
        result = 1.0 - F.max_pool2d(1.0 - result, kernel_size=3, stride=1, padding=1)
    return result


def _can_postprocess_on_torch(config: PostProcessConfig, image_area: int) -> bool:
    return not config.fill_holes and config.min_area == 0 and int(round(config.min_area_ratio * image_area)) == 0


def apply_postprocess_tensor(preds: torch.Tensor, config: Optional[PostProcessConfig]) -> torch.Tensor:
    if config is None or not config.enabled:
        return preds.float()

    image_area = int(preds.shape[-1] * preds.shape[-2])
    if _can_postprocess_on_torch(config, image_area):
        result = preds.float()
        if config.close_iters > 0:
            result = _torch_erode(_torch_dilate(result, config.close_iters), config.close_iters)
        if config.open_iters > 0:
            result = _torch_dilate(_torch_erode(result, config.open_iters), config.open_iters)
        return (result > 0.5).float()

    processed = []
    for item in preds.detach().cpu().numpy():
        mask = apply_postprocess_mask(item[0] > 0.5, config)
        processed.append(mask.astype(np.float32)[None, ...])
    array = np.stack(processed, axis=0)
    return torch.from_numpy(array).to(device=preds.device, dtype=preds.dtype)
