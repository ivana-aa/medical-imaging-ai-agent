import argparse
import csv
import json
import itertools
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sized, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageEnhance
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def parse_size(value: Optional[str]) -> Optional[Tuple[int, int]]:
    if value is None or value == "":
        return None
    text = value.lower().replace("x", ",")
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("size must look like 512,512 or 512x512")
    height, width = int(parts[0]), int(parts[1])
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("size values must be positive")
    return height, width


def build_threshold_grid(min_value: float, max_value: float, steps: int) -> list[float]:
    if steps < 2:
        raise argparse.ArgumentTypeError("threshold steps must be at least 2")
    if not 0.0 < min_value < max_value < 1.0:
        raise argparse.ArgumentTypeError("threshold range must satisfy 0 < min < max < 1")
    return [round(float(value), 4) for value in np.linspace(min_value, max_value, steps)]


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


def loader_sample_count(loader: DataLoader) -> int:
    return len(cast(Sized, loader.dataset))


class SegmentationDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        image_size: Optional[Tuple[int, int]] = None,
        augment: str = "off",
    ) -> None:
        self.root = root
        self.split = split
        self.image_dir = root / split / "images"
        self.label_dir = root / split / "labels"
        self.image_size = image_size
        self.augment = augment

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image folder not found: {self.image_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Label folder not found: {self.label_dir}")

        label_by_stem = {
            path.stem: path
            for path in self.label_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        self.samples = []
        for image_path in sorted(self.image_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_path = label_by_stem.get(image_path.stem)
            if label_path is not None:
                self.samples.append((image_path, label_path))

        if not self.samples:
            raise RuntimeError(f"No image/label pairs found in {self.image_dir} and {self.label_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label_path = self.samples[index]

        image = Image.open(image_path).convert("L")
        mask = Image.open(label_path).convert("L")

        if self.image_size is not None:
            height, width = self.image_size
            image = image.resize((width, height), Image.Resampling.BILINEAR)
            mask = mask.resize((width, height), Image.Resampling.NEAREST)

        if self.augment != "off":
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            if self.augment == "strong":
                if random.random() < 0.5:
                    turns = random.choice([1, 2, 3])
                    image = image.rotate(90 * turns, resample=Image.Resampling.BILINEAR)
                    mask = mask.rotate(90 * turns, resample=Image.Resampling.NEAREST)
                if random.random() < 0.7:
                    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.20))
                if random.random() < 0.5:
                    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.90, 1.12))

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        if self.augment == "strong" and random.random() < 0.35:
            noise = np.random.normal(loc=0.0, scale=0.02, size=image_array.shape).astype(np.float32)
            image_array = np.clip(image_array + noise, 0.0, 1.0)
        mask_array = (np.asarray(mask, dtype=np.float32) > 127.0).astype(np.float32)

        image_tensor = torch.from_numpy(image_array).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
        return image_tensor, mask_tensor, image_path.name


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class DiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersection = (probs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        return bce + (1.0 - dice.mean())


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersection = (probs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, probs, 1.0 - probs)
        alpha_t = torch.where(targets > 0.5, self.alpha, 1.0 - self.alpha)
        loss = alpha_t * torch.pow(1.0 - pt, self.gamma) * bce
        return loss.mean()


class ComboSegmentationLoss(nn.Module):
    def __init__(
        self,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        focal_weight: float = 0.0,
        dice_smooth: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.dice = DiceLoss(smooth=dice_smooth)
        self.bce = nn.BCEWithLogitsLoss()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = logits.new_tensor(0.0)
        if self.dice_weight > 0:
            loss = loss + self.dice_weight * self.dice(logits, targets)
        if self.bce_weight > 0:
            loss = loss + self.bce_weight * self.bce(logits, targets)
        if self.focal_weight > 0:
            loss = loss + self.focal_weight * self.focal(logits, targets)
        return loss


@dataclass
class Metrics:
    loss: float
    dice: float
    iou: float


@dataclass
class ThresholdMetrics:
    threshold: float
    dice: float
    iou: float
    postprocess: Optional[dict] = None


@dataclass(frozen=True)
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
        pixels: list[tuple[int, int]] = []

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


def _fill_binary_holes(mask: np.ndarray) -> np.ndarray:
    if mask.all():
        return mask

    background = ~mask.astype(bool)
    height, width = background.shape
    visited = np.zeros_like(background, dtype=bool)
    stack: list[tuple[int, int]] = []

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

    holes = background & ~visited
    return mask | holes


def apply_postprocess_mask(mask: np.ndarray, config: PostProcessConfig) -> np.ndarray:
    if not config.enabled:
        return mask.astype(bool, copy=False)

    result = mask.astype(bool, copy=True)
    if config.close_iters > 0:
        result = _binary_erode(_binary_dilate(result, config.close_iters), config.close_iters)
    if config.open_iters > 0:
        result = _binary_dilate(_binary_erode(result, config.open_iters), config.open_iters)
    if config.fill_holes:
        result = _fill_binary_holes(result)

    min_area = max(int(config.min_area), int(round(result.size * config.min_area_ratio)))
    if min_area > 1:
        result = _remove_small_components(result, min_area)
    return result


def _torch_dilate(mask: torch.Tensor, iterations: int) -> torch.Tensor:
    result = mask
    for _ in range(max(iterations, 0)):
        result = F.max_pool2d(result, kernel_size=3, stride=1, padding=1)
    return result


def _torch_erode(mask: torch.Tensor, iterations: int) -> torch.Tensor:
    result = mask
    for _ in range(max(iterations, 0)):
        result = 1.0 - F.max_pool2d(1.0 - result, kernel_size=3, stride=1, padding=1)
    return result


def _can_postprocess_on_torch(config: PostProcessConfig, image_area: int) -> bool:
    min_area = max(int(config.min_area), int(round(image_area * config.min_area_ratio)))
    return not config.fill_holes and min_area <= 1


def apply_postprocess_tensor(preds: torch.Tensor, config: Optional[PostProcessConfig]) -> torch.Tensor:
    if config is None or not config.enabled:
        return preds

    image_area = int(preds.shape[-1] * preds.shape[-2])
    if _can_postprocess_on_torch(config, image_area):
        result = preds
        if config.close_iters > 0:
            result = _torch_erode(_torch_dilate(result, config.close_iters), config.close_iters)
        if config.open_iters > 0:
            result = _torch_dilate(_torch_erode(result, config.open_iters), config.open_iters)
        return (result > 0.5).to(dtype=preds.dtype)

    device = preds.device
    pred_np = preds.detach().cpu().numpy().astype(bool)
    processed = np.zeros_like(pred_np, dtype=np.float32)
    for batch_index in range(pred_np.shape[0]):
        processed[batch_index, 0] = apply_postprocess_mask(pred_np[batch_index, 0], config).astype(np.float32)
    return torch.from_numpy(processed).to(device=device, dtype=preds.dtype)


def binary_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    postprocess: Optional[PostProcessConfig] = None,
):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    preds = apply_postprocess_tensor(preds, postprocess)
    targets = (targets > 0.5).float()

    dims = (1, 2, 3)
    intersection = (preds * targets).sum(dim=dims)
    union = preds.sum(dim=dims) + targets.sum(dim=dims) - intersection
    denom = preds.sum(dim=dims) + targets.sum(dim=dims)

    dice = (2.0 * intersection + 1.0) / (denom + 1.0)
    iou = (intersection + 1.0) / (union + 1.0)
    return dice.mean().item(), iou.mean().item()


def predict_logits(model: nn.Module, images: torch.Tensor, tta: str = "off") -> torch.Tensor:
    if tta == "off":
        return model(images)
    if tta != "flip":
        raise ValueError(f"Unsupported TTA mode: {tta}")

    probabilities = []
    transforms = [
        (lambda x: x, lambda x: x),
        (lambda x: torch.flip(x, dims=[-1]), lambda x: torch.flip(x, dims=[-1])),
        (lambda x: torch.flip(x, dims=[-2]), lambda x: torch.flip(x, dims=[-2])),
        (lambda x: torch.flip(x, dims=[-2, -1]), lambda x: torch.flip(x, dims=[-2, -1])),
    ]
    for apply_transform, undo_transform in transforms:
        logits = model(apply_transform(images))
        probabilities.append(torch.sigmoid(undo_transform(logits)))

    mean_probability = torch.stack(probabilities, dim=0).mean(dim=0).clamp(1e-6, 1.0 - 1e-6)
    return torch.logit(mean_probability)


def make_loader(
    dataset_root: Path,
    split: str,
    image_size: Optional[Tuple[int, int]],
    batch_size: int,
    num_workers: int,
    augment: str,
    shuffle: bool,
) -> DataLoader:
    dataset = SegmentationDataset(
        root=dataset_root,
        split=split,
        image_size=image_size,
        augment=augment,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    max_grad_norm: float = 0.0,
) -> Metrics:
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_items = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for images, masks, _ in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        batch_size = images.size(0)
        dice, iou = binary_metrics(logits.detach(), masks)
        total_loss += loss.item() * batch_size
        total_dice += dice * batch_size
        total_iou += iou * batch_size
        total_items += batch_size
        pbar.set_postfix(loss=total_loss / total_items, dice=total_dice / total_items)

    return Metrics(
        loss=total_loss / total_items,
        dice=total_dice / total_items,
        iou=total_iou / total_items,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split_name: str = "val",
    threshold: float = 0.5,
    postprocess: Optional[PostProcessConfig] = None,
    tta: str = "off",
) -> Metrics:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_items = 0

    pbar = tqdm(loader, desc=split_name, leave=False)
    for images, masks, _ in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)
        metric_logits = logits if tta == "off" else predict_logits(model, images, tta=tta)
        batch_size = images.size(0)
        dice, iou = binary_metrics(metric_logits, masks, threshold=threshold, postprocess=postprocess)
        total_loss += loss.item() * batch_size
        total_dice += dice * batch_size
        total_iou += iou * batch_size
        total_items += batch_size
        pbar.set_postfix(loss=total_loss / total_items, dice=total_dice / total_items)

    return Metrics(
        loss=total_loss / total_items,
        dice=total_dice / total_items,
        iou=total_iou / total_items,
    )


@torch.no_grad()
def find_best_threshold(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    thresholds: list[float],
    split_name: str = "val-threshold",
    postprocess_configs: Optional[list[PostProcessConfig]] = None,
    search_output_path: Optional[Path] = None,
    tta: str = "off",
) -> ThresholdMetrics:
    model.eval()
    configs = postprocess_configs or [PostProcessConfig(enabled=False)]
    candidates = [(threshold, config) for threshold in thresholds for config in configs]
    totals = [
        {"threshold": threshold, "postprocess": config, "dice": 0.0, "iou": 0.0, "items": 0}
        for threshold, config in candidates
    ]

    pbar = tqdm(loader, desc=split_name, leave=False)
    for images, masks, _ in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = predict_logits(model, images, tta=tta)
        batch_size = images.size(0)
        for values in totals:
            dice, iou = binary_metrics(
                logits,
                masks,
                threshold=float(values["threshold"]),
                postprocess=cast(PostProcessConfig, values["postprocess"]),
            )
            values["dice"] = float(values["dice"]) + dice * batch_size
            values["iou"] = float(values["iou"]) + iou * batch_size
            values["items"] = int(values["items"]) + batch_size

    rows = []
    best = ThresholdMetrics(threshold=0.5, dice=-1.0, iou=-1.0, postprocess=None)
    for values in totals:
        items = max(int(values["items"]), 1)
        dice = float(values["dice"]) / items
        iou = float(values["iou"]) / items
        threshold = float(values["threshold"])
        config = cast(PostProcessConfig, values["postprocess"])
        row = {"threshold": threshold, "dice": dice, "iou": iou, **config.to_dict()}
        rows.append(row)
        if dice > best.dice:
            best = ThresholdMetrics(
                threshold=threshold,
                dice=dice,
                iou=iou,
                postprocess=config.to_dict(),
            )

    if search_output_path is not None:
        write_search_results(search_output_path, rows)
    return best


def write_search_results(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: row["dice"], reverse=True)
    fieldnames = [
        "threshold",
        "enabled",
        "close_iters",
        "open_iters",
        "fill_holes",
        "min_area",
        "min_area_ratio",
        "dice",
        "iou",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ordered)


@torch.no_grad()
def save_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    threshold: float,
    postprocess: Optional[PostProcessConfig] = None,
    tta: str = "off",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    for images, _, names in tqdm(loader, desc="predict"):
        images = images.to(device, non_blocking=True)
        logits = predict_logits(model, images, tta=tta)
        preds = (torch.sigmoid(logits) > threshold).float()
        preds = apply_postprocess_tensor(preds, postprocess)
        masks = preds.byte().cpu().numpy() * 255
        for mask, name in zip(masks, names):
            mask_image = Image.fromarray(mask[0].astype(np.uint8), mode="L")
            mask_image.save(output_dir / Path(name).with_suffix(".png").name)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_dice: float,
    best_threshold: float,
    args: argparse.Namespace,
    best_postprocess: Optional[dict] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_dice": best_dice,
            "best_threshold": best_threshold,
            "best_postprocess": best_postprocess,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "args": vars(args),
        },
        path,
    )


def append_history(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_threshold_info(path: Path, epoch: int, metrics: ThresholdMetrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "epoch": epoch,
                "best_threshold": metrics.threshold,
                "val_dice": metrics.dice,
                "val_iou": metrics.iou,
                "postprocess": metrics.postprocess,
            },
            f,
            indent=2,
        )


def read_history(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []

    rows: list[dict[str, float]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({key: float(value) for key, value in row.items() if value not in (None, "")})
    return rows


def plot_training_curves(history_path: Path, output_path: Path) -> None:
    history = read_history(history_path)
    if not history:
        return

    epochs = [int(row["epoch"]) for row in history]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 1500, 520
    margin = 46
    panel_gap = 34
    panel_width = (width - margin * 2 - panel_gap * 2) // 3
    panel_height = height - 120
    top = 64
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    plots: list[tuple[str, str, str, str]] = [
        ("Loss", "train_loss", "val_loss", "lower is better"),
        ("Dice", "train_dice", "val_dice", "higher is better"),
        ("IoU", "train_iou", "val_iou", "higher is better"),
    ]

    for index, (title, train_key, val_key, ylabel) in enumerate(plots):
        left = margin + index * (panel_width + panel_gap)
        box = (left, top, left + panel_width, top + panel_height)
        train_values = [row[train_key] for row in history]
        val_values = [row[val_key] for row in history]
        draw_metric_panel(draw, box, title, ylabel, epochs, train_values, val_values)

    draw.text((margin, 18), "U-Net fitting curves", fill=(20, 20, 20))
    draw.text((margin, height - 34), "blue=train, red=val", fill=(70, 70, 70))
    image.save(output_path)


def draw_metric_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    ylabel: str,
    epochs: list[int],
    train_values: list[float],
    val_values: list[float],
) -> None:
    left, top, right, bottom = box
    plot_left = left + 58
    plot_top = top + 42
    plot_right = right - 18
    plot_bottom = bottom - 50
    values = train_values + val_values
    y_min = min(values)
    y_max = max(values)
    if abs(y_max - y_min) < 1e-8:
        y_min -= 0.5
        y_max += 0.5
    padding = (y_max - y_min) * 0.08
    y_min -= padding
    y_max += padding

    draw.rectangle(box, outline=(215, 215, 215), width=1)
    draw.text((left + 12, top + 12), title, fill=(20, 20, 20))
    draw.text((left + 12, bottom - 28), "epoch", fill=(70, 70, 70))
    draw.text((left + 12, top + 34), ylabel, fill=(70, 70, 70))

    for grid_index in range(5):
        y = plot_top + round((plot_bottom - plot_top) * grid_index / 4)
        draw.line((plot_left, y, plot_right, y), fill=(235, 235, 235), width=1)
        value = y_max - (y_max - y_min) * grid_index / 4
        draw.text((left + 8, y - 7), f"{value:.3f}", fill=(85, 85, 85))

    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=(80, 80, 80), width=1)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=(80, 80, 80), width=1)

    def points(series: list[float]) -> list[tuple[int, int]]:
        if len(series) == 1:
            x_positions = [(plot_left + plot_right) // 2]
        else:
            x_positions = [
                plot_left + round((plot_right - plot_left) * i / (len(series) - 1))
                for i in range(len(series))
            ]
        return [
            (
                x,
                plot_bottom - round((plot_bottom - plot_top) * (value - y_min) / (y_max - y_min)),
            )
            for x, value in zip(x_positions, series)
        ]

    train_points = points(train_values)
    val_points = points(val_values)
    if len(train_points) > 1:
        draw.line(train_points, fill=(37, 99, 235), width=3)
        draw.line(val_points, fill=(220, 38, 38), width=3)
    for x, y in train_points:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(37, 99, 235))
    for x, y in val_points:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(220, 38, 38))

    draw.text((plot_left, plot_bottom + 10), str(epochs[0]), fill=(85, 85, 85))
    draw.text((plot_right - 28, plot_bottom + 10), str(epochs[-1]), fill=(85, 85, 85))


def colored_overlay(image: np.ndarray, label: np.ndarray, pred: np.ndarray) -> np.ndarray:
    base = np.stack([image, image, image], axis=-1)
    overlay = base.copy()
    label_mask = label > 0.5
    pred_mask = pred > 0.5

    overlay[label_mask, 1] = 1.0
    overlay[label_mask, 0] *= 0.35
    overlay[label_mask, 2] *= 0.35

    overlay[pred_mask, 0] = 1.0
    overlay[pred_mask, 1] *= 0.45
    overlay[pred_mask, 2] *= 0.45

    intersection = label_mask & pred_mask
    overlay[intersection, 0] = 1.0
    overlay[intersection, 1] = 1.0
    overlay[intersection, 2] = 0.0
    return np.clip(overlay, 0.0, 1.0)


@torch.no_grad()
def save_visual_samples(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_path: Path,
    threshold: float,
    max_samples: int,
    postprocess: Optional[PostProcessConfig] = None,
    tta: str = "off",
) -> None:
    if max_samples <= 0:
        return

    model.eval()
    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]] = []
    for images, masks, names in loader:
        images = images.to(device, non_blocking=True)
        logits = predict_logits(model, images, tta=tta)
        probs = torch.sigmoid(logits).cpu().numpy()
        image_arrays = images.cpu().numpy()
        mask_arrays = masks.cpu().numpy()

        for image, mask, prob, name in zip(image_arrays, mask_arrays, probs, names):
            pred = (prob[0] > threshold).astype(np.float32)
            if postprocess is not None and postprocess.enabled:
                pred = apply_postprocess_mask(pred > 0.5, postprocess).astype(np.float32)
            samples.append((image[0], mask[0], prob[0], pred, str(name)))
            if len(samples) >= max_samples:
                break
        if len(samples) >= max_samples:
            break

    if not samples:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tile_size = 220
    title_height = 36
    gap = 12
    columns = 4
    rows = len(samples)
    canvas_width = columns * tile_size + (columns + 1) * gap
    canvas_height = rows * (tile_size + title_height) + (rows + 1) * gap
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")

    for row_index, (image, label, prob, pred, name) in enumerate(samples):
        tiles = [
            make_visual_tile(gray_to_rgb(image), f"{name} | image", tile_size, title_height),
            make_visual_tile(gray_to_rgb(label), "label", tile_size, title_height),
            make_visual_tile(probability_to_rgb(prob), "prediction probability", tile_size, title_height),
            make_visual_tile(float_rgb_to_image(colored_overlay(image, label, pred)), "overlay: label green, pred red", tile_size, title_height),
        ]
        y = gap + row_index * (tile_size + title_height + gap)
        for col_index, tile in enumerate(tiles):
            x = gap + col_index * (tile_size + gap)
            canvas.paste(tile, (x, y))

    canvas.save(output_path)


def gray_to_rgb(array: np.ndarray) -> Image.Image:
    clipped = np.clip(array, 0.0, 1.0)
    gray = (clipped * 255).astype(np.uint8)
    return Image.fromarray(gray, mode="L").convert("RGB")


def probability_to_rgb(array: np.ndarray) -> Image.Image:
    clipped = np.clip(array, 0.0, 1.0)
    red = (clipped * 255).astype(np.uint8)
    green = (np.sqrt(clipped) * 180).astype(np.uint8)
    blue = ((1.0 - clipped) * 80).astype(np.uint8)
    rgb = np.stack([red, green, blue], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def float_rgb_to_image(array: np.ndarray) -> Image.Image:
    clipped = np.clip(array, 0.0, 1.0)
    return Image.fromarray((clipped * 255).astype(np.uint8), mode="RGB")


def make_visual_tile(image: Image.Image, title: str, tile_size: int, title_height: int) -> Image.Image:
    tile = Image.new("RGB", (tile_size, tile_size + title_height), "white")
    draw = ImageDraw.Draw(tile)
    draw.text((6, 8), title[:42], fill=(25, 25, 25))
    resized = image.resize((tile_size, tile_size), Image.Resampling.BILINEAR)
    tile.paste(resized, (0, title_height))
    return tile


def build_postprocess_configs(args: argparse.Namespace) -> list[PostProcessConfig]:
    disabled = PostProcessConfig(enabled=False)
    if not args.postprocess and not args.postprocess_search:
        return [disabled]

    if not args.postprocess_search:
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

    configs = [disabled]
    for close_iters, open_iters, fill_holes, min_area, min_area_ratio in itertools.product(
        args.post_search_close_iters,
        args.post_search_open_iters,
        args.post_search_fill_holes,
        args.post_search_min_areas,
        args.post_search_min_area_ratios,
    ):
        config = PostProcessConfig(
            enabled=True,
            close_iters=close_iters,
            open_iters=open_iters,
            fill_holes=fill_holes,
            min_area=min_area,
            min_area_ratio=min_area_ratio,
        )
        if config.to_dict() != disabled.to_dict():
            configs.append(config)

    unique: dict[tuple, PostProcessConfig] = {}
    for config in configs:
        unique[
            (
                config.enabled,
                config.close_iters,
                config.open_iters,
                config.fill_holes,
                config.min_area,
                config.min_area_ratio,
            )
        ] = config
    return list(unique.values())


def postprocess_from_dict(value: Optional[dict]) -> Optional[PostProcessConfig]:
    if not value or not value.get("enabled"):
        return None
    return PostProcessConfig(
        enabled=True,
        close_iters=int(value.get("close_iters", 0)),
        open_iters=int(value.get("open_iters", 0)),
        fill_holes=bool(value.get("fill_holes", False)),
        min_area=int(value.get("min_area", 0)),
        min_area_ratio=float(value.get("min_area_ratio", 0.0)),
    )


def build_criterion(args: argparse.Namespace) -> nn.Module:
    if args.loss == "bce":
        return nn.BCEWithLogitsLoss()
    if args.loss == "dice":
        return DiceLoss(smooth=args.dice_smooth)
    if args.loss == "focal":
        return FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    if args.loss == "dice_bce":
        return ComboSegmentationLoss(
            dice_weight=args.loss_dice_weight,
            bce_weight=args.loss_bce_weight,
            focal_weight=0.0,
            dice_smooth=args.dice_smooth,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
        )
    if args.loss == "combo":
        if args.loss_dice_weight <= 0 and args.loss_bce_weight <= 0 and args.loss_focal_weight <= 0:
            raise ValueError("At least one combo loss weight must be greater than 0.")
        return ComboSegmentationLoss(
            dice_weight=args.loss_dice_weight,
            bce_weight=args.loss_bce_weight,
            focal_weight=args.loss_focal_weight,
            dice_smooth=args.dice_smooth,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
        )
    raise ValueError(f"Unsupported loss: {args.loss}")


def build_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer):
    if args.scheduler == "none":
        return None
    if args.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            min_lr=args.min_lr,
        )
    if args.scheduler == "cosine":
        t_max = args.scheduler_t_max if args.scheduler_t_max > 0 else max(args.epochs, 1)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=args.min_lr)
    if args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(args.scheduler_step_size, 1),
            gamma=args.scheduler_gamma,
        )
    raise ValueError(f"Unsupported scheduler: {args.scheduler}")


def step_scheduler(scheduler, scheduler_name: str, metric: float) -> None:
    if scheduler is None:
        return
    if scheduler_name == "plateau":
        scheduler.step(metric)
    else:
        scheduler.step()


def save_run_config(args: argparse.Namespace, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = vars(args).copy()
    for key, value in list(values.items()):
        if isinstance(value, Path):
            values[key] = str(value)
        elif isinstance(value, tuple):
            values[key] = list(value)
    with path.open("w", encoding="utf-8") as f:
        json.dump(values, f, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a binary U-Net for this Dataset folder.")
    parser.add_argument("--data-dir", type=Path, default=Path("Dataset"), help="Dataset root folder.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--image-size", type=parse_size, default=None, help="Optional H,W resize, for example 512,512.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--augment", choices=["off", "basic", "strong"], default="basic", help="Training-time augmentation strength.")
    parser.add_argument("--loss", choices=["dice_bce", "combo", "bce", "dice", "focal"], default="dice_bce", help="Segmentation loss family.")
    parser.add_argument("--loss-dice-weight", type=float, default=1.0, help="Dice term weight for dice_bce/combo loss.")
    parser.add_argument("--loss-bce-weight", type=float, default=1.0, help="BCE term weight for dice_bce/combo loss.")
    parser.add_argument("--loss-focal-weight", type=float, default=0.0, help="Focal term weight for combo loss.")
    parser.add_argument("--dice-smooth", type=float, default=1.0, help="Smoothing constant for Dice loss.")
    parser.add_argument("--focal-alpha", type=float, default=0.25, help="Positive-class alpha for focal loss.")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Gamma for focal loss.")
    parser.add_argument("--scheduler", choices=["plateau", "cosine", "step", "none"], default="plateau", help="Learning-rate scheduler.")
    parser.add_argument("--scheduler-factor", type=float, default=0.5, help="ReduceLROnPlateau decay factor.")
    parser.add_argument("--scheduler-patience", type=int, default=5, help="ReduceLROnPlateau patience.")
    parser.add_argument("--scheduler-step-size", type=int, default=10, help="StepLR step size in epochs.")
    parser.add_argument("--scheduler-gamma", type=float, default=0.5, help="StepLR decay factor.")
    parser.add_argument("--scheduler-t-max", type=int, default=0, help="CosineAnnealingLR T_max. 0 means --epochs.")
    parser.add_argument("--min-lr", type=float, default=0.0, help="Minimum LR for plateau/cosine scheduler.")
    parser.add_argument("--tta", choices=["off", "flip"], default="off", help="Validation/test-time augmentation.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threshold-min", type=float, default=0.10, help="Minimum validation threshold to scan.")
    parser.add_argument("--threshold-max", type=float, default=0.90, help="Maximum validation threshold to scan.")
    parser.add_argument("--threshold-steps", type=int, default=17, help="Number of thresholds to scan on validation.")
    parser.add_argument("--early-stopping-patience", type=int, default=0, help="Stop after N epochs without threshold-optimized val Dice improvement. 0 disables.")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0, help="Minimum val Dice improvement required to reset early stopping.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs") / "unet")
    parser.add_argument("--predict-test", action="store_true", help="Save test predictions after training.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint to load before training/eval.")
    parser.add_argument("--reset-optimizer", action="store_true", help="Load checkpoint weights but start a fresh optimizer at --lr.")
    parser.add_argument("--max-grad-norm", type=float, default=0.0, help="Clip gradient norm when greater than 0.")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate the checkpoint.")
    parser.add_argument("--no-visuals", action="store_true", help="Disable curve plots and validation sample images.")
    parser.add_argument("--visualize-every", type=int, default=1, help="Save visuals every N epochs.")
    parser.add_argument("--visualize-samples", type=int, default=4, help="Number of validation samples to visualize.")
    parser.add_argument("--launch-monitor", action="store_true", help="Launch the popup fitting monitor window.")
    parser.add_argument("--monitor-refresh-ms", type=int, default=2000, help="Popup monitor refresh interval in ms.")
    parser.add_argument("--postprocess", action="store_true", help="Enable fixed morphology post-processing.")
    parser.add_argument("--postprocess-search", action="store_true", help="Search threshold and morphology post-processing on validation.")
    parser.add_argument("--post-close-iters", type=int, default=0, help="Fixed binary closing iterations when --postprocess is enabled.")
    parser.add_argument("--post-open-iters", type=int, default=0, help="Fixed binary opening iterations when --postprocess is enabled.")
    parser.add_argument("--post-fill-holes", action="store_true", help="Fill enclosed holes when --postprocess is enabled.")
    parser.add_argument("--post-min-area", type=int, default=0, help="Remove connected components smaller than this pixel count.")
    parser.add_argument("--post-min-area-ratio", type=float, default=0.0, help="Remove components smaller than this image-area ratio.")
    parser.add_argument("--post-search-close-iters", type=parse_int_list, default=parse_int_list("0,1"), help="Comma list for closing iterations.")
    parser.add_argument("--post-search-open-iters", type=parse_int_list, default=parse_int_list("0"), help="Comma list for opening iterations.")
    parser.add_argument("--post-search-fill-holes", type=parse_bool_list, default=parse_bool_list("false,true"), help="Comma bool list for hole filling.")
    parser.add_argument("--post-search-min-areas", type=parse_int_list, default=parse_int_list("0,16,64"), help="Comma list of min component areas.")
    parser.add_argument("--post-search-min-area-ratios", type=parse_float_list, default=parse_float_list("0"), help="Comma list of min component area ratios.")
    parser.add_argument("--postprocess-search-output", type=Path, default=None, help="CSV path for threshold/postprocess ablation results.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    seed_everything(args.seed)

    if args.launch_monitor:
        monitor_script = Path(__file__).with_name("fit_monitor.py")
        subprocess.Popen(
            [
                sys.executable,
                str(monitor_script),
                "--run-dir",
                str(args.output_dir),
                "--refresh-ms",
                str(args.monitor_refresh_ms),
            ]
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.checkpoint is not None:
        checkpoint_args = torch.load(args.checkpoint, map_location="cpu", weights_only=False).get("args", {})
        args.base_channels = int(checkpoint_args.get("base_channels", args.base_channels))
        if args.image_size is None and checkpoint_args.get("image_size"):
            image_size = checkpoint_args["image_size"]
            args.image_size = (int(image_size[0]), int(image_size[1]))
        print(f"Checkpoint config: base_channels={args.base_channels}, image_size={args.image_size}")

    train_loader = None
    if not args.eval_only:
        train_loader = make_loader(
            args.data_dir,
            "train",
            args.image_size,
            args.batch_size,
            args.num_workers,
            augment=args.augment,
            shuffle=True,
        )
        print(f"Train samples: {loader_sample_count(train_loader)}")

    val_loader = make_loader(
        args.data_dir,
        "val",
        args.image_size,
        args.batch_size,
        args.num_workers,
        augment="off",
        shuffle=False,
    )
    print(f"Val samples: {loader_sample_count(val_loader)}")

    model = UNet(in_channels=1, out_channels=1, base_channels=args.base_channels).to(device)
    criterion = build_criterion(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = build_scheduler(args, optimizer)

    start_epoch = 1
    best_dice = 0.0
    best_threshold = args.threshold
    best_path = args.output_dir / "best_unet.pt"
    last_path = args.output_dir / "last_unet.pt"
    history_path = args.output_dir / "history.csv"
    threshold_info_path = args.output_dir / "best_threshold.json"
    save_run_config(args, args.output_dir / "ablation_config.json")
    threshold_grid = build_threshold_grid(args.threshold_min, args.threshold_max, args.threshold_steps)
    postprocess_configs = build_postprocess_configs(args)
    search_output_path = args.postprocess_search_output
    if search_output_path is None and args.postprocess_search:
        search_output_path = args.output_dir / "postprocess_ablation.csv"
    best_postprocess = None

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint and not args.eval_only and not args.reset_optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if (
            scheduler is not None
            and checkpoint.get("scheduler_state") is not None
            and not args.eval_only
            and not args.reset_optimizer
        ):
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state"])
            except Exception as exc:
                print(f"Scheduler state skipped: {exc}")
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_dice = float(checkpoint.get("best_dice", 0.0))
        best_threshold = float(checkpoint.get("best_threshold", args.threshold))
        best_postprocess = checkpoint.get("best_postprocess")
        print(f"Loaded checkpoint: {args.checkpoint}")

    if args.eval_only:
        threshold_metrics = find_best_threshold(
            model,
            val_loader,
            device,
            threshold_grid,
            postprocess_configs=postprocess_configs,
            search_output_path=search_output_path,
            tta=args.tta,
        )
        selected_postprocess = postprocess_from_dict(threshold_metrics.postprocess)
        metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            split_name="val",
            threshold=threshold_metrics.threshold,
            postprocess=selected_postprocess,
            tta=args.tta,
        )
        print(
            f"val loss={metrics.loss:.4f} dice={metrics.dice:.4f} iou={metrics.iou:.4f} | "
            f"best_threshold={threshold_metrics.threshold:.2f} | "
            f"postprocess={threshold_metrics.postprocess}"
        )
        if search_output_path is not None:
            save_threshold_info(args.output_dir / "best_threshold_postprocess_eval.json", 0, threshold_metrics)
            print(f"Postprocess ablation saved to: {search_output_path}")
    else:
        scaler = None
        if device.type == "cuda" and not args.no_amp:
            scaler = torch.cuda.amp.GradScaler()

        assert train_loader is not None
        epochs_without_improvement = 0
        for epoch in range(start_epoch, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")
            train_metrics = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                scaler,
                max_grad_norm=args.max_grad_norm,
            )
            epoch_search_output = None
            if search_output_path is not None:
                epoch_search_output = search_output_path
            epoch_threshold_metrics = find_best_threshold(
                model,
                val_loader,
                device,
                threshold_grid,
                postprocess_configs=postprocess_configs,
                search_output_path=epoch_search_output,
                tta=args.tta,
            )
            selected_postprocess = postprocess_from_dict(epoch_threshold_metrics.postprocess)
            val_metrics = evaluate(
                model,
                val_loader,
                criterion,
                device,
                split_name="val",
                threshold=epoch_threshold_metrics.threshold,
                postprocess=selected_postprocess,
                tta=args.tta,
            )
            step_scheduler(scheduler, args.scheduler, epoch_threshold_metrics.dice)

            row = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_metrics.loss,
                "train_dice": train_metrics.dice,
                "train_iou": train_metrics.iou,
                "val_loss": val_metrics.loss,
                "val_dice": val_metrics.dice,
                "val_iou": val_metrics.iou,
                "val_best_threshold": epoch_threshold_metrics.threshold,
                "val_best_dice": epoch_threshold_metrics.dice,
                "val_best_iou": epoch_threshold_metrics.iou,
            }
            append_history(history_path, row)
            save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_dice, best_threshold, args, best_postprocess)

            improved = epoch_threshold_metrics.dice > best_dice + args.early_stopping_min_delta
            if improved:
                best_dice = epoch_threshold_metrics.dice
                best_threshold = epoch_threshold_metrics.threshold
                best_postprocess = epoch_threshold_metrics.postprocess
                epochs_without_improvement = 0
                save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_dice, best_threshold, args, best_postprocess)
                save_threshold_info(threshold_info_path, epoch, epoch_threshold_metrics)
            else:
                epochs_without_improvement += 1

            if not args.no_visuals and args.visualize_every > 0 and epoch % args.visualize_every == 0:
                visuals_dir = args.output_dir / "visuals"
                plot_training_curves(history_path, visuals_dir / "training_curves.png")
                save_visual_samples(
                    model,
                    val_loader,
                    device,
                    visuals_dir / f"val_epoch_{epoch:03d}.png",
                    epoch_threshold_metrics.threshold,
                    args.visualize_samples,
                    selected_postprocess,
                    args.tta,
                )

            print(
                f"train loss={train_metrics.loss:.4f} dice={train_metrics.dice:.4f} iou={train_metrics.iou:.4f} | "
                f"val loss={val_metrics.loss:.4f} dice={val_metrics.dice:.4f} iou={val_metrics.iou:.4f} | "
                f"epoch_threshold={epoch_threshold_metrics.threshold:.2f} | "
                f"postprocess={epoch_threshold_metrics.postprocess} | "
                f"best_dice={best_dice:.4f} best_threshold={best_threshold:.2f}"
            )

            if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                print(
                    f"Early stopping at epoch {epoch}: no val Dice improvement for "
                    f"{epochs_without_improvement} epoch(s)."
                )
                break

    if args.predict_test:
        if best_path.exists() and not args.eval_only:
            checkpoint = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state"])
            best_threshold = float(checkpoint.get("best_threshold", best_threshold))
            best_postprocess = checkpoint.get("best_postprocess", best_postprocess)
        test_loader = make_loader(
            args.data_dir,
            "test",
            args.image_size,
            args.batch_size,
            args.num_workers,
            augment="off",
            shuffle=False,
        )
        print(f"Test samples: {loader_sample_count(test_loader)}")
        selected_postprocess = postprocess_from_dict(best_postprocess)
        test_metrics = evaluate(
            model,
            test_loader,
            criterion,
            device,
            split_name="test",
            threshold=best_threshold,
            postprocess=selected_postprocess,
            tta=args.tta,
        )
        print(
            f"test loss={test_metrics.loss:.4f} dice={test_metrics.dice:.4f} iou={test_metrics.iou:.4f} "
            f"threshold={best_threshold:.2f} postprocess={best_postprocess}"
        )
        save_predictions(
            model,
            test_loader,
            device,
            args.output_dir / "test_predictions",
            best_threshold,
            selected_postprocess,
            args.tta,
        )
        print(f"Predictions saved to: {args.output_dir / 'test_predictions'}")


if __name__ == "__main__":
    main()
