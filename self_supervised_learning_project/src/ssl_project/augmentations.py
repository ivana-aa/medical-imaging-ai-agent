from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from .config import as_tuple


def _normalize_values(channels: int) -> tuple[list[float], list[float]]:
    if channels == 1:
        return [0.5], [0.5]
    if channels == 3:
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    raise ValueError(f"Unsupported channel count: {channels}")


class SimCLRTransform:
    def __init__(self, image_size: int, channels: int, augment_config: dict[str, Any]) -> None:
        self.image_size = image_size
        self.channels = channels
        self.crop_scale = as_tuple(augment_config.get("crop_scale", [0.55, 1.0]), 2, "crop_scale")
        self.horizontal_flip_p = float(augment_config.get("horizontal_flip_p", 0.5))
        self.vertical_flip_p = float(augment_config.get("vertical_flip_p", 0.5))
        self.color_jitter_p = float(augment_config.get("color_jitter_p", 0.8))
        self.brightness = float(augment_config.get("brightness", 0.25))
        self.contrast = float(augment_config.get("contrast", 0.25))
        self.gaussian_blur_p = float(augment_config.get("gaussian_blur_p", 0.35))
        self.mean, self.std = _normalize_values(channels)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = self._random_resized_crop(image)
        if random.random() < self.horizontal_flip_p:
            image = ImageOps.mirror(image)
        if random.random() < self.vertical_flip_p:
            image = ImageOps.flip(image)
        if random.random() < self.color_jitter_p:
            image = self._jitter(image)
        if random.random() < self.gaussian_blur_p:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 2.0)))
        return pil_to_normalized_tensor(image, self.channels, self.mean, self.std)

    def _random_resized_crop(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        area = width * height
        min_scale, max_scale = self.crop_scale
        for _ in range(10):
            target_area = random.uniform(min_scale, max_scale) * area
            aspect_ratio = random.uniform(0.85, 1.15)
            crop_width = int(round((target_area * aspect_ratio) ** 0.5))
            crop_height = int(round((target_area / aspect_ratio) ** 0.5))
            if 0 < crop_width <= width and 0 < crop_height <= height:
                left = random.randint(0, width - crop_width)
                top = random.randint(0, height - crop_height)
                image = image.crop((left, top, left + crop_width, top + crop_height))
                return image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        image = image.crop((left, top, left + side, top + side))
        return image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)

    def _jitter(self, image: Image.Image) -> Image.Image:
        if self.brightness > 0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            image = ImageEnhance.Brightness(image).enhance(max(0.0, factor))
        if self.contrast > 0:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            image = ImageEnhance.Contrast(image).enhance(max(0.0, factor))
        return image


class EvalImageTransform:
    def __init__(self, image_size: int, channels: int) -> None:
        self.image_size = image_size
        self.channels = channels
        self.mean, self.std = _normalize_values(channels)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        return pil_to_normalized_tensor(image, self.channels, self.mean, self.std)


class ReconstructionImageTransform:
    def __init__(self, image_size: int, channels: int, augment_config: dict[str, Any]) -> None:
        self.image_size = image_size
        self.channels = channels
        self.horizontal_flip_p = float(augment_config.get("horizontal_flip_p", 0.5))
        self.vertical_flip_p = float(augment_config.get("vertical_flip_p", 0.5))
        self.mean, self.std = _normalize_values(channels)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        if random.random() < self.horizontal_flip_p:
            image = ImageOps.mirror(image)
        if random.random() < self.vertical_flip_p:
            image = ImageOps.flip(image)
        return pil_to_normalized_tensor(image, self.channels, self.mean, self.std)


class MaskTransform:
    def __init__(self, image_size: int) -> None:
        self.image_size = image_size

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0)


class SegmentationTrainPairTransform:
    def __init__(self, image_size: int, channels: int, augment_config: dict[str, Any]) -> None:
        self.image_size = image_size
        self.channels = channels
        self.rotation_degrees = float(augment_config.get("rotation_degrees", 0.0))
        self.translate_fraction = float(augment_config.get("translate_fraction", 0.0))
        self.scale_range = as_tuple(augment_config.get("scale_range", [1.0, 1.0]), 2, "scale_range")
        self.horizontal_flip_p = float(augment_config.get("horizontal_flip_p", 0.0))
        self.vertical_flip_p = float(augment_config.get("vertical_flip_p", 0.0))
        self.brightness = float(augment_config.get("brightness", 0.0))
        self.contrast = float(augment_config.get("contrast", 0.0))
        self.noise_std = float(augment_config.get("noise_std", 0.0))
        self.mean, self.std = _normalize_values(channels)

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)

        image, mask = self._scale_pair(image, mask)
        if self.horizontal_flip_p > 0 and random.random() < self.horizontal_flip_p:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)
        if self.vertical_flip_p > 0 and random.random() < self.vertical_flip_p:
            image = ImageOps.flip(image)
            mask = ImageOps.flip(mask)

        if self.rotation_degrees > 0 or self.translate_fraction > 0:
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            max_translate = int(round(self.image_size * self.translate_fraction))
            translate = (
                random.randint(-max_translate, max_translate) if max_translate > 0 else 0,
                random.randint(-max_translate, max_translate) if max_translate > 0 else 0,
            )
            image = image.rotate(
                angle,
                resample=Image.Resampling.BILINEAR,
                translate=translate,
                fillcolor=0,
            )
            mask = mask.rotate(
                angle,
                resample=Image.Resampling.NEAREST,
                translate=translate,
                fillcolor=0,
            )

        image = self._jitter(image)
        image_tensor = pil_to_normalized_tensor(image, self.channels, self.mean, self.std)
        if self.noise_std > 0:
            image_tensor = image_tensor + torch.randn_like(image_tensor) * self.noise_std

        mask_array = np.asarray(mask, dtype=np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
        return image_tensor, mask_tensor

    def _scale_pair(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        min_scale, max_scale = self.scale_range
        scale = random.uniform(min_scale, max_scale)
        if abs(scale - 1.0) < 1e-3:
            return image, mask

        scaled_size = max(1, int(round(self.image_size * scale)))
        image = image.resize((scaled_size, scaled_size), Image.Resampling.BILINEAR)
        mask = mask.resize((scaled_size, scaled_size), Image.Resampling.NEAREST)

        if scaled_size >= self.image_size:
            max_offset = scaled_size - self.image_size
            left = random.randint(0, max_offset)
            top = random.randint(0, max_offset)
            crop_box = (left, top, left + self.image_size, top + self.image_size)
            return image.crop(crop_box), mask.crop(crop_box)

        left = random.randint(0, self.image_size - scaled_size)
        top = random.randint(0, self.image_size - scaled_size)
        image_canvas = Image.new(image.mode, (self.image_size, self.image_size), color=0)
        mask_canvas = Image.new("L", (self.image_size, self.image_size), color=0)
        image_canvas.paste(image, (left, top))
        mask_canvas.paste(mask, (left, top))
        return image_canvas, mask_canvas

    def _jitter(self, image: Image.Image) -> Image.Image:
        if self.brightness > 0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            image = ImageEnhance.Brightness(image).enhance(max(0.0, factor))
        if self.contrast > 0:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            image = ImageEnhance.Contrast(image).enhance(max(0.0, factor))
        return image


def pil_to_normalized_tensor(
    image: Image.Image,
    channels: int,
    mean: list[float],
    std: list[float],
) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if channels == 1:
        if array.ndim == 3:
            array = array[..., 0]
        tensor = torch.from_numpy(array).unsqueeze(0)
    else:
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean_tensor = torch.tensor(mean, dtype=tensor.dtype).view(-1, 1, 1)
    std_tensor = torch.tensor(std, dtype=tensor.dtype).view(-1, 1, 1)
    return (tensor - mean_tensor) / std_tensor


def build_simclr_transform(
    image_size: int,
    channels: int,
    augment_config: dict[str, Any],
) -> SimCLRTransform:
    return SimCLRTransform(image_size, channels, augment_config)


def build_eval_image_transform(image_size: int, channels: int) -> EvalImageTransform:
    return EvalImageTransform(image_size, channels)


def build_reconstruction_transform(
    image_size: int,
    channels: int,
    augment_config: dict[str, Any],
) -> ReconstructionImageTransform:
    return ReconstructionImageTransform(image_size, channels, augment_config)


def build_mask_transform(image_size: int) -> MaskTransform:
    return MaskTransform(image_size)


def build_segmentation_train_pair_transform(
    image_size: int,
    channels: int,
    augment_config: dict[str, Any],
) -> SegmentationTrainPairTransform:
    return SegmentationTrainPairTransform(image_size, channels, augment_config)
