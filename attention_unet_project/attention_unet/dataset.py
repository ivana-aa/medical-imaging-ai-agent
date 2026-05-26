from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional, Sized, Tuple, cast

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Subset


IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def parse_size(value: Optional[str]) -> Optional[Tuple[int, int]]:
    if value is None or value == "":
        return None
    if isinstance(value, tuple):
        return value
    text = str(value).lower().replace("x", ",")
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("size must look like 256,256 or 256x256")
    height, width = int(parts[0]), int(parts[1])
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("size values must be positive")
    return height, width


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


def make_loader(
    data_dir: Path,
    split: str,
    image_size: Optional[Tuple[int, int]],
    batch_size: int,
    num_workers: int,
    augment: str = "off",
    shuffle: bool = False,
    max_samples: int = 0,
) -> DataLoader:
    dataset = SegmentationDataset(data_dir, split, image_size=image_size, augment=augment)
    if max_samples > 0:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
