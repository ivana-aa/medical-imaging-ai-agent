from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def list_images(path: Path) -> list[Path]:
    return sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


def split_image_dir(root: Path, split: str, image_dir_name: str = "images") -> Path:
    image_dir = root / split / image_dir_name
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    return image_dir


def open_image(path: Path, channels: int) -> Image.Image:
    image = Image.open(path)
    if channels == 1:
        return image.convert("L")
    if channels == 3:
        return image.convert("RGB")
    raise ValueError(f"Unsupported channel count: {channels}")


class UnlabeledImageDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        image_dir_name: str = "images",
        image_channels: int = 1,
    ) -> None:
        self.image_dir = split_image_dir(root, split, image_dir_name)
        self.image_channels = image_channels
        self.paths = list_images(self.image_dir)
        if not self.paths:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Image.Image, str]:
        path = self.paths[index]
        return open_image(path, self.image_channels), str(path)


class ContrastiveImageDataset(Dataset):
    def __init__(
        self,
        base_dataset: UnlabeledImageDataset,
        transform: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        image, path = self.base_dataset[index]
        view_a = self.transform(image)
        view_b = self.transform(image)
        return view_a, view_b, path


class SegmentationPairDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        image_transform: Callable[[Image.Image], torch.Tensor],
        mask_transform: Callable[[Image.Image], torch.Tensor],
        pair_transform: Callable[[Image.Image, Image.Image], tuple[torch.Tensor, torch.Tensor]] | None = None,
        image_dir_name: str = "images",
        label_dir_name: str = "labels",
        image_channels: int = 1,
    ) -> None:
        self.image_channels = image_channels
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.pair_transform = pair_transform
        self.image_dir = split_image_dir(root, split, image_dir_name)
        self.label_dir = root / split / label_dir_name
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Label directory does not exist: {self.label_dir}")

        label_by_stem = {path.stem: path for path in list_images(self.label_dir)}
        pairs: list[tuple[Path, Path]] = []
        for image_path in list_images(self.image_dir):
            label_path = label_by_stem.get(image_path.stem)
            if label_path is not None:
                pairs.append((image_path, label_path))
        if not pairs:
            raise FileNotFoundError(f"No image/label pairs found for split '{split}'")
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        image_path, label_path = self.pairs[index]
        image = open_image(image_path, self.image_channels)
        mask = Image.open(label_path).convert("L")
        if self.pair_transform is not None:
            image_tensor, mask_tensor = self.pair_transform(image, mask)
        else:
            image_tensor = self.image_transform(image)
            mask_tensor = self.mask_transform(mask)
        mask_tensor = (mask_tensor > 0.5).float()
        return image_tensor, mask_tensor, str(image_path)
