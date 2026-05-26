from __future__ import annotations

import argparse
from pathlib import Path

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def list_images(path: Path) -> list[Path]:
    return sorted(
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


def check_split(root: Path, split: str, image_dir_name: str, label_dir_name: str) -> None:
    image_dir = root / split / image_dir_name
    label_dir = root / split / label_dir_name
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    images = list_images(image_dir)
    print(f"{split}: images={len(images)} dir={image_dir}")

    if label_dir.exists():
        labels = list_images(label_dir)
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        missing_labels = sorted(image_stems - label_stems)
        extra_labels = sorted(label_stems - image_stems)
        print(
            f"{split}: labels={len(labels)} paired={len(image_stems & label_stems)} "
            f"missing_labels={len(missing_labels)} extra_labels={len(extra_labels)}"
        )
        if missing_labels:
            print(f"{split}: first missing labels: {missing_labels[:5]}")
        if extra_labels:
            print(f"{split}: first extra labels: {extra_labels[:5]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check dataset split structure.")
    parser.add_argument("--root", type=Path, default=Path("../Dataset"))
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--image-dir-name", default="images")
    parser.add_argument("--label-dir-name", default="labels")
    args = parser.parse_args()

    root = args.root
    if not root.is_absolute():
        root = (Path(__file__).resolve().parents[1] / root).resolve()
    for split in args.splits:
        check_split(root, split, args.image_dir_name, args.label_dir_name)


if __name__ == "__main__":
    main()
