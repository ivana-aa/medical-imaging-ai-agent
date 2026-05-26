from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ssl_project.augmentations import build_eval_image_transform, build_mask_transform
from ssl_project.config import load_config, resolve_path, section
from ssl_project.data import SegmentationPairDataset
from ssl_project.losses import soft_dice_loss
from ssl_project.metrics import SegmentationMetrics, binary_segmentation_metrics
from ssl_project.models import ResNetEncoder, ResNetUNet
from ssl_project.utils import AverageMeter, ensure_dir, get_device, seed_everything, write_json


def build_loader(
    data_root: Path,
    split: str,
    data_config: dict[str, Any],
    image_transform: Any,
    mask_transform: Any,
) -> DataLoader:
    dataset = SegmentationPairDataset(
        root=data_root,
        split=split,
        image_transform=image_transform,
        mask_transform=mask_transform,
        image_dir_name=str(data_config.get("image_dir_name", "images")),
        label_dir_name=str(data_config.get("label_dir_name", "labels")),
        image_channels=int(data_config.get("image_channels", 1)),
    )
    return DataLoader(
        dataset,
        batch_size=int(data_config.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(data_config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )


def evaluate(
    model: ResNetUNet,
    loader: DataLoader,
    device: torch.device,
    dice_weight: float,
    threshold: float,
    desc: str,
) -> SegmentationMetrics:
    model.eval()
    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()

    progress = tqdm(loader, desc=desc, leave=False, disable=not sys.stderr.isatty())
    with torch.no_grad():
        for images, masks, _ in progress:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(images)
            bce = F.binary_cross_entropy_with_logits(logits, masks)
            dice_loss = soft_dice_loss(logits, masks)
            loss = bce + dice_weight * dice_loss
            metrics = binary_segmentation_metrics(logits, masks, threshold=threshold)

            batch_size = images.shape[0]
            loss_meter.update(float(loss.item()), batch_size)
            dice_meter.update(metrics["dice"], batch_size)
            iou_meter.update(metrics["iou"], batch_size)
            precision_meter.update(metrics["precision"], batch_size)
            recall_meter.update(metrics["recall"], batch_size)
            progress.set_postfix(loss=f"{loss_meter.avg:.4f}", dice=f"{dice_meter.avg:.4f}")

    return SegmentationMetrics(
        loss=loss_meter.avg,
        dice=dice_meter.avg,
        iou=iou_meter.avg,
        precision=precision_meter.avg,
        recall=recall_meter.avg,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved ResNet-UNet decoder checkpoint.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None, help="Override project.output_dir.")
    parser.add_argument("--threshold", type=float, default=None, help="Override training.threshold.")
    args = parser.parse_args()

    config = load_config(args.config)
    project_config = section(config, "project")
    data_config = section(config, "data")
    model_config = section(config, "model")
    training_config = section(config, "training")
    if args.output_dir is not None:
        project_config["output_dir"] = args.output_dir
    if args.threshold is not None:
        training_config["threshold"] = float(args.threshold)

    seed_everything(int(project_config.get("seed", 42)))
    output_dir = ensure_dir(resolve_path(project_config.get("output_dir", "runs/decoder"), PROJECT_ROOT))

    data_root = resolve_path(data_config.get("root", "../Dataset"), PROJECT_ROOT)
    image_channels = int(data_config.get("image_channels", 1))
    image_size = int(data_config.get("image_size", 128))
    image_transform = build_eval_image_transform(image_size, image_channels)
    mask_transform = build_mask_transform(image_size)
    split = str(args.split)
    split_key = f"{split}_split"
    loader = build_loader(
        data_root,
        str(data_config.get(split_key, split)),
        data_config,
        image_transform,
        mask_transform,
    )

    device = get_device()
    encoder = ResNetEncoder(
        backbone=str(model_config.get("backbone", "resnet18")),
        in_channels=image_channels,
    )
    model = ResNetUNet(
        encoder=encoder,
        freeze_encoder=bool(model_config.get("freeze_encoder", False)),
    ).to(device)

    checkpoint_path = resolve_path(args.checkpoint, PROJECT_ROOT)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    metrics = evaluate(
        model,
        loader,
        device,
        dice_weight=float(training_config.get("dice_weight", 1.0)),
        threshold=float(training_config.get("threshold", 0.5)),
        desc=f"evaluate {split}",
    )
    result = {
        "split": split,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "threshold": float(training_config.get("threshold", 0.5)),
        "metrics": metrics,
    }
    output_name = args.output_name or f"{split}_metrics.json"
    write_json(output_dir / output_name, result)
    print(
        "split={split} checkpoint_epoch={epoch} threshold={threshold:.3f} loss={loss:.5f} dice={dice:.5f} "
        "iou={iou:.5f} precision={precision:.5f} recall={recall:.5f}".format(
            split=split,
            epoch=result["checkpoint_epoch"],
            threshold=result["threshold"],
            loss=metrics.loss,
            dice=metrics.dice,
            iou=metrics.iou,
            precision=metrics.precision,
            recall=metrics.recall,
        )
    )


if __name__ == "__main__":
    main()
