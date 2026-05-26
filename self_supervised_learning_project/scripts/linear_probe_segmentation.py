from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ssl_project.augmentations import build_eval_image_transform, build_mask_transform
from ssl_project.config import load_config, resolve_path, section
from ssl_project.data import SegmentationPairDataset
from ssl_project.losses import soft_dice_loss
from ssl_project.metrics import SegmentationMetrics, binary_segmentation_metrics
from ssl_project.models.resnet import ResNetEncoder, load_encoder_state
from ssl_project.utils import AverageMeter, ensure_dir, get_device, seed_everything, write_json


class FrozenSegmentationProbe(nn.Module):
    def __init__(self, encoder: ResNetEncoder) -> None:
        super().__init__()
        self.encoder = encoder
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.head = nn.Conv2d(encoder.feature_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.encoder.forward_features(x)
        logits = self.head(features)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def build_loader(
    data_root: Path,
    split: str,
    data_config: dict[str, Any],
    image_transform: Any,
    mask_transform: Any,
    shuffle: bool,
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
        shuffle=shuffle,
        num_workers=int(data_config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(
    model: FrozenSegmentationProbe,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    dice_weight: float,
    threshold: float,
    desc: str,
) -> SegmentationMetrics:
    training = optimizer is not None
    model.train(mode=training)
    model.encoder.eval()

    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()

    progress = tqdm(loader, desc=desc, leave=False, disable=not sys.stderr.isatty())
    for images, masks, _ in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        bce = F.binary_cross_entropy_with_logits(logits, masks)
        dice_loss = soft_dice_loss(logits, masks)
        loss = bce + dice_weight * dice_loss

        if training:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
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


def save_probe(
    path: Path,
    model: FrozenSegmentationProbe,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: SegmentationMetrics,
    config: dict[str, Any],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "metrics": metrics.__dict__,
            "config": config,
            "head": model.head.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a frozen-encoder segmentation probe.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None, help="Resume from a saved probe checkpoint.")
    parser.add_argument("--stop-epoch", type=int, default=None, help="Stop after this absolute epoch.")
    args = parser.parse_args()

    config = load_config(args.config)
    project_config = section(config, "project")
    data_config = section(config, "data")
    model_config = section(config, "model")
    training_config = section(config, "training")

    seed_everything(int(project_config.get("seed", 42)))
    output_dir = ensure_dir(resolve_path(project_config.get("output_dir", "runs/linear_probe_segmentation"), PROJECT_ROOT))
    write_json(output_dir / "config.json", config)

    data_root = resolve_path(data_config.get("root", "../Dataset"), PROJECT_ROOT)
    image_channels = int(data_config.get("image_channels", 1))
    image_size = int(data_config.get("image_size", 256))
    image_transform = build_eval_image_transform(image_size, image_channels)
    mask_transform = build_mask_transform(image_size)
    train_loader = build_loader(
        data_root,
        str(data_config.get("train_split", "train")),
        data_config,
        image_transform,
        mask_transform,
        shuffle=True,
    )
    val_loader = build_loader(
        data_root,
        str(data_config.get("val_split", "val")),
        data_config,
        image_transform,
        mask_transform,
        shuffle=False,
    )

    device = get_device()
    encoder = ResNetEncoder(
        backbone=str(model_config.get("backbone", "resnet18")),
        in_channels=image_channels,
    )
    checkpoint_path = resolve_path(model_config.get("checkpoint", "runs/simclr_medical/best.pt"), PROJECT_ROOT)
    load_encoder_state(encoder, str(checkpoint_path), map_location=device)
    model = FrozenSegmentationProbe(encoder).to(device)

    optimizer = torch.optim.AdamW(
        model.head.parameters(),
        lr=float(training_config.get("lr", 1e-3)),
        weight_decay=float(training_config.get("weight_decay", 1e-4)),
    )

    start_epoch = 1
    history: list[dict[str, Any]] = []
    history_path = output_dir / "history.json"
    if history_path.exists():
        import json

        payload = json.loads(history_path.read_text(encoding="utf-8"))
        history = payload.get("history", [])
    if args.resume is not None:
        resume_path = resolve_path(args.resume, PROJECT_ROOT)
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.head.load_state_dict(checkpoint["head"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1

    dice_weight = float(training_config.get("dice_weight", 1.0))
    threshold = float(training_config.get("threshold", 0.5))
    epochs = int(args.stop_epoch or training_config.get("epochs", 20))

    best_dice = -1.0
    if history:
        best_dice = max(float(item["val"]["dice"]) for item in history if "val" in item)
    for epoch in range(start_epoch, epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            dice_weight=dice_weight,
            threshold=threshold,
            desc=f"probe train {epoch}",
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                None,
                device,
                dice_weight=dice_weight,
                threshold=threshold,
                desc=f"probe val {epoch}",
            )

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
        )
        write_json(output_dir / "history.json", {"history": history})
        save_probe(output_dir / "last_probe.pt", model, optimizer, epoch, val_metrics, config)
        if val_metrics.dice > best_dice:
            best_dice = val_metrics.dice
            save_probe(output_dir / "best_probe.pt", model, optimizer, epoch, val_metrics, config)

        print(
            "epoch={epoch:03d} train_loss={train_loss:.5f} "
            "val_loss={val_loss:.5f} val_dice={val_dice:.5f} val_iou={val_iou:.5f}".format(
                epoch=epoch,
                train_loss=train_metrics.loss,
                val_loss=val_metrics.loss,
                val_dice=val_metrics.dice,
                val_iou=val_metrics.iou,
            )
        )

    print(f"Saved segmentation probe checkpoints to: {output_dir}")


if __name__ == "__main__":
    main()
