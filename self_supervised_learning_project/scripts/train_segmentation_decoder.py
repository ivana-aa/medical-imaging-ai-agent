from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ssl_project.augmentations import (
    build_eval_image_transform,
    build_mask_transform,
    build_segmentation_train_pair_transform,
)
from ssl_project.config import load_config, resolve_path, section
from ssl_project.data import SegmentationPairDataset
from ssl_project.losses import soft_dice_loss, soft_tversky_loss
from ssl_project.metrics import SegmentationMetrics, binary_segmentation_metrics
from ssl_project.models import ResNetEncoder, ResNetUNet
from ssl_project.models.resnet import load_encoder_state
from ssl_project.utils import AverageMeter, ensure_dir, get_device, seed_everything, write_json


def build_loader(
    data_root: Path,
    split: str,
    data_config: dict[str, Any],
    image_transform: Any,
    mask_transform: Any,
    shuffle: bool,
    pair_transform: Any = None,
) -> DataLoader:
    dataset = SegmentationPairDataset(
        root=data_root,
        split=split,
        image_transform=image_transform,
        mask_transform=mask_transform,
        pair_transform=pair_transform,
        image_dir_name=str(data_config.get("image_dir_name", "images")),
        label_dir_name=str(data_config.get("label_dir_name", "labels")),
        image_channels=int(data_config.get("image_channels", 1)),
    )
    if shuffle:
        train_fraction = float(data_config.get("train_fraction", 1.0))
        if not 0.0 < train_fraction <= 1.0:
            raise ValueError(f"train_fraction must be in (0, 1], got {train_fraction}")
        if train_fraction < 1.0:
            original_size = len(dataset)
            generator = torch.Generator().manual_seed(int(data_config.get("subset_seed", 42)))
            subset_size = max(1, int(round(original_size * train_fraction)))
            indices = torch.randperm(original_size, generator=generator)[:subset_size].tolist()
            dataset = Subset(dataset, indices)
            print(f"Using {subset_size} / {original_size} labeled train pairs.")
    return DataLoader(
        dataset,
        batch_size=int(data_config.get("batch_size", 8)),
        shuffle=shuffle,
        num_workers=int(data_config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )


def build_optimizer(model: ResNetUNet, training_config: dict[str, Any]) -> torch.optim.Optimizer:
    encoder_lr = float(training_config.get("encoder_lr", training_config.get("lr", 1e-4)))
    decoder_lr = float(training_config.get("decoder_lr", training_config.get("lr", 1e-3)))
    weight_decay = float(training_config.get("weight_decay", 1e-4))

    decoder_params = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("encoder.") and parameter.requires_grad
    ]
    groups: list[dict[str, Any]] = [{"params": decoder_params, "lr": decoder_lr}]
    encoder_params = [parameter for parameter in model.encoder.parameters() if parameter.requires_grad]
    if encoder_params:
        groups.append({"params": encoder_params, "lr": encoder_lr})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def run_epoch(
    model: ResNetUNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    training_config: dict[str, Any],
    threshold: float,
    desc: str,
) -> SegmentationMetrics:
    training = optimizer is not None
    model.train(mode=training)
    if model.freeze_encoder:
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
        loss = compute_segmentation_loss(logits, masks, training_config)

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


def compute_segmentation_loss(
    logits: torch.Tensor,
    masks: torch.Tensor,
    training_config: dict[str, Any],
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, masks)
    loss_type = str(training_config.get("loss_type", "bce_dice")).lower()
    if loss_type == "bce_dice":
        dice_loss = soft_dice_loss(logits, masks)
        return bce + float(training_config.get("dice_weight", 1.0)) * dice_loss
    if loss_type == "bce_tversky":
        tversky_loss = soft_tversky_loss(
            logits,
            masks,
            alpha=float(training_config.get("tversky_alpha", 0.7)),
            beta=float(training_config.get("tversky_beta", 0.3)),
        )
        return bce + float(training_config.get("tversky_weight", 1.0)) * tversky_loss
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def save_checkpoint(
    path: Path,
    model: ResNetUNet,
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
            "model": model.state_dict(),
            "encoder": model.encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a ResNet-UNet segmentation decoder.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None, help="Resume from a saved decoder checkpoint.")
    parser.add_argument("--stop-epoch", type=int, default=None, help="Stop after this absolute epoch.")
    parser.add_argument("--subset-seed", type=int, default=None, help="Override data.subset_seed.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override project.output_dir.")
    args = parser.parse_args()

    config = load_config(args.config)
    project_config = section(config, "project")
    data_config = section(config, "data")
    augment_config = section(config, "augment")
    model_config = section(config, "model")
    training_config = section(config, "training")
    if args.subset_seed is not None:
        data_config["subset_seed"] = int(args.subset_seed)
    if args.output_dir is not None:
        project_config["output_dir"] = args.output_dir

    seed_everything(int(project_config.get("seed", 42)))
    output_dir = ensure_dir(resolve_path(project_config.get("output_dir", "runs/decoder"), PROJECT_ROOT))
    write_json(output_dir / "config.json", config)

    data_root = resolve_path(data_config.get("root", "../Dataset"), PROJECT_ROOT)
    image_channels = int(data_config.get("image_channels", 1))
    image_size = int(data_config.get("image_size", 128))
    image_transform = build_eval_image_transform(image_size, image_channels)
    mask_transform = build_mask_transform(image_size)
    pair_transform = None
    if bool(augment_config.get("enabled", False)):
        pair_transform = build_segmentation_train_pair_transform(
            image_size,
            image_channels,
            augment_config,
        )
        print(f"Enabled paired train augmentation: {augment_config}")

    train_loader = build_loader(
        data_root,
        str(data_config.get("train_split", "train")),
        data_config,
        image_transform,
        mask_transform,
        shuffle=True,
        pair_transform=pair_transform,
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
    checkpoint_value = model_config.get("checkpoint", "")
    if checkpoint_value:
        checkpoint_path = resolve_path(checkpoint_value, PROJECT_ROOT)
        load_encoder_state(encoder, str(checkpoint_path), map_location=device)
        print(f"Loaded encoder checkpoint: {checkpoint_path}")
    else:
        print("Training from random encoder initialization.")

    model = ResNetUNet(
        encoder=encoder,
        freeze_encoder=bool(model_config.get("freeze_encoder", False)),
    ).to(device)
    optimizer = build_optimizer(model, training_config)

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
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"Resumed decoder checkpoint: {resume_path}")

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
            training_config=training_config,
            threshold=threshold,
            desc=f"decoder train {epoch}",
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                None,
                device,
                training_config=training_config,
                threshold=threshold,
                desc=f"decoder val {epoch}",
            )

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        write_json(output_dir / "history.json", {"history": history})
        save_checkpoint(output_dir / "last_decoder.pt", model, optimizer, epoch, val_metrics, config)
        if val_metrics.dice > best_dice:
            best_dice = val_metrics.dice
            save_checkpoint(output_dir / "best_decoder.pt", model, optimizer, epoch, val_metrics, config)

        print(
            "epoch={epoch:03d} train_loss={train_loss:.5f} train_dice={train_dice:.5f} "
            "val_loss={val_loss:.5f} val_dice={val_dice:.5f} val_iou={val_iou:.5f}".format(
                epoch=epoch,
                train_loss=train_metrics.loss,
                train_dice=train_metrics.dice,
                val_loss=val_metrics.loss,
                val_dice=val_metrics.dice,
                val_iou=val_metrics.iou,
            )
        )

    print(f"Saved decoder checkpoints to: {output_dir}")


if __name__ == "__main__":
    main()
