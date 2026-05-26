from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from ssl_project.augmentations import (  # noqa: E402
    build_eval_image_transform,
    build_mask_transform,
    build_segmentation_train_pair_transform,
)
from ssl_project.config import load_config, resolve_path, section  # noqa: E402
from ssl_project.data import SegmentationPairDataset  # noqa: E402
from ssl_project.models import ResNetEncoder, ResNetUNet  # noqa: E402
from ssl_project.models.resnet import load_encoder_state  # noqa: E402
from ssl_project.utils import ensure_dir, get_device, seed_everything, write_json  # noqa: E402
from train_segmentation_decoder import (  # noqa: E402
    build_loader,
    build_optimizer,
    run_epoch,
    save_checkpoint,
)


def select_train_indices(dataset_size: int, data_config: dict[str, Any]) -> list[int]:
    train_fraction = float(data_config.get("train_fraction", 1.0))
    if not 0.0 < train_fraction <= 1.0:
        raise ValueError(f"train_fraction must be in (0, 1], got {train_fraction}")
    if train_fraction >= 1.0:
        return list(range(dataset_size))
    generator = torch.Generator().manual_seed(int(data_config.get("subset_seed", 42)))
    subset_size = max(1, int(round(dataset_size * train_fraction)))
    return torch.randperm(dataset_size, generator=generator)[:subset_size].tolist()


def build_train_subset(
    data_root: Path,
    data_config: dict[str, Any],
    image_transform: Any,
    mask_transform: Any,
    pair_transform: Any = None,
) -> tuple[Subset, list[int]]:
    dataset = SegmentationPairDataset(
        root=data_root,
        split=str(data_config.get("train_split", "train")),
        image_transform=image_transform,
        mask_transform=mask_transform,
        pair_transform=pair_transform,
        image_dir_name=str(data_config.get("image_dir_name", "images")),
        label_dir_name=str(data_config.get("label_dir_name", "labels")),
        image_channels=int(data_config.get("image_channels", 1)),
    )
    indices = select_train_indices(len(dataset), data_config)
    print(f"Using {len(indices)} / {len(dataset)} labeled train pairs.")
    return Subset(dataset, indices), indices


def compute_mining_rows(
    model: ResNetUNet,
    dataset: Subset,
    indices: list[int],
    data_config: dict[str, Any],
    hard_config: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    threshold = float(hard_config.get("threshold", 0.458))
    loader = DataLoader(
        dataset,
        batch_size=int(data_config.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(data_config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )

    rows: list[dict[str, Any]] = []
    model.eval()
    progress = tqdm(loader, desc="mine hard train cases", leave=False, disable=not sys.stderr.isatty())
    with torch.no_grad():
        offset = 0
        for images, masks, paths in progress:
            images = images.to(device, non_blocking=True)
            masks = (masks >= 0.5).float().to(device, non_blocking=True)
            probabilities = torch.sigmoid(model(images))
            preds = (probabilities >= threshold).float()
            dims = tuple(range(1, preds.ndim))
            tp = torch.sum(preds * masks, dim=dims)
            fp = torch.sum(preds * (1.0 - masks), dim=dims)
            fn = torch.sum((1.0 - preds) * masks, dim=dims)
            dice = (2.0 * tp + 1e-7) / (2.0 * tp + fp + fn + 1e-7)
            iou = (tp + 1e-7) / (tp + fp + fn + 1e-7)
            target_area = torch.sum(masks, dim=dims)
            pred_area = torch.sum(preds, dim=dims)
            max_prob = torch.amax(probabilities, dim=dims)

            for batch_index, path in enumerate(paths):
                subset_pos = offset + batch_index
                rows.append(
                    {
                        "subset_index": int(subset_pos),
                        "dataset_index": int(indices[subset_pos]),
                        "image_path": str(path),
                        "dice": float(dice[batch_index].item()),
                        "iou": float(iou[batch_index].item()),
                        "target_area": float(target_area[batch_index].item()),
                        "pred_area": float(pred_area[batch_index].item()),
                        "prob_max": float(max_prob[batch_index].item()),
                    }
                )
            offset += len(paths)
    return rows


def assign_hard_weights(rows: list[dict[str, Any]], hard_config: dict[str, Any]) -> list[float]:
    positive_weight = float(hard_config.get("positive_weight", 1.5))
    low_dice_cutoff = float(hard_config.get("low_dice_cutoff", 0.85))
    low_dice_weight = float(hard_config.get("low_dice_weight", 4.0))
    empty_positive_weight = float(hard_config.get("empty_positive_weight", 8.0))
    max_weight = float(hard_config.get("max_weight", empty_positive_weight))

    weights: list[float] = []
    for row in rows:
        weight = 1.0
        is_positive = float(row["target_area"]) > 0.0
        if is_positive:
            weight = max(weight, positive_weight)
            if float(row["dice"]) < low_dice_cutoff:
                weight = max(weight, low_dice_weight)
            if float(row["pred_area"]) <= 0.0:
                weight = max(weight, empty_positive_weight)
        weight = min(weight, max_weight)
        row["hard_weight"] = float(weight)
        weights.append(float(weight))
    return weights


def build_weighted_loader(
    dataset: Subset,
    weights: list[float],
    data_config: dict[str, Any],
    hard_config: dict[str, Any],
) -> DataLoader:
    multiplier = float(hard_config.get("num_samples_multiplier", 1.0))
    num_samples = max(1, int(round(len(weights) * multiplier)))
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
    )
    return DataLoader(
        dataset,
        batch_size=int(data_config.get("batch_size", 8)),
        sampler=sampler,
        num_workers=int(data_config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )


def summarize_mining_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    weights = torch.tensor([float(row["hard_weight"]) for row in rows], dtype=torch.float32)
    positives = [row for row in rows if float(row["target_area"]) > 0.0]
    low_dice = [row for row in positives if float(row["dice"]) < 0.85]
    empty_positive = [row for row in positives if float(row["pred_area"]) <= 0.0]
    return {
        "case_count": len(rows),
        "positive_cases": len(positives),
        "low_dice_positive_cases": len(low_dice),
        "empty_positive_cases": len(empty_positive),
        "weight_min": float(weights.min().item()),
        "weight_mean": float(weights.mean().item()),
        "weight_max": float(weights.max().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Hard-case weighted fine-tuning for segmentation decoder.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True, help="Decoder checkpoint to fine-tune from.")
    parser.add_argument("--stop-epoch", type=int, default=None, help="Stop after this absolute epoch.")
    parser.add_argument("--subset-seed", type=int, default=None, help="Override data.subset_seed.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override project.output_dir.")
    parser.add_argument("--resume-optimizer", action="store_true", help="Also resume optimizer state.")
    args = parser.parse_args()

    config = load_config(args.config)
    project_config = section(config, "project")
    data_config = section(config, "data")
    augment_config = section(config, "augment")
    model_config = section(config, "model")
    training_config = section(config, "training")
    hard_config = section(config, "hard_mining")
    if args.subset_seed is not None:
        data_config["subset_seed"] = int(args.subset_seed)
    if args.output_dir is not None:
        project_config["output_dir"] = args.output_dir

    seed_everything(int(project_config.get("seed", 42)))
    output_dir = ensure_dir(resolve_path(project_config.get("output_dir", "runs/decoder_hard"), PROJECT_ROOT))
    write_json(output_dir / "config.json", config)

    data_root = resolve_path(data_config.get("root", "../Dataset"), PROJECT_ROOT)
    image_channels = int(data_config.get("image_channels", 1))
    image_size = int(data_config.get("image_size", 128))
    image_transform = build_eval_image_transform(image_size, image_channels)
    mask_transform = build_mask_transform(image_size)

    device = get_device()
    encoder = ResNetEncoder(
        backbone=str(model_config.get("backbone", "resnet18")),
        in_channels=image_channels,
    )
    checkpoint_value = model_config.get("checkpoint", "")
    if checkpoint_value:
        encoder_checkpoint = resolve_path(checkpoint_value, PROJECT_ROOT)
        load_encoder_state(encoder, str(encoder_checkpoint), map_location=device)
        print(f"Loaded encoder checkpoint: {encoder_checkpoint}")

    model = ResNetUNet(
        encoder=encoder,
        freeze_encoder=bool(model_config.get("freeze_encoder", False)),
    ).to(device)
    optimizer = build_optimizer(model, training_config)

    resume_path = resolve_path(args.resume, PROJECT_ROOT)
    checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if args.resume_optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    print(f"Resumed decoder model: {resume_path}")

    mining_subset, train_indices = build_train_subset(
        data_root,
        data_config,
        image_transform,
        mask_transform,
        pair_transform=None,
    )
    mining_rows = compute_mining_rows(model, mining_subset, train_indices, data_config, hard_config, device)
    weights = assign_hard_weights(mining_rows, hard_config)
    mining_summary = summarize_mining_rows(mining_rows)
    write_json(
        output_dir / "hard_mining_train_cases.json",
        {
            "summary": mining_summary,
            "hard_mining": hard_config,
            "cases": mining_rows,
        },
    )
    print(
        "hard_mining cases={case_count} positives={positive_cases} low_dice={low_dice_positive_cases} "
        "empty_positive={empty_positive_cases} weight_mean={weight_mean:.3f} weight_max={weight_max:.1f}".format(
            **mining_summary
        )
    )

    pair_transform = None
    if bool(augment_config.get("enabled", False)):
        pair_transform = build_segmentation_train_pair_transform(
            image_size,
            image_channels,
            augment_config,
        )
        print(f"Enabled paired train augmentation: {augment_config}")
    train_subset, _ = build_train_subset(
        data_root,
        data_config,
        image_transform,
        mask_transform,
        pair_transform=pair_transform,
    )
    train_loader = build_weighted_loader(train_subset, weights, data_config, hard_config)
    val_loader = build_loader(
        data_root,
        str(data_config.get("val_split", "val")),
        data_config,
        image_transform,
        mask_transform,
        shuffle=False,
    )

    history: list[dict[str, Any]] = []
    threshold = float(training_config.get("threshold", 0.5))
    epochs = int(args.stop_epoch or training_config.get("epochs", start_epoch))
    best_dice = -1.0
    for epoch in range(start_epoch, epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            training_config=training_config,
            threshold=threshold,
            desc=f"hard train {epoch}",
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                None,
                device,
                training_config=training_config,
                threshold=threshold,
                desc=f"hard val {epoch}",
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

    print(f"Saved hard-mined decoder checkpoints to: {output_dir}")


if __name__ == "__main__":
    main()
