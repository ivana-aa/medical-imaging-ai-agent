from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .dataset import loader_sample_count, make_loader, parse_size
from .losses import build_criterion
from .metrics import Metrics, evaluate_loader, find_best_threshold
from .model import AttentionUNet
from .postprocess import (
    build_postprocess_configs,
    build_threshold_grid,
    parse_bool_list,
    parse_float_list,
    parse_int_list,
    postprocess_from_dict,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
DEFAULT_DATA_DIR = REPOSITORY_ROOT / "Dataset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "attention_unet_fair_seed42"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _namespace_to_jsonable(args: argparse.Namespace) -> dict:
    result = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, tuple):
            result[key] = list(value)
        else:
            result[key] = value
    return result


def save_run_config(args: argparse.Namespace, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_namespace_to_jsonable(args), indent=2), encoding="utf-8")


def append_history(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_threshold_info(path: Path, epoch: int, metrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "epoch": epoch,
                "best_threshold": metrics.threshold,
                "val_dice": metrics.dice,
                "val_iou": metrics.iou,
                "val_precision": metrics.precision,
                "val_recall": metrics.recall,
                "postprocess": metrics.postprocess,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_dice: float,
    best_threshold: float,
    args: argparse.Namespace,
    best_postprocess: Optional[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": "AttentionUNet",
            "epoch": epoch,
            "best_dice": best_dice,
            "best_threshold": best_threshold,
            "best_postprocess": best_postprocess,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "args": _namespace_to_jsonable(args),
        },
        path,
    )


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
        t_max = args.scheduler_t_max or args.epochs
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=args.min_lr)
    if args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.scheduler_step_size,
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


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool,
    max_grad_norm: float,
) -> Metrics:
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_samples = 0

    from .metrics import binary_metrics

    for images, masks, _ in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)

        if use_amp:
            scaler.scale(loss).backward()
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        metrics = binary_metrics(logits.detach(), masks, threshold=0.5)
        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_dice += metrics.dice * batch_size
        total_iou += metrics.iou * batch_size
        total_precision += metrics.precision * batch_size
        total_recall += metrics.recall * batch_size
        total_samples += batch_size

    return Metrics(
        loss=total_loss / total_samples,
        dice=total_dice / total_samples,
        iou=total_iou / total_samples,
        precision=total_precision / total_samples,
        recall=total_recall / total_samples,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an independent Attention U-Net segmentation model.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--image-size", type=parse_size, default=(256, 256))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=0, help="Optional smoke-test cap; 0 uses all train samples.")
    parser.add_argument("--max-val-samples", type=int, default=0, help="Optional smoke-test cap; 0 uses all val samples.")
    parser.add_argument("--augment", choices=["off", "basic", "strong"], default="basic")
    parser.add_argument("--loss", choices=["dice_bce", "combo", "bce", "dice", "focal"], default="combo")
    parser.add_argument("--loss-dice-weight", type=float, default=1.0)
    parser.add_argument("--loss-bce-weight", type=float, default=1.0)
    parser.add_argument("--loss-focal-weight", type=float, default=0.5)
    parser.add_argument("--dice-smooth", type=float, default=1.0)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--scheduler", choices=["plateau", "cosine", "step", "none"], default="plateau")
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--scheduler-step-size", type=int, default=10)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--scheduler-t-max", type=int, default=0)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--tta", choices=["off", "flip"], default="off")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threshold-min", type=float, default=0.20)
    parser.add_argument("--threshold-max", type=float, default=0.60)
    parser.add_argument("--threshold-steps", type=int, default=17)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--reset-optimizer", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument("--postprocess", action="store_true")
    parser.add_argument("--postprocess-search", action="store_true")
    parser.add_argument("--post-close-iters", type=int, default=0)
    parser.add_argument("--post-open-iters", type=int, default=0)
    parser.add_argument("--post-fill-holes", action="store_true")
    parser.add_argument("--post-min-area", type=int, default=0)
    parser.add_argument("--post-min-area-ratio", type=float, default=0.0)
    parser.add_argument("--post-search-close-iters", type=parse_int_list, default=parse_int_list("0,1"))
    parser.add_argument("--post-search-open-iters", type=parse_int_list, default=parse_int_list("0"))
    parser.add_argument("--post-search-fill-holes", type=parse_bool_list, default=parse_bool_list("false,true"))
    parser.add_argument("--post-search-min-areas", type=parse_int_list, default=parse_int_list("0,16,64"))
    parser.add_argument("--post-search-min-area-ratios", type=parse_float_list, default=parse_float_list("0"))
    parser.add_argument("--postprocess-search-output", type=Path, default=None)
    return parser


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = make_loader(
        args.data_dir,
        "train",
        args.image_size,
        args.batch_size,
        args.num_workers,
        augment=args.augment,
        shuffle=True,
        max_samples=args.max_train_samples,
    )
    val_loader = make_loader(
        args.data_dir,
        "val",
        args.image_size,
        args.batch_size,
        args.num_workers,
        augment="off",
        shuffle=False,
        max_samples=args.max_val_samples,
    )
    print(f"Train samples: {loader_sample_count(train_loader)}")
    print(f"Val samples: {loader_sample_count(val_loader)}")

    model = AttentionUNet(in_channels=1, out_channels=1, base_channels=args.base_channels).to(device)
    criterion = build_criterion(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = build_scheduler(args, optimizer)

    start_epoch = 1
    best_dice = 0.0
    best_threshold = args.threshold
    best_postprocess = None
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint and not args.reset_optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler is not None and checkpoint.get("scheduler_state") is not None and not args.reset_optimizer:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_dice = float(checkpoint.get("best_dice", 0.0))
        best_threshold = float(checkpoint.get("best_threshold", args.threshold))
        best_postprocess = checkpoint.get("best_postprocess")
        print(f"Loaded Attention U-Net checkpoint: {args.checkpoint}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(args, args.output_dir / "ablation_config.json")
    best_path = args.output_dir / "best_attention_unet.pt"
    last_path = args.output_dir / "last_attention_unet.pt"
    history_path = args.output_dir / "history.csv"
    threshold_info_path = args.output_dir / "best_threshold.json"
    threshold_grid = build_threshold_grid(args.threshold_min, args.threshold_max, args.threshold_steps)
    postprocess_configs = build_postprocess_configs(args)
    search_output_path = args.postprocess_search_output
    if search_output_path is None and args.postprocess_search:
        search_output_path = args.output_dir / "postprocess_ablation.csv"

    epochs_without_improvement = 0
    use_amp = device.type == "cuda" and not args.no_amp
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            use_amp=use_amp,
            max_grad_norm=args.max_grad_norm,
        )
        threshold_metrics = find_best_threshold(
            model,
            val_loader,
            device,
            threshold_grid,
            postprocess_configs,
            split_name="val-threshold",
            search_output_path=search_output_path,
            tta=args.tta,
        )
        selected_postprocess = postprocess_from_dict(threshold_metrics.postprocess)
        val_metrics = evaluate_loader(
            model,
            val_loader,
            criterion,
            device,
            split_name="val",
            threshold=threshold_metrics.threshold,
            postprocess=selected_postprocess,
            tta=args.tta,
        )
        step_scheduler(scheduler, args.scheduler, threshold_metrics.dice)

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics.loss,
            "train_dice": train_metrics.dice,
            "train_iou": train_metrics.iou,
            "train_precision": train_metrics.precision,
            "train_recall": train_metrics.recall,
            "val_loss": val_metrics.loss,
            "val_dice": val_metrics.dice,
            "val_iou": val_metrics.iou,
            "val_precision": val_metrics.precision,
            "val_recall": val_metrics.recall,
            "val_best_threshold": threshold_metrics.threshold,
            "val_best_dice": threshold_metrics.dice,
            "val_best_iou": threshold_metrics.iou,
            "val_best_precision": threshold_metrics.precision,
            "val_best_recall": threshold_metrics.recall,
        }
        append_history(history_path, row)
        save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_dice, best_threshold, args, best_postprocess)

        improved = threshold_metrics.dice > best_dice + args.early_stopping_min_delta
        if improved:
            best_dice = threshold_metrics.dice
            best_threshold = threshold_metrics.threshold
            best_postprocess = threshold_metrics.postprocess
            epochs_without_improvement = 0
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_dice, best_threshold, args, best_postprocess)
            save_threshold_info(threshold_info_path, epoch, threshold_metrics)
        else:
            epochs_without_improvement += 1

        print(
            f"train loss={train_metrics.loss:.4f} dice={train_metrics.dice:.4f} iou={train_metrics.iou:.4f} | "
            f"val loss={val_metrics.loss:.4f} dice={val_metrics.dice:.4f} iou={val_metrics.iou:.4f} | "
            f"epoch_threshold={threshold_metrics.threshold:.2f} postprocess={threshold_metrics.postprocess} | "
            f"best_dice={best_dice:.4f} best_threshold={best_threshold:.2f}"
        )

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch}: no val Dice improvement for "
                f"{epochs_without_improvement} epoch(s)."
            )
            break


def main(argv: Optional[list[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
