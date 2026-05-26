from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ssl_project.augmentations import build_reconstruction_transform
from ssl_project.config import load_config, resolve_path, section
from ssl_project.data import UnlabeledImageDataset
from ssl_project.models import MaskedAutoencoder, ResNetEncoder
from ssl_project.utils import AverageMeter, ensure_dir, get_device, seed_everything, write_json


class ReconstructionDataset(Dataset):
    def __init__(self, base_dataset: UnlabeledImageDataset, transform: Any) -> None:
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        image, path = self.base_dataset[index]
        return self.transform(image), path


def make_block_mask(
    images: torch.Tensor,
    mask_ratio: float,
    patch_size: int,
) -> torch.Tensor:
    batch_size, _, height, width = images.shape
    grid_h = max(1, height // patch_size)
    grid_w = max(1, width // patch_size)
    num_patches = grid_h * grid_w
    num_mask = max(1, min(num_patches, int(round(num_patches * mask_ratio))))

    masks = torch.zeros(batch_size, num_patches, device=images.device)
    random_scores = torch.rand(batch_size, num_patches, device=images.device)
    indices = random_scores.argsort(dim=1)[:, :num_mask]
    masks.scatter_(1, indices, 1.0)
    masks = masks.view(batch_size, 1, grid_h, grid_w)
    return F.interpolate(masks, size=(height, width), mode="nearest")


def reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_name: str,
) -> torch.Tensor:
    if loss_name == "l1":
        loss = (reconstruction - target).abs()
    elif loss_name == "smooth_l1":
        loss = F.smooth_l1_loss(reconstruction, target, reduction="none")
    elif loss_name == "mse":
        loss = (reconstruction - target).pow(2)
    else:
        raise ValueError(f"Unsupported reconstruction loss: {loss_name}")
    return (loss * mask).sum() / (mask.sum() * target.shape[1]).clamp_min(1.0)


def build_optimizer(model: torch.nn.Module, training_config: dict[str, Any]) -> torch.optim.Optimizer:
    lr = float(training_config.get("lr", 1e-3))
    weight_decay = float(training_config.get("weight_decay", 1e-4))
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def save_checkpoint(
    path: Path,
    model: MaskedAutoencoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    loss: float,
    config: dict[str, Any],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "loss": loss,
            "config": config,
            "model": model.state_dict(),
            "encoder": model.encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
        },
        path,
    )


def train_one_epoch(
    model: MaskedAutoencoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    training_config: dict[str, Any],
) -> float:
    model.train()
    meter = AverageMeter()
    mask_ratio = float(training_config.get("mask_ratio", 0.6))
    patch_size = int(training_config.get("patch_size", 16))
    fill_value = float(training_config.get("mask_fill_value", 0.0))
    loss_name = str(training_config.get("loss", "smooth_l1")).lower()

    progress = tqdm(loader, desc=f"masked pretrain {epoch}", leave=False, disable=not sys.stderr.isatty())
    for images, _ in progress:
        images = images.to(device, non_blocking=True)
        mask = make_block_mask(images, mask_ratio=mask_ratio, patch_size=patch_size)
        masked_images = images * (1.0 - mask) + fill_value * mask

        optimizer.zero_grad(set_to_none=True)
        reconstruction = model(masked_images)
        loss = reconstruction_loss(reconstruction, images, mask, loss_name)
        loss.backward()
        optimizer.step()

        meter.update(float(loss.item()), images.shape[0])
        progress.set_postfix(loss=f"{meter.avg:.5f}")
    return meter.avg


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain an encoder with masked image reconstruction.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None, help="Resume from a saved masked pretrain checkpoint.")
    parser.add_argument(
        "--resume-model-only",
        action="store_true",
        help="Load model weights from --resume but reset optimizer and scheduler state.",
    )
    parser.add_argument("--stop-epoch", type=int, default=None, help="Stop after this absolute epoch.")
    args = parser.parse_args()

    config = load_config(args.config)
    project_config = section(config, "project")
    data_config = section(config, "data")
    augment_config = section(config, "augment")
    model_config = section(config, "model")
    training_config = section(config, "training")

    seed_everything(int(project_config.get("seed", 42)))
    output_dir = ensure_dir(resolve_path(project_config.get("output_dir", "runs/masked_autoencoder"), PROJECT_ROOT))
    write_json(output_dir / "config.json", config)

    data_root = resolve_path(data_config.get("root", "../Dataset"), PROJECT_ROOT)
    image_channels = int(data_config.get("image_channels", 1))
    image_size = int(data_config.get("image_size", 128))
    base_dataset = UnlabeledImageDataset(
        root=data_root,
        split=str(data_config.get("train_split", "train")),
        image_dir_name=str(data_config.get("image_dir_name", "images")),
        image_channels=image_channels,
    )
    transform = build_reconstruction_transform(image_size, image_channels, augment_config)
    dataset = ReconstructionDataset(base_dataset, transform)
    loader = DataLoader(
        dataset,
        batch_size=int(data_config.get("batch_size", 16)),
        shuffle=True,
        num_workers=int(data_config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    device = get_device()
    encoder = ResNetEncoder(
        backbone=str(model_config.get("backbone", "resnet18")),
        in_channels=image_channels,
    )
    model = MaskedAutoencoder(encoder=encoder, out_channels=image_channels).to(device)
    optimizer = build_optimizer(model, training_config)

    start_epoch = 1
    resume_checkpoint: dict[str, Any] | None = None
    history: list[dict[str, float]] = []
    history_path = output_dir / "history.json"
    if history_path.exists():
        import json

        payload = json.loads(history_path.read_text(encoding="utf-8"))
        history = payload.get("history", [])
    if args.resume is not None:
        resume_path = resolve_path(args.resume, PROJECT_ROOT)
        resume_checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(resume_checkpoint["model"])
        if not args.resume_model_only and "optimizer" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer"])
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        if args.resume_model_only:
            print(f"Loaded masked pretrain model weights: {resume_path}")
        else:
            print(f"Resumed masked pretrain checkpoint: {resume_path}")

    scheduler = None
    if str(training_config.get("scheduler", "cosine")).lower() == "cosine":
        t_max = int(training_config.get("scheduler_t_max", training_config.get("epochs", 20)))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
        )
        if (
            resume_checkpoint is not None
            and not args.resume_model_only
            and resume_checkpoint.get("scheduler") is not None
        ):
            scheduler.load_state_dict(resume_checkpoint["scheduler"])

    best_loss = float("inf")
    if history:
        best_loss = min(float(item["loss"]) for item in history if "loss" in item)
    if resume_checkpoint is not None and "loss" in resume_checkpoint:
        best_loss = min(best_loss, float(resume_checkpoint["loss"]))
    epochs = int(args.stop_epoch or training_config.get("epochs", 20))
    save_every = int(training_config.get("save_every", 10))
    for epoch in range(start_epoch, epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, device, epoch, training_config)
        if scheduler is not None:
            scheduler.step()

        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append({"epoch": epoch, "loss": loss, "lr": current_lr})
        write_json(output_dir / "history.json", {"history": history})
        save_checkpoint(output_dir / "last.pt", model, optimizer, scheduler, epoch, loss, config)
        if epoch % save_every == 0:
            save_checkpoint(output_dir / f"epoch_{epoch:03d}.pt", model, optimizer, scheduler, epoch, loss, config)
        if loss < best_loss:
            best_loss = loss
            save_checkpoint(output_dir / "best.pt", model, optimizer, scheduler, epoch, loss, config)

        print(f"epoch={epoch:03d} masked_loss={loss:.6f} lr={current_lr:.6g}")

    print(f"Saved masked autoencoder checkpoints to: {output_dir}")


if __name__ == "__main__":
    main()
