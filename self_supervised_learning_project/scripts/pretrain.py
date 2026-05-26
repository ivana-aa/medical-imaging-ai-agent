from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ssl_project.augmentations import build_simclr_transform
from ssl_project.config import load_config, resolve_path, section
from ssl_project.data import ContrastiveImageDataset, UnlabeledImageDataset
from ssl_project.losses import NTXentLoss
from ssl_project.models import ResNetEncoder, SimCLR
from ssl_project.utils import AverageMeter, ensure_dir, get_device, seed_everything, write_json


def build_optimizer(model: torch.nn.Module, training_config: dict[str, Any]) -> torch.optim.Optimizer:
    lr = float(training_config.get("lr", 3e-4))
    weight_decay = float(training_config.get("weight_decay", 1e-4))
    optimizer_name = str(training_config.get("optimizer", "adamw")).lower()
    if optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def save_checkpoint(
    path: Path,
    model: SimCLR,
    optimizer: torch.optim.Optimizer,
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
        },
        path,
    )


def train_one_epoch(
    model: SimCLR,
    loader: DataLoader,
    criterion: NTXentLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int,
) -> float:
    model.train()
    meter = AverageMeter()
    progress = tqdm(loader, desc=f"pretrain epoch {epoch}", leave=False, disable=not sys.stderr.isatty())
    for step, (view_a, view_b, _) in enumerate(progress, start=1):
        view_a = view_a.to(device, non_blocking=True)
        view_b = view_b.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        z_a = model(view_a)
        z_b = model(view_b)
        loss = criterion(z_a, z_b)
        loss.backward()
        optimizer.step()

        batch_size = view_a.shape[0]
        meter.update(float(loss.item()), batch_size)
        if step % log_interval == 0:
            progress.set_postfix(loss=f"{meter.avg:.4f}")
    return meter.avg


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain a SimCLR encoder from scratch.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    project_config = section(config, "project")
    data_config = section(config, "data")
    augment_config = section(config, "augment")
    model_config = section(config, "model")
    training_config = section(config, "training")

    seed_everything(int(project_config.get("seed", 42)))
    output_dir = ensure_dir(resolve_path(project_config.get("output_dir", "runs/simclr"), PROJECT_ROOT))
    write_json(output_dir / "config.json", config)

    data_root = resolve_path(data_config.get("root", "../Dataset"), PROJECT_ROOT)
    image_channels = int(data_config.get("image_channels", 1))
    image_size = int(data_config.get("image_size", 256))
    train_split = str(data_config.get("train_split", "train"))
    image_dir_name = str(data_config.get("image_dir_name", "images"))

    base_dataset = UnlabeledImageDataset(
        root=data_root,
        split=train_split,
        image_dir_name=image_dir_name,
        image_channels=image_channels,
    )
    transform = build_simclr_transform(image_size, image_channels, augment_config)
    dataset = ContrastiveImageDataset(base_dataset, transform)
    loader = DataLoader(
        dataset,
        batch_size=int(data_config.get("batch_size", 16)),
        shuffle=True,
        num_workers=int(data_config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    if len(loader) == 0:
        raise ValueError("No training batches. Increase dataset size or reduce batch_size.")

    device = get_device()
    encoder = ResNetEncoder(
        backbone=str(model_config.get("backbone", "resnet18")),
        in_channels=image_channels,
    )
    model = SimCLR(
        encoder=encoder,
        projection_hidden_dim=int(model_config.get("projection_hidden_dim", 512)),
        projection_dim=int(model_config.get("projection_dim", 128)),
    ).to(device)
    criterion = NTXentLoss(temperature=float(training_config.get("temperature", 0.2)))
    optimizer = build_optimizer(model, training_config)

    scheduler = None
    if str(training_config.get("scheduler", "cosine")).lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(training_config.get("epochs", 20)),
        )

    best_loss = float("inf")
    epochs = int(training_config.get("epochs", 20))
    save_every = int(training_config.get("save_every", 1))
    log_interval = int(training_config.get("log_interval", 20))

    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, loader, criterion, optimizer, device, epoch, log_interval)
        if scheduler is not None:
            scheduler.step()

        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append({"epoch": epoch, "loss": loss, "lr": current_lr})
        write_json(output_dir / "history.json", {"history": history})
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, loss, config)
        if epoch % save_every == 0:
            save_checkpoint(output_dir / f"epoch_{epoch:03d}.pt", model, optimizer, epoch, loss, config)
        if loss < best_loss:
            best_loss = loss
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, loss, config)

        print(f"epoch={epoch:03d} loss={loss:.5f} lr={current_lr:.6g}")

    print(f"Saved SimCLR checkpoints to: {output_dir}")


if __name__ == "__main__":
    main()
