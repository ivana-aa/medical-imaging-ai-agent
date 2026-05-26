from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.2) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        if z_i.shape != z_j.shape:
            raise ValueError("z_i and z_j must have the same shape")
        batch_size = z_i.shape[0]
        if batch_size < 2:
            raise ValueError("NT-Xent requires batch_size >= 2")

        z = torch.cat([z_i, z_j], dim=0)
        z = F.normalize(z, dim=1)
        similarity = torch.matmul(z, z.T) / self.temperature

        self_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        similarity = similarity.masked_fill(self_mask, float("-inf"))

        positive_index = torch.arange(2 * batch_size, device=z.device)
        positive_index = (positive_index + batch_size) % (2 * batch_size)
        positives = similarity[torch.arange(2 * batch_size, device=z.device), positive_index].unsqueeze(1)

        negative_mask = ~self_mask
        negative_mask[torch.arange(2 * batch_size, device=z.device), positive_index] = False
        negatives = similarity[negative_mask].view(2 * batch_size, -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z.device)
        return F.cross_entropy(logits, labels)


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    intersection = torch.sum(probs * targets, dim=dims)
    denominator = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


def soft_tversky_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.7,
    beta: float = 0.3,
    smooth: float = 1.0,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    true_positive = torch.sum(probs * targets, dim=dims)
    false_positive = torch.sum(probs * (1.0 - targets), dim=dims)
    false_negative = torch.sum((1.0 - probs) * targets, dim=dims)
    tversky = (true_positive + smooth) / (
        true_positive + alpha * false_positive + beta * false_negative + smooth
    )
    return 1.0 - tversky.mean()
