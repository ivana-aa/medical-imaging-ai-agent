from __future__ import annotations

import torch
from torch import nn

from .resnet import ResNetEncoder


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLR(nn.Module):
    def __init__(
        self,
        encoder: ResNetEncoder,
        projection_hidden_dim: int = 512,
        projection_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.projector = ProjectionHead(
            input_dim=encoder.feature_dim,
            hidden_dim=projection_hidden_dim,
            output_dim=projection_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.projector(features)
