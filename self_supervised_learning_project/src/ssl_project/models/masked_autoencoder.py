from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .resnet import ResNetEncoder


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.net(x)


class MaskedAutoencoder(nn.Module):
    """CNN masked image model that reconstructs only masked regions."""

    def __init__(self, encoder: ResNetEncoder, out_channels: int = 1) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.Sequential(
            DecoderBlock(encoder.feature_dim, 256),
            DecoderBlock(256, 128),
            DecoderBlock(128, 64),
            DecoderBlock(64, 32),
            DecoderBlock(32, 16),
        )
        self.head = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        features = self.encoder.forward_features(x)
        reconstruction = self.head(self.decoder(features))
        if reconstruction.shape[-2:] != input_size:
            reconstruction = F.interpolate(
                reconstruction,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )
        return reconstruction
