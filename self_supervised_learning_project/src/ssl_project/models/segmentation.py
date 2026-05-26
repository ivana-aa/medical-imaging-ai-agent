from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .resnet import ResNetEncoder


class ConvBlock(nn.Module):
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
        return self.net(x)


class ResNetUNet(nn.Module):
    """Small decoder for dense segmentation on top of the project ResNet encoder."""

    def __init__(self, encoder: ResNetEncoder, freeze_encoder: bool = False) -> None:
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)

        self.decode3 = ConvBlock(512 + 256, 256)
        self.decode2 = ConvBlock(256 + 128, 128)
        self.decode1 = ConvBlock(128 + 64, 64)
        self.decode0 = ConvBlock(64, 64)
        self.refine = ConvBlock(64, 32)
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    @staticmethod
    def _upsample_like(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        if self.freeze_encoder:
            with torch.no_grad():
                features = self.encoder.forward_pyramid(x)
        else:
            features = self.encoder.forward_pyramid(x)

        y = self._upsample_like(features["layer4"], features["layer3"])
        y = self.decode3(torch.cat([y, features["layer3"]], dim=1))

        y = self._upsample_like(y, features["layer2"])
        y = self.decode2(torch.cat([y, features["layer2"]], dim=1))

        y = self._upsample_like(y, features["layer1"])
        y = self.decode1(torch.cat([y, features["layer1"]], dim=1))

        y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)
        y = self.decode0(y)
        y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)
        y = self.refine(y)
        logits = self.head(y)
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return logits
