from __future__ import annotations

import torch
from torch import nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNetEncoder(nn.Module):
    def __init__(self, backbone: str = "resnet18", in_channels: int = 1) -> None:
        super().__init__()
        layers_by_name = {
            "resnet18": [2, 2, 2, 2],
            "resnet34": [3, 4, 6, 3],
        }
        if backbone not in layers_by_name:
            raise ValueError(f"Unsupported backbone '{backbone}'. Use one of: {sorted(layers_by_name)}")
        self.backbone_name = backbone
        self.in_channels = in_channels
        self.feature_dim = 512
        self._current_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        layers = layers_by_name[backbone]
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self._init_weights()

    def _make_layer(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(self._current_channels, out_channels, stride=stride)]
        self._current_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(self._current_channels, out_channels))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)

    def forward_pyramid(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        stem = self.stem(x)
        layer1 = self.layer1(stem)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return {
            "stem": stem,
            "layer1": layer1,
            "layer2": layer2,
            "layer3": layer3,
            "layer4": layer4,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        pooled = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(features, 1), 1)
        return pooled


def load_encoder_state(encoder: ResNetEncoder, checkpoint_path: str, map_location: str | torch.device = "cpu") -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    state = checkpoint.get("encoder")
    if state is None:
        state = checkpoint.get("encoder_state")
    if state is None:
        model_state = checkpoint.get("model")
        if model_state is None:
            model_state = checkpoint.get("model_state")
        if model_state is not None:
            state = {
                key.removeprefix("encoder."): value
                for key, value in model_state.items()
                if key.startswith("encoder.")
            }
    if not state:
        raise KeyError(f"Could not find encoder state in checkpoint: {checkpoint_path}")
    encoder.load_state_dict(state, strict=True)
    return checkpoint
