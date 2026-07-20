from typing import Tuple

import torch
from torch import nn


class SimpleCNN(nn.Module):
    """Small CNN intended for Raspberry Pi image classification experiments."""

    def __init__(self, num_classes: int, input_size: Tuple[int, int] = (64, 64)) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def _torchvision_model(model_name: str, num_classes: int) -> nn.Module:
    try:
        from torchvision import models
    except ImportError as exc:
        raise RuntimeError(
            f"The {model_name} model requires torchvision to be installed."
        ) from exc

    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features, num_classes
        )
        return model
    if model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features, num_classes
        )
        return model
    if model_name == "swin_t":
        model = models.swin_t(weights=None)
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model
    raise ValueError(f"Unknown torchvision model: {model_name}")


def get_model(model_name: str, num_classes: int, input_size: Tuple[int, int] = (64, 64)) -> nn.Module:
    if model_name in {"simple_cnn", "SimpleCNN"}:
        return SimpleCNN(num_classes=num_classes, input_size=input_size)
    normalized_name = model_name.lower().replace("-", "_")
    aliases = {
        "resnet_18": "resnet18",
        "mobilenetv3_large": "mobilenet_v3_large",
        "mobilenetv3_small": "mobilenet_v3_small",
        "swin_tiny": "swin_t",
    }
    normalized_name = aliases.get(normalized_name, normalized_name)
    if normalized_name in {
        "resnet18",
        "mobilenet_v3_large",
        "mobilenet_v3_small",
        "swin_t",
    }:
        return _torchvision_model(normalized_name, num_classes)
    raise ValueError(f"Unknown model: {model_name}")
