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


def get_model(model_name: str, num_classes: int, input_size: Tuple[int, int] = (64, 64)) -> nn.Module:
    if model_name in {"simple_cnn", "SimpleCNN"}:
        return SimpleCNN(num_classes=num_classes, input_size=input_size)
    raise ValueError(f"Unknown model: {model_name}")
