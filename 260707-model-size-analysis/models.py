from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

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


def _make_divisible(value: float, divisor: int = 8, minimum: int = 8) -> int:
    return max(minimum, int(round(value / divisor) * divisor))


class ScalableCNN(nn.Module):
    """CNN whose depth is explicit and whose width can be PAM-calibrated.

    The network keeps a simple Conv-BN-ReLU structure for clean hardware
    interpretation. Width changes the channel count; depth changes the number
    of convolution blocks. Spatial downsampling is inserted periodically so the
    model can grow without exploding activation memory.
    """

    def __init__(
        self,
        num_classes: int,
        input_size: Tuple[int, int] = (64, 64),
        *,
        depth: int = 5,
        width_multiplier: float = 1.0,
        base_channels: int = 16,
        max_channels: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if width_multiplier <= 0:
            raise ValueError("width_multiplier must be > 0")

        layers: list[nn.Module] = []
        in_channels = 3
        out_channels = _make_divisible(base_channels * width_multiplier)

        for block_idx in range(depth):
            if block_idx > 0 and block_idx % 2 == 0:
                out_channels = min(max_channels, _make_divisible(out_channels * 2))
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            if block_idx < depth - 1 and block_idx % 2 == 1:
                layers.append(nn.MaxPool2d(2))
            in_channels = out_channels

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(in_channels, num_classes),
        )
        self.model_metadata: Dict[str, Any] = {
            "model_depth": depth,
            "model_width_multiplier": width_multiplier,
            "model_target_pam_mb": "",
            "model_estimated_pam_mb": "",
            "model_parameter_count": count_parameters(self),
            "input_size": f"{input_size[0]}x{input_size[1]}",
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def get_model(
    model_name: str,
    num_classes: int,
    input_size: Tuple[int, int] = (64, 64),
    *,
    batch_size: int = 16,
    model_depth: int = 5,
    width_multiplier: float = 1.0,
    target_pam_mb: Optional[float] = None,
    pam_device: str = "cpu",
    pam_calibration_steps: int = 12,
) -> nn.Module:
    if model_name in {"simple_cnn", "SimpleCNN"}:
        model = SimpleCNN(num_classes=num_classes, input_size=input_size)
        model.model_metadata = {
            "model_depth": 3,
            "model_width_multiplier": 1.0,
            "model_target_pam_mb": "",
            "model_estimated_pam_mb": "",
            "model_parameter_count": count_parameters(model),
            "input_size": f"{input_size[0]}x{input_size[1]}",
        }
        return model
    if model_name in {"pam_cnn", "adaptive_cnn", "scalable_cnn", "ScalableCNN"}:
        image_size = int(input_size[0])

        def build_model(width: float) -> nn.Module:
            return ScalableCNN(
                num_classes=num_classes,
                input_size=input_size,
                depth=model_depth,
                width_multiplier=width,
            )

        if target_pam_mb is not None and target_pam_mb > 0:
            from pam import calibrate_width_for_pam

            calibrated = calibrate_width_for_pam(
                build_model=build_model,
                target_pam_mb=target_pam_mb,
                batch_size=batch_size,
                image_size=image_size,
                device=pam_device,
                steps=pam_calibration_steps,
            )
            model = calibrated.model
            model.model_metadata.update(
                {
                    "model_width_multiplier": calibrated.width_multiplier,
                    "model_target_pam_mb": target_pam_mb,
                    "model_estimated_pam_mb": calibrated.estimated_pam_mb,
                    "model_parameter_count": count_parameters(model),
                }
            )
            return model

        model = build_model(width_multiplier)
        model.model_metadata.update(
            {
                "model_width_multiplier": width_multiplier,
                "model_parameter_count": count_parameters(model),
            }
        )
        return model
    raise ValueError(f"Unknown model: {model_name}")
