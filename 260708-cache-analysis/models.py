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


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("Attention dimension must be divisible by the number of heads.")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.last_attention: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch, tokens, 3, self.heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv.unbind(0)
        weights = ((query @ key.transpose(-2, -1)) * self.scale).softmax(-1)
        self.last_attention = weights.detach().float().cpu().contiguous()
        attended = (weights @ value).transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj(attended)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int) -> None:
        super().__init__()
        hidden = dim * mlp_ratio
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.fc2(self.gelu(self.fc1(self.norm2(x))))


class TinyViT(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        input_size: Tuple[int, int] = (32, 32),
        patch_size: int = 4,
        embed_dim: int = 128,
        heads: int = 4,
        mlp_ratio: int = 2,
        depth: int = 4,
    ) -> None:
        super().__init__()
        height, width = input_size
        if height != width:
            raise ValueError("TinyViT requires a square input.")
        if height % patch_size != 0:
            raise ValueError("TinyViT input size must be divisible by patch size.")

        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, stride=patch_size)
        token_count = (height // patch_size) ** 2 + 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, token_count, embed_dim))
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.classifier(self.norm(x)[:, 0])


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


def _build_resnet18(num_classes: int, input_size: Tuple[int, int]) -> nn.Module:
    """Build a randomly initialized torchvision ResNet18 for local training."""
    try:
        from torchvision.models import resnet18
    except ImportError as exc:
        raise RuntimeError(
            "The resnet18 model requires torchvision. Install torch and torchvision "
            "from the PyTorch CPU wheel index on Raspberry Pi."
        ) from exc

    # weights=None avoids an implicit network download and keeps every trial local.
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.model_metadata = {
        "model_depth": 18,
        "model_width_multiplier": 1.0,
        "model_target_pam_mb": "",
        "model_estimated_pam_mb": "",
        "model_parameter_count": count_parameters(model),
        "input_size": f"{input_size[0]}x{input_size[1]}",
        "pretrained": False,
    }
    return model


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
    if model_name.lower() in {"resnet18", "resnet_18"}:
        return _build_resnet18(num_classes=num_classes, input_size=input_size)
    if model_name.lower() in {"tiny_vit", "tinyvit", "vit"}:
        model = TinyViT(
            num_classes=num_classes,
            input_size=input_size,
            depth=model_depth,
        )
        model.model_metadata = {
            "model_depth": model_depth,
            "model_width_multiplier": 1.0,
            "model_target_pam_mb": "",
            "model_estimated_pam_mb": "",
            "model_parameter_count": count_parameters(model),
            "input_size": f"{input_size[0]}x{input_size[1]}",
            "patch_size": 4,
            "embed_dim": 128,
            "attention_heads": 4,
            "mlp_ratio": 2,
            "pretrained": False,
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
