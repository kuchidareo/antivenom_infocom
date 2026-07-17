"""Theoretical peak activation memory estimation and model calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn


PAM_MODULES = (
    nn.Conv2d,
    nn.BatchNorm2d,
    nn.ReLU,
    nn.ReLU6,
    nn.MaxPool2d,
    nn.AvgPool2d,
    nn.AdaptiveAvgPool2d,
    nn.Linear,
)


@dataclass(frozen=True)
class CalibrationResult:
    model: nn.Module
    width_multiplier: float
    estimated_pam_mb: float


def dtype_nbytes(dtype: torch.dtype) -> int:
    if dtype in (torch.float16, torch.bfloat16):
        return 2
    if dtype in (torch.float64, torch.int64):
        return 8
    return 4


def estimate_activation_pam_mb(
    model: nn.Module,
    batch_size: int,
    image_size: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> float:
    """Estimate training activation memory from forward hook outputs.

    This is a theoretical PAM proxy, not real process memory. It sums outputs
    from activation-producing modules during a dummy forward pass and treats
    them as saved tensors required for backward. It intentionally excludes
    parameter, optimizer, allocator, Python, dataloader, and OS memory.
    """
    device = torch.device(device)
    was_training = model.training
    model = model.to(device=device, dtype=dtype)
    model.eval()

    total_bytes = 0
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    def tensor_bytes(obj: object) -> int:
        if isinstance(obj, torch.Tensor):
            return obj.numel() * dtype_nbytes(obj.dtype)
        if isinstance(obj, (tuple, list)):
            return sum(tensor_bytes(item) for item in obj)
        if isinstance(obj, dict):
            return sum(tensor_bytes(item) for item in obj.values())
        return 0

    def hook(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
        nonlocal total_bytes
        total_bytes += tensor_bytes(output)

    for module in model.modules():
        if isinstance(module, PAM_MODULES):
            hooks.append(module.register_forward_hook(hook))

    try:
        with torch.no_grad():
            dummy = torch.zeros(batch_size, 3, image_size, image_size, device=device, dtype=dtype)
            _ = model(dummy)
    finally:
        for handle in hooks:
            handle.remove()
        model.train(was_training)

    return total_bytes / (1024.0 * 1024.0)


def calibrate_width_for_pam(
    build_model: Callable[[float], nn.Module],
    target_pam_mb: float,
    batch_size: int,
    image_size: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    min_width: float = 0.05,
    max_width: float = 16.0,
    steps: int = 18,
) -> CalibrationResult:
    """Binary-search a width multiplier close to the target theoretical PAM."""
    low = min_width
    high = min_width
    best_width = min_width
    best_pam = float("inf")

    def estimate(width: float) -> float:
        model = build_model(width)
        return estimate_activation_pam_mb(model, batch_size, image_size, dtype=dtype, device=device)

    while high < max_width:
        current = estimate(high)
        if abs(current - target_pam_mb) < abs(best_pam - target_pam_mb):
            best_width, best_pam = high, current
        if current >= target_pam_mb:
            break
        low = high
        high *= 2.0
    high = min(high, max_width)

    for _ in range(steps):
        mid = (low + high) / 2.0
        current = estimate(mid)
        if abs(current - target_pam_mb) < abs(best_pam - target_pam_mb):
            best_width, best_pam = mid, current
        if current < target_pam_mb:
            low = mid
        else:
            high = mid

    final_model = build_model(best_width)
    final_pam = estimate_activation_pam_mb(final_model, batch_size, image_size, dtype=dtype, device=device)
    return CalibrationResult(model=final_model, width_multiplier=best_width, estimated_pam_mb=final_pam)
