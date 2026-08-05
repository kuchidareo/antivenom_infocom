"""Epoch-level activation and activation-gradient entropy summaries.

Tensor references are captured at layer boundaries, but entropy is calculated
only after the complete forward/backward PMU batch has finished. This keeps the
entropy kernels outside every measured leaf-layer region and avoids writing a
large per-batch trace.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from experiment_config import CONDITION_COLUMNS


SUMMARY_COLUMNS = [
    "timestamp",
    "timestamp_unix",
    *CONDITION_COLUMNS,
    "condition",
    "round",
    "epoch",
    "phase",
    "layer_index",
    "layer_name",
    "layer_type",
    "family",
    "metric",
    "mean",
    "std",
    "batches",
]


@dataclass
class RunningStatistics:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        if not math.isfinite(value):
            raise ValueError(f"Entropy metric is not finite: {value}")
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    @property
    def std(self) -> float:
        return math.sqrt(max(self.m2, 0.0) / (self.count - 1)) if self.count > 1 else 0.0


@dataclass
class ForwardRecord:
    module: torch.nn.Module
    layer_index: int
    layer_name: str
    layer_type: str
    input_tensor: torch.Tensor
    output_tensor: torch.Tensor


def _first_tensor(value: Any) -> Optional[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        return next((item for item in value if isinstance(item, torch.Tensor)), None)
    if isinstance(value, dict):
        return next((item for item in value.values() if isinstance(item, torch.Tensor)), None)
    return None


def _pair(value: Any, *, fallback: Any = None) -> Tuple[int, int]:
    if value is None:
        value = fallback
    if isinstance(value, tuple):
        return int(value[0]), int(value[1])
    return int(value), int(value)


@torch.no_grad()
def conditional_entropy(sequence: torch.Tensor, num_states: int) -> float:
    """Return H(S_t | S_{t-1}) without transitions across sample rows."""
    if sequence.ndim != 2:
        raise ValueError(f"Expected a 2D sequence, got shape={tuple(sequence.shape)}")
    if sequence.shape[1] < 2 or sequence.numel() == 0:
        return 0.0
    sequence = sequence.long()
    previous = sequence[:, :-1].reshape(-1)
    current = sequence[:, 1:].reshape(-1)
    counts = torch.bincount(
        previous * num_states + current,
        minlength=num_states * num_states,
    ).reshape(num_states, num_states).double()
    total = counts.sum()
    if total <= 0:
        return 0.0
    row_totals = counts.sum(dim=1)
    entropy = torch.zeros((), dtype=torch.double, device=counts.device)
    for row, row_total in zip(counts, row_totals):
        if row_total > 0:
            probabilities = row[row > 0] / row_total
            entropy += (row_total / total) * (-(probabilities * probabilities.log2()).sum())
    return float(entropy.item())


@torch.no_grad()
def marginal_entropy(sequence: torch.Tensor, num_states: int) -> float:
    if sequence.ndim != 2:
        raise ValueError(f"Expected a 2D sequence, got shape={tuple(sequence.shape)}")
    if sequence.numel() == 0:
        return 0.0
    counts = torch.bincount(sequence.long().reshape(-1), minlength=num_states).double()
    probabilities = counts[counts > 0] / counts.sum()
    return float((-(probabilities * probabilities.log2()).sum()).item())


@torch.no_grad()
def zero_nonzero_rates(tensor: torch.Tensor) -> Dict[str, float]:
    contiguous = tensor.detach().contiguous()
    zero_rate = float(contiguous.eq(0).double().mean().item())
    return {
        "zero_rate": zero_rate,
        "nonzero_rate": 1.0 - zero_rate,
    }


@torch.no_grad()
def binary_mask_conditional_entropy(tensor: torch.Tensor) -> float:
    contiguous = tensor.detach().contiguous()
    mask = contiguous.ne(0)
    sequence = mask.long().reshape(mask.shape[0], -1)
    return conditional_entropy(sequence, 2)


@torch.no_grad()
def binary_mask_metrics(tensor: torch.Tensor) -> Dict[str, float]:
    return {
        "mask_conditional_entropy_bits": binary_mask_conditional_entropy(tensor),
        **zero_nonzero_rates(tensor),
    }


@torch.no_grad()
def conv_input_conditional_entropy(x: torch.Tensor, conv: torch.nn.Conv2d) -> float:
    mask = x.detach().ne(0).float()
    patches = F.unfold(
        mask,
        kernel_size=conv.kernel_size,
        dilation=conv.dilation,
        padding=conv.padding,
        stride=conv.stride,
    )
    sequence = patches.transpose(1, 2).reshape(mask.shape[0], -1)
    return conditional_entropy(sequence, 2)


@torch.no_grad()
def maxpool_winner_metrics(x: torch.Tensor, pool: torch.nn.MaxPool2d) -> Dict[str, float]:
    kernel_h, kernel_w = _pair(pool.kernel_size)
    stride_h, stride_w = _pair(pool.stride, fallback=pool.kernel_size)
    padding_h, padding_w = _pair(pool.padding)
    dilation_h, dilation_w = _pair(pool.dilation)
    _, indices = F.max_pool2d(
        x.detach(),
        kernel_size=pool.kernel_size,
        stride=pool.stride,
        padding=pool.padding,
        dilation=pool.dilation,
        ceil_mode=pool.ceil_mode,
        return_indices=True,
    )
    input_width = x.shape[-1]
    output_height, output_width = indices.shape[-2:]
    device = indices.device
    base_h = (
        torch.arange(output_height, device=device).view(1, 1, output_height, 1)
        * stride_h
        - padding_h
    )
    base_w = (
        torch.arange(output_width, device=device).view(1, 1, 1, output_width)
        * stride_w
        - padding_w
    )
    winner_h = indices // input_width
    winner_w = indices % input_width
    local_h = torch.div(winner_h - base_h, dilation_h, rounding_mode="floor")
    local_w = torch.div(winner_w - base_w, dilation_w, rounding_mode="floor")
    local_index = local_h * kernel_w + local_w
    if bool(((local_index < 0) | (local_index >= kernel_h * kernel_w)).any().item()):
        raise RuntimeError("MaxPool returned a winner outside its logical pooling window.")
    sequence = local_index.reshape(local_index.shape[0], -1)
    states = kernel_h * kernel_w
    return {
        "winner_index_entropy_bits": marginal_entropy(sequence, states),
        "winner_index_conditional_entropy_bits": conditional_entropy(sequence, states),
    }


class LayerEntropyLogger:
    """Collect online epoch summaries for Conv, ReLU, and MaxPool tensors."""

    def __init__(self, *, path: Path, condition: Dict[str, Any]) -> None:
        self.path = Path(path)
        self.condition = dict(condition)
        self._batch_context: Dict[str, Any] = {}
        self._forward: Dict[Tuple[int, int], ForwardRecord] = {}
        self._gradients: Dict[Tuple[int, int], torch.Tensor] = {}
        self._statistics: Dict[Tuple[Any, ...], RunningStatistics] = {}
        self._closed = False

    def begin_batch(self, *, epoch: int, batch_idx: int, round_id: Any = 0) -> None:
        if self._batch_context:
            raise RuntimeError("An entropy batch is already active.")
        self._batch_context = {"round": round_id, "epoch": epoch, "batch_idx": batch_idx}
        self._forward = {}
        self._gradients = {}

    def capture_forward(
        self,
        *,
        module: torch.nn.Module,
        layer_index: int,
        layer_name: str,
        layer_type: str,
        invocation_index: int,
        inputs: Any,
        output: Any,
    ) -> None:
        if not self._batch_context or not isinstance(
            module, (torch.nn.Conv2d, torch.nn.ReLU, torch.nn.MaxPool2d)
        ):
            return
        input_tensor = _first_tensor(inputs)
        output_tensor = _first_tensor(output)
        if input_tensor is None or output_tensor is None:
            return
        key = (layer_index, invocation_index)
        self._forward[key] = ForwardRecord(
            module=module,
            layer_index=layer_index,
            layer_name=layer_name,
            layer_type=layer_type,
            input_tensor=input_tensor.detach(),
            output_tensor=output_tensor.detach(),
        )

    def capture_backward(
        self,
        *,
        layer_index: int,
        invocation_index: int,
        grad_outputs: Any,
    ) -> None:
        if not self._batch_context:
            return
        gradient = _first_tensor(grad_outputs)
        if gradient is not None and (layer_index, invocation_index) in self._forward:
            self._gradients[(layer_index, invocation_index)] = gradient.detach()

    def end_batch(self) -> None:
        if not self._batch_context:
            raise RuntimeError("No entropy batch is active.")
        for key in sorted(self._forward):
            record = self._forward[key]
            family = self._family(record.module)
            output_metrics = zero_nonzero_rates(record.output_tensor)
            self._update(record, family, "forward", "output_zero_rate", output_metrics["zero_rate"])
            self._update(
                record, family, "forward", "output_nonzero_rate", output_metrics["nonzero_rate"]
            )
            if isinstance(record.module, torch.nn.ReLU):
                self._update(
                    record,
                    family,
                    "forward",
                    "mask_conditional_entropy_bits",
                    binary_mask_conditional_entropy(record.output_tensor),
                )
            elif isinstance(record.module, torch.nn.Conv2d):
                self._update(
                    record,
                    family,
                    "forward",
                    "input_mask_conditional_entropy_bits",
                    conv_input_conditional_entropy(record.input_tensor, record.module),
                )
            elif isinstance(record.module, torch.nn.MaxPool2d):
                for metric, value in maxpool_winner_metrics(
                    record.input_tensor, record.module
                ).items():
                    self._update(record, family, "forward", metric, value)

            gradient = self._gradients.get(key)
            if gradient is not None:
                gradient_metrics = binary_mask_metrics(gradient)
                self._update(
                    record,
                    family,
                    "backward",
                    "output_gradient_mask_conditional_entropy_bits",
                    gradient_metrics["mask_conditional_entropy_bits"],
                )
                self._update(
                    record,
                    family,
                    "backward",
                    "output_gradient_zero_rate",
                    gradient_metrics["zero_rate"],
                )
                self._update(
                    record,
                    family,
                    "backward",
                    "output_gradient_nonzero_rate",
                    gradient_metrics["nonzero_rate"],
                )
        self._clear_batch()

    def abort_batch(self) -> None:
        self._clear_batch()

    def close(self) -> None:
        if self._closed:
            return
        self.abort_batch()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now()
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for key in sorted(self._statistics, key=lambda value: tuple(map(str, value))):
                (
                    round_id,
                    epoch,
                    phase,
                    layer_index,
                    layer_name,
                    layer_type,
                    family,
                    metric,
                ) = key
                statistics = self._statistics[key]
                row = {
                    "timestamp": timestamp.isoformat(timespec="microseconds"),
                    "timestamp_unix": timestamp.timestamp(),
                    **self.condition,
                    "condition": self.condition.get("scenario", ""),
                    "round": round_id,
                    "epoch": epoch,
                    "phase": phase,
                    "layer_index": layer_index,
                    "layer_name": layer_name,
                    "layer_type": layer_type,
                    "family": family,
                    "metric": metric,
                    "mean": statistics.mean,
                    "std": statistics.std,
                    "batches": statistics.count,
                }
                writer.writerow(row)
        temporary.replace(self.path)
        self._closed = True

    def _update(
        self,
        record: ForwardRecord,
        family: str,
        phase: str,
        metric: str,
        value: float,
    ) -> None:
        key = (
            self._batch_context["round"],
            self._batch_context["epoch"],
            phase,
            record.layer_index,
            record.layer_name,
            record.layer_type,
            family,
            metric,
        )
        self._statistics.setdefault(key, RunningStatistics()).update(value)

    @staticmethod
    def _family(module: torch.nn.Module) -> str:
        if isinstance(module, torch.nn.Conv2d):
            return "conv"
        if isinstance(module, torch.nn.ReLU):
            return "relu"
        if isinstance(module, torch.nn.MaxPool2d):
            return "maxpool"
        raise TypeError(f"Unsupported entropy layer: {type(module).__name__}")

    def _clear_batch(self) -> None:
        self._batch_context = {}
        self._forward = {}
        self._gradients = {}
