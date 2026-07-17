import csv
import json
import math
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
import torch

from experiment_config import CONDITION_COLUMNS


GRADIENT_STREAM_COLUMNS = [
    "timestamp",
    "timestamp_unix",
    *CONDITION_COLUMNS,
    "round",
    "epoch",
    "batch_idx",
    "phase",
    "inspect_event",
    "line_size",
    "momentum",
    "name",
    "numel",
    "p_shape",
    "p_dtype",
    "p_contig",
    "p_stride",
    "p_bytes",
    "grad_none",
    "g_dtype",
    "g_contig",
    "g_sparse",
    "g_stride",
    "g_bytes",
    "buf_exists",
    "buf_dtype",
    "buf_contig",
    "buf_stride",
    "buf_bytes",
    "uses_momentum",
    "read_streams",
    "write_streams",
    "stream_count_total",
    "sgd_read_bytes",
    "sgd_write_bytes",
    "sgd_total_bytes",
    "sgd_read_cache_lines",
    "sgd_write_cache_lines",
    "sgd_total_cache_lines",
    "stream_size_class",
    "short_stream_pressure",
    "p_cache_line_util_proxy",
    "g_cache_line_util_proxy",
    "buf_cache_line_util_proxy",
    "min_cache_line_util_proxy",
    "mean_cache_line_util_proxy",
    "layout_penalty",
    "grad_l1",
    "grad_l2",
    "grad_mean_abs",
    "grad_std_abs",
    "grad_cv_abs",
    "grad_max_abs",
    "grad_max_over_mean",
    "grad_zero_ratio",
    "grad_tiny_ratio",
    "grad_finite_all",
    "grad_positive_ratio",
    "grad_negative_ratio",
    "grad_sign_balance",
    "update_l2",
    "update_mean_abs",
    "update_max_abs",
    "update_to_param_l2",
    "value_penalty",
    "hardware_risk_raw",
    "traffic_share",
    "hardware_risk_weighted",
]


def _bytes(t: Optional[torch.Tensor]) -> int:
    if t is None:
        return 0
    return t.numel() * t.element_size()


def _cache_lines(nbytes: int, line_size: int = 64) -> int:
    return int(math.ceil(nbytes / line_size)) if nbytes > 0 else 0


def _size_class(n_cache_lines: int) -> str:
    if n_cache_lines <= 2:
        return "tiny"
    if n_cache_lines <= 16:
        return "small"
    if n_cache_lines <= 256:
        return "medium"
    return "large"


def _is_dense_stride(t: Optional[torch.Tensor]) -> Optional[bool]:
    if t is None:
        return None
    return t.is_contiguous()


def _cache_line_utilization_proxy(t: Optional[torch.Tensor], line_size: int = 64) -> Optional[float]:
    """
    Approximate how efficiently a sequential logical traversal uses cache lines.

    For contiguous tensors, assume full utilization. For non-contiguous tensors,
    use the last-dimension stride as a crude penalty. This is a proxy, not a
    hardware measurement.
    """
    if t is None:
        return None

    if t.is_sparse:
        return 0.0

    if t.is_contiguous():
        return 1.0

    if t.dim() == 0:
        return 1.0

    elem_size = t.element_size()
    elems_per_line = line_size // elem_size
    last_stride = abs(t.stride()[-1])
    if last_stride <= 1:
        return 0.75

    return max(1.0 / min(last_stride, elems_per_line), 1.0 / elems_per_line)


@torch.no_grad()
def inspect_sgd_layer_memory_metrics(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    momentum: float = 0.0,
    line_size: int = 64,
) -> pd.DataFrame:
    """
    Inspect layer-wise metrics designed around SGD optimizer-step memory streams.

    Run after loss.backward() and before optimizer.step().

    This does not measure hardware events directly. It produces hardware-aware
    proxies for SGD read/write traffic, cache-line footprint, stream pressure,
    layout, gradient/update value distribution, and traffic-weighted risk.
    """
    rows = []

    for name, p in model.named_parameters():
        g = p.grad

        state = {}
        if optimizer is not None and p in optimizer.state:
            state = optimizer.state[p]

        buf = state.get("momentum_buffer", None)

        has_grad = g is not None
        has_buf = buf is not None
        uses_momentum = has_buf or momentum != 0.0

        p_bytes = _bytes(p)
        g_bytes = _bytes(g)
        b_bytes = _bytes(buf)

        read_bytes = 0
        write_bytes = 0
        read_streams = 0
        write_streams = 0

        if has_grad:
            read_bytes += g_bytes
            read_streams += 1

            read_bytes += p_bytes
            write_bytes += p_bytes
            read_streams += 1
            write_streams += 1

            if uses_momentum:
                if has_buf:
                    read_bytes += b_bytes
                    write_bytes += b_bytes
                    read_streams += 1
                    write_streams += 1
                else:
                    write_bytes += p_bytes
                    write_streams += 1

        total_bytes = read_bytes + write_bytes
        total_cache_lines = _cache_lines(total_bytes, line_size)
        read_cache_lines = _cache_lines(read_bytes, line_size)
        write_cache_lines = _cache_lines(write_bytes, line_size)

        stream_count_total = read_streams + write_streams
        short_stream_pressure = (
            stream_count_total / math.sqrt(total_cache_lines + 1.0)
            if total_cache_lines > 0
            else 0.0
        )

        p_contig = p.is_contiguous()
        g_contig = _is_dense_stride(g)
        b_contig = _is_dense_stride(buf)

        p_util = _cache_line_utilization_proxy(p, line_size)
        g_util = _cache_line_utilization_proxy(g, line_size)
        b_util = _cache_line_utilization_proxy(buf, line_size)

        util_values = [x for x in [p_util, g_util, b_util] if x is not None]
        min_cache_line_util = min(util_values) if util_values else None
        mean_cache_line_util = sum(util_values) / len(util_values) if util_values else None

        layout_penalty = 1.0
        if not p_contig:
            layout_penalty *= 2.0
        if g is not None and not g_contig:
            layout_penalty *= 2.0
        if buf is not None and not b_contig:
            layout_penalty *= 2.0
        if g is not None and getattr(g, "is_sparse", False):
            layout_penalty *= 4.0

        if g is not None:
            gd = g.detach()
            abs_g = gd.abs()

            grad_l2 = float(torch.linalg.vector_norm(gd).item())
            grad_l1 = float(abs_g.sum().item())
            grad_mean_abs = float(abs_g.mean().item())
            grad_std_abs = float(abs_g.std(unbiased=False).item())
            grad_max_abs = float(abs_g.max().item())
            grad_zero_ratio = float((gd == 0).float().mean().item())
            grad_finite_all = bool(torch.isfinite(gd).all().item())

            if grad_mean_abs > 0:
                grad_cv_abs = grad_std_abs / grad_mean_abs
                grad_max_over_mean = grad_max_abs / grad_mean_abs
            else:
                grad_cv_abs = float("inf")
                grad_max_over_mean = float("inf")

            if gd.dtype == torch.float32:
                grad_tiny_ratio = float(((gd != 0) & (abs_g < 1e-38)).float().mean().item())
            else:
                grad_tiny_ratio = None

            grad_positive_ratio = float((gd > 0).float().mean().item())
            grad_negative_ratio = float((gd < 0).float().mean().item())
            grad_sign_balance = abs(grad_positive_ratio - grad_negative_ratio)
        else:
            grad_l2 = None
            grad_l1 = None
            grad_mean_abs = None
            grad_std_abs = None
            grad_max_abs = None
            grad_zero_ratio = None
            grad_finite_all = None
            grad_cv_abs = None
            grad_max_over_mean = None
            grad_tiny_ratio = None
            grad_positive_ratio = None
            grad_negative_ratio = None
            grad_sign_balance = None

        update_l2 = None
        update_mean_abs = None
        update_max_abs = None
        update_to_param_l2 = None

        if g is not None:
            if uses_momentum and buf is not None:
                upd = momentum * buf.detach() + g.detach()
            else:
                upd = g.detach()

            abs_u = upd.abs()
            update_l2 = float(torch.linalg.vector_norm(upd).item())
            update_mean_abs = float(abs_u.mean().item())
            update_max_abs = float(abs_u.max().item())

            p_l2 = float(torch.linalg.vector_norm(p.detach()).item())
            update_to_param_l2 = update_l2 / (p_l2 + 1e-12)

        value_penalty = 1.0
        if grad_tiny_ratio is not None:
            value_penalty += min(grad_tiny_ratio, 1.0)
        if grad_zero_ratio is not None:
            value_penalty += 0.1 * grad_zero_ratio
        if grad_finite_all is False:
            value_penalty *= 10.0

        hardware_risk_raw = (
            layout_penalty
            * (1.0 + short_stream_pressure)
            * value_penalty
            * max(stream_count_total, 1)
        )

        rows.append(
            {
                "name": name,
                "numel": p.numel(),
                "p_shape": tuple(p.shape),
                "p_dtype": str(p.dtype),
                "p_contig": p_contig,
                "p_stride": tuple(p.stride()),
                "p_bytes": p_bytes,
                "grad_none": g is None,
                "g_dtype": None if g is None else str(g.dtype),
                "g_contig": None if g is None else g_contig,
                "g_sparse": None if g is None else g.is_sparse,
                "g_stride": None if g is None else tuple(g.stride()),
                "g_bytes": g_bytes,
                "buf_exists": buf is not None,
                "buf_dtype": None if buf is None else str(buf.dtype),
                "buf_contig": None if buf is None else b_contig,
                "buf_stride": None if buf is None else tuple(buf.stride()),
                "buf_bytes": b_bytes,
                "uses_momentum": uses_momentum,
                "read_streams": read_streams,
                "write_streams": write_streams,
                "stream_count_total": stream_count_total,
                "sgd_read_bytes": read_bytes,
                "sgd_write_bytes": write_bytes,
                "sgd_total_bytes": total_bytes,
                "sgd_read_cache_lines": read_cache_lines,
                "sgd_write_cache_lines": write_cache_lines,
                "sgd_total_cache_lines": total_cache_lines,
                "stream_size_class": _size_class(total_cache_lines),
                "short_stream_pressure": short_stream_pressure,
                "p_cache_line_util_proxy": p_util,
                "g_cache_line_util_proxy": g_util,
                "buf_cache_line_util_proxy": b_util,
                "min_cache_line_util_proxy": min_cache_line_util,
                "mean_cache_line_util_proxy": mean_cache_line_util,
                "layout_penalty": layout_penalty,
                "grad_l1": grad_l1,
                "grad_l2": grad_l2,
                "grad_mean_abs": grad_mean_abs,
                "grad_std_abs": grad_std_abs,
                "grad_cv_abs": grad_cv_abs,
                "grad_max_abs": grad_max_abs,
                "grad_max_over_mean": grad_max_over_mean,
                "grad_zero_ratio": grad_zero_ratio,
                "grad_tiny_ratio": grad_tiny_ratio,
                "grad_finite_all": grad_finite_all,
                "grad_positive_ratio": grad_positive_ratio,
                "grad_negative_ratio": grad_negative_ratio,
                "grad_sign_balance": grad_sign_balance,
                "update_l2": update_l2,
                "update_mean_abs": update_mean_abs,
                "update_max_abs": update_max_abs,
                "update_to_param_l2": update_to_param_l2,
                "value_penalty": value_penalty,
                "hardware_risk_raw": hardware_risk_raw,
            }
        )

    df = pd.DataFrame(rows)
    total_bytes = df["sgd_total_bytes"].sum()
    if total_bytes > 0:
        df["traffic_share"] = df["sgd_total_bytes"] / total_bytes
    else:
        df["traffic_share"] = 0.0

    df["hardware_risk_weighted"] = df["hardware_risk_raw"] * df["traffic_share"]
    return df


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, tuple):
        return json.dumps(list(value))
    if isinstance(value, list):
        return json.dumps(value)
    return value


class GradientStreamLogger:
    def __init__(
        self,
        *,
        path: Union[str, Path],
        condition: Dict[str, Any],
        line_size: int = 64,
        momentum: float = 0.9,
    ) -> None:
        self.path = Path(path)
        self.condition = dict(condition)
        self.line_size = line_size
        self.momentum = momentum
        self._lock = threading.Lock()
        self._initialized = False

    def write(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        state: Dict[str, Any],
        inspect_event: str,
    ) -> None:
        now = datetime.now()
        df = inspect_sgd_layer_memory_metrics(
            model,
            optimizer=optimizer,
            momentum=self.momentum,
            line_size=self.line_size,
        )
        rows = df.to_dict("records")
        for row in rows:
            row.update(
                {
                    "timestamp": now.isoformat(timespec="microseconds"),
                    "timestamp_unix": now.timestamp(),
                    "inspect_event": inspect_event,
                    "line_size": self.line_size,
                    "momentum": self.momentum,
                }
            )
            row.update(self.condition)
            row.update(state)
            for key, value in list(row.items()):
                row[key] = _csv_value(value)
            for column in GRADIENT_STREAM_COLUMNS:
                row.setdefault(column, "")

        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if self._initialized or self.path.exists() else "w"
            with self.path.open(mode, newline="") as f:
                writer = csv.DictWriter(f, fieldnames=GRADIENT_STREAM_COLUMNS, extrasaction="ignore")
                if mode == "w":
                    writer.writeheader()
                writer.writerows(rows)
            self._initialized = True
