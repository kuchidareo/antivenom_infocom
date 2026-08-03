"""Controlled Conv entropy-chain replay with replay-only PMU logging.

One process executes one condition. Tensor generation, entropy analysis, and
warm-up finish before the internal PhasePerfLogger enables counters around the
replay. The JSON and matching ``*_perf.csv`` share the same run ID, trial ID,
and device ID.

The measured replay is either Conv2d with autograd enabled, or Conv2d followed
by detach, inference-only ReLU, and inference-only MaxPool2d. It does not
generate tensors or masks, calculate entropy, run backward, or calculate the
checksum. All conditions use the same shape, active rate, and nonzero FP32
value multiset for a given seed. Only the zero/nonzero spatial arrangement
changes across low/mid/high.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from perf_logger import (
    JETSON_PERF_EVENTS,
    RPI_PERF_EVENTS,
    X86_PERF_EVENTS,
    PhasePerfLogger,
    TrainingState,
    default_perf_events_for_host,
    resolve_perf_event_spec,
)


@dataclass(frozen=True)
class Config:
    operator: str
    chain: str
    regime: str
    seed: int
    trial_id: str
    device_id: str
    batch_size: int
    channels: int
    height: int
    width: int
    activation_rate: float
    pool_size: int
    pool_stride: int
    conv_out_channels: int
    conv_kernel_size: int
    conv_stride: int
    conv_padding: int
    bank_size: int
    warmup_bank_size: int
    warmup: int
    repeats: int
    threads: int
    start_delay: float
    output_dir: str
    perf_profile: str
    perf_events: str
    perf_binary: str
    enable_perf: bool
    perf_control_fifo: str
    perf_control_ack_fifo: str


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--operator", choices=("relu", "maxpool", "conv"), required=True
    )
    p.add_argument(
        "--chain",
        choices=(
            "relu_only",
            "maxpool_only",
            "conv_only",
            "conv_relu_pool",
            "conv_relu_pool_autograd",
        ),
        default="conv_only",
    )
    p.add_argument("--regime", choices=("low", "mid", "high"), required=True)
    # Accepted only so older launchers fail gracefully while the single-tensor
    # behavior is removed. Every run now uses the dataset-style input bank.
    p.add_argument("--temporal", choices=("stable", "changing"), help=argparse.SUPPRESS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trial-id", default="trial_0")
    p.add_argument(
        "--device-id", default=os.environ.get("DEVICE_ID", socket.gethostname())
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--height", type=int, default=32)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--activation-rate", type=float, default=0.5)
    p.add_argument("--pool-size", type=int, default=2)
    p.add_argument("--pool-stride", type=int, default=2)
    p.add_argument("--conv-out-channels", type=int, default=64)
    p.add_argument("--conv-kernel-size", type=int, default=3)
    p.add_argument("--conv-stride", type=int, default=1)
    p.add_argument("--conv-padding", type=int, default=1)
    p.add_argument("--bank-size", type=int, default=250,
                   help="Number of unique prebuilt measurement tensors")
    p.add_argument("--warmup-bank-size", type=int, default=16,
                   help="Number of tensors reserved exclusively for warm-up")
    p.add_argument("--warmup", type=int, default=None,
                   help="Default: 1000 for ReLU/MaxPool and 100 for Conv")
    p.add_argument("--repeats", type=int, default=None,
                   help="Default: 50000 for ReLU/MaxPool and 250 for Conv")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--start-delay", type=float, default=0.0,
                   help="Seconds to wait after READY before the replay starts")
    p.add_argument("--output-dir", default="controlled_entropy_results")
    p.add_argument(
        "--perf-profile", choices=("auto", "x86", "rpi", "jetson"), default="auto"
    )
    p.add_argument(
        "--perf-events", default="",
        help="Comma-separated override; empty selects the host profile",
    )
    p.add_argument("--perf-binary", default="perf")
    p.add_argument("--disable-perf", dest="enable_perf", action="store_false")
    p.add_argument(
        "--perf-control-fifo", default="",
        help="perf-record control FIFO used to enable counters around replay only",
    )
    p.add_argument(
        "--perf-control-ack-fifo", default="",
        help="Acknowledgement FIFO paired with --perf-control-fifo",
    )
    p.set_defaults(enable_perf=True)
    a = p.parse_args()
    expected_operator = {
        "relu_only": "relu",
        "maxpool_only": "maxpool",
        "conv_only": "conv",
        "conv_relu_pool": "conv",
        "conv_relu_pool_autograd": "conv",
    }[a.chain]
    if a.operator != expected_operator:
        p.error(f"chain {a.chain} requires --operator {expected_operator}")
    if not 0 < a.activation_rate < 1:
        p.error("activation-rate must be in (0, 1)")
    for name in (
        "batch_size", "channels", "height", "width", "pool_size",
        "pool_stride", "conv_out_channels", "conv_kernel_size", "conv_stride",
        "bank_size", "warmup_bank_size", "threads",
    ):
        if getattr(a, name) <= 0:
            p.error(f"{name.replace('_', '-')} must be positive")
    if a.conv_padding < 0:
        p.error("conv-padding must be nonnegative")
    if a.warmup is None:
        a.warmup = 100 if a.operator == "conv" else 1_000
    if a.repeats is None:
        a.repeats = 250 if a.operator == "conv" else 1_000
    if a.warmup < 0 or a.start_delay < 0:
        p.error("warmup and start-delay must be nonnegative")
    if a.repeats <= 0:
        p.error("repeats must be positive")
    if a.bank_size < a.repeats:
        p.error(
            "Conv requires bank-size >= repeats so every measured call uses "
            "a unique input tensor"
        )
    if a.operator == "conv":
        conv_height = (
            a.height + 2 * a.conv_padding - a.conv_kernel_size
        ) // a.conv_stride + 1
        conv_width = (
            a.width + 2 * a.conv_padding - a.conv_kernel_size
        ) // a.conv_stride + 1
        if conv_height <= 0 or conv_width <= 0:
            p.error("Conv configuration produces an empty output")
        if (
            a.chain != "conv_only"
            and (a.pool_size > conv_height or a.pool_size > conv_width)
        ):
            p.error("pool-size must fit within the Conv output height and width")
    elif a.operator == "maxpool" and (
        a.pool_size > a.height or a.pool_size > a.width
    ):
        p.error("pool-size must fit within the input height and width")
    if bool(a.perf_control_fifo) != bool(a.perf_control_ack_fifo):
        p.error("perf control and acknowledgement FIFOs must be specified together")
    if a.enable_perf and a.perf_control_fifo:
        p.error("internal perf and external perf-record control cannot both be enabled")
    values = vars(a)
    values.pop("temporal", None)
    return Config(**values)


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def smooth_scores(scores: torch.Tensor, regime: str) -> torch.Tensor:
    """Change spatial correlation length, not the number of active entries."""
    if regime == "high":
        return scores
    kernel, passes = (5, 2) if regime == "mid" else (17, 4)
    for _ in range(passes):
        scores = F.avg_pool2d(scores, kernel, stride=1, padding=kernel // 2)
    return scores


def exact_rate_mask(shape: tuple[int, int, int, int], rate: float, regime: str, seed: int) -> torch.Tensor:
    """Exactly K active coordinates per sample, with controlled clustering."""
    n, c, h, w = shape
    g = torch.Generator().manual_seed(seed)
    scores = smooth_scores(torch.randn(shape, generator=g), regime)
    flat = scores.reshape(n, -1)
    k = int(round(rate * flat.shape[1]))
    selected = flat.topk(k, dim=1, largest=True).indices
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask.scatter_(1, selected, True)
    return mask.reshape(shape)


def shared_value_multisets(shape: tuple[int, int, int, int], rate: float, seed: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return identical positive/negative FP32 multisets for every regime."""
    n, c, h, w = shape; elements = c * h * w
    k = int(round(rate * elements)); g = torch.Generator().manual_seed(seed + 9_000_001)
    positives = [torch.rand(k, generator=g).clamp_min_(torch.finfo(torch.float32).eps) for _ in range(n)]
    negatives = [-torch.rand(elements - k, generator=g).clamp_min_(torch.finfo(torch.float32).eps) for _ in range(n)]
    return positives, negatives


def materialize_tensor(
    mask: torch.Tensor,
    positives: list[torch.Tensor],
) -> torch.Tensor:
    """Place an identical positive FP32 multiset on each regime's active mask."""
    result = torch.empty(mask.shape, dtype=torch.float32)
    for sample in range(mask.shape[0]):
        flat_mask = mask[sample].reshape(-1); flat = result[sample].reshape(-1)
        flat[flat_mask] = positives[sample]
        flat[~flat_mask] = 0.0
    return result.contiguous()


def transition_metrics(sequence: torch.Tensor, states: int) -> dict[str, float]:
    previous = sequence[:, :-1].reshape(-1).long(); current = sequence[:, 1:].reshape(-1).long()
    counts = torch.bincount(previous * states + current, minlength=states * states).reshape(states, states).double()
    total = counts.sum(); row_total = counts.sum(1); entropy = 0.0
    for row, n in zip(counts, row_total):
        if n > 0:
            p = row[row > 0] / n
            entropy += float(n / total) * float(-(p * p.log2()).sum())
    same = float(counts.diag().sum() / total)
    return {"conditional_entropy_bits": entropy, "flip_rate": 1.0 - same, "same_state_rate": same,
            "transitions": int(total)}


def ordered_mask(mask: torch.Tensor, order: str) -> torch.Tensor:
    n, c, h, w = mask.shape
    if order == "width": return mask.reshape(n * c * h, w)
    if order == "height": return mask.permute(0, 1, 3, 2).contiguous().reshape(n * c * w, h)
    if order == "channel": return mask.permute(0, 2, 3, 1).contiguous().reshape(n * h * w, c)
    if order == "nchw_memory": return mask.reshape(n, c * h * w)  # never cross sample boundaries
    raise KeyError(order)


def patch_transition_metrics(mask: torch.Tensor, kernel: int = 3, stride: int = 1,
                             padding: int = 1) -> dict[str, float]:
    """Binary transitions in output-raster then C,KH,KW Conv read order."""
    n, c, h, w = mask.shape
    patches = F.unfold(mask.float(), kernel_size=kernel, stride=stride,
                       padding=padding).long().transpose(1, 2)
    return transition_metrics(patches.reshape(n, -1), 2)


def mask_metrics(mask: torch.Tensor, prefix: str) -> dict[str, float]:
    output: dict[str, float] = {
        f"{prefix}_activation_rate": float(mask.float().mean()),
        f"{prefix}_zero_fraction": float((~mask).float().mean()),
    }
    for order in ("width", "height", "channel", "nchw_memory"):
        for key, value in transition_metrics(ordered_mask(mask, order), 2).items():
            output[f"{prefix}_{order}_{key}"] = value
    for key, value in patch_transition_metrics(mask).items():
        output[f"{prefix}_conv3x3_patch_{key}"] = value
    return output


def pool_argmax_metrics(x: torch.Tensor, kernel: int, stride: int) -> dict[str, float]:
    values, indices = F.max_pool2d(x, kernel, stride, return_indices=True)
    input_width = x.shape[-1]; oh, ow = values.shape[-2:]
    base_h = torch.arange(oh).view(1, 1, oh, 1) * stride
    base_w = torch.arange(ow).view(1, 1, 1, ow) * stride
    local = (indices // input_width - base_h) * kernel + (indices % input_width - base_w)
    output = mask_metrics(values != 0, "pool_output")
    for order in ("width", "height", "channel", "nchw_memory"):
        for key, value in transition_metrics(ordered_mask(local, order), kernel * kernel).items():
            output[f"pool_argmax_{order}_{key}"] = value
    return output


def conv_patch_metrics(mask: torch.Tensor, kernel: int, stride: int,
                       padding: int) -> dict[str, float]:
    """Describe the exact binary receptive-field stream presented to Conv.

    These metrics are computed before timing. `exact_patch_*` treats the full
    CxKxK support mask as one state, whereas `patch_stream_*` treats the scalar
    zero/nonzero sequence as a first-order process.
    """
    n = mask.shape[0]
    patches = F.unfold(mask.float(), kernel_size=kernel, stride=stride,
                       padding=padding).transpose(1, 2).bool()  # [N,L,C*K*K]
    flat_patches = patches.reshape(-1, patches.shape[-1])
    active_count = flat_patches.sum(1)
    unique, counts = torch.unique(flat_patches, dim=0, return_counts=True)
    total = int(counts.sum())
    probabilities = counts.double() / total
    exact_entropy = float(-(probabilities * probabilities.log2()).sum())
    collision = (float((counts.double() * (counts.double()-1)).sum() / (total*(total-1)))
                 if total > 1 else 0.0)
    duplicate_occurrence = 1.0 - unique.shape[0] / total
    # Never compare the last output position of one image with the first of the
    # next image. This is exact full-patch repetition, not scalar similarity.
    adjacent_same = float((patches[:, 1:] == patches[:, :-1]).all(-1).float().mean())
    intersection = (patches[:, 1:] & patches[:, :-1]).sum(-1).float()
    union = (patches[:, 1:] | patches[:, :-1]).sum(-1).float()
    adjacent_jaccard = float(torch.where(union > 0, intersection / union, torch.ones_like(union)).mean())
    stream = transition_metrics(patches.reshape(n, -1), 2)
    count_values, count_counts = torch.unique(active_count, return_counts=True)
    count_p = count_counts.double() / count_counts.sum()
    active_count_entropy = float(-(count_p * count_p.log2()).sum())
    return {
        "conv_all_zero_patch_fraction": float((active_count == 0).float().mean()),
        "conv_patch_active_count_mean": float(active_count.float().mean()),
        "conv_patch_active_count_std": float(active_count.float().std(unbiased=False)),
        "conv_patch_active_count_entropy_bits": active_count_entropy,
        "conv_exact_patch_entropy_bits": exact_entropy,
        "conv_exact_patch_unique_fraction": unique.shape[0] / total,
        "conv_exact_patch_collision_probability": collision,
        "conv_duplicate_patch_occurrence_fraction": duplicate_occurrence,
        "conv_adjacent_exact_patch_same_fraction": adjacent_same,
        "conv_adjacent_patch_jaccard": adjacent_jaccard,
        **{f"conv_patch_stream_{key}": value for key, value in stream.items()},
    }


def make_bank(
    cfg: Config,
    count: int,
    *,
    seed_offset: int,
    metadata_samples: int,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    """Build a dataset-like bank whose tensors have distinct values and layouts.

    The value seed does not depend on the entropy regime. Consequently, tensor i
    in low and high runs has the same FP32 value multiset; only its spatial mask
    differs. Metadata is intentionally sampled because exact Conv patch analysis
    over every large bank item would dominate experiment preparation.
    """
    shape = (cfg.batch_size, cfg.channels, cfg.height, cfg.width)
    tensors = torch.empty((count, *shape), dtype=torch.float32)
    metadata: list[dict[str, float]] = []
    for index in range(count):
        item_seed = cfg.seed + seed_offset + 100_003 * index
        positives, _ = shared_value_multisets(
            shape, cfg.activation_rate, item_seed
        )
        mask = exact_rate_mask(shape, cfg.activation_rate, cfg.regime, item_seed)
        tensor = materialize_tensor(mask, positives)
        tensors[index].copy_(tensor)
        if index < metadata_samples:
            row = {"bank_index": index, **mask_metrics(mask, "input")}
            if cfg.operator == "relu":
                row.update(mask_metrics(F.relu(tensor) != 0, "relu_output"))
            elif cfg.operator == "maxpool":
                row.update(
                    pool_argmax_metrics(tensor, cfg.pool_size, cfg.pool_stride)
                )
            else:
                row.update(conv_patch_metrics(mask, cfg.conv_kernel_size,
                                              cfg.conv_stride, cfg.conv_padding))
            metadata.append(row)
    return tensors, metadata


def make_replay_order(count: int, seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(count, generator=generator).tolist()


def make_operator(cfg: Config):
    """Build the selected chain once; Conv uses deterministic shared weights."""
    if cfg.operator == "relu":
        return F.relu, None, {}
    if cfg.operator == "maxpool":
        operator = lambda x: F.max_pool2d(
            x, kernel_size=cfg.pool_size, stride=cfg.pool_stride
        )
        return operator, None, {}

    generator = torch.Generator().manual_seed(cfg.seed + 7_000_003)
    weight = torch.empty(cfg.conv_out_channels, cfg.channels,
                         cfg.conv_kernel_size, cfg.conv_kernel_size)
    bound = 1 / math.sqrt(cfg.channels * cfg.conv_kernel_size * cfg.conv_kernel_size)
    weight.uniform_(-bound, bound, generator=generator)
    weight = weight.contiguous()
    weight.requires_grad_(True)

    def operator(x: torch.Tensor) -> torch.Tensor:
        conv_output = F.conv2d(
            x,
            weight,
            bias=None,
            stride=cfg.conv_stride,
            padding=cfg.conv_padding,
        )
        if cfg.chain == "conv_only":
            return conv_output
        if cfg.chain == "conv_relu_pool_autograd":
            relu_output = F.relu(conv_output)
            return F.max_pool2d(
                relu_output,
                kernel_size=cfg.pool_size,
                stride=cfg.pool_stride,
            )
        detached = conv_output.detach()
        with torch.no_grad():
            relu_output = F.relu(detached)
            return F.max_pool2d(
                relu_output,
                kernel_size=cfg.pool_size,
                stride=cfg.pool_stride,
            )

    metadata = {"conv_weight_l2": float(weight.detach().norm()),
                "conv_weight_sum": float(weight.detach().sum()),
                "conv_weight_shape": list(weight.shape),
                "conv_weight_requires_grad": weight.requires_grad}
    return operator, weight, metadata


def prepare_replay_bank(bank: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Prepare gradient-requiring leaf inputs before the measured replay."""
    return tuple(tensor.detach().requires_grad_(True) for tensor in bank.unbind(0))


def validate_autograd_boundaries(
    cfg: Config,
    x: torch.Tensor,
    weight: torch.Tensor | None,
) -> dict[str, object]:
    """Assert the selected chain's autograd state before warm-up and PMU use."""
    assert x.requires_grad

    if cfg.operator == "relu":
        with torch.enable_grad():
            relu_output = F.relu(x)
        assert relu_output.requires_grad
        assert relu_output.grad_fn is not None
        metadata = {
            "chain": cfg.chain,
            "conv_autograd_enabled": False,
            "relu_autograd_enabled": True,
            "maxpool_autograd_enabled": False,
            "conv_output_shape": None,
            "relu_output_shape": list(relu_output.shape),
            "pool_output_shape": None,
            "conv_output_requires_grad": False,
            "relu_output_requires_grad": True,
            "pool_output_requires_grad": False,
        }
        del relu_output
        return metadata

    if cfg.operator == "maxpool":
        with torch.enable_grad():
            pool_output = F.max_pool2d(
                x, kernel_size=cfg.pool_size, stride=cfg.pool_stride
            )
        assert pool_output.requires_grad
        assert pool_output.grad_fn is not None
        metadata = {
            "chain": cfg.chain,
            "conv_autograd_enabled": False,
            "relu_autograd_enabled": False,
            "maxpool_autograd_enabled": True,
            "conv_output_shape": None,
            "relu_output_shape": None,
            "pool_output_shape": list(pool_output.shape),
            "conv_output_requires_grad": False,
            "relu_output_requires_grad": False,
            "pool_output_requires_grad": True,
        }
        del pool_output
        return metadata

    assert weight is not None
    assert weight.requires_grad

    with torch.enable_grad():
        conv_output = F.conv2d(
            x,
            weight,
            bias=None,
            stride=cfg.conv_stride,
            padding=cfg.conv_padding,
        )
    assert conv_output.requires_grad
    assert conv_output.grad_fn is not None

    if cfg.chain == "conv_relu_pool_autograd":
        relu_output = F.relu(conv_output)
        pool_output = F.max_pool2d(
            relu_output,
            kernel_size=cfg.pool_size,
            stride=cfg.pool_stride,
        )
        assert relu_output.requires_grad
        assert pool_output.requires_grad
        assert relu_output.grad_fn is not None
        assert pool_output.grad_fn is not None
        relu_autograd_enabled = True
        maxpool_autograd_enabled = True
    else:
        detached = conv_output.detach()
        assert not detached.requires_grad
        assert detached.grad_fn is None

        with torch.no_grad():
            relu_output = F.relu(detached)
            pool_output = F.max_pool2d(
                relu_output,
                kernel_size=cfg.pool_size,
                stride=cfg.pool_stride,
            )
        assert not relu_output.requires_grad
        assert not pool_output.requires_grad
        assert relu_output.grad_fn is None
        assert pool_output.grad_fn is None
        relu_autograd_enabled = False
        maxpool_autograd_enabled = False

    metadata = {
        "chain": cfg.chain,
        "conv_autograd_enabled": True,
        "relu_autograd_enabled": relu_autograd_enabled,
        "maxpool_autograd_enabled": maxpool_autograd_enabled,
        "conv_output_shape": list(conv_output.shape),
        "relu_output_shape": list(relu_output.shape),
        "pool_output_shape": list(pool_output.shape),
        "conv_output_requires_grad": conv_output.requires_grad,
        "relu_output_requires_grad": relu_output.requires_grad,
        "pool_output_requires_grad": pool_output.requires_grad,
    }
    del pool_output, relu_output, conv_output
    return metadata


def warmup(cfg: Config, bank, operator) -> None:
    with torch.enable_grad():
        for index in range(cfg.warmup):
            output = operator(bank[index % len(bank)])
    if cfg.warmup:
        del output


def replay(
    cfg: Config,
    bank,
    replay_order: list[int],
    operator,
) -> tuple[float, torch.Tensor]:

    # This is the target replay region. Tensor generation, entropy analysis,
    # JSON serialization, and RNG are all outside it.
    with torch.enable_grad():
        start = time.perf_counter_ns()
        for index in range(cfg.repeats):
            bank_index = replay_order[index % len(replay_order)]
            output = operator(bank[bank_index])
        elapsed = (time.perf_counter_ns() - start) / 1e9
    # Earlier iteration graphs are released when output is overwritten. Only
    # the final output survives so checksum can be calculated after PMU disable.
    return elapsed, output


def detected_host() -> str:
    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return socket.gethostname()


def is_jetson() -> bool:
    return Path("/etc/nv_tegra_release").exists() or Path(
        "/sys/devices/platform/tegra-soc"
    ).exists()


def selected_perf_events(cfg: Config, host: str) -> tuple[list[str], dict[str, str]]:
    if cfg.perf_events.strip():
        events = [item.strip() for item in cfg.perf_events.split(",") if item.strip()]
        if not events:
            raise ValueError("--perf-events did not contain an event name")
        return events, {}
    if cfg.perf_profile == "x86":
        candidates = list(X86_PERF_EVENTS)
    elif cfg.perf_profile == "rpi":
        candidates = list(RPI_PERF_EVENTS)
    elif cfg.perf_profile == "jetson":
        candidates = list(JETSON_PERF_EVENTS)
    elif is_jetson():
        candidates = list(JETSON_PERF_EVENTS)
    elif platform.machine().lower() in {"x86_64", "amd64", "i386", "i686"}:
        candidates = list(X86_PERF_EVENTS)
    else:
        candidates = default_perf_events_for_host(host)

    events: list[str] = []
    unavailable: dict[str, str] = {}
    for event in candidates:
        try:
            resolve_perf_event_spec(event, cfg.perf_binary)
        except (OSError, ValueError) as exc:
            unavailable[event] = str(exc)
        else:
            events.append(event)
    if not events:
        raise RuntimeError("No events from the selected perf profile are supported")
    return events, unavailable


def run_identifier(cfg: Config) -> str:
    run_id = (
        f"{cfg.operator}_{cfg.chain}_{cfg.regime}_dataset_seed{cfg.seed}_{cfg.trial_id}_"
        f"b{cfg.batch_size}_c{cfg.channels}_h{cfg.height}_w{cfg.width}"
    )
    if cfg.operator == "conv":
        run_id += (
            f"_oc{cfg.conv_out_channels}_k{cfg.conv_kernel_size}_"
            f"s{cfg.conv_stride}_p{cfg.conv_padding}"
        )
    return run_id


def controlled_perf_record_replay(
    cfg: Config,
    bank,
    replay_order: list[int],
    operator,
) -> tuple[float, torch.Tensor]:
    """Enable an externally launched perf-record process for replay only."""
    with open(cfg.perf_control_fifo, "w", buffering=1) as control:
        control.write("enable\n")
        with open(cfg.perf_control_ack_fifo, "r", buffering=1) as acknowledgement:
            if acknowledgement.readline().strip() != "ack":
                raise RuntimeError("perf record did not acknowledge enable")
        try:
            return replay(cfg, bank, replay_order, operator)
        finally:
            control.write("disable\n")
            with open(cfg.perf_control_ack_fifo, "r", buffering=1) as acknowledgement:
                if acknowledgement.readline().strip() != "ack":
                    raise RuntimeError("perf record did not acknowledge disable")


def main() -> None:
    cfg = parse_args()
    seed_everything(cfg.seed)
    torch.set_num_threads(cfg.threads)
    torch.set_num_interop_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", str(cfg.threads))

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_id = run_identifier(cfg)
    host = detected_host()

    bank_build_start = time.perf_counter()
    bank, entropy_rows = make_bank(
        cfg,
        cfg.bank_size,
        seed_offset=0,
        metadata_samples=min(4, cfg.bank_size),
    )
    warmup_bank, _ = make_bank(
        cfg,
        cfg.warmup_bank_size,
        seed_offset=1_000_000_007,
        metadata_samples=0,
    )
    bank_build_seconds = time.perf_counter() - bank_build_start
    replay_order = make_replay_order(cfg.bank_size, cfg.seed + 8_000_003)
    operator, weight, operator_metadata = make_operator(cfg)
    replay_bank = prepare_replay_bank(bank)
    warmup_replay_bank = prepare_replay_bank(warmup_bank)
    entropy_summary: dict[str, float] = {}
    for key in entropy_rows[0]:
        if key == "bank_index":
            continue
        values = [
            row[key] for row in entropy_rows
            if isinstance(row.get(key), (int, float))
        ]
        if values:
            entropy_summary[key] = float(np.mean(values))

    print("RUN_ID", run_id, flush=True)
    print("ENTROPY", json.dumps(entropy_summary, sort_keys=True), flush=True)
    chain_metadata = validate_autograd_boundaries(
        cfg, warmup_replay_bank[0], weight
    )
    print("AUTOGRAD_BOUNDARIES", json.dumps(chain_metadata, sort_keys=True), flush=True)
    warmup(cfg, warmup_replay_bank, operator)

    state = TrainingState(round=0, epoch=0, batch_idx=0, phase="replay")
    condition = {
        "experiment_id": "controlled_entropy_operator_chain",
        "run_id": run_id,
        "operator": cfg.operator,
        "chain": cfg.chain,
        "regime": cfg.regime,
        "temporal": "dataset_bank",
        **chain_metadata,
        "seed": cfg.seed,
        "trial_id": cfg.trial_id,
        "device_id": cfg.device_id,
        "host": host,
        "batch_size": cfg.batch_size,
        "channels": cfg.channels,
        "height": cfg.height,
        "width": cfg.width,
        "activation_rate": cfg.activation_rate,
        "pool_size": cfg.pool_size,
        "pool_stride": cfg.pool_stride,
        "conv_out_channels": cfg.conv_out_channels,
        "conv_kernel_size": cfg.conv_kernel_size,
        "conv_stride": cfg.conv_stride,
        "conv_padding": cfg.conv_padding,
        "bank_size": len(bank),
        "warmup_bank_size": len(warmup_bank),
        "bank_tensor_bytes": int(bank[0].numel() * bank.element_size()),
        "bank_working_set_bytes": int(bank.numel() * bank.element_size()),
        "unique_inputs_consumed": min(cfg.repeats, len(bank)),
        "maximum_input_reuse_count": math.ceil(cfg.repeats / len(bank)),
        "entropy_metadata_sample_count": len(entropy_rows),
        "warmup": cfg.warmup,
        "repeats": cfg.repeats,
        "threads": cfg.threads,
    }
    perf_path = out / f"{run_id}_perf.csv"
    logger = None
    events: list[str] = []
    unavailable_events: dict[str, str] = {}
    if cfg.enable_perf:
        events, unavailable_events = selected_perf_events(cfg, host)
        if unavailable_events:
            print(
                "PERF_SKIPPED_EVENTS",
                json.dumps(unavailable_events, sort_keys=True),
                flush=True,
            )
        logger = PhasePerfLogger(
            log_dir=str(out),
            condition=condition,
            training_state=state,
            events=events,
            perf_binary=cfg.perf_binary,
            path=str(perf_path),
        )
        logger.start()

    print(
        "READY",
        json.dumps({
            "pid": os.getpid(),
            "start_delay": cfg.start_delay,
            "internal_perf": cfg.enable_perf,
            "perf_events": events,
        }),
        flush=True,
    )
    if cfg.start_delay > 0:
        time.sleep(cfg.start_delay)

    try:
        if cfg.perf_control_fifo:
            elapsed, output = controlled_perf_record_replay(
                cfg, replay_bank, replay_order, operator
            )
        elif logger is None:
            elapsed, output = replay(cfg, replay_bank, replay_order, operator)
        else:
            with logger.measure_phase():
                elapsed, output = replay(cfg, replay_bank, replay_order, operator)
    finally:
        if logger is not None:
            logger.stop()

    # Checksum and graph teardown occur after external/internal PMU disable.
    checksum = float(output.detach().sum())
    del output

    result = {
        "run_id": run_id,
        "config": asdict(cfg),
        "host": host,
        "perf_events": events,
        "perf_unavailable_events": unavailable_events,
        "perf_csv": str(perf_path) if cfg.enable_perf else None,
        "entropy_mean": entropy_summary,
        "operator_metadata": operator_metadata,
        "chain_metadata": chain_metadata,
        **chain_metadata,
        "entropy_per_bank_tensor": entropy_rows,
        "condition": condition,
        "bank_build_seconds": bank_build_seconds,
        "elapsed_seconds": elapsed,
        "nanoseconds_per_call": elapsed * 1e9 / cfg.repeats,
        "checksum": checksum,
        "pid": os.getpid(),
    }
    path = out / f"{run_id}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        "RESULT",
        json.dumps({
            "run_id": run_id,
            "elapsed_seconds": elapsed,
            "nanoseconds_per_call": result["nanoseconds_per_call"],
            "checksum": checksum,
            "json": str(path),
            "perf_csv": result["perf_csv"],
        }),
        flush=True,
    )


if __name__ == "__main__":
    main()
