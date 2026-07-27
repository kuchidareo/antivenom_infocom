"""Controlled ReLU/MaxPool entropy replay with replay-only PMU logging."""

from __future__ import annotations

import argparse
import json
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
    regime: str
    temporal: str
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
    bank_size: int
    warmup: int
    repeats: int
    threads: int
    start_delay: float
    output_dir: str
    perf_profile: str
    perf_events: str
    perf_binary: str
    enable_perf: bool


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", choices=("relu", "maxpool"), required=True)
    parser.add_argument("--regime", choices=("low", "mid", "high"), required=True)
    parser.add_argument("--temporal", choices=("stable", "changing"), default="stable")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trial-id", default="trial_0")
    parser.add_argument("--device-id", default=os.environ.get("DEVICE_ID", socket.gethostname()))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--activation-rate", type=float, default=0.5)
    parser.add_argument("--pool-size", type=int, default=2)
    parser.add_argument("--pool-stride", type=int, default=2)
    parser.add_argument(
        "--bank-size", type=int, default=16,
        help="Number of prebuilt tensors used only in changing mode",
    )
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument("--repeats", type=int, default=50_000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--start-delay", type=float, default=0.0,
        help="Seconds to wait after READY before the replay starts",
    )
    parser.add_argument("--output-dir", default="controlled_entropy_results")
    parser.add_argument(
        "--perf-profile", choices=("auto", "x86", "rpi", "jetson"), default="auto",
    )
    parser.add_argument(
        "--perf-events", default="",
        help="Comma-separated override; empty selects the host profile",
    )
    parser.add_argument("--perf-binary", default="perf")
    parser.add_argument("--disable-perf", dest="enable_perf", action="store_false")
    parser.set_defaults(enable_perf=True)
    args = parser.parse_args()

    if not 0 < args.activation_rate < 1:
        parser.error("activation-rate must be in (0, 1)")
    for name in (
        "batch_size", "channels", "height", "width", "pool_size",
        "pool_stride", "bank_size", "repeats", "threads",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"{name.replace('_', '-')} must be positive")
    if args.warmup < 0 or args.start_delay < 0:
        parser.error("warmup and start-delay must be nonnegative")
    if args.operator == "maxpool" and (
        args.pool_size > args.height or args.pool_size > args.width
    ):
        parser.error("pool-size must fit within the input height and width")
    return Config(**vars(args))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def smooth_scores(scores: torch.Tensor, regime: str) -> torch.Tensor:
    """Change spatial correlation length, not the active-entry count."""
    if regime == "high":
        return scores
    kernel, passes = (5, 2) if regime == "mid" else (17, 4)
    for _ in range(passes):
        scores = F.avg_pool2d(scores, kernel, stride=1, padding=kernel // 2)
    return scores


def exact_rate_mask(
    shape: tuple[int, int, int, int], rate: float, regime: str, seed: int
) -> torch.Tensor:
    """Select exactly K active coordinates per sample."""
    generator = torch.Generator().manual_seed(seed)
    scores = smooth_scores(torch.randn(shape, generator=generator), regime)
    flat = scores.reshape(shape[0], -1)
    active_count = int(round(rate * flat.shape[1]))
    selected = flat.topk(active_count, dim=1, largest=True).indices
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask.scatter_(1, selected, True)
    return mask.reshape(shape)


def shared_value_multisets(
    shape: tuple[int, int, int, int], rate: float, seed: int
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return identical positive/negative FP32 multisets for every regime."""
    samples, channels, height, width = shape
    elements = channels * height * width
    active_count = int(round(rate * elements))
    generator = torch.Generator().manual_seed(seed + 9_000_001)
    epsilon = torch.finfo(torch.float32).eps
    positives = [
        torch.rand(active_count, generator=generator).clamp_min_(epsilon)
        for _ in range(samples)
    ]
    negatives = [
        -torch.rand(elements - active_count, generator=generator).clamp_min_(epsilon)
        for _ in range(samples)
    ]
    return positives, negatives


def materialize_tensor(
    mask: torch.Tensor,
    positives: list[torch.Tensor],
    negatives: list[torch.Tensor],
    operator: str,
) -> torch.Tensor:
    """ReLU gets signed preactivations; MaxPool gets nonnegative input."""
    result = torch.empty(mask.shape, dtype=torch.float32)
    for sample in range(mask.shape[0]):
        flat_mask = mask[sample].reshape(-1)
        flat = result[sample].reshape(-1)
        flat[flat_mask] = positives[sample]
        flat[~flat_mask] = negatives[sample] if operator == "relu" else 0.0
    return result.contiguous()


def transition_metrics(sequence: torch.Tensor, states: int) -> dict[str, float]:
    previous = sequence[:, :-1].reshape(-1).long()
    current = sequence[:, 1:].reshape(-1).long()
    counts = torch.bincount(
        previous * states + current, minlength=states * states
    ).reshape(states, states).double()
    total = counts.sum()
    row_total = counts.sum(1)
    entropy = 0.0
    for row, count in zip(counts, row_total):
        if count > 0:
            probability = row[row > 0] / count
            entropy += float(count / total) * float(
                -(probability * probability.log2()).sum()
            )
    same = float(counts.diag().sum() / total)
    return {
        "conditional_entropy_bits": entropy,
        "flip_rate": 1.0 - same,
        "same_state_rate": same,
        "transitions": int(total),
    }


def ordered_mask(mask: torch.Tensor, order: str) -> torch.Tensor:
    samples, channels, height, width = mask.shape
    if order == "width":
        return mask.reshape(samples * channels * height, width)
    if order == "height":
        return mask.permute(0, 1, 3, 2).contiguous().reshape(
            samples * channels * width, height
        )
    if order == "channel":
        return mask.permute(0, 2, 3, 1).contiguous().reshape(
            samples * height * width, channels
        )
    if order == "nchw_memory":
        return mask.reshape(samples, channels * height * width)
    raise KeyError(order)


def patch_transition_metrics(mask: torch.Tensor) -> dict[str, float]:
    """Binary transitions in canonical C,KH,KW order for all 3x3 patches."""
    samples = mask.shape[0]
    patches = F.unfold(mask.float(), kernel_size=3, padding=1).long().transpose(1, 2)
    return transition_metrics(patches.reshape(samples, -1), 2)


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
    input_width = x.shape[-1]
    output_height, output_width = values.shape[-2:]
    base_height = torch.arange(output_height).view(1, 1, output_height, 1) * stride
    base_width = torch.arange(output_width).view(1, 1, 1, output_width) * stride
    local = (
        (indices // input_width - base_height) * kernel
        + (indices % input_width - base_width)
    )
    output = mask_metrics(values != 0, "pool_output")
    for order in ("width", "height", "channel", "nchw_memory"):
        for key, value in transition_metrics(
            ordered_mask(local, order), kernel * kernel
        ).items():
            output[f"pool_argmax_{order}_{key}"] = value
    return output


def make_bank(cfg: Config) -> tuple[list[torch.Tensor], list[dict[str, float]]]:
    shape = (cfg.batch_size, cfg.channels, cfg.height, cfg.width)
    positives, negatives = shared_value_multisets(shape, cfg.activation_rate, cfg.seed)
    count = 1 if cfg.temporal == "stable" else cfg.bank_size
    tensors: list[torch.Tensor] = []
    metadata: list[dict[str, float]] = []
    for index in range(count):
        mask = exact_rate_mask(
            shape, cfg.activation_rate, cfg.regime, cfg.seed + 100_003 * index
        )
        tensor = materialize_tensor(mask, positives, negatives, cfg.operator)
        row = {"bank_index": index, **mask_metrics(mask, "input")}
        if cfg.operator == "relu":
            row.update(mask_metrics(F.relu(tensor) != 0, "relu_output"))
        else:
            row.update(pool_argmax_metrics(tensor, cfg.pool_size, cfg.pool_stride))
        tensors.append(tensor)
        metadata.append(row)
    return tensors, metadata


def operator_function(cfg: Config):
    if cfg.operator == "relu":
        return F.relu
    return lambda x: F.max_pool2d(x, cfg.pool_size, cfg.pool_stride)


@torch.inference_mode()
def warmup(cfg: Config, bank: list[torch.Tensor]) -> None:
    operator = operator_function(cfg)
    for index in range(cfg.warmup):
        operator(bank[index % len(bank)])


@torch.inference_mode()
def replay(cfg: Config, bank: list[torch.Tensor]) -> tuple[float, float]:
    operator = operator_function(cfg)
    start = time.perf_counter_ns()
    for index in range(cfg.repeats):
        output = operator(bank[index % len(bank)])
    elapsed = (time.perf_counter_ns() - start) / 1e9
    checksum = float(output.sum())
    return elapsed, checksum


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
    return (
        f"{cfg.operator}_{cfg.regime}_{cfg.temporal}_seed{cfg.seed}_{cfg.trial_id}_"
        f"b{cfg.batch_size}_c{cfg.channels}_h{cfg.height}_w{cfg.width}"
    )


def main() -> None:
    cfg = parse_args()
    seed_everything(cfg.seed)
    torch.set_num_threads(cfg.threads)
    torch.set_num_interop_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", str(cfg.threads))

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_identifier(cfg)
    host = detected_host()

    bank, entropy_rows = make_bank(cfg)
    entropy_summary: dict[str, float] = {}
    for key in entropy_rows[0]:
        values = [
            row[key] for row in entropy_rows
            if isinstance(row.get(key), (int, float))
        ]
        if values:
            entropy_summary[key] = float(np.mean(values))

    print("RUN_ID", run_id, flush=True)
    print("ENTROPY", json.dumps(entropy_summary, sort_keys=True), flush=True)
    warmup(cfg, bank)

    state = TrainingState(round=0, epoch=0, batch_idx=0, phase="replay")
    condition = {
        "experiment_id": "controlled_relu_maxpool_entropy",
        "run_id": run_id,
        "operator": cfg.operator,
        "regime": cfg.regime,
        "temporal": cfg.temporal,
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
        "bank_size": len(bank),
        "warmup": cfg.warmup,
        "repeats": cfg.repeats,
        "threads": cfg.threads,
    }
    perf_path = output_dir / f"{run_id}_perf.csv"
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
            log_dir=str(output_dir),
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
        if logger is None:
            elapsed, checksum = replay(cfg, bank)
        else:
            with logger.measure_phase():
                elapsed, checksum = replay(cfg, bank)
    finally:
        if logger is not None:
            logger.stop()

    result = {
        "run_id": run_id,
        "config": asdict(cfg),
        "host": host,
        "perf_events": events,
        "perf_unavailable_events": unavailable_events,
        "perf_csv": str(perf_path) if cfg.enable_perf else None,
        "entropy_mean": entropy_summary,
        "entropy_per_bank_tensor": entropy_rows,
        "elapsed_seconds": elapsed,
        "nanoseconds_per_call": elapsed * 1e9 / cfg.repeats,
        "checksum": checksum,
        "pid": os.getpid(),
    }
    result_path = output_dir / f"{run_id}.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        "RESULT",
        json.dumps({
            "run_id": run_id,
            "elapsed_seconds": elapsed,
            "nanoseconds_per_call": result["nanoseconds_per_call"],
            "checksum": checksum,
            "json": str(result_path),
            "perf_csv": result["perf_csv"],
        }),
        flush=True,
    )


if __name__ == "__main__":
    main()
