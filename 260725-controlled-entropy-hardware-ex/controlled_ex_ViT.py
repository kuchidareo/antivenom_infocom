"""Controlled Softmax score-row entropy replay with replay-only PMU logging.

Hypothesis
----------
Repeated attention-score rows give Softmax a low-entropy computation stream:
the max-reduction comparisons, S-max operands, exp operands, sums, and outputs
repeat from row to row.  As complete score rows become less predictable, time
and hardware cost may increase.

The exact FP32 logit multiset is identical in every row and every regime.
Consequently shape, work count, row mean/variance, sorted Softmax probability
values, probability entropy, maximum probability, and top-k mass are controls.
Only complete-row repetition changes:

  low:  one permutation is repeated for every query row;
  mid:  a small bank of permutations is repeated in contiguous query blocks;
  high: every query row uses an independently generated permutation.

Only torch.softmax(logits, dim=-1) is inside the measured replay loop. Tensor
construction, validation, entropy analysis, and warm-up finish before READY.

Internal PMU measurement is enabled by default and gates counters around only
the replay. For external attachment, pass ``--disable-perf --start-delay 10``.

Example with external whole-process measurement:

  perf stat -x, -o softmax_low.csv \
    -e instructions,cycles,L1-dcache-loads,L1-dcache-load-misses,branch-misses \
    -- python controlled_vit_attention_entropy.py --regime low

Cleaner attach workflow:

  python controlled_vit_attention_entropy.py --regime low --start-delay 10 &
  pid=$!
  perf stat -x, -o softmax_low.csv -e instructions,cycles,... -p "$pid"
  wait "$pid"

Run low/mid/high as separate processes and repeat across several seeds. Replace
generic perf events with the Arm/raw PMU events supported by the target CPU.
"""

from __future__ import annotations

import argparse
import hashlib
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
    regime: str
    seed: int
    trial_id: str
    device_id: str
    batch_size: int
    grid_size: int
    heads: int
    mid_prototypes: int
    warmup: int
    repeats: int
    threads: int
    start_delay: float
    output_dir: str
    perf_profile: str
    perf_events: str
    perf_binary: str
    enable_perf: bool

    @property
    def tokens(self) -> int:
        return self.grid_size * self.grid_size + 1


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Controlled low/mid/high score-row entropy Softmax replay"
    )
    parser.add_argument("--regime", choices=("low", "mid", "middle", "high"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trial-id", default="trial_0")
    parser.add_argument("--device-id", default=os.environ.get("DEVICE_ID", socket.gethostname()))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grid-size", type=int, default=8,
                        help="8 gives 64 patch tokens plus one CLS-sized token")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--mid-prototypes", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=10_000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--start-delay", type=float, default=0.0)
    parser.add_argument("--output-dir", default="controlled_vit_softmax_results")
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
    values = vars(parser.parse_args())
    if values["regime"] == "middle":
        values["regime"] = "mid"
    if values["mid_prototypes"] < 2:
        parser.error("--mid-prototypes must be at least 2")
    for name in ("batch_size", "grid_size", "heads", "repeats", "threads"):
        if values[name] <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if values["warmup"] < 0 or values["start_delay"] < 0:
        parser.error("--warmup and --start-delay must be nonnegative")
    return Config(**values)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def entropy_from_counts(counts: torch.Tensor) -> float:
    counts = counts.double().reshape(-1)
    counts = counts[counts > 0]
    if counts.numel() == 0:
        return 0.0
    probabilities = counts / counts.sum()
    return float(-(probabilities * probabilities.log2()).sum())


def conditional_entropy(previous: torch.Tensor, current: torch.Tensor, states: int) -> float:
    encoded = previous.reshape(-1).long() * states + current.reshape(-1).long()
    counts = torch.bincount(encoded, minlength=states * states).reshape(states, states).double()
    total = counts.sum()
    result = 0.0
    for row, row_total in zip(counts, counts.sum(1)):
        if row_total > 0:
            result += float(row_total / total) * entropy_from_counts(row)
    return result


def common_logit_values(tokens: int, seed: int) -> torch.Tensor:
    """Return one exact, non-degenerate FP32 multiset shared by all rows."""
    generator = torch.Generator().manual_seed(seed + 2_000_003)
    values = torch.randn(tokens, generator=generator, dtype=torch.float32) * 0.55
    # Keep Softmax non-uniform without making it almost one-hot.
    values[0] = 2.4
    values[1] = 1.5
    values[2] = 0.9
    values = values.sort(descending=True).values.contiguous()
    if torch.unique(values).numel() != tokens:
        raise AssertionError("Common logits must be unique so value states are unambiguous")
    return values


def make_permutations(cfg: Config) -> tuple[torch.Tensor, torch.Tensor]:
    """Return row permutations [Q,K] and integer prototype IDs [Q]."""
    queries = keys = cfg.tokens
    generator = torch.Generator().manual_seed(cfg.seed + 3_000_017)

    if cfg.regime == "low":
        prototype = torch.randperm(keys, generator=generator)
        permutations = prototype.unsqueeze(0).repeat(queries, 1)
        prototype_ids = torch.zeros(queries, dtype=torch.long)
    elif cfg.regime == "mid":
        count = min(cfg.mid_prototypes, queries)
        prototypes = torch.stack([torch.randperm(keys, generator=generator) for _ in range(count)])
        block_size = math.ceil(queries / count)
        prototype_ids = (torch.arange(queries) // block_size).clamp_max(count - 1)
        permutations = prototypes[prototype_ids]
    else:
        permutations = torch.stack([torch.randperm(keys, generator=generator) for _ in range(queries)])
        prototype_ids = torch.arange(queries, dtype=torch.long)

    return permutations.contiguous(), prototype_ids


def make_logits(cfg: Config) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = common_logit_values(cfg.tokens, cfg.seed)
    permutations, prototype_ids = make_permutations(cfg)
    rows = values[permutations]  # [queries, keys]
    # Batch and head are exact replicas: only query-row structure is manipulated.
    logits = rows.view(1, 1, cfg.tokens, cfg.tokens).expand(
        cfg.batch_size, cfg.heads, -1, -1
    ).clone().contiguous()
    return logits, values, prototype_ids


def running_max_signature(rows: torch.Tensor) -> torch.Tensor:
    """Binary record-update signature, used as a max-reduction proxy."""
    cumulative = rows.cummax(dim=-1).values
    signature = torch.ones_like(rows, dtype=torch.long)
    signature[:, 1:] = (rows[:, 1:] > cumulative[:, :-1]).long()
    return signature


def analyze(logits: torch.Tensor, values: torch.Tensor, prototype_ids: torch.Tensor) -> dict[str, float | int | str]:
    rows = logits[0, 0]  # [query,key]; batch/head are deliberate replicas.
    queries, keys = rows.shape

    # Every row must contain exactly the same FP32 multiset.
    sorted_rows = rows.sort(dim=-1).values
    expected = values.sort().values.unsqueeze(0).expand_as(sorted_rows)
    if not torch.equal(sorted_rows, expected):
        raise AssertionError("A score row changed the controlled logit multiset")
    if not logits.is_contiguous():
        raise AssertionError("Softmax input must have identical contiguous layout")

    probabilities = rows.softmax(dim=-1)
    probability_entropy = -(probabilities * probabilities.clamp_min(
        torch.finfo(probabilities.dtype).tiny
    ).log2()).sum(-1)
    if not torch.allclose(probability_entropy, probability_entropy[:1].expand_as(probability_entropy),
                          rtol=1e-6, atol=1e-6):
        raise AssertionError("Row probability entropy is not controlled")

    previous, current = rows[:-1], rows[1:]
    exact_rows = (previous == current).all(-1)
    element_persistence = (previous == current).float().mean()
    relative_l2 = ((current - previous).norm(dim=-1) /
                   previous.norm(dim=-1).clamp_min(1e-12)).mean()
    cosine = torch.nn.functional.cosine_similarity(previous, current, dim=-1).mean()

    # Values are unique and shared, so their sorted rank is an exact discrete state.
    ascending = values.sort().values
    value_states = torch.searchsorted(ascending, rows)
    value_conditional = conditional_entropy(value_states[:-1], value_states[1:], keys)

    comparison = running_max_signature(rows)
    comparison_flip = (comparison[:-1] != comparison[1:]).float().mean()
    comparison_conditional = conditional_entropy(comparison[:-1], comparison[1:], 2)
    comparison_exact = (comparison[:-1] == comparison[1:]).all(-1).float().mean()

    unique_rows = torch.unique(rows, dim=0).shape[0]
    pattern_counts = torch.bincount(prototype_ids)
    control_hash = hashlib.sha256(expected[0].numpy().tobytes()).hexdigest()
    return {
        "score_row_pattern_entropy_bits": entropy_from_counts(pattern_counts),
        "score_value_conditional_entropy_bits": value_conditional,
        "adjacent_row_exact_repetition_rate": float(exact_rows.float().mean()),
        "adjacent_element_persistence_rate": float(element_persistence),
        "adjacent_row_relative_l2": float(relative_l2),
        "adjacent_row_cosine": float(cosine),
        "unique_score_rows": int(unique_rows),
        "unique_score_row_fraction": float(unique_rows / queries),
        "comparison_signature_conditional_entropy_bits": comparison_conditional,
        "comparison_signature_flip_rate": float(comparison_flip),
        "comparison_signature_exact_repetition_rate": float(comparison_exact),
        "attention_probability_entropy_bits": float(probability_entropy.mean()),
        "attention_probability_entropy_std": float(probability_entropy.std()),
        "attention_max_probability": float(probabilities.max(-1).values.mean()),
        "attention_top3_mass": float(probabilities.topk(3, dim=-1).values.sum(-1).mean()),
        "logit_mean": float(rows.mean()),
        "logit_variance": float(rows.var(unbiased=False)),
        "logit_multiset_sha256": control_hash,
        "rows": int(queries),
        "columns": int(keys),
    }


@torch.inference_mode()
def warmup(logits: torch.Tensor, count: int) -> None:
    for _ in range(count):
        torch.softmax(logits, dim=-1)


@torch.inference_mode()
def replay(logits: torch.Tensor, repeats: int) -> tuple[float, float]:
    start = time.perf_counter_ns()
    for _ in range(repeats):
        output = torch.softmax(logits, dim=-1)
    elapsed = (time.perf_counter_ns() - start) / 1e9
    return elapsed, float(output.sum())


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
        f"softmax_score_rows_{cfg.regime}_seed{cfg.seed}_{cfg.trial_id}_"
        f"b{cfg.batch_size}_g{cfg.grid_size}_h{cfg.heads}_m{cfg.mid_prototypes}"
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

    logits, values, prototype_ids = make_logits(cfg)
    metrics = analyze(logits, values, prototype_ids)
    print("RUN_ID", run_id, flush=True)
    print("STRUCTURE", json.dumps(metrics, sort_keys=True), flush=True)

    warmup(logits, cfg.warmup)

    state = TrainingState(round=0, epoch=0, batch_idx=0, phase="replay")
    condition = {
        "experiment_id": "controlled_vit_softmax_score_row_entropy",
        "run_id": run_id,
        "operator": "softmax",
        "regime": cfg.regime,
        "temporal": "stable",
        "seed": cfg.seed,
        "trial_id": cfg.trial_id,
        "device_id": cfg.device_id,
        "host": host,
        "batch_size": cfg.batch_size,
        "grid_size": cfg.grid_size,
        "tokens": cfg.tokens,
        "heads": cfg.heads,
        "mid_prototypes": cfg.mid_prototypes,
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
            elapsed, checksum = replay(logits, cfg.repeats)
        else:
            with logger.measure_phase():
                elapsed, checksum = replay(logits, cfg.repeats)
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
        "structure_metrics": metrics,
        "elapsed_seconds": elapsed,
        "nanoseconds_per_softmax": elapsed * 1e9 / cfg.repeats,
        "checksum": checksum,
        "pid": os.getpid(),
    }
    path = output_dir / f"{run_id}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT", json.dumps({
        "run_id": run_id,
        "elapsed_seconds": elapsed,
        "nanoseconds_per_softmax": result["nanoseconds_per_softmax"],
        "checksum": checksum,
        "json": str(path),
        "perf_csv": result["perf_csv"],
    }), flush=True)


if __name__ == "__main__":
    main()
