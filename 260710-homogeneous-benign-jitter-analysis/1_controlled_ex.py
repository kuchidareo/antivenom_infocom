#!/usr/bin/env python3
"""
controlled_perf_conv.py

One-file controlled PyTorch CPU backward microbenchmark.

It:
  1. Discovers usable perf events on the current Linux CPU.
  2. Runs controlled conv2d backward benchmarks where only grad_output structure changes.
  3. Records perf counters into a long-form CSV.
  4. Creates summary CSVs and a multi-page PDF/PNG visualizations.

Main comparisons:
  - random vs rank1
  - classwise vs shuffled
  - backward-data ("input") vs backward-weights ("weight")
  - interaction with CPU thread count

Example:
  python controlled_perf_conv.py all --threads 1,2,4 --repeats 15 --cpu-list 0-3

Outputs:
  perf_results/raw_perf.csv
  perf_results/run_metadata.csv
  perf_results/summary.csv
  perf_results/comparisons.csv
  perf_results/perf_report.pdf
  perf_results/*.png

Requirements:
  Linux, perf, Python 3.10+, torch, pandas, matplotlib
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

# Keep imports out of controller startup where possible.
CONDITIONS = ("random", "classwise", "rank1", "shuffled")
TARGETS = ("input", "weight")
CORE_EVENTS = ("cycles", "instructions")

# Candidate names are deliberately conservative. perf aliases differ by CPU/PMU.
# The first working candidate in each logical family is selected.
EVENT_FAMILIES: dict[str, tuple[str, ...]] = {
    # Core execution counters
    "cycles": (
        "cycles",
        "cpu-cycles",
    ),
    "instructions": (
        "instructions",
    ),

    # L1D traffic
    "l1d_read_access": (
        "L1-dcache-loads",
        "cpu/L1-dcache-loads/",
    ),
    "l1d_read_miss": (
        "L1-dcache-load-misses",
        "cpu/L1-dcache-load-misses/",
    ),
    "l1d_write_access": (
        "L1-dcache-stores",
        "cpu/L1-dcache-stores/",
    ),

    # L1D replacement and miss pressure
    "l1d_replacement": (
        "l1d.replacement",
    ),
    "l1d_pending": (
        "l1d_pend_miss.pending",
    ),
    "l1d_pending_cycles": (
        "l1d_pend_miss.pending_cycles",
    ),
    "l1d_fb_full": (
        "l1d_pend_miss.fb_full",
    ),
    "l1d_fb_full_periods": (
        "l1d_pend_miss.fb_full_periods",
    ),
    "l1d_l2_stall": (
        "l1d_pend_miss.l2_stall",
    ),

    # L2 demand-read traffic
    "l2d_read_access": (
        "l2_rqsts.all_demand_data_rd",
        "l2_rqsts.references",
    ),
    "l2d_read_hit": (
        "l2_rqsts.demand_data_rd_hit",
    ),
    "l2d_read_miss": (
        "l2_rqsts.demand_data_rd_miss",
        "l2_rqsts.miss",
    ),

    # L2 write-side traffic: RFO
    "l2d_rfo_access": (
        "l2_rqsts.all_rfo",
    ),
    "l2d_rfo_hit": (
        "l2_rqsts.rfo_hit",
    ),
    "l2d_rfo_miss": (
        "l2_rqsts.rfo_miss",
    ),

    # L2 writeback and line movement
    "l2d_writeback": (
        "l2_trans.l2_wb",
        "l2_lines_out.non_silent",
    ),
    "l2d_lines_in": (
        "l2_lines_in.all",
    ),
    "l2d_lines_out_dirty": (
        "l2_lines_out.non_silent",
    ),
    "l2d_lines_out_clean": (
        "l2_lines_out.silent",
    ),

    # LLC supplementary counters
    "llc_read_access": (
        "LLC-loads",
        "cpu/LLC-loads/",
    ),
    "llc_read_miss": (
        "LLC-load-misses",
        "cpu/LLC-load-misses/",
    ),
    "llc_write_access": (
        "LLC-stores",
        "cpu/LLC-stores/",
    ),
    "llc_write_miss": (
        "LLC-store-misses",
        "cpu/LLC-store-misses/",
    ),

    # Optional bus counters
    "bus_read_access": (
        "bus_access_rd",
        "bus-access-rd",
    ),
    "bus_write_access": (
        "bus_access_wr",
        "bus-access-wr",
    ),
    "bus_access": (
        "bus_access",
        "bus-access",
    ),
    "bus_cycles": (
        "bus-cycles",
    ),
}

DISPLAY_NAMES = {
    "cycles": "Cycles",
    "instructions": "Instructions",

    "l1d_read_access": "L1D read accesses",
    "l1d_read_miss": "L1D read misses",
    "l1d_write_access": "L1D write accesses",
    "l1d_replacement": "L1D replacements",
    "l1d_pending": "Outstanding L1D miss occupancy",
    "l1d_pending_cycles": "Cycles with outstanding L1D misses",
    "l1d_fb_full": "Cycles stalled by full L1D fill buffers",
    "l1d_fb_full_periods": "L1D fill-buffer-full periods",
    "l1d_l2_stall": "Cycles L1D waited for L2 resources",

    "l2d_read_access": "L2 demand data reads",
    "l2d_read_hit": "L2 demand data read hits",
    "l2d_read_miss": "L2 demand data read misses",

    "l2d_rfo_access": "L2 RFO requests",
    "l2d_rfo_hit": "L2 RFO hits",
    "l2d_rfo_miss": "L2 RFO misses",

    "l2d_writeback": "L2 writebacks",
    "l2d_lines_in": "L2 lines filled",
    "l2d_lines_out_dirty": "Dirty L2 lines evicted",
    "l2d_lines_out_clean": "Clean L2 lines evicted",

    "llc_read_access": "LLC reads",
    "llc_read_miss": "LLC read misses",
    "llc_write_access": "LLC writes",
    "llc_write_miss": "LLC write misses",

    "bus_read_access": "Bus read accesses",
    "bus_write_access": "Bus write accesses",
    "bus_access": "Bus accesses",
    "bus_cycles": "Bus cycles",
}

@dataclass(frozen=True)
class BenchConfig:
    batch: int = 16
    in_channels: int = 32
    out_channels: int = 32
    height: int = 56
    width: int = 56
    kernel_size: int = 3
    padding: int = 1
    classes: int = 4
    warmup: int = 15
    iterations: int = 80
    seed: int = 20260710


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr, flush=True)


def run_checked(
    command: Sequence[str],
    *,
    env: dict[str, str] | None = None,
    capture_output: bool = True,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        env=env,
        text=True,
        capture_output=capture_output,
        check=True,
        timeout=timeout,
    )


def perf_list_text() -> str:
    result = subprocess.run(
        ["perf", "list"],
        text=True,
        capture_output=True,
        check=False,
    )
    return (result.stdout or "") + "\n" + (result.stderr or "")


def event_mentioned(perf_text: str, event: str) -> bool:
    # perf list formatting and case differ by architecture/version.
    pattern = rf"(?<![\w.-]){re.escape(event)}(?![\w.-])"
    return re.search(pattern, perf_text, flags=re.IGNORECASE) is not None


def probe_event(event: str) -> bool:
    """Actually ask perf to open the event; perf list alone can be misleading."""
    command = [
        "perf", "stat",
        "-e", event,
        "--",
        "true",
    ]
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    text = (result.stdout or "") + (result.stderr or "")
    bad_markers = (
        "not supported",
        "unknown event",
        "event syntax error",
        "<not supported>",
        "no permission",
        "permission denied",
    )
    return result.returncode == 0 and not any(
        marker in text.lower() for marker in bad_markers
    )


def discover_events() -> tuple[dict[str, str], dict[str, str]]:
    text = perf_list_text()
    selected: dict[str, str] = {}
    unavailable: dict[str, str] = {}

    for logical_name, candidates in EVENT_FAMILIES.items():
        found = None

        # Prefer events explicitly reported by perf list.
        ordered = [
            *[e for e in candidates if event_mentioned(text, e)],
            *[e for e in candidates if not event_mentioned(text, e)],
        ]
        seen: set[str] = set()

        for event in ordered:
            if event in seen:
                continue
            seen.add(event)
            if probe_event(event):
                found = event
                break

        if found is not None:
            selected[logical_name] = found
        else:
            unavailable[logical_name] = "|".join(candidates)

    return selected, unavailable


def parse_numeric(value: str) -> float | None:
    cleaned = value.strip().replace(",", "")
    if not cleaned:
        return None
    if cleaned.startswith("<"):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_perf_file(path: Path) -> list[dict[str, object]]:
    """
    Parse perf stat -x ';' output across perf versions.

    Typical fields:
      value ; unit ; event ; runtime ; running_pct ; metric_value ; metric_unit
    """
    rows: list[dict[str, object]] = []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            fields = [field.strip() for field in line.split(";")]
            if len(fields) < 3:
                continue

            value = parse_numeric(fields[0])
            unit = fields[1] if len(fields) > 1 else ""
            event = fields[2] if len(fields) > 2 else ""
            runtime_ns = parse_numeric(fields[3]) if len(fields) > 3 else None
            running_pct = parse_numeric(fields[4]) if len(fields) > 4 else None

            # Ignore elapsed-time footer and non-event lines.
            if not event or "seconds time elapsed" in event:
                continue

            rows.append({
                "perf_event": event,
                "value": value,
                "unit": unit,
                "runtime_ns": runtime_ns,
                "running_pct": running_pct,
            })

    return rows


def normalize_event_name(name: str) -> str:
    # Remove privilege suffixes such as :u and event modifiers.
    return name.strip().split(":")[0].lower()


def map_actual_to_logical(
    actual_event: str,
    selected: dict[str, str],
) -> str:
    actual_norm = normalize_event_name(actual_event)
    for logical, candidate in selected.items():
        if normalize_event_name(candidate) == actual_norm:
            return logical
    return actual_event


def event_passes(
    selected: dict[str, str],
    max_events_per_pass: int,
) -> list[list[str]]:
    """
    cycles+instructions are included in every pass.
    Extra PMU events are split to reduce multiplexing.
    """
    core = [selected[name] for name in CORE_EVENTS if name in selected]
    extras = [
        event
        for logical, event in selected.items()
        if logical not in CORE_EVENTS
    ]

    if not extras:
        return [core]

    room = max(1, max_events_per_pass - len(core))
    passes: list[list[str]] = []
    for start in range(0, len(extras), room):
        passes.append(core + extras[start:start + room])
    return passes


def parse_worker_json(stdout: str) -> dict[str, object]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError(f"Worker did not emit JSON. Output:\n{stdout[-2000:]}")


def make_worker_command(
    script_path: Path,
    condition: str,
    target: str,
    threads: int,
    cfg: BenchConfig,
) -> list[str]:
    return [
        sys.executable,
        str(script_path),
        "worker",
        "--condition", condition,
        "--target", target,
        "--threads", str(threads),
        "--batch", str(cfg.batch),
        "--in-channels", str(cfg.in_channels),
        "--out-channels", str(cfg.out_channels),
        "--height", str(cfg.height),
        "--width", str(cfg.width),
        "--kernel-size", str(cfg.kernel_size),
        "--padding", str(cfg.padding),
        "--classes", str(cfg.classes),
        "--warmup", str(cfg.warmup),
        "--iterations", str(cfg.iterations),
        "--seed", str(cfg.seed),
    ]


def make_env(threads: int) -> dict[str, str]:
    env = os.environ.copy()
    thread_text = str(threads)
    env.update({
        "OMP_NUM_THREADS": thread_text,
        "MKL_NUM_THREADS": thread_text,
        "OPENBLAS_NUM_THREADS": thread_text,
        "NUMEXPR_NUM_THREADS": thread_text,
        "VECLIB_MAXIMUM_THREADS": thread_text,
        # Helpful for reproducibility where GNU OpenMP is in use.
        "OMP_DYNAMIC": "FALSE",
    })
    return env


def taskset_prefix(cpu_list: str | None) -> list[str]:
    if not cpu_list:
        return []
    return ["taskset", "-c", cpu_list]


def append_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = list(rows[0].keys())
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def controller(args: argparse.Namespace) -> None:
    if shutil.which("perf") is None:
        raise SystemExit("perf was not found in PATH.")
    if args.cpu_list and shutil.which("taskset") is None:
        raise SystemExit("taskset was requested but was not found in PATH.")

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    raw_csv = outdir / "raw_perf.csv"
    metadata_csv = outdir / "run_metadata.csv"

    if args.overwrite:
        for path in (raw_csv, metadata_csv):
            if path.exists():
                path.unlink()

    selected, unavailable = discover_events()

    print("Selected perf events:")
    for logical, actual in selected.items():
        print(f"  {logical:20s} -> {actual}")
    if unavailable:
        print("\nUnavailable event families:")
        for logical in unavailable:
            print(f"  {logical}")

    if "cycles" not in selected or "instructions" not in selected:
        raise SystemExit("cycles and instructions must be available.")

    passes = event_passes(selected, args.max_events_per_pass)
    print(f"\nUsing {len(passes)} perf pass(es) per trial.")

    threads = [int(item) for item in args.threads.split(",") if item.strip()]
    conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]
    targets = [x.strip() for x in args.targets.split(",") if x.strip()]

    unknown_conditions = sorted(set(conditions) - set(CONDITIONS))
    unknown_targets = sorted(set(targets) - set(TARGETS))
    if unknown_conditions:
        raise SystemExit(f"Unknown conditions: {unknown_conditions}")
    if unknown_targets:
        raise SystemExit(f"Unknown targets: {unknown_targets}")

    cfg = BenchConfig(
        batch=args.batch,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        height=args.height,
        width=args.width,
        kernel_size=args.kernel_size,
        padding=args.padding,
        classes=args.classes,
        warmup=args.warmup,
        iterations=args.iterations,
        seed=args.seed,
    )

    script_path = Path(__file__).resolve()
    jobs = [
        (thread_count, target, condition, repeat_index)
        for thread_count in threads
        for target in targets
        for condition in conditions
        for repeat_index in range(args.repeats)
    ]

    rng = random.Random(args.order_seed)
    rng.shuffle(jobs)

    total_invocations = len(jobs) * len(passes)
    invocation_index = 0

    for threads_n, target, condition, repeat_index in jobs:
        for pass_index, pass_events in enumerate(passes):
            invocation_index += 1
            print(
                f"[{invocation_index}/{total_invocations}] "
                f"threads={threads_n} target={target} condition={condition} "
                f"repeat={repeat_index} pass={pass_index}",
                flush=True,
            )

            worker_cmd = make_worker_command(
                script_path, condition, target, threads_n, cfg
            )
            full_worker_cmd = taskset_prefix(args.cpu_list) + worker_cmd

            with tempfile.NamedTemporaryFile(
                prefix="perf_stat_",
                suffix=".csv",
                delete=False,
            ) as temporary:
                perf_output_path = Path(temporary.name)

            perf_cmd = [
                "perf", "stat",
                "-x", ";",
                "--no-big-num",
                "-o", str(perf_output_path),
                "-e", ",".join(pass_events),
                "--",
                *full_worker_cmd,
            ]

            started_at = time.time()
            result = subprocess.run(
                perf_cmd,
                env=make_env(threads_n),
                text=True,
                capture_output=True,
                check=False,
            )
            ended_at = time.time()

            try:
                if result.returncode != 0:
                    eprint("Command failed:", " ".join(perf_cmd))
                    eprint(result.stderr[-4000:])
                    raise RuntimeError(
                        f"perf/worker failed with return code {result.returncode}"
                    )

                worker = parse_worker_json(result.stdout)
                perf_rows = parse_perf_file(perf_output_path)
            finally:
                perf_output_path.unlink(missing_ok=True)

            run_id = (
                f"t{threads_n}_{target}_{condition}_"
                f"r{repeat_index}_p{pass_index}"
            )

            common = {
                "run_id": run_id,
                "repeat": repeat_index,
                "pass": pass_index,
                "condition": condition,
                "target": target,
                "threads": threads_n,
                "cpu_list": args.cpu_list or "",
                "iterations": cfg.iterations,
                "warmup": cfg.warmup,
                "effective_rank": worker["effective_rank"],
                "gradient_norm": worker["gradient_norm"],
                "zero_fraction": worker["zero_fraction"],
                "subnormal_fraction": worker["subnormal_fraction"],
                "worker_total_ns": worker["total_ns"],
                "worker_ns_per_iteration": worker["ns_per_iteration"],
                "wall_seconds": ended_at - started_at,
            }

            output_rows = []
            for perf_row in perf_rows:
                logical = map_actual_to_logical(
                    str(perf_row["perf_event"]), selected
                )
                value = perf_row["value"]
                value_per_iteration = (
                    float(value) / cfg.iterations
                    if value is not None
                    else None
                )
                output_rows.append({
                    **common,
                    "logical_event": logical,
                    "display_event": DISPLAY_NAMES.get(logical, logical),
                    "perf_event": perf_row["perf_event"],
                    "value": value,
                    "value_per_iteration": value_per_iteration,
                    "unit": perf_row["unit"],
                    "runtime_ns": perf_row["runtime_ns"],
                    "running_pct": perf_row["running_pct"],
                })

            append_csv(raw_csv, output_rows)
            append_csv(metadata_csv, [{
                **common,
                "selected_events_json": json.dumps(selected, sort_keys=True),
                "unavailable_events_json": json.dumps(
                    unavailable, sort_keys=True
                ),
                "torch_version": worker["torch_version"],
                "mkldnn_available": worker["mkldnn_available"],
                "platform": platform.platform(),
                "python": sys.version.split()[0],
            }])

    print(f"\nRaw measurements written to {raw_csv}")
    analyze_results(outdir)


def effective_rank_sample_axis(x: "torch.Tensor") -> float:
    import torch

    matrix = x.reshape(x.shape[0], -1).double()
    singular_values = torch.linalg.svdvals(matrix)
    total = singular_values.sum()
    if total.item() == 0:
        return 0.0
    probabilities = singular_values / total
    probabilities = probabilities[probabilities > 0]
    entropy = -(probabilities * probabilities.log()).sum()
    return float(entropy.exp().item())


def tensor_value_stats(x: "torch.Tensor") -> dict[str, float]:
    import torch

    ax = x.abs()
    tiny = torch.finfo(x.dtype).tiny
    return {
        "gradient_norm": float(x.norm().item()),
        "zero_fraction": float((x == 0).float().mean().item()),
        "subnormal_fraction": float(
            ((ax > 0) & (ax < tiny)).float().mean().item()
        ),
    }


def make_gradients(
    shape: tuple[int, int, int, int],
    classes: int,
    seed: int,
) -> dict[str, "torch.Tensor"]:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    batch, channels, height, width = shape

    random_grad = torch.randn(
        shape, generator=generator, dtype=torch.float32
    )
    reference_norm = random_grad.norm()

    prototypes = torch.randn(
        classes,
        channels,
        height,
        width,
        generator=generator,
        dtype=torch.float32,
    )
    labels = torch.arange(batch) % classes
    classwise = prototypes[labels].clone()
    classwise.mul_(reference_norm / classwise.norm())

    prototype = torch.randn(
        1,
        channels,
        height,
        width,
        generator=generator,
        dtype=torch.float32,
    )
    rank1 = prototype.expand(batch, -1, -1, -1).clone()
    rank1.mul_(reference_norm / rank1.norm())

    permutation = torch.randperm(
        classwise.numel(), generator=generator
    )
    shuffled = classwise.flatten()[permutation].reshape(shape).clone()

    return {
        "random": random_grad.contiguous(),
        "classwise": classwise.contiguous(),
        "rank1": rank1.contiguous(),
        "shuffled": shuffled.contiguous(),
    }


def worker(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(args.threads)
    # Must be called before inter-op work begins. It can fail if an embedding
    # environment has already initialized the pool, so keep the worker isolated.
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)

    x = torch.randn(
        args.batch,
        args.in_channels,
        args.height,
        args.width,
        dtype=torch.float32,
        requires_grad=True,
    )
    weight = torch.randn(
        args.out_channels,
        args.in_channels,
        args.kernel_size,
        args.kernel_size,
        dtype=torch.float32,
        requires_grad=True,
    )
    y = F.conv2d(x, weight, padding=args.padding)

    gradients = make_gradients(
        tuple(y.shape),
        classes=args.classes,
        seed=args.seed + 1,
    )

    # Same storage/alignment for all conditions.
    dy = torch.empty_like(y)
    dy.copy_(gradients[args.condition])

    if args.target == "input":
        differentiated_inputs = (x,)
    elif args.target == "weight":
        differentiated_inputs = (weight,)
    else:
        raise ValueError(args.target)

    def backward_once() -> None:
        torch.autograd.grad(
            outputs=y,
            inputs=differentiated_inputs,
            grad_outputs=dy,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )

    for _ in range(args.warmup):
        backward_once()

    start = time.perf_counter_ns()
    for _ in range(args.iterations):
        backward_once()
    end = time.perf_counter_ns()

    stats = tensor_value_stats(dy)
    result = {
        "condition": args.condition,
        "target": args.target,
        "threads": args.threads,
        "iterations": args.iterations,
        "total_ns": end - start,
        "ns_per_iteration": (end - start) / args.iterations,
        "effective_rank": effective_rank_sample_axis(dy),
        **stats,
        "torch_version": torch.__version__,
        "mkldnn_available": torch.backends.mkldnn.is_available(),
    }
    print(json.dumps(result, sort_keys=True), flush=True)


def safe_ratio(numerator: "pd.Series", denominator: "pd.Series") -> "pd.Series":
    import numpy as np

    return numerator / denominator.replace(0, np.nan)


def prepare_wide(raw: "pd.DataFrame") -> "pd.DataFrame":
    import pandas as pd

    identity = [
        "run_id", "repeat", "pass", "condition", "target", "threads",
        "iterations", "effective_rank", "gradient_norm", "zero_fraction",
        "subnormal_fraction", "worker_ns_per_iteration",
    ]

    # Different perf passes repeat cycles/instructions. Keep pass-level rows,
    # then aggregate later rather than incorrectly joining unrelated runs.
    wide = (
        raw.pivot_table(
            index=identity,
            columns="logical_event",
            values="value_per_iteration",
            aggfunc="first",
        )
        .reset_index()
    )
    wide.columns.name = None

    if "cycles" in wide and "instructions" in wide:
        wide["ipc"] = safe_ratio(wide["instructions"], wide["cycles"])
        wide["cycles_per_instruction"] = safe_ratio(
            wide["cycles"], wide["instructions"]
        )

    for event in [
        "l1d_read_miss", "l1d_writeback",
        "l2d_read_miss", "l2d_writeback",
        "bus_read_access", "bus_write_access", "bus_access",
    ]:
        if event in wide and "instructions" in wide:
            wide[f"{event}_mpki"] = (
                1000.0 * safe_ratio(wide[event], wide["instructions"])
            )
        if event in wide and "cycles" in wide:
            wide[f"{event}_per_kcycle"] = (
                1000.0 * safe_ratio(wide[event], wide["cycles"])
            )

    if "l1d_read_miss" in wide and "l1d_read_access" in wide:
        wide["l1d_read_miss_rate"] = safe_ratio(
            wide["l1d_read_miss"], wide["l1d_read_access"]
        )
    if "l2d_read_miss" in wide and "l2d_read_access" in wide:
        wide["l2d_read_miss_rate"] = safe_ratio(
            wide["l2d_read_miss"], wide["l2d_read_access"]
        )

    return wide


def summarize(raw: "pd.DataFrame") -> "pd.DataFrame":
    import pandas as pd

    group_columns = [
        "condition", "target", "threads",
        "logical_event", "display_event",
    ]
    summary = (
        raw.groupby(group_columns, dropna=False)["value_per_iteration"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )
    summary["cv"] = summary["std"] / summary["mean"]
    return summary


def paired_comparisons(raw: "pd.DataFrame") -> "pd.DataFrame":
    import numpy as np
    import pandas as pd

    comparisons = [
        ("random", "rank1", "Random − rank1"),
        ("classwise", "shuffled", "Classwise − shuffled"),
    ]

    grouped = (
        raw.groupby(
            ["condition", "target", "threads", "logical_event"],
            dropna=False,
        )["value_per_iteration"]
        .median()
        .reset_index()
    )

    rows: list[dict[str, object]] = []
    for left, right, label in comparisons:
        left_df = grouped[grouped["condition"] == left].rename(
            columns={"value_per_iteration": "left_value"}
        )
        right_df = grouped[grouped["condition"] == right].rename(
            columns={"value_per_iteration": "right_value"}
        )
        merged = left_df.merge(
            right_df,
            on=["target", "threads", "logical_event"],
            suffixes=("_left", "_right"),
        )

        for _, row in merged.iterrows():
            denominator = row["right_value"]
            pct = (
                100.0 * (row["left_value"] - denominator) / denominator
                if denominator not in (0, None) and np.isfinite(denominator)
                else np.nan
            )
            rows.append({
                "comparison": label,
                "left_condition": left,
                "right_condition": right,
                "target": row["target"],
                "threads": row["threads"],
                "logical_event": row["logical_event"],
                "left_median": row["left_value"],
                "right_median": row["right_value"],
                "percent_difference_left_vs_right": pct,
            })

    return pd.DataFrame(rows)


def analyze_results(outdir: Path) -> None:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    raw_path = outdir / "raw_perf.csv"
    if not raw_path.exists():
        raise SystemExit(f"Missing {raw_path}")

    raw = pd.read_csv(raw_path)
    raw["value_per_iteration"] = pd.to_numeric(
        raw["value_per_iteration"], errors="coerce"
    )
    raw["running_pct"] = pd.to_numeric(
        raw["running_pct"], errors="coerce"
    )

    summary = summarize(raw)
    comparisons = paired_comparisons(raw)
    wide = prepare_wide(raw)

    summary.to_csv(outdir / "summary.csv", index=False)
    comparisons.to_csv(outdir / "comparisons.csv", index=False)
    wide.to_csv(outdir / "wide_runs.csv", index=False)

    pdf_path = outdir / "perf_report.pdf"

    # Metrics ordered by interpretability.
    preferred = [
        "cycles", "instructions",
        "l1d_read_access", "l1d_read_miss", "l1d_write_access",
        "l1d_writeback",
        "l2d_read_access", "l2d_read_miss", "l2d_write_access",
        "l2d_writeback",
        "bus_read_access", "bus_write_access", "bus_access", "bus_cycles",
    ]
    available_metrics = [
        metric for metric in preferred
        if metric in set(raw["logical_event"])
    ]

    with PdfPages(pdf_path) as pdf:
        # Page 1: rank sanity check.
        rank_df = (
            raw.groupby("condition", as_index=False)["effective_rank"]
            .median()
            .sort_values("effective_rank")
        )
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.bar(rank_df["condition"], rank_df["effective_rank"])
        ax.set_title("Constructed gradient effective rank")
        ax.set_ylabel("Effective rank on sample axis")
        ax.set_xlabel("Gradient condition")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig)
        fig.savefig(outdir / "effective_rank.png", dpi=180)
        plt.close(fig)

        # One page per raw counter; line over thread count exposes interaction.
        for metric in available_metrics:
            subset = summary[summary["logical_event"] == metric]
            if subset.empty:
                continue

            for target in TARGETS:
                target_df = subset[subset["target"] == target]
                if target_df.empty:
                    continue

                fig, ax = plt.subplots(figsize=(9.5, 5.8))
                for condition in CONDITIONS:
                    condition_df = (
                        target_df[target_df["condition"] == condition]
                        .sort_values("threads")
                    )
                    if condition_df.empty:
                        continue
                    ax.plot(
                        condition_df["threads"],
                        condition_df["median"],
                        marker="o",
                        label=condition,
                    )

                ax.set_title(
                    f"{DISPLAY_NAMES.get(metric, metric)} per backward iteration "
                    f"— target={target}"
                )
                ax.set_xlabel("PyTorch intra-op threads")
                ax.set_ylabel("Counter value / iteration")
                ax.set_xticks(sorted(target_df["threads"].unique()))
                ax.grid(alpha=0.3)
                ax.legend()
                fig.tight_layout()
                pdf.savefig(fig)
                filename = f"{metric}_{target}_threads.png"
                fig.savefig(outdir / filename, dpi=180)
                plt.close(fig)

        # Important percent comparisons.
        if not comparisons.empty:
            for comparison_name in comparisons["comparison"].unique():
                comp = comparisons[
                    comparisons["comparison"] == comparison_name
                ]
                for target in TARGETS:
                    target_df = comp[comp["target"] == target]
                    if target_df.empty:
                        continue

                    pivot = target_df.pivot_table(
                        index="logical_event",
                        columns="threads",
                        values="percent_difference_left_vs_right",
                        aggfunc="first",
                    )
                    if pivot.empty:
                        continue

                    fig, ax = plt.subplots(figsize=(10.5, max(5.5, 0.42 * len(pivot))))
                    image = ax.imshow(
                        pivot.to_numpy(),
                        aspect="auto",
                        interpolation="nearest",
                    )
                    ax.set_title(f"{comparison_name}: percent difference — target={target}")
                    ax.set_xlabel("Threads")
                    ax.set_ylabel("Counter")
                    ax.set_xticks(range(len(pivot.columns)))
                    ax.set_xticklabels([str(x) for x in pivot.columns])
                    ax.set_yticks(range(len(pivot.index)))
                    ax.set_yticklabels([
                        DISPLAY_NAMES.get(x, x) for x in pivot.index
                    ])
                    colorbar = fig.colorbar(image, ax=ax)
                    colorbar.set_label("% difference (left relative to right)")
                    fig.tight_layout()
                    pdf.savefig(fig)
                    safe_name = re.sub(r"[^a-z0-9]+", "_", comparison_name.lower()).strip("_")
                    fig.savefig(
                        outdir / f"comparison_{safe_name}_{target}.png",
                        dpi=180,
                    )
                    plt.close(fig)

        # Perf multiplexing diagnostic.
        running = raw.dropna(subset=["running_pct"])
        if not running.empty:
            diagnostic = (
                running.groupby("logical_event", as_index=False)["running_pct"]
                .median()
                .sort_values("running_pct")
            )
            fig, ax = plt.subplots(figsize=(9, 5.5))
            ax.barh(
                [DISPLAY_NAMES.get(x, x) for x in diagnostic["logical_event"]],
                diagnostic["running_pct"],
            )
            ax.axvline(90, linestyle="--")
            ax.set_title("Median perf counter running percentage")
            ax.set_xlabel("Running percentage")
            ax.set_ylabel("Counter")
            ax.grid(axis="x", alpha=0.3)
            fig.tight_layout()
            pdf.savefig(fig)
            fig.savefig(outdir / "perf_running_percentage.png", dpi=180)
            plt.close(fig)

    print(f"Summary written to {outdir / 'summary.csv'}")
    print(f"Comparisons written to {outdir / 'comparisons.csv'}")
    print(f"Visual report written to {pdf_path}")

    low_running = raw[
        raw["running_pct"].notna() & (raw["running_pct"] < 90)
    ]
    if not low_running.empty:
        affected = sorted(low_running["logical_event"].unique())
        print(
            "WARNING: Some counters ran below 90% and may be multiplexed: "
            + ", ".join(affected),
            file=sys.stderr,
        )


def add_benchmark_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--in-channels", type=int, default=32)
    parser.add_argument("--out-channels", type=int, default=32)
    parser.add_argument("--height", type=int, default=56)
    parser.add_argument("--width", type=int, default=56)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--padding", type=int, default=1)
    parser.add_argument("--classes", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260710)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Controlled PyTorch conv backward + perf experiment"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="Collect perf measurements"
    )
    run_parser.add_argument("--outdir", default="perf_results")
    run_parser.add_argument("--threads", default="1,2,4")
    run_parser.add_argument(
        "--conditions",
        default=",".join(CONDITIONS),
    )
    run_parser.add_argument(
        "--targets",
        default=",".join(TARGETS),
    )
    run_parser.add_argument("--repeats", type=int, default=12)
    run_parser.add_argument(
        "--cpu-list",
        default=None,
        help="taskset CPU list, e.g. 0-3. Omit to disable pinning.",
    )
    run_parser.add_argument(
        "--max-events-per-pass",
        type=int,
        default=4,
        help="Includes cycles and instructions. Lower values reduce multiplexing.",
    )
    run_parser.add_argument("--order-seed", type=int, default=9137)
    run_parser.add_argument("--overwrite", action="store_true")
    add_benchmark_arguments(run_parser)

    plot_parser = subparsers.add_parser(
        "plot", help="Regenerate CSV summaries and figures"
    )
    plot_parser.add_argument("--outdir", default="perf_results")

    all_parser = subparsers.add_parser(
        "all", help="Collect measurements, then visualize"
    )
    for action in run_parser._actions:
        if action.dest in {"help"}:
            continue
        if action.dest == "outdir":
            all_parser.add_argument("--outdir", default="perf_results")
        elif action.dest == "threads":
            all_parser.add_argument("--threads", default="1,2,4")
        elif action.dest == "conditions":
            all_parser.add_argument(
                "--conditions", default=",".join(CONDITIONS)
            )
        elif action.dest == "targets":
            all_parser.add_argument(
                "--targets", default=",".join(TARGETS)
            )
        elif action.dest == "repeats":
            all_parser.add_argument("--repeats", type=int, default=12)
        elif action.dest == "cpu_list":
            all_parser.add_argument("--cpu-list", default=None)
        elif action.dest == "max_events_per_pass":
            all_parser.add_argument(
                "--max-events-per-pass", type=int, default=4
            )
        elif action.dest == "order_seed":
            all_parser.add_argument("--order-seed", type=int, default=9137)
        elif action.dest == "overwrite":
            all_parser.add_argument("--overwrite", action="store_true")
        elif action.dest in {
            "batch", "in_channels", "out_channels", "height", "width",
            "kernel_size", "padding", "classes", "warmup", "iterations", "seed",
        }:
            option = "--" + action.dest.replace("_", "-")
            all_parser.add_argument(
                option, type=action.type, default=action.default
            )

    worker_parser = subparsers.add_parser(
        "worker", help=argparse.SUPPRESS
    )
    worker_parser.add_argument(
        "--condition", choices=CONDITIONS, required=True
    )
    worker_parser.add_argument(
        "--target", choices=TARGETS, required=True
    )
    worker_parser.add_argument("--threads", type=int, required=True)
    add_benchmark_arguments(worker_parser)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "worker":
        worker(args)
    elif args.command in {"run", "all"}:
        controller(args)
    elif args.command == "plot":
        analyze_results(Path(args.outdir).resolve())
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

