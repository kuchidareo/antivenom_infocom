#!/usr/bin/env python3
"""Plot matched forward PMU trajectories aligned by instruction progress."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = SCRIPT_DIR / "0_raw_data_visualization.py"
DEFAULT_INPUT_DIR = SCRIPT_DIR / "collected_logs_0718"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "visualization_instruction_trajectory_0718"

ATTACK_LABELS = {
    "unlearnable_examples": "Unlearnable examples",
    "availability_shortcuts": "Availability shortcuts",
    "random_label_flipping": "Random label flipping",
}
BASELINE_COLOR = "#202124"
POISONED_COLOR = "#c23b4a"

METRICS = {
    "cpi": {
        "label": "CPI",
        "numerator": ("perf_cycles",),
    },
    "l1d_access": {
        "label": "L1D read accesses / instruction",
        "numerator": ("perf_l1d_cache_rd",),
    },
    "l1d_miss": {
        "label": "L1D read refills / instruction",
        "numerator": ("perf_l1d_cache_refill_rd",),
    },
    "task_clock": {
        "label": "Task-clock (ms) / instruction",
        "numerator": ("perf_task_clock",),
    },
    "context_switches": {
        "label": "Context switches / instruction",
        "numerator": ("perf_context_switches",),
    },
    "cpu_migrations": {
        "label": "CPU migrations / instruction",
        "numerator": ("perf_cpu_migrations",),
    },
    "page_faults": {
        "label": "Page faults / instruction",
        "numerator": ("perf_page_faults",),
    },
    "l1d_write_access": {
        "label": "L1D write accesses / instruction",
        "numerator": ("perf_l1d_cache_wr",),
    },
    "l1d_write_miss": {
        "label": "L1D write refills / instruction",
        "numerator": ("perf_l1d_cache_refill_wr",),
    },
    "l2d_access": {
        "label": "L2D read accesses / instruction",
        "numerator": ("perf_l2d_cache_rd",),
    },
    "llc_miss_proxy": {
        "label": "L2D read refills / instruction (LLC proxy)",
        "numerator": ("perf_l2d_cache_refill_rd",),
    },
    "l2d_write_access": {
        "label": "L2D write accesses / instruction",
        "numerator": ("perf_l2d_cache_wr",),
    },
    "l2d_write_miss": {
        "label": "L2D write refills / instruction",
        "numerator": ("perf_l2d_cache_refill_wr",),
    },
    "bus_read_access": {
        "label": "Bus read accesses / instruction",
        "numerator": ("perf_bus_access_rd",),
    },
    "bus_write_access": {
        "label": "Bus write accesses / instruction",
        "numerator": ("perf_bus_access_wr",),
    },
    "memory_access": {
        "label": "Memory accesses / instruction",
        "numerator": ("perf_mem_access",),
    },
    "branch_miss": {
        "label": "Branch misses / instruction",
        "numerator": ("perf_branch_misses",),
    },
    "vfp": {
        "label": "VFP operations / instruction",
        "numerator": ("perf_vfp_spec",),
    },
    "ase": {
        "label": "ASE operations / instruction",
        "numerator": ("perf_ase_spec",),
    },
    "speculative_instructions": {
        "label": "Speculative instructions / instruction",
        "numerator": ("perf_inst_spec",),
    },
}

HARDWARE_METRICS = {
    **{
        f"system_cpu_core_{core}": {
            "label": f"System CPU core {core} utilization (percentage points)",
            "source": f"system_cpu_core_{core}",
            "kind": "gauge",
            "scale": 1.0,
        }
        for core in range(4)
    },
    "system_cpu_frequency": {
        "label": "Sampled CPU frequency (MHz)",
        "source": "system_cpu_freq_core_0",
        "kind": "gauge",
        "scale": 1.0,
    },
    "system_memory_percent": {
        "label": "System memory utilization (percentage points)",
        "source": "system_memory_percent",
        "kind": "gauge",
        "scale": 1.0,
    },
    "system_memory_used": {
        "label": "System memory used (GiB)",
        "source": "system_memory_used",
        "kind": "gauge",
        "scale": 1.0 / 2**30,
    },
    "system_memory_available": {
        "label": "System memory available (GiB)",
        "source": "system_memory_available",
        "kind": "gauge",
        "scale": 1.0 / 2**30,
    },
    "process_cpu_percent": {
        "label": "Process CPU utilization (percentage points)",
        "source": "process_cpu_percent",
        "kind": "gauge",
        "scale": 1.0,
    },
    "process_memory_rss": {
        "label": "Process RSS (GiB)",
        "source": "process_memory_rss",
        "kind": "gauge",
        "scale": 1.0 / 2**30,
    },
    "process_memory_vms": {
        "label": "Process VMS (GiB)",
        "source": "process_memory_vms",
        "kind": "gauge",
        "scale": 1.0 / 2**30,
    },
    "process_memory_percent": {
        "label": "Process memory utilization (percentage points)",
        "source": "process_memory_percent",
        "kind": "gauge",
        "scale": 1.0,
    },
    "process_ctx_switches_voluntary": {
        "label": "Voluntary context-switch increments / instruction",
        "source": "process_ctx_switches_voluntary",
        "kind": "counter",
        "scale": 1.0,
    },
    "process_ctx_switches_involuntary": {
        "label": "Involuntary context-switch increments / instruction",
        "source": "process_ctx_switches_involuntary",
        "kind": "counter",
        "scale": 1.0,
    },
    "process_minor_faults": {
        "label": "Minor-fault increments / instruction",
        "source": "process_minor_faults",
        "kind": "counter",
        "scale": 1.0,
    },
}
MAIN_METRICS = ("cpi", "l1d_miss", "llc_miss_proxy")
DIFFERENCE_METRICS = (
    "l1d_access",
    "l1d_miss",
    "l2d_access",
    "llc_miss_proxy",
    "branch_miss",
    "vfp",
)
DIFFERENCE_COLORS = {
    "l1d_access": "#007f83",
    "l1d_miss": "#c23b4a",
    "l2d_access": "#2d8f55",
    "llc_miss_proxy": "#386cb0",
    "branch_miss": "#7a5195",
    "vfp": "#d27c2c",
}
DIFFERENCE_LABELS = {
    "l1d_access": "Δ L1D reads / I",
    "l1d_miss": "Δ L1D refills / I",
    "l2d_access": "Δ L2D reads / I",
    "llc_miss_proxy": "Δ L2D refills / I\n(LLC proxy)",
    "branch_miss": "Δ branch misses / I",
    "vfp": "Δ VFP / I",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--attack", choices=tuple(ATTACK_LABELS), default="availability_shortcuts"
    )
    parser.add_argument("--poisoned-count", type=int, default=4)
    parser.add_argument(
        "--rounds",
        nargs="+",
        type=int,
        default=(0, 7, 14),
        help="Zero-based FL rounds to average (default: 0 7 14).",
    )
    parser.add_argument("--grid-points", type=int, default=201)
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


def load_base_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fl_raw_visualization", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def select_poisoned_pairs(
    base: ModuleType,
    input_dir: Path,
    attack: str,
    poisoned_count: int,
):
    pairs = []
    for device, runs in base.discover_runs(input_dir).items():
        baseline = [run for run in runs if run.global_method == "clean"]
        attacked = [
            run
            for run in runs
            if run.global_method == attack
            and run.poisoned_count == poisoned_count
            and run.locally_poisoned
        ]
        if attacked:
            if len(baseline) != 1 or len(attacked) != 1:
                raise ValueError(f"{device}: baseline/poisoned run is not unique")
            pairs.append((baseline[0], attacked[0]))
    if len(pairs) != poisoned_count:
        raise ValueError(
            f"Expected {poisoned_count} poisoned client pairs, found {len(pairs)}"
        )
    return pairs


def interpolate_batch(
    batch: pd.DataFrame,
    grid: np.ndarray,
) -> dict[str, np.ndarray] | None:
    instructions = batch["perf_instructions"].to_numpy(dtype=float)
    valid_instructions = np.isfinite(instructions) & (instructions > 0)
    if valid_instructions.sum() < 2:
        return None
    batch = batch.loc[valid_instructions].copy()
    instructions = batch["perf_instructions"].to_numpy(dtype=float)
    total_instructions = instructions.sum()
    progress = (np.cumsum(instructions) - 0.5 * instructions) / total_instructions

    result = {}
    for metric, spec in METRICS.items():
        numerator = batch[list(spec["numerator"])].sum(axis=1, min_count=1).to_numpy()
        values = numerator / instructions
        valid = np.isfinite(progress) & np.isfinite(values)
        if valid.sum() < 2:
            continue
        result[metric] = np.interp(
            grid,
            progress[valid],
            values[valid],
            left=values[valid][0],
            right=values[valid][-1],
        )
    return result


def client_trajectory(
    run,
    rounds: tuple[int, ...],
    grid: np.ndarray,
) -> dict[str, np.ndarray]:
    event_columns = sorted(
        {
            "perf_instructions",
            *(column for spec in METRICS.values() for column in spec["numerator"]),
        }
    )
    columns = ["round", "batch_idx", "phase", "timestamp_unix", *event_columns]
    frame = pd.read_csv(run.perf_path, usecols=columns, low_memory=False)
    csv_rounds = {round_id + 1 for round_id in rounds}
    frame = frame.loc[
        frame["phase"].eq("forward") & frame["round"].isin(csv_rounds)
    ].copy()
    for column in ["timestamp_unix", *event_columns]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    batches: dict[str, list[np.ndarray]] = {metric: [] for metric in METRICS}
    for _, batch in frame.groupby(["round", "batch_idx"], sort=True):
        batch = batch.sort_values("timestamp_unix")
        trajectory = interpolate_batch(batch, grid)
        if trajectory is None:
            continue
        for metric, values in trajectory.items():
            batches[metric].append(values)

    expected_batches = len(rounds) * frame["batch_idx"].nunique()
    if not batches["cpi"] or len(batches["cpi"]) != expected_batches:
        raise ValueError(
            f"{run.device}: expected {expected_batches} forward batches, "
            f"found {len(batches['cpi'])}"
        )
    return {
        metric: np.mean(np.vstack(values), axis=0)
        for metric, values in batches.items()
        if values
    }


def calculate_trajectories(pairs, rounds: tuple[int, ...], grid: np.ndarray):
    devices = []
    baseline = {metric: [] for metric in METRICS}
    poisoned = {metric: [] for metric in METRICS}
    for baseline_run, poisoned_run in pairs:
        devices.append(poisoned_run.device)
        baseline_client = client_trajectory(baseline_run, rounds, grid)
        poisoned_client = client_trajectory(poisoned_run, rounds, grid)
        for metric in METRICS:
            baseline[metric].append(baseline_client[metric])
            poisoned[metric].append(poisoned_client[metric])
    return (
        devices,
        {metric: np.vstack(values) for metric, values in baseline.items()},
        {metric: np.vstack(values) for metric, values in poisoned.items()},
    )


def interpolate_hardware_batch(
    perf_batch: pd.DataFrame,
    hardware_batch: pd.DataFrame,
    grid: np.ndarray,
) -> dict[str, np.ndarray] | None:
    perf_batch = perf_batch.sort_values("timestamp_unix").dropna(
        subset=["timestamp_unix", "perf_instructions"]
    )
    perf_batch = perf_batch.loc[perf_batch["perf_instructions"].gt(0)]
    if len(perf_batch) < 2 or len(hardware_batch) < 2:
        return None

    perf_time = perf_batch["timestamp_unix"].to_numpy(dtype=float)
    instructions = perf_batch["perf_instructions"].to_numpy(dtype=float)
    total_instructions = instructions.sum()
    cumulative_instructions = np.cumsum(instructions)
    perf_progress = (
        cumulative_instructions - 0.5 * instructions
    ) / total_instructions

    result = {}
    for metric, spec in HARDWARE_METRICS.items():
        selected = hardware_batch[["timestamp_unix", spec["source"]]].dropna()
        selected = selected.sort_values("timestamp_unix")
        if len(selected) < 2:
            continue
        hardware_time = selected["timestamp_unix"].to_numpy(dtype=float)
        raw_values = selected[spec["source"]].to_numpy(dtype=float) * spec["scale"]

        if spec["kind"] == "gauge":
            progress = np.interp(
                hardware_time,
                perf_time,
                perf_progress,
                left=perf_progress[0],
                right=perf_progress[-1],
            )
            result[metric] = np.interp(
                grid,
                progress,
                raw_values,
                left=raw_values[0],
                right=raw_values[-1],
            )
            continue

        hardware_cumulative_instructions = np.interp(
            hardware_time,
            perf_time,
            cumulative_instructions,
            left=0.0,
            right=total_instructions,
        )
        delta_counter = np.diff(raw_values)
        delta_instructions = np.diff(hardware_cumulative_instructions)
        progress = (
            hardware_cumulative_instructions[1:]
            + hardware_cumulative_instructions[:-1]
        ) / (2.0 * total_instructions)
        values = delta_counter / delta_instructions
        valid = (
            np.isfinite(progress)
            & np.isfinite(values)
            & (delta_counter >= 0)
            & (delta_instructions > 0)
        )
        if valid.sum() < 2:
            continue
        result[metric] = np.interp(
            grid,
            progress[valid],
            values[valid],
            left=values[valid][0],
            right=values[valid][-1],
        )
    return result


def client_hardware_trajectory(
    run,
    rounds: tuple[int, ...],
    grid: np.ndarray,
) -> dict[str, np.ndarray]:
    csv_rounds = {round_id + 1 for round_id in rounds}
    perf = pd.read_csv(
        run.perf_path,
        usecols=[
            "round",
            "batch_idx",
            "phase",
            "timestamp_unix",
            "perf_instructions",
        ],
        low_memory=False,
    )
    main_path = run.perf_path.with_name(
        run.perf_path.name.removesuffix("_perf.csv") + ".csv"
    )
    hardware_sources = sorted(
        {spec["source"] for spec in HARDWARE_METRICS.values()}
    )
    hardware = pd.read_csv(
        main_path,
        usecols=[
            "round",
            "batch_idx",
            "phase",
            "timestamp_unix",
            *hardware_sources,
        ],
        low_memory=False,
    )
    perf = perf.loc[
        perf["phase"].eq("forward") & perf["round"].isin(csv_rounds)
    ].copy()
    hardware = hardware.loc[
        hardware["phase"].eq("forward") & hardware["round"].isin(csv_rounds)
    ].copy()
    for column in ["timestamp_unix", "perf_instructions"]:
        perf[column] = pd.to_numeric(perf[column], errors="coerce")
    for column in ["timestamp_unix", *hardware_sources]:
        hardware[column] = pd.to_numeric(hardware[column], errors="coerce")

    perf_groups = {
        key: batch for key, batch in perf.groupby(["round", "batch_idx"])
    }
    hardware_groups = {
        key: batch for key, batch in hardware.groupby(["round", "batch_idx"])
    }
    batches: dict[str, list[np.ndarray]] = {
        metric: [] for metric in HARDWARE_METRICS
    }
    for key in sorted(set(perf_groups) & set(hardware_groups)):
        trajectory = interpolate_hardware_batch(
            perf_groups[key], hardware_groups[key], grid
        )
        if trajectory is None:
            continue
        for metric, values in trajectory.items():
            batches[metric].append(values)

    expected_batches = len(rounds) * perf["batch_idx"].nunique()
    missing = [
        metric
        for metric, values in batches.items()
        if len(values) != expected_batches
    ]
    if missing:
        raise ValueError(
            f"{run.device}: incomplete hardware trajectories for {missing}"
        )
    return {
        metric: np.mean(np.vstack(values), axis=0)
        for metric, values in batches.items()
    }


def calculate_hardware_trajectories(
    pairs,
    rounds: tuple[int, ...],
    grid: np.ndarray,
):
    devices = []
    baseline = {metric: [] for metric in HARDWARE_METRICS}
    poisoned = {metric: [] for metric in HARDWARE_METRICS}
    for baseline_run, poisoned_run in pairs:
        devices.append(poisoned_run.device)
        baseline_client = client_hardware_trajectory(baseline_run, rounds, grid)
        poisoned_client = client_hardware_trajectory(poisoned_run, rounds, grid)
        for metric in HARDWARE_METRICS:
            baseline[metric].append(baseline_client[metric])
            poisoned[metric].append(poisoned_client[metric])
    return (
        devices,
        {metric: np.vstack(values) for metric, values in baseline.items()},
        {metric: np.vstack(values) for metric, values in poisoned.items()},
    )


def summarize(
    devices: list[str],
    grid: np.ndarray,
    baseline: dict[str, np.ndarray],
    poisoned: dict[str, np.ndarray],
    metric_specs: dict = METRICS,
) -> tuple[dict, pd.DataFrame]:
    statistics = {}
    records = []
    for metric, spec in metric_specs.items():
        paired_difference = poisoned[metric] - baseline[metric]
        statistics[metric] = {
            "baseline_mean": baseline[metric].mean(axis=0),
            "baseline_std": baseline[metric].std(axis=0, ddof=1),
            "poisoned_mean": poisoned[metric].mean(axis=0),
            "poisoned_std": poisoned[metric].std(axis=0, ddof=1),
            "difference_mean": paired_difference.mean(axis=0),
            "difference_std": paired_difference.std(axis=0, ddof=1),
        }
        for client_index, device in enumerate(devices):
            for progress_index, progress in enumerate(grid):
                records.append(
                    {
                        "device_id": device,
                        "metric": metric,
                        "metric_label": spec["label"],
                        "instruction_progress": progress,
                        "baseline": baseline[metric][client_index, progress_index],
                        "poisoned": poisoned[metric][client_index, progress_index],
                        "paired_difference": paired_difference[
                            client_index, progress_index
                        ],
                    }
                )
    return statistics, pd.DataFrame(records)


def save_main_figure(
    statistics: dict,
    grid: np.ndarray,
    path: Path,
    title: str,
    dpi: int,
) -> None:
    x = grid * 100.0
    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for axis, metric in zip(axes, MAIN_METRICS):
        stats = statistics[metric]
        axis.fill_between(
            x,
            stats["baseline_mean"] - stats["baseline_std"],
            stats["baseline_mean"] + stats["baseline_std"],
            color=BASELINE_COLOR,
            alpha=0.13,
            linewidth=0,
        )
        axis.fill_between(
            x,
            stats["poisoned_mean"] - stats["poisoned_std"],
            stats["poisoned_mean"] + stats["poisoned_std"],
            color=POISONED_COLOR,
            alpha=0.16,
            linewidth=0,
        )
        axis.plot(x, stats["baseline_mean"], color=BASELINE_COLOR, linewidth=1.8, label="Matched baseline")
        axis.plot(x, stats["poisoned_mean"], color=POISONED_COLOR, linewidth=1.8, label="Poisoned")
        axis.set_ylabel(METRICS[metric]["label"])
        axis.grid(True, color="#d8dadd", linewidth=0.6, alpha=0.8)
        axis.margins(x=0)
    axes[0].legend(loc="upper right", frameon=False, ncol=2)
    axes[-1].set_xlabel("Normalized cumulative instruction progress (%)")
    axes[-1].set_xlim(0, 100)
    figure.suptitle(title, fontsize=14, y=0.995)
    figure.tight_layout(rect=(0.01, 0.01, 0.99, 0.965), h_pad=1.0)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def save_difference_figure(
    statistics: dict,
    grid: np.ndarray,
    path: Path,
    title: str,
    dpi: int,
) -> None:
    x = grid * 100.0
    figure, axes = plt.subplots(
        len(DIFFERENCE_METRICS),
        1,
        figsize=(12, 2.8 * len(DIFFERENCE_METRICS)),
        sharex=True,
    )
    for axis, metric in zip(axes, DIFFERENCE_METRICS):
        stats = statistics[metric]
        mean = stats["difference_mean"]
        std = stats["difference_std"]
        color = DIFFERENCE_COLORS[metric]
        axis.fill_between(x, mean - std, mean + std, color=color, alpha=0.17, linewidth=0)
        axis.plot(x, mean, color=color, linewidth=1.8)
        axis.axhline(0, color="#4b4d50", linewidth=0.8)
        peak_index = int(np.nanargmax(np.abs(mean)))
        peak_x = x[peak_index]
        peak_y = mean[peak_index]
        axis.scatter([peak_x], [peak_y], color=color, s=28, zorder=3)
        axis.annotate(
            f"peak {peak_x:.1f}%: {peak_y:+.3g}",
            (peak_x, peak_y),
            xytext=(6, 7 if peak_y >= 0 else -13),
            textcoords="offset points",
            fontsize=8,
            color=color,
        )
        axis.set_ylabel(DIFFERENCE_LABELS[metric], fontsize=9)
        axis.grid(True, color="#d8dadd", linewidth=0.6, alpha=0.8)
        axis.margins(x=0, y=0.16)
    axes[-1].set_xlabel("Normalized cumulative instruction progress (%)")
    axes[-1].set_xlim(0, 100)
    figure.suptitle(f"{title}\nPaired difference: poisoned − matched baseline", fontsize=14, y=0.995)
    figure.tight_layout(rect=(0.01, 0.01, 0.99, 0.955), h_pad=1.0)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def save_all_difference_figure(
    statistics: dict,
    metric_specs: dict,
    grid: np.ndarray,
    path: Path,
    title: str,
    dpi: int,
) -> None:
    metrics = list(metric_specs)
    x = grid * 100.0
    figure, axes = plt.subplots(
        len(metrics),
        1,
        figsize=(12, 2.35 * len(metrics)),
        sharex=True,
        squeeze=False,
    )
    color_map = plt.get_cmap("tab10")
    for index, (axis, metric) in enumerate(zip(axes[:, 0], metrics)):
        stats = statistics[metric]
        mean = stats["difference_mean"]
        std = stats["difference_std"]
        color = color_map(index % 10)
        axis.fill_between(
            x, mean - std, mean + std, color=color, alpha=0.16, linewidth=0
        )
        axis.plot(x, mean, color=color, linewidth=1.5)
        axis.axhline(0, color="#4b4d50", linewidth=0.75)
        axis.text(
            0.008,
            0.93,
            metric_specs[metric]["label"],
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72},
        )
        axis.set_ylabel("Δ", fontsize=8)
        axis.tick_params(axis="both", labelsize=7)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        axis.grid(True, color="#d8dadd", linewidth=0.5, alpha=0.75)
        axis.margins(x=0, y=0.16)
    axes[-1, 0].set_xlabel("Normalized cumulative instruction progress (%)")
    axes[-1, 0].set_xlim(0, 100)
    figure.suptitle(
        f"{title}\nPaired difference: poisoned − matched baseline",
        fontsize=14,
        y=0.998,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.992,
        bottom=0.025,
        top=0.965,
        hspace=0.38,
    )
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    rounds = tuple(args.rounds)
    if not rounds or any(round_id < 0 or round_id > 14 for round_id in rounds):
        raise ValueError("--rounds must contain zero-based values from 0 through 14")
    if args.grid_points < 20:
        raise ValueError("--grid-points must be at least 20")

    base = load_base_module()
    pairs = select_poisoned_pairs(
        base,
        args.input_dir.resolve(),
        args.attack,
        args.poisoned_count,
    )
    grid = np.linspace(0.0, 1.0, args.grid_points)
    devices, baseline, poisoned = calculate_trajectories(pairs, rounds, grid)
    statistics, trajectories = summarize(
        devices, grid, baseline, poisoned, METRICS
    )
    hardware_devices, hardware_baseline, hardware_poisoned = (
        calculate_hardware_trajectories(pairs, rounds, grid)
    )
    hardware_statistics, hardware_trajectories = summarize(
        hardware_devices,
        grid,
        hardware_baseline,
        hardware_poisoned,
        HARDWARE_METRICS,
    )

    output_dir = args.output_dir.resolve() / args.attack
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories.to_csv(output_dir / "instruction_aligned_trajectories.csv", index=False)
    hardware_trajectories.to_csv(
        output_dir / "hardware_instruction_aligned_trajectories.csv", index=False
    )

    round_label = ", ".join(map(str, rounds))
    title = (
        f"{ATTACK_LABELS[args.attack]}: forward instruction-aligned trajectory "
        f"(matched poisoned clients n={len(pairs)}; rounds {round_label})"
    )
    save_main_figure(
        statistics,
        grid,
        output_dir / "instruction_aligned_cache_spectrum.pdf",
        title,
        args.dpi,
    )
    save_difference_figure(
        statistics,
        grid,
        output_dir / "difference_spectrum.pdf",
        title,
        args.dpi,
    )
    save_all_difference_figure(
        statistics,
        METRICS,
        grid,
        output_dir / "all_perf_difference_spectrum.pdf",
        f"{title}: all perf metrics per instruction",
        args.dpi,
    )
    save_all_difference_figure(
        hardware_statistics,
        HARDWARE_METRICS,
        grid,
        output_dir / "hardware_difference_spectrum.pdf",
        f"{title}: hardware and process metrics",
        args.dpi,
    )
    print(f"Saved instruction-aligned analysis to {output_dir}")


if __name__ == "__main__":
    main()
