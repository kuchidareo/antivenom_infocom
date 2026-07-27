#!/usr/bin/env python3
"""Plot matched robustness runs aligned by within-batch instruction progress."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import re
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "collected_logs"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "visualization_instruction_trajectory"
BASELINE_DATASET = "kuchidareo/small_trashnet"
BASELINE_MODEL = "simple_cnn"
BASELINE_BATCH_SIZE = 16
BASELINE_AUGMENTATION = "baseline"
BASELINE_COLOR = "#202124"
SCENARIO_COLOR = "#c23b4a"

ATTACK_LABELS = {
    "availability_shortcuts": "Availability shortcuts",
    "badsampling": "BadSampler",
    "unlearnable_examples": "Unlearnable examples",
    "random_label_flipping": "Random label flipping",
}

# perf_instructions is the progress coordinate and denominator, not a response metric.
PERF_METRICS = {
    "cpi": {"label": "Cycles / instruction", "source": "perf_cycles"},
    "task_clock": {
        "label": "Task-clock (ms) / instruction",
        "source": "perf_task_clock",
    },
    "context_switches": {
        "label": "Context switches / instruction",
        "source": "perf_context_switches",
    },
    "cpu_migrations": {
        "label": "CPU migrations / instruction",
        "source": "perf_cpu_migrations",
    },
    "page_faults": {
        "label": "Page faults / instruction",
        "source": "perf_page_faults",
    },
    "branches": {
        "label": "Retired branches / instruction",
        "source": "perf_br_retired",
    },
    "branch_misses": {
        "label": "Branch mispredictions / instruction",
        "source": "perf_br_mis_pred_retired",
    },
    "l1d_access": {
        "label": "L1D accesses / instruction",
        "source": "perf_l1d_cache",
    },
    "l1d_refill": {
        "label": "L1D refills / instruction",
        "source": "perf_l1d_cache_refill",
    },
    "l1d_writeback": {
        "label": "L1D writebacks / instruction",
        "source": "perf_l1d_cache_wb",
    },
    "l2d_access": {
        "label": "L2D accesses / instruction",
        "source": "perf_l2d_cache",
    },
    "l2d_refill": {
        "label": "L2D refills / instruction",
        "source": "perf_l2d_cache_refill",
    },
    "l2d_writeback": {
        "label": "L2D writebacks / instruction",
        "source": "perf_l2d_cache_wb",
    },
    "bus_access": {
        "label": "Bus accesses / instruction",
        "source": "perf_bus_access",
    },
    "memory_access": {
        "label": "Memory accesses / instruction",
        "source": "perf_mem_access",
    },
    "speculative_instructions": {
        "label": "Speculative instructions / instruction",
        "source": "perf_inst_spec",
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
    **{
        f"system_cpu_frequency_core_{core}": {
            "label": f"Sampled CPU frequency core {core} (MHz)",
            "source": f"system_cpu_freq_core_{core}",
            "kind": "gauge",
            "scale": 1.0,
        }
        for core in range(4)
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

FOCUSED_METRICS = ("cpi", "l1d_refill", "l2d_refill")
DIFFERENCE_METRICS = (
    "l1d_access",
    "l1d_refill",
    "l2d_access",
    "l2d_refill",
    "branch_misses",
    "speculative_instructions",
)


@dataclass(frozen=True)
class Run:
    perf_path: Path
    hardware_path: Path
    device: str
    condition: str
    dataset: str
    partition_method: str
    noniid_alpha: str
    model: str
    batch_size: int
    augmentation: str
    background_enabled: bool
    background_group: str
    background_profile: str
    local_epochs: int
    trial_id: str
    start_time: float

    @property
    def config_key(self) -> tuple[object, ...]:
        background_group = self.background_group if self.background_enabled else "none"
        background_profile = self.background_profile if self.background_enabled else "none"
        return (
            self.dataset,
            self.partition_method,
            self.noniid_alpha,
            self.model,
            self.batch_size,
            self.augmentation,
            self.background_enabled,
            background_group,
            background_profile,
        )

    @property
    def config_name(self) -> str:
        parts = []
        if self.dataset != BASELINE_DATASET:
            parts.append(f"dataset_{safe_name(self.dataset.rsplit('/', 1)[-1])}")
        if self.partition_method != "iid":
            parts.append(f"partition_{safe_name(self.partition_method)}")
        if self.model != BASELINE_MODEL:
            parts.append(f"model_{safe_name(self.model)}")
        if self.batch_size != BASELINE_BATCH_SIZE:
            parts.append(f"batch_size_{self.batch_size}")
        if self.augmentation != BASELINE_AUGMENTATION:
            parts.append(f"augmentation_{safe_name(self.augmentation)}")
        if self.background_enabled:
            parts.append(
                "background_"
                f"{safe_name(self.background_group)}_{safe_name(self.background_profile)}"
            )
        return "__".join(parts) if parts else "baseline"


def safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("_") or "value"


def parse_bool(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--attacks",
        nargs="+",
        choices=tuple(ATTACK_LABELS),
        help="Attack methods to process (default: every matched attack).",
    )
    parser.add_argument(
        "--configurations",
        nargs="+",
        help="Configuration directory names to process (default: all).",
    )
    parser.add_argument(
        "--epochs",
        nargs="+",
        type=int,
        default=(0, 4, 9),
        help="Local epochs whose batches are averaged (default: 0 4 9).",
    )
    parser.add_argument("--phase", choices=("forward", "backward"), default="forward")
    parser.add_argument("--grid-points", type=int, default=201)
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


def read_run(path: Path) -> Run:
    with path.open(newline="") as handle:
        row = next(csv.DictReader(handle), None)
    if row is None:
        raise ValueError(f"Empty perf CSV: {path}")
    hardware_path = path.with_name(path.name.removesuffix("_perf.csv") + ".csv")
    if not hardware_path.is_file():
        raise FileNotFoundError(f"Missing hardware log for {path}: {hardware_path}")
    return Run(
        perf_path=path,
        hardware_path=hardware_path,
        device=row.get("device_id") or path.parent.parent.name,
        condition=row.get("poisoning_method") or "clean",
        dataset=row.get("dataset") or "",
        partition_method=row.get("partition_method") or "iid",
        noniid_alpha=row.get("noniid_alpha") or "",
        model=row.get("model") or BASELINE_MODEL,
        batch_size=int(row.get("batch_size") or BASELINE_BATCH_SIZE),
        augmentation=row.get("augmentation_profile") or BASELINE_AUGMENTATION,
        background_enabled=parse_bool(row.get("background_workload_enabled")),
        background_group=row.get("background_workload_group") or "none",
        background_profile=row.get("background_workload_profile") or "none",
        local_epochs=int(row.get("local_epochs") or 0),
        trial_id=row.get("trial_id") or "",
        start_time=float(row.get("timestamp_unix") or 0.0),
    )


def discover_pairs(input_dir: Path) -> dict[tuple[str, str], list[tuple[Run, Run]]]:
    paths = sorted(input_dir.rglob("*_perf.csv"))
    if not paths:
        raise FileNotFoundError(f"No *_perf.csv files found below {input_dir}")
    candidates: dict[tuple[str, tuple[object, ...], str], list[Run]] = {}
    for path in paths:
        run = read_run(path)
        candidates.setdefault((run.device, run.config_key, run.condition), []).append(run)

    selected = {key: max(runs, key=lambda run: run.start_time) for key, runs in candidates.items()}
    pairs: dict[tuple[str, str], list[tuple[Run, Run]]] = {}
    for (device, config_key, condition), scenario in selected.items():
        if condition == "clean":
            continue
        baseline = selected.get((device, config_key, "clean"))
        if baseline is None:
            warnings.warn(
                f"Skipping {device}/{condition}/{scenario.config_name}: "
                "no exactly matched clean run"
            )
            continue
        pairs.setdefault((condition, scenario.config_name), []).append((baseline, scenario))
    return {key: sorted(value, key=lambda pair: pair[1].device) for key, value in sorted(pairs.items())}


def interpolate(values: np.ndarray, progress: np.ndarray, grid: np.ndarray) -> np.ndarray | None:
    valid = np.isfinite(values) & np.isfinite(progress)
    if not valid.any():
        return None
    values = values[valid]
    progress = progress[valid]
    if len(values) == 1:
        return np.full_like(grid, values[0], dtype=float)
    unique_progress, indices = np.unique(progress, return_index=True)
    values = values[indices]
    if len(values) == 1:
        return np.full_like(grid, values[0], dtype=float)
    return np.interp(grid, unique_progress, values, left=values[0], right=values[-1])


def load_perf(run: Run, epochs: tuple[int, ...], phase: str) -> pd.DataFrame:
    sources = [spec["source"] for spec in PERF_METRICS.values()]
    columns = [
        "epoch",
        "batch_idx",
        "phase",
        "timestamp_unix",
        "perf_status",
        "perf_instructions",
        *sources,
    ]
    frame = pd.read_csv(run.perf_path, usecols=columns, low_memory=False)
    frame = frame.loc[
        frame["phase"].eq(phase)
        & frame["epoch"].isin(epochs)
        & frame["perf_status"].eq("ok")
    ].copy()
    for column in ["timestamp_unix", "perf_instructions", *sources]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def perf_trajectory(
    run: Run, epochs: tuple[int, ...], phase: str, grid: np.ndarray
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    frame = load_perf(run, epochs, phase)
    if frame.empty:
        raise ValueError(f"No {phase} samples for epochs {epochs} in {run.perf_path}")
    batches = {metric: [] for metric in PERF_METRICS}
    total_batches = frame.groupby(["epoch", "batch_idx"]).ngroups
    for _, batch in frame.groupby(["epoch", "batch_idx"], sort=True):
        batch = batch.sort_values("timestamp_unix")
        instructions = batch["perf_instructions"].to_numpy(dtype=float)
        valid = np.isfinite(instructions) & (instructions > 0)
        if not valid.any():
            continue
        batch = batch.loc[valid]
        instructions = instructions[valid]
        progress = (np.cumsum(instructions) - 0.5 * instructions) / instructions.sum()
        for metric, spec in PERF_METRICS.items():
            values = batch[spec["source"]].to_numpy(dtype=float) / instructions
            trajectory = interpolate(values, progress, grid)
            if trajectory is not None:
                batches[metric].append(trajectory)
    result = {
        metric: np.nanmean(np.vstack(values), axis=0)
        for metric, values in batches.items()
        if values
    }
    coverage = [
        {
            "metric_group": "perf",
            "metric": metric,
            "usable_batches": len(values),
            "total_batches": total_batches,
        }
        for metric, values in batches.items()
    ]
    return result, coverage


def available_hardware_metrics(runs: list[Run]) -> dict[str, dict[str, object]]:
    available = dict(HARDWARE_METRICS)
    for run in runs:
        header = pd.read_csv(run.hardware_path, nrows=0).columns
        available = {
            metric: spec
            for metric, spec in available.items()
            if spec["source"] in header
        }
    usable = {}
    for metric, spec in available.items():
        if all(
            pd.read_csv(run.hardware_path, usecols=[spec["source"]])[spec["source"]]
            .notna()
            .any()
            for run in runs
        ):
            usable[metric] = spec
    return usable


def hardware_trajectory(
    run: Run,
    epochs: tuple[int, ...],
    phase: str,
    grid: np.ndarray,
    metrics: dict[str, dict[str, object]],
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    perf = load_perf(run, epochs, phase)[
        ["epoch", "batch_idx", "timestamp_unix", "perf_instructions"]
    ]
    sources = sorted({str(spec["source"]) for spec in metrics.values()})
    hardware = pd.read_csv(
        run.hardware_path,
        usecols=["epoch", "batch_idx", "phase", "timestamp_unix", *sources],
        low_memory=False,
    )
    hardware = hardware.loc[
        hardware["phase"].eq(phase) & hardware["epoch"].isin(epochs)
    ].copy()
    for column in ["timestamp_unix", *sources]:
        hardware[column] = pd.to_numeric(hardware[column], errors="coerce")

    perf_groups = {key: value for key, value in perf.groupby(["epoch", "batch_idx"])}
    hardware_groups = {
        key: value for key, value in hardware.groupby(["epoch", "batch_idx"])
    }
    common = sorted(set(perf_groups) & set(hardware_groups))
    batches = {metric: [] for metric in metrics}
    for key in common:
        perf_batch = perf_groups[key].sort_values("timestamp_unix").dropna()
        perf_batch = perf_batch.loc[perf_batch["perf_instructions"].gt(0)]
        if perf_batch.empty:
            continue
        perf_time = perf_batch["timestamp_unix"].to_numpy(dtype=float)
        instructions = perf_batch["perf_instructions"].to_numpy(dtype=float)
        cumulative = np.cumsum(instructions)
        total = cumulative[-1]
        perf_progress = (cumulative - 0.5 * instructions) / total

        hardware_batch = hardware_groups[key].sort_values("timestamp_unix")
        for metric, spec in metrics.items():
            selected = hardware_batch[["timestamp_unix", str(spec["source"])]].dropna()
            if selected.empty:
                continue
            time = selected["timestamp_unix"].to_numpy(dtype=float)
            raw = selected[str(spec["source"])].to_numpy(dtype=float) * float(spec["scale"])
            if spec["kind"] == "gauge":
                progress = np.interp(
                    time, perf_time, perf_progress,
                    left=perf_progress[0], right=perf_progress[-1],
                )
                trajectory = interpolate(raw, progress, grid)
            else:
                if len(raw) < 2:
                    continue
                cumulative_at_sample = np.interp(
                    time, perf_time, cumulative, left=0.0, right=total
                )
                delta_counter = np.diff(raw)
                delta_instructions = np.diff(cumulative_at_sample)
                progress = (
                    cumulative_at_sample[1:] + cumulative_at_sample[:-1]
                ) / (2.0 * total)
                values = np.divide(
                    delta_counter,
                    delta_instructions,
                    out=np.full_like(delta_counter, np.nan),
                    where=(delta_counter >= 0) & (delta_instructions > 0),
                )
                trajectory = interpolate(values, progress, grid)
            if trajectory is not None:
                batches[metric].append(trajectory)
    result = {
        metric: np.nanmean(np.vstack(values), axis=0)
        for metric, values in batches.items()
        if values
    }
    coverage = [
        {
            "metric_group": "hardware",
            "metric": metric,
            "usable_batches": len(values),
            "total_batches": len(common),
        }
        for metric, values in batches.items()
    ]
    return result, coverage


def calculate_group(
    pairs: list[tuple[Run, Run]],
    epochs: tuple[int, ...],
    phase: str,
    grid: np.ndarray,
) -> tuple[list[str], tuple[dict, dict, dict], tuple[dict, dict, dict], list[dict]]:
    hardware_specs = available_hardware_metrics(
        [run for pair in pairs for run in pair]
    )
    perf_by_group = {"baseline": [], "scenario": []}
    hardware_by_group = {"baseline": [], "scenario": []}
    coverage = []
    devices = []
    for baseline_run, scenario_run in pairs:
        devices.append(scenario_run.device)
        for group, run in (("baseline", baseline_run), ("scenario", scenario_run)):
            perf_result, perf_coverage = perf_trajectory(run, epochs, phase, grid)
            hardware_result, hardware_coverage = hardware_trajectory(
                run, epochs, phase, grid, hardware_specs
            )
            perf_by_group[group].append(perf_result)
            hardware_by_group[group].append(hardware_result)
            for row in [*perf_coverage, *hardware_coverage]:
                coverage.append(
                    {
                        "device_id": run.device,
                        "group": group,
                        "source_file": str(run.perf_path),
                        **row,
                    }
                )

    perf_metrics = {
        metric: spec
        for metric, spec in PERF_METRICS.items()
        if all(metric in result for group in perf_by_group.values() for result in group)
    }
    hardware_specs = {
        metric: spec
        for metric, spec in hardware_specs.items()
        if all(metric in result for group in hardware_by_group.values() for result in group)
    }
    baseline_perf = {
        metric: np.vstack([result[metric] for result in perf_by_group["baseline"]])
        for metric in perf_metrics
    }
    scenario_perf = {
        metric: np.vstack([result[metric] for result in perf_by_group["scenario"]])
        for metric in perf_metrics
    }
    baseline_hardware = {
        metric: np.vstack([result[metric] for result in hardware_by_group["baseline"]])
        for metric in hardware_specs
    }
    scenario_hardware = {
        metric: np.vstack([result[metric] for result in hardware_by_group["scenario"]])
        for metric in hardware_specs
    }
    return (
        devices,
        (baseline_perf, scenario_perf, perf_metrics),
        (baseline_hardware, scenario_hardware, hardware_specs),
        coverage,
    )


def summarize(
    devices: list[str],
    grid: np.ndarray,
    baseline: dict[str, np.ndarray],
    scenario: dict[str, np.ndarray],
    specs: dict[str, dict[str, object]],
) -> tuple[dict[str, dict[str, np.ndarray]], pd.DataFrame]:
    statistics = {}
    records = []
    ddof = 1 if len(devices) > 1 else 0
    for metric, spec in specs.items():
        difference = scenario[metric] - baseline[metric]
        statistics[metric] = {
            "baseline_mean": baseline[metric].mean(axis=0),
            "baseline_std": baseline[metric].std(axis=0, ddof=ddof),
            "scenario_mean": scenario[metric].mean(axis=0),
            "scenario_std": scenario[metric].std(axis=0, ddof=ddof),
            "difference_mean": difference.mean(axis=0),
            "difference_std": difference.std(axis=0, ddof=ddof),
        }
        for device_index, device in enumerate(devices):
            for progress_index, progress in enumerate(grid):
                records.append(
                    {
                        "device_id": device,
                        "metric": metric,
                        "metric_label": spec["label"],
                        "instruction_progress": progress,
                        "baseline": baseline[metric][device_index, progress_index],
                        "scenario": scenario[metric][device_index, progress_index],
                        "paired_difference": difference[device_index, progress_index],
                    }
                )
    return statistics, pd.DataFrame(records)


def save_focused(
    statistics: dict,
    specs: dict,
    metrics: tuple[str, ...],
    grid: np.ndarray,
    path: Path,
    title: str,
    difference: bool,
    dpi: int,
) -> None:
    selected = [metric for metric in metrics if metric in statistics]
    x = grid * 100.0
    figure, axes = plt.subplots(
        len(selected), 1, figsize=(12, 3.0 * len(selected)), sharex=True, squeeze=False
    )
    colors = plt.get_cmap("tab10")
    for index, (axis, metric) in enumerate(zip(axes[:, 0], selected)):
        stats = statistics[metric]
        if difference:
            mean, std = stats["difference_mean"], stats["difference_std"]
            color = colors(index)
            axis.fill_between(x, mean - std, mean + std, color=color, alpha=0.16)
            axis.plot(x, mean, color=color, linewidth=1.7)
            axis.axhline(0, color="#4b4d50", linewidth=0.8)
            axis.set_ylabel(f"Δ {specs[metric]['label']}", fontsize=8)
        else:
            for group, color, label in (
                ("baseline", BASELINE_COLOR, "Matched clean"),
                ("scenario", SCENARIO_COLOR, "Scenario"),
            ):
                mean, std = stats[f"{group}_mean"], stats[f"{group}_std"]
                axis.fill_between(x, mean - std, mean + std, color=color, alpha=0.14)
                axis.plot(x, mean, color=color, linewidth=1.7, label=label)
            axis.set_ylabel(str(specs[metric]["label"]), fontsize=8)
        axis.tick_params(labelsize=7)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        axis.grid(True, color="#d8dadd", linewidth=0.55, alpha=0.8)
        axis.margins(x=0, y=0.12)
    if not difference:
        axes[0, 0].legend(loc="upper right", frameon=False, ncol=2)
    axes[-1, 0].set_xlabel("Normalized cumulative instruction progress (%)")
    axes[-1, 0].set_xlim(0, 100)
    suffix = "\nPaired difference: scenario - matched clean" if difference else ""
    figure.suptitle(title + suffix, fontsize=13, y=0.998)
    figure.tight_layout(rect=(0.01, 0.01, 0.99, 0.955), h_pad=1.0)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def save_all_differences(
    statistics: dict,
    specs: dict,
    grid: np.ndarray,
    path: Path,
    title: str,
    dpi: int,
) -> None:
    metrics = list(specs)
    x = grid * 100.0
    figure, axes = plt.subplots(
        len(metrics), 1, figsize=(12, 2.35 * len(metrics)),
        sharex=True, squeeze=False,
    )
    colors = plt.get_cmap("tab10")
    for index, (axis, metric) in enumerate(zip(axes[:, 0], metrics)):
        stats = statistics[metric]
        mean, std = stats["difference_mean"], stats["difference_std"]
        color = colors(index % 10)
        axis.fill_between(x, mean - std, mean + std, color=color, alpha=0.16)
        axis.plot(x, mean, color=color, linewidth=1.5)
        axis.axhline(0, color="#4b4d50", linewidth=0.75)
        axis.text(
            0.008, 0.93, str(specs[metric]["label"]), transform=axis.transAxes,
            ha="left", va="top", fontsize=8, color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72},
        )
        axis.set_ylabel("Δ", fontsize=8)
        axis.tick_params(labelsize=7)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        axis.grid(True, color="#d8dadd", linewidth=0.5, alpha=0.75)
        axis.margins(x=0, y=0.16)
    axes[-1, 0].set_xlabel("Normalized cumulative instruction progress (%)")
    axes[-1, 0].set_xlim(0, 100)
    figure.suptitle(
        title + "\nPaired difference: scenario - matched clean",
        fontsize=13, y=0.998,
    )
    figure.subplots_adjust(left=0.075, right=0.992, bottom=0.025, top=0.965, hspace=0.38)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    epochs = tuple(dict.fromkeys(args.epochs))
    if not epochs or any(epoch < 0 for epoch in epochs):
        raise ValueError("--epochs must contain non-negative local epoch numbers")
    if args.grid_points < 20:
        raise ValueError("--grid-points must be at least 20")

    pairs_by_group = discover_pairs(args.input_dir.resolve())
    attacks = set(args.attacks or ATTACK_LABELS)
    configurations = set(args.configurations or ())
    selected = {
        key: pairs
        for key, pairs in pairs_by_group.items()
        if key[0] in attacks and (not configurations or key[1] in configurations)
    }
    if not selected:
        raise ValueError("No matched attack/configuration pairs satisfy the filters")

    grid = np.linspace(0.0, 1.0, args.grid_points)
    output_root = args.output_dir.resolve()
    selected_rows = []
    print(f"Found {len(selected)} matched attack/configuration groups")
    for (attack, configuration), pairs in selected.items():
        devices, perf_group, hardware_group, coverage = calculate_group(
            pairs, epochs, args.phase, grid
        )
        baseline_perf, scenario_perf, perf_specs = perf_group
        baseline_hardware, scenario_hardware, hardware_specs = hardware_group
        perf_statistics, perf_records = summarize(
            devices, grid, baseline_perf, scenario_perf, perf_specs
        )
        hardware_statistics, hardware_records = summarize(
            devices, grid, baseline_hardware, scenario_hardware, hardware_specs
        )

        output_dir = output_root / safe_name(attack) / safe_name(configuration)
        output_dir.mkdir(parents=True, exist_ok=True)
        perf_records.to_csv(output_dir / "instruction_aligned_perf.csv", index=False)
        hardware_records.to_csv(
            output_dir / "instruction_aligned_hardware.csv", index=False
        )
        pd.DataFrame(coverage).to_csv(output_dir / "batch_coverage.csv", index=False)

        label = ATTACK_LABELS.get(attack, attack.replace("_", " "))
        title = (
            f"{label} / {configuration.replace('_', ' ')}: {args.phase} trajectory "
            f"(matched devices n={len(devices)}; epochs {', '.join(map(str, epochs))})"
        )
        save_focused(
            perf_statistics, perf_specs, FOCUSED_METRICS, grid,
            output_dir / "instruction_aligned_cache_spectrum.pdf",
            title, False, args.dpi,
        )
        save_focused(
            perf_statistics, perf_specs, DIFFERENCE_METRICS, grid,
            output_dir / "difference_spectrum.pdf",
            title, True, args.dpi,
        )
        save_all_differences(
            perf_statistics, perf_specs, grid,
            output_dir / "all_perf_difference_spectrum.pdf",
            f"{title}: all perf metrics per instruction", args.dpi,
        )
        save_all_differences(
            hardware_statistics, hardware_specs, grid,
            output_dir / "hardware_difference_spectrum.pdf",
            f"{title}: hardware and process metrics", args.dpi,
        )
        for baseline, scenario in pairs:
            selected_rows.append(
                {
                    "attack": attack,
                    "configuration": configuration,
                    "device_id": scenario.device,
                    "baseline_file": str(baseline.perf_path),
                    "scenario_file": str(scenario.perf_path),
                }
            )
        print(f"Saved {attack}/{configuration} ({len(devices)} devices)", flush=True)

    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selected_rows).to_csv(output_root / "selected_runs.csv", index=False)
    print(f"Saved instruction-aligned robustness analysis to {output_root}")


if __name__ == "__main__":
    main()
