#!/usr/bin/env python3
"""Visualize local-ML raw telemetry after fixed-window averaging.

This script keeps the telemetry values in their original units, but normalizes
trace length by averaging each run into a fixed number of sequential windows.
By default it uses 128 windows. This is useful when runs have different sample
counts but we want to inspect coarse temporal patterns without resampling every
point or applying z-score/min-max scaling.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache-antivenom")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_CONDITIONS = [
    "clean",
    "random_label_flipping",
    "target_label_flipping",
    "availability_shortcuts",
    "unlearnable_examples",
]

CPU_CORE_COLUMNS = [
    "system_cpu_core_0",
    "system_cpu_core_1",
    "system_cpu_core_2",
    "system_cpu_core_3",
]

COUNTER_COLUMNS = [
    "process_ctx_switches_voluntary",
    "process_ctx_switches_involuntary",
    "process_minor_faults",
]

PLOT_COLUMNS = [
    "system_cpu_core_0",
    "system_cpu_core_1",
    "system_cpu_core_2",
    "system_cpu_core_3",
    "system_memory_percent",
    "process_ctx_switches_voluntary",
    "process_ctx_switches_involuntary",
    "process_minor_faults",
]

DISPLAY_NAMES = {
    "system_cpu_core_0": "cpu_core_0_ranked_highest",
    "system_cpu_core_1": "cpu_core_1_ranked",
    "system_cpu_core_2": "cpu_core_2_ranked",
    "system_cpu_core_3": "cpu_core_3_ranked_lowest",
    "system_memory_percent": "memory_percent",
    "process_ctx_switches_voluntary": "voluntary_context_switch_delta",
    "process_ctx_switches_involuntary": "involuntary_context_switch_delta",
    "process_minor_faults": "minor_fault_delta",
}

COLORS = {
    "clean": "#1f77b4",
    "random_label_flipping": "#2ca02c",
    "target_label_flipping": "#9467bd",
    "availability_shortcuts": "#ff7f0e",
    "unlearnable_examples": "#d62728",
}


@dataclass(frozen=True)
class RunInfo:
    path: Path
    device_id: str
    trial_id: str
    poisoning_method: str
    attack_name: str
    timestamp_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average raw local-ML telemetry into fixed windows and plot it."
    )
    parser.add_argument("--input_dir", default="collected_logs")
    parser.add_argument("--output_dir", default="normalized_raw_data_visualization_result")
    parser.add_argument("--device", default="192.168.0.112")
    parser.add_argument("--trial_id", default="trial_0")
    parser.add_argument(
        "--epoch",
        type=int,
        default=0,
        help="Local-ML epoch to plot. Use a negative value to plot the full run.",
    )
    parser.add_argument("--num_windows", type=int, default=128)
    parser.add_argument(
        "--conditions",
        default=",".join(DEFAULT_CONDITIONS),
        help="Comma-separated poisoning_method values to plot.",
    )
    parser.add_argument(
        "--include_idle",
        action="store_true",
        help="Keep idle rows inside the selected epoch.",
    )
    parser.add_argument(
        "--physical_cpu_cores",
        action="store_true",
        help="Plot physical CPU core columns instead of per-timestamp ranked cores.",
    )
    parser.add_argument(
        "--save_individual",
        action="store_true",
        help="Also save one figure per metric.",
    )
    parser.add_argument(
        "--save_window_csv",
        action="store_true",
        help="Save the 128-window averaged telemetry values as CSV.",
    )
    return parser.parse_args()


def split_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def safe_name(value: object) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_")


def read_first_row(path: Path) -> Optional[Dict[str, str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return next(reader, None)


def is_hardware_csv(path: Path) -> bool:
    if path.name.endswith("_metrics.csv"):
        return False
    first = read_first_row(path)
    if not first:
        return False
    return all(column in first for column in PLOT_COLUMNS)


def discover_runs(input_dir: Path, device: str, trial_id: str) -> List[RunInfo]:
    local_dir = input_dir / device / "local_ml"
    if not local_dir.exists():
        raise FileNotFoundError(f"Local-ML log directory does not exist: {local_dir}")

    runs: List[RunInfo] = []
    for path in sorted(local_dir.glob("*.csv")):
        if not is_hardware_csv(path):
            continue
        first = read_first_row(path)
        if not first:
            continue
        if first.get("run_type") != "local_ml":
            continue
        if first.get("trial_id") != trial_id:
            continue
        runs.append(
            RunInfo(
                path=path,
                device_id=first.get("device_id") or device,
                trial_id=first.get("trial_id") or "",
                poisoning_method=first.get("poisoning_method") or "clean",
                attack_name=first.get("attack_name") or "",
                timestamp_name=path.stem,
            )
        )
    return runs


def choose_runs_by_condition(runs: Sequence[RunInfo], conditions: Sequence[str]) -> Dict[str, RunInfo]:
    selected: Dict[str, RunInfo] = {}
    for condition in conditions:
        candidates = sorted(
            [run for run in runs if run.poisoning_method == condition],
            key=lambda run: run.timestamp_name,
        )
        if not candidates:
            print(f"warning: no run found for condition={condition}")
            continue
        if len(candidates) > 1:
            print(
                f"warning: multiple runs found for condition={condition}; "
                f"using latest {candidates[-1].path.name}"
            )
        selected[condition] = candidates[-1]
    if not selected:
        raise ValueError("No matching local-ML hardware runs were found.")
    return selected


def sort_cpu_cores_per_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in CPU_CORE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing CPU columns: {missing}")
    out = df.copy()
    cpu_values = out.loc[:, CPU_CORE_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    sorted_values = np.sort(cpu_values, axis=1)[:, ::-1]
    out.loc[:, CPU_CORE_COLUMNS] = sorted_values
    return out


def apply_counter_deltas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in COUNTER_COLUMNS:
        values = pd.to_numeric(out[column], errors="coerce")
        deltas = values.diff().fillna(0.0)
        out[column] = deltas.mask(deltas < 0.0, 0.0)
    return out


def load_run_trace(
    run: RunInfo,
    epoch: int,
    include_idle: bool,
    rank_cpu_cores: bool,
) -> pd.DataFrame:
    df = pd.read_csv(run.path)
    required = ["timestamp_unix", "epoch", "phase", *PLOT_COLUMNS]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{run.path} is missing columns: {missing}")

    for column in ["timestamp_unix", "epoch", *PLOT_COLUMNS]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if not np.isfinite(df[["timestamp_unix", *PLOT_COLUMNS]].to_numpy(dtype=float)).all():
        raise ValueError(f"{run.path} contains NaN/inf in plotted columns.")

    if rank_cpu_cores:
        df = sort_cpu_cores_per_timestamp(df)
    df = apply_counter_deltas(df)

    if epoch >= 0:
        df = df[df["epoch"] == epoch].copy()
    if not include_idle:
        df = df[df["phase"] != "idle"].copy()
    if df.empty:
        raise ValueError(f"No rows remain after filtering {run.path} for epoch={epoch}.")

    df = df.sort_values("timestamp_unix").reset_index(drop=True)
    df["relative_time"] = df["timestamp_unix"] - float(df["timestamp_unix"].iloc[0])
    df["condition"] = run.poisoning_method
    df["attack_name"] = run.attack_name
    df["source_file"] = run.path.name
    return df


def average_to_windows(values: np.ndarray, num_windows: int) -> np.ndarray:
    """Average a 1D series into num_windows contiguous bins."""
    if num_windows <= 0:
        raise ValueError("--num_windows must be positive.")
    if values.ndim != 1:
        raise ValueError(f"values must be 1D, got shape {values.shape}")
    if len(values) == 0:
        raise ValueError("Cannot average an empty time series.")

    edges = np.linspace(0, len(values), num_windows + 1)
    out = np.empty(num_windows, dtype=float)
    for idx in range(num_windows):
        start = int(np.floor(edges[idx]))
        end = int(np.floor(edges[idx + 1]))
        if idx == num_windows - 1:
            end = len(values)
        if end <= start:
            nearest = min(start, len(values) - 1)
            out[idx] = float(values[nearest])
        else:
            out[idx] = float(np.mean(values[start:end]))
    return out


def build_window_dataframe(
    traces: Dict[str, pd.DataFrame],
    selected_runs: Dict[str, RunInfo],
    num_windows: int,
) -> pd.DataFrame:
    rows = []
    x = np.linspace(0.0, 1.0, num_windows)
    for condition, df in traces.items():
        run = selected_runs[condition]
        for column in PLOT_COLUMNS:
            averaged = average_to_windows(df[column].to_numpy(dtype=float), num_windows)
            for window_idx, value in enumerate(averaged):
                rows.append(
                    {
                        "condition": condition,
                        "attack_name": run.attack_name,
                        "source_file": run.path.name,
                        "metric": DISPLAY_NAMES[column],
                        "source_column": column,
                        "window_idx": window_idx,
                        "normalized_time": float(x[window_idx]),
                        "value": float(value),
                    }
                )
    return pd.DataFrame(rows)


def y_label_for_column(column: str) -> str:
    if column in CPU_CORE_COLUMNS or column == "system_memory_percent":
        return "raw mean percent"
    return "raw mean delta/sample"


def plot_combined(
    window_df: pd.DataFrame,
    output_dir: Path,
    device: str,
    trial_id: str,
    epoch: int,
    num_windows: int,
    rank_cpu_cores: bool,
) -> Path:
    n_rows = len(PLOT_COLUMNS)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2.45 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    for ax, column in zip(axes, PLOT_COLUMNS):
        metric_name = DISPLAY_NAMES[column]
        sub = window_df[window_df["source_column"] == column]
        for condition, cdf in sub.groupby("condition", sort=False):
            ax.plot(
                cdf["normalized_time"],
                cdf["value"],
                label=condition,
                color=COLORS.get(condition),
                linewidth=1.55,
                alpha=0.94,
            )
        ax.set_title(metric_name, fontsize=10)
        ax.set_ylabel(y_label_for_column(column))
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel(f"normalized time ({num_windows} averaged windows)")
    axes[0].legend(loc="upper right", ncol=2, fontsize=8)
    epoch_label = "full_run" if epoch < 0 else f"epoch_{epoch}"
    cpu_label = "ranked_cpu" if rank_cpu_cores else "physical_cpu"
    fig.suptitle(
        f"{device} local_ml {trial_id} {epoch_label} / {num_windows} averaged windows ({cpu_label})",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    output_path = (
        output_dir
        / f"windowed_raw_timeseries_{safe_name(device)}_{safe_name(trial_id)}_{epoch_label}_{num_windows}windows_{cpu_label}.png"
    )
    fig.savefig(output_path, dpi=180)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)
    return output_path


def plot_individual(
    window_df: pd.DataFrame,
    output_dir: Path,
    device: str,
    trial_id: str,
    epoch: int,
    num_windows: int,
    rank_cpu_cores: bool,
) -> None:
    epoch_label = "full_run" if epoch < 0 else f"epoch_{epoch}"
    cpu_label = "ranked_cpu" if rank_cpu_cores else "physical_cpu"
    for column in PLOT_COLUMNS:
        metric_name = DISPLAY_NAMES[column]
        sub = window_df[window_df["source_column"] == column]
        fig, ax = plt.subplots(figsize=(11, 4))
        for condition, cdf in sub.groupby("condition", sort=False):
            ax.plot(
                cdf["normalized_time"],
                cdf["value"],
                label=condition,
                color=COLORS.get(condition),
                linewidth=1.6,
                alpha=0.94,
            )
        ax.set_title(f"{device} {trial_id} {epoch_label} / {metric_name}")
        ax.set_xlabel(f"normalized time ({num_windows} averaged windows)")
        ax.set_ylabel(y_label_for_column(column))
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        output_path = (
            output_dir
            / f"windowed_raw_{safe_name(metric_name)}_{safe_name(device)}_{safe_name(trial_id)}_{epoch_label}_{num_windows}windows_{cpu_label}.png"
        )
        fig.savefig(output_path, dpi=180)
        fig.savefig(output_path.with_suffix(".pdf"))
        plt.close(fig)


def write_selection_summary(
    selected_runs: Dict[str, RunInfo],
    traces: Dict[str, pd.DataFrame],
    output_dir: Path,
    device: str,
    trial_id: str,
    epoch: int,
    num_windows: int,
) -> Path:
    rows = []
    for condition, run in selected_runs.items():
        df = traces[condition]
        rows.append(
            {
                "device": device,
                "trial_id": trial_id,
                "epoch": "full_run" if epoch < 0 else epoch,
                "num_windows": num_windows,
                "condition": condition,
                "attack_name": run.attack_name,
                "source_file": str(run.path),
                "n_raw_samples": len(df),
                "duration_seconds": float(df["relative_time"].max()),
            }
        )
    output_path = output_dir / f"windowed_raw_selection_{safe_name(device)}_{safe_name(trial_id)}.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = split_csv_list(args.conditions)
    selected_runs = choose_runs_by_condition(
        discover_runs(input_dir=input_dir, device=args.device, trial_id=args.trial_id),
        conditions,
    )

    rank_cpu_cores = not args.physical_cpu_cores
    traces: Dict[str, pd.DataFrame] = {}
    for condition, run in selected_runs.items():
        print(f"loading condition={condition} file={run.path.name}")
        traces[condition] = load_run_trace(
            run=run,
            epoch=args.epoch,
            include_idle=args.include_idle,
            rank_cpu_cores=rank_cpu_cores,
        )

    window_df = build_window_dataframe(
        traces=traces,
        selected_runs=selected_runs,
        num_windows=args.num_windows,
    )
    figure_path = plot_combined(
        window_df=window_df,
        output_dir=output_dir,
        device=args.device,
        trial_id=args.trial_id,
        epoch=args.epoch,
        num_windows=args.num_windows,
        rank_cpu_cores=rank_cpu_cores,
    )
    if args.save_individual:
        plot_individual(
            window_df=window_df,
            output_dir=output_dir,
            device=args.device,
            trial_id=args.trial_id,
            epoch=args.epoch,
            num_windows=args.num_windows,
            rank_cpu_cores=rank_cpu_cores,
        )
    if args.save_window_csv:
        csv_path = (
            output_dir
            / f"windowed_raw_values_{safe_name(args.device)}_{safe_name(args.trial_id)}_{args.num_windows}windows.csv"
        )
        window_df.to_csv(csv_path, index=False)
        print(f"saved windowed values: {csv_path}")

    summary_path = write_selection_summary(
        selected_runs=selected_runs,
        traces=traces,
        output_dir=output_dir,
        device=args.device,
        trial_id=args.trial_id,
        epoch=args.epoch,
        num_windows=args.num_windows,
    )
    print(f"saved figure: {figure_path}")
    print(f"saved selection summary: {summary_path}")


if __name__ == "__main__":
    main()
