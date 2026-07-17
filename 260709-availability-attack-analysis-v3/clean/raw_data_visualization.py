#!/usr/bin/env python3
"""Plot raw local-ML hardware traces for one device and one epoch.

The plots compare clean, label-flipping, unlearnable-examples, and shortcut
availability-attack runs on the same axes. CPU core columns are converted to
ranked core load per timestamp, matching the OT preprocessing: core 0 is the
highest-loaded core and core 3 is the lowest-loaded core. Context-switch and
minor-fault counters are converted to per-sample deltas before plotting.

The optional phase grids place hardware metrics in rows and batches in columns.
Separate forward and backward figures overlay clean, availability-shortcuts,
and unlearnable-examples traces using elapsed time relative to each phase.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

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

GRID_CONDITIONS = [
    "clean",
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
        description="Visualize raw local-ML hardware traces for one device/trial/epoch."
    )
    parser.add_argument("--input_dir", default="collected_logs")
    parser.add_argument("--output_dir", default="raw_data_visualization_result")
    parser.add_argument("--device", default="192.168.0.112")
    parser.add_argument("--trial_id", default="trial_0")
    parser.add_argument(
        "--epoch",
        type=int,
        default=0,
        help="Local-ML epoch to plot. Use a negative value to plot the full run.",
    )
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
        "--phase_filter",
        default="training",
        choices=["training", "evaluation", "all"],
        help=(
            "Rows to plot. training keeps forward/backward/optimizer_step; "
            "evaluation keeps evaluation rows; all keeps every non-idle phase."
        ),
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
        "--save_grid",
        action="store_true",
        help=(
            "Also save metric-by-batch grids for forward and backward phases."
        ),
    )
    parser.add_argument(
        "--grid_conditions",
        default=",".join(GRID_CONDITIONS),
        help="Comma-separated conditions overlaid in the phase grids.",
    )
    parser.add_argument(
        "--batch_start",
        type=int,
        default=0,
        help="First batch index to include in each grid (inclusive).",
    )
    parser.add_argument(
        "--batch_end",
        type=int,
        default=None,
        help="Last batch index to include in each grid (inclusive).",
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
    by_condition: Dict[str, List[RunInfo]] = {condition: [] for condition in conditions}
    for run in runs:
        if run.poisoning_method in by_condition:
            by_condition[run.poisoning_method].append(run)

    selected: Dict[str, RunInfo] = {}
    for condition in conditions:
        candidates = sorted(by_condition.get(condition, []), key=lambda item: item.timestamp_name)
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


def apply_counter_deltas_per_phase_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Convert cumulative counters to deltas independently inside each phase batch."""
    out = df.copy()
    for column in COUNTER_COLUMNS:
        values = pd.to_numeric(out[column], errors="coerce")
        deltas = values.groupby(out["phase_batch_index"], sort=False).diff().fillna(0.0)
        out[column] = deltas.mask(deltas < 0.0, 0.0)
    return out


def _assign_phase_batch_index(df: pd.DataFrame, phase: str) -> pd.DataFrame:
    """Assign batch IDs before discarding rows from the other phases."""
    out = df.sort_values("timestamp_unix").reset_index(drop=True).copy()
    is_selected_phase = out["phase"].astype(str).eq(phase)

    # The logger's batch_idx is more reliable when a short backward or optimizer
    # phase falls between two 10 Hz samples and is therefore not observed.
    batch_column = next(
        (
            column
            for column in ("batch_idx", "batch_index", "batch_id", "batch")
            if column in out.columns
        ),
        None,
    )
    if batch_column is not None:
        logged_batch = pd.to_numeric(out[batch_column], errors="coerce")
        if logged_batch[is_selected_phase].notna().all():
            out["phase_batch_index"] = logged_batch
        else:
            batch_column = None

    if batch_column is None:
        phase_start = is_selected_phase & ~is_selected_phase.shift(fill_value=False)
        out["phase_batch_index"] = phase_start.cumsum() - 1

    out = out[is_selected_phase].copy()
    if out.empty:
        return out
    out["phase_batch_index"] = out["phase_batch_index"].astype(int)
    return out


def load_phase_trace(
    run: RunInfo,
    epoch: int,
    rank_cpu_cores: bool,
    phase: str,
) -> pd.DataFrame:
    """Load one phase while preserving transitions used as batch boundaries."""
    if phase not in {"forward", "backward"}:
        raise ValueError(f"Unsupported grid phase: {phase}")
    df = pd.read_csv(run.path)
    required = ["timestamp_unix", "epoch", "phase", *PLOT_COLUMNS]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{run.path} is missing columns: {missing}")

    for column in ["timestamp_unix", "epoch", *PLOT_COLUMNS]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if not np.isfinite(df[["timestamp_unix", *PLOT_COLUMNS]].to_numpy(dtype=float)).all():
        raise ValueError(f"{run.path} contains NaN/inf in plotted columns.")

    if epoch >= 0:
        df = df[df["epoch"] == epoch].copy()
    if df.empty:
        raise ValueError(f"No rows found in {run.path} for epoch={epoch}.")

    if rank_cpu_cores:
        df = sort_cpu_cores_per_timestamp(df)
    df = _assign_phase_batch_index(df, phase=phase)
    if df.empty:
        raise ValueError(f"No {phase} rows found in {run.path} for epoch={epoch}.")

    df["phase_relative_time"] = (
        df["timestamp_unix"]
        - df.groupby("phase_batch_index")["timestamp_unix"].transform("min")
    )
    df = apply_counter_deltas_per_phase_batch(df)
    df["grid_phase"] = phase
    df["condition"] = run.poisoning_method
    df["attack_name"] = run.attack_name
    df["source_file"] = run.path.name
    return df.reset_index(drop=True)


def load_run_trace(
    run: RunInfo,
    epoch: int,
    include_idle: bool,
    rank_cpu_cores: bool,
    phase_filter: str,
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
    if phase_filter == "training":
        df = df[df["phase"].isin(["forward", "backward", "optimizer_step"])].copy()
    elif phase_filter == "evaluation":
        df = df[df["phase"] == "evaluation"].copy()
    if df.empty:
        raise ValueError(
            f"No rows remain after filtering {run.path} for epoch={epoch}, phase_filter={phase_filter}."
        )

    df = df.sort_values("timestamp_unix").reset_index(drop=True)
    df["relative_time"] = df["timestamp_unix"] - float(df["timestamp_unix"].iloc[0])
    df["condition"] = run.poisoning_method
    df["attack_name"] = run.attack_name
    df["source_file"] = run.path.name
    return df


def load_all_traces(
    selected_runs: Dict[str, RunInfo],
    epoch: int,
    include_idle: bool,
    rank_cpu_cores: bool,
    phase_filter: str,
) -> Dict[str, pd.DataFrame]:
    traces: Dict[str, pd.DataFrame] = {}
    for condition, run in selected_runs.items():
        print(f"loading condition={condition} file={run.path.name}")
        traces[condition] = load_run_trace(
            run=run,
            epoch=epoch,
            include_idle=include_idle,
            rank_cpu_cores=rank_cpu_cores,
            phase_filter=phase_filter,
        )
    return traces


def y_label_for_column(column: str) -> str:
    if column in CPU_CORE_COLUMNS or column == "system_memory_percent":
        return "percent"
    return "delta/sample"


def plot_combined(
    traces: Dict[str, pd.DataFrame],
    output_dir: Path,
    device: str,
    trial_id: str,
    epoch: int,
    rank_cpu_cores: bool,
    phase_filter: str,
) -> Path:
    n_rows = len(PLOT_COLUMNS)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2.45 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    for ax, column in zip(axes, PLOT_COLUMNS):
        for condition, df in traces.items():
            ax.plot(
                df["relative_time"],
                df[column],
                label=condition,
                color=COLORS.get(condition),
                linewidth=1.35,
                alpha=0.92,
            )
        ax.set_title(DISPLAY_NAMES[column], fontsize=10)
        ax.set_ylabel(y_label_for_column(column))
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("relative time (s)")
    axes[0].legend(loc="upper right", ncol=2, fontsize=8)
    epoch_label = "full_run" if epoch < 0 else f"epoch_{epoch}"
    cpu_label = "ranked_cpu" if rank_cpu_cores else "physical_cpu"
    fig.suptitle(f"{device} local_ml {trial_id} {epoch_label} {phase_filter} ({cpu_label})", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    output_path = (
        output_dir
        / f"raw_timeseries_{safe_name(device)}_{safe_name(trial_id)}_{epoch_label}_{safe_name(phase_filter)}_{cpu_label}.png"
    )
    fig.savefig(output_path, dpi=180)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)
    return output_path


def plot_individual(
    traces: Dict[str, pd.DataFrame],
    output_dir: Path,
    device: str,
    trial_id: str,
    epoch: int,
    rank_cpu_cores: bool,
    phase_filter: str,
) -> None:
    epoch_label = "full_run" if epoch < 0 else f"epoch_{epoch}"
    cpu_label = "ranked_cpu" if rank_cpu_cores else "physical_cpu"
    for column in PLOT_COLUMNS:
        fig, ax = plt.subplots(figsize=(11, 4))
        for condition, df in traces.items():
            ax.plot(
                df["relative_time"],
                df[column],
                label=condition,
                color=COLORS.get(condition),
                linewidth=1.45,
                alpha=0.92,
            )
        ax.set_title(f"{device} {trial_id} {epoch_label} {phase_filter} / {DISPLAY_NAMES[column]}")
        ax.set_xlabel("relative time (s)")
        ax.set_ylabel(y_label_for_column(column))
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        output_path = (
            output_dir
            / (
                f"raw_{safe_name(DISPLAY_NAMES[column])}_{safe_name(device)}_{safe_name(trial_id)}_"
                f"{epoch_label}_{safe_name(phase_filter)}_{cpu_label}.png"
            )
        )
        fig.savefig(output_path, dpi=180)
        fig.savefig(output_path.with_suffix(".pdf"))
        plt.close(fig)


def plot_phase_grid(
    traces: Dict[str, pd.DataFrame],
    output_dir: Path,
    device: str,
    trial_id: str,
    epoch: int,
    rank_cpu_cores: bool,
    batch_start: int,
    batch_end: Optional[int],
    phase: str,
) -> Path:
    """Plot metrics by row and batches by column for one training phase."""
    if phase not in {"forward", "backward"}:
        raise ValueError(f"Unsupported grid phase: {phase}")
    if epoch < 0:
        raise ValueError("Phase grids require one epoch; set --epoch to 0 or greater.")
    if batch_start < 0:
        raise ValueError("--batch_start must be zero or greater.")
    if batch_end is not None and batch_end < batch_start:
        raise ValueError("--batch_end must be greater than or equal to --batch_start.")
    if not traces:
        raise ValueError(f"No {phase} traces were loaded.")

    common_batches: Optional[set[int]] = None
    for df in traces.values():
        observed = set(df["phase_batch_index"].astype(int).unique())
        common_batches = observed if common_batches is None else common_batches & observed

    batch_ids = sorted(
        int(batch_id)
        for batch_id in (common_batches or set())
        if batch_id >= batch_start and (batch_end is None or batch_id <= batch_end)
    )
    if not batch_ids:
        counts = {
            condition: sorted(df["phase_batch_index"].astype(int).unique().tolist())
            for condition, df in traces.items()
        }
        raise ValueError(
            f"No common {phase} batches remain after applying the requested range. "
            f"Observed batches by condition: {counts}"
        )

    n_metrics = len(PLOT_COLUMNS)
    n_batches = len(batch_ids)
    fig, axes = plt.subplots(
        n_metrics,
        n_batches,
        figsize=(3.2 * n_batches, 2.2 * n_metrics),
        squeeze=False,
        sharex=False,
        sharey="row",
    )

    for row_index, metric in enumerate(PLOT_COLUMNS):
        for column_index, batch_id in enumerate(batch_ids):
            ax = axes[row_index, column_index]
            for condition, df in traces.items():
                batch_df = df[df["phase_batch_index"] == batch_id]
                if batch_df.empty:
                    continue
                ax.plot(
                    batch_df["phase_relative_time"],
                    batch_df[metric],
                    label=condition,
                    color=COLORS.get(condition),
                    linewidth=1.2,
                    alpha=0.9,
                )

            if row_index == 0:
                ax.set_title(f"batch {batch_id}", fontsize=9)
            if column_index == 0:
                label = DISPLAY_NAMES[metric]
                if not rank_cpu_cores and metric in CPU_CORE_COLUMNS:
                    label = metric.replace("system_", "") + "_physical"
                ax.set_ylabel(label, fontsize=8)
            if row_index == n_metrics - 1:
                ax.set_xlabel("elapsed time (s)", fontsize=8)
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=7)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(labels),
            fontsize=9,
            bbox_to_anchor=(0.5, 0.995),
        )

    cpu_label = "ranked_cpu" if rank_cpu_cores else "physical_cpu"
    fig.suptitle(
        f"{device} local_ml {trial_id} epoch_{epoch} {phase} batches ({cpu_label})",
        y=1.01,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.975])

    range_label = f"batches_{batch_ids[0]}-{batch_ids[-1]}"
    output_path = output_dir / (
        f"{phase}_grid_{safe_name(device)}_{safe_name(trial_id)}_epoch_{epoch}_"
        f"{range_label}_{cpu_label}.png"
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"{phase} grid common batches: {batch_ids}")
    return output_path


def write_selection_summary(
    selected_runs: Dict[str, RunInfo],
    traces: Dict[str, pd.DataFrame],
    output_dir: Path,
    device: str,
    trial_id: str,
    epoch: int,
    phase_filter: str,
) -> Path:
    rows = []
    for condition, run in selected_runs.items():
        df = traces[condition]
        rows.append(
            {
                "device": device,
                "trial_id": trial_id,
                "epoch": "full_run" if epoch < 0 else epoch,
                "phase_filter": phase_filter,
                "condition": condition,
                "attack_name": run.attack_name,
                "source_file": str(run.path),
                "n_samples": len(df),
                "duration_seconds": float(df["relative_time"].max()),
            }
        )
    output_path = (
        output_dir
        / f"raw_timeseries_selection_{safe_name(device)}_{safe_name(trial_id)}_{safe_name(phase_filter)}.csv"
    )
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = split_csv_list(args.conditions)
    runs = discover_runs(input_dir=input_dir, device=args.device, trial_id=args.trial_id)
    selected_runs = choose_runs_by_condition(runs, conditions)
    traces = load_all_traces(
        selected_runs=selected_runs,
        epoch=args.epoch,
        include_idle=args.include_idle,
        rank_cpu_cores=not args.physical_cpu_cores,
        phase_filter=args.phase_filter,
    )

    figure_path = plot_combined(
        traces=traces,
        output_dir=output_dir,
        device=args.device,
        trial_id=args.trial_id,
        epoch=args.epoch,
        rank_cpu_cores=not args.physical_cpu_cores,
        phase_filter=args.phase_filter,
    )
    if args.save_individual:
        plot_individual(
            traces=traces,
            output_dir=output_dir,
            device=args.device,
            trial_id=args.trial_id,
            epoch=args.epoch,
            rank_cpu_cores=not args.physical_cpu_cores,
            phase_filter=args.phase_filter,
        )
    summary_path = write_selection_summary(
        selected_runs=selected_runs,
        traces=traces,
        output_dir=output_dir,
        device=args.device,
        trial_id=args.trial_id,
        epoch=args.epoch,
        phase_filter=args.phase_filter,
    )
    print(f"saved figure: {figure_path}")
    print(f"saved selection summary: {summary_path}")

    if args.save_grid:
        grid_conditions = split_csv_list(args.grid_conditions)
        grid_selected_runs = choose_runs_by_condition(runs, grid_conditions)
        for phase in ("forward", "backward"):
            phase_traces = {
                condition: load_phase_trace(
                    run=run,
                    epoch=args.epoch,
                    rank_cpu_cores=not args.physical_cpu_cores,
                    phase=phase,
                )
                for condition, run in grid_selected_runs.items()
            }
            phase_figure_path = plot_phase_grid(
                traces=phase_traces,
                output_dir=output_dir,
                device=args.device,
                trial_id=args.trial_id,
                epoch=args.epoch,
                rank_cpu_cores=not args.physical_cpu_cores,
                batch_start=args.batch_start,
                batch_end=args.batch_end,
                phase=phase,
            )
            print(f"saved {phase} grid: {phase_figure_path}")


if __name__ == "__main__":
    main()
