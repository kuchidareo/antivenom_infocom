#!/usr/bin/env python3
"""Plot raw hardware and perf data for matched RPi and Jetson CPU runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "collected_logs" / "logs"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "visualization"

DEVICES = ("192.168.0.112", "192.168.0.141", "cloudlab_test")
METHODS = ("clean", "unlearnable_examples", "availability_shortcuts")
SEQUENCE_METHODS = ("clean", "availability_shortcuts")
METHOD_LABELS = {
    "clean": "Clean",
    "unlearnable_examples": "Unlearnable examples",
    "availability_shortcuts": "Availability shortcuts",
}
METHOD_COLORS = {
    "clean": "#202124",
    "unlearnable_examples": "#d1495b",
    "availability_shortcuts": "#00798c",
}


@dataclass(frozen=True)
class MetricSpec:
    label: str
    source: str
    columns: Mapping[str, str]


PERF_METRICS = (
    MetricSpec("CPU cycles", "perf", {DEVICES[0]: "perf_cycles", DEVICES[1]: "perf_cycles"}),
    MetricSpec("Instructions", "perf", {DEVICES[0]: "perf_instructions", DEVICES[1]: "perf_instructions"}),
    MetricSpec("Task clock (ms)", "perf", {DEVICES[0]: "perf_task_clock", DEVICES[1]: "perf_task_clock"}),
    MetricSpec("Context switches", "perf", {DEVICES[0]: "perf_context_switches", DEVICES[1]: "perf_context_switches"}),
    MetricSpec("CPU migrations", "perf", {DEVICES[0]: "perf_cpu_migrations", DEVICES[1]: "perf_cpu_migrations"}),
    MetricSpec("Page faults", "perf", {DEVICES[0]: "perf_page_faults", DEVICES[1]: "perf_page_faults"}),
    MetricSpec("Branch operations", "perf", {DEVICES[0]: "perf_branches", DEVICES[1]: "perf_br_retired"}),
    MetricSpec("Branch misses", "perf", {DEVICES[0]: "perf_branch_misses", DEVICES[1]: "perf_br_mis_pred_retired"}),
    MetricSpec("L1D read / access", "perf", {DEVICES[0]: "perf_l1d_cache_rd", DEVICES[1]: "perf_l1d_cache"}),
    MetricSpec("L1D read refill / refill", "perf", {DEVICES[0]: "perf_l1d_cache_refill_rd", DEVICES[1]: "perf_l1d_cache_refill"}),
    MetricSpec("L1D write access", "perf", {DEVICES[0]: "perf_l1d_cache_wr"}),
    MetricSpec("L1D write refill", "perf", {DEVICES[0]: "perf_l1d_cache_refill_wr"}),
    MetricSpec("L1D writeback", "perf", {DEVICES[1]: "perf_l1d_cache_wb"}),
    MetricSpec("L2D read / access", "perf", {DEVICES[0]: "perf_l2d_cache_rd", DEVICES[1]: "perf_l2d_cache"}),
    MetricSpec("L2D read refill / refill", "perf", {DEVICES[0]: "perf_l2d_cache_refill_rd", DEVICES[1]: "perf_l2d_cache_refill"}),
    MetricSpec("L2D write access", "perf", {DEVICES[0]: "perf_l2d_cache_wr"}),
    MetricSpec("L2D write refill", "perf", {DEVICES[0]: "perf_l2d_cache_refill_wr"}),
    MetricSpec("L2D writeback", "perf", {DEVICES[1]: "perf_l2d_cache_wb"}),
    MetricSpec("Bus read / access", "perf", {DEVICES[0]: "perf_bus_access_rd", DEVICES[1]: "perf_bus_access"}),
    MetricSpec("Bus write access", "perf", {DEVICES[0]: "perf_bus_access_wr"}),
    MetricSpec("Memory accesses", "perf", {DEVICES[0]: "perf_mem_access", DEVICES[1]: "perf_mem_access"}),
    MetricSpec("Advanced SIMD (ASE_SPEC)", "perf", {DEVICES[0]: "perf_ase_spec"}),
    MetricSpec("Floating point (VFP_SPEC)", "perf", {DEVICES[0]: "perf_vfp_spec"}),
    MetricSpec("Speculative instructions", "perf", {DEVICES[0]: "perf_inst_spec", DEVICES[1]: "perf_inst_spec"}),
)

ELAPSED_METRICS = (
    MetricSpec("Phase elapsed time (s)", "perf", {device: "phase_elapsed_sec" for device in DEVICES}),
)

HARDWARE_METRICS = (
    MetricSpec("System CPU average (%)", "main", {device: "system_cpu_average_percent" for device in DEVICES}),
    MetricSpec("System CPU core 0 (%)", "main", {device: "system_cpu_core_0" for device in DEVICES}),
    MetricSpec("System CPU core 1 (%)", "main", {device: "system_cpu_core_1" for device in DEVICES}),
    MetricSpec("System CPU core 2 (%)", "main", {device: "system_cpu_core_2" for device in DEVICES}),
    MetricSpec("System CPU core 3 (%)", "main", {device: "system_cpu_core_3" for device in DEVICES}),
    MetricSpec("Process CPU (%)", "main", {device: "process_cpu_percent" for device in DEVICES}),
    MetricSpec("System memory (%)", "main", {device: "system_memory_percent" for device in DEVICES}),
    MetricSpec("System memory used (bytes)", "main", {device: "system_memory_used" for device in DEVICES}),
    MetricSpec("System memory available (bytes)", "main", {device: "system_memory_available" for device in DEVICES}),
    MetricSpec("Process RSS (bytes)", "main", {device: "process_memory_rss" for device in DEVICES}),
    MetricSpec("Process VMS (bytes)", "main", {device: "process_memory_vms" for device in DEVICES}),
    MetricSpec("Process memory (%)", "main", {device: "process_memory_percent" for device in DEVICES}),
)

METRICS = PERF_METRICS + HARDWARE_METRICS + ELAPSED_METRICS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-time-epoch", type=int, default=0)
    parser.add_argument(
        "--bin-ms",
        type=float,
        default=0.0,
        help="Batch-time alignment width in ms. The default 0 uses each file's native interval.",
    )
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def main_csv_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*.csv")
        if not path.name.endswith(("_metrics.csv", "_perf.csv"))
    )


def add_derived_columns(frame: pd.DataFrame, run_start: float) -> pd.DataFrame:
    frame = frame.copy()
    timestamps = pd.to_numeric(frame["timestamp_unix"], errors="coerce")
    frame["elapsed_sec"] = timestamps - run_start
    phase_keys = ["epoch", "batch_idx", "phase"]
    if "perf_interval_ms" in frame.columns and all(column in frame.columns for column in phase_keys):
        phase_span = timestamps.groupby([frame[column] for column in phase_keys]).transform(
            lambda values: values.max() - values.min()
        )
        interval_sec = pd.to_numeric(frame["perf_interval_ms"], errors="coerce") / 1000.0
        frame["phase_elapsed_sec"] = phase_span + interval_sec
    cpu_columns = [
        column
        for column in (f"system_cpu_core_{index}" for index in range(4))
        if column in frame.columns
    ]
    if cpu_columns:
        frame["system_cpu_average_percent"] = frame[cpu_columns].apply(
            pd.to_numeric, errors="coerce"
        ).mean(axis=1)
    return frame


def load_runs(input_dir: Path) -> dict[tuple[str, str], dict[str, pd.DataFrame]]:
    candidates: dict[tuple[str, str], tuple[Path, pd.DataFrame]] = {}
    for path in main_csv_files(input_dir):
        frame = pd.read_csv(path, low_memory=False)
        if frame.empty:
            continue
        device = str(frame.iloc[0].get("device_id", ""))
        method = str(frame.iloc[0].get("poisoning_method", ""))
        if device not in DEVICES or method not in METHODS:
            continue
        key = (device, method)
        if key not in candidates or path.name > candidates[key][0].name:
            candidates[key] = (path, frame)

    expected = {(device, method) for device in DEVICES for method in METHODS}
    missing = sorted(expected - candidates.keys())
    if missing:
        raise FileNotFoundError(f"Missing matched runs: {missing}")

    runs: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}
    for key, (main_path, main_frame) in candidates.items():
        perf_path = main_path.with_name(f"{main_path.stem}_perf.csv")
        if not perf_path.exists():
            raise FileNotFoundError(f"Missing perf CSV: {perf_path}")
        run_start = float(pd.to_numeric(main_frame["timestamp_unix"], errors="coerce").min())
        runs[key] = {
            "main": add_derived_columns(main_frame, run_start),
            "perf": add_derived_columns(pd.read_csv(perf_path, low_memory=False), run_start),
        }
    return runs


def sequence_csv_files(input_dir: Path) -> list[Path]:
    files = []
    for path in main_csv_files(input_dir):
        columns = pd.read_csv(path, nrows=0).columns
        if "training_sequence" in columns:
            files.append(path)
    return files


def load_sequence_runs(input_dir: Path) -> dict[str, dict[str, object]]:
    candidates: dict[str, tuple[Path, pd.DataFrame]] = {}
    for path in sequence_csv_files(input_dir):
        frame = pd.read_csv(path, low_memory=False)
        if frame.empty:
            continue
        sequence = str(frame.iloc[0].get("training_sequence", ""))
        if not sequence:
            continue
        if sequence not in candidates or path.name > candidates[sequence][0].name:
            candidates[sequence] = (path, frame)

    if not candidates:
        raise FileNotFoundError(f"No sequence runs found below {input_dir}")

    runs: dict[str, dict[str, object]] = {}
    for sequence, (main_path, main_frame) in sorted(candidates.items()):
        perf_path = main_path.with_name(f"{main_path.stem}_perf.csv")
        if not perf_path.exists():
            raise FileNotFoundError(f"Missing perf CSV: {perf_path}")
        run_start = float(pd.to_numeric(main_frame["timestamp_unix"], errors="coerce").min())
        runs[sequence] = {
            "device": str(main_frame.iloc[0].get("device_id", "")),
            "main": add_derived_columns(main_frame, run_start),
            "perf": add_derived_columns(pd.read_csv(perf_path, low_memory=False), run_start),
        }
    return runs


def is_sequence_input(input_dir: Path) -> bool:
    return bool(sequence_csv_files(input_dir))


def summarize_epochs(
    frame: pd.DataFrame,
    column: str,
    phase: str,
    normalize_per_instruction: bool,
) -> pd.DataFrame:
    required = ["epoch", "batch_idx", "phase", column]
    if normalize_per_instruction:
        required.append("perf_instructions")
    if any(name not in frame.columns for name in required):
        return pd.DataFrame()

    selected = frame.loc[
        frame["phase"].eq(phase),
        list(dict.fromkeys(required)),
    ].copy()
    selected["epoch"] = pd.to_numeric(selected["epoch"], errors="coerce")
    selected["batch_idx"] = pd.to_numeric(selected["batch_idx"], errors="coerce")
    selected["value"] = pd.to_numeric(selected[column], errors="coerce")
    if normalize_per_instruction:
        instructions = pd.to_numeric(selected["perf_instructions"], errors="coerce")
        selected["value"] = selected["value"] / instructions.where(instructions > 0)

    selected = selected.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["epoch", "batch_idx", "value"]
    )
    if selected.empty:
        return pd.DataFrame()

    per_batch = (
        selected.groupby(["epoch", "batch_idx"], as_index=False)["value"]
        .mean()
        .sort_values(["epoch", "batch_idx"])
    )
    return per_batch.groupby("epoch")["value"].agg(
        mean="mean", std="std", batches="count"
    ).reset_index()


def summarize_batch_time(
    frame: pd.DataFrame,
    column: str,
    phase: str,
    epoch: int,
    bin_sec: float,
    normalize_per_instruction: bool,
) -> pd.DataFrame:
    required = ["timestamp_unix", "epoch", "batch_idx", "phase", column]
    if normalize_per_instruction:
        required.append("perf_instructions")
    if any(name not in frame.columns for name in required):
        return pd.DataFrame()

    selected = frame.loc[
        frame["phase"].eq(phase) & pd.to_numeric(frame["epoch"], errors="coerce").eq(epoch),
        list(dict.fromkeys(required)),
    ].copy()
    selected["timestamp_unix"] = pd.to_numeric(selected["timestamp_unix"], errors="coerce")
    selected["batch_idx"] = pd.to_numeric(selected["batch_idx"], errors="coerce")
    selected["value"] = pd.to_numeric(selected[column], errors="coerce")
    if normalize_per_instruction:
        instructions = pd.to_numeric(selected["perf_instructions"], errors="coerce")
        selected["value"] = selected["value"] / instructions.where(instructions > 0)
    selected = selected.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["timestamp_unix", "batch_idx", "value"]
    )
    if selected.empty:
        return pd.DataFrame()

    if bin_sec <= 0:
        if "perf_interval_ms" in frame.columns:
            native_ms = pd.to_numeric(frame["perf_interval_ms"], errors="coerce").dropna()
            bin_sec = float(native_ms.median() / 1000.0)
        else:
            ordered = selected.sort_values(["batch_idx", "timestamp_unix"])
            native_deltas = ordered.groupby("batch_idx")["timestamp_unix"].diff()
            native_deltas = native_deltas[native_deltas > 0]
            if native_deltas.empty:
                return pd.DataFrame()
            bin_sec = float(native_deltas.median())

    selected["batch_time_sec"] = selected["timestamp_unix"] - selected.groupby(
        "batch_idx"
    )["timestamp_unix"].transform("min")
    selected["time_sec"] = (selected["batch_time_sec"] / bin_sec).round() * bin_sec
    per_batch = selected.groupby(["batch_idx", "time_sec"], as_index=False)["value"].mean()
    return per_batch.groupby("time_sec")["value"].agg(
        mean="mean", std="std", batches="count"
    ).reset_index()


def plot_metric(
    axis: plt.Axes,
    runs: dict[tuple[str, str], dict[str, pd.DataFrame]],
    device: str,
    phase: str,
    metric: MetricSpec,
    normalize_per_instruction: bool,
    x_mode: str,
    batch_time_epoch: int,
    bin_sec: float,
) -> None:
    column = metric.columns.get(device)
    normalize_metric = normalize_per_instruction and column != "phase_elapsed_sec"
    plotted = False
    if column is not None:
        for method in METHODS:
            frame = runs[(device, method)][metric.source]
            if x_mode == "epoch":
                summary = summarize_epochs(frame, column, phase, normalize_metric)
                x_column = "epoch"
            else:
                summary = summarize_batch_time(
                    frame,
                    column,
                    phase,
                    batch_time_epoch,
                    bin_sec,
                    normalize_metric,
                )
                x_column = "time_sec"
            if summary.empty:
                continue
            color = METHOD_COLORS[method]
            axis.fill_between(
                summary[x_column],
                summary["mean"] - summary["std"].fillna(0),
                summary["mean"] + summary["std"].fillna(0),
                color=color,
                alpha=0.14,
                linewidth=0,
            )
            axis.plot(
                summary[x_column],
                summary["mean"],
                color=color,
                linewidth=1.15,
                marker="o",
                markersize=1.8,
                alpha=0.95,
            )
            plotted = True

    if not plotted:
        axis.text(0.5, 0.5, "N/A", transform=axis.transAxes, ha="center", va="center", color="#777777")
        axis.set_xticks([])
        axis.set_yticks([])

    label = f"{metric.label} / instruction" if normalize_metric else metric.label
    axis.set_title(f"{label} | {device}", fontsize=8, pad=3)
    if x_mode == "epoch":
        axis.set_xlabel("Epoch", fontsize=7)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        axis.set_xlabel("Time within batch phase (s)", fontsize=7)
    axis.tick_params(axis="both", labelsize=6)
    axis.grid(True, color="#d9d9d9", linewidth=0.4, alpha=0.7)
    axis.margins(x=0.02, y=0.08)


def save_phase_figure(
    runs: dict[tuple[str, str], dict[str, pd.DataFrame]],
    phase: str,
    output_dir: Path,
    dpi: int,
    normalize_per_instruction: bool = False,
    x_mode: str = "epoch",
    batch_time_epoch: int = 0,
    bin_sec: float = 0.1,
) -> Path:
    metrics = PERF_METRICS + ELAPSED_METRICS if normalize_per_instruction else METRICS
    figure, axes = plt.subplots(
        nrows=len(metrics),
        ncols=len(DEVICES),
        figsize=(18, len(metrics) * 2.15),
        sharex=False,
        sharey=False,
        squeeze=False,
    )
    for row, metric in enumerate(metrics):
        for column, device in enumerate(DEVICES):
            plot_metric(
                axes[row, column],
                runs,
                device,
                phase,
                metric,
                normalize_per_instruction,
                x_mode,
                batch_time_epoch,
                bin_sec,
            )

    handles = [
        Line2D([0], [0], color=METHOD_COLORS[method], linewidth=1.5, label=METHOD_LABELS[method])
        for method in METHODS
    ]
    handles.append(Patch(facecolor="#777777", alpha=0.18, label="Mean +/- 1 SD across batch averages"))
    value_kind = "Perf Metrics per Instruction" if normalize_per_instruction else "Raw Metrics"
    comparison = (
        "Epoch Means and Batch Standard Deviation"
        if x_mode == "epoch"
        else f"Epoch {batch_time_epoch} Batch-Time Mean and Standard Deviation"
    )
    figure.suptitle(
        f"{value_kind}: {phase.capitalize()} | {comparison}",
        fontsize=16,
        y=0.999,
    )
    figure.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 0.997))
    figure.tight_layout(rect=(0.015, 0.005, 0.995, 0.992), h_pad=1.0, w_pad=1.2)

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "metrics_per_instruction" if normalize_per_instruction else "raw_metrics"
    suffix = "" if x_mode == "epoch" else f"_epoch{batch_time_epoch}_batch_time"
    output_path = output_dir / f"{prefix}_{phase}{suffix}.png"
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path


def sequence_label(sequence: str) -> str:
    labels = {
        "clean_to_availability_shortcuts": "Clean -> Availability shortcuts",
        "availability_shortcuts_to_clean": "Availability shortcuts -> Clean",
    }
    return labels.get(sequence, sequence.replace("_to_", " -> ").replace("_", " ").title())


def plot_sequence_metric(
    axis: plt.Axes,
    run: dict[str, object],
    sequence: str,
    phase: str,
    metric: MetricSpec,
    normalize_per_instruction: bool,
    x_mode: str,
    batch_time_epoch: int,
    bin_sec: float,
) -> None:
    device = str(run["device"])
    column = metric.columns.get(device)
    normalize_metric = normalize_per_instruction and column != "phase_elapsed_sec"
    plotted = False

    if column is not None:
        frame = run[metric.source]
        assert isinstance(frame, pd.DataFrame)
        method_column = (
            "input_poisoning_method"
            if "input_poisoning_method" in frame.columns
            else "poisoning_method"
        )
        for method in SEQUENCE_METHODS:
            method_frame = frame.loc[frame[method_column].eq(method)]
            if x_mode == "epoch":
                summary = summarize_epochs(method_frame, column, phase, normalize_metric)
                x_column = "epoch"
            else:
                summary = summarize_batch_time(
                    method_frame,
                    column,
                    phase,
                    batch_time_epoch,
                    bin_sec,
                    normalize_metric,
                )
                x_column = "time_sec"
            if summary.empty:
                continue
            color = METHOD_COLORS[method]
            axis.fill_between(
                summary[x_column],
                summary["mean"] - summary["std"].fillna(0),
                summary["mean"] + summary["std"].fillna(0),
                color=color,
                alpha=0.14,
                linewidth=0,
            )
            axis.plot(
                summary[x_column],
                summary["mean"],
                color=color,
                linewidth=1.15,
                marker="o",
                markersize=2.2,
                alpha=0.95,
            )
            plotted = True

    if not plotted:
        axis.text(
            0.5,
            0.5,
            "N/A",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color="#777777",
        )
        axis.set_xticks([])
        axis.set_yticks([])

    label = f"{metric.label} / instruction" if normalize_metric else metric.label
    axis.set_title(f"{label} | {sequence_label(sequence)} | {device}", fontsize=8, pad=3)
    if x_mode == "epoch":
        axis.axvline(9.5, color="#777777", linestyle="--", linewidth=0.8, alpha=0.8)
        axis.set_xlabel("Global epoch", fontsize=7)
        axis.set_xlim(-0.4, 19.4)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        axis.set_xlabel("Time within batch phase (s)", fontsize=7)
    axis.tick_params(axis="both", labelsize=6)
    axis.grid(True, color="#d9d9d9", linewidth=0.4, alpha=0.7)
    axis.margins(x=0.02, y=0.08)


def save_sequence_phase_figure(
    runs: dict[str, dict[str, object]],
    phase: str,
    output_dir: Path,
    dpi: int,
    normalize_per_instruction: bool = False,
    x_mode: str = "epoch",
    batch_time_epoch: int = 0,
    bin_sec: float = 0.1,
) -> Path:
    metrics = PERF_METRICS + ELAPSED_METRICS if normalize_per_instruction else METRICS
    sequences = list(runs)
    figure, axes = plt.subplots(
        nrows=len(metrics),
        ncols=len(sequences),
        figsize=(9 * len(sequences), len(metrics) * 2.15),
        sharex=False,
        sharey=False,
        squeeze=False,
    )
    for row, metric in enumerate(metrics):
        for column_index, sequence in enumerate(sequences):
            plot_sequence_metric(
                axes[row, column_index],
                runs[sequence],
                sequence,
                phase,
                metric,
                normalize_per_instruction,
                x_mode,
                batch_time_epoch,
                bin_sec,
            )

    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            linewidth=1.5,
            label=METHOD_LABELS[method],
        )
        for method in SEQUENCE_METHODS
    ]
    handles.append(Patch(facecolor="#777777", alpha=0.18, label="Mean +/- 1 SD across batch averages"))
    if x_mode == "epoch":
        handles.append(Line2D([0], [0], color="#777777", linestyle="--", linewidth=0.8, label="Stage boundary"))
    value_kind = "Perf Metrics per Instruction" if normalize_per_instruction else "Raw Metrics"
    comparison = (
        "20-Epoch Sequence Means and Batch Standard Deviation"
        if x_mode == "epoch"
        else f"Global Epoch {batch_time_epoch} Batch-Time Mean and Standard Deviation"
    )
    figure.suptitle(f"{value_kind}: {phase.capitalize()} | {comparison}", fontsize=16, y=0.999)
    figure.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        bbox_to_anchor=(0.5, 0.997),
    )
    figure.tight_layout(rect=(0.015, 0.005, 0.995, 0.992), h_pad=1.0, w_pad=1.2)

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "metrics_per_instruction" if normalize_per_instruction else "raw_metrics"
    suffix = "" if x_mode == "epoch" else f"_epoch{batch_time_epoch}_batch_time"
    output_path = output_dir / f"{prefix}_{phase}{suffix}.png"
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path


def main() -> None:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    if args.batch_time_epoch < 0 or args.bin_ms < 0:
        raise ValueError("--batch-time-epoch and --bin-ms must be non-negative")
    input_dir = args.input_dir.resolve()
    sequence_mode = is_sequence_input(input_dir)
    runs = load_sequence_runs(input_dir) if sequence_mode else load_runs(input_dir)
    for phase in ("forward", "backward"):
        for normalize_per_instruction in (False, True):
            for x_mode in ("epoch", "batch_time"):
                save_function = save_sequence_phase_figure if sequence_mode else save_phase_figure
                path = save_function(
                    runs,
                    phase,
                    args.output_dir.resolve(),
                    args.dpi,
                    normalize_per_instruction,
                    x_mode,
                    args.batch_time_epoch,
                    args.bin_ms / 1000.0,
                )
                print(f"Saved {path}")


if __name__ == "__main__":
    main()
