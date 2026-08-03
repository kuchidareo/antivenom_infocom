"""Visualize controlled entropy metadata and replay-only PMU measurements."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/antivenom-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.ticker import ScalarFormatter


REGIMES = ("low", "mid", "high")
OPERATORS = ("relu", "maxpool", "conv")
TEMPORALS = ("stable", "changing")
COLORS = {"low": "#2878B5", "mid": "#E07A1F", "high": "#C23B4A"}
PANEL_COLORS = {"stable": "#2878B5", "changing": "#C23B4A"}
OPERATOR_COLORS = {"relu": "#2878B5", "maxpool": "#C23B4A", "conv": "#2F855A"}
OPERATOR_MARKERS = {"relu": "o", "maxpool": "s", "conv": "^"}
OPERATOR_LABELS = {"relu": "ReLU", "maxpool": "MaxPool", "conv": "Conv2D"}

PERF_METADATA_COLUMNS = {
    "perf_pid",
    "perf_measurement_mode",
    "perf_scope",
    "perf_elapsed_sec",
    "perf_interval_ms",
    "perf_phase_start_timestamp",
    "perf_phase_start_unix",
    "perf_phase_end_timestamp",
    "perf_phase_end_unix",
    "perf_phase_duration_sec",
    "perf_events",
    "perf_status",
    "perf_error",
}

METRIC_LABELS = {
    "perf_cycles": "Cycles",
    "perf_instructions": "Instructions",
    "perf_task_clock": "Task clock (ms)",
    "perf_context_switches": "Context switches",
    "perf_cpu_migrations": "CPU migrations",
    "perf_page_faults": "Page faults",
    "perf_branches": "Branches",
    "perf_branch_misses": "Branch misses",
    "perf_br_retired": "Retired branches",
    "perf_br_mis_pred_retired": "Mispredicted retired branches",
    "perf_l1_dcache_loads": "L1D loads",
    "perf_l1_dcache_load_misses": "L1D load misses",
    "perf_l1d_cache": "L1D accesses",
    "perf_l1d_cache_refill": "L1D refills",
    "perf_l1d_cache_wb": "L1D writebacks",
    "perf_l1d_cache_rd": "L1D read accesses",
    "perf_l1d_cache_refill_rd": "L1D read refills",
    "perf_l1d_cache_wr": "L1D write accesses",
    "perf_l1d_cache_refill_wr": "L1D write refills",
    "perf_l2d_cache": "L2D accesses",
    "perf_l2d_cache_refill": "L2D refills",
    "perf_l2d_cache_wb": "L2D writebacks",
    "perf_l2d_cache_rd": "L2D read accesses",
    "perf_l2d_cache_refill_rd": "L2D read refills",
    "perf_l2d_cache_wr": "L2D write accesses",
    "perf_l2d_cache_refill_wr": "L2D write refills",
    "perf_bus_access": "Bus accesses",
    "perf_bus_access_rd": "Bus read accesses",
    "perf_bus_access_wr": "Bus write accesses",
    "perf_mem_access": "Memory accesses",
    "perf_ase_spec": "ASE speculative operations",
    "perf_vfp_spec": "VFP speculative operations",
    "perf_inst_spec": "Speculative operations",
}

ENTROPY_METRICS = (
    "entropy_input_activation_rate",
    "entropy_input_width_conditional_entropy_bits",
    "entropy_input_height_conditional_entropy_bits",
    "entropy_input_channel_conditional_entropy_bits",
    "entropy_input_nchw_memory_conditional_entropy_bits",
    "entropy_input_conv3x3_patch_conditional_entropy_bits",
    "entropy_input_width_flip_rate",
    "entropy_input_height_flip_rate",
    "entropy_input_channel_flip_rate",
    "entropy_input_nchw_memory_flip_rate",
    "entropy_input_conv3x3_patch_flip_rate",
    "entropy_conv_patch_active_count_std",
    "entropy_conv_patch_active_count_entropy_bits",
    "entropy_conv_exact_patch_entropy_bits",
    "entropy_conv_exact_patch_unique_fraction",
    "entropy_conv_exact_patch_collision_probability",
    "entropy_conv_duplicate_patch_occurrence_fraction",
    "entropy_conv_adjacent_exact_patch_same_fraction",
    "entropy_conv_adjacent_patch_jaccard",
    "entropy_conv_patch_stream_conditional_entropy_bits",
    "entropy_conv_patch_stream_flip_rate",
)

ENTROPY_LABELS = {
    "entropy_input_activation_rate": "Input activation rate",
    "entropy_input_width_conditional_entropy_bits": "Width transition entropy (bits)",
    "entropy_input_height_conditional_entropy_bits": "Height transition entropy (bits)",
    "entropy_input_channel_conditional_entropy_bits": "Channel transition entropy (bits)",
    "entropy_input_nchw_memory_conditional_entropy_bits": "NCHW transition entropy (bits)",
    "entropy_input_conv3x3_patch_conditional_entropy_bits": "3x3 patch transition entropy (bits)",
    "entropy_input_width_flip_rate": "Width flip rate",
    "entropy_input_height_flip_rate": "Height flip rate",
    "entropy_input_channel_flip_rate": "Channel flip rate",
    "entropy_input_nchw_memory_flip_rate": "NCHW flip rate",
    "entropy_input_conv3x3_patch_flip_rate": "3x3 patch flip rate",
    "entropy_conv_patch_active_count_std": "Conv patch active-count SD",
    "entropy_conv_patch_active_count_entropy_bits": "Conv patch active-count entropy (bits)",
    "entropy_conv_exact_patch_entropy_bits": "Exact Conv patch entropy (bits)",
    "entropy_conv_exact_patch_unique_fraction": "Unique Conv patch fraction",
    "entropy_conv_exact_patch_collision_probability": "Exact Conv patch collision probability",
    "entropy_conv_duplicate_patch_occurrence_fraction": "Duplicate Conv patch occurrence fraction",
    "entropy_conv_adjacent_exact_patch_same_fraction": "Adjacent exact Conv patch match fraction",
    "entropy_conv_adjacent_patch_jaccard": "Adjacent Conv patch Jaccard",
    "entropy_conv_patch_stream_conditional_entropy_bits": "Conv patch-stream entropy (bits)",
    "entropy_conv_patch_stream_flip_rate": "Conv patch-stream flip rate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", type=Path, default=Path("controlled_entropy_results")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("visualization"))
    parser.add_argument("--format", choices=("pdf", "png"), default="pdf")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--metrics",
        default="",
        help="Optional comma-separated perf column names, with or without perf_",
    )
    return parser.parse_args()


def safe_name(value: Any) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value)).strip("_") or "unknown"


def numeric(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def local_perf_path(input_dir: Path, result: dict[str, Any]) -> Path | None:
    run_id = result.get("run_id")
    local = input_dir / f"{run_id}_perf.csv"
    if local.is_file():
        return local
    configured = result.get("perf_csv")
    if configured and Path(configured).is_file():
        return Path(configured)
    return None


def load_perf_row(path: Path) -> dict[str, Any]:
    with path.open(newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row.get("perf_status") == "ok" and row.get("phase") == "replay"
        ]
    if not rows:
        raise ValueError(f"No successful replay PMU row in {path}")
    if len(rows) != 1:
        raise ValueError(f"Expected one replay PMU row in {path}, found {len(rows)}")
    return rows[0]


def load_runs(input_dir: Path) -> pd.DataFrame:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for json_path in sorted(input_dir.glob("*.json")):
        try:
            result = json.loads(json_path.read_text())
            config = result["config"]
            record: dict[str, Any] = {
                "run_id": result["run_id"],
                "source_json": str(json_path),
                "operator": config["operator"],
                "regime": config["regime"],
                "temporal": config["temporal"],
                "seed": config["seed"],
                "trial_id": config.get("trial_id", "trial_0"),
                "device_id": config.get("device_id") or result.get("host", "unknown"),
                "host": result.get("host", ""),
                "batch_size": config["batch_size"],
                "channels": config["channels"],
                "height": config["height"],
                "width": config["width"],
                "activation_rate": config["activation_rate"],
                "pool_size": config["pool_size"],
                "pool_stride": config["pool_stride"],
                "conv_out_channels": config.get("conv_out_channels", 64),
                "conv_kernel_size": config.get("conv_kernel_size", 3),
                "conv_stride": config.get("conv_stride", 1),
                "conv_padding": config.get("conv_padding", 1),
                "bank_size": 1 if config["temporal"] == "stable" else config["bank_size"],
                "warmup": config["warmup"],
                "repeats": config["repeats"],
                "threads": config["threads"],
                "elapsed_seconds": result["elapsed_seconds"],
                "nanoseconds_per_call": result["nanoseconds_per_call"],
            }
            record.update(
                {f"entropy_{key}": value for key, value in result["entropy_mean"].items()}
            )
            perf_path = local_perf_path(input_dir, result)
            if perf_path is not None:
                perf_row = load_perf_row(perf_path)
                record["source_perf_csv"] = str(perf_path)
                for key, value in perf_row.items():
                    if key.startswith("perf_"):
                        record[key] = numeric(value)
            else:
                record["source_perf_csv"] = ""
            records.append(record)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{json_path.name}: {exc}")

    if errors:
        raise ValueError("Unable to load result files:\n  " + "\n  ".join(errors))
    if not records:
        raise FileNotFoundError(f"No controlled-experiment JSON files in {input_dir}")

    frame = pd.DataFrame.from_records(records)
    frame["regime"] = pd.Categorical(frame["regime"], REGIMES, ordered=True)
    return frame.sort_values(
        ["device_id", "operator", "temporal", "regime", "seed", "trial_id"]
    ).reset_index(drop=True)


def perf_metric_columns(frame: pd.DataFrame) -> list[str]:
    metrics: list[str] = []
    for column in frame.columns:
        if not column.startswith("perf_"):
            continue
        if column in PERF_METADATA_COLUMNS:
            continue
        if column.endswith("_enabled_pct") or column.endswith("_runtime_pct"):
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            metrics.append(column)
    return metrics


def select_metrics(available: list[str], requested: str) -> list[str]:
    if not requested.strip():
        return available
    selected: list[str] = []
    missing: list[str] = []
    for token in requested.split(","):
        token = token.strip()
        if not token:
            continue
        column = token if token.startswith("perf_") else f"perf_{token}"
        if column in available:
            selected.append(column)
        else:
            missing.append(column)
    if missing:
        raise ValueError(f"Requested metrics are unavailable: {', '.join(missing)}")
    return selected


def metric_label(metric: str, *, per_instruction: bool = False) -> str:
    label = METRIC_LABELS.get(metric, metric.removeprefix("perf_").replace("_", " ").title())
    return f"{label} / instruction" if per_instruction else label


def style_axis(axis: Axes) -> None:
    axis.grid(axis="y", color="#D9DEE5", linewidth=0.7, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=8)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 4))
    axis.yaxis.set_major_formatter(formatter)


def plot_condition_panel(
    axis: Axes,
    frame: pd.DataFrame,
    metric: str,
    operator: str,
    temporal: str,
) -> None:
    subset = frame[(frame["operator"] == operator) & (frame["temporal"] == temporal)]
    x_values: list[int] = []
    means: list[float] = []
    deviations: list[float] = []
    for x_index, regime in enumerate(REGIMES):
        values = pd.to_numeric(
            subset.loc[subset["regime"] == regime, metric], errors="coerce"
        ).dropna()
        if values.empty:
            continue
        x_values.append(x_index)
        means.append(float(values.mean()))
        deviations.append(float(values.std(ddof=1)) if len(values) > 1 else 0.0)
        jitter = np.linspace(-0.06, 0.06, len(values)) if len(values) > 1 else [0.0]
        axis.scatter(
            x_index + np.asarray(jitter), values,
            color=COLORS[regime], s=18, alpha=0.45, linewidths=0, zorder=2,
        )
    if x_values:
        axis.plot(x_values, means, color="#24292F", marker="o", markersize=4, linewidth=1.4)
        deviations_array = np.asarray(deviations)
        means_array = np.asarray(means)
        axis.errorbar(
            x_values, means_array, yerr=deviations_array, fmt="none",
            ecolor="#24292F", elinewidth=0.9, capsize=3, capthick=0.9, zorder=4,
        )
        if np.any(deviations_array > 0):
            axis.fill_between(
                x_values, means_array - deviations_array, means_array + deviations_array,
                color="#6E7781", alpha=0.16, linewidth=0,
            )
    axis.set_xticks(range(len(REGIMES)), [item.title() for item in REGIMES])
    style_axis(axis)


def plot_metric_grid(
    frame: pd.DataFrame,
    metrics: Iterable[str],
    output_path: Path,
    *,
    per_instruction: bool,
    dpi: int,
) -> None:
    metrics = list(metrics)
    if not metrics:
        return
    panels = [(operator, temporal) for operator in OPERATORS for temporal in TEMPORALS]
    figure, axes = plt.subplots(
        len(metrics), len(panels),
        figsize=(4.0 * len(panels), max(3.2, 2.35 * len(metrics))),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, metric in enumerate(metrics):
        for column_index, (operator, temporal) in enumerate(panels):
            axis = axes[row_index, column_index]
            plot_condition_panel(axis, frame, metric, operator, temporal)
            if row_index == 0:
                axis.set_title(
                    f"{OPERATOR_LABELS[operator]} | {temporal.title()}",
                    fontsize=10,
                )
            if column_index == 0:
                axis.set_ylabel(metric_label(metric, per_instruction=per_instruction), fontsize=8)
            if row_index != len(metrics) - 1:
                axis.tick_params(labelbottom=False)
    figure.suptitle("Mean +/- 1 SD; dots are individual process runs", fontsize=11)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def add_per_instruction_metrics(
    frame: pd.DataFrame, metrics: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    if "perf_instructions" not in frame or frame["perf_instructions"].isna().all():
        return frame, []
    output = frame.copy()
    denominator = pd.to_numeric(output["perf_instructions"], errors="coerce").replace(0, np.nan)
    normalized: list[str] = []
    for metric in metrics:
        if metric == "perf_instructions":
            continue
        name = f"{metric}_per_instruction"
        output[name] = pd.to_numeric(output[metric], errors="coerce") / denominator
        normalized.append(name)
        METRIC_LABELS[name] = METRIC_LABELS.get(
            metric, metric.removeprefix("perf_").replace("_", " ").title()
        )
    return output, normalized


def plot_entropy(frame: pd.DataFrame, output_path: Path, dpi: int) -> None:
    metrics = [metric for metric in ENTROPY_METRICS if metric in frame]
    if not metrics:
        return
    figure, axes = plt.subplots(
        len(metrics), 2,
        figsize=(9, max(3.2, 2.25 * len(metrics))),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, metric in enumerate(metrics):
        for column_index, temporal in enumerate(TEMPORALS):
            axis = axes[row_index, column_index]
            subset = frame[frame["temporal"] == temporal]
            for operator in OPERATORS:
                marker = OPERATOR_MARKERS[operator]
                color = OPERATOR_COLORS[operator]
                means = []
                deviations = []
                for x_index, regime in enumerate(REGIMES):
                    values = pd.to_numeric(
                        subset.loc[
                            (subset["operator"] == operator) & (subset["regime"] == regime),
                            metric,
                        ], errors="coerce"
                    ).dropna()
                    means.append(float(values.mean()) if not values.empty else np.nan)
                    deviations.append(
                        float(values.std(ddof=1)) if len(values) > 1 else 0.0
                    )
                    if not values.empty:
                        jitter = (
                            np.linspace(-0.05, 0.05, len(values))
                            if len(values) > 1 else [0.0]
                        )
                        axis.scatter(
                            x_index + np.asarray(jitter), values, marker=marker,
                            color=color,
                            s=20, alpha=0.45, linewidths=0, zorder=3,
                        )
                means_array = np.asarray(means)
                deviations_array = np.asarray(deviations)
                x_values = np.arange(len(REGIMES))
                axis.plot(
                    x_values, means_array, marker=marker, markersize=4,
                    linewidth=1.25, color=color,
                    label=OPERATOR_LABELS[operator],
                )
                axis.errorbar(
                    x_values, means_array, yerr=deviations_array, fmt="none",
                    ecolor=color,
                    elinewidth=0.9, capsize=3, capthick=0.9,
                )
                if np.any(deviations_array > 0):
                    axis.fill_between(
                        x_values, means_array - deviations_array,
                        means_array + deviations_array,
                        color=color,
                        alpha=0.10, linewidth=0,
                    )
            axis.set_xticks(range(len(REGIMES)), [item.title() for item in REGIMES])
            style_axis(axis)
            if row_index == 0:
                axis.set_title(temporal.title(), fontsize=10)
                axis.legend(frameon=False, fontsize=8)
            if column_index == 0:
                axis.set_ylabel(ENTROPY_LABELS.get(metric, metric), fontsize=8)
            if row_index != len(metrics) - 1:
                axis.tick_params(labelbottom=False)
    figure.suptitle("Entropy controls: mean +/- 1 SD; dots are individual runs", fontsize=11)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_runtime(frame: pd.DataFrame, output_path: Path, dpi: int) -> None:
    metrics = ("elapsed_seconds", "nanoseconds_per_call")
    labels = ("Replay elapsed time (s)", "Nanoseconds / operator call")
    figure, axes = plt.subplots(
        2, len(OPERATORS), figsize=(4.5 * len(OPERATORS), 6.5),
        squeeze=False, constrained_layout=True,
    )
    for row_index, (metric, label) in enumerate(zip(metrics, labels)):
        for column_index, operator in enumerate(OPERATORS):
            axis = axes[row_index, column_index]
            subset = frame[frame["operator"] == operator]
            for temporal in TEMPORALS:
                means = []
                deviations = []
                for regime in REGIMES:
                    values = pd.to_numeric(
                        subset.loc[
                            (subset["temporal"] == temporal) & (subset["regime"] == regime),
                            metric,
                        ], errors="coerce"
                    ).dropna()
                    means.append(float(values.mean()) if not values.empty else np.nan)
                    deviations.append(
                        float(values.std(ddof=1)) if len(values) > 1 else 0.0
                    )
                    if not values.empty:
                        x_index = REGIMES.index(regime)
                        jitter = (
                            np.linspace(-0.05, 0.05, len(values))
                            if len(values) > 1 else [0.0]
                        )
                        axis.scatter(
                            x_index + np.asarray(jitter), values,
                            color=PANEL_COLORS[temporal], s=20, alpha=0.45,
                            linewidths=0, zorder=3,
                        )
                means_array = np.asarray(means)
                deviation_array = np.asarray(deviations)
                x_values = np.arange(len(REGIMES))
                axis.plot(
                    x_values, means_array, marker="o", markersize=4, linewidth=1.4,
                    color=PANEL_COLORS[temporal], label=temporal.title(),
                )
                axis.errorbar(
                    x_values, means_array, yerr=deviation_array, fmt="none",
                    ecolor=PANEL_COLORS[temporal], elinewidth=0.9,
                    capsize=3, capthick=0.9,
                )
                if np.any(deviation_array > 0):
                    axis.fill_between(
                        x_values, means_array - deviation_array, means_array + deviation_array,
                        color=PANEL_COLORS[temporal], alpha=0.14, linewidth=0,
                    )
            axis.set_xticks(range(len(REGIMES)), [item.title() for item in REGIMES])
            style_axis(axis)
            if row_index == 0:
                axis.set_title(OPERATOR_LABELS[operator], fontsize=10)
            if column_index == 0:
                axis.set_ylabel(label, fontsize=9)
            axis.legend(frameon=False, fontsize=8)
    figure.suptitle("Runtime: mean +/- 1 SD; dots are individual process runs", fontsize=11)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def validate_configuration(frame: pd.DataFrame, device_id: str) -> None:
    columns = [
        "batch_size", "channels", "height", "width", "activation_rate",
        "pool_size", "pool_stride", "conv_out_channels", "conv_kernel_size",
        "conv_stride", "conv_padding", "warmup", "repeats", "threads",
    ]
    for operator, subset in frame.groupby("operator", observed=True):
        varying = [
            column for column in columns
            if subset[column].nunique(dropna=False) > 1
        ]
        if varying:
            raise ValueError(
                f"Device {device_id}, operator {operator} has mixed controlled "
                f"configurations in: {', '.join(varying)}. Use a separate input "
                "directory for each configuration."
            )


def render_device(
    frame: pd.DataFrame,
    output_dir: Path,
    file_format: str,
    dpi: int,
    requested_metrics: str,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    available = perf_metric_columns(frame)
    metrics = select_metrics(available, requested_metrics)
    written: list[Path] = []

    summary_path = output_dir / "controlled_entropy_summary.csv"
    frame.to_csv(summary_path, index=False)
    written.append(summary_path)

    if metrics:
        raw_path = output_dir / f"pmu_raw.{file_format}"
        plot_metric_grid(
            frame, metrics, raw_path, per_instruction=False, dpi=dpi
        )
        written.append(raw_path)

        normalized_frame, normalized_metrics = add_per_instruction_metrics(frame, metrics)
        if normalized_metrics:
            normalized_path = output_dir / f"pmu_per_instruction.{file_format}"
            plot_metric_grid(
                normalized_frame, normalized_metrics, normalized_path,
                per_instruction=True, dpi=dpi,
            )
            written.append(normalized_path)

    entropy_path = output_dir / f"entropy_control.{file_format}"
    plot_entropy(frame, entropy_path, dpi)
    if entropy_path.exists():
        written.append(entropy_path)

    runtime_path = output_dir / f"runtime.{file_format}"
    plot_runtime(frame, runtime_path, dpi)
    written.append(runtime_path)
    return written


def main() -> None:
    args = parse_args()
    frame = load_runs(args.input_dir.resolve())
    devices = [str(value) for value in frame["device_id"].dropna().unique()]
    all_written: list[Path] = []
    for device_id in devices:
        subset = frame[frame["device_id"].astype(str) == device_id].copy()
        validate_configuration(subset, device_id)
        device_output = (
            args.output_dir.resolve()
            if len(devices) == 1
            else args.output_dir.resolve() / safe_name(device_id)
        )
        all_written.extend(
            render_device(
                subset, device_output, args.format, args.dpi, args.metrics
            )
        )

    print(f"Loaded {len(frame)} runs from {args.input_dir.resolve()}")
    print(f"Devices: {', '.join(devices)}")
    for path in all_written:
        print(path)


if __name__ == "__main__":
    main()
