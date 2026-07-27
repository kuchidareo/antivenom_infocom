#!/usr/bin/env python3
"""Plot loss and forward L1D behavior over each batch's early instructions."""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "collected_logs"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "visualization_batch_loss_cache_early"


def load_common_module():
    path = SCRIPT_DIR / "5_batch_loss_cache_visualization.py"
    spec = importlib.util.spec_from_file_location("batch_loss_cache_common", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load shared analysis functions from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMMON = load_common_module()
SERIES_COLORS = {
    "clean": "#2468a2",
    "availability_shortcuts": "#c43c39",
    "clean_to_availability_shortcuts": "#2468a2",
    "availability_shortcuts_to_clean": "#c43c39",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--instruction-end-percent",
        type=float,
        default=20.0,
        help="End of the within-batch instruction range (default: 20).",
    )
    parser.add_argument(
        "--device",
        action="append",
        help="Restrict the plot to a device ID; repeat for multiple devices.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def series_label(series: str) -> str:
    labels = {
        "clean": "Clean",
        "availability_shortcuts": "Availability shortcuts",
        "clean_to_availability_shortcuts": "Clean -> availability shortcuts",
        "availability_shortcuts_to_clean": "Availability shortcuts -> clean",
    }
    return labels.get(series, series.replace("_", " ").title())


def read_run(perf_path: Path) -> dict[str, object]:
    with perf_path.open(newline="") as handle:
        row = next(csv.DictReader(handle), None)
    if row is None:
        raise ValueError(f"Empty perf CSV: {perf_path}")
    method = row.get("poisoning_method") or "unknown"
    sequence = row.get("training_sequence") or ""
    return {
        "perf_path": perf_path,
        "metrics_path": perf_path.with_name(
            perf_path.name.replace("_perf.csv", "_metrics.csv")
        ),
        "device_id": row.get("device_id") or perf_path.parent.parent.name,
        "method": method,
        "series": sequence or method,
        "training_sequence": sequence,
        "dataset": row.get("dataset", ""),
        "partition_method": row.get("partition_method", ""),
        "model": row.get("model", ""),
        "batch_size": int(row.get("batch_size") or 0),
        "augmentation": row.get("augmentation_profile", ""),
        "background_enabled": COMMON.parse_bool(
            row.get("background_workload_enabled")
        ),
        "trial_id": row.get("trial_id", ""),
        "start_time": float(row.get("timestamp_unix") or 0.0),
    }


def is_compatible_run(run: dict[str, object]) -> bool:
    augmentation = str(run["augmentation"])
    return (
        run["method"] in COMMON.METHODS
        and run["dataset"] == COMMON.DATASET
        and run["partition_method"] == "iid"
        and run["model"] == COMMON.MODEL
        and run["batch_size"] == COMMON.BATCH_SIZE
        and augmentation in {"", COMMON.AUGMENTATION}
        and not run["background_enabled"]
    )


def discover_available_runs(
    input_dir: Path, selected_devices: set[str]
) -> dict[tuple[str, str], dict[str, object]]:
    candidates: dict[tuple[str, str], list[dict[str, object]]] = {}
    for perf_path in sorted(input_dir.rglob("*_perf.csv")):
        try:
            run = read_run(perf_path)
        except ValueError as exc:
            warnings.warn(str(exc))
            continue
        if not is_compatible_run(run):
            continue
        device = str(run["device_id"])
        if selected_devices and device not in selected_devices:
            continue
        if not Path(run["metrics_path"]).exists():
            warnings.warn(f"Missing metrics CSV for {perf_path}")
            continue
        candidates.setdefault((device, str(run["series"])), []).append(run)

    selected: dict[tuple[str, str], dict[str, object]] = {}
    for key, runs in candidates.items():
        runs.sort(key=lambda run: float(run["start_time"]))
        selected[key] = runs[-1]
        if len(runs) > 1:
            warnings.warn(
                f"{key}: selected latest {Path(runs[-1]['perf_path']).name}; "
                f"ignored {len(runs) - 1} older run(s)"
            )
    if not selected:
        raise ValueError(f"No compatible perf/metrics run pairs found below {input_dir}")
    return selected


def early_instruction_window(batch: pd.DataFrame, fraction: float) -> pd.Series:
    batch = batch.sort_values(["timestamp_unix", "perf_elapsed_sec"])
    instructions = batch["perf_instructions"].clip(lower=0).to_numpy(dtype=float)
    l1d_accesses = batch["perf_l1d_cache"].clip(lower=0).to_numpy(dtype=float)
    total_instructions = float(instructions.sum())
    target_instructions = total_instructions * fraction

    selected_instructions = 0.0
    selected_l1d = 0.0
    selected_intervals = 0
    equivalent_intervals = 0.0
    remaining = target_instructions
    for interval_instructions, interval_l1d in zip(instructions, l1d_accesses):
        if remaining <= 0:
            break
        if interval_instructions <= 0:
            continue
        overlap = min(interval_instructions, remaining)
        interval_fraction = overlap / interval_instructions
        selected_instructions += overlap
        selected_l1d += interval_l1d * interval_fraction
        selected_intervals += 1
        equivalent_intervals += interval_fraction
        remaining -= overlap

    ratio = (
        selected_l1d / selected_instructions
        if selected_instructions > 0
        else np.nan
    )
    return pd.Series(
        {
            "batch_total_instructions": total_instructions,
            "window_instructions": selected_instructions,
            "window_l1d_accesses": selected_l1d,
            "window_l1d_accesses_per_instruction": ratio,
            "window_interval_count": selected_intervals,
            "window_equivalent_intervals": equivalent_intervals,
            "window_coverage_percent": (
                100.0 * selected_instructions / total_instructions
                if total_instructions > 0
                else np.nan
            ),
        }
    )


def load_batch_data(run: dict[str, object], fraction: float) -> pd.DataFrame:
    perf_path = Path(run["perf_path"])
    required = {
        "epoch",
        "batch_idx",
        "phase",
        "perf_status",
        "timestamp_unix",
        "perf_elapsed_sec",
        "perf_l1d_cache",
        "perf_instructions",
    }
    header = set(pd.read_csv(perf_path, nrows=0).columns)
    missing = required - header
    if missing:
        raise ValueError(f"{perf_path} is missing columns: {sorted(missing)}")

    perf = pd.read_csv(perf_path, usecols=sorted(required))
    perf = perf[(perf["phase"] == "forward") & (perf["perf_status"] == "ok")].copy()
    numeric = (
        "epoch",
        "batch_idx",
        "timestamp_unix",
        "perf_elapsed_sec",
        "perf_l1d_cache",
        "perf_instructions",
    )
    for column in numeric:
        perf[column] = pd.to_numeric(perf[column], errors="coerce")
    perf = perf.dropna(subset=list(numeric))

    windows = (
        perf.groupby(["epoch", "batch_idx"], sort=True)
        .apply(early_instruction_window, fraction=fraction, include_groups=False)
        .reset_index()
    )

    metrics_path = Path(run["metrics_path"])
    metric_header = set(pd.read_csv(metrics_path, nrows=0).columns)
    optional_metadata = [
        column
        for column in (
            "poisoning_method",
            "input_poisoning_method",
            "training_sequence",
            "stage_index",
            "stage_epoch",
            "model_state_condition",
        )
        if column in metric_header
    ]
    metrics = pd.read_csv(
        metrics_path,
        usecols=["epoch", "batch_idx", "metric_event", "loss", *optional_metadata],
    )
    metrics = metrics[metrics["metric_event"] == "train_batch"].copy()
    for column in ("epoch", "batch_idx", "loss"):
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    metrics = metrics.dropna(subset=["epoch", "batch_idx", "loss"])
    metrics = metrics.drop_duplicates(["epoch", "batch_idx"], keep="last")

    joined = windows.merge(
        metrics[["epoch", "batch_idx", "loss", *optional_metadata]],
        on=["epoch", "batch_idx"],
        how="inner",
        validate="one_to_one",
    ).sort_values(["epoch", "batch_idx"])
    joined = joined.dropna(
        subset=["window_l1d_accesses", "window_l1d_accesses_per_instruction", "loss"]
    ).reset_index(drop=True)
    if joined.empty:
        raise ValueError(f"No early forward windows could be joined to {metrics_path}")

    joined["epoch"] = joined["epoch"].astype(int)
    joined["batch_idx"] = joined["batch_idx"].astype(int)
    joined["global_batch_idx"] = np.arange(1, len(joined) + 1)
    joined["instruction_range_start_percent"] = 0.0
    joined["instruction_range_end_percent"] = fraction * 100.0
    joined["device_id"] = str(run["device_id"])
    joined["method"] = str(run["method"])
    joined["series"] = str(run["series"])
    joined["trial_id"] = str(run["trial_id"])
    joined["perf_path"] = str(perf_path)
    return joined


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    values = [
        "window_l1d_accesses",
        "window_l1d_accesses_per_instruction",
        "window_instructions",
        "window_coverage_percent",
        "loss",
    ]
    summary = (
        data.groupby(["series", "global_batch_idx"], as_index=False)[values]
        .agg(["mean", "std"])
    )
    summary.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    return summary.rename(
        columns={"series_": "series", "global_batch_idx_": "global_batch_idx"}
    )


def plot(
    data: pd.DataFrame,
    summary: pd.DataFrame,
    output_path: Path,
    end_percent: float,
    dpi: int,
) -> None:
    metrics = (
        (
            "window_l1d_accesses",
            f"L1D accesses in first {end_percent:g}% of batch instructions",
        ),
        (
            "window_l1d_accesses_per_instruction",
            f"L1D accesses / instruction in first {end_percent:g}%",
        ),
    )
    figure, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    boundaries, epoch_labels = COMMON.epoch_boundaries(data)
    available_series = sorted(summary["series"].unique())
    fallback_colors = plt.get_cmap("tab10")
    colors = {
        series: SERIES_COLORS.get(series, fallback_colors(index % 10))
        for index, series in enumerate(available_series)
    }

    for axis, (metric, ylabel) in zip(axes, metrics):
        loss_axis = axis.twinx()
        for series in available_series:
            values = summary[summary["series"] == series]
            x = values["global_batch_idx"].to_numpy()
            color = colors[series]
            axis.plot(
                x,
                values[f"{metric}_mean"],
                color=color,
                linewidth=1.5,
                marker="o",
                markersize=2.5,
            )
            loss_axis.plot(
                x,
                values["loss_mean"],
                color=color,
                linewidth=1.15,
                linestyle="--",
                alpha=0.72,
            )

        for boundary in boundaries:
            axis.axvline(boundary, color="#b8bcc2", linewidth=0.65, alpha=0.65)
        axis.set_ylabel(ylabel)
        loss_axis.set_ylabel("Training loss", color="#5c6066")
        loss_axis.tick_params(axis="y", labelcolor="#5c6066")
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
        axis.grid(axis="y", color="#d8dadd", linewidth=0.55, alpha=0.8)
        axis.margins(x=0.01)

    axes[-1].set_xlabel("Global batch index (continuous across epochs)")
    axes[-1].set_xlim(0.5, float(data["global_batch_idx"].max()) + 0.5)
    for position, label in epoch_labels:
        axes[0].text(
            position,
            1.035,
            label,
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=7,
            color="#5c6066",
        )

    handles = [
        Line2D(
            [0], [0], color=colors[series], linewidth=1.8,
            label=series_label(series),
        )
        for series in available_series
    ]
    handles.extend(
        [
            Line2D([0], [0], color="#333333", linewidth=1.6, label="Forward metric"),
            Line2D(
                [0], [0], color="#333333", linewidth=1.3,
                linestyle="--", label="Batch training loss",
            ),
        ]
    )
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=min(4, len(handles)),
        frameon=False,
    )
    figure.suptitle(
        "Available training runs: early-forward L1D behavior and loss\n"
        f"Instruction range 0-{end_percent:g}%; mean across "
        f"{data['device_id'].nunique()} device(s)",
        y=0.99,
        fontsize=13,
    )
    figure.tight_layout(rect=(0.02, 0.02, 0.98, 0.86), h_pad=1.4)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not 0 < args.instruction_end_percent <= 100:
        raise ValueError("--instruction-end-percent must be in (0, 100]")
    fraction = args.instruction_end_percent / 100.0
    runs = discover_available_runs(args.input_dir.resolve(), set(args.device or ()))

    frames = []
    selected_rows = []
    for (device, series), run in sorted(runs.items()):
        frame = load_batch_data(run, fraction)
        frames.append(frame)
        selected_rows.append(
            {
                "device_id": device,
                "series": series,
                "initial_method": run["method"],
                "training_sequence": run["training_sequence"],
                "trial_id": run["trial_id"],
                "perf_path": run["perf_path"],
                "metrics_path": run["metrics_path"],
                "joined_batches": len(frame),
            }
        )

    data = pd.concat(frames, ignore_index=True)
    summary = summarize(data)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / "forward_first_20pct_cache_vs_loss.csv", index=False)
    summary.to_csv(
        output_dir / "forward_first_20pct_cache_vs_loss_summary.csv", index=False
    )
    pd.DataFrame(selected_rows).to_csv(output_dir / "selected_runs.csv", index=False)
    output_path = output_dir / "forward_first_20pct_cache_vs_loss.png"
    plot(data, summary, output_path, args.instruction_end_percent, args.dpi)

    coverage = data["window_coverage_percent"]
    print(f"Saved {output_path}")
    print(
        f"Devices: {', '.join(sorted(data['device_id'].unique()))}; "
        f"series: {', '.join(sorted(data['series'].unique()))}; "
        f"batches: {len(data)}; instruction coverage: "
        f"{coverage.min():.6f}-{coverage.max():.6f}%"
    )


if __name__ == "__main__":
    main()
