#!/usr/bin/env python3
"""Visualize raw CPU, psutil, and perf samples by phase, metric, and batch."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from average_metric_visualization import plot_batch_variances
from analysis import (
    PERF_COUNTER_COLUMNS,
    add_counter_deltas,
    color_for_method,
    discover_run_files,
    load_all,
)
from raw_data_visualization_0714 import (
    PERF_NORMALIZATION_SUFFIXES,
    normalize_metadata,
    perf_trial_values,
    plot_configuration,
    safe_name,
    summarize,
)


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "collected_logs" / "logs"
DEFAULT_OUTPUT = HERE / "result" / "raw_data_visualization"

CPU_METRICS = [
    "process_cpu_percent",
    "system_cpu_core_0",
    "system_cpu_core_1",
    "system_cpu_core_2",
    "system_cpu_core_3",
    "system_cpu_freq_mhz",
    "system_cpu_freq_core_0_mhz",
    "system_cpu_freq_core_1_mhz",
    "system_cpu_freq_core_2_mhz",
    "system_cpu_freq_core_3_mhz",
    "process_ctx_switches_voluntary_delta",
    "process_ctx_switches_involuntary_delta",
]
PERF_METRICS = list(PERF_COUNTER_COLUMNS)
METHOD_ORDER = ["clean", "unlearnable_examples", "availability_shortcuts"]
METHOD_LABELS = {
    "clean": "Clean",
    "unlearnable_examples": "Unlearnable examples",
    "availability_shortcuts": "Availability shortcuts",
}


def select_latest_runs(perf: pd.DataFrame, hardware: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    columns = ["run_id", "poisoning_method"]
    sources = [df[columns] for df in (perf, hardware) if set(columns).issubset(df.columns)]
    if not sources:
        raise ValueError("No hardware or perf CSV contains run and poisoning metadata")
    metadata = pd.concat(sources, ignore_index=True).drop_duplicates()
    latest = metadata.sort_values("run_id").groupby("poisoning_method", sort=False).tail(1)
    methods = [method for method in METHOD_ORDER if method in set(latest["poisoning_method"])]
    methods += [method for method in latest["poisoning_method"] if method not in methods]
    run_ids = set(latest["run_id"])
    if "run_id" in perf:
        perf = perf[perf["run_id"].isin(run_ids)].copy()
    if "run_id" in hardware:
        hardware = hardware[hardware["run_id"].isin(run_ids)].copy()
    return perf, hardware, methods


def normalize_frequency_columns(hardware: pd.DataFrame) -> pd.DataFrame:
    hardware = hardware.copy()
    for core in range(4):
        source = f"system_cpu_freq_core_{core}"
        target = f"{source}_mhz"
        if target not in hardware and source in hardware:
            hardware[target] = hardware[source]
    return hardware


def prepare(df: pd.DataFrame, epoch: int, metrics: list[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    df = df[df["epoch"] == epoch].copy()
    for column in ["timestamp_unix", "batch_idx", *metrics]:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    keys = ["run_id", "phase", "batch_idx"]
    df["elapsed"] = df["timestamp_unix"] - df.groupby(keys)["timestamp_unix"].transform("min")
    return df


def available_metrics(perf: pd.DataFrame, hardware: pd.DataFrame, phases: list[str]) -> list[tuple[str, pd.DataFrame]]:
    rows = []
    for metric, source in [
        *((metric, hardware) for metric in CPU_METRICS),
        *((metric, perf) for metric in PERF_METRICS),
    ]:
        if metric in source and source.loc[source["phase"].isin(phases), metric].notna().any():
            rows.append((metric, source))
    return rows


def metric_batch_variances(
    metric_rows: list[tuple[str, pd.DataFrame]],
    sources: tuple[pd.DataFrame, pd.DataFrame],
    phases: list[str],
) -> pd.DataFrame:
    group_keys = ["run_id", "poisoning_method", "epoch", "phase"]
    batch_keys = [*group_keys, "batch_idx"]
    averages = []
    for metric, source in metric_rows:
        subset = source[source["phase"].isin(phases)].copy()
        for column in ["epoch", "batch_idx", metric]:
            subset[column] = pd.to_numeric(subset[column], errors="coerce")
        subset = subset.dropna(subset=["epoch", "batch_idx", metric])
        if not subset.empty:
            averages.append(subset.groupby(batch_keys, observed=True)[metric].mean().rename(metric))
    if not averages:
        return pd.DataFrame(columns=group_keys)

    batch_summary = pd.concat(averages, axis=1).reset_index()
    metrics = [metric for metric, _ in metric_rows if metric in batch_summary]
    # The samples inside one batch first form a representative batch mean.
    # The reported epoch/phase statistic is then the sample variance between
    # those batch means; it is not an equal-weight mean across batches.
    summary = batch_summary.groupby(group_keys, observed=True, as_index=False)[metrics].var(ddof=1)
    summary = summary.rename(columns={metric: f"{metric}_batch_variance" for metric in metrics})

    timestamp_parts = []
    timestamp_columns = [*batch_keys, "timestamp_unix"]
    for source in sources:
        if set(timestamp_columns).issubset(source.columns):
            samples = source.loc[source["phase"].isin(phases), timestamp_columns].copy()
            samples["timestamp_unix"] = pd.to_numeric(samples["timestamp_unix"], errors="coerce")
            timestamp_parts.append(samples.dropna(subset=["epoch", "batch_idx", "timestamp_unix"]))
    if timestamp_parts:
        timestamps = pd.concat(timestamp_parts, ignore_index=True)
        batch_elapsed = timestamps.groupby(batch_keys, observed=True)["timestamp_unix"].agg(
            lambda values: values.max() - values.min()
        )
        elapsed = (
            batch_elapsed.groupby(group_keys, observed=True)
            .var(ddof=1)
            .rename("elapsed_time_sec_batch_variance")
        )
        summary = summary.merge(elapsed.reset_index(), on=group_keys, how="left")

    summary["epoch"] = summary["epoch"].astype(int)
    method_order = {method: index for index, method in enumerate(METHOD_ORDER)}
    phase_order = {phase: index for index, phase in enumerate(phases)}
    summary["_method_order"] = summary["poisoning_method"].map(method_order).fillna(len(method_order))
    summary["_phase_order"] = summary["phase"].map(phase_order).fillna(len(phase_order))
    return summary.sort_values(["_method_order", "epoch", "_phase_order"]).drop(
        columns=["_method_order", "_phase_order"]
    )


def number(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    if value == 0 or 0.01 <= abs(value) < 10_000:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.1e}"


def plot_phase(
    phase: str,
    epoch: int,
    metric_rows: list[tuple[str, pd.DataFrame]],
    methods: list[str],
    max_batches: int,
    output: Path,
    dpi: int,
) -> None:
    batch_ids = sorted(
        {
            int(batch)
            for _, source in metric_rows
            for batch in source.loc[source["phase"] == phase, "batch_idx"].dropna().unique()
        }
    )
    if max_batches:
        batch_ids = batch_ids[:max_batches]
    if not batch_ids:
        print(f"Skipping {phase}: no samples", flush=True)
        return

    rows, columns = len(metric_rows), len(batch_ids)
    fig, ax = plt.subplots(figsize=(max(16, columns * 1.55), max(10, rows * 0.72 + 2)))
    ax.set_xlim(-2.0, columns)
    ax.set_ylim(-0.8, rows + 0.8)
    ax.axis("off")

    # Every metric uses one shared raw y scale across its batch row. Every
    # batch uses one shared elapsed-time x scale down its column.
    x_limits = {}
    for batch in batch_ids:
        maxima = []
        for _, source in metric_rows:
            values = source.loc[(source["phase"] == phase) & (source["batch_idx"] == batch), "elapsed"]
            if values.notna().any():
                maxima.append(float(values.max()))
        x_limits[batch] = max(maxima, default=0.1) or 0.1

    segments = {method: [] for method in methods}
    for row_index, (metric, source) in enumerate(metric_rows):
        y_base = rows - row_index - 1
        phase_values = source.loc[source["phase"] == phase, metric].dropna()
        low, high = float(phase_values.min()), float(phase_values.max())
        if high <= low:
            high = low + 1.0
        ax.text(
            -0.08,
            y_base + 0.5,
            f"{metric}\n{number(low)} .. {number(high)}",
            ha="right",
            va="center",
            fontsize=6,
        )
        for column, batch in enumerate(batch_ids):
            subset = source[(source["phase"] == phase) & (source["batch_idx"] == batch)]
            for method in methods:
                values = subset.loc[subset["poisoning_method"] == method, ["elapsed", metric]].dropna()
                if values.empty:
                    continue
                x = column + 0.06 + 0.88 * values["elapsed"].to_numpy() / x_limits[batch]
                y = y_base + 0.08 + 0.84 * (values[metric].to_numpy() - low) / (high - low)
                segments[method].append(np.column_stack([x, y]))

    for method in methods:
        if segments[method]:
            points = np.concatenate(segments[method])
            ax.add_collection(
                LineCollection(segments[method], colors=color_for_method(method), linewidths=0.75, alpha=0.9)
            )
            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=4,
                color=color_for_method(method),
                linewidths=0,
                alpha=0.9,
                zorder=3,
            )

    ax.vlines(range(columns + 1), 0, rows, color="#b8b8b8", linewidth=0.35)
    ax.hlines(range(rows + 1), 0, columns, color="#b8b8b8", linewidth=0.35)
    ax.vlines([column + 0.5 for column in range(columns)], 0, rows, color="#dddddd", linewidth=0.25)
    for column, batch in enumerate(batch_ids):
        ax.text(column + 0.5, rows + 0.12, f"Batch {batch}", ha="center", va="bottom", fontsize=7)
        ax.text(column + 0.5, -0.18, f"0 .. {x_limits[batch]:.2f} s", ha="center", va="top", fontsize=6)

    handles = [
        Line2D(
            [0],
            [0],
            color=color_for_method(method),
            label=METHOD_LABELS.get(method, method),
            linewidth=1.2,
            marker="o",
            markersize=3,
        )
        for method in methods
    ]
    has_perf = any(name.startswith("perf_") for name, _ in metric_rows)
    source_label = "CPU, psutil, and perf" if has_perf else "CPU and psutil"
    fig.suptitle(f"Raw {source_label} values: {phase}, epoch {epoch}", fontsize=13, y=0.995)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.978), ncol=len(handles), frameon=False)
    fig.text(0.54, 0.008, "Elapsed time within phase and batch", ha="center", fontsize=8)
    fig.subplots_adjust(left=0.01, right=0.998, top=0.955, bottom=0.035)
    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)


def save_epoch_perf_trajectories(
    perf: pd.DataFrame,
    phases: list[str],
    output_dir: Path,
    dpi: int,
) -> int:
    """Save raw and normalized perf trajectories across every available epoch."""
    perf_values, raw_metrics, normalized_metrics = perf_trial_values(
        normalize_metadata(perf),
        phases,
    )
    if perf_values.empty:
        print("Skipping epoch perf trajectories: no usable perf samples", flush=True)
        return 0

    summary = summarize(perf_values)
    trajectory_dir = output_dir / "epoch_trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    perf_values.to_csv(trajectory_dir / "epoch_perf_trial_values.csv", index=False)
    summary.to_csv(trajectory_dir / "epoch_perf_summary.csv", index=False)

    figure_count = 0
    configurations = summary[["device_id", "dataset_name", "model"]].drop_duplicates()
    for _, configuration in configurations.iterrows():
        device = str(configuration["device_id"])
        dataset = str(configuration["dataset_name"])
        model = str(configuration["model"])
        subset = summary[
            (summary["device_id"].astype(str) == device)
            & (summary["dataset_name"] == dataset)
            & (summary["model"] == model)
        ]
        available = set(subset["metric"])
        configuration_dir = trajectory_dir / f"{safe_name(device)}_{safe_name(dataset)}_{safe_name(model)}"

        raw_for_plot = [metric for metric in raw_metrics if metric in available]
        if raw_for_plot:
            plot_configuration(
                summary,
                device,
                dataset,
                model,
                phases,
                raw_for_plot,
                configuration_dir / "perf_totals.png",
                dpi,
                qualifier="perf counter totals per epoch and phase",
            )
            figure_count += 1

        for suffix, label in PERF_NORMALIZATION_SUFFIXES.items():
            metrics_for_plot = [metric for metric in normalized_metrics[suffix] if metric in available]
            if not metrics_for_plot:
                continue
            plot_configuration(
                summary,
                device,
                dataset,
                model,
                phases,
                metrics_for_plot,
                configuration_dir / f"perf_{suffix}.png",
                dpi,
                qualifier=f"perf counters {label}",
            )
            figure_count += 1

    print(
        f"Wrote {figure_count} ten-epoch perf trajectory plots under {trajectory_dir}",
        flush=True,
    )
    return figure_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epoch", "--epochs", dest="epoch", type=int, default=0)
    parser.add_argument("--phases", default="forward,backward")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=110)
    args = parser.parse_args()

    phases = [phase.strip() for phase in args.phases.split(",") if phase.strip()]
    runs = discover_run_files(args.input_dir)
    perf, hardware, _ = load_all(runs)
    hardware = normalize_frequency_columns(hardware)
    perf, hardware, methods = select_latest_runs(perf, hardware)
    hardware = add_counter_deltas(hardware)
    available_epochs = sorted(
        {
            int(epoch)
            for source in (perf, hardware)
            if "epoch" in source
            for epoch in pd.to_numeric(source["epoch"], errors="coerce").dropna()
        }
    )
    if args.epoch not in available_epochs:
        parser.error(f"epoch {args.epoch} is unavailable; choose from {available_epochs}")

    all_metric_rows = available_metrics(perf, hardware, phases)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # This uses all epochs. ``--epoch`` only controls the detailed batch-grid
    # plots written below.
    save_epoch_perf_trajectories(perf, phases, args.output_dir, args.dpi)

    columns = ["run_id", "poisoning_method"]
    selected = pd.concat([source[columns] for source in (perf, hardware) if set(columns).issubset(source.columns)])
    selected.drop_duplicates().sort_values("poisoning_method").to_csv(args.output_dir / "selected_runs.csv", index=False)
    variances = metric_batch_variances(all_metric_rows, (perf, hardware), phases)
    variances_path = args.output_dir / "batch_metric_variances.csv"
    variances.to_csv(variances_path, index=False)
    variance_plot_path = args.output_dir / "epoch_metric_variances.png"
    variance_metric_count, variance_epoch_count = plot_batch_variances(
        variances,
        variance_plot_path,
        args.dpi,
    )
    print(f"Wrote {variances_path}", flush=True)
    print(
        f"Wrote {variance_plot_path} "
        f"({variance_metric_count} metrics, {variance_epoch_count} epochs)",
        flush=True,
    )

    perf = prepare(perf, args.epoch, PERF_METRICS)
    hardware = prepare(hardware, args.epoch, CPU_METRICS)
    metric_rows = available_metrics(perf, hardware, phases)
    print(f"Epoch {args.epoch}; methods: {', '.join(methods)}; metrics: {len(metric_rows)}", flush=True)
    for phase in phases:
        path = args.output_dir / f"raw_{phase}.png"
        print(f"Rendering {phase} -> {path}", flush=True)
        plot_phase(phase, args.epoch, metric_rows, methods, args.max_batches, path, args.dpi)
        if path.exists():
            print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    main()
