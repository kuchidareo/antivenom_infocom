#!/usr/bin/env python3
"""Analyze cache/perf behavior during optimizer_step.

This script is intended for the cache-analysis quick runs where each training
run writes three CSVs with the same timestamp stem:

    YYYYMMDDHHMMSS.csv          hardware/phase log
    YYYYMMDDHHMMSS_perf.csv     perf stat interval log
    YYYYMMDDHHMMSS_metrics.csv  loss/accuracy log

Main interpretation:
    The primary question is whether availability poisoning changes cache and
    memory-traffic behavior specifically during optimizer_step. The script
    therefore reports both all-phase summaries and optimizer_step-focused
    clean-vs-poisoned ratios.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PERF_COUNTER_COLUMNS = [
    "perf_cycles",
    "perf_instructions",
    "perf_branches",
    "perf_branch_misses",
    "perf_task_clock",
    "perf_context_switches",
    "perf_cpu_migrations",
    "perf_page_faults",
    "perf_l1d_cache_rd",
    "perf_l1d_cache_refill_rd",
    "perf_l1d_cache_wr",
    "perf_l1d_cache_refill_wr",
    "perf_l2d_cache_rd",
    "perf_l2d_cache_refill_rd",
    "perf_l2d_cache_wr",
    "perf_l2d_cache_refill_wr",
    "perf_bus_access_rd",
    "perf_bus_access_wr",
    "perf_mem_access",
    "perf_ase_spec",
    "perf_vfp_spec",
    "perf_inst_spec",
    "perf_br_retired",
    "perf_br_mis_pred_retired",
    "perf_l1d_cache",
    "perf_l1d_cache_refill",
    "perf_l1d_cache_wb",
    "perf_l2d_cache",
    "perf_l2d_cache_refill",
    "perf_l2d_cache_wb",
    "perf_bus_access",
]

HARDWARE_COLUMNS = [
    "system_cpu_core_0",
    "system_cpu_core_1",
    "system_cpu_core_2",
    "system_cpu_core_3",
    "system_cpu_freq_mhz",
    "system_cpu_freq_core_0_mhz",
    "system_cpu_freq_core_1_mhz",
    "system_cpu_freq_core_2_mhz",
    "system_cpu_freq_core_3_mhz",
    "system_memory_percent",
    "process_cpu_percent",
    "process_memory_rss",
    "process_ctx_switches_voluntary",
    "process_ctx_switches_involuntary",
    "process_minor_faults",
]

DERIVED_PERF_COLUMNS = [
    "ipc",
    "branch_miss_rate",
    "task_clock_utilization_pct",
    "context_switches_per_sec",
    "cpu_migrations_per_sec",
    "page_faults_per_sec",
    "l1d_read_refill_rate",
    "l1d_write_refill_rate",
    "l2d_read_refill_rate",
    "l2d_write_refill_rate",
    "l1d_refill_total",
    "l2d_refill_total",
    "bus_access_total",
    "l1d_refill_per_kinst",
    "l2d_refill_per_kinst",
    "bus_access_per_kinst",
    "mem_access_per_kinst",
    "ase_spec_per_kinst",
    "vfp_spec_per_kinst",
    "inst_spec_per_kinst",
    "l1d_refill_per_mcycle",
    "l2d_refill_per_mcycle",
    "bus_access_per_mcycle",
]

DEFAULT_PLOT_METRICS = [
    "perf_cycles",
    "perf_instructions",
    "perf_branches",
    "perf_branch_misses",
    "perf_task_clock",
    "perf_context_switches",
    "perf_cpu_migrations",
    "perf_page_faults",
    "perf_l1d_cache_rd",
    "perf_l1d_cache_refill_rd",
    "perf_l1d_cache_wr",
    "perf_l1d_cache_refill_wr",
    "perf_l2d_cache_rd",
    "perf_l2d_cache_refill_rd",
    "perf_l2d_cache_wr",
    "perf_l2d_cache_refill_wr",
    "perf_bus_access_rd",
    "perf_bus_access_wr",
    "perf_mem_access",
    "perf_ase_spec",
    "perf_vfp_spec",
    "perf_inst_spec",
    "perf_br_retired",
    "perf_br_mis_pred_retired",
    "perf_l1d_cache",
    "perf_l1d_cache_refill",
    "perf_l1d_cache_wb",
    "perf_l2d_cache",
    "perf_l2d_cache_refill",
    "perf_l2d_cache_wb",
    "perf_bus_access",
    "ipc",
    "branch_miss_rate",
    "task_clock_utilization_pct",
    "context_switches_per_sec",
    "cpu_migrations_per_sec",
    "page_faults_per_sec",
    "l1d_read_refill_rate",
    "l1d_write_refill_rate",
    "l2d_read_refill_rate",
    "l2d_write_refill_rate",
    "l1d_refill_per_kinst",
    "l2d_refill_per_kinst",
    "bus_access_per_kinst",
    "mem_access_per_kinst",
    "ase_spec_per_kinst",
    "vfp_spec_per_kinst",
    "inst_spec_per_kinst",
]

ID_COLUMNS = [
    "run_id",
    "source_file",
    "poisoning_method",
    "trial_id",
    "device_id",
    "host",
    "client_id",
    "model",
    "phase",
]


@dataclass(frozen=True)
class RunFiles:
    run_id: str
    hardware_csv: Optional[Path]
    perf_csv: Optional[Path]
    metrics_csv: Optional[Path]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def discover_run_files(input_dir: Path) -> List[RunFiles]:
    csvs = sorted(input_dir.rglob("*.csv"))
    grouped: Dict[str, Dict[str, Path]] = {}
    for path in csvs:
        stem = path.stem
        if stem.endswith("_perf"):
            run_id = stem[: -len("_perf")]
            kind = "perf"
        elif stem.endswith("_metrics"):
            run_id = stem[: -len("_metrics")]
            kind = "metrics"
        else:
            run_id = stem
            kind = "hardware"
        grouped.setdefault(run_id, {})[kind] = path

    runs = [
        RunFiles(
            run_id=run_id,
            hardware_csv=items.get("hardware"),
            perf_csv=items.get("perf"),
            metrics_csv=items.get("metrics"),
        )
        for run_id, items in sorted(grouped.items())
    ]
    if not runs:
        raise FileNotFoundError(f"No CSV files found under {input_dir}")
    return runs


def read_csv(path: Path, run_id: str, source_kind: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["run_id"] = run_id
    df["source_file"] = str(path)
    df["source_kind"] = source_kind
    if "timestamp_unix" in df.columns:
        df["timestamp_unix"] = pd.to_numeric(df["timestamp_unix"], errors="coerce")
        first = df["timestamp_unix"].dropna().iloc[0] if df["timestamp_unix"].notna().any() else np.nan
        df["relative_time_sec"] = df["timestamp_unix"] - first
    else:
        df["relative_time_sec"] = np.arange(len(df), dtype=float)
    return df


def load_all(runs: Sequence[RunFiles]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    perf_parts = []
    hw_parts = []
    metric_parts = []
    for run in runs:
        if run.perf_csv is not None:
            perf_parts.append(read_csv(run.perf_csv, run.run_id, "perf"))
        if run.hardware_csv is not None:
            hw_parts.append(read_csv(run.hardware_csv, run.run_id, "hardware"))
        if run.metrics_csv is not None:
            metric_parts.append(read_csv(run.metrics_csv, run.run_id, "metrics"))

    perf = pd.concat(perf_parts, ignore_index=True) if perf_parts else pd.DataFrame()
    hardware = pd.concat(hw_parts, ignore_index=True) if hw_parts else pd.DataFrame()
    metrics = pd.concat(metric_parts, ignore_index=True) if metric_parts else pd.DataFrame()
    return perf, hardware, metrics


def numeric(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")


def add_perf_derived(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    numeric(df, PERF_COUNTER_COLUMNS + ["perf_interval_ms", "perf_elapsed_sec"])

    eps = 1e-12
    interval_sec = df.get("perf_interval_ms", pd.Series(100.0, index=df.index)).fillna(100.0) / 1000.0
    for column in PERF_COUNTER_COLUMNS:
        if column in df.columns:
            df[f"{column}_per_sec"] = df[column] / interval_sec.replace(0, np.nan)

    cycles = df.get("perf_cycles", pd.Series(np.nan, index=df.index))
    instructions = df.get("perf_instructions", pd.Series(np.nan, index=df.index))
    branches = df.get("perf_branches", pd.Series(np.nan, index=df.index))
    branch_misses = df.get("perf_branch_misses", pd.Series(np.nan, index=df.index))
    task_clock_ms = df.get("perf_task_clock", pd.Series(np.nan, index=df.index))
    context_switches = df.get("perf_context_switches", pd.Series(np.nan, index=df.index))
    cpu_migrations = df.get("perf_cpu_migrations", pd.Series(np.nan, index=df.index))
    page_faults = df.get("perf_page_faults", pd.Series(np.nan, index=df.index))
    l1_rd = df.get("perf_l1d_cache_rd", pd.Series(np.nan, index=df.index))
    l1_rd_refill = df.get("perf_l1d_cache_refill_rd", pd.Series(np.nan, index=df.index))
    l1_wr = df.get("perf_l1d_cache_wr", pd.Series(np.nan, index=df.index))
    l1_wr_refill = df.get("perf_l1d_cache_refill_wr", pd.Series(np.nan, index=df.index))
    l2_rd = df.get("perf_l2d_cache_rd", pd.Series(np.nan, index=df.index))
    l2_rd_refill = df.get("perf_l2d_cache_refill_rd", pd.Series(np.nan, index=df.index))
    l2_wr = df.get("perf_l2d_cache_wr", pd.Series(np.nan, index=df.index))
    l2_wr_refill = df.get("perf_l2d_cache_refill_wr", pd.Series(np.nan, index=df.index))
    bus_rd = df.get("perf_bus_access_rd", pd.Series(np.nan, index=df.index))
    bus_wr = df.get("perf_bus_access_wr", pd.Series(np.nan, index=df.index))
    mem_access = df.get("perf_mem_access", pd.Series(np.nan, index=df.index))
    ase_spec = df.get("perf_ase_spec", pd.Series(np.nan, index=df.index))
    vfp_spec = df.get("perf_vfp_spec", pd.Series(np.nan, index=df.index))
    inst_spec = df.get("perf_inst_spec", pd.Series(np.nan, index=df.index))

    df["ipc"] = instructions / (cycles + eps)
    df["branch_miss_rate"] = branch_misses / (branches + eps)
    df["task_clock_utilization_pct"] = task_clock_ms / (interval_sec * 1000.0 + eps) * 100.0
    df["context_switches_per_sec"] = context_switches / (interval_sec + eps)
    df["cpu_migrations_per_sec"] = cpu_migrations / (interval_sec + eps)
    df["page_faults_per_sec"] = page_faults / (interval_sec + eps)
    df["l1d_read_refill_rate"] = l1_rd_refill / (l1_rd + eps)
    df["l1d_write_refill_rate"] = l1_wr_refill / (l1_wr + eps)
    df["l2d_read_refill_rate"] = l2_rd_refill / (l2_rd + eps)
    df["l2d_write_refill_rate"] = l2_wr_refill / (l2_wr + eps)
    df["l1d_refill_total"] = l1_rd_refill.fillna(0) + l1_wr_refill.fillna(0)
    df["l2d_refill_total"] = l2_rd_refill.fillna(0) + l2_wr_refill.fillna(0)
    df["bus_access_total"] = bus_rd.fillna(0) + bus_wr.fillna(0)
    df["l1d_refill_per_kinst"] = df["l1d_refill_total"] / (instructions / 1000.0 + eps)
    df["l2d_refill_per_kinst"] = df["l2d_refill_total"] / (instructions / 1000.0 + eps)
    df["bus_access_per_kinst"] = df["bus_access_total"] / (instructions / 1000.0 + eps)
    df["mem_access_per_kinst"] = mem_access / (instructions / 1000.0 + eps)
    df["ase_spec_per_kinst"] = ase_spec / (instructions / 1000.0 + eps)
    df["vfp_spec_per_kinst"] = vfp_spec / (instructions / 1000.0 + eps)
    df["inst_spec_per_kinst"] = inst_spec / (instructions / 1000.0 + eps)
    df["l1d_refill_per_mcycle"] = df["l1d_refill_total"] / (cycles / 1_000_000.0 + eps)
    df["l2d_refill_per_mcycle"] = df["l2d_refill_total"] / (cycles / 1_000_000.0 + eps)
    df["bus_access_per_mcycle"] = df["bus_access_total"] / (cycles / 1_000_000.0 + eps)
    return df


def add_counter_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Add deltas for monotonic hardware counters, grouped by run.

    psutil context switches and /proc minor faults are cumulative counters. For
    phase summaries, the per-sample delta is more interpretable than the raw
    absolute value.
    """
    if df.empty:
        return df
    df = df.copy()
    counter_cols = [
        "process_ctx_switches_voluntary",
        "process_ctx_switches_involuntary",
        "process_minor_faults",
    ]
    numeric(df, HARDWARE_COLUMNS)
    for column in counter_cols:
        if column in df.columns:
            delta = df.groupby("run_id")[column].diff()
            df[f"{column}_delta"] = delta.clip(lower=0)
    return df


def summarize(df: pd.DataFrame, value_columns: Sequence[str], group_columns: Sequence[str]) -> pd.DataFrame:
    available = [column for column in value_columns if column in df.columns]
    if df.empty or not available:
        return pd.DataFrame()
    numeric(df, available)
    summary = (
        df.groupby(list(group_columns), dropna=False)[available]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in col if part != "").rstrip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in summary.columns
    ]
    return summary


def build_clean_ratio_table(summary: pd.DataFrame, metrics: Sequence[str], phase: str) -> pd.DataFrame:
    if summary.empty or "poisoning_method" not in summary.columns:
        return pd.DataFrame()
    rows = []
    phase_summary = summary[summary["phase"] == phase].copy() if "phase" in summary.columns else summary.copy()
    for metric in metrics:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        count_col = f"{metric}_count"
        if mean_col not in phase_summary.columns:
            continue
        clean_rows = phase_summary[phase_summary["poisoning_method"] == "clean"]
        if clean_rows.empty:
            continue
        clean_mean = float(clean_rows[mean_col].iloc[0])
        clean_std = float(clean_rows[std_col].iloc[0]) if std_col in clean_rows.columns else np.nan
        for _, row in phase_summary.iterrows():
            method = row["poisoning_method"]
            value = float(row[mean_col])
            rows.append(
                {
                    "phase": phase,
                    "metric": metric,
                    "poisoning_method": method,
                    "mean": value,
                    "std": row.get(std_col, np.nan),
                    "count": row.get(count_col, np.nan),
                    "clean_mean": clean_mean,
                    "clean_std": clean_std,
                    "delta_vs_clean": value - clean_mean,
                    "ratio_vs_clean": value / clean_mean if abs(clean_mean) > 1e-12 else np.nan,
                    "percent_change_vs_clean": ((value - clean_mean) / clean_mean * 100.0)
                    if abs(clean_mean) > 1e-12
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def condition_order(df: pd.DataFrame) -> List[str]:
    if "poisoning_method" not in df.columns:
        return []
    methods = [m for m in df["poisoning_method"].dropna().unique().tolist()]
    if "clean" in methods:
        return ["clean", *[m for m in methods if m != "clean"]]
    return methods


def color_for_method(method: str) -> str:
    colors = {
        "clean": "#1f77b4",
        "availability_shortcuts": "#d62728",
        "unlearnable_examples": "#ff7f0e",
        "random_label_flipping": "#9467bd",
        "target_label_flipping": "#2ca02c",
    }
    return colors.get(method, "#7f7f7f")


def plot_optimizer_ratios(ratio_df: pd.DataFrame, output_path: Path) -> None:
    if ratio_df.empty:
        return
    plot_df = ratio_df[ratio_df["poisoning_method"] != "clean"].copy()
    if plot_df.empty:
        return
    metrics = plot_df["metric"].drop_duplicates().tolist()
    methods = condition_order(plot_df)
    x = np.arange(len(metrics))
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(max(10, len(metrics) * 1.3), 5))
    for idx, method in enumerate(methods):
        sub = plot_df[plot_df["poisoning_method"] == method].set_index("metric")
        values = [sub.loc[m, "ratio_vs_clean"] if m in sub.index else np.nan for m in metrics]
        ax.bar(x + idx * width, values, width=width, label=method, color=color_for_method(method))
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(metrics, rotation=35, ha="right")
    ax.set_ylabel("Ratio vs clean mean")
    ax.set_title("optimizer_step perf-derived metrics: attack / clean")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_phase_bars(summary: pd.DataFrame, metrics: Sequence[str], output_dir: Path, prefix: str) -> None:
    if summary.empty or "phase" not in summary.columns:
        return
    methods = condition_order(summary)
    phases = [p for p in ["forward", "backward", "optimizer_step", "evaluation", "idle", "finished"] if p in set(summary["phase"])]
    for metric in metrics:
        mean_col = f"{metric}_mean"
        if mean_col not in summary.columns:
            continue
        fig, ax = plt.subplots(figsize=(9, 4.5))
        x = np.arange(len(phases))
        width = 0.8 / max(len(methods), 1)
        for idx, method in enumerate(methods):
            sub = summary[summary["poisoning_method"] == method].set_index("phase")
            values = [sub.loc[p, mean_col] if p in sub.index else np.nan for p in phases]
            ax.bar(x + idx * width, values, width=width, label=method, color=color_for_method(method))
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(phases, rotation=20, ha="right")
        ax.set_title(f"{metric} by phase")
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}_phase_bar_{metric}.png", dpi=180)
        plt.close(fig)


def plot_timeseries(df: pd.DataFrame, metrics: Sequence[str], output_dir: Path, prefix: str) -> None:
    if df.empty:
        return
    for metric in metrics:
        if metric not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(11, 4.5))
        for (method, run_id), sub in df.groupby(["poisoning_method", "run_id"], dropna=False):
            sub = sub.sort_values("relative_time_sec")
            ax.plot(
                sub["relative_time_sec"],
                sub[metric],
                label=f"{method}:{run_id}",
                color=color_for_method(str(method)),
                alpha=0.9,
                linewidth=1.2,
            )
        ax.set_title(f"{metric} over relative time")
        ax.set_xlabel("Relative time (sec)")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}_timeseries_{metric}.png", dpi=180)
        plt.close(fig)


def plot_phase_timeline(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty or "phase" not in df.columns:
        return
    phase_to_y = {
        "idle": 0,
        "forward": 1,
        "backward": 2,
        "optimizer_step": 3,
        "evaluation": 4,
        "finished": 5,
    }
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for (method, run_id), sub in df.groupby(["poisoning_method", "run_id"], dropna=False):
        sub = sub.sort_values("relative_time_sec")
        y = sub["phase"].map(phase_to_y)
        ax.scatter(
            sub["relative_time_sec"],
            y,
            s=10,
            label=f"{method}:{run_id}",
            color=color_for_method(str(method)),
            alpha=0.7,
        )
    ax.set_yticks(list(phase_to_y.values()))
    ax.set_yticklabels(list(phase_to_y.keys()))
    ax.set_xlabel("Relative time (sec)")
    ax.set_title("Training phase timeline")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_loss_metrics(metrics_df: pd.DataFrame, output_path: Path) -> None:
    if metrics_df.empty or "loss" not in metrics_df.columns:
        return
    df = metrics_df.copy()
    numeric(df, ["loss", "accuracy", "timestamp_unix"])
    batch_df = df[df.get("metric_event", "") == "train_batch"].copy()
    if batch_df.empty:
        return
    batch_df["batch_order"] = batch_df.groupby("run_id").cumcount()

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for (method, run_id), sub in batch_df.groupby(["poisoning_method", "run_id"], dropna=False):
        color = color_for_method(str(method))
        axes[0].plot(sub["batch_order"], sub["loss"], label=f"{method}:{run_id}", color=color)
        if "accuracy" in sub.columns:
            axes[1].plot(sub["batch_order"], sub["accuracy"], label=f"{method}:{run_id}", color=color)
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Train-batch loss")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_xlabel("Batch index in run")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_text_report(
    *,
    output_path: Path,
    runs: Sequence[RunFiles],
    perf: pd.DataFrame,
    ratio: pd.DataFrame,
    target_phase: str,
) -> None:
    lines = []
    lines.append("# Cache Analysis Report")
    lines.append("")
    lines.append(f"Runs discovered: {len(runs)}")
    for run in runs:
        lines.append(
            f"- {run.run_id}: hardware={bool(run.hardware_csv)} perf={bool(run.perf_csv)} metrics={bool(run.metrics_csv)}"
        )
    lines.append("")
    if not perf.empty:
        lines.append("Perf samples by poisoning_method and phase:")
        phase_counts = (
            perf.groupby(["poisoning_method", "phase"], dropna=False)
            .size()
            .reset_index(name="count")
            .to_string(index=False)
        )
        lines.append("```")
        lines.append(phase_counts)
        lines.append("```")
        lines.append("")
    if not ratio.empty:
        lines.append(f"All raw perf counters during `{target_phase}`:")
        raw = ratio[ratio["metric"].isin(PERF_COUNTER_COLUMNS)].copy()
        raw = raw.sort_values(["metric", "poisoning_method"])
        if not raw.empty:
            lines.append("```")
            lines.append(
                raw[
                    [
                        "poisoning_method",
                        "metric",
                        "mean",
                        "clean_mean",
                        "ratio_vs_clean",
                        "percent_change_vs_clean",
                    ]
                ].to_string(index=False)
            )
            lines.append("```")
            lines.append("")

        lines.append(f"All derived perf metrics during `{target_phase}`:")
        derived = ratio[ratio["metric"].isin(DERIVED_PERF_COLUMNS)].copy()
        derived = derived.sort_values(["metric", "poisoning_method"])
        if not derived.empty:
            lines.append("```")
            lines.append(
                derived[
                    [
                        "poisoning_method",
                        "metric",
                        "mean",
                        "clean_mean",
                        "ratio_vs_clean",
                        "percent_change_vs_clean",
                    ]
                ].to_string(index=False)
            )
            lines.append("```")
            lines.append("")

        lines.append(f"Top absolute percent changes vs clean during `{target_phase}`:")
        top = ratio[ratio["poisoning_method"] != "clean"].copy()
        if not top.empty:
            top["abs_percent_change"] = top["percent_change_vs_clean"].abs()
            top = top.sort_values("abs_percent_change", ascending=False).head(12)
            lines.append("```")
            lines.append(
                top[
                    [
                        "poisoning_method",
                        "metric",
                        "mean",
                        "clean_mean",
                        "ratio_vs_clean",
                        "percent_change_vs_clean",
                    ]
                ].to_string(index=False)
            )
            lines.append("```")
    ensure_dir(output_path.parent)
    output_path.write_text("\n".join(lines) + "\n")


def parse_metric_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze perf/cache logs from cache-analysis runs.")
    parser.add_argument("--input_dir", default="collected_logs")
    parser.add_argument("--output_dir", default="result")
    parser.add_argument("--target_phase", default="optimizer_step")
    parser.add_argument("--plot_metrics", default=",".join(DEFAULT_PLOT_METRICS))
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = ensure_dir(Path(args.output_dir))
    plot_dir = ensure_dir(output_dir / "plots")
    plot_metrics = parse_metric_list(args.plot_metrics)

    runs = discover_run_files(input_dir)
    perf, hardware, metrics = load_all(runs)
    perf = add_perf_derived(perf)
    hardware = add_counter_deltas(hardware)

    if not perf.empty:
        save_dataframe(perf, output_dir / "perf_samples_with_derived.csv")
    if not hardware.empty:
        save_dataframe(hardware, output_dir / "hardware_samples_with_deltas.csv")
    if not metrics.empty:
        save_dataframe(metrics, output_dir / "metrics_samples.csv")

    perf_summary = summarize(
        perf,
        [*PERF_COUNTER_COLUMNS, *[f"{c}_per_sec" for c in PERF_COUNTER_COLUMNS], *DERIVED_PERF_COLUMNS],
        ["poisoning_method", "phase"],
    )
    hardware_summary = summarize(
        hardware,
        [
            *HARDWARE_COLUMNS,
            "process_ctx_switches_voluntary_delta",
            "process_ctx_switches_involuntary_delta",
            "process_minor_faults_delta",
        ],
        ["poisoning_method", "phase"],
    )
    optimizer_perf = perf[perf["phase"] == args.target_phase].copy() if not perf.empty else pd.DataFrame()
    optimizer_hardware = (
        hardware[hardware["phase"] == args.target_phase].copy() if not hardware.empty else pd.DataFrame()
    )
    optimizer_perf_summary = summarize(
        optimizer_perf,
        [*PERF_COUNTER_COLUMNS, *DERIVED_PERF_COLUMNS],
        ["poisoning_method"],
    )
    optimizer_hardware_summary = summarize(
        optimizer_hardware,
        [
            *HARDWARE_COLUMNS,
            "process_ctx_switches_voluntary_delta",
            "process_ctx_switches_involuntary_delta",
            "process_minor_faults_delta",
        ],
        ["poisoning_method"],
    )

    perf_ratio = build_clean_ratio_table(perf_summary, [*PERF_COUNTER_COLUMNS, *DERIVED_PERF_COLUMNS], args.target_phase)
    hardware_ratio = build_clean_ratio_table(
        hardware_summary,
        [
            *HARDWARE_COLUMNS,
            "process_ctx_switches_voluntary_delta",
            "process_ctx_switches_involuntary_delta",
            "process_minor_faults_delta",
        ],
        args.target_phase,
    )

    save_dataframe(perf_summary, output_dir / "perf_summary_by_phase.csv")
    save_dataframe(hardware_summary, output_dir / "hardware_summary_by_phase.csv")
    save_dataframe(optimizer_perf_summary, output_dir / "optimizer_step_perf_summary.csv")
    save_dataframe(optimizer_hardware_summary, output_dir / "optimizer_step_hardware_summary.csv")
    save_dataframe(perf_ratio, output_dir / "optimizer_step_perf_ratio_vs_clean.csv")
    save_dataframe(hardware_ratio, output_dir / "optimizer_step_hardware_ratio_vs_clean.csv")

    plot_optimizer_ratios(perf_ratio[perf_ratio["metric"].isin(plot_metrics)], plot_dir / "optimizer_step_perf_ratio_vs_clean.png")
    plot_phase_bars(perf_summary, plot_metrics, plot_dir, "perf")
    plot_timeseries(perf, plot_metrics, plot_dir, "perf")
    plot_timeseries(optimizer_perf, plot_metrics, plot_dir, "optimizer_step_perf")
    plot_phase_timeline(perf, plot_dir / "perf_phase_timeline.png")

    hw_plot_metrics = [
        "system_cpu_core_0",
        "system_cpu_core_1",
        "system_cpu_core_2",
        "system_cpu_core_3",
        "system_cpu_freq_mhz",
        "system_cpu_freq_core_0_mhz",
        "system_cpu_freq_core_1_mhz",
        "system_cpu_freq_core_2_mhz",
        "system_cpu_freq_core_3_mhz",
        "process_cpu_percent",
        "process_memory_rss",
        "process_minor_faults_delta",
    ]
    plot_phase_bars(hardware_summary, hw_plot_metrics, plot_dir, "hardware")
    plot_timeseries(hardware, hw_plot_metrics, plot_dir, "hardware")
    plot_phase_timeline(hardware, plot_dir / "hardware_phase_timeline.png")
    plot_loss_metrics(metrics, plot_dir / "train_loss_accuracy.png")

    write_text_report(
        output_path=output_dir / "analysis_report.md",
        runs=runs,
        perf=perf,
        ratio=perf_ratio,
        target_phase=args.target_phase,
    )

    print(f"Discovered runs: {len(runs)}")
    print(f"Perf samples: {len(perf)}")
    print(f"Hardware samples: {len(hardware)}")
    print(f"Metrics samples: {len(metrics)}")
    print(f"Wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()
