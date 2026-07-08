#!/usr/bin/env python3
"""Analyze model-size experiment logs.

Goal:
    Check whether increasing CNN model size increases optimizer_step memory and
    CPU behavior differences between clean and availability-shortcut runs.

Inputs are hardware CSVs and *_metrics.csv files collected from
quick_model_size_check.zsh:

    collected_logs/model_size_quick/<host>/<model_label>/<timestamp>.csv
    collected_logs/model_size_quick/<host>/<model_label>/<timestamp>_metrics.csv

The script writes summary CSVs and plots under result/ by default.
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


PHASE_ORDER = ["forward", "backward", "optimizer_step", "evaluation", "idle", "finished"]

HARDWARE_METRICS = [
    "system_cpu_core_0",
    "system_cpu_core_1",
    "system_cpu_core_2",
    "system_cpu_core_3",
    "process_cpu_percent",
    "system_memory_percent",
    "process_memory_rss",
    "process_memory_vms",
    "process_memory_percent",
]

COUNTER_COLUMNS = [
    "process_ctx_switches_voluntary",
    "process_ctx_switches_involuntary",
    "process_minor_faults",
]

COUNTER_DELTA_COLUMNS = [f"{column}_delta" for column in COUNTER_COLUMNS]

SUMMARY_METRICS = [
    *HARDWARE_METRICS,
    *COUNTER_DELTA_COLUMNS,
]

MODEL_LABEL_BY_NAME = {
    "simple_cnn": "simple",
    "pam_cnn": "pam500mb",
}


@dataclass(frozen=True)
class RunFiles:
    run_id: str
    hardware_csv: Optional[Path]
    metrics_csv: Optional[Path]
    host_from_path: str
    model_label_from_path: str


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def discover_runs(input_dir: Path) -> List[RunFiles]:
    csvs = sorted(input_dir.rglob("*.csv"))
    grouped: Dict[Tuple[str, str, str], Dict[str, Path]] = {}
    for path in csvs:
        stem = path.stem
        if stem.endswith("_metrics"):
            run_id = stem[: -len("_metrics")]
            kind = "metrics"
        else:
            run_id = stem
            kind = "hardware"

        parts = path.parts
        host = ""
        model_label = ""
        if len(parts) >= 3:
            model_label = path.parent.name
            host = path.parent.parent.name
        key = (host, model_label, run_id)
        grouped.setdefault(key, {})[kind] = path

    runs = [
        RunFiles(
            run_id=run_id,
            hardware_csv=items.get("hardware"),
            metrics_csv=items.get("metrics"),
            host_from_path=host,
            model_label_from_path=model_label,
        )
        for (host, model_label, run_id), items in sorted(grouped.items())
    ]
    if not runs:
        raise FileNotFoundError(f"No CSV files found under {input_dir}")
    return runs


def read_csv(path: Path, run: RunFiles, source_kind: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["run_id"] = run.run_id
    df["source_file"] = str(path)
    df["source_kind"] = source_kind
    df["host_from_path"] = run.host_from_path
    df["model_label"] = run.model_label_from_path
    if "host" not in df.columns or df["host"].isna().all():
        df["host"] = run.host_from_path
    if "model" in df.columns:
        model_label_from_model = df["model"].map(MODEL_LABEL_BY_NAME).fillna(df["model"].astype(str))
        df["model_label"] = df["model_label"].where(df["model_label"].astype(str) != "", model_label_from_model)
    if "timestamp_unix" in df.columns:
        df["timestamp_unix"] = pd.to_numeric(df["timestamp_unix"], errors="coerce")
        first = df["timestamp_unix"].dropna().iloc[0] if df["timestamp_unix"].notna().any() else np.nan
        df["relative_time_sec"] = df["timestamp_unix"] - first
    else:
        df["relative_time_sec"] = np.arange(len(df), dtype=float)
    return df


def load_all(runs: Sequence[RunFiles]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    hardware_parts = []
    metrics_parts = []
    for run in runs:
        if run.hardware_csv is not None:
            hardware_parts.append(read_csv(run.hardware_csv, run, "hardware"))
        if run.metrics_csv is not None:
            metrics_parts.append(read_csv(run.metrics_csv, run, "metrics"))
    hardware = pd.concat(hardware_parts, ignore_index=True) if hardware_parts else pd.DataFrame()
    metrics = pd.concat(metrics_parts, ignore_index=True) if metrics_parts else pd.DataFrame()
    return hardware, metrics


def numeric(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")


def add_derived_hardware(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    numeric(df, [*HARDWARE_METRICS, *COUNTER_COLUMNS, "timestamp_unix", "model_estimated_pam_mb", "model_parameter_count"])
    for column in COUNTER_COLUMNS:
        if column in df.columns:
            delta = df.groupby("run_id")[column].diff()
            df[f"{column}_delta"] = delta.clip(lower=0)
    core_cols = [f"system_cpu_core_{idx}" for idx in range(4) if f"system_cpu_core_{idx}" in df.columns]
    if core_cols:
        df["system_cpu_mean_4cores"] = df[core_cols].mean(axis=1)
        df["system_cpu_max_4cores"] = df[core_cols].max(axis=1)
    if "process_memory_rss" in df.columns:
        df["process_memory_rss_mb"] = df["process_memory_rss"] / (1024 * 1024)
    if "process_memory_vms" in df.columns:
        df["process_memory_vms_mb"] = df["process_memory_vms"] / (1024 * 1024)
    return df


def phase_duration_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    duration = (
        df.groupby(["host", "model_label", "model", "poisoning_method", "phase", "run_id"], dropna=False)
        .agg(
            samples=("phase", "size"),
            duration_sec=("timestamp_unix", lambda s: max(len(s) - 1, 0) * 0.1),
            first_relative_sec=("relative_time_sec", "min"),
            last_relative_sec=("relative_time_sec", "max"),
        )
        .reset_index()
    )
    return duration


def summarize_by_phase(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [m for m in [*SUMMARY_METRICS, "system_cpu_mean_4cores", "system_cpu_max_4cores", "process_memory_rss_mb"] if m in df.columns]
    if df.empty or not metrics:
        return pd.DataFrame()
    summary = (
        df.groupby(["host", "model_label", "model", "poisoning_method", "phase"], dropna=False)[metrics]
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


def optimizer_summary(df: pd.DataFrame) -> pd.DataFrame:
    opt = df[df["phase"] == "optimizer_step"].copy()
    metrics = [m for m in [*SUMMARY_METRICS, "system_cpu_mean_4cores", "system_cpu_max_4cores", "process_memory_rss_mb"] if m in opt.columns]
    if opt.empty or not metrics:
        return pd.DataFrame()
    summary = (
        opt.groupby(["host", "model_label", "model", "poisoning_method", "run_id"], dropna=False)[metrics]
        .agg(["count", "mean", "std", "median", "min", "max", "sum"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in col if part != "").rstrip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in summary.columns
    ]
    return summary


def optimizer_ratio_vs_clean(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows = []
    id_cols = ["host", "model_label", "model"]
    metric_means = [col for col in summary.columns if col.endswith("_mean")]
    for keys, group in summary.groupby(id_cols, dropna=False):
        clean = group[group["poisoning_method"] == "clean"]
        if clean.empty:
            continue
        for mean_col in metric_means:
            metric = mean_col[: -len("_mean")]
            clean_mean = float(clean[mean_col].iloc[0])
            for _, row in group.iterrows():
                value = float(row[mean_col])
                host, model_label, model = keys
                rows.append(
                    {
                        "host": host,
                        "model_label": model_label,
                        "model": model,
                        "poisoning_method": row["poisoning_method"],
                        "metric": metric,
                        "mean": value,
                        "clean_mean": clean_mean,
                        "delta_vs_clean": value - clean_mean,
                        "ratio_vs_clean": value / clean_mean if abs(clean_mean) > 1e-12 else np.nan,
                        "percent_change_vs_clean": (value - clean_mean) / clean_mean * 100.0
                        if abs(clean_mean) > 1e-12
                        else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def amplification_summary(ratio: pd.DataFrame) -> pd.DataFrame:
    """Compare poisoned-vs-clean gap between PAM and simple for each device.

    A positive amplification_delta_abs means the absolute clean-vs-poison gap is
    larger for PAM than for simple for that metric on that host.
    """
    if ratio.empty:
        return pd.DataFrame()
    poison = ratio[ratio["poisoning_method"] != "clean"].copy()
    rows = []
    for (host, metric), group in poison.groupby(["host", "metric"], dropna=False):
        simple = group[group["model_label"] == "simple"]
        pam = group[group["model_label"].str.startswith("pam", na=False)]
        if simple.empty or pam.empty:
            continue
        simple_pct = float(simple["percent_change_vs_clean"].iloc[0])
        pam_pct = float(pam["percent_change_vs_clean"].iloc[0])
        rows.append(
            {
                "host": host,
                "metric": metric,
                "simple_percent_change": simple_pct,
                "pam_percent_change": pam_pct,
                "amplification_delta_abs": abs(pam_pct) - abs(simple_pct),
                "amplification_delta_signed": pam_pct - simple_pct,
            }
        )
    return pd.DataFrame(rows)


def metrics_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()
    df = metrics_df.copy()
    numeric(df, ["loss", "accuracy", "num_examples"])
    event_col = "metric_event"
    if event_col not in df.columns:
        return pd.DataFrame()
    keep = df[df[event_col].isin(["train_summary", "eval_summary", "train_epoch"])].copy()
    if keep.empty:
        return pd.DataFrame()
    cols = [
        "host",
        "model_label",
        "model",
        "poisoning_method",
        "run_id",
        "metric_event",
        "metric_split",
        "loss",
        "accuracy",
        "num_examples",
    ]
    return keep[[col for col in cols if col in keep.columns]]


def save_df(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def color_for_method(method: str) -> str:
    return {
        "clean": "#1f77b4",
        "availability_shortcuts": "#d62728",
    }.get(str(method), "#7f7f7f")


def host_label(host: str) -> str:
    return {
        "192.168.0.112": "RPI4",
        "192.168.0.141": "Jetson-CPU",
    }.get(str(host), str(host))


def plot_optimizer_ratios(ratio: pd.DataFrame, output_dir: Path, metrics: Sequence[str]) -> None:
    if ratio.empty:
        return
    poison = ratio[ratio["poisoning_method"] != "clean"].copy()
    if poison.empty:
        return
    for metric in metrics:
        sub = poison[poison["metric"] == metric].copy()
        if sub.empty:
            continue
        hosts = sub["host"].drop_duplicates().tolist()
        models = sub["model_label"].drop_duplicates().tolist()
        x = np.arange(len(hosts))
        width = 0.8 / max(len(models), 1)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for idx, model_label in enumerate(models):
            model_sub = sub[sub["model_label"] == model_label].set_index("host")
            values = [model_sub.loc[h, "percent_change_vs_clean"] if h in model_sub.index else np.nan for h in hosts]
            ax.bar(x + idx * width, values, width=width, label=model_label)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels([host_label(h) for h in hosts])
        ax.set_ylabel("Poisoned vs clean change (%)")
        ax.set_title(f"optimizer_step clean-vs-shortcut gap: {metric}")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"optimizer_gap_{metric}.png", dpi=180)
        plt.close(fig)


def plot_phase_durations(duration: pd.DataFrame, output_dir: Path) -> None:
    if duration.empty:
        return
    for host in duration["host"].drop_duplicates():
        host_df = duration[duration["host"] == host]
        for model_label in host_df["model_label"].drop_duplicates():
            sub = host_df[host_df["model_label"] == model_label]
            phases = [p for p in PHASE_ORDER if p in set(sub["phase"])]
            methods = [m for m in ["clean", "availability_shortcuts"] if m in set(sub["poisoning_method"])]
            if not phases or not methods:
                continue
            x = np.arange(len(phases))
            width = 0.8 / max(len(methods), 1)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for idx, method in enumerate(methods):
                method_sub = sub[sub["poisoning_method"] == method].set_index("phase")
                values = [method_sub.loc[p, "duration_sec"] if p in method_sub.index else np.nan for p in phases]
                ax.bar(x + idx * width, values, width=width, label=method, color=color_for_method(method))
            ax.set_xticks(x + width * (len(methods) - 1) / 2)
            ax.set_xticklabels(phases, rotation=25, ha="right")
            ax.set_ylabel("Duration estimate (sec)")
            ax.set_title(f"{host_label(host)} {model_label}: phase duration")
            ax.legend()
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(output_dir / f"phase_duration_{host}_{model_label}.png", dpi=180)
            plt.close(fig)


def plot_timeseries(df: pd.DataFrame, output_dir: Path, metrics: Sequence[str]) -> None:
    if df.empty:
        return
    for host in df["host"].drop_duplicates():
        for model_label in df[df["host"] == host]["model_label"].drop_duplicates():
            sub = df[(df["host"] == host) & (df["model_label"] == model_label)].copy()
            for metric in metrics:
                if metric not in sub.columns:
                    continue
                fig, ax = plt.subplots(figsize=(10, 4.5))
                for method, method_sub in sub.groupby("poisoning_method", dropna=False):
                    method_sub = method_sub.sort_values("relative_time_sec")
                    ax.plot(
                        method_sub["relative_time_sec"],
                        method_sub[metric],
                        label=str(method),
                        color=color_for_method(str(method)),
                        linewidth=1.1,
                    )
                ax.set_xlabel("Relative time (sec)")
                ax.set_ylabel(metric)
                ax.set_title(f"{host_label(host)} {model_label}: {metric}")
                ax.grid(alpha=0.25)
                ax.legend()
                fig.tight_layout()
                fig.savefig(output_dir / f"timeseries_{host}_{model_label}_{metric}.png", dpi=180)
                plt.close(fig)


def write_report(
    output_path: Path,
    runs: Sequence[RunFiles],
    ratio: pd.DataFrame,
    amplification: pd.DataFrame,
    duration: pd.DataFrame,
) -> None:
    lines = ["# Model Size Analysis Report", ""]
    lines.append(f"Runs discovered: {len(runs)}")
    for run in runs:
        lines.append(
            f"- {run.host_from_path}/{run.model_label_from_path}/{run.run_id}: "
            f"hardware={bool(run.hardware_csv)} metrics={bool(run.metrics_csv)}"
        )
    lines.append("")

    if not duration.empty:
        lines.append("Optimizer-step duration estimate:")
        opt_dur = duration[duration["phase"] == "optimizer_step"].copy()
        lines.append("```")
        lines.append(
            opt_dur[
                ["host", "model_label", "poisoning_method", "samples", "duration_sec"]
            ].to_string(index=False)
        )
        lines.append("```")
        lines.append("")

    if not ratio.empty:
        key_metrics = [
            "system_cpu_mean_4cores",
            "process_cpu_percent",
            "process_memory_rss_mb",
            "process_ctx_switches_voluntary_delta",
            "process_ctx_switches_involuntary_delta",
            "process_minor_faults_delta",
        ]
        key = ratio[(ratio["poisoning_method"] != "clean") & (ratio["metric"].isin(key_metrics))].copy()
        lines.append("Optimizer-step poisoned-vs-clean gaps:")
        lines.append("```")
        lines.append(
            key[
                [
                    "host",
                    "model_label",
                    "metric",
                    "mean",
                    "clean_mean",
                    "percent_change_vs_clean",
                ]
            ].to_string(index=False)
        )
        lines.append("```")
        lines.append("")

    if not amplification.empty:
        lines.append("PAM amplification vs simple, positive means larger absolute poisoned-vs-clean gap:")
        top = amplification.sort_values("amplification_delta_abs", ascending=False).head(20)
        lines.append("```")
        lines.append(top.to_string(index=False))
        lines.append("```")
    ensure_dir(output_path.parent)
    output_path.write_text("\n".join(lines) + "\n")


def parse_metric_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze model-size local ML hardware logs.")
    parser.add_argument("--input_dir", default="collected_logs")
    parser.add_argument("--output_dir", default="result")
    parser.add_argument(
        "--plot_metrics",
        default="system_cpu_mean_4cores,process_cpu_percent,process_memory_rss_mb,process_minor_faults_delta",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = ensure_dir(Path(args.output_dir))
    plot_dir = ensure_dir(output_dir / "plots")
    plot_metrics = parse_metric_list(args.plot_metrics)

    runs = discover_runs(input_dir)
    hardware, metrics = load_all(runs)
    hardware = add_derived_hardware(hardware)

    duration = phase_duration_summary(hardware)
    phase_summary = summarize_by_phase(hardware)
    opt_summary = optimizer_summary(hardware)
    ratio = optimizer_ratio_vs_clean(opt_summary)
    amplification = amplification_summary(ratio)
    train_metrics = metrics_summary(metrics)

    save_df(hardware, output_dir / "hardware_samples_with_deltas.csv")
    save_df(metrics, output_dir / "metrics_samples.csv")
    save_df(duration, output_dir / "phase_duration_summary.csv")
    save_df(phase_summary, output_dir / "hardware_summary_by_phase.csv")
    save_df(opt_summary, output_dir / "optimizer_step_hardware_summary.csv")
    save_df(ratio, output_dir / "optimizer_step_ratio_vs_clean.csv")
    save_df(amplification, output_dir / "pam_amplification_vs_simple.csv")
    save_df(train_metrics, output_dir / "train_eval_metrics_summary.csv")

    plot_optimizer_ratios(ratio, plot_dir, plot_metrics)
    plot_phase_durations(duration, plot_dir)
    plot_timeseries(hardware, plot_dir, plot_metrics)

    write_report(output_dir / "analysis_report.md", runs, ratio, amplification, duration)

    print(f"Discovered runs: {len(runs)}")
    print(f"Hardware samples: {len(hardware)}")
    print(f"Metrics samples: {len(metrics)}")
    print(f"Wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()
