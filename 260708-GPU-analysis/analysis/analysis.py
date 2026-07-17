#!/usr/bin/env python3
"""Analyze Jetson jtop metrics by training phase.

This script compares clean vs availability_shortcuts for CUDA local-ML runs.
It uses only training phases: forward, backward, and optimizer_step.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


TRAINING_PHASES = ["forward", "backward", "optimizer_step"]
CONDITIONS = ["clean", "availability_shortcuts"]
GPU_METRICS = [
    "jtop_gpu",
    "jtop_ram",
    "jtop_power_tot",
    "jtop_power_vdd_cpu_gpu_cv",
    "jtop_power_vdd_soc",
    "jtop_temp_cpu",
    "jtop_temp_gpu",
    "jtop_temp_soc0",
    "jtop_temp_soc1",
    "jtop_temp_soc2",
    "jtop_temp_tj",
]
META_COLUMNS = [
    "device",
    "source_file",
    "poisoning_method",
    "trial_id",
    "run_role",
    "phase",
    "torch_device",
    "cuda_device_name",
]


def is_hardware_csv(path: Path) -> bool:
    return path.suffix == ".csv" and not path.name.endswith("_metrics.csv")


def read_one_csv(path: Path, device: str) -> tuple[pd.DataFrame | None, Dict[str, object]]:
    header = pd.read_csv(path, nrows=0)
    missing_metrics = [col for col in GPU_METRICS if col not in header.columns]
    status: Dict[str, object] = {
        "device": device,
        "source_file": path.name,
        "has_jtop_columns": len(missing_metrics) == 0,
        "missing_jtop_columns": ",".join(missing_metrics),
        "n_rows": 0,
        "n_training_rows": 0,
        "n_valid_jtop_rows": 0,
        "poisoning_method": "",
        "trial_id": "",
    }
    if missing_metrics:
        return None, status

    usecols = [col for col in META_COLUMNS[2:] + GPU_METRICS if col in header.columns]
    df = pd.read_csv(path, usecols=usecols)
    status["n_rows"] = len(df)
    if df.empty:
        return None, status

    df["device"] = device
    df["source_file"] = path.name
    for col in META_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    status["poisoning_method"] = str(df["poisoning_method"].iloc[0])
    status["trial_id"] = str(df["trial_id"].iloc[0])

    df = df[df["phase"].isin(TRAINING_PHASES)].copy()
    df = df[df["poisoning_method"].isin(CONDITIONS)].copy()
    status["n_training_rows"] = len(df)
    if df.empty:
        return None, status

    for col in GPU_METRICS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    valid_mask = df[GPU_METRICS].notna().any(axis=1)
    status["n_valid_jtop_rows"] = int(valid_mask.sum())
    df = df[valid_mask].copy()
    if df.empty:
        return None, status

    return df[META_COLUMNS + GPU_METRICS], status


def load_logs(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    statuses: List[Dict[str, object]] = []
    for device_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        for path in sorted(device_dir.rglob("*.csv")):
            if not is_hardware_csv(path):
                continue
            df, status = read_one_csv(path, device_dir.name)
            statuses.append(status)
            if df is not None:
                frames.append(df)
    if not frames:
        return pd.DataFrame(columns=META_COLUMNS + GPU_METRICS), pd.DataFrame(statuses)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(statuses)


def build_run_phase_summary(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "device",
        "source_file",
        "poisoning_method",
        "trial_id",
        "run_role",
        "phase",
        "torch_device",
        "cuda_device_name",
    ]
    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False, sort=True):
        row = dict(zip(group_cols, keys))
        row["n_samples"] = len(sub)
        for metric in GPU_METRICS:
            values = sub[metric].dropna()
            row[f"{metric}_mean"] = float(values.mean()) if not values.empty else float("nan")
            row[f"{metric}_std"] = float(values.std(ddof=0)) if len(values) > 1 else 0.0
            row[f"{metric}_min"] = float(values.min()) if not values.empty else float("nan")
            row[f"{metric}_max"] = float(values.max()) if not values.empty else float("nan")
            row[f"{metric}_median"] = float(values.median()) if not values.empty else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def build_phase_summary(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["device", "poisoning_method", "phase"]
    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False, sort=True):
        row = dict(zip(group_cols, keys))
        row["n_samples"] = len(sub)
        row["n_runs"] = sub["source_file"].nunique()
        for metric in GPU_METRICS:
            values = sub[metric].dropna()
            row[f"{metric}_mean"] = float(values.mean()) if not values.empty else float("nan")
            row[f"{metric}_std"] = float(values.std(ddof=0)) if len(values) > 1 else 0.0
            row[f"{metric}_min"] = float(values.min()) if not values.empty else float("nan")
            row[f"{metric}_max"] = float(values.max()) if not values.empty else float("nan")
            row[f"{metric}_median"] = float(values.median()) if not values.empty else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def build_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (device, phase), sub in summary.groupby(["device", "phase"], sort=True):
        clean = sub[sub["poisoning_method"] == "clean"]
        attack = sub[sub["poisoning_method"] == "availability_shortcuts"]
        if clean.empty or attack.empty:
            continue
        clean_row = clean.iloc[0]
        attack_row = attack.iloc[0]
        row = {
            "device": device,
            "phase": phase,
            "clean_n_samples": int(clean_row["n_samples"]),
            "attack_n_samples": int(attack_row["n_samples"]),
            "clean_n_runs": int(clean_row["n_runs"]),
            "attack_n_runs": int(attack_row["n_runs"]),
        }
        for metric in GPU_METRICS:
            c = float(clean_row[f"{metric}_mean"])
            a = float(attack_row[f"{metric}_mean"])
            row[f"{metric}_clean_mean"] = c
            row[f"{metric}_attack_mean"] = a
            row[f"{metric}_delta"] = a - c
            row[f"{metric}_ratio"] = a / c if c != 0 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _ensure_matplotlib(output_dir: Path):
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib_cache"))
    import matplotlib.pyplot as plt

    return plt


def plot_metric_bars(summary: pd.DataFrame, output_dir: Path) -> None:
    plt = _ensure_matplotlib(output_dir)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    phase_order = TRAINING_PHASES
    colors = {"clean": "tab:blue", "availability_shortcuts": "tab:orange"}
    for metric in GPU_METRICS:
        metric_col = f"{metric}_mean"
        devices = sorted(summary["device"].dropna().unique())
        if not devices:
            continue
        fig, axes = plt.subplots(
            len(devices),
            1,
            figsize=(9, max(3.0 * len(devices), 3.5)),
            squeeze=False,
            sharex=True,
        )
        for ax, device in zip(axes[:, 0], devices):
            sub = summary[summary["device"] == device]
            x = range(len(phase_order))
            width = 0.36
            for offset, condition in [(-width / 2, "clean"), (width / 2, "availability_shortcuts")]:
                values = []
                for phase in phase_order:
                    match = sub[(sub["phase"] == phase) & (sub["poisoning_method"] == condition)]
                    values.append(float(match[metric_col].iloc[0]) if not match.empty else float("nan"))
                ax.bar([i + offset for i in x], values, width=width, label=condition, color=colors[condition])
            ax.set_title(f"{device} / {metric}")
            ax.set_ylabel("mean")
            ax.set_xticks(list(x))
            ax.set_xticklabels(phase_order)
            ax.grid(axis="y", alpha=0.3)
            ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{metric}_clean_vs_availability_by_phase.png", dpi=180)
        plt.close(fig)


def plot_delta_heatmap(comparison: pd.DataFrame, output_dir: Path) -> None:
    if comparison.empty:
        return
    plt = _ensure_matplotlib(output_dir)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for device, sub in comparison.groupby("device", sort=True):
        matrix = []
        ylabels = []
        for metric in GPU_METRICS:
            values = []
            for phase in TRAINING_PHASES:
                match = sub[sub["phase"] == phase]
                values.append(float(match[f"{metric}_delta"].iloc[0]) if not match.empty else float("nan"))
            matrix.append(values)
            ylabels.append(metric)

        fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(ylabels))))
        im = ax.imshow(matrix, aspect="auto", cmap="coolwarm")
        ax.set_title(f"{device}: availability_shortcuts - clean")
        ax.set_xticks(range(len(TRAINING_PHASES)))
        ax.set_xticklabels(TRAINING_PHASES)
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels)
        fig.colorbar(im, ax=ax, label="mean delta")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{device}_gpu_metric_delta_heatmap.png", dpi=180)
        plt.close(fig)


def print_report(summary: pd.DataFrame, comparison: pd.DataFrame, statuses: pd.DataFrame) -> None:
    print("GPU/jtop phase analysis")
    print(f"usable rows: {int(summary['n_samples'].sum()) if not summary.empty else 0}")
    if not statuses.empty:
        missing = statuses[(statuses["has_jtop_columns"] == False) | (statuses["n_valid_jtop_rows"] == 0)]
        print(f"hardware csv files checked: {len(statuses)}")
        print(f"files without usable jtop rows: {len(missing)}")
    if comparison.empty:
        print("No clean-vs-availability comparison rows were available.")
        return

    display_metrics = ["jtop_gpu", "jtop_power_tot", "jtop_power_vdd_cpu_gpu_cv", "jtop_temp_gpu", "jtop_ram"]
    for _, row in comparison.sort_values(["device", "phase"]).iterrows():
        print(f"\n{row['device']} phase={row['phase']}")
        print(
            f"  samples clean/attack: {row['clean_n_samples']}/{row['attack_n_samples']} "
            f"runs clean/attack: {row['clean_n_runs']}/{row['attack_n_runs']}"
        )
        for metric in display_metrics:
            print(
                f"  {metric}: clean={row[f'{metric}_clean_mean']:.3f} "
                f"attack={row[f'{metric}_attack_mean']:.3f} "
                f"delta={row[f'{metric}_delta']:.3f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="collected_logs")
    parser.add_argument("--output_dir", default="result")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, statuses = load_logs(input_dir)
    statuses.to_csv(output_dir / "gpu_jtop_file_status.csv", index=False)
    if df.empty:
        raise SystemExit(f"No usable jtop rows found under {input_dir}")

    df.to_csv(output_dir / "gpu_jtop_training_rows.csv", index=False)
    run_phase_summary = build_run_phase_summary(df)
    phase_summary = build_phase_summary(df)
    comparison = build_comparison(phase_summary)

    run_phase_summary.to_csv(output_dir / "gpu_jtop_run_phase_summary.csv", index=False)
    phase_summary.to_csv(output_dir / "gpu_jtop_phase_summary.csv", index=False)
    comparison.to_csv(output_dir / "gpu_jtop_clean_vs_availability_comparison.csv", index=False)

    plot_metric_bars(phase_summary, output_dir)
    plot_delta_heatmap(comparison, output_dir)
    print_report(phase_summary, comparison, statuses)
    print(f"\nSaved outputs to {output_dir}")


if __name__ == "__main__":
    main()
