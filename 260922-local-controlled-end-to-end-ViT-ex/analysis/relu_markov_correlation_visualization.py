#!/usr/bin/env python3
"""Compare ReLU output sparsity with downstream MaxPool Markov entropy."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba


SCENARIOS = [
    "baseline",
    "moderate_augmentation",
    "strong_augmentation",
    "availability_shortcuts",
    "badsampler",
]
LABELS = {
    "baseline": "Clean",
    "moderate_augmentation": "Moderate aug.",
    "strong_augmentation": "Strong aug.",
    "availability_shortcuts": "Availability shortcuts",
    "badsampler": "BadSampler",
}
COLORS = {
    "baseline": "#2878B5",
    "moderate_augmentation": "#59A14F",
    "strong_augmentation": "#F28E2B",
    "availability_shortcuts": "#D62728",
    "badsampler": "#9467BD",
}
RELU_TO_POOL = {2: 3, 6: 7}


def plot_epoch_trajectory(
    ax: plt.Axes,
    points: pd.DataFrame,
    y_column: str,
    scenario: str,
) -> None:
    """Draw an epoch-ordered trajectory with increasing opacity."""
    points = points.sort_values("epoch")
    x = points["relu_output_zero_rate"].to_numpy(float) * 100.0
    y = points[y_column].to_numpy(float)
    epochs = points["epoch"].to_numpy(int)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y, epochs = x[finite], y[finite], epochs[finite]
    if not len(x):
        return

    color = COLORS[scenario]
    alphas = np.linspace(0.22, 1.0, len(x))
    point_colors = [to_rgba(color, alpha) for alpha in alphas]
    ax.plot(x, y, color=color, alpha=0.34, linewidth=1.1, zorder=1)
    ax.scatter(
        x,
        y,
        s=34,
        c=point_colors,
        edgecolors="none",
        label=LABELS[scenario],
        zorder=2,
    )

    # Arrow every three transitions keeps the direction readable without covering points.
    for index in range(0, len(x) - 1, 3):
        target = min(index + 1, len(x) - 1)
        ax.annotate(
            "",
            xy=(x[target], y[target]),
            xytext=(x[index], y[index]),
            arrowprops={"arrowstyle": "-|>", "color": color, "alpha": 0.75, "lw": 1.2},
            zorder=3,
        )

    ax.scatter(x[0], y[0], s=72, facecolors="white", edgecolors=color, linewidths=1.8, zorder=4)
    ax.scatter(x[-1], y[-1], s=105, marker="*", color=color, edgecolors="white", linewidths=0.6, zorder=4)
    ax.annotate(f"start e{epochs[0]}", (x[0], y[0]), xytext=(5, 5), textcoords="offset points", color=color, fontsize=7)
    ax.annotate(f"end e{epochs[-1]}", (x[-1], y[-1]), xytext=(5, -10), textcoords="offset points", color=color, fontsize=7)


def read_csvs(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    if not frames:
        raise FileNotFoundError("No matching CSV files were found.")
    return pd.concat(frames, ignore_index=True)


def load_joined(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = read_csvs(sorted(input_dir.rglob("*_entropy_summary.csv")))
    markov = read_csvs(sorted(input_dir.rglob("*_maxpool_markov.csv")))
    perf = read_csvs(sorted(input_dir.rglob("*_layer_perf.csv")))

    relu = summary[
        (summary["layer_type"] == "ReLU")
        & (summary["metric"] == "output_zero_rate")
        & (summary["phase"] == "forward")
        & summary["layer_index"].isin(RELU_TO_POOL)
    ].copy()
    relu["pool_layer_index"] = relu["layer_index"].map(RELU_TO_POOL)
    relu = relu.rename(
        columns={
            "layer_index": "relu_layer_index",
            "layer_name": "relu_layer_name",
            "mean": "relu_output_zero_rate",
            "std": "relu_output_zero_rate_std",
        }
    )

    keys = ["device_id", "trial_id", "scenario", "experiment_mode", "epoch"]
    markov_epoch = (
        markov.groupby(keys + ["layer_index", "layer_name"], as_index=False)
        .agg(
            markov_entropy_bits=("position_markov_entropy_rate_bits", "mean"),
            markov_entropy_std=("position_markov_entropy_rate_bits", "std"),
            markov_batches=("position_markov_entropy_rate_bits", "size"),
        )
        .rename(columns={"layer_index": "pool_layer_index", "layer_name": "pool_layer_name"})
    )
    joined = relu.merge(markov_epoch, on=keys + ["pool_layer_index"], how="inner")
    joined = joined[joined["scenario"].isin(SCENARIOS)].copy()
    if joined.empty:
        raise ValueError("No matching ReLU/MaxPool epoch observations were found.")

    perf = perf[
        (perf["phase"] == "forward")
        & (perf["layer_type"] == "MaxPool2d")
        & perf["layer_index"].isin(RELU_TO_POOL.values())
        & (perf["perf_status"] == "ok")
    ].copy()
    for column in ["perf_branches", "perf_branch_misses", "perf_branches_running_pct", "perf_branch_misses_running_pct"]:
        perf[column] = pd.to_numeric(perf[column], errors="coerce")
    perf = perf[
        (perf["perf_branches_running_pct"] >= 95.0)
        & (perf["perf_branch_misses_running_pct"] >= 95.0)
        & (perf["perf_branches"] > 0)
    ]
    perf_epoch = (
        perf.groupby(keys + ["layer_index", "layer_name"], as_index=False)
        .agg(
            branch_misses_per_invocation=("perf_branch_misses", "mean"),
            branch_misses_std=("perf_branch_misses", "std"),
            branch_misses_total=("perf_branch_misses", "sum"),
            branches_total=("perf_branches", "sum"),
            perf_invocations=("perf_branch_misses", "size"),
        )
        .rename(columns={"layer_index": "pool_layer_index", "layer_name": "pool_layer_name"})
    )
    perf_epoch["branch_miss_fraction"] = perf_epoch["branch_misses_total"] / perf_epoch["branches_total"]
    perf_epoch["branch_miss_percent"] = 100.0 * perf_epoch["branch_miss_fraction"]
    hardware = relu.merge(perf_epoch, on=keys + ["pool_layer_index"], how="inner")
    hardware = hardware[hardware["scenario"].isin(SCENARIOS)].copy()
    if hardware.empty:
        raise ValueError("No matching ReLU/MaxPool branch-counter observations were found.")
    return joined, hardware


def correlation_rows(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["device_id", "trial_id", "experiment_mode", "relu_layer_name", "pool_layer_name"]
    for keys, group in data.groupby(group_columns, dropna=False):
        x = group["relu_output_zero_rate"].to_numpy(float)
        y = group["markov_entropy_bits"].to_numpy(float)
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        pearson = float(np.corrcoef(x, y)[0, 1]) if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0 else np.nan
        spearman = float(pd.Series(x).corr(pd.Series(y), method="spearman")) if len(x) >= 2 else np.nan
        rows.append(dict(zip(group_columns, keys)) | {"pearson_r": pearson, "spearman_rho": spearman, "observations": len(x)})
    return pd.DataFrame(rows)


def plot_device(data: pd.DataFrame, output_dir: Path, device: str) -> Path:
    subset = data[data["device_id"].astype(str) == device]
    modes = [mode for mode in ["train", "frozen_replay"] if mode in set(subset["experiment_mode"])]
    pairs = sorted(subset[["relu_layer_index", "pool_layer_index"]].drop_duplicates().itertuples(index=False, name=None))
    fig, axes = plt.subplots(len(pairs), len(modes), figsize=(7 * len(modes), 5 * len(pairs)), squeeze=False)

    for row, (relu_index, pool_index) in enumerate(pairs):
        for col, mode in enumerate(modes):
            ax = axes[row, col]
            panel = subset[(subset["relu_layer_index"] == relu_index) & (subset["experiment_mode"] == mode)]
            for scenario in SCENARIOS:
                points = panel[panel["scenario"] == scenario]
                plot_epoch_trajectory(ax, points, "markov_entropy_bits", scenario)
            x = panel["relu_output_zero_rate"].to_numpy(float) * 100
            y = panel["markov_entropy_bits"].to_numpy(float)
            finite = np.isfinite(x) & np.isfinite(y)
            if finite.sum() >= 2 and np.std(x[finite]) > 0:
                slope, intercept = np.polyfit(x[finite], y[finite], 1)
                line_x = np.linspace(x[finite].min(), x[finite].max(), 100)
                ax.plot(line_x, slope * line_x + intercept, color="black", linewidth=1.3, linestyle="--")
                r = np.corrcoef(x[finite], y[finite])[0, 1]
                ax.text(0.03, 0.97, f"Pearson r = {r:.3f}", transform=ax.transAxes, va="top")
            ax.set_title(f"{mode}: ReLU {relu_index} -> MaxPool {pool_index}")
            ax.set_xlabel("ReLU output zero rate (%)")
            ax.set_ylabel("MaxPool position Markov entropy (bits)")
            ax.grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
    fig.suptitle(f"ReLU sparsity vs downstream MaxPool Markov entropy: {device}", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = output_dir / f"relu_zero_rate_vs_maxpool_markov_{device}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_hardware_device(
    data: pd.DataFrame,
    output_dir: Path,
    device: str,
    metric: str,
    ylabel: str,
) -> Path:
    subset = data[data["device_id"].astype(str) == device]
    modes = [mode for mode in ["train", "frozen_replay"] if mode in set(subset["experiment_mode"])]
    pairs = sorted(subset[["relu_layer_index", "pool_layer_index"]].drop_duplicates().itertuples(index=False, name=None))
    fig, axes = plt.subplots(len(pairs), len(modes), figsize=(7 * len(modes), 5 * len(pairs)), squeeze=False)
    correlation_records = []

    for row, (relu_index, pool_index) in enumerate(pairs):
        for col, mode in enumerate(modes):
            ax = axes[row, col]
            panel = subset[(subset["relu_layer_index"] == relu_index) & (subset["experiment_mode"] == mode)]
            for scenario in SCENARIOS:
                points = panel[panel["scenario"] == scenario]
                plot_epoch_trajectory(ax, points, metric, scenario)
            x = panel["relu_output_zero_rate"].to_numpy(float) * 100
            y = panel[metric].to_numpy(float)
            finite = np.isfinite(x) & np.isfinite(y)
            if finite.sum() >= 2 and np.std(x[finite]) > 0 and np.std(y[finite]) > 0:
                slope, intercept = np.polyfit(x[finite], y[finite], 1)
                line_x = np.linspace(x[finite].min(), x[finite].max(), 100)
                ax.plot(line_x, slope * line_x + intercept, color="black", linewidth=1.3, linestyle="--")
                r = float(np.corrcoef(x[finite], y[finite])[0, 1])
                rho = float(pd.Series(x[finite]).corr(pd.Series(y[finite]), method="spearman"))
                ax.text(0.03, 0.97, f"Pearson r = {r:.3f}\nSpearman rho = {rho:.3f}", transform=ax.transAxes, va="top")
                correlation_records.append({
                    "device_id": device,
                    "experiment_mode": mode,
                    "relu_layer_index": relu_index,
                    "pool_layer_index": pool_index,
                    "hardware_metric": metric,
                    "pearson_r": r,
                    "spearman_rho": rho,
                    "observations": int(finite.sum()),
                })
            ax.set_title(f"{mode}: ReLU {relu_index} -> MaxPool {pool_index}")
            ax.set_xlabel("ReLU output zero rate (%)")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
    fig.suptitle(f"ReLU sparsity vs MaxPool {ylabel}: {device}", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = output_dir / f"relu_zero_rate_vs_maxpool_{metric}_{device}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path, correlation_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data, hardware = load_joined(args.input_dir.resolve())
    data.to_csv(args.output_dir / "relu_markov_joined_epoch_data.csv", index=False)
    hardware.to_csv(args.output_dir / "relu_branch_miss_joined_epoch_data.csv", index=False)
    correlations = correlation_rows(data)
    correlations.to_csv(args.output_dir / "relu_markov_correlations.csv", index=False)
    hardware_correlations = []
    for device in sorted(data["device_id"].astype(str).unique()):
        print(plot_device(data, args.output_dir, device))
        for metric, ylabel in [
            ("branch_miss_percent", "branch-miss fraction (%)"),
            ("branch_misses_per_invocation", "raw branch misses / invocation"),
        ]:
            path, records = plot_hardware_device(hardware, args.output_dir, device, metric, ylabel)
            print(path)
            hardware_correlations.extend(records)
    pd.DataFrame(hardware_correlations).to_csv(
        args.output_dir / "relu_branch_miss_correlations.csv", index=False
    )
    print(args.output_dir / "relu_markov_correlations.csv")


if __name__ == "__main__":
    main()
