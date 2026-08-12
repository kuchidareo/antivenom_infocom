#!/usr/bin/env python3
"""Correlate real-training MaxPool Markov entropy with layer PMU branch misses."""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/antivenom-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCENARIO_LABELS = {
    "baseline": "Clean",
    "moderate_augmentation": "Moderate augmentation",
    "strong_augmentation": "Strong augmentation",
    "availability_shortcuts": "Availability shortcuts",
    "badsampler": "BadSampler",
}
SCENARIO_COLORS = {
    "baseline": "#2563eb",
    "moderate_augmentation": "#16a34a",
    "strong_augmentation": "#dc2626",
    "availability_shortcuts": "#9333ea",
    "badsampler": "#ea580c",
}
MODE_LABELS = {"train": "Training", "frozen_replay": "Frozen replay"}

JOIN_KEYS = [
    "device_id",
    "trial_id",
    "scenario",
    "experiment_mode",
    "round",
    "epoch",
    "batch_idx",
    "layer_index",
    "layer_name",
    "invocation_index",
]
ENTROPY_COLUMN = "position_markov_entropy_rate_bits"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--input-dir", type=Path, default=script_dir.parent / "collected_logs"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=script_dir / "visualization" / "maxpool_markov"
    )
    parser.add_argument(
        "--minimum-running-percent",
        type=float,
        default=95.0,
        help="Exclude PMU rows below this running percentage.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def safe_name(value: object) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value)).strip("_") or "unknown"


def find_perf_path(markov_path: Path) -> Path:
    expected = markov_path.with_name(
        markov_path.name.replace("_maxpool_markov.csv", "_layer_perf.csv")
    )
    if expected.is_file():
        return expected
    candidates = sorted(markov_path.parent.glob("*_layer_perf.csv"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        f"Cannot identify one layer PMU CSV for {markov_path}; candidates={candidates}"
    )


def canonicalize_branch_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    aliases = {
        "perf_branch_loads": "perf_branches",
        "perf_branch_instructions": "perf_branches",
        "perf_branch_load_misses": "perf_branch_misses",
        "perf_branch_loads_running_pct": "perf_branches_running_pct",
        "perf_branch_instructions_running_pct": "perf_branches_running_pct",
        "perf_branch_load_misses_running_pct": "perf_branch_misses_running_pct",
    }
    for source, target in aliases.items():
        if target not in frame.columns and source in frame.columns:
            frame[target] = frame[source]
    return frame


def numeric(frame: pd.DataFrame, columns: Iterable[str], source: Path) -> None:
    for column in columns:
        if column not in frame.columns:
            raise ValueError(f"{source} is missing required column: {column}")
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any():
            raise ValueError(f"{source} contains invalid numeric values in {column}")


def load_joined(input_dir: Path, minimum_running_percent: float) -> pd.DataFrame:
    markov_paths = sorted(input_dir.rglob("*_maxpool_markov.csv"))
    if not markov_paths:
        raise FileNotFoundError(f"No *_maxpool_markov.csv files found under {input_dir}")

    frames = []
    for markov_path in markov_paths:
        perf_path = find_perf_path(markov_path)
        entropy = pd.read_csv(markov_path, low_memory=False)
        perf = canonicalize_branch_columns(pd.read_csv(perf_path, low_memory=False))
        if entropy.empty:
            raise ValueError(f"MaxPool Markov CSV is empty: {markov_path}")

        missing_entropy = sorted(
            set(JOIN_KEYS + [ENTROPY_COLUMN, "windows", "logical_comparisons"])
            - set(entropy.columns)
        )
        missing_perf = sorted(
            set(JOIN_KEYS + ["phase", "layer_type", "perf_branches", "perf_branch_misses"])
            - set(perf.columns)
        )
        if missing_entropy:
            raise ValueError(f"{markov_path} is missing columns: {missing_entropy}")
        if missing_perf:
            raise ValueError(f"{perf_path} is missing branch PMU columns: {missing_perf}")

        entropy = entropy[(entropy["phase"] == "forward") & (entropy["layer_type"] == "MaxPool2d")].copy()
        perf = perf[(perf["phase"] == "forward") & (perf["layer_type"] == "MaxPool2d")].copy()
        if entropy.empty or perf.empty:
            raise ValueError(f"Missing forward MaxPool rows in {markov_path} or {perf_path}")

        integer_columns = [
            "round",
            "epoch",
            "batch_idx",
            "layer_index",
            "invocation_index",
        ]
        numeric(entropy, [*integer_columns, ENTROPY_COLUMN, "windows", "logical_comparisons"], markov_path)
        numeric(perf, [*integer_columns, "perf_branches", "perf_branch_misses"], perf_path)
        for column in integer_columns:
            entropy[column] = entropy[column].astype(int)
            perf[column] = perf[column].astype(int)

        if entropy.duplicated(JOIN_KEYS).any():
            raise ValueError(f"Duplicate MaxPool entropy join keys in {markov_path}")
        if perf.duplicated(JOIN_KEYS).any():
            raise ValueError(f"Duplicate MaxPool PMU join keys in {perf_path}")

        keep_perf = [
            *JOIN_KEYS,
            "duration_ns",
            "perf_events",
            "perf_branches",
            "perf_branch_misses",
        ]
        running_columns = [
            column
            for column in (
                "perf_branches_running_pct",
                "perf_branch_misses_running_pct",
            )
            if column in perf.columns
        ]
        keep_perf.extend(running_columns)
        joined = entropy.merge(
            perf[keep_perf],
            on=JOIN_KEYS,
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
        unmatched = joined["_merge"] != "both"
        if unmatched.any():
            examples = joined.loc[unmatched, [*JOIN_KEYS, "_merge"]].head(5)
            raise ValueError(
                f"Entropy/PMU join mismatch for {markov_path}:\n{examples.to_string(index=False)}"
            )
        joined = joined.drop(columns="_merge")
        joined["entropy_source_file"] = str(markov_path)
        joined["perf_source_file"] = str(perf_path)

        if running_columns:
            for column in running_columns:
                joined[column] = pd.to_numeric(joined[column], errors="coerce")
            joined["minimum_branch_running_percent"] = joined[running_columns].min(axis=1)
        else:
            joined["minimum_branch_running_percent"] = 100.0
        joined["pmu_valid"] = (
            (joined["minimum_branch_running_percent"] >= minimum_running_percent)
            & (joined["perf_branches"] > 0)
            & (joined["logical_comparisons"] > 0)
        )
        joined["branch_miss_fraction"] = (
            joined["perf_branch_misses"] / joined["perf_branches"]
        )
        joined["branch_miss_percent"] = 100.0 * joined["branch_miss_fraction"]
        joined["branch_misses_per_logical_comparison"] = (
            joined["perf_branch_misses"] / joined["logical_comparisons"]
        )
        joined["retired_branches_per_logical_comparison"] = (
            joined["perf_branches"] / joined["logical_comparisons"]
        )
        frames.append(joined)

    return pd.concat(frames, ignore_index=True)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(
        pd.Series(x).rank(method="average").to_numpy(),
        pd.Series(y).rank(method="average").to_numpy(),
    )


def centered_correlation(data: pd.DataFrame, response: str) -> float:
    centered = data.copy()
    group = ["scenario", "epoch"]
    centered["_x"] = centered[ENTROPY_COLUMN] - centered.groupby(group)[
        ENTROPY_COLUMN
    ].transform("mean")
    centered["_y"] = centered[response] - centered.groupby(group)[response].transform(
        "mean"
    )
    return pearson(centered["_x"].to_numpy(), centered["_y"].to_numpy())


def fit_line(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < 3 or np.std(x) == 0:
        return
    slope, intercept = np.polyfit(x, y, 1)
    grid = np.linspace(float(x.min()), float(x.max()), 200)
    ax.plot(grid, slope * grid + intercept, color="#111827", linewidth=1.8)


def plot_group(
    data: pd.DataFrame,
    *,
    device: str,
    experiment_mode: str,
    layer_index: int,
    layer_name: str,
    output_dir: Path,
    dpi: int,
) -> Path:
    responses = (
        ("branch_miss_percent", "Branch misses / retired branches (%)"),
        (
            "branch_misses_per_logical_comparison",
            "Observed branch misses / logical MaxPool comparisons",
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.8), squeeze=False)
    x = data[ENTROPY_COLUMN].to_numpy(dtype=float)
    for axis, (response, ylabel) in zip(axes[0], responses):
        y = data[response].to_numpy(dtype=float)
        for scenario, scenario_data in data.groupby("scenario", observed=True):
            axis.scatter(
                scenario_data[ENTROPY_COLUMN],
                scenario_data[response],
                s=18,
                alpha=0.42,
                color=SCENARIO_COLORS.get(str(scenario), "#64748b"),
                label=SCENARIO_LABELS.get(str(scenario), str(scenario)),
            )
        fit_line(axis, x, y)
        raw_pearson = pearson(x, y)
        raw_spearman = spearman(x, y)
        within = centered_correlation(data, response)
        axis.text(
            0.025,
            0.975,
            f"Batch Pearson r = {raw_pearson:.3f}\n"
            f"Batch Spearman rho = {raw_spearman:.3f}\n"
            f"Within scenario/epoch r = {within:.3f}\n"
            f"Valid batches = {len(data)}",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9},
        )
        axis.set_xlabel("Position-aware MaxPool Markov entropy (bits/comparison)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.22)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(5, len(labels)),
            frameon=False,
            bbox_to_anchor=(0.5, 0.97),
        )
    fig.suptitle(
        f"{device} | {MODE_LABELS.get(experiment_mode, experiment_mode)} | "
        f"layer {layer_index}: {layer_name}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / (
        f"entropy_branch_miss_{safe_name(device)}_{safe_name(experiment_mode)}_"
        f"layer_{layer_index}_{safe_name(layer_name)}.png"
    )
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def correlation_summary(valid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = [
        "device_id",
        "experiment_mode",
        "layer_index",
        "layer_name",
    ]
    responses = ("branch_miss_percent", "branch_misses_per_logical_comparison")
    for keys, group in valid.groupby(group_columns, observed=True):
        x = group[ENTROPY_COLUMN].to_numpy(dtype=float)
        for response in responses:
            y = group[response].to_numpy(dtype=float)
            rows.append(
                {
                    **dict(zip(group_columns, keys)),
                    "response": response,
                    "pearson_r": pearson(x, y),
                    "spearman_rho": spearman(x, y),
                    "within_scenario_epoch_pearson_r": centered_correlation(
                        group, response
                    ),
                    "batches": len(group),
                    "scenarios": group["scenario"].nunique(),
                    "epochs": group["epoch"].nunique(),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    joined = load_joined(input_dir, args.minimum_running_percent)
    output_dir.mkdir(parents=True, exist_ok=True)
    joined.to_csv(output_dir / "maxpool_markov_pmu_joined.csv", index=False)

    valid = joined[joined["pmu_valid"]].copy()
    if valid.empty:
        raise ValueError(
            "No valid MaxPool PMU rows remain after branch counter quality filtering."
        )
    summary = correlation_summary(valid)
    summary.to_csv(output_dir / "maxpool_markov_correlation_summary.csv", index=False)

    generated = []
    group_columns = ["device_id", "experiment_mode", "layer_index", "layer_name"]
    for keys, group in valid.groupby(group_columns, observed=True):
        device, mode, layer_index, layer_name = keys
        generated.append(
            plot_group(
                group,
                device=str(device),
                experiment_mode=str(mode),
                layer_index=int(layer_index),
                layer_name=str(layer_name),
                output_dir=output_dir / safe_name(device),
                dpi=args.dpi,
            )
        )

    invalid = int((~joined["pmu_valid"]).sum())
    print(f"Loaded and joined {len(joined)} MaxPool batch/layer rows")
    print(f"Valid PMU rows: {len(valid)}; excluded rows: {invalid}")
    print(f"Saved {len(generated)} correlation figures under {output_dir}")
    print(f"Correlation summary: {output_dir / 'maxpool_markov_correlation_summary.csv'}")


if __name__ == "__main__":
    main()
