#!/usr/bin/env python3
"""Plot ReLU output zero rate for the five end-to-end scenarios."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/antivenom-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCENARIOS = (
    "baseline",
    "moderate_augmentation",
    "strong_augmentation",
    "availability_shortcuts",
    "badsampler",
)
LABELS = {
    "baseline": "Clean",
    "moderate_augmentation": "Moderate aug.",
    "strong_augmentation": "Strong aug.",
    "availability_shortcuts": "Availability shortcuts",
    "badsampler": "BadSampler",
}
COLORS = {
    "baseline": "#2563eb",
    "moderate_augmentation": "#16a34a",
    "strong_augmentation": "#dc2626",
    "availability_shortcuts": "#9333ea",
    "badsampler": "#ea580c",
}
MODES = ("train", "frozen_replay")
MODE_LABELS = {"train": "Training", "frozen_replay": "Frozen replay"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--input-dir", type=Path, default=script_dir.parent / "collected_logs"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "visualization" / "relu_sparsity",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    return float(np.average(values.to_numpy(float), weights=weights.to_numpy(float)))


def load_relu_rows(input_dir: Path) -> pd.DataFrame:
    paths = sorted(input_dir.rglob("*_entropy_summary.csv"))
    if not paths:
        raise FileNotFoundError(f"No *_entropy_summary.csv files under {input_dir}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        required = {
            "device_id",
            "trial_id",
            "scenario",
            "experiment_mode",
            "epoch",
            "layer_index",
            "layer_name",
            "layer_type",
            "metric",
            "mean",
            "std",
            "batches",
        }
        if not required.issubset(frame.columns):
            continue
        frame = frame[
            (frame["layer_type"] == "ReLU")
            & (frame["metric"] == "output_zero_rate")
            & frame["scenario"].isin(SCENARIOS)
            & frame["experiment_mode"].isin(MODES)
        ].copy()
        if frame.empty:
            continue
        frame["source_file"] = str(path)
        frames.append(frame)
    if not frames:
        raise ValueError(
            "No ReLU output_zero_rate rows were found. Markov-only runs do not "
            "contain ReLU sparsity; use a full-layer entropy run."
        )
    data = pd.concat(frames, ignore_index=True)
    for column in ("epoch", "layer_index", "mean", "std", "batches"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data[["epoch", "layer_index", "mean", "batches"]].isna().any().any():
        raise ValueError("Invalid numeric values in ReLU entropy summaries.")
    return data


def build_summary(data: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "device_id",
        "trial_id",
        "scenario",
        "experiment_mode",
        "layer_index",
        "layer_name",
    ]
    rows = []
    for values, group in data.groupby(keys, observed=True):
        rows.append(
            {
                **dict(zip(keys, values)),
                "relu_output_zero_rate": weighted_mean(group["mean"], group["batches"]),
                "epochs": group["epoch"].nunique(),
                "batch_observations": int(group["batches"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(keys)


def build_overall(summary: pd.DataFrame) -> pd.DataFrame:
    keys = ["device_id", "trial_id", "scenario", "experiment_mode"]
    rows = []
    for values, group in summary.groupby(keys, observed=True):
        rows.append(
            {
                **dict(zip(keys, values)),
                "relu_output_zero_rate": weighted_mean(
                    group["relu_output_zero_rate"], group["batch_observations"]
                ),
                "relu_layers": group["layer_index"].nunique(),
                "batch_layer_observations": int(group["batch_observations"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(keys)


def validate_five_conditions(summary: pd.DataFrame) -> None:
    for keys, group in summary.groupby(
        ["device_id", "trial_id", "experiment_mode"], observed=True
    ):
        scenarios = set(group["scenario"])
        missing = set(SCENARIOS) - scenarios
        if missing:
            raise ValueError(f"{keys} is missing scenarios: {sorted(missing)}")


def plot_device(
    summary: pd.DataFrame,
    overall: pd.DataFrame,
    *,
    device: str,
    output_dir: Path,
    dpi: int,
) -> Path:
    device_summary = summary[summary["device_id"] == device]
    layers = (
        device_summary[["layer_index", "layer_name"]]
        .drop_duplicates()
        .sort_values("layer_index")
    )
    row_labels = [
        f"Layer {int(row.layer_index)}: {row.layer_name}"
        for row in layers.itertuples(index=False)
    ] + ["All ReLU layers"]
    fig, axes = plt.subplots(
        len(row_labels),
        2,
        figsize=(14.5, max(7.0, 3.0 * len(row_labels))),
        squeeze=False,
        sharey=True,
    )
    x = np.arange(len(SCENARIOS))
    for column, mode in enumerate(MODES):
        mode_data = device_summary[device_summary["experiment_mode"] == mode]
        for row_index, layer in layers.reset_index(drop=True).iterrows():
            axis = axes[row_index, column]
            layer_data = mode_data[mode_data["layer_index"] == layer["layer_index"]]
            values = [
                float(layer_data[layer_data["scenario"] == scenario]["relu_output_zero_rate"].mean())
                for scenario in SCENARIOS
            ]
            axis.bar(x, values, color=[COLORS[scenario] for scenario in SCENARIOS])
            axis.set_title(f"{MODE_LABELS[mode]} | {row_labels[row_index]}")
            axis.set_ylim(0.0, 1.0)
            axis.grid(axis="y", alpha=0.22)
            axis.set_ylabel("ReLU output zero rate")
            for index, value in enumerate(values):
                axis.text(index, value + 0.015, f"{100*value:.1f}%", ha="center", fontsize=8)

        axis = axes[-1, column]
        mode_overall = overall[
            (overall["device_id"] == device)
            & (overall["experiment_mode"] == mode)
        ]
        values = [
            float(mode_overall[mode_overall["scenario"] == scenario]["relu_output_zero_rate"].mean())
            for scenario in SCENARIOS
        ]
        axis.bar(x, values, color=[COLORS[scenario] for scenario in SCENARIOS])
        axis.set_title(f"{MODE_LABELS[mode]} | All ReLU layers")
        axis.set_ylim(0.0, 1.0)
        axis.grid(axis="y", alpha=0.22)
        axis.set_ylabel("ReLU output zero rate")
        for index, value in enumerate(values):
            axis.text(index, value + 0.015, f"{100*value:.1f}%", ha="center", fontsize=8)

    for axis in axes[-1]:
        axis.set_xticks(x, [LABELS[scenario] for scenario in SCENARIOS], rotation=25, ha="right")
    for axis in axes[:-1].flat:
        axis.set_xticks(x, [])
    fig.suptitle(f"{device}: ReLU output sparsity across five conditions", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"relu_output_zero_rate_{device}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    data = load_relu_rows(args.input_dir.expanduser().resolve())
    summary = build_summary(data)
    validate_five_conditions(summary)
    overall = build_overall(summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "relu_sparsity_by_layer.csv", index=False)
    overall.to_csv(output_dir / "relu_sparsity_overall.csv", index=False)
    paths = [
        plot_device(
            summary,
            overall,
            device=str(device),
            output_dir=output_dir,
            dpi=args.dpi,
        )
        for device in sorted(summary["device_id"].unique())
    ]
    print(f"Loaded {len(data)} ReLU epoch/layer summaries")
    print(f"Saved {len(paths)} figures under {output_dir}")


if __name__ == "__main__":
    main()
