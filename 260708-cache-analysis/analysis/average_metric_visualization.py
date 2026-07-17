#!/usr/bin/env python3
"""Plot epoch-level forward and backward between-batch variances."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from analysis import color_for_method


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "result" / "raw_data_visualization" / "batch_metric_variances.csv"
DEFAULT_OUTPUT = HERE / "result" / "raw_data_visualization" / "epoch_metric_variances.png"
ID_COLUMNS = {"run_id", "poisoning_method", "epoch", "phase"}
METHOD_ORDER = ["clean", "unlearnable_examples", "availability_shortcuts"]
METHOD_LABELS = {
    "clean": "Clean",
    "unlearnable_examples": "Unlearnable examples",
    "availability_shortcuts": "Availability shortcuts",
}


def plot_batch_variances(data: pd.DataFrame, output: Path, dpi: int = 110) -> tuple[int, int]:
    metrics = [column for column in data.columns if column not in ID_COLUMNS]
    phases = [phase for phase in ["forward", "backward"] if phase in set(data["phase"])]
    methods = [method for method in METHOD_ORDER if method in set(data["poisoning_method"])]
    if not metrics or not phases or not methods:
        raise ValueError("input data has no plottable metrics, phases, or poisoning methods")

    for column in ["epoch", *metrics]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    fig, axes = plt.subplots(
        len(metrics),
        len(phases),
        figsize=(16, max(12, len(metrics) * 1.65)),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    epochs = sorted(data["epoch"].dropna().astype(int).unique())
    for row, metric in enumerate(metrics):
        for column, phase in enumerate(phases):
            ax = axes[row, column]
            subset = data[data["phase"] == phase]
            for method in methods:
                values = subset[subset["poisoning_method"] == method].sort_values("epoch")
                ax.plot(
                    values["epoch"],
                    values[metric],
                    color=color_for_method(method),
                    linewidth=1.1,
                    marker="o",
                    markersize=3,
                )
            ax.set_xticks(epochs)
            ax.tick_params(axis="both", labelsize=6, labelbottom=True)
            ax.grid(True, color="#d8d8d8", linewidth=0.5, alpha=0.8)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useOffset=False)
            if column == 0:
                ax.set_ylabel(metric, fontsize=7)
            if row == 0:
                ax.set_title(phase.capitalize(), fontsize=10)
            if row == len(metrics) - 1:
                ax.set_xlabel("Epoch", fontsize=8)

    handles = [
        Line2D(
            [0],
            [0],
            color=color_for_method(method),
            marker="o",
            markersize=4,
            linewidth=1.3,
            label=METHOD_LABELS.get(method, method),
        )
        for method in methods
    ]
    fig.suptitle("Forward and backward variance across batches", fontsize=14, y=0.998)
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 0.993))
    fig.subplots_adjust(left=0.16, right=0.99, top=0.975, bottom=0.02, hspace=0.65, wspace=0.12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)
    return len(metrics), len(epochs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=110)
    args = parser.parse_args()

    try:
        metric_count, epoch_count = plot_batch_variances(
            pd.read_csv(args.input_csv),
            args.output,
            args.dpi,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Wrote {args.output} ({metric_count} metrics, {epoch_count} epochs)", flush=True)


if __name__ == "__main__":
    main()
