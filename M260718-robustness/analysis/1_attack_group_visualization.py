#!/usr/bin/env python3
"""Compare clean and poisoned client groups for each FL attack scenario."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "collected_logs"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "visualization"
BASE_SCRIPT = SCRIPT_DIR / "0_raw_data_visualization.py"

ATTACKS = (
    "unlearnable_examples",
    "availability_shortcuts",
    "random_label_flipping",
)
ATTACK_LABELS = {
    "unlearnable_examples": "Unlearnable examples",
    "availability_shortcuts": "Availability shortcuts",
    "random_label_flipping": "Random label flipping",
}
POISONED_COUNTS = (1, 4, 7, 10)
PHASES = ("forward", "backward")

GROUP_COLORS = {
    "baseline": "#202124",
    "attack_clean": "#007f83",
    "attack_poisoned": "#c23b4a",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=140)
    return parser.parse_args()


def load_base_module() -> ModuleType:
    """Load the numbered base script without duplicating its CSV logic."""
    module_name = "fl_raw_data_visualization"
    spec = importlib.util.spec_from_file_location(module_name, BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def select_runs(runs_by_device: dict, attack: str, poisoned_count: int):
    baseline = []
    attacked = []
    for runs in runs_by_device.values():
        baseline_matches = [
            run
            for run in runs
            if run.global_method == "clean" and run.poisoned_count == 0
        ]
        attack_matches = [
            run
            for run in runs
            if run.global_method == attack and run.poisoned_count == poisoned_count
        ]
        if len(baseline_matches) != 1 or len(attack_matches) != 1:
            raise ValueError(
                f"Expected one baseline and one {attack}/{poisoned_count} run "
                f"for each device"
            )
        baseline.extend(baseline_matches)
        attacked.extend(attack_matches)

    attack_clean = [run for run in attacked if not run.locally_poisoned]
    attack_poisoned = [run for run in attacked if run.locally_poisoned]
    if len(attack_poisoned) != poisoned_count:
        raise ValueError(
            f"{attack}/{poisoned_count}: found {len(attack_poisoned)} poisoned clients"
        )
    return {
        "baseline": baseline,
        "attack_clean": attack_clean,
        "attack_poisoned": attack_poisoned,
    }


def group_statistics(runs, summaries, phase: str, metric_column: str):
    if not runs:
        return None
    values = pd.concat(
        {
            run.device: summaries[run.perf_path][phase][metric_column]
            for run in runs
        },
        axis=1,
    )
    return values.mean(axis=1), values.std(axis=1, ddof=1)


def plot_group(
    axis: plt.Axes,
    rounds: pd.Index,
    mean: pd.Series,
    std: pd.Series,
    color: str,
) -> None:
    axis.plot(
        rounds,
        mean,
        color=color,
        linewidth=1.7,
        marker="o",
        markersize=3.0,
    )
    if std.notna().any():
        lower = mean - std.fillna(0.0)
        upper = mean + std.fillna(0.0)
        axis.fill_between(rounds, lower, upper, color=color, alpha=0.15, linewidth=0)


def save_scenario_figure(
    base: ModuleType,
    attack: str,
    poisoned_count: int,
    groups: dict,
    summaries: dict,
    output_dir: Path,
    dpi: int,
) -> Path:
    metrics = [
        metric
        for metric in base.METRICS
        if any(
            summaries[run.perf_path][phase][metric.column].notna().any()
            for runs in groups.values()
            for run in runs
            for phase in PHASES
        )
    ]
    figure, axes = plt.subplots(
        nrows=len(metrics),
        ncols=2,
        figsize=(18, max(3.0 * len(metrics), 8)),
        squeeze=False,
    )

    for row, metric in enumerate(metrics):
        for column, phase in enumerate(PHASES):
            axis = axes[row, column]
            for group_name, runs in groups.items():
                statistics = group_statistics(
                    runs, summaries, phase, metric.column
                )
                if statistics is None:
                    continue
                mean, std = statistics
                plot_group(
                    axis,
                    mean.index,
                    mean,
                    std,
                    GROUP_COLORS[group_name],
                )

            axis.set_ylabel(metric.label, fontsize=8)
            axis.set_xlim(1, 15)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=15))
            axis.tick_params(axis="both", labelsize=7)
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
            axis.grid(True, color="#d8dadd", linewidth=0.55, alpha=0.8)
            axis.margins(y=0.08)
            if row == 0:
                axis.set_title(phase.capitalize(), fontsize=11, pad=7)
            if row == len(metrics) - 1:
                axis.set_xlabel("Federated learning round", fontsize=9)

    legend_specs = (
        ("baseline", "Baseline clean", len(groups["baseline"])),
        ("attack_clean", "Attack-run clean", len(groups["attack_clean"])),
        ("attack_poisoned", "Attack-run poisoned", len(groups["attack_poisoned"])),
    )
    handles = [
        Line2D(
            [0],
            [0],
            color=GROUP_COLORS[group_name],
            linewidth=1.7,
            marker="o",
            markersize=4,
            label=f"{label} (n={size}, mean +/- 1 SD)",
        )
        for group_name, label, size in legend_specs
        if size > 0
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.986),
        ncol=len(handles),
        fontsize=9,
        frameon=False,
        handlelength=3.0,
    )
    figure.suptitle(
        f"{ATTACK_LABELS[attack]}: {poisoned_count} poisoned clients",
        fontsize=15,
        y=0.999,
    )
    figure.tight_layout(rect=(0.025, 0.01, 0.995, 0.958), h_pad=1.1, w_pad=1.2)

    attack_dir = output_dir / attack
    attack_dir.mkdir(parents=True, exist_ok=True)
    path = attack_dir / f"poisoned_{poisoned_count}_clients.png"
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def main() -> None:
    args = parse_args()
    base = load_base_module()
    runs_by_device = base.discover_runs(args.input_dir.resolve())

    all_runs = [run for runs in runs_by_device.values() for run in runs]
    print(f"Aggregating {len(all_runs)} runs across {len(runs_by_device)} clients ...")
    summaries = {}
    for index, run in enumerate(all_runs, start=1):
        summaries[run.perf_path] = base.aggregate_run(run)
        if index % 13 == 0:
            print(f"Aggregated {index}/{len(all_runs)} runs", flush=True)

    for attack in ATTACKS:
        for poisoned_count in POISONED_COUNTS:
            groups = select_runs(runs_by_device, attack, poisoned_count)
            path = save_scenario_figure(
                base,
                attack,
                poisoned_count,
                groups,
                summaries,
                args.output_dir.resolve(),
                args.dpi,
            )
            print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
