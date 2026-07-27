#!/usr/bin/env python3
"""Plot per-batch perf traces for selected FL rounds."""

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
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "collected_logs_0718"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "visualization_detail_0718"
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
PHASES = ("forward", "backward")
DEFAULT_ROUNDS = (0, 7, 14)

GROUPS = (
    ("baseline", "Baseline clean", "#202124"),
    ("attack_clean", "Attack-run clean", "#007f83"),
    ("attack_poisoned", "Attack-run poisoned", "#c23b4a"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--poisoned-count", type=int, default=4)
    parser.add_argument(
        "--rounds",
        type=int,
        nargs="+",
        default=DEFAULT_ROUNDS,
        help="Zero-based rounds to display (default: 0 7 14).",
    )
    parser.add_argument(
        "--time-step-ms",
        type=float,
        default=100.0,
        help="Alignment grid interval in milliseconds (default: 100).",
    )
    parser.add_argument(
        "--plot-mode",
        choices=("all", "raw", "per-instruction"),
        default="all",
        help="Generate raw, per-instruction, or both figure sets (default: all).",
    )
    parser.add_argument("--dpi", type=int, default=140)
    return parser.parse_args()


def load_base_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fl_raw_visualization", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def raw_metrics(base: ModuleType) -> list[tuple[str, str]]:
    metrics = [("perf_instructions", "Retired instructions")]
    metrics.extend(
        (metric.column, metric.label.removesuffix(" / instruction"))
        for metric in base.METRICS
    )
    return metrics


def select_groups(runs_by_device: dict, attack: str, poisoned_count: int) -> dict:
    groups = {name: [] for name, _, _ in GROUPS}
    for device, runs in runs_by_device.items():
        baseline = [
            run
            for run in runs
            if run.global_method == "clean" and run.poisoned_count == 0
        ]
        attacked = [
            run
            for run in runs
            if run.global_method == attack
            and run.poisoned_count == poisoned_count
        ]
        if len(baseline) != 1 or len(attacked) != 1:
            raise ValueError(
                f"{device}: expected one baseline and one "
                f"{attack}/{poisoned_count} run"
            )
        groups["baseline"].append(baseline[0])
        target = "attack_poisoned" if attacked[0].locally_poisoned else "attack_clean"
        groups[target].append(attacked[0])

    if len(groups["attack_poisoned"]) != poisoned_count:
        raise ValueError(
            f"{attack}/{poisoned_count}: found "
            f"{len(groups['attack_poisoned'])} locally poisoned clients"
        )
    return groups


def load_run_traces(
    run,
    metrics: list[tuple[str, str]],
    display_rounds: tuple[int, ...],
) -> dict[tuple[str, int], pd.DataFrame]:
    csv_rounds = {round_id + 1 for round_id in display_rounds}
    columns = [
        "timestamp_unix",
        "round",
        "batch_idx",
        "phase",
        *(name for name, _ in metrics),
    ]
    frame = pd.read_csv(run.perf_path, usecols=columns, low_memory=False)
    frame["round"] = pd.to_numeric(frame["round"], errors="coerce")
    frame = frame.loc[
        frame["round"].isin(csv_rounds) & frame["phase"].isin(PHASES)
    ].copy()
    frame["timestamp_unix"] = pd.to_numeric(frame["timestamp_unix"], errors="coerce")
    frame["batch_idx"] = pd.to_numeric(frame["batch_idx"], errors="coerce")
    for name, _ in metrics:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")

    traces: dict[tuple[str, int], pd.DataFrame] = {}
    for display_round in display_rounds:
        csv_round = display_round + 1
        for phase in PHASES:
            raw_trace = frame.loc[
                frame["round"].eq(csv_round) & frame["phase"].eq(phase)
            ].dropna(subset=["timestamp_unix", "batch_idx"])
            raw_trace = raw_trace.sort_values("timestamp_unix").reset_index(drop=True)
            if raw_trace.empty:
                raise ValueError(
                    f"No {phase} samples for displayed round {display_round} "
                    f"(CSV round {csv_round}) in {run.perf_path}"
                )
            grouped = raw_trace.groupby("batch_idx", sort=True)
            raw_trace["batch_elapsed_sec"] = (
                raw_trace["timestamp_unix"]
                - grouped["timestamp_unix"].transform("min")
            )
            raw_trace["batch_idx"] = raw_trace["batch_idx"].astype(int)
            traces[(phase, display_round)] = raw_trace
    return traces


def maximum_batch_duration(
    groups: dict,
    traces: dict[Path, dict[tuple[str, int], pd.DataFrame]],
    phase: str,
    round_id: int,
) -> float:
    return max(
        traces[run.perf_path][(phase, round_id)]["batch_elapsed_sec"].max()
        for runs in groups.values()
        for run in runs
    )


def interpolate_trace(
    trace: pd.DataFrame,
    metric: str,
    grid: np.ndarray,
    per_instruction: bool,
) -> np.ndarray | None:
    columns = ["batch_elapsed_sec", metric]
    if per_instruction:
        columns.append("perf_instructions")
    valid = trace[columns].dropna()
    if per_instruction:
        valid = valid.loc[valid["perf_instructions"].gt(0)].copy()
        valid[metric] = valid[metric] / valid["perf_instructions"]
    if len(valid) < 2:
        return None
    valid = valid.groupby("batch_elapsed_sec", as_index=False)[metric].mean()
    x = valid["batch_elapsed_sec"].to_numpy()
    y = valid[metric].to_numpy()
    result = np.full(grid.shape, np.nan, dtype=float)
    inside = (grid >= x[0]) & (grid <= x[-1])
    result[inside] = np.interp(grid[inside], x, y)
    return result


def aligned_group_statistics(
    runs: list,
    traces: dict[Path, dict[tuple[str, int], pd.DataFrame]],
    phase: str,
    round_id: int,
    metric: str,
    grid: np.ndarray,
    per_instruction: bool,
) -> pd.DataFrame | None:
    # Average clients first, leaving exactly one aligned trace per batch_idx.
    clients_by_batch: dict[int, list[np.ndarray]] = {}
    for run in runs:
        trace = traces[run.perf_path][(phase, round_id)]
        for batch_idx, batch_trace in trace.groupby("batch_idx", sort=True):
            values = interpolate_trace(
                batch_trace, metric, grid, per_instruction
            )
            if values is not None:
                clients_by_batch.setdefault(int(batch_idx), []).append(values)
    if not clients_by_batch:
        return None

    batch_traces = []
    for client_traces in clients_by_batch.values():
        batch_traces.append(pd.DataFrame(client_traces).mean(axis=0).to_numpy())
    batches = pd.DataFrame(batch_traces)
    result = pd.DataFrame(
        {
            "elapsed_time_sec": grid,
            "mean": batches.mean(axis=0).to_numpy(),
            "minimum": batches.min(axis=0).to_numpy(),
            "maximum": batches.max(axis=0).to_numpy(),
            "contributing_batch_count": batches.count(axis=0).to_numpy(),
        }
    )
    result["range"] = result["maximum"] - result["minimum"]
    return result


def calculate_statistics(
    attack: str,
    poisoned_count: int,
    groups: dict,
    traces: dict[Path, dict[tuple[str, int], pd.DataFrame]],
    metrics: list[tuple[str, str]],
    rounds: tuple[int, ...],
    time_step_sec: float,
    per_instruction: bool,
) -> tuple[dict, pd.DataFrame]:
    statistics = {}
    records = []
    for phase in PHASES:
        for round_id in rounds:
            duration = maximum_batch_duration(groups, traces, phase, round_id)
            grid = np.arange(0.0, duration + time_step_sec / 2.0, time_step_sec)
            for metric, label in metrics:
                for group_name, _, _ in GROUPS:
                    result = aligned_group_statistics(
                        groups[group_name],
                        traces,
                        phase,
                        round_id,
                        metric,
                        grid,
                        per_instruction,
                    )
                    if result is None:
                        continue
                    statistics[(group_name, phase, round_id, metric)] = result
                    for row in result.itertuples(index=False):
                        records.append(
                            {
                                "attack": attack,
                                "poisoned_client_count": poisoned_count,
                                "group": group_name,
                                "phase": phase,
                                "round": round_id,
                                "csv_round": round_id + 1,
                                "metric": metric,
                                "metric_label": (
                                    f"{label} / instruction"
                                    if per_instruction
                                    else label
                                ),
                                "normalization": (
                                    "per_instruction" if per_instruction else "raw"
                                ),
                                "elapsed_time_sec": row.elapsed_time_sec,
                                "mean": row.mean,
                                "minimum": row.minimum,
                                "maximum": row.maximum,
                                "range": row.range,
                                "contributing_batch_count": row.contributing_batch_count,
                            }
                        )
    return statistics, pd.DataFrame(records)


def save_phase_figure(
    attack: str,
    poisoned_count: int,
    phase: str,
    groups: dict,
    statistics: dict,
    metrics: list[tuple[str, str]],
    rounds: tuple[int, ...],
    output_dir: Path,
    dpi: int,
    per_instruction: bool,
) -> Path:
    available = [
        (metric, label)
        for metric, label in metrics
        if any(
            (group_name, phase, round_id, metric) in statistics
            for group_name, _, _ in GROUPS
            for round_id in rounds
        )
    ]
    figure, axes = plt.subplots(
        len(available),
        len(rounds),
        figsize=(7.0 * len(rounds), max(2.65 * len(available), 8)),
        squeeze=False,
        sharey="row",
    )

    for row, (metric, label) in enumerate(available):
        for column, round_id in enumerate(rounds):
            axis = axes[row, column]
            for group_name, _, color in GROUPS:
                result = statistics.get((group_name, phase, round_id, metric))
                if result is None:
                    continue
                x = result["elapsed_time_sec"].to_numpy(dtype=float)
                mean = result["mean"].to_numpy(dtype=float)
                minimum = result["minimum"].to_numpy(dtype=float)
                maximum = result["maximum"].to_numpy(dtype=float)
                axis.fill_between(
                    x,
                    minimum,
                    maximum,
                    color=color,
                    alpha=0.13,
                    linewidth=0,
                )
                axis.plot(
                    x,
                    mean,
                    color=color,
                    linewidth=1.7,
                    marker="o",
                    markersize=2.2,
                )

            valid_times = [
                result.loc[
                    result["contributing_batch_count"].gt(0), "elapsed_time_sec"
                ].max()
                for group_name, _, _ in GROUPS
                if (
                    result := statistics.get(
                        (group_name, phase, round_id, metric)
                    )
                )
                is not None
            ]
            axis.set_xlim(0.0, max(valid_times))
            axis.grid(True, color="#d8dadd", linewidth=0.5, alpha=0.75)
            axis.tick_params(axis="both", labelsize=7)
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
            axis.margins(y=0.08)
            if column == 0:
                axis.set_ylabel(
                    f"{label} / instruction" if per_instruction else label,
                    fontsize=8,
                )
            if row == 0:
                axis.set_title(f"Round {round_id}", fontsize=11)
            if row == len(available) - 1:
                axis.set_xlabel("Elapsed time within batch (seconds)", fontsize=9)

    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            linewidth=1.8,
            label=f"{label} (n={len(groups[name])} clients; mean and batch range)",
        )
        for name, label, color in GROUPS
        if groups[name]
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.987),
        ncol=len(handles),
        frameon=False,
        fontsize=9,
    )
    value_type = "perf metrics per instruction" if per_instruction else "raw perf metrics"
    figure.suptitle(
        f"{ATTACK_LABELS[attack]}: {phase.capitalize()} within-batch {value_type} "
        f"({poisoned_count} poisoned clients; band=min-max across batches)",
        fontsize=15,
        y=0.999,
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.992,
        bottom=0.025,
        top=0.955,
        hspace=0.42,
        wspace=0.12,
    )

    attack_dir = output_dir / attack
    attack_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_per_instruction" if per_instruction else ""
    path = attack_dir / f"{phase}{suffix}.png"
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def main() -> None:
    args = parse_args()
    rounds = tuple(args.rounds)
    if not rounds or any(round_id < 0 or round_id > 14 for round_id in rounds):
        raise ValueError("--rounds must contain zero-based values from 0 through 14")
    if args.time_step_ms <= 0:
        raise ValueError("--time-step-ms must be positive")
    time_step_sec = args.time_step_ms / 1000.0

    base = load_base_module()
    metrics = raw_metrics(base)
    runs_by_device = base.discover_runs(args.input_dir.resolve())
    output_dir = args.output_dir.resolve()

    baseline_cache: dict[Path, dict[tuple[str, int], pd.DataFrame]] = {}
    for attack in ATTACKS:
        print(f"Processing {attack} ...", flush=True)
        groups = select_groups(runs_by_device, attack, args.poisoned_count)
        traces = dict(baseline_cache)
        for runs in groups.values():
            for run in runs:
                if run.perf_path not in traces:
                    traces[run.perf_path] = load_run_traces(run, metrics, rounds)
        if not baseline_cache:
            baseline_cache = {
                run.perf_path: traces[run.perf_path]
                for run in groups["baseline"]
            }

        attack_dir = output_dir / attack
        attack_dir.mkdir(parents=True, exist_ok=True)
        modes = []
        if args.plot_mode in ("all", "raw"):
            modes.append((False, metrics, "raw_timeseries_ranges.csv"))
        if args.plot_mode in ("all", "per-instruction"):
            modes.append(
                (True, metrics[1:], "per_instruction_timeseries_ranges.csv")
            )

        for per_instruction, mode_metrics, range_name in modes:
            statistics, ranges = calculate_statistics(
                attack,
                args.poisoned_count,
                groups,
                traces,
                mode_metrics,
                rounds,
                time_step_sec,
                per_instruction,
            )
            range_path = attack_dir / range_name
            ranges.to_csv(range_path, index=False)
            print(f"Saved {range_path}")

            for phase in PHASES:
                path = save_phase_figure(
                    attack,
                    args.poisoned_count,
                    phase,
                    groups,
                    statistics,
                    mode_metrics,
                    rounds,
                    output_dir,
                    args.dpi,
                    per_instruction,
                )
                print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
