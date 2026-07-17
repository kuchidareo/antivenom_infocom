#!/usr/bin/env python3
"""Compare CloudLab clean and availability runs using perf values per instruction."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "collected_logs" / "logs" / "cloudlab_device1"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "visualization" / "cloudlab-test"
METHODS = ("clean", "availability_shortcuts")
PHASES = ("forward", "backward")
METHOD_LABELS = {
    "clean": "Clean",
    "availability_shortcuts": "Availability shortcuts",
}
METHOD_COLORS = {
    "clean": "#202124",
    "availability_shortcuts": "#007f83",
}


@dataclass(frozen=True)
class Metric:
    column: str
    label: str


# Instructions are the denominator and are therefore not plotted as a constant 1.
PERF_METRICS = (
    Metric("perf_cycles", "CPU cycles / instruction"),
    Metric("perf_task_clock", "Task clock (ms) / instruction"),
    Metric("perf_context_switches", "Context switches / instruction"),
    Metric("perf_cpu_migrations", "CPU migrations / instruction"),
    Metric("perf_page_faults", "Page faults / instruction"),
    Metric("perf_branches", "Branch operations / instruction"),
    Metric("perf_branch_misses", "Branch misses / instruction"),
    Metric("perf_cache_references", "Cache references / instruction"),
    Metric("perf_cache_misses", "Cache misses / instruction"),
    Metric("perf_l1d_cache_rd", "L1D read accesses / instruction"),
    Metric("perf_l1d_cache_refill_rd", "L1D read refills / instruction"),
    Metric("perf_l1d_cache_wr", "L1D write accesses / instruction"),
    Metric("perf_l1d_cache_refill_wr", "L1D write refills / instruction"),
    Metric("perf_l2d_cache_rd", "L2D read accesses / instruction"),
    Metric("perf_l2d_cache_refill_rd", "L2D read refills / instruction"),
    Metric("perf_l2d_cache_wr", "L2D write accesses / instruction"),
    Metric("perf_l2d_cache_refill_wr", "L2D write refills / instruction"),
    Metric("perf_bus_access_rd", "Bus read accesses / instruction"),
    Metric("perf_bus_access_wr", "Bus write accesses / instruction"),
    Metric("perf_mem_access", "Memory accesses / instruction"),
    Metric("perf_ase_spec", "Advanced SIMD (ASE_SPEC) / instruction"),
    Metric("perf_vfp_spec", "Floating point (VFP_SPEC) / instruction"),
    Metric("perf_inst_spec", "Speculative instructions / instruction"),
)


@dataclass(frozen=True)
class Trial:
    method: str
    trial_id: str
    path: Path
    data: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device-label", default="cloudlab-test")
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def perf_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.rglob("*_perf.csv"))
    if not files:
        raise FileNotFoundError(f"No perf CSVs found below {input_dir}")
    return files


def describe_perf_failures(failures: list[tuple[Path, str]]) -> str:
    errors = sorted({error for _, error in failures if error})
    details = "; ".join(errors) if errors else "unknown perf error"
    return (
        f"No usable perf instruction samples were found. "
        f"{len(failures)} analysis files are empty or contain failed perf rows. "
        f"Error: {details}. "
        "The experiment must be recollected with perf events supported by the "
        "CloudLab CPU before metrics per instruction can be calculated."
    )


def load_trials(input_dir: Path) -> list[Trial]:
    trials: list[Trial] = []
    failures: list[tuple[Path, str]] = []
    for path in perf_files(input_dir):
        frame = pd.read_csv(path, low_memory=False)
        if frame.empty:
            failures.append((path, "empty perf CSV"))
            continue

        method = str(frame.iloc[0].get("poisoning_method", ""))
        run_role = str(frame.iloc[0].get("run_role", ""))
        trial_id = str(frame.iloc[0].get("trial_id", ""))
        if run_role != "analysis" or method not in METHODS:
            continue

        statuses = frame.get("perf_status", pd.Series(index=frame.index, dtype=object))
        instructions = pd.to_numeric(frame.get("perf_instructions"), errors="coerce")
        usable = statuses.eq("ok") & instructions.gt(0)
        if not usable.any():
            error_values = frame.get(
                "perf_error", pd.Series(index=frame.index, dtype=object)
            ).dropna()
            failures.append(
                (path, str(error_values.iloc[0]) if not error_values.empty else "")
            )
            continue

        trials.append(Trial(method, trial_id, path, frame.loc[usable].copy()))

    if not trials:
        raise RuntimeError(describe_perf_failures(failures))

    coverage = {(trial.method, trial.trial_id) for trial in trials}
    for method in METHODS:
        method_trials = {trial_id for value, trial_id in coverage if value == method}
        if not method_trials:
            raise FileNotFoundError(f"No usable {method} trials found below {input_dir}")
    return trials


def aggregate_trial(trial: Trial) -> dict[str, pd.DataFrame]:
    summaries: dict[str, pd.DataFrame] = {}
    frame = trial.data
    epochs = pd.to_numeric(frame["epoch"], errors="coerce")
    instructions = pd.to_numeric(frame["perf_instructions"], errors="coerce")

    for phase in PHASES:
        phase_mask = frame["phase"].eq(phase) & epochs.notna() & instructions.gt(0)
        result = pd.DataFrame(index=sorted(epochs[phase_mask].astype(int).unique()))
        result.index.name = "epoch"
        for metric in PERF_METRICS:
            if metric.column not in frame.columns:
                result[metric.column] = float("nan")
                continue
            values = pd.to_numeric(frame[metric.column], errors="coerce")
            valid = phase_mask & values.notna()
            numerator = values[valid].groupby(epochs[valid].astype(int)).sum()
            denominator = instructions[valid].groupby(epochs[valid].astype(int)).sum()
            result[metric.column] = numerator / denominator
        summaries[phase] = result
    return summaries


def method_statistics(
    trials: list[Trial],
    summaries: dict[Path, dict[str, pd.DataFrame]],
    method: str,
    phase: str,
    metric: str,
) -> pd.DataFrame:
    matching = [trial for trial in trials if trial.method == method]
    values = pd.concat(
        {
            trial.trial_id: summaries[trial.path][phase][metric]
            for trial in matching
        },
        axis=1,
    )
    return pd.DataFrame(
        {
            "mean": values.mean(axis=1),
            "std": values.std(axis=1, ddof=1),
            "trials": values.count(axis=1),
        }
    )


def save_phase_figure(
    trials: list[Trial],
    summaries: dict[Path, dict[str, pd.DataFrame]],
    phase: str,
    device_label: str,
    output_dir: Path,
    dpi: int,
) -> Path:
    metrics = [
        metric
        for metric in PERF_METRICS
        if any(
            summaries[trial.path][phase][metric.column].notna().any()
            for trial in trials
        )
    ]
    if not metrics:
        raise RuntimeError(f"No usable {phase} perf metrics were found")

    figure, axes = plt.subplots(
        nrows=len(metrics),
        ncols=1,
        figsize=(14, max(3.0 * len(metrics), 8)),
        squeeze=False,
    )
    for axis, metric in zip(axes[:, 0], metrics):
        for method in METHODS:
            stats = method_statistics(
                trials, summaries, method, phase, metric.column
            ).dropna(subset=["mean"])
            if stats.empty:
                continue
            color = METHOD_COLORS[method]
            std = stats["std"].fillna(0.0)
            axis.fill_between(
                stats.index,
                stats["mean"] - std,
                stats["mean"] + std,
                color=color,
                alpha=0.15,
                linewidth=0,
            )
            axis.plot(
                stats.index,
                stats["mean"],
                color=color,
                linewidth=1.6,
                marker="o",
                markersize=3.2,
            )

        axis.set_ylabel(metric.label, fontsize=8)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.tick_params(axis="both", labelsize=7)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        axis.grid(True, color="#d8dadd", linewidth=0.55, alpha=0.8)
        axis.margins(x=0.02, y=0.08)

    axes[-1, 0].set_xlabel("Local training epoch", fontsize=9)
    trial_counts = {
        method: len({trial.trial_id for trial in trials if trial.method == method})
        for method in METHODS
    }
    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            linewidth=1.6,
            marker="o",
            markersize=4,
            label=f"{METHOD_LABELS[method]} (n={trial_counts[method]} trials)",
        )
        for method in METHODS
    ]
    handles.append(Patch(color="#777777", alpha=0.15, label="Mean +/- 1 SD"))
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.988),
        ncol=3,
        fontsize=9,
        frameon=False,
    )
    figure.suptitle(
        f"{device_label}: {phase.capitalize()} perf metrics per instruction",
        fontsize=15,
        y=0.999,
    )
    figure.tight_layout(rect=(0.04, 0.01, 0.995, 0.955), h_pad=1.1)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"metrics_per_instruction_{phase}.png"
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def main() -> None:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")

    trials = load_trials(args.input_dir.resolve())
    summaries = {trial.path: aggregate_trial(trial) for trial in trials}
    methods = ", ".join(
        f"{method}={sum(trial.method == method for trial in trials)}"
        for method in METHODS
    )
    print(f"Loaded CloudLab trials: {methods}")
    for phase in PHASES:
        path = save_phase_figure(
            trials,
            summaries,
            phase,
            args.device_label,
            args.output_dir.resolve(),
            args.dpi,
        )
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
