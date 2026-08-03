#!/usr/bin/env python3
"""Visualize epoch-wise batch counter/instruction totals without inverse fitting.

Every point starts from a batch-level ratio: reliable counter intervals are
summed, matching retired instructions are summed, and counter/instructions is
calculated for that batch. Clean and target batches are matched by trial,
epoch, and batch index. Batch ratios are averaged within each trial and epoch;
lines and bands then show the mean and standard deviation across trials.

The script also creates normalized and raw-total batch-index trajectory figures
for every counter. Each figure has one panel per epoch and shows the mean and
standard deviation across trials at every exact batch index.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-batch-efficiency")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
SEVEN_PATH = SCRIPT_DIR / "7_batch_calibrated_shape_comparison.py"
SEVEN_SPEC = importlib.util.spec_from_file_location("batch_calibrated_shape", SEVEN_PATH)
assert SEVEN_SPEC is not None and SEVEN_SPEC.loader is not None
SEVEN = importlib.util.module_from_spec(SEVEN_SPEC)
sys.modules.setdefault(SEVEN_SPEC.name, SEVEN)
SEVEN_SPEC.loader.exec_module(SEVEN)

DEFAULT_COLLECTION = SCRIPT_DIR / "cache_0727_jetson_cpu_20_trials"
DEFAULT_INPUT = DEFAULT_COLLECTION / "192.168.0.141"
DEFAULT_OUTPUT = DEFAULT_COLLECTION / "batch_efficiency_visualization"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--datasets", nargs="+", choices=("cifar10", "trashnet"), default=("cifar10",))
    parser.add_argument("--models", nargs="+", choices=("cnn", "vit"), default=("cnn",))
    parser.add_argument(
        "--target", choices=tuple(SEVEN.ROLE_LABELS), default="strong_augmentation"
    )
    parser.add_argument("--phase", choices=("forward", "backward"), default="forward")
    parser.add_argument("--epochs", nargs="+", type=int, default=list(range(10)))
    parser.add_argument(
        "--counters",
        nargs="+",
        choices=tuple(SEVEN.COUNTER_ALIASES),
        default=list(SEVEN.DEFAULT_COUNTERS),
    )
    parser.add_argument("--pmu-min-running", type=float, default=20.0)
    parser.add_argument("--min-instruction-mass", type=float, default=0.90)
    parser.add_argument("--edge-weight", type=float, default=0.5)
    parser.add_argument("--phase-label-lag", type=int, choices=(0, 1), default=1)
    parser.add_argument("--min-matched-batches", type=int, default=6)
    parser.add_argument("--expected-trials", type=int, default=20)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf"), default=("png",))
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.pmu_min_running <= 100:
        parser.error("--pmu-min-running must be in [0, 100]")
    if not 0 < args.min_instruction_mass <= 1:
        parser.error("--min-instruction-mass must be in (0, 1]")
    if not 0 < args.edge_weight <= 1:
        parser.error("--edge-weight must be in (0, 1]")
    if args.min_matched_batches < 2 or args.columns < 1:
        parser.error("minimum matched batches must be >= 2 and columns >= 1")
    return args


def collect_batch_values(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    input_dir = SEVEN.resolve_input_dir(args.input_dir)
    indexed = SEVEN.index_runs(SEVEN.discover_runs(input_dir))
    rows: list[dict[str, object]] = []
    skipped: list[str] = []

    groups = []
    for dataset in args.datasets:
        for model in args.models:
            clean_condition = SEVEN.condition_name(dataset, model, "clean")
            target_condition = SEVEN.condition_name(dataset, model, args.target)
            clean_trials = {trial for condition, trial in indexed if condition == clean_condition}
            target_trials = {trial for condition, trial in indexed if condition == target_condition}
            trials = sorted(clean_trials & target_trials)
            if not trials:
                warnings.warn(f"No paired runs for {clean_condition} and {target_condition}")
                continue
            groups.append((dataset, model, clean_condition, target_condition, trials))

    total = sum(len(trials) * len(args.epochs) * len(args.counters) for *_, trials in groups)
    position = 0
    for dataset, model, clean_condition, target_condition, trials in groups:
        for trial_id in trials:
            clean_run = SEVEN.load_run(indexed[(clean_condition, trial_id)], args.phase_label_lag)
            target_run = SEVEN.load_run(indexed[(target_condition, trial_id)], args.phase_label_lag)
            for epoch in args.epochs:
                for counter in args.counters:
                    position += 1
                    if position == 1 or position % 100 == 0 or position == total:
                        print(
                            f"[batch efficiency {position}/{total}] "
                            f"{dataset}/{model}/{trial_id}/epoch{epoch}/{counter}",
                            flush=True,
                        )
                    clean_obs, clean_totals, clean_reasons = SEVEN.batch_observations(
                        clean_run,
                        counter=counter,
                        epoch=epoch,
                        phase=args.phase,
                        pmu_min_running=args.pmu_min_running,
                        min_instruction_mass=args.min_instruction_mass,
                        edge_weight=args.edge_weight,
                    )
                    target_obs, target_totals, target_reasons = SEVEN.batch_observations(
                        target_run,
                        counter=counter,
                        epoch=epoch,
                        phase=args.phase,
                        pmu_min_running=args.pmu_min_running,
                        min_instruction_mass=args.min_instruction_mass,
                        edge_weight=args.edge_weight,
                    )
                    del clean_obs, target_obs
                    matched = sorted(
                        set(clean_totals.get("batch_idx", pd.Series(dtype=int)).astype(int))
                        & set(target_totals.get("batch_idx", pd.Series(dtype=int)).astype(int))
                    )
                    if len(matched) < args.min_matched_batches:
                        skipped.append(
                            f"{dataset}/{model}/{trial_id}/epoch{epoch}/{counter}: "
                            f"{len(matched)} matched batches; clean={clean_reasons}; "
                            f"target={target_reasons}"
                        )
                        continue
                    for role, condition, totals in (
                        ("clean", clean_condition, clean_totals),
                        ("target", target_condition, target_totals),
                    ):
                        selected = totals[totals["batch_idx"].isin(matched)]
                        for item in selected.itertuples():
                            rows.append({
                                "dataset": dataset,
                                "model": model,
                                "target_role": args.target,
                                "phase": args.phase,
                                "condition": condition,
                                "role": role,
                                "trial_id": trial_id,
                                "epoch": epoch,
                                "counter": counter,
                                "batch_idx": int(item.batch_idx),
                                "counter_total": float(item.counter_total),
                                "instruction_total": float(item.instruction_total),
                                "counter_per_instruction": float(item.counter_per_instruction),
                                "reliable_intervals": int(item.reliable_intervals),
                                "available_intervals": int(item.available_intervals),
                                "reliable_instruction_mass": float(item.reliable_instruction_mass),
                                "mean_running_percentage": float(item.mean_running_percentage),
                                "matched_batches": len(matched),
                            })
    return pd.DataFrame(rows), skipped


def summarize(
    values: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pooled_batches = values.groupby(
        ["dataset", "model", "target_role", "phase", "counter", "epoch", "role"],
        as_index=False,
    ).agg(
        batch_observations=("counter_per_instruction", "size"),
        trials=("trial_id", "nunique"),
        mean_counter_per_instruction=("counter_per_instruction", "mean"),
        std_counter_per_instruction=("counter_per_instruction", "std"),
        median_counter_per_instruction=("counter_per_instruction", "median"),
        mean_counter_total=("counter_total", "mean"),
        std_counter_total=("counter_total", "std"),
        median_counter_total=("counter_total", "median"),
        mean_instruction_total=("instruction_total", "mean"),
        std_instruction_total=("instruction_total", "std"),
        mean_running_percentage=("mean_running_percentage", "mean"),
        mean_reliable_instruction_mass=("reliable_instruction_mass", "mean"),
    )
    trial = values.groupby(
        [
            "dataset", "model", "target_role", "phase", "counter", "epoch", "role",
            "trial_id",
        ],
        as_index=False,
    ).agg(
        batches=("batch_idx", "nunique"),
        mean_batch_counter_per_instruction=("counter_per_instruction", "mean"),
        std_within_trial_batches=("counter_per_instruction", "std"),
    )
    epoch = trial.groupby(
        ["dataset", "model", "target_role", "phase", "counter", "epoch", "role"],
        as_index=False,
    ).agg(
        trials=("trial_id", "nunique"),
        mean_batches_per_trial=("batches", "mean"),
        mean_counter_per_instruction=("mean_batch_counter_per_instruction", "mean"),
        std_counter_per_instruction=("mean_batch_counter_per_instruction", "std"),
        median_counter_per_instruction=("mean_batch_counter_per_instruction", "median"),
        mean_within_trial_batch_std=("std_within_trial_batches", "mean"),
    )
    batch_index = values.groupby(
        [
            "dataset", "model", "target_role", "phase", "counter", "epoch", "role",
            "batch_idx",
        ],
        as_index=False,
    ).agg(
        trials=("trial_id", "nunique"),
        mean_counter_per_instruction=("counter_per_instruction", "mean"),
        std_counter_per_instruction=("counter_per_instruction", "std"),
        median_counter_per_instruction=("counter_per_instruction", "median"),
        mean_counter_total=("counter_total", "mean"),
        std_counter_total=("counter_total", "std"),
        median_counter_total=("counter_total", "median"),
        mean_instruction_total=("instruction_total", "mean"),
        std_instruction_total=("instruction_total", "std"),
        mean_running_percentage=("mean_running_percentage", "mean"),
        mean_reliable_instruction_mass=("reliable_instruction_mass", "mean"),
    )
    return epoch, trial, pooled_batches, batch_index


def plot_model(
    summary: pd.DataFrame,
    *,
    dataset: str,
    model: str,
    target: str,
    phase: str,
    counters: list[str],
    columns: int,
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> None:
    available = [
        counter
        for counter in counters
        if not summary[
            summary["dataset"].eq(dataset)
            & summary["model"].eq(model)
            & summary["counter"].eq(counter)
        ].empty
    ]
    if not available:
        return
    rows = math.ceil(len(available) / columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(4.2 * columns, 3.2 * rows), squeeze=False
    )
    colors = {"clean": "#25282b", "target": "#c84c3a"}
    labels = {"clean": "clean", "target": SEVEN.ROLE_LABELS[target]}
    for axis, counter in zip(axes.flat, available):
        selected = summary[
            summary["dataset"].eq(dataset)
            & summary["model"].eq(model)
            & summary["counter"].eq(counter)
        ]
        for role in ("clean", "target"):
            curve = selected[selected["role"].eq(role)].sort_values("epoch")
            if curve.empty:
                continue
            x = curve["epoch"].to_numpy(dtype=float)
            mean = curve["mean_counter_per_instruction"].to_numpy(dtype=float)
            sd = curve["std_counter_per_instruction"].to_numpy(dtype=float)
            axis.fill_between(x, mean - sd, mean + sd, color=colors[role], alpha=0.13)
            axis.plot(x, mean, color=colors[role], marker="o", markersize=3.5, label=labels[role])
        axis.set_title(counter, fontsize=9)
        axis.set_xticks(sorted(selected["epoch"].astype(int).unique()))
        axis.grid(True, color="#d9dde1", linewidth=0.5)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
    for axis in axes.flat[len(available):]:
        axis.set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=8)
    figure.supxlabel("Epoch")
    figure.supylabel("Batch total counter / batch total instructions")
    figure.suptitle(
        f"{dataset} | {model.upper()} | {phase} | clean vs {SEVEN.ROLE_LABELS[target]}\n"
        "Per trial: mean of matched batch ratios; line/band: mean +/- SD across trials",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.02, 0.02, 1.0, 0.94))
    prefix = output_dir / dataset / model / target / f"{phase}_batch_efficiency"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        kwargs = {"dpi": dpi} if extension == "png" else {}
        figure.savefig(
            prefix.with_suffix(f".{extension}"), facecolor="white", **kwargs
        )
    plt.close(figure)


def plot_batch_index_metric(
    summary: pd.DataFrame,
    *,
    dataset: str,
    model: str,
    target: str,
    phase: str,
    counter: str,
    epochs: list[int],
    output_dir: Path,
    formats: list[str],
    dpi: int,
    normalize_by_instructions: bool,
) -> bool:
    selected = summary[
        summary["dataset"].eq(dataset)
        & summary["model"].eq(model)
        & summary["counter"].eq(counter)
        & summary["epoch"].isin(epochs)
    ]
    if selected.empty:
        return False

    epoch_order = [epoch for epoch in epochs if epoch in set(selected["epoch"].astype(int))]
    columns = min(5, max(1, len(epoch_order)))
    rows = math.ceil(len(epoch_order) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.0 * columns, 3.1 * rows),
        squeeze=False,
        sharey=True,
    )
    colors = {"clean": "#25282b", "target": "#c84c3a"}
    labels = {"clean": "clean", "target": SEVEN.ROLE_LABELS[target]}
    if normalize_by_instructions:
        mean_column = "mean_counter_per_instruction"
        std_column = "std_counter_per_instruction"
        y_label = "Batch total counter / batch total instructions"
        filename_suffix = "batch_trend_by_epoch"
        value_label = "counter/instruction"
    else:
        mean_column = "mean_counter_total"
        std_column = "std_counter_total"
        y_label = "Raw batch counter total"
        filename_suffix = "raw_batch_total_trend_by_epoch"
        value_label = "raw counter total"

    finite_low: list[float] = []
    finite_high: list[float] = []
    for epoch in epoch_order:
        epoch_data = selected[selected["epoch"].eq(epoch)]
        for role in ("clean", "target"):
            curve = epoch_data[epoch_data["role"].eq(role)]
            mean = curve[mean_column].to_numpy(dtype=float)
            sd = curve[std_column].fillna(0.0).to_numpy(dtype=float)
            finite_low.extend((mean - sd)[np.isfinite(mean - sd)].tolist())
            finite_high.extend((mean + sd)[np.isfinite(mean + sd)].tolist())
    y_low = min(finite_low, default=0.0)
    y_high = max(finite_high, default=1.0)
    if not np.isfinite(y_low) or not np.isfinite(y_high):
        y_low, y_high = 0.0, 1.0
    if y_high <= y_low:
        padding = max(abs(y_high) * 0.05, 1e-12)
    else:
        padding = (y_high - y_low) * 0.05
    y_limits = (min(0.0, y_low - padding), y_high + padding)

    for axis, epoch in zip(axes.flat, epoch_order):
        epoch_data = selected[selected["epoch"].eq(epoch)]
        for role in ("clean", "target"):
            curve = epoch_data[epoch_data["role"].eq(role)].sort_values("batch_idx")
            if curve.empty:
                continue
            x = curve["batch_idx"].to_numpy(dtype=float)
            mean = curve[mean_column].to_numpy(dtype=float)
            sd = curve[std_column].fillna(0.0).to_numpy(dtype=float)
            axis.fill_between(x, mean - sd, mean + sd, color=colors[role], alpha=0.13)
            axis.plot(
                x,
                mean,
                color=colors[role],
                marker="o",
                markersize=2.8,
                linewidth=1.2,
                label=labels[role],
            )
        batch_indices = sorted(epoch_data["batch_idx"].astype(int).unique())
        axis.set_title(f"Epoch {epoch}", fontsize=9)
        axis.set_xticks(batch_indices)
        axis.set_ylim(*y_limits)
        axis.grid(True, color="#d9dde1", linewidth=0.5)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
    for axis in axes.flat[len(epoch_order):]:
        axis.set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=8)
    figure.supxlabel("Batch index", y=0.035)
    figure.supylabel(y_label, x=0.012, fontsize=10)
    figure.suptitle(
        f"{dataset} | {model.upper()} | {phase} | {counter}\n"
        f"clean vs {SEVEN.ROLE_LABELS[target]}: {value_label}, mean +/- SD across trials",
        fontsize=11,
        y=0.985,
    )
    left_margin = 0.15 if columns <= 2 else 0.07
    top_margin = 0.76 if rows == 1 else 0.84
    figure.subplots_adjust(
        left=left_margin,
        right=0.99,
        bottom=0.14,
        top=top_margin,
        wspace=0.18,
        hspace=0.38,
    )
    prefix = output_dir / dataset / model / target / f"{counter}_{filename_suffix}"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        kwargs = {"dpi": dpi} if extension == "png" else {}
        figure.savefig(
            prefix.with_suffix(f".{extension}"), facecolor="white", **kwargs
        )
    plt.close(figure)
    return True


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    marker = output_dir / "epoch_batch_efficiency_summary.csv"
    if marker.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists below {output_dir}; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    values, skipped = collect_batch_values(args)
    if values.empty:
        raise RuntimeError("No matched reliable batch totals were available")
    epoch_summary, trial_summary, pooled_batch_summary, batch_index_summary = summarize(values)
    values.to_csv(output_dir / "batch_efficiency_values.csv", index=False)
    epoch_summary.to_csv(output_dir / "epoch_batch_efficiency_summary.csv", index=False)
    trial_summary.to_csv(output_dir / "trial_epoch_batch_efficiency_summary.csv", index=False)
    pooled_batch_summary.to_csv(output_dir / "pooled_batch_diagnostics.csv", index=False)
    batch_index_summary.to_csv(output_dir / "batch_index_efficiency_summary.csv", index=False)

    figures = 0
    for dataset in args.datasets:
        for model in args.models:
            before = figures
            selected = epoch_summary[
                epoch_summary["dataset"].eq(dataset) & epoch_summary["model"].eq(model)
            ]
            if not selected.empty:
                plot_model(
                    epoch_summary,
                    dataset=dataset,
                    model=model,
                    target=args.target,
                    phase=args.phase,
                    counters=list(args.counters),
                    columns=args.columns,
                    output_dir=output_dir,
                    formats=list(args.formats),
                    dpi=args.dpi,
                )
                figures += 1
                for counter in args.counters:
                    for normalize_by_instructions in (True, False):
                        if plot_batch_index_metric(
                            batch_index_summary,
                            dataset=dataset,
                            model=model,
                            target=args.target,
                            phase=args.phase,
                            counter=counter,
                            epochs=list(args.epochs),
                            output_dir=output_dir,
                            formats=list(args.formats),
                            dpi=args.dpi,
                            normalize_by_instructions=normalize_by_instructions,
                        ):
                            figures += 1
            if figures == before:
                warnings.warn(f"No figure data for {dataset}/{model}")

    run_summary = {
        "input_dir": str(SEVEN.resolve_input_dir(args.input_dir)),
        "output_dir": str(output_dir),
        "datasets": list(args.datasets),
        "models": list(args.models),
        "target": args.target,
        "phase": args.phase,
        "phase_label_lag_rows": args.phase_label_lag,
        "pmu_min_running_percentage": args.pmu_min_running,
        "minimum_reliable_instruction_mass": args.min_instruction_mass,
        "ratio_definition": "per batch: reliable counter total / matching instruction total",
        "trial_value_definition": "mean of matched batch counter/instruction ratios within each trial and epoch",
        "line_definition": "mean across trial-level epoch values",
        "band_definition": "standard deviation across trial-level epoch values",
        "batch_index_figure_definition": (
            "for each epoch and exact batch_idx, mean and standard deviation across trials "
            "of both the batch counter/instruction ratio and the raw batch counter total"
        ),
        "inverse_problem_used": False,
        "figures": figures,
        "batch_rows": len(values),
        "skipped_count": len(skipped),
        "skipped": skipped,
        "measurement_note": (
            "Batch totals are reconstructed by summing PMU-filtered 50 Hz intervals; "
            "they are not from an independent synchronous batch counter read."
        ),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2) + "\n")
    print(f"Wrote {figures} figures and {len(values)} batch rows to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
