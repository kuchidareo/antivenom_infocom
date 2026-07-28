#!/usr/bin/env python3
"""Plot epoch trends for training loss and forward perf batch totals."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-motivational-raw")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "motivational_0727_30_trials" / "192.168.0.112"
DEFAULT_OUTPUT = SCRIPT_DIR / "motivational_0727_30_trials" / "raw_visualization"

CONDITIONS = {
    "clean": "Clean",
    "moderate_augmentation": "Moderate augmentation",
    "strong_augmentation": "Strong augmentation",
    "availability_shortcut": "Availability shortcut",
}
COLORS = {
    "clean": "#2878b5",
    "moderate_augmentation": "#3a923a",
    "strong_augmentation": "#d18b24",
    "availability_shortcut": "#c84c3a",
}

PERF_METRICS = {
    "perf_cycles": "Cycles",
    "perf_instructions": "Instructions",
    "perf_task_clock": "Task clock",
    "perf_context_switches": "Context switches",
    "perf_cpu_migrations": "CPU migrations",
    "perf_page_faults": "Page faults",
    "perf_branches": "Branches",
    "perf_branch_misses": "Branch misses",
    "perf_l1d_cache_rd": "L1D read accesses",
    "perf_l1d_cache_refill_rd": "L1D read refills",
    "perf_l1d_cache_wr": "L1D write accesses",
    "perf_l1d_cache_refill_wr": "L1D write refills",
    "perf_l2d_cache_rd": "L2D read accesses",
    "perf_l2d_cache_refill_rd": "L2D read refills",
    "perf_l2d_cache_wr": "L2D write accesses",
    "perf_l2d_cache_refill_wr": "L2D write refills",
    "perf_bus_access_rd": "Bus read accesses",
    "perf_bus_access_wr": "Bus write accesses",
    "perf_mem_access": "Memory accesses",
    "perf_ase_spec": "ASE speculative ops",
    "perf_vfp_spec": "VFP speculative ops",
    "perf_inst_spec": "Speculative instructions",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phase", default="forward")
    parser.add_argument("--models", nargs="+", choices=("cnn", "vit"), default=("cnn", "vit"))
    parser.add_argument("--include-partial-batch", action="store_true")
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def condition_parts(name: str) -> tuple[str, str] | None:
    for model in ("cnn", "vit"):
        prefix = f"cifar10_{model}_"
        if name.startswith(prefix):
            role = name.removeprefix(prefix)
            if role in CONDITIONS:
                return model, role
    return None


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def load_metrics(
    path: Path,
) -> tuple[str, list[dict[str, object]], set[tuple[int, int]], dict[int, int], dict[int, int]]:
    frame = pd.read_csv(path, low_memory=False)
    required = {"trial_id", "epoch", "batch_idx", "metric_event", "loss", "num_examples"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing metrics columns {sorted(missing)}")
    trial_id = str(frame["trial_id"].dropna().iloc[0])

    epoch_rows = frame[frame["metric_event"].astype(str).eq("train_epoch")].copy()
    epoch_rows["epoch"] = numeric(epoch_rows, "epoch")
    epoch_rows["loss"] = numeric(epoch_rows, "loss")
    loss_rows = [
        {
            "trial_id": trial_id,
            "epoch": int(row.epoch),
            "metric": "training_loss",
            "value": float(row.loss),
        }
        for row in epoch_rows.dropna(subset=["epoch", "loss"]).itertuples()
    ]

    batches = frame[frame["metric_event"].astype(str).eq("train_batch")].copy()
    for column in ("epoch", "batch_idx", "num_examples"):
        batches[column] = numeric(batches, column)
    batches = batches.dropna(subset=["epoch", "batch_idx", "num_examples"])
    allowed: set[tuple[int, int]] = set()
    expected_by_epoch: dict[int, int] = {}
    all_expected_by_epoch: dict[int, int] = {}
    for epoch, group in batches.groupby("epoch"):
        epoch_id = int(epoch)
        all_expected_by_epoch[epoch_id] = int(group["batch_idx"].nunique())
        full_size = float(group["num_examples"].max())
        full = group[group["num_examples"].eq(full_size)]
        pairs = {(epoch_id, int(batch)) for batch in full["batch_idx"]}
        allowed.update(pairs)
        expected_by_epoch[epoch_id] = len(pairs)
    return trial_id, loss_rows, allowed, expected_by_epoch, all_expected_by_epoch


def load_perf(
    path: Path,
    *,
    trial_id: str,
    phase: str,
    allowed_batches: set[tuple[int, int]] | None,
    expected_by_epoch: dict[int, int],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[str]]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    metric_columns = [column for column in PERF_METRICS if column in header]
    required = ["epoch", "batch_idx", "phase"]
    if "perf_status" in header:
        required.append("perf_status")
    frame = pd.read_csv(path, usecols=required + metric_columns, low_memory=False)
    for column in ("epoch", "batch_idx", *metric_columns):
        frame[column] = numeric(frame, column)
    selected = frame[frame["phase"].astype(str).eq(phase)].copy()
    if "perf_status" in selected:
        selected = selected[selected["perf_status"].astype(str).eq("ok")]
    selected = selected.dropna(subset=["epoch", "batch_idx"])
    selected["epoch"] = selected["epoch"].astype(int)
    selected["batch_idx"] = selected["batch_idx"].astype(int)
    if allowed_batches is not None:
        pairs = pd.Series(list(zip(selected["epoch"], selected["batch_idx"])), index=selected.index)
        selected = selected[pairs.isin(allowed_batches)]

    batch_totals = selected.groupby(["epoch", "batch_idx"])[metric_columns].sum(min_count=1)
    epoch_means = batch_totals.groupby(level="epoch").mean()
    values: list[dict[str, object]] = []
    for epoch, row in epoch_means.iterrows():
        for metric in metric_columns:
            value = row[metric]
            if pd.notna(value):
                values.append(
                    {
                        "trial_id": trial_id,
                        "epoch": int(epoch),
                        "metric": metric,
                        "value": float(value),
                    }
                )

    normalized_values: list[dict[str, object]] = []
    instruction_column = "perf_instructions"
    if instruction_column in batch_totals:
        instructions = batch_totals[instruction_column].where(batch_totals[instruction_column] > 0)
        ratios = batch_totals.drop(columns=[instruction_column]).div(instructions, axis=0)
        normalized_epoch_means = ratios.groupby(level="epoch").mean()
        for epoch, row in normalized_epoch_means.iterrows():
            for metric in ratios.columns:
                value = row[metric]
                if pd.notna(value):
                    normalized_values.append(
                        {
                            "trial_id": trial_id,
                            "epoch": int(epoch),
                            "metric": metric,
                            "value": float(value),
                        }
                    )

    observed = selected.groupby("epoch")["batch_idx"].nunique()
    coverage = [
        {
            "trial_id": trial_id,
            "epoch": epoch,
            "expected_batches": expected,
            "observed_batches": int(observed.get(epoch, 0)),
            "coverage_fraction": float(observed.get(epoch, 0) / expected) if expected else math.nan,
        }
        for epoch, expected in sorted(expected_by_epoch.items())
    ]
    return values, normalized_values, coverage, metric_columns


def discover(input_dir: Path, models: set[str]) -> list[tuple[str, str, Path, Path]]:
    runs: list[tuple[str, str, Path, Path]] = []
    for condition_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        parts = condition_parts(condition_dir.name)
        if parts is None or parts[0] not in models:
            continue
        model, role = parts
        for perf_path in sorted(condition_dir.glob("*_perf.csv")):
            prefix = perf_path.name.removesuffix("_perf.csv")
            metrics_path = perf_path.with_name(f"{prefix}_metrics.csv")
            if not metrics_path.is_file():
                warnings.warn(f"Skipping {perf_path}: matched metrics file is missing")
                continue
            runs.append((model, role, perf_path, metrics_path))
    if not runs:
        raise FileNotFoundError(f"No matched perf/metrics runs found below {input_dir}")
    return runs


def summarize(values: pd.DataFrame) -> pd.DataFrame:
    return (
        values.groupby(["model", "condition", "metric", "epoch"], as_index=False)["value"]
        .agg(trial_count="count", mean="mean", std="std")
        .sort_values(["model", "metric", "condition", "epoch"])
    )


def plot_model(
    summary: pd.DataFrame,
    model: str,
    output_dir: Path,
    dpi: int,
    partial: bool,
    *,
    per_instruction: bool,
) -> None:
    selected = summary[summary["model"].eq(model)]
    available_perf = [metric for metric in PERF_METRICS if metric in set(selected["metric"])]
    metrics = ["training_loss", *available_perf]
    columns = 6
    rows = math.ceil(len(metrics) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(23, 3.25 * rows), squeeze=False)
    flat_axes = axes.ravel()
    for axis, metric in zip(flat_axes, metrics):
        metric_frame = selected[selected["metric"].eq(metric)]
        for condition in CONDITIONS:
            group = metric_frame[metric_frame["condition"].eq(condition)].sort_values("epoch")
            if group.empty:
                continue
            x = group["epoch"].to_numpy(dtype=float)
            mean = group["mean"].to_numpy(dtype=float)
            std = group["std"].fillna(0.0).to_numpy(dtype=float)
            color = COLORS[condition]
            axis.fill_between(x, mean - std, mean + std, color=color, alpha=0.13, linewidth=0)
            axis.plot(x, mean, color=color, marker="o", markersize=3.0, linewidth=1.5, label=CONDITIONS[condition])
        metric_title = PERF_METRICS[metric] if metric != "training_loss" else "Training loss"
        if per_instruction and metric != "training_loss":
            metric_title += " / instruction"
        axis.set_title(metric_title, fontsize=10)
        axis.set_xlabel("Epoch")
        if metric == "training_loss":
            axis.set_ylabel("Loss")
        elif per_instruction:
            axis.set_ylabel("Mean batch ratio")
        else:
            axis.set_ylabel("Mean batch total")
        axis.set_xticks(range(10))
        axis.grid(True, color="#d9dde1", linewidth=0.55)
        if metric != "training_loss":
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
    for axis in flat_axes[len(metrics):]:
        axis.set_visible(False)

    handles, labels = flat_axes[0].get_legend_handles_labels()
    batch_policy = "including partial batches" if partial else "excluding the final partial batch"
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.985))
    value_description = (
        "trial-level mean of per-batch counter/instruction ratios"
        if per_instruction
        else "trial-level mean batch totals"
    )
    figure.suptitle(
        f"CIFAR-10 {model.upper()} forward execution by epoch\n"
        f"Line: mean across {value_description}; band: +/- 1 SD across trials; {batch_policy}",
        fontsize=13,
        y=1.015,
    )
    figure.tight_layout(rect=(0.01, 0.01, 0.99, 0.955))
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "per_instruction_metrics" if per_instruction else "raw_metrics"
    prefix = output_dir / f"cifar10_{model}_forward_epoch_{suffix}"
    figure.savefig(prefix.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)
    gc.collect()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    runs = discover(input_dir, set(args.models))
    value_rows: list[dict[str, object]] = []
    normalized_value_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    available_metrics: set[str] = set()

    for index, (model, condition, perf_path, metrics_path) in enumerate(runs, start=1):
        trial_id, losses, full_batches, full_expected, all_expected = load_metrics(metrics_path)
        allowed = None if args.include_partial_batch else full_batches
        expected = all_expected if args.include_partial_batch else full_expected
        perf_values, normalized_perf_values, coverage, columns = load_perf(
            perf_path,
            trial_id=trial_id,
            phase=args.phase,
            allowed_batches=allowed,
            expected_by_epoch=expected,
        )
        for row in (*losses, *perf_values):
            row.update(model=model, condition=condition)
            value_rows.append(row)
        for row in (*losses, *normalized_perf_values):
            row.update(model=model, condition=condition)
            normalized_value_rows.append(row)
        for row in coverage:
            row.update(model=model, condition=condition)
            coverage_rows.append(row)
        available_metrics.update(columns)
        if index % 20 == 0 or index == len(runs):
            print(f"Loaded {index}/{len(runs)} runs", flush=True)

    values = pd.DataFrame(value_rows)
    normalized_values = pd.DataFrame(normalized_value_rows)
    coverage = pd.DataFrame(coverage_rows)
    summary = summarize(values)
    normalized_summary = summarize(normalized_values)
    output_dir.mkdir(parents=True, exist_ok=True)
    values.to_csv(output_dir / "trial_epoch_values.csv", index=False)
    summary.to_csv(output_dir / "epoch_summary.csv", index=False)
    normalized_values.to_csv(output_dir / "trial_epoch_per_instruction_values.csv", index=False)
    normalized_summary.to_csv(output_dir / "epoch_per_instruction_summary.csv", index=False)
    coverage.to_csv(output_dir / "forward_batch_coverage.csv", index=False)

    coverage_summary = (
        coverage.groupby(["model", "condition"], as_index=False)
        .agg(
            trials=("trial_id", "nunique"),
            median_coverage=("coverage_fraction", "median"),
            mean_coverage=("coverage_fraction", "mean"),
            minimum_coverage=("coverage_fraction", "min"),
        )
    )
    coverage_summary.to_csv(output_dir / "forward_batch_coverage_summary.csv", index=False)
    for row in coverage_summary.itertuples():
        if row.median_coverage < 0.8:
            warnings.warn(
                f"{row.model}/{row.condition}: median forward batch coverage is "
                f"{row.median_coverage:.1%}; perf means use observed batches only"
            )

    for model in args.models:
        plot_model(
            summary,
            model,
            output_dir,
            args.dpi,
            args.include_partial_batch,
            per_instruction=False,
        )
        plot_model(
            normalized_summary,
            model,
            output_dir,
            args.dpi,
            args.include_partial_batch,
            per_instruction=True,
        )

    run_summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "phase": args.phase,
        "models": list(args.models),
        "run_count": len(runs),
        "trials_per_condition": {
            f"{row.model}/{row.condition}": int(row.trials)
            for row in coverage_summary.itertuples()
        },
        "epochs": sorted(values["epoch"].dropna().astype(int).unique().tolist()),
        "perf_metrics": sorted(available_metrics),
        "partial_batch_policy": "included" if args.include_partial_batch else "excluded",
        "aggregation": (
            "Sum perf interval increments within each observed forward batch; average observed "
            "batch totals within each trial/epoch; then mean and sample SD across trials."
        ),
        "per_instruction_aggregation": (
            "For each observed forward batch, divide its summed counter increments by its summed "
            "retired instructions; average batch ratios within each trial/epoch; then mean and "
            "sample SD across trials. Instructions/instructions is omitted because it is identically one."
        ),
        "coverage_warning": (
            "Missing forward batches are not imputed as zero. Low 10 Hz coverage, especially for "
            "CNN, means those curves summarize observed batches rather than every executed batch."
        ),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2) + "\n")
    print(f"Saved figures and CSV summaries to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
