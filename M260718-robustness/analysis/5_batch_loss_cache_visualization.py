#!/usr/bin/env python3
"""Plot reliability-filtered forward counters and batch loss by global batch."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "collected_logs"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "visualization_batch_loss_cache"

DATASET = "kuchidareo/small_trashnet"
MODEL = "simple_cnn"
BATCH_SIZE = 16
AUGMENTATION = "baseline"
METHODS = ("clean", "availability_shortcuts")
COLORS = {
    "clean": "#2468a2",
    "availability_shortcuts": "#c43c39",
}
LABELS = {
    "clean": "Clean",
    "availability_shortcuts": "Availability shortcuts",
}

PLOT_METRICS = (
    ("l1d_accesses", "Total L1D accesses per forward batch"),
    ("instructions", "Total instructions per forward batch"),
    ("l1d_accesses_per_instruction", "L1D accesses / instruction"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--device",
        action="append",
        help="Restrict the plot to a device ID; repeat for multiple devices.",
    )
    parser.add_argument(
        "--minimum-running-pct",
        type=float,
        default=20.0,
        help=(
            "Reject a batch metric when any contributing perf interval ran its "
            "required PMU event for less than this percentage of the interval. "
            "Counts are already multiplex-scaled by perf and are not rescaled."
        ),
    )
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    if not 0.0 <= args.minimum_running_pct <= 100.0:
        parser.error("--minimum-running-pct must be between 0 and 100")
    return args


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def read_metadata(perf_path: Path) -> dict[str, object]:
    with perf_path.open(newline="") as handle:
        row = next(csv.DictReader(handle), None)
    if row is None:
        raise ValueError(f"Empty perf CSV: {perf_path}")

    return {
        "perf_path": perf_path,
        "metrics_path": perf_path.with_name(
            perf_path.name.replace("_perf.csv", "_metrics.csv")
        ),
        "device_id": row.get("device_id") or perf_path.parent.parent.name,
        "method": row.get("poisoning_method", ""),
        "dataset": row.get("dataset", ""),
        "partition_method": row.get("partition_method", ""),
        "model": row.get("model", ""),
        "batch_size": int(row.get("batch_size") or 0),
        "augmentation": row.get("augmentation_profile", ""),
        "background_enabled": parse_bool(row.get("background_workload_enabled")),
        "trial_id": row.get("trial_id", ""),
        "start_time": float(row.get("timestamp_unix") or 0.0),
    }


def is_target_run(run: dict[str, object]) -> bool:
    return (
        run["method"] in METHODS
        and run["dataset"] == DATASET
        and run["partition_method"] == "iid"
        and run["model"] == MODEL
        and run["batch_size"] == BATCH_SIZE
        and run["augmentation"] == AUGMENTATION
        and not run["background_enabled"]
    )


def discover_runs(
    input_dir: Path, selected_devices: set[str]
) -> dict[tuple[str, str], dict[str, object]]:
    candidates: dict[tuple[str, str], list[dict[str, object]]] = {}
    for perf_path in sorted(input_dir.rglob("*_perf.csv")):
        try:
            run = read_metadata(perf_path)
        except ValueError as exc:
            warnings.warn(str(exc))
            continue
        if not is_target_run(run):
            continue
        device = str(run["device_id"])
        if selected_devices and device not in selected_devices:
            continue
        metrics_path = Path(run["metrics_path"])
        if not metrics_path.exists():
            warnings.warn(f"Missing metrics CSV for {perf_path}")
            continue
        candidates.setdefault((device, str(run["method"])), []).append(run)

    selected: dict[tuple[str, str], dict[str, object]] = {}
    for key, runs in candidates.items():
        runs.sort(key=lambda run: float(run["start_time"]))
        selected[key] = runs[-1]
        if len(runs) > 1:
            warnings.warn(
                f"{key}: selected latest {Path(runs[-1]['perf_path']).name}; "
                f"ignored {len(runs) - 1} older run(s)"
            )

    devices = sorted({device for device, _ in selected})
    paired_devices = [
        device
        for device in devices
        if all((device, method) in selected for method in METHODS)
    ]
    if not paired_devices:
        raise ValueError(
            "No device has matched clean and availability_shortcuts baseline runs"
        )
    missing = set(devices) - set(paired_devices)
    if missing:
        warnings.warn(f"Skipping devices without both conditions: {sorted(missing)}")
    return {
        key: run for key, run in selected.items() if key[0] in paired_devices
    }


def summarize_perf_batch(group: pd.DataFrame, minimum_running_pct: float) -> pd.Series:
    instructions = group["perf_instructions"]
    instruction_running = group["perf_instructions_enabled_pct"]
    l1d_accesses = group["perf_l1d_cache"]
    l1d_running = group["perf_l1d_cache_enabled_pct"]

    instruction_reliable = bool(
        instructions.notna().all()
        and instructions.ge(0).all()
        and instruction_running.notna().all()
        and instruction_running.ge(minimum_running_pct).all()
    )
    l1d_reliable = bool(
        l1d_accesses.notna().all()
        and l1d_accesses.ge(0).all()
        and l1d_running.notna().all()
        and l1d_running.ge(minimum_running_pct).all()
    )

    instruction_total = float(instructions.sum()) if instruction_reliable else np.nan
    l1d_total = float(l1d_accesses.sum()) if l1d_reliable else np.nan
    jointly_reliable = instruction_reliable and l1d_reliable
    ratio = (
        l1d_total / instruction_total
        if jointly_reliable and instruction_total > 0
        else np.nan
    )
    return pd.Series(
        {
            "l1d_accesses": l1d_total,
            "instructions": instruction_total,
            "l1d_accesses_per_instruction": ratio,
            "perf_intervals": len(group),
            "instructions_reliable": instruction_reliable,
            "l1d_accesses_reliable": l1d_reliable,
            "jointly_reliable": jointly_reliable,
            "instructions_running_pct_min": instruction_running.min(skipna=True),
            "instructions_running_pct_mean": instruction_running.mean(skipna=True),
            "l1d_running_pct_min": l1d_running.min(skipna=True),
            "l1d_running_pct_mean": l1d_running.mean(skipna=True),
        }
    )


def load_batch_data(
    run: dict[str, object], minimum_running_pct: float
) -> pd.DataFrame:
    perf_path = Path(run["perf_path"])
    perf_header = set(pd.read_csv(perf_path, nrows=0).columns)
    required = {
        "epoch",
        "batch_idx",
        "phase",
        "perf_status",
        "perf_l1d_cache",
        "perf_l1d_cache_enabled_pct",
        "perf_instructions",
        "perf_instructions_enabled_pct",
    }
    missing = required - perf_header
    if missing:
        raise ValueError(f"{perf_path} is missing columns: {sorted(missing)}")

    perf = pd.read_csv(perf_path, usecols=sorted(required))
    perf = perf[(perf["phase"] == "forward") & (perf["perf_status"] == "ok")].copy()
    for column in (
        "epoch",
        "batch_idx",
        "perf_l1d_cache",
        "perf_l1d_cache_enabled_pct",
        "perf_instructions",
        "perf_instructions_enabled_pct",
    ):
        perf[column] = pd.to_numeric(perf[column], errors="coerce")
    perf = perf.dropna(subset=["epoch", "batch_idx"])

    # perf stat has already scaled each multiplexed interval count by
    # time_enabled/time_running. Percentage-running is therefore a reliability
    # check only; scaling these values again would double-correct them.
    batches = (
        perf.groupby(["epoch", "batch_idx"], sort=True)
        .apply(
            summarize_perf_batch,
            minimum_running_pct=minimum_running_pct,
            include_groups=False,
        )
        .reset_index()
        .sort_values(["epoch", "batch_idx"])
    )

    metrics_path = Path(run["metrics_path"])
    metrics = pd.read_csv(
        metrics_path,
        usecols=["epoch", "batch_idx", "metric_event", "loss"],
    )
    metrics = metrics[metrics["metric_event"] == "train_batch"].copy()
    for column in ("epoch", "batch_idx", "loss"):
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    metrics = metrics.dropna(subset=["epoch", "batch_idx", "loss"])
    metrics = metrics.drop_duplicates(["epoch", "batch_idx"], keep="last")

    joined = batches.merge(
        metrics[["epoch", "batch_idx", "loss"]],
        on=["epoch", "batch_idx"],
        how="inner",
        validate="one_to_one",
    )
    if joined.empty:
        raise ValueError(f"No forward batches could be joined to loss in {perf_path}")

    joined["epoch"] = joined["epoch"].astype(int)
    joined["batch_idx"] = joined["batch_idx"].astype(int)
    joined = joined.sort_values(["epoch", "batch_idx"]).reset_index(drop=True)
    joined["global_batch_idx"] = np.arange(1, len(joined) + 1)
    joined["device_id"] = str(run["device_id"])
    joined["method"] = str(run["method"])
    joined["trial_id"] = str(run["trial_id"])
    joined["perf_path"] = str(perf_path)
    return joined


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    value_columns = [metric for metric, _ in PLOT_METRICS] + ["loss"]
    summary = (
        data.groupby(["method", "global_batch_idx"], as_index=False)[value_columns]
        .agg(["mean", "std"])
    )
    summary.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    return summary.rename(
        columns={"method_": "method", "global_batch_idx_": "global_batch_idx"}
    )


def epoch_boundaries(data: pd.DataFrame) -> tuple[list[float], list[tuple[float, str]]]:
    mapping = (
        data.groupby(["epoch", "global_batch_idx"], as_index=False)
        .size()
        .groupby("epoch")["global_batch_idx"]
        .agg(["min", "max"])
        .sort_index()
    )
    boundaries = [float(row["max"]) + 0.5 for _, row in mapping.iloc[:-1].iterrows()]
    labels = [
        ((float(row["min"]) + float(row["max"])) / 2.0, f"Epoch {int(epoch)}")
        for epoch, row in mapping.iterrows()
    ]
    return boundaries, labels


def plot(
    data: pd.DataFrame,
    summary: pd.DataFrame,
    output_path: Path,
    dpi: int,
    minimum_running_pct: float,
) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    boundaries, epoch_labels = epoch_boundaries(data)

    for axis, (metric, ylabel) in zip(axes, PLOT_METRICS):
        loss_axis = axis.twinx()
        for method in METHODS:
            values = summary[summary["method"] == method]
            x = values["global_batch_idx"].to_numpy()
            color = COLORS[method]
            axis.plot(
                x,
                values[f"{metric}_mean"],
                color=color,
                linewidth=1.5,
                marker="o",
                markersize=2.5,
            )
            loss_axis.plot(
                x,
                values["loss_mean"],
                color=color,
                linewidth=1.15,
                linestyle="--",
                alpha=0.72,
            )

        for boundary in boundaries:
            axis.axvline(boundary, color="#b8bcc2", linewidth=0.65, alpha=0.65)
        axis.set_ylabel(ylabel)
        loss_axis.set_ylabel("Training loss", color="#5c6066")
        loss_axis.tick_params(axis="y", labelcolor="#5c6066")
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
        axis.grid(axis="y", color="#d8dadd", linewidth=0.55, alpha=0.8)
        axis.margins(x=0.01)

    axes[-1].set_xlabel("Global batch index (continuous across epochs)")
    axes[-1].set_xlim(0.5, float(data["global_batch_idx"].max()) + 0.5)
    for position, label in epoch_labels:
        axes[0].text(
            position,
            1.015,
            label,
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=7,
            color="#5c6066",
        )

    legend = [
        Line2D([0], [0], color=COLORS[method], linewidth=1.8, label=LABELS[method])
        for method in METHODS
    ]
    legend.extend(
        [
            Line2D([0], [0], color="#333333", linewidth=1.6, label="Forward metric"),
            Line2D(
                [0], [0], color="#333333", linewidth=1.3,
                linestyle="--", label="Batch training loss",
            ),
        ]
    )
    figure.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.947),
        ncol=4,
        frameon=False,
    )
    device_count = data["device_id"].nunique()
    figure.suptitle(
        "Clean vs availability-shortcut training: forward L1D behavior and loss\n"
        f"Small TrashNet, IID, SimpleCNN, batch size 16; mean across {device_count} device(s)\n"
        f"perf-scaled counts; complete batches require every event interval >= "
        f"{minimum_running_pct:g}% running",
        y=0.992,
        fontsize=13,
    )
    figure.tight_layout(rect=(0.02, 0.02, 0.98, 0.89), h_pad=1.4)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    runs = discover_runs(input_dir, set(args.device or ()))

    frames = []
    selected_rows = []
    for (device, method), run in sorted(runs.items()):
        frame = load_batch_data(run, args.minimum_running_pct)
        frames.append(frame)
        selected_rows.append(
            {
                "device_id": device,
                "method": method,
                "trial_id": run["trial_id"],
                "perf_path": run["perf_path"],
                "metrics_path": run["metrics_path"],
                "joined_batches": len(frame),
                "reliable_instruction_batches": int(frame["instructions_reliable"].sum()),
                "reliable_l1d_batches": int(frame["l1d_accesses_reliable"].sum()),
                "jointly_reliable_batches": int(frame["jointly_reliable"].sum()),
                "minimum_running_pct": args.minimum_running_pct,
            }
        )

    data = pd.concat(frames, ignore_index=True)
    summary = summarize(data)
    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / "forward_batch_cache_vs_loss.csv", index=False)
    summary.to_csv(output_dir / "forward_batch_cache_vs_loss_summary.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(output_dir / "selected_runs.csv", index=False)
    output_path = output_dir / "forward_batch_cache_vs_loss.png"
    plot(data, summary, output_path, args.dpi, args.minimum_running_pct)

    batches_per_epoch = (
        data.groupby(["device_id", "method", "epoch"])["batch_idx"]
        .nunique()
        .astype(int)
    )
    print(f"Saved {output_path}")
    print(
        f"Paired devices: {', '.join(sorted(data['device_id'].unique()))}; "
        f"batches per epoch: {sorted(batches_per_epoch.unique())}"
    )


if __name__ == "__main__":
    main()
