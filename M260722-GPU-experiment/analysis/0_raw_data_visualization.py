#!/usr/bin/env python3
"""Plot GPU-experiment hardware and NCU metric trends by local epoch."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import re
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "collected_logs"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "visualization"
PHASES = ("forward", "backward")

CONDITION_LABELS = {
    "clean": "Clean",
    "availability_shortcuts": "Availability shortcuts",
    "unlearnable_examples": "Unlearnable examples",
    "random_label_flipping": "Random label flipping",
    "badsampling": "BadSampler",
}
CONDITION_COLORS = {
    "clean": "#202124",
    "availability_shortcuts": "#007f83",
    "unlearnable_examples": "#c23b4a",
    "random_label_flipping": "#386cb0",
    "badsampling": "#d27c2c",
}


@dataclass(frozen=True)
class HardwareMetric:
    column: str
    label: str
    kind: str = "gauge"
    scale: float = 1.0


HARDWARE_METRICS = (
    *(HardwareMetric(f"system_cpu_core_{core}", f"System CPU core {core} utilization (%)") for core in range(4)),
    *(HardwareMetric(f"system_cpu_freq_core_{core}", f"CPU frequency core {core} (MHz)") for core in range(4)),
    HardwareMetric("system_memory_percent", "System memory utilization (%)"),
    HardwareMetric("system_memory_used", "System memory used (GiB)", scale=1.0 / 2**30),
    HardwareMetric("system_memory_available", "System memory available (GiB)", scale=1.0 / 2**30),
    HardwareMetric("process_cpu_percent", "Process CPU utilization (%)"),
    HardwareMetric("process_memory_rss", "Process RSS (GiB)", scale=1.0 / 2**30),
    HardwareMetric("process_memory_vms", "Process VMS (GiB)", scale=1.0 / 2**30),
    HardwareMetric("process_memory_percent", "Process memory utilization (%)"),
    HardwareMetric(
        "process_ctx_switches_voluntary",
        "Voluntary context-switch increments",
        kind="counter",
    ),
    HardwareMetric(
        "process_ctx_switches_involuntary",
        "Involuntary context-switch increments",
        kind="counter",
    ),
    HardwareMetric("process_minor_faults", "Minor-fault increments", kind="counter"),
)

NCU_LABELS = {
    "gpu__time_duration.sum": "GPU duration",
    "sm__cycles_elapsed.sum": "SM cycles elapsed",
    "smsp__inst_executed.sum": "SMSP instructions executed",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum": "L1/TEX global-load requests",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": "L1/TEX global-load sectors",
    "lts__t_sectors_op_read.sum": "L2 read sectors",
    "lts__t_sectors_op_write.sum": "L2 write sectors",
    "dram__bytes_read.sum": "DRAM bytes read",
    "dram__bytes_write.sum": "DRAM bytes written",
}


@dataclass(frozen=True)
class Run:
    path: Path
    condition: str
    device: str
    trial: str
    start_time: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--conditions",
        nargs="+",
        help="Conditions to plot (default: every discovered condition).",
    )
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def read_first_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        row = next(csv.DictReader(handle), None)
    if row is None:
        raise ValueError(f"Empty CSV: {path}")
    return row


def is_hardware_log(path: Path) -> bool:
    if path.name.endswith(("_metrics.csv", "_perf.csv")):
        return False
    columns = set(pd.read_csv(path, nrows=0).columns)
    return {"epoch", "phase", "poisoning_method", "timestamp_unix"}.issubset(columns) and any(
        metric.column in columns for metric in HARDWARE_METRICS
    )


def discover_hardware_runs(input_dir: Path) -> list[Run]:
    candidates: dict[tuple[str, str, str], list[Run]] = {}
    for path in sorted(input_dir.rglob("*.csv")):
        if not is_hardware_log(path):
            continue
        row = read_first_row(path)
        run = Run(
            path=path,
            condition=row.get("poisoning_method") or "clean",
            device=row.get("device_id") or "unknown_device",
            trial=row.get("trial_id") or "",
            start_time=float(row.get("timestamp_unix") or 0.0),
        )
        candidates.setdefault((run.device, run.condition, run.trial), []).append(run)
    return sorted(
        (max(runs, key=lambda run: run.start_time) for runs in candidates.values()),
        key=lambda run: (run.device, run.condition, run.trial),
    )


def available_hardware_metrics(runs: list[Run]) -> tuple[HardwareMetric, ...]:
    available = []
    for metric in HARDWARE_METRICS:
        if all(
            metric.column in pd.read_csv(run.path, nrows=0).columns
            and pd.read_csv(run.path, usecols=[metric.column])[metric.column].notna().any()
            for run in runs
        ):
            available.append(metric)
    return tuple(available)


def aggregate_hardware_run(
    run: Run, metrics: tuple[HardwareMetric, ...]
) -> pd.DataFrame:
    columns = ["timestamp_unix", "epoch", "phase", *(metric.column for metric in metrics)]
    frame = pd.read_csv(run.path, usecols=columns, low_memory=False)
    frame["epoch"] = pd.to_numeric(frame["epoch"], errors="coerce")
    frame["timestamp_unix"] = pd.to_numeric(frame["timestamp_unix"], errors="coerce")
    for metric in metrics:
        frame[metric.column] = pd.to_numeric(frame[metric.column], errors="coerce")
        if metric.kind == "counter":
            values = frame.sort_values("timestamp_unix")[metric.column]
            increments = values.diff()
            frame[f"{metric.column}__value"] = increments.where(increments.ge(0)) * metric.scale
        else:
            frame[f"{metric.column}__value"] = frame[metric.column] * metric.scale

    rows = []
    for (epoch, phase), selected in frame.loc[
        frame["epoch"].notna() & frame["phase"].isin(PHASES)
    ].groupby(["epoch", "phase"]):
        for metric in metrics:
            values = selected[f"{metric.column}__value"].dropna()
            if values.empty:
                continue
            value = values.sum() if metric.kind == "counter" else values.mean()
            rows.append(
                {
                    "device_id": run.device,
                    "trial_id": run.trial,
                    "condition": run.condition,
                    "phase": phase,
                    "epoch": int(epoch),
                    "metric": metric.column,
                    "metric_label": metric.label,
                    "unit": "count" if metric.kind == "counter" else "",
                    "aggregation": "positive increment sum" if metric.kind == "counter" else "sample mean",
                    "value": value,
                    "sample_count": len(values),
                    "source_file": str(run.path),
                }
            )
    return pd.DataFrame(rows)


def condition_style(condition: str) -> dict[str, object]:
    return {
        "color": CONDITION_COLORS.get(condition, "#7a5195"),
        "label": CONDITION_LABELS.get(condition, condition.replace("_", " ")),
        "linewidth": 1.7,
        "marker": "o",
        "markersize": 3.5,
    }


def save_metric_figure(
    summary: pd.DataFrame,
    phase: str,
    path: Path,
    title: str,
    dpi: int,
) -> None:
    selected = summary.loc[summary["phase"].eq(phase)].copy()
    metrics = selected[["metric", "metric_label"]].drop_duplicates().itertuples(index=False)
    metrics = list(metrics)
    if not metrics:
        warnings.warn(f"No {phase} metrics available for {path}")
        return
    figure, axes = plt.subplots(
        len(metrics), 1, figsize=(13, max(2.65 * len(metrics), 8)),
        sharex=True, squeeze=False,
    )
    for axis, metric_row in zip(axes[:, 0], metrics):
        metric_frame = selected.loc[selected["metric"].eq(metric_row.metric)]
        for condition, values in metric_frame.groupby("condition", sort=False):
            epoch_values = values.groupby("epoch")["value"].mean().sort_index()
            axis.plot(epoch_values.index, epoch_values.values, **condition_style(condition))
        axis.set_ylabel(metric_row.metric_label, fontsize=8)
        axis.tick_params(axis="both", labelsize=7)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(True, color="#d8dadd", linewidth=0.55, alpha=0.8)
        axis.margins(x=0.02, y=0.1)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.986), ncol=4, frameon=False)
    axes[-1, 0].set_xlabel("Local training epoch")
    figure.suptitle(f"{title}: {phase.capitalize()}", fontsize=14, y=0.999)
    figure.tight_layout(rect=(0.025, 0.01, 0.995, 0.955), h_pad=0.9)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def parse_numeric(value: object) -> float:
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"nan", "none", "n/a"}:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def infer_epoch(path: Path, frame: pd.DataFrame) -> pd.Series | None:
    if "epoch" in frame.columns:
        epochs = pd.to_numeric(frame["epoch"], errors="coerce")
        if epochs.notna().any():
            return epochs
    matches = re.findall(r"(?:^|[_-])epoch[_-]?(\d+)(?:[_-]|$)", path.stem, re.IGNORECASE)
    if matches:
        return pd.Series(float(matches[-1]), index=frame.index)
    if "report" in frame.columns:
        extracted = frame["report"].astype(str).str.extract(
            r"(?:^|[_-])epoch[_-]?(\d+)(?:[_-]|$)", expand=False
        )
        epochs = pd.to_numeric(extracted, errors="coerce")
        if epochs.notna().any():
            return epochs
    return None


def discover_ncu_summary(input_dir: Path) -> tuple[pd.DataFrame, list[Path]]:
    records = []
    missing_epoch = []
    required = {"condition", "phase", "metric", "value"}
    for path in sorted(input_dir.rglob("*.csv")):
        header = set(pd.read_csv(path, nrows=0).columns)
        if not required.issubset(header):
            continue
        frame = pd.read_csv(path, low_memory=False)
        epochs = infer_epoch(path, frame)
        if epochs is None:
            missing_epoch.append(path)
            continue
        frame["epoch"] = epochs
        frame["value_numeric"] = frame["value"].map(parse_numeric)
        frame = frame.loc[
            frame["epoch"].notna()
            & frame["value_numeric"].notna()
            & frame["phase"].isin(PHASES)
        ].copy()
        for (condition, phase, epoch, metric, unit), selected in frame.groupby(
            ["condition", "phase", "epoch", "metric", "unit"], dropna=False
        ):
            records.append(
                {
                    "device_id": "ncu_replay",
                    "trial_id": "",
                    "condition": condition,
                    "phase": phase,
                    "epoch": int(epoch),
                    "metric": metric,
                    "metric_label": NCU_LABELS.get(metric, metric),
                    "unit": unit,
                    "aggregation": "range-row sum",
                    "value": selected["value_numeric"].sum(),
                    "sample_count": len(selected),
                    "source_file": str(path),
                }
            )
    return pd.DataFrame(records), missing_epoch


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    hardware_runs = discover_hardware_runs(input_dir)
    if args.conditions:
        requested = set(args.conditions)
        hardware_runs = [run for run in hardware_runs if run.condition in requested]
    if not hardware_runs:
        raise FileNotFoundError(f"No GPU-experiment hardware logs found below {input_dir}")

    metrics = available_hardware_metrics(hardware_runs)
    hardware_summary = pd.concat(
        [aggregate_hardware_run(run, metrics) for run in hardware_runs],
        ignore_index=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    hardware_summary.to_csv(output_dir / "epoch_hardware_summary.csv", index=False)
    device_label = ", ".join(sorted({run.device for run in hardware_runs}))
    for phase in PHASES:
        save_metric_figure(
            hardware_summary,
            phase,
            output_dir / f"hardware_metrics_by_epoch_{phase}.pdf",
            f"{device_label}: hardware and process metrics by epoch",
            args.dpi,
        )

    ncu_summary, missing_epoch = discover_ncu_summary(input_dir)
    if args.conditions and not ncu_summary.empty:
        ncu_summary = ncu_summary.loc[ncu_summary["condition"].isin(args.conditions)]
    if not ncu_summary.empty:
        ncu_summary.to_csv(output_dir / "epoch_ncu_summary.csv", index=False)
        for phase in PHASES:
            save_metric_figure(
                ncu_summary,
                phase,
                output_dir / f"ncu_metrics_by_epoch_{phase}.pdf",
                "Nsight Compute metrics by epoch",
                args.dpi,
            )
        print(f"Saved {ncu_summary['metric'].nunique()} NCU metric trends")
    else:
        warnings.warn(
            "No epoch-annotated NCU metric CSVs were found; NCU epoch plots were not created."
        )
    if missing_epoch:
        warnings.warn(
            "NCU metric CSVs without an epoch annotation were skipped: "
            + ", ".join(str(path) for path in missing_epoch)
        )
    print(
        f"Saved {len(metrics)} hardware/process metric trends for "
        f"{len(hardware_runs)} runs to {output_dir}"
    )


if __name__ == "__main__":
    main()
