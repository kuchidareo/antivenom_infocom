#!/usr/bin/env python3
"""Plot TensorFlow local-training perf counters normalized by instructions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "collected_logs"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "visualization"
PHASES = ("forward", "backward")

METHODS = (
    "clean",
    "unlearnable_examples",
    "availability_shortcuts",
    "random_label_flipping",
    "target_label_flipping",
)
METHOD_LABELS = {
    "clean": "Clean",
    "unlearnable_examples": "Unlearnable examples",
    "availability_shortcuts": "Availability shortcuts",
    "random_label_flipping": "Random label flipping",
    "target_label_flipping": "Target label flipping",
}
METHOD_COLORS = {
    "clean": "#202124",
    "unlearnable_examples": "#c23b4a",
    "availability_shortcuts": "#007f83",
    "random_label_flipping": "#386cb0",
    "target_label_flipping": "#a05a00",
}
METHOD_MARKERS = {
    "clean": "o",
    "unlearnable_examples": "s",
    "availability_shortcuts": "^",
    "random_label_flipping": "D",
    "target_label_flipping": "v",
}


@dataclass(frozen=True)
class Metric:
    column: str
    label: str


METRICS = (
    Metric("perf_cycles", "CPU cycles / instruction"),
    Metric("perf_task_clock", "Task clock (ms) / instruction"),
    Metric("perf_context_switches", "Context switches / instruction"),
    Metric("perf_cpu_migrations", "CPU migrations / instruction"),
    Metric("perf_page_faults", "Page faults / instruction"),
    Metric("perf_br_retired", "Retired branches / instruction"),
    Metric("perf_br_mis_pred_retired", "Mispredicted branches / instruction"),
    Metric("perf_l1d_cache", "L1D accesses / instruction"),
    Metric("perf_l1d_cache_refill", "L1D refills / instruction"),
    Metric("perf_l1d_cache_wb", "L1D writebacks / instruction"),
    Metric("perf_l2d_cache", "L2D accesses / instruction"),
    Metric("perf_l2d_cache_refill", "L2D refills / instruction"),
    Metric("perf_l2d_cache_wb", "L2D writebacks / instruction"),
    Metric("perf_bus_access", "Bus accesses / instruction"),
    Metric("perf_mem_access", "Memory accesses / instruction"),
    Metric("perf_inst_spec", "Speculative instructions / instruction"),
)


@dataclass(frozen=True)
class Run:
    device: str
    client: str
    method: str
    framework: str
    perf_path: Path
    start_time: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=140)
    return parser.parse_args()


def read_run(perf_path: Path) -> Run:
    metadata = pd.read_csv(
        perf_path,
        nrows=1,
        usecols=[
            "timestamp_unix",
            "framework",
            "device_id",
            "client_id",
            "poisoning_method",
        ],
    ).iloc[0]
    framework = str(metadata["framework"]).strip().lower()
    if framework != "tensorflow":
        raise ValueError(f"Expected TensorFlow log, found {framework!r}: {perf_path}")
    return Run(
        device=str(metadata["device_id"]),
        client=str(metadata["client_id"]),
        method=str(metadata["poisoning_method"]),
        framework=framework,
        perf_path=perf_path,
        start_time=float(metadata["timestamp_unix"]),
    )


def discover_runs(input_dir: Path) -> dict[str, list[Run]]:
    perf_paths = sorted(input_dir.glob("logs_*/local_ml/*_perf.csv"))
    if not perf_paths:
        raise FileNotFoundError(f"No TensorFlow perf CSVs found below {input_dir}")

    candidates: dict[tuple[str, str], list[Run]] = {}
    for path in perf_paths:
        run = read_run(path)
        if run.method not in METHODS:
            warnings.warn(f"Skipping unsupported method {run.method!r}: {path}")
            continue
        candidates.setdefault((run.device, run.method), []).append(run)

    by_device: dict[str, list[Run]] = {}
    for (device, method), method_runs in candidates.items():
        method_runs.sort(key=lambda run: run.start_time)
        selected = method_runs[-1]
        by_device.setdefault(device, []).append(selected)
        if len(method_runs) > 1:
            warnings.warn(
                f"{device}/{method}: using latest {selected.perf_path.name}; "
                f"ignoring {len(method_runs) - 1} older run(s)."
            )

    for device, runs in by_device.items():
        present = {run.method for run in runs}
        missing = [method for method in METHODS if method not in present]
        if missing:
            warnings.warn(f"{device} is missing methods: {', '.join(missing)}")
        runs.sort(key=lambda run: METHODS.index(run.method))
    return dict(sorted(by_device.items()))


def aggregate_run(run: Run) -> dict[str, pd.DataFrame]:
    requested = [
        "epoch",
        "phase",
        "perf_instructions",
        *(metric.column for metric in METRICS),
    ]
    frame = pd.read_csv(run.perf_path, usecols=requested, low_memory=False)
    frame["epoch"] = pd.to_numeric(frame["epoch"], errors="coerce")
    frame["perf_instructions"] = pd.to_numeric(
        frame["perf_instructions"], errors="coerce"
    )
    epochs = sorted(int(epoch) for epoch in frame["epoch"].dropna().unique() if epoch >= 0)
    if not epochs:
        raise ValueError(f"No training epochs found in {run.perf_path}")

    summaries: dict[str, pd.DataFrame] = {}
    for phase in PHASES:
        selected = frame.loc[frame["phase"].eq(phase)].copy()
        if selected.empty:
            raise ValueError(f"No {phase} samples in {run.perf_path}")

        result = pd.DataFrame(index=epochs, dtype=float)
        for metric in METRICS:
            values = pd.to_numeric(selected[metric.column], errors="coerce")
            valid = values.notna() & selected["perf_instructions"].gt(0)
            numerator = values[valid].groupby(selected.loc[valid, "epoch"]).sum()
            denominator = (
                selected.loc[valid, "perf_instructions"]
                .groupby(selected.loc[valid, "epoch"])
                .sum()
            )
            result[metric.column] = numerator / denominator
        result.index.name = "epoch"
        summaries[phase] = result
    return summaries


def line_style(method: str) -> dict[str, object]:
    return {
        "color": METHOD_COLORS[method],
        "marker": METHOD_MARKERS[method],
        "markersize": 3.2,
        "linewidth": 1.45,
        "alpha": 0.95,
    }


def save_device_figure(
    device: str,
    runs: list[Run],
    summaries: dict[Path, dict[str, pd.DataFrame]],
    output_dir: Path,
    dpi: int,
) -> Path:
    available_metrics = [
        metric
        for metric in METRICS
        if any(
            summaries[run.perf_path][phase][metric.column].notna().any()
            for run in runs
            for phase in PHASES
        )
    ]
    if not available_metrics:
        raise ValueError(f"No usable perf metrics for {device}")

    figure, axes = plt.subplots(
        nrows=len(available_metrics),
        ncols=len(PHASES),
        figsize=(18, max(3.0 * len(available_metrics), 8)),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    for row, metric in enumerate(available_metrics):
        for column, phase in enumerate(PHASES):
            axis = axes[row, column]
            for run in runs:
                values = summaries[run.perf_path][phase][metric.column]
                axis.plot(values.index, values, **line_style(run.method))
            axis.set_ylabel(metric.label, fontsize=8)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
            axis.tick_params(axis="both", labelsize=7)
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
            axis.grid(True, color="#d8dadd", linewidth=0.55, alpha=0.8)
            axis.margins(x=0.03, y=0.08)
            if row == 0:
                axis.set_title(phase.capitalize(), fontsize=11, pad=7)
            if row == len(available_metrics) - 1:
                axis.set_xlabel("Local training epoch", fontsize=9)

    handles = [
        Line2D([0], [0], label=METHOD_LABELS[run.method], **line_style(run.method))
        for run in runs
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.987),
        ncol=len(handles),
        fontsize=9,
        frameon=False,
        handlelength=2.8,
    )
    figure.suptitle(
        f"{device}: TensorFlow perf metrics per instruction",
        fontsize=15,
        y=0.999,
    )
    figure.tight_layout(rect=(0.025, 0.01, 0.995, 0.958), h_pad=1.1, w_pad=1.2)

    device_dir = output_dir / device
    device_dir.mkdir(parents=True, exist_ok=True)
    path = device_dir / "tensorflow_perf_metrics_per_instruction.png"
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def main() -> None:
    args = parse_args()
    runs_by_device = discover_runs(args.input_dir.resolve())
    all_runs = [run for runs in runs_by_device.values() for run in runs]
    print(
        f"Aggregating {len(all_runs)} TensorFlow runs across "
        f"{len(runs_by_device)} devices ..."
    )
    summaries = {
        run.perf_path: aggregate_run(run)
        for run in all_runs
    }
    for device, runs in runs_by_device.items():
        path = save_device_figure(
            device,
            runs,
            summaries,
            args.output_dir.resolve(),
            args.dpi,
        )
        print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
