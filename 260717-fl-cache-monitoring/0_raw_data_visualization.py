#!/usr/bin/env python3
"""Plot per-round FL perf counters normalized by retired instructions."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path

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
POISONED_COUNTS = (1, 4, 7, 10)

METHOD_ORDER = {
    "clean": 0,
    "unlearnable_examples": 1,
    "availability_shortcuts": 2,
    "random_label_flipping": 3,
}
METHOD_LABELS = {
    "clean": "Clean",
    "unlearnable_examples": "Unlearnable",
    "availability_shortcuts": "Shortcut",
    "random_label_flipping": "Label flip",
}
METHOD_COLORS = {
    "clean": "#202124",
    "unlearnable_examples": "#c23b4a",
    "availability_shortcuts": "#007f83",
    "random_label_flipping": "#386cb0",
}
COUNT_STYLES = {
    0: ("-", "o"),
    1: (":", "o"),
    4: ("--", "s"),
    7: ("-.", "^"),
    10: ("-", "D"),
}


@dataclass(frozen=True)
class Metric:
    column: str
    label: str


# perf_branches is intentionally absent: it is empty throughout this collection.
METRICS = (
    Metric("perf_cycles", "CPU cycles / instruction"),
    Metric("perf_task_clock", "Task clock (ms) / instruction"),
    Metric("perf_context_switches", "Context switches / instruction"),
    Metric("perf_cpu_migrations", "CPU migrations / instruction"),
    Metric("perf_page_faults", "Page faults / instruction"),
    Metric("perf_branch_misses", "Branch misses / instruction"),
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
class Run:
    device: str
    client: str
    perf_path: Path
    start_time: float
    poisoned_count: int
    locally_poisoned: bool
    local_method: str
    global_method: str = ""

    @property
    def condition_key(self) -> tuple[int, int]:
        return METHOD_ORDER[self.global_method], self.poisoned_count

    @property
    def label(self) -> str:
        if self.global_method == "clean":
            return "Clean baseline"
        status = "local poisoned" if self.locally_poisoned else "local clean"
        return (
            f"{METHOD_LABELS[self.global_method]}, "
            f"{self.poisoned_count} poisoned ({status})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=140)
    return parser.parse_args()


def read_run_metadata(perf_path: Path) -> Run:
    main_path = perf_path.with_name(perf_path.name.removesuffix("_perf.csv") + ".csv")
    if not main_path.is_file():
        raise FileNotFoundError(f"Missing main log for {perf_path}: {main_path}")

    with main_path.open(newline="") as handle:
        row = next(csv.DictReader(handle))

    return Run(
        device=row["device_id"],
        client=row["client_id"],
        perf_path=perf_path,
        start_time=float(row["timestamp_unix"]),
        poisoned_count=int(row["poisoned_client_count"]),
        locally_poisoned=row["is_poisoned_client"].strip().lower() == "true",
        local_method=row["poisoning_method"],
    )


def temporal_clusters(runs: list[Run], maximum_gap_sec: float = 60.0) -> list[list[Run]]:
    clusters: list[list[Run]] = []
    for run in sorted(runs, key=lambda item: item.start_time):
        if not clusters or run.start_time - clusters[-1][-1].start_time > maximum_gap_sec:
            clusters.append([run])
        else:
            clusters[-1].append(run)
    return clusters


def discover_runs(input_dir: Path) -> dict[str, list[Run]]:
    perf_paths = sorted(input_dir.glob("192.168.0.*/fl/*_perf.csv"))
    if not perf_paths:
        raise FileNotFoundError(f"No client perf CSVs found below {input_dir}")

    raw_runs = [read_run_metadata(path) for path in perf_paths]
    resolved: list[Run] = []
    for cluster in temporal_clusters(raw_runs):
        devices = {run.device for run in cluster}
        counts = {run.poisoned_count for run in cluster}
        methods = {run.local_method for run in cluster if run.local_method != "clean"}
        if len(cluster) != 10 or len(devices) != 10:
            raise ValueError(
                "Expected ten synchronized client runs; found "
                f"{len(cluster)} records and {len(devices)} devices near "
                f"timestamp {cluster[0].start_time}"
            )
        if len(counts) != 1:
            raise ValueError(f"Inconsistent poisoned-client counts in run cluster: {counts}")

        poisoned_count = next(iter(counts))
        if poisoned_count == 0:
            global_method = "clean"
        elif len(methods) == 1:
            global_method = next(iter(methods))
        else:
            raise ValueError(
                f"Could not identify one attack method near {cluster[0].start_time}: {methods}"
            )
        if global_method not in METHOD_ORDER:
            raise ValueError(f"Unsupported poisoning method: {global_method}")
        resolved.extend(replace(run, global_method=global_method) for run in cluster)

    by_device: dict[str, list[Run]] = {}
    for run in resolved:
        by_device.setdefault(run.device, []).append(run)

    expected_conditions = {
        (METHOD_ORDER["clean"], 0),
        *{
            (METHOD_ORDER[method], count)
            for method in METHOD_ORDER
            if method != "clean"
            for count in (1, 4, 7, 10)
        },
    }
    for device, device_runs in by_device.items():
        actual = {run.condition_key for run in device_runs}
        if len(device_runs) != 13 or actual != expected_conditions:
            raise ValueError(
                f"{device} does not have the expected 13 unique conditions: "
                f"found {len(device_runs)} runs and {len(actual)} conditions"
            )
        device_runs.sort(key=lambda run: run.condition_key)
    return dict(sorted(by_device.items()))


def aggregate_run(run: Run) -> dict[str, pd.DataFrame]:
    columns = ["round", "phase", "perf_instructions", *(m.column for m in METRICS)]
    frame = pd.read_csv(run.perf_path, usecols=columns, low_memory=False)
    frame["round"] = pd.to_numeric(frame["round"], errors="coerce")
    frame["perf_instructions"] = pd.to_numeric(
        frame["perf_instructions"], errors="coerce"
    )

    summaries: dict[str, pd.DataFrame] = {}
    for phase in PHASES:
        selected = frame.loc[
            frame["phase"].eq(phase) & frame["round"].between(1, 15), columns
        ].copy()
        if selected.empty:
            raise ValueError(f"No {phase} samples in {run.perf_path}")

        result = pd.DataFrame(index=range(1, 16), dtype=float)
        for metric in METRICS:
            values = pd.to_numeric(selected[metric.column], errors="coerce")
            valid = values.notna() & selected["perf_instructions"].gt(0)
            numerator = values[valid].groupby(selected.loc[valid, "round"]).sum()
            denominator = (
                selected.loc[valid, "perf_instructions"]
                .groupby(selected.loc[valid, "round"])
                .sum()
            )
            result[metric.column] = numerator / denominator
        result.index.name = "round"
        summaries[phase] = result
    return summaries


def line_style(run: Run) -> dict[str, object]:
    linestyle, marker = COUNT_STYLES[run.poisoned_count]
    local_attack = run.locally_poisoned or run.global_method == "clean"
    return {
        "color": METHOD_COLORS[run.global_method],
        "linestyle": linestyle,
        "marker": marker,
        "markersize": 3.2,
        "markerfacecolor": (
            METHOD_COLORS[run.global_method] if local_attack else "white"
        ),
        "markeredgewidth": 0.8,
        "linewidth": 1.45 if local_attack else 1.05,
        "alpha": 0.95 if local_attack else 0.68,
    }


def save_phase_figure(
    device: str,
    runs: list[Run],
    summaries: dict[Path, dict[str, pd.DataFrame]],
    phase: str,
    poisoned_count: int,
    output_dir: Path,
    dpi: int,
) -> Path:
    scenario_runs = [
        run
        for run in runs
        if run.global_method == "clean" or run.poisoned_count == poisoned_count
    ]
    available_metrics = [
        metric
        for metric in METRICS
        if any(
            summaries[run.perf_path][phase][metric.column].notna().any()
            for run in scenario_runs
        )
    ]
    figure, axes = plt.subplots(
        nrows=len(available_metrics),
        ncols=1,
        figsize=(14, max(3.0 * len(available_metrics), 8)),
        squeeze=False,
    )

    for axis, metric in zip(axes[:, 0], available_metrics):
        for run in scenario_runs:
            values = summaries[run.perf_path][phase][metric.column]
            axis.plot(values.index, values, **line_style(run))
        axis.set_ylabel(metric.label, fontsize=8)
        axis.set_xlim(1, 15)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=15))
        axis.tick_params(axis="both", labelsize=7)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        axis.grid(True, color="#d8dadd", linewidth=0.55, alpha=0.8)
        axis.margins(y=0.08)

    axes[-1, 0].set_xlabel("Federated learning round", fontsize=9)
    handles = [
        Line2D([0], [0], label=run.label, **line_style(run))
        for run in scenario_runs
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.988),
        ncol=4,
        fontsize=8,
        frameon=False,
        handlelength=3.2,
    )
    figure.suptitle(
        f"{device}: {phase.capitalize()} perf metrics per instruction "
        f"({poisoned_count} poisoned clients)",
        fontsize=15,
        y=0.999,
    )
    figure.tight_layout(rect=(0.04, 0.01, 0.995, 0.955), h_pad=1.1)

    device_dir = output_dir / f"poisoned_{poisoned_count}_clients" / device
    device_dir.mkdir(parents=True, exist_ok=True)
    path = device_dir / f"metrics_per_instruction_{phase}.png"
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    runs_by_device = discover_runs(input_dir)

    print(f"Found {sum(map(len, runs_by_device.values()))} runs for {len(runs_by_device)} clients")
    for device, runs in runs_by_device.items():
        print(f"Aggregating {device} ({len(runs)} conditions) ...", flush=True)
        summaries = {run.perf_path: aggregate_run(run) for run in runs}
        for poisoned_count in POISONED_COUNTS:
            for phase in PHASES:
                path = save_phase_figure(
                    device,
                    runs,
                    summaries,
                    phase,
                    poisoned_count,
                    output_dir,
                    args.dpi,
                )
                print(f"Saved {path}")


if __name__ == "__main__":
    main()
