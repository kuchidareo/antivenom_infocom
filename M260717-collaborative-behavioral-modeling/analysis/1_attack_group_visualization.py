#!/usr/bin/env python3
"""Visualize clean versus attacked local-ML perf behavior across epochs."""

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
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "collected_logs" / "test"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "visualization" / "test"
PHASES = ("forward", "backward")
CONDITION_COLORS = {
    "clean": "#2468a2",
    "availability_shortcuts": "#c43c39",
}
CONDITION_LABELS = {
    "clean": "Clean",
    "availability_shortcuts": "Availability shortcuts",
}
DATASET_LABELS = {
    "kuchidareo/small_trashnet": "Small TrashNet",
    "kuchidareo/chinese_trafficsign_dataset": "Chinese traffic signs",
    "uoft-cs/cifar10": "CIFAR-10",
}


@dataclass(frozen=True)
class Run:
    perf_path: Path
    device: str
    client: str
    dataset: str
    partition_method: str
    condition: str
    model: str
    batch_size: int
    background_enabled: bool
    background_group: str
    background_profile: str
    start_time: float

    @property
    def scenario_key(self) -> tuple[str, str, str, str, int, str, str]:
        background_group = self.background_group if self.background_enabled else "none"
        background_profile = self.background_profile if self.background_enabled else "none"
        return (
            self.device,
            self.dataset,
            self.partition_method,
            self.model,
            self.batch_size,
            background_group,
            background_profile,
        )


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    numerator: str
    denominator: str | None = None
    scale: float = 1.0


METRICS = (
    Metric(
        "instructions_billion",
        "Instructions (billions)",
        "perf_instructions",
        scale=1e-9,
    ),
    Metric(
        "task_clock_seconds",
        "Task clock (s)",
        "perf_task_clock",
        scale=1e-3,
    ),
    Metric(
        "cycles_per_instruction",
        "Cycles / instruction",
        "perf_cycles",
        "perf_instructions",
    ),
    Metric(
        "context_switches_per_million_instructions",
        "Context switches / 1M instructions",
        "perf_context_switches",
        "perf_instructions",
        1e6,
    ),
    Metric(
        "cpu_migrations_per_million_instructions",
        "CPU migrations / 1M instructions",
        "perf_cpu_migrations",
        "perf_instructions",
        1e6,
    ),
    Metric(
        "page_faults_per_million_instructions",
        "Page faults / 1M instructions",
        "perf_page_faults",
        "perf_instructions",
        1e6,
    ),
    Metric(
        "l1_loads_per_instruction",
        "L1D loads / instruction",
        "perf_l1_dcache_loads",
        "perf_instructions",
    ),
    Metric(
        "l1_load_misses_per_instruction",
        "L1D load misses / instruction",
        "perf_l1_dcache_load_misses",
        "perf_instructions",
    ),
    Metric(
        "l1_load_miss_percent",
        "L1D load miss rate (%)",
        "perf_l1_dcache_load_misses",
        "perf_l1_dcache_loads",
        100.0,
    ),
    Metric(
        "l2_demand_references_per_instruction",
        "L2 demand references / instruction",
        "perf_l2_rqsts_all_demand_references",
        "perf_instructions",
    ),
    Metric(
        "l2_demand_misses_per_instruction",
        "L2 demand misses / instruction",
        "perf_l2_rqsts_all_demand_miss",
        "perf_instructions",
    ),
    Metric(
        "l2_demand_miss_percent",
        "L2 demand miss rate (%)",
        "perf_l2_rqsts_all_demand_miss",
        "perf_l2_rqsts_all_demand_references",
        100.0,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--attack",
        default="availability_shortcuts",
        help="Poisoning method to compare with clean (default: %(default)s).",
    )
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("_") or "value"


def read_run(perf_path: Path) -> Run:
    with perf_path.open(newline="") as handle:
        row = next(csv.DictReader(handle), None)
    if row is None:
        raise ValueError(f"Empty perf CSV: {perf_path}")
    return Run(
        perf_path=perf_path,
        device=row["device_id"],
        client=row["client_id"],
        dataset=row["dataset"],
        partition_method=row["partition_method"],
        condition=row["poisoning_method"],
        model=row["model"],
        batch_size=int(row["batch_size"]),
        background_enabled=row["background_workload_enabled"].strip().lower() == "true",
        background_group=row["background_workload_group"],
        background_profile=row["background_workload_profile"],
        start_time=float(row["timestamp_unix"]),
    )


def discover_runs(input_dir: Path, attack: str) -> dict[tuple, dict[str, Run]]:
    perf_paths = sorted(input_dir.rglob("*_perf.csv"))
    if not perf_paths:
        raise FileNotFoundError(f"No *_perf.csv files found below {input_dir}")

    candidates: dict[tuple, dict[str, list[Run]]] = {}
    for perf_path in perf_paths:
        run = read_run(perf_path)
        if run.condition not in {"clean", attack}:
            continue
        candidates.setdefault(run.scenario_key, {}).setdefault(run.condition, []).append(run)

    selected: dict[tuple, dict[str, Run]] = {}
    for scenario, by_condition in candidates.items():
        missing = {"clean", attack} - set(by_condition)
        if missing:
            warnings.warn(f"Skipping {scenario}: missing conditions {sorted(missing)}")
            continue

        selected[scenario] = {}
        for condition, runs in by_condition.items():
            runs.sort(key=lambda item: item.start_time)
            selected[scenario][condition] = runs[-1]
            if len(runs) > 1:
                warnings.warn(
                    f"{scenario}/{condition}: using latest {runs[-1].perf_path.name} "
                    f"and ignoring {len(runs) - 1} earlier run(s)."
                )

        clients = {run.client for run in selected[scenario].values()}
        if len(clients) != 1:
            warnings.warn(
                f"{scenario}: clean and attack use different partitions {sorted(clients)}."
            )

    if not selected:
        raise ValueError(f"No complete clean/{attack} scenario pairs found in {input_dir}")
    return dict(sorted(selected.items()))


def available_metrics(runs: dict[str, Run]) -> tuple[Metric, ...]:
    common_columns = set.intersection(
        *(set(pd.read_csv(run.perf_path, nrows=0).columns) for run in runs.values())
    )
    return tuple(
        metric
        for metric in METRICS
        if metric.numerator in common_columns
        and (metric.denominator is None or metric.denominator in common_columns)
    )


def required_columns(metrics: tuple[Metric, ...], header: set[str]) -> list[str]:
    result = ["epoch", "phase", "perf_status"]
    event_columns: list[str] = []
    for metric in metrics:
        result.append(metric.numerator)
        event_columns.append(metric.numerator)
        if metric.denominator:
            result.append(metric.denominator)
            event_columns.append(metric.denominator)
    for event_column in event_columns:
        enabled_column = f"{event_column}_enabled_pct"
        if enabled_column in header:
            result.append(enabled_column)
    return list(dict.fromkeys(result))


def aggregate_run(
    run: Run,
    metrics: tuple[Metric, ...],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    header = set(pd.read_csv(run.perf_path, nrows=0).columns)
    columns = required_columns(metrics, header)
    missing = [column for column in columns if column not in header]
    if missing:
        raise ValueError(f"{run.perf_path} is missing columns: {missing}")

    frame = pd.read_csv(run.perf_path, usecols=columns, low_memory=False)
    frame = frame.loc[frame["perf_status"].eq("ok")].copy()
    frame["epoch"] = pd.to_numeric(frame["epoch"], errors="coerce")
    numeric_columns = [column for column in columns if column not in {"phase", "perf_status"}]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")

    summaries: dict[str, pd.DataFrame] = {}
    quality_rows: list[dict[str, float | int | str]] = []
    enabled_columns = [column for column in columns if column.endswith("_enabled_pct")]
    for phase in PHASES:
        selected = frame.loc[frame["phase"].eq(phase) & frame["epoch"].notna()].copy()
        if selected.empty:
            raise ValueError(f"No {phase} samples in {run.perf_path}")
        selected["epoch"] = selected["epoch"].astype(int)
        epochs = sorted(selected["epoch"].unique())
        result = pd.DataFrame(index=epochs, dtype=float)
        result.index.name = "epoch"

        for metric in metrics:
            numerator = selected.groupby("epoch")[metric.numerator].sum(min_count=1)
            if metric.denominator is None:
                result[metric.key] = numerator * metric.scale
            else:
                denominator = selected.groupby("epoch")[metric.denominator].sum(min_count=1)
                result[metric.key] = numerator.div(denominator.where(denominator > 0)) * metric.scale

        for epoch, epoch_frame in selected.groupby("epoch"):
            enabled = epoch_frame[enabled_columns].stack().dropna()
            quality_rows.append(
                {
                    "phase": phase,
                    "epoch": int(epoch),
                    "sample_count": len(epoch_frame),
                    "enabled_pct_mean": float(enabled.mean()) if not enabled.empty else float("nan"),
                    "enabled_pct_min": float(enabled.min()) if not enabled.empty else float("nan"),
                }
            )
        summaries[phase] = result

    return summaries, pd.DataFrame(quality_rows)


def plot_scenario(
    scenario: tuple,
    runs: dict[str, Run],
    summaries: dict[str, dict[str, pd.DataFrame]],
    quality: dict[str, pd.DataFrame],
    metrics: tuple[Metric, ...],
    output_dir: Path,
    attack: str,
    dpi: int,
) -> Path:
    device, dataset, partition, model, batch_size, background_group, background_profile = scenario
    figure, axes = plt.subplots(
        nrows=len(metrics),
        ncols=len(PHASES),
        figsize=(18, 2.65 * len(metrics)),
        squeeze=False,
    )

    for row_index, metric in enumerate(metrics):
        for column_index, phase in enumerate(PHASES):
            axis = axes[row_index, column_index]
            for condition in ("clean", attack):
                values = summaries[condition][phase][metric.key]
                axis.plot(
                    values.index,
                    values,
                    color=CONDITION_COLORS.get(condition, "#555555"),
                    linewidth=1.7,
                    marker="o",
                    markersize=3.2,
                    label=CONDITION_LABELS.get(condition, condition),
                )
            axis.set_ylabel(metric.label, fontsize=8)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.tick_params(axis="both", labelsize=7)
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
            axis.grid(True, color="#d8dadd", linewidth=0.55, alpha=0.8)
            axis.margins(x=0.03, y=0.08)
            if row_index == 0:
                axis.set_title(phase.capitalize(), fontsize=11, pad=7)
            if row_index == len(metrics) - 1:
                axis.set_xlabel("Local training epoch", fontsize=9)

    handles = [
        Line2D(
            [0],
            [0],
            color=CONDITION_COLORS.get(condition, "#555555"),
            linewidth=1.7,
            marker="o",
            markersize=4,
            label=CONDITION_LABELS.get(condition, condition),
        )
        for condition in ("clean", attack)
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.986),
        ncol=2,
        fontsize=9,
        frameon=False,
    )

    quality_text = []
    for condition in ("clean", attack):
        minimum = quality[condition]["enabled_pct_min"].min()
        quality_text.append(f"{CONDITION_LABELS.get(condition, condition)} min enabled={minimum:.1f}%")
    title_dataset = DATASET_LABELS.get(dataset, dataset)
    figure.suptitle(
        f"{device} | {title_dataset} | {partition} | {model}, batch {batch_size} | "
        f"bg={background_group}/{background_profile}\n"
        + " | ".join(quality_text),
        fontsize=14,
        y=0.999,
    )
    figure.tight_layout(rect=(0.025, 0.01, 0.995, 0.953), h_pad=1.0, w_pad=1.2)

    scenario_dir = (
        output_dir
        / safe_name(device)
        / safe_name(partition)
        / f"model_{safe_name(model)}"
        / f"batch_{batch_size}"
        / f"bg_{safe_name(background_group)}_{safe_name(background_profile)}"
    )
    scenario_dir.mkdir(parents=True, exist_ok=True)
    path = scenario_dir / f"{safe_name(dataset)}_clean_vs_{safe_name(attack)}.png"
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    scenarios = discover_runs(input_dir, args.attack)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(scenarios)} complete clean/{args.attack} scenarios in {input_dir}")
    summary_records: list[dict[str, object]] = []
    quality_records: list[dict[str, object]] = []

    for scenario, runs in scenarios.items():
        scenario_metrics = available_metrics(runs)
        if not scenario_metrics:
            warnings.warn(f"Skipping {scenario}: no common perf metrics are available.")
            continue
        unavailable = [metric.key for metric in METRICS if metric not in scenario_metrics]
        if unavailable:
            warnings.warn(
                f"{scenario}: omitting unavailable metrics: {', '.join(unavailable)}"
            )
        run_summaries: dict[str, dict[str, pd.DataFrame]] = {}
        run_quality: dict[str, pd.DataFrame] = {}
        for condition, run in runs.items():
            run_summaries[condition], run_quality[condition] = aggregate_run(
                run, scenario_metrics
            )
            for phase, frame in run_summaries[condition].items():
                for epoch, row in frame.iterrows():
                    for metric in scenario_metrics:
                        summary_records.append(
                            {
                                "device": run.device,
                                "dataset": run.dataset,
                                "partition_method": run.partition_method,
                                "client_id": run.client,
                                "model": run.model,
                                "batch_size": run.batch_size,
                                "condition": condition,
                                "phase": phase,
                                "epoch": int(epoch),
                                "metric": metric.key,
                                "value": row[metric.key],
                                "source_file": str(run.perf_path),
                            }
                        )
            for record in run_quality[condition].to_dict("records"):
                quality_records.append(
                    {
                        "device": run.device,
                        "dataset": run.dataset,
                        "partition_method": run.partition_method,
                        "client_id": run.client,
                        "condition": condition,
                        "source_file": str(run.perf_path),
                        **record,
                    }
                )

        path = plot_scenario(
            scenario,
            runs,
            run_summaries,
            run_quality,
            scenario_metrics,
            output_dir,
            args.attack,
            args.dpi,
        )
        print(f"Saved {path}", flush=True)

    pd.DataFrame(summary_records).to_csv(output_dir / "epoch_perf_summary.csv", index=False)
    pd.DataFrame(quality_records).to_csv(output_dir / "perf_sampling_quality.csv", index=False)
    print(f"Saved {output_dir / 'epoch_perf_summary.csv'}")
    print(f"Saved {output_dir / 'perf_sampling_quality.csv'}")


if __name__ == "__main__":
    main()
