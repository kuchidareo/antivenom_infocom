#!/usr/bin/env python3
"""Compare each local-ML robustness scenario with the clean baseline."""

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
DEFAULT_INPUT_DIR = SCRIPT_DIR / "collected_logs"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "visualization"
PHASES = ("forward", "backward")

BASELINE_DATASET = "kuchidareo/small_trashnet"
BASELINE_MODEL = "simple_cnn"
BASELINE_BATCH_SIZE = 16
BASELINE_AUGMENTATION = "baseline"

GROUP_COLORS = {
    "baseline": "#2468a2",
    "scenario": "#c43c39",
}


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    numerator: str
    denominator: str | None = None
    scale: float = 1.0


METRICS = (
    Metric("instructions_billion", "Instructions (billions)", "perf_instructions", scale=1e-9),
    Metric("task_clock_seconds", "Task clock (s)", "perf_task_clock", scale=1e-3),
    Metric("cycles_per_instruction", "Cycles / instruction", "perf_cycles", "perf_instructions"),
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
        "branch_mispredictions_per_instruction",
        "Branch mispredictions / instruction",
        "perf_br_mis_pred_retired",
        "perf_instructions",
    ),
    Metric("l1d_accesses_per_instruction", "L1D accesses / instruction", "perf_l1d_cache", "perf_instructions"),
    Metric("l1d_refills_per_instruction", "L1D refills / instruction", "perf_l1d_cache_refill", "perf_instructions"),
    Metric("l1d_refill_percent", "L1D refill fraction (%)", "perf_l1d_cache_refill", "perf_l1d_cache", 100.0),
    Metric("l1d_writebacks_per_instruction", "L1D writebacks / instruction", "perf_l1d_cache_wb", "perf_instructions"),
    Metric("l2d_accesses_per_instruction", "L2D accesses / instruction", "perf_l2d_cache", "perf_instructions"),
    Metric("l2d_refills_per_instruction", "L2D refills / instruction", "perf_l2d_cache_refill", "perf_instructions"),
    Metric("l2d_refill_percent", "L2D refill fraction (%)", "perf_l2d_cache_refill", "perf_l2d_cache", 100.0),
    Metric("l2d_writebacks_per_instruction", "L2D writebacks / instruction", "perf_l2d_cache_wb", "perf_instructions"),
    Metric("bus_accesses_per_instruction", "Bus accesses / instruction", "perf_bus_access", "perf_instructions"),
    Metric("memory_accesses_per_instruction", "Memory accesses / instruction", "perf_mem_access", "perf_instructions"),
    Metric("speculative_instructions_per_instruction", "Speculative instructions / instruction", "perf_inst_spec", "perf_instructions"),
)


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
    augmentation_profile: str
    background_enabled: bool
    background_group: str
    background_profile: str
    trial_id: str
    measurement_mode: str
    start_time: float

    @property
    def scenario(self) -> str:
        if self.condition != "clean":
            return f"attack_{self.condition}"
        if self.background_enabled:
            return f"background_{self.background_group}_{self.background_profile}"
        if self.partition_method != "iid":
            return f"partition_{self.partition_method}_{dataset_slug(self.dataset)}"
        if self.dataset != BASELINE_DATASET:
            return f"dataset_{dataset_slug(self.dataset)}"
        if self.batch_size != BASELINE_BATCH_SIZE:
            return f"batch_size_{self.batch_size}"
        if self.model != BASELINE_MODEL:
            return f"model_{self.model}"
        if self.augmentation_profile != BASELINE_AUGMENTATION:
            return f"augmentation_{self.augmentation_profile}"
        return "baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("_") or "value"


def dataset_slug(dataset: str) -> str:
    return safe_name(dataset.rsplit("/", 1)[-1])


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def read_run(perf_path: Path) -> Run:
    with perf_path.open(newline="") as handle:
        row = next(csv.DictReader(handle), None)
    if row is None:
        raise ValueError(f"Empty perf CSV: {perf_path}")
    if row.get("run_type") not in {"", None, "local_ml"}:
        raise ValueError(f"Not a local_ml perf CSV: {perf_path}")
    return Run(
        perf_path=perf_path,
        device=row.get("device_id") or perf_path.parent.parent.name,
        client=row.get("client_id", ""),
        dataset=row.get("dataset", ""),
        partition_method=row.get("partition_method", "iid"),
        condition=row.get("poisoning_method", "clean"),
        model=row.get("model", BASELINE_MODEL),
        batch_size=int(row.get("batch_size") or BASELINE_BATCH_SIZE),
        augmentation_profile=row.get("augmentation_profile") or BASELINE_AUGMENTATION,
        background_enabled=parse_bool(row.get("background_workload_enabled", "false")),
        background_group=row.get("background_workload_group") or "none",
        background_profile=row.get("background_workload_profile") or "none",
        trial_id=row.get("trial_id", ""),
        measurement_mode=row.get("perf_measurement_mode", "unknown"),
        start_time=float(row.get("timestamp_unix") or 0.0),
    )


def discover_runs(input_dir: Path) -> tuple[dict[str, Run], dict[str, dict[str, Run]]]:
    perf_paths = sorted(input_dir.rglob("*_perf.csv"))
    if not perf_paths:
        raise FileNotFoundError(f"No *_perf.csv files found below {input_dir}")

    candidates: dict[str, dict[str, list[Run]]] = {}
    for perf_path in perf_paths:
        try:
            run = read_run(perf_path)
        except ValueError as exc:
            warnings.warn(str(exc))
            continue
        candidates.setdefault(run.scenario, {}).setdefault(run.device, []).append(run)

    if "baseline" not in candidates:
        raise ValueError(
            "No clean baseline was found. Expected clean Small TrashNet, IID, "
            "SimpleCNN, batch size 16, baseline augmentation, and no background workload."
        )

    baseline = select_latest_by_device(candidates.pop("baseline"), "baseline")
    scenarios: dict[str, dict[str, Run]] = {}
    for scenario, by_device in sorted(candidates.items()):
        selected = select_latest_by_device(by_device, scenario)
        paired = set(selected) & set(baseline)
        missing = set(selected) - paired
        if missing:
            warnings.warn(
                f"Skipping devices without a baseline for {scenario}: {sorted(missing)}"
            )
        selected = {device: selected[device] for device in sorted(paired)}
        if selected:
            scenarios[scenario] = selected

    if not scenarios:
        raise ValueError(f"No robustness scenarios were found below {input_dir}")
    return baseline, scenarios


def select_latest_by_device(
    candidates: dict[str, list[Run]], scenario: str
) -> dict[str, Run]:
    selected: dict[str, Run] = {}
    for device, runs in candidates.items():
        runs.sort(key=lambda run: run.start_time)
        selected[device] = runs[-1]
        if len(runs) > 1:
            ignored = ", ".join(run.perf_path.name for run in runs[:-1])
            warnings.warn(
                f"{device}/{scenario}: using latest {runs[-1].perf_path.name}; "
                f"ignoring {ignored}."
            )
    return selected


def available_metrics(runs: list[Run]) -> tuple[Metric, ...]:
    common_columns = set.intersection(
        *(set(pd.read_csv(run.perf_path, nrows=0).columns) for run in runs)
    )
    return tuple(
        metric
        for metric in METRICS
        if metric.numerator in common_columns
        and (metric.denominator is None or metric.denominator in common_columns)
    )


def required_columns(metrics: tuple[Metric, ...], header: set[str]) -> list[str]:
    columns = ["epoch", "phase", "perf_status"]
    event_columns: list[str] = []
    for metric in metrics:
        columns.append(metric.numerator)
        event_columns.append(metric.numerator)
        if metric.denominator:
            columns.append(metric.denominator)
            event_columns.append(metric.denominator)
    for event_column in event_columns:
        enabled_column = f"{event_column}_enabled_pct"
        if enabled_column in header:
            columns.append(enabled_column)
    return list(dict.fromkeys(columns))


def aggregate_run(
    run: Run, metrics: tuple[Metric, ...]
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    header = set(pd.read_csv(run.perf_path, nrows=0).columns)
    columns = required_columns(metrics, header)
    frame = pd.read_csv(run.perf_path, usecols=columns, low_memory=False)
    frame = frame.loc[frame["perf_status"].eq("ok")].copy()
    frame["epoch"] = pd.to_numeric(frame["epoch"], errors="coerce")
    numeric = [column for column in columns if column not in {"phase", "perf_status"}]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")

    summaries: dict[str, pd.DataFrame] = {}
    quality_rows: list[dict[str, float | int | str]] = []
    enabled_columns = [column for column in columns if column.endswith("_enabled_pct")]
    for phase in PHASES:
        selected = frame.loc[frame["phase"].eq(phase) & frame["epoch"].notna()].copy()
        if selected.empty:
            raise ValueError(f"No {phase} samples in {run.perf_path}")
        selected["epoch"] = selected["epoch"].astype(int)
        result = pd.DataFrame(index=sorted(selected["epoch"].unique()), dtype=float)
        result.index.name = "epoch"
        for metric in metrics:
            numerator = selected.groupby("epoch")[metric.numerator].sum(min_count=1)
            if metric.denominator is None:
                result[metric.key] = numerator * metric.scale
            else:
                denominator = selected.groupby("epoch")[metric.denominator].sum(min_count=1)
                result[metric.key] = (
                    numerator.div(denominator.where(denominator > 0)) * metric.scale
                )
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


def plot_trace(
    axis: plt.Axes, values: pd.Series, color: str, label: str
) -> None:
    axis.plot(
        values.index,
        values,
        color=color,
        linewidth=1.7,
        marker="o",
        markersize=3.2,
        label=label,
    )


def save_device_scenario_figure(
    device: str,
    scenario: str,
    baseline_run: Run,
    scenario_run: Run,
    summaries: dict[Path, dict[str, pd.DataFrame]],
    metrics: tuple[Metric, ...],
    output_dir: Path,
    dpi: int,
) -> Path:
    figure, axes = plt.subplots(
        nrows=len(metrics),
        ncols=len(PHASES),
        figsize=(18, max(2.7 * len(metrics), 8)),
        squeeze=False,
    )

    for row, metric in enumerate(metrics):
        for column, phase in enumerate(PHASES):
            axis = axes[row, column]
            for name, run in (("baseline", baseline_run), ("scenario", scenario_run)):
                values = summaries[run.perf_path][phase][metric.key]
                label = "Clean baseline" if name == "baseline" else scenario.replace("_", " ")
                plot_trace(axis, values, GROUP_COLORS[name], label)
            axis.set_ylabel(metric.label, fontsize=8)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.tick_params(axis="both", labelsize=7)
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
            axis.grid(True, color="#d8dadd", linewidth=0.55, alpha=0.8)
            axis.margins(x=0.03, y=0.08)
            if row == 0:
                axis.set_title(phase.capitalize(), fontsize=11, pad=7)
            if row == len(metrics) - 1:
                axis.set_xlabel("Local training epoch", fontsize=9)

    handles = [
        Line2D([0], [0], color=GROUP_COLORS["baseline"], marker="o", label="Clean baseline"),
        Line2D([0], [0], color=GROUP_COLORS["scenario"], marker="o", label=scenario.replace("_", " ")),
    ]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.986), ncol=2, frameon=False)
    modes = sorted({baseline_run.measurement_mode, scenario_run.measurement_mode})
    figure.suptitle(
        f"{device}: clean baseline vs {scenario.replace('_', ' ')}\n"
        f"perf mode={','.join(modes)}",
        fontsize=14,
        y=0.999,
    )
    figure.tight_layout(rect=(0.025, 0.01, 0.995, 0.95), h_pad=1.0, w_pad=1.2)
    device_dir = output_dir / safe_name(device)
    device_dir.mkdir(parents=True, exist_ok=True)
    path = device_dir / f"{safe_name(scenario)}.png"
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    baseline, scenarios = discover_runs(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Found clean baselines for {len(baseline)} devices and "
        f"{len(scenarios)} robustness scenarios in {input_dir}"
    )

    summary_records: list[dict[str, object]] = []
    quality_records: list[dict[str, object]] = []
    selected_records: list[dict[str, object]] = []

    for scenario, scenario_runs in scenarios.items():
        devices = sorted(set(baseline) & set(scenario_runs))
        all_runs = [*(baseline[device] for device in devices), *(scenario_runs[device] for device in devices)]
        metrics = available_metrics(all_runs)
        if not metrics:
            warnings.warn(f"Skipping {scenario}: no common perf metrics")
            continue

        summaries: dict[Path, dict[str, pd.DataFrame]] = {}
        quality: dict[Path, pd.DataFrame] = {}
        for group, runs in (("baseline", baseline), ("scenario", scenario_runs)):
            for device in devices:
                run = runs[device]
                if run.perf_path not in summaries:
                    summaries[run.perf_path], quality[run.perf_path] = aggregate_run(run, metrics)
                selected_records.append(
                    {
                        "scenario": scenario,
                        "group": group,
                        "device": device,
                        "condition": run.condition,
                        "source_file": str(run.perf_path),
                    }
                )
                for phase, frame in summaries[run.perf_path].items():
                    for epoch, row in frame.iterrows():
                        for metric in metrics:
                            summary_records.append(
                                {
                                    "scenario": scenario,
                                    "group": group,
                                    "device": device,
                                    "phase": phase,
                                    "epoch": int(epoch),
                                    "metric": metric.key,
                                    "value": row[metric.key],
                                    "source_file": str(run.perf_path),
                                }
                            )
                for record in quality[run.perf_path].to_dict("records"):
                    quality_records.append(
                        {
                            "scenario": scenario,
                            "group": group,
                            "device": device,
                            "source_file": str(run.perf_path),
                            **record,
                        }
                    )

        for device in devices:
            path = save_device_scenario_figure(
                device,
                scenario,
                baseline[device],
                scenario_runs[device],
                summaries,
                metrics,
                output_dir,
                args.dpi,
            )
            print(f"Saved {path}", flush=True)

    pd.DataFrame(summary_records).to_csv(output_dir / "epoch_perf_summary.csv", index=False)
    pd.DataFrame(quality_records).to_csv(output_dir / "perf_sampling_quality.csv", index=False)
    pd.DataFrame(selected_records).drop_duplicates().to_csv(
        output_dir / "selected_runs.csv", index=False
    )
    print(f"Saved summary CSVs in {output_dir}")


if __name__ == "__main__":
    main()
