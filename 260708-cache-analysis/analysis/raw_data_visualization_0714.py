#!/usr/bin/env python3
"""Plot epoch-level hardware metrics across dataset/model configurations."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from analysis import PERF_COUNTER_COLUMNS, add_counter_deltas, color_for_method


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "collected_logs_0714"
DEFAULT_OUTPUT = HERE / "result" / "raw_data_visualization_0714"

CPU_METRICS = [
    "process_cpu_percent",
    "system_cpu_core_0",
    "system_cpu_core_1",
    "system_cpu_core_2",
    "system_cpu_core_3",
    "system_cpu_freq_mhz",
    "system_cpu_freq_core_0_mhz",
    "system_cpu_freq_core_1_mhz",
    "system_cpu_freq_core_2_mhz",
    "system_cpu_freq_core_3_mhz",
    "process_ctx_switches_voluntary_delta",
    "process_ctx_switches_involuntary_delta",
]
PERF_METRICS = list(PERF_COUNTER_COLUMNS)
PERF_NORMALIZATION_SUFFIXES = {
    "per_elapsed_sec": "per elapsed second",
    "per_task_clock_sec": "per task-clock second",
    "per_instruction": "per instruction",
}
METHOD_ORDER = ["clean", "unlearnable_examples", "availability_shortcuts", "random_label_flipping"]
METHOD_LABELS = {
    "clean": "Clean",
    "unlearnable_examples": "Unlearnable examples",
    "availability_shortcuts": "Availability shortcuts",
    "random_label_flipping": "Random label flipping",
}


def run_id(path: Path, root: Path) -> str:
    stem = path.stem.removesuffix("_perf").removesuffix("_metrics")
    return str(path.parent.relative_to(root) / stem)


def load_csvs(root: Path, kind: str) -> tuple[pd.DataFrame, int, int]:
    if kind == "perf":
        paths = sorted(root.rglob("*_perf.csv"))
    elif kind == "metrics":
        paths = sorted(root.rglob("*_metrics.csv"))
    else:
        paths = [
            path
            for path in sorted(root.rglob("*.csv"))
            if not path.stem.endswith(("_perf", "_metrics"))
        ]

    frames = []
    empty = failed = 0
    for path in paths:
        frame = pd.read_csv(path)
        if frame.empty:
            empty += 1
            continue
        if kind == "perf" and not (frame.get("perf_status", pd.Series(dtype=str)) == "ok").any():
            failed += 1
            continue
        frame["run_id"] = run_id(path, root)
        frames.append(frame)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), empty, failed


def normalize_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["dataset_name"] = frame["dataset"].astype(str).str.rsplit("/", n=1).str[-1]
    for core in range(4):
        source = f"system_cpu_freq_core_{core}"
        target = f"{source}_mhz"
        if source in frame and target not in frame:
            frame[target] = frame[source]
    return frame


def available_metrics(hardware: pd.DataFrame, perf: pd.DataFrame, phases: list[str]) -> list[tuple[str, pd.DataFrame]]:
    metrics = []
    for name, source in [
        *((name, hardware) for name in CPU_METRICS),
        *((name, perf) for name in PERF_METRICS),
    ]:
        if name in source and "phase" in source:
            values = pd.to_numeric(source.loc[source["phase"].isin(phases), name], errors="coerce")
            if values.notna().any():
                metrics.append((name, source))
    return metrics


def perf_trial_values(
    perf: pd.DataFrame,
    phases: list[str],
) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    """Aggregate perf counters before applying normalization.

    Perf counters are interval totals. Ratios must therefore be calculated as
    ratio-of-sums for each run/epoch/phase, rather than as the mean of
    per-interval ratios. ``perf_elapsed_sec`` is cumulative within a run, so
    its first difference gives the observed wall duration represented by each
    sample. The nominal interval is only used when that difference is missing
    or non-positive.
    """
    empty_groups = {suffix: [] for suffix in PERF_NORMALIZATION_SUFFIXES}
    if perf.empty or "phase" not in perf:
        return pd.DataFrame(), [], empty_groups

    group = [
        "device_id",
        "dataset_name",
        "model",
        "poisoning_method",
        "trial_id",
        "run_id",
        "epoch",
        "phase",
    ]
    required = [*group, "perf_elapsed_sec", "perf_interval_ms", "timestamp_unix"]
    counters = [
        metric
        for metric in PERF_METRICS
        if metric in perf and pd.to_numeric(perf[metric], errors="coerce").notna().any()
    ]
    columns = [column for column in [*required, "perf_status", *counters] if column in perf]
    subset = perf.loc[perf["phase"].isin(phases), columns].copy()
    if "perf_status" in subset:
        subset = subset[subset["perf_status"].eq("ok")].copy()
    if subset.empty or not counters:
        return pd.DataFrame(), [], empty_groups

    for column in ["epoch", "perf_elapsed_sec", "perf_interval_ms", "timestamp_unix", *counters]:
        if column in subset:
            subset[column] = pd.to_numeric(subset[column], errors="coerce")
    subset = subset.dropna(subset=[*group, "epoch"])
    subset = subset.sort_values(["run_id", "perf_elapsed_sec", "timestamp_unix"], na_position="last")

    observed = subset.groupby("run_id", observed=True)["perf_elapsed_sec"].diff()
    first_in_run = subset.groupby("run_id", observed=True).cumcount().eq(0)
    observed = observed.mask(first_in_run, subset["perf_elapsed_sec"])
    nominal = subset.get("perf_interval_ms", pd.Series(100.0, index=subset.index)) / 1000.0
    subset["_observed_elapsed_sec"] = observed.where(observed.gt(0), nominal)
    subset["_observed_elapsed_sec"] = subset["_observed_elapsed_sec"].where(
        np.isfinite(subset["_observed_elapsed_sec"]), nominal
    )

    totals = (
        subset.groupby(group, observed=True, as_index=False)[[*counters, "_observed_elapsed_sec"]]
        .sum(min_count=1)
    )
    raw_metrics = [*counters, "perf_observed_elapsed_sec"]
    normalized_metrics = {
        suffix: [f"{metric}_{suffix}" for metric in counters]
        for suffix in PERF_NORMALIZATION_SUFFIXES
    }
    rows = []
    for _, record in totals.iterrows():
        identifiers = {column: record[column] for column in group if column != "run_id"}
        elapsed_sec = float(record["_observed_elapsed_sec"])
        task_clock_ms = float(record.get("perf_task_clock", np.nan))
        instructions = float(record.get("perf_instructions", np.nan))
        denominators = {
            "per_elapsed_sec": elapsed_sec,
            "per_task_clock_sec": task_clock_ms / 1000.0,
            "per_instruction": instructions,
        }

        for metric in counters:
            value = float(record[metric])
            rows.append({**identifiers, "metric": metric, "value": value})
            for suffix, denominator in denominators.items():
                normalized = value / denominator if np.isfinite(denominator) and denominator > 0 else np.nan
                rows.append(
                    {
                        **identifiers,
                        "metric": f"{metric}_{suffix}",
                        "value": normalized,
                    }
                )
        rows.append(
            {
                **identifiers,
                "metric": "perf_observed_elapsed_sec",
                "value": elapsed_sec,
            }
        )

    values = pd.DataFrame(rows)
    return values, raw_metrics, normalized_metrics


def trial_values(metric_rows: list[tuple[str, pd.DataFrame]], phases: list[str]) -> pd.DataFrame:
    group = [
        "device_id",
        "dataset_name",
        "model",
        "poisoning_method",
        "trial_id",
        "epoch",
        "phase",
        "batch_idx",
    ]
    parts = []
    for metric, source in metric_rows:
        subset = source.loc[source["phase"].isin(phases), [*group, metric]].copy()
        for column in ["epoch", "batch_idx", metric]:
            subset[column] = pd.to_numeric(subset[column], errors="coerce")
        subset = subset.dropna(subset=["epoch", "batch_idx", metric])
        if subset.empty:
            continue
        # First average samples within a batch, then give every batch equal weight.
        batches = subset.groupby(group, observed=True, as_index=False)[metric].mean()
        per_trial = batches.groupby(group[:-1], observed=True, as_index=False)[metric].mean()
        parts.append(per_trial.rename(columns={metric: "value"}).assign(metric=metric))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def elapsed_trial_values(hardware: pd.DataFrame, phases: list[str]) -> pd.DataFrame:
    group = [
        "device_id",
        "dataset_name",
        "model",
        "poisoning_method",
        "trial_id",
        "epoch",
        "phase",
        "batch_idx",
    ]
    subset = hardware.loc[hardware["phase"].isin(phases), [*group, "timestamp_unix"]].copy()
    for column in ["epoch", "batch_idx", "timestamp_unix"]:
        subset[column] = pd.to_numeric(subset[column], errors="coerce")
    subset = subset.dropna(subset=["epoch", "batch_idx", "timestamp_unix"])
    batches = subset.groupby(group, observed=True)["timestamp_unix"].agg(lambda values: values.max() - values.min())
    per_trial = batches.groupby(group[:-1], observed=True).mean().rename("value").reset_index()
    return per_trial.assign(metric="elapsed_time_sec")


def summarize(values: pd.DataFrame) -> pd.DataFrame:
    group = ["device_id", "dataset_name", "model", "poisoning_method", "epoch", "phase", "metric"]
    summary = values.groupby(group, observed=True)["value"].agg(mean="mean", std="std", trials="count").reset_index()
    summary["epoch"] = summary["epoch"].astype(int)
    return summary.sort_values(group).reset_index(drop=True)


def convergence_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for event, prefix in [("train_epoch", "train"), ("eval_summary", "evaluation")]:
        subset = metrics[metrics["metric_event"] == event].copy()
        if subset.empty:
            continue
        for measure in ["accuracy", "loss"]:
            subset[measure] = pd.to_numeric(subset[measure], errors="coerce")
        subset["epoch"] = pd.to_numeric(subset["epoch"], errors="coerce")
        identifiers = [
            "device_id",
            "dataset_name",
            "model",
            "poisoning_method",
            "trial_id",
            "epoch",
        ]
        long = subset.melt(
            id_vars=identifiers,
            value_vars=["accuracy", "loss"],
            var_name="measure",
            value_name="value",
        ).dropna(subset=["epoch", "value"])
        long["metric"] = prefix + "_" + long["measure"]
        parts.append(long.drop(columns="measure"))
    if not parts:
        return pd.DataFrame()

    values = pd.concat(parts, ignore_index=True)
    group = ["device_id", "dataset_name", "model", "poisoning_method", "epoch", "metric"]
    summary = values.groupby(group, observed=True)["value"].agg(mean="mean", std="std", trials="count").reset_index()
    summary["epoch"] = summary["epoch"].astype(int)
    return summary.sort_values(group).reset_index(drop=True)


def safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value)


def plot_configuration(
    data: pd.DataFrame,
    device: str,
    dataset: str,
    model: str,
    phases: list[str],
    metrics: list[str],
    output: Path,
    dpi: int,
    qualifier: str = "raw counters and hardware metrics",
) -> None:
    fig, axes = plt.subplots(
        len(metrics),
        len(phases),
        figsize=(16, max(12, len(metrics) * 1.65)),
        sharex=True,
        sharey=False,
        squeeze=False,
    )
    subset = data[
        (data["device_id"].astype(str) == device)
        & (data["dataset_name"] == dataset)
        & (data["model"] == model)
    ]
    epochs = sorted(subset["epoch"].unique())
    methods = [method for method in METHOD_ORDER if method in set(subset["poisoning_method"])]

    for row, metric in enumerate(metrics):
        for column, phase in enumerate(phases):
            ax = axes[row, column]
            panel = subset[(subset["phase"] == phase) & (subset["metric"] == metric)]
            for method in methods:
                line = panel[panel["poisoning_method"] == method].sort_values("epoch")
                if line.empty:
                    continue
                ax.plot(
                    line["epoch"],
                    line["mean"],
                    color=color_for_method(method),
                    marker="o",
                    markersize=3,
                    linewidth=1.1,
                )
            ax.set_xticks(epochs)
            ax.tick_params(axis="both", labelsize=6, labelbottom=True)
            ax.grid(True, color="#d7d7d7", linewidth=0.5)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useOffset=False)
            if column == 0:
                ax.set_ylabel(metric, fontsize=7)
            if row == 0:
                ax.set_title(phase.capitalize(), fontsize=10)
            if row == len(metrics) - 1:
                ax.set_xlabel("Epoch", fontsize=8)

    handles = [
        Line2D([0], [0], color=color_for_method(method), marker="o", linewidth=1.3, label=METHOD_LABELS[method])
        for method in methods
    ]
    fig.suptitle(
        f"{device}: {dataset} / {model} - {qualifier}",
        fontsize=14,
        y=0.999,
    )
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.987), ncol=len(handles), frameon=False)
    fig.subplots_adjust(left=0.16, right=0.99, top=0.955, bottom=0.02, hspace=0.65, wspace=0.12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)


def plot_convergence_grid(
    data: pd.DataFrame,
    device: str,
    metric: str,
    datasets: list[str],
    models: list[str],
    output: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(
        len(datasets),
        len(models),
        figsize=(6.4 * len(models), 4.0 * len(datasets)),
        sharex=False,
        sharey=False,
        squeeze=False,
    )
    subset = data[(data["device_id"].astype(str) == device) & (data["metric"] == metric)]
    methods = [method for method in METHOD_ORDER if method in set(subset["poisoning_method"])]

    for row, dataset in enumerate(datasets):
        for column, model in enumerate(models):
            ax = axes[row, column]
            panel = subset[(subset["dataset_name"] == dataset) & (subset["model"] == model)]
            if panel.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="#666666")
            for method in methods:
                line = panel[panel["poisoning_method"] == method].sort_values("epoch")
                if line.empty:
                    continue
                ax.plot(
                    line["epoch"],
                    line["mean"],
                    color=color_for_method(method),
                    marker="o",
                    markersize=4,
                    linewidth=1.3,
                )
            epochs = sorted(panel["epoch"].unique())
            if epochs:
                ax.set_xticks(epochs)
            ax.set_title(f"{dataset} / {model}", fontsize=10)
            ax.set_xlabel("Epoch", fontsize=8)
            ax.set_ylabel(metric, fontsize=8)
            ax.grid(True, color="#d7d7d7", linewidth=0.5)
            ax.tick_params(labelsize=7)

    handles = [
        Line2D([0], [0], color=color_for_method(method), marker="o", linewidth=1.3, label=METHOD_LABELS[method])
        for method in methods
    ]
    qualifier = "final train-partition evaluation" if metric.startswith("evaluation_") else "epoch training metric"
    fig.suptitle(f"{device}: {metric} ({qualifier})", fontsize=13, y=0.995)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=len(handles), frameon=False)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.84, bottom=0.08, hspace=0.32, wspace=0.22)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phases", default="forward,backward")
    parser.add_argument("--dpi", type=int, default=110)
    args = parser.parse_args()

    phases = [phase.strip() for phase in args.phases.split(",") if phase.strip()]
    hardware, hardware_empty, _ = load_csvs(args.input_dir, "hardware")
    perf, perf_empty, perf_failed = load_csvs(args.input_dir, "perf")
    metric_logs, metrics_empty, _ = load_csvs(args.input_dir, "metrics")
    if hardware.empty:
        parser.error(f"no hardware samples found under {args.input_dir}")
    hardware = add_counter_deltas(normalize_metadata(hardware))
    perf = normalize_metadata(perf)
    metric_logs = normalize_metadata(metric_logs)

    hardware_metric_rows = available_metrics(hardware, pd.DataFrame(), phases)
    perf_values, raw_perf_metrics, normalized_perf_metrics = perf_trial_values(perf, phases)
    value_parts = [
        trial_values(hardware_metric_rows, phases),
        elapsed_trial_values(hardware, phases),
        perf_values,
    ]
    values = pd.concat([part for part in value_parts if not part.empty], ignore_index=True)
    summary = summarize(values)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trial_values_path = args.output_dir / "epoch_metric_trial_values.csv"
    values.to_csv(trial_values_path, index=False)
    summary_path = args.output_dir / "epoch_metric_summary.csv"
    summary.to_csv(summary_path, index=False)
    normalized_names = {
        metric
        for metrics_for_normalization in normalized_perf_metrics.values()
        for metric in metrics_for_normalization
    }
    normalized_summary_path = args.output_dir / "epoch_perf_normalized_summary.csv"
    summary[summary["metric"].isin(normalized_names)].to_csv(normalized_summary_path, index=False)
    convergence = convergence_summary(metric_logs)
    convergence_path = args.output_dir / "convergence_summary.csv"
    convergence.to_csv(convergence_path, index=False)

    devices = sorted(summary["device_id"].astype(str).unique())
    available_summary_metrics = set(summary["metric"])
    metrics = [
        *[name for name, _ in hardware_metric_rows if name in available_summary_metrics],
        *[name for name in raw_perf_metrics if name in available_summary_metrics],
        "elapsed_time_sec",
    ]
    print(f"Hardware files: {hardware['run_id'].nunique()} ({hardware_empty} empty)", flush=True)
    print(f"Perf files: {perf['run_id'].nunique() if not perf.empty else 0} usable, {perf_empty} empty, {perf_failed} failed", flush=True)
    print(f"Metric files: {metric_logs['run_id'].nunique()} ({metrics_empty} empty)", flush=True)
    print(f"Devices: {', '.join(devices)}", flush=True)
    print(f"Wrote {trial_values_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {normalized_summary_path}", flush=True)
    print(f"Wrote {convergence_path}", flush=True)

    for old_figure in args.output_dir.rglob("*.png"):
        old_figure.unlink()

    figure_count = 0
    for device in devices:
        device_summary = summary[summary["device_id"].astype(str) == device]
        datasets = sorted(device_summary["dataset_name"].unique())
        models = sorted(device_summary["model"].unique())
        for dataset in datasets:
            for model in models:
                if device_summary[
                    (device_summary["dataset_name"] == dataset) & (device_summary["model"] == model)
                ].empty:
                    continue
                configuration = device_summary[
                    (device_summary["dataset_name"] == dataset) & (device_summary["model"] == model)
                ]
                configuration_metrics = [metric for metric in metrics if metric in set(configuration["metric"])]
                path = args.output_dir / safe_name(device) / f"{safe_name(dataset)}_{safe_name(model)}.png"
                plot_configuration(summary, device, dataset, model, phases, configuration_metrics, path, args.dpi)
                figure_count += 1

                for suffix, label in PERF_NORMALIZATION_SUFFIXES.items():
                    normalized_metrics = [
                        metric
                        for metric in normalized_perf_metrics[suffix]
                        if metric in set(configuration["metric"])
                    ]
                    if not normalized_metrics:
                        continue
                    normalized_path = (
                        args.output_dir
                        / safe_name(device)
                        / f"{safe_name(dataset)}_{safe_name(model)}_{suffix}.png"
                    )
                    plot_configuration(
                        summary,
                        device,
                        dataset,
                        model,
                        phases,
                        normalized_metrics,
                        normalized_path,
                        args.dpi,
                        qualifier=f"perf counters {label}",
                    )
                    figure_count += 1

    convergence_metrics = ["train_accuracy", "train_loss", "evaluation_accuracy", "evaluation_loss"]
    all_datasets = sorted(set(summary["dataset_name"]) | set(convergence["dataset_name"]))
    all_models = sorted(set(summary["model"]) | set(convergence["model"]))
    convergence_figure_count = 0
    for device in devices:
        for metric in convergence_metrics:
            if convergence[(convergence["device_id"].astype(str) == device) & (convergence["metric"] == metric)].empty:
                continue
            path = args.output_dir / safe_name(device) / "convergence" / f"{metric}.png"
            plot_convergence_grid(convergence, device, metric, all_datasets, all_models, path, args.dpi)
            convergence_figure_count += 1
    print(
        f"Wrote {figure_count} hardware figures and {convergence_figure_count} convergence figures under {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
