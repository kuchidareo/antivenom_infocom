"""Visualize controlled ViT Softmax structure and replay-only PMU results."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/antivenom-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.ticker import ScalarFormatter


REGIMES = ("low", "mid", "high")
REGIME_COLORS = {"low": "#2878B5", "mid": "#E07A1F", "high": "#C23B4A"}

PERF_METADATA_COLUMNS = {
    "perf_pid",
    "perf_measurement_mode",
    "perf_scope",
    "perf_elapsed_sec",
    "perf_interval_ms",
    "perf_phase_start_timestamp",
    "perf_phase_start_unix",
    "perf_phase_end_timestamp",
    "perf_phase_end_unix",
    "perf_phase_duration_sec",
    "perf_events",
    "perf_status",
    "perf_error",
}

PERF_LABELS = {
    "perf_cycles": "Cycles",
    "perf_instructions": "Instructions",
    "perf_task_clock": "Task clock (ms)",
    "perf_context_switches": "Context switches",
    "perf_cpu_migrations": "CPU migrations",
    "perf_page_faults": "Page faults",
    "perf_branches": "Branches",
    "perf_branch_misses": "Branch misses",
    "perf_br_retired": "Retired branches",
    "perf_br_mis_pred_retired": "Mispredicted retired branches",
    "perf_l1_dcache_loads": "L1D loads",
    "perf_l1_dcache_load_misses": "L1D load misses",
    "perf_l1d_cache": "L1D accesses",
    "perf_l1d_cache_refill": "L1D refills",
    "perf_l1d_cache_wb": "L1D writebacks",
    "perf_l1d_cache_rd": "L1D read accesses",
    "perf_l1d_cache_refill_rd": "L1D read refills",
    "perf_l1d_cache_wr": "L1D write accesses",
    "perf_l1d_cache_refill_wr": "L1D write refills",
    "perf_l2d_cache": "L2D accesses",
    "perf_l2d_cache_refill": "L2D refills",
    "perf_l2d_cache_wb": "L2D writebacks",
    "perf_l2d_cache_rd": "L2D read accesses",
    "perf_l2d_cache_refill_rd": "L2D read refills",
    "perf_l2d_cache_wr": "L2D write accesses",
    "perf_l2d_cache_refill_wr": "L2D write refills",
    "perf_bus_access": "Bus accesses",
    "perf_bus_access_rd": "Bus read accesses",
    "perf_bus_access_wr": "Bus write accesses",
    "perf_mem_access": "Memory accesses",
    "perf_ase_spec": "ASE speculative operations",
    "perf_vfp_spec": "VFP speculative operations",
    "perf_inst_spec": "Speculative operations",
}

STRUCTURE_METRICS = {
    "structure_score_row_pattern_entropy_bits": "Score-row pattern entropy (bits)",
    "structure_score_value_conditional_entropy_bits": "Score-value conditional entropy (bits)",
    "structure_unique_score_rows": "Unique score rows",
    "structure_unique_score_row_fraction": "Unique score-row fraction",
    "structure_adjacent_row_exact_repetition_rate": "Adjacent exact-row repetition rate",
    "structure_adjacent_element_persistence_rate": "Adjacent element persistence rate",
    "structure_adjacent_row_relative_l2": "Adjacent-row relative L2",
    "structure_adjacent_row_cosine": "Adjacent-row cosine similarity",
    "structure_comparison_signature_conditional_entropy_bits": (
        "Comparison-signature conditional entropy (bits)"
    ),
    "structure_comparison_signature_flip_rate": "Comparison-signature flip rate",
    "structure_comparison_signature_exact_repetition_rate": (
        "Comparison-signature exact repetition rate"
    ),
}

CONTROL_METRICS = {
    "structure_attention_probability_entropy_bits": "Attention probability entropy (bits)",
    "structure_attention_probability_entropy_std": "Within-run probability entropy SD",
    "structure_attention_max_probability": "Maximum attention probability",
    "structure_attention_top3_mass": "Attention top-3 mass",
    "structure_logit_mean": "Logit mean",
    "structure_logit_variance": "Logit variance",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", type=Path, default=Path("controlled_vit_softmax_results")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("visualization_ViT_softmax")
    )
    parser.add_argument("--format", choices=("pdf", "png"), default="pdf")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--metrics",
        default="",
        help="Optional comma-separated perf columns, with or without perf_",
    )
    return parser.parse_args()


def safe_name(value: Any) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value)).strip("_") or "unknown"


def numeric(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def local_perf_path(input_dir: Path, result: dict[str, Any]) -> Path | None:
    local = input_dir / f"{result.get('run_id')}_perf.csv"
    if local.is_file():
        return local
    configured = result.get("perf_csv")
    if configured and Path(configured).is_file():
        return Path(configured)
    return None


def load_perf_row(path: Path) -> dict[str, Any]:
    with path.open(newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("perf_status") == "ok" and row.get("phase") == "replay"
        ]
    if not rows:
        raise ValueError(f"No successful replay PMU row in {path}")
    if len(rows) != 1:
        raise ValueError(f"Expected one replay PMU row in {path}, found {len(rows)}")
    return rows[0]


def load_runs(input_dir: Path) -> pd.DataFrame:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for json_path in sorted(input_dir.glob("*.json")):
        try:
            result = json.loads(json_path.read_text())
            config = result["config"]
            structure = result["structure_metrics"]
            record: dict[str, Any] = {
                "run_id": result["run_id"],
                "source_json": str(json_path),
                "operator": "softmax",
                "regime": config["regime"],
                "seed": config["seed"],
                "trial_id": config.get("trial_id", "trial_0"),
                "device_id": config.get("device_id") or result.get("host", "unknown"),
                "host": result.get("host", ""),
                "batch_size": config["batch_size"],
                "grid_size": config["grid_size"],
                "tokens": int(config["grid_size"]) ** 2 + 1,
                "heads": config["heads"],
                "mid_prototypes": config["mid_prototypes"],
                "warmup": config["warmup"],
                "repeats": config["repeats"],
                "threads": config["threads"],
                "elapsed_seconds": result["elapsed_seconds"],
                "nanoseconds_per_softmax": result["nanoseconds_per_softmax"],
                "logit_multiset_sha256": structure["logit_multiset_sha256"],
            }
            record.update(
                {f"structure_{key}": value for key, value in structure.items()}
            )
            perf_path = local_perf_path(input_dir, result)
            if perf_path is not None:
                perf_row = load_perf_row(perf_path)
                if perf_row.get("run_id") != result["run_id"]:
                    raise ValueError(f"PMU run_id does not match {result['run_id']}")
                record["source_perf_csv"] = str(perf_path)
                for key, value in perf_row.items():
                    if key.startswith("perf_"):
                        record[key] = numeric(value)
            else:
                record["source_perf_csv"] = ""
            records.append(record)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{json_path.name}: {exc}")

    if errors:
        raise ValueError("Unable to load result files:\n  " + "\n  ".join(errors))
    if not records:
        raise FileNotFoundError(f"No ViT Softmax result JSON files in {input_dir}")

    frame = pd.DataFrame.from_records(records)
    unexpected = sorted(set(frame["regime"].astype(str)) - set(REGIMES))
    if unexpected:
        raise ValueError(f"Unexpected entropy regimes: {', '.join(unexpected)}")
    frame["regime"] = pd.Categorical(frame["regime"], REGIMES, ordered=True)
    return frame.sort_values(
        ["device_id", "regime", "seed", "trial_id"]
    ).reset_index(drop=True)


def perf_metric_columns(frame: pd.DataFrame) -> list[str]:
    metrics: list[str] = []
    for column in frame.columns:
        if not column.startswith("perf_") or column in PERF_METADATA_COLUMNS:
            continue
        if column.endswith("_enabled_pct") or column.endswith("_runtime_pct"):
            continue
        if pd.to_numeric(frame[column], errors="coerce").notna().any():
            metrics.append(column)
    return metrics


def select_metrics(available: list[str], requested: str) -> list[str]:
    if not requested.strip():
        return available
    selected: list[str] = []
    missing: list[str] = []
    for token in requested.split(","):
        token = token.strip()
        if not token:
            continue
        column = token if token.startswith("perf_") else f"perf_{token}"
        if column in available:
            selected.append(column)
        else:
            missing.append(column)
    if missing:
        raise ValueError(f"Requested PMU metrics are unavailable: {', '.join(missing)}")
    return selected


def style_axis(axis: Axes) -> None:
    axis.grid(axis="y", color="#D9DEE5", linewidth=0.7, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=8)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 4))
    axis.yaxis.set_major_formatter(formatter)


def plot_regime_series(
    axis: Axes,
    frame: pd.DataFrame,
    metric: str,
    minimum_relative_halfspan: float = 0.0,
) -> None:
    x_values: list[int] = []
    means: list[float] = []
    deviations: list[float] = []
    for index, regime in enumerate(REGIMES):
        values = pd.to_numeric(
            frame.loc[frame["regime"] == regime, metric], errors="coerce"
        ).dropna()
        if values.empty:
            continue
        x_values.append(index)
        means.append(float(values.mean()))
        deviations.append(float(values.std(ddof=1)) if len(values) > 1 else 0.0)
        jitter = np.linspace(-0.055, 0.055, len(values)) if len(values) > 1 else [0.0]
        axis.scatter(
            index + np.asarray(jitter),
            values,
            color=REGIME_COLORS[regime],
            s=22,
            alpha=0.55,
            linewidths=0,
            zorder=3,
        )
    if x_values:
        mean_array = np.asarray(means)
        deviation_array = np.asarray(deviations)
        axis.plot(
            x_values, mean_array, color="#24292F", marker="o", markersize=4,
            linewidth=1.4, zorder=2,
        )
        axis.errorbar(
            x_values, mean_array, yerr=deviation_array, fmt="none",
            ecolor="#24292F", elinewidth=0.9, capsize=3, capthick=0.9, zorder=4,
        )
        if np.any(deviation_array > 0):
            axis.fill_between(
                x_values,
                mean_array - deviation_array,
                mean_array + deviation_array,
                color="#6E7781",
                alpha=0.16,
                linewidth=0,
            )
    axis.set_xticks(range(len(REGIMES)), [item.title() for item in REGIMES])
    style_axis(axis)
    if minimum_relative_halfspan > 0:
        values = pd.to_numeric(frame[metric], errors="coerce").dropna()
        if not values.empty:
            center = float(values.mean())
            data_halfspan = max(
                abs(float(values.max()) - center),
                abs(center - float(values.min())),
            )
            halfspan = max(
                data_halfspan * 1.1,
                abs(center) * minimum_relative_halfspan,
                1e-12,
            )
            axis.set_ylim(center - halfspan, center + halfspan)


def plot_metric_rows(
    frame: pd.DataFrame,
    metrics: Iterable[str],
    labels: dict[str, str],
    output_path: Path,
    title: str,
    dpi: int,
    minimum_relative_halfspan: float = 0.0,
) -> None:
    metrics = [metric for metric in metrics if metric in frame]
    if not metrics:
        return
    figure, axes = plt.subplots(
        len(metrics), 1,
        figsize=(7.4, max(3.2, 2.25 * len(metrics))),
        squeeze=False,
        constrained_layout=True,
    )
    for index, metric in enumerate(metrics):
        axis = axes[index, 0]
        plot_regime_series(axis, frame, metric, minimum_relative_halfspan)
        axis.set_ylabel(labels.get(metric, readable_name(metric)), fontsize=8)
        if index != len(metrics) - 1:
            axis.tick_params(labelbottom=False)
    figure.suptitle(
        f"{title}\nMean +/- 1 SD; dots are individual process runs", fontsize=11
    )
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def readable_name(metric: str) -> str:
    return metric.removeprefix("perf_").replace("_", " ").title()


def perf_label(metric: str) -> str:
    base = metric.removesuffix("_per_instruction")
    label = PERF_LABELS.get(base, readable_name(base))
    return f"{label} / instruction" if metric.endswith("_per_instruction") else label


def add_per_instruction_metrics(
    frame: pd.DataFrame, metrics: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    if "perf_instructions" not in frame:
        return frame, []
    denominator = pd.to_numeric(
        frame["perf_instructions"], errors="coerce"
    ).replace(0, np.nan)
    if denominator.notna().sum() == 0:
        return frame, []
    output = frame.copy()
    normalized: list[str] = []
    for metric in metrics:
        if metric == "perf_instructions":
            continue
        name = f"{metric}_per_instruction"
        output[name] = pd.to_numeric(output[metric], errors="coerce") / denominator
        normalized.append(name)
    return output, normalized


def validate_configuration(frame: pd.DataFrame, device_id: str) -> None:
    columns = [
        "batch_size", "grid_size", "tokens", "heads", "mid_prototypes",
        "warmup", "repeats", "threads",
    ]
    varying = [column for column in columns if frame[column].nunique(dropna=False) > 1]
    if varying:
        raise ValueError(
            f"Device {device_id} has mixed controlled configurations in: "
            f"{', '.join(varying)}. Use a separate input directory per configuration."
        )


def render_device(
    frame: pd.DataFrame,
    output_dir: Path,
    file_format: str,
    dpi: int,
    requested_metrics: str,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    available = perf_metric_columns(frame)
    perf_metrics = select_metrics(available, requested_metrics)
    written: list[Path] = []

    summary_path = output_dir / "controlled_vit_softmax_summary.csv"
    frame.to_csv(summary_path, index=False)
    written.append(summary_path)

    structure_path = output_dir / f"structure_change.{file_format}"
    plot_metric_rows(
        frame, STRUCTURE_METRICS, STRUCTURE_METRICS, structure_path,
        "Manipulated score-row structure", dpi,
    )
    written.append(structure_path)

    controls_path = output_dir / f"controlled_quantities.{file_format}"
    plot_metric_rows(
        frame, CONTROL_METRICS, CONTROL_METRICS, controls_path,
        "Quantities held constant across regimes", dpi,
        minimum_relative_halfspan=0.01,
    )
    written.append(controls_path)

    if perf_metrics:
        raw_path = output_dir / f"pmu_raw.{file_format}"
        plot_metric_rows(
            frame, perf_metrics, {item: perf_label(item) for item in perf_metrics},
            raw_path, "Replay-only PMU counters", dpi,
        )
        written.append(raw_path)

        normalized_frame, normalized_metrics = add_per_instruction_metrics(
            frame, perf_metrics
        )
        if normalized_metrics:
            normalized_path = output_dir / f"pmu_per_instruction.{file_format}"
            plot_metric_rows(
                normalized_frame,
                normalized_metrics,
                {item: perf_label(item) for item in normalized_metrics},
                normalized_path,
                "Replay-only PMU counters per instruction",
                dpi,
            )
            written.append(normalized_path)

    runtime_metrics = ("elapsed_seconds", "nanoseconds_per_softmax")
    runtime_labels = {
        "elapsed_seconds": "Replay elapsed time (s)",
        "nanoseconds_per_softmax": "Nanoseconds / Softmax call",
    }
    runtime_path = output_dir / f"runtime.{file_format}"
    plot_metric_rows(
        frame, runtime_metrics, runtime_labels, runtime_path,
        "Softmax replay runtime", dpi,
    )
    written.append(runtime_path)
    return written


def main() -> None:
    args = parse_args()
    frame = load_runs(args.input_dir.resolve())
    devices = [str(value) for value in frame["device_id"].dropna().unique()]
    all_written: list[Path] = []
    for device_id in devices:
        subset = frame[frame["device_id"].astype(str) == device_id].copy()
        validate_configuration(subset, device_id)
        device_output = (
            args.output_dir.resolve()
            if len(devices) == 1
            else args.output_dir.resolve() / safe_name(device_id)
        )
        all_written.extend(
            render_device(
                subset, device_output, args.format, args.dpi, args.metrics
            )
        )

    print(f"Loaded {len(frame)} runs from {args.input_dir.resolve()}")
    print(f"Devices: {', '.join(devices)}")
    for path in all_written:
        print(path)


if __name__ == "__main__":
    main()
