#!/usr/bin/env python3
"""Visualize all configured scenarios for every epoch and batch.

Each batch-level value is the sum of the 15 measured leaf-layer values for a
single phase. Ratios are calculated after summation, not averaged per layer.
The output grid places normal training on the left and frozen replay on the
right. Every subplot is one batch and contains all available scenario
trajectories over epochs 1-15.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/antivenom-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, ScalarFormatter


SCENARIO_LABELS = {
    "baseline": "Clean",
    "moderate_augmentation": "Moderate augmentation",
    "strong_augmentation": "Strong augmentation",
    "availability_shortcuts": "Availability shortcuts",
    "badsampler": "BadSampler",
}
SCENARIO_COLORS = {
    "baseline": "#2563eb",
    "moderate_augmentation": "#16a34a",
    "strong_augmentation": "#dc2626",
    "availability_shortcuts": "#9333ea",
    "badsampler": "#ea580c",
}
MODE_LABELS = {
    "train": "Training",
    "frozen_replay": "Frozen replay",
}

BASIC_ADDITIVE_METRICS = {
    "duration_ms": "Elapsed time (ms)",
    "perf_cycles": "Cycles",
    "perf_instructions": "Instructions",
    "perf_branches": "Branches",
    "perf_branch_misses": "Branch misses",
    "perf_l1_dcache_loads": "L1D loads",
    "perf_l1_dcache_load_misses": "L1D load misses",
}
BASIC_DERIVED_METRICS = {
    "ipc": "Instructions per cycle",
    "branch_miss_percent": "Branch miss rate (%)",
    "l1d_load_miss_percent": "L1D load miss rate (%)",
}
BASIC_RAW_LAYER_METRICS = {
    "duration_ns": "Elapsed time (ns)",
    "perf_cycles_raw": "Raw cycles",
    "perf_instructions_raw": "Raw instructions",
    "perf_branches_raw": "Raw branch loads",
    "perf_branch_misses_raw": "Raw branch-load misses",
    "perf_l1_dcache_loads_raw": "Raw L1D loads",
    "perf_l1_dcache_load_misses_raw": "Raw L1D load misses",
}
TRANSLATION_ADDITIVE_METRICS = {
    "duration_ms": "Elapsed time (ms)",
    "perf_arm_l1i_cache_access": "L1I cache accesses",
    "perf_arm_l1i_cache_refill": "L1I cache refills",
    "perf_arm_itlb_access": "iTLB accesses",
    "perf_arm_itlb_refill": "iTLB refills",
    "perf_arm_dtlb_load_refill": "dTLB load refills",
    "perf_arm_ld_spec": "Speculatively executed loads",
}
TRANSLATION_DERIVED_METRICS = {
    "l1i_cache_refill_percent": "L1I cache refill rate (%)",
    "itlb_refill_percent": "iTLB refill rate (%)",
    "dtlb_refills_per_1k_ld_spec": "dTLB load refills per 1K speculative loads",
}
TRANSLATION_RAW_LAYER_METRICS = {
    "duration_ns": "Elapsed time (ns)",
    "perf_arm_l1i_cache_access_raw": "Raw L1I cache accesses",
    "perf_arm_l1i_cache_refill_raw": "Raw L1I cache refills",
    "perf_arm_itlb_access_raw": "Raw iTLB accesses",
    "perf_arm_itlb_refill_raw": "Raw iTLB refills",
    "perf_arm_dtlb_load_refill_raw": "Raw dTLB load refills",
    "perf_arm_ld_spec_raw": "Raw speculatively executed loads",
}
X86_TRANSLATION_ADDITIVE_METRICS = {
    "duration_ms": "Elapsed time (ms)",
    "perf_instructions": "Instructions",
    "perf_itlb_load_misses": "iTLB load misses",
    "perf_dtlb_load_misses": "dTLB load misses",
    "perf_l1_icache_load_misses": "L1I cache load misses",
    "perf_mem_inst_retired_all_loads": "Retired load instructions",
}
X86_TRANSLATION_DERIVED_METRICS = {
    "itlb_misses_per_1k_instructions": "iTLB misses per 1K instructions",
    "l1i_misses_per_1k_instructions": "L1I misses per 1K instructions",
    "dtlb_misses_per_1k_retired_loads": "dTLB misses per 1K retired loads",
}
X86_TRANSLATION_RAW_LAYER_METRICS = {
    "duration_ns": "Elapsed time (ns)",
    "perf_instructions_raw": "Raw instructions",
    "perf_itlb_load_misses_raw": "Raw iTLB load misses",
    "perf_dtlb_load_misses_raw": "Raw dTLB load misses",
    "perf_l1_icache_load_misses_raw": "Raw L1I cache load misses",
    "perf_mem_inst_retired_all_loads_raw": "Raw retired load instructions",
}
X86_DTLB_ADDITIVE_METRICS = {
    "duration_ms": "Elapsed time (ms)",
    "perf_dtlb_loads": "dTLB load accesses",
    "perf_dtlb_load_misses": "dTLB load misses",
    "perf_dtlb_stores": "dTLB store accesses",
    "perf_dtlb_store_misses": "dTLB store misses",
}
X86_DTLB_DERIVED_METRICS = {
    "dtlb_load_miss_percent": "dTLB load miss rate (%)",
    "dtlb_store_miss_percent": "dTLB store miss rate (%)",
}
X86_DTLB_RAW_LAYER_METRICS = {
    "duration_ns": "Elapsed time (ns)",
    "perf_dtlb_loads_raw": "Raw dTLB load accesses",
    "perf_dtlb_load_misses_raw": "Raw dTLB load misses",
    "perf_dtlb_stores_raw": "Raw dTLB store accesses",
    "perf_dtlb_store_misses_raw": "Raw dTLB store misses",
}
JETSON_DTLB_ADDITIVE_METRICS = {
    "duration_ms": "Elapsed time (ms)",
    "perf_dtlb_loads": "dTLB load accesses",
    "perf_dtlb_load_misses": "dTLB load misses",
}
JETSON_DTLB_DERIVED_METRICS = {
    "dtlb_load_miss_percent": "dTLB load miss rate (%)",
}
JETSON_DTLB_RAW_LAYER_METRICS = {
    "duration_ns": "Elapsed time (ns)",
    "perf_dtlb_loads_raw": "Raw dTLB load accesses",
    "perf_dtlb_load_misses_raw": "Raw dTLB load misses",
}
RPI_DTLB_ADDITIVE_METRICS = {
    "duration_ms": "Elapsed time (ms)",
    "perf_dtlb_load_misses": "dTLB load misses",
    "perf_dtlb_store_misses": "dTLB store misses",
}
RPI_DTLB_DERIVED_METRICS: Dict[str, str] = {}
RPI_DTLB_RAW_LAYER_METRICS = {
    "duration_ns": "Elapsed time (ns)",
    "perf_dtlb_load_misses_raw": "Raw dTLB load misses",
    "perf_dtlb_store_misses_raw": "Raw dTLB store misses",
}
METRIC_CONFIGS = {
    "basic": (
        BASIC_ADDITIVE_METRICS,
        BASIC_DERIVED_METRICS,
        BASIC_RAW_LAYER_METRICS,
    ),
    "translation": (
        TRANSLATION_ADDITIVE_METRICS,
        TRANSLATION_DERIVED_METRICS,
        TRANSLATION_RAW_LAYER_METRICS,
    ),
    "translation_x86": (
        X86_TRANSLATION_ADDITIVE_METRICS,
        X86_TRANSLATION_DERIVED_METRICS,
        X86_TRANSLATION_RAW_LAYER_METRICS,
    ),
    "dtlb_x86": (
        X86_DTLB_ADDITIVE_METRICS,
        X86_DTLB_DERIVED_METRICS,
        X86_DTLB_RAW_LAYER_METRICS,
    ),
    "dtlb_jetson": (
        JETSON_DTLB_ADDITIVE_METRICS,
        JETSON_DTLB_DERIVED_METRICS,
        JETSON_DTLB_RAW_LAYER_METRICS,
    ),
    "dtlb_rpi": (
        RPI_DTLB_ADDITIVE_METRICS,
        RPI_DTLB_DERIVED_METRICS,
        RPI_DTLB_RAW_LAYER_METRICS,
    ),
}
REQUIRED_COLUMNS = {
    "device_id",
    "trial_id",
    "scenario",
    "experiment_mode",
    "phase",
    "epoch",
    "batch_idx",
    "layer_index",
    "layer_name",
    "layer_type",
    "duration_ms",
    "duration_ns",
    "perf_events",
}

PERF_COLUMN_ALIASES = {
    "perf_branch_loads": "perf_branches",
    "perf_branch_load_misses": "perf_branch_misses",
    "perf_branch_loads_raw": "perf_branches_raw",
    "perf_branch_load_misses_raw": "perf_branch_misses_raw",
}


def discover_layer_csvs(input_dir: Path) -> List[Path]:
    paths = sorted(input_dir.rglob("*_layer_perf.csv"))
    if not paths:
        raise FileNotFoundError(f"No *_layer_perf.csv files found under {input_dir}")
    return paths


def classify_perf_events(events: object) -> str:
    value = str(events)
    if "arm_l1i_cache_access" in value:
        return "translation"
    if "mem_inst_retired.all_loads" in value:
        return "translation_x86"
    if "dTLB-stores" in value:
        return "dtlb_x86"
    if "dTLB-loads" in value:
        return "dtlb_jetson"
    if "dTLB-store-misses" in value:
        return "dtlb_rpi"
    return "basic"


def detect_perf_preset(data: pd.DataFrame) -> str:
    presets = set(data["perf_profile"].dropna().astype(str).unique())
    if len(presets) != 1:
        raise ValueError(
            "Input contains incompatible PMU event profiles. "
            "Point --input-dir to one preset directory at a time."
        )
    return presets.pop()


def load_layer_data(paths: Iterable[Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    inventory: List[Dict[str, object]] = []
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        for source, canonical in PERF_COLUMN_ALIASES.items():
            if canonical not in frame.columns and source in frame.columns:
                frame[canonical] = frame[source]
        missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        frame["source_file"] = str(path)
        frames.append(frame)
        inventory.append(
            {
                "source_file": str(path),
                "rows": len(frame),
                "device_id": first_value(frame, "device_id"),
                "scenario": first_value(frame, "scenario"),
                "experiment_mode": first_value(frame, "experiment_mode"),
                "trial_id": first_value(frame, "trial_id"),
                "perf_profile": classify_perf_events(first_value(frame, "perf_events")),
                "epochs": frame["epoch"].nunique(),
                "batches": frame["batch_idx"].nunique(),
                "layers": frame["layer_index"].nunique(),
            }
        )

    data = pd.concat(frames, ignore_index=True)
    data = data[
        data["scenario"].isin(SCENARIO_LABELS)
        & data["experiment_mode"].isin(MODE_LABELS)
        & data["phase"].isin(["forward", "backward"])
    ].copy()
    if data.empty:
        raise ValueError("No configured train/frozen layer rows were found.")

    data["perf_profile"] = data["perf_events"].map(classify_perf_events)
    base_numeric_columns = ["epoch", "batch_idx", "layer_index"]
    metric_columns = sorted(
        {
            column
            for additive, _, raw in METRIC_CONFIGS.values()
            for column in [*additive, *raw]
            if column in data.columns
        }
    )
    for column in [*base_numeric_columns, *metric_columns]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    invalid_base = data[base_numeric_columns].isna().any(axis=1)
    if invalid_base.any():
        raise ValueError(f"Found {int(invalid_base.sum())} rows with invalid indices.")
    for perf_profile, profile_data in data.groupby("perf_profile", observed=True):
        additive_metrics, _, raw_layer_metrics = METRIC_CONFIGS[str(perf_profile)]
        required_metrics = [*additive_metrics, *raw_layer_metrics]
        missing_metrics = sorted(set(required_metrics) - set(data.columns))
        if missing_metrics:
            raise ValueError(
                f"{perf_profile} PMU data is missing columns: {', '.join(missing_metrics)}"
            )
        invalid_metrics = profile_data[required_metrics].isna().any(axis=1)
        if invalid_metrics.any():
            raise ValueError(
                f"Found {int(invalid_metrics.sum())} invalid {perf_profile} PMU rows."
            )
    data["epoch_display"] = data["epoch"].astype(int) + 1
    data["batch_idx"] = data["batch_idx"].astype(int)
    data["layer_index"] = data["layer_index"].astype(int)
    return data, pd.DataFrame(inventory)


def first_value(frame: pd.DataFrame, column: str) -> object:
    values = frame[column].dropna()
    return "" if values.empty else values.iloc[0]


def validate_layout(data: pd.DataFrame) -> None:
    grouped = data.groupby(
        [
            "perf_profile",
            "device_id",
            "scenario",
            "experiment_mode",
            "trial_id",
            "phase",
            "epoch_display",
            "batch_idx",
        ],
        observed=True,
    ).size()
    wrong = grouped[grouped != 15]
    if not wrong.empty:
        examples = wrong.head(10).to_dict()
        raise ValueError(
            "Every phase/batch must contain 15 leaf-layer rows; "
            f"found mismatches: {examples}"
        )


def build_layer_summary(
    data: pd.DataFrame,
    additive_metrics: Dict[str, str],
    raw_layer_metrics: Dict[str, str],
) -> pd.DataFrame:
    columns = [
        "device_id",
        "trial_id",
        "scenario",
        "experiment_mode",
        "phase",
        "epoch_display",
        "batch_idx",
        "layer_index",
        "layer_name",
        "layer_type",
        *additive_metrics,
        *raw_layer_metrics,
    ]
    return data[columns].sort_values(
        [
            "device_id",
            "scenario",
            "experiment_mode",
            "trial_id",
            "phase",
            "epoch_display",
            "batch_idx",
            "layer_index",
        ]
    )


def build_raw_layer_epoch_statistics(
    data: pd.DataFrame, raw_layer_metrics: Dict[str, str]
) -> pd.DataFrame:
    id_columns = [
        "device_id",
        "scenario",
        "experiment_mode",
        "phase",
        "epoch_display",
        "layer_index",
        "layer_name",
        "layer_type",
    ]
    long_data = data.melt(
        id_vars=id_columns,
        value_vars=list(raw_layer_metrics),
        var_name="metric",
        value_name="value",
    )
    return (
        long_data.groupby([*id_columns, "metric"], observed=True)["value"]
        .agg(["mean", "std", "median", "min", "max", "count"])
        .reset_index()
        .sort_values(
            [
                "device_id",
                "experiment_mode",
                "phase",
                "metric",
                "layer_index",
                "scenario",
                "epoch_display",
            ]
        )
    )


def build_batch_totals(
    data: pd.DataFrame, additive_metrics: Dict[str, str], perf_preset: str
) -> pd.DataFrame:
    group_columns = [
        "device_id",
        "trial_id",
        "scenario",
        "experiment_mode",
        "phase",
        "epoch_display",
        "batch_idx",
    ]
    totals = (
        data.groupby(group_columns, observed=True)[list(additive_metrics)]
        .sum(min_count=1)
        .reset_index()
    )
    if perf_preset == "basic":
        totals["ipc"] = safe_ratio(totals["perf_instructions"], totals["perf_cycles"])
        totals["branch_miss_percent"] = 100.0 * safe_ratio(
            totals["perf_branch_misses"], totals["perf_branches"]
        )
        totals["l1d_load_miss_percent"] = 100.0 * safe_ratio(
            totals["perf_l1_dcache_load_misses"], totals["perf_l1_dcache_loads"]
        )
    elif perf_preset == "translation":
        totals["l1i_cache_refill_percent"] = 100.0 * safe_ratio(
            totals["perf_arm_l1i_cache_refill"],
            totals["perf_arm_l1i_cache_access"],
        )
        totals["itlb_refill_percent"] = 100.0 * safe_ratio(
            totals["perf_arm_itlb_refill"], totals["perf_arm_itlb_access"]
        )
        totals["dtlb_refills_per_1k_ld_spec"] = 1000.0 * safe_ratio(
            totals["perf_arm_dtlb_load_refill"], totals["perf_arm_ld_spec"]
        )
    elif perf_preset == "translation_x86":
        totals["itlb_misses_per_1k_instructions"] = 1000.0 * safe_ratio(
            totals["perf_itlb_load_misses"], totals["perf_instructions"]
        )
        totals["l1i_misses_per_1k_instructions"] = 1000.0 * safe_ratio(
            totals["perf_l1_icache_load_misses"], totals["perf_instructions"]
        )
        totals["dtlb_misses_per_1k_retired_loads"] = 1000.0 * safe_ratio(
            totals["perf_dtlb_load_misses"],
            totals["perf_mem_inst_retired_all_loads"],
        )
    elif perf_preset == "dtlb_x86":
        totals["dtlb_load_miss_percent"] = 100.0 * safe_ratio(
            totals["perf_dtlb_load_misses"], totals["perf_dtlb_loads"]
        )
        totals["dtlb_store_miss_percent"] = 100.0 * safe_ratio(
            totals["perf_dtlb_store_misses"], totals["perf_dtlb_stores"]
        )
    elif perf_preset == "dtlb_jetson":
        totals["dtlb_load_miss_percent"] = 100.0 * safe_ratio(
            totals["perf_dtlb_load_misses"], totals["perf_dtlb_loads"]
        )
    elif perf_preset == "dtlb_rpi":
        pass
    else:
        raise ValueError(f"Unsupported PMU preset: {perf_preset}")
    return totals


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def aggregate_trials(batch_totals: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        batch_totals.groupby(
            [
                "device_id",
                "scenario",
                "experiment_mode",
                "phase",
                "epoch_display",
                "batch_idx",
            ],
            observed=True,
        )[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


def plot_batch_epoch_grid(
    batch_totals: pd.DataFrame,
    *,
    device_id: str,
    phase: str,
    metric: str,
    plot_metrics: Dict[str, str],
    output_dir: Path,
) -> Optional[Path]:
    summary = aggregate_trials(batch_totals, metric)
    summary = summary[(summary["device_id"] == device_id) & (summary["phase"] == phase)]
    batches = sorted(int(value) for value in summary["batch_idx"].unique())
    if not batches:
        raise ValueError(f"No batch data for device={device_id}, phase={phase}")
    if len(batches) > 16:
        raise ValueError(f"The fixed grid supports at most 16 batches; got {len(batches)}")

    finite_means = summary["mean"].to_numpy(dtype=float)
    finite_means = finite_means[np.isfinite(finite_means)]
    if finite_means.size == 0:
        print(
            f"Skipping unavailable metric: device={device_id} phase={phase} "
            f"metric={metric} (no finite values)"
        )
        return None
    y_min = float(finite_means.min())
    y_max = float(finite_means.max())
    y_padding = max((y_max - y_min) * 0.06, abs(y_max) * 0.01, 1e-12)

    # Matplotlib 3.11 can produce a NaN tick-space calculation when 32 axes
    # share a scientific-notation y-axis, so apply identical limits manually.
    fig, axes = plt.subplots(4, 8, figsize=(28, 14), sharex=True, sharey=False)
    for mode_index, mode in enumerate(("train", "frozen_replay")):
        for position in range(16):
            row = position // 4
            column = position % 4 + 4 * mode_index
            axis = axes[row, column]
            if position >= len(batches):
                axis.set_visible(False)
                continue
            batch_idx = batches[position]
            for scenario in SCENARIO_LABELS:
                trace = summary[
                    (summary["experiment_mode"] == mode)
                    & (summary["scenario"] == scenario)
                    & (summary["batch_idx"] == batch_idx)
                ].sort_values("epoch_display")
                if trace.empty:
                    continue
                x = trace["epoch_display"].to_numpy(dtype=float)
                mean = trace["mean"].to_numpy(dtype=float)
                std = trace["std"].fillna(0.0).to_numpy(dtype=float)
                axis.plot(
                    x,
                    mean,
                    color=SCENARIO_COLORS[scenario],
                    label=SCENARIO_LABELS[scenario],
                    linewidth=1.6,
                    marker="o",
                    markersize=2.7,
                )
                if np.any(std > 0):
                    axis.fill_between(
                        x,
                        mean - std,
                        mean + std,
                        color=SCENARIO_COLORS[scenario],
                        alpha=0.14,
                        linewidth=0,
                    )
            axis.set_title(f"{MODE_LABELS[mode]} - batch {batch_idx}", fontsize=8)
            axis.grid(True, alpha=0.25)
            axis.set_ylim(y_min - y_padding, y_max + y_padding)
            axis.tick_params(labelsize=7)
            axis.xaxis.set_ticks([1, 5, 10, 15])
            axis.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            if row == 3:
                axis.set_xlabel("Epoch", fontsize=8)
            if column in (0, 4):
                axis.set_ylabel(plot_metrics[metric], fontsize=8)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.972),
        )
    fig.suptitle(
        f"{device_id}: {phase} - {plot_metrics[metric]} by batch",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"batch_epoch_{phase}_{metric}.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_raw_layer_epoch_grid(
    statistics: pd.DataFrame,
    *,
    device_id: str,
    experiment_mode: str,
    metric: str,
    raw_layer_metrics: Dict[str, str],
    output_dir: Path,
) -> Path:
    selected = statistics[
        (statistics["device_id"] == device_id)
        & (statistics["experiment_mode"] == experiment_mode)
        & (statistics["metric"] == metric)
    ]
    layers = (
        selected[["layer_index", "layer_name", "layer_type"]]
        .drop_duplicates()
        .sort_values("layer_index")
    )
    if layers.empty:
        raise ValueError(
            f"No raw layer data for device={device_id}, "
            f"mode={experiment_mode}, metric={metric}."
        )
    n_layers = len(layers)
    fig, axes = plt.subplots(
        n_layers,
        2,
        figsize=(16, max(4.0, 2.35 * n_layers)),
        squeeze=False,
        sharex=True,
        sharey=False,
    )
    for row_index, layer in layers.reset_index(drop=True).iterrows():
        layer_index = int(layer["layer_index"])
        layer_name = str(layer["layer_name"])
        layer_type = str(layer["layer_type"])
        layer_data = selected[selected["layer_index"] == layer_index]

        for column_index, phase in enumerate(("forward", "backward")):
            axis = axes[row_index, column_index]
            phase_data = layer_data[layer_data["phase"] == phase]
            phase_means = phase_data["mean"].to_numpy(dtype=float)
            phase_stds = phase_data["std"].fillna(0.0).to_numpy(dtype=float)
            finite = np.isfinite(phase_means) & np.isfinite(phase_stds)
            if not finite.any():
                axis.set_visible(False)
                continue
            phase_means = phase_means[finite]
            phase_stds = phase_stds[finite]
            y_min = max(0.0, float(np.min(phase_means - phase_stds)))
            y_max = float(np.max(phase_means + phase_stds))
            y_padding = max((y_max - y_min) * 0.06, abs(y_max) * 0.01, 1e-12)
            for scenario in SCENARIO_LABELS:
                trace = phase_data[phase_data["scenario"] == scenario].sort_values(
                    "epoch_display"
                )
                if trace.empty:
                    continue
                x = trace["epoch_display"].to_numpy(dtype=float)
                mean = trace["mean"].to_numpy(dtype=float)
                std = trace["std"].fillna(0.0).to_numpy(dtype=float)
                axis.plot(
                    x,
                    mean,
                    color=SCENARIO_COLORS[scenario],
                    label=SCENARIO_LABELS[scenario],
                    linewidth=1.4,
                    marker="o",
                    markersize=2.2,
                )
                if np.any(std > 0):
                    axis.fill_between(
                        x,
                        mean - std,
                        mean + std,
                        color=SCENARIO_COLORS[scenario],
                        alpha=0.12,
                        linewidth=0,
                    )
            axis.set_title(
                f"{phase.capitalize()} | layer {layer_index}: "
                f"{layer_name} ({layer_type})",
                fontsize=9,
            )
            axis.set_ylim(y_min - y_padding, y_max + y_padding)
            axis.set_xticks([1, 5, 10, 15])
            axis.grid(True, alpha=0.25)
            axis.tick_params(labelsize=7)
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
            axis.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            if row_index == n_layers - 1:
                axis.set_xlabel("Epoch", fontsize=8)
            if column_index == 0:
                axis.set_ylabel(raw_layer_metrics[metric], fontsize=8)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(5, len(labels)),
            frameon=False,
            bbox_to_anchor=(0.5, 0.972),
        )
    fig.suptitle(
        f"{device_id}: {MODE_LABELS[experiment_mode]} - "
        f"{raw_layer_metrics[metric]} per layer (batch mean +/- std)",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"raw_layer_epoch_{experiment_mode}_{metric}.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def save_counter_quality(data: pd.DataFrame, output_path: Path) -> None:
    columns = [column for column in data.columns if column.endswith("_running_pct")]
    rows = []
    for (perf_profile, device, scenario, mode, phase), group in data.groupby(
        ["perf_profile", "device_id", "scenario", "experiment_mode", "phase"],
        observed=True,
    ):
        for column in columns:
            values = pd.to_numeric(group[column], errors="coerce")
            if not values.notna().any():
                continue
            rows.append(
                {
                    "perf_profile": perf_profile,
                    "device_id": device,
                    "scenario": scenario,
                    "experiment_mode": mode,
                    "phase": phase,
                    "event": column.removeprefix("perf_").removesuffix("_running_pct"),
                    "minimum_running_percent": values.min(),
                    "mean_running_percent": values.mean(),
                    "rows": values.notna().sum(),
                }
            )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).resolve().parent
    parser.add_argument("--input-dir", type=Path, default=script_dir / "collected_logs")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "visualization")
    parser.add_argument(
        "--perf-preset",
        choices=("auto", "basic", "translation", "dtlb"),
        default="auto",
        help=(
            "Select PMU logs. 'translation' includes both Arm and Intel event "
            "profiles. 'dtlb' includes the target-specific laptop, Jetson, "
            "and Raspberry Pi dTLB profiles. Defaults to auto."
        ),
    )
    args = parser.parse_args()

    paths = discover_layer_csvs(args.input_dir.resolve())
    data, inventory = load_layer_data(paths)
    available_profiles = sorted(str(value) for value in data["perf_profile"].unique())
    requested_profiles = {
        "auto": set(METRIC_CONFIGS),
        "basic": {"basic"},
        "translation": {"translation", "translation_x86"},
        "dtlb": {"dtlb_x86", "dtlb_jetson", "dtlb_rpi"},
    }[args.perf_preset]
    data = data[data["perf_profile"].isin(requested_profiles)].copy()
    inventory = inventory[inventory["perf_profile"].isin(requested_profiles)].copy()
    if data.empty:
        raise ValueError(
            f"No {args.perf_preset} PMU rows were found. "
            f"Available profiles: {available_profiles}"
        )
    validate_layout(data)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory.to_csv(output_dir / "input_inventory.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    save_counter_quality(data, output_dir / "perf_counter_running_quality.csv")

    generated: List[Path] = []
    layer_summaries: List[pd.DataFrame] = []
    batch_total_frames: List[pd.DataFrame] = []
    raw_statistic_frames: List[pd.DataFrame] = []
    detected_profiles = sorted(str(value) for value in data["perf_profile"].unique())
    for perf_profile in detected_profiles:
        profile_data = data[data["perf_profile"] == perf_profile].copy()
        additive_metrics, derived_metrics, raw_layer_metrics = METRIC_CONFIGS[perf_profile]
        plot_metrics = {**additive_metrics, **derived_metrics}
        layer_summary = build_layer_summary(
            profile_data, additive_metrics, raw_layer_metrics
        )
        batch_totals = build_batch_totals(
            profile_data, additive_metrics, perf_profile
        )
        raw_layer_statistics = build_raw_layer_epoch_statistics(
            profile_data, raw_layer_metrics
        )
        for frame in (layer_summary, batch_totals, raw_layer_statistics):
            frame.insert(0, "perf_profile", perf_profile)
        layer_summaries.append(layer_summary)
        batch_total_frames.append(batch_totals)
        raw_statistic_frames.append(raw_layer_statistics)

        for device_id in sorted(
            str(value) for value in batch_totals["device_id"].unique()
        ):
            device_dir = output_dir / device_id / perf_profile
            for phase in ("forward", "backward"):
                for metric in plot_metrics:
                    path = plot_batch_epoch_grid(
                        batch_totals,
                        device_id=device_id,
                        phase=phase,
                        metric=metric,
                        plot_metrics=plot_metrics,
                        output_dir=device_dir,
                    )
                    if path is not None:
                        generated.append(path)
            for experiment_mode in MODE_LABELS:
                for metric in raw_layer_metrics:
                    generated.append(
                        plot_raw_layer_epoch_grid(
                            raw_layer_statistics,
                            device_id=device_id,
                            experiment_mode=experiment_mode,
                            metric=metric,
                            raw_layer_metrics=raw_layer_metrics,
                            output_dir=device_dir / "raw_layers",
                        )
                    )

    pd.concat(layer_summaries, ignore_index=True).to_csv(
        output_dir / "layer_batch_epoch_metrics.csv", index=False
    )
    pd.concat(batch_total_frames, ignore_index=True).to_csv(
        output_dir / "batch_epoch_phase_totals.csv", index=False
    )
    pd.concat(raw_statistic_frames, ignore_index=True).to_csv(
        output_dir / "raw_layer_epoch_statistics.csv", index=False
    )

    print(f"Loaded {len(data)} layer rows from {len(paths)} CSV files.")
    print(f"Requested PMU preset: {args.perf_preset}")
    print(f"Detected PMU profiles: {','.join(detected_profiles)}")
    print(f"Saved {len(generated)} figures under {output_dir}.")
    print(f"Batch totals: {output_dir / 'batch_epoch_phase_totals.csv'}")


if __name__ == "__main__":
    main()
