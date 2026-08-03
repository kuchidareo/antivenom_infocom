#!/usr/bin/env python3
"""Compare forward-phase PMU metrics with clean across ten Raspberry Pi devices.

Each device is one independent paired unit. The three trials within a device are
aggregated before inference, so trials, epochs, batches, and perf intervals are
not incorrectly treated as independent replicates.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-forward-metric-significance")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "collected_logs"
DEFAULT_OUTPUT = SCRIPT_DIR / "forward_epochwise_metric_significance"
DEFAULT_DEVICES = tuple(f"192.168.0.{host}" for host in range(112, 122))
STAGE_PARTS = ("full", "phase1_cifar10_cnn")
BASELINE = "clean"
TARGETS = (
    "moderate_augmentation",
    "strong_augmentation",
    "availability_shortcuts",
    "badsampling",
    "non_iid",
)
DISPLAY_NAMES = {
    "moderate_augmentation": "Moderate augmentation",
    "strong_augmentation": "Strong augmentation",
    "availability_shortcuts": "Availability shortcuts",
    "badsampling": "BadSampler",
    "non_iid": "Non-IID",
}
COUNTERS = {
    "cycles": "perf_cycles",
    "task_clock": "perf_task_clock",
    "context_switches": "perf_context_switches",
    "cpu_migrations": "perf_cpu_migrations",
    "page_faults": "perf_page_faults",
    "branch_misses": "perf_branch_misses",
    "l1d_read_access": "perf_l1d_cache_rd",
    "l1d_read_refill": "perf_l1d_cache_refill_rd",
    "l1d_write_access": "perf_l1d_cache_wr",
    "l1d_write_refill": "perf_l1d_cache_refill_wr",
    "l2d_read_access": "perf_l2d_cache_rd",
    "l2d_read_refill": "perf_l2d_cache_refill_rd",
    "l2d_write_access": "perf_l2d_cache_wr",
    "l2d_write_refill": "perf_l2d_cache_refill_wr",
    "bus_read_access": "perf_bus_access_rd",
    "bus_write_access": "perf_bus_access_wr",
    "memory_access": "perf_mem_access",
    "ase_spec": "perf_ase_spec",
    "vfp_spec": "perf_vfp_spec",
    "inst_spec": "perf_inst_spec",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--devices", nargs="+", default=DEFAULT_DEVICES)
    parser.add_argument("--epochs", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--coverage-threshold", type=float, default=20.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf"), default=("png",))
    args = parser.parse_args()
    if not args.epochs or any(epoch < 0 for epoch in args.epochs):
        parser.error("--epochs must contain non-negative values")
    if not 0 <= args.coverage_threshold <= 100:
        parser.error("--coverage-threshold must be between 0 and 100")
    if not 0 < args.alpha < 1:
        parser.error("--alpha must be between 0 and 1")
    return args


def partial_batches(metrics_path: Path) -> set[tuple[int, int]]:
    if not metrics_path.is_file():
        return set()
    frame = pd.read_csv(metrics_path, low_memory=False)
    required = {"epoch", "batch_idx", "num_examples"}
    if not required <= set(frame.columns):
        return set()
    if "metric_event" in frame:
        frame = frame[frame["metric_event"].astype(str).eq("train_batch")]
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=list(required))
    output: set[tuple[int, int]] = set()
    for epoch, group in frame.groupby("epoch"):
        full_batch = group["num_examples"].max()
        for row in group[group["num_examples"].lt(full_batch)].itertuples():
            output.add((int(epoch), int(row.batch_idx)))
    return output


def load_corrected_forward(path: Path) -> tuple[pd.DataFrame, str, str]:
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty or "trial_id" not in frame:
        raise ValueError("empty or metadata-only perf file")
    frame = frame.sort_values("perf_elapsed_sec", kind="stable").reset_index(drop=True)
    for column in ("phase", "epoch", "batch_idx", "round"):
        if column in frame:
            frame[f"corrected_{column}"] = frame[column].shift(1)

    epoch = pd.to_numeric(frame["corrected_epoch"], errors="coerce")
    batch = pd.to_numeric(frame["corrected_batch_idx"], errors="coerce")
    mask = (
        frame["corrected_phase"].astype(str).str.lower().eq("forward")
        & frame["perf_status"].astype(str).str.lower().eq("ok")
        & epoch.notna()
        & batch.notna()
    )
    prefix = path.name.removesuffix("_perf.csv")
    partial = partial_batches(path.with_name(f"{prefix}_metrics.csv"))
    if partial:
        keys = pd.Series(
            list(zip(epoch.fillna(-1).astype(int), batch.fillna(-1).astype(int))),
            index=frame.index,
        )
        mask &= ~keys.isin(partial)
    selected = frame.loc[mask].copy()
    trials = frame["trial_id"].dropna().astype(str).unique()
    devices = frame["device_id"].dropna().astype(str).unique()
    if len(trials) != 1 or len(devices) != 1:
        raise ValueError(f"expected one trial/device, got trials={trials}, devices={devices}")
    return selected, str(devices[0]), str(trials[0])


def aggregate_run(
    path: Path,
    condition: str,
    epochs: set[int],
    threshold: float,
) -> list[dict[str, object]]:
    selected, device_id, trial_id = load_corrected_forward(path)
    selected_epoch = pd.to_numeric(selected["corrected_epoch"], errors="coerce")
    instructions = pd.to_numeric(selected["perf_instructions"], errors="coerce")
    instruction_coverage = pd.to_numeric(
        selected["perf_instructions_enabled_pct"], errors="coerce"
    )
    rows: list[dict[str, object]] = []
    for epoch in sorted(epochs):
        in_epoch = selected_epoch.eq(epoch)
        instruction_valid = (
            in_epoch
            & instructions.gt(0)
            & np.isfinite(instructions)
            & instruction_coverage.ge(threshold)
        )
        instruction_total = float(instructions[instruction_valid].sum())
        if instruction_total > 0:
            rows.append({
                "device_id": device_id,
                "condition": condition,
                "trial_id": trial_id,
                "epoch": epoch,
                "counter": "instructions_total",
                "counter_total": instruction_total,
                "instruction_total": instruction_total,
                "value": instruction_total,
                "unit": "instructions per forward epoch",
                "retained_rows": int(instruction_valid.sum()),
                "counter_running_pct_mean": float(
                    instruction_coverage[instruction_valid].mean()
                ),
            })
        for counter, column in COUNTERS.items():
            coverage_column = f"{column}_enabled_pct"
            if column not in selected or coverage_column not in selected:
                continue
            values = pd.to_numeric(selected[column], errors="coerce")
            coverage = pd.to_numeric(selected[coverage_column], errors="coerce")
            valid = (
                instruction_valid
                & values.ge(0)
                & np.isfinite(values)
                & coverage.ge(threshold)
            )
            denominator = float(instructions[valid].sum())
            numerator = float(values[valid].sum())
            if not valid.any() or denominator <= 0:
                continue
            rows.append({
                "device_id": device_id,
                "condition": condition,
                "trial_id": trial_id,
                "epoch": epoch,
                "counter": counter,
                "counter_total": numerator,
                "instruction_total": denominator,
                "value": numerator / denominator,
                "unit": "counter / instruction",
                "retained_rows": int(valid.sum()),
                "counter_running_pct_mean": float(coverage[valid].mean()),
            })
    return rows


def collect(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    expected_conditions = (BASELINE, *TARGETS)
    for device_index, device_id in enumerate(args.devices, start=1):
        stage = args.input_dir.resolve() / device_id / Path(*STAGE_PARTS)
        if not stage.is_dir():
            warnings.warn(f"Skipping {device_id}: missing {stage}")
            continue
        print(f"[device {device_index}/{len(args.devices)}] {device_id}", flush=True)
        for condition in expected_conditions:
            files = sorted((stage / condition).glob("*_perf.csv"))
            valid_runs = 0
            for path in files:
                try:
                    rows.extend(
                        aggregate_run(
                            path, condition, set(args.epochs), args.coverage_threshold
                        )
                    )
                    valid_runs += 1
                except ValueError as exc:
                    warnings.warn(f"Skipping {path}: {exc}")
            print(f"  {condition}: {valid_runs} valid runs", flush=True)
    if not rows:
        raise RuntimeError("No valid forward PMU observations were found")
    return pd.DataFrame(rows)


def aggregate_trials(values: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["device_id", "condition", "epoch", "counter"]
    for key, group in values.groupby(keys, sort=False):
        counter = str(key[-1])
        if counter == "instructions_total":
            value = float(group["value"].mean())
            counter_total = float(group["counter_total"].mean())
            instruction_total = float(group["instruction_total"].mean())
        else:
            counter_total = float(group["counter_total"].sum())
            instruction_total = float(group["instruction_total"].sum())
            value = counter_total / instruction_total if instruction_total > 0 else math.nan
        rows.append({
            **dict(zip(keys, key)),
            "trials": int(group["trial_id"].nunique()),
            "counter_total": counter_total,
            "instruction_total": instruction_total,
            "value": value,
            "unit": str(group["unit"].iloc[0]),
            "retained_rows": int(group["retained_rows"].sum()),
            "counter_running_pct_mean": float(
                np.average(
                    group["counter_running_pct_mean"],
                    weights=np.maximum(group["retained_rows"], 1),
                )
            ),
        })
    return pd.DataFrame(rows)


def aggregate_epochs(device_epochs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["device_id", "condition", "counter"]
    for key, group in device_epochs.groupby(keys, sort=False):
        counter = str(key[-1])
        if counter == "instructions_total":
            value = float(group["value"].sum())
        else:
            denominator = float(group["instruction_total"].sum())
            value = float(group["counter_total"].sum()) / denominator if denominator > 0 else math.nan
        rows.append({
            **dict(zip(keys, key)),
            "epochs": int(group["epoch"].nunique()),
            "value": value,
            "unit": str(group["unit"].iloc[0]),
        })
    return pd.DataFrame(rows)


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    observed = abs(float(differences.mean()))
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(differences))))
    null = np.abs(signs @ differences / len(differences))
    return float(np.count_nonzero(null >= observed - 1e-15) / len(null))


def bh(values: pd.Series) -> np.ndarray:
    p = values.to_numpy(dtype=float)
    output = np.full(p.shape, np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return output
    selected = p[valid]
    order = np.argsort(selected)
    ranked = selected[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    output[valid] = restored
    return output


def compare_group(group: pd.DataFrame, condition: str) -> dict[str, object] | None:
    wide = group.pivot(index="device_id", columns="condition", values="value")
    if BASELINE not in wide or condition not in wide:
        return None
    wide = wide[[BASELINE, condition]].dropna()
    if len(wide) < 2:
        return None
    clean = wide[BASELINE].to_numpy(dtype=float)
    target = wide[condition].to_numpy(dtype=float)
    if np.all(clean > 0) and np.all(target > 0):
        differences = np.log(target / clean)
        effect = 100.0 * np.expm1(differences.mean())
        test_scale = "paired log ratio"
    else:
        differences = target - clean
        effect = 100.0 * (target.mean() / clean.mean() - 1.0) if clean.mean() else math.nan
        test_scale = "paired raw difference"
    return {
        "condition": condition,
        "paired_devices": len(wide),
        "clean_mean": float(clean.mean()),
        "clean_std": float(clean.std(ddof=1)),
        "condition_mean": float(target.mean()),
        "condition_std": float(target.std(ddof=1)),
        "paired_change_pct": float(effect),
        "devices_increased": int(np.count_nonzero(target > clean)),
        "devices_decreased": int(np.count_nonzero(target < clean)),
        "test_scale": test_scale,
        "paired_exact_p": exact_sign_flip_pvalue(differences),
    }


def epoch_tests(device_epochs: pd.DataFrame, alpha: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for condition in TARGETS:
        selected = device_epochs[device_epochs["condition"].isin((BASELINE, condition))]
        for (epoch, counter), group in selected.groupby(["epoch", "counter"], sort=False):
            result = compare_group(group, condition)
            if result is None:
                continue
            rows.append({
                **result,
                "epoch": int(epoch),
                "counter": str(counter),
                "unit": str(group["unit"].iloc[0]),
            })
    tests = pd.DataFrame(rows)
    tests["fdr_q_all_tests"] = bh(tests["paired_exact_p"])
    condition_adjusted = []
    for _, group in tests.groupby("condition", sort=False):
        condition_adjusted.append(
            pd.Series(bh(group["paired_exact_p"]), index=group.index)
        )
    tests["fdr_q_within_condition"] = pd.concat(condition_adjusted).sort_index()
    epoch_adjusted = []
    for _, group in tests.groupby(["condition", "epoch"], sort=False):
        epoch_adjusted.append(pd.Series(bh(group["paired_exact_p"]), index=group.index))
    tests["fdr_q_within_condition_epoch"] = pd.concat(epoch_adjusted).sort_index()
    tests["significant_all_tests"] = tests["fdr_q_all_tests"].le(alpha)
    tests["significant_within_condition"] = tests["fdr_q_within_condition"].le(alpha)
    tests["significant_within_condition_epoch"] = tests[
        "fdr_q_within_condition_epoch"
    ].le(alpha)
    return tests


def overall_tests(device_overall: pd.DataFrame, alpha: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for condition in TARGETS:
        selected = device_overall[device_overall["condition"].isin((BASELINE, condition))]
        for counter, group in selected.groupby("counter", sort=False):
            result = compare_group(group, condition)
            if result is None:
                continue
            rows.append({
                **result,
                "counter": str(counter),
                "unit": str(group["unit"].iloc[0]),
            })
    tests = pd.DataFrame(rows)
    tests["fdr_q_all_tests"] = bh(tests["paired_exact_p"])
    adjusted = []
    for _, group in tests.groupby("condition", sort=False):
        adjusted.append(pd.Series(bh(group["paired_exact_p"]), index=group.index))
    tests["fdr_q_within_condition"] = pd.concat(adjusted).sort_index()
    tests["significant_all_tests"] = tests["fdr_q_all_tests"].le(alpha)
    tests["significant_within_condition"] = tests["fdr_q_within_condition"].le(alpha)
    return tests


def save_figure(figure: plt.Figure, prefix: Path, formats: list[str], dpi: int) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        kwargs = {"dpi": dpi} if extension == "png" else {}
        figure.savefig(prefix.with_suffix(f".{extension}"), bbox_inches="tight", **kwargs)
    plt.close(figure)


def heatmap_limit(values: np.ndarray) -> float:
    finite = np.abs(values[np.isfinite(values)])
    if not len(finite):
        return 1.0
    return max(float(np.quantile(finite, 0.95)), np.finfo(float).eps)


def plot_overall(tests: pd.DataFrame, output: Path, formats: list[str], dpi: int) -> None:
    effects = tests.pivot(index="counter", columns="condition", values="paired_change_pct")
    effects = effects.reindex(columns=TARGETS)
    q_values = tests.pivot(index="counter", columns="condition", values="fdr_q_all_tests").reindex(
        index=effects.index, columns=effects.columns
    )
    values = effects.to_numpy(dtype=float)
    limit = heatmap_limit(values)
    figure, axis = plt.subplots(figsize=(12.5, max(6.5, 0.38 * len(effects))))
    image = axis.imshow(
        np.clip(values, -limit, limit), aspect="auto", cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
    )
    axis.set_xticks(np.arange(len(effects.columns)), [DISPLAY_NAMES[c] for c in effects.columns], rotation=25, ha="right")
    axis.set_yticks(np.arange(len(effects.index)), effects.index)
    for row in range(len(effects.index)):
        for column in range(len(effects.columns)):
            value = values[row, column]
            if not np.isfinite(value):
                continue
            star = "*" if float(q_values.iloc[row, column]) <= 0.05 else ""
            axis.text(column, row, f"{value:+.1f}%{star}", ha="center", va="center", fontsize=7)
    axis.set_title(
        "Forward PMU change from clean across all 10 epochs\n"
        "Device-paired effects; * = BH q <= 0.05 across all condition/metric tests"
    )
    figure.colorbar(image, ax=axis, label="Paired geometric change from clean (%)")
    figure.tight_layout()
    save_figure(figure, output / "overall_clean_comparison_heatmap", formats, dpi)


def plot_epoch_condition(
    tests: pd.DataFrame,
    condition: str,
    output: Path,
    formats: list[str],
    dpi: int,
) -> None:
    selected = tests[tests["condition"].eq(condition)]
    effects = selected.pivot(index="counter", columns="epoch", values="paired_change_pct")
    q_values = selected.pivot(index="counter", columns="epoch", values="fdr_q_all_tests").reindex(
        index=effects.index, columns=effects.columns
    )
    values = effects.to_numpy(dtype=float)
    limit = heatmap_limit(values)
    figure, axis = plt.subplots(figsize=(13.0, max(6.5, 0.36 * len(effects))))
    image = axis.imshow(
        np.clip(values, -limit, limit), aspect="auto", cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
    )
    axis.set_xticks(np.arange(len(effects.columns)), effects.columns)
    axis.set_yticks(np.arange(len(effects.index)), effects.index)
    significant = q_values.to_numpy(dtype=float) <= 0.05
    yy, xx = np.where(significant)
    if len(xx):
        axis.scatter(xx, yy, marker="*", s=22, color="black")
    axis.set_xlabel("Epoch")
    axis.set_title(
        f"{DISPLAY_NAMES[condition]} vs clean: forward PMU effect by epoch\n"
        "Stars indicate BH q <= 0.05 across all epoch/condition/metric tests"
    )
    figure.colorbar(image, ax=axis, label="Paired geometric change from clean (%)")
    figure.tight_layout()
    save_figure(figure, output / f"epoch_effect_heatmap_{condition}", formats, dpi)


def write_significance_summary(tests: pd.DataFrame, path: Path, alpha: float) -> None:
    lines = [
        "Forward metric significance summary",
        f"Primary criterion: BH q <= {alpha:g} across all {len(tests)} "
        "condition-by-metric tests.",
        "Each device is one paired unit; three trials and ten epochs are aggregated within device.",
        "",
    ]
    for condition in TARGETS:
        selected = tests[
            tests["condition"].eq(condition) & tests["significant_all_tests"]
        ].sort_values("fdr_q_all_tests")
        lines.append(DISPLAY_NAMES[condition])
        if selected.empty:
            lines.append("  No metrics significant after global BH correction.")
        else:
            for row in selected.itertuples():
                lines.append(
                    f"  {row.counter}: change={row.paired_change_pct:+.2f}%, "
                    f"p={row.paired_exact_p:.4g}, q={row.fdr_q_all_tests:.4g}, "
                    f"devices={row.paired_devices}"
                )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots = args.output_dir / "plots"

    run_values = collect(args)
    device_epochs = aggregate_trials(run_values)
    device_overall = aggregate_epochs(device_epochs)
    per_epoch = epoch_tests(device_epochs, args.alpha)
    overall = overall_tests(device_overall, args.alpha)

    run_values.to_csv(args.output_dir / "trial_epoch_forward_metrics.csv", index=False)
    device_epochs.to_csv(args.output_dir / "device_epoch_forward_metrics.csv", index=False)
    device_overall.to_csv(args.output_dir / "device_overall_forward_metrics.csv", index=False)
    per_epoch.to_csv(args.output_dir / "clean_comparison_by_epoch.csv", index=False)
    overall.to_csv(args.output_dir / "clean_comparison_overall.csv", index=False)
    write_significance_summary(overall, args.output_dir / "significant_metrics_summary.txt", args.alpha)

    plot_overall(overall, plots, list(args.formats), args.dpi)
    for condition in TARGETS:
        plot_epoch_condition(per_epoch, condition, plots, list(args.formats), args.dpi)

    summary = {
        "devices_requested": list(args.devices),
        "devices_observed": sorted(run_values["device_id"].unique()),
        "conditions": [BASELINE, *TARGETS],
        "epochs": list(args.epochs),
        "coverage_threshold": args.coverage_threshold,
        "phase": "forward",
        "phase_annotation_correction": "one perf interval backward shift",
        "independent_unit": "device",
        "within_device_aggregation": "three trials, then ten epochs for overall tests",
        "overall_tests": len(overall),
        "overall_significant_global_fdr": int(overall["significant_all_tests"].sum()),
        "epoch_tests": len(per_epoch),
        "epoch_significant_global_fdr": int(per_epoch["significant_all_tests"].sum()),
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Wrote results to {args.output_dir}")
    print((args.output_dir / "significant_metrics_summary.txt").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
