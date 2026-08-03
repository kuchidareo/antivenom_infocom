#!/usr/bin/env python3
"""Compare batch-calibrated, instruction-aligned counter shapes.

The analysis separates each batch into:

* scale: total counter / total retired instructions; and
* shape: interval counter mass / total batch counter mass.

Shape profiles are nonnegative and constrained to integrate to one. PMU
running percentage filters and weights interval observations. Clean and target
runs are paired by trial, epoch, and batch index; statistical inference uses
the paired trial-level shape differences, not individual intervals or batches.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-batch-calibrated-shapes")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from execution_profiles import (
    COUNTER_ALIASES,
    DEFAULT_COUNTERS,
    RunFiles,
    build_overlap_matrix,
    discover_runs,
    parse_numeric,
    partial_batches,
    resolve_counter_columns,
    resolve_structural_columns,
    second_difference,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_COLLECTION = SCRIPT_DIR / "cache_0727_jetson_cpu_20_trials"
DEFAULT_INPUT = DEFAULT_COLLECTION / "192.168.0.141"
DEFAULT_OUTPUT = DEFAULT_COLLECTION / "batch_calibrated_shape_comparison"

ROLE_LABELS = {
    "strong_augmentation": "strong augmentation",
    "moderate_augmentation": "moderate augmentation",
    "availability_shortcut": "availability shortcut",
    "non_iid": "non-IID",
}
COLORS = {"clean": "#25282b", "target": "#c84c3a", "difference": "#236a9a"}


@dataclass
class LoadedRun:
    files: RunFiles
    frame: pd.DataFrame
    counters: dict[str, str | None]
    partial: set[tuple[int, int]]


@dataclass
class ShapeFit:
    values: np.ndarray
    success: bool
    message: str
    objective: float
    iterations: int
    rank: int
    condition_number: float
    residual_rmse: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--datasets", nargs="+", choices=("cifar10", "trashnet"), default=("cifar10",))
    parser.add_argument("--models", nargs="+", choices=("cnn", "vit"), default=("cnn", "vit"))
    parser.add_argument("--target", choices=tuple(ROLE_LABELS), default="strong_augmentation")
    parser.add_argument("--phase", choices=("forward", "backward"), default="forward")
    parser.add_argument("--epochs", nargs="+", type=int, default=list(range(10)))
    parser.add_argument(
        "--counters", nargs="+", choices=tuple(COUNTER_ALIASES), default=list(DEFAULT_COUNTERS)
    )
    parser.add_argument("--n-bins", type=int, default=4)
    parser.add_argument("--pmu-min-running", type=float, default=20.0)
    parser.add_argument("--min-instruction-mass", type=float, default=0.90)
    parser.add_argument("--edge-weight", type=float, default=0.5)
    parser.add_argument("--phase-label-lag", type=int, choices=(0, 1), default=1)
    parser.add_argument("--min-matched-batches", type=int, default=6)
    parser.add_argument("--min-paired-trials", type=int, default=8)
    parser.add_argument("--expected-trials", type=int, default=20)
    parser.add_argument("--shape-smoothness", type=float, default=0.25)
    parser.add_argument("--huber-delta", type=float, default=0.08)
    parser.add_argument("--optimizer-maxiter", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=4999)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--random-seed", type=int, default=260729)
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf"), default=("pdf",))
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.n_bins < 2:
        parser.error("--n-bins must be at least 2")
    if not 0 <= args.pmu_min_running <= 100:
        parser.error("--pmu-min-running must be in [0, 100]")
    if not 0 < args.min_instruction_mass <= 1:
        parser.error("--min-instruction-mass must be in (0, 1]")
    if not 0 < args.edge_weight <= 1:
        parser.error("--edge-weight must be in (0, 1]")
    if args.min_matched_batches < 2 or args.min_paired_trials < 2:
        parser.error("minimum batch/trial counts must be at least 2")
    if args.permutations < 99:
        parser.error("--permutations must be at least 99")
    if args.shape_smoothness < 0 or args.huber_delta <= 0:
        parser.error("shape smoothness must be nonnegative and Huber delta positive")
    return args


def resolve_input_dir(path: Path) -> Path:
    path = path.resolve()
    if path.is_dir() and any(path.rglob("*_perf.csv")):
        direct = [child for child in path.iterdir() if child.is_dir() and any(child.glob("*_perf.csv"))]
        if direct:
            return path
        devices = [child for child in path.iterdir() if child.is_dir() and any(child.rglob("*_perf.csv"))]
        if len(devices) == 1:
            return devices[0]
    raise FileNotFoundError(f"Could not resolve a device log directory below {path}")


def condition_name(dataset: str, model: str, role: str) -> str:
    if role == "clean":
        return f"{dataset}_iid" if model == "cnn" else f"{dataset}_vit"
    prefix = dataset if model == "cnn" else f"{dataset}_vit"
    return f"{prefix}_{role}"


def index_runs(runs: list[RunFiles]) -> dict[tuple[str, str], RunFiles]:
    result: dict[tuple[str, str], RunFiles] = {}
    for run in runs:
        key = (run.condition, run.trial_id)
        if key in result:
            warnings.warn(f"Duplicate run for {key}; retaining {run.perf_path.name}")
        result[key] = run
    return result


def running_column(counter_column: str, columns: pd.Index) -> str | None:
    # perf_logger.py's historical name is misleading: enabled_pct stores the
    # perf CSV percent-running field, while runtime_pct stores runtime ns.
    candidate = f"{counter_column}_enabled_pct"
    return candidate if candidate in columns else None


def load_run(files: RunFiles, phase_label_lag: int) -> LoadedRun:
    frame = pd.read_csv(files.perf_path, low_memory=False)
    structural = resolve_structural_columns(frame.columns)
    timestamp = structural.get("timestamp")
    if timestamp is None:
        raise ValueError(f"Missing timestamp column in {files.perf_path}")
    frame["__timestamp"] = parse_numeric(frame[timestamp])
    frame = frame.sort_values("__timestamp", kind="stable").reset_index(drop=True)
    for key in ("epoch", "batch_idx", "phase"):
        column = structural.get(key)
        if column is None:
            raise ValueError(f"Missing {key} column in {files.perf_path}")
        frame[f"__{key}"] = frame[column].shift(phase_label_lag)
    status = structural.get("status")
    frame["__status"] = frame[status] if status else "ok"
    return LoadedRun(
        files=files,
        frame=frame,
        counters=resolve_counter_columns(frame.columns),
        partial=partial_batches(files.metrics_path)[0],
    )


def batch_observations(
    run: LoadedRun,
    *,
    counter: str,
    epoch: int,
    phase: str,
    pmu_min_running: float,
    min_instruction_mass: float,
    edge_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    frame = run.frame
    instruction_column = run.counters.get("instructions")
    counter_column = run.counters.get(counter)
    if instruction_column is None or counter_column is None:
        return pd.DataFrame(), pd.DataFrame(), ["counter or instruction column unavailable"]
    instruction_running_column = running_column(instruction_column, frame.columns)
    counter_running_column = running_column(counter_column, frame.columns)
    if instruction_running_column is None or counter_running_column is None:
        return pd.DataFrame(), pd.DataFrame(), ["PMU running-percentage column unavailable"]

    selected = frame[
        parse_numeric(frame["__epoch"]).eq(epoch)
        & frame["__phase"].astype(str).str.lower().eq(phase.lower())
        & frame["__status"].astype(str).str.lower().eq("ok")
    ].copy()
    selected["__batch"] = parse_numeric(selected["__batch_idx"])
    selected["__instructions"] = parse_numeric(selected[instruction_column])
    selected["__instruction_running"] = parse_numeric(selected[instruction_running_column])
    selected["__counter"] = parse_numeric(selected[counter_column])
    selected["__counter_running"] = parse_numeric(selected[counter_running_column])

    observation_rows: list[dict[str, object]] = []
    total_rows: list[dict[str, object]] = []
    skipped: list[str] = []
    for batch_value, batch in selected.groupby("__batch", sort=True):
        if not np.isfinite(batch_value):
            continue
        batch_idx = int(batch_value)
        if (epoch, batch_idx) in run.partial:
            continue
        batch = batch.sort_values("__timestamp", kind="stable").copy()
        instruction_ok = (
            np.isfinite(batch["__instructions"])
            & batch["__instructions"].gt(0)
            & np.isfinite(batch["__instruction_running"])
            & batch["__instruction_running"].gt(0)
        )
        batch = batch.loc[instruction_ok].copy()
        if batch.empty:
            skipped.append(f"batch {batch_idx}: no valid instruction intervals")
            continue
        instructions = batch["__instructions"].to_numpy(dtype=float)
        instruction_total_all = float(instructions.sum())
        cumulative = np.cumsum(instructions)
        starts = np.r_[0.0, cumulative[:-1]] / instruction_total_all
        ends = cumulative / instruction_total_all
        counters = batch["__counter"].to_numpy(dtype=float)
        instruction_running = batch["__instruction_running"].to_numpy(dtype=float)
        counter_running = batch["__counter_running"].to_numpy(dtype=float)
        reliable = (
            np.isfinite(counters)
            & (counters >= 0)
            & np.isfinite(counter_running)
            & (counter_running >= pmu_min_running)
            & (instruction_running >= pmu_min_running)
        )
        reliable_positions = np.flatnonzero(reliable)
        if len(reliable_positions) == 0:
            skipped.append(f"batch {batch_idx}: no intervals pass PMU threshold")
            continue
        instruction_mass = float(instructions[reliable].sum() / instruction_total_all)
        if instruction_mass < min_instruction_mass:
            skipped.append(
                f"batch {batch_idx}: reliable instruction mass {instruction_mass:.3f}"
            )
            continue
        counter_total = float(counters[reliable].sum())
        instruction_total = float(instructions[reliable].sum())
        if counter_total <= 0 or instruction_total <= 0:
            skipped.append(f"batch {batch_idx}: nonpositive reliable total")
            continue
        raw_weights = np.minimum(
            instruction_running[reliable_positions], counter_running[reliable_positions]
        ) / 100.0
        if len(batch) > 1:
            edge = (reliable_positions == 0) | (reliable_positions == len(batch) - 1)
            raw_weights[edge] *= edge_weight
        raw_weights = np.maximum(raw_weights, np.finfo(float).eps)
        # Equal total influence per batch prevents a slower batch with more
        # emitted intervals from becoming an independent larger replicate.
        normalized_weights = raw_weights / raw_weights.sum()
        for local_index, position in enumerate(reliable_positions):
            observation_rows.append({
                "batch_idx": batch_idx,
                "a": float(starts[position]),
                "b": float(ends[position]),
                "width": float(ends[position] - starts[position]),
                "counter_fraction": float(counters[position] / counter_total),
                "weight": float(normalized_weights[local_index]),
                "pmu_running_fraction": float(
                    min(instruction_running[position], counter_running[position]) / 100.0
                ),
                "edge_interval": bool(position == 0 or position == len(batch) - 1),
            })
        total_rows.append({
            "batch_idx": batch_idx,
            "counter_total": counter_total,
            "instruction_total": instruction_total,
            "counter_per_instruction": counter_total / instruction_total,
            "reliable_intervals": int(len(reliable_positions)),
            "available_intervals": int(len(batch)),
            "reliable_instruction_mass": instruction_mass,
            "mean_running_percentage": float(
                np.minimum(
                    instruction_running[reliable_positions],
                    counter_running[reliable_positions],
                ).mean()
            ),
        })
    return pd.DataFrame(observation_rows), pd.DataFrame(total_rows), skipped


def huber(values: np.ndarray, delta: float) -> np.ndarray:
    absolute = np.abs(values)
    return np.where(absolute <= delta, 0.5 * values * values, delta * (absolute - 0.5 * delta))


def fit_unit_shape(
    observations: pd.DataFrame,
    *,
    n_bins: int,
    smoothness: float,
    huber_delta: float,
    maxiter: int,
) -> ShapeFit:
    w, _ = build_overlap_matrix(
        observations["a"].to_numpy(dtype=float),
        observations["b"].to_numpy(dtype=float),
        n_bins,
    )
    values = observations["counter_fraction"].to_numpy(dtype=float)
    weights = observations["weight"].to_numpy(dtype=float)
    d2 = second_difference(n_bins)
    weighted_w = w * np.sqrt(weights[:, None])
    singular = np.linalg.svd(weighted_w, compute_uv=False)
    tolerance = singular[0] * max(weighted_w.shape) * np.finfo(float).eps if len(singular) else 0
    rank = int(np.count_nonzero(singular > tolerance))
    condition = float(singular[0] / singular[-1]) if len(singular) and singular[-1] > tolerance else math.inf

    def objective(shape: np.ndarray) -> float:
        residual = w @ shape - values
        loss = float(np.sum(weights * huber(residual, huber_delta)))
        if len(d2) and smoothness > 0:
            loss += float(smoothness * np.sum((d2 @ shape) ** 2))
        return loss

    result = minimize(
        objective,
        np.ones(n_bins, dtype=float),
        method="SLSQP",
        bounds=[(0.0, None)] * n_bins,
        constraints={"type": "eq", "fun": lambda shape: float(shape.mean() - 1.0)},
        options={"maxiter": maxiter, "ftol": 1e-11},
    )
    shape = np.asarray(result.x, dtype=float)
    residual = values - w @ shape
    return ShapeFit(
        values=shape,
        success=bool(result.success and np.all(np.isfinite(shape))),
        message=str(result.message),
        objective=float(result.fun),
        iterations=int(getattr(result, "nit", 0)),
        rank=rank,
        condition_number=condition,
        residual_rmse=float(np.sqrt(np.average(residual * residual, weights=weights))),
    )


def paired_shape_test(
    differences: np.ndarray,
    *,
    permutations: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    trials, bins = differences.shape
    mean = differences.mean(axis=0)
    sd = differences.std(axis=0, ddof=1)
    se = sd / math.sqrt(trials)
    safe_se = np.maximum(se, np.finfo(float).eps)
    observed_global = float(np.mean(mean * mean))
    signs = rng.choice((-1.0, 1.0), size=(permutations, trials))
    null_means = signs @ differences / trials
    null_global = np.mean(null_means * null_means, axis=1)
    global_p = (1 + np.count_nonzero(null_global >= observed_global - 1e-15)) / (
        permutations + 1
    )
    null_t = np.abs(null_means / safe_se[None, :])
    observed_t = np.abs(mean / safe_se)
    maximum_t = null_t.max(axis=1)
    pointwise_fwer = (
        1 + np.count_nonzero(maximum_t[:, None] >= observed_t[None, :] - 1e-15, axis=0)
    ) / (permutations + 1)
    critical = float(np.quantile(maximum_t, 0.95))
    return {
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci_low": mean - critical * se,
        "ci_high": mean + critical * se,
        "global_statistic": observed_global,
        "global_p": float(global_p),
        "pointwise_fwer_p": pointwise_fwer,
        "simultaneous_critical_t": critical,
    }


def paired_scalar_pvalue(
    clean: np.ndarray,
    target: np.ndarray,
    *,
    permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if np.all(clean > 0) and np.all(target > 0):
        differences = np.log(target / clean)
        change = float(100.0 * np.expm1(differences.mean()))
    else:
        differences = target - clean
        change = float(100.0 * (target.mean() / clean.mean() - 1.0)) if clean.mean() else math.nan
    observed = abs(float(differences.mean()))
    signs = rng.choice((-1.0, 1.0), size=(permutations, len(differences)))
    null = np.abs(signs @ differences / len(differences))
    p_value = (1 + np.count_nonzero(null >= observed - 1e-15)) / (permutations + 1)
    return change, float(p_value)


def benjamini_hochberg(values: pd.Series) -> np.ndarray:
    selected = values.to_numpy(dtype=float)
    order = np.argsort(selected)
    ranked = selected[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "_-" else "_" for character in value)


def step_xy(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, 1.0, len(values) + 1)
    return np.repeat(edges, 2)[1:-1], np.repeat(values, 2)


def save_figure(figure: plt.Figure, prefix: Path, formats: list[str], dpi: int) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        kwargs = {"dpi": dpi} if extension == "png" else {}
        figure.savefig(prefix.with_suffix(f".{extension}"), bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(figure)


def plot_shape_epochs(
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    *,
    model: str,
    counter: str,
    epochs: list[int],
    target_label: str,
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> None:
    columns = min(5, len(epochs))
    rows = math.ceil(len(epochs) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(3.8 * columns, 3.0 * rows), squeeze=False)
    for axis, epoch in zip(axes.flat, epochs):
        selected = summary[
            summary["model"].eq(model)
            & summary["counter"].eq(counter)
            & summary["epoch"].eq(epoch)
        ]
        for role, color, label in (
            ("clean", COLORS["clean"], "clean"),
            ("target", COLORS["target"], target_label),
        ):
            curve = selected[selected["role"].eq(role)].sort_values("bin_index")
            if curve.empty:
                continue
            mean = curve["shape_mean"].to_numpy(dtype=float)
            sd = curve["shape_std"].to_numpy(dtype=float)
            x, y = step_xy(mean)
            _, low = step_xy(np.maximum(mean - sd, 0.0))
            _, high = step_xy(mean + sd)
            axis.fill_between(x, low, high, color=color, alpha=0.13)
            axis.plot(x, y, color=color, linewidth=1.6, label=label)
        test = tests[
            tests["model"].eq(model)
            & tests["counter"].eq(counter)
            & tests["epoch"].eq(epoch)
        ]
        suffix = f"; q={test['shape_fdr_q_within_metric'].iloc[0]:.3g}" if not test.empty else ""
        axis.set_title(f"epoch {epoch}{suffix}", fontsize=9)
        axis.set_xlim(0, 1)
        axis.set_ylim(bottom=0)
        axis.grid(True, color="#d9dde1", linewidth=0.5)
    for axis in axes.flat[len(epochs):]:
        axis.set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=8)
    figure.supxlabel("Relative retired-instruction progress")
    figure.supylabel("Unit-integral counter shape density")
    figure.suptitle(
        f"CIFAR-10 | {model.upper()} | forward | {counter}\n"
        "Bands are trial SD; each trial shape integrates to one",
        fontsize=11,
    )
    figure.tight_layout(rect=(0.02, 0.02, 1.0, 0.92))
    save_figure(figure, output_dir / f"{safe_name(counter)}_shape_epochs", formats, dpi)


def plot_difference_epochs(
    points: pd.DataFrame,
    tests: pd.DataFrame,
    *,
    model: str,
    counter: str,
    epochs: list[int],
    target_label: str,
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> None:
    columns = min(5, len(epochs))
    rows = math.ceil(len(epochs) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(3.8 * columns, 3.0 * rows), squeeze=False)
    for axis, epoch in zip(axes.flat, epochs):
        selected = points[
            points["model"].eq(model)
            & points["counter"].eq(counter)
            & points["epoch"].eq(epoch)
        ].sort_values("bin_index")
        if not selected.empty:
            mean = selected["difference_mean"].to_numpy(dtype=float)
            low = selected["simultaneous_ci_low"].to_numpy(dtype=float)
            high = selected["simultaneous_ci_high"].to_numpy(dtype=float)
            x, y = step_xy(mean)
            _, low_step = step_xy(low)
            _, high_step = step_xy(high)
            axis.fill_between(x, low_step, high_step, color=COLORS["difference"], alpha=0.18)
            axis.plot(x, y, color=COLORS["difference"], linewidth=1.7)
            significant = selected["pointwise_fwer_p"].le(0.05).to_numpy()
            centers = selected["progress_center"].to_numpy(dtype=float)
            if significant.any():
                axis.scatter(centers[significant], mean[significant], color="#111111", s=15, zorder=3)
        test = tests[
            tests["model"].eq(model)
            & tests["counter"].eq(counter)
            & tests["epoch"].eq(epoch)
        ]
        suffix = f"; q={test['shape_fdr_q_within_metric'].iloc[0]:.3g}" if not test.empty else ""
        axis.set_title(f"epoch {epoch}{suffix}", fontsize=9)
        axis.axhline(0.0, color="#56595c", linewidth=0.8)
        axis.set_xlim(0, 1)
        axis.grid(True, color="#d9dde1", linewidth=0.5)
    for axis in axes.flat[len(epochs):]:
        axis.set_visible(False)
    figure.supxlabel("Relative retired-instruction progress")
    figure.supylabel(f"{target_label} - clean shape density")
    figure.suptitle(
        f"CIFAR-10 | {model.upper()} | forward | {counter}\n"
        "Bands are trial-level simultaneous 95% sign-flip intervals",
        fontsize=11,
    )
    figure.tight_layout(rect=(0.02, 0.02, 1.0, 0.92))
    save_figure(figure, output_dir / f"{safe_name(counter)}_shape_difference_epochs", formats, dpi)


def plot_total_efficiency(
    totals: pd.DataFrame,
    tests: pd.DataFrame,
    *,
    model: str,
    counter: str,
    target_label: str,
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> None:
    selected = totals[totals["model"].eq(model) & totals["counter"].eq(counter)]
    selected_tests = tests[tests["model"].eq(model) & tests["counter"].eq(counter)]
    if selected.empty:
        return
    figure, axis = plt.subplots(figsize=(8.8, 4.8))
    for role, color, label in (
        ("clean", COLORS["clean"], "clean"),
        ("target", COLORS["target"], target_label),
    ):
        curve = selected[selected["role"].eq(role)].sort_values("epoch")
        axis.fill_between(
            curve["epoch"], curve["efficiency_mean"] - curve["efficiency_std"],
            curve["efficiency_mean"] + curve["efficiency_std"], color=color, alpha=0.13,
        )
        axis.plot(curve["epoch"], curve["efficiency_mean"], color=color, marker="o", label=label)
    significant_epochs = selected_tests.loc[
        selected_tests["total_fdr_q_within_metric"].le(0.05), "epoch"
    ].astype(int).tolist()
    axis.set_title(
        f"CIFAR-10 | {model.upper()} | forward | {counter} total efficiency\n"
        f"significant epochs after per-metric FDR: {significant_epochs or 'none'}"
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Total counter / total instructions")
    axis.grid(True, color="#d9dde1", linewidth=0.55)
    axis.legend(frameon=False)
    figure.tight_layout()
    save_figure(figure, output_dir / f"{safe_name(counter)}_total_efficiency", formats, dpi)


def main() -> int:
    args = parse_args()
    input_dir = resolve_input_dir(args.input_dir)
    output_dir = args.output_dir.resolve()
    marker = output_dir / "shape_global_tests.csv"
    if marker.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists below {output_dir}; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(input_dir)
    indexed = index_runs(runs)
    trial_shape_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    skipped: list[str] = []
    requested_groups: list[tuple[str, str, str, str, list[str]]] = []
    for dataset in args.datasets:
        for model in args.models:
            clean_condition = condition_name(dataset, model, "clean")
            target_condition = condition_name(dataset, model, args.target)
            clean_trials = {trial for condition, trial in indexed if condition == clean_condition}
            target_trials = {trial for condition, trial in indexed if condition == target_condition}
            trials = sorted(clean_trials & target_trials)
            if not trials:
                warnings.warn(f"Skipping {dataset}/{model}: no paired {clean_condition}/{target_condition} runs")
                continue
            requested_groups.append((dataset, model, clean_condition, target_condition, trials))

    total_fits = sum(len(trials) * len(args.epochs) * len(args.counters) for *_, trials in requested_groups)
    fit_index = 0
    for dataset, model, clean_condition, target_condition, trials in requested_groups:
        for trial_id in trials:
            try:
                clean_run = load_run(indexed[(clean_condition, trial_id)], args.phase_label_lag)
                target_run = load_run(indexed[(target_condition, trial_id)], args.phase_label_lag)
            except ValueError as exc:
                skipped.append(f"{dataset}/{model}/{trial_id}: {exc}")
                continue
            for epoch in args.epochs:
                for counter in args.counters:
                    fit_index += 1
                    if fit_index == 1 or fit_index % 100 == 0 or fit_index == total_fits:
                        print(
                            f"[paired shape {fit_index}/{total_fits}] "
                            f"{dataset}/{model}/{trial_id}/epoch{epoch}/{counter}",
                            flush=True,
                        )
                    clean_obs, clean_totals, clean_skips = batch_observations(
                        clean_run, counter=counter, epoch=epoch, phase=args.phase,
                        pmu_min_running=args.pmu_min_running,
                        min_instruction_mass=args.min_instruction_mass,
                        edge_weight=args.edge_weight,
                    )
                    target_obs, target_totals, target_skips = batch_observations(
                        target_run, counter=counter, epoch=epoch, phase=args.phase,
                        pmu_min_running=args.pmu_min_running,
                        min_instruction_mass=args.min_instruction_mass,
                        edge_weight=args.edge_weight,
                    )
                    matched = sorted(
                        set(clean_totals.get("batch_idx", pd.Series(dtype=int)).astype(int))
                        & set(target_totals.get("batch_idx", pd.Series(dtype=int)).astype(int))
                    )
                    if len(matched) < args.min_matched_batches:
                        skipped.append(
                            f"{dataset}/{model}/{trial_id}/epoch{epoch}/{counter}: "
                            f"{len(matched)} matched reliable batches"
                        )
                        continue
                    clean_obs = clean_obs[clean_obs["batch_idx"].isin(matched)].copy()
                    target_obs = target_obs[target_obs["batch_idx"].isin(matched)].copy()
                    clean_totals = clean_totals[clean_totals["batch_idx"].isin(matched)].copy()
                    target_totals = target_totals[target_totals["batch_idx"].isin(matched)].copy()
                    clean_fit = fit_unit_shape(
                        clean_obs, n_bins=args.n_bins, smoothness=args.shape_smoothness,
                        huber_delta=args.huber_delta, maxiter=args.optimizer_maxiter,
                    )
                    target_fit = fit_unit_shape(
                        target_obs, n_bins=args.n_bins, smoothness=args.shape_smoothness,
                        huber_delta=args.huber_delta, maxiter=args.optimizer_maxiter,
                    )
                    if not clean_fit.success or not target_fit.success:
                        skipped.append(
                            f"{dataset}/{model}/{trial_id}/epoch{epoch}/{counter}: optimizer failure"
                        )
                        continue
                    clean_efficiency = float(
                        clean_totals["counter_total"].sum() / clean_totals["instruction_total"].sum()
                    )
                    target_efficiency = float(
                        target_totals["counter_total"].sum() / target_totals["instruction_total"].sum()
                    )
                    edges = np.linspace(0.0, 1.0, args.n_bins + 1)
                    for role, fit, efficiency in (
                        ("clean", clean_fit, clean_efficiency),
                        ("target", target_fit, target_efficiency),
                    ):
                        for bin_index, shape in enumerate(fit.values):
                            trial_shape_rows.append({
                                "dataset": dataset,
                                "model": model,
                                "target_role": args.target,
                                "phase": args.phase,
                                "trial_id": trial_id,
                                "epoch": epoch,
                                "counter": counter,
                                "role": role,
                                "bin_index": bin_index,
                                "progress_start": edges[bin_index],
                                "progress_end": edges[bin_index + 1],
                                "progress_center": (edges[bin_index] + edges[bin_index + 1]) / 2,
                                "shape_density": float(shape),
                                "total_efficiency": efficiency,
                                "reconstructed_counter_per_instruction": float(efficiency * shape),
                                "matched_batches": len(matched),
                            })
                    diagnostic_rows.append({
                        "dataset": dataset,
                        "model": model,
                        "target_role": args.target,
                        "phase": args.phase,
                        "trial_id": trial_id,
                        "epoch": epoch,
                        "counter": counter,
                        "matched_batches": len(matched),
                        "clean_observations": len(clean_obs),
                        "target_observations": len(target_obs),
                        "clean_rank": clean_fit.rank,
                        "target_rank": target_fit.rank,
                        "clean_condition_number": clean_fit.condition_number,
                        "target_condition_number": target_fit.condition_number,
                        "clean_residual_rmse": clean_fit.residual_rmse,
                        "target_residual_rmse": target_fit.residual_rmse,
                        "clean_objective": clean_fit.objective,
                        "target_objective": target_fit.objective,
                        "clean_total_efficiency": clean_efficiency,
                        "target_total_efficiency": target_efficiency,
                        "clean_dropped_batch_reasons": json.dumps(clean_skips),
                        "target_dropped_batch_reasons": json.dumps(target_skips),
                    })

    trial_shapes = pd.DataFrame(trial_shape_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    if trial_shapes.empty:
        raise RuntimeError("No paired unit-integral shape profiles were fitted")

    shape_summary_rows: list[dict[str, object]] = []
    pointwise_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    total_summary_rows: list[dict[str, object]] = []
    grouped = trial_shapes.groupby(["dataset", "model", "target_role", "phase", "counter", "epoch"])
    test_number = 0
    for key, group in grouped:
        dataset, model, target_role, phase, counter, epoch = key
        clean = group[group["role"].eq("clean")].pivot(
            index="trial_id", columns="bin_index", values="shape_density"
        )
        target = group[group["role"].eq("target")].pivot(
            index="trial_id", columns="bin_index", values="shape_density"
        )
        trials = clean.index.intersection(target.index)
        if len(trials) < args.min_paired_trials:
            skipped.append(f"{dataset}/{model}/epoch{epoch}/{counter}: {len(trials)} paired fits")
            continue
        clean_array = clean.loc[trials].sort_index(axis=1).to_numpy(dtype=float)
        target_array = target.loc[trials].sort_index(axis=1).to_numpy(dtype=float)
        differences = target_array - clean_array
        rng = np.random.default_rng(args.random_seed + 1009 * test_number)
        test_number += 1
        result = paired_shape_test(differences, permutations=args.permutations, rng=rng)
        clean_efficiency = group[group["role"].eq("clean")].drop_duplicates("trial_id").set_index("trial_id")["total_efficiency"].reindex(trials).to_numpy(dtype=float)
        target_efficiency = group[group["role"].eq("target")].drop_duplicates("trial_id").set_index("trial_id")["total_efficiency"].reindex(trials).to_numpy(dtype=float)
        total_change, total_p = paired_scalar_pvalue(
            clean_efficiency, target_efficiency, permutations=args.permutations, rng=rng
        )
        test_rows.append({
            "dataset": dataset,
            "model": model,
            "target_role": target_role,
            "phase": phase,
            "counter": counter,
            "epoch": int(epoch),
            "paired_trials": len(trials),
            "shape_global_statistic": result["global_statistic"],
            "shape_global_permutation_p": result["global_p"],
            "total_efficiency_clean_mean": float(clean_efficiency.mean()),
            "total_efficiency_clean_std": float(clean_efficiency.std(ddof=1)),
            "total_efficiency_target_mean": float(target_efficiency.mean()),
            "total_efficiency_target_std": float(target_efficiency.std(ddof=1)),
            "total_efficiency_paired_change_pct": total_change,
            "total_efficiency_permutation_p": total_p,
        })
        for role, values in (("clean", clean_array), ("target", target_array)):
            for bin_index in range(args.n_bins):
                shape_summary_rows.append({
                    "dataset": dataset,
                    "model": model,
                    "target_role": target_role,
                    "phase": phase,
                    "counter": counter,
                    "epoch": int(epoch),
                    "role": role,
                    "bin_index": bin_index,
                    "progress_start": bin_index / args.n_bins,
                    "progress_end": (bin_index + 1) / args.n_bins,
                    "progress_center": (bin_index + 0.5) / args.n_bins,
                    "shape_mean": float(values[:, bin_index].mean()),
                    "shape_std": float(values[:, bin_index].std(ddof=1)),
                    "paired_trials": len(trials),
                })
        for bin_index in range(args.n_bins):
            pointwise_rows.append({
                "dataset": dataset,
                "model": model,
                "target_role": target_role,
                "phase": phase,
                "counter": counter,
                "epoch": int(epoch),
                "bin_index": bin_index,
                "progress_start": bin_index / args.n_bins,
                "progress_end": (bin_index + 1) / args.n_bins,
                "progress_center": (bin_index + 0.5) / args.n_bins,
                "difference_mean": float(result["mean"][bin_index]),
                "difference_std": float(result["sd"][bin_index]),
                "difference_se": float(result["se"][bin_index]),
                "simultaneous_ci_low": float(result["ci_low"][bin_index]),
                "simultaneous_ci_high": float(result["ci_high"][bin_index]),
                "pointwise_fwer_p": float(result["pointwise_fwer_p"][bin_index]),
                "paired_trials": len(trials),
            })
        for role, values in (("clean", clean_efficiency), ("target", target_efficiency)):
            total_summary_rows.append({
                "dataset": dataset,
                "model": model,
                "target_role": target_role,
                "phase": phase,
                "counter": counter,
                "epoch": int(epoch),
                "role": role,
                "efficiency_mean": float(values.mean()),
                "efficiency_std": float(values.std(ddof=1)),
                "paired_trials": len(trials),
            })

    tests = pd.DataFrame(test_rows)
    if tests.empty:
        raise RuntimeError("No metric/epoch had enough paired trial profiles")
    adjusted = []
    for _, group in tests.groupby(["dataset", "model", "counter"], sort=False):
        group = group.copy()
        group["shape_fdr_q_within_metric"] = benjamini_hochberg(
            group["shape_global_permutation_p"]
        )
        group["total_fdr_q_within_metric"] = benjamini_hochberg(
            group["total_efficiency_permutation_p"]
        )
        group["shape_significant"] = group["shape_fdr_q_within_metric"].le(args.alpha)
        group["total_significant"] = group["total_fdr_q_within_metric"].le(args.alpha)
        adjusted.append(group)
    tests = pd.concat(adjusted, ignore_index=True)
    shape_summary = pd.DataFrame(shape_summary_rows)
    pointwise = pd.DataFrame(pointwise_rows)
    total_summary = pd.DataFrame(total_summary_rows)

    trial_shapes.to_csv(output_dir / "trial_shape_profiles.csv", index=False)
    diagnostics.to_csv(output_dir / "fit_diagnostics.csv", index=False)
    shape_summary.to_csv(output_dir / "epoch_shape_summary.csv", index=False)
    pointwise.to_csv(output_dir / "shape_pointwise_tests.csv", index=False)
    tests.to_csv(output_dir / "shape_global_tests.csv", index=False)
    total_summary.to_csv(output_dir / "total_efficiency_summary.csv", index=False)

    figures = 0
    if not args.no_plots:
        for dataset, model, _, _, _ in requested_groups:
            destination = output_dir / dataset / model / args.target
            for counter in args.counters:
                selected_tests = tests[
                    tests["dataset"].eq(dataset)
                    & tests["model"].eq(model)
                    & tests["counter"].eq(counter)
                ]
                if selected_tests.empty:
                    continue
                plot_epochs = sorted(selected_tests["epoch"].astype(int).unique())
                plot_shape_epochs(
                    shape_summary, tests, model=model, counter=counter, epochs=plot_epochs,
                    target_label=ROLE_LABELS[args.target], output_dir=destination,
                    formats=list(args.formats), dpi=args.dpi,
                )
                plot_difference_epochs(
                    pointwise, tests, model=model, counter=counter, epochs=plot_epochs,
                    target_label=ROLE_LABELS[args.target], output_dir=destination,
                    formats=list(args.formats), dpi=args.dpi,
                )
                plot_total_efficiency(
                    total_summary, tests, model=model, counter=counter,
                    target_label=ROLE_LABELS[args.target], output_dir=destination,
                    formats=list(args.formats), dpi=args.dpi,
                )
                figures += 3

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "datasets": list(args.datasets),
        "models": list(args.models),
        "target": args.target,
        "phase": args.phase,
        "phase_label_lag_rows": args.phase_label_lag,
        "pmu_min_running_percentage": args.pmu_min_running,
        "minimum_reliable_instruction_mass": args.min_instruction_mass,
        "edge_interval_weight": args.edge_weight,
        "n_bins": args.n_bins,
        "shape_definition": "interval counter / reliable batch counter total; unit integral",
        "scale_definition": "reliable batch counter total / matching instruction total",
        "batch_weighting": "equal total weight per batch; interval weights use PMU running percentage and edge reliability",
        "statistical_unit": "paired training trial",
        "multiple_testing": "BH independently across requested epochs for each dataset/model/counter",
        "completed_tests": len(tests),
        "significant_shape_tests": int(tests["shape_significant"].sum()),
        "significant_total_tests": int(tests["total_significant"].sum()),
        "figures": figures,
        "skipped_count": len(skipped),
        "skipped": skipped,
        "sample_pairing_assumption": (
            "Matching trial_id, epoch, and batch_idx are assumed to preserve sample order; "
            "the logs do not contain sample IDs."
        ),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"Wrote {len(tests)} tests and {figures} figures to {output_dir}; "
        f"shape significant={summary['significant_shape_tests']}, "
        f"total significant={summary['significant_total_tests']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
