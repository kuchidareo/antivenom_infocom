#!/usr/bin/env python3
"""Core utilities for distributional execution-profile estimation."""

from __future__ import annotations

import csv
import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize, lsq_linear


STRUCTURAL_ALIASES = {
    "epoch": ("epoch", "local_epoch"),
    "batch_idx": ("batch_idx", "batch", "batch_id"),
    "phase": ("phase", "training_phase"),
    "timestamp": ("timestamp_unix", "perf_elapsed_sec", "timestamp"),
    "status": ("perf_status", "status"),
    "metric_event": ("metric_event", "event"),
    "num_examples": ("num_examples", "examples", "batch_size_actual"),
}

# Alias resolution is intentionally centralized. The first matching name wins.
COUNTER_ALIASES = {
    "instructions": ("perf_instructions", "instructions", "instructions:u"),
    "cycles": ("perf_cycles", "cycles", "cycles:u"),
    "task_clock": ("perf_task_clock", "task_clock", "task-clock"),
    "branches": ("perf_br_retired", "perf_branches", "branches", "br_retired"),
    "branch_misses": (
        "perf_br_mis_pred_retired",
        "perf_branch_misses",
        "branch-misses",
        "br_mis_pred_retired",
    ),
    "l1d_access": ("perf_l1d_cache", "perf_l1d_cache_rd", "l1d_cache", "l1d_cache_rd"),
    "l1d_refill": (
        "perf_l1d_cache_refill",
        "perf_l1d_cache_refill_rd",
        "l1d_cache_refill",
    ),
    "l1d_writeback": ("perf_l1d_cache_wb", "l1d_cache_wb", "l1d_writeback"),
    "l2d_access": ("perf_l2d_cache", "perf_l2d_cache_rd", "l2d_cache", "l2d_cache_rd"),
    "l2d_refill": (
        "perf_l2d_cache_refill",
        "perf_l2d_cache_refill_rd",
        "l2d_cache_refill",
    ),
    "l2d_writeback": ("perf_l2d_cache_wb", "l2d_cache_wb", "l2d_writeback"),
    "bus_access": ("perf_bus_access", "perf_bus_access_rd", "bus_access"),
    "memory_access": ("perf_mem_access", "perf_memory_access", "mem_access"),
    "speculative_instructions": ("perf_inst_spec", "inst_spec"),
    "context_switches": ("perf_context_switches", "context-switches", "context_switches"),
    "cpu_migrations": ("perf_cpu_migrations", "cpu-migrations", "cpu_migrations"),
    "page_faults": ("perf_page_faults", "page-faults", "page_faults"),
}

DEFAULT_COUNTERS = tuple(name for name in COUNTER_ALIASES if name != "instructions")
DEFAULT_BIN_CANDIDATES = (4, 8, 12, 16, 24, 32)
INVALID_TEXT = {"", "nan", "none", "<not counted>", "<not supported>"}


@dataclass(frozen=True)
class RunFiles:
    condition: str
    run_id: str
    perf_path: Path
    metrics_path: Path | None
    hardware_path: Path | None
    metadata: dict[str, object]


@dataclass(frozen=True)
class BinDiagnostics:
    k: int
    rank: int
    condition_number: float
    singular_values: np.ndarray
    coverage: np.ndarray
    accepted: bool
    reason: str


@dataclass(frozen=True)
class ProfileFit:
    mu: np.ndarray
    q: np.ndarray
    tau2: float
    predicted: np.ndarray
    residuals: np.ndarray
    success: bool
    message: str
    objective: float
    iterations: int
    y_scale: float


def normalized_name(value: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", str(value).lower())


def resolve_alias(columns: Iterable[str], aliases: Sequence[str]) -> str | None:
    columns = list(columns)
    for alias in aliases:
        if alias in columns:
            return alias
    normalized = {normalized_name(column): column for column in columns}
    for alias in aliases:
        match = normalized.get(normalized_name(alias))
        if match is not None:
            return match
    return None


def resolve_structural_columns(columns: Iterable[str]) -> dict[str, str | None]:
    return {name: resolve_alias(columns, aliases) for name, aliases in STRUCTURAL_ALIASES.items()}


def resolve_counter_columns(columns: Iterable[str]) -> dict[str, str | None]:
    return {name: resolve_alias(columns, aliases) for name, aliases in COUNTER_ALIASES.items()}


def read_first_row(path: Path) -> dict[str, object]:
    with path.open(newline="") as handle:
        return next(csv.DictReader(handle), {})


def discover_runs(input_dir: Path) -> list[RunFiles]:
    runs: list[RunFiles] = []
    for perf_path in sorted(input_dir.rglob("*_perf.csv")):
        prefix = perf_path.name.removesuffix("_perf.csv")
        metrics_path = perf_path.with_name(f"{prefix}_metrics.csv")
        hardware_path = perf_path.with_name(f"{prefix}.csv")
        metadata = read_first_row(perf_path)
        runs.append(
            RunFiles(
                condition=perf_path.parent.name,
                run_id=str(metadata.get("experiment_id") or prefix),
                perf_path=perf_path,
                metrics_path=metrics_path if metrics_path.is_file() else None,
                hardware_path=hardware_path if hardware_path.is_file() else None,
                metadata=metadata,
            )
        )
    if not runs:
        raise FileNotFoundError(f"No *_perf.csv files found below {input_dir}")
    return runs


def parse_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series.dtype):
        return pd.to_numeric(series, errors="coerce")
    cleaned = series.astype("string").str.strip().str.lower()
    cleaned = cleaned.mask(cleaned.isin(INVALID_TEXT)).str.replace(",", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def associated_pmu_columns(counter_column: str, columns: Iterable[str]) -> tuple[str | None, str | None]:
    columns = set(columns)
    enabled = f"{counter_column}_enabled_pct"
    runtime = f"{counter_column}_runtime_pct"
    return (enabled if enabled in columns else None, runtime if runtime in columns else None)


def detect_scaling_layout(frame: pd.DataFrame, counter_column: str) -> dict[str, object]:
    enabled_col, runtime_col = associated_pmu_columns(counter_column, frame.columns)
    result: dict[str, object] = {
        "counter_column": counter_column,
        "enabled_column": enabled_col,
        "runtime_column": runtime_col,
        "layout": "none",
        "count_already_scaled": False,
    }
    if enabled_col is None or runtime_col is None:
        return result
    enabled = parse_numeric(frame[enabled_col]).dropna()
    runtime = parse_numeric(frame[runtime_col]).dropna()
    if enabled.empty or runtime.empty:
        return result
    enabled_is_pct = bool(enabled.between(0, 100).mean() > 0.95)
    runtime_is_pct = bool(runtime.between(0, 100).mean() > 0.95)
    perf_stat_source = "perf_elapsed_sec" in frame.columns and "perf_events" in frame.columns
    if enabled_is_pct and not runtime_is_pct:
        # perf_logger.py stores perf field 6 (% running) under enabled_pct and
        # field 5 (counter runtime in ns) under runtime_pct.
        result.update(
            layout="legacy_perf_logger_running_pct_and_runtime_ns",
            running_percentage_column=enabled_col,
            runtime_ns_column=runtime_col,
            count_already_scaled=perf_stat_source,
        )
    elif enabled_is_pct and runtime_is_pct:
        result.update(layout="enabled_and_running_percentages", count_already_scaled=perf_stat_source)
    else:
        result.update(layout="unrecognized", count_already_scaled=perf_stat_source)
    return result


def scaled_counter_values(
    frame: pd.DataFrame,
    counter_column: str,
    mode: str,
) -> tuple[pd.Series, dict[str, object]]:
    values = parse_numeric(frame[counter_column]).astype(float)
    info = detect_scaling_layout(frame, counter_column)
    info["requested_mode"] = mode
    info["applied"] = False
    info["factor_description"] = "none"
    if mode == "off":
        return values, info
    if bool(info["count_already_scaled"]):
        info["factor_description"] = "perf stat printed count is already multiplex-scaled"
        if mode == "on":
            warnings.warn(
                f"{counter_column}: --pmu-scaling on requested, but the source is perf stat "
                "and the count is already scaled; refusing a second correction"
            )
        return values, info
    enabled_col = info.get("enabled_column")
    runtime_col = info.get("runtime_column")
    factor: pd.Series | None = None
    if info["layout"] == "enabled_and_running_percentages" and enabled_col and runtime_col:
        enabled = parse_numeric(frame[str(enabled_col)]).astype(float)
        running = parse_numeric(frame[str(runtime_col)]).astype(float)
        factor = enabled / running.where(running > 0)
        info["factor_description"] = "enabled_percentage / running_percentage"
    elif info["layout"] == "legacy_perf_logger_running_pct_and_runtime_ns" and enabled_col:
        running_pct = parse_numeric(frame[str(enabled_col)]).astype(float)
        factor = 100.0 / running_pct.where(running_pct > 0)
        info["factor_description"] = "100 / running_percentage"
    if factor is None:
        if mode == "on":
            warnings.warn(f"{counter_column}: PMU scaling requested but usable timing fields are unavailable")
        return values, info
    valid = factor.replace([np.inf, -np.inf], np.nan).between(1.0, 100.0)
    values = values.where(~valid, values * factor)
    info["applied"] = True
    info["corrected_rows"] = int(valid.sum())
    return values, info


def partial_batches(metrics_path: Path | None) -> tuple[set[tuple[int, int]], dict[tuple[int, int], int]]:
    if metrics_path is None:
        return set(), {}
    frame = pd.read_csv(metrics_path, low_memory=False)
    schema = resolve_structural_columns(frame.columns)
    required = (schema["epoch"], schema["batch_idx"], schema["num_examples"])
    if any(column is None for column in required):
        return set(), {}
    selected = frame.copy()
    event_col = schema["metric_event"]
    if event_col is not None:
        selected = selected[selected[event_col].astype(str).eq("train_batch")]
    epoch_col, batch_col, examples_col = required
    selected[epoch_col] = parse_numeric(selected[epoch_col])
    selected[batch_col] = parse_numeric(selected[batch_col])
    selected[examples_col] = parse_numeric(selected[examples_col])
    selected = selected.dropna(subset=[epoch_col, batch_col, examples_col])
    sizes = {
        (int(row[epoch_col]), int(row[batch_col])): int(row[examples_col])
        for _, row in selected.iterrows()
    }
    partial: set[tuple[int, int]] = set()
    for epoch, group in selected.groupby(epoch_col):
        full_size = int(group[examples_col].max())
        for _, row in group.iterrows():
            if int(row[examples_col]) < full_size:
                partial.add((int(epoch), int(row[batch_col])))
    return partial, sizes


def build_forward_observations(
    frame: pd.DataFrame,
    *,
    epoch: int,
    phase: str,
    instruction_column: str,
    counter_column: str,
    partial: set[tuple[int, int]],
    include_partial: bool,
    pmu_scaling: str,
    nonnegative: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    schema = resolve_structural_columns(frame.columns)
    required = (schema["epoch"], schema["batch_idx"], schema["phase"], schema["timestamp"])
    if any(column is None for column in required):
        raise ValueError(f"Missing required perf columns: {schema}")
    epoch_col, batch_col, phase_col, timestamp_col = required
    work = frame.copy()
    work[epoch_col] = parse_numeric(work[epoch_col])
    work[batch_col] = parse_numeric(work[batch_col])
    work[timestamp_col] = parse_numeric(work[timestamp_col])
    work["__instructions"] = parse_numeric(work[instruction_column])
    work["__counter"], scaling = scaled_counter_values(work, counter_column, pmu_scaling)
    mask = work[epoch_col].eq(epoch) & work[phase_col].astype(str).str.lower().eq(phase.lower())
    status_col = schema["status"]
    if status_col is not None:
        mask &= work[status_col].astype(str).str.lower().eq("ok")
    work = work.loc[mask].copy()
    total_phase_rows = len(work)
    invalid_instruction_rows = int((~np.isfinite(work["__instructions"]) | (work["__instructions"] <= 0)).sum())
    records: list[dict[str, object]] = []
    excluded_partial_rows = 0
    invalid_counter_rows = 0
    for batch_value, batch in work.groupby(batch_col, sort=True):
        batch_idx = int(batch_value)
        if not include_partial and (epoch, batch_idx) in partial:
            excluded_partial_rows += len(batch)
            continue
        batch = batch.sort_values(timestamp_col)
        valid_instructions = np.isfinite(batch["__instructions"]) & batch["__instructions"].gt(0)
        batch = batch.loc[valid_instructions].copy()
        if batch.empty:
            continue
        instructions = batch["__instructions"].to_numpy(dtype=float)
        total = float(instructions.sum())
        if not np.isfinite(total) or total <= 0:
            continue
        cumulative = np.cumsum(instructions)
        starts = np.r_[0.0, cumulative[:-1]] / total
        ends = cumulative / total
        counters = batch["__counter"].to_numpy(dtype=float)
        valid_counter = np.isfinite(counters)
        if nonnegative:
            valid_counter &= counters >= 0
        invalid_counter_rows += int((~valid_counter).sum())
        for position in np.flatnonzero(valid_counter):
            records.append(
                {
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "timestamp": float(batch.iloc[position][timestamp_col]),
                    "a": float(starts[position]),
                    "b": float(ends[position]),
                    "width": float(ends[position] - starts[position]),
                    "instructions": float(instructions[position]),
                    "observed_increment": float(counters[position]),
                }
            )
    observations = pd.DataFrame.from_records(records)
    diagnostics = {
        "phase_rows": total_phase_rows,
        "invalid_instruction_rows": invalid_instruction_rows,
        "invalid_counter_rows": invalid_counter_rows,
        "excluded_partial_rows": excluded_partial_rows,
        "used_interval_observations": len(observations),
        "used_batches": int(observations["batch_idx"].nunique()) if not observations.empty else 0,
        "intervals_per_batch": (
            observations.groupby("batch_idx").size().astype(int).to_dict() if not observations.empty else {}
        ),
        "scaling": scaling,
    }
    return observations, diagnostics


def build_overlap_matrix(starts: np.ndarray, ends: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float)
    if starts.shape != ends.shape or starts.ndim != 1:
        raise ValueError("starts and ends must be one-dimensional arrays with equal shape")
    if not np.all(np.isfinite(starts)) or not np.all(np.isfinite(ends)):
        raise ValueError("interval boundaries must be finite")
    if np.any(starts < -1e-12) or np.any(ends > 1 + 1e-12) or np.any(ends <= starts):
        raise ValueError("intervals must satisfy 0 <= start < end <= 1")
    edges = np.linspace(0.0, 1.0, k + 1)
    left = np.maximum(starts[:, None], edges[:-1][None, :])
    right = np.minimum(ends[:, None], edges[1:][None, :])
    return np.maximum(0.0, right - left), edges


def diagnose_bins(w: np.ndarray, k: int, *, min_coverage: int, max_condition: float) -> BinDiagnostics:
    singular = np.linalg.svd(w, compute_uv=False)
    tolerance = max(w.shape) * np.finfo(float).eps * (singular[0] if singular.size else 0.0)
    rank = int((singular > tolerance).sum())
    condition = float(singular[0] / singular[-1]) if singular.size and singular[-1] > tolerance else math.inf
    coverage = (w > 1e-12).sum(axis=0).astype(int)
    reasons = []
    if len(w) < 4 * k:
        reasons.append(f"N={len(w)} < 4K={4 * k}")
    if coverage.size == 0 or int(coverage.min()) < min_coverage:
        reasons.append(f"minimum bin coverage {int(coverage.min()) if coverage.size else 0} < {min_coverage}")
    if rank < k:
        reasons.append(f"rank {rank} < K={k}")
    if not np.isfinite(condition) or condition > max_condition:
        reasons.append(f"condition number {condition:.3g} exceeds {max_condition:.3g}")
    return BinDiagnostics(k, rank, condition, singular, coverage, not reasons, "; ".join(reasons))


def choose_bin_count(
    starts: np.ndarray,
    ends: np.ndarray,
    *,
    requested: int | None,
    candidates: Sequence[int] = DEFAULT_BIN_CANDIDATES,
    min_coverage: int = 4,
    max_condition: float = 1e8,
) -> tuple[np.ndarray | None, np.ndarray | None, BinDiagnostics, list[BinDiagnostics]]:
    choices = [requested] if requested is not None else sorted(set(candidates), reverse=True)
    attempted: list[BinDiagnostics] = []
    for k in choices:
        if k is None or k < 2:
            raise ValueError("K must be at least 2")
        w, edges = build_overlap_matrix(starts, ends, int(k))
        diagnostic = diagnose_bins(w, int(k), min_coverage=min_coverage, max_condition=max_condition)
        attempted.append(diagnostic)
        if diagnostic.accepted:
            return w, edges, diagnostic, attempted
    final = attempted[-1]
    return None, None, final, attempted


def softplus(values: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(values))) + np.maximum(values, 0)


def sigmoid(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def inverse_softplus(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=float), 1e-12)
    return np.where(values > 20.0, values, np.log(np.expm1(values)))


def second_difference(k: int) -> np.ndarray:
    if k < 3:
        return np.zeros((0, k), dtype=float)
    matrix = np.zeros((k - 2, k), dtype=float)
    for row in range(k - 2):
        matrix[row, row : row + 3] = (1.0, -2.0, 1.0)
    return matrix


def fit_profile(
    w: np.ndarray,
    y: np.ndarray,
    *,
    tau2: float | None,
    mean_smoothness: float,
    variance_smoothness: float,
    nonnegative_mean: bool = True,
    maxiter: int = 1500,
) -> ProfileFit:
    w = np.asarray(w, dtype=float)
    y = np.asarray(y, dtype=float)
    if w.ndim != 2 or y.shape != (len(w),) or len(y) == 0:
        raise ValueError("W and y have incompatible or empty shapes")
    if not np.all(np.isfinite(w)) or not np.all(np.isfinite(y)):
        raise ValueError("W and y must be finite")
    rate = y / np.maximum(w.sum(axis=1), 1e-12)
    positive_scale = np.median(np.abs(y[y != 0])) if np.any(y != 0) else 0.0
    y_scale = float(max(positive_scale, np.std(y), np.percentile(np.abs(y), 75) * 0.1, 1.0))
    yn = y / y_scale
    k = w.shape[1]
    d2 = second_difference(k)
    try:
        if nonnegative_mean:
            initial_mu = lsq_linear(
                w, yn, bounds=(0.0, np.inf), lsmr_tol="auto"
            ).x
        else:
            initial_mu = np.linalg.lstsq(w, yn, rcond=None)[0]
    except Exception:
        initial_mu = np.full(k, float(np.mean(rate) / y_scale))
    if nonnegative_mean:
        initial_mu = np.maximum(initial_mu, 0.0)
    initial_residual = yn - w @ initial_mu
    median_width = max(float(np.median(w.sum(axis=1))), 1e-6)
    initial_q = np.full(k, max(float(np.var(initial_residual) / median_width), 1e-5))
    if tau2 is None:
        tau2n = max(float(np.var(initial_residual) * 0.02), 1e-8)
    else:
        if tau2 < 0:
            raise ValueError("tau2 must be non-negative")
        tau2n = float(tau2 / (y_scale * y_scale))
    q_floor = 1e-10
    # Optimize the mean directly. A softplus mean can become trapped near zero:
    # its sigmoid derivative vanishes after L-BFGS drives a bin negative, even
    # when observations at that progress position require a positive rate.
    theta0 = np.r_[initial_mu, inverse_softplus(initial_q)]

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        mu = theta[:k]
        raw_q = theta[k:]
        q = softplus(raw_q) + q_floor
        prediction = w @ mu
        variance = np.maximum(w @ q + tau2n, 1e-12)
        residual = yn - prediction
        value = float(np.sum(residual * residual / variance + np.log(variance)))
        d_prediction = -2.0 * residual / variance
        d_variance = 1.0 / variance - residual * residual / (variance * variance)
        grad_mu = w.T @ d_prediction
        grad_q = w.T @ d_variance
        if len(d2):
            mean_curvature = d2 @ mu
            value += mean_smoothness * float(mean_curvature @ mean_curvature)
            grad_mu += 2.0 * mean_smoothness * (d2.T @ mean_curvature)
            log_q = np.log(q)
            variance_curvature = d2 @ log_q
            value += variance_smoothness * float(variance_curvature @ variance_curvature)
            grad_q += (
                2.0 * variance_smoothness * (d2.T @ variance_curvature) / q
            )
        grad_q *= sigmoid(raw_q)
        return value, np.r_[grad_mu, grad_q]

    mean_bounds = [(0.0, None)] * k if nonnegative_mean else [(None, None)] * k
    result = minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        jac=True,
        bounds=[*mean_bounds, *[(-30.0, 30.0)] * k],
        options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-7, "maxls": 40},
    )
    mu = result.x[:k]
    q = softplus(result.x[k:]) + q_floor
    mu *= y_scale
    q *= y_scale * y_scale
    tau2_raw = tau2n * y_scale * y_scale
    predicted = w @ mu
    residuals = y - predicted
    return ProfileFit(
        mu=mu,
        q=q,
        tau2=tau2_raw,
        predicted=predicted,
        residuals=residuals,
        success=bool(result.success and np.all(np.isfinite(mu)) and np.all(q > 0)),
        message=str(result.message),
        objective=float(result.fun),
        iterations=int(result.nit),
        y_scale=y_scale,
    )


def cross_validate_batches(
    observations: pd.DataFrame,
    *,
    k: int,
    folds: int,
    tau2: float | None,
    mean_smoothness: float,
    variance_smoothness: float,
    random_seed: int,
) -> dict[str, float | int]:
    batches = np.array(sorted(observations["batch_idx"].unique()), dtype=int)
    folds = min(folds, len(batches))
    if folds < 2:
        return {"cv_folds": 0, "cv_observations": 0, "cv_rmse": math.nan, "cv_mae": math.nan, "cv_nll": math.nan}
    rng = np.random.default_rng(random_seed)
    rng.shuffle(batches)
    predictions: list[np.ndarray] = []
    actuals: list[np.ndarray] = []
    variances: list[np.ndarray] = []
    for held_out in np.array_split(batches, folds):
        test_mask = observations["batch_idx"].isin(held_out).to_numpy()
        train = observations.loc[~test_mask]
        test = observations.loc[test_mask]
        if train.empty or test.empty:
            continue
        w_train, _ = build_overlap_matrix(train["a"].to_numpy(), train["b"].to_numpy(), k)
        w_test, _ = build_overlap_matrix(test["a"].to_numpy(), test["b"].to_numpy(), k)
        try:
            fitted = fit_profile(
                w_train,
                train["observed_increment"].to_numpy(),
                tau2=tau2,
                mean_smoothness=mean_smoothness,
                variance_smoothness=variance_smoothness,
            )
        except (ValueError, FloatingPointError):
            continue
        predictions.append(w_test @ fitted.mu)
        actuals.append(test["observed_increment"].to_numpy(dtype=float))
        variances.append(np.maximum(w_test @ fitted.q + fitted.tau2, 1e-12))
    if not predictions:
        return {"cv_folds": 0, "cv_observations": 0, "cv_rmse": math.nan, "cv_mae": math.nan, "cv_nll": math.nan}
    predicted = np.concatenate(predictions)
    actual = np.concatenate(actuals)
    variance = np.concatenate(variances)
    residual = actual - predicted
    return {
        "cv_folds": folds,
        "cv_observations": len(actual),
        "cv_rmse": float(np.sqrt(np.mean(residual * residual))),
        "cv_mae": float(np.mean(np.abs(residual))),
        "cv_nll": float(np.mean(residual * residual / variance + np.log(variance))),
    }


def residual_statistics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    residual = observed - predicted
    denominator = float(np.sum((observed - observed.mean()) ** 2))
    return {
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std(ddof=1)) if len(residual) > 1 else 0.0,
        "residual_rmse": float(np.sqrt(np.mean(residual * residual))),
        "residual_mae": float(np.mean(np.abs(residual))),
        "reconstruction_r2": float(1.0 - np.sum(residual * residual) / denominator) if denominator > 0 else math.nan,
    }


def inspect_runs(runs: Sequence[RunFiles], phase: str) -> dict[str, object]:
    report: dict[str, object] = {"phase": phase, "runs": [], "interpretation": {}}
    all_columns: list[str] = []
    for run in runs:
        frame = pd.read_csv(run.perf_path, low_memory=False)
        columns = list(frame.columns)
        if not all_columns:
            all_columns = columns
        schema = resolve_structural_columns(columns)
        counters = resolve_counter_columns(columns)
        epoch_col, batch_col, phase_col = schema["epoch"], schema["batch_idx"], schema["phase"]
        forward_counts: dict[str, int] = {}
        interval_summary: dict[str, float | int] = {}
        phase_group_counts = pd.Series(dtype=int)
        epochs: list[int] = []
        batches: list[int] = []
        if epoch_col and batch_col and phase_col:
            selected = frame[frame[phase_col].astype(str).str.lower().eq(phase.lower())].copy()
            selected[epoch_col] = parse_numeric(selected[epoch_col])
            selected[batch_col] = parse_numeric(selected[batch_col])
            epochs = sorted(selected[epoch_col].dropna().astype(int).unique().tolist())
            batches = sorted(selected[batch_col].dropna().astype(int).unique().tolist())
            forward_counts = {str(int(key)): int(value) for key, value in selected.groupby(epoch_col).size().items()}
            phase_group_counts = selected.groupby([epoch_col, batch_col]).size()
            if len(phase_group_counts):
                interval_summary = {
                    "observed_batch_groups": int(len(phase_group_counts)),
                    "minimum_among_observed": int(phase_group_counts.min()),
                    "median_among_observed": float(phase_group_counts.median()),
                    "maximum_among_observed": int(phase_group_counts.max()),
                }
        partial, sizes = partial_batches(run.metrics_path)
        if sizes:
            expected_index = pd.MultiIndex.from_tuples(sorted(sizes), names=[epoch_col, batch_col])
            complete_counts = phase_group_counts.reindex(expected_index, fill_value=0)
            interval_summary.update(
                expected_batch_groups=int(len(complete_counts)),
                zero_interval_batches=int((complete_counts == 0).sum()),
                minimum_including_zero=int(complete_counts.min()),
                median_including_zero=float(complete_counts.median()),
                maximum_including_zero=int(complete_counts.max()),
            )
        missing = {}
        nonnumeric = {}
        for canonical, column in counters.items():
            if column is None:
                missing[canonical] = len(frame)
                continue
            parsed = parse_numeric(frame[column])
            missing[canonical] = int(parsed.isna().sum())
            nonnumeric[canonical] = int((frame[column].notna() & parsed.isna()).sum())
        instruction_col = counters["instructions"]
        interval_evidence = {}
        if instruction_col:
            values = parse_numeric(frame[instruction_col]).dropna().to_numpy(dtype=float)
            interval_evidence = {
                "nondecreasing_difference_fraction": float(np.mean(np.diff(values) >= 0)) if len(values) > 1 else math.nan,
                "negative_values": int((values < 0).sum()),
            }
        scaling = detect_scaling_layout(frame, instruction_col) if instruction_col else {}
        report["runs"].append(
            {
                "condition": run.condition,
                "run_id": run.run_id,
                "files": {
                    "perf": str(run.perf_path),
                    "metrics": str(run.metrics_path) if run.metrics_path else None,
                    "hardware": str(run.hardware_path) if run.hardware_path else None,
                },
                "rows": len(frame),
                "resolved_structural_columns": schema,
                "resolved_counter_columns": counters,
                "epochs_in_phase": epochs,
                "batches_in_phase": batches,
                "epochs_in_metrics": sorted({key[0] for key in sizes}),
                "batches_in_metrics": sorted({key[1] for key in sizes}),
                "phase_rows_per_epoch": forward_counts,
                "intervals_per_observed_batch": interval_summary,
                "partial_batches": [list(key) for key in sorted(partial)],
                "batch_sizes": {f"{epoch}:{batch}": size for (epoch, batch), size in sorted(sizes.items())},
                "missing_values_by_counter": missing,
                "unparsed_nonempty_values_by_counter": nonnumeric,
                "interval_increment_evidence": interval_evidence,
                "pmu_scaling_layout": scaling,
            }
        )
    report["perf_columns"] = all_columns
    report["interpretation"] = {
        "counter_semantics": "perf stat -I interval increments; values are not cumulative",
        "pmu_scaling": (
            "perf stat printed counts are already multiplex-scaled. In these files *_enabled_pct "
            "contains percent-running and *_runtime_pct contains runtime nanoseconds; auto mode "
            "therefore applies no additional factor."
        ),
        "phase_annotation": (
            "phase/epoch/batch are TrainingState snapshots when each interval row is emitted; "
            "a boundary interval can straddle adjacent phases"
        ),
    }
    return report


def write_investigation(report: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data_investigation.json").write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    lines = [
        "# Data Investigation",
        "",
        f"Analyzed phase: `{report['phase']}`",
        "",
        "## Interpretation",
        "",
    ]
    for key, value in dict(report["interpretation"]).items():
        lines.append(f"- **{key}**: {value}")
    lines.extend(["", "## Runs", ""])
    for row in report["runs"]:
        lines.extend(
            [
                f"### {row['condition']}",
                "",
                f"- Perf file: `{row['files']['perf']}`",
                f"- Rows: {row['rows']}",
                f"- Epochs: {row['epochs_in_phase']}",
                f"- Batches: {row['batches_in_phase']}",
                f"- Phase rows per epoch: `{row['phase_rows_per_epoch']}`",
                f"- Intervals per observed batch: `{row['intervals_per_observed_batch']}`",
                f"- Partial batches from metrics: `{row['partial_batches']}`",
                "",
            ]
        )
    lines.extend(["## Perf Columns", "", "```text", *report["perf_columns"], "```", ""])
    (output_dir / "data_investigation.md").write_text("\n".join(lines))


__all__ = [
    "COUNTER_ALIASES",
    "DEFAULT_BIN_CANDIDATES",
    "DEFAULT_COUNTERS",
    "BinDiagnostics",
    "ProfileFit",
    "RunFiles",
    "build_forward_observations",
    "build_overlap_matrix",
    "choose_bin_count",
    "cross_validate_batches",
    "discover_runs",
    "fit_profile",
    "inspect_runs",
    "partial_batches",
    "residual_statistics",
    "resolve_counter_columns",
    "resolve_structural_columns",
    "write_investigation",
]
