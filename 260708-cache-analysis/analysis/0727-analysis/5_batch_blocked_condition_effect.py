#!/usr/bin/env python3
"""Run the 4_ epoch-wise distribution analysis after blocking batch identity.

Counter values remain raw interval increments. Retired instructions are used
only to construct relative forward progress in [0, 1]. Clean and target runs
are paired by trial, epoch, and batch index. A shared additive batch-rate
nuisance term is estimated jointly and removed before the same inverse profile,
Gaussian aggregation, Wasserstein permutation tests, and 2x5 epoch figures used
by 4_epochwise_distribution_comparison.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-batch-blocked-raw")
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from execution_profiles import (
    DEFAULT_COUNTERS,
    RunFiles,
    build_forward_observations,
    build_overlap_matrix,
    discover_runs,
    fit_profile,
    partial_batches,
    resolve_counter_columns,
    resolve_structural_columns,
    second_difference,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_COLLECTION = SCRIPT_DIR / "cache_0727_jetson_cpu_20_trials"
DEFAULT_INPUT = DEFAULT_COLLECTION / "192.168.0.141"
DEFAULT_OUTPUT = DEFAULT_COLLECTION / "batch_blocked_raw_epochwise_comparison"

FOUR_PATH = SCRIPT_DIR / "4_epochwise_distribution_comparison.py"
FOUR_SPEC = importlib.util.spec_from_file_location("epochwise_distribution", FOUR_PATH)
assert FOUR_SPEC is not None and FOUR_SPEC.loader is not None
FOUR = importlib.util.module_from_spec(FOUR_SPEC)
sys.modules.setdefault(FOUR_SPEC.name, FOUR)
FOUR_SPEC.loader.exec_module(FOUR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--datasets", nargs="+", choices=("cifar10", "trashnet"),
        default=("cifar10", "trashnet"),
    )
    parser.add_argument(
        "--models", nargs="+", choices=("cnn", "vit"), default=("cnn", "vit")
    )
    parser.add_argument(
        "--targets", nargs="+", choices=tuple(FOUR.TARGET_LABELS),
        default=tuple(FOUR.TARGET_LABELS),
    )
    parser.add_argument("--epochs", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--counters", nargs="+", default=None)
    parser.add_argument("--n-bins", type=int, default=4)
    parser.add_argument("--progress-points", type=int, default=101)
    parser.add_argument("--expected-trials", type=int, default=20)
    parser.add_argument("--min-paired-trials", type=int, default=8)
    parser.add_argument("--min-matched-batches", type=int, default=6)
    parser.add_argument("--mean-smoothness", type=float, default=2.0)
    parser.add_argument("--variance-smoothness", type=float, default=2.0)
    parser.add_argument("--batch-ridge", type=float, default=1e-4)
    parser.add_argument("--tau2", type=float)
    parser.add_argument("--optimizer-maxiter", type=int, default=1500)
    parser.add_argument("--permutations", type=int, default=4999)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--pmu-scaling", choices=("auto", "on", "off"), default="auto")
    partial = parser.add_mutually_exclusive_group()
    partial.add_argument("--include-partial-batch", action="store_true")
    partial.add_argument("--exclude-partial-batch", action="store_true")
    parser.add_argument("--random-seed", type=int, default=260728)
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf"), default=("pdf",))
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.n_bins < 2 or args.progress_points < 11:
        parser.error("--n-bins must be >= 2 and --progress-points must be >= 11")
    if args.min_paired_trials < 2 or args.min_matched_batches < 2:
        parser.error("minimum trial and batch counts must be at least 2")
    if args.permutations < 99:
        parser.error("--permutations must be at least 99")
    if args.tau2 is not None and args.tau2 < 0:
        parser.error("--tau2 must be non-negative")
    if any(value < 0 for value in (
        args.mean_smoothness, args.variance_smoothness, args.batch_ridge
    )):
        parser.error("smoothness and ridge parameters must be non-negative")
    return args


def run_index(runs: list[RunFiles]) -> dict[tuple[str, str], RunFiles]:
    output: dict[tuple[str, str], RunFiles] = {}
    for run in runs:
        key = (run.condition, run.trial_id)
        if key in output:
            warnings.warn(f"Duplicate run for {key}; using {run.perf_path.name}")
        output[key] = run
    return output


class RawObservationLoader:
    """Load raw counter increments while retaining instruction progress."""

    def __init__(self, include_partial: bool, pmu_scaling: str):
        self.include_partial = include_partial
        self.pmu_scaling = pmu_scaling
        self.frames: dict[Path, pd.DataFrame] = {}
        self.partial: dict[Path, set[tuple[int, int]]] = {}
        self.observations: dict[tuple[Path, str, int], pd.DataFrame] = {}

    def frame(self, run: RunFiles) -> pd.DataFrame:
        if run.perf_path not in self.frames:
            self.frames[run.perf_path] = pd.read_csv(run.perf_path, low_memory=False)
            self.partial[run.perf_path] = partial_batches(run.metrics_path)[0]
        return self.frames[run.perf_path]

    def get(self, run: RunFiles, counter: str, epoch: int) -> pd.DataFrame:
        key = (run.perf_path, counter, epoch)
        if key in self.observations:
            return self.observations[key]
        frame = self.frame(run)
        counters = resolve_counter_columns(frame.columns)
        structural = resolve_structural_columns(frame.columns)
        instruction_column = counters.get("instructions")
        counter_column = counters.get(counter)
        if instruction_column is None or counter_column is None:
            result = pd.DataFrame()
        else:
            columns = {column for column in structural.values() if column is not None}
            columns.update((instruction_column, counter_column))
            for suffix in ("_enabled_pct", "_runtime_pct"):
                candidate = f"{counter_column}{suffix}"
                if candidate in frame.columns:
                    columns.add(candidate)
            for candidate in ("perf_elapsed_sec", "perf_events"):
                if candidate in frame.columns:
                    columns.add(candidate)
            compact = frame.loc[:, sorted(columns)]
            epoch_column = structural.get("epoch")
            if epoch_column is not None:
                compact = compact.loc[
                    pd.to_numeric(compact[epoch_column], errors="coerce").eq(epoch)
                ]
            result, _ = build_forward_observations(
                compact,
                epoch=epoch,
                phase="forward",
                instruction_column=instruction_column,
                counter_column=counter_column,
                partial=self.partial[run.perf_path],
                include_partial=self.include_partial,
                pmu_scaling=self.pmu_scaling,
                counter_normalization="raw",
                nonnegative=True,
            )
        self.observations[key] = result
        return result


def batch_contrasts(observations: pd.DataFrame, batches: list[int]) -> np.ndarray:
    """Return constant-rate batch effects constrained to sum to zero."""
    if len(batches) <= 1:
        return np.empty((len(observations), 0))
    positions = {batch: index for index, batch in enumerate(batches)}
    output = np.zeros((len(observations), len(batches) - 1))
    for row, item in enumerate(observations.itertuples()):
        position = positions[int(item.batch_idx)]
        if position < len(batches) - 1:
            output[row, position] = float(item.width)
        else:
            output[row, :] = -float(item.width)
    return output


def shared_batch_adjustment(
    clean: pd.DataFrame,
    target: pd.DataFrame,
    *,
    k: int,
    mean_smoothness: float,
    batch_ridge: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Remove a nuisance rate shared by the matching clean/target batch."""
    batches = sorted(
        set(clean["batch_idx"].astype(int)) & set(target["batch_idx"].astype(int))
    )
    if len(batches) < 2:
        raise ValueError("fewer than two matched batches")
    clean = clean[clean["batch_idx"].isin(batches)].copy()
    target = target[target["batch_idx"].isin(batches)].copy()
    wc, edges = build_overlap_matrix(clean["a"].to_numpy(), clean["b"].to_numpy(), k)
    wt, _ = build_overlap_matrix(target["a"].to_numpy(), target["b"].to_numpy(), k)
    bc = batch_contrasts(clean, batches)
    bt = batch_contrasts(target, batches)
    nuisance_count = len(batches) - 1
    design = np.vstack((
        np.hstack((wc, np.zeros_like(wc), bc)),
        np.hstack((np.zeros_like(wt), wt, bt)),
    ))
    observed = np.r_[
        clean["observed_increment"].to_numpy(dtype=float),
        target["observed_increment"].to_numpy(dtype=float),
    ]
    widths = np.r_[clean["width"].to_numpy(), target["width"].to_numpy()]
    rates = observed / np.maximum(widths, 1e-12)
    finite_nonzero = np.abs(rates[np.isfinite(rates) & (rates != 0)])
    scale = float(max(
        np.median(finite_nonzero) if len(finite_nonzero) else 0.0,
        np.std(rates),
        1.0,
    ))
    weights = 1.0 / np.sqrt(np.maximum(widths, np.median(widths) * 0.25))
    weighted_design = design * weights[:, None]
    weighted_observed = observed / scale * weights
    penalties: list[np.ndarray] = []
    d2 = second_difference(k)
    if len(d2) and mean_smoothness > 0:
        for offset in (0, k):
            penalty = np.zeros((len(d2), 2 * k + nuisance_count))
            penalty[:, offset : offset + k] = math.sqrt(mean_smoothness) * d2
            penalties.append(penalty)
    if nuisance_count and batch_ridge > 0:
        penalty = np.zeros((nuisance_count, 2 * k + nuisance_count))
        penalty[:, 2 * k :] = math.sqrt(batch_ridge) * np.eye(nuisance_count)
        penalties.append(penalty)
    augmented_design = np.vstack((weighted_design, *penalties)) if penalties else weighted_design
    augmented_observed = np.r_[
        weighted_observed,
        np.zeros(sum(len(penalty) for penalty in penalties)),
    ]
    bounds_low = np.r_[np.zeros(2 * k), np.full(nuisance_count, -np.inf)]
    fitted = lsq_linear(
        augmented_design,
        augmented_observed,
        bounds=(bounds_low, np.full(2 * k + nuisance_count, np.inf)),
        lsmr_tol="auto",
        max_iter=1000,
    )
    rank = int(np.linalg.matrix_rank(weighted_design))
    identifiable = rank == design.shape[1]
    if not fitted.success or not identifiable:
        raise ValueError(
            f"batch-adjustment design is not identifiable (rank={rank}, parameters={design.shape[1]})"
        )
    nuisance_parameters = fitted.x[2 * k :] * scale
    batch_rates = {
        batch: float(nuisance_parameters[index])
        for index, batch in enumerate(batches[:-1])
    }
    batch_rates[batches[-1]] = float(-nuisance_parameters.sum())
    for frame in (clean, target):
        nuisance = frame["batch_idx"].map(batch_rates).to_numpy(dtype=float)
        frame["observed_increment"] = (
            frame["observed_increment"].to_numpy(dtype=float)
            - frame["width"].to_numpy(dtype=float) * nuisance
        )
    predicted = design @ fitted.x * scale
    residual = observed - predicted
    diagnostics = {
        "matched_batches": batches,
        "batch_rates": batch_rates,
        "observations": len(observed),
        "parameters": design.shape[1],
        "rank": rank,
        "residual_rmse": float(np.sqrt(np.mean(residual * residual))),
        "joint_cost": float(fitted.cost * scale * scale),
        "edges": edges,
    }
    return clean, target, diagnostics


def resample_piecewise(
    values: np.ndarray, edges: np.ndarray, progress: np.ndarray
) -> np.ndarray:
    indexes = np.minimum(np.searchsorted(edges[1:], progress, side="right"), len(values) - 1)
    return np.asarray(values, dtype=float)[indexes]


def fit_adjusted_profiles(
    clean: pd.DataFrame,
    target: pd.DataFrame,
    *,
    progress: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, object]:
    clean, target, adjustment = shared_batch_adjustment(
        clean,
        target,
        k=args.n_bins,
        mean_smoothness=args.mean_smoothness,
        batch_ridge=args.batch_ridge,
    )
    profiles = []
    fits = []
    for observations in (clean, target):
        w, edges = build_overlap_matrix(
            observations["a"].to_numpy(), observations["b"].to_numpy(), args.n_bins
        )
        fitted = fit_profile(
            w,
            observations["observed_increment"].to_numpy(dtype=float),
            tau2=args.tau2,
            mean_smoothness=args.mean_smoothness,
            variance_smoothness=args.variance_smoothness,
            nonnegative_mean=True,
            maxiter=args.optimizer_maxiter,
        )
        if not np.all(np.isfinite(fitted.mu)) or not np.all(fitted.q > 0):
            raise ValueError("inverse profile contains invalid values")
        width = max(float(observations["width"].median()), 1e-12)
        mean = resample_piecewise(fitted.mu, edges, progress)
        q = resample_piecewise(fitted.q, edges, progress)
        rate_variance = np.maximum(q / width + fitted.tau2 / (width * width), 0.0)
        profiles.append((mean, rate_variance))
        fits.append(fitted)
    return {
        "clean_mean": profiles[0][0],
        "clean_variance": profiles[0][1],
        "target_mean": profiles[1][0],
        "target_variance": profiles[1][1],
        "adjustment": adjustment,
        "clean_fit": fits[0],
        "target_fit": fits[1],
    }


def main() -> int:
    args = parse_args()
    input_dir = FOUR.resolve_input_dir(args.input_dir)
    output_dir = args.output_dir.resolve()
    marker = output_dir / "epoch_global_tests.csv"
    if marker.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists below {output_dir}; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    include_partial = bool(args.include_partial_batch and not args.exclude_partial_batch)

    runs = discover_runs(input_dir)
    indexed = run_index(runs)
    available = {run.condition for run in runs}
    maps = FOUR.condition_maps(
        list(args.datasets), list(args.models), list(args.targets), available
    )
    counters = list(args.counters or DEFAULT_COUNTERS)
    progress = np.linspace(0.0, 1.0, args.progress_points)
    loader = RawObservationLoader(include_partial, args.pmu_scaling)
    profile_store: dict[
        tuple[str, str, str, str, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    diagnostic_rows: list[dict[str, object]] = []
    nuisance_rows: list[dict[str, object]] = []
    skipped: list[str] = []
    total = sum(
        (len(conditions) - 1) * len(counters) * len(args.epochs)
        for conditions in maps.values()
    )
    position = 0
    for (dataset, model), conditions in maps.items():
        clean_condition = conditions["clean"]
        for target_role, target_condition in conditions.items():
            if target_role == "clean":
                continue
            clean_trials = {trial for condition, trial in indexed if condition == clean_condition}
            target_trials = {trial for condition, trial in indexed if condition == target_condition}
            trials = sorted(clean_trials & target_trials)
            for counter in counters:
                for epoch in args.epochs:
                    position += 1
                    print(
                        f"[raw blocked fit {position}/{total}] "
                        f"{dataset}/{model}/{target_role}/{counter}/epoch {epoch}",
                        flush=True,
                    )
                    for trial in trials:
                        clean = loader.get(indexed[(clean_condition, trial)], counter, epoch)
                        target = loader.get(indexed[(target_condition, trial)], counter, epoch)
                        matched = set(clean.get("batch_idx", pd.Series(dtype=int)).astype(int)) & set(
                            target.get("batch_idx", pd.Series(dtype=int)).astype(int)
                        )
                        if len(matched) < args.min_matched_batches:
                            skipped.append(
                                f"{dataset}/{model}/{target_role}/{counter}/epoch{epoch}/{trial}: "
                                f"{len(matched)} matched batches"
                            )
                            continue
                        try:
                            fitted = fit_adjusted_profiles(
                                clean, target, progress=progress, args=args
                            )
                        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                            skipped.append(
                                f"{dataset}/{model}/{target_role}/{counter}/epoch{epoch}/{trial}: {exc}"
                            )
                            continue
                        key = (dataset, model, target_role, counter, epoch, trial)
                        profile_store[key] = (
                            fitted["clean_mean"], fitted["clean_variance"],
                            fitted["target_mean"], fitted["target_variance"],
                        )
                        adjustment = fitted["adjustment"]
                        diagnostic_rows.append({
                            "dataset": dataset,
                            "model": model,
                            "target_role": target_role,
                            "counter": counter,
                            "epoch": epoch,
                            "trial_id": trial,
                            "matched_batches": len(adjustment["matched_batches"]),
                            "observations": adjustment["observations"],
                            "parameters": adjustment["parameters"],
                            "rank": adjustment["rank"],
                            "joint_cost": adjustment["joint_cost"],
                            "joint_residual_rmse": adjustment["residual_rmse"],
                            "clean_optimizer_success": fitted["clean_fit"].success,
                            "target_optimizer_success": fitted["target_fit"].success,
                            "clean_objective": fitted["clean_fit"].objective,
                            "target_objective": fitted["target_fit"].objective,
                        })
                        for batch_idx, nuisance_rate in adjustment["batch_rates"].items():
                            nuisance_rows.append({
                                "dataset": dataset,
                                "model": model,
                                "target_role": target_role,
                                "counter": counter,
                                "epoch": epoch,
                                "trial_id": trial,
                                "batch_idx": batch_idx,
                                "estimated_shared_batch_rate": nuisance_rate,
                            })

    profile_rows: list[dict[str, object]] = []
    pointwise_rows: list[dict[str, object]] = []
    epoch_test_rows: list[dict[str, object]] = []
    distance_rows: list[dict[str, object]] = []
    test_index = 0
    for (dataset, model), conditions in maps.items():
        for target_role in args.targets:
            if target_role not in conditions:
                continue
            for counter in counters:
                for epoch in args.epochs:
                    trials = sorted(
                        key[5] for key in profile_store
                        if key[:5] == (dataset, model, target_role, counter, epoch)
                    )
                    if len(trials) < args.min_paired_trials:
                        skipped.append(
                            f"{dataset}/{model}/{target_role}/{counter}/epoch{epoch}: "
                            f"only {len(trials)} adjusted trial profiles"
                        )
                        continue
                    arrays = [
                        np.stack([profile_store[(dataset, model, target_role, counter, epoch, trial)][i]
                                  for trial in trials])
                        for i in range(4)
                    ]
                    rng = np.random.default_rng(args.random_seed + 1009 * test_index)
                    test_index += 1
                    result = FOUR.paired_distribution_permutation(
                        *arrays, progress, args.permutations, rng
                    )
                    scale = max(
                        float(np.median(np.abs(result["reference_mean"]))),
                        np.finfo(float).eps,
                    )
                    epoch_test_rows.append({
                        "dataset": dataset,
                        "model": model,
                        "counter": counter,
                        "target_role": target_role,
                        "epoch": epoch,
                        "paired_trials": len(trials),
                        "global_wasserstein_squared": result["global_wasserstein_squared"],
                        "global_wasserstein": math.sqrt(max(result["global_wasserstein_squared"], 0)),
                        "normalized_global_w2": (
                            math.sqrt(max(result["global_wasserstein_squared"], 0)) / scale
                        ),
                        "global_permutation_p": result["global_p"],
                    })
                    distance_rows.extend(FOUR.trial_epoch_rows(
                        dataset, model, counter, target_role, epoch, trials,
                        *arrays, progress,
                    ))
                    for role, mean, variance in (
                        ("normal_reference", result["reference_mean"], result["reference_variance"]),
                        ("target", result["target_mean"], result["target_variance"]),
                    ):
                        for index, point in enumerate(progress):
                            profile_rows.append({
                                "dataset": dataset,
                                "model": model,
                                "counter": counter,
                                "target_role": target_role,
                                "epoch": epoch,
                                "profile_role": role,
                                "progress": point,
                                "mean_rate": mean[index],
                                "rate_variance": variance[index],
                                "rate_sd": math.sqrt(max(variance[index], 0)),
                                "paired_trials": len(trials),
                            })
                    for index, point in enumerate(progress):
                        reference = float(result["reference_mean"][index])
                        target_value = float(result["target_mean"][index])
                        pointwise_rows.append({
                            "dataset": dataset,
                            "model": model,
                            "counter": counter,
                            "target_role": target_role,
                            "epoch": epoch,
                            "progress": point,
                            "reference_mean_rate": reference,
                            "target_mean_rate": target_value,
                            "mean_rate_difference": target_value - reference,
                            "wasserstein_squared": result["wasserstein_squared"][index],
                            "wasserstein_distance": math.sqrt(max(result["wasserstein_squared"][index], 0)),
                            "pointwise_permutation_p": result["pointwise_p"][index],
                            "pointwise_fwer_p": result["pointwise_fwer_p"][index],
                            "paired_trials": len(trials),
                        })

    epoch_tests = pd.DataFrame(epoch_test_rows)
    profiles = pd.DataFrame(profile_rows)
    pointwise = pd.DataFrame(pointwise_rows)
    distances = pd.DataFrame(distance_rows)
    if epoch_tests.empty:
        raise RuntimeError("No epoch had enough identifiable adjusted trial profiles")
    adjusted = []
    for _, group in epoch_tests.groupby(["dataset", "model"], sort=False):
        group = group.copy()
        group["global_fdr_q"] = FOUR.benjamini_hochberg(
            group["global_permutation_p"].to_numpy(dtype=float)
        )
        group["global_fdr_significant"] = group["global_fdr_q"].le(args.alpha)
        adjusted.append(group)
    epoch_tests = pd.concat(adjusted, ignore_index=True)
    trends = pd.DataFrame(FOUR.trend_test_rows(
        distances, list(args.epochs), args.permutations, args.random_seed + 700_001
    ))
    if not trends.empty:
        adjusted_trends = []
        for _, group in trends.groupby(["dataset", "model"], sort=False):
            group = group.copy()
            group["slope_fdr_q"] = FOUR.benjamini_hochberg(
                group["slope_sign_flip_p"].to_numpy(dtype=float)
            )
            group["omnibus_fdr_q"] = FOUR.benjamini_hochberg(
                group["epoch_omnibus_permutation_p"].to_numpy(dtype=float)
            )
            adjusted_trends.append(group)
        trends = pd.concat(adjusted_trends, ignore_index=True)
    distance_summary = FOUR.epoch_distance_summary(distances)

    tables = {
        "epoch_profiles.csv": profiles,
        "epoch_global_tests.csv": epoch_tests,
        "epoch_pointwise_tests.csv": pointwise,
        "trial_epoch_distances.csv": distances,
        "epoch_distance_summary.csv": distance_summary,
        "longitudinal_trend_tests.csv": trends,
        "batch_adjustment_diagnostics.csv": pd.DataFrame(diagnostic_rows),
        "batch_nuisance_rates.csv": pd.DataFrame(nuisance_rows),
    }
    for name, frame in tables.items():
        frame.to_csv(output_dir / name, index=False)

    figures = 0
    for (dataset, model), conditions in maps.items():
        for target_role in args.targets:
            if target_role not in conditions:
                continue
            target_dir = output_dir / dataset / model / target_role
            for counter in counters:
                selected = epoch_tests[
                    epoch_tests["dataset"].eq(dataset)
                    & epoch_tests["model"].eq(model)
                    & epoch_tests["target_role"].eq(target_role)
                    & epoch_tests["counter"].eq(counter)
                ]
                if selected.empty:
                    continue
                plot_epochs = [epoch for epoch in args.epochs if epoch in set(selected["epoch"])]
                FOUR.plot_epoch_profiles(
                    profiles, epoch_tests,
                    dataset=dataset, model=model, counter=counter,
                    target_role=target_role, epochs=plot_epochs,
                    output_dir=target_dir, formats=list(args.formats), dpi=args.dpi,
                    counter_normalization="raw",
                )
                FOUR.plot_difference_heatmap(
                    pointwise, epoch_tests,
                    dataset=dataset, model=model, counter=counter,
                    target_role=target_role, output_dir=target_dir,
                    formats=list(args.formats), dpi=args.dpi, alpha=args.alpha,
                    counter_normalization="raw",
                )
                figures += 2
                if not trends.empty:
                    FOUR.plot_distance_trend(
                        distances, distance_summary, trends,
                        dataset=dataset, model=model, counter=counter,
                        target_role=target_role, output_dir=target_dir,
                        formats=list(args.formats), dpi=args.dpi,
                    )
                    figures += 1

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "counter_normalization": "raw",
        "instruction_usage": "relative progress axis only",
        "n_bins": args.n_bins,
        "partial_batches_included": include_partial,
        "successful_trial_profiles": len(profile_store),
        "epoch_tests": len(epoch_tests),
        "figures": figures,
        "skipped_count": len(skipped),
        "skipped": skipped,
        "batch_model": (
            "A shared additive counter/progress rate is fitted for each matched "
            "trial/epoch/batch and removed from both conditions before separate "
            "raw inverse-profile fits. Batch effects sum to zero within a fit."
        ),
        "sample_identity_note": (
            "Logs do not contain sample IDs. Matching assumes equal trial, epoch, "
            "and batch index correspond to the deterministic batch order."
        ),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {len(tables)} tables and {figures} figures to {output_dir}")
    print(f"Skipped entries: {len(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
