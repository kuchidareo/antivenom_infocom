#!/usr/bin/env python3
"""Pool trial intervals, fit execution profiles, and compare conditions.

Every forward batch is normalized independently. Only after its instruction
progress and counter/instruction observations have been constructed are rows
from repeated trials pooled. This supports low-frequency logs where a single
trial does not contain enough forward intervals for an identifiable profile.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from execution_profiles import (
    COUNTER_ALIASES,
    DEFAULT_COUNTERS,
    build_overlap_matrix,
    build_forward_observations,
    choose_bin_count,
    cross_validate_batches,
    diagnose_bins,
    discover_runs,
    fit_profile,
    partial_batches,
    residual_statistics,
    resolve_counter_columns,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_COLLECTION = (
    REPOSITORY_ROOT
    / "M260727-motivational-study"
    / "analysis"
    / "motivational_0727_30_trials"
)
DEFAULT_INPUT = DEFAULT_COLLECTION / "192.168.0.112"
DEFAULT_OUTPUT = DEFAULT_COLLECTION / "pooled_distribution_analysis"
ROLES = (
    "clean",
    "moderate_augmentation",
    "strong_augmentation",
    "availability_shortcut",
    "non_iid",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--datasets", nargs="+", choices=("cifar10", "trashnet"), default=("cifar10",))
    parser.add_argument("--models", nargs="+", choices=("cnn", "vit"), default=("cnn", "vit"))
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--epochs", nargs="+", type=int, default=list(range(10)))
    parser.add_argument(
        "--counters",
        nargs="+",
        choices=tuple(COUNTER_ALIASES),
        default=list(DEFAULT_COUNTERS),
    )
    parser.add_argument("--phase", default="forward")
    parser.add_argument("--n-bins", type=int)
    parser.add_argument("--tau2", type=float)
    parser.add_argument("--mean-smoothness", type=float, default=1.0)
    parser.add_argument("--variance-smoothness", type=float, default=1.0)
    parser.add_argument("--min-bin-coverage", type=int, default=4)
    parser.add_argument("--max-condition-number", type=float, default=1e8)
    parser.add_argument("--optimizer-maxiter", type=int, default=1500)
    parser.add_argument("--cv-folds", type=int, default=0)
    parser.add_argument("--pmu-scaling", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--include-partial-batch", action="store_true")
    parser.add_argument("--expected-trials", type=int, default=30)
    parser.add_argument("--min-paired-epochs", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=260727)
    parser.add_argument(
        "--no-scalar-fallback",
        action="store_true",
        help="Skip rank-1 epochs instead of estimating one whole-forward counter/instruction value.",
    )
    parser.add_argument("--skip-comparison", action="store_true")
    parser.add_argument("--save-observations", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.epochs or any(epoch < 0 for epoch in args.epochs):
        parser.error("--epochs must contain non-negative values")
    if args.n_bins is not None and args.n_bins < 2:
        parser.error("--n-bins must be at least 2")
    if args.cv_folds < 0:
        parser.error("--cv-folds must be non-negative")
    return args


def requested_conditions(args: argparse.Namespace, available: set[str]) -> list[str]:
    if args.conditions:
        requested = list(dict.fromkeys(args.conditions))
    else:
        requested = [
            f"{dataset}_{model}_{role}"
            for dataset in args.datasets
            for model in args.models
            for role in ROLES
        ]
    missing = sorted(set(requested) - available)
    if missing:
        warnings.warn(f"Unavailable conditions will be skipped: {missing}")
    return [condition for condition in requested if condition in available]


def load_condition_runs(runs: list) -> list[dict[str, object]]:
    loaded: list[dict[str, object]] = []
    for run_index, run in enumerate(runs):
        perf = pd.read_csv(run.perf_path, low_memory=False)
        loaded.append(
            {
                "run": run,
                "run_index": run_index,
                "perf": perf,
                "resolved": resolve_counter_columns(perf.columns),
                "partial": partial_batches(run.metrics_path)[0],
            }
        )
    return loaded


def pool_observations(
    loaded: list[dict[str, object]],
    *,
    epoch: int,
    counter: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, object]]:
    pieces: list[pd.DataFrame] = []
    trial_interval_counts: dict[str, int] = {}
    scaling: dict[str, object] = {}
    unavailable_trials: list[str] = []
    excluded_partial_rows = 0

    for item in loaded:
        run = item["run"]
        resolved = item["resolved"]
        instruction_column = resolved.get("instructions")
        counter_column = resolved.get(counter)
        if instruction_column is None or counter_column is None:
            unavailable_trials.append(run.trial_id)
            continue
        observations, construction = build_forward_observations(
            item["perf"],
            epoch=epoch,
            phase=args.phase,
            instruction_column=instruction_column,
            counter_column=counter_column,
            partial=item["partial"],
            include_partial=args.include_partial_batch,
            pmu_scaling=args.pmu_scaling,
            counter_normalization="per_instruction",
        )
        excluded_partial_rows += int(construction["excluded_partial_rows"])
        trial_interval_counts[run.trial_id] = len(observations)
        scaling[run.trial_id] = construction["scaling"]
        if observations.empty:
            continue
        observations = observations.copy()
        observations["source_run_id"] = run.run_id
        observations["source_trial_id"] = run.trial_id
        observations["source_batch_idx"] = observations["batch_idx"].astype(int)
        observations["batch_idx"] = (
            int(item["run_index"]) * 10_000 + observations["batch_idx"].astype(int)
        )
        pieces.append(observations)

    pooled = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    details = {
        "trials_available": len(loaded),
        "trials_contributing": int(pooled["source_trial_id"].nunique()) if not pooled.empty else 0,
        "used_batches": int(pooled["batch_idx"].nunique()) if not pooled.empty else 0,
        "trial_interval_counts": trial_interval_counts,
        "excluded_partial_rows": excluded_partial_rows,
        "unavailable_trials": unavailable_trials,
        "pmu_scaling_by_trial": scaling,
    }
    return pooled, details


def fit_pooled_profiles(
    input_dir: Path,
    profile_dir: Path,
    conditions: list[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_runs = discover_runs(input_dir)
    grouped: dict[str, list] = {condition: [] for condition in conditions}
    for run in all_runs:
        if run.condition in grouped:
            grouped[run.condition].append(run)

    profile_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    observation_rows: list[dict[str, object]] = []
    total_groups = len(conditions) * len(args.epochs) * len(args.counters)
    group_index = 0

    for condition in conditions:
        runs = grouped[condition]
        if args.expected_trials and len(runs) != args.expected_trials:
            warnings.warn(
                f"{condition}: found {len(runs)} trials; expected {args.expected_trials}"
            )
        print(f"Loading {condition}: {len(runs)} trials", flush=True)
        loaded = load_condition_runs(runs)
        pooled_run_id = f"pooled:{condition}:{len(runs)}-trials"
        for counter in args.counters:
            for epoch in args.epochs:
                group_index += 1
                print(
                    f"[{group_index}/{total_groups}] {condition} epoch={epoch} counter={counter}",
                    flush=True,
                )
                base = {
                    "condition": condition,
                    "run_id": pooled_run_id,
                    "trial_id": "pooled",
                    "epoch": epoch,
                    "counter": counter,
                    "phase": args.phase,
                    "counter_normalization": "per_instruction",
                    "partial_batch_policy": "included" if args.include_partial_batch else "excluded",
                }
                try:
                    observations, construction = pool_observations(
                        loaded, epoch=epoch, counter=counter, args=args
                    )
                except (KeyError, ValueError) as exc:
                    diagnostic_rows.append({**base, "status": "skipped", "reason": str(exc)})
                    continue
                n = len(observations)
                if n == 0:
                    diagnostic_rows.append(
                        {
                            **base,
                            "status": "skipped",
                            "reason": "no usable pooled forward observations",
                            "N": 0,
                            "used_batches": 0,
                            "construction": json.dumps(construction, sort_keys=True),
                        }
                    )
                    continue

                w, edges, bin_diagnostic, attempts = choose_bin_count(
                    observations["a"].to_numpy(),
                    observations["b"].to_numpy(),
                    requested=args.n_bins,
                    min_coverage=args.min_bin_coverage,
                    max_condition=args.max_condition_number,
                )
                if w is None or edges is None:
                    reason = "no identifiable K: " + " | ".join(
                        f"K={item.k}: {item.reason}" for item in attempts
                    )
                    if args.no_scalar_fallback or n < 4:
                        diagnostic_rows.append(
                            {
                                **base,
                                "status": "skipped",
                                "reason": reason,
                                "N": n,
                                "used_batches": construction["used_batches"],
                                "construction": json.dumps(construction, sort_keys=True),
                            }
                        )
                        continue
                    w, edges = build_overlap_matrix(
                        observations["a"].to_numpy(), observations["b"].to_numpy(), 1
                    )
                    bin_diagnostic = diagnose_bins(
                        w, 1, min_coverage=args.min_bin_coverage,
                        max_condition=args.max_condition_number,
                    )
                    profile_resolution = "whole_forward_scalar"
                    resolution_note = reason
                else:
                    profile_resolution = "progress_profile"
                    resolution_note = ""

                observed = observations["observed_increment"].to_numpy(dtype=float)
                try:
                    fitted = fit_profile(
                        w,
                        observed,
                        tau2=args.tau2,
                        mean_smoothness=args.mean_smoothness,
                        variance_smoothness=args.variance_smoothness,
                        maxiter=args.optimizer_maxiter,
                    )
                except (FloatingPointError, ValueError) as exc:
                    diagnostic_rows.append(
                        {
                            **base,
                            "status": "skipped",
                            "reason": f"optimizer input failure: {exc}",
                            "N": n,
                            "K": bin_diagnostic.k,
                            "profile_resolution": profile_resolution,
                        }
                    )
                    continue
                cv = cross_validate_batches(
                    observations,
                    k=bin_diagnostic.k,
                    folds=args.cv_folds,
                    tau2=args.tau2,
                    mean_smoothness=args.mean_smoothness,
                    variance_smoothness=args.variance_smoothness,
                    random_seed=args.random_seed + 1009 * epoch + group_index,
                )
                diagnostic_rows.append(
                    {
                        **base,
                        "status": "ok" if fitted.success else "optimizer_warning",
                        "reason": "" if fitted.success else fitted.message,
                        "N": n,
                        "K": bin_diagnostic.k,
                        "used_batches": construction["used_batches"],
                        "trials_contributing": construction["trials_contributing"],
                        "profile_resolution": profile_resolution,
                        "resolution_note": resolution_note,
                        "rank": bin_diagnostic.rank,
                        "condition_number": bin_diagnostic.condition_number,
                        "singular_values": json.dumps(bin_diagnostic.singular_values.tolist()),
                        "bin_coverage": json.dumps(bin_diagnostic.coverage.tolist()),
                        "optimizer_success": fitted.success,
                        "optimizer_message": fitted.message,
                        "optimizer_iterations": fitted.iterations,
                        "final_objective": fitted.objective,
                        "tau_squared": fitted.tau2,
                        "representative_interval_width": float(observations["width"].median()),
                        "mean_smoothness": args.mean_smoothness,
                        "variance_smoothness": args.variance_smoothness,
                        "construction": json.dumps(construction, sort_keys=True),
                        **residual_statistics(observed, fitted.predicted),
                        **cv,
                    }
                )
                for bin_index, (start, end, mean_rate, variance_rate) in enumerate(
                    zip(edges[:-1], edges[1:], fitted.mu, fitted.q)
                ):
                    profile_rows.append(
                        {
                            "condition": condition,
                            "run_id": pooled_run_id,
                            "trial_id": "pooled",
                            "epoch": epoch,
                            "counter": counter,
                            "bin_index": bin_index,
                            "progress_start": start,
                            "progress_end": end,
                            "progress_center": (start + end) / 2.0,
                            "estimated_mean_rate": mean_rate,
                            "estimated_variance_rate": variance_rate,
                            "counter_normalization": "per_instruction",
                            "tau_squared": fitted.tau2,
                            "N": n,
                            "K": bin_diagnostic.k,
                            "used_batches": construction["used_batches"],
                            "optimizer_success": fitted.success,
                            "profile_resolution": profile_resolution,
                        }
                    )
                if args.save_observations:
                    for observation_index, row in observations.reset_index(drop=True).iterrows():
                        observation_rows.append(
                            {
                                "condition": condition,
                                "run_id": pooled_run_id,
                                "trial_id": row["source_trial_id"],
                                "epoch": epoch,
                                "counter": counter,
                                "observation_index": observation_index,
                                "batch_idx": int(row["source_batch_idx"]),
                                "progress_start": row["a"],
                                "progress_end": row["b"],
                                "batch_total_instructions": row["batch_total_instructions"],
                                "raw_counter_increment": row["raw_counter_increment"],
                                "observed_increment": row["observed_increment"],
                                "predicted_increment": fitted.predicted[observation_index],
                                "residual": fitted.residuals[observation_index],
                            }
                        )

    profile_dir.mkdir(parents=True, exist_ok=True)
    profiles = pd.DataFrame(profile_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    observations = pd.DataFrame(observation_rows)
    profiles.to_csv(profile_dir / "profiles.csv", index=False)
    diagnostics.to_csv(profile_dir / "diagnostics.csv", index=False)
    if args.save_observations:
        observations.to_csv(profile_dir / "observations.csv", index=False)
    summary = {
        "input_dir": str(input_dir),
        "profile_dir": str(profile_dir),
        "pooling_unit": "condition, epoch, counter",
        "normalization": (
            "Each batch is independently converted to instruction progress and interval "
            "counter/batch-total-instructions before observations are pooled across trials"
        ),
        "resolution_policy": (
            "Use the largest identifiable multi-bin progress profile. If the overlap matrix "
            "is rank 1, use K=1 and report only a whole-forward counter/instruction distribution."
        ),
        "conditions": conditions,
        "epochs": args.epochs,
        "counters": args.counters,
        "observations_saved": args.save_observations,
        "successful_profiles": int(diagnostics["status"].eq("ok").sum()) if not diagnostics.empty else 0,
        "skipped_profiles": int(diagnostics["status"].eq("skipped").sum()) if not diagnostics.empty else 0,
    }
    (profile_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return profiles, diagnostics


def run_comparison(
    input_dir: Path,
    profile_dir: Path,
    comparison_dir: Path,
    args: argparse.Namespace,
) -> None:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "2_augmentation_statistical_comparison.py"),
        "--input-dir",
        str(input_dir),
        "--profile-dir",
        str(profile_dir),
        "--output-dir",
        str(comparison_dir),
        "--datasets",
        *args.datasets,
        "--models",
        *args.models,
        "--epochs",
        *(str(epoch) for epoch in args.epochs),
        "--min-paired-epochs",
        str(args.min_paired_epochs),
        "--no-auto-estimate",
        "--overwrite",
    ]
    if args.counters:
        command.extend(("--counters", *args.counters))
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    profile_dir = output_dir / "profile_cache"
    comparison_dir = output_dir / "comparison"
    marker = profile_dir / "profiles.csv"
    if marker.exists() and not args.overwrite:
        raise FileExistsError(f"{marker} exists; pass --overwrite or choose another output directory")

    runs = discover_runs(input_dir)
    available = {run.condition for run in runs}
    conditions = requested_conditions(args, available)
    if not conditions:
        raise ValueError("No requested motivational-study conditions were found")
    profiles, diagnostics = fit_pooled_profiles(input_dir, profile_dir, conditions, args)
    successful = diagnostics[diagnostics["status"].eq("ok")]
    print(
        f"Pooled profiles: {len(successful)} successful, "
        f"{diagnostics['status'].eq('skipped').sum()} skipped; rows={len(profiles)}",
        flush=True,
    )
    if not args.skip_comparison:
        run_comparison(input_dir, profile_dir, comparison_dir, args)
    print(f"Saved pooled analysis to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
