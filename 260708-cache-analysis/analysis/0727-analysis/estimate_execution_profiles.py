#!/usr/bin/env python3
"""Estimate instruction-progress distributional execution profiles from perf logs."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-execution-profiles")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from execution_profiles import (
    COUNTER_ALIASES,
    DEFAULT_COUNTERS,
    build_forward_observations,
    choose_bin_count,
    cross_validate_batches,
    discover_runs,
    fit_profile,
    inspect_runs,
    partial_batches,
    residual_statistics,
    resolve_counter_columns,
    write_investigation,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "cache_0727_jetson_cpu" / "192.168.0.141"
DEFAULT_OUTPUT = DEFAULT_INPUT / "profile_outputs"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phase", default="forward")
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--epochs", nargs="+", type=int, default=list(range(10)))
    parser.add_argument(
        "--counters",
        nargs="+",
        choices=tuple(COUNTER_ALIASES),
        default=list(DEFAULT_COUNTERS),
    )
    parser.add_argument("--n-bins", type=int)
    parser.add_argument("--tau2", type=float)
    parser.add_argument("--mean-smoothness", type=float, default=1.0)
    parser.add_argument("--variance-smoothness", type=float, default=1.0)
    partial = parser.add_mutually_exclusive_group()
    partial.add_argument("--include-partial-batch", action="store_true")
    partial.add_argument("--exclude-partial-batch", action="store_false", dest="include_partial_batch")
    parser.set_defaults(include_partial_batch=False)
    parser.add_argument("--pmu-scaling", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--min-bin-coverage", type=int, default=4)
    parser.add_argument("--max-condition-number", type=float, default=1e8)
    parser.add_argument("--optimizer-maxiter", type=int, default=1500)
    parser.add_argument("--random-seed", type=int, default=260727)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not args.epochs or any(epoch < 0 for epoch in args.epochs):
        parser.error("--epochs must contain non-negative epoch numbers")
    if args.n_bins is not None and args.n_bins < 2:
        parser.error("--n-bins must be at least 2")
    if args.tau2 is not None and args.tau2 < 0:
        parser.error("--tau2 must be non-negative")
    if args.mean_smoothness < 0 or args.variance_smoothness < 0:
        parser.error("smoothness coefficients must be non-negative")
    if args.cv_folds < 0:
        parser.error("--cv-folds must be non-negative")
    return args


def safe_name(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "._-" else "_" for character in value)
    return cleaned.strip("_") or "value"


def step_coordinates(edges: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.repeat(edges, 2)[1:-1], np.repeat(values, 2)


def plot_profile(
    *,
    observations: pd.DataFrame,
    edges: np.ndarray,
    mu: np.ndarray,
    q: np.ndarray,
    tau2: float,
    condition: str,
    epoch: int,
    counter: str,
    n: int,
    k: int,
    batches: int,
    output_prefix: Path,
    dpi: int,
) -> None:
    centers = (edges[:-1] + edges[1:]) / 2.0
    x_step, mu_step = step_coordinates(edges, mu)
    _, q_step = step_coordinates(edges, q)
    representative_width = float(observations["width"].median())
    rate_sd = np.sqrt(np.maximum(representative_width * q + tau2, 0.0)) / representative_width
    _, lower_step = step_coordinates(edges, mu - rate_sd)
    _, upper_step = step_coordinates(edges, mu + rate_sd)
    observed_rate = observations["observed_increment"] / observations["width"]
    midpoint = (observations["a"] + observations["b"]) / 2.0

    figure, axes = plt.subplots(2, 1, figsize=(11.5, 7.8), sharex=True, gridspec_kw={"height_ratios": (2.0, 1.0)})
    mean_axis, variance_axis = axes
    mean_axis.fill_between(
        x_step,
        lower_step,
        upper_step,
        color="#3f78a8",
        alpha=0.18,
        label=f"Approx. rate variation for median interval width h={representative_width:.3f}",
    )
    mean_axis.plot(x_step, mu_step, color="#155b8a", linewidth=2.0, label="Estimated mean increment rate")
    mean_axis.scatter(midpoint, observed_rate, s=15, color="#c84c3a", alpha=0.58, label="Observed interval rate")
    for start, end, rate in zip(observations["a"], observations["b"], observed_rate):
        mean_axis.hlines(rate, start, end, color="#c84c3a", alpha=0.18, linewidth=0.8)
    mean_axis.set_ylabel("Counter increment / progress")
    mean_axis.grid(True, color="#d9dde1", linewidth=0.55)
    mean_axis.legend(loc="best", fontsize=8, frameon=False)
    mean_axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))

    variance_axis.plot(x_step, q_step, color="#6e4a8e", linewidth=1.8)
    variance_axis.fill_between(x_step, 0.0, q_step, color="#6e4a8e", alpha=0.15)
    rug_height = float(np.nanmax(q)) * 0.035 if np.any(np.isfinite(q)) and np.nanmax(q) > 0 else 1.0
    variance_axis.vlines(midpoint, 0, rug_height, color="#56595c", alpha=0.25, linewidth=0.5)
    variance_axis.set_ylabel("Variance rate q(p)")
    variance_axis.set_xlabel("Relative retired-instruction progress p")
    variance_axis.set_xlim(0.0, 1.0)
    variance_axis.set_ylim(bottom=0.0)
    variance_axis.grid(True, color="#d9dde1", linewidth=0.55)
    variance_axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))

    figure.suptitle(
        f"{condition} | epoch {epoch:02d} | {counter}\n"
        f"N={n} interval observations, K={k} bins, batches={batches}; "
        "band is a representative-interval variation scale, not a confidence interval",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.01, 0.01, 0.99, 0.93))
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_prefix.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_reconstruction(
    observations: pd.DataFrame,
    predicted: np.ndarray,
    *,
    condition: str,
    epoch: int,
    counter: str,
    path: Path,
    dpi: int,
) -> None:
    observed = observations["observed_increment"].to_numpy(dtype=float)
    residual = observed - predicted
    midpoint = (observations["a"].to_numpy() + observations["b"].to_numpy()) / 2.0
    lower = float(min(observed.min(), predicted.min()))
    upper = float(max(observed.max(), predicted.max()))
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    axes[0].scatter(observed, predicted, s=18, alpha=0.65, color="#276b91")
    axes[0].plot([lower, upper], [lower, upper], color="#343638", linewidth=1.0, linestyle="--")
    axes[0].set_xlabel("Observed interval increment")
    axes[0].set_ylabel("Reconstructed interval increment")
    axes[0].grid(True, color="#d9dde1", linewidth=0.55)
    axes[1].scatter(midpoint, residual, s=18, alpha=0.65, color="#b64b40")
    axes[1].axhline(0.0, color="#343638", linewidth=1.0, linestyle="--")
    axes[1].set_xlabel("Interval progress midpoint")
    axes[1].set_ylabel("Observed - reconstructed")
    axes[1].grid(True, color="#d9dde1", linewidth=0.55)
    figure.suptitle(f"Reconstruction check | {condition} | epoch {epoch:02d} | {counter}")
    figure.tight_layout(rect=(0.01, 0.01, 0.99, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_overviews(profiles: pd.DataFrame, output_dir: Path, dpi: int) -> int:
    if profiles.empty:
        return 0
    count = 0
    epoch_colors = plt.get_cmap("viridis", 10)
    for (condition, counter), group in profiles.groupby(["condition", "counter"], sort=True):
        figure, axes = plt.subplots(2, 1, figsize=(10.8, 7.0), sharex=True)
        for epoch, epoch_frame in group.groupby("epoch", sort=True):
            epoch_frame = epoch_frame.sort_values("bin_index")
            color = epoch_colors(int(epoch) % 10)
            axes[0].plot(
                epoch_frame["progress_center"],
                epoch_frame["estimated_mean_rate"],
                color=color,
                linewidth=1.5,
                label=f"epoch {int(epoch)}",
            )
            axes[1].plot(
                epoch_frame["progress_center"],
                epoch_frame["estimated_variance_rate"],
                color=color,
                linewidth=1.5,
            )
        axes[0].set_ylabel("Mean increment rate")
        axes[1].set_ylabel("Variance rate")
        axes[1].set_xlabel("Relative retired-instruction progress p")
        axes[1].set_xlim(0, 1)
        for axis in axes:
            axis.grid(True, color="#d9dde1", linewidth=0.55)
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
        axes[0].legend(ncol=5, fontsize=7, frameon=False)
        figure.suptitle(f"{condition} | {counter} | epoch profiles")
        figure.tight_layout(rect=(0.01, 0.01, 0.99, 0.94))
        path = output_dir / safe_name(str(condition)) / "overviews" / f"{safe_name(str(counter))}_epochs_overlay.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        count += 1
    return count


def write_tables(output_dir: Path, profiles: list[dict], diagnostics: list[dict], observations: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    profile_frame = pd.DataFrame(profiles)
    diagnostic_frame = pd.DataFrame(diagnostics)
    observation_frame = pd.DataFrame(observations)
    profile_frame.to_csv(output_dir / "profiles.csv", index=False)
    diagnostic_frame.to_csv(output_dir / "diagnostics.csv", index=False)
    observation_frame.to_csv(output_dir / "observations.csv", index=False)
    try:
        profile_frame.to_parquet(output_dir / "profiles.parquet", index=False)
        diagnostic_frame.to_parquet(output_dir / "diagnostics.parquet", index=False)
    except (ImportError, ValueError, OSError) as exc:
        warnings.warn(f"Parquet output skipped: {exc}")
    return profile_frame, diagnostic_frame


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    marker_files = (output_dir / "profiles.csv", output_dir / "run_summary.json")
    if not args.overwrite and any(path.exists() for path in marker_files):
        raise FileExistsError(f"Output already exists below {output_dir}; pass --overwrite to replace generated tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(input_dir)
    if args.conditions:
        requested = set(args.conditions)
        runs = [run for run in runs if run.condition in requested]
        missing_conditions = sorted(requested - {run.condition for run in runs})
        if missing_conditions:
            warnings.warn(f"Requested conditions not found: {missing_conditions}")
    if not runs:
        raise ValueError("No runs remain after condition filtering")

    investigation = inspect_runs(runs, args.phase)
    write_investigation(investigation, output_dir)
    profile_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    observation_rows: list[dict] = []
    warning_messages: list[str] = []

    for run_index, run in enumerate(runs):
        print(f"[{run_index + 1}/{len(runs)}] {run.condition}: {run.perf_path.name}", flush=True)
        perf = pd.read_csv(run.perf_path, low_memory=False)
        resolved = resolve_counter_columns(perf.columns)
        instruction_column = resolved.get("instructions")
        partial, batch_sizes = partial_batches(run.metrics_path)
        if instruction_column is None:
            message = f"{run.condition}: retired-instruction column is unavailable; all counters skipped"
            warnings.warn(message)
            warning_messages.append(message)
            continue
        for counter in args.counters:
            counter_column = resolved.get(counter)
            if counter_column is None:
                message = f"{run.condition}: counter {counter!r} is unavailable and was skipped"
                warnings.warn(message)
                warning_messages.append(message)
                for epoch in args.epochs:
                    diagnostic_rows.append(
                        {
                            "condition": run.condition,
                            "run_id": run.run_id,
                            "epoch": epoch,
                            "counter": counter,
                            "status": "skipped",
                            "reason": "counter column unavailable",
                        }
                    )
                continue
            for epoch in args.epochs:
                base_diagnostic = {
                    "condition": run.condition,
                    "run_id": run.run_id,
                    "epoch": epoch,
                    "counter": counter,
                    "phase": args.phase,
                    "instruction_column": instruction_column,
                    "counter_column": counter_column,
                    "partial_batch_policy": "included" if args.include_partial_batch else "excluded",
                }
                try:
                    observations, construction = build_forward_observations(
                        perf,
                        epoch=epoch,
                        phase=args.phase,
                        instruction_column=instruction_column,
                        counter_column=counter_column,
                        partial=partial,
                        include_partial=args.include_partial_batch,
                        pmu_scaling=args.pmu_scaling,
                    )
                except (ValueError, KeyError) as exc:
                    diagnostic_rows.append({**base_diagnostic, "status": "skipped", "reason": str(exc)})
                    warning_messages.append(f"{run.condition}/epoch {epoch}/{counter}: {exc}")
                    continue
                n = len(observations)
                if n == 0:
                    diagnostic_rows.append(
                        {
                            **base_diagnostic,
                            "status": "skipped",
                            "reason": "no usable forward interval observations",
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
                    diagnostic_rows.append(
                        {
                            **base_diagnostic,
                            "status": "skipped",
                            "reason": reason,
                            "N": n,
                            "used_batches": construction["used_batches"],
                            "intervals_per_batch": json.dumps(construction["intervals_per_batch"], sort_keys=True),
                            "construction": json.dumps(construction, sort_keys=True),
                        }
                    )
                    warning_messages.append(f"{run.condition}/epoch {epoch}/{counter}: {reason}")
                    continue
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
                except (ValueError, FloatingPointError) as exc:
                    diagnostic_rows.append(
                        {
                            **base_diagnostic,
                            "status": "skipped",
                            "reason": f"optimizer input failure: {exc}",
                            "N": n,
                            "K": bin_diagnostic.k,
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
                    random_seed=args.random_seed + 1009 * epoch + run_index,
                )
                residual = residual_statistics(observed, fitted.predicted)
                diagnostic_rows.append(
                    {
                        **base_diagnostic,
                        "status": "ok" if fitted.success else "optimizer_warning",
                        "reason": "" if fitted.success else fitted.message,
                        "N": n,
                        "K": bin_diagnostic.k,
                        "used_batches": construction["used_batches"],
                        "intervals_per_batch": json.dumps(construction["intervals_per_batch"], sort_keys=True),
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
                        "pmu_scaling": json.dumps(construction["scaling"], sort_keys=True),
                        "construction": json.dumps(construction, sort_keys=True),
                        **residual,
                        **cv,
                    }
                )
                for bin_index, (start, end, mean_rate, variance_rate) in enumerate(
                    zip(edges[:-1], edges[1:], fitted.mu, fitted.q)
                ):
                    profile_rows.append(
                        {
                            "condition": run.condition,
                            "run_id": run.run_id,
                            "epoch": epoch,
                            "counter": counter,
                            "bin_index": bin_index,
                            "progress_start": start,
                            "progress_end": end,
                            "progress_center": (start + end) / 2.0,
                            "estimated_mean_rate": mean_rate,
                            "estimated_variance_rate": variance_rate,
                            "tau_squared": fitted.tau2,
                            "N": n,
                            "K": bin_diagnostic.k,
                            "used_batches": construction["used_batches"],
                            "optimizer_success": fitted.success,
                        }
                    )
                for observation_index, (_, row) in enumerate(observations.iterrows()):
                    observation_rows.append(
                        {
                            "condition": run.condition,
                            "run_id": run.run_id,
                            "epoch": epoch,
                            "counter": counter,
                            "observation_index": observation_index,
                            "batch_idx": int(row["batch_idx"]),
                            "progress_start": row["a"],
                            "progress_end": row["b"],
                            "observed_increment": row["observed_increment"],
                            "predicted_increment": fitted.predicted[observation_index],
                            "residual": fitted.residuals[observation_index],
                        }
                    )
                if not args.no_plots:
                    epoch_dir = output_dir / safe_name(run.condition) / f"epoch_{epoch:02d}"
                    prefix = epoch_dir / f"{safe_name(counter)}_profile"
                    plot_profile(
                        observations=observations,
                        edges=edges,
                        mu=fitted.mu,
                        q=fitted.q,
                        tau2=fitted.tau2,
                        condition=run.condition,
                        epoch=epoch,
                        counter=counter,
                        n=n,
                        k=bin_diagnostic.k,
                        batches=int(construction["used_batches"]),
                        output_prefix=prefix,
                        dpi=args.dpi,
                    )
                    plot_reconstruction(
                        observations,
                        fitted.predicted,
                        condition=run.condition,
                        epoch=epoch,
                        counter=counter,
                        path=epoch_dir / f"{safe_name(counter)}_reconstruction.png",
                        dpi=args.dpi,
                    )

    profile_frame, diagnostic_frame = write_tables(output_dir, profile_rows, diagnostic_rows, observation_rows)
    overview_count = 0 if args.no_plots else plot_overviews(profile_frame, output_dir, args.dpi)
    successful = int(diagnostic_frame["status"].isin(["ok", "optimizer_warning"]).sum()) if not diagnostic_frame.empty else 0
    skipped = int(diagnostic_frame["status"].eq("skipped").sum()) if not diagnostic_frame.empty else 0
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "conditions": [run.condition for run in runs],
        "epochs": args.epochs,
        "counters": args.counters,
        "phase": args.phase,
        "include_partial_batch": args.include_partial_batch,
        "pmu_scaling": args.pmu_scaling,
        "successful_epoch_counter_profiles": successful,
        "skipped_epoch_counter_profiles": skipped,
        "profile_rows": len(profile_frame),
        "overview_plots": overview_count,
        "warnings": warning_messages,
        "investigation": "data_investigation.json",
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    print(
        f"Completed: {successful} profiles, {skipped} skipped; outputs: {output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
