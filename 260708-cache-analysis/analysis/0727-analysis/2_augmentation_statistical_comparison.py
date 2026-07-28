#!/usr/bin/env python3
"""Compare progress-aligned execution distributions across training conditions."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-distribution-comparison")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_COLLECTION = SCRIPT_DIR / "cache_0727_jetson_cpu_20_trials"
DEFAULT_INPUT = DEFAULT_COLLECTION / "192.168.0.141"
DEFAULT_PROFILE_DIR = DEFAULT_COLLECTION / "distribution_profile_cache_per_instruction"
DEFAULT_OUTPUT = DEFAULT_COLLECTION / "distribution_statistical_comparison_per_instruction"
TARGET_LABELS = {
    "availability_shortcut": "shortcut",
    "non_iid": "non-IID",
    "strong_augmentation": "strong augmentation",
}
COLORS = {
    "clean": "#2878b5",
    "moderate": "#3a923a",
    "normal_reference": "#202326",
    "availability_shortcut": "#c84c3a",
    "non_iid": "#8b5ea7",
    "strong_augmentation": "#d18b24",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--profile-dir", type=Path, default=DEFAULT_PROFILE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--datasets", nargs="+", choices=("cifar10", "trashnet"), default=("cifar10", "trashnet"))
    parser.add_argument("--models", nargs="+", choices=("cnn", "vit"), default=("cnn", "vit"))
    parser.add_argument("--epochs", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--counters", nargs="+")
    parser.add_argument("--progress-points", type=int, default=101)
    parser.add_argument("--min-paired-epochs", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument(
        "--profile-cv-folds",
        type=int,
        default=0,
        help="Batch CV folds used only when profiles must be estimated; 0 avoids repeated fitting.",
    )
    parser.add_argument("--rebuild-profiles", action="store_true")
    parser.add_argument("--no-auto-estimate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.progress_points < 11:
        parser.error("--progress-points must be at least 11")
    if args.min_paired_epochs < 2:
        parser.error("--min-paired-epochs must be at least 2")
    if not 0.0 < args.alpha < 1.0:
        parser.error("--alpha must be in (0, 1)")
    if not args.epochs or any(epoch < 0 for epoch in args.epochs):
        parser.error("--epochs must contain non-negative values")
    if args.profile_cv_folds < 0:
        parser.error("--profile-cv-folds must be non-negative")
    return args


def condition_name(dataset: str, model: str, role: str) -> str:
    prefix = dataset if model == "cnn" else f"{dataset}_vit"
    if role == "clean":
        return prefix if model == "vit" else f"{dataset}_iid"
    return f"{prefix}_{role}"


def condition_name_candidates(dataset: str, model: str, role: str) -> tuple[str, ...]:
    """Support both the cache-study and motivational-study directory names."""
    canonical = condition_name(dataset, model, role)
    motivational = f"{dataset}_{model}_{role}"
    return tuple(dict.fromkeys((canonical, motivational)))


def requested_condition_map(
    datasets: list[str],
    models: list[str],
    available: set[str] | None = None,
) -> dict[tuple[str, str], dict[str, str]]:
    output: dict[tuple[str, str], dict[str, str]] = {}
    for dataset in datasets:
        for model in models:
            conditions: dict[str, str] = {}
            roles = {
                "clean": "clean",
                "moderate": "moderate_augmentation",
                **{role: role for role in TARGET_LABELS},
            }
            for role, directory_role in roles.items():
                candidates = condition_name_candidates(dataset, model, directory_role)
                conditions[role] = next(
                    (candidate for candidate in candidates if available is not None and candidate in available),
                    candidates[0],
                )
            output[(dataset, model)] = conditions
    return output


def available_raw_conditions(input_dir: Path) -> set[str]:
    if not input_dir.is_dir():
        return set()
    return {
        path.name
        for path in input_dir.iterdir()
        if path.is_dir() and any(path.glob("*_perf.csv"))
    }


def resolve_input_dir(path: Path) -> Path:
    path = path.resolve()
    if available_raw_conditions(path):
        return path
    candidates = [
        child
        for child in path.iterdir()
        if child.is_dir() and available_raw_conditions(child)
    ] if path.is_dir() else []
    if len(candidates) == 1:
        print(f"Detected device log directory: {candidates[0]}", flush=True)
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No condition directories containing *_perf.csv were found in {path} "
            "or one directory below it"
        )
    raise ValueError(
        f"Multiple device log directories found below {path}: "
        f"{[candidate.name for candidate in candidates]}; pass one device directory"
    )


def estimate_profiles(
    input_dir: Path,
    profile_dir: Path,
    conditions: list[str],
    epochs: list[int],
    counters: list[str] | None,
    cv_folds: int,
) -> None:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "estimate_execution_profiles.py"),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(profile_dir),
        "--conditions",
        *conditions,
        "--epochs",
        *(str(epoch) for epoch in epochs),
        "--counter-normalization",
        "per_instruction",
        "--cv-folds",
        str(cv_folds),
    ]
    if counters:
        command.extend(("--counters", *counters))
    command.extend(("--no-plots", "--overwrite"))
    print("Estimating missing progress profiles (plots disabled)...", flush=True)
    subprocess.run(command, check=True)


def ensure_profile_tables(args: argparse.Namespace, required: set[str]) -> tuple[Path, Path]:
    profile_dir = args.profile_dir.resolve()
    profiles_path = profile_dir / "profiles.csv"
    diagnostics_path = profile_dir / "diagnostics.csv"
    existing: set[str] = set()
    compatible = False
    if profiles_path.exists():
        columns = pd.read_csv(profiles_path, nrows=0).columns
        if "counter_normalization" in columns:
            cached = pd.read_csv(profiles_path, usecols=["condition", "counter_normalization"])
            compatible = cached["counter_normalization"].astype(str).eq("per_instruction").all()
            if compatible:
                existing = set(cached["condition"].dropna().astype(str))
        if not compatible:
            warnings.warn(f"Ignoring incompatible non-normalized profile cache: {profiles_path}")
    missing = required - existing
    if args.rebuild_profiles or missing:
        if args.no_auto_estimate:
            raise FileNotFoundError(
                f"Profile cache lacks conditions {sorted(missing)}; rerun without --no-auto-estimate"
            )
        estimate_profiles(
            args.input_dir.resolve(),
            profile_dir,
            sorted(required),
            list(args.epochs),
            args.counters,
            args.profile_cv_folds,
        )
    if not profiles_path.exists() or not diagnostics_path.exists():
        raise FileNotFoundError(f"Missing profile tables below {profile_dir}")
    return profiles_path, diagnostics_path


def successful_rows(frame: pd.DataFrame) -> pd.DataFrame:
    success = frame["optimizer_success"]
    if success.dtype != bool:
        success = success.astype(str).str.lower().isin(("true", "1", "yes"))
    numeric = frame["estimated_mean_rate"].notna() & frame["estimated_variance_rate"].notna()
    return frame.loc[success & numeric].copy()


def resample_profile(group: pd.DataFrame, progress: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    ordered = group.sort_values("bin_index")
    starts = ordered["progress_start"].to_numpy(dtype=float)
    ends = ordered["progress_end"].to_numpy(dtype=float)
    if len(starts) == 0 or starts[0] > 1e-8 or ends[-1] < 1.0 - 1e-8:
        raise ValueError("profile bins do not cover [0, 1]")
    index = np.searchsorted(ends, progress, side="right")
    index = np.minimum(index, len(ends) - 1)
    if np.any(progress < starts[index] - 1e-8) or np.any(progress > ends[index] + 1e-8):
        raise ValueError("profile bins contain a progress gap")
    mu = ordered["estimated_mean_rate"].to_numpy(dtype=float)[index]
    q = ordered["estimated_variance_rate"].to_numpy(dtype=float)[index]
    tau2 = float(ordered["tau_squared"].iloc[0])
    return mu, np.maximum(q, 0.0), max(tau2, 0.0)


def representative_widths(diagnostics: pd.DataFrame) -> dict[tuple[str, str, str, int], float]:
    valid = diagnostics[diagnostics["status"].astype(str).eq("ok")].copy()
    if "run_id" not in valid:
        valid["run_id"] = "run_0"
    valid["representative_interval_width"] = pd.to_numeric(
        valid["representative_interval_width"], errors="coerce"
    )
    return {
        (str(row.condition), str(row.run_id), str(row.counter), int(row.epoch)): float(row.representative_interval_width)
        for row in valid.itertuples()
        if np.isfinite(row.representative_interval_width) and row.representative_interval_width > 0
    }


def profile_store(
    profiles: pd.DataFrame,
    diagnostics: pd.DataFrame,
    progress: np.ndarray,
) -> dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]]:
    profiles = profiles.copy()
    if "run_id" not in profiles:
        profiles["run_id"] = "run_0"
    widths = representative_widths(diagnostics)
    run_store: dict[tuple[str, str, str, int], tuple[np.ndarray, np.ndarray]] = {}
    for key, group in successful_rows(profiles).groupby(
        ["condition", "run_id", "counter", "epoch"], sort=False
    ):
        condition, run_id, counter, epoch = str(key[0]), str(key[1]), str(key[2]), int(key[3])
        width = widths.get((condition, run_id, counter, epoch))
        if width is None:
            warnings.warn(f"Skipping {key}: missing representative interval width")
            continue
        try:
            mu, q, tau2 = resample_profile(group, progress)
        except ValueError as error:
            warnings.warn(f"Skipping {key}: {error}")
            continue
        # q is a variance density. Convert it to the variance of an interval
        # rate at the observed representative progress width before comparison.
        rate_variance = q / width + tau2 / (width * width)
        run_store[(condition, run_id, counter, epoch)] = (
            mu,
            np.maximum(rate_variance, 0.0),
        )

    grouped: dict[tuple[str, str, int], list[tuple[np.ndarray, np.ndarray]]] = {}
    for (condition, _run_id, counter, epoch), distribution in run_store.items():
        grouped.setdefault((condition, counter, epoch), []).append(distribution)

    store: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]] = {}
    for key, distributions in grouped.items():
        store[key] = aggregate_gaussians(
            np.stack([distribution[0] for distribution in distributions]),
            np.stack([distribution[1] for distribution in distributions]),
        )
    return store


def aggregate_gaussians(mu: np.ndarray, variance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if mu.shape != variance.shape or mu.ndim != 2 or len(mu) == 0:
        raise ValueError("mu and variance must be non-empty matrices with matching shapes")
    mean = mu.mean(axis=0)
    total_variance = (variance + mu * mu).mean(axis=0) - mean * mean
    return mean, np.maximum(total_variance, 0.0)


def moment_match_pair(
    first_mu: np.ndarray,
    first_variance: np.ndarray,
    second_mu: np.ndarray,
    second_variance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    means = np.stack((first_mu, second_mu))
    variances = np.stack((first_variance, second_variance))
    return aggregate_gaussians(means, variances)


def wasserstein_squared(
    first_mu: np.ndarray,
    first_variance: np.ndarray,
    second_mu: np.ndarray,
    second_variance: np.ndarray,
) -> np.ndarray:
    return (first_mu - second_mu) ** 2 + (
        np.sqrt(np.maximum(first_variance, 0.0))
        - np.sqrt(np.maximum(second_variance, 0.0))
    ) ** 2


def paired_distribution_test(
    normal_mu: np.ndarray,
    normal_variance: np.ndarray,
    target_mu: np.ndarray,
    target_variance: np.ndarray,
    progress: np.ndarray,
) -> dict[str, np.ndarray | float | int]:
    n = len(normal_mu)
    if not (normal_mu.shape == normal_variance.shape == target_mu.shape == target_variance.shape):
        raise ValueError("paired distribution arrays must have matching shapes")
    if n > 16:
        raise ValueError("exact paired permutation is limited to 16 pairs")

    reference_mean, reference_variance = aggregate_gaussians(normal_mu, normal_variance)
    target_mean, target_total_variance = aggregate_gaussians(target_mu, target_variance)
    observed = wasserstein_squared(
        reference_mean, reference_variance, target_mean, target_total_variance
    )
    observed_global = float(np.trapezoid(observed, progress))
    permutations = np.empty((2**n, len(progress)), dtype=float)
    global_statistics = np.empty(2**n, dtype=float)

    for permutation_index, choices in enumerate(itertools.product((False, True), repeat=n)):
        swap = np.asarray(choices, dtype=bool)[:, None]
        left_mu = np.where(swap, target_mu, normal_mu)
        left_variance = np.where(swap, target_variance, normal_variance)
        right_mu = np.where(swap, normal_mu, target_mu)
        right_variance = np.where(swap, normal_variance, target_variance)
        left_mean, left_total_variance = aggregate_gaussians(left_mu, left_variance)
        right_mean, right_total_variance = aggregate_gaussians(right_mu, right_variance)
        distance = wasserstein_squared(
            left_mean, left_total_variance, right_mean, right_total_variance
        )
        permutations[permutation_index] = distance
        global_statistics[permutation_index] = np.trapezoid(distance, progress)

    tolerance = max(abs(observed_global), 1.0) * 1e-12
    global_p = float(np.mean(global_statistics >= observed_global - tolerance))
    pointwise_p = np.mean(permutations >= observed[None, :] - tolerance, axis=0)
    scale = np.maximum(np.median(permutations, axis=0), np.finfo(float).eps)
    maximum_statistics = np.max(permutations / scale[None, :], axis=1)
    pointwise_fwer_p = np.mean(
        maximum_statistics[:, None] >= observed[None, :] / scale[None, :] - 1e-12,
        axis=0,
    )
    return {
        "n": n,
        "reference_mean": reference_mean,
        "reference_variance": reference_variance,
        "target_mean": target_mean,
        "target_variance": target_total_variance,
        "wasserstein_squared": observed,
        "global_wasserstein_squared": observed_global,
        "global_p": global_p,
        "pointwise_p": pointwise_p,
        "pointwise_fwer_p": pointwise_fwer_p,
    }


def benjamini_hochberg(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape, np.nan)
    valid = np.isfinite(values)
    if not valid.any():
        return result
    selected = values[valid]
    order = np.argsort(selected)
    ranked = selected[order] * len(selected) / np.arange(1, len(selected) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    restored = np.empty_like(ranked)
    restored[order] = np.minimum(ranked, 1.0)
    result[valid] = restored
    return result


def collect_epoch_pairs(
    store: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]],
    conditions: dict[str, str],
    counter: str,
    target_role: str,
    epochs: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    normal_mu: list[np.ndarray] = []
    normal_variance: list[np.ndarray] = []
    target_mu: list[np.ndarray] = []
    target_variance: list[np.ndarray] = []
    used_epochs: list[int] = []
    for epoch in epochs:
        keys = (
            (conditions["clean"], counter, epoch),
            (conditions["moderate"], counter, epoch),
            (conditions[target_role], counter, epoch),
        )
        if any(key not in store for key in keys):
            continue
        clean, moderate, target = (store[key] for key in keys)
        mean, variance = moment_match_pair(clean[0], clean[1], moderate[0], moderate[1])
        normal_mu.append(mean)
        normal_variance.append(variance)
        target_mu.append(target[0])
        target_variance.append(target[1])
        used_epochs.append(epoch)
    return (
        np.asarray(normal_mu),
        np.asarray(normal_variance),
        np.asarray(target_mu),
        np.asarray(target_variance),
        used_epochs,
    )


def condition_aggregate_rows(
    store: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]],
    condition: str,
    role: str,
    dataset: str,
    model: str,
    counter: str,
    epochs: list[int],
    progress: np.ndarray,
) -> list[dict[str, object]]:
    available = [(epoch, store[(condition, counter, epoch)]) for epoch in epochs if (condition, counter, epoch) in store]
    if not available:
        return []
    mean, variance = aggregate_gaussians(
        np.stack([item[1][0] for item in available]),
        np.stack([item[1][1] for item in available]),
    )
    return [
        {
            "dataset": dataset,
            "model": model,
            "role": role,
            "condition": condition,
            "counter": counter,
            "progress": float(point),
            "mean_rate": float(mean[index]),
            "rate_variance": float(variance[index]),
            "rate_sd": float(np.sqrt(variance[index])),
            "epochs": len(available),
        }
        for index, point in enumerate(progress)
    ]


def plot_metric(
    pointwise: pd.DataFrame,
    aggregates: pd.DataFrame,
    *,
    dataset: str,
    model: str,
    counter: str,
    output_dir: Path,
    alpha: float,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(11.2, 10.2), sharex=True)
    mean_axis, sd_axis, distance_axis = axes
    selected_aggregates = aggregates[
        aggregates["dataset"].eq(dataset)
        & aggregates["model"].eq(model)
        & aggregates["counter"].eq(counter)
    ]
    for role, group in selected_aggregates.groupby("role", sort=False):
        label = role.replace("_", " ")
        linewidth = 2.4 if role == "normal_reference" else 1.45
        alpha_value = 1.0 if role == "normal_reference" else 0.85
        mean_axis.plot(group["progress"], group["mean_rate"], color=COLORS[role], linewidth=linewidth, alpha=alpha_value, label=label)
        sd_axis.plot(group["progress"], group["rate_sd"], color=COLORS[role], linewidth=linewidth, alpha=alpha_value, label=label)

    selected_points = pointwise[
        pointwise["dataset"].eq(dataset)
        & pointwise["model"].eq(model)
        & pointwise["counter"].eq(counter)
    ]
    for role, group in selected_points.groupby("target_role", sort=False):
        color = COLORS[role]
        distance_axis.plot(group["progress"], group["wasserstein_distance"], color=color, linewidth=1.7, label=TARGET_LABELS[role])
        significant = group["pointwise_fwer_p"].le(alpha).to_numpy(dtype=bool)
        distance_axis.scatter(
            group.loc[significant, "progress"],
            group.loc[significant, "wasserstein_distance"],
            color=color,
            s=13,
            marker="o",
            zorder=3,
        )

    mean_axis.set_ylabel("Mean counter / instruction")
    sd_axis.set_ylabel("Counter/instruction\nrate SD")
    distance_axis.set_ylabel("Gaussian W2 distance")
    distance_axis.set_xlabel("Relative retired-instruction progress")
    distance_axis.set_xlim(0.0, 1.0)
    for axis in axes:
        axis.grid(True, color="#d9dde1", linewidth=0.55)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
        axis.legend(loc="best", fontsize=8, frameon=False, ncol=2)
    figure.suptitle(
        f"{dataset} | {model.upper()} | {counter}\n"
        f"Normal reference = moment-matched clean + moderate; dots: pointwise max-permutation p <= {alpha:g}",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.01, 0.01, 0.99, 0.94))
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / counter
    figure.savefig(prefix.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_global_summary(tests: pd.DataFrame, output_dir: Path, alpha: float, dpi: int) -> None:
    if tests.empty:
        return
    comparisons = tests["target_role"].drop_duplicates().tolist()
    metrics = sorted(tests["counter"].unique())
    matrix = tests.pivot(index="counter", columns="target_role", values="normalized_global_w2").reindex(index=metrics, columns=comparisons)
    q_values = tests.pivot(index="counter", columns="target_role", values="global_fdr_q").reindex(index=metrics, columns=comparisons)
    figure, axis = plt.subplots(figsize=(2.3 + 2.2 * len(comparisons), 1.6 + 0.42 * len(metrics)))
    image = axis.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="magma")
    axis.set_xticks(np.arange(len(comparisons)), [TARGET_LABELS[item] for item in comparisons], rotation=20, ha="right")
    axis.set_yticks(np.arange(len(metrics)), metrics)
    for row in range(len(metrics)):
        for column in range(len(comparisons)):
            value = matrix.iloc[row, column]
            if not np.isfinite(value):
                continue
            marker = "*" if q_values.iloc[row, column] <= alpha else ""
            axis.text(column, row, f"{value:.2g}{marker}", ha="center", va="center", color="white", fontsize=8)
    axis.set_title("Integrated distribution distance / median normal |mean|\n* BH-FDR significant global paired permutation test")
    figure.colorbar(image, ax=axis, label="Normalized integrated W2")
    figure.tight_layout()
    figure.savefig(output_dir / "global_distribution_tests.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(output_dir / "global_distribution_tests.pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    input_dir = resolve_input_dir(args.input_dir)
    args.input_dir = input_dir
    output_dir = args.output_dir.resolve()
    if not args.overwrite and (output_dir / "global_distribution_tests.csv").exists():
        raise FileExistsError(f"Output exists below {output_dir}; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_conditions = available_raw_conditions(input_dir)
    maps = requested_condition_map(list(args.datasets), list(args.models), raw_conditions)
    usable_maps: dict[tuple[str, str], dict[str, str]] = {}
    required: set[str] = set()
    warnings_list: list[str] = []
    for key, conditions in maps.items():
        missing_normal = [role for role in ("clean", "moderate") if conditions[role] not in raw_conditions]
        if missing_normal:
            message = f"Skipping {key}: missing normal conditions {missing_normal}"
            warnings.warn(message)
            warnings_list.append(message)
            continue
        usable = {role: name for role, name in conditions.items() if name in raw_conditions}
        for role in TARGET_LABELS:
            if role not in usable:
                message = f"{key}: {role} unavailable and will be skipped"
                warnings.warn(message)
                warnings_list.append(message)
        usable_maps[key] = usable
        required.update(usable.values())

    if not required:
        raise RuntimeError(f"No requested condition groups are available below {input_dir}")

    profiles_path, diagnostics_path = ensure_profile_tables(args, required)
    profiles = pd.read_csv(profiles_path)
    diagnostics = pd.read_csv(diagnostics_path)
    runs_per_condition = (
        profiles.groupby("condition")["run_id"].nunique().astype(int).to_dict()
        if "run_id" in profiles
        else {condition: 1 for condition in profiles["condition"].dropna().astype(str).unique()}
    )
    profiles = profiles[profiles["condition"].isin(required) & profiles["epoch"].isin(args.epochs)]
    diagnostics = diagnostics[diagnostics["condition"].isin(required) & diagnostics["epoch"].isin(args.epochs)]
    if args.counters:
        profiles = profiles[profiles["counter"].isin(args.counters)]
        diagnostics = diagnostics[diagnostics["counter"].isin(args.counters)]
    progress = np.linspace(0.0, 1.0, args.progress_points)
    store = profile_store(profiles, diagnostics, progress)
    counters = sorted(profiles["counter"].dropna().astype(str).unique())

    aggregate_rows: list[dict[str, object]] = []
    pointwise_rows: list[dict[str, object]] = []
    global_rows: list[dict[str, object]] = []
    for (dataset, model), conditions in usable_maps.items():
        for counter in counters:
            for role, condition in conditions.items():
                aggregate_rows.extend(
                    condition_aggregate_rows(
                        store, condition, role, dataset, model, counter,
                        list(args.epochs), progress,
                    )
                )
            for target_role in TARGET_LABELS:
                if target_role not in conditions:
                    continue
                normal_mu, normal_variance, target_mu, target_variance, used_epochs = collect_epoch_pairs(
                    store, conditions, counter, target_role, list(args.epochs)
                )
                if len(used_epochs) < args.min_paired_epochs:
                    message = f"Skipping {(dataset, model, counter, target_role)}: only {len(used_epochs)} paired epochs"
                    warnings.warn(message)
                    warnings_list.append(message)
                    continue
                result = paired_distribution_test(
                    normal_mu, normal_variance, target_mu, target_variance, progress
                )
                normal_scale = max(float(np.median(np.abs(result["reference_mean"]))), np.finfo(float).eps)
                global_rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "counter": counter,
                        "target_role": target_role,
                        "comparison": f"{TARGET_LABELS[target_role]} vs clean+moderate",
                        "paired_epochs": len(used_epochs),
                        "epochs": json.dumps(used_epochs),
                        "global_wasserstein_squared": result["global_wasserstein_squared"],
                        "global_wasserstein": math.sqrt(float(result["global_wasserstein_squared"])),
                        "normalized_global_w2": math.sqrt(float(result["global_wasserstein_squared"])) / normal_scale,
                        "global_permutation_p": result["global_p"],
                    }
                )
                pointwise_q = benjamini_hochberg(np.asarray(result["pointwise_p"]))
                for index, point in enumerate(progress):
                    reference_mean = float(result["reference_mean"][index])
                    target_mean = float(result["target_mean"][index])
                    denominator = abs(reference_mean) + abs(target_mean)
                    pointwise_rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "counter": counter,
                            "target_role": target_role,
                            "comparison": f"{TARGET_LABELS[target_role]} vs clean+moderate",
                            "progress": float(point),
                            "reference_mean_rate": reference_mean,
                            "target_mean_rate": target_mean,
                            "mean_rate_difference": target_mean - reference_mean,
                            "symmetric_percent_difference": 200.0 * (target_mean - reference_mean) / denominator if denominator else 0.0,
                            "reference_rate_variance": float(result["reference_variance"][index]),
                            "target_rate_variance": float(result["target_variance"][index]),
                            "wasserstein_squared": float(result["wasserstein_squared"][index]),
                            "wasserstein_distance": math.sqrt(float(result["wasserstein_squared"][index])),
                            "pointwise_permutation_p": float(result["pointwise_p"][index]),
                            "pointwise_fdr_q": float(pointwise_q[index]),
                            "pointwise_fwer_p": float(result["pointwise_fwer_p"][index]),
                            "paired_epochs": len(used_epochs),
                        }
                    )

            normal_rows = []
            for epoch in args.epochs:
                clean_key = (conditions["clean"], counter, epoch)
                moderate_key = (conditions["moderate"], counter, epoch)
                if clean_key in store and moderate_key in store:
                    mean, variance = moment_match_pair(*store[clean_key], *store[moderate_key])
                    normal_rows.append((mean, variance))
            if normal_rows:
                mean, variance = aggregate_gaussians(
                    np.stack([item[0] for item in normal_rows]),
                    np.stack([item[1] for item in normal_rows]),
                )
                aggregate_rows.extend(
                    {
                        "dataset": dataset,
                        "model": model,
                        "role": "normal_reference",
                        "condition": "clean+moderate",
                        "counter": counter,
                        "progress": float(point),
                        "mean_rate": float(mean[index]),
                        "rate_variance": float(variance[index]),
                        "rate_sd": float(np.sqrt(variance[index])),
                        "epochs": len(normal_rows),
                    }
                    for index, point in enumerate(progress)
                )

    global_frame = pd.DataFrame(global_rows)
    pointwise_frame = pd.DataFrame(pointwise_rows)
    aggregate_frame = pd.DataFrame(aggregate_rows)
    if global_frame.empty:
        raise RuntimeError("No comparisons had enough paired epoch profiles")
    adjusted_parts = []
    for _, group in global_frame.groupby(["dataset", "model"], sort=False):
        group = group.copy()
        group["global_fdr_q"] = benjamini_hochberg(group["global_permutation_p"].to_numpy())
        group["global_fdr_significant"] = group["global_fdr_q"].le(args.alpha)
        adjusted_parts.append(group)
    global_frame = pd.concat(adjusted_parts, ignore_index=True)

    aggregate_frame.to_csv(output_dir / "aggregated_distributions.csv", index=False)
    pointwise_frame.to_csv(output_dir / "pointwise_distribution_tests.csv", index=False)
    global_frame.to_csv(output_dir / "global_distribution_tests.csv", index=False)
    global_frame[global_frame["global_fdr_significant"]].to_csv(
        output_dir / "significant_global_distribution_tests.csv", index=False
    )
    for (dataset, model), group in global_frame.groupby(["dataset", "model"], sort=False):
        destination = output_dir / dataset / model
        for counter in sorted(group["counter"].unique()):
            plot_metric(
                pointwise_frame, aggregate_frame,
                dataset=dataset, model=model, counter=counter,
                output_dir=destination / "metrics", alpha=args.alpha, dpi=args.dpi,
            )
        plot_global_summary(group, destination, args.alpha, args.dpi)

    summary = {
        "input_dir": str(input_dir),
        "profile_dir": str(args.profile_dir.resolve()),
        "output_dir": str(output_dir),
        "datasets": list(args.datasets),
        "models": list(args.models),
        "epochs": list(args.epochs),
        "progress_points": args.progress_points,
        "counter_normalization": "Per batch: interval counter increments divided by that batch's total retired instructions before progress-profile fitting",
        "normal_reference": "Equal-weight Gaussian moment match of clean and moderate augmentation at each epoch and progress point",
        "variance_conversion": "rate_variance(p; h) = q(p) / h + tau_squared / h^2, using each fitted epoch's representative interval width h",
        "global_statistic": "Integral over instruction progress of Gaussian 2-Wasserstein distance squared",
        "test": "Exact epoch-paired label-swap permutation test between the normal reference and each target distribution",
        "pointwise_correction": "Both BH-FDR and max-permutation FWER p-values are saved",
        "global_multiple_testing": "BH-FDR within each dataset/model across counters and target conditions",
        "run_aggregation": "For each condition/counter/epoch, independent run profiles are combined by Gaussian moment matching before epoch-paired comparison",
        "runs_per_condition": runs_per_condition,
        "inference_scope": "Epoch-paired comparison of profiles estimated from all available runs; epochs remain the permutation-test pairs",
        "warnings": warnings_list,
        "successful_comparisons": int(len(global_frame)),
        "globally_significant": int(global_frame["global_fdr_significant"].sum()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        global_frame[
            ["dataset", "model", "counter", "comparison", "paired_epochs", "normalized_global_w2", "global_permutation_p", "global_fdr_q", "global_fdr_significant"]
        ].to_string(index=False)
    )
    print(f"Saved distribution comparison to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
