#!/usr/bin/env python3
"""Compare backward progress-aligned execution distributions at each epoch.

The independent statistical unit is a training trial.  Epoch is a repeated
measurement within a trial; batches and perf intervals are not treated as
independent replicates.  Cached inverse-problem profiles must use the
raw or per-instruction mode requested on the command line. Backward profiles
use retired-instruction progress computed independently within every backward
batch phase.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-backward-epochwise-distributions")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy import stats


SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_PHASE = "backward"
DEFAULT_COLLECTION = SCRIPT_DIR / "cache_0727_jetson_cpu_20_trials"
DEFAULT_INPUT = DEFAULT_COLLECTION / "192.168.0.141"
DEFAULT_PROFILE_DIR = DEFAULT_COLLECTION / "distribution_profile_cache_backward_per_instruction"
DEFAULT_RAW_PROFILE_DIR = DEFAULT_COLLECTION / "distribution_profile_cache_backward_raw"
DEFAULT_OUTPUT = DEFAULT_COLLECTION / "backward_epochwise_clean_baseline_comparison_per_instruction"
DEFAULT_RAW_OUTPUT = DEFAULT_COLLECTION / "backward_epochwise_clean_baseline_comparison_raw"

TARGET_LABELS = {
    "moderate_augmentation": "moderate augmentation",
    "availability_shortcut": "shortcut",
    "non_iid": "non-IID",
    "strong_augmentation": "strong augmentation",
}
COLORS = {
    "normal_reference": "#202326",
    "moderate_augmentation": "#3a923a",
    "availability_shortcut": "#c84c3a",
    "non_iid": "#8b5ea7",
    "strong_augmentation": "#d18b24",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--profile-dir",
        type=Path,
        help=(
            "Profile cache. When omitted, select a compatible raw or per-instruction "
            "cache automatically."
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--counter-normalization",
        choices=("raw", "per_instruction"),
        default="per_instruction",
        help=(
            "raw fits counter increments directly; per_instruction divides each interval "
            "counter by its batch's total retired instructions before fitting."
        ),
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=("cifar10", "trashnet"),
        default=("cifar10", "trashnet"),
    )
    parser.add_argument(
        "--models", nargs="+", choices=("cnn", "vit"), default=("cnn", "vit")
    )
    parser.add_argument(
        "--targets", nargs="+", choices=tuple(TARGET_LABELS),
        default=tuple(TARGET_LABELS),
    )
    parser.add_argument("--epochs", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--counters", nargs="+")
    parser.add_argument("--progress-points", type=int, default=101)
    parser.add_argument("--expected-trials", type=int, default=20)
    parser.add_argument("--min-paired-trials", type=int, default=8)
    parser.add_argument("--permutations", type=int, default=4999)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--random-seed", type=int, default=260728)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf"), default=("pdf",))
    parser.add_argument("--profile-cv-folds", type=int, default=0)
    parser.add_argument(
        "--auto-estimate", action="store_true",
        help="Estimate all requested profiles when the selected cache is incomplete.",
    )
    parser.add_argument("--rebuild-profiles", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.epochs or any(epoch < 0 for epoch in args.epochs):
        parser.error("--epochs must contain non-negative values")
    if args.progress_points < 11:
        parser.error("--progress-points must be at least 11")
    if args.expected_trials < 1 or args.min_paired_trials < 2:
        parser.error("trial counts must be positive and min-paired-trials must be at least 2")
    if args.permutations < 99:
        parser.error("--permutations must be at least 99")
    if not 0.0 < args.alpha < 1.0:
        parser.error("--alpha must be in (0, 1)")
    return args


def condition_name(dataset: str, model: str, role: str) -> str:
    prefix = dataset if model == "cnn" else f"{dataset}_vit"
    if role == "clean":
        return prefix if model == "vit" else f"{dataset}_iid"
    return f"{prefix}_{role}"


def condition_maps(
    datasets: list[str], models: list[str], targets: list[str], available: set[str]
) -> dict[tuple[str, str], dict[str, str]]:
    output: dict[tuple[str, str], dict[str, str]] = {}
    for dataset in datasets:
        for model in models:
            roles = ("clean", *targets)
            conditions = {role: condition_name(dataset, model, role) for role in roles}
            if conditions["clean"] not in available:
                warnings.warn(f"Skipping {(dataset, model)}: clean baseline is unavailable")
                continue
            output[(dataset, model)] = {
                role: condition for role, condition in conditions.items() if condition in available
            }
            for role in targets:
                if conditions[role] not in available:
                    warnings.warn(f"{(dataset, model)}: {role} is unavailable and will be skipped")
    return output


def available_conditions(input_dir: Path) -> set[str]:
    return {
        path.name for path in input_dir.iterdir()
        if path.is_dir() and any(path.glob("*_perf.csv"))
    } if input_dir.is_dir() else set()


def resolve_input_dir(path: Path) -> Path:
    path = path.resolve()
    if available_conditions(path):
        return path
    candidates = [
        child for child in path.iterdir()
        if child.is_dir() and available_conditions(child)
    ] if path.is_dir() else []
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No condition directories containing *_perf.csv below {path}")
    raise ValueError(f"Multiple device directories below {path}; pass one explicitly")


def estimate_profiles(args: argparse.Namespace, conditions: set[str]) -> None:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "estimate_execution_profiles.py"),
        "--input-dir", str(args.input_dir),
        "--output-dir", str(args.profile_dir),
        "--phase", ANALYSIS_PHASE,
        "--conditions", *sorted(conditions),
        "--epochs", *(str(epoch) for epoch in args.epochs),
        "--counter-normalization", args.counter_normalization,
        "--cv-folds", str(args.profile_cv_folds),
        "--no-plots", "--overwrite",
    ]
    if args.counters:
        command.extend(("--counters", *args.counters))
    print(
        f"Estimating requested {ANALYSIS_PHASE} "
        f"{args.counter_normalization} profiles...",
        flush=True,
    )
    subprocess.run(command, check=True)


def profile_cache_paths(args: argparse.Namespace, required: set[str]) -> tuple[Path, Path]:
    profiles_path = args.profile_dir / "profiles.csv"
    diagnostics_path = args.profile_dir / "diagnostics.csv"
    existing: set[str] = set()
    compatible = False
    if profiles_path.exists() and diagnostics_path.exists():
        compatible = (
            cache_normalization(profiles_path) == args.counter_normalization
            and cache_phase(args.profile_dir) == ANALYSIS_PHASE
        )
        if compatible:
            cache_info = pd.read_csv(profiles_path, usecols=("condition",))
            existing = set(cache_info["condition"].dropna().astype(str))
    missing = required - existing
    if args.rebuild_profiles or missing or not compatible:
        if not (args.auto_estimate or args.rebuild_profiles):
            raise FileNotFoundError(
                f"The {ANALYSIS_PHASE} {args.counter_normalization} profile cache is "
                "missing, incompatible, or incomplete. "
                "Missing conditions: "
                f"{sorted(missing)}. Finish the current profile-estimation process, or rerun "
                "this command with --auto-estimate. Raw and per-instruction fitted profiles "
                "are not interchangeable because normalization is batch-specific."
            )
        estimate_profiles(args, required)
    return profiles_path, diagnostics_path


def cache_normalization(profiles_path: Path) -> str | None:
    if not profiles_path.is_file():
        return None
    columns = pd.read_csv(profiles_path, nrows=0).columns
    if "counter_normalization" not in columns:
        # The original cache predates the explicit column and was fitted from
        # unnormalized counter increments.
        return "raw"
    values = pd.read_csv(profiles_path, usecols=("counter_normalization",))[
        "counter_normalization"
    ].dropna().astype(str).unique()
    return str(values[0]) if len(values) == 1 else None


def cache_phase(profile_dir: Path) -> str | None:
    summary_path = profile_dir / "run_summary.json"
    if summary_path.is_file():
        try:
            value = json.loads(summary_path.read_text()).get("phase")
            if value:
                return str(value).lower()
        except (OSError, ValueError, TypeError):
            pass
    diagnostics_path = profile_dir / "diagnostics.csv"
    if diagnostics_path.is_file():
        columns = pd.read_csv(diagnostics_path, nrows=0).columns
        if "phase" in columns:
            values = pd.read_csv(diagnostics_path, usecols=("phase",))["phase"]
            values = values.dropna().astype(str).str.lower().unique()
            if len(values) == 1:
                return str(values[0])
    return None


def cached_conditions(
    profile_dir: Path, normalization: str, phase: str = ANALYSIS_PHASE
) -> set[str]:
    profiles_path = profile_dir / "profiles.csv"
    diagnostics_path = profile_dir / "diagnostics.csv"
    if not profiles_path.is_file() or not diagnostics_path.is_file():
        return set()
    if (
        cache_normalization(profiles_path) != normalization
        or cache_phase(profile_dir) != phase
    ):
        return set()
    information = pd.read_csv(profiles_path, usecols=("condition",))
    return set(information["condition"].dropna().astype(str))


def resolve_profile_dir(
    requested: Path | None,
    input_dir: Path,
    required: set[str],
    normalization: str,
) -> Path:
    if requested is not None:
        return requested.resolve()
    collection_dir = input_dir.parent
    if normalization == "raw":
        candidates = sorted(collection_dir.glob("distribution_profile_cache_backward_raw*"))
    else:
        candidates = sorted(
            collection_dir.glob("distribution_profile_cache_backward_per_instruction*")
        )
    compatible = [
        path for path in candidates
        if required <= cached_conditions(path, normalization)
    ]
    if compatible:
        # Prefer the cache with the least unrelated condition data, then the
        # shortest name. This selects a specialized cache when one exists.
        selected = min(
            compatible,
            key=lambda path: (
                len(cached_conditions(path, normalization) - required), len(path.name)
            ),
        )
        print(f"Auto-selected compatible profile cache: {selected}", flush=True)
        return selected.resolve()
    fallback = (
        DEFAULT_RAW_PROFILE_DIR.name
        if normalization == "raw"
        else DEFAULT_PROFILE_DIR.name
    )
    return (collection_dir / fallback).resolve()


def successful_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    success = frame["optimizer_success"]
    if success.dtype != bool:
        success = success.astype(str).str.lower().isin(("true", "1", "yes"))
    finite = frame["estimated_mean_rate"].notna() & frame["estimated_variance_rate"].notna()
    return frame.loc[success & finite].copy()


def representative_widths(
    diagnostics: pd.DataFrame,
) -> dict[tuple[str, str, str, int], float]:
    status = diagnostics["status"].astype(str).eq("ok")
    valid = diagnostics.loc[status].copy()
    valid["representative_interval_width"] = pd.to_numeric(
        valid["representative_interval_width"], errors="coerce"
    )
    return {
        (str(row.condition), str(row.run_id), str(row.counter), int(row.epoch)):
            float(row.representative_interval_width)
        for row in valid.itertuples()
        if np.isfinite(row.representative_interval_width)
        and row.representative_interval_width > 0
    }


def resample_profile(
    group: pd.DataFrame, progress: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    ordered = group.sort_values("bin_index")
    starts = ordered["progress_start"].to_numpy(dtype=float)
    ends = ordered["progress_end"].to_numpy(dtype=float)
    if len(starts) == 0 or starts[0] > 1e-8 or ends[-1] < 1.0 - 1e-8:
        raise ValueError("profile bins do not cover [0, 1]")
    index = np.minimum(np.searchsorted(ends, progress, side="right"), len(ends) - 1)
    if np.any(progress < starts[index] - 1e-8) or np.any(progress > ends[index] + 1e-8):
        raise ValueError("profile bins contain a progress gap")
    mu = ordered["estimated_mean_rate"].to_numpy(dtype=float)[index]
    q = ordered["estimated_variance_rate"].to_numpy(dtype=float)[index]
    tau2 = max(float(ordered["tau_squared"].iloc[0]), 0.0)
    return mu, np.maximum(q, 0.0), tau2


def aggregate_gaussians(
    means: np.ndarray, variances: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = means.mean(axis=0)
    variance = (variances + means * means).mean(axis=0) - mean * mean
    return mean, np.maximum(variance, 0.0)


def trial_profile_store(
    profiles: pd.DataFrame, diagnostics: pd.DataFrame, progress: np.ndarray
) -> dict[tuple[str, str, str, int], tuple[np.ndarray, np.ndarray]]:
    widths = representative_widths(diagnostics)
    by_trial: dict[
        tuple[str, str, str, int], list[tuple[np.ndarray, np.ndarray]]
    ] = {}
    grouped = successful_profiles(profiles).groupby(
        ["condition", "run_id", "trial_id", "counter", "epoch"], sort=False
    )
    for key, group in grouped:
        condition, run_id, trial_id, counter, epoch = (
            str(key[0]), str(key[1]), str(key[2]), str(key[3]), int(key[4])
        )
        width = widths.get((condition, run_id, counter, epoch))
        if width is None:
            continue
        try:
            mu, q, tau2 = resample_profile(group, progress)
        except ValueError:
            continue
        rate_variance = np.maximum(q / width + tau2 / (width * width), 0.0)
        by_trial.setdefault((condition, trial_id, counter, epoch), []).append(
            (mu, rate_variance)
        )

    output: dict[tuple[str, str, str, int], tuple[np.ndarray, np.ndarray]] = {}
    for key, values in by_trial.items():
        if len(values) == 1:
            output[key] = values[0]
        else:
            output[key] = aggregate_gaussians(
                np.stack([value[0] for value in values]),
                np.stack([value[1] for value in values]),
            )
            warnings.warn(f"Moment-matched {len(values)} duplicate runs for {key}")
    return output


def paired_epoch_profiles(
    store: dict[tuple[str, str, str, int], tuple[np.ndarray, np.ndarray]],
    conditions: dict[str, str],
    target_role: str,
    counter: str,
    epoch: int,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    clean = conditions["clean"]
    target = conditions[target_role]
    trials = sorted({
        key[1] for key in store
        if key[0] == clean and key[2] == counter and key[3] == epoch
        and (target, key[1], counter, epoch) in store
    })
    normal_mu, normal_variance, target_mu, target_variance = [], [], [], []
    for trial in trials:
        normal = store[(clean, trial, counter, epoch)]
        selected = store[(target, trial, counter, epoch)]
        normal_mu.append(normal[0])
        normal_variance.append(normal[1])
        target_mu.append(selected[0])
        target_variance.append(selected[1])
    return (
        trials,
        np.asarray(normal_mu), np.asarray(normal_variance),
        np.asarray(target_mu), np.asarray(target_variance),
    )


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


def paired_distribution_permutation(
    normal_mu: np.ndarray,
    normal_variance: np.ndarray,
    target_mu: np.ndarray,
    target_variance: np.ndarray,
    progress: np.ndarray,
    permutations: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    n = len(normal_mu)
    reference_mean, reference_variance = aggregate_gaussians(normal_mu, normal_variance)
    target_mean, target_total_variance = aggregate_gaussians(target_mu, target_variance)
    observed = wasserstein_squared(
        reference_mean, reference_variance, target_mean, target_total_variance
    )
    observed_global = float(np.trapezoid(observed, progress))

    normal_second = normal_variance + normal_mu * normal_mu
    target_second = target_variance + target_mu * target_mu
    swaps = rng.integers(0, 2, size=(permutations, n), dtype=np.int8).astype(float)
    left_mean = normal_mu.mean(axis=0) + swaps @ (target_mu - normal_mu) / n
    right_mean = target_mu.mean(axis=0) - swaps @ (target_mu - normal_mu) / n
    left_second = normal_second.mean(axis=0) + swaps @ (target_second - normal_second) / n
    right_second = target_second.mean(axis=0) - swaps @ (target_second - normal_second) / n
    left_variance = np.maximum(left_second - left_mean * left_mean, 0.0)
    right_variance = np.maximum(right_second - right_mean * right_mean, 0.0)
    null_distance = wasserstein_squared(
        left_mean, left_variance, right_mean, right_variance
    )
    null_global = np.trapezoid(null_distance, progress, axis=1)
    tolerance = max(abs(observed_global), 1.0) * 1e-12
    global_p = (1.0 + np.count_nonzero(null_global >= observed_global - tolerance)) / (
        permutations + 1.0
    )
    pointwise_p = (
        1.0 + np.count_nonzero(null_distance >= observed[None, :] - tolerance, axis=0)
    ) / (permutations + 1.0)
    scale = np.maximum(np.median(null_distance, axis=0), np.finfo(float).eps)
    maximum = np.max(null_distance / scale[None, :], axis=1)
    pointwise_fwer = (
        1.0 + np.count_nonzero(
            maximum[:, None] >= observed[None, :] / scale[None, :] - 1e-12,
            axis=0,
        )
    ) / (permutations + 1.0)
    return {
        "reference_mean": reference_mean,
        "reference_variance": reference_variance,
        "target_mean": target_mean,
        "target_variance": target_total_variance,
        "wasserstein_squared": observed,
        "global_wasserstein_squared": observed_global,
        "global_p": float(global_p),
        "pointwise_p": pointwise_p,
        "pointwise_fwer_p": pointwise_fwer,
    }


def benjamini_hochberg(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape, np.nan)
    valid = np.isfinite(values)
    if not valid.any():
        return result
    selected = values[valid]
    order = np.argsort(selected)
    adjusted = selected[order] * len(selected) / np.arange(1, len(selected) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    result[valid] = restored
    return result


def trial_epoch_rows(
    dataset: str,
    model: str,
    counter: str,
    target_role: str,
    epoch: int,
    trials: list[str],
    normal_mu: np.ndarray,
    normal_variance: np.ndarray,
    target_mu: np.ndarray,
    target_variance: np.ndarray,
    progress: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for index, trial in enumerate(trials):
        distance_squared = wasserstein_squared(
            normal_mu[index], normal_variance[index],
            target_mu[index], target_variance[index],
        )
        integrated_squared = float(np.trapezoid(distance_squared, progress))
        rows.append({
            "dataset": dataset,
            "model": model,
            "counter": counter,
            "target_role": target_role,
            "trial_id": trial,
            "epoch": epoch,
            "integrated_w2_squared": integrated_squared,
            "integrated_w2": math.sqrt(max(integrated_squared, 0.0)),
            "integrated_mean_difference": float(
                np.trapezoid(target_mu[index] - normal_mu[index], progress)
            ),
        })
    return rows


def sign_flip_pvalue(
    values: np.ndarray, permutations: int, rng: np.random.Generator
) -> float:
    observed = abs(float(values.mean()))
    signs = rng.choice((-1.0, 1.0), size=(permutations, len(values)))
    null = np.abs(signs @ values / len(values))
    return float((1 + np.count_nonzero(null >= observed - 1e-15)) / (permutations + 1))


def repeated_epoch_permutation_pvalue(
    group: pd.DataFrame,
    permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Permute epoch labels within trials, retaining missing-epoch patterns."""
    epoch_values = sorted(group["epoch"].astype(int).unique())
    epoch_index = {epoch: index for index, epoch in enumerate(epoch_values)}
    observed_sums = np.zeros(len(epoch_values), dtype=float)
    null_sums = np.zeros((permutations, len(epoch_values)), dtype=float)
    counts = np.zeros(len(epoch_values), dtype=float)

    for _, trial in group.groupby("trial_id", sort=False):
        trial = trial.sort_values("epoch")
        positions = np.asarray([epoch_index[int(epoch)] for epoch in trial["epoch"]])
        values = trial["integrated_w2"].to_numpy(dtype=float)
        centered = values - values.mean()
        observed_sums[positions] += centered
        counts[positions] += 1.0

        # Every row is an independent within-trial permutation. Assigning the
        # shuffled values back to the fixed observed epoch positions preserves
        # each trial's missing-epoch pattern.
        order = np.argsort(rng.random((permutations, len(centered))), axis=1)
        null_sums[:, positions] += centered[order]

    valid = counts > 0
    observed = float(np.sum(observed_sums[valid] ** 2 / counts[valid]))
    null_statistics = np.sum(
        null_sums[:, valid] ** 2 / counts[None, valid], axis=1
    )
    p_value = (1 + np.count_nonzero(null_statistics >= observed - 1e-15)) / (
        permutations + 1
    )
    return observed, float(p_value)


def trend_test_rows(
    distances: pd.DataFrame,
    epochs: list[int],
    permutations: int,
    random_seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    grouped = distances.groupby(["dataset", "model", "counter", "target_role"], sort=False)
    for group_index, (key, group) in enumerate(grouped):
        selected_epochs = sorted(set(epochs) & set(group["epoch"].astype(int)))
        slope_values: list[float] = []
        retained_trials: list[str] = []
        for trial_id, trial in group.groupby("trial_id", sort=False):
            trial = trial[trial["epoch"].isin(selected_epochs)].sort_values("epoch")
            if trial["epoch"].nunique() < 3:
                continue
            x = trial["epoch"].to_numpy(dtype=float)
            y = trial["integrated_w2"].to_numpy(dtype=float)
            x -= x.mean()
            slope_values.append(float(y @ x / (x @ x)))
            retained_trials.append(str(trial_id))
        if len(slope_values) < 3 or len(selected_epochs) < 3:
            continue
        slopes = np.asarray(slope_values)
        rng = np.random.default_rng(random_seed + 7919 * group_index)
        slope_p = sign_flip_pvalue(slopes, permutations, rng)
        bootstrap = rng.choice(slopes, size=(permutations, len(slopes)), replace=True).mean(axis=1)
        omnibus_group = group[group["trial_id"].astype(str).isin(retained_trials)]
        omnibus_statistic, omnibus_p = repeated_epoch_permutation_pvalue(
            omnibus_group, permutations, rng
        )
        rows.append({
            "dataset": key[0],
            "model": key[1],
            "counter": key[2],
            "target_role": key[3],
            "slope_trials": len(slopes),
            "minimum_epochs_per_slope": 3,
            "epochs": json.dumps(selected_epochs),
            "mean_w2_slope_per_epoch": float(slopes.mean()),
            "median_w2_slope_per_epoch": float(np.median(slopes)),
            "slope_ci95_low": float(np.quantile(bootstrap, 0.025)),
            "slope_ci95_high": float(np.quantile(bootstrap, 0.975)),
            "slope_sign_flip_p": slope_p,
            "epoch_omnibus_statistic": omnibus_statistic,
            "epoch_omnibus_permutation_p": omnibus_p,
        })
    return rows


def epoch_distance_summary(distances: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["dataset", "model", "counter", "target_role", "epoch"]
    for key, group in distances.groupby(keys, sort=False):
        values = group["integrated_w2"].to_numpy(dtype=float)
        mean = float(values.mean())
        if len(values) > 1:
            sem = stats.sem(values)
            half_width = float(stats.t.ppf(0.975, len(values) - 1) * sem)
            standard_deviation = float(values.std(ddof=1))
        else:
            half_width = math.nan
            standard_deviation = math.nan
        rows.append({
            **dict(zip(keys, key)),
            "trials": len(values),
            "mean_integrated_w2": mean,
            "std_integrated_w2": standard_deviation,
            "ci95_low": mean - half_width,
            "ci95_high": mean + half_width,
        })
    return pd.DataFrame(rows)


def save_figure(figure: plt.Figure, prefix: Path, formats: list[str], dpi: int) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        kwargs = {"dpi": dpi} if extension == "png" else {}
        figure.savefig(
            prefix.with_suffix(f".{extension}"), bbox_inches="tight", facecolor="white", **kwargs
        )
    plt.close(figure)


def plot_epoch_profiles(
    profiles: pd.DataFrame,
    tests: pd.DataFrame,
    *,
    dataset: str,
    model: str,
    counter: str,
    target_role: str,
    epochs: list[int],
    output_dir: Path,
    formats: list[str],
    dpi: int,
    counter_normalization: str,
) -> None:
    columns = min(5, len(epochs))
    rows = math.ceil(len(epochs) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(3.7 * columns, 3.0 * rows), squeeze=False)
    for axis, epoch in zip(axes.flat, epochs):
        selected = profiles[
            profiles["dataset"].eq(dataset)
            & profiles["model"].eq(model)
            & profiles["counter"].eq(counter)
            & profiles["target_role"].eq(target_role)
            & profiles["epoch"].eq(epoch)
        ]
        for profile_role, color, label in (
            ("normal_reference", COLORS["normal_reference"], "clean IID"),
            ("target", COLORS[target_role], TARGET_LABELS[target_role]),
        ):
            curve = selected[selected["profile_role"].eq(profile_role)]
            if curve.empty:
                continue
            x = curve["progress"].to_numpy(dtype=float)
            mean = curve["mean_rate"].to_numpy(dtype=float)
            sd = curve["rate_sd"].to_numpy(dtype=float)
            axis.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.12)
            axis.plot(x, mean, color=color, linewidth=1.6, label=label)
        test = tests[
            tests["dataset"].eq(dataset)
            & tests["model"].eq(model)
            & tests["counter"].eq(counter)
            & tests["target_role"].eq(target_role)
            & tests["epoch"].eq(epoch)
        ]
        suffix = f"; q={test['global_fdr_q'].iloc[0]:.3g}" if not test.empty else ""
        axis.set_title(f"epoch {epoch}{suffix}", fontsize=9)
        axis.set_xlim(0, 1)
        axis.grid(True, color="#d9dde1", linewidth=0.5)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
    for axis in axes.flat[len(epochs):]:
        axis.set_visible(False)
    axes.flat[0].legend(fontsize=7, frameon=False)
    figure.supxlabel("Relative retired-instruction progress")
    figure.supylabel(
        "Counter / instruction-progress"
        if counter_normalization == "raw"
        else "Counter / instruction"
    )
    figure.suptitle(
        f"{dataset} | {model.upper()} | backward | {counter} | "
        f"{TARGET_LABELS[target_role]}\n"
        "Bands show moment-matched rate SD, not uncertainty of the trial mean",
        fontsize=11,
    )
    figure.tight_layout(rect=(0.02, 0.02, 1.0, 0.93))
    save_figure(figure, output_dir / f"{counter}_epoch_profiles", formats, dpi)


def plot_difference_heatmap(
    points: pd.DataFrame,
    tests: pd.DataFrame,
    *,
    dataset: str,
    model: str,
    counter: str,
    target_role: str,
    output_dir: Path,
    formats: list[str],
    dpi: int,
    alpha: float,
    counter_normalization: str,
) -> None:
    selected = points[
        points["dataset"].eq(dataset)
        & points["model"].eq(model)
        & points["counter"].eq(counter)
        & points["target_role"].eq(target_role)
    ]
    difference = selected.pivot(index="epoch", columns="progress", values="mean_rate_difference")
    fwer = selected.pivot(index="epoch", columns="progress", values="pointwise_fwer_p")
    epoch_q = tests[
        tests["dataset"].eq(dataset)
        & tests["model"].eq(model)
        & tests["counter"].eq(counter)
        & tests["target_role"].eq(target_role)
    ].set_index("epoch")["global_fdr_q"]
    if difference.empty:
        return
    values = difference.to_numpy(dtype=float)
    limit = max(float(np.nanmax(np.abs(values))), np.finfo(float).eps)
    figure, axis = plt.subplots(figsize=(11.0, 5.0))
    image = axis.imshow(
        values,
        aspect="auto",
        origin="lower",
        extent=(0, 1, -0.5, len(difference.index) - 0.5),
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
    )
    significant = fwer.to_numpy(dtype=float) <= alpha
    for row, epoch in enumerate(difference.index):
        if epoch_q.get(epoch, 1.0) > alpha:
            significant[row] = False
    yy, xx = np.where(significant)
    if len(xx):
        progress_values = difference.columns.to_numpy(dtype=float)
        axis.scatter(progress_values[xx], yy, s=4, color="black", alpha=0.65)
    axis.set_yticks(np.arange(len(difference.index)), difference.index)
    axis.set_xlabel("Relative retired-instruction progress")
    axis.set_ylabel("Epoch")
    axis.set_title(
        f"{dataset} | {model.upper()} | backward | {counter} | "
        f"{TARGET_LABELS[target_role]} - normal\n"
        f"Dots require epoch BH q and pointwise max-permutation p <= {alpha:g}"
    )
    difference_label = (
        "Mean counter/instruction-progress difference"
        if counter_normalization == "raw"
        else "Mean counter/instruction difference"
    )
    figure.colorbar(image, ax=axis, label=difference_label)
    figure.tight_layout()
    save_figure(
        figure, output_dir / f"{counter}_epoch_progress_difference", formats, dpi
    )


def plot_distance_trend(
    distances: pd.DataFrame,
    summary: pd.DataFrame,
    trends: pd.DataFrame,
    *,
    dataset: str,
    model: str,
    counter: str,
    target_role: str,
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> None:
    selected = distances[
        distances["dataset"].eq(dataset)
        & distances["model"].eq(model)
        & distances["counter"].eq(counter)
        & distances["target_role"].eq(target_role)
    ]
    selected_summary = summary[
        summary["dataset"].eq(dataset)
        & summary["model"].eq(model)
        & summary["counter"].eq(counter)
        & summary["target_role"].eq(target_role)
    ].sort_values("epoch")
    selected_trend = trends[
        trends["dataset"].eq(dataset)
        & trends["model"].eq(model)
        & trends["counter"].eq(counter)
        & trends["target_role"].eq(target_role)
    ]
    figure, axis = plt.subplots(figsize=(8.8, 5.0))
    for _, trial in selected.groupby("trial_id", sort=False):
        trial = trial.sort_values("epoch")
        axis.plot(trial["epoch"], trial["integrated_w2"], color="#8a8d90", alpha=0.22, linewidth=0.7)
    axis.fill_between(
        selected_summary["epoch"].to_numpy(dtype=float),
        selected_summary["ci95_low"].to_numpy(dtype=float),
        selected_summary["ci95_high"].to_numpy(dtype=float),
        color=COLORS[target_role], alpha=0.2, label="95% CI of trial mean",
    )
    axis.plot(
        selected_summary["epoch"], selected_summary["mean_integrated_w2"],
        color=COLORS[target_role], marker="o", linewidth=2.0, label="trial mean",
    )
    annotation = ""
    if not selected_trend.empty:
        row = selected_trend.iloc[0]
        annotation = (
            f"slope={row['mean_w2_slope_per_epoch']:.3g}/epoch, "
            f"slope q={row['slope_fdr_q']:.3g}, omnibus q={row['omnibus_fdr_q']:.3g}"
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Integrated Gaussian W2 distance")
    axis.set_title(
        f"{dataset} | {model.upper()} | backward | {counter} | "
        f"{TARGET_LABELS[target_role]} vs normal\n{annotation}"
    )
    axis.grid(True, color="#d9dde1", linewidth=0.55)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    save_figure(
        figure, output_dir / f"{counter}_integrated_distance_trend", formats, dpi
    )


def main() -> int:
    args = parse_args()
    args.input_dir = resolve_input_dir(args.input_dir)
    if args.output_dir is None:
        args.output_dir = (
            DEFAULT_RAW_OUTPUT
            if args.counter_normalization == "raw"
            else DEFAULT_OUTPUT
        )
    output_dir = args.output_dir.resolve()
    marker = output_dir / "epoch_global_tests.csv"
    if marker.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists below {output_dir}; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_conditions = available_conditions(args.input_dir)
    maps = condition_maps(
        list(args.datasets), list(args.models), list(args.targets), raw_conditions
    )
    required = {condition for conditions in maps.values() for condition in conditions.values()}
    if not required:
        raise RuntimeError("No requested condition groups are available")
    args.profile_dir = resolve_profile_dir(
        args.profile_dir, args.input_dir, required, args.counter_normalization
    )
    profiles_path, diagnostics_path = profile_cache_paths(args, required)
    print(f"Loading cached profiles: {profiles_path}", flush=True)
    profiles = pd.read_csv(profiles_path)
    diagnostics = pd.read_csv(diagnostics_path)
    actual_normalization = cache_normalization(profiles_path)
    if actual_normalization != args.counter_normalization:
        raise ValueError(
            f"Requested {args.counter_normalization}, but cache contains "
            f"{actual_normalization or 'mixed/unknown'} profiles"
        )
    profiles = profiles[
        profiles["condition"].isin(required) & profiles["epoch"].isin(args.epochs)
    ]
    diagnostics = diagnostics[
        diagnostics["condition"].isin(required) & diagnostics["epoch"].isin(args.epochs)
    ]
    if args.counters:
        profiles = profiles[profiles["counter"].isin(args.counters)]
        diagnostics = diagnostics[diagnostics["counter"].isin(args.counters)]
    progress = np.linspace(0.0, 1.0, args.progress_points)
    store = trial_profile_store(profiles, diagnostics, progress)
    counters = sorted(profiles["counter"].dropna().astype(str).unique())

    profile_rows: list[dict[str, object]] = []
    pointwise_rows: list[dict[str, object]] = []
    epoch_test_rows: list[dict[str, object]] = []
    distance_rows: list[dict[str, object]] = []
    test_index = 0
    available_targets = sum(
        role in conditions
        for conditions in maps.values()
        for role in args.targets
    )
    total_epoch_tests = available_targets * len(counters) * len(args.epochs)
    skipped: list[str] = []
    for (dataset, model), conditions in maps.items():
        for target_role in args.targets:
            if target_role not in conditions:
                continue
            for counter in counters:
                for epoch in args.epochs:
                    print(
                        f"[epoch test {test_index + 1}/{total_epoch_tests}] "
                        f"{dataset}/{model}/{target_role}/{counter}/epoch {epoch}",
                        flush=True,
                    )
                    trials, normal_mu, normal_variance, target_mu, target_variance = paired_epoch_profiles(
                        store, conditions, target_role, counter, epoch
                    )
                    if len(trials) < args.min_paired_trials:
                        skipped.append(
                            f"{dataset}/{model}/{target_role}/{counter}/epoch {epoch}: "
                            f"{len(trials)} paired trials"
                        )
                        continue
                    if len(trials) != args.expected_trials:
                        warnings.warn(
                            f"{dataset}/{model}/{target_role}/{counter}/epoch {epoch}: "
                            f"using {len(trials)} of expected {args.expected_trials} paired trials"
                        )
                    rng = np.random.default_rng(args.random_seed + 1009 * test_index)
                    test_index += 1
                    result = paired_distribution_permutation(
                        normal_mu, normal_variance, target_mu, target_variance,
                        progress, args.permutations, rng,
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
                        "global_wasserstein": math.sqrt(
                            max(float(result["global_wasserstein_squared"]), 0.0)
                        ),
                        "normalized_global_w2": math.sqrt(
                            max(float(result["global_wasserstein_squared"]), 0.0)
                        ) / scale,
                        "global_permutation_p": result["global_p"],
                    })
                    distance_rows.extend(trial_epoch_rows(
                        dataset, model, counter, target_role, epoch, trials,
                        normal_mu, normal_variance, target_mu, target_variance, progress,
                    ))
                    for profile_role, mean, variance in (
                        ("normal_reference", result["reference_mean"], result["reference_variance"]),
                        ("target", result["target_mean"], result["target_variance"]),
                    ):
                        profile_rows.extend({
                            "dataset": dataset,
                            "model": model,
                            "counter": counter,
                            "target_role": target_role,
                            "epoch": epoch,
                            "profile_role": profile_role,
                            "progress": float(point),
                            "mean_rate": float(mean[index]),
                            "rate_variance": float(variance[index]),
                            "rate_sd": math.sqrt(max(float(variance[index]), 0.0)),
                            "paired_trials": len(trials),
                        } for index, point in enumerate(progress))
                    for index, point in enumerate(progress):
                        reference_mean = float(result["reference_mean"][index])
                        selected_mean = float(result["target_mean"][index])
                        pointwise_rows.append({
                            "dataset": dataset,
                            "model": model,
                            "counter": counter,
                            "target_role": target_role,
                            "epoch": epoch,
                            "progress": float(point),
                            "reference_mean_rate": reference_mean,
                            "target_mean_rate": selected_mean,
                            "mean_rate_difference": selected_mean - reference_mean,
                            "wasserstein_squared": float(result["wasserstein_squared"][index]),
                            "wasserstein_distance": math.sqrt(
                                max(float(result["wasserstein_squared"][index]), 0.0)
                            ),
                            "pointwise_permutation_p": float(result["pointwise_p"][index]),
                            "pointwise_fwer_p": float(result["pointwise_fwer_p"][index]),
                            "paired_trials": len(trials),
                        })

    epoch_tests = pd.DataFrame(epoch_test_rows)
    profiles_out = pd.DataFrame(profile_rows)
    pointwise = pd.DataFrame(pointwise_rows)
    distances = pd.DataFrame(distance_rows)
    if epoch_tests.empty:
        raise RuntimeError("No epoch had enough paired trial profiles for comparison")

    adjusted = []
    for _, group in epoch_tests.groupby(["dataset", "model"], sort=False):
        group = group.copy()
        group["global_fdr_q"] = benjamini_hochberg(
            group["global_permutation_p"].to_numpy(dtype=float)
        )
        group["global_fdr_significant"] = group["global_fdr_q"].le(args.alpha)
        adjusted.append(group)
    epoch_tests = pd.concat(adjusted, ignore_index=True)

    print("Running trial-level longitudinal trend tests...", flush=True)
    trends = pd.DataFrame(trend_test_rows(
        distances, list(args.epochs), args.permutations, args.random_seed + 700_001
    ))
    if not trends.empty:
        adjusted_trends = []
        for _, group in trends.groupby(["dataset", "model"], sort=False):
            group = group.copy()
            group["slope_fdr_q"] = benjamini_hochberg(
                group["slope_sign_flip_p"].to_numpy(dtype=float)
            )
            group["omnibus_fdr_q"] = benjamini_hochberg(
                group["epoch_omnibus_permutation_p"].to_numpy(dtype=float)
            )
            adjusted_trends.append(group)
        trends = pd.concat(adjusted_trends, ignore_index=True)
    distance_summary = epoch_distance_summary(distances)

    profiles_out.to_csv(output_dir / "epoch_profiles.csv", index=False)
    epoch_tests.to_csv(output_dir / "epoch_global_tests.csv", index=False)
    pointwise.to_csv(output_dir / "epoch_pointwise_tests.csv", index=False)
    distances.to_csv(output_dir / "trial_epoch_distances.csv", index=False)
    distance_summary.to_csv(output_dir / "epoch_distance_summary.csv", index=False)
    trends.to_csv(output_dir / "trend_tests.csv", index=False)

    print(f"Rendering {len(trends)} metric/comparison figure sets...", flush=True)
    for plot_index, row in enumerate(trends.itertuples(), start=1):
        print(
            f"[plot {plot_index}/{len(trends)}] "
            f"{row.dataset}/{row.model}/{row.target_role}/{row.counter}",
            flush=True,
        )
        destination = output_dir / row.dataset / row.model / row.target_role
        plot_epoch_profiles(
            profiles_out, epoch_tests,
            dataset=row.dataset, model=row.model, counter=row.counter,
            target_role=row.target_role, epochs=list(args.epochs), output_dir=destination,
            formats=list(args.formats), dpi=args.dpi,
            counter_normalization=args.counter_normalization,
        )
        plot_difference_heatmap(
            pointwise, epoch_tests,
            dataset=row.dataset, model=row.model, counter=row.counter,
            target_role=row.target_role, output_dir=destination,
            formats=list(args.formats), dpi=args.dpi, alpha=args.alpha,
            counter_normalization=args.counter_normalization,
        )
        plot_distance_trend(
            distances, distance_summary, trends,
            dataset=row.dataset, model=row.model, counter=row.counter,
            target_role=row.target_role, output_dir=destination,
            formats=list(args.formats), dpi=args.dpi,
        )

    summary = {
        "input_dir": str(args.input_dir),
        "profile_dir": str(args.profile_dir),
        "output_dir": str(output_dir),
        "phase": ANALYSIS_PHASE,
        "counter_normalization": args.counter_normalization,
        "normalization_definition": (
            "Counter increments were fitted directly on relative instruction progress."
            if args.counter_normalization == "raw"
            else "Each interval counter was divided by its own batch's total retired "
            "instructions before inverse-profile fitting."
        ),
        "normal_reference": (
            "Within every trial and epoch, the clean IID Gaussian profile is the sole baseline."
        ),
        "epoch_test": (
            "Trial-paired Monte Carlo label-swap test of integrated Gaussian W2 squared. "
            "BH correction is applied across target/counter/epoch tests within dataset/model."
        ),
        "trend_test": (
            "A slope is fitted to each trial's integrated W2 across epoch. The mean slope uses "
            "a trial-level sign-flip test. A within-trial epoch-label permutation is the "
            "nonlinear epoch omnibus test and supports missing fitted epochs."
        ),
        "pointwise_test": "Max-permutation FWER correction across instruction progress.",
        "permutations": args.permutations,
        "progress_points": args.progress_points,
        "expected_trials": args.expected_trials,
        "completed_epoch_tests": len(epoch_tests),
        "completed_trend_tests": len(trends),
        "skipped_count": len(skipped),
        "skipped": skipped,
        "limitation": (
            "A fitted K=1 profile represents only a whole-backward scalar; "
            "it does not identify where within backward execution a change occurred."
        ),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"Wrote {len(epoch_tests)} epoch tests and {len(trends)} trend tests to {output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
