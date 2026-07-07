#!/usr/bin/env python3
"""Unsupervised condition separation from Fourier OT distances.

This script reuses ``4_fourier_preprocessed_ot_embedding.py`` as the Fourier-OT
module. The default path does not use a fixed clean reference. For each device
type, it:

1. Loads runs from all physical devices in that device type.
2. Splits telemetry by epoch/round.
3. Applies Fourier preprocessing to each run segment.
4. Computes pairwise OT distances between every run and every other run.
5. Converts each run's pairwise distance profile into a run-level feature vector.
6. Fits local PCA and unsupervised two-cluster k-means within that device type.

There is intentionally no fixed hand-written threshold. The output is primarily
for distribution inspection. The known clean/poisoning labels from the CSVs are
used only for post-hoc reporting and plots, not for fitting PCA or k-means.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
FOURIER_MODULE_PATH = SCRIPT_DIR / "4_fourier_preprocessed_ot_embedding.py"


def _load_fourier_module() -> Any:
    spec = importlib.util.spec_from_file_location("fourier_ot_base", FOURIER_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Fourier OT module: {FOURIER_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fourier_base = _load_fourier_module()
ot_base = fourier_base.ot_base


DEFAULT_FEATURE_COLUMNS = fourier_base.DEFAULT_FEATURE_COLUMNS
DEFAULT_COST_TYPES = fourier_base.DEFAULT_COST_TYPES
DEVICE_TYPE_BY_HOST = {
    "192.168.0.115": "rpi3",
    "192.168.0.116": "rpi4",
    "192.168.0.117": "rpi4",
    "192.168.0.118": "rpi4",
    "192.168.0.119": "rpi4",
    "192.168.0.120": "rpi4",
    "192.168.0.121": "rpi4",
    "192.168.0.131": "laptop",
    "192.168.0.141": "jetson_cpu",
    "192.168.0.142": "jetson_cpu",
}


def safe_name(value: str) -> str:
    return str(value).replace("/", "_").replace(" ", "_").replace(":", "_")


def host_from_group_label(group_label: str) -> str:
    return str(group_label).split("/", 1)[0]


def device_type_for_group_label(group_label: str) -> str:
    host = host_from_group_label(group_label)
    return DEVICE_TYPE_BY_HOST.get(host, f"unknown_{host}")


def numeric_segment_sort_key(item: Tuple[str, str]) -> Tuple[str, int, str]:
    segment_type, segment_id = item
    try:
        return (segment_type, int(segment_id), "")
    except ValueError:
        return (segment_type, 10**9, segment_id)


def compute_embedding_clean_distances(
    rows: List[Dict[str, Any]],
    embeddings: Dict[Tuple[str, str, str], np.ndarray],
) -> None:
    """Add distances from each embedding to the clean embedding centroid.

    This is still unsupervised at classifier time. The clean centroid is a
    diagnostic feature derived from the experimental reference design; labels
    are not used to set a decision threshold.
    """
    scopes = sorted({(row["reference_run_id"], row["cost_type"]) for row in rows})
    for reference_run_id, cost_type in scopes:
        scoped_rows = [
            row
            for row in rows
            if row["reference_run_id"] == reference_run_id and row["cost_type"] == cost_type
        ]
        clean_embeddings = []
        for row in scoped_rows:
            if row["target_group"] == "clean":
                key = (reference_run_id, cost_type, row["target_run_id"])
                clean_embeddings.append(embeddings[key])
        if not clean_embeddings:
            centroid = None
        else:
            centroid = np.vstack(clean_embeddings).mean(axis=0)
        for row in scoped_rows:
            if centroid is None:
                row["embedding_l2_to_clean_centroid"] = np.nan
                continue
            key = (reference_run_id, cost_type, row["target_run_id"])
            row["embedding_l2_to_clean_centroid"] = float(np.linalg.norm(embeddings[key] - centroid))


def compute_fourier_summary_for_group(
    *,
    input_dir: Path,
    feature_columns: Sequence[str],
    cost_types: Sequence[str],
    reference_trial_ids_value: str,
    window_size: int,
    sinkhorn_reg: float,
    use_sinkhorn: bool,
    sinkhorn_num_iter: int,
    sinkhorn_stop_thr: float,
    normalize_solver_cost: bool,
    pca_components_for_residual: int,
    max_samples_per_run: int,
    segment_by: str,
    fourier_num_bins: int,
    fourier_spectrum: str,
    fourier_drop_dc: bool,
    fourier_detrend: bool,
    fourier_log_amplitude: bool,
    fourier_normalize: bool,
    verbose_ot: bool,
) -> pd.DataFrame:
    segmented_data, _metadata = ot_base.load_segmented_data_from_csv_dir(
        input_dir=input_dir,
        feature_columns=feature_columns,
        segment_by=segment_by,
        max_samples_per_run=max_samples_per_run,
    )

    all_rows: List[Dict[str, Any]] = []
    for segment_type, segment_id in sorted(segmented_data.keys(), key=numeric_segment_sort_key):
        print(f"  segment_type={segment_type} segment_id={segment_id}")
        transformed = fourier_base.transform_data_fourier(
            segmented_data[(segment_type, segment_id)],
            num_bins=fourier_num_bins,
            spectrum=fourier_spectrum,
            drop_dc=fourier_drop_dc,
            detrend=fourier_detrend,
            log_amplitude=fourier_log_amplitude,
            normalize=fourier_normalize,
        )
        for metric_index, source_column in enumerate(feature_columns):
            metric_name = ot_base.metric_name_for_column(source_column)
            metric_transform = f"{ot_base.metric_transform_for_column(source_column)}+fourier"
            metric_data = ot_base.slice_data_for_metric(transformed, metric_index)
            reference_trial_ids = ot_base.resolve_reference_trial_ids(metric_data, reference_trial_ids_value)
            print(f"    metric={metric_name} reference={reference_trial_ids[0]}")

            if verbose_ot:
                rows, embeddings = ot_base.compute_all_ot_embeddings(
                    data=metric_data,
                    reference_trial_ids=reference_trial_ids,
                    cost_types=cost_types,
                    window_size=window_size,
                    sinkhorn_reg=sinkhorn_reg,
                    use_sinkhorn=use_sinkhorn,
                    sinkhorn_num_iter=sinkhorn_num_iter,
                    sinkhorn_stop_thr=sinkhorn_stop_thr,
                    normalize_solver_cost=normalize_solver_cost,
                )
            else:
                with contextlib.redirect_stdout(io.StringIO()):
                    rows, embeddings = ot_base.compute_all_ot_embeddings(
                        data=metric_data,
                        reference_trial_ids=reference_trial_ids,
                        cost_types=cost_types,
                        window_size=window_size,
                        sinkhorn_reg=sinkhorn_reg,
                        use_sinkhorn=use_sinkhorn,
                        sinkhorn_num_iter=sinkhorn_num_iter,
                        sinkhorn_stop_thr=sinkhorn_stop_thr,
                        normalize_solver_cost=normalize_solver_cost,
                    )
            pcas = ot_base.fit_clean_pcas(
                summary_rows=rows,
                embeddings=embeddings,
                pca_components_for_residual=pca_components_for_residual,
            )
            ot_base.compute_residual_scores(rows, embeddings, pcas)
            compute_embedding_clean_distances(rows, embeddings)

            for row in rows:
                row["segment_type"] = segment_type
                row["segment_id"] = segment_id
                row["metric_name"] = metric_name
                row["source_column"] = source_column
                row["metric_transform"] = metric_transform
                row["preprocess"] = "fourier"
                row["fourier_spectrum"] = fourier_spectrum
                row["fourier_num_bins"] = fourier_num_bins
                row["fourier_drop_dc"] = fourier_drop_dc
                row["fourier_detrend"] = fourier_detrend
                row["fourier_log_amplitude"] = fourier_log_amplitude
                row["fourier_normalize"] = fourier_normalize
            all_rows.extend(rows)

    if not all_rows:
        raise ValueError(f"No Fourier OT rows were computed for {input_dir}")
    return pd.DataFrame(all_rows)


def is_reference_trial(trial_id: str) -> bool:
    return str(trial_id).startswith("reference_")


def run_sort_key(run: Dict[str, Any]) -> Tuple[int, int, str]:
    trial_id = str(run["target_trial_id"])
    prefix = 0 if trial_id.startswith("reference_") else 1
    try:
        number = int(trial_id.split("_", 1)[1])
    except (IndexError, ValueError):
        number = 10**9
    return (prefix, number, str(run["poisoning_type"]))


def collect_runs_from_data(data: Dict[str, Dict[str, Any]], include_reference_runs: bool) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for trial_id in sorted(data.keys(), key=ot_base._numeric_trial_id):
        if not include_reference_runs and is_reference_trial(trial_id):
            continue
        if "clean" in data[trial_id]:
            runs.append(
                {
                    "target_trial_id": trial_id,
                    "target_run_id": f"{trial_id}_clean",
                    "target_group": "clean",
                    "poisoning_type": "none",
                    "condition": "clean",
                    "run": data[trial_id]["clean"],
                }
            )
        for poisoning_type, run in sorted(data[trial_id].get("poisoning", {}).items()):
            runs.append(
                {
                    "target_trial_id": trial_id,
                    "target_run_id": f"{trial_id}_{poisoning_type}",
                    "target_group": "poisoning",
                    "poisoning_type": poisoning_type,
                    "condition": poisoning_type,
                    "run": run,
                }
            )
    return sorted(runs, key=run_sort_key)


def fast_uniform_1d_w2_distance(features_a: np.ndarray, features_b: np.ndarray) -> Optional[float]:
    """Exact squared W2 distance for equal-mass 1D empirical distributions.

    Pairwise classifier mode uses one metric at a time, so c2_value features are
    one-dimensional Fourier magnitudes. For uniform weights in 1D, the exact OT
    plan is monotone matching after sorting. This avoids thousands of Sinkhorn
    solves without changing the pairwise c2_value distance.
    """
    if features_a.ndim != 2 or features_b.ndim != 2:
        return None
    if features_a.shape[1] != 1 or features_b.shape[1] != 1:
        return None
    if features_a.shape[0] != features_b.shape[0]:
        return None
    a = np.sort(features_a[:, 0])
    b = np.sort(features_b[:, 0])
    return float(np.mean((a - b) ** 2))


def pairwise_ot_distance(
    features_a: np.ndarray,
    features_b: np.ndarray,
    args: argparse.Namespace,
) -> float:
    fast_distance = fast_uniform_1d_w2_distance(features_a, features_b)
    if fast_distance is not None:
        return fast_distance
    if args.verbose_ot:
        plan, cost, _cost_scale = ot_base.solve_ot(
            features_a,
            features_b,
            reg=args.sinkhorn_reg,
            use_sinkhorn=args.use_sinkhorn,
            sinkhorn_num_iter=args.sinkhorn_num_iter,
            sinkhorn_stop_thr=args.sinkhorn_stop_thr,
            normalize_solver_cost=args.normalize_solver_cost,
        )
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            plan, cost, _cost_scale = ot_base.solve_ot(
                features_a,
                features_b,
                reg=args.sinkhorn_reg,
                use_sinkhorn=args.use_sinkhorn,
                sinkhorn_num_iter=args.sinkhorn_num_iter,
                sinkhorn_stop_thr=args.sinkhorn_stop_thr,
                normalize_solver_cost=args.normalize_solver_cost,
            )
    return float(np.sum(plan * cost))


def aggregate_run_features(summary_df: pd.DataFrame, include_reference_runs: bool) -> pd.DataFrame:
    working = summary_df.copy()
    if not include_reference_runs:
        working = working[~working["target_trial_id"].astype(str).map(is_reference_trial)].copy()
    if working.empty:
        raise ValueError("No target runs remain after filtering reference runs.")

    base_cols = [
        "target_trial_id",
        "target_run_id",
        "target_group",
        "poisoning_type",
    ]
    value_cols = [
        "ot_cost",
        "tangent_norm",
        "residual_norm",
        "residual_ratio",
        "embedding_l2_to_clean_centroid",
    ]
    frames = []
    for (metric_name, cost_type), sub in working.groupby(["metric_name", "cost_type"], sort=False):
        grouped = sub.groupby(base_cols, sort=False)[value_cols].agg(["mean", "std", "max"])
        grouped.columns = [
            f"{metric_name}__{cost_type}__{value_col}__{stat}"
            for value_col, stat in grouped.columns.to_flat_index()
        ]
        frames.append(grouped)

    features = pd.concat(frames, axis=1).reset_index()
    features = features.fillna(0.0)
    features["is_poisoned"] = (features["target_group"] == "poisoning").astype(int)
    features["condition"] = np.where(
        features["target_group"] == "clean",
        "clean",
        features["poisoning_type"].astype(str),
    )
    return features


def compute_pairwise_fourier_run_features_for_device(
    *,
    group_label: str,
    group_dir: Path,
    args: argparse.Namespace,
    feature_columns: Sequence[str],
    cost_types: Sequence[str],
) -> pd.DataFrame:
    print(f"Computing pairwise Fourier OT features for device={group_label} input={group_dir}")
    segmented_data, _metadata = ot_base.load_segmented_data_from_csv_dir(
        input_dir=group_dir,
        feature_columns=feature_columns,
        segment_by=args.segment_by,
        max_samples_per_run=args.max_samples_per_run,
    )
    include_reference_runs = not args.exclude_reference_runs
    run_feature_rows: Dict[str, Dict[str, Any]] = {}

    for segment_type, segment_id in sorted(segmented_data.keys(), key=numeric_segment_sort_key):
        print(f"  segment_type={segment_type} segment_id={segment_id}")
        transformed = fourier_base.transform_data_fourier(
            segmented_data[(segment_type, segment_id)],
            num_bins=args.fourier_num_bins,
            spectrum=args.fourier_spectrum,
            drop_dc=args.fourier_drop_dc,
            detrend=args.fourier_detrend,
            log_amplitude=args.fourier_log_amplitude,
            normalize=args.fourier_normalize,
        )
        runs = collect_runs_from_data(transformed, include_reference_runs=include_reference_runs)
        if len(runs) < 2:
            raise ValueError(f"Need at least two runs for pairwise analysis in {group_dir}")

        for run in runs:
            run_id = run["target_run_id"]
            row = run_feature_rows.setdefault(
                run_id,
                {
                    "target_trial_id": run["target_trial_id"],
                    "target_run_id": run_id,
                    "target_group": run["target_group"],
                    "poisoning_type": run["poisoning_type"],
                    "condition": run["condition"],
                    "is_poisoned": int(run["target_group"] == "poisoning"),
                },
            )
            if row["condition"] != run["condition"]:
                raise ValueError(f"Conflicting condition metadata for run_id={run_id}")

        for metric_index, source_column in enumerate(feature_columns):
            metric_name = ot_base.metric_name_for_column(source_column)
            print(f"    metric={metric_name}")
            metric_runs = [
                {
                    **run,
                    "run": run["run"][:, [metric_index]],
                }
                for run in runs
            ]
            for cost_type in cost_types:
                features = [
                    ot_base.build_features_for_cost(
                        run["run"],
                        cost_type=cost_type,
                        window_size=args.window_size,
                    )
                    for run in metric_runs
                ]
                distances_by_run = {run["target_run_id"]: [] for run in metric_runs}
                for i in range(len(metric_runs)):
                    for j in range(i + 1, len(metric_runs)):
                        distance = pairwise_ot_distance(features[i], features[j], args)
                        distances_by_run[metric_runs[i]["target_run_id"]].append(distance)
                        distances_by_run[metric_runs[j]["target_run_id"]].append(distance)

                prefix = f"{metric_name}__{cost_type}__{segment_type}_{segment_id}__pairwise_ot"
                for run in metric_runs:
                    values = np.asarray(distances_by_run[run["target_run_id"]], dtype=np.float64)
                    row = run_feature_rows[run["target_run_id"]]
                    row[f"{prefix}__mean"] = float(values.mean())
                    row[f"{prefix}__std"] = float(values.std(ddof=0))
                    row[f"{prefix}__min"] = float(values.min())
                    row[f"{prefix}__max"] = float(values.max())
                    row[f"{prefix}__median"] = float(np.median(values))

    features_df = pd.DataFrame(list(run_feature_rows.values())).fillna(0.0)
    physical_device = host_from_group_label(group_label)
    device_type = device_type_for_group_label(group_label)
    features_df["source_group_label"] = group_label
    features_df["physical_device"] = physical_device
    features_df["device_type"] = device_type
    return features_df


def compute_pairwise_fourier_run_features_for_pooled_group(
    *,
    group_label: str,
    grouped_dirs: Sequence[Tuple[str, Path]],
    args: argparse.Namespace,
    feature_columns: Sequence[str],
    cost_types: Sequence[str],
) -> pd.DataFrame:
    """Build no-reference pairwise OT features across a pooled group.

    Unlike the older per-device pairwise path, this compares every run in the
    group against every other run in the group. For device_type grouping, that
    means RPI4 runs are compared across all RPI4 physical devices, Jetson runs
    across both Jetsons, etc.
    """
    print(f"Computing pooled pairwise Fourier OT features for group={group_label}")
    include_reference_runs = not args.exclude_reference_runs
    pooled_segments: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    aggregate_distance_matrix: Optional[np.ndarray] = None
    aggregate_run_ids: Optional[List[str]] = None
    aggregate_count = 0

    for source_group_label, group_dir in grouped_dirs:
        print(f"  loading device={source_group_label} input={group_dir}")
        segmented_data, _metadata = ot_base.load_segmented_data_from_csv_dir(
            input_dir=group_dir,
            feature_columns=feature_columns,
            segment_by=args.segment_by,
            max_samples_per_run=args.max_samples_per_run,
        )
        physical_device = host_from_group_label(source_group_label)
        device_type = device_type_for_group_label(source_group_label)
        for segment_key in sorted(segmented_data.keys(), key=numeric_segment_sort_key):
            transformed = fourier_base.transform_data_fourier(
                segmented_data[segment_key],
                num_bins=args.fourier_num_bins,
                spectrum=args.fourier_spectrum,
                drop_dc=args.fourier_drop_dc,
                detrend=args.fourier_detrend,
                log_amplitude=args.fourier_log_amplitude,
                normalize=args.fourier_normalize,
            )
            for run in collect_runs_from_data(transformed, include_reference_runs=include_reference_runs):
                pooled_run = dict(run)
                pooled_run["source_group_label"] = source_group_label
                pooled_run["physical_device"] = physical_device
                pooled_run["device_type"] = device_type
                pooled_run["global_run_id"] = f"{physical_device}__{run['target_run_id']}"
                pooled_segments.setdefault(segment_key, []).append(pooled_run)

    run_feature_rows: Dict[str, Dict[str, Any]] = {}
    for segment_key, runs in sorted(pooled_segments.items(), key=lambda item: numeric_segment_sort_key(item[0])):
        segment_type, segment_id = segment_key
        runs = sorted(runs, key=lambda run: (run["physical_device"], *run_sort_key(run)))
        print(f"  segment_type={segment_type} segment_id={segment_id} n_runs={len(runs)}")
        if len(runs) < 2:
            raise ValueError(f"Need at least two runs for pooled pairwise analysis in {group_label}")

        for run in runs:
            global_run_id = run["global_run_id"]
            run_feature_rows.setdefault(
                global_run_id,
                {
                    "target_trial_id": run["target_trial_id"],
                    "target_run_id": run["target_run_id"],
                    "target_group": run["target_group"],
                    "poisoning_type": run["poisoning_type"],
                    "condition": run["condition"],
                    "is_poisoned": int(run["target_group"] == "poisoning"),
                    "source_group_label": run["source_group_label"],
                    "physical_device": run["physical_device"],
                    "device_type": run["device_type"],
                },
            )

        for metric_index, source_column in enumerate(feature_columns):
            metric_name = ot_base.metric_name_for_column(source_column)
            print(f"    metric={metric_name}")
            metric_runs = [
                {
                    **run,
                    "run": run["run"][:, [metric_index]],
                }
                for run in runs
            ]
            for cost_type in cost_types:
                features = [
                    ot_base.build_features_for_cost(
                        run["run"],
                        cost_type=cost_type,
                        window_size=args.window_size,
                    )
                    for run in metric_runs
                ]
                n = len(metric_runs)
                distance_matrix = np.zeros((n, n), dtype=np.float64)
                for i in range(n):
                    for j in range(i + 1, n):
                        distance = pairwise_ot_distance(features[i], features[j], args)
                        distance_matrix[i, j] = distance
                        distance_matrix[j, i] = distance

                current_run_ids = [run["global_run_id"] for run in metric_runs]
                positive = distance_matrix[distance_matrix > 0.0]
                if positive.size:
                    scale = float(np.median(positive))
                    if not np.isfinite(scale) or scale <= 0.0:
                        scale = 1.0
                    normalized_matrix = distance_matrix / scale
                    if aggregate_distance_matrix is None:
                        aggregate_distance_matrix = np.zeros_like(normalized_matrix, dtype=np.float64)
                        aggregate_run_ids = current_run_ids
                    elif aggregate_run_ids != current_run_ids:
                        raise ValueError("Run ordering changed across pairwise distance matrices.")
                    aggregate_distance_matrix += normalized_matrix
                    aggregate_count += 1

                prefix = f"{metric_name}__{cost_type}__{segment_type}_{segment_id}__pooled_pairwise_ot"
                for i, run in enumerate(metric_runs):
                    values = np.delete(distance_matrix[i], i)
                    sorted_values = np.sort(values)
                    k = max(1, min(int(args.knn_k), len(sorted_values)))
                    far_k = max(1, min(int(args.far_k), len(sorted_values)))
                    row = run_feature_rows[run["global_run_id"]]
                    row[f"{prefix}__mean"] = float(values.mean())
                    row[f"{prefix}__std"] = float(values.std(ddof=0))
                    row[f"{prefix}__min"] = float(values.min())
                    row[f"{prefix}__max"] = float(values.max())
                    row[f"{prefix}__median"] = float(np.median(values))
                    row[f"{prefix}__q25"] = float(np.percentile(values, 25))
                    row[f"{prefix}__q75"] = float(np.percentile(values, 75))
                    row[f"{prefix}__knn{k}_mean"] = float(sorted_values[:k].mean())
                    row[f"{prefix}__far{far_k}_mean"] = float(sorted_values[-far_k:].mean())

    if aggregate_distance_matrix is not None and aggregate_run_ids is not None and aggregate_count > 0:
        aggregate_distance_matrix = aggregate_distance_matrix / float(aggregate_count)
        for i, global_run_id in enumerate(aggregate_run_ids):
            row = run_feature_rows[global_run_id]
            values = np.delete(aggregate_distance_matrix[i], i)
            sorted_values = np.sort(values)
            k = max(1, min(int(args.knn_k), len(sorted_values)))
            far_k = max(1, min(int(args.far_k), len(sorted_values)))
            row["aggregate_pairdist_mean"] = float(values.mean())
            row["aggregate_pairdist_std"] = float(values.std(ddof=0))
            row["aggregate_pairdist_median"] = float(np.median(values))
            row[f"aggregate_pairdist_knn{k}_mean"] = float(sorted_values[:k].mean())
            row[f"aggregate_pairdist_far{far_k}_mean"] = float(sorted_values[-far_k:].mean())
            for j, distance in enumerate(aggregate_distance_matrix[i]):
                if i == j:
                    # Avoid a one-hot self-zero feature dominating the clustering.
                    distance = row["aggregate_pairdist_median"]
                row[f"aggregate_pairdist_to_run_{j:03d}"] = float(distance)

    return pd.DataFrame(list(run_feature_rows.values())).fillna(0.0)


def feature_columns_for_classifier(run_df: pd.DataFrame, feature_set: str = "auto") -> List[str]:
    if feature_set in {"auto", "pairwise_distance_rows"}:
        pairwise_cols = [
            column
            for column in run_df.columns
            if column.startswith("aggregate_pairdist_to_run_")
            and pd.api.types.is_numeric_dtype(run_df[column])
            and float(run_df[column].abs().sum()) > 1e-12
            and float(run_df[column].std(ddof=0)) > 1e-12
        ]
        if pairwise_cols or feature_set == "pairwise_distance_rows":
            return pairwise_cols

    excluded = {
        "target_trial_id",
        "target_run_id",
        "target_group",
        "poisoning_type",
        "is_poisoned",
        "condition",
        "device_group",
        "group_label",
        "source_group_label",
        "physical_device",
        "device_type",
    }
    return [
        column
        for column in run_df.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(run_df[column])
    ]


def robust_scale(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.median(x, axis=0)
    q25 = np.percentile(x, 25, axis=0)
    q75 = np.percentile(x, 75, axis=0)
    scale = q75 - q25
    scale[scale < 1e-12] = 1.0
    return (x - center) / scale, center, scale


def pca_2d(x: np.ndarray) -> np.ndarray:
    if x.shape[0] == 0:
        return np.empty((0, 2))
    pca = ot_base.PCANP(n_components=min(2, x.shape[0], x.shape[1])).fit(x)
    coords = pca.transform(x)
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(coords.shape[0])])
    return coords[:, :2]


def kmeans2(x: np.ndarray, max_iter: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Small deterministic k-means with k=2, no sklearn dependency."""
    if x.shape[0] < 2:
        return np.zeros(x.shape[0], dtype=int), np.repeat(x.mean(axis=0, keepdims=True), 2, axis=0)
    norms = np.linalg.norm(x, axis=1)
    first = int(np.argmin(norms))
    second = int(np.argmax(norms))
    if first == second:
        second = 1 if first == 0 else 0
    centers = np.vstack([x[first], x[second]]).astype(np.float64)
    labels = np.zeros(x.shape[0], dtype=int)
    for _ in range(max_iter):
        distances = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for cluster_id in range(2):
            members = x[labels == cluster_id]
            if len(members) == 0:
                centers[cluster_id] = x[int(np.argmax(norms))]
            else:
                centers[cluster_id] = members.mean(axis=0)
    return labels, centers


def best_binary_cluster_mapping(labels: np.ndarray, truth: np.ndarray) -> Tuple[np.ndarray, int, float]:
    """Map two arbitrary cluster ids to clean/poison labels for evaluation only."""
    labels = labels.astype(int)
    truth = truth.astype(int)
    pred_a = labels.copy()
    pred_b = 1 - labels
    acc_a = float((pred_a == truth).mean()) if len(truth) else np.nan
    acc_b = float((pred_b == truth).mean()) if len(truth) else np.nan
    if acc_a >= acc_b:
        poison_cluster = 1
        return pred_a, poison_cluster, acc_a
    poison_cluster = 0
    return pred_b, poison_cluster, acc_b


def add_unsupervised_scores(run_df: pd.DataFrame, feature_set: str = "auto") -> Tuple[pd.DataFrame, List[str]]:
    out = run_df.copy()
    feature_cols = feature_columns_for_classifier(out, feature_set=feature_set)
    if not feature_cols:
        raise ValueError("No numeric run-level features available for classifier.")
    x = out[feature_cols].to_numpy(dtype=np.float64)
    x_scaled, _, _ = robust_scale(x)
    x_scaled = np.nan_to_num(x_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    coords = pca_2d(x_scaled)
    labels, centers = kmeans2(coords)
    distances_to_centers = np.linalg.norm(coords[:, None, :] - centers[None, :, :], axis=2)
    own_cluster_distance = distances_to_centers[np.arange(len(labels)), labels]
    nearest_other_distance = distances_to_centers[np.arange(len(labels)), 1 - labels]
    margin = nearest_other_distance - own_cluster_distance
    truth = out["is_poisoned"].to_numpy(dtype=int)
    posthoc_predicted, posthoc_poison_cluster, posthoc_accuracy = best_binary_cluster_mapping(labels, truth)

    out["unsupervised_score"] = own_cluster_distance
    out["pca_cluster_distance"] = own_cluster_distance
    out["pca_cluster_margin"] = margin
    out["cluster_id"] = labels
    out["posthoc_poison_cluster"] = posthoc_poison_cluster
    out["posthoc_predicted_poisoned"] = posthoc_predicted
    out["posthoc_cluster_accuracy"] = posthoc_accuracy
    # Kept for compatibility with older result readers. This is post-hoc only,
    # because unsupervised cluster ids have no inherent clean/poison direction.
    out["predicted_anomaly_cluster"] = posthoc_predicted
    out["pca_x"] = coords[:, 0]
    out["pca_y"] = coords[:, 1]
    return out, feature_cols


def summarize_classifier(run_df: pd.DataFrame, group_label: str) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "group_label": group_label,
        "n_runs": int(len(run_df)),
        "n_clean": int((run_df["is_poisoned"] == 0).sum()),
        "n_poisoned": int((run_df["is_poisoned"] == 1).sum()),
    }
    predicted = run_df["posthoc_predicted_poisoned"].astype(int)
    raw_cluster = run_df["cluster_id"].astype(int)
    truth = run_df["is_poisoned"].astype(int)
    summary.update(
        {
            "tp": int(((predicted == 1) & (truth == 1)).sum()),
            "fp": int(((predicted == 1) & (truth == 0)).sum()),
            "tn": int(((predicted == 0) & (truth == 0)).sum()),
            "fn": int(((predicted == 0) & (truth == 1)).sum()),
        }
    )
    denom = len(run_df)
    summary["cluster_label_match_accuracy"] = float(((predicted == truth).sum()) / denom) if denom else np.nan
    raw_acc = float((raw_cluster == truth).mean()) if denom else np.nan
    flipped_acc = float(((1 - raw_cluster) == truth).mean()) if denom else np.nan
    summary["best_cluster_label_accuracy"] = max(raw_acc, flipped_acc)
    summary["raw_cluster_accuracy"] = raw_acc
    summary["flipped_cluster_accuracy"] = flipped_acc
    summary["posthoc_poison_cluster"] = int(run_df["posthoc_poison_cluster"].iloc[0]) if len(run_df) else -1
    for cluster_id, sub in run_df.groupby("cluster_id", sort=True):
        summary[f"cluster_{cluster_id}_n"] = int(len(sub))
        summary[f"cluster_{cluster_id}_poisoned_fraction"] = float(sub["is_poisoned"].mean())
    for condition, sub in run_df.groupby("condition", sort=False):
        summary[f"mean_score_{condition}"] = float(sub["unsupervised_score"].mean())
        summary[f"std_score_{condition}"] = float(sub["unsupervised_score"].std(ddof=0))
    return summary


def make_plots(run_df: pd.DataFrame, output_dir: Path, group_label: str) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib_cache"))
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots" / safe_name(group_label)
    plot_dir.mkdir(parents=True, exist_ok=True)
    palette = {
        "clean": "tab:blue",
        "unlearnable_examples": "tab:red",
        "availability_shortcuts": "tab:orange",
        "random_label_flipping": "tab:green",
        "target_label_flipping": "tab:purple",
    }

    fig, ax = plt.subplots(figsize=(7, 4))
    conditions = list(dict.fromkeys(run_df["condition"].tolist()))
    data = [run_df.loc[run_df["condition"] == condition, "unsupervised_score"].to_numpy() for condition in conditions]
    try:
        ax.boxplot(data, tick_labels=conditions, showfliers=True)
    except TypeError:
        ax.boxplot(data, labels=conditions, showfliers=True)
    for idx, condition in enumerate(conditions, start=1):
        y = run_df.loc[run_df["condition"] == condition, "unsupervised_score"].to_numpy()
        x = np.full(len(y), idx, dtype=float)
        if len(y):
            x += np.linspace(-0.08, 0.08, len(y))
        ax.scatter(x, y, color=palette.get(condition, "tab:gray"), alpha=0.75, s=28)
    ax.set_title(f"Unsupervised score distribution: {group_label}")
    ax.set_ylabel("robust feature-space norm")
    ax.grid(True, axis="y", alpha=0.3)
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    fig.savefig(plot_dir / "score_distribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for condition, sub in run_df.groupby("condition", sort=False):
        ax.scatter(
            sub["pca_x"],
            sub["pca_y"],
            color=palette.get(condition, "tab:gray"),
            label=condition,
            s=36,
            alpha=0.8,
        )
    ax.set_title(f"Run-level feature PCA: {group_label}")
    ax.set_xlabel("PCA x")
    ax.set_ylabel("PCA y")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_dir / "feature_pca_by_condition.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for cluster_id, sub in run_df.groupby("cluster_id", sort=True):
        ax.scatter(
            sub["pca_x"],
            sub["pca_y"],
            label=f"cluster {cluster_id}",
            s=36,
            alpha=0.8,
        )
    ax.set_title(f"Unsupervised clusters: {group_label}")
    ax.set_xlabel("PCA x")
    ax.set_ylabel("PCA y")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_dir / "feature_pca_by_cluster.png", dpi=180)
    plt.close(fig)


def compute_device_run_features(
    *,
    group_label: str,
    group_dir: Path,
    args: argparse.Namespace,
    feature_columns: Sequence[str],
    cost_types: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Computing Fourier OT features for device={group_label} input={group_dir}")
    summary_df = compute_fourier_summary_for_group(
        input_dir=group_dir,
        feature_columns=feature_columns,
        cost_types=cost_types,
        reference_trial_ids_value=args.reference_trial_id or args.reference_trial_ids,
        window_size=args.window_size,
        sinkhorn_reg=args.sinkhorn_reg,
        use_sinkhorn=args.use_sinkhorn,
        sinkhorn_num_iter=args.sinkhorn_num_iter,
        sinkhorn_stop_thr=args.sinkhorn_stop_thr,
        normalize_solver_cost=args.normalize_solver_cost,
        pca_components_for_residual=args.pca_components_for_residual,
        max_samples_per_run=args.max_samples_per_run,
        segment_by=args.segment_by,
        fourier_num_bins=args.fourier_num_bins,
        fourier_spectrum=args.fourier_spectrum,
        fourier_drop_dc=args.fourier_drop_dc,
        fourier_detrend=args.fourier_detrend,
        fourier_log_amplitude=args.fourier_log_amplitude,
        fourier_normalize=args.fourier_normalize,
        verbose_ot=args.verbose_ot,
    )
    run_features = aggregate_run_features(summary_df, include_reference_runs=args.include_reference_runs)
    physical_device = host_from_group_label(group_label)
    device_type = device_type_for_group_label(group_label)
    summary_df["source_group_label"] = group_label
    summary_df["physical_device"] = physical_device
    summary_df["device_type"] = device_type
    run_features["source_group_label"] = group_label
    run_features["physical_device"] = physical_device
    run_features["device_type"] = device_type
    return summary_df, run_features


def run_one_group(
    *,
    group_label: str,
    group_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    feature_columns: Sequence[str],
    cost_types: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    summary_df, run_features = compute_device_run_features(
        group_label=group_label,
        group_dir=group_dir,
        args=args,
        feature_columns=feature_columns,
        cost_types=cost_types,
    )
    run_features["group_label"] = group_label
    scored, feature_cols = add_unsupervised_scores(run_features, feature_set=args.classifier_feature_set)
    make_plots(scored, output_dir, group_label)
    group_summary = summarize_classifier(scored, group_label)
    group_summary["n_features"] = len(feature_cols)
    group_summary["feature_columns"] = ",".join(feature_cols)
    return summary_df, scored, group_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="collected_logs")
    parser.add_argument("--output_dir", default="fourier_classifier_result")
    parser.add_argument(
        "--analysis_mode",
        choices=["pairwise", "reference"],
        default="pairwise",
        help="pairwise uses no fixed reference; reference uses script-4 style clean-reference OT features.",
    )
    parser.add_argument(
        "--grouping",
        choices=["device", "device_type"],
        default="device_type",
        help="Cluster per physical device or pool runs by device type before clustering.",
    )
    parser.add_argument("--reference_trial_ids", default="global_reference")
    parser.add_argument("--reference_trial_id", default="")
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--sinkhorn_reg", type=float, default=1.0)
    parser.add_argument("--sinkhorn_num_iter", type=int, default=300)
    parser.add_argument("--sinkhorn_stop_thr", type=float, default=1e-6)
    parser.add_argument("--normalize_solver_cost", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_sinkhorn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pca_components_for_residual", type=int, default=3)
    parser.add_argument("--feature_columns", default=",".join(DEFAULT_FEATURE_COLUMNS))
    parser.add_argument("--cost_types", default=",".join(DEFAULT_COST_TYPES))
    parser.add_argument("--segment_by", choices=["auto", "epoch", "round", "none"], default="auto")
    parser.add_argument("--max_samples_per_run", type=int, default=0)
    parser.add_argument("--fourier_num_bins", type=int, default=128)
    parser.add_argument("--fourier_spectrum", choices=["magnitude", "power"], default="magnitude")
    parser.add_argument("--fourier_drop_dc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fourier_detrend", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fourier_log_amplitude", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fourier_normalize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include_reference_runs", action="store_true")
    parser.add_argument(
        "--exclude_reference_runs",
        action="store_true",
        help="Only for --analysis_mode pairwise. By default reference_* runs are ordinary clean runs.",
    )
    parser.add_argument("--verbose_ot", action="store_true")
    parser.add_argument("--knn_k", type=int, default=5)
    parser.add_argument("--far_k", type=int, default=5)
    parser.add_argument(
        "--classifier_feature_set",
        choices=["auto", "all", "pairwise_distance_rows"],
        default="auto",
        help="auto uses pooled pairwise distance-row features when present, otherwise all numeric features.",
    )
    parser.add_argument(
        "--max_groups",
        type=int,
        default=0,
        help="Optional debug limit on discovered device groups. 0 means all.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = Path(args.input_dir)
    feature_columns = ot_base.parse_feature_columns(args.feature_columns)
    cost_types = ot_base.parse_cost_types(args.cost_types)
    groups = ot_base.discover_input_groups(input_dir)
    if args.max_groups > 0:
        groups = groups[: args.max_groups]

    all_summary = []
    all_runs = []
    group_summaries = []
    if args.analysis_mode == "pairwise":
        if args.grouping == "device":
            grouped_dirs = [(group_label, [(group_label, group_dir)]) for group_label, group_dir in groups]
        else:
            type_to_groups: Dict[str, List[Tuple[str, Path]]] = {}
            for group_label, group_dir in groups:
                type_to_groups.setdefault(device_type_for_group_label(group_label), []).append((group_label, group_dir))
            grouped_dirs = sorted(type_to_groups.items())

        for group_label, grouped_group_dirs in grouped_dirs:
            group_features = compute_pairwise_fourier_run_features_for_pooled_group(
                group_label=str(group_label),
                grouped_dirs=grouped_group_dirs,
                args=args,
                feature_columns=feature_columns,
                cost_types=cost_types,
            )
            group_features["group_label"] = str(group_label)
            scored_df, feature_cols = add_unsupervised_scores(
                group_features,
                feature_set=args.classifier_feature_set,
            )
            group_output_dir = output_dir if len(grouped_dirs) == 1 else output_dir / safe_name(str(group_label))
            make_plots(scored_df, group_output_dir, str(group_label))
            group_summary = summarize_classifier(scored_df, str(group_label))
            group_summary["analysis_mode"] = "pairwise"
            group_summary["n_features"] = len(feature_cols)
            group_summary["feature_columns"] = ",".join(feature_cols)
            group_summary["n_physical_devices"] = int(scored_df["physical_device"].nunique())
            group_summary["physical_devices"] = ",".join(sorted(scored_df["physical_device"].unique()))
            all_runs.append(scored_df)
            group_summaries.append(group_summary)

    elif args.grouping == "device":
        for group_label, group_dir in groups:
            group_output_dir = output_dir if len(groups) == 1 else output_dir / group_label
            summary_df, scored_df, group_summary = run_one_group(
                group_label=group_label,
                group_dir=group_dir,
                output_dir=group_output_dir,
                args=args,
                feature_columns=feature_columns,
                cost_types=cost_types,
            )
            summary_df["group_label"] = group_label
            all_summary.append(summary_df)
            all_runs.append(scored_df)
            group_summaries.append(group_summary)
    else:
        type_to_groups: Dict[str, List[Tuple[str, Path]]] = {}
        for group_label, group_dir in groups:
            type_to_groups.setdefault(device_type_for_group_label(group_label), []).append((group_label, group_dir))

        for device_type, typed_groups in sorted(type_to_groups.items()):
            print(f"Running Fourier unsupervised classifier for device_type={device_type}")
            typed_summary = []
            typed_run_features = []
            for group_label, group_dir in typed_groups:
                summary_df, run_features = compute_device_run_features(
                    group_label=group_label,
                    group_dir=group_dir,
                    args=args,
                    feature_columns=feature_columns,
                    cost_types=cost_types,
                )
                summary_df["group_label"] = device_type
                typed_summary.append(summary_df)
                typed_run_features.append(run_features)

            combined_features = pd.concat(typed_run_features, ignore_index=True)
            combined_features["group_label"] = device_type
            scored_df, feature_cols = add_unsupervised_scores(
                combined_features,
                feature_set=args.classifier_feature_set,
            )
            make_plots(scored_df, output_dir / device_type, device_type)
            group_summary = summarize_classifier(scored_df, device_type)
            group_summary["n_features"] = len(feature_cols)
            group_summary["feature_columns"] = ",".join(feature_cols)
            group_summary["n_physical_devices"] = int(scored_df["physical_device"].nunique())
            group_summary["physical_devices"] = ",".join(sorted(scored_df["physical_device"].unique()))

            all_summary.append(pd.concat(typed_summary, ignore_index=True))
            all_runs.append(scored_df)
            group_summaries.append(group_summary)

    if all_summary:
        summary_all = pd.concat(all_summary, ignore_index=True)
    else:
        summary_all = pd.DataFrame()
    runs_all = pd.concat(all_runs, ignore_index=True)
    report = pd.DataFrame(group_summaries)
    summary_all.to_csv(output_dir / "fourier_ot_point_summary.csv", index=False)
    runs_all.to_csv(output_dir / "fourier_run_level_unsupervised_scores.csv", index=False)
    report.to_csv(output_dir / "fourier_unsupervised_classifier_report.csv", index=False)
    print(f"Saved outputs to {output_dir}")
    print(
        report[
            [
                "group_label",
                "n_runs",
                "n_clean",
                "n_poisoned",
                "best_cluster_label_accuracy",
                "cluster_label_match_accuracy",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
