#!/usr/bin/env python3
"""Unsupervised condition separation from raw OT distances.

This is the non-Fourier counterpart of ``5_fourier_classification_performance.py``.
The default path does not use a fixed clean reference. For each device type, it:

1. Loads runs from all physical devices in that device type.
2. Splits telemetry by epoch/round.
3. Builds raw OT features directly from the selected telemetry columns.
4. For each device_type/trial/epoch subset, builds a central reference from the
   runs in that subset.
5. Computes each run's OT distance and displacement embedding from that central
   reference.
6. Fits unsupervised two-cluster k-means separately for each subset.

Known clean/poisoning labels are used only for post-hoc reporting and plots.
They are not used to build the central reference, fit k-means, or set a threshold.
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
OT_MODULE_PATH = SCRIPT_DIR / "1_calculate_ot_embedding.py"


def _load_ot_module() -> Any:
    spec = importlib.util.spec_from_file_location("raw_ot_base", OT_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load raw OT module: {OT_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ot_base = _load_ot_module()


DEFAULT_FEATURE_COLUMNS = [
    "system_cpu_core_1",
    "system_cpu_core_2",
    "system_cpu_core_3",
]
DEFAULT_COST_TYPES = [
    "c2_value_shape",
]
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


def exact_uniform_1d_w2_distance(features_a: np.ndarray, features_b: np.ndarray) -> Optional[float]:
    """Exact squared W2 distance for 1D uniform empirical distributions.

    This keeps raw c1/c2/c2_value_shape pairwise analysis fast without changing
    the 1D OT distance. It also supports unequal run lengths by integrating the
    squared difference between empirical quantile functions.
    """
    if features_a.ndim != 2 or features_b.ndim != 2:
        return None
    if features_a.shape[1] != 1 or features_b.shape[1] != 1:
        return None

    a = np.sort(features_a[:, 0].astype(np.float64, copy=False))
    b = np.sort(features_b[:, 0].astype(np.float64, copy=False))
    n = len(a)
    m = len(b)
    if n == 0 or m == 0:
        return None

    # Exact quantile integration. The empirical quantile functions are constant
    # on intervals whose boundaries are multiples of 1/n and 1/m.
    boundaries = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 1.0, n + 1, dtype=np.float64),
                np.linspace(0.0, 1.0, m + 1, dtype=np.float64),
            ]
        )
    )
    left = boundaries[:-1]
    right = boundaries[1:]
    widths = right - left
    mids = (left + right) * 0.5
    idx_a = np.minimum((mids * n).astype(np.int64), n - 1)
    idx_b = np.minimum((mids * m).astype(np.int64), m - 1)
    diff = a[idx_a] - b[idx_b]
    return float(np.sum(widths * diff * diff))


def fixed_quantile_support(features: np.ndarray, num_bins: int) -> Optional[np.ndarray]:
    """Represent a 1D empirical distribution on a fixed quantile support.

    For c2_value_shape this gives every run the same number of sorted support
    points. It is not the final embedding in central-reference mode; it is the
    source/target support used to compute a transport plan.
    """
    if features.ndim != 2 or features.shape[1] != 1 or features.shape[0] == 0:
        return None
    bins = int(num_bins)
    if bins <= 0:
        raise ValueError("embedding_bins must be positive.")
    values = np.sort(features[:, 0].astype(np.float64, copy=False))
    quantiles = (np.arange(bins, dtype=np.float64) + 0.5) / bins
    positions = quantiles * (len(values) - 1)
    left = np.floor(positions).astype(np.int64)
    right = np.ceil(positions).astype(np.int64)
    weight = positions - left
    return (1.0 - weight) * values[left] + weight * values[right]


def transport_displacement_embedding(
    source_support: np.ndarray,
    target_support: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, float]:
    """Compute mass-weighted barycentric displacement from an OT plan.

    For each source bin i:
        T_i = sum_j P_ij target_j / (sum_j P_ij + eps)
        e_i = a_i * (T_i - source_i)

    The returned embedding has one value per source bin, not a flattened
    transport matrix.
    """
    source_features = np.asarray(source_support, dtype=np.float32).reshape(-1, 1)
    target_features = np.asarray(target_support, dtype=np.float32).reshape(-1, 1)
    if (
        source_features.shape == target_features.shape
        and source_features.shape[1] == 1
        and np.all(np.diff(source_features[:, 0]) >= -1e-12)
        and np.all(np.diff(target_features[:, 0]) >= -1e-12)
    ):
        # Exact 1D OT for equal uniform masses: the optimal plan is monotone
        # matching. This is equivalent to P_ii = 1/n, so
        # e_i = a_i * (target_i - source_i).
        n = source_features.shape[0]
        row_mass = 1.0 / n
        displacement = target_features[:, 0] - source_features[:, 0]
        embedding = row_mass * displacement
        ot_cost = float(row_mass * np.sum(displacement * displacement))
        return embedding.astype(np.float64, copy=False), ot_cost

    if args.verbose_ot:
        plan, cost, _cost_scale = ot_base.solve_ot(
            source_features,
            target_features,
            reg=args.sinkhorn_reg,
            use_sinkhorn=args.use_sinkhorn,
            sinkhorn_num_iter=args.sinkhorn_num_iter,
            sinkhorn_stop_thr=args.sinkhorn_stop_thr,
            normalize_solver_cost=args.normalize_solver_cost,
        )
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            plan, cost, _cost_scale = ot_base.solve_ot(
                source_features,
                target_features,
                reg=args.sinkhorn_reg,
                use_sinkhorn=args.use_sinkhorn,
                sinkhorn_num_iter=args.sinkhorn_num_iter,
                sinkhorn_stop_thr=args.sinkhorn_stop_thr,
                normalize_solver_cost=args.normalize_solver_cost,
            )
    row_mass = plan.sum(axis=1)
    transported = (plan @ target_features[:, 0]) / (row_mass + 1e-12)
    displacement = transported - source_features[:, 0]
    embedding = row_mass * displacement
    return embedding.astype(np.float64, copy=False), float(np.sum(plan * cost))


def pairwise_ot_distance(
    features_a: np.ndarray,
    features_b: np.ndarray,
    args: argparse.Namespace,
) -> float:
    fast_distance = exact_uniform_1d_w2_distance(features_a, features_b)
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


def compute_embedding_clean_distances(
    rows: List[Dict[str, Any]],
    embeddings: Dict[Tuple[str, str, str], np.ndarray],
) -> None:
    """Reference-mode diagnostic: distance to clean tangent-embedding centroid."""
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
        centroid = np.vstack(clean_embeddings).mean(axis=0) if clean_embeddings else None
        for row in scoped_rows:
            if centroid is None:
                row["embedding_l2_to_clean_centroid"] = np.nan
                continue
            key = (reference_run_id, cost_type, row["target_run_id"])
            row["embedding_l2_to_clean_centroid"] = float(np.linalg.norm(embeddings[key] - centroid))


def compute_raw_summary_for_group(
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
        segment_data = segmented_data[(segment_type, segment_id)]
        for metric_index, source_column in enumerate(feature_columns):
            metric_name = ot_base.metric_name_for_column(source_column)
            metric_data = ot_base.slice_data_for_metric(segment_data, metric_index)
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
                row["metric_transform"] = ot_base.metric_transform_for_column(source_column)
                row["preprocess"] = "raw"
            all_rows.extend(rows)

    if not all_rows:
        raise ValueError(f"No raw OT rows were computed for {input_dir}")
    return pd.DataFrame(all_rows)


def aggregate_reference_run_features(summary_df: pd.DataFrame, include_reference_runs: bool) -> pd.DataFrame:
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


def compute_pairwise_raw_run_features_for_pooled_group(
    *,
    group_label: str,
    grouped_dirs: Sequence[Tuple[str, Path]],
    args: argparse.Namespace,
    feature_columns: Sequence[str],
    cost_types: Sequence[str],
) -> pd.DataFrame:
    """Build no-reference raw pairwise OT features across a pooled group."""
    print(f"Computing pooled pairwise raw OT features for group={group_label}")
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
            for run in collect_runs_from_data(segmented_data[segment_key], include_reference_runs=include_reference_runs):
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
                    "global_run_id": global_run_id,
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
            metric_runs = [{**run, "run": run["run"][:, [metric_index]]} for run in runs]
            for cost_type in cost_types:
                features = [
                    ot_base.build_features_for_cost(
                        run["run"],
                        cost_type=cost_type,
                        window_size=args.window_size,
                    )
                    for run in metric_runs
                ]
                quantile_supports = [
                    fixed_quantile_support(feature, args.embedding_bins)
                    for feature in features
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

                prefix = f"{metric_name}__{cost_type}__{segment_type}_{segment_id}__raw_pairwise_ot"
                emb_prefix = f"{metric_name}__{cost_type}__{segment_type}_{segment_id}__quantile_embedding"
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
                    support = quantile_supports[i]
                    if support is not None:
                        row[f"{emb_prefix}__mean"] = float(np.mean(support))
                        row[f"{emb_prefix}__std"] = float(np.std(support))
                        row[f"{emb_prefix}__min"] = float(np.min(support))
                        row[f"{emb_prefix}__max"] = float(np.max(support))
                        for emb_idx, emb_value in enumerate(support):
                            row[f"{emb_prefix}_{emb_idx:03d}"] = float(emb_value)

    if aggregate_distance_matrix is not None and aggregate_run_ids is not None and aggregate_count > 0:
        aggregate_distance_matrix = aggregate_distance_matrix / float(aggregate_count)
        for i, global_run_id in enumerate(aggregate_run_ids):
            row = run_feature_rows[global_run_id]
            row["aggregate_pairdist_order_index"] = int(i)
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
                    distance = row["aggregate_pairdist_median"]
                row[f"aggregate_pairdist_to_run_{j:03d}"] = float(distance)

    return pd.DataFrame(list(run_feature_rows.values())).fillna(0.0)


def compute_central_reference_run_features_for_pooled_group(
    *,
    group_label: str,
    grouped_dirs: Sequence[Tuple[str, Path]],
    args: argparse.Namespace,
    feature_columns: Sequence[str],
    cost_types: Sequence[str],
) -> pd.DataFrame:
    """Build central-reference OT/embedding features per trial and segment.

    The central reference is computed independently for each
    device_type/trial/epoch subset as the mean fixed-quantile distribution of
    all runs in that subset. This is unsupervised: clean/poison labels are not
    used to build the center.
    """
    print(f"Computing central-reference raw OT features for group={group_label}")
    pooled: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}

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
            segment_type, segment_id = segment_key
            runs = collect_runs_from_data(segmented_data[segment_key], include_reference_runs=False)
            for run in runs:
                trial_id = str(run["target_trial_id"])
                if is_reference_trial(trial_id):
                    continue
                pooled_run = dict(run)
                pooled_run["source_group_label"] = source_group_label
                pooled_run["physical_device"] = physical_device
                pooled_run["device_type"] = device_type
                pooled_run["segment_type"] = segment_type
                pooled_run["segment_id"] = segment_id
                pooled_run["analysis_group_id"] = f"{group_label}__{trial_id}__{segment_type}_{segment_id}"
                pooled_run["global_run_id"] = (
                    f"{physical_device}__{trial_id}__{run['condition']}__{segment_type}_{segment_id}"
                )
                pooled.setdefault((trial_id, segment_type, segment_id), []).append(pooled_run)

    run_feature_rows: List[Dict[str, Any]] = []
    for (trial_id, segment_type, segment_id), runs in sorted(
        pooled.items(),
        key=lambda item: (ot_base._numeric_trial_id(item[0][0]), *numeric_segment_sort_key((item[0][1], item[0][2]))),
    ):
        runs = sorted(runs, key=lambda run: (run["physical_device"], run["condition"]))
        print(f"  trial_id={trial_id} segment_type={segment_type} segment_id={segment_id} n_runs={len(runs)}")
        if len(runs) < 2:
            continue

        rows_by_run_id: Dict[str, Dict[str, Any]] = {}
        for run in runs:
            rows_by_run_id[run["global_run_id"]] = {
                "analysis_group_id": run["analysis_group_id"],
                "target_trial_id": run["target_trial_id"],
                "target_run_id": run["target_run_id"],
                "target_group": run["target_group"],
                "poisoning_type": run["poisoning_type"],
                "condition": run["condition"],
                "is_poisoned": int(run["target_group"] == "poisoning"),
                "source_group_label": run["source_group_label"],
                "physical_device": run["physical_device"],
                "device_type": run["device_type"],
                "segment_type": segment_type,
                "segment_id": segment_id,
                "global_run_id": run["global_run_id"],
            }

        for metric_index, source_column in enumerate(feature_columns):
            metric_name = ot_base.metric_name_for_column(source_column)
            print(f"    metric={metric_name}")
            metric_runs = [{**run, "run": run["run"][:, [metric_index]]} for run in runs]
            for cost_type in cost_types:
                if cost_type != "c2_value_shape":
                    raise ValueError("Central-reference mode currently expects --cost_types c2_value_shape.")
                supports = []
                support_run_ids = []
                for run in metric_runs:
                    features = ot_base.build_features_for_cost(
                        run["run"],
                        cost_type=cost_type,
                        window_size=args.window_size,
                    )
                    support = fixed_quantile_support(features, args.embedding_bins)
                    if support is None:
                        continue
                    supports.append(support)
                    support_run_ids.append(run["global_run_id"])
                if len(supports) < 2:
                    continue
                stacked = np.vstack(supports)
                central_reference = stacked.mean(axis=0)
                prefix = f"{metric_name}__{cost_type}__central_reference"
                for global_run_id, target_support in zip(support_run_ids, supports):
                    transport_embedding, ot_cost = transport_displacement_embedding(
                        central_reference,
                        target_support,
                        args,
                    )
                    row = rows_by_run_id[global_run_id]
                    row[f"{prefix}__ot_cost"] = ot_cost
                    row[f"{prefix}__tangent_norm"] = float(np.linalg.norm(transport_embedding))
                    row[f"{prefix}__embedding_mean"] = float(transport_embedding.mean())
                    row[f"{prefix}__embedding_std"] = float(transport_embedding.std(ddof=0))
                    row[f"{prefix}__embedding_min"] = float(transport_embedding.min())
                    row[f"{prefix}__embedding_max"] = float(transport_embedding.max())
                    for emb_idx, emb_value in enumerate(transport_embedding):
                        row[f"{prefix}__emb_{emb_idx:03d}"] = float(emb_value)

        run_feature_rows.extend(rows_by_run_id.values())

    return pd.DataFrame(run_feature_rows).fillna(0.0)


def compute_reference_raw_run_features(
    *,
    group_label: str,
    group_dir: Path,
    args: argparse.Namespace,
    feature_columns: Sequence[str],
    cost_types: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Computing reference raw OT/embedding features for device={group_label} input={group_dir}")
    summary_df = compute_raw_summary_for_group(
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
        verbose_ot=args.verbose_ot,
    )
    run_features = aggregate_reference_run_features(summary_df, include_reference_runs=args.include_reference_runs)
    physical_device = host_from_group_label(group_label)
    device_type = device_type_for_group_label(group_label)
    summary_df["source_group_label"] = group_label
    summary_df["physical_device"] = physical_device
    summary_df["device_type"] = device_type
    run_features["source_group_label"] = group_label
    run_features["physical_device"] = physical_device
    run_features["device_type"] = device_type
    return summary_df, run_features


def feature_columns_for_classifier(run_df: pd.DataFrame, feature_set: str = "auto") -> List[str]:
    embedding_cols = [
        column
        for column in run_df.columns
        if (
            "__quantile_embedding_" in column
            or "__central_reference__emb_" in column
            or "__central_reference__ot_cost" in column
            or "__central_reference__tangent_norm" in column
        )
        and pd.api.types.is_numeric_dtype(run_df[column])
        and float(run_df[column].std(ddof=0)) > 1e-12
    ]
    pairwise_cols = [
        column
        for column in run_df.columns
        if column.startswith("aggregate_pairdist_to_run_")
        and pd.api.types.is_numeric_dtype(run_df[column])
        and float(run_df[column].abs().sum()) > 1e-12
        and float(run_df[column].std(ddof=0)) > 1e-12
    ]
    if feature_set == "embedding":
        return embedding_cols
    if feature_set == "pairwise_distance_rows":
        return pairwise_cols
    if feature_set in {"auto", "embedding_and_distance"} and (embedding_cols or pairwise_cols):
        return embedding_cols + pairwise_cols

    excluded = {
        "global_run_id",
        "analysis_group_id",
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
        "segment_type",
        "segment_id",
        "aggregate_pairdist_order_index",
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


def classical_mds_2d(distance_matrix: np.ndarray) -> np.ndarray:
    """Plot-only 2D embedding from pairwise OT distances."""
    if distance_matrix.shape[0] == 0:
        return np.empty((0, 2))
    d = np.asarray(distance_matrix, dtype=np.float64)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)
    n = d.shape[0]
    if n == 1:
        return np.zeros((1, 2), dtype=np.float64)
    d2 = d * d
    centering = np.eye(n) - np.full((n, n), 1.0 / n)
    gram = -0.5 * centering @ d2 @ centering
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order[:2]], 0.0)
    eigvecs = eigvecs[:, order[:2]]
    coords = eigvecs * np.sqrt(eigvals)[None, :]
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(n)])
    return coords[:, :2]


def plot_coords_from_pairwise_distances(run_df: pd.DataFrame, fallback_features: np.ndarray) -> np.ndarray:
    pairwise_cols = sorted(
        [
            column
            for column in run_df.columns
            if column.startswith("aggregate_pairdist_to_run_")
            and pd.api.types.is_numeric_dtype(run_df[column])
        ]
    )
    if not pairwise_cols:
        return pca_2d(fallback_features)
    distance_matrix = run_df[pairwise_cols].to_numpy(dtype=np.float64)
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        return pca_2d(fallback_features)
    return classical_mds_2d(distance_matrix)


def kmeans2(x: np.ndarray, max_iter: int = 200) -> Tuple[np.ndarray, np.ndarray]:
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
    labels = labels.astype(int)
    truth = truth.astype(int)
    pred_a = labels.copy()
    pred_b = 1 - labels
    acc_a = float((pred_a == truth).mean()) if len(truth) else np.nan
    acc_b = float((pred_b == truth).mean()) if len(truth) else np.nan
    if acc_a >= acc_b:
        return pred_a, 1, acc_a
    return pred_b, 0, acc_b


def add_unsupervised_scores(run_df: pd.DataFrame, feature_set: str = "auto") -> Tuple[pd.DataFrame, List[str]]:
    out = run_df.copy()
    feature_cols = feature_columns_for_classifier(out, feature_set=feature_set)
    if not feature_cols:
        raise ValueError("No numeric run-level features available for classifier.")
    x = out[feature_cols].to_numpy(dtype=np.float64)
    x_scaled, _, _ = robust_scale(x)
    x_scaled = np.nan_to_num(x_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    labels, centers = kmeans2(x_scaled)
    distances_to_centers = np.linalg.norm(x_scaled[:, None, :] - centers[None, :, :], axis=2)
    own_cluster_distance = distances_to_centers[np.arange(len(labels)), labels]
    nearest_other_distance = distances_to_centers[np.arange(len(labels)), 1 - labels]
    coords = plot_coords_from_pairwise_distances(out, x_scaled)
    truth = out["is_poisoned"].to_numpy(dtype=int)
    posthoc_predicted, posthoc_poison_cluster, posthoc_accuracy = best_binary_cluster_mapping(labels, truth)

    out["unsupervised_score"] = own_cluster_distance
    out["cluster_distance"] = own_cluster_distance
    out["cluster_margin"] = nearest_other_distance - own_cluster_distance
    out["cluster_id"] = labels
    out["posthoc_poison_cluster"] = posthoc_poison_cluster
    out["posthoc_predicted_poisoned"] = posthoc_predicted
    out["posthoc_cluster_accuracy"] = posthoc_accuracy
    out["predicted_anomaly_cluster"] = posthoc_predicted
    out["embedding_x"] = coords[:, 0]
    out["embedding_y"] = coords[:, 1]
    out["pca_x"] = coords[:, 0]
    out["pca_y"] = coords[:, 1]
    return out, feature_cols


def add_unsupervised_scores_by_analysis_group(
    run_df: pd.DataFrame,
    feature_set: str = "auto",
) -> Tuple[pd.DataFrame, List[str]]:
    if "analysis_group_id" not in run_df.columns:
        scored, feature_cols = add_unsupervised_scores(run_df, feature_set=feature_set)
        return scored, feature_cols

    scored_parts = []
    all_feature_cols: List[str] = []
    for analysis_group_id, sub in run_df.groupby("analysis_group_id", sort=False):
        if len(sub) < 2:
            continue
        scored, feature_cols = add_unsupervised_scores(sub.copy(), feature_set=feature_set)
        scored["analysis_group_id"] = analysis_group_id
        scored_parts.append(scored)
        all_feature_cols.extend(feature_cols)
    if not scored_parts:
        raise ValueError("No analysis group had enough runs for unsupervised scoring.")
    unique_feature_cols = list(dict.fromkeys(all_feature_cols))
    return pd.concat(scored_parts, ignore_index=True), unique_feature_cols


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
    summary["cluster_label_match_accuracy"] = float((predicted == truth).mean()) if denom else np.nan
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
    ax.set_title(f"Raw OT unsupervised score: {group_label}")
    ax.set_ylabel("distance to assigned cluster center")
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
    ax.set_title(f"Raw OT run embedding view: {group_label}")
    ax.set_xlabel("embedding x")
    ax.set_ylabel("embedding y")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_dir / "embedding_view_by_condition.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for cluster_id, sub in run_df.groupby("cluster_id", sort=True):
        ax.scatter(sub["pca_x"], sub["pca_y"], label=f"cluster {cluster_id}", s=36, alpha=0.8)
    ax.set_title(f"Raw OT unsupervised clusters: {group_label}")
    ax.set_xlabel("embedding x")
    ax.set_ylabel("embedding y")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_dir / "embedding_view_by_cluster.png", dpi=180)
    plt.close(fig)


def group_plot_coordinates(sub: pd.DataFrame, feature_cols: Sequence[str]) -> np.ndarray:
    """Plot-only 2D view of the actual k-means feature space for one group.

    The classifier runs on robust-scaled embedding features. This function uses
    the same local scaling, adds the zero central-reference vector, projects to
    2D, and shifts coordinates so the central reference appears at (0, 0).
    """
    available = [column for column in feature_cols if column in sub.columns]
    if not available:
        return sub[["embedding_x", "embedding_y"]].to_numpy(dtype=np.float64)

    x = sub[available].to_numpy(dtype=np.float64)
    x_scaled, center, scale = robust_scale(x)
    x_scaled = np.nan_to_num(x_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    ref_scaled = (np.zeros((1, len(available)), dtype=np.float64) - center) / scale
    ref_scaled = np.nan_to_num(ref_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    coords = pca_2d(np.vstack([ref_scaled, x_scaled]))
    ref_coord = coords[0]
    return coords[1:] - ref_coord


def make_trial_epoch_point_plots(
    run_df: pd.DataFrame,
    output_dir: Path,
    group_label: str,
    feature_cols: Sequence[str],
) -> None:
    """Save per-trial grids showing every epoch's classification points."""
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib_cache"))
    import matplotlib.pyplot as plt

    if "analysis_group_id" not in run_df.columns:
        return

    plot_dir = output_dir / "plots" / safe_name(group_label) / "trial_epoch_points"
    plot_dir.mkdir(parents=True, exist_ok=True)
    palette = {
        "clean": "tab:blue",
        "unlearnable_examples": "tab:red",
        "availability_shortcuts": "tab:orange",
        "random_label_flipping": "tab:green",
        "target_label_flipping": "tab:purple",
    }
    marker_by_cluster = {0: "o", 1: "s"}

    def epoch_sort_key(value: Any) -> Tuple[int, str]:
        text = str(value)
        try:
            return int(text), ""
        except ValueError:
            return 10**9, text

    for trial_id, trial_df in run_df.groupby("target_trial_id", sort=False):
        epochs = sorted(trial_df["segment_id"].astype(str).unique(), key=epoch_sort_key)
        if not epochs:
            continue
        ncols = 5
        nrows = int(np.ceil(len(epochs) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.8 * nrows), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")

        for idx, epoch_id in enumerate(epochs):
            ax = axes[idx // ncols][idx % ncols]
            sub = trial_df[trial_df["segment_id"].astype(str) == epoch_id].copy()
            coords = group_plot_coordinates(sub, feature_cols)
            sub["plot_x"] = coords[:, 0]
            sub["plot_y"] = coords[:, 1]
            ax.axis("on")
            ax.scatter([0.0], [0.0], color="black", marker="+", s=80, linewidths=1.6, label="central ref")
            for _, row in sub.iterrows():
                condition = str(row["condition"])
                cluster_id = int(row["cluster_id"])
                is_misclassified = int(row["posthoc_predicted_poisoned"]) != int(row["is_poisoned"])
                ax.scatter(
                    row["plot_x"],
                    row["plot_y"],
                    color=palette.get(condition, "tab:gray"),
                    marker=marker_by_cluster.get(cluster_id, "o"),
                    s=38,
                    alpha=0.85,
                    edgecolors="black" if is_misclassified else "none",
                    linewidths=1.1 if is_misclassified else 0.0,
                )
            accuracy = float((sub["posthoc_predicted_poisoned"].astype(int) == sub["is_poisoned"].astype(int)).mean())
            ax.set_title(f"epoch {epoch_id} acc={accuracy:.2f}", fontsize=9)
            ax.axhline(0.0, color="0.85", linewidth=0.7)
            ax.axvline(0.0, color="0.85", linewidth=0.7)
            ax.grid(True, alpha=0.2)
            ax.tick_params(labelsize=7)

        handles = [
            plt.Line2D([0], [0], marker="+", color="black", linestyle="", markersize=9, label="central ref"),
            plt.Line2D([0], [0], marker="o", color="black", linestyle="", markersize=6, label="cluster 0"),
            plt.Line2D([0], [0], marker="s", color="black", linestyle="", markersize=6, label="cluster 1"),
        ]
        for condition, color in palette.items():
            if condition in set(trial_df["condition"].astype(str)):
                handles.append(
                    plt.Line2D([0], [0], marker="o", color=color, linestyle="", markersize=6, label=condition)
                )
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                markerfacecolor="white",
                markeredgecolor="black",
                linestyle="",
                markersize=6,
                label="misclassified",
            )
        )
        fig.suptitle(f"{group_label} {trial_id}: central-reference transport embeddings", fontsize=12)
        fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)), fontsize=8)
        fig.tight_layout(rect=(0, 0.08, 1, 0.94))
        fig.savefig(plot_dir / f"{safe_name(str(trial_id))}_epoch_points.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="collected_logs")
    parser.add_argument("--output_dir", default="raw_ot_classifier_result")
    parser.add_argument(
        "--analysis_mode",
        choices=["central", "pairwise", "reference"],
        default="central",
        help=(
            "central builds one unsupervised central reference per device_type/trial/epoch; "
            "pairwise compares every run with every other run; reference uses a clean fixed reference."
        ),
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
    parser.add_argument(
        "--embedding_bins",
        type=int,
        default=128,
        help="Fixed quantile bins for no-reference c2_value_shape run embeddings.",
    )
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
        choices=["auto", "all", "embedding", "pairwise_distance_rows", "embedding_and_distance"],
        default="auto",
    )
    parser.add_argument("--max_groups", type=int, default=0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_columns = ot_base.parse_feature_columns(args.feature_columns)
    cost_types = ot_base.parse_cost_types(args.cost_types)
    groups = ot_base.discover_input_groups(input_dir)
    if args.max_groups > 0:
        groups = groups[: args.max_groups]

    all_summary = []
    all_runs = []
    group_summaries = []

    if args.analysis_mode == "central":
        if args.grouping == "device":
            grouped_dirs = [(group_label, [(group_label, group_dir)]) for group_label, group_dir in groups]
        else:
            type_to_groups: Dict[str, List[Tuple[str, Path]]] = {}
            for group_label, group_dir in groups:
                type_to_groups.setdefault(device_type_for_group_label(group_label), []).append((group_label, group_dir))
            grouped_dirs = sorted(type_to_groups.items())

        for group_label, grouped_group_dirs in grouped_dirs:
            group_features = compute_central_reference_run_features_for_pooled_group(
                group_label=str(group_label),
                grouped_dirs=grouped_group_dirs,
                args=args,
                feature_columns=feature_columns,
                cost_types=cost_types,
            )
            group_features["group_label"] = str(group_label)
            scored_df, feature_cols = add_unsupervised_scores_by_analysis_group(
                group_features,
                feature_set=args.classifier_feature_set,
            )
            group_output_dir = output_dir if len(grouped_dirs) == 1 else output_dir / safe_name(str(group_label))
            make_plots(scored_df, group_output_dir, str(group_label))
            make_trial_epoch_point_plots(scored_df, group_output_dir, str(group_label), feature_cols)
            for analysis_group_id, sub in scored_df.groupby("analysis_group_id", sort=False):
                group_summary = summarize_classifier(sub, str(analysis_group_id))
                group_summary["analysis_mode"] = "central"
                group_summary["device_group"] = str(group_label)
                group_summary["analysis_group_id"] = str(analysis_group_id)
                group_summary["target_trial_id"] = str(sub["target_trial_id"].iloc[0])
                group_summary["segment_type"] = str(sub["segment_type"].iloc[0])
                group_summary["segment_id"] = str(sub["segment_id"].iloc[0])
                group_summary["n_features"] = len(feature_cols)
                group_summary["feature_columns"] = ",".join(feature_cols)
                group_summary["n_physical_devices"] = int(sub["physical_device"].nunique())
                group_summary["physical_devices"] = ",".join(sorted(sub["physical_device"].unique()))
                group_summaries.append(group_summary)
            all_runs.append(scored_df)

    elif args.analysis_mode == "pairwise":
        if args.grouping == "device":
            grouped_dirs = [(group_label, [(group_label, group_dir)]) for group_label, group_dir in groups]
        else:
            type_to_groups: Dict[str, List[Tuple[str, Path]]] = {}
            for group_label, group_dir in groups:
                type_to_groups.setdefault(device_type_for_group_label(group_label), []).append((group_label, group_dir))
            grouped_dirs = sorted(type_to_groups.items())

        for group_label, grouped_group_dirs in grouped_dirs:
            group_features = compute_pairwise_raw_run_features_for_pooled_group(
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
    else:
        if args.grouping == "device":
            grouped_reference = [(group_label, [(group_label, group_dir)]) for group_label, group_dir in groups]
        else:
            type_to_groups: Dict[str, List[Tuple[str, Path]]] = {}
            for group_label, group_dir in groups:
                type_to_groups.setdefault(device_type_for_group_label(group_label), []).append((group_label, group_dir))
            grouped_reference = sorted(type_to_groups.items())

        for group_label, grouped_group_dirs in grouped_reference:
            typed_summary = []
            typed_run_features = []
            for source_group_label, group_dir in grouped_group_dirs:
                summary_df, run_features = compute_reference_raw_run_features(
                    group_label=source_group_label,
                    group_dir=group_dir,
                    args=args,
                    feature_columns=feature_columns,
                    cost_types=cost_types,
                )
                summary_df["group_label"] = str(group_label)
                typed_summary.append(summary_df)
                typed_run_features.append(run_features)

            combined_features = pd.concat(typed_run_features, ignore_index=True)
            combined_features["group_label"] = str(group_label)
            scored_df, feature_cols = add_unsupervised_scores(
                combined_features,
                feature_set=args.classifier_feature_set,
            )
            group_output_dir = output_dir if len(grouped_reference) == 1 else output_dir / safe_name(str(group_label))
            make_plots(scored_df, group_output_dir, str(group_label))
            group_summary = summarize_classifier(scored_df, str(group_label))
            group_summary["analysis_mode"] = "reference"
            group_summary["n_features"] = len(feature_cols)
            group_summary["feature_columns"] = ",".join(feature_cols)
            group_summary["n_physical_devices"] = int(scored_df["physical_device"].nunique())
            group_summary["physical_devices"] = ",".join(sorted(scored_df["physical_device"].unique()))
            all_summary.append(pd.concat(typed_summary, ignore_index=True))
            all_runs.append(scored_df)
            group_summaries.append(group_summary)

    summary_all = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    runs_all = pd.concat(all_runs, ignore_index=True)
    report = pd.DataFrame(group_summaries)
    summary_all.to_csv(output_dir / "raw_ot_point_summary.csv", index=False)
    runs_all.to_csv(output_dir / "raw_ot_run_level_unsupervised_scores.csv", index=False)
    report.to_csv(output_dir / "raw_ot_unsupervised_classifier_report.csv", index=False)
    aggregate_report = pd.DataFrame()
    if "device_group" in report.columns:
        aggregate_rows = []
        for device_group, sub in report.groupby("device_group", sort=False):
            tp = int(sub["tp"].sum())
            fp = int(sub["fp"].sum())
            tn = int(sub["tn"].sum())
            fn = int(sub["fn"].sum())
            total = tp + fp + tn + fn
            aggregate_rows.append(
                {
                    "device_group": device_group,
                    "n_analysis_groups": int(len(sub)),
                    "n_runs": int(sub["n_runs"].sum()),
                    "n_clean": int(sub["n_clean"].sum()),
                    "n_poisoned": int(sub["n_poisoned"].sum()),
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                    "overall_accuracy": float((tp + tn) / total) if total else np.nan,
                    "mean_group_accuracy": float(sub["cluster_label_match_accuracy"].mean()),
                    "std_group_accuracy": float(sub["cluster_label_match_accuracy"].std(ddof=0)),
                }
            )
        aggregate_report = pd.DataFrame(aggregate_rows)
        aggregate_report.to_csv(output_dir / "raw_ot_unsupervised_classifier_report_by_device_group.csv", index=False)
    print(f"Saved outputs to {output_dir}")
    if not aggregate_report.empty:
        print(aggregate_report.to_string(index=False))
    else:
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
