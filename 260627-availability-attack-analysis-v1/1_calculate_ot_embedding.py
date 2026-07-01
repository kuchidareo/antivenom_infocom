#!/usr/bin/env python3
"""Compute OT distances and Wasserstein tangent embeddings for hardware traces.

Interpretation notes:
- ot_cost measures how far a target run is from a clean reference run under a selected cost.
- c1_time captures phase/time alignment differences.
- c2_value captures resource-value distribution shifts.
- c2_value_shape captures value-distribution shape after removing each run's mean and scale.
- c3_window_abs captures local waveform differences including level, amplitude, and shape.
- c3_window_shape captures local waveform shape differences after local normalization.
- tangent embedding captures the OT-induced direction of distributional change from the clean reference to the target.
- residual_norm captures how much of that displacement is not explained by clean-to-clean shifts across trials.
- High ot_cost and high residual_norm means the target is far from the reference and moves in a direction not typical of clean trial variation.
- This does not directly prove poisoning; it provides a system-trace distributional signature associated with the target condition.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


COST_TYPES = ("c1_time", "c2_value", "c2_value_shape", "c3_window_abs", "c3_window_shape")
DEFAULT_GLOBAL_REFERENCE_TRIAL_ID = "reference_0"
LEGACY_GLOBAL_REFERENCE_TRIAL_ID = "trial_0"
DEFAULT_CLEAN_BASELINE_TRIAL_IDS = tuple(f"reference_{idx}" for idx in range(5))
LEGACY_CLEAN_BASELINE_TRIAL_IDS = tuple(f"trial_{idx}" for idx in range(5))

DEFAULT_FEATURE_COLUMNS = [
    "system_cpu_core_0",
    "system_cpu_core_1",
    "system_cpu_core_2",
    "system_cpu_core_3",
    "system_memory_percent",
    "process_ctx_switches_voluntary",
    "process_ctx_switches_involuntary",
    "process_minor_faults",
]

CPU_CORE_COLUMNS = [
    "system_cpu_core_0",
    "system_cpu_core_1",
    "system_cpu_core_2",
    "system_cpu_core_3",
]

METRIC_NAME_BY_COLUMN = {
    "system_cpu_core_0": "cpu_0",
    "system_cpu_core_1": "cpu_1",
    "system_cpu_core_2": "cpu_2",
    "system_cpu_core_3": "cpu_3",
    "system_memory_percent": "memory",
    "process_ctx_switches_voluntary": "voluntary_context",
    "process_ctx_switches_involuntary": "involuntary_context",
    "process_minor_faults": "minor_fault",
}

COUNTER_COLUMNS = {
    "process_ctx_switches_voluntary",
    "process_ctx_switches_involuntary",
    "process_minor_faults",
}

SUMMARY_COLUMNS = [
    "reference_run_id",
    "reference_trial_id",
    "target_trial_id",
    "target_run_id",
    "segment_type",
    "segment_id",
    "metric_name",
    "source_column",
    "metric_transform",
    "target_group",
    "poisoning_type",
    "cost_type",
    "ot_cost",
    "tangent_norm",
    "pca_x",
    "pca_y",
    "residual_norm",
    "residual_ratio",
    "ref_length",
    "target_length",
    "feature_dim",
    "ot_solver_cost_scale",
]

ZSCORE_COLUMNS = [
    "z_ot_cost",
    "z_tangent_norm",
    "z_residual_norm",
    "z_residual_ratio",
]


class PCANP:
    def __init__(self, n_components: int) -> None:
        self.requested_components = int(n_components)
        self.n_components_: int = 0
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "PCANP":
        if x.ndim != 2:
            raise ValueError(f"PCA input must be 2D, got shape {x.shape}")
        max_components = min(x.shape[0], x.shape[1], self.requested_components)
        self.n_components_ = max(1, max_components)
        self.mean_ = x.mean(axis=0)
        centered = x - self.mean_
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        self.components_ = vt[: self.n_components_]
        if self.components_.shape[0] < self.n_components_:
            self.n_components_ = self.components_.shape[0]
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA has not been fitted.")
        return (x - self.mean_) @ self.components_.T

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA has not been fitted.")
        return z @ self.components_ + self.mean_


@dataclass
class Target:
    target_trial_id: str
    target_run_id: str
    target_group: str
    poisoning_type: str
    run: np.ndarray


def _trial_id(value: Any) -> str:
    text = str(value)
    return text if text.startswith(("trial_", "reference_")) else f"trial_{text}"


def _numeric_trial_id(trial_id: str) -> int:
    prefix_rank = 0 if trial_id.startswith("reference_") else 1
    try:
        return prefix_rank * 10**8 + int(trial_id.split("_", 1)[1])
    except (IndexError, ValueError):
        return 10**9


def validate_data(data: Dict[str, Dict[str, Any]]) -> int:
    if not data:
        raise ValueError("No runs were loaded.")

    feature_dim: Optional[int] = None
    for trial_id, trial_data in data.items():
        if "clean" not in trial_data:
            raise ValueError(f"{trial_id} is missing a clean run.")
        poisoning = trial_data.get("poisoning", {})
        if poisoning is None:
            trial_data["poisoning"] = {}
            poisoning = {}
        if not isinstance(poisoning, dict):
            raise ValueError(f"{trial_id} poisoning field must be a dict.")

        runs = [("clean", trial_data["clean"])]
        runs.extend((f"poisoning.{name}", run) for name, run in poisoning.items())
        for label, run in runs:
            if not isinstance(run, np.ndarray):
                raise ValueError(f"{trial_id}.{label} must be a numpy array.")
            if run.ndim != 2:
                raise ValueError(f"{trial_id}.{label} must be 2D, got shape {run.shape}")
            if run.shape[0] == 0 or run.shape[1] == 0:
                raise ValueError(f"{trial_id}.{label} must be non-empty, got shape {run.shape}")
            if not np.isfinite(run).all():
                raise ValueError(f"{trial_id}.{label} contains NaN or inf values.")
            if feature_dim is None:
                feature_dim = run.shape[1]
            elif run.shape[1] != feature_dim:
                raise ValueError(
                    f"{trial_id}.{label} feature dimension {run.shape[1]} differs from expected {feature_dim}."
                )
    return int(feature_dim)


def collect_targets(data: Dict[str, Dict[str, Any]], reference_trial_id: str) -> List[Target]:
    if reference_trial_id not in data:
        raise ValueError(f"reference_trial_id={reference_trial_id!r} does not exist.")

    targets: List[Target] = []
    for trial_id in sorted(data.keys(), key=_numeric_trial_id):
        targets.append(
            Target(
                target_trial_id=trial_id,
                target_run_id=f"{trial_id}_clean",
                target_group="clean",
                poisoning_type="none",
                run=data[trial_id]["clean"],
            )
        )
        for poisoning_type, run in sorted(data[trial_id].get("poisoning", {}).items()):
            targets.append(
                Target(
                    target_trial_id=trial_id,
                    target_run_id=f"{trial_id}_{poisoning_type}",
                    target_group="poisoning",
                    poisoning_type=poisoning_type,
                    run=run,
                )
            )
    return targets


def resolve_reference_trial_ids(data: Dict[str, Dict[str, Any]], value: str) -> List[str]:
    # The OT tangent space must use one fixed global anchor. Additional clean
    # reference runs are targets used to estimate clean variation, not separate
    # OT anchors.
    if value in {"", "default", "global", "global_reference", "reference_0", "first", "first5"}:
        if DEFAULT_GLOBAL_REFERENCE_TRIAL_ID in data:
            return [DEFAULT_GLOBAL_REFERENCE_TRIAL_ID]
        if LEGACY_GLOBAL_REFERENCE_TRIAL_ID in data:
            return [LEGACY_GLOBAL_REFERENCE_TRIAL_ID]
        raise ValueError(
            "Global reference trial does not exist. Expected "
            f"{DEFAULT_GLOBAL_REFERENCE_TRIAL_ID!r}, or legacy {LEGACY_GLOBAL_REFERENCE_TRIAL_ID!r}."
        )
    if value in {"all_clean", "all"}:
        return sorted(data.keys(), key=_numeric_trial_id)
    trial_ids = [_trial_id(item.strip()) for item in value.split(",") if item.strip()]
    if not trial_ids:
        raise ValueError("At least one reference trial id is required.")
    missing = [trial_id for trial_id in trial_ids if trial_id not in data]
    if missing:
        raise ValueError(f"Reference trial ids do not exist: {missing}")
    return trial_ids


def parse_cost_types(value: str) -> List[str]:
    cost_types = [item.strip() for item in value.split(",") if item.strip()]
    if not cost_types:
        raise ValueError("At least one cost type is required.")
    unknown = [cost_type for cost_type in cost_types if cost_type not in COST_TYPES]
    if unknown:
        raise ValueError(f"Unknown cost types: {unknown}. Valid values: {COST_TYPES}")
    return cost_types


def metric_name_for_column(column: str) -> str:
    return METRIC_NAME_BY_COLUMN.get(column, column)


def metric_transform_for_column(column: str) -> str:
    if column in CPU_CORE_COLUMNS:
        return "sorted_desc"
    return "delta" if column in COUNTER_COLUMNS else "raw"


def sort_cpu_cores_per_timestamp(df: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    """Sort CPU core utilizations by load at each timestamp.

    The physical Raspberry Pi core IDs can swap relative load across time. For
    this analysis, CPU metrics are rank-based: system_cpu_core_0 is rewritten to
    the highest core utilization at that timestamp and system_cpu_core_3 to the
    lowest.
    """
    available = [column for column in CPU_CORE_COLUMNS if column in feature_columns]
    if len(available) != len(CPU_CORE_COLUMNS):
        return df
    out = df.copy()
    cpu_values = out.loc[:, CPU_CORE_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    sorted_values = np.sort(cpu_values, axis=1)[:, ::-1]
    out.loc[:, CPU_CORE_COLUMNS] = sorted_values
    return out


def apply_counter_deltas(df: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    """Convert cumulative counter columns to per-sample increments.

    Context switches and minor faults are cumulative counters in the logger.
    OT should compare the local activity trend, so those columns are replaced
    by row-to-row deltas before epoch/round segmentation.
    """
    out = df.copy()
    for column in feature_columns:
        if column not in COUNTER_COLUMNS:
            continue
        values = pd.to_numeric(out[column], errors="coerce")
        deltas = values.diff().fillna(0.0)
        deltas = deltas.mask(deltas < 0.0, 0.0)
        out[column] = deltas
    return out


def slice_data_for_metric(data: Dict[str, Dict[str, Any]], metric_index: int) -> Dict[str, Dict[str, Any]]:
    sliced: Dict[str, Dict[str, Any]] = {}
    for trial_id, trial_data in data.items():
        sliced_trial: Dict[str, Any] = {"poisoning": {}}
        if "clean" in trial_data:
            sliced_trial["clean"] = trial_data["clean"][:, [metric_index]]
        for poisoning_type, run in trial_data.get("poisoning", {}).items():
            sliced_trial["poisoning"][poisoning_type] = run[:, [metric_index]]
        sliced[trial_id] = sliced_trial
    return sliced


def deterministic_downsample_run(run: np.ndarray, max_samples: int) -> np.ndarray:
    """Keep at most max_samples rows while preserving start/end and temporal coverage.

    Full OT builds an n_ref by n_target cost matrix. For long 10 FPS traces, c3
    window costs can exceed RAM. This option is an explicit approximation for
    memory-bounded analysis; leave max_samples <= 0 to use full-resolution traces.
    """
    if max_samples <= 0 or run.shape[0] <= max_samples:
        return run
    indices = np.linspace(0, run.shape[0] - 1, max_samples)
    indices = np.unique(np.round(indices).astype(int))
    return run[indices]


def make_time_features(run: np.ndarray) -> np.ndarray:
    if run.shape[0] == 1:
        tau = np.array([0.0], dtype=np.float32)
    else:
        tau = np.linspace(0.0, 1.0, run.shape[0], dtype=np.float32)
    return tau[:, None]


def make_value_features(run: np.ndarray) -> np.ndarray:
    return np.asarray(run, dtype=np.float32)


def make_value_shape_features(run: np.ndarray) -> np.ndarray:
    x = np.asarray(run, dtype=np.float32)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    return (x - mean) / std


def make_window_features(run: np.ndarray, window_size: int = 5, z_normalize_window: bool = False) -> np.ndarray:
    h = int(window_size)
    if h < 0:
        raise ValueError("window_size must be >= 0.")
    x = np.asarray(run, dtype=np.float32)
    padded = np.pad(x, ((h, h), (0, 0)), mode="edge")
    windows = np.empty((x.shape[0], 2 * h + 1, x.shape[1]), dtype=np.float32)
    for i in range(x.shape[0]):
        windows[i] = padded[i : i + 2 * h + 1]
    flat = windows.reshape(x.shape[0], -1)
    if z_normalize_window:
        mean = flat.mean(axis=1, keepdims=True)
        std = flat.std(axis=1, keepdims=True)
        std[std < 1e-12] = 1.0
        flat = (flat - mean) / std
    return flat


def build_features_for_cost(run: np.ndarray, cost_type: str, window_size: int) -> np.ndarray:
    if cost_type == "c1_time":
        return make_time_features(run)
    if cost_type == "c2_value":
        return make_value_features(run)
    if cost_type == "c2_value_shape":
        return make_value_shape_features(run)
    if cost_type == "c3_window_abs":
        return make_window_features(run, window_size=window_size, z_normalize_window=False)
    if cost_type == "c3_window_shape":
        return make_window_features(run, window_size=window_size, z_normalize_window=True)
    raise ValueError(f"Unknown cost_type={cost_type!r}. Expected one of {COST_TYPES}.")


def compute_cost_matrix(ref_features: np.ndarray, target_features: np.ndarray) -> np.ndarray:
    ref_norm = np.sum(ref_features * ref_features, axis=1)[:, None]
    target_norm = np.sum(target_features * target_features, axis=1)[None, :]
    cost = ref_norm + target_norm - 2.0 * (ref_features @ target_features.T)
    return np.maximum(cost, 0.0).astype(np.float32, copy=False)


def normalize_cost_for_solver(cost: np.ndarray) -> Tuple[np.ndarray, float]:
    """Scale a cost matrix for stable Sinkhorn without changing raw features.

    Raw hardware counters can produce squared costs around 1e14-1e15. Passing
    those directly to Sinkhorn with a small regularization makes exp(-C/reg)
    underflow to zero, which can yield an all-zero finite transport plan. The
    transport plan is computed on a numerically scaled cost matrix, while the
    reported ot_cost is still evaluated against the original raw cost matrix.
    """
    positive = cost[np.isfinite(cost) & (cost > 0)]
    if positive.size == 0:
        return cost, 1.0
    scale = float(np.median(positive))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.mean(positive))
    if not np.isfinite(scale) or scale <= 0.0:
        return cost, 1.0
    return (cost / scale).astype(np.float32, copy=False), scale


def solve_ot(
    ref_features: np.ndarray,
    target_features: np.ndarray,
    reg: float = 1.0,
    use_sinkhorn: bool = True,
    sinkhorn_num_iter: int = 300,
    sinkhorn_stop_thr: float = 1e-6,
    normalize_solver_cost: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    try:
        import ot
    except ImportError as exc:
        raise RuntimeError("POT is required. Install with: pip install POT") from exc

    n = ref_features.shape[0]
    m = target_features.shape[0]
    a = np.full(n, 1.0 / n, dtype=np.float32)
    b = np.full(m, 1.0 / m, dtype=np.float32)
    raw_cost = compute_cost_matrix(ref_features, target_features)
    solver_cost, cost_scale = normalize_cost_for_solver(raw_cost) if normalize_solver_cost else (raw_cost, 1.0)

    if use_sinkhorn:
        try:
            plan = ot.sinkhorn(a, b, solver_cost, reg=reg, numItermax=sinkhorn_num_iter, stopThr=sinkhorn_stop_thr)
            if not np.isfinite(plan).all():
                raise FloatingPointError("Sinkhorn produced NaN or inf values.")
            plan_mass = float(plan.sum())
            if not np.isfinite(plan_mass) or plan_mass < 0.5:
                raise FloatingPointError(f"Sinkhorn produced invalid transport mass: {plan_mass}")
            return plan, raw_cost, cost_scale
        except Exception as exc:
            print(f"  Sinkhorn failed ({exc}); falling back to exact EMD.")

    plan = ot.emd(a, b, solver_cost)
    if not np.isfinite(plan).all():
        raise FloatingPointError("EMD produced NaN or inf values.")
    plan_mass = float(plan.sum())
    if not np.isfinite(plan_mass) or plan_mass < 0.5:
        raise FloatingPointError(f"EMD produced invalid transport mass: {plan_mass}")
    return plan, raw_cost, cost_scale


def compute_barycentric_embedding(
    ref_features: np.ndarray,
    target_features: np.ndarray,
    plan: np.ndarray,
) -> np.ndarray:
    row_mass = plan.sum(axis=1, keepdims=True)
    transported = (plan @ target_features) / (row_mass + 1e-12)
    displacement = transported - ref_features
    return displacement.reshape(-1)


def _build_raw_features(
    run: np.ndarray,
    cost_type: str,
    window_size: int,
) -> np.ndarray:
    return build_features_for_cost(run, cost_type=cost_type, window_size=window_size)


def compute_all_ot_embeddings(
    data: Dict[str, Dict[str, Any]],
    reference_trial_ids: Sequence[str],
    cost_types: Sequence[str] = COST_TYPES,
    window_size: int = 5,
    sinkhorn_reg: float = 1.0,
    use_sinkhorn: bool = True,
    sinkhorn_num_iter: int = 300,
    sinkhorn_stop_thr: float = 1e-6,
    normalize_solver_cost: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str, str], np.ndarray]]:
    feature_dim = validate_data(data)
    if not reference_trial_ids:
        raise ValueError("At least one reference trial id is required.")
    missing_refs = [trial_id for trial_id in reference_trial_ids if trial_id not in data]
    if missing_refs:
        raise ValueError(f"Reference trial ids do not exist: {missing_refs}")
    targets = collect_targets(data, reference_trial_ids[0])

    rows: List[Dict[str, Any]] = []
    embeddings: Dict[Tuple[str, str, str], np.ndarray] = {}

    for reference_trial_id in reference_trial_ids:
        reference_run = data[reference_trial_id]["clean"]
        reference_run_id = f"{reference_trial_id}_clean"
        print(f"reference_trial_id={reference_trial_id}")
        for cost_type in cost_types:
            print(f"  cost_type={cost_type}")
            ref_features = _build_raw_features(reference_run, cost_type, window_size)
            for target in targets:
                print(f"    target_trial_id={target.target_trial_id} target_run_id={target.target_run_id}")
                target_features = _build_raw_features(target.run, cost_type, window_size)
                plan, cost, cost_scale = solve_ot(
                    ref_features,
                    target_features,
                    reg=sinkhorn_reg,
                    use_sinkhorn=use_sinkhorn,
                    sinkhorn_num_iter=sinkhorn_num_iter,
                    sinkhorn_stop_thr=sinkhorn_stop_thr,
                    normalize_solver_cost=normalize_solver_cost,
                )
                embedding = compute_barycentric_embedding(ref_features, target_features, plan)
                ot_cost = float(np.sum(plan * cost))
                key = (reference_run_id, cost_type, target.target_run_id)
                embeddings[key] = embedding
                rows.append(
                    {
                        "reference_run_id": reference_run_id,
                        "reference_trial_id": reference_trial_id,
                        "target_trial_id": target.target_trial_id,
                        "target_run_id": target.target_run_id,
                        "target_group": target.target_group,
                        "poisoning_type": target.poisoning_type,
                        "cost_type": cost_type,
                        "ot_cost": ot_cost,
                        "tangent_norm": float(np.linalg.norm(embedding)),
                        "ref_length": int(reference_run.shape[0]),
                        "target_length": int(target.run.shape[0]),
                        "feature_dim": int(feature_dim),
                        "ot_solver_cost_scale": float(cost_scale),
                    }
                )
    return rows, embeddings


def fit_clean_pcas(
    summary_rows: Sequence[Dict[str, Any]],
    embeddings: Dict[Tuple[str, str, str], np.ndarray],
    pca_components_for_residual: int = 3,
) -> Dict[Tuple[str, str], Dict[str, PCANP]]:
    pcas: Dict[Tuple[str, str], Dict[str, PCANP]] = {}
    scopes = sorted({(row["reference_run_id"], row["cost_type"]) for row in summary_rows})
    for reference_run_id, cost_type in scopes:
        clean_ids = [
            row["target_run_id"]
            for row in summary_rows
            if row["reference_run_id"] == reference_run_id
            and row["cost_type"] == cost_type
            and row["target_group"] == "clean"
        ]
        if not clean_ids:
            raise ValueError(f"No clean embeddings found for {reference_run_id}, {cost_type}.")
        x = np.vstack([embeddings[(reference_run_id, cost_type, run_id)] for run_id in clean_ids])
        pca_2d = PCANP(n_components=min(2, x.shape[0], x.shape[1])).fit(x)
        pca_residual = PCANP(n_components=min(pca_components_for_residual, x.shape[0], x.shape[1])).fit(x)
        pcas[(reference_run_id, cost_type)] = {"pca_2d": pca_2d, "pca_residual": pca_residual}
    return pcas


def compute_residual_scores(
    summary_rows: List[Dict[str, Any]],
    embeddings: Dict[Tuple[str, str, str], np.ndarray],
    pcas: Dict[Tuple[str, str], Dict[str, PCANP]],
) -> None:
    for row in summary_rows:
        reference_run_id = row["reference_run_id"]
        cost_type = row["cost_type"]
        embedding = embeddings[(reference_run_id, cost_type, row["target_run_id"])][None, :]
        pca_2d = pcas[(reference_run_id, cost_type)]["pca_2d"]
        coords = pca_2d.transform(embedding).reshape(-1)
        row["pca_x"] = float(coords[0]) if coords.size > 0 else 0.0
        row["pca_y"] = float(coords[1]) if coords.size > 1 else 0.0

        pca_residual = pcas[(reference_run_id, cost_type)]["pca_residual"]
        embedding_hat = pca_residual.inverse_transform(pca_residual.transform(embedding))
        residual = embedding - embedding_hat
        tangent_norm = float(np.linalg.norm(embedding))
        residual_norm = float(np.linalg.norm(residual))
        row["tangent_norm"] = tangent_norm
        row["residual_norm"] = residual_norm
        row["residual_ratio"] = residual_norm / (tangent_norm + 1e-12)


def build_summary_dataframe(summary_rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(summary_rows)
    for column in SUMMARY_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
    return df[SUMMARY_COLUMNS].copy()


def add_clean_zscores(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    zdf = df.copy()
    baseline_rows: List[Dict[str, Any]] = []
    metric_pairs = [
        ("ot_cost", "z_ot_cost"),
        ("tangent_norm", "z_tangent_norm"),
        ("residual_norm", "z_residual_norm"),
        ("residual_ratio", "z_residual_ratio"),
    ]
    group_cols = ["cost_type"]
    if "metric_name" in df.columns:
        group_cols.insert(0, "metric_name")
    if {"segment_type", "segment_id"}.issubset(df.columns):
        group_cols.extend(["segment_type", "segment_id"])
    for group_key, sub in df.groupby(group_cols, sort=False):
        if isinstance(group_key, tuple):
            group_values = dict(zip(group_cols, group_key))
            metric_name = group_values.get("metric_name", "")
            cost_type = group_values["cost_type"]
            segment_type = group_values.get("segment_type", "")
            segment_id = group_values.get("segment_id", "")
        else:
            metric_name = ""
            cost_type = group_key
            segment_type = ""
            segment_id = ""
        clean = sub[
            (sub["target_group"] == "clean")
            & (sub["reference_trial_id"].astype(str) != sub["target_trial_id"].astype(str))
        ]
        if clean.empty:
            clean = sub[sub["target_group"] == "clean"]
        stats: Dict[str, Any] = {
            "metric_name": metric_name,
            "cost_type": cost_type,
            "segment_type": segment_type,
            "segment_id": segment_id,
            "n_clean_targets": int(len(clean)),
        }
        for raw_col, z_col in metric_pairs:
            mean_value = float(clean[raw_col].mean())
            std_value = float(clean[raw_col].std(ddof=0))
            zdf.loc[sub.index, z_col] = (sub[raw_col] - mean_value) / (std_value + 1e-12)
            stats[f"mean_{raw_col}"] = mean_value
            stats[f"std_{raw_col}"] = std_value
        baseline_rows.append(stats)
    baseline = pd.DataFrame(baseline_rows)[
        [
            "metric_name",
            "cost_type",
            "segment_type",
            "segment_id",
            "mean_ot_cost",
            "std_ot_cost",
            "mean_tangent_norm",
            "std_tangent_norm",
            "mean_residual_norm",
            "std_residual_norm",
            "mean_residual_ratio",
            "std_residual_ratio",
            "n_clean_targets",
        ]
    ]
    return zdf[SUMMARY_COLUMNS + ZSCORE_COLUMNS], baseline


def build_tangent_embeddings_dataframe(
    summary_rows: Sequence[Dict[str, Any]],
    embeddings: Dict[Tuple[str, str, str], np.ndarray],
) -> pd.DataFrame:
    max_dim = max(len(embedding) for embedding in embeddings.values())
    meta_cols = [
        "reference_run_id",
        "reference_trial_id",
        "target_trial_id",
        "target_run_id",
        "target_group",
        "poisoning_type",
        "cost_type",
    ]
    rows = []
    for row in summary_rows:
        emb = embeddings[(row["reference_run_id"], row["cost_type"], row["target_run_id"])]
        out = {col: row[col] for col in meta_cols}
        for idx in range(max_dim):
            out[f"emb_{idx}"] = emb[idx] if idx < len(emb) else ""
        rows.append(out)
    return pd.DataFrame(rows)


def _plot_color(row: pd.Series) -> str:
    if row["target_group"] == "clean":
        return "tab:blue"
    palette = {
        "unlearnable_examples": "tab:red",
        "random_label_flipping": "tab:green",
        "target_label_flipping": "tab:purple",
        "availability_shortcuts": "tab:orange",
    }
    return palette.get(str(row["poisoning_type"]), "tab:red")


def make_plots(df: pd.DataFrame, output_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib_cache"))
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plots. Install with: pip install matplotlib") from exc

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_specs = [
        ("pca_x", "pca_y", "pca_scatter_{cost_type}.png", "PCA x", "PCA y"),
        ("ot_cost", "residual_norm", "ot_vs_residual_{cost_type}.png", "OT cost", "Residual norm"),
        ("z_ot_cost", "z_residual_norm", "z_ot_vs_z_residual_{cost_type}.png", "Z OT cost", "Z residual norm"),
        (
            "z_ot_cost",
            "z_residual_ratio",
            "z_ot_vs_z_residual_ratio_{cost_type}.png",
            "Z OT cost",
            "Z residual ratio",
        ),
    ]

    for cost_type, sub in df.groupby("cost_type", sort=False):
        for x_col, y_col, filename_template, xlabel, ylabel in plot_specs:
            fig, ax = plt.subplots(figsize=(7, 5))
            for _, row in sub.iterrows():
                label = "clean" if row["target_group"] == "clean" else row["poisoning_type"]
                ax.scatter(row[x_col], row[y_col], color=_plot_color(row), label=label)
                ax.annotate(str(row["target_trial_id"]), (row[x_col], row[y_col]), fontsize=8)
            handles, labels = ax.get_legend_handles_labels()
            dedup = dict(zip(labels, handles))
            ax.legend(dedup.values(), dedup.keys(), loc="best")
            ax.set_title(cost_type)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(plot_dir / filename_template.format(cost_type=cost_type), dpi=180)
            plt.close(fig)


def save_outputs(
    output_dir: Path,
    summary_df: pd.DataFrame,
    zscore_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / "ot_embedding_summary.csv", index=False)
    zscore_df.to_csv(output_dir / "ot_embedding_summary_zscored.csv", index=False)
    baseline_df.to_csv(output_dir / "clean_baseline_stats.csv", index=False)
    make_plots(zscore_df, output_dir)


def _segment_column_for_csv(path: Path, df: pd.DataFrame, segment_by: str) -> Optional[str]:
    mode = segment_by.lower()
    if mode == "none":
        return None
    if mode in {"epoch", "round"}:
        if mode not in df.columns:
            raise ValueError(f"{path} is missing requested segment column: {mode}")
        return mode
    if mode != "auto":
        raise ValueError("--segment_by must be one of: auto, epoch, round, none")

    run_type = str(df.iloc[0].get("run_type", "")).lower() if not df.empty else ""
    path_text = str(path).lower()
    if "local_ml" in run_type or "/local_ml/" in path_text:
        if "epoch" not in df.columns:
            raise ValueError(f"{path} looks like local_ml but is missing epoch column.")
        return "epoch"
    if "round" in df.columns:
        return "round"
    return None


def _read_hardware_csv(path: Path, feature_columns: Sequence[str]) -> Tuple[np.ndarray, Dict[str, str]]:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} is empty.")
    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing feature columns: {missing}")
    df = sort_cpu_cores_per_timestamp(df, feature_columns)
    df = apply_counter_deltas(df, feature_columns)
    values = df.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        bad = np.argwhere(~np.isfinite(values))
        raise ValueError(f"{path} contains NaN/inf in feature data; first bad index={bad[0].tolist()}")
    first = {column: str(df.iloc[0].get(column, "")) for column in df.columns}
    return values, first


def _read_hardware_csv_segments(
    path: Path,
    feature_columns: Sequence[str],
    segment_by: str,
    max_samples_per_run: int,
) -> List[Tuple[str, str, np.ndarray, Dict[str, str], int]]:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} is empty.")
    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing feature columns: {missing}")
    df = sort_cpu_cores_per_timestamp(df, feature_columns)
    df = apply_counter_deltas(df, feature_columns)

    segment_column = _segment_column_for_csv(path, df, segment_by)
    if segment_column is None:
        groups = [("full_run", "all", df)]
    else:
        numeric_segment = pd.to_numeric(df[segment_column], errors="coerce")
        if numeric_segment.notna().sum() == 0:
            raise ValueError(f"{path} has no valid numeric values in segment column {segment_column}.")
        working = df.loc[numeric_segment.notna()].copy()
        working["_analysis_segment"] = numeric_segment.loc[numeric_segment.notna()].astype(int).to_numpy()
        groups = [
            (segment_column, str(int(segment_id)), group.drop(columns=["_analysis_segment"]))
            for segment_id, group in working.groupby("_analysis_segment", sort=True)
        ]

    out: List[Tuple[str, str, np.ndarray, Dict[str, str], int]] = []
    for segment_type, segment_id, segment_df in groups:
        values = segment_df.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            bad = np.argwhere(~np.isfinite(values))
            raise ValueError(f"{path} contains NaN/inf in feature data; first bad index={bad[0].tolist()}")
        original_length = values.shape[0]
        values = deterministic_downsample_run(values, max_samples_per_run)
        first = {column: str(segment_df.iloc[0].get(column, "")) for column in segment_df.columns}
        first["analysis_segment_type"] = segment_type
        first["analysis_segment_id"] = segment_id
        out.append((segment_type, segment_id, values, first, original_length))
    return out


def _is_hardware_csv(path: Path) -> bool:
    return (
        path.suffix == ".csv"
        and not path.name.endswith("_metrics.csv")
        and path.name not in {
            "ot_embedding_summary.csv",
            "ot_embedding_summary_zscored.csv",
            "tangent_embeddings.csv",
            "clean_baseline_stats.csv",
        }
    )


def _csv_paths_under(input_dir: Path) -> List[Path]:
    return sorted(p for p in input_dir.rglob("*.csv") if _is_hardware_csv(p))


def discover_input_groups(input_dir: Path) -> List[Tuple[str, Path]]:
    """Return independent analysis groups for an input path.

    Passing a single device directory such as ``collected_logs/192.168.0.112/local_ml``
    is treated as one group. Passing the whole ``collected_logs`` directory is split
    into one local-ML group per device, because FL logs contain many clients per
    trial and should not be merged with local ML clean/poisoned runs.
    """
    if not input_dir.exists():
        raise ValueError(f"input_dir does not exist: {input_dir}")

    if input_dir.name == "local_ml":
        return [(input_dir.parent.name, input_dir)]

    local_ml_dirs = sorted(
        path for path in input_dir.rglob("local_ml")
        if path.is_dir() and _csv_paths_under(path)
    )
    if local_ml_dirs:
        groups = []
        for path in local_ml_dirs:
            try:
                label = str(path.relative_to(input_dir))
            except ValueError:
                label = path.name
            groups.append((label, path))
        return groups

    return [(input_dir.name, input_dir)]


def load_data_from_csv_paths(
    csv_paths: Sequence[Path],
    feature_columns: Sequence[str],
    max_samples_per_run: int = 0,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, str]]]:
    if not csv_paths:
        raise ValueError("No hardware CSV files found.")

    data: Dict[str, Dict[str, Any]] = {}
    metadata: Dict[str, Dict[str, str]] = {}
    seen: Dict[Tuple[str, str], Path] = {}

    for path in csv_paths:
        run, first = _read_hardware_csv(path, feature_columns)
        original_length = run.shape[0]
        run = deterministic_downsample_run(run, max_samples_per_run)
        trial_id = _trial_id(first.get("trial_id", ""))
        poisoning_type = str(first.get("poisoning_method", "") or "clean")
        key = (trial_id, poisoning_type)
        if key in seen:
            raise ValueError(
                "Duplicate run for "
                f"trial={trial_id} poisoning_method={poisoning_type}: {seen[key]} and {path}. "
                "This group still contains duplicate local runs."
            )
        seen[key] = path
        data.setdefault(trial_id, {"poisoning": {}})
        if poisoning_type == "clean":
            data[trial_id]["clean"] = run
            run_id = f"{trial_id}_clean"
        else:
            data[trial_id]["poisoning"][poisoning_type] = run
            run_id = f"{trial_id}_{poisoning_type}"
        metadata[run_id] = {
            **first,
            "source_path": str(path),
            "original_length": str(original_length),
            "analysis_length": str(run.shape[0]),
        }

    validate_data(data)
    return data, metadata


def load_segmented_data_from_csv_paths(
    csv_paths: Sequence[Path],
    feature_columns: Sequence[str],
    segment_by: str,
    max_samples_per_run: int = 0,
) -> Tuple[Dict[Tuple[str, str], Dict[str, Dict[str, Any]]], Dict[str, Dict[str, str]]]:
    if not csv_paths:
        raise ValueError("No hardware CSV files found.")

    segmented_data: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
    metadata: Dict[str, Dict[str, str]] = {}
    seen: Dict[Tuple[str, str, str, str], Path] = {}

    for path in csv_paths:
        for segment_type, segment_id, run, first, original_length in _read_hardware_csv_segments(
            path=path,
            feature_columns=feature_columns,
            segment_by=segment_by,
            max_samples_per_run=max_samples_per_run,
        ):
            trial_id = _trial_id(first.get("trial_id", ""))
            poisoning_type = str(first.get("poisoning_method", "") or "clean")
            segment_key = (segment_type, segment_id)
            seen_key = (segment_type, segment_id, trial_id, poisoning_type)
            if seen_key in seen:
                raise ValueError(
                    "Duplicate segment for "
                    f"segment={segment_key} trial={trial_id} poisoning_method={poisoning_type}: "
                    f"{seen[seen_key]} and {path}."
                )
            seen[seen_key] = path

            data = segmented_data.setdefault(segment_key, {})
            data.setdefault(trial_id, {"poisoning": {}})
            if poisoning_type == "clean":
                data[trial_id]["clean"] = run
                run_id = f"{trial_id}_clean"
            else:
                data[trial_id]["poisoning"][poisoning_type] = run
                run_id = f"{trial_id}_{poisoning_type}"
            metadata[f"{segment_type}_{segment_id}_{run_id}"] = {
                **first,
                "source_path": str(path),
                "original_length": str(original_length),
                "analysis_length": str(run.shape[0]),
            }

    for segment_key, data in segmented_data.items():
        try:
            validate_data(data)
        except ValueError as exc:
            raise ValueError(f"Invalid data in segment {segment_key}: {exc}") from exc
    return segmented_data, metadata


def load_data_from_csv_dir(
    input_dir: Path,
    feature_columns: Sequence[str],
    max_samples_per_run: int = 0,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, str]]]:
    return load_data_from_csv_paths(_csv_paths_under(input_dir), feature_columns, max_samples_per_run)


def load_segmented_data_from_csv_dir(
    input_dir: Path,
    feature_columns: Sequence[str],
    segment_by: str,
    max_samples_per_run: int = 0,
) -> Tuple[Dict[Tuple[str, str], Dict[str, Dict[str, Any]]], Dict[str, Dict[str, str]]]:
    return load_segmented_data_from_csv_paths(
        _csv_paths_under(input_dir),
        feature_columns,
        segment_by,
        max_samples_per_run,
    )


def parse_feature_columns(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def run_one_group(
    *,
    input_dir: Path,
    output_dir: Path,
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
) -> None:
    segmented_data, _metadata = load_segmented_data_from_csv_dir(
        input_dir=input_dir,
        feature_columns=feature_columns,
        segment_by=segment_by,
        max_samples_per_run=max_samples_per_run,
    )
    if max_samples_per_run > 0:
        print(f"Using deterministic downsampling within each segment: max_samples_per_run={max_samples_per_run}")

    all_summary_rows: List[Dict[str, Any]] = []
    for segment_type, segment_id in sorted(segmented_data.keys(), key=lambda item: (item[0], int(item[1]) if item[1].isdigit() else item[1])):
        print(f"segment_type={segment_type} segment_id={segment_id}")
        segment_data = segmented_data[(segment_type, segment_id)]
        for metric_index, source_column in enumerate(feature_columns):
            metric_name = metric_name_for_column(source_column)
            metric_transform = metric_transform_for_column(source_column)
            data = slice_data_for_metric(segment_data, metric_index)
            reference_trial_ids = resolve_reference_trial_ids(data, reference_trial_ids_value)
            print(
                f"metric_name={metric_name} source_column={source_column} "
                f"metric_transform={metric_transform}"
            )
            print(f"Using global clean reference trial: {reference_trial_ids[0]}")
            summary_rows, embeddings = compute_all_ot_embeddings(
                data=data,
                reference_trial_ids=reference_trial_ids,
                cost_types=cost_types,
                window_size=window_size,
                sinkhorn_reg=sinkhorn_reg,
                use_sinkhorn=use_sinkhorn,
                sinkhorn_num_iter=sinkhorn_num_iter,
                sinkhorn_stop_thr=sinkhorn_stop_thr,
                normalize_solver_cost=normalize_solver_cost,
            )
            pcas = fit_clean_pcas(
                summary_rows=summary_rows,
                embeddings=embeddings,
                pca_components_for_residual=pca_components_for_residual,
            )
            compute_residual_scores(summary_rows, embeddings, pcas)
            for row in summary_rows:
                row["segment_type"] = segment_type
                row["segment_id"] = segment_id
                row["metric_name"] = metric_name
                row["source_column"] = source_column
                row["metric_transform"] = metric_transform
            all_summary_rows.extend(summary_rows)

    summary_df = build_summary_dataframe(all_summary_rows)
    zscore_df, baseline_df = add_clean_zscores(summary_df)
    save_outputs(output_dir, summary_df, zscore_df, baseline_df)
    print(f"Saved outputs to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
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
    parser.add_argument("--cost_types", default=",".join(COST_TYPES))
    parser.add_argument(
        "--segment_by",
        default="auto",
        choices=["auto", "epoch", "round", "none"],
        help="Analysis unit. auto uses epoch for local_ml logs and round for FL logs.",
    )
    parser.add_argument(
        "--max_samples_per_run",
        type=int,
        default=0,
        help="Optional deterministic per-run sample cap. 0 keeps full-resolution traces.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    feature_columns = parse_feature_columns(args.feature_columns)
    cost_types = parse_cost_types(args.cost_types)
    reference_trial_ids_value = args.reference_trial_id or args.reference_trial_ids

    groups = discover_input_groups(input_dir)
    if len(groups) > 1:
        print(f"Discovered {len(groups)} local_ml analysis groups under {input_dir}")
    for group_label, group_dir in groups:
        group_output_dir = output_dir if len(groups) == 1 else output_dir / group_label
        print(f"Running OT analysis for group={group_label} input={group_dir}")
        run_one_group(
            input_dir=group_dir,
            output_dir=group_output_dir,
            feature_columns=feature_columns,
            cost_types=cost_types,
            reference_trial_ids_value=reference_trial_ids_value,
            window_size=args.window_size,
            sinkhorn_reg=args.sinkhorn_reg,
            use_sinkhorn=args.use_sinkhorn,
            sinkhorn_num_iter=args.sinkhorn_num_iter,
            sinkhorn_stop_thr=args.sinkhorn_stop_thr,
            normalize_solver_cost=args.normalize_solver_cost,
            pca_components_for_residual=args.pca_components_for_residual,
            max_samples_per_run=args.max_samples_per_run,
            segment_by=args.segment_by,
        )


if __name__ == "__main__":
    main()
