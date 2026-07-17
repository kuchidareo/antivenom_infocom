#!/usr/bin/env python3
"""Heterogeneous-device pooled-reference OT/CUSUM classifier.

This version does not reuse ``result/`` because those residuals were computed
with a homogeneous per-device reference. Instead, it loads raw local_ml logs,
builds a pooled reference from reference_0..reference_4 across all devices, and
then computes OT residuals against that pooled reference.

Default focus:
- phase: backward
- cost: c3_window_shape
- metrics: CPU rank cores 1, 2, 3
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_ANALYSIS_PHASE = "backward"
DEFAULT_COST_TYPES = "c3_window_shape"
DEFAULT_LOAD_FEATURE_COLUMNS = (
    "system_cpu_core_0,system_cpu_core_1,system_cpu_core_2,system_cpu_core_3"
)
DEFAULT_ANALYSIS_FEATURE_COLUMNS = (
    "system_cpu_core_1,system_cpu_core_2,system_cpu_core_3"
)
DEFAULT_REFERENCE_TRIAL_IDS = tuple(f"reference_{idx}" for idx in range(5))

DEVICE_TYPE_BY_ID = {
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


@dataclass
class RunRecord:
    device_id: str
    device_type: str
    analysis_group: str
    segment_type: str
    segment_id: str
    analysis_phase: str
    trial_id: str
    run_id: str
    target_group: str
    poisoning_type: str
    values: np.ndarray


def load_ot_module(script_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("ot_embedding_module_for_hetero", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import OT module: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def numeric_segment_sort_key(value: Any) -> Tuple[int, Any]:
    text = str(value)
    try:
        return (0, int(float(text)))
    except ValueError:
        return (1, text)


def trial_sort_key(value: str) -> Tuple[int, int, str]:
    if value.startswith("reference_"):
        prefix = 0
    elif value.startswith("trial_"):
        prefix = 1
    else:
        prefix = 2
    try:
        number = int(value.split("_", 1)[1])
    except (IndexError, ValueError):
        number = 10**9
    return (prefix, number, value)


def device_id_from_group(group_label: str, group_dir: Path) -> str:
    parts = str(group_label).replace("\\", "/").split("/")
    return parts[0] if parts and parts[0] else group_dir.parent.name


def device_type_from_id(device_id: str) -> str:
    return DEVICE_TYPE_BY_ID.get(str(device_id), "unknown")


def condition_label(target_group: str, poisoning_type: str) -> str:
    return "clean" if target_group == "clean" else poisoning_type


def discover_device_groups(ot_module: Any, input_dir: Path) -> List[Tuple[str, Path]]:
    groups = [
        (label, path)
        for label, path in ot_module.discover_input_groups(input_dir)
        if path.name == "local_ml" or str(label).endswith("local_ml")
    ]
    groups = sorted(groups, key=lambda item: str(item[0]))
    if not groups:
        raise ValueError(f"No local_ml groups found under {input_dir}")
    return groups


def collect_records(
    *,
    ot_module: Any,
    input_dir: Path,
    load_feature_columns: Sequence[str],
    analysis_feature_columns: Sequence[str],
    analysis_phase: str,
    segment_by: str,
    max_samples_per_run: int,
) -> Tuple[List[RunRecord], List[str]]:
    missing = [col for col in analysis_feature_columns if col not in load_feature_columns]
    if missing:
        raise ValueError(f"analysis columns must be included in load columns; missing={missing}")
    analysis_indices = [load_feature_columns.index(col) for col in analysis_feature_columns]

    records: List[RunRecord] = []
    groups = discover_device_groups(ot_module, input_dir)
    print(f"Loading {len(groups)} local_ml groups from {input_dir}")
    for group_label, group_dir in groups:
        device_id = device_id_from_group(group_label, group_dir)
        device_type = device_type_from_id(device_id)
        print(f"  loading {device_id} ({device_type})")
        segmented_data, _metadata = ot_module.load_segmented_data_from_csv_dir(
            input_dir=group_dir,
            feature_columns=load_feature_columns,
            segment_by=segment_by,
            max_samples_per_run=max_samples_per_run,
        )
        segment_keys = sorted(
            [key for key in segmented_data if str(key[2]) == analysis_phase],
            key=lambda key: (key[0], numeric_segment_sort_key(key[1]), key[2]),
        )
        for segment_type, segment_id, phase in segment_keys:
            data = segmented_data[(segment_type, segment_id, phase)]
            for trial_id, trial_data in sorted(data.items(), key=lambda item: trial_sort_key(str(item[0]))):
                if "clean" in trial_data:
                    records.append(
                        RunRecord(
                            device_id=device_id,
                            device_type=device_type,
                            analysis_group=str(group_label),
                            segment_type=segment_type,
                            segment_id=str(segment_id),
                            analysis_phase=str(phase),
                            trial_id=str(trial_id),
                            run_id=f"{device_id}__{trial_id}_clean",
                            target_group="clean",
                            poisoning_type="none",
                            values=trial_data["clean"][:, analysis_indices].astype(np.float32, copy=False),
                        )
                    )
                for poisoning_type, run in sorted(trial_data.get("poisoning", {}).items()):
                    records.append(
                        RunRecord(
                            device_id=device_id,
                            device_type=device_type,
                            analysis_group=str(group_label),
                            segment_type=segment_type,
                            segment_id=str(segment_id),
                            analysis_phase=str(phase),
                            trial_id=str(trial_id),
                            run_id=f"{device_id}__{trial_id}_{poisoning_type}",
                            target_group="poisoning",
                            poisoning_type=str(poisoning_type),
                            values=run[:, analysis_indices].astype(np.float32, copy=False),
                        )
                    )
    if not records:
        raise ValueError(f"No records collected for phase={analysis_phase}")
    return records, list(analysis_feature_columns)


def common_ordering_vector(feature_clouds: Sequence[np.ndarray]) -> np.ndarray:
    dim = int(feature_clouds[0].shape[1])
    if dim == 1:
        return np.ones(1, dtype=np.float32)
    stacked = np.vstack(feature_clouds).astype(np.float32, copy=False)
    if stacked.shape[0] > 50000:
        idx = np.linspace(0, stacked.shape[0] - 1, 50000)
        stacked = stacked[np.round(idx).astype(int)]
    centered = stacked - stacked.mean(axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        vec = vt[0].astype(np.float32, copy=False)
    except np.linalg.LinAlgError:
        vec = np.zeros(dim, dtype=np.float32)
        vec[0] = 1.0
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm < 1e-12:
        vec = np.zeros(dim, dtype=np.float32)
        vec[0] = 1.0
    else:
        vec = vec / norm
    return vec


def fixed_support(features: np.ndarray, support_bins: int, ordering_vector: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    projection = features[:, 0] if features.shape[1] == 1 else features @ ordering_vector
    ordered = features[np.argsort(projection, kind="mergesort")]
    n = ordered.shape[0]
    out = np.empty((support_bins, ordered.shape[1]), dtype=np.float32)
    edges = np.linspace(0, n, support_bins + 1)
    for idx in range(support_bins):
        start = int(math.floor(edges[idx]))
        end = int(math.floor(edges[idx + 1]))
        if end <= start:
            nearest = min(max(int(round((edges[idx] + edges[idx + 1]) / 2.0)), 0), n - 1)
            out[idx] = ordered[nearest]
        else:
            out[idx] = ordered[start:end].mean(axis=0)
    return out


def barycentric_embedding(source: np.ndarray, target: np.ndarray, plan: np.ndarray) -> np.ndarray:
    row_mass = plan.sum(axis=1, keepdims=True)
    transported = (plan @ target) / (row_mass + 1e-12)
    return (row_mass * (transported - source)).reshape(-1)


def compute_residuals(
    *,
    records: Sequence[RunRecord],
    analysis_feature_columns: Sequence[str],
    cost_types: Sequence[str],
    reference_trial_ids: Sequence[str],
    support_bins: int,
    window_size: int,
    sinkhorn_reg: float,
    sinkhorn_num_iter: int,
    sinkhorn_stop_thr: float,
    use_sinkhorn: bool,
    normalize_solver_cost: bool,
    pca_components_for_residual: int,
    ot_module: Any,
) -> pd.DataFrame:
    grouped: Dict[Tuple[str, str], List[RunRecord]] = {}
    for record in records:
        grouped.setdefault((record.segment_type, record.segment_id), []).append(record)

    rows_all: List[Dict[str, Any]] = []
    for (segment_type, segment_id), segment_records in sorted(
        grouped.items(), key=lambda item: (item[0][0], numeric_segment_sort_key(item[0][1]))
    ):
        print(f"segment={segment_type}:{segment_id} records={len(segment_records)}")
        for metric_idx, source_column in enumerate(analysis_feature_columns):
            metric_name = ot_module.metric_name_for_column(source_column)
            metric_records = [
                RunRecord(**{**record.__dict__, "values": record.values[:, [metric_idx]]})
                for record in segment_records
            ]
            for cost_type in cost_types:
                print(f"  metric={metric_name} cost={cost_type}")
                features_by_run: Dict[str, np.ndarray] = {}
                reference_features = []
                reference_run_ids = []
                for record in metric_records:
                    features = ot_module.build_features_for_cost(record.values, cost_type, window_size)
                    features_by_run[record.run_id] = features
                    if record.target_group == "clean" and record.trial_id in reference_trial_ids:
                        reference_features.append(features)
                        reference_run_ids.append(record.run_id)
                if not reference_features:
                    raise ValueError(f"No pooled reference rows for segment={segment_id} metric={metric_name}")

                ordering_vector = common_ordering_vector(reference_features)
                supports = {
                    run_id: fixed_support(features, support_bins, ordering_vector)
                    for run_id, features in features_by_run.items()
                }
                central_reference = np.stack([supports[run_id] for run_id in reference_run_ids], axis=0).mean(axis=0)

                embeddings: Dict[str, np.ndarray] = {}
                rows: List[Dict[str, Any]] = []
                for record in metric_records:
                    target_support = supports[record.run_id]
                    plan, cost, cost_scale = ot_module.solve_ot(
                        central_reference,
                        target_support,
                        reg=sinkhorn_reg,
                        use_sinkhorn=use_sinkhorn,
                        sinkhorn_num_iter=sinkhorn_num_iter,
                        sinkhorn_stop_thr=sinkhorn_stop_thr,
                        normalize_solver_cost=normalize_solver_cost,
                    )
                    embedding = barycentric_embedding(central_reference, target_support, plan)
                    embeddings[record.run_id] = embedding
                    rows.append(
                        {
                            "device_id": record.device_id,
                            "device_type": record.device_type,
                            "analysis_group": record.analysis_group,
                            "global_run_id": record.run_id,
                            "target_run_id": record.run_id.split("__", 1)[1],
                            "target_trial_id": record.trial_id,
                            "target_group": record.target_group,
                            "poisoning_type": record.poisoning_type,
                            "condition_label": condition_label(record.target_group, record.poisoning_type),
                            "is_reference_baseline": bool(
                                record.target_group == "clean"
                                and record.trial_id in reference_trial_ids
                            ),
                            "segment_type": segment_type,
                            "segment_id": segment_id,
                            "analysis_phase": record.analysis_phase,
                            "metric_name": metric_name,
                            "source_column": source_column,
                            "cost_type": cost_type,
                            "ot_cost": float(np.sum(plan * cost)),
                            "ot_solver_cost_scale": float(cost_scale),
                            "support_bins": int(support_bins),
                            "reference_pool_size": int(len(reference_run_ids)),
                            "reference_device_types": ",".join(sorted({r.device_type for r in metric_records if r.run_id in reference_run_ids})),
                        }
                    )

                clean_x = np.vstack([embeddings[run_id] for run_id in reference_run_ids])
                pca_2d = ot_module.PCANP(n_components=min(2, clean_x.shape[0], clean_x.shape[1])).fit(clean_x)
                pca_resid = ot_module.PCANP(
                    n_components=min(pca_components_for_residual, clean_x.shape[0], clean_x.shape[1])
                ).fit(clean_x)

                ref_residuals = []
                for row in rows:
                    emb = embeddings[row["global_run_id"]][None, :]
                    coords = pca_2d.transform(emb).reshape(-1)
                    emb_hat = pca_resid.inverse_transform(pca_resid.transform(emb))
                    residual = emb - emb_hat
                    row["pca_x"] = float(coords[0]) if coords.size else 0.0
                    row["pca_y"] = float(coords[1]) if coords.size > 1 else 0.0
                    row["tangent_norm"] = float(np.linalg.norm(emb))
                    row["residual_norm"] = float(np.linalg.norm(residual))
                    row["residual_ratio"] = row["residual_norm"] / (row["tangent_norm"] + 1e-12)
                    if row["global_run_id"] in reference_run_ids:
                        ref_residuals.append(row["residual_norm"])
                ref_arr = np.asarray(ref_residuals, dtype=float)
                ref_mean = float(ref_arr.mean())
                ref_std = float(ref_arr.std(ddof=0))
                for row in rows:
                    row["z_residual_score"] = (row["residual_norm"] - ref_mean) / (ref_std + 1e-12)
                    row["distributed_reference_residual_mean"] = ref_mean
                    row["distributed_reference_residual_std"] = ref_std
                    row["distributed_reference_runs_for_zscore"] = int(len(reference_run_ids))
                rows_all.extend(rows)
    return pd.DataFrame(rows_all)


def compute_cusum(
    df: pd.DataFrame,
    *,
    reference_trial_ids: Sequence[str],
    clean_quantile: float,
    cusum_k_std: float,
    min_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    row_frames = []
    run_rows: List[Dict[str, Any]] = []
    report_rows: List[Dict[str, Any]] = []

    for (metric_name, cost_type), scope in df.groupby(["metric_name", "cost_type"], sort=False):
        reference_scope = scope[
            (scope["target_group"] == "clean")
            & (scope["target_trial_id"].astype(str).isin(reference_trial_ids))
        ].copy()
        clean_values = reference_scope["z_residual_score"].to_numpy(dtype=float)
        clean_center = float(np.median(clean_values))
        clean_std = float(np.std(clean_values, ddof=0))
        if not np.isfinite(clean_std) or clean_std < 1e-12:
            clean_std = 1.0
        k_value = float(cusum_k_std * clean_std)

        clean_max = []
        for _, run_df in reference_scope.groupby("global_run_id", sort=False):
            s_value = 0.0
            max_value = 0.0
            run_df = run_df.sort_values("segment_id", key=lambda col: col.map(numeric_segment_sort_key))
            for score in run_df["z_residual_score"].to_numpy(dtype=float):
                s_value = max(0.0, s_value + score - clean_center - k_value)
                max_value = max(max_value, s_value)
            clean_max.append(max_value)
        threshold = max(float(np.quantile(clean_max, clean_quantile)), min_threshold)

        scoped_rows = []
        scoped_run_rows = []
        for run_id, run_df in scope.groupby("global_run_id", sort=False):
            run_df = run_df.copy().sort_values("segment_id", key=lambda col: col.map(numeric_segment_sort_key))
            s_value = 0.0
            scores = []
            alarms = []
            for score in run_df["z_residual_score"].to_numpy(dtype=float):
                s_value = max(0.0, s_value + score - clean_center - k_value)
                scores.append(s_value)
                alarms.append(bool(s_value > threshold))
            run_df["cusum_step_idx"] = np.arange(len(run_df), dtype=int)
            run_df["cusum_score"] = scores
            run_df["cusum_alarm"] = alarms
            run_df["cusum_threshold"] = threshold
            run_df["cusum_center_clean_median"] = clean_center
            run_df["cusum_clean_std"] = clean_std
            run_df["cusum_k"] = k_value
            scoped_rows.append(run_df)

            first = run_df.iloc[0]
            is_reference_baseline = bool(first.get("is_reference_baseline", False))
            max_score = float(np.max(scores)) if scores else 0.0
            alarm_idx = next((idx for idx, flag in enumerate(alarms) if flag), -1)
            if not is_reference_baseline:
                scoped_run_rows.append(
                    {
                        "metric_name": metric_name,
                        "cost_type": cost_type,
                        "device_id": first["device_id"],
                        "device_type": first["device_type"],
                        "analysis_group": first["analysis_group"],
                        "global_run_id": run_id,
                        "target_run_id": first["target_run_id"],
                        "target_trial_id": first["target_trial_id"],
                        "target_group": first["target_group"],
                        "poisoning_type": first["poisoning_type"],
                        "condition_label": first["condition_label"],
                        "is_reference_baseline": is_reference_baseline,
                        "max_z_residual_score": float(run_df["z_residual_score"].max()),
                        "mean_z_residual_score": float(run_df["z_residual_score"].mean()),
                        "max_cusum_score": max_score,
                        "cusum_threshold": threshold,
                        "first_alarm_index": alarm_idx,
                        "first_alarm_segment_id": "" if alarm_idx < 0 else str(run_df.iloc[alarm_idx]["segment_id"]),
                        "actual_anomaly": first["target_group"] != "clean",
                        "predicted_anomaly": bool(max_score > threshold),
                        "n_segments": int(len(run_df)),
                        "n_clean_baseline_runs": int(reference_scope["global_run_id"].nunique()),
                        "n_clean_baseline_rows": int(len(reference_scope)),
                        "n_clean_baseline_device_types": int(reference_scope["device_type"].nunique()),
                    }
                )
        row_frames.extend(scoped_rows)
        run_rows.extend(scoped_run_rows)

        run_report_df = pd.DataFrame(scoped_run_rows)
        if run_report_df.empty:
            continue
        y_true = run_report_df["actual_anomaly"].astype(bool).to_numpy()
        y_pred = run_report_df["predicted_anomaly"].astype(bool).to_numpy()
        tp = int(np.sum(y_true & y_pred))
        tn = int(np.sum(~y_true & ~y_pred))
        fp = int(np.sum(~y_true & y_pred))
        fn = int(np.sum(y_true & ~y_pred))
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        report_rows.append(
            {
                "metric_name": metric_name,
                "cost_type": cost_type,
                "clean_quantile": clean_quantile,
                "cusum_threshold": threshold,
                "n_runs": int(len(run_report_df)),
                "n_clean_baseline_runs": int(reference_scope["global_run_id"].nunique()),
                "n_clean_baseline_rows": int(len(reference_scope)),
                "n_clean_baseline_device_types": int(reference_scope["device_type"].nunique()),
                "n_clean_runs": int((~y_true).sum()),
                "n_anomaly_runs": int(y_true.sum()),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": (tp + tn) / max(len(run_report_df), 1),
            }
        )
    return pd.concat(row_frames, ignore_index=True), pd.DataFrame(run_rows), pd.DataFrame(report_rows)


def make_plots(row_df: pd.DataFrame, run_df: pd.DataFrame, output_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib_cache"))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots.")
        return
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    palette = {
        "clean": "tab:blue",
        "availability_shortcuts": "tab:orange",
        "unlearnable_examples": "tab:red",
    }
    plot_df = row_df.copy()
    if "is_reference_baseline" in plot_df.columns:
        plot_df = plot_df[~plot_df["is_reference_baseline"].astype(bool)].copy()

    for (metric_name, cost_type), sub in plot_df.groupby(["metric_name", "cost_type"], sort=False):
        fig, ax = plt.subplots(figsize=(8, 5))
        for condition, cond_df in sub.groupby("condition_label", sort=False):
            stats = cond_df.groupby("cusum_step_idx")["cusum_score"].agg(["mean", "std"]).reset_index()
            x = stats["cusum_step_idx"].to_numpy(dtype=float)
            y = stats["mean"].to_numpy(dtype=float)
            std = stats["std"].fillna(0.0).to_numpy(dtype=float)
            color = palette.get(str(condition), "tab:gray")
            ax.plot(x, y, color=color, label=str(condition), linewidth=1.8)
            ax.fill_between(x, y - std, y + std, color=color, alpha=0.12)
        ax.axhline(float(sub["cusum_threshold"].iloc[0]), color="black", linestyle="--", linewidth=1.0)
        ax.set_title(f"Hetero pooled-reference CUSUM: {metric_name} / {cost_type}")
        ax.set_xlabel("epoch index")
        ax.set_ylabel("CUSUM anomaly score")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / f"all_hetero__{metric_name}__{cost_type}__cusum_trace.png", dpi=180)
        plt.close(fig)

    for (metric_name, cost_type), sub in plot_df.groupby(["metric_name", "cost_type"], sort=False):
        fig, ax = plt.subplots(figsize=(8, 5))
        for condition, cond_df in sub.groupby("condition_label", sort=False):
            color = palette.get(str(condition), "tab:gray")
            ax.scatter(
                cond_df["ot_cost"],
                cond_df["residual_norm"],
                s=18,
                alpha=0.65,
                color=color,
                label=str(condition),
            )
        ax.set_title(f"OT cost vs residual: {metric_name} / {cost_type}")
        ax.set_xlabel("OT cost")
        ax.set_ylabel("residual norm")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / f"all_hetero__{metric_name}__{cost_type}__ot_vs_residual.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        for device_type, dev_df in sub.groupby("device_type", sort=False):
            ax.scatter(
                dev_df["ot_cost"],
                dev_df["residual_norm"],
                s=18,
                alpha=0.65,
                label=str(device_type),
            )
        ax.set_title(f"OT cost vs residual by device: {metric_name} / {cost_type}")
        ax.set_xlabel("OT cost")
        ax.set_ylabel("residual norm")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / f"all_hetero__{metric_name}__{cost_type}__ot_vs_residual_by_device.png", dpi=180)
        plt.close(fig)


def save_outputs(
    residual_df: pd.DataFrame,
    row_df: pd.DataFrame,
    run_df: pd.DataFrame,
    report_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    residual_df.to_csv(output_dir / "backward_distributed_residual_scores.csv", index=False)
    row_df.to_csv(output_dir / "backward_cusum_rows.csv", index=False)
    run_df.to_csv(output_dir / "backward_cusum_run_summary.csv", index=False)
    report_df.to_csv(output_dir / "backward_cusum_detection_report.csv", index=False)
    y_true = run_df["actual_anomaly"].astype(bool).to_numpy()
    y_pred = run_df["predicted_anomaly"].astype(bool).to_numpy()
    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    pd.DataFrame(
        [{
            "scope": "all",
            "n_runs": int(len(run_df)),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp + 1e-12),
            "recall": tp / (tp + fn + 1e-12),
            "accuracy": (tp + tn) / max(len(run_df), 1),
        }]
    ).to_csv(output_dir / "backward_cusum_detection_report_overall.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="collected_logs")
    parser.add_argument("--output_dir", default="distributed_classification_result")
    parser.add_argument("--ot_module_path", default="1_calculate_ot_embedding.py")
    parser.add_argument("--analysis_phase", default=DEFAULT_ANALYSIS_PHASE)
    parser.add_argument("--cost_types", default=DEFAULT_COST_TYPES)
    parser.add_argument("--load_feature_columns", default=DEFAULT_LOAD_FEATURE_COLUMNS)
    parser.add_argument("--analysis_feature_columns", default=DEFAULT_ANALYSIS_FEATURE_COLUMNS)
    parser.add_argument("--reference_trial_ids", default=",".join(DEFAULT_REFERENCE_TRIAL_IDS))
    parser.add_argument("--segment_by", default="auto", choices=["auto", "epoch", "round", "none"])
    parser.add_argument("--support_bins", type=int, default=32)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--sinkhorn_reg", type=float, default=1.0)
    parser.add_argument("--sinkhorn_num_iter", type=int, default=300)
    parser.add_argument("--sinkhorn_stop_thr", type=float, default=1e-6)
    parser.add_argument("--normalize_solver_cost", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_sinkhorn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pca_components_for_residual", type=int, default=3)
    parser.add_argument("--max_samples_per_run", type=int, default=0)
    parser.add_argument("--clean_quantile", type=float, default=0.95)
    parser.add_argument("--cusum_k_std", type=float, default=0.5)
    parser.add_argument("--min_threshold", type=float, default=1e-9)
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = base_dir / input_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    ot_module_path = Path(args.ot_module_path)
    if not ot_module_path.is_absolute():
        ot_module_path = base_dir / ot_module_path

    ot_module = load_ot_module(ot_module_path)
    load_feature_columns = parse_csv_list(args.load_feature_columns)
    analysis_feature_columns = parse_csv_list(args.analysis_feature_columns)
    cost_types = ot_module.parse_cost_types(args.cost_types)
    reference_trial_ids = parse_csv_list(args.reference_trial_ids)

    records, analysis_feature_columns = collect_records(
        ot_module=ot_module,
        input_dir=input_dir,
        load_feature_columns=load_feature_columns,
        analysis_feature_columns=analysis_feature_columns,
        analysis_phase=args.analysis_phase,
        segment_by=args.segment_by,
        max_samples_per_run=args.max_samples_per_run,
    )
    print(
        f"Collected records={len(records)} devices={len({r.device_id for r in records})} "
        f"device_types={sorted({r.device_type for r in records})}"
    )
    residual_df = compute_residuals(
        records=records,
        analysis_feature_columns=analysis_feature_columns,
        cost_types=cost_types,
        reference_trial_ids=reference_trial_ids,
        support_bins=args.support_bins,
        window_size=args.window_size,
        sinkhorn_reg=args.sinkhorn_reg,
        sinkhorn_num_iter=args.sinkhorn_num_iter,
        sinkhorn_stop_thr=args.sinkhorn_stop_thr,
        use_sinkhorn=args.use_sinkhorn,
        normalize_solver_cost=args.normalize_solver_cost,
        pca_components_for_residual=args.pca_components_for_residual,
        ot_module=ot_module,
    )
    row_df, run_df, report_df = compute_cusum(
        residual_df,
        reference_trial_ids=reference_trial_ids,
        clean_quantile=args.clean_quantile,
        cusum_k_std=args.cusum_k_std,
        min_threshold=args.min_threshold,
    )
    save_outputs(residual_df, row_df, run_df, report_df, output_dir)
    if not args.no_plots:
        make_plots(row_df, run_df, output_dir)

    print(f"Saved hetero distributed outputs to {output_dir}")
    print(report_df.to_string(index=False))


if __name__ == "__main__":
    main()
