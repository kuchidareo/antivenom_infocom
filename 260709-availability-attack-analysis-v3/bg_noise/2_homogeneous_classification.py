#!/usr/bin/env python3
"""CUSUM anomaly analysis from OT residual scores.

This script intentionally reuses ``1_calculate_ot_embedding.py`` as the OT
module. The module computes OT tangent embeddings, clean-PCA residuals, and
post-hoc z-scores. This file adds a statistical anomaly layer on top of
``z_residual_norm``.

Default analysis focuses on the backward phase because that is where the
gradient/update signal is expected to be most visible before optimizer_step.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_ANALYSIS_PHASE = "backward"
DEFAULT_COST_TYPES = "c3_window_shape"
DEFAULT_REFERENCE_BASELINE_TRIAL_IDS = tuple(f"reference_{idx}" for idx in range(5))


def load_ot_module(script_path: Path) -> Any:
    if not script_path.exists():
        raise FileNotFoundError(f"OT module not found: {script_path}")
    spec = importlib.util.spec_from_file_location("ot_embedding_module", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_feature_columns(value: str, ot_module: Any) -> List[str]:
    if value:
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(ot_module.DEFAULT_FEATURE_COLUMNS)


def parse_cost_types(value: str, ot_module: Any) -> List[str]:
    return ot_module.parse_cost_types(value)


def parse_filter_values(*values: str) -> List[str]:
    filters: List[str] = []
    for value in values:
        if not value:
            continue
        filters.extend(item.strip() for item in value.split(",") if item.strip())
    return filters


def parse_reference_baseline_trial_ids(value: str) -> List[str]:
    ids = [item.strip() for item in value.split(",") if item.strip()]
    if not ids:
        raise ValueError("At least one reference baseline trial id is required.")
    return ids


def parse_optional_trial_ids(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def add_reference_baseline_column(df: pd.DataFrame, reference_baseline_trial_ids: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    if "is_reference_baseline" in out.columns:
        out["is_reference_baseline"] = out["is_reference_baseline"].astype(bool)
    else:
        out["is_reference_baseline"] = (
            (out["target_group"].astype(str) == "clean")
            & (out["target_trial_id"].astype(str).isin(reference_baseline_trial_ids))
        )
    return out


def analysis_group_matches(analysis_group: str, filters: Sequence[str]) -> bool:
    if not filters:
        return True
    group = str(analysis_group)
    group_parts = set(part for part in group.replace("\\", "/").split("/") if part)
    for value in filters:
        text = str(value).strip()
        if not text:
            continue
        if group == text or text in group_parts or text in group:
            return True
    return False


def compute_or_reuse_ot_outputs(
    *,
    ot_module: Any,
    input_dir: Path,
    ot_output_dir: Path,
    feature_columns: Sequence[str],
    cost_types: Sequence[str],
    reference_trial_ids_value: str,
    clean_baseline_trial_ids_value: str,
    window_size: int,
    sinkhorn_reg: float,
    use_sinkhorn: bool,
    sinkhorn_num_iter: int,
    sinkhorn_stop_thr: float,
    normalize_solver_cost: bool,
    pca_components_for_residual: int,
    max_samples_per_run: int,
    segment_by: str,
    force_recompute_ot: bool,
    analysis_group_filters: Sequence[str],
) -> None:
    existing = list(ot_output_dir.rglob("ot_embedding_summary_zscored.csv"))
    if existing and not force_recompute_ot:
        print(f"Reusing existing OT outputs under {ot_output_dir}")
        return

    groups = ot_module.discover_input_groups(input_dir)
    split_outputs_by_group = len(groups) > 1
    groups = [
        (group_label, group_dir)
        for group_label, group_dir in groups
        if analysis_group_matches(group_label, analysis_group_filters)
    ]
    if not groups:
        raise ValueError(f"No input groups matched filters={list(analysis_group_filters)} under {input_dir}")
    if len(groups) > 1:
        print(f"Discovered {len(groups)} local_ml analysis groups under {input_dir}")

    for group_label, group_dir in groups:
        group_output_dir = ot_output_dir / group_label if split_outputs_by_group else ot_output_dir
        print(f"Running OT residual analysis for group={group_label} input={group_dir}")
        ot_module.run_one_group(
            input_dir=group_dir,
            output_dir=group_output_dir,
            feature_columns=feature_columns,
            cost_types=cost_types,
            reference_trial_ids_value=reference_trial_ids_value,
            clean_baseline_trial_ids_value=clean_baseline_trial_ids_value,
            window_size=window_size,
            sinkhorn_reg=sinkhorn_reg,
            use_sinkhorn=use_sinkhorn,
            sinkhorn_num_iter=sinkhorn_num_iter,
            sinkhorn_stop_thr=sinkhorn_stop_thr,
            normalize_solver_cost=normalize_solver_cost,
            pca_components_for_residual=pca_components_for_residual,
            max_samples_per_run=max_samples_per_run,
            segment_by=segment_by,
        )


def load_zscore_outputs(ot_output_dir: Path) -> pd.DataFrame:
    paths = sorted(ot_output_dir.rglob("ot_embedding_summary_zscored.csv"))
    if not paths:
        raise FileNotFoundError(f"No ot_embedding_summary_zscored.csv found under {ot_output_dir}")

    frames = []
    for path in paths:
        df = pd.read_csv(path)
        try:
            analysis_group = str(path.parent.relative_to(ot_output_dir))
        except ValueError:
            analysis_group = path.parent.name
        if analysis_group in {"", "."}:
            analysis_group = ot_output_dir.name
        df.insert(0, "analysis_group", analysis_group)
        df.insert(1, "source_summary_path", str(path))
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    if "z_residual_norm" not in out.columns:
        raise ValueError("OT output is missing z_residual_norm.")
    out["z_residual_score"] = pd.to_numeric(out["z_residual_norm"], errors="coerce")
    return out


def _segment_sort_value(value: Any) -> Tuple[int, Any]:
    text = str(value)
    try:
        return (0, int(float(text)))
    except ValueError:
        return (1, text)


def _condition_label(row: pd.Series) -> str:
    if str(row.get("target_group", "")) == "clean":
        return "clean"
    return str(row.get("poisoning_type", "poisoning"))


def _target_sort_cols(df: pd.DataFrame) -> List[str]:
    cols = []
    for col in ["segment_type", "segment_id", "target_trial_id", "target_run_id"]:
        if col in df.columns:
            cols.append(col)
    return cols


def compute_cusum_scores(
    df: pd.DataFrame,
    *,
    analysis_phase: str,
    reference_baseline_trial_ids: Sequence[str],
    clean_quantile: float,
    cusum_k_std: float,
    min_threshold: float,
    include_baseline_runs_in_eval: bool,
    evaluation_trial_ids: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "analysis_phase" not in df.columns:
        raise ValueError("Expected analysis_phase column from phase-specific OT output.")

    phase_df = df[df["analysis_phase"].astype(str) == analysis_phase].copy()
    if phase_df.empty:
        available = sorted(df["analysis_phase"].astype(str).unique().tolist())
        raise ValueError(f"No rows found for analysis_phase={analysis_phase!r}. Available phases: {available}")

    required = ["analysis_group", "metric_name", "cost_type", "target_run_id", "target_group"]
    missing = [col for col in required if col not in phase_df.columns]
    if missing:
        raise ValueError(f"Missing required columns for CUSUM analysis: {missing}")

    phase_df["z_residual_score"] = pd.to_numeric(phase_df["z_residual_score"], errors="coerce")
    phase_df = phase_df[np.isfinite(phase_df["z_residual_score"])].copy()
    phase_df = add_reference_baseline_column(phase_df, reference_baseline_trial_ids)
    phase_df["condition_label"] = phase_df.apply(_condition_label, axis=1)
    phase_df["residual_norm"] = pd.to_numeric(phase_df["residual_norm"], errors="coerce")
    phase_df = phase_df[np.isfinite(phase_df["residual_norm"])].copy()

    z_group_cols = ["analysis_group", "metric_name", "cost_type"]
    if {"segment_type", "segment_id"}.issubset(phase_df.columns):
        z_group_cols.extend(["segment_type", "segment_id"])
    z_frames = []
    for _, z_group in phase_df.groupby(z_group_cols, sort=False):
        reference_rows = z_group[
            (z_group["target_group"].astype(str) == "clean")
            & (z_group["is_reference_baseline"].astype(bool))
        ]
        if reference_rows.empty:
            continue
        mean_value = float(reference_rows["residual_norm"].mean())
        std_value = float(reference_rows["residual_norm"].std(ddof=0))
        z_group = z_group.copy()
        z_group["z_residual_score"] = (z_group["residual_norm"] - mean_value) / (std_value + 1e-12)
        z_group["reference_residual_mean"] = mean_value
        z_group["reference_residual_std"] = std_value
        z_group["reference_rows_for_zscore"] = int(len(reference_rows))
        z_group["reference_runs_for_zscore"] = int(reference_rows["target_run_id"].nunique())
        z_frames.append(z_group)
    if not z_frames:
        raise ValueError(
            "No reference baseline rows found for z-score recomputation. "
            f"Expected clean target_trial_id in {list(reference_baseline_trial_ids)}."
        )
    phase_df = pd.concat(z_frames, ignore_index=True)
    evaluation_trial_set = set(evaluation_trial_ids)

    row_outputs: List[pd.DataFrame] = []
    run_rows: List[Dict[str, Any]] = []
    report_rows: List[Dict[str, Any]] = []

    scope_cols = ["analysis_group", "metric_name", "cost_type"]
    for scope_key, scope in phase_df.groupby(scope_cols, sort=False):
        scope_values = dict(zip(scope_cols, scope_key))
        clean_scope = scope[
            (scope["target_group"].astype(str) == "clean")
            & (scope["is_reference_baseline"].astype(bool))
        ].copy()
        if clean_scope.empty:
            print(f"Skipping CUSUM scope without clean baseline: {scope_values}")
            continue

        clean_values = clean_scope["z_residual_score"].to_numpy(dtype=float)
        clean_center = float(np.median(clean_values))
        clean_std = float(np.std(clean_values, ddof=0))
        if not math.isfinite(clean_std) or clean_std < 1e-12:
            clean_std = 1.0
        k_value = float(cusum_k_std * clean_std)

        clean_max_scores = []
        clean_run_scores: Dict[str, float] = {}
        for run_id, run_df in clean_scope.groupby("target_run_id", sort=False):
            s_value = 0.0
            max_value = 0.0
            run_df = run_df.sort_values(
                by=_target_sort_cols(run_df),
                key=lambda col: col.map(_segment_sort_value) if col.name == "segment_id" else col,
            )
            for score in run_df["z_residual_score"].to_numpy(dtype=float):
                s_value = max(0.0, s_value + (score - clean_center - k_value))
                max_value = max(max_value, s_value)
            clean_max_scores.append(max_value)
            clean_run_scores[str(run_id)] = max_value

        if clean_max_scores:
            threshold = float(np.quantile(clean_max_scores, clean_quantile))
        else:
            threshold = min_threshold
        threshold = max(threshold, min_threshold)

        scoped_row_frames = []
        scoped_run_rows = []
        for run_id, run_df in scope.groupby("target_run_id", sort=False):
            run_df = run_df.copy()
            run_df = run_df.sort_values(
                by=_target_sort_cols(run_df),
                key=lambda col: col.map(_segment_sort_value) if col.name == "segment_id" else col,
            )
            s_value = 0.0
            scores = []
            alarms = []
            for score in run_df["z_residual_score"].to_numpy(dtype=float):
                s_value = max(0.0, s_value + (score - clean_center - k_value))
                scores.append(s_value)
                alarms.append(bool(s_value > threshold))

            run_df["cusum_step_idx"] = np.arange(len(run_df), dtype=int)
            run_df["cusum_score"] = scores
            run_df["cusum_alarm"] = alarms
            run_df["cusum_threshold"] = threshold
            run_df["cusum_center_clean_median"] = clean_center
            run_df["cusum_clean_std"] = clean_std
            run_df["cusum_k"] = k_value
            scoped_row_frames.append(run_df)

            max_score = float(np.max(scores)) if scores else 0.0
            alarm_positions = [idx for idx, flag in enumerate(alarms) if flag]
            first_alarm_index = alarm_positions[0] if alarm_positions else -1
            first_alarm_segment_id = ""
            if first_alarm_index >= 0 and "segment_id" in run_df.columns:
                first_alarm_segment_id = str(run_df.iloc[first_alarm_index]["segment_id"])

            first = run_df.iloc[0]
            is_reference_baseline = bool(first.get("is_reference_baseline", False))
            actual_anomaly = str(first["target_group"]) != "clean"
            predicted_anomaly = max_score > threshold
            include_in_eval = (not is_reference_baseline) or include_baseline_runs_in_eval
            if evaluation_trial_set:
                include_in_eval = include_in_eval and str(first.get("target_trial_id", "")) in evaluation_trial_set
            if include_in_eval:
                scoped_run_rows.append(
                    {
                        **scope_values,
                        "analysis_phase": analysis_phase,
                        "target_run_id": run_id,
                        "target_trial_id": first.get("target_trial_id", ""),
                        "target_group": first.get("target_group", ""),
                        "poisoning_type": first.get("poisoning_type", ""),
                        "condition_label": first.get("condition_label", ""),
                        "is_reference_baseline": is_reference_baseline,
                        "max_z_residual_score": float(run_df["z_residual_score"].max()),
                        "mean_z_residual_score": float(run_df["z_residual_score"].mean()),
                        "max_cusum_score": max_score,
                        "cusum_threshold": threshold,
                        "first_alarm_index": first_alarm_index,
                        "first_alarm_segment_id": first_alarm_segment_id,
                        "actual_anomaly": actual_anomaly,
                        "predicted_anomaly": predicted_anomaly,
                        "n_segments": int(len(run_df)),
                        "n_clean_baseline_rows": int(len(clean_scope)),
                        "n_clean_baseline_runs": int(clean_scope["target_run_id"].nunique()),
                    }
                )

        if scoped_row_frames:
            row_outputs.append(pd.concat(scoped_row_frames, ignore_index=True))
        run_rows.extend(scoped_run_rows)

        run_report_df = pd.DataFrame(scoped_run_rows)
        if not run_report_df.empty:
            y_true = run_report_df["actual_anomaly"].astype(bool).to_numpy()
            y_pred = run_report_df["predicted_anomaly"].astype(bool).to_numpy()
            tp = int(np.sum(y_true & y_pred))
            tn = int(np.sum(~y_true & ~y_pred))
            fp = int(np.sum(~y_true & y_pred))
            fn = int(np.sum(y_true & ~y_pred))
            precision = tp / (tp + fp + 1e-12)
            recall = tp / (tp + fn + 1e-12)
            f1 = 2 * precision * recall / (precision + recall + 1e-12)
            accuracy = (tp + tn) / max(len(run_report_df), 1)
            report_rows.append(
                {
                    **scope_values,
                    "analysis_phase": analysis_phase,
                    "clean_quantile": clean_quantile,
                    "cusum_k_std": cusum_k_std,
                    "cusum_threshold": threshold,
                    "clean_center_median": clean_center,
                    "clean_std": clean_std,
                    "n_runs": int(len(run_report_df)),
                    "n_clean_runs": int((~y_true).sum()),
                    "n_anomaly_runs": int(y_true.sum()),
                    "tp": tp,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "accuracy": accuracy,
                }
            )

    if not row_outputs:
        raise ValueError("CUSUM analysis produced no rows.")

    row_df = pd.concat(row_outputs, ignore_index=True)
    run_df = pd.DataFrame(run_rows)
    report_df = pd.DataFrame(report_rows)
    return row_df, run_df, report_df


def make_plots(row_df: pd.DataFrame, run_df: pd.DataFrame, output_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib_cache"))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping CUSUM plots.")
        return

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    palette = {
        "clean": "tab:blue",
        "unlearnable_examples": "tab:red",
        "availability_shortcuts": "tab:orange",
        "random_label_flipping": "tab:green",
        "target_label_flipping": "tab:purple",
    }
    plot_row_df = row_df.copy()
    if "is_reference_baseline" in plot_row_df.columns:
        plot_row_df = plot_row_df[~plot_row_df["is_reference_baseline"].astype(bool)].copy()

    for (analysis_group, metric_name, cost_type), sub in plot_row_df.groupby(
        ["analysis_group", "metric_name", "cost_type"], sort=False
    ):
        safe_group = str(analysis_group).replace("/", "__")
        safe_metric = str(metric_name).replace("/", "_")
        safe_cost = str(cost_type).replace("/", "_")
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        for run_id, run_sub in sub.groupby("target_run_id", sort=False):
            condition = str(run_sub.iloc[0]["condition_label"])
            color = palette.get(condition, "tab:gray")
            x = np.arange(len(run_sub))
            label = f"{condition}:{run_id}"
            axes[0].plot(x, run_sub["z_residual_score"], color=color, alpha=0.75, linewidth=1.2, label=label)
            axes[1].plot(x, run_sub["cusum_score"], color=color, alpha=0.75, linewidth=1.2, label=label)
        threshold = float(sub["cusum_threshold"].iloc[0])
        axes[1].axhline(threshold, color="black", linestyle="--", linewidth=1.0, label="clean CUSUM threshold")
        axes[0].set_ylabel("z residual")
        axes[1].set_ylabel("CUSUM")
        axes[1].set_xlabel("segment index")
        axes[0].set_title(f"{analysis_group} / {metric_name} / {cost_type}")
        for ax in axes:
            ax.grid(True, alpha=0.3)
        handles, labels = axes[1].get_legend_handles_labels()
        dedup = dict(zip(labels, handles))
        axes[1].legend(dedup.values(), dedup.keys(), fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{safe_group}__{safe_metric}__{safe_cost}__cusum.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        for condition, cond_df in sub.groupby("condition_label", sort=False):
            condition = str(condition)
            color = palette.get(condition, "tab:gray")
            stats = (
                cond_df.groupby("cusum_step_idx", sort=True)["cusum_score"]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            x = stats["cusum_step_idx"].to_numpy(dtype=float)
            y = stats["mean"].to_numpy(dtype=float)
            std = stats["std"].fillna(0.0).to_numpy(dtype=float)
            ax.plot(x, y, color=color, linewidth=1.8, label=condition)
            ax.fill_between(x, y - std, y + std, color=color, alpha=0.12, linewidth=0)
        threshold = float(sub["cusum_threshold"].iloc[0])
        ax.axhline(threshold, color="black", linestyle="--", linewidth=1.0, label="clean CUSUM threshold")
        ax.set_title(f"Anomaly score trace: {analysis_group} / {metric_name} / {cost_type}")
        ax.set_xlabel("segment index")
        ax.set_ylabel("CUSUM anomaly score")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(
            plot_dir / f"{safe_group}__{safe_metric}__{safe_cost}__condition_mean_cusum_trace.png",
            dpi=180,
        )
        plt.close(fig)

        if {"pca_x", "pca_y"}.issubset(sub.columns):
            pca_sub = sub.copy()
            pca_sub["pca_x"] = pd.to_numeric(pca_sub["pca_x"], errors="coerce")
            pca_sub["pca_y"] = pd.to_numeric(pca_sub["pca_y"], errors="coerce")
            pca_sub = pca_sub[np.isfinite(pca_sub["pca_x"]) & np.isfinite(pca_sub["pca_y"])]
            if not pca_sub.empty:
                fig, ax = plt.subplots(figsize=(7, 6))
                for condition, cond_df in pca_sub.groupby("condition_label", sort=False):
                    condition = str(condition)
                    color = palette.get(condition, "tab:gray")
                    ax.scatter(
                        cond_df["pca_x"],
                        cond_df["pca_y"],
                        s=28,
                        alpha=0.75,
                        color=color,
                        label=condition,
                    )
                ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
                ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.35)
                ax.set_title(f"Tangent PCA: {analysis_group} / {metric_name} / {cost_type}")
                ax.set_xlabel("clean-PCA tangent component 1")
                ax.set_ylabel("clean-PCA tangent component 2")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8, loc="best")
                fig.tight_layout()
                fig.savefig(plot_dir / f"{safe_group}__{safe_metric}__{safe_cost}__tangent_pca.png", dpi=180)
                plt.close(fig)

    if not run_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = []
        values = []
        colors = []
        for _, row in run_df.sort_values(["analysis_group", "metric_name", "cost_type", "target_run_id"]).iterrows():
            labels.append(f"{row['analysis_group']}|{row['metric_name']}|{row['cost_type']}|{row['target_run_id']}")
            values.append(row["max_cusum_score"])
            colors.append(palette.get(str(row["condition_label"]), "tab:gray"))
        ax.bar(np.arange(len(values)), values, color=colors)
        ax.set_ylabel("max CUSUM")
        ax.set_xticks([])
        ax.set_title("Run-level max CUSUM scores")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / "run_level_max_cusum.png", dpi=180)
        plt.close(fig)


def save_outputs(
    row_df: pd.DataFrame,
    run_df: pd.DataFrame,
    report_df: pd.DataFrame,
    output_dir: Path,
    analysis_phase: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    phase = str(analysis_phase)
    row_df.to_csv(output_dir / f"{phase}_z_residual_cusum_rows.csv", index=False)

    pca_cols = [
        col
        for col in [
            "analysis_group",
            "source_summary_path",
            "analysis_phase",
            "segment_type",
            "segment_id",
            "metric_name",
            "reference_run_id",
            "reference_trial_id",
            "target_trial_id",
            "target_run_id",
            "target_group",
            "poisoning_type",
            "condition_label",
            "cost_type",
            "ot_cost",
            "tangent_norm",
            "pca_x",
            "pca_y",
            "residual_norm",
            "residual_ratio",
            "z_residual_score",
            "cusum_score",
            "cusum_alarm",
        ]
        if col in row_df.columns
    ]
    if {"pca_x", "pca_y"}.issubset(row_df.columns):
        row_df[pca_cols].to_csv(output_dir / f"{phase}_tangent_pca_coordinates.csv", index=False)

    run_df.to_csv(output_dir / f"{phase}_cusum_run_summary.csv", index=False)
    report_df.to_csv(output_dir / f"{phase}_cusum_detection_report.csv", index=False)

    aggregate_rows = []
    if not run_df.empty:
        y_true = run_df["actual_anomaly"].astype(bool).to_numpy()
        y_pred = run_df["predicted_anomaly"].astype(bool).to_numpy()
        tp = int(np.sum(y_true & y_pred))
        tn = int(np.sum(~y_true & ~y_pred))
        fp = int(np.sum(~y_true & y_pred))
        fn = int(np.sum(y_true & ~y_pred))
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        aggregate_rows.append(
            {
                "scope": "all",
                "n_runs": int(len(run_df)),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": (tp + tn) / max(len(run_df), 1),
            }
        )
    pd.DataFrame(aggregate_rows).to_csv(output_dir / f"{phase}_cusum_detection_report_overall.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="collected_logs")
    parser.add_argument("--output_dir", default="classification_result")
    parser.add_argument(
        "--ot_output_dir",
        default="",
        help="Directory for OT intermediate outputs. Default: output_dir/ot_outputs",
    )
    parser.add_argument("--ot_module_path", default="1_calculate_ot_embedding.py")
    parser.add_argument("--analysis_phase", default=DEFAULT_ANALYSIS_PHASE)
    parser.add_argument("--reference_trial_ids", default="global_reference")
    parser.add_argument("--reference_trial_id", default="")
    parser.add_argument(
        "--reference_baseline_trial_ids",
        default=",".join(DEFAULT_REFERENCE_BASELINE_TRIAL_IDS),
        help=(
            "Clean trial ids used for OT PCA/z-score and CUSUM baseline. "
            "Default uses reference_0..reference_4; include trial_* ids to calibrate bg-noise clean jitter."
        ),
    )
    parser.add_argument(
        "--include_baseline_runs_in_eval",
        action="store_true",
        help=(
            "Include clean runs used for baseline calibration in the run-level evaluation output. "
            "Useful when trial_* clean bg-noise runs are intentionally part of the benign jitter baseline."
        ),
    )
    parser.add_argument(
        "--evaluation_trial_ids",
        default="",
        help=(
            "Optional comma-separated target trial ids to include in detection metrics, "
            "e.g. trial_0,trial_1,trial_2,trial_3,trial_4."
        ),
    )
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--sinkhorn_reg", type=float, default=1.0)
    parser.add_argument("--sinkhorn_num_iter", type=int, default=300)
    parser.add_argument("--sinkhorn_stop_thr", type=float, default=1e-6)
    parser.add_argument("--normalize_solver_cost", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_sinkhorn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pca_components_for_residual", type=int, default=3)
    parser.add_argument("--feature_columns", default="")
    parser.add_argument("--cost_types", default=DEFAULT_COST_TYPES)
    parser.add_argument(
        "--device_id",
        default="",
        help="Optional comma-separated device ids to analyze, e.g. 192.168.0.112.",
    )
    parser.add_argument(
        "--analysis_group_filter",
        default="",
        help="Optional comma-separated analysis group filters, e.g. 192.168.0.112/local_ml.",
    )
    parser.add_argument("--segment_by", default="auto", choices=["auto", "epoch", "round", "none"])
    parser.add_argument("--max_samples_per_run", type=int, default=0)
    parser.add_argument("--force_recompute_ot", action="store_true")
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
    ot_output_dir = Path(args.ot_output_dir) if args.ot_output_dir else output_dir / "ot_outputs"
    if not ot_output_dir.is_absolute():
        ot_output_dir = base_dir / ot_output_dir
    ot_module_path = Path(args.ot_module_path)
    if not ot_module_path.is_absolute():
        ot_module_path = base_dir / ot_module_path

    ot_module = load_ot_module(ot_module_path)
    feature_columns = parse_feature_columns(args.feature_columns, ot_module)
    cost_types = parse_cost_types(args.cost_types, ot_module)
    reference_trial_ids_value = args.reference_trial_id or args.reference_trial_ids
    reference_baseline_trial_ids = parse_reference_baseline_trial_ids(args.reference_baseline_trial_ids)
    evaluation_trial_ids = parse_optional_trial_ids(args.evaluation_trial_ids)
    analysis_group_filters = parse_filter_values(args.device_id, args.analysis_group_filter)

    compute_or_reuse_ot_outputs(
        ot_module=ot_module,
        input_dir=input_dir,
        ot_output_dir=ot_output_dir,
        feature_columns=feature_columns,
        cost_types=cost_types,
        reference_trial_ids_value=reference_trial_ids_value,
        clean_baseline_trial_ids_value=",".join(reference_baseline_trial_ids),
        window_size=args.window_size,
        sinkhorn_reg=args.sinkhorn_reg,
        use_sinkhorn=args.use_sinkhorn,
        sinkhorn_num_iter=args.sinkhorn_num_iter,
        sinkhorn_stop_thr=args.sinkhorn_stop_thr,
        normalize_solver_cost=args.normalize_solver_cost,
        pca_components_for_residual=args.pca_components_for_residual,
        max_samples_per_run=args.max_samples_per_run,
        segment_by=args.segment_by,
        force_recompute_ot=args.force_recompute_ot,
        analysis_group_filters=analysis_group_filters,
    )

    zscore_df = load_zscore_outputs(ot_output_dir)
    if analysis_group_filters:
        zscore_df = zscore_df[
            zscore_df["analysis_group"].astype(str).map(
                lambda value: analysis_group_matches(value, analysis_group_filters)
            )
        ].copy()
        if zscore_df.empty:
            raise ValueError(
                f"No OT z-score rows matched device/group filters={analysis_group_filters} "
                f"under {ot_output_dir}"
            )
    zscore_df = zscore_df[zscore_df["cost_type"].astype(str).isin(cost_types)].copy()
    if zscore_df.empty:
        raise ValueError(f"No OT z-score rows matched requested cost_types={cost_types}")
    row_df, run_df, report_df = compute_cusum_scores(
        zscore_df,
        analysis_phase=args.analysis_phase,
        reference_baseline_trial_ids=reference_baseline_trial_ids,
        clean_quantile=args.clean_quantile,
        cusum_k_std=args.cusum_k_std,
        min_threshold=args.min_threshold,
        include_baseline_runs_in_eval=args.include_baseline_runs_in_eval,
        evaluation_trial_ids=evaluation_trial_ids,
    )
    save_outputs(row_df, run_df, report_df, output_dir, args.analysis_phase)
    if not args.no_plots:
        make_plots(row_df, run_df, output_dir)

    print(f"Saved CUSUM classification outputs to {output_dir}")
    if not report_df.empty:
        cols = [
            "analysis_group",
            "metric_name",
            "cost_type",
            "n_runs",
            "n_clean_runs",
            "n_anomaly_runs",
            "precision",
            "recall",
            "f1",
            "accuracy",
        ]
        print(report_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
