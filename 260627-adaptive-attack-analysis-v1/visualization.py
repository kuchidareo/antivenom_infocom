#!/usr/bin/env python3
"""Visualize OT distances and tangent embeddings.

This script reads outputs produced by calculate_ot_embedding.py.

It creates two main figures:
1. OT distance by trial, one subplot row per cost type.
2. 2D PCA of tangent embeddings, one subplot row per cost type.

For PCA plots, the clean reference target, usually trial_0_clean, is shifted to
(0, 0). When multiple devices are loaded, the centroid of all reference points
for a cost type is used as the origin.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


SUMMARY_NAME = "ot_embedding_summary_zscored.csv"
EMBEDDING_NAME = "tangent_embeddings.csv"


def parse_trial_id(value: object) -> int:
    text = str(value)
    match = re.search(r"(\d+)$", text)
    return int(match.group(1)) if match else 0


def result_group_label(result_file: Path, input_dir: Path) -> str:
    parent = result_file.parent
    try:
        rel = parent.relative_to(input_dir)
        return str(rel)
    except ValueError:
        return parent.name


def discover_result_dirs(input_dir: Path) -> List[Path]:
    summary_files = sorted(input_dir.rglob(SUMMARY_NAME))
    dirs = [path.parent for path in summary_files]
    if not dirs:
        raise ValueError(f"No result directories containing {SUMMARY_NAME} under {input_dir}")
    return dirs


def load_summary(result_dirs: Sequence[Path], input_dir: Path) -> pd.DataFrame:
    frames = []
    for result_dir in result_dirs:
        df = pd.read_csv(result_dir / SUMMARY_NAME)
        if "segment_type" not in df.columns:
            df["segment_type"] = "full_run"
        if "segment_id" not in df.columns:
            df["segment_id"] = "all"
        if "metric_name" not in df.columns:
            df["metric_name"] = "all_metrics"
        df["segment_index"] = df["segment_id"].map(parse_trial_id)
        group = result_group_label(result_dir / SUMMARY_NAME, input_dir)
        df = df.assign(
            result_group=group,
            device_id=group.split("/")[0],
            trial_index=df["target_trial_id"].map(parse_trial_id),
            condition=np.where(df["target_group"] == "clean", "clean", df["poisoning_type"]),
        )
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_embeddings(result_dirs: Sequence[Path], input_dir: Path) -> pd.DataFrame:
    frames = []
    for result_dir in result_dirs:
        df = pd.read_csv(result_dir / EMBEDDING_NAME)
        group = result_group_label(result_dir / EMBEDDING_NAME, input_dir)
        df = df.assign(
            result_group=group,
            device_id=group.split("/")[0],
            trial_index=df["target_trial_id"].map(parse_trial_id),
            condition=np.where(df["target_group"] == "clean", "clean", df["poisoning_type"]),
        )
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def pca_2d(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"PCA input must be 2D, got {x.shape}")
    centered = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2]
    coords = centered @ components.T
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(coords.shape[0])])
    return coords


def cost_types_in_order(df: pd.DataFrame) -> List[str]:
    preferred = ["c1_time", "c2_value", "c3_window_abs", "c3_window_shape"]
    found = list(df["cost_type"].dropna().unique())
    ordered = [cost for cost in preferred if cost in found]
    ordered.extend(cost for cost in found if cost not in ordered)
    return ordered


def metric_names_in_order(df: pd.DataFrame) -> List[str]:
    preferred = [
        "cpu_0",
        "cpu_1",
        "cpu_2",
        "cpu_3",
        "memory",
        "voluntary_context",
        "involuntary_context",
        "minor_fault",
        "all_metrics",
    ]
    found = list(df["metric_name"].dropna().unique())
    ordered = [metric for metric in preferred if metric in found]
    ordered.extend(metric for metric in found if metric not in ordered)
    return ordered


def safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def color_for_condition(condition: str) -> str:
    colors = {
        "clean": "tab:blue",
        "adaptive": "tab:red",
        "blurring": "tab:orange",
        "label_flip": "tab:green",
        "backdoor": "tab:purple",
    }
    return colors.get(str(condition), "tab:gray")


def marker_for_condition(condition: str) -> str:
    return "o" if condition == "clean" else "s"


def plot_ot_distance(summary: pd.DataFrame, output_dir: Path, y_column: str = "ot_cost") -> None:
    x_column = "segment_index" if "segment_index" in summary.columns else "trial_index"
    x_label = "epoch/round" if x_column == "segment_index" else "trial"

    for metric_name in metric_names_in_order(summary):
        metric_df = summary[summary["metric_name"] == metric_name].copy()
        cost_types = cost_types_in_order(metric_df)
        fig, axes = make_row_axes(len(cost_types), width=8.5, row_height=3.0)

        for ax, cost_type in zip(axes, cost_types):
            sub = metric_df[metric_df["cost_type"] == cost_type].copy()
            grouped = (
                sub.groupby(["condition", x_column], as_index=False)[y_column]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            for condition, cdf in grouped.groupby("condition", sort=False):
                cdf = cdf.sort_values(x_column)
                color = color_for_condition(condition)
                ax.plot(
                    cdf[x_column],
                    cdf["mean"],
                    marker=marker_for_condition(condition),
                    color=color,
                    label=condition,
                )
                std = cdf["std"].fillna(0.0)
                ax.fill_between(
                    cdf[x_column].to_numpy(),
                    (cdf["mean"] - std).to_numpy(),
                    (cdf["mean"] + std).to_numpy(),
                    color=color,
                    alpha=0.16,
                    linewidth=0,
                )
            ax.set_title(f"{metric_name} / {cost_type}")
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_column)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")

        fig.tight_layout()
        suffix = f"{safe_name(metric_name)}_{safe_name(y_column)}"
        fig.savefig(output_dir / f"ot_distance_by_{x_label.replace('/', '_')}_{suffix}.png", dpi=180)
        fig.savefig(output_dir / f"ot_distance_by_{x_label.replace('/', '_')}_{suffix}.pdf")
        close_figure(fig)


def embedding_matrix_for_cost(embeddings: pd.DataFrame, cost_type: str) -> Tuple[pd.DataFrame, np.ndarray]:
    sub = embeddings[embeddings["cost_type"] == cost_type].copy()
    emb_cols = [column for column in sub.columns if column.startswith("emb_")]
    xdf = sub[emb_cols].apply(pd.to_numeric, errors="coerce")
    xdf = xdf.dropna(axis=1, how="all")
    x = xdf.to_numpy(dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError(f"Embedding matrix for {cost_type} contains NaN/inf after dropping empty columns.")
    return sub.reset_index(drop=True), x


def compute_pca_coordinates(embeddings: pd.DataFrame, reference_trial_id: str) -> pd.DataFrame:
    rows = []
    for result_group, group_df in embeddings.groupby("result_group", sort=False):
        for cost_type in cost_types_in_order(group_df):
            sub, x = embedding_matrix_for_cost(group_df, cost_type)
            coords = pca_2d(x)
            reference_mask = (
                (sub["target_group"] == "clean")
                & (sub["target_trial_id"].astype(str) == reference_trial_id)
            )
            if not reference_mask.any():
                raise ValueError(f"No reference clean rows found for {result_group}, {cost_type}, {reference_trial_id}")
            origin = coords[reference_mask.to_numpy()].mean(axis=0)
            coords = coords - origin
            out = sub[
                [
                    "result_group",
                    "device_id",
                    "target_trial_id",
                    "target_run_id",
                    "target_group",
                    "poisoning_type",
                    "condition",
                    "trial_index",
                    "cost_type",
                ]
            ].copy()
            out["pca_x"] = coords[:, 0]
            out["pca_y"] = coords[:, 1]
            out["is_reference"] = reference_mask.to_numpy()
            out["pca_scope"] = "per_result_group"
            rows.append(out)
    return pd.concat(rows, ignore_index=True)


def compute_pca_coordinates_from_result_dirs(
    result_dirs: Sequence[Path],
    input_dir: Path,
    reference_trial_id: str,
) -> pd.DataFrame:
    """Compute PCA coordinates without concatenating wide embedding CSVs."""
    rows = []
    for result_dir in result_dirs:
        group = result_group_label(result_dir / EMBEDDING_NAME, input_dir)
        device_id = group.split("/")[0]
        print(f"Computing embedding PCA for {group}")
        df = pd.read_csv(result_dir / EMBEDDING_NAME)
        for cost_type in cost_types_in_order(df):
            sub = df[df["cost_type"] == cost_type].copy()
            emb_cols = [column for column in sub.columns if column.startswith("emb_")]
            xdf = sub[emb_cols].apply(pd.to_numeric, errors="coerce")
            xdf = xdf.dropna(axis=1, how="all")
            x = xdf.to_numpy(dtype=np.float64)
            if not np.isfinite(x).all():
                raise ValueError(f"Embedding matrix for {group}, {cost_type} contains NaN/inf.")
            coords = pca_2d(x)
            reference_mask = (
                (sub["target_group"] == "clean")
                & (sub["target_trial_id"].astype(str) == reference_trial_id)
            )
            if not reference_mask.any():
                raise ValueError(f"No reference clean rows found for {group}, {cost_type}, {reference_trial_id}")
            origin = coords[reference_mask.to_numpy()].mean(axis=0)
            coords = coords - origin
            out = sub[
                [
                    "target_trial_id",
                    "target_run_id",
                    "target_group",
                    "poisoning_type",
                    "cost_type",
                ]
            ].copy()
            out["result_group"] = group
            out["device_id"] = device_id
            out["trial_index"] = out["target_trial_id"].map(parse_trial_id)
            out["condition"] = np.where(out["target_group"] == "clean", "clean", out["poisoning_type"])
            out["pca_x"] = coords[:, 0]
            out["pca_y"] = coords[:, 1]
            out["is_reference"] = reference_mask.to_numpy()
            out["pca_scope"] = "per_result_group"
            rows.append(out)
    return pd.concat(rows, ignore_index=True)


def compute_pca_coordinates_from_summary(summary: pd.DataFrame, reference_trial_id: str = "") -> pd.DataFrame:
    """Use precomputed embedding PCA coordinates from OT summary and shift reference to zero."""
    required = {"pca_x", "pca_y", "result_group", "cost_type", "target_group", "target_trial_id", "metric_name"}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"Summary is missing PCA columns: {missing}")

    rows = []
    group_cols = ["result_group", "metric_name", "cost_type", "reference_run_id"]
    if {"segment_type", "segment_id"}.issubset(summary.columns):
        group_cols.extend(["segment_type", "segment_id"])
    for group_key, sub in summary.groupby(
        group_cols,
        sort=False,
    ):
        result_group, metric_name, cost_type, reference_run_id = group_key[:4]
        sub = sub.copy()
        origin_trial_id = reference_trial_id or str(sub["reference_trial_id"].iloc[0])
        reference = sub[
            (sub["target_group"] == "clean")
            & (sub["target_trial_id"].astype(str) == origin_trial_id)
        ]
        if reference.empty:
            reference_trial = str(sub["reference_trial_id"].iloc[0])
            reference = sub[
                (sub["target_group"] == "clean")
                & (sub["target_trial_id"].astype(str) == reference_trial)
            ]
            origin_trial_id = reference_trial
        if reference.empty:
            raise ValueError(f"No reference clean row found for {result_group}, {cost_type}, {reference_run_id}")
        origin = reference[["pca_x", "pca_y"]].mean().to_numpy(dtype=np.float64)
        sub["pca_x"] = pd.to_numeric(sub["pca_x"], errors="raise") - origin[0]
        sub["pca_y"] = pd.to_numeric(sub["pca_y"], errors="raise") - origin[1]
        sub["is_reference"] = (
            (sub["target_group"] == "clean")
            & (sub["target_trial_id"].astype(str) == origin_trial_id)
        )
        sub["pca_scope"] = "precomputed_per_result_group"
        rows.append(
            sub[
                [
                    "result_group",
                    "device_id",
                    "target_trial_id",
                    "target_run_id",
                    "target_group",
                    "poisoning_type",
                    "condition",
                    "trial_index",
                    "segment_type",
                    "segment_id",
                    "segment_index",
                    "metric_name",
                    "cost_type",
                    "pca_x",
                    "pca_y",
                    "is_reference",
                    "pca_scope",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True)


def plot_embedding_pca(coords: pd.DataFrame, output_dir: Path) -> None:
    for metric_name in metric_names_in_order(coords):
        metric_df = coords[coords["metric_name"] == metric_name].copy()
        cost_types = cost_types_in_order(metric_df)
        fig, axes = make_row_axes(len(cost_types), width=8.5, row_height=3.2)

        for ax, cost_type in zip(axes, cost_types):
            sub = metric_df[metric_df["cost_type"] == cost_type].copy()
            for condition, cdf in sub.groupby("condition", sort=False):
                ax.scatter(
                    cdf["pca_x"],
                    cdf["pca_y"],
                    s=38,
                    alpha=0.72,
                    marker=marker_for_condition(condition),
                    color=color_for_condition(condition),
                    label=condition,
                )
            ax.scatter(
                [0.0],
                [0.0],
                s=150,
                marker="*",
                color="black",
                label="global reference",
                zorder=5,
            )
            for _, row in sub.iterrows():
                ax.annotate(str(row["trial_index"]), (row["pca_x"], row["pca_y"]), fontsize=7, alpha=0.65)
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
            ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.35)
            ax.set_title(f"{metric_name} / {cost_type} embedding PCA, reference shifted to 0")
            ax.set_xlabel("PC1, shifted so reference is 0")
            ax.set_ylabel("PC2, shifted so reference is 0")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")

        fig.tight_layout()
        suffix = safe_name(metric_name)
        fig.savefig(output_dir / f"embedding_pca_by_cost_{suffix}.png", dpi=180)
        fig.savefig(output_dir / f"embedding_pca_by_cost_{suffix}.pdf")
        close_figure(fig)


def make_row_axes(n_rows: int, width: float, row_height: float):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(n_rows, 1, figsize=(width, row_height * n_rows), squeeze=False)
    return fig, list(axes[:, 0])


def close_figure(fig) -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="result")
    parser.add_argument("--output_dir", default="visualization_result")
    parser.add_argument("--reference_trial_id", default="")
    parser.add_argument("--ot_y", default="ot_cost", choices=["ot_cost", "z_ot_cost"])
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib_cache"))

    result_dirs = discover_result_dirs(input_dir)
    print(f"Found {len(result_dirs)} result directories")
    summary = load_summary(result_dirs, input_dir)

    plot_ot_distance(summary, output_dir, y_column=args.ot_y)

    coords = compute_pca_coordinates_from_summary(summary, reference_trial_id=args.reference_trial_id)
    plot_embedding_pca(coords, output_dir)

    print(f"Saved visualization outputs to {output_dir}")


if __name__ == "__main__":
    main()
