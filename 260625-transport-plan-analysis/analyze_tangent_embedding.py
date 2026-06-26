#!/usr/bin/env python3
"""Wasserstein tangent embedding analysis for time-series run metrics.

The core API accepts already-loaded runs:

    clean_runs: list[np.ndarray]  # each (n_i, d)
    test_runs: list[np.ndarray]   # each (m_i, d)
    test_labels: list[str]

It also provides a small CLI for the CSV logs in this analysis directory.
POT (`ot`) and scikit-learn are used when installed. Lightweight local
fallbacks are included so the pipeline remains executable in minimal
environments.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "results" / ".mplconfig"))

import matplotlib.pyplot as plt

try:  # Prefer POT when available.
    import ot as pot
except Exception:  # pragma: no cover - depends on local environment.
    pot = None

try:  # Prefer sklearn when available.
    from sklearn.decomposition import PCA as SklearnPCA
    from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
except Exception:  # pragma: no cover - depends on local environment.
    SklearnPCA = None
    SklearnStandardScaler = None


EPS = 1e-12
DEFAULT_WINDOW_SIZE = 5
CORE_COLS = ["core_0", "core_1", "core_2", "core_3"]


class SimpleStandardScaler:
    """Small StandardScaler fallback with the sklearn methods used here."""

    def fit(self, x: np.ndarray) -> "SimpleStandardScaler":
        x = np.asarray(x, dtype=float)
        self.mean_ = np.nanmean(x, axis=0)
        self.scale_ = np.nanstd(x, axis=0)
        self.scale_ = np.where(self.scale_ > EPS, self.scale_, 1.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=float) - self.mean_) / self.scale_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


class SimplePCA:
    """Small PCA fallback with transform/inverse_transform."""

    def __init__(self, n_components: int | None = None) -> None:
        self.n_components = n_components

    def fit(self, x: np.ndarray) -> "SimplePCA":
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError(f"PCA input must be 2D, got shape {x.shape}")
        self.mean_ = np.mean(x, axis=0)
        centered = x - self.mean_
        max_components = min(centered.shape)
        n_components = max_components if self.n_components is None else int(self.n_components)
        n_components = max(0, min(n_components, max_components))
        if n_components == 0 or centered.size == 0:
            self.components_ = np.zeros((0, x.shape[1]), dtype=float)
            self.explained_variance_ratio_ = np.zeros(0, dtype=float)
            return self
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        self.components_ = vt[:n_components]
        variances = (singular_values * singular_values) / max(1, x.shape[0] - 1)
        total = float(np.sum(variances))
        if total > 0:
            self.explained_variance_ratio_ = variances[:n_components] / total
        else:
            self.explained_variance_ratio_ = np.zeros(n_components, dtype=float)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return (x - self.mean_) @ self.components_.T

    def inverse_transform(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        return scores @ self.components_ + self.mean_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


@dataclass
class OTDisplacementResult:
    plan: np.ndarray
    cost_matrix: np.ndarray
    ot_cost: float
    barycentric_projection: np.ndarray
    displacement: np.ndarray
    embedding: np.ndarray
    reg: float


@dataclass
class TangentEmbeddingResult:
    feature_type: str
    z_normalize_window: bool
    scaler: Any
    pca: Any
    reference_features: np.ndarray
    clean_embeddings: np.ndarray
    test_embeddings: np.ndarray
    clean_scores: pd.DataFrame
    test_scores: pd.DataFrame
    all_scores: pd.DataFrame
    clean_ot: list[OTDisplacementResult]
    test_ot: list[OTDisplacementResult]


def _as_2d_run(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"Each run must be 1D or 2D, got shape {arr.shape}")
    if len(arr) == 0:
        raise ValueError("Runs must contain at least one point.")
    return arr


def normalized_time(n_points: int) -> np.ndarray:
    if n_points <= 1:
        return np.zeros((n_points, 1), dtype=float)
    return np.linspace(0.0, 1.0, n_points, dtype=float)[:, None]


def make_delta_features(x: np.ndarray) -> np.ndarray:
    """Create z_i = [x_i, delta x_i, tau_i] for one variable-length run."""

    arr = _as_2d_run(x)
    delta = np.zeros_like(arr)
    if len(arr) > 1:
        delta[1:] = arr[1:] - arr[:-1]
    tau = normalized_time(len(arr))
    return np.concatenate([arr, delta, tau], axis=1)


def make_window_features(
    x: np.ndarray,
    window_size: int = DEFAULT_WINDOW_SIZE,
    z_normalize_window: bool = False,
) -> np.ndarray:
    """Create local-window features [x_{i-w}, ..., x_i, ..., x_{i+w}, tau_i]."""

    arr = _as_2d_run(x)
    w = max(0, int(window_size))
    if w == 0:
        windows = arr[:, None, :]
    else:
        padded = np.pad(arr, ((w, w), (0, 0)), mode="edge")
        width = 2 * w + 1
        windows = np.stack([padded[i : i + width] for i in range(len(arr))], axis=0)

    if z_normalize_window:
        mean = windows.mean(axis=1, keepdims=True)
        std = windows.std(axis=1, keepdims=True)
        std = np.where(std > EPS, std, 1.0)
        windows = (windows - mean) / std

    flat = windows.reshape(len(arr), -1)
    tau = normalized_time(len(arr))
    return np.concatenate([flat, tau], axis=1)


def make_features(
    x: np.ndarray,
    feature_type: str,
    window_size: int = DEFAULT_WINDOW_SIZE,
    z_normalize_window: bool = False,
) -> np.ndarray:
    if feature_type == "delta":
        return make_delta_features(x)
    if feature_type == "window":
        return make_window_features(
            x,
            window_size=window_size,
            z_normalize_window=z_normalize_window,
        )
    raise ValueError(f"Unsupported feature_type: {feature_type}")


def fit_feature_scaler(clean_feature_runs: list[np.ndarray]) -> Any:
    """Fit StandardScaler on clean feature points only."""

    all_clean_points = np.vstack(clean_feature_runs)
    if SklearnStandardScaler is not None:
        return SklearnStandardScaler().fit(all_clean_points)
    return SimpleStandardScaler().fit(all_clean_points)


def squared_euclidean_cost(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - y[None, :, :]
    return np.sum(diff * diff, axis=2)


def uniform_weights(n_points: int) -> np.ndarray:
    if n_points <= 0:
        raise ValueError("Cannot build weights for empty support.")
    return np.full(n_points, 1.0 / n_points, dtype=float)


def choose_sinkhorn_reg(cost_matrix: np.ndarray, reg_scale: float) -> float:
    positive = cost_matrix[np.isfinite(cost_matrix) & (cost_matrix > 0)]
    if positive.size == 0:
        return 1e-3
    return max(float(np.median(positive)) * float(reg_scale), 1e-6)


def sinkhorn_fallback(
    a: np.ndarray,
    b: np.ndarray,
    cost_matrix: np.ndarray,
    reg: float,
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> np.ndarray:
    """Numerically simple entropic OT fallback used when POT is unavailable."""

    reg = max(float(reg), 1e-9)
    scaled = -cost_matrix / reg
    scaled = scaled - np.max(scaled)
    k_mat = np.exp(scaled)
    k_mat = np.maximum(k_mat, 1e-300)

    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(int(max_iter)):
        prev = u.copy()
        kv = np.maximum(k_mat @ v, 1e-300)
        u = a / kv
        ktu = np.maximum(k_mat.T @ u, 1e-300)
        v = b / ktu
        if np.max(np.abs(u - prev)) <= tol:
            break

    plan = (u[:, None] * k_mat) * v[None, :]
    total = float(np.sum(plan))
    if total > 0:
        plan /= total
    return plan


def compute_ot_displacement(
    reference_features: np.ndarray,
    target_features: np.ndarray,
    method: str = "sinkhorn",
    reg: float | None = None,
    reg_scale: float = 0.05,
    max_iter: int = 1000,
) -> OTDisplacementResult:
    """Compute OT plan, barycentric projection, and flattened displacement.

    OT is from fixed reference support q_a to target support y_b.
    POT is used when installed; otherwise `sinkhorn_fallback` is used for the
    default entropic method.
    """

    q = _as_2d_run(reference_features)
    y = _as_2d_run(target_features)
    a = uniform_weights(len(q))
    b = uniform_weights(len(y))
    cost_matrix = squared_euclidean_cost(q, y)
    chosen_reg = choose_sinkhorn_reg(cost_matrix, reg_scale) if reg is None else float(reg)

    method = str(method).lower()
    if pot is not None:
        if method == "sinkhorn":
            plan = pot.sinkhorn(a, b, cost_matrix, reg=chosen_reg, numItermax=int(max_iter))
        elif method == "emd":
            plan = pot.emd(a, b, cost_matrix)
        else:
            raise ValueError(f"Unsupported OT method: {method}")
    else:
        if method != "sinkhorn":
            raise ImportError("POT is required for method='emd'. Install with `pip install pot`.")
        plan = sinkhorn_fallback(a, b, cost_matrix, reg=chosen_reg, max_iter=int(max_iter))

    row_mass = np.sum(plan, axis=1)
    safe_mass = np.where(row_mass > EPS, row_mass, 1.0)
    barycentric_projection = (plan @ y) / safe_mass[:, None]
    barycentric_projection[row_mass <= EPS] = q[row_mass <= EPS]
    displacement = barycentric_projection - q
    ot_cost = float(np.sum(plan * cost_matrix))

    return OTDisplacementResult(
        plan=np.asarray(plan, dtype=float),
        cost_matrix=cost_matrix,
        ot_cost=ot_cost,
        barycentric_projection=barycentric_projection,
        displacement=displacement,
        embedding=displacement.ravel(),
        reg=chosen_reg,
    )


def make_tangent_embeddings(
    clean_runs: list[np.ndarray],
    test_runs: list[np.ndarray],
    feature_type: str = "delta",
    window_size: int = DEFAULT_WINDOW_SIZE,
    z_normalize_window: bool = False,
    ot_method: str = "sinkhorn",
    reg: float | None = None,
    reg_scale: float = 0.05,
    max_iter: int = 1000,
    reference_index: int = 0,
    exclude_reference_from_clean_calibration: bool = True,
) -> dict[str, Any]:
    """Build scaled features and tangent embeddings for clean/test runs."""

    if not clean_runs:
        raise ValueError("clean_runs must contain at least one run.")
    if reference_index < 0 or reference_index >= len(clean_runs):
        raise IndexError(f"reference_index {reference_index} outside clean_runs.")

    clean_features = [
        make_features(run, feature_type, window_size, z_normalize_window)
        for run in clean_runs
    ]
    test_features = [
        make_features(run, feature_type, window_size, z_normalize_window)
        for run in test_runs
    ]

    scaler = fit_feature_scaler(clean_features)
    clean_scaled = [scaler.transform(features) for features in clean_features]
    test_scaled = [scaler.transform(features) for features in test_features]
    reference_features = clean_scaled[reference_index]

    clean_indices = list(range(len(clean_scaled)))
    if exclude_reference_from_clean_calibration and len(clean_indices) > 1:
        clean_indices = [idx for idx in clean_indices if idx != reference_index]

    clean_ot = [
        compute_ot_displacement(
            reference_features,
            clean_scaled[idx],
            method=ot_method,
            reg=reg,
            reg_scale=reg_scale,
            max_iter=max_iter,
        )
        for idx in clean_indices
    ]
    test_ot = [
        compute_ot_displacement(
            reference_features,
            features,
            method=ot_method,
            reg=reg,
            reg_scale=reg_scale,
            max_iter=max_iter,
        )
        for features in test_scaled
    ]

    clean_embeddings = np.vstack([item.embedding for item in clean_ot]) if clean_ot else np.zeros((0, reference_features.size))
    test_embeddings = np.vstack([item.embedding for item in test_ot]) if test_ot else np.zeros((0, reference_features.size))

    return {
        "scaler": scaler,
        "reference_features": reference_features,
        "clean_indices": clean_indices,
        "clean_embeddings": clean_embeddings,
        "test_embeddings": test_embeddings,
        "clean_ot": clean_ot,
        "test_ot": test_ot,
        "feature_type": feature_type,
        "z_normalize_window": z_normalize_window,
    }


def fit_clean_pca(
    clean_embeddings: np.ndarray,
    n_components: int | None = None,
) -> Any:
    """Fit PCA on clean tangent embeddings only."""

    clean_embeddings = np.asarray(clean_embeddings, dtype=float)
    if clean_embeddings.ndim != 2:
        raise ValueError(f"clean_embeddings must be 2D, got {clean_embeddings.shape}")
    if clean_embeddings.shape[0] == 0:
        raise ValueError("Need at least one clean tangent embedding for PCA.")

    if n_components is None:
        n_components = min(2, max(1, clean_embeddings.shape[0] - 1), clean_embeddings.shape[1])
    else:
        n_components = min(max(0, int(n_components)), clean_embeddings.shape[0], clean_embeddings.shape[1])

    if SklearnPCA is not None:
        return SklearnPCA(n_components=n_components).fit(clean_embeddings)
    return SimplePCA(n_components=n_components).fit(clean_embeddings)


def compute_residual_scores(
    embeddings: np.ndarray,
    pca: Any,
    ot_costs: Iterable[float],
    labels: Iterable[str],
    run_ids: Iterable[str] | None = None,
    eps: float = EPS,
) -> pd.DataFrame:
    """Compute tangent norm, residual norm, and residual ratio."""

    embeddings = np.asarray(embeddings, dtype=float)
    if embeddings.ndim == 1:
        embeddings = embeddings[None, :]
    reconstructed = pca.inverse_transform(pca.transform(embeddings))
    residual = embeddings - reconstructed
    tangent_norm = np.linalg.norm(embeddings, axis=1)
    residual_norm = np.linalg.norm(residual, axis=1)
    residual_ratio = residual_norm / (tangent_norm + float(eps))

    labels = list(labels)
    ot_costs = list(ot_costs)
    if run_ids is None:
        run_ids = [f"run_{idx}" for idx in range(len(labels))]
    else:
        run_ids = list(run_ids)

    return pd.DataFrame(
        {
            "run_id": run_ids,
            "label": labels,
            "ot_cost": ot_costs,
            "tangent_norm": tangent_norm,
            "residual_norm": residual_norm,
            "residual_ratio": residual_ratio,
        }
    )


def pca_coordinates(pca: Any, embeddings: np.ndarray, n_dims: int = 2) -> np.ndarray:
    coords = pca.transform(np.asarray(embeddings, dtype=float))
    if coords.shape[1] >= n_dims:
        return coords[:, :n_dims]
    padded = np.zeros((coords.shape[0], n_dims), dtype=float)
    padded[:, : coords.shape[1]] = coords
    return padded


def run_tangent_space_analysis(
    clean_runs: list[np.ndarray],
    test_runs: list[np.ndarray],
    test_labels: list[str],
    feature_type: str = "delta",
    window_size: int = DEFAULT_WINDOW_SIZE,
    z_normalize_window: bool = False,
    pca_components: int | None = None,
    ot_method: str = "sinkhorn",
    reg: float | None = None,
    reg_scale: float = 0.05,
    max_iter: int = 1000,
    output_dir: Path | None = None,
    run_ids: list[str] | None = None,
    clean_run_ids: list[str] | None = None,
) -> TangentEmbeddingResult:
    """End-to-end tangent-space analysis for one feature configuration."""

    if len(test_runs) != len(test_labels):
        raise ValueError("test_runs and test_labels must have the same length.")
    embedding_data = make_tangent_embeddings(
        clean_runs=clean_runs,
        test_runs=test_runs,
        feature_type=feature_type,
        window_size=window_size,
        z_normalize_window=z_normalize_window,
        ot_method=ot_method,
        reg=reg,
        reg_scale=reg_scale,
        max_iter=max_iter,
    )
    pca = fit_clean_pca(embedding_data["clean_embeddings"], n_components=pca_components)

    clean_labels = ["clean_calibration"] * len(embedding_data["clean_ot"])
    if clean_run_ids is None:
        clean_run_ids = [f"clean_{idx}" for idx in embedding_data["clean_indices"]]
    else:
        clean_run_ids = [clean_run_ids[idx] for idx in embedding_data["clean_indices"]]

    clean_scores = compute_residual_scores(
        embedding_data["clean_embeddings"],
        pca,
        [item.ot_cost for item in embedding_data["clean_ot"]],
        clean_labels,
        run_ids=clean_run_ids,
    )
    test_scores = compute_residual_scores(
        embedding_data["test_embeddings"],
        pca,
        [item.ot_cost for item in embedding_data["test_ot"]],
        test_labels,
        run_ids=run_ids,
    )
    all_scores = pd.concat([clean_scores, test_scores], ignore_index=True)
    all_scores["feature_type"] = feature_type
    all_scores["z_normalize_window"] = bool(z_normalize_window)

    result = TangentEmbeddingResult(
        feature_type=feature_type,
        z_normalize_window=bool(z_normalize_window),
        scaler=embedding_data["scaler"],
        pca=pca,
        reference_features=embedding_data["reference_features"],
        clean_embeddings=embedding_data["clean_embeddings"],
        test_embeddings=embedding_data["test_embeddings"],
        clean_scores=clean_scores,
        test_scores=test_scores,
        all_scores=all_scores,
        clean_ot=embedding_data["clean_ot"],
        test_ot=embedding_data["test_ot"],
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        all_scores.to_csv(output_dir / "tangent_scores.csv", index=False)
        plot_ot_vs_residual(all_scores, output_dir / "ot_cost_vs_residual_norm.png")
        plot_residual_ratio_by_label(all_scores, output_dir / "residual_ratio_by_label.png")
        all_embeddings = np.vstack([result.clean_embeddings, result.test_embeddings])
        plot_pca_scatter(
            pca=pca,
            embeddings=all_embeddings,
            labels=all_scores["label"].tolist(),
            output_path=output_dir / "tangent_pca_scatter.png",
        )
        plot_displacement_over_reference_index(
            result=result,
            output_path=output_dir / "selected_displacements_over_reference_index.png",
        )

    return result


def label_colors(labels: list[str]) -> dict[str, Any]:
    unique = list(dict.fromkeys(labels))
    cmap = plt.get_cmap("tab10")
    return {label: cmap(idx % 10) for idx, label in enumerate(unique)}


def plot_ot_vs_residual(scores: pd.DataFrame, output_path: Path) -> None:
    colors = label_colors(scores["label"].tolist())
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, group in scores.groupby("label"):
        ax.scatter(group["ot_cost"], group["residual_norm"], label=label, color=colors[label], alpha=0.8)
    ax.set_xlabel("OT cost")
    ax.set_ylabel("Residual norm")
    ax.set_title("OT distance vs tangent residual")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_residual_ratio_by_label(scores: pd.DataFrame, output_path: Path) -> None:
    labels = list(dict.fromkeys(scores["label"].tolist()))
    data = [scores.loc[scores["label"] == label, "residual_ratio"].to_numpy() for label in labels]
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(labels)), 5))
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_xlabel("Label")
    ax.set_ylabel("Residual ratio")
    ax.set_title("Residual ratio by label")
    ax.grid(True, axis="y", alpha=0.25)
    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_pca_scatter(
    pca: Any,
    embeddings: np.ndarray,
    labels: list[str],
    output_path: Path,
) -> None:
    coords = pca_coordinates(pca, embeddings, n_dims=2)
    colors = label_colors(labels)
    fig, ax = plt.subplots(figsize=(7, 5))
    for label in dict.fromkeys(labels):
        mask = np.array(labels) == label
        ax.scatter(coords[mask, 0], coords[mask, 1], label=label, color=colors[label], alpha=0.8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Tangent embedding PCA")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_displacement_over_reference_index(
    result: TangentEmbeddingResult,
    output_path: Path,
    max_runs: int = 6,
) -> None:
    selected = result.test_ot[:max_runs]
    if not selected:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, item in enumerate(selected):
        per_ref_norm = np.linalg.norm(item.displacement, axis=1)
        ax.plot(per_ref_norm, linewidth=1.5, label=f"test_{idx}")
    ax.set_xlabel("Reference support index")
    ax.set_ylabel("Displacement norm")
    ax.set_title("Barycentric displacement over reference support")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_comparison_experiments(
    clean_runs: list[np.ndarray],
    test_runs: list[np.ndarray],
    test_labels: list[str],
    output_dir: Path,
    window_size: int = DEFAULT_WINDOW_SIZE,
    pca_components: int | None = None,
    ot_method: str = "sinkhorn",
    reg_scale: float = 0.05,
    max_iter: int = 1000,
    run_ids: list[str] | None = None,
    clean_run_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Run delta/window/window-z experiments and save comparable summaries."""

    configs = [
        ("delta", False, "delta"),
        ("window", False, "window_raw"),
        ("window", True, "window_z_normalized"),
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for feature_type, z_norm, name in configs:
        result = run_tangent_space_analysis(
            clean_runs=clean_runs,
            test_runs=test_runs,
            test_labels=test_labels,
            feature_type=feature_type,
            window_size=window_size,
            z_normalize_window=z_norm,
            pca_components=pca_components,
            ot_method=ot_method,
            reg_scale=reg_scale,
            max_iter=max_iter,
            output_dir=output_dir / name,
            run_ids=run_ids,
            clean_run_ids=clean_run_ids,
        )
        frame = result.all_scores.copy()
        frame["experiment"] = name
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output_dir / "tangent_embedding_comparison_scores.csv", index=False)
    summary = (
        combined.groupby(["experiment", "label"])
        .agg(
            ot_cost_mean=("ot_cost", "mean"),
            ot_cost_median=("ot_cost", "median"),
            residual_norm_mean=("residual_norm", "mean"),
            residual_norm_median=("residual_norm", "median"),
            residual_ratio_mean=("residual_ratio", "mean"),
            residual_ratio_median=("residual_ratio", "median"),
            n=("run_id", "size"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "tangent_embedding_comparison_summary.csv", index=False)
    plot_experiment_metric_comparison(
        combined,
        metric="residual_ratio",
        output_path=output_dir / "comparison_residual_ratio_by_label.png",
    )
    plot_experiment_metric_comparison(
        combined,
        metric="residual_norm",
        output_path=output_dir / "comparison_residual_norm_by_label.png",
    )
    plot_experiment_metric_comparison(
        combined,
        metric="ot_cost",
        output_path=output_dir / "comparison_ot_cost_by_label.png",
    )
    return combined


def plot_experiment_metric_comparison(scores: pd.DataFrame, metric: str, output_path: Path) -> None:
    labels = list(dict.fromkeys(scores["label"].tolist()))
    experiments = list(dict.fromkeys(scores["experiment"].tolist()))
    x = np.arange(len(labels))
    width = 0.8 / max(1, len(experiments))
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(labels)), 5))
    for idx, experiment in enumerate(experiments):
        means = [
            scores.loc[(scores["experiment"] == experiment) & (scores["label"] == label), metric].mean()
            for label in labels
        ]
        ax.bar(x + (idx - (len(experiments) - 1) / 2) * width, means, width=width, label=experiment)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by label and feature experiment")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_core_list(value: object) -> list[float] | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    if isinstance(parsed, (list, tuple)):
        return [float(v) for v in parsed]
    return None


def load_cpu_core_run(csv_path: Path, n_cores: int = 4) -> np.ndarray:
    """Load sorted per-core CPU usage from one ml_running CSV."""

    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            values = parse_core_list(row.get("cpu_per_core"))
            if values is None or len(values) == 0:
                continue
            values = sorted(values, reverse=True)
            if len(values) < n_cores:
                values = values + [float("nan")] * (n_cores - len(values))
            rows.append(values[:n_cores])
    arr = np.asarray(rows, dtype=float)
    arr = arr[np.isfinite(arr).all(axis=1)]
    if len(arr) == 0:
        raise ValueError(f"No valid cpu_per_core rows in {csv_path}")
    return arr


def resample_run(x: np.ndarray, max_points: int) -> np.ndarray:
    arr = _as_2d_run(x)
    if max_points <= 0 or len(arr) <= max_points:
        return arr
    idx = np.unique(np.linspace(0, len(arr) - 1, max_points).round().astype(int))
    return arr[idx]


def load_runs_from_dir(directory: Path, max_runs: int = 0, max_points: int = 0) -> tuple[list[np.ndarray], list[str]]:
    files = sorted(directory.glob("*.csv"))
    if max_runs > 0:
        files = files[:max_runs]
    runs = [resample_run(load_cpu_core_run(path), max_points=max_points) for path in files]
    run_ids = [path.stem for path in files]
    return runs, run_ids


def parse_test_dir_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        path = Path(spec)
        return path.name, path
    label, path = spec.split("=", 1)
    return label.strip(), Path(path).expanduser()


def _self_test() -> None:
    rng = np.random.default_rng(7)
    clean_runs = [rng.normal(0, 1, size=(40, 4)).cumsum(axis=0) * 0.05 + 50 for _ in range(4)]
    test_runs = [
        rng.normal(0, 1, size=(42, 4)).cumsum(axis=0) * 0.05 + 50,
        rng.normal(0, 1, size=(42, 4)).cumsum(axis=0) * 0.12 + np.array([50, 53, 47, 51]),
    ]
    labels = ["clean_like", "shifted"]
    result = run_tangent_space_analysis(
        clean_runs=clean_runs,
        test_runs=test_runs,
        test_labels=labels,
        feature_type="delta",
        pca_components=2,
    )
    print(result.test_scores.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--clean-dir", type=Path, default=None)
    parser.add_argument(
        "--test-dir",
        action="append",
        default=[],
        help="Label and directory as label=/path/to/csv_dir. May be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--pca-components", type=int, default=None)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--max-points", type=int, default=250)
    parser.add_argument("--reg-scale", type=float, default=0.05)
    parser.add_argument("--ot-method", choices=["sinkhorn", "emd"], default="sinkhorn")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return

    analysis_root = args.analysis_root.resolve()
    clean_dir = args.clean_dir.resolve() if args.clean_dir else analysis_root / "logs" / "baseclean"
    output_dir = args.output_dir.resolve() if args.output_dir else analysis_root / "results" / "tangent_embedding"
    test_specs = args.test_dir or [f"baseblurring={analysis_root / 'logs' / 'baseblurring'}"]

    clean_runs, clean_ids = load_runs_from_dir(clean_dir, max_runs=args.max_runs, max_points=args.max_points)
    test_runs: list[np.ndarray] = []
    test_labels: list[str] = []
    test_ids: list[str] = []
    for spec in test_specs:
        label, directory = parse_test_dir_spec(spec)
        runs, ids = load_runs_from_dir(directory.resolve(), max_runs=args.max_runs, max_points=args.max_points)
        test_runs.extend(runs)
        test_labels.extend([label] * len(runs))
        test_ids.extend([f"{label}_{run_id}" for run_id in ids])

    combined = run_comparison_experiments(
        clean_runs=clean_runs,
        test_runs=test_runs,
        test_labels=test_labels,
        output_dir=output_dir,
        window_size=args.window_size,
        pca_components=args.pca_components,
        ot_method=args.ot_method,
        reg_scale=args.reg_scale,
        run_ids=test_ids,
        clean_run_ids=clean_ids,
    )
    print(f"Wrote tangent embedding outputs to: {output_dir}")
    print(
        combined.groupby(["experiment", "label"])
        [["ot_cost", "residual_norm", "residual_ratio"]]
        .mean()
        .reset_index()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
