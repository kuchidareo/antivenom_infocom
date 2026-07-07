#!/usr/bin/env python3
"""Cross-device OT comparison for local-ML hardware traces.

This script compares device classes instead of comparing attacks within one
device. For each device-type pair, it computes OT distances and tangent
embeddings for:

    clean -> clean
    clean -> poisoned
    poisoned -> clean
    poisoned -> poisoned

The poisoned cases are evaluated for both availability attacks:
unlearnable_examples and availability_shortcuts. Plots keep those two attacks
inside the same figure for each of the 12 high-level conditions:

    3 device pairs x 4 clean/poison direction conditions.

The implementation reuses the feature builders and POT-based OT solver from
1_calculate_ot_embedding.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
OT_MODULE_PATH = SCRIPT_DIR / "1_calculate_ot_embedding.py"


def _load_ot_module() -> Any:
    spec = importlib.util.spec_from_file_location("ot_embedding_base", OT_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load OT helper module: {OT_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ot_base = _load_ot_module()


DEFAULT_DEVICE_GROUPS = {
    "rpi4": ["192.168.0.116"],
    "rpi3": ["192.168.0.115"],
    "jetson_cpu": ["192.168.0.141"],
}

ALL_DEVICE_GROUPS = {
    "rpi4": [
        "192.168.0.116",
        "192.168.0.117",
        "192.168.0.118",
        "192.168.0.119",
        "192.168.0.120",
        "192.168.0.121",
    ],
    "rpi3": ["192.168.0.115"],
    "jetson_cpu": ["192.168.0.141", "192.168.0.142"],
}

DEVICE_TYPE_PAIRS = [
    ("rpi4", "rpi3"),
    ("rpi4", "jetson_cpu"),
    ("rpi3", "jetson_cpu"),
]

CONDITION_PAIRS = [
    ("clean", "clean"),
    ("clean", "poisoned"),
    ("poisoned", "clean"),
    ("poisoned", "poisoned"),
]
ATTACK_RELATED_CONDITIONS = {
    "clean_to_poisoned",
    "poisoned_to_clean",
    "poisoned_to_poisoned",
}

DEFAULT_ATTACKS = ["unlearnable_examples", "availability_shortcuts"]
DEFAULT_TRIALS = [f"trial_{idx}" for idx in range(5)]
DEFAULT_FEATURE_COLUMNS = [
    "system_cpu_core_1",
    "system_cpu_core_2",
    "system_cpu_core_3",
]
DEFAULT_COST_TYPES = [
    "c2_value",
    "c2_value_shape",
    "c3_window_shape",
]


@dataclass(frozen=True)
class RunRecord:
    device: str
    device_group: str
    trial_id: str
    poisoning_method: str
    segment_type: str
    segment_id: str
    values: np.ndarray
    source_path: str
    original_length: int


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def is_hardware_csv(path: Path) -> bool:
    return (
        path.suffix == ".csv"
        and not path.name.endswith("_metrics.csv")
        and path.name
        not in {
            "ot_embedding_summary.csv",
            "ot_embedding_summary_zscored.csv",
            "clean_baseline_stats.csv",
            "cross_device_ot_summary.csv",
            "cross_device_embedding_pca.csv",
        }
    )


def normalized_trial_id(value: Any) -> str:
    text = str(value)
    return text if text.startswith(("trial_", "reference_")) else f"trial_{text}"


def condition_name(ref_condition: str, target_condition: str) -> str:
    return f"{ref_condition}_to_{target_condition}"


def safe_name(value: str) -> str:
    return (
        str(value)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("->", "_to_")
        .replace(":", "_")
    )


def fixed_window_average(values: np.ndarray, num_bins: int) -> np.ndarray:
    """Convert a variable-length trace to fixed bins by window averaging.

    OT itself can handle variable support sizes, but tangent PCA needs a fixed
    embedding dimension. The binning is therefore explicit and controlled by
    --num_bins. If a segment has fewer samples than bins, linear interpolation
    is used to avoid empty windows.
    """
    x = np.asarray(values, dtype=np.float32)
    if num_bins <= 0:
        return x
    if x.shape[0] == num_bins:
        return x
    if x.shape[0] < 2:
        return np.repeat(x, num_bins, axis=0)
    if x.shape[0] < num_bins:
        old_pos = np.linspace(0.0, 1.0, x.shape[0], dtype=np.float32)
        new_pos = np.linspace(0.0, 1.0, num_bins, dtype=np.float32)
        out = np.empty((num_bins, x.shape[1]), dtype=np.float32)
        for col_idx in range(x.shape[1]):
            out[:, col_idx] = np.interp(new_pos, old_pos, x[:, col_idx])
        return out

    indices = np.array_split(np.arange(x.shape[0]), num_bins)
    return np.vstack([x[idx].mean(axis=0) for idx in indices]).astype(np.float32)


def segment_column_for_df(path: Path, df: pd.DataFrame, segment_by: str) -> Optional[str]:
    mode = segment_by.lower()
    if mode == "none":
        return None
    if mode in {"epoch", "round"}:
        if mode not in df.columns:
            raise ValueError(f"{path} is missing requested segment column: {mode}")
        return mode
    if mode != "auto":
        raise ValueError("--segment_by must be one of: auto, epoch, round, none")
    if "epoch" in df.columns:
        return "epoch"
    if "round" in df.columns:
        return "round"
    return None


def load_local_ml_records(
    input_dir: Path,
    device_groups: Dict[str, List[str]],
    feature_columns: Sequence[str],
    segment_by: str,
    num_bins: int,
) -> Dict[Tuple[str, str, str, str, str], RunRecord]:
    device_to_group = {
        device: group_name
        for group_name, devices in device_groups.items()
        for device in devices
    }
    records: Dict[Tuple[str, str, str, str, str], RunRecord] = {}

    for device, device_group in sorted(device_to_group.items()):
        local_ml_dir = input_dir / device / "local_ml"
        if not local_ml_dir.exists():
            print(f"warning: missing local_ml directory for {device}: {local_ml_dir}")
            continue
        for path in sorted(local_ml_dir.glob("*.csv")):
            if not is_hardware_csv(path):
                continue
            df = pd.read_csv(path)
            if df.empty:
                continue
            missing = [column for column in feature_columns if column not in df.columns]
            if missing:
                raise ValueError(f"{path} is missing feature columns: {missing}")

            df = ot_base.sort_cpu_cores_per_timestamp(df, feature_columns)
            df = ot_base.apply_counter_deltas(df, feature_columns)
            segment_column = segment_column_for_df(path, df, segment_by)
            if segment_column is None:
                groups = [("full_run", "all", df)]
            else:
                numeric_segment = pd.to_numeric(df[segment_column], errors="coerce")
                working = df.loc[numeric_segment.notna()].copy()
                working["_analysis_segment"] = numeric_segment.loc[numeric_segment.notna()].astype(int).to_numpy()
                groups = [
                    (segment_column, str(int(segment_id)), group.drop(columns=["_analysis_segment"]))
                    for segment_id, group in working.groupby("_analysis_segment", sort=True)
                ]

            first = df.iloc[0]
            trial_id = normalized_trial_id(first.get("trial_id", ""))
            poisoning_method = str(first.get("poisoning_method", "") or "clean")
            for segment_type, segment_id, segment_df in groups:
                values = (
                    segment_df.loc[:, feature_columns]
                    .apply(pd.to_numeric, errors="coerce")
                    .to_numpy(dtype=np.float32)
                )
                if not np.isfinite(values).all():
                    bad = np.argwhere(~np.isfinite(values))[0].tolist()
                    raise ValueError(f"{path} has NaN/inf feature data; first bad index={bad}")
                original_length = int(values.shape[0])
                values = fixed_window_average(values, num_bins)
                key = (device_group, device, trial_id, poisoning_method, segment_id)
                if key in records:
                    raise ValueError(f"Duplicate run segment for {key}: {records[key].source_path} and {path}")
                records[key] = RunRecord(
                    device=device,
                    device_group=device_group,
                    trial_id=trial_id,
                    poisoning_method=poisoning_method,
                    segment_type=segment_type,
                    segment_id=segment_id,
                    values=values,
                    source_path=str(path),
                    original_length=original_length,
                )
    return records


def get_record(
    records: Dict[Tuple[str, str, str, str, str], RunRecord],
    device_group: str,
    device: str,
    trial_id: str,
    poisoning_method: str,
    segment_id: str,
) -> Optional[RunRecord]:
    return records.get((device_group, device, trial_id, poisoning_method, segment_id))


def sorted_segment_ids(records: Dict[Tuple[str, str, str, str, str], RunRecord]) -> List[str]:
    def key_func(value: str) -> Tuple[int, Any]:
        try:
            return (0, int(value))
        except ValueError:
            return (1, value)

    return sorted({key[-1] for key in records.keys()}, key=key_func)


def compute_cross_device_ot(
    records: Dict[Tuple[str, str, str, str, str], RunRecord],
    device_groups: Dict[str, List[str]],
    feature_columns: Sequence[str],
    cost_types: Sequence[str],
    attacks: Sequence[str],
    trial_ids: Sequence[str],
    window_size: int,
    sinkhorn_reg: float,
    sinkhorn_num_iter: int,
    sinkhorn_stop_thr: float,
    use_sinkhorn: bool,
    normalize_solver_cost: bool,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    rows: List[Dict[str, Any]] = []
    embeddings: Dict[str, np.ndarray] = {}
    segment_ids = sorted_segment_ids(records)

    for source_group, target_group in DEVICE_TYPE_PAIRS:
        source_devices = device_groups.get(source_group, [])
        target_devices = device_groups.get(target_group, [])
        if not source_devices or not target_devices:
            continue
        device_pair = f"{source_group}_vs_{target_group}"
        print(f"device_pair={device_pair}")

        for segment_id in segment_ids:
            for metric_index, source_column in enumerate(feature_columns):
                metric_name = ot_base.metric_name_for_column(source_column)
                metric_transform = ot_base.metric_transform_for_column(source_column)
                for cost_type in cost_types:
                    print(f"  segment={segment_id} metric={metric_name} cost_type={cost_type}")
                    for ref_condition, target_condition in CONDITION_PAIRS:
                        cond_name = condition_name(ref_condition, target_condition)
                        for attack in attacks:
                            ref_poisoning = "clean" if ref_condition == "clean" else attack
                            target_poisoning = "clean" if target_condition == "clean" else attack
                            for trial_id in trial_ids:
                                for source_device in source_devices:
                                    ref = get_record(
                                        records,
                                        source_group,
                                        source_device,
                                        trial_id,
                                        ref_poisoning,
                                        segment_id,
                                    )
                                    if ref is None:
                                        continue
                                    for target_device in target_devices:
                                        target = get_record(
                                            records,
                                            target_group,
                                            target_device,
                                            trial_id,
                                            target_poisoning,
                                            segment_id,
                                        )
                                        if target is None:
                                            continue

                                        ref_run = ref.values[:, [metric_index]]
                                        target_run = target.values[:, [metric_index]]
                                        ref_features = ot_base.build_features_for_cost(
                                            ref_run, cost_type=cost_type, window_size=window_size
                                        )
                                        target_features = ot_base.build_features_for_cost(
                                            target_run, cost_type=cost_type, window_size=window_size
                                        )
                                        plan, cost, cost_scale = ot_base.solve_ot(
                                            ref_features,
                                            target_features,
                                            reg=sinkhorn_reg,
                                            use_sinkhorn=use_sinkhorn,
                                            sinkhorn_num_iter=sinkhorn_num_iter,
                                            sinkhorn_stop_thr=sinkhorn_stop_thr,
                                            normalize_solver_cost=normalize_solver_cost,
                                        )
                                        embedding = ot_base.compute_barycentric_embedding(
                                            ref_features, target_features, plan
                                        )
                                        embedding_id = (
                                            f"{device_pair}|{cond_name}|{attack}|{trial_id}|"
                                            f"{source_device}|{target_device}|{segment_id}|"
                                            f"{metric_name}|{cost_type}"
                                        )
                                        embeddings[embedding_id] = embedding.astype(np.float32, copy=False)
                                        rows.append(
                                            {
                                                "embedding_id": embedding_id,
                                                "device_pair": device_pair,
                                                "source_device_group": source_group,
                                                "target_device_group": target_group,
                                                "source_device": source_device,
                                                "target_device": target_device,
                                                "condition": cond_name,
                                                "reference_condition": ref_condition,
                                                "target_condition": target_condition,
                                                "attack_type": attack,
                                                "reference_poisoning_method": ref_poisoning,
                                                "target_poisoning_method": target_poisoning,
                                                "trial_id": trial_id,
                                                "segment_type": ref.segment_type,
                                                "segment_id": segment_id,
                                                "metric_name": metric_name,
                                                "source_column": source_column,
                                                "metric_transform": metric_transform,
                                                "cost_type": cost_type,
                                                "ot_cost": float(np.sum(plan * cost)),
                                                "tangent_norm": float(np.linalg.norm(embedding)),
                                                "ref_length": int(ref.values.shape[0]),
                                                "target_length": int(target.values.shape[0]),
                                                "ref_original_length": int(ref.original_length),
                                                "target_original_length": int(target.original_length),
                                                "feature_dim": int(ref_features.shape[1]),
                                                "ot_solver_cost_scale": float(cost_scale),
                                                "reference_source_path": ref.source_path,
                                                "target_source_path": target.source_path,
                                            }
                                        )

    if not rows:
        raise ValueError("No cross-device comparisons were computed. Check device groups, trials, and attacks.")
    return pd.DataFrame(rows), embeddings


def add_embedding_pca(df: pd.DataFrame, embeddings: Dict[str, np.ndarray]) -> pd.DataFrame:
    out = df.copy()
    out["pca_x"] = np.nan
    out["pca_y"] = np.nan
    group_cols = ["segment_type", "segment_id", "metric_name", "cost_type"]
    for _, sub in out.groupby(group_cols, sort=False):
        vectors = np.vstack([embeddings[embedding_id] for embedding_id in sub["embedding_id"]])
        pca = ot_base.PCANP(n_components=min(2, vectors.shape[0], vectors.shape[1])).fit(vectors)
        coords = pca.transform(vectors)
        out.loc[sub.index, "pca_x"] = coords[:, 0] if coords.shape[1] > 0 else 0.0
        out.loc[sub.index, "pca_y"] = coords[:, 1] if coords.shape[1] > 1 else 0.0
    return out


def mean_std_by_epoch(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(
            [
                "device_pair",
                "condition",
                "attack_type",
                "segment_id",
                "metric_name",
                "cost_type",
            ],
            sort=False,
        )["ot_cost"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["std"] = grouped["std"].fillna(0.0)
    grouped["segment_num"] = pd.to_numeric(grouped["segment_id"], errors="coerce")
    return grouped


def make_ot_distance_plots(df: pd.DataFrame, output_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib_cache"))
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots" / "ot_distance"
    plot_dir.mkdir(parents=True, exist_ok=True)
    stats = mean_std_by_epoch(df)
    cost_types = list(dict.fromkeys(df["cost_type"].tolist()))
    metrics = list(dict.fromkeys(df["metric_name"].tolist()))
    colors = {
        "unlearnable_examples": "tab:red",
        "availability_shortcuts": "tab:orange",
    }

    for device_pair in sorted(df["device_pair"].unique()):
        for cond in [condition_name(*pair) for pair in CONDITION_PAIRS]:
            sub = stats[(stats["device_pair"] == device_pair) & (stats["condition"] == cond)]
            if sub.empty:
                continue
            fig, axes = plt.subplots(
                nrows=len(cost_types),
                ncols=len(metrics),
                figsize=(4.8 * len(metrics), 3.0 * len(cost_types)),
                squeeze=False,
                sharex=True,
            )
            for row_idx, cost_type in enumerate(cost_types):
                for col_idx, metric_name in enumerate(metrics):
                    ax = axes[row_idx][col_idx]
                    cell = sub[(sub["cost_type"] == cost_type) & (sub["metric_name"] == metric_name)]
                    for attack_type, attack_df in cell.groupby("attack_type", sort=False):
                        attack_df = attack_df.sort_values(["segment_num", "segment_id"])
                        if attack_df["segment_num"].notna().all():
                            x = attack_df["segment_num"].to_numpy(dtype=float)
                        else:
                            x = np.arange(len(attack_df), dtype=float)
                        y = attack_df["mean"].to_numpy(dtype=float)
                        std = attack_df["std"].to_numpy(dtype=float)
                        label = str(attack_type)
                        color = "tab:blue" if cond == "clean_to_clean" else colors.get(label, None)
                        ax.plot(x, y, marker="o", linewidth=1.6, label=label, color=color)
                        if len(y) > 1:
                            ax.fill_between(x, y - std, y + std, alpha=0.15, color=color)
                    if cond in ATTACK_RELATED_CONDITIONS:
                        clean_cell = stats[
                            (stats["device_pair"] == device_pair)
                            & (stats["condition"] == "clean_to_clean")
                            & (stats["cost_type"] == cost_type)
                            & (stats["metric_name"] == metric_name)
                        ]
                        if not clean_cell.empty:
                            clean_baseline = (
                                clean_cell.groupby(["segment_id"], sort=False)
                                .agg(
                                    mean=("mean", "mean"),
                                    std=("mean", "std"),
                                    segment_num=("segment_num", "first"),
                                )
                                .reset_index()
                                .sort_values(["segment_num", "segment_id"])
                            )
                            clean_baseline["std"] = clean_baseline["std"].fillna(0.0)
                            if clean_baseline["segment_num"].notna().all():
                                clean_x = clean_baseline["segment_num"].to_numpy(dtype=float)
                            else:
                                clean_x = np.arange(len(clean_baseline), dtype=float)
                            clean_y = clean_baseline["mean"].to_numpy(dtype=float)
                            clean_std = clean_baseline["std"].to_numpy(dtype=float)
                            ax.plot(
                                clean_x,
                                clean_y,
                                marker="o",
                                linewidth=1.8,
                                label="clean",
                                color="tab:blue",
                            )
                            if len(clean_y) > 1:
                                ax.fill_between(
                                    clean_x,
                                    clean_y - clean_std,
                                    clean_y + clean_std,
                                    alpha=0.12,
                                    color="tab:blue",
                                )
                    ax.set_title(f"{metric_name} / {cost_type}")
                    ax.set_ylabel("OT distance")
                    ax.grid(True, alpha=0.3)
                    if row_idx == len(cost_types) - 1:
                        ax.set_xlabel("epoch")
            handles, labels = axes[0][0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper center", ncol=2)
            fig.suptitle(f"{device_pair}: {cond}", y=0.995)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
            fig.savefig(plot_dir / f"ot_distance_{safe_name(device_pair)}_{safe_name(cond)}.png", dpi=180)
            plt.close(fig)


def make_embedding_pca_plots(df: pd.DataFrame, output_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib_cache"))
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots" / "embedding_pca"
    plot_dir.mkdir(parents=True, exist_ok=True)
    cost_types = list(dict.fromkeys(df["cost_type"].tolist()))
    metrics = list(dict.fromkeys(df["metric_name"].tolist()))
    colors = {
        "unlearnable_examples": "tab:red",
        "availability_shortcuts": "tab:orange",
    }

    for device_pair in sorted(df["device_pair"].unique()):
        for cond in [condition_name(*pair) for pair in CONDITION_PAIRS]:
            sub = df[(df["device_pair"] == device_pair) & (df["condition"] == cond)]
            if sub.empty:
                continue
            fig, axes = plt.subplots(
                nrows=len(cost_types),
                ncols=len(metrics),
                figsize=(4.8 * len(metrics), 3.0 * len(cost_types)),
                squeeze=False,
            )
            for row_idx, cost_type in enumerate(cost_types):
                for col_idx, metric_name in enumerate(metrics):
                    ax = axes[row_idx][col_idx]
                    cell = sub[(sub["cost_type"] == cost_type) & (sub["metric_name"] == metric_name)]
                    for attack_type, attack_df in cell.groupby("attack_type", sort=False):
                        label = str(attack_type)
                        color = "tab:blue" if cond == "clean_to_clean" else colors.get(label, None)
                        ax.scatter(
                            attack_df["pca_x"],
                            attack_df["pca_y"],
                            s=22,
                            alpha=0.72,
                            label=label,
                            color=color,
                        )
                    if cond in ATTACK_RELATED_CONDITIONS:
                        clean_cell = df[
                            (df["device_pair"] == device_pair)
                            & (df["condition"] == "clean_to_clean")
                            & (df["cost_type"] == cost_type)
                            & (df["metric_name"] == metric_name)
                        ]
                        if not clean_cell.empty:
                            clean_cell = clean_cell.drop_duplicates(
                                subset=[
                                    "source_device",
                                    "target_device",
                                    "trial_id",
                                    "segment_id",
                                    "metric_name",
                                    "cost_type",
                                ]
                            )
                            ax.scatter(
                                clean_cell["pca_x"],
                                clean_cell["pca_y"],
                                s=24,
                                alpha=0.72,
                                label="clean",
                                color="tab:blue",
                            )
                    ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.25)
                    ax.axvline(0.0, color="black", linewidth=0.7, alpha=0.25)
                    ax.set_title(f"{metric_name} / {cost_type}")
                    ax.set_xlabel("PCA x")
                    ax.set_ylabel("PCA y")
                    ax.grid(True, alpha=0.3)
            handles, labels = axes[0][0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper center", ncol=2)
            fig.suptitle(f"{device_pair}: {cond}", y=0.995)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
            fig.savefig(plot_dir / f"embedding_pca_{safe_name(device_pair)}_{safe_name(cond)}.png", dpi=180)
            plt.close(fig)


def save_outputs(
    df: pd.DataFrame,
    embeddings: Dict[str, np.ndarray],
    output_dir: Path,
    save_embeddings: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "cross_device_ot_summary.csv", index=False)
    pca_columns = [
        "embedding_id",
        "device_pair",
        "condition",
        "attack_type",
        "source_device",
        "target_device",
        "trial_id",
        "segment_type",
        "segment_id",
        "metric_name",
        "source_column",
        "cost_type",
        "pca_x",
        "pca_y",
        "tangent_norm",
        "ot_cost",
    ]
    df.loc[:, pca_columns].to_csv(output_dir / "cross_device_embedding_pca.csv", index=False)
    make_ot_distance_plots(df, output_dir)
    make_embedding_pca_plots(df, output_dir)
    if save_embeddings:
        np.savez_compressed(output_dir / "cross_device_tangent_embeddings.npz", **embeddings)


def build_device_groups(args: argparse.Namespace) -> Dict[str, List[str]]:
    if args.pairing_strategy == "all":
        groups = {name: list(devices) for name, devices in ALL_DEVICE_GROUPS.items()}
    else:
        groups = {name: list(devices) for name, devices in DEFAULT_DEVICE_GROUPS.items()}

    overrides = {
        "rpi4": args.rpi4_devices,
        "rpi3": args.rpi3_devices,
        "jetson_cpu": args.jetson_devices,
    }
    for group_name, value in overrides.items():
        if value:
            groups[group_name] = parse_csv_list(value)
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="collected_logs")
    parser.add_argument("--output_dir", default="cross_device_result")
    parser.add_argument("--pairing_strategy", choices=["representative", "all"], default="representative")
    parser.add_argument("--rpi4_devices", default="")
    parser.add_argument("--rpi3_devices", default="")
    parser.add_argument("--jetson_devices", default="")
    parser.add_argument("--attacks", default=",".join(DEFAULT_ATTACKS))
    parser.add_argument("--trials", default=",".join(DEFAULT_TRIALS))
    parser.add_argument("--feature_columns", default=",".join(DEFAULT_FEATURE_COLUMNS))
    parser.add_argument("--cost_types", default=",".join(DEFAULT_COST_TYPES))
    parser.add_argument("--segment_by", choices=["auto", "epoch", "round", "none"], default="epoch")
    parser.add_argument("--num_bins", type=int, default=128)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--sinkhorn_reg", type=float, default=1.0)
    parser.add_argument("--sinkhorn_num_iter", type=int, default=300)
    parser.add_argument("--sinkhorn_stop_thr", type=float, default=1e-6)
    parser.add_argument("--normalize_solver_cost", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_sinkhorn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_embeddings", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    device_groups = build_device_groups(args)
    feature_columns = ot_base.parse_feature_columns(args.feature_columns)
    cost_types = ot_base.parse_cost_types(args.cost_types)
    attacks = parse_csv_list(args.attacks)
    trial_ids = parse_csv_list(args.trials)

    print("device_groups:", device_groups)
    print("feature_columns:", feature_columns)
    print("cost_types:", cost_types)
    print("attacks:", attacks)
    print("trials:", trial_ids)
    print(f"num_bins={args.num_bins}")

    records = load_local_ml_records(
        input_dir=input_dir,
        device_groups=device_groups,
        feature_columns=feature_columns,
        segment_by=args.segment_by,
        num_bins=args.num_bins,
    )
    print(f"loaded_segments={len(records)}")

    df, embeddings = compute_cross_device_ot(
        records=records,
        device_groups=device_groups,
        feature_columns=feature_columns,
        cost_types=cost_types,
        attacks=attacks,
        trial_ids=trial_ids,
        window_size=args.window_size,
        sinkhorn_reg=args.sinkhorn_reg,
        sinkhorn_num_iter=args.sinkhorn_num_iter,
        sinkhorn_stop_thr=args.sinkhorn_stop_thr,
        use_sinkhorn=args.use_sinkhorn,
        normalize_solver_cost=args.normalize_solver_cost,
    )
    df = add_embedding_pca(df, embeddings)
    save_outputs(df, embeddings, output_dir, save_embeddings=args.save_embeddings)
    print(f"saved outputs to {output_dir}")
    print(f"rows={len(df)}")


if __name__ == "__main__":
    main()
