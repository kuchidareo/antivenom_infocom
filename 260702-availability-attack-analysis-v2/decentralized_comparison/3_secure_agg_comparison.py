#!/usr/bin/env python3
"""Three-device averaged-reference OT comparison.

This script models a simple decentralized/secure-aggregation view over three
representative devices:

    RPI4, RPI3, Jetson-CPU

For each case, it first averages the selected telemetry feature traces from the
three devices. That averaged trace becomes the reference support. OT distance
and the tangent embedding are then computed from this averaged reference to
each participating device trace.

Cases:

    clean_clean_clean
    poisoned_clean_clean
    clean_poisoned_clean
    clean_clean_poisoned

For poisoned cases, both availability attacks are evaluated:
unlearnable_examples and availability_shortcuts.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
CROSS_MODULE_PATH = SCRIPT_DIR / "2_cross_device_ot_comparison.py"


def _load_cross_module() -> Any:
    spec = importlib.util.spec_from_file_location("cross_device_base", CROSS_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module: {CROSS_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cross_base = _load_cross_module()
ot_base = cross_base.ot_base


DEFAULT_DEVICE_GROUPS = {
    "rpi4": ["192.168.0.116"],
    "rpi3": ["192.168.0.115"],
    "jetson_cpu": ["192.168.0.141"],
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

DEVICE_ORDER = ["rpi4", "rpi3", "jetson_cpu"]
CASE_DEFINITIONS = [
    ("clean_clean_clean", {"rpi4": "clean", "rpi3": "clean", "jetson_cpu": "clean"}),
    ("poisoned_clean_clean", {"rpi4": "poisoned", "rpi3": "clean", "jetson_cpu": "clean"}),
    ("clean_poisoned_clean", {"rpi4": "clean", "rpi3": "poisoned", "jetson_cpu": "clean"}),
    ("clean_clean_poisoned", {"rpi4": "clean", "rpi3": "clean", "jetson_cpu": "poisoned"}),
]


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def safe_name(value: str) -> str:
    return (
        str(value)
        .replace(" ", "_")
        .replace("/", "_")
        .replace(":", "_")
        .replace(",", "_")
    )


def sorted_segment_ids(records: Dict[Tuple[str, str, str, str, str], Any]) -> List[str]:
    def key_func(value: str) -> Tuple[int, Any]:
        try:
            return (0, int(value))
        except ValueError:
            return (1, value)

    return sorted({key[-1] for key in records.keys()}, key=key_func)


def get_single_device(device_groups: Dict[str, List[str]], group_name: str) -> str:
    devices = device_groups.get(group_name, [])
    if len(devices) != 1:
        raise ValueError(
            f"3_secure_agg_comparison.py expects exactly one device for {group_name}; "
            f"got {devices}. Pass --rpi4_device/--rpi3_device/--jetson_device."
        )
    return devices[0]


def poisoning_for_condition(condition: str, attack_type: str) -> str:
    return "clean" if condition == "clean" else attack_type


def case_attack_types(case_name: str, attacks: Sequence[str]) -> List[str]:
    return ["none"] if case_name == "clean_clean_clean" else list(attacks)


def build_device_groups(args: argparse.Namespace) -> Dict[str, List[str]]:
    return {
        "rpi4": [args.rpi4_device],
        "rpi3": [args.rpi3_device],
        "jetson_cpu": [args.jetson_device],
    }


def add_embedding_pca(df: pd.DataFrame, embeddings: Dict[str, np.ndarray]) -> pd.DataFrame:
    out = df.copy()
    out["pca_x"] = np.nan
    out["pca_y"] = np.nan
    group_cols = ["segment_type", "segment_id", "metric_name", "cost_type"]
    for _, sub in out.groupby(group_cols, sort=False):
        x = np.vstack([embeddings[embedding_id] for embedding_id in sub["embedding_id"]])
        pca = ot_base.PCANP(n_components=min(2, x.shape[0], x.shape[1])).fit(x)
        coords = pca.transform(x)
        out.loc[sub.index, "pca_x"] = coords[:, 0] if coords.shape[1] > 0 else 0.0
        out.loc[sub.index, "pca_y"] = coords[:, 1] if coords.shape[1] > 1 else 0.0
    return out


def compute_secure_agg_ot(
    records: Dict[Tuple[str, str, str, str, str], Any],
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
    devices = {group: get_single_device(device_groups, group) for group in DEVICE_ORDER}
    segment_ids = sorted_segment_ids(records)

    for case_name, case_conditions in CASE_DEFINITIONS:
        for attack_type in case_attack_types(case_name, attacks):
            print(f"case={case_name} attack_type={attack_type}")
            for segment_id in segment_ids:
                for metric_index, source_column in enumerate(feature_columns):
                    metric_name = ot_base.metric_name_for_column(source_column)
                    metric_transform = ot_base.metric_transform_for_column(source_column)
                    for cost_type in cost_types:
                        print(f"  segment={segment_id} metric={metric_name} cost_type={cost_type}")
                        for trial_id in trial_ids:
                            device_records = []
                            missing = []
                            for device_group in DEVICE_ORDER:
                                condition = case_conditions[device_group]
                                poisoning_method = poisoning_for_condition(condition, attack_type)
                                device = devices[device_group]
                                record = records.get(
                                    (device_group, device, trial_id, poisoning_method, segment_id)
                                )
                                if record is None:
                                    missing.append((device_group, device, trial_id, poisoning_method, segment_id))
                                else:
                                    device_records.append((device_group, condition, poisoning_method, record))
                            if missing:
                                continue

                            runs = [record.values[:, [metric_index]] for _, _, _, record in device_records]
                            reference_run = np.mean(np.stack(runs, axis=0), axis=0).astype(np.float32)
                            reference_features = ot_base.build_features_for_cost(
                                reference_run, cost_type=cost_type, window_size=window_size
                            )

                            for target_group, target_condition, target_poisoning, target_record in device_records:
                                target_run = target_record.values[:, [metric_index]]
                                target_features = ot_base.build_features_for_cost(
                                    target_run, cost_type=cost_type, window_size=window_size
                                )
                                plan, cost, cost_scale = ot_base.solve_ot(
                                    reference_features,
                                    target_features,
                                    reg=sinkhorn_reg,
                                    use_sinkhorn=use_sinkhorn,
                                    sinkhorn_num_iter=sinkhorn_num_iter,
                                    sinkhorn_stop_thr=sinkhorn_stop_thr,
                                    normalize_solver_cost=normalize_solver_cost,
                                )
                                embedding = ot_base.compute_barycentric_embedding(
                                    reference_features, target_features, plan
                                )
                                reference_id = (
                                    f"avg_{case_name}_{attack_type}_{trial_id}_"
                                    f"{segment_id}_{metric_name}_{cost_type}"
                                )
                                embedding_id = f"{reference_id}_to_{target_group}"
                                embeddings[embedding_id] = embedding.astype(np.float32, copy=False)
                                rows.append(
                                    {
                                        "embedding_id": embedding_id,
                                        "reference_id": reference_id,
                                        "case_name": case_name,
                                        "attack_type": attack_type,
                                        "trial_id": trial_id,
                                        "segment_type": target_record.segment_type,
                                        "segment_id": segment_id,
                                        "metric_name": metric_name,
                                        "source_column": source_column,
                                        "metric_transform": metric_transform,
                                        "cost_type": cost_type,
                                        "target_device_group": target_group,
                                        "target_device": target_record.device,
                                        "target_condition": target_condition,
                                        "target_poisoning_method": target_poisoning,
                                        "ot_cost": float(np.sum(plan * cost)),
                                        "tangent_norm": float(np.linalg.norm(embedding)),
                                        "reference_length": int(reference_run.shape[0]),
                                        "target_length": int(target_run.shape[0]),
                                        "target_original_length": int(target_record.original_length),
                                        "feature_dim": int(reference_features.shape[1]),
                                        "ot_solver_cost_scale": float(cost_scale),
                                        "rpi4_condition": case_conditions["rpi4"],
                                        "rpi3_condition": case_conditions["rpi3"],
                                        "jetson_cpu_condition": case_conditions["jetson_cpu"],
                                        "rpi4_device": devices["rpi4"],
                                        "rpi3_device": devices["rpi3"],
                                        "jetson_cpu_device": devices["jetson_cpu"],
                                        "target_source_path": target_record.source_path,
                                    }
                                )

    if not rows:
        raise ValueError("No secure-aggregation comparisons were computed. Check devices, trials, and attacks.")
    return pd.DataFrame(rows), embeddings


def mean_std_by_epoch(df: pd.DataFrame) -> pd.DataFrame:
    stats = (
        df.groupby(
            [
                "case_name",
                "attack_type",
                "target_device_group",
                "target_condition",
                "segment_id",
                "metric_name",
                "cost_type",
            ],
            sort=False,
        )["ot_cost"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    stats["std"] = stats["std"].fillna(0.0)
    stats["segment_num"] = pd.to_numeric(stats["segment_id"], errors="coerce")
    return stats


def device_color(device_group: str) -> str:
    return {
        "rpi4": "tab:blue",
        "rpi3": "tab:green",
        "jetson_cpu": "tab:purple",
    }.get(device_group, "tab:gray")


def line_style_for_attack(attack_type: str) -> str:
    return "--" if attack_type == "availability_shortcuts" else "-"


def device_marker(device_group: str) -> str:
    return {
        "rpi4": "o",
        "rpi3": "s",
        "jetson_cpu": "^",
    }.get(device_group, "o")


def device_line_style(device_group: str) -> str:
    return {
        "rpi4": "-",
        "rpi3": "--",
        "jetson_cpu": ":",
    }.get(device_group, "-")


def condition_color(target_condition: str, attack_type: str) -> str:
    if target_condition == "clean":
        return "tab:blue"
    return {
        "unlearnable_examples": "tab:red",
        "availability_shortcuts": "tab:orange",
    }.get(attack_type, "tab:red")


def target_label(target_group: str, target_condition: str, attack_type: str) -> str:
    label = f"{target_group}: {target_condition}"
    if target_condition == "poisoned":
        label = f"{label} ({attack_type})"
    elif attack_type not in {"none", ""}:
        # The target itself is clean. The attack name only distinguishes which
        # poisoned trace was used by the other device when building the averaged
        # reference.
        label = f"{label} ({attack_type} case)"
    return label


def make_ot_distance_plots(df: pd.DataFrame, output_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib_cache"))
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots" / "ot_distance"
    plot_dir.mkdir(parents=True, exist_ok=True)
    stats = mean_std_by_epoch(df)
    cost_types = list(dict.fromkeys(df["cost_type"].tolist()))
    metrics = list(dict.fromkeys(df["metric_name"].tolist()))

    for case_name in [case[0] for case in CASE_DEFINITIONS]:
        sub = stats[stats["case_name"] == case_name]
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
                for (attack_type, target_group, target_condition), line_df in cell.groupby(
                    ["attack_type", "target_device_group", "target_condition"], sort=False
                ):
                    line_df = line_df.sort_values(["segment_num", "segment_id"])
                    if line_df["segment_num"].notna().all():
                        x = line_df["segment_num"].to_numpy(dtype=float)
                    else:
                        x = np.arange(len(line_df), dtype=float)
                    y = line_df["mean"].to_numpy(dtype=float)
                    std = line_df["std"].to_numpy(dtype=float)
                    label = target_label(str(target_group), str(target_condition), str(attack_type))
                    color = condition_color(str(target_condition), str(attack_type))
                    ax.plot(
                        x,
                        y,
                        marker=device_marker(str(target_group)),
                        linewidth=1.6,
                        linestyle=device_line_style(str(target_group)),
                        label=label,
                        color=color,
                    )
                    if len(y) > 1:
                        ax.fill_between(x, y - std, y + std, alpha=0.12, color=color)
                ax.set_title(f"{metric_name} / {cost_type}")
                ax.set_ylabel("OT distance")
                ax.grid(True, alpha=0.3)
                if row_idx == len(cost_types) - 1:
                    ax.set_xlabel("epoch")
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=3)
        fig.suptitle(f"averaged reference: {case_name}", y=0.995)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        fig.savefig(plot_dir / f"ot_distance_{safe_name(case_name)}.png", dpi=180)
        plt.close(fig)


def make_embedding_pca_plots(df: pd.DataFrame, output_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib_cache"))
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots" / "embedding_pca"
    plot_dir.mkdir(parents=True, exist_ok=True)
    cost_types = list(dict.fromkeys(df["cost_type"].tolist()))
    metrics = list(dict.fromkeys(df["metric_name"].tolist()))

    for case_name in [case[0] for case in CASE_DEFINITIONS]:
        sub = df[df["case_name"] == case_name]
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
                for (attack_type, target_group, target_condition), point_df in cell.groupby(
                    ["attack_type", "target_device_group", "target_condition"], sort=False
                ):
                    label = target_label(str(target_group), str(target_condition), str(attack_type))
                    ax.scatter(
                        point_df["pca_x"],
                        point_df["pca_y"],
                        s=22,
                        alpha=0.72,
                        label=label,
                        color=condition_color(str(target_condition), str(attack_type)),
                        marker=device_marker(str(target_group)),
                    )
                ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.25)
                ax.axvline(0.0, color="black", linewidth=0.7, alpha=0.25)
                ax.set_title(f"{metric_name} / {cost_type}")
                ax.set_xlabel("PCA x")
                ax.set_ylabel("PCA y")
                ax.grid(True, alpha=0.3)
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=3)
        fig.suptitle(f"averaged reference: {case_name}", y=0.995)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        fig.savefig(plot_dir / f"embedding_pca_{safe_name(case_name)}.png", dpi=180)
        plt.close(fig)


def save_outputs(
    df: pd.DataFrame,
    embeddings: Dict[str, np.ndarray],
    output_dir: Path,
    save_embeddings: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "secure_agg_ot_summary.csv", index=False)
    pca_cols = [
        "embedding_id",
        "reference_id",
        "case_name",
        "attack_type",
        "target_device_group",
        "target_device",
        "target_condition",
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
    df.loc[:, pca_cols].to_csv(output_dir / "secure_agg_embedding_pca.csv", index=False)
    make_ot_distance_plots(df, output_dir)
    make_embedding_pca_plots(df, output_dir)
    if save_embeddings:
        np.savez_compressed(output_dir / "secure_agg_tangent_embeddings.npz", **embeddings)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="collected_logs")
    parser.add_argument("--output_dir", default="secure_agg_result")
    parser.add_argument("--rpi4_device", default=DEFAULT_DEVICE_GROUPS["rpi4"][0])
    parser.add_argument("--rpi3_device", default=DEFAULT_DEVICE_GROUPS["rpi3"][0])
    parser.add_argument("--jetson_device", default=DEFAULT_DEVICE_GROUPS["jetson_cpu"][0])
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

    records = cross_base.load_local_ml_records(
        input_dir=input_dir,
        device_groups=device_groups,
        feature_columns=feature_columns,
        segment_by=args.segment_by,
        num_bins=args.num_bins,
    )
    print(f"loaded_segments={len(records)}")

    df, embeddings = compute_secure_agg_ot(
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
