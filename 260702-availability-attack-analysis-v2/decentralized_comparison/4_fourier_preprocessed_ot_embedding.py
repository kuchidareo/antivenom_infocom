#!/usr/bin/env python3
"""Fourier-preprocessed OT distances and tangent embeddings.

This is intentionally close to ``1_calculate_ot_embedding.py``:

1. Load hardware CSVs.
2. Segment local-ML traces by epoch by default.
3. Sort CPU cores per timestamp and convert counter metrics to deltas through
   the existing helper code.
4. Convert each variable-length epoch trace into a fixed number of time bins.
5. Apply a real FFT along time for each metric.
6. Run the same POT-based OT and tangent-embedding pipeline in frequency space.

Interpretation:
- The OT support points are frequency bins, not raw time samples.
- ``c1_time`` compares the normalized frequency-bin position. This captures
  where mass is shifted along the frequency axis.
- ``c2_value`` compares Fourier magnitude distributions. This captures how much
  frequency energy differs at each support point.
- Shape costs are intentionally not used by default here. They can still be
  requested manually through ``--cost_types`` if needed.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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


DEFAULT_FEATURE_COLUMNS = [
    "system_cpu_core_1",
    "system_cpu_core_2",
    "system_cpu_core_3",
]
DEFAULT_COST_TYPES = [
    "c1_time",
    "c2_value",
]
EXTRA_SUMMARY_COLUMNS = [
    "preprocess",
    "fourier_spectrum",
    "fourier_num_bins",
    "fourier_drop_dc",
    "fourier_detrend",
    "fourier_log_amplitude",
    "fourier_normalize",
]


def fixed_window_average(values: np.ndarray, num_bins: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    if num_bins <= 0 or x.shape[0] == num_bins:
        return x
    if x.shape[0] < 2:
        return np.repeat(x, max(num_bins, 1), axis=0)
    if x.shape[0] < num_bins:
        old_pos = np.linspace(0.0, 1.0, x.shape[0], dtype=np.float32)
        new_pos = np.linspace(0.0, 1.0, num_bins, dtype=np.float32)
        out = np.empty((num_bins, x.shape[1]), dtype=np.float32)
        for col_idx in range(x.shape[1]):
            out[:, col_idx] = np.interp(new_pos, old_pos, x[:, col_idx])
        return out
    indices = np.array_split(np.arange(x.shape[0]), num_bins)
    return np.vstack([x[idx].mean(axis=0) for idx in indices]).astype(np.float32)


def fourier_transform_run(
    run: np.ndarray,
    *,
    num_bins: int,
    spectrum: str,
    drop_dc: bool,
    detrend: bool,
    log_amplitude: bool,
    normalize: bool,
) -> np.ndarray:
    """Return frequency-domain metric traces with shape (n_freq_bins, n_metrics)."""
    x = fixed_window_average(run, num_bins=num_bins)
    x = np.asarray(x, dtype=np.float32)
    if detrend:
        x = x - x.mean(axis=0, keepdims=True)

    fft = np.fft.rfft(x, axis=0)
    if spectrum == "magnitude":
        values = np.abs(fft)
    elif spectrum == "power":
        values = np.abs(fft) ** 2
    else:
        raise ValueError("--fourier_spectrum must be one of: magnitude, power")

    values = values.astype(np.float32, copy=False)
    if drop_dc and values.shape[0] > 1:
        values = values[1:]
    if log_amplitude:
        values = np.log1p(values)
    if normalize:
        denom = np.linalg.norm(values, axis=0, keepdims=True)
        denom[denom < 1e-12] = 1.0
        values = values / denom
    if values.shape[0] == 0:
        raise ValueError("Fourier transform produced zero frequency bins. Disable --drop_dc or increase --num_bins.")
    if not np.isfinite(values).all():
        bad = np.argwhere(~np.isfinite(values))[0].tolist()
        raise ValueError(f"Fourier transform produced NaN/inf; first bad index={bad}")
    return values.astype(np.float32, copy=False)


def transform_data_fourier(
    data: Dict[str, Dict[str, Any]],
    *,
    num_bins: int,
    spectrum: str,
    drop_dc: bool,
    detrend: bool,
    log_amplitude: bool,
    normalize: bool,
) -> Dict[str, Dict[str, Any]]:
    transformed: Dict[str, Dict[str, Any]] = {}
    for trial_id, trial_data in data.items():
        out_trial: Dict[str, Any] = {"poisoning": {}}
        if "clean" in trial_data:
            out_trial["clean"] = fourier_transform_run(
                trial_data["clean"],
                num_bins=num_bins,
                spectrum=spectrum,
                drop_dc=drop_dc,
                detrend=detrend,
                log_amplitude=log_amplitude,
                normalize=normalize,
            )
        for poisoning_type, run in trial_data.get("poisoning", {}).items():
            out_trial["poisoning"][poisoning_type] = fourier_transform_run(
                run,
                num_bins=num_bins,
                spectrum=spectrum,
                drop_dc=drop_dc,
                detrend=detrend,
                log_amplitude=log_amplitude,
                normalize=normalize,
            )
        transformed[trial_id] = out_trial
    ot_base.validate_data(transformed)
    return transformed


def add_clean_zscores_preserve_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    zdf = df.copy()
    baseline_rows: List[Dict[str, Any]] = []
    metric_pairs = [
        ("ot_cost", "z_ot_cost"),
        ("tangent_norm", "z_tangent_norm"),
        ("residual_norm", "z_residual_norm"),
        ("residual_ratio", "z_residual_ratio"),
    ]
    group_cols = ["metric_name", "cost_type", "segment_type", "segment_id"]
    for group_key, sub in df.groupby(group_cols, sort=False):
        group_values = dict(zip(group_cols, group_key))
        clean = sub[
            (sub["target_group"] == "clean")
            & (sub["reference_trial_id"].astype(str) != sub["target_trial_id"].astype(str))
        ]
        if clean.empty:
            clean = sub[sub["target_group"] == "clean"]
        stats: Dict[str, Any] = {
            **group_values,
            "n_clean_targets": int(len(clean)),
        }
        for raw_col, z_col in metric_pairs:
            mean_value = float(clean[raw_col].mean())
            std_value = float(clean[raw_col].std(ddof=0))
            zdf.loc[sub.index, z_col] = (sub[raw_col] - mean_value) / (std_value + 1e-12)
            stats[f"mean_{raw_col}"] = mean_value
            stats[f"std_{raw_col}"] = std_value
        baseline_rows.append(stats)
    return zdf, pd.DataFrame(baseline_rows)


def safe_name(value: str) -> str:
    return str(value).replace("/", "_").replace(" ", "_").replace(":", "_")


def mean_std_by_epoch(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(
            ["metric_name", "cost_type", "poisoning_type", "target_group", "segment_id"],
            sort=False,
        )["ot_cost"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["std"] = grouped["std"].fillna(0.0)
    grouped["segment_num"] = pd.to_numeric(grouped["segment_id"], errors="coerce")
    return grouped


def plot_color(poisoning_type: str, target_group: str) -> str:
    if target_group == "clean" or poisoning_type == "none":
        return "tab:blue"
    return {
        "unlearnable_examples": "tab:red",
        "availability_shortcuts": "tab:orange",
        "random_label_flipping": "tab:green",
        "target_label_flipping": "tab:purple",
    }.get(poisoning_type, "tab:red")


def make_ot_distance_plots(df: pd.DataFrame, output_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib_cache"))
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots" / "ot_distance"
    plot_dir.mkdir(parents=True, exist_ok=True)
    stats = mean_std_by_epoch(df)
    metrics = list(dict.fromkeys(df["metric_name"].tolist()))
    cost_types = list(dict.fromkeys(df["cost_type"].tolist()))

    for metric_name in metrics:
        sub_metric = stats[stats["metric_name"] == metric_name]
        if sub_metric.empty:
            continue
        fig, axes = plt.subplots(
            nrows=len(cost_types),
            ncols=1,
            figsize=(8.0, 3.0 * len(cost_types)),
            squeeze=False,
            sharex=True,
        )
        for row_idx, cost_type in enumerate(cost_types):
            ax = axes[row_idx][0]
            cell = sub_metric[sub_metric["cost_type"] == cost_type]
            for (poisoning_type, target_group), line_df in cell.groupby(
                ["poisoning_type", "target_group"], sort=False
            ):
                line_df = line_df.sort_values(["segment_num", "segment_id"])
                if line_df["segment_num"].notna().all():
                    x = line_df["segment_num"].to_numpy(dtype=float)
                else:
                    x = np.arange(len(line_df), dtype=float)
                y = line_df["mean"].to_numpy(dtype=float)
                std = line_df["std"].to_numpy(dtype=float)
                label = "clean" if target_group == "clean" else str(poisoning_type)
                color = plot_color(str(poisoning_type), str(target_group))
                ax.plot(x, y, marker="o", linewidth=1.6, label=label, color=color)
                if len(y) > 1:
                    ax.fill_between(x, y - std, y + std, alpha=0.12, color=color)
            ax.set_title(f"{metric_name} / {cost_type}")
            ax.set_ylabel("OT distance")
            ax.grid(True, alpha=0.3)
            if row_idx == len(cost_types) - 1:
                ax.set_xlabel("epoch")
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            dedup = dict(zip(labels, handles))
            fig.legend(dedup.values(), dedup.keys(), loc="upper center", ncol=min(4, len(dedup)))
        fig.suptitle(f"Fourier OT distance: {metric_name}", y=0.995)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        fig.savefig(plot_dir / f"fourier_ot_distance_{safe_name(metric_name)}.png", dpi=180)
        plt.close(fig)


def make_embedding_pca_plots(df: pd.DataFrame, output_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib_cache"))
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots" / "embedding_pca"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for (metric_name, cost_type), sub in df.groupby(["metric_name", "cost_type"], sort=False):
        fig, ax = plt.subplots(figsize=(7, 5))
        for (poisoning_type, target_group), point_df in sub.groupby(["poisoning_type", "target_group"], sort=False):
            label = "clean" if target_group == "clean" else str(poisoning_type)
            ax.scatter(
                point_df["pca_x"],
                point_df["pca_y"],
                s=22,
                alpha=0.72,
                label=label,
                color=plot_color(str(poisoning_type), str(target_group)),
            )
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            dedup = dict(zip(labels, handles))
            ax.legend(dedup.values(), dedup.keys(), loc="best")
        ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.25)
        ax.axvline(0.0, color="black", linewidth=0.7, alpha=0.25)
        ax.set_title(f"Fourier embedding PCA: {metric_name} / {cost_type}")
        ax.set_xlabel("PCA x")
        ax.set_ylabel("PCA y")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            plot_dir / f"fourier_embedding_pca_{safe_name(metric_name)}_{safe_name(cost_type)}.png",
            dpi=180,
        )
        plt.close(fig)


def save_outputs(
    output_dir: Path,
    summary_df: pd.DataFrame,
    zscore_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    embeddings: Dict[Tuple[str, str, str], np.ndarray],
    save_embeddings: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / "fourier_ot_embedding_summary.csv", index=False)
    zscore_df.to_csv(output_dir / "fourier_ot_embedding_summary_zscored.csv", index=False)
    baseline_df.to_csv(output_dir / "fourier_clean_baseline_stats.csv", index=False)
    pca_cols = [
        "reference_run_id",
        "reference_trial_id",
        "target_trial_id",
        "target_run_id",
        "target_group",
        "poisoning_type",
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
    zscore_df.loc[:, pca_cols].to_csv(output_dir / "fourier_embedding_pca.csv", index=False)
    make_ot_distance_plots(zscore_df, output_dir)
    make_embedding_pca_plots(zscore_df, output_dir)
    if save_embeddings:
        np.savez_compressed(
            output_dir / "fourier_tangent_embeddings.npz",
            **{f"{k[0]}__{k[1]}__{k[2]}": v for k, v in embeddings.items()},
        )


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
    fourier_num_bins: int,
    fourier_spectrum: str,
    fourier_drop_dc: bool,
    fourier_detrend: bool,
    fourier_log_amplitude: bool,
    fourier_normalize: bool,
    save_embeddings: bool,
) -> None:
    segmented_data, _metadata = ot_base.load_segmented_data_from_csv_dir(
        input_dir=input_dir,
        feature_columns=feature_columns,
        segment_by=segment_by,
        max_samples_per_run=max_samples_per_run,
    )

    all_summary_rows: List[Dict[str, Any]] = []
    all_embeddings: Dict[Tuple[str, str, str], np.ndarray] = {}
    for segment_type, segment_id in sorted(
        segmented_data.keys(),
        key=lambda item: (item[0], int(item[1]) if str(item[1]).isdigit() else item[1]),
    ):
        print(f"segment_type={segment_type} segment_id={segment_id}")
        transformed_segment_data = transform_data_fourier(
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
            data = ot_base.slice_data_for_metric(transformed_segment_data, metric_index)
            reference_trial_ids = ot_base.resolve_reference_trial_ids(data, reference_trial_ids_value)
            print(
                f"metric_name={metric_name} source_column={source_column} "
                f"metric_transform={metric_transform}"
            )
            print(f"Using global clean reference trial: {reference_trial_ids[0]}")
            summary_rows, embeddings = ot_base.compute_all_ot_embeddings(
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
            pcas = ot_base.fit_clean_pcas(
                summary_rows=summary_rows,
                embeddings=embeddings,
                pca_components_for_residual=pca_components_for_residual,
            )
            ot_base.compute_residual_scores(summary_rows, embeddings, pcas)
            for row in summary_rows:
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
            scoped_embeddings = {
                (f"{segment_type}_{segment_id}_{metric_name}_{key[0]}", key[1], key[2]): value
                for key, value in embeddings.items()
            }
            for row in summary_rows:
                row["reference_run_id"] = f"{segment_type}_{segment_id}_{metric_name}_{row['reference_run_id']}"
            all_embeddings.update(scoped_embeddings)
            all_summary_rows.extend(summary_rows)

    summary_df = pd.DataFrame(all_summary_rows)
    for column in ot_base.SUMMARY_COLUMNS + EXTRA_SUMMARY_COLUMNS:
        if column not in summary_df.columns:
            summary_df[column] = np.nan
    summary_df = summary_df[ot_base.SUMMARY_COLUMNS + EXTRA_SUMMARY_COLUMNS].copy()
    zscore_df, baseline_df = add_clean_zscores_preserve_columns(summary_df)
    save_outputs(output_dir, summary_df, zscore_df, baseline_df, all_embeddings, save_embeddings=save_embeddings)
    print(f"Saved Fourier OT outputs to {output_dir}")


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
    parser.add_argument("--cost_types", default=",".join(DEFAULT_COST_TYPES))
    parser.add_argument("--segment_by", default="auto", choices=["auto", "epoch", "round", "none"])
    parser.add_argument("--max_samples_per_run", type=int, default=0)
    parser.add_argument("--fourier_num_bins", type=int, default=128)
    parser.add_argument("--fourier_spectrum", choices=["magnitude", "power"], default="magnitude")
    parser.add_argument("--fourier_drop_dc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fourier_detrend", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fourier_log_amplitude", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fourier_normalize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save_embeddings", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    feature_columns = ot_base.parse_feature_columns(args.feature_columns)
    cost_types = ot_base.parse_cost_types(args.cost_types)
    reference_trial_ids_value = args.reference_trial_id or args.reference_trial_ids

    groups = ot_base.discover_input_groups(input_dir)
    if len(groups) > 1:
        print(f"Discovered {len(groups)} local_ml analysis groups under {input_dir}")
    for group_label, group_dir in groups:
        group_output_dir = output_dir if len(groups) == 1 else output_dir / group_label
        print(f"Running Fourier OT analysis for group={group_label} input={group_dir}")
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
            fourier_num_bins=args.fourier_num_bins,
            fourier_spectrum=args.fourier_spectrum,
            fourier_drop_dc=args.fourier_drop_dc,
            fourier_detrend=args.fourier_detrend,
            fourier_log_amplitude=args.fourier_log_amplitude,
            fourier_normalize=args.fourier_normalize,
            save_embeddings=args.save_embeddings,
        )


if __name__ == "__main__":
    main()
