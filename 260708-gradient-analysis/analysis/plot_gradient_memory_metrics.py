import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = [
    "p_bytes",
    "g_bytes",
    "buf_bytes",
    "read_streams",
    "write_streams",
    "stream_count_total",
    "sgd_read_bytes",
    "sgd_write_bytes",
    "sgd_total_bytes",
    "sgd_read_cache_lines",
    "sgd_write_cache_lines",
    "sgd_total_cache_lines",
    "traffic_share",
    "short_stream_pressure",
    "p_cache_line_util_proxy",
    "g_cache_line_util_proxy",
    "buf_cache_line_util_proxy",
    "min_cache_line_util_proxy",
    "mean_cache_line_util_proxy",
    "layout_penalty",
    "grad_l1",
    "grad_l2",
    "grad_mean_abs",
    "grad_std_abs",
    "grad_cv_abs",
    "grad_max_abs",
    "grad_max_over_mean",
    "grad_zero_ratio",
    "grad_tiny_ratio",
    "grad_positive_ratio",
    "grad_negative_ratio",
    "grad_sign_balance",
    "update_l2",
    "update_mean_abs",
    "update_max_abs",
    "update_to_param_l2",
    "value_penalty",
    "hardware_risk_raw",
    "hardware_risk_weighted",
]


def load_gradient_streams(log_dir: Path) -> pd.DataFrame:
    files = sorted(log_dir.glob("*_gradient_streams.csv"))
    if not files:
        raise FileNotFoundError(f"No *_gradient_streams.csv files found in {log_dir}")

    frames = [pd.read_csv(path) for path in files]
    df = pd.concat(frames, ignore_index=True)
    required = {"poisoning_method", "name", *METRICS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in gradient stream CSVs: {missing}")
    return df


def summarize_by_layer(df: pd.DataFrame) -> pd.DataFrame:
    numeric_metrics = [metric for metric in METRICS if metric in df.columns]
    agg_dict = {metric: (metric, "mean") for metric in numeric_metrics}
    agg = df.groupby(["poisoning_method", "name"], as_index=False).agg(**agg_dict)
    return agg


def build_delta_table(summary: pd.DataFrame) -> pd.DataFrame:
    clean = summary[summary["poisoning_method"] == "clean"].set_index("name")
    shortcut = summary[summary["poisoning_method"] == "availability_shortcuts"].set_index("name")
    if clean.empty or shortcut.empty:
        raise ValueError("Need both clean and availability_shortcuts rows to build the comparison.")

    common = clean.index.intersection(shortcut.index)
    rows = []
    value_columns = [metric for metric in METRICS if metric in clean.columns and metric in shortcut.columns]
    for name in common:
        row = {"layer": name}
        for column in value_columns:
            clean_value = clean.at[name, column]
            shortcut_value = shortcut.at[name, column]
            row[f"clean_{column}"] = clean_value
            row[f"shortcut_{column}"] = shortcut_value
            row[f"delta_{column}"] = shortcut_value - clean_value
            row[f"delta_pct_{column}"] = (shortcut_value - clean_value) / (clean_value + 1e-12) * 100.0
        rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values("clean_sgd_total_bytes", ascending=True)


def _metric_title(metric: str) -> str:
    return metric.replace("_", " ")


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def make_selected_metric_bar_plot(delta: pd.DataFrame, out_path: Path) -> None:
    selected_metrics = [
        "grad_l2",
        "grad_mean_abs",
        "update_l2",
        "hardware_risk_weighted",
    ]
    fig, axes = plt.subplots(
        nrows=len(selected_metrics),
        ncols=1,
        figsize=(13, 12),
        sharex=True,
        constrained_layout=True,
    )

    layer_labels = delta["layer"].tolist()
    x = range(len(layer_labels))
    bar_width = 0.38

    metric_titles = {
        "grad_l2": "Gradient L2",
        "grad_mean_abs": "Gradient Mean Absolute Value",
        "update_l2": "Effective SGD Update L2",
        "hardware_risk_weighted": "Traffic-Weighted Hardware Risk Proxy",
    }

    for ax, metric in zip(axes, selected_metrics):
        clean_values = delta[f"clean_{metric}"].to_numpy()
        shortcut_values = delta[f"shortcut_{metric}"].to_numpy()
        ax.bar([i - bar_width / 2 for i in x], clean_values, width=bar_width, label="clean", color="#4C78A8")
        ax.bar(
            [i + bar_width / 2 for i in x],
            shortcut_values,
            width=bar_width,
            label="availability_shortcuts",
            color="#F58518",
        )
        ax.set_ylabel(metric_titles[metric])
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xticks(list(x))
    axes[-1].set_xticklabels(layer_labels, rotation=45, ha="right")
    axes[-1].set_xlabel("Layer")
    fig.suptitle("Clean vs Availability Shortcuts: Per-Layer SGD Optimizer-Step Metrics", fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_all_metrics_by_layer_plot(delta: pd.DataFrame, out_path: Path) -> None:
    metrics = [metric for metric in METRICS if f"clean_{metric}" in delta.columns]
    layer_labels = delta["layer"].tolist()
    x = np.arange(len(layer_labels))
    bar_width = 0.38

    ncols = 3
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(22, max(4 * nrows, 12)),
        constrained_layout=True,
    )
    axes_flat = np.asarray(axes).reshape(-1)

    for ax, metric in zip(axes_flat, metrics):
        clean_values = delta[f"clean_{metric}"].to_numpy(dtype=float)
        shortcut_values = delta[f"shortcut_{metric}"].to_numpy(dtype=float)
        ax.bar(x - bar_width / 2, clean_values, width=bar_width, label="clean", color="#4C78A8")
        ax.bar(x + bar_width / 2, shortcut_values, width=bar_width, label="availability_shortcuts", color="#F58518")
        ax.set_title(_metric_title(metric), fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelrotation=80, labelsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(layer_labels, ha="right")

    for ax in axes_flat[len(metrics):]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Clean vs Availability Shortcuts: All SGD Memory/Gradient Metrics by Layer", fontsize=16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_per_layer_metric_plots(delta: pd.DataFrame, out_dir: Path) -> None:
    metrics = [metric for metric in METRICS if f"clean_{metric}" in delta.columns]
    out_dir.mkdir(parents=True, exist_ok=True)

    for _, row in delta.iterrows():
        layer = row["layer"]
        clean_values = np.array([float(row[f"clean_{metric}"]) for metric in metrics], dtype=float)
        shortcut_values = np.array([float(row[f"shortcut_{metric}"]) for metric in metrics], dtype=float)

        # One plot per layer, normalized per metric so bytes, ratios, and gradients
        # can be compared in a single view. Clean is 1.0 unless the clean value is 0.
        denom = np.where(np.abs(clean_values) > 1e-12, clean_values, 1.0)
        clean_norm = np.where(np.abs(clean_values) > 1e-12, 1.0, 0.0)
        shortcut_norm = shortcut_values / denom

        x = np.arange(len(metrics))
        bar_width = 0.38
        fig, ax = plt.subplots(figsize=(20, 7), constrained_layout=True)
        ax.bar(x - bar_width / 2, clean_norm, width=bar_width, label="clean", color="#4C78A8")
        ax.bar(x + bar_width / 2, shortcut_norm, width=bar_width, label="availability_shortcuts / clean", color="#F58518")
        ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=80, ha="right", fontsize=8)
        ax.set_ylabel("Normalized value per metric")
        ax.set_title(f"Clean vs Availability Shortcuts: {layer}")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper right")
        fig.savefig(out_dir / f"{_safe_filename(layer)}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-dir",
        default="../logs/gradient_memory_metrics_shortcut_now",
        help="Directory containing *_gradient_streams.csv files.",
    )
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    df = load_gradient_streams(Path(args.log_dir))
    summary = summarize_by_layer(df)
    delta = build_delta_table(summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    delta.to_csv(output_dir / "gradient_memory_metrics_clean_vs_shortcut_by_layer.csv", index=False)
    make_selected_metric_bar_plot(delta, output_dir / "gradient_memory_metrics_clean_vs_shortcut_by_layer.png")
    make_all_metrics_by_layer_plot(delta, output_dir / "gradient_memory_metrics_clean_vs_shortcut_all_metrics_by_layer.png")
    make_per_layer_metric_plots(delta, output_dir / "per_layer_metric_bars")

    print(f"Wrote {output_dir / 'gradient_memory_metrics_clean_vs_shortcut_by_layer.csv'}")
    print(f"Wrote {output_dir / 'gradient_memory_metrics_clean_vs_shortcut_by_layer.png'}")
    print(f"Wrote {output_dir / 'gradient_memory_metrics_clean_vs_shortcut_all_metrics_by_layer.png'}")
    print(f"Wrote {output_dir / 'per_layer_metric_bars'}")


if __name__ == "__main__":
    main()
