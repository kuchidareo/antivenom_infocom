#!/usr/bin/env python3
"""Decompose c7 window cost into amount-vs-shape terms by CPU core."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_ANALYSIS_ROOT = Path("/home/user/kuchida/antivenom_infocom/260625-transport-plan-analysis")
DEFAULT_SOURCE_ROOT = Path("/home/user/kuchida/antivenom_ewsn/2nd-submission")
DEFAULT_OT_SCRIPT = (
    DEFAULT_SOURCE_ROOT
    / "senario_evaluation"
    / "5_cost_function_ot"
    / "011_unbinned_ot_analysis.py"
)
CORE_COLS = ["core_0", "core_1", "core_2", "core_3"]


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pairwise_squared_l2(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    diff = left[:, None, :] - right[None, :, :]
    return np.sum(diff * diff, axis=2)


def decompose_window_cost(
    ot: Any,
    ref_measure: dict[str, np.ndarray],
    target_measure: dict[str, np.ndarray],
    plan: np.ndarray,
    window_size: int,
) -> dict[str, float]:
    ref_window = ot._shape_features(
        ref_measure,
        window_size=window_size,
        z_normalize_window=False,
        pad_mode="edge",
    )
    target_window = ot._shape_features(
        target_measure,
        window_size=window_size,
        z_normalize_window=False,
        pad_mode="edge",
    )

    ref_mean = ref_window.mean(axis=1, keepdims=True)
    target_mean = target_window.mean(axis=1, keepdims=True)
    window_width = ref_window.shape[1]

    level_cost = window_width * (ref_mean - target_mean.T) ** 2
    centered_cost = pairwise_squared_l2(
        ref_window - ref_mean,
        target_window - target_mean,
    )

    ref_std = ref_window.std(axis=1, keepdims=True)
    target_std = target_window.std(axis=1, keepdims=True)
    ref_std = np.where(ref_std > 1e-8, ref_std, 1.0)
    target_std = np.where(target_std > 1e-8, target_std, 1.0)
    zshape_cost = pairwise_squared_l2(
        (ref_window - ref_mean) / ref_std,
        (target_window - target_mean) / target_std,
    )

    level_weighted = float(np.sum(plan * level_cost))
    centered_weighted = float(np.sum(plan * centered_cost))
    total = level_weighted + centered_weighted
    return {
        "level_cost": level_weighted,
        "centered_shape_cost": centered_weighted,
        "zshape_cost_same_plan": float(np.sum(plan * zshape_cost)),
        "level_share": level_weighted / total if total > 0 else float("nan"),
        "centered_shape_share": centered_weighted / total if total > 0 else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--ot-script", type=Path, default=DEFAULT_OT_SCRIPT)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--reg-scale", type=float, default=0.05)
    args = parser.parse_args()

    analysis_root = args.analysis_root.resolve()
    output_dir = analysis_root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))

    import matplotlib.pyplot as plt

    base_analysis = load_module("base_transport_analysis", analysis_root / "analyze_transport_plan_c3_c7.py")
    ot = base_analysis.load_ot_module(args.ot_script.resolve())
    reference_path, clean_targets, blurring_targets = base_analysis.discover_runs(
        analysis_root / "logs",
        base_analysis.DEFAULT_REFERENCE,
    )

    runs = [
        ("reference", "reference", reference_path),
        *[("clean_vs_clean", path.stem, path) for path in clean_targets],
        *[("clean_vs_blurring", path.stem, path) for path in blurring_targets],
    ]

    run_measures = {}
    raw_rows = []
    for comparison, label, path in runs:
        df = base_analysis.load_run(path, comparison=comparison, label=label)
        prepared = ot._prepare_df(df)
        if comparison != "reference":
            for signal in CORE_COLS:
                values = prepared[signal].astype(float)
                raw_rows.append(
                    {
                        "comparison": comparison,
                        "run_label": label,
                        "signal": signal,
                        "mean": values.mean(),
                        "std": values.std(),
                        "p10": values.quantile(0.10),
                        "p50": values.quantile(0.50),
                        "p90": values.quantile(0.90),
                    }
                )
        run_measures[(comparison, label)] = {
            "path": path,
            "measures": base_analysis.prepare_run_measures(ot, df),
        }

    reference_measures = {
        signal: base_analysis.aggregate_measure(
            ot,
            run_measures[("reference", "reference")]["measures"][signal],
        )
        for signal in CORE_COLS
    }

    rows = []
    for (comparison, label), data in run_measures.items():
        if comparison == "reference":
            continue
        for epoch in sorted(data["measures"]["core_0"]):
            for signal in CORE_COLS:
                ref_measure = reference_measures[signal]
                target_measure = data["measures"][signal][epoch]
                plan, cost, _, _ = ot._transport_plan(
                    ref_measure,
                    target_measure,
                    cost_function="c7_window",
                    window_size=int(args.window_size),
                    reg_scale=float(args.reg_scale),
                    return_details=True,
                )
                pieces = decompose_window_cost(
                    ot=ot,
                    ref_measure=ref_measure,
                    target_measure=target_measure,
                    plan=plan,
                    window_size=int(args.window_size),
                )
                rows.append(
                    {
                        "comparison": comparison,
                        "run_label": label,
                        "epoch": epoch,
                        "signal": signal,
                        "ot_distance": float(np.sum(plan * cost)),
                        **pieces,
                    }
                )

    decomposition = pd.DataFrame(rows)
    raw_stats = pd.DataFrame(raw_rows)
    mean_decomposition = (
        decomposition.groupby(["comparison", "signal"])
        [
            [
                "ot_distance",
                "level_cost",
                "centered_shape_cost",
                "zshape_cost_same_plan",
                "level_share",
                "centered_shape_share",
            ]
        ]
        .mean()
        .reset_index()
    )
    mean_raw_stats = (
        raw_stats.groupby(["comparison", "signal"])[["mean", "std", "p10", "p50", "p90"]]
        .mean()
        .reset_index()
    )

    deltas = []
    for signal in CORE_COLS:
        clean = mean_decomposition[
            (mean_decomposition["comparison"] == "clean_vs_clean")
            & (mean_decomposition["signal"] == signal)
        ].iloc[0]
        blur = mean_decomposition[
            (mean_decomposition["comparison"] == "clean_vs_blurring")
            & (mean_decomposition["signal"] == signal)
        ].iloc[0]
        deltas.append(
            {
                "signal": signal,
                "ot_distance_delta": blur["ot_distance"] - clean["ot_distance"],
                "level_cost_delta": blur["level_cost"] - clean["level_cost"],
                "centered_shape_cost_delta": blur["centered_shape_cost"] - clean["centered_shape_cost"],
                "zshape_cost_same_plan_delta": blur["zshape_cost_same_plan"] - clean["zshape_cost_same_plan"],
            }
        )
    delta_df = pd.DataFrame(deltas)

    decomposition.to_csv(output_dir / "c7_core_window_decomposition_per_plan.csv", index=False)
    mean_decomposition.to_csv(output_dir / "c7_core_window_decomposition_summary.csv", index=False)
    delta_df.to_csv(output_dir / "c7_core_window_decomposition_deltas.csv", index=False)
    raw_stats.to_csv(output_dir / "c7_core_raw_cpu_stats_per_run.csv", index=False)
    mean_raw_stats.to_csv(output_dir / "c7_core_raw_cpu_stats_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(CORE_COLS))
    width = 0.36
    clean = mean_decomposition[mean_decomposition["comparison"] == "clean_vs_clean"].set_index("signal").loc[CORE_COLS]
    blur = mean_decomposition[mean_decomposition["comparison"] == "clean_vs_blurring"].set_index("signal").loc[CORE_COLS]
    ax.bar(x - width / 2, clean["level_cost"], width, label="clean level")
    ax.bar(
        x - width / 2,
        clean["centered_shape_cost"],
        width,
        bottom=clean["level_cost"],
        label="clean centered shape",
    )
    ax.bar(x + width / 2, blur["level_cost"], width, label="blurring level")
    ax.bar(
        x + width / 2,
        blur["centered_shape_cost"],
        width,
        bottom=blur["level_cost"],
        label="blurring centered shape",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(CORE_COLS)
    ax.set_ylabel("Weighted c7 cost")
    ax.set_title("c7 window cost decomposition by sorted CPU core")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "c7_core_window_decomposition.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(delta_df["signal"], delta_df["level_cost_delta"], label="level delta")
    ax.bar(
        delta_df["signal"],
        delta_df["centered_shape_cost_delta"],
        bottom=delta_df["level_cost_delta"],
        label="centered shape delta",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Blurring minus clean c7 cost")
    ax.set_title("What changes in c7: amount vs local shape")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "c7_core_window_decomposition_delta.png", dpi=180)
    plt.close(fig)

    print("c7 core decomposition deltas:")
    print(delta_df.to_string(index=False))
    print(f"Wrote decomposition outputs to: {output_dir}")


if __name__ == "__main__":
    main()
