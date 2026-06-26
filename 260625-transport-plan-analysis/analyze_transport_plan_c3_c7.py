#!/usr/bin/env python3
"""Analyze clean-vs-clean and clean-vs-blurring OT transport plans.

This script focuses on the two cost functions that looked different in the
existing cost-function OT results:

* c3_time: absolute normalized time mismatch.
* c7_window: local-window signal-shape mismatch.

It expects raw logs copied under:

    logs/baseclean/*.csv
    logs/baseblurring/*.csv

The reference clean run defaults to baseclean/20260403_163816.csv to match the
existing `base_clean_vs_blurring` analysis.
"""

from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "results" / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SOURCE_ROOT = Path("/home/user/kuchida/antivenom_ewsn/2nd-submission")
DEFAULT_OT_SCRIPT = (
    DEFAULT_SOURCE_ROOT
    / "senario_evaluation"
    / "5_cost_function_ot"
    / "011_unbinned_ot_analysis.py"
)
DEFAULT_REFERENCE = "20260403_163816.csv"
CORE_COLS = ["core_0", "core_1", "core_2", "core_3"]
SIGNALS = CORE_COLS
COST_FUNCTIONS = ["c3_time", "c7_window"]


def load_ot_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("cost_ot", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load OT module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def extract_run_info(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        event = str(row.get("event") or "").strip()
        if not event.startswith("{"):
            continue
        try:
            payload = json.loads(event)
        except json.JSONDecodeError:
            continue
        run_info = payload.get("run_info")
        if isinstance(run_info, dict):
            return run_info
    return {}


def parse_core_list(value: object) -> list[float] | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    if isinstance(parsed, (list, tuple)):
        return [float(x) for x in parsed]
    return None


def load_run(path: Path, comparison: str, label: str) -> pd.DataFrame:
    rows = read_csv_rows(path)
    run_info = extract_run_info(rows)
    df = pd.DataFrame(rows)
    df["comparison"] = comparison
    df["run_label"] = label
    df["run_csv"] = str(path)
    poison_type = str(run_info.get("poison_type", comparison))
    if poison_type == "none":
        poison_type = "clean"
    df["poisoning_type"] = poison_type
    df["poison_frac"] = pd.to_numeric(run_info.get("poison_frac"), errors="coerce")
    return df


def discover_runs(logs_dir: Path, reference_name: str) -> tuple[Path, list[Path], list[Path]]:
    clean_dir = logs_dir / "baseclean"
    blurring_dir = logs_dir / "baseblurring"
    if not clean_dir.exists():
        raise FileNotFoundError(f"Missing clean logs dir: {clean_dir}")
    if not blurring_dir.exists():
        raise FileNotFoundError(f"Missing blurring logs dir: {blurring_dir}")

    clean_runs = sorted(clean_dir.glob("*.csv"))
    blurring_runs = sorted(blurring_dir.glob("*.csv"))
    reference = clean_dir / reference_name
    if not reference.exists():
        if not clean_runs:
            raise FileNotFoundError(f"No clean CSVs found in {clean_dir}")
        reference = clean_runs[0]
        print(f"Reference {reference_name} not found; using {reference.name}")
    clean_targets = [path for path in clean_runs if path.resolve() != reference.resolve()]
    return reference, clean_targets, blurring_runs


def prepare_run_measures(ot: Any, df: pd.DataFrame) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    prepared = ot._prepare_df(df)
    measures: dict[str, dict[int, dict[str, np.ndarray]]] = {signal: {} for signal in SIGNALS}
    for epoch, group in prepared.groupby("epoch"):
        epoch_idx = int(epoch)
        for col in CORE_COLS:
            measures[col][epoch_idx] = ot._epoch_measure(group, col)
    return measures


def aggregate_measure(ot: Any, measures: dict[int, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return ot._aggregate_run_measure([measures[epoch] for epoch in sorted(measures)])


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float).ravel()
    weights = np.asarray(weights, dtype=float).ravel()
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[mask]
    weights = weights[mask]
    if values.size == 0:
        return float("nan")
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights) / np.sum(weights)
    return float(np.interp(float(q), cdf, values))


def plan_metrics(
    plan: np.ndarray,
    cost: np.ndarray,
    ref: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
) -> dict[str, float]:
    if plan.size == 0 or not np.isfinite(plan).any():
        return defaultdict(lambda: float("nan"))

    ref_t = np.asarray(ref["t"], dtype=float)[:, None]
    target_t = np.asarray(target["t"], dtype=float)[None, :]
    ref_x = np.asarray(ref["x"], dtype=float)[:, None]
    target_x = np.asarray(target["x"], dtype=float)[None, :]
    signed_shift = target_t - ref_t
    abs_shift = np.abs(signed_shift)
    abs_value_delta = np.abs(target_x - ref_x)
    contribution = plan * cost
    mass_total = float(plan.sum())

    diag_005 = float(plan[abs_shift <= 0.05].sum())
    diag_010 = float(plan[abs_shift <= 0.10].sum())
    ref_early_to_target_late = float(plan[(ref_t <= 0.33) & (target_t >= 0.67)].sum())
    ref_late_to_target_early = float(plan[(ref_t >= 0.67) & (target_t <= 0.33)].sum())

    return {
        "mass_total": mass_total,
        "ot_distance": float(contribution.sum()),
        "mean_cost_given_plan": float(contribution.sum() / mass_total) if mass_total else float("nan"),
        "mean_signed_time_shift": float((plan * signed_shift).sum()),
        "mean_abs_time_shift": float((plan * abs_shift).sum()),
        "p50_abs_time_shift": weighted_quantile(abs_shift, plan, 0.50),
        "p90_abs_time_shift": weighted_quantile(abs_shift, plan, 0.90),
        "diagonal_mass_t005": diag_005,
        "diagonal_mass_t010": diag_010,
        "off_diagonal_mass_t005": 1.0 - diag_005,
        "off_diagonal_mass_t010": 1.0 - diag_010,
        "ref_early_to_target_late_mass": ref_early_to_target_late,
        "ref_late_to_target_early_mass": ref_late_to_target_early,
        "mean_abs_value_delta": float((plan * abs_value_delta).sum()),
        "cost_contribution_p90": weighted_quantile(contribution, plan, 0.90),
    }


def top_contributors(
    plan: np.ndarray,
    cost: np.ndarray,
    ref: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    top_n: int,
) -> list[dict[str, float]]:
    if plan.size == 0:
        return []
    contribution = plan * cost
    flat = contribution.ravel()
    if flat.size == 0:
        return []
    top_n = min(int(top_n), flat.size)
    idxs = np.argpartition(flat, -top_n)[-top_n:]
    idxs = idxs[np.argsort(flat[idxs])[::-1]]
    rows = []
    for rank, flat_idx in enumerate(idxs, start=1):
        i, j = np.unravel_index(int(flat_idx), contribution.shape)
        rows.append(
            {
                "rank": rank,
                "ref_index": int(i),
                "target_index": int(j),
                "ref_t_norm": float(ref["t"][i]),
                "target_t_norm": float(target["t"][j]),
                "signed_time_shift": float(target["t"][j] - ref["t"][i]),
                "abs_time_shift": float(abs(target["t"][j] - ref["t"][i])),
                "ref_value": float(ref["x"][i]),
                "target_value": float(target["x"][j]),
                "abs_value_delta": float(abs(target["x"][j] - ref["x"][i])),
                "plan_mass": float(plan[i, j]),
                "pair_cost": float(cost[i, j]),
                "cost_contribution": float(contribution[i, j]),
            }
        )
    return rows


def resample_plan(plan: np.ndarray, rows: int = 80, cols: int = 80) -> np.ndarray:
    if plan.size == 0:
        return np.zeros((rows, cols), dtype=float)
    r_src = np.linspace(0.0, 1.0, plan.shape[0])
    c_src = np.linspace(0.0, 1.0, plan.shape[1])
    r_dst = np.linspace(0.0, 1.0, rows)
    c_dst = np.linspace(0.0, 1.0, cols)
    tmp = np.vstack([np.interp(r_dst, r_src, plan[:, j]) for j in range(plan.shape[1])]).T
    return np.vstack([np.interp(c_dst, c_src, tmp[i, :]) for i in range(tmp.shape[0])])


def standard_error(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1) / math.sqrt(len(values)))


def save_epoch_plots(output_dir: Path, combined: pd.DataFrame) -> None:
    for cost_function in COST_FUNCTIONS:
        sub = combined[combined["cost_function"] == cost_function]
        for metric in ["combined_ot_distance", "combined_mean_abs_time_shift", "combined_diagonal_mass_t005"]:
            fig, ax = plt.subplots(figsize=(9, 5))
            for comparison, group in sub.groupby("comparison"):
                agg = (
                    group.groupby("epoch")[metric]
                    .agg(["mean", standard_error])
                    .reset_index()
                    .rename(columns={"standard_error": "se"})
                )
                ax.plot(agg["epoch"], agg["mean"], marker="o", label=comparison)
                ax.fill_between(
                    agg["epoch"],
                    agg["mean"] - agg["se"],
                    agg["mean"] + agg["se"],
                    alpha=0.18,
                )
            ax.set_title(f"{cost_function}: {metric}")
            ax.set_xlabel("Target epoch")
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / f"{cost_function}_{metric}_by_epoch.png", dpi=180)
            plt.close(fig)


def save_signal_delta_plot(output_dir: Path, summary: pd.DataFrame) -> None:
    for cost_function in COST_FUNCTIONS:
        sub = summary[summary["cost_function"] == cost_function]
        metric = "ot_distance"
        pivot = (
            sub.groupby(["comparison", "signal"])[metric]
            .mean()
            .unstack("comparison")
            .reindex(SIGNALS)
        )
        if "clean_vs_clean" not in pivot or "clean_vs_blurring" not in pivot:
            continue
        delta = pivot["clean_vs_blurring"] - pivot["clean_vs_clean"]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(delta.index, delta.values)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"{cost_function}: blurring minus clean OT distance by signal")
        ax.set_xlabel("Signal")
        ax.set_ylabel("OT distance delta")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"{cost_function}_signal_ot_distance_delta.png", dpi=180)
        plt.close(fig)


def save_representative_heatmaps(
    output_dir: Path,
    representative_plans: dict[tuple[str, str], dict[str, Any]],
) -> None:
    for cost_function in COST_FUNCTIONS:
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(11, 4.8),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        for ax, comparison in zip(axes, ["clean_vs_clean", "clean_vs_blurring"], strict=True):
            item = representative_plans.get((cost_function, comparison))
            if item is None:
                ax.set_title(f"{comparison}: no plan")
                ax.axis("off")
                continue
            plan_small = resample_plan(item["plan"])
            im = ax.imshow(
                np.log10(plan_small + 1e-12),
                origin="lower",
                aspect="auto",
                extent=[0, 1, 0, 1],
                cmap="magma",
            )
            ax.plot([0, 1], [0, 1], color="cyan", linewidth=1.0, alpha=0.8)
            ax.set_title(
                f"{comparison}\n{item['run_label']} epoch {item['epoch']} {item['signal']}"
            )
            ax.set_xlabel("target normalized time")
            ax.set_ylabel("reference normalized time")
        fig.colorbar(im, ax=axes, label="log10(plan mass)")
        fig.suptitle(f"{cost_function}: representative high-OT transport plans")
        fig.savefig(output_dir / f"{cost_function}_representative_transport_plan_heatmaps.png", dpi=180)
        plt.close(fig)


def build_combined_rows(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_cols = [
        "ot_distance",
        "mean_abs_time_shift",
        "mean_signed_time_shift",
        "diagonal_mass_t005",
        "off_diagonal_mass_t005",
        "mean_abs_value_delta",
    ]
    group_cols = ["comparison", "run_label", "epoch", "cost_function"]
    for key, group in summary.groupby(group_cols):
        row = dict(zip(group_cols, key, strict=True))
        cores = group[group["signal"].isin(CORE_COLS)]
        for metric in metric_cols:
            core_value = float(cores[metric].mean()) if not cores.empty else float("nan")
            row[f"core_mean_{metric}"] = core_value
            row[f"combined_{metric}"] = core_value
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["cost_function", "comparison", "run_label", "epoch"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--logs-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--ot-script", type=Path, default=DEFAULT_OT_SCRIPT)
    parser.add_argument("--reference-name", default=DEFAULT_REFERENCE)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--reg-scale", type=float, default=0.05)
    parser.add_argument("--window-size", type=int, default=5)
    args = parser.parse_args()

    analysis_root = args.analysis_root.resolve()
    logs_dir = args.logs_dir.resolve() if args.logs_dir else analysis_root / "logs"
    output_dir = args.output_dir.resolve() if args.output_dir else analysis_root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))

    ot = load_ot_module(args.ot_script.resolve())
    reference_path, clean_targets, blurring_targets = discover_runs(logs_dir, args.reference_name)

    runs = [
        ("reference", "reference", reference_path),
        *[("clean_vs_clean", path.stem, path) for path in clean_targets],
        *[("clean_vs_blurring", path.stem, path) for path in blurring_targets],
    ]

    run_measures = {}
    for comparison, label, path in runs:
        df = load_run(path, comparison=comparison, label=label)
        run_measures[(comparison, label)] = {
            "path": path,
            "measures": prepare_run_measures(ot, df),
        }

    reference_measures = {
        signal: aggregate_measure(ot, run_measures[("reference", "reference")]["measures"][signal])
        for signal in SIGNALS
    }

    summary_rows = []
    top_rows = []
    representative_plans: dict[tuple[str, str], dict[str, Any]] = {}

    target_items = [
        ((comparison, label), data)
        for (comparison, label), data in run_measures.items()
        if comparison != "reference"
    ]

    for cost_function in COST_FUNCTIONS:
        for (comparison, label), data in target_items:
            epochs = sorted(data["measures"]["core_0"])
            for epoch in epochs:
                for signal in SIGNALS:
                    ref_measure = reference_measures[signal]
                    target_measure = data["measures"][signal][epoch]
                    plan, cost, _, details = ot._transport_plan(
                        ref_measure,
                        target_measure,
                        reg_scale=float(args.reg_scale),
                        window_size=int(args.window_size),
                        cost_function=cost_function,
                        return_details=True,
                    )
                    metrics = plan_metrics(plan, cost, ref_measure, target_measure)
                    summary_row = {
                        "comparison": comparison,
                        "run_label": label,
                        "run_csv": str(data["path"]),
                        "epoch": epoch,
                        "signal": signal,
                        "cost_function": cost_function,
                        "reg": details["reg"],
                        "n_ref": len(ref_measure["t"]),
                        "n_target": len(target_measure["t"]),
                        "time_cost_mean": details["time_cost_mean"],
                        "window_cost_mean": details["window_cost_mean"],
                        **metrics,
                    }
                    summary_rows.append(summary_row)

                    rep_key = (cost_function, comparison)
                    previous = representative_plans.get(rep_key)
                    if previous is None or summary_row["ot_distance"] > previous["ot_distance"]:
                        representative_plans[rep_key] = {
                            "plan": plan,
                            "cost": cost,
                            "ot_distance": summary_row["ot_distance"],
                            "run_label": label,
                            "epoch": epoch,
                            "signal": signal,
                        }

                    for top in top_contributors(plan, cost, ref_measure, target_measure, args.top_n):
                        top_rows.append(
                            {
                                "comparison": comparison,
                                "run_label": label,
                                "epoch": epoch,
                                "signal": signal,
                                "cost_function": cost_function,
                                **top,
                            }
                        )

    summary = pd.DataFrame(summary_rows)
    top = pd.DataFrame(top_rows)
    combined = build_combined_rows(summary)

    signal_summary = (
        summary.groupby(["cost_function", "comparison", "signal"])
        .agg(
            ot_distance_mean=("ot_distance", "mean"),
            ot_distance_median=("ot_distance", "median"),
            mean_abs_time_shift=("mean_abs_time_shift", "mean"),
            mean_signed_time_shift=("mean_signed_time_shift", "mean"),
            diagonal_mass_t005=("diagonal_mass_t005", "mean"),
            off_diagonal_mass_t005=("off_diagonal_mass_t005", "mean"),
            mean_abs_value_delta=("mean_abs_value_delta", "mean"),
            num_plans=("ot_distance", "size"),
        )
        .reset_index()
    )

    diff_rows = []
    for cost_function in COST_FUNCTIONS:
        for signal in SIGNALS:
            clean = signal_summary[
                (signal_summary["cost_function"] == cost_function)
                & (signal_summary["comparison"] == "clean_vs_clean")
                & (signal_summary["signal"] == signal)
            ]
            blur = signal_summary[
                (signal_summary["cost_function"] == cost_function)
                & (signal_summary["comparison"] == "clean_vs_blurring")
                & (signal_summary["signal"] == signal)
            ]
            if clean.empty or blur.empty:
                continue
            diff_rows.append(
                {
                    "cost_function": cost_function,
                    "signal": signal,
                    "ot_distance_delta_blurring_minus_clean": float(
                        blur["ot_distance_mean"].iloc[0] - clean["ot_distance_mean"].iloc[0]
                    ),
                    "abs_time_shift_delta_blurring_minus_clean": float(
                        blur["mean_abs_time_shift"].iloc[0] - clean["mean_abs_time_shift"].iloc[0]
                    ),
                    "diagonal_mass_delta_blurring_minus_clean": float(
                        blur["diagonal_mass_t005"].iloc[0] - clean["diagonal_mass_t005"].iloc[0]
                    ),
                    "abs_value_delta_blurring_minus_clean": float(
                        blur["mean_abs_value_delta"].iloc[0] - clean["mean_abs_value_delta"].iloc[0]
                    ),
                }
            )

    summary.to_csv(output_dir / "transport_plan_plan_summary.csv", index=False)
    combined.to_csv(output_dir / "transport_plan_combined_epoch_summary.csv", index=False)
    signal_summary.to_csv(output_dir / "transport_plan_signal_summary.csv", index=False)
    pd.DataFrame(diff_rows).to_csv(output_dir / "transport_plan_signal_differences.csv", index=False)
    top.to_csv(output_dir / "transport_plan_top_cost_contributors.csv", index=False)

    save_epoch_plots(output_dir, combined)
    save_signal_delta_plot(output_dir, summary)
    save_representative_heatmaps(output_dir, representative_plans)

    print(f"Reference: {reference_path}")
    print(f"Clean targets: {len(clean_targets)}")
    print(f"Blurring targets: {len(blurring_targets)}")
    print(f"Wrote results to: {output_dir}")
    print("\nSignal-level blurring minus clean differences:")
    if diff_rows:
        print(pd.DataFrame(diff_rows).to_string(index=False))
    else:
        print("No differences computed.")


if __name__ == "__main__":
    main()
