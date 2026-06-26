#!/usr/bin/env python3
"""Run c7 window decomposition across batch/device poisoning logs."""

from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import json
import math
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
POISON_DIRS = ["blurring", "occlusion", "label-flip"]


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_run_info(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
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


def load_run(path: Path, batch_id: str, condition: str) -> pd.DataFrame:
    info = extract_run_info(path)
    df = pd.read_csv(path)
    df["batch_id"] = batch_id
    df["condition"] = condition
    df["run_label"] = path.stem
    df["run_csv"] = str(path)
    poison = str(info.get("poison_type", condition))
    df["poisoning_type"] = "clean" if poison == "none" else poison
    df["poison_frac"] = pd.to_numeric(info.get("poison_frac"), errors="coerce")
    df["platform"] = str(info.get("platform", ""))
    return df


def pairwise_squared_l2(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    diff = left[:, None, :] - right[None, :, :]
    return np.sum(diff * diff, axis=2)


def aggregate_measure(ot: Any, measures: dict[int, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return ot._aggregate_run_measure([measures[epoch] for epoch in sorted(measures)])


def thin_measure(measure: dict[str, np.ndarray], max_points: int) -> dict[str, np.ndarray]:
    n = len(measure["t"])
    if max_points <= 0 or n <= max_points:
        return measure
    idx = np.unique(np.linspace(0, n - 1, max_points).round().astype(int))
    thinned = {
        "t": np.asarray(measure["t"])[idx],
        "x": np.asarray(measure["x"])[idx],
        "w": np.asarray(measure["w"])[idx],
        "_shape_cache": {},
        "_delta_cache": None,
        "_frequency_cache": {},
        "_curvature_cache": None,
        "_zscore_value_cache": None,
        "_local_variance_cache": {},
    }
    weight_sum = float(np.sum(thinned["w"]))
    if weight_sum > 0:
        thinned["w"] = thinned["w"] / weight_sum
    return thinned


def prepare_measures(ot: Any, df: pd.DataFrame) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    prepared = ot._prepare_df(df)
    measures: dict[str, dict[int, dict[str, np.ndarray]]] = {signal: {} for signal in CORE_COLS}
    for epoch, group in prepared.groupby("epoch"):
        epoch_idx = int(epoch)
        for signal in CORE_COLS:
            measures[signal][epoch_idx] = ot._epoch_measure(group, signal)
    return measures


def raw_core_stats(ot: Any, df: pd.DataFrame) -> list[dict[str, Any]]:
    prepared = ot._prepare_df(df)
    rows = []
    for signal in CORE_COLS:
        values = prepared[signal].astype(float)
        rows.append(
            {
                "signal": signal,
                "mean": values.mean(),
                "std": values.std(),
                "p10": values.quantile(0.10),
                "p50": values.quantile(0.50),
                "p90": values.quantile(0.90),
            }
        )
    return rows


def decompose_window_cost(
    ot: Any,
    ref_measure: dict[str, np.ndarray],
    target_measure: dict[str, np.ndarray],
    plan: np.ndarray,
    window_size: int,
) -> dict[str, float]:
    ref_window = ot._shape_features(ref_measure, window_size, False, "edge")
    target_window = ot._shape_features(target_measure, window_size, False, "edge")

    ref_mean = ref_window.mean(axis=1, keepdims=True)
    target_mean = target_window.mean(axis=1, keepdims=True)
    width = ref_window.shape[1]
    level_cost = width * (ref_mean - target_mean.T) ** 2
    centered_cost = pairwise_squared_l2(ref_window - ref_mean, target_window - target_mean)

    level = float(np.sum(plan * level_cost))
    centered = float(np.sum(plan * centered_cost))
    total = level + centered
    return {
        "level_cost": level,
        "centered_shape_cost": centered,
        "level_share": level / total if total > 0 else float("nan"),
        "centered_shape_share": centered / total if total > 0 else float("nan"),
    }


def discover_batches(batch_root: Path) -> dict[str, dict[str, list[Path]]]:
    result: dict[str, dict[str, list[Path]]] = {}
    for batch_dir in sorted(batch_root.glob("logs_batch_*")):
        inner = batch_dir / "logs_batch"
        if not inner.exists():
            continue
        batch_id = batch_dir.name.replace("logs_batch_", "batch_")
        result[batch_id] = {}
        for condition in ["clean", *POISON_DIRS]:
            files = sorted((inner / condition).glob("*.csv"))
            if files:
                result[batch_id][condition] = files
    return result


def choose_reference(clean_files: list[Path]) -> Path:
    scored = []
    for path in clean_files:
        info = extract_run_info(path)
        frac = pd.to_numeric(info.get("poison_frac"), errors="coerce")
        frac_value = float(frac) if pd.notna(frac) else -math.inf
        scored.append((frac_value, path.name, path))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--batch-root", type=Path, default=None)
    parser.add_argument("--ot-script", type=Path, default=DEFAULT_OT_SCRIPT)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--reg-scale", type=float, default=0.05)
    parser.add_argument("--max-ref-points", type=int, default=240)
    parser.add_argument("--max-target-points", type=int, default=120)
    args = parser.parse_args()

    analysis_root = args.analysis_root.resolve()
    batch_root = args.batch_root.resolve() if args.batch_root else analysis_root / "logs" / "batches"
    output_dir = analysis_root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))

    import matplotlib.pyplot as plt

    ot = load_module("cost_ot", args.ot_script.resolve())
    batches = discover_batches(batch_root)
    if not batches:
        raise FileNotFoundError(f"No batch logs found under {batch_root}")

    plan_rows = []
    raw_rows = []
    reference_rows = []

    for batch_id, conditions in batches.items():
        clean_files = conditions.get("clean", [])
        if not clean_files:
            continue
        reference = choose_reference(clean_files)
        reference_rows.append({"batch_id": batch_id, "reference_csv": str(reference)})
        reference_df = load_run(reference, batch_id=batch_id, condition="reference")
        reference_measures_by_epoch = prepare_measures(ot, reference_df)
        reference_measures = {
            signal: thin_measure(
                aggregate_measure(ot, reference_measures_by_epoch[signal]),
                max_points=int(args.max_ref_points),
            )
            for signal in CORE_COLS
        }

        targets: list[tuple[str, Path]] = []
        targets.extend(("clean", path) for path in clean_files if path.resolve() != reference.resolve())
        for poison in POISON_DIRS:
            targets.extend((poison, path) for path in conditions.get(poison, []))

        for condition, path in targets:
            df = load_run(path, batch_id=batch_id, condition=condition)
            info = extract_run_info(path)
            comparison = "clean_vs_clean" if condition == "clean" else f"clean_vs_{condition}"
            for row in raw_core_stats(ot, df):
                raw_rows.append(
                    {
                        "batch_id": batch_id,
                        "condition": condition,
                        "comparison": comparison,
                        "run_label": path.stem,
                        "poison_frac": info.get("poison_frac"),
                        **row,
                    }
                )

            target_measures = prepare_measures(ot, df)
            for epoch in sorted(target_measures["core_0"]):
                for signal in CORE_COLS:
                    ref_measure = reference_measures[signal]
                    target_measure = thin_measure(
                        target_measures[signal][epoch],
                        max_points=int(args.max_target_points),
                    )
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
                    plan_rows.append(
                        {
                            "batch_id": batch_id,
                            "condition": condition,
                            "comparison": comparison,
                            "run_label": path.stem,
                            "run_csv": str(path),
                            "poison_frac": info.get("poison_frac"),
                            "epoch": epoch,
                            "signal": signal,
                            "ot_distance": float(np.sum(plan * cost)),
                            **pieces,
                        }
                    )

    plan_df = pd.DataFrame(plan_rows)
    raw_df = pd.DataFrame(raw_rows)
    ref_df = pd.DataFrame(reference_rows)

    summary = (
        plan_df.groupby(["batch_id", "comparison", "condition", "signal"])
        [
            [
                "ot_distance",
                "level_cost",
                "centered_shape_cost",
                "level_share",
                "centered_shape_share",
            ]
        ]
        .mean()
        .reset_index()
    )

    baseline = summary[summary["comparison"] == "clean_vs_clean"].copy()
    deltas = []
    for _, row in summary[summary["comparison"] != "clean_vs_clean"].iterrows():
        base = baseline[
            (baseline["batch_id"] == row["batch_id"])
            & (baseline["signal"] == row["signal"])
        ]
        if base.empty:
            continue
        base_row = base.iloc[0]
        deltas.append(
            {
                "batch_id": row["batch_id"],
                "condition": row["condition"],
                "comparison": row["comparison"],
                "signal": row["signal"],
                "ot_distance_delta": row["ot_distance"] - base_row["ot_distance"],
                "level_cost_delta": row["level_cost"] - base_row["level_cost"],
                "centered_shape_cost_delta": row["centered_shape_cost"] - base_row["centered_shape_cost"],
            }
        )
    delta_df = pd.DataFrame(deltas)

    condition_summary = (
        delta_df.groupby(["condition", "signal"])
        [
            [
                "ot_distance_delta",
                "level_cost_delta",
                "centered_shape_cost_delta",
            ]
        ]
        .agg(["mean", "median", "std", "count"])
    )
    condition_summary.columns = ["_".join(col).strip() for col in condition_summary.columns.to_flat_index()]
    condition_summary = condition_summary.reset_index()

    verdict_rows = []
    for condition, group in delta_df.groupby("condition"):
        core0 = group[group["signal"] == "core_0"]["centered_shape_cost_delta"]
        other = group[group["signal"].isin(["core_1", "core_2", "core_3"])]["centered_shape_cost_delta"]
        verdict_rows.append(
            {
                "condition": condition,
                "core0_centered_shape_delta_mean": core0.mean(),
                "core123_centered_shape_delta_mean": other.mean(),
                "core123_minus_core0": other.mean() - core0.mean(),
                "batches": group["batch_id"].nunique(),
                "same_trend": bool(other.mean() > core0.mean()),
            }
        )
    verdict = pd.DataFrame(verdict_rows)

    ref_df.to_csv(output_dir / "batch_c7_references.csv", index=False)
    plan_df.to_csv(output_dir / "batch_c7_core_window_decomposition_per_plan.csv", index=False)
    raw_df.to_csv(output_dir / "batch_c7_core_raw_cpu_stats_per_run.csv", index=False)
    summary.to_csv(output_dir / "batch_c7_core_window_decomposition_summary.csv", index=False)
    delta_df.to_csv(output_dir / "batch_c7_core_window_decomposition_deltas.csv", index=False)
    condition_summary.to_csv(output_dir / "batch_c7_condition_signal_delta_summary.csv", index=False)
    verdict.to_csv(output_dir / "batch_c7_trend_verdict.csv", index=False)

    for metric in ["ot_distance_delta", "centered_shape_cost_delta", "level_cost_delta"]:
        plot = condition_summary.pivot(index="condition", columns="signal", values=f"{metric}_mean")
        plot = plot.reindex(POISON_DIRS)
        fig, ax = plt.subplots(figsize=(9, 5))
        plot[CORE_COLS].plot(kind="bar", ax=ax)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"c7 {metric}: poisoning minus clean-vs-clean")
        ax.set_xlabel("Poisoning condition")
        ax.set_ylabel(metric)
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"batch_c7_{metric}_by_condition.png", dpi=180)
        plt.close(fig)

    print(f"Batches analyzed: {len(batches)}")
    print(f"Wrote outputs to: {output_dir}")
    print("\nTrend verdict:")
    print(verdict.to_string(index=False))
    print("\nCondition/signal mean deltas:")
    print(condition_summary.to_string(index=False))


if __name__ == "__main__":
    main()
