"""Analyze replay-only perf-stat TopdownL1 measurements."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/antivenom-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = (
    "tma_retiring",
    "tma_backend_bound",
    "tma_frontend_bound",
    "tma_bad_speculation",
)
BAD_SPEC_METRICS = (
    "tma_bad_speculation",
    "tma_branch_mispredicts",
    "tma_machine_clears",
    "tma_mispredicts_resteers",
    "tma_clears_resteers",
)
ALL_METRICS = tuple(dict.fromkeys((*METRICS, *BAD_SPEC_METRICS)))
LABELS = {
    "tma_retiring": "Retiring",
    "tma_backend_bound": "Backend bound",
    "tma_frontend_bound": "Frontend bound",
    "tma_bad_speculation": "Bad speculation",
    "tma_branch_mispredicts": "Branch mispredicts",
    "tma_machine_clears": "Machine clears",
    "tma_mispredicts_resteers": "Mispredict resteers",
    "tma_clears_resteers": "Machine-clear resteers",
}
COLORS = {
    "tma_retiring": "#2F855A",
    "tma_backend_bound": "#C23B4A",
    "tma_frontend_bound": "#2878B5",
    "tma_bad_speculation": "#E07A1F",
    "tma_branch_mispredicts": "#D95F02",
    "tma_machine_clears": "#7570B3",
    "tma_mispredicts_resteers": "#E7298A",
    "tma_clears_resteers": "#66A61E",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--minimum-running-percent", type=float, default=99.0)
    parser.add_argument("--sum-tolerance", type=float, default=2.0)
    parser.add_argument("--format", choices=("png", "pdf"), default="png")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def numeric(value: Any) -> float:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return np.nan


def metric_name(unit: str) -> str:
    match = re.search(r"tma_[a-z_]+", unit)
    return match.group(0) if match else ""


def load_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Topdown output: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def load_runs(
    manifest: pd.DataFrame,
    minimum_running_percent: float,
    sum_tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for _, meta in manifest.iterrows():
        result_path = Path(meta["json_path"])
        result = json.loads(result_path.read_text())
        repeats = numeric(result.get("config", {}).get("repeats"))
        metric_pass = str(meta.get("metric_pass", "topdown_l1") or "topdown_l1")
        entries = load_json_lines(Path(meta["topdown_jsonl"]))
        metrics: dict[str, float] = {}
        running_values: list[float] = []
        for entry in entries:
            name = metric_name(str(entry.get("metric-unit", "")))
            if name in ALL_METRICS:
                metrics[name] = numeric(entry.get("metric-value"))
            event = str(entry.get("event", "")).strip()
            if event:
                running = numeric(entry.get("pcnt-running"))
                if np.isfinite(running):
                    running_values.append(running)
                raw_rows.append({
                    **meta.to_dict(),
                    "event": event,
                    "counter_value": numeric(entry.get("counter-value")),
                    "event_runtime": numeric(entry.get("event-runtime")),
                    "running_percent": running,
                    "repeats": repeats,
                    "counter_per_call": (
                        numeric(entry.get("counter-value")) / repeats
                        if np.isfinite(repeats) and repeats != 0 else np.nan
                    ),
                })

        required_metrics = (
            BAD_SPEC_METRICS if metric_pass == "bad_speculation" else METRICS
        )
        missing = [name for name in required_metrics if name not in metrics]
        metric_sum = sum(metrics.get(name, np.nan) for name in METRICS)
        bad_spec_component_sum = (
            metrics.get("tma_branch_mispredicts", np.nan)
            + metrics.get("tma_machine_clears", np.nan)
        )
        bad_spec_balance_error = (
            bad_spec_component_sum - metrics.get("tma_bad_speculation", np.nan)
        )
        min_running = min(running_values) if running_values else np.nan
        running_valid = (
            np.isfinite(min_running) and min_running >= minimum_running_percent
        )
        if metric_pass == "bad_speculation":
            valid = (
                not missing
                and running_valid
                and np.isfinite(bad_spec_balance_error)
                and abs(bad_spec_balance_error) <= sum_tolerance
            )
        else:
            valid = (
                not missing
                and np.isfinite(metric_sum)
                and abs(metric_sum - 100.0) <= sum_tolerance
                and running_valid
            )
        run_rows.append({
            **meta.to_dict(),
            "metric_pass": metric_pass,
            **{f"{name}_percent": metrics.get(name, np.nan) for name in ALL_METRICS},
            "topdown_sum_percent": metric_sum,
            "bad_spec_component_sum_percent": bad_spec_component_sum,
            "bad_spec_balance_error_pp": bad_spec_balance_error,
            "minimum_event_running_percent": min_running,
            "missing_metrics": ",".join(missing),
            "valid": valid,
            "elapsed_seconds": numeric(result.get("elapsed_seconds")),
            "nanoseconds_per_call": numeric(result.get("nanoseconds_per_call")),
            "repeats": repeats,
        })
    return pd.DataFrame.from_records(run_rows), pd.DataFrame.from_records(raw_rows)


def summarize(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_columns = [f"{name}_percent" for name in METRICS]
    valid = runs[(runs["valid"]) & (runs["metric_pass"] == "topdown_l1")].copy()
    if valid.empty:
        return pd.DataFrame(), pd.DataFrame()
    summary = (
        valid.groupby(["operator", "scope", "regime"], observed=True)
        [metric_columns + ["nanoseconds_per_call"]]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(item) for item in column if item).rstrip("_")
        if isinstance(column, tuple) else str(column)
        for column in summary.columns
    ]

    comparisons: list[dict[str, Any]] = []
    for (operator, scope), subset in summary.groupby(["operator", "scope"], observed=True):
        row: dict[str, Any] = {"operator": operator, "scope": scope}
        for name in METRICS:
            column = f"{name}_percent_mean"
            low = subset.loc[subset["regime"] == "low", column]
            high = subset.loc[subset["regime"] == "high", column]
            low_value = float(low.iloc[0]) if not low.empty else np.nan
            high_value = float(high.iloc[0]) if not high.empty else np.nan
            row[f"low_{name}_percent"] = low_value
            row[f"high_{name}_percent"] = high_value
            row[f"high_minus_low_{name}_pp"] = high_value - low_value
        comparisons.append(row)
    return summary, pd.DataFrame.from_records(comparisons)


def summarize_bad_speculation(
    runs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_columns = [f"{name}_percent" for name in BAD_SPEC_METRICS]
    valid = runs[
        (runs["valid"]) & (runs["metric_pass"] == "bad_speculation")
    ].copy()
    if valid.empty:
        return pd.DataFrame(), pd.DataFrame()
    summary = (
        valid.groupby(["operator", "scope", "regime"], observed=True)
        [metric_columns + ["nanoseconds_per_call"]]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(item) for item in column if item).rstrip("_")
        if isinstance(column, tuple) else str(column)
        for column in summary.columns
    ]

    comparisons: list[dict[str, Any]] = []
    for (operator, scope), subset in summary.groupby(
        ["operator", "scope"], observed=True
    ):
        row: dict[str, Any] = {"operator": operator, "scope": scope}
        for name in BAD_SPEC_METRICS:
            column = f"{name}_percent_mean"
            low = subset.loc[subset["regime"] == "low", column]
            high = subset.loc[subset["regime"] == "high", column]
            low_value = float(low.iloc[0]) if not low.empty else np.nan
            high_value = float(high.iloc[0]) if not high.empty else np.nan
            row[f"low_{name}_percent"] = low_value
            row[f"high_{name}_percent"] = high_value
            row[f"high_minus_low_{name}_pp"] = high_value - low_value
        comparisons.append(row)
    return summary, pd.DataFrame.from_records(comparisons)


def plot_comparisons(
    comparison: pd.DataFrame,
    output_dir: Path,
    file_format: str,
    dpi: int,
) -> list[Path]:
    written: list[Path] = []
    for _, row in comparison.iterrows():
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
        bottoms = np.zeros(2)
        for name in METRICS:
            values = np.array([
                row[f"low_{name}_percent"], row[f"high_{name}_percent"]
            ])
            axes[0].bar(["Low", "High"], values, bottom=bottoms,
                        color=COLORS[name], label=LABELS[name])
            bottoms += values
        axes[0].set_ylim(0, 105)
        axes[0].set_ylabel("Pipeline slots (%)")
        axes[0].set_title("TopdownL1 composition")
        axes[0].legend(frameon=False, fontsize=8)

        deltas = [row[f"high_minus_low_{name}_pp"] for name in METRICS]
        labels = [LABELS[name] for name in METRICS]
        colors = [COLORS[name] for name in METRICS]
        axes[1].barh(labels, deltas, color=colors)
        axes[1].axvline(0, color="#24292F", linewidth=0.8)
        axes[1].set_xlabel("High - low (percentage points)")
        axes[1].set_title("Entropy-associated shift")
        for axis in axes:
            axis.grid(axis="y", alpha=0.25)
            axis.spines[["top", "right"]].set_visible(False)
        figure.suptitle(f"{row['operator']} | {row['scope']} | TopdownL1")
        path = output_dir / (
            f"topdownL1_{row['operator']}_{row['scope']}.{file_format}"
        )
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        written.append(path)
    return written


def plot_bad_speculation(
    comparison: pd.DataFrame,
    output_dir: Path,
    file_format: str,
    dpi: int,
) -> list[Path]:
    written: list[Path] = []
    for _, row in comparison.iterrows():
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

        bottoms = np.zeros(2)
        for name in ("tma_branch_mispredicts", "tma_machine_clears"):
            values = np.array([
                row[f"low_{name}_percent"], row[f"high_{name}_percent"]
            ])
            axes[0].bar(
                ["Low", "High"], values, bottom=bottoms,
                color=COLORS[name], label=LABELS[name],
            )
            bottoms += values
        totals = np.array([
            row["low_tma_bad_speculation_percent"],
            row["high_tma_bad_speculation_percent"],
        ])
        axes[0].scatter([0, 1], totals, marker="_", s=350, linewidth=2,
                        color="#24292F", label="Bad speculation total")
        axes[0].set_ylabel("Pipeline slots (%)")
        axes[0].set_title("Bad-speculation decomposition")
        axes[0].legend(frameon=False, fontsize=8)

        resteer_metrics = ("tma_mispredicts_resteers", "tma_clears_resteers")
        x = np.arange(len(resteer_metrics))
        width = 0.36
        low_values = [row[f"low_{name}_percent"] for name in resteer_metrics]
        high_values = [row[f"high_{name}_percent"] for name in resteer_metrics]
        axes[1].bar(x - width / 2, low_values, width, color="#2878B5", label="Low")
        axes[1].bar(x + width / 2, high_values, width, color="#C23B4A", label="High")
        axes[1].set_xticks(x, [LABELS[name] for name in resteer_metrics], rotation=15)
        axes[1].set_ylabel("Cycles / slots (%)")
        axes[1].set_title("Recovery/resteer pressure")
        axes[1].legend(frameon=False)

        deltas = [row[f"high_minus_low_{name}_pp"] for name in BAD_SPEC_METRICS]
        labels = [LABELS[name] for name in BAD_SPEC_METRICS]
        colors = [COLORS[name] for name in BAD_SPEC_METRICS]
        axes[2].barh(labels, deltas, color=colors)
        axes[2].axvline(0, color="#24292F", linewidth=0.8)
        axes[2].set_xlabel("High - low (percentage points)")
        axes[2].set_title("Entropy-associated shift")

        for axis in axes:
            axis.grid(axis="y", alpha=0.25)
            axis.spines[["top", "right"]].set_visible(False)
        figure.suptitle(f"{row['operator']} | {row['scope']} | Bad speculation")
        path = output_dir / (
            f"bad_speculation_{row['operator']}_{row['scope']}.{file_format}"
        )
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        written.append(path)
    return written


def summarize_bad_spec_raw_events(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty or "metric_pass" not in raw.columns:
        return pd.DataFrame()
    selected_events = (
        "TOPDOWN.SLOTS",
        "BR_MISP_RETIRED.ALL_BRANCHES",
        "MACHINE_CLEARS.COUNT",
        "INT_MISC.CLEAR_RESTEER_CYCLES",
    )
    selected = raw[
        (raw["metric_pass"] == "bad_speculation")
        & (raw["event"].isin(selected_events))
    ].copy()
    if selected.empty:
        return pd.DataFrame()
    # Some metric formulas request the same raw event more than once.
    selected = selected.drop_duplicates(
        ["operator", "scope", "regime", "trial_id", "event"]
    )
    return (
        selected.groupby(
            ["operator", "scope", "regime", "event"], observed=True,
            as_index=False,
        )["counter_per_call"]
        .agg(["mean", "std", "count"])
    )


def plot_bad_spec_raw_events(
    summary: pd.DataFrame,
    output_dir: Path,
    file_format: str,
    dpi: int,
) -> list[Path]:
    written: list[Path] = []
    if summary.empty:
        return written
    for (operator, scope), subset in summary.groupby(
        ["operator", "scope"], observed=True
    ):
        events = list(dict.fromkeys(subset["event"]))
        figure, axes = plt.subplots(
            1, len(events), figsize=(4.2 * len(events), 4.1), squeeze=False,
            constrained_layout=True,
        )
        for axis, event in zip(axes[0], events):
            event_rows = subset[subset["event"] == event]
            means = event_rows.set_index("regime")["mean"]
            stds = event_rows.set_index("regime")["std"]
            values = [means.get(regime, np.nan) for regime in ("low", "high")]
            errors = [stds.get(regime, 0.0) for regime in ("low", "high")]
            errors = [0.0 if not np.isfinite(value) else value for value in errors]
            axis.bar(
                ["Low", "High"], values, yerr=errors, capsize=3,
                color=["#2878B5", "#C23B4A"],
            )
            axis.set_title(event, fontsize=9)
            axis.set_ylabel("Counter / operator call")
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
            axis.grid(axis="y", alpha=0.25)
            axis.spines[["top", "right"]].set_visible(False)
        figure.suptitle(f"{operator} | {scope} | Bad-speculation raw events")
        path = output_dir / (
            f"bad_speculation_raw_{operator}_{scope}.{file_format}"
        )
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        written.append(path)
    return written


def run_analysis(
    input_dir: Path,
    output_dir: Path,
    minimum_running_percent: float = 99.0,
    sum_tolerance: float = 2.0,
    file_format: str = "png",
    dpi: int = 180,
) -> tuple[pd.DataFrame, list[Path]]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = input_dir / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path, dtype=str).fillna("")
    required = {
        "operator", "regime", "trial_id", "scope", "json_path",
        "topdown_jsonl",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(
            f"Topdown manifest is missing columns: {', '.join(missing)}"
        )
    runs, raw = load_runs(
        manifest, minimum_running_percent, sum_tolerance
    )
    summary, comparison = summarize(runs)
    bad_spec_summary, bad_spec_comparison = summarize_bad_speculation(runs)
    bad_spec_raw_summary = summarize_bad_spec_raw_events(raw)

    runs.to_csv(output_dir / "topdown_runs.csv", index=False)
    raw.to_csv(output_dir / "topdown_raw_events.csv", index=False)
    summary.to_csv(output_dir / "topdown_summary.csv", index=False)
    comparison.to_csv(output_dir / "topdown_high_vs_low.csv", index=False)
    bad_spec_summary.to_csv(
        output_dir / "bad_speculation_summary.csv", index=False
    )
    bad_spec_comparison.to_csv(
        output_dir / "bad_speculation_high_vs_low.csv", index=False
    )
    bad_spec_raw_summary.to_csv(
        output_dir / "bad_speculation_raw_events.csv", index=False
    )
    written = plot_comparisons(comparison, output_dir, file_format, dpi)
    written.extend(plot_bad_speculation(
        bad_spec_comparison, output_dir, file_format, dpi
    ))
    written.extend(plot_bad_spec_raw_events(
        bad_spec_raw_summary, output_dir, file_format, dpi
    ))

    print(f"Loaded {len(runs)} Topdown metric runs from {input_dir}")
    print(f"Valid runs: {int(runs['valid'].sum())}/{len(runs)}")
    if (~runs["valid"]).any():
        print("Invalid runs are retained in topdown_runs.csv but excluded from summaries.")
    print(f"Output: {output_dir}")
    for path in written:
        print(path)
    return runs, written


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else input_dir / "analysis"
    run_analysis(
        input_dir=input_dir,
        output_dir=output_dir,
        minimum_running_percent=args.minimum_running_percent,
        sum_tolerance=args.sum_tolerance,
        file_format=args.format,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
