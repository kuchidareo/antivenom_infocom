"""Compare low/high entropy perf-record profiles for MaxPool and Conv2D."""

from __future__ import annotations

import argparse
import csv
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


REGIMES = ("low", "high")
COLORS = {"low": "#2878B5", "high": "#C23B4A"}
PERF_NUMBER = r"[0-9][0-9,.]*(?:[KkMmGgTt])?"
EVENT_HEADER = re.compile(
    rf"^# Samples:\s*({PERF_NUMBER})\s+of event '(.+)'$"
)
EVENT_COUNT = re.compile(
    rf"^# Event count \(approx\.\):\s*({PERF_NUMBER})"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--format", choices=("png", "pdf"), default="png")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def safe_name(value: Any) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value)).strip("_") or "unknown"


def base_event_name(event: str) -> str:
    return re.sub(r":(?:u|k|uk|ku)$", "", event.strip())


def parse_perf_number(value: str) -> float:
    """Parse perf's compact counts, for example 52K or 1.3M."""
    cleaned = value.strip().replace(",", "")
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KkMmGgTt]?)", cleaned)
    if not match:
        raise ValueError(f"Invalid perf number: {value!r}")
    scale = {
        "": 1.0,
        "k": 1e3,
        "m": 1e6,
        "g": 1e9,
        "t": 1e12,
    }[match.group(2).lower()]
    return float(match.group(1)) * scale


def resolve_run_directory(path: Path) -> Path:
    """Accept either one run directory or its timestamped parent directory."""
    path = path.resolve()
    if (path / "manifest.csv").is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {path}")

    candidates = [
        manifest.parent
        for manifest in path.glob("*/manifest.csv")
        if manifest.is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No manifest.csv found in {path} or its immediate subdirectories"
        )
    selected = max(
        candidates,
        key=lambda candidate: (candidate / "manifest.csv").stat().st_mtime,
    )
    print(f"Selected latest perf-record run: {selected}")
    return selected


def read_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing manifest: {path}")
    frame = pd.read_csv(path, dtype=str).fillna("")
    required = {
        "operator", "regime", "trial_id", "scope", "pass_id", "event",
        "run_id", "json_path", "perf_data", "flat_report",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Manifest is missing columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError(f"Manifest contains no profile rows: {path}")
    return frame


def parse_flat_report(path: Path) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    rows: list[dict[str, Any]] = []
    event_stats: dict[str, dict[str, float]] = {}
    current_event = ""
    if not path.is_file():
        raise FileNotFoundError(f"Missing perf report: {path}")

    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        match = EVENT_HEADER.match(line)
        if match:
            current_event = match.group(2)
            event_stats[current_event] = {
                "sample_count": parse_perf_number(match.group(1)),
                "event_count_approx": np.nan,
            }
            continue
        count_match = EVENT_COUNT.match(line)
        if count_match and current_event:
            event_stats[current_event]["event_count_approx"] = parse_perf_number(
                count_match.group(1)
            )
            continue
        if not current_event or not line or line.startswith("#") or ";" not in line:
            continue
        fields = [field.strip() for field in line.split(";", 5)]
        if len(fields) != 6 or not fields[0].endswith("%"):
            continue
        try:
            overhead = float(fields[0].removesuffix("%").strip())
            samples = float(fields[1].replace(",", ""))
            period = float(fields[2].replace(",", ""))
        except ValueError:
            continue
        rows.append({
            "recorded_event": current_event,
            "event": base_event_name(current_event),
            "overhead_percent": overhead,
            "symbol_samples": samples,
            "period": period,
            "comm": fields[3],
            "dso": fields[4],
            "symbol": re.sub(r"^\[[^]]+\]\s*", "", fields[5]).strip(),
        })
    return pd.DataFrame.from_records(rows), event_stats


def load_profiles(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    symbol_frames: list[pd.DataFrame] = []
    event_rows: list[dict[str, Any]] = []
    cache: dict[Path, tuple[pd.DataFrame, dict[str, dict[str, float]]]] = {}

    for report_name, report_manifest in manifest.groupby("flat_report", sort=False):
        report_path = Path(report_name)
        parsed, stats = cache.setdefault(report_path, parse_flat_report(report_path))
        for _, meta in report_manifest.iterrows():
            event = meta["event"]
            matching_names = [name for name in stats if base_event_name(name) == event]
            stat = stats[matching_names[0]] if matching_names else {
                "sample_count": 0.0,
                "event_count_approx": np.nan,
            }
            event_rows.append({
                **meta.to_dict(),
                "sample_count": stat["sample_count"],
                "event_count_approx": stat["event_count_approx"],
            })
            if parsed.empty:
                continue
            subset = parsed[parsed["event"] == event].copy()
            if subset.empty:
                continue
            for column, value in meta.items():
                subset[column] = value
            symbol_frames.append(subset)

    symbols = (
        pd.concat(symbol_frames, ignore_index=True)
        if symbol_frames else pd.DataFrame()
    )
    return symbols, pd.DataFrame.from_records(event_rows)


def aggregate_symbols(symbols: pd.DataFrame, key: str) -> pd.DataFrame:
    if symbols.empty:
        return pd.DataFrame()
    grouped = (
        symbols.groupby(
            ["operator", "scope", "event", "regime", "trial_id", key],
            observed=True,
            as_index=False,
        )["overhead_percent"].sum()
    )
    return (
        grouped.groupby(["operator", "scope", "event", "regime", key], observed=True)
        ["overhead_percent"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_overhead_percent", "std": "std_overhead_percent"})
    )


def complete_regime_table(frame: pd.DataFrame, category: str) -> pd.DataFrame:
    pivot = frame.pivot_table(
        index=category, columns="regime", values="mean_overhead_percent",
        aggfunc="first", fill_value=0.0,
    )
    for regime in REGIMES:
        if regime not in pivot:
            pivot[regime] = 0.0
    pivot["max_overhead"] = pivot[list(REGIMES)].max(axis=1)
    pivot["high_minus_low"] = pivot["high"] - pivot["low"]
    return pivot.sort_values("max_overhead", ascending=False)


def plot_profile_comparison(
    summary: pd.DataFrame,
    category: str,
    output_dir: Path,
    top_n: int,
    file_format: str,
    dpi: int,
) -> list[Path]:
    written: list[Path] = []
    if summary.empty:
        return written
    for (operator, scope, event), subset in summary.groupby(
        ["operator", "scope", "event"], observed=True
    ):
        table = complete_regime_table(subset, category).head(top_n).iloc[::-1]
        if table.empty:
            continue
        y = np.arange(len(table))
        figure, axes = plt.subplots(
            1, 2,
            figsize=(20, max(4.5, 0.42 * len(table) + 2.0)),
        )
        height = 0.36
        axes[0].barh(y - height / 2, table["low"], height, color=COLORS["low"], label="Low")
        axes[0].barh(y + height / 2, table["high"], height, color=COLORS["high"], label="High")
        axes[0].set_yticks(y, table.index)
        axes[0].set_xlabel("Self overhead (%)")
        axes[0].set_title("Low vs high entropy")
        axes[0].legend(frameon=False)

        differences = table["high_minus_low"]
        diff_colors = np.where(differences >= 0, COLORS["high"], COLORS["low"])
        axes[1].barh(y, differences, color=diff_colors)
        axes[1].axvline(0, color="#24292F", linewidth=0.8)
        axes[1].set_yticks(y, table.index)
        axes[1].set_xlabel("High - low self overhead (percentage points)")
        axes[1].set_title("Profile-share difference")
        for axis in axes:
            axis.grid(axis="x", alpha=0.25)
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(labelsize=8)
        figure.suptitle(f"{operator} | {scope} | {event} | {category}")
        figure.subplots_adjust(left=0.24, right=0.98, bottom=0.10, top=0.88, wspace=0.62)
        path = output_dir / (
            f"{category}_{safe_name(operator)}_{safe_name(scope)}_"
            f"{safe_name(event)}.{file_format}"
        )
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        written.append(path)
    return written


def load_runtime_rows(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    unique = manifest.drop_duplicates(["json_path", "scope", "pass_id"])
    for _, meta in unique.iterrows():
        path = Path(meta["json_path"])
        if not path.is_file():
            continue
        result = json.loads(path.read_text())
        entropy = result.get("entropy_mean", {})
        config = result.get("config", {})
        rows.append({
            "run_id": meta["run_id"],
            "operator": meta["operator"],
            "regime": meta["regime"],
            "scope": meta["scope"],
            "pass_id": meta["pass_id"],
            "trial_id": meta["trial_id"],
            "elapsed_seconds": result.get("elapsed_seconds", np.nan),
            "nanoseconds_per_call": result.get("nanoseconds_per_call", np.nan),
            "repeats": config.get("repeats", np.nan),
            "input_nchw_entropy_bits": entropy.get(
                "input_nchw_memory_conditional_entropy_bits", np.nan
            ),
            "conv_exact_patch_entropy_bits": entropy.get(
                "conv_exact_patch_entropy_bits", np.nan
            ),
        })
    return pd.DataFrame.from_records(rows)


def add_event_normalizations(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()

    enriched = events.copy()
    json_cache: dict[Path, tuple[float, float]] = {}
    elapsed_values: list[float] = []
    repeat_values: list[float] = []
    for path_text in enriched["json_path"]:
        path = Path(path_text)
        if path not in json_cache:
            result = json.loads(path.read_text())
            config = result.get("config", {})
            json_cache[path] = (
                float(result.get("elapsed_seconds", np.nan)),
                float(config.get("repeats", np.nan)),
            )
        elapsed, repeats = json_cache[path]
        elapsed_values.append(elapsed)
        repeat_values.append(repeats)

    enriched["elapsed_seconds"] = elapsed_values
    enriched["repeats"] = repeat_values
    enriched["event_count_approx"] = pd.to_numeric(
        enriched["event_count_approx"], errors="coerce"
    )
    enriched["event_count_per_call"] = (
        enriched["event_count_approx"] / enriched["repeats"].replace(0, np.nan)
    )
    enriched["event_count_per_second"] = (
        enriched["event_count_approx"]
        / enriched["elapsed_seconds"].replace(0, np.nan)
    )

    group_columns = [
        "operator", "regime", "trial_id", "scope", "pass_id",
    ]
    instruction_counts = (
        enriched[enriched["event"] == "instructions"]
        .groupby(group_columns, observed=True)["event_count_approx"]
        .mean()
        .rename("instruction_count")
        .reset_index()
    )
    enriched = enriched.merge(instruction_counts, on=group_columns, how="left")
    enriched["event_count_per_instruction"] = (
        enriched["event_count_approx"]
        / enriched["instruction_count"].replace(0, np.nan)
    )
    return enriched


def summarize_event_counts(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    numeric = events.copy()
    value_columns = (
        "event_count_approx",
        "event_count_per_call",
        "event_count_per_second",
        "event_count_per_instruction",
    )
    for column in ("sample_count", *value_columns):
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    grouped = (
        numeric.groupby(["operator", "scope", "event", "regime"], observed=True)
        .agg(
            mean_event_count=("event_count_approx", "mean"),
            std_event_count=("event_count_approx", "std"),
            mean_event_count_per_call=("event_count_per_call", "mean"),
            std_event_count_per_call=("event_count_per_call", "std"),
            mean_event_count_per_second=("event_count_per_second", "mean"),
            std_event_count_per_second=("event_count_per_second", "std"),
            mean_event_count_per_instruction=("event_count_per_instruction", "mean"),
            std_event_count_per_instruction=("event_count_per_instruction", "std"),
            mean_sample_count=("sample_count", "mean"),
            trials=("trial_id", "nunique"),
        )
        .reset_index()
    )
    records: list[dict[str, Any]] = []
    for (operator, scope, event), subset in grouped.groupby(
        ["operator", "scope", "event"], observed=True
    ):
        values: dict[str, Any] = {
            "operator": operator, "scope": scope, "event": event,
        }
        for regime in REGIMES:
            row = subset[subset["regime"] == regime]
            for metric in (
                "event_count",
                "event_count_per_call",
                "event_count_per_second",
                "event_count_per_instruction",
            ):
                values[f"{regime}_{metric}"] = (
                    float(row[f"mean_{metric}"].iloc[0])
                    if not row.empty else np.nan
                )
                values[f"{regime}_{metric}_std"] = (
                    float(row[f"std_{metric}"].iloc[0])
                    if not row.empty else np.nan
                )
            values[f"{regime}_sample_count"] = (
                float(row["mean_sample_count"].iloc[0]) if not row.empty else 0.0
            )
            values[f"{regime}_trials"] = (
                int(row["trials"].iloc[0]) if not row.empty else 0
            )
        low = values["low_event_count"]
        high = values["high_event_count"]
        values["high_minus_low"] = high - low
        values["high_minus_low_percent"] = (
            100.0 * (high - low) / low if np.isfinite(low) and low != 0 else np.nan
        )
        records.append(values)
    return pd.DataFrame.from_records(records)


def plot_event_counts(
    summary: pd.DataFrame,
    output_dir: Path,
    file_format: str,
    dpi: int,
    metric: str = "event_count",
    ylabel: str = "Approximate event count",
    filename_suffix: str = "",
) -> list[Path]:
    written: list[Path] = []
    if summary.empty:
        return written
    for (operator, scope), subset in summary.groupby(["operator", "scope"], observed=True):
        subset = subset.sort_values("event").reset_index(drop=True)
        columns = 3
        rows = int(np.ceil(len(subset) / columns))
        figure, axes = plt.subplots(
            rows, columns, figsize=(4.2 * columns, 3.4 * rows),
            squeeze=False, constrained_layout=True,
        )
        for axis, (_, row) in zip(axes.flat, subset.iterrows()):
            values = [row[f"low_{metric}"], row[f"high_{metric}"]]
            errors = [row[f"low_{metric}_std"], row[f"high_{metric}_std"]]
            errors = [0.0 if not np.isfinite(value) else value for value in errors]
            axis.bar(REGIMES, values, yerr=errors, capsize=3,
                     color=[COLORS[item] for item in REGIMES])
            axis.set_title(str(row["event"]), fontsize=9)
            axis.set_ylabel(ylabel)
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
            axis.grid(axis="y", alpha=0.25)
            axis.spines[["top", "right"]].set_visible(False)
            low_value, high_value = values
            delta = (
                100.0 * (high_value - low_value) / low_value
                if np.isfinite(low_value) and low_value != 0 else np.nan
            )
            axis.text(
                0.5, 0.97,
                f"high-low: {delta:+.1f}%" if np.isfinite(delta) else "high-low: n/a",
                transform=axis.transAxes, ha="center", va="top", fontsize=8,
            )
        for axis in axes.flat[len(subset):]:
            axis.set_visible(False)
        figure.suptitle(f"Replay {ylabel.lower()} | {operator} | {scope}")
        path = output_dir / (
            f"event_counts{filename_suffix}_{safe_name(operator)}_"
            f"{safe_name(scope)}.{file_format}"
        )
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        written.append(path)
    return written


def build_derived_metrics(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    index = ["operator", "scope", "regime", "trial_id"]
    wide = events.pivot_table(
        index=index,
        columns="event",
        values="event_count_approx",
        aggfunc="mean",
    ).reset_index()
    definitions = {
        "IPC": ("instructions", "cycles"),
        "L1D load miss fraction": ("L1-dcache-load-misses", "L1-dcache-loads"),
        "Cache miss fraction": ("cache-misses", "cache-references"),
    }
    rows: list[dict[str, Any]] = []
    for _, source in wide.iterrows():
        for metric, (numerator, denominator) in definitions.items():
            if numerator not in wide.columns or denominator not in wide.columns:
                continue
            denominator_value = source[denominator]
            value = (
                source[numerator] / denominator_value
                if pd.notna(denominator_value) and denominator_value != 0
                else np.nan
            )
            rows.append({
                **{column: source[column] for column in index},
                "metric": metric,
                "value": value,
            })
    return pd.DataFrame.from_records(rows)


def plot_derived_metrics(
    derived: pd.DataFrame,
    output_dir: Path,
    file_format: str,
    dpi: int,
) -> list[Path]:
    written: list[Path] = []
    if derived.empty:
        return written
    for (operator, scope), subset in derived.groupby(["operator", "scope"], observed=True):
        metrics = list(dict.fromkeys(subset["metric"]))
        figure, axes = plt.subplots(
            1, len(metrics), figsize=(4.4 * len(metrics), 4.0), squeeze=False,
            constrained_layout=True,
        )
        for axis, metric in zip(axes[0], metrics):
            metric_rows = subset[subset["metric"] == metric]
            means = metric_rows.groupby("regime", observed=True)["value"].mean()
            stds = metric_rows.groupby("regime", observed=True)["value"].std()
            values = [means.get(regime, np.nan) for regime in REGIMES]
            errors = [stds.get(regime, 0.0) for regime in REGIMES]
            errors = [0.0 if not np.isfinite(value) else value for value in errors]
            axis.bar(
                REGIMES, values, yerr=errors, capsize=3,
                color=[COLORS[item] for item in REGIMES],
            )
            axis.set_title(metric)
            axis.grid(axis="y", alpha=0.25)
            axis.spines[["top", "right"]].set_visible(False)
        figure.suptitle(f"Derived hardware ratios | {operator} | {scope}")
        path = output_dir / (
            f"derived_metrics_{safe_name(operator)}_{safe_name(scope)}.{file_format}"
        )
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        written.append(path)
    return written


def plot_runtime(runtime: pd.DataFrame, output_path: Path, dpi: int) -> None:
    if runtime.empty:
        return
    aggregate = (
        runtime.groupby(["operator", "scope", "regime"], observed=True)
        ["nanoseconds_per_call"].mean().reset_index()
    )
    groups = list(aggregate.groupby(["operator", "scope"], observed=True))
    figure, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 4), squeeze=False)
    for axis, ((operator, scope), subset) in zip(axes[0], groups):
        values = [
            float(subset.loc[subset["regime"] == regime, "nanoseconds_per_call"].mean())
            for regime in REGIMES
        ]
        axis.bar(REGIMES, values, color=[COLORS[item] for item in REGIMES])
        axis.set_title(f"{operator} | {scope}")
        axis.set_ylabel("Nanoseconds / call")
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Profiled replay runtime")
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    input_dir = resolve_run_directory(args.input_dir)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir else input_dir / "visualization"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = input_dir / "manifest.csv"
    manifest_columns = set(pd.read_csv(manifest_path, nrows=0).columns)
    if "topdown_jsonl" in manifest_columns:
        from topdown_analysis import run_analysis as run_topdown_analysis

        print("Detected Topdown metric manifest; using Topdown visualization.")
        run_topdown_analysis(
            input_dir=input_dir,
            output_dir=output_dir,
            file_format=args.format,
            dpi=args.dpi,
        )
        return

    manifest = read_manifest(manifest_path)
    symbols, events = load_profiles(manifest)
    events = add_event_normalizations(events)
    if symbols.empty or events["event_count_approx"].notna().sum() == 0:
        raise RuntimeError(
            "No perf counter/profile rows were parsed. Check the perf-report format."
        )
    symbol_summary = aggregate_symbols(symbols, "symbol")
    dso_summary = aggregate_symbols(symbols, "dso")
    event_count_summary = summarize_event_counts(events)
    derived_metrics = build_derived_metrics(events)
    runtime = load_runtime_rows(manifest)

    manifest.to_csv(output_dir / "manifest_resolved.csv", index=False)
    events.to_csv(output_dir / "event_profile_summary.csv", index=False)
    event_count_summary.to_csv(output_dir / "event_count_comparison.csv", index=False)
    derived_metrics.to_csv(output_dir / "derived_metrics.csv", index=False)
    symbols.to_csv(output_dir / "symbol_samples.csv", index=False)
    symbol_summary.to_csv(output_dir / "symbol_profile_summary.csv", index=False)
    dso_summary.to_csv(output_dir / "dso_profile_summary.csv", index=False)
    runtime.to_csv(output_dir / "runtime_entropy_summary.csv", index=False)

    written = []
    written.extend(plot_profile_comparison(
        symbol_summary, "symbol", output_dir, args.top_n,
        args.format, args.dpi,
    ))
    written.extend(plot_profile_comparison(
        dso_summary, "dso", output_dir, min(args.top_n, 10),
        args.format, args.dpi,
    ))
    written.extend(plot_event_counts(
        event_count_summary, output_dir, args.format, args.dpi,
    ))
    written.extend(plot_event_counts(
        event_count_summary, output_dir, args.format, args.dpi,
        metric="event_count_per_call",
        ylabel="Approximate event count / call",
        filename_suffix="_per_call",
    ))
    written.extend(plot_event_counts(
        event_count_summary, output_dir, args.format, args.dpi,
        metric="event_count_per_second",
        ylabel="Approximate event count / second",
        filename_suffix="_per_second",
    ))
    written.extend(plot_event_counts(
        event_count_summary, output_dir, args.format, args.dpi,
        metric="event_count_per_instruction",
        ylabel="Approximate event count / instruction",
        filename_suffix="_per_instruction",
    ))
    written.extend(plot_derived_metrics(
        derived_metrics, output_dir, args.format, args.dpi,
    ))
    runtime_path = output_dir / f"runtime.{args.format}"
    plot_runtime(runtime, runtime_path, args.dpi)
    if runtime_path.exists():
        written.append(runtime_path)

    print(f"Loaded {len(manifest)} event profiles from {input_dir}")
    print(f"Parsed {len(symbols)} symbol rows")
    print(f"Output: {output_dir}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
