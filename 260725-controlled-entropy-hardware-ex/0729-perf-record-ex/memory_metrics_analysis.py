"""Analyze controlled operator perf-stat memory metric passes."""

from __future__ import annotations

import argparse
import ast
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


COLORS = {"low": "#2878B5", "high": "#C23B4A"}
PROFILE_KEYS = (
    "operator", "chain", "regime", "pair_id", "trial_id", "seed", "device_id", "scope"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--minimum-running-percent", type=float, default=99.0)
    parser.add_argument("--terminal-warmup-levels", type=int, default=3)
    parser.add_argument("--maximum-cv-percent", type=float, default=5.0)
    parser.add_argument("--maximum-warmup-range-percent", type=float, default=5.0)
    parser.add_argument("--format", choices=("png", "pdf"), default="png")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def numeric(value: Any) -> float:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return np.nan


def normalize_event_name(value: str) -> str:
    name = value.strip().strip("/")
    return re.sub(r"(?::[ukhp]+)$", "", name, flags=re.IGNORECASE)


def load_json_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid perf JSON at {path}:{line_number}: {exc}") from exc
    return rows


def read_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing manifest: {path}")
    manifest = pd.read_csv(path)
    if "chain" not in manifest.columns:
        manifest["chain"] = "legacy_operator_only"
    required = {
        "run_kind", "pair_id", "operator", "regime", "scope",
        "source_group", "pass_name", "pass_mode", "warmup", "repeats",
        "json_path", "perf_jsonl",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing columns: {', '.join(missing)}")
    return manifest


def read_metric_plan(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    targets = pd.read_csv(input_dir / "metric_targets.csv")
    formulas = pd.read_csv(input_dir / "metric_formulas.csv")
    passes = pd.read_csv(input_dir / "pass_plan.csv")
    pass_events = {
        str(row["pass_name"]): str(row["events"]).split("|")
        for _, row in passes.iterrows()
    }
    return targets, formulas, pass_events


def load_results(
    manifest: pd.DataFrame,
    pass_events: dict[str, list[str]],
    minimum_running_percent: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for _, meta in manifest.iterrows():
        result_path = Path(meta["json_path"])
        perf_path = Path(meta["perf_jsonl"])
        if not result_path.is_file() or not perf_path.is_file():
            raise FileNotFoundError(f"Missing result: {result_path} or {perf_path}")
        result = json.loads(result_path.read_text())
        entries = load_json_lines(perf_path)
        counts: dict[str, float] = {}
        running_values: list[float] = []
        not_counted = 0
        expected_events = pass_events.get(str(meta["pass_name"]))
        if expected_events is None and meta["run_kind"] == "warmup_calibration":
            expected_events = ["cycles", "instructions"]
        perf_entries = [entry for entry in entries if str(entry.get("event", "")).strip()]
        if expected_events is not None and len(perf_entries) != len(expected_events):
            raise ValueError(
                f"{perf_path} has {len(perf_entries)} perf events; "
                f"the pass plan expects {len(expected_events)}"
            )
        for event_index, entry in enumerate(perf_entries):
            event = normalize_event_name(str(entry.get("event", "")))
            requested_event = (
                expected_events[event_index] if expected_events is not None else event
            )
            count = numeric(entry.get("counter-value"))
            running = numeric(entry.get("pcnt-running"))
            if event:
                counts[requested_event] = count
                if np.isfinite(running):
                    running_values.append(running)
                if not np.isfinite(count):
                    not_counted += 1
                event_rows.append({
                    **meta.to_dict(),
                    "event": event,
                    "requested_event": requested_event,
                    "counter_value": count,
                    "counter_per_call": count / numeric(meta["repeats"]),
                    "running_percent": running,
                })

        minimum_running = min(running_values) if running_values else np.nan
        high_confidence = bool(
            not_counted == 0
            and np.isfinite(minimum_running)
            and minimum_running >= minimum_running_percent
        )
        entropy = result.get("entropy_mean", {})
        repeats = numeric(meta["repeats"])
        row: dict[str, Any] = {
            **meta.to_dict(),
            "elapsed_seconds": numeric(result.get("elapsed_seconds")),
            "nanoseconds_per_call": numeric(result.get("nanoseconds_per_call")),
            "conv_patch_stream_conditional_entropy_bits": numeric(
                entropy.get("conv_patch_stream_conditional_entropy_bits")
            ),
            "minimum_running_percent": minimum_running,
            "not_counted_events": not_counted,
            "high_confidence": high_confidence,
        }
        for event, value in counts.items():
            key = re.sub(r"[^a-z0-9]+", "_", event.lower()).strip("_")
            row[key] = value
            row[f"{key}_per_call"] = value / repeats if repeats else np.nan

        cycles = counts.get("cycles", np.nan)
        instructions = counts.get("instructions", np.nan)
        row["ipc"] = instructions / cycles if cycles else np.nan
        run_rows.append(row)
    return pd.DataFrame(run_rows), pd.DataFrame(event_rows)


class FormulaEvaluator:
    """Evaluate perf metric expressions from raw replay-normalized counters."""

    def __init__(self, formulas: dict[str, str], values: dict[str, float]):
        self.formulas = formulas
        self.values = values
        self.cache: dict[str, tuple[float, set[str]]] = {}
        self.symbols = sorted(
            set(formulas) | set(values), key=len, reverse=True
        )

    def resolve(self, name: str) -> tuple[float, set[str]]:
        if name in self.cache:
            return self.cache[name]
        if name in self.values:
            result = (self.values[name], {name})
            self.cache[name] = result
            return result
        expression = self.formulas.get(name)
        if expression is None:
            return np.nan, {name}

        rendered = expression
        dependencies: set[str] = set()
        replacements: dict[str, float] = {}
        replacement_index = 0
        for symbol in self.symbols:
            if symbol not in rendered:
                continue
            if symbol in self.formulas:
                value, raw_dependencies = self.resolve(symbol)
            else:
                value, raw_dependencies = self.values[symbol], {symbol}
            placeholder = f"value_{replacement_index}"
            replacement_index += 1
            rendered = rendered.replace(symbol, placeholder)
            replacements[placeholder] = value
            dependencies.update(raw_dependencies)

        try:
            tree = ast.parse(rendered, mode="eval")
            allowed_nodes = (
                ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub,
                ast.Mult, ast.Div, ast.USub, ast.UAdd, ast.Call, ast.Name,
                ast.Load, ast.Constant,
            )
            for node in ast.walk(tree):
                if not isinstance(node, allowed_nodes):
                    raise ValueError(f"Unsupported expression node: {type(node).__name__}")
                if isinstance(node, ast.Call) and not (
                    isinstance(node.func, ast.Name) and node.func.id in {"min", "max"}
                ):
                    raise ValueError("Only min() and max() are allowed")
                if isinstance(node, ast.Name) and node.id not in set(replacements) | {"min", "max"}:
                    raise ValueError(f"Unresolved symbol: {node.id}")
            value = float(eval(
                compile(tree, "<perf-metric>", "eval"),
                {"__builtins__": {}, "min": min, "max": max},
                replacements,
            ))
            if not np.isfinite(value):
                value = np.nan
        except (ArithmeticError, TypeError, ValueError, SyntaxError):
            value = np.nan
        result = (value, dependencies)
        self.cache[name] = result
        return result


def reconstruct_metrics(
    runs: pd.DataFrame,
    events: pd.DataFrame,
    targets: pd.DataFrame,
    formula_frame: pd.DataFrame,
    minimum_running_percent: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    formulas = dict(zip(formula_frame["metric"], formula_frame["expression"]))
    main_runs = runs[runs["run_kind"] == "main"].copy()
    main_events = events[events["run_kind"] == "main"].copy()
    timing = (
        main_runs.groupby(list(PROFILE_KEYS), observed=True)["nanoseconds_per_call"]
        .median()
        .reset_index()
    )
    timing_map = {
        tuple(row[key] for key in PROFILE_KEYS): numeric(row["nanoseconds_per_call"])
        for _, row in timing.iterrows()
    }

    raw_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for profile_key, group in main_events.groupby(list(PROFILE_KEYS), observed=True):
        profile = dict(zip(PROFILE_KEYS, profile_key))
        values: dict[str, float] = {}
        running_map: dict[str, float] = {}
        for event, event_group in group.groupby("requested_event", observed=True):
            value = pd.to_numeric(event_group["counter_per_call"], errors="coerce").mean()
            running = pd.to_numeric(event_group["running_percent"], errors="coerce").min()
            values[str(event)] = float(value)
            running_map[str(event)] = float(running)
            raw_rows.append({
                **profile,
                "source_group": "RawEvents",
                "pass_name": "replayed_raw_events",
                "metric": f"raw::{event}::per_call",
                "value": value,
                "minimum_running_percent": running,
                "high_confidence": bool(
                    np.isfinite(value)
                    and np.isfinite(running)
                    and running >= minimum_running_percent
                ),
            })

        nanoseconds = timing_map.get(profile_key, np.nan)
        values["duration_time"] = nanoseconds / 1e9
        if "CPU_CLK_UNHALTED.REF_TSC" in values:
            values["TSC"] = values["CPU_CLK_UNHALTED.REF_TSC"]
            running_map["TSC"] = running_map.get("CPU_CLK_UNHALTED.REF_TSC", np.nan)
        evaluator = FormulaEvaluator(formulas, values)
        target_scope = "system" if profile["scope"] == "system" else "core"
        selected_targets = targets[targets["scope_class"] == target_scope]
        for _, target in selected_targets.iterrows():
            formula_value, dependencies = evaluator.resolve(str(target["metric"]))
            display_scale = numeric(target["display_scale"])
            value = formula_value * display_scale
            event_dependencies = sorted(
                dependency for dependency in dependencies if dependency != "duration_time"
            )
            dependency_running = [
                running_map.get(dependency, np.nan) for dependency in event_dependencies
            ]
            minimum_running = (
                min(dependency_running) if dependency_running else 100.0
            )
            high_confidence = bool(
                np.isfinite(value)
                and all(np.isfinite(item) for item in dependency_running)
                and minimum_running >= minimum_running_percent
            )
            metric_rows.append({
                **profile,
                "source_group": target["source_group"],
                "pass_name": "reconstructed",
                "metric": target["metric"],
                "formula_value": formula_value,
                "display_scale": display_scale,
                "value": value,
                "dependencies": ";".join(event_dependencies),
                "dependency_count": len(event_dependencies),
                "minimum_running_percent": minimum_running,
                "high_confidence": high_confidence,
            })

        if profile["scope"] != "system":
            instructions = values.get("INST_RETIRED.ANY", np.nan)
            cycles = values.get("CPU_CLK_UNHALTED.THREAD", np.nan)
            for metric, value, dependencies in (
                ("nanoseconds_per_call", nanoseconds, []),
                ("instructions_per_cycle", instructions / cycles if cycles else np.nan,
                 ["INST_RETIRED.ANY", "CPU_CLK_UNHALTED.THREAD"]),
            ):
                dependency_running = [running_map.get(item, np.nan) for item in dependencies]
                minimum_running = min(dependency_running) if dependency_running else 100.0
                metric_rows.append({
                    **profile,
                    "source_group": "CoreSummary",
                    "pass_name": "reconstructed",
                    "metric": metric,
                    "formula_value": value,
                    "display_scale": 1.0,
                    "value": value,
                    "dependencies": ";".join(dependencies),
                    "dependency_count": len(dependencies),
                    "minimum_running_percent": minimum_running,
                    "high_confidence": bool(
                        np.isfinite(value)
                        and all(np.isfinite(item) for item in dependency_running)
                        and minimum_running >= minimum_running_percent
                    ),
                })
    return pd.DataFrame(metric_rows), pd.DataFrame(raw_rows)


def cv_percent(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) < 2 or np.isclose(values.mean(), 0.0):
        return np.nan
    return float(100.0 * values.std(ddof=1) / abs(values.mean()))


def long_measurements(metrics: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "operator", "chain", "scope", "source_group", "pass_name", "regime",
        "pair_id", "trial_id", "metric", "value",
        "minimum_running_percent", "high_confidence",
    ]
    frames = [frame[columns] for frame in (metrics, raw) if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def summarize_measurements(
    measurements: pd.DataFrame,
    maximum_cv_percent: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if measurements.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    keys = ["operator", "chain", "scope", "source_group", "pass_name", "regime", "metric"]
    summary = (
        measurements.groupby(keys, observed=True)["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    stability_rows = []
    for group_key, group in measurements.groupby(keys, observed=True):
        cv = cv_percent(group["value"])
        stability_rows.append({
            **dict(zip(keys, group_key)),
            "cv_percent": cv,
            "minimum_running_percent": group["minimum_running_percent"].min(),
            "all_high_confidence": bool(group["high_confidence"].all()),
            "stable": bool(np.isfinite(cv) and cv <= maximum_cv_percent),
        })

    paired_rows = []
    pair_keys = ["operator", "chain", "scope", "source_group", "pass_name", "pair_id", "metric"]
    for group_key, group in measurements.groupby(pair_keys, observed=True):
        low = group[group["regime"] == "low"]["value"]
        high = group[group["regime"] == "high"]["value"]
        if len(low) != 1 or len(high) != 1:
            continue
        low_value, high_value = float(low.iloc[0]), float(high.iloc[0])
        paired_rows.append({
            **dict(zip(pair_keys, group_key)),
            "low": low_value,
            "high": high_value,
            "high_minus_low": high_value - low_value,
            "high_minus_low_percent": (
                100.0 * (high_value - low_value) / abs(low_value)
                if not np.isclose(low_value, 0.0) else np.nan
            ),
        })
    return summary, pd.DataFrame(stability_rows), pd.DataFrame(paired_rows)


def summarize_warmup(
    runs: pd.DataFrame,
    terminal_count: int,
    maximum_range_percent: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    calibration = runs[runs["run_kind"] == "warmup_calibration"].copy()
    if calibration.empty:
        return pd.DataFrame(), pd.DataFrame()
    metrics = ["nanoseconds_per_call", "ipc"]
    summary = (
        calibration.groupby(["operator", "chain", "regime", "warmup"], observed=True)[metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(item) for item in column if item).rstrip("_")
        if isinstance(column, tuple) else str(column)
        for column in summary.columns
    ]
    rows = []
    for (operator, chain, regime), group in summary.groupby(
        ["operator", "chain", "regime"], observed=True
    ):
        terminal = group.sort_values("warmup").tail(terminal_count)
        for metric in metrics:
            values = terminal[f"{metric}_mean"].dropna()
            mean = values.mean()
            relative_range = (
                100.0 * (values.max() - values.min()) / abs(mean)
                if len(values) >= 2 and not np.isclose(mean, 0.0) else np.nan
            )
            rows.append({
                "operator": operator,
                "chain": chain,
                "regime": regime,
                "metric": metric,
                "terminal_levels": ",".join(
                    str(int(value)) for value in terminal["warmup"]
                ),
                "terminal_relative_range_percent": relative_range,
                "stable": bool(
                    np.isfinite(relative_range)
                    and relative_range <= maximum_range_percent
                ),
            })
    return summary, pd.DataFrame(rows)


def safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def plot_measurements(
    measurements: pd.DataFrame,
    output_dir: Path,
    fmt: str,
    dpi: int,
) -> list[Path]:
    written: list[Path] = []
    if measurements.empty:
        return written
    group_keys = ["operator", "chain", "scope", "source_group"]
    for (operator, chain, scope, source_group), group in measurements.groupby(
        group_keys, observed=True
    ):
        metric_names = sorted(group["metric"].unique())
        metrics_per_page = 12
        for page_index, start in enumerate(range(0, len(metric_names), metrics_per_page)):
            page_metrics = metric_names[start : start + metrics_per_page]
            ncols = 3
            nrows = int(np.ceil(len(page_metrics) / ncols))
            figure, axes = plt.subplots(
                nrows, ncols, figsize=(4.3 * ncols, 3.3 * nrows), squeeze=False,
                constrained_layout=True,
            )
            for ax, metric in zip(axes.flat, page_metrics):
                data = group[group["metric"] == metric]
                for _, pair in data.groupby(["pass_name", "pair_id"], observed=True):
                    values = []
                    for regime in ("low", "high"):
                        selected = pair[pair["regime"] == regime]["value"]
                        values.append(
                            float(selected.iloc[0]) if len(selected) == 1 else np.nan
                        )
                    ax.plot((0, 1), values, color="#999999", alpha=0.5, linewidth=1)
                    ax.scatter((0, 1), values, c=(COLORS["low"], COLORS["high"]), s=24)
                ax.set_xticks((0, 1), ("low", "high"))
                ax.set_title(metric, fontsize=9)
                ax.grid(axis="y", alpha=0.25)
            for ax in axes.flat[len(page_metrics):]:
                ax.set_visible(False)
            figure.suptitle(f"{operator}/{chain}: {source_group} ({scope})")
            page_suffix = f"_page_{page_index + 1:02d}" if len(metric_names) > metrics_per_page else ""
            path = output_dir / (
                f"metrics_{safe_name(operator)}_{safe_name(chain)}_{safe_name(scope)}_"
                f"{safe_name(source_group)}{page_suffix}.{fmt}"
            )
            figure.savefig(path, dpi=dpi)
            plt.close(figure)
            written.append(path)
    return written


def plot_warmup(summary: pd.DataFrame, output_dir: Path, fmt: str, dpi: int) -> list[Path]:
    written: list[Path] = []
    if summary.empty:
        return written
    for (operator, chain), group in summary.groupby(
        ["operator", "chain"], observed=True
    ):
        figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
        for ax, metric in zip(axes, ("nanoseconds_per_call", "ipc")):
            for regime in ("low", "high"):
                data = group[group["regime"] == regime].sort_values("warmup")
                ax.errorbar(
                    data["warmup"], data[f"{metric}_mean"],
                    yerr=data[f"{metric}_std"].fillna(0.0), marker="o",
                    capsize=3, color=COLORS[regime], label=regime,
                )
            ax.set_title(metric)
            ax.set_xlabel("Warm-up operator calls")
            ax.grid(alpha=0.25)
        axes[0].legend(frameon=False)
        figure.suptitle(f"Warm-up convergence: {operator}/{chain}")
        path = output_dir / f"warmup_{safe_name(operator)}_{safe_name(chain)}.{fmt}"
        figure.savefig(path, dpi=dpi)
        plt.close(figure)
        written.append(path)
    return written


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else input_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(input_dir / "manifest.csv")
    targets, formulas, pass_events = read_metric_plan(input_dir)
    runs, events = load_results(
        manifest, pass_events, args.minimum_running_percent
    )
    perf_metrics, raw_measurements = reconstruct_metrics(
        runs, events, targets, formulas, args.minimum_running_percent
    )
    measurements = long_measurements(perf_metrics, raw_measurements)
    summary, replicate_stability, paired = summarize_measurements(
        measurements, args.maximum_cv_percent
    )
    warmup_summary, warmup_stability = summarize_warmup(
        runs, args.terminal_warmup_levels, args.maximum_warmup_range_percent
    )

    outputs = {
        "runs.csv": runs,
        "raw_events.csv": events,
        "reconstructed_metrics.csv": perf_metrics,
        "raw_event_measurements.csv": raw_measurements,
        "measurements.csv": measurements,
        "summary.csv": summary,
        "replicate_stability.csv": replicate_stability,
        "paired_differences.csv": paired,
        "warmup_convergence.csv": warmup_summary,
        "warmup_stability.csv": warmup_stability,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, index=False)
    written = [
        *plot_measurements(measurements, output_dir, args.format, args.dpi),
        *plot_warmup(warmup_summary, output_dir, args.format, args.dpi),
    ]

    low_running = runs[~runs["high_confidence"]]
    print(f"Loaded {len(runs)} perf-stat passes")
    print(f"Passes below {args.minimum_running_percent:.1f}% running: {len(low_running)}")
    if not low_running.empty:
        print(low_running[[
            "operator", "chain", "regime", "scope", "source_group", "pass_name",
            "minimum_running_percent", "not_counted_events",
        ]].to_string(index=False))
    if not warmup_stability.empty:
        passed = int(warmup_stability["stable"].sum())
        print(f"Warm-up stability passed: {passed}/{len(warmup_stability)}")
        print(warmup_stability.to_string(index=False))
    if not replicate_stability.empty:
        passed = int(replicate_stability["stable"].sum())
        print(f"Replicate stability passed: {passed}/{len(replicate_stability)}")
    print(f"Analysis output: {output_dir}")
    for path in written:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
