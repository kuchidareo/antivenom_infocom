from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


PERF_EVENT_PASSES: Mapping[str, Mapping[str, Sequence[str]]] = {
    "core": {
        "events": ("cycles", "instructions", "task-clock"),
        "event_options": ("{cycles,instructions}", "task-clock"),
    },
    "l1": {
        "events": (
            "instructions",
            "l1d_cache_rd",
            "l1d_cache_refill_rd",
            "l1d_cache_wr",
            "l1d_cache_refill_wr",
            "task-clock",
        ),
        "event_options": (
            "{instructions,l1d_cache_rd,l1d_cache_refill_rd,l1d_cache_wr,l1d_cache_refill_wr}",
            "task-clock",
        ),
    },
    "l2": {
        "events": (
            "instructions",
            "l2d_cache_rd",
            "l2d_cache_refill_rd",
            "l2d_cache_wr",
            "l2d_cache_refill_wr",
            "task-clock",
        ),
        "event_options": (
            "{instructions,l2d_cache_rd,l2d_cache_refill_rd,l2d_cache_wr,l2d_cache_refill_wr}",
            "task-clock",
        ),
    },
}

REGIMES = (
    "large-stable",
    "large-unstable",
    "small-stable",
    "small-unstable",
)

# Balanced four-condition order: each regime appears once in every run-order
# position over four replicates, while immediate carryover is counterbalanced.
COUNTERBALANCED_ORDERS = (
    ("large-stable", "large-unstable", "small-unstable", "small-stable"),
    ("large-unstable", "small-stable", "large-stable", "small-unstable"),
    ("small-stable", "small-unstable", "large-unstable", "large-stable"),
    ("small-unstable", "large-stable", "small-stable", "large-unstable"),
)

CONTRASTS = {
    "stability_at_large": ("large-stable", "large-unstable"),
    "stability_at_small": ("small-stable", "small-unstable"),
    "magnitude_when_stable": ("small-stable", "large-stable"),
    "magnitude_when_unstable": ("small-unstable", "large-unstable"),
    "original_combined": ("small-stable", "large-unstable"),
}
GRADIENT_LAYOUT_FIELDS = (
    "gradient_bank_shape",
    "gradient_bank_stride",
    "gradient_bank_contiguous",
    "gradient_bank_buffer_count",
    "gradient_bank_unique_buffer_addresses",
    "gradient_buffer_bytes",
    "gradient_buffer_offsets_bytes",
    "gradient_bank_base_ptr_mod64",
    "gradient_bank_base_ptr_mod4096",
)
GRADIENT_IDENTITY_FIELDS = (
    *GRADIENT_LAYOUT_FIELDS,
    "gradient_mean",
    "gradient_std",
    "gradient_l2_mean",
    "gradient_l2_min",
    "gradient_l2_max",
    "adjacent_gradient_cosine",
)

EVENT_COLUMNS = [
    "replicate",
    "run_order",
    "seed",
    "regime",
    "gradient_magnitude",
    "gradient_direction",
    "gradient_scale",
    "event_pass",
    "event",
    "total_count",
    "unit",
    "running_time_ns",
    "running_percentage",
    "running_status",
    "steps",
    "count_per_backward",
    "scaled_by_perf",
    "raw_perf_path",
]

REPLICATE_COLUMNS = [
    "replicate",
    "run_order",
    "seed",
    "regime",
    "gradient_magnitude",
    "gradient_direction",
    "gradient_scale",
    "steps",
    "gradient_mean",
    "gradient_std",
    "gradient_l2_mean",
    "gradient_l2_min",
    "gradient_l2_max",
    "adjacent_gradient_cosine",
    "gradient_bank_shape",
    "gradient_bank_stride",
    "gradient_bank_buffer_count",
    "gradient_bank_unique_buffer_addresses",
    "gradient_buffer_bytes",
    "gradient_buffer_offsets_bytes",
    "gradient_bank_base_ptr_mod64",
    "gradient_bank_base_ptr_mod4096",
    "backward_mean_ms",
    "backward_median_ms",
    "backward_p95_ms",
    "backward_total_ms",
    "cycles_per_backward",
    "instructions_per_backward",
    "instructions_per_cycle",
    "task_clock_per_backward_ms",
    "l1_instructions_per_backward",
    "l1_access_total",
    "l1_refill_total",
    "l1_refill_fraction",
    "l2_instructions_per_backward",
    "l2_access_total",
    "l2_refill_total",
    "l2_refill_fraction",
    "minimum_running_percentage",
    "all_events_valid",
    "all_events_ideal",
]

PAIRED_METRICS = (
    "backward_median_ms",
    "instructions_per_backward",
    "l1_refill_fraction",
    "l2_refill_fraction",
    "task_clock_per_backward_ms",
)


def parse_perf_number(value: str) -> Optional[float]:
    stripped = value.strip().replace(",", "")
    if not stripped or stripped.startswith("<not"):
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def normalize_event_name(event: str) -> str:
    return event.strip().split(":", 1)[0]


def parse_perf_stat_summary(path: Path, expected_events: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    """Parse non-interval `perf stat -x ';'` output.

    The fields used here are count, unit, event, time-running in nanoseconds,
    and percentage-running. These are the documented perf CSV summary fields.
    """
    expected = set(expected_events)
    parsed: Dict[str, Dict[str, Any]] = {}

    with path.open(errors="replace") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [part.strip() for part in line.split(";")]
            if len(parts) < 3:
                continue
            event = normalize_event_name(parts[2])
            if event not in expected:
                continue
            parsed[event] = {
                "count": parse_perf_number(parts[0]),
                "unit": parts[1],
                "running_time_ns": parse_perf_number(parts[3]) if len(parts) > 3 else None,
                "running_percentage": parse_perf_number(parts[4]) if len(parts) > 4 else None,
            }

    missing = expected - parsed.keys()
    if missing:
        raise RuntimeError(f"Missing perf events in {path}: {sorted(missing)}")
    return parsed


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def benchmark_arguments(
    args: argparse.Namespace,
    regime: str,
    seed: int,
    timing_path: Path,
    summary_path: Path,
) -> List[str]:
    return [
        "--regime", regime,
        "--steps", str(args.steps),
        "--warmup", str(args.warmup),
        "--threads", str(args.threads),
        "--seed", str(seed),
        "--batch-size", str(args.batch_size),
        "--channels", str(args.channels),
        "--spatial-size", str(args.spatial_size),
        "--gradient-bank-size", str(args.gradient_bank_size),
        "--large-scale", str(args.large_scale),
        "--small-scale", str(args.small_scale),
        "--output", str(timing_path),
        "--summary-output", str(summary_path),
    ]


def running_status(
    count: Optional[float],
    percentage: Optional[float],
    *,
    warning_threshold: float,
    invalid_threshold: float,
) -> str:
    if count is None or percentage is None or percentage < invalid_threshold:
        return "invalid"
    if percentage < warning_threshold:
        return "warning"
    return "ideal"


def run_perf_pass(
    *,
    args: argparse.Namespace,
    replicate: int,
    run_order: int,
    seed: int,
    regime: str,
    pass_name: str,
    pass_config: Mapping[str, Sequence[str]],
    output_dir: Path,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    events = tuple(pass_config["events"])
    event_options = tuple(pass_config["event_options"])
    run_dir = output_dir / "raw" / f"replicate_{replicate:02d}" / f"order_{run_order}_{regime}" / pass_name
    run_dir.mkdir(parents=True, exist_ok=True)
    timing_path = run_dir / "backward_times.csv"
    benchmark_summary_path = run_dir / "benchmark_summary.json"
    raw_perf_path = run_dir / "perf_stat.csv"

    with tempfile.TemporaryDirectory(prefix="controlled-perf-") as temporary_dir:
        control_fifo = Path(temporary_dir) / "control.fifo"
        ack_fifo = Path(temporary_dir) / "ack.fifo"
        os.mkfifo(control_fifo)
        os.mkfifo(ack_fifo)
        control_holder = os.open(control_fifo, os.O_RDWR | os.O_NONBLOCK)
        ack_holder = os.open(ack_fifo, os.O_RDWR | os.O_NONBLOCK)
        try:
            workload = [
                args.python,
                str(args.benchmark),
                *benchmark_arguments(args, regime, seed, timing_path, benchmark_summary_path),
                "--perf-control-fifo", str(control_fifo),
                "--perf-ack-fifo", str(ack_fifo),
            ]
            command = [
                args.perf_binary,
                "stat",
                "--delay=-1",
                f"--control=fifo:{control_fifo},{ack_fifo}",
                "-x", ";",
                "-o", str(raw_perf_path),
            ]
            if not args.scale_counts:
                command.append("--no-scale")
            for event_option in event_options:
                command.extend(("-e", event_option))
            command.extend(("--", *workload))
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        finally:
            os.close(control_holder)
            os.close(ack_holder)

    (run_dir / "benchmark_stdout.txt").write_text(completed.stdout)
    if completed.stderr:
        (run_dir / "perf_stderr.txt").write_text(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"perf failed: replicate={replicate} regime={regime} pass={pass_name} "
            f"exit={completed.returncode}: {completed.stderr.strip()}"
        )

    with benchmark_summary_path.open() as file:
        benchmark_summary = json.load(file)
    perf_results = parse_perf_stat_summary(raw_perf_path, events)

    event_rows: List[Dict[str, Any]] = []
    for event in events:
        result = perf_results[event]
        count = result["count"]
        percentage = result["running_percentage"]
        status = running_status(
            count,
            percentage,
            warning_threshold=args.running_warning_percentage,
            invalid_threshold=args.running_invalid_percentage,
        )
        event_rows.append(
            {
                "replicate": replicate,
                "run_order": run_order,
                "seed": seed,
                "regime": regime,
                "gradient_magnitude": benchmark_summary["gradient_magnitude"],
                "gradient_direction": benchmark_summary["gradient_direction"],
                "gradient_scale": benchmark_summary["gradient_scale"],
                "event_pass": pass_name,
                "event": event,
                "total_count": "" if count is None else count,
                "unit": result["unit"],
                "running_time_ns": "" if result["running_time_ns"] is None else result["running_time_ns"],
                "running_percentage": "" if percentage is None else percentage,
                "running_status": status,
                "steps": args.steps,
                "count_per_backward": "" if count is None else count / args.steps,
                "scaled_by_perf": args.scale_counts,
                "raw_perf_path": str(raw_perf_path),
            }
        )
    return benchmark_summary, event_rows


def validate_pass_identity(summaries: Sequence[Mapping[str, Any]]) -> None:
    reference = summaries[0]
    for summary in summaries[1:]:
        for field in GRADIENT_IDENTITY_FIELDS:
            if summary[field] != reference[field]:
                raise RuntimeError(f"Controlled workload changed between perf passes: {field}")


def validate_paired_layout(stable: Mapping[str, Any], unstable: Mapping[str, Any]) -> None:
    for field in GRADIENT_LAYOUT_FIELDS:
        if stable[field] != unstable[field]:
            raise RuntimeError(f"Gradient bank layout differs between paired regimes: {field}")


def count_for(rows: Mapping[tuple[str, str], Mapping[str, Any]], event_pass: str, event: str) -> float:
    value = rows[(event_pass, event)]["total_count"]
    if value == "":
        return float("nan")
    return float(value)


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


def build_replicate_summary(
    *,
    replicate: int,
    run_order: int,
    seed: int,
    regime: str,
    core_benchmark_summary: Mapping[str, Any],
    event_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    rows = {(str(row["event_pass"]), str(row["event"])): row for row in event_rows}
    steps = int(core_benchmark_summary["steps"])
    cycles = count_for(rows, "core", "cycles")
    instructions = count_for(rows, "core", "instructions")
    task_clock = count_for(rows, "core", "task-clock")
    l1_instructions = count_for(rows, "l1", "instructions")
    l1_access = count_for(rows, "l1", "l1d_cache_rd") + count_for(rows, "l1", "l1d_cache_wr")
    l1_refill = count_for(rows, "l1", "l1d_cache_refill_rd") + count_for(rows, "l1", "l1d_cache_refill_wr")
    l2_instructions = count_for(rows, "l2", "instructions")
    l2_access = count_for(rows, "l2", "l2d_cache_rd") + count_for(rows, "l2", "l2d_cache_wr")
    l2_refill = count_for(rows, "l2", "l2d_cache_refill_rd") + count_for(rows, "l2", "l2d_cache_refill_wr")
    percentages = [float(row["running_percentage"]) for row in event_rows if row["running_percentage"] != ""]

    return {
        "replicate": replicate,
        "run_order": run_order,
        "seed": seed,
        "regime": regime,
        **core_benchmark_summary,
        "cycles_per_backward": cycles / steps,
        "instructions_per_backward": instructions / steps,
        "instructions_per_cycle": ratio(instructions, cycles),
        "task_clock_per_backward_ms": task_clock / steps,
        "l1_instructions_per_backward": l1_instructions / steps,
        "l1_access_total": l1_access,
        "l1_refill_total": l1_refill,
        "l1_refill_fraction": ratio(l1_refill, l1_access),
        "l2_instructions_per_backward": l2_instructions / steps,
        "l2_access_total": l2_access,
        "l2_refill_total": l2_refill,
        "l2_refill_fraction": ratio(l2_refill, l2_access),
        "minimum_running_percentage": min(percentages) if percentages else float("nan"),
        "all_events_valid": all(row["running_status"] != "invalid" for row in event_rows),
        "all_events_ideal": all(row["running_status"] == "ideal" for row in event_rows),
    }


def build_contrast_rows(
    replicate: int,
    summaries: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for contrast, (left_regime, right_regime) in CONTRASTS.items():
        left = summaries[left_regime]
        right = summaries[right_regime]
        for metric in PAIRED_METRICS:
            left_value = float(left[metric])
            right_value = float(right[metric])
            difference = left_value - right_value
            rows.append(
                {
                    "replicate": replicate,
                    "seed": left["seed"],
                    "contrast": contrast,
                    "left_regime": left_regime,
                    "right_regime": right_regime,
                    "left_run_order": left["run_order"],
                    "right_run_order": right["run_order"],
                    "metric": metric,
                    "left_value": left_value,
                    "right_value": right_value,
                    "difference_left_minus_right": difference,
                    "percent_change_vs_right": (
                        100.0 * difference / right_value if right_value else float("nan")
                    ),
                    "both_events_valid": bool(
                        left["all_events_valid"] and right["all_events_valid"]
                    ),
                }
            )
    return rows


def build_hypothesis_check(
    replicate: int,
    summaries: Mapping[str, Mapping[str, Any]],
    instructions_tolerance_percent: float,
) -> Dict[str, Any]:
    stable = summaries["small-stable"]
    unstable = summaries["large-unstable"]

    def percent_change(metric: str) -> float:
        left = float(stable[metric])
        right = float(unstable[metric])
        return 100.0 * (left - right) / right if right else float("nan")

    row = {
        "replicate": replicate,
        "seed": stable["seed"],
        "contrast": "original_combined",
        "left_regime": "small-stable",
        "right_regime": "large-unstable",
        "check_backward_time_lower": (
            float(stable["backward_median_ms"]) < float(unstable["backward_median_ms"])
        ),
        "check_instructions_similar": (
            abs(percent_change("instructions_per_backward")) <= instructions_tolerance_percent
        ),
        "check_l1_refill_fraction_lower": (
            float(stable["l1_refill_fraction"]) < float(unstable["l1_refill_fraction"])
        ),
        "check_l2_refill_fraction_lower": (
            float(stable["l2_refill_fraction"]) < float(unstable["l2_refill_fraction"])
        ),
        "check_task_clock_lower": (
            float(stable["task_clock_per_backward_ms"])
            < float(unstable["task_clock_per_backward_ms"])
        ),
        "all_events_valid": bool(
            stable["all_events_valid"] and unstable["all_events_valid"]
        ),
    }
    row["supports_hypothesis"] = all(
        value for key, value in row.items() if key.startswith("check_") or key == "all_events_valid"
    )
    return row


def aggregate_metric(values: Sequence[float]) -> Dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def aggregate_results(
    replicate_rows: Sequence[Mapping[str, Any]],
    contrast_rows: Sequence[Mapping[str, Any]],
    hypothesis_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"regimes": {}, "contrasts": {}}
    for regime in REGIMES:
        selected = [row for row in replicate_rows if row["regime"] == regime]
        result["regimes"][regime] = {
            metric: aggregate_metric([float(row[metric]) for row in selected])
            for metric in PAIRED_METRICS
        }
    for contrast in CONTRASTS:
        result["contrasts"][contrast] = {}
        for metric in PAIRED_METRICS:
            selected = [
                row for row in contrast_rows
                if row["contrast"] == contrast and row["metric"] == metric
            ]
            result["contrasts"][contrast][metric] = aggregate_metric(
                [float(row["difference_left_minus_right"]) for row in selected]
            )
    result["replicates"] = len(hypothesis_rows)
    result["supporting_replicates"] = sum(
        bool(row["supports_hypothesis"]) for row in hypothesis_rows
    )
    result["all_replicates_support_hypothesis"] = all(
        bool(row["supports_hypothesis"]) for row in hypothesis_rows
    )
    result["interpretation"] = (
        "A supporting result indicates that small, directionally stable gradients are associated "
        "with higher Conv2D backward throughput and a lower cache-refill fraction under an "
        "otherwise fixed execution graph. It does not directly establish prefetch accuracy."
    )
    return result


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run paired, counterbalanced controlled backward experiments with grouped perf events."
    )
    parser.add_argument("--output-dir", default=str(script_dir / "perf_results"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--perf-binary", default="perf")
    parser.add_argument("--benchmark", type=Path, default=script_dir / "controlled_cache_test.py")
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--spatial-size", type=int, default=32)
    parser.add_argument("--gradient-bank-size", type=int, default=16)
    parser.add_argument("--large-scale", type=float, default=1e-1)
    parser.add_argument("--small-scale", type=float, default=1e-4)
    parser.add_argument("--running-warning-percentage", type=float, default=99.0)
    parser.add_argument("--running-invalid-percentage", type=float, default=95.0)
    parser.add_argument("--instructions-tolerance-percent", type=float, default=5.0)
    parser.add_argument(
        "--scale-counts",
        action="store_true",
        help="Allow perf to scale multiplexed counts. The default uses --no-scale.",
    )
    args = parser.parse_args()

    if shutil.which(args.perf_binary) is None:
        parser.error(f"perf executable was not found: {args.perf_binary}")
    if args.replicates <= 0 or args.steps <= 0:
        parser.error("--replicates and --steps must be positive")
    if not 0 <= args.running_invalid_percentage <= args.running_warning_percentage <= 100:
        parser.error("Require 0 <= invalid running percentage <= warning percentage <= 100")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_event_rows: List[Dict[str, Any]] = []
    replicate_rows: List[Dict[str, Any]] = []
    contrast_rows: List[Dict[str, Any]] = []
    hypothesis_rows: List[Dict[str, Any]] = []

    for replicate in range(1, args.replicates + 1):
        paired_seed = args.seed + replicate - 1
        regime_order = COUNTERBALANCED_ORDERS[(replicate - 1) % len(COUNTERBALANCED_ORDERS)]
        current_pair: Dict[str, Dict[str, Any]] = {}

        for run_order, regime in enumerate(regime_order, start=1):
            pass_summaries: List[Dict[str, Any]] = []
            regime_event_rows: List[Dict[str, Any]] = []
            for pass_name, pass_config in PERF_EVENT_PASSES.items():
                print(
                    f"replicate={replicate} order={run_order} seed={paired_seed} "
                    f"regime={regime} pass={pass_name}",
                    flush=True,
                )
                benchmark_summary, event_rows = run_perf_pass(
                    args=args,
                    replicate=replicate,
                    run_order=run_order,
                    seed=paired_seed,
                    regime=regime,
                    pass_name=pass_name,
                    pass_config=pass_config,
                    output_dir=output_dir,
                )
                pass_summaries.append(benchmark_summary)
                regime_event_rows.extend(event_rows)
                all_event_rows.extend(event_rows)

            validate_pass_identity(pass_summaries)
            summary = build_replicate_summary(
                replicate=replicate,
                run_order=run_order,
                seed=paired_seed,
                regime=regime,
                core_benchmark_summary=pass_summaries[0],
                event_rows=regime_event_rows,
            )
            replicate_rows.append(summary)
            current_pair[regime] = summary

        layout_reference = current_pair[REGIMES[0]]
        for regime in REGIMES[1:]:
            validate_paired_layout(layout_reference, current_pair[regime])
        contrast_rows.extend(build_contrast_rows(replicate, current_pair))
        hypothesis_rows.append(
            build_hypothesis_check(
                replicate, current_pair, args.instructions_tolerance_percent
            )
        )

    write_csv(output_dir / "perf_event_summary.csv", EVENT_COLUMNS, all_event_rows)
    write_csv(output_dir / "replicate_summary.csv", REPLICATE_COLUMNS, replicate_rows)
    write_csv(
        output_dir / "paired_comparison.csv",
        list(contrast_rows[0].keys()),
        contrast_rows,
    )
    write_csv(
        output_dir / "hypothesis_checks.csv",
        list(hypothesis_rows[0].keys()),
        hypothesis_rows,
    )

    aggregate = aggregate_results(replicate_rows, contrast_rows, hypothesis_rows)
    with (output_dir / "aggregate_summary.json").open("w") as file:
        json.dump(aggregate, file, indent=2)
        file.write("\n")
    print(json.dumps(aggregate, indent=2))

    invalid_events = [row for row in all_event_rows if row["running_status"] == "invalid"]
    warning_events = [row for row in all_event_rows if row["running_status"] == "warning"]
    if invalid_events:
        print(
            f"INVALID: {len(invalid_events)} event measurements ran below "
            f"{args.running_invalid_percentage:g}% or were unavailable.",
            file=sys.stderr,
        )
    elif warning_events:
        print(
            f"WARNING: {len(warning_events)} event measurements ran below "
            f"{args.running_warning_percentage:g}%.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
