import argparse
import csv
import math
import statistics
from pathlib import Path
from collections import Counter
from typing import Dict, Iterable, List, Tuple


def percentile(values: List[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    index = int(round((len(ordered) - 1) * quantile))
    return ordered[index]


def metric_paths(input_dir: Path, latest_run: bool) -> List[Path]:
    paths = list(input_dir.glob("*_metrics.csv"))
    if not paths:
        return []
    if latest_run:
        return [max(paths, key=lambda path: path.stat().st_mtime_ns)]
    return sorted(paths)


def load_forward_rows(paths: Iterable[Path]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("metric_event") != "train_batch":
                    continue
                try:
                    rows.append(
                        {
                            "batch_size_actual": float(row["batch_size_actual"]),
                            "forward_elapsed_ms": float(row["forward_elapsed_ms"]),
                            "epoch": int(row["epoch"]),
                            "batch_idx": int(row["batch_idx"]),
                        }
                    )
                except (KeyError, TypeError, ValueError):
                    continue
    return rows


def phase_counts(path: Path, phase: str) -> Counter[Tuple[int, int]]:
    counts: Counter[Tuple[int, int]] = Counter()
    if not path.exists():
        raise FileNotFoundError(f"Missing telemetry companion file: {path}")
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("phase") != phase:
                continue
            if path.name.endswith("_perf.csv") and row.get("perf_status") != "ok":
                continue
            try:
                counts[(int(row["epoch"]), int(row["batch_idx"]))] += 1
            except (KeyError, TypeError, ValueError):
                continue
    return counts


def perf_enabled_percentages(path: Path, phase: str) -> List[float]:
    values: List[float] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("phase") != phase or row.get("perf_status") != "ok":
                continue
            for column, value in row.items():
                if not column.endswith("_enabled_pct") or not value:
                    continue
                try:
                    values.append(float(value))
                except ValueError:
                    continue
    return values


def report_actual_coverage(
    *,
    metrics_path: Path,
    full_rows: List[Dict[str, float]],
    minimum_samples: float,
    fail_below: bool,
) -> None:
    run_stem = metrics_path.name.removesuffix("_metrics.csv")
    full_keys = [(int(row["epoch"]), int(row["batch_idx"])) for row in full_rows]
    companion_paths = {
        "hardware": metrics_path.with_name(f"{run_stem}.csv"),
        "perf": metrics_path.with_name(f"{run_stem}_perf.csv"),
    }
    for telemetry_name, path in companion_paths.items():
        counts = phase_counts(path, "forward")
        per_batch = [counts[key] for key in full_keys]
        zero_batches = sum(value == 0 for value in per_batch)
        median_samples = statistics.median(per_batch) if per_batch else 0.0
        print(f"{telemetry_name}_forward_rows={sum(per_batch)}")
        print(f"{telemetry_name}_forward_samples_per_batch_min={min(per_batch, default=0)}")
        print(f"{telemetry_name}_forward_samples_per_batch_median={median_samples:.3f}")
        print(f"{telemetry_name}_forward_zero_coverage_batches={zero_batches}")
        if telemetry_name == "perf":
            enabled = perf_enabled_percentages(path, "forward")
            print(f"perf_forward_enabled_pct_min={min(enabled, default=math.nan):.3f}")
            print(
                "perf_forward_enabled_pct_median="
                f"{statistics.median(enabled) if enabled else math.nan:.3f}"
            )
        if fail_below and (zero_batches or median_samples < minimum_samples):
            raise SystemExit(
                f"Pilot rejected: {telemetry_name} forward coverage has median "
                f"{median_samples:.2f} samples and {zero_batches} zero-coverage batches; "
                f"required median is {minimum_samples:.2f} with no zero-coverage batches."
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--configured-batch-size", type=int, required=True)
    parser.add_argument("--hardware-fps", type=float, default=10.0)
    parser.add_argument("--minimum-expected-samples", type=float, default=2.0)
    parser.add_argument("--minimum-actual-samples", type=float, default=2.0)
    parser.add_argument("--expected-batches", type=int, default=0)
    parser.add_argument("--latest-run", action="store_true")
    parser.add_argument("--fail-below", action="store_true")
    args = parser.parse_args()

    paths = metric_paths(args.input_dir, args.latest_run)
    if not paths:
        raise SystemExit(f"No metrics CSV files found in {args.input_dir}")
    rows = load_forward_rows(paths)
    full_rows = [
        row
        for row in rows
        if int(row["batch_size_actual"]) == args.configured_batch_size
    ]
    full = [row["forward_elapsed_ms"] for row in full_rows]
    partial = [
        row["forward_elapsed_ms"]
        for row in rows
        if int(row["batch_size_actual"]) != args.configured_batch_size
    ]
    if not full:
        raise SystemExit(f"No full-batch forward timings found in {args.input_dir}")
    if args.expected_batches > 0 and len(rows) != args.expected_batches:
        raise SystemExit(
            f"Pilot rejected: expected {args.expected_batches} total batches, "
            f"found {len(rows)} in {paths[0]}"
        )

    median_ms = statistics.median(full)
    expected_samples = median_ms * args.hardware_fps / 1000.0
    print(f"timing_dir={args.input_dir}")
    print(f"total_forward_count={len(rows)}")
    print(f"full_forward_count={len(full)}")
    print(f"full_forward_min_ms={min(full):.3f}")
    print(f"full_forward_median_ms={median_ms:.3f}")
    print(f"full_forward_p95_ms={percentile(full, 0.95):.3f}")
    print(f"partial_forward_count={len(partial)}")
    if partial:
        print(f"partial_forward_median_ms={statistics.median(partial):.3f}")
    print(f"hardware_fps={args.hardware_fps:.3f}")
    print(f"expected_samples_per_median_forward={expected_samples:.3f}")

    if args.fail_below and expected_samples < args.minimum_expected_samples:
        raise SystemExit(
            "Pilot rejected: median full-batch forward is expected to receive "
            f"{expected_samples:.2f} hardware samples, below the required "
            f"{args.minimum_expected_samples:.2f}."
        )

    if args.latest_run:
        report_actual_coverage(
            metrics_path=paths[0],
            full_rows=full_rows,
            minimum_samples=args.minimum_actual_samples,
            fail_below=args.fail_below,
        )


if __name__ == "__main__":
    main()
