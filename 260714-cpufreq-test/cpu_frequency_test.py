#!/usr/bin/env python3
"""
CPU frequency sampling comparison.

Compares:
1. High-rate frequency samples, nominally every 1 ms
2. A single point sampled at 10 FPS, every 100 ms
3. Time-weighted average frequency inside each 100 ms window
4. Minimum and maximum frequency inside each window

Outputs:
- cpu_frequency_raw.csv
- cpu_frequency_windows.csv
- cpu_frequency_comparison.png

Install:
    python -m pip install psutil matplotlib

Run:
    python cpu_frequency_test.py

Examples:
    python cpu_frequency_test.py --duration 15
    python cpu_frequency_test.py --sample-ms 1 --fps 10
    python cpu_frequency_test.py --per-cpu 0
"""

from __future__ import annotations

import argparse
import csv
import math
import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import psutil


@dataclass
class Sample:
    """One frequency observation."""

    time_s: float
    frequency_mhz: float


@dataclass
class WindowResult:
    """Aggregated values for one low-FPS window."""

    start_s: float
    end_s: float
    center_s: float
    point_time_s: float
    point_frequency_mhz: float
    average_frequency_mhz: float
    minimum_frequency_mhz: float
    maximum_frequency_mhz: float
    standard_deviation_mhz: float
    sample_count: int


def read_frequency_mhz(cpu_index: int | None) -> float | None:
    """
    Read the CPU frequency exposed by psutil.

    cpu_index=None:
        Use psutil's system-wide current-frequency value.

    cpu_index=N:
        Use the frequency reported for logical CPU N, where supported.
    """
    try:
        if cpu_index is None:
            freq = psutil.cpu_freq(percpu=False)
            if freq is None or freq.current <= 0:
                return None
            return float(freq.current)

        frequencies = psutil.cpu_freq(percpu=True)
        if not frequencies:
            return None

        if cpu_index < 0 or cpu_index >= len(frequencies):
            raise ValueError(
                f"CPU index {cpu_index} is outside the valid range "
                f"0..{len(frequencies) - 1}"
            )

        current = frequencies[cpu_index].current
        if current <= 0:
            return None

        return float(current)

    except (NotImplementedError, OSError):
        return None


def collect_samples(
    duration_s: float,
    sample_interval_s: float,
    cpu_index: int | None,
) -> tuple[list[Sample], float]:
    """
    Collect frequency samples using an absolute-deadline scheduler.

    Absolute deadlines reduce accumulated drift. Actual timestamps are still
    recorded because sleep timing is not exact on normal desktop operating
    systems.
    """
    samples: list[Sample] = []

    start_ns = time.perf_counter_ns()
    duration_ns = int(duration_s * 1_000_000_000)
    interval_ns = int(sample_interval_s * 1_000_000_000)
    end_ns = start_ns + duration_ns

    next_deadline_ns = start_ns

    while True:
        now_ns = time.perf_counter_ns()

        if now_ns >= end_ns:
            break

        if now_ns < next_deadline_ns:
            remaining_s = (next_deadline_ns - now_ns) / 1_000_000_000
            time.sleep(remaining_s)
            continue

        frequency = read_frequency_mhz(cpu_index)
        observed_ns = time.perf_counter_ns()

        if frequency is not None and math.isfinite(frequency):
            samples.append(
                Sample(
                    time_s=(observed_ns - start_ns) / 1_000_000_000,
                    frequency_mhz=frequency,
                )
            )

        next_deadline_ns += interval_ns

        # If sampling was delayed significantly, skip stale deadlines rather
        # than rapidly issuing many back-to-back reads.
        current_ns = time.perf_counter_ns()
        if current_ns > next_deadline_ns + interval_ns:
            missed = (current_ns - next_deadline_ns) // interval_ns
            next_deadline_ns += missed * interval_ns

    actual_duration_s = (
        time.perf_counter_ns() - start_ns
    ) / 1_000_000_000

    return samples, actual_duration_s


def time_weighted_average(
    samples: list[Sample],
    window_start_s: float,
    window_end_s: float,
) -> float:
    """
    Estimate the time-weighted average using zero-order hold.

    Each observed frequency is treated as valid until the following sample.
    The first sample in the window is also extended backwards to the beginning
    of the window when necessary.
    """
    if not samples:
        return math.nan

    weighted_sum = 0.0
    covered_duration = 0.0

    for index, sample in enumerate(samples):
        segment_start = max(sample.time_s, window_start_s)

        if index + 1 < len(samples):
            next_time = samples[index + 1].time_s
        else:
            next_time = window_end_s

        segment_end = min(next_time, window_end_s)

        if segment_end > segment_start:
            duration = segment_end - segment_start
            weighted_sum += sample.frequency_mhz * duration
            covered_duration += duration

    first = samples[0]

    if first.time_s > window_start_s:
        duration = min(first.time_s, window_end_s) - window_start_s

        if duration > 0:
            weighted_sum += first.frequency_mhz * duration
            covered_duration += duration

    if covered_duration <= 0:
        return statistics.fmean(
            sample.frequency_mhz for sample in samples
        )

    return weighted_sum / covered_duration


def aggregate_windows(
    samples: list[Sample],
    window_s: float,
    duration_s: float,
) -> list[WindowResult]:
    """
    Split high-rate samples into fixed windows.

    The "10 FPS point" is the sample closest to the beginning of each window.
    It represents what a simple low-rate polling program might observe.
    """
    results: list[WindowResult] = []

    window_start = 0.0

    while window_start < duration_s:
        window_end = min(window_start + window_s, duration_s)

        inside = [
            sample
            for sample in samples
            if window_start <= sample.time_s < window_end
        ]

        if inside:
            point = min(
                inside,
                key=lambda sample: abs(sample.time_s - window_start),
            )

            values = [sample.frequency_mhz for sample in inside]

            result = WindowResult(
                start_s=window_start,
                end_s=window_end,
                center_s=(window_start + window_end) / 2,
                point_time_s=point.time_s,
                point_frequency_mhz=point.frequency_mhz,
                average_frequency_mhz=time_weighted_average(
                    inside,
                    window_start,
                    window_end,
                ),
                minimum_frequency_mhz=min(values),
                maximum_frequency_mhz=max(values),
                standard_deviation_mhz=(
                    statistics.pstdev(values)
                    if len(values) >= 2
                    else 0.0
                ),
                sample_count=len(inside),
            )
            results.append(result)

        window_start = window_end

    return results


def save_raw_csv(samples: list[Sample], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["time_s", "frequency_mhz"])

        for sample in samples:
            writer.writerow(
                [
                    f"{sample.time_s:.9f}",
                    f"{sample.frequency_mhz:.6f}",
                ]
            )


def save_window_csv(
    windows: list[WindowResult],
    path: Path,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "window_start_s",
                "window_end_s",
                "window_center_s",
                "point_time_s",
                "point_frequency_mhz",
                "weighted_average_frequency_mhz",
                "minimum_frequency_mhz",
                "maximum_frequency_mhz",
                "standard_deviation_mhz",
                "sample_count",
            ]
        )

        for window in windows:
            writer.writerow(
                [
                    f"{window.start_s:.9f}",
                    f"{window.end_s:.9f}",
                    f"{window.center_s:.9f}",
                    f"{window.point_time_s:.9f}",
                    f"{window.point_frequency_mhz:.6f}",
                    f"{window.average_frequency_mhz:.6f}",
                    f"{window.minimum_frequency_mhz:.6f}",
                    f"{window.maximum_frequency_mhz:.6f}",
                    f"{window.standard_deviation_mhz:.6f}",
                    window.sample_count,
                ]
            )


def create_plot(
    samples: list[Sample],
    windows: list[WindowResult],
    fps: float,
    output_path: Path,
    show_plot: bool,
) -> None:
    raw_times = [sample.time_s for sample in samples]
    raw_frequencies = [sample.frequency_mhz for sample in samples]

    point_times = [window.point_time_s for window in windows]
    point_frequencies = [
        window.point_frequency_mhz for window in windows
    ]

    window_centers = [window.center_s for window in windows]
    average_frequencies = [
        window.average_frequency_mhz for window in windows
    ]
    minimum_frequencies = [
        window.minimum_frequency_mhz for window in windows
    ]
    maximum_frequencies = [
        window.maximum_frequency_mhz for window in windows
    ]

    figure, axis = plt.subplots(figsize=(13, 7))

    axis.plot(
        raw_times,
        raw_frequencies,
        linewidth=0.7,
        alpha=0.55,
        label="High-rate raw samples",
    )

    axis.fill_between(
        window_centers,
        minimum_frequencies,
        maximum_frequencies,
        alpha=0.18,
        label="Min–max inside each window",
    )

    axis.plot(
        window_centers,
        average_frequencies,
        linewidth=2.2,
        label=f"Windowed time-weighted average ({fps:g} FPS)",
    )

    axis.scatter(
        point_times,
        point_frequencies,
        s=28,
        zorder=4,
        label=f"Single point sampled at {fps:g} FPS",
    )

    axis.set_title(
        "CPU frequency: low-rate point samples versus "
        "high-rate window averages"
    )
    axis.set_xlabel("Elapsed time [s]")
    axis.set_ylabel("Frequency [MHz]")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    figure.savefig(output_path, dpi=160)

    if show_plot:
        plt.show()
    else:
        plt.close(figure)


def print_summary(
    samples: list[Sample],
    windows: list[WindowResult],
    requested_sample_ms: float,
    fps: float,
    actual_duration_s: float,
) -> None:
    intervals_ms = [
        (samples[index].time_s - samples[index - 1].time_s) * 1000
        for index in range(1, len(samples))
    ]

    differences = [
        abs(
            window.point_frequency_mhz
            - window.average_frequency_mhz
        )
        for window in windows
    ]

    ranges = [
        window.maximum_frequency_mhz
        - window.minimum_frequency_mhz
        for window in windows
    ]

    print()
    print("Measurement summary")
    print("-------------------")
    print(f"OS:                    {platform.platform()}")
    print(f"psutil version:        {psutil.__version__}")
    print(f"Logical CPUs:          {psutil.cpu_count(logical=True)}")
    print(f"Actual duration:       {actual_duration_s:.3f} s")
    print(f"Raw sample count:      {len(samples)}")
    print(f"Requested interval:    {requested_sample_ms:.3f} ms")
    print(f"Low-rate comparison:   {fps:g} FPS")
    print(f"Window count:          {len(windows)}")

    if intervals_ms:
        print(
            f"Actual interval mean:  "
            f"{statistics.fmean(intervals_ms):.3f} ms"
        )
        print(
            f"Actual interval median:"
            f" {statistics.median(intervals_ms):.3f} ms"
        )
        print(
            f"Actual interval max:   "
            f"{max(intervals_ms):.3f} ms"
        )

    if differences:
        print()
        print("Difference between one point and window average")
        print(
            f"Mean absolute error:   "
            f"{statistics.fmean(differences):.2f} MHz"
        )
        print(
            f"Maximum difference:    "
            f"{max(differences):.2f} MHz"
        )

    if ranges:
        print()
        print("Variation inside each low-rate window")
        print(
            f"Mean min–max range:    "
            f"{statistics.fmean(ranges):.2f} MHz"
        )
        print(
            f"Maximum min–max range: "
            f"{max(ranges):.2f} MHz"
        )

    print()
    print(
        "Note: these values are psutil/OS frequency observations, "
        "not APERF/MPERF cumulative-counter measurements."
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare low-rate CPU-frequency polling against "
            "high-rate window averages."
        )
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Measurement duration in seconds. Default: 10",
    )
    parser.add_argument(
        "--sample-ms",
        type=float,
        default=1.0,
        help="Requested high-rate sampling interval in ms. Default: 1",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Low-rate sampling/window frequency. Default: 10",
    )
    parser.add_argument(
        "--per-cpu",
        type=int,
        default=None,
        metavar="CPU",
        help=(
            "Read one logical CPU where supported. "
            "Default: system-wide psutil value"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Output directory. Default: current directory",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the plot without opening a GUI window",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    if args.duration <= 0:
        raise ValueError("--duration must be greater than zero")

    if args.sample_ms <= 0:
        raise ValueError("--sample-ms must be greater than zero")

    if args.fps <= 0:
        raise ValueError("--fps must be greater than zero")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_interval_s = args.sample_ms / 1000
    window_s = 1 / args.fps

    initial_frequency = read_frequency_mhz(args.per_cpu)
    if initial_frequency is None:
        print(
            "CPU frequency is unavailable through psutil on this system."
        )
        print(
            "Try another OS-specific backend such as turbostat, "
            "Windows Performance Counters, or powermetrics."
        )
        return 1

    target_description = (
        f"logical CPU {args.per_cpu}"
        if args.per_cpu is not None
        else "system-wide psutil frequency"
    )

    print(f"Target: {target_description}")
    print(
        f"Collecting for {args.duration:g} seconds at a requested "
        f"{args.sample_ms:g} ms interval..."
    )

    samples, actual_duration_s = collect_samples(
        duration_s=args.duration,
        sample_interval_s=sample_interval_s,
        cpu_index=args.per_cpu,
    )

    if len(samples) < 2:
        print("Not enough valid samples were collected.")
        return 1

    windows = aggregate_windows(
        samples=samples,
        window_s=window_s,
        duration_s=args.duration,
    )

    raw_csv_path = output_dir / "cpu_frequency_raw.csv"
    window_csv_path = output_dir / "cpu_frequency_windows.csv"
    plot_path = output_dir / "cpu_frequency_comparison.png"

    save_raw_csv(samples, raw_csv_path)
    save_window_csv(windows, window_csv_path)

    create_plot(
        samples=samples,
        windows=windows,
        fps=args.fps,
        output_path=plot_path,
        show_plot=not args.no_show,
    )

    print_summary(
        samples=samples,
        windows=windows,
        requested_sample_ms=args.sample_ms,
        fps=args.fps,
        actual_duration_s=actual_duration_s,
    )

    print()
    print(f"Raw CSV:     {raw_csv_path.resolve()}")
    print(f"Window CSV:  {window_csv_path.resolve()}")
    print(f"Plot:        {plot_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
