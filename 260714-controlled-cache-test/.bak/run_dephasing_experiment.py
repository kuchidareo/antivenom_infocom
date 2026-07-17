from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import queue
import random
import shutil
import signal
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from dephasing_worker import worker_process_main


EVENT_CANDIDATES: Mapping[str, tuple[str, ...]] = {
    "cycles": ("cycles", "cpu-cycles"),
    "instructions": ("instructions",),
    "backend_stall": ("stall_backend", "stalled-cycles-backend"),
    "task_clock": ("task-clock",),
    "l2_read_access": ("l2d_cache_rd",),
    "l2_write_access": ("l2d_cache_wr",),
    "l2_access": ("l2d_cache",),
    "l2_read_refill": ("l2d_cache_refill_rd",),
    "l2_write_refill": ("l2d_cache_refill_wr",),
    "l2_refill": ("l2d_cache_refill",),
    "bus_read_access": ("bus_access_rd", "bus-access-rd"),
    "bus_write_access": ("bus_access_wr", "bus-access-wr"),
    "bus_access": ("bus_access", "bus-access"),
    "memory_access": ("mem_access",),
}

PASS_LOGICAL_EVENTS: Mapping[str, tuple[str, ...]] = {
    "core": ("cycles", "instructions", "backend_stall", "task_clock"),
    "l2": (
        "l2_read_access",
        "l2_write_access",
        "l2_access",
        "l2_read_refill",
        "l2_write_refill",
        "l2_refill",
        "task_clock",
    ),
    "bus": ("bus_read_access", "bus_write_access", "bus_access", "memory_access", "task_clock"),
}

REQUIRED_EVENTS = ("cycles", "instructions", "task_clock")

PERF_LONG_COLUMNS = [
    "replicate",
    "offset_delta_ms",
    "event_pass",
    "interval_end_sec",
    "interval_start_ns",
    "interval_end_ns",
    "cpu_id",
    "event",
    "perf_event",
    "count",
    "unit",
    "running_time_ns",
    "running_percentage",
    "running_status",
    "overlaps_workload",
    "fully_inside_workload",
    "analysis_interval",
]


def parse_float(value: str) -> Optional[float]:
    cleaned = value.strip().replace(",", "")
    if not cleaned or cleaned.startswith("<"):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def normalize_event_name(value: str) -> str:
    return value.strip().split(" [", 1)[0].split(":", 1)[0].lower()


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: Sequence[float], q: float) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return float("nan")
    if len(finite) == 1:
        return finite[0]
    position = (len(finite) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return finite[lower]
    fraction = position - lower
    return finite[lower] * (1.0 - fraction) + finite[upper] * fraction


def coefficient_of_variation(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("nan")
    mean = statistics.mean(finite)
    if mean == 0.0:
        return float("nan")
    return statistics.pstdev(finite) / mean


def pearson_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    pairs = [
        (a, b)
        for a, b in zip(left, right)
        if math.isfinite(a) and math.isfinite(b)
    ]
    if len(pairs) < 3:
        return float("nan")
    xs, ys = zip(*pairs)
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else float("nan")


def parse_number_list(value: str, value_type: Any) -> list[Any]:
    result = [value_type(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("Expected a non-empty comma-separated list")
    return result


def probe_perf_event(perf_binary: str, event: str, cpu_id: int) -> bool:
    with tempfile.TemporaryDirectory(prefix="dephasing-probe-") as temporary_dir:
        output = Path(temporary_dir) / "perf.csv"
        command = [
            perf_binary,
            "stat",
            "-a",
            "-A",
            "-C",
            str(cpu_id),
            "-x",
            ";",
            "-o",
            str(output),
            "-e",
            event,
            "--",
            "sleep",
            "0.02",
        ]
        completed = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "LC_ALL": "C"},
            check=False,
        )
        text = output.read_text(errors="replace") if output.exists() else ""
        lowered = f"{text}\n{completed.stderr}".lower()
        return completed.returncode == 0 and not any(
            marker in lowered
            for marker in ("<not supported>", "<not counted>", "permission error", "no permission")
        )


def discover_events(perf_binary: str, cpu_id: int) -> tuple[dict[str, str], dict[str, list[str]]]:
    selected: dict[str, str] = {}
    unavailable: dict[str, list[str]] = {}
    for logical_name, candidates in EVENT_CANDIDATES.items():
        for candidate in candidates:
            if probe_perf_event(perf_binary, candidate, cpu_id):
                selected[logical_name] = candidate
                break
        if logical_name not in selected:
            unavailable[logical_name] = list(candidates)
    missing = [event for event in REQUIRED_EVENTS if event not in selected]
    if missing:
        raise RuntimeError(
            "Required perf events are unavailable: "
            f"{missing}. Run `sudo sysctl kernel.perf_event_paranoid=-1` and check `perf list`."
        )
    return selected, unavailable


def build_event_passes(selected: Mapping[str, str]) -> dict[str, dict[str, str]]:
    passes: dict[str, dict[str, str]] = {
        "core": {
            name: selected[name]
            for name in PASS_LOGICAL_EVENTS["core"]
            if name in selected
        }
    }

    l2_names: list[str] = []
    if "l2_read_access" in selected and "l2_write_access" in selected:
        l2_names.extend(("l2_read_access", "l2_write_access"))
    elif "l2_access" in selected:
        l2_names.append("l2_access")
    if "l2_read_refill" in selected and "l2_write_refill" in selected:
        l2_names.extend(("l2_read_refill", "l2_write_refill"))
    elif "l2_refill" in selected:
        l2_names.append("l2_refill")
    if l2_names:
        if "task_clock" in selected:
            l2_names.append("task_clock")
        passes["l2"] = {name: selected[name] for name in l2_names}

    bus_names: list[str] = []
    if "bus_read_access" in selected and "bus_write_access" in selected:
        bus_names.extend(("bus_read_access", "bus_write_access"))
    elif "bus_access" in selected:
        bus_names.append("bus_access")
    if "memory_access" in selected:
        bus_names.append("memory_access")
    if bus_names:
        if "task_clock" in selected:
            bus_names.append("task_clock")
        passes["bus"] = {name: selected[name] for name in bus_names}
    return passes


class PerfIntervalSession:
    def __init__(
        self,
        *,
        perf_binary: str,
        cpu_ids: Sequence[int],
        events: Mapping[str, str],
        interval_ms: int,
        output_path: Path,
        scale_counts: bool,
    ) -> None:
        self.perf_binary = perf_binary
        self.cpu_ids = list(cpu_ids)
        self.events = dict(events)
        self.interval_ms = interval_ms
        self.output_path = output_path
        self.scale_counts = scale_counts
        self.process: Optional[subprocess.Popen[str]] = None
        self.temporary_directory: Optional[tempfile.TemporaryDirectory[str]] = None
        self.control_fifo: Optional[Path] = None
        self.ack_fifo: Optional[Path] = None
        self.control_holder: Optional[int] = None
        self.ack_holder: Optional[int] = None
        self.control_file: Any = None
        self.ack_file: Any = None
        self.enabled = False
        self.launch_monotonic_ns = 0
        self.enable_monotonic_ns = 0
        self.disable_monotonic_ns = 0
        self.closed = False
        self.stderr_text = ""

    def start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary_directory = tempfile.TemporaryDirectory(prefix="dephasing-perf-")
        temporary_path = Path(self.temporary_directory.name)
        self.control_fifo = temporary_path / "control.fifo"
        self.ack_fifo = temporary_path / "ack.fifo"
        os.mkfifo(self.control_fifo)
        os.mkfifo(self.ack_fifo)
        self.control_holder = os.open(self.control_fifo, os.O_RDWR | os.O_NONBLOCK)
        self.ack_holder = os.open(self.ack_fifo, os.O_RDWR | os.O_NONBLOCK)

        hardware_events = [event for logical, event in self.events.items() if logical != "task_clock"]
        command = [
            self.perf_binary,
            "stat",
            "-a",
            "-A",
            "-C",
            ",".join(str(cpu) for cpu in self.cpu_ids),
            "-I",
            str(self.interval_ms),
            "-x",
            ";",
            "-o",
            str(self.output_path),
            "--delay=-1",
            f"--control=fifo:{self.control_fifo},{self.ack_fifo}",
        ]
        if not self.scale_counts:
            command.append("--no-scale")
        if hardware_events:
            grouped = "{" + ",".join(hardware_events) + "}"
            command.extend(("-e", grouped))
        if "task_clock" in self.events:
            command.extend(("-e", self.events["task_clock"]))
        command.extend(("--", "sleep", "86400"))

        before_ns = time.perf_counter_ns()
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "LC_ALL": "C"},
            start_new_session=True,
        )
        after_ns = time.perf_counter_ns()
        self.launch_monotonic_ns = (before_ns + after_ns) // 2
        self.control_file = self.control_fifo.open("w", buffering=1)
        self.ack_file = self.ack_fifo.open("r", buffering=1)
        time.sleep(0.03)
        if self.process.poll() is not None:
            stderr = self.process.stderr.read() if self.process.stderr else ""
            raise RuntimeError(f"perf exited before measurement: {stderr.strip()}")

    def _command(self, command: str) -> None:
        self.control_file.write(f"{command}\n")
        acknowledgment = self.ack_file.readline().replace("\x00", "").strip()
        if acknowledgment != "ack":
            raise RuntimeError(f"perf did not acknowledge {command!r}: {acknowledgment!r}")

    def enable(self) -> None:
        self._command("enable")
        self.enable_monotonic_ns = time.perf_counter_ns()
        self.enabled = True

    def disable(self) -> None:
        if self.enabled:
            self._command("disable")
            self.disable_monotonic_ns = time.perf_counter_ns()
            self.enabled = False

    def close(self) -> str:
        if self.closed:
            return self.stderr_text
        stderr = ""
        try:
            self.disable()
        finally:
            if self.process is not None and self.process.poll() is None:
                os.killpg(self.process.pid, signal.SIGINT)
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGTERM)
                    self.process.wait(timeout=5)
            if self.process is not None and self.process.stderr is not None:
                stderr = self.process.stderr.read()
            for file in (self.control_file, self.ack_file):
                if file is not None:
                    file.close()
            for descriptor in (self.control_holder, self.ack_holder):
                if descriptor is not None:
                    os.close(descriptor)
            if self.temporary_directory is not None:
                self.temporary_directory.cleanup()
        self.stderr_text = stderr
        self.closed = True
        return self.stderr_text


def collect_messages(
    message_queue: Any,
    expected_kind: str,
    expected_count: int,
    processes: Sequence[Any],
    timeout_sec: float,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    deadline = time.monotonic() + timeout_sec
    while len(messages) < expected_count:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Timed out waiting for {expected_kind} worker messages")
        try:
            message = message_queue.get(timeout=min(1.0, remaining))
        except queue.Empty:
            failed = [process for process in processes if process.exitcode not in (None, 0)]
            if failed:
                raise RuntimeError(f"Worker exited early with codes {[p.exitcode for p in failed]}")
            continue
        if message.get("kind") == "error":
            raise RuntimeError(
                f"Worker {message.get('worker_id')} failed: {message.get('error')}\n"
                f"{message.get('traceback', '')}"
            )
        if message.get("kind") != expected_kind:
            raise RuntimeError(f"Expected {expected_kind!r}, received {message!r}")
        messages.append(dict(message))
    return sorted(messages, key=lambda item: int(item["worker_id"]))


def read_timing_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(newline="") as file:
            for row in csv.DictReader(file):
                converted: dict[str, Any] = dict(row)
                for name in (
                    "replicate",
                    "worker_id",
                    "cpu_id",
                    "step",
                    "planned_first_start_ns",
                    "start_ns",
                    "end_ns",
                    "duration_ns",
                ):
                    converted[name] = int(float(converted[name]))
                converted["offset_delta_ms"] = float(converted["offset_delta_ms"])
                converted["duration_ms"] = float(converted["duration_ms"])
                rows.append(converted)
    return rows


def summarize_worker_timing(
    timing_rows: Sequence[Mapping[str, Any]], measured_messages: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    by_worker: dict[int, list[Mapping[str, Any]]] = {}
    by_step: dict[int, list[Mapping[str, Any]]] = {}
    for row in timing_rows:
        by_worker.setdefault(int(row["worker_id"]), []).append(row)
        by_step.setdefault(int(row["step"]), []).append(row)

    first_starts = [min(int(row["start_ns"]) for row in rows) for rows in by_worker.values()]
    last_ends = [max(int(row["end_ns"]) for row in rows) for rows in by_worker.values()]
    worker_means = [statistics.mean(float(row["duration_ms"]) for row in rows) for rows in by_worker.values()]
    all_durations = [float(row["duration_ms"]) for row in timing_rows]
    iteration_start_skews = [
        (max(int(row["start_ns"]) for row in rows) - min(int(row["start_ns"]) for row in rows))
        / 1_000_000.0
        for rows in by_step.values()
        if len(rows) == len(by_worker)
    ]
    iteration_duration_cvs = [
        coefficient_of_variation([float(row["duration_ms"]) for row in rows])
        for rows in by_step.values()
        if len(rows) == len(by_worker)
    ]
    wall_time_sec = (max(last_ends) - min(first_starts)) / 1_000_000_000.0
    operations = len(timing_rows)
    process_cpu_time_sec = sum(
        int(message["process_cpu_time_ns"]) for message in measured_messages
    ) / 1_000_000_000.0
    return {
        "worker_count": len(by_worker),
        "operations": operations,
        "workload_start_ns": min(first_starts),
        "workload_end_ns": max(last_ends),
        "wall_time_sec": wall_time_sec,
        "aggregate_throughput_backward_per_sec": operations / wall_time_sec,
        "first_start_skew_ms": (max(first_starts) - min(first_starts)) / 1_000_000.0,
        "completion_skew_ms": (max(last_ends) - min(last_ends)) / 1_000_000.0,
        "iteration_start_skew_median_ms": statistics.median(iteration_start_skews),
        "iteration_start_skew_p95_ms": percentile(iteration_start_skews, 0.95),
        "duration_mean_ms": statistics.mean(all_durations),
        "duration_median_ms": statistics.median(all_durations),
        "duration_p95_ms": percentile(all_durations, 0.95),
        "worker_mean_duration_cv": coefficient_of_variation(worker_means),
        "iteration_duration_cv_mean": statistics.mean(iteration_duration_cvs),
        "worker_process_cpu_time_sec": process_cpu_time_sec,
        "effective_parallelism": process_cpu_time_sec / wall_time_sec,
    }


def execute_worker_run(
    *,
    args: argparse.Namespace,
    context: Any,
    replicate: int,
    offset_delta_ms: float,
    event_pass: str,
    run_dir: Path,
    cpu_ids: Sequence[int],
    interval_ms: int,
    perf_events: Optional[Mapping[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]], Optional[PerfIntervalSession]]:
    run_dir.mkdir(parents=True, exist_ok=True)
    message_queue = context.Queue()
    release_event = context.Event()
    measurement_closed_event = context.Event()
    release_ns = context.Value("q", 0)
    processes = []
    timing_paths: list[Path] = []
    perf_session: Optional[PerfIntervalSession] = None

    if perf_events is not None:
        perf_session = PerfIntervalSession(
            perf_binary=args.perf_binary,
            cpu_ids=cpu_ids,
            events=perf_events,
            interval_ms=interval_ms,
            output_path=run_dir / "perf_interval_raw.csv",
            scale_counts=args.scale_counts,
        )

    try:
        for worker_id, cpu_id in enumerate(cpu_ids):
            timing_path = run_dir / f"worker_{worker_id}_timings.csv"
            timing_paths.append(timing_path)
            config = {
                "replicate": replicate,
                "offset_delta_ms": offset_delta_ms,
                "event_pass": event_pass,
                "worker_id": worker_id,
                "cpu_id": cpu_id,
                "start_offset_ms": worker_id * offset_delta_ms,
                "steps": args.steps,
                "warmup": args.warmup,
                "batch_size": args.batch_size,
                "channels": args.channels,
                "spatial_size": args.spatial_size,
                "gradient_bank_size": args.gradient_bank_size,
                "gradient_scale": args.gradient_scale,
                "model_seed": args.seed + (replicate - 1) * 1000 + worker_id,
                "gradient_seed": args.seed + (replicate - 1) * 1000 + 900,
                "spin_threshold_us": args.spin_threshold_us,
                "worker_timeout_sec": args.worker_timeout_sec,
                "timing_path": str(timing_path),
            }
            process = context.Process(
                target=worker_process_main,
                args=(
                    config,
                    message_queue,
                    release_event,
                    release_ns,
                    measurement_closed_event,
                ),
                name=f"dephase-worker-{worker_id}",
            )
            process.start()
            processes.append(process)

        ready_messages = collect_messages(
            message_queue, "ready", len(cpu_ids), processes, args.worker_timeout_sec
        )
        for message in ready_messages:
            if message["affinity"] != [message["cpu_id"]] or message["torch_num_threads"] != 1:
                raise RuntimeError(f"Worker is not correctly pinned/single-threaded: {message}")

        if perf_session is not None:
            # Start perf after warmup so disabled 2 ms rows do not dominate the
            # raw file and perf setup cannot perturb worker initialization.
            perf_session.start()
            perf_session.enable()
        release_ns.value = time.perf_counter_ns() + int(args.start_lead_ms * 1_000_000.0)
        release_event.set()
        measured_messages = collect_messages(
            message_queue, "measured", len(cpu_ids), processes, args.worker_timeout_sec
        )
        if perf_session is not None:
            perf_session.disable()
            perf_stderr = perf_session.close()
            if perf_stderr:
                (run_dir / "perf_stderr.txt").write_text(perf_stderr)
        measurement_closed_event.set()

        for process in processes:
            process.join(timeout=args.worker_timeout_sec)
            if process.exitcode != 0:
                raise RuntimeError(f"Worker {process.name} exited with code {process.exitcode}")

        timing_rows = read_timing_rows(timing_paths)
        timing_summary = summarize_worker_timing(timing_rows, measured_messages)
        timing_summary.update(
            {
                "replicate": replicate,
                "offset_delta_ms": offset_delta_ms,
                "event_pass": event_pass,
                "interval_ms": interval_ms if perf_events is not None else "",
                "worker_cpus": ",".join(str(cpu) for cpu in cpu_ids),
            }
        )
        (run_dir / "worker_setup.json").write_text(json.dumps(ready_messages, indent=2) + "\n")
        return timing_summary, timing_rows, perf_session
    except BaseException:
        release_event.set()
        measurement_closed_event.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=3)
        raise
    finally:
        if perf_session is not None:
            stderr = perf_session.close()
            if stderr:
                (run_dir / "perf_stderr.txt").write_text(stderr)


def perf_running_status(
    count: Optional[float], value: Optional[float], warning: float, invalid: float
) -> str:
    if count is None or value is None or value < invalid:
        return "invalid"
    if value < warning:
        return "warning"
    return "ideal"


def parse_perf_intervals(
    *,
    path: Path,
    selected_events: Mapping[str, str],
    perf_disable_ns: int,
    interval_ms: int,
    workload_start_ns: int,
    workload_end_ns: int,
    replicate: int,
    offset_delta_ms: float,
    event_pass: str,
    warning_percentage: float,
    invalid_percentage: float,
) -> list[dict[str, Any]]:
    actual_to_logical = {
        normalize_event_name(actual): logical for logical, actual in selected_events.items()
    }
    parsed_rows: list[dict[str, Any]] = []
    with path.open(errors="replace") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = [field.strip() for field in line.split(";")]
            if len(fields) < 5:
                continue
            elapsed = parse_float(fields[0])
            cpu_text = fields[1]
            if elapsed is None or not cpu_text.upper().startswith("CPU"):
                continue
            try:
                cpu_id = int(cpu_text[3:])
            except ValueError:
                continue
            actual_event = normalize_event_name(fields[4])
            logical_event = actual_to_logical.get(actual_event)
            if logical_event is None:
                continue
            count = parse_float(fields[2])
            runtime = parse_float(fields[5]) if len(fields) > 5 else None
            percentage = parse_float(fields[6]) if len(fields) > 6 else None
            parsed_rows.append(
                {
                    "elapsed": elapsed,
                    "cpu_id": cpu_id,
                    "logical_event": logical_event,
                    "perf_event": fields[4],
                    "count": count,
                    "unit": fields[3],
                    "runtime": runtime,
                    "percentage": percentage,
                }
            )

    active_elapsed_values = [
        float(row["elapsed"]) for row in parsed_rows if row["count"] is not None
    ]
    if not active_elapsed_values:
        raise RuntimeError(
            f"perf produced no counted intervals in {path}. Check event support and perf permissions."
        )
    # `perf -I` continues printing <not counted> after a control FIFO disable.
    # Anchor the final interval containing counts to the acknowledged disable
    # timestamp instead of assuming that perf's elapsed clock starts at Popen.
    final_active_elapsed = max(active_elapsed_values)
    first_active_elapsed = min(active_elapsed_values)
    rows: list[dict[str, Any]] = []
    for parsed in parsed_rows:
        elapsed = float(parsed["elapsed"])
        if elapsed < first_active_elapsed or elapsed > final_active_elapsed:
            continue
        cpu_id = int(parsed["cpu_id"])
        logical_event = str(parsed["logical_event"])
        count = parsed["count"]
        runtime = parsed["runtime"]
        percentage = parsed["percentage"]
        interval_end_ns = perf_disable_ns + int(
            (elapsed - final_active_elapsed) * 1_000_000_000.0
        )
        interval_start_ns = interval_end_ns - interval_ms * 1_000_000
        overlaps = interval_end_ns > workload_start_ns and interval_start_ns < workload_end_ns
        fully_inside = interval_start_ns >= workload_start_ns and interval_end_ns <= workload_end_ns
        rows.append(
            {
                "replicate": replicate,
                "offset_delta_ms": offset_delta_ms,
                "event_pass": event_pass,
                "interval_end_sec": elapsed,
                "interval_start_ns": interval_start_ns,
                "interval_end_ns": interval_end_ns,
                "cpu_id": cpu_id,
                "event": logical_event,
                "perf_event": parsed["perf_event"],
                "count": "" if count is None else count,
                "unit": parsed["unit"],
                "running_time_ns": "" if runtime is None else runtime,
                "running_percentage": "" if percentage is None else percentage,
                "running_status": perf_running_status(
                    count, percentage, warning_percentage, invalid_percentage
                ),
                "overlaps_workload": overlaps,
                "fully_inside_workload": fully_inside,
                "analysis_interval": False,
            }
        )

    complete_intervals = sorted(
        {
            float(row["interval_end_sec"])
            for row in rows
            if bool(row["fully_inside_workload"])
        }
    )
    use_full = len(complete_intervals) >= 3
    for row in rows:
        row["analysis_interval"] = (
            bool(row["fully_inside_workload"]) if use_full else bool(row["overlaps_workload"])
        )
    return rows


def sum_optional(values: Sequence[Optional[float]]) -> float:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return sum(finite) if finite else float("nan")


def build_wide_perf_rows(long_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, int], dict[str, Any]] = {}
    for row in long_rows:
        key = (float(row["interval_end_sec"]), int(row["cpu_id"]))
        wide = grouped.setdefault(
            key,
            {
                "replicate": row["replicate"],
                "offset_delta_ms": row["offset_delta_ms"],
                "event_pass": row["event_pass"],
                "interval_end_sec": row["interval_end_sec"],
                "interval_start_ns": row["interval_start_ns"],
                "interval_end_ns": row["interval_end_ns"],
                "cpu_id": row["cpu_id"],
                "analysis_interval": row["analysis_interval"],
            },
        )
        wide[str(row["event"])] = (
            float(row["count"]) if row["count"] != "" else float("nan")
        )

    result = []
    for wide in grouped.values():
        l2_access = wide.get("l2_access", float("nan"))
        if not math.isfinite(l2_access):
            l2_access = sum_optional(
                [wide.get("l2_read_access"), wide.get("l2_write_access")]
            )
        l2_refill = wide.get("l2_refill", float("nan"))
        if not math.isfinite(l2_refill):
            l2_refill = sum_optional(
                [wide.get("l2_read_refill"), wide.get("l2_write_refill")]
            )
        bus_access = wide.get("bus_access", float("nan"))
        if not math.isfinite(bus_access):
            bus_access = sum_optional(
                [wide.get("bus_read_access"), wide.get("bus_write_access")]
            )
        wide["l2_access_total"] = l2_access
        wide["l2_refill_total"] = l2_refill
        wide["bus_access_total"] = bus_access
        result.append(wide)
    return sorted(result, key=lambda row: (float(row["interval_end_sec"]), int(row["cpu_id"])))


def synchrony_metrics(
    wide_rows: Sequence[Mapping[str, Any]], cpu_ids: Sequence[int]
) -> dict[str, Any]:
    selected = [row for row in wide_rows if bool(row["analysis_interval"])]
    signal = ""
    for candidate in ("bus_access_total", "l2_access_total", "memory_access", "l2_refill_total"):
        if any(math.isfinite(float(row.get(candidate, float("nan")))) for row in selected):
            signal = candidate
            break
    if not signal:
        return {
            "synchrony_signal": "",
            "cross_core_correlation": float("nan"),
            "simultaneous_burst_fraction": float("nan"),
            "aggregate_peak_to_mean": float("nan"),
            "request_total": float("nan"),
            "synchrony_interval_count": 0,
        }

    by_interval: dict[float, dict[int, float]] = {}
    by_cpu: dict[int, list[float]] = {cpu: [] for cpu in cpu_ids}
    for row in selected:
        value = float(row.get(signal, float("nan")))
        by_interval.setdefault(float(row["interval_end_sec"]), {})[int(row["cpu_id"])] = value
    complete = [values for _, values in sorted(by_interval.items()) if all(cpu in values for cpu in cpu_ids)]
    for values in complete:
        for cpu in cpu_ids:
            by_cpu[cpu].append(values[cpu])

    correlations = []
    for index, left_cpu in enumerate(cpu_ids):
        for right_cpu in cpu_ids[index + 1 :]:
            correlation = pearson_correlation(by_cpu[left_cpu], by_cpu[right_cpu])
            if math.isfinite(correlation):
                correlations.append(correlation)
    thresholds = {cpu: percentile(by_cpu[cpu], 0.90) for cpu in cpu_ids}
    required_high_cores = max(1, math.ceil(0.75 * len(cpu_ids)))
    simultaneous = 0
    aggregate = []
    for values in complete:
        high_count = sum(values[cpu] >= thresholds[cpu] for cpu in cpu_ids)
        simultaneous += high_count >= required_high_cores
        aggregate.append(sum(values[cpu] for cpu in cpu_ids))
    mean_aggregate = statistics.mean(aggregate) if aggregate else float("nan")
    return {
        "synchrony_signal": signal,
        "cross_core_correlation": statistics.mean(correlations) if correlations else float("nan"),
        "simultaneous_burst_fraction": simultaneous / len(complete) if complete else float("nan"),
        "aggregate_peak_to_mean": (
            percentile(aggregate, 0.95) / mean_aggregate
            if aggregate and mean_aggregate != 0.0
            else float("nan")
        ),
        "request_total": sum(aggregate),
        "synchrony_interval_count": len(complete),
    }


def summarize_perf_pass(
    long_rows: Sequence[Mapping[str, Any]], wide_rows: Sequence[Mapping[str, Any]], cpu_ids: Sequence[int]
) -> dict[str, Any]:
    selected = [row for row in wide_rows if bool(row["analysis_interval"])]

    def total(name: str) -> float:
        return sum_optional([float(row.get(name, float("nan"))) for row in selected])

    cycles = total("cycles")
    instructions = total("instructions")
    backend_stall = total("backend_stall")
    l2_access = total("l2_access_total")
    l2_refill = total("l2_refill_total")
    bus_access = total("bus_access_total")
    task_clock = total("task_clock")
    selected_long = [row for row in long_rows if bool(row["analysis_interval"])]
    running_percentages = [
        float(row["running_percentage"])
        for row in selected_long
        if row["running_percentage"] != ""
    ]
    return {
        **synchrony_metrics(wide_rows, cpu_ids),
        "cycles_total": cycles,
        "instructions_total": instructions,
        "ipc": instructions / cycles if cycles and math.isfinite(cycles) else float("nan"),
        "backend_stall_total": backend_stall,
        "backend_stall_fraction": (
            backend_stall / cycles if cycles and math.isfinite(backend_stall) else float("nan")
        ),
        "l2_access_total": l2_access,
        "l2_refill_total": l2_refill,
        "l2_refill_fraction": (
            l2_refill / l2_access if l2_access and math.isfinite(l2_refill) else float("nan")
        ),
        "bus_access_total": bus_access,
        "bus_access_per_task_clock_ms": (
            bus_access / task_clock if task_clock and math.isfinite(bus_access) else float("nan")
        ),
        "task_clock_total_ms": task_clock,
        "minimum_running_percentage": min(running_percentages) if running_percentages else float("nan"),
        "all_events_valid": bool(selected_long) and all(
            row["running_status"] != "invalid" for row in selected_long
        ),
        "all_events_ideal": bool(selected_long) and all(
            row["running_status"] == "ideal" for row in selected_long
        ),
    }


def choose_interval_ms(
    args: argparse.Namespace,
    context: Any,
    output_dir: Path,
    cpu_ids: Sequence[int],
    event_passes: Mapping[str, Mapping[str, str]],
) -> tuple[int, list[dict[str, Any]]]:
    if args.sampling_interval_ms != "auto":
        return int(args.sampling_interval_ms), []

    calibration_dir = output_dir / "sampling_overhead_calibration"
    baseline, _, _ = execute_worker_run(
        args=args,
        context=context,
        replicate=0,
        offset_delta_ms=0.0,
        event_pass="monitor_off",
        run_dir=calibration_dir / "monitor_off",
        cpu_ids=cpu_ids,
        interval_ms=0,
        perf_events=None,
    )
    heaviest_pass_name = max(event_passes, key=lambda name: len(event_passes[name]))
    calibration_rows = []
    chosen = args.interval_candidates_ms[-1]
    for interval_ms in args.interval_candidates_ms:
        monitored, _, session = execute_worker_run(
            args=args,
            context=context,
            replicate=0,
            offset_delta_ms=0.0,
            event_pass=f"monitor_on_{heaviest_pass_name}",
            run_dir=calibration_dir / f"monitor_on_{interval_ms}ms",
            cpu_ids=cpu_ids,
            interval_ms=interval_ms,
            perf_events=event_passes[heaviest_pass_name],
        )
        overhead = 100.0 * (
            monitored["wall_time_sec"] - baseline["wall_time_sec"]
        ) / baseline["wall_time_sec"]
        row = {
            "interval_ms": interval_ms,
            "event_pass": heaviest_pass_name,
            "monitor_off_wall_time_sec": baseline["wall_time_sec"],
            "monitor_on_wall_time_sec": monitored["wall_time_sec"],
            "overhead_percent": overhead,
            "accepted": overhead <= args.max_monitoring_overhead_percent,
            "raw_perf_path": str(session.output_path) if session is not None else "",
        }
        calibration_rows.append(row)
        if row["accepted"]:
            chosen = interval_ms
            break
    return chosen, calibration_rows


def combine_condition_summary(
    *,
    replicate: int,
    offset_delta_ms: float,
    pass_rows: Sequence[Mapping[str, Any]],
    single_worker_throughput: float,
    worker_count: int,
) -> dict[str, Any]:
    by_pass = {str(row["event_pass"]): row for row in pass_rows}
    timing_source = by_pass.get("core", pass_rows[0])

    synchrony_source = next(
        (
            by_pass[name]
            for name in ("bus", "l2", "core")
            if name in by_pass and str(by_pass[name].get("synchrony_signal", ""))
        ),
        {},
    )

    throughput = float(timing_source["aggregate_throughput_backward_per_sec"])
    return {
        "replicate": replicate,
        "offset_delta_ms": offset_delta_ms,
        "worker_count": worker_count,
        "first_start_skew_ms": timing_source["first_start_skew_ms"],
        "completion_skew_ms": timing_source["completion_skew_ms"],
        "iteration_start_skew_median_ms": timing_source["iteration_start_skew_median_ms"],
        "iteration_start_skew_p95_ms": timing_source["iteration_start_skew_p95_ms"],
        "duration_median_ms": timing_source["duration_median_ms"],
        "worker_mean_duration_cv": timing_source["worker_mean_duration_cv"],
        "iteration_duration_cv_mean": timing_source["iteration_duration_cv_mean"],
        "wall_time_sec": timing_source["wall_time_sec"],
        "aggregate_throughput_backward_per_sec": throughput,
        "effective_parallelism": timing_source["effective_parallelism"],
        "single_worker_throughput_backward_per_sec": single_worker_throughput,
        "parallel_efficiency": throughput / (worker_count * single_worker_throughput),
        "synchrony_signal": synchrony_source.get("synchrony_signal", ""),
        "cross_core_correlation": synchrony_source.get(
            "cross_core_correlation", float("nan")
        ),
        "simultaneous_burst_fraction": synchrony_source.get(
            "simultaneous_burst_fraction", float("nan")
        ),
        "aggregate_peak_to_mean": synchrony_source.get(
            "aggregate_peak_to_mean", float("nan")
        ),
        "request_total": synchrony_source.get("request_total", float("nan")),
        "backend_stall_fraction": by_pass.get("core", {}).get(
            "backend_stall_fraction", float("nan")
        ),
        "l2_refill_fraction": by_pass.get("l2", {}).get(
            "l2_refill_fraction", float("nan")
        ),
        "bus_access_total": by_pass.get("bus", {}).get(
            "bus_access_total", float("nan")
        ),
        "bus_access_per_task_clock_ms": by_pass.get("bus", {}).get(
            "bus_access_per_task_clock_ms", float("nan")
        ),
        "minimum_running_percentage": min(
            (
                float(row["minimum_running_percentage"])
                for row in pass_rows
                if math.isfinite(float(row["minimum_running_percentage"]))
            ),
            default=float("nan"),
        ),
        "all_events_valid": all(bool(row["all_events_valid"]) for row in pass_rows),
        "all_events_ideal": all(bool(row["all_events_ideal"]) for row in pass_rows),
    }


def aggregate_conditions(condition_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "first_start_skew_ms",
        "iteration_start_skew_median_ms",
        "duration_median_ms",
        "iteration_duration_cv_mean",
        "aggregate_throughput_backward_per_sec",
        "effective_parallelism",
        "parallel_efficiency",
        "cross_core_correlation",
        "simultaneous_burst_fraction",
        "aggregate_peak_to_mean",
        "request_total",
        "backend_stall_fraction",
        "l2_refill_fraction",
        "bus_access_total",
    )
    rows = []
    for delta in sorted({float(row["offset_delta_ms"]) for row in condition_rows}):
        selected = [row for row in condition_rows if float(row["offset_delta_ms"]) == delta]
        output: dict[str, Any] = {"offset_delta_ms": delta, "replicates": len(selected)}
        for metric in metrics:
            values = [float(row[metric]) for row in selected if math.isfinite(float(row[metric]))]
            output[f"{metric}_mean"] = statistics.mean(values) if values else float("nan")
            output[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        rows.append(output)
    return rows


def make_plots(
    output_dir: Path,
    condition_rows: Sequence[Mapping[str, Any]],
    perf_wide_rows: Sequence[Mapping[str, Any]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; skipping plots", flush=True)
        return

    metrics = (
        ("cross_core_correlation", "Cross-core correlation"),
        ("simultaneous_burst_fraction", "Simultaneous burst fraction"),
        ("aggregate_peak_to_mean", "Aggregate P95 / mean"),
        ("backend_stall_fraction", "Backend stall / cycles"),
        ("parallel_efficiency", "Parallel efficiency"),
        ("aggregate_throughput_backward_per_sec", "Aggregate backward/s"),
    )
    deltas = sorted({float(row["offset_delta_ms"]) for row in condition_rows})
    fig, axes = plt.subplots(3, 2, figsize=(11, 11), squeeze=False)
    for axis, (metric, label) in zip(axes.flat, metrics):
        means = []
        errors = []
        for delta in deltas:
            values = [
                float(row[metric])
                for row in condition_rows
                if float(row["offset_delta_ms"]) == delta and math.isfinite(float(row[metric]))
            ]
            means.append(statistics.mean(values) if values else float("nan"))
            errors.append(statistics.stdev(values) if len(values) > 1 else 0.0)
            axis.scatter([delta] * len(values), values, color="#4C78A8", alpha=0.55, s=22)
        axis.errorbar(deltas, means, yerr=errors, color="#D1495B", marker="o", capsize=3)
        axis.set_xlabel("Initial offset delta (ms)")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.25)
    fig.suptitle("Controlled dephasing: lockstep-contention metrics")
    fig.tight_layout()
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / "dephasing_dose_response.png", dpi=180, bbox_inches="tight")
    fig.savefig(plot_dir / "dephasing_dose_response.pdf", bbox_inches="tight")
    plt.close(fig)

    for replicate in sorted({int(row["replicate"]) for row in condition_rows}):
        for delta in deltas:
            candidates = [
                row
                for row in perf_wide_rows
                if int(row["replicate"]) == replicate
                and float(row["offset_delta_ms"]) == delta
                and bool(row["analysis_interval"])
            ]
            signal = ""
            pass_name = ""
            for candidate_pass, candidate_signal in (
                ("bus", "bus_access_total"),
                ("l2", "l2_access_total"),
                ("bus", "memory_access"),
                ("l2", "l2_refill_total"),
            ):
                if any(
                    row["event_pass"] == candidate_pass
                    and math.isfinite(float(row.get(candidate_signal, float("nan"))))
                    for row in candidates
                ):
                    pass_name = candidate_pass
                    signal = candidate_signal
                    break
            if not signal:
                continue
            selected = [row for row in candidates if row["event_pass"] == pass_name]
            first_end_ns = min(int(row["interval_end_ns"]) for row in selected)
            fig, axis = plt.subplots(figsize=(9, 4.5))
            for cpu_id in sorted({int(row["cpu_id"]) for row in selected}):
                cpu_rows = sorted(
                    (row for row in selected if int(row["cpu_id"]) == cpu_id),
                    key=lambda row: int(row["interval_end_ns"]),
                )
                axis.plot(
                    [
                        (int(row["interval_end_ns"]) - first_end_ns) / 1_000_000.0
                        for row in cpu_rows
                    ],
                    [float(row.get(signal, float("nan"))) for row in cpu_rows],
                    label=f"CPU {cpu_id}",
                    linewidth=1.2,
                )
            axis.set_xlabel("Measurement time (ms)")
            axis.set_ylabel(f"{signal} per interval")
            axis.set_title(
                f"Replicate {replicate}, initial delta {delta:g} ms ({pass_name} pass)"
            )
            axis.grid(True, alpha=0.25)
            axis.legend(ncol=4)
            fig.tight_layout()
            filename = f"request_trace_rep{replicate:02d}_delta_{delta:g}ms.png"
            fig.savefig(plot_dir / filename, dpi=180, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Run four single-thread Conv2D backward workers pinned to separate cores, "
            "sweep their one-time initial offsets, and measure per-core request synchrony."
        )
    )
    parser.add_argument("--output-dir", default=str(script_dir / "dephasing_results"))
    parser.add_argument("--perf-binary", default="perf")
    parser.add_argument("--cpus", default="0,1,2,3")
    parser.add_argument("--offset-deltas-ms", default="0,0.25,0.5,1,2")
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--spatial-size", type=int, default=32)
    parser.add_argument("--gradient-bank-size", type=int, default=16)
    parser.add_argument("--gradient-scale", type=float, default=1e-1)
    parser.add_argument(
        "--sampling-interval-ms",
        default="auto",
        help="Per-core perf interval in milliseconds, or 'auto' to test 2,5,10 ms.",
    )
    parser.add_argument("--interval-candidates-ms", default="2,5,10")
    parser.add_argument("--max-monitoring-overhead-percent", type=float, default=2.0)
    parser.add_argument("--start-lead-ms", type=float, default=25.0)
    parser.add_argument("--spin-threshold-us", type=float, default=200.0)
    parser.add_argument("--seed", type=int, default=260714)
    parser.add_argument("--worker-timeout-sec", type=float, default=600.0)
    parser.add_argument("--running-warning-percentage", type=float, default=99.0)
    parser.add_argument("--running-invalid-percentage", type=float, default=95.0)
    parser.add_argument("--scale-counts", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if shutil.which(args.perf_binary) is None:
        parser.error(f"perf executable was not found: {args.perf_binary}")
    args.cpu_ids = parse_number_list(args.cpus, int)
    args.offset_deltas_ms = parse_number_list(args.offset_deltas_ms, float)
    args.interval_candidates_ms = parse_number_list(args.interval_candidates_ms, int)
    if len(args.cpu_ids) != 4 or len(set(args.cpu_ids)) != 4:
        parser.error("--cpus must contain exactly four distinct CPU IDs")
    available_cpus = os.sched_getaffinity(0)
    missing_cpus = sorted(set(args.cpu_ids) - available_cpus)
    if missing_cpus:
        parser.error(f"Requested CPUs are outside this process affinity: {missing_cpus}")
    if args.replicates <= 0 or args.steps <= 0 or args.warmup < 0:
        parser.error("--replicates and --steps must be positive; --warmup must be non-negative")
    if args.sampling_interval_ms != "auto":
        try:
            explicit_interval = int(args.sampling_interval_ms)
        except ValueError:
            parser.error("--sampling-interval-ms must be 'auto' or a positive integer")
        if explicit_interval <= 0:
            parser.error("--sampling-interval-ms must be positive")
    if not 0 <= args.running_invalid_percentage <= args.running_warning_percentage <= 100:
        parser.error("Require 0 <= invalid running percentage <= warning percentage <= 100")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_events, unavailable_events = discover_events(args.perf_binary, args.cpu_ids[0])
    event_passes = build_event_passes(selected_events)
    print("Selected perf events:", flush=True)
    for logical, actual in selected_events.items():
        print(f"  {logical:22s} -> {actual}", flush=True)
    if unavailable_events:
        print("Unavailable optional event families: " + ", ".join(unavailable_events), flush=True)

    context = mp.get_context("spawn")
    interval_ms, calibration_rows = choose_interval_ms(
        args, context, output_dir, args.cpu_ids, event_passes
    )
    print(f"Using perf sampling interval: {interval_ms} ms", flush=True)
    if calibration_rows:
        write_csv(
            output_dir / "sampling_overhead_calibration.csv",
            calibration_rows,
            list(calibration_rows[0].keys()),
        )
        if not any(bool(row["accepted"]) for row in calibration_rows):
            print(
                "WARNING: no tested interval kept monitoring overhead below "
                f"{args.max_monitoring_overhead_percent:g}%.",
                flush=True,
            )

    configuration = {
        **vars(args),
        "cpu_ids": args.cpu_ids,
        "offset_deltas_ms": args.offset_deltas_ms,
        "interval_candidates_ms": args.interval_candidates_ms,
        "selected_events": selected_events,
        "unavailable_events": unavailable_events,
        "event_passes": event_passes,
        "selected_sampling_interval_ms": interval_ms,
        "parallel_efficiency_definition": (
            "aggregate_throughput / (worker_count * single_worker_throughput)"
        ),
        "interpretation": (
            "Offsets are applied once before the first measured backward. No per-iteration "
            "sleep is used. Per-core interval counts are system-wide on the pinned CPUs."
        ),
    }
    with (output_dir / "experiment_config.json").open("w") as file:
        json.dump(configuration, file, indent=2, default=str)
        file.write("\n")

    all_timing_rows: list[dict[str, Any]] = []
    all_perf_long_rows: list[dict[str, Any]] = []
    all_perf_wide_rows: list[dict[str, Any]] = []
    pass_summary_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []

    for replicate in range(1, args.replicates + 1):
        print(f"replicate={replicate}: single-worker baseline", flush=True)
        baseline_dir = output_dir / "raw" / f"replicate_{replicate:02d}" / "single_worker_baseline"
        baseline, baseline_timings, _ = execute_worker_run(
            args=args,
            context=context,
            replicate=replicate,
            offset_delta_ms=0.0,
            event_pass="single_worker_baseline",
            run_dir=baseline_dir,
            cpu_ids=[args.cpu_ids[0]],
            interval_ms=0,
            perf_events=None,
        )
        all_timing_rows.extend(baseline_timings)
        offset_order = list(args.offset_deltas_ms)
        random.Random(args.seed + replicate - 1).shuffle(offset_order)

        for offset_delta_ms in offset_order:
            current_pass_rows = []
            for pass_name, pass_events in event_passes.items():
                print(
                    f"replicate={replicate} delta={offset_delta_ms:g} ms pass={pass_name}",
                    flush=True,
                )
                run_dir = (
                    output_dir
                    / "raw"
                    / f"replicate_{replicate:02d}"
                    / f"delta_{offset_delta_ms:g}ms"
                    / pass_name
                )
                timing_summary, timing_rows, perf_session = execute_worker_run(
                    args=args,
                    context=context,
                    replicate=replicate,
                    offset_delta_ms=offset_delta_ms,
                    event_pass=pass_name,
                    run_dir=run_dir,
                    cpu_ids=args.cpu_ids,
                    interval_ms=interval_ms,
                    perf_events=pass_events,
                )
                if perf_session is None:
                    raise RuntimeError("Internal error: perf session was not created")
                long_rows = parse_perf_intervals(
                    path=perf_session.output_path,
                    selected_events=pass_events,
                    perf_disable_ns=perf_session.disable_monotonic_ns,
                    interval_ms=interval_ms,
                    workload_start_ns=int(timing_summary["workload_start_ns"]),
                    workload_end_ns=int(timing_summary["workload_end_ns"]),
                    replicate=replicate,
                    offset_delta_ms=offset_delta_ms,
                    event_pass=pass_name,
                    warning_percentage=args.running_warning_percentage,
                    invalid_percentage=args.running_invalid_percentage,
                )
                wide_rows = build_wide_perf_rows(long_rows)
                pass_summary = {
                    **timing_summary,
                    **summarize_perf_pass(long_rows, wide_rows, args.cpu_ids),
                    "raw_perf_path": str(perf_session.output_path),
                }
                current_pass_rows.append(pass_summary)
                pass_summary_rows.append(pass_summary)
                all_timing_rows.extend(timing_rows)
                all_perf_long_rows.extend(long_rows)
                all_perf_wide_rows.extend(wide_rows)

            condition_rows.append(
                combine_condition_summary(
                    replicate=replicate,
                    offset_delta_ms=offset_delta_ms,
                    pass_rows=current_pass_rows,
                    single_worker_throughput=float(
                        baseline["aggregate_throughput_backward_per_sec"]
                    ),
                    worker_count=len(args.cpu_ids),
                )
            )

    timing_fields = list(all_timing_rows[0].keys())
    write_csv(output_dir / "worker_timings.csv", all_timing_rows, timing_fields)
    write_csv(output_dir / "perf_intervals_long.csv", all_perf_long_rows, PERF_LONG_COLUMNS)
    wide_fields = sorted({key for row in all_perf_wide_rows for key in row.keys()})
    write_csv(output_dir / "perf_intervals_wide.csv", all_perf_wide_rows, wide_fields)
    pass_fields = sorted({key for row in pass_summary_rows for key in row.keys()})
    write_csv(output_dir / "pass_summary.csv", pass_summary_rows, pass_fields)
    condition_fields = list(condition_rows[0].keys())
    write_csv(output_dir / "condition_summary.csv", condition_rows, condition_fields)
    aggregate_rows = aggregate_conditions(condition_rows)
    write_csv(output_dir / "aggregate_summary.csv", aggregate_rows, list(aggregate_rows[0].keys()))
    if not args.no_plots:
        make_plots(output_dir, condition_rows, all_perf_wide_rows)

    print(f"Completed controlled dephasing experiment: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
