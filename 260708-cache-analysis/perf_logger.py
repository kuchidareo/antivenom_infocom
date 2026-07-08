import csv
import os
import re
import shutil
import socket
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from experiment_config import CONDITION_COLUMNS, yyyymmddhhmmss_log_path
from hardware_logger import TrainingState


DEFAULT_PERF_EVENTS = [
    "cycles",
    "instructions",
    "l1d_cache_rd",
    "l1d_cache_refill_rd",
    "l1d_cache_wr",
    "l1d_cache_refill_wr",
    "l2d_cache_rd",
    "l2d_cache_refill_rd",
    "l2d_cache_wr",
    "l2d_cache_refill_wr",
    "bus_access_rd",
    "bus_access_wr",
]


BASE_COLUMNS = [
    "timestamp",
    "timestamp_unix",
    *CONDITION_COLUMNS,
    "round",
    "epoch",
    "batch_idx",
    "phase",
    "perf_pid",
    "perf_elapsed_sec",
    "perf_interval_ms",
    "perf_events",
    "perf_status",
    "perf_error",
]


def sanitize_event_name(event_name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", event_name.strip()).strip("_").lower()
    return cleaned or "event"


def perf_event_columns(events: Sequence[str]) -> List[str]:
    return [f"perf_{sanitize_event_name(event)}" for event in events]


def perf_event_enabled_columns(events: Sequence[str]) -> List[str]:
    return [f"perf_{sanitize_event_name(event)}_enabled_pct" for event in events]


def perf_event_runtime_columns(events: Sequence[str]) -> List[str]:
    return [f"perf_{sanitize_event_name(event)}_runtime_pct" for event in events]


def perf_csv_columns(events: Sequence[str]) -> List[str]:
    return [
        *BASE_COLUMNS,
        *perf_event_columns(events),
        *perf_event_enabled_columns(events),
        *perf_event_runtime_columns(events),
    ]


def parse_perf_number(value: str) -> Any:
    value = value.strip()
    if not value or value.startswith("<not"):
        return ""
    value = value.replace(",", "")
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return ""


def parse_perf_stat_csv_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse one `perf stat -I ... -x,` output line.

    perf prints interval stat records to stderr. With CSV output, the common
    layout is:

        elapsed_sec,count,unit,event,runtime_pct,enabled_pct,...

    Some versions insert empty fields or reorder trailing metrics slightly, so
    this parser only relies on the first four fields and keeps best-effort
    parsing for runtime/enabled percentages.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    parts = [part.strip() for part in stripped.split(",")]
    if len(parts) < 4:
        return None
    try:
        elapsed_sec = float(parts[0])
    except ValueError:
        return None

    count = parse_perf_number(parts[1])
    event = parts[3].strip()
    if not event:
        return None

    runtime_pct = parse_perf_number(parts[4]) if len(parts) > 4 else ""
    enabled_pct = parse_perf_number(parts[5]) if len(parts) > 5 else ""
    return {
        "elapsed_sec": elapsed_sec,
        "event": event,
        "count": count,
        "runtime_pct": runtime_pct,
        "enabled_pct": enabled_pct,
    }


def check_perf_available(perf_binary: str = "perf") -> None:
    if shutil.which(perf_binary) is None:
        raise FileNotFoundError(f"perf binary not found: {perf_binary}")


def perf_paranoid_command() -> str:
    return "sudo sysctl kernel.perf_event_paranoid=-1"


class PerfLogger:
    """10 FPS perf counter logger with training-state annotations.

    Interpretation notes:
    - The logger samples hardware counters with `perf stat -I`.
    - Each row is one sampling interval, not one training batch.
    - The phase/epoch/batch columns are a snapshot of the shared TrainingState
      when the interval is written, so optimizer_step intervals can be selected
      later for cache-behavior analysis.
    - If perf reports `<not counted>` for an event, the CSV value is left empty.
    """

    def __init__(
        self,
        *,
        log_dir: str,
        condition: Dict[str, Any],
        training_state: Optional[TrainingState] = None,
        pid: Optional[int] = None,
        fps: float = 10.0,
        events: Optional[Iterable[str]] = None,
        perf_binary: str = "perf",
        path: Optional[str] = None,
        require_perf_available: bool = True,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive.")
        self.log_dir = log_dir
        self.condition = dict(condition)
        self.training_state = training_state or TrainingState()
        self.pid = pid or os.getpid()
        self.interval_ms = max(1, int(round(1000.0 / fps)))
        self.events = list(events or DEFAULT_PERF_EVENTS)
        if not self.events:
            raise ValueError("At least one perf event is required.")
        self.perf_binary = perf_binary
        if require_perf_available:
            check_perf_available(perf_binary)
        self.path = Path(path) if path else yyyymmddhhmmss_log_path(log_dir, suffix="_perf.csv")
        self.columns = perf_csv_columns(self.events)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._process: Optional[subprocess.Popen[str]] = None
        self._error = ""
        self._event_to_column = {
            event: f"perf_{sanitize_event_name(event)}"
            for event in self.events
        }

    def __enter__(self) -> "PerfLogger":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()

    def start(self) -> None:
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="perf-logger", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=3)
        if self._thread is not None:
            self._thread.join(timeout=5)

    def command(self) -> List[str]:
        return [
            self.perf_binary,
            "stat",
            "-I",
            str(self.interval_ms),
            "-x",
            ",",
            "-e",
            ",".join(self.events),
            "-p",
            str(self.pid),
        ]

    def _run(self) -> None:
        with self.path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns, extrasaction="ignore")
            writer.writeheader()
            try:
                self._process = subprocess.Popen(
                    self.command(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
            except OSError as exc:
                self._error = str(exc)
                writer.writerow(self._base_row(perf_status="failed_to_start", perf_error=self._error))
                f.flush()
                return

            current_elapsed: Optional[float] = None
            current_counts: Dict[str, Any] = {}
            current_runtime: Dict[str, Any] = {}
            current_enabled: Dict[str, Any] = {}

            assert self._process.stderr is not None
            for line in self._process.stderr:
                if self._stop_event.is_set():
                    break
                parsed = parse_perf_stat_csv_line(line)
                if parsed is None:
                    if "No permission" in line or "Permission" in line:
                        self._error = line.strip()
                    continue

                elapsed = float(parsed["elapsed_sec"])
                if current_elapsed is not None and elapsed != current_elapsed:
                    writer.writerow(
                        self._sample_row(
                            elapsed_sec=current_elapsed,
                            counts=current_counts,
                            runtime=current_runtime,
                            enabled=current_enabled,
                        )
                    )
                    f.flush()
                    current_counts = {}
                    current_runtime = {}
                    current_enabled = {}

                current_elapsed = elapsed
                event = str(parsed["event"])
                event_column = self._event_to_column.get(event, f"perf_{sanitize_event_name(event)}")
                current_counts[event_column] = parsed["count"]
                current_runtime[f"{event_column}_runtime_pct"] = parsed["runtime_pct"]
                current_enabled[f"{event_column}_enabled_pct"] = parsed["enabled_pct"]

            if current_elapsed is not None:
                writer.writerow(
                    self._sample_row(
                        elapsed_sec=current_elapsed,
                        counts=current_counts,
                        runtime=current_runtime,
                        enabled=current_enabled,
                    )
                )
                f.flush()

            return_code = self._process.poll() if self._process is not None else None
            if return_code not in (0, None) and not self._stop_event.is_set():
                self._error = self._error or f"perf exited with code {return_code}"
                writer.writerow(self._base_row(perf_status="failed", perf_error=self._error))
                f.flush()

    def _base_row(self, *, perf_status: str, perf_error: str = "", elapsed_sec: Any = "") -> Dict[str, Any]:
        now = datetime.now()
        row: Dict[str, Any] = {
            "timestamp": now.isoformat(timespec="microseconds"),
            "timestamp_unix": now.timestamp(),
            "perf_pid": self.pid,
            "perf_elapsed_sec": elapsed_sec,
            "perf_interval_ms": self.interval_ms,
            "perf_events": ",".join(self.events),
            "perf_status": perf_status,
            "perf_error": perf_error,
        }
        row.update(self.condition)
        if not row.get("host"):
            try:
                row["host"] = socket.gethostbyname(socket.gethostname())
            except OSError:
                row["host"] = socket.gethostname()
        row.update(self.training_state.snapshot())
        for column in self.columns:
            row.setdefault(column, "")
        return row

    def _sample_row(
        self,
        *,
        elapsed_sec: float,
        counts: Dict[str, Any],
        runtime: Dict[str, Any],
        enabled: Dict[str, Any],
    ) -> Dict[str, Any]:
        row = self._base_row(perf_status="ok", perf_error=self._error, elapsed_sec=elapsed_sec)
        row.update(counts)
        row.update(runtime)
        row.update(enabled)
        return row


__all__ = [
    "DEFAULT_PERF_EVENTS",
    "PerfLogger",
    "check_perf_available",
    "parse_perf_stat_csv_line",
    "perf_csv_columns",
    "perf_paranoid_command",
]
