import csv
import ctypes
import errno
import fcntl
import os
import platform
import re
import shutil
import socket
import struct
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set

from experiment_config import CONDITION_COLUMNS, yyyymmddhhmmss_log_path
from hardware_logger import TrainingState


COMMON_PERF_EVENTS = [
    "cycles",
    "instructions",
    "task-clock",
    "context-switches",
    "cpu-migrations",
    "page-faults",
]

# Keep the default x86 set focused enough to limit PMU multiplexing.  The L1
# aliases are portable perf hardware-cache events, while the L2 pair is the
# Intel PMU's direct demand-reference/miss pair available on the CloudLab
# machines used by this experiment.  LLC events are intentionally excluded:
# LLC normally means L3 on these systems and must not be interpreted as L2.
X86_PERF_EVENTS = [
    *COMMON_PERF_EVENTS,
    "L1-dcache-loads",
    "L1-dcache-load-misses",
    "l2_rqsts.all_demand_references",
    "l2_rqsts.all_demand_miss",
]

RPI_PERF_EVENTS = [
    *COMMON_PERF_EVENTS,
    "branches",
    "branch-misses",
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
    "mem_access",
    "ase_spec",
    "vfp_spec",
    "inst_spec",
]

JETSON_PERF_EVENTS = [
    *COMMON_PERF_EVENTS,
    "br_retired",
    "br_mis_pred_retired",
    "l1d_cache",
    "l1d_cache_refill",
    "l1d_cache_wb",
    "l2d_cache",
    "l2d_cache_refill",
    "l2d_cache_wb",
    "bus_access",
    "mem_access",
    "inst_spec",
]

JETSON_HOSTS = {"192.168.0.141", "192.168.0.142"}
RPI_HOSTS = {f"192.168.0.{last_octet}" for last_octet in range(112, 122)}
DEFAULT_PERF_EVENTS = RPI_PERF_EVENTS


def default_perf_events_for_host(host: str) -> List[str]:
    normalized_host = host.strip()
    if normalized_host in JETSON_HOSTS:
        return list(JETSON_PERF_EVENTS)
    if normalized_host in RPI_HOSTS:
        return list(RPI_PERF_EVENTS)
    if platform.machine().lower() in {"x86_64", "amd64", "i386", "i686"}:
        return list(X86_PERF_EVENTS)
    return list(RPI_PERF_EVENTS)


BASE_COLUMNS = [
    "timestamp",
    "timestamp_unix",
    *CONDITION_COLUMNS,
    "round",
    "epoch",
    "batch_idx",
    "phase",
    "perf_pid",
    "perf_measurement_mode",
    "perf_scope",
    "perf_elapsed_sec",
    "perf_interval_ms",
    "perf_phase_start_timestamp",
    "perf_phase_start_unix",
    "perf_phase_end_timestamp",
    "perf_phase_end_unix",
    "perf_phase_duration_sec",
    "perf_events",
    "perf_status",
    "perf_error",
]


PERF_TYPE_HARDWARE = 0
PERF_TYPE_SOFTWARE = 1
PERF_TYPE_HW_CACHE = 3

PERF_FORMAT_TOTAL_TIME_ENABLED = 1 << 0
PERF_FORMAT_TOTAL_TIME_RUNNING = 1 << 1
PERF_FLAG_FD_CLOEXEC = 1 << 3

PERF_EVENT_IOC_ENABLE = 0x2400
PERF_EVENT_IOC_DISABLE = 0x2401
PERF_EVENT_IOC_RESET = 0x2403


class PerfEventAttr(ctypes.Structure):
    """Linux ``perf_event_attr`` through the config3 field (136 bytes)."""

    _fields_ = [
        ("type", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("config", ctypes.c_uint64),
        ("sample_period", ctypes.c_uint64),
        ("sample_type", ctypes.c_uint64),
        ("read_format", ctypes.c_uint64),
        ("flags", ctypes.c_uint64),
        ("wakeup_events", ctypes.c_uint32),
        ("bp_type", ctypes.c_uint32),
        ("config1", ctypes.c_uint64),
        ("config2", ctypes.c_uint64),
        ("branch_sample_type", ctypes.c_uint64),
        ("sample_regs_user", ctypes.c_uint64),
        ("sample_stack_user", ctypes.c_uint32),
        ("clockid", ctypes.c_int32),
        ("sample_regs_intr", ctypes.c_uint64),
        ("aux_watermark", ctypes.c_uint32),
        ("sample_max_stack", ctypes.c_uint16),
        ("reserved2", ctypes.c_uint16),
        ("aux_sample_size", ctypes.c_uint32),
        ("reserved3", ctypes.c_uint32),
        ("sig_data", ctypes.c_uint64),
        ("config3", ctypes.c_uint64),
    ]


@dataclass(frozen=True)
class PerfEventSpec:
    name: str
    type: int
    config: int
    config1: int = 0
    config2: int = 0
    config3: int = 0


@dataclass
class OpenPerfCounter:
    event: str
    tid: int
    fd: int


GENERIC_PERF_EVENT_SPECS = {
    "cycles": (PERF_TYPE_HARDWARE, 0),
    "instructions": (PERF_TYPE_HARDWARE, 1),
    "cache-references": (PERF_TYPE_HARDWARE, 2),
    "cache-misses": (PERF_TYPE_HARDWARE, 3),
    "branches": (PERF_TYPE_HARDWARE, 4),
    "branch-instructions": (PERF_TYPE_HARDWARE, 4),
    "branch-misses": (PERF_TYPE_HARDWARE, 5),
    "bus-cycles": (PERF_TYPE_HARDWARE, 6),
    "cpu-clock": (PERF_TYPE_SOFTWARE, 0),
    "task-clock": (PERF_TYPE_SOFTWARE, 1),
    "page-faults": (PERF_TYPE_SOFTWARE, 2),
    "faults": (PERF_TYPE_SOFTWARE, 2),
    "context-switches": (PERF_TYPE_SOFTWARE, 3),
    "cs": (PERF_TYPE_SOFTWARE, 3),
    "cpu-migrations": (PERF_TYPE_SOFTWARE, 4),
    "migrations": (PERF_TYPE_SOFTWARE, 4),
    "minor-faults": (PERF_TYPE_SOFTWARE, 5),
    "major-faults": (PERF_TYPE_SOFTWARE, 6),
    "L1-dcache-loads": (PERF_TYPE_HW_CACHE, 0),
    "L1-dcache-load-misses": (PERF_TYPE_HW_CACHE, 1 << 16),
    "L1-dcache-stores": (PERF_TYPE_HW_CACHE, 1 << 8),
    "L1-dcache-store-misses": (PERF_TYPE_HW_CACHE, (1 << 8) | (1 << 16)),
}


def _perf_event_open_syscall_number() -> int:
    machine = platform.machine().lower()
    syscall_numbers = {
        "x86_64": 298,
        "amd64": 298,
        "aarch64": 241,
        "arm64": 241,
        "armv7l": 364,
        "armv6l": 364,
    }
    try:
        return syscall_numbers[machine]
    except KeyError as exc:
        raise RuntimeError(f"perf_event_open is not configured for architecture {machine}.") from exc


def _parse_attr_integer(value: str) -> int:
    return int(value.strip().split()[0], 0)


def resolve_perf_event_spec(event: str, perf_binary: str = "perf") -> PerfEventSpec:
    """Resolve a perf event name to the kernel type/config tuple.

    Generic events are encoded locally. Architecture-specific aliases are
    resolved once through ``perf stat -vv`` before training begins.
    """
    generic = GENERIC_PERF_EVENT_SPECS.get(event)
    if generic is not None:
        return PerfEventSpec(event, *generic)

    result = subprocess.run(
        [perf_binary, "stat", "-vv", "--no-scale", "-e", event, "--", "true"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "LC_ALL": "C"},
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()
        message = detail[0] if detail else f"perf exited with code {result.returncode}"
        raise ValueError(f"Unable to resolve perf event {event!r}: {message}")

    values: Dict[str, int] = {}
    inside_attr = False
    for line in result.stderr.splitlines():
        if line.strip() == "perf_event_attr:":
            inside_attr = True
            continue
        if not inside_attr:
            continue
        stripped = line.strip()
        if stripped.startswith("sys_perf_event_open:") or stripped.startswith("---"):
            if values:
                break
            continue
        match = re.match(r"^(type|config|config1|config2|config3)\s+(.+)$", stripped)
        if match:
            values[match.group(1)] = _parse_attr_integer(match.group(2))

    if "type" not in values or "config" not in values:
        raise ValueError(f"perf did not expose type/config while resolving event {event!r}.")
    return PerfEventSpec(
        name=event,
        type=values["type"],
        config=values["config"],
        config1=values.get("config1", 0),
        config2=values.get("config2", 0),
        config3=values.get("config3", 0),
    )


def _open_perf_counter(spec: PerfEventSpec, tid: int, *, inherit: bool) -> int:
    attr = PerfEventAttr()
    attr.type = spec.type
    attr.size = ctypes.sizeof(PerfEventAttr)
    attr.config = spec.config
    attr.config1 = spec.config1
    attr.config2 = spec.config2
    attr.config3 = spec.config3
    attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING
    # disabled | inherit | exclude_guest. Match perf-stat's normal host scope.
    attr.flags = 1 | ((1 << 1) if inherit else 0) | (1 << 20)

    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.syscall(
        _perf_event_open_syscall_number(),
        ctypes.byref(attr),
        tid,
        -1,
        -1,
        PERF_FLAG_FD_CLOEXEC,
    )
    if fd < 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), f"event={spec.name} tid={tid}")
    return int(fd)


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
    """Parse one interval record emitted by ``perf stat -I ... -x,``."""
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
    """Collect interval perf counters with shared training-state annotations."""

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
        with self.path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.columns, extrasaction="ignore")
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
                file.flush()
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
                    message = line.strip()
                    if message and not message.startswith("#") and not self._error:
                        self._error = message
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
                    file.flush()
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
                file.flush()

            return_code = self._process.poll() if self._process is not None else None
            if return_code not in (0, None) and not self._stop_event.is_set():
                self._error = self._error or f"perf exited with code {return_code}"
                writer.writerow(self._base_row(perf_status="failed", perf_error=self._error))
                file.flush()

    def _base_row(self, *, perf_status: str, perf_error: str = "", elapsed_sec: Any = "") -> Dict[str, Any]:
        now = datetime.now()
        row: Dict[str, Any] = {
            "timestamp": now.isoformat(timespec="microseconds"),
            "timestamp_unix": now.timestamp(),
            "perf_pid": self.pid,
            "perf_measurement_mode": "interval",
            "perf_scope": "process_leader_with_inheritance",
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


class PhasePerfLogger:
    """Measure complete training operations rather than periodic intervals.

    Counters are opened before training and gated with perf ioctls around each
    forward, backward, and optimizer operation. Existing process threads are
    attached individually with inheritance so PyTorch workers created later
    are included as well. Logger thread IDs can be
    excluded to keep monitoring work outside the measured process scope.
    """

    def __init__(
        self,
        *,
        log_dir: str,
        condition: Dict[str, Any],
        training_state: Optional[TrainingState] = None,
        pid: Optional[int] = None,
        events: Optional[Iterable[str]] = None,
        perf_binary: str = "perf",
        path: Optional[str] = None,
        excluded_tids: Optional[Iterable[int]] = None,
        require_perf_available: bool = True,
    ) -> None:
        self.log_dir = log_dir
        self.condition = dict(condition)
        self.training_state = training_state or TrainingState()
        self.pid = pid or os.getpid()
        self.events = list(events or DEFAULT_PERF_EVENTS)
        if not self.events:
            raise ValueError("At least one perf event is required.")
        self.perf_binary = perf_binary
        if require_perf_available:
            check_perf_available(perf_binary)
        self.path = Path(path) if path else yyyymmddhhmmss_log_path(log_dir, suffix="_perf.csv")
        self.columns = perf_csv_columns(self.events)
        self.excluded_tids: Set[int] = {int(tid) for tid in (excluded_tids or []) if tid}
        self._specs: List[PerfEventSpec] = []
        self._counters: List[OpenPerfCounter] = []
        self._file: Any = None
        self._writer: Optional[csv.DictWriter] = None
        self._active = False
        self._lock = threading.Lock()

    def __enter__(self) -> "PhasePerfLogger":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()

    def _target_tids(self) -> List[int]:
        task_dir = Path(f"/proc/{self.pid}/task")
        tids = sorted(
            int(entry.name)
            for entry in task_dir.iterdir()
            if entry.name.isdigit() and int(entry.name) not in self.excluded_tids
        )
        if self.pid not in tids:
            raise RuntimeError(f"Process leader {self.pid} is unavailable for perf measurement.")
        return tids

    def start(self) -> None:
        if self._file is not None:
            return
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self._specs = [resolve_perf_event_spec(event, self.perf_binary) for event in self.events]
        target_tids = self._target_tids()
        try:
            for tid in target_tids:
                for spec in self._specs:
                    try:
                        # Every attached training thread inherits into workers it
                        # creates after logger startup. Existing threads are each
                        # opened exactly once, so this does not double count them.
                        fd = _open_perf_counter(spec, tid, inherit=True)
                    except OSError as exc:
                        # A worker can disappear while /proc is being enumerated.
                        if tid != self.pid and exc.errno in {errno.ENOENT, errno.ESRCH}:
                            break
                        raise
                    self._counters.append(OpenPerfCounter(spec.name, tid, fd))
        except Exception:
            self._close_counters()
            raise

        self._file = self.path.open("w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.columns, extrasaction="ignore")
        self._writer.writeheader()
        self._file.flush()

    def stop(self) -> None:
        with self._lock:
            if self._active:
                self._disable_all()
                self._active = False
            self._close_counters()
            if self._file is not None:
                self._file.flush()
                self._file.close()
                self._file = None
                self._writer = None

    def _close_counters(self) -> None:
        for counter in self._counters:
            try:
                os.close(counter.fd)
            except OSError:
                pass
        self._counters.clear()

    def _reset_and_enable_all(self) -> None:
        for counter in self._counters:
            fcntl.ioctl(counter.fd, PERF_EVENT_IOC_RESET, 0)
        for counter in self._counters:
            fcntl.ioctl(counter.fd, PERF_EVENT_IOC_ENABLE, 0)

    def _disable_all(self) -> None:
        for counter in self._counters:
            try:
                fcntl.ioctl(counter.fd, PERF_EVENT_IOC_DISABLE, 0)
            except OSError as exc:
                if exc.errno not in {errno.ENOENT, errno.ESRCH}:
                    raise

    def _read_all(self) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        raw_by_event: Dict[str, float] = {event: 0.0 for event in self.events}
        enabled_by_event: Dict[str, int] = {event: 0 for event in self.events}
        running_by_event: Dict[str, int] = {event: 0 for event in self.events}

        for counter in self._counters:
            payload = os.read(counter.fd, 24)
            if len(payload) != 24:
                raise OSError(f"Short perf counter read for {counter.event} on tid {counter.tid}.")
            value, time_enabled, time_running = struct.unpack("QQQ", payload)
            scaled = float(value)
            if 0 < time_running < time_enabled:
                scaled *= time_enabled / time_running
            raw_by_event[counter.event] += scaled
            enabled_by_event[counter.event] += time_enabled
            running_by_event[counter.event] += time_running

        counts: Dict[str, Any] = {}
        runtime: Dict[str, Any] = {}
        enabled: Dict[str, Any] = {}
        for event in self.events:
            column = f"perf_{sanitize_event_name(event)}"
            value = raw_by_event[event]
            # perf stat reports software clocks in milliseconds.
            if event in {"task-clock", "cpu-clock"}:
                value /= 1_000_000.0
            counts[column] = value
            runtime[f"{column}_runtime_pct"] = running_by_event[event]
            total_enabled = enabled_by_event[event]
            enabled[f"{column}_enabled_pct"] = (
                100.0 * running_by_event[event] / total_enabled if total_enabled > 0 else ""
            )
        return counts, runtime, enabled

    @contextmanager
    def measure_phase(self) -> Iterator[None]:
        if self._writer is None or self._file is None:
            raise RuntimeError("PhasePerfLogger must be started before measuring a phase.")
        with self._lock:
            if self._active:
                raise RuntimeError("Nested phase perf measurements are not supported.")
            state = self.training_state.snapshot()
            self._reset_and_enable_all()
            self._active = True
            start_wall = datetime.now()
            start_ns = time.perf_counter_ns()

        operation_error = ""
        try:
            yield
        except BaseException as exc:
            operation_error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            end_ns = time.perf_counter_ns()
            end_wall = datetime.now()
            with self._lock:
                try:
                    self._disable_all()
                    counts, runtime, enabled = self._read_all()
                    # End the hardware-state phase before CSV I/O begins.
                    self.training_state.update(phase="idle")
                    status = "operation_failed" if operation_error else "ok"
                    row = self._phase_row(
                        state=state,
                        start_wall=start_wall,
                        end_wall=end_wall,
                        duration_sec=(end_ns - start_ns) / 1_000_000_000.0,
                        perf_status=status,
                        perf_error=operation_error,
                    )
                    row.update(counts)
                    row.update(runtime)
                    row.update(enabled)
                    self._writer.writerow(row)
                    self._file.flush()
                finally:
                    self._active = False

    def _phase_row(
        self,
        *,
        state: Dict[str, Any],
        start_wall: datetime,
        end_wall: datetime,
        duration_sec: float,
        perf_status: str,
        perf_error: str,
    ) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "timestamp": start_wall.isoformat(timespec="microseconds"),
            "timestamp_unix": start_wall.timestamp(),
            "perf_pid": self.pid,
            "perf_measurement_mode": "phase",
            "perf_scope": "all_existing_threads_plus_inherited_new_threads",
            "perf_elapsed_sec": duration_sec,
            "perf_interval_ms": "",
            "perf_phase_start_timestamp": start_wall.isoformat(timespec="microseconds"),
            "perf_phase_start_unix": start_wall.timestamp(),
            "perf_phase_end_timestamp": end_wall.isoformat(timespec="microseconds"),
            "perf_phase_end_unix": end_wall.timestamp(),
            "perf_phase_duration_sec": duration_sec,
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
        row.update(state)
        for column in self.columns:
            row.setdefault(column, "")
        return row


__all__ = [
    "COMMON_PERF_EVENTS",
    "DEFAULT_PERF_EVENTS",
    "JETSON_PERF_EVENTS",
    "PhasePerfLogger",
    "PerfLogger",
    "RPI_PERF_EVENTS",
    "X86_PERF_EVENTS",
    "check_perf_available",
    "default_perf_events_for_host",
    "parse_perf_stat_csv_line",
    "perf_csv_columns",
    "perf_paranoid_command",
    "resolve_perf_event_spec",
]
