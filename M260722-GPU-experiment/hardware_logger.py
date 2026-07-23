import csv
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import psutil

from experiment_config import CSV_COLUMNS, yyyymmddhhmmss_log_path


PHASES = {
    "idle",
    "forward",
    "backward",
    "optimizer_step",
    "evaluation",
    "aggregation",
    "finished",
}


def read_minor_faults(pid: int) -> int:
    """Read minflt from /proc/<pid>/stat on Linux.

    /proc fields before the closing ')' may contain spaces in the process name,
    so parse from the right side of the command field. minflt is field 10 in
    procfs documentation and index 7 after removing pid/comm.
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        after_comm = stat.rsplit(")", 1)[1].strip().split()
        return int(after_comm[7])
    except (FileNotFoundError, IndexError, ValueError, PermissionError):
        return 0


@dataclass
class TrainingState:
    round: Any = ""
    epoch: Any = ""
    batch_idx: Any = ""
    phase: str = "idle"
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(
        self,
        *,
        round: Any = None,
        epoch: Any = None,
        batch_idx: Any = None,
        phase: Optional[str] = None,
    ) -> None:
        if phase is not None and phase not in PHASES:
            raise ValueError(f"Invalid phase {phase}. Expected one of {sorted(PHASES)}")
        with self._lock:
            if round is not None:
                self.round = round
            if epoch is not None:
                self.epoch = epoch
            if batch_idx is not None:
                self.batch_idx = batch_idx
            if phase is not None:
                self.phase = phase

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "round": self.round,
                "epoch": self.epoch,
                "batch_idx": self.batch_idx,
                "phase": self.phase,
            }


class HardwareLogger:
    def __init__(
        self,
        *,
        log_dir: str,
        condition: Dict[str, Any],
        training_state: Optional[TrainingState] = None,
        pid: Optional[int] = None,
        fps: float = 10.0,
        cpu_freq_sample_ms: float = 1.0,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be greater than zero")
        if cpu_freq_sample_ms < 0:
            raise ValueError("cpu_freq_sample_ms must be zero or greater")
        self.log_dir = log_dir
        self.condition = dict(condition)
        self.training_state = training_state or TrainingState()
        self.pid = pid or os.getpid()
        self.interval = 1.0 / fps
        self.cpu_freq_interval = cpu_freq_sample_ms / 1000.0
        self.path = yyyymmddhhmmss_log_path(log_dir)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cpu_freq_thread: Optional[threading.Thread] = None
        self._cpu_freq_lock = threading.Lock()
        self._cpu_freq_sums = [0.0, 0.0, 0.0, 0.0]
        self._cpu_freq_counts = [0, 0, 0, 0]
        self._process = psutil.Process(self.pid)

    def __enter__(self) -> "HardwareLogger":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()

    def start(self) -> None:
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        # Create the file synchronously so permission failures stop the run
        # before training starts instead of being hidden in the logger thread.
        with self.path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
        self._process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None, percpu=True)
        if self.cpu_freq_interval > 0:
            self._cpu_freq_thread = threading.Thread(
                target=self._run_cpu_frequency,
                name="cpu-frequency-logger",
                daemon=True,
            )
            self._cpu_freq_thread.start()
        self._thread = threading.Thread(target=self._run, name="hardware-logger", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.training_state.update(phase="finished")
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._cpu_freq_thread is not None:
            self._cpu_freq_thread.join(timeout=5)

    def _run(self) -> None:
        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            while not self._stop_event.is_set():
                writer.writerow(self._sample())
                f.flush()
                self._stop_event.wait(self.interval)
            writer.writerow(self._sample())
            f.flush()

    def _run_cpu_frequency(self) -> None:
        """Accumulate high-rate samples for the next 10 FPS hardware row."""
        interval_ns = max(1, int(self.cpu_freq_interval * 1_000_000_000))
        next_deadline_ns = time.perf_counter_ns()

        while not self._stop_event.is_set():
            now_ns = time.perf_counter_ns()
            if now_ns < next_deadline_ns:
                wait_seconds = (next_deadline_ns - now_ns) / 1_000_000_000
                if self._stop_event.wait(wait_seconds):
                    break
                continue

            frequencies = self._read_cpu_frequencies()
            with self._cpu_freq_lock:
                for core_index, frequency in enumerate(frequencies):
                    if frequency != "":
                        self._cpu_freq_sums[core_index] += float(frequency)
                        self._cpu_freq_counts[core_index] += 1

            next_deadline_ns += interval_ns
            current_ns = time.perf_counter_ns()
            if current_ns > next_deadline_ns + interval_ns:
                missed = (current_ns - next_deadline_ns) // interval_ns
                next_deadline_ns += missed * interval_ns

    @staticmethod
    def _read_cpu_frequencies() -> list[Any]:
        try:
            frequencies = psutil.cpu_freq(percpu=True) or []
        except (NotImplementedError, OSError):
            frequencies = []
        return [
            frequencies[index].current if len(frequencies) > index else ""
            for index in range(4)
        ]

    def _consume_cpu_frequency_averages(self) -> list[Any]:
        if self.cpu_freq_interval <= 0:
            return self._read_cpu_frequencies()

        with self._cpu_freq_lock:
            averages = [
                self._cpu_freq_sums[index] / self._cpu_freq_counts[index]
                if self._cpu_freq_counts[index] > 0
                else ""
                for index in range(4)
            ]
            self._cpu_freq_sums = [0.0, 0.0, 0.0, 0.0]
            self._cpu_freq_counts = [0, 0, 0, 0]

        if all(value == "" for value in averages):
            return self._read_cpu_frequencies()
        return averages

    def _sample(self) -> Dict[str, Any]:
        now = datetime.now()
        cpu = psutil.cpu_percent(interval=None, percpu=True)
        cpu_freq = self._consume_cpu_frequency_averages()
        mem = psutil.virtual_memory()
        try:
            proc_mem = self._process.memory_info()
            proc_percent = self._process.memory_percent()
            proc_cpu = self._process.cpu_percent(interval=None)
            switches = self._process.num_ctx_switches()
            voluntary = switches.voluntary
            involuntary = switches.involuntary
            rss = proc_mem.rss
            vms = proc_mem.vms
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            proc_percent = proc_cpu = voluntary = involuntary = rss = vms = 0

        row = {
            "timestamp": now.isoformat(timespec="microseconds"),
            "timestamp_unix": now.timestamp(),
            "system_cpu_core_0": cpu[0] if len(cpu) > 0 else "",
            "system_cpu_core_1": cpu[1] if len(cpu) > 1 else "",
            "system_cpu_core_2": cpu[2] if len(cpu) > 2 else "",
            "system_cpu_core_3": cpu[3] if len(cpu) > 3 else "",
            "system_cpu_freq_core_0": cpu_freq[0],
            "system_cpu_freq_core_1": cpu_freq[1],
            "system_cpu_freq_core_2": cpu_freq[2],
            "system_cpu_freq_core_3": cpu_freq[3],
            "system_memory_percent": mem.percent,
            "system_memory_used": mem.used,
            "system_memory_available": mem.available,
            "process_cpu_percent": proc_cpu,
            "process_memory_rss": rss,
            "process_memory_vms": vms,
            "process_memory_percent": proc_percent,
            "process_ctx_switches_voluntary": voluntary,
            "process_ctx_switches_involuntary": involuntary,
            "process_minor_faults": read_minor_faults(self.pid),
        }
        row.update(self.condition)
        if not row.get("host"):
            try:
                row["host"] = socket.gethostbyname(socket.gethostname())
            except OSError:
                row["host"] = socket.gethostname()
        row.update(self.training_state.snapshot())
        for column in CSV_COLUMNS:
            row.setdefault(column, "")
        return row
