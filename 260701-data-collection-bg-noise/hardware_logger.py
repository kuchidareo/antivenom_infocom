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
    ) -> None:
        self.log_dir = log_dir
        self.condition = dict(condition)
        self.training_state = training_state or TrainingState()
        self.pid = pid or os.getpid()
        self.interval = 1.0 / fps
        self.path = yyyymmddhhmmss_log_path(log_dir)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._process = psutil.Process(self.pid)

    def __enter__(self) -> "HardwareLogger":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()

    def start(self) -> None:
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self._process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None, percpu=True)
        self._thread = threading.Thread(target=self._run, name="hardware-logger", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.training_state.update(phase="finished")
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        with self.path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            while not self._stop_event.is_set():
                writer.writerow(self._sample())
                f.flush()
                self._stop_event.wait(self.interval)
            writer.writerow(self._sample())
            f.flush()

    def _sample(self) -> Dict[str, Any]:
        now = datetime.now()
        cpu = psutil.cpu_percent(interval=None, percpu=True)
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
