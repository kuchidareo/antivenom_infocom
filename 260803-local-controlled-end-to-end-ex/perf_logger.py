"""Layer-boundary Linux perf counters for the controlled end-to-end experiment.

Counters are opened once, then reset/enabled immediately before each leaf
module operation and disabled immediately after it. No interval sampling thread
is used. Rows stay in memory until the current batch backward pass completes,
so CSV I/O is outside every measured layer region.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import errno
import fcntl
import json
import os
import platform
import re
import shutil
import socket
import struct
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from experiment_config import CONDITION_COLUMNS


DEFAULT_PERF_EVENTS = [
    "cycles",
    "instructions",
    "branches",
    "branch-misses",
    "L1-dcache-loads",
    "L1-dcache-load-misses",
]
MAX_PERF_EVENTS = 6

PERF_TYPE_HARDWARE = 0
PERF_TYPE_SOFTWARE = 1
PERF_TYPE_HW_CACHE = 3
PERF_FORMAT_TOTAL_TIME_ENABLED = 1 << 0
PERF_FORMAT_TOTAL_TIME_RUNNING = 1 << 1
PERF_FLAG_FD_CLOEXEC = 1 << 3
PERF_IOC_FLAG_GROUP = 1
PERF_EVENT_IOC_ENABLE = 0x2400
PERF_EVENT_IOC_DISABLE = 0x2401
PERF_EVENT_IOC_RESET = 0x2403


class PerfEventAttr(ctypes.Structure):
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


@dataclass
class CounterGroup:
    tid: int
    leader_fd: int
    counters: List[OpenPerfCounter]


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
    "context-switches": (PERF_TYPE_SOFTWARE, 3),
    "cpu-migrations": (PERF_TYPE_SOFTWARE, 4),
    "minor-faults": (PERF_TYPE_SOFTWARE, 5),
    "major-faults": (PERF_TYPE_SOFTWARE, 6),
    "L1-dcache-loads": (PERF_TYPE_HW_CACHE, 0),
    "L1-dcache-load-misses": (PERF_TYPE_HW_CACHE, 1 << 16),
    "L1-dcache-stores": (PERF_TYPE_HW_CACHE, 1 << 8),
    "L1-dcache-store-misses": (PERF_TYPE_HW_CACHE, (1 << 8) | (1 << 16)),
}


def sanitize_event_name(event_name: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", event_name).strip("_").lower() or "event"


def parse_perf_events(value: str) -> List[str]:
    events = [event.strip() for event in value.split(",") if event.strip()]
    events = events or list(DEFAULT_PERF_EVENTS)
    if len(events) > MAX_PERF_EVENTS:
        raise ValueError(
            f"At most {MAX_PERF_EVENTS} simultaneous perf events are supported; got {len(events)}."
        )
    if len(set(events)) != len(events):
        raise ValueError("Perf event names must be unique.")
    return events


def _perf_event_open_syscall_number() -> int:
    machine = platform.machine().lower()
    numbers = {
        "x86_64": 298,
        "amd64": 298,
        "aarch64": 241,
        "arm64": 241,
        "armv7l": 364,
        "armv6l": 364,
    }
    if machine not in numbers:
        raise RuntimeError(f"perf_event_open is not configured for architecture {machine}.")
    return numbers[machine]


def _parse_attr_integer(value: str) -> int:
    return int(value.strip().split()[0], 0)


def resolve_perf_event_spec(event: str, perf_binary: str = "perf") -> PerfEventSpec:
    generic = GENERIC_PERF_EVENT_SPECS.get(event)
    if generic is not None:
        return PerfEventSpec(event, *generic)

    if shutil.which(perf_binary) is None:
        raise FileNotFoundError(f"perf binary not found: {perf_binary}")
    result = subprocess.run(
        [perf_binary, "stat", "-vv", "--no-scale", "-e", event, "--", "true"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "LC_ALL": "C"},
        check=False,
    )
    if result.returncode != 0:
        detail = next((line for line in result.stderr.splitlines() if line.strip()), "")
        raise ValueError(f"Unable to resolve perf event {event!r}: {detail or result.returncode}")

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
        raise ValueError(f"perf did not expose type/config for event {event!r}.")
    return PerfEventSpec(
        event,
        values["type"],
        values["config"],
        values.get("config1", 0),
        values.get("config2", 0),
        values.get("config3", 0),
    )


def _open_perf_counter(
    spec: PerfEventSpec,
    tid: int,
    *,
    group_fd: int,
    group_leader: bool,
    inherit: bool,
) -> int:
    attr = PerfEventAttr()
    attr.type = spec.type
    attr.size = ctypes.sizeof(PerfEventAttr)
    attr.config = spec.config
    attr.config1 = spec.config1
    attr.config2 = spec.config2
    attr.config3 = spec.config3
    attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING
    # disabled (leader), inherit, exclude_kernel, exclude_hv, exclude_guest.
    # exclude_user remains clear, so only user-space execution is counted.
    attr.flags = (
        (1 if group_leader else 0)
        | ((1 << 1) if inherit else 0)
        | (1 << 5)
        | (1 << 6)
        | (1 << 20)
    )
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.syscall(
        _perf_event_open_syscall_number(),
        ctypes.byref(attr),
        tid,
        -1,
        group_fd,
        PERF_FLAG_FD_CLOEXEC,
    )
    if fd < 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number), f"event={spec.name} tid={tid}")
    return int(fd)


def _open_counter_groups(
    specs: Sequence[PerfEventSpec], pid: int, *, inherit: bool
) -> List[CounterGroup]:
    task_dir = Path(f"/proc/{pid}/task")
    tids = sorted(int(path.name) for path in task_dir.iterdir() if path.name.isdigit())
    if pid not in tids:
        raise RuntimeError(f"Process leader {pid} is unavailable for perf measurement.")

    groups: List[CounterGroup] = []
    try:
        for tid in tids:
            counters: List[OpenPerfCounter] = []
            leader_fd = -1
            for index, spec in enumerate(specs):
                try:
                    fd = _open_perf_counter(
                        spec,
                        tid,
                        group_fd=leader_fd,
                        group_leader=index == 0,
                        inherit=inherit,
                    )
                except OSError as exc:
                    if tid != pid and exc.errno in {errno.ENOENT, errno.ESRCH}:
                        for counter in counters:
                            os.close(counter.fd)
                        counters = []
                        break
                    raise
                if index == 0:
                    leader_fd = fd
                counters.append(OpenPerfCounter(spec.name, tid, fd))
            if counters:
                groups.append(CounterGroup(tid, leader_fd, counters))
    except BaseException:
        _close_counter_groups(groups)
        raise
    return groups


def _close_counter_groups(groups: Iterable[CounterGroup]) -> None:
    for group in groups:
        for counter in group.counters:
            try:
                os.close(counter.fd)
            except OSError:
                pass


def validate_perf_events(events: Sequence[str], perf_binary: str = "perf") -> None:
    parsed = parse_perf_events(",".join(events))
    specs = [resolve_perf_event_spec(event, perf_binary) for event in parsed]
    groups = _open_counter_groups(specs, os.getpid(), inherit=False)
    _close_counter_groups(groups)


def _first_tensor(value: Any) -> Optional[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _tensor_shape(value: Any) -> str:
    tensor = _first_tensor(value)
    return "" if tensor is None else json.dumps(list(tensor.shape), separators=(",", ":"))


def _tensor_dtype(value: Any) -> str:
    tensor = _first_tensor(value)
    return "" if tensor is None else str(tensor.dtype)


def _event_columns(events: Sequence[str]) -> List[str]:
    columns: List[str] = []
    for event in events:
        base = f"perf_{sanitize_event_name(event)}"
        columns.extend(
            [
                f"{base}_raw",
                base,
                f"{base}_time_enabled_ns",
                f"{base}_time_running_ns",
                f"{base}_running_pct",
            ]
        )
    return columns


BASE_COLUMNS = [
    "timestamp",
    "timestamp_unix",
    "end_timestamp",
    "end_timestamp_unix",
    *CONDITION_COLUMNS,
    "round",
    "epoch",
    "batch_idx",
    "phase",
    "layer_index",
    "layer_name",
    "layer_type",
    "execution_order_index",
    "invocation_index",
    "input_shape",
    "output_shape",
    "input_dtype",
    "output_dtype",
    "duration_ns",
    "duration_ms",
    "perf_pid",
    "perf_scope",
    "perf_measurement_mode",
    "perf_events",
    "perf_target_thread_count",
    "torch_num_threads",
    "torch_num_interop_threads",
    "perf_status",
    "perf_error",
]


class LayerPerfLogger:
    """Measure every leaf module forward and autograd-node backward operation."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        path: os.PathLike[str] | str,
        condition: Dict[str, Any],
        events: Optional[Sequence[str]] = None,
        pid: Optional[int] = None,
        perf_binary: str = "perf",
        strict_batch_layers: bool = True,
    ) -> None:
        self.model = model
        self.path = Path(path)
        self.condition = dict(condition)
        self.events = parse_perf_events(",".join(events or DEFAULT_PERF_EVENTS))
        self.pid = int(pid or os.getpid())
        self.perf_binary = perf_binary
        self.strict_batch_layers = strict_batch_layers
        self.columns = [*BASE_COLUMNS, *_event_columns(self.events)]
        self.leaf_modules: List[Tuple[str, torch.nn.Module]] = [
            (name, module)
            for name, module in model.named_modules()
            if name and not any(module.children())
        ]
        if not self.leaf_modules:
            raise ValueError("The model does not contain any named leaf modules.")
        self._layer_by_module = {
            module: (index, name, type(module).__name__)
            for index, (name, module) in enumerate(self.leaf_modules)
        }
        self._specs: List[PerfEventSpec] = []
        self._groups: List[CounterGroup] = []
        self._module_handles: List[Any] = []
        self._node_handles: List[Any] = []
        self._file: Any = None
        self._writer: Optional[csv.DictWriter] = None
        self._batch_active = False
        self._batch_context: Dict[str, Any] = {}
        self._pending_rows: List[Dict[str, Any]] = []
        self._phase_order = {"forward": 0, "backward": 0}
        self._invocations: Dict[int, int] = {}
        self._active_region: Optional[Dict[str, Any]] = None
        self._time_baseline: Dict[int, Tuple[int, int]] = {}
        self._lock = threading.RLock()

    @property
    def expected_rows_per_batch(self) -> int:
        return 2 * len(self.leaf_modules)

    def __enter__(self) -> "LayerPerfLogger":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()

    def start(self) -> None:
        if self._file is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._specs = [resolve_perf_event_spec(event, self.perf_binary) for event in self.events]
        try:
            self._groups = _open_counter_groups(self._specs, self.pid, inherit=True)
            self._file = self.path.open("w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=self.columns, extrasaction="ignore")
            self._writer.writeheader()
            self._file.flush()
            for module in self._layer_by_module:
                self._module_handles.append(module.register_forward_pre_hook(self._forward_pre_hook))
                self._module_handles.append(module.register_forward_hook(self._forward_hook))
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        with self._lock:
            self.abort_batch()
            for handle in self._module_handles:
                handle.remove()
            self._module_handles.clear()
            for handle in self._node_handles:
                handle.remove()
            self._node_handles.clear()
            _close_counter_groups(self._groups)
            self._groups.clear()
            if self._file is not None:
                self._file.flush()
                self._file.close()
                self._file = None
                self._writer = None

    def begin_batch(self, *, epoch: int, batch_idx: int, round_id: Any = 0) -> None:
        with self._lock:
            if self._batch_active:
                raise RuntimeError("A layer-perf batch is already active.")
            self._batch_context = {"round": round_id, "epoch": epoch, "batch_idx": batch_idx}
            self._pending_rows = []
            self._phase_order = {"forward": 0, "backward": 0}
            self._invocations = {}
            self._batch_active = True

    def flush_batch(self) -> None:
        with self._lock:
            if not self._batch_active:
                raise RuntimeError("No layer-perf batch is active.")
            if self._active_region is not None:
                raise RuntimeError("Cannot flush while a layer PMU region is active.")
            counts = {
                phase: sum(row["phase"] == phase for row in self._pending_rows)
                for phase in ("forward", "backward")
            }
            expected = len(self.leaf_modules)
            if self.strict_batch_layers and counts != {"forward": expected, "backward": expected}:
                raise RuntimeError(
                    "Incomplete layer PMU batch: "
                    f"expected {expected} forward and backward rows, got {counts}."
                )
            assert self._writer is not None and self._file is not None
            self._writer.writerows(self._pending_rows)
            self._file.flush()
            self._clear_batch()

    def abort_batch(self) -> None:
        with self._lock:
            if self._active_region is not None:
                self._disable_groups()
                self._active_region = None
            self._clear_batch()

    def _clear_batch(self) -> None:
        for handle in self._node_handles:
            handle.remove()
        self._node_handles.clear()
        self._pending_rows = []
        self._batch_context = {}
        self._batch_active = False

    def _forward_pre_hook(self, module: torch.nn.Module, inputs: Tuple[Any, ...]) -> None:
        if not self._batch_active:
            return
        layer_index, layer_name, layer_type = self._layer_by_module[module]
        invocation_index = self._invocations.get(layer_index, 0)
        self._invocations[layer_index] = invocation_index + 1
        self._begin_region(
            {
                **self._batch_context,
                "phase": "forward",
                "layer_index": layer_index,
                "layer_name": layer_name,
                "layer_type": layer_type,
                "execution_order_index": self._phase_order["forward"],
                "invocation_index": invocation_index,
                "input_shape": _tensor_shape(inputs),
                "input_dtype": _tensor_dtype(inputs),
            }
        )
        self._phase_order["forward"] += 1

    def _forward_hook(
        self, module: torch.nn.Module, inputs: Tuple[Any, ...], output: Any
    ) -> None:
        if not self._batch_active:
            return
        context = dict(self._active_region or {})
        context["output_shape"] = _tensor_shape(output)
        context["output_dtype"] = _tensor_dtype(output)
        self._pending_rows.append(self._end_region(context))

        output_tensor = _first_tensor(output)
        node = None if output_tensor is None else output_tensor.grad_fn
        if node is None:
            return

        backward_context = {
            **self._batch_context,
            "phase": "backward",
            "layer_index": context["layer_index"],
            "layer_name": context["layer_name"],
            "layer_type": context["layer_type"],
            "invocation_index": context["invocation_index"],
            "input_shape": context.get("input_shape", ""),
            "output_shape": context.get("output_shape", ""),
            "input_dtype": context.get("input_dtype", ""),
            "output_dtype": context.get("output_dtype", ""),
        }

        def backward_pre_hook(grad_outputs: Tuple[Any, ...]) -> Tuple[Any, ...]:
            local_context = dict(backward_context)
            local_context["execution_order_index"] = self._phase_order["backward"]
            self._phase_order["backward"] += 1
            self._begin_region(local_context)
            return grad_outputs

        def backward_hook(
            grad_inputs: Tuple[Any, ...], grad_outputs: Tuple[Any, ...]
        ) -> Tuple[Any, ...]:
            self._pending_rows.append(self._end_region(dict(self._active_region or {})))
            return grad_inputs

        self._node_handles.append(node.register_prehook(backward_pre_hook))
        self._node_handles.append(node.register_hook(backward_hook))

    def _begin_region(self, context: Dict[str, Any]) -> None:
        with self._lock:
            if self._active_region is not None:
                raise RuntimeError(
                    "Nested layer PMU regions are not supported: "
                    f"{self._active_region.get('layer_name')} -> {context.get('layer_name')}"
                )
            self._reset_enable_groups()
            context["_start_wall"] = datetime.now()
            context["_start_ns"] = time.perf_counter_ns()
            self._active_region = context

    def _end_region(self, context: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if self._active_region is None:
                raise RuntimeError("A layer PMU region ended without a matching start.")
            end_ns = time.perf_counter_ns()
            end_wall = datetime.now()
            self._disable_groups()
            event_values = self._read_groups()
            start_wall = self._active_region["_start_wall"]
            start_ns = self._active_region["_start_ns"]
            row: Dict[str, Any] = {
                "timestamp": start_wall.isoformat(timespec="microseconds"),
                "timestamp_unix": start_wall.timestamp(),
                "end_timestamp": end_wall.isoformat(timespec="microseconds"),
                "end_timestamp_unix": end_wall.timestamp(),
                "duration_ns": end_ns - start_ns,
                "duration_ms": (end_ns - start_ns) / 1_000_000.0,
                "perf_pid": self.pid,
                "perf_scope": "user_all_existing_threads_plus_inherited_new_threads",
                "perf_measurement_mode": "leaf_layer_boundary",
                "perf_events": ",".join(self.events),
                "perf_target_thread_count": len(self._groups),
                "torch_num_threads": torch.get_num_threads(),
                "torch_num_interop_threads": torch.get_num_interop_threads(),
                "perf_status": "ok",
                "perf_error": "",
            }
            row.update(self.condition)
            row.update({key: value for key, value in context.items() if not key.startswith("_")})
            row.update(event_values)
            if not row.get("host"):
                try:
                    row["host"] = socket.gethostbyname(socket.gethostname())
                except OSError:
                    row["host"] = socket.gethostname()
            for column in self.columns:
                row.setdefault(column, "")
            self._active_region = None
            return row

    def _reset_enable_groups(self) -> None:
        for group in self._groups:
            fcntl.ioctl(group.leader_fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP)
        self._time_baseline = {}
        for group in self._groups:
            for counter in group.counters:
                payload = os.read(counter.fd, 24)
                if len(payload) != 24:
                    raise OSError(
                        f"Short baseline perf read for event={counter.event} tid={counter.tid}."
                    )
                _, time_enabled, time_running = struct.unpack("QQQ", payload)
                self._time_baseline[counter.fd] = (time_enabled, time_running)
        for group in self._groups:
            fcntl.ioctl(group.leader_fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP)

    def _disable_groups(self) -> None:
        for group in self._groups:
            try:
                fcntl.ioctl(group.leader_fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP)
            except OSError as exc:
                if exc.errno not in {errno.ENOENT, errno.ESRCH}:
                    raise

    def _read_groups(self) -> Dict[str, Any]:
        raw = {event: 0 for event in self.events}
        scaled = {event: 0.0 for event in self.events}
        enabled = {event: 0 for event in self.events}
        running = {event: 0 for event in self.events}
        for group in self._groups:
            for counter in group.counters:
                payload = os.read(counter.fd, 24)
                if len(payload) != 24:
                    raise OSError(
                        f"Short perf read for event={counter.event} tid={counter.tid}."
                    )
                value, total_enabled, total_running = struct.unpack("QQQ", payload)
                baseline_enabled, baseline_running = self._time_baseline.get(counter.fd, (0, 0))
                time_enabled = max(0, total_enabled - baseline_enabled)
                time_running = max(0, total_running - baseline_running)
                raw[counter.event] += value
                enabled[counter.event] += time_enabled
                running[counter.event] += time_running
                if time_running > 0:
                    scaled[counter.event] += value * time_enabled / time_running

        values: Dict[str, Any] = {}
        for event in self.events:
            base = f"perf_{sanitize_event_name(event)}"
            values[f"{base}_raw"] = raw[event]
            values[base] = scaled[event] if running[event] > 0 else ""
            values[f"{base}_time_enabled_ns"] = enabled[event]
            values[f"{base}_time_running_ns"] = running[event]
            values[f"{base}_running_pct"] = (
                100.0 * running[event] / enabled[event] if enabled[event] else ""
            )
        return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate direct layer perf events.")
    parser.add_argument("--check-events", default=",".join(DEFAULT_PERF_EVENTS))
    parser.add_argument("--perf-binary", default="perf")
    args = parser.parse_args()
    events = parse_perf_events(args.check_events)
    validate_perf_events(events, args.perf_binary)
    print(f"perf events ok ({len(events)}): {','.join(events)}")


if __name__ == "__main__":
    main()
