from __future__ import annotations

import csv
import gc
import os
import time
import traceback
from pathlib import Path
from typing import Any, Mapping


def _configure_single_thread_runtime(cpu_id: int) -> None:
    # These variables must be fixed before importing torch in a spawned worker.
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.sched_setaffinity(0, {cpu_id})


def _standardized_random(torch: Any, shape: tuple[int, ...], generator: Any) -> Any:
    tensor = torch.randn(shape, generator=generator, dtype=torch.float32)
    tensor.sub_(tensor.mean())
    tensor.div_(tensor.std(unbiased=False) + 1e-12)
    return tensor


def _make_aligned_gradient_bank(torch: Any, gradients: list[Any], alignment_bytes: int = 4096) -> Any:
    stacked = torch.stack(gradients)
    element_size = stacked.element_size()
    padding_elements = alignment_bytes // element_size
    backing = torch.empty(stacked.numel() + padding_elements, dtype=stacked.dtype)
    offset_elements = ((-backing.data_ptr()) % alignment_bytes) // element_size
    aligned = backing[offset_elements : offset_elements + stacked.numel()].view_as(stacked)
    aligned.copy_(stacked)
    if aligned.data_ptr() % alignment_bytes != 0:
        raise RuntimeError("Could not align the gradient bank")
    return aligned


def _wait_until_ns(target_ns: int, spin_threshold_us: float) -> None:
    spin_ns = max(0, int(spin_threshold_us * 1_000.0))
    while True:
        remaining_ns = target_ns - time.perf_counter_ns()
        if remaining_ns <= 0:
            return
        if remaining_ns > spin_ns:
            time.sleep((remaining_ns - spin_ns) / 1_000_000_000.0)
        else:
            # A short final spin avoids scheduler-sized errors in sub-ms offsets.
            while time.perf_counter_ns() < target_ns:
                pass
            return


def _write_timings(path: Path, metadata: Mapping[str, Any], starts: list[int], ends: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "replicate",
        "offset_delta_ms",
        "event_pass",
        "worker_id",
        "cpu_id",
        "step",
        "planned_first_start_ns",
        "start_ns",
        "end_ns",
        "duration_ns",
        "duration_ms",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for step, (start_ns, end_ns) in enumerate(zip(starts, ends)):
            writer.writerow(
                {
                    **metadata,
                    "step": step,
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                    "duration_ns": end_ns - start_ns,
                    "duration_ms": (end_ns - start_ns) / 1_000_000.0,
                }
            )


def worker_process_main(
    config: Mapping[str, Any],
    ready_queue: Any,
    release_event: Any,
    release_ns: Any,
    measurement_closed_event: Any,
) -> None:
    """Run one fixed Conv2D backward stream on one CPU.

    The worker warms up before reporting ready. It applies its phase offset only
    once, before the first measured backward. No sleep is inserted between
    measured iterations.
    """
    worker_id = int(config["worker_id"])
    cpu_id = int(config["cpu_id"])
    starts: list[int] = []
    ends: list[int] = []

    try:
        _configure_single_thread_runtime(cpu_id)
        import torch
        import torch.nn.functional as functional

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        model_generator = torch.Generator().manual_seed(int(config["model_seed"]))
        gradient_generator = torch.Generator().manual_seed(int(config["gradient_seed"]))
        batch_size = int(config["batch_size"])
        channels = int(config["channels"])
        spatial_size = int(config["spatial_size"])

        x = torch.randn(
            batch_size,
            channels,
            spatial_size,
            spatial_size,
            generator=model_generator,
            dtype=torch.float32,
            requires_grad=True,
        )
        weight = torch.randn(
            channels,
            channels,
            3,
            3,
            generator=model_generator,
            dtype=torch.float32,
        )
        weight.mul_(0.05)
        weight.requires_grad_(True)
        output = functional.conv2d(x, weight, bias=None, stride=1, padding=1)

        direction = _standardized_random(torch, tuple(output.shape), gradient_generator)
        direction.mul_(float(config["gradient_scale"]))
        bank_size = int(config["gradient_bank_size"])
        gradient_bank = _make_aligned_gradient_bank(
            torch, [direction.clone() for _ in range(bank_size)]
        )

        for step in range(int(config["warmup"])):
            grad_x, grad_weight = torch.autograd.grad(
                outputs=output,
                inputs=(x, weight),
                grad_outputs=gradient_bank[step % bank_size],
                retain_graph=True,
                create_graph=False,
            )
            del grad_x, grad_weight

        ready_queue.put(
            {
                "kind": "ready",
                "worker_id": worker_id,
                "cpu_id": cpu_id,
                "pid": os.getpid(),
                "affinity": sorted(os.sched_getaffinity(0)),
                "torch_num_threads": torch.get_num_threads(),
                "torch_num_interop_threads": torch.get_num_interop_threads(),
                "gradient_std": float(gradient_bank.std(unbiased=False).item()),
                "gradient_bank_base_mod4096": gradient_bank.data_ptr() % 4096,
            }
        )

        if not release_event.wait(timeout=float(config["worker_timeout_sec"])):
            raise TimeoutError("Timed out waiting for the common release event")

        planned_first_start_ns = int(release_ns.value) + int(
            float(config["start_offset_ms"]) * 1_000_000.0
        )
        _wait_until_ns(planned_first_start_ns, float(config["spin_threshold_us"]))

        gc_was_enabled = gc.isenabled()
        gc.disable()
        measured_process_start_ns = time.process_time_ns()
        try:
            for step in range(int(config["steps"])):
                grad_output = gradient_bank[step % bank_size]
                start_ns = time.perf_counter_ns()
                grad_x, grad_weight = torch.autograd.grad(
                    outputs=output,
                    inputs=(x, weight),
                    grad_outputs=grad_output,
                    retain_graph=True,
                    create_graph=False,
                )
                end_ns = time.perf_counter_ns()
                starts.append(start_ns)
                ends.append(end_ns)
                del grad_x, grad_weight
        finally:
            measured_process_end_ns = time.process_time_ns()
            if gc_was_enabled:
                gc.enable()

        ready_queue.put(
            {
                "kind": "measured",
                "worker_id": worker_id,
                "cpu_id": cpu_id,
                "pid": os.getpid(),
                "planned_first_start_ns": planned_first_start_ns,
                "first_start_ns": starts[0],
                "last_end_ns": ends[-1],
                "process_cpu_time_ns": measured_process_end_ns - measured_process_start_ns,
                "duration_mean_ms": sum(e - s for s, e in zip(starts, ends))
                / len(starts)
                / 1_000_000.0,
            }
        )

        # Keep output serialization outside the perf-enabled region.
        if not measurement_closed_event.wait(timeout=float(config["worker_timeout_sec"])):
            raise TimeoutError("Timed out waiting for measurement closure")

        metadata = {
            "replicate": config["replicate"],
            "offset_delta_ms": config["offset_delta_ms"],
            "event_pass": config["event_pass"],
            "worker_id": worker_id,
            "cpu_id": cpu_id,
            "planned_first_start_ns": planned_first_start_ns,
        }
        _write_timings(Path(str(config["timing_path"])), metadata, starts, ends)
    except BaseException as exc:
        ready_queue.put(
            {
                "kind": "error",
                "worker_id": worker_id,
                "cpu_id": cpu_id,
                "pid": os.getpid(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        measurement_closed_event.set()
        raise
