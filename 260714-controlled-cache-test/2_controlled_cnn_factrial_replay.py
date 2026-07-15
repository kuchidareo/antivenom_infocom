from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * q)
    return ordered[index]


class PerfControl:
    """Synchronize the measured loop with ``perf stat --control=fifo:...``."""

    def __init__(self, control_fifo: str | None = None, ack_fifo: str | None = None):
        if bool(control_fifo) != bool(ack_fifo):
            raise ValueError("Both perf control and acknowledgment FIFOs are required")
        self.control_fifo = control_fifo
        self.ack_fifo = ack_fifo
        self._control = None
        self._ack = None
        self.enabled = False

    def __enter__(self):
        if self.control_fifo is not None:
            self._control = open(self.control_fifo, "w", buffering=1)
            self._ack = open(self.ack_fifo, "r", buffering=1)
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            if self.enabled:
                self.disable()
        finally:
            if self._control is not None:
                self._control.close()
            if self._ack is not None:
                self._ack.close()

    def _command(self, command: str) -> None:
        if self._control is None or self._ack is None:
            return
        self._control.write(f"{command}\n")
        acknowledgment = self._ack.readline().replace("\x00", "").strip()
        if acknowledgment != "ack":
            raise RuntimeError(
                f"perf did not acknowledge {command!r}; received {acknowledgment!r}"
            )

    def enable(self) -> None:
        self._command("enable")
        self.enabled = True

    def disable(self) -> None:
        try:
            self._command("disable")
        finally:
            self.enabled = False


def make_aligned_empty(
    shape: Sequence[int],
    dtype: torch.dtype = torch.float32,
    alignment_bytes: int = 4096,
) -> torch.Tensor:
    element_size = torch.empty((), dtype=dtype).element_size()
    if alignment_bytes % element_size != 0:
        raise ValueError("alignment_bytes must be divisible by the tensor element size")

    element_count = math.prod(shape)
    padding_elements = alignment_bytes // element_size
    backing = torch.empty(element_count + padding_elements, dtype=dtype)
    offset_bytes = (-backing.data_ptr()) % alignment_bytes
    offset_elements = offset_bytes // element_size
    aligned = backing[offset_elements : offset_elements + element_count].view(*shape)

    if aligned.data_ptr() % alignment_bytes != 0:
        raise RuntimeError("Could not align a replay bank")
    return aligned


def load_payload(path: Path) -> Dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a dictionary payload in {path}")
    return payload


def metadata_tag(payload: Mapping[str, Any], path: Path) -> Dict[str, Any]:
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    keys = (
        "payload_version",
        "architecture",
        "condition",
        "seed",
        "epoch",
        "layer",
        "pair_count",
    )
    return {"path": str(path)} | {key: metadata.get(key) for key in keys}


def conv_parameters(payload: Mapping[str, Any], path: Path) -> Dict[str, Any]:
    metadata = payload.get("metadata", {})
    conv = metadata.get("conv") if isinstance(metadata, Mapping) else None
    if not isinstance(conv, Mapping):
        raise ValueError(f"Missing Conv metadata in {path}")

    required = ("stride", "padding", "dilation", "groups")
    missing = [name for name in required if name not in conv]
    if missing:
        raise ValueError(f"Missing Conv metadata {missing} in {path}")

    def pair(value: Any) -> Tuple[int, int]:
        if isinstance(value, int):
            return value, value
        values = tuple(int(item) for item in value)
        if len(values) != 2:
            raise ValueError(f"Expected a pair in Conv metadata, received {value!r}")
        return values

    return {
        "stride": pair(conv["stride"]),
        "padding": pair(conv["padding"]),
        "dilation": pair(conv["dilation"]),
        "groups": int(conv["groups"]),
        "in_channels": int(conv.get("in_channels", -1)),
        "out_channels": int(conv.get("out_channels", -1)),
        "kernel_size": pair(conv.get("kernel_size", (0, 0))),
    }


def require_tensor(payload: Mapping[str, Any], key: str, path: Path) -> torch.Tensor:
    value = payload.get(key)
    if not isinstance(value, torch.Tensor):
        if key == "activation_bank":
            raise ValueError(
                f"{path} has no activation_bank. Rerun the updated Kaggle capture "
                "to create payload_version=2."
            )
        raise ValueError(f"{path} has no Tensor field {key!r}")
    if value.ndim < 2:
        raise ValueError(f"{key} in {path} must include bank and tensor dimensions")
    return value.detach().to(device="cpu", dtype=torch.float32).contiguous()


def materialize_bank(source: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    bank = make_aligned_empty((len(indices), *source.shape[1:]), dtype=torch.float32)
    for destination, source_index in enumerate(indices):
        bank[destination].copy_(source[source_index])
    return bank


def bank_layout(bank: torch.Tensor) -> Dict[str, Any]:
    count = len(bank)
    addresses = [bank[index].data_ptr() for index in range(count)]
    buffer_bytes = bank[0].numel() * bank.element_size()
    offsets = [address - addresses[0] for address in addresses]
    expected_offsets = [index * buffer_bytes for index in range(count)]

    if not bank.is_contiguous():
        raise RuntimeError("Replay banks must be contiguous")
    if len(set(addresses)) != count:
        raise RuntimeError("Every replay entry must have a distinct address")
    if offsets != expected_offsets:
        raise RuntimeError("Replay entries do not have sequential offsets")

    return {
        "shape": list(bank.shape),
        "stride": list(bank.stride()),
        "contiguous": bank.is_contiguous(),
        "buffer_count": count,
        "unique_buffer_addresses": len(set(addresses)),
        "buffer_bytes": buffer_bytes,
        "buffer_offsets_bytes": offsets,
        "base_ptr_mod64": addresses[0] % 64,
        "base_ptr_mod4096": addresses[0] % 4096,
    }


def bank_statistics(bank: torch.Tensor) -> Dict[str, float]:
    flat = bank.reshape(len(bank), -1)
    if len(flat) > 1:
        adjacent_cosine = F.cosine_similarity(
            flat[:-1], flat[1:], dim=1, eps=1e-12
        ).mean().item()
        adjacent_relative_delta = (
            (flat[1:] - flat[:-1]).norm(dim=1)
            / flat[:-1].norm(dim=1).clamp_min(1e-12)
        ).mean().item()
    else:
        adjacent_cosine = 1.0
        adjacent_relative_delta = 0.0

    norms = torch.linalg.vector_norm(flat, dim=1)
    return {
        "mean": flat.mean().item(),
        "std": flat.std(unbiased=False).item(),
        "rms": flat.square().mean().sqrt().item(),
        "l2_mean": norms.mean().item(),
        "l2_min": norms.min().item(),
        "l2_max": norms.max().item(),
        "adjacent_cosine": adjacent_cosine,
        "adjacent_relative_delta": adjacent_relative_delta,
    }


def build_indices(
    mode: str,
    count: int,
    order: str,
    seed: int,
) -> list[int]:
    if mode == "held":
        if order != "matched":
            raise ValueError("Non-matched order is only meaningful for stream mode")
        return [0] * count

    indices = list(range(count))
    if order == "matched":
        return indices
    if order == "cyclic-shift":
        return indices[1:] + indices[:1]
    if order == "permuted":
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(count, generator=generator).tolist()
        if count > 1 and permutation == indices:
            permutation = permutation[1:] + permutation[:1]
        return permutation
    raise ValueError(order)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Factorial replay of real SimpleCNN Conv input activations and "
            "Conv output gradients captured from paired training probes."
        )
    )
    parser.add_argument("--activation-source", type=Path, required=True)
    parser.add_argument(
        "--gradient-source",
        type=Path,
        help="Defaults to --activation-source.",
    )
    parser.add_argument(
        "--weight-source",
        type=Path,
        help="Defaults to --activation-source.",
    )
    parser.add_argument(
        "--activation-mode",
        choices=("held", "stream"),
        required=True,
        help="held repeats entry 0; stream replays all captured activations.",
    )
    parser.add_argument(
        "--gradient-mode",
        choices=("held", "stream"),
        required=True,
        help="held repeats entry 0; stream replays all captured gradients.",
    )
    parser.add_argument(
        "--gradient-order",
        choices=("matched", "cyclic-shift", "permuted"),
        default="matched",
        help="Reorder only the gradient stream to test activation-gradient pairing.",
    )
    parser.add_argument(
        "--gradient-target",
        choices=("both", "input", "weight"),
        default="both",
        help="Which Conv backward result to request from autograd.",
    )
    parser.add_argument("--bank-size", type=int)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--perf-control-fifo", help=argparse.SUPPRESS)
    parser.add_argument("--perf-ack-fifo", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.threads <= 0:
        parser.error("--threads must be positive")
    if args.bank_size is not None and args.bank_size <= 0:
        parser.error("--bank-size must be positive")

    # Configure the CPU runtime before loading or copying any large tensors.
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    gradient_path = args.gradient_source or args.activation_source
    weight_path = args.weight_source or args.activation_source
    paths = {
        "activation": args.activation_source.resolve(),
        "gradient": gradient_path.resolve(),
        "weight": weight_path.resolve(),
    }

    payload_cache: Dict[Path, Dict[str, Any]] = {}
    for path in sorted(set(paths.values()), key=str):
        if not path.exists():
            parser.error(f"Capture payload does not exist: {path}")
        payload_cache[path] = load_payload(path)

    activation_payload = payload_cache[paths["activation"]]
    gradient_payload = payload_cache[paths["gradient"]]
    weight_payload = payload_cache[paths["weight"]]
    activation_source = require_tensor(
        activation_payload, "activation_bank", paths["activation"]
    )
    gradient_source = require_tensor(
        gradient_payload, "gradient_bank", paths["gradient"]
    )
    if activation_source.ndim != 5 or gradient_source.ndim != 5:
        parser.error(
            "This Conv2D replay expects banks shaped "
            "[bank, batch, channels, height, width]: "
            f"activation={tuple(activation_source.shape)}, "
            f"gradient={tuple(gradient_source.shape)}"
        )
    weight_value = weight_payload.get("weight")
    if not isinstance(weight_value, torch.Tensor):
        parser.error(
            f"{paths['weight']} has no weight Tensor. Rerun payload_version=2 capture."
        )
    weight_value = weight_value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if weight_value.ndim != 4:
        parser.error(f"Conv2D weight must be rank 4, received {tuple(weight_value.shape)}")
    bias_value = weight_payload.get("bias")
    if bias_value is not None and not isinstance(bias_value, torch.Tensor):
        parser.error(f"{paths['weight']} has a non-Tensor bias field")
    if isinstance(bias_value, torch.Tensor):
        bias_value = (
            bias_value.detach().to(device="cpu", dtype=torch.float32).contiguous()
        )
        if bias_value.ndim != 1 or bias_value.numel() != weight_value.shape[0]:
            parser.error(
                "Conv2D bias is incompatible with weight: "
                f"bias={tuple(bias_value.shape)}, weight={tuple(weight_value.shape)}"
            )

    activation_conv = conv_parameters(activation_payload, paths["activation"])
    gradient_conv = conv_parameters(gradient_payload, paths["gradient"])
    weight_conv = conv_parameters(weight_payload, paths["weight"])
    if activation_conv != gradient_conv or activation_conv != weight_conv:
        parser.error(
            "Activation, gradient, and weight payloads describe different Conv shapes: "
            f"activation={activation_conv}, gradient={gradient_conv}, weight={weight_conv}"
        )

    available_count = min(len(activation_source), len(gradient_source))
    bank_size = args.bank_size or available_count
    if bank_size > available_count:
        parser.error(
            f"Requested bank size {bank_size}, but only {available_count} paired entries exist"
        )

    activation_indices = build_indices(
        args.activation_mode,
        bank_size,
        "matched",
        args.seed,
    )
    gradient_indices = build_indices(
        args.gradient_mode,
        bank_size,
        args.gradient_order,
        args.seed,
    )

    activation_bank = materialize_bank(activation_source, activation_indices)
    gradient_bank = materialize_bank(gradient_source, gradient_indices)
    weight = weight_value.clone().requires_grad_(True)
    bias = (
        bias_value.clone().requires_grad_(True)
        if isinstance(bias_value, torch.Tensor) else None
    )

    if activation_bank.shape[2] != weight.shape[1] * activation_conv["groups"]:
        parser.error(
            "Activation channels and Conv weight are incompatible: "
            f"activation={tuple(activation_bank.shape)}, weight={tuple(weight.shape)}"
        )

    # The source payloads are intentionally removed before graph construction
    # so all regimes retain the same live Tensor set during measurement.
    source_tags = {
        name: metadata_tag(payload_cache[path], path)
        for name, path in paths.items()
    }
    del activation_source, gradient_source, weight_value, bias_value
    del activation_payload, gradient_payload, weight_payload, payload_cache
    gc.collect()

    activation_views = tuple(
        activation_bank[index].detach().requires_grad_(True)
        for index in range(bank_size)
    )
    outputs = tuple(
        F.conv2d(
            activation,
            weight,
            bias=bias,
            stride=activation_conv["stride"],
            padding=activation_conv["padding"],
            dilation=activation_conv["dilation"],
            groups=activation_conv["groups"],
        )
        for activation in activation_views
    )

    for index, output in enumerate(outputs):
        if tuple(output.shape) != tuple(gradient_bank[index].shape):
            parser.error(
                "Captured gradient shape does not match replay Conv output: "
                f"output={tuple(output.shape)}, gradient={tuple(gradient_bank[index].shape)}"
            )

    if args.gradient_target == "both":
        input_sets = tuple(
            (
                (activation_views[index], weight, bias)
                if bias is not None else (activation_views[index], weight)
            )
            for index in range(bank_size)
        )
    elif args.gradient_target == "input":
        input_sets = tuple((activation_views[index],) for index in range(bank_size))
    else:
        input_sets = tuple((weight,) for _ in range(bank_size))

    result = None
    for step in range(args.warmup):
        index = step % bank_size
        result = torch.autograd.grad(
            outputs=outputs[index],
            inputs=input_sets[index],
            grad_outputs=gradient_bank[index],
            retain_graph=True,
            create_graph=False,
        )
    del result

    gc.disable()
    backward_times_ms: list[float] = []
    measurement_start_ns = None
    measurement_end_ns = None

    try:
        with PerfControl(args.perf_control_fifo, args.perf_ack_fifo) as perf_control:
            perf_control.enable()
            measurement_start_ns = time.perf_counter_ns()

            for step in range(args.steps):
                index = step % bank_size
                start_ns = time.perf_counter_ns()

                result = torch.autograd.grad(
                    outputs=outputs[index],
                    inputs=input_sets[index],
                    grad_outputs=gradient_bank[index],
                    retain_graph=True,
                    create_graph=False,
                )

                end_ns = time.perf_counter_ns()
                backward_times_ms.append((end_ns - start_ns) / 1_000_000)
                del result

            measurement_end_ns = time.perf_counter_ns()
            perf_control.disable()
    finally:
        gc.enable()

    if measurement_start_ns is None or measurement_end_ns is None:
        raise RuntimeError("The measured loop did not complete")

    # These full-bank scans intentionally run after perf has been disabled so
    # summary construction cannot precondition the measured cache state.
    activation_layout = bank_layout(activation_bank)
    gradient_layout = bank_layout(gradient_bank)
    activation_stats = bank_statistics(activation_bank)
    gradient_stats = bank_statistics(gradient_bank)

    summary = {
        "framework": "pytorch",
        "pytorch_version": torch.__version__,
        "experiment": "real-conv-activation-gradient-factorial-replay",
        "activation_mode": args.activation_mode,
        "gradient_mode": args.gradient_mode,
        "gradient_order": args.gradient_order,
        "gradient_target": args.gradient_target,
        "steps": args.steps,
        "warmup": args.warmup,
        "threads": args.threads,
        "seed": args.seed,
        "bank_size": bank_size,
        "activation_indices": activation_indices,
        "gradient_indices": gradient_indices,
        "sources": source_tags,
        "conv": activation_conv,
        "weight_shape": list(weight.shape),
        "weight_mean": weight.detach().mean().item(),
        "weight_std": weight.detach().std(unbiased=False).item(),
        "weight_rms": weight.detach().square().mean().sqrt().item(),
        "bias_included_in_default_both_target": bias is not None,
        "bias_shape": list(bias.shape) if bias is not None else None,
        "activation_bank": activation_layout | activation_stats,
        "gradient_bank": gradient_layout | gradient_stats,
        "backward_mean_ms": statistics.mean(backward_times_ms),
        "backward_median_ms": statistics.median(backward_times_ms),
        "backward_p95_ms": percentile(backward_times_ms, 0.95),
        "backward_total_ms": sum(backward_times_ms),
        "measurement_total_wall_ms": (
            measurement_end_ns - measurement_start_ns
        ) / 1_000_000,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "step",
                "bank_index",
                "activation_index",
                "gradient_index",
                "activation_mode",
                "gradient_mode",
                "gradient_order",
                "gradient_target",
                "backward_ms",
            ]
        )
        for step, elapsed_ms in enumerate(backward_times_ms):
            index = step % bank_size
            writer.writerow(
                [
                    step,
                    index,
                    activation_indices[index],
                    gradient_indices[index],
                    args.activation_mode,
                    args.gradient_mode,
                    args.gradient_order,
                    args.gradient_target,
                    elapsed_ms,
                ]
            )

    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_output.open("w") as file:
            json.dump(summary, file, indent=2)
            file.write("\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

