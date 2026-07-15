import argparse
import csv
import gc
import json
import statistics
import time

import torch
import torch.nn.functional as F


def standardized_random(shape, generator):
    tensor = torch.randn(shape, generator=generator, dtype=torch.float32)
    tensor.sub_(tensor.mean())
    tensor.div_(tensor.std(unbiased=False) + 1e-12)
    return tensor


def make_aligned_gradient_bank(gradients, alignment_bytes=4096):
    stacked = torch.stack(gradients)
    element_size = stacked.element_size()
    if alignment_bytes % element_size != 0:
        raise ValueError("alignment_bytes must be divisible by the tensor element size")

    padding_elements = alignment_bytes // element_size
    backing = torch.empty(
        stacked.numel() + padding_elements,
        dtype=stacked.dtype,
    )
    offset_bytes = (-backing.data_ptr()) % alignment_bytes
    offset_elements = offset_bytes // element_size
    aligned = backing[offset_elements : offset_elements + stacked.numel()].view_as(stacked)
    aligned.copy_(stacked)
    if aligned.data_ptr() % alignment_bytes != 0:
        raise RuntimeError("Could not align the gradient bank")
    return aligned


def percentile(values, q):
    ordered = sorted(values)
    index = round((len(ordered) - 1) * q)
    return ordered[index]


class PerfControl:
    """Synchronize the timed loop with `perf stat --control=fifo:...`."""

    def __init__(self, control_fifo=None, ack_fifo=None):
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

    def _command(self, command):
        if self._control is None or self._ack is None:
            return
        self._control.write(f"{command}\n")
        acknowledgment = self._ack.readline().replace("\x00", "").strip()
        if acknowledgment != "ack":
            raise RuntimeError(
                f"perf did not acknowledge {command!r}; received {acknowledgment!r}"
            )

    def enable(self):
        self._command("enable")
        self.enabled = True

    def disable(self):
        try:
            self._command("disable")
        finally:
            self.enabled = False


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--regime",
        choices=[
            "large-stable",
            "large-unstable",
            "small-stable",
            "small-unstable",
        ],
        required=True,
    )

    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--spatial-size", type=int, default=32)
    parser.add_argument("--gradient-bank-size", type=int, default=16)

    # Replace these with gradient standard deviations measured from the real
    # model when those values are available.
    parser.add_argument("--large-scale", type=float, default=1e-1)
    parser.add_argument("--small-scale", type=float, default=1e-4)

    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output")
    parser.add_argument("--perf-control-fifo", help=argparse.SUPPRESS)
    parser.add_argument("--perf-ack-fifo", help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.threads <= 0:
        parser.error("--threads must be positive")
    if args.batch_size <= 0 or args.channels <= 0 or args.spatial_size <= 0:
        parser.error("--batch-size, --channels, and --spatial-size must be positive")
    if args.gradient_bank_size <= 0:
        parser.error("--gradient-bank-size must be positive")

    # Fix PyTorch parallel execution across regimes.
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    batch = args.batch_size
    channels = args.channels
    size = args.spatial_size

    # Input and weights are identical across regimes for the same seed.
    x = torch.randn(
        batch,
        channels,
        size,
        size,
        generator=generator,
        dtype=torch.float32,
        requires_grad=True,
    )

    weight = torch.randn(
        channels,
        channels,
        3,
        3,
        generator=generator,
        dtype=torch.float32,
    )
    weight.mul_(0.05)
    weight.requires_grad_(True)

    # Build the forward graph once so the timed region contains backward only.
    y = F.conv2d(x, weight, bias=None, stride=1, padding=1)

    output_shape = tuple(y.shape)
    bank_size = args.gradient_bank_size

    scale = args.large_scale if args.regime.startswith("large-") else args.small_scale
    is_stable = args.regime.endswith("-stable")

    if is_stable:
        # Every buffer contains the same direction and magnitude.
        direction = standardized_random(output_shape, generator)
        direction.mul_(scale)

        gradient_bank = make_aligned_gradient_bank(
            [direction.clone() for _ in range(bank_size)]
        )
    else:
        # Buffers contain independent directions at the selected magnitude.
        gradients = []

        for _ in range(bank_size):
            gradient = standardized_random(output_shape, generator)
            gradient.mul_(scale)
            gradients.append(gradient)

        gradient_bank = make_aligned_gradient_bank(gradients)

    flat_bank = gradient_bank.reshape(bank_size, -1)
    buffer_addresses = [gradient_bank[index].data_ptr() for index in range(bank_size)]
    buffer_bytes = gradient_bank[0].numel() * gradient_bank.element_size()
    buffer_offsets_bytes = [address - buffer_addresses[0] for address in buffer_addresses]
    expected_offsets_bytes = [index * buffer_bytes for index in range(bank_size)]

    if not gradient_bank.is_contiguous():
        raise RuntimeError("gradient_bank must be contiguous in both regimes")
    if len(set(buffer_addresses)) != bank_size:
        raise RuntimeError("Every gradient bank entry must use a distinct buffer address")
    if buffer_offsets_bytes != expected_offsets_bytes:
        raise RuntimeError(
            "Gradient bank buffers are not laid out with the expected identical sequential access pattern"
        )

    if bank_size > 1:
        adjacent_cosine = F.cosine_similarity(
            flat_bank[:-1],
            flat_bank[1:],
            dim=1,
        ).mean().item()
    else:
        adjacent_cosine = 1.0

    gradient_mean = flat_bank.mean().item()
    gradient_std = flat_bank.std(unbiased=False).item()
    gradient_l2_values = torch.linalg.vector_norm(flat_bank, dim=1)

    # Initialize backward kernels and the thread pool before timing.
    result = None

    for step in range(args.warmup):
        grad_output = gradient_bank[step % bank_size]

        result = torch.autograd.grad(
            outputs=y,
            inputs=(x, weight),
            grad_outputs=grad_output,
            retain_graph=True,
            create_graph=False,
        )

    del result
    gc.disable()

    backward_times_ms = []

    try:
        with PerfControl(args.perf_control_fifo, args.perf_ack_fifo) as perf_control:
            perf_control.enable()

            for step in range(args.steps):
                grad_output = gradient_bank[step % bank_size]

                start_ns = time.perf_counter_ns()

                grad_x, grad_weight = torch.autograd.grad(
                    outputs=y,
                    inputs=(x, weight),
                    grad_outputs=grad_output,
                    retain_graph=True,
                    create_graph=False,
                )

                end_ns = time.perf_counter_ns()

                backward_times_ms.append((end_ns - start_ns) / 1_000_000)

                # Keep only the most recent outputs alive.
                del grad_x
                del grad_weight

            perf_control.disable()
    finally:
        gc.enable()

    summary = {
        "regime": args.regime,
        "gradient_magnitude": "large" if args.regime.startswith("large-") else "small",
        "gradient_direction": "stable" if is_stable else "unstable",
        "gradient_scale": scale,
        "steps": args.steps,
        "threads": args.threads,
        "gradient_mean": gradient_mean,
        "gradient_std": gradient_std,
        "gradient_l2_mean": gradient_l2_values.mean().item(),
        "gradient_l2_min": gradient_l2_values.min().item(),
        "gradient_l2_max": gradient_l2_values.max().item(),
        "adjacent_gradient_cosine": adjacent_cosine,
        "gradient_bank_shape": list(gradient_bank.shape),
        "gradient_bank_stride": list(gradient_bank.stride()),
        "gradient_bank_contiguous": gradient_bank.is_contiguous(),
        "gradient_bank_buffer_count": bank_size,
        "gradient_bank_unique_buffer_addresses": len(set(buffer_addresses)),
        "gradient_buffer_bytes": buffer_bytes,
        "gradient_buffer_offsets_bytes": buffer_offsets_bytes,
        "gradient_bank_base_ptr_mod64": buffer_addresses[0] % 64,
        "gradient_bank_base_ptr_mod4096": buffer_addresses[0] % 4096,
        "backward_mean_ms": statistics.mean(backward_times_ms),
        "backward_median_ms": statistics.median(backward_times_ms),
        "backward_p95_ms": percentile(backward_times_ms, 0.95),
        "backward_total_ms": sum(backward_times_ms),
    }

    # Write after measurement so file I/O is outside the timed region.
    with open(args.output, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "regime", "backward_ms"])

        for step, elapsed_ms in enumerate(backward_times_ms):
            writer.writerow([step, args.regime, elapsed_ms])

    if args.summary_output:
        with open(args.summary_output, "w") as file:
            json.dump(summary, file, indent=2)
            file.write("\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
