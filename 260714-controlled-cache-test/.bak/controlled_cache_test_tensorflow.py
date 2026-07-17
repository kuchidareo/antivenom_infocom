import argparse
import csv
import gc
import json
import os
import statistics
import time


def percentile(values, q):
    ordered = sorted(values)
    index = round((len(ordered) - 1) * q)
    return ordered[index]


class PerfControl:
    """Synchronize the measured loop with ``perf stat --control=fifo:...``."""

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


def random_float32(rng, shape, np):
    """Generate float32 normals without requiring a recent NumPy version."""
    try:
        return rng.standard_normal(shape, dtype=np.float32)
    except TypeError:
        return rng.standard_normal(shape).astype(np.float32)


def standardized_random(rng, shape, np):
    array = random_float32(rng, shape, np)
    array -= array.mean(dtype=np.float32)
    array /= array.std(dtype=np.float32) + np.float32(1e-12)
    return array


def make_aligned_array(shape, np, alignment_bytes=4096):
    """Return an aligned NumPy view and the backing allocation that owns it."""
    element_count = int(np.prod(shape))
    data_bytes = element_count * np.dtype(np.float32).itemsize
    backing = np.empty(data_bytes + alignment_bytes, dtype=np.uint8)
    offset_bytes = (-backing.ctypes.data) % alignment_bytes
    aligned_bytes = backing[offset_bytes : offset_bytes + data_bytes]
    aligned = aligned_bytes.view(np.float32).reshape(shape)

    if aligned.ctypes.data % alignment_bytes != 0:
        raise RuntimeError("Could not align the NumPy gradient bank")

    return aligned, backing


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

    # TensorFlow and some CPU kernels read these values during import. Importing
    # TensorFlow here also ensures that no tensor has initialized the runtime yet.
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(args.threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    try:
        import numpy as np
        import tensorflow as tf
    except ImportError as error:
        parser.error(f"TensorFlow and NumPy are required: {error}")

    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    # CPU eager execution is normally synchronous. Make this explicit so that
    # perf_counter_ns() ends after each backward operation has completed.
    set_sync = getattr(tf.config.experimental, "set_synchronous_execution", None)
    if set_sync is not None:
        set_sync(True)

    tf.random.set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    batch = args.batch_size
    channels = args.channels
    size = args.spatial_size
    bank_size = args.gradient_bank_size

    # TensorFlow CPU Conv2D uses NHWC. The element counts and convolution
    # dimensions are the same as in the PyTorch NCHW experiment.
    input_shape = (batch, size, size, channels)
    filter_shape = (3, 3, channels, channels)
    output_shape = input_shape

    # Input, weights, allocation order, and random-number consumption are
    # identical across regimes for the same seed.
    x_numpy = random_float32(rng, input_shape, np)
    weight_numpy = random_float32(rng, filter_shape, np)
    weight_numpy *= np.float32(0.05)

    scale = args.large_scale if args.regime.startswith("large-") else args.small_scale
    is_stable = args.regime.endswith("-stable")

    # Generate all independent directions in every regime first. Stable then
    # overwrites the values in place. This equalizes allocation and RNG history.
    gradient_bank_numpy, gradient_bank_backing = make_aligned_array(
        (bank_size, *output_shape), np
    )
    for index in range(bank_size):
        gradient_bank_numpy[index] = standardized_random(rng, output_shape, np)

    if is_stable:
        for index in range(1, bank_size):
            gradient_bank_numpy[index] = gradient_bank_numpy[0]

    gradient_bank_numpy *= np.float32(scale)

    flat_bank = gradient_bank_numpy.reshape(bank_size, -1)
    source_buffer_addresses = [
        gradient_bank_numpy[index].ctypes.data for index in range(bank_size)
    ]
    buffer_bytes = gradient_bank_numpy[0].nbytes
    source_buffer_offsets_bytes = [
        address - source_buffer_addresses[0] for address in source_buffer_addresses
    ]
    expected_offsets_bytes = [index * buffer_bytes for index in range(bank_size)]

    if not gradient_bank_numpy.flags.c_contiguous:
        raise RuntimeError("The source gradient bank must be C-contiguous")
    if len(set(source_buffer_addresses)) != bank_size:
        raise RuntimeError("Every source gradient bank entry must have a distinct address")
    if source_buffer_offsets_bytes != expected_offsets_bytes:
        raise RuntimeError("The source gradient buffers are not sequentially laid out")

    if bank_size > 1:
        left = flat_bank[:-1]
        right = flat_bank[1:]
        numerator = np.sum(left * right, axis=1, dtype=np.float64)
        denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        adjacent_cosine = float(np.mean(numerator / denominator))
    else:
        adjacent_cosine = 1.0

    gradient_mean = float(flat_bank.mean(dtype=np.float64))
    gradient_std = float(flat_bank.std(dtype=np.float64))
    gradient_l2_values = np.linalg.norm(flat_bank, axis=1)

    x = tf.constant(x_numpy, dtype=tf.float32)
    weight = tf.constant(weight_numpy, dtype=tf.float32)

    # Convert the buffers before warmup. Using a Python tuple avoids running a
    # TensorFlow StridedSlice/copy operation inside the perf-controlled loop.
    gradient_buffers = tuple(
        tf.constant(gradient_bank_numpy[index], dtype=tf.float32)
        for index in range(bank_size)
    )

    # Record forward exactly once. A persistent tape is the TensorFlow analogue
    # of retain_graph=True: each measured call performs backward only.
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        tape.watch(x)
        tape.watch(weight)
        y = tf.nn.conv2d(
            x,
            weight,
            strides=[1, 1, 1, 1],
            padding="SAME",
            data_format="NHWC",
        )

    if tuple(y.shape) != output_shape:
        raise RuntimeError(f"Unexpected Conv2D output shape: {tuple(y.shape)}")

    # Initialize TensorFlow's gradient kernels and thread pools before timing.
    result = None
    for step in range(args.warmup):
        grad_output = gradient_buffers[step % bank_size]
        result = tape.gradient(
            y,
            (x, weight),
            output_gradients=grad_output,
            unconnected_gradients=tf.UnconnectedGradients.NONE,
        )

        if result[0] is None or result[1] is None:
            raise RuntimeError("TensorFlow did not produce both Conv2D gradients")

    del result
    gc.disable()

    backward_times_ms = []
    measurement_start_ns = None
    measurement_end_ns = None

    try:
        with PerfControl(args.perf_control_fifo, args.perf_ack_fifo) as perf_control:
            perf_control.enable()
            measurement_start_ns = time.perf_counter_ns()

            for step in range(args.steps):
                grad_output = gradient_buffers[step % bank_size]
                start_ns = time.perf_counter_ns()

                grad_x, grad_weight = tape.gradient(
                    y,
                    (x, weight),
                    output_gradients=grad_output,
                    unconnected_gradients=tf.UnconnectedGradients.NONE,
                )

                end_ns = time.perf_counter_ns()
                backward_times_ms.append((end_ns - start_ns) / 1_000_000)

                del grad_x
                del grad_weight

            measurement_end_ns = time.perf_counter_ns()
            perf_control.disable()
    finally:
        gc.enable()
        del tape

    if measurement_start_ns is None or measurement_end_ns is None:
        raise RuntimeError("The measured loop did not complete")

    source_stride_elements = [
        stride // gradient_bank_numpy.itemsize for stride in gradient_bank_numpy.strides
    ]

    summary = {
        "framework": "tensorflow",
        "tensorflow_version": tf.__version__,
        "execution_mode": "eager-persistent-gradient-tape",
        "data_format": "NHWC",
        "regime": args.regime,
        "gradient_magnitude": "large" if args.regime.startswith("large-") else "small",
        "gradient_direction": "stable" if is_stable else "unstable",
        "gradient_scale": scale,
        "steps": args.steps,
        "threads": args.threads,
        "gradient_mean": gradient_mean,
        "gradient_std": gradient_std,
        "gradient_l2_mean": float(gradient_l2_values.mean()),
        "gradient_l2_min": float(gradient_l2_values.min()),
        "gradient_l2_max": float(gradient_l2_values.max()),
        "adjacent_gradient_cosine": adjacent_cosine,
        "gradient_bank_shape": list(gradient_bank_numpy.shape),
        "gradient_bank_stride": source_stride_elements,
        "gradient_bank_contiguous": bool(gradient_bank_numpy.flags.c_contiguous),
        "gradient_bank_buffer_count": bank_size,
        "gradient_bank_unique_buffer_addresses": len(set(source_buffer_addresses)),
        "gradient_buffer_bytes": buffer_bytes,
        "gradient_buffer_offsets_bytes": source_buffer_offsets_bytes,
        "gradient_bank_base_ptr_mod64": source_buffer_addresses[0] % 64,
        "gradient_bank_base_ptr_mod4096": source_buffer_addresses[0] % 4096,
        "gradient_bank_address_scope": (
            "aligned NumPy source; TensorFlow public APIs do not expose or guarantee "
            "the addresses of the copied EagerTensor buffers"
        ),
        "tensorflow_gradient_buffer_count": len(gradient_buffers),
        "backward_mean_ms": statistics.mean(backward_times_ms),
        "backward_median_ms": statistics.median(backward_times_ms),
        "backward_p95_ms": percentile(backward_times_ms, 0.95),
        "backward_total_ms": sum(backward_times_ms),
        "measurement_total_wall_ms": (
            measurement_end_ns - measurement_start_ns
        ) / 1_000_000,
    }

    # Keep the aligned owner alive until all TensorFlow buffers and statistics
    # have been constructed, even though TensorFlow normally copies the data.
    _ = gradient_bank_backing

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

