#!/usr/bin/env python3
"""Measure TinyViT training memory without requiring a prepared dataset."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import resource
import tempfile
import threading
import time
from typing import Iterable

import torch

from models import get_model


MIB = 1024 * 1024


def tensor_storage_bytes(tensors: Iterable[torch.Tensor]) -> int:
    storages: dict[int, int] = {}
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            continue
        storage = tensor.untyped_storage()
        storages[storage.data_ptr()] = storage.nbytes()
    return sum(storages.values())


def current_rss_bytes() -> int:
    with open("/proc/self/status", encoding="ascii") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("VmRSS is unavailable in /proc/self/status")


def peak_rss_bytes() -> int:
    # Linux reports ru_maxrss in KiB.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


class RssSampler:
    def __init__(self, interval_seconds: float = 0.001) -> None:
        self.interval_seconds = interval_seconds
        self.peak = current_rss_bytes()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.peak = max(self.peak, current_rss_bytes())

    def __enter__(self) -> "RssSampler":
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, current_rss_bytes(), peak_rss_bytes())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--limit-mb", type=float, default=500.0)
    parser.add_argument(
        "--with-operator-logging",
        action="store_true",
        help="Include one task-clock PMU operator trace.",
    )
    parser.add_argument(
        "--with-entropy-logging",
        action="store_true",
        help="Attach entropy summarization to the operator trace.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.torch_threads < 1:
        raise ValueError("batch-size and torch-threads must be positive")
    if args.with_entropy_logging and not args.with_operator_logging:
        raise ValueError("--with-entropy-logging requires --with-operator-logging")
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(260922)

    import_baseline = current_rss_bytes()
    model = get_model(
        "tiny_vit",
        num_classes=args.num_classes,
        input_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        model_depth=args.depth,
        vit_patch_size=args.patch_size,
        vit_embed_dim=args.embed_dim,
        vit_heads=args.heads,
        vit_mlp_ratio=args.mlp_ratio,
    )
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    images = torch.randn(args.batch_size, 3, args.image_size, args.image_size)
    labels = torch.randint(args.num_classes, (args.batch_size,))
    before_training = current_rss_bytes()

    parameter_storages = {
        parameter.untyped_storage().data_ptr() for parameter in model.parameters()
    }
    saved_activation_storages: dict[int, int] = {}

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        storage = tensor.untyped_storage()
        pointer = storage.data_ptr()
        if pointer not in parameter_storages:
            saved_activation_storages[pointer] = storage.nbytes()
        return tensor

    def unpack(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    with tempfile.TemporaryDirectory() as temporary_directory:
        logger_context = nullcontext(None)
        if args.with_operator_logging:
            from perf_logger import LayerPerfLogger

            temporary = Path(temporary_directory)
            entropy = None
            if args.with_entropy_logging:
                from entropy_logger import LayerEntropyLogger

                entropy = LayerEntropyLogger(
                    path=temporary / "probe_entropy_summary.csv",
                    condition={},
                )
            logger_context = LayerPerfLogger(
                model=model,
                path=temporary / "probe_layer_perf.csv",
                condition={},
                events=["task-clock"],
                observer=entropy,
            )

        with RssSampler() as sampler, logger_context as operator_logger:
            optimizer.zero_grad(set_to_none=True)
            if operator_logger is not None:
                operator_logger.begin_batch(epoch=0, batch_idx=0)
            with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                logits = model(images)
                loss = criterion(logits, labels)
                saved_activation_bytes = sum(saved_activation_storages.values())
                loss.backward()
            if operator_logger is not None:
                operator_logger.flush_batch()
            optimizer.step()

    parameter_bytes = tensor_storage_bytes(model.parameters())
    gradient_bytes = tensor_storage_bytes(
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    )
    optimizer_bytes = tensor_storage_bytes(
        value
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )
    process_peak = max(sampler.peak, peak_rss_bytes())
    result = {
        "configuration": {
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "depth": args.depth,
            "patch_size": args.patch_size,
            "embed_dim": args.embed_dim,
            "heads": args.heads,
            "mlp_ratio": args.mlp_ratio,
            "torch_threads": args.torch_threads,
            "operator_logging": args.with_operator_logging,
            "entropy_logging": args.with_entropy_logging,
        },
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "parameter_mb": parameter_bytes / MIB,
        "gradient_mb": gradient_bytes / MIB,
        "adam_state_mb": optimizer_bytes / MIB,
        "saved_activation_storage_mb": saved_activation_bytes / MIB,
        "training_tensor_mb": (
            parameter_bytes + gradient_bytes + optimizer_bytes + saved_activation_bytes
        )
        / MIB,
        "rss_after_import_mb": import_baseline / MIB,
        "rss_before_training_mb": before_training / MIB,
        "peak_process_rss_mb": process_peak / MIB,
        "incremental_peak_from_import_mb": (process_peak - import_baseline) / MIB,
        "incremental_training_peak_from_ready_mb": (
            process_peak - before_training
        )
        / MIB,
        "limit_mb": args.limit_mb,
        "saved_activations_under_limit": saved_activation_bytes <= args.limit_mb * MIB,
        "training_working_set_under_limit": (
            process_peak - before_training
        )
        <= args.limit_mb * MIB,
        "process_rss_under_limit": process_peak <= args.limit_mb * MIB,
        "pid": os.getpid(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
