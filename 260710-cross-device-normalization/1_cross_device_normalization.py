"""CLI entry point for cross-device normalization experiments."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

os.environ["MPLCONFIGDIR"] = str(Path(".matplotlib-cache").resolve())
os.environ["XDG_CACHE_HOME"] = str(Path(".cache").resolve())

from cross_device_normalization import ExperimentConfig, run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-device local CPU waveform normalization with OT."
    )
    parser.add_argument("--logs-dir", type=Path, default=Path("collected_logs"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cross_device_normalization"),
    )
    parser.add_argument(
        "--signal",
        choices=["process_cpu_percent", "system_cpu_mean"],
        default="process_cpu_percent",
    )
    parser.add_argument("--resampled-length", type=int, default=512)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--window-stride", type=int, default=4)
    parser.add_argument("--sinkhorn-reg", type=float, default=0.05)
    parser.add_argument("--lambda-mean", type=float, default=1.0)
    parser.add_argument("--lambda-std", type=float, default=1.0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--experiment",
        choices=["all", "shape_only", "shape_level_scale"],
        default="all",
    )
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--rpi4-reference-run")
    parser.add_argument("--jetson-reference-run")
    parser.add_argument("--rpi3-reference-run")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s:%(name)s:%(message)s",
    )
    config = ExperimentConfig(
        logs_dir=args.logs_dir,
        output_dir=args.output_dir,
        signal=args.signal,
        resampled_length=args.resampled_length,
        window_size=args.window_size,
        window_stride=args.window_stride,
        sinkhorn_reg=args.sinkhorn_reg,
        lambda_mean=args.lambda_mean,
        lambda_std=args.lambda_std,
        random_seed=args.random_seed,
        experiment=args.experiment,
        inspect_only=args.inspect_only,
        rpi4_reference_run=args.rpi4_reference_run,
        jetson_reference_run=args.jetson_reference_run,
        rpi3_reference_run=args.rpi3_reference_run,
    )
    run(config)


if __name__ == "__main__":
    main()
