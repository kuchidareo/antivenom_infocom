from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Tuple


Condition = Tuple[str, str]


def condition_name(condition: Condition) -> str:
    activation_mode, gradient_mode = condition
    return f"activation-{activation_mode}_gradient-{gradient_mode}"


def balanced_order(replicate: int) -> List[Condition]:
    # A four-period cyclic Latin square. Across each block of four replicates,
    # every condition appears exactly once in every execution position.
    base: List[Condition] = [
        ("held", "held"),
        ("held", "stream"),
        ("stream", "stream"),
        ("stream", "held"),
    ]
    shift = replicate % len(base)
    return base[shift:] + base[:shift]


def mean_or_nan(values: List[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def sample_std_or_zero(values: List[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run and aggregate an order-balanced 2x2 CNN replay experiment."
    )
    parser.add_argument("--activation-source", type=Path, required=True)
    parser.add_argument("--gradient-source", type=Path)
    parser.add_argument("--weight-source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--bank-size", type=int)
    parser.add_argument(
        "--gradient-target",
        choices=("both", "input", "weight"),
        default="both",
    )
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if args.replicates <= 0:
        parser.error("--replicates must be positive")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.threads <= 0:
        parser.error("--threads must be positive")
    if args.bank_size is not None and args.bank_size <= 0:
        parser.error("--bank-size must be positive")

    script = Path(__file__).with_name("controlled_cnn_factorial_replay.py").resolve()
    if not script.exists():
        parser.error(f"Replay script not found: {script}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    run_rows: List[Dict[str, object]] = []
    condition_values: Dict[Condition, List[float]] = defaultdict(list)
    position_values: Dict[int, List[float]] = defaultdict(list)

    for replicate_index in range(args.replicates):
        replicate_number = replicate_index + 1
        for position_index, condition in enumerate(balanced_order(replicate_index)):
            position = position_index + 1
            activation_mode, gradient_mode = condition
            name = condition_name(condition)
            stem = f"replicate_{replicate_number:02d}_position_{position:02d}_{name}"
            csv_path = raw_dir / f"{stem}.csv"
            json_path = raw_dir / f"{stem}.json"

            command = [
                sys.executable,
                str(script),
                "--activation-source",
                str(args.activation_source),
                "--activation-mode",
                activation_mode,
                "--gradient-mode",
                gradient_mode,
                "--gradient-order",
                "matched",
                "--gradient-target",
                args.gradient_target,
                "--steps",
                str(args.steps),
                "--warmup",
                str(args.warmup),
                "--threads",
                str(args.threads),
                "--seed",
                str(args.seed),
                "--output",
                str(csv_path),
                "--summary-output",
                str(json_path),
            ]
            if args.gradient_source:
                command.extend(["--gradient-source", str(args.gradient_source)])
            if args.weight_source:
                command.extend(["--weight-source", str(args.weight_source)])
            if args.bank_size is not None:
                command.extend(["--bank-size", str(args.bank_size)])

            print(
                f"replicate={replicate_number}/{args.replicates} "
                f"position={position}/4 {name}",
                flush=True,
            )
            subprocess.run(command, check=True)

            with json_path.open() as file:
                summary = json.load(file)
            median_ms = float(summary["backward_median_ms"])
            condition_values[condition].append(median_ms)
            position_values[position].append(median_ms)
            run_rows.append(
                {
                    "replicate": replicate_number,
                    "position": position,
                    "activation_mode": activation_mode,
                    "gradient_mode": gradient_mode,
                    "gradient_target": args.gradient_target,
                    "backward_mean_ms": float(summary["backward_mean_ms"]),
                    "backward_median_ms": median_ms,
                    "backward_p95_ms": float(summary["backward_p95_ms"]),
                    "measurement_total_wall_ms": float(
                        summary["measurement_total_wall_ms"]
                    ),
                    "summary_path": str(json_path),
                }
            )

    run_summary_path = args.output_dir / "run_summary.csv"
    with run_summary_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(run_rows[0]))
        writer.writeheader()
        writer.writerows(run_rows)

    condition_rows = []
    cell_means: Dict[Condition, float] = {}
    for condition in (
        ("held", "held"),
        ("held", "stream"),
        ("stream", "held"),
        ("stream", "stream"),
    ):
        values = condition_values[condition]
        mean_median = mean_or_nan(values)
        cell_means[condition] = mean_median
        condition_rows.append(
            {
                "activation_mode": condition[0],
                "gradient_mode": condition[1],
                "replicates": len(values),
                "mean_of_run_medians_ms": mean_median,
                "std_of_run_medians_ms": sample_std_or_zero(values),
                "min_run_median_ms": min(values),
                "max_run_median_ms": max(values),
            }
        )

    condition_summary_path = args.output_dir / "condition_summary.csv"
    with condition_summary_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(condition_rows[0]))
        writer.writeheader()
        writer.writerows(condition_rows)

    hh = cell_means[("held", "held")]
    hs = cell_means[("held", "stream")]
    sh = cell_means[("stream", "held")]
    ss = cell_means[("stream", "stream")]
    grand_mean = statistics.mean((hh, hs, sh, ss))
    activation_effect = ((sh + ss) - (hh + hs)) / 2
    gradient_effect = ((hs + ss) - (hh + sh)) / 2
    interaction = (ss - sh) - (hs - hh)

    aggregate = {
        "replicates": args.replicates,
        "response": "mean of per-run backward medians",
        "cell_means_ms": {
            condition_name(condition): value
            for condition, value in cell_means.items()
        },
        "grand_mean_ms": grand_mean,
        "activation_stream_minus_held_effect_ms": activation_effect,
        "activation_stream_minus_held_effect_pct_of_grand_mean": (
            100 * activation_effect / grand_mean
        ),
        "gradient_stream_minus_held_effect_ms": gradient_effect,
        "gradient_stream_minus_held_effect_pct_of_grand_mean": (
            100 * gradient_effect / grand_mean
        ),
        "activation_gradient_interaction_ms": interaction,
        "activation_gradient_interaction_pct_of_grand_mean": (
            100 * interaction / grand_mean
        ),
        "execution_position_mean_medians_ms": {
            str(position): statistics.mean(values)
            for position, values in sorted(position_values.items())
        },
        "run_summary": str(run_summary_path),
        "condition_summary": str(condition_summary_path),
    }
    aggregate_path = args.output_dir / "aggregate_summary.json"
    with aggregate_path.open("w") as file:
        json.dump(aggregate, file, indent=2)
        file.write("\n")

    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()

