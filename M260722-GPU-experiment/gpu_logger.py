#!/usr/bin/env python3
"""Controlled Nsight Compute replay for clean and shortcut-poisoned inputs.

This is intentionally not a continuous sampler. The parent process launches an
NCU worker for one NVTX-marked forward or backward range. Every worker loads the
same checkpoint and replays the same deterministic batch sequence without an
optimizer step, allowing clean and poisoned inputs to be compared at a fixed
model state.
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import os
import re
import shutil
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch

from dataset_preparation import get_dataloader, get_num_classes
from experiment_config import DEFAULT_AUGMENT, DEFAULT_BASE_SEED, set_all_seeds
from models import get_model


PRIMARY_METRICS = [
    "gpu__time_duration.sum",
    "sm__cycles_elapsed.sum",
    "smsp__inst_executed.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "lts__t_sectors_op_read.sum",
    "lts__t_sectors_op_write.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
]

PROFILE_RANGE_BY_PHASE = {
    "forward": "PROFILE_FORWARD",
    "backward": "PROFILE_BACKWARD",
}

DEFAULT_CONDITIONS = ["clean", "availability_shortcuts"]
NCU_TEMPORARY_PATHS = [Path("/tmp/nsight-compute-lock")]


@contextmanager
def nvtx_range(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def parse_json_dict(value: str) -> Dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("--augment must be a JSON object")
    return parsed


def parse_conditions(value: str) -> List[str]:
    conditions = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in conditions if item not in DEFAULT_CONDITIONS]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unsupported condition(s): {','.join(invalid)}. "
            f"Allowed: {','.join(DEFAULT_CONDITIONS)}"
        )
    if not conditions:
        raise argparse.ArgumentTypeError("At least one condition is required")
    return list(dict.fromkeys(conditions))


def find_ncu(explicit_path: str = "") -> str:
    candidates: List[str] = []
    if explicit_path:
        candidates.append(explicit_path)
    env_path = os.environ.get("NCU", "")
    if env_path:
        candidates.append(env_path)
    path_match = shutil.which("ncu")
    if path_match:
        candidates.append(path_match)
    candidates.extend(
        [
            "/usr/local/cuda/bin/ncu",
            *sorted(glob.glob("/usr/local/cuda-*/bin/ncu"), reverse=True),
            *sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"), reverse=True),
        ]
    )
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())
    raise FileNotFoundError(
        "Nsight Compute CLI (ncu) was not found. Set --ncu-path or NCU."
    )


def sudo_prefix(use_sudo: bool) -> List[str]:
    return ["sudo", "-n"] if use_sudo else []


def authenticate_sudo() -> None:
    print("Authenticating once for Nsight Compute hardware counters...")
    subprocess.run(["sudo", "-v"], check=True)


def cleanup_ncu_temporary_files(*, use_sudo: bool) -> None:
    """Remove only known disposable NCU files, never reports or CUDA JIT caches."""
    for path in NCU_TEMPORARY_PATHS:
        command = [*sudo_prefix(use_sudo), "rm", "-f", str(path)]
        subprocess.run(command, check=False)


def remove_file(path: Path, *, use_sudo: bool) -> None:
    if not path.exists():
        return
    subprocess.run([*sudo_prefix(use_sudo), "rm", "-f", str(path)], check=True)


def restore_invoking_user_ownership(output_dir: Path) -> None:
    """Avoid root-owned CSVs when the parent was started through sudo."""
    if os.geteuid() != 0:
        return
    sudo_uid = os.environ.get("SUDO_UID")
    sudo_gid = os.environ.get("SUDO_GID")
    if not sudo_uid or not sudo_gid:
        return
    uid = int(sudo_uid)
    gid = int(sudo_gid)
    if output_dir.exists():
        os.chown(output_dir, uid, gid)
        for path in output_dir.glob("*.csv"):
            os.chown(path, uid, gid)


def query_available_metrics(ncu: str, *, use_sudo: bool) -> tuple[List[str], str]:
    bases = [metric.rsplit(".", 1)[0] for metric in PRIMARY_METRICS]
    command = [
        *sudo_prefix(use_sudo),
        ncu,
        "--query-metrics",
        "--query-metrics-mode",
        "suffix",
        "--metrics",
        ",".join(bases),
        "--devices",
        "0",
    ]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    query_text = result.stdout + result.stderr
    metric_names = set(re.findall(r"^([A-Za-z0-9_.]+)\s+", query_text, re.MULTILINE))
    selected = [metric for metric in PRIMARY_METRICS if metric in metric_names]
    return selected, query_text


def create_fixed_checkpoint(args: argparse.Namespace, checkpoint_path: Path) -> None:
    set_all_seeds(args.seed)
    num_classes = get_num_classes(args.data_dir, dataset_name=args.dataset)
    model = get_model(args.model, num_classes=num_classes)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model": args.model,
            "num_classes": num_classes,
            "seed": args.seed,
        },
        checkpoint_path,
    )


def load_fixed_model(args: argparse.Namespace) -> torch.nn.Module:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    expected_model = str(checkpoint.get("model", args.model))
    if expected_model != args.model:
        raise ValueError(
            f"Checkpoint model is {expected_model}, but --model is {args.model}."
        )
    num_classes = int(checkpoint["num_classes"])
    model = get_model(args.model, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model


def build_replay_loader(args: argparse.Namespace):
    augment = dict(DEFAULT_AUGMENT)
    augment.update(parse_json_dict(args.augment))
    # Controlled replay must not introduce a new random flip on each NCU pass.
    augment["horizontal_flip"] = False
    return get_dataloader(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        client_id=args.client_id,
        poisoning_method=args.condition,
        split="train",
        augment=augment,
        batch_size=args.batch_size,
        shuffle=False,
    )


def replay_batches(
    *,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    batches: Iterable[Any],
    target_batch: int,
    profile_phase: str,
) -> List[Dict[str, float]]:
    if profile_phase not in PROFILE_RANGE_BY_PHASE:
        raise ValueError(f"Unknown profile phase: {profile_phase}")
    if target_batch < 0:
        raise ValueError("target_batch must be zero or greater")

    model.eval()
    model.zero_grad(set_to_none=True)
    timing_events: List[Dict[str, Any]] = []
    target_seen = False

    for batch_id, (images, labels) in enumerate(batches):
        if batch_id > target_batch:
            break
        images = images.cuda(non_blocking=False)
        labels = labels.cuda(non_blocking=False)
        is_target = batch_id == target_batch
        model.zero_grad(set_to_none=True)

        forward_start = torch.cuda.Event(enable_timing=True)
        forward_end = torch.cuda.Event(enable_timing=True)
        backward_start = torch.cuda.Event(enable_timing=True)
        backward_end = torch.cuda.Event(enable_timing=True)

        forward_name = (
            PROFILE_RANGE_BY_PHASE["forward"]
            if is_target and profile_phase == "forward"
            else f"batch_{batch_id}_forward"
        )
        forward_start.record()
        with nvtx_range(forward_name):
            logits = model(images)
        forward_end.record()

        loss = criterion(logits, labels)
        backward_name = (
            PROFILE_RANGE_BY_PHASE["backward"]
            if is_target and profile_phase == "backward"
            else f"batch_{batch_id}_backward"
        )
        backward_start.record()
        with nvtx_range(backward_name):
            loss.backward()
        backward_end.record()

        timing_events.append(
            {
                "batch_id": batch_id,
                "forward_start": forward_start,
                "forward_end": forward_end,
                "backward_start": backward_start,
                "backward_end": backward_end,
            }
        )
        target_seen = target_seen or is_target

    if not target_seen:
        raise ValueError(
            f"target_batch={target_batch} does not exist; loader has fewer batches."
        )

    torch.cuda.synchronize()
    timings: List[Dict[str, float]] = []
    for row in timing_events:
        timings.append(
            {
                "batch_id": int(row["batch_id"]),
                "forward_ms": float(row["forward_start"].elapsed_time(row["forward_end"])),
                "backward_ms": float(row["backward_start"].elapsed_time(row["backward_end"])),
            }
        )
    return timings


def worker_main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the NCU replay worker.")
    set_all_seeds(args.seed)
    torch.cuda.set_device(0)
    model = load_fixed_model(args).cuda()
    criterion = torch.nn.CrossEntropyLoss()
    loader = build_replay_loader(args)
    timings = replay_batches(
        model=model,
        criterion=criterion,
        batches=loader,
        target_batch=args.target_batch,
        profile_phase=args.profile_phase,
    )
    target_timing = next(row for row in timings if row["batch_id"] == args.target_batch)
    print(
        json.dumps(
            {
                "condition": args.condition,
                "epoch": args.epoch,
                "profile_phase": args.profile_phase,
                "target_batch": args.target_batch,
                "forward_ms": target_timing["forward_ms"],
                "backward_ms": target_timing["backward_ms"],
                "cuda_device": torch.cuda.get_device_name(0),
            },
            sort_keys=True,
        )
    )


def worker_arguments(
    args: argparse.Namespace,
    *,
    condition: str,
    phase: str,
    checkpoint_path: Path,
) -> List[str]:
    return [
        str(Path(__file__).resolve()),
        "--worker",
        "--condition",
        condition,
        "--epoch",
        str(args.epoch),
        "--profile-phase",
        phase,
        "--target-batch",
        str(args.target_batch),
        "--checkpoint",
        str(checkpoint_path.resolve()),
        "--dataset",
        args.dataset,
        "--data-dir",
        str(Path(args.data_dir).resolve()),
        "--client-id",
        args.client_id,
        "--model",
        args.model,
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--augment",
        args.augment,
    ]


def launch_ncu(
    args: argparse.Namespace,
    *,
    ncu: str,
    metrics: Sequence[str],
    condition: str,
    phase: str,
    checkpoint_path: Path,
) -> Dict[str, Any]:
    range_name = PROFILE_RANGE_BY_PHASE[phase]
    report_stem = args.output_dir / (
        f"{args.run_id}_ncu_{condition}_epoch_{args.epoch}_"
        f"{phase}_batch_{args.target_batch}"
    )
    command = [
        *sudo_prefix(not args.no_sudo),
        ncu,
        "--replay-mode",
        "app-range",
        "--cache-control",
        "none",
        "--clock-control",
        "none",
        "--nvtx",
        "--nvtx-include",
        f"{range_name}/",
        "--metrics",
        ",".join(metrics),
        "--export",
        str(report_stem),
        "--force-overwrite",
        sys.executable,
        *worker_arguments(
            args,
            condition=condition,
            phase=phase,
            checkpoint_path=checkpoint_path,
        ),
    ]
    print("Running:")
    print(" ".join(command))
    report_path = report_stem.with_suffix(".ncu-rep")
    metrics_csv = report_stem.with_name(f"{report_stem.name}_metrics.csv")
    if not args.dry_run:
        try:
            subprocess.run(command, check=True)
            export_report_metrics(
                ncu=ncu,
                report_path=report_path,
                output_path=metrics_csv,
                metrics=metrics,
                condition=condition,
                epoch=args.epoch,
                phase=phase,
                target_batch=args.target_batch,
            )
        finally:
            remove_file(report_path, use_sudo=not args.no_sudo)
    return {
        "condition": condition,
        "epoch": args.epoch,
        "phase": phase,
        "target_batch": args.target_batch,
        "metrics_csv": str(metrics_csv),
    }


def export_report_metrics(
    *,
    ncu: str,
    report_path: Path,
    output_path: Path,
    metrics: Sequence[str],
    condition: str,
    epoch: int,
    phase: str,
    target_batch: int,
) -> None:
    result = subprocess.run(
        [ncu, "--import", str(report_path), "--csv", "--page", "raw"],
        check=True,
        text=True,
        capture_output=True,
    )
    rows = list(csv.reader(io.StringIO(result.stdout)))
    if len(rows) < 3:
        raise RuntimeError(f"NCU raw CSV export was empty for {report_path}")

    header = rows[0]
    units = rows[1]
    metric_indices = {
        metric: header.index(metric)
        for metric in metrics
        if metric in header
    }
    if not metric_indices:
        raise RuntimeError(f"Selected metrics were absent from NCU report {report_path}")

    kernel_index = header.index("Kernel Name") if "Kernel Name" in header else None
    data_rows = rows[2:]
    if kernel_index is not None:
        range_rows = [row for row in data_rows if len(row) > kernel_index and row[kernel_index] == "range"]
        if range_rows:
            data_rows = range_rows

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "condition",
                "epoch",
                "phase",
                "target_batch",
                "range_row",
                "metric",
                "unit",
                "value",
                "report",
            ],
        )
        writer.writeheader()
        for range_row, row in enumerate(data_rows):
            for metric, index in metric_indices.items():
                writer.writerow(
                    {
                        "condition": condition,
                        "epoch": epoch,
                        "phase": phase,
                        "target_batch": target_batch,
                        "range_row": range_row,
                        "metric": metric,
                        "unit": units[index] if index < len(units) else "",
                        "value": row[index] if index < len(row) else "",
                        "report": str(report_path),
                    }
                )


def write_run_summary(
    *,
    args: argparse.Namespace,
    ncu: str,
    selected: Sequence[str],
    missing: Sequence[str],
    runs: Sequence[Dict[str, Any]],
) -> None:
    condition_tag = "-".join(args.conditions)
    output_path = args.output_dir / (
        f"{args.run_id}_ncu_{condition_tag}_epoch_{args.epoch}_profile_runs.csv"
    )
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "run_id",
                "condition",
                "epoch",
                "phase",
                "target_batch",
                "dataset",
                "client_id",
                "model",
                "batch_size",
                "seed",
                "ncu",
                "selected_metrics",
                "missing_metrics",
                "checkpoint",
                "metrics_csv",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    **run,
                    "run_id": args.run_id,
                    "epoch": args.epoch,
                    "dataset": args.dataset,
                    "client_id": args.client_id,
                    "model": args.model,
                    "batch_size": args.batch_size,
                    "seed": args.seed,
                    "ncu": ncu,
                    "selected_metrics": ",".join(selected),
                    "missing_metrics": ",".join(missing),
                    "checkpoint": args.checkpoint,
                }
            )


def parent_main(args: argparse.Namespace) -> None:
    try:
        run_parent(args)
    finally:
        if not args.keep_ncu_temporary_files:
            cleanup_ncu_temporary_files(use_sudo=not args.no_sudo)
        restore_invoking_user_ownership(args.output_dir)


def run_parent(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    ncu = find_ncu(args.ncu_path)
    if not args.no_sudo and not args.skip_sudo_auth:
        authenticate_sudo()

    selected, query_text = query_available_metrics(ncu, use_sudo=not args.no_sudo)
    missing = [metric for metric in PRIMARY_METRICS if metric not in selected]
    if not selected:
        raise RuntimeError("None of the requested NCU metrics are available.")

    print("Selected metrics:")
    for metric in selected:
        print(f"  {metric}")
    if missing:
        print("Unavailable metrics (continuing without them):")
        for metric in missing:
            print(f"  {metric}")

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else args.output_dir / "fixed_checkpoint.pt"
    checkpoint_was_created = not bool(args.checkpoint)
    if args.checkpoint:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        checkpoint_epoch = checkpoint.get("epoch")
        if args.epoch < 0 and checkpoint_epoch is not None:
            args.epoch = int(checkpoint_epoch)
        elif checkpoint_epoch is not None and args.epoch != int(checkpoint_epoch):
            raise ValueError(
                f"--epoch={args.epoch} does not match checkpoint epoch "
                f"{checkpoint_epoch}."
            )
        checkpoint_condition = checkpoint.get("condition")
        if (
            checkpoint_condition is not None
            and len(args.conditions) == 1
            and args.conditions[0] != checkpoint_condition
        ):
            raise ValueError(
                f"--conditions={args.conditions[0]} does not match checkpoint "
                f"condition {checkpoint_condition}."
            )
        if checkpoint.get("seed") is not None:
            args.seed = int(checkpoint["seed"])
    else:
        create_fixed_checkpoint(args, checkpoint_path)
    if args.epoch < 0:
        args.epoch = 0
    args.checkpoint = str(checkpoint_path)

    try:
        phases = ["forward", "backward"] if args.profile_phase == "both" else [args.profile_phase]
        runs: List[Dict[str, Any]] = []
        for condition in args.conditions:
            for phase in phases:
                runs.append(
                    launch_ncu(
                        args,
                        ncu=ncu,
                        metrics=selected,
                        condition=condition,
                        phase=phase,
                        checkpoint_path=checkpoint_path,
                    )
                )
        write_run_summary(
            args=args,
            ncu=ncu,
            selected=selected,
            missing=missing,
            runs=runs,
        )
    finally:
        if checkpoint_was_created:
            remove_file(checkpoint_path, use_sudo=False)
    print(f"NCU controlled replay finished: {args.output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile one deterministic forward/backward batch with Nsight Compute."
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dataset", default="kuchidareo/small_trashnet")
    parser.add_argument("--data-dir", default="../iid-data")
    parser.add_argument("--client-id", default="client_1")
    parser.add_argument("--model", default="simple_cnn")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--target-batch", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument(
        "--augment",
        default=json.dumps(
            {
                "enabled": True,
                "resize": [224, 224],
                "horizontal_flip": False,
                "normalize": True,
            }
        ),
    )
    parser.add_argument(
        "--conditions",
        type=parse_conditions,
        default=DEFAULT_CONDITIONS,
        help="Comma-separated conditions (default: clean,availability_shortcuts).",
    )
    parser.add_argument(
        "--condition",
        choices=DEFAULT_CONDITIONS,
        default="clean",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=-1,
        help="Training epoch represented by --checkpoint; inferred when available.",
    )
    parser.add_argument(
        "--profile-phase",
        choices=["both", "forward", "backward"],
        default="both",
    )
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output-dir", type=Path, default=Path("logs/local_ml"))
    parser.add_argument("--ncu-path", default="")
    parser.add_argument(
        "--no-sudo",
        action="store_true",
        help="Run NCU without sudo (for systems where counters are already accessible).",
    )
    parser.add_argument(
        "--skip-sudo-auth",
        action="store_true",
        help="Skip sudo -v; subsequent NCU commands still use sudo -n.",
    )
    parser.add_argument(
        "--keep-ncu-temporary-files",
        action="store_true",
        help="Keep known NCU temporary files such as /tmp/nsight-compute-lock.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.worker:
        if args.profile_phase == "both":
            raise ValueError("A worker must profile exactly one phase.")
        if not args.checkpoint:
            raise ValueError("A replay worker requires --checkpoint.")
        worker_main(args)
        return
    parent_main(args)


if __name__ == "__main__":
    main()
