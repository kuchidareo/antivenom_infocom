"""Command line interface for campaign generation, validation and inspection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from .generate import generate_manifest, load_targets, plan_batches
from .schema import (
    CampaignValidationError,
    load_json,
    manifest_datasets,
    select_assignment,
    validate_manifest,
    write_json_atomic,
)


def _generate(arguments: argparse.Namespace) -> int:
    space = load_json(arguments.space)
    targets = load_targets(arguments.targets)
    manifest = generate_manifest(space, targets)
    write_json_atomic(arguments.output, manifest)
    print(
        "Generated {} with {} hardware assignments and {} contexts each".format(
            arguments.output,
            len(manifest["assignments"]),
            manifest["contexts_per_hardware"],
        )
    )
    return 0


def _validate(arguments: argparse.Namespace) -> int:
    manifest = load_json(arguments.manifest)
    validate_manifest(manifest)
    print(
        "Valid campaign: id={} assignments={} contexts_per_hardware={}".format(
            manifest["campaign_id"],
            len(manifest["assignments"]),
            manifest["contexts_per_hardware"],
        )
    )
    return 0


def _batches(arguments: argparse.Namespace) -> int:
    targets = load_targets(arguments.targets)
    document = plan_batches(targets, arguments.max_size)
    write_json_atomic(arguments.output, document)
    print("Generated {} batches in {}".format(len(document["batches"]), arguments.output))
    return 0


def _plan(arguments: argparse.Namespace) -> int:
    manifest = load_json(arguments.manifest)
    validate_manifest(manifest)
    assignment = select_assignment(
        manifest, arguments.cluster, arguments.hardware_type, arguments.node_id
    )
    calibration = manifest["calibration"]
    if calibration.get("enabled", True):
        context = calibration["context"]
        print(
            "calibration {} dataset={} model={} batch={} epochs={} trials={} conditions={}".format(
                context["context_id"],
                context["dataset"],
                context["model"],
                context["batch_size"],
                context["local_epochs"],
                context["analysis_trials"],
                ",".join(calibration["conditions"]),
            )
        )
    for index, context in enumerate(assignment["contexts"], 1):
        print(
            "{:02d} {} dataset={} model={} batch={} partition={} bg={} conditions={}".format(
                index,
                context["context_id"],
                context["dataset"],
                context["model"],
                context["batch_size"],
                context["partition_method"],
                context["background_workload"],
                ",".join(manifest["conditions"]),
            )
        )
    return 0


def _datasets(arguments: argparse.Namespace) -> int:
    manifest = load_json(arguments.manifest)
    validate_manifest(manifest)
    datasets = manifest_datasets(
        manifest,
        cluster=arguments.cluster,
        hardware_type=arguments.hardware_type,
        node_id=arguments.node_id,
    )
    print(",".join(datasets))
    return 0


def _status(arguments: argparse.Namespace) -> int:
    campaign_directory = arguments.state_root / arguments.campaign_id
    if not campaign_directory.is_dir():
        raise CampaignValidationError(
            "Campaign state directory does not exist: {}".format(campaign_directory)
        )

    print("hardware\tstatus\tdone\tfailed\trunning\tdry_run")
    for hardware_directory in sorted(
        path for path in campaign_directory.iterdir() if path.is_dir()
    ):
        runs_directory = hardware_directory / "runs"
        counts = {"done": 0, "failed": 0, "running": 0, "dry_run": 0}
        if runs_directory.is_dir():
            for run_directory in runs_directory.iterdir():
                if not run_directory.is_dir():
                    continue
                if (run_directory / "done.json").exists():
                    counts["done"] += 1
                elif (run_directory / "failed.json").exists():
                    counts["failed"] += 1
                elif (run_directory / "running.json").exists():
                    counts["running"] += 1
                elif (run_directory / "dry-run.json").exists():
                    counts["dry_run"] += 1
        summary_path = hardware_directory / "summary.json"
        status = "unknown"
        if summary_path.is_file():
            status = str(load_json(summary_path).get("status", "unknown"))
        print(
            "{}\t{}\t{}\t{}\t{}\t{}".format(
                hardware_directory.name,
                status,
                counts["done"],
                counts["failed"],
                counts["running"],
                counts["dry_run"],
            )
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="generate a deterministic manifest")
    generate.add_argument("--space", required=True, type=Path)
    generate.add_argument("--targets", required=True, type=Path)
    generate.add_argument("--output", required=True, type=Path)
    generate.set_defaults(handler=_generate)

    validate = commands.add_parser("validate", help="validate a generated manifest")
    validate.add_argument("--manifest", required=True, type=Path)
    validate.set_defaults(handler=_validate)

    batches = commands.add_parser("batches", help="split targets into cluster batches")
    batches.add_argument("--targets", required=True, type=Path)
    batches.add_argument("--max-size", type=int, default=10)
    batches.add_argument("--output", required=True, type=Path)
    batches.set_defaults(handler=_batches)

    plan = commands.add_parser("plan", help="print one node's assigned contexts")
    plan.add_argument("--manifest", required=True, type=Path)
    plan.add_argument("--cluster", required=True)
    plan.add_argument("--hardware-type", required=True)
    plan.add_argument("--node-id", default="")
    plan.set_defaults(handler=_plan)

    datasets = commands.add_parser(
        "datasets", help="print comma-separated datasets needed by one node"
    )
    datasets.add_argument("--manifest", required=True, type=Path)
    datasets.add_argument("--cluster", required=True)
    datasets.add_argument("--hardware-type", required=True)
    datasets.add_argument("--node-id", default="")
    datasets.set_defaults(handler=_datasets)

    status = commands.add_parser("status", help="summarize runtime campaign state")
    status.add_argument("--state-root", required=True, type=Path)
    status.add_argument("--campaign-id", required=True)
    status.set_defaults(handler=_status)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        return int(arguments.handler(arguments))
    except (CampaignValidationError, OSError, json.JSONDecodeError) as error:
        print("Campaign error: {}".format(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
