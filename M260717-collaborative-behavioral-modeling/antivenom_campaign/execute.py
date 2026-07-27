"""Execute one hardware assignment from a generated campaign manifest."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .schema import (
    CampaignValidationError,
    load_json,
    select_assignment,
    slug,
    validate_manifest,
    write_json_atomic,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def context_run_id(kind: str, context: Mapping[str, Any]) -> str:
    return "{}_{}".format(kind, context["context_id"])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_event(path: Path, status: str, **extra: Any) -> None:
    document: Dict[str, Any] = {"status": status, "timestamp": now()}
    document.update(extra)
    write_json_atomic(path, document)


class CampaignExecutor:
    def __init__(
        self,
        manifest_path: Path,
        cluster: str,
        hardware_type: str,
        node_id: str,
        runner: Path,
        python: Path,
        state_root: Path,
        log_root: Path,
        result_root: Path,
        restart: bool = False,
        dry_run: bool = False,
        heartbeat_seconds: int = 30,
    ) -> None:
        self.manifest_path = manifest_path.resolve()
        self.manifest = load_json(self.manifest_path)
        validate_manifest(self.manifest)
        self.manifest_sha256 = sha256_file(self.manifest_path)
        self.cluster = cluster
        self.hardware_type = hardware_type
        self.node_id = node_id
        self.assignment = select_assignment(
            self.manifest, cluster, hardware_type, node_id
        )
        self.runner = runner.resolve()
        self.python = python.resolve()
        self.restart = restart
        self.dry_run = dry_run
        self.heartbeat_seconds = max(5, heartbeat_seconds)

        campaign_id = str(self.manifest["campaign_id"])
        hardware_key = slug(
            "{}_{}_{}".format(cluster, hardware_type, node_id or "default")
        )
        self.state_dir = state_root / campaign_id / hardware_key
        self.log_dir = log_root / campaign_id / hardware_key
        self.result_dir = result_root / campaign_id / hardware_key
        for directory in (self.state_dir, self.log_dir, self.result_dir):
            directory.mkdir(parents=True, exist_ok=True)

        if not self.runner.is_file():
            raise CampaignValidationError(
                "Local experiment runner does not exist: {}".format(self.runner)
            )
        if not os.access(str(self.python), os.X_OK):
            raise CampaignValidationError(
                "Python is not executable: {}".format(self.python)
            )

    def _run_context(
        self,
        kind: str,
        context: Mapping[str, Any],
        conditions: Sequence[str],
    ) -> int:
        run_id = context_run_id(kind, context)
        state_dir = self.state_dir / "runs" / run_id
        log_dir = self.log_dir / run_id
        result_dir = self.result_dir / run_id
        for directory in (state_dir, log_dir, result_dir):
            directory.mkdir(parents=True, exist_ok=True)

        context_path = state_dir / "context.json"
        done_path = state_dir / "done.json"
        failed_path = state_dir / "failed.json"
        running_path = state_dir / "running.json"
        heartbeat_path = state_dir / "heartbeat.json"
        dry_run_path = state_dir / "dry-run.json"
        console_path = log_dir / "console.log"
        write_json_atomic(context_path, dict(context))

        if done_path.exists() and not self.restart:
            print("Skipping completed context: {}".format(run_id), flush=True)
            return 0

        if self.restart:
            for path in (
                done_path,
                failed_path,
                running_path,
                heartbeat_path,
                dry_run_path,
            ):
                path.unlink(missing_ok=True)

        command = [
            "zsh",
            str(self.runner),
            "context",
            str(context_path),
            ",".join(conditions),
        ]
        environment = dict(os.environ)
        environment.update(
            {
                "PYTHON": str(self.python),
                "LOG_DIR": str(log_dir),
                "ANTIVENOM_CONTEXT_ID": str(context["context_id"]),
                "ANTIVENOM_RUN_ID": run_id,
                "ANTIVENOM_RESULT_DIR": str(result_dir),
            }
        )

        print(
            "Starting {}: model={} dataset={} batch={} conditions={}".format(
                run_id,
                context["model"],
                context["dataset"],
                context["batch_size"],
                ",".join(conditions),
            ),
            flush=True,
        )
        print("  console log: {}".format(console_path), flush=True)
        _write_event(
            running_path,
            "running",
            run_id=run_id,
            pid=None,
            command=command,
            conditions=list(conditions),
            context_id=context["context_id"],
        )

        if self.dry_run:
            print("  dry-run command: {}".format(" ".join(command)), flush=True)
            _write_event(dry_run_path, "dry_run", run_id=run_id, exit_code=0)
            running_path.unlink(missing_ok=True)
            return 0

        started = time.monotonic()
        with console_path.open("a", encoding="utf-8") as console:
            console.write("\n=== {} {} ===\n".format(now(), run_id))
            console.flush()
            process = subprocess.Popen(
                command,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=console,
                stderr=subprocess.STDOUT,
            )
            _write_event(
                running_path,
                "running",
                run_id=run_id,
                pid=process.pid,
                command=command,
                conditions=list(conditions),
                context_id=context["context_id"],
            )

            while True:
                exit_code = process.poll()
                elapsed_seconds = int(time.monotonic() - started)
                _write_event(
                    heartbeat_path,
                    "running" if exit_code is None else "exited",
                    run_id=run_id,
                    pid=process.pid,
                    elapsed_seconds=elapsed_seconds,
                )
                if exit_code is not None:
                    break
                time.sleep(self.heartbeat_seconds)

        running_path.unlink(missing_ok=True)
        elapsed_seconds = int(time.monotonic() - started)
        if exit_code == 0:
            failed_path.unlink(missing_ok=True)
            _write_event(
                done_path,
                "done",
                run_id=run_id,
                exit_code=0,
                elapsed_seconds=elapsed_seconds,
                console_log=str(console_path),
                result_directory=str(result_dir),
            )
            print("Completed {} in {}s".format(run_id, elapsed_seconds), flush=True)
        else:
            done_path.unlink(missing_ok=True)
            _write_event(
                failed_path,
                "failed",
                run_id=run_id,
                exit_code=exit_code,
                elapsed_seconds=elapsed_seconds,
                console_log=str(console_path),
            )
            print(
                "Failed {} with exit code {}; see {}".format(
                    run_id, exit_code, console_path
                ),
                file=sys.stderr,
                flush=True,
            )
        return int(exit_code)

    def run(self) -> int:
        campaign_id = str(self.manifest["campaign_id"])
        summary_path = self.state_dir / "summary.json"
        _write_event(
            summary_path,
            "running",
            campaign_id=campaign_id,
            cluster=self.cluster,
            hardware_type=self.hardware_type,
            node_id=self.node_id,
            manifest=str(self.manifest_path),
            manifest_sha256=self.manifest_sha256,
        )

        calibration = self.manifest["calibration"]
        completed = 0
        if calibration.get("enabled", True):
            exit_code = self._run_context(
                "calibration",
                calibration["context"],
                calibration["conditions"],
            )
            if exit_code != 0:
                _write_event(
                    summary_path,
                    "failed",
                    campaign_id=campaign_id,
                    failed_run=context_run_id("calibration", calibration["context"]),
                    completed_contexts=completed,
                )
                return exit_code

        for context in self.assignment["contexts"]:
            exit_code = self._run_context("context", context, self.manifest["conditions"])
            if exit_code != 0:
                _write_event(
                    summary_path,
                    "failed",
                    campaign_id=campaign_id,
                    failed_run=context_run_id("context", context),
                    completed_contexts=completed,
                )
                return exit_code
            completed += 1

        _write_event(
            summary_path,
            "done" if not self.dry_run else "dry_run",
            campaign_id=campaign_id,
            completed_contexts=completed,
            expected_contexts=len(self.assignment["contexts"]),
            calibration_completed=bool(calibration.get("enabled", True)),
        )
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--cluster", required=True)
    parser.add_argument("--hardware-type", required=True)
    parser.add_argument("--node-id", default="")
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--state-root", required=True, type=Path)
    parser.add_argument("--log-root", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--heartbeat-seconds", type=int, default=30)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        executor = CampaignExecutor(
            manifest_path=arguments.manifest,
            cluster=arguments.cluster,
            hardware_type=arguments.hardware_type,
            node_id=arguments.node_id,
            runner=arguments.runner,
            python=arguments.python,
            state_root=arguments.state_root,
            log_root=arguments.log_root,
            result_root=arguments.result_root,
            restart=arguments.restart,
            dry_run=arguments.dry_run,
            heartbeat_seconds=arguments.heartbeat_seconds,
        )
        return executor.run()
    except CampaignValidationError as error:
        print("Campaign error: {}".format(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
