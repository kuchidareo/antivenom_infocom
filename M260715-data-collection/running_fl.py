import argparse
import random
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List

from dataset_preparation import prepare_dataset
from experiment_config import (
    DEFAULT_FL_CLIENT_SERVER_ADDRESS,
    DEFAULT_FL_LOG_DIR,
    DEFAULT_FL_LOCAL_EPOCHS,
    DEFAULT_FL_NUM_ROUNDS,
    DEFAULT_FL_SERVER_BIND_ADDRESS,
    DEFAULT_FL_TRIALS,
    DEFAULT_FL_POISONING_METHODS,
    DEFAULT_REMOTE_PYTHON,
    DEFAULT_REMOTE_PROJECT_DIR,
    DEFAULT_SSH_USER,
    DEVICES,
    POISONING_ATTACK_METHODS,
    POISONING_METHOD_CLEAN,
    add_common_args,
    augment_from_args,
    parse_poisoning_methods,
)


def _arg(name: str, value: object) -> List[str]:
    return [name, str(value)]


def _active_devices(args: argparse.Namespace) -> List[dict]:
    configured_devices = DEVICES[: args.num_clients]
    requested = {
        client_id.strip()
        for client_id in args.active_client_ids.split(",")
        if client_id.strip()
    }
    devices = [
        device
        for device in configured_devices
        if not requested or device["client_id"] in requested
    ]
    unknown = requested - {device["client_id"] for device in devices}
    if unknown:
        raise ValueError(f"Unknown active client IDs: {','.join(sorted(unknown))}")
    if not devices:
        raise ValueError("No active FL clients were selected.")
    return devices


def _select_poisoned_clients(active_devices: List[dict], count: int, seed: int) -> List[str]:
    if count < 0 or count > len(active_devices):
        raise ValueError(
            f"poisoned_client_count={count} is invalid for {len(active_devices)} active clients."
        )
    rng = random.Random(seed)
    selected = rng.sample([device["client_id"] for device in active_devices], count)
    return sorted(selected, key=lambda client_id: int(client_id.split("_")[-1]))


def _client_command(args: argparse.Namespace, device: dict, seed: int, poisoned_ids: List[str]) -> str:
    client_args = [
        args.remote_python,
        "fl_client.py",
        *_arg("--server-address", args.client_server_address),
        *_arg("--dataset", args.dataset),
        *_arg("--data-dir", args.data_dir),
        *_arg("--log-dir", args.log_dir),
        *_arg("--model", args.model),
        *_arg("--batch-size", args.batch_size),
        *_arg("--local-epochs", args.local_epochs),
        *_arg("--num-rounds", args.num_rounds),
        *_arg("--learning-rate", args.learning_rate),
        *_arg("--num-clients", args.num_clients),
        *_arg("--trial-id", args.trial_id),
        *_arg("--seed", seed),
        *_arg("--client-id", device["client_id"]),
        *_arg("--device-id", device["host"]),
        *_arg("--host", device["host"]),
        *_arg("--augment", args.augment),
        *_arg("--poisoned-client-count", len(poisoned_ids)),
        *_arg("--poisoned-client-ids", ",".join(poisoned_ids)),
        *_arg("--poisoning-method", args.poisoning_method),
        *_arg("--run-role", args.run_role),
        *_arg("--cpu-freq-sample-ms", args.cpu_freq_sample_ms),
        *_arg("--perf-fps", args.perf_fps),
        *_arg("--perf-events", args.perf_events),
    ]
    client_args.append("--enable-perf" if args.enable_perf else "--disable-perf")
    quoted = " ".join(shlex.quote(part) for part in client_args)
    return f"cd {shlex.quote(args.remote_project_dir)} && {quoted}"


def _start_server(args: argparse.Namespace, seed: int, poisoned_ids: List[str]) -> subprocess.Popen:
    command = [
        sys.executable,
        "fl_server.py",
        *_arg("--server-address", args.server_address),
        *_arg("--dataset", args.dataset),
        *_arg("--data-dir", args.data_dir),
        *_arg("--log-dir", args.log_dir),
        *_arg("--model", args.model),
        *_arg("--batch-size", args.batch_size),
        *_arg("--local-epochs", args.local_epochs),
        *_arg("--num-rounds", args.num_rounds),
        *_arg("--learning-rate", args.learning_rate),
        *_arg("--num-clients", args.num_clients),
        *_arg("--trial-id", args.trial_id),
        *_arg("--seed", seed),
        *_arg("--augment", args.augment),
        *_arg("--poisoned-client-count", len(poisoned_ids)),
        *_arg("--poisoned-client-ids", ",".join(poisoned_ids)),
        *_arg("--poisoning-method", args.poisoning_method),
        *_arg("--run-role", args.run_role),
        *_arg("--min-clients", len(_active_devices(args))),
        *_arg("--cpu-freq-sample-ms", args.cpu_freq_sample_ms),
    ]
    if args.server_log_hardware:
        command.append("--server-log-hardware")
    return subprocess.Popen(command, cwd=Path(__file__).resolve().parent)


def _start_clients(args: argparse.Namespace, seed: int, poisoned_ids: List[str]) -> List[subprocess.Popen]:
    processes = []
    for device in _active_devices(args):
        remote_command = _client_command(args, device, seed, poisoned_ids)
        target = f"{args.ssh_user}@{device['host']}" if args.ssh_user else device["host"]
        command = ["ssh", target, remote_command]
        if args.ssh_password:
            command = [
                "sshpass",
                "-p",
                args.ssh_password,
                "ssh",
                "-o",
                "StrictHostKeyChecking=accept-new",
                target,
                remote_command,
            ]
        if args.dry_run:
            redacted = ["******" if part == args.ssh_password else part for part in command]
            print(" ".join(shlex.quote(part) for part in redacted))
            continue
        processes.append(subprocess.Popen(command))
    return processes


def _terminate(processes: List[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.time() + 10
    for process in processes:
        timeout = max(0, deadline - time.time())
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()


def run_trial(args: argparse.Namespace, poisoned_client_count: int, poisoning_method: str, trial_id: int) -> None:
    seed = args.seed + trial_id
    active_devices = _active_devices(args)
    poisoned_ids = _select_poisoned_clients(active_devices, poisoned_client_count, seed)
    args.trial_id = trial_id
    args.poisoning_method = poisoning_method
    args.run_role = "clean_baseline" if poisoning_method == POISONING_METHOD_CLEAN else "attack_analysis"
    print(
        f"trial={trial_id} seed={seed} poisoning_method={poisoning_method} "
        f"run_role={args.run_role} "
        f"active_clients={[device['client_id'] for device in active_devices]} "
        f"poisoned_client_count={poisoned_client_count} ids={poisoned_ids}"
    )
    if args.dry_run:
        _start_clients(args, seed, poisoned_ids)
        return

    server = _start_server(args, seed, poisoned_ids)
    clients: List[subprocess.Popen] = []
    try:
        time.sleep(args.client_start_delay)
        clients = _start_clients(args, seed, poisoned_ids)
        server.wait()
    finally:
        _terminate(clients)
        if server.poll() is None:
            server.send_signal(signal.SIGINT)
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.set_defaults(
        local_epochs=DEFAULT_FL_LOCAL_EPOCHS,
        num_rounds=DEFAULT_FL_NUM_ROUNDS,
        log_dir=DEFAULT_FL_LOG_DIR,
    )
    parser.add_argument("--server-address", default=DEFAULT_FL_SERVER_BIND_ADDRESS)
    parser.add_argument("--client-server-address", default=DEFAULT_FL_CLIENT_SERVER_ADDRESS)
    parser.add_argument("--remote-project-dir", default=DEFAULT_REMOTE_PROJECT_DIR)
    parser.add_argument("--remote-python", default=DEFAULT_REMOTE_PYTHON)
    parser.add_argument("--ssh-user", default=DEFAULT_SSH_USER)
    parser.add_argument("--ssh-password", default="")
    parser.add_argument(
        "--active-client-ids",
        default="",
        help="Comma-separated reachable client IDs. Empty uses every configured device.",
    )
    parser.add_argument("--poisoned-client-counts", default="1,4")
    parser.add_argument(
        "--poisoning-methods",
        default=",".join(DEFAULT_FL_POISONING_METHODS),
        help=(
            "Comma-separated clean/attack methods. Clean runs once with zero poisoned clients. "
            f"Allowed: {POISONING_METHOD_CLEAN},{','.join(POISONING_ATTACK_METHODS)}"
        ),
    )
    parser.add_argument("--trials", type=int, default=DEFAULT_FL_TRIALS)
    parser.add_argument("--client-start-delay", type=float, default=3.0)
    parser.add_argument("--server-log-hardware", action="store_true")
    perf_group = parser.add_mutually_exclusive_group()
    perf_group.add_argument("--enable-perf", dest="enable_perf", action="store_true")
    perf_group.add_argument("--disable-perf", dest="enable_perf", action="store_false")
    parser.set_defaults(enable_perf=True)
    parser.add_argument(
        "--perf-events",
        default="",
        help="Client perf events. Empty lets each client select its host-specific profile.",
    )
    parser.add_argument("--perf-fps", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    augment = augment_from_args(args)
    if not args.dry_run:
        prepare_dataset(
            data_dir=args.data_dir,
            dataset_name=args.dataset,
            num_clients=args.num_clients,
            seed=args.seed,
            resize=augment.get("resize", [64, 64]),
            batch_size=args.batch_size,
        )
    counts = [int(item.strip()) for item in args.poisoned_client_counts.split(",") if item.strip()]
    poisoning_methods = parse_poisoning_methods(args.poisoning_methods, include_clean=True)
    for poisoning_method in poisoning_methods:
        method_counts = [0] if poisoning_method == POISONING_METHOD_CLEAN else counts
        for count in method_counts:
            for trial_id in range(args.trials):
                run_trial(args, count, poisoning_method, trial_id)


if __name__ == "__main__":
    main()
