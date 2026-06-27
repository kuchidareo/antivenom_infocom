import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


DATASET_NAME = "kuchidareo/small_trashnet"
DEFAULT_DATA_DIR = "data"
DEFAULT_LOG_DIR = "logs"
DEFAULT_LOCAL_ML_LOG_DIR = "logs/local_ml"
DEFAULT_FL_LOG_DIR = "logs/fl"
DEFAULT_MODEL = "simple_cnn"
DEFAULT_BATCH_SIZE = 16
DEFAULT_LOCAL_EPOCHS = 1
DEFAULT_NUM_ROUNDS = 10
DEFAULT_LOCAL_ML_EPOCHS = 10
DEFAULT_LOCAL_ML_GLOBAL_CLEAN_REFERENCE_TRIALS = 5
DEFAULT_LOCAL_ML_ANALYSIS_TRIALS = 5
DEFAULT_LOCAL_ML_TRIALS = DEFAULT_LOCAL_ML_ANALYSIS_TRIALS
DEFAULT_CLEAN_REFERENCE_TRIALS = DEFAULT_LOCAL_ML_GLOBAL_CLEAN_REFERENCE_TRIALS
DEFAULT_OT_REFERENCE_TRIAL_IDS = [f"reference_{idx}" for idx in range(DEFAULT_CLEAN_REFERENCE_TRIALS)]
DEFAULT_FL_LOCAL_EPOCHS = 1
DEFAULT_FL_NUM_ROUNDS = 10
DEFAULT_FL_TRIALS = 5
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_NUM_CLIENTS = 10
DEFAULT_BASE_SEED = 260626
DEFAULT_SERVER_ADDRESS = "192.168.0.110:8080"
DEFAULT_FL_SERVER_BIND_ADDRESS = "0.0.0.0:8080"
DEFAULT_FL_CLIENT_SERVER_ADDRESS = "192.168.0.110:8080"
DEFAULT_REMOTE_PROJECT_DIR = "/home/rasheed/kuchida/antivenom_infocom/260626-data-collection-v1"
DEFAULT_REMOTE_PYTHON = "/home/rasheed/kuchida/antivenom_infocom/venv/bin/python"
DEFAULT_SSH_USER = "rasheed"

DEVICES = [
    {"client_id": "client_0", "host": "192.168.0.112"},
    {"client_id": "client_1", "host": "192.168.0.113"},
    {"client_id": "client_2", "host": "192.168.0.114"},
    {"client_id": "client_3", "host": "192.168.0.115"},
    {"client_id": "client_4", "host": "192.168.0.116"},
    {"client_id": "client_5", "host": "192.168.0.117"},
    {"client_id": "client_6", "host": "192.168.0.118"},
    {"client_id": "client_7", "host": "192.168.0.119"},
    {"client_id": "client_8", "host": "192.168.0.120"},
    {"client_id": "client_9", "host": "192.168.0.121"},
]

DEFAULT_AUGMENT = {
    "enabled": True,
    "resize": [224, 224],
    "horizontal_flip": True,
    "normalize": True,
}

CSV_COLUMNS = [
    "timestamp",
    "timestamp_unix",
    "experiment_id",
    "run_type",
    "run_role",
    "device_id",
    "client_id",
    "host",
    "trial_id",
    "seed",
    "dataset",
    "dataset_split",
    "partition_method",
    "num_clients",
    "client_partition_id",
    "model",
    "batch_size",
    "local_epochs",
    "num_rounds",
    "learning_rate",
    "augment_enabled",
    "augment_resize",
    "augment_horizontal_flip",
    "augment_normalize",
    "poisoning_method",
    "is_poisoned_client",
    "poisoned_client_count",
    "poisoned_client_ids",
    "poison_fraction",
    "attack_name",
    "round",
    "epoch",
    "batch_idx",
    "phase",
    "system_cpu_core_0",
    "system_cpu_core_1",
    "system_cpu_core_2",
    "system_cpu_core_3",
    "system_memory_percent",
    "system_memory_used",
    "system_memory_available",
    "process_cpu_percent",
    "process_memory_rss",
    "process_memory_vms",
    "process_memory_percent",
    "process_ctx_switches_voluntary",
    "process_ctx_switches_involuntary",
    "process_minor_faults",
]

CONDITION_COLUMNS = [
    "experiment_id",
    "run_type",
    "run_role",
    "device_id",
    "client_id",
    "host",
    "trial_id",
    "seed",
    "dataset",
    "dataset_split",
    "partition_method",
    "num_clients",
    "client_partition_id",
    "model",
    "batch_size",
    "local_epochs",
    "num_rounds",
    "learning_rate",
    "augment_enabled",
    "augment_resize",
    "augment_horizontal_flip",
    "augment_normalize",
    "poisoning_method",
    "is_poisoned_client",
    "poisoned_client_count",
    "poisoned_client_ids",
    "poison_fraction",
    "attack_name",
]

METRIC_COLUMNS = [
    "timestamp",
    "timestamp_unix",
    *CONDITION_COLUMNS,
    "round",
    "epoch",
    "batch_idx",
    "phase",
    "metric_event",
    "metric_split",
    "loss",
    "accuracy",
    "num_examples",
]


def parse_json_dict(value: Optional[str], default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if value is None or value == "":
        return dict(default or {})
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("Expected a JSON object.")
    return parsed


def parse_client_ids(value: Optional[str]) -> List[str]:
    if value is None or value == "":
        return []
    if value.strip().startswith("["):
        parsed = json.loads(value)
        return [str(item) for item in parsed]
    return [item.strip() for item in value.split(",") if item.strip()]


def client_index(client_id: str) -> int:
    if client_id.startswith("client_"):
        return int(client_id.split("_", 1)[1])
    return int(client_id)


def get_device(client_id: str) -> Dict[str, str]:
    for device in DEVICES:
        if device["client_id"] == client_id:
            return device
    return {"client_id": client_id, "host": ""}


def select_poisoned_clients(num_clients: int, poisoned_client_count: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    selected = rng.sample(range(num_clients), poisoned_client_count)
    return [f"client_{idx}" for idx in sorted(selected)]


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def yyyymmddhhmmss_log_path(log_dir: str, suffix: str = ".csv") -> Path:
    """Return a collision-resistant path using yyyymmddhhmmss format."""
    base = datetime.now().strftime("%Y%m%d%H%M%S")
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / f"{base}{suffix}"
    if not candidate.exists():
        return candidate
    for idx in range(1, 1000):
        candidate = directory / f"{base}_{idx:03d}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create a unique log filename in {directory}")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--local-epochs", type=int, default=DEFAULT_LOCAL_EPOCHS)
    parser.add_argument("--num-rounds", type=int, default=DEFAULT_NUM_ROUNDS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--num-clients", type=int, default=DEFAULT_NUM_CLIENTS)
    parser.add_argument("--trial-id", default="trial_0")
    parser.add_argument("--seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--device-id", default="")
    parser.add_argument("--client-id", default="client_0")
    parser.add_argument("--host", default="")
    parser.add_argument("--augment", default=json.dumps(DEFAULT_AUGMENT))
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--experiment-id", default="")


def augment_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    augment = dict(DEFAULT_AUGMENT)
    augment.update(parse_json_dict(args.augment))
    return augment


def condition_columns(
    *,
    args: argparse.Namespace,
    run_type: str,
    poisoning_method: str,
    is_poisoned_client: bool,
    poisoned_client_count: int = 0,
    poisoned_client_ids: Optional[Iterable[str]] = None,
    poison_fraction: float = 0.0,
    attack_name: str = "",
) -> Dict[str, Any]:
    augment = augment_from_args(args)
    client_id = getattr(args, "client_id", "")
    host = getattr(args, "host", "") or get_device(client_id).get("host", "")
    trial_id = getattr(args, "trial_id", "trial_0")
    experiment_id = getattr(args, "experiment_id", "") or (
        f"{run_type}_{poisoning_method}_{trial_id}_{client_id}"
    )
    poisoned_ids = list(poisoned_client_ids or [])
    return {
        "experiment_id": experiment_id,
        "run_type": run_type,
        "run_role": getattr(args, "run_role", ""),
        "device_id": getattr(args, "device_id", "") or host,
        "client_id": client_id,
        "host": host,
        "trial_id": trial_id,
        "seed": getattr(args, "seed", DEFAULT_BASE_SEED),
        "dataset": getattr(args, "dataset", DATASET_NAME),
        "dataset_split": getattr(args, "dataset_split", "train"),
        "partition_method": "iid",
        "num_clients": getattr(args, "num_clients", DEFAULT_NUM_CLIENTS),
        "client_partition_id": client_id,
        "model": getattr(args, "model", DEFAULT_MODEL),
        "batch_size": getattr(args, "batch_size", DEFAULT_BATCH_SIZE),
        "local_epochs": getattr(args, "local_epochs", DEFAULT_LOCAL_EPOCHS),
        "num_rounds": getattr(args, "num_rounds", DEFAULT_NUM_ROUNDS),
        "learning_rate": getattr(args, "learning_rate", DEFAULT_LEARNING_RATE),
        "augment_enabled": bool(augment.get("enabled", True)),
        "augment_resize": "x".join(str(v) for v in augment.get("resize", [])),
        "augment_horizontal_flip": bool(augment.get("horizontal_flip", False)),
        "augment_normalize": bool(augment.get("normalize", False)),
        "poisoning_method": poisoning_method,
        "is_poisoned_client": bool(is_poisoned_client),
        "poisoned_client_count": poisoned_client_count,
        "poisoned_client_ids": ",".join(poisoned_ids),
        "poison_fraction": poison_fraction,
        "attack_name": attack_name,
    }
