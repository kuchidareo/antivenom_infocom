import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


DATASET_NAME = os.environ.get("DATASET_NAME", "kuchidareo/small_trashnet")
DEFAULT_DATA_DIR = os.environ.get("DATA_DIR", "../iid-data")
DEFAULT_LOG_DIR = "logs/local_ml"
DEFAULT_MODEL = "simple_cnn"
DEFAULT_FRAMEWORK = "tensorflow"
DEFAULT_BATCH_SIZE = 16
DEFAULT_LOCAL_EPOCHS = 10
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_NUM_CLIENTS = 10
DEFAULT_PARTITION_METHOD = "iid"
DEFAULT_NONIID_ALPHA = 0.3
DEFAULT_BASE_SEED = 260626
DEFAULT_CPU_FREQ_SAMPLE_MS = 1.0
DEFAULT_REMOTE_PROJECT_DIR = (
    "/home/rasheed/kuchida/antivenom_infocom/M260719-robustness-library"
)
DEFAULT_REMOTE_PYTHON = f"{DEFAULT_REMOTE_PROJECT_DIR}/.venv/bin/python"
DEFAULT_BOOTSTRAP_PYTHON = "/home/rasheed/kuchida/antivenom_infocom/venv/bin/python"
DEFAULT_SSH_USER = "rasheed"

POISONING_METHOD_CLEAN = "clean"
POISONING_METHOD_UNLEARNABLE_EXAMPLES = "unlearnable_examples"
POISONING_METHOD_RANDOM_LABEL_FLIPPING = "random_label_flipping"
POISONING_METHOD_TARGET_LABEL_FLIPPING = "target_label_flipping"
POISONING_METHOD_AVAILABILITY_SHORTCUTS = "availability_shortcuts"
POISONING_METHODS = [
    POISONING_METHOD_CLEAN,
    POISONING_METHOD_UNLEARNABLE_EXAMPLES,
    POISONING_METHOD_RANDOM_LABEL_FLIPPING,
    POISONING_METHOD_TARGET_LABEL_FLIPPING,
    POISONING_METHOD_AVAILABILITY_SHORTCUTS,
]

ATTACK_NAME_BY_POISONING_METHOD = {
    POISONING_METHOD_CLEAN: "",
    POISONING_METHOD_UNLEARNABLE_EXAMPLES: "unlearnable_examples_min_min_samplewise",
    POISONING_METHOD_RANDOM_LABEL_FLIPPING: "random_label_flipping_1_over_6",
    POISONING_METHOD_TARGET_LABEL_FLIPPING: "target_label_flipping_5_to_3",
    POISONING_METHOD_AVAILABILITY_SHORTCUTS: "availability_shortcuts_synthetic_classwise",
}

DEVICES = [
    {"client_id": "client_1", "host": "192.168.0.141"},
    {"client_id": "client_2", "host": "192.168.0.142"},
]

DEFAULT_AUGMENT = {
    "enabled": True,
    "resize": [224, 224],
    "horizontal_flip": True,
    "normalize": True,
}

CONDITION_COLUMNS = [
    "experiment_id",
    "run_type",
    "run_role",
    "framework",
    "device_id",
    "client_id",
    "host",
    "trial_id",
    "seed",
    "dataset",
    "dataset_split",
    "partition_method",
    "noniid_alpha",
    "num_clients",
    "client_partition_id",
    "model",
    "batch_size",
    "local_epochs",
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
    "background_workload_enabled",
    "background_workload_group",
    "background_workload_profile",
    "cpu_freq_sample_ms",
]

CSV_COLUMNS = [
    "timestamp",
    "timestamp_unix",
    *CONDITION_COLUMNS,
    "round",
    "epoch",
    "batch_idx",
    "phase",
    "system_cpu_core_0",
    "system_cpu_core_1",
    "system_cpu_core_2",
    "system_cpu_core_3",
    "system_cpu_freq_core_0",
    "system_cpu_freq_core_1",
    "system_cpu_freq_core_2",
    "system_cpu_freq_core_3",
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


def parse_json_dict(value: Optional[str]) -> Dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("Expected a JSON object.")
    return parsed


def get_device(client_id: str) -> Dict[str, str]:
    for device in DEVICES:
        if device["client_id"] == client_id:
            return device
    return {"client_id": client_id, "host": ""}


def parse_poisoning_methods(value: Optional[str]) -> List[str]:
    if value is None or value.strip() in {"", "all"}:
        return list(POISONING_METHODS)
    methods = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [method for method in methods if method not in POISONING_METHODS]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unknown poisoning method(s): {','.join(invalid)}. "
            f"Allowed: {','.join(POISONING_METHODS)}"
        )
    return list(dict.fromkeys(methods))


def attack_name_for_poisoning_method(poisoning_method: str) -> str:
    return ATTACK_NAME_BY_POISONING_METHOD.get(poisoning_method, poisoning_method)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def yyyymmddhhmmss_log_path(log_dir: str, suffix: str = ".csv") -> Path:
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
    parser.add_argument("--model", choices=[DEFAULT_MODEL], default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--local-epochs", type=int, default=DEFAULT_LOCAL_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--num-clients", type=int, default=DEFAULT_NUM_CLIENTS)
    parser.add_argument(
        "--partition-method",
        choices=["iid"],
        default=DEFAULT_PARTITION_METHOD,
    )
    parser.add_argument("--noniid-alpha", type=float, default=DEFAULT_NONIID_ALPHA)
    parser.add_argument("--trial-id", default="trial_0")
    parser.add_argument("--seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--device-id", default="")
    parser.add_argument("--client-id", default="client_1")
    parser.add_argument("--host", default="")
    parser.add_argument("--augment", default=json.dumps(DEFAULT_AUGMENT))
    parser.add_argument("--dataset-split", choices=["train"], default="train")
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--run-role", default="analysis")
    parser.add_argument(
        "--cpu-freq-sample-ms",
        type=float,
        default=DEFAULT_CPU_FREQ_SAMPLE_MS,
    )


def augment_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    augment = dict(DEFAULT_AUGMENT)
    augment.update(parse_json_dict(args.augment))
    resize = augment.get("resize")
    if not isinstance(resize, list) or len(resize) != 2 or any(int(v) <= 0 for v in resize):
        raise ValueError("augment.resize must contain two positive integers")
    augment["resize"] = [int(v) for v in resize]
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
    client_id = args.client_id
    host = args.host or get_device(client_id).get("host", "")
    experiment_id = args.experiment_id or (
        f"tensorflow_{run_type}_{poisoning_method}_{args.trial_id}_{client_id}"
    )
    return {
        "experiment_id": experiment_id,
        "run_type": run_type,
        "run_role": args.run_role,
        "framework": DEFAULT_FRAMEWORK,
        "device_id": args.device_id or host,
        "client_id": client_id,
        "host": host,
        "trial_id": args.trial_id,
        "seed": args.seed,
        "dataset": args.dataset,
        "dataset_split": args.dataset_split,
        "partition_method": args.partition_method,
        "noniid_alpha": "",
        "num_clients": args.num_clients,
        "client_partition_id": client_id,
        "model": args.model,
        "batch_size": args.batch_size,
        "local_epochs": args.local_epochs,
        "learning_rate": args.learning_rate,
        "augment_enabled": bool(augment.get("enabled", True)),
        "augment_resize": "x".join(str(v) for v in augment["resize"]),
        "augment_horizontal_flip": bool(augment.get("horizontal_flip", False)),
        "augment_normalize": bool(augment.get("normalize", False)),
        "poisoning_method": poisoning_method,
        "is_poisoned_client": bool(is_poisoned_client),
        "poisoned_client_count": poisoned_client_count,
        "poisoned_client_ids": ",".join(poisoned_client_ids or []),
        "poison_fraction": poison_fraction,
        "attack_name": attack_name,
        "background_workload_enabled": False,
        "background_workload_group": "",
        "background_workload_profile": "",
        "cpu_freq_sample_ms": args.cpu_freq_sample_ms,
    }
