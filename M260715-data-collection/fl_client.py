import argparse
from contextlib import nullcontext
from typing import Dict, List

import flwr as fl

from dataset_preparation import get_dataloader, get_num_classes, get_poison_fraction, prepare_dataset
from experiment_config import (
    DEFAULT_SERVER_ADDRESS,
    POISONING_ATTACK_METHODS,
    POISONING_METHOD_CLEAN,
    add_common_args,
    attack_name_for_poisoning_method,
    augment_from_args,
    condition_columns,
    get_device as get_device_config,
    parse_client_ids,
    set_all_seeds,
)
from hardware_logger import HardwareLogger, TrainingState
from metrics_logger import MetricsLogger
from models import get_model
from perf_logger import DEFAULT_PERF_EVENTS, PerfLogger, default_perf_events_for_host
from training_utils import evaluate_model, get_parameters, set_parameters, train_model


def parse_perf_events(value: str, *, host: str = "") -> List[str]:
    if not value:
        return default_perf_events_for_host(host)
    return [event.strip() for event in value.split(",") if event.strip()]


class TrashNetFlowerClient(fl.client.NumPyClient):
    def __init__(self, args: argparse.Namespace, state: TrainingState) -> None:
        self.args = args
        self.state = state
        self.augment = augment_from_args(args)
        self.num_classes = get_num_classes(args.data_dir, dataset_name=args.dataset)
        resize = self.augment.get("resize", [64, 64])
        self.model = get_model(
            args.model,
            num_classes=self.num_classes,
            input_size=(int(resize[0]), int(resize[1])),
        )
        self.metrics_logger = None

    def get_parameters(self, config: Dict[str, str]) -> List:
        return get_parameters(self.model)

    def fit(self, parameters: List, config: Dict[str, str]):
        current_round = int(config.get("round", 0))
        poisoned_ids = parse_client_ids(str(config.get("poisoned_client_ids", self.args.poisoned_client_ids)))
        configured_method = str(config.get("poisoning_method", self.args.poisoning_method))
        poisoning_method = configured_method if self.args.client_id in poisoned_ids else POISONING_METHOD_CLEAN
        set_all_seeds(int(config.get("seed", self.args.seed)))
        set_parameters(self.model, parameters)
        train_loader = get_dataloader(
            data_dir=self.args.data_dir,
            dataset_name=self.args.dataset,
            client_id=self.args.client_id,
            poisoning_method=poisoning_method,
            split=self.args.dataset_split,
            augment=self.augment,
            batch_size=int(config.get("batch_size", self.args.batch_size)),
            shuffle=True,
        )
        metrics = train_model(
            model=self.model,
            train_loader=train_loader,
            epochs=int(config.get("local_epochs", self.args.local_epochs)),
            learning_rate=float(config.get("learning_rate", self.args.learning_rate)),
            state=self.state,
            round_id=current_round,
            metrics_logger=self.metrics_logger,
        )
        metrics.update({"client_id": self.args.client_id, "poisoning_method": poisoning_method})
        return get_parameters(self.model), len(train_loader.dataset), metrics

    def evaluate(self, parameters: List, config: Dict[str, str]):
        current_round = int(config.get("round", 0))
        poisoned_ids = parse_client_ids(str(config.get("poisoned_client_ids", self.args.poisoned_client_ids)))
        configured_method = str(config.get("poisoning_method", self.args.poisoning_method))
        poisoning_method = configured_method if self.args.client_id in poisoned_ids else POISONING_METHOD_CLEAN
        set_parameters(self.model, parameters)
        eval_loader = get_dataloader(
            data_dir=self.args.data_dir,
            dataset_name=self.args.dataset,
            client_id=self.args.client_id,
            poisoning_method=poisoning_method,
            split=self.args.dataset_split,
            augment=self.augment,
            batch_size=self.args.batch_size,
            shuffle=False,
        )
        metrics = evaluate_model(
            model=self.model,
            data_loader=eval_loader,
            state=self.state,
            round_id=current_round,
            metrics_logger=self.metrics_logger,
        )
        return float(metrics["loss"]), len(eval_loader.dataset), {"accuracy": float(metrics["accuracy"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--server-address", default=DEFAULT_SERVER_ADDRESS)
    parser.add_argument("--poisoned-client-count", type=int, default=0)
    parser.add_argument("--poisoned-client-ids", default="")
    parser.add_argument("--poison-fraction", type=float, default=1.0)
    parser.add_argument(
        "--poisoning-method",
        choices=[POISONING_METHOD_CLEAN, *POISONING_ATTACK_METHODS],
        default=POISONING_ATTACK_METHODS[0],
    )
    perf_group = parser.add_mutually_exclusive_group()
    perf_group.add_argument("--enable-perf", dest="enable_perf", action="store_true")
    perf_group.add_argument("--disable-perf", dest="enable_perf", action="store_false")
    parser.set_defaults(enable_perf=True)
    parser.add_argument(
        "--perf-events",
        default="",
        help=(
            "Comma-separated event list. Empty selects the Raspberry Pi or Jetson profile "
            f"from --host. Raspberry Pi default: {','.join(DEFAULT_PERF_EVENTS)}"
        ),
    )
    parser.add_argument("--perf-fps", type=float, default=10.0)
    args = parser.parse_args()

    device = get_device_config(args.client_id)
    if not args.host:
        args.host = device.get("host", "")
    if not args.device_id:
        args.device_id = args.host
    poisoned_ids = parse_client_ids(args.poisoned_client_ids)
    is_poisoned = args.client_id in poisoned_ids
    poisoning_method = args.poisoning_method if is_poisoned else POISONING_METHOD_CLEAN

    set_all_seeds(args.seed)
    augment = augment_from_args(args)
    prepare_dataset(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        num_clients=args.num_clients,
        seed=args.seed,
        resize=augment.get("resize", [64, 64]),
        batch_size=args.batch_size,
    )

    state = TrainingState(round=0, epoch=0, batch_idx=0, phase="idle")
    poison_fraction = get_poison_fraction(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        client_id=args.client_id,
        poisoning_method=poisoning_method,
        split=args.dataset_split,
    )
    condition = condition_columns(
        args=args,
        run_type="fl_client",
        poisoning_method=poisoning_method,
        is_poisoned_client=is_poisoned,
        poisoned_client_count=args.poisoned_client_count,
        poisoned_client_ids=poisoned_ids,
        poison_fraction=poison_fraction if is_poisoned else 0.0,
        attack_name=attack_name_for_poisoning_method(poisoning_method) if is_poisoned else "",
    )
    client = TrashNetFlowerClient(args, state)
    with HardwareLogger(
        log_dir=args.log_dir,
        condition=condition,
        training_state=state,
        cpu_freq_sample_ms=args.cpu_freq_sample_ms,
    ) as logger:
        perf_context = nullcontext()
        if args.enable_perf:
            perf_context = PerfLogger(
                log_dir=args.log_dir,
                condition=condition,
                training_state=state,
                path=logger.path.with_name(f"{logger.path.stem}_perf.csv"),
                events=parse_perf_events(args.perf_events, host=args.host),
                fps=args.perf_fps,
            )
        client.metrics_logger = MetricsLogger(
            path=logger.path.with_name(f"{logger.path.stem}_metrics.csv"),
            condition=condition,
        )
        with perf_context:
            fl.client.start_numpy_client(server_address=args.server_address, client=client)


if __name__ == "__main__":
    main()
