import argparse
import json
import socket
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import flwr as fl
from flwr.common import Metrics, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy

from dataset_preparation import get_dataloader, get_num_classes, prepare_dataset
from experiment_config import (
    DEFAULT_SERVER_ADDRESS,
    POISONING_ATTACK_METHODS,
    POISONING_METHOD_CLEAN,
    add_common_args,
    attack_name_for_poisoning_method,
    augment_from_args,
    condition_columns,
    parse_client_ids,
    select_poisoned_clients,
    set_all_seeds,
    yyyymmddhhmmss_log_path,
)
from hardware_logger import HardwareLogger, TrainingState
from metrics_logger import MetricsLogger
from models import get_model
from training_utils import evaluate_model, get_parameters, set_parameters


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total = sum(num_examples for num_examples, _ in metrics)
    if total == 0:
        return {}
    aggregated: Dict[str, float] = {}
    for num_examples, metric in metrics:
        for key, value in metric.items():
            if isinstance(value, (int, float)):
                aggregated[key] = aggregated.get(key, 0.0) + float(value) * num_examples / total
    return aggregated


class AnnotatedFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, state: TrainingState, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.state = state

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        self.state.update(round=server_round, phase="aggregation")
        aggregated = super().aggregate_fit(server_round, results, failures)
        self.state.update(round=server_round, phase="idle")
        return aggregated


def make_strategy(
    args: argparse.Namespace,
    poisoned_ids: List[str],
    state: TrainingState,
    metrics_logger: MetricsLogger,
) -> fl.server.strategy.Strategy:
    num_classes = get_num_classes(args.data_dir, dataset_name=args.dataset)
    augment = augment_from_args(args)
    resize = augment.get("resize", [64, 64])
    model = get_model(
        args.model,
        num_classes=num_classes,
        input_size=(int(resize[0]), int(resize[1])),
    )
    initial_parameters = fl.common.ndarrays_to_parameters(get_parameters(model))
    evaluation_augment = dict(augment)
    evaluation_augment["horizontal_flip"] = False
    clean_test_loader = get_dataloader(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        client_id="all",
        poisoning_method=POISONING_METHOD_CLEAN,
        split="test",
        augment=evaluation_augment,
        batch_size=args.batch_size,
        shuffle=False,
    )

    def fit_config(server_round: int) -> Dict[str, Scalar]:
        return {
            "round": server_round,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "local_epochs": args.local_epochs,
            "learning_rate": args.learning_rate,
            "poisoned_client_ids": ",".join(poisoned_ids),
            "poisoned_client_count": len(poisoned_ids),
            "poisoning_method": args.poisoning_method,
        }

    def evaluate_global_model(server_round: int, parameters: List, config: Dict[str, Scalar]):
        if server_round == 0:
            return None
        set_parameters(model, parameters)
        result = evaluate_model(
            model=model,
            data_loader=clean_test_loader,
            state=state,
            round_id=server_round,
            metrics_logger=metrics_logger,
            metric_event="global_clean_test_round",
            metric_split="clean_test",
            condition_overrides={
                "dataset_split": "test",
                "client_partition_id": "all",
            },
        )
        print(
            f"global_clean_test round={server_round} loss={result['loss']:.6f} "
            f"accuracy={result['accuracy']:.4f} examples={int(result['num_examples'])}"
        )
        return float(result["loss"]), {
            "accuracy": float(result["accuracy"]),
            "num_examples": int(result["num_examples"]),
        }

    min_clients = args.min_clients or args.num_clients
    return AnnotatedFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=0,
        min_available_clients=min_clients,
        initial_parameters=initial_parameters,
        on_fit_config_fn=fit_config,
        evaluate_fn=evaluate_global_model,
        fit_metrics_aggregation_fn=weighted_average,
        state=state,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--server-address", default=DEFAULT_SERVER_ADDRESS)
    parser.add_argument("--poisoned-client-count", type=int, default=1)
    parser.add_argument("--poisoned-client-ids", default="")
    parser.add_argument(
        "--poisoning-method",
        choices=[POISONING_METHOD_CLEAN, *POISONING_ATTACK_METHODS],
        default=POISONING_ATTACK_METHODS[0],
    )
    parser.add_argument("--min-clients", type=int, default=0)
    parser.add_argument("--server-log-hardware", action="store_true")
    parser.add_argument("--metadata-dir", default="logs/server_metadata")
    args = parser.parse_args()
    args.client_id = "server"
    args.device_id = args.device_id or "server"
    args.host = args.host or socket.gethostname()

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
    poisoned_ids = parse_client_ids(args.poisoned_client_ids)
    if not poisoned_ids:
        poisoned_ids = select_poisoned_clients(args.num_clients, args.poisoned_client_count, args.seed)

    metadata_dir = Path(args.metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / (
        f"fl_trial_{args.trial_id}_{args.poisoning_method}_poisoned_{len(poisoned_ids)}.json"
    )
    metadata_path.write_text(
        json.dumps(
            {
                "trial_id": args.trial_id,
                "seed": args.seed,
                "poisoning_method": args.poisoning_method,
                "run_role": args.run_role,
                "poisoned_client_ids": poisoned_ids,
                "min_clients": args.min_clients or args.num_clients,
            },
            indent=2,
        )
    )

    state = TrainingState(round=0, phase="idle")
    server_condition = POISONING_METHOD_CLEAN if not poisoned_ids else "mixed"
    condition = condition_columns(
        args=args,
        run_type="fl_server",
        poisoning_method=server_condition,
        is_poisoned_client=False,
        poisoned_client_count=len(poisoned_ids),
        poisoned_client_ids=poisoned_ids,
        poison_fraction=0.0,
        attack_name=(
            ""
            if args.poisoning_method == POISONING_METHOD_CLEAN
            else attack_name_for_poisoning_method(args.poisoning_method)
        ),
    )

    if args.server_log_hardware:
        with HardwareLogger(
            log_dir=args.log_dir,
            condition=condition,
            training_state=state,
            cpu_freq_sample_ms=args.cpu_freq_sample_ms,
        ) as logger:
            metrics_logger = MetricsLogger(
                path=logger.path.with_name(f"{logger.path.stem}_metrics.csv"),
                condition=condition,
            )
            strategy = make_strategy(args, poisoned_ids, state, metrics_logger)
            fl.server.start_server(
                server_address=args.server_address,
                config=fl.server.ServerConfig(num_rounds=args.num_rounds),
                strategy=strategy,
            )
    else:
        metrics_logger = MetricsLogger(
            path=yyyymmddhhmmss_log_path(args.log_dir, suffix="_metrics.csv"),
            condition=condition,
        )
        strategy = make_strategy(args, poisoned_ids, state, metrics_logger)
        fl.server.start_server(
            server_address=args.server_address,
            config=fl.server.ServerConfig(num_rounds=args.num_rounds),
            strategy=strategy,
        )


if __name__ == "__main__":
    main()
