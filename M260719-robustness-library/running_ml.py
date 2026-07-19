import os

# This experiment measures CPU behavior. Set this before importing TensorFlow.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
from contextlib import nullcontext
from typing import List

from dataset_preparation import (
    get_dataset,
    get_num_classes,
    get_poison_fraction,
    validate_prepared_dataset,
)
from experiment_config import (
    DEFAULT_LOG_DIR,
    POISONING_METHODS,
    POISONING_METHOD_CLEAN,
    add_common_args,
    attack_name_for_poisoning_method,
    augment_from_args,
    condition_columns,
    get_device,
    parse_poisoning_methods,
    set_all_seeds,
)
from hardware_logger import HardwareLogger, TrainingState
from metrics_logger import MetricsLogger
from models import get_model
from perf_logger import DEFAULT_PERF_EVENTS, PerfLogger, default_perf_events_for_host
from training_utils import evaluate_model, train_model


def parse_perf_events(value: str, *, host: str = "") -> List[str]:
    if not value:
        return default_perf_events_for_host(host)
    return [event.strip() for event in value.split(",") if event.strip()]


def run_one_local(args: argparse.Namespace, poisoning_method: str) -> str:
    set_all_seeds(args.seed)
    augment = augment_from_args(args)
    num_classes = get_num_classes(args.data_dir, args.dataset)
    model = get_model(
        args.model,
        num_classes=num_classes,
        input_size=tuple(augment["resize"]),
    )
    train_dataset = get_dataset(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        client_id=args.client_id,
        poisoning_method=poisoning_method,
        split=args.dataset_split,
        augment=augment,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    evaluation_augment = dict(augment)
    evaluation_augment["horizontal_flip"] = False
    clean_test_dataset = get_dataset(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        client_id="all",
        poisoning_method=POISONING_METHOD_CLEAN,
        split="test",
        augment=evaluation_augment,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )
    state = TrainingState(round=0, epoch=0, batch_idx=0, phase="idle")
    poison_fraction = get_poison_fraction(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        client_id=args.client_id,
        poisoning_method=poisoning_method,
        split=args.dataset_split,
    )
    is_poisoned = poisoning_method != POISONING_METHOD_CLEAN
    condition = condition_columns(
        args=args,
        run_type="local_ml",
        poisoning_method=poisoning_method,
        is_poisoned_client=is_poisoned,
        poisoned_client_count=1 if is_poisoned else 0,
        poisoned_client_ids=[args.client_id] if is_poisoned else [],
        poison_fraction=poison_fraction,
        attack_name=attack_name_for_poisoning_method(poisoning_method),
    )

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
        metrics_logger = MetricsLogger(
            path=logger.path.with_name(f"{logger.path.stem}_metrics.csv"),
            condition=condition,
        )

        with perf_context:
            def evaluate_clean_test(epoch: int) -> None:
                result = evaluate_model(
                    model=model,
                    data_dataset=clean_test_dataset,
                    state=state,
                    round_id=0,
                    metrics_logger=metrics_logger,
                    metric_event="clean_test_epoch",
                    metric_split="clean_test",
                    condition_overrides={
                        "dataset_split": "test",
                        "client_partition_id": "all",
                    },
                )
                print(
                    f"clean_test epoch={epoch} loss={result['loss']:.6f} "
                    f"accuracy={result['accuracy']:.4f} "
                    f"examples={int(result['num_examples'])}",
                    flush=True,
                )

            result = train_model(
                model=model,
                train_dataset=train_dataset,
                epochs=args.local_epochs,
                learning_rate=args.learning_rate,
                state=state,
                round_id=0,
                metrics_logger=metrics_logger,
                epoch_end_callback=evaluate_clean_test,
            )
            print(
                f"finished method={poisoning_method} "
                f"loss={result['train_loss']:.6f} "
                f"accuracy={result['train_accuracy']:.4f}",
                flush=True,
            )
    return str(logger.path)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.set_defaults(log_dir=DEFAULT_LOG_DIR)
    parser.add_argument(
        "--poisoning-method",
        default="all",
        help=f"One method, a comma-separated subset, or all: {','.join(POISONING_METHODS)}",
    )
    parser.add_argument("--trials", type=int, default=1)
    perf_group = parser.add_mutually_exclusive_group()
    perf_group.add_argument("--enable-perf", dest="enable_perf", action="store_true")
    perf_group.add_argument("--disable-perf", dest="enable_perf", action="store_false")
    parser.set_defaults(enable_perf=True)
    parser.add_argument("--perf-events", default="")
    parser.add_argument("--perf-fps", type=float, default=10.0)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0 or args.local_epochs <= 0 or args.trials <= 0:
        parser.error("batch-size, local-epochs, and trials must be positive")

    device = get_device(args.client_id)
    if not args.host:
        args.host = device.get("host", "")
    if not args.device_id:
        args.device_id = args.host

    methods = parse_poisoning_methods(args.poisoning_method)
    validate_prepared_dataset(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        client_id=args.client_id,
        poisoning_methods=methods,
    )
    if args.validate_only:
        print(
            f"validated dataset={args.dataset} client={args.client_id} "
            f"methods={','.join(methods)}"
        )
        return

    base_seed = args.seed
    for trial in range(args.trials):
        args.trial_id = f"trial_{trial}"
        args.seed = base_seed + trial
        for method in methods:
            run_one_local(args, method)


if __name__ == "__main__":
    main()
