import argparse

from dataset_preparation import get_dataloader, get_num_classes, get_poison_fraction, prepare_dataset
from experiment_config import (
    DEFAULT_LOCAL_ML_LOG_DIR,
    DEFAULT_LOCAL_ML_ANALYSIS_TRIALS,
    DEFAULT_LOCAL_ML_EPOCHS,
    DEFAULT_LOCAL_ML_GLOBAL_CLEAN_REFERENCE_TRIALS,
    DEFAULT_LOCAL_ML_POISONING_METHODS,
    POISONING_METHODS,
    POISONING_METHOD_CLEAN,
    add_common_args,
    attack_name_for_poisoning_method,
    augment_from_args,
    condition_columns,
    get_device as get_device_config,
    parse_poisoning_methods,
    set_all_seeds,
)
from hardware_logger import HardwareLogger, TrainingState
from metrics_logger import MetricsLogger
from models import get_model
from perf_logger import DEFAULT_PERF_EVENTS, PerfLogger
from training_utils import evaluate_model, train_model


def run_one_local(args: argparse.Namespace, poisoning_method: str) -> str:
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
    num_classes = get_num_classes(args.data_dir, dataset_name=args.dataset)
    resize = augment.get("resize", [64, 64])
    input_size = (int(resize[0]), int(resize[1]))
    model = get_model(
        args.model,
        num_classes=num_classes,
        input_size=input_size,
        batch_size=args.batch_size,
        model_depth=args.model_depth,
        width_multiplier=args.model_width_multiplier,
        target_pam_mb=args.model_target_pam_mb,
        pam_calibration_steps=args.model_pam_calibration_steps,
    )
    model_metadata = getattr(model, "model_metadata", {})
    args.resolved_model_width_multiplier = model_metadata.get(
        "model_width_multiplier", args.model_width_multiplier
    )
    args.resolved_model_estimated_pam_mb = model_metadata.get("model_estimated_pam_mb", "")
    args.resolved_model_parameter_count = model_metadata.get("model_parameter_count", "")
    train_loader = get_dataloader(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        client_id=args.client_id,
        poisoning_method=poisoning_method,
        split=args.dataset_split,
        augment=augment,
        batch_size=args.batch_size,
        shuffle=True,
    )
    eval_loader = get_dataloader(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        client_id=args.client_id,
        poisoning_method=poisoning_method,
        split=args.dataset_split,
        augment=augment,
        batch_size=args.batch_size,
        shuffle=False,
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
        run_type="local_ml",
        poisoning_method=poisoning_method,
        is_poisoned_client=(poisoning_method != POISONING_METHOD_CLEAN),
        poisoned_client_count=1 if poisoning_method != POISONING_METHOD_CLEAN else 0,
        poisoned_client_ids=[args.client_id] if poisoning_method != POISONING_METHOD_CLEAN else [],
        poison_fraction=poison_fraction,
        attack_name=attack_name_for_poisoning_method(poisoning_method),
    )
    with HardwareLogger(
        log_dir=args.log_dir,
        condition=condition,
        training_state=state,
        fps=args.hardware_fps,
    ) as logger:
        perf_path = logger.path.with_name(f"{logger.path.stem}_perf.csv")
        metrics_logger = MetricsLogger(
            path=logger.path.with_name(f"{logger.path.stem}_metrics.csv"),
            condition=condition,
        )
        with PerfLogger(
            log_dir=args.log_dir,
            condition=condition,
            training_state=state,
            path=perf_path,
            events=parse_perf_events(args.perf_events),
            fps=args.perf_fps,
        ):
            train_model(
                model=model,
                train_loader=train_loader,
                epochs=args.local_epochs,
                learning_rate=args.learning_rate,
                state=state,
                round_id=0,
                metrics_logger=metrics_logger,
            )
            evaluate_model(
                model=model,
                data_loader=eval_loader,
                state=state,
                round_id=0,
                metrics_logger=metrics_logger,
            )
    return str(logger.path)


def parse_perf_events(value: str):
    if not value:
        return DEFAULT_PERF_EVENTS
    return [event.strip() for event in value.split(",") if event.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.set_defaults(local_epochs=DEFAULT_LOCAL_ML_EPOCHS, log_dir=DEFAULT_LOCAL_ML_LOG_DIR)
    parser.add_argument(
        "--poisoning-method",
        default="all",
        help=(
            "Use one method, a comma-separated subset, or all/both. "
            f"Allowed methods: {','.join(POISONING_METHODS)}"
        ),
    )
    parser.add_argument("--reference-trials", type=int, default=DEFAULT_LOCAL_ML_GLOBAL_CLEAN_REFERENCE_TRIALS)
    parser.add_argument("--trials", type=int, default=DEFAULT_LOCAL_ML_ANALYSIS_TRIALS)
    parser.add_argument(
        "--perf-events",
        default="",
        help=(
            "Comma-separated perf event list. Defaults to the expanded cache-analysis event set: "
            f"{','.join(DEFAULT_PERF_EVENTS)}"
        ),
    )
    parser.add_argument(
        "--perf-fps",
        type=float,
        default=10.0,
        help="perf stat sampling frequency. The default 10 FPS uses 100 ms intervals.",
    )
    parser.add_argument(
        "--hardware-fps",
        type=float,
        default=10.0,
        help="psutil hardware sampling frequency. The default 10 FPS uses 100 ms intervals.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    device = get_device_config(args.client_id)
    if not args.host:
        args.host = device.get("host", "")
    if not args.device_id:
        args.device_id = args.host

    augment = augment_from_args(args)
    prepare_dataset(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        num_clients=args.num_clients,
        seed=args.seed,
        resize=augment.get("resize", [64, 64]),
        batch_size=args.batch_size,
    )
    if args.prepare_only:
        return

    methods = (
        parse_poisoning_methods(args.poisoning_method, include_clean=True)
        if args.poisoning_method in {"all", "both"} or "," in args.poisoning_method
        else [args.poisoning_method]
    )
    if args.poisoning_method == "all":
        methods = list(DEFAULT_LOCAL_ML_POISONING_METHODS)
    base_seed = args.seed

    for reference_idx in range(args.reference_trials):
        args.trial_id = f"reference_{reference_idx}"
        args.seed = base_seed + reference_idx
        args.run_role = "global_clean_reference"
        run_one_local(args, POISONING_METHOD_CLEAN)

    analysis_seed_offset = 1000
    for trial in range(args.trials):
        for method in methods:
            args.trial_id = f"trial_{trial}"
            args.seed = base_seed + analysis_seed_offset + trial
            args.run_role = "analysis"
            run_one_local(args, method)


if __name__ == "__main__":
    main()
