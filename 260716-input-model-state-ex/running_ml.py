import argparse

import torch

from dataset_preparation import (
    get_dataloader,
    get_num_classes,
    get_poison_fraction,
    prepare_dataset,
)
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


AUGMENTATION_SEQUENCE_PROFILES = ("baseline", "moderate", "strong")


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
        test_fraction=args.test_fraction,
        test_seed=args.test_seed,
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


def parse_input_sequences(value: str):
    sequences = []
    for sequence_text in (item.strip() for item in value.split(",")):
        if not sequence_text:
            continue
        methods = tuple(item.strip() for item in sequence_text.split(":"))
        if len(methods) not in {1, 2} or any(
            method not in POISONING_METHODS for method in methods
        ):
            raise argparse.ArgumentTypeError(
                "Each input sequence must contain one valid method, or two valid methods "
                "separated by ':'. "
                f"Received {sequence_text!r}."
            )
        sequences.append(methods)
    if not sequences:
        raise argparse.ArgumentTypeError("At least one input sequence is required.")
    return sequences


def parse_augmentation_sequences(value: str):
    sequences = []
    for sequence_text in (item.strip() for item in value.split(",")):
        if not sequence_text:
            continue
        profiles = tuple(item.strip() for item in sequence_text.split(":"))
        if len(profiles) not in {1, 2} or any(
            profile not in AUGMENTATION_SEQUENCE_PROFILES for profile in profiles
        ):
            raise argparse.ArgumentTypeError(
                "Each augmentation sequence must contain one or two profiles "
                f"from {AUGMENTATION_SEQUENCE_PROFILES}, separated by ':'. "
                f"Received {sequence_text!r}."
            )
        sequences.append(profiles)
    if not sequences:
        raise argparse.ArgumentTypeError("At least one augmentation sequence is required.")
    return sequences


def augment_for_profile(args: argparse.Namespace, profile: str) -> dict:
    augment = augment_from_args(args)
    augment["_profile"] = profile
    return augment


def run_one_input_sequence(
    args: argparse.Namespace,
    first_method: str,
    second_method: str | None,
    stage_epochs: int,
) -> str:
    """Train one model across one or two input regimes.

    The model parameters and Adam state are retained at the stage boundary.
    Hardware and perf loggers remain open across both stages when a transition
    is requested.
    """
    set_all_seeds(args.seed)
    augment = augment_from_args(args)
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

    methods = (first_method,) if second_method is None else (first_method, second_method)
    total_epochs = stage_epochs * len(methods)
    sequence = "_to_".join(methods)
    print(
        f"Starting sequence={sequence} trial={args.trial_id} "
        f"stage_epochs={stage_epochs} total_epochs={total_epochs}"
    )
    loaders = {
        method: get_dataloader(
            data_dir=args.data_dir,
            dataset_name=args.dataset,
            client_id=args.client_id,
            poisoning_method=method,
            split=args.dataset_split,
            augment=augment,
            batch_size=args.batch_size,
            shuffle=True,
        )
        for method in set(methods)
    }
    evaluation_augment = dict(augment)
    evaluation_augment["horizontal_flip"] = False
    clean_evaluation_loader = get_dataloader(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        client_id="all",
        poisoning_method=POISONING_METHOD_CLEAN,
        split="test",
        augment=evaluation_augment,
        batch_size=args.batch_size,
        shuffle=False,
    )
    poison_fractions = {
        method: get_poison_fraction(
            data_dir=args.data_dir,
            dataset_name=args.dataset,
            client_id=args.client_id,
            poisoning_method=method,
            split=args.dataset_split,
        )
        for method in set(methods)
    }

    original_local_epochs = args.local_epochs
    original_experiment_id = args.experiment_id
    args.local_epochs = total_epochs
    args.evaluation_method = POISONING_METHOD_CLEAN
    args.evaluation_split = "clean_test"
    args.evaluation_num_examples = len(clean_evaluation_loader.dataset)
    args.experiment_id = original_experiment_id or (
        f"input_model_state_{sequence}_{args.trial_id}_{args.client_id}"
    )
    condition = condition_columns(
        args=args,
        run_type="local_ml_input_model_state",
        poisoning_method=sequence,
        is_poisoned_client=any(method != POISONING_METHOD_CLEAN for method in methods),
        poisoned_client_count=1,
        poisoned_client_ids=[args.client_id],
        poison_fraction=max(poison_fractions.values()),
        attack_name="input_model_state_transition",
    )
    args.local_epochs = original_local_epochs
    args.experiment_id = original_experiment_id

    state = TrainingState(
        round=0,
        epoch=0,
        batch_idx=0,
        phase="idle",
        training_sequence=sequence,
        stage_index=0,
        stage_epoch=0,
        input_poisoning_method=first_method,
        model_state_condition="stage_started_from_initial",
    )
    state.update_condition(
        poisoning_method=first_method,
        is_poisoned_client=(first_method != POISONING_METHOD_CLEAN),
        poisoned_client_count=1 if first_method != POISONING_METHOD_CLEAN else 0,
        poisoned_client_ids=args.client_id if first_method != POISONING_METHOD_CLEAN else "",
        poison_fraction=poison_fractions[first_method],
        attack_name=attack_name_for_poisoning_method(first_method),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

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
            for stage_index, method in enumerate(methods):
                model_state_condition = (
                    "stage_started_from_initial"
                    if stage_index == 0
                    else f"stage_started_from_{first_method}_{stage_epochs}_epochs"
                )
                state.update(
                    epoch=stage_index * stage_epochs,
                    batch_idx=0,
                    phase="idle",
                    training_sequence=sequence,
                    stage_index=stage_index,
                    stage_epoch=0,
                    input_poisoning_method=method,
                    model_state_condition=model_state_condition,
                )
                state.update_condition(
                    poisoning_method=method,
                    is_poisoned_client=(method != POISONING_METHOD_CLEAN),
                    poisoned_client_count=1 if method != POISONING_METHOD_CLEAN else 0,
                    poisoned_client_ids=args.client_id if method != POISONING_METHOD_CLEAN else "",
                    poison_fraction=poison_fractions[method],
                    attack_name=attack_name_for_poisoning_method(method),
                )
                print(
                    f"  stage={stage_index} global_epochs="
                    f"{stage_index * stage_epochs}-{(stage_index + 1) * stage_epochs - 1} "
                    f"input={method} model_state={model_state_condition}"
                )

                def evaluate_clean_data(global_epoch: int, current_stage_epoch: int) -> None:
                    state.update(
                        epoch=global_epoch,
                        batch_idx=0,
                        phase="evaluation",
                        stage_epoch=current_stage_epoch,
                        input_poisoning_method=POISONING_METHOD_CLEAN,
                    )
                    state.update_condition(
                        dataset_split="test",
                        client_partition_id="all",
                        poisoning_method=POISONING_METHOD_CLEAN,
                        is_poisoned_client=False,
                        poisoned_client_count=0,
                        poisoned_client_ids="",
                        poison_fraction=0.0,
                        attack_name="",
                    )
                    result = evaluate_model(
                        model=model,
                        data_loader=clean_evaluation_loader,
                        state=state,
                        round_id=0,
                        metrics_logger=metrics_logger,
                        metric_event="clean_test_epoch",
                        metric_split="clean_test",
                    )
                    print(
                        f"    clean_test epoch={global_epoch} "
                        f"loss={result['loss']:.6f} accuracy={result['accuracy']:.4f}"
                    )
                    state.update(
                        epoch=global_epoch,
                        batch_idx=0,
                        stage_epoch=current_stage_epoch,
                        input_poisoning_method=method,
                    )
                    state.update_condition(
                        dataset_split=args.dataset_split,
                        client_partition_id=args.client_id,
                        poisoning_method=method,
                        is_poisoned_client=(method != POISONING_METHOD_CLEAN),
                        poisoned_client_count=1 if method != POISONING_METHOD_CLEAN else 0,
                        poisoned_client_ids=args.client_id if method != POISONING_METHOD_CLEAN else "",
                        poison_fraction=poison_fractions[method],
                        attack_name=attack_name_for_poisoning_method(method),
                    )

                train_model(
                    model=model,
                    train_loader=loaders[method],
                    epochs=stage_epochs,
                    learning_rate=args.learning_rate,
                    state=state,
                    round_id=0,
                    metrics_logger=metrics_logger,
                    optimizer=optimizer,
                    epoch_offset=stage_index * stage_epochs,
                    training_sequence=sequence,
                    stage_index=stage_index,
                    input_poisoning_method=method,
                    model_state_condition=model_state_condition,
                    epoch_end_callback=evaluate_clean_data,
                )
    print(f"Completed sequence={sequence}; hardware_log={logger.path}")
    return str(logger.path)


def run_one_augmentation_sequence(
    args: argparse.Namespace,
    first_profile: str,
    second_profile: str | None,
    stage_epochs: int,
) -> str:
    """Train continuously while switching only the saved augmentation input."""
    set_all_seeds(args.seed)
    profiles = (
        (first_profile,)
        if second_profile is None
        else (first_profile, second_profile)
    )
    augments = {profile: augment_for_profile(args, profile) for profile in set(profiles)}
    resize = augments[first_profile].get("resize", [64, 64])
    input_size = (int(resize[0]), int(resize[1]))
    num_classes = get_num_classes(args.data_dir, dataset_name=args.dataset)
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
    args.resolved_model_estimated_pam_mb = model_metadata.get(
        "model_estimated_pam_mb", ""
    )
    args.resolved_model_parameter_count = model_metadata.get(
        "model_parameter_count", ""
    )

    loaders = {
        profile: get_dataloader(
            data_dir=args.data_dir,
            dataset_name=args.dataset,
            client_id=args.client_id,
            poisoning_method=POISONING_METHOD_CLEAN,
            split=args.dataset_split,
            augment=augments[profile],
            batch_size=args.batch_size,
            shuffle=True,
        )
        for profile in set(profiles)
    }
    evaluation_augment = augment_for_profile(args, "baseline")
    evaluation_augment["horizontal_flip"] = False
    clean_evaluation_loader = get_dataloader(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        client_id="all",
        poisoning_method=POISONING_METHOD_CLEAN,
        split="test",
        augment=evaluation_augment,
        batch_size=args.batch_size,
        shuffle=False,
    )

    total_epochs = stage_epochs * len(profiles)
    sequence = "_to_".join(profiles)
    training_sequence = f"augmentation_{sequence}"
    print(
        f"Starting augmentation sequence={sequence} trial={args.trial_id} "
        f"stage_epochs={stage_epochs} total_epochs={total_epochs}"
    )

    original_local_epochs = args.local_epochs
    original_experiment_id = args.experiment_id
    args.local_epochs = total_epochs
    args.evaluation_method = POISONING_METHOD_CLEAN
    args.evaluation_split = "clean_test"
    args.evaluation_num_examples = len(clean_evaluation_loader.dataset)
    args.experiment_id = original_experiment_id or (
        f"augmentation_state_{sequence}_{args.trial_id}_{args.client_id}"
    )
    condition = condition_columns(
        args=args,
        run_type="local_ml_augmentation_state",
        poisoning_method=POISONING_METHOD_CLEAN,
        is_poisoned_client=False,
        poisoned_client_count=0,
        poisoned_client_ids=[],
        poison_fraction=0.0,
        attack_name="augmentation_state_transition",
    )
    args.local_epochs = original_local_epochs
    args.experiment_id = original_experiment_id

    state = TrainingState(
        round=0,
        epoch=0,
        batch_idx=0,
        phase="idle",
        training_sequence=training_sequence,
        stage_index=0,
        stage_epoch=0,
        input_poisoning_method=POISONING_METHOD_CLEAN,
        input_augmentation_profile=first_profile,
        model_state_condition="stage_started_from_initial",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

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
            for stage_index, profile in enumerate(profiles):
                stage_augment = augments[profile]
                model_state_condition = (
                    "stage_started_from_initial"
                    if stage_index == 0
                    else f"stage_started_from_{first_profile}_{stage_epochs}_epochs"
                )
                state.update(
                    epoch=stage_index * stage_epochs,
                    batch_idx=0,
                    phase="idle",
                    training_sequence=training_sequence,
                    stage_index=stage_index,
                    stage_epoch=0,
                    input_poisoning_method=POISONING_METHOD_CLEAN,
                    input_augmentation_profile=profile,
                    model_state_condition=model_state_condition,
                )
                state.update_condition(
                    dataset_split=args.dataset_split,
                    client_partition_id=args.client_id,
                    augmentation_profile=profile,
                    augment_enabled=bool(stage_augment.get("enabled", True)),
                    augment_resize="x".join(str(value) for value in stage_augment.get("resize", [])),
                    augment_horizontal_flip=bool(stage_augment.get("horizontal_flip", False)),
                    augment_normalize=bool(stage_augment.get("normalize", False)),
                    poisoning_method=POISONING_METHOD_CLEAN,
                    is_poisoned_client=False,
                    poisoned_client_count=0,
                    poisoned_client_ids="",
                    poison_fraction=0.0,
                )
                print(
                    f"  stage={stage_index} global_epochs="
                    f"{stage_index * stage_epochs}-{(stage_index + 1) * stage_epochs - 1} "
                    f"augmentation={profile} model_state={model_state_condition}"
                )

                def evaluate_clean_data(global_epoch: int, current_stage_epoch: int) -> None:
                    state.update(
                        epoch=global_epoch,
                        batch_idx=0,
                        phase="evaluation",
                        stage_epoch=current_stage_epoch,
                        input_augmentation_profile="baseline",
                    )
                    state.update_condition(
                        dataset_split="test",
                        client_partition_id="all",
                        augmentation_profile="baseline",
                    )
                    result = evaluate_model(
                        model=model,
                        data_loader=clean_evaluation_loader,
                        state=state,
                        round_id=0,
                        metrics_logger=metrics_logger,
                        metric_event="clean_test_epoch",
                        metric_split="clean_test",
                    )
                    print(
                        f"    clean_test epoch={global_epoch} "
                        f"loss={result['loss']:.6f} accuracy={result['accuracy']:.4f}"
                    )
                    state.update(
                        epoch=global_epoch,
                        batch_idx=0,
                        stage_epoch=current_stage_epoch,
                        input_augmentation_profile=profile,
                    )
                    state.update_condition(
                        dataset_split=args.dataset_split,
                        client_partition_id=args.client_id,
                        augmentation_profile=profile,
                    )

                train_model(
                    model=model,
                    train_loader=loaders[profile],
                    epochs=stage_epochs,
                    learning_rate=args.learning_rate,
                    state=state,
                    round_id=0,
                    metrics_logger=metrics_logger,
                    optimizer=optimizer,
                    epoch_offset=stage_index * stage_epochs,
                    training_sequence=training_sequence,
                    stage_index=stage_index,
                    input_poisoning_method=POISONING_METHOD_CLEAN,
                    input_augmentation_profile=profile,
                    model_state_condition=model_state_condition,
                    epoch_end_callback=evaluate_clean_data,
                )
    print(f"Completed augmentation sequence={sequence}; hardware_log={logger.path}")
    return str(logger.path)


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
    parser.add_argument(
        "--input-sequences",
        default="",
        help=(
            "Run one-stage or continuous two-stage input/model-state experiments. "
            "Separate stages with ':' and sequences with ','. Examples: "
            "availability_shortcuts or clean:availability_shortcuts. "
            "Empty keeps the legacy independent-run mode."
        ),
    )
    parser.add_argument(
        "--stage-epochs",
        type=int,
        default=10,
        help="Epochs per stage in --input-sequences mode.",
    )
    parser.add_argument(
        "--augmentation-sequences",
        default="",
        help=(
            "Run a continuous one- or two-stage augmentation experiment while "
            "retaining model and Adam state. Example: baseline:strong."
        ),
    )
    args = parser.parse_args()

    if args.input_sequences and args.augmentation_sequences:
        parser.error("Use only one of --input-sequences and --augmentation-sequences.")
    parsed_input_sequences = (
        parse_input_sequences(args.input_sequences) if args.input_sequences else []
    )
    parsed_augmentation_sequences = (
        parse_augmentation_sequences(args.augmentation_sequences)
        if args.augmentation_sequences
        else []
    )

    device = get_device_config(args.client_id)
    if not args.host:
        args.host = device.get("host", "")
    if not args.device_id:
        args.device_id = args.host

    augment = augment_from_args(args)
    if parsed_augmentation_sequences:
        prepare_scenarios = POISONING_METHOD_CLEAN
    elif parsed_input_sequences:
        prepare_scenarios = sorted(
            {method for sequence in parsed_input_sequences for method in sequence}
        )
    else:
        prepare_scenarios = "all"
    prepare_dataset(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        num_clients=args.num_clients,
        seed=args.seed,
        resize=augment.get("resize", [64, 64]),
        batch_size=args.batch_size,
        test_fraction=args.test_fraction,
        test_seed=args.test_seed,
        prepare_scenarios=prepare_scenarios,
    )
    if args.prepare_only:
        return

    if parsed_augmentation_sequences:
        if args.stage_epochs <= 0:
            parser.error("--stage-epochs must be positive.")
        base_seed = args.seed
        for trial in range(args.trials):
            args.trial_id = f"trial_{trial}"
            args.seed = base_seed + 1000 + trial
            args.run_role = "augmentation_state_analysis"
            for profiles in parsed_augmentation_sequences:
                run_one_augmentation_sequence(
                    args,
                    first_profile=profiles[0],
                    second_profile=profiles[1] if len(profiles) == 2 else None,
                    stage_epochs=args.stage_epochs,
                )
        return

    if parsed_input_sequences:
        if args.stage_epochs <= 0:
            parser.error("--stage-epochs must be positive.")
        base_seed = args.seed
        for trial in range(args.trials):
            args.trial_id = f"trial_{trial}"
            args.seed = base_seed + 1000 + trial
            args.run_role = "input_model_state_analysis"
            for methods in parsed_input_sequences:
                run_one_input_sequence(
                    args,
                    first_method=methods[0],
                    second_method=methods[1] if len(methods) == 2 else None,
                    stage_epochs=args.stage_epochs,
                )
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
