import argparse
import copy
import csv
from pathlib import Path

import torch

from dataset_preparation import (
    dataset_slug,
    evaluation_augment_from_training,
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
    POISONING_METHOD_BADSAMPLING,
    POISONING_METHOD_CLEAN,
    add_common_args,
    attack_name_for_poisoning_method,
    augment_from_args,
    condition_columns,
    get_device as get_device_config,
    parse_poisoning_methods,
    set_all_seeds,
    yyyymmddhhmmss_log_path,
)
from entropy_logger import LayerEntropyLogger
from metrics_logger import MetricsLogger
from models import get_model
from perf_logger import DEFAULT_PERF_EVENTS, LayerPerfLogger, parse_perf_events
from training_state import TrainingState
from training_utils import evaluate_model, replay_model_with_frozen_state, train_model


def require_prepared_dataset(args: argparse.Namespace, poisoning_method: str) -> None:
    root = Path(args.data_dir) / dataset_slug(args.dataset)
    metadata_path = root / "partition_metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Prepared metadata is missing: {metadata_path}. Sync the prepared dataset "
            "before running the experiment."
        )

    with metadata_path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    metadata_method = (
        POISONING_METHOD_CLEAN
        if poisoning_method == POISONING_METHOD_BADSAMPLING
        else poisoning_method
    )
    matching_rows = [
        row
        for row in rows
        if row.get("client_id") == args.client_id
        and row.get("dataset_split") == args.dataset_split
        and row.get("poisoning_method") == metadata_method
        and row.get("partition_method", "iid") == args.partition_method
    ]
    if not matching_rows:
        raise ValueError(
            f"No prepared rows for dataset={args.dataset}, client={args.client_id}, "
            f"split={args.dataset_split}, poisoning={metadata_method}, "
            f"partition={args.partition_method} in {metadata_path}."
        )

    if poisoning_method == POISONING_METHOD_BADSAMPLING:
        plan = root / "poisoned" / POISONING_METHOD_BADSAMPLING / args.client_id / "sampling_plan.json"
        if not plan.is_file():
            raise FileNotFoundError(f"Prepared BadSampler plan is missing: {plan}")


def run_one_local(args: argparse.Namespace, poisoning_method: str) -> str:
    set_all_seeds(args.seed)
    augment = augment_from_args(args)
    require_prepared_dataset(args, poisoning_method)
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
    initial_model_state = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    checkpoint_path = (
        Path(args.checkpoint_path)
        if args.checkpoint_path
        else Path(args.log_dir) / f"{args.trial_id}_final_model.pt"
    )
    args.checkpoint_path = str(checkpoint_path)
    checkpoint_payload = None
    if args.experiment_mode == "frozen_replay":
        checkpoint_payload = load_checkpoint(
            model=model,
            path=checkpoint_path,
            expected_model=args.model,
            expected_num_classes=num_classes,
            expected_input_size=input_size,
        )
    model_metadata = getattr(model, "model_metadata", {})
    args.resolved_model_width_multiplier = model_metadata.get(
        "model_width_multiplier", args.model_width_multiplier
    )
    args.resolved_model_estimated_pam_mb = model_metadata.get("model_estimated_pam_mb", "")
    args.resolved_model_parameter_count = model_metadata.get("model_parameter_count", "")
    surrogate_model = None
    if poisoning_method == POISONING_METHOD_BADSAMPLING:
        surrogate_model = copy.deepcopy(model)
        sampling_state = initial_model_state
        if args.experiment_mode == "frozen_replay":
            assert checkpoint_payload is not None
            sampling_state = checkpoint_payload.get("sampling_surrogate_state_dict")
            if sampling_state is None:
                raise ValueError(
                    "BadSampler checkpoint is missing its initial sampling surrogate state."
                )
        surrogate_model.load_state_dict(sampling_state, strict=True)
    train_loader = get_dataloader(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        client_id=args.client_id,
        poisoning_method=poisoning_method,
        split=args.dataset_split,
        augment=augment,
        batch_size=args.batch_size,
        shuffle=True,
        surrogate_model=surrogate_model,
        badsampler_kappa=args.badsampler_kappa,
        badsampler_seed=args.seed,
        badsampler_run_name=f"{args.model}_{args.trial_id}_seed_{args.seed}",
        badsampler_num_epochs=args.local_epochs,
        max_samples=args.max_train_samples,
        subset_seed=args.seed,
    )
    clean_test_loader = None
    if args.experiment_mode == "train":
        evaluation_augment = evaluation_augment_from_training(augment)
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
    warm_up_cpu_workers(model, input_size=input_size, batch_size=args.batch_size)
    layer_perf_path = yyyymmddhhmmss_log_path(
        args.log_dir, suffix="_layer_perf.csv"
    )
    metrics_path = layer_perf_path.with_name(
        layer_perf_path.name.replace("_layer_perf.csv", "_metrics.csv")
    )
    entropy_path = layer_perf_path.with_name(
        layer_perf_path.name.replace("_layer_perf.csv", "_entropy_summary.csv")
    )
    metrics_logger = MetricsLogger(path=metrics_path, condition=condition)
    events = parse_perf_events(args.perf_events)
    entropy_logger = LayerEntropyLogger(
        path=entropy_path,
        condition=condition,
        maxpool_markov_only=args.maxpool_markov_only,
    )
    with LayerPerfLogger(
        model=model,
        path=layer_perf_path,
        condition=condition,
        events=events,
        observer=entropy_logger,
        module_types=(torch.nn.MaxPool2d,) if args.maxpool_markov_only else None,
    ) as layer_perf_logger:
        print(
            f"layer_perf={layer_perf_path} leaf_layers={len(layer_perf_logger.leaf_modules)} "
            f"rows_per_batch={layer_perf_logger.expected_rows_per_batch} "
            f"entropy_summary={entropy_path} "
            f"maxpool_markov={entropy_logger.maxpool_markov_path} "
            f"torch_threads={torch.get_num_threads()} "
            f"torch_interop_threads={torch.get_num_interop_threads()}"
        )

        def evaluate_clean_test(epoch: int) -> None:
            assert clean_test_loader is not None
            result = evaluate_model(
                model=model,
                data_loader=clean_test_loader,
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
                f"train_condition={poisoning_method} eval_dataset=clean_test "
                f"epoch={epoch} loss={result['loss']:.6f} "
                f"accuracy={result['accuracy']:.4f} "
                f"examples={int(result['num_examples'])}"
            )

        if args.experiment_mode == "train":
            train_model(
                model=model,
                train_loader=train_loader,
                epochs=args.local_epochs,
                learning_rate=args.learning_rate,
                state=state,
                round_id=0,
                metrics_logger=metrics_logger,
                layer_perf_logger=layer_perf_logger,
                epoch_end_callback=evaluate_clean_test,
            )
        else:
            result = replay_model_with_frozen_state(
                model=model,
                data_loader=train_loader,
                epochs=args.local_epochs,
                state=state,
                round_id=0,
                metrics_logger=metrics_logger,
                layer_perf_logger=layer_perf_logger,
            )
            print(
                f"frozen_replay_complete checkpoint={checkpoint_path} "
                f"loss={result['loss']:.6f} accuracy={result['accuracy']:.4f} "
                "model_state_unchanged=true"
            )
    if args.experiment_mode == "train":
        save_checkpoint(
            model=model,
            path=checkpoint_path,
            model_name=args.model,
            num_classes=num_classes,
            input_size=input_size,
            condition=condition,
            sampling_surrogate_state=(
                initial_model_state
                if poisoning_method == POISONING_METHOD_BADSAMPLING
                else None
            ),
        )
        print(f"saved_final_model={checkpoint_path}")
    return str(layer_perf_path)


def save_checkpoint(
    *,
    model: torch.nn.Module,
    path: Path,
    model_name: str,
    num_classes: int,
    input_size: tuple[int, int],
    condition: dict,
    sampling_surrogate_state: dict | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_name": model_name,
            "num_classes": num_classes,
            "input_size": list(input_size),
            "state_dict": model.state_dict(),
            "condition": condition,
            "sampling_surrogate_state_dict": sampling_surrogate_state,
        },
        path,
    )


def load_checkpoint(
    *,
    model: torch.nn.Module,
    path: Path,
    expected_model: str,
    expected_num_classes: int,
    expected_input_size: tuple[int, int],
) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Frozen replay checkpoint is missing: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("model_name") != expected_model:
        raise ValueError(
            f"Checkpoint model mismatch: {payload.get('model_name')} != {expected_model}"
        )
    if int(payload.get("num_classes", -1)) != expected_num_classes:
        raise ValueError("Checkpoint class count does not match the prepared dataset.")
    if tuple(payload.get("input_size", ())) != expected_input_size:
        raise ValueError("Checkpoint input size does not match the current augmentation.")
    model.load_state_dict(payload["state_dict"], strict=True)
    return payload


def warm_up_cpu_workers(
    model: torch.nn.Module, *, input_size: tuple[int, int], batch_size: int
) -> None:
    """Create PyTorch CPU worker threads before perf opens per-thread counters."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(batch_size, 3, input_size[0], input_size[1]))
    model.train(was_training)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.set_defaults(local_epochs=DEFAULT_LOCAL_ML_EPOCHS, log_dir=DEFAULT_LOCAL_ML_LOG_DIR)
    parser.add_argument(
        "--poisoning-method",
        default=POISONING_METHOD_CLEAN,
        help=(
            "Use one method, a comma-separated subset, or all/both. "
            f"Allowed methods: {','.join(POISONING_METHODS)}"
        ),
    )
    parser.add_argument("--reference-trials", type=int, default=DEFAULT_LOCAL_ML_GLOBAL_CLEAN_REFERENCE_TRIALS)
    parser.add_argument("--trials", type=int, default=DEFAULT_LOCAL_ML_ANALYSIS_TRIALS)
    parser.add_argument(
        "--perf-events",
        default=",".join(DEFAULT_PERF_EVENTS),
        help=(
            "At most six comma-separated events measured simultaneously at each layer: "
            f"{','.join(DEFAULT_PERF_EVENTS)}"
        ),
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=0,
        help="Set PyTorch intra-op threads; 0 keeps the environment default.",
    )
    parser.add_argument(
        "--maxpool-markov-only",
        action="store_true",
        help="Instrument only MaxPool2d layers and write batch-level Markov metrics.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    if args.torch_threads < 0:
        parser.error("--torch-threads must be >= 0")
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)

    device = get_device_config(args.client_id)
    if not args.host:
        args.host = device.get("host", "")
    if not args.device_id:
        args.device_id = args.host

    if args.prepare_only:
        augment = augment_from_args(args)
        prepare_dataset(
            data_dir=args.data_dir,
            dataset_name=args.dataset,
            num_clients=args.num_clients,
            seed=args.seed,
            resize=augment.get("resize", [64, 64]),
            batch_size=args.batch_size,
            prepare_scenarios=POISONING_METHOD_CLEAN,
            partition_method=args.partition_method,
            noniid_alpha=args.noniid_alpha,
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

    for trial in range(args.trials):
        for method in methods:
            args.trial_id = f"trial_{trial}"
            args.seed = base_seed + trial
            args.run_role = "analysis"
            run_one_local(args, method)


if __name__ == "__main__":
    main()
