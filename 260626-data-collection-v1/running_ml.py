import argparse

from dataset_preparation import get_dataloader, get_num_classes, prepare_dataset
from experiment_config import (
    add_common_args,
    augment_from_args,
    condition_columns,
    get_device as get_device_config,
    set_all_seeds,
)
from hardware_logger import HardwareLogger, TrainingState
from models import get_model
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
    num_classes = get_num_classes(args.data_dir)
    model = get_model(args.model, num_classes=num_classes)
    train_loader = get_dataloader(
        data_dir=args.data_dir,
        client_id=args.client_id,
        poisoning_method=poisoning_method,
        split=args.dataset_split,
        augment=augment,
        batch_size=args.batch_size,
        shuffle=True,
    )
    eval_loader = get_dataloader(
        data_dir=args.data_dir,
        client_id=args.client_id,
        poisoning_method=poisoning_method,
        split=args.dataset_split,
        augment=augment,
        batch_size=args.batch_size,
        shuffle=False,
    )
    state = TrainingState(round=0, epoch=0, batch_idx=0, phase="idle")
    condition = condition_columns(
        args=args,
        run_type="local_ml",
        poisoning_method=poisoning_method,
        is_poisoned_client=(poisoning_method != "clean"),
        poisoned_client_count=1 if poisoning_method != "clean" else 0,
        poisoned_client_ids=[args.client_id] if poisoning_method != "clean" else [],
        poison_fraction=1.0 if poisoning_method != "clean" else 0.0,
        attack_name="adaptive_min_min_samplewise" if poisoning_method == "adaptive" else "",
    )
    with HardwareLogger(log_dir=args.log_dir, condition=condition, training_state=state) as logger:
        train_model(
            model=model,
            train_loader=train_loader,
            epochs=args.local_epochs,
            learning_rate=args.learning_rate,
            state=state,
            round_id=0,
        )
        evaluate_model(model=model, data_loader=eval_loader, state=state, round_id=0)
    return str(logger.path)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--poisoning-method", choices=["clean", "adaptive", "both"], default="both")
    parser.add_argument("--trials", type=int, default=5)
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

    methods = ["clean", "adaptive"] if args.poisoning_method == "both" else [args.poisoning_method]
    base_seed = args.seed
    for trial in range(args.trials):
        for method in methods:
            args.trial_id = trial
            args.seed = base_seed + trial
            run_one_local(args, method)


if __name__ == "__main__":
    main()
