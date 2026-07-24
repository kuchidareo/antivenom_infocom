import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch

from dataset_preparation import (
    evaluation_augment_from_training,
    get_dataloader,
    get_num_classes,
    get_poison_fraction,
)
from experiment_config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LOCAL_ML_LOG_DIR,
    POISONING_METHODS,
    POISONING_METHOD_CLEAN,
    add_common_args,
    attack_name_for_poisoning_method,
    augment_from_args,
    condition_columns,
    set_all_seeds,
)
from hardware_logger import TrainingState
from models import get_model, get_monitoring_layers
from perf_logger import PhasePerfLogger, default_perf_events_for_host


MONITOR_COLUMNS = [
    "data_condition",
    "model_condition",
    "checkpoint_path",
    "model_training_seed",
    "model_training_epochs",
    "model_training_batch_size",
    "measurement_scope",
    "layer_index",
    "layer_name",
    "layer_type",
    "layer_call_index",
]


def parse_conditions(value: str) -> List[str]:
    conditions: List[str] = []
    for condition in (item.strip() for item in value.split(",")):
        if not condition:
            continue
        if condition not in POISONING_METHODS:
            raise ValueError(
                f"Unknown condition {condition!r}; choose from {', '.join(POISONING_METHODS)}"
            )
        if condition not in conditions:
            conditions.append(condition)
    if not conditions:
        raise ValueError("At least one condition is required.")
    return conditions


def parse_perf_events(value: str, host: str) -> List[str]:
    if not value:
        return default_perf_events_for_host(host)
    events = [event.strip() for event in value.split(",") if event.strip()]
    if not events:
        raise ValueError("--perf-events did not contain any event names.")
    return events


def checkpoint_path(
    model_dir: Path,
    model_name: str,
    client_id: str,
    model_condition: str,
    trial_id: str,
) -> Path:
    return model_dir / model_name / client_id / model_condition / f"{trial_id}.pt"


def load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Trained model checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Invalid trained model checkpoint: {path}")
    return checkpoint


def validate_checkpoint(
    checkpoint: Dict[str, Any],
    *,
    path: Path,
    model_name: str,
    client_id: str,
    model_condition: str,
    trial_id: str,
) -> None:
    expected = {
        "model": model_name,
        "client_id": client_id,
        "training_condition": model_condition,
        "trial_id": trial_id,
    }
    mismatches = [
        f"{key}={checkpoint.get(key)!r} (expected {value!r})"
        for key, value in expected.items()
        if checkpoint.get(key) != value
    ]
    if mismatches:
        raise ValueError(f"Checkpoint metadata mismatch in {path}: {'; '.join(mismatches)}")


def make_loader(args: argparse.Namespace, data_condition: str) -> Any:
    if data_condition == "badsampling":
        raise ValueError(
            "Controlled BadSampler replay needs an explicit saved sampling plan and "
            "is not yet enabled as an input condition."
        )
    evaluation_augment = evaluation_augment_from_training(augment_from_args(args))
    return get_dataloader(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        client_id=args.client_id,
        poisoning_method=data_condition,
        split=args.dataset_split,
        augment=evaluation_augment,
        batch_size=args.batch_size,
        shuffle=False,
    )


def limited_batches(loader: Iterable[Any], max_batches: int) -> Iterable[Any]:
    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        yield batch_idx, batch


def monitor_metadata(
    *,
    data_condition: str,
    model_condition: str,
    checkpoint_path: Path,
    checkpoint: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "data_condition": data_condition,
        "model_condition": model_condition,
        "checkpoint_path": str(checkpoint_path),
        "model_training_seed": checkpoint.get("seed", ""),
        "model_training_epochs": checkpoint.get("local_epochs", ""),
        "model_training_batch_size": checkpoint.get("batch_size", ""),
    }


class LayerPerfHooks:
    def __init__(
        self,
        *,
        logger: PhasePerfLogger,
        state: TrainingState,
        layers: List[Tuple[str, torch.nn.Module]],
        metadata: Dict[str, Any],
    ) -> None:
        self.logger = logger
        self.state = state
        self.layers = layers
        self.metadata = metadata
        self.handles: List[Any] = []
        self.active: Dict[torch.nn.Module, Any] = {}
        self.call_counts: Dict[str, int] = {}

    def __enter__(self) -> "LayerPerfHooks":
        for layer_index, (layer_name, module) in enumerate(self.layers):
            pre_hook = self._make_pre_hook(layer_index, layer_name)
            post_hook = self._make_post_hook(module)
            self.handles.append(module.register_forward_pre_hook(pre_hook))
            self.handles.append(module.register_forward_hook(post_hook))
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if self.active:
            for context in list(self.active.values()):
                context.__exit__(exc_type, exc, tb)
            self.active.clear()

    def start_batch(self, batch_idx: int) -> None:
        if self.active:
            raise RuntimeError("A layer measurement remained active across batches.")
        self.call_counts.clear()
        self.state.update(round=0, epoch=0, batch_idx=batch_idx, phase="forward")

    def _make_pre_hook(self, layer_index: int, layer_name: str) -> Any:
        def pre_hook(module: torch.nn.Module, inputs: Any) -> None:
            if self.active:
                active_names = ", ".join(type(item).__name__ for item in self.active)
                raise RuntimeError(
                    "Selected monitoring layers overlap; active parent modules: "
                    f"{active_names}"
                )
            call_index = self.call_counts.get(layer_name, 0)
            self.call_counts[layer_name] = call_index + 1
            self.state.update(phase="forward")
            context = self.logger.measure_phase(
                row_overrides={
                    **self.metadata,
                    "measurement_scope": "logical_layer",
                    "layer_index": layer_index,
                    "layer_name": layer_name,
                    "layer_type": type(module).__name__,
                    "layer_call_index": call_index,
                }
            )
            context.__enter__()
            self.active[module] = context

        return pre_hook

    def _make_post_hook(self, expected_module: torch.nn.Module) -> Any:
        def post_hook(module: torch.nn.Module, inputs: Any, output: Any) -> None:
            if module is not expected_module or module not in self.active:
                raise RuntimeError("Layer perf hook state is inconsistent.")
            context = self.active.pop(module)
            context.__exit__(None, None, None)

        return post_hook


def condition_metadata(
    args: argparse.Namespace, data_condition: str
) -> Dict[str, Any]:
    poison_fraction = get_poison_fraction(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        client_id=args.client_id,
        poisoning_method=data_condition,
        split=args.dataset_split,
    )
    return condition_columns(
        args=args,
        run_type="controlled_forward",
        poisoning_method=data_condition,
        is_poisoned_client=(data_condition != POISONING_METHOD_CLEAN),
        poisoned_client_count=int(data_condition != POISONING_METHOD_CLEAN),
        poisoned_client_ids=(
            [args.client_id] if data_condition != POISONING_METHOD_CLEAN else []
        ),
        poison_fraction=poison_fraction,
        attack_name=attack_name_for_poisoning_method(data_condition),
    )


def run_matrix_condition(
    args: argparse.Namespace,
    *,
    run_id: str,
    data_condition: str,
    model_condition: str,
    perf_events: List[str],
) -> Dict[str, Any]:
    path = checkpoint_path(
        Path(args.model_dir),
        args.model,
        args.client_id,
        model_condition,
        args.trial_id,
    )
    checkpoint = load_checkpoint(path)
    validate_checkpoint(
        checkpoint,
        path=path,
        model_name=args.model,
        client_id=args.client_id,
        model_condition=model_condition,
        trial_id=args.trial_id,
    )
    num_classes = get_num_classes(args.data_dir, dataset_name=args.dataset)
    if int(checkpoint.get("num_classes", -1)) != num_classes:
        raise ValueError(
            f"Checkpoint {path} has {checkpoint.get('num_classes')} classes; "
            f"prepared dataset has {num_classes}."
        )
    model = get_model(args.model, num_classes=num_classes)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(torch.device("cpu"))
    model.eval()
    set_all_seeds(args.seed)

    output_dir = Path(args.output_dir) / args.model / args.client_id
    output_dir.mkdir(parents=True, exist_ok=True)
    file_stem = (
        f"{run_id}_data-{data_condition}_model-{model_condition}"
    )
    whole_path = output_dir / f"{file_stem}_forward_perf.csv"
    layer_path = output_dir / f"{file_stem}_layer_perf.csv"
    summary_path = output_dir / f"{file_stem}_summary.csv"

    state = TrainingState(round=0, epoch=0, batch_idx=0, phase="idle")
    condition = condition_metadata(args, data_condition)
    condition["experiment_id"] = (
        f"{run_id}_{args.model}_{args.client_id}_"
        f"data-{data_condition}_model-{model_condition}"
    )
    condition["run_role"] = "controlled_forward_monitor"
    metadata = monitor_metadata(
        data_condition=data_condition,
        model_condition=model_condition,
        checkpoint_path=path,
        checkpoint=checkpoint,
    )

    if args.warmup_batches > 0:
        warmup_loader = make_loader(args, data_condition)
        with torch.no_grad():
            for _, (images, _) in limited_batches(
                warmup_loader, args.warmup_batches
            ):
                model(images.to("cpu"))

    criterion = torch.nn.CrossEntropyLoss()
    summaries: List[Dict[str, Any]] = []
    whole_outputs: List[torch.Tensor] = []
    whole_logger = PhasePerfLogger(
        log_dir=str(output_dir),
        condition=condition,
        training_state=state,
        path=str(whole_path),
        events=perf_events,
        extra_columns=MONITOR_COLUMNS,
    )
    with whole_logger, torch.no_grad():
        for batch_idx, (images, labels) in limited_batches(
            make_loader(args, data_condition), args.max_batches
        ):
            state.update(round=0, epoch=0, batch_idx=batch_idx, phase="forward")
            with whole_logger.measure_phase(
                row_overrides={
                    **metadata,
                    "measurement_scope": "whole_forward",
                }
            ):
                outputs = model(images.to("cpu"))
            whole_outputs.append(outputs.detach().cpu())
            labels = labels.to("cpu")
            summaries.append(
                {
                    **metadata,
                    "model": args.model,
                    "client_id": args.client_id,
                    "trial_id": args.trial_id,
                    "batch_idx": batch_idx,
                    "num_examples": int(labels.numel()),
                    "loss": float(criterion(outputs, labels).item()),
                    "accuracy": float(
                        (outputs.argmax(dim=1) == labels).float().mean().item()
                    ),
                    "layer_replay_max_abs_diff": "",
                }
            )

    if not summaries:
        raise RuntimeError(
            f"No batches were available for data condition {data_condition!r}."
        )

    layers = get_monitoring_layers(model, args.model)
    layer_logger = PhasePerfLogger(
        log_dir=str(output_dir),
        condition=condition,
        training_state=state,
        path=str(layer_path),
        events=perf_events,
        extra_columns=MONITOR_COLUMNS,
    )
    with layer_logger, LayerPerfHooks(
        logger=layer_logger,
        state=state,
        layers=layers,
        metadata=metadata,
    ) as hooks, torch.no_grad():
        for batch_idx, (images, _) in limited_batches(
            make_loader(args, data_condition), args.max_batches
        ):
            hooks.start_batch(batch_idx)
            replay_output = model(images.to("cpu")).detach().cpu()
            difference = float(
                (replay_output - whole_outputs[batch_idx]).abs().max().item()
            )
            summaries[batch_idx]["layer_replay_max_abs_diff"] = difference
            if difference > args.replay_tolerance:
                raise RuntimeError(
                    f"Layer replay changed model output at batch {batch_idx}: "
                    f"max_abs_diff={difference}"
                )

    with summary_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    print(
        f"monitored data={data_condition} model_state={model_condition} "
        f"batches={len(summaries)} whole={whole_path} layers={layer_path}"
    )
    return {
        "model": args.model,
        "client_id": args.client_id,
        "data_condition": data_condition,
        "model_condition": model_condition,
        "checkpoint_path": str(path),
        "num_batches": len(summaries),
        "num_layers": len(layers),
        "whole_perf_path": str(whole_path),
        "layer_perf_path": str(layer_path),
        "summary_path": str(summary_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.set_defaults(batch_size=DEFAULT_BATCH_SIZE)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument(
        "--output-dir", default=f"{DEFAULT_LOCAL_ML_LOG_DIR}/controlled_forward"
    )
    parser.add_argument(
        "--data-conditions", default="clean,availability_shortcuts"
    )
    parser.add_argument(
        "--model-conditions", default="clean,availability_shortcuts"
    )
    parser.add_argument("--perf-events", default="")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Maximum monitored batches per matrix cell; 0 monitors all batches.",
    )
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--replay-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--run-id", default=datetime.now().strftime("%Y%m%d%H%M%S")
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.max_batches < 0 or args.warmup_batches < 0:
        parser.error("--max-batches and --warmup-batches cannot be negative")

    data_conditions = parse_conditions(args.data_conditions)
    model_conditions = parse_conditions(args.model_conditions)
    perf_events = parse_perf_events(args.perf_events, args.host)
    manifest: List[Dict[str, Any]] = []
    for model_condition in model_conditions:
        for data_condition in data_conditions:
            manifest.append(
                run_matrix_condition(
                    args,
                    run_id=args.run_id,
                    data_condition=data_condition,
                    model_condition=model_condition,
                    perf_events=perf_events,
                )
            )

    manifest_path = (
        Path(args.output_dir)
        / args.model
        / args.client_id
        / f"{args.run_id}_monitor_manifest.csv"
    )
    with manifest_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"monitor_manifest={manifest_path}")


if __name__ == "__main__":
    main()
