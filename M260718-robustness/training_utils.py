from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch

from hardware_logger import TrainingState


@contextmanager
def training_phase(
    *,
    state: TrainingState,
    phase_perf_logger: Any,
    round_id: Any,
    epoch: Any,
    batch_idx: Any,
    phase: str,
) -> Iterator[None]:
    """Align the shared phase annotation with optional gated perf counters."""
    state.update(round=round_id, epoch=epoch, batch_idx=batch_idx, phase=phase)
    if phase_perf_logger is None:
        yield
        return
    try:
        with phase_perf_logger.measure_phase():
            yield
    finally:
        state.update(phase="idle")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(
    *,
    model: torch.nn.Module,
    train_loader: Any,
    epochs: int,
    learning_rate: float,
    state: TrainingState,
    round_id: Any = 0,
    metrics_logger: Any = None,
    epoch_end_callback: Optional[Callable[[int], None]] = None,
    phase_perf_logger: Any = None,
) -> Dict[str, float]:
    device = get_device()
    model.to(device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_seen = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            with training_phase(
                state=state,
                phase_perf_logger=phase_perf_logger,
                round_id=round_id,
                epoch=epoch,
                batch_idx=batch_idx,
                phase="forward",
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            with training_phase(
                state=state,
                phase_perf_logger=phase_perf_logger,
                round_id=round_id,
                epoch=epoch,
                batch_idx=batch_idx,
                phase="backward",
            ):
                optimizer.zero_grad()
                loss.backward()

            with training_phase(
                state=state,
                phase_perf_logger=phase_perf_logger,
                round_id=round_id,
                epoch=epoch,
                batch_idx=batch_idx,
                phase="optimizer_step",
            ):
                optimizer.step()

            batch_size = labels.size(0)
            batch_correct = int((outputs.argmax(dim=1) == labels).sum().item())
            total_loss += float(loss.item()) * batch_size
            total_correct += batch_correct
            total_seen += batch_size
            epoch_loss += float(loss.item()) * batch_size
            epoch_correct += batch_correct
            epoch_seen += batch_size
            if metrics_logger is not None:
                metric_state = state.snapshot()
                metric_state["phase"] = "optimizer_step"
                metrics_logger.write(
                    state=metric_state,
                    metric_event="train_batch",
                    metric_split="train",
                    loss=float(loss.item()),
                    accuracy=batch_correct / max(batch_size, 1),
                    num_examples=float(batch_size),
                )
        if metrics_logger is not None:
            metrics_logger.write(
                state=state.snapshot(),
                metric_event="train_epoch",
                metric_split="train",
                loss=epoch_loss / max(epoch_seen, 1),
                accuracy=epoch_correct / max(epoch_seen, 1),
                num_examples=float(epoch_seen),
            )
        if epoch_end_callback is not None:
            epoch_end_callback(epoch)

    state.update(round=round_id, phase="idle")
    if metrics_logger is not None:
        metrics_logger.write(
            state=state.snapshot(),
            metric_event="train_summary",
            metric_split="train",
            loss=total_loss / max(total_seen, 1),
            accuracy=total_correct / max(total_seen, 1),
            num_examples=float(total_seen),
        )
    return {
        "train_loss": total_loss / max(total_seen, 1),
        "train_accuracy": total_correct / max(total_seen, 1),
        "num_examples": float(total_seen),
    }


@torch.no_grad()
def evaluate_model(
    *,
    model: torch.nn.Module,
    data_loader: Any,
    state: TrainingState,
    round_id: Any = 0,
    metrics_logger: Any = None,
    metric_event: str = "eval_summary",
    metric_split: str = "eval",
    condition_overrides: Optional[Dict[str, Any]] = None,
    phase_perf_logger: Any = None,
) -> Dict[str, float]:
    device = get_device()
    model.to(device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for batch_idx, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        with training_phase(
            state=state,
            phase_perf_logger=phase_perf_logger,
            round_id=round_id,
            epoch=state.snapshot()["epoch"],
            batch_idx=batch_idx,
            phase="evaluation",
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((outputs.argmax(dim=1) == labels).sum().item())
        total_seen += batch_size
    state.update(round=round_id, phase="idle")
    if metrics_logger is not None:
        metric_state = state.snapshot()
        metric_state["phase"] = "evaluation"
        metrics_logger.write(
            state=metric_state,
            metric_event=metric_event,
            metric_split=metric_split,
            loss=total_loss / max(total_seen, 1),
            accuracy=total_correct / max(total_seen, 1),
            num_examples=float(total_seen),
            condition_overrides=condition_overrides,
        )
    return {
        "loss": total_loss / max(total_seen, 1),
        "accuracy": total_correct / max(total_seen, 1),
        "num_examples": float(total_seen),
    }


def get_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    return [value.detach().cpu().numpy() for _, value in model.state_dict().items()]


def set_parameters(model: torch.nn.Module, parameters: Iterable[np.ndarray]) -> None:
    state_dict = model.state_dict()
    new_state = {
        key: torch.tensor(value)
        for key, value in zip(state_dict.keys(), parameters)
    }
    model.load_state_dict(new_state, strict=True)
