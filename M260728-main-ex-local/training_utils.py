import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from hardware_logger import TrainingState


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
        train_iterator = iter(train_loader)
        batch_idx = 0
        while True:
            state.update(round=round_id, epoch=epoch, batch_idx=batch_idx, phase="dataloading")
            dataloading_start_ns = time.perf_counter_ns()
            try:
                images, labels = next(train_iterator)
            except StopIteration:
                state.update(phase="idle")
                break
            dataloading_elapsed_ms = (time.perf_counter_ns() - dataloading_start_ns) / 1_000_000
            batch_start_ns = time.perf_counter_ns()
            images, labels = images.to(device), labels.to(device)

            state.update(round=round_id, epoch=epoch, batch_idx=batch_idx, phase="forward")
            forward_start_ns = time.perf_counter_ns()
            outputs = model(images)
            loss = criterion(outputs, labels)
            forward_elapsed_ms = (time.perf_counter_ns() - forward_start_ns) / 1_000_000

            state.update(round=round_id, epoch=epoch, batch_idx=batch_idx, phase="backward")
            backward_start_ns = time.perf_counter_ns()
            optimizer.zero_grad()
            loss.backward()
            backward_elapsed_ms = (time.perf_counter_ns() - backward_start_ns) / 1_000_000

            state.update(round=round_id, epoch=epoch, batch_idx=batch_idx, phase="optimizer_step")
            optimizer_start_ns = time.perf_counter_ns()
            optimizer.step()
            optimizer_step_elapsed_ms = (
                time.perf_counter_ns() - optimizer_start_ns
            ) / 1_000_000
            batch_elapsed_ms = (time.perf_counter_ns() - batch_start_ns) / 1_000_000

            batch_size = labels.size(0)
            batch_correct = int((outputs.argmax(dim=1) == labels).sum().item())
            total_loss += float(loss.item()) * batch_size
            total_correct += batch_correct
            total_seen += batch_size
            epoch_loss += float(loss.item()) * batch_size
            epoch_correct += batch_correct
            epoch_seen += batch_size
            if metrics_logger is not None:
                metrics_logger.write(
                    state=state.snapshot(),
                    metric_event="train_batch",
                    metric_split="train",
                    loss=float(loss.item()),
                    accuracy=batch_correct / max(batch_size, 1),
                    num_examples=float(batch_size),
                    batch_size_actual=batch_size,
                    dataloading_elapsed_ms=dataloading_elapsed_ms,
                    forward_elapsed_ms=forward_elapsed_ms,
                    backward_elapsed_ms=backward_elapsed_ms,
                    optimizer_step_elapsed_ms=optimizer_step_elapsed_ms,
                    batch_elapsed_ms=batch_elapsed_ms,
                )
            state.update(phase="idle")
            batch_idx += 1
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
) -> Dict[str, float]:
    device = get_device()
    model.to(device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    state.update(round=round_id, phase="evaluation")
    for batch_idx, (images, labels) in enumerate(data_loader):
        state.update(round=round_id, batch_idx=batch_idx, phase="evaluation")
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((outputs.argmax(dim=1) == labels).sum().item())
        total_seen += batch_size
    state.update(round=round_id, phase="idle")
    if metrics_logger is not None:
        metrics_logger.write(
            state=state.snapshot(),
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
