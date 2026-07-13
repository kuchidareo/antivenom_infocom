from typing import Any, Dict, Iterable, List, Tuple

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
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_seen = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            state.update(round=round_id, epoch=epoch, batch_idx=batch_idx, phase="forward")
            outputs = model(images)
            loss = criterion(outputs, labels)

            state.update(round=round_id, epoch=epoch, batch_idx=batch_idx, phase="backward")
            optimizer.zero_grad()
            loss.backward()

            state.update(round=round_id, epoch=epoch, batch_idx=batch_idx, phase="optimizer_step")
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
                metrics_logger.write(
                    state=state.snapshot(),
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
            metric_event="eval_summary",
            metric_split="eval",
            loss=total_loss / max(total_seen, 1),
            accuracy=total_correct / max(total_seen, 1),
            num_examples=float(total_seen),
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
