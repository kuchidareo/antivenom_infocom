from typing import Any, Callable, Dict, Optional

import tensorflow as tf

from hardware_logger import TrainingState


def train_model(
    *,
    model: tf.keras.Model,
    train_dataset: Any,
    epochs: int,
    learning_rate: float,
    state: TrainingState,
    round_id: Any = 0,
    metrics_logger: Any = None,
    epoch_end_callback: Optional[Callable[[int], None]] = None,
) -> Dict[str, float]:
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_seen = 0
        iterator = iter(train_dataset)
        batch_idx = 0
        while True:
            state.update(round=round_id, epoch=epoch, batch_idx=batch_idx, phase="idle")
            try:
                images, labels = next(iterator)
            except StopIteration:
                break

            state.update(round=round_id, epoch=epoch, batch_idx=batch_idx, phase="forward")
            with tf.GradientTape() as tape:
                outputs = model(images, training=True)
                loss = loss_fn(labels, outputs)

            state.update(round=round_id, epoch=epoch, batch_idx=batch_idx, phase="backward")
            gradients = tape.gradient(loss, model.trainable_variables)
            gradient_pairs = [
                (gradient, variable)
                for gradient, variable in zip(gradients, model.trainable_variables)
                if gradient is not None
            ]

            state.update(round=round_id, epoch=epoch, batch_idx=batch_idx, phase="optimizer_step")
            optimizer.apply_gradients(gradient_pairs)
            state.update(round=round_id, epoch=epoch, batch_idx=batch_idx, phase="idle")

            batch_size = int(tf.shape(labels)[0].numpy())
            batch_loss = float(loss.numpy())
            predictions = tf.argmax(outputs, axis=1, output_type=labels.dtype)
            batch_correct = int(tf.reduce_sum(tf.cast(predictions == labels, tf.int32)).numpy())
            total_loss += batch_loss * batch_size
            total_correct += batch_correct
            total_seen += batch_size
            epoch_loss += batch_loss * batch_size
            epoch_correct += batch_correct
            epoch_seen += batch_size
            if metrics_logger is not None:
                metrics_logger.write(
                    state=state.snapshot(),
                    metric_event="train_batch",
                    metric_split="train",
                    loss=batch_loss,
                    accuracy=batch_correct / max(batch_size, 1),
                    num_examples=float(batch_size),
                )
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


def evaluate_model(
    *,
    model: tf.keras.Model,
    data_dataset: Any,
    state: TrainingState,
    round_id: Any = 0,
    metrics_logger: Any = None,
    metric_event: str = "eval_summary",
    metric_split: str = "eval",
    condition_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    state.update(round=round_id, phase="evaluation")

    for batch_idx, (images, labels) in enumerate(data_dataset):
        state.update(round=round_id, batch_idx=batch_idx, phase="evaluation")
        outputs = model(images, training=False)
        loss = loss_fn(labels, outputs)
        batch_size = int(tf.shape(labels)[0].numpy())
        predictions = tf.argmax(outputs, axis=1, output_type=labels.dtype)
        total_loss += float(loss.numpy()) * batch_size
        total_correct += int(tf.reduce_sum(tf.cast(predictions == labels, tf.int32)).numpy())
        total_seen += batch_size

    state.update(round=round_id, phase="idle")
    result = {
        "loss": total_loss / max(total_seen, 1),
        "accuracy": total_correct / max(total_seen, 1),
        "num_examples": float(total_seen),
    }
    if metrics_logger is not None:
        metrics_logger.write(
            state=state.snapshot(),
            metric_event=metric_event,
            metric_split=metric_split,
            loss=result["loss"],
            accuracy=result["accuracy"],
            num_examples=result["num_examples"],
            condition_overrides=condition_overrides,
        )
    return result
