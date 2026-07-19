from typing import Tuple

import tensorflow as tf


def get_model(
    model_name: str,
    num_classes: int,
    input_size: Tuple[int, int] = (224, 224),
) -> tf.keras.Model:
    if model_name not in {"simple_cnn", "SimpleCNN"}:
        raise ValueError(f"Unknown TensorFlow model: {model_name}")

    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(*input_size, 3)),
            tf.keras.layers.Conv2D(16, 3, padding="same"),
            tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(32, 3, padding="same"),
            tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(64, 3, padding="same"),
            tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5),
            tf.keras.layers.ReLU(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(num_classes),
        ],
        name="simple_cnn",
    )
