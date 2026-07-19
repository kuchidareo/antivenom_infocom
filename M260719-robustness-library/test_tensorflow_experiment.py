import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

from dataset_preparation import (
    METADATA_NAME,
    PREPARED_MARKER,
    get_num_classes,
    validate_prepared_dataset,
)
from experiment_config import POISONING_METHODS


class PreparedDatasetTests(unittest.TestCase):
    def test_validates_all_iid_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            root = data_dir / "small_trashnet"
            root.mkdir()
            (root / PREPARED_MARKER).write_text("{}")

            fieldnames = [
                "image_path",
                "label",
                "client_id",
                "partition_method",
                "dataset_split",
                "is_poisoned",
                "label_changed",
                "poisoning_method",
            ]
            rows = []
            for method_index, method in enumerate(POISONING_METHODS):
                image_path = root / f"{method}.jpeg"
                image_path.touch()
                rows.append(
                    {
                        "image_path": str(image_path),
                        "label": method_index % 2,
                        "client_id": "client_1",
                        "partition_method": "iid",
                        "dataset_split": "train",
                        "is_poisoned": str(method != "clean"),
                        "label_changed": "False",
                        "poisoning_method": method,
                    }
                )
            test_path = root / "test.jpeg"
            test_path.touch()
            rows.append(
                {
                    "image_path": str(test_path),
                    "label": 1,
                    "client_id": "client_1",
                    "partition_method": "iid",
                    "dataset_split": "test",
                    "is_poisoned": "False",
                    "label_changed": "False",
                    "poisoning_method": "clean",
                }
            )
            with (root / METADATA_NAME).open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            validate_prepared_dataset(
                data_dir=str(data_dir),
                dataset_name="kuchidareo/small_trashnet",
                client_id="client_1",
                poisoning_methods=POISONING_METHODS,
            )
            self.assertEqual(get_num_classes(str(data_dir), "kuchidareo/small_trashnet"), 2)


@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow is not installed")
class TensorFlowModelTests(unittest.TestCase):
    def test_cpu_only(self) -> None:
        import tensorflow as tf

        self.assertEqual(tf.config.list_physical_devices("GPU"), [])

    def test_simple_cnn_output_shape(self) -> None:
        import tensorflow as tf

        from models import get_model

        model = get_model("simple_cnn", num_classes=6, input_size=(224, 224))
        output = model(tf.zeros((2, 224, 224, 3)), training=False)
        self.assertEqual(tuple(output.shape), (2, 6))


if __name__ == "__main__":
    unittest.main()
