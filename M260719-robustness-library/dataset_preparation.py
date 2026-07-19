import csv
from pathlib import Path
from typing import Any, Dict, List, Optional


METADATA_NAME = "partition_metadata.csv"
PREPARED_MARKER = "PREPARED"


def dataset_slug(dataset_name: str) -> str:
    return dataset_name.rstrip("/").split("/")[-1]


def dataset_root(data_dir: str, dataset_name: str) -> Path:
    return Path(data_dir).expanduser().resolve() / dataset_slug(dataset_name)


def load_metadata_records(
    *,
    data_dir: str,
    dataset_name: str,
    client_id: str,
    poisoning_method: str,
    split: str,
) -> List[Dict[str, Any]]:
    root = dataset_root(data_dir, dataset_name)
    metadata_path = root / METADATA_NAME
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Prepared metadata not found: {metadata_path}. "
            "Run the repository-root dataset_preparation.py first."
        )

    selected: List[Dict[str, Any]] = []
    with metadata_path.open(newline="") as file:
        for row in csv.DictReader(file):
            if (
                (client_id in {"all", "*"} or row["client_id"] == client_id)
                and row["poisoning_method"] == poisoning_method
                and row["dataset_split"] == split
            ):
                record = dict(row)
                path = Path(record["image_path"]).expanduser()
                if not path.is_absolute():
                    path = (Path.cwd() / path).resolve()
                record["image_path"] = str(path)
                selected.append(record)
    return selected


def validate_prepared_dataset(
    *,
    data_dir: str,
    dataset_name: str,
    client_id: str,
    poisoning_methods: List[str],
) -> None:
    root = dataset_root(data_dir, dataset_name)
    if not (root / PREPARED_MARKER).exists():
        raise FileNotFoundError(f"Prepared dataset marker not found: {root / PREPARED_MARKER}")

    for method in poisoning_methods:
        records = load_metadata_records(
            data_dir=data_dir,
            dataset_name=dataset_name,
            client_id=client_id,
            poisoning_method=method,
            split="train",
        )
        if not records:
            raise ValueError(f"No train records for {client_id}/{method}")
        if any(record.get("partition_method") != "iid" for record in records):
            raise ValueError(f"Expected IID records for {client_id}/{method}")
        missing = [record["image_path"] for record in records if not Path(record["image_path"]).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing image referenced by metadata: {missing[0]}")

    test_records = load_metadata_records(
        data_dir=data_dir,
        dataset_name=dataset_name,
        client_id="all",
        poisoning_method="clean",
        split="test",
    )
    if not test_records:
        raise ValueError("The prepared dataset has no clean test records")
    missing_test = [
        record["image_path"]
        for record in test_records
        if not Path(record["image_path"]).is_file()
    ]
    if missing_test:
        raise FileNotFoundError(f"Missing clean test image: {missing_test[0]}")


def get_num_classes(data_dir: str, dataset_name: str) -> int:
    metadata_path = dataset_root(data_dir, dataset_name) / METADATA_NAME
    with metadata_path.open(newline="") as file:
        labels = {int(row["label"]) for row in csv.DictReader(file)}
    if not labels:
        raise ValueError(f"No labels found in {metadata_path}")
    return len(labels)


def get_poison_fraction(
    *,
    data_dir: str,
    dataset_name: str,
    client_id: str,
    poisoning_method: str,
    split: str,
) -> float:
    records = load_metadata_records(
        data_dir=data_dir,
        dataset_name=dataset_name,
        client_id=client_id,
        poisoning_method=poisoning_method,
        split=split,
    )
    poisoned = sum(
        record.get("is_poisoned", "").lower() == "true"
        or record.get("label_changed", "").lower() == "true"
        for record in records
    )
    return poisoned / len(records) if records else 0.0


def get_dataset(
    *,
    data_dir: str,
    dataset_name: str,
    client_id: str,
    poisoning_method: str,
    split: str,
    augment: Dict[str, Any],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Any:
    import tensorflow as tf

    records = load_metadata_records(
        data_dir=data_dir,
        dataset_name=dataset_name,
        client_id=client_id,
        poisoning_method=poisoning_method,
        split=split,
    )
    if not records:
        raise ValueError(f"No records for client={client_id}, method={poisoning_method}, split={split}")

    paths = [record["image_path"] for record in records]
    labels = [int(record["label"]) for record in records]
    resize = tuple(int(value) for value in augment["resize"])
    horizontal_flip = bool(augment.get("enabled", True) and augment.get("horizontal_flip", False))
    normalize = bool(augment.get("normalize", False))
    mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

    def decode(path: Any, label: Any) -> Any:
        image = tf.io.decode_image(
            tf.io.read_file(path), channels=3, expand_animations=False
        )
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, resize, antialias=True)
        image = tf.cast(image, tf.float32) / 255.0
        if horizontal_flip:
            image = tf.image.random_flip_left_right(image, seed=seed)
        if normalize:
            image = (image - mean) / std
        return image, tf.cast(label, tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(len(records), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.map(decode, num_parallel_calls=1, deterministic=True)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    options = tf.data.Options()
    options.experimental_deterministic = True
    return dataset.with_options(options)
