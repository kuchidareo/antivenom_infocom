import argparse
import colorsys
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

try:
    import experiment_config as _experiment_config
except ModuleNotFoundError:
    _experiment_config = None


def _config_value(name: str, fallback: Any) -> Any:
    if _experiment_config is None:
        return fallback
    return getattr(_experiment_config, name, fallback)


DATASET_NAME = _config_value("DATASET_NAME", "kuchidareo/small_trashnet")
DEFAULT_AUGMENT = _config_value(
    "DEFAULT_AUGMENT",
    {
        "enabled": True,
        "resize": [224, 224],
        "horizontal_flip": True,
        "normalize": True,
    },
)
DEFAULT_AUGMENTATION_PROFILE = _config_value("DEFAULT_AUGMENTATION_PROFILE", "baseline")
DEFAULT_AUGMENTATION_SEED = _config_value("DEFAULT_AUGMENTATION_SEED", 260719)
AUGMENTATION_PROFILES = _config_value(
    "AUGMENTATION_PROFILES",
    {
        "baseline": DEFAULT_AUGMENT,
        "moderate": {
            "enabled": True,
            "resize": [224, 224],
            "random_resized_crop_scale": [0.8, 1.0],
            "random_resized_crop_ratio": [0.85, 1.18],
            "horizontal_flip": True,
            "rotation_degrees": 15,
            "color_jitter": [0.2, 0.2, 0.2, 0.05],
            "normalize": True,
        },
        "strong": {
            "enabled": True,
            "resize": [224, 224],
            "random_resized_crop_scale": [0.4, 1.0],
            "random_resized_crop_ratio": [0.6, 1.67],
            "horizontal_flip": True,
            "vertical_flip": True,
            "rotation_degrees": 45,
            "perspective_distortion": 0.5,
            "perspective_probability": 0.5,
            "color_jitter": [0.6, 0.6, 0.6, 0.2],
            "random_grayscale_probability": 0.2,
            "gaussian_blur_kernel_size": 9,
            "gaussian_blur_probability": 0.3,
            "random_erasing_probability": 0.5,
            "random_erasing_scale": [0.05, 0.35],
            "normalize": True,
        },
    },
)
DEFAULT_AVAILABILITY_SHORTCUT_EPS = _config_value("DEFAULT_AVAILABILITY_SHORTCUT_EPS", 6.0)
DEFAULT_AVAILABILITY_SHORTCUT_PATCH_SIZE = _config_value(
    "DEFAULT_AVAILABILITY_SHORTCUT_PATCH_SIZE", 8
)
DEFAULT_BATCH_SIZE = _config_value("DEFAULT_BATCH_SIZE", 16)
DEFAULT_BADSAMPLER_KAPPA = _config_value("DEFAULT_BADSAMPLER_KAPPA", 2.0)
DEFAULT_DATA_DIR = _config_value("DEFAULT_DATA_DIR", "iid-data")
DEFAULT_NUM_CLIENTS = _config_value("DEFAULT_NUM_CLIENTS", 10)
DEFAULT_TEST_FRACTION = _config_value("DEFAULT_TEST_FRACTION", 0.2)
DEFAULT_TEST_SEED = _config_value("DEFAULT_TEST_SEED", 260626)
DEFAULT_RANDOM_LABEL_FLIP_FRACTION = _config_value(
    "DEFAULT_RANDOM_LABEL_FLIP_FRACTION", 1.0 / 6.0
)
DEFAULT_TARGET_LABEL_FLIP_REPLACEMENT_LABEL = _config_value(
    "DEFAULT_TARGET_LABEL_FLIP_REPLACEMENT_LABEL", 3
)
DEFAULT_TARGET_LABEL_FLIP_TARGET_LABEL = _config_value(
    "DEFAULT_TARGET_LABEL_FLIP_TARGET_LABEL", 5
)
DEFAULT_PARTITION_METHOD = _config_value("DEFAULT_PARTITION_METHOD", "iid")
DEFAULT_NONIID_ALPHA = _config_value("DEFAULT_NONIID_ALPHA", 0.3)

POISONING_METHOD_AVAILABILITY_SHORTCUTS = _config_value(
    "POISONING_METHOD_AVAILABILITY_SHORTCUTS", "availability_shortcuts"
)
POISONING_METHOD_CLEAN = _config_value("POISONING_METHOD_CLEAN", "clean")
POISONING_METHOD_RANDOM_LABEL_FLIPPING = _config_value(
    "POISONING_METHOD_RANDOM_LABEL_FLIPPING", "random_label_flipping"
)
POISONING_METHOD_TARGET_LABEL_FLIPPING = _config_value(
    "POISONING_METHOD_TARGET_LABEL_FLIPPING", "target_label_flipping"
)
POISONING_METHOD_UNLEARNABLE_EXAMPLES = _config_value(
    "POISONING_METHOD_UNLEARNABLE_EXAMPLES", "unlearnable_examples"
)
POISONING_METHOD_BADSAMPLING = _config_value(
    "POISONING_METHOD_BADSAMPLING", "badsampling"
)


def set_all_seeds(seed: int) -> None:
    configured = getattr(_experiment_config, "set_all_seeds", None)
    if configured is not None:
        configured(seed)
        return
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def augment_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    configured = getattr(_experiment_config, "augment_from_args", None)
    if configured is not None:
        return configured(args)
    profile = getattr(args, "augmentation_profile", DEFAULT_AUGMENTATION_PROFILE)
    if profile not in AUGMENTATION_PROFILES:
        raise ValueError(
            f"Unknown augmentation profile {profile!r}; "
            f"choose one of {', '.join(AUGMENTATION_PROFILES)}"
        )
    augment = dict(AUGMENTATION_PROFILES[profile])
    value = getattr(args, "augment", "{}")
    if isinstance(value, str):
        value = json.loads(value)
    augment.update(dict(value))
    augment["_profile"] = profile
    return augment


def add_common_args(parser: argparse.ArgumentParser) -> None:
    configured = getattr(_experiment_config, "add_common_args", None)
    if configured is not None:
        configured(parser)
        return
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--num-clients", type=int, default=DEFAULT_NUM_CLIENTS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--augmentation-profile",
        choices=tuple(AUGMENTATION_PROFILES),
        default=DEFAULT_AUGMENTATION_PROFILE,
    )
    parser.add_argument(
        "--augment",
        default="{}",
        help="JSON overrides applied after the selected augmentation profile.",
    )


METADATA_NAME = "partition_metadata.csv"
PREPARED_MARKER = "PREPARED"
REFERENCE_DATASET_NAME = "kuchidareo/small_trashnet"
MATERIALIZED_AUGMENTATION_DATASET_SLUGS = {"small_trashnet", "cifar10"}
MATCH_REFERENCE_SIZE_DATASETS = {
    "kuchidareo/chinese_trafficsign_dataset",
    "uoft-cs/cifar10",
}
PREPARE_SCENARIO_ALL = "all"
PREPARE_SCENARIOS = [
    POISONING_METHOD_CLEAN,
    POISONING_METHOD_UNLEARNABLE_EXAMPLES,
    POISONING_METHOD_RANDOM_LABEL_FLIPPING,
    POISONING_METHOD_TARGET_LABEL_FLIPPING,
    POISONING_METHOD_AVAILABILITY_SHORTCUTS,
    POISONING_METHOD_BADSAMPLING,
]
AVAILABILITY_SHORTCUT_GENERATOR = "classwise_color_patch_jpeg_v4"
AVAILABILITY_SHORTCUT_CLASS_SEP = 10.0
AVAILABILITY_SHORTCUT_REFERENCE_IMAGE_SIZE = 32
AVAILABILITY_SHORTCUT_FREQUENCY = 3.0
AVAILABILITY_SHORTCUT_MANIFEST_NAME = "shortcut_bank.json"
AVAILABILITY_SHORTCUT_JPEG_QUALITY = 100
AVAILABILITY_SHORTCUT_JPEG_SUBSAMPLING = 0
AUGMENTATION_PROFILE_DIR = "augmented"
AUGMENTATION_VARIANT_NAMES = ("moderate", "strong")
BADSAMPLER_PLAN_NAME = "sampling_plan.json"
UE_GENERATION_VERSION = "alternating_samplewise_min_min_v2"
UE_MANIFEST_NAME = "ue_manifest.json"
DEFAULT_UNLEARNABLE_REPO = str(
    Path(__file__).resolve().parent / "Unlearnable-Examples"
)

METADATA_FIELDNAMES = [
    "image_path",
    "label",
    "class_name",
    "original_label",
    "original_class_name",
    "label_changed",
    "label_flip_fraction",
    "target_label",
    "replacement_label",
    "shortcut_eps",
    "shortcut_patch_size",
    "shortcut_frequency",
    "shortcut_seed",
    "shortcut_num_classes",
    "shortcut_generator",
    "shortcut_operation",
    "shortcut_class_sep",
    "shortcut_jpeg_quality",
    "shortcut_jpeg_subsampling",
    "ue_epsilon",
    "ue_steps",
    "ue_step_size",
    "ue_warmup_epochs",
    "ue_outer_iterations",
    "ue_surrogate_steps_per_outer",
    "ue_stop_error",
    "ue_surrogate",
    "ue_generation_version",
    "client_id",
    "partition_id",
    "partition_method",
    "noniid_alpha",
    "dataset_split",
    "is_poisoned",
    "poisoning_method",
    "source_index",
    "relative_path",
]


def _require_torch():
    import torch
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, Dataset

    return torch, transforms, DataLoader, Dataset


def _safe_class_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(name))


def dataset_slug(dataset_name: Optional[str] = None) -> str:
    name = dataset_name or os.environ.get("DATASET_NAME", DATASET_NAME)
    slug = name.rstrip("/").split("/")[-1]
    return _safe_class_name(slug)


def _dataset_root(data_dir: str, dataset_name: Optional[str] = None) -> Path:
    return Path(data_dir) / dataset_slug(dataset_name)


def _canonical_partition_method(partition_method: str) -> str:
    method = partition_method.lower()
    if method == "iid":
        return "iid"
    if method in {"dirichlet_noniid", "non_iid_dirichlet", "noniid_dirichlet"}:
        return "dirichlet_noniid"
    raise ValueError(
        f"Unknown partition_method={partition_method!r}. "
        "Use 'iid' or 'dirichlet_noniid'."
    )


def _metadata_partition_method(rows: Sequence[Dict[str, Any]]) -> str:
    methods = {
        _canonical_partition_method(row["partition_method"])
        for row in rows
        if row.get("partition_method")
    }
    if len(methods) > 1:
        raise ValueError(f"Metadata contains mixed partition methods: {sorted(methods)}")
    # Metadata produced before partition_method was recorded was IID.
    return next(iter(methods), "iid")


def _fill_partition_metadata(
    rows: Sequence[Dict[str, Any]],
    *,
    partition_method: str,
    noniid_alpha: float,
) -> bool:
    changed = False
    alpha_value: Any = noniid_alpha if partition_method != "iid" else ""
    for row in rows:
        if not row.get("partition_method"):
            row["partition_method"] = partition_method
            changed = True
        if "noniid_alpha" not in row or (
            partition_method != "iid" and not row.get("noniid_alpha")
        ):
            row["noniid_alpha"] = alpha_value
            changed = True
    return changed


def _resolve_metadata_image_path(
    image_path: str,
    data_dir: str,
    dataset_name: Optional[str] = None,
) -> str:
    """Resolve old metadata paths after moving data_dir.

    Older metadata stores paths such as data/small_trashnet/clean/...
    When data_dir is now ../iid-data, the correct path is
    ../iid-data/small_trashnet/clean/...
    """
    root = _dataset_root(data_dir, dataset_name)
    raw = Path(image_path)
    slug = dataset_slug(dataset_name)
    candidates: List[Path] = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        parts = raw.parts
        if slug in parts:
            slug_idx = parts.index(slug)
            candidates.append(root.joinpath(*parts[slug_idx + 1 :]))
        candidates.extend([root / raw, Path(data_dir) / raw, raw])

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0] if candidates else raw)


def prepared_data_exists(
    data_dir: str,
    num_clients: int = DEFAULT_NUM_CLIENTS,
    dataset_name: Optional[str] = None,
    partition_method: Optional[str] = None,
    expected_ue_manifest: Optional[Dict[str, Any]] = None,
) -> bool:
    root = _dataset_root(data_dir, dataset_name)
    if not (root / METADATA_NAME).exists() or not (root / PREPARED_MARKER).exists():
        return False
    rows = _read_metadata_rows(root)
    if partition_method is not None and _metadata_partition_method(rows) != _canonical_partition_method(
        partition_method
    ):
        return False
    if not any(
        row.get("poisoning_method") == POISONING_METHOD_CLEAN
        and row.get("dataset_split") == "test"
        for row in rows
    ):
        return False
    shortcut_rows = [
        row
        for row in rows
        if row.get("poisoning_method") == POISONING_METHOD_AVAILABILITY_SHORTCUTS
    ]
    if not shortcut_rows or any(
        row.get("shortcut_generator") != AVAILABILITY_SHORTCUT_GENERATOR
        for row in shortcut_rows
    ):
        return False
    if expected_ue_manifest is not None and not _unlearnable_manifest_matches(
        root / "poisoned" / POISONING_METHOD_UNLEARNABLE_EXAMPLES,
        expected_ue_manifest,
    ):
        return False
    required_modes = [
        ("clean",),
        (f"poisoned/{POISONING_METHOD_UNLEARNABLE_EXAMPLES}",),
        (f"poisoned/{POISONING_METHOD_RANDOM_LABEL_FLIPPING}",),
        (f"poisoned/{POISONING_METHOD_TARGET_LABEL_FLIPPING}",),
        (f"poisoned/{POISONING_METHOD_AVAILABILITY_SHORTCUTS}",),
        (f"poisoned/{POISONING_METHOD_BADSAMPLING}",),
    ]
    for mode_options in required_modes:
        for idx in range(num_clients):
            if not any((root / mode / f"client_{idx}").exists() for mode in mode_options):
                return False
    return True


def _extract_image_label(example: Dict[str, Any]) -> Tuple[Image.Image, int]:
    image = None
    for key in ("image", "img", "pixel_values"):
        if key in example:
            image = example[key]
            break
    if image is None:
        for value in example.values():
            if isinstance(value, Image.Image):
                image = value
                break
    label = None
    for key in ("label", "labels", "fine_label", "class", "target"):
        if key in example:
            label = example[key]
            break
    if image is None or label is None:
        raise ValueError(f"Could not infer image/label fields from dataset example keys: {list(example.keys())}")
    if not isinstance(image, Image.Image):
        if isinstance(image, (str, os.PathLike)):
            image = Image.open(image)
        else:
            try:
                import numpy as np

                image = Image.fromarray(np.asarray(image))
            except Exception as exc:
                raise ValueError(f"Could not convert image value of type {type(image)!r} to PIL.Image") from exc
    return image.convert("RGB"), int(label)


def _class_names(ds: Any) -> List[str]:
    features = getattr(ds, "features", {})
    label_feature = None
    if hasattr(features, "get"):
        for key in ("label", "labels", "fine_label", "class", "target"):
            label_feature = features.get(key)
            if label_feature is not None:
                break
    names = getattr(label_feature, "names", None)
    if names:
        return list(names)
    labels = sorted({_extract_image_label(row)[1] for row in ds})
    return [str(label) for label in labels]


def _iter_splits(dataset_dict: Any) -> Iterable[Tuple[str, Any]]:
    if hasattr(dataset_dict, "keys"):
        for split in dataset_dict.keys():
            yield split, dataset_dict[split]
    else:
        yield "train", dataset_dict


def _should_match_reference_size(dataset_name: str) -> bool:
    return dataset_name in MATCH_REFERENCE_SIZE_DATASETS


def _reference_clean_split_counts(
    data_dir: str,
    reference_dataset_name: str = REFERENCE_DATASET_NAME,
) -> Dict[str, int]:
    metadata_path = _dataset_root(data_dir, reference_dataset_name) / METADATA_NAME
    if metadata_path.exists():
        counts: Dict[str, int] = {}
        with metadata_path.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("poisoning_method") == POISONING_METHOD_CLEAN:
                    split = row.get("dataset_split", "train")
                    counts[split] = counts.get(split, 0) + 1
        if counts:
            return counts

    from datasets import load_dataset

    reference = load_dataset(reference_dataset_name)
    raw_counts = {split: len(ds) for split, ds in _iter_splits(reference)}
    if "test" not in raw_counts and "train" in raw_counts:
        test_count = int(round(raw_counts["train"] * DEFAULT_TEST_FRACTION))
        return {
            "train": raw_counts["train"] - test_count,
            "test": test_count,
        }
    return raw_counts


def _balanced_subset_indices(ds: Any, target_count: int, seed: int) -> List[int]:
    from collections import defaultdict

    if target_count >= len(ds):
        return list(range(len(ds)))
    if target_count <= 0:
        return []

    by_label: Dict[int, List[int]] = defaultdict(list)
    for idx, example in enumerate(ds):
        _, label = _extract_image_label(example)
        by_label[label].append(idx)

    rng = random.Random(seed)
    total = sum(len(indices) for indices in by_label.values())
    allocations: Dict[int, int] = {}
    fractions: List[Tuple[float, int]] = []
    allocated = 0
    for label, indices in by_label.items():
        exact = target_count * len(indices) / total
        count = min(len(indices), int(exact))
        allocations[label] = count
        allocated += count
        fractions.append((exact - int(exact), label))

    for _, label in sorted(fractions, reverse=True):
        if allocated >= target_count:
            break
        if allocations[label] < len(by_label[label]):
            allocations[label] += 1
            allocated += 1

    selected: List[int] = []
    for label in sorted(by_label):
        indices = list(by_label[label])
        rng.shuffle(indices)
        selected.extend(indices[: allocations[label]])
    rng.shuffle(selected)
    return selected[:target_count]


def _maybe_match_reference_split_size(
    ds: Any,
    *,
    dataset_name: str,
    split: str,
    data_dir: str,
    seed: int,
    has_explicit_test_split: bool,
) -> Tuple[Any, List[int]]:
    source_indices = list(range(len(ds)))
    if not _should_match_reference_size(dataset_name):
        return ds, source_indices

    reference_counts = _reference_clean_split_counts(data_dir)
    if split == "train" and not has_explicit_test_split:
        target_count = sum(reference_counts.values())
    else:
        target_count = reference_counts.get(split, 0)
    target_count = min(target_count, len(ds))
    selected_indices = _balanced_subset_indices(ds, target_count, seed)
    return ds.select(selected_indices), selected_indices


def _assign_iid_partitions(ds: Any, num_clients: int, seed: int) -> Dict[int, str]:
    import random
    from collections import defaultdict

    by_label: Dict[int, List[int]] = defaultdict(list)
    for idx, example in enumerate(ds):
        _, label = _extract_image_label(example)
        by_label[label].append(idx)

    assignment: Dict[int, str] = {}
    rng = random.Random(seed)
    for label in sorted(by_label):
        indices = list(by_label[label])
        rng.shuffle(indices)
        for offset, item_idx in enumerate(indices):
            assignment[item_idx] = f"client_{offset % num_clients}"
    return assignment


def _assign_dirichlet_noniid_partitions(
    ds: Any,
    num_clients: int,
    seed: int,
    alpha: float,
) -> Dict[int, str]:
    """Assign label-skew partitions while guaranteeing non-empty clients."""
    import numpy as np
    from collections import defaultdict

    if alpha <= 0:
        raise ValueError(f"noniid_alpha must be > 0, got {alpha}")
    if num_clients <= 0:
        raise ValueError(f"num_clients must be > 0, got {num_clients}")

    by_label: Dict[int, List[int]] = defaultdict(list)
    for idx, example in enumerate(ds):
        _, label = _extract_image_label(example)
        by_label[label].append(idx)

    rng = np.random.default_rng(seed)
    client_indices: Dict[int, List[int]] = {idx: [] for idx in range(num_clients)}
    for label in sorted(by_label):
        indices = np.asarray(by_label[label], dtype=np.int64)
        rng.shuffle(indices)
        proportions = rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
        cuts = (np.cumsum(proportions)[:-1] * len(indices)).astype(int)
        for client_idx, shard in enumerate(np.split(indices, cuts)):
            client_indices[client_idx].extend(int(item) for item in shard)

    for empty_client in [idx for idx, values in client_indices.items() if not values]:
        donor = max(client_indices, key=lambda idx: len(client_indices[idx]))
        if len(client_indices[donor]) <= 1:
            raise ValueError("Could not rebalance non-IID partitions without empty clients.")
        client_indices[empty_client].append(client_indices[donor].pop())

    return {
        item_idx: f"client_{client_idx}"
        for client_idx, indices in client_indices.items()
        for item_idx in indices
    }


def _assign_partitions(
    ds: Any,
    *,
    num_clients: int,
    seed: int,
    partition_method: str,
    noniid_alpha: float,
) -> Dict[int, str]:
    method = _canonical_partition_method(partition_method)
    if method == "iid":
        return _assign_iid_partitions(ds, num_clients, seed)
    if method == "dirichlet_noniid":
        return _assign_dirichlet_noniid_partitions(ds, num_clients, seed, noniid_alpha)
    raise AssertionError(f"Unhandled canonical partition method: {method}")


def _save_jpeg(
    image: Image.Image,
    path: Path,
    *,
    quality: int = 95,
    subsampling: Optional[int] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    options: Dict[str, Any] = {"format": "JPEG", "quality": quality}
    if subsampling is not None:
        options["subsampling"] = subsampling
    image.convert("RGB").save(path, **options)


def _read_metadata_rows(root: Path) -> List[Dict[str, Any]]:
    path = root / METADATA_NAME
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_metadata_rows(root: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with (root / METADATA_NAME).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _normalize_metadata_paths(
    rows: Sequence[Dict[str, Any]],
    *,
    data_dir: str,
    dataset_name: str,
) -> bool:
    root = _dataset_root(data_dir, dataset_name)
    resolved_root = root.resolve()
    changed = False
    for row in rows:
        resolved_path = Path(
            _resolve_metadata_image_path(row["image_path"], data_dir, dataset_name)
        ).resolve()
        try:
            relative_path = resolved_path.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(
                f"Metadata image is outside dataset root {resolved_root}: {resolved_path}"
            ) from exc
        normalized_path = str(root / relative_path)
        if row.get("image_path") != normalized_path:
            row["image_path"] = normalized_path
            changed = True
    return changed


def prune_unreferenced_images(*, data_dir: str, dataset_name: str) -> int:
    root = _dataset_root(data_dir, dataset_name)
    rows = _read_metadata_rows(root)
    if not rows:
        raise FileNotFoundError(f"Prepared metadata not found in {root}")
    referenced = {
        Path(_resolve_metadata_image_path(row["image_path"], data_dir, dataset_name)).resolve()
        for row in rows
    }
    removed = 0
    image_paths = (
        path
        for suffix in ("*.jpeg", "*.jpg", "*.png")
        for path in root.rglob(suffix)
    )
    for image_path in image_paths:
        if image_path.resolve() not in referenced:
            image_path.unlink()
            removed += 1
    for directory in sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        try:
            directory.rmdir()
        except OSError:
            pass
    return removed


def _add_clean_test_split(
    rows: Sequence[Dict[str, Any]],
    *,
    test_fraction: float,
    test_seed: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """Create a deterministic clean-only test split from prepared train rows.

    The corresponding source IDs are removed from every poisoned condition, so
    no poisoned image can enter evaluation and no test source can enter
    training through another condition.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1.")
    copied = [dict(row) for row in rows]
    if any(
        row.get("poisoning_method") == POISONING_METHOD_CLEAN
        and row.get("dataset_split") == "test"
        for row in copied
    ):
        return copied, 0

    clean_train = [
        row for row in copied
        if row.get("poisoning_method") == POISONING_METHOD_CLEAN
        and row.get("dataset_split") == "train"
    ]
    if not clean_train:
        raise ValueError("Cannot create a clean test split: no clean train rows exist.")

    by_label: Dict[str, List[str]] = {}
    for row in clean_train:
        source_index = str(row.get("source_index", ""))
        if not source_index:
            raise ValueError("Cannot split metadata row without source_index.")
        by_label.setdefault(str(row["label"]), []).append(source_index)

    candidates_by_label = {
        label: sorted(set(source_indices), key=int)
        for label, source_indices in by_label.items()
    }
    eligible = {
        label: source_indices
        for label, source_indices in candidates_by_label.items()
        if len(source_indices) >= 2
    }
    if not eligible:
        raise ValueError("No class has enough samples to create a clean test split.")

    target_test_count = int(round(len(clean_train) * test_fraction))
    max_test_count = sum(len(source_indices) - 1 for source_indices in eligible.values())
    target_test_count = max(1, min(target_test_count, max_test_count))

    eligible_total = sum(len(source_indices) for source_indices in eligible.values())
    allocations: Dict[str, int] = {}
    remainders: List[Tuple[float, str]] = []
    allocated = 0
    for label, source_indices in eligible.items():
        exact = target_test_count * len(source_indices) / eligible_total
        count = min(int(exact), len(source_indices) - 1)
        allocations[label] = count
        allocated += count
        remainders.append((exact - int(exact), label))

    while allocated < target_test_count:
        made_progress = False
        for _, label in sorted(remainders, reverse=True):
            if allocations[label] >= len(eligible[label]) - 1:
                continue
            allocations[label] += 1
            allocated += 1
            made_progress = True
            if allocated == target_test_count:
                break
        if not made_progress:
            raise ValueError("Could not allocate the requested clean test split.")

    rng = random.Random(test_seed)
    test_source_indices = set()
    for label, candidates in eligible.items():
        rng.shuffle(candidates)
        test_source_indices.update(candidates[: allocations[label]])

    result: List[Dict[str, Any]] = []
    for row in copied:
        source_index = str(row.get("source_index", ""))
        is_selected_train_source = (
            row.get("dataset_split") == "train"
            and source_index in test_source_indices
        )
        if not is_selected_train_source:
            result.append(row)
            continue
        if row.get("poisoning_method") == POISONING_METHOD_CLEAN:
            row["dataset_split"] = "test"
            result.append(row)
        # Poisoned rows for selected test source IDs are intentionally omitted.

    return result, len(test_source_indices)


def _mode_complete(root: Path, mode: str, num_clients: int) -> bool:
    return all((root / mode / f"client_{idx}").exists() for idx in range(num_clients))


def _unlearnable_manifest(
    *,
    epsilon: float,
    steps: int,
    step_size: float,
    warmup_epochs: int,
    outer_iterations: int,
    surrogate_steps_per_outer: int,
    stop_error: float,
    seed: int,
    resize: Sequence[int],
    unlearnable_repo: str,
) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {
        "method": POISONING_METHOD_UNLEARNABLE_EXAMPLES,
        "epsilon": float(epsilon),
        "steps": int(steps),
        "step_size": float(step_size),
        "warmup_epochs": int(warmup_epochs),
        "outer_iterations": int(outer_iterations),
        "surrogate_steps_per_outer": int(surrogate_steps_per_outer),
        "stop_error": float(stop_error),
        "seed": int(seed),
        "resize": [int(value) for value in resize],
        "surrogate": "resnet18",
        "repo": str(Path(unlearnable_repo).resolve()),
        "generation_version": UE_GENERATION_VERSION,
    }
    serialized = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    manifest["cache_key"] = hashlib.sha256(serialized.encode()).hexdigest()
    return manifest


def _unlearnable_manifest_matches(output_root: Path, expected: Dict[str, Any]) -> bool:
    manifest_path = output_root / UE_MANIFEST_NAME
    try:
        actual = json.loads(manifest_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    return actual.get("cache_key") == expected.get("cache_key")


def _metadata_has_method(rows: Sequence[Dict[str, Any]], poisoning_method: str) -> bool:
    return any(row.get("poisoning_method") == poisoning_method for row in rows)


def _parse_prepare_scenarios(value: Optional[Any]) -> List[str]:
    if value is None or value == "" or value == PREPARE_SCENARIO_ALL:
        return list(PREPARE_SCENARIOS)
    if isinstance(value, str):
        scenarios = [item.strip() for item in value.split(",") if item.strip()]
    else:
        scenarios = [str(item).strip() for item in value if str(item).strip()]
    if not scenarios:
        return list(PREPARE_SCENARIOS)
    if PREPARE_SCENARIO_ALL in scenarios:
        if len(scenarios) > 1:
            raise ValueError(f"{PREPARE_SCENARIO_ALL!r} cannot be combined with specific scenarios.")
        return list(PREPARE_SCENARIOS)
    unknown = [scenario for scenario in scenarios if scenario not in PREPARE_SCENARIOS]
    if unknown:
        raise ValueError(
            f"Unknown prepare scenario(s): {unknown}. "
            f"Use {PREPARE_SCENARIO_ALL!r} or one or more of: {', '.join(PREPARE_SCENARIOS)}"
        )
    return scenarios


def _scenario_mode(scenario: str) -> str:
    if scenario == POISONING_METHOD_CLEAN:
        return "clean"
    return f"poisoned/{scenario}"


def _scenario_complete(
    root: Path,
    scenario: str,
    rows: Sequence[Dict[str, Any]],
    num_clients: int,
    *,
    expected_ue_manifest: Optional[Dict[str, Any]] = None,
) -> bool:
    if scenario == POISONING_METHOD_BADSAMPLING:
        return all(
            (root / "poisoned" / scenario / f"client_{idx}" / BADSAMPLER_PLAN_NAME).exists()
            for idx in range(num_clients)
        )
    if not (_mode_complete(root, _scenario_mode(scenario), num_clients) and _metadata_has_method(rows, scenario)):
        return False
    if scenario == POISONING_METHOD_UNLEARNABLE_EXAMPLES:
        if expected_ue_manifest is None:
            return False
        return _unlearnable_manifest_matches(
            root / "poisoned" / POISONING_METHOD_UNLEARNABLE_EXAMPLES,
            expected_ue_manifest,
        )
    if scenario == POISONING_METHOD_AVAILABILITY_SHORTCUTS:
        shortcut_rows = [
            row
            for row in rows
            if row.get("poisoning_method") == POISONING_METHOD_AVAILABILITY_SHORTCUTS
        ]
        return bool(shortcut_rows) and all(
            row.get("shortcut_generator") == AVAILABILITY_SHORTCUT_GENERATOR
            for row in shortcut_rows
        )
    return True


def _clean_records_from_metadata(
    rows: Sequence[Dict[str, Any]],
    *,
    data_dir: str,
    dataset_name: Optional[str],
) -> List[Dict[str, Any]]:
    records = [dict(row) for row in rows if row.get("poisoning_method") == POISONING_METHOD_CLEAN]
    for record in records:
        resolved = _resolve_metadata_image_path(record["image_path"], data_dir, dataset_name)
        record["image_path"] = resolved
        record["clean_path"] = resolved
    return records


def _class_names_from_records(records: Sequence[Dict[str, Any]]) -> List[str]:
    label_to_name = {
        int(record["label"]): str(record.get("class_name", record["label"]))
        for record in records
    }
    return [label_to_name[label] for label in sorted(label_to_name)]


def _pil_from_tensor(tensor: Any) -> Image.Image:
    import torchvision.transforms.functional as TF

    return TF.to_pil_image(tensor.detach().cpu().clamp(0, 1))


def _build_unlearnable_example_images(
    clean_records: Sequence[Dict[str, Any]],
    *,
    output_root: Path,
    num_classes: int,
    resize: Sequence[int],
    seed: int,
    epsilon: float,
    num_steps: int,
    step_size: float,
    batch_size: int,
    unlearnable_repo: str,
    warmup_epochs: int,
    outer_iterations: int,
    surrogate_steps_per_outer: int,
    stop_error: float,
) -> Dict[str, Any]:
    import numpy as np

    torch, transforms, DataLoader, Dataset = _require_torch()

    repo_path = Path(unlearnable_repo).resolve()
    toolbox_path = repo_path / "toolbox.py"
    if not toolbox_path.is_file():
        raise FileNotFoundError(
            f"Official Unlearnable-Examples repository was not found: {repo_path}"
        )
    module_spec = importlib.util.spec_from_file_location(
        "_antivenom_unlearnable_examples_toolbox",
        toolbox_path,
    )
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Could not load official UE toolbox: {toolbox_path}")
    toolbox_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(toolbox_module)
    PerturbationTool = toolbox_module.PerturbationTool

    if not clean_records:
        raise ValueError("Cannot generate unlearnable examples without clean training records.")
    if len(resize) != 2 or any(int(value) <= 0 for value in resize):
        raise ValueError(f"UE resize must contain two positive dimensions, got {resize!r}.")
    if epsilon <= 0 or num_steps <= 0 or step_size <= 0:
        raise ValueError("UE epsilon, steps, and step size must be positive.")
    if warmup_epochs < 0 or outer_iterations <= 0 or surrogate_steps_per_outer < 0:
        raise ValueError(
            "UE warm-up epochs and surrogate steps must be non-negative, and outer "
            "iterations must be positive."
        )
    if not 0.0 <= stop_error <= 1.0:
        raise ValueError("UE stop error must be in [0, 1].")

    set_all_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.Resize(tuple(resize)), transforms.ToTensor()])

    class CleanRecordDataset(Dataset):
        def __init__(self, records: Sequence[Dict[str, Any]]) -> None:
            self.records = list(records)

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, idx: int) -> Tuple[Any, int, int]:
            record = self.records[idx]
            image = Image.open(record["clean_path"]).convert("RGB")
            return transform(image), int(record["label"]), idx

    loader = DataLoader(
        CleanRecordDataset(clean_records),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    from torchvision.models import resnet18

    model = resnet18(weights=None, num_classes=num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / UE_MANIFEST_NAME
    manifest_path.unlink(missing_ok=True)
    image_height, image_width = (int(resize[0]), int(resize[1]))
    delta_path = output_root / "perturbations.npy"
    delta_bank = np.lib.format.open_memmap(
        delta_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(clean_records), 3, image_height, image_width),
    )
    delta_bank[:] = 0.0
    delta_bank.flush()

    for warmup_epoch in range(warmup_epochs):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for images, labels, _ in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * labels.numel()
            total_examples += labels.numel()
        print(
            f"UE warmup={warmup_epoch + 1}/{warmup_epochs}, "
            f"loss={total_loss / max(total_examples, 1):.6f}",
            flush=True,
        )

    tool = PerturbationTool(
        seed=seed,
        epsilon=epsilon / 255.0,
        num_steps=num_steps,
        step_size=step_size / 255.0,
    )
    surrogate_iterator = iter(loader)
    completed_outer_iterations = 0
    poisoned_train_accuracy = 0.0

    for outer in range(outer_iterations):
        model.train()
        for parameter in model.parameters():
            parameter.requires_grad_(True)

        for _ in range(surrogate_steps_per_outer):
            try:
                images, labels, indices = next(surrogate_iterator)
            except StopIteration:
                surrogate_iterator = iter(loader)
                images, labels, indices = next(surrogate_iterator)

            index_array = indices.numpy()
            current_delta = torch.from_numpy(
                np.array(delta_bank[index_array], copy=True)
            )
            poisoned_images = (images + current_delta).clamp(0.0, 1.0)
            poisoned_images = poisoned_images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(poisoned_images), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        for images, labels, indices in loader:
            index_array = indices.numpy()
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            current_delta = torch.from_numpy(
                np.array(delta_bank[index_array], copy=True)
            ).to(device, non_blocking=True)
            _, updated_delta = tool.min_min_attack(
                images,
                labels,
                model,
                optimizer,
                criterion,
                random_noise=current_delta,
            )
            delta_bank[index_array] = (
                updated_delta.detach().cpu().numpy().astype(np.float32, copy=False)
            )
        delta_bank.flush()

        total = 0
        correct = 0
        model.eval()
        with torch.no_grad():
            for images, labels, indices in loader:
                index_array = indices.numpy()
                delta = torch.from_numpy(np.array(delta_bank[index_array], copy=True))
                poisoned_images = (images + delta).clamp(0.0, 1.0)
                poisoned_images = poisoned_images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                predictions = model(poisoned_images).argmax(dim=1)
                total += labels.numel()
                correct += int((predictions == labels).sum())

        completed_outer_iterations = outer + 1
        poisoned_train_accuracy = correct / max(total, 1)
        poisoned_train_error = 1.0 - poisoned_train_accuracy
        print(
            f"UE outer={completed_outer_iterations}/{outer_iterations}, "
            f"poisoned_train_accuracy={poisoned_train_accuracy:.4f}, "
            f"error={poisoned_train_error:.4f}",
            flush=True,
        )

        checkpoint_path = output_root / "generator_last.pt"
        temporary_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "outer_iteration": completed_outer_iterations,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "delta_path": str(delta_path),
                "poisoned_train_accuracy": poisoned_train_accuracy,
                "epsilon": epsilon,
                "num_steps": num_steps,
                "step_size": step_size,
                "seed": seed,
                "resize": list(resize),
            },
            temporary_checkpoint,
        )
        os.replace(temporary_checkpoint, checkpoint_path)

        if poisoned_train_error <= stop_error:
            break

    for images, _, indices in loader:
        index_array = indices.numpy()
        delta = torch.from_numpy(np.array(delta_bank[index_array], copy=True))
        poisoned_images = (images + delta).clamp(0.0, 1.0)
        for tensor, record_idx in zip(poisoned_images, indices):
            record = clean_records[int(record_idx)]
            relative_path = Path(record["relative_path"]).with_suffix(".png")
            out_path = output_root / record["client_id"] / relative_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            _pil_from_tensor(tensor).save(out_path, format="PNG")

    manifest = _unlearnable_manifest(
        epsilon=epsilon,
        steps=num_steps,
        step_size=step_size,
        warmup_epochs=warmup_epochs,
        outer_iterations=outer_iterations,
        surrogate_steps_per_outer=surrogate_steps_per_outer,
        stop_error=stop_error,
        seed=seed,
        resize=resize,
        unlearnable_repo=unlearnable_repo,
    )
    manifest.update(
        {
            "completed_outer_iterations": completed_outer_iterations,
            "poisoned_train_accuracy": poisoned_train_accuracy,
            "delta_path": str(delta_path),
        }
    )
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_manifest, manifest_path)
    return manifest


def _copy_clean_image(record: Dict[str, Any], output_root: Path) -> Path:
    out_path = output_root / record["client_id"] / record["relative_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(record["clean_path"], out_path)
    return out_path


def _class_name_for_label(class_names: Sequence[str], label: int) -> str:
    return class_names[label] if 0 <= label < len(class_names) else str(label)


def _make_label_flip_rows(
    clean_records: Sequence[Dict[str, Any]],
    *,
    method: str,
    output_root: Path,
    class_names: Sequence[str],
    seed: int,
    random_flip_fraction: float,
    target_label: int,
    replacement_label: int,
) -> List[Dict[str, Any]]:
    labels = sorted({int(record["label"]) for record in clean_records})
    if len(labels) < 2:
        raise ValueError("Label flipping requires at least two classes.")

    rng = random.Random(seed)
    selected_random_indices = set()
    if method == POISONING_METHOD_RANDOM_LABEL_FLIPPING:
        if not 0.0 <= random_flip_fraction <= 1.0:
            raise ValueError(f"random_flip_fraction must be in [0, 1], got {random_flip_fraction}")
        flip_count = int(len(clean_records) * random_flip_fraction)
        selected_random_indices = set(rng.sample(range(len(clean_records)), flip_count))

    rows: List[Dict[str, Any]] = []
    for idx, record in enumerate(clean_records):
        original_label = int(record["label"])
        new_label = original_label
        label_changed = False

        if method == POISONING_METHOD_RANDOM_LABEL_FLIPPING and idx in selected_random_indices:
            choices = [label for label in labels if label != original_label]
            new_label = rng.choice(choices)
            label_changed = True
        elif method == POISONING_METHOD_TARGET_LABEL_FLIPPING and original_label == target_label:
            if replacement_label == target_label:
                raise ValueError("target_label and replacement_label must differ for target label flipping.")
            new_label = replacement_label
            label_changed = True

        image_path = _copy_clean_image(record, output_root)
        poisoned = dict(record)
        poisoned.update(
            {
                "image_path": str(image_path),
                "label": new_label,
                "class_name": _class_name_for_label(class_names, new_label),
                "is_poisoned": label_changed,
                "poisoning_method": method,
                "original_label": original_label,
                "original_class_name": record["class_name"],
                "label_changed": label_changed,
                "label_flip_fraction": random_flip_fraction if method == POISONING_METHOD_RANDOM_LABEL_FLIPPING else "",
                "target_label": target_label if method == POISONING_METHOD_TARGET_LABEL_FLIPPING else "",
                "replacement_label": replacement_label if method == POISONING_METHOD_TARGET_LABEL_FLIPPING else "",
            }
        )
        rows.append(poisoned)
    return rows


def _l2norm_limit_from_linf(linf: float, feature_dim: int) -> float:
    # Same projection scale used by Availability-Attacks-Create-Shortcuts.
    return float((linf**2 * feature_dim) ** 0.5)


def _normalize_l2norm(data: Any, norm_limit: float) -> Any:
    import numpy as np

    flat = data.reshape(data.shape[0], -1)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return (flat / norms * norm_limit).reshape(data.shape)


def _make_availability_shortcut_rows(
    clean_records: Sequence[Dict[str, Any]],
    *,
    output_root: Path,
    class_names: Sequence[str],
    seed: int,
    resize: Sequence[int],
    eps: float,
    patch_size: int,
) -> List[Dict[str, Any]]:
    del eps  # The visible validation attack overwrites a patch and is not epsilon-bounded.
    if not clean_records:
        return []
    if len(resize) != 2 or any(int(value) <= 0 for value in resize):
        raise ValueError(f"Shortcut resize must contain two positive dimensions, got {resize!r}.")
    if patch_size <= 0:
        raise ValueError(f"shortcut patch size must be positive, got {patch_size}.")

    labels = sorted({int(record["label"]) for record in clean_records})
    num_classes = len(labels)
    if labels != list(range(num_classes)):
        raise ValueError(
            "Availability-shortcut labels must be contiguous from zero; "
            f"got {labels}."
        )

    colors = [
        tuple(
            int(round(channel * 255))
            for channel in colorsys.hsv_to_rgb(class_index / num_classes, 1.0, 1.0)
        )
        for class_index in range(num_classes)
    ]
    height, width = (int(resize[0]), int(resize[1]))
    materialized_patch_size = min(patch_size, height, width)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "generator": AVAILABILITY_SHORTCUT_GENERATOR,
        "application": "materialized_before_training_augmentation",
        "operation": "overwrite_top_left_patch_jpeg",
        "image_format": "JPEG",
        "jpeg_quality": AVAILABILITY_SHORTCUT_JPEG_QUALITY,
        "jpeg_subsampling": AVAILABILITY_SHORTCUT_JPEG_SUBSAMPLING,
        "seed": int(seed),
        "patch_size": materialized_patch_size,
        "num_classes": num_classes,
        "class_names": list(class_names),
        "requested_resize": [int(value) for value in resize],
    }
    (output_root / AVAILABILITY_SHORTCUT_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    rows: List[Dict[str, Any]] = []
    for record in clean_records:
        label = int(record["label"])
        image = Image.open(record["clean_path"]).convert("RGB").resize((width, height))
        image.paste(
            colors[label],
            (0, 0, materialized_patch_size, materialized_patch_size),
        )
        relative_path = Path(record["relative_path"]).with_suffix(".jpeg")
        out_path = output_root / str(record["client_id"]) / relative_path
        _save_jpeg(
            image,
            out_path,
            quality=AVAILABILITY_SHORTCUT_JPEG_QUALITY,
            subsampling=AVAILABILITY_SHORTCUT_JPEG_SUBSAMPLING,
        )
        poisoned = dict(record)
        poisoned.update(
            {
                "image_path": str(out_path),
                "relative_path": str(relative_path),
                "label": label,
                "class_name": _class_name_for_label(class_names, label),
                "is_poisoned": True,
                "poisoning_method": POISONING_METHOD_AVAILABILITY_SHORTCUTS,
                "original_label": label,
                "original_class_name": record["class_name"],
                "label_changed": False,
                "shortcut_eps": "",
                "shortcut_patch_size": materialized_patch_size,
                "shortcut_frequency": "",
                "shortcut_seed": seed,
                "shortcut_num_classes": num_classes,
                "shortcut_generator": AVAILABILITY_SHORTCUT_GENERATOR,
                "shortcut_operation": "overwrite_top_left_patch_jpeg",
                "shortcut_class_sep": "",
                "shortcut_jpeg_quality": AVAILABILITY_SHORTCUT_JPEG_QUALITY,
                "shortcut_jpeg_subsampling": AVAILABILITY_SHORTCUT_JPEG_SUBSAMPLING,
            }
        )
        rows.append(poisoned)
    return rows


# Disabled legacy implementation retained for comparison. Dataset preparation now
# materializes the class-color patch as JPEG above.
def _make_availability_shortcut_rows_sklearn_disabled(
    clean_records: Sequence[Dict[str, Any]],
    *,
    output_root: Path,
    class_names: Sequence[str],
    seed: int,
    resize: Sequence[int],
    eps: float,
    patch_size: int,
) -> List[Dict[str, Any]]:
    import numpy as np
    try:
        from sklearn.datasets import make_classification
    except ImportError as exc:
        raise RuntimeError(
            "availability_shortcuts requires scikit-learn. "
            "Install it with: pip install scikit-learn"
        ) from exc

    if patch_size <= 0:
        raise ValueError(f"patch_size must be > 0, got {patch_size}")
    height = int(resize[0])
    width = int(resize[1]) if len(resize) > 1 else height
    if not clean_records:
        return []

    labels = sorted({int(record["label"]) for record in clean_records})
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    num_classes = len(labels)
    # The official attack defines patch_size on a 32x32 image. Preserve its
    # feature grid when training images are resized to 224x224; using 8px
    # patches directly at 224px would change 48 synthetic features into 2352.
    reference_size = AVAILABILITY_SHORTCUT_REFERENCE_IMAGE_SIZE
    grid_h = (reference_size + patch_size - 1) // patch_size
    grid_w = (reference_size + patch_size - 1) // patch_size
    n_random_features = grid_h * grid_w * 3

    # The official implementation generates sample-specific synthetic features
    # with sklearn.make_classification and then assigns same-label features to
    # each training image. TrashNet is imbalanced, so generate a balanced pool
    # large enough to cover the largest real class and retry if rounding leaves
    # any class short.
    class_counts = {
        label: sum(int(record["label"]) == label for record in clean_records)
        for label in labels
    }
    generated_count = max(len(clean_records), num_classes * max(class_counts.values()))
    generated_data = None
    generated_labels = None
    for attempt in range(6):
        candidate_data, candidate_labels = make_classification(
            n_samples=generated_count,
            n_features=n_random_features,
            n_classes=num_classes,
            n_informative=n_random_features,
            n_redundant=0,
            n_repeated=0,
            class_sep=AVAILABILITY_SHORTCUT_CLASS_SEP,
            flip_y=0.0,
            n_clusters_per_class=1,
            random_state=seed + attempt,
        )
        candidate_counts = np.bincount(candidate_labels, minlength=num_classes)
        if all(
            int(candidate_counts[label_to_index[label]]) >= required
            for label, required in class_counts.items()
        ):
            generated_data = candidate_data.astype(np.float32, copy=False)
            generated_labels = candidate_labels
            break
        generated_count *= 2
    if generated_data is None or generated_labels is None:
        raise RuntimeError(
            "Could not generate enough class-matched availability shortcut samples "
            f"for class counts {class_counts}."
        )

    generated_data = generated_data.reshape(generated_count, grid_h, grid_w, 3)
    output_patch_h = (height + grid_h - 1) // grid_h
    output_patch_w = (width + grid_w - 1) // grid_w
    linf = float(eps) / 255.0
    l2_limit = _l2norm_limit_from_linf(linf, height * width * 3)

    class_shortcut_features = {
        label: generated_data[generated_labels == label_to_index[label]][: class_counts[label]]
        for label in labels
    }
    class_offsets = {label: 0 for label in labels}

    rows: List[Dict[str, Any]] = []
    for record in clean_records:
        label = int(record["label"])
        shortcut_features = class_shortcut_features[label][class_offsets[label]]
        class_offsets[label] += 1
        shortcut = np.repeat(shortcut_features, output_patch_w, axis=1)
        shortcut = np.repeat(shortcut, output_patch_h, axis=0)[:height, :width, :]
        shortcut = _normalize_l2norm(shortcut[None, ...], l2_limit)[0]

        image = Image.open(record["clean_path"]).convert("RGB").resize((width, height))
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        poisoned_array = np.clip(image_array + shortcut, 0.0, 1.0)
        poisoned_image = Image.fromarray((poisoned_array * 255.0).round().astype(np.uint8), mode="RGB")

        out_path = output_root / record["client_id"] / record["relative_path"]
        _save_jpeg(
            poisoned_image,
            out_path,
            quality=AVAILABILITY_SHORTCUT_JPEG_QUALITY,
            subsampling=AVAILABILITY_SHORTCUT_JPEG_SUBSAMPLING,
        )
        poisoned = dict(record)
        poisoned.update(
            {
                "image_path": str(out_path),
                "label": label,
                "class_name": _class_name_for_label(class_names, label),
                "is_poisoned": True,
                "poisoning_method": POISONING_METHOD_AVAILABILITY_SHORTCUTS,
                "original_label": label,
                "original_class_name": record["class_name"],
                "label_changed": False,
                "shortcut_eps": eps,
                "shortcut_patch_size": patch_size,
                "shortcut_generator": "sklearn_make_classification_class_sep_10_jpeg_q100_v2",
                "shortcut_class_sep": AVAILABILITY_SHORTCUT_CLASS_SEP,
                "shortcut_jpeg_quality": AVAILABILITY_SHORTCUT_JPEG_QUALITY,
                "shortcut_jpeg_subsampling": AVAILABILITY_SHORTCUT_JPEG_SUBSAMPLING,
            }
        )
        rows.append(poisoned)
    return rows


def _append_unlearnable_example_rows(
    metadata_rows: List[Dict[str, Any]],
    clean_records: Sequence[Dict[str, Any]],
    *,
    root: Path,
    num_classes: int,
    resize: Sequence[int],
    seed: int,
    poison_epsilon: float,
    poison_steps: int,
    poison_step_size: float,
    batch_size: int,
    unlearnable_repo: str,
    poison_warmup_epochs: int,
    poison_outer_iterations: int,
    poison_surrogate_steps_per_outer: int,
    poison_stop_error: float,
) -> None:
    _build_unlearnable_example_images(
        clean_records,
        output_root=root / "poisoned" / POISONING_METHOD_UNLEARNABLE_EXAMPLES,
        num_classes=num_classes,
        resize=resize,
        seed=seed,
        epsilon=poison_epsilon,
        num_steps=poison_steps,
        step_size=poison_step_size,
        batch_size=batch_size,
        unlearnable_repo=unlearnable_repo,
        warmup_epochs=poison_warmup_epochs,
        outer_iterations=poison_outer_iterations,
        surrogate_steps_per_outer=poison_surrogate_steps_per_outer,
        stop_error=poison_stop_error,
    )

    for record in clean_records:
        ue_relative_path = Path(record["relative_path"]).with_suffix(".png")
        poisoned_path = (
            root
            / "poisoned"
            / POISONING_METHOD_UNLEARNABLE_EXAMPLES
            / record["client_id"]
            / ue_relative_path
        )
        poisoned = dict(record)
        poisoned.update(
            {
                "image_path": str(poisoned_path),
                "relative_path": str(ue_relative_path),
                "is_poisoned": True,
                "poisoning_method": POISONING_METHOD_UNLEARNABLE_EXAMPLES,
                "ue_epsilon": poison_epsilon,
                "ue_steps": poison_steps,
                "ue_step_size": poison_step_size,
                "ue_warmup_epochs": poison_warmup_epochs,
                "ue_outer_iterations": poison_outer_iterations,
                "ue_surrogate_steps_per_outer": poison_surrogate_steps_per_outer,
                "ue_stop_error": poison_stop_error,
                "ue_surrogate": "resnet18",
                "ue_generation_version": UE_GENERATION_VERSION,
            }
        )
        metadata_rows.append(poisoned)


def _write_badsampler_plans(
    *,
    root: Path,
    clean_records: Sequence[Dict[str, Any]],
    kappa: float,
) -> Path:
    attack_root = root / "poisoned" / POISONING_METHOD_BADSAMPLING
    by_client: Dict[str, List[Dict[str, Any]]] = {}
    for record in clean_records:
        by_client.setdefault(str(record["client_id"]), []).append(record)

    for client_id, records in sorted(by_client.items()):
        ordered = sorted(
            records,
            key=lambda record: (
                int(record.get("source_index", 0)),
                str(record.get("relative_path", "")),
            ),
        )
        candidates = [
            {
                "position": position,
                "source_index": int(record["source_index"]),
                "relative_path": str(record["relative_path"]),
                "label": int(record["label"]),
            }
            for position, record in enumerate(ordered)
        ]
        plan = {
            "attack": POISONING_METHOD_BADSAMPLING,
            "version": 1,
            "client_id": client_id,
            "dataset_split": "train",
            "source_poisoning_method": POISONING_METHOD_CLEAN,
            "images_or_labels_modified": False,
            "ranking": "computed_from_surrogate_loss_at_runtime",
            "selection": {
                "strategy": "top_k_loss",
                "kappa": float(kappa),
                "pool_size": "min(num_candidates, kappa * batch_size)",
                "replacement": True,
                "num_samples": "num_candidates",
            },
            "num_candidates": len(candidates),
            "candidates": candidates,
        }
        client_dir = attack_root / client_id
        client_dir.mkdir(parents=True, exist_ok=True)
        (client_dir / BADSAMPLER_PLAN_NAME).write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n"
        )
    return attack_root


def _append_requested_poisoning_rows(
    metadata_rows: List[Dict[str, Any]],
    clean_records: Sequence[Dict[str, Any]],
    *,
    requested_scenarios: Sequence[str],
    root: Path,
    class_names: Sequence[str],
    seed: int,
    resize: Sequence[int],
    poison_epsilon: float,
    poison_steps: int,
    poison_step_size: float,
    batch_size: int,
    unlearnable_repo: str,
    poison_warmup_epochs: int,
    poison_outer_iterations: int,
    poison_surrogate_steps_per_outer: int,
    poison_stop_error: float,
    random_label_flip_fraction: float,
    target_label: int,
    replacement_label: int,
    shortcut_eps: float,
    shortcut_patch_size: int,
) -> None:
    if POISONING_METHOD_BADSAMPLING in requested_scenarios:
        _write_badsampler_plans(
            root=root,
            clean_records=clean_records,
            kappa=DEFAULT_BADSAMPLER_KAPPA,
        )

    if POISONING_METHOD_UNLEARNABLE_EXAMPLES in requested_scenarios:
        _append_unlearnable_example_rows(
            metadata_rows,
            clean_records,
            root=root,
            num_classes=len({int(row["label"]) for row in clean_records}),
            resize=resize,
            seed=seed,
            poison_epsilon=poison_epsilon,
            poison_steps=poison_steps,
            poison_step_size=poison_step_size,
            batch_size=batch_size,
            unlearnable_repo=unlearnable_repo,
            poison_warmup_epochs=poison_warmup_epochs,
            poison_outer_iterations=poison_outer_iterations,
            poison_surrogate_steps_per_outer=poison_surrogate_steps_per_outer,
            poison_stop_error=poison_stop_error,
        )

    if POISONING_METHOD_RANDOM_LABEL_FLIPPING in requested_scenarios:
        metadata_rows.extend(
            _make_label_flip_rows(
                clean_records,
                method=POISONING_METHOD_RANDOM_LABEL_FLIPPING,
                output_root=root / "poisoned" / POISONING_METHOD_RANDOM_LABEL_FLIPPING,
                class_names=class_names,
                seed=seed + 17,
                random_flip_fraction=random_label_flip_fraction,
                target_label=target_label,
                replacement_label=replacement_label,
            )
        )

    if POISONING_METHOD_TARGET_LABEL_FLIPPING in requested_scenarios:
        metadata_rows.extend(
            _make_label_flip_rows(
                clean_records,
                method=POISONING_METHOD_TARGET_LABEL_FLIPPING,
                output_root=root / "poisoned" / POISONING_METHOD_TARGET_LABEL_FLIPPING,
                class_names=class_names,
                seed=seed + 31,
                random_flip_fraction=random_label_flip_fraction,
                target_label=target_label,
                replacement_label=replacement_label,
            )
        )

    if POISONING_METHOD_AVAILABILITY_SHORTCUTS in requested_scenarios:
        metadata_rows.extend(
            _make_availability_shortcut_rows(
                clean_records,
                output_root=root / "poisoned" / POISONING_METHOD_AVAILABILITY_SHORTCUTS,
                class_names=class_names,
                seed=seed,
                resize=resize,
                eps=shortcut_eps,
                patch_size=shortcut_patch_size,
            )
        )


def prepare_dataset(
    *,
    data_dir: str = DEFAULT_DATA_DIR,
    dataset_name: str = DATASET_NAME,
    num_clients: int = DEFAULT_NUM_CLIENTS,
    seed: int = 0,
    force: bool = False,
    resize: Sequence[int] = (64, 64),
    poison_epsilon: float = 16.0,
    poison_steps: int = 20,
    poison_step_size: float = 1.6,
    poison_warmup_epochs: int = 10,
    poison_outer_iterations: int = 10,
    poison_surrogate_steps_per_outer: int = 100,
    poison_stop_error: float = 0.01,
    batch_size: int = DEFAULT_BATCH_SIZE,
    unlearnable_repo: str = DEFAULT_UNLEARNABLE_REPO,
    random_label_flip_fraction: float = DEFAULT_RANDOM_LABEL_FLIP_FRACTION,
    target_label: int = DEFAULT_TARGET_LABEL_FLIP_TARGET_LABEL,
    replacement_label: int = DEFAULT_TARGET_LABEL_FLIP_REPLACEMENT_LABEL,
    shortcut_eps: float = DEFAULT_AVAILABILITY_SHORTCUT_EPS,
    shortcut_patch_size: int = DEFAULT_AVAILABILITY_SHORTCUT_PATCH_SIZE,
    prepare_scenarios: Optional[Any] = PREPARE_SCENARIO_ALL,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    test_seed: int = DEFAULT_TEST_SEED,
    partition_method: str = DEFAULT_PARTITION_METHOD,
    noniid_alpha: float = DEFAULT_NONIID_ALPHA,
) -> Path:
    os.environ["DATASET_NAME"] = dataset_name
    partition_method = _canonical_partition_method(partition_method)
    root = _dataset_root(data_dir, dataset_name)
    requested_scenarios = _parse_prepare_scenarios(prepare_scenarios)
    expected_ue_manifest = _unlearnable_manifest(
        epsilon=poison_epsilon,
        steps=poison_steps,
        step_size=poison_step_size,
        warmup_epochs=poison_warmup_epochs,
        outer_iterations=poison_outer_iterations,
        surrogate_steps_per_outer=poison_surrogate_steps_per_outer,
        stop_error=poison_stop_error,
        seed=seed,
        resize=resize,
        unlearnable_repo=unlearnable_repo,
    )
    existing_rows = _read_metadata_rows(root)
    if existing_rows and _metadata_partition_method(existing_rows) != partition_method:
        if not force or POISONING_METHOD_CLEAN not in requested_scenarios:
            raise ValueError(
                f"Prepared data in {root} uses partition_method="
                f"{_metadata_partition_method(existing_rows)!r}, but {partition_method!r} "
                "was requested. Use the correct data directory or regenerate clean data "
                "with --force."
            )
    metadata_changed = bool(existing_rows) and _fill_partition_metadata(
        existing_rows,
        partition_method=partition_method,
        noniid_alpha=noniid_alpha,
    )
    if existing_rows and _normalize_metadata_paths(
        existing_rows,
        data_dir=data_dir,
        dataset_name=dataset_name,
    ):
        metadata_changed = True
    if metadata_changed:
        _write_metadata_rows(root, existing_rows)
    if existing_rows and not any(
        row.get("poisoning_method") == POISONING_METHOD_CLEAN
        and row.get("dataset_split") == "test"
        for row in existing_rows
    ):
        existing_rows, created_test_samples = _add_clean_test_split(
            existing_rows,
            test_fraction=test_fraction,
            test_seed=test_seed,
        )
        _write_metadata_rows(root, existing_rows)
        (root / PREPARED_MARKER).write_text(
            json.dumps(
                {
                    "dataset": dataset_name,
                    "num_clients": num_clients,
                    "seed": seed,
                    "test_fraction": test_fraction,
                    "test_seed": test_seed,
                    "partition_method": partition_method,
                    "noniid_alpha": noniid_alpha if partition_method != "iid" else "",
                }
            )
        )
        print(
            f"Created clean-only test split with {created_test_samples} samples "
            f"in {root / METADATA_NAME}"
        )
    if (
        prepared_data_exists(
            data_dir,
            num_clients,
            dataset_name,
            partition_method,
            expected_ue_manifest,
        )
        and not force
        and set(requested_scenarios) == set(PREPARE_SCENARIOS)
    ):
        return _finalize_prepared_dataset(root)

    existing_rows = _read_metadata_rows(root)
    clean_records_from_existing = _clean_records_from_metadata(
        existing_rows,
        data_dir=data_dir,
        dataset_name=dataset_name,
    )
    if clean_records_from_existing and _mode_complete(root, "clean", num_clients) and (
        not force or POISONING_METHOD_CLEAN not in requested_scenarios
    ):
        missing_or_requested = [
            scenario
            for scenario in requested_scenarios
            if scenario != POISONING_METHOD_CLEAN
            and (
                force
                or not _scenario_complete(
                    root,
                    scenario,
                    existing_rows,
                    num_clients,
                    expected_ue_manifest=expected_ue_manifest,
                )
            )
        ]
        if not missing_or_requested:
            return _finalize_prepared_dataset(root)
        metadata_rows = [row for row in existing_rows if row.get("poisoning_method") not in set(missing_or_requested)]
        class_names = _class_names_from_records(clean_records_from_existing)
        training_clean_records = [
            record for record in clean_records_from_existing
            if record.get("dataset_split") == "train"
        ]
        _append_requested_poisoning_rows(
            metadata_rows,
            training_clean_records,
            requested_scenarios=missing_or_requested,
            root=root,
            class_names=class_names,
            seed=seed,
            resize=resize,
            poison_epsilon=poison_epsilon,
            poison_steps=poison_steps,
            poison_step_size=poison_step_size,
            batch_size=batch_size,
            unlearnable_repo=unlearnable_repo,
            poison_warmup_epochs=poison_warmup_epochs,
            poison_outer_iterations=poison_outer_iterations,
            poison_surrogate_steps_per_outer=poison_surrogate_steps_per_outer,
            poison_stop_error=poison_stop_error,
            random_label_flip_fraction=random_label_flip_fraction,
            target_label=target_label,
            replacement_label=replacement_label,
            shortcut_eps=shortcut_eps,
            shortcut_patch_size=shortcut_patch_size,
        )
        _write_metadata_rows(root, metadata_rows)
        (root / PREPARED_MARKER).write_text(
            json.dumps(
                {
                    "dataset": dataset_name,
                    "num_clients": num_clients,
                    "seed": seed,
                    "test_fraction": test_fraction,
                    "test_seed": test_seed,
                    "partition_method": partition_method,
                    "noniid_alpha": noniid_alpha if partition_method != "iid" else "",
                }
            )
        )
        return _finalize_prepared_dataset(root)

    if POISONING_METHOD_CLEAN not in requested_scenarios:
        raise FileNotFoundError(
            "Clean prepared data is required before generating poisoning-only scenarios. "
            f"Run with --prepare-scenarios {POISONING_METHOD_CLEAN} or --prepare-scenarios {PREPARE_SCENARIO_ALL} first."
        )

    from datasets import load_dataset

    root.mkdir(parents=True, exist_ok=True)
    dataset_dict = load_dataset(dataset_name)
    split_names = {split for split, _ in _iter_splits(dataset_dict)}
    has_explicit_test_split = "test" in split_names
    metadata_rows: List[Dict[str, Any]] = []
    clean_records: List[Dict[str, Any]] = []
    class_names: List[str] = []

    for split, raw_ds in _iter_splits(dataset_dict):
        ds, source_indices = _maybe_match_reference_split_size(
            raw_ds,
            dataset_name=dataset_name,
            split=split,
            data_dir=data_dir,
            seed=seed,
            has_explicit_test_split=has_explicit_test_split,
        )
        if len(ds) == 0:
            continue
        names = _class_names(ds)
        if not class_names:
            class_names = names
        assignments = _assign_partitions(
            ds,
            num_clients=num_clients,
            seed=seed,
            partition_method=partition_method,
            noniid_alpha=noniid_alpha,
        )
        for idx, example in enumerate(ds):
            source_index = source_indices[idx]
            image, label = _extract_image_label(example)
            class_name = names[label] if label < len(names) else str(label)
            client_id = assignments[idx]
            relative_path = Path(_safe_class_name(class_name)) / f"{split}_{source_index:06d}.jpeg"
            clean_path = root / "clean" / client_id / relative_path
            _save_jpeg(image, clean_path)
            record = {
                "source_index": source_index,
                "image_path": str(clean_path),
                "clean_path": str(clean_path),
                "relative_path": str(relative_path),
                "label": label,
                "class_name": class_name,
                "original_label": label,
                "original_class_name": class_name,
                "label_changed": False,
                "label_flip_fraction": "",
                "target_label": "",
                "replacement_label": "",
                "shortcut_eps": "",
                "shortcut_patch_size": "",
                "shortcut_frequency": "",
                "shortcut_seed": "",
                "shortcut_num_classes": "",
                "shortcut_generator": "",
                "shortcut_operation": "",
                "shortcut_class_sep": "",
                "shortcut_jpeg_quality": "",
                "shortcut_jpeg_subsampling": "",
                "client_id": client_id,
                "partition_id": client_id,
                "partition_method": partition_method,
                "noniid_alpha": noniid_alpha if partition_method != "iid" else "",
                "dataset_split": split,
                "is_poisoned": False,
                "poisoning_method": POISONING_METHOD_CLEAN,
            }
            clean_records.append(record)
            metadata_rows.append(record)

    metadata_rows, _ = _add_clean_test_split(
        metadata_rows,
        test_fraction=test_fraction,
        test_seed=test_seed,
    )
    clean_records = _clean_records_from_metadata(
        metadata_rows,
        data_dir=data_dir,
        dataset_name=dataset_name,
    )
    training_clean_records = [
        record for record in clean_records
        if record.get("dataset_split") == "train"
    ]

    _append_requested_poisoning_rows(
        metadata_rows,
        training_clean_records,
        requested_scenarios=[scenario for scenario in requested_scenarios if scenario != POISONING_METHOD_CLEAN],
        root=root,
        class_names=class_names,
        seed=seed,
        resize=resize,
        poison_epsilon=poison_epsilon,
        poison_steps=poison_steps,
        poison_step_size=poison_step_size,
        batch_size=batch_size,
        unlearnable_repo=unlearnable_repo,
        poison_warmup_epochs=poison_warmup_epochs,
        poison_outer_iterations=poison_outer_iterations,
        poison_surrogate_steps_per_outer=poison_surrogate_steps_per_outer,
        poison_stop_error=poison_stop_error,
        random_label_flip_fraction=random_label_flip_fraction,
        target_label=target_label,
        replacement_label=replacement_label,
        shortcut_eps=shortcut_eps,
        shortcut_patch_size=shortcut_patch_size,
    )

    _write_metadata_rows(root, metadata_rows)
    (root / PREPARED_MARKER).write_text(
        json.dumps(
            {
                "dataset": dataset_name,
                "num_clients": num_clients,
                "seed": seed,
                "test_fraction": test_fraction,
                "test_seed": test_seed,
                "partition_method": partition_method,
                "noniid_alpha": noniid_alpha if partition_method != "iid" else "",
            }
        )
    )
    return _finalize_prepared_dataset(root)


def _materialized_augmentation_supported(dataset_name: str) -> bool:
    return dataset_slug(dataset_name) in MATERIALIZED_AUGMENTATION_DATASET_SLUGS


def _augmentation_profile_for_dataset(
    profile_name: str,
    dataset_name: str,
) -> Dict[str, Any]:
    profile = dict(AUGMENTATION_PROFILES[profile_name])
    if dataset_slug(dataset_name) == "cifar10":
        profile["resize"] = [32, 32]
    return profile


def write_augmentation_profile_manifests(
    root: Path,
    dataset_name: Optional[str] = None,
) -> Path:
    effective_dataset_name = dataset_name or root.name
    profile_root = root / AUGMENTATION_PROFILE_DIR
    profile_root.mkdir(parents=True, exist_ok=True)
    for profile_name in AUGMENTATION_VARIANT_NAMES:
        profile_dir = profile_root / profile_name
        profile_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "profile": profile_name,
            "application": "materialized_once",
            "source": "clean_train",
            "test_augmentation": False,
            "transform": _augmentation_profile_for_dataset(
                profile_name, effective_dataset_name
            ),
        }
        (profile_dir / "augmentation.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
    return profile_root


def _augmented_clean_path(root: Path, profile_name: str, record: Dict[str, Any]) -> Path:
    return (
        root
        / AUGMENTATION_PROFILE_DIR
        / profile_name
        / record["client_id"]
        / record["relative_path"]
    )


def prepare_augmentation_variants(
    *,
    data_dir: str,
    dataset_name: Optional[str] = None,
    profiles: Sequence[str] = AUGMENTATION_VARIANT_NAMES,
    seed: int = DEFAULT_AUGMENTATION_SEED,
    force: bool = False,
) -> Path:
    root = _dataset_root(data_dir, dataset_name)
    effective_dataset_name = dataset_name or DATASET_NAME
    rows = _read_metadata_rows(root)
    clean_training_rows = [
        dict(row)
        for row in rows
        if row.get("poisoning_method") == POISONING_METHOD_CLEAN
        and row.get("dataset_split") == "train"
    ]
    if not clean_training_rows:
        raise FileNotFoundError(
            f"No clean training records found in {root / METADATA_NAME}"
        )

    torch, _, _, _ = _require_torch()
    for profile_name in profiles:
        if profile_name not in AUGMENTATION_VARIANT_NAMES:
            raise ValueError(
                f"Cannot materialize augmentation profile {profile_name!r}; "
                f"choose from {', '.join(AUGMENTATION_VARIANT_NAMES)}"
            )
        profile_dir = root / AUGMENTATION_PROFILE_DIR / profile_name
        marker_path = profile_dir / PREPARED_MARKER
        marker = {
            "dataset": dataset_name or DATASET_NAME,
            "profile": profile_name,
            "seed": seed,
            "source": "clean_train",
            "num_images": len(clean_training_rows),
            "transform": _augmentation_profile_for_dataset(
                profile_name, effective_dataset_name
            ),
        }
        marker_matches = False
        if marker_path.exists() and not force:
            try:
                marker_matches = json.loads(marker_path.read_text()) == marker
            except (json.JSONDecodeError, OSError):
                marker_matches = False
            if marker_matches and all(
                _augmented_clean_path(root, profile_name, record).exists()
                for record in clean_training_rows
            ):
                continue

        materialize_augment = _augmentation_profile_for_dataset(
            profile_name, effective_dataset_name
        )
        materialize_augment["normalize"] = False
        transform = build_transform(materialize_augment)
        for record in clean_training_rows:
            output_path = _augmented_clean_path(root, profile_name, record)
            if output_path.exists() and not force and not marker_path.exists():
                continue
            source_path = _resolve_metadata_image_path(
                record["image_path"], data_dir, dataset_name
            )
            profile_offset = AUGMENTATION_VARIANT_NAMES.index(profile_name) * 1_000_000
            sample_seed = seed + profile_offset + int(record["source_index"])
            random.seed(sample_seed)
            torch.manual_seed(sample_seed)
            with Image.open(source_path) as source_image:
                augmented_image = transform(source_image.convert("RGB"))
            _save_jpeg(_pil_from_tensor(augmented_image), output_path)

        marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    return root / AUGMENTATION_PROFILE_DIR


def _finalize_prepared_dataset(root: Path) -> Path:
    if root.name in MATERIALIZED_AUGMENTATION_DATASET_SLUGS:
        write_augmentation_profile_manifests(root, root.name)
    return root


def evaluation_augment_from_training(augment: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "enabled": False,
        "resize": list(augment.get("resize", [224, 224])),
        "normalize": bool(augment.get("normalize", False)),
    }


def build_transform(augment: Optional[Dict[str, Any]] = None) -> Any:
    _, transforms, _, _ = _require_torch()
    augment = dict(DEFAULT_AUGMENT if augment is None else augment)
    enabled = bool(augment.get("enabled", True))
    ops: List[Any] = []
    resize = augment.get("resize", [64, 64])
    crop_scale = augment.get("random_resized_crop_scale")
    if enabled and resize and crop_scale:
        crop_ratio = augment.get("random_resized_crop_ratio", [0.75, 1.3333333333])
        ops.append(
            transforms.RandomResizedCrop(
                tuple(resize),
                scale=tuple(crop_scale),
                ratio=tuple(crop_ratio),
            )
        )
    elif resize:
        ops.append(transforms.Resize(tuple(resize)))
    if enabled and augment.get("horizontal_flip", False):
        ops.append(transforms.RandomHorizontalFlip())
    if enabled and augment.get("vertical_flip", False):
        ops.append(transforms.RandomVerticalFlip())
    if enabled and augment.get("rotation_degrees"):
        ops.append(transforms.RandomRotation(float(augment["rotation_degrees"])))
    if enabled and augment.get("perspective_distortion"):
        ops.append(
            transforms.RandomPerspective(
                distortion_scale=float(augment["perspective_distortion"]),
                p=float(augment.get("perspective_probability", 0.5)),
            )
        )
    if enabled and augment.get("color_jitter"):
        jitter = augment["color_jitter"]
        ops.append(transforms.ColorJitter(*jitter))
    if enabled and augment.get("random_grayscale_probability"):
        ops.append(
            transforms.RandomGrayscale(
                p=float(augment["random_grayscale_probability"])
            )
        )
    if enabled and augment.get("gaussian_blur_probability"):
        blur = transforms.GaussianBlur(
            kernel_size=int(augment.get("gaussian_blur_kernel_size", 5))
        )
        ops.append(
            transforms.RandomApply(
                [blur], p=float(augment["gaussian_blur_probability"])
            )
        )
    ops.append(transforms.ToTensor())
    if enabled and augment.get("random_erasing_probability"):
        ops.append(
            transforms.RandomErasing(
                p=float(augment["random_erasing_probability"]),
                scale=tuple(augment.get("random_erasing_scale", [0.02, 0.33])),
            )
        )
    if augment.get("normalize", False):
        ops.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(ops)


class LocalImageDataset:
    def __init__(
        self,
        *,
        data_dir: str,
        dataset_name: Optional[str] = None,
        client_id: str,
        poisoning_method: str,
        augmentation_profile: str = DEFAULT_AUGMENTATION_PROFILE,
        split: str = "train",
        transform: Any = None,
    ) -> None:
        _, _, _, Dataset = _require_torch()

        class _Dataset(Dataset):
            def __init__(self, outer: "LocalImageDataset") -> None:
                self.outer = outer

            def __len__(self) -> int:
                return len(self.outer.records)

            def __getitem__(self, idx: int) -> Tuple[Any, int]:
                record = self.outer.records[idx]
                image = Image.open(record["image_path"]).convert("RGB")
                if self.outer.transform is not None:
                    image = self.outer.transform(image)
                return image, int(record["label"])

        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.client_id = client_id
        self.poisoning_method = poisoning_method
        self.augmentation_profile = augmentation_profile
        self.split = split
        self.transform = transform
        self.records = load_metadata_records(
            data_dir=data_dir,
            dataset_name=dataset_name,
            client_id=client_id,
            poisoning_method=poisoning_method,
            split=split,
        )
        if poisoning_method == POISONING_METHOD_AVAILABILITY_SHORTCUTS:
            if split != "train":
                raise ValueError(
                    "Availability shortcuts are only defined for the training split; "
                    "use clean data for evaluation."
                )
            if not self.records:
                raise ValueError(
                    f"No availability-shortcut records found for client {client_id!r}."
                )
            generators = {record.get("shortcut_generator", "") for record in self.records}
            if generators != {AVAILABILITY_SHORTCUT_GENERATOR}:
                raise ValueError(
                    "Availability-shortcut metadata is stale or inconsistent. "
                    "Regenerate it with dataset_preparation.py --force; "
                    f"found generators={sorted(generators)!r}."
                )

        if poisoning_method == POISONING_METHOD_BADSAMPLING:
            if split != "train":
                raise ValueError("BadSampler is only valid for the training split.")
            self.records.sort(
                key=lambda record: (
                    int(record.get("source_index", 0)),
                    str(record.get("relative_path", "")),
                )
            )
            _validate_badsampler_plan(
                root=_dataset_root(data_dir, dataset_name),
                client_id=client_id,
                records=self.records,
            )
        if augmentation_profile != DEFAULT_AUGMENTATION_PROFILE and split == "train":
            if poisoning_method != POISONING_METHOD_CLEAN:
                raise ValueError(
                    "Saved augmentation profiles are clean-data conditions and cannot "
                    f"be combined with poisoning_method={poisoning_method!r}."
                )
            root = _dataset_root(data_dir, dataset_name)
            for record in self.records:
                augmented_path = _augmented_clean_path(root, augmentation_profile, record)
                if not augmented_path.exists():
                    raise FileNotFoundError(
                        f"Missing saved augmented image: {augmented_path}. "
                        "Run dataset_preparation.py first."
                    )
                record["image_path"] = str(augmented_path)
        self._dataset = _Dataset(self)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        return self._dataset[idx]


def load_metadata_records(
    *,
    data_dir: str,
    dataset_name: Optional[str] = None,
    client_id: str,
    poisoning_method: str,
    split: str = "train",
) -> List[Dict[str, Any]]:
    path = _dataset_root(data_dir, dataset_name) / METADATA_NAME
    if not path.exists():
        raise FileNotFoundError(f"Prepared metadata not found: {path}. Run dataset_preparation.py first.")
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    source_method = (
        POISONING_METHOD_CLEAN
        if poisoning_method == POISONING_METHOD_BADSAMPLING
        else poisoning_method
    )
    selected: List[Dict[str, Any]] = []
    for row in rows:
        if (
            (client_id in {"all", "*"} or row["client_id"] == client_id)
            and row["poisoning_method"] == source_method
            and row["dataset_split"] == split
        ):
            row = dict(row)
            row["image_path"] = _resolve_metadata_image_path(row["image_path"], data_dir, dataset_name)
            selected.append(row)
    return selected


def _validate_badsampler_plan(
    *,
    root: Path,
    client_id: str,
    records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    plan_path = (
        root
        / "poisoned"
        / POISONING_METHOD_BADSAMPLING
        / client_id
        / BADSAMPLER_PLAN_NAME
    )
    if not plan_path.exists():
        raise FileNotFoundError(
            f"Missing BadSampler plan: {plan_path}. Run dataset_preparation.py first."
        )
    plan = json.loads(plan_path.read_text())
    expected = [
        (int(record["source_index"]), str(record["relative_path"]), int(record["label"]))
        for record in records
    ]
    actual = [
        (int(item["source_index"]), str(item["relative_path"]), int(item["label"]))
        for item in plan.get("candidates", [])
    ]
    if actual != expected:
        raise ValueError(
            f"BadSampler plan does not match current clean records: {plan_path}. "
            "Regenerate the badsampling scenario."
        )
    return plan


def calculate_badsampler_losses(
    *,
    model: Any,
    dataset: Any,
    batch_size: int,
    device: Optional[Any] = None,
) -> Any:
    torch, _, DataLoader, _ = _require_torch()
    if len(dataset) == 0:
        raise ValueError("BadSampler cannot score an empty local dataset.")
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    original_training = bool(model.training)
    try:
        original_device = next(model.parameters()).device
    except StopIteration:
        original_device = torch.device("cpu")

    python_rng_state = random.getstate()
    torch_rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        model.to(target_device)
        model.eval()
        loader = DataLoader(
            dataset,
            batch_size=max(1, batch_size),
            shuffle=False,
            num_workers=0,
        )
        losses = []
        with torch.no_grad():
            for images, labels in loader:
                outputs = model(images.to(target_device))
                batch_losses = torch.nn.functional.cross_entropy(
                    outputs,
                    labels.to(target_device),
                    reduction="none",
                )
                losses.extend(batch_losses.detach().cpu().tolist())
        return torch.tensor(losses, dtype=torch.float64)
    finally:
        model.to(original_device)
        model.train(original_training)
        random.setstate(python_rng_state)
        torch.random.set_rng_state(torch_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)


def make_badsampler(
    *,
    losses: Any,
    batch_size: int,
    kappa: float,
    seed: int,
) -> Tuple[Any, List[int]]:
    torch, _, _, _ = _require_torch()
    from torch.utils.data import WeightedRandomSampler

    if len(losses) == 0:
        raise ValueError("BadSampler requires at least one loss value.")
    if batch_size <= 0 or kappa <= 0:
        raise ValueError("BadSampler batch_size and kappa must be positive.")
    pool_size = min(len(losses), max(1, math.ceil(kappa * batch_size)))
    hard_positions = torch.topk(losses, k=pool_size, largest=True).indices
    weights = torch.zeros(len(losses), dtype=torch.float64)
    weights[hard_positions] = 1.0
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(losses),
        replacement=True,
        generator=generator,
    )
    return sampler, [int(position) for position in hard_positions.tolist()]


def _write_badsampler_runtime_ranking(
    *,
    root: Path,
    client_id: str,
    records: Sequence[Dict[str, Any]],
    losses: Any,
    hard_positions: Sequence[int],
    batch_size: int,
    kappa: float,
    seed: int,
    run_name: str,
    num_epochs: int,
) -> Path:
    torch, _, _, _ = _require_torch()
    hard_set = set(hard_positions)
    ranked_positions = sorted(
        range(len(records)), key=lambda position: float(losses[position]), reverse=True
    )
    ranking = []
    for rank, position in enumerate(ranked_positions, start=1):
        record = records[position]
        ranking.append(
            {
                "rank": rank,
                "position": position,
                "source_index": int(record["source_index"]),
                "relative_path": str(record["relative_path"]),
                "label": int(record["label"]),
                "loss": float(losses[position]),
                "in_sampling_pool": position in hard_set,
            }
        )
    safe_run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name).strip("_") or "run"
    output_dir = root / "poisoned" / POISONING_METHOD_BADSAMPLING / client_id / "rankings"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{safe_run_name}.json"
    weights = torch.zeros(len(records), dtype=torch.float64)
    weights[list(hard_positions)] = 1.0
    preview_generator = torch.Generator().manual_seed(seed)
    sampled_positions_by_epoch = []
    sampled_source_indices_by_epoch = []
    for _ in range(max(0, num_epochs)):
        sampled_positions = torch.multinomial(
            weights,
            num_samples=len(records),
            replacement=True,
            generator=preview_generator,
        ).tolist()
        sampled_positions_by_epoch.append(sampled_positions)
        sampled_source_indices_by_epoch.append(
            [int(records[position]["source_index"]) for position in sampled_positions]
        )

    payload = {
        "attack": POISONING_METHOD_BADSAMPLING,
        "client_id": client_id,
        "seed": seed,
        "batch_size": batch_size,
        "kappa": kappa,
        "pool_size": len(hard_positions),
        "num_samples_per_epoch": len(records),
        "replacement": True,
        "num_epochs": num_epochs,
        "sampled_positions_by_epoch": sampled_positions_by_epoch,
        "sampled_source_indices_by_epoch": sampled_source_indices_by_epoch,
        "ranking": ranking,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output_path


def get_poison_fraction(
    *,
    data_dir: str,
    dataset_name: Optional[str] = None,
    client_id: str,
    poisoning_method: str,
    split: str = "train",
) -> float:
    records = load_metadata_records(
        data_dir=data_dir,
        dataset_name=dataset_name,
        client_id=client_id,
        poisoning_method=poisoning_method,
        split=split,
    )
    if not records:
        return 0.0
    if poisoning_method == POISONING_METHOD_BADSAMPLING:
        return 1.0
    poisoned = 0
    for record in records:
        is_poisoned = str(record.get("is_poisoned", "")).lower() == "true"
        label_changed = str(record.get("label_changed", "")).lower() == "true"
        if is_poisoned or label_changed:
            poisoned += 1
    return poisoned / len(records)


def get_num_classes(data_dir: str, dataset_name: Optional[str] = None) -> int:
    path = _dataset_root(data_dir, dataset_name) / METADATA_NAME
    if not path.exists():
        raise FileNotFoundError(f"Prepared metadata not found: {path}")
    with path.open(newline="") as f:
        return len({int(row["label"]) for row in csv.DictReader(f)})


def get_dataloader(
    *,
    data_dir: str,
    dataset_name: Optional[str] = None,
    client_id: str,
    poisoning_method: str,
    split: str,
    augment: Dict[str, Any],
    batch_size: int,
    shuffle: bool,
    surrogate_model: Any = None,
    badsampler_kappa: float = DEFAULT_BADSAMPLER_KAPPA,
    badsampler_seed: int = 0,
    badsampler_run_name: str = "run",
    badsampler_device: Optional[Any] = None,
    badsampler_num_epochs: int = 1,
) -> Any:
    _, _, DataLoader, _ = _require_torch()
    augmentation_profile = str(
        augment.get("_profile", DEFAULT_AUGMENTATION_PROFILE)
    )
    loader_augment = augment
    if augmentation_profile != DEFAULT_AUGMENTATION_PROFILE and split == "train":
        loader_augment = evaluation_augment_from_training(augment)
    dataset = LocalImageDataset(
        data_dir=data_dir,
        dataset_name=dataset_name,
        client_id=client_id,
        poisoning_method=poisoning_method,
        augmentation_profile=augmentation_profile,
        split=split,
        transform=build_transform(loader_augment),
    )
    if poisoning_method != POISONING_METHOD_BADSAMPLING:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    if surrogate_model is None:
        raise ValueError("BadSampler requires surrogate_model to rank local samples.")
    losses = calculate_badsampler_losses(
        model=surrogate_model,
        dataset=dataset,
        batch_size=batch_size,
        device=badsampler_device,
    )
    sampler, hard_positions = make_badsampler(
        losses=losses,
        batch_size=batch_size,
        kappa=badsampler_kappa,
        seed=badsampler_seed,
    )
    ranking_path = _write_badsampler_runtime_ranking(
        root=_dataset_root(data_dir, dataset_name),
        client_id=client_id,
        records=dataset.records,
        losses=losses,
        hard_positions=hard_positions,
        batch_size=batch_size,
        kappa=badsampler_kappa,
        seed=badsampler_seed,
        run_name=badsampler_run_name,
        num_epochs=badsampler_num_epochs,
    )
    print(
        f"BadSampler client={client_id} candidates={len(dataset)} "
        f"pool={len(hard_positions)} replacement=true ranking={ranking_path}"
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    if "--test-fraction" not in parser._option_string_actions:
        parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    if "--test-seed" not in parser._option_string_actions:
        parser.add_argument("--test-seed", type=int, default=DEFAULT_TEST_SEED)
    if "--partition-method" not in parser._option_string_actions:
        parser.add_argument(
            "--partition-method",
            choices=["iid", "dirichlet_noniid"],
            default=DEFAULT_PARTITION_METHOD,
        )
    if "--noniid-alpha" not in parser._option_string_actions:
        parser.add_argument("--noniid-alpha", type=float, default=DEFAULT_NONIID_ALPHA)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--prepare-scenarios",
        default=PREPARE_SCENARIO_ALL,
        help=(
            "Comma-separated dataset scenarios to generate. "
            f"Use {PREPARE_SCENARIO_ALL!r} or exact names: {', '.join(PREPARE_SCENARIOS)}. "
            f"Example: --prepare-scenarios {POISONING_METHOD_AVAILABILITY_SHORTCUTS}"
        ),
    )
    parser.add_argument("--poison-epsilon", type=float, default=16.0)
    parser.add_argument("--poison-steps", type=int, default=20)
    parser.add_argument("--poison-step-size", type=float, default=1.6)
    parser.add_argument("--poison-warmup-epochs", type=int, default=10)
    parser.add_argument("--poison-outer-iterations", type=int, default=10)
    parser.add_argument("--poison-surrogate-steps-per-outer", type=int, default=100)
    parser.add_argument("--poison-stop-error", type=float, default=0.01)
    parser.add_argument("--unlearnable-repo", default=DEFAULT_UNLEARNABLE_REPO)
    parser.add_argument("--random-label-flip-fraction", type=float, default=DEFAULT_RANDOM_LABEL_FLIP_FRACTION)
    parser.add_argument("--target-label", type=int, default=DEFAULT_TARGET_LABEL_FLIP_TARGET_LABEL)
    parser.add_argument("--replacement-label", type=int, default=DEFAULT_TARGET_LABEL_FLIP_REPLACEMENT_LABEL)
    parser.add_argument("--shortcut-eps", type=float, default=DEFAULT_AVAILABILITY_SHORTCUT_EPS)
    parser.add_argument("--shortcut-patch-size", type=int, default=DEFAULT_AVAILABILITY_SHORTCUT_PATCH_SIZE)
    parser.add_argument(
        "--prune-unreferenced",
        action="store_true",
        help="Delete image files under this dataset root that are not referenced by metadata.",
    )
    args = parser.parse_args()
    augment = augment_from_args(args)
    resize = augment.get("resize", [64, 64])
    root = prepare_dataset(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        num_clients=args.num_clients,
        seed=args.seed,
        force=args.force,
        resize=resize,
        poison_epsilon=args.poison_epsilon,
        poison_steps=args.poison_steps,
        poison_step_size=args.poison_step_size,
        poison_warmup_epochs=args.poison_warmup_epochs,
        poison_outer_iterations=args.poison_outer_iterations,
        poison_surrogate_steps_per_outer=args.poison_surrogate_steps_per_outer,
        poison_stop_error=args.poison_stop_error,
        batch_size=args.batch_size,
        unlearnable_repo=args.unlearnable_repo,
        random_label_flip_fraction=args.random_label_flip_fraction,
        target_label=args.target_label,
        replacement_label=args.replacement_label,
        shortcut_eps=args.shortcut_eps,
        shortcut_patch_size=args.shortcut_patch_size,
        test_fraction=args.test_fraction,
        test_seed=args.test_seed,
        prepare_scenarios=args.prepare_scenarios,
        partition_method=args.partition_method,
        noniid_alpha=args.noniid_alpha,
    )
    requested_scenarios = _parse_prepare_scenarios(args.prepare_scenarios)
    has_augmentation_variants = (
        _materialized_augmentation_supported(args.dataset)
        and POISONING_METHOD_CLEAN in requested_scenarios
    )
    if has_augmentation_variants:
        prepare_augmentation_variants(
            data_dir=args.data_dir,
            dataset_name=args.dataset,
            force=args.force,
        )
    print(root)
    if has_augmentation_variants:
        print(f"Augmentation profiles: {root / AUGMENTATION_PROFILE_DIR}")
    if args.prune_unreferenced:
        removed = prune_unreferenced_images(
            data_dir=args.data_dir,
            dataset_name=args.dataset,
        )
        print(f"Pruned {removed} unreferenced JPEG files from {root}")


if __name__ == "__main__":
    main()
