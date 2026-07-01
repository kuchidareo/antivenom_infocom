import argparse
import csv
import json
import os
import random
import sys
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

from experiment_config import (
    DATASET_NAME,
    DEFAULT_AUGMENT,
    DEFAULT_AVAILABILITY_SHORTCUT_EPS,
    DEFAULT_AVAILABILITY_SHORTCUT_PATCH_SIZE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATA_DIR,
    DEFAULT_NUM_CLIENTS,
    DEFAULT_RANDOM_LABEL_FLIP_FRACTION,
    DEFAULT_TARGET_LABEL_FLIP_REPLACEMENT_LABEL,
    DEFAULT_TARGET_LABEL_FLIP_TARGET_LABEL,
    POISONING_METHOD_AVAILABILITY_SHORTCUTS,
    POISONING_METHOD_CLEAN,
    POISONING_METHOD_RANDOM_LABEL_FLIPPING,
    POISONING_METHOD_TARGET_LABEL_FLIPPING,
    POISONING_METHOD_UNLEARNABLE_EXAMPLES,
    add_common_args,
    augment_from_args,
    client_index,
    set_all_seeds,
)


METADATA_NAME = "partition_metadata.csv"
PREPARED_MARKER = "PREPARED"
PREPARE_SCENARIO_ALL = "all"
PREPARE_SCENARIOS = [
    POISONING_METHOD_CLEAN,
    POISONING_METHOD_UNLEARNABLE_EXAMPLES,
    POISONING_METHOD_RANDOM_LABEL_FLIPPING,
    POISONING_METHOD_TARGET_LABEL_FLIPPING,
    POISONING_METHOD_AVAILABILITY_SHORTCUTS,
]

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
    "client_id",
    "partition_id",
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


def _dataset_root(data_dir: str) -> Path:
    return Path(data_dir) / "small_trashnet"


def prepared_data_exists(data_dir: str, num_clients: int = DEFAULT_NUM_CLIENTS) -> bool:
    root = _dataset_root(data_dir)
    if not (root / METADATA_NAME).exists() or not (root / PREPARED_MARKER).exists():
        return False
    required_modes = [
        ("clean",),
        (f"poisoned/{POISONING_METHOD_UNLEARNABLE_EXAMPLES}",),
        (f"poisoned/{POISONING_METHOD_RANDOM_LABEL_FLIPPING}",),
        (f"poisoned/{POISONING_METHOD_TARGET_LABEL_FLIPPING}",),
        (f"poisoned/{POISONING_METHOD_AVAILABILITY_SHORTCUTS}",),
    ]
    for mode_options in required_modes:
        for idx in range(num_clients):
            if not any((root / mode / f"client_{idx}").exists() for mode in mode_options):
                return False
    return True


def _extract_image_label(example: Dict[str, Any]) -> Tuple[Image.Image, int]:
    image = example.get("image") or example.get("img")
    if image is None:
        for value in example.values():
            if isinstance(value, Image.Image):
                image = value
                break
    label = example.get("label")
    if label is None:
        label = example.get("labels")
    if image is None or label is None:
        raise ValueError(f"Could not infer image/label fields from dataset example keys: {list(example.keys())}")
    return image.convert("RGB"), int(label)


def _class_names(ds: Any) -> List[str]:
    features = getattr(ds, "features", {})
    label_feature = features.get("label") if hasattr(features, "get") else None
    names = getattr(label_feature, "names", None)
    if names:
        return list(names)
    labels = sorted({int(row["label"]) for row in ds if "label" in row})
    return [str(label) for label in labels]


def _iter_splits(dataset_dict: Any) -> Iterable[Tuple[str, Any]]:
    if hasattr(dataset_dict, "keys"):
        for split in dataset_dict.keys():
            yield split, dataset_dict[split]
    else:
        yield "train", dataset_dict


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


def _save_jpeg(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="JPEG", quality=95)


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


def _mode_complete(root: Path, mode: str, num_clients: int) -> bool:
    return all((root / mode / f"client_{idx}").exists() for idx in range(num_clients))


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


def _scenario_complete(root: Path, scenario: str, rows: Sequence[Dict[str, Any]], num_clients: int) -> bool:
    return _mode_complete(root, _scenario_mode(scenario), num_clients) and _metadata_has_method(rows, scenario)


def _clean_records_from_metadata(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records = [dict(row) for row in rows if row.get("poisoning_method") == POISONING_METHOD_CLEAN]
    for record in records:
        record["clean_path"] = record.get("clean_path") or record["image_path"]
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
) -> None:
    torch, transforms, DataLoader, Dataset = _require_torch()
    from models import get_model

    repo_path = Path(unlearnable_repo).resolve()
    if repo_path.exists():
        sys.path.insert(0, str(repo_path))
    try:
        from toolbox import PerturbationTool
    except Exception:
        PerturbationTool = None

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

    loader = DataLoader(CleanRecordDataset(clean_records), batch_size=batch_size, shuffle=False, num_workers=0)
    model = get_model("simple_cnn", num_classes=num_classes, input_size=(int(resize[0]), int(resize[1]))).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # One light surrogate pass keeps the unlearnable-example perturbation deterministic
    # without turning data preparation into a full training run.
    model.train()
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

    model.eval()
    if PerturbationTool is not None:
        tool = PerturbationTool(
            seed=seed,
            epsilon=epsilon / 255.0,
            num_steps=num_steps,
            step_size=step_size / 255.0,
        )
    else:
        tool = None

    for images, labels, indices in loader:
        images, labels = images.to(device), labels.to(device)
        if tool is not None:
            poisoned, _ = tool.min_min_attack(images, labels, model, optimizer, criterion)
        else:
            poisoned = _fallback_min_min(images, labels, model, criterion, epsilon / 255.0, step_size / 255.0, num_steps)
        for tensor, record_idx in zip(poisoned, indices):
            record = clean_records[int(record_idx)]
            out_path = output_root / record["client_id"] / record["relative_path"]
            _save_jpeg(_pil_from_tensor(tensor), out_path)


def _fallback_min_min(images: Any, labels: Any, model: Any, criterion: Any, epsilon: float, step_size: float, steps: int) -> Any:
    import torch

    eta = torch.zeros_like(images)
    perturbed = images.detach().clone()
    for _ in range(steps):
        perturbed.requires_grad_(True)
        model.zero_grad()
        loss = criterion(model(perturbed), labels)
        loss.backward()
        eta = torch.clamp(eta - step_size * perturbed.grad.detach().sign(), -epsilon, epsilon)
        perturbed = torch.clamp(images + eta, 0, 1).detach()
    return perturbed


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
    import numpy as np

    if patch_size <= 0:
        raise ValueError(f"patch_size must be > 0, got {patch_size}")
    height = int(resize[0])
    width = int(resize[1]) if len(resize) > 1 else height
    labels = sorted({int(record["label"]) for record in clean_records})
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    num_classes = len(labels)
    grid_h = (height + patch_size - 1) // patch_size
    grid_w = (width + patch_size - 1) // patch_size

    rng = np.random.default_rng(seed)
    class_grids = rng.normal(loc=0.0, scale=1.0, size=(num_classes, grid_h, grid_w, 3)).astype(np.float32)
    linf = float(eps) / 255.0
    l2_limit = _l2norm_limit_from_linf(linf, height * width * 3)

    rows: List[Dict[str, Any]] = []
    for record in clean_records:
        label = int(record["label"])
        class_idx = label_to_index[label]
        # Synthetic shortcuts are class-correlated low-frequency grids. A tiny
        # sample-specific jitter prevents every image in a class from receiving
        # an identical perturbation while preserving the class shortcut.
        low_res = class_grids[class_idx] + rng.normal(0.0, 0.05, size=(grid_h, grid_w, 3)).astype(np.float32)
        shortcut = np.repeat(np.repeat(low_res, patch_size, axis=0), patch_size, axis=1)[:height, :width, :]
        shortcut = _normalize_l2norm(shortcut[None, ...], l2_limit)[0]

        image = Image.open(record["clean_path"]).convert("RGB").resize((width, height))
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        poisoned_array = np.clip(image_array + shortcut, 0.0, 1.0)
        poisoned_image = Image.fromarray((poisoned_array * 255.0).round().astype(np.uint8), mode="RGB")

        out_path = output_root / record["client_id"] / record["relative_path"]
        _save_jpeg(poisoned_image, out_path)
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
    )

    for record in clean_records:
        poisoned_path = (
            root
            / "poisoned"
            / POISONING_METHOD_UNLEARNABLE_EXAMPLES
            / record["client_id"]
            / record["relative_path"]
        )
        poisoned = dict(record)
        poisoned.update(
            {
                "image_path": str(poisoned_path),
                "is_poisoned": True,
                "poisoning_method": POISONING_METHOD_UNLEARNABLE_EXAMPLES,
            }
        )
        metadata_rows.append(poisoned)


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
    random_label_flip_fraction: float,
    target_label: int,
    replacement_label: int,
    shortcut_eps: float,
    shortcut_patch_size: int,
) -> None:
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
                seed=seed + 47,
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
    poison_epsilon: float = 8.0,
    poison_steps: int = 5,
    poison_step_size: float = 0.8,
    batch_size: int = DEFAULT_BATCH_SIZE,
    unlearnable_repo: str = "../Unlearnable-Examples",
    random_label_flip_fraction: float = DEFAULT_RANDOM_LABEL_FLIP_FRACTION,
    target_label: int = DEFAULT_TARGET_LABEL_FLIP_TARGET_LABEL,
    replacement_label: int = DEFAULT_TARGET_LABEL_FLIP_REPLACEMENT_LABEL,
    shortcut_eps: float = DEFAULT_AVAILABILITY_SHORTCUT_EPS,
    shortcut_patch_size: int = DEFAULT_AVAILABILITY_SHORTCUT_PATCH_SIZE,
    prepare_scenarios: Optional[Any] = PREPARE_SCENARIO_ALL,
) -> Path:
    root = _dataset_root(data_dir)
    requested_scenarios = _parse_prepare_scenarios(prepare_scenarios)
    if prepared_data_exists(data_dir, num_clients) and not force and set(requested_scenarios) == set(PREPARE_SCENARIOS):
        return root

    existing_rows = _read_metadata_rows(root)
    clean_records_from_existing = _clean_records_from_metadata(existing_rows)
    if clean_records_from_existing and _mode_complete(root, "clean", num_clients) and (
        not force or POISONING_METHOD_CLEAN not in requested_scenarios
    ):
        missing_or_requested = [
            scenario
            for scenario in requested_scenarios
            if scenario != POISONING_METHOD_CLEAN
            and (force or not _scenario_complete(root, scenario, existing_rows, num_clients))
        ]
        if not missing_or_requested:
            return root
        metadata_rows = [row for row in existing_rows if row.get("poisoning_method") not in set(missing_or_requested)]
        class_names = _class_names_from_records(clean_records_from_existing)
        _append_requested_poisoning_rows(
            metadata_rows,
            clean_records_from_existing,
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
            random_label_flip_fraction=random_label_flip_fraction,
            target_label=target_label,
            replacement_label=replacement_label,
            shortcut_eps=shortcut_eps,
            shortcut_patch_size=shortcut_patch_size,
        )
        _write_metadata_rows(root, metadata_rows)
        (root / PREPARED_MARKER).write_text(json.dumps({"dataset": dataset_name, "num_clients": num_clients, "seed": seed}))
        return root

    if POISONING_METHOD_CLEAN not in requested_scenarios:
        raise FileNotFoundError(
            "Clean prepared data is required before generating poisoning-only scenarios. "
            f"Run with --prepare-scenarios {POISONING_METHOD_CLEAN} or --prepare-scenarios {PREPARE_SCENARIO_ALL} first."
        )

    from datasets import load_dataset

    root.mkdir(parents=True, exist_ok=True)
    dataset_dict = load_dataset(dataset_name)
    metadata_rows: List[Dict[str, Any]] = []
    clean_records: List[Dict[str, Any]] = []
    class_names: List[str] = []

    for split, ds in _iter_splits(dataset_dict):
        names = _class_names(ds)
        if not class_names:
            class_names = names
        assignments = _assign_iid_partitions(ds, num_clients, seed)
        for idx, example in enumerate(ds):
            image, label = _extract_image_label(example)
            class_name = names[label] if label < len(names) else str(label)
            client_id = assignments[idx]
            relative_path = Path(_safe_class_name(class_name)) / f"{split}_{idx:06d}.jpeg"
            clean_path = root / "clean" / client_id / relative_path
            _save_jpeg(image, clean_path)
            record = {
                "source_index": idx,
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
                "client_id": client_id,
                "partition_id": client_id,
                "dataset_split": split,
                "is_poisoned": False,
                "poisoning_method": POISONING_METHOD_CLEAN,
            }
            clean_records.append(record)
            metadata_rows.append(record)

    _append_requested_poisoning_rows(
        metadata_rows,
        clean_records,
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
        random_label_flip_fraction=random_label_flip_fraction,
        target_label=target_label,
        replacement_label=replacement_label,
        shortcut_eps=shortcut_eps,
        shortcut_patch_size=shortcut_patch_size,
    )

    _write_metadata_rows(root, metadata_rows)
    (root / PREPARED_MARKER).write_text(json.dumps({"dataset": dataset_name, "num_clients": num_clients, "seed": seed}))
    return root


def build_transform(augment: Optional[Dict[str, Any]] = None) -> Any:
    _, transforms, _, _ = _require_torch()
    augment = dict(DEFAULT_AUGMENT if augment is None else augment)
    enabled = bool(augment.get("enabled", True))
    ops: List[Any] = []
    resize = augment.get("resize", [64, 64])
    if resize:
        ops.append(transforms.Resize(tuple(resize)))
    if enabled and augment.get("horizontal_flip", False):
        ops.append(transforms.RandomHorizontalFlip())
    ops.append(transforms.ToTensor())
    if augment.get("normalize", False):
        ops.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(ops)


class LocalImageDataset:
    def __init__(
        self,
        *,
        data_dir: str,
        client_id: str,
        poisoning_method: str,
        split: str = "train",
        transform: Any = None,
    ) -> None:
        torch, _, _, Dataset = _require_torch()

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
        self.client_id = client_id
        self.poisoning_method = poisoning_method
        self.split = split
        self.transform = transform
        self.records = load_metadata_records(
            data_dir=data_dir,
            client_id=client_id,
            poisoning_method=poisoning_method,
            split=split,
        )
        self._dataset = _Dataset(self)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        return self._dataset[idx]


def load_metadata_records(
    *,
    data_dir: str,
    client_id: str,
    poisoning_method: str,
    split: str = "train",
) -> List[Dict[str, Any]]:
    path = _dataset_root(data_dir) / METADATA_NAME
    if not path.exists():
        raise FileNotFoundError(f"Prepared metadata not found: {path}. Run dataset_preparation.py first.")
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return [
        row
        for row in rows
        if row["client_id"] == client_id
        and row["poisoning_method"] == poisoning_method
        and row["dataset_split"] == split
    ]


def get_poison_fraction(
    *,
    data_dir: str,
    client_id: str,
    poisoning_method: str,
    split: str = "train",
) -> float:
    records = load_metadata_records(
        data_dir=data_dir,
        client_id=client_id,
        poisoning_method=poisoning_method,
        split=split,
    )
    if not records:
        return 0.0
    poisoned = 0
    for record in records:
        is_poisoned = str(record.get("is_poisoned", "")).lower() == "true"
        label_changed = str(record.get("label_changed", "")).lower() == "true"
        if is_poisoned or label_changed:
            poisoned += 1
    return poisoned / len(records)


def get_num_classes(data_dir: str) -> int:
    path = _dataset_root(data_dir) / METADATA_NAME
    if not path.exists():
        raise FileNotFoundError(f"Prepared metadata not found: {path}")
    with path.open(newline="") as f:
        return len({int(row["label"]) for row in csv.DictReader(f)})


def get_dataloader(
    *,
    data_dir: str,
    client_id: str,
    poisoning_method: str,
    split: str,
    augment: Dict[str, Any],
    batch_size: int,
    shuffle: bool,
) -> Any:
    _, _, DataLoader, _ = _require_torch()
    dataset = LocalImageDataset(
        data_dir=data_dir,
        client_id=client_id,
        poisoning_method=poisoning_method,
        split=split,
        transform=build_transform(augment),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
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
    parser.add_argument("--poison-epsilon", type=float, default=8.0)
    parser.add_argument("--poison-steps", type=int, default=5)
    parser.add_argument("--poison-step-size", type=float, default=0.8)
    parser.add_argument("--unlearnable-repo", default="../Unlearnable-Examples")
    parser.add_argument("--random-label-flip-fraction", type=float, default=DEFAULT_RANDOM_LABEL_FLIP_FRACTION)
    parser.add_argument("--target-label", type=int, default=DEFAULT_TARGET_LABEL_FLIP_TARGET_LABEL)
    parser.add_argument("--replacement-label", type=int, default=DEFAULT_TARGET_LABEL_FLIP_REPLACEMENT_LABEL)
    parser.add_argument("--shortcut-eps", type=float, default=DEFAULT_AVAILABILITY_SHORTCUT_EPS)
    parser.add_argument("--shortcut-patch-size", type=int, default=DEFAULT_AVAILABILITY_SHORTCUT_PATCH_SIZE)
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
        batch_size=args.batch_size,
        unlearnable_repo=args.unlearnable_repo,
        random_label_flip_fraction=args.random_label_flip_fraction,
        target_label=args.target_label,
        replacement_label=args.replacement_label,
        shortcut_eps=args.shortcut_eps,
        shortcut_patch_size=args.shortcut_patch_size,
        prepare_scenarios=args.prepare_scenarios,
    )
    print(root)


if __name__ == "__main__":
    main()
