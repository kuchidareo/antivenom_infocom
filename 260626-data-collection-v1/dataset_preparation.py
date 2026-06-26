import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

from experiment_config import (
    DATASET_NAME,
    DEFAULT_AUGMENT,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATA_DIR,
    DEFAULT_NUM_CLIENTS,
    add_common_args,
    augment_from_args,
    client_index,
    set_all_seeds,
)


METADATA_NAME = "partition_metadata.csv"
PREPARED_MARKER = "PREPARED"


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
    for mode in ("clean", "poisoned/adaptive"):
        for idx in range(num_clients):
            if not (root / mode / f"client_{idx}").exists():
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


def _pil_from_tensor(tensor: Any) -> Image.Image:
    import torchvision.transforms.functional as TF

    return TF.to_pil_image(tensor.detach().cpu().clamp(0, 1))


def _build_adaptive_images(
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

    # One light surrogate pass keeps the adaptive perturbation deterministic
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
) -> Path:
    if prepared_data_exists(data_dir, num_clients) and not force:
        return _dataset_root(data_dir)

    from datasets import load_dataset

    root = _dataset_root(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    dataset_dict = load_dataset(dataset_name)
    metadata_rows: List[Dict[str, Any]] = []
    clean_records: List[Dict[str, Any]] = []

    for split, ds in _iter_splits(dataset_dict):
        names = _class_names(ds)
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
                "client_id": client_id,
                "partition_id": client_id,
                "dataset_split": split,
                "is_poisoned": False,
                "poisoning_method": "clean",
            }
            clean_records.append(record)
            metadata_rows.append(record)

    _build_adaptive_images(
        clean_records,
        output_root=root / "poisoned" / "adaptive",
        num_classes=len({int(row["label"]) for row in clean_records}),
        resize=resize,
        seed=seed,
        epsilon=poison_epsilon,
        num_steps=poison_steps,
        step_size=poison_step_size,
        batch_size=batch_size,
        unlearnable_repo=unlearnable_repo,
    )

    for record in clean_records:
        poisoned_path = root / "poisoned" / "adaptive" / record["client_id"] / record["relative_path"]
        poisoned = dict(record)
        poisoned.update(
            {
                "image_path": str(poisoned_path),
                "is_poisoned": True,
                "poisoning_method": "adaptive",
            }
        )
        metadata_rows.append(poisoned)

    fieldnames = [
        "image_path",
        "label",
        "class_name",
        "client_id",
        "partition_id",
        "dataset_split",
        "is_poisoned",
        "poisoning_method",
        "source_index",
        "relative_path",
    ]
    with (root / METADATA_NAME).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metadata_rows)
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
    parser.add_argument("--poison-epsilon", type=float, default=8.0)
    parser.add_argument("--poison-steps", type=int, default=5)
    parser.add_argument("--poison-step-size", type=float, default=0.8)
    parser.add_argument("--unlearnable-repo", default="../Unlearnable-Examples")
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
    )
    print(root)


if __name__ == "__main__":
    main()
