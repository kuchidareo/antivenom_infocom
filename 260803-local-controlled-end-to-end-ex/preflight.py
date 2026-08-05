import argparse
import csv
import json
import math
from pathlib import Path

from dataset_preparation import dataset_slug, get_dataloader, get_num_classes
from models import get_model
from perf_logger import DEFAULT_PERF_EVENTS, parse_perf_events, validate_perf_events


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the layer-level PMU experiment.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--dataset", default="uoft-cs/cifar10")
    parser.add_argument("--client-id", default="client_0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--expected-batches", type=int, default=16)
    parser.add_argument("--perf-events", default=",".join(DEFAULT_PERF_EVENTS))
    args = parser.parse_args()

    events = parse_perf_events(args.perf_events)
    validate_perf_events(events)

    results = {}
    loader_scenarios = (
        ("baseline", "baseline", "clean"),
        ("moderate_augmentation", "moderate", "clean"),
        ("strong_augmentation", "strong", "clean"),
        ("availability_shortcuts", "baseline", "availability_shortcuts"),
    )
    for scenario, profile, poisoning_method in loader_scenarios:
        augment = {
            "enabled": True,
            "_profile": profile,
            "resize": [32, 32],
            "horizontal_flip": False,
            "normalize": True,
        }
        loader = get_dataloader(
            data_dir=args.data_dir,
            dataset_name=args.dataset,
            client_id=args.client_id,
            poisoning_method=poisoning_method,
            split="train",
            augment=augment,
            batch_size=args.batch_size,
            shuffle=False,
        )
        sample_count = len(loader.dataset)
        batch_count = math.ceil(sample_count / args.batch_size)
        images, labels = next(iter(loader))
        if tuple(images.shape[1:]) != (3, 32, 32):
            raise ValueError(f"Unexpected {scenario} input shape: {tuple(images.shape)}")
        if images.shape[0] != labels.shape[0]:
            raise ValueError(f"Image/label batch mismatch for scenario={scenario}.")
        results[scenario] = (sample_count, batch_count)

    root = Path(args.data_dir) / dataset_slug(args.dataset)
    metadata_path = root / "partition_metadata.csv"
    with metadata_path.open(newline="") as file:
        metadata_rows = list(csv.DictReader(file))
    clean_records = [
        row
        for row in metadata_rows
        if row.get("client_id") == args.client_id
        and row.get("dataset_split") == "train"
        and row.get("partition_method", "iid") == "iid"
        and row.get("poisoning_method") == "clean"
    ]
    plan_path = root / "poisoned" / "badsampling" / args.client_id / "sampling_plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(f"BadSampler plan is missing: {plan_path}")
    plan = json.loads(plan_path.read_text())
    candidate_count = len(plan.get("candidates", []))
    if candidate_count != len(clean_records):
        raise ValueError(
            f"BadSampler candidates={candidate_count}, clean records={len(clean_records)}"
        )
    results["badsampler"] = (
        len(clean_records),
        math.ceil(len(clean_records) / args.batch_size),
    )

    if args.expected_batches > 0:
        unexpected = {
            profile: count
            for profile, (_, count) in results.items()
            if count != args.expected_batches
        }
        if unexpected:
            raise ValueError(
                f"Expected {args.expected_batches} batches at batch size {args.batch_size}; "
                f"got {unexpected}."
            )

    model = get_model(
        "simple_cnn",
        num_classes=get_num_classes(args.data_dir, dataset_name=args.dataset),
        input_size=(32, 32),
        batch_size=args.batch_size,
    )
    leaves = [name for name, module in model.named_modules() if name and not any(module.children())]
    if len(leaves) != 15:
        raise ValueError(f"Expected 15 SimpleCNN leaf modules, got {len(leaves)}: {leaves}")

    print(f"perf_events={','.join(events)}")
    for scenario, (samples, batches) in results.items():
        print(f"scenario={scenario} samples={samples} batches={batches}")
    print(f"leaf_layers={len(leaves)} names={','.join(leaves)}")


if __name__ == "__main__":
    main()
