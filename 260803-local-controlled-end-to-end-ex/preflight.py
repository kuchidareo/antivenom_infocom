import argparse
import math

from dataset_preparation import get_dataloader, get_num_classes
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
    for profile in ("baseline", "strong"):
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
            poisoning_method="clean",
            split="train",
            augment=augment,
            batch_size=args.batch_size,
            shuffle=False,
        )
        sample_count = len(loader.dataset)
        batch_count = math.ceil(sample_count / args.batch_size)
        images, labels = next(iter(loader))
        if tuple(images.shape[1:]) != (3, 32, 32):
            raise ValueError(f"Unexpected {profile} input shape: {tuple(images.shape)}")
        if images.shape[0] != labels.shape[0]:
            raise ValueError(f"Image/label batch mismatch for profile={profile}.")
        results[profile] = (sample_count, batch_count)

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
    for profile, (samples, batches) in results.items():
        print(f"profile={profile} samples={samples} batches={batches}")
    print(f"leaf_layers={len(leaves)} names={','.join(leaves)}")


if __name__ == "__main__":
    main()
