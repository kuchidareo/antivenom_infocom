import argparse
import json
import statistics
import time

import torch

from models import get_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-depth", type=int, default=3)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    model = get_model(
        args.model,
        num_classes=args.num_classes,
        input_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        model_depth=args.model_depth,
    ).cpu()
    model.train()
    inputs = torch.randn(args.batch_size, 3, args.image_size, args.image_size)
    criterion = torch.nn.CrossEntropyLoss()
    targets = torch.randint(0, args.num_classes, (args.batch_size,))

    warmup_loss = criterion(model(inputs), targets)
    del warmup_loss

    timings_ms = []
    for _ in range(args.repeats):
        start_ns = time.perf_counter_ns()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        timings_ms.append((time.perf_counter_ns() - start_ns) / 1_000_000)
        del outputs
        del loss

    print(
        json.dumps(
            {
                "model": args.model,
                "model_depth": args.model_depth,
                "batch_size": args.batch_size,
                "image_size": args.image_size,
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "forward_min_ms": min(timings_ms),
                "forward_median_ms": statistics.median(timings_ms),
                "forward_max_ms": max(timings_ms),
                "expected_samples_at_10_fps": statistics.median(timings_ms) / 100.0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
