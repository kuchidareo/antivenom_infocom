from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import running_ml


class _Context:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None


class AugmentationSequenceTests(unittest.TestCase):
    def test_parser_accepts_baseline_to_strong(self) -> None:
        self.assertEqual(
            running_ml.parse_augmentation_sequences("baseline:strong"),
            [("baseline", "strong")],
        )

    def test_two_stages_share_model_and_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = Namespace(
                seed=42,
                dataset="uoft-cs/cifar10",
                data_dir="iid-data",
                client_id="client_1",
                dataset_split="train",
                batch_size=16,
                model="simple_cnn",
                model_depth=3,
                model_width_multiplier=1.0,
                model_target_pam_mb=0.0,
                model_pam_calibration_steps=1,
                learning_rate=0.001,
                local_epochs=10,
                experiment_id="test_sequence",
                log_dir=temporary,
                hardware_fps=50.0,
                perf_fps=50.0,
                perf_events="cycles,instructions",
                augment='{"enabled": true, "resize": [32, 32], "normalize": true}',
                num_clients=10,
                num_rounds=10,
                host="192.168.0.141",
                device_id="192.168.0.141",
                trial_id="trial_0",
                run_role="augmentation_state_analysis",
                test_fraction=0.2,
                test_seed=260626,
            )
            model = torch.nn.Linear(4, 2)
            optimizer = object()
            train_model = mock.Mock(return_value={})
            hardware_context = _Context(Path(temporary) / "run.csv")

            def loader(**kwargs):
                if kwargs["split"] == "test":
                    return Namespace(dataset=list(range(20)))
                return Namespace(profile=kwargs["augment"]["_profile"])

            with (
                mock.patch.object(running_ml, "set_all_seeds"),
                mock.patch.object(running_ml, "get_num_classes", return_value=10),
                mock.patch.object(running_ml, "get_model", return_value=model),
                mock.patch.object(running_ml.torch.optim, "Adam", return_value=optimizer),
                mock.patch.object(running_ml, "get_dataloader", side_effect=loader),
                mock.patch.object(running_ml, "condition_columns", return_value={}),
                mock.patch.object(running_ml, "HardwareLogger", return_value=hardware_context),
                mock.patch.object(running_ml, "PerfLogger", return_value=_Context()),
                mock.patch.object(running_ml, "MetricsLogger", return_value=mock.Mock()),
                mock.patch.object(running_ml, "train_model", train_model),
            ):
                running_ml.run_one_augmentation_sequence(
                    args,
                    first_profile="baseline",
                    second_profile="strong",
                    stage_epochs=10,
                )

            self.assertEqual(train_model.call_count, 2)
            first = train_model.call_args_list[0].kwargs
            second = train_model.call_args_list[1].kwargs
            self.assertIs(first["model"], second["model"])
            self.assertIs(first["optimizer"], second["optimizer"])
            self.assertEqual((first["epoch_offset"], second["epoch_offset"]), (0, 10))
            self.assertEqual(
                (first["input_augmentation_profile"], second["input_augmentation_profile"]),
                ("baseline", "strong"),
            )


if __name__ == "__main__":
    unittest.main()
