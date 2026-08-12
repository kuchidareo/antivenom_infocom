import csv
import tempfile
import unittest
from pathlib import Path

import torch

from entropy_logger import (
    LayerEntropyLogger,
    _maxpool_comparison_outcomes,
    maxpool_position_markov_metrics,
)


class MaxPoolMarkovTests(unittest.TestCase):
    def test_comparison_order_matches_nchw_window_scan(self):
        tensor = torch.tensor(
            [
                [[
                    [4.0, 1.0, 3.0, 2.0],
                    [0.0, 2.0, 5.0, 1.0],
                    [9.0, 8.0, 1.0, 7.0],
                    [6.0, 5.0, 4.0, 3.0],
                ]]
            ]
        )
        pool = torch.nn.MaxPool2d(2, stride=2)
        outcomes = _maxpool_comparison_outcomes(tensor, pool)
        expected = torch.tensor(
            [
                [True, False, False, False],  # [4, 1, 0, 2]
                [True, False, True, False],   # [3, 2, 5, 1]
                [True, False, False, False],  # [9, 8, 6, 5]
                [True, True, False, False],   # [1, 7, 4, 3]
            ]
        )
        torch.testing.assert_close(outcomes, expected)

    def test_metrics_count_four_comparisons_per_window(self):
        generator = torch.Generator().manual_seed(260803)
        tensor = torch.randn(2, 3, 8, 8, generator=generator)
        metrics = maxpool_position_markov_metrics(
            tensor, torch.nn.MaxPool2d(2, stride=2)
        )
        self.assertEqual(metrics["windows"], 2 * 3 * 4 * 4)
        self.assertEqual(metrics["logical_comparisons"], 4 * metrics["windows"])
        self.assertEqual(metrics["position_p1_b1"], 1.0)
        self.assertGreaterEqual(metrics["position_markov_entropy_rate_bits"], 0.0)
        self.assertLessEqual(metrics["position_markov_entropy_rate_bits"], 1.0)

    def test_logger_writes_one_compact_row_per_maxpool_invocation(self):
        with tempfile.TemporaryDirectory() as directory:
            summary_path = Path(directory) / "run_entropy_summary.csv"
            logger = LayerEntropyLogger(
                path=summary_path,
                condition={"scenario": "baseline", "device_id": "test"},
            )
            pool = torch.nn.MaxPool2d(2, stride=2)
            tensor = torch.randn(2, 3, 8, 8)
            logger.begin_batch(epoch=0, batch_idx=0, round_id=0)
            logger.capture_forward(
                module=pool,
                layer_index=3,
                layer_name="features.3",
                layer_type="MaxPool2d",
                invocation_index=0,
                inputs=(tensor,),
                output=pool(tensor),
            )
            logger.end_batch()
            logger.close()

            with logger.maxpool_markov_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["layer_name"], "features.3")
            self.assertEqual(int(rows[0]["logical_comparisons"]), 384)


if __name__ == "__main__":
    unittest.main()
