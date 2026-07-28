from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).with_name("5_batch_blocked_condition_effect.py")
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("batch_blocked_effect", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BatchBlockedFitTests(unittest.TestCase):
    @staticmethod
    def observations(mu: np.ndarray, effects: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
        clean_rows = []
        target_rows = []
        k = len(mu)
        batch_offsets = np.linspace(-0.3, 0.3, 8)
        edges = np.linspace(0.0, 1.0, k + 1)
        for batch, offset in enumerate(batch_offsets):
            for index in range(k):
                width = edges[index + 1] - edges[index]
                common = {
                    "batch_idx": batch,
                    "a": edges[index],
                    "b": edges[index + 1],
                    "width": width,
                }
                clean_rows.append(common | {"observed_increment": width * (mu[index] + offset)})
                target_rows.append(common | {
                    "observed_increment": width * (mu[index] + offset + effects[index])
                })
        return pd.DataFrame(clean_rows), pd.DataFrame(target_rows)

    def test_joint_fit_removes_shared_batch_offsets_from_raw_increments(self) -> None:
        mu = np.array([2.0, 3.0, 4.0, 5.0])
        effect = np.array([0.2, 0.4, 0.6, 0.8])
        clean, target = self.observations(mu, effect)
        adjusted_clean, adjusted_target, diagnostics = MODULE.shared_batch_adjustment(
            clean,
            target,
            k=4,
            mean_smoothness=0,
            batch_ridge=0,
        )
        expected_clean = []
        expected_target = []
        for _ in range(8):
            expected_clean.extend(0.25 * mu)
            expected_target.extend(0.25 * (mu + effect))
        np.testing.assert_allclose(
            adjusted_clean["observed_increment"], expected_clean, atol=1e-8
        )
        np.testing.assert_allclose(
            adjusted_target["observed_increment"], expected_target, atol=1e-8
        )
        self.assertEqual(diagnostics["rank"], diagnostics["parameters"])

    def test_resampling_preserves_raw_piecewise_profile_values(self) -> None:
        values = np.array([10.0, 20.0, 30.0, 40.0])
        edges = np.linspace(0.0, 1.0, 5)
        progress = np.array([0.0, 0.24, 0.25, 0.74, 1.0])
        np.testing.assert_allclose(
            MODULE.resample_piecewise(values, edges, progress),
            [10.0, 10.0, 20.0, 30.0, 40.0],
        )


if __name__ == "__main__":
    unittest.main()
