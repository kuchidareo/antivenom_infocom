from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).with_name("2_augmentation_statistical_comparison.py")
SPEC = importlib.util.spec_from_file_location("distributional_comparison", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DistributionAggregationTests(unittest.TestCase):
    def test_collection_root_resolves_single_device_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            device = root / "192.168.0.141"
            condition = device / "cifar10_iid"
            condition.mkdir(parents=True)
            (condition / "run_perf.csv").touch()
            self.assertEqual(MODULE.resolve_input_dir(root), device)

    def test_piecewise_profile_is_resampled_without_interpolation(self) -> None:
        frame = pd.DataFrame(
            {
                "bin_index": [0, 1],
                "progress_start": [0.0, 0.5],
                "progress_end": [0.5, 1.0],
                "estimated_mean_rate": [10.0, 30.0],
                "estimated_variance_rate": [4.0, 16.0],
                "tau_squared": [2.0, 2.0],
            }
        )
        mean, variance_density, tau2 = MODULE.resample_profile(
            frame, np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        )
        np.testing.assert_allclose(mean, [10.0, 10.0, 30.0, 30.0, 30.0])
        np.testing.assert_allclose(variance_density, [4.0, 4.0, 16.0, 16.0, 16.0])
        self.assertEqual(tau2, 2.0)

    def test_gaussian_moment_aggregation_includes_between_epoch_variance(self) -> None:
        mean, variance = MODULE.aggregate_gaussians(
            np.array([[0.0, 2.0], [2.0, 4.0]]),
            np.array([[1.0, 1.0], [1.0, 1.0]]),
        )
        np.testing.assert_allclose(mean, [1.0, 3.0])
        np.testing.assert_allclose(variance, [2.0, 2.0])


class DistributionPermutationTests(unittest.TestCase):
    def arrays(self, shift: float) -> tuple[np.ndarray, ...]:
        progress = np.linspace(0.0, 1.0, 21)
        normal_mean = np.zeros((8, len(progress)))
        normal_variance = np.ones_like(normal_mean)
        target_mean = np.full_like(normal_mean, shift)
        target_variance = np.ones_like(normal_mean)
        return normal_mean, normal_variance, target_mean, target_variance, progress

    def test_identical_distributions_have_unit_permutation_p(self) -> None:
        result = MODULE.paired_distribution_test(*self.arrays(0.0))
        self.assertEqual(result["global_p"], 1.0)
        np.testing.assert_allclose(result["wasserstein_squared"], 0.0)

    def test_consistent_shift_is_detected(self) -> None:
        result = MODULE.paired_distribution_test(*self.arrays(4.0))
        self.assertAlmostEqual(result["global_wasserstein_squared"], 16.0)
        self.assertLessEqual(result["global_p"], 0.01)
        self.assertTrue(np.all(np.asarray(result["pointwise_fwer_p"]) <= 0.01))


if __name__ == "__main__":
    unittest.main()
