#!/usr/bin/env python3
"""Synthetic and edge-case tests for execution-profile estimation."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from execution_profiles import (
    build_forward_observations,
    build_overlap_matrix,
    fit_profile,
    resolve_counter_columns,
)


class OverlapMatrixTests(unittest.TestCase):
    def test_overlap_integrates_interval_width(self) -> None:
        starts = np.array([0.0, 0.1, 0.45, 0.8])
        ends = np.array([0.2, 0.6, 0.9, 1.0])
        matrix, edges = build_overlap_matrix(starts, ends, 4)
        np.testing.assert_allclose(matrix.sum(axis=1), ends - starts, atol=1e-12)
        np.testing.assert_allclose(edges, np.linspace(0, 1, 5))
        self.assertEqual(matrix.shape, (4, 4))

    def test_invalid_intervals_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_overlap_matrix(np.array([0.4]), np.array([0.3]), 4)
        with self.assertRaises(ValueError):
            build_overlap_matrix(np.array([-0.1]), np.array([0.3]), 4)


class SyntheticRecoveryTests(unittest.TestCase):
    @staticmethod
    def synthetic(seed: int = 91) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        starts: list[float] = []
        ends: list[float] = []
        for _ in range(100):
            widths = rng.dirichlet(np.full(rng.integers(4, 9), 2.0))
            boundaries = np.r_[0.0, np.cumsum(widths)]
            starts.extend(boundaries[:-1])
            ends.extend(boundaries[1:])
        matrix, _ = build_overlap_matrix(np.array(starts), np.array(ends), 4)
        true_mu = np.array([35.0, 80.0, 45.0, 65.0])
        true_q = np.array([20.0, 45.0, 15.0, 35.0])
        tau2 = 0.5
        mean = matrix @ true_mu
        variance = matrix @ true_q + tau2
        observed = rng.normal(mean, np.sqrt(variance))
        return matrix, observed, true_mu, true_q

    def test_profile_shape_and_positive_variance_are_recovered(self) -> None:
        matrix, observed, true_mu, true_q = self.synthetic()
        fitted = fit_profile(
            matrix,
            observed,
            tau2=0.5,
            mean_smoothness=0.01,
            variance_smoothness=0.05,
        )
        self.assertTrue(np.isfinite(fitted.objective))
        self.assertTrue(np.all(fitted.q > 0))
        self.assertLess(np.sqrt(np.mean((fitted.mu - true_mu) ** 2)), 12.0)
        self.assertGreater(float(np.corrcoef(fitted.q, true_q)[0, 1]), 0.8)
        # With +log(v), variance remains finite instead of drifting to its bound.
        self.assertLess(float(np.max(fitted.q)), 1e5)

    def test_seeded_input_is_reproducible(self) -> None:
        first = self.synthetic(123)
        second = self.synthetic(123)
        for left, right in zip(first, second):
            np.testing.assert_allclose(left, right)

    def test_positive_edge_bin_does_not_stick_at_zero(self) -> None:
        rng = np.random.default_rng(42)
        true_mu = np.array([50.0, 90.0, 70.0, 55.0])
        true_q = np.array([600.0, 900.0, 700.0, 500.0])
        starts: list[float] = []
        ends: list[float] = []
        observed: list[float] = []
        edges = np.linspace(0.0, 1.0, 5)

        for _ in range(13):
            boundaries = np.r_[0.0, np.cumsum(rng.dirichlet(np.ones(4)))]
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                overlap = np.maximum(
                    0.0,
                    np.minimum(end, edges[1:]) - np.maximum(start, edges[:-1]),
                )
                value = overlap @ true_mu + rng.normal(
                    0.0, np.sqrt(overlap @ true_q + 0.5)
                )
                starts.append(float(start))
                ends.append(float(end))
                observed.append(max(0.0, float(value)))

        matrix, _ = build_overlap_matrix(np.array(starts), np.array(ends), 4)
        fitted = fit_profile(
            matrix,
            np.array(observed),
            tau2=0.5,
            mean_smoothness=0.2,
            variance_smoothness=0.2,
            maxiter=300,
        )

        self.assertTrue(fitted.success, fitted.message)
        self.assertGreater(fitted.mu[0], 5.0)


class ObservationConstructionTests(unittest.TestCase):
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp_unix": [1.0, 1.02, 1.04, 1.06, 1.08],
                "perf_elapsed_sec": [0.02, 0.04, 0.06, 0.08, 0.10],
                "perf_events": ["cycles,instructions"] * 5,
                "epoch": [0] * 5,
                "batch_idx": [0, 0, 0, 1, 1],
                "phase": ["forward"] * 5,
                "perf_status": ["ok"] * 5,
                "perf_instructions": [10, 20, 30, 5, 5],
                "perf_cycles": [20, 30, 40, np.nan, 12],
                "perf_cycles_enabled_pct": [50.0] * 5,
                "perf_cycles_runtime_pct": [1_000_000] * 5,
            }
        )

    def test_partial_batch_and_missing_counter_are_safe(self) -> None:
        observations, diagnostics = build_forward_observations(
            self.frame(),
            epoch=0,
            phase="forward",
            instruction_column="perf_instructions",
            counter_column="perf_cycles",
            partial={(0, 1)},
            include_partial=False,
            pmu_scaling="auto",
        )
        self.assertEqual(len(observations), 3)
        self.assertEqual(diagnostics["excluded_partial_rows"], 2)
        self.assertFalse(bool(diagnostics["scaling"]["applied"]))
        np.testing.assert_allclose(observations["b"].iloc[-1], 1.0)

    def test_empty_epoch_returns_empty_observations(self) -> None:
        observations, diagnostics = build_forward_observations(
            self.frame(),
            epoch=9,
            phase="forward",
            instruction_column="perf_instructions",
            counter_column="perf_cycles",
            partial=set(),
            include_partial=False,
            pmu_scaling="auto",
        )
        self.assertTrue(observations.empty)
        self.assertEqual(diagnostics["used_interval_observations"], 0)

    def test_missing_counter_alias_is_reported_without_exception(self) -> None:
        resolved = resolve_counter_columns(["epoch", "perf_instructions"])
        self.assertEqual(resolved["instructions"], "perf_instructions")
        self.assertIsNone(resolved["l2d_refill"])


if __name__ == "__main__":
    unittest.main()
