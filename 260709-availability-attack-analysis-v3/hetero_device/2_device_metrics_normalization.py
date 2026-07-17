#!/usr/bin/env python3
"""Device-metric normalization helpers for hetero-device analysis.

This file intentionally keeps the reference/test split explicit:
- reference_* clean runs define the benign jitter baseline.
- trial_* clean and poisoning runs are evaluation targets.

Do not use trial_* clean rows to fit a benign baseline; that leaks clean test
data into the detector.
"""

from __future__ import annotations

from typing import Iterable, List


DEFAULT_REFERENCE_BASELINE_TRIAL_IDS = tuple(f"reference_{idx}" for idx in range(5))


def parse_trial_id_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def is_reference_baseline_trial(trial_id: object, reference_trial_ids: Iterable[str] | None = None) -> bool:
    references = set(reference_trial_ids or DEFAULT_REFERENCE_BASELINE_TRIAL_IDS)
    return str(trial_id) in references


def is_evaluation_trial(trial_id: object, reference_trial_ids: Iterable[str] | None = None) -> bool:
    return not is_reference_baseline_trial(trial_id, reference_trial_ids)


def add_reference_split_columns(df, trial_col: str = "target_trial_id", reference_trial_ids: Iterable[str] | None = None):
    """Return a copy of df with baseline/evaluation split columns.

    The function accepts a pandas DataFrame without importing pandas at module
    import time, so scripts can import these helpers in lightweight contexts.
    """
    out = df.copy()
    out["is_reference_baseline"] = out[trial_col].map(
        lambda trial_id: is_reference_baseline_trial(trial_id, reference_trial_ids)
    )
    out["is_evaluation_target"] = ~out["is_reference_baseline"]
    return out
