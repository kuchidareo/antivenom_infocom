"""Compatibility wrapper for the repository-wide dataset preparation module."""

import runpy
from pathlib import Path

_SHARED_PATH = Path(__file__).resolve().parents[1] / "dataset_preparation.py"
_SHARED_NAMESPACE = runpy.run_path(
    str(_SHARED_PATH),
    run_name=f"{__name__}._shared_dataset_preparation",
)
globals().update(
    {
        name: value
        for name, value in _SHARED_NAMESPACE.items()
        if name not in {"__name__", "__file__", "__package__", "__spec__"}
    }
)

if __name__ == "__main__":
    _SHARED_NAMESPACE["main"]()

