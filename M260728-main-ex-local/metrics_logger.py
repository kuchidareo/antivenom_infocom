import csv
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from experiment_config import METRIC_COLUMNS


class MetricsLogger:
    def __init__(self, *, path: Union[str, Path], condition: Dict[str, Any]) -> None:
        self.path = Path(path)
        self.condition = dict(condition)
        self._lock = threading.Lock()
        self._initialized = False

    def write(
        self,
        *,
        state: Dict[str, Any],
        metric_event: str,
        metric_split: str,
        loss: Optional[float] = None,
        accuracy: Optional[float] = None,
        num_examples: Optional[float] = None,
        batch_size_actual: Optional[int] = None,
        dataloading_elapsed_ms: Optional[float] = None,
        forward_elapsed_ms: Optional[float] = None,
        backward_elapsed_ms: Optional[float] = None,
        optimizer_step_elapsed_ms: Optional[float] = None,
        batch_elapsed_ms: Optional[float] = None,
        condition_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = datetime.now()
        row = {
            "timestamp": now.isoformat(timespec="microseconds"),
            "timestamp_unix": now.timestamp(),
            "metric_event": metric_event,
            "metric_split": metric_split,
            "loss": "" if loss is None else loss,
            "accuracy": "" if accuracy is None else accuracy,
            "num_examples": "" if num_examples is None else num_examples,
            "batch_size_actual": "" if batch_size_actual is None else batch_size_actual,
            "dataloading_elapsed_ms": (
                "" if dataloading_elapsed_ms is None else dataloading_elapsed_ms
            ),
            "forward_elapsed_ms": "" if forward_elapsed_ms is None else forward_elapsed_ms,
            "backward_elapsed_ms": "" if backward_elapsed_ms is None else backward_elapsed_ms,
            "optimizer_step_elapsed_ms": (
                "" if optimizer_step_elapsed_ms is None else optimizer_step_elapsed_ms
            ),
            "batch_elapsed_ms": "" if batch_elapsed_ms is None else batch_elapsed_ms,
        }
        row.update(self.condition)
        row.update(condition_overrides or {})
        row.update(state)
        for column in METRIC_COLUMNS:
            row.setdefault(column, "")

        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if self._initialized or self.path.exists() else "w"
            with self.path.open(mode, newline="") as f:
                writer = csv.DictWriter(f, fieldnames=METRIC_COLUMNS, extrasaction="ignore")
                if mode == "w":
                    writer.writeheader()
                writer.writerow(row)
            self._initialized = True
