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
        }
        row.update(self.condition)
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
