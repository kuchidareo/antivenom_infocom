import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


PHASES = {
    "idle",
    "dataloading",
    "forward",
    "backward",
    "optimizer_step",
    "evaluation",
    "finished",
}


@dataclass
class TrainingState:
    round: Any = ""
    epoch: Any = ""
    batch_idx: Any = ""
    phase: str = "idle"
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(
        self,
        *,
        round: Any = None,
        epoch: Any = None,
        batch_idx: Any = None,
        phase: Optional[str] = None,
    ) -> None:
        if phase is not None and phase not in PHASES:
            raise ValueError(f"Invalid phase {phase}. Expected one of {sorted(PHASES)}")
        with self._lock:
            if round is not None:
                self.round = round
            if epoch is not None:
                self.epoch = epoch
            if batch_idx is not None:
                self.batch_idx = batch_idx
            if phase is not None:
                self.phase = phase

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "round": self.round,
                "epoch": self.epoch,
                "batch_idx": self.batch_idx,
                "phase": self.phase,
            }
