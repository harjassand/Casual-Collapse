import json
import os
from typing import Any, Dict

from torch.utils.tensorboard import SummaryWriter


class RunLogger:
    def __init__(self, run_dir: str) -> None:
        os.makedirs(run_dir, exist_ok=True)
        self.run_dir = run_dir
        self.metrics_path = os.path.join(run_dir, "metrics.jsonl")
        self._writer = SummaryWriter(log_dir=run_dir)

    def log(self, step: int, metrics: Dict[str, Any]) -> None:
        record = {"step": step}
        record.update(metrics)
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self._writer.add_scalar(key, value, step)

    def close(self) -> None:
        self._writer.flush()
        self._writer.close()
