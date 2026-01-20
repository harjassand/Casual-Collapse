from typing import Dict, List, Optional, Tuple

import numpy as np


class StatsBuffer:
    def __init__(self, num_codes: int, num_envs: int, max_per_code: int = 512) -> None:
        self.num_codes = num_codes
        self.num_envs = num_envs
        self.max_per_code = max_per_code
        self.counts = np.zeros(num_codes, dtype=np.int64)
        self.env_counts = np.zeros((num_envs, num_codes), dtype=np.int64)
        self.future_counts: Dict[int, Dict[int, int]] = {c: {} for c in range(num_codes)}
        self.future_counts_env: Dict[Tuple[int, int], Dict[int, int]] = {}
        self.embeddings: Dict[int, List[np.ndarray]] = {c: [] for c in range(num_codes)}
        self.embedding_labels: Dict[int, List[int]] = {c: [] for c in range(num_codes)}

    def update(self, codes: np.ndarray, futures: np.ndarray, env_ids: np.ndarray, embeddings: Optional[np.ndarray] = None) -> None:
        # codes: [N], futures: [N] discrete labels, env_ids: [N]
        for idx, code in enumerate(codes):
            code = int(code)
            future = int(futures[idx])
            env_id = int(env_ids[idx])
            self.counts[code] += 1
            if env_id < self.num_envs:
                self.env_counts[env_id, code] += 1

            self.future_counts[code][future] = self.future_counts[code].get(future, 0) + 1
            key = (env_id, code)
            if key not in self.future_counts_env:
                self.future_counts_env[key] = {}
            self.future_counts_env[key][future] = self.future_counts_env[key].get(future, 0) + 1

            if embeddings is not None:
                if len(self.embeddings[code]) < self.max_per_code:
                    self.embeddings[code].append(embeddings[idx])
                    self.embedding_labels[code].append(future)

    def distribution(self, code: int, num_classes: int) -> np.ndarray:
        counts = np.zeros(num_classes, dtype=np.float64)
        for k, v in self.future_counts[code].items():
            if k < num_classes:
                counts[k] = v
        if counts.sum() == 0:
            return np.ones(num_classes, dtype=np.float64) / num_classes
        return counts / counts.sum()

    def distribution_env(self, env_id: int, code: int, num_classes: int) -> np.ndarray:
        counts = np.zeros(num_classes, dtype=np.float64)
        key = (env_id, code)
        if key in self.future_counts_env:
            for k, v in self.future_counts_env[key].items():
                if k < num_classes:
                    counts[k] = v
        if counts.sum() == 0:
            return np.ones(num_classes, dtype=np.float64) / num_classes
        return counts / counts.sum()

    def code_embeddings(self, code: int) -> np.ndarray:
        if len(self.embeddings[code]) == 0:
            return np.zeros((0,))
        return np.stack(self.embeddings[code], axis=0)

    def code_labels(self, code: int) -> np.ndarray:
        if len(self.embedding_labels[code]) == 0:
            return np.zeros((0,), dtype=np.int64)
        return np.array(self.embedding_labels[code], dtype=np.int64)

    def reset(self) -> None:
        self.counts[:] = 0
        self.env_counts[:] = 0
        self.future_counts = {c: {} for c in range(self.num_codes)}
        self.future_counts_env = {}
        self.embeddings = {c: [] for c in range(self.num_codes)}
        self.embedding_labels = {c: [] for c in range(self.num_codes)}
