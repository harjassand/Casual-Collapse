from typing import Dict, List, Optional

import numpy as np


class RolloutBuffer:
    def __init__(self, max_size: int = 10000) -> None:
        self.max_size = max_size
        self.obs: List[np.ndarray] = []
        self.actions: List[Optional[np.ndarray]] = []
        self.future_obs: List[np.ndarray] = []
        self.future_actions: List[Optional[np.ndarray]] = []
        self.env_ids: List[int] = []
        self.futures: List[int] = []

    def add(
        self,
        obs: np.ndarray,
        action: Optional[np.ndarray],
        future_obs: np.ndarray,
        future_actions: Optional[np.ndarray],
        env_id: int,
        future_label: int,
    ) -> None:
        if len(self.obs) >= self.max_size:
            self.obs.pop(0)
            self.actions.pop(0)
            self.future_obs.pop(0)
            self.future_actions.pop(0)
            self.env_ids.pop(0)
            self.futures.pop(0)
        self.obs.append(obs)
        self.actions.append(action)
        self.future_obs.append(future_obs)
        self.future_actions.append(future_actions)
        self.env_ids.append(env_id)
        self.futures.append(future_label)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idx = np.random.choice(len(self.obs), size=batch_size, replace=False)
        obs = np.stack([self.obs[i] for i in idx], axis=0)
        future_obs = np.stack([self.future_obs[i] for i in idx], axis=0)
        actions = None
        if self.actions[0] is not None:
            actions = np.stack([self.actions[i] for i in idx], axis=0)
        future_actions = None
        if self.future_actions[0] is not None:
            future_actions = np.stack([self.future_actions[i] for i in idx], axis=0)
        env_ids = np.array([self.env_ids[i] for i in idx], dtype=np.int64)
        futures = np.array([self.futures[i] for i in idx], dtype=np.int64)
        return {
            "obs": obs,
            "actions": actions,
            "future_obs": future_obs,
            "future_actions": future_actions,
            "env_ids": env_ids,
            "futures": futures,
        }

    def __len__(self) -> int:
        return len(self.obs)
