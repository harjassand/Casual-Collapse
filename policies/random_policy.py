from typing import Optional

import numpy as np


class RandomPolicy:
    def __init__(self, action_dim: int = 0, action_scale: float = 0.1) -> None:
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.rng = np.random.default_rng()

    def act(self, obs: np.ndarray) -> Optional[np.ndarray]:
        if self.action_dim == 0:
            return None
        return self.rng.normal(0.0, self.action_scale, size=(self.action_dim,)).astype(np.float32)
