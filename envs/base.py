from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np


class BaseEnv(ABC):
    @abstractmethod
    def reset(self, seed: Optional[int] = None, env_id: Optional[int] = None) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Optional[np.ndarray]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def do_intervention(self, spec: Dict[str, Any]) -> None:
        raise NotImplementedError
