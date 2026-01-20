from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


def default_uncertainty(preds: np.ndarray) -> float:
    # preds: [M, ...]
    return float(np.var(preds, axis=0).mean())


def default_cost(action: np.ndarray) -> float:
    return float(np.linalg.norm(action))


class ActiveInfoGainPolicy:
    def __init__(
        self,
        action_dim: int,
        num_candidates: int = 16,
        action_scale: float = 0.3,
        cost_weight: float = 0.0,
        uncertainty_fn: Optional[Callable[[np.ndarray], float]] = None,
        cost_fn: Optional[Callable[[np.ndarray], float]] = None,
    ) -> None:
        self.action_dim = action_dim
        self.num_candidates = num_candidates
        self.action_scale = action_scale
        self.cost_weight = cost_weight
        self.uncertainty_fn = uncertainty_fn or default_uncertainty
        self.cost_fn = cost_fn or default_cost
        self.rng = np.random.default_rng()

    def act(self, obs: np.ndarray, predict_fn: Callable[[np.ndarray], np.ndarray]) -> Tuple[np.ndarray, Dict[str, Any]]:
        best_action = None
        best_score = -1e9
        best_uncertainty = 0.0
        for _ in range(self.num_candidates):
            action = self.rng.normal(0.0, self.action_scale, size=(self.action_dim,)).astype(np.float32)
            preds = predict_fn(action)
            uncertainty = self.uncertainty_fn(preds)
            score = uncertainty - self.cost_weight * self.cost_fn(action)
            if score > best_score:
                best_score = score
                best_action = action
                best_uncertainty = uncertainty
        info = {"expected_gain": best_uncertainty, "score": best_score}
        return best_action, info
