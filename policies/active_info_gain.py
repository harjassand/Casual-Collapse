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
        top_k: int = 5,
        uncertainty_fn: Optional[Callable[[np.ndarray], float]] = None,
        cost_fn: Optional[Callable[[np.ndarray], float]] = None,
    ) -> None:
        self.action_dim = action_dim
        self.num_candidates = num_candidates
        self.action_scale = action_scale
        self.cost_weight = cost_weight
        self.top_k = top_k
        self.uncertainty_fn = uncertainty_fn or default_uncertainty
        self.cost_fn = cost_fn or default_cost
        self.rng = np.random.default_rng()

    def act(self, obs: np.ndarray, predict_fn: Callable[[np.ndarray], np.ndarray]) -> Tuple[np.ndarray, Dict[str, Any]]:
        actions = []
        scores = []
        costs = []
        uncertainties = []
        for _ in range(self.num_candidates):
            action = self.rng.normal(0.0, self.action_scale, size=(self.action_dim,)).astype(np.float32)
            preds = predict_fn(action)
            uncertainty = self.uncertainty_fn(preds)
            cost = self.cost_fn(action)
            score = uncertainty - self.cost_weight * cost
            actions.append(action)
            scores.append(score)
            costs.append(cost)
            uncertainties.append(uncertainty)
        best_idx = int(np.argmax(scores))
        best_action = actions[best_idx]
        top_k = min(self.top_k, len(scores))
        top_idx = np.argsort(scores)[-top_k:][::-1]
        info = {
            "expected_gain": float(uncertainties[best_idx]),
            "score": float(scores[best_idx]),
            "cost": float(costs[best_idx]),
            "action": best_action.tolist(),
            "candidate_scores": [float(scores[i]) for i in range(len(scores))],
            "candidate_costs": [float(costs[i]) for i in range(len(costs))],
            "top_scores": [float(scores[i]) for i in top_idx],
            "top_costs": [float(costs[i]) for i in top_idx],
            "top_actions": [actions[i].tolist() for i in top_idx],
        }
        return best_action, info
