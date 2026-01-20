from typing import Any, Dict, Optional, Tuple

import numpy as np

from envs.base import BaseEnv


class HMMCausalEnv(BaseEnv):
    def __init__(
        self,
        num_states: int = 4,
        num_actions: int = 2,
        obs_dim: Optional[int] = None,
        spurious_noise: float = 0.1,
        transition_noise: float = 0.05,
        spurious_flip_on_odd: bool = True,
    ) -> None:
        self.num_states = num_states
        self.num_actions = num_actions
        self.obs_dim = obs_dim or (num_states + 1)
        self.spurious_noise = spurious_noise
        self.transition_noise = transition_noise
        self.spurious_flip_on_odd = spurious_flip_on_odd
        self.rng = np.random.default_rng()
        self.env_id = 0
        self.latent_state = 0
        self.intervention: Dict[str, Any] = {}
        self.intervention_active = False
        self.t = 0
        self.spurious_key = "spurious_bit"
        self.causal_key = "latent_state"
        self.spurious_value = 0
        self.transition = self._init_transition()

    def _init_transition(self) -> np.ndarray:
        base = np.full((self.num_states, self.num_states),
                       self.transition_noise / (self.num_states - 1))
        np.fill_diagonal(base, 1.0 - self.transition_noise)
        transitions = np.stack([np.roll(base, shift=a, axis=1) for a in range(self.num_actions)], axis=0)
        return transitions

    def reset(self, seed: Optional[int] = None, env_id: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if env_id is not None:
            self.env_id = env_id
        self.latent_state = int(self.rng.integers(self.num_states))
        self.intervention = {}
        self.intervention_active = False
        self.t = 0
        return self._observe()

    def do_intervention(self, spec: Dict[str, Any]) -> None:
        # Example specs: {"set_latent": 2}, {"set_spurious": 1}, {"duration": 1}
        spec = dict(spec)
        duration = spec.get("duration")
        if "type" in spec:
            if spec["type"] in ("set_z", "set_latent"):
                spec = {"set_latent": spec.get("value", spec.get("latent", 0))}
            elif spec["type"] in ("set_spurious", "set_s"):
                spec = {"set_spurious": spec.get("value", 0)}
            if duration is not None:
                spec["duration"] = duration
        self.intervention = spec
        self.intervention_active = bool(spec)

    def step(self, action: Optional[np.ndarray]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self.t += 1
        if action is None:
            action_idx = 0
        elif isinstance(action, np.ndarray):
            action_idx = int(action) if action.size == 1 else int(action[0])
        else:
            action_idx = int(action)
        action_idx = int(np.clip(action_idx, 0, self.num_actions - 1))

        if "set_latent" in self.intervention:
            self.latent_state = int(self.intervention["set_latent"])
        else:
            probs = self.transition[action_idx, self.latent_state]
            self.latent_state = int(self.rng.choice(self.num_states, p=probs))

        obs = self._observe()
        reward = 0.0
        done = False
        info = {
            "t": self.t,
            "latent_state": self.latent_state,
            "env_id": self.env_id,
            "spurious_key": self.spurious_key,
            "spurious_value": int(self.spurious_value),
            "causal_key": self.causal_key,
            "causal_value": int(self.latent_state),
            "intervention_active": self.intervention_active,
            "intervention_spec": self.intervention if self.intervention_active else {},
        }
        if self.intervention.get("duration", 0) == 1:
            self.intervention = {}
            self.intervention_active = False
        return obs, reward, done, info

    def _observe(self) -> np.ndarray:
        signal = np.zeros(self.num_states, dtype=np.float32)
        signal[self.latent_state] = 1.0
        noise = self.rng.normal(0.0, 0.05, size=signal.shape).astype(np.float32)
        signal = signal + noise

        spur = int(self.latent_state % 2)
        if self.rng.random() < self.spurious_noise:
            spur ^= 1
        if self.spurious_flip_on_odd and self.env_id % 2 == 1:
            spur = 1 - spur
        if "set_spurious" in self.intervention:
            spur = int(self.intervention["set_spurious"])
        self.spurious_value = spur

        obs = np.zeros(self.obs_dim, dtype=np.float32)
        dims = min(self.num_states, self.obs_dim - 1)
        obs[:dims] = signal[:dims]
        obs[-1] = float(spur)
        if self.obs_dim > self.num_states + 1:
            extra = self.rng.normal(0.0, 0.1, size=(self.obs_dim - self.num_states - 1,)).astype(np.float32)
            obs[self.num_states:-1] = extra
        return obs

    def get_ground_truth(self) -> Dict[str, Any]:
        return {
            "t": self.t,
            "env_id": self.env_id,
            "latent_state": int(self.latent_state),
            "spurious_key": self.spurious_key,
            "spurious_value": int(self.spurious_value),
            "causal_key": self.causal_key,
            "causal_value": int(self.latent_state),
            "intervention_active": self.intervention_active,
            "intervention_spec": self.intervention if self.intervention_active else {},
        }
