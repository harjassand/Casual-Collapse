from typing import Any, Dict, Optional, Tuple

import numpy as np

from envs.base import BaseEnv


class MechanismShiftEnv(BaseEnv):
    def __init__(
        self,
        elasticity_env0: float = 0.9,
        elasticity_env1: float = 0.2,
        dt: float = 0.1,
    ) -> None:
        self.elasticity_env0 = elasticity_env0
        self.elasticity_env1 = elasticity_env1
        self.dt = dt
        self.rng = np.random.default_rng()
        self.env_id = 0
        self.state = None
        self.intervention: Dict[str, Any] = {}
        self.intervention_active = False
        self.t = 0
        self.spurious_key = "none"
        self.causal_key = "elasticity"

    def reset(self, seed: Optional[int] = None, env_id: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if env_id is not None:
            self.env_id = env_id
        positions = self.rng.uniform(0.2, 0.8, size=(2,)).astype(np.float32)
        velocities = self.rng.normal(0.0, 0.2, size=(2,)).astype(np.float32)
        self.state = {"pos": positions, "vel": velocities}
        self.intervention = {}
        self.intervention_active = False
        self.t = 0
        return self._observe()

    def do_intervention(self, spec: Dict[str, Any]) -> None:
        spec = dict(spec)
        duration = spec.get("duration")
        if "type" in spec:
            if spec["type"] == "set_pos":
                spec = {"set_pos": spec.get("value", {})}
            elif spec["type"] == "set_vel":
                spec = {"set_vel": spec.get("value", {})}
            if duration is not None:
                spec["duration"] = duration
        self.intervention = spec
        self.intervention_active = bool(spec)

    def step(self, action: Optional[np.ndarray]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self.t += 1
        if self.intervention.get("set_pos"):
            idx = int(self.intervention["set_pos"]["obj"])
            self.state["pos"][idx] = float(self.intervention["set_pos"]["value"])
        if self.intervention.get("set_vel"):
            idx = int(self.intervention["set_vel"]["obj"])
            self.state["vel"][idx] = float(self.intervention["set_vel"]["value"])
        if action is not None:
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            if action.size >= 2:
                self.state["vel"] += action[:2]

        self.state["pos"] = self.state["pos"] + self.state["vel"] * self.dt
        self._handle_bounds()
        self._handle_collision()

        obs = self._observe()
        reward = 0.0
        done = False
        info = {
            "t": self.t,
            "latent_state": {
                "pos": self.state["pos"].copy(),
                "vel": self.state["vel"].copy(),
            },
            "env_id": self.env_id,
            "elasticity": self._elasticity(),
            "spurious_key": self.spurious_key,
            "spurious_value": 0.0,
            "causal_key": self.causal_key,
            "causal_value": float(self._elasticity()),
            "intervention_active": self.intervention_active,
            "intervention_spec": self.intervention if self.intervention_active else {},
        }
        if self.intervention.get("duration", 0) == 1:
            self.intervention = {}
            self.intervention_active = False
        return obs, reward, done, info

    def _elasticity(self) -> float:
        return self.elasticity_env0 if self.env_id % 2 == 0 else self.elasticity_env1

    def _handle_bounds(self) -> None:
        for i in range(2):
            if self.state["pos"][i] < 0.0:
                self.state["pos"][i] = 0.0
                self.state["vel"][i] *= -0.8
            if self.state["pos"][i] > 1.0:
                self.state["pos"][i] = 1.0
                self.state["vel"][i] *= -0.8

    def _handle_collision(self) -> None:
        pos = self.state["pos"]
        vel = self.state["vel"]
        if abs(pos[0] - pos[1]) < 0.05:
            e = self._elasticity()
            v0, v1 = vel[0], vel[1]
            vel[0] = e * v1
            vel[1] = e * v0
        self.state["vel"] = vel

    def _observe(self) -> np.ndarray:
        return np.concatenate([self.state["pos"], self.state["vel"]], axis=0).astype(np.float32)

    def get_ground_truth(self) -> Dict[str, Any]:
        return {
            "t": self.t,
            "env_id": self.env_id,
            "latent_state": {
                "pos": self.state["pos"].copy(),
                "vel": self.state["vel"].copy(),
            },
            "spurious_key": self.spurious_key,
            "spurious_value": 0.0,
            "causal_key": self.causal_key,
            "causal_value": float(self._elasticity()),
            "intervention_active": self.intervention_active,
            "intervention_spec": self.intervention if self.intervention_active else {},
        }
