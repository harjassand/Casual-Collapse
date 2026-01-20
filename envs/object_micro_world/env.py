from typing import Any, Dict, Optional, Tuple

import numpy as np

from envs.base import BaseEnv


class ObjectMicroWorldEnv(BaseEnv):
    def __init__(
        self,
        num_objects: int = 3,
        obs_mode: str = "structured",
        image_size: int = 32,
        spurious_noise: float = 0.1,
        dt: float = 0.1,
        spurious_flip_on_odd: bool = True,
    ) -> None:
        self.num_objects = num_objects
        self.obs_mode = obs_mode
        self.image_size = image_size
        self.spurious_noise = spurious_noise
        self.dt = dt
        self.spurious_flip_on_odd = spurious_flip_on_odd
        self.rng = np.random.default_rng()
        self.env_id = 0
        self.state = None
        self.background = 0.0
        self.intervention: Dict[str, Any] = {}
        self.intervention_active = False
        self.t = 0
        self.spurious_key = "background"
        self.causal_key = "objects"

    def reset(self, seed: Optional[int] = None, env_id: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if env_id is not None:
            self.env_id = env_id
        positions = self.rng.uniform(0.2, 0.8, size=(self.num_objects, 2)).astype(np.float32)
        velocities = self.rng.normal(0.0, 0.2, size=(self.num_objects, 2)).astype(np.float32)
        colors = self.rng.integers(0, 3, size=(self.num_objects,)).astype(np.int64)
        self.state = {"pos": positions, "vel": velocities, "color": colors}
        self.background = float(self._background_from_event())
        self.intervention = {}
        self.intervention_active = False
        self.t = 0
        return self._observe()

    def do_intervention(self, spec: Dict[str, Any]) -> None:
        # Example specs: {"set_pos": {"obj": 0, "value": [0.5, 0.5]}},
        # {"set_vel": {"obj": 1, "value": [0.0, 0.3]}}, {"set_background": 1.0}
        spec = dict(spec)
        duration = spec.get("duration")
        if "type" in spec:
            if spec["type"] == "set_pos":
                spec = {"set_pos": spec.get("value", {})}
            elif spec["type"] == "set_vel":
                spec = {"set_vel": spec.get("value", {})}
            elif spec["type"] == "set_background":
                spec = {"set_background": spec.get("value", 0.0)}
            if duration is not None:
                spec["duration"] = duration
        self.intervention = spec
        self.intervention_active = bool(spec)

    def step(self, action: Optional[np.ndarray]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self.t += 1
        if self.intervention.get("set_pos"):
            obj = int(self.intervention["set_pos"]["obj"])
            self.state["pos"][obj] = np.array(self.intervention["set_pos"]["value"], dtype=np.float32)
        if self.intervention.get("set_vel"):
            obj = int(self.intervention["set_vel"]["obj"])
            self.state["vel"][obj] = np.array(self.intervention["set_vel"]["value"], dtype=np.float32)
        if action is not None:
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            if action.size >= 2:
                self.state["vel"][0] = self.state["vel"][0] + action[:2]

        self.state["pos"] = self.state["pos"] + self.state["vel"] * self.dt
        self._handle_bounds()

        if "set_background" in self.intervention:
            self.background = float(self.intervention["set_background"])
        else:
            self.background = float(self._background_from_event())

        obs = self._observe()
        reward = 0.0
        done = False
        info = {
            "t": self.t,
            "latent_state": {
                "pos": self.state["pos"].copy(),
                "vel": self.state["vel"].copy(),
                "color": self.state["color"].copy(),
            },
            "env_id": self.env_id,
            "background": self.background,
            "event": int(self._event_label()),
            "spurious_key": self.spurious_key,
            "spurious_value": float(self.background),
            "causal_key": self.causal_key,
            "causal_value": {
                "pos": self.state["pos"].copy(),
                "vel": self.state["vel"].copy(),
                "color": self.state["color"].copy(),
            },
            "intervention_active": self.intervention_active,
            "intervention_spec": self.intervention if self.intervention_active else {},
        }
        if self.intervention.get("duration", 0) == 1:
            self.intervention = {}
            self.intervention_active = False
        return obs, reward, done, info

    def _handle_bounds(self) -> None:
        pos = self.state["pos"]
        vel = self.state["vel"]
        for i in range(self.num_objects):
            for d in range(2):
                if pos[i, d] < 0.0:
                    pos[i, d] = 0.0
                    vel[i, d] *= -0.8
                if pos[i, d] > 1.0:
                    pos[i, d] = 1.0
                    vel[i, d] *= -0.8
        self.state["pos"] = pos
        self.state["vel"] = vel

    def _event_label(self) -> bool:
        pos = self.state["pos"][0]
        vel = self.state["vel"][0]
        future_pos = pos + vel * (self.dt * 5)
        return bool(future_pos[0] > 0.8)

    def _background_from_event(self) -> float:
        event = self._event_label()
        spur = int(event)
        if self.rng.random() < self.spurious_noise:
            spur ^= 1
        if self.spurious_flip_on_odd and self.env_id % 2 == 1:
            spur = 1 - spur
        return float(spur)

    def _observe(self) -> np.ndarray:
        if self.obs_mode == "image":
            return self._render_image()
        return self._structured_obs()

    def _structured_obs(self) -> np.ndarray:
        pos = self.state["pos"]
        vel = self.state["vel"]
        color = self.state["color"].astype(np.float32)[:, None] / 2.0
        background = np.full((self.num_objects, 1), self.background, dtype=np.float32)
        obs = np.concatenate([pos, vel, color, background], axis=1)
        return obs.astype(np.float32)

    def _render_image(self) -> np.ndarray:
        size = self.image_size
        image = np.ones((size, size, 3), dtype=np.float32)
        if self.background > 0.5:
            image *= np.array([0.2, 0.2, 0.6], dtype=np.float32)
        else:
            image *= np.array([0.6, 0.6, 0.2], dtype=np.float32)
        for i in range(self.num_objects):
            x, y = self.state["pos"][i]
            cx = int(x * (size - 1))
            cy = int(y * (size - 1))
            color = np.array([
                1.0 if self.state["color"][i] == 0 else 0.2,
                1.0 if self.state["color"][i] == 1 else 0.2,
                1.0 if self.state["color"][i] == 2 else 0.2,
            ], dtype=np.float32)
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    px = np.clip(cx + dx, 0, size - 1)
                    py = np.clip(cy + dy, 0, size - 1)
                    image[py, px] = color
        return image.transpose(2, 0, 1)

    def get_ground_truth(self) -> Dict[str, Any]:
        return {
            "t": self.t,
            "env_id": self.env_id,
            "latent_state": {
                "pos": self.state["pos"].copy(),
                "vel": self.state["vel"].copy(),
                "color": self.state["color"].copy(),
            },
            "spurious_key": self.spurious_key,
            "spurious_value": float(self.background),
            "causal_key": self.causal_key,
            "causal_value": {
                "pos": self.state["pos"].copy(),
                "vel": self.state["vel"].copy(),
                "color": self.state["color"].copy(),
            },
            "intervention_active": self.intervention_active,
            "intervention_spec": self.intervention if self.intervention_active else {},
        }
