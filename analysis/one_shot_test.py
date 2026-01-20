import argparse
import json
import os
import sys
from typing import Any, Dict

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs import HMMCausalEnv, MechanismShiftEnv, ObjectMicroWorldEnv
from models.causal_collapse_model import CausalCollapseModel


def format_obs(obs: np.ndarray) -> np.ndarray:
    if obs.ndim == 1:
        return obs[None, :]
    return obs


def make_env(cfg: Dict[str, Any]):
    env_type = cfg["type"]
    if env_type == "hmm":
        return HMMCausalEnv(
            num_states=cfg["num_states"],
            num_actions=cfg["num_actions"],
            obs_dim=cfg["obs_dim"],
            spurious_noise=cfg["spurious_noise"],
            transition_noise=cfg["transition_noise"],
            spurious_flip_on_odd=cfg.get("spurious_flip_on_odd", True),
        )
    if env_type == "object":
        return ObjectMicroWorldEnv(
            num_objects=cfg["num_objects"],
            obs_mode=cfg["obs_mode"],
            image_size=cfg["image_size"],
            spurious_noise=cfg["spurious_noise"],
            dt=cfg["dt"],
            spurious_flip_on_odd=cfg.get("spurious_flip_on_odd", True),
        )
    if env_type == "mechanism":
        return MechanismShiftEnv(
            elasticity_env0=cfg["elasticity_env0"],
            elasticity_env1=cfg["elasticity_env1"],
            dt=cfg["dt"],
        )
    raise ValueError(f"Unknown env type {env_type}")


def one_shot_spec(env_type: str) -> Dict[str, Any]:
    if env_type == "hmm":
        return {
            "spec": {"set_latent": 0, "duration": 1},
            "targeted_mechanism": "latent_transition",
            "expected_change": "latent clamp changes immediate future observations",
        }
    if env_type == "object":
        return {
            "spec": {"set_vel": {"obj": 0, "value": [0.6, 0.0]}, "duration": 1},
            "targeted_mechanism": "object_velocity",
            "expected_change": "velocity intervention shifts next-step position",
        }
    return {
        "spec": {"set_vel": {"obj": 0, "value": 1.0}, "duration": 1},
        "targeted_mechanism": "collision_elasticity",
        "expected_change": "velocity intervention reveals collision elasticity via altered post-collision velocities",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="analysis/one_shot_result.json")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalCollapseModel(cfg["model"]).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    env_cfg: Dict[str, Any] = cfg["env"]
    env = make_env(env_cfg)

    env_id = env_cfg["test_env_ids"][0]
    obs = format_obs(env.reset(seed=cfg["seed"], env_id=env_id))

    shot = one_shot_spec(env_cfg["type"])
    env.do_intervention(shot["spec"])
    next_obs, _, _, _ = env.step(None)
    next_obs = format_obs(next_obs)

    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    next_t = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        out = model(obs_t, None, horizon=1)
        pred = out["preds"][:, 0]
        loss_before = torch.mean((pred - next_t) ** 2).item()
        pred_before = pred.cpu().numpy().tolist()

    # One-step update on dynamics to fit intervention
    model.train()
    optimizer = torch.optim.SGD(model.dynamics.parameters(), lr=cfg["eval"].get("one_shot_lr", 1e-2))
    out = model(obs_t, None, horizon=1)
    pred = out["preds"][:, 0]
    loss = torch.mean((pred - next_t) ** 2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(obs_t, None, horizon=1)
        pred = out["preds"][:, 0]
        loss_after = torch.mean((pred - next_t) ** 2).item()
        pred_after = pred.cpu().numpy().tolist()

    result = {
        "intervention": shot["spec"],
        "targeted_mechanism": shot["targeted_mechanism"],
        "expected_change": shot["expected_change"],
        "seed": int(cfg["seed"]),
        "env_id": int(env_id),
        "env_type": env_cfg["type"],
        "loss_before": loss_before,
        "loss_after": loss_after,
        "loss_delta": loss_before - loss_after,
        "pred_before": pred_before,
        "pred_after": pred_after,
        "true_next": next_obs.tolist(),
    }

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
