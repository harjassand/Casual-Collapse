import argparse
import json
from typing import Any, Dict

import numpy as np
import torch

from envs import MechanismShiftEnv
from models.causal_collapse_model import CausalCollapseModel


def format_obs(obs: np.ndarray) -> np.ndarray:
    if obs.ndim == 1:
        return obs[None, :]
    return obs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="analysis/one_shot.json")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalCollapseModel(cfg["model"]).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    env_cfg: Dict[str, Any] = cfg["env"]
    env = MechanismShiftEnv(
        elasticity_env0=env_cfg.get("elasticity_env0", 0.9),
        elasticity_env1=env_cfg.get("elasticity_env1", 0.2),
        dt=env_cfg.get("dt", 0.1),
    )

    env_id = env_cfg["test_env_ids"][0]
    obs = format_obs(env.reset(seed=cfg["seed"], env_id=env_id))

    # One-shot intervention: set velocity of object 0 to high magnitude
    env.do_intervention({"set_vel": {"obj": 0, "value": 1.0}, "duration": 1})
    next_obs, _, _, _ = env.step(None)
    next_obs = format_obs(next_obs)

    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    next_t = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        out = model(obs_t, None, horizon=1)
        pred = out["preds"][:, 0]
        loss_before = torch.mean((pred - next_t) ** 2).item()

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

    result = {
        "intervention": {"set_vel": {"obj": 0, "value": 1.0}},
        "targeted_mechanism": "collision_elasticity",
        "loss_before": loss_before,
        "loss_after": loss_after,
    }

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
