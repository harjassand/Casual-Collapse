import json
import os
from typing import Any, Dict, List

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LinearRegression

from envs import HMMCausalEnv, MechanismShiftEnv, ObjectMicroWorldEnv
from models.causal_collapse_model import CausalCollapseModel
from utils.metrics import active_code_count, code_perplexity


def make_env(cfg: Dict[str, Any]):
    env_type = cfg["type"]
    if env_type == "hmm":
        return HMMCausalEnv(
            num_states=cfg["num_states"],
            num_actions=cfg["num_actions"],
            obs_dim=cfg["obs_dim"],
            spurious_noise=cfg["spurious_noise"],
            transition_noise=cfg["transition_noise"],
        )
    if env_type == "object":
        return ObjectMicroWorldEnv(
            num_objects=cfg["num_objects"],
            obs_mode=cfg["obs_mode"],
            image_size=cfg["image_size"],
            spurious_noise=cfg["spurious_noise"],
            dt=cfg["dt"],
        )
    if env_type == "mechanism":
        return MechanismShiftEnv(
            elasticity_env0=cfg["elasticity_env0"],
            elasticity_env1=cfg["elasticity_env1"],
            dt=cfg["dt"],
        )
    raise ValueError(f"Unknown env type {env_type}")


def format_obs(obs: np.ndarray) -> np.ndarray:
    if obs.ndim == 1:
        return obs[None, :]
    return obs


def collect_rollouts(env, model, env_id: int, steps: int, horizon: int, device: torch.device):
    obs = env.reset(seed=None, env_id=env_id)
    obs = format_obs(obs)
    preds = []
    trues = []
    codes = []
    for _ in range(steps):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        out = model(obs_t, None, horizon)
        pred = out["preds"].detach().cpu().numpy()[0]
        preds.append(pred)

        next_obs, _, _, info = env.step(None)
        next_obs = format_obs(next_obs)
        trues.append(next_obs)

        code = out["vq_stats"]["indices"].detach().cpu().numpy()[0]
        codes.append(code)
        obs = next_obs
    return np.array(preds), np.array(trues), np.array(codes)


def linear_probe_drift(codes: np.ndarray, labels: np.ndarray) -> np.ndarray:
    # codes: [N, K], labels: [N]
    one_hot = np.eye(np.max(codes) + 1)[codes.reshape(-1)].reshape(codes.shape[0], -1)
    reg = LinearRegression().fit(one_hot, labels)
    return reg.coef_


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = cfg_dict["model"]
    model = CausalCollapseModel(model_cfg).to(device)
    ckpt = torch.load(cfg_dict["eval"]["ckpt_path"], map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    env_cfg = cfg_dict["env"]
    env = make_env(env_cfg)

    metrics: Dict[str, Any] = {}
    in_envs = env_cfg["train_env_ids"]
    ood_envs = env_cfg["test_env_ids"]

    rollouts_dump = {"preds": [], "trues": [], "codes": [], "env_ids": [], "intervention_id": []}

    def eval_envs(env_ids: List[int], prefix: str) -> None:
        mse_list = []
        code_counts = []
        for env_id in env_ids:
            preds, trues, codes = collect_rollouts(
                env, model, env_id, cfg_dict["eval"]["steps"], cfg_dict["train"]["horizon"], device
            )
            mse = float(np.mean((preds - trues) ** 2))
            mse_list.append(mse)
            code_counts.append(codes.reshape(-1))
            if cfg_dict["eval"]["save_rollouts"]:
                rollouts_dump["preds"].append(preds)
                rollouts_dump["trues"].append(trues)
                rollouts_dump["codes"].append(codes)
                rollouts_dump["env_ids"].append(np.full(preds.shape[0], env_id))
                rollouts_dump["intervention_id"].append(np.full(preds.shape[0], -1))
        all_codes = np.concatenate(code_counts)
        counts = np.bincount(all_codes, minlength=model_cfg["num_codes"])
        metrics[f"{prefix}/mse"] = float(np.mean(mse_list))
        metrics[f"{prefix}/perplexity"] = code_perplexity(counts)
        metrics[f"{prefix}/active_codes"] = active_code_count(counts, cfg_dict["eval"]["active_code_threshold"])

    eval_envs(in_envs, "in")
    eval_envs(ood_envs, "ood")

    # Interventions
    interventional_scores = []
    for idx, spec in enumerate(cfg_dict["eval"]["interventions"]):
        env.do_intervention(spec)
        preds, trues, _ = collect_rollouts(
            env, model, env_cfg["test_env_ids"][0], cfg_dict["eval"]["steps"], cfg_dict["train"]["horizon"], device
        )
        interventional_scores.append(float(np.mean((preds - trues) ** 2)))
        if cfg_dict["eval"]["save_rollouts"]:
            rollouts_dump["preds"].append(preds)
            rollouts_dump["trues"].append(trues)
            rollouts_dump["codes"].append(np.zeros((preds.shape[0], model_cfg["num_slots"]), dtype=np.int64))
            rollouts_dump["env_ids"].append(np.full(preds.shape[0], env_cfg["test_env_ids"][0]))
            rollouts_dump["intervention_id"].append(np.full(preds.shape[0], idx))
    metrics["interventional/mse"] = float(np.mean(interventional_scores)) if interventional_scores else 0.0

    # Invariance probe drift
    drift = []
    for env_id in in_envs:
        _, _, codes = collect_rollouts(
            env, model, env_id, cfg_dict["eval"]["steps"], cfg_dict["train"]["horizon"], device
        )
        labels = np.repeat(env_id, codes.shape[0])
        drift.append(linear_probe_drift(codes.reshape(codes.shape[0], -1), labels))
    if drift:
        drift = np.stack(drift, axis=0)
        mean_w = drift.mean(axis=0)
        metrics["invariance/probe_drift"] = float(np.mean(np.linalg.norm(drift - mean_w, axis=1)))

    out_path = cfg_dict["eval"]["output_path"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if cfg_dict["eval"]["save_rollouts"]:
        rollouts_path = cfg_dict["eval"]["rollouts_path"]
        os.makedirs(os.path.dirname(rollouts_path), exist_ok=True)
        np.savez(
            rollouts_path,
            preds=np.concatenate(rollouts_dump["preds"], axis=0),
            trues=np.concatenate(rollouts_dump["trues"], axis=0),
            codes=np.concatenate(rollouts_dump["codes"], axis=0),
            env_ids=np.concatenate(rollouts_dump["env_ids"], axis=0),
            intervention_id=np.concatenate(rollouts_dump["intervention_id"], axis=0),
        )


if __name__ == "__main__":
    main()
