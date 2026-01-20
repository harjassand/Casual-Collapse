import argparse
import json
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering

from envs import HMMCausalEnv, MechanismShiftEnv, ObjectMicroWorldEnv
from models.causal_collapse_model import CausalCollapseModel
from utils.metrics import clustering_scores, purity_scores, js_divergence


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


def format_obs(obs: np.ndarray) -> np.ndarray:
    if obs.ndim == 1:
        return obs[None, :]
    return obs


def get_future_label(env_type: str, info: Dict[str, Any]) -> int:
    if env_type == "hmm":
        return int(info["latent_state"])
    if env_type == "object":
        return int(info["event"])
    if env_type == "mechanism":
        pos = info["latent_state"]["pos"]
        return int(abs(pos[0] - pos[1]) < 0.1)
    return 0


def set_state(env, env_type: str, state: Any) -> None:
    if env_type == "hmm":
        env.latent_state = int(state)
    elif env_type == "object":
        env.state = {
            "pos": state["pos"].copy(),
            "vel": state["vel"].copy(),
            "color": state["color"].copy(),
        }
    elif env_type == "mechanism":
        env.state = {"pos": state["pos"].copy(), "vel": state["vel"].copy()}


def empirical_future_distribution(env, env_type: str, state: Any, env_id: int, horizon: int, rollouts: int, num_classes: int) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=np.float64)
    for _ in range(rollouts):
        env.env_id = env_id
        set_state(env, env_type, state)
        info = {}
        for _ in range(horizon):
            _, _, _, info = env.step(None)
        label = get_future_label(env_type, info)
        counts[label] += 1
    return counts / counts.sum()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="analysis/alignment_metrics.json")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalCollapseModel(cfg["model"]).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    env_cfg = cfg["env"]
    env = make_env(env_cfg)
    env_id = env_cfg["train_env_ids"][0]
    env.reset(seed=cfg["eval"].get("alignment_seed", cfg["seed"]), env_id=env_id)

    histories = []
    labels = []
    codes = []

    obs = env.reset(seed=cfg["eval"].get("alignment_seed", cfg["seed"]), env_id=env_id)
    obs = format_obs(obs)
    for _ in range(cfg["eval"]["alignment_samples"]):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            out = model.encode(obs_t)
            code_vec = out["vq_stats"]["indices"].cpu().numpy().reshape(-1)
            code = 0
            base = cfg["model"]["num_codes"]
            for idx, val in enumerate(code_vec):
                code += int(val) * (base ** idx)
        next_obs, _, _, info = env.step(None)
        state = info.get("latent_state")
        if env_cfg["type"] == "hmm":
            state = info["latent_state"]
        histories.append(state)
        codes.append(code)
        obs = format_obs(next_obs)

    num_classes = cfg["train"]["num_future_classes"]
    dists = []
    for state in histories:
        dist = empirical_future_distribution(
            env, env_cfg["type"], state, env_id, cfg["train"]["horizon"], cfg["eval"]["alignment_rollouts"], num_classes
        )
        dists.append(dist)
    dists = np.stack(dists, axis=0)

    # Cluster by JS divergence
    n = dists.shape[0]
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist_mat[i, j] = js_divergence(dists[i], dists[j])
            dist_mat[j, i] = dist_mat[i, j]

    num_clusters = cfg["eval"].get("alignment_clusters", min(cfg["model"]["num_codes"], n))
    try:
        clusterer = AgglomerativeClustering(n_clusters=num_clusters, metric="precomputed", linkage="average")
    except TypeError:
        clusterer = AgglomerativeClustering(n_clusters=num_clusters, affinity="precomputed", linkage="average")
    cluster_labels = clusterer.fit_predict(dist_mat)

    scores = clustering_scores(cluster_labels, np.array(codes))
    purity, inv_purity = purity_scores(cluster_labels, np.array(codes))

    unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
    unique_codes, code_counts = np.unique(np.array(codes), return_counts=True)
    out = {
        "ari": scores["ari"],
        "nmi": scores["nmi"],
        "purity": purity,
        "inverse_purity": inv_purity,
        "num_clusters": int(num_clusters),
        "cluster_counts": {int(k): int(v) for k, v in zip(unique_clusters, cluster_counts)},
        "code_counts": {int(k): int(v) for k, v in zip(unique_codes, code_counts)},
        "alignment_samples": int(cfg["eval"]["alignment_samples"]),
        "alignment_rollouts": int(cfg["eval"]["alignment_rollouts"]),
        "horizon": int(cfg["train"]["horizon"]),
        "env_id": int(env_id),
        "seed": int(cfg["eval"].get("alignment_seed", cfg["seed"])),
        "future_summary": "future_label",
    }

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
