import json
import os
import sys
from typing import Any, Dict, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LinearRegression

from envs import HMMCausalEnv, MechanismShiftEnv, ObjectMicroWorldEnv
from models.causal_collapse_model import CausalCollapseModel
from losses.modularity import total_correlation_penalty
from utils.device import resolve_device
from utils.metrics import active_code_count, code_perplexity


def apply_repr_mode(cfg: DictConfig, cfg_dict: Dict[str, Any]) -> None:
    repr_mode = cfg_dict.get("model", {}).get("repr_mode")
    if not repr_mode:
        return
    if repr_mode == "discrete_only":
        cfg.model.use_quantizer = True
        cfg.model.use_residual = False
    elif repr_mode == "continuous_only":
        cfg.model.use_quantizer = False
        cfg.model.use_residual = False
    elif repr_mode == "multiscale":
        cfg.model.use_quantizer = True
        cfg.model.use_residual = True
    cfg_dict["model"]["use_quantizer"] = bool(cfg.model.use_quantizer)
    cfg_dict["model"]["use_residual"] = bool(cfg.model.use_residual)


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


def future_label_from_info(env_type: str, info: Dict[str, Any]) -> int:
    if env_type == "hmm":
        return int(info["latent_state"])
    if env_type == "object":
        return int(info.get("event", 0))
    if env_type == "mechanism":
        pos = info["latent_state"]["pos"]
        return int(abs(pos[0] - pos[1]) < 0.1)
    return 0


def collect_rollouts(
    env,
    model,
    env_id: int,
    steps: int,
    horizon: int,
    device: torch.device,
    seed: int,
    env_type: str,
    intervention_spec: Dict[str, Any] = None,
):
    obs_batch = []
    trues = []
    labels = []
    for i in range(steps):
        obs = env.reset(seed=seed + i, env_id=env_id)
        if intervention_spec:
            env.do_intervention(intervention_spec)
        obs = format_obs(obs)
        obs_batch.append(obs)

        future_obs = []
        info = {}
        for _ in range(horizon):
            next_obs, _, _, info = env.step(None)
            next_obs = format_obs(next_obs)
            future_obs.append(next_obs)
        trues.append(np.stack(future_obs, axis=0))
        labels.append(future_label_from_info(env_type, info))

    obs_batch = np.stack(obs_batch, axis=0)
    trues = np.stack(trues, axis=0)
    labels = np.asarray(labels, dtype=np.int64)

    obs_t = torch.tensor(obs_batch, dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(obs_t, None, horizon)
    preds = out["preds"].detach().cpu().numpy()
    codes = out["vq_stats"]["indices"].detach().cpu().numpy()
    label_logits = out.get("label_logits")
    return preds, trues, codes, out, labels, label_logits


def rep_usage_metrics(model, out, future_obs: np.ndarray, horizon: int, pred_mode: str, labels: np.ndarray) -> Dict[str, float]:
    device = out["preds"].device
    if pred_mode == "label_ce":
        labels_t = torch.tensor(labels, dtype=torch.long, device=device)
        ce_base = float(F.cross_entropy(out["label_logits"], labels_t).item())
        metric_base = ce_base
    else:
        future_t = torch.tensor(future_obs, dtype=torch.float32, device=device)
        mse_base = torch.mean((out["preds"] - future_t) ** 2).item()
        metric_base = mse_base

    z = out["quantized"]
    if model.cfg.get("use_residual", False):
        z = torch.cat([z, out["residual"]], dim=-1)
    perm = torch.randperm(z.shape[0], device=device)
    z_shuf = z[perm]
    if pred_mode == "label_ce":
        logits_shuf = model.label_head(z_shuf.reshape(z_shuf.shape[0], -1))
        ce_pert = float(F.cross_entropy(logits_shuf, labels_t).item())
        metric_pert = ce_pert
    else:
        adj = None
        if model.graph_infer is not None:
            adj = model.graph_infer(out["quantized"][perm])
        with torch.no_grad():
            preds_shuf = model.dynamics.rollout(z_shuf, None, adj, horizon)
            preds_shuf = model.decoder(preds_shuf)
        metric_pert = float(torch.mean((preds_shuf - future_t) ** 2).item())
    mse_delta = metric_pert - metric_base
    mse_ratio = metric_pert / max(metric_base, 1e-8)
    return {
        "mse_base": metric_base,
        "mse_perturbed": metric_pert,
        "mse_delta": mse_delta,
        "mse_ratio": mse_ratio,
    }


def linear_probe_drift(codes: np.ndarray, labels: np.ndarray) -> np.ndarray:
    # codes: [N, K], labels: [N]
    one_hot = np.eye(np.max(codes) + 1)[codes.reshape(-1)].reshape(codes.shape[0], -1)
    reg = LinearRegression().fit(one_hot, labels)
    return reg.coef_


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    apply_repr_mode(cfg, cfg_dict)
    device = resolve_device(cfg_dict.get("device", "auto"))

    model_cfg = cfg_dict["model"]
    model = CausalCollapseModel(model_cfg).to(device)
    ckpt = torch.load(cfg_dict["eval"]["ckpt_path"], map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    env_cfg = cfg_dict["env"]
    env = make_env(env_cfg)
    pred_mode = cfg_dict.get("loss", {}).get("prediction") or env_cfg.get("prediction", "mse")
    cfg_dict["loss"]["prediction"] = pred_mode

    metrics: Dict[str, Any] = {}
    in_envs = env_cfg["train_env_ids"]
    ood_envs = env_cfg["test_env_ids"]

    rollouts_dump = {"preds": [], "trues": [], "codes": [], "env_ids": [], "intervention_id": []}
    per_env_risks: Dict[str, float] = {}
    in_env_risks: List[float] = []
    rep_usage_stats: List[Dict[str, float]] = []

    def eval_envs(env_ids: List[int], prefix: str) -> None:
        mse_list = []
        code_counts = []
        tc_list = []
        for env_id in env_ids:
            preds, trues, codes, out, labels, label_logits = collect_rollouts(
                env,
                model,
                env_id,
                cfg_dict["eval"]["steps"],
                cfg_dict["train"]["horizon"],
                device,
                seed=cfg_dict["eval"]["eval_seed"] + int(env_id),
                env_type=env_cfg["type"],
            )
            if pred_mode == "label_ce":
                labels_t = torch.tensor(labels, dtype=torch.long, device=device)
                ce = float(F.cross_entropy(label_logits, labels_t).item())
                mse = ce
            else:
                mse = float(np.mean((preds - trues) ** 2))
            mse_list.append(mse)
            per_env_risks[f"{prefix}/risk_env_{int(env_id)}"] = mse
            per_env_risks[str(env_id)] = mse
            if prefix == "in":
                in_env_risks.append(mse)
            code_counts.append(codes.reshape(-1))
            rep_usage_stats.append(rep_usage_metrics(model, out, trues, cfg_dict["train"]["horizon"], pred_mode, labels))
            if env_cfg["type"] == "mechanism":
                with torch.no_grad():
                    tc = total_correlation_penalty(out["quantized"]).item()
                tc_list.append(tc)
            if cfg_dict["eval"]["save_rollouts"]:
                rollouts_dump["preds"].append(preds)
                rollouts_dump["trues"].append(trues)
                rollouts_dump["codes"].append(codes)
                rollouts_dump["env_ids"].append(np.full(preds.shape[0], env_id))
                rollouts_dump["intervention_id"].append(np.full(preds.shape[0], -1))
        all_codes = np.concatenate(code_counts)
        counts = np.bincount(all_codes, minlength=model_cfg["num_codes"])
        probs = counts / max(counts.sum(), 1)
        entropy = float(max(0.0, -np.sum(probs * np.log(probs + 1e-8))))
        metrics[f"{prefix}/mse"] = float(np.mean(mse_list))
        metrics[f"{prefix}/perplexity"] = code_perplexity(counts)
        metrics[f"{prefix}/active_codes"] = active_code_count(counts, cfg_dict["eval"]["active_code_threshold"])
        metrics[f"{prefix}/entropy"] = entropy
        if tc_list:
            metrics[f"{prefix}/total_correlation"] = float(np.mean(tc_list))

    eval_envs(in_envs, "in")
    eval_envs(ood_envs, "ood")

    all_envs = env_cfg.get("all_env_ids") or sorted(set(in_envs + ood_envs))
    extra_envs = [e for e in all_envs if e not in in_envs and e not in ood_envs]
    if extra_envs:
        eval_envs(extra_envs, "all")

    if env_cfg["type"] == "mechanism":
        metrics["mechanism/total_correlation_in"] = metrics.get("in/total_correlation", None)
        metrics["mechanism/total_correlation_ood"] = metrics.get("ood/total_correlation", None)

    # Interventions
    interventional_scores = []
    interventional_details = []
    interventions = cfg_dict["eval"].get("interventions") or env_cfg.get("interventions", [])
    for idx, spec in enumerate(interventions):
        preds, trues, _, out, labels, label_logits = collect_rollouts(
            env,
            model,
            env_cfg["test_env_ids"][0],
            cfg_dict["eval"]["steps"],
            cfg_dict["train"]["horizon"],
            device,
            seed=cfg_dict["eval"]["eval_seed"] + idx,
            env_type=env_cfg["type"],
            intervention_spec=spec,
        )
        if pred_mode == "label_ce":
            labels_t = torch.tensor(labels, dtype=torch.long, device=device)
            metric_val = float(F.cross_entropy(label_logits, labels_t).item())
        else:
            metric_val = float(np.mean((preds - trues) ** 2))
        interventional_scores.append(metric_val)
        interventional_details.append({
            "spec": spec,
            "mse": metric_val,
            "env_id": int(env_cfg["test_env_ids"][0]),
            "rollouts": int(cfg_dict["eval"]["steps"]),
            "seed": int(cfg_dict["eval"]["eval_seed"]) + idx,
        })
        rep_usage_stats.append(rep_usage_metrics(model, out, trues, cfg_dict["train"]["horizon"], pred_mode, labels))
        if cfg_dict["eval"]["save_rollouts"]:
            rollouts_dump["preds"].append(preds)
            rollouts_dump["trues"].append(trues)
            rollouts_dump["codes"].append(np.zeros((preds.shape[0], model_cfg["num_slots"]), dtype=np.int64))
            rollouts_dump["env_ids"].append(np.full(preds.shape[0], env_cfg["test_env_ids"][0]))
            rollouts_dump["intervention_id"].append(np.full(preds.shape[0], idx))
    metrics["interventional/mse"] = float(np.mean(interventional_scores)) if interventional_scores else 0.0
    metrics["interventional/details"] = interventional_details
    metrics["interventional/specs"] = interventions

    # Invariance probe drift
    drift = []
    for env_id in in_envs:
        _, _, codes, _, _, _ = collect_rollouts(
            env,
            model,
            env_id,
            cfg_dict["eval"]["steps"],
            cfg_dict["train"]["horizon"],
            device,
            seed=cfg_dict["eval"]["eval_seed"] + int(env_id),
            env_type=env_cfg["type"],
        )
        labels = np.repeat(env_id, codes.shape[0])
        drift.append(linear_probe_drift(codes.reshape(codes.shape[0], -1), labels))
    if drift:
        drift = np.stack(drift, axis=0)
        mean_w = drift.mean(axis=0)
        metrics["invariance/probe_drift"] = float(np.mean(np.linalg.norm(drift - mean_w, axis=1)))
    metrics.update(per_env_risks)
    if in_env_risks:
        metrics["invariance/risk_variance"] = float(np.var(in_env_risks))
    if rep_usage_stats:
        mse_base = float(np.mean([m["mse_base"] for m in rep_usage_stats]))
        mse_pert = float(np.mean([m["mse_perturbed"] for m in rep_usage_stats]))
        mse_delta = float(np.mean([m["mse_delta"] for m in rep_usage_stats]))
        mse_ratio = float(np.mean([m["mse_ratio"] for m in rep_usage_stats]))
    else:
        mse_base = mse_pert = mse_delta = mse_ratio = 0.0
    metrics["stats/rep_usage_delta"] = mse_delta
    metrics["eval/seed"] = int(cfg_dict["eval"]["eval_seed"])
    metrics["eval/steps"] = int(cfg_dict["eval"]["steps"])
    metrics["eval/horizon"] = int(cfg_dict["train"]["horizon"])

    per_env_map = {}
    for env_id in all_envs:
        key = f"in/risk_env_{int(env_id)}"
        if key in per_env_risks:
            per_env_map[str(env_id)] = per_env_risks[key]
            continue
        key = f"ood/risk_env_{int(env_id)}"
        if key in per_env_risks:
            per_env_map[str(env_id)] = per_env_risks[key]
            continue
        key = f"all/risk_env_{int(env_id)}"
        if key in per_env_risks:
            per_env_map[str(env_id)] = per_env_risks[key]

    metric_name = "ce" if pred_mode == "label_ce" else "mse"
    metrics["in_distribution"] = {"metric": metric_name, "value": metrics.get("in/mse", None)}
    metrics["ood"] = {"metric": metric_name, "value": metrics.get("ood/mse", None)}
    metrics["per_env_risks"] = per_env_map
    metrics["risk_variance"] = metrics.get("invariance/risk_variance", None)
    metrics["complexity"] = {
        "perplexity": metrics.get("in/perplexity", None),
        "active_codes": metrics.get("in/active_codes", None),
        "entropy_or_proxy": metrics.get("in/entropy", None),
        "codebook_size": int(model_cfg["num_codes"]),
        "active_code_threshold": float(cfg_dict["eval"]["active_code_threshold"]),
    }
    metrics["interventional"] = {
        "metric": metric_name,
        "details": interventional_details,
        "rollouts": int(cfg_dict["eval"]["steps"]),
        "seed": int(cfg_dict["eval"]["eval_seed"]),
    }
    metrics["rep_usage_test"] = {
        "metric": metric_name,
        "mse_base": mse_base,
        "mse_perturbed": mse_pert,
        "mse_delta": mse_delta,
        "mse_ratio": mse_ratio,
    }
    metrics["rollout_counts"] = {
        "steps": int(cfg_dict["eval"]["steps"]),
        "horizon": int(cfg_dict["train"]["horizon"]),
        "num_envs": int(len(all_envs)),
    }
    metrics["seeds"] = {"eval_seed": int(cfg_dict["eval"]["eval_seed"])}
    metrics["env_sets"] = {
        "train_env_ids": [int(e) for e in in_envs],
        "test_env_ids": [int(e) for e in ood_envs],
        "all_env_ids": [int(e) for e in all_envs],
    }

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
