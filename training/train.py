import copy
import json
import os
import sys
from typing import Any, Dict, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from collapse_operator.merge_split import merge_split_operator
from collapse_operator.stats_buffer import StatsBuffer
from envs import HMMCausalEnv, MechanismShiftEnv, ObjectMicroWorldEnv
from losses.invariance_irm import irm_penalty_envs
from losses.invariance_rex import rex_penalty
from losses.modularity import cross_jacobian_penalty, total_correlation_penalty
from losses.vib import code_entropy
from models.causal_collapse_model import CausalCollapseModel
from policies.active_info_gain import ActiveInfoGainPolicy
from policies.random_policy import RandomPolicy
from training.buffer import RolloutBuffer
from utils.device import resolve_device
from utils.logger import RunLogger
from utils.metrics import active_code_count
from utils.seed import set_seed


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


def future_label_from_info(env_type: str, info: Dict[str, Any]) -> int:
    if env_type == "hmm":
        return int(info["latent_state"])
    if env_type == "object":
        return int(info["event"])
    if env_type == "mechanism":
        pos = info["latent_state"]["pos"]
        return int(abs(pos[0] - pos[1]) < 0.1)
    return 0


def format_obs(obs: np.ndarray) -> np.ndarray:
    if obs.ndim == 1:
        return obs[None, :]
    return obs


def collect_rollout(
    env,
    policy,
    horizon: int,
    env_id: int,
    env_type: str,
) -> Dict[str, Any]:
    obs = env.reset(seed=None, env_id=env_id)
    obs = format_obs(obs)
    obs0 = obs
    actions = []
    future_obs = []
    future_actions = []
    info = {}
    for _ in range(horizon):
        if hasattr(policy, "act"):
            action = policy.act(obs)
        else:
            action = None
        next_obs, _, _, info = env.step(action)
        next_obs = format_obs(next_obs)
        actions.append(action if action is not None else np.zeros((0,), dtype=np.float32))
        future_actions.append(action if action is not None else np.zeros((0,), dtype=np.float32))
        future_obs.append(next_obs)
        obs = next_obs
    future_obs_arr = np.stack(future_obs, axis=0)
    actions_arr = None
    if actions:
        actions_arr = np.stack(actions, axis=0)
    future_actions_arr = None
    if future_actions:
        future_actions_arr = np.stack(future_actions, axis=0)
    return {
        "obs": obs0,
        "action": actions_arr[0] if actions_arr is not None else None,
        "future_obs": future_obs_arr,
        "future_actions": future_actions_arr,
        "future_label": future_label_from_info(env_type, info),
        "env_id": env_id,
    }


def make_policy(cfg: Dict[str, Any], action_dim: int):
    if cfg["type"] == "active":
        return ActiveInfoGainPolicy(
            action_dim=action_dim,
            num_candidates=cfg["num_candidates"],
            action_scale=cfg["action_scale"],
            cost_weight=cfg["cost_weight"],
            top_k=cfg.get("top_k", 5),
        )
    return RandomPolicy(action_dim=action_dim, action_scale=cfg["action_scale"])


def build_predict_fn(model: CausalCollapseModel, ensemble: List[CausalCollapseModel], device: torch.device, obs: np.ndarray):
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        enc = model.encode(obs_t)
        z = enc["quantized"]
        if model.cfg.get("use_residual", False):
            z = torch.cat([z, enc["residual"]], dim=-1)
    def predict(action: np.ndarray) -> np.ndarray:
        action_t = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
        preds = []
        for m in ensemble:
            with torch.no_grad():
                z_next = m.dynamics.forward(z, action_t, None)
                preds.append(z_next.cpu().numpy())
        return np.stack(preds, axis=0)
    return predict


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    apply_repr_mode(cfg, cfg_dict)
    set_seed(cfg.seed)
    device = resolve_device(cfg_dict.get("device", "auto"))

    env_cfg = cfg_dict["env"]
    env = make_env(env_cfg)

    model_cfg = cfg_dict["model"]
    pred_mode = cfg_dict.get("loss", {}).get("prediction") or env_cfg.get("prediction", "mse")
    cfg_dict["loss"]["prediction"] = pred_mode
    cfg.loss.prediction = pred_mode
    model = CausalCollapseModel(model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_dict["train"]["lr"])

    policy_cfg = cfg_dict["policy"]
    action_dim = model_cfg.get("action_dim", 0)
    policy = make_policy(policy_cfg, action_dim)

    ensemble = []
    for _ in range(policy_cfg.get("ensemble_size", 1)):
        member = copy.deepcopy(model).to(device)
        member.eval()
        ensemble.append(member)

    buffer = RolloutBuffer(max_size=cfg_dict["train"]["buffer_size"])
    logger = RunLogger(cfg_dict["train"]["run_dir"])
    os.makedirs(cfg_dict["train"]["run_dir"], exist_ok=True)
    with open(os.path.join(cfg_dict["train"]["run_dir"], "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=2)
    config_snapshot_path = os.path.join(cfg_dict["train"]["run_dir"], "config_snapshot.yaml")
    OmegaConf.save(cfg, config_snapshot_path)
    operator_events: List[Dict[str, Any]] = []
    pred_history: List[float] = []

    stats = StatsBuffer(
        num_codes=model_cfg["num_codes"],
        num_envs=len(env_cfg["train_env_ids"]),
        max_per_code=cfg_dict["train"]["stats_max_per_code"],
    )

    env_ids = env_cfg["train_env_ids"]
    env_id_map = {int(e): i for i, e in enumerate(env_ids)}
    horizon = cfg_dict["train"]["horizon"]
    last_gain = 0.0
    last_usage: Optional[np.ndarray] = None
    last_policy_info: Dict[str, Any] = {}
    dead_counts = np.zeros(model_cfg["num_codes"], dtype=np.int64)
    vq_cfg = cfg_dict.get("vq", {})
    dead_cfg = vq_cfg.get("dead_code_reinit", {})
    dead_enabled = bool(dead_cfg.get("enabled", False))
    dead_min_usage = float(dead_cfg.get("min_usage", cfg_dict["train"].get("dead_code_threshold", 0.0)))
    dead_window = int(dead_cfg.get("window_steps", cfg_dict["train"].get("dead_code_patience", 0)))
    dead_strategy = dead_cfg.get("strategy", "sample_encoder")
    vq_entropy_reg = float(vq_cfg.get("entropy_reg", cfg_dict["loss"].get("vq_entropy_reg", 0.0)))
    vq_events: List[Dict[str, Any]] = []

    for step in range(cfg_dict["train"]["num_steps"]):
        env_id = int(np.random.choice(env_ids))
        if policy_cfg["type"] == "active":
            obs = env.reset(seed=None, env_id=env_id)
            obs = format_obs(obs)
            predict_fn = build_predict_fn(model, ensemble, device, obs)
            action, info = policy.act(obs, predict_fn)
            last_gain = float(info.get("expected_gain", 0.0))
            last_policy_info = info
            next_obs, _, _, info = env.step(action)
            next_obs = format_obs(next_obs)
            future_obs = [next_obs]
            future_actions = [action if action is not None else np.zeros((0,), dtype=np.float32)]
            for _ in range(horizon - 1):
                predict_fn = build_predict_fn(model, ensemble, device, next_obs)
                action, info = policy.act(next_obs, predict_fn)
                last_gain = float(info.get("expected_gain", last_gain))
                last_policy_info = info
                next_obs, _, _, info = env.step(action)
                next_obs = format_obs(next_obs)
                future_obs.append(next_obs)
                future_actions.append(action if action is not None else np.zeros((0,), dtype=np.float32))
            buffer.add(
                obs=obs,
                action=action,
                future_obs=np.stack(future_obs, axis=0),
                future_actions=np.stack(future_actions, axis=0),
                env_id=env_id,
                future_label=future_label_from_info(env_cfg["type"], info),
            )
        else:
            rollout = collect_rollout(env, policy, horizon, env_id, env_cfg["type"])
            buffer.add(**rollout)

        if len(buffer) < cfg_dict["train"]["batch_size"]:
            continue

        batch = buffer.sample(cfg_dict["train"]["batch_size"])
        obs = torch.tensor(batch["obs"], dtype=torch.float32, device=device)
        future_obs = torch.tensor(batch["future_obs"], dtype=torch.float32, device=device)
        actions = None
        if batch["future_actions"] is not None and action_dim > 0:
            actions = torch.tensor(batch["future_actions"], dtype=torch.float32, device=device)
        env_ids_batch = batch["env_ids"]
        future_labels = batch["futures"]

        if obs.ndim == 2:
            obs = obs.unsqueeze(1)
        if future_obs.ndim == 3:
            future_obs = future_obs.unsqueeze(2)

        out = model(obs, actions, horizon=horizon)
        preds = out["preds"]
        if pred_mode == "label_ce":
            labels_t = torch.tensor(future_labels, dtype=torch.long, device=device)
            if "label_logits" not in out:
                raise ValueError("label_ce requires model.label_head and label_logits")
            pred_loss = F.cross_entropy(out["label_logits"], labels_t)
        else:
            pred_loss = torch.mean((preds - future_obs) ** 2)
        pred_history.append(float(pred_loss.item()))
        if len(pred_history) > cfg_dict["train"]["pred_plateau_window"]:
            pred_history.pop(0)

        usage = out["vq_stats"].get("soft_usage", out["vq_stats"]["usage"])
        usage_hard = out["vq_stats"]["usage"]
        kl_term = (usage * (usage + 1e-8).log()).sum() + np.log(model_cfg["num_codes"])
        entropy = code_entropy(usage)
        log_likelihood = -pred_loss
        vib_loss = kl_term - cfg_dict["loss"]["beta"] * log_likelihood
        last_usage = usage_hard.detach().cpu().numpy()

        # Invariance penalties
        risks = []
        logits_list = []
        targets_list = []
        for env_id in np.unique(env_ids_batch):
            idx = env_ids_batch == env_id
            if idx.sum() == 0:
                continue
            if pred_mode == "label_ce":
                env_logits = out["label_logits"][idx]
                env_labels = labels_t[idx]
                risk = F.cross_entropy(env_logits, env_labels)
                risks.append(risk)
                logits_list.append(env_logits)
                targets_list.append(env_labels)
            else:
                env_pred = preds[idx]
                env_future = future_obs[idx]
                risk = torch.mean((env_pred - env_future) ** 2)
                risks.append(risk)
                logits_list.append(env_pred.reshape(env_pred.shape[0], -1))
                targets_list.append(env_future.reshape(env_future.shape[0], -1))

        rex = rex_penalty(risks) if cfg_dict["loss"]["use_rex"] else torch.tensor(0.0, device=device)
        if cfg_dict["loss"]["use_irm"]:
            loss_fn = F.cross_entropy if pred_mode == "label_ce" else torch.nn.functional.mse_loss
            irm = irm_penalty_envs(logits_list, targets_list, loss_fn)
        else:
            irm = torch.tensor(0.0, device=device)
        inv_penalty = rex + irm

        # Modularity
        mod_penalty = torch.tensor(0.0, device=device)
        if cfg_dict["loss"]["modularity"] == "cross_jacobian":
            z = out["quantized"].detach().requires_grad_(True)
            z_next = model.dynamics.forward(z, None, None)
            mod_penalty = cross_jacobian_penalty(z_next, z)
        elif cfg_dict["loss"]["modularity"] == "total_correlation":
            z = out["quantized"]
            mod_penalty = total_correlation_penalty(z)

        logic_loss = torch.tensor(0.0, device=device)
        if model.logic_layer is not None:
            logic_out = model.logic_layer(out["vq_stats"]["indices"])
            logic_loss = logic_out["logic_loss"]

        vq_entropy_reg_term = -entropy
        total_loss = (
            pred_loss
            + cfg_dict["loss"]["gamma"] * vib_loss
            + cfg_dict["loss"]["lambda"] * inv_penalty
            + cfg_dict["loss"]["mu"] * mod_penalty
            + cfg_dict["loss"]["nu"] * logic_loss
            + vq_entropy_reg * vq_entropy_reg_term
            + out["vq_loss"]
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Dead-code revival (optional)
        if dead_enabled and dead_window > 0 and model.use_quantizer and model.quantizer is not None:
            usage_np = last_usage if last_usage is not None else usage_hard.detach().cpu().numpy()
            for i, u in enumerate(usage_np):
                if u < dead_min_usage:
                    dead_counts[i] += 1
                else:
                    dead_counts[i] = 0
            revive = np.where(dead_counts >= dead_window)[0]
            if revive.size > 0:
                with torch.no_grad():
                    if dead_strategy == "random_batch":
                        flat = out["quantized"].detach().reshape(-1, out["quantized"].shape[-1])
                    else:
                        flat = out["slots"].detach().reshape(-1, out["slots"].shape[-1])
                    for idx in revive:
                        sample_idx = int(np.random.randint(flat.shape[0]))
                        vec = flat[sample_idx]
                        model.quantizer.codebook[idx].copy_(vec)
                        if getattr(model.quantizer, "use_ema", False):
                            model.quantizer.ema_counts[idx] = 1.0
                            model.quantizer.ema_means[idx].copy_(vec)
                        vq_events.append({
                            "step": step,
                            "code_id": int(idx),
                            "prior_usage": float(usage_np[idx]),
                            "strategy": dead_strategy,
                        })
                        dead_counts[idx] = 0

        # Update stats buffer
        codes_np = out["vq_stats"]["indices"].reshape(-1).cpu().numpy()
        emb_np = out["quantized"].reshape(-1, out["quantized"].shape[-1]).detach().cpu().numpy()
        env_idx = np.array([env_id_map[int(e)] for e in env_ids_batch], dtype=np.int64)
        env_np = np.repeat(env_idx, obs.shape[1])
        future_np = np.repeat(future_labels, obs.shape[1])
        stats.update(codes_np, future_np, env_np, embeddings=emb_np)

        if step % cfg_dict["train"]["operator_interval"] == 0 and step > 0:
            perplexity = float(out["vq_stats"]["perplexity"].item())
            plateau = False
            if len(pred_history) == cfg_dict["train"]["pred_plateau_window"]:
                plateau = abs(pred_history[-1] - pred_history[0]) < cfg_dict["train"]["pred_plateau_threshold"]
            if (
                (perplexity < cfg_dict["train"]["perplexity_threshold"] or plateau)
                and model.use_quantizer
                and cfg_dict["train"]["enable_operator"]
            ):
                codebook = model.quantizer.codebook.detach().cpu().numpy()
                ops = merge_split_operator(
                    stats,
                    codebook,
                    num_classes=cfg_dict["train"]["num_future_classes"],
                    delta_merge=cfg_dict["operator"]["delta_merge"],
                    delta_merge_inv=cfg_dict["operator"]["delta_merge_inv"],
                    delta_split=cfg_dict["operator"]["delta_split"],
                    eps_split_gain=cfg_dict["operator"]["eps_split_gain"],
                    env_ids=list(range(len(env_cfg["train_env_ids"]))),
                )
                with torch.no_grad():
                    model.quantizer.codebook.copy_(torch.tensor(codebook, device=device))
                logger.log(step, {
                    "operator/merges": len(ops["merges"]),
                    "operator/splits": len(ops["splits"]),
                })
                for merge in ops["merges"]:
                    event = {
                        "step": step,
                        "type": "merge",
                        "perplexity": perplexity,
                        "plateau_trigger": plateau,
                        "delta_merge": cfg_dict["operator"]["delta_merge"],
                        "delta_merge_inv": cfg_dict["operator"]["delta_merge_inv"],
                    }
                    event.update(merge)
                    operator_events.append(event)
                for split in ops["splits"]:
                    event = {
                        "step": step,
                        "type": "split",
                        "perplexity": perplexity,
                        "plateau_trigger": plateau,
                        "delta_split": cfg_dict["operator"]["delta_split"],
                        "eps_split_gain": cfg_dict["operator"]["eps_split_gain"],
                    }
                    event.update(split)
                    operator_events.append(event)
                stats.reset()

        if step % cfg_dict["train"]["log_interval"] == 0:
            counts = out["vq_stats"]["usage"].detach().cpu().numpy()
            active_codes = active_code_count(counts, cfg_dict["train"]["active_code_threshold"])
            risk_metrics = {}
            for env_id, risk in zip(np.unique(env_ids_batch), risks):
                risk_metrics[f"risk/env_{int(env_id)}"] = float(risk.item())
            code_metrics = {f"code_usage/{i}": float(counts[i]) for i in range(len(counts))}
            entropy_val = max(0.0, float(code_entropy(out["vq_stats"]["usage"]).item()))
            p_min = float(out["vq_stats"]["usage"].min().item())
            p_max = float(out["vq_stats"]["usage"].max().item())
            degenerate = int(
                entropy_val < cfg_dict["train"]["degenerate_entropy_threshold"]
                and float(pred_loss.item()) > cfg_dict["train"]["degenerate_pred_threshold"]
            )
            rep_usage = 0.0
            with torch.no_grad():
                perm = torch.randperm(obs.shape[0], device=device)
                z = out["quantized"]
                if model_cfg.get("use_residual", False):
                    z = torch.cat([z, out["residual"]], dim=-1)
                z_shuf = z[perm]
                if pred_mode == "label_ce":
                    logits_shuf = model.label_head(z_shuf.reshape(z_shuf.shape[0], -1))
                    ce_shuf = F.cross_entropy(logits_shuf, labels_t)
                    rep_usage = float(ce_shuf.item()) - float(pred_loss.item())
                else:
                    adj = out["adj"][perm] if out["adj"] is not None else None
                    preds_shuf = model.dynamics.rollout(z_shuf, actions, adj, horizon)
                    preds_shuf = model.decoder(preds_shuf)
                    rep_usage = float(torch.mean((preds_shuf - future_obs) ** 2).item()) - float(pred_loss.item())
            adj_metrics = {}
            if out["adj"] is not None:
                adj = out["adj"]
                edge_prob = adj.mean()
                edge_entropy = -(edge_prob * (edge_prob + 1e-8).log() + (1 - edge_prob) * (1 - edge_prob + 1e-8).log())
                adj_metrics = {
                    "graph/edge_prob": float(edge_prob.item()),
                    "graph/edge_entropy": float(edge_entropy.item()),
                }
            logger.log(step, {
                "loss/pred": float(pred_loss.item()),
                "loss/vib": float(vib_loss.item()),
                "loss/vib_kl": float(kl_term.item()),
                "loss/vib_ll": float(log_likelihood.item()),
                "loss/inv": float(inv_penalty.item()),
                "loss/mod": float(mod_penalty.item()),
                "loss/logic": float(logic_loss.item()),
                "loss/vq_entropy_reg": float(vq_entropy_reg_term.item()),
                "loss/vq_entropy_reg_weight": float(vq_entropy_reg),
                "loss/vq": float(out["vq_loss"].item()),
                "stats/perplexity": float(out["vq_stats"]["perplexity"].item()),
                "stats/active_codes": active_codes,
                "stats/entropy": entropy_val,
                "stats/p_min": p_min,
                "stats/p_max": p_max,
                "stats/degenerate": degenerate,
                "stats/rep_usage_delta": rep_usage,
                "stats/dead_code_count": int(np.sum(dead_counts >= dead_window)) if dead_window > 0 else 0,
                "stats/rex": float(rex.item()) if isinstance(rex, torch.Tensor) else 0.0,
                "stats/irm": float(irm.item()) if isinstance(irm, torch.Tensor) else 0.0,
                "policy/expected_gain": last_gain,
                "policy/score": float(last_policy_info.get("score", 0.0)),
                "policy/cost": float(last_policy_info.get("cost", 0.0)),
                "policy/action": last_policy_info.get("action", []),
                "policy/top_scores": last_policy_info.get("top_scores", []),
                **risk_metrics,
                **code_metrics,
                **adj_metrics,
            })

        if step % cfg_dict["train"]["checkpoint_interval"] == 0 and step > 0:
            ckpt_path = os.path.join(cfg_dict["train"]["run_dir"], f"model_{step}.pt")
            torch.save({"model": model.state_dict(), "cfg": cfg_dict}, ckpt_path)

        if policy_cfg["type"] == "active" and step % policy_cfg.get("ensemble_sync", 100) == 0:
            for m in ensemble:
                m.load_state_dict(copy.deepcopy(model.state_dict()))

    logger.close()

    metrics_path = os.path.join(cfg_dict["train"]["run_dir"], "metrics.jsonl")
    train_metrics_path = os.path.join(cfg_dict["train"]["run_dir"], "train_metrics.json")
    if os.path.exists(metrics_path):
        records = []
        with open(metrics_path, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
        with open(train_metrics_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
    else:
        with open(train_metrics_path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)

    if last_usage is not None:
        code_usage_path = os.path.join(cfg_dict["train"]["run_dir"], "code_usage.json")
        active_codes = active_code_count(last_usage, cfg_dict["train"]["active_code_threshold"])
        entropy_val = float(-np.sum(last_usage * np.log(last_usage + 1e-8)))
        code_usage = {
            "usage": last_usage.tolist(),
            "perplexity": float(np.exp(entropy_val)),
            "entropy": float(max(0.0, entropy_val)),
            "active_codes": int(active_codes),
            "p_min": float(np.min(last_usage)),
            "p_max": float(np.max(last_usage)),
        }
        with open(code_usage_path, "w", encoding="utf-8") as f:
            json.dump(code_usage, f, indent=2)

    if cfg_dict["train"]["enable_operator"]:
        operator_path = os.path.join(cfg_dict["train"]["run_dir"], "operator_events.json")
        with open(operator_path, "w", encoding="utf-8") as f:
            json.dump(operator_events, f, indent=2)

    if dead_enabled:
        vq_events_path = os.path.join(cfg_dict["train"]["run_dir"], "vq_events.json")
        with open(vq_events_path, "w", encoding="utf-8") as f:
            json.dump(vq_events, f, indent=2)

    checkpoint_path = os.path.join(cfg_dict["train"]["run_dir"], "checkpoint.pt")
    torch.save({"model": model.state_dict(), "cfg": cfg_dict}, checkpoint_path)


if __name__ == "__main__":
    main()
