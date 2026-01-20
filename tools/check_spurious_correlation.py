import argparse
import json
import os
import sys

import numpy as np
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs import HMMCausalEnv, MechanismShiftEnv, ObjectMicroWorldEnv


def make_env(cfg):
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


def target_from_info(env_type, info):
    if env_type == "hmm":
        return int(info["latent_state"]) % 2
    if env_type == "object":
        return int(info.get("event", 0))
    if env_type == "mechanism":
        return float(info.get("causal_value", 0.0))
    return 0.0


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def collect(env, env_type, env_id, steps, intervention=None):
    env.reset(seed=123, env_id=env_id)
    if intervention:
        env.do_intervention(intervention)
    spurious = []
    target = []
    last_info = None
    for _ in range(steps):
        _, _, _, info = env.step(None)
        last_info = info
        spurious.append(float(info["spurious_value"]))
        target.append(target_from_info(env_type, info))
    intervention_active = bool(last_info.get("intervention_active")) if last_info else False
    intervention_spec = last_info.get("intervention_spec") if last_info else {}
    return spurious, target, intervention_active, intervention_spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="hmm")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output_path", type=str, default="")
    args = parser.parse_args()

    with open(f"configs/env/{args.env}.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env = make_env(cfg)
    env_type = cfg["type"]

    output = {
        "env": env_type,
        "train": [],
        "test": [],
        "interventions": [],
    }
    print(f"env={env_type}")
    for split, env_ids in [("train", cfg["train_env_ids"]), ("test", cfg["test_env_ids"])]:
        for env_id in env_ids:
            spurious, target, intervention_active, intervention_spec = collect(env, env_type, env_id, args.steps)
            corr = pearson(spurious, target)
            record = {
                "env_id": int(env_id),
                "corr": corr,
                "mean_spurious": float(np.mean(spurious)),
                "intervention_active": bool(intervention_active),
                "intervention_spec": intervention_spec,
            }
            output[split].append(record)
            print(f"{split} env_id={env_id} corr={corr:.3f} mean_spurious={np.mean(spurious):.3f}")

    if cfg.get("interventions"):
        print("interventions:")
        for spec in cfg["interventions"]:
            spurious, target, intervention_active, intervention_spec = collect(
                env, env_type, cfg["train_env_ids"][0], args.steps, intervention=spec
            )
            corr = pearson(spurious, target)
            record = {
                "spec": spec,
                "corr": corr,
                "mean_spurious": float(np.mean(spurious)),
                "intervention_active": bool(intervention_active),
                "intervention_spec": intervention_spec,
            }
            output["interventions"].append(record)
            print(f"  spec={spec} corr={corr:.3f} mean_spurious={np.mean(spurious):.3f}")
    else:
        print("interventions: none")

    if args.output_path:
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
