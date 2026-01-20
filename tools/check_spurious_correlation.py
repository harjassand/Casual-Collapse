import argparse
import math

import numpy as np
import yaml

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
    for _ in range(steps):
        _, _, _, info = env.step(None)
        spurious.append(float(info["spurious_value"]))
        target.append(target_from_info(env_type, info))
    return spurious, target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="hmm")
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    with open(f"configs/env/{args.env}.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env = make_env(cfg)
    env_type = cfg["type"]

    print(f"env={env_type}")
    for split, env_ids in [("train", cfg["train_env_ids"]), ("test", cfg["test_env_ids"])]:
        for env_id in env_ids:
            spurious, target = collect(env, env_type, env_id, args.steps)
            corr = pearson(spurious, target)
            print(f"{split} env_id={env_id} corr={corr:.3f} mean_spurious={np.mean(spurious):.3f}")

    if cfg.get("interventions"):
        print("interventions:")
        for spec in cfg["interventions"]:
            spurious, target = collect(env, env_type, cfg["train_env_ids"][0], args.steps, intervention=spec)
            corr = pearson(spurious, target)
            print(f"  spec={spec} corr={corr:.3f} mean_spurious={np.mean(spurious):.3f}")
    else:
        print("interventions: none")


if __name__ == "__main__":
    main()
