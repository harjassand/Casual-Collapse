import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import yaml
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs import HMMCausalEnv, MechanismShiftEnv, ObjectMicroWorldEnv


def make_env(cfg: Dict):
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


def target_from_info(env_type: str, info: Dict) -> float:
    if env_type == "hmm":
        return float(int(info["latent_state"]) % 2)
    if env_type == "object":
        return float(int(info.get("event", 0)))
    if env_type == "mechanism":
        return float(info.get("causal_value", 0.0))
    return 0.0


def collect(env, env_type: str, env_id: int, steps: int) -> Tuple[np.ndarray, np.ndarray]:
    env.reset(seed=123, env_id=env_id)
    xs = []
    ys = []
    for _ in range(steps):
        _, _, _, info = env.step(None)
        xs.append([float(info["spurious_value"])])
        ys.append(target_from_info(env_type, info))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="hmm")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--output_path", type=str, default="")
    args = parser.parse_args()

    with open(f"configs/env/{args.env}.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env = make_env(cfg)
    env_type = cfg["type"]

    X_train, y_train = [], []
    X_test, y_test = [], []
    for env_id in cfg["train_env_ids"]:
        xs, ys = collect(env, env_type, env_id, args.steps)
        X_train.append(xs)
        y_train.append(ys)
    for env_id in cfg["test_env_ids"]:
        xs, ys = collect(env, env_type, env_id, args.steps)
        X_test.append(xs)
        y_test.append(ys)

    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    if env_type in ("hmm", "object"):
        model = LogisticRegression(max_iter=500)
        model.fit(X_train, y_train)
        train_metric = float(model.score(X_train, y_train))
        test_metric = float(model.score(X_test, y_test))
        metric_name = "accuracy"
        probe_type = "logistic_regression"
    else:
        model = LinearRegression()
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_metric = float(mean_squared_error(y_train, train_pred))
        test_metric = float(mean_squared_error(y_test, test_pred))
        metric_name = "mse"
        probe_type = "linear_regression"

    output = {
        "env": env_type,
        "probe_type": probe_type,
        "metric": metric_name,
        "features": "spurious_only",
        "train_metric": train_metric,
        "test_metric": test_metric,
        "steps_per_env": int(args.steps),
        "train_env_ids": [int(e) for e in cfg["train_env_ids"]],
        "test_env_ids": [int(e) for e in cfg["test_env_ids"]],
    }

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
