import argparse
import json
import os
import sys
from typing import Any, Dict, List

import yaml


def require(condition: bool, message: str, errors: List[str]) -> None:
    if not condition:
        errors.append(message)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_eval_metrics(eval_metrics: Dict[str, Any], errors: List[str]) -> None:
    require("in_distribution" in eval_metrics, "eval_metrics missing in_distribution", errors)
    require("ood" in eval_metrics, "eval_metrics missing ood", errors)
    require("per_env_risks" in eval_metrics, "eval_metrics missing per_env_risks", errors)
    require("risk_variance" in eval_metrics, "eval_metrics missing risk_variance", errors)
    require("complexity" in eval_metrics, "eval_metrics missing complexity", errors)
    require("interventional" in eval_metrics, "eval_metrics missing interventional", errors)
    require("rep_usage_test" in eval_metrics, "eval_metrics missing rep_usage_test", errors)
    require("rollout_counts" in eval_metrics, "eval_metrics missing rollout_counts", errors)
    require("seeds" in eval_metrics, "eval_metrics missing seeds", errors)

    if "complexity" in eval_metrics:
        comp = eval_metrics["complexity"]
        for key in ["perplexity", "active_codes", "entropy_or_proxy"]:
            require(key in comp, f"complexity missing {key}", errors)

    if "interventional" in eval_metrics:
        inter = eval_metrics["interventional"]
        require("details" in inter, "interventional missing details", errors)
        if isinstance(inter.get("details"), list):
            for idx, item in enumerate(inter["details"]):
                for key in ["spec", "mse", "env_id", "rollouts", "seed"]:
                    require(key in item, f"interventional.details[{idx}] missing {key}", errors)

    if "rep_usage_test" in eval_metrics:
        rep = eval_metrics["rep_usage_test"]
        for key in ["metric", "value"]:
            require(key in rep, f"rep_usage_test missing {key}", errors)

    if "per_env_risks" in eval_metrics:
        per_env = eval_metrics["per_env_risks"]
        require(isinstance(per_env, dict), "per_env_risks must be a dict", errors)
        require(len(per_env) > 0, "per_env_risks is empty", errors)


def check_alignment(alignment: Dict[str, Any], errors: List[str]) -> None:
    for key in ["ARI", "NMI", "alignment_rollouts", "alignment_samples", "num_clusters"]:
        require(key in alignment, f"alignment_metrics missing {key}", errors)


def check_one_shot(one_shot: Dict[str, Any], errors: List[str]) -> None:
    for key in ["intervention", "targeted_mechanism", "loss_before", "loss_after", "loss_delta"]:
        require(key in one_shot, f"one_shot_result missing {key}", errors)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=str)
    args = parser.parse_args()

    run_dir = args.run_dir
    errors: List[str] = []

    required_files = [
        "config_snapshot.yaml",
        "train_metrics.json",
        "eval_metrics.json",
        "alignment_metrics.json",
        "one_shot_result.json",
        "checkpoint.pt",
    ]

    for fname in required_files:
        require(os.path.exists(os.path.join(run_dir, fname)), f"missing {fname}", errors)

    config_path = os.path.join(run_dir, "config_snapshot.yaml")
    if os.path.exists(config_path):
        cfg = load_yaml(config_path)
        use_quantizer = cfg.get("model", {}).get("use_quantizer", True)
        enable_operator = cfg.get("train", {}).get("enable_operator", True)
        if use_quantizer:
            require(os.path.exists(os.path.join(run_dir, "code_usage.json")), "missing code_usage.json", errors)
        if enable_operator:
            require(os.path.exists(os.path.join(run_dir, "operator_events.json")), "missing operator_events.json", errors)

    eval_path = os.path.join(run_dir, "eval_metrics.json")
    if os.path.exists(eval_path):
        eval_metrics = load_json(eval_path)
        check_eval_metrics(eval_metrics, errors)

    alignment_path = os.path.join(run_dir, "alignment_metrics.json")
    if os.path.exists(alignment_path):
        alignment = load_json(alignment_path)
        check_alignment(alignment, errors)

    one_shot_path = os.path.join(run_dir, "one_shot_result.json")
    if os.path.exists(one_shot_path):
        one_shot = load_json(one_shot_path)
        check_one_shot(one_shot, errors)

    if errors:
        print("FAIL")
        for err in errors:
            print(f"- {err}")
        sys.exit(1)

    print("PASS")


if __name__ == "__main__":
    main()
