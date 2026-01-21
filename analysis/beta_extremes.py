import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def last_train_metrics(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    data = load_json(path)
    if isinstance(data, list) and data:
        return data[-1]
    return {}


def run_one(env: str, model: str, preset: str, run_dir: str, beta: float, lam: float, mu: float, seed: int, overrides: List[str]) -> Dict[str, Any]:
    os.makedirs(run_dir, exist_ok=True)
    train_cmd = [
        sys.executable,
        "training/train.py",
        f"env={env}",
        f"model={model}",
        f"preset={preset}",
        f"train.run_dir={run_dir}",
        f"loss.beta={beta}",
        f"loss.lambda={lam}",
        f"loss.mu={mu}",
        f"seed={seed}",
    ] + overrides
    subprocess.run(train_cmd, check=True)

    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    eval_cmd = [
        sys.executable,
        "training/rollout_eval.py",
        f"env={env}",
        f"model={model}",
        f"preset={preset}",
        f"eval.ckpt_path={ckpt_path}",
        f"eval.output_path={os.path.join(run_dir, 'eval_metrics.json')}",
    ] + overrides
    subprocess.run(eval_cmd, check=True)

    eval_metrics = load_json(os.path.join(run_dir, "eval_metrics.json"))
    train_metrics = last_train_metrics(os.path.join(run_dir, "train_metrics.json"))
    cfg = load_json(os.path.join(run_dir, "config.json"))
    return {
        "run_dir": run_dir,
        "beta": beta,
        "lambda": lam,
        "mu": mu,
        "seed": seed,
        "repr_mode": cfg.get("model", {}).get("repr_mode"),
        "vq_use_ema": cfg.get("model", {}).get("vq_use_ema"),
        "vq_entropy_reg": cfg.get("vq", {}).get("entropy_reg"),
        "metrics": {
            "in_distribution": eval_metrics.get("in_distribution"),
            "ood": eval_metrics.get("ood"),
            "interventional": eval_metrics.get("interventional"),
            "complexity": eval_metrics.get("complexity"),
        },
        "losses": {
            "pred": train_metrics.get("loss/pred"),
            "vib": train_metrics.get("loss/vib"),
            "vib_kl": train_metrics.get("loss/vib_kl"),
            "vib_ll": train_metrics.get("loss/vib_ll"),
            "inv": train_metrics.get("loss/inv"),
            "mod": train_metrics.get("loss/mod"),
            "vq": train_metrics.get("loss/vq"),
            "vq_entropy_reg": train_metrics.get("loss/vq_entropy_reg"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="hmm")
    parser.add_argument("--model", type=str, default="hmm")
    parser.add_argument("--preset", type=str, default="quick")
    parser.add_argument("--beta_low", type=float, required=True)
    parser.add_argument("--beta_high", type=float, required=True)
    parser.add_argument("--lambda", dest="lam", type=float, default=0.0)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base_run_dir", type=str, default="runs/diagnostics")
    parser.add_argument("--overrides", type=str, nargs="*", default=[])
    args = parser.parse_args()

    out_dir = os.path.join(args.base_run_dir, args.env)
    os.makedirs(out_dir, exist_ok=True)

    runs = []
    runs.append(run_one(args.env, args.model, args.preset, os.path.join(out_dir, f"beta_{args.beta_low}"), args.beta_low, args.lam, args.mu, args.seed, args.overrides))
    runs.append(run_one(args.env, args.model, args.preset, os.path.join(out_dir, f"beta_{args.beta_high}"), args.beta_high, args.lam, args.mu, args.seed, args.overrides))

    output = {
        "env": args.env,
        "model": args.model,
        "beta_low": args.beta_low,
        "beta_high": args.beta_high,
        "lambda": args.lam,
        "mu": args.mu,
        "seed": args.seed,
        "runs": runs,
    }

    output_path = os.path.join(out_dir, "beta_extremes.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
