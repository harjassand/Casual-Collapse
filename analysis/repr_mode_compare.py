import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_one(env: str, model: str, preset: str, run_dir: str, repr_mode: str, beta: float, lam: float, mu: float, seed: int, overrides: List[str]) -> Dict[str, Any]:
    os.makedirs(run_dir, exist_ok=True)
    train_cmd = [
        sys.executable,
        "training/train.py",
        f"env={env}",
        f"model={model}",
        f"preset={preset}",
        f"train.run_dir={run_dir}",
        f"model.repr_mode={repr_mode}",
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
        f"model.repr_mode={repr_mode}",
    ] + overrides
    subprocess.run(eval_cmd, check=True)

    eval_metrics = load_json(os.path.join(run_dir, "eval_metrics.json"))
    return {
        "run_dir": run_dir,
        "repr_mode": repr_mode,
        "in_distribution": eval_metrics.get("in_distribution"),
        "ood": eval_metrics.get("ood"),
        "interventional": eval_metrics.get("interventional"),
        "complexity": eval_metrics.get("complexity"),
        "rep_usage_test": eval_metrics.get("rep_usage_test"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="hmm")
    parser.add_argument("--model", type=str, default="hmm")
    parser.add_argument("--preset", type=str, default="quick")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lambda", dest="lam", type=float, default=0.0)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base_run_dir", type=str, default="runs/diagnostics")
    parser.add_argument("--overrides", type=str, nargs="*", default=[])
    args = parser.parse_args()

    out_dir = os.path.join(args.base_run_dir, args.env)
    os.makedirs(out_dir, exist_ok=True)

    modes = ["continuous_only", "discrete_only", "multiscale"]
    rows = []
    for mode in modes:
        run_dir = os.path.join(out_dir, f"repr_{mode}")
        rows.append(run_one(args.env, args.model, args.preset, run_dir, mode, args.beta, args.lam, args.mu, args.seed, args.overrides))

    output = {
        "env": args.env,
        "model": args.model,
        "beta": args.beta,
        "lambda": args.lam,
        "mu": args.mu,
        "seed": args.seed,
        "rows": rows,
    }

    output_path = os.path.join(out_dir, "repr_mode_compare.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
