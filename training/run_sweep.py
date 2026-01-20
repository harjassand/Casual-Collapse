import argparse
import glob
import itertools
import json
import os
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--betas", type=float, nargs="+", required=True)
    parser.add_argument("--lambdas", type=float, nargs="+", required=True)
    parser.add_argument("--mus", type=float, nargs="+", default=[0.0])
    parser.add_argument("--base_run_dir", type=str, default="runs/sweep")
    args = parser.parse_args()

    os.makedirs(args.base_run_dir, exist_ok=True)

    for beta, lam, mu in itertools.product(args.betas, args.lambdas, args.mus):
        run_dir = os.path.join(args.base_run_dir, f"beta_{beta}_lambda_{lam}_mu_{mu}")
        cmd = [
            "python3",
            "training/train.py",
            f"train.run_dir={run_dir}",
            f"loss.beta={beta}",
            f"loss.lambda={lam}",
            f"loss.mu={mu}",
        ]
        subprocess.run(cmd, check=True)
        config_path = os.path.join(run_dir, "config.json")
        ckpt_path = os.path.join(run_dir, "checkpoint.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = None
        if ckpt_path is None and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            num_steps = int(cfg["train"]["num_steps"])
            interval = int(cfg["train"]["checkpoint_interval"])
            last_step = num_steps - (num_steps % interval)
            if last_step == 0:
                last_step = interval
            candidate = os.path.join(run_dir, f"model_{last_step}.pt")
            if os.path.exists(candidate):
                ckpt_path = candidate
        if ckpt_path is None:
            checkpoints = glob.glob(os.path.join(run_dir, "model_*.pt"))
            if checkpoints:
                ckpt_path = sorted(checkpoints)[-1]
        if ckpt_path is None:
            raise FileNotFoundError(f"No checkpoint found in {run_dir}")
        eval_cmd = [
            "python3",
            "training/rollout_eval.py",
            f"eval.ckpt_path={ckpt_path}",
            f"eval.output_path={os.path.join(run_dir, 'eval_metrics.json')}",
        ]
        subprocess.run(eval_cmd, check=True)


if __name__ == "__main__":
    main()
