import argparse
import glob
import itertools
import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="hmm")
    parser.add_argument("--model", type=str, default="hmm")
    parser.add_argument("--betas", type=float, nargs="+", required=True)
    parser.add_argument("--lambdas", type=float, nargs="+", required=True)
    parser.add_argument("--mus", type=float, nargs="+", default=[0.0])
    parser.add_argument("--base_run_dir", type=str, default="runs/sweep")
    parser.add_argument("--preset", type=str, default="default")
    parser.add_argument("--overrides", type=str, nargs="*", default=[])
    parser.add_argument("--with_alignment", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.base_run_dir, exist_ok=True)

    for beta, lam, mu in itertools.product(args.betas, args.lambdas, args.mus):
        run_dir = os.path.join(args.base_run_dir, f"beta_{beta}_lambda_{lam}_mu_{mu}")
        cmd = [
            sys.executable,
            "training/train.py",
            f"env={args.env}",
            f"model={args.model}",
            f"preset={args.preset}",
            f"train.run_dir={run_dir}",
            f"loss.beta={beta}",
            f"loss.lambda={lam}",
            f"loss.mu={mu}",
        ] + args.overrides
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
            sys.executable,
            "training/rollout_eval.py",
            f"env={args.env}",
            f"model={args.model}",
            f"preset={args.preset}",
            f"eval.ckpt_path={ckpt_path}",
            f"eval.output_path={os.path.join(run_dir, 'eval_metrics.json')}",
        ] + args.overrides
        subprocess.run(eval_cmd, check=True)
        if args.with_alignment:
            align_cmd = [
                sys.executable,
                "analysis/causal_state_alignment.py",
                f"--config={config_path}",
                f"--ckpt={ckpt_path}",
                f"--output_path={os.path.join(run_dir, 'alignment_metrics.json')}",
            ]
            subprocess.run(align_cmd, check=True)


if __name__ == "__main__":
    main()
