import argparse
import glob
import os
import glob
import json
import subprocess
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="hmm")
    parser.add_argument("--model", type=str, default="hmm")
    parser.add_argument("--base_run_dir", type=str, default="runs/ablations")
    parser.add_argument("--configs_dir", type=str, default="configs/ablation")
    args = parser.parse_args()

    os.makedirs(args.base_run_dir, exist_ok=True)

    for path in sorted(glob.glob(os.path.join(args.configs_dir, "*.yaml"))):
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        name = cfg["name"]
        overrides = cfg.get("overrides", [])
        run_dir = os.path.join(args.base_run_dir, name)
        cmd = [
            "python3",
            "training/train.py",
            f"env={args.env}",
            f"model={args.model}",
            f"train.run_dir={run_dir}",
        ] + overrides
        subprocess.run(cmd, check=True)
        config_path = os.path.join(run_dir, "config.json")
        ckpt_path = os.path.join(run_dir, "checkpoint.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = None
        if ckpt_path is None and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg_loaded = json.load(f)
            num_steps = int(cfg_loaded["train"]["num_steps"])
            interval = int(cfg_loaded["train"]["checkpoint_interval"])
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
            f"env={args.env}",
            f"model={args.model}",
            f"eval.ckpt_path={ckpt_path}",
            f"eval.output_path={run_dir}/eval_metrics.json",
        ]
        subprocess.run(eval_cmd, check=True)

        align_cmd = [
            "python3",
            "analysis/causal_state_alignment.py",
            f"--config={config_path}",
            f"--ckpt={ckpt_path}",
            f"--output_path={run_dir}/alignment_metrics.json",
        ]
        subprocess.run(align_cmd, check=True)

        one_shot_cmd = [
            "python3",
            "analysis/one_shot_test.py",
            f"--config={config_path}",
            f"--ckpt={ckpt_path}",
            f"--output_path={run_dir}/one_shot_result.json",
        ]
        subprocess.run(one_shot_cmd, check=True)

    matrix_path = os.path.join(args.base_run_dir, "ablation_matrix.json")
    matrix_cmd = [
        "python3",
        "analysis/ablation_matrix.py",
        f"--runs_dir={args.base_run_dir}",
        f"--output_path={matrix_path}",
    ]
    subprocess.run(matrix_cmd, check=True)


if __name__ == "__main__":
    main()
