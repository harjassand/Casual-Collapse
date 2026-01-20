import argparse
import glob
import os
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
        eval_cmd = [
            "python3",
            "training/rollout_eval.py",
            f"env={args.env}",
            f"model={args.model}",
            f"eval.ckpt_path={run_dir}/model_1500.pt",
            f"eval.output_path={run_dir}/eval_metrics.json",
        ]
        subprocess.run(eval_cmd, check=True)


if __name__ == "__main__":
    main()
