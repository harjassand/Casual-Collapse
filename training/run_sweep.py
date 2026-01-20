import argparse
import itertools
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


if __name__ == "__main__":
    main()
