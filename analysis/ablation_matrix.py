import argparse
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument("--output_path", type=str, default="analysis/ablation_matrix.json")
    args = parser.parse_args()

    rows = []
    for name in os.listdir(args.runs_dir):
        run_dir = os.path.join(args.runs_dir, name)
        config_path = os.path.join(run_dir, "config.json")
        eval_path = os.path.join(run_dir, "eval_metrics.json")
        if not os.path.exists(config_path) or not os.path.exists(eval_path):
            continue
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        with open(eval_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        rows.append({
            "run": name,
            "beta": cfg["loss"]["beta"],
            "lambda": cfg["loss"]["lambda"],
            "mu": cfg["loss"]["mu"],
            "use_irm": cfg["loss"]["use_irm"],
            "use_rex": cfg["loss"]["use_rex"],
            "modularity": cfg["loss"]["modularity"],
            "active": cfg["policy"]["type"] == "active",
            "in_mse": metrics.get("in/mse", None),
            "ood_mse": metrics.get("ood/mse", None),
            "interventional_mse": metrics.get("interventional/mse", None),
        })

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
