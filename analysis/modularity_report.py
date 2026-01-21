import argparse
import json
import os
from typing import Any, Dict, List


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_run_dir", type=str, default="runs/sweep_mechanism_modularity")
    parser.add_argument("--output_path", type=str, default="runs/sweep_mechanism_modularity/modularity_report.json")
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    if os.path.exists(args.base_run_dir):
        for name in sorted(os.listdir(args.base_run_dir)):
            run_dir = os.path.join(args.base_run_dir, name)
            eval_path = os.path.join(run_dir, "eval_metrics.json")
            cfg_path = os.path.join(run_dir, "config.json")
            if not os.path.exists(eval_path) or not os.path.exists(cfg_path):
                continue
            metrics = load_json(eval_path)
            cfg = load_json(cfg_path)
            rows.append({
                "run_dir": run_dir,
                "beta": cfg.get("loss", {}).get("beta"),
                "lambda": cfg.get("loss", {}).get("lambda"),
                "mu": cfg.get("loss", {}).get("mu"),
                "in_mse": metrics.get("in_distribution", {}).get("value"),
                "ood_mse": metrics.get("ood", {}).get("value"),
                "in_total_correlation": metrics.get("in/total_correlation"),
                "ood_total_correlation": metrics.get("ood/total_correlation"),
                "mechanism_tc_in": metrics.get("mechanism/total_correlation_in"),
                "mechanism_tc_ood": metrics.get("mechanism/total_correlation_ood"),
                "probe_drift": metrics.get("invariance/probe_drift"),
            })

    report = {
        "base_run_dir": args.base_run_dir,
        "rows": rows,
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
