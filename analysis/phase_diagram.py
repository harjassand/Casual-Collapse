import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np


def load_last_metric(path: str, key: str) -> float:
    last = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if key in record:
                last = record[key]
    return float(last) if last is not None else float("nan")


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument("--q", type=float, default=0.05)
    parser.add_argument("--output_path", type=str, default="analysis/phase_diagram.json")
    args = parser.parse_args()

    runs = []
    for name in os.listdir(args.runs_dir):
        run_dir = os.path.join(args.runs_dir, name)
        metrics_path = os.path.join(run_dir, "metrics.jsonl")
        config_path = os.path.join(run_dir, "config.json")
        eval_path = os.path.join(run_dir, "eval_metrics.json")
        if not os.path.exists(metrics_path) or not os.path.exists(config_path):
            continue
        cfg = load_config(config_path)
        beta = cfg["loss"]["beta"]
        lam = cfg["loss"]["lambda"]
        mu = cfg["loss"]["mu"]
        C = load_last_metric(metrics_path, "stats/entropy")
        if os.path.exists(eval_path):
            with open(eval_path, "r", encoding="utf-8") as f:
                eval_metrics = json.load(f)
            G = eval_metrics.get("ood/mse", float("nan"))
        else:
            G = load_last_metric(metrics_path, "loss/pred")
        runs.append({"beta": beta, "lambda": lam, "mu": mu, "C": C, "G": G})

    grouped: Dict[Tuple[float, float], List[Dict]] = {}
    for run in runs:
        key = (run["lambda"], run["mu"])
        grouped.setdefault(key, []).append(run)

    transitions = []
    thresholds = []
    for key, items in grouped.items():
        items = sorted(items, key=lambda x: x["beta"])
        deltas_c = [items[i + 1]["C"] - items[i]["C"] for i in range(len(items) - 1)]
        deltas_g = [items[i + 1]["G"] - items[i]["G"] for i in range(len(items) - 1)]
        if not deltas_c or not deltas_g:
            continue
        thresh_c = np.quantile(np.abs(deltas_c), 1 - args.q)
        thresh_g = np.quantile(np.abs(deltas_g), 1 - args.q)
        thresholds.append({
            "lambda": key[0],
            "mu": key[1],
            "threshold_c": float(thresh_c),
            "threshold_g": float(thresh_g),
        })
        for i, (dc, dg) in enumerate(zip(deltas_c, deltas_g)):
            if abs(dc) >= thresh_c and abs(dg) >= thresh_g:
                transitions.append({
                    "lambda": key[0],
                    "mu": key[1],
                    "beta": items[i + 1]["beta"],
                    "delta_C": float(dc),
                    "delta_G": float(dg),
                })

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(
            {"runs": runs, "transitions": transitions, "thresholds": thresholds, "quantile": args.q},
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
