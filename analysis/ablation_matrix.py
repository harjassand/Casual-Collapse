import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


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
        alignment_path = os.path.join(run_dir, "alignment_metrics.json")
        one_shot_path = os.path.join(run_dir, "one_shot_result.json")
        alignment = {}
        one_shot = {}
        if os.path.exists(alignment_path):
            with open(alignment_path, "r", encoding="utf-8") as f:
                alignment = json.load(f)
        if os.path.exists(one_shot_path):
            with open(one_shot_path, "r", encoding="utf-8") as f:
                one_shot = json.load(f)
        rows.append({
            "run": name,
            "beta": cfg["loss"]["beta"],
            "lambda": cfg["loss"]["lambda"],
            "mu": cfg["loss"]["mu"],
            "use_irm": cfg["loss"]["use_irm"],
            "use_rex": cfg["loss"]["use_rex"],
            "modularity": cfg["loss"]["modularity"],
            "active": cfg["policy"]["type"] == "active",
            "operator": cfg["train"].get("enable_operator", True),
            "ib": cfg["loss"]["beta"] > 0.0,
            "in_mse": metrics.get("in/mse", None),
            "ood_mse": metrics.get("ood/mse", None),
            "interventional_mse": metrics.get("interventional/mse", None),
            "in_perplexity": metrics.get("in/perplexity", None),
            "ood_perplexity": metrics.get("ood/perplexity", None),
            "in_active_codes": metrics.get("in/active_codes", None),
            "ood_active_codes": metrics.get("ood/active_codes", None),
            "risk_variance": metrics.get("invariance/risk_variance", metrics.get("risk_variance", None)),
            "rep_usage_delta": metrics.get("stats/rep_usage_delta", None),
            "alignment_ari": alignment.get("ARI", alignment.get("ari")),
            "alignment_nmi": alignment.get("NMI", alignment.get("nmi")),
            "one_shot_loss_before": one_shot.get("loss_before"),
            "one_shot_loss_after": one_shot.get("loss_after"),
            "one_shot_loss_delta": one_shot.get("loss_delta"),
        })

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
