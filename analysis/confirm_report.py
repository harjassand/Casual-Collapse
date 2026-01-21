import argparse
import json
import os
from typing import Any, Dict, List, Optional


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def variance(values: List[float]) -> Optional[float]:
    if not values:
        return None
    mu = sum(values) / len(values)
    return float(sum((x - mu) ** 2 for x in values) / len(values))


def parse_point(name: str) -> str:
    if "_beta_" in name:
        return name.split("_beta_")[0]
    return "unknown"


def parse_seed(name: str) -> Optional[int]:
    if "_seed_" not in name:
        return None
    try:
        return int(name.split("_seed_")[-1])
    except ValueError:
        return None


def interventional_score(metrics: Dict[str, Any]) -> Optional[float]:
    inter = metrics.get("interventional", {})
    if "value" in inter:
        return float(inter["value"])
    details = inter.get("details", [])
    if not details:
        return None
    vals = []
    for item in details:
        if "mse" in item:
            vals.append(float(item["mse"]))
        elif "kl" in item:
            vals.append(float(item["kl"]))
    return mean(vals)


def collect_runs(base_dir: str) -> List[Dict[str, Any]]:
    rows = []
    if not os.path.exists(base_dir):
        return rows
    for name in sorted(os.listdir(base_dir)):
        run_dir = os.path.join(base_dir, name)
        if not os.path.isdir(run_dir):
            continue
        eval_path = os.path.join(run_dir, "eval_metrics.json")
        align_path = os.path.join(run_dir, "alignment_metrics.json")
        one_shot_path = os.path.join(run_dir, "one_shot_result.json")
        config_path = os.path.join(run_dir, "config.json")
        if not os.path.exists(eval_path) or not os.path.exists(config_path):
            continue
        metrics = load_json(eval_path)
        cfg = load_json(config_path)
        align = load_json(align_path) if os.path.exists(align_path) else {}
        one_shot = load_json(one_shot_path) if os.path.exists(one_shot_path) else {}
        rows.append({
            "name": name,
            "point": parse_point(name),
            "seed": parse_seed(name),
            "beta": cfg.get("loss", {}).get("beta"),
            "lambda": cfg.get("loss", {}).get("lambda"),
            "mu": cfg.get("loss", {}).get("mu"),
            "in_distribution": metrics.get("in_distribution", {}).get("value"),
            "ood": metrics.get("ood", {}).get("value"),
            "interventional": interventional_score(metrics),
            "active_codes": metrics.get("complexity", {}).get("active_codes"),
            "entropy": metrics.get("complexity", {}).get("entropy_or_proxy"),
            "ari": align.get("ARI"),
            "nmi": align.get("NMI"),
            "one_shot_delta": one_shot.get("loss_delta"),
        })
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_point: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_point.setdefault(row["point"], []).append(row)
    summary = {}
    for point, items in by_point.items():
        summary[point] = {
            "count": len(items),
            "seeds": sorted({r["seed"] for r in items if r["seed"] is not None}),
            "beta": items[0].get("beta"),
            "lambda": items[0].get("lambda"),
            "mu": items[0].get("mu"),
            "in_distribution_mean": mean([r["in_distribution"] for r in items if r["in_distribution"] is not None]),
            "in_distribution_var": variance([r["in_distribution"] for r in items if r["in_distribution"] is not None]),
            "ood_mean": mean([r["ood"] for r in items if r["ood"] is not None]),
            "ood_var": variance([r["ood"] for r in items if r["ood"] is not None]),
            "interventional_mean": mean([r["interventional"] for r in items if r["interventional"] is not None]),
            "interventional_var": variance([r["interventional"] for r in items if r["interventional"] is not None]),
            "active_codes_mean": mean([r["active_codes"] for r in items if r["active_codes"] is not None]),
            "entropy_mean": mean([r["entropy"] for r in items if r["entropy"] is not None]),
            "ari_mean": mean([r["ari"] for r in items if r["ari"] is not None]),
            "nmi_mean": mean([r["nmi"] for r in items if r["nmi"] is not None]),
            "one_shot_delta_mean": mean([r["one_shot_delta"] for r in items if r["one_shot_delta"] is not None]),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hmm_dir", type=str, default="runs/confirm_hmm")
    parser.add_argument("--object_dir", type=str, default="runs/confirm_object")
    parser.add_argument("--output_path", type=str, default="runs/reports/confirm_report.json")
    args = parser.parse_args()

    report = {}
    for env, path in (("hmm", args.hmm_dir), ("object", args.object_dir)):
        rows = collect_runs(path)
        report[env] = {
            "runs_dir": path,
            "rows": rows,
            "summary": summarize(rows),
        }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
