import argparse
import itertools
import json
import os
import subprocess
import sys
from typing import Any, Dict, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="hmm")
    parser.add_argument("--model", type=str, default="hmm")
    parser.add_argument("--preset", type=str, default="quick")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0)
    parser.add_argument("--mus", type=float, nargs="+", required=True)
    parser.add_argument("--entropy_regs", type=float, nargs="+", required=True)
    parser.add_argument("--use_ema", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--base_run_dir", type=str, default="runs/util_sweep")
    parser.add_argument("--dead_enabled", type=int, nargs="+", default=[0])
    parser.add_argument("--dead_min_usage", type=float, default=0.01)
    parser.add_argument("--dead_window", type=int, default=20)
    parser.add_argument("--dead_strategy", type=str, default="sample_encoder")
    args = parser.parse_args()

    os.makedirs(args.base_run_dir, exist_ok=True)

    run_rows: List[Dict[str, Any]] = []
    for mu, ent, ema, dead_enabled, seed in itertools.product(args.mus, args.entropy_regs, args.use_ema, args.dead_enabled, args.seeds):
        ema_bool = bool(int(ema))
        dead_bool = bool(int(dead_enabled))
        run_name = f"mu_{mu}_ent_{ent}_ema_{int(ema_bool)}_dead_{int(dead_bool)}_seed_{seed}"
        run_dir = os.path.join(args.base_run_dir, run_name)
        overrides = [
            f"env={args.env}",
            f"model={args.model}",
            f"preset={args.preset}",
            f"train.run_dir={run_dir}",
            f"loss.beta={args.beta}",
            f"loss.lambda={args.lam}",
            f"loss.mu={mu}",
            f"vq.entropy_reg={ent}",
            f"vq.dead_code_reinit.enabled={dead_bool}",
            f"vq.dead_code_reinit.min_usage={args.dead_min_usage}",
            f"vq.dead_code_reinit.window_steps={args.dead_window}",
            f"vq.dead_code_reinit.strategy={args.dead_strategy}",
            f"model.vq_use_ema={ema_bool}",
            f"seed={seed}",
        ]
        eval_path = os.path.join(run_dir, "eval_metrics.json")
        if not os.path.exists(eval_path):
            train_cmd = [sys.executable, "training/train.py"] + overrides
            subprocess.run(train_cmd, check=True)

            ckpt_path = os.path.join(run_dir, "checkpoint.pt")
            eval_cmd = [
                sys.executable,
                "training/rollout_eval.py",
                f"env={args.env}",
                f"model={args.model}",
                f"preset={args.preset}",
                f"eval.ckpt_path={ckpt_path}",
                f"eval.output_path={eval_path}",
                f"model.vq_use_ema={ema_bool}",
                f"vq.entropy_reg={ent}",
            ]
            subprocess.run(eval_cmd, check=True)

        with open(eval_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        comp = metrics.get("complexity", {})
        run_rows.append({
            "run_dir": run_dir,
            "mu": mu,
            "vq_use_ema": ema_bool,
            "vq_entropy_reg": ent,
            "vq_dead_enabled": dead_bool,
            "seed": seed,
            "active_codes": comp.get("active_codes"),
            "perplexity": comp.get("perplexity"),
            "entropy": comp.get("entropy_or_proxy"),
            "in_distribution": metrics.get("in_distribution"),
            "ood": metrics.get("ood"),
            "interventional": metrics.get("interventional"),
        })

    summary_rows: List[Dict[str, Any]] = []
    for mu, ent, ema, dead_enabled in itertools.product(args.mus, args.entropy_regs, args.use_ema, args.dead_enabled):
        ema_bool = bool(int(ema))
        dead_bool = bool(int(dead_enabled))
        subset = [r for r in run_rows if r["mu"] == mu and r["vq_entropy_reg"] == ent and r["vq_use_ema"] == ema_bool and r["vq_dead_enabled"] == dead_bool]
        if not subset:
            continue
        active = [r["active_codes"] for r in subset if r["active_codes"] is not None]
        entropy = [r["entropy"] for r in subset if r["entropy"] is not None]
        perplex = [r["perplexity"] for r in subset if r["perplexity"] is not None]
        summary_rows.append({
            "mu": mu,
            "vq_use_ema": ema_bool,
            "vq_entropy_reg": ent,
            "vq_dead_enabled": dead_bool,
            "active_codes_mean": float(sum(active) / len(active)) if active else None,
            "active_codes_min": min(active) if active else None,
            "active_codes_max": max(active) if active else None,
            "entropy_mean": float(sum(entropy) / len(entropy)) if entropy else None,
            "entropy_var": float((sum((x - (sum(entropy) / len(entropy))) ** 2 for x in entropy) / len(entropy))) if entropy else None,
            "perplexity_mean": float(sum(perplex) / len(perplex)) if perplex else None,
            "perplexity_var": float((sum((x - (sum(perplex) / len(perplex))) ** 2 for x in perplex) / len(perplex))) if perplex else None,
            "seeds": args.seeds,
        })

    report = {
        "env": args.env,
        "model": args.model,
        "beta": args.beta,
        "lambda": args.lam,
        "mus": args.mus,
        "entropy_regs": args.entropy_regs,
        "use_ema": [bool(int(e)) for e in args.use_ema],
        "dead_enabled": [bool(int(e)) for e in args.dead_enabled],
        "seeds": args.seeds,
        "runs": run_rows,
        "summary": summary_rows,
    }

    report_path = os.path.join(args.base_run_dir, "utilization_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
