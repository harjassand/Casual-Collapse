import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def best_from_ablation(path: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    notes = []
    if not os.path.exists(path):
        notes.append(f"missing ablation_matrix: {path}")
        return None, notes
    rows = load_json(path)
    best = None
    best_score = float("inf")
    for row in rows:
        ood = row.get("ood_mse")
        inter = row.get("interventional_mse")
        if ood is None:
            continue
        score = float(ood)
        if inter is not None:
            score += float(inter)
        if score < best_score:
            best_score = score
            best = row
    if best is None:
        notes.append("no valid rows for best config")
    return best, notes


def sweep_summary(run_dir: str) -> Tuple[Dict[str, Any], List[str]]:
    notes = []
    phase_path = os.path.join(run_dir, "phase_diagram.json")
    summary: Dict[str, Any] = {}
    if os.path.exists(phase_path):
        phase = load_json(phase_path)
        summary["transitions"] = phase.get("transitions", [])
        summary["quantile"] = phase.get("quantile")
        summary["beta_grid"] = phase.get("beta_grid")
        summary["lambda_grid"] = phase.get("lambda_grid")
        if not summary["transitions"]:
            notes.append("no transition detected")
    else:
        notes.append(f"missing phase_diagram: {phase_path}")

    max_active = 0
    max_nmi = None
    for name in os.listdir(run_dir) if os.path.exists(run_dir) else []:
        eval_path = os.path.join(run_dir, name, "eval_metrics.json")
        if not os.path.exists(eval_path):
            continue
        metrics = load_json(eval_path)
        active = metrics.get("complexity", {}).get("active_codes")
        if active is not None:
            max_active = max(max_active, int(active))
        align_path = os.path.join(run_dir, name, "alignment_metrics.json")
        if os.path.exists(align_path):
            align = load_json(align_path)
            nmi = align.get("NMI")
            if nmi is not None:
                max_nmi = max(float(nmi), max_nmi or float("-inf"))
    summary["max_active_codes"] = int(max_active)
    summary["max_nmi"] = max_nmi
    if max_active <= 1:
        notes.append("degenerate complexity (active_codes <= 1)")
    return summary, notes


def load_spurious(path: str) -> Optional[Dict[str, Any]]:
    if os.path.exists(path):
        return load_json(path)
    return None


def load_utilization_summary(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"path": path, "stable_configs": []}
    data = load_json(path)
    summary = data.get("summary", [])
    stable = []
    for row in summary:
        if row.get("active_codes_min", 0) >= 4 and (row.get("entropy_mean") or 0) > 0:
            stable.append(row)
    stable = sorted(
        stable,
        key=lambda r: (r.get("active_codes_mean") or 0, r.get("entropy_mean") or 0),
        reverse=True,
    )
    return {"path": path, "stable_configs": stable}


def load_confirm_report(path: str) -> Optional[Dict[str, Any]]:
    if os.path.exists(path):
        return load_json(path)
    return None


def load_modularity_report(path: str) -> Optional[Dict[str, Any]]:
    if os.path.exists(path):
        return load_json(path)
    return None


def build_report(env: str, ablation_dir: str, sweep_dir: str, spurious_path: str) -> Dict[str, Any]:
    ablation_path = os.path.join(ablation_dir, "ablation_matrix.json")
    best, notes = best_from_ablation(ablation_path)
    sweep, sweep_notes = sweep_summary(sweep_dir)
    notes.extend(sweep_notes)
    return {
        "env": env,
        "best_config": best,
        "sweep_summary": sweep,
        "spurious_probe": load_spurious(spurious_path),
        "notes": notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hmm_ablations", type=str, default="runs/ablations_hmm_quick")
    parser.add_argument("--object_ablations", type=str, default="runs/ablations_object_quick")
    parser.add_argument("--mechanism_ablations", type=str, default="runs/ablations_mechanism_quick")
    parser.add_argument("--hmm_sweep", type=str, default="runs/sweep_hmm_quick")
    parser.add_argument("--object_sweep", type=str, default="runs/sweep_object_quick")
    parser.add_argument("--mechanism_sweep", type=str, default="runs/sweep_mechanism_quick")
    parser.add_argument("--util_hmm", type=str, default="runs/util_sweep_hmm")
    parser.add_argument("--util_object", type=str, default="runs/util_sweep_object")
    parser.add_argument("--util_mechanism", type=str, default="runs/util_sweep_mechanism")
    parser.add_argument("--nondeg_hmm", type=str, default="runs/sweep_hmm_nondeg")
    parser.add_argument("--nondeg_object", type=str, default="runs/sweep_object_nondeg")
    parser.add_argument("--nondeg_mechanism", type=str, default="runs/sweep_mechanism_nondeg")
    parser.add_argument("--confirm_hmm", type=str, default="runs/confirm_hmm")
    parser.add_argument("--confirm_object", type=str, default="runs/confirm_object")
    parser.add_argument("--confirm_mechanism", type=str, default="runs/confirm_mechanism")
    parser.add_argument("--confirm_report", type=str, default="runs/reports/confirm_report.json")
    parser.add_argument("--modularity_report", type=str, default="runs/sweep_mechanism_modularity/modularity_report.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output_path", type=str, default="runs/reports/summary.json")
    parser.add_argument("--output_md", type=str, default="runs/reports/summary.md")
    args = parser.parse_args()

    quick_sweeps = {
        "hmm": "runs/sweep_hmm_quick",
        "object": "runs/sweep_object_quick",
        "mechanism": "runs/sweep_mechanism_quick",
    }

    report = {
        "hmm": build_report(
            "hmm",
            args.hmm_ablations,
            args.hmm_sweep,
            "runs/diagnostics/hmm/spurious_probe.json",
        ),
        "object": build_report(
            "object",
            args.object_ablations,
            args.object_sweep,
            "runs/diagnostics/object/spurious_probe.json",
        ),
        "mechanism": build_report(
            "mechanism",
            args.mechanism_ablations,
            args.mechanism_sweep,
            "runs/diagnostics/mechanism/spurious_probe.json",
        ),
        "utilization_sweeps": {
            "hmm": args.util_hmm,
            "object": args.util_object,
            "mechanism": args.util_mechanism,
        },
        "utilization_summary": {
            "hmm": load_utilization_summary(os.path.join(args.util_hmm, "utilization_report.json")),
            "object": load_utilization_summary(os.path.join(args.util_object, "utilization_report.json")),
            "mechanism": load_utilization_summary(os.path.join(args.util_mechanism, "utilization_report.json")),
        },
        "nondeg_sweeps": {
            "hmm": args.nondeg_hmm,
            "object": args.nondeg_object,
            "mechanism": args.nondeg_mechanism,
        },
        "confirm_runs": {
            "hmm": args.confirm_hmm,
            "object": args.confirm_object,
            "mechanism": args.confirm_mechanism,
        },
        "confirm_report": args.confirm_report,
        "modularity_report": args.modularity_report,
        "confirm_report_data": load_confirm_report(args.confirm_report),
        "modularity_report_data": load_modularity_report(args.modularity_report),
        "quick_sweeps": quick_sweeps,
        "seeds": args.seeds,
    }

    def hypothesis_status(env: str) -> str:
        sweep = report[env]["sweep_summary"]
        transitions = sweep.get("transitions", [])
        max_nmi = sweep.get("max_nmi")
        if not transitions:
            return "no transition detected"
        if max_nmi is None:
            return "transition detected; alignment missing"
        return "transition detected; alignment present"

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    lines = [
        "# Summary Report",
        "",
        "## Environment stress",
    ]
    for env in ("hmm", "object"):
        probe = report[env]["spurious_probe"] or {}
        lines.append(f"- {env}: train {probe.get('metric')}={probe.get('train_metric')} test {probe.get('metric')}={probe.get('test_metric')}")
    lines.append("")
    lines.append("## Degeneracy diagnosis")
    for env in ("hmm", "object", "mechanism"):
        sweep = report[env]["sweep_summary"]
        lines.append(f"- {env}: max_active_codes={sweep.get('max_active_codes')} transitions={len(sweep.get('transitions', []))}")
    for env, path in report["quick_sweeps"].items():
        if os.path.exists(path):
            quick, _ = sweep_summary(path)
            lines.append(f"- {env} (quick): max_active_codes={quick.get('max_active_codes')} transitions={len(quick.get('transitions', []))}")
    lines.append("")
    lines.append("## Utilization sweeps")
    for env, info in report["utilization_summary"].items():
        lines.append(f"- {env}: {info.get('path')}")
        stable = info.get("stable_configs", [])[:3]
        if stable:
            for row in stable:
                lines.append(
                    f"- {env} top: mu={row.get('mu')} ema={row.get('vq_use_ema')} ent={row.get('vq_entropy_reg')} "
                    f"active_min={row.get('active_codes_min')} entropy_mean={row.get('entropy_mean')}"
                )
        else:
            lines.append(f"- {env} top: no stable configs found")
    lines.append("")
    lines.append("## Non-degenerate sweeps")
    for env, path in report["nondeg_sweeps"].items():
        lines.append(f"- {env}: {path}")
    lines.append("")
    lines.append("## Confirmation runs")
    for env, path in report["confirm_runs"].items():
        lines.append(f"- {env}: {path}")
    confirm = report.get("confirm_report_data") or {}
    for env in ("hmm", "object"):
        summary = (confirm.get(env) or {}).get("summary", {})
        for point, stats in summary.items():
            lines.append(
                f"- {env} {point}: ood_mean={stats.get('ood_mean')} interventional_mean={stats.get('interventional_mean')} "
                f"active_codes_mean={stats.get('active_codes_mean')}"
            )
    lines.append("")
    lines.append("## Mechanism modularity")
    mod = report.get("modularity_report_data") or {}
    rows = mod.get("rows", [])
    if rows:
        for row in rows:
            lines.append(
                f"- mu={row.get('mu')} ood_mse={row.get('ood_mse')} "
                f"tc_in={row.get('mechanism_tc_in')} tc_ood={row.get('mechanism_tc_ood')} "
                f"probe_drift={row.get('probe_drift')}"
            )
    else:
        lines.append("- no modularity report found")
    lines.append("")
    lines.append("## Hypothesis status")
    for env in ("hmm", "object"):
        lines.append(f"- {env}: {hypothesis_status(env)}")
    lines.append("")
    lines.append("## Seeds")
    lines.append(f"- {report['seeds']}")

    os.makedirs(os.path.dirname(args.output_md), exist_ok=True)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
