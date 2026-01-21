#!/usr/bin/env bash
set -e

source .venv_check/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# Wait for utilization reports
while [ ! -f runs/util_sweep_hmm/utilization_report.json ] || [ ! -f runs/util_sweep_object/utilization_report.json ]; do
  echo "Waiting for utilization reports..."
  sleep 60
done

# Pick stable config from utilization report
pick_cfg() {
  python3 - "$1" <<'PY'
import json, sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
cands = []
for row in data.get("summary", []):
    if (row.get("active_codes_min") or 0) >= 4 and (row.get("entropy_mean") or 0) > 0:
        cands.append(row)
if not cands:
    print("NONE")
    sys.exit(0)
cands.sort(key=lambda r: (r.get("active_codes_mean",0), r.get("entropy_mean",0)), reverse=True)
best = cands[0]
print(f"{best['mu']} {best['vq_use_ema']} {best['vq_entropy_reg']}")
PY
}

HMM_CFG=$(pick_cfg runs/util_sweep_hmm/utilization_report.json)
OBJ_CFG=$(pick_cfg runs/util_sweep_object/utilization_report.json)

if [ "$HMM_CFG" = "NONE" ] || [ "$OBJ_CFG" = "NONE" ]; then
  echo "No stable non-degenerate config found. Stop."
  exit 1
fi

read HMM_MU HMM_EMA HMM_ENT <<< "$HMM_CFG"
read OBJ_MU OBJ_EMA OBJ_ENT <<< "$OBJ_CFG"

# Non-degenerate sweeps (HMM + Object)
python3 training/run_sweep.py --env hmm --model hmm --preset quick \
  --betas 0.0 0.05 0.1 0.2 0.5 1.0 2.0 4.0 \
  --lambdas 0.0 0.1 1.0 10.0 --mus ${HMM_MU} \
  --base_run_dir runs/sweep_hmm_nondeg \
  --overrides vq.entropy_reg=${HMM_ENT} vq.dead_code_reinit.enabled=true \
             vq.dead_code_reinit.min_usage=0.01 vq.dead_code_reinit.window_steps=20 \
             model.vq_use_ema=${HMM_EMA}

python3 analysis/phase_diagram.py --base_run_dir runs/sweep_hmm_nondeg --output_path runs/sweep_hmm_nondeg/phase_diagram.json

python3 training/run_sweep.py --env object --model object --preset quick \
  --betas 0.0 0.05 0.1 0.2 0.5 1.0 2.0 4.0 \
  --lambdas 0.0 0.1 1.0 10.0 --mus ${OBJ_MU} \
  --base_run_dir runs/sweep_object_nondeg \
  --overrides vq.entropy_reg=${OBJ_ENT} vq.dead_code_reinit.enabled=true \
             vq.dead_code_reinit.min_usage=0.01 vq.dead_code_reinit.window_steps=20 \
             model.vq_use_ema=${OBJ_EMA}

python3 analysis/phase_diagram.py --base_run_dir runs/sweep_object_nondeg --output_path runs/sweep_object_nondeg/phase_diagram.json

# Summary report
python3 analysis/summary_report.py --output_path runs/reports/summary.json --output_md runs/reports/summary.md
