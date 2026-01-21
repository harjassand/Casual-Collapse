#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

PY=".venv_check/bin/python3"
if [ ! -x "${PY}" ]; then
  echo "Missing venv python at ${PY}. Activate your venv or update PY." >&2
  exit 1
fi

log() {
  echo "[$(date +%H:%M:%S)] $*"
}

run_beta_extremes() {
  local env=$1
  local model=$2
  log "Beta extremes ${env}"
  "${PY}" analysis/beta_extremes.py --env "${env}" --model "${model}" --preset quick \
    --beta_low 0.0 --beta_high 50.0 --lambda 0.0 --mu 0.0 \
    --base_run_dir runs/diagnostics \
    --overrides model.repr_mode=discrete_only model.vq_use_ema=false vq.entropy_reg=0.1 \
    vq.dead_code_reinit.enabled=true vq.dead_code_reinit.min_usage=0.01 vq.dead_code_reinit.window_steps=20 \
    vq.dead_code_reinit.strategy=sample_encoder
}

run_repr_compare() {
  local env=$1
  local model=$2
  log "Repr mode compare ${env}"
  "${PY}" analysis/repr_mode_compare.py --env "${env}" --model "${model}" --preset quick \
    --beta 1.0 --lambda 1.0 --mu 0.0 --base_run_dir runs/diagnostics \
    --overrides model.vq_use_ema=false vq.entropy_reg=0.1 \
    vq.dead_code_reinit.enabled=true vq.dead_code_reinit.min_usage=0.01 vq.dead_code_reinit.window_steps=20 \
    vq.dead_code_reinit.strategy=sample_encoder
}

run_util_sweep() {
  local env=$1
  local model=$2
  local out_dir=$3
  log "Util sweep ${env}"
  "${PY}" training/run_util_sweep.py --env "${env}" --model "${model}" --preset quick \
    --beta 1.0 --lambda 1.0 \
    --mus 0.0 0.0001 0.001 0.01 0.1 1.0 \
    --entropy_regs 0.0 0.0001 0.001 0.01 0.1 \
    --use_ema 0 1 --dead_enabled 0 1 --seeds 0 1 2 \
    --base_run_dir "${out_dir}" \
    --dead_min_usage 0.01 --dead_window 20 --dead_strategy sample_encoder \
    --repr_mode discrete_only
}

pick_cfg() {
  local report=$1
  "${PY}" - "$report" <<'PY'
import json, sys
path=sys.argv[1]
data=json.load(open(path))
rows=data.get("summary", [])
best=None
for row in rows:
    if row.get("active_codes_min", 0) < 4:
        continue
    ent=row.get("entropy_mean") or 0
    if ent <= 0:
        continue
    score = -(row.get("active_codes_mean") or 0) - ent
    if best is None or score < best[0]:
        best = (score, row)
if best is None:
    print("NO_STABLE")
else:
    row=best[1]
    print(f"{row['mu']} {row['vq_use_ema']} {row['vq_entropy_reg']} {row['vq_dead_enabled']}")
PY
}

run_sweep() {
  local env=$1
  local model=$2
  local util_dir=$3
  local out_dir=$4
  local cfg
  cfg=$(pick_cfg "${util_dir}/utilization_report.json")
  if [ "${cfg}" = "NO_STABLE" ]; then
    echo "No stable config found for ${env}" >&2
    exit 1
  fi
  read -r MU EMA ENT DEAD <<< "${cfg}"
  log "Sweep ${env} using mu=${MU} ema=${EMA} ent=${ENT} dead=${DEAD}"
  "${PY}" training/run_sweep.py --env "${env}" --model "${model}" --preset quick \
    --betas 0.0 0.05 0.1 0.2 0.5 1.0 2.0 4.0 \
    --lambdas 0.0 0.1 1.0 10.0 --mus 0.0 \
    --base_run_dir "${out_dir}" --with_alignment \
    --overrides model.repr_mode=discrete_only model.vq_use_ema=${EMA} vq.entropy_reg=${ENT} \
      vq.dead_code_reinit.enabled=${DEAD} vq.dead_code_reinit.min_usage=0.01 vq.dead_code_reinit.window_steps=20 \
      vq.dead_code_reinit.strategy=sample_encoder
  "${PY}" analysis/phase_diagram.py --base_run_dir "${out_dir}" \
    --output_path "${out_dir}/phase_diagram.json"
}

log "Diagnostics"
run_beta_extremes hmm hmm
run_beta_extremes object object
run_repr_compare hmm hmm
run_repr_compare object object

log "Utilization sweeps"
run_util_sweep hmm hmm runs/util_sweep_hmm_label
run_util_sweep object object runs/util_sweep_object_label

log "Non-degenerate sweeps"
run_sweep hmm hmm runs/util_sweep_hmm_label runs/sweep_hmm_nondeg_label
run_sweep object object runs/util_sweep_object_label runs/sweep_object_nondeg_label

log "Summary report"
"${PY}" analysis/summary_report.py \
  --hmm_sweep runs/sweep_hmm_nondeg_label --object_sweep runs/sweep_object_nondeg_label --mechanism_sweep runs/sweep_mechanism_modularity \
  --util_hmm runs/util_sweep_hmm_label --util_object runs/util_sweep_object_label --util_mechanism runs/util_sweep_mechanism \
  --nondeg_hmm runs/sweep_hmm_nondeg_label --nondeg_object runs/sweep_object_nondeg_label --nondeg_mechanism runs/sweep_mechanism_modularity \
  --confirm_hmm runs/confirm_hmm --confirm_object runs/confirm_object --confirm_mechanism runs/confirm_mechanism \
  --confirm_report runs/reports/confirm_report.json --modularity_report runs/sweep_mechanism_modularity/modularity_report.json \
  --output_path runs/reports/summary.json --output_md runs/reports/summary.md

log "Done"
