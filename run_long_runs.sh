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

run_full_stack() {
  local env=$1
  local model=$2
  local run_dir=$3
  shift 3
  local overrides=("$@")

  if [ -f "${run_dir}/eval_metrics.json" ] && [ -f "${run_dir}/alignment_metrics.json" ] && [ -f "${run_dir}/one_shot_result.json" ]; then
    log "Skip existing run ${run_dir}"
    return
  fi

  log "Train ${run_dir}"
  "${PY}" training/train.py \
    "env=${env}" "model=${model}" "preset=default" "train.run_dir=${run_dir}" \
    "${overrides[@]}"

  local ckpt="${run_dir}/checkpoint.pt"
  eval_overrides=()
  for ov in "${overrides[@]}"; do
    case "${ov}" in
      seed=*) ;; # skip seed for eval
      *) eval_overrides+=("${ov}") ;;
    esac
  done
  log "Eval ${run_dir}"
  "${PY}" training/rollout_eval.py \
    "env=${env}" "model=${model}" "preset=default" \
    "eval.ckpt_path=${ckpt}" "eval.output_path=${run_dir}/eval_metrics.json" \
    "${eval_overrides[@]}"

  log "Alignment ${run_dir}"
  "${PY}" analysis/causal_state_alignment.py \
    "--config=${run_dir}/config.json" "--ckpt=${ckpt}" \
    "--output_path=${run_dir}/alignment_metrics.json"

  log "One-shot ${run_dir}"
  "${PY}" analysis/one_shot_test.py \
    "--config=${run_dir}/config.json" "--ckpt=${ckpt}" \
    "--output_path=${run_dir}/one_shot_result.json"

  log "Validate ${run_dir}"
  "${PY}" tools/validate_run_dir.py "${run_dir}"
}

log "Update HMM utilization sweep (seed 2)"
"${PY}" training/run_util_sweep.py --env hmm --model hmm --preset quick_noema \
  --beta 1.0 --lambda 1.0 \
  --mus 0.0 0.0001 0.001 0.01 0.1 1.0 \
  --entropy_regs 0.0 0.0001 0.001 0.01 0.1 \
  --use_ema 0 1 --seeds 0 1 2 \
  --base_run_dir runs/util_sweep_hmm

# HMM confirm points (lambda=0.1, betas=0.0/0.2/4.0)
HMM_LAM=0.1
HMM_PRE=0.0
HMM_TRANS=0.2
HMM_POST=4.0
HMM_MU=0.0
HMM_ENT=0.1
HMM_EMA=false

for seed in 0 1 2; do
  run_full_stack hmm hmm "runs/confirm_hmm/pre_beta_${HMM_PRE}_lambda_${HMM_LAM}_seed_${seed}" \
    "loss.beta=${HMM_PRE}" "loss.lambda=${HMM_LAM}" "loss.mu=${HMM_MU}" \
    "model.vq_use_ema=${HMM_EMA}" "vq.entropy_reg=${HMM_ENT}" \
    "vq.dead_code_reinit.enabled=true" "vq.dead_code_reinit.min_usage=0.01" \
    "vq.dead_code_reinit.window_steps=20" "vq.dead_code_reinit.strategy=sample_encoder" \
    "seed=${seed}"
  run_full_stack hmm hmm "runs/confirm_hmm/transition_beta_${HMM_TRANS}_lambda_${HMM_LAM}_seed_${seed}" \
    "loss.beta=${HMM_TRANS}" "loss.lambda=${HMM_LAM}" "loss.mu=${HMM_MU}" \
    "model.vq_use_ema=${HMM_EMA}" "vq.entropy_reg=${HMM_ENT}" \
    "vq.dead_code_reinit.enabled=true" "vq.dead_code_reinit.min_usage=0.01" \
    "vq.dead_code_reinit.window_steps=20" "vq.dead_code_reinit.strategy=sample_encoder" \
    "seed=${seed}"
  run_full_stack hmm hmm "runs/confirm_hmm/post_beta_${HMM_POST}_lambda_${HMM_LAM}_seed_${seed}" \
    "loss.beta=${HMM_POST}" "loss.lambda=${HMM_LAM}" "loss.mu=${HMM_MU}" \
    "model.vq_use_ema=${HMM_EMA}" "vq.entropy_reg=${HMM_ENT}" \
    "vq.dead_code_reinit.enabled=true" "vq.dead_code_reinit.min_usage=0.01" \
    "vq.dead_code_reinit.window_steps=20" "vq.dead_code_reinit.strategy=sample_encoder" \
    "seed=${seed}"
done

# Object confirm points (lambda=1.0, betas=0.0/1.0/4.0)
OBJ_LAM=1.0
OBJ_PRE=0.0
OBJ_TRANS=1.0
OBJ_POST=4.0
OBJ_MU=0.0
OBJ_ENT=0.0
OBJ_EMA=false

for seed in 0 1 2; do
  run_full_stack object object "runs/confirm_object/pre_beta_${OBJ_PRE}_lambda_${OBJ_LAM}_seed_${seed}" \
    "loss.beta=${OBJ_PRE}" "loss.lambda=${OBJ_LAM}" "loss.mu=${OBJ_MU}" \
    "model.vq_use_ema=${OBJ_EMA}" "vq.entropy_reg=${OBJ_ENT}" \
    "vq.dead_code_reinit.enabled=true" "vq.dead_code_reinit.min_usage=0.01" \
    "vq.dead_code_reinit.window_steps=20" "vq.dead_code_reinit.strategy=sample_encoder" \
    "seed=${seed}"
  run_full_stack object object "runs/confirm_object/transition_beta_${OBJ_TRANS}_lambda_${OBJ_LAM}_seed_${seed}" \
    "loss.beta=${OBJ_TRANS}" "loss.lambda=${OBJ_LAM}" "loss.mu=${OBJ_MU}" \
    "model.vq_use_ema=${OBJ_EMA}" "vq.entropy_reg=${OBJ_ENT}" \
    "vq.dead_code_reinit.enabled=true" "vq.dead_code_reinit.min_usage=0.01" \
    "vq.dead_code_reinit.window_steps=20" "vq.dead_code_reinit.strategy=sample_encoder" \
    "seed=${seed}"
  run_full_stack object object "runs/confirm_object/post_beta_${OBJ_POST}_lambda_${OBJ_LAM}_seed_${seed}" \
    "loss.beta=${OBJ_POST}" "loss.lambda=${OBJ_LAM}" "loss.mu=${OBJ_MU}" \
    "model.vq_use_ema=${OBJ_EMA}" "vq.entropy_reg=${OBJ_ENT}" \
    "vq.dead_code_reinit.enabled=true" "vq.dead_code_reinit.min_usage=0.01" \
    "vq.dead_code_reinit.window_steps=20" "vq.dead_code_reinit.strategy=sample_encoder" \
    "seed=${seed}"
done

run_ablation_variants() {
  local env=$1
  local model=$2
  local base_dir=$3
  local beta=$4
  local lambda=$5
  local ent=$6
  local ema=$7

  mkdir -p "${base_dir}"
  local common=(
    "loss.beta=${beta}" "loss.lambda=${lambda}" "loss.mu=0.0"
    "model.vq_use_ema=${ema}" "vq.entropy_reg=${ent}"
    "vq.dead_code_reinit.enabled=true" "vq.dead_code_reinit.min_usage=0.01"
    "vq.dead_code_reinit.window_steps=20" "vq.dead_code_reinit.strategy=sample_encoder"
    "seed=0"
  )

  run_full_stack "${env}" "${model}" "${base_dir}/predictive_only" \
    "loss.beta=0.0" "loss.lambda=0.0" "loss.mu=0.0" \
    "loss.use_rex=false" "loss.use_irm=false" "loss.modularity=none" "policy=random" \
    "${common[@]}"

  run_full_stack "${env}" "${model}" "${base_dir}/ib_only" \
    "loss.beta=${beta}" "loss.lambda=0.0" "loss.mu=0.0" \
    "loss.use_rex=false" "loss.use_irm=false" "loss.modularity=none" "policy=random" \
    "${common[@]}"

  run_full_stack "${env}" "${model}" "${base_dir}/invariance_rex_only" \
    "loss.beta=0.0" "loss.lambda=${lambda}" "loss.mu=0.0" \
    "loss.use_rex=true" "loss.use_irm=false" "loss.modularity=none" "policy=random" \
    "${common[@]}"

  run_full_stack "${env}" "${model}" "${base_dir}/invariance_irm_only" \
    "loss.beta=0.0" "loss.lambda=${lambda}" "loss.mu=0.0" \
    "loss.use_rex=false" "loss.use_irm=true" "loss.modularity=none" "policy=random" \
    "${common[@]}"

  run_full_stack "${env}" "${model}" "${base_dir}/ib_rex" \
    "loss.beta=${beta}" "loss.lambda=${lambda}" "loss.mu=0.0" \
    "loss.use_rex=true" "loss.use_irm=false" "loss.modularity=none" "policy=random" \
    "${common[@]}"

  run_full_stack "${env}" "${model}" "${base_dir}/ib_rex_mod" \
    "loss.beta=${beta}" "loss.lambda=${lambda}" "loss.mu=0.1" \
    "loss.use_rex=true" "loss.use_irm=false" "loss.modularity=total_correlation" "policy=random" \
    "${common[@]}"

  run_full_stack "${env}" "${model}" "${base_dir}/ib_rex_mod_active" \
    "loss.beta=${beta}" "loss.lambda=${lambda}" "loss.mu=0.1" \
    "loss.use_rex=true" "loss.use_irm=false" "loss.modularity=total_correlation" "policy=active" \
    "${common[@]}"

  run_full_stack "${env}" "${model}" "${base_dir}/operator_off" \
    "loss.beta=${beta}" "loss.lambda=${lambda}" "loss.mu=0.0" \
    "loss.use_rex=true" "loss.use_irm=false" "loss.modularity=none" "policy=random" \
    "train.enable_operator=false" "${common[@]}"

  "${PY}" analysis/ablation_matrix.py --runs_dir "${base_dir}" --output_path "${base_dir}/ablation_matrix.json"
}

log "Transition ablations (HMM)"
run_ablation_variants hmm hmm "runs/confirm_hmm/transition_ablation" "${HMM_TRANS}" "${HMM_LAM}" "${HMM_ENT}" "${HMM_EMA}"

log "Transition ablations (Object)"
run_ablation_variants object object "runs/confirm_object/transition_ablation" "${OBJ_TRANS}" "${OBJ_LAM}" "${OBJ_ENT}" "${OBJ_EMA}"

log "Mechanism modularity sweep"
"${PY}" training/run_sweep.py --env mechanism --model mechanism --preset default \
  --betas 1.0 --lambdas 1.0 --mus 0.0 0.01 0.1 1.0 \
  --base_run_dir runs/sweep_mechanism_modularity \
  --overrides "vq.entropy_reg=0.1" "vq.dead_code_reinit.enabled=true" \
              "vq.dead_code_reinit.min_usage=0.01" "vq.dead_code_reinit.window_steps=20" \
              "vq.dead_code_reinit.strategy=sample_encoder" "model.vq_use_ema=false"

"${PY}" analysis/phase_diagram.py --base_run_dir runs/sweep_mechanism_modularity \
  --output_path runs/sweep_mechanism_modularity/phase_diagram.json
"${PY}" analysis/modularity_report.py --base_run_dir runs/sweep_mechanism_modularity \
  --output_path runs/sweep_mechanism_modularity/modularity_report.json

log "Reports"
"${PY}" analysis/confirm_report.py --hmm_dir runs/confirm_hmm --object_dir runs/confirm_object \
  --output_path runs/reports/confirm_report.json
"${PY}" analysis/summary_report.py \
  --hmm_sweep runs/sweep_hmm_nondeg --object_sweep runs/sweep_object_nondeg --mechanism_sweep runs/sweep_mechanism_modularity \
  --util_hmm runs/util_sweep_hmm --util_object runs/util_sweep_object --util_mechanism runs/util_sweep_mechanism \
  --nondeg_hmm runs/sweep_hmm_nondeg --nondeg_object runs/sweep_object_nondeg --nondeg_mechanism runs/sweep_mechanism_modularity \
  --confirm_hmm runs/confirm_hmm --confirm_object runs/confirm_object --confirm_mechanism runs/confirm_mechanism \
  --output_path runs/reports/summary.json --output_md runs/reports/summary.md

log "Done"
