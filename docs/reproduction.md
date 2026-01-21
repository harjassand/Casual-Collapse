# Reproduction Guide

All experiments are configured via Hydra configs in `configs/`.

## Install dependencies

```
python3 -m pip install -r requirements.txt
```

## Static checks

```
python3 -m compileall -q .
```

## Smoke test

```
python3 smoke_test.py
```

## Self-check (full end-to-end)

```
python3 self_check.py
```

## Spurious correlation checks

```
python3 tools/check_spurious_correlation.py --env hmm --steps 200
python3 tools/check_spurious_correlation.py --env object --steps 200
python3 tools/check_spurious_correlation.py --env mechanism --steps 200
```

## Spurious-only probe

```
python3 analysis/spurious_probe.py --env hmm --steps 500 --output_path runs/diagnostics/hmm/spurious_probe.json
python3 analysis/spurious_probe.py --env object --steps 500 --output_path runs/diagnostics/object/spurious_probe.json
python3 analysis/spurious_probe.py --env mechanism --steps 500 --output_path runs/diagnostics/mechanism/spurious_probe.json
```

## Beta extremes diagnostic

```
python3 analysis/beta_extremes.py --env hmm --model hmm --preset quick \
  --beta_low 0.0 --beta_high 50.0 --lambda 0.0 --mu 0.0 \
  --base_run_dir runs/diagnostics
python3 analysis/beta_extremes.py --env object --model object --preset quick \
  --beta_low 0.0 --beta_high 50.0 --lambda 0.0 --mu 0.0 \
  --base_run_dir runs/diagnostics
```

## Representation mode compare (continuous vs discrete vs multiscale)

```
python3 analysis/repr_mode_compare.py --env hmm --model hmm --preset quick \
  --beta 1.0 --lambda 1.0 --mu 0.0 --base_run_dir runs/diagnostics
python3 analysis/repr_mode_compare.py --env object --model object --preset quick \
  --beta 1.0 --lambda 1.0 --mu 0.0 --base_run_dir runs/diagnostics
```

## Utilization sweep (VQ usage stability)

```
python3 training/run_util_sweep.py --env hmm --model hmm --preset quick \
  --beta 1.0 --lambda 1.0 \
  --mus 0.0 0.0001 0.001 0.01 0.1 1.0 \
  --entropy_regs 0.0 0.0001 0.001 0.01 0.1 \
  --use_ema 0 1 --dead_enabled 0 1 --seeds 0 1 2 \
  --base_run_dir runs/util_sweep_hmm
```

## Train (HMM baseline)

```
python3 training/train.py env=hmm model=hmm train.run_dir=runs/hmm_baseline train.num_future_classes=4
```

This writes `config_snapshot.yaml`, `train_metrics.json`, `code_usage.json`, and (if enabled) `operator_events.json` into the run directory.

## Train (Object micro-world)

```
python3 training/train.py env=object model=object train.run_dir=runs/object_baseline train.num_future_classes=2
```

## Train (Mechanism shift)

```
python3 training/train.py env=mechanism model=mechanism train.run_dir=runs/mechanism_baseline train.num_future_classes=2
```

## Enable invariance + modularity

```
python3 training/train.py env=hmm model=hmm loss.use_rex=true loss.lambda=1.0 loss.modularity=total_correlation loss.mu=0.1 train.run_dir=runs/hmm_inv_mod
```

## Active interventions (information-gain policy)

```
python3 training/train.py env=object model=object policy=active train.run_dir=runs/object_active train.num_future_classes=2
```

## Evaluate a checkpoint

```
python3 training/rollout_eval.py env=hmm model=hmm eval.ckpt_path=runs/hmm_baseline/checkpoint.pt eval.output_path=runs/hmm_baseline/eval_metrics.json
```

## Sweep over (beta, lambda)

```
python3 training/run_sweep.py --env hmm --model hmm --preset quick --betas 0.1 0.5 1.0 2.0 --lambdas 0.0 0.1 1.0 --mus 0.0 --base_run_dir runs/sweep --with_alignment
```

## Phase diagram + transition detection

```
python3 analysis/phase_diagram.py --base_run_dir runs/sweep --output_path analysis/phase_diagram.json
```

## Causal-state alignment

```
python3 analysis/causal_state_alignment.py --config runs/hmm_baseline/config.json --ckpt runs/hmm_baseline/checkpoint.pt --output_path runs/hmm_baseline/alignment_metrics.json
```

## One-shot causality test (mechanism shift)

```
python3 analysis/one_shot_test.py --config runs/mechanism_baseline/config.json --ckpt runs/mechanism_baseline/checkpoint.pt --output_path runs/mechanism_baseline/one_shot_result.json
```

## Ablation matrix table

```
python3 analysis/ablation_matrix.py --base_run_dir runs --output_path analysis/ablation_matrix.json
```

## Run full ablation suite (HMM defaults)

```
python3 training/run_ablations.py --env hmm --model hmm --preset quick --base_run_dir runs/ablations
```

This produces `runs/ablations/ablation_matrix.json` via `analysis/ablation_matrix.py`.

## Validate a run directory

```
python3 validate_run_dir.py runs/hmm_baseline
```

## Summary report

```
python3 analysis/summary_report.py --output_path runs/reports/summary.json --output_md runs/reports/summary.md
```
