# Causal Collapse Testbed

This repository implements an empirical testbed for “Causal Collapse” experiments:
learning discrete/structured representations under compression and invariance
pressures, with explicit causal-state alignment and interventional evaluation.

## What’s inside

- Environments: HMM causal process, object micro-world, mechanism shift
- Models: slot encoder, VQ quantizer, dynamics, decoder, optional logic layer
- Losses: VIB, invariance (REx/IRM), modularity, VQ losses
- Causal collapse operator: merge/split over code usage + predictive stats
- Evaluation: OOD, interventional, causal-state alignment, phase diagnostics
- Utilities: sweep runners, ablations, validators, summary reports

## Quick start

```bash
python3 -m pip install -r requirements.txt
python3 -m compileall -q .
python3 smoke_test.py
```

## Core workflows

```bash
# Self-check (end-to-end)
python3 self_check.py

# Spurious-only probe
python3 analysis/spurious_probe.py --env hmm --steps 500 --output_path runs/diagnostics/hmm/spurious_probe.json

# Utilization sweep
python3 training/run_util_sweep.py --env hmm --model hmm --preset quick \
  --beta 1.0 --lambda 1.0 \
  --mus 0.0 0.0001 0.001 0.01 0.1 1.0 \
  --entropy_regs 0.0 0.0001 0.001 0.01 0.1 \
  --use_ema 0 1 --dead_enabled 0 1 --seeds 0 1 2 \
  --base_run_dir runs/util_sweep_hmm

# Sweep (beta, lambda) with alignment
python3 training/run_sweep.py --env hmm --model hmm --preset quick \
  --betas 0.0 0.05 0.1 0.2 0.5 1.0 2.0 4.0 \
  --lambdas 0.0 0.1 1.0 10.0 --mus 0.0 \
  --base_run_dir runs/sweep_hmm --with_alignment
python3 analysis/phase_diagram.py --base_run_dir runs/sweep_hmm --output_path runs/sweep_hmm/phase_diagram.json
```

## Docs

- `docs/reproduction.md`: exact runbook and commands
- `docs/math_spec.md`: objective + parameter map
- `docs/envs.md`: environment definitions and interventions

## Artifacts

All experiment outputs are under `runs/`, including:

- `eval_metrics.json`
- `alignment_metrics.json`
- `one_shot_result.json`
- `phase_diagram.json`
- `ablation_matrix.json`
- `reports/summary.json` and `reports/summary.md`

## Validation

```bash
python3 tools/validate_run_dir.py <run_dir>
```

