# Reproduction Guide

All experiments are configured via Hydra configs in `configs/`.

## Train (HMM baseline)

```
python3 training/train.py env=hmm model=hmm train.run_dir=runs/hmm_baseline train.num_future_classes=4
```

## Train (Object micro-world)

```
python3 training/train.py env=object model=default train.run_dir=runs/object_baseline train.num_future_classes=2
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
python3 training/train.py env=object model=default policy=active train.run_dir=runs/object_active train.num_future_classes=2
```

## Evaluate a checkpoint

```
python3 training/rollout_eval.py env=hmm model=hmm eval.ckpt_path=runs/hmm_baseline/model_1500.pt eval.output_path=runs/hmm_baseline/eval_metrics.json
```

## Sweep over (beta, lambda)

```
python3 training/run_sweep.py --betas 0.1 0.5 1.0 2.0 --lambdas 0.0 0.1 1.0 --mus 0.0 --base_run_dir runs/sweep
```

## Phase diagram + transition detection

```
python3 analysis/phase_diagram.py --runs_dir runs/sweep --output_path analysis/phase_diagram.json
```

## Causal-state alignment

```
python3 analysis/causal_state_alignment.py --config runs/hmm_baseline/config.json --ckpt runs/hmm_baseline/model_1500.pt --output_path analysis/causal_alignment.json
```

## One-shot causality test (mechanism shift)

```
python3 analysis/one_shot_test.py --config runs/mechanism_baseline/config.json --ckpt runs/mechanism_baseline/model_1500.pt --output_path analysis/one_shot.json
```

## Ablation matrix table

```
python3 analysis/ablation_matrix.py --runs_dir runs --output_path analysis/ablation_matrix.json
```

## Run full ablation suite (HMM defaults)

```
python3 training/run_ablations.py --env hmm --model hmm --base_run_dir runs/ablations
```
