# Mathematical Specification (Implementation Summary)

This document maps the project definitions to the concrete implementation in this repository. Symbols follow the project statement.

## Variables

- Observation at time t: `X_t` (structured vectors by default; images supported for the object micro-world).
- Environment index: `E` (`env_id` integer; even/odd parity flips spurious correlations).
- Representation: `S_t` is the slot-wise discrete code `C_t` with optional residual `R_t`.
- Future summary `T(X_{>t})`: discrete labels defined per environment and used for the collapse operator and causal alignment:
  - HMM: `T` is the latent state `Z_t` (integer).
  - Object micro-world: `T` is the binary event label (object-0 future boundary event).
  - Mechanism shift: `T` is a binary proximity label for the two objects.

## Predictive Information Bottleneck

Implemented in `losses/vib.py` and `training/train.py`:

- Variational KL term for discrete codes:
  - `KL(q(C|X) || p(C))`, with uniform prior over `M` codes, approximated using code usage entropy:
    - `KL ≈ log(M) - H(C)` with `H(C)` estimated from usage counts per batch.
- The training loop logs both the KL proxy and the predictive log-likelihood term separately.
- Predictive term:
  - `E[log p_theta(X_{>t} | S_t)]` approximated by negative MSE on multi-step predictions.

Combined VIB loss:

```
L_vib = KL(q(C|X)||p(C)) - beta * E[log p_theta(X_{>t}|S_t)]
```

## Invariance Penalties

Two penalties are implemented:

- REx (risk variance): `losses/invariance_rex.py`
- IRMv1 (gradient penalty): `losses/invariance_irm.py`

Both operate over per-environment risks computed in `training/train.py`.
IRM uses a fixed scalar head `w0 = 1.0` via the scale parameter trick in `irm_penalty`.

## Modularity Penalties

At least one modularity regularizer is implemented in `losses/modularity.py`:

- Cross-Jacobian penalty (default optional)
- Total correlation proxy (off-diagonal covariance)

## Full Objective

`training/train.py` combines:

```
L = L_pred
  + gamma * L_vib
  + lambda * R_inv
  + mu * R_mod
  + nu * L_logic
  + L_vq
```

Where `L_pred` is MSE on multi-step predictions and `L_vq` is the VQ-VAE codebook + commitment loss.

## Representation modes

Configured via `use_quantizer` and `use_residual` in model configs:

- Discrete-only: `use_quantizer: true`, `use_residual: false`
- Continuous-only: `use_quantizer: false`
- Multiscale: `use_quantizer: true`, `use_residual: true`

## Causal Collapse Operator

`collapse_operator/merge_split.py` uses discrete future distributions:

- Merge if JS divergence and environment-conditional JS are below thresholds.
- Split if within-code entropy is above threshold and the predicted NLL improves.

Thresholds are set in `configs/operator/default.yaml` and logged.
