# Environment Definitions

## HMM Causal Environment (`envs/hmm_causal_env/env.py`)

- Latent state `Z_t` is discrete with transition matrices per action.
- Observation `X_t` is a noisy one-hot of `Z_t` plus a spurious bit.
- Spurious bit correlates with `Z_t` in even `env_id` and flips in odd `env_id`.
- Interventions:
  - `{"set_latent": k}` fixes the latent state.
  - `{"set_spurious": b}` fixes the spurious bit.

## Object Micro-World (`envs/object_micro_world/env.py`)

- 2D objects with position, velocity, color; background is a binary attribute.
- Event label: whether object 0 crosses a boundary within a short horizon.
- Background is spuriously correlated with the event during training (even `env_id`) and flips at test.
- Interventions:
  - `{"set_pos": {"obj": i, "value": [x,y]}}`
  - `{"set_vel": {"obj": i, "value": [vx,vy]}}`
  - `{"set_background": b}`

## Mechanism Shift Environment (`envs/mechanism_shift_env/env.py`)

- Two 1D objects with elastic collisions.
- Elasticity parameter differs across environments (even vs odd `env_id`).
- Interventions:
  - `{"set_pos": {"obj": i, "value": x}}`
  - `{"set_vel": {"obj": i, "value": v}}`
