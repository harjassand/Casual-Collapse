# Environment Definitions

## HMM Causal Environment (`envs/hmm_causal_env/env.py`)

- Latent state `Z_t` is discrete with transition matrices per action.
- Observation `X_t` is a noisy one-hot of `Z_t` plus a spurious bit.
- Spurious bit correlates with `Z_t` in even `env_id` and flips in odd `env_id` when `spurious_flip_on_odd: true`.
- Interventions:
  - `{"set_latent": k}` or `{"type":"set_z","value":k}` fixes the latent state.
  - `{"set_spurious": b}` fixes the spurious bit.

Info fields include `env_id`, `t`, `latent_state`, `spurious_key/value`, `causal_key/value`, `intervention_active`, and `intervention_spec`.

## Object Micro-World (`envs/object_micro_world/env.py`)

- 2D objects with position, velocity, color; background is a binary attribute.
- Event label: whether object 0 crosses a boundary within a short horizon.
- Background is spuriously correlated with the event during training (even `env_id`) and flips at test when `spurious_flip_on_odd: true`.
- Interventions:
  - `{"set_pos": {"obj": i, "value": [x,y]}}`
  - `{"set_vel": {"obj": i, "value": [vx,vy]}}`
  - `{"set_background": b}`

Info fields include `env_id`, `t`, `latent_state`, `spurious_key/value` (background), `causal_key/value` (object state), `intervention_active`, and `intervention_spec`.

## Mechanism Shift Environment (`envs/mechanism_shift_env/env.py`)

- Two 1D objects with elastic collisions.
- Elasticity parameter differs across environments (even vs odd `env_id`).
- Interventions:
  - `{"set_pos": {"obj": i, "value": x}}`
  - `{"set_vel": {"obj": i, "value": v}}`

Info fields include `env_id`, `t`, `latent_state`, `spurious_key/value` (none), `causal_key/value` (elasticity), `intervention_active`, and `intervention_spec`.
