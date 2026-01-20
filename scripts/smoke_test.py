import json

import numpy as np
import torch

from envs import HMMCausalEnv, ObjectMicroWorldEnv, MechanismShiftEnv
from models.causal_collapse_model import CausalCollapseModel


REQUIRED_INFO_KEYS = {
    "env_id",
    "t",
    "latent_state",
    "intervention_active",
    "intervention_spec",
    "spurious_key",
    "spurious_value",
    "causal_key",
    "causal_value",
}


def check_info(info):
    missing = REQUIRED_INFO_KEYS - set(info.keys())
    if missing:
        raise AssertionError(f"Missing info keys: {missing}")


def run_env(env, intervention):
    obs = env.reset(seed=123, env_id=0)
    for _ in range(5):
        obs, _, _, info = env.step(None)
        check_info(info)
    env.do_intervention(intervention)
    obs, _, _, info = env.step(None)
    check_info(info)
    gt = env.get_ground_truth()
    missing = REQUIRED_INFO_KEYS - set(gt.keys())
    if missing:
        raise AssertionError(f"Missing ground truth keys: {missing}")
    return obs


def main() -> None:
    hmm = HMMCausalEnv()
    obj = ObjectMicroWorldEnv()
    mech = MechanismShiftEnv()

    run_env(hmm, {"set_latent": 1, "duration": 1})
    run_env(obj, {"set_pos": {"obj": 0, "value": [0.5, 0.5]}, "duration": 1})
    run_env(mech, {"set_vel": {"obj": 0, "value": 0.5}, "duration": 1})

    model_cfg = {
        "input_type": "structured",
        "obs_dim": 6,
        "num_slots": 1,
        "slot_dim": 8,
        "num_codes": 4,
        "use_quantizer": True,
        "commitment_weight": 0.25,
        "vq_decay": 0.99,
        "vq_use_ema": True,
        "use_graph": False,
        "use_residual": True,
        "residual_dim": 8,
        "action_dim": 0,
        "dyn_hidden": 32,
        "recon": False,
        "use_logic": False,
        "logic_predicates": [],
        "logic_constraints": [],
    }
    model = CausalCollapseModel(model_cfg)
    obs = torch.zeros(1, 1, model_cfg["obs_dim"], dtype=torch.float32)
    out = model(obs, None, horizon=2)
    assert out["preds"].shape[1] == 2

    print(json.dumps({"status": "ok"}))


if __name__ == "__main__":
    main()
