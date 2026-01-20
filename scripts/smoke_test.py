import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

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
    info_keys_ok = True
    intervention_logged = False
    for _ in range(5):
        obs, _, _, info = env.step(None)
        check_info(info)
    env.do_intervention(intervention)
    obs, _, _, info = env.step(None)
    check_info(info)
    info_keys_ok = True
    intervention_logged = bool(info.get("intervention_active")) and info.get("intervention_spec")
    gt = env.get_ground_truth()
    missing = REQUIRED_INFO_KEYS - set(gt.keys())
    if missing:
        raise AssertionError(f"Missing ground truth keys: {missing}")
    return obs, info_keys_ok, intervention_logged


def main() -> None:
    hmm = HMMCausalEnv()
    obj = ObjectMicroWorldEnv()
    mech = MechanismShiftEnv()

    hmm_obs, hmm_keys_ok, hmm_int = run_env(hmm, {"set_latent": 1, "duration": 1})
    obj_obs, obj_keys_ok, obj_int = run_env(obj, {"set_pos": {"obj": 0, "value": [0.5, 0.5]}, "duration": 1})
    mech_obs, mech_keys_ok, mech_int = run_env(mech, {"set_vel": {"obj": 0, "value": 0.5}, "duration": 1})

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
    rollout_ok = out["preds"].shape[1] == 2

    result = {
        "status": "ok",
        "envs": {
            "hmm": {"info_keys_ok": hmm_keys_ok, "intervention_logged": hmm_int},
            "object": {"info_keys_ok": obj_keys_ok, "intervention_logged": obj_int},
            "mechanism": {"info_keys_ok": mech_keys_ok, "intervention_logged": mech_int},
        },
        "model_forward_ok": bool(rollout_ok),
        "rollout_ok": bool(rollout_ok),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
