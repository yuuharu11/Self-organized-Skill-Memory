import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path('/work/.skillnet-state/worktrees/robocasa-memory/skill_moe/skillnet/examples/memory_pilot/robocasa/expert2_prep.py')
SPEC = importlib.util.spec_from_file_location('expert2_prep_test', SCRIPT)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_initial_history_mask_and_phase_contract():
    assert not mod._initial_mask(0).any()
    assert mod._initial_mask(4).sum() == 4
    assert mod._initial_mask(25).all()
    assert mod.T0S == (.1, .2, .3)


def test_scale_is_applied_to_motion_only_by_chain_contract():
    action = np.arange(12, dtype=np.float32)[None]
    action[:, 5:11] *= 2
    np.testing.assert_array_equal(action[:, :5], np.arange(5, dtype=np.float32)[None])
    np.testing.assert_array_equal(action[:, 11], np.array([11], np.float32))
    np.testing.assert_array_equal(action[:, 5:11], np.arange(5, 11, dtype=np.float32)[None] * 2)


def test_policy_storage_action_order_roundtrip():
    from experiment import action_policy_to_storage, action_storage_to_policy
    a = np.arange(12, dtype=np.float32).reshape(2, 6, 1).reshape(2, 6) if False else np.arange(24, dtype=np.float32).reshape(2, 12)
    np.testing.assert_array_equal(action_storage_to_policy(action_policy_to_storage(a)), a)

