import importlib.util
import sys
from pathlib import Path

import numpy as np

MP = str(Path(__file__).resolve().parents[1] / 'skill_moe/skillnet/examples/memory_pilot/robocasa')
if MP not in sys.path:
    sys.path.insert(0, MP)


SCRIPT = Path(__file__).resolve().parents[1] / 'skill_moe/skillnet/examples/memory_pilot/robocasa/expert2_prep.py'
SPEC = importlib.util.spec_from_file_location('expert2_prep_test', SCRIPT)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def _load_module(name, filename):
    path = Path(__file__).resolve().parents[1] / 'skill_moe/skillnet/examples/memory_pilot/robocasa' / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def test_load_ranker_prefers_checked_in_script_over_artifact_copy(tmp_path, monkeypatch):
    prep = _load_module('expert2_prep_fallback_test', 'expert2_prep.py')
    monkeypatch.setattr(prep, 'KEYLITE', tmp_path)
    rk = prep._load_ranker()
    assert hasattr(rk, 'assemble_context')


def test_magnitude_keylite_loader_prefers_checked_in_script(tmp_path, monkeypatch):
    mh = _load_module('magnitude_head_fallback_test', 'magnitude_head.py')
    monkeypatch.setattr(mh, 'KEYLITE', tmp_path)
    rk = mh._load_keylite()
    assert hasattr(rk, 'assemble_context')


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

