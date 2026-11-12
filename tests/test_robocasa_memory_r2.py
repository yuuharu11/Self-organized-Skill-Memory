import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skill_moe/skillnet/examples/robocasa"))
sys.path.insert(0, str(ROOT / "skill_moe/skillnet/examples/memory_pilot/robocasa"))
from serve_memory_policy import MemoryBridgePolicy, _context_net_class  # noqa: E402
import experiment  # noqa: E402


def test_r2_selects_positive_s_shared_type_and_rejects_negative():
    ref = np.zeros((50, 12), np.float32)
    ref[:10, 0] = np.linspace(0.1, 1.0, 10)
    query_storage = experiment.action_policy_to_storage(ref[:10])
    query_shape, _ = experiment.encode_shape(query_storage)
    policy = MemoryBridgePolicy.__new__(MemoryBridgePolicy)
    policy.context_pool = {
        "length": np.array([10, 10], np.int32),
        "shape": np.stack([query_shape, -query_shape]),
    }
    policy.context_norm = {
        "feature_mean": np.zeros(2048, np.float32),
        "feature_std": np.ones(2048, np.float32),
        "state_mean": np.zeros(16, np.float32),
        "state_std": np.ones(16, np.float32),
    }
    policy.context_net = _context_net_class()(2064).eval()
    policy.p5_device = "cpu"
    selected, score, _, dot, error = policy._r2_select(ref, np.zeros(2048, np.float32), np.zeros(16, np.float32))
    assert selected == 0
    assert dot > 0
    assert score <= 0
    assert error < 1e-6
