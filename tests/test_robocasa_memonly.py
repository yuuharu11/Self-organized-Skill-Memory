"""CPU-only contract checks for memory-only action assembly."""
import pathlib
import sys
import inspect

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skill_moe/skillnet/examples/robocasa"))
sys.path.insert(0, str(ROOT / "skill_moe/skillnet/examples/memory_pilot/robocasa"))
from serve_memonly_policy import (MemoryOnlyPolicy, append_history, clip_magnitude_multiplier,
                                  materialize_unit, unit_metadata)


def test_memory_only_action_assembly_contract():
    h = np.zeros((10, 12), np.float32)
    mask = np.zeros(10, bool)
    first = np.arange(5 * 12, dtype=np.float32).reshape(5, 12)
    h, mask = append_history(h, mask, first)
    assert mask.sum() == 5
    assert np.array_equal(h[-5:], first)
    second = (100 + np.arange(8 * 12, dtype=np.float32)).reshape(8, 12)
    h, mask = append_history(h, mask, second)
    assert mask.all()
    assert np.array_equal(h, np.concatenate((first[-2:], second), axis=0))
    x = np.arange(12 * 12, dtype=np.float32).reshape(12, 12)
    grip = np.array([-1, 1, -1, 1, -1], np.float32)
    y = materialize_unit(x, 2.0, grip)
    assert np.array_equal(y[:, :6], x[:, :6] * 2)
    assert np.array_equal(y[:5, 6], grip)
    assert np.array_equal(y[5:, 6], x[5:, 6])
    assert np.array_equal(y[:, 7:], x[:, 7:])
    rng = np.random.default_rng(4)
    pool = {"action10": rng.normal(size=(3, 10, 12)).astype(np.float32),
            "key": rng.normal(size=(3, 18)).astype(np.float32)}
    meta, magnitude = unit_metadata(pool)
    assert meta.shape == (3, 19)
    assert magnitude.shape == (3,)
    assert np.isfinite(meta).all()
    source = inspect.getsource(MemoryOnlyPolicy.infer)
    assert "sample_actions" not in source
    assert "sample_bridge" not in source
    assert "self.action_expert_calls" in source


def test_speed_ablation_multiplier_is_applied_before_clip():
    assert np.isclose(clip_magnitude_multiplier(.58, 1.5), .87)
    assert np.isclose(clip_magnitude_multiplier(.58, 2.0), 1.16)
    assert clip_magnitude_multiplier(3.0, 2.0) == 4.0
