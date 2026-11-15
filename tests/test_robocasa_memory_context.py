import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skill_moe/skillnet/examples/memory_pilot/robocasa"))
import context_key


def test_pi05_eval_noise_matches_gripper01_query_seed_rule():
    import jax

    row = {"task": 2, "episode": 33, "frame": 70}
    seed_value = row["task"] * 1_000_000 + row["episode"] * 1000 + row["frame"]
    expected = jax.random.normal(
        jax.random.fold_in(jax.random.key(4101), seed_value), (1, 50, 32))
    np.testing.assert_array_equal(np.asarray(context_key._pi05_eval_noise(row)), np.asarray(expected))


def test_old_zero_noise_action_cache_is_rejected(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "feature_manifest.json").write_text(
        '{"evaluation_actions_noise":"zero noise, 10-step pi05 sampler"}')
    with pytest.raises(ValueError, match="different noise rule"):
        context_key._require_current_pi05_noise_manifest(tmp_path)

