import json

import numpy as np

from skill_moe.skillnet.examples.emergent.light.extract_libero10_features import split_initial_states
from skill_moe.skillnet.examples.emergent.light.libero10_features import (
    _action_chunks,
    _motion,
    initial_state_id,
    join_existing_and_libero10,
)


def test_split_is_seeded_and_keeps_initial_state_groups_together():
    rows = [{"episode_index": i, "task_index": i // 8, "init_state_id": f"t{i // 8}-s{i % 8}"}
            for i in range(16)]
    first = split_initial_states(rows, seed=73)
    assert first == split_initial_states(rows, seed=73)
    assert set(first.values()) == {"store", "query"}
    assert sum(v == "query" for v in first.values()) == 4


def test_action_windows_motion_and_state_group_id():
    action = np.arange(30, dtype=np.float32).reshape(10, 3)
    chunks, mask = _action_chunks(action, np.array([0, 5, 10]), 4)
    assert chunks.shape == (3, 4, 3) and mask.shape == (3, 4)
    np.testing.assert_array_equal(chunks[1], action[5:9])
    assert mask[2].sum() == 0

    state = np.zeros((10, 8), np.float32)
    state[:, 0] = np.arange(10) * 0.1
    motion, motion_mask = _motion(state, np.array([0, 5, 9]), 3)
    assert motion.shape == (3, 3, 9)
    assert motion_mask.tolist() == [[True] * 3, [True] * 3, [False] * 3]
    np.testing.assert_allclose(motion[0, :, 3:7], np.array([[0, 0, 0, 1]] * 3), atol=1e-6)
    assert initial_state_id(state[0], 2) == initial_state_id(state[0].copy(), 2)
    assert initial_state_id(state[0], 2) != initial_state_id(state[0], 3)


def _write_cache(root, name, *, task_index, split=None, camera=False):
    feature = root / "feature-cache"
    epdir = feature / "episodes"
    epdir.mkdir(parents=True)
    camera_dir = root / "camera-features" / "episodes"
    language_dir = root / "camera-language-features" / "episodes"
    if camera:
        camera_dir.mkdir(parents=True)
        language_dir.mkdir(parents=True)
    steps = np.array([0, 5], np.int32)
    state = np.zeros((10, 8), np.float32)
    action = np.ones((10, 7), np.float32)
    init = initial_state_id(state[0], task_index)
    ep = epdir / f"{name}.npz"
    payload = {"step": steps, "env_step": np.arange(10, dtype=np.int32), "proprio": state[steps],
               "env_state": state, "action": action, "f": np.ones((2, 4), np.float16),
               "k": np.ones((2, 4), np.float16), "task_index": np.asarray(task_index, np.int32)}
    if split:
        payload.update(init_state_id=np.asarray(init), split=np.asarray(split))
    np.savez_compressed(ep, **payload)
    entry = {"episode_id": name, "suite": "libero10" if split else "libero_spatial",
             "task": "test", "task_index": task_index, "n_rows": 2, "cache": str(ep)}
    if split:
        entry.update(init_state_id=init, split=split)
    (feature / "manifest.json").write_text(json.dumps({"episodes": [entry]}))
    if camera:
        np.savez_compressed(camera_dir / f"{name}.npz", step=steps, base=np.ones((2, 3), np.float16),
                            wrist=np.ones((2, 3), np.float16) * 2)
        np.savez_compressed(language_dir / f"{name}.npz", step=steps, base_lang=np.ones((2, 3), np.float16) * 3,
                            wrist_lang=np.ones((2, 3), np.float16) * 4)
    return root


def test_join_existing_cache_with_libero10_and_mark_missing_modalities(tmp_path):
    existing = _write_cache(tmp_path / "existing", "old-ep", task_index=10)
    new = _write_cache(tmp_path / "libero10", "new-ep", task_index=2, split="query", camera=True)
    combined = join_existing_and_libero10(existing / "feature-cache", new, horizon=4)
    assert combined["f"].shape == (4, 4) and combined["f"].dtype == np.float16
    assert combined["action_chunk"].shape == (4, 4, 7)
    assert combined["motion"].shape == (4, 4, 9)
    assert combined["episode_id"].tolist() == ["old-ep", "old-ep", "new-ep", "new-ep"]
    assert combined["has_wrist_lang"].tolist() == [False, False, True, True]
    assert np.isnan(combined["wrist_lang"][:2]).all()
    np.testing.assert_array_equal(combined["wrist_lang"][2:], np.ones((2, 3), np.float16) * 4)
