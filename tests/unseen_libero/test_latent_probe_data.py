from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = (Path(__file__).resolve().parents[2] / "skill_moe" / "skillnet" / "examples" /
               "memory_pilot" / "unseen_libero" / "probes/latent_probe_data.py")
SPEC = importlib.util.spec_from_file_location("latent_probe_data", MODULE_PATH)
latent_probe_data = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(latent_probe_data)


def test_libero40_keys_mirror_seg2_fit_window_contract(tmp_path):
    root = tmp_path / "run"
    rows = root / "rows"
    rows.mkdir(parents=True)
    episodes = [
        {"task": 0, "episode": 1, "memory": False, "length": 20, "path": "unused"},
        {"task": 2, "episode": 2, "memory": False, "length": 20, "path": "unused"},
        {"task": 3, "episode": 3, "memory": True, "length": 20, "path": "unused"},
    ]
    (root / "data_manifest.json").write_text(json.dumps({"episodes": episodes, "stride": 10}))
    # Two run rows represent twenty trace steps for each stored episode.
    for name, values in {
        "task": np.array([0, 0, 2, 2, 3, 3]),
        "episode": np.array([1, 1, 2, 2, 3, 3]),
        "frame": np.array([0, 10, 0, 10, 0, 10]),
    }.items():
        np.save(rows / f"{name}.npy", values)
    keys = latent_probe_data.build_libero40_fit_keys(root, heldout_tasks=(2,))
    assert keys == [(0, 1, 0), (0, 1, 5), (0, 1, 10)]


def test_eval_fixed_rows_and_dev_eval_episode_disjoint(tmp_path):
    root = tmp_path / "unseen"
    root.mkdir()
    for name, data in {
        "task": np.array([6, 6, 0, 0]),
        "episode": np.array([1, 1, 2, 2]),
        "frame": np.array([0, 10, 0, 10]),
        "is_dev": np.array([True, True, False, False]),
    }.items():
        np.save(root / f"{name}.npy", data)
    keys = latent_probe_data.build_eval_keys(root, expected_rows=2)
    assert keys == [(0, 2, 0), (0, 2, 10)]
    assert latent_probe_data.assert_dev_eval_disjoint(
        [{"task": 6, "episode": 1}], keys) == {"dev_episodes": 1, "eval_episodes": 1, "overlap": 0}
    with pytest.raises(AssertionError, match="overlap"):
        latent_probe_data.assert_dev_eval_disjoint(
            [{"task": 0, "episode": 2}], keys)


def test_hdf5_orientation_is_exact_180_degree_pixel_reversal():
    image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    actual = latent_probe_data.orient_libero_hdf5_frame(image)
    assert actual.flags["C_CONTIGUOUS"]
    assert np.array_equal(actual, image[::-1, ::-1])


def test_action_targets_rms_switch_and_segment_lengths():
    physical = np.zeros((26, 7), np.float32)
    physical[:, 0] = 3.0
    physical[8:, 6] = 1.0
    normalized = physical * 0.5
    target = latent_probe_data.action_targets(physical, normalized, 5, [(0, 10), (10, 26)])
    assert target["actions_physical"].shape == (10, 7)
    assert np.allclose(target["rms6_physical"], [3, 0, 0, 0, 0, 0])
    assert np.allclose(target["rms6_normalized"], [1.5, 0, 0, 0, 0, 0])
    assert target["first_gripper_switch"] == 3
    assert target["current_segment_remaining"] == 5
    assert target["next_segment_length"] == 16
    assert target["gripper_signs"].tolist() == [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]


def test_storage_estimates_are_memmap_sized_and_vjepa_two_camera():
    e = latent_probe_data.array_storage_estimate(3, 128, 128)
    assert e["image_bytes"] == 3 * 4 * 128 * 128 * 3
    v = latent_probe_data.vjepa_storage_estimate(2)
    assert v["tokens_per_camera_shape_per_row"] == [1, 256, 1024]
    assert v["two_camera_unpooled_bytes"] == 2 * 2 * 256 * 1024 * 2
    assert v["two_camera_spatial_mean_bytes"] == 2 * 2 * 1024 * 2


def test_image_pixel_equality_assertion():
    expected = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    actual = expected.copy()
    assert np.array_equal(expected, actual)
    actual[0, 0, 0] ^= 1
    assert not np.array_equal(expected, actual)


@pytest.mark.skipif(not Path("/work/.skillnet-state/outputs/memory-pilot/unseen-libero/unseen-rows-02/manifest.json").exists(),
                    reason="requires the read-only unseen-rows-02 source cache")
def test_actual_unseen_hdf5_orientation_matches_cached_pixels():
    root = Path("/work/.skillnet-state/outputs/memory-pilot/unseen-libero/unseen-rows-02")
    result = latent_probe_data.verify_hdf5_image_parity(root)
    assert result["base_exact"] and result["wrist_exact"]


@pytest.mark.skipif(not Path("/work/.skillnet-state/outputs/memory-pilot/unseen-libero/seg-context-01/seg5-cache/manifest.json").exists(),
                    reason="requires the read-only SEG2 frame cache")
def test_real_fit_keys_exactly_match_seg2_query_builder():
    import importlib.util
    import sys

    run_file = MODULE_PATH.parent / "seg_context_run.py"
    sys.path.insert(0, str(run_file.parent))
    spec = importlib.util.spec_from_file_location("seg_context_run_for_lpd_test", run_file)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    run_root = Path("/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/run-01")
    extra_root = Path("/work/.skillnet-state/outputs/memory-pilot/unseen-libero/seg-context-01/seg5-cache")
    store = module.FrameStore(run_root, extra_root)
    actual = [tuple(map(int, k)) for k in module.build_fit_queries(store)["key"]]
    expected = latent_probe_data.build_libero40_fit_keys(run_root)
    assert actual == expected
