import json

import numpy as np
import pytest

from skill_moe.skillnet.examples.memory_pilot.unseen_libero.key_inputs_vjepa import (
    load_vjepa_tokens,
    offset_vjepa_groups,
    preceding_frame_keys,
    stable_row_map,
    vjepa_group_ids,
)


def test_exact_key_load_preserves_requested_order_and_groups(tmp_path):
    root = tmp_path / "libero40"
    root.mkdir()
    keys = np.array([[2, 7, 11], [1, 3, 5]], np.int32)
    base = np.zeros((2, 64, 1024), np.float16)
    wrist = np.zeros_like(base)
    base[0].fill(11); wrist[0].fill(12)
    base[1].fill(21); wrist[1].fill(22)
    np.save(root / "keys.npy", keys)
    np.save(root / "base.npy", base)
    np.save(root / "wrist.npy", wrist)
    (root / "manifest.json").write_text(json.dumps({"complete": True}))

    tokens, groups = load_vjepa_tokens(keys[::-1], cache_root=tmp_path, domain="libero40")
    assert tokens.shape == (2, 128, 1024)
    assert groups.shape == (2, 128)
    assert np.all(tokens[0, :64] == 21) and np.all(tokens[0, 64:] == 22)
    assert np.all(tokens[1, :64] == 11) and np.all(tokens[1, 64:] == 12)
    np.testing.assert_array_equal(groups[0], np.r_[np.zeros(64), np.ones(64)])


def test_lookup_rejects_missing_exact_key_instead_of_approximating(tmp_path):
    root = tmp_path / "libero90"
    root.mkdir()
    np.save(root / "keys.npy", np.array([[4, 9, 10]], np.int32))
    np.save(root / "base.npy", np.zeros((1, 64, 1024), np.float16))
    np.save(root / "wrist.npy", np.zeros((1, 64, 1024), np.float16))
    (root / "manifest.json").write_text(json.dumps({"complete": True}))
    with pytest.raises(KeyError, match="exact V-JEPA row keys missing"):
        load_vjepa_tokens(np.array([[4, 9, 11]], np.int32), cache_root=tmp_path, domain="libero90")


def test_previous_current_pair_never_reads_future_frame():
    keys = np.array([[0, 2, 0], [5, 8, 23], [7, 1, 100]], np.int32)
    pairs = preceding_frame_keys(keys, frame_step=1)
    np.testing.assert_array_equal(pairs[:, 0], [[0, 2, 0], [5, 8, 22], [7, 1, 99]])
    np.testing.assert_array_equal(pairs[:, 1], keys)
    assert np.all(pairs[:, 0, 2] <= pairs[:, 1, 2])
    assert not np.any(pairs[:, :, 2] > keys[:, None, 2])


def test_group_offsets_do_not_collide_with_vlm_groups():
    groups = vjepa_group_ids(2)
    shifted = offset_vjepa_groups(groups, vlm_group_count=4)
    np.testing.assert_array_equal(np.unique(shifted), [4, 5])
    assert set(np.unique(shifted)).isdisjoint({0, 1, 2, 3})


def test_row_map_rejects_duplicate_identity():
    with pytest.raises(ValueError, match="duplicate row key"):
        stable_row_map(np.array([[1, 2, 3], [1, 2, 3]], np.int32))
