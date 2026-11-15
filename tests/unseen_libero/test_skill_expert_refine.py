import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[2] / 'skill_moe/skillnet/examples/memory_pilot/unseen_libero/skill_expert_refine.py'
spec = importlib.util.spec_from_file_location('skill_expert_refine_tested', SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_unit_window_uses_natural_prefix_and_holds_short_tail():
    units = np.array([
        [[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14], [15, 16, 17, 18, 19, 20, 21]],
    ], dtype=np.float32)
    got = mod.unit_window(units, horizon=5)
    np.testing.assert_array_equal(got[0, :3], units[0])
    np.testing.assert_array_equal(got[0, 3:], np.repeat(units[0, -1:], 2, axis=0))


def test_nearest_unit_excludes_query_task_and_uses_arm_only():
    target = np.zeros((1, 10, 7), np.float32)
    candidates = np.zeros((2, 10, 7), np.float32)
    candidates[0, :, :6] = 0.01
    candidates[0, :, 6] = 1.0
    candidates[1, :, :6] = 0.2
    candidates[1, :, 6] = 0.0
    selected, _ = mod.nearest_unit_ids(target, candidates, np.array([4, 5]), query_task=np.array([4]))
    np.testing.assert_array_equal(selected, [1])


def test_draft_factory_keeps_raw_unit_motion_and_copies_pi05_gripper():
    target = np.zeros((1, 10, 7), np.float32)
    unit = np.zeros((1, 10, 7), np.float32)
    unit[:, :, :6] = .125
    unit[:, :, 6] = -1
    normal = np.zeros((1, 10, 32), np.float32)
    normal[:, :, 6] = .75
    bank = {'actions': unit, 'task': np.array([8], np.int32)}
    draft, ix, _ = mod.draft_factory(target, normal, bank, query_task=np.array([2]))
    np.testing.assert_array_equal(ix, [0])
    np.testing.assert_allclose(draft[0, :, :6], .125)
    np.testing.assert_allclose(draft[0, :, 6], .75)
    np.testing.assert_array_equal(draft[:, :, 7:], 0)


def test_mse6_is_summed_over_six_channels_and_divided_by_70():
    p = np.ones((2, 10, 32), np.float32)
    y = np.zeros_like(p)
    np.testing.assert_allclose(mod._mse6(p, y), np.full(2, 6 / 7))


def test_flow_samples_are_seeded_for_the_full_effective_batch_before_slicing():
    import jax
    key = jax.random.key(20261104)
    full = mod._sample_flow_inputs(key, (8, 10, 32))
    again = mod._sample_flow_inputs(key, (8, 10, 32))
    for x, y in zip(full, again, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    parts = [tuple(np.asarray(x)[i:i + 2] for x in full) for i in range(0, 8, 2)]
    for field, expected in enumerate(full):
        np.testing.assert_array_equal(
            np.concatenate([part[field] for part in parts], axis=0), np.asarray(expected))


def test_cached_selection_checks_row_alignment_and_materializes_draft(tmp_path):
    bank = {
        'actions': [np.full((5, 7), .25, np.float32), np.full((7, 7), -.5, np.float32)],
        'length': np.array([5, 7], np.int32),
        'task': np.array([1, 2], np.int32),
        'episode': np.array([3, 4], np.int32),
    }
    digest = mod._unit_bank_fingerprint(bank)
    mod._write_json(tmp_path / 'selection_manifest.json', {'unit_bank_fingerprint_sha256': digest})
    mod._write_npz(tmp_path / 'train_a.npz', row_ids=np.array([10, 11]),
                   selected_unit=np.array([1, 0]), arm6_mse=np.array([.1, .2]))
    _, selected = mod.load_unit_selections(tmp_path, bank, {'train_a': np.array([10, 11])})
    normal = np.zeros((2, 10, 32), np.float32)
    normal[:, :, 6] = .75
    draft = mod.draft_from_unit_ids(normal, bank, selected['train_a']['selected_unit'])
    np.testing.assert_allclose(draft[0, :, :6], -.5)
    np.testing.assert_allclose(draft[0, :, 6], .75)
    np.testing.assert_allclose(draft[1, :, :6], .25)
    with np.testing.assert_raises_regex(ValueError, 'misaligned'):
        mod.load_unit_selections(tmp_path, bank, {'train_a': np.array([11, 10])})


def test_l90x_source_task_mapping_uses_exact_normalized_instruction():
    mapped = mod.map_instruction_tasks(
        {'16': 'Turn on the stove', '9': 'pick up the book and place it in the back compartment of the caddy'},
        {'episodes': [
            {'task_id': 20, 'instruction': ' turn on   THE stove '},
            {'task_id': 44, 'instruction': 'Turn on the stove'},
            {'task_id': 77, 'instruction': 'Pick up the book and place it in the back compartment of the caddy'},
            {'task_id': 2, 'instruction': 'an unrelated instruction'},
        ]})
    assert mapped == {20: 16, 44: 16, 77: 9, 2: -1}


def test_oracle_dev_is_blocked_for_production_checkpoint_selection():
    with np.testing.assert_raises_regex(ValueError, 'KEY2-F-retrieved Dev drafts'):
        mod.assert_dev_retrieval_for_training({'dev_draft_selection_method': 'oracle_pending_key2f_retrieval'})
    mod.assert_dev_retrieval_for_training({'dev_draft_selection_method': 'oracle_smoke_only'},
                                          allow_oracle_smoke=True)
    mod.assert_dev_retrieval_for_training({'dev_draft_selection_method': 'key2f_retrieval'})
