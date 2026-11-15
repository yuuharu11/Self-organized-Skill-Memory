import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'skill_moe' / 'skillnet' / 'examples' / 'memory_pilot' / 'unseen_libero'
sys.path.insert(0, str(SCRIPT))
import compression_matched as cm  # noqa: E402


def test_dct_roundtrip_full_coefficients_and_truncation():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(6, 10, 7))
    full = cm.dct_decode(cm.dct_encode(x, 10), 10)
    assert np.allclose(full, x, atol=1e-12)
    reduced = cm.dct_decode(cm.dct_encode(x, 3), 10)
    assert reduced.shape == x.shape
    assert np.mean((reduced - x) ** 2) > 0


def test_keyframe_values_count_and_decode_shape():
    x = np.arange(2 * 10 * 7, dtype=np.float32).reshape(2, 10, 7)
    enc = cm.keyframe_encode(x, 4)
    assert enc.shape == (2, 4, 7)
    dec = cm.keyframe_decode(enc)
    assert dec.shape == x.shape
    assert np.array_equal(dec[:, 0], x[:, 0])
    assert np.array_equal(dec[:, -1], x[:, -1])


def test_pca_fit_restore_dimension_and_training_only_input_contract():
    rng = np.random.default_rng(8)
    train = rng.normal(size=(32, 10, 7))
    mean, basis = cm.pca_fit(train, 14)
    assert mean.shape == (70,)
    assert basis.shape == (14, 70)
    restored = cm.pca_restore(cm.pca_project(train[:3], mean, basis), mean, basis)
    assert restored.shape == (3, 10, 7)


def test_greedy_cover_threshold_and_no_merge():
    x = np.array([[[0.]], [[0.01]], [[1.]]], dtype=np.float32)
    assert np.array_equal(cm.greedy_cover(x, None), np.arange(3))
    cover = cm.greedy_cover(x, .001)
    assert len(cover) < 3
    assert cover.min() >= 0 and cover.max() < len(x)


def test_anchor_cut_known_speed_and_gripper_edges():
    # Translation has a minimum around the pause; gripper switch is mandatory.
    trace = np.zeros((20, 7), dtype=np.float32)
    trace[:6, 0] = .1
    trace[6:13, 0] = 0.
    trace[13:, 0] = .1
    trace[10:, 6] = 1.
    spans = cm.motion.anchor_spans(trace, min_length=3)
    assert spans[0][0] == 0 and spans[-1][1] == len(trace)
    assert any(10 in (a, b) for a, b in spans)
    assert all(b > a for a, b in spans)


def test_low_frequency_and_cluster_diagnostics_are_finite():
    rng = np.random.default_rng(12)
    actions = [rng.normal(size=(10 + i % 3, 7)).astype(np.float32) for i in range(80)]
    low = cm.low_frequency_concentration(actions)
    assert 0 <= low['q25'] <= low['median'] <= low['q75'] <= 1
    phase = np.stack([cm.phase_resample(a) for a in actions])
    sep = cm.between_within_ratio(phase, 8)
    assert sep['occupied_clusters'] <= 8
    assert np.isfinite(sep['ratio']) and sep['ratio'] >= 0


def test_nearest_templates_uses_original_action_space():
    q = np.zeros((2, 10, 7), np.float32)
    bank = np.stack([np.ones((10, 7)), np.zeros((10, 7))]).astype(np.float32)
    idx, err = cm.nearest_templates(q, bank)
    assert np.array_equal(idx, [1, 1])
    assert np.all(err == 0)


def test_training_selector_excludes_validation_rows():
    rows = {'task': np.array([0, 0, 1, 1]),
            'validation': np.array([False, True, False, True]),
            'actions': np.arange(4 * 10 * 7, dtype=np.float32).reshape(4, 10, 7)}
    got = cm.actions_for_tasks(rows, [0, 1])
    assert np.array_equal(got, rows['actions'][[0, 2]])


def test_variable_storage_reconstructs_to_each_original_length():
    rng = np.random.default_rng(91)
    examples = [rng.normal(size=(n, 7)).astype(np.float32) for n in (6, 13, 24)]
    dct_items = cm.compressed_variable(examples, 'dct', 3)
    raw_phase_items = cm.compressed_variable(examples, 'raw32', 224)
    assert [len(x) for x in dct_items] == [6, 13, 24]
    assert [len(x) for x in raw_phase_items] == [6, 13, 24]
    assert all(x.shape[1] == 7 for x in dct_items + raw_phase_items)


def test_matched_value_counts_are_equal_for_fixed_representations():
    for v in (14, 21, 28):
        k = v // 7
        x = np.zeros((3, 10, 7), np.float32)
        assert cm.dct_encode(x, k).shape[-2] * 7 == v
        assert cm.keyframe_encode(x, k).shape[-2] * 7 == v


def test_cluster_diagnostics_subsample_each_segmentation_to_same_count():
    groups = {
        'anchor': [np.full((i + 2, 7), i, np.float32) for i in range(9)],
        'fixed10': [np.full((10, 7), 100 + i, np.float32) for i in range(6)],
        'fixed_mean': [np.full((17, 7), 200 + i, np.float32) for i in range(7)],
    }
    first, n_first = cm.equal_count_samples(groups, seed=31)
    again, n_again = cm.equal_count_samples(groups, seed=31)
    assert n_first == n_again == 6
    assert {name: len(items) for name, items in first.items()} == {
        'anchor': 6, 'fixed10': 6, 'fixed_mean': 6,
    }
    for name in groups:
        assert all(np.array_equal(a, b) for a, b in zip(first[name], again[name]))
