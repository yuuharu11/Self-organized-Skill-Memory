import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "skill_moe/skillnet/examples/memory_pilot/unseen_libero/key_query_k2.py"
spec = importlib.util.spec_from_file_location("unseen_key_query_k2_test", SCRIPT)
k2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k2)


def test_pool_prefix_all_layers_4x4_and_valid_instruction_mean():
    # 18 layers, one batch, 3x256 image slots, two prompt tokens, width 2.
    x = np.zeros((18, 1, 770, 2), np.float32)
    x[:, :, :256] = 1
    x[:, :, 256:512] = 2
    x[:, :, 512:768] = 99  # masked third camera must not appear
    x[:, :, 768] = 4
    x[:, :, 769] = 8
    pooled, mask = k2.pool_prefix_layer_values(x, np.asarray([[True, False]]))
    assert pooled.shape == (1, 18, 33, 2)
    assert np.all(pooled[:, :, :16] == 1)
    assert np.all(pooled[:, :, 16:32] == 2)
    assert np.all(pooled[:, :, 32] == 4)
    assert mask.shape == (1, 33) and mask.all()


def test_history_only_uses_preceding_chunks_and_zero_masks_at_episode_start():
    rows = {"task": np.asarray([4, 4, 4]), "episode": np.asarray([9, 9, 9]),
            "frame": np.asarray([0, 10, 20]), "state": np.arange(24, dtype=np.float32).reshape(3, 8)}
    actions = np.arange(3 * 10 * 7, dtype=np.float32).reshape(3, 10, 7)
    out = k2.build_history_arrays(rows, np.asarray([0, 1, 2]), actions)
    assert not out["action_mask_10"][0].any()
    assert out["action_10"][0].sum() == 0
    np.testing.assert_array_equal(out["action_10"][1], actions[0])
    np.testing.assert_array_equal(out["action_30"][2], np.concatenate([np.zeros((10, 7)), actions[0], actions[1]]))
    assert not out["action_mask_50"][1, :40].any()
    assert out["action_mask_50"][1, 40:].all()


def test_history_unit_codes_and_mask_are_bounded():
    x = np.zeros((30, 7), np.float32)
    x[:15, 0] = np.linspace(0, 1, 15)
    x[15:, 1] = np.linspace(0, 1, 15)
    x[20:, 6] = 1
    code, mask = k2.history_unit_codes(x)
    assert code.shape == (3, 21)
    assert mask.shape == (3,) and mask.any()
    assert np.isfinite(code).all()


def test_h3_trims_unavailable_padding_and_preserves_empty_history_mask():
    real = np.zeros((10, 7), np.float32)
    real[:, 0] = np.linspace(-0.4, 0.5, 10)
    real[6:, 6] = 1.0
    padded = np.concatenate([np.zeros((20, 7), np.float32), real])
    valid = np.concatenate([np.zeros(20, bool), np.ones(10, bool)])
    codes, mask = k2.history_unit_codes(padded, valid)
    expected_codes, expected_mask = k2.history_unit_codes(real, np.ones(10, bool))
    np.testing.assert_array_equal(codes, expected_codes)
    np.testing.assert_array_equal(mask, expected_mask)

    empty_codes, empty_mask = k2.history_unit_codes(np.zeros((30, 7), np.float32),
                                                     np.zeros(30, bool))
    assert not empty_mask.any()
    assert not empty_codes.any()
    # A real, executed zero-action segment remains distinguishable from
    # unavailable history because its mask is true.
    _, executed_zero_mask = k2.history_unit_codes(np.zeros((10, 7), np.float32),
                                                   np.ones(10, bool))
    assert executed_zero_mask.any()

    histories = {"action_10": np.stack([np.zeros((10, 7), np.float32), real]),
                 "action_mask_10": np.stack([np.zeros(10, bool), np.ones(10, bool)])}
    features = k2.history_only_inputs(histories, length=10, encoder="h3")
    assert not features[0, -3:].any()
    assert features[1, -3:].any()


def test_h3_rejects_non_suffix_history_mask():
    with pytest.raises(ValueError, match="contiguous suffix"):
        k2.history_unit_codes(np.zeros((4, 7), np.float32),
                              np.asarray([True, False, True, True]))


def test_top1_is_candidate_local_and_respects_exclusions():
    scores = np.asarray([[100, 2, 3], [5, 4, 8]], np.float32)
    allowed = np.asarray([[False, True, True], [True, False, True]])
    np.testing.assert_array_equal(k2.masked_top1(scores, allowed), [2, 2])
    with pytest.raises(ValueError):
        k2.masked_top1(scores, np.zeros_like(allowed))


def test_k1_task_and_cross_task_exclusions_remain_the_shared_contract():
    k1 = k2._k1()
    pool = {"source_task": np.asarray([0, 1, 2, 3])}
    allow = k1.filter_cross_task(pool, np.asarray([0, 2]), heldout=(3,))
    np.testing.assert_array_equal(allow, [[False, True, True, False], [True, True, False, False]])


def test_layer_query_variants_and_history_encoders_smoke():
    import torch
    torch.set_num_threads(8)
    tokens = torch.randn(4, 33, 256)
    mask = torch.ones(4, 33, dtype=torch.bool)
    state = torch.randn(4, 8)
    hist = torch.randn(4, 64)
    for fusion in ("attention", "concat", "visual", "history"):
        model = k2.make_k2_layer_query(history_mode=fusion, history_dim=64)
        result = model(tokens, mask, state, hist)
        if isinstance(result, tuple): result = result[0]
        assert result.shape == (4, 256)
    gru = k2.K2HistoryGRU()
    acts = torch.randn(4, 30, 7)
    hmask = torch.ones(4, 30, dtype=torch.bool)
    assert gru(acts, hmask).shape == (4, 128)


def test_one_step_listwise_retriever_fit_and_rank():
    rng = np.random.default_rng(3)
    n, k = 8, 5
    tokens = rng.normal(size=(n, 33, 256)).astype(np.float32)
    token_mask = np.ones((n, 33), bool)
    state = rng.normal(size=(n, 8)).astype(np.float32)
    history = rng.normal(size=(n, 16)).astype(np.float32)
    shapes = rng.normal(size=(k, 10, 7)).astype(np.float32)
    allowed = np.ones((n, k), bool)
    allowed[:, 0] = False
    teacher = allowed.astype(np.float32) / allowed.sum(1, keepdims=True)
    model = k2.fit_layer_retriever(tokens, token_mask, state, history, shapes,
        teacher, allowed, np.arange(n), epochs=1, batch_size=4,
        history_mode="attention", device="cpu", threads=8,
        train_metadata={"episode":np.repeat(np.arange(4),2),"frame":np.tile([0,10],4),
                        "hard_context":rng.normal(size=(n,70)).astype(np.float32)},
        hard_negative_weight=.05)
    scores, rank = k2.score_layer_retriever(model, tokens, token_mask, state,
        history, shapes, allowed, batch_size=4)
    assert scores.shape == (n, k) and rank.shape == (n, k)
    assert np.all(allowed[np.arange(n), rank[:, 0]])


def test_layer_selection_and_history_dropout_are_deterministic():
    assert k2.select_top_layers({2: .3, 7: .1, 1: .2, 4: .4}) == (7, 1, 2)
    x = np.ones((20, 9), np.float32)
    a = k2.drop_history_modality(x, .5, seed=4)
    b = k2.drop_history_modality(x, .5, seed=4)
    np.testing.assert_array_equal(a, b)
    assert np.any(np.all(a == 0, axis=1)) and np.any(np.all(a == 1, axis=1))
    assert k2.drop_history_modality(x, .5, seed=4, training=False).sum() == x.sum()


def test_elmo_scalar_mixer_fit_score_and_layer_weights():
    rng = np.random.default_rng(9)
    n, layers, candidates = 8, 3, 5
    x = rng.normal(size=(n, layers, 4, 16)).astype(np.float32)
    mask = np.ones((n, 4), bool)
    state = rng.normal(size=(n, 8)).astype(np.float32)
    hist = np.zeros((n, 1), np.float32)
    shapes = rng.normal(size=(candidates, 10, 7)).astype(np.float32)
    allowed = np.ones((n, candidates), bool)
    teacher = np.full((n, candidates), 1 / candidates, np.float32)
    fit = k2.fit_elmo_layer_mixer(x, mask, state, hist, shapes, teacher, allowed,
                                  np.arange(n), epochs=1, batch_size=4)
    scores, ranks = k2.score_elmo_layer_mixer(fit, x, mask, state, hist, shapes, allowed)
    assert scores.shape == (n, candidates)
    assert ranks.shape == (n, candidates)
    assert fit["layer_weights"].shape == (layers,)
    np.testing.assert_allclose(fit["layer_weights"].sum(), 1., atol=1e-6)


def test_top_three_layer_concatenation_preserves_layer_identity():
    x=np.stack([np.full((2,3,4),float(i),np.float32) for i in range(5)],axis=1)
    y=k2.combine_layer_tokens(x,(1,3,4),method="concat")
    assert y.shape==(2,3,12)
    np.testing.assert_array_equal(y[0,0],[1,1,1,1,3,3,3,3,4,4,4,4])
