import numpy as np
import torch

from skill_moe.skillnet.examples.emergent.light.intermediate.clip_retrieval import (
    DualEncoder,
    FrozenEmbeddingMemory,
    _fit_m6_units,
    attach_subtask_texts,
    fit_dual_encoder,
    mean_token_embedding,
    plan_text_for_segment,
    retrieve_embedding_top1,
    split_suite_mask,
)


def test_split_masks_keep_heldout_suite_out_of_query_tower_fit():
    tasks = np.arange(40)
    split_a_train = split_suite_mask(tasks, range(20, 40))
    split_b_train = split_suite_mask(tasks, range(10, 30))
    assert set(tasks[split_a_train]) == set(range(20, 40))
    assert set(tasks[split_b_train]) == set(range(10, 30))
    assert not split_a_train[10:20].any()
    assert not split_b_train[30:40].any()


def test_query_tower_fit_rejects_a_heldout_suite_row():
    rows = {"task_index": np.asarray([20, 30])}
    try:
        fit_dual_encoder(rows, np.zeros((2, 4), np.float32), use_subtask=False,
                         nearby_positive=False, seed=0, epochs=1, patience=1,
                         device=torch.device("cpu"), allowed_task_indices=range(20, 30))
    except ValueError as exc:
        assert "non-training suites" in str(exc)
    else:
        raise AssertionError("held-out suite was accepted for query-tower training")


def test_memory_add_does_not_mutate_frozen_query_or_skill_tower():
    tower = DualEncoder(feature_dim=5, key_dim=3, state_dim=4, base_dim=6,
                        num_classes=2, use_subtask=True)
    for parameter in tower.parameters():
        parameter.requires_grad_(False)
    before = {name: value.detach().clone() for name, value in tower.state_dict().items()}
    memory = FrozenEmbeddingMemory()
    q = np.zeros((3, 96), np.float32)
    h = np.ones((3, 96), np.float32)
    b = np.full((3, 6), 2.0, np.float32)
    memory.add(q, h, b)
    after = tower.state_dict()
    assert all(torch.equal(before[name], after[name]) for name in before)
    assert all(not parameter.requires_grad for parameter in tower.parameters())
    np.testing.assert_array_equal(memory.query_embeddings, q)
    np.testing.assert_array_equal(memory.skill_embeddings, h)
    np.testing.assert_array_equal(memory.bases, b)


def test_symmetric_retrieval_uses_query_tower_item_vectors():
    # Both columns represent g-space. The nearest item is selected by g·g,
    # distinct from scoring the same query against h-space.
    query = np.asarray([[1.0, 0.0], [0.0, 1.0]], np.float32)
    stored_g = np.asarray([[0.9, 0.1], [0.1, 0.9], [-1.0, 0.0]], np.float32)
    stored_h = np.asarray([[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]], np.float32)
    np.testing.assert_array_equal(retrieve_embedding_top1(query, stored_g), [0, 1])
    np.testing.assert_array_equal(retrieve_embedding_top1(query, stored_h), [1, 0])


def test_embedding_retrieval_rejects_empty_or_mismatched_memory():
    query = np.ones((1, 3), np.float32)
    try:
        retrieve_embedding_top1(query, np.ones((2, 4), np.float32))
    except ValueError as exc:
        assert "dimensions differ" in str(exc)
    else:
        raise AssertionError("dimension mismatch was accepted")
    try:
        retrieve_embedding_top1(query, np.empty((0, 3), np.float32))
    except ValueError as exc:
        assert "empty memory" in str(exc)
    else:
        raise AssertionError("empty memory was accepted")


def test_query_tower_has_independent_normalized_32d_modalities_and_ablation():
    tower = DualEncoder(feature_dim=5, key_dim=3, state_dim=4, base_dim=6,
                        num_classes=2, use_subtask=True)
    task = torch.tensor([0, 1])
    feature = torch.randn(2, 5)
    key = torch.randn(2, 3)
    state = torch.randn(2, 4)
    task_embedding = tower.class_embedding(task)
    e_img = torch.nn.functional.normalize(
        tower.image_tower(torch.cat([feature, key], dim=-1)), dim=-1)
    e_state = torch.nn.functional.normalize(tower.state_tower(state), dim=-1)
    e_sub = torch.nn.functional.normalize(tower.subtask_tower(task_embedding), dim=-1)
    assert e_img.shape == e_state.shape == e_sub.shape == (2, 32)
    torch.testing.assert_close(e_img.norm(dim=-1), torch.ones(2))
    torch.testing.assert_close(e_state.norm(dim=-1), torch.ones(2))
    torch.testing.assert_close(e_sub.norm(dim=-1), torch.ones(2))

    full = tower.encode_query(task, feature, key, state)
    no_img = tower.encode_query(task, feature, key, state, include_img=False)
    no_state = tower.encode_query(task, feature, key, state, include_state=False)
    no_subtask = tower.encode_query(task, feature, key, state, include_subtask=False)
    assert full.shape == no_img.shape == no_state.shape == no_subtask.shape == (2, 96)
    for encoded in (full, no_img, no_state, no_subtask):
        torch.testing.assert_close(encoded.norm(dim=-1), torch.ones(2))
    assert tower.modality_weights().keys() == {"image", "state", "subtask"}


def test_functional_units_follow_m6_probe_and_epsilon_calibration():
    class DummyM1:
        action_shape = (1, 2)

        def prior(self, feature, state):
            return np.zeros((len(feature), 12), np.float32), np.zeros((len(feature), 4), np.float32)

        def decode(self, base, residual, feature, state):
            return np.asarray(base, np.float32)[:, None, :2]

    centers = np.asarray([[0.0, 0.0], [5.0, 0.0]], np.float32)
    base = np.zeros((60, 12), np.float32)
    base[:, :2] = np.concatenate([
        np.repeat(centers[0][None], 30, axis=0),
        np.repeat(centers[1][None], 30, axis=0),
    ])
    episodes = np.asarray([f"ep-{i // 10}" for i in range(60)])
    train = {"f": np.zeros((60, 3), np.float32),
             "s": np.zeros((60, 2), np.float32),
             "episode_id": episodes}
    builder, labels = _fit_m6_units(
        DummyM1(), train, base, seed=3, device=torch.device("cpu"))
    assert builder._functional_probe_count == 32
    assert builder.RANDOM_DISTANCE_PAIRS == 64
    assert builder.kappa == 0.1 and builder.min_support == 10 and builder.min_episodes == 3
    assert builder.epsilon == max(0.1 * np.median(builder._random_pair_distances), 1e-12)
    assert len(builder.prototypes) == 2
    assert np.mean(labels >= 0) == 1.0


def test_plan_segment_text_join_preserves_rows_and_maps_minus_one_to_empty(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "manifest.json").write_text(
        '{"episodes":[{"episode_id":"ep-a","task":"pick and place"}]}'
    )
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        '{"pick and place":{"plan":["pick the mug","place the mug"]}}'
    )
    rows = {"episode_id": np.asarray(["ep-a", "ep-a", "ep-a"]),
            "skill_segment": np.asarray([0, 1, -1]),
            "skill_class": np.asarray([0, 1, -1]),
            "a": np.zeros((3, 2, 7), np.float32)}
    joined = attach_subtask_texts(rows, cache_dir=cache, annotations_path=annotations)
    assert len(joined["subtask_text"]) == len(rows["a"])
    assert joined["subtask_text"].tolist() == ["pick the mug", "place the mug", ""]
    assert plan_text_for_segment("pick and place", -1,
                                 {"pick and place": {"plan": ["pick", "place"]}}) == ""


def test_frozen_text_token_embeddings_are_averaged_and_empty_is_zero():
    class DummyTokenizer:
        def encode(self, text, add_bos=False):
            assert add_bos is False
            return {"pick mug": [1, 3], "place mug": [2]}[text]

    table = np.arange(20, dtype=np.float32).reshape(5, 4)
    np.testing.assert_array_equal(mean_token_embedding("pick mug", DummyTokenizer(), table),
                                  (table[1] + table[3]) / 2)
    np.testing.assert_array_equal(mean_token_embedding("", DummyTokenizer(), table), np.zeros(4))


def test_sentence_subtask_mode_accepts_frozen_embedding_and_zeroes_missing_segment():
    tower = DualEncoder(feature_dim=5, key_dim=3, state_dim=4, base_dim=6,
                        num_classes=2, subtask_mode="sentence", subtask_input_dim=12)
    sentence = torch.randn(2, 12)
    sentence[0].zero_()
    query = tower.encode_query(sentence, torch.randn(2, 5), torch.randn(2, 3), torch.randn(2, 4))
    assert query.shape == (2, 96)
    torch.testing.assert_close(query.norm(dim=-1), torch.ones(2))
    task_component = tower.subtask_tower(sentence)
    projected = torch.nn.functional.normalize(task_component, dim=-1)
    # The zero vector is explicitly masked after E_sub, including its learned bias.
    masked = torch.where(sentence.abs().sum(dim=-1, keepdim=True) > 0,
                         projected, torch.zeros_like(projected))
    assert torch.count_nonzero(masked[0]) == 0
    assert torch.count_nonzero(masked[1]) > 0


def test_sentence_mode_query_tower_fits_frozen_vectors_on_training_suites_only():
    rng = np.random.default_rng(21)
    n = 48
    rows = {
        "task_index": np.full(n, 20, np.int64),
        "skill_class": np.tile([0, 1], n // 2),
        "episode_id": np.repeat([f"ep-{i}" for i in range(8)], n // 8),
        "f": rng.normal(size=(n, 5)).astype(np.float32),
        "k": rng.normal(size=(n, 3)).astype(np.float32),
        "s": rng.normal(size=(n, 4)).astype(np.float32),
        "a": rng.normal(size=(n, 2, 7)).astype(np.float32),
        "subtask_embedding": rng.normal(size=(n, 24)).astype(np.float32),
    }
    base = rng.normal(size=(n, 12)).astype(np.float32)
    model, fit = fit_dual_encoder(
        rows, base, use_subtask=True, subtask_mode="sentence", nearby_positive=False,
        seed=5, epochs=1, patience=1, device=torch.device("cpu"),
        allowed_task_indices=[20])
    q = model.encode_query(torch.as_tensor(rows["subtask_embedding"][:4]),
                           torch.as_tensor(rows["f"][:4]), torch.as_tensor(rows["k"][:4]),
                           torch.as_tensor(rows["s"][:4]))
    assert fit["subtask_mode"] == "sentence"
    assert fit["train_task_classes"] == [0, 1]
    assert q.shape == (4, 96)
