import numpy as np

from skill_moe.skillnet.examples.emergent.light.methods.m0 import M0
from skill_moe.skillnet.examples.emergent.light.methods.m4 import M4
from skill_moe.skillnet.examples.emergent.light.skill_bench import (
    _initialize_standard_units, _memory_add, _memory_add_many_standard, _novelty_scores_many,
)


def test_two_stage_novelty_separates_memory_items_from_skill_units():
    rng = np.random.default_rng(44)
    train_base = np.concatenate([
        rng.normal(loc=(-2., -2.), scale=.1, size=(40, 2)),
        rng.normal(loc=(2., 2.), scale=.1, size=(40, 2)),
    ]).astype(np.float32)
    method = M0()
    method.tau = 1.0
    _initialize_standard_units(method, train_base)
    method.memory = [{"key": np.zeros(2, np.float32), "base": train_base[i],
                      "meta": {"episode_id": f"seed-{i}", "init_state_id": f"init-{i}"}}
                     for i in range(10)]
    old_units = len(method._d3_unit_centers)
    known = method._d3_unit_centers[:1]
    candidate_base = np.concatenate([known, np.asarray([[12., -12.]], np.float32)], axis=0)
    candidate_key = np.asarray([[8., 0.], [0., 0.]], np.float32)
    metas = [
        {"episode_id": "new-key", "init_state_id": "new-init-1", "segment_id": "new-key@0"},
        {"episode_id": "known-key-new-base", "init_state_id": "new-init-2", "segment_id": "known-key-new-base@0"},
    ]
    scores = _novelty_scores_many(method, candidate_key, candidate_base, metas)
    accepted, reasons = _memory_add_many_standard(method, candidate_key, list(candidate_base), metas)
    assert scores[0] < 1e-5  # R4 measures new-unit novelty; new-key items are tracked separately.
    assert scores[1] > scores[0]
    assert accepted.tolist() == [True, True]
    assert reasons["new_key_items"] == 1
    assert reasons["new_skill_units"] == 1
    assert len(method._d3_unit_centers) == old_units + 1
    assert method.memory[-2]["meta"]["key_novel"] and not method.memory[-2]["meta"]["skill_unit_added"]
    assert not method.memory[-1]["meta"]["key_novel"] and method.memory[-1]["meta"]["skill_unit_added"]


def test_base_novelty_add_rate_is_neither_zero_nor_one():
    rng = np.random.default_rng(45)
    train_base = np.concatenate([
        rng.normal(loc=(-2., 0.), scale=.1, size=(40, 2)),
        rng.normal(loc=(2., 0.), scale=.1, size=(40, 2)),
    ]).astype(np.float32)
    method = M0(); method.tau = 1.0
    _initialize_standard_units(method, train_base)
    method.memory = [{"key": np.zeros(2, np.float32), "base": x,
                      "meta": {"episode_id": f"seed-{i}", "init_state_id": f"seed-init-{i}"}}
                     for i, x in enumerate(train_base[:8])]
    center = method._d3_unit_centers[0]
    candidates = np.concatenate([np.repeat(center[None], 5, axis=0),
                                 np.asarray([[10., 10.], [-10., 10.], [10., -10.], [-10., -10.], [0., 14.]], np.float32)])
    keys = np.zeros((len(candidates), 2), np.float32)
    metas = [{"episode_id": f"candidate-{i}", "init_state_id": f"candidate-init-{i}"} for i in range(len(candidates))]
    accepted, reasons = _memory_add_many_standard(method, keys, list(candidates), metas)
    rate = reasons["new_skill_units"] / len(candidates)
    assert 0 < rate < 1
    assert accepted.sum() < len(candidates)


def test_m4_capacity_miss_is_recorded_without_touching_method_file():
    rng = np.random.default_rng(46)
    n = 32
    train = {"actions": rng.normal(size=(n, 4, 7)).astype(np.float32),
             "state": rng.normal(size=(n, 8)).astype(np.float32),
             "feature": rng.normal(size=(n, 6)).astype(np.float32),
             "key": rng.normal(size=(n, 6)).astype(np.float32),
             "episode_id": np.asarray([f"episode-{i}" for i in range(n)]),
             "init_state_id": np.asarray([f"init-{i}" for i in range(n)])}
    method = M4(epochs=1, batch_size=8, width=16, base_dim=6, residual_dim=3, max_prototypes=8)
    method.fit(train)
    base = np.asarray([[100.0] * method.base_dim], dtype=np.float32)
    method.max_prototypes = len(method.prototypes)
    method.novelty_distance_threshold = 0.0
    ok, reason = _memory_add(method, np.zeros(6, np.float32), base,
                             {"episode_id": "new", "init_state_id": "new-init"})
    assert ok and reason == "vigilance_miss_capacity_reached"
    assert len(method.memory) == 1 and method.memory[0]["code"] < len(method.prototypes)


def test_m4_item_gate_uses_key_novelty_separately_from_unit_novelty():
    rng = np.random.default_rng(47); n = 32
    train = {"actions": rng.normal(size=(n, 4, 7)).astype(np.float32),
             "state": rng.normal(size=(n, 8)).astype(np.float32),
             "feature": rng.normal(size=(n, 6)).astype(np.float32),
             "key": rng.normal(size=(n, 6)).astype(np.float32),
             "episode_id": np.asarray([f"episode-{i}" for i in range(n)]),
             "init_state_id": np.asarray([f"init-{i}" for i in range(n)])}
    method = M4(epochs=1, batch_size=8, width=16, base_dim=6, residual_dim=3, max_prototypes=8)
    method.fit(train); method.tau = .5; method.novelty_distance_threshold = 1.0
    base = method.prototypes[:1].copy(); key = np.asarray([1., 0., 0., 0., 0., 0.], np.float32)
    method.memory = [{"key": key.copy(), "code": 0, "meta": {"episode_id": "stored", "init_state_id": "stored-init"}}]
    accepted, reason = _memory_add(method, key, base, {"episode_id": "known", "init_state_id": "known-init"})
    assert not accepted and reason == "known_key_known_base"
    accepted, reason = _memory_add(method, -key, base, {"episode_id": "novel-key", "init_state_id": "novel-init"})
    assert accepted and reason in {"resonance_frozen", "resonance_updated"}
    assert method.memory[-1]["meta"]["key_novel"]
    assert not method.memory[-1]["meta"]["skill_unit_added"]
