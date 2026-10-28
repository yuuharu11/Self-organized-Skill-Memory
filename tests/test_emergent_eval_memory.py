import numpy as np
import pytest

from skill_moe.skillnet.examples.emergent.eval.memory import EmergentMemory, KeyEncoder, MemoryConfig
from skill_moe.skillnet.examples.memory_pilot.memory import namespace_init_state_id


def _sid(i):
    return namespace_init_state_id("s", 0, i)


def _reference(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return {"vlm_img_nolang": rng.normal(size=(n, 40)), "vlm_img_lang": rng.normal(size=(n, 40)) + 5,
            "proprio": rng.normal(size=(n, 8))}


def _memory(keys, config=MemoryConfig(), z_dim=3, per_episode=5, horizon=10):
    enc = KeyEncoder.fit({"vlm_img_nolang": keys, "proprio": np.zeros((len(keys), 2))}, n_components=keys.shape[1])
    mem = EmergentMemory(enc, config, horizon=horizon)
    pk = enc.transform({"vlm_img_nolang": keys, "proprio": np.zeros((len(keys), 2))})
    for i, k in enumerate(pk):
        ep = i // per_episode
        mem.add(k, np.full(z_dim, float(i)), episode_id=f"e{ep}", init_state_id=_sid(ep), step=10 * (i % per_episode))
    mem.calibrate()
    return mem, enc


def test_key_encoder_is_fixed_and_language_switch_selects_block(tmp_path):
    ref = _reference()
    enc = KeyEncoder.fit(ref, include_language=False, n_components=5)
    enc_l = KeyEncoder.fit(ref, include_language=True, n_components=5)
    assert enc.blocks == ("vlm_img_nolang", "proprio") and enc_l.blocks == ("vlm_img_lang", "proprio")
    new = _reference(10, seed=1)
    a = enc.transform(new)
    # Changing the unused language block does not change the no-language key.
    changed = dict(new, vlm_img_lang=new["vlm_img_lang"] * 100)
    np.testing.assert_allclose(enc.transform(changed), a)
    assert not np.allclose(enc_l.transform(changed), enc_l.transform(new))
    enc.save(tmp_path / "k.npz")
    np.testing.assert_allclose(KeyEncoder.load(tmp_path / "k.npz").transform(new), a)
    assert KeyEncoder.load(tmp_path / "k.npz").fingerprint == enc.fingerprint
    with pytest.raises(ValueError):
        KeyEncoder.fit({"proprio": ref["proprio"]}, include_language=False)
    frac = KeyEncoder.fit(ref, n_components=0.5)
    assert 1 <= frac.components.shape[0] < 48


def test_threshold_is_relative_to_memory_scale():
    # Keys are added directly in key space, so no standardization hides the scale.
    rng = np.random.default_rng(0)
    keys = rng.normal(size=(100, 4))
    queries = rng.normal(size=(30, 4)) * 1.3
    enc = KeyEncoder.fit({"vlm_img_nolang": keys, "proprio": np.zeros((100, 2))}, n_components=4)
    decisions, taus = [], []
    for scale in (1.0, 100.0):
        mem = EmergentMemory(enc, MemoryConfig(), horizon=10)
        for i, k in enumerate(keys * scale):
            mem.add(k, np.zeros(3), episode_id=f"e{i // 5}", init_state_id=_sid(i // 5), step=10 * (i % 5))
        taus.append(mem.calibrate())
        decisions.append([mem.retrieve(q * scale, episode_id=f"q{j}", init_state_id=_sid(1000 + j)).novel
                          for j, q in enumerate(queries)])
        assert mem.tau == pytest.approx(np.quantile(mem.nn_distances(), 0.9))
    assert decisions[0] == decisions[1] and 0 < sum(decisions[0]) < len(queries)
    assert taus[1] == pytest.approx(100 * taus[0])


def test_add_only_novel_executed_full_interval_and_log():
    rng = np.random.default_rng(1)
    mem, enc = _memory(rng.normal(size=(50, 4)))
    n0 = len(mem.store)
    near = mem.store.entries[0].key + 1e-6
    far = mem.store.entries[0].key + 1e3
    kw = dict(init_state_id=_sid(999), step=0)
    _, added = mem.observe(near, np.ones(3), executed=True, interval_len=10, episode_id="new1", **kw)
    assert added is None and mem.log[-1]["reason"] == "near_existing"
    _, added = mem.observe(far, np.ones(3), executed=False, interval_len=10, episode_id="new2", **kw)
    assert added is None and mem.log[-1]["reason"] == "not_executed"
    _, added = mem.observe(far, np.ones(3), executed=True, interval_len=4, episode_id="new3", **kw)
    assert added is None and mem.log[-1]["reason"] == "interval_truncated"
    _, added = mem.observe(far, np.full(3, np.nan), executed=True, interval_len=10, episode_id="new4", **kw)
    assert added is None and mem.log[-1]["reason"] == "z_missing"
    assert len(mem.store) == n0
    _, added = mem.observe(far, np.ones(3), executed=True, interval_len=10, episode_id="new5", **kw)
    assert added is not None and len(mem.store) == n0 + 1
    event = mem.log[-1]
    for field in ("added", "nearest_distance", "tau", "quantile", "candidate_ids", "distances", "weights"):
        assert field in event
    assert event["nearest_distance"] > event["tau"]


def test_add_rate_matches_quantile_on_exchangeable_data():
    rng = np.random.default_rng(2)
    keys = rng.normal(size=(600, 3))
    mem, enc = _memory(keys, MemoryConfig(threshold_quantile=0.8), per_episode=1)
    added = 0
    queries = rng.normal(size=(600, 3))
    pk = enc.transform({"vlm_img_nolang": queries, "proprio": np.zeros((600, 2))})
    for j, k in enumerate(pk):
        _, e = mem.observe(k, np.zeros(3), executed=True, interval_len=10, episode_id=f"q{j}",
                           init_state_id=_sid(10_000 + j), step=0)
        added += e is not None
    assert 0.12 < added / 600 < 0.28  # about 1 - 0.8


def test_topk_blend_and_exclusions():
    keys = np.arange(20, dtype=float)[:, None] * np.array([[1.0, 0.0]])
    mem, enc = _memory(keys, MemoryConfig(k=3, temperature=0.5), per_episode=1)
    q = mem.store.entries[5].key
    r = mem.retrieve(q, episode_id="e5", init_state_id=_sid(5))
    assert len(r.candidate_ids) == 3 and r.weights.sum() == pytest.approx(1.0)
    assert mem.store.entries[5].metadata.entry_id not in r.candidate_ids  # same episode excluded
    values = {e.metadata.entry_id: e.value for e in mem.store.entries}
    np.testing.assert_allclose(r.z, sum(w * values[c] for w, c in zip(r.weights, r.candidate_ids)), rtol=1e-5)
    assert r.weights[0] >= r.weights[-1]
    r2 = mem.retrieve(q, episode_id="other", init_state_id=_sid(500))
    assert mem.store.entries[5].metadata.entry_id == r2.candidate_ids[0]


def test_fixed_vs_adaptive_threshold_and_persistence(tmp_path):
    rng = np.random.default_rng(3)
    mem, _ = _memory(rng.normal(size=(40, 3)))
    tau = mem.tau
    for j in range(5):
        mem.observe(rng.normal(size=3) * 10, np.zeros(3), executed=True, interval_len=10, episode_id=f"n{j}",
                    init_state_id=_sid(900 + j), step=0)
    assert mem.tau == tau  # fixed mode does not move
    ad, _ = _memory(rng.normal(size=(40, 3)), MemoryConfig(threshold_mode="adaptive"))
    t0 = ad.tau
    for j in range(5):
        ad.observe(rng.normal(size=3) * 10, np.zeros(3), executed=True, interval_len=10, episode_id=f"n{j}",
                   init_state_id=_sid(900 + j), step=0)
    ad.retrieve(np.zeros(3), episode_id="x", init_state_id=_sid(0))
    assert ad.tau != t0
    mem.save(tmp_path / "m")
    loaded = EmergentMemory.load(tmp_path / "m")
    q = rng.normal(size=3)
    a = mem.retrieve(q, episode_id="x", init_state_id=_sid(1), count_usage=False)
    b = loaded.retrieve(q, episode_id="x", init_state_id=_sid(1), count_usage=False)
    assert a.candidate_ids == b.candidate_ids and loaded.tau == mem.tau
    np.testing.assert_allclose(a.z, b.z)


def test_memory_threshold_requires_minimum_entries():
    mem, _ = _memory(np.random.default_rng(4).normal(size=(5, 3)), MemoryConfig(min_entries_for_threshold=10))
    assert mem.tau is None
    _, added = mem.observe(np.zeros(3), np.zeros(3), executed=True, interval_len=10, episode_id="z",
                           init_state_id=_sid(77), step=0)
    assert added is not None and mem.log[-1]["reason"] == "added_no_threshold_yet"
