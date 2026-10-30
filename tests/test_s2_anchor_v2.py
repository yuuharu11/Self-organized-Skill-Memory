import importlib.util
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
ROOT = Path(__file__).resolve().parents[1] / "skill_moe/skillnet/examples/memory_pilot/anchor_flow"
import sys
sys.path.insert(0, str(ROOT))
from memory import (anchor_v2_components, anchor_v2_t0, prepare_centered_online_query,
                    prepare_online_query, sample_memory_anchor)
from openpi.models import pi0_anchor_flow as pi0_anchor_flow_module
from openpi.models.pi0_anchor_flow import Pi0AnchorFlow

spec = importlib.util.spec_from_file_location("s2_policy_for_test", ROOT / "s2_policy.py")
s2_policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s2_policy)


def _fake_memory():
    mem = SimpleNamespace()
    mem.temperature = .5
    mem.decode_batch_sizes = []
    mem.memory = []
    for i, key in enumerate(([1., 0.], [.8, .2], [0., 1.], [-1., 0.], [0., -1.], [.2, .8])):
        mem.memory.append({"key": np.asarray(key, np.float32), "base": np.array([i, 1.], np.float32),
            "unit_id": i % 3, "meta": {"episode_id": f"ep{i}", "init_state_id": f"init{i}",
                                        "task": f"task{i}", "raw_action": np.full((10, 32), i, np.float32)}})
    mem.prior = lambda key, state: (np.zeros((1, 2), np.float32), np.ones((1, 1), np.float32))
    def decode(b, r, f, s):
        b = np.asarray(b)
        mem.decode_batch_sizes.append(len(b))
        return np.broadcast_to(b[:, 0, None, None], (len(b), 10, 32)).copy()
    mem.decode = decode
    mem._centers = [np.array([float(i), 1.], np.float32) for i in range(3)]
    return mem


def test_centered_vectorized_query_matches_naive_and_excludes_both_ids():
    mem = _fake_memory()
    query = np.array([1., .1], np.float32)
    meta = {"episode_id": "ep0", "init_state_id": "init1"}
    prepared = prepare_centered_online_query(mem, query, meta, top_k=5, temperature=.03)
    keys = np.stack([r["key"] for r in mem.memory]).astype(np.float32)
    mean = keys.mean(axis=0)
    keys -= mean
    keys /= np.maximum(np.linalg.norm(keys, axis=1, keepdims=True), 1e-8)
    q = query - mean
    q /= np.linalg.norm(q)
    sims = keys @ q
    for i, row in enumerate(mem.memory):
        if row["meta"]["episode_id"] == meta["episode_id"] or row["meta"]["init_state_id"] == meta["init_state_id"]:
            sims[i] = -np.inf
    expected = np.lexsort((np.arange(len(sims)), -sims))[:min(5, np.isfinite(sims).sum())]
    np.testing.assert_array_equal(prepared["_prepared_candidates"], expected)
    np.testing.assert_allclose(prepared["_anchor_v2_similarities"], sims[expected], atol=1e-7)


def test_anchor_drafts_are_weighted_and_log_top_five_units():
    mem = _fake_memory()
    meta = prepare_centered_online_query(mem, np.array([1., .2]),
        {"episode_id": "new", "init_state_id": "new-init"}, temperature=.03)
    latent, records = anchor_v2_components(mem, np.array([1., .2]), np.zeros(32), meta)
    batch_before = len(mem.decode_batch_sizes)
    each, each_records = anchor_v2_components(mem, np.array([1., .2]), np.zeros(32), meta,
                                               source="dec_each")
    raw, raw_records = anchor_v2_components(mem, np.array([1., .2]), np.zeros(32), meta, source="raw")
    assert latent.shape == each.shape == raw.shape == (10, 32)
    assert len(records) == len(each_records) == len(raw_records) == 5
    np.testing.assert_allclose(sum(x["weight"] for x in records), 1., atol=1e-6)
    assert all("similarity" in x and "unit_id" in x for x in records)
    assert mem.decode_batch_sizes[batch_before:] == [5]


def test_dec_each_matches_naive_loop_and_leaves_other_sources_unchanged():
    mem = _fake_memory()
    key = np.array([1., .2], np.float32)
    state = np.arange(32, dtype=np.float32)
    meta = prepare_centered_online_query(mem, key,
        {"episode_id": "new", "init_state_id": "new-init"}, temperature=.05)
    dec_latent_before, _ = anchor_v2_components(mem, key, state, meta, source="dec_latent")
    raw_before, _ = anchor_v2_components(mem, key, state, meta, source="raw")
    ids = meta["_prepared_candidates"]
    weights = meta["_anchor_v2_weights"]
    rows = [mem.memory[int(i)] for i in ids]
    _, residual = mem.prior(key[None], state[None])
    loop_values = []
    for row in rows:
        loop_values.append(mem.decode(np.asarray(row["base"], np.float32)[None], residual,
                                      key[None], state[None])[0])
    expected = np.einsum("k,khd->hd", weights, np.asarray(loop_values))
    mem.decode_batch_sizes.clear()
    actual, _ = anchor_v2_components(mem, key, state, meta, source="dec_each")
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
    assert mem.decode_batch_sizes == [5]
    dec_latent_after, _ = anchor_v2_components(mem, key, state, meta, source="dec_latent")
    raw_after, _ = anchor_v2_components(mem, key, state, meta, source="raw")
    np.testing.assert_array_equal(dec_latent_before, dec_latent_after)
    np.testing.assert_array_equal(raw_before, raw_after)


def test_similarity_gate_and_calibration_bins():
    table = {"edges": [.3, .6], "t0": [.8, .5, .2], "gate_threshold": .4}
    assert anchor_v2_t0(.2, mode="calibration", table=table) == (1., 0, .4)
    assert anchor_v2_t0(.5, mode="calibration", table=table) == (.5, 1, .4)
    assert anchor_v2_t0(.9, mode="fixed", fixed_t0=.4, gate_threshold=.4)[0] == .4
    assert anchor_v2_t0(.4, mode="fixed", fixed_t0=.4, gate_threshold=.4)[0] == 1.
    assert anchor_v2_t0(-.9, mode="one")[0] == 1.
    median_table = {"top1_similarity_edges": [.5], "t0_by_bin": [.6, .3],
                    "holdout_top1_similarities": [.2, .4, .6, .8]}
    assert anchor_v2_t0(.7, mode="calibration", table=median_table) == (.3, 1, .5)
    assert anchor_v2_t0(.5, mode="fixed", fixed_t0=.4, table=median_table) == (1., None, .5)


def test_t0_one_calls_the_exact_ordinary_sampler_path():
    calls = []
    def sampler(rng, obs, **kwargs):
        calls.append(kwargs)
        return np.asarray(kwargs["x_src"]).copy()
    noise = jnp.arange(320, dtype=jnp.float32).reshape(1, 10, 32)
    anchor = np.full((10, 32), -99., np.float32)
    actual = s2_policy.sample_anchor_v2_actions(sampler, "rng", "obs", noise=noise,
                                                 anchor=anchor, t0=1., num_steps=10)
    expected = sampler("rng", "obs", num_steps=10, noise=noise, x_src=noise, t0=1.)
    np.testing.assert_array_equal(actual, expected)
    assert calls[-2].keys() == calls[-1].keys()
    for key in calls[-1]:
        if key == "x_src" or key == "noise":
            np.testing.assert_array_equal(calls[-2][key], calls[-1][key])
        else:
            assert calls[-2][key] == calls[-1][key]
    np.testing.assert_array_equal(calls[-2]["x_src"], noise)


def test_legacy_condition_candidates_and_output_stay_identical_after_v2_index():
    mem = _fake_memory()
    key = np.array([1., .3], np.float32)
    meta = {"episode_id": "q", "init_state_id": "q-init"}
    before_meta = prepare_online_query(mem, key, meta)
    before = sample_memory_anchor(mem, key, np.zeros(32), seed=123, meta=before_meta)
    before_ids = before_meta["_prepared_candidates"].copy()
    prepare_centered_online_query(mem, key, meta, temperature=.03)
    after_meta = prepare_online_query(mem, key, meta)
    after = sample_memory_anchor(mem, key, np.zeros(32), seed=123, meta=after_meta)
    np.testing.assert_array_equal(before_ids, after_meta["_prepared_candidates"])
    np.testing.assert_array_equal(before, after)


def test_prefix_key_cache_matches_key_and_reuses_forward(monkeypatch):
    class ToyPaliGemma:
        def __init__(self):
            self.prefix_calls = 0

        def llm(self, inputs, **kwargs):
            prefix, suffix = inputs
            if prefix is not None:
                self.prefix_calls += 1
                cache = np.asarray(prefix).sum(axis=1)
                return ((prefix + 2.0, None), cache, None)
            return ((None, suffix + kwargs["kv_cache"][:, None, :1]), None, None)

    class ToyModel:
        action_horizon = 1
        action_dim = 1
        positional_encoding = False

        def __init__(self):
            self.PaliGemma = ToyPaliGemma()

        def embed_prefix(self, observation):
            tokens = np.array([[[1., 2., 3.], [4., 5., 6.]]], np.float32)
            valid = np.array([[True, True]])
            ar = np.zeros((1, 2), bool)
            return tokens, valid, ar

        def embed_suffix(self, observation, x, time):
            return x, np.ones((1, 1), bool), np.zeros((1, 1), bool), None

        def action_out_proj(self, x):
            return x

    toy = ToyModel()
    monkeypatch.setattr(pi0_anchor_flow_module._model, "preprocess_observation",
                        lambda rng, obs, train=False: obs)
    monkeypatch.setattr(pi0_anchor_flow_module, "make_attn_mask",
                        lambda valid, ar: np.asarray(valid)[:, None, :] & np.asarray(valid)[:, :, None])
    monkeypatch.setattr(pi0_anchor_flow_module, "integrate_velocity",
                        lambda fn, x, starts, steps: x + fn(x, starts[:, None, None]))
    observation = SimpleNamespace(state=np.zeros((1, 1), np.float32), skills=None, skill_mask=None)

    legacy_key = Pi0AnchorFlow.encode_prefix_key(toy, observation)
    combined_key, prefix_cache = Pi0AnchorFlow.encode_prefix_key_and_cache(toy, observation)
    np.testing.assert_array_equal(combined_key, legacy_key)
    assert toy.PaliGemma.prefix_calls == 2

    # Skill annotations are not part of the retrieval-key contract. Removing
    # their variable-length leaves before JIT must preserve the key exactly.
    from openpi.models import model as model_module
    annotated = model_module.Observation(
        images={}, image_masks={}, state=np.zeros((1, 1), np.float32),
        skills=np.array([[1, 2, 3]], np.int32), skill_mask=np.ones((1, 3), bool),
    )
    key_only = s2_policy.S2WebsocketPolicy._key_only_observation(annotated)
    np.testing.assert_array_equal(
        Pi0AnchorFlow.encode_prefix_key(toy, annotated),
        Pi0AnchorFlow.encode_prefix_key(toy, key_only),
    )
    assert key_only.skills is None and key_only.skill_mask is None

    source = np.array([[[.25]]], np.float32)
    toy.PaliGemma.prefix_calls = 0
    expected = Pi0AnchorFlow._integrate_from_anchor(
        toy, observation, x_src=source, t0=.5, num_steps=2,
    )
    assert toy.PaliGemma.prefix_calls == 1
    actual = Pi0AnchorFlow._integrate_from_anchor(
        toy, observation, x_src=source, t0=.5, num_steps=2,
        prefix_context=prefix_cache,
    )
    assert toy.PaliGemma.prefix_calls == 1
    np.testing.assert_array_equal(actual, expected)
