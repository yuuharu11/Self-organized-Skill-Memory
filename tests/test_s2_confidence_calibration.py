"""Focused S2 calibration, candidate caching, and actual sampler-call tests."""
from pathlib import Path
import itertools
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / 'skill_moe/skillnet/examples/memory_pilot/anchor_flow'
sys.path.insert(0, str(HERE))
from confidence_calibration import calibrated_t0, denoising_steps, fit_monotone, load_calibration
from memory import _retrieved_components, prepare_online_query


@pytest.mark.parametrize('t0,n', [(0., 2), (.05, 2), (.15, 2), (.2, 2), (.25, 2), (.4, 4), (.6, 6), (.8, 8), (1., 10)])
def test_steps(t0, n):
    assert denoising_steps(t0) == n


def test_fit_matches_exhaustive_monotone_optimum():
    confidence = np.arange(12.)
    errors = np.random.default_rng(37).uniform(size=(3, 12))
    grid = [.2, .6, 1.]
    fit = fit_monotone(confidence, errors, grid, bins=3)
    best = np.inf
    for choices in itertools.product(range(3), repeat=3):
        if any(grid[choices[i]] < grid[choices[i+1]] for i in range(2)):
            continue
        best = min(best, sum(errors[choices[b], b*4:(b+1)*4].sum() for b in range(3))/12)
    assert fit['validation_mse'] == pytest.approx(best)
    assert all(x >= y for x, y in zip(fit['t0'], fit['t0'][1:]))


def test_lookup_and_reject_invalid_tables(tmp_path):
    p = tmp_path/'table.json'
    table = dict(suite='libero90', query_text_source='instruction_only', edges=[.5],
                 t0=[.8, .2], metric='one_minus_normalized_entropy', temperature=.001)
    p.write_text(json.dumps(table))
    table = load_calibration(p)
    assert calibrated_t0(table, .9, .9)[:2] == (.8, 0)
    assert calibrated_t0(table, .9, .4)[:2] == (.2, 1)
    table['t0'] = [.2, .8]; p.write_text(json.dumps(table))
    with pytest.raises(ValueError): load_calibration(p)


def test_empty_public_object_annotation_is_integer_and_observation_valid():
    import jax.numpy as jnp
    from openpi.models.model import ModelType, Observation
    from openpi.policies.libero_policy import LiberoInputs
    from s2_policy import integer_object_ids
    raw = {'observation/state':np.zeros(8,np.float32),
           'observation/image':np.zeros((224,224,3),np.uint8),
           'observation/wrist_image':np.zeros((224,224,3),np.uint8),
           'objects':'[]', 'skills':'[0]'}
    transformed = LiberoInputs(ModelType.PI05)(raw)
    assert transformed['objects'].dtype.kind == 'f'
    fixed = integer_object_ids(transformed)
    assert fixed['objects'].dtype == np.int32
    import jax
    obs = Observation.from_dict(jax.tree.map(lambda x:jnp.asarray(x)[None],fixed))
    assert obs.objects.dtype == jnp.int32
    np.testing.assert_array_equal(obs.objects,np.zeros((1,4),np.int32))
    original=np.asarray([0,1,2,3],np.int64)
    np.testing.assert_array_equal(integer_object_ids({'objects':original})['objects'],original)
    with pytest.raises(ValueError):integer_object_ids({'objects':np.array([.5])})


def test_cached_candidates_equal_original_with_ties_and_exclusions():
    rng = np.random.default_rng(19)
    rows = [dict(key=rng.normal(size=8), base=np.zeros(2), unit_id=i%3,
                 meta=dict(episode_id=f'e{i%7}', init_state_id=f's{i%5}')) for i in range(40)]
    rows[9]['key'] = rows[8]['key'].copy()
    bank = SimpleNamespace(memory=rows, temperature=.002)
    for key in [rng.normal(size=8), rows[8]['key']]:
        meta = dict(episode_id='e3', init_state_id='s1')
        expected, weights = _retrieved_components(bank, key, top_k=5, meta=meta)
        actual, cached_weights = _retrieved_components(bank, key, top_k=5,
                            meta=prepare_online_query(bank, key, meta))
        assert [id(x) for x in actual] == [id(x) for x in expected]
        np.testing.assert_array_equal(weights, cached_weights)


@pytest.mark.parametrize('condition', ['anchor_entropy_current_memory', 'anchor_memory_shuffled', 'anchor_disabled_t0_1'])
@pytest.mark.parametrize('t0', [.2, .6, 1.])
def test_infer_passes_actual_steps_and_logs_confidence(monkeypatch, condition, t0):
    import jax.numpy as jnp
    import s2_policy as s
    p = s.S2WebsocketPolicy.__new__(s.S2WebsocketPolicy)
    p._query_index = 0; p._seed = 7
    p._model = SimpleNamespace(action_horizon=10, action_dim=32)
    p._input_transform = lambda raw: {'state': np.zeros(32, np.float32)}
    p._output_transform = lambda out: out
    p._encode_key = lambda obs: np.array([[1., 0.]])
    p._calibration = dict(metric='max_cosine_similarity', edges=[], t0=[t0], temperature=.001)
    p._memory_for = lambda *args: SimpleNamespace(key_text_source='instruction_only')
    class ObservationStub(SimpleNamespace):
        def replace(self, **updates):
            return type(self)(**{**vars(self), **updates})

    monkeypatch.setattr(s._model.Observation, 'from_dict', lambda inputs: ObservationStub(state=inputs['state']))
    monkeypatch.setattr(s, 'prepare_online_query', lambda memory, key, meta: meta)
    monkeypatch.setattr(s, 'retrieval_confidence', lambda *a, **kw: (.83, .17))
    monkeypatch.setattr(s, 'retrieval_records', lambda *a, **kw: [])
    monkeypatch.setattr(s, 'sample_memory_anchor', lambda *a, **kw: (np.zeros((10,32)), dict(unit_id=3, task='libero90')))
    calls = []
    def sample(*args, **kw):
        calls.append(kw)
        return jnp.zeros((1, 10, 32))
    p._sample_actions = sample
    out = p.infer(dict(prompt='open the drawer', s2_condition=condition, s2_rng_seed=123))
    debug = out['s2_debug']
    expected = 10 if condition == 'anchor_disabled_t0_1' else denoising_steps(t0)
    assert calls[0]['num_steps'] == debug['n'] == debug['num_steps'] == expected
    if condition == 'anchor_disabled_t0_1':
        assert debug['t0'] == 1.
        assert debug['normalized_retrieval_entropy'] is None
        np.testing.assert_array_equal(calls[0]['noise'], calls[0]['x_src'])
    else:
        assert debug['t0'] == t0
        assert debug['normalized_retrieval_entropy'] == .17
        assert debug['max_cosine_similarity'] == .83
