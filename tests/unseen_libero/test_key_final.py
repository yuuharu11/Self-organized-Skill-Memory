import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'skill_moe/skillnet/examples/memory_pilot/unseen_libero'))
import key_final as kf
import key_grid as kg


def _split(n=6):
    rng = np.random.default_rng(0)
    return {'history': rng.normal(size=(n, 10, 7)).astype(np.float32),
            'history_mask': rng.random((n, 10)) > .3,
            'state': rng.normal(size=(n, 8)).astype(np.float32)}


def test_hs_features_match_keygrid_vector():
    s = _split()
    assert np.array_equal(kf.hs_features(s), kg.make_vector_features(s, 'history_state'))


def test_query_models_shapes():
    s = _split(3)
    t = {k: torch.as_tensor(v) for k, v in s.items()}
    for query in ('mqe', 'rqe'):
        for key in ('x1_mk1b', 'dct'):
            m = kf.make_query({'query': query, 'input': 'e_h', 'key': key})
            q = m(history=t['history'], history_mask=t['history_mask'], state=t['state'])
            assert q.shape == (3, 18 if key == 'dct' else 256)
    m = kf.make_query({'query': 'hsmlp', 'input': 'e_h', 'key': 'x1_mk1c'})
    q = m(hs=torch.as_tensor(kf.hs_features(s)))
    assert q.shape == (3, 256) and torch.allclose(q.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_keygrid_key_equals_pairmlp_score():
    torch.manual_seed(0)
    ref = kg.PairMLPRanker(88, shared=False)
    key = kf.KeyGridKey()
    key.tower.load_state_dict(ref.key_tower.state_dict())
    q = torch.randn(4, 88); k = torch.randn(7, 88)
    qv = torch.nn.functional.normalize(ref.query_tower(q), dim=-1)
    assert torch.allclose(ref(q, k), qv @ key(k).T, atol=1e-5)


def test_metrics_rank_and_recall():
    scores = np.array([[3., 2., 1.], [1., 2., 3.]], np.float32)
    errors = np.array([[.1, .2, .3], [.1, .2, .3]], np.float32)
    split = {'task': np.array([0, 0]), 'hard': np.array([False, True])}
    m = kf.metrics(scores, errors, split, np.array([0, 1, 1]))
    assert m['all']['recall_at']['1'] == .5 and m['all']['oracle_rank_median'] == 2.
    assert abs(m['all']['top1_arm6_mse'] - .2) < 1e-6
    assert m['all']['other_task_top1_fraction'] == .5
    assert m['hard']['top1_arm6_mse'] == np.float32(.3)


def test_score_modes():
    q = torch.tensor([[1., 0.]]); k = torch.tensor([[1., 0.], [0., 1.]])
    s0 = torch.tensor(0.)
    assert torch.equal(kf.score(q, k, 'dot', s0), torch.tensor([[1., 0.]]))
    assert torch.allclose(kf.score(q, k, 'dct', s0), torch.tensor([[0., -2.]]))
