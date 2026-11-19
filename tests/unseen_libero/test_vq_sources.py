import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'skill_moe/skillnet/examples/memory_pilot/unseen_libero'))
import key_final as kf
import vq_sources as vq


def _split(n=6, seed=0):
    rng = np.random.default_rng(seed)
    return {'history': rng.normal(size=(n, 10, 7)).astype(np.float32),
            'history_mask': rng.random((n, 10)) > .3,
            'state': rng.normal(size=(n, 8)).astype(np.float32)}


def test_hs_matches_c1b_features():
    s = _split()
    assert np.array_equal(vq._hs(s), kf.hs_features(s))


def test_film_identity_at_init_equals_c1b_tower():
    s = _split(5)
    hs = torch.as_tensor(vq._hs(s))
    lang = torch.randn(5, 2048)
    img = torch.randn(5, 32, 2048)
    torch.manual_seed(3)
    base = kf.HSMLPQuery('x1')
    for cond in ({}, {'L': 'L1b'}, {'I': 'I1'}, {'L': 'L1c', 'I': 'I2'}):
        torch.manual_seed(3)
        m = vq.build_model(cond)
        # same parameter init order as C1b for the shared tower
        assert torch.equal(m.l1.weight, base.tower[0].weight)
        assert torch.equal(m.l2.weight, base.tower[2].weight)
        with torch.no_grad():
            q = m(hs, lang if cond.get('L') else None, img if cond.get('I') else None)
            assert torch.allclose(q, base(hs=hs), atol=1e-6)


def test_film_condition_changes_query_after_update():
    m = vq.build_model({'L': 'L1b', 'I': 'I1'})
    with torch.no_grad():
        m.film[-1].weight.normal_()
    hs = torch.randn(2, 88)
    a = m(hs, torch.randn(2, 2048), torch.randn(2, 32, 2048))
    b = m(hs, torch.randn(2, 2048), torch.randn(2, 32, 2048))
    assert not torch.allclose(a, b)


def test_subsets_definitions():
    n = 4
    s = {'key': np.array([[0, 0, 0], [0, 0, 30], [0, 0, 40], [0, 0, 50]]),
         'history': np.zeros((n, 10, 7), np.float32), 'history_mask': np.ones((n, 10), bool),
         'actions': np.zeros((n, 10, 7), np.float32)}
    s['history_mask'][0] = False
    s['history'][:, :, 6] = -1; s['actions'][:, :, 6] = -1
    s['actions'][1, :, 6] = 1                       # gripper switch at now
    s['history'][2, :, 0] = .5; s['actions'][2, :, 0] = -.5   # xyz reversal
    s['history'][3, :, 0] = .5; s['actions'][3, :, 0] = .5    # continuing
    sub = vq.subsets(s)
    assert sub['first_nohist'].tolist() == [True, False, False, False]
    assert sub['early'].tolist() == [True, False, False, False]
    assert sub['grip_switch'].tolist() == [False, True, False, False]
    assert sub['dir_reverse'].tolist() == [False, False, True, False]
    assert sub['post_switch'].tolist() == [False, True, True, False]
