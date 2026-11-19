import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'skill_moe/skillnet/examples/memory_pilot/unseen_libero'))
import vq_geometry as vg


def test_goal_entities_order_and_dedup(tmp_path):
    p = tmp_path / 'x.bddl'
    p.write_text('(define (problem p) (:obj_of_interest a b)\n (:goal (And (On moka_pot_1 flat_stove_1_cook_region)'
                 ' (On moka_pot_2 flat_stove_1_cook_region) (Turnon flat_stove_1)) ) )')
    ents, atoms = vg.goal_entities(str(p))
    assert ents == ['moka_pot_1', 'flat_stove_1_cook_region', 'moka_pot_2', 'flat_stove_1']
    assert atoms[0] == ('On', 'moka_pot_1', 'flat_stove_1_cook_region')
    assert atoms[2] == ('Turnon', 'flat_stove_1')


def test_noop_keep_openvla_rule():
    a = np.zeros((5, 7), np.float32)
    a[:, -1] = [-1, -1, 1, 1, 1]
    a[3, 0] = .5
    keep = vg.noop_keep(a)
    # OpenVLA: before any kept action a zero arm action is a no-op whatever the gripper;
    # afterwards it is kept only if the gripper differs from the last kept action
    assert keep.tolist() == [False, False, False, True, False]
    b = np.zeros((3, 7), np.float32); b[0, 0] = .5; b[:, -1] = [-1, 1, 1]
    assert vg.noop_keep(b).tolist() == [True, True, False]


def test_subsets_definitions():
    n = 4
    h = np.zeros((n, 10, 7), np.float32)
    m = np.ones((n, 10), bool)
    a = np.zeros((n, 10, 7), np.float32)
    h[..., 6] = -1
    a[..., 6] = -1
    m[0] = False                      # first unit
    h[1, 5:, 6] = 1; a[1, :, 6] = 1   # gripper switch inside history
    h[2, :, 0] = 1; a[2, :, 0] = -1   # direction reversal
    h[3, :, 0] = 1; a[3, :, 0] = 1    # straight continuation
    s = vg.subsets({'history': h, 'history_mask': m, 'actions': a})
    assert s['first'].tolist() == [True, False, False, False]
    assert s['switch'].tolist() == [False, True, False, False]
    assert s['turn'].tolist() == [False, False, True, False]
    assert s['post'].tolist() == [False, True, True, False]


def test_film_identity_init_matches_plain_tower():
    _, Q = vg._models()
    torch.manual_seed(0)
    q = Q(5, 'film')
    hs, g = torch.randn(3, 88), torch.randn(3, 5)
    ref = torch.nn.functional.normalize(q.l2(torch.nn.functional.gelu(q.l1(hs))), dim=-1)
    assert torch.allclose(q(hs=hs, geo=g), ref, atol=1e-6)
    assert torch.allclose(q(hs=hs, geo=g), q(hs=hs, geo=torch.zeros_like(g)), atol=1e-6)


def test_paired_bootstrap_sign():
    a = np.ones(20) * .2
    b = np.ones(20) * .1
    m, lo, hi = vg.paired_bootstrap(a, b, np.repeat(np.arange(5), 4), n=200)
    assert abs(m - .1) < 1e-9 and lo <= m <= hi


def test_rate_matched_rows_align_with_original_keys():
    import pytest
    d = vg.OUT / 'unseen20'
    if not (d / 'DONE').exists():
        pytest.skip('unseen20 not built')
    r = vg.load_rate_matched('unseen20')
    orig = np.load(vg.KF_DATA / 'unseen__key.npy')
    assert np.array_equal(r['key'], orig[r['row']])
    assert r['actions'].shape[1:] == (10, 7) and r['history'].shape[1:] == (10, 7)
    # frame-0 rows have no history; others have all 10 steps at 20 Hz
    f0 = r['key'][:, 2] == 0
    assert not r['history_mask'][f0].any()
