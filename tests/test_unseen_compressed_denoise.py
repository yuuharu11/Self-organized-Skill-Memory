import importlib.util
from pathlib import Path

import numpy as np
import pytest

P = (Path(__file__).resolve().parents[1] /
     'skill_moe/skillnet/examples/memory_pilot/unseen_libero/compressed_denoise.py')
S = importlib.util.spec_from_file_location('unseen_compressed_denoise_test', P)
M = importlib.util.module_from_spec(S)
S.loader.exec_module(M)


def test_active7_actions_are_zero_padded_in_model_space():
    a = np.ones((2, 10, 7), np.float32)
    y = M._pad_actions(a)
    assert y.shape == (2, 10, 32)
    np.testing.assert_array_equal(y[:, :, :7], a)
    assert np.count_nonzero(y[:, :, 7:]) == 0
    np.testing.assert_array_equal(M._pad_actions(y), y)


def test_procrustes_recovers_shared_rotation_and_scale():
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(12)
    shape = rng.normal(size=(10, 6))
    rot = Rotation.from_rotvec([.2, -.1, .3]).as_matrix()
    scale = 1.7
    target = np.concatenate([shape[:, :3] @ rot, shape[:, 3:6] @ rot], axis=1) * scale
    theta, aligned, err = M._procrustes(shape, target)
    assert theta.shape == (4,)
    np.testing.assert_allclose(aligned, target, atol=1e-5)
    assert err < 1e-10
    assert theta[3] == pytest.approx(np.log(scale), abs=1e-5)


def test_variable_draft_planner_can_cross_unit_boundary_and_excludes_task():
    class Memory:
        form = 'as_is'
        lengths = np.array([5, 5], np.int32)

    bank = {'memory': Memory(), 'task': np.array([2, 3]), 'source_row': np.array([21, 31]),
            'source_theta': np.zeros((2, 4), np.float32)}
    a = np.zeros((5, 7), np.float32); a[:, 0] = .1
    b = np.zeros((5, 7), np.float32); b[:, 1] = .2
    target = np.concatenate([a, b])
    selected = M._select_sequence(bank, [a, b], target, 0, 1,
                                  form='as_is', seed=4)
    assert len(selected['records']) >= 2
    assert np.all(selected['unit_ids'][:5] == 0)
    assert np.all(selected['unit_ids'][5:] == 1)
    assert all(rec['source_task'] != 1 for rec in selected['records'])
    np.testing.assert_allclose(selected['draft'], target, atol=1e-6)


def test_unseen_numeric_task_id_does_not_exclude_libero40_memory_task():
    class Memory:
        form = 'as_is'
        lengths = np.array([10, 10], np.int32)

    bank = {'memory': Memory(), 'task': np.array([2, 3]),
            'source_row': np.array([21, 31]),
            'source_theta': np.zeros((2, 4), np.float32)}
    a = np.zeros((10, 7), np.float32); a[:, 0] = .2
    b = np.zeros((10, 7), np.float32); b[:, 1] = .2
    # In a distinct suite namespace, query task 2 can use LIBERO-40 task 2.
    selected = M._select_sequence(bank, [a, b], a, 0, 2, form='as_is',
                                  exclude_task=False)
    assert selected['records'][0]['source_task'] == 2
    # Training rows share the memory namespace and still exclude that task.
    selected_train = M._select_sequence(bank, [a, b], a, 0, 2, form='as_is',
                                        exclude_task=True)
    assert selected_train['records'][0]['source_task'] == 3


def test_candidate_oracle_uses_global_memory_row_ids_and_excludes_task():
    actions = np.zeros((8, 10, 32), np.float32)
    actions[2, :, 0] = .2
    actions[5, :, 0] = .01
    actions[7, :, 0] = .1
    tasks = np.array([0, 0, 1, 1, 2, 2, 2, 3])
    qid = np.array([6])  # query task 2; row 5 itself is excluded with the full task.
    qtask = np.array([2])
    picked, draft, err = M._raw_oracle(actions, qid, qtask,
                                       np.array([2, 5, 7]), tasks)
    assert picked.tolist() == [7]
    assert draft.shape == (1, 10, 7)
    assert err[0] == pytest.approx(.01/7)
    picked_all, _, _ = M._raw_oracle(actions, qid, qtask,
                                     np.array([2, 5, 7]), tasks,
                                     exclude_task=False)
    assert picked_all.tolist() == [5]


def test_shape_draft_uses_source_pose_not_target_oracle_pose():
    from scipy.spatial.transform import Rotation
    import variable_shape_memory as vs

    class Memory:
        form = 'rms'
        lengths = np.array([10], np.int32)

    rng = np.random.default_rng(42)
    shape = np.zeros((10, 7), np.float32)
    shape[:, :6] = rng.normal(size=(10, 6)).astype(np.float32)
    shape[:, 6] = 1.
    source_theta = np.zeros((1, 4), np.float32)
    bank = {'memory': Memory(), 'task': np.array([99]),
            'source_row': np.array([17]), 'source_theta': source_theta}
    rot = Rotation.from_rotvec([.3, -.2, .1]).as_matrix()
    target_a = shape.copy()
    target_a[:, :6] = np.concatenate([shape[:, :3] @ rot,
                                      shape[:, 3:6] @ rot], axis=1) * 1.4
    target_a[:, :6] = vs.to_normalized(target_a[:, :6])
    target_b = shape.copy()
    target_b[:, :6] = np.concatenate([shape[:, :3] @ rot.T,
                                      shape[:, 3:6] @ rot.T], axis=1) * .7
    target_b[:, :6] = vs.to_normalized(target_b[:, :6])

    got_a = M._select_sequence(bank, [shape], target_a, 0, 2, form='rms',
                               exclude_task=False)
    got_b = M._select_sequence(bank, [shape], target_b, 0, 2, form='rms',
                               exclude_task=False)
    expected_motion = M._apply_theta(shape, source_theta[0], vs)
    expected = np.concatenate([expected_motion, shape[:, 6:7]], axis=1)
    np.testing.assert_allclose(got_a['draft'], expected, atol=1e-6)
    np.testing.assert_allclose(got_b['draft'], expected, atol=1e-6)
    assert not np.allclose(got_a['theta_target'], got_b['theta_target'])


def test_condition_list_and_t0_ranges_are_fixed():
    assert len(M.CONDS) == 10
    assert M._t0_range('A-vm-08-hi') == (.2, .6)
    assert M._t0_range('A-vm-08') == (.05, .4)
    assert M._t0_range('C-vm-08') == (0., 1.)


def test_theta_normalization_uses_only_valid_training_tokens():
    src=np.zeros((2,3,4),np.float32)
    tgt=np.arange(24,dtype=np.float32).reshape(2,3,4)
    units=np.array([[0,0,-1],[1,-1,-1]])
    mean,std=M._theta_stats(src,tgt,units)
    np.testing.assert_allclose(mean,tgt[units>=0].mean(0))
    np.testing.assert_allclose(std,tgt[units>=0].std(0))


def test_unit_pose_diagnostic_handles_multiple_units_and_active7():
    import variable_shape_memory as vs
    shape=np.zeros((10,7),np.float32);shape[:,0]=1.;shape[:,6]=1.
    theta=np.zeros((10,4),np.float32);theta[5:,3]=np.log(2.)
    decoded=M._decode_theta_tokens(shape,theta,vs)
    assert decoded.shape==(10,7)
    np.testing.assert_allclose(decoded[:5,:6],M._apply_theta(shape[:5],theta[0],vs),atol=1e-6)
    np.testing.assert_allclose(decoded[5:,:6],M._apply_theta(shape[5:],theta[5],vs),atol=1e-6)
    np.testing.assert_array_equal(decoded[:,6],shape[:,6])


def test_vectorized_oracle_matches_scalar_scoring_and_ties():
    import variable_shape_memory as vs
    rng=np.random.default_rng(820)
    for n in [1,5,10]:
        cand=rng.normal(size=(19,n,7)).astype(np.float32)
        cand[1]=cand[0]
        target=rng.normal(size=(n,7)).astype(np.float32)
        for form in ['as_is','rms']:
            actual=M._candidate_errors(cand,target,form)
            if form=='as_is':expected=np.array([np.mean((c-target)**2) for c in cand])
            else:
                expected=[]
                for c in cand:
                    _,aligned,_=M._procrustes(c[:,:6],vs.to_physical(target[:,:6]))
                    pred=np.concatenate([vs.to_normalized(aligned),c[:,6:7]],axis=1)
                    expected.append(np.mean((pred-target)**2))
                expected=np.array(expected)
            np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-7)
            assert np.argmin(actual)==np.argmin(expected)
            assert actual[0]==actual[1]


def test_theta_filter_preserves_frozen_vision_head():
    from openpi.shared.nnx_utils import PathRegex
    filt=PathRegex(M.THETA_FILTER)
    assert filt(('head','kernel'),None)
    assert filt(('head','bias'),None)
    assert not filt(('base','PaliGemma','img','head','kernel'),None)
    assert not filt(('base','action_out_proj','kernel'),None)


def test_parallel_selection_preserves_row_order_and_seeded_units():
    from concurrent.futures import ThreadPoolExecutor
    class Memory:
        form='rms'
        lengths=np.array([5,5,10],np.int32)
    rng=np.random.default_rng(992)
    decoded=[rng.normal(size=(n,7)).astype(np.float32) for n in Memory.lengths]
    bank={'memory':Memory(),'task':np.array([1,2,3]),'source_row':np.array([10,20,30]),'source_theta':np.zeros((3,4),np.float32)}
    target=rng.normal(size=(40,7)).astype(np.float32)
    def one(i):return M._select_sequence(bank,decoded,target,i,0,form='rms',random_pick=True,seed=100+i)
    serial=[one(i) for i in range(20)]
    with ThreadPoolExecutor(max_workers=8) as pool:parallel=list(pool.map(one,range(20)))
    for a,b in zip(serial,parallel):
        np.testing.assert_array_equal(a['draft'],b['draft'])
        assert a['records']==b['records']


def test_query_windows_keep_long_unit_phase_and_do_not_reselect_at_window_boundary():
    class Memory:
        form='as_is'
        lengths=np.array([15],np.int32)
    unit=np.zeros((15,7),np.float32);unit[:,0]=np.arange(15)/100
    bank={'memory':Memory(),'task':np.array([3]),'source_row':np.array([40]),'source_theta':np.zeros((1,4),np.float32)}
    trace=np.concatenate([unit,unit,unit])
    left=M._select_sequence(bank,[unit],trace,0,1,form='as_is',seed=4)
    middle=M._select_sequence(bank,[unit],trace,10,1,form='as_is',seed=4)
    right=M._select_sequence(bank,[unit],trace,20,1,form='as_is',seed=4)
    np.testing.assert_array_equal(np.concatenate([left['draft'],middle['draft'],right['draft']]),trace[:30])
    assert middle['records'][0]['phase']==10
    assert middle['records'][0]['length']==5
    assert middle['records'][1]['episode_start']==15
    assert right['records'][0]['phase']==5


def test_episode_plan_is_independent_of_query_order_and_preserves_random_units():
    class Memory:
        form='as_is'
        lengths=np.array([5,15],np.int32)
    units=[np.full((5,7),.1,np.float32),np.full((15,7),.2,np.float32)]
    def bank():return {'memory':Memory(),'task':np.array([2,3]),'source_row':np.array([1,2]),'source_theta':np.zeros((2,4),np.float32)}
    trace=np.zeros((50,7),np.float32);a=bank();b=bank()
    first=M._select_sequence(a,units,trace,30,1,form='as_is',random_pick=True,seed=55)
    M._select_sequence(b,units,trace,0,1,form='as_is',random_pick=True,seed=55)
    second=M._select_sequence(b,units,trace,30,1,form='as_is',random_pick=True,seed=55)
    np.testing.assert_array_equal(first['draft'],second['draft']);assert first['records']==second['records']


def test_paired_bootstrap_restricts_episode_ids_to_the_hard_subset():
    draft=np.arange(6,dtype=float);generated=draft+.5;episode=np.repeat([10,20,30],2)
    hard=np.array([True,True,False,False,False,False])
    out=M._bootstrap_all_and_hard(generated,draft,episode,hard)
    assert out['all']['rows']==6 and out['all']['episodes']==3
    assert out['hard']['rows']==2 and out['hard']['episodes']==1
    assert out['all']['mean']==pytest.approx(.5)
    np.testing.assert_allclose(out['hard']['ci95'],[.5,.5])
    assert M._bootstrap_all_and_hard(generated,draft,episode,np.zeros(6,bool))['hard'] is None
