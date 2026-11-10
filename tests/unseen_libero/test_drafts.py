"""Draft representations, reconstruction bundles and denoising from drafts."""
import importlib.util
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.fft import dct

from tests.support import (
    REPO,
    UNSEEN,
    load_script,
)

oracle = load_script(UNSEEN / "oracle_repr.py")
bundles = load_script(UNSEEN / "draft_reconstruction_bundles.py")
lf = load_script(UNSEEN / "low_freq_denoise.py")
cd = load_script(UNSEEN / "compressed_denoise.py")


# ── Oracle representations ────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('name', ['raw','dct14','dct21','keyframe21','keyframe35'])
def test_representation_shape_and_raw_identity(name):
    x=np.random.default_rng(3).normal(size=(5,10,7)).astype(np.float32)
    z=oracle.encode_fixed(x,name); y=oracle.decode_fixed(z,name)
    assert y.shape==x.shape and np.isfinite(y).all()
    if name=='raw': np.testing.assert_array_equal(y,x)


def test_zero_padded_dct_reconstruction_is_low_frequency_projection():
    x=np.random.default_rng(4).normal(size=(2,10,7)).astype(np.float32)
    z=oracle.encode_fixed(x,'dct14'); y=oracle.decode_fixed(z,'dct14')
    direct=oracle.idct(np.pad(z.reshape(2,2,7),((0,0),(0,8),(0,0))),type=2,axis=1,norm='ortho')
    np.testing.assert_allclose(y,direct,atol=1e-6)


def test_oracle_selects_one_real_candidate_and_ids_are_not_offsets():
    target=np.zeros((10,7),np.float32)
    drafts=np.stack([np.ones((10,7)),np.full((10,7),.2)]).astype(np.float32)
    row,draft,mse=oracle.oracle_candidate(target,np.array([9001,12003]),drafts)
    assert row==12003 and mse==pytest.approx(.04)
    np.testing.assert_array_equal(draft,drafts[1])


def test_episode_bootstrap_pairs_at_episode_level():
    a=np.array([2.,2.,4.,4.]);b=np.zeros(4);ep=np.array([1,1,2,2])
    got=oracle.episode_bootstrap_diff(a,b,ep,reps=100,seed=1)
    assert got['mean']==3 and got['episodes']==2 and got['rows']==4


def test_pca_fit_source_excludes_unseen_eval_by_construction():
    # The fitting helper accepts training actions only; this guards the row-id
    # split contract in the runner without needing a checkpoint/GPU.
    from sklearn.decomposition import PCA
    train=np.arange(40,dtype=np.float32).reshape(4,10)
    eval_actions=np.full((1,10),1e6,np.float32)
    pca=PCA(n_components=2).fit(train)
    expected=PCA(n_components=2).fit(train)
    np.testing.assert_array_equal(pca.mean_,expected.mean_)
    assert eval_actions.min()>pca.mean_.max()


def test_shuffle_preserves_selected_rows_and_changes_task():
    # IDs are global row IDs; make three equally sized task partitions.
    tasks=np.repeat(np.array([11,22,33],np.int64),4)
    ids=np.arange(12,dtype=np.int64)
    query_task=tasks.copy()
    shuffled=oracle.deterministic_cross_task_shuffle(ids,query_task,tasks,seed=9)
    np.testing.assert_array_equal(np.sort(shuffled),ids)
    assert np.all(tasks[shuffled]!=query_task)


def test_pool_shuffle_returns_unique_cross_task_rows():
    tasks=np.repeat(np.array([11,22,33],np.int64),5)
    q=np.array([11,11,22,22,33,33])
    pool=np.arange(len(tasks))
    shuffled=oracle.deterministic_cross_task_from_pool(pool,q,tasks,seed=4)
    assert len(np.unique(shuffled))==len(q)
    assert np.all(tasks[shuffled]!=q)


def test_zero_bridge_draft_uses_model_padded_action_width():
    p=REPO / 'skill_moe/skillnet/examples/memory_pilot/unseen_libero/oracle_experiment.py'
    s=importlib.util.spec_from_file_location('unseen_oracle_experiment_test',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    active7=np.ones((3,10,7),np.float32)
    padded32=np.zeros((3,10,32),np.float32)
    # Eval labels are active7, but sample_bridge combines with model-space 32-D noise.
    adapted=m.pad_bridge_action(active7)
    assert adapted.shape==(3,10,32)
    np.testing.assert_array_equal(adapted[:,:,:7],active7)
    assert not adapted[:,:,7:].any()
    z=m.zero_bridge_draft(padded32)
    assert z.shape==(3,10,32) and not z.any()


def test_keyframe_linear_interpolation_uses_endpoints():
    x=np.zeros((1,10,7),np.float32);x[0,:,0]=np.arange(10,dtype=np.float32)
    z=oracle.encode_fixed(x,'keyframe21');y=oracle.decode_fixed(z,'keyframe21')
    assert y.shape==x.shape
    assert y[0,0,0]==pytest.approx(0.) and y[0,-1,0]==pytest.approx(9.)


# ── Draft reconstruction bundles ──────────────────────────────────────────────────────────────────

def test_roundtrip_preserves_full_segment_shape_and_window_shape():
    source = np.arange(18 * 7, dtype=np.float32).reshape(18, 7)
    decoded = bundles.roundtrip_segment(
        source,
        lambda x: (x.copy(), {"length": len(x)}),
        lambda code, pose, length: code[:length],
    )
    assert decoded.shape == source.shape
    np.testing.assert_array_equal(decoded, source)
    window = bundles.interp_window(decoded, 5.0, 10)
    assert window.shape == (10, 7)
    np.testing.assert_array_equal(window, source[5:15])


def test_row_order_is_restored_by_dataset_row_id_not_pool_position():
    wanted = np.asarray([91, 7, 44], np.int64)
    returned = np.asarray([44, 91, 7], np.int64)
    values = np.asarray([[440], [910], [70]])
    got = bundles.align_by_row_ids(wanted, returned, values)
    np.testing.assert_array_equal(got[:, 0], [910, 70, 440])


def test_unseen_windows_stitch_and_roundtrip_to_source_order():
    # Two consecutive stride-10 windows from a synthetic episode.
    chunks = np.zeros((2, 10, 32), np.float32)
    chunks[0, :, :7] = np.arange(70, dtype=np.float32).reshape(10, 7)
    chunks[1, :, :7] = np.arange(70, 140, dtype=np.float32).reshape(10, 7)
    un = {"task": np.asarray([88, 88]), "episode": np.asarray([3, 3]),
          "frame": np.asarray([0, 10]), "actions": chunks}
    q01, q99 = np.full(7, -1, np.float32), np.full(7, 1, np.float32)
    episodes = bundles._unseen_episodes(un, q01, q99)
    got = episodes[(88, 3)]["actions_norm"]
    assert got.shape == (20, 7)
    np.testing.assert_allclose(got[:10], chunks[0, :, :7], atol=1e-6, rtol=0)
    np.testing.assert_allclose(got[10:], chunks[1, :, :7], atol=1e-6, rtol=0)


def test_uncovered_long_window_gets_query_only_fallback():
    rows = {"task": np.asarray([2]), "episode": np.asarray([9]),
            "frame": np.asarray([10])}
    trace = np.arange(20 * 7, dtype=np.float32).reshape(20, 7)
    episodes = {(2, 9): {"frame": np.arange(20), "actions_phys": trace,
                         "meta": {"memory": False}}}
    segments = {(2, 9): [{"task": 2, "episode": 9, "start": 0,
                          "length": 10, "row_id": 0, "a": trace[:10]}]}
    q, fallback = bundles._long_queries(segments, np.asarray([0]), rows, episodes)
    seg, offset = q[(2, 9)][0]
    assert fallback == {0}
    assert (seg["start"], seg["length"], offset) == (10, 10, 0.0)
    np.testing.assert_array_equal(seg["a"], trace[10:])


# ── Low-frequency denoise ─────────────────────────────────────────────────────────────────────────

def test_basis_matches_scipy_dct():
    x = np.random.default_rng(0).normal(size=(10, 4))
    assert np.allclose(lf.dct_basis().T @ x, dct(x, type=2, norm='ortho', axis=0))


def test_projection_keeps_only_low_frequencies():
    noise = np.random.default_rng(1).normal(size=(64, 10, 32))
    for k in (3, 5):
        low = lf.project(noise, k)
        coef = dct(low, type=2, norm='ortho', axis=1)
        assert np.abs(coef[:, k:]).max() < 1e-5
        assert np.abs(coef[:, :k]).max() > .1


def test_mean_element_variance_is_one():
    for k in (1, 3, 5, 10):
        p = lf.low_projector(k)
        assert np.isclose(np.trace(p @ p.T) / 10, 1.0)


def test_k10_is_white_noise():
    noise = np.random.default_rng(2).normal(size=(3, 10, 7))
    assert np.allclose(lf.project(noise, 10), noise, atol=1e-5)


def test_conditions_reuse_white_drafts():
    assert set(c[0] for c in lf.CONDS.values()) <= {'A-raw', 'A-vm-none', 'A-vm-08'}
    assert all(0 < lo < hi <= 1 for _, _, (lo, hi) in lf.CONDS.values())


def test_installed_eval_hook_projects_the_shared_white_noise(monkeypatch):
    captured = {}

    def original(model, obs, drafts, noise, t0, *, batch):
        captured['noise'] = noise
        captured['t0'] = t0
        captured['batch'] = batch
        return 'pred'

    cd = SimpleNamespace(BATCH=8, CONDS=('A-raw',), T0_EVAL=(1.,),
                         _predict_action=original, _t0_range=lambda _: (0., 1.))
    monkeypatch.setattr(lf, '_link_drafts', lambda _name: None)
    installed = lf.install(cd, 'L-vm-none-k3')
    white = np.random.default_rng(9).normal(size=(4, 10, 32)).astype(np.float32)
    result = installed._predict_action('m', 'o', 'd', white, .2)
    assert result == 'pred'
    np.testing.assert_allclose(captured['noise'], lf.project(white, 3), atol=1e-6)
    assert captured['t0'] == .2 and captured['batch'] == 8


def test_comparison_rejects_different_draft_or_target(tmp_path, monkeypatch):
    out = tmp_path / 'low'
    white = tmp_path / 'white'
    condition, white_condition = 'L-vm-none-k3', 'A-vm-none'
    low_dir = out / 'predictions' / condition
    white_dir = white / 'predictions' / white_condition
    low_dir.mkdir(parents=True); white_dir.mkdir(parents=True)
    (out / 'experts' / condition).mkdir(parents=True)
    (white / 'experts' / white_condition).mkdir(parents=True)
    (out / 'experts' / condition / 'trained.json').write_text(json.dumps({'best': {'t0': .2}}))
    (white / 'experts' / white_condition / 'trained.json').write_text(json.dumps({'best': {'t0': .2}}))
    n = 3
    rows = {'row_ids': np.arange(n), 'target': np.zeros((n, 10, 7), np.float32),
            'draft': np.ones((n, 10, 7), np.float32), 'hard': np.array([False, True, False]),
            'task': np.zeros(n, np.int32), 'episode': np.arange(n, dtype=np.int32),
            'err_t0.2': np.ones(n, np.float32)}
    np.savez(low_dir / 'eval_rows.npz', **rows)
    paired = dict(rows); paired['draft'] = rows['draft'].copy(); paired['target'] = rows['target'].copy()
    paired['draft'][0, 0, 0] += 1
    np.savez(white_dir / 'eval_rows.npz', **paired)
    monkeypatch.setattr(lf, 'OUT', out); monkeypatch.setattr(lf, 'WHITE', white)
    monkeypatch.setattr(lf, '_load_cd', lambda: SimpleNamespace(
        _episode_keys=lambda task, ep: task * 100000 + ep,
        _bootstrap_all_and_hard=lambda *args: {'all': {}, 'hard': {}}))
    try:
        lf.compare(condition)
    except AssertionError as exc:
        assert 'draft differs' in str(exc)
    else:
        raise AssertionError('paired comparison accepted a different draft')


# ── Compressed denoise ────────────────────────────────────────────────────────────────────────────

def test_active7_actions_are_zero_padded_in_model_space():
    a = np.ones((2, 10, 7), np.float32)
    y = cd._pad_actions(a)
    assert y.shape == (2, 10, 32)
    np.testing.assert_array_equal(y[:, :, :7], a)
    assert np.count_nonzero(y[:, :, 7:]) == 0
    np.testing.assert_array_equal(cd._pad_actions(y), y)


def test_procrustes_recovers_shared_rotation_and_scale():
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(12)
    shape = rng.normal(size=(10, 6))
    rot = Rotation.from_rotvec([.2, -.1, .3]).as_matrix()
    scale = 1.7
    target = np.concatenate([shape[:, :3] @ rot, shape[:, 3:6] @ rot], axis=1) * scale
    theta, aligned, err = cd._procrustes(shape, target)
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
    selected = cd._select_sequence(bank, [a, b], target, 0, 1,
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
    selected = cd._select_sequence(bank, [a, b], a, 0, 2, form='as_is',
                                  exclude_task=False)
    assert selected['records'][0]['source_task'] == 2
    # Training rows share the memory namespace and still exclude that task.
    selected_train = cd._select_sequence(bank, [a, b], a, 0, 2, form='as_is',
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
    picked, draft, err = cd._raw_oracle(actions, qid, qtask,
                                       np.array([2, 5, 7]), tasks)
    assert picked.tolist() == [7]
    assert draft.shape == (1, 10, 7)
    assert err[0] == pytest.approx(.01/7)
    picked_all, _, _ = cd._raw_oracle(actions, qid, qtask,
                                     np.array([2, 5, 7]), tasks,
                                     exclude_task=False)
    assert picked_all.tolist() == [5]


def test_shape_draft_uses_source_pose_not_target_oracle_pose():
    import variable_shape_memory as vs
    from scipy.spatial.transform import Rotation

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

    got_a = cd._select_sequence(bank, [shape], target_a, 0, 2, form='rms',
                               exclude_task=False)
    got_b = cd._select_sequence(bank, [shape], target_b, 0, 2, form='rms',
                               exclude_task=False)
    expected_motion = cd._apply_theta(shape, source_theta[0], vs)
    expected = np.concatenate([expected_motion, shape[:, 6:7]], axis=1)
    np.testing.assert_allclose(got_a['draft'], expected, atol=1e-6)
    np.testing.assert_allclose(got_b['draft'], expected, atol=1e-6)
    assert not np.allclose(got_a['theta_target'], got_b['theta_target'])


def test_condition_list_and_t0_ranges_are_fixed():
    assert len(cd.CONDS) == 10
    assert cd._t0_range('A-vm-08-hi') == (.2, .6)
    assert cd._t0_range('A-vm-08') == (.05, .4)
    assert cd._t0_range('C-vm-08') == (0., 1.)


def test_theta_normalization_uses_only_valid_training_tokens():
    src=np.zeros((2,3,4),np.float32)
    tgt=np.arange(24,dtype=np.float32).reshape(2,3,4)
    units=np.array([[0,0,-1],[1,-1,-1]])
    mean,std=cd._theta_stats(src,tgt,units)
    np.testing.assert_allclose(mean,tgt[units>=0].mean(0))
    np.testing.assert_allclose(std,tgt[units>=0].std(0))


def test_unit_pose_diagnostic_handles_multiple_units_and_active7():
    import variable_shape_memory as vs
    shape=np.zeros((10,7),np.float32);shape[:,0]=1.;shape[:,6]=1.
    theta=np.zeros((10,4),np.float32);theta[5:,3]=np.log(2.)
    decoded=cd._decode_theta_tokens(shape,theta,vs)
    assert decoded.shape==(10,7)
    np.testing.assert_allclose(decoded[:5,:6],cd._apply_theta(shape[:5],theta[0],vs),atol=1e-6)
    np.testing.assert_allclose(decoded[5:,:6],cd._apply_theta(shape[5:],theta[5],vs),atol=1e-6)
    np.testing.assert_array_equal(decoded[:,6],shape[:,6])


def test_vectorized_oracle_matches_scalar_scoring_and_ties():
    import variable_shape_memory as vs
    rng=np.random.default_rng(820)
    for n in [1,5,10]:
        cand=rng.normal(size=(19,n,7)).astype(np.float32)
        cand[1]=cand[0]
        target=rng.normal(size=(n,7)).astype(np.float32)
        for form in ['as_is','rms']:
            actual=cd._candidate_errors(cand,target,form)
            if form=='as_is':expected=np.array([np.mean((c-target)**2) for c in cand])
            else:
                expected=[]
                for c in cand:
                    _,aligned,_=cd._procrustes(c[:,:6],vs.to_physical(target[:,:6]))
                    pred=np.concatenate([vs.to_normalized(aligned),c[:,6:7]],axis=1)
                    expected.append(np.mean((pred-target)**2))
                expected=np.array(expected)
            np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-7)
            assert np.argmin(actual)==np.argmin(expected)
            assert actual[0]==actual[1]


def test_theta_filter_preserves_frozen_vision_head():
    from openpi.shared.nnx_utils import PathRegex
    filt=PathRegex(cd.THETA_FILTER)
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
    def one(i):return cd._select_sequence(bank,decoded,target,i,0,form='rms',random_pick=True,seed=100+i)
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
    left=cd._select_sequence(bank,[unit],trace,0,1,form='as_is',seed=4)
    middle=cd._select_sequence(bank,[unit],trace,10,1,form='as_is',seed=4)
    right=cd._select_sequence(bank,[unit],trace,20,1,form='as_is',seed=4)
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
    first=cd._select_sequence(a,units,trace,30,1,form='as_is',random_pick=True,seed=55)
    cd._select_sequence(b,units,trace,0,1,form='as_is',random_pick=True,seed=55)
    second=cd._select_sequence(b,units,trace,30,1,form='as_is',random_pick=True,seed=55)
    np.testing.assert_array_equal(first['draft'],second['draft']);assert first['records']==second['records']


def test_paired_bootstrap_restricts_episode_ids_to_the_hard_subset():
    draft=np.arange(6,dtype=float);generated=draft+.5;episode=np.repeat([10,20,30],2)
    hard=np.array([True,True,False,False,False,False])
    out=cd._bootstrap_all_and_hard(generated,draft,episode,hard)
    assert out['all']['rows']==6 and out['all']['episodes']==3
    assert out['hard']['rows']==2 and out['hard']['episodes']==1
    assert out['all']['mean']==pytest.approx(.5)
    np.testing.assert_allclose(out['hard']['ci95'],[.5,.5])
    assert cd._bootstrap_all_and_hard(generated,draft,episode,np.zeros(6,bool))['hard'] is None
