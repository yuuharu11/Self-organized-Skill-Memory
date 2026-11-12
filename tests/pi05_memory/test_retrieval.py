"""Memory core and retrieval keys."""
import importlib.util
import json

import numpy as np
import pytest
import torch

from skill_moe.skillnet.examples.memory_pilot.pi05_memory.score_audit.retrieval_4b import (
    _rank_matrix,
    stable_topk,
    standardized,
)
from tests.support import (
    PI05,
    add_path,
    load_script,
)

CORE_SCRIPT = PI05 / "core.py"
core = load_script(CORE_SCRIPT)
add_path(PI05 / "key_mechanism")
key_mechanism = load_script(PI05 / "key_mechanism/run.py")
local_key = load_script(PI05 / "local_key/run.py")
topk_long = load_script(PI05 / "topk_long/run.py")


# ── Dense bridge and non-averaged token retrieval — Checks for the new dense bridge and non-averaged token retrieval contract. ───

def test_bridge_endpoints_and_velocity_reach_target():
    rng = np.random.default_rng(12)
    target, draft, noise = rng.normal(size=(3,2,10,7))
    t0 = np.array([.2,.6])
    x, v, t = core.bridge_batch(target,draft,noise,t0,np.ones(2))
    np.testing.assert_allclose(x-t[:,None,None]*v,target,atol=1e-12)
    x0, _, times = core.bridge_batch(target,draft,noise,t0,np.zeros(2))
    np.testing.assert_allclose(x0,target)
    np.testing.assert_equal(times,0)


def test_ordinary_flow_is_special_case():
    a=np.ones((1,10,7));eps=np.full_like(a,2.)
    x,v,t=core.bridge_batch(a,-a,eps,[1.],[.5])
    np.testing.assert_allclose(x,1.5)
    np.testing.assert_allclose(v,eps-a)


def test_maxsim_keeps_distinct_input_modes_and_ignores_padding():
    q=np.array([[1.,0.],[0.,1.],[99.,99.]])
    bank=np.array([[[0.,1.],[1.,0.],[-99.,99.]],[[1.,0.],[1.,0.],[99.,99.]]])
    group=np.array([0,0,-1])
    scores=core.late_interaction(q,bank,group,np.stack([group,group]))
    np.testing.assert_allclose(scores,[1.,.5])


def test_modalities_cannot_match_each_other():
    q=np.eye(2);bank=np.array([q,q[::-1]])
    groups=np.array([0,1])
    np.testing.assert_allclose(core.late_interaction(q,bank,groups,np.stack([groups,groups])),[1.,0.])


def test_self_episode_is_excluded_and_empty_bank_rejected():
    np.testing.assert_equal(core.exclude_episode([1.,.8,.2],[5,6,5],5),[-np.inf,.8,-np.inf])
    with pytest.raises(ValueError):
        core.exclude_episode([1.],[5],5)


def test_calibration_can_choose_all_fallback_without_forcing_memory_use():
    path = CORE_SCRIPT.with_name('calibrate.py')
    spec = importlib.util.spec_from_file_location('pi05_memory_calibrate',path)
    calibration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(calibration)
    scores = np.linspace(.2,.99,100)
    errors = np.tile([.1,.2,.3,.4],(100,1))
    selected = calibration.choose_schedule(scores,errors)
    np.testing.assert_equal(calibration.apply_schedule(scores,selected),0)


def test_calibration_uses_monotone_noise_and_applies_frozen_edges():
    path = CORE_SCRIPT.with_name('calibrate.py')
    spec = importlib.util.spec_from_file_location('pi05_memory_calibrate',path)
    calibration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(calibration)
    scores = np.linspace(0,1,100)
    errors = np.ones((100,4))
    errors[:40,0]=0
    errors[40:80,2]=0
    errors[80:,3]=0
    selected = calibration.choose_schedule(scores,errors)
    modes = calibration.apply_schedule(scores,selected)
    assert np.all(np.diff(modes)>=0)
    assert selected['dev_mse'] < .05
    np.testing.assert_equal(calibration.apply_schedule(np.array([-10,10]),selected),[0,3])


def test_paired_success_summary_and_initial_state_integrity(monkeypatch):
    monkeypatch.syspath_prepend(str(CORE_SCRIPT.parent))
    spec=importlib.util.spec_from_file_location('pi05_memory_evaluate',CORE_SCRIPT.with_name('evaluate.py'))
    evaluation=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    rows=[]
    for initial,(a,b) in enumerate([(0,1),(0,1),(1,0),(1,1)]):
        for condition,success in [('normal',a),('memory',b)]:
            rows.append({'task_suite':'libero_10','task_id':0,'trial_index':initial,
                         'condition':condition,'success':success,'status':'complete',
                         'initial_state_sha256':str(initial)})
    summary=evaluation.summarize(rows)['libero_10']
    assert summary['paired']==4
    assert summary['rescues']==2 and summary['harms']==1
    assert summary['paired_difference']==.25
    assert summary['conditions']['memory']['success_rate']==.75
    rows[-1]['initial_state_sha256']='wrong'
    with pytest.raises(ValueError,match='initial states'):
        evaluation.summarize(rows)


def test_calibration_episode_split_is_disjoint_and_window_count_independent(monkeypatch):
    monkeypatch.syspath_prepend(str(CORE_SCRIPT.parent))
    from run import calibration_episodes
    rows={'task':np.repeat([0,1],5),'episode':np.arange(10),'validation':np.ones(10,bool)}
    dev,evaluation=calibration_episodes(rows)
    assert len(dev)==6 and len(evaluation)==4
    assert not set(dev)&set(evaluation)
    repeated={k:np.repeat(v,np.arange(1,11)) for k,v in rows.items()}
    a,b=calibration_episodes(repeated)
    np.testing.assert_array_equal(dev,a)
    np.testing.assert_array_equal(evaluation,b)


# ── Key mechanism ─────────────────────────────────────────────────────────────────────────────────

def _fake_rows():
    # Deliberately choose dataset row IDs with a different order from pool positions.
    actions = np.zeros((9, 10, 7), np.float32)
    for i in range(9):
        actions[i] = i / 10
    task = np.array([0, 10, 10, 11, 11, 12, 12, 13, 13], np.int32)
    return {"actions": actions, "task": task}


def test_retrieve_scores_actions_by_candidate_row_id_and_excludes_same_task():
    rows = _fake_rows()
    qids = np.array([0, 5])
    mids = np.array([8, 2, 6, 1, 7, 3])
    qfeat = np.array([[1., 0.], [0., 1.]], np.float32)
    mfeat = np.array([[0., 1.], [1., 0.], [0., .9], [.8, .2], [0., .8], [.7, .3]], np.float32)
    cand, score, mse = key_mechanism._retrieve(qfeat, mfeat, qids, mids, rows, rows["actions"], k=3)
    assert cand[0, 0] == 2  # highest score but same task? query task 0, so allowed
    assert 6 not in cand[1]  # query task 12 excluded
    expected = np.mean((rows["actions"][qids][:, None, :, :7] - rows["actions"][cand][..., :7]) ** 2, axis=(-1, -2))
    np.testing.assert_allclose(mse, expected, rtol=0, atol=1e-7)
    # Explicitly catch the former pool-position / dataset-row-ID confusion.
    wrong = np.mean((rows["actions"][qids[0]][None, :, :7] - rows["actions"][np.arange(3)][..., :7]) ** 2, axis=(-1, -2))
    assert not np.allclose(mse[0], wrong)
    assert np.all(np.diff(score[0]) <= 0)


def test_holdout_tasks_are_suite_balanced_and_long_is_rejected():
    hold = key_mechanism.choose_holdout_tasks()
    assert hold.tolist() == [17, 19, 28, 29, 35, 38]
    assert sum(10 <= x < 20 for x in hold) == 2
    assert sum(20 <= x < 30 for x in hold) == 2
    assert sum(30 <= x < 40 for x in hold) == 2
    rows = {"task": np.array([9, 10, 39], np.int32)}
    with pytest.raises(AssertionError):
        key_mechanism.assert_no_long(np.array([0]), rows, "bad")
    key_mechanism.assert_no_long(np.array([1, 2]), rows, "good")


def test_attention_models_shapes_and_masked_attention():
    torch.set_num_threads(1)
    from models import M1AttentionKey, M2SalientTokens, M3CandidateJudge
    tokens = torch.randn(2, 12, 2048)
    context = torch.randn(2, 79)
    mask = torch.ones(2, 12, dtype=torch.bool)
    mask[:, -2:] = False
    m1 = M1AttentionKey(slots=3, dim=24, token_count=12, heads=4)
    key, _, att = m1.encode_tokens(tokens, context, mask)
    assert key.shape == (2, 24)
    assert att.shape == (2, 3, 12)
    assert torch.all(att[:, :, -2:] == 0)
    m2 = M2SalientTokens(dim=24, token_count=12)
    key2, _, weights = m2.encode_tokens(tokens, context, mask)
    assert key2.shape == (2, 24)
    assert weights.shape == (2, 12)
    assert torch.all(weights[:, -2:] == 0)
    judge = M3CandidateJudge(dim=24, heads=4)
    score = judge(torch.randn(2, 12, 24), mask, torch.randn(2, 5, 10, 7), torch.randn(2, 5, 8))
    assert score.shape == (2, 5)


def test_cpu_state_dict_snapshot_keeps_live_model_on_device():
    from models import M1AttentionKey
    model = M1AttentionKey(slots=3, dim=24, token_count=12, heads=4)
    model.to(key_mechanism.DEVICE)
    before = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
    snapshot = key_mechanism._cpu_state_dict(model)
    # The live model must stay where it was so post-save encoding still hits DEVICE.
    assert next(model.parameters()).device.type == torch.device(key_mechanism.DEVICE).type
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key].cpu(), value)
        assert snapshot[key].device.type == 'cpu'
        torch.testing.assert_close(snapshot[key], value.cpu())


def test_gap_threshold_reads_corrected_then_summary(tmp_path):
    (tmp_path / 'summary_corrected.json').write_text(json.dumps({'threshold': 0.02}))
    (tmp_path / 'summary.json').write_text(json.dumps({'threshold': 0.05}))
    assert key_mechanism._gap_threshold(tmp_path) == 0.02
    (tmp_path / 'summary_corrected.json').unlink()
    assert key_mechanism._gap_threshold(tmp_path) == 0.05
    (tmp_path / 'summary.json').unlink()
    with pytest.raises(FileNotFoundError):
        key_mechanism._gap_threshold(tmp_path)


# ── Local key ─────────────────────────────────────────────────────────────────────────────────────

def test_project_and_patch_grid_mapping():
    t=np.eye(4,dtype=np.float32)
    assert np.allclose(local_key.project_world_to_pixel([128.,128.,1.],t),[111.125,111.125])
    ix=local_key.patch_ids_for_point([112.,112.],3)
    assert len(ix)==9 and ix.tolist()==[119,120,121,135,136,137,151,152,153]
    assert len(local_key.patch_ids_for_point([0.,0.],5))==9


def test_state_standardization_is_fit_on_training_only():
    train=np.asarray([[0.,2.],[2.,4.]],np.float32);other=np.asarray([[1.,3.],[100.,100.]],np.float32)
    tr,ot,spec=local_key._standardize(train,other)
    assert np.allclose(tr.mean(0),0.,atol=1e-6)
    assert np.allclose(spec['mean'],[1.,3.])
    assert ot[0].tolist()==[0.,0.] and ot[1,0]>90


def test_pairwise_and_candidate_action_mse():
    q=np.zeros((2,10,7),np.float32);m=np.ones((3,10,7),np.float32)
    assert np.allclose(local_key._row_mse(q,m),1.)
    c=np.stack([np.zeros((2,10,7)),np.ones((2,10,7))],axis=1).astype(np.float32)
    assert np.allclose(local_key._candidate_mse(q,c),[[0.,1.],[0.,1.]])
    m=local_key._metrics('test',np.zeros((2,64),np.int64),np.ones((2,64),np.float32),
                   np.ones((2,64),np.float32),None,np.asarray([0,1]),{'episode':np.asarray([3,4])})
    assert np.isnan(m['recall8']).all()


def test_recall8_counts_only_first_eight_ranked_candidates():
    cand=np.tile(np.arange(64,dtype=np.int64),(1,1))
    scores=np.arange(64,dtype=np.float32)[None,:]
    # Relevant items exist only at ranks 9..12. They belong in the denominator
    # but cannot count as retrieved by Recall@8.
    candidate_mse=np.ones((1,64),np.float32)
    candidate_mse[0,8:12]=0.
    all_mse=np.zeros((1,4),np.float32)
    got=local_key._metrics('recall',cand,scores,candidate_mse,all_mse,np.asarray([100]),{'episode':np.asarray([7])})
    assert got['query_positive']==[True]
    assert got['recall8']==[0.]


def test_retrieve_maps_pool_position_back_to_dataset_row_id():
    # Memory pool order is [row 4, row 1], deliberately different from dataset order.
    rows={'task':np.asarray([0,1,9,2,3]),'episode':np.asarray([0,1,9,2,3])}
    actions=np.zeros((5,10,7),np.float32)
    actions[4]=.25
    actions[1]=1.
    actions[0]=-.75  # the old bug accidentally compared against this dataset row
    qfeat=np.asarray([[1.,0.]],np.float32)
    mfeat=np.asarray([[1.,0.],[0.,1.]],np.float32)
    qids=np.asarray([2]);mids=np.asarray([4,1])
    cand,score,mse=local_key._retrieve(qfeat,mfeat,qids,mids,rows,actions,k=2,exclusion='task')
    assert cand.tolist()==[[4,1]]
    assert np.allclose(mse[0,0],local_key._row_mse(actions[[2]],actions[[4]])[0,0])
    assert not np.isclose(mse[0,0],local_key._row_mse(actions[[2]],actions[[0]])[0,0])


def test_key_blocks_are_independently_normalized_and_weighted():
    v=np.asarray([[3.,4.]],np.float32);s=np.asarray([[0.,2.]],np.float32)
    h=np.asarray([[1.,0.]],np.float32)
    out=local_key._join_key_blocks(v,s,h,(.5,.25,.25))
    assert out.shape==(1,6) and np.isclose(np.linalg.norm(out),1.)
    assert np.allclose(out[0,:2],np.asarray([.6,.8],np.float32)*np.sqrt(.5))
    assert np.allclose(out[0,2:4],np.asarray([0.,1.],np.float32)*.5)
    assert np.allclose(out[0,4:6],np.asarray([1.,0.],np.float32)*.5)


def test_episode_bootstrap_is_paired_and_reproducible():
    d=np.asarray([-1.,-2.,1.,2.]);ep=np.asarray([10,10,20,20])
    a=local_key._episode_bootstrap(d,ep,reps=100);b=local_key._episode_bootstrap(d,ep,reps=100)
    assert a==b and a['episodes']==2 and a['rows']==4


def test_long_leakage_assertion_uses_task_ids():
    rows={'task':np.asarray([10,11,0,1])}
    local_key._assert_no_long(np.asarray([0,1]),np.asarray([1]),rows)
    try:local_key._assert_no_long(np.asarray([0,2]),np.asarray([1]),rows)
    except AssertionError:pass
    else:raise AssertionError('Long row must be rejected from fitting')


def test_direct_mlp_small_cpu_fit_and_prediction():
    rng=np.random.default_rng(3);x=rng.normal(size=(24,6)).astype(np.float32);y=np.repeat(x[:,:1,None],70,axis=2).reshape(24,10,7)
    model,meta=local_key._fit_direct(x[:18],y[:18],x[18:],y[18:],threads=1)
    pred=local_key._predict_direct(model,x[18:],meta['mean'],meta['std'])
    assert pred.shape==(6,10,7) and np.isfinite(pred).all() and meta['epochs']>0


def test_contrastive_projection_uses_cross_task_action_positives():
    rng=np.random.default_rng(9);x=rng.normal(size=(32,12)).astype(np.float32)
    a=np.zeros((32,10,7),np.float32);a[16:]=.01
    task=np.repeat(np.arange(4),8);devx=rng.normal(size=(12,12)).astype(np.float32)
    deva=np.zeros((12,10,7),np.float32);devtask=np.repeat(np.arange(3),4)
    model,meta=local_key._fit_projection(x,a,task,devx,deva,devtask,threads=1)
    out=local_key._project_features(model,x,meta['feature_mean'],meta['feature_std'])
    assert out.shape==(32,128) and np.isfinite(out).all() and meta['tau'] in (.005,.01,.02)


# ── Retrieval 4b ranking ──────────────────────────────────────────────────────────────────────────

def test_stable_topk_uses_row_id_for_ties():
    ids, scores = stable_topk(np.asarray([.2, .5, .5, .1]), np.asarray([9, 8, 3, 1]), k=3)
    np.testing.assert_array_equal(ids, [3, 8, 9])
    np.testing.assert_allclose(scores, [.5, .5, .2], rtol=0, atol=1e-7)


def test_vectorized_fixed_keys_obey_task_and_episode_exclusions():
    x = np.asarray([[1, 0], [.9, .1], [0, 1], [-1, 0], [0, -1]], np.float32)
    rows = {'task': np.asarray([0, 0, 1, 2, 3]), 'episode': np.asarray([0, 1, 2, 3, 4])}
    qids = np.asarray([0]); mids = np.arange(5)
    task_cand, _ = _rank_matrix(x, qids, mids, rows, 'task', 'cos', batch_rows=1, k=3)
    ep_cand, _ = _rank_matrix(x, qids, mids, rows, 'episode', 'cos', batch_rows=1, k=3)
    assert 0 not in task_cand[0]
    assert 1 not in task_cand[0]
    assert 0 not in ep_cand[0]
    assert 1 in ep_cand[0]


def test_standardization_is_fitted_only_on_supplied_train_rows():
    x = np.asarray([[0., 1.], [2., 3.], [100., 100.]], np.float32)
    z, mean, std = standardized(x, np.asarray([0, 1]))
    np.testing.assert_allclose(mean, [1., 2.])
    np.testing.assert_allclose(std, [1., 1.])
    np.testing.assert_allclose(z[2], [99., 98.])


# ── Top-k on Long ─────────────────────────────────────────────────────────────────────────────────

def test_medoid_returns_consensus_candidate():
    x = np.asarray([[[[0.]], [[2.]], [[10.]]], [[[5.]], [[5.1]], [[20.]]]], np.float32)
    got = topk_long.select_medoid(x)
    assert got.tolist() == [1, 1]


def test_min_edit_and_near_normal():
    outputs = np.asarray([[[[0.]], [[2.]], [[4.]]], [[[9.]], [[8.]], [[7.]]]], np.float32)
    drafts = np.asarray([[[[1.]], [[2.2]], [[3.]]], [[[9.1]], [[8.9]], [[6.]]]], np.float32)
    assert topk_long.select_min_edit(outputs, drafts).tolist() == [1, 0]
    ref = np.asarray([[[3.8]], [[7.1]]], np.float32)
    assert topk_long.select_near(outputs, ref).tolist() == [2, 2]


def test_episode_bootstrap_is_reproducible_and_paired():
    a = np.asarray([1., 2., 4., 8.])
    b = np.asarray([2., 3., 3., 9.])
    ep = np.asarray(['a', 'a', 'b', 'c'])
    mask = np.asarray([True, True, True, False])
    ci1 = topk_long.episode_bootstrap_ci(a, b, ep, mask, reps=200, seed=19)
    ci2 = topk_long.episode_bootstrap_ci(a, b, ep, mask, reps=200, seed=19)
    assert ci1 == ci2
    assert ci1[0] <= -0.5 <= ci1[1]


def test_candidate_validation_fails_closed_for_nonmemory_and_duplicates():
    rows = {
        'memory': np.asarray([True, True, False]),
        'task': np.asarray([0, 1, 0]),
        'episode': np.asarray([4, 4, 8]),
    }
    topk_long.validate_candidate_rows(rows, np.asarray([[1]]), np.asarray([0]), np.asarray([4]), 'task')
    try:
        topk_long.validate_candidate_rows(rows, np.asarray([[2]]), np.asarray([0]), np.asarray([4]), 'task')
    except AssertionError as exc:
        assert 'fixed memory' in str(exc)
    else:
        raise AssertionError('nonmemory candidate was accepted')
    try:
        topk_long.validate_candidate_rows(rows, np.asarray([[1, 1]]), np.asarray([0]), np.asarray([4]), 'task')
    except AssertionError as exc:
        assert 'duplicate' in str(exc)
    else:
        raise AssertionError('duplicate candidate was accepted')


def test_episode_exclusion_is_task_namespaced():
    rows = {
        'memory': np.asarray([True, True]),
        'task': np.asarray([0, 1]),
        'episode': np.asarray([7, 7]),
    }
    # Episode numbers can repeat in another task; only (task, episode) is identity.
    topk_long.validate_candidate_rows(rows, np.asarray([[1]]), np.asarray([0]), np.asarray([7]), 'episode')
    try:
        topk_long.validate_candidate_rows(rows, np.asarray([[0]]), np.asarray([0]), np.asarray([7]), 'episode')
    except AssertionError as exc:
        assert 'same-episode' in str(exc)
    else:
        raise AssertionError('same episode candidate was accepted')


def test_long_mapping_is_verified_from_task_text_and_data_filenames():
    if not topk_long.RUN.exists():
        pytest.skip('requires the local run-01 artifact')
    rows, _, _, long_rows = topk_long.load_inputs()
    mapping = topk_long.verify_long_task_mapping(rows)
    assert set(mapping) == {str(i) for i in range(10)}
    assert len(long_rows) > 0
    assert 'turn_on_the_stove' in mapping['3']['demo_file']
