import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path('/work/.skillnet-state/worktrees/latent-probe')
MODULE = ROOT / 'skill_moe/skillnet/examples/memory_pilot/unseen_libero/latent_probe_gate.py'
spec = importlib.util.spec_from_file_location('latent_probe_gate', MODULE)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def test_scalar_oracle_cannot_be_worse_than_fixed_source_scale():
    rng = np.random.default_rng(13)
    canonical = rng.normal(size=(12, 10, 6))
    source_scale = rng.uniform(.1, 3., size=12)
    source = gate.scale_without_rotation(canonical, source_scale)
    target = rng.normal(size=(12, 10, 6))
    oracle = gate.fit_nonnegative_scale(canonical, target)
    source_error = np.sum((source-target)**2, axis=(1, 2))
    oracle_error = np.sum((gate.scale_without_rotation(canonical, oracle)-target)**2, axis=(1, 2))
    assert np.all(oracle_error <= source_error + 1e-9)


def test_no_rotation_preserves_direction_and_cross_products():
    rng = np.random.default_rng(22)
    x = rng.normal(size=(4, 10, 6))
    y = gate.scale_without_rotation(x, np.array([.25, .5, 1., 2.]))
    assert np.allclose(y[:, :, :3], x[:, :, :3] * np.array([.25,.5,1.,2.])[:,None,None])
    # A scalar size operation preserves every normalized vector and axis relation;
    # a hidden rotation would violate the component-wise proportionality below.
    for i in range(len(x)):
        nz = np.abs(x[i]) > 1e-8
        ratios = y[i][nz] / x[i][nz]
        assert np.allclose(ratios, ratios[0])


def test_hold_last_and_pi05_orientation_reference_cache():
    paths = [gate.PRED/'pi05_eval_rot180.npz', gate.PRED/'pi05_eval_asis.npz',
             gate.SCALE/'prep/W_inputs_eval.npz', gate.ROWS/'actions.npy', gate.EVAL_ROWS]
    if not all(p.exists() for p in paths):
        pytest.skip('production LIBERO cache not mounted; cache-backed reference check skipped')
    result = gate.audit_orientation_and_hold_last(assert_reference=True)
    assert .105 <= result['pi05_rot180_mse6_over7'] <= .125
    assert .175 <= result['pi05_asis_mse6_over7'] <= .210
    assert .100 <= result['hold_last_eval_mse6_over7'] <= .135
    assert .175 <= result['hold_last_hard_mse6_over7'] <= .230
    assert result['corrected_action_cache_audit']['actions_equal'] is True


def test_pi05_rot180_is_not_silently_interchanged_with_asis():
    correct = np.zeros((2,10,7), np.float32)
    target = correct.copy(); target[...,0] = 1.
    rot180 = target.copy()
    asis = correct.copy()
    assert np.mean(gate.mse6_over7(rot180,target)) < np.mean(gate.mse6_over7(asis,target))


def test_switch_metrics_report_true_switch_subset():
    target = np.zeros((2,10,7), np.float32); target[:,:,6] = -1.
    target[0,5:,6] = 1.
    pred = target.copy()
    m = gate.gripper_metrics(pred,target)
    assert m['sign_agreement'] == 1.
    assert m['switch_rows'] == 1
    assert m['switch_precision_on_true_switch_rows'] == 1.
    assert m['switch_recall_on_true_switch_rows'] == 1.
    assert m['matched_switch_time_abs_error_steps'] == 0.


def test_episode_bootstrap_is_paired_and_deterministic():
    a = np.array([2.,3.,5.,7.]); b=np.array([1.,1.,1.,1.]); ep=np.array([1,1,2,2])
    x=gate.episode_bootstrap_diff(a,b,ep,reps=500,seed=5)
    y=gate.episode_bootstrap_diff(a,b,ep,reps=500,seed=5)
    assert x==y and x['mean']==3.25 and x['episodes']==2


def test_frozen_p5_and_r5_selectors_return_pool_ids_without_same_task():
    required=[gate.SCALE/'prep/W_pool.npz',gate.SCALE/'prep/W_inputs_dev.npz',
              gate.SCALE/'selection.json',gate.R5_CHECKPOINT,gate.ROWS/'tokens.npy']
    if not all(p.exists() for p in required):
        pytest.skip('selector artifacts are not mounted; real-selector CPU smoke skipped')
    pool=gate._load_combined_pool()
    src=dict(np.load(gate.SCALE/'prep/W_inputs_dev.npz'))
    inp={k:np.asarray(v[:3]) for k,v in src.items() if hasattr(v,'shape') and len(v.shape)>0 and len(v)==len(src['row'])}
    allowed=gate.candidate_mask('dev',inp,pool)
    p5,_=gate._load_p5_picks('dev',pool,inp,allowed)
    r5=gate._load_r5_picks('dev',pool,inp,allowed)
    assert p5.shape==r5.shape==(3,)
    assert np.all(allowed[np.arange(3),p5])
    assert np.all(allowed[np.arange(3),r5])
    assert np.all((0<=p5)&(p5<len(pool['task'])))
    assert np.all((0<=r5)&(r5<len(pool['task'])))
    # Exercise the actual amplitude path on only three Dev rows; this is a CPU
    # smoke, not the full Dev/Eval gate.
    targets={k:np.load(gate.ROWS/f'{k}.npy',mmap_mode='r') for k in ('task','episode','frame','actions')}
    row_idx=gate._align_unseen_rows(inp,targets)
    y_norm=np.asarray(targets['actions'][row_idx,:,:7],np.float32)
    pi_z=np.load(gate.PRED/'pi05_dev_rot180.npz');pi=np.asarray(gate._pred_actions(pi_z)[:3],np.float32)
    pi_phys=gate.vs.to_physical(pi[:,:,:6])
    types=gate.w_scale_only_types(pool)[r5]
    src=np.asarray(pool['phys'][r5],np.float32)
    amps=gate.amplitude_predictions(types,src,pi_phys,inp['hist'],inp['has_hist'])
    oracle_scale=gate.fit_oracle_scale(types,y_norm[:,:,:6])
    oracle=gate.scale_without_rotation(types,oracle_scale)
    source_full=np.zeros_like(y_norm);source_full[:,:,:6]=gate.vs.to_normalized(src)
    oracle_full=np.zeros_like(y_norm);oracle_full[:,:,:6]=gate.vs.to_normalized(oracle)
    source_error=gate.mse6_over7(source_full,y_norm)
    oracle_error=gate.mse6_over7(oracle_full,y_norm)
    assert np.all(np.isfinite(amps['S-pi05'])) and np.all(np.isfinite(amps['S-prev']))
    assert np.all(oracle_error<=source_error+1e-7)


def test_dev_and_eval_task_id_namespaces_do_not_create_false_exclusion():
    # A numeric task ID collision across LIBERO-90 and LIBERO-40 is not a
    # semantic same-task match. The Dev rule is explicitly cross_task=False.
    fake_inp={'task':np.asarray([10]),'row':np.asarray([123])}
    fake_pool={'shape':np.zeros((2,10,7),np.float32),
               'task':np.asarray([10,4]),'keep':np.asarray([True,True])}
    got=gate.candidate_mask('dev',fake_inp,fake_pool)
    assert got.tolist()==[[True,True]]


def test_corrected_wrist_tokens_are_joined_by_task_episode_frame():
    corrected=gate.ROWS/'tokens.npy'; groups=gate.ROWS/'groups.npy'
    required=[corrected,groups,gate.ROWS/'task.npy',gate.ROWS/'episode.npy',gate.ROWS/'frame.npy',
              gate.SCALE/'prep/W_inputs_dev.npz']
    if not all(p.exists() for p in required):
        pytest.skip('corrected row/token cache unavailable; source assertion skipped')
    meta={k:np.load(gate.ROWS/f'{k}.npy',mmap_mode='r') for k in ('task','episode','frame','tokens','groups')}
    src=dict(np.load(gate.SCALE/'prep/W_inputs_dev.npz'))
    inp={k:np.asarray(v[:2]) for k,v in src.items() if hasattr(v,'shape') and len(v.shape)>0 and len(v)==len(src['row'])}
    actual=gate._wrist_tokens_for_input(inp,corrected=True)
    ix=gate._align_unseen_rows(inp,meta)
    expected=np.stack([np.asarray(meta['tokens'][j][np.asarray(meta['groups'][j])==1],np.float32) for j in ix])
    assert actual.shape==(2,256,2048)
    assert np.array_equal(actual,expected)
    legacy=gate._wrist_tokens_for_input(inp,corrected=False)
    assert not np.array_equal(actual,legacy), 'fixture should expose the legacy 180-degree token cache'


def test_eval_mask_is_loaded_verbatim_with_row_order_assertion():
    p=gate.EVAL_ROWS
    if not p.exists(): pytest.skip('lib-closed Eval record unavailable')
    rec=np.load(p)
    inp={'row':np.asarray(rec['row_id']), 'task':np.asarray(rec['task'])}
    pool={'shape':np.zeros((rec['allowed'].shape[1],10,7),np.float32),
          'task':np.zeros(rec['allowed'].shape[1],np.int64),
          'keep':np.ones(rec['allowed'].shape[1],bool)}
    allowed=gate.candidate_mask('eval',inp,pool,eval_record=rec)
    assert np.array_equal(allowed,rec['allowed'])
    with pytest.raises(AssertionError):
        gate.candidate_mask('eval',{'row':inp['row'][::-1],'task':inp['task'][::-1]},pool,eval_record=rec)

