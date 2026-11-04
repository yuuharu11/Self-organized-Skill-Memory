"""Checks for the new dense bridge and non-averaged token retrieval contract."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_path = Path(__file__).resolve().parents[1] / 'skill_moe/skillnet/examples/memory_pilot/pi05_memory/core.py'
_spec = importlib.util.spec_from_file_location('pi05_memory_core', _path)
core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(core)


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
    path = _path.with_name('calibrate.py')
    spec = importlib.util.spec_from_file_location('pi05_memory_calibrate',path)
    calibration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(calibration)
    scores = np.linspace(.2,.99,100)
    errors = np.tile([.1,.2,.3,.4],(100,1))
    selected = calibration.choose_schedule(scores,errors)
    np.testing.assert_equal(calibration.apply_schedule(scores,selected),0)


def test_calibration_uses_monotone_noise_and_applies_frozen_edges():
    path = _path.with_name('calibrate.py')
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
    monkeypatch.syspath_prepend(str(_path.parent))
    spec=importlib.util.spec_from_file_location('pi05_memory_evaluate',_path.with_name('evaluate.py'))
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
    monkeypatch.syspath_prepend(str(_path.parent))
    from run import calibration_episodes
    rows={'task':np.repeat([0,1],5),'episode':np.arange(10),'validation':np.ones(10,bool)}
    dev,evaluation=calibration_episodes(rows)
    assert len(dev)==6 and len(evaluation)==4
    assert not set(dev)&set(evaluation)
    repeated={k:np.repeat(v,np.arange(1,11)) for k,v in rows.items()}
    a,b=calibration_episodes(repeated)
    np.testing.assert_array_equal(dev,a)
    np.testing.assert_array_equal(evaluation,b)
