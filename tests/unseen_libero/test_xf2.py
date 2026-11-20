import numpy as np
import pytest

from skill_moe.skillnet.examples.memory_pilot.unseen_libero import xf2_run as X


def test_episode_fold_excludes_heldout_from_fit_and_memory():
    assignment={(t,e):(e%5) for t in (0,1) for e in range(5)}
    task=np.repeat([0,0,0,0,0,1,1,1,1,1],2)
    episode=np.repeat(list(range(5))*2,2)
    fit,held,mem=X.fold_partition(task,episode,task,episode,assignment,2)
    held_pairs={(int(task[i]),int(episode[i])) for i in held}
    assert held_pairs and not held_pairs & {(int(task[i]),int(episode[i])) for i in fit}
    assert not held_pairs & {(int(task[i]),int(episode[i])) for i in mem}


def test_oof_isotonic_prediction_does_not_use_held_fold_labels():
    x=np.arange(20,dtype=float);folds=np.repeat(np.arange(4),5);y=x**2
    p,_=X.fit_oof_isotonic(x,y,folds)
    y2=y.copy();y2[folds==1]+=10000
    p2,_=X.fit_oof_isotonic(x,y2,folds)
    np.testing.assert_allclose(p[folds==1],p2[folds==1])


def test_t0_rule_and_fallback_threshold():
    np.testing.assert_allclose(X.variable_t0([0.,1.,4.],1),[.05,.5,.6])
    x=np.arange(100,dtype=float)
    out=X.calibration_threshold(x,.2)
    assert out['realized_rate'] <= .2
    assert out['rule'].startswith('fallback iff ehat > threshold')


def test_mcnemar_direction_is_xf2_only_then_reference_only():
    x={(0,0):True,(0,1):False,(0,2):True}
    r={(0,0):False,(0,1):True,(0,2):True}
    out=X.paired_counts(x,r)
    assert out['xf2_only']==1 and out['reference_only']==1
    assert out['success_difference_check']


def test_inference_retrieval_function_has_no_target_parameter():
    # Target actions are only consumed after ranking for the teacher/error arrays.
    import inspect
    args=inspect.signature(X._score_rows).parameters
    assert 'target' not in args
    assert 'pi05' not in args and 'task_id' not in args
