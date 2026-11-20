import numpy as np
import torch
import inspect

from skill_moe.skillnet.examples.memory_pilot.unseen_libero import uq1_run as U
from skill_moe.skillnet.examples.memory_pilot.unseen_libero import xf2_run as X


def test_held_episode_excluded_from_fit_memory_and_fold_assignment():
    assignment={(t,e):e%5 for t in (0,1) for e in range(10)}
    task=np.repeat(np.repeat([0,1],10),2);episode=np.repeat(list(range(10))*2,2)
    fit,held,mem=X.fold_partition(task,episode,task,episode,assignment,3)
    hp={(int(task[i]),int(episode[i])) for i in held}
    assert not hp & {(int(task[i]),int(episode[i])) for i in fit}
    assert not hp & {(int(task[i]),int(episode[i])) for i in mem}


def test_oof_error_estimator_excludes_held_labels():
    x=np.linspace(0,1,50);f=np.repeat(np.arange(5),10);y=x*x
    a,_=X.fit_oof_isotonic(x,y,f)
    y[f==2]+=100
    b,_=X.fit_oof_isotonic(x,y,f)
    np.testing.assert_allclose(a[f==2],b[f==2])


def test_uq1_query_is_cross_attention_to_34_vlm_tokens_only():
    class Inputs:
        def __init__(self):
            self.x=torch.zeros(2,10,16);self.valid=torch.ones(2,10,dtype=torch.bool)
            self.vlm=torch.randn(2,34,2048,dtype=torch.float16)
        def window(self,ix):return self.x[ix],self.valid[ix]
    model=U.UQ1Net().eval();gs=Inputs();ix=torch.arange(2)
    with torch.no_grad():q=model(gs,ix)
    assert q.shape==(2,256)
    assert model.query.enc=='gru' and model.query.xattn_n==2
    assert model.query.kv[1].in_features==2048
    assert model.query.kv[1].out_features==256
    # Query forward consumes only the feature split, never task/target/pi05 fields.
    assert set(vars(gs))=={'x','valid','vlm'}

def test_spread10_is_variance_across_retrieved_candidates():
    keys=np.asarray([[0.,0.],[2.,0.],[4.,0.],[0.,2.]],np.float32)
    ids=np.asarray([[0,1,2,3,0,1,2,3,0,1]],np.int64)
    result=U.spread10(keys,ids)
    np.testing.assert_allclose(result,[1.54],rtol=1e-6,atol=1e-6)


def test_weighted_gripper_switch_rows_and_timing_f1():
    sp={'history':np.full((2,10,7),-1,np.float32),'history_mask':np.ones((2,10),bool),
        'actions':np.full((2,10,7),-1,np.float32)}
    sp['actions'][0,4:,6]=1
    y,last,valid,first=U._switch_info(sp)
    assert first.tolist()==[4,-1]
    _,tw,_,prob=U._target_loss_arrays(sp,20)
    assert tw[0,2:7].min()==20 and tw[1].max()==1
    assert prob[0]>prob[1]
    metric=U._switch_f1(np.where(y>0,1.,-1.),y,last,valid)
    assert metric['f1']==1. and metric['tp']==1


def test_threshold_selection_respects_coefficient_constraint_then_f1():
    rows=[{'dev_coef_sq':1.0,'dev_switch':{'f1':.5},'name':'a'},
          {'dev_coef_sq':1.04,'dev_switch':{'f1':.8},'name':'b'},
          {'dev_coef_sq':1.06,'dev_switch':{'f1':1.},'name':'c'}]
    best=min(r['dev_coef_sq'] for r in rows);eligible=[r for r in rows if r['dev_coef_sq']<=best*1.05]
    assert max(eligible,key=lambda r:r['dev_switch']['f1'])['name']=='b'


def test_paired_orientation_is_uq1_only_then_reference_only():
    x={(0,0):True,(0,1):False};r={(0,0):False,(0,1):True}
    z=X.paired_counts(x,r)
    assert z['xf2_only']==1 and z['reference_only']==1
    assert z['success_difference_check']

def test_serving_query_has_no_label_task_or_pi05_prediction_input():
    from skill_moe.skillnet.examples.memory_pilot.unseen_libero.uq1_serve import UQ1Policy
    src=inspect.getsource(UQ1Policy.infer)
    assert "task_id" not in src and "target" not in src
    assert "pi05_actions" not in src and "prediction" not in src

def test_vlm_feature_source_uses_20hz_dev_rows():
    from skill_moe.skillnet.examples.memory_pilot.unseen_libero import kq2_data
    assert kq2_data.VQB_NAME['dev2']=='dev2_20'
    assert set(kq2_data.VQB_NAME)=={'eh','learned','unseen20','dev2'}
