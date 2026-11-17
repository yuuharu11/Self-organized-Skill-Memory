import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'skill_moe/skillnet/examples/memory_pilot/unseen_libero'))
import key_kv_cache as kv1


def test_compacts_valid_prefix_tokens_and_keeps_rope_kv():
    rng=np.random.default_rng(4)
    k=rng.normal(size=(18,2,968,1,256)).astype(np.float32)
    v=rng.normal(size=k.shape).astype(np.float32)
    mask=np.zeros((2,968),bool);mask[:,:512]=True;mask[:,768:788]=True
    packed,valid=kv1.pack_prefix_kv(k,v,mask)
    assert packed.shape==(2,18,2,532,256)
    assert valid.shape==(2,532) and valid.all()
    assert np.array_equal(packed[:, :, 0],np.transpose(k[:,:,np.r_[0:512,768:788],0,:],(1,0,2,3)).astype(np.float16))
    assert np.array_equal(packed[:, :, 1],np.transpose(v[:,:,np.r_[0:512,768:788],0,:],(1,0,2,3)).astype(np.float16))


def test_final_tokens_compact_and_group_padding():
    h=np.ones((1,968,2048),np.float32);mask=np.zeros((1,968),bool)
    mask[:,:512]=True;mask[:,768:780]=True
    tokens,groups=kv1.final_tokens_from_prefix(h,mask)
    assert tokens.shape==(1,532,2048) and groups.shape==(1,532)
    assert np.all(groups[0,:256]==0) and np.all(groups[0,256:512]==1)
    assert np.all(groups[0,512:524]==2) and np.all(groups[0,524:]==-1)
    assert np.allclose(np.linalg.norm(tokens[0,:524],axis=-1),1.,atol=1e-3)


def test_exact_join_uses_identity_and_fails_closed():
    q=np.asarray([[9,1,80],[2,3,10]],np.int32);c=np.asarray([[2,3,10],[9,1,80],[1,0,0]],np.int32)
    values=np.arange(6).reshape(3,2)
    assert np.array_equal(kv1.exact_join_rows(q,c,values),values[[1,0]])
    with pytest.raises(KeyError):kv1.exact_join_rows([[3,4,5]],c,values)
    with pytest.raises(ValueError):kv1.exact_join_rows(q,[[1,1,1],[1,1,1]],values[:2])


def test_l90x_manifest_excludes_fixed_dev_eval_tasks():
    tasks=[i for i in range(90) if i not in set(kv1.DEV_TASKS+kv1.EVAL_TASKS)]
    assert len(tasks)==78
    kv1.validate_l90x_manifest({'tasks':tasks,'episodes':[]})
    with pytest.raises(AssertionError):kv1.validate_l90x_manifest({'tasks':tasks+[6],'episodes':[]})


def test_relative_precision_parity_gate_matches_spec():
    ref=np.asarray([[10.,-20.],[2.,4.]],np.float32)
    candidate=ref*np.asarray([[1.001,0.999],[1.002,.998]],np.float32)
    metrics=kv1.prefix_parity_metrics(ref,candidate)
    assert metrics['relative_median']<.01
    assert metrics['token_cosine_median']>=.999
    assert metrics['mean_pooled_cosine']>=.999
    assert kv1.parity_passes(metrics)
    bad=ref*1.1
    assert not kv1.parity_passes(kv1.prefix_parity_metrics(ref,bad))


def test_training_candidates_remove_entire_learned_episodes():
    task=np.asarray([0,0,0,1,1,2]);episode=np.asarray([4,5,6,4,5,4])
    ids=kv1.training_bank_indices(task,episode,{(0,5),(1,4)})
    assert np.array_equal(ids,np.asarray([0,2,4,5]))
    assert not ({(int(task[i]),int(episode[i])) for i in ids}&{(0,5),(1,4)})


def test_source_grouped_batches_preserves_source_batch_and_target_positions():
    source=np.asarray([[0,0,i] for i in range(10)],np.int32)
    targets=np.asarray([[0,0,0,5],[0,0,0,4],[0,0,0,9]],np.int32)
    batches=kv1.source_grouped_batches(targets,{'run':source},source_priority={0:('run',)},batch_size=4)
    by_group={int(x['source_indices'][0]//4):x for x in batches}
    b=by_group[1]
    assert np.array_equal(b['source_indices'],[4,5,6,7])
    pairs=dict(zip(b['target_indices'].tolist(),b['target_positions'].tolist()))
    assert pairs=={0:1,1:0}
    tail=by_group[2]
    assert np.array_equal(tail['source_indices'],[8,9])


@pytest.mark.parametrize('mode,expected', [('kv_l',5),('kv_h',11),('kv_lh',16)])
def test_reader_modes_have_symmetric_query_key_shapes(mode,expected):
    model=kv1.KVReader(mode,layer_count=3,context_dim=16,token_count=5)
    b,t=2,12
    x=torch.randn(b,3,2,t,256);valid=torch.ones(b,t,dtype=torch.bool)
    hist=torch.randn(b,10,7);hmask=torch.ones(b,10,dtype=torch.bool);state=torch.randn(b,8)
    z=model(x,valid,hist,hmask,state)
    assert z.shape==(b,expected,16) and torch.isfinite(z).all()
    assert model(x,valid,hist,hmask,state).shape[1]==expected


def test_layerwise_reader_and_layer_subset():
    model=kv1.KVReader('kv_l',layer_count=3,context_dim=16,selected_layers=(0,2),layerwise=True)
    x=torch.randn(1,3,2,8,256);valid=torch.ones(1,8,dtype=torch.bool)
    z=model(x,valid,torch.zeros(1,10,7),torch.zeros(1,10,dtype=torch.bool),torch.zeros(1,8))
    assert z.shape==(1,5,16) and torch.isfinite(z).all()
