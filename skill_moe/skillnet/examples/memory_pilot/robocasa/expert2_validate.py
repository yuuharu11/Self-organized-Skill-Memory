"""Re-evaluate RM-EXP2 checkpoints with a live NNX model state.

The training loop's module_jit helper freezes state at construction time. This
script deliberately uses nnx.jit with the model as an argument so each saved
checkpoint is evaluated using its own updated weights.
"""
from __future__ import annotations

import json, pickle, sys
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
import expert, expert2_train as train

def validate(out: Path, threads=8, fraction=.45):
    import jax, jax.numpy as jnp, torch
    import flax.nnx as nnx
    from flax import serialization
    from openpi.shared.nnx_utils import PathRegex
    from openpi.shared import normalize
    torch.set_num_threads(threads); torch.cuda.set_per_process_memory_fraction(fraction)
    out=Path(out); data=np.load(train.PREP/'drafts/memory_dynamic.npz',allow_pickle=False)
    _,val_ids,_=train.prepare_full_cache(out,threads=threads)
    raw_draft=np.asarray(data['draft'],np.float32); raw_target=np.asarray(data['target'],np.float32)
    lengths=np.asarray(data['first_unit_length'],np.int32)
    stats=normalize.load(train.CHECKPOINT/'assets')['actions']
    draft=np.stack([expert._draft_to_model(x,stats) for x in raw_draft]).astype(np.float32)
    target=np.stack([expert._draft_to_model(x,stats) for x in raw_target]).astype(np.float32)
    cache={}; tree_def=None
    for task in sorted(np.unique(np.asarray(data['task'])[np.r_[np.flatnonzero(~np.asarray(data['validation'],bool)),val_ids]])):
        payload=pickle.load((out/'obs_cache'/f'task_{int(task)}.pkl').open('rb'))
        if tree_def is None: tree_def=payload['tree_def']
        if payload['tree_def']!=tree_def: raise AssertionError('cache tree mismatch')
        for pos,row in enumerate(payload['row_ids']): cache[int(row)]=[np.asarray(x[pos]) for x in payload['leaves']]
    def batch(ids):
        leaves=[jnp.asarray(np.stack([cache[int(i)][j] for i in ids])) for j in range(len(cache[int(ids[0])]))]
        return jax.tree.unflatten(tree_def,leaves)
    model,_=expert._load_expert_model(); filt=nnx.All(nnx.Param,PathRegex(r'.*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*'))
    @nnx.jit(static_argnames=('num_steps',))
    def evaluate(m,obs,dn,eps,starts,*,num_steps=10):
        return m.sample_bridge(jax.random.key(0),obs,dn,starts,num_steps=num_steps,noise=eps)
    ids=np.asarray(val_ids,np.int64); rows=[]; ep=np.asarray(data['episode']); task=np.asarray(data['task']); frame=np.asarray(data['frame'])
    for cp in sorted((out/'expert').glob('expert_*.msgpack')):
        selected=nnx.state(model,filt); loaded=serialization.from_bytes(selected.to_pure_dict(),cp.read_bytes()); selected.replace_by_pure_dict(loaded); nnx.update(model,selected)
        pred=[]
        for st in range(0,len(ids),4):
            bid=ids[st:st+4]; obs=batch(bid); dn=jnp.asarray(draft[bid]); eps=np.stack([np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(4101),int(i)),(1,50,32)))[0] for i in bid]); starts=jnp.full((len(bid),),.2,jnp.float32)
            pred.extend(np.asarray(evaluate(model,obs,dn,jnp.asarray(eps),starts,num_steps=10)))
        pred=np.asarray(pred); dif=(pred[:,:5,:6]-target[ids,:5,:6])**2
        unit=[]
        for p,i in zip(pred,ids,strict=True):
            l=int(np.clip(lengths[i],1,50)); unit.append(np.mean((p[:l,:6]-target[i,:l,:6])**2))
        rec={'checkpoint':cp.name,'rows':len(ids),'first5_mse6':float(dif.mean()),'unit_mse6':float(np.mean(unit))}
        rows.append(rec); print(json.dumps(rec),flush=True)
    best=min(rows,key=lambda r:r['first5_mse6'])
    (out/'expert'/'validation_recomputed.json').write_text(json.dumps({'method':'nnx.jit(model as dynamic argument); checkpoint weights loaded before evaluation','results':rows,'best':best},indent=2)+'\n')
    (out/'expert'/'best.json').write_text(json.dumps({'checkpoint':best['checkpoint'],'steps':int(best['checkpoint'].split('_')[1].split('.')[0]),'t0':.2,'val_first5_mse6':best['first5_mse6'],'val_unit_mse6':best['unit_mse6'],'selection':'fixed memory-side 300-row validation; recomputed with live checkpoint state'},indent=2)+'\n')
    return best

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--threads',type=int,default=8);p.add_argument('--fraction',type=float,default=.45);a=p.parse_args();validate(a.out,a.threads,a.fraction)
