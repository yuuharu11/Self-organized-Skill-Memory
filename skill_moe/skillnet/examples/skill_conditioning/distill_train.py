"""Matched ordinary-flow fine tuning; C is a target source, never a student input."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2))


class MatchedData:
    def __init__(self,root):
        index=json.loads((root/'data/index.json').read_text())
        self.shards=index['shards']
        self.arrays=[np.load(row['path'],allow_pickle=False) for row in self.shards]
        self.normal=[i for i,row in enumerate(self.shards) if row['kind']=='normal']
        self.parents=sorted({r['parent'] for r in self.shards if r['kind']=='recovery'})
        self.recovery={p:[i for i,r in enumerate(self.shards) if r['kind']=='recovery' and r['parent']==p] for p in self.parents}
        self.normal_task={r['task']:[i for i in self.normal if self.shards[i]['task']==r['task']] for r in self.shards if r['kind']=='recovery'}
        # Materialize to prevent repeatedly inflating NPZ members per sample.
        self.arrays=[{k:a[k] for k in a.files} for a in self.arrays]

    def batch(self,update,condition):
        # Independent RNG per slot: replacing the last two slots cannot perturb
        # the common six normal samples or the next update's schedule.
        rows=[];identities=[]
        for slot in range(8):
            rng=np.random.default_rng(np.random.SeedSequence([7,update,slot]))
            if slot<6:
                counts=np.array([self.shards[i]['samples'] for i in self.normal])
                shard=int(rng.choice(self.normal,p=counts/counts.sum()))
            else:
                parent=self.parents[int(rng.integers(len(self.parents)))]
                teachers=self.recovery[parent]
                teacher=int(rng.choice(teachers))
                task=self.shards[teacher]['task']
                pool=[teacher] if condition=='M1' else self.normal_task[task]
                shard=int(rng.choice(pool))
            sample=int(rng.integers(self.shards[shard]['samples']))
            rows.append({key:values[sample] for key,values in self.arrays[shard].items()})
            identities.append(dict(shard=self.shards[shard]['id'],row=sample,task=self.shards[shard]['task']))
        keys=rows[0].keys()
        return {k:np.stack([r[k] for r in rows]) for k in keys},identities


def main(root):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import optax
    import orbax.checkpoint as ocp
    from smoke import load_model,parameter_digest
    from distill_loss import ordinary_flow_loss
    from openpi.models.pi0_config_anchor_flow import action_expert_trainable_filter
    from openpi.models import model as model_lib
    if jax.default_backend()!='gpu':raise RuntimeError('GPU required')
    manifest=json.loads((root/'manifest.json').read_text())
    dataset=MatchedData(root)
    model,config=load_model(manifest['base_checkpoint'])
    trainable=action_expert_trainable_filter()
    frozen=nnx.All(nnx.Param,nnx.Not(trainable))
    before_frozen=parameter_digest(model,frozen)
    before_trainable=parameter_digest(model,trainable)
    base_params=jax.tree.map(lambda v:np.asarray(v),nnx.state(model).to_pure_dict())
    tx=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(1e-6,b1=.9,b2=.95,weight_decay=1e-5))
    opt_state=tx.init(nnx.state(model,trainable))

    @nnx.jit
    def grad_step(rng,current,obs,actions):
        def loss_fn(m,key):
            flow,balance=ordinary_flow_loss(m,key,obs,actions,train=True)
            return jnp.mean(flow)+balance,(jnp.mean(flow),balance)
        return nnx.value_and_grad(loss_fn,argnums=nnx.DiffState(0,trainable),has_aux=True)(current,rng)

    @nnx.jit
    def apply(current,state,grads):
        params=nnx.state(current,trainable)
        updates,state=tx.update(grads,state,params)
        nnx.update(current,optax.apply_updates(params,updates))
        return state

    micro=manifest['microbatch']
    def update(condition,step,state):
        nonlocal micro
        batch,ids=dataset.batch(step,condition)
        total=None;losses=[]
        for lo in range(0,8,micro):
            b={k:v[lo:lo+micro] for k,v in batch.items()}
            images={name:jnp.asarray(b[key],jnp.float32)/255.*2.-1. for name,key in [('base_0_rgb','image_base'),('left_wrist_0_rgb','image_left_wrist')]}
            images['right_wrist_0_rgb']=jnp.full_like(images['base_0_rgb'],-1.)
            obs=model_lib.Observation(images=images,image_masks={name:jnp.full((micro,),name!='right_wrist_0_rgb') for name in images},state=jnp.asarray(b['state']),tokenized_prompt=jnp.asarray(b['tokenized_prompt'],jnp.int32),tokenized_prompt_mask=jnp.asarray(b['tokenized_prompt_mask'],bool),skills=jnp.asarray(b['skills'],jnp.int32),skill_mask=jnp.asarray(b['skill_mask'],bool),objects=jnp.asarray(b['objects'],jnp.int32))
            (loss,aux),grads=grad_step(jax.random.fold_in(jax.random.fold_in(jax.random.key(7),step),lo//micro),model,obs,jnp.asarray(b['actions']))
            total=jax.tree.map(lambda x:x/(8//micro),grads) if total is None else jax.tree.map(lambda x,y:x+y/(8//micro),total,grads)
            losses.append(loss)
        state=apply(model,state,total)
        loss=float(jnp.mean(jnp.stack(losses)))
        if not np.isfinite(loss):raise ValueError('nonfinite training loss')
        return state,loss,ids

    def restore():
        state=nnx.state(model);state.replace_by_pure_dict(base_params);nnx.update(model,state)
        if parameter_digest(model,trainable)!=before_trainable:raise AssertionError('base reset failed')
        return tx.init(nnx.state(model,trainable))

    print('PROFILE_START',flush=True)
    profile=[]
    for i in range(22):
        tick=time.monotonic()
        try:opt_state,loss,_=update('M1',i,opt_state)
        except Exception as exc:
            if i==0 and micro==2 and ('RESOURCE_EXHAUSTED' in str(exc) or 'out of memory' in str(exc).lower()):
                micro=1;opt_state=restore();opt_state,loss,_=update('M1',i,opt_state)
            else:raise
        profile.append(time.monotonic()-tick)
        print(json.dumps(dict(phase='profile',step=i+1,loss=loss,seconds=profile[-1])),flush=True)
    if parameter_digest(model,frozen)!=before_frozen:raise AssertionError('profile mutated frozen parameters')
    opt_state=restore()
    rate=max(np.mean(profile[2:]),np.quantile(profile[2:],.9))
    # Reserve model serialization/hash/reinitialization overhead and 15% timing margin.
    available=manifest['train_deadline']-time.time()-600
    steps=min(1000,int(available/(2*rate*1.15))//100*100)
    budget=dict(updates=steps,microbatch=micro,accumulation=8//micro,profile_seconds=profile,seconds_per_update=float(rate),decided_epoch=time.time(),train_deadline=manifest['train_deadline'],optimizer=dict(lr=1e-6,b1=.9,b2=.95,weight_decay=1e-5,clip=1.),frozen_before=before_frozen,trainable_before=before_trainable,reset_after_profile=True)
    write(root/'training_budget.json',budget)
    if steps<200:raise RuntimeError('fewer than 200 matched updates fit before training deadline')
    for condition in ('M0','M1'):
        if condition=='M1':opt_state=restore()
        directory=root/'models'/condition;directory.mkdir(parents=True,exist_ok=False)
        started=time.monotonic()
        with (directory/'training.jsonl').open('w') as log:
            for i in range(steps):
                if time.time()>=manifest['train_deadline']:raise TimeoutError('training deadline, no unequal-budget comparison')
                opt_state,loss,ids=update(condition,i,opt_state)
                row=dict(step=i+1,loss=loss,samples=ids,elapsed=time.monotonic()-started)
                log.write(json.dumps(row)+'\n');log.flush()
                if (i+1)%10==0:print(json.dumps(dict(model=condition,step=i+1,loss=loss,elapsed=row['elapsed'])),flush=True)
        after_frozen=parameter_digest(model,frozen);after_trainable=parameter_digest(model,trainable)
        if after_frozen!=before_frozen or after_trainable==before_trainable:raise AssertionError('frozen/update invariant failed')
        ocp.PyTreeCheckpointer().save(str(directory/'params'),nnx.state(model).to_pure_dict())
        (directory/'assets').symlink_to(Path(manifest['base_checkpoint']).parent/'assets',target_is_directory=True)
        write(directory/'result.json',dict(status='complete',updates=steps,frozen_before=before_frozen,frozen_after=after_frozen,trainable_after=after_trainable,seconds=time.monotonic()-started,checkpoint=str(directory/'params')))
    (root/'TRAIN_READY').write_text(str(time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    try:main(a.root)
    except Exception as exc:write(a.root/'TRAIN_ERROR.json',dict(error=repr(exc)));raise
