"""Episode-held-out, monotone confidence calibration; never simulator labels."""
import argparse
from itertools import combinations_with_replacement
import json
from pathlib import Path

import numpy as np


def choose_schedule(scores, errors):
    """Columns: normal fallback, bridge .6, .4, .2; high score -> less noise."""
    if errors.shape != (len(scores),4) or not np.isfinite(errors).all():
        raise ValueError('expected finite [N,4] paired generation errors')
    edges = np.quantile(scores,[.2,.4,.6,.8])
    bins = np.searchsorted(edges,scores,side='right')
    options = list(combinations_with_replacement(range(4),5))
    def key(option):
        loss = float(errors[np.arange(len(scores)),np.asarray(option)[bins]].mean())
        # Ties prefer the safer unmodified policy.
        return loss,sum(option),option
    best = min(options,key=key)
    return {'edges':edges.tolist(),'modes':list(best),'dev_mse':key(best)[0],
            'mode_meaning':['fallback','bridge_t0_0.6','bridge_t0_0.4','bridge_t0_0.2']}


def apply_schedule(scores,schedule):
    bins = np.searchsorted(schedule['edges'],scores,side='right')
    return np.asarray(schedule['modes'])[bins]


def main():
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from openpi.shared.nnx_utils import module_jit, PathRegex
    from run import load_policy,load_rows,obs_batch,write_json,log,calibration_episodes
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    args=parser.parse_args()
    rows=load_rows(args.output)
    drafts=np.load(args.output/'drafts.npy',mmap_mode='r')
    scores=np.load(args.output/'retrieved_score.npy',mmap_mode='r')
    rng=np.random.default_rng(20261105)
    dev_episodes,eval_episodes=calibration_episodes(rows)
    ids=[];is_dev=[]
    for task in np.unique(rows['task']):
        episodes=np.unique(rows['episode'][(rows['task']==task)&rows['validation']])
        for episode in episodes:
            candidates=np.flatnonzero(rows['episode']==episode)
            selected=rng.choice(candidates,min(2,len(candidates)),replace=False)
            ids.extend(selected.tolist());is_dev.extend([episode in dev_episodes]*len(selected))
    ids=np.asarray(ids);is_dev=np.asarray(is_dev)
    base,_=load_policy(args.checkpoint)
    baseline=module_jit(base.sample_actions,static_argnames=('num_steps',))
    probe_obs=obs_batch(rows,ids[:4])
    probe_noise=jnp.zeros((len(ids[:4]),10,32))
    fallback_before=np.asarray(baseline(jax.random.key(19),probe_obs,num_steps=10,noise=probe_noise))
    # Freeze the baseline closure before overlaying only action-expert leaves.
    filt=nnx.All(nnx.Param,PathRegex(r'.*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*'))
    selection=args.output/'best_bridge.json'
    if not selection.exists():selection=args.output/'latest_bridge.json'
    checkpoint=Path(json.loads(selection.read_text())['path'])
    with ocp.PyTreeCheckpointer() as checkpointer:
        partial=checkpointer.restore(checkpoint)
    state=nnx.state(base,filt)
    state.replace_by_pure_dict(jax.tree.map(jnp.asarray,partial));nnx.update(base,state)
    fallback_after=np.asarray(baseline(jax.random.key(19),probe_obs,num_steps=10,noise=probe_noise))
    np.testing.assert_array_equal(fallback_before,fallback_after)
    bridge=module_jit(base.sample_bridge,static_argnames=('num_steps',))
    errors=np.zeros((len(ids),4),np.float64)
    for noise_seed in [4101,4102]:
        for start in range(0,len(ids),4):
            selected=ids[start:start+4]
            obs=obs_batch(rows,selected)
            # Per-row keys are independent of calibration batch composition.
            eps=np.stack([np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(noise_seed),int(i)),(10,32))) for i in selected])
            pred=baseline(jax.random.key(0),obs,num_steps=10,noise=jnp.asarray(eps))
            target=rows['actions'][selected]
            errors[start:start+len(selected),0]+=np.mean((np.asarray(pred)[:,:,:7]-target[:,:,:7])**2,axis=(1,2))/2
            key=jax.random.fold_in(jax.random.key(noise_seed),start)
            for ci,t0 in enumerate([.6,.4,.2],1):
                pred=bridge(key,obs,jnp.asarray(drafts[selected]),jnp.full(len(selected),t0),num_steps=10,noise=jnp.asarray(eps))
                errors[start:start+len(selected),ci]+=np.mean((np.asarray(pred)[:,:,:7]-target[:,:,:7])**2,axis=(1,2))/2
            log(stage='calibrate',noise=noise_seed,rows=start+len(selected),total=len(ids))
    schedule=choose_schedule(scores[ids][is_dev],errors[is_dev])
    modes=apply_schedule(scores[ids][~is_dev],schedule)
    eval_errors=errors[~is_dev]
    result={'schedule':schedule,'dev_rows':int(is_dev.sum()),'eval_rows':int((~is_dev).sum()),
        'eval_baseline_mse':float(eval_errors[:,0].mean()),
        'eval_adaptive_mse':float(eval_errors[np.arange(len(modes)),modes].mean()),
        'eval_fixed_bridge_mse':eval_errors[:,1:].mean(0).tolist(),
        'eval_fallback_fraction':float(np.mean(modes==0)),
        'fallback_identical_after_bridge_overlay':True,
        'metric':'normalized final-action MSE, active7; not closed-loop success',
        'note':'calibration/evaluation episodes are disjoint for new modules; public base may have seen both'}
    np.savez(args.output/'calibration_errors.npz',row_ids=ids,is_dev=is_dev,errors=errors,scores=scores[ids])
    write_json(args.output/'calibration.json',result)


if __name__=='__main__':
    main()
