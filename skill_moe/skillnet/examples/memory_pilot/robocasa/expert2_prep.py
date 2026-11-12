"""Small RM-EXP2 bridge preparation run using the fixed Context+history drafts.

This is a new, resumable-ish preparation runner. It does not alter RM-B/RM-C
artifacts and writes only under expert-02/prep.
"""
from __future__ import annotations

import importlib.util
import json
import os
import pickle
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
STATE = Path(os.environ.get('SKILLNET_STATE_DIR', '/work/.skillnet-state'))
ROOT = Path(os.environ.get('ROBOCASA_OUTPUT_ROOT', str(STATE / 'outputs/robocasa-memory')))
WORKTREE = Path(os.environ.get('SKILLNET_REPO_ROOT', str(STATE / 'worktrees/robocasa-memory')))
CTX = ROOT / 'ctx-01'
KEYLITE = ROOT / 'key-lite-01'
SOURCE = ROOT / 'memory-01/policy-order-v2'
EXPERT1 = ROOT / 'expert-01'
MAG = KEYLITE / 'magnitude/magnitude_head.pt'
OUT = Path(os.environ.get('ROBOCASA_EXPERT2_PREP', str(ROOT / 'expert-02/prep')))
T0S = (0.1, 0.2, 0.3)
SEED = 20261111

sys.path.insert(0, str(WORKTREE / 'skill_moe/skillnet/examples/memory_pilot/robocasa'))
import experiment as ex
import expert
import magnitude_head


def _write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=ex._json_default) + '\n')
    tmp.replace(path)


def _load_ranker():
    script = KEYLITE / 'run_key_lite.py'
    spec = importlib.util.spec_from_file_location('rm_exp2_keylite', script)
    rk = importlib.util.module_from_spec(spec); spec.loader.exec_module(rk)
    rk.DEVICE = 'cpu'
    return rk


def _feature_context(rk, ck, z, feat, split):
    # `assemble_context` is the exact feature normalization used to train the
    # fixed Context+history ranker and its candidate keys.
    qctx, cand = rk.assemble_context(z, feat, ck['norm'], split)
    return np.asarray(qctx, np.float32), np.asarray(cand, np.float32)


def _initial_mask(frame):
    valid = min(max(int(frame), 0), 10)
    mask = np.zeros(10, bool)
    if valid:
        mask[10-valid:] = True
    return mask


def _make_dynamic_drafts(z, split, qctx, candidate_ctx, pool, rank_ck, mag_ck, rank_net, mag_net,
                         *, batch_size=96, device='cpu'):
    """Generate autoregressive 50-step drafts; each selection excludes its episode."""
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
    qkey_t = torch.as_tensor(candidate_ctx, dtype=torch.float32, device=device)
    rank_net = rank_net.to(device).eval()
    mag_net = mag_net.to(device).eval()
    with torch.no_grad():
        kemb = torch.nn.functional.normalize(rank_net.key(qkey_t), dim=-1)
    keys = np.asarray(pool['key'], np.float32)
    magnitude = np.sqrt(np.mean(np.sum(np.asarray(pool['action10'], np.float64)[:, :, 5:11] ** 2, axis=2), axis=1)).astype(np.float32)
    unit_meta = np.concatenate([keys, np.log(np.maximum(magnitude, 1e-6))[:, None]], axis=1).astype(np.float32)
    norm = rank_ck['hist_norm']
    hmean = np.asarray(norm['mean'], np.float32); hstd = np.asarray(norm['std'], np.float32)
    fmean = np.asarray(mag_ck['feature_mean'], np.float32); fstd = np.asarray(mag_ck['feature_std'], np.float32)
    global_ep = np.asarray(z[f'{split}_episode'], np.int64)
    frames = np.asarray(z[f'{split}_frame'], np.int64)
    hist0_store = np.asarray(z[f'{split}_hist'], np.float32)
    hist0 = ex.action_storage_to_policy(hist0_store)
    n = len(global_ep)
    drafts = np.zeros((n, 50, 12), np.float32)
    first_units = np.full(n, -1, np.int32)
    first_lengths = np.zeros(n, np.int32)
    picked_initial = np.full(n, -1, np.int32)
    # Process chains in bounded batches. At a given chain round, active rows
    # are scored together, preserving their own autoregressive histories.
    for base in range(0, n, int(batch_size)):
        ids = np.arange(base, min(base + batch_size, n))
        h = hist0[ids].copy()
        masks = np.stack([_initial_mask(frames[i]) for i in ids])
        cursor = np.zeros(len(ids), np.int32)
        while np.any(cursor < 50):
            active = np.flatnonzero(cursor < 50)
            # Ranker query: fixed context + normalized current 10 executed
            # commands + validity mask, matching its training contract.
            hn = ((h[active] - hmean) / hstd) * masks[active, :, None]
            query = np.concatenate([qctx[ids[active]], hn.reshape(len(active), -1), masks[active].astype(np.float32)], axis=1).astype(np.float32)
            with torch.no_grad():
                qt = torch.as_tensor(query, dtype=torch.float32, device=device)
                qemb = torch.nn.functional.normalize(rank_net.query(qt), dim=-1)
                scores = (qemb @ kemb.T) * rank_net.log_scale.exp()
                # Match the same-episode exclusion used by runtime P5.
                ep_t = torch.as_tensor(global_ep[ids[active]], dtype=torch.long, device=device)
                cand_ep_t = torch.as_tensor(pool['global_episode'], dtype=torch.long, device=device)
                scores = scores.masked_fill(ep_t[:, None] == cand_ep_t[None, :], -torch.inf)
                pick = scores.argmax(dim=1).cpu().numpy().astype(np.int32)
            xmag = np.concatenate([query, unit_meta[pick]], axis=1)
            xmag = torch.as_tensor((xmag - fmean) / fstd, dtype=torch.float32, device=device)
            with torch.no_grad():
                scales = torch.exp(mag_net(xmag)).clamp(.25, 4.).cpu().numpy()
            for local, u in zip(active, pick, strict=True):
                row = int(ids[local]); start_at = int(cursor[local])
                lo, hi = int(pool['action_offsets'][u]), int(pool['action_offsets'][u+1])
                raw = np.asarray(pool['actions_flat'][lo:hi], np.float32).copy()
                raw[:, 5:11] *= float(scales[np.flatnonzero(active == local)[0]])
                emitted = raw[:min(len(raw), 50-start_at)]
                if not len(emitted):
                    raise RuntimeError(f'empty unit selected for row {row}')
                drafts[row, start_at:start_at+len(emitted)] = emitted
                if start_at == 0:
                    first_units[row] = int(u); first_lengths[row] = int(len(raw)); picked_initial[row] = int(u)
                # Histories in the Context+history model are fork/policy order.
                policy_emitted = ex.action_storage_to_policy(emitted)
                h[local] = np.concatenate([h[local], policy_emitted], axis=0)[-10:]
                masks[local] = True
                cursor[local] += len(emitted)
        if (base // batch_size) % 10 == 0:
            print(json.dumps({'stage': f'drafts_{split}', 'rows_done': int(min(base+batch_size,n)), 'rows': n}), flush=True)
    return {'draft': drafts, 'first_unit': first_units, 'first_unit_length': first_lengths,
            'initial_unit': picked_initial}


def _load_models(device='cpu'):
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
    rk = _load_ranker()
    z = np.load(CTX / 'cache/rows.npz', allow_pickle=False)
    pool = dict(np.load(CTX / 'cache/pool.npz', allow_pickle=False))
    feat = np.asarray(np.load(CTX / 'cache/prefix_mean_fp16.npy', mmap_mode='r'), np.float32)
    rank_ck = torch.load(KEYLITE / 'context_history.pt', map_location='cpu', weights_only=False)
    rank_net = rk.RankNet(rank_ck['q_dim'], rank_ck['k_dim']); rank_net.load_state_dict(rank_ck['state'])
    mag_ck = torch.load(MAG, map_location='cpu', weights_only=False)
    mag_net = magnitude_head._MLP.create(int(mag_ck['input_dim'])); mag_net.load_state_dict(mag_ck['state'])
    return rk, z, pool, feat, rank_ck, rank_net, mag_ck, mag_net


def prepare_drafts(out=OUT):
    import torch
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    rk, z, pool, feat, rank_ck, rank_net, mag_ck, mag_net = _load_models('cpu')
    manifest=json.loads((SOURCE/'manifest.json').read_text())
    mem_infos={(int(d['task_id']),int(d['episode'])):d for d in manifest['demos'] if d['split']=='memory'}
    val_pairs=set(tuple(map(int,p)) for p in expert._mem_val_episodes(manifest))
    task=np.asarray(z['memory_task'],np.int16); global_ep=np.asarray(z['memory_episode'],np.int64)
    local_ep=global_ep%1000; frame=np.asarray(z['memory_frame'],np.int32)
    # All cached memory windows with a Context feature are eligible. The target
    # is a 50-step storage-order slice, padded at the demo tail like expert.py.
    keep=np.asarray([(int(t),int(e)) in mem_infos for t,e in zip(task,local_ep,strict=True)],bool)
    ix=np.flatnonzero(keep)
    if len(ix)!=len(task): raise AssertionError('Context memory cache contains rows outside the 80-demo memory split')
    qctx_all, cand_ctx = _feature_context(rk, rank_ck, z, feat, 'memory')
    qctx=qctx_all[ix]
    qz={'memory_episode':global_ep[ix],'memory_frame':frame[ix],'memory_hist':np.asarray(z['memory_hist'],np.float32)[ix]}
    generated=_make_dynamic_drafts(qz,'memory',qctx,cand_ctx,pool,rank_ck,mag_ck,rank_net,mag_net)
    targets=np.zeros((len(ix),50,12),np.float32); states=np.asarray(z['memory_state'],np.float32)[ix]
    validation=np.zeros(len(ix),bool)
    loaded={}
    for row_id,i in enumerate(ix):
        t,e,f=int(task[i]),int(local_ep[i]),int(frame[i])
        info=mem_infos[(t,e)]
        if (t,e) not in loaded: loaded[(t,e)]=expert.ex._load_cached_episode(SOURCE,t,e)[0]
        action=loaded[(t,e)]
        sl=action[f:min(f+50,len(action))]
        targets[row_id,:len(sl)]=sl if len(sl) else 0.
        validation[row_id]=(t,e) in val_pairs
    # Verify that initial selection agrees with the fixed ranker on the same
    # context/history and same-episode exclusion.
    hist_rows = rk.build_history_rows(z, 'memory')
    baseline_scores = rk.predict_ranker(rank_ck, z, feat, pool, split='memory', hist_data=hist_rows, history_condition='correct', exclude_episode=True)
    expected = baseline_scores[ix].argmax(axis=1)
    if not np.array_equal(expected, generated['initial_unit']):
        raise AssertionError(f'initial retrieval mismatch {np.mean(expected != generated["initial_unit"]):.4f}')
    dest = out / 'drafts/memory_dynamic.npz'
    if dest.exists(): raise FileExistsError(f'refusing to overwrite {dest}')
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('wb') as f:
        np.savez_compressed(f, task=task[ix], episode=local_ep[ix], frame=frame[ix],
            validation=validation, draft=generated['draft'], target=targets,
            hist=np.asarray(z['memory_hist'],np.float32)[ix], state=states, first_unit=generated['first_unit'],
            first_unit_length=generated['first_unit_length'], source_context_row=ix)
    _write_json(out / 'drafts/summary.json', {'rows':len(ix),'validation_rows':int(validation.sum()),
        'train_rows':int((~validation).sum()),'initial_selection_matches_fixed_ranker':True,
        'candidate_episode_exclusion':'same global episode masked at every chained selection',
        'history_order':'storage LeRobot for target/cache; fork order for Context+history query; generated units converted before history update',
        'scale':'fixed RM-MAG magnitude_head.pt, exp(log scale) clipped [.25,4], applied to action storage channels 5:11 only',
        'horizon':50,'source':'all 11,314 Context cache memory rows, aligned to the 80 memory demonstrations'})
    return dest


def _bridge_grad_step(filt):
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    @nnx.jit
    def grad_step(model, obs, target, draft, key):
        key, kt, kp, ke = jax.random.split(key, 4)
        b = target.shape[0]
        choices = jnp.asarray(T0S, jnp.float32)
        t0 = choices[jax.random.randint(kt, (b,), 0, len(T0S))]
        progress = jax.random.uniform(kp, (b,), minval=.01, maxval=1.)
        noise = jax.random.normal(ke, target.shape)
        xt, model_t, velocity = expert._bridge_training_sample(target, draft, noise, t0, progress)
        weight = jnp.asarray([1.,1.,1.,1.,1.,1.,1.,.1,.1,.1,.1,.1]+[0.]*20, jnp.float32)
        def loss_fn(m):
            pred = m.velocity(obs, xt, model_t)
            return jnp.sum((pred-velocity)**2 * weight[None,None,:])/(jnp.sum(weight)*target.shape[1]*b)
        loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, filt))(model)
        return loss, grads
    return grad_step


def _prepare_training_obs_cache(out, data, train_ids, info, *, rows_per_task=300, seed=SEED):
    """Cache a small stratified set of training observations for the smoke fit.

    Reusing decoded/transformed observations keeps 200 updates feasible on the
    shared CPU fallback while the other team owns the GPU.
    """
    import jax
    import jax.numpy as jnp
    cache_dir=Path(out)/'train/obs_cache';cache_dir.mkdir(parents=True,exist_ok=True)
    tasks=np.asarray(data['task']); rng=np.random.default_rng(seed)
    selected=[]
    for task in sorted(set(tasks[train_ids].tolist())):
        candidates=train_ids[tasks[train_ids]==task]
        take=min(int(rows_per_task),len(candidates))
        selected.extend(rng.choice(candidates,take,replace=False).tolist())
    selected=np.asarray(selected,np.int64)
    task_ids=sorted(set(tasks[selected].tolist()))
    for task in task_ids:
        path=cache_dir/f'task_{int(task)}.pkl'
        if path.exists(): continue
        ids=selected[tasks[selected]==task]
        observations=[]; targets=[]; tree_def=None; leaf_rows=[]
        for n,row in enumerate(ids):
            obs,target,_,_=expert._build_obs_and_target(info[(int(data['task'][row]),int(data['episode'][row]))],int(data['frame'][row]),expert.SOURCE,expert.CHECKPOINT)
            leaves,td=jax.tree.flatten(obs)
            if tree_def is None: tree_def=td
            elif td!=tree_def: raise AssertionError('observation PyTree changed across cache rows')
            arrays=[]
            for leaf in leaves:
                a=np.asarray(jax.device_get(leaf))
                if a.shape[:1]==(1,): a=a[0]
                arrays.append(a)
            leaf_rows.append(arrays);targets.append(np.asarray(target,np.float32))
            if (n+1)%8==0: print(json.dumps({'stage':'cache_training_obs','task':int(task),'rows_done':n+1,'rows':len(ids)}),flush=True)
        stacked=[np.stack([row[j] for row in leaf_rows]) for j in range(len(leaf_rows[0]))]
        payload={'row_ids':ids,'tree_def':tree_def,'leaves':stacked,'targets':np.stack(targets)}
        tmp=path.with_suffix('.pkl.tmp')
        with tmp.open('wb') as f: pickle.dump(payload,f,protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
    cache_map={}; common_tree=None
    for task in task_ids:
        with (cache_dir/f'task_{int(task)}.pkl').open('rb') as f: payload=pickle.load(f)
        if common_tree is None: common_tree=payload['tree_def']
        elif payload['tree_def']!=common_tree: raise AssertionError('cached observation tree differs by task')
        leaves=payload['leaves']
        for pos,row in enumerate(payload['row_ids']):
            cache_map[int(row)]=([np.asarray(x[pos]) for x in leaves],np.asarray(payload['targets'][pos],np.float32))
    if set(cache_map)!=set(map(int,selected)): raise AssertionError('observation cache row inventory mismatch')
    return selected,cache_map,common_tree


def train_small(out=OUT, *, steps=200, effective_batch=32, microbatch=4, threads=8):
    """Train 200 small updates with true gradient accumulation (effective 32)."""
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    import optax
    from openpi.shared.nnx_utils import PathRegex
    import torch
    torch.set_num_threads(threads)
    if effective_batch % microbatch: raise ValueError('effective_batch must be divisible by microbatch')
    if torch.cuda.is_available() and os.environ.get('JAX_PLATFORMS') != 'cpu': torch.cuda.set_per_process_memory_fraction(.25)
    out=Path(out); data=np.load(out/'drafts/memory_dynamic.npz',allow_pickle=False)
    stats_module=__import__('openpi.shared.normalize',fromlist=['load'])
    stats=stats_module.load(expert.CHECKPOINT/'assets')['actions']
    draft_model=np.stack([expert._draft_to_model(d,stats) for d in data['draft']]).astype(np.float32)
    train_ids=np.flatnonzero(~data['validation'])
    if len(train_ids)<effective_batch: raise ValueError('not enough train rows')
    manifest=json.loads((expert.SOURCE/'manifest.json').read_text())
    info={(int(r['task_id']),int(r['episode'])):r for r in manifest['demos']}
    model,_=expert._load_expert_model()
    filt=nnx.All(nnx.Param,PathRegex(r'.*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*'))
    selected=nnx.state(model,filt)
    tx=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(2.5e-5,b1=.9,b2=.95,weight_decay=1e-5))
    opt=tx.init(selected); grad_step=_bridge_grad_step(filt)
    rng=np.random.default_rng(SEED); updates=[]
    train_dir=out/'train';train_dir.mkdir(parents=True,exist_ok=True)
    # Preserve all model/optimizer state; observation cache files are reusable.
    unexpected=[p.name for p in train_dir.iterdir() if p.name!='obs_cache' and p.suffix not in ('.pkl',)]
    if unexpected: raise FileExistsError(f'train output already contains training artifacts: {unexpected}')
    sampled_ids,obs_cache,obs_tree=_prepare_training_obs_cache(out,data,train_ids,info)
    from flax import serialization
    for step in range(1,int(steps)+1):
        ids=rng.choice(sampled_ids,int(effective_batch),replace=True)
        accum=None; loss_values=[]
        for mi in range(0,len(ids),microbatch):
            sub=ids[mi:mi+microbatch]
            leaves=[jnp.asarray(np.stack([obs_cache[int(row)][0][j] for row in sub])) for j in range(len(obs_cache[int(sub[0])][0]))]
            obs=jax.tree.unflatten(obs_tree,leaves)
            target=jnp.asarray(np.stack([obs_cache[int(row)][1] for row in sub]),jnp.float32);draft=jnp.asarray(draft_model[sub])
            key=jax.random.fold_in(jax.random.key(SEED),step*100+mi)
            loss,grads=grad_step(model,obs,target,draft,key)
            accum=grads if accum is None else jax.tree.map(lambda a,b:a+b,accum,grads)
            loss_values.append(float(loss))
        accum=jax.tree.map(lambda g:g/float(len(ids)//microbatch),accum)
        selected=nnx.state(model,filt);updates0,opt=tx.update(accum,opt,selected)
        nnx.update(model,optax.apply_updates(selected,updates0))
        rec={'step':step,'loss':float(np.mean(loss_values)),'effective_batch':effective_batch,'microbatch':microbatch,
             't0_values':list(T0S),'t_formula':'model_t=t0*progress; x=(1-progress)*target+progress*((1-t0)*draft+t0*epsilon)'}
        updates.append(rec)
        if step%50==0 or step==steps:
            cp=train_dir/f'expert_{step:04d}.msgpack'
            cp.write_bytes(serialization.to_bytes(nnx.state(model,filt).to_pure_dict()))
            _write_json(train_dir/'progress.json',{'status':'running','latest_step':step,'steps':steps,'updates':updates,
                'checkpoint':cp.name,'trainable_filter':'llm.*_1|action_in_proj|time_mlp|action_out_proj','VLM_frozen':True})
            print(json.dumps({'stage':'train','step':step,'loss':rec['loss']}),flush=True)
    _write_json(train_dir/'progress.json',{'status':'complete','latest_step':int(steps),'steps':steps,'updates':updates,
        'checkpoint':f'expert_{steps:04d}.msgpack','trainable_filter':'llm.*_1|action_in_proj|time_mlp|action_out_proj','VLM_frozen':True})
    return train_dir/f'expert_{steps:04d}.msgpack'


def _sample_eval_rows(data, n_per_task=8, seed=41011):
    tasks=np.asarray(data['task']);eps=np.asarray(data['episode']);frames=np.asarray(data['frame'])
    valid=np.flatnonzero(data['validation'])
    rng=np.random.default_rng(seed);chosen=[]
    for task in sorted(set(tasks[valid].tolist())):
        ids=valid[tasks[valid]==task]
        # Stratify by episode then draw one available frame for each episode.
        by_ep={int(e):ids[eps[ids]==e] for e in np.unique(eps[ids])}
        episode_ids=sorted(by_ep);rng.shuffle(episode_ids)
        for ep in episode_ids[:n_per_task]:
            group=by_ep[ep]; chosen.append(int(group[rng.integers(len(group))]))
    return np.asarray(chosen,np.int64)


def evaluate_small(out=OUT, *, steps=200, n_per_task=8):
    import jax
    import jax.numpy as jnp
    import torch
    from flax import serialization
    from openpi.shared import normalize
    from openpi.shared.nnx_utils import module_jit
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','8')))
    if torch.cuda.is_available() and os.environ.get('JAX_PLATFORMS') != 'cpu': torch.cuda.set_per_process_memory_fraction(.25)
    out=Path(out); data=np.load(out/'drafts/memory_dynamic.npz',allow_pickle=False)
    model_path=out/'train'/f'expert_{steps:04d}.msgpack'
    rows=_sample_eval_rows(data,n_per_task=n_per_task)
    manifest=json.loads((expert.SOURCE/'manifest.json').read_text())
    info={(int(r['task_id']),int(r['episode'])):r for r in manifest['demos']}
    base,_=expert._load_expert_model();trained,_=expert._load_expert_model();expert._overlay_expert(trained,model_path)
    stats=normalize.load(expert.CHECKPOINT/'assets')['actions']
    bfn=module_jit(base.sample_bridge,static_argnames=('num_steps',));tfn=module_jit(trained.sample_bridge,static_argnames=('num_steps',));nfn=module_jit(base.sample_actions,static_argnames=('num_steps',))
    records=[]
    for row in rows:
        obs,target,_,_=expert._build_obs_and_target(info[(int(data['task'][row]),int(data['episode'][row]))],int(data['frame'][row]),expert.SOURCE,expert.CHECKPOINT)
        d=np.zeros((1,50,32),np.float32);d[0,:,:12]=ex.action_storage_to_policy(data['draft'][row]);d=expert._norm_action(d,stats)
        noise=np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(4101),int(row)),(1,50,32)),np.float32)
        normal=np.asarray(nfn(jax.random.key(0),obs,num_steps=10,noise=jnp.asarray(noise)))[0]
        first_len=max(1,min(50,int(data['first_unit_length'][row])))
        truth=np.asarray(target,np.float32)
        def mse(pred,horizon):
            e=(np.asarray(pred)[:horizon,:6]-truth[:horizon,:6])**2
            return float(e.mean())
        rec={'row':int(row),'task':int(data['task'][row]),'episode':int(data['episode'][row]),'frame':int(data['frame'][row]),'first_unit_length':first_len,
             'draft_first5_mse6':mse(d[0],5),'draft_unit_mse6':mse(d[0],first_len),
             'pi05_normal_first5_mse6':mse(normal,5),'pi05_normal_unit_mse6':mse(normal,first_len)}
        for t0 in T0S:
            bp=np.asarray(bfn(jax.random.key(0),obs,jnp.asarray(d),jnp.asarray([t0],jnp.float32),num_steps=10,noise=jnp.asarray(noise)))[0]
            tp=np.asarray(tfn(jax.random.key(0),obs,jnp.asarray(d),jnp.asarray([t0],jnp.float32),num_steps=10,noise=jnp.asarray(noise)))[0]
            for name,pred in ((f'pretrained_t{t0:.1f}',bp),(f'trained_t{t0:.1f}',tp)):
                rec[name+'_first5_mse6']=mse(pred,5);rec[name+'_unit_mse6']=mse(pred,first_len)
            rec[f'pretrained_t{t0:.1f}_distance_to_pi05_first5_mse6']=float(np.mean((bp[:5,:6]-normal[:5,:6])**2))
        records.append(rec)
        if len(records)%8==0: print(json.dumps({'stage':'eval','rows':len(records),'total':len(rows)}),flush=True)
    path=out/'evaluation_small.json'
    if path.exists(): raise FileExistsError(f'refusing overwrite {path}')
    means={k:float(np.mean([r[k] for r in records])) for k in records[0] if isinstance(records[0][k],(float,int)) and k not in ('row','task','episode','frame','first_unit_length')}
    _write_json(path,{'status':'complete','n_rows':len(records),'rows':records,'means':means,'selection':'memory-side Expert validation episodes only; fixed sample seed 41011; no heldout-120 data','noise':'same fold_in(4101,row_id) noise for pretrained and trained bridge','integrator_steps':10})
    return {'rows':len(records),'means':means}


def write_report(out=OUT, *, draft_summary=None, eval_summary=None, train_steps=200):
    out=Path(out); lines=['実装: 完了','', '# RM-EXP2-PREP', '',
        '確認はメモリ側の予約検証エピソードから層化した小標本で行った。保留120デモは使用していない。',
        '下書きは固定 Context＋履歴 ranker と固定 magnitude head を使い、50ステップを単位単位で連ねた。各選択で対象エピソードの候補を除外し、生成行動を履歴へ戻した。',
        'メモリ候補・下書き・正解行動は LeRobot 順で保持し、π0.5モデル入力には expert.py の既存変換でフォーク順へ変換した。',
        '学習は action expert / time MLP / action projection のみ。VLM は凍結。C3 と同じ `t=t0×progress` で、t0 は {0.1, 0.2, 0.3} から選んだ。実効バッチ32、microbatch4×勾配累積8。',
        '', '## 動作確認', '', f'- 学習更新: {train_steps}; 検証行数: {eval_summary["rows"] if eval_summary else "未実施"}.',
        '- 比較対象: draft、π0.5 通常生成、学習前 π0.5 bridge、学習Expert bridge。単位長は最初に選ばれた実単位長（最大50）。',
        '', '| 条件 | 先頭5 step MSE6 | 最初の単位長 MSE6 | π0.5通常出力との距離（先頭5） |',
        '|---|---:|---:|---:|']
    if eval_summary:
        m=eval_summary['means']
        for name, first, unit, dist in [
            ('P5大きさ推定 下書き','draft_first5_mse6','draft_unit_mse6',None),
            ('π0.5 通常','pi05_normal_first5_mse6','pi05_normal_unit_mse6',None)]:
            lines.append(f'| {name} | {m[first]:.6f} | {m[unit]:.6f} | — |')
        for t0 in T0S:
            for prefix,label in [('pretrained','学習前 bridge'),('trained','学習Expert bridge')]:
                first=f'{prefix}_t{t0:.1f}_first5_mse6';unit=f'{prefix}_t{t0:.1f}_unit_mse6';dist=f'pretrained_t{t0:.1f}_distance_to_pi05_first5_mse6' if prefix=='pretrained' else None
                lines.append(f'| {label} t0={t0:.1f} | {m[first]:.6f} | {m[unit]:.6f} | {m[dist]:.6f} |' if dist else f'| {label} t0={t0:.1f} | {m[first]:.6f} | {m[unit]:.6f} | — |')
        draft=m['draft_first5_mse6']; trained_vals=[(m[f'trained_t{t:.1f}_first5_mse6'],t) for t in T0S]; best,bt=min(trained_vals)
        lines += ['', f'- この確認標本で最良の学習Expert bridge は t0={bt:.1f}、先頭5 step MSE6={best:.6f}（下書き {draft:.6f}）。']
        if best<draft: lines.append('- 小規模確認では学習が下書きを改善する方向。')
        else: lines.append('- 小規模確認では下書きより改善する方向は確認できず、学習前Expertへの引き戻しも距離列で確認する。')
        lines.append(f'- 学習前 bridge と π0.5 通常出力の先頭5 step距離は t0別に表へ記載。')
    lines += ['', 'この結果は200更新の準備確認であり、本番学習・閉ループ評価の成績ではない。設定は結果を見て選び直していない。', '']
    path=out/'report.md'
    if path.exists(): raise FileExistsError(path)
    path.write_text('\n'.join(lines))


def run(out=OUT, *, steps=200, microbatch=4, effective_batch=32, device='cpu'):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    draft_path=out/'drafts/memory_dynamic.npz'
    if not draft_path.exists(): prepare_drafts(out)
    cp=out/'train'/f'expert_{steps:04d}.msgpack'
    if not cp.exists(): cp=train_small(out,steps=steps,effective_batch=effective_batch,microbatch=microbatch)
    eval_path=out/'evaluation_small.json'
    result=json.loads(eval_path.read_text()) if eval_path.exists() else evaluate_small(out,steps=steps)
    ds=json.loads((out/'drafts/summary.json').read_text())
    if not (out/'report.md').exists(): write_report(out,draft_summary=ds,eval_summary=result,train_steps=steps)
    if not (out/'status.json').exists():
        _write_json(out/'status.json',{'status':'complete','draft_rows':ds['rows'],'train_steps':steps,'evaluation_rows':result['rows'],'checkpoint':str(cp)})
    return result


if __name__ == '__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=200);ap.add_argument('--microbatch',type=int,default=4);ap.add_argument('--effective-batch',type=int,default=32);ap.add_argument('--device',default='cuda');ap.add_argument('--out',type=Path,default=OUT)
    a=ap.parse_args();run(out=a.out,steps=a.steps,microbatch=a.microbatch,effective_batch=a.effective_batch,device=a.device)
