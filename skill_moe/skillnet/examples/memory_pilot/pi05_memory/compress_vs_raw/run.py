#!/usr/bin/env python3
"""Fair offline comparison of raw and M1-b12 retrieved drafts."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from common import (FAIR, MEMORY, OUT, ROOT, RUN, T0S, base_module,
                    decoded_candidate_drafts, episode_ids, fair_module,
                    classify_diff, require_new_artifacts)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=True) + '\n')
    tmp.replace(path)


def _load_codec(base):
    return base.restore_codec(RUN)


def prepare(out=OUT):
    """Materialize b12 drafts for the already selected, fixed E-M2 candidates."""
    require_new_artifacts((out / 'drafts_prepared.npz', out / 'prepare.json'))
    fair = fair_module()
    base = base_module()
    rows = base.load_rows(RUN)
    actions = np.load(RUN / 'rows/actions.npy', mmap_mode='r', allow_pickle=False)
    features = np.load(RUN / 'pooled.npy', mmap_mode='r', allow_pickle=False)
    states = rows['state']
    bases = np.load(RUN / 'bases.npy', mmap_mode='r', allow_pickle=False)
    saved_train = np.load(FAIR / 'E-M2/training_drafts.npz', allow_pickle=False)
    train_q = np.asarray(saved_train['query_rows'], np.int64)
    train_c = np.asarray(saved_train['selected_candidate_rows'], np.int64)
    expected_train = fair.training_rows(rows)
    if not np.array_equal(train_q, expected_train):
        raise AssertionError('E-M2 train row sequence does not match fair-expert')
    if np.any(np.asarray(rows['task'][train_q]) < 10):
        raise AssertionError('Long contamination in training queries')
    pool = np.asarray(np.load(MEMORY / 'units.npz', allow_pickle=False)['pool_row_ids'], np.int64)
    fair.validate_candidates(train_c[:, None], train_q, rows, pool)
    batch_rows = np.load(FAIR / 'batch_row_ids.npy', allow_pickle=False)
    batch_sha = hashlib.sha256(batch_rows.tobytes()).hexdigest()
    trained = json.loads((FAIR / 'E-M2/trained.json').read_text())
    if batch_sha != trained['batch_row_ids_sha256']:
        raise AssertionError('shared E-M2 batch sequence hash mismatch')

    # Use precisely the same fixed Dev candidates and rank mixture as E-M2.
    dev_q = fair.validation_rows(rows)
    cand_file = np.load(FAIR / 'candidate_lists.npz', allow_pickle=False)
    qmap = {int(r): i for i, r in enumerate(cand_file['query_rows'])}
    dev_cands = np.stack([cand_file['candidate_rows'][qmap[int(r)]] for r in dev_q])
    dev_pick, dev_ranks = fair.sample_m2_training_draft(
        dev_cands, np.random.default_rng(fair.SEED + 1902))
    fair.validate_candidates(dev_pick[:, None], dev_q, rows, pool)

    # Fixed Long Eval rows, same candidate lists as E-M2 evaluation.
    eval_pred = np.load(FAIR / 'evaluation-final/offline_predictions.npz', allow_pickle=False)
    corrected_shuffle = np.load(
        FAIR / 'corrected-top1-shuffle-01/top1_shuffle_predictions.npz', allow_pickle=False)
    long_q = np.asarray(eval_pred['row_ids'], np.int64)
    long_m2 = np.asarray(eval_pred['m2_candidate_rows'], np.int64)[:, :1]
    if np.any(np.asarray(rows['task'][long_q]) >= 10) or len(long_q) != 548:
        raise AssertionError('Long evaluation rows malformed')
    if not np.array_equal(corrected_shuffle['query_rows'], long_q) or not np.array_equal(
            corrected_shuffle['source_candidate_rows'], long_m2[:, 0]):
        raise AssertionError('corrected Top-1 shuffle does not align with E-M2 query/candidate rows')
    shuffle_ids = np.asarray(corrected_shuffle['shuffled_candidate_rows'], np.int64)
    if not np.array_equal(np.sort(shuffle_ids), np.sort(long_m2[:, 0])):
        raise AssertionError('corrected shuffle does not preserve the exact M2 Top-1 multiset')
    if np.any(np.asarray(rows['task'])[shuffle_ids] == np.asarray(rows['task'])[long_q]):
        raise AssertionError('corrected shuffle contains a same-task candidate')
    m2_cache = np.load(ROOT / 'key-mechanism-01/production/candidate_rows.npz', allow_pickle=False)
    if not np.array_equal(m2_cache['query_rows'], long_q) or not np.array_equal(
            m2_cache['M2_salient_maxsim_rows'][:, :1], long_m2):
        raise AssertionError('cached M2 candidate row IDs differ from E-M2 evaluation')
    oracle = np.asarray(m2_cache['action_oracle_rows'], np.int64)[:, :1]
    shuffle = shuffle_ids[:, None]
    # Oracle is the raw active-action nearest neighbor over the 3,857-row pool.
    target = np.asarray(actions[long_q, :, :7], np.float32)
    pool_a = np.asarray(actions[pool, :, :7], np.float32)
    oracle_rows_check = np.empty(len(long_q), np.int64)
    for lo in range(0, len(long_q), 16):
        sl = slice(lo, min(lo + 16, len(long_q)))
        oracle_mse = np.mean((target[sl, None] - pool_a[None]) ** 2, axis=(2, 3))
        oracle_rows_check[sl] = pool[np.argmin(oracle_mse, axis=1)]
    if not np.array_equal(oracle_rows_check, oracle[:, 0]):
        raise AssertionError('saved action oracle is not the exact raw-action nearest row')
    fair.validate_candidates(oracle, long_q, rows, pool)
    fair.validate_candidates(shuffle, long_q, rows, pool)

    codec = _load_codec(base)
    train_dec = decoded_candidate_drafts(codec, bases, features, states, train_q, train_c[:, None])[:, 0]
    dev_dec = decoded_candidate_drafts(codec, bases, features, states, dev_q, dev_pick[:, None])[:, 0]
    eval_candidates = np.concatenate([long_m2, oracle, shuffle], axis=1)
    eval_dec = decoded_candidate_drafts(codec, bases, features, states, long_q, eval_candidates)
    # Persist compact active32 drafts and identity metadata for resumable stages.
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'drafts_prepared.npz', train_query_rows=train_q,
                        train_candidate_rows=train_c, train_drafts=train_dec,
                        dev_query_rows=dev_q, dev_candidate_rows=dev_pick,
                        dev_candidate_ranks=dev_ranks, dev_drafts=dev_dec,
                        eval_query_rows=long_q, m2_candidate_rows=long_m2[:, 0],
                        oracle_candidate_rows=oracle[:, 0], shuffle_candidate_rows=shuffle[:, 0],
                        eval_decoded_drafts=eval_dec,
                        eval_raw_drafts=np.asarray(actions[eval_candidates], np.float32),
                        eval_target=target, eval_noise=eval_pred['noise'],
                        hard_mask=eval_pred['hard_mask'], task_ids=eval_pred['task_ids'],
                        episode_ids=eval_pred['episode_ids'],
                        candidate_identity='dataset_row_ids', batch_row_ids_sha256=batch_sha)
    write_json(out / 'prepare.json', {
        'train_rows': int(len(train_q)), 'dev_rows': int(len(dev_q)), 'long_eval_rows': int(len(long_q)),
        'long_hard_rows': int(np.sum(eval_pred['hard_mask'])), 'memory_rows': int(len(pool)),
        'train_candidate_ids_exactly_E-M2': True, 'dev_candidate_rule': 'E-M2 fixed seed 20263006 rank mixture',
        'batch_row_ids_sha256': batch_sha, 'codec': 'run-01 M1 b12/r4; fit across all 40 tasks including Long',
        'candidate_indexing': 'direct dataset row ID; never memory-pool offset',
        'shuffle': 'corrected fair-expert exact Top-1 multiset permutation; all rows cross-task',
        'oracle_rows_match_exact_raw_action_argmin': True,
    })
    print(json.dumps({'stage': 'prepare', 'train': len(train_q), 'dev': len(dev_q), 'eval': len(long_q)}), flush=True)


import os as _os
_REPO = Path(__file__).resolve().parents[6]
_STATE_DIR = Path(_os.environ.get('SKILLNET_STATE_DIR', _REPO / '.skillnet-state'))
def train(out=OUT, *, steps=4000, batch=8, save_every=500, threads=8):
    import torch
    torch.set_num_threads(threads)
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    import optax
    import orbax.checkpoint as ocp
    from openpi.shared.nnx_utils import PathRegex, module_jit
    fair = fair_module(); base = base_module()
    rows = base.load_rows(RUN)
    prep = np.load(out / 'drafts_prepared.npz', allow_pickle=False)
    train_q = np.asarray(prep['train_query_rows'], np.int64)
    if not np.array_equal(train_q, fair.training_rows(rows)) or np.any(np.asarray(rows['task'][train_q]) < 10):
        raise AssertionError('Long or changed training query rows')
    batches = np.load(FAIR / 'batch_row_ids.npy', allow_pickle=False)
    if steps != 4000 or batch != 8 or save_every != 500:
        raise ValueError('fair comparison requires exact E-M2 4,000/8/500 schedule')
    e2trained = json.loads((FAIR / 'E-M2/trained.json').read_text())
    if hashlib.sha256(batches.tobytes()).hexdigest() != e2trained['batch_row_ids_sha256']:
        raise AssertionError('E-M2 batch_row_ids mismatch')
    train_pos = {int(r): i for i, r in enumerate(train_q)}
    model, _ = base.load_policy(_STATE_DIR / 'checkpoints/openpi-assets/checkpoints/pi05_libero')
    filt = nnx.All(nnx.Param, PathRegex(r'.*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*'))
    selected = nnx.state(model, filt)
    if sum(x.value.size for x in selected.flat_state().values()) < 100_000_000:
        raise ValueError('unexpected trainable action-expert parameter count')
    tx = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(2.5e-5, b1=.9, b2=.95, weight_decay=1e-5))
    opt = tx.init(selected); step_fn = base.make_bridge_step(filt, tx)
    jkey = jax.random.key(fair.SEED)
    expdir = out / 'E-b12'
    if expdir.exists():
        raise FileExistsError(f'refusing to overwrite {expdir}')
    (expdir / 'bridge').mkdir(parents=True)
    val_q = np.asarray(prep['dev_query_rows'], np.int64)
    val_draft = np.asarray(prep['dev_drafts'], np.float32)
    val_target = np.asarray(rows['actions'][val_q], np.float32)[:, :, :7]
    val_noise = np.stack([np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(88261), int(r)), (10,32))) for r in val_q])
    best, logs = float('inf'), []
    start = time.monotonic()
    for step, ids in enumerate(batches, start=1):
        draft = np.asarray(prep['train_drafts'][[train_pos[int(r)] for r in ids]], np.float32)
        jkey, subkey = jax.random.split(jkey)
        loss, opt = step_fn(model, opt, base.obs_batch(rows, ids), jnp.asarray(rows['actions'][ids]), jnp.asarray(draft), subkey)
        loss = float(loss)
        if not np.isfinite(loss): raise FloatingPointError(f'non-finite loss at {step}')
        if step % save_every == 0:
            ckpt = expdir / 'bridge' / f'{step:06d}'
            ocp.PyTreeCheckpointer().save(ckpt, nnx.state(model, filt).to_pure_dict())
            bridge = module_jit(model.sample_bridge, static_argnames=('num_steps',))
            vals = {}
            val_obs = base.obs_batch(rows, val_q)
            for t0 in T0S:
                pred = []
                for lo in range(0, len(val_q), batch):
                    sl = slice(lo, min(lo+batch, len(val_q)))
                    p = bridge(jax.random.key(0), jax.tree.map(lambda x: x[sl], val_obs), jnp.asarray(val_draft[sl]),
                               jnp.full((len(val_q[sl]),), t0), num_steps=10, noise=jnp.asarray(val_noise[sl]))
                    pred.append(np.asarray(p)[:, :, :7])
                vals[str(t0)] = float(np.mean((np.concatenate(pred)-val_target)**2))
            chosen = min(vals, key=vals.get); vmse = vals[chosen]
            logs.append({'step': step, 'loss': loss, 'dev_mse_by_t0': vals, 'selected_t0': float(chosen), 'elapsed_s': time.monotonic()-start})
            if vmse < best:
                best = vmse
                fair._write_json(expdir / 'best_bridge.json', {'path': str(ckpt), 'steps': step, 't0': float(chosen),
                    'validation_generated_active7_mse': vmse, 'selection': 'fixed E-M2 Dev candidates, seed 20263006 rank mixture; six heldout tasks; no Long'})
            fair._write_json(expdir / 'training_progress.json', {'status':'running','steps_done':step,'logs':logs,'best_mse':best})
            print(json.dumps({'stage':'train','step':step,'train_loss':loss,'dev':vals,'elapsed_s':time.monotonic()-start}), flush=True)
    fair._write_json(expdir / 'trained.json', {'expert':'E-b12','steps':steps,'seed':fair.SEED,'batch_size':batch,
        'train_rows':len(train_q),'dev_rows':len(val_q),'batch_row_ids_sha256':hashlib.sha256(batches.tobytes()).hexdigest(),
        'best':json.loads((expdir/'best_bridge.json').read_text()),'optimizer':'AdamW lr=2.5e-5 b1=.9 b2=.95 wd=1e-5; clip=1',
        'draft':'M1 codec decode of exactly E-M2 selected_candidate_rows', 'training_logs':logs,'seconds':time.monotonic()-start})
    fair._write_json(expdir / 'training_progress.json', {'status':'complete','steps_done':steps,'logs':logs,'best_mse':best})


def evaluate(out=OUT, *, batch=8, threads=8):
    require_new_artifacts((out / 'predictions_and_errors.npz', out / 'summary.json',
                           out / 'decision.json'))
    import torch
    torch.set_num_threads(threads)
    import jax
    import jax.numpy as jnp
    from openpi.shared.nnx_utils import module_jit
    fair = fair_module(); base = base_module(); rows = base.load_rows(RUN)
    prep = np.load(out / 'drafts_prepared.npz', allow_pickle=False)
    official = np.load(FAIR / 'evaluation-final/offline_predictions.npz', allow_pickle=False)
    q = np.asarray(prep['eval_query_rows'], np.int64)
    if not np.array_equal(q, official['row_ids']) or not np.array_equal(prep['eval_target'], official['target']) or not np.array_equal(prep['eval_noise'], official['noise']):
        raise AssertionError('reused predictions differ in query/target/noise alignment')
    hard = np.asarray(prep['hard_mask'], bool); episode = episode_ids(rows, q)
    target = np.asarray(prep['eval_target'], np.float32); noise = np.asarray(prep['eval_noise'], np.float32)
    names = ('m2','oracle','shuffle'); draft_index = {n:i for i,n in enumerate(names)}
    raw = np.asarray(prep['eval_raw_drafts'], np.float32); dec = np.asarray(prep['eval_decoded_drafts'], np.float32)
    if raw.shape != dec.shape or raw.shape[:2] != (len(q),3): raise AssertionError('bad candidate draft shape')
    # Load saved E-M2 weights and the newly trained E-b12 weights. Each model gets its own Dev-selected t0.
    specs = {
        'E-M2': json.loads((FAIR/'E-M2/best_bridge.json').read_text()),
        'E-b12': json.loads((out/'E-b12/best_bridge.json').read_text()),
    }
    preds = {}; mse = {}; official_normal = np.asarray(official['pred_official_normal'], np.float32)
    preds['official_normal'] = official_normal
    for expert, spec in specs.items():
        ckpt = Path(spec['path'])
        policy, _ = base.load_policy(_STATE_DIR / 'checkpoints/openpi-assets/checkpoints/pi05_libero')
        fair._skill_run().load_expert(policy, ckpt)
        fn = module_jit(policy.sample_bridge, static_argnames=('num_steps',))
        obs = base.obs_batch(rows, q)
        drafts = raw if expert == 'E-M2' else dec
        for kind, idx in draft_index.items():
            for t0 in T0S:
                outp=[]
                for lo in range(0,len(q),batch):
                    sl=slice(lo,min(lo+batch,len(q)))
                    p=fn(jax.random.key(0),jax.tree.map(lambda x:x[sl],obs),jnp.asarray(drafts[sl,idx]),
                         jnp.full((len(q[sl]),),t0),num_steps=10,noise=jnp.asarray(noise[sl]))
                    outp.append(np.asarray(p)[:,:,:7])
                preds[f'{expert}/{kind}/t0={t0:.1f}']=np.concatenate(outp).astype(np.float32)
        # Zero baseline for both trainable Experts.
        zero=np.zeros((len(q),10,32),np.float32)
        for t0 in T0S:
            outp=[]
            for lo in range(0,len(q),batch):
                sl=slice(lo,min(lo+batch,len(q)))
                p=fn(jax.random.key(0),jax.tree.map(lambda x:x[sl],obs),jnp.asarray(zero[sl]),jnp.full((len(q[sl]),),t0),
                     num_steps=10,noise=jnp.asarray(noise[sl]))
                outp.append(np.asarray(p)[:,:,:7])
            preds[f'{expert}/zero/t0={t0:.1f}']=np.concatenate(outp).astype(np.float32)
        del policy,fn
        jax.clear_caches()
    # Existing E-zero + zero and official normal are held fixed baselines.
    for t0 in T0S:
        preds[f'E-zero/zero/t0={t0:.1f}']=np.asarray(official[f'pred_E-zero_zero_t0_{t0:.1f}'],np.float32)
    for k,v in preds.items(): mse[k]=np.mean((v-target)**2,axis=(1,2)).astype(np.float32)
    # Existing E-M2+M2 is a reproducibility check, not a substitute for inference.
    for t0 in T0S:
        old=np.asarray(official[f'pred_E-M2_M2_top1_t0_{t0:.1f}'],np.float32)
        if not np.allclose(preds[f'E-M2/m2/t0={t0:.1f}'],old,rtol=0,atol=2e-5):
            raise AssertionError(f'E-M2 reused reference mismatch at t0={t0}')
    def summarize(x, mask=None):
        x=np.asarray(x); x=x if mask is None else x[mask]
        return {'mean':float(x.mean()),'median':float(np.median(x)),'n':int(len(x))}
    summaries={k:{'all':summarize(v),'hard':summarize(v,hard)} for k,v in mse.items()}
    pairs={}
    # Each Expert's own Dev-selected t0, as preregistered.
    t_raw=float(specs['E-M2']['t0']); t_comp=float(specs['E-b12']['t0'])
    def diff(a,b,mask): return fair._bootstrap_diff(a[mask],b[mask],episode[mask],reps=5000,seed=4101)
    for subset,mask in [('all',np.ones(len(q),bool)),('hard',hard)]:
        pairs[f'C1_{subset}']=diff(mse[f'E-M2/m2/t0={t_raw:.1f}'],mse[f'E-b12/m2/t0={t_comp:.1f}'],mask)
        pairs[f'C2_{subset}']=diff(mse[f'E-M2/oracle/t0={t_raw:.1f}'],mse[f'E-b12/oracle/t0={t_comp:.1f}'],mask)
        pairs[f'C3_{subset}']=diff(mse[f'E-b12/m2/t0={t_comp:.1f}'],mse[f'E-b12/shuffle/t0={t_comp:.1f}'],mask)
    pre={}
    for kind,i in draft_index.items():
        pre[kind]={'raw_mse':summarize(np.mean((raw[:,i,:,:7]-target)**2,axis=(1,2))),
                   'decoded_mse':summarize(np.mean((dec[:,i,:,:7]-target)**2,axis=(1,2))),
                   'compression_delta_mse':summarize(np.mean((dec[:,i,:,:7]-raw[:,i,:,:7])**2,axis=(1,2)))}
    sensitivity={}
    for expert,drafts in [('E-M2',raw),('E-b12',dec)]:
        sensitivity[expert]={}
        for t0 in T0S:
            p1=preds[f'{expert}/m2/t0={t0:.1f}']; ps=preds[f'{expert}/shuffle/t0={t0:.1f}']
            pd=np.sqrt(np.mean((p1-ps)**2,axis=(1,2)))
            dd=np.sqrt(np.mean((drafts[:,0,:,:7]-drafts[:,2,:,:7])**2,axis=(1,2)))
            sensitivity[expert][str(t0)]={'mean_output_distance':float(pd.mean()),'mean_draft_distance':float(dd.mean()),
                                          'mean_ratio':float(np.mean(pd/np.maximum(dd,1e-8)))}
    result={'rows':len(q),'hard_rows':int(hard.sum()),'selected':specs,'metrics':summaries,'paired_episode_bootstrap':pairs,
            'pre_generation_drafts':pre,'output_sensitivity':sensitivity,
            'threshold':'95% episode-cluster bootstrap, 5,000 replicates; seed 4101',
            'selection_note':'each Expert evaluated at its own Dev-selected checkpoint and t0; comparisons are not same-t0 causal contrasts',
            'decision':{k:{'all':classify_diff(pairs[f'{k}_all']),'hard':classify_diff(pairs[f'{k}_hard'])} for k in ('C1','C2','C3')}}
    np.savez_compressed(out/'predictions_and_errors.npz',row_ids=q,target=target,hard_mask=hard,episode_cluster=episode,
                        raw_drafts=raw,decoded_drafts=dec,**{f'pred_{k.replace("/","_").replace("=","_")}':v for k,v in preds.items()},
                        **{f'mse_{k.replace("/","_").replace("=","_")}':v for k,v in mse.items()})
    write_json(out/'summary.json',result)
    write_json(out/'decision.json',{'C1':result['decision']['C1'],'C2':result['decision']['C2'],'C3':result['decision']['C3'],
                                    'selected':{'E-M2':specs['E-M2'],'E-b12':specs['E-b12']}})


def main():
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['prepare','train','evaluate'])
    p.add_argument('--output',type=Path,default=OUT); p.add_argument('--steps',type=int,default=4000)
    p.add_argument('--batch',type=int,default=8); p.add_argument('--save-every',type=int,default=500); p.add_argument('--threads',type=int,default=8)
    a=p.parse_args(); a.output=a.output.resolve()
    if not 1<=a.threads<=16: raise ValueError('threads must be 1..16')
    if a.stage=='prepare': prepare(a.output)
    elif a.stage=='train': train(a.output,steps=a.steps,batch=a.batch,save_every=a.save_every,threads=a.threads)
    else: evaluate(a.output,batch=a.batch,threads=a.threads)

if __name__=='__main__': main()
