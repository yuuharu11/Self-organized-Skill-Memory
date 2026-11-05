"""Offline A-E memory-gap comparison for the frozen pi05-memory correction expert."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
PI05 = HERE.parent
REPO = HERE.parents[4]
import os as _os
_REPO = Path(__file__).resolve().parents[6]
_STATE_DIR = Path(_os.environ.get('SKILLNET_STATE_DIR', _REPO / '.skillnet-state'))
_PI05_ROOT = Path(_os.environ.get('SKILLNET_PI05_ROOT', _STATE_DIR / 'outputs/memory-pilot/pi05-memory40-20261104'))
RUN = _PI05_ROOT / 'run-01'
AUDIT = _PI05_ROOT / 'score-audit-01'
OUT = _PI05_ROOT / 'memory-gap-01'
OFFICIAL = _STATE_DIR / 'checkpoints/openpi-assets/checkpoints/pi05_libero'
NOISE_SEEDS = (4101, 4102)
T0S = (0.6, 0.4, 0.2)
BATCH = 4
sys.path.insert(0, str(PI05))


def _draft_run_module():
    import importlib.util
    p = PI05 / 'draft_sensitivity' / 'run.py'
    spec = importlib.util.spec_from_file_location('pi05_draft_sensitivity_run', p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_rows(run: Path):
    return {p.stem: np.load(p, mmap_mode='r', allow_pickle=False) for p in (run / 'rows').glob('*.npy')}


def _episode_ids(rows: dict, row_ids: np.ndarray):
    return np.asarray([f"{int(t)}:{int(e)}" for t, e in zip(
        np.asarray(rows['task'])[row_ids], np.asarray(rows['episode'])[row_ids], strict=True)])


def _dev_hard_threshold(run: Path, rows: dict):
    z = np.load(run / 'calibration_errors.npz', allow_pickle=False)
    dev = z['is_dev']
    ids = z['row_ids'][dev]
    # Column 0 is normal pi05 active7 MSE, averaged across the two fixed noise seeds.
    err = np.asarray(z['errors'][dev, 0], np.float64)
    clusters = _episode_ids(rows, ids)
    episode_err = np.asarray([err[clusters == ep].mean() for ep in np.unique(clusters)])
    return float(np.quantile(episode_err, .8)), {
        'dev_rows': int(len(ids)), 'dev_episodes': int(len(episode_err)),
        'dev_episode_normal_mse_q80': float(np.quantile(episode_err, .8)),
        'dev_episode_normal_mse_median': float(np.median(episode_err)),
        'dev_source': str(run / 'calibration_errors.npz'),
        'dev_error_column': 'normal pi05, active7 MSE, mean over noise seeds 4101/4102',
    }


def _candidate_selection(run: Path, audit: Path, rows: dict, query_rows: np.ndarray):
    oracle = np.load(audit / 'oracle_action_mse.npy', mmap_mode='r', allow_pickle=False)
    om = np.load(audit / 'oracle_meta.npz', allow_pickle=False)
    qm = np.load(audit / 'query_meta.npz', allow_pickle=False)
    if not np.array_equal(om['query_rows'], qm['query_rows']):
        raise AssertionError('oracle_meta/query_meta query row order differs')
    if not np.array_equal(om['query_task'], qm['task']) or not np.array_equal(om['query_episode'], qm['episode']):
        raise AssertionError('oracle_meta/query_meta task or episode metadata differs')
    if oracle.shape != (len(om['query_rows']), len(om['memory_rows'])):
        raise AssertionError(f'oracle matrix/meta shape mismatch: {oracle.shape}')
    qmap = {int(row): i for i, row in enumerate(om['query_rows'])}
    missing = [int(i) for i in query_rows if int(i) not in qmap]
    if missing:
        raise ValueError(f'{len(missing)} Eval rows absent from oracle matrix; examples {missing[:5]}')
    qix = np.asarray([qmap[int(i)] for i in query_rows], np.int64)
    mem_rows = np.asarray(om['memory_rows'], np.int64)
    qt = np.asarray(rows['task'])[query_rows]
    qe = np.asarray(rows['episode'])[query_rows]
    mt = np.asarray(rows['task'])[mem_rows]
    me = np.asarray(rows['episode'])[mem_rows]
    selections = {}
    selected_mse = {}
    for exclusion in ('episode', 'task'):
        chosen = np.empty(len(query_rows), np.int64)
        values = np.empty(len(query_rows), np.float32)
        for j, (qi, task, episode) in enumerate(zip(qix, qt, qe, strict=True)):
            allowed = (mt != task) if exclusion == 'task' else ~((mt == task) & (me == episode))
            if not allowed.any():
                raise ValueError(f'empty oracle candidate set for row {query_rows[j]} exclusion={exclusion}')
            row = np.asarray(oracle[qi], np.float32)
            local = int(np.argmin(row[allowed]))
            allowed_cols = np.flatnonzero(allowed)
            col = int(allowed_cols[local])
            # Explicitly assert that selected candidate is the permitted oracle-table minimum.
            assert np.isclose(row[col], row[allowed_cols].min(), rtol=0, atol=0)
            chosen[j] = mem_rows[col]
            values[j] = row[col]
        selections[exclusion] = chosen
        selected_mse[exclusion] = values
    return selections, selected_mse, qix, mem_rows


def _make_drafts(run: Path, audit: Path, rows: dict, ids: np.ndarray):
    from run import restore_codec
    mod = _draft_run_module()
    base_drafts, _, shuffle_src = mod.make_drafts(run, rows, ids)
    codec = restore_codec(run)
    pooled = np.asarray(np.load(run / 'pooled.npy', mmap_mode='r', allow_pickle=False)[ids], np.float32)
    state = np.asarray(rows['state'][ids], np.float32)
    actions = np.asarray(rows['actions'][ids], np.float32)
    residual = np.asarray(np.load(run / 'predicted_residual.npy', mmap_mode='r', allow_pickle=False)[ids], np.float32)
    b_gt, _ = codec.encode(actions, state)
    e_draft = np.asarray(codec.decode(b_gt, residual, pooled, state), np.float32)
    chosen, oracle_values, qix, mem_rows = _candidate_selection(run, audit, rows, ids)
    bases = np.load(run / 'bases.npy', mmap_mode='r', allow_pickle=False)
    drafts = dict(base_drafts)
    for exclusion in ('episode', 'task'):
        cand = chosen[exclusion]
        d = codec.decode(np.asarray(bases[cand], np.float32), residual, pooled, state)
        drafts[f'D_{exclusion}'] = np.asarray(d, np.float32)
        # Preserve exactly the action row's [10,32] values, including the original padding channels.
        drafts[f'Draw_{exclusion}'] = np.asarray(rows['actions'][cand], np.float32)
    drafts['E'] = e_draft
    for name, d in drafts.items():
        if d.shape != (len(ids), 10, 32) or not np.isfinite(d).all():
            raise ValueError(f'invalid draft {name}: {d.shape}')
    return drafts, chosen, oracle_values, shuffle_src


def _noise(jax, seed: int, row_id: int):
    return np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(seed), int(row_id)), (10, 32)), np.float32)


def _gpu_used_mib():
    import subprocess
    try:
        s = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                    text=True, timeout=5).splitlines()[0]
        return int(s.strip())
    except Exception:
        return None


def _infer(run: Path, ids: np.ndarray, rows: dict, drafts: dict, *, output: Path):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from openpi.shared.nnx_utils import PathRegex, module_jit
    from run import load_policy, obs_batch

    best = json.loads((run / 'best_bridge.json').read_text())
    best_path = Path(best['path'])
    if best_path != run / 'bridge/001500' or int(best['steps']) != 1500:
        raise ValueError(f'wrong selected expert checkpoint: {best}')
    base, _ = load_policy(OFFICIAL)
    baseline = module_jit(base.sample_actions, static_argnames=('num_steps',))
    filt = nnx.All(nnx.Param, PathRegex(r'.*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*'))
    # Verify exact normal route after overlay on an initial fixed 16-row probe.
    probe_ids = ids[:16]
    probe_obs = obs_batch(rows, probe_ids)
    probe_eps = jnp.asarray(np.stack([_noise(jax, NOISE_SEEDS[0], i) for i in probe_ids]))
    pre = np.asarray(baseline(jax.random.key(0), probe_obs, num_steps=10, noise=probe_eps))
    with ocp.PyTreeCheckpointer() as ckpt:
        partial = ckpt.restore(best_path)
    state = nnx.state(base, filt)
    state.replace_by_pure_dict(jax.tree.map(jnp.asarray, partial))
    nnx.update(base, state)
    post = np.asarray(baseline(jax.random.key(0), probe_obs, num_steps=10, noise=probe_eps))
    fallback_equal = bool(np.array_equal(pre, post))
    if not fallback_equal:
        raise AssertionError('overlay changed normal fallback output')
    bridge = module_jit(base.sample_bridge, static_argnames=('num_steps',))

    # One baseline and all draft/t0 bridges. Shared epsilon and batch bridge key across each row's conditions.
    specs = [('A_normal', None, None)]
    for t0 in T0S:
        for draft_name in ('zero', 'vlm_retrieved', 'D_episode', 'D_task',
                           'Draw_episode', 'Draw_task', 'E'):
            specs.append((f'{draft_name}_t0_{t0:.1f}', draft_name, t0))
    names = [s[0] for s in specs]
    preds = np.empty((len(specs), len(NOISE_SEEDS), len(ids), 10, 7), np.float32)
    started = time.monotonic(); batch_num = 0
    for ni, seed in enumerate(NOISE_SEEDS):
        for start in range(0, len(ids), BATCH):
            end = min(start + BATCH, len(ids)); batch_ids = ids[start:end]
            obs = obs_batch(rows, batch_ids)
            eps = jnp.asarray(np.stack([_noise(jax, seed, i) for i in batch_ids]))
            preds[0, ni, start:end] = np.asarray(baseline(jax.random.key(0), obs, num_steps=10, noise=eps))[:, :, :7]
            key = jax.random.fold_in(jax.random.key(seed), start)
            for ci, (_, dname, t0) in enumerate(specs[1:], 1):
                p = bridge(key, obs, jnp.asarray(drafts[dname][start:end]),
                           jnp.full((end-start,), t0), num_steps=10, noise=eps)
                preds[ci, ni, start:end] = np.asarray(p)[:, :, :7]
            batch_num += 1
            if batch_num % 20 == 0 or end == len(ids):
                with (output / 'progress.jsonl').open('a') as f:
                    f.write(json.dumps({'event':'infer','seed':seed,'rows_done':end,'rows_total':len(ids),
                                        'batches_done':batch_num,'elapsed_s':time.monotonic()-started,
                                        'gpu_used_mib':_gpu_used_mib()})+'\n')
    return names, preds, fallback_equal, time.monotonic()-started


def _bootstrap_delta(a, b, ep_clusters, selected, *, seed=20261104, reps=5000):
    idx = np.flatnonzero(selected)
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    cl = np.asarray(ep_clusters)[idx]
    blocks = [idx[cl == e] for e in np.unique(cl)]
    if not blocks:
        return [float('nan'), float('nan')]
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, np.float64)
    for k in range(reps):
        sample = rng.integers(0, len(blocks), size=len(blocks))
        rows = np.concatenate([blocks[i] for i in sample])
        draws[k] = d[rows].mean()
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def _summarize(names, preds, actions, drafts, ids, rows, threshold):
    errors = np.mean((preds - actions[None, None]) ** 2, axis=(-1, -2))  # [condition,noise,row], active7
    rowerr = errors.mean(axis=1)
    baseline_ix = names.index('A_normal')
    baseline = rowerr[baseline_ix]
    ep = _episode_ids(rows, ids)
    hard = baseline >= threshold
    n_hard_top5 = max(1, int(np.ceil(hard.sum() * .05)))
    hard_top5 = np.zeros(len(ids), bool)
    hard_inds = np.flatnonzero(hard)
    hard_top5[hard_inds[np.argsort(baseline[hard_inds])[-n_hard_top5:]]] = True
    masks = {'all': np.ones(len(ids), bool), 'hard': hard, 'hard_top5': hard_top5}
    results = {'threshold':float(threshold),'hard_rows':int(hard.sum()),'hard_top5_rows':int(hard_top5.sum()),
               'masks':{},'conditions':{},'G':{}}
    for group, mask in masks.items():
        results['masks'][group] = {'rows':int(mask.sum()),'episodes':int(len(np.unique(ep[mask]))) if mask.any() else 0}
        results['conditions'][group] = {}
        if not mask.any():
            continue
        for ci, name in enumerate(names):
            v = rowerr[ci]
            selected = v[mask]
            item={'mean_mse':float(selected.mean()),'median_mse':float(np.median(selected)),'rows':int(mask.sum())}
            if name != 'A_normal' and mask.any():
                b = rowerr[names.index(f'zero_t0_{name.rsplit("_t0_",1)[1]}')]
                # Difference vs zero draft (B), paired at row and episode level.
                item['delta_vs_B_mean']=float((v-b)[mask].mean())
                item['delta_vs_B_ci95']=_bootstrap_delta(v,b,ep,mask)
                dname = name.rsplit('_t0_',1)[0]
                if dname in drafts:
                    draft_err=np.mean((drafts[dname][:,:,:7]-actions)**2,axis=(1,2))
                    item['draft_mse_mean']=float(draft_err[mask].mean())
            results['conditions'][group][name]=item
        if not mask.any():
            continue
        for t0 in T0S:
            bname=f'zero_t0_{t0:.1f}'; ename=f'E_t0_{t0:.1f}'
            bmean=results['conditions'][group][bname]['mean_mse']
            emean=results['conditions'][group][ename]['mean_mse']
            denom=bmean-emean
            for draft in ('D_episode','D_task','Draw_episode','Draw_task'):
                cname=f'{draft}_t0_{t0:.1f}'; x=results['conditions'][group][cname]['mean_mse']
                results['G'].setdefault(group,{}).setdefault(draft,{})[f't0_{t0:.1f}']={
                    'G':float((bmean-x)/denom) if abs(denom)>1e-12 else None,
                    'B_mse':bmean,'X_mse':x,'E_mse':emean}
    return results, errors


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,default=RUN); ap.add_argument('--audit',type=Path,default=AUDIT)
    ap.add_argument('--output',type=Path,default=OUT); ap.add_argument('--batch-size',type=int,default=BATCH)
    ap.add_argument('--row-limit',type=int,default=None)
    a=ap.parse_args(); out=a.output
    if a.batch_size != BATCH: raise ValueError('batch size is fixed to 4 for parity with calibrate.py')
    out.mkdir(parents=True,exist_ok=True)
    for name in ('protocol.json','predictions_and_errors.npz','summary.json','summary_corrected.json'):
        if (out/name).exists(): raise FileExistsError(out/name)
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION','.22')
    mod=_draft_run_module(); rows=_load_rows(a.run)
    ids_all=np.flatnonzero(rows['validation'])
    # Reuse exact eval episode split from the prior run helper.
    _, eval_eps=mod.calibration_episodes(rows)
    ids=np.flatnonzero(rows['validation'] & np.isin(rows['episode'],eval_eps))
    if len(ids)!=1318: raise AssertionError(f'expected 1318 Eval rows; got {len(ids)}')
    if a.row_limit is not None:
        if a.row_limit < 1: raise ValueError('--row-limit must be positive')
        # Spread smoke rows across tasks so the cross-task shuffle is defined.
        chosen=[]
        per_task=max(1,int(np.ceil(a.row_limit / len(np.unique(rows['task'][ids])))))
        for task in np.unique(rows['task'][ids]):
            chosen.extend(ids[np.asarray(rows['task'])[ids]==task][:per_task].tolist())
        ids=np.asarray(sorted(chosen[:a.row_limit]),np.int64)
    threshold, dev_meta=_dev_hard_threshold(a.run,rows)
    drafts,chosen,oracle_values,shuffle_src=_make_drafts(a.run,a.audit,rows,ids)
    protocol={'source_run':str(a.run),'audit_dir':str(a.audit),'best_bridge':json.loads((a.run/'best_bridge.json').read_text()),
              'official_checkpoint':str(OFFICIAL),'rows':len(ids),'tasks':int(len(np.unique(rows['task'][ids]))),
              'episodes':int(len(np.unique(_episode_ids(rows,ids)))),'row_ids':ids.tolist(),
              'noise_seeds':list(NOISE_SEEDS),'noise_rule':'jax.random.normal(fold_in(key(seed),row_id), [10,32])',
              'bridge_key_rule':'fold_in(key(seed), batch_start), batch_size=4, 10 steps',
              't0_values':list(T0S),'dev_threshold':dev_meta,'hard_rule':'Eval baseline row MSE (averaged over two noise seeds) >= Dev episode-mean q80',
              'hard_top5_rule':'top ceil(5% of hard rows) by Eval normal baseline error',
              'candidate_exclusions':{'episode':'exclude matching (task,episode)','task':'exclude matching task'},
              'query_rows_oracle_index': 'mapped by oracle_meta.query_rows and verified against query_meta order',
              'candidate_rows':{k:v.tolist() for k,v in chosen.items()},'candidate_oracle_mse':{k:v.tolist() for k,v in oracle_values.items()},
              'shuffle_source_rows':shuffle_src.tolist(),'metric':'active7 normalized action MSE averaged over 10x7',
              'target_constraints':'D/E are oracle diagnostics; not executable method results',
              'mem_fraction':os.getenv('XLA_PYTHON_CLIENT_MEM_FRACTION')}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    with (out/'progress.jsonl').open('w') as f: f.write(json.dumps({'event':'prepared','rows':len(ids),'hard_threshold':threshold})+'\n')
    names,preds,fallback_equal,inference_s=_infer(a.run,ids,rows,drafts,output=out)
    actions=np.asarray(rows['actions'][ids,:,:7],np.float32)
    result,err=_summarize(names,preds,actions,drafts,ids,rows,threshold)
    # Exact parity checks against completed draft-sensitivity full Eval output.
    prior=_PI05_ROOT / 'draft-sensitivity-01'
    prev=np.load(prior/'predictions_and_errors.npz',allow_pickle=False)
    prev_names=prev['condition_names'].tolist()
    prev_map={int(row):i for i,row in enumerate(prev['row_ids'])}
    try: prev_ix=np.asarray([prev_map[int(row)] for row in ids],np.int64)
    except KeyError as exc: raise AssertionError(f'row absent from draft-sensitivity: {exc}') from exc
    parity={}
    for ours,theirs in [('A_normal','normal'),('vlm_retrieved_t0_0.6','vlm_retrieved_t0_0.6')]:
        x=err[names.index(ours)].mean(0); y=prev['row_mse'][prev_names.index(theirs)][:,prev_ix].mean(0)
        diff=float(np.max(np.abs(x-y)))
        parity[ours]={'reference':theirs,'max_abs_row_mse_difference':diff,'allclose_atol_1e-6':bool(np.allclose(x,y,atol=1e-6,rtol=0))}
        if not parity[ours]['allclose_atol_1e-6']: raise AssertionError(f'{ours} parity failed: {diff}')
    result.update({'rows':len(ids),'tasks':int(len(np.unique(rows['task'][ids]))),'episodes':int(len(np.unique(_episode_ids(rows,ids)))),
                   'smoke_row_limit':a.row_limit,
                   'fallback_identical_before_after_overlay':fallback_equal,'inference_seconds':inference_s,
                   'parity_with_draft_sensitivity':parity,'oracle_candidate_assertion':True})
    # D selection values are already asserted against every allowed table row in _candidate_selection.
    np.savez_compressed(out/'predictions_and_errors.npz',row_ids=ids,task_ids=np.asarray(rows['task'])[ids],
                        episode_ids=np.asarray(rows['episode'])[ids],condition_names=np.asarray(names),
                        noise_seeds=np.asarray(NOISE_SEEDS),actions=actions,predictions=preds,row_mse=err,
                        draft_names=np.asarray(list(drafts)),drafts=np.stack(list(drafts.values())),
                        chosen_candidate_rows=np.stack([chosen['episode'],chosen['task']]),
                        chosen_oracle_mse=np.stack([oracle_values['episode'],oracle_values['task']]))
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    # Downstream stages (skill-memories, local-key, key-mechanism) read the hard-row
    # threshold from summary_corrected.json; it carries the same evaluated result.
    (out/'summary_corrected.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'rows':len(ids),'conditions':len(names),'inference_seconds':inference_s,
                      'parity':parity,'hard_threshold':threshold,'hard_rows':result['hard_rows']},indent=2))

if __name__=='__main__': main()
