"""EVT1: frozen-query transfer diagnostics; Eval actions are scoring-only.

No training or model/hyperparameter selection is performed. Existing LIB1
metrics are reproduced before two prespecified scoring sensitivities: a common
scale-only metric and removal of cross-suite integer-ID candidate exclusion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scene_invariance as s1
import scene_invariance_v2 as v2
import scale_only_query as sq
import theta_regression as tr
import variable_shape_memory as vs

ROOT = Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero')
OUT = ROOT / 'eval-transfer-01'
INV = ROOT / 'scene-invariance-01/step2/attempt-fix2'
SEEDS = (4101, 4102, 4103)
MODELS = ('R5', 'R0', 'P5')


def plain(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, Path): return str(x)
    raise TypeError(type(x).__name__)


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=plain, allow_nan=False) + '\n')
    tmp.replace(path)


def log(out, stage, **extra):
    obj = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'stage': stage, **extra}
    line = json.dumps(obj, ensure_ascii=False, default=plain)
    print(line, flush=True)
    with (out / 'progress.log').open('a') as f: f.write(line + '\n')


def load_npz(path):
    with np.load(path, allow_pickle=False) as z: return {k: z[k] for k in z.files}


def checkpoints(name):
    if name == 'R5':
        return {4101: INV / 'ablation_tuning/R5_from_R4_k5_w0.1.pt',
                **{seed: INV / f'noise_history_models/R5_seed{seed}.pt' for seed in SEEDS[1:]}}
    if name == 'R0':
        return {4101: INV / 'ablation_tuning/R0.pt',
                **{seed: INV / f'noise_history_models/R0_seed{seed}.pt' for seed in SEEDS[1:]}}
    return {seed: INV / f'noise_history_models/P5_seed{seed}.pt' for seed in SEEDS}


def score_embeddings(net, name, ds, pool, *, device, zero=False, batch=32):
    import torch
    from torch.nn import functional as F
    scores, vectors, raw_norm = [], [], []
    net.eval()
    with torch.no_grad():
        if name == 'R5':
            pa = torch.as_tensor(pool, device=device)
            keys = F.normalize(torch.cat([net.encode_keys(pa[i:i+256]) for i in range(0, len(pa), 256)]), dim=-1)
        else:
            keys = F.normalize(net.key(torch.as_tensor(pool.reshape(len(pool), -1), device=device)), dim=-1)
        for start in range(0, len(ds['task']), batch):
            sl = slice(start, start + batch)
            hist = torch.as_tensor(ds['history'][sl], device=device, dtype=torch.float32)
            if zero: hist = torch.zeros_like(hist)
            state = torch.as_tensor(ds['state'][sl], device=device, dtype=torch.float32)
            tok = None
            if name != 'P5':
                tok = torch.as_tensor(ds['tokens'][sl], device=device, dtype=torch.float32)
            if name == 'R5':
                q, _ = net.encode_query(tok, hist, state, token_groups=torch.as_tensor(ds['groups'], device=device))
                sc = net.scores(q, keys)
                vec = q.flatten(1) / np.sqrt(q.shape[1])
            else:
                if tok is not None: tok = tok[:, ds['groups'] == 1]
                mask = None if tok is None else torch.ones(tok.shape[:2], device=device, dtype=torch.bool)
                q, _ = net(hist, state, tok, mask)
                vec = F.normalize(q, dim=-1)
                sc = vec @ keys.T * net.log_scale.exp()
            raw_norm.append(q.norm(dim=-1).cpu().numpy().reshape(len(q), -1).mean(1))
            scores.append(sc.cpu().numpy())
            vectors.append(vec.cpu().numpy())
    return np.concatenate(scores), np.concatenate(vectors), np.concatenate(raw_norm)


def get_tokens(path, groups_path, row_ids, out, label):
    """One host gather, retaining exactly LIB1's first-row valid-group convention."""
    mem = np.load(path, mmap_mode='r')
    g = np.asarray(np.load(groups_path, mmap_mode='r')[0])
    cols = np.flatnonzero(g >= 0)
    vals = np.empty((len(row_ids), len(cols), mem.shape[-1]), np.float16)
    wrist_norm, pooled = [], []
    for start in range(0, len(row_ids), 32):
        ids = row_ids[start:start+32]
        vals[start:start+len(ids)] = mem[ids][:, cols]
        wrist = vals[start:start+len(ids), g[cols] == 1].astype(np.float32)
        wrist_norm.append(np.linalg.norm(wrist, axis=-1).mean(1))
        pooled.append(wrist.mean(1))
    log(out, 'tokens_loaded', split=label, rows=len(row_ids), shape=vals.shape)
    return vals, g[cols], np.concatenate(wrist_norm), np.concatenate(pooled)


def row_metrics(scores, errors, allowed):
    """Positive Spearman, original clipped recovery and top-k diagnostics."""
    s = np.where(allowed, scores, -np.inf)
    order = np.argsort(-s, axis=1, kind='stable')
    n, m = errors.shape
    ar = np.arange(n)
    oracle = np.min(np.where(allowed, errors, np.inf), axis=1)
    rand = np.sum(np.where(allowed, errors, 0), axis=1) / allowed.sum(1)
    top = errors[ar, order[:, 0]]
    raw_rec = np.divide(rand-top, rand-oracle, out=np.zeros(n), where=(rand-oracle)>1e-8)
    result = {'top1_error': top, 'oracle': oracle, 'random': rand,
              'recovery': np.clip(raw_rec, 0, 1), 'unclipped_recovery': raw_rec,
              'spearman': np.zeros(n)}
    # Average tie ranks, just as the original LIB1 implementation.
    groups = {}
    for i, a in enumerate(allowed): groups.setdefault(a.tobytes(), []).append(i)
    for ids0 in groups.values():
        ids = np.asarray(ids0)
        cols = np.flatnonzero(allowed[ids[0]])
        a = rankdata(scores[ids][:, cols], axis=1)
        b = rankdata(-errors[ids][:, cols], axis=1)
        a -= a.mean(1, keepdims=True); b -= b.mean(1, keepdims=True)
        den = np.sqrt((a*a).sum(1)*(b*b).sum(1))
        result['spearman'][ids] = np.divide((a*b).sum(1), den, out=np.zeros(len(ids)), where=den>0)
    good = np.zeros_like(allowed)
    for i in range(n):
        cols = np.flatnonzero(allowed[i])
        # All ties at the 10th percentile are included; chance is measured explicitly.
        threshold = np.quantile(errors[i, cols], .10)
        good[i] = allowed[i] & (errors[i] <= threshold)
    for k in (1, 5, 20):
        chosen = order[:, :k]
        ek = np.take_along_axis(errors, chosen, axis=1)
        hits = np.take_along_axis(good, chosen, axis=1).sum(1)
        result[f'top{k}_best_error'] = ek.min(1)
        result[f'top{k}_mean_error'] = ek.mean(1)
        result[f'top{k}_bottom10_recall'] = hits / good.sum(1)
        result[f'top{k}_bottom10_hit'] = (hits > 0).astype(float)
        result[f'top{k}_chance_recall'] = k / allowed.sum(1)
    return result


def aggregate(metrics, mask=None):
    if mask is None: mask = np.ones(len(next(iter(metrics.values()))), bool)
    return {'rows': int(mask.sum()), **{k: float(np.mean(v[mask])) for k, v in metrics.items()}}


def summary(values):
    x = np.asarray(values, np.float64)
    return {'mean': float(x.mean()), 'sd': float(x.std()),
            'q10': float(np.quantile(x, .1)), 'q50': float(np.quantile(x, .5)), 'q90': float(np.quantile(x, .9)),
            'q95': float(np.quantile(x, .95))}


def landscape(errors, allowed, energy):
    oracle = np.min(np.where(allowed, errors, np.inf), axis=1)
    rand = np.sum(np.where(allowed, errors, 0), axis=1)/allowed.sum(1)
    stdev = np.sqrt(np.sum(np.where(allowed, (errors-rand[:, None])**2, 0), axis=1)/allowed.sum(1))
    qs = np.array([np.quantile(e[a], [.1,.5,.9]) for e, a in zip(errors, allowed)])
    return {'rows': len(errors), 'oracle': summary(oracle), 'random': summary(rand),
            'gap': summary(rand-oracle), 'relative_gap': summary((rand-oracle)/np.maximum(rand, 1e-12)),
            'row_error_cv': summary(stdev/np.maximum(rand, 1e-12)),
            'oracle_over_query_energy': summary(oracle/np.maximum(energy, 1e-12)),
            'random_over_query_energy': summary(rand/np.maximum(energy, 1e-12)),
            'relative_p90_p10_width': summary((qs[:,2]-qs[:,0])/np.maximum(qs[:,1],1e-12)),
            'near_oracle_1pct_gap_fraction': float(np.mean(np.sum(allowed & (errors <= oracle[:,None]+.01*(rand-oracle)[:,None]), axis=1)/allowed.sum(1)))}


def provenance(errors, allowed, pool, names):
    pick = np.argmin(np.where(allowed, errors, np.inf), axis=1)
    tasks, units = np.asarray(pool['source_task']), np.asarray(pool['source_unit'])
    def counts(x):
        ids, c = np.unique(x, return_counts=True)
        return [{'id': int(i), 'count': int(n), 'fraction': float(n/len(x)),
                 **({'name': names.get(int(i), '')} if x is not None and np.array_equal(x, tasks[pick]) else {})}
                for i, n in sorted(zip(ids, c), key=lambda p: -p[1])]
    return {'oracle_candidate': pick.tolist(), 'task_counts': counts(tasks[pick]),
            'unit_counts': counts(units[pick]),
            'candidate_counts': counts(pick),
            'ties_within_1e_8_mean': float(np.mean(np.sum(allowed & (errors <= errors[np.arange(len(errors)),pick,None]+1e-8), axis=1)))}


def concentration(scores, allowed, tasks, pool):
    rank = np.argsort(-np.where(allowed, scores, -np.inf), axis=1, kind='stable')
    out = {}
    rng = np.random.default_rng(4101)
    a = rng.integers(len(rank), size=10000); b = rng.integers(len(rank), size=10000)
    valid = a != b; a, b = a[valid], b[valid]
    cross = tasks[a] != tasks[b]
    for k in (1, 10):
        ids = rank[:, :k]
        cnt = np.bincount(ids.flatten(), minlength=scores.shape[1])
        p = cnt[cnt>0]/cnt.sum()
        ov = np.array([len(np.intersect1d(ids[i], ids[j], assume_unique=True)) for i,j in zip(a,b)])
        top = np.argsort(-cnt)[:10]
        # Uniform allowed-candidate expected intersection, accounting for the masks.
        chance = np.array([k*k*np.sum(allowed[i]&allowed[j])/(allowed[i].sum()*allowed[j].sum()) for i,j in zip(a[:1000],b[:1000])])
        out[f'top{k}'] = {'unique_candidates': int((cnt>0).sum()), 'max_fraction_of_queries': float(cnt.max()/len(scores)),
                         'effective_candidates_entropy': float(np.exp(-np.sum(p*np.log(p)))),
                         'mean_intersection': float(ov.mean()), 'mean_jaccard': float(np.mean(ov/(2*k-ov))),
                         'cross_task_mean_intersection': float(ov[cross].mean()),
                         'uniform_allowed_expected_intersection': float(chance.mean()),
                         'most_selected': [{'candidate': int(j), 'query_fraction': float(cnt[j]/len(scores)),
                                            'source_task': int(pool['source_task'][j]), 'source_unit': int(pool['source_unit'][j])} for j in top]}
    return out


def ood_measure(ref, hold, ev):
    """Reference-only shrinkage covariance; query coordinates match the score dot product."""
    from sklearn.covariance import LedoitWolf
    from scipy.spatial.distance import cdist
    cov = LedoitWolf().fit(np.asarray(ref, np.float64))
    out = {'reference_rows': len(ref), 'dimension': ref.shape[1], 'shrinkage': float(cov.shrinkage_)}
    for label, x in [('lib40_holdout', hold), ('lib90_eval', ev)]:
        distances = np.empty(len(x))
        for st in range(0, len(x), 256):
            distances[st:st+256] = cdist(x[st:st+256], ref, metric='euclidean').min(1)
        md = np.sqrt(np.maximum(cov.mahalanobis(x), 0))
        out[label] = {'nearest_unit_sphere_euclidean': summary(distances), 'mahalanobis_shrinkage': summary(md)}
        if label == 'lib40_holdout': hold_nn, hold_md = distances, md
        else:
            out['eval_above_holdout_q95_fraction'] = {
                'nearest_distance': float(np.mean(distances>np.quantile(hold_nn,.95))),
                'mahalanobis': float(np.mean(md>np.quantile(hold_md,.95)))}
    return out


def bootstrap_diff(x, y, episodes, reps=3000):
    _, inv = np.unique(episodes, return_inverse=True)
    count = np.bincount(inv)
    sums = np.bincount(inv, weights=np.asarray(x)-np.asarray(y))
    draw = np.random.default_rng(4101).integers(len(count), size=(reps, len(count)))
    vals = sums[draw].sum(1)/count[draw].sum(1)
    return {'mean': float(np.mean(np.asarray(x)-np.asarray(y))), 'ci95': np.quantile(vals,[.025,.975]).tolist()}


def run(out=OUT, *, device='cuda', ref_rows=4096):
    import torch
    started = time.time()
    threads = int(os.environ.get('OMP_NUM_THREADS', '8'))
    torch.set_num_threads(threads)
    if device.startswith('cuda'): torch.cuda.set_per_process_memory_fraction(.10)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if (out/'results.json').exists(): raise FileExistsError('results.json already exists; no overwrite')
    log(out, 'start', device=device, threads=threads)
    train = load_npz(ROOT/'scene-invariance-01/fit_rows.npz')
    saved = load_npz(ROOT/'lib-closed-01/eval_rows.npz')
    inp = load_npz(s1.SOURCE/'prep/W_inputs_eval.npz')
    assert np.array_equal(inp['row'], saved['row_id'])
    assert np.array_equal(inp['task'], saved['task'])
    assert np.array_equal(inp['task']*100000+inp['episode'], saved['episode'])
    fit = np.flatnonzero(train['fit']); hold = np.flatnonzero(train['select'])
    rng = np.random.default_rng(4101)
    ref = np.sort(rng.choice(fit, min(ref_rows, len(fit)), replace=False))
    loadids = np.r_[ref, hold]
    t40, g40, n40, pooled40 = get_tokens(s1.TOKENS, s1.GROUPS, train['row_id'][loadids], out, 'lib40_ref_plus_holdout')
    t90, g90, n90, pooled90 = get_tokens(s1.UNSEEN_ROWS/'tokens.npy', s1.UNSEEN_ROWS/'groups.npy', saved['row_id'], out, 'lib90_eval')
    d40 = {'task': train['task'][loadids], 'state': train['state'][loadids], 'history': train['history'][loadids], 'tokens': t40, 'groups': g40}
    ev = {'task': saved['task'], 'state': inp['state'], 'history': inp['hist'], 'tokens': t90, 'groups': g90}
    assert set(train['task'][ref]).isdisjoint(set(train['task'][hold]))
    # Candidate provenance is reconstructed from training-only cached records.
    pool0 = tr.candidate_shape_pool(rep='C_vm_none')
    keep = ~np.isin(pool0['source_task'], s1.HOLDOUT)
    pool = {k: np.asarray(v)[keep] for k,v in pool0.items()}
    np.testing.assert_allclose(pool['shape'], train['pool_shape'], atol=1e-7)
    np.testing.assert_array_equal(pool['source_task'], train['pool_task'])
    wpool = load_npz(s1.SOURCE/'prep/W_pool.npz')
    np.testing.assert_allclose(wpool['canon'], pool['shape'][:,:,:6], atol=1e-7)
    np.testing.assert_array_equal(wpool['task'], pool['source_task'])
    names40 = s1._task_names()
    # Task names without importing simulator modules.
    import ast
    path = HERE.parents[2]/'third_party/libero/libero/libero/benchmark/libero_suite_task_map.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='libero_task_map' for t in n.targets))
    names90 = dict(enumerate(ast.literal_eval(node.value)['libero_90']))
    phys90 = np.asarray(vs.to_physical(saved['target']), np.float32)
    # The existing 40 labels are scale-only on decoded source-oriented candidates.
    err_ss = sq.scale_only_error(phys90, wpool['phys'], 'S-shared')
    sample40 = sq.scale_only_error(train['target'][hold[:32]], wpool['phys'], 'S-shared')
    np.testing.assert_allclose(sample40, train['error'][hold[:32]], rtol=2e-4, atol=2e-6)
    log(out, 'common_error_computed', max_l40_error_verification=float(np.abs(sample40-train['error'][hold[:32]]).max()))
    a40 = train['allowed'][hold]
    a90 = saved['allowed']; all90 = np.ones_like(a90, bool)
    specs = {'L40-Sshared': (train['error'][hold], a40),
             'L90-original-Rso3': (saved['errors'], a90),
             'L90-Sshared': (err_ss, a90), 'L90-Sshared-all': (err_ss, all90),
             'L90-Rso3-all': (saved['errors'], all90)}
    energy40 = np.mean(train['target'][hold]**2, axis=(1,2))
    energy90 = np.mean(phys90**2, axis=(1,2))
    results = {'status': 'running', 'protocol': {'model_selection': 'none; existing frozen 3-seed checkpoints only',
                'reference_seed': 4101, 'reference_fit_rows': len(ref), 'lib40_holdout_rows': len(hold),
                'lib90_eval_rows': len(saved['task']), 'hard_rows': int(saved['hard'].sum()),
                'good_candidates': 'error <= per-row allowed-candidate 10th percentile; include ties',
                'recovery': 'mean clip((random-top1)/(random-oracle),0,1); raw also provided',
                'topk_error': 'minimum error within retrieved k, and mean of all retrieved k, both provided',
                'ood_reference': 'fixed sample of fit rows only; shrinkage covariance fitted to embeddings, no Eval labels',
                'no_training': True, 'gpu_fraction_cap': .10, 'threads': threads,
                'cpu_affinity': sorted(os.sched_getaffinity(0))},
              'landscapes': {}, 'oracle_provenance': {}, 'models': {}, 'task_names_40': names40, 'task_names_90': names90,
              'candidate_mask_audit': {'excluded_per_eval_row': summary((~a90).sum(1)),
                 'rows_affected': int(np.any(~a90, axis=1).sum()),
                 'numeric_id_equality_cross_suite': True,
                 'all_combinations_equal_instruction': None}}
    collisions=[]
    for task in np.unique(ev['task']):
        ids = np.flatnonzero((pool['source_task']==task))
        if len(ids): collisions.append({'lib90_task':int(task),'lib90_name':names90[int(task)],
                                        'lib40_name':names40[int(task)], 'removed_candidates':len(ids)})
    results['candidate_mask_audit']['collisions'] = collisions
    for label, (errors, allowed) in specs.items():
        energy = energy40 if label.startswith('L40') else energy90
        results['landscapes'][label] = landscape(errors, allowed, energy)
        results['oracle_provenance'][label] = provenance(errors, allowed, pool, names40)
    results['wrist_tokens'] = {'definition': 'mean over 256 wrist tokens of per-token L2 norm (2048 dims)',
        'fit_reference': summary(n40[:len(ref)]), 'lib40_holdout': summary(n40[len(ref):]), 'lib90_eval': summary(n90),
        'eval_above_holdout_q95': float(np.mean(n90>np.quantile(n40[len(ref):],.95)))}
    save(out/'data_protocol.json', results)
    np.savez_compressed(out/'diagnostic_labels.npz', lib40_holdout_indices=hold, fit_reference_indices=ref,
        eval_task=ev['task'], eval_episode=saved['episode'], eval_hard=saved['hard'],
        errors_original=saved['errors'], errors_Sshared=err_ss, allowed_original=a90,
        pool_task=pool['source_task'], pool_unit=pool['source_unit'], pool_source_row=pool['source_row'])
    all_seed_metrics = {}
    for name in MODELS:
        per = []
        for seed, cp in checkpoints(name).items():
            log(out, 'inference_start', model=name, seed=seed, checkpoint=str(cp))
            ck = torch.load(cp, map_location='cpu', weights_only=False)
            if name=='R5': net = v2._load_ablation_checkpoint(cp, v2.ablation_config('R5'), device)
            else:
                net = s1._make_model('P5' if name=='P5' else 'A', 1, 2048)
                net.load_state_dict(ck['state_dict']); net.to(device)
            del ck
            sc40, q40, qn40 = score_embeddings(net, name, d40, pool['shape'], device=device)
            sc90, q90, qn90 = score_embeddings(net, name, ev, pool['shape'], device=device)
            zero40, _, _ = score_embeddings(net, name, {**d40, 'task':d40['task'][len(ref):],
                'state':d40['state'][len(ref):], 'history':d40['history'][len(ref):], 'tokens':d40['tokens'][len(ref):]},
                pool['shape'], device=device, zero=True)
            zero90, _, _ = score_embeddings(net, name, ev, pool['shape'], device=device, zero=True)
            np.savez_compressed(out/f'{name}-s{seed}-inference.npz', lib40_scores=sc40[len(ref):], eval_scores=sc90,
                lib40_zero_scores=zero40, eval_zero_scores=zero90, fit_embedding=q40[:len(ref)],
                holdout_embedding=q40[len(ref):], eval_embedding=q90)
            record = {'seed':seed, 'checkpoint':str(cp), 'checkpoint_sha256': hashlib.sha256(cp.read_bytes()).hexdigest(),
                      'metrics':{}, 'task_metrics':{}, 'concentration':{}}
            for label, (errors, allowed) in specs.items():
                for hist in ('true','zero'):
                    sc = (sc40[len(ref):] if hist=='true' else zero40) if label.startswith('L40') else (sc90 if hist=='true' else zero90)
                    rows = row_metrics(sc, errors, allowed)
                    key = f'{label}|{hist}'
                    all_seed_metrics[name,seed,key] = rows
                    record['metrics'][key] = aggregate(rows)
                    if label.startswith('L90'):
                        record['metrics'][key]['hard'] = aggregate(rows, saved['hard'])
                        record['task_metrics'][key] = {str(int(t)):aggregate(rows,ev['task']==t) for t in np.unique(ev['task'])}
                    if label in ('L40-Sshared', 'L90-original-Rso3') and hist=='true':
                        tasks = train['task'][hold] if label.startswith('L40') else ev['task']
                        record['concentration'][label] = concentration(sc, allowed, tasks, pool)
            rep = all_seed_metrics[name,seed,'L90-original-Rso3|true']
            si = SEEDS.index(seed)
            record['reproduction'] = {k:float(np.max(np.abs(rep[k]-saved[f'{name}_{field}'][si])))
                for k,field in [('top1_error','row_type_error'),('recovery','row_recovery'),('spearman','row_rho')]}
            # Ranking ties may differ at numerical precision, but gross drift is an error.
            assert abs(rep['spearman'].mean()-saved[f'{name}_row_rho'][si].mean()) < 2e-4
            assert abs(rep['top1_error'].mean()-saved[f'{name}_row_type_error'][si].mean()) < 1e-5
            log(out, 'ood_start', model=name, seed=seed)
            record['ood'] = ood_measure(q40[:len(ref)], q40[len(ref):], q90)
            record['query_raw_norm'] = {'fit':summary(qn40[:len(ref)]),'holdout':summary(qn40[len(ref):]),'eval':summary(qn90)}
            per.append(record)
            save(out/f'{name}-s{seed}-diagnostics.json', record)
            del net
            if device.startswith('cuda'): torch.cuda.empty_cache()
            log(out, 'seed_complete', model=name, seed=seed, reproduction=record['reproduction'])
        keys = list(per[0]['metrics'])
        result = {'seeds':per, 'metrics':{}, 'task_metrics':{}}
        for key in keys:
            metrics = {k:np.mean([all_seed_metrics[name,seed,key][k] for seed in SEEDS],axis=0)
                       for k in all_seed_metrics[name,SEEDS[0],key]}
            result['metrics'][key] = aggregate(metrics)
            if key.startswith('L90'):
                result['metrics'][key]['hard'] = aggregate(metrics, saved['hard'])
                result['task_metrics'][key] = {str(int(t)):aggregate(metrics,ev['task']==t) for t in np.unique(ev['task'])}
            result['metrics'][key]['seed_sd_spearman'] = float(np.std([p['metrics'][key]['spearman'] for p in per],ddof=1))
        results['models'][name] = result
        save(out/'partial_results.json',results)
    results['paired_scoring_sensitivity'] = {}
    for name in MODELS:
        rows = {label:np.mean([all_seed_metrics[name,s,label+'|true']['spearman'] for s in SEEDS],axis=0)
                for label in ('L90-original-Rso3','L90-Sshared','L90-Sshared-all')}
        results['paired_scoring_sensitivity'][name] = {
            'Sshared_minus_original_rho':bootstrap_diff(rows['L90-Sshared'],rows['L90-original-Rso3'],saved['episode']),
            'allmask_minus_originalmask_rho':bootstrap_diff(rows['L90-Sshared-all'],rows['L90-Sshared'],saved['episode'])}
    results['status']='complete'; results['elapsed_seconds']=time.time()-started
    results['run_provenance']={'source':str(Path(__file__).resolve()),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
        'module_imports':{m.__name__:m.__file__ for m in (s1,v2,sq,tr,vs)},
        'python':sys.executable,'missing_common_rule_file':not Path('/work/docs/memory_pilot_gates.md').exists()}
    save(out/'results.json',results)
    write_report(out,results)
    (out/'DONE').write_text('complete\n')
    log(out,'complete',elapsed_seconds=results['elapsed_seconds'])


def write_report(out, r):
    """Report is generated only from frozen saved diagnostics, never handwritten metrics."""
    r['task_names_40']={int(k):v for k,v in r['task_names_40'].items()}
    r['task_names_90']={int(k):v for k,v in r['task_names_90'].items()}
    lines=['実行: 完了','',
      'EVT1：既存の3 seed（4101–4103）の R5・R0・P5 を固定したオフライン診断。学習・モデル選択は行わず、Eval の正解は採点だけに使用した。表は行別指標を seed 平均後に集計。LIBERO-40 は保留8タスク5,012窓、LIBERO-90 は10タスク619窓（hard 134）。','',
      '## 先に確認された採点の不一致','',
      '`scene_invariance.py:prepare` は既存 `W_err_S-shared_train.npz`（共通の非負倍率だけ、回転なし）の教師・採点誤差を読む。一方 `_step1_eval_data` は `key_query_probe.shape_error_matrix`（SO(3)回転＋共通倍率を正解に合わせる）で Eval を採点している。候補も、前者は元のデモ姿勢に復元した物理行動、後者は正規化した型。型誤差の定義を跨いだ .63→.04 は、そのまま転移量と解釈できない。','',
      'さらに Eval の既存 `allowed` は LIBERO-90 task ID と LIBERO-40 source task ID の数値が同じ候補を除外する。以下はまず既存採点を再現し、その後、同一 S-shared 誤差、さらに全候補許可を順に比較する。いずれも既存モデルの再採点であり、この結果によるモデル再選択はない。','',
      '## 全体：採点条件を揃えるとどうなるか','',
      '| 採点条件 | モデル | 1位型誤差 | 回収率 | Spearman ± seed SD |','|---|---|---:|---:|---:|']
    for label in ('L40-Sshared','L90-original-Rso3','L90-Sshared','L90-Sshared-all'):
        for name in MODELS:
            m=r['models'][name]['metrics'][label+'|true']
            lines.append(f"| {label} | {name} | {m['top1_error']:.5f} | {m['recovery']:.3f} | {m['spearman']:.3f} ± {m['seed_sd_spearman']:.3f} |")
    lines += ['','回収率は元コードと同じく `clip((random−top1)/(random−oracle),0,1)` の窓平均。random は候補を一様に選ぶ際の期待誤差（乱数を1回引いた値ではない）。負の回収率を0に切るため、ランダムな並べ方でも平均0とは限らない。未クリップ値は results.json に併記。','',
      '| モデル | 同一 S-shared − 旧採点の Spearman差 [episode bootstrap 95%] | 全候補 − 数値ID除外の差 [95%] |','|---|---:|---:|']
    for name,v in r['paired_scoring_sensitivity'].items():
        a=v['Sshared_minus_original_rho'];b=v['allmask_minus_originalmask_rho']
        lines.append(f"| {name} | {a['mean']:+.3f} [{a['ci95'][0]:+.3f},{a['ci95'][1]:+.3f}] | {b['mean']:+.3f} [{b['ci95'][0]:+.3f},{b['ci95'][1]:+.3f}] |")
    lines += ['','CI は固定10タスク内のデモ・エピソードを3,000回再標本化。学習seedやタスク母集団の不確実性までは含まない。','',
      '## タスク別：既存採点の再現と共通尺度の診断','',
      '各セルは 回収率 / Spearman。Sshared-all は LIBERO-40 と同じ誤差を使い、LIBERO-90 と LIBERO-40 の数値ID衝突を除いた採点。','',
      '| LIBERO-90 task | n | R5 旧 | R0 旧 | P5 旧 | R5 Sshared-all | R0 Sshared-all | P5 Sshared-all | 旧oracle/random | 共通oracle/random |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    lab=load_npz(out/'diagnostic_labels.npz')
    def pair(m):return f"{m['recovery']:.3f}/{m['spearman']:.3f}"
    for t in sorted(np.unique(lab['eval_task'])):
        old=[r['models'][n]['task_metrics']['L90-original-Rso3|true'][str(t)] for n in MODELS]
        new=[r['models'][n]['task_metrics']['L90-Sshared-all|true'][str(t)] for n in MODELS]
        lines.append(f"| {t}: {r['task_names_90'][int(t)]} | {old[0]['rows']} | "+' | '.join(pair(m) for m in old+new)+f" | {old[0]['oracle']:.5f}/{old[0]['random']:.5f} | {new[0]['oracle']:.5f}/{new[0]['random']:.5f} |")
    lines += ['','## 候補側：並べ方で稼げる余地と平坦さ','',
      '| 条件 | oracle平均 | random平均 | 差の平均 | 相対差 (random−oracle)/random | 候補誤差CV平均 | 幅(p90−p10)/p50 | oracle/正解エネルギー |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for label in ('L40-Sshared','L90-original-Rso3','L90-Sshared-all'):
        l=r['landscapes'][label]
        lines.append(f"| {label} | {l['oracle']['mean']:.5f} | {l['random']['mean']:.5f} | {l['gap']['mean']:.5f} | {l['relative_gap']['mean']:.3f} | {l['row_error_cv']['mean']:.3f} | {l['relative_p90_p10_width']['mean']:.3f} | {l['oracle_over_query_energy']['mean']:.3f} |")
    lines += ['','正解エネルギーは物理座標6軸の二乗平均。タスクの動きの大きさが異なるため、絶対誤差だけで分布距離を断定しない。窓ごとの分布の10/50/90/95%点は results.json。','',
      '| 条件 | oracle出どころ上位5タスク（窓数） | 上位5単位ID（窓数） | oracle同点候補数平均 |','|---|---|---|---:|']
    for label in ('L40-Sshared','L90-original-Rso3','L90-Sshared-all'):
        p=r['oracle_provenance'][label]
        tasks='; '.join(f"{x['id']}:{x['count']} ({r['task_names_40'][x['id']]})" for x in p['task_counts'][:5])
        units='; '.join(f"{x['id']}:{x['count']}" for x in p['unit_counts'][:5])
        lines.append(f"| {label} | {tasks} | {units} | {p['ties_within_1e_8_mean']:.2f} |")
    lines += ['','単位IDはキャッシュの source_unit（1,595候補を構成する個々の元単位）であり、57クラスタ等のIDではない。全窓の oracle候補・出どころと全順位の集計は results.json と diagnostic_labels.npz。','',
      'Evalタスク別のoracle出どころ。表は各採点で最も多い2つの `LIBERO-40 task ID / source_unit ID : 窓数`。','',
      '| Eval task | 旧採点のtask/unit | 共通尺度・全候補のtask/unit |','|---|---|---|']
    for t in sorted(np.unique(lab['eval_task'])):
        cells=[]
        for label in ('L90-original-Rso3','L90-Sshared-all'):
            picks=np.asarray(r['oracle_provenance'][label]['oracle_candidate'])[lab['eval_task']==t]
            pairs=np.stack([lab['pool_task'][picks],lab['pool_unit'][picks]],axis=1)
            units,counts=np.unique(pairs,axis=0,return_counts=True)
            order=np.argsort(-counts,kind='stable')[:2]
            cells.append('; '.join(f'{units[j,0]}/{units[j,1]}:{counts[j]}' for j in order))
        lines.append(f"| {t} | {cells[0]} | {cells[1]} |")
    lines += ['','## クエリ側：分布外距離と履歴ゼロ化','',
      '| モデル / seed | 埋め込み次元 | NN中央値 40保留/90 | Mahalanobis中央値 40保留/90 | 90が40保留95%点を超える割合 NN/MD |','|---|---:|---:|---:|---:|']
    for name in MODELS:
        for p in r['models'][name]['seeds']:
            o=p['ood'];h=o['lib40_holdout'];e=o['lib90_eval'];f=o['eval_above_holdout_q95_fraction']
            lines.append(f"| {name}/{p['seed']} | {o['dimension']} | {h['nearest_unit_sphere_euclidean']['q50']:.3f}/{e['nearest_unit_sphere_euclidean']['q50']:.3f} | {h['mahalanobis_shrinkage']['q50']:.2f}/{e['mahalanobis_shrinkage']['q50']:.2f} | {f['nearest_distance']:.3f}/{f['mahalanobis']:.3f} |")
    lines += ['','参照はLIBERO-40学習用から seed 4101で固定抽出した4,096窓。共分散は参照だけで Ledoit–Wolf 推定し、Evalで調整しない。R5は5×256の正規化queryを連結/√5（候補スコアの内積と同じ空間）、R0/P5はL2正規化256次元。異なるモデルのMDの絶対値を直接比較しない。LIBERO-40保留側は過去のモデル選択に使われているため、その性能を独立な汎化推定値と解釈しない。','',
      '| 手首トークンの平均L2ノルム | 平均 | 中央値 | 90%点 |','|---|---:|---:|---:|']
    for k in ('fit_reference','lib40_holdout','lib90_eval'):
        v=r['wrist_tokens'][k];lines.append(f"| {k} | {v['mean']:.2f} | {v['q50']:.2f} | {v['q90']:.2f} |")
    lines += ['','重要：キャッシュ生成元 `pi05_memory/core.py:prefix_features` は、保存前に各トークンをL2正規化している。そのため約1というノルムは構築上の性質で、VLMの生の活性ノルムの分布差は今回のキャッシュから判定できない。ノルムが一致することを分布内の証拠にしない。']
    lines += ['','| 採点条件 | モデル | Spearman 真→履歴ゼロ | 回収率 真→ゼロ |','|---|---|---:|---:|']
    for label in ('L40-Sshared','L90-original-Rso3','L90-Sshared-all'):
        for name in MODELS:
            a=r['models'][name]['metrics'][label+'|true'];b=r['models'][name]['metrics'][label+'|zero']
            lines.append(f"| {label} | {name} | {a['spearman']:.3f}→{b['spearman']:.3f} | {a['recovery']:.3f}→{b['recovery']:.3f} |")
    lines += ['','## キー側：同じ候補への集中','',
      '| モデル | データ | top1異なり数 | top1最大窓率 | top10有効候補数(exp entropy) | top10共通件数（全組/異タスク組） | 一様選択なら共通件数 |','|---|---|---:|---:|---:|---:|---:|']
    for name in MODELS:
        for label in ('L40-Sshared','L90-original-Rso3'):
            cs=[p['concentration'][label] for p in r['models'][name]['seeds']]
            avg=lambda k,f:float(np.mean([c[k][f] for c in cs]))
            lines.append(f"| {name} | {label} | {avg('top1','unique_candidates'):.1f} | {avg('top1','max_fraction_of_queries'):.3f} | {avg('top10','effective_candidates_entropy'):.1f} | {avg('top10','mean_intersection'):.2f}/{avg('top10','cross_task_mean_intersection'):.2f} | {avg('top10','uniform_allowed_expected_intersection'):.3f} |")
    lines += ['','重なりはseed固定で抽出した最大10,000組の異なる窓ペア。候補ごとの頻度・上位タスク/単位は各seed diagnostics.json。同一エピソードの近接窓も含むため、異タスク組を併記。集中は分布外の原因とも結果ともなりうる。窓数が5,012対619で異なるため、異なり数の単純比較はできない。','']
    if 'matched_concentration' in r:
        lines += ['同数対照：40保留から619窓を30通り、固定seedで非復元抽出。40欄は30抽出×3学習seedの平均、90欄は3 seed平均。ペアの共通件数は全異タスク・異窓ペアから厳密計算。','',
          '| モデル | top1異なり数 40同数/90 | top1最大率 40同数/90 | top10有効候補数 40同数/90 | top10共通件数 40同数/90 | 異タスク共通件数 40同数/90 |','|---|---:|---:|---:|---:|---:|']
        for name,v in r['matched_concentration'].items():
            h=v['holdout_matched_mean'];e=v['eval_seed_mean']
            p=lambda key:f"{h[key]:.3f}/{e[key]:.3f}"
            lines.append(f"| {name} | {p('top1_unique')} | {p('top1_max_fraction')} | {p('top10_effective')} | {p('top10_pair_intersection')} | {p('top10_cross_task_intersection')} |")
        lines += ['','候補が共通すること自体は同じ運動の検索でも起こる。全クエリが同じ候補しか返さないような全面的な崩壊か、比較的緩やかな偏りかをこの表から区別する。','']
    lines += [
      '## 指標だけの問題か：上位k件と良い候補の再現率','',
      '型誤差が許可候補の下位10%に入るものを「良い候補」とする（同点は全て含む）。Recallはその良い集合の何割を回収したか。Hitは少なくとも1件含む窓率。top-k誤差は検索されたk件中の最小であり、実行時に正解で選べるという意味ではない。','',
      '| 条件 | モデル | top1 / top5-best / top20-best 誤差 | Recall@1/5/20 | Hit@1/5/20 |','|---|---|---:|---:|---:|']
    for label in ('L40-Sshared','L90-original-Rso3','L90-Sshared-all'):
        for name in MODELS:
            m=r['models'][name]['metrics'][label+'|true']
            vals=lambda pattern,fmt:'/'.join(format(m[pattern.format(k=k)],fmt) for k in (1,5,20))
            lines.append(f"| {label} | {name} | {vals('top{k}_best_error','.5f')} | {vals('top{k}_bottom10_recall','.3f')} | {vals('top{k}_bottom10_hit','.3f')} |")
    lines += ['','上位k平均誤差、hard行、未クリップ回収率も results.json に保存。Recallの一様選択基準はk/許可候補数、top1のHit基準は約10%。','',
      '## 監査・範囲','',
      f"- 推論・主診断の計算時間: {r['elapsed_seconds']/60:.1f}分（別途、実装・レポート集計・数値監査）。既存特徴だけで診断し、π0.5推論・追加学習なし。GPU割当上限10%、CPU8スレッド・8–15コア、nice10、PREALLOCATE=false。",
      '- 既存Evalの3指標を行単位で照合。seed別の最大差は各 diagnostics.json の reproduction。旧ファイルは読み取りのみ。',
      '- Evalのスコア・query embedding・履歴ゼロのスコアを seed別 inference.npz に保存。画像の向きは既存の修正済みキャッシュを使用。',
      f"- 数値ID除外が影響する窓数: {r['candidate_mask_audit']['rows_affected']} / 619。具体的な指示文の対は results.json の collisions。",
      '- 指定された /work/docs/memory_pilot_gates.md は存在せず、依頼文の制約に従った。',
      '- この診断から閉ループ成功率、因果関係、LIBERO-90を見たことがあるかどうかは断定しない。今後モデルを変更する場合は別の開発データを用い、今回のEvalを選択に使わない。','',
      '## 仮説と次の実験','',
      'この節は、保存した results.json に基づく診断後の解釈を別途追記する。モデルの再選択には使わない。','']
    if (out/'audit.json').exists():
        a=json.loads((out/'audit.json').read_text())
        audit_line=f"- 数値監査: {a['independent_numeric_assertions']}件通過（[audit.json](audit.json)）。保存スコアから独立にSpearman・top-k最小誤差を計算、候補重なりを全ペア列挙で確認、旧Eval誤差8窓をSO(3)＋倍率のコードで再計算して一致を確認した。"
        drift=[]
        for name,rows in a['old_result_row_drift'].items():
            for row in rows:
                if row['different_error_rows_gt_1e_8']:
                    drift.append('{} seed {}: {}窓、行最大差 {:.3g}、全窓平均差 {:+.3g}'.format(name,row['seed'],row['different_error_rows_gt_1e_8'],row['max_absolute_row_difference'],row['mean_error_difference']))
        drift_line='- 旧記録との1位型誤差の不一致（差>1e−8）: '+('; '.join(drift) if drift else 'なし')+'。詳細は [ranking-drift.json](ranking-drift.json)。近接候補の順が異なる可能性があるが、原因を断定せず、再計算値をそのまま集計している。'
        pos=lines.index('## 仮説と次の実験')
        lines[pos:pos]=[audit_line,drift_line,'']
    if 'matched_concentration' in r:
        lines=lines[:-2]
        rr=r['models']['R5']['metrics'];land=r['landscapes'];c=r['matched_concentration']['R5']
        h=c['holdout_matched_mean'];e=c['eval_seed_mean']
        md=float(np.mean([p['ood']['eval_above_holdout_q95_fraction']['mahalanobis'] for p in r['models']['R5']['seeds']]))
        nn=float(np.mean([p['ood']['eval_above_holdout_q95_fraction']['nearest_distance'] for p in r['models']['R5']['seeds']]))
        diff=r['paired_scoring_sensitivity']['R5']['Sshared_minus_original_rho']
        ts=r['models']['R5']['task_metrics']['L90-Sshared-all|true']
        lowest=sorted(ts,key=lambda k:ts[k]['recovery'])[:3]
        low_scores=', '.join(format(ts[t]['recovery'], '.3f') for t in lowest)
        lines += [
          f"1. **最も強い根拠：落差の大部分は学習時とEval時の採点対象の違い。** R5 Spearmanは同じスコアで {rr['L90-original-Rso3|true']['spearman']:.3f}→{rr['L90-Sshared|true']['spearman']:.3f}、差 {diff['mean']:+.3f}、エピソードbootstrap95% [{diff['ci95'][0]:+.3f},{diff['ci95'][1]:+.3f}]。回転を無視する型の並べ方と、向きも含む型の並べ方は別課題であり、元の .63→.04 を画像クエリの全面失敗とするのは不適切。数値ID除外の影響はこれに比べ小さい。**次の実験：** Evalに触れない固定の40保留/別Dev窓で、source姿勢を復元したS-sharedと回転許容の二つを同じ固定スコアに当て、タスク集合間で採点定義・候補順序・許可マスクを一致させる回帰テストを実施する。",
          '',
          f"2. **中程度の根拠：未学習タスクの一部はメモリで表現しにくく、同時に並べ方も悪くなる。** 共通尺度のoracle/正解エネルギーは40保留 {land['L40-Sshared']['oracle_over_query_energy']['mean']:.3f}、90 {land['L90-Sshared-all']['oracle_over_query_energy']['mean']:.3f}。oracleとrandomの相対差は {land['L40-Sshared']['relative_gap']['mean']:.3f}→{land['L90-Sshared-all']['relative_gap']['mean']:.3f} に縮むが、90でも大きな余地があり『どれを選んでも同じ』ではない。低回収率のtaskは {', '.join(lowest)}（各 {low_scores}）。R5 top20内の最良誤差も {rr['L90-Sshared-all|true']['top20_best_error']:.5f} と全候補oracle {rr['L90-Sshared-all|true']['oracle']:.5f} に届かず、全順位Spearmanだけの問題でもない。**次の実験：** 別Devタスクで元のメモリと、Eval由来デモを使わず40内の窓位置・長さを網羅した候補集合を固定し、oracle誤差と同じ固定検索器のtop20-bestを比較する。候補被覆と並べ方の寄与を分ける。",
          '',
          f"3. **弱めの根拠：クエリの見え方・文脈への写像が一部で移らず、候補の選択が偏る。** R5の90クエリのうち40保留95%点を超える割合はNN距離 {nn:.3f}、Mahalanobis {md:.3f} で、大多数が極端な分布外という証拠はない。同数比較のtop10有効候補数は40 {h['top10_effective']:.1f}→90 {e['top10_effective']:.1f}、異タスク間の共通件数は {h['top10_cross_task_intersection']:.3f}→{e['top10_cross_task_intersection']:.3f}。全面的に同じ候補を返す崩壊ではない。履歴ゼロでも共通尺度のSpearmanは {rr['L90-Sshared-all|zero']['spearman']:.3f}（真 {rr['L90-Sshared-all|true']['spearman']:.3f}）で、履歴だけが原因とも言えない。正規化済み手首ノルムは生の活性変化を検出できない。**次の実験：** 別Devの同じ局面について画像だけ・状態だけ・履歴だけを対応付きで差し替える固定モデルの入力介入を行い、query距離と候補順位の変化を比較する。新しい検索器を学習する前に、画像/状態/履歴のどの経路で偏るかを切り分ける。",
          '',
          'これらは診断後の仮説であり、2・3の因果関係や閉ループでの改善は未検証。新規学習の根拠としては、まず採点契約を揃えることを優先する。','']
    (out/'report.md').write_text('\n'.join(lines))


def exact_rank_concentration(rank, tasks, candidates=1595):
    """Size-matched counts and exact all-pair overlap without pair sampling."""
    n=len(rank);out={}
    for k in (1,10):
        ids=rank[:,:k]
        cnt=np.bincount(ids.ravel(),minlength=candidates)
        p=cnt[cnt>0]/cnt.sum()
        task_counts=np.array([np.sum(tasks==t) for t in np.unique(tasks)])
        within=np.zeros(candidates,dtype=np.int64)
        for t in np.unique(tasks):
            c=np.bincount(ids[tasks==t].ravel(),minlength=candidates)
            within+=c*c
        out[f'top{k}_unique']=int((cnt>0).sum())
        out[f'top{k}_max_fraction']=float(cnt.max()/n)
        out[f'top{k}_effective']=float(np.exp(-np.sum(p*np.log(p))))
        out[f'top{k}_pair_intersection']=float(np.sum(cnt*(cnt-1))/(n*(n-1)))
        out[f'top{k}_cross_task_intersection']=float((np.sum(cnt*cnt)-within.sum())/(n*n-np.sum(task_counts**2)))
    return out


def finalize(out):
    """Report-only additions from saved scores; no model inference or selection."""
    out=Path(out);r=json.loads((out/'results.json').read_text())
    if 'matched_concentration' in r:raise FileExistsError('already finalized')
    train=load_npz(ROOT/'scene-invariance-01/fit_rows.npz')
    hold=np.flatnonzero(train['select']);a40=train['allowed'][hold]
    d=load_npz(out/'diagnostic_labels.npz')
    n=len(d['eval_task']);matched={}
    for name in MODELS:
        hs=[];es=[]
        for seed in SEEDS:
            with np.load(out/f'{name}-s{seed}-inference.npz') as z:
                h=np.argsort(-np.where(a40,z['lib40_scores'],-np.inf),axis=1,kind='stable')[:,:10]
                e=np.argsort(-np.where(d['allowed_original'],z['eval_scores'],-np.inf),axis=1,kind='stable')[:,:10]
            es.append(exact_rank_concentration(e,d['eval_task']))
            rng=np.random.default_rng(4101)
            for _ in range(30):
                ix=rng.choice(len(h),n,replace=False)
                hs.append(exact_rank_concentration(h[ix],train['task'][hold][ix]))
        matched[name]={'holdout_rows':n,'resamples':30,'seeds':list(SEEDS),
            'holdout_matched_mean':{k:float(np.mean([h[k] for h in hs])) for k in hs[0]},
            'holdout_matched_q05_q95':{k:np.quantile([h[k] for h in hs],[.05,.95]).tolist() for k in hs[0]},
            'eval_seed_mean':{k:float(np.mean([e[k] for e in es])) for k in es[0]}}
    r['matched_concentration']=matched
    metric_rank=row_metrics(-d['errors_Sshared'],d['errors_original'],d['allowed_original'])['spearman']
    r['rank_agreement_between_error_definitions']={'overall':summary(metric_rank),
       'by_task':{str(int(t)):summary(metric_rank[d['eval_task']==t]) for t in np.unique(d['eval_task'])}}
    r['report_finalization_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    r['report_finalization']='added sample-size matched candidate concentration and diagnostic hypotheses; saved inference unchanged'
    (out/'results.json').rename(out/'inference-results.json')
    (out/'report.md').rename(out/'report-inference.md')
    save(out/'results.json',r)
    write_report(out,r)
    log(out,'report_finalized',matched_rows=n)


def audit(out):
    """Independent scalar checks of persisted scores and label definitions."""
    out=Path(out)
    d=load_npz(out/'diagnostic_labels.npz')
    original=load_npz(ROOT/'lib-closed-01/eval_rows.npz')
    checks=0;drift={};details=[]
    for name in MODELS:
        mismatch=[]
        for si,seed in enumerate(SEEDS):
            with np.load(out/f'{name}-s{seed}-inference.npz') as z:
                sc=np.asarray(z['eval_scores'])
            allow=d['allowed_original'];errors=d['errors_original']
            chosen=np.argmax(np.where(allow,sc,-np.inf),axis=1)
            selected=errors[np.arange(len(sc)),chosen]
            difference=np.abs(selected-original[f'{name}_row_type_error'][si])
            mismatch.append({'seed':seed,'different_error_rows_gt_1e_8':int(np.sum(difference>1e-8)),
                             'max_absolute_row_difference':float(difference.max()),
                             'mean_error_difference':float(np.mean(selected-original[f'{name}_row_type_error'][si]))})
            for i in np.flatnonzero(difference>1e-8):
                previous_error=original[f'{name}_row_type_error'][si,i]
                old_candidates=np.flatnonzero(np.isclose(errors[i],previous_error,rtol=0,atol=1e-10))
                ranking=np.argsort(-np.where(allow[i],sc[i],-np.inf),kind='stable')
                details.append({'model':name,'seed':seed,'row':int(i),'task':int(original['task'][i]),
                    'candidate_now':int(chosen[i]),'candidate_matching_old_error':old_candidates.tolist(),
                    'score_now':float(sc[i,chosen[i]]),'scores_old_candidates':sc[i,old_candidates].tolist(),
                    'top2_score_margin':float(sc[i,ranking[0]]-sc[i,ranking[1]]),
                    'old_error':float(previous_error),'new_error':float(selected[i])})
            for label,err,a in [('L90-original-Rso3',errors,allow),('L90-Sshared-all',d['errors_Sshared'],np.ones_like(allow))]:
                metrics=row_metrics(sc,err,a)
                chosen=np.argmax(np.where(a,sc,-np.inf),axis=1)
                np.testing.assert_array_equal(metrics['top1_error'],err[np.arange(len(sc)),chosen]);checks+=1
                for i in np.random.default_rng(4101).choice(len(sc),12,replace=False):
                    rho=spearmanr(sc[i,a[i]],-err[i,a[i]]).statistic
                    np.testing.assert_allclose(metrics['spearman'][i],rho,atol=1e-12);checks+=1
                    order=sorted(np.flatnonzero(a[i]),key=lambda j:(-float(sc[i,j]),j))
                    for k in (1,5,20):
                        np.testing.assert_allclose(metrics[f'top{k}_best_error'][i],min(err[i,j] for j in order[:k]),atol=1e-8);checks+=1
        drift[name]=mismatch
    # Prove the original Eval matrix implements R+scale, not just a naming difference.
    import key_query_probe as kq
    train=load_npz(ROOT/'scene-invariance-01/fit_rows.npz')
    q=vs.to_physical(original['target'][:8]).astype(np.float32)
    q=np.concatenate([q,np.zeros((len(q),10,1),np.float32)],axis=-1)
    recomputed=kq.shape_error_matrix(q,train['pool_shape'],vs,device='cpu',batch=8)
    np.testing.assert_allclose(recomputed,original['errors'][:8],rtol=2e-5,atol=2e-7);checks+=1
    ranking=np.array([[0,1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9,10],[5,6,7,8,9,10,11,12,13,14]])
    ts=np.array([0,0,1]);m=exact_rank_concentration(ranking,ts,15)
    for k in (1,10):
        allpair=[len(set(ranking[i,:k])&set(ranking[j,:k])) for i in range(3) for j in range(3) if i!=j]
        cross=[len(set(ranking[i,:k])&set(ranking[j,:k])) for i in range(3) for j in range(3) if ts[i]!=ts[j]]
        np.testing.assert_allclose(m[f'top{k}_pair_intersection'],np.mean(allpair));checks+=1
        np.testing.assert_allclose(m[f'top{k}_cross_task_intersection'],np.mean(cross));checks+=1
    result={'status':'passed','independent_numeric_assertions':checks,'old_result_row_drift':drift,
       'old_eval_Rso3_matrix_recalculated_rows':8,
       'new_model_training_or_selection':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save(out/'audit.json',result)
    save(out/'ranking-drift.json',details)
    print(json.dumps(result,default=plain,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--device',default='cuda');p.add_argument('--ref-rows',type=int,default=4096)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--finalize',action='store_true')
    p.add_argument('--audit',action='store_true')
    p.add_argument('--report-only',action='store_true')
    a=p.parse_args()
    if a.self_test:
        error=np.repeat(np.arange(40,dtype=float)[None],2,axis=0)
        allowed=np.ones_like(error,bool);allowed[:,0]=False
        score=-error;score[:,0]=1e9
        m=row_metrics(score,error,allowed)
        np.testing.assert_allclose(m['spearman'],1.)
        np.testing.assert_allclose(m['top1_error'],1.)
        np.testing.assert_allclose(m['top5_bottom10_recall'],1.)
        np.testing.assert_allclose(m['top1_bottom10_recall'],.25)
        np.testing.assert_allclose(m['recovery'],1.)
        opposite=row_metrics(error,error,np.ones_like(allowed))
        np.testing.assert_allclose(opposite['spearman'],-1.)
        np.testing.assert_allclose(opposite['recovery'],0.)
        ties=row_metrics(np.ones_like(error),np.ones_like(error),allowed)
        np.testing.assert_allclose(ties['spearman'],0.)
        np.testing.assert_allclose(ties['top5_bottom10_recall'],5/39)
        print('self-test: passed (mask, rank sign, recall denominator, clipped recovery, constant/tie handling)')
    elif a.finalize:finalize(a.out)
    elif a.audit:audit(a.out)
    elif a.report_only:write_report(a.out,json.loads((a.out/'results.json').read_text()))
    else:run(a.out,device=a.device,ref_rows=a.ref_rows)


if __name__=='__main__':main()
