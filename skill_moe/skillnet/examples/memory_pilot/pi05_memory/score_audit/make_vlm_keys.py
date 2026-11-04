#!/usr/bin/env python3
"""Exact chunked VLM-token key variants for pi05 memory retrieval."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/run-01')
OUT = Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/score-audit-01')
sys.path.insert(0, str(HERE.parent))
import run  # noqa: E402


RAW_METHODS = ('current', 'base_only', 'wrist_only', 'instruction_only', 'cameras_only',
               'top10pct', 'top25pct', 'q75', 'idf_weighted')
TRANSFORMS = ('centered', 'whitened')


def _group_stats(tokens: np.ndarray, groups: np.ndarray, mem_ids: np.ndarray,
                 sample_rows: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Groupwise channel mean/std on deterministic memory-only token sample."""
    rng = np.random.default_rng(20261104)
    sample = np.sort(rng.choice(mem_ids, min(sample_rows, len(mem_ids)), replace=False))
    sums = np.zeros((3, 2048), np.float64)
    sqs = np.zeros((3, 2048), np.float64)
    counts = np.zeros(3, np.int64)
    for start in range(0, len(sample), 8):
        ids = sample[start:start+8]
        x = np.asarray(tokens[ids], dtype=np.float32)
        g = np.asarray(groups[ids])
        for group in range(3):
            vals = x[g == group]
            if vals.size:
                vals = vals.reshape(-1, 2048)
                sums[group] += vals.sum(0, dtype=np.float64)
                sqs[group] += np.square(vals, dtype=np.float32).sum(0, dtype=np.float64)
                counts[group] += len(vals)
    mean = (sums / counts[:, None]).astype(np.float32)
    var = np.maximum(sqs / counts[:, None] - mean.astype(np.float64)**2, 1e-6)
    std = np.sqrt(var).astype(np.float32)
    return mean, std


def _load_bank(torch, tokens, groups, mem_ids, *, mode: str, mean=None, std=None,
               cap_fraction=.24):
    torch.cuda.set_per_process_memory_fraction(cap_fraction, 0)
    n = len(mem_ids)
    bank = torch.empty((n, 712, 2048), device='cuda', dtype=torch.float16)
    bank_group = np.asarray(groups[mem_ids])
    means = torch.as_tensor(mean, device='cuda') if mean is not None else None
    stds = torch.as_tensor(std, device='cuda') if std is not None else None
    with torch.inference_mode():
        for start in range(0, n, 8):
            end = min(n, start + 8)
            if mode == 'raw':
                # Match run.retrieve's fp16 bank normalization operation order.
                xh = torch.as_tensor(np.asarray(tokens[mem_ids[start:end]]), device='cuda', dtype=torch.float16)
                denom = torch.linalg.vector_norm(xh.float(), dim=-1, keepdim=True).clamp_min(1e-8).half()
                bank[start:end] = xh / denom
                continue
            x = torch.as_tensor(np.asarray(tokens[mem_ids[start:end]]), device='cuda', dtype=torch.float32)
            if mode in TRANSFORMS:
                for group in range(3):
                    mask = torch.as_tensor(bank_group[start:end] == group, device='cuda')
                    if mode == 'centered':
                        x[mask] -= means[group]
                    else:
                        x[mask] = (x[mask] - means[group]) / stds[group]
            x = x / torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-8)
            bank[start:end] = x.half()
    return bank, bank_group


def _query_tokens(torch, x: np.ndarray, g: np.ndarray, *, mode: str, mean=None, std=None):
    q = torch.as_tensor(np.asarray(x).copy(), device='cuda', dtype=torch.float32)
    for group in range(3):
        pos = torch.as_tensor(g == group, device='cuda')
        if mode == 'centered':
            q[pos] -= torch.as_tensor(mean[group], device='cuda')
        elif mode == 'whitened':
            q[pos] = (q[pos] - torch.as_tensor(mean[group], device='cuda')) / torch.as_tensor(std[group], device='cuda')
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-8)
    return q.half()


def _score_values(torch, token_max: 'torch.Tensor', qgroups: np.ndarray,
                  idf_weights: np.ndarray) -> tuple[dict[str, 'torch.Tensor'], 'torch.Tensor']:
    group_score = []
    aggs = {}
    idf_tensor = torch.as_tensor(idf_weights, device=token_max.device, dtype=torch.float32)
    for group in range(3):
        pos = torch.as_tensor(qgroups == group, device=token_max.device)
        vals = token_max[:, pos].float()
        mean = vals.mean(1)
        group_score.append(mean)
        aggs[f'g{group}'] = mean
        n = vals.shape[1]
        for pct, name in ((.10, 'top10pct'), (.25, 'top25pct')):
            k = max(1, int(np.ceil(n * pct)))
            aggs.setdefault(name, []).append(torch.sort(vals, dim=1, descending=True).values[:, :k].mean(1))
        aggs.setdefault('q75', []).append(torch.quantile(vals, .75, dim=1))
        w = idf_tensor[pos]
        aggs.setdefault('idf_weighted', []).append((vals * w[None]).sum(1) / w.sum().clamp_min(1e-8))
    aggs['current'] = torch.stack(group_score, 1).mean(1)
    aggs['base_only'] = group_score[0]
    aggs['wrist_only'] = group_score[1]
    aggs['instruction_only'] = group_score[2]
    aggs['cameras_only'] = torch.stack(group_score[:2], 1).mean(1)
    for key in ('top10pct', 'top25pct', 'q75', 'idf_weighted'):
        aggs[key] = torch.stack(aggs[key], 1).mean(1)
    return aggs, torch.stack(group_score, 1)


def _score_sweep(torch, tokens, groups, mem_ids, query_ids, rows, *, mode,
                 out, block=32, limit=None, cap_fraction=.24):
    mean = std = None
    if mode in TRANSFORMS:
        mean, std = _group_stats(tokens, groups, mem_ids)
        np.savez(out / f'{mode}_stats.npz', mean=mean, std=std)
    bank, bank_groups = _load_bank(torch, tokens, groups, mem_ids, mode=mode, mean=mean, std=std,
                                   cap_fraction=cap_fraction)
    # A memory-only token-direction rarity proxy: tokens aligned with the
    # groupwise bank centroid receive less weight.
    centroids = np.zeros((3, 2048), np.float32)
    for group in range(3):
        c = mean[group] if mean is not None else np.asarray(tokens[mem_ids[:min(256, len(mem_ids))]], np.float32)[
            np.asarray(groups[mem_ids[:min(256, len(mem_ids))]]) == group].mean(0).mean(0)
        centroids[group] = c / max(np.linalg.norm(c), 1e-8)
    mem_ep = np.asarray(rows['episode'][mem_ids])
    mem_task = np.asarray(rows['task'][mem_ids])
    text_cols = np.flatnonzero(np.any(np.asarray(groups[mem_ids])[:, 512:] == 2, axis=0))
    text_stop = 512 + int(text_cols[-1]) + 1
    methods = RAW_METHODS if mode == 'raw' else (f'{mode}_current',)
    outputs = {}
    for method in methods:
        outputs[method] = {}
        for exclusion in ('task', 'episode'):
            outputs[method][exclusion] = {
                'cand_rows': np.empty((len(query_ids), 64), np.int64),
                'score': np.empty((len(query_ids), 64), np.float32),
                'score_group': np.empty((len(query_ids), 64, 3), np.float32),
                'token_maxsim': np.empty((len(query_ids), 64, 712), np.float16),
            }
    start_time = time.monotonic()
    with torch.inference_mode():
        for qi, row_id in enumerate(query_ids):
            qgroups = np.asarray(groups[row_id])
            q = _query_tokens(torch, np.asarray(tokens[row_id]), qgroups, mode=mode, mean=mean, std=std)
            # IDF proxy weights based on agreement with memory-only group centroid.
            idf = np.ones(712, np.float32)
            for group in range(3):
                pos = np.flatnonzero(qgroups == group)
                if len(pos):
                    qc = torch.as_tensor(centroids[group], device='cuda')
                    sim = (q[torch.as_tensor(pos, device='cuda')].float() @ qc).cpu().numpy()
                    idf[pos] = 1.0 / np.maximum(.05 + np.maximum(sim, 0), .05)
            all_scores = {name: np.empty(len(mem_ids), np.float32) for name in methods}
            all_groups = np.empty((len(mem_ids), 3), np.float32)
            all_tokens = np.empty((len(mem_ids), 712), np.float16)
            all_tokens.fill(np.nan)
            for start in range(0, len(mem_ids), block):
                end = min(len(mem_ids), start + block)
                chunk_max = torch.full((end-start, 712), float('nan'), device='cuda', dtype=torch.float32)
                for group, (left, right) in enumerate(((0, 256), (256, 512), (512, text_stop))):
                    qpos_np = np.flatnonzero(qgroups == group)
                    if not len(qpos_np):
                        continue
                    qpos = torch.as_tensor(qpos_np, device='cuda')
                    sim = torch.matmul(q[qpos][None], bank[start:end, left:right].transpose(1, 2)).float()
                    valid = torch.as_tensor(bank_groups[start:end, None, left:right] == group, device='cuda')
                    sim.masked_fill_(~valid, -torch.inf)
                    vals = sim.amax(-1)
                    chunk_max[:, qpos] = vals
                score_dict, gscore = _score_values(torch, chunk_max, qgroups, idf)
                for name in methods:
                    actual = name if name in score_dict else 'current'
                    all_scores[name][start:end] = score_dict[actual].float().cpu().numpy()
                all_groups[start:end] = gscore.float().cpu().numpy()
                all_tokens[start:end] = chunk_max.cpu().numpy()
            for method in methods:
                for exclusion in ('task', 'episode'):
                    eligible = (mem_task != int(rows['task'][row_id]) if exclusion == 'task'
                                else mem_ep != int(rows['episode'][row_id]))
                    values = all_scores[method].copy()
                    values[~eligible] = -np.inf
                    order = np.argsort(-values, kind='stable')[:64]
                    if len(order) < 64 or not np.isfinite(values[order]).all():
                        raise ValueError(f'fewer than 64 candidates for {exclusion} exclusion')
                    dest = outputs[method][exclusion]
                    dest['cand_rows'][qi] = mem_ids[order]
                    dest['score'][qi] = values[order]
                    dest['score_group'][qi] = all_groups[order]
                    dest['token_maxsim'][qi] = all_tokens[order]
            if qi % 8 == 0:
                print(json.dumps({'stage': 'vlm_scores', 'mode': mode, 'query': qi+1,
                                  'total': len(query_ids), 'elapsed': time.monotonic()-start_time}), flush=True)
    del bank, bank_groups
    torch.cuda.empty_cache()
    return outputs, {'seconds': time.monotonic()-start_time, 'mode': mode, 'methods': list(methods)}


def _add_drafts(torch, method_outputs, rows, tokens, groups, mem_ids, query_ids, out):
    from run import restore_codec
    model = restore_codec(ROOT)
    bases = np.load(ROOT / 'bases.npy', mmap_mode='r', allow_pickle=False)
    pooled = np.load(ROOT / 'pooled.npy', mmap_mode='r', allow_pickle=False)
    residual = np.load(ROOT / 'predicted_residual.npy', mmap_mode='r', allow_pickle=False)
    for method, exclusions in method_outputs.items():
        for exclusion, data in exclusions.items():
            cand = data['cand_rows']
            qf = np.asarray(pooled[query_ids], np.float32)
            qr = np.asarray(residual[query_ids], np.float32)
            qs = np.asarray(rows['state'][query_ids], np.float32)
            draft = np.empty((len(query_ids), 64, 10, 7), np.float32)
            dist = np.empty((len(query_ids), 64), np.float32)
            first = np.empty_like(dist); last = np.empty_like(dist)
            truth = np.asarray(rows['actions'][query_ids, :, :7], np.float32)
            for start in range(0, len(query_ids), 32):
                end = min(len(query_ids), start+32)
                cr = cand[start:end].reshape(-1)
                n = len(cr)
                pred = model.decode(np.asarray(bases[cr]), np.repeat(qr[start:end], 64, axis=0),
                                    np.repeat(qf[start:end], 64, axis=0), np.repeat(qs[start:end], 64, axis=0))
                pred = pred[:, :, :7].reshape(end-start, 64, 10, 7).astype(np.float32)
                draft[start:end] = pred
                delta = pred - truth[start:end, None]
                dist[start:end] = np.mean(delta**2, axis=(2, 3))
                first[start:end] = np.mean(delta[:, :, 0]**2, axis=2)
                last[start:end] = np.mean(delta[:, :, -1]**2, axis=2)
            data.update(draft=draft, dist=dist, first_step_dist=first, last_step_dist=last,
                        cand_task=np.asarray(rows['task'][cand], np.int32),
                        cand_episode=np.asarray(rows['episode'][cand], np.int32))
            qsplit = _query_splits(rows, query_ids)
            path = out / 'keys' / f'{method}_{exclusion}.npz'
            np.savez_compressed(path, query_rows=query_ids.astype(np.int64), cand_rows=cand.astype(np.int64),
                                score=data['score'], exclusion=np.asarray(exclusion),
                                description=np.asarray(_description(method)),
                                score_group=data['score_group'], token_maxsim=data['token_maxsim'],
                                draft=draft, dist=dist, first_step_dist=first, last_step_dist=last,
                                cand_task=data['cand_task'], cand_episode=data['cand_episode'],
                                query_split=qsplit, query_task=np.asarray(rows['task'][query_ids]),
                                query_episode=np.asarray(rows['episode'][query_ids]))


def _query_splits(rows, query_ids):
    from oracle import query_split
    return query_split(rows, query_ids)


def _description(method):
    return {
        'current': 'Exact grouped token MaxSim, equal mean over base/wrist/instruction.',
        'base_only': 'Grouped token MaxSim, base-camera group only.',
        'wrist_only': 'Grouped token MaxSim, wrist-camera group only.',
        'instruction_only': 'Grouped token MaxSim, instruction group only.',
        'cameras_only': 'Equal mean of base and wrist group MaxSim; instruction excluded.',
        'top10pct': 'Per group, average highest 10% query-token MaxSim values, then mean groups.',
        'top25pct': 'Per group, average highest 25% query-token MaxSim values, then mean groups.',
        'q75': 'Per group 75th percentile query-token MaxSim, then mean groups.',
        'idf_weighted': 'Per-group query-token MaxSim weighted inversely by similarity to memory token centroid.',
        'centered_current': 'Memory-only group-centroid token centering, L2, exact grouped MaxSim.',
        'whitened_current': 'Memory-only groupwise diagonal token whitening (sampled stats), L2, exact grouped MaxSim.',
        'pooled': 'Cosine of cached valid-token pooled 2048-D feature.',
    }.get(method, method)


def _pooled_keys(out, rows, mem_ids, query_ids, limit=None):
    pooled = np.load(ROOT / 'pooled.npy', mmap_mode='r', allow_pickle=False)
    mem = np.asarray(pooled[mem_ids], np.float32)
    mem /= np.maximum(np.linalg.norm(mem, axis=1, keepdims=True), 1e-8)
    q = np.asarray(pooled[query_ids], np.float32)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)
    scores = q @ mem.T
    model = __import__('run').restore_codec(ROOT)
    bases = np.load(ROOT / 'bases.npy', mmap_mode='r', allow_pickle=False)
    residual = np.load(ROOT / 'predicted_residual.npy', mmap_mode='r', allow_pickle=False)
    result = {}
    for exclusion in ('task', 'episode'):
        cand = np.empty((len(query_ids), 64), np.int64); val = np.empty((len(query_ids), 64), np.float32)
        for qi, row in enumerate(query_ids):
            eligible = (np.asarray(rows['task'][mem_ids]) != rows['task'][row] if exclusion == 'task'
                        else np.asarray(rows['episode'][mem_ids]) != rows['episode'][row])
            sc = scores[qi].copy(); sc[~eligible] = -np.inf
            ix = np.argsort(-sc, kind='stable')[:64]
            cand[qi] = mem_ids[ix]; val[qi] = sc[ix]
        split = _query_splits(rows, query_ids)
        drafts = np.empty((len(query_ids), 64, 10, 7), np.float32)
        truth = np.asarray(rows['actions'][query_ids, :, :7], np.float32)
        dist = np.empty((len(query_ids), 64), np.float32); first=np.empty_like(dist); last=np.empty_like(dist)
        for start in range(0, len(query_ids), 32):
            end=min(len(query_ids),start+32); cr=cand[start:end].reshape(-1)
            pred=model.decode(np.asarray(bases[cr]),np.repeat(np.asarray(residual[query_ids[start:end]]),64,axis=0),
                np.repeat(np.asarray(pooled[query_ids[start:end]]),64,axis=0),
                np.repeat(np.asarray(rows['state'][query_ids[start:end]]),64,axis=0))[:,:,:7].reshape(end-start,64,10,7)
            drafts[start:end]=pred
            d=pred-truth[start:end,None];dist[start:end]=np.mean(d*d,axis=(2,3));first[start:end]=np.mean(d[:,:,0]**2,axis=2);last[start:end]=np.mean(d[:,:,-1]**2,axis=2)
        tg=np.asarray(rows['task'][cand]); eg=np.asarray(rows['episode'][cand])
        np.savez_compressed(out/'keys'/f'pooled_{exclusion}.npz',query_rows=query_ids,cand_rows=cand,score=val,
            exclusion=np.asarray(exclusion),description=np.asarray(_description('pooled')),draft=drafts,dist=dist,
            first_step_dist=first,last_step_dist=last,cand_task=tg,cand_episode=eg,query_split=split,
            query_task=np.asarray(rows['task'][query_ids]),query_episode=np.asarray(rows['episode'][query_ids]))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--limit',type=int)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--block',type=int,default=128)
    parser.add_argument('--cap-fraction',type=float,default=.24)
    args=parser.parse_args()
    import torch
    torch.set_num_threads(8)
    if not torch.cuda.is_available(): raise RuntimeError('VLM token MaxSim smoke requires CUDA')
    torch.cuda.set_per_process_memory_fraction(args.cap_fraction,0)
    rows=run.load_rows(ROOT)
    tokens=np.load(ROOT/'tokens.npy',mmap_mode='r',allow_pickle=False)
    groups=np.load(ROOT/'groups.npy',mmap_mode='r',allow_pickle=False)
    mem_ids=np.load(ROOT/'memory_row_ids.npy',mmap_mode='r',allow_pickle=False)
    query_ids=np.flatnonzero(rows['validation'])
    if args.limit is not None: query_ids=query_ids[:args.limit]
    out=args.output
    if (out/'vlm_key_smoke.json').exists() or any((out/'keys').glob('*.npz')):
        raise FileExistsError(f'refusing to overwrite VLM key cache in {out}')
    (out/'keys').mkdir(parents=True,exist_ok=True)
    timings=[]
    all_outputs={}
    for mode in ('raw','centered','whitened'):
        outputs,timing=_score_sweep(torch,tokens,groups,mem_ids,query_ids,rows,mode=mode,out=out,
                                    block=args.block,cap_fraction=args.cap_fraction)
        all_outputs.update(outputs);timings.append(timing)
        _add_drafts(torch,outputs,rows,tokens,groups,mem_ids,query_ids,out)
    _pooled_keys(out,rows,mem_ids,query_ids)
    saved=np.load(ROOT/'retrieved_row.npy',mmap_mode='r',allow_pickle=False)
    exact=np.load(out/'keys/current_episode.npz',allow_pickle=False)
    matches=exact['cand_rows'][:,0] == saved[query_ids]
    if not matches.all():
        bad=np.flatnonzero(~matches)[:10]
        raise AssertionError(f'current Top-1 disagrees with run.retrieve on {len(bad)} examples; query offsets={bad.tolist()}')
    old_scores=np.asarray(np.load(ROOT/'retrieved_score.npy',mmap_mode='r',allow_pickle=False)[query_ids])
    score_delta=np.asarray(exact['score'][:,0])-old_scores
    if not np.allclose(exact['score'][:,0],old_scores,rtol=1e-6,atol=1e-6):
        raise AssertionError(f'current Top-1 scores differ from run.retrieve, max abs delta={np.max(np.abs(score_delta))}')
    summary={'queries':len(query_ids),'memory_rows':len(mem_ids),'K':64,'cap_fraction':args.cap_fraction,
        'cuda_peak_allocated_bytes':int(torch.cuda.max_memory_allocated()),
        'cuda_peak_reserved_bytes':int(torch.cuda.max_memory_reserved()),'current_episode_top1_matches':int(matches.sum()),
        'current_episode_score_max_abs_delta':float(np.max(np.abs(score_delta))),
        'timings':timings,'seconds_total_scoring':sum(x['seconds'] for x in timings),
        'full_query_estimate_seconds':sum(x['seconds'] for x in timings)*3277/len(query_ids),
        'methods':list(RAW_METHODS)+['centered_current','whitened_current','pooled'],
        'note':'The task-excluded current key is intentionally different from the saved same-episode-excluded current retrieval.'}
    (out/'vlm_key_smoke.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
