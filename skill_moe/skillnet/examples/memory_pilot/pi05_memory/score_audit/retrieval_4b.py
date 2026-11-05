#!/usr/bin/env python3
"""Stage 4-B: retrieve raw action-near demonstrations from inference-time keys."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
PI05 = HERE.parent
REPO = HERE.parents[5]
import os as _os
_REPO = Path(__file__).resolve().parents[6]
_STATE_DIR = Path(_os.environ.get('SKILLNET_STATE_DIR', _REPO / '.skillnet-state'))
_PI05_ROOT = Path(_os.environ.get('SKILLNET_PI05_ROOT', _STATE_DIR / 'outputs/memory-pilot/pi05-memory40-20261104'))
RUN = _PI05_ROOT / 'run-01'
AUDIT = _PI05_ROOT / 'score-audit-01'
GAP = _PI05_ROOT / 'memory-gap-01'
OUT = _PI05_ROOT / 'retrieval-4b-01'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PI05))
import keys
import learned_key

VLM_METHODS = ('current', 'centered_current', 'wrist_only', 'cameras_only', 'pooled',
               'top10pct', 'top25pct', 'q75', 'idf_weighted', 'whitened_current')
REFERENCE_METHODS = ('pi05_pred_to_memory_action', 'codec_prior_b')
FIXED_METHODS = ('wrist_state_cos', 'wrist_state_std_l2',
                 'wrist_state_history_cos', 'wrist_state_history_std_l2')
TAUS = (.005, .01, .02)
T0S = (.2, .4)


def _gap_module():
    path = PI05 / 'memory_gap' / 'run.py'
    spec = importlib.util.spec_from_file_location('pi05_memory_gap_run_4b', path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def stable_topk(scores: np.ndarray, row_ids: np.ndarray, k: int = 64) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, np.float32).reshape(-1)
    row_ids = np.asarray(row_ids, np.int64).reshape(-1)
    if len(scores) != len(row_ids) or not np.isfinite(scores).all() or len(scores) < k:
        raise ValueError('invalid candidate score vector')
    ix = np.lexsort((row_ids, -scores))[:k]
    return row_ids[ix], scores[ix]


def standardized(x: np.ndarray, train_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, np.float32)
    mean = x[train_ids].mean(0, dtype=np.float64).astype(np.float32)
    std = x[train_ids].std(0, dtype=np.float64).astype(np.float32)
    std[std < 1e-5] = 1.
    return ((x - mean) / std).astype(np.float32), mean, std


def _rank_matrix(x: np.ndarray, qids: np.ndarray, mids: np.ndarray, rows: dict,
                 exclusion: str, metric: str, *, batch_rows: int = 64,
                 k: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact vectorized top-64 scores for a fixed feature metric."""
    x = np.asarray(x, np.float32)
    mem = x[mids]
    q = x[qids]
    if metric == 'cos':
        mem = mem / np.maximum(np.linalg.norm(mem, axis=1, keepdims=True), 1e-8)
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)
    elif metric != 'std_l2':
        raise ValueError(metric)
    mem_norm = np.einsum('ij,ij->i', mem, mem) if metric == 'std_l2' else None
    if len(mids) < k: raise ValueError(f'need at least k={k} rows')
    cand = np.empty((len(qids), k), np.int64)
    score_out = np.empty((len(qids), k), np.float32)
    mt = np.asarray(rows['task'])[mids]
    me = np.asarray(rows['episode'])[mids]
    for start in range(0, len(qids), batch_rows):
        end = min(start + batch_rows, len(qids))
        qb = q[start:end]
        dot = qb @ mem.T
        if metric == 'std_l2':
            qn = np.einsum('ij,ij->i', qb, qb)
            scores = -(qn[:, None] + mem_norm[None, :] - 2 * dot)
        else:
            scores = dot
        for j, qrow in enumerate(qids[start:end]):
            allowed = mt != int(rows['task'][qrow]) if exclusion == 'task' else me != int(rows['episode'][qrow])
            values = np.asarray(scores[j], np.float32).copy()
            values[~allowed] = -np.inf
            ix = np.lexsort((mids, -values))[:k]
            if len(ix) != k or not np.isfinite(values[ix]).all():
                raise ValueError(f'less than {k} eligible rows for {qrow}/{exclusion}')
            cand[start+j], score_out[start+j] = mids[ix], values[ix]
    return cand, score_out


def _save_key_and_top8(out: Path, name: str, exclusion: str, qids: np.ndarray,
                       cand: np.ndarray, score: np.ndarray, rows: dict,
                       qsplit: np.ndarray, description: str) -> Path:
    dest = out / 'keys' / f'{name}_{exclusion}.npz'
    if dest.exists():
        raise FileExistsError(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Raw [10,32] chunks are retained so Stage 3 can consume the same examples.
    acts = np.asarray(rows['actions'][cand[:, :8]], np.float32)
    np.savez_compressed(dest, query_rows=qids.astype(np.int64), cand_rows=cand.astype(np.int64),
                        score=score.astype(np.float32), exclusion=np.asarray(exclusion),
                        description=np.asarray(description), query_split=qsplit.astype('U4'),
                        query_task=np.asarray(rows['task'][qids], np.int32),
                        query_episode=np.asarray(rows['episode'][qids], np.int32),
                        cand_actions=acts.astype(np.float32))
    return dest


def _load_existing_keys(out: Path, qids: np.ndarray, qsplit: np.ndarray, rows: dict) -> list[dict]:
    result = []
    for exclusion in ('task', 'episode'):
        for name in VLM_METHODS:
            path = AUDIT / 'vlm-full' / 'keys' / f'{name}_{exclusion}.npz'
            if not path.exists():
                continue
            z = np.load(path, allow_pickle=False)
            if not np.array_equal(z['query_rows'], qids):
                raise AssertionError(f'query row mismatch in {path}')
            cand, score = np.asarray(z['cand_rows']), np.asarray(z['score'])
            target = _save_key_and_top8(out, name, exclusion, qids, cand, score, rows, qsplit,
                                         f'Reused frozen score-audit VLM key: {name}.')
            result.append({'name': name, 'exclusion': exclusion, 'cand': cand, 'score': score,
                           'path': str(target), 'source': str(path), 'family': 'VLM'})
        for name in REFERENCE_METHODS:
            path = AUDIT / 'nonvlm-full' / 'keys' / f'{name}_{exclusion}.npz'
            if not path.exists():
                raise FileNotFoundError(path)
            z = np.load(path, allow_pickle=False)
            if not np.array_equal(z['query_rows'], qids):
                raise AssertionError(f'query row mismatch in {path}')
            cand, score = np.asarray(z['cand_rows']), np.asarray(z['score'])
            desc = ('π0.5予測行動に依存する query key と保存実演行動との負MSE。' if name.startswith('pi05')
                    else 'VLM pooled/state からの frozen codec prior b̂ の cosine。')
            target = _save_key_and_top8(out, name, exclusion, qids, cand, score, rows, qsplit, desc)
            result.append({'name': name, 'exclusion': exclusion, 'cand': cand, 'score': score,
                           'path': str(target), 'source': str(path), 'family': 'reference'})
    return result


def _direct_keys(out: Path, qids: np.ndarray, qsplit: np.ndarray, rows: dict,
                 wrist: np.ndarray, *, batch_rows: int = 64) -> tuple[list[dict], dict]:
    train_ids = np.flatnonzero(~rows['memory'] & ~rows['validation'])
    feature_specs = {}
    x0 = learned_key.wrist_key_inputs(wrist, rows, include_history=False)
    x1 = learned_key.wrist_key_inputs(wrist, rows, include_history=True)
    feature_specs['wrist_state'] = x0
    feature_specs['wrist_state_history'] = x1
    direct = []
    prepared = {}
    mids = np.flatnonzero(rows['memory'])
    for feature_name, raw in feature_specs.items():
        x, mean, std = standardized(raw, train_ids)
        prepared[feature_name] = {'x': x, 'mean': mean, 'std': std}
        np.savez_compressed(out / f'{feature_name}_normalization.npz', mean=mean, std=std,
                            fit_rows=train_ids)
        for metric, suffix in (('cos', 'cos'), ('std_l2', 'std_l2')):
            for exclusion in ('task', 'episode'):
                cand, score = _rank_matrix(x, qids, mids, rows, exclusion, metric, batch_rows=batch_rows)
                name = f'{feature_name}_{suffix}'
                path = _save_key_and_top8(out, name, exclusion, qids, cand, score, rows, qsplit,
                    f'{feature_name}: wrist token mean + proprio' +
                    (' + previous wrist/state/action chunk and availability flag' if feature_name.endswith('history') else '') +
                    f'; train-standardized {metric} score, no labels or future action at query time.')
                direct.append({'name': name, 'exclusion': exclusion, 'cand': cand, 'score': score,
                               'path': str(path), 'family': 'wrist+state' if feature_name == 'wrist_state' else 'wrist+state+history'})
    return direct, prepared


def _learned_keys(out: Path, qids: np.ndarray, qsplit: np.ndarray, rows: dict,
                  prepared: dict, *, pair_count: int, epochs: int, threads: int,
                  batch_rows: int = 64) -> tuple[list[dict], dict]:
    mids = np.flatnonzero(rows['memory'])
    learned = []; fit_summaries = {}
    for feature_name in ('wrist_state', 'wrist_state_history'):
        model_dir = out / f'{feature_name}_learned'
        fit, embedding = learned_key.fit_wrist_contrastive(
            prepared[feature_name]['x'], rows, out=model_dir, run=RUN,
            epochs=epochs, pair_count=pair_count, threads=threads)
        fit_summaries[feature_name] = fit
        embed_file = out / f'{feature_name}_embedding.npy'
        if embed_file.exists(): raise FileExistsError(embed_file)
        np.save(embed_file, embedding.astype(np.float32), allow_pickle=False)
        for exclusion in ('task', 'episode'):
            cand, score = _rank_matrix(embedding, qids, mids, rows, exclusion, 'cos', batch_rows=batch_rows)
            name = f'{feature_name}_contrastive'
            path = _save_key_and_top8(out, name, exclusion, qids, cand, score, rows, qsplit,
                f'64D contrastive key from {feature_name}; trained on non-memory/non-validation rows using cross-task action-near positives; threshold selected on Dev only.')
            learned.append({'name': name, 'exclusion': exclusion, 'cand': cand, 'score': score,
                            'path': str(path), 'family': 'learned wrist key', 'selected_threshold': fit['selected_positive_action_mse_max']})
    (out / 'learned_fit.json').write_text(json.dumps(fit_summaries, ensure_ascii=False, indent=2) + '\n')
    return learned, fit_summaries


def _recall_metrics(items: list[dict], qids: np.ndarray, qsplit: np.ndarray,
                    oracle: np.ndarray, oracle_qids: np.ndarray, memids: np.ndarray,
                    rows: dict, *, bootstraps: int = 1000) -> list[dict]:
    qmap = {int(q): i for i, q in enumerate(oracle_qids)}
    mcols = {int(m): i for i, m in enumerate(memids)}
    qoi = np.asarray([qmap[int(q)] for q in qids], np.int64)
    mt = np.asarray(rows['task'])[memids]; me = np.asarray(rows['episode'])[memids]
    qt = np.asarray(rows['task'])[qids]; qe = np.asarray(rows['episode'])[qids]
    split_results = []
    for item in items:
        cand = item['cand']
        # Map global candidate row IDs to columns of the full oracle matrix once.
        ccols = np.asarray([[mcols[int(x)] for x in row] for row in cand], np.int64)
        dists = np.asarray(oracle[qoi[:, None], ccols], np.float32)
        for split in ('dev', 'eval'):
            qmask = qsplit == split
            for exclusion in ('task', 'episode'):
                if item['exclusion'] != exclusion: continue
                allowed_presence = np.zeros(len(qids), bool)
                for qi, oi in enumerate(qoi):
                    allowed = mt != qt[qi] if exclusion == 'task' else me != qe[qi]
                    allowed_presence[qi] = np.any(np.asarray(oracle[oi], np.float32)[allowed] <= .01)
                for tau in TAUS:
                    present = np.zeros(len(qids), bool); hit1 = np.zeros(len(qids), bool)
                    recall1 = np.zeros(len(qids), np.float32); recall8 = np.zeros(len(qids), np.float32); recall64 = np.zeros(len(qids), np.float32)
                    for qi, oi in enumerate(qoi):
                        allowed = mt != qt[qi] if exclusion == 'task' else me != qe[qi]
                        poscols = np.flatnonzero(allowed & (np.asarray(oracle[oi], np.float32) <= tau))
                        if not len(poscols): continue
                        present[qi] = True
                        posrows = {int(memids[c]) for c in poscols}
                        ids = list(map(int, cand[qi]))
                        recall1[qi] = float(ids[0] in posrows) / len(posrows)
                        recall8[qi] = len(posrows.intersection(ids[:8])) / len(posrows)
                        recall64[qi] = len(posrows.intersection(ids[:64])) / len(posrows)
                        hit1[qi] = ids[0] in posrows
                    selected = qmask & present
                    rec = {'method': item['name'], 'family': item['family'], 'exclusion': exclusion,
                           'split': split, 'tau': tau, 'queries': int(qmask.sum()),
                           'oracle_presence_rate': float(present[qmask].mean()) if qmask.any() else None,
                           'recall_at_1': float(recall1[selected].mean()) if selected.any() else None,
                           'recall_at_8': float(recall8[selected].mean()) if selected.any() else None,
                           'recall_at_64': float(recall64[selected].mean()) if selected.any() else None,
                           'top1_hit_rate_all_queries': float(hit1[qmask].mean()) if qmask.any() else None,
                           'top1_hit_rate_given_oracle_exists': float(hit1[selected].mean()) if selected.any() else None,
                           'top1_raw_action_mse_mean': float(dists[qmask, 0].mean()) if qmask.any() else None,
                           'queries_with_oracle': int(selected.sum())}
                    split_results.append(rec)
    return split_results


def _eval_raw_drafts(out: Path, items: list[dict], rows: dict, *, batch_size: int = 4,
                     condition_batch: int = 8, smoke_rows: int | None = None) -> dict:
    """Run the memory-gap bridge with selected stored raw action chunks."""
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from openpi.shared.nnx_utils import PathRegex, module_jit
    gap = _gap_module()
    core_spec = importlib.util.spec_from_file_location('pi05_memory_core_run_4b', PI05 / 'run.py')
    policy = importlib.util.module_from_spec(core_spec)
    assert core_spec.loader is not None
    core_spec.loader.exec_module(policy)
    source = np.load(GAP / 'predictions_and_errors.npz', allow_pickle=False)
    eval_ids = np.asarray(source['row_ids'], np.int64)
    if smoke_rows is not None: eval_ids = eval_ids[:smoke_rows]
    byrow = {int(q): j for j, q in enumerate(np.flatnonzero(rows['validation']))}
    conditions = []
    for item in items:
        for t0 in T0S:
            conditions.append({'name': item['name'], 'exclusion': item['exclusion'], 't0': t0,
                               'cand_rows': np.asarray(item['cand'])[:, 0]})
    # Reuse the already computed zero-draft output from the exact memory-gap evaluation.
    zero_cond = {}
    started = time.monotonic()
    best = json.loads((RUN / 'best_bridge.json').read_text())
    if int(best['steps']) != 1500: raise ValueError(f'unexpected bridge checkpoint {best}')
    base, _ = policy.load_policy(gap.OFFICIAL)
    baseline = module_jit(base.sample_actions, static_argnames=('num_steps',))
    probe = eval_ids[:min(4, len(eval_ids))]
    eps = jnp.asarray(np.stack([gap._noise(jax, gap.NOISE_SEEDS[0], int(i)) for i in probe]))
    pred_orig = np.asarray(baseline(jax.random.key(0), policy.obs_batch(rows, probe), num_steps=10, noise=eps))
    filt = nnx.All(nnx.Param, PathRegex(r'.*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*'))
    with ocp.PyTreeCheckpointer() as ckpt:
        partial = ckpt.restore(Path(best['path']))
    state = nnx.state(base, filt)
    state.replace_by_pure_dict(jax.tree.map(jnp.asarray, partial)); nnx.update(base, state)
    pred = np.asarray(baseline(jax.random.key(0), policy.obs_batch(rows, probe), num_steps=10, noise=eps))
    fallback_equal = bool(np.array_equal(pred, pred_orig))
    if not fallback_equal: raise AssertionError('corrected checkpoint changed normal fallback')
    sample_bridge = module_jit(base.sample_bridge, static_argnames=('num_steps',))

    method_names = [f"{c['name']}|{c['exclusion']}|t0={c['t0']:.1f}" for c in conditions]
    err = np.empty((len(conditions), len(gap.NOISE_SEEDS), len(eval_ids)), np.float32)
    truth = np.asarray(rows['actions'][eval_ids, :, :7], np.float32)
    # Match Stage 2's Dev-only q80 threshold and apply it to Eval normal-policy errors.
    hard_threshold = float(json.loads((GAP / 'protocol.json').read_text())['dev_threshold']['dev_episode_normal_mse_q80'])
    normal_ix = int(np.flatnonzero(source['condition_names'] == 'A_normal')[0])
    baseline_err = np.mean(source['row_mse'][normal_ix], axis=0)
    source_lookup_hard = {int(q): i for i, q in enumerate(source['row_ids'])}
    hard_mask = np.asarray([baseline_err[source_lookup_hard[int(q)]] >= hard_threshold for q in eval_ids], bool)
    oracle_meta = np.load(AUDIT / 'oracle_meta.npz', allow_pickle=False)
    oracle = np.load(AUDIT / 'oracle_action_mse.npy', mmap_mode='r', allow_pickle=False)
    oracle_q = {int(q): i for i, q in enumerate(oracle_meta['query_rows'])}
    oracle_m = {int(m): i for i, m in enumerate(oracle_meta['memory_rows'])}
    q_oracle_ix = np.asarray([oracle_q[int(q)] for q in eval_ids], np.int64)
    for seed_ix, seed in enumerate(gap.NOISE_SEEDS):
        for bstart in range(0, len(eval_ids), batch_size):
            qrows = eval_ids[bstart:bstart + batch_size]
            b = len(qrows)
            obs = policy.obs_batch(rows, qrows)
            eps0 = np.stack([gap._noise(jax, int(seed), int(q)) for q in qrows])
            for cstart in range(0, len(conditions), condition_batch):
                cs = conditions[cstart:cstart + condition_batch]
                # Keep a fixed compile shape for the last chunk.
                actual_c = len(cs)
                if actual_c < condition_batch:
                    cs = cs + [conditions[0]] * (condition_batch - actual_c)
                drafts = np.stack([rows['actions'][c['cand_rows'][[byrow[int(q)] for q in qrows]]] for c in cs], axis=0)
                # [C,B,10,32] -> [B*C,10,32], aligned with repeated observation rows.
                draft_batch = jnp.asarray(drafts.transpose(1, 0, 2, 3).reshape(b * condition_batch, 10, 32))
                obs_batch = jax.tree.map(lambda x: jnp.repeat(x, condition_batch, axis=0), obs)
                t0_batch = jnp.asarray(np.tile(np.asarray([c['t0'] for c in cs], np.float32), b))
                noise_batch = jnp.asarray(np.repeat(eps0[:, None], condition_batch, axis=1).reshape(b * condition_batch, 10, 32))
                key = jax.random.fold_in(jax.random.key(int(seed)), int(bstart))
                output = np.asarray(sample_bridge(key, obs_batch, draft_batch, t0_batch,
                                                 num_steps=10, noise=noise_batch))[:, :, :7]
                output = output.reshape(b, condition_batch, 10, 7).transpose(1, 0, 2, 3)
                delta = output[:actual_c] - truth[bstart:bstart+b][None]
                err[cstart:cstart+actual_c, seed_ix, bstart:bstart+b] = np.mean(delta**2, axis=(2, 3))
            if bstart % 128 == 0 or bstart + b == len(eval_ids):
                with (out / 'progress.jsonl').open('a') as f:
                    f.write(json.dumps({'stage':'raw_draft_inference','seed':int(seed),
                                        'rows_done':int(bstart+b),'rows':int(len(eval_ids)),
                                        'elapsed_s':time.monotonic()-started})+'\n')
    # Pull the exact matching zero predictions from the previous Stage-2 run.
    zero_errors = np.empty((len(T0S), len(gap.NOISE_SEEDS), len(eval_ids)), np.float32)
    source_rows = np.asarray(source['row_ids'])
    source_lookup = {int(q): i for i, q in enumerate(source_rows)}
    source_cond_idx = {float(t0): int(np.flatnonzero(source['condition_names'] == f'zero_t0_{t0:.1f}')[0]) for t0 in T0S}
    for ti, t0 in enumerate(T0S):
        ix = np.asarray([source_lookup[int(q)] for q in eval_ids], np.int64)
        zero_errors[ti] = source['row_mse'][source_cond_idx[t0]][:, ix]
    metrics = []
    for ci, condition in enumerate(conditions):
        t0 = condition['t0']; zero = zero_errors[T0S.index(t0)]
        v = err[ci]
        cand_rows = condition['cand_rows'][np.asarray([byrow[int(q)] for q in eval_ids])]
        cand_cols = np.asarray([oracle_m[int(m)] for m in cand_rows], np.int64)
        raw_dist = np.asarray(oracle[q_oracle_ix, cand_cols], np.float32)
        record = {'method': condition['name'], 'exclusion': condition['exclusion'], 't0': t0,
                  'rows': int(len(eval_ids)), 'hard_rows': int(hard_mask.sum()),
                  'hard_threshold': hard_threshold, 'raw_draft_action_mse_all': float(raw_dist.mean()),
                  'final_mse_all': float(v.mean()), 'median_final_mse_all': float(np.median(v)),
                  'zero_final_mse_all': float(zero.mean()),
                  'fraction_better_than_zero_all': float(np.mean(v < zero))}
        if hard_mask.any():
            record.update(raw_draft_action_mse_hard=float(raw_dist[hard_mask].mean()),
                          final_mse_hard=float(v[:, hard_mask].mean()),
                          median_final_mse_hard=float(np.median(v[:, hard_mask])),
                          zero_final_mse_hard=float(zero[:, hard_mask].mean()),
                          fraction_better_than_zero_hard=float(np.mean(v[:, hard_mask] < zero[:, hard_mask])))
        else:
            record.update(raw_draft_action_mse_hard=None, final_mse_hard=None,
                          median_final_mse_hard=None, zero_final_mse_hard=None,
                          fraction_better_than_zero_hard=None)
        metrics.append(record)
    error_path = out / ('raw_draft_final_mse_smoke.npz' if smoke_rows is not None else 'raw_draft_final_mse.npz')
    if error_path.exists(): raise FileExistsError(error_path)
    np.savez_compressed(error_path, eval_rows=eval_ids, condition_names=np.asarray(method_names),
                        noise_seeds=np.asarray(gap.NOISE_SEEDS), final_mse=err, zero_final_mse=zero_errors,
                        zero_t0s=np.asarray(T0S), hard_mask=hard_mask,
                        hard_threshold=np.asarray(hard_threshold), raw_draft_action_mse=np.asarray([
                            np.asarray(oracle[q_oracle_ix, np.asarray([oracle_m[int(m)] for m in c['cand_rows'][[byrow[int(q)] for q in eval_ids]]], np.int64)], np.float32)
                            for c in conditions], np.float32))
    return {'eval_rows': eval_ids, 'conditions': metrics, 'fallback_equal': fallback_equal,
            'seconds': time.monotonic()-started, 'hard_rows': int(hard_mask.sum()),
            'hard_threshold': hard_threshold, 'zero_errors': zero_errors, 'error_file': str(error_path)}


def inference_only(out: Path, *, smoke_rows: int | None = None) -> dict:
    rows = keys.load_rows(RUN)
    items = []
    for path in sorted((out / 'keys').glob('*.npz')):
        z = np.load(path, allow_pickle=False)
        stem = path.stem
        exclusion = 'task' if stem.endswith('_task') else 'episode'
        name = stem[:-(len(exclusion)+1)]
        family = ('VLM' if name in VLM_METHODS else
                  'reference' if name in REFERENCE_METHODS else
                  'learned wrist key' if name.endswith('_contrastive') else
                  'wrist+state' if name.startswith('wrist_state_') else 'wrist+state+history')
        items.append({'name':name,'exclusion':exclusion,'cand':z['cand_rows'],'score':z['score'],
                      'path':str(path),'family':family})
    result = _eval_raw_drafts(out, items, rows, smoke_rows=smoke_rows)
    if smoke_rows is None:
        results_file = out / 'results.json'
        payload = json.loads(results_file.read_text())
        payload['final_raw_draft'] = {'eval_rows': int(len(result['eval_rows'])),
                                      'conditions': result['conditions'],
                                      'fallback_equal': result['fallback_equal'],
                                      'seconds': result['seconds'], 'hard_rows': result['hard_rows'],
                                      'hard_threshold': result['hard_threshold'],
                                      'error_file': result['error_file']}
        results_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + '\n')
    return result


def build(out: Path = OUT, *, pair_count: int = 10_000_000, epochs: int = 4,
          threads: int = 8, infer: bool = True, smoke_rows: int | None = None) -> dict:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'output is not empty; refusing overwrite: {out}')
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    rows = keys.load_rows(RUN)
    meta = np.load(AUDIT / 'oracle_meta.npz', allow_pickle=False)
    qids = np.asarray(meta['query_rows'], np.int64); mids = np.asarray(meta['memory_rows'], np.int64)
    qsplit = np.asarray(meta['query_split']).astype('U4')
    assert np.array_equal(qids, np.flatnonzero(rows['validation']))
    oracle = np.load(AUDIT / 'oracle_action_mse.npy', mmap_mode='r', allow_pickle=False)
    _save_existing = _load_existing_keys(out, qids, qsplit, rows)
    feature_file = out / 'wrist_token_mean.npy'
    wrist = learned_key.extract_wrist_token_mean(RUN, rows, feature_file)
    direct, prepared = _direct_keys(out, qids, qsplit, rows, wrist)
    learned, fit_summaries = _learned_keys(out, qids, qsplit, rows, prepared,
                                           pair_count=pair_count, epochs=epochs, threads=threads)
    items = _save_existing + direct + learned
    recall = _recall_metrics(items, qids, qsplit, oracle, qids, mids, rows)
    eval_final = None
    if infer:
        eval_final = _eval_raw_drafts(out, items, rows, smoke_rows=smoke_rows)
    result = {'queries': int(len(qids)), 'memory_rows': int(len(mids)),
              'dev_queries': int(np.sum(qsplit == 'dev')), 'eval_queries': int(np.sum(qsplit == 'eval')),
              'methods': [{'name':x['name'],'family':x['family'],'exclusion':x['exclusion'],'path':x['path']} for x in items],
              'learned_fit': fit_summaries, 'recall': recall,
              'final_raw_draft': eval_final, 'seconds': time.monotonic()-started,
              'oracle_source': str(AUDIT / 'oracle_action_mse.npy'), 'output': str(out)}
    (out / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True) + '\n')
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=OUT)
    ap.add_argument('--pair-count', type=int, default=10_000_000)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--no-inference', action='store_true')
    ap.add_argument('--smoke-rows', type=int, default=None)
    ap.add_argument('--inference-only', action='store_true')
    args = ap.parse_args()
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '.20')
    if args.inference_only:
        result = inference_only(args.output, smoke_rows=args.smoke_rows)
        print(json.dumps({'eval_rows':len(result['eval_rows']),'conditions':len(result['conditions']),
                          'fallback_equal':result['fallback_equal'],'seconds':result['seconds'],
                          'error_file':result['error_file']}, indent=2))
        return
    result = build(args.output, pair_count=args.pair_count, epochs=args.epochs,
                   threads=args.threads, infer=not args.no_inference, smoke_rows=args.smoke_rows)
    print(json.dumps({'output':result['output'],'methods':len(result['methods']),
                      'queries':result['queries'],'seconds':result['seconds']},indent=2))


if __name__ == '__main__': main()
