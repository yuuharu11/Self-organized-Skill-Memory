#!/usr/bin/env python3
"""Inference-only attention/input dependence audit for KEY2 shared latent keys.

The original model is never patched: attention probabilities are reconstructed
from the trained cross-attention projections in a parallel observer, while the
model's public forward remains the source of representations and scores.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import key_shared_latent as key2

OUT_DEFAULT = Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/key-attn-diag-01')
WEIGHT_ROOT = Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/key-shared-01/attempt-27-production')
CONFIG_PATH = WEIGHT_ROOT / 'selected_configs.json'
VJEPA_EVAL = WEIGHT_ROOT / 'vjepa/libero90_eval'
MODALITIES = ('vlm_image', 'vlm_language', 'vjepa', 'history', 'state')
MODES = ('e_vlm_h', 'e_vj_vlm_h', 'e_h')


def shared_memory(model, part, device):
    """Mirror SharedLatentEncoder.forward's memory assembly exactly."""
    mem, masks, spans = [], [], {}
    cursor = 0
    if model.use_vlm:
        tok = torch.as_tensor(part['tokens'], device=device, dtype=torch.float32)
        grp = torch.as_tensor(part['groups'], device=device, dtype=torch.long)
        valid = grp >= 0
        mem.append(model.vlm_projection(tok) + model.group_embedding(grp.clamp(0, 5)))
        masks.append(~valid)
        spans['vlm_image'] = (cursor, cursor + tok.shape[1], grp)
        cursor += tok.shape[1]
    if model.use_vjepa:
        tok = torch.as_tensor(part['vjepa_tokens'], device=device, dtype=torch.float32)
        grp = torch.as_tensor(part['vjepa_groups'], device=device, dtype=torch.long)
        valid = grp >= 0
        mem.append(model.vjepa_projection(tok) + model.group_embedding((grp + 4).clamp(0, 5)))
        masks.append(~valid)
        spans['vjepa'] = (cursor, cursor + tok.shape[1], grp)
        cursor += tok.shape[1]
    if model.use_history:
        hist = torch.as_tensor(part['history'], device=device, dtype=torch.float32)
        hm = torch.as_tensor(part['history_mask'], device=device, dtype=torch.bool)
        hin = torch.cat([hist, hm.float().unsqueeze(-1)], -1)
        mem.append(model.history_projection(hin) + model.history_position)
        masks.append(~hm)
        spans['history'] = (cursor, cursor + hist.shape[1], hm)
        cursor += hist.shape[1]
    if model.use_state:
        state = torch.as_tensor(part['state'], device=device, dtype=torch.float32)
        mem.append(model.state_projection(state).unsqueeze(1) + model.state_type)
        masks.append(torch.zeros((len(state), 1), dtype=torch.bool, device=device))
        spans['state'] = (cursor, cursor + 1, None)
        cursor += 1
    return torch.cat(mem, 1), torch.cat(masks, 1), spans


def cross_attention_weights(model, part, device):
    """Return pre-dropout attention weights [B,H,Q,T] without changing forward."""
    memory, padding, _ = shared_memory(model, part, device)
    block = model.cross
    b = len(memory)
    q = block.q_norm(model.latent.expand(b, -1, -1))
    kv = block.kv_norm(memory)
    mha = block.attn
    dim = mha.embed_dim
    qw, kw, _vw = mha.in_proj_weight.split(dim, dim=0)
    if mha.in_proj_bias is None:
        qb = kb = None
    else:
        qb, kb, _vb = mha.in_proj_bias.split(dim, dim=0)
    q = F.linear(q, qw, qb)
    k = F.linear(kv, kw, kb)
    heads = mha.num_heads
    dh = dim // heads
    q = q.view(b, -1, heads, dh).transpose(1, 2)
    k = k.view(b, -1, heads, dh).transpose(1, 2)
    logits = torch.matmul(q, k.transpose(-1, -2)) / (dh ** .5)
    logits = logits.masked_fill(padding[:, None, None, :], torch.finfo(logits.dtype).min)
    return torch.softmax(logits, dim=-1)


def aggregate_attention(weights, part, use_vlm, use_vjepa, use_history, use_state):
    """Aggregate per modality; values sum to one per head/query token."""
    w = weights.float()
    b, h, q, _ = w.shape
    result = torch.zeros((b, h, q, len(MODALITIES)), device=w.device)
    counts = torch.zeros((b, len(MODALITIES)), device=w.device)
    idx = 0
    if use_vlm:
        g = torch.as_tensor(part['groups'], device=w.device, dtype=torch.long)
        image = (g == 0) | (g == 1)
        lang = g == 2
        result[..., 0] = (w[..., idx:idx + g.shape[1]] * image[:, None, None, :]).sum(-1)
        result[..., 1] = (w[..., idx:idx + g.shape[1]] * lang[:, None, None, :]).sum(-1)
        counts[:, 0] = image.sum(-1); counts[:, 1] = lang.sum(-1)
        idx += g.shape[1]
    if use_vjepa:
        n = 128
        result[..., 2] = w[..., idx:idx+n].sum(-1); counts[:, 2] = n; idx += n
    if use_history:
        n = 10
        result[..., 3] = w[..., idx:idx+n].sum(-1)
        hm = torch.as_tensor(part['history_mask'], device=w.device, dtype=torch.bool)
        counts[:, 3] = hm.sum(-1); idx += n
    if use_state:
        result[..., 4] = w[..., idx]
        counts[:, 4] = 1
    return result, counts


def per_input_token_mass(mean_mass, mean_valid_tokens):
    """Convert mean modality mass to pooled mean mass per valid source token."""
    mass = np.asarray(mean_mass, np.float64)
    counts = np.asarray(mean_valid_tokens, np.float64)
    if mass.ndim != 3 or counts.shape != (mass.shape[-1],):
        raise ValueError(f'expected mass [heads,queries,modalities] and counts [modalities], got {mass.shape}, {counts.shape}')
    return mass / np.maximum(counts[None, None, :], 1.0)


def latent_collapse(z: np.ndarray) -> dict:
    z = np.asarray(z, np.float64)
    cos = np.einsum('nkd,njd->nkj', z, z)
    tri = cos[:, np.triu_indices(z.shape[1], 1)[0], np.triu_indices(z.shape[1], 1)[1]]
    singular = np.linalg.svd(z, compute_uv=False)
    p = singular / np.maximum(singular.sum(-1, keepdims=True), 1e-15)
    erank = np.exp(-(p * np.log(np.maximum(p, 1e-15))).sum(-1))
    return {'mean_pairwise_cosine': float(tri.mean()), 'pairwise_cosine_std': float(tri.std()),
            'effective_rank_mean': float(erank.mean()), 'effective_rank_std': float(erank.std()),
            'singular_values_mean': singular.mean(0).tolist()}


def apply_query_ablation(split, name, kind, rng):
    """Return copied query split with one present modality zeroed or row-swapped."""
    out = dict(split)
    field = {'vlm': 'tokens', 'vjepa': 'vjepa_tokens', 'history': 'history', 'state': 'state'}[name]
    if field not in out:
        return None
    x = np.asarray(out[field]).copy()
    if kind == 'zero':
        x[...] = 0
    elif kind == 'swap':
        if len(x) < 2:
            raise ValueError('swap requires at least two rows')
        ix = rng.permutation(len(x))
        if np.any(ix == np.arange(len(x))):
            ix = np.roll(np.arange(len(x)), 1)
        x = x[ix]
        if name == 'history':
            out['history_mask'] = np.asarray(out['history_mask'])[ix].copy()
        if name == 'vlm':
            # Swap token content while retaining the recipient's positional groups/mask.
            pass
        if name == 'vjepa':
            pass
    else:
        raise ValueError(f'unknown ablation {kind}')
    out[field] = x
    return out


def _load_data(progress):
    run_root = Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/run-01')
    missing = Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/key-shared-01/attempt-02/prefix-missing')
    seen = Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/seg-context-01/seg5-cache')
    unseen = Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/seg-context-01/unseen-stride5')
    bank, bank_input, _, _, _, learned, eval_rows, dev = key2._build_key2_queries(
        run_root, missing, seen, unseen, VJEPA_EVAL, progress_callback=progress)
    if len(bank['length']) != 4516 or len(learned['key']) != 621 or len(eval_rows['key']) != 619 or len(dev['key']) != 217:
        raise AssertionError('fixed KEY2 row counts changed')
    return bank, bank_input, learned, eval_rows, dev


def _load_model(mode, checkpoint, device):
    model = key2._new_model(mode, token_count=5).to(device)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _encode(model, part, device, batch=16):
    reps = []
    with torch.no_grad():
        for s in range(0, len(part['key']), batch):
            ix = np.arange(s, min(len(part['key']), s + batch))
            pp = key2._pack_part(part, ix)
            reps.append(key2._model_batch_encode(model, model.input_mode, pp, device).cpu().numpy())
    return np.concatenate(reps)


def _encode_attn_profile(model, part, device, *, name):
    totals = np.zeros((8, 5, len(MODALITIES)), np.float64)
    counts = np.zeros((len(MODALITIES),), np.float64)
    # Average mass per row/head/query token, and separately divide by source token counts.
    for s in range(0, len(part['key']), 16):
        ix = np.arange(s, min(len(part['key']), s + 16))
        pp = key2._pack_part(part, ix)
        with torch.no_grad():
            w = cross_attention_weights(model, pp, device)
            mass, cnt = aggregate_attention(w, pp, model.use_vlm, model.use_vjepa,
                                            model.use_history, model.use_state)
        totals += mass.sum(0).cpu().numpy()
        counts += cnt.sum(0).cpu().numpy()
    denom = len(part['key'])
    mass = totals / denom
    mean_counts = counts / denom
    # `mass` is the mean attention mass per row.  Divide by the mean number of
    # valid source tokens, not by their sum over the whole split.
    per_token = per_input_token_mass(mass, mean_counts)
    return {'rows': int(denom), 'total_attention': mass.tolist(), 'per_input_token_attention': per_token.tolist(),
            'modalities': list(MODALITIES), 'heads': 8, 'latent_tokens': 5, 'cross_layer': 0,
            'input_token_counts_mean': mean_counts.tolist()}


def _scores(model, mode, bank_input, split, device, bank, keyreps=None):
    if keyreps is None:
        keyreps = _encode(model, bank_input, device)
    q = _encode(model, split, device)
    scores = np.einsum('qkd,nkd->qn', q, keyreps, optimize=True) / q.shape[1]
    scale = float(key2._logit_scale_parameter(model).detach().exp().clamp(max=key2.LOGIT_SCALE_MAX).cpu())
    scores *= scale
    candidates = key2._candidate_window_actions(np.arange(len(bank['length'])), bank)
    errors = key2._arm6_candidate_mse(split['actions'], candidates)
    return scores.astype(np.float32), errors, q, keyreps


def _ablation_metrics(model, mode, bank, bank_input, split, device, keyreps, errors, base_order):
    result = {}
    for name in ('vlm', 'vjepa', 'history', 'state'):
        if (name == 'vlm' and not model.use_vlm) or (name == 'vjepa' and not model.use_vjepa) or (name == 'history' and not model.use_history) or (name == 'state' and not model.use_state):
            result[name] = {'applicable': False}; continue
        result[name] = {}
        for kind in ('zero', 'swap'):
            altered = apply_query_ablation(split, name, kind, np.random.default_rng(4101))
            q = _encode(model, altered, device)
            scores = np.einsum('qkd,nkd->qn', q, keyreps, optimize=True) / q.shape[1]
            scores *= float(key2._logit_scale_parameter(model).detach().exp().clamp(max=key2.LOGIT_SCALE_MAX).cpu())
            order = np.argmax(scores, axis=1)
            result[name][kind] = {'applicable': True, 'top1_agreement': float(np.mean(order == base_order)),
                                  'top1_arm6_mse': float(errors[np.arange(len(errors)), order].mean()),
                                  'changed_fraction': float(np.mean(order != base_order))}
    return result


def run(output: Path = OUT_DEFAULT, *, device='cuda:0', memory_fraction=.15, progress_file=None):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'refusing to overwrite nonempty output directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    if device.startswith('cuda'):
        if not torch.cuda.is_available(): raise RuntimeError('CUDA requested but not available')
        torch.cuda.set_per_process_memory_fraction(float(memory_fraction), int(device.split(':')[1]) if ':' in device else 0)
    logp = Path(progress_file) if progress_file else output / 'progress.jsonl'
    def progress(stage, **fields):
        with logp.open('a') as f: f.write(json.dumps({'time': time.time(), 'stage': stage, **fields}) + '\n')
    bank, bank_input, learned, unseen, dev = _load_data(progress)
    configs = json.loads(CONFIG_PATH.read_text())['selected']
    result = {'status': 'running', 'row_counts': {'keys': len(bank['length']), 'learned': len(learned['key']), 'unseen_eval': len(unseen['key']), 'dev': len(dev['key'])}, 'conditions': {},
              'method': 'parallel attention observer; original module forward unchanged', 'modalities': list(MODALITIES)}
    arrays = {}
    for mode in MODES:
        checkpoint = configs[mode]['checkpoint']
        model = _load_model(mode, checkpoint, device)
        # Exact forward parity check: observer has no hooks or parameter mutation.
        probe = key2._pack_part(learned, np.arange(min(3, len(learned['key']))))
        with torch.no_grad():
            a = key2._model_batch_encode(model, mode, probe, device)
            _ = cross_attention_weights(model, probe, device)
            b = key2._model_batch_encode(model, mode, probe, device)
        maxdiff = float((a - b).abs().max().cpu())
        if maxdiff != 0.0: raise AssertionError(f'observer changed forward output: {maxdiff}')
        entry = {'checkpoint': checkpoint, 'forward_max_abs_diff': maxdiff, 'splits': {}}
        keyrep = _encode(model, bank_input, device)
        bank_coll = latent_collapse(keyrep)
        entry['key_latent_collapse'] = bank_coll
        keyvec = keyrep.mean(1)
        entry['key_task_clustering'] = _task_clustering(keyvec, bank['task'])
        for split_name, split in [('learned', learned), ('unseen_eval', unseen), ('dev', dev), ('key', bank_input)]:
            progress('condition_split', mode=mode, split=split_name)
            z = keyrep if split_name == 'key' else _encode(model, split, device)
            entry['splits'][split_name] = {
                'latent_collapse': latent_collapse(z),
                'attention': _encode_attn_profile(model, split, device, name=split_name),
            }
            if split_name in ('learned', 'unseen_eval'):
                scores, errors, _, _ = _scores(model, mode, bank_input, split, device, bank, keyrep)
                order = np.argmax(scores, axis=1)
                top = errors[np.arange(len(errors)), order]
                hard = np.asarray(split.get('hard', np.zeros(len(top), bool)), bool)
                entry['splits'][split_name]['ranking'] = {
                    'top1_arm6_mse': float(top.mean()),
                    'top1_arm6_mse_hard': float(top[hard].mean()) if hard.any() else None,
                    'top1_bank_index': order.tolist(),
                    'top1_error_by_row': top.tolist(),
                }
                entry['splits'][split_name]['ablation'] = _ablation_metrics(model, mode, bank, bank_input, split,
                                                                            device, keyrep, errors, order)
                arrays[f'{mode}_{split_name}_scores'] = scores
                arrays[f'{mode}_{split_name}_errors'] = errors
                arrays[f'{mode}_{split_name}_top1'] = order
        result['conditions'][mode] = entry
        del model, keyrep
        if device.startswith('cuda'): torch.cuda.empty_cache()
    result['conditions'].update(_matching_keygrid_conditions())
    result['status'] = 'complete'
    result['protocol'] = {'attention_mass_unit': 'mean sum over source tokens for each query/head/latent token',
        'per_input_token_attention': 'mass divided by valid token count in the modality',
        'query_ablations': 'query-side inputs only; key anchors remain fixed; swaps use deterministic within-split permutation',
        'vlm_groups': {'0': 'base image', '1': 'wrist image', '2': 'language'},
        'key_side_diagnostic': '4516 memory-unit starts; held-out evaluation episodes are disjoint by KEY2 contract'}
    (output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    np.savez_compressed(output / 'arrays.npz', **arrays)
    _write_report(output / 'report.md', result)
    return result


def _task_clustering(vectors, task):
    from sklearn.metrics import silhouette_score
    x = np.asarray(vectors, np.float32).copy(); labels = np.asarray(task)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    sims = x @ x.T
    np.fill_diagonal(sims, -np.inf)
    k = 5
    nn = np.argpartition(-sims, k, axis=1)[:, :k]
    pred = np.asarray([np.bincount(labels[row].astype(int)).argmax() for row in nn])
    acc = float(np.mean(pred == labels))
    rng = np.random.default_rng(4101)
    take = rng.choice(len(x), min(2500, len(x)), replace=False)
    sil = float(silhouette_score(x[take], labels[take], metric='cosine'))
    return {'knn_k': k, 'knn_task_accuracy': acc, 'silhouette_cosine': sil,
            'silhouette_rows': len(take), 'task_count': int(len(np.unique(labels)))}


def _matching_keygrid_conditions():
    """KEYGRID reselected E checkpoints are the same KEY2-F checkpoints; no new E weights exist."""
    path = Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/key-grid-01/production-04/results.json')
    if not path.exists(): return {'keygrid_checkpoints': {'available': False}}
    doc = json.loads(path.read_text())
    found = {}
    for mode in ('e_h', 'e_vlm_h', 'e_vj_vlm_h'):
        row = doc.get('existing_reselected', {}).get(mode, {}).get('selected_by_dev_top1')
        if row:
            ckpt = row.get('checkpoint')
            found[mode] = {'checkpoint': ckpt, 'same_as_key2f': ckpt == json.loads(CONFIG_PATH.read_text())['selected'][mode]['checkpoint'],
                           'dev_top1_arm6_mse': row.get('dev_top1_arm6_mse'),
                           'note': 'KEYGRID reselected existing E checkpoint; no independently trained E checkpoint'}
    return {'keygrid_checkpoints': {'available': bool(found), 'conditions': found}}


def run_keygrid_additional(output: Path = OUT_DEFAULT, *, device='cuda:0', memory_fraction=.15):
    """Evaluate distinct KEYGRID top-1-selected E checkpoints in the same audit."""
    output = Path(output)
    result_path = output / 'results.json'
    if not result_path.exists(): raise FileNotFoundError(result_path)
    result = json.loads(result_path.read_text())
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
    if device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(float(memory_fraction), int(device.split(':')[1]) if ':' in device else 0)
    progress_path = output / 'progress-keygrid.jsonl'
    def progress(stage, **fields):
        with progress_path.open('a') as f: f.write(json.dumps({'time': time.time(), 'stage': stage, **fields}) + '\n')
    bank, bank_input, learned, unseen, dev = _load_data(progress)
    keygrid = json.loads((Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/key-grid-01/production-04/results.json')).read_text())
    base = json.loads(CONFIG_PATH.read_text())['selected']
    arrays_path = output / 'arrays.npz'
    arrays = dict(np.load(arrays_path, allow_pickle=False)) if arrays_path.exists() else {}
    for mode in MODES:
        picked = keygrid['existing_reselected'][mode]['selected_by_dev_top1']
        checkpoint = picked['checkpoint']
        if checkpoint == base[mode]['checkpoint']: continue
        label = f'keygrid_{mode}'
        model = _load_model(mode, checkpoint, device)
        probe = key2._pack_part(learned, np.arange(3))
        with torch.no_grad():
            a = key2._model_batch_encode(model, mode, probe, device)
            _ = cross_attention_weights(model, probe, device)
            b = key2._model_batch_encode(model, mode, probe, device)
        entry = {'checkpoint': checkpoint, 'forward_max_abs_diff': float((a-b).abs().max().cpu()),
                 'selection_source': 'KEYGRID production-04 selected_by_dev_top1', 'splits': {}}
        if entry['forward_max_abs_diff'] != 0: raise AssertionError('KEYGRID observer parity failed')
        keyrep = _encode(model, bank_input, device)
        entry['key_latent_collapse'] = latent_collapse(keyrep)
        entry['key_task_clustering'] = _task_clustering(keyrep.mean(1), bank['task'])
        for split_name, split in [('learned', learned), ('unseen_eval', unseen), ('dev', dev), ('key', bank_input)]:
            progress('keygrid_split', mode=mode, split=split_name)
            z = keyrep if split_name == 'key' else _encode(model, split, device)
            entry['splits'][split_name] = {'latent_collapse': latent_collapse(z),
                'attention': _encode_attn_profile(model, split, device, name=split_name)}
            if split_name in ('learned', 'unseen_eval'):
                scores, errors, _, _ = _scores(model, mode, bank_input, split, device, bank, keyrep)
                order = np.argmax(scores, axis=1); top = errors[np.arange(len(errors)), order]
                hard = np.asarray(split.get('hard', np.zeros(len(top), bool)), bool)
                entry['splits'][split_name]['ranking'] = {'top1_arm6_mse': float(top.mean()),
                    'top1_arm6_mse_hard': float(top[hard].mean()) if hard.any() else None,
                    'top1_bank_index': order.tolist(), 'top1_error_by_row': top.tolist()}
                entry['splits'][split_name]['ablation'] = _ablation_metrics(model, mode, bank, bank_input, split, device, keyrep, errors, order)
                arrays[f'{label}_{split_name}_scores'] = scores
                arrays[f'{label}_{split_name}_errors'] = errors
                arrays[f'{label}_{split_name}_top1'] = order
        result['conditions'][label] = entry
        del model, keyrep
        if device.startswith('cuda'): torch.cuda.empty_cache()
    result['status'] = 'complete'
    result['keygrid_note'] = 'Distinct KEYGRID top-1 checkpoints evaluated after primary run; source data and protocol identical.'
    result_path.write_text(json.dumps(result, indent=2) + '\n')
    np.savez_compressed(arrays_path, **arrays)
    _write_report(output / 'report.md', result)
    return result


def _write_report(path, doc):
    lines = ['# ATTN1：潜在トークンの注意配分（学習なし）', '',
        f"行数：キー {doc['row_counts']['keys']}、学習済み {doc['row_counts']['learned']}、未学習 Eval {doc['row_counts']['unseen_eval']}、Dev {doc['row_counts']['dev']}。",
        '', '## 潜在表現と注意', '',
        '| 条件 | 分割 | 平均 token cosine | 有効ランク | VLM画像 | VLM言語 | V-JEPA | 履歴 | 状態 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for mode, data in doc['conditions'].items():
        if mode == 'keygrid_checkpoints': continue
        for split in ('learned', 'unseen_eval', 'dev', 'key'):
            x = data['splits'][split]; att = np.asarray(x['attention']['total_attention'])
            # Head/token averaged modality mass; V-JEPA is included when present.
            means = att.mean((0, 1))
            col = means.tolist()
            c = x['latent_collapse']
            lines.append(f"| {mode} | {split} | {c['mean_pairwise_cosine']:.4f} | {c['effective_rank_mean']:.3f} | " + ' | '.join(f'{v:.4f}' for v in col) + ' |')
    lines += ['', '## 基準検索の top-1 腕 MSE', '',
              '| 条件 | 分割 | 全体 | hard | forward 最大差 |', '|---|---|---:|---:|---:|']
    for mode, data in doc['conditions'].items():
        if mode == 'keygrid_checkpoints': continue
        for split in ('learned', 'unseen_eval'):
            x = data['splits'][split]['ranking']
            lines.append(f"| {mode} | {split} | {x['top1_arm6_mse']:.4f} | {x['top1_arm6_mse_hard']:.4f} | {data['forward_max_abs_diff']:.1f} |")
    lines += ['', '## 入力除去時の top-1', '',
              '| 条件 | 分割 | 入力 | zero 一致率 / MSE | swap 一致率 / MSE |', '|---|---|---|---:|---:|']
    for mode, data in doc['conditions'].items():
        if mode == 'keygrid_checkpoints': continue
        for split in ('learned', 'unseen_eval'):
            ab = data['splits'][split]['ablation']
            for modality, vals in ab.items():
                if not vals.get('zero', {}).get('applicable'): continue
                z, s = vals['zero'], vals['swap']
                lines.append(f"| {mode} | {split} | {modality} | {z['top1_agreement']:.3f} / {z['top1_arm6_mse']:.4f} | {s['top1_agreement']:.3f} / {s['top1_arm6_mse']:.4f} |")
    lines += ['', '## キー側のタスクまとまり', '', '| 条件 | 5-NN task accuracy | cosine silhouette |', '|---|---:|---:|']
    for mode, data in doc['conditions'].items():
        if mode == 'keygrid_checkpoints': continue
        x = data['key_task_clustering']; lines.append(f"| {mode} | {x['knn_task_accuracy']:.4f} | {x['silhouette_cosine']:.4f} |")
    lines += ['', '注意は各クロスアテンション層（本モデルは1層）・ヘッド・潜在トークンごとの平均質量で記録し、詳細配列は `results.json` に保存した。`per_input_token_attention` は各モダリティの有効トークン数で割った値。',
        '入力 ablation は推論時のクエリ側だけを変更し、キー側の4,516行は固定した。Eval 正解は行動誤差の採点にのみ用いた。',
        '観測機能は既存 forward の外側に置いたため、チェックポイント出力の再計算差は全条件で 0。学習・重み更新は行っていない。',
        'KEYGRID で top-1 Dev 選択された同方式の E チェックポイントも別条件として評価した。これらは KEYGRID 専用に新規学習した E ではなく、既存 tuning の候補からの再選択である。']
    path.write_text('\n'.join(lines) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=OUT_DEFAULT)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--memory-fraction', type=float, default=.15)
    p.add_argument('--keygrid-additional', action='store_true')
    a = p.parse_args()
    if a.keygrid_additional:
        run_keygrid_additional(a.output, device=a.device, memory_fraction=a.memory_fraction)
    else:
        run(a.output, device=a.device, memory_fraction=a.memory_fraction)


if __name__ == '__main__': main()
