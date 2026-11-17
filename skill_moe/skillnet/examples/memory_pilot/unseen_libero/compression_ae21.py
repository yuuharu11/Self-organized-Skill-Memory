#!/usr/bin/env python3
"""AE21 versus classical compression on the CMP3 fixed and variable windows.

The production path fits representations only on rows['memory']; LIBERO-90
tasks 6 and 7 are used for model selection, and the other evaluation sets are
scored only.  `--smoke` exercises the complete data path on a small subset.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'pi05_memory')]
import compression_matched as c2
import compression_seen_cuts as c3
import motion_core_eval as motion
from draft_reconstruction import ROOT, UNSEEN_ROWS
from variable_length_error import stitch

OUT_DEFAULT = ROOT / 'unseen-libero' / 'compress-ae-01' / 'production'
SEED = 20261116
AXES = ('x', 'y', 'z', 'rx', 'ry', 'rz', 'gripper')
VAR_CFG = {'mode': 'all', 'min_len': 5, 'width': 3}


class ActionAE:
    """MLP AE whose only input is an action tensor (N,T,7)."""
    def __init__(self, width=128, depth=2, latent=21, device='cpu', steps=10):
        import torch
        from torch import nn
        self.torch = torch
        self.device = torch.device(device)
        self.steps = int(steps)
        layers = []
        d = self.steps * 7
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.GELU()]
            d = width
        layers.append(nn.Linear(d, latent))
        self.encoder = nn.Sequential(*layers).to(self.device)
        layers = []
        d = latent
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.GELU()]
            d = width
        layers.append(nn.Linear(d, self.steps * 7))
        self.decoder = nn.Sequential(*layers).to(self.device)

    def encode(self, actions):
        x = self.torch.as_tensor(actions, dtype=self.torch.float32, device=self.device)
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.steps, 7):
            raise ValueError(f'expected actions [N,{self.steps},7], got {tuple(x.shape)}')
        return self.encoder(x.reshape(len(x), self.steps * 7))

    def decode(self, latent):
        z = self.torch.as_tensor(latent, dtype=self.torch.float32, device=self.device)
        return self.decoder(z).reshape(len(z), self.steps, 7)

    def __call__(self, actions):
        return self.decode(self.encode(actions))


def _set_seed(seed):
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(8)


def train_ae(train_actions, dev_actions, *, seed, width=128, depth=2,
             lr=3e-4, weight_decay=1e-5, max_epochs=80, patience=8,
             batch_size=256, device='cpu'):
    """Fit on actions only; early stopping observes Dev reconstruction loss."""
    import torch
    _set_seed(seed)
    steps = int(np.asarray(train_actions).shape[1])
    if np.asarray(train_actions).shape[2] != 7:
        raise ValueError('AE accepts action values only, exactly seven axes')
    model = ActionAE(width, depth, 21, device, steps=steps)
    opt = torch.optim.AdamW(list(model.encoder.parameters()) + list(model.decoder.parameters()),
                            lr=lr, weight_decay=weight_decay)
    train = torch.as_tensor(np.asarray(train_actions, np.float32), device=device)
    dev = torch.as_tensor(np.asarray(dev_actions, np.float32), device=device)
    if train.ndim != 3 or tuple(train.shape[1:]) != (steps, 7):
        raise ValueError(f'AE training input must be action windows [N,{steps},7]')
    if not len(train) or not len(dev):
        raise ValueError('empty train or Dev actions')
    best, best_state, best_epoch, stale = math.inf, None, 0, 0
    gen = torch.Generator(device='cpu').manual_seed(seed)
    for epoch in range(1, max_epochs + 1):
        model.encoder.train(); model.decoder.train()
        order = torch.randperm(len(train), generator=gen).to(device)
        for ids in order.split(batch_size):
            pred = model(train[ids])
            loss = torch.mean((pred - train[ids]) ** 2)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        model.encoder.eval(); model.decoder.eval()
        with torch.no_grad():
            val = torch.cat([torch.mean((model(x) - x) ** 2).reshape(1)
                             for x in dev.split(batch_size)]).mean().item()
        if val < best - 1e-8:
            best, best_epoch, stale = val, epoch, 0
            best_state = ({k: v.detach().cpu().clone() for k, v in model.encoder.state_dict().items()},
                          {k: v.detach().cpu().clone() for k, v in model.decoder.state_dict().items()})
        else:
            stale += 1
            if stale >= patience:
                break
    model.encoder.load_state_dict(best_state[0]); model.decoder.load_state_dict(best_state[1])
    model.encoder.eval(); model.decoder.eval()
    return model, {'best_dev_mse': float(best), 'best_epoch': int(best_epoch), 'epochs_run': int(epoch),
                   'seed': int(seed), 'width': width, 'depth': depth, 'lr': lr,
                   'weight_decay': weight_decay}


def dev_actions_from_cache():
    """Only fixed Dev task IDs 6 and 7; never returns Eval targets."""
    required = ('actions', 'task', 'episode', 'frame', 'is_dev')
    data = {k: np.load(UNSEEN_ROWS / f'{k}.npy', mmap_mode='r') for k in required}
    ix = np.flatnonzero(np.isin(np.asarray(data['task']), (6, 7)) & np.asarray(data['is_dev'], bool))
    if not len(ix):
        raise RuntimeError('no LIBERO-90 Dev rows for tasks 6/7')
    return np.asarray(data['actions'][ix, :, :7], np.float32), {
        'task_ids': [6, 7], 'rows': int(len(ix)),
        'episodes': sorted({(int(data['task'][i]), int(data['episode'][i])) for i in ix}),
        'row_ids': ix,
    }


def memory_fit_windows(d):
    """Assert all AE/PCA fit examples are rows['memory'], never validation."""
    rows = d['rows']; mask = np.asarray(rows['memory'], bool)
    if np.any(mask & np.asarray(rows['validation'], bool)):
        raise AssertionError('memory fit includes validation rows')
    ids = np.flatnonzero(mask)
    if not np.array_equal(ids, np.asarray(d['mem_ids'])):
        raise AssertionError('memory ids differ from prepared memory IDs')
    return np.asarray(rows['actions'][ids, :, :7], np.float32), ids


def assert_fit_is_memory_only(rows, fit_ids):
    fit_ids = np.asarray(fit_ids, np.int64)
    memory = np.asarray(rows['memory'], bool)
    validation = np.asarray(rows['validation'], bool)
    if len(np.unique(fit_ids)) != len(fit_ids) or np.any(fit_ids < 0) or np.any(fit_ids >= len(memory)):
        raise AssertionError('invalid or duplicate fit row IDs')
    if not np.all(memory[fit_ids]) or np.any(validation[fit_ids]):
        raise AssertionError('representation fit includes non-memory or validation rows')
    return True


def variable_memory(d):
    items, lengths = c3.spans(d['memory_traces'], VAR_CFG)
    if len(items) != c3.G1_EXPECT['units']:
        raise AssertionError(f"expected {c3.G1_EXPECT['units']} variable units, got {len(items)}")
    return items, lengths


def variable_fit_phase(items, lengths):
    del lengths
    return np.stack([c2.phase_resample(x, 32) for x in items]).astype(np.float32)


def variable_ae_templates(model, items):
    phase = variable_fit_phase(items, None)
    rec_phase = predict(model, phase)
    return [np.stack([np.interp(np.linspace(0, 1, len(x)), np.linspace(0, 1, 32), rec_phase[i, :, j])
                      for j in range(7)], axis=-1).astype(np.float32)
            for i, x in enumerate(items)]


def predict(model, actions, batch_size=512):
    import torch
    x = np.asarray(actions, np.float32)
    out = []
    with torch.no_grad():
        for st in range(0, len(x), batch_size):
            out.append(model(torch.as_tensor(x[st:st+batch_size], dtype=torch.float32,
                                             device=model.device)).detach().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.empty_like(x)


def per_row_axis_mse(pred, target):
    """Per-axis squared error contribution; sum over seven equals active7 MSE."""
    err = (np.asarray(pred, np.float64)[..., :7] - np.asarray(target, np.float64)[..., :7]) ** 2
    return err.sum(axis=1) / (10.0 * 7.0)


def phase_batch(actions, phases=32):
    a = np.asarray(actions, np.float32)
    return np.stack([c2.phase_resample(x, phases) for x in a])


def phase_restore_batch(actions, rec_phase):
    actions = np.asarray(actions, np.float32)
    rec_phase = np.asarray(rec_phase, np.float32)
    dst = np.linspace(0., 1., 10)
    src = np.linspace(0., 1., rec_phase.shape[1])
    return np.stack([np.stack([np.interp(dst, src, rec_phase[i, :, j]) for j in range(7)], -1)
                     for i in range(len(actions))]).astype(np.float32)


def summarize_errors(pred, target, hard):
    axis = per_row_axis_mse(pred, target)
    err = axis.sum(axis=1)
    hard = np.asarray(hard, bool)
    def agg(ix):
        if not np.any(ix): return None
        return {'n': int(ix.sum()), 'mse': float(err[ix].mean()),
                'per_axis': {AXES[i]: float(axis[ix, i].mean()) for i in range(7)},
                'axis_sum': float(axis[ix].mean(axis=0).sum())}
    return {'all': agg(np.ones(len(err), bool)), 'hard': agg(hard), 'row_mse': err.tolist(),
            'row_axis': axis.tolist(), 'hard_mask': hard.tolist()}


def retrieve_fixed(templates, targets, *, arm_only=False):
    t = np.asarray(targets, np.float32); c = np.asarray(templates, np.float32)
    dims = 6 if arm_only else 7
    ix = np.empty(len(t), np.int64)
    for i in range(len(t)):
        ix[i] = np.argmin(np.mean((c[..., :dims] - t[i, :, :dims]) ** 2, axis=(1, 2)))
    chosen = c[ix].copy()
    if arm_only:
        chosen[..., 6] = t[..., 6]
    return chosen, ix


def retrieve_variable(templates, lengths, data, *, arm_only=False):
    """CMP3 DP playback; zero grip for the arm-only choice, score in arm axes."""
    import optimal_units_eval as opt
    traces = data['traces']; rows_of = data['rows_of']
    dim = 6 if arm_only else 7
    grouped = {}
    for i, (a, n) in enumerate(zip(templates, lengths)):
        x = np.asarray(a, np.float64).copy()
        if arm_only: x[:, 6] = 0.
        grouped.setdefault(int(n), []).append(x)
    drafts = {n: np.stack(v) for n, v in grouped.items()}; lens = sorted(drafts)
    pred = np.empty_like(data['targets'], dtype=np.float64)
    for key, trace in traces.items():
        scoring_trace = np.asarray(trace, np.float64).copy()
        if arm_only: scoring_trace[:, 6] = 0.
        table = opt.best_cost_table(scoring_trace, drafts)
        budget = max(1, math.ceil(len(trace) / 10))
        _, seq = opt.best_sequence(table, lens, len(trace), budget)
        joined = np.empty_like(scoring_trace)
        for start, length in seq:
            piece = table[start, length][1]
            joined[start:start + len(piece)] = piece
        pred[rows_of[key]] = joined.reshape(-1, 10, 7)
    if arm_only:
        pred[..., 6] = data['targets'][..., 6]
    return pred


def _boot_ci(a, b, episodes, *, mask=None, n_boot=5000, seed=SEED):
    diff = np.asarray(a, float) - np.asarray(b, float)
    eps = np.asarray(episodes)
    if mask is not None:
        mask = np.asarray(mask, bool)
        diff, eps = diff[mask], eps[mask]
    unique = np.unique(eps)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    rows = {e: np.flatnonzero(eps == e) for e in unique}
    for i in range(n_boot):
        chosen = rng.choice(unique, len(unique), replace=True)
        ids = np.concatenate([rows[e] for e in chosen])
        vals[i] = diff[ids].mean()
    return {'mean_difference': float(diff.mean()), 'ci95': np.quantile(vals, [.025, .975]).tolist(),
            'bootstrap_samples': n_boot}


def _episode_keys(data):
    ep = np.empty(len(data['targets']), dtype=object)
    for k, ids in data['rows_of'].items(): ep[ids] = f'{int(k[0])}:{int(k[1])}'
    return ep


def _fit_pca_templates(memory, fit):
    mean, basis = c2.pca_fit(fit, 21)
    return c2.pca_restore(c2.pca_project(memory, mean, basis), mean, basis).astype(np.float32), (mean, basis)


def _eval_template_method(templates, evals, method, *, variable=False, lengths=None):
    row = {'method': method, 'eval': {}, 'self_reconstruction': {}}
    for split, data in evals.items():
        pred7, _ = (retrieve_variable(templates, lengths, data) if variable else
                    retrieve_fixed(templates, data['targets']))
        pred6, _ = (retrieve_variable(templates, lengths, data, arm_only=True) if variable else
                    retrieve_fixed(templates, data['targets'], arm_only=True))
        row['eval'][split] = {
            'upper7': summarize_errors(pred7, data['targets'], data['hard']),
            'upper6_choice_arm_score': summarize_errors(pred6, data['targets'], data['hard'])}
    return row


def fixed_compression_metrics(recon, evals):
    result = {}
    for split, data in evals.items():
        result[split] = summarize_errors(recon, data['targets'], data['hard'])
    return result


def _fixed_representations(memory, fit, ae):
    pca_rec, pca_params = _fit_pca_templates(memory, fit)
    dct_rec = c2.dct_decode(c2.dct_encode(memory, 3), 10).astype(np.float32)
    key_rec = c2.keyframe_decode(c2.keyframe_encode(memory, 3), 10).astype(np.float32)
    ae_rec = predict(ae, memory)
    return {'raw70': memory, 'dct21': dct_rec, 'pca21': pca_rec,
            'keyframe21': key_rec, 'ae21': ae_rec}, pca_params


def _aggregate_seed_summaries(summaries):
    """Average rowwise summaries across independently trained AE seeds."""
    all_rows = [np.asarray(x['row_mse'], np.float64) for x in summaries]
    all_axes = [np.asarray(x['row_axis'], np.float64) for x in summaries]
    row = np.mean(all_rows, axis=0)
    axes = np.mean(all_axes, axis=0)
    def agg(mask):
        mask = np.asarray(mask, bool)
        return {'n': int(mask.sum()), 'mse': float(row[mask].mean()),
                'per_axis': {AXES[i]: float(axes[mask, i].mean()) for i in range(7)},
                'axis_sum': float(axes[mask].mean(0).sum())}
    return {'all':agg(np.ones(len(row),bool)),
            'hard':agg(np.asarray(summaries[0]['hard_mask'],bool)),
            'row_mse':row.tolist(),'row_axis':axes.tolist(),
            'seed_range_all':[float(x['all']['mse']) for x in summaries],
            'seed_range_hard':[float(x['hard']['mse']) if x['hard'] else None for x in summaries]}


def _variable_representations(items, ae, fit_phase):
    dct = c2.compressed_variable(items, 'dct', 3)
    mean, basis = c2.pca_fit(fit_phase, 21)
    pca = c2.compressed_variable(items, 'pca', 21, mean, basis)
    ae = variable_ae_templates(ae, items)
    return {'raw': [np.asarray(x, np.float32) for x in items],
            'dct22': dct, 'pca22': pca, 'ae22': ae}, (mean, basis)


def _axis_metrics_by_split(pred, targets, hard):
    return summarize_errors(pred, targets, hard)


def run(smoke=False, output=OUT_DEFAULT, *, device=None):
    if output.exists():
        raise FileExistsError(f'refusing to overwrite output directory: {output}')
    import torch
    torch.set_num_threads(8)
    d, evals = c3.prepare()
    g1 = c3.verify_g1(d, evals['unseen'])
    full_counts = {k:{'tasks':len(v['task_ids']),'demos':len(v['demos']),
                      'windows':len(v['targets']),'hard':int(np.asarray(v['hard']).sum())}
                   for k,v in evals.items()}
    memory, memory_ids = memory_fit_windows(d)
    assert_fit_is_memory_only(d['rows'], memory_ids)
    if not np.array_equal(memory, d['memory']): raise AssertionError('memory fit rows mismatch')
    # Dev only: rows marked Dev in unseen cache, fixed to LIBERO-90 task IDs 6,7.
    dev_x, dev_meta = dev_actions_from_cache()
    if device is None: device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if smoke:
        # Include both hard and ordinary windows; bound fitting and epochs only.
        for k, v in list(evals.items()): evals[k] = c3.subset_eval(v, 2)
        dev_x = dev_x[:min(32, len(dev_x))]
    # Small declared grid, selected on Dev reconstruction error only.
    grid = ([{'width': 32, 'depth': 1, 'lr': 1e-3, 'weight_decay': 0.0}]
            if smoke else [dict(width=w, depth=dep, lr=lr, weight_decay=wd)
                           for w, dep, lr, wd in ((64,1,1e-3,0.), (64,2,3e-4,1e-5),
                                                 (128,1,3e-4,1e-5), (128,2,1e-3,0.))])
    fit_fixed = memory
    var_items, var_lengths = variable_memory(d)
    fit_phase = variable_fit_phase(var_items, var_lengths)
    fit_var = fit_phase
    if smoke:
        fit_fixed = fit_fixed[:512]
        fit_var = fit_var[:512]
    selected = {}
    for name, fit_data, dev_data in (('fixed', fit_fixed, dev_x), ('variable', fit_var, phase_batch(dev_x, 32))):
        candidates = []
        for gi, cfg in enumerate(grid):
            model, info = train_ae(fit_data, dev_data, seed=SEED + gi, max_epochs=3 if smoke else 80,
                                   patience=2 if smoke else 8, device=device, **cfg)
            candidates.append((info['best_dev_mse'], gi, model, info))
        _, gi, model, info = min(candidates, key=lambda x: (x[0], x[1]))
        selected[name] = {'grid_index': gi, 'config': info, 'model': model}
    # Three final seeds at the Dev-selected architecture; smoke uses one light seed.
    seed_models = {'fixed': [], 'variable': []}
    final_seeds = [SEED] if smoke else [SEED, SEED + 1, SEED + 2]
    for name, fit_data, dev_data in (('fixed', memory, dev_x), ('variable', fit_phase, phase_batch(dev_x, 32))):
        cfg = {k: selected[name]['config'][k] for k in ('width', 'depth', 'lr', 'weight_decay')}
        for seed in final_seeds:
            model, info = train_ae(fit_data, dev_data, seed=seed, max_epochs=4 if smoke else 120,
                                   patience=2 if smoke else 12, device=device, **cfg)
            seed_models[name].append((model, info))
    # Classical bases are fit only on memory windows, matching AE's training set.
    fixed_repr, pca_params = _fixed_representations(memory, memory, seed_models['fixed'][0][0])
    var_repr_by_seed = [_variable_representations(var_items, model, fit_phase) for model, _ in seed_models['variable']]
    variable_repr, var_pca_params = var_repr_by_seed[0]
    result = {
        'mode': 'smoke' if smoke else 'full', 'seed': SEED, 'device': str(device),
        'protocol': {'AE_fit_rows': 'rows[\'memory\'] only; action tensors only, fixed 10x7 and variable phase32x7',
                     'PCA_fit_rows': 'same rows[\'memory\'] only',
                     'dev': 'LIBERO-90 rows with task in {6,7} and is_dev=true; Dev reconstruction MSE selects grid/early stop',
                     'eval': 'CMP3 learned and unseen sets; target actions used only for nearest template selection/scoring',
                     'variable_segmentation': 'motion minima + gripper, min length 5, width 3; input AE phase32 plus stored length metadata',
                     'AE_values': {'fixed': 21, 'variable': 22}, 'seeds': final_seeds,
                     'G1_reproduction': g1, 'memory_fit_row_ids': memory_ids.tolist()},
        'evaluation_sets': c3.unit_meta(d, evals), 'full_evaluation_counts': full_counts,
        'dev': {k:v for k,v in dev_meta.items() if k != 'row_ids'},
        'selection': {k:{'grid_index':v['grid_index'],'config':v['config']} for k,v in selected.items()},
        'fixed': {'methods': [], 'seed_results': []}, 'variable': {'methods': [], 'seed_results': []},
    }
    # Fixed methods: compression self-reconstruction and nearest-template upper bounds.
    fixed_seed_recon = [predict(model, memory) for model, _ in seed_models['fixed']]
    for method, rec in fixed_repr.items():
        comp = {}
        method_seed_summaries = {s: [] for s in evals}
        eval_seed_summaries = {s: {'upper7': [], 'upper6_choice_arm_score': []} for s in evals}
        for split, e in evals.items():
            # Encode/decode each target independently for compression-only error.
            tgt = e['targets']
            if method == 'raw70': rr = tgt.copy()
            elif method == 'dct21': rr = c2.dct_decode(c2.dct_encode(tgt, 3), 10).astype(np.float32)
            elif method == 'keyframe21': rr = c2.keyframe_decode(c2.keyframe_encode(tgt, 3), 10).astype(np.float32)
            elif method == 'pca21': rr = c2.pca_restore(c2.pca_project(tgt, *pca_params), *pca_params).astype(np.float32)
            elif method == 'ae21':
                seed_summaries=[]
                for rr_seed in [predict(model, tgt) for model,_ in seed_models['fixed']]:
                    seed_summaries.append(summarize_errors(rr_seed,tgt,e['hard']))
                comp[split] = _aggregate_seed_summaries(seed_summaries)
            else: rr = predict(seed_models['fixed'][0][0], tgt)
            if method != 'ae21': comp[split] = summarize_errors(rr, tgt, e['hard'])
        method_row = {'method': method, 'values_per_unit': 70 if method == 'raw70' else 21,
                      'compression': comp, 'eval': {}}
        for split, e in evals.items():
            reps_by_seed = fixed_seed_recon if method=='ae21' else [rec]
            for reps in reps_by_seed:
                upper7, ix7 = retrieve_fixed(reps, e['targets'])
                upper6, ix6 = retrieve_fixed(reps, e['targets'], arm_only=True)
                eval_seed_summaries[split]['upper7'].append(summarize_errors(upper7,e['targets'],e['hard']))
                eval_seed_summaries[split]['upper6_choice_arm_score'].append(summarize_errors(upper6,e['targets'],e['hard']))
                if method != 'ae21':
                    method_row['eval'][split] = {'upper7':eval_seed_summaries[split]['upper7'][0],
                        'upper6_choice_arm_score':eval_seed_summaries[split]['upper6_choice_arm_score'][0],
                        'chosen7':ix7.tolist(),'chosen6':ix6.tolist()}
            if method=='ae21':
                method_row['eval'][split]={'upper7':_aggregate_seed_summaries(eval_seed_summaries[split]['upper7']),
                    'upper6_choice_arm_score':_aggregate_seed_summaries(eval_seed_summaries[split]['upper6_choice_arm_score'])}
        result['fixed']['methods'].append(method_row)
    # AE seed range and paired episode bootstrap against DCT/PCA (per evaluation set).
    for seed, (model, info) in enumerate(seed_models['fixed']):
        rr = predict(model, memory)
        result['fixed']['seed_results'].append({'seed': final_seeds[seed], 'train': info,
            'memory_self_reconstruction_mse': float(np.mean((rr-memory)**2))})
    # Variable methods and G1 baseline validation.
    lengths = var_lengths
    var_ae_templates_by_seed = [variable_ae_templates(model,var_items) for model,_ in seed_models['variable']]
    for method, templates in variable_repr.items():
        unit_count = len(templates)
        comp = {}
        for split,e in evals.items():
            if method == 'raw': cr = e['targets'].copy()
            else:
                cr = np.empty_like(e['targets'])
                # Each evaluation window is encoded as an independent 10-frame sequence;
                # PCA and AE use the fixed phase convention, DCT is direct.
                if method == 'dct22': cr = c2.dct_decode(c2.dct_encode(e['targets'], 3), 10).astype(np.float32)
                elif method == 'pca22':
                    phase_rec = c2.pca_restore(c2.pca_project(phase_batch(e['targets'],32), *var_pca_params), *var_pca_params).astype(np.float32)
                    cr = phase_restore_batch(e['targets'], phase_rec)
                elif method == 'ae22':
                    phase_targets = phase_batch(e['targets'], 32)
                    comp[split] = _aggregate_seed_summaries([
                        summarize_errors(phase_restore_batch(e['targets'], predict(model, phase_targets)),
                                         e['targets'],e['hard']) for model,_ in seed_models['variable']])
                else: raise ValueError(method)
            if method != 'ae22': comp[split] = summarize_errors(cr, e['targets'], e['hard'])
        row = {'method': method, 'units': unit_count, 'values_per_unit': {'raw': None, 'dct22':22, 'pca22':22, 'ae22':22}[method],
               'compression': comp, 'eval': {}}
        for split,e in evals.items():
            model_templates = var_ae_templates_by_seed if method=='ae22' else [templates]
            scores7=[]; scores6=[]
            for ts in model_templates:
                p7 = retrieve_variable(ts, lengths, e)
                p6 = retrieve_variable(ts, lengths, e, arm_only=True)
                scores7.append(summarize_errors(p7,e['targets'],e['hard']))
                scores6.append(summarize_errors(p6,e['targets'],e['hard']))
            row['eval'][split] = {'upper7':_aggregate_seed_summaries(scores7),
                                  'upper6_choice_arm_score':_aggregate_seed_summaries(scores6)}
        result['variable']['methods'].append(row)
    dct_var = next(x for x in result['variable']['methods'] if x['method']=='dct22')
    if not smoke:
        cmp3_m = dct_var['eval']['unseen']['upper7']['all']['mse']
        cmp3_h = dct_var['eval']['unseen']['upper7']['hard']['mse']
        if abs(cmp3_m-c3.G1_EXPECT['all_mse'])>2e-6 or abs(cmp3_h-c3.G1_EXPECT['hard_mse'])>2e-6:
            raise RuntimeError(f'variable DCT retrieval differs from CMP3 G1 baseline: {cmp3_m}/{cmp3_h}')
    for seed, (model, info) in enumerate(seed_models['variable']):
        phase_rec = predict(model, fit_phase)
        result['variable']['seed_results'].append({'seed':final_seeds[seed], 'train':info,
            'memory_phase_self_reconstruction_mse':float(np.mean((phase_rec-fit_phase)**2)),
            'memory_units':len(var_items)})
    # Compare AE/DCT/PCA using paired episode bootstrap on each split; retain row errors.
    result['paired_ci'] = {}
    for category in ('fixed','variable'):
        result['paired_ci'][category] = {}
        rows = result[category]['methods']
        ae_name = 'ae21' if category=='fixed' else 'ae22'
        for split,e in evals.items():
            ae_row = next(x for x in rows if x['method']==ae_name)
            for baseline in (('dct21','pca21') if category=='fixed' else ('dct22','pca22')):
                base_row = next(x for x in rows if x['method']==baseline)
                hardmask = np.asarray(e['hard'],bool)
                paired = {}
                for section in ('compression','upper7','upper6_choice_arm_score'):
                    aa = ae_row['compression'][split] if section=='compression' else ae_row['eval'][split][section]
                    bb = base_row['compression'][split] if section=='compression' else base_row['eval'][split][section]
                    av = np.asarray(aa['row_mse']); bv = np.asarray(bb['row_mse'])
                    paired[section] = {
                        'all': _boot_ci(av,bv,_episode_keys(e),n_boot=200 if smoke else 5000),
                        'hard': _boot_ci(av,bv,_episode_keys(e),mask=hardmask,n_boot=200 if smoke else 5000),
                    }
                result['paired_ci'][category][split+':'+baseline] = paired
    # Exact CMP3 consistency checks on full split outputs, separate from smoke subsets.
    result['cmp3_checks'] = {'G1_variable_dct3': g1,
        'fixed_dct21_smoke_or_full': {s: next(x for x in result['fixed']['methods'] if x['method']=='dct21')['eval'][s]['upper7']['all']['mse']
                                      for s in evals},
        'custom_variable_dct_full_assert': not smoke}
    # Convert internal row arrays to compact NPZ; JSON contains aggregate summaries only.
    result = _json_safe(result)
    output.mkdir(parents=True, exist_ok=False)
    (output/'results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    (output/'report.md').write_text(render_report(result))
    return result


def _json_safe(obj):
    if isinstance(obj, dict): return {k:_json_safe(v) for k,v in obj.items() if k not in ('row_mse','row_axis','chosen7','chosen6')}
    if isinstance(obj, (list,tuple)): return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic): return obj.item()
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj


def render_report(r):
    lines=['# AE21 と古典圧縮の比較','',
      ('**スモークのみ。以下は本番比較ではない。**' if r['mode']=='smoke' else '**本番結果。**'),
      '誤差は10×7の正規化行動空間で、各軸寄与はその軸の二乗誤差和÷70。7軸の合計がactive7 MSE。AEは行動だけを入力し、潜在21値（可変長は長さメタデータを加えて22値）から復元する。',
      f"Dev: LIBERO-90 task 6/7 の {r['dev']['rows']} 行のみ。メモリ fit: {r['evaluation_sets']['memory']['tasks']}タスク・{r['evaluation_sets']['memory']['demos']}デモ・{r['evaluation_sets']['memory']['windows']}行。",
      f"評価窓: learned {r['evaluation_sets']['learned']['tasks']}タスク/{r['evaluation_sets']['learned']['demos']}デモ/{r['evaluation_sets']['learned']['windows']}窓 (hard {r['evaluation_sets']['learned']['hard']}); unseen {r['evaluation_sets']['unseen']['tasks']}タスク/{r['evaluation_sets']['unseen']['demos']}デモ/{r['evaluation_sets']['unseen']['windows']}窓 (hard {r['evaluation_sets']['unseen']['hard']}).",'',
      f"全評価予定件数（CMP3）：learned {r['full_evaluation_counts']['learned']['windows']}窓/hard {r['full_evaluation_counts']['learned']['hard']}、unseen {r['full_evaluation_counts']['unseen']['windows']}窓/hard {r['full_evaluation_counts']['unseen']['hard']}。",'',
      '可変長 G1 の再現: '+json.dumps(r['protocol']['G1_reproduction'],ensure_ascii=False), '',
      '## 固定10窓：圧縮誤差と最近傍上限','',
      '| method | values | split | compression all/hard | upper 7-axis all/hard | upper arm6 all/hard |','|---|---:|---|---:|---:|---:|']
    for m in r['fixed']['methods']:
        for s in ('learned','unseen'):
            c=m['compression'][s]; a=m['eval'][s]['upper7']; b=m['eval'][s]['upper6_choice_arm_score']
            tag=''
            if m['method']=='ae21':
                tag=f" (seed range all {min(c['seed_range_all']):.5f}–{max(c['seed_range_all']):.5f}; hard {min(c['seed_range_hard']):.5f}–{max(c['seed_range_hard']):.5f})"
            lines.append(f"| {m['method']}{tag} | {m['values_per_unit']} | {s} | {c['all']['mse']:.6f}/{c['hard']['mse'] if c['hard'] else float('nan'):.6f} | {a['all']['mse']:.6f}/{a['hard']['mse'] if a['hard'] else float('nan'):.6f} | {b['all']['mse']:.6f}/{b['hard']['mse'] if b['hard'] else float('nan'):.6f} |")
    lines += ['', '## 可変長：速度極小＋グリッパー切替','',
      '| method | units | values | split | compression all/hard | upper 7-axis all/hard | upper arm6 all/hard |','|---|---:|---:|---|---:|---:|---:|']
    for m in r['variable']['methods']:
        for s in ('learned','unseen'):
            c=m['compression'][s]; a=m['eval'][s]['upper7']; b=m['eval'][s]['upper6_choice_arm_score']
            tag=''
            if m['method']=='ae22':
                tag=f" (seed range all {min(c['seed_range_all']):.5f}–{max(c['seed_range_all']):.5f}; hard {min(c['seed_range_hard']):.5f}–{max(c['seed_range_hard']):.5f})"
            lines.append(f"| {m['method']}{tag} | {m['units']} | {m['values_per_unit']} | {s} | {c['all']['mse']:.6f}/{c['hard']['mse'] if c['hard'] else float('nan'):.6f} | {a['all']['mse']:.6f}/{a['hard']['mse'] if a['hard'] else float('nan'):.6f} | {b['all']['mse']:.6f}/{b['hard']['mse'] if b['hard'] else float('nan'):.6f} |")
    lines += ['', '## 軸ごとの誤差寄与','',
      '各値は二乗誤差和÷70。行ごとに7軸を足すと、その行のactive7 MSEになる。AEは最終seed群の行別誤差平均を掲載。seed ごとの平均・範囲は results.json の各指標の seed_range_all / seed_range_hard と seed_results を参照。','',
      '| representation | set | metric | subset | x | y | z | rx | ry | rz | grip | sum |','|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for cat in ('fixed','variable'):
        for m in r[cat]['methods']:
            for split in ('learned','unseen'):
                for metric,summary in [('compression',m['compression'][split]),
                    ('upper7',m['eval'][split]['upper7']),
                    ('upper6_choice_arm_score',m['eval'][split]['upper6_choice_arm_score'])]:
                    for subset in ('all','hard'):
                        v=summary[subset]
                        if v is None: continue
                        vals=[v['per_axis'][a] for a in AXES]
                        lines.append(f"| {m['method']} | {split} | {metric} | {subset} | " + ' | '.join(f'{x:.6f}' for x in vals) + f" | {v['axis_sum']:.6f} |")
    lines += ['', '## 対応 bootstrap 95%区間','', '| category | split / baseline | metric / subset | AE−baseline mean | 95% CI |','|---|---|---|---:|---:|']
    for cat,vals in r['paired_ci'].items():
        for k,vsets in vals.items():
            for metric,subsets in vsets.items():
                for subset,v in subsets.items(): lines.append(f"| {cat} | {k} | {metric} / {subset} | {v['mean_difference']:.6f} | [{v['ci95'][0]:.6f}, {v['ci95'][1]:.6f}] |")
    lines += ['', 'CMP3 G1 の既知値は units=4516, unseen all/hard=.0344479/.0684542。全件評価時に照合し、一致しない場合は停止する。AE seed と Dev 選択は results.json に記録。','',
      '## 本番実行コマンド','', '```bash','cd /work && source env/activate.sh',
      f"taskset -c 8-47 nice -n 10 env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 python skill_moe/skillnet/examples/memory_pilot/unseen_libero/compression_ae21.py --full --output {OUT_DEFAULT}",'```','']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True); g.add_argument('--smoke',action='store_true'); g.add_argument('--full',action='store_true')
    p.add_argument('--output',type=Path,default=OUT_DEFAULT); p.add_argument('--device',default=None)
    a=p.parse_args(); r=run(a.smoke,a.output,device=a.device)
    print(json.dumps({'output':str(a.output),'mode':r['mode'],'G1':r['protocol']['G1_reproduction']},indent=2))

if __name__=='__main__': main()
