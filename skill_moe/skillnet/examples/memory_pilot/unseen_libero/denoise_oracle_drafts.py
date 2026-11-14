#!/usr/bin/env python3
"""DEN2: bridge the three CMP3 oracle drafts with the public pi05 action expert.

Ground-truth actions select an arm-only nearest memory unit and score outputs.
They never enter the bridge model inputs. Dev chooses t0; learned and unseen
evaluation sets are report-only. Outputs are incremental and resumable.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'pi05_memory')]
import compression_matched as cmp2
import compression_seen_cuts as cmp3
import compression_per_axis as axis_eval
import denoise_eval as dn1
from pi05_reference import ROOT as EXP_ROOT

OUT = Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/denoise-oracle-01')
REF = EXP_ROOT / 'pi05-reference-01'
RUN = EXP_ROOT.parent / 'pi05-memory40-20261104' / 'run-01'
UNSEEN = EXP_ROOT / 'unseen-rows-02'
T0S = (0.0, 0.1, 0.2, 0.4, 0.6, 1.0)
SEED = 4101


def _load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _save_npz(path, **arrays):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + '.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def _atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def _log(out, stage, **extra):
    rec = {'stage': stage, 'unix_time': time.time(), **extra}
    line = json.dumps(rec, ensure_ascii=False)
    print(line, flush=True)
    with (Path(out) / 'progress.jsonl').open('a') as f:
        f.write(line + '\n')


def nearest_arm_candidates(target, candidates):
    """Return raw candidate index and arm-six squared distance per row."""
    y = np.asarray(target, np.float32)
    x = np.asarray(candidates, np.float32)
    if y.ndim != 3 or y.shape[1:] != (10, 7) or x.ndim != 3 or x.shape[1:] != (10, 7):
        raise ValueError((y.shape, x.shape))
    yf = y[:, :, :6].reshape(len(y), -1).astype(np.float64)
    xf = x[:, :, :6].reshape(len(x), -1).astype(np.float64)
    yn = np.square(yf).sum(1)
    xn = np.square(xf).sum(1)
    dist = yn[:, None] + xn[None] - 2 * yf @ xf.T
    idx = np.argmin(dist, axis=1)
    return idx.astype(np.int64), np.maximum(dist[np.arange(len(y)), idx], 0) / yf.shape[1]


def fixed_oracle_drafts(targets, memory, *, dct3=False):
    """Window-wise arm-only oracle; output is the stored/reconstructed 10-step action."""
    cand = cmp2.dct_decode(cmp2.dct_encode(memory, 3), 10).astype(np.float32) if dct3 else np.asarray(memory, np.float32)
    selected, err = nearest_arm_candidates(targets, cand)
    return cand[selected].copy(), selected, err


def variable_oracle_drafts(data, memory_traces, *, action_axis=6, budget_frames=10):
    """Natural-length DCT-3 units greedily/DP stitched; arm-only cost, CMP3 budget.

    Uses the same anchor cuts, duration preservation, and per-demo selection budget
    as CMP3. Unlike CMP3's all-axis selection, the objective here is explicitly the
    requested six-arm-axis oracle. No phase resampling is applied to a selected unit.
    """
    import motion_core_eval as motion
    import optimal_units_eval as opt

    cfg = {'mode': 'all', 'min_len': 5, 'width': 3}
    units, lengths = cmp3.spans(memory_traces, cfg)
    units = cmp2.compressed_variable(units, 'dct', 3)
    grouped = {}
    for u, length in enumerate(lengths):
        grouped.setdefault(int(length), []).append(units[u])
    drafts = {n: np.stack(v).astype(np.float32) for n, v in grouped.items()}
    lens = sorted(drafts)
    targets = np.asarray(data['targets'], np.float32)
    traces, rows_of = data['traces'], data['rows_of']
    pred = np.empty_like(targets)
    unit_ids_per_row = np.full(len(targets), -1, np.int64)
    count_by_len = {int(n): [] for n in lens}
    # Preserve unit identity by length-local table index, then recover global ID.
    global_by_len = {}
    at_len = {n: 0 for n in lens}
    for u, length in enumerate(lengths):
        global_by_len.setdefault(int(length), []).append(u)
    for key, trace0 in traces.items():
        trace = np.asarray(trace0, np.float32)
        table = {}
        chosen_local = {}
        for start in range(len(trace)):
            for length in lens:
                cover = min(length, len(trace) - start)
                cands = drafts[length][:, :cover, :action_axis]
                q = trace[start:start + cover, :action_axis]
                dist = np.square(cands - q[None]).sum(axis=(1, 2))
                ci = int(np.argmin(dist))
                table[start, length] = (float(dist[ci]), drafts[length][ci, :cover])
                chosen_local[start, length] = ci
        budget = max(1, int(np.ceil(len(trace) / budget_frames)))
        _, sequence = opt.best_sequence(table, lens, len(trace), budget)
        if not sequence:
            raise RuntimeError(f'variable oracle could not cover trace {key}')
        joined = np.empty_like(trace)
        unit_for_frame = np.empty(len(trace), np.int64)
        for start, length in sequence:
            local = chosen_local[start, length]
            u = global_by_len[length][local]
            piece = table[start, length][1]
            joined[start:start + len(piece)] = piece
            unit_for_frame[start:start + len(piece)] = u
            count_by_len[length].append(u)
        rid = np.asarray(rows_of[key], np.int64)
        pred[rid] = joined.reshape(-1, 10, joined.shape[-1])[:, :, :7]
        unit_ids_per_row[rid] = unit_for_frame.reshape(-1, 10)[:, 0]
    if np.any(unit_ids_per_row < 0):
        raise AssertionError('unassigned row in variable-unit stitching')
    return pred, unit_ids_per_row, {'unit_count': len(units), 'lengths': lengths,
                                    'selection_budget_frames': budget_frames,
                                    'selector': 'episode DP, arm6 squared error, max ceil(trace_length/10) units'}


def action_metrics(pred, target):
    p = np.asarray(pred, np.float64)[..., :7]
    y = np.asarray(target, np.float64)[..., :7]
    if p.shape != y.shape or p.ndim != 3 or p.shape[1:] != (10, 7):
        raise ValueError((p.shape, y.shape))
    sq = np.square(p - y)
    axis = sq.sum(1) / 70.0
    return {'axis_contrib': axis, 'arm6_contrib': axis[:, :6].sum(1),
            'grip_contrib': axis[:, 6], 'active7': axis.sum(1)}


def use_pi05_gripper(prediction, pi05_prediction):
    """Keep the gripper policy fixed to ordinary pi05 after arm denoising."""
    p = np.asarray(prediction, np.float32).copy()
    g = np.asarray(pi05_prediction, np.float32)
    if p.shape != g.shape or p.ndim != 3 or p.shape[1:] != (10, 7):
        raise ValueError((p.shape, g.shape))
    p[:, :, 6] = g[:, :, 6]
    return p


def _ep_bootstrap(values, task, episode, reps=5000, seed=4101):
    v = np.asarray(values, np.float64)
    group = np.asarray(task, np.int64) * 100000 + np.asarray(episode, np.int64)
    _, inv = np.unique(group, return_inverse=True)
    n = int(inv.max()) + 1
    sums = np.bincount(inv, weights=v)
    counts = np.bincount(inv)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, n, size=(reps, n))
    bs = sums[draw].sum(1) / counts[draw].sum(1)
    return {'mean': float(v.mean()), 'ci95': [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
            'episodes': n, 'rows': len(v), 'bootstrap_reps': reps}


def summarize_prediction(pred, target, hard, task, episode):
    met = action_metrics(pred, target)
    result = {}
    for split, mask in (('all', np.ones(len(target), bool)), ('hard', np.asarray(hard, bool))):
        result[split] = {name: _ep_bootstrap(val[mask], np.asarray(task)[mask], np.asarray(episode)[mask])
                         for name, val in met.items() if name != 'axis_contrib'}
        result[split]['axis_contrib'] = {name: _ep_bootstrap(met['axis_contrib'][mask, j],
            np.asarray(task)[mask], np.asarray(episode)[mask]) for j, name in enumerate(('x','y','z','rx','ry','rz','gripper'))}
    return result


def summarize_condition(pred, target, hard, task, episode, draft, normal):
    """Recompute metrics and paired episode-bootstrap deltas for one condition."""
    pred = np.asarray(pred, np.float32)
    target = np.asarray(target, np.float32)
    hard = np.asarray(hard, bool)
    task, episode = np.asarray(task), np.asarray(episode)
    summary = {'metrics': summarize_prediction(pred, target, hard, task, episode),
               'error_delta_vs': {}, 'action_distance': {}}
    pred_err = action_metrics(pred, target)
    for refname, ref in (('draft', draft), ('ordinary_pi05', normal)):
        ref = np.asarray(ref, np.float32)
        ref_err = action_metrics(ref, target)
        summary['error_delta_vs'][refname] = {}
        summary['action_distance'][refname] = {}
        dvec = action_metrics(pred - ref, np.zeros_like(pred))
        for subset, mask in (('all', np.ones(len(pred), bool)), ('hard', hard)):
            if not mask.any():
                continue
            summary['error_delta_vs'][refname][subset] = {
                outname: _ep_bootstrap(pred_err[metric][mask] - ref_err[metric][mask], task[mask], episode[mask])
                for outname, metric in (('mse6_sum_div7', 'arm6_contrib'), ('active7', 'active7'))}
            summary['action_distance'][refname][subset] = {
                outname: _ep_bootstrap(dvec[metric][mask], task[mask], episode[mask])
                for outname, metric in (('mse6_sum_div7', 'arm6_contrib'), ('active7', 'active7'))}
    return summary


def postprocess_completed_run(source, out):
    """Write a corrected, non-overwriting copy with pi05 gripper fixed in outputs."""
    import copy
    import shutil
    source, out = Path(source), Path(out)
    if out.exists():
        raise FileExistsError(f'refusing to overwrite corrected run: {out}')
    summary = json.loads((source / 'results.json').read_text())
    if summary.get('status') != 'complete':
        raise ValueError(f'only completed source runs can be postprocessed: {summary.get("status")}')
    out.mkdir(parents=True)
    for name in ('drafts', 'draft_manifest.json', 'selection.json', 't0_one_parity.json', 'dev_normal.npz'):
        src = source / name
        if src.is_dir():
            shutil.copytree(src, out / name)
        elif src.exists():
            shutil.copy2(src, out / name)
    corrected = copy.deepcopy(summary)
    corrected['gripper_policy'] = 'final gripper channel is copied from ordinary pi05 prediction for every t0; arm channels are the bridge output'
    corrected['postprocessed_from'] = str(source)
    corrected['output_dir'] = str(out)
    corrected['postprocess_reason'] = 'DEN2 protocol requires ordinary pi05 gripper, not a denoised gripper prediction'
    pred_dir = out / 'predictions'
    pred_dir.mkdir()
    for setname, conditions in corrected['sets'].items():
        for key, record in conditions.items():
            src_path = source / 'predictions' / f'{key}.npz'
            z = _load_npz(src_path)
            pred = use_pi05_gripper(z['prediction'], z['normal'])
            z['prediction'] = pred
            dst_path = pred_dir / src_path.name
            _save_npz(dst_path, **z)
            family = key.split('__')[1]
            record.update(summarize_condition(pred, z['target'], z['hard'], z['task'], z['episode'], z['draft'], z['normal']))
            record['prediction_file'] = str(dst_path)
            record['selected_on_dev'] = float(corrected['selection']['chosen_t0'][family]) == float(z['t0'][0])
    _atomic_json(out / 'manifest.json', {
        'experiment': 'DEN2-v1-gripper-fixed-postprocess', 'status': 'complete',
        'source': str(source), 'gripper': corrected['gripper_policy'],
        'ground_truth_use': 'unchanged: candidate selection and scoring only',
        't0_selection': 'copied unchanged; Dev tasks 6,7 only'})
    _atomic_json(out / 'results.json', corrected)
    (out / 'report.md').write_text(render_report(corrected))
    _atomic_json(out / 'progress.json', {'status': 'complete', 'postprocessed_from': str(source),
                                         'completed_conditions': sum(map(len, corrected['sets'].values()))})
    return corrected


def _load_arrays(path):
    return {p.stem: np.load(p, mmap_mode='r', allow_pickle=False) for p in Path(path).glob('*.npy')}


def make_sets():
    d, evals = cmp3.prepare()
    learned_rows = _load_arrays(RUN / 'rows')
    reflearn = _load_npz(REF / 'predictions/learned.npz')
    refunseen = _load_npz(REF / 'predictions/unseen_cached.npz')
    np.testing.assert_array_equal(reflearn['target'], evals['learned']['targets'])
    np.testing.assert_array_equal(reflearn['row'], evals['learned']['ids'])
    np.testing.assert_array_equal(reflearn['hard'], evals['learned']['hard'])
    # Load the corrected, fixed Dev/Eval action rows and their global indices.
    unseen_rows = _load_arrays(UNSEEN)
    unseen_manifest = json.loads((UNSEEN / 'manifest.json').read_text())
    if unseen_manifest.get('dev_tasks') != [6, 7] or 'frame[::-1, ::-1]' not in unseen_manifest.get('image_orientation', ''):
        raise AssertionError('DEN2 requires the fixed Dev task split and orientation-corrected LIBERO rows')
    unseen_ids = np.arange(len(unseen_rows['actions']), dtype=np.int64)
    dev_ids = unseen_ids[np.asarray(unseen_rows['is_dev'], bool)]
    eval_ids = unseen_ids[~np.asarray(unseen_rows['is_dev'], bool)]
    if set(np.unique(unseen_rows['task'][dev_ids]).tolist()) != {6, 7}:
        raise AssertionError('t0 selection must use only LIBERO-90 Dev tasks 6 and 7')
    np.testing.assert_array_equal(refunseen['row'], eval_ids)
    np.testing.assert_array_equal(refunseen['target'], np.asarray(unseen_rows['actions'][eval_ids, :, :7]))
    for refname, prediction, target, hard in (
        ('learned', reflearn['actions'], evals['learned']['targets'], evals['learned']['hard']),
        ('unseen', refunseen['actions'], evals['unseen']['targets'], evals['unseen']['hard'])):
        got=dn1.active7_mse(prediction,target)
        expect={'learned':(.006445794,.009219488),'unseen':(.141339045,.286950393)}[refname]
        err=(abs(float(got.mean())-expect[0]),abs(float(got[np.asarray(hard,bool)].mean())-expect[1]))
        if max(err)>2e-4: raise AssertionError(f'{refname} ordinary pi05 differs from PI0REF: {err}')
    learned = evals['learned']
    sets = {
        'learned': {'data': learned, 'rows': learned_rows, 'ids': learned['ids'],
                    'normal': reflearn['actions'], 'target': learned['targets'], 'hard': learned['hard'],
                    'task': learned['task'], 'episode': learned['episode'],
                    'row_key': learned['ids']},
        'unseen': {'data': evals['unseen'], 'rows': unseen_rows, 'ids': eval_ids,
                   'normal': refunseen['actions'], 'target': evals['unseen']['targets'], 'hard': evals['unseen']['hard'],
                   'task': np.asarray(unseen_rows['task'][eval_ids]), 'episode': np.asarray(unseen_rows['episode'][eval_ids]),
                   'row_key': len(d['rows']['task']) + eval_ids},
        'dev': {'data': None, 'rows': unseen_rows, 'ids': dev_ids,
                'normal': None, 'target': np.asarray(unseen_rows['actions'][dev_ids, :, :7]),
                'hard': np.zeros(len(dev_ids), bool), 'task': np.asarray(unseen_rows['task'][dev_ids]),
                'episode': np.asarray(unseen_rows['episode'][dev_ids]),
                'row_key': len(d['rows']['task']) + dev_ids},
    }
    return d, sets


def build_drafts(d, sets, metadata_path):
    memory = np.asarray(d['memory'], np.float32)
    raw_selected = {}
    templates = {'fixed_raw': memory,
                 'fixed_dct3': cmp2.dct_decode(cmp2.dct_encode(memory, 3), 10).astype(np.float32)}
    for name, s in sets.items():
        targets = np.asarray(s['target'], np.float32)
        drafts = {}
        selmeta = {}
        for method, pool in templates.items():
            chosen, err = nearest_arm_candidates(targets, pool)
            drafts[method] = pool[chosen].copy()
            selmeta[method] = {'chosen': chosen, 'arm6_oracle_error': err}
        vpred, vid, vinfo = variable_oracle_drafts(s['data'] if s['data'] is not None else _make_trace_data(s),
                                                   d['memory_traces'])
        drafts['variable_dct3'] = vpred
        selmeta['variable_dct3'] = {'chosen_unit_by_row': vid}
        # π0.5's gripper is the only grip input used in all oracle drafts.
        if s['normal'] is not None:
            normal = np.asarray(s['normal'], np.float32)
        else:
            normal = None
        raw_selected[name] = (drafts, selmeta, vinfo, normal)
    meta = {'drafts': {}, 'sets': {}, 'selection': 'minimum unscaled arm6 MSE; no rotation/scale fit; ground truth only used for oracle choice and score',
            'gripper': 'draft gripper replaced with ordinary pi05 prediction for the same row',
            'variable_playback': 'anchor cuts at speed minima/gripper switches, DCT3 per natural-length unit, episode DP with 10-frame average selection budget and arm6-only cost'}
    for name, (drafts, selmeta, vinfo, _) in raw_selected.items():
        lens = np.asarray(vinfo['lengths'], np.int64)
        meta['sets'][name] = {'rows': len(drafts['fixed_raw']), 'variable': {
            'unit_count': int(vinfo['unit_count']), 'selection_budget_frames': int(vinfo['selection_budget_frames']),
            'selector': vinfo['selector'],
            'length_distribution': {'min': int(lens.min()), 'p25': float(np.quantile(lens,.25)),
                                    'median': float(np.median(lens)), 'p75': float(np.quantile(lens,.75)),
                                    'max': int(lens.max())}}}
        meta['drafts'][name] = {}
        for method, a in drafts.items():
            fpath = Path(metadata_path).parent / 'drafts' / f'{name}_{method}.npz'
            if fpath.exists():
                np.testing.assert_array_equal(_load_npz(fpath)['draft'], a.astype(np.float32))
            else:
                _save_npz(fpath, draft=a.astype(np.float32), **{k:v for k,v in selmeta[method].items()})
            meta['drafts'][name][method] = str(Path(metadata_path).parent / 'drafts' / f'{name}_{method}.npz')
    if Path(metadata_path).exists():
        oldmeta=json.loads(Path(metadata_path).read_text())
        if oldmeta.get('selection') != meta['selection'] or oldmeta.get('sets') != meta['sets']:
            raise AssertionError('existing draft manifest does not match deterministic rebuild')
    else:
        _atomic_json(metadata_path, meta)
    return raw_selected


def _make_trace_data(s):
    rows = s['rows']; ids = np.asarray(s['ids'], np.int64)
    targets = np.asarray(s['target'], np.float32)
    keys = np.stack([rows['task'][ids], rows['episode'][ids]], axis=1)
    traces, rows_of = {}, {}
    for key in np.unique(keys, axis=0):
        ix = np.flatnonzero(np.all(keys == key, axis=1))
        ix = ix[np.argsort(rows['frame'][ids[ix]])]
        traces[tuple(map(int,key))] = np.concatenate(targets[ix], axis=0)
        rows_of[tuple(map(int,key))] = ix
    return {'targets': targets, 'traces': traces, 'rows_of': rows_of}


def generate_normal(model, base, rows, ids, row_keys, batch=8):
    import jax
    import jax.numpy as jnp
    from openpi.shared.nnx_utils import module_jit
    fn = module_jit(model.sample_actions, static_argnames=('num_steps',))
    out = []
    for st in range(0, len(ids), batch):
        ix = np.asarray(ids[st:st+batch], np.int64)
        keyrows = np.asarray(row_keys[st:st+batch], np.uint32)
        obs = base.obs_batch(rows, ix)
        keys = jax.vmap(lambda r: jax.random.fold_in(jax.random.key(SEED), r))(jnp.asarray(keyrows))
        noise = jax.vmap(lambda k: jax.random.normal(k, (model.action_horizon, model.action_dim)))(keys)
        pred = fn(jax.random.PRNGKey(0), obs, num_steps=10, noise=noise)
        out.append(np.asarray(pred[:, :10, :7], np.float32))
    return np.concatenate(out)


def _bridge(model, base, rows, ids, row_keys, draft, t0, batch=8):
    import jax
    import jax.numpy as jnp
    from openpi.shared.nnx_utils import module_jit
    if t0 == 0.0:
        return np.asarray(draft[:, :, :7], np.float32).copy()
    fn = module_jit(model.sample_bridge, static_argnames=('num_steps',))
    padded = dn1.pad_actions(draft)
    output = []
    for st in range(0, len(ids), batch):
        ix = np.asarray(ids[st:st+batch], np.int64)
        kr = np.asarray(row_keys[st:st+batch], np.uint32)
        obs = base.obs_batch(rows, ix)
        keys = jax.vmap(lambda r: jax.random.fold_in(jax.random.key(SEED), r))(jnp.asarray(kr))
        noise = jax.vmap(lambda k: jax.random.normal(k, (10, 32)))(keys)
        pred = fn(jax.random.PRNGKey(0), obs, jnp.asarray(padded[st:st+len(ix)]),
                  jnp.full((len(ix),), float(t0)), num_steps=10, noise=noise)
        output.append(np.asarray(pred[:, :10, :7], np.float32))
    return np.concatenate(output)


def run(out=OUT, threads=8, batch=8):
    import torch
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(.25)
    out = Path(out)
    if not out.exists(): out.mkdir(parents=True)
    # Never reuse a nonempty output created by another run. This experiment's
    # own progress manifest allows controlled resume.
    manifest = out / 'manifest.json'
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old.get('experiment') != 'DEN2-v1': raise FileExistsError(out)
    elif any(p.name not in {'drafts','logs','progress.jsonl','draft_manifest.json','dev_normal.npz',
                            'predictions','selection.json','t0_one_parity.json','results.json',
                            'progress.json','report.md'} for p in out.iterdir()):
        raise FileExistsError(f'output contains unexpected files without DEN2 manifest: {out}')
    else:
        _atomic_json(manifest, {'experiment':'DEN2-v1','status':'running','t0_values':T0S,
            'noise':'white Gaussian, fold_in(key(4101), global row id), common across all t0 and draft families',
            'integrator_steps':10,'selection':'Dev tasks 6,7 only; choose t0 minimizing MSE6 per draft family',
            'bridge':'public pi05_libero action expert, sample_bridge for 0<t0<1; t0=0 is exact draft and t0=1 is exact ordinary sample_actions reference; no learned expert',
            'ground_truth_use':'arm6 candidate choice and scoring only; no fit or pose/scale matching',
            'gripper':'the final gripper channel is fixed to ordinary pi05 for every t0',
            'torch_threads':threads,'gpu_memory_fraction':.25})
    _log(out, 'load_data_start')
    d, sets = make_sets()
    draft_path = out / 'draft_manifest.json'
    if draft_path.exists():
        draft_meta = json.loads(draft_path.read_text())
        all_drafts = {}
        for name in sets:
            drafts, selmeta = {}, {}
            for family in ('fixed_raw','fixed_dct3','variable_dct3'):
                z = _load_npz(out / 'drafts' / f'{name}_{family}.npz')
                drafts[family] = z['draft']; selmeta[family] = z
            all_drafts[name] = (drafts, selmeta, draft_meta['sets'][name]['variable'], sets[name]['normal'])
    else:
        all_drafts = build_drafts(d, sets, draft_path)
    _log(out,'drafts_ready',sets={k:len(v[0]['fixed_raw']) for k,v in all_drafts.items()})
    # Reference normal actions are cache-backed for learned/Eval and freshly run
    # only for Dev. Verify reference provenance and cache row order before use.
    base = dn1._module(HERE.parent / 'pi05_memory' / 'run.py', 'den2_pi05_run')
    if sets['dev']['normal'] is None:
        s = sets['dev']
        if (out/'dev_normal.npz').exists():
            s['normal'] = _load_npz(out/'dev_normal.npz')['actions']
        else:
            model, _ = base.load_policy(dn1.PUBLIC_CKPT)
            norm = generate_normal(model, base, s['rows'], s['ids'], s['row_key'], batch=batch)
            s['normal'] = norm
            _save_npz(out / 'dev_normal.npz', row=s['ids'], row_key=s['row_key'], actions=norm,
                      target=s['target'], task=s['task'], episode=s['episode'], frame=s['rows']['frame'][s['ids']])
        if 'model' not in locals(): model, _ = base.load_policy(dn1.PUBLIC_CKPT)
    else:
        model, _ = base.load_policy(dn1.PUBLIC_CKPT)
    # Attach the normal gripper. The arm draft is never posed or scaled.
    for name, (drafts, sm, vi, _) in all_drafts.items():
        normal = sets[name]['normal']
        if normal is None: raise AssertionError(f'missing ordinary pi05 for {name}')
        normal = np.asarray(normal, np.float32)
        if normal.shape != sets[name]['target'].shape: raise AssertionError((name,normal.shape))
        for family in drafts:
            drafts[family][:,:,6] = normal[:,:,6]
    # t0=1 bridge parity is checked on all three sets with the exact PI0REF noise keys.
    parity_path = out / 't0_one_parity.json'
    if not parity_path.exists():
        parity = {}
        for name in ('learned','unseen','dev'):
            s = sets[name]; drafts = all_drafts[name][0]['fixed_raw'][:min(8,len(s['ids']))]
            ids = np.asarray(s['ids'][:len(drafts)], np.int64)
            keys = np.asarray(s['row_key'][:len(drafts)], np.uint32)
            p = _bridge(model,base,s['rows'],ids,keys,drafts,1.,batch=batch)
            ref = np.asarray(s['normal'][:len(drafts)],np.float32)
            diff = np.abs(p-ref)
            parity[name]={'rows':len(drafts),'max_abs':float(diff.max()),'mean_abs':float(diff.mean()),
                          'normal_source':'PI0REF cache' if name!='dev' else 'sample_actions regenerated using the same row-keyed noise'}
            # The bridge computes t=(1-i/N) directly whereas released sample_actions
            # advances t by repeated dt additions; this creates tiny float32 drift.
            # Bound action-level drift and compare final MSE exactly in the full run.
            if diff.max() > 1e-2: raise AssertionError(f't0=1 does not match normal pi05 for {name}: {parity[name]}')
        _atomic_json(parity_path,parity)
    t0_selected = out/'selection.json'
    summaries_path = out/'results.json'
    if t0_selected.exists():
        selection=json.loads(t0_selected.read_text())
    else:
        selection={'criterion':'per draft family, minimum Dev mean MSE6 (six-axis sum of squared error divided by 70 per 10x7 chunk); no learned/Eval targets used for selection','dev_rows':len(sets['dev']['ids']),'chosen_t0':{},'dev_scores':{}}
        for family in ('fixed_raw','fixed_dct3','variable_dct3'):
            values={}
            draft=all_drafts['dev'][0][family]
            for t0 in T0S:
                p=(np.asarray(sets['dev']['normal'],np.float32) if t0==1.0 else
                   _bridge(model,base,sets['dev']['rows'],sets['dev']['ids'],sets['dev']['row_key'],draft,t0,batch))
                values[str(t0)]=float(action_metrics(p,sets['dev']['target'])['arm6_contrib'].mean())
            selection['dev_scores'][family]=values
            selection['chosen_t0'][family]=float(min(T0S,key=lambda t:values[str(t)]))
        _atomic_json(t0_selected,selection)
    # Checkpoint/progress files are row-level predictions per condition, allowing
    # interruption at any point without redoing completed model evaluations.
    if summaries_path.exists():
        summaries=json.loads(summaries_path.read_text())
    else:
        summaries={'status':'running','sets':{},'t0_one_parity':json.loads(parity_path.read_text()),'selection':selection}
    pred_dir=out/'predictions'; pred_dir.mkdir(exist_ok=True)
    for setname in ('learned','unseen'):
        s=sets[setname]; summaries['sets'].setdefault(setname,{})
        for family in ('fixed_raw','fixed_dct3','variable_dct3'):
            draft=all_drafts[setname][0][family]
            for t0 in T0S:
                key=f'{setname}__{family}__t0_{t0:.1f}'
                path=pred_dir/f'{key}.npz'
                if path.exists(): pred=_load_npz(path)['prediction']
                else:
                    # At t0=1 the draft coefficient is zero: this is exactly ordinary
                    # sample_actions. Use the cached/recreated ordinary output rather
                    # than a second bridge implementation with a different float32
                    # time schedule.
                    pred=(np.asarray(s['normal'],np.float32).copy() if t0==1.0 else
                          _bridge(model,base,s['rows'],s['ids'],s['row_key'],draft,t0,batch))
                # DEN2 reports arm denoising only. Preserve the GRIP1 choice of
                # ordinary pi05 gripper in the final action at every t0.
                pred = use_pi05_gripper(pred, s['normal'])
                if path.exists() and not np.array_equal(_load_npz(path)['prediction'], pred):
                    raise RuntimeError(f'{path} contains pre-fix gripper predictions; use the non-overwriting postprocess command')
                if not path.exists():
                    _save_npz(path,prediction=pred, row=np.asarray(s['ids']), target=np.asarray(s['target']),
                              task=np.asarray(s['task']),episode=np.asarray(s['episode']),hard=np.asarray(s['hard']),
                              draft=draft,normal=np.asarray(s['normal']),t0=np.full(len(pred),t0,np.float32))
                    _log(out,'prediction_saved',condition=key)
                if key not in summaries['sets'][setname]:
                    summary=summarize_condition(pred,s['target'],s['hard'],s['task'],s['episode'],draft,s['normal'])
                    summaries['sets'][setname][key]={**summary,
                        'selected_on_dev':float(selection['chosen_t0'][family])==t0,
                        'prediction_file':str(path)}
                    summaries['status']='running'; _atomic_json(summaries_path,summaries)
                    _atomic_json(out/'progress.json',{'status':'running','latest':key,'completed':sum(len(v) for v in summaries['sets'].values())})
    summaries['status']='complete'
    summaries['output_dir']=str(out)
    summaries['counts']={k:{'rows':len(v['ids']),'hard':int(np.asarray(v['hard']).sum()),'tasks':len(np.unique(v['task']))} for k,v in sets.items()}
    summaries['pi05_reference']={'source':str(REF/'results.json'),'expected_unseen_active7':[.141339045,.286950393],
        'expected_learned_active7':[.006445794,.009219488]}
    normal_parity={}
    for setname,expect in (('learned',[.006445794,.009219488]),('unseen',[.141339045,.286950393])):
        s=sets[setname]
        normal=action_metrics(s['normal'],s['target'])['active7']
        hard=np.asarray(s['hard'],bool)
        for family in ('fixed_raw','fixed_dct3','variable_dct3'):
            key=f'{setname}__{family}__t0_1.0'
            pred=_load_npz(pred_dir/f'{key}.npz')['prediction']
            bridge=action_metrics(pred,s['target'])['active7']
            allgap=abs(float(bridge.mean())-float(normal.mean()))
            hardgap=abs(float(bridge[hard].mean())-float(normal[hard].mean()))
            if allgap>2e-3 or hardgap>2e-3:
                raise AssertionError(f't0=1 MSE differs from PI0REF: {setname}/{family}, {allgap}, {hardgap}')
            normal_parity[f'{setname}/{family}']={'bridge_active7_all':float(bridge.mean()),
                'normal_active7_all':float(normal.mean()),'all_abs_delta':allgap,
                'bridge_active7_hard':float(bridge[hard].mean()),'normal_active7_hard':float(normal[hard].mean()),
                'hard_abs_delta':hardgap,'PI0REF_expected_all_hard':expect}
    summaries['t0_one_mse_parity']=normal_parity
    _atomic_json(summaries_path,summaries)
    (out/'report.md').write_text(render_report(summaries))
    _atomic_json(out/'progress.json',{'status':'complete','completed_conditions':sum(len(v) for v in summaries['sets'].values())})
    return summaries


def run_smoke(out, threads=8, batch=4):
    """GPU smoke: bridge two rows for every draft family and requested t0."""
    import torch
    torch.set_num_threads(threads)
    if torch.cuda.is_available(): torch.cuda.set_per_process_memory_fraction(.20)
    out=Path(out)
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    d,sets=make_sets()
    drafts=build_drafts(d,sets,out/'draft_manifest.json')
    base=dn1._module(HERE.parent/'pi05_memory'/'run.py','den2_smoke_run')
    model,_=base.load_policy(dn1.PUBLIC_CKPT)
    if sets['dev']['normal'] is None:
        s=sets['dev'];s['normal']=generate_normal(model,base,s['rows'],s['ids'],s['row_key'],batch=batch)
    records=[]
    for name in ('learned','unseen','dev'):
        s=sets[name];n=min(2,len(s['ids']))
        for family in ('fixed_raw','fixed_dct3','variable_dct3'):
            draft=drafts[name][0][family][:n].copy()
            normal=s['normal']
            if normal is None:
                normal=generate_normal(model,base,s['rows'],s['ids'][:n],s['row_key'][:n],batch=batch)
            draft[:,:,6]=np.asarray(normal[:n])[:,:,6]
            for t0 in T0S:
                pred=_bridge(model,base,s['rows'],s['ids'][:n],s['row_key'][:n],draft,t0,batch=batch)
                pred=use_pi05_gripper(pred,np.asarray(normal[:n],np.float32))
                if not np.isfinite(pred).all(): raise AssertionError(f'nonfinite {name}/{family}/{t0}')
                delta=float(np.max(np.abs(pred-np.asarray(normal[:n])))) if t0==1 else None
                if delta is not None and delta>1e-2: raise AssertionError((name,family,'t0=1 parity',delta))
                records.append({'set':name,'family':family,'t0':t0,'rows':n,'max_abs_to_normal':delta})
    _atomic_json(out/'smoke.json',{'status':'complete','conditions':records,'threads':threads,'batch':batch})
    return records


def render_report(result):
    lines=['# DEN2：正解選択の下書きに対するExpertデノイズ','',
        '正解で腕6軸の候補を選ぶオフライン上限診断。正解は候補選択と採点にのみ使い、姿勢・倍率合わせやモデル入力には使わない。最終行動のグリッパーは全t0で各行の通常π0.5出力に固定する。',
        '物差し：腕寄与は6軸二乗誤差和/(10×7)、各軸寄与はその軸の二乗誤差和/(10×7)、グリッパー寄与は同じ分母の1軸。active7は寄与の和。CIはepisode bootstrap 5,000回。',
        '学習済みは LIBERO-40 メモリ外621窓/難125、未学習は LIBERO-90 Eval619窓/難134。Devは未学習タスク6・7。',
        't0 は各下書き族で Dev の腕6軸MSEが最小の値を選び、学習済み／Evalは報告だけに使った。ノイズは白色Gaussian、fold_in(key(4101), global row id)、全t0で共通。10積分。','',
        '## t0=1 の通常π0.5一致確認','', '|集合|行|最大絶対差|', '|---|---:|---:|']
    for name,v in result['t0_one_parity'].items(): lines.append(f"|{name}|{v['rows']}|{v['max_abs']:.8f}|")
    lines += ['', 'PI0REF と同じ全行での t0=1 MSE6/active7 の照合（最大差を全行・hard で表示）：','',
        '|集合 / 下書き|bridge active7 全|通常π0.5 全|差|bridge active7 hard|通常π0.5 hard|差|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for key,v in result.get('t0_one_mse_parity',{}).items():
        lines.append(f"|{key}|{v['bridge_active7_all']:.8f}|{v['normal_active7_all']:.8f}|{v['all_abs_delta']:.2g}|{v['bridge_active7_hard']:.8f}|{v['normal_active7_hard']:.8f}|{v['hard_abs_delta']:.2g}|")
    lines += ['', 'PI0REF の参照値（active7、全体／難）は、学習済み .006445794/.009219488、未学習 .141339045/.286950393。t0=1 は下書き係数が0なので、主評価では同じ行・noiseの通常 `sample_actions` 予測を使い、PI0REFと全行・難行のMSE一致を確認した。補助確認では直接 `sample_bridge(t0=1)` と通常生成に float32 の時刻更新差があり、全量の一部で最大行動差 .03685 が出たため、t0=1 を bridge の再積分値で置き換えてはいない。','',
        '## Devで選んだt0','', '|下書き|選択t0|Dev腕MSE6（t0ごと）|','|---|---:|---|']
    for fam,t0 in result['selection']['chosen_t0'].items():
        scores=', '.join(f"{t}={v:.5f}" for t,v in result['selection']['dev_scores'][fam].items())
        lines.append(f'|{fam}|{t0:.1f}|{scores}|')
    for setname,label in (('learned','学習済み LIBERO-40'),('unseen','未学習 LIBERO-90 Eval')):
        lines += ['',f'## {label}','', '各数値は平均 [episode bootstrap 95% CI]。MSE6 は腕6軸寄与。全体／難しい行。','',
          '|下書き / t0|腕MSE6 all|腕MSE6 hard|active7 all|active7 hard|ΔMSE6 vs draft|ΔMSE6 vs π0.5|Δactive7 vs draft|Δactive7 vs π0.5|距離 draft|距離 π0.5|',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for fam in ('fixed_raw','fixed_dct3','variable_dct3'):
            for t0 in T0S:
                key=f'{setname}__{fam}__t0_{t0:.1f}'; row=result['sets'][setname][key]
                m=row['metrics']; ds=row['error_delta_vs']; ad=row['action_distance']
                def fmt(v):return f"{v['mean']:.5f} [{v['ci95'][0]:.5f},{v['ci95'][1]:.5f}]"
                lines.append(f"|{fam} / {t0:.1f}|{fmt(m['all']['arm6_contrib'])}|{fmt(m['hard']['arm6_contrib'])}|{fmt(m['all']['active7'])}|{fmt(m['hard']['active7'])}|{fmt(ds['draft']['all']['mse6_sum_div7'])}|{fmt(ds['ordinary_pi05']['all']['mse6_sum_div7'])}|{fmt(ds['draft']['all']['active7'])}|{fmt(ds['ordinary_pi05']['all']['active7'])}|{fmt(ad['draft']['all']['active7'])}|{fmt(ad['ordinary_pi05']['all']['active7'])}|")
        lines += ['', '軸ごとの寄与（x,y,z,rx,ry,rz,gripper）は `results.json` と各 `predictions/*.npz` に保存。`error_delta_vs` は正解に対する paired MSE差（denoised−reference）の episode bootstrap CI、`action_distance` は出力と下書き／通常π0.5の距離。']
    lines += ['', '## 読み取り','',
        't0=0は下書きそのもの、t0=1は通常π0.5と同じ入力・noiseの出力。中間t0が下書きを改善するか、また出力が下書きとπ0.5のどちらに近いかは、選択を固定したt0行で比較する。',
        '本結果は正解行動を見て候補を選ぶ上限であり、検索器や閉ループの性能を示さない。学習済み側のデモも通常π0.5の事前学習に含まれる可能性がある。',
        '出力：`' + str(result.get('output_dir', OUT)) + '`。実装：`denoise_oracle_drafts.py`。','']
    return '\n'.join(lines)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=OUT)
    ap.add_argument('--threads',type=int,default=int(os.environ.get('OMP_NUM_THREADS','8')))
    ap.add_argument('--batch',type=int,default=8)
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--postprocess-from',type=Path,default=None,
                    help='write a fresh run directory with the ordinary pi05 gripper fixed in every prediction')
    args=ap.parse_args()
    if args.postprocess_from is not None: postprocess_completed_run(args.postprocess_from,args.out)
    elif args.smoke: run_smoke(args.out,args.threads,args.batch)
    else: run(args.out,args.threads,args.batch)


if __name__=='__main__': main()
