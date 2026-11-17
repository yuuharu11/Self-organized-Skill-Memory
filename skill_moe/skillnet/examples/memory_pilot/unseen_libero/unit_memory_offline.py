#!/usr/bin/env python3
"""Offline scoring for the exact unit-memory selectors used by memory_closed_loop.

This is a smoke-first companion to the closed-loop report.  It imports the saved
pool/query checkpoints and the serving selector; it does not fit a selector or
choose settings.  ``--smoke`` is the default-safe mode; the full offline run is
explicitly requested with ``--full``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'pi05_memory')]
STATE = Path(os.environ.get('SKILLNET_STATE_DIR', '/work/.skillnet-state'))
ROOT = STATE / 'outputs/memory-pilot/unseen-libero'
OUT = ROOT / 'unit-memory-offline-01'
CLUSTER = ROOT / 'cluster-pools-01'
POOL_NAMES = ('unit_none', 'unit_04', 'unit_08', 'unit_16')
LONG_TASKS = (0, 10, 13, 31, 35, 43, 57, 61, 84, 88)


def row_loss(pred, target, axes: int, window: slice) -> np.ndarray:
    """Per-window error under the project's sum-of-squares / 70 convention."""
    p = np.asarray(pred, np.float64)[:, window, :axes]
    y = np.asarray(target, np.float64)[:, window, :axes]
    return np.square(p - y).sum(axis=(1, 2)) / (p.shape[1] * 7.0)


def dct3_pi05(actions):
    """Closed-loop normal baseline's motion-only DCT-3 ablation; preserve grip."""
    from scipy.fft import dct, idct
    a = np.asarray(actions, np.float32)
    out = a.copy()
    c = dct(a[..., :6], type=2, norm='ortho', axis=1)
    c[:, 3:, :] = 0
    out[..., :6] = idct(c, type=2, norm='ortho', axis=1).astype(np.float32)
    return out


def episode_bootstrap(values, task, episode, mask=None, *, reps=5000, seed=4101):
    """Mean and task-scoped episode bootstrap interval, optionally over paired diffs."""
    v = np.asarray(values, np.float64)
    t, e = np.asarray(task), np.asarray(episode)
    keep = np.ones(len(v), bool) if mask is None else np.asarray(mask, bool)
    v, t, e = v[keep], t[keep], e[keep]
    keys = np.asarray([f'{int(a)}:{int(b)}' for a, b in zip(t, e)])
    unique, inv = np.unique(keys, return_inverse=True)
    sums = np.bincount(inv, weights=v); counts = np.bincount(inv)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), (reps, len(unique)))
    boot = sums[draws].sum(1) / counts[draws].sum(1)
    return {'rows': int(len(v)), 'episodes': int(len(unique)), 'mean': float(v.mean()),
            'ci95': [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]}


def _learned_data(d, c3, hard=None):
    """The CMP3 learned set: validation rows from non-memory held-out demos."""
    from compressed_denoise import RUN
    rows = d['rows']; selected = c3.select_seen(rows)
    ids = np.asarray(selected['ids'], np.int64)
    action_all = np.asarray(rows['actions'])[:, :, :7]
    state_all = np.load(RUN / 'rows/state.npy', mmap_mode='r', allow_pickle=False)
    task_all = np.asarray(rows['task']); ep_all = np.asarray(rows['episode']); frame_all = np.asarray(rows['frame'])
    index = {(int(t), int(e), int(f)): i for i, (t, e, f) in enumerate(zip(task_all, ep_all, frame_all))}
    hist = np.zeros((len(ids), 10, 7), np.float32)
    for n, i in enumerate(ids):
        prev = index.get((int(task_all[i]), int(ep_all[i]), int(frame_all[i]) - 10))
        if prev is not None:
            hist[n] = action_all[prev]
    state = np.asarray(state_all[ids, :8], np.float32)
    pi_file = np.load(ROOT / 'pi05-reference-01/predictions/learned.npz', allow_pickle=False)
    loc = {int(r): j for j, r in enumerate(pi_file['row'])}
    if any(int(i) not in loc for i in ids):
        raise AssertionError('corrected learned π0.5 cache misses selected CMP3 rows')
    pi = np.asarray(pi_file['actions'][[loc[int(i)] for i in ids]], np.float32)
    target = np.asarray(selected['targets'], np.float32)
    if not np.allclose(target, pi_file['target'][[loc[int(i)] for i in ids]], atol=1e-7):
        raise AssertionError('learned π0.5 cache target does not match CMP3 selection')
    if np.asarray(rows['memory'])[ids].any() or not np.asarray(rows['validation'])[ids].all():
        raise AssertionError('learned evaluation rows must be held-out validation demos outside memory')
    if np.intersect1d(ids, np.flatnonzero(~np.asarray(rows['validation'], bool))).size:
        raise AssertionError('learned evaluation rows entered selector training rows')
    return {'name': 'learned', 'row': ids, 'target': target, 'pi05': pi,
            'hist': hist, 'state': state, 'task': np.asarray(selected['task']),
            'episode': np.asarray(selected['episode']), 'frame': np.asarray(selected['frame']),
            'hard': np.asarray(hard if hard is not None else np.zeros(len(ids), bool), bool),
            'audit': {'target_rows': len(ids), 'validation_only': True, 'memory_overlap': 0,
                      'selector_train_overlap': 0,
                      'history_source': 'ground-truth recorded demo actions at frame-10; zero when unavailable',
                      'pi05_cache': 'pi05-reference-01/predictions/learned.npz'}}


def _unseen_data(d, c3):
    """The CMP3 unseen Eval, with corrected official π0.5 predictions aligned by row ID."""
    wpath = ROOT / 'scale-only-02/production/prep/W_inputs_eval.npz'
    w = np.load(wpath, allow_pickle=False)
    k3 = np.load(ROOT / 'key-query-03/prep.npz', allow_pickle=False)
    if not (np.array_equal(k3['eval_task'], w['task']) and
            np.allclose(k3['eval_hist'], w['hist'], atol=0) and
            np.allclose(k3['eval_state'], w['state'], atol=0)):
        raise AssertionError('unseen K1 P5 query inputs differ from the closed-loop-aligned Eval cache')
    rows = np.asarray(w['row'], np.int64)
    if len(rows) != 619 or not np.array_equal(rows, np.arange(113, 732)):
        raise AssertionError('unseen W_inputs row order differs from the fixed 619-row Eval')
    unseen = ROOT / 'unseen-rows-02'
    task = np.load(unseen / 'task.npy', mmap_mode='r', allow_pickle=False)
    ep = np.load(unseen / 'episode.npy', mmap_mode='r', allow_pickle=False)
    frame = np.load(unseen / 'frame.npy', mmap_mode='r', allow_pickle=False)
    act = np.load(unseen / 'actions.npy', mmap_mode='r', allow_pickle=False)
    target = np.asarray(act[rows, :, :7], np.float32)
    if not np.allclose(target, np.asarray(d['eval'], np.float32), atol=1e-7):
        raise AssertionError('unseen target action rows differ from CMP3 Eval order')
    if not np.array_equal(np.asarray(task[rows]), np.asarray(w['task'])):
        raise AssertionError('unseen task metadata is not aligned')
    official = np.load(ROOT / 'fast-query-01/production/predictions/pi05_eval_rot180.npz', allow_pickle=False)
    loc = {int(r): j for j, r in enumerate(official['row'])}
    if any(int(r) not in loc for r in rows):
        raise AssertionError('corrected official π0.5 cache misses Eval rows')
    pi = np.asarray(official['actions'][[loc[int(r)] for r in rows]], np.float32)
    asis = np.load(ROOT / 'fast-query-01/production/predictions/pi05_eval_asis.npz', allow_pickle=False)
    aloc = {int(r): j for j, r in enumerate(asis['row'])}
    old_pi = np.asarray(asis['actions'][[aloc[int(r)] for r in rows]], np.float32)
    return {'name': 'unseen', 'row': rows, 'target': target, 'pi05': pi,
            'hist': np.asarray(w['hist'], np.float32), 'state': np.asarray(w['state'], np.float32),
            'task': np.asarray(w['task'], np.int64), 'episode': np.asarray(w['episode'], np.int64),
            'frame': np.asarray(w['frame'], np.int64), 'hard': np.asarray(d['hard'], bool),
            'audit': {'target_rows': len(rows), 'pi05_cache': 'fast-query-01/production/predictions/pi05_eval_rot180.npz',
                      'history_source': 'precomputed K1/scale-only query history from recorded demo actions',
                      'π0.5_cache_max_diff_vs_scale_only02': float(np.max(np.abs(pi - w['pi05']))),
                      'π0.5_eval_MSE_active7': float(np.square(pi - target).mean()),
                      'max_diff_vs_pi05_eval_asis': float(np.max(np.abs(pi - old_pi))),
                      'asis_eval_MSE_active7': float(np.square(old_pi - target).mean())}}


def _selectors():
    """Instantiate the exact serving selector, swapping only its persisted pool/checkpoint."""
    import memory_closed_loop as mcl
    selectors = {'window_frag_1595': mcl.MemorySelector()}
    old_pool, old_ckpt = os.environ.get('MEM_POOL'), os.environ.get('MEM_CKPT')
    try:
        for name in POOL_NAMES:
            os.environ['MEM_POOL'] = str(CLUSTER / f'pool_{name}.npz')
            os.environ['MEM_CKPT'] = str(CLUSTER / f'query_{name}.pt')
            selectors[name] = mcl.MemorySelector()
    finally:
        if old_pool is None: os.environ.pop('MEM_POOL', None)
        else: os.environ['MEM_POOL'] = old_pool
        if old_ckpt is None: os.environ.pop('MEM_CKPT', None)
        else: os.environ['MEM_CKPT'] = old_ckpt
    return selectors


def _long_exclusion_task(task):
    """Map canonical LIBERO-90 Long task IDs to the LIBERO-40 task used by closed-loop."""
    from memory_closed_loop import LONG_TO_40
    order = {task_id: i for i, task_id in enumerate(LONG_TASKS)}
    return LONG_TO_40[order[int(task)]]


def task_exclusion_mask(candidate_task, query_task, *, unseen):
    """Closed-loop task mask, accounting for LIBERO-90 vs LIBERO-40 ID spaces."""
    q = (np.asarray([_long_exclusion_task(t) for t in query_task], np.int64)
         if unseen else np.asarray(query_task, np.int64))
    return np.asarray(candidate_task)[None, :] != q[:, None]


def _select(selector, data):
    picks = np.empty(len(data['row']), np.int64)
    if data['name'] == 'unseen':
        for task in np.unique(data['task']):
            ix = np.flatnonzero(data['task'] == task)
            picks[ix] = selector.pick(data['hist'][ix], data['state'][ix], _long_exclusion_task(task))
    else:
        picks[:] = selector.pick(data['hist'], data['state'], None)
    return picks


def _pred_with_pi05_pose(selector, picks, pi05):
    # Exactly the serving path: Procrustes pose the chosen unit, then replace only arm axes;
    # π0.5 remains the gripper source.
    posed = selector.pose_to(np.asarray(picks, np.int64), np.asarray(pi05, np.float32))
    pred = np.asarray(pi05, np.float32).copy()
    pred[:, :, :6] = posed[:, :, :6]
    return pred


def _upper_oracle(target, candidates, candidate_task, query_task, *, exclude):
    import magnitude_bounds as mb
    from variable_shape_memory import to_normalized
    y = np.asarray(target, np.float32)
    qphys = mb.vs.to_physical(y[:, :, :6]).astype(np.float32)
    pool = np.asarray(candidates, np.float32)
    pool_task = np.asarray(candidate_task)
    qt = (np.asarray([_long_exclusion_task(t) for t in query_task], np.int64)
          if exclude else np.asarray(query_task))
    if exclude:
        picks, rot, scale, _ = mb.best_oracle_types(qphys, pool[:, :, :6], pool_task=pool_task,
                                                     query_task=qt, chunk=128)
    else:
        picks, rot, scale, _ = mb.best_oracle_types(qphys, pool[:, :, :6], chunk=128)
    shape = pool[picks].copy()
    motion = mb.apply_rotation(shape[:, :, :6], rot) * scale[:, None, None]
    pred = np.asarray(y, np.float32).copy()
    pred[:, :, :6] = to_normalized(motion).astype(np.float32)
    pred[:, :, 6] = y[:, :, 6]
    return picks, pred


def _oracle_type_pi_pose(data, candidates, candidate_task, exclude):
    """Oracle S-shared type choice, but infer its orientation/scale from π0.5."""
    import scale_only_query as sq
    import magnitude_bounds as mb
    pool = np.asarray(candidates, np.float32)
    qphys = mb.vs.to_physical(data['target'][:, :, :6]).astype(np.float32)
    err = sq.scale_only_error(qphys, pool[:, :, :6], 'S-shared')
    if exclude:
        mask = task_exclusion_mask(candidate_task, data['task'], unseen=True)
        err[~mask] = np.inf
    picks = np.argmin(err, axis=1)
    # Reuse the serving selector's Procrustes decoder; do not fit pose or scale to the target.
    import memory_closed_loop as mcl
    tmp = object.__new__(mcl.MemorySelector)
    tmp.shapes = pool
    import variable_shape_memory as vs
    tmp.vs = vs
    return picks, _pred_with_pi05_pose(tmp, picks, data['pi05'])


def _type_recovery(data, candidates, picks):
    """Historical greedy-pool gap recovery: (random-selected)/(random-oracle)."""
    import key_query_probe as kq
    import variable_shape_memory as vs
    q = kq._motion_shape_from_actions(data['target'], vs)
    err = kq.shape_error_matrix(q, candidates, vs, batch=64, device='cpu')
    chosen = err[np.arange(len(err)), np.asarray(picks)]
    oracle = err.min(1); random = err.mean(1)
    denom = random - oracle
    return {'selected_shape_error_mean': float(chosen.mean()), 'oracle_shape_error_mean': float(oracle.mean()),
            'random_shape_error_mean': float(random.mean()),
            'gap_recovery': float(np.mean(random - chosen) / max(float(np.mean(denom)), 1e-12)),
            'strict_top1': float(np.mean(chosen <= oracle + 1e-8))}


def _metric_summary(pred, data, *, reps=5000):
    result = {}
    pi = np.asarray(data['pi05'], np.float32)
    for name, sl in [('first5', slice(0, 5)), ('continuation5_9', slice(5, 10)), ('full10', slice(0, 10))]:
        result[name] = {}
        for metric, axes in [('active7', 7), ('arm6_over7', 6)]:
            loss = row_loss(pred, data['target'], axes, sl)
            pi_loss = row_loss(pi, data['target'], axes, sl)
            result[name][metric] = {'all': episode_bootstrap(loss, data['task'], data['episode'], reps=reps),
                                    'hard': episode_bootstrap(loss, data['task'], data['episode'], data['hard'], reps=reps),
                                    'delta_vs_pi05': {
                                        'all': episode_bootstrap(loss - pi_loss, data['task'], data['episode'], reps=reps),
                                        'hard': episode_bootstrap(loss - pi_loss, data['task'], data['episode'], data['hard'], reps=reps)}}
    return result


def reproduce_mag1_supplement():
    """Repeat supplement-corrected-metrics' MAG1 A6/T3 reconstruction verbatim."""
    import magnitude_bounds as mb
    import scale_only_query as sq
    p = ROOT / 'magnitude-01/eval_rows.npz'
    z = np.load(p, allow_pickle=False)
    mag_input = mb._load_rows('eval')
    pi_phys = mb.vs.to_physical(np.asarray(mag_input['pi05'])[:, :, :6])
    out = []
    for i in range(len(z['row'])):
        template_phys = mb.vs.to_physical(z['template'][i, :, :6])
        rot, amp = sq.procrustes_rs(template_phys, pi_phys[i])
        out.append(mb.rollout(z['template'][i], amp, 10, rotation=rot)[:, :6])
    pred = np.asarray(out, np.float32)
    target = np.asarray(z['target'][:, :, :6], np.float32)
    loss = np.square(pred - target).sum((1, 2)) / 70.0
    hard = np.load(ROOT / 'lib-closed-01/eval_rows.npz', allow_pickle=False)['hard'].astype(bool)
    return {'all_full10': float(loss.mean()), 'hard_full10': float(loss[hard].mean()),
            'rows': len(loss), 'hard_rows': int(hard.sum()),
            'pi05_source': 'scale-only-02/production/prep/W_inputs_eval.npz via magnitude_bounds._load_rows',
            'note': 'Recomputed exactly as supplement-corrected-metrics/recompute.py from MAG1 saved oracle-type templates.',
            'pred': pred}


def run(*, smoke=False, out=None):
    import compression_seen_cuts as c3
    import cluster_pools as cp
    import memory_closed_loop as mcl
    import variable_shape_memory as vs
    from scipy.stats import spearmanr

    out = Path(out or (OUT / ('smoke-07' if smoke else 'production')))
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'refusing to overwrite non-empty output: {out}')
    out.mkdir(parents=True, exist_ok=True)
    d, evals = c3.prepare()
    datasets = {'learned': _learned_data(d, c3, evals['learned']['hard']), 'unseen': _unseen_data(d, c3)}
    full_unseen_for_recovery = dict(datasets['unseen'])
    if smoke:
        for k, data in datasets.items():
            full_n = len(data['row'])
            hard_ix = np.flatnonzero(data['hard']); easy_ix = np.flatnonzero(~data['hard'])
            take = np.r_[easy_ix[:4], hard_ix[:4]]
            if len(take) < min(8, full_n): take = np.r_[take, np.setdiff1d(np.arange(full_n), take)[:8-len(take)]]
            take = take[:min(8, full_n)]
            n = len(take)
            for key, value in list(data.items()):
                if isinstance(value, np.ndarray) and len(value) == full_n: data[key] = value[take]
            data['audit']['smoke_rows'] = n
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
    selectors = _selectors()
    task_counts = {'normal_pi05': [32, 46], 'pi05_dct3': [34, 47], 'window_frag_1595': [39, 20],
                   'unit_none': [27, 19], 'unit_04': [43, 11], 'unit_08': [43, 17], 'unit_16': [33, 14]}
    results = {'protocol': {'eval_sets': c3.unit_meta(d, evals), 'pi05_unseen':
                 'fast-query-01/production/predictions/pi05_eval_rot180.npz (corrected image orientation)',
                 'pi05_learned': 'pi05-reference-01/predictions/learned.npz (row-aligned held-out demos)',
                 'selector': 'memory_closed_loop.MemorySelector + persisted cluster_pools query checkpoints',
                 'pose_grip': 'MemorySelector.pose_to (π0.5 Procrustes); overwrite arm only, π0.5 grip retained',
                 'offline_history': ('Offline query history uses recorded ground-truth demonstration actions (preceding frame-10 window; '
                                     'unseen history is the K1/scale-only precomputed cache). Closed-loop history instead uses the '
                                     'previously executed policy actions; therefore history inputs differ.'),
                 'hard': 'CMP3 fixed hard masks', 'smoke': bool(smoke)},
                'datasets': {}, 'closed_loop_successes': task_counts, 'legacy_recovery': {}, 'mag1_check': {}}
    stored_predictions = {}
    for split, data in datasets.items():
        split_res = {'audit': data['audit'], 'conditions': {}}
        predictions = {'normal_pi05': data['pi05'], 'pi05_dct3': dct3_pi05(data['pi05']),
                       'previous_hold': np.repeat(data['hist'][:, -1:, :7], 10, axis=1)}
        candidates_by_name = {'window_frag_1595': selectors['window_frag_1595'].shapes}
        for name in POOL_NAMES: candidates_by_name[name] = selectors[name].shapes
        for name, selector in selectors.items():
            picks = _select(selector, data)
            pred = _pred_with_pi05_pose(selector, picks, data['pi05'])
            predictions[name] = pred
            # Type-only oracle selects the true S-shared motion type but retains π0.5 pose fitting.
            mapped_exclusion = split == 'unseen'
            type_picks, type_pred = _oracle_type_pi_pose(data, selector.shapes, selector.src_task, mapped_exclusion)
            upper_picks, upper_pred = _upper_oracle(data['target'], selector.shapes, selector.src_task,
                                                      data['task'], exclude=mapped_exclusion)
            predictions[name + '|oracle_type_pi05_pose'] = type_pred
            predictions[name + '|oracle_unit_pose_scale_grip'] = upper_pred
            split_res['conditions'][name] = {
                'candidate_count': len(selector.shapes),
                'inference': _metric_summary(pred, data, reps=1000 if smoke else 5000),
                'oracle_type_pi05_pose': _metric_summary(type_pred, data, reps=1000 if smoke else 5000),
                'oracle_unit_pose_scale_grip': _metric_summary(upper_pred, data, reps=1000 if smoke else 5000),
                'type_recovery': _type_recovery(data, selector.shapes, picks),
                'selected_pool_indices': picks.tolist() if smoke else None,
                'oracle_type_indices': type_picks.tolist() if smoke else None,
                'oracle_upper_indices': upper_picks.tolist() if smoke else None}
            if split == 'unseen':
                results['legacy_recovery'][name] = _type_recovery(data, selector.shapes,
                                                                   selector.pick(data['hist'], data['state'], None))
        for name in ('normal_pi05', 'pi05_dct3', 'previous_hold'):
            pred = predictions[name]
            split_res['conditions'].setdefault(name, {})['inference'] = _metric_summary(pred, data, reps=1000 if smoke else 5000)
        results['datasets'][split] = split_res
        stored_predictions[split] = {'row': data['row'], 'target': data['target'], 'pi05': data['pi05'],
                                     'hard': data['hard'], 'task': data['task'], 'episode': data['episode'],
                                     **{f'pred__{k}': v for k, v in predictions.items()}}
    mag = reproduce_mag1_supplement()
    results['mag1_check'] = {k: v for k, v in mag.items() if k != 'pred'}
    # This is a cross-report consistency check, not a selector or setting choice.
    results['mag1_check']['expected_supplement'] = {'all_full10': .1348, 'hard_full10': .2921}
    results['mag1_check']['difference_from_magnitude01'] = {
        'magnitude01_first5_all': .1459741506933179, 'magnitude01_continuation_all': .16863476342983508,
        'magnitude01_active6_mean_scale_first5': .1459741506933179,
        'converted_to_sum6_over70': [.1459741506933179 * 6 / 7, .16863476342983508 * 6 / 7],
        'note': 'The metric discrepancy is the denominator: magnitude-01 axis6 uses mean over 5x6 (.14597/.16863); the supplement reports six-axis squared sum/70, i.e. multiply by 6/7 (.12512/.14454). The hard masks also differ: magnitude-01 uses its Dev π0.5 q80 mask (179 rows), supplement reuses the common 134-row mask.'}
    # Optional stored selector IDs do not include the query inputs, so exact rollout row matching is not auditable.
    results['selector_log_audit'] = {'closed_loop_results_have_selected_unit_ids': True,
        'same_observation_history_state_recorded': False,
        'note': 'Rollout result.json stores selected unit IDs but not the corresponding query history/state; matching those IDs to offline rows is not testable.'}
    # Type-recovery is a scoring-only check. Compute it on all 619 Eval windows even in smoke
    # so the historical 55% / 76% definition is checked on its original denominator.
    recovery_expected = {x['pool']: x for x in json.loads((CLUSTER / 'offline.json').read_text())}
    full_recovery = {}
    for name in POOL_NAMES:
        selector = selectors[name]
        if smoke:
            pick_full = _select(selector, full_unseen_for_recovery)
            rec = _type_recovery(full_unseen_for_recovery, selector.shapes, pick_full)
        else:
            rec = results['legacy_recovery'][name]
        old = recovery_expected[name]
        full_recovery[name] = {**rec, 'reference_gap': float(old['gap']),
                               'gap_abs_difference': float(abs(rec['gap_recovery'] - old['gap'])),
                               'reference_eval_shape_selected': float(old['eval_shape_selected'])}
    results['legacy_recovery_full_eval_check'] = full_recovery
    rank_rows = []
    for name, counts in task_counts.items():
        if name in ('normal_pi05', 'pi05_dct3') or name in ('window_frag_1595', *POOL_NAMES):
            rank_rows.append((name, float(counts[0]), float(counts[1])))
    results['offline_success_rank_correlation'] = {}
    for split, sdata in results['datasets'].items():
        values = {}
        for scope in ('all', 'hard'):
            for win in ('first5', 'full10'):
                for key in ('active7', 'arm6_over7'):
                    x = []; y90 = []; y10 = []
                    for name, c90, c10 in rank_rows:
                        if name not in sdata['conditions'] or 'inference' not in sdata['conditions'][name]: continue
                        x.append(sdata['conditions'][name]['inference'][win][key][scope]['mean'])
                        y90.append(c90); y10.append(c10)
                    values[f'{scope}/{win}/{key}'] = {'n': len(x),
                        'libero90_spearman': float(spearmanr(x, y90).statistic) if len(x) >= 3 else None,
                        'libero10_spearman': float(spearmanr(x, y10).statistic) if len(x) >= 3 else None}
        results['offline_success_rank_correlation'][split] = values
    (out / 'results.json').write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    np.savez_compressed(out / 'predictions.npz', **{f'{s}__{k}': v for s, arrays in stored_predictions.items() for k, v in arrays.items()})
    _write_report(out / 'report.md', results)
    return results


def _write_report(path, result):
    lines = ['# 単位メモリ閉ループ条件のオフライン MSE', '',
             '## 主表', '',
             '各窓の MSE は当該時間区間の二乗誤差合計/(区間長×7)。従って full10 は active7 平均（÷70）、腕6軸は6軸二乗和÷70。first5/continuation は各々÷35。全体・難を分け、π0.5との差は episode bootstrap 95% CI。', '',
             '| 集合 | プール | 選択／姿勢 | active7 all (first5/full10) | active7 hard | arm6/7 all (first5/full10) | arm6/7 hard | 閉ループ (90/10) |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for split, data in result['datasets'].items():
        for name, c in data['conditions'].items():
            for mode in ('inference', 'oracle_type_pi05_pose', 'oracle_unit_pose_scale_grip'):
                if mode not in c: continue
                m = c[mode]
                a = m['first5']['active7']['all']['mean']; af = m['full10']['active7']['all']['mean']
                ah = m['first5']['active7']['hard']['mean']; afh = m['full10']['active7']['hard']['mean']
                b = m['first5']['arm6_over7']['all']['mean']; bf = m['full10']['arm6_over7']['all']['mean']
                bh = m['first5']['arm6_over7']['hard']['mean']; bhf = m['full10']['arm6_over7']['hard']['mean']
                count = result['closed_loop_successes'].get(name, [None, None])
                count_txt = f'{count[0]}/100 · {count[1]}/50' if count[0] is not None else '—'
                count_in_pool = c.get('candidate_count')
                pool_label = f'{name} (K={count_in_pool})' if count_in_pool is not None else name
                lines.append(f'| {split} | {pool_label} | {mode} | {a:.4f} / {af:.4f} | {ah:.4f} / {afh:.4f} | {b:.4f} / {bf:.4f} | {bh:.4f} / {bhf:.4f} | {count_txt} |')
    lines += ['', '## 閉ループ成功数と回収率', '',
              '| 条件 | LIBERO-90 /100 | LIBERO-10 /50 |', '|---|---:|---:|']
    for name, (a, b) in result['closed_loop_successes'].items(): lines.append(f'| {name} | {a} | {b} |')
    lines += ['', '| Eval pool | legacy gap recovery | strict top1 | selected shape error | oracle shape error | random shape error |', '|---|---:|---:|---:|---:|---:|']
    for name, x in result['legacy_recovery'].items(): lines.append(f"| {name} | {x['gap_recovery']:.3f} | {x['strict_top1']:.3f} | {x['selected_shape_error_mean']:.5f} | {x['oracle_shape_error_mean']:.5f} | {x['random_shape_error_mean']:.5f} |")
    if 'legacy_recovery_full_eval_check' in result:
        lines += ['', '全619行の旧 gap recovery 再現チェック（選択クエリは同じ P5_hist weights、ここだけ旧定義に合わせ same-task mask なし）:', '',
                  '| Pool | measured | reference | absolute diff |', '|---|---:|---:|---:|']
        for name, x in result['legacy_recovery_full_eval_check'].items():
            lines.append(f"| {name} | {x['gap_recovery']:.4f} | {x['reference_gap']:.4f} | {x['gap_abs_difference']:.6f} |")
    lines += ['', '回収率は従来 `cluster_pools.offline.json` と同じ `(random-selected)/(random-oracle)` の型誤差ギャップ。', '',
              '### π0.5 との差（full10 episode bootstrap 95% CI）', '',
              '| 集合 | プール | 選択 | active7 Δ all / hard | arm6/7 Δ all / hard |', '|---|---|---|---:|---:|']
    for split, data in result['datasets'].items():
        for name, c in data['conditions'].items():
            for mode in ('inference', 'oracle_type_pi05_pose', 'oracle_unit_pose_scale_grip'):
                if mode not in c: continue
                m = c[mode]['full10']
                a = m['active7']['delta_vs_pi05']['all']; ah = m['active7']['delta_vs_pi05']['hard']
                b = m['arm6_over7']['delta_vs_pi05']['all']; bh = m['arm6_over7']['delta_vs_pi05']['hard']
                pool = f'{name} (K={c.get("candidate_count")})' if c.get('candidate_count') is not None else name
                lines.append(f'| {split} | {pool} | {mode} | {a["mean"]:+.4f} [{a["ci95"][0]:+.4f},{a["ci95"][1]:+.4f}] / {ah["mean"]:+.4f} [{ah["ci95"][0]:+.4f},{ah["ci95"][1]:+.4f}] | {b["mean"]:+.4f} [{b["ci95"][0]:+.4f},{b["ci95"][1]:+.4f}] / {bh["mean"]:+.4f} [{bh["ci95"][0]:+.4f},{bh["ci95"][1]:+.4f}] |')
    lines += ['',
              '### オフライン誤差と閉ループ成功数の順位相関（Spearman）', '',
              '| 集合 | 評価範囲 | active7 full10 | arm6/70 full10 |', '|---|---|---:|---:|']
    for split, vals in result['offline_success_rank_correlation'].items():
        for scope in ('all', 'hard'):
            a = vals[f'{scope}/full10/active7']['libero90_spearman']
            b = vals[f'{scope}/full10/arm6_over7']['libero90_spearman']
            av = f'{a:.3f}' if a is not None else 'NA'; bv = f'{b:.3f}' if b is not None else 'NA'
            lines.append(f'| {split} | {scope} vs LIBERO-90 success | {av} | {bv} |')
    lines += ['', '## MAG1 A6/T3 再現と π0.5 版', '',
              f"補足 `supplement-corrected-metrics-01/recompute.py` と同じ再構成: all={result['mag1_check']['all_full10']:.4f}, hard={result['mag1_check']['hard_full10']:.4f}; 指定値 .1348/.2921。",
              f"π0.5 は未学習 Eval に `fast-query-01/production/predictions/pi05_eval_rot180.npz`、学習済みに `pi05-reference-01/predictions/learned.npz` を使用。{result['mag1_check']['note']}",
              f"別報告との差は主に分母。magnitude-01 の axis6 指標は平均二乗誤差（/60）で first5=.14597、continuation=.16863。補足の sum6/70 に換算すると .12512/.14454、full10=.13483。hard集合も magnitude-01 の179行から補足の共通134行に変わる。今回の予測は rot180 で、scale-only-02 のπ0.5とmax差 {result['datasets']['unseen']['audit']['π0.5_cache_max_diff_vs_scale_only02']:.6g}、旧 asis と max差 {result['datasets']['unseen']['audit']['max_diff_vs_pi05_eval_asis']:.6g}。", '',
              '## 再現性と制約', '',
              '推論選択は serving `memory_closed_loop.MemorySelector` と保存済み pool-specific checkpoint を使用し、Procrustes はその `pose_to` を呼ぶ。rolloutログは selected unit ID を持つが対応する history/state を保存しないため、同一シミュレータ問い合わせをオフライン行へ厳密に対応づける照合はできない。',
              'オフラインの履歴はデモに記録された正解行動（直前の frame-10 窓。未存在ならゼロ）から作る。一方、閉ループの履歴は直前に実際に実行した方策の行動であるため、履歴入力は一致しない。',
              'Selector学習とメモリ構築は既存のLIBERO-40成果物を読み込み、新たな学習・設定選択はしていない。スモーク出力は8行/集合。', '']
    Path(path).write_text('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='8 rows per evaluation set (default)')
    ap.add_argument('--full', action='store_true', help='all 619 + 621 rows; writes production/')
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    if a.smoke and a.full: ap.error('choose only one of --smoke or --full')
    run(smoke=not a.full, out=a.out)


if __name__ == '__main__':
    main()
