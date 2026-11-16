#!/usr/bin/env python3
"""MAG3: offline orientation x amplitude bounds, with a gated paired rollout.

The 95-unit pool stores canonical (rotation/scale-free) motion; its source pose
is reconstructed from the frozen LIBERO-40 medoids. The 1,595-window W pool
retains its source direction. Offline unit identity is oracle-selected within
each pool's type metric; orientation and amplitude are then varied separately.

Run from /work after ``source env/activate.sh``. Outputs are write-once under
``outputs/memory-pilot/unseen-libero/magnitude-03``. The offline stage is CPU
only; if the Dev gate passes, ``run`` validates the server/offline draft on ten
Dev windows before launching up to three GPU servers at .15 memory each.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import compressed_denoise as cd
import magnitude_bounds as mag1
import memory_closed_loop as legacy
import scale_only_query as sq
import variable_shape_memory as vs

ROOT = Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero')
OUT = ROOT / 'magnitude-03'
POOL95 = ROOT / 'cluster-pools-01/pool_unit_08.npz'
WPOOL = ROOT / 'scale-only-02/production/prep/W_pool.npz'
WPREP = ROOT / 'scale-only-02/production/prep'
WMODELS = ROOT / 'scale-only-02/production/models'
WSELECTION = ROOT / 'scale-only-02/production/selection.json'
CHECKPOINT = Path('/work/.skillnet-state/checkpoints/openpi-assets/checkpoints/pi05_libero')
HOLDOUT = (2, 7, 12, 17, 22, 27, 32, 37)
ORIENTATIONS = ('O1_source', 'O2_pi05', 'O3_history', 'O4_oracle')
AMPLITUDES = ('A1_oracle', 'A2_history', 'A3_pi05', 'A4_source')
SPLITS = ('dev', 'eval')
EPS = 1e-10


def _json(path: Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def _scale(x):
    return float(np.sqrt(np.mean(np.sum(np.asarray(x, np.float64) ** 2, axis=-1))))


def _unitize(x):
    a = np.asarray(x, np.float64)
    return a / max(_scale(a), EPS)


def _pool95():
    """Load canonical shapes and reconstruct their saved source rotation/scale."""
    z = np.load(POOL95, allow_pickle=False)
    pool = np.asarray(z['shape'], np.float64)
    task = np.asarray(z['source_task'], np.int64)
    rows = {k: np.load(cd.RUN / 'rows' / f'{k}.npy', mmap_mode='r', allow_pickle=False)
            for k in ('task', 'episode', 'frame')}
    mem_ids = np.asarray(np.load(cd.RUN / 'memory_row_ids.npy', allow_pickle=False), np.int64)
    bank = cd._bank_from_memory(rows, mem_ids, mode='variable', form='rms', tau=.08)
    if bank['unit_count'] != 133:
        raise AssertionError(f'expected 133 pre-filter medoids, got {bank["unit_count"]}')
    actions = np.load(cd.RUN / 'rows' / 'actions.npy', mmap_mode='r', allow_pickle=False)
    traces, _ = cd._stitch(actions, rows['task'], rows['episode'], rows['frame'], mem_ids)
    decoded = cd._bank_decode(bank)
    source_rot, source_scale = [], []
    for t, e, start, length, shape in zip(bank['task'], bank['episode'], bank['start'], bank['memory'].lengths, decoded):
        if int(t) in HOLDOUT:
            continue
        i = len(source_rot)
        if i >= len(pool) or not np.allclose(mag1.normalize_shape(shape, 10), pool[i], atol=2e-5):
            raise AssertionError(f'pool_unit_08 source-pose order mismatch at index {i}')
        trace = traces[(int(t), int(e))][int(start):int(start + length), :6]
        r, s = vs.pose_of(trace, 'rms')
        source_rot.append(np.asarray(r, np.float64))
        source_scale.append(float(s))
    if len(source_rot) != 95:
        raise AssertionError(f'expected 95 source poses, got {len(source_rot)}')
    return {'name': 'pool_unit_08', 'pool': pool[:, :, :6], 'task': task,
            'source_rot': np.asarray(source_rot), 'source_scale': np.asarray(source_scale),
            'base': pool[:, :, :6], 'rotation_is_stored': True}


def _pool_w():
    """Load scale-only W, whose original source directions are retained."""
    z = np.load(WPOOL, allow_pickle=False)
    phys = np.asarray(z['phys'], np.float64)
    return {'name': 'scale_only_W', 'pool': phys, 'task': np.asarray(z['task'], np.int64),
            'source_rot': np.broadcast_to(np.eye(3), (len(phys), 3, 3)).copy(),
            'source_scale': np.asarray([_scale(x) for x in phys]),
            'base': np.asarray([_unitize(x) for x in phys]), 'rotation_is_stored': False}


def _minimal_row_rotation(source, target):
    """Smallest row-vector rotation mapping source direction onto target."""
    from scipy.spatial.transform import Rotation
    a = np.asarray(source, np.float64); b = np.asarray(target, np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < EPS or nb < EPS:
        return np.eye(3), False
    a, b = a / na, b / nb
    c = float(np.clip(np.dot(a, b), -1., 1.))
    axis = np.cross(a, b); s = float(np.linalg.norm(axis))
    if s < EPS:
        if c > 0:
            return np.eye(3), True
        # Deterministic axis for the antipodal case.
        ref = np.eye(3)[int(np.argmin(np.abs(a)))]
        axis = np.cross(a, ref); axis /= np.linalg.norm(axis)
        rot_col = Rotation.from_rotvec(axis * np.pi).as_matrix()
    else:
        rot_col = Rotation.from_rotvec(axis / s * np.arctan2(s, c)).as_matrix()
    # Actions are row vectors (x @ R), while scipy's matrix acts on columns.
    return rot_col.T, True


def _history_direction(shape, history_physical):
    """Rotate the unit's leading 3-step translation toward prior 10-step motion."""
    src = np.asarray(shape, np.float64)[:min(3, len(shape)), :3].sum(0)
    hist = np.asarray(history_physical, np.float64)[-10:, :3].sum(0)
    return _minimal_row_rotation(src, hist)


def _rotation_for(pack, idx, orientation, pi_physical, hist_physical, oracle_rotation=None,
                  history_available=True):
    if orientation == 'O1_source':
        return np.asarray(pack['source_rot'][idx], np.float64), False
    if orientation == 'O2_pi05':
        return sq.procrustes_rs(pack['base'][idx], pi_physical)[0], False
    if orientation == 'O3_history':
        if not history_available:
            return np.asarray(pack['source_rot'][idx], np.float64), True
        r, ok = _history_direction(pack['base'][idx], hist_physical)
        if not ok:
            return np.asarray(pack['source_rot'][idx], np.float64), True
        return r, False
    if orientation == 'O4_oracle':
        if oracle_rotation is None:
            raise ValueError('O4_oracle requires a target-fitted rotation')
        return np.asarray(oracle_rotation, np.float64), False
    raise ValueError(orientation)


def _oracle_choices(pack, target_physical, query_task):
    """Choose each pool's unit under its own target type metric, excluding own task."""
    if pack['name'] == 'pool_unit_08':
        pick, oracle_rot, _, type_err = mag1.best_oracle_types(target_physical, pack['pool'],
            pool_task=pack['task'], query_task=query_task)
        return pick, oracle_rot, type_err
    errors = sq.scale_only_error(target_physical, pack['pool'], 'S-shared')
    allowed = np.asarray(pack['task'])[None, :] != np.asarray(query_task)[:, None]
    errors[~allowed] = np.inf
    pick = errors.argmin(1)
    oracle_rot = np.empty((len(pick), 3, 3), np.float64)
    for i, j in enumerate(pick):
        oracle_rot[i] = sq.procrustes_rs(pack['pool'][j], target_physical[i])[0]
    return pick, oracle_rot, errors[np.arange(len(pick)), pick]


def render_draft(pack, idx, pi05_norm, history_norm, orientation, amplitude,
                 *, oracle_rotation=None, history_available=True, no_history_scale='source'):
    """Render one 10-action draft; shared by offline evaluation and server hook."""
    idx = int(idx)
    ref = np.asarray(pi05_norm, np.float64)
    hist_n = np.asarray(history_norm, np.float64)
    ref_phys = vs.to_physical(ref[:, :6])
    hist_phys = vs.to_physical(hist_n[:, :6]) if len(hist_n) else np.zeros((0, 6), np.float64)
    r, fallback = _rotation_for(pack, idx, orientation, ref_phys, hist_phys,
        oracle_rotation=oracle_rotation, history_available=history_available)
    oriented = mag1.apply_rotation(pack['base'][idx], r)
    if amplitude == 'A1_oracle':
        # Upper-bound scale after fixing the selected orientation.
        scale = mag1.scalar_fit(oriented, ref_target_physical) if False else None
        raise ValueError('A1_oracle is offline-only and must be supplied through render_oracle_draft')
    if amplitude == 'A2_history':
        if not history_available:
            scale = float(pack['source_scale'][idx]) if no_history_scale == 'source' else 0.0
        else:
            scale = _scale(hist_phys) / max(_scale(oriented), EPS)
    elif amplitude == 'A3_pi05':
        scale = float(mag1.scalar_fit(oriented, ref_phys))
    elif amplitude == 'A4_source':
        scale = float(pack['source_scale'][idx])
    else:
        raise ValueError(amplitude)
    physical = oriented * float(scale)
    out = np.asarray(ref, np.float32).copy()
    out[:, :6] = vs.to_normalized(physical).astype(np.float32)
    return out, {'rotation': r, 'scale': float(scale), 'history_fallback': bool(fallback)}


def render_oracle_draft(pack, idx, pi05_norm, history_norm, target_physical, orientation,
                        oracle_rotation, *, history_available=True):
    ref = np.asarray(pi05_norm, np.float64)
    ref_phys = vs.to_physical(ref[:, :6])
    hist_n = np.asarray(history_norm, np.float64)
    hist_phys = vs.to_physical(hist_n[:, :6]) if len(hist_n) else np.zeros((0, 6), np.float64)
    r, fallback = _rotation_for(pack, int(idx), orientation, ref_phys, hist_phys,
        oracle_rotation=oracle_rotation, history_available=history_available)
    oriented = mag1.apply_rotation(pack['base'][int(idx)], r)
    scale = float(mag1.scalar_fit(oriented, target_physical))
    out = ref.astype(np.float32).copy()
    out[:, :6] = vs.to_normalized(oriented * scale).astype(np.float32)
    return out, {'rotation': r, 'scale': scale, 'history_fallback': bool(fallback)}


def _selector95():
    os.environ['MEM_POOL'] = str(POOL95)
    os.environ['MEM_CKPT'] = str(ROOT / 'cluster-pools-01/query_unit_08.pt')
    return legacy.MemorySelector()


def _selector_w():
    import torch
    sel = json.loads(WSELECTION.read_text())['selected']['W|S-shared|P5|4101']['holdout']
    frac = float(sel['frac'])
    ckpt = WMODELS / f'W_S-shared_P5_s4101_f{frac:.4f}.pt'
    rec = torch.load(ckpt, map_location='cpu', weights_only=False)
    pool = dict(np.load(WPREP / 'W_pool.npz', allow_pickle=False))
    ddev = dict(np.load(WPREP / 'W_inputs_dev.npz', allow_pickle=False))
    xdim = len(sq.query_inputs(ddev, 'P5')[0])
    kinput = sq.key_inputs('W', pool, 'S-shared')
    net = sq.make_net('W', 'P5', xdim, kinput.shape[1])
    net.load_state_dict(rec['holdout']['state']); net.eval()
    with torch.no_grad():
        keys = net.key(torch.as_tensor(kinput, dtype=torch.float32))
        keys = torch.nn.functional.normalize(keys, dim=-1)
    class Selector:
        def __init__(self):
            self.torch = torch; self.pool = pool; self.net = net; self.keys = keys; self.vs = vs
            # legacy.infer uses physical candidate shapes for its retrieval diagnostic.
            self.shapes = np.concatenate([np.asarray(pool['phys'], np.float32),
                np.zeros((len(pool['phys']), 10, 1), np.float32)], axis=2)
            self.src_task = np.asarray(pool['task'], np.int64)
        def pick(self, history, state, exclude_task=None):
            inp = {'hist': np.asarray(history, np.float32), 'state': np.asarray(state, np.float32)}
            x = torch.as_tensor(sq.query_inputs(inp, 'P5'), dtype=torch.float32)
            with torch.no_grad():
                scores = torch.nn.functional.normalize(net.query(x), dim=-1) @ keys.T * net.log_scale.exp()
                if exclude_task is not None:
                    scores[:, torch.as_tensor(self.src_task == int(exclude_task))] = -1e9
                self.last_scores = scores.numpy()
                return scores.argmax(-1).numpy()
        def confidence(self, scores, temperature=.05):
            v = np.asarray(scores, np.float64); v = v[v > -1e8]
            if len(v) < 2: return {'p_top1':1.,'margin':0.,'eff_candidates':1.,'cos_top1':float(v.max()) if len(v) else 0.}
            z = (v-v.max())/float(temperature); p = np.exp(z); p /= p.sum()
            top = np.sort(v)[::-1]
            return {'p_top1':float(p.max()),'margin':float(top[0]-top[1]),
                'eff_candidates':float(np.exp(-(p*np.log(p+1e-30)).sum())), 'cos_top1':float(top[0])}
    return Selector()


def _selector(pool_name):
    return _selector95() if pool_name == 'pool_unit_08' else _selector_w()


def make_pose_hook(pack, orientation, amplitude, selector, *, context):
    """Install the same selector/pose operation used by the live policy adapter."""
    original_pick = selector.pick
    def pick(history, state, exclude_task=None):
        context['history'] = np.asarray(history, np.float32).copy()
        context['state'] = np.asarray(state, np.float32).copy()
        context['history_available'] = bool(context.get('history_available', np.any(np.abs(context['history']) > 1e-8)))
        return original_pick(history, state, exclude_task)
    def pose_to(idx, ref_norm):
        ids = np.asarray(idx).reshape(-1)
        history = np.asarray(context['history'], np.float32)
        outputs = []
        for i, j in enumerate(ids):
            h = np.asarray(context.get('history_override', history[i]), np.float32)
            if h.ndim == 3: h = h[i]
            out, dbg = render_draft(pack, int(j), np.asarray(ref_norm)[i], h,
                orientation, amplitude, history_available=bool(context.get('history_available', True)))
            context['last_render'] = {'unit':int(j), 'scale':dbg['scale'], 'fallback':dbg['history_fallback']}
            outputs.append(out)
        return np.asarray(outputs, np.float32)
    selector.pick = pick
    selector.pose_to = pose_to
    return selector


def _inputs_and_targets():
    inputs = {s: mag1._load_rows(s) for s in SPLITS}
    actions = np.load(mag1.ROWS / 'actions.npy', mmap_mode='r', allow_pickle=False)
    targets = {s: np.asarray(actions[np.asarray(inputs[s]['row'], np.int64), :, :7], np.float32) for s in SPLITS}
    traces = mag1._traces()
    segments = {s: mag1._segments_by_row(inputs[s], traces) for s in SPLITS}
    return inputs, targets, segments


def _target_windows(target, segments):
    truth = np.zeros((len(target), 10, 7), np.float32)
    for i, (trace, _) in enumerate(segments):
        n = min(10, len(trace))
        truth[i, :n] = trace[:n]
        if n < 10:
            truth[i, n:] = truth[i, max(n - 1, 0)]
    return truth


def _metrics(pred, truth, task, episode):
    ep = mag1.episode_cluster(task, episode)
    out = {}
    for win, sl in (('first5', slice(0, 5)), ('continuation5_9', slice(5, 10))):
        err = ((np.asarray(pred)[:, sl, :6] - truth[:, sl, :6]) ** 2).mean((1, 2))
        out[win] = mag1.paired_bootstrap(err, ep)
        out[f'{win}_row'] = err
    return out


def _score_pool(pack, split, inputs, target, segments):
    yphys = vs.to_physical(target[:, :, :6]).astype(np.float64)
    pi = np.asarray(inputs['pi05'], np.float32)
    pi_phys = vs.to_physical(pi[:, :, :6])
    hist_norm = np.asarray(inputs['hist'], np.float32)[:, :, :7]
    hist_phys = vs.to_physical(hist_norm[:, :, :6])
    pick, oracle_r, type_err = _oracle_choices(pack, yphys, inputs['task'])
    truth = _target_windows(target, segments)
    all_rows = []
    pred_by_key = {}
    for o in ORIENTATIONS:
        for a in AMPLITUDES:
            pred = np.zeros((len(pick), 10, 7), np.float32)
            scales = np.zeros(len(pick), np.float64)
            fallbacks = np.zeros(len(pick), bool)
            for i, j in enumerate(pick):
                if a == 'A1_oracle':
                    p, dbg = render_oracle_draft(pack, j, pi[i], hist_norm[i], yphys[i], o,
                        oracle_r[i], history_available=bool(np.any(np.abs(hist_norm[i]) > 1e-8)))
                else:
                    p, dbg = render_draft(pack, j, pi[i], hist_norm[i], o, a,
                        oracle_rotation=oracle_r[i], history_available=bool(np.any(np.abs(hist_norm[i]) > 1e-8)))
                pred[i] = p; scales[i] = dbg['scale']; fallbacks[i] = dbg['history_fallback']
            metrics = _metrics(pred, truth, inputs['task'], inputs['episode'])
            for win in ('first5', 'continuation5_9'):
                b = metrics[win]
                all_rows.append({'pool': pack['name'], 'split': split, 'orientation': o,
                    'amplitude': a, 'window': win, 'rows': b['rows'], 'episodes': b['episodes'],
                    'mean': b['mean'], 'ci95': b['ci95']})
            pred_by_key[(o, a)] = pred
            if o == 'O1_source' and a == 'A2_history':
                np.savez_compressed(OUT / 'offline' / f'{pack["name"]}_{split}_selected.npz',
                    rows=inputs['row'], task=inputs['task'], episode=inputs['episode'], pick=pick,
                    type_error=type_err, oracle_rotation=oracle_r, scale=scales,
                    history_fallback=fallbacks, prediction=pred, truth=truth)
    # Baselines are row-aligned and use exactly the same segment windows.
    hold = np.repeat(hist_norm[:, -1:, :7], 10, axis=1)
    for name, pred in (('pi05', pi), ('previous_hold', hold)):
        for win, sl in (('first5', slice(0, 5)), ('continuation5_9', slice(5, 10))):
            err = ((pred[:, sl, :6] - truth[:, sl, :6]) ** 2).mean((1, 2))
            b = mag1.paired_bootstrap(err, mag1.episode_cluster(inputs['task'], inputs['episode']))
            all_rows.append({'pool': pack['name'], 'split': split, 'orientation': name,
                'amplitude': 'baseline', 'window': win, 'rows': b['rows'], 'episodes': b['episodes'],
                'mean': b['mean'], 'ci95': b['ci95']})
    return all_rows, {'pick': pick, 'oracle_rotation': oracle_r, 'type_error': type_err,
                      'truth': truth, 'pi': pi, 'hist': hist_norm, 'pred_by_key': pred_by_key}


def _summary_and_gate(rows):
    keyed = {(r['pool'], r['split'], r['orientation'], r['amplitude'], r['window']): r['mean'] for r in rows}
    pools = ('pool_unit_08', 'scale_only_W')
    options = []
    for pool in pools:
        for o in ('O1_source', 'O2_pi05', 'O3_history', 'O4_oracle'):
            for a in ('A2_history', 'A3_pi05'):
                dev5 = keyed[(pool, 'dev', o, a, 'first5')]
                dev59 = keyed[(pool, 'dev', o, a, 'continuation5_9')]
                ev5 = keyed[(pool, 'eval', o, a, 'first5')]
                ev59 = keyed[(pool, 'eval', o, a, 'continuation5_9')]
                options.append({'pool': pool, 'orientation': o, 'amplitude': a,
                    'dev_first5': dev5, 'dev_continuation5_9': dev59, 'dev_mean': (dev5+dev59)/2,
                    'eval_first5': ev5, 'eval_continuation5_9': ev59, 'eval_mean': (ev5+ev59)/2})
    baseline = {}
    for split in SPLITS:
        v5 = keyed[('pool_unit_08', split, 'pi05', 'baseline', 'first5')]
        v59 = keyed[('pool_unit_08', split, 'pi05', 'baseline', 'continuation5_9')]
        h5 = keyed[('pool_unit_08', split, 'previous_hold', 'baseline', 'first5')]
        h59 = keyed[('pool_unit_08', split, 'previous_hold', 'baseline', 'continuation5_9')]
        baseline[split] = {'pi05_first5': v5, 'pi05_continuation5_9': v59, 'pi05_mean': (v5+v59)/2,
                           'previous_hold_first5': h5, 'previous_hold_continuation5_9': h59,
                           'previous_hold_mean': (h5+h59)/2}
    selectable = []
    for pool in pools:
        for o in ('O1_source','O2_pi05','O3_history'):
            for a in ('A2_history','A3_pi05','A4_source'):
                v5 = keyed[(pool, 'dev', o, a, 'first5')]
                v59 = keyed[(pool, 'dev', o, a, 'continuation5_9')]
                e5 = keyed[(pool, 'eval', o, a, 'first5')]
                e59 = keyed[(pool, 'eval', o, a, 'continuation5_9')]
                selectable.append({'pool': pool, 'orientation': o, 'amplitude': a,
                    'dev_first5': v5, 'dev_continuation5_9': v59, 'dev_mean': (v5+v59)/2,
                    'eval_first5': e5, 'eval_continuation5_9': e59, 'eval_mean': (e5+e59)/2})
    selectable.sort(key=lambda x: x['dev_mean'])
    best = selectable[0]
    gate = {'passed': bool(best['dev_mean'] < baseline['dev']['pi05_mean']),
            'selected': best, 'pi05_dev_mean': baseline['dev']['pi05_mean'],
            'rule': 'a non-oracle O1-O3 x A2-A4 tuple must beat Dev pi0.5 mean across first5 and continuation5_9; Eval is report-only'}
    return options, baseline, selectable, gate


def offline(output=OUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'offline').mkdir(parents=True, exist_ok=True)
    inputs, targets, segments = _inputs_and_targets()
    packs = (_pool95(), _pool_w())
    rows, _ = [], {}
    for pack in packs:
        for split in SPLITS:
            r, state = _score_pool(pack, split, inputs[split], targets[split], segments[split])
            rows.extend(r); _['%s_%s' % (pack['name'], split)] = state
    summary, baseline, gate_options, gate = _summary_and_gate(rows)
    report = {'experiment': 'MAG3', 'duration': 'T3 fixed 10', 'axes': '6-axis normalized action MSE',
        'type_selection': {'pool_unit_08': 'target Procrustes residual; same-task candidate excluded',
            'scale_only_W': 'target S-shared residual; same-task candidate excluded'},
        'orientation_rules': {'O1_source': 'source theta rotation; W uses its retained source direction',
            'O2_pi05': 'Procrustes rotation to the cached pi0.5 10-action chunk',
            'O3_history': 'minimum rotation from unit first-3-step translation to prior-10-step net translation; no-history/degenerate direction falls back to source rotation',
            'O4_oracle': 'target-fitted Procrustes rotation, diagnostic upper bound'},
        'amplitude_rules': {'A1_oracle': 'nonnegative target fit after chosen orientation',
            'A2_history': 'prior action RMS speed / oriented template RMS speed',
            'A3_pi05': 'nonnegative scalar fit of cached pi0.5 chunk; no extra rotation',
            'A4_source': 'saved source scale'},
        'baseline': baseline, 'summary_orientation_x_amplitude': summary,
        'gate_candidates': gate_options, 'gate': gate, 'combinations': rows,
        'dev_used_for_rule_selection': True, 'eval_used_for_selection': False}
    _json(output / 'offline_report.json', report)
    lines = ['# MAG3 offline: orientation × amplitude', '',
        'Selection uses Dev only; Eval is reporting-only. Every candidate type is oracle-selected within its pool type metric, same task excluded. T3 uses ten action samples; MSE is six-axis normalized action MSE.', '',
        '## Summary: orientation × amplitude', '',
        '| Pool | Orientation | Amplitude | Dev first5 | Dev 5–9 | Eval first5 | Eval 5–9 |',
        '|---|---|---|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['pool']} | {r['orientation']} | {r['amplitude']} | {r['dev_first5']:.6f} | {r['dev_continuation5_9']:.6f} | {r['eval_first5']:.6f} | {r['eval_continuation5_9']:.6f} |")
    lines += ['', 'Baselines (pool-independent):', '', '| Split | π0.5 first5 | π0.5 5–9 | hold-last first5 | hold-last 5–9 |', '|---|---:|---:|---:|---:|']
    for s in SPLITS:
        b = baseline[s]
        lines.append(f"| {s} | {b['pi05_first5']:.6f} | {b['pi05_continuation5_9']:.6f} | {b['previous_hold_first5']:.6f} | {b['previous_hold_continuation5_9']:.6f} |")
    lines += ['', '## Dev gate', '', f"Passed: **{gate['passed']}**. Selected tuple: **{gate['selected']['pool']} / {gate['selected']['orientation']} / {gate['selected']['amplitude']}**, Dev mean {gate['selected']['dev_mean']:.6f}, pi0.5 Dev mean {gate['pi05_dev_mean']:.6f}.", '',
        '## All combinations', '', '| Pool | Split | Orientation | Amplitude | Window | MSE | 95% episode bootstrap CI |', '|---|---|---|---|---|---:|---:|']
    for r in rows:
        lines.append(f"| {r['pool']} | {r['split']} | {r['orientation']} | {r['amplitude']} | {r['window']} | {r['mean']:.6f} | [{r['ci95'][0]:.6f}, {r['ci95'][1]:.6f}] |")
    (output / 'offline_report.md').write_text('\n'.join(lines) + '\n')
    _json(output / 'gate.json', gate)
    return report


def verify_server_parity(pack, selected, output, *, n_windows=10):
    """Integration check of the live selector->pose hook against offline drafts."""
    inputs = mag1._load_rows('dev')
    wdev = dict(np.load(WPREP / 'W_inputs_dev.npz', allow_pickle=False))
    if not np.array_equal(np.asarray(wdev['row']), np.asarray(inputs['row'])):
        raise AssertionError('W P5 Dev rows do not align with the frozen MAG1 Dev rows')
    if not np.allclose(np.asarray(wdev['hist']), np.asarray(inputs['hist']), atol=1e-7):
        raise AssertionError('W P5 and MAG1 Dev histories are not identical')
    selector = _selector(pack['name'])
    context = {'history_available': False}
    hooked = make_pose_hook(pack, selected['orientation'], selected['amplitude'], selector, context=context)
    n = min(int(n_windows), len(inputs['row']))
    rows = []
    max_abs = 0.0
    for i in range(n):
        history = np.asarray(inputs['hist'][i:i+1, :, :7], np.float32)
        state = np.asarray(wdev['state'][i:i+1], np.float32)
        pi = np.asarray(inputs['pi05'][i:i+1], np.float32)
        context['history_available'] = bool(np.any(np.abs(history) > 1e-8))
        context.pop('history_override', None)
        idx = hooked.pick(history, state, None)
        server_draft = hooked.pose_to(idx, pi)
        offline_draft, _ = render_draft(pack, int(np.asarray(idx).reshape(-1)[0]), pi[0], history[0],
            selected['orientation'], selected['amplitude'], history_available=context['history_available'])
        delta = np.abs(np.asarray(server_draft)[0, :, :6] - offline_draft[:, :6])
        err = float(delta.max())
        max_abs = max(max_abs, err)
        rows.append({'row': int(inputs['row'][i]), 'pool_index': int(np.asarray(idx).reshape(-1)[0]),
                     'max_abs_6axis': err, 'pass': err <= 1e-5})
    report = {'status': 'complete', 'pool': pack['name'], 'orientation': selected['orientation'],
        'amplitude': selected['amplitude'], 'method': 'actual selector.pick and installed server pose_to hook vs offline render_draft on identical Dev cache rows/actions/history',
        'windows': n, 'max_abs_6axis': max_abs, 'tolerance': 1e-5,
        'pass': bool(n >= 10 and max_abs <= 1e-5), 'rows': rows}
    _json(Path(output) / 'parity_check.json', report)
    if not report['pass']:
        raise AssertionError(f'server/offline parity failed: n={n}, max_abs={max_abs:g}')
    return report


def _make_policy(pack, selected, checkpoint):
    import threading
    orientation, amplitude = selected['orientation'], selected['amplitude']
    if pack['name'] == 'pool_unit_08':
        os.environ['MEM_POOL'] = str(POOL95)
        os.environ['MEM_CKPT'] = str(ROOT / 'cluster-pools-01/query_unit_08.pt')
    else:
        os.environ.pop('MEM_POOL', None); os.environ.pop('MEM_CKPT', None)
    policy = legacy.make_policy(checkpoint)
    selector = _selector(pack['name'])
    context = {'history_available': False}
    hooked = make_pose_hook(pack, orientation, amplitude, selector, context=context)
    policy.sel = hooked
    original_infer = policy.infer
    def infer(observation):
        episode = str(observation['s2_episode_id'])
        qid = int(observation.get('s2_query_id', 0)) % 1000
        previous = [] if qid == 0 else policy.hist.get(episode, [])
        context['history_available'] = bool(previous)
        context['history_override'] = np.asarray(previous[-10:], np.float32).reshape(-1, 7) if previous else np.zeros((0, 7), np.float32)
        result = original_infer(observation)
        dbg = result.get('s2_debug', {})
        dbg.update({'mag3_orientation': orientation, 'mag3_amplitude': amplitude,
            'mag3_pi05_arm_axes_used': ['scale only'] if amplitude == 'A3_pi05' else [],
            'mag3_pi05_gripper_used': True,
            'mag3_scale': float(context.get('last_render', {}).get('scale', 0.)),
            'mag3_history_fallback': bool(context.get('last_render', {}).get('fallback', False))})
        result['s2_debug'] = dbg
        return result
    policy.infer = infer
    return policy


def serve(args):
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '2')))
    pack = _pool95() if args.pool == 'pool_unit_08' else _pool_w()
    selected = {'pool': args.pool, 'orientation': args.orientation, 'amplitude': args.amplitude}
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    policy = _make_policy(pack, selected, args.checkpoint)
    WebsocketPolicyServer(policy, host='127.0.0.1', port=args.port,
        metadata={'experiment': 'MAG3', **selected}).serve_forever()


def _mcnemar(left, right, keys):
    from scipy.stats import binomtest
    succ = lambda z: z.get('status') == 'complete' and bool(z.get('success'))
    wins = sum(succ(left[k]) and not succ(right[k]) for k in keys)
    losses = sum(succ(right[k]) and not succ(left[k]) for k in keys)
    n = wins + losses
    return {'left_only': int(wins), 'right_only': int(losses), 'discordant': int(n),
        'p_two_sided_exact': float(binomtest(min(wins, losses), n, .5).pvalue) if n else 1.}


def _rollout_diagnostics(records):
    scales, unique, repeat, fallbacks = [], [], [], 0
    for row in records.values():
        events = [e for e in row.get('selected_units_and_t0', []) if 'mag3_scale' in e]
        units = [int(e['unit']) for e in events if 'unit' in e]
        scales.extend(float(e['mag3_scale']) for e in events)
        unique.append(len(set(units)))
        if len(units) > 1:
            repeat.append(float(np.mean(np.asarray(units[1:]) == np.asarray(units[:-1]))))
        fallbacks += sum(bool(e.get('mag3_history_fallback', False)) for e in events)
    return {'trials': len(records), 'mean_unique_units': float(np.mean(unique)) if unique else None,
        'median_unique_units': float(np.median(unique)) if unique else None,
        'same_unit_reselection_fraction': float(np.mean(repeat)) if repeat else None,
        'scale_quantiles': {str(q): float(np.quantile(scales, q)) for q in (0,.1,.5,.9,1)} if scales else {},
        'history_fallback_events': int(fallbacks)}


def run_closed_loop(output, selected, args):
    import subprocess
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import run as harness
    out = Path(output) / 'closed_loop'
    out.mkdir(parents=True, exist_ok=True)
    tasks = [int(x['task_id']) for x in json.loads((ROOT / 'baseline-01/plan.json').read_text())['selected_tasks']]
    servers, handles = [], []
    for slot in range(args.servers):
        port = harness.free_port()
        folder = Path(output) / 'servers' / str(slot); folder.mkdir(parents=True, exist_ok=True)
        handle = (folder / 'server.log').open('a'); handles.append(handle)
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
            XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION='.15',
            MAG3_POOL=selected['pool'])
        proc = subprocess.Popen(['taskset','-c','8-15','nice','-n','10',sys.executable,
            str(Path(__file__).resolve()),'serve','--checkpoint',str(args.checkpoint),'--port',str(port),
            '--pool',selected['pool'],'--orientation',selected['orientation'],'--amplitude',selected['amplitude']],
            env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        servers.append((proc, port))
    try:
        for proc, port in servers:
            harness.ready(proc, port)
        folder = out / 'mag3'; folder.mkdir(parents=True, exist_ok=True)
        finished = harness.read_terminal(folder / 'rollouts.jsonl')
        jobs = [(task, init) for task in tasks for init in range(args.inits)
                if (task,init) not in finished or finished[(task,init)].get('status') != 'complete']
        import threading
        lock = threading.Lock()
        def execute(task, init, port):
            job = {'host':'127.0.0.1','port':port,'suite':'libero_90','task_id':task,
                'trial_index':init,'seed_base':7,'condition':'mag3','memory_snapshot':'none',
                'query_text_source':'instruction_only','checkpoint':str(args.checkpoint),
                'wait_steps':10,'replan_steps':5,'max_steps':520,'record':init<2}
            root = folder / 'attempts' / f'task{task:02d}-init{init:02d}'
            existing = list(root.glob('attempt-*')) if root.exists() else []
            start = max([int(x.name.split('-')[-1]) for x in existing], default=0)+1
            row = None
            for attempt in range(start,start+2):
                row = harness.trial(job,root/f'attempt-{attempt}',args.timeout,attempt)
                harness.append(folder/'attempts.jsonl',row)
                if row['status']=='complete': break
            row.update(task_id=task,trial_index=init,task_suite='libero_90')
            harness.append(folder/'rollouts.jsonl',row)
            with lock:
                done = harness.read_terminal(folder/'rollouts.jsonl')
                harness.write(folder/'progress.json',{'processed':len(done),'scheduled':len(tasks)*args.inits,
                    'successes':sum(r.get('status')=='complete' and bool(r.get('success')) for r in done.values()),
                    'updated_unix':time.time()})
        def lane(slot):
            proc, port = servers[slot % len(servers)]
            for task, init in jobs[slot::len(servers)]:
                if proc.poll() is not None: raise RuntimeError(f'MAG3 server {slot} exited')
                execute(task,init,port)
        with ThreadPoolExecutor(max_workers=len(servers)) as pool:
            for f in as_completed([pool.submit(lane,i) for i in range(len(servers))]): f.result()
    finally:
        for p, _ in servers: harness.terminate(p)
        for h in handles: h.close()
    # Paired comparisons use the same task and public init index as baseline-01.
    references = {'baseline-01':ROOT/'baseline-01/production/rollouts.jsonl',
        'unit_08':ROOT/'closed-loop-memory-01/libero90_unit_08/memory/rollouts.jsonl',
        'R5':ROOT/'lib-closed-01/R5/rollouts.jsonl','state_only':ROOT/'lib-closed-01/state_only/rollouts.jsonl'}
    current = harness.read_terminal(out/'mag3/rollouts.jsonl')
    paired = {}
    summary = {}
    for name, path in references.items():
        ref = harness.read_terminal(path)
        keys = sorted(k for k in current if k in ref)
        paired[name] = {'n_paired':len(keys),'reference_successes':sum(bool(ref[k].get('success')) for k in keys),
            'mcnemar':_mcnemar(current,ref,keys)}
    keys = list(current)
    success_count = sum(r.get('status')=='complete' and bool(r.get('success')) for r in current.values())
    summary = {'selected':selected,'trials':len(keys),'successes':success_count,'paired_comparisons':paired,
        'diagnostics':_rollout_diagnostics(current),'seed_protocol':'run.py seed_base=7, trial_index=public initial state index; 10 selected tasks x 10 inits',
        'policy_axes':{'query':'pool-specific P5 for scale_only_W; query_unit_08 for pool_unit_08',
            'arm':'selected memory shape with selected orientation and scalar amplitude',
            'gripper':'pi0.5 dim 6','replan_steps':5,'action_horizon':10}}
    _json(Path(output)/'closed_loop_report.json',summary)
    return summary


def _init_output(output):
    output = Path(output)
    manifest = output / 'run_manifest.json'
    if output.exists():
        if not manifest.exists():
            raise FileExistsError(f'refusing to overwrite unrelated output directory: {output}')
        return output
    output.mkdir(parents=True)
    _json(manifest, {'experiment':'MAG3','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'write_once':True,'suite':'libero_90'})
    return output


def run(args):
    out = _init_output(args.output)
    report_path = out / 'offline_report.json'
    offline_report = json.loads(report_path.read_text()) if report_path.exists() else offline(out)
    gate = offline_report['gate']
    selected = gate['selected']
    pack = _pool95() if selected['pool'] == 'pool_unit_08' else _pool_w()
    parity_path = out / 'parity_check.json'
    if parity_path.exists():
        parity = json.loads(parity_path.read_text())
        if not parity.get('pass'): raise AssertionError('saved parity check did not pass')
    else:
        parity = verify_server_parity(pack, selected, out, n_windows=10)
    result = {'gate':gate,'parity_check':parity,'closed_loop':None}
    if gate['passed']:
        if not (out / 'closed_loop_report.json').exists():
            result['closed_loop'] = run_closed_loop(out, selected, args)
        else:
            result['closed_loop'] = json.loads((out/'closed_loop_report.json').read_text())
    result['closed_loop_gate_passed'] = bool(gate['passed'])
    _json(out / 'run_summary.json', result)
    _write_report(out, offline_report, result)
    (out/'DONE').write_text('complete\n')
    return result


def _write_report(out, offline_report, result):
    lines = ['# MAG3: orientation and amplitude', '',
        f"Dev gate: **{'PASS' if result['gate']['passed'] else 'FAIL'}**; selected {result['gate']['selected']['pool']} / {result['gate']['selected']['orientation']} / {result['gate']['selected']['amplitude']} (Dev mean {result['gate']['selected']['dev_mean']:.6f}; pi0.5 {result['gate']['pi05_dev_mean']:.6f}).", '',
        f"Server/offline draft parity: {result['parity_check']['windows']} Dev windows, maximum 6-axis difference {result['parity_check']['max_abs_6axis']:.8g} (limit 1e-5), {'passed' if result['parity_check']['pass'] else 'failed'}.", '',
        '## Orientation × amplitude summary (MSE, first5 / continuation5–9)', '',
        '| Pool | Orientation | Amplitude | Dev first5 | Dev 5–9 | Eval first5 | Eval 5–9 |', '|---|---|---|---:|---:|---:|---:|']
    for r in offline_report['summary_orientation_x_amplitude']:
        lines.append(f"| {r['pool']} | {r['orientation']} | {r['amplitude']} | {r['dev_first5']:.6f} | {r['dev_continuation5_9']:.6f} | {r['eval_first5']:.6f} | {r['eval_continuation5_9']:.6f} |")
    lines += ['', '## Dev gate candidates', '', '| Pool | Orientation | Amplitude | Dev mean | Eval mean (report only) |', '|---|---|---|---:|---:|']
    for r in offline_report['gate_candidates']:
        lines.append(f"| {r['pool']} | {r['orientation']} | {r['amplitude']} | {r['dev_mean']:.6f} | {r['eval_mean']:.6f} |")
    if result['closed_loop'] is None:
        lines += ['', '## Closed loop', '', 'Not run: the Dev-only non-oracle gate did not pass.']
    else:
        cl = result['closed_loop']
        lines += ['', '## Closed loop', '', f"Success: **{cl['successes']}/{cl['trials']}**.", '',
            '| Comparator | Paired n | Comparator successes | MAG3-only / comparator-only | Exact McNemar p |', '|---|---:|---:|---:|---:|']
        for name, d in cl['paired_comparisons'].items():
            m=d['mcnemar']; lines.append(f"| {name} | {d['n_paired']} | {d['reference_successes']} | {m['left_only']}/{m['right_only']} | {m['p_two_sided_exact']:.4g} |")
        lines += ['', 'Runtime diagnostics:', '', '```json', json.dumps(cl['diagnostics'], indent=2), '```']
    lines += ['', 'Full offline combinations, bootstrap intervals, per-row picks, and predictions are in `offline_report.json` and `offline/`.', '']
    (Path(out)/'report.md').write_text('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run'); r.add_argument('--output',type=Path,default=OUT)
    r.add_argument('--checkpoint',type=Path,default=CHECKPOINT); r.add_argument('--servers',type=int,default=3)
    r.add_argument('--inits',type=int,default=10); r.add_argument('--timeout',type=int,default=900)
    s = sub.add_parser('serve'); s.add_argument('--checkpoint',type=Path,required=True); s.add_argument('--port',type=int,required=True)
    s.add_argument('--pool',choices=('pool_unit_08','scale_only_W'),required=True)
    s.add_argument('--orientation',choices=ORIENTATIONS,required=True); s.add_argument('--amplitude',choices=AMPLITUDES,required=True)
    a = p.parse_args()
    if a.cmd == 'run':
        if not 1 <= a.servers <= 3: raise ValueError('MAG3 supports at most three GPU servers')
        result = run(a); print(json.dumps(result,indent=2,ensure_ascii=False))
    else:
        serve(a)


if __name__ == '__main__': main()
