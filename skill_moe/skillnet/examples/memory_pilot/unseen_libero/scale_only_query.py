"""S1: motion-memory retrieval with scale-only types (direction kept, gripper separate).

The memory types used by K1-K4 and unit_retrieval removed rotation (SO(3)) and scale.  Here a type
keeps its direction; only its size is removed, and the gripper is kept out of the type.

Pools (LIBERO-40 memory demos, holdout-task units removed)
  U  variable-length units cut at speed minima + gripper switches (as unit_retrieval), stored as the
     physical DCT-3 of the 6 motion axes + length (4,516 units, 3,504 kept).  Direction is not removed.
  W  the K1-K4 10-frame windows (2,049, 1,595 kept), decoded with their own source pose (theta_src),
     i.e. the stored motion in its original direction and size.

Type conditions
  S-shared  one scale over the 6 motion axes (physical units):      e = min_{s>=0} ||s u - q||^2
  S-sep     separate position / rotation scales:                    e = sum_g min_{s_g>=0} ||s_g u_g - q_g||^2
  R-so3     control: the current SO(3) + one-scale Procrustes error
  Errors are mean over frames x 6 axes; NO rotation is applied for S-*.

Query: K1 P5 (last 10 executed actions + state -> MLP; key = type -> MLP; cosine x learned scale;
listwise loss to a softmax of the type error; candidates of the query's own task excluded).  On W also
P6 (final-layer token group means + history + state).  Three seeds.  Teacher temperature (relative to
the median random-oracle gap) and epoch are selected on the LIBERO-40 holdout tasks; a Dev-selected
(LIBERO-90 tasks 6, 7) variant is reported alongside.  LIBERO-90 Eval is never used for selection.

Size at inference (no ground truth):  stored | hist (speed of the last executed actions) |
pi05 (speed of the pi0.5 chunk available at the start) | oracle s* (upper bound, labelled).
R-so3 uses theta_src (= stored), hist (stored direction, history speed), theta_pi05 (Procrustes onto
the pi0.5 chunk) and theta* (Procrustes onto the target, upper bound).
Gripper: G1 (hold the last executed gripper command; at an episode start threshold the state s6).

Ground-truth actions are used only (a) for the selection teacher on the LIBERO-40 side (and Dev type
errors for the Dev-selected variant) and (b) for scoring, in ``score_*`` functions.  The generation
functions (``generate_w`` / ``generate_u``) refuse inputs that contain target keys.

P0 (VLM group means only, P6 network) is trained and scored when ``modes`` in prep.json lists it.

Orientation fix: scale-only-01 used Dev/Eval VLM means and pi0.5 chunks computed on images that were rotated
180 degrees.  ``refresh`` builds a new run (scale-only-02) from a finished one: the LIBERO-40 side (pools,
train inputs, teacher errors, trained query models, holdout selection) is copied unchanged and only the
image-derived Dev/Eval inputs (``mean``, ``pi05``; U ``pi05_now`` / ``pi05_span``) are replaced by the
orientation_fix.py outputs.  The Dev-selected states of models whose input uses the VLM mean were chosen on
the old Dev features and are flagged in the report.

Stages:  python scale_only_query.py {prepare,refresh,train,evaluate,report,all} [--smoke] [--out DIR]
         refresh: --src <finished run> --fix <orientation_fix output>; then train --modes P0, evaluate, report
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', Path(__file__).resolve().parents[5] / '.skillnet-state')).expanduser()
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = STATE_DIR / 'outputs/memory-pilot/unseen-libero'
OUT = ROOT / 'scale-only-01'
K1 = ROOT / 'key-query-01'
K3 = ROOT / 'key-query-03'
K4 = ROOT / 'key-query-04'
UR = ROOT / 'unit-retrieval-01'
THETA_FEATURES = ROOT / 'theta-regression-01' / 'production' / 'features'
HOLDOUT = (2, 7, 12, 17, 22, 27, 32, 37)
SEEDS = (4101, 4102, 4103)
CONDS = ('S-shared', 'S-sep', 'R-so3')
FRACS = (1 / 3, 1 / 10, 1 / 30)
RULES = ('stored', 'hist', 'pi05', 'oracle')
MODES = {'W': ('P5', 'P6'), 'U': ('P5',)}
MEAN_MODES = ('P0', 'P6')          # query inputs that contain the VLM group means
EPOCHS = {'W': 20, 'U': 40}
L16 = 16
KDCT = 3
EPS = 1e-12
TARGET_KEYS = ('target', 'targets', 'target16', 'target_norm', 'actions', 'y', 'truth')
BOOT_SEED = 4101


def _mods():
    import compressed_denoise as cd, variable_shape_memory as vs, key_query_probe as kq
    return cd, vs, kq


def _atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp'); tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False)); tmp.replace(path)


def _savez(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + '.tmp.npz'); np.savez(tmp, **arrays); tmp.replace(path)


def log(out, **rec):
    rec = {'time': time.strftime('%Y-%m-%dT%H:%M:%S'), **rec}
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    with open(Path(out) / 'progress.log', 'a') as f: f.write(json.dumps(rec, ensure_ascii=False) + '\n')


# ======================================================================== geometry
def groups_of(cond):
    """Axis groups that share one scale: S-shared/R-so3 -> one group of 6, S-sep -> position, rotation."""
    if cond == 'S-sep': return (slice(0, 3), slice(3, 6))
    if cond in ('S-shared', 'R-so3'): return (slice(0, 6),)
    raise ValueError(cond)


def rms_speed(x, g=slice(0, 6), count=None):
    """Root-mean-square per-frame norm over the axes in g.  x [...,T,6]; count = valid trailing frames."""
    x = np.asarray(x, np.float64)[..., g]
    if count is None: return np.sqrt((x ** 2).sum((-1, -2)) / x.shape[-2])
    return np.sqrt((x ** 2).sum((-1, -2)) / np.maximum(np.asarray(count, np.float64), 1))


def type_of(x, cond):
    """Scale-free type of physical motion x [...,T,6] (direction kept); R-so3 not handled here."""
    if cond not in ('S-shared', 'S-sep'): raise ValueError('type_of is for scale-only conditions')
    x = np.asarray(x, np.float64); out = np.empty_like(x)
    for g in groups_of(cond):
        out[..., g] = x[..., g] / np.maximum(rms_speed(x, g), 1e-9)[..., None, None]
    return out


def scale_only_error(q, u, cond, chunk=2048):
    """[N,K] min_{s_g>=0} sum_g ||s_g u_g - q_g||^2 / (T*6).  q [N,T,6], u [K,T,6] physical.  No rotation."""
    q = np.asarray(q, np.float64); u = np.asarray(u, np.float64)
    if q.ndim != 3 or u.ndim != 3 or q.shape[1:] != u.shape[1:] or q.shape[2] != 6: raise ValueError((q.shape, u.shape))
    n, t, _ = q.shape; out = np.empty((n, len(u)), np.float32)
    flat_u = [u[:, :, g].reshape(len(u), -1) for g in groups_of(cond)]
    uu = [(fu ** 2).sum(1) for fu in flat_u]
    for st in range(0, n, chunk):
        acc = np.zeros((min(chunk, n - st), len(u)))
        for g, fu, nu in zip(groups_of(cond), flat_u, uu):
            fq = q[st:st + chunk, :, g].reshape(-1, fu.shape[1])
            dot = np.maximum(fq @ fu.T, 0.)
            acc += (fq ** 2).sum(1)[:, None] - np.where(nu[None] > EPS, dot ** 2 / np.maximum(nu[None], EPS), 0.)
        out[st:st + chunk] = np.maximum(acc, 0.) / (t * 6)
    return out


def closed_form_scale(u, q, cond):
    """Per-pair optimal non-negative scales [N,G] for u, q [N,T,6] (no rotation)."""
    u = np.asarray(u, np.float64); q = np.asarray(q, np.float64)
    return np.stack([np.maximum((u[..., g] * q[..., g]).sum((-1, -2)), 0.) / np.maximum((u[..., g] ** 2).sum((-1, -2)), EPS)
                     for g in groups_of(cond)], -1)


def apply_scales(u, s, cond):
    u = np.asarray(u, np.float64).copy()
    for j, g in enumerate(groups_of(cond)): u[..., g] *= np.asarray(s)[..., j, None, None]
    return u


def match_speed(u, ref, cond, ref_count=None):
    """Rescale u so each scale group's rms speed equals that of ref (direction of u untouched)."""
    s = np.stack([rms_speed(ref, g, ref_count) / np.maximum(rms_speed(u, g), 1e-9) for g in groups_of(cond)], -1)
    return apply_scales(u, s, cond), s


def procrustes_rs(y, x):
    """Rotation r [3,3] and scale s >= 0 so that [y_t r, y_r r] * s best matches x (as compressed_denoise._procrustes)."""
    y = np.asarray(y, np.float64); x = np.asarray(x, np.float64)
    cross = y[:, :3].T @ x[:, :3] + y[:, 3:6].T @ x[:, 3:6]
    uu, _, vt = np.linalg.svd(cross); corr = np.eye(3); corr[-1, -1] = np.linalg.det(uu @ vt); r = uu @ corr @ vt
    yr = np.concatenate([y[:, :3] @ r, y[:, 3:6] @ r], 1)
    return r, max(float((yr * x).sum() / max((y * y).sum(), EPS)), 0.)


def rotate_scale(y, r, s):
    y = np.asarray(y, np.float64)
    return np.concatenate([y[:, :3] @ r, y[:, 3:6] @ r], 1) * s


def procrustes_error_w(q, u, device):
    """R-so3 control for 10-frame windows (key_query_probe.shape_error_matrix on padded shapes)."""
    _, vs, kq = _mods()
    pad = lambda a: np.concatenate([np.asarray(a, np.float32), np.zeros((*a.shape[:2], 1), np.float32)], -1)
    return kq.shape_error_matrix(pad(q), pad(u), vs, batch=512 if str(device).startswith('cuda') else 8, device=device)


def procrustes_error_u(q16, u16, device):
    import unit_retrieval as ur
    return ur.procrustes_err(u16, q16, device=device)


def type_errors(q, u, cond, device='cpu'):
    """Teacher / scoring type error for one condition (physical, mean over frames x 6)."""
    if cond == 'R-so3':
        return procrustes_error_w(q, u, device) if q.shape[1] == 10 else procrustes_error_u(q, u, device)
    return scale_only_error(q, u, cond)


# ======================================================================== DCT units
def dct3_values(phys):
    """Physical DCT-3 values (18) of a [T,6] motion, normalised by sqrt(T) like variable_shape_memory.encode."""
    from scipy.fft import dct
    p = np.asarray(phys, np.float64); c = np.zeros((KDCT, 6))
    full = dct(p, type=2, norm='ortho', axis=0); c[:min(KDCT, len(p))] = full[:KDCT]
    return (c / np.sqrt(len(p))).ravel()


def dct3_decode(values, n):
    """Decode 18 DCT-3 values at length n -> [n,6] physical."""
    from scipy.fft import idct
    c = np.zeros((n, 6)); k = min(KDCT, n); c[:k] = np.asarray(values[:KDCT * 6], np.float64).reshape(KDCT, 6)[:k] * np.sqrt(n)
    return idct(c, type=2, norm='ortho', axis=0)


# ======================================================================== data: pools
def build_w_pool():
    _, vs, _ = _mods(); import theta_regression as tr
    p0 = tr.candidate_shape_pool(rep='C_vm_none'); keep = ~np.isin(p0['source_task'], HOLDOUT)
    shape = p0['shape'][keep]; theta = p0['source_theta'][keep]
    dec = np.stack([tr.decode_theta(s, t, vs) for s, t in zip(shape, theta)])
    phys = vs.to_physical(dec[:, :, :6]).astype(np.float32)
    k3 = np.load(K3 / 'prep.npz')
    if not np.allclose(k3['pool_shape'], shape): raise AssertionError('W pool differs from the K1-K4 pool')
    return {'phys': phys, 'canon': shape[:, :, :6].astype(np.float32), 'task': p0['source_task'][keep].astype(np.int64),
            'keep': np.ones(len(shape), bool)}


def build_u_pool():
    cd, vs, _ = _mods()
    R = cd.RUN / 'rows'; ld = lambda k: np.load(R / f'{k}.npy', mmap_mode='r', allow_pickle=False)
    task, ep, fr = (np.asarray(ld(k)) for k in ('task', 'episode', 'frame'))
    mem = np.asarray(np.load(cd.RUN / 'memory_row_ids.npy'), np.int64)
    traces, _ = cd._stitch(ld('actions'), task, ep, fr, mem)
    vals, rms_vals, meta = [], [], []
    for key in sorted(traces):          # same order as compressed_denoise._bank_from_memory(mode='variable')
        t = traces[key]
        for s, e in cd._anchor_spans(t):
            if e <= s: continue
            vals.append(dct3_values(vs.to_physical(t[s:e, :6]))); rms_vals.append(vs.encode(t, s, e, 'rms', 1)[:KDCT * 6])
            meta.append((key[0], key[1], s, e - s))
    meta = np.asarray(meta); cached = np.load(UR / 'units.npz')
    if not (len(meta) == len(cached['task']) and (meta[:, 0] == cached['task']).all() and (meta[:, 2] == cached['start']).all()
            and (meta[:, 3] == cached['lengths']).all() and np.allclose(np.asarray(rms_vals), cached['values'][:, :KDCT * 6], atol=1e-5)):
        raise AssertionError('U units differ from unit_retrieval units')
    vals = np.asarray(vals)
    return {'values': vals.astype(np.float32), 'canon_values': np.asarray(rms_vals, np.float32), 'lengths': meta[:, 3].astype(np.int64),
            'task': meta[:, 0].astype(np.int64), 'episode': meta[:, 1].astype(np.int64), 'start': meta[:, 2].astype(np.int64),
            'keep': ~np.isin(meta[:, 0], HOLDOUT), 'phys16': np.stack([dct3_decode(v, L16) for v in vals]).astype(np.float32)}


# ======================================================================== data: queries
def _resample(x, n=L16):
    x = np.asarray(x, np.float64); t = np.linspace(0, len(x) - 1, n)
    return np.stack([np.interp(t, np.arange(len(x)), x[:, j]) for j in range(x.shape[1])], 1)


def build_segments(actions, task, episode, frame, state, ids, pi05=None, hard_by_row=None):
    """Speed-minimum segments with past-only inputs (as unit_retrieval.segments_from) + scoring targets.

    Returns (inputs, targets).  ``inputs`` holds only what is available at the segment start: the
    unit_retrieval history (last 10 executed actions, zeros if s<10), the available past (up to 10
    frames, front-padded) and its count, the last executed action, the state of the row containing s,
    and the pi0.5 chunk of that row from s on (observation at the row start <= s).  ``targets`` holds the
    true segment (normalised, variable length) and its physical resampling, for teacher/scoring only.
    """
    cd, vs, _ = _mods()
    ids = np.asarray(ids, np.int64); pos_of = {int(r): i for i, r in enumerate(ids)}
    traces, row_at = cd._stitch(actions, task, episode, frame, ids)
    first = {}
    for r, (epk, off) in row_at.items(): first.setdefault(epk, {})[off] = r
    I = {k: [] for k in ('task', 'episode', 'start', 'row', 'hist', 'past', 'n_past', 'last', 'state', 'pi05_now', 'n_pi05', 'pi05_span', 'hard')}
    T = {'target_norm': [], 'target16': [], 'length': []}
    for epk, tr in sorted(traces.items()):
        for s, e in cd._anchor_spans(tr):
            if e - s < 2: continue
            r = first[epk][(s // 10) * 10]; npast = min(s, 10)
            past = np.zeros((10, 7), np.float32)
            if npast: past[10 - npast:] = tr[s - npast:s, :7]
            I['task'].append(epk[0]); I['episode'].append(epk[1]); I['start'].append(s); I['row'].append(r)
            I['hist'].append(tr[s - 10:s, :7] if s >= 10 else np.zeros((10, 7), np.float32))
            I['past'].append(past); I['n_past'].append(npast)
            I['last'].append(tr[s - 1, :7] if s >= 1 else np.zeros(7, np.float32))
            I['state'].append(np.asarray(state[r][:8], np.float32))
            I['hard'].append(bool(hard_by_row.get(int(r), False)) if hard_by_row is not None else False)
            if pi05 is not None:
                now = np.zeros((10, 7), np.float32); m = 10 - s % 10; now[:m] = pi05[pos_of[r], s % 10:, :7]
                I['pi05_now'].append(now); I['n_pi05'].append(m)
                span = []
                for f in range(s, e):
                    rr = first[epk][(f // 10) * 10]; span.append(pi05[pos_of[rr], f % 10, :7])
                I['pi05_span'].append(np.asarray(span, np.float32))
            T['target_norm'].append(np.asarray(tr[s:e, :7], np.float32)); T['length'].append(e - s)
            T['target16'].append(_resample(vs.to_physical(tr[s:e, :6])))
    inputs = {k: np.asarray(v) for k, v in I.items() if k not in ('pi05_span', 'pi05_now', 'n_pi05')}
    if pi05 is not None:
        inputs['pi05_now'] = np.asarray(I['pi05_now'], np.float32); inputs['n_pi05'] = np.asarray(I['n_pi05'])
        inputs['pi05_span_flat'], inputs['pi05_span_off'] = _flatten(I['pi05_span'])
    tgt = {'target16': np.asarray(T['target16'], np.float32), 'length': np.asarray(T['length'])}
    tgt['target_flat'], tgt['target_off'] = _flatten(T['target_norm'])
    return inputs, tgt


def _flatten(seqs):
    off = np.r_[0, np.cumsum([len(s) for s in seqs])].astype(np.int64)
    return (np.concatenate(seqs).astype(np.float32) if len(seqs) else np.zeros((0, 7), np.float32)), off


def _unflatten(flat, off, i):
    return flat[off[i]:off[i + 1]]


def _subset(n, limit):
    return np.arange(n) if limit is None or limit >= n else np.unique(np.linspace(0, n - 1, limit).astype(np.int64))


# ======================================================================== prepare
def prepare(out, *, limit=None, device='cuda'):
    """Pools, past-only query inputs, LIBERO-40 teacher errors and Dev selection errors (no Eval targets)."""
    cd, vs, kq = _mods(); out = Path(out); P = out / 'prep'; P.mkdir(parents=True, exist_ok=True)
    if (P / 'DONE').exists(): return
    base, rows, unseen, splits = kq._production_data_splits(include_eval_targets=False)
    U = kq.UNSEEN; lu = lambda k: np.load(U / f'{k}.npy', mmap_mode='r', allow_pickle=False)
    ut, ue, uf = (np.asarray(lu(k)) for k in ('task', 'episode', 'frame'))
    train_ids, dev_ids, eval_ids = (np.asarray(splits[s][1]) for s in ('train', 'dev', 'eval'))
    k3 = np.load(K3 / 'prep.npz'); k4 = np.load(K4 / 'prep.npz'); k1 = np.load(K1 / 'production' / 'evaluation' / 'rows.npz')
    if not (np.array_equal(k4['eval_rows'], eval_ids) and np.array_equal(k1['row_ids'], eval_ids) and np.array_equal(k4['dev_rows'], dev_ids)):
        raise AssertionError('cached K1/K3/K4 rows are misaligned with the fixed split')
    pi_dev = np.asarray(np.load(K1 / 'policy_features' / 'dev_actions.npy'), np.float32)
    pi_eval = np.asarray(np.load(K1 / 'policy_features' / 'eval_actions.npy'), np.float32)
    hard_by_row = {int(r): bool(h) for r, h in zip(eval_ids, k1['hard'])}
    # ---------------- W
    pool = build_w_pool(); _savez(P / 'W_pool.npz', **pool)
    sub = _subset(len(train_ids), limit)
    tr_task = np.asarray(rows['task'][train_ids], np.int64)[sub]
    winp = {'train': {'hist': k3['train_hist'][sub], 'state': k3['train_state'][sub], 'mean': k3['train_mean'][sub].reshape(len(sub), -1),
                      'task': tr_task, 'frame': np.asarray(rows['frame'][train_ids])[sub], 'episode': np.asarray(rows['episode'][train_ids])[sub]},
            'dev': {'hist': k4['dev_hist'], 'state': k4['dev_state'], 'mean': np.asarray(np.load(THETA_FEATURES / 'dev.npy'), np.float32).reshape(len(dev_ids), -1),
                    'task': ut[dev_ids].astype(np.int64), 'frame': uf[dev_ids], 'episode': ue[dev_ids], 'pi05': pi_dev[:, :, :7], 'row': dev_ids},
            'eval': {'hist': k3['eval_hist'], 'state': k3['eval_state'], 'mean': k3['eval_mean'].reshape(len(eval_ids), -1),
                     'task': ut[eval_ids].astype(np.int64), 'frame': uf[eval_ids], 'episode': ue[eval_ids], 'pi05': pi_eval[:, :, :7],
                     'hard': np.asarray(k1['hard'], bool), 'row': eval_ids}}
    if not np.array_equal(k3['train_task'][sub], tr_task) or not np.array_equal(k3['eval_task'], winp['eval']['task']):
        raise AssertionError('K3 inputs misaligned')
    for s, d in winp.items(): d['has_hist'] = np.abs(d['hist']).sum((1, 2)) > 0; _savez(P / f'W_inputs_{s}.npz', **d)
    # LIBERO-40 labels (teacher) and Dev labels (Dev selection only); Eval labels are not read here.
    q_train = vs.to_physical(np.asarray(rows['actions'][train_ids[sub], :, :6], np.float32)).astype(np.float32)
    q_dev = vs.to_physical(np.asarray(lu('actions')[dev_ids, :, :6], np.float32)).astype(np.float32)
    for cond in CONDS:
        for split, q in (('train', q_train), ('dev', q_dev)):
            E = type_errors(q, pool['phys'], cond, device)
            _savez(P / f'W_err_{cond}_{split}.npz', err=E.astype(np.float32))
        log(out, stage='prepare', pool='W', cond=cond, rows=len(q_train))
    ref = k3['train_err'][sub[:200]]; mine = np.load(P / 'W_err_R-so3_train.npz')['err'][:200]
    w_check = float(np.max(np.abs(ref - mine)))
    # ---------------- U
    upool = build_u_pool(); _savez(P / 'U_pool.npz', **upool)
    R = cd.RUN / 'rows'; ld = lambda k: np.load(R / f'{k}.npy', mmap_mode='r', allow_pickle=False)
    l40_in, l40_t = build_segments(ld('actions'), np.asarray(rows['task']), np.asarray(rows['episode']), np.asarray(rows['frame']),
                                   ld('state'), train_ids)
    cached = np.load(UR / 'segments_libero40.npz')
    if not (np.array_equal(cached['start'], l40_in['start']) and np.array_equal(cached['task'], l40_in['task'])
            and np.allclose(cached['hist'], l40_in['hist']) and np.allclose(cached['target'], l40_t['target16'], atol=1e-4)):
        raise AssertionError('LIBERO-40 segments differ from unit_retrieval')
    ssub = _subset(len(l40_in['task']), limit)
    l40_in = {k: v[ssub] for k, v in l40_in.items()}; l40_q = l40_t['target16'][ssub]
    dev_in, dev_t = build_segments(lu('actions'), ut, ue, uf, lu('state'), dev_ids, pi05=pi_dev)
    ev_in, _ = build_segments(lu('actions'), ut, ue, uf, lu('state'), eval_ids, pi05=pi_eval, hard_by_row=hard_by_row)
    cu = np.load(UR / 'segments_unseen.npz')
    if not (np.array_equal(cu['start'], ev_in['start']) and np.array_equal(cu['task'], ev_in['task']) and np.allclose(cu['hist'], ev_in['hist'])):
        raise AssertionError('unseen Eval segments differ from unit_retrieval')
    for name, d in (('train', l40_in), ('dev', dev_in), ('eval', ev_in)):
        d = dict(d); d['has_hist'] = d['n_past'] > 0; _savez(P / f'U_inputs_{name}.npz', **d)
    _savez(P / 'U_dev_targets.npz', **dev_t)
    for cond in CONDS:
        for split, q in (('train', l40_q), ('dev', dev_t['target16'])):
            E = type_errors(q, upool['phys16'], cond, device)
            _savez(P / f'U_err_{cond}_{split}.npz', err=E.astype(np.float32))
        log(out, stage='prepare', pool='U', cond=cond, rows=len(l40_q))
    ue_ref = np.load(UR / 'err_libero40.npz')['err'][ssub[:200]]; ue_mine = np.load(P / 'U_err_R-so3_train.npz')['err'][:200]
    meta = {'limit': limit, 'W_train_rows': int(len(sub)), 'U_train_segments': int(len(ssub)), 'W_pool': int(len(pool['phys'])),
            'U_pool': int(len(upool['keep'])), 'U_pool_kept': int(upool['keep'].sum()), 'W_dev': int(len(dev_ids)), 'W_eval': int(len(eval_ids)),
            'U_dev': int(len(dev_in['task'])), 'U_eval': int(len(ev_in['task'])), 'U_eval_hard': int(ev_in['hard'].sum()),
            'check_W_Rso3_vs_K3_maxabs': w_check, 'check_U_Rso3_vs_unit_retrieval_maxabs': float(np.max(np.abs(ue_ref - ue_mine))),
            'gripper_threshold': gripper_threshold(rows, train_ids), 'global_mean_action': global_mean_action(rows, train_ids).tolist(),
            'eval_targets_read': False}
    _atomic_json(P / 'prep.json', meta); (P / 'DONE').write_text('ok\n'); log(out, stage='prepare', done=True, **{k: v for k, v in meta.items() if k != 'global_mean_action'})


def modes_of(out):
    """Query modes of a run (prep.json 'modes'; scale-only-01 predates the key and uses MODES)."""
    p = Path(out) / 'prep' / 'prep.json'
    m = json.loads(p.read_text()).get('modes') if p.exists() else None
    return {k: tuple(v) for k, v in m.items()} if m else MODES


def refresh(out, src, fix, *, modes=None):
    """New run from a finished one with corrected image-derived Dev/Eval inputs (see module docstring).

    Copied unchanged from ``src``: pools, LIBERO-40 train inputs and teacher errors, Dev type errors, U Dev
    targets, trained models, selection.json.  Replaced from ``fix`` (orientation_fix.py): W ``mean`` (VLM
    group means) and ``pi05`` (K1 noise key) for Dev/Eval, and the U segments' pi0.5 inputs (rebuilt with
    build_segments from past-only data, exactly as prepare).  No ground-truth action of Eval is read.
    """
    import shutil
    cd, vs, kq = _mods(); out, src, fix = Path(out), Path(src), Path(fix); P, SP = out / 'prep', src / 'prep'
    if (P / 'DONE').exists(): return
    if out.exists() and any(out.iterdir()): raise FileExistsError(f'{out} is not empty; runs are never overwritten')
    P.mkdir(parents=True, exist_ok=True); (out / 'models').mkdir(parents=True, exist_ok=True)
    replaced = {f'{pn}_inputs_{s}.npz' for pn in ('W', 'U') for s in ('dev', 'eval')}
    for f in sorted(SP.iterdir()):
        if f.name not in replaced | {'DONE', 'prep.json'}: shutil.copy2(f, P / f.name)
    for f in sorted((src / 'models').glob('*.pt')): shutil.copy2(f, out / 'models' / f.name)
    shutil.copy2(src / 'selection.json', out / 'selection.json')
    U = kq.UNSEEN; lu = lambda k: np.load(U / f'{k}.npy', mmap_mode='r', allow_pickle=False)
    ut, ue, uf = (np.asarray(lu(k)) for k in ('task', 'episode', 'frame'))
    k1 = np.load(K1 / 'production' / 'evaluation' / 'rows.npz'); hard_by_row = {int(r): bool(h) for r, h in zip(k1['row_ids'], k1['hard'])}
    pi, check = {}, {}
    for split in ('dev', 'eval'):
        old = dict(np.load(SP / f'W_inputs_{split}.npz')); ids = old['row']
        z = np.load(fix / 'pi05' / f'{split}_k1.npz')
        if not np.array_equal(z['row'], ids): raise AssertionError(f'{split}: pi0.5 rows misaligned')
        mean = np.asarray(np.load(fix / 'features' / f'{split}.npy'), np.float32).reshape(len(ids), -1)
        if mean.shape != old['mean'].shape: raise AssertionError('VLM mean shape changed')
        pi[split] = np.asarray(z['actions'], np.float32)
        new = {**old, 'mean': mean, 'pi05': pi[split][:, :, :7]}
        _savez(P / f'W_inputs_{split}.npz', **new)
        check[f'W_{split}_mean_change_rms'] = float(np.sqrt(((mean - old['mean']) ** 2).mean()))
        check[f'W_{split}_pi05_change_rms'] = float(np.sqrt(((new['pi05'] - old['pi05']) ** 2).mean()))
        uold = dict(np.load(SP / f'U_inputs_{split}.npz'))
        unew, _ = build_segments(lu('actions'), ut, ue, uf, lu('state'), ids, pi05=pi[split],
                                 hard_by_row=hard_by_row if split == 'eval' else None)
        unew = dict(unew); unew['has_hist'] = unew['n_past'] > 0
        for k in uold:                      # everything but the pi0.5 inputs must be identical
            if k in ('pi05_now', 'pi05_span_flat'): continue
            if not np.array_equal(uold[k], unew[k]): raise AssertionError(f'U {split}: {k} changed')
        _savez(P / f'U_inputs_{split}.npz', **unew)
    meta = json.loads((SP / 'prep.json').read_text())
    meta.update(refreshed_from=str(src), image_inputs_from=str(fix), unseen_rows=str(U), refresh_checks=check,
                modes={k: list(v) for k, v in (modes or MODES).items()},
                note='Dev/Eval の VLM 平均と π0.5 出力を、向きを直した画像で取り直した（orientation_fix.py）。LIBERO-40 側'
                     '（プール、学習入力、教師、学習済みクエリ、保留タスクでの選択）は scale-only-01 のまま。'
                     'VLM 平均を使うクエリ（P6）の「Dev で選んだ状態」は、旧い（逆さの）Dev 特徴で選ばれたまま。')
    _atomic_json(P / 'prep.json', meta); (P / 'DONE').write_text('ok\n'); log(out, stage='refresh', **check)


def correlations(out, *, seeds=SEEDS, device='cpu'):
    """Per-query Spearman / Pearson between a query's candidate scores and -(true type error) on Eval.
    Learned queries: holdout-selected state, mean over seeds of the per-seed averages.  pi05_nn: score =
    -(type error between the pi0.5 chunk and the candidate), W only.  Eval targets are read here for scoring."""
    import torch
    from scipy.stats import spearmanr, pearsonr
    cd, vs, kq = _mods(); out = Path(out); P = out / 'prep'; MD = modes_of(out)
    sel = json.loads((out / 'selection.json').read_text())['selected']
    ua = np.load(kq.UNSEEN / 'actions.npy', mmap_mode='r', allow_pickle=False)          # scoring only

    def corr(sc, E, A, hard):
        sp = np.array([spearmanr(sc[i, A[i]], -E[i, A[i]]).statistic for i in range(len(E))])
        pe = np.array([pearsonr(sc[i, A[i]], -E[i, A[i]]).statistic for i in range(len(E))])
        return sp.mean(), sp[hard].mean(), pe.mean(), pe[hard].mean()

    def net_scores(pool_name, cond, mode, seed, inp, pool):
        s = sel[f'{pool_name}|{cond}|{mode}|{seed}']['holdout']
        r = torch.load(out / 'models' / f"{pool_name}_{cond}_{mode}_s{seed}_f{s['frac']:.4f}.pt", weights_only=False)
        X = torch.as_tensor(query_inputs(inp, mode)); KI = torch.as_tensor(key_inputs(pool_name, pool, cond))
        net = make_net(pool_name, mode, X.shape[1], KI.shape[1]); net.load_state_dict(r['holdout']['state']); net.eval()
        with torch.no_grad(): return net(X, KI).numpy()

    def put(key, vals):
        a = np.asarray(vals, np.float64)
        res[key] = {m: [float(a[:, j].mean()), float(a[:, j].min()), float(a[:, j].max())]
                    for j, m in enumerate(('spearman', 'spearman_hard', 'pearson', 'pearson_hard'))}
    res = {}
    pool = dict(np.load(P / 'W_pool.npz')); inp = dict(np.load(P / 'W_inputs_eval.npz'))
    y = np.asarray(ua[inp['row'], :, :6], np.float32); yphys = vs.to_physical(y).astype(np.float32)
    A = allowed_matrix(inp['task'], pool['task'], pool['keep'], False); hard = inp['hard'].astype(bool)
    qpi = vs.to_physical(np.asarray(inp['pi05'])[:, :, :6]).astype(np.float32)
    for cond in CONDS:
        E = type_errors(yphys, pool['phys'], cond, device)
        for mode in MD['W']: put(f'W|{cond}|{mode}', [corr(net_scores('W', cond, mode, s, inp, pool), E, A, hard) for s in seeds])
        put(f'W|{cond}|pi05_nn', [corr(-type_errors(qpi, pool['phys'], cond, device), E, A, hard)])
        log(out, stage='correlations', pool='W', cond=cond)
    upool = dict(np.load(P / 'U_pool.npz')); uin = dict(np.load(P / 'U_inputs_eval.npz'))
    _, ut_all = build_segments(ua, *(np.asarray(np.load(kq.UNSEEN / f'{k}.npy', mmap_mode='r')) for k in ('task', 'episode', 'frame')),
                               np.load(kq.UNSEEN / 'state.npy', mmap_mode='r'), inp['row'])
    A = allowed_matrix(uin['task'], upool['task'], upool['keep'], False); hard = uin['hard'].astype(bool)
    for cond in CONDS:
        E = type_errors(ut_all['target16'], upool['phys16'], cond, device)
        for mode in MD['U']: put(f'U|{cond}|{mode}', [corr(net_scores('U', cond, mode, s, uin, upool), E, A, hard) for s in seeds])
        log(out, stage='correlations', pool='U', cond=cond)
    _atomic_json(out / 'evaluation' / 'correlations.json', res)
    return res


def gripper_threshold(rows, train_ids):
    """LIBERO-40 state-s6 threshold between closing and opening commands (theta_flow.evaluate rule)."""
    st = np.asarray(rows['state'][train_ids, 6]); a0 = np.asarray(rows['actions'][train_ids, 0, 6])
    return float((st[a0 < 0].mean() + st[a0 > 0].mean()) / 2)


def global_mean_action(rows, train_ids):
    return np.asarray(rows['actions'][train_ids, :, :7], np.float64).reshape(-1, 7).mean(0)


# ======================================================================== models
def query_inputs(d, mode):
    x = np.concatenate([np.asarray(d['hist'], np.float32).reshape(len(d['hist']), -1), np.asarray(d['state'], np.float32)], 1)
    if mode == 'P6': x = np.concatenate([np.asarray(d['mean'], np.float32), x], 1)
    elif mode == 'P0': x = np.asarray(d['mean'], np.float32).reshape(len(d['mean']), -1)
    elif mode != 'P5': raise ValueError(mode)
    for k in TARGET_KEYS:
        if k in d: raise ValueError(f'query inputs must not carry {k}')
    return x.astype(np.float32)


def key_inputs(pool_name, pool, cond):
    """Key = the type only (scale removed for S-*, rotation+scale removed for R-so3); gripper never included."""
    if pool_name == 'W':
        k = pool['canon'] if cond == 'R-so3' else type_of(pool['phys'], cond)
        return np.asarray(k, np.float32).reshape(len(k), -1)
    if cond == 'R-so3': v = np.asarray(pool['canon_values'], np.float64)
    else:
        v = np.asarray(pool['values'], np.float64).reshape(-1, KDCT, 6).copy()
        for g in groups_of(cond): v[:, :, g] /= np.maximum(rms_speed(pool['phys16'], g), 1e-9)[:, None, None]
        v = v.reshape(len(v), -1)
    return np.concatenate([v, np.log(pool['lengths'])[:, None] / 3], 1).astype(np.float32)


def make_net(pool_name, mode, in_dim, key_dim, d=256):
    import torch, torch.nn as nn
    F = nn.functional

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            if pool_name == 'U':      # unit_retrieval recipe
                self.query = nn.Sequential(nn.Linear(in_dim, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, d))
                init = 20.
            elif mode in ('P6', 'P0'):  # K3 'concat' (~ K1 P6); P0 = the same network on the VLM means only
                self.query = nn.Sequential(nn.Linear(in_dim, 768), nn.GELU(), nn.Linear(768, d)); init = 10.
            else:                     # K3 'hist' (= K1 P5)
                self.query = nn.Sequential(nn.Linear(in_dim, 512), nn.GELU(), nn.Linear(512, d)); init = 10.
            self.key = nn.Sequential(nn.Linear(key_dim, 512), nn.GELU(), nn.Linear(512, d))
            self.log_scale = nn.Parameter(torch.tensor(float(np.log(init))))

        def forward(self, x, kin):
            return F.normalize(self.query(x), dim=-1) @ F.normalize(self.key(kin), dim=-1).T * self.log_scale.exp()
    return Net()


def allowed_matrix(query_task, pool_task, keep, cross_task):
    a = np.broadcast_to(np.asarray(keep, bool), (len(query_task), len(pool_task))).copy()
    if cross_task: a &= np.asarray(pool_task)[None, :] != np.asarray(query_task)[:, None]
    if not a.any(1).all(): raise ValueError('a query has no allowed candidate')
    return a


def teacher_tau(E, A, frac):
    Em = np.where(A, E, np.inf); orc = Em.min(1); rnd = np.where(A, E, 0).sum(1) / A.sum(1)
    return float(frac * np.median(rnd - orc))


def train_one(pool_name, mode, cond, seed, frac, data, *, epochs, device):
    """One query model; returns holdout-best and Dev-best states.  Uses LIBERO-40 fit rows only for gradients."""
    import torch
    torch.manual_seed(seed)
    X, KI, E, A, fit, va = data['X'], data['KI'], data['E'], data['A'], data['fit'], data['va']
    tau = teacher_tau(E[fit], A[fit], frac)
    Xt = torch.as_tensor(X, device=device); Kt = torch.as_tensor(KI, device=device)
    Et = torch.as_tensor(np.where(A, E, 1e3).astype(np.float32), device=device); At = torch.as_tensor(A, device=device)
    Xd = torch.as_tensor(data['Xdev'], device=device); Ad = torch.as_tensor(data['Adev'], device=device)
    net = make_net(pool_name, mode, X.shape[1], KI.shape[1]).to(device)
    lr = 1e-3; opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-4)
    nb = (len(fit) + 255) // 256
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * nb, pct_start=.05) if pool_name == 'W' else None
    best = {'holdout': (np.inf, None, 0), 'dev': (np.inf, None, 0)}; curve = []

    def pick(xs, am):
        out = []
        with torch.no_grad():
            for i in range(0, len(xs), 2048):
                out.append(net(xs[i:i + 2048], Kt).masked_fill(~am[i:i + 2048], -1e9).argmax(-1).cpu().numpy())
        return np.concatenate(out)
    for ep in range(epochs):
        net.train(); perm = np.random.default_rng(seed + ep).permutation(fit)
        for i in range(0, len(perm), 256):
            b = torch.as_tensor(perm[i:i + 256], device=device)
            Eb = Et[b]; teacher = torch.softmax(-(Eb - Eb.min(1, keepdim=True).values) / tau, -1) * At[b]
            teacher = teacher / teacher.sum(1, keepdim=True)
            logit = net(Xt[b], Kt).masked_fill(~At[b], -1e9)
            loss = -(teacher * torch.log_softmax(logit, -1)).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if sched is not None: sched.step()
        net.eval()
        pv = pick(Xt[torch.as_tensor(va, device=device)], At[torch.as_tensor(va, device=device)])
        v = float(E[va, pv].mean())
        pd = pick(Xd, Ad); dv = float(data['Edev'][np.arange(len(pd)), pd].mean())
        curve.append({'epoch': ep + 1, 'holdout_top1_type_error': v, 'dev_top1_type_error': dv, 'loss': float(loss.detach())})
        state = None
        for key, val in (('holdout', v), ('dev', dv)):
            if val < best[key][0]:
                state = state or {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
                best[key] = (val, state, ep + 1)
    return {'tau': tau, 'frac': frac, 'curve': curve,
            'holdout': {'value': best['holdout'][0], 'epoch': best['holdout'][2], 'state': best['holdout'][1]},
            'dev': {'value': best['dev'][0], 'epoch': best['dev'][2], 'state': best['dev'][1]}}


def load_train_data(out, pool_name, cond, mode):
    P = Path(out) / 'prep'
    pool = dict(np.load(P / f'{pool_name}_pool.npz')); tr = dict(np.load(P / f'{pool_name}_inputs_train.npz'))
    dv = dict(np.load(P / f'{pool_name}_inputs_dev.npz'))
    E = np.load(P / f'{pool_name}_err_{cond}_train.npz')['err']; Edev = np.load(P / f'{pool_name}_err_{cond}_dev.npz')['err']
    A = allowed_matrix(tr['task'], pool['task'], pool['keep'], True)
    hold = np.isin(tr['task'], HOLDOUT)
    Adev = allowed_matrix(dv['task'], pool['task'], pool['keep'], False)   # unseen tasks never share a LIBERO-40 task
    return pool, {'X': query_inputs(tr, mode), 'KI': key_inputs(pool_name, pool, cond), 'E': E, 'A': A,
                  'fit': np.flatnonzero(~hold), 'va': np.flatnonzero(hold), 'Xdev': query_inputs(dv, mode), 'Adev': Adev, 'Edev': Edev}


def train(out, *, seeds=SEEDS, fracs=FRACS, epochs=None, device='cuda', configs=None):
    import torch
    out = Path(out); M = out / 'models'; M.mkdir(parents=True, exist_ok=True)
    plan = [(p, c, m) for p in ('W', 'U') for c in CONDS for m in modes_of(out)[p]] if configs is None else configs
    for pool_name, cond, mode in plan:
        pool, data = load_train_data(out, pool_name, cond, mode)
        for seed in seeds:
            for frac in fracs:
                path = M / f'{pool_name}_{cond}_{mode}_s{seed}_f{frac:.4f}.pt'
                if path.exists(): continue
                t0 = time.time()
                res = train_one(pool_name, mode, cond, seed, frac, data, epochs=epochs or EPOCHS[pool_name], device=device)
                res.update(pool=pool_name, cond=cond, mode=mode, seed=seed, epochs=epochs or EPOCHS[pool_name],
                           fit_rows=int(len(data['fit'])), holdout_rows=int(len(data['va'])), dev_rows=int(len(data['Xdev'])))
                tmp = path.with_suffix('.tmp'); torch.save(res, tmp); tmp.replace(path)
                log(out, stage='train', pool=pool_name, cond=cond, mode=mode, seed=seed, frac=round(frac, 4), tau=res['tau'],
                    holdout=res['holdout']['value'], holdout_epoch=res['holdout']['epoch'], dev=res['dev']['value'],
                    dev_epoch=res['dev']['epoch'], sec=round(time.time() - t0, 1))
        del data
    select(out, seeds=seeds, fracs=fracs, plan=plan)


def select(out, *, seeds, fracs, plan):
    """Choose teacher temperature (+ epoch) per seed on the LIBERO-40 holdout; Dev choice kept separately."""
    import torch
    out = Path(out); prev = out / 'selection.json'
    sel = json.loads(prev.read_text())['selected'] if prev.exists() else {}      # keep the other models' choices
    for pool_name, cond, mode in plan:
        for seed in seeds:
            runs = []
            for frac in fracs:
                r = torch.load(out / 'models' / f'{pool_name}_{cond}_{mode}_s{seed}_f{frac:.4f}.pt', weights_only=False)
                runs.append({'frac': frac, 'tau': r['tau'], 'holdout': r['holdout']['value'], 'holdout_epoch': r['holdout']['epoch'],
                             'dev': r['dev']['value'], 'dev_epoch': r['dev']['epoch']})
            h = min(runs, key=lambda x: (x['holdout'], x['frac'])); d = min(runs, key=lambda x: (x['dev'], x['frac']))
            sel[f'{pool_name}|{cond}|{mode}|{seed}'] = {'holdout': h, 'dev': d, 'runs': runs}
    _atomic_json(out / 'selection.json', {'selection': 'LIBERO-40 holdout tasks (top-1 type error within the condition); '
                                                       'Dev (LIBERO-90 tasks 6,7) choice reported separately; Eval never used',
                                          'holdout_tasks': list(HOLDOUT), 'selected': sel})


def score_models(out, pool_name, cond, mode, seed, split, device='cpu'):
    """Pool-local picks for dev/eval rows under the holdout- and Dev-selected states (no targets used)."""
    import torch
    out = Path(out); P = out / 'prep'; sel = json.loads((out / 'selection.json').read_text())['selected'][f'{pool_name}|{cond}|{mode}|{seed}']
    pool = dict(np.load(P / f'{pool_name}_pool.npz')); d = dict(np.load(P / f'{pool_name}_inputs_{split}.npz'))
    X = torch.as_tensor(query_inputs(d, mode)); KI = torch.as_tensor(key_inputs(pool_name, pool, cond))
    A = torch.as_tensor(allowed_matrix(d['task'], pool['task'], pool['keep'], False))
    picks = {}
    for which in ('holdout', 'dev'):
        r = torch.load(out / 'models' / f"{pool_name}_{cond}_{mode}_s{seed}_f{sel[which]['frac']:.4f}.pt", weights_only=False)
        net = make_net(pool_name, mode, X.shape[1], KI.shape[1]); net.load_state_dict(r[which]['state']); net.eval()
        with torch.no_grad(): picks[which] = net(X, KI).masked_fill(~A, -1e9).argmax(-1).numpy()
    return picks


# ======================================================================== generation (no targets)
def _check_no_targets(inp):
    bad = [k for k in inp if k in TARGET_KEYS]
    if bad: raise ValueError(f'generation inputs must not contain ground-truth keys: {bad}')


def gripper_g1(inp, thr, pool_name):
    """Hold the last executed gripper command; at an episode start threshold the state s6."""
    if pool_name == 'W':
        first = np.asarray(inp['frame']) == 0; last = np.asarray(inp['hist'])[:, -1, 6]
    else:
        first = np.asarray(inp['start']) == 0; last = np.asarray(inp['last'])[:, 6]
    return np.where(first, np.where(np.asarray(inp['state'])[:, 6] > thr, -1., 1.), last).astype(np.float32)


def generate_w(pool, pick, cond, rule, inp):
    """Normalised 6-axis actions [N,10,6] for picked W windows.  rule in stored/hist/pi05 (no ground truth)."""
    _check_no_targets(inp); _, vs, _ = _mods()
    u = np.asarray(pool['phys'], np.float64)[np.asarray(pick)]
    if rule == 'stored': out = u
    elif rule == 'hist':
        ref = vs.to_physical(np.asarray(inp['hist'])[:, :, :6]); g = 'S-sep' if cond == 'S-sep' else 'S-shared'
        out, _ = match_speed(u, ref, g); has = np.asarray(inp['has_hist'], bool)
        out = np.where(has[:, None, None], out, u)          # no history -> stored size
    elif rule == 'pi05':
        ref = vs.to_physical(np.asarray(inp['pi05'])[:, :, :6])
        if cond == 'R-so3': out = np.stack([rotate_scale(u[i], *procrustes_rs(u[i], ref[i])) for i in range(len(u))])
        else: out, _ = match_speed(u, ref, cond)
    else: raise ValueError(rule)
    return np.asarray(vs.to_normalized(out), np.float32)


def generate_u(pool, pick, cond, rule, inp):
    """List of normalised [Lu,6] actions for picked U units at their stored length (no ground truth)."""
    _check_no_targets(inp); _, vs, _ = _mods()
    res = []
    for i, k in enumerate(np.asarray(pick)):
        u = dct3_decode(pool['values'][k], int(pool['lengths'][k]))
        if rule == 'stored': o = u
        elif rule == 'hist':
            n = int(inp['n_past'][i])
            if n == 0: o = u
            else:
                ref = vs.to_physical(np.asarray(inp['past'][i])[10 - n:, :6]); g = 'S-sep' if cond == 'S-sep' else 'S-shared'
                o, _ = match_speed(u[None], ref[None], g); o = o[0]
        elif rule == 'pi05':
            m = int(inp['n_pi05'][i]); ref = vs.to_physical(np.asarray(inp['pi05_now'][i])[:m, :6])
            if cond == 'R-so3':
                j = min(m, len(u)); o = rotate_scale(u, *procrustes_rs(u[:j], ref[:j]))
            else:
                o, _ = match_speed(u[None], ref[None], cond); o = o[0]
        else: raise ValueError(rule)
        res.append(np.asarray(vs.to_normalized(o), np.float32))
    return res


# ======================================================================== scoring (ground truth only here)
def six_axis(pred, y):
    """Six-axis MSE as in theta_flow.six_axis_mse: mean over frames of the 6-axis squared error sum / 7."""
    return ((np.asarray(pred)[..., :6] - np.asarray(y)[..., :6]) ** 2).sum(-1).mean(-1) / 7


def grip_part(g, y):
    return ((np.asarray(g)[:, None] - np.asarray(y)[..., 6]) ** 2).mean(-1) / 7


def oracle_w(pool, pick, cond, y_norm):
    """Upper bound: S-* closed-form s* to the target; R-so3 theta* (Procrustes onto the target)."""
    _, vs, _ = _mods(); u = np.asarray(pool['phys'], np.float64)[np.asarray(pick)]; y = vs.to_physical(np.asarray(y_norm)[:, :, :6])
    if cond == 'R-so3': out = np.stack([rotate_scale(u[i], *procrustes_rs(u[i], y[i])) for i in range(len(u))])
    else: out = apply_scales(u, closed_form_scale(u, y, cond), cond)
    return np.asarray(vs.to_normalized(out), np.float32)


def oracle_u(pool, pick, cond, targets):
    _, vs, _ = _mods(); res = []
    for i, k in enumerate(np.asarray(pick)):
        u = dct3_decode(pool['values'][k], int(pool['lengths'][k])); y = vs.to_physical(targets[i][:, :6]); j = min(len(u), len(y))
        if cond == 'R-so3': o = rotate_scale(u, *procrustes_rs(u[:j], y[:j]))
        else: o = apply_scales(u, closed_form_scale(u[None, :j], y[None, :j], cond)[0], cond)
        res.append(np.asarray(vs.to_normalized(o), np.float32))
    return res


def u_errors(preds, targets, grip):
    """Per segment: full (all true frames; a shorter unit holds its last frame, a longer one is cut) and
    overlap (only frames covered by both) six-axis MSE, plus the G1 gripper part on the same frames."""
    out = {k: np.zeros(len(preds)) for k in ('full', 'overlap', 'grip_full', 'grip_overlap', 'n_overlap')}
    for i, (p, y) in enumerate(zip(preds, targets)):
        Lq = len(y); j = min(len(p), Lq); ext = p[np.minimum(np.arange(Lq), len(p) - 1)]
        out['full'][i] = six_axis(ext, y); out['overlap'][i] = six_axis(p[:j], y[:j]); out['n_overlap'][i] = j
        out['grip_full'][i] = ((grip[i] - y[:, 6]) ** 2).mean() / 7; out['grip_overlap'][i] = ((grip[i] - y[:j, 6]) ** 2).mean() / 7
    return out


def u_overlap_baseline(const_or_seq, targets, n_overlap):
    """A baseline's six-axis MSE on exactly the frames a method was scored on (first n frames)."""
    return np.array([six_axis(const_or_seq[i][:int(n)], targets[i][:int(n)]) for i, n in enumerate(n_overlap)])


def type_metrics(E, pick, allowed):
    Em = np.where(allowed, E, np.inf); orc = Em.min(1); rnd = np.where(allowed, E, 0).sum(1) / allowed.sum(1)
    ch = E[np.arange(len(pick)), pick]
    return {'type_error': float(ch.mean()), 'gap_recovery': float((rnd - ch).mean() / max((rnd - orc).mean(), EPS)),
            'strict_top1': float((ch <= orc + .1 * (rnd - orc)).mean())}


def boot(v, episode):
    _, _, kq = _mods(); return kq.episode_bootstrap(v, episode, reps=5000, seed=BOOT_SEED)


def boot_diff(a, b, episode):
    cd, _, _ = _mods(); return cd._bootstrap_diff(a, b, episode, reps=5000, seed=BOOT_SEED)


def evaluate(out, *, seeds=SEEDS, device='cpu'):
    """Score frozen picks on unseen LIBERO-90 Eval (and Dev for reference).  Eval targets are read here only."""
    cd, vs, kq = _mods(); out = Path(out); P = out / 'prep'; meta = json.loads((P / 'prep.json').read_text())
    thr = meta['gripper_threshold']; gmean = np.asarray(meta['global_mean_action'], np.float32)
    ua = np.load(kq.UNSEEN / 'actions.npy', mmap_mode='r', allow_pickle=False)          # scoring only
    rows = {}; summary = {'W': {}, 'U': {}}; rng = np.random.default_rng(BOOT_SEED); MD = modes_of(out)
    for split in ('eval', 'dev'):
        # ---------------------------------------------------------------- W
        pool = dict(np.load(P / 'W_pool.npz')); inp = dict(np.load(P / f'W_inputs_{split}.npz'))
        y = np.asarray(ua[inp['row'], :, :7], np.float32); n = len(y)
        ep = inp['task'].astype(np.int64) * 100000 + inp['episode'].astype(np.int64)
        hard = inp['hard'].astype(bool) if 'hard' in inp else np.zeros(n, bool)
        A = allowed_matrix(inp['task'], pool['task'], pool['keep'], False)
        g1 = gripper_g1(inp, thr, 'W'); gp = grip_part(g1, y)
        base = {'keep_prev': np.repeat(inp['hist'][:, -1:, :], 10, 1), 'zero': np.zeros_like(y),
                'global_mean': np.broadcast_to(gmean, y.shape), 'pi05': inp['pi05']}
        R = {}
        for name, pred in base.items():
            R[f'W|base|{name}'] = six_axis(pred, y)
        R['W|base|keep_prev|active7'] = six_axis(base['keep_prev'], y) + grip_part(base['keep_prev'][:, 0, 6], y)
        R['W|base|pi05|active7'] = ((inp['pi05'] - y) ** 2).mean((1, 2))
        rnd = np.stack([rng.choice(np.flatnonzero(A[i]), 5) for i in range(n)])
        yphys = vs.to_physical(y[:, :, :6])
        types = {}
        for cond in CONDS:
            E = type_errors(yphys.astype(np.float32), pool['phys'], cond, device); orc = np.where(A, E, np.inf).argmin(1)
            Epi = type_errors(vs.to_physical(np.asarray(inp['pi05'])[:, :, :6]).astype(np.float32), pool['phys'], cond, device)
            picks = {'random': None, 'oracle_type': orc, 'pi05_nn': np.where(A, Epi, np.inf).argmin(1)}   # no target
            for mode in MD['W']:
                for seed in seeds:
                    pk = score_models(out, 'W', cond, mode, seed, split)
                    for which in ('holdout', 'dev'):
                        picks[f'{mode}|s{seed}|{which}'] = pk[which]
                        types[f'W|{cond}|{mode}|s{seed}|{which}'] = type_metrics(E, pk[which], A)
            for key, pk in picks.items():
                for rule in RULES:
                    if key == 'random':
                        v = np.mean([_w_rule(pool, rnd[:, j], cond, rule, inp, y) for j in range(5)], 0)
                    else: v = _w_rule(pool, pk, cond, rule, inp, y)
                    R[f'W|{cond}|{key}|{rule}'] = v; R[f'W|{cond}|{key}|{rule}|active7'] = v + gp
        rows[f'W|{split}'] = {'values': R, 'episode': ep, 'hard': hard}
        summary['W'][split] = {'rows': n, 'hard_rows': int(hard.sum()), 'G1_gripper_part': float(gp.mean()), 'types': types}
        log(out, stage='evaluate', pool='W', split=split, rows=n)
        # ---------------------------------------------------------------- U
        upool = dict(np.load(P / 'U_pool.npz')); uin = dict(np.load(P / f'U_inputs_{split}.npz'))
        uids = np.asarray(np.load(P / f'W_inputs_{split}.npz')['row'])
        chk, ut_all = build_segments(ua, *(np.asarray(np.load(kq.UNSEEN / f'{k}.npy', mmap_mode='r')) for k in ('task', 'episode', 'frame')),
                                     np.load(kq.UNSEEN / 'state.npy', mmap_mode='r'), uids)
        if not (np.array_equal(chk['start'], uin['start']) and np.array_equal(chk['episode'], uin['episode']) and np.array_equal(chk['task'], uin['task'])):
            raise AssertionError('U scoring targets misaligned')
        tg = [_unflatten(ut_all['target_flat'], ut_all['target_off'], i) for i in range(len(uin['task']))]
        n = len(tg); ep = uin['task'].astype(np.int64) * 100000 + uin['episode'].astype(np.int64); hard = uin['hard'].astype(bool)
        A = allowed_matrix(uin['task'], upool['task'], upool['keep'], False)
        g1 = gripper_g1(uin, thr, 'U')
        pi_span = [_unflatten(uin['pi05_span_flat'], uin['pi05_span_off'], i) for i in range(n)]
        consts = {'keep_prev': [np.repeat(uin['last'][i][None], len(tg[i]), 0) for i in range(n)],
                  'zero': [np.zeros((len(t), 7), np.float32) for t in tg], 'global_mean': [np.repeat(gmean[None], len(t), 0) for t in tg],
                  'pi05': pi_span}
        R = {}
        for name, seq in consts.items(): R[f'U|base|{name}'] = np.array([six_axis(s, t) for s, t in zip(seq, tg)])
        R['U|base|keep_prev|active7'] = R['U|base|keep_prev'] + np.array([((uin['last'][i][6] - t[:, 6]) ** 2).mean() / 7 for i, t in enumerate(tg)])
        R['U|base|pi05|active7'] = np.array([((s - t) ** 2).mean() for s, t in zip(pi_span, tg)])
        rnd = np.stack([rng.choice(np.flatnonzero(A[i]), 5) for i in range(n)])
        q16 = ut_all['target16']; types = {}
        for cond in CONDS:
            E = type_errors(q16, upool['phys16'], cond, 'cuda' if device == 'cuda' else 'cpu'); orc = np.where(A, E, np.inf).argmin(1)
            picks = {'random': None, 'oracle_type': orc}
            for seed in seeds:
                pk = score_models(out, 'U', cond, 'P5', seed, split)
                for which in ('holdout', 'dev'):
                    picks[f'P5|s{seed}|{which}'] = pk[which]; types[f'U|{cond}|P5|s{seed}|{which}'] = type_metrics(E, pk[which], A)
            for key, pk in picks.items():
                for rule in RULES:
                    draws = [rnd[:, j] for j in range(5)] if key == 'random' else [pk]
                    acc = None
                    for d in draws:
                        preds = oracle_u(upool, d, cond, tg) if rule == 'oracle' else generate_u(upool, d, cond, rule, uin)
                        e = u_errors(preds, tg, g1)
                        e['keep_overlap'] = u_overlap_baseline(consts['keep_prev'], tg, e['n_overlap'])
                        acc = e if acc is None else {k: acc[k] + e[k] for k in acc}
                    e = {k: v / len(draws) for k, v in acc.items()}
                    R[f'U|{cond}|{key}|{rule}'] = e['full']; R[f'U|{cond}|{key}|{rule}|active7'] = e['full'] + e['grip_full']
                    R[f'U|{cond}|{key}|{rule}|overlap'] = e['overlap']; R[f'U|{cond}|{key}|{rule}|overlap_keep'] = e['keep_overlap']
                    R[f'U|{cond}|{key}|{rule}|n_overlap'] = e['n_overlap']
        rows[f'U|{split}'] = {'values': R, 'episode': ep, 'hard': hard}
        summary['U'][split] = {'rows': n, 'hard_rows': int(hard.sum()), 'types': types,
                               'segment_length_median': float(np.median([len(t) for t in tg]))}
        log(out, stage='evaluate', pool='U', split=split, rows=n)
    E_dir = out / 'evaluation'; E_dir.mkdir(parents=True, exist_ok=True)
    flat = {}
    for tag, d in rows.items():
        flat[f'{tag}|episode'] = d['episode']; flat[f'{tag}|hard'] = d['hard']
        for k, v in d['values'].items(): flat[f'{tag}||{k}'] = np.asarray(v, np.float64)
    _savez(E_dir / 'rows.npz', **flat)
    summary.update(seeds=list(seeds), eval_targets_used_only_for_scoring=True, gripper_threshold=thr)
    _atomic_json(E_dir / 'summary.json', summary)


def _w_rule(pool, pick, cond, rule, inp, y):
    pred = oracle_w(pool, pick, cond, y) if rule == 'oracle' else generate_w(pool, pick, cond, rule, inp)
    return six_axis(pred, y)


# ======================================================================== report
COND_JA = {'S-shared': 'S-shared（6軸で1つの大きさ）', 'S-sep': 'S-sep（位置と回転で別の大きさ）', 'R-so3': 'R-so3（対照：回転＋大きさ）'}
RULE_JA = {'stored': '保存のまま', 'hist': '履歴の速さ', 'pi05': 'π0.5 の大きさ', 'oracle': 'oracle s*'}
RULE_R = {'stored': 'θ_src', 'hist': '履歴の速さ', 'pi05': 'θπ0.5', 'oracle': 'θ*'}


def aggregate(out, *, seeds=SEEDS):
    out = Path(out); z = np.load(out / 'evaluation' / 'rows.npz'); summ = json.loads((out / 'evaluation' / 'summary.json').read_text())
    res = {'tables': {}, 'ci': {}, 'types': {}}; MD = modes_of(out)

    def get(tag, key): return z[f'{tag}||{key}']
    for pool_name in ('W', 'U'):
        for split in ('eval', 'dev'):
            tag = f'{pool_name}|{split}'; ep = z[f'{tag}|episode']; hard = z[f'{tag}|hard'].astype(bool)
            T = {}

            def stat(vs_):
                alls = [float(v.mean()) for v in vs_]; hs = [float(v[hard].mean()) for v in vs_] if hard.any() else [None]
                m = lambda a: None if a[0] is None else [float(np.mean(a)), float(np.min(a)), float(np.max(a))]
                return {'all': m(alls), 'hard': m(hs)}
            for b in ('keep_prev', 'zero', 'global_mean', 'pi05'):
                T[f'base|{b}'] = stat([get(tag, f'{pool_name}|base|{b}')])
            for b in ('keep_prev', 'pi05'):
                T[f'base|{b}|active7'] = stat([get(tag, f'{pool_name}|base|{b}|active7')])
            for cond in CONDS:
                for rule in RULES:
                    for suffix in ('', '|active7') + (('|overlap', '|overlap_keep') if pool_name == 'U' else ()):
                        for key in ('random', 'oracle_type', 'pi05_nn'):
                            if f'{tag}||{pool_name}|{cond}|{key}|{rule}{suffix}' not in z: continue   # pi05_nn: W only, newer runs
                            T[f'{cond}|{key}|{rule}{suffix}'] = stat([get(tag, f'{pool_name}|{cond}|{key}|{rule}{suffix}')])
                        for mode in MD[pool_name]:
                            for which in ('holdout', 'dev'):
                                T[f'{cond}|{mode}|{which}|{rule}{suffix}'] = stat(
                                    [get(tag, f'{pool_name}|{cond}|{mode}|s{s}|{which}|{rule}{suffix}') for s in seeds])
            res['tables'][tag] = T
            if split != 'eval': continue
            C = {}
            mean_seeds = lambda cond, mode, rule, which='holdout', sfx='': np.mean(
                [get(tag, f'{pool_name}|{cond}|{mode}|s{s}|{which}|{rule}{sfx}') for s in seeds], 0)
            keep = get(tag, f'{pool_name}|base|keep_prev')
            for mode in MD[pool_name]:
                for rule in RULES:
                    r = mean_seeds('R-so3', mode, rule)
                    for cond in CONDS:
                        a = mean_seeds(cond, mode, rule)
                        if cond != 'R-so3':
                            C[f'{mode}|{rule}|{cond}-R-so3'] = {'all': boot_diff(a, r, ep), 'hard': boot_diff(a[hard], r[hard], ep[hard]) if hard.any() else None}
                        C[f'{mode}|{rule}|{cond}-keep_prev'] = {'all': boot_diff(a, keep, ep), 'hard': boot_diff(a[hard], keep[hard], ep[hard]) if hard.any() else None}
                        if pool_name == 'U':
                            ao = mean_seeds(cond, mode, rule, sfx='|overlap'); ko = mean_seeds(cond, mode, rule, sfx='|overlap_keep')
                            C[f'{mode}|{rule}|{cond}-keep_prev|overlap'] = {'all': boot_diff(ao, ko, ep)}
                        C[f'{mode}|{rule}|{cond}|abs'] = boot(a, ep)
            C['pi05-keep_prev'] = {'all': boot_diff(get(tag, f'{pool_name}|base|pi05'), keep, ep)}
            res['ci'][tag] = C
        ty = {}
        for key, v in summ[pool_name]['eval']['types'].items():
            _, cond, mode, seed, which = key.split('|'); ty.setdefault(f'{cond}|{mode}|{which}', []).append(v)
        res['types'][pool_name] = {k: {m: [float(np.mean([x[m] for x in v])), float(np.min([x[m] for x in v])), float(np.max([x[m] for x in v]))]
                                       for m in v[0]} for k, v in ty.items()}
    res['summary'] = summ
    _atomic_json(out / 'results.json', res)
    return res


def _f(x, d=3):
    return '—' if x is None else f'{x:.{d}f}'.lstrip('0').replace('-0.', '-.') if abs(x) < 1 else f'{x:.{d}f}'


def _cell(s, rng=True):
    if s is None or s['all'] is None: return '—'
    a, h = s['all'], s['hard']
    txt = f"{_f(a[0])}" + (f" ({_f(a[1])}〜{_f(a[2])})" if rng and a[2] - a[1] > 5e-4 else '')
    if h is not None: txt += f" / {_f(h[0])}" + (f" ({_f(h[1])}〜{_f(h[2])})" if rng and h[2] - h[1] > 5e-4 else '')
    return txt


def _ci(c):
    if c is None: return '—'
    return f"{c['mean']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]"


def report(out, *, seeds=SEEDS, title_suffix=''):
    out = Path(out); res = aggregate(out, seeds=seeds); prep = json.loads((out / 'prep' / 'prep.json').read_text())
    sel = json.loads((out / 'selection.json').read_text())['selected']; MODES = modes_of(out)
    L = [f'# S1：型を「大きさだけ正規化」にした運動メモリの検索{title_suffix}', '',
         '主指標＝未学習 LIBERO-90 Eval の最終 6 軸行動 MSE（正規化行動、theta_flow.six_axis_mse と同じ定義）。'
         '各セルは「全体 / 難しい行」、学習クエリは 3 seed の平均（括弧は seed 間の最小〜最大）。'
         '型誤差は条件ごとに定義が違うので条件間比較には使わない。', '',
         f"W：窓 {res['summary']['W']['eval']['rows']}（難 {res['summary']['W']['eval']['hard_rows']}）、候補 {prep['W_pool']}。"
         f"U：区間 {res['summary']['U']['eval']['rows']}（難 {res['summary']['U']['eval']['hard_rows']}＝区間の始まりを含む窓が難しい行）、候補 {prep['U_pool_kept']}。"
         f"G1 のしきい値 {prep['gripper_threshold']:.3f}。選択は LIBERO-40 保留8タスク、Eval は選択に使わない。", '']
    for pool_name in ('W', 'U'):
        T = res['tables'][f'{pool_name}|eval']
        L += [f'## {pool_name}：6 軸 MSE（保留タスクで選択）' + ('（全区間：短い単位は最後の値を保持、長い単位は打ち切り）' if pool_name == 'U' else ''), '',
              '| 条件 | 型の選び方 | 保存のまま（R: θ_src） | 履歴の速さ | π0.5 の大きさ（R: θπ0.5） | oracle s*（R: θ*） |', '|---|---|---:|---:|---:|---:|']
        for cond in CONDS:
            for key, lab in [(f'{m}|holdout', f'学習クエリ {m}') for m in MODES[pool_name]] + [('pi05_nn', 'π0.5 出力の最近傍（学習なし）'), ('random', 'random 型（5回平均）'), ('oracle_type', 'oracle 型')]:
                if f'{cond}|{key}|stored' not in T: continue
                L.append(f'| {cond} | {lab} | ' + ' | '.join(_cell(T[f'{cond}|{key}|{r}']) for r in RULES) + ' |')
        L += ['', '| 基準 | 6 軸 MSE（全体 / 難しい行） |', '|---|---:|']
        for b, lab in (('keep_prev', '直前の行動を保つ'), ('zero', 'ゼロ（正規化 0）'), ('global_mean', '全体平均（LIBERO-40）'), ('pi05', 'π0.5')):
            L.append(f'| {lab} | {_cell(T["base|" + b], False)} |')
        L += ['', f'### {pool_name}：差の 95% 区間（episode bootstrap、seed 平均の行ごとの値で対応あり）', '',
              '| クエリ | 大きさ | S-shared − R-so3 | S-sep − R-so3 | S-shared − 直前保持 | S-sep − 直前保持 | R-so3 − 直前保持 |', '|---|---|---:|---:|---:|---:|---:|']
        C = res['ci'][f'{pool_name}|eval']
        for mode in MODES[pool_name]:
            for rule in RULES:
                L.append(f'| {mode} | {RULE_JA[rule]} | ' + ' | '.join(_ci(C[f'{mode}|{rule}|{c}-R-so3']['all']) for c in ('S-shared', 'S-sep')) + ' | '
                         + ' | '.join(_ci(C[f'{mode}|{rule}|{c}-keep_prev']['all']) for c in CONDS) + ' |')
        L.append(f"\nπ0.5 − 直前保持：{_ci(C['pi05-keep_prev']['all'])}。難しい行の区間は results.json の ci。")
        if pool_name == 'U':
            L += ['', '### U：重なったフレームだけで採点（仕様どおり）。直前保持も同じフレームで計算', '',
                  '| 条件 | 大きさ | 学習クエリ P5 | 直前保持（同じフレーム） | 差 [95%] |', '|---|---|---:|---:|---:|']
            for cond in CONDS:
                for rule in RULES:
                    L.append(f"| {cond} | {RULE_JA[rule]} | {_cell(T[f'{cond}|P5|holdout|{rule}|overlap'])} | {_cell(T[f'{cond}|P5|holdout|{rule}|overlap_keep'])} | "
                             f"{_ci(C[f'P5|{rule}|{cond}-keep_prev|overlap']['all'])} |")
        L += ['', f'### {pool_name}：G1 込み active7 MSE（副指標）', '', '| 条件 | クエリ | ' + ' | '.join(RULE_JA[r] for r in RULES) + ' |', '|---|---|' + '---:|' * 4]
        for cond in CONDS:
            for m in MODES[pool_name]:
                L.append(f'| {cond} | {m} | ' + ' | '.join(_cell(T[f'{cond}|{m}|holdout|{r}|active7']) for r in RULES) + ' |')
        L.append(f"| 基準 | 直前保持 / π0.5 | {_cell(T['base|keep_prev|active7'], False)} | {_cell(T['base|pi05|active7'], False)} | | |")
        L += ['', f'### {pool_name}：Dev で選んだ場合（Eval の 6 軸 MSE）', '', '| 条件 | クエリ | ' + ' | '.join(RULE_JA[r] for r in RULES) + ' |', '|---|---|' + '---:|' * 4]
        for cond in CONDS:
            for m in MODES[pool_name]:
                L.append(f'| {cond} | {m} | ' + ' | '.join(_cell(T[f'{cond}|{m}|dev|{r}']) for r in RULES) + ' |')
        D = res['tables'][f'{pool_name}|dev']
        L += ['', f'### {pool_name}：Dev（LIBERO-90 タスク 6, 7）自体の 6 軸 MSE（保留タスクで選択、参考）', '',
              '| 条件 | クエリ | ' + ' | '.join(RULE_JA[r] for r in RULES) + ' |', '|---|---|' + '---:|' * 4]
        for cond in CONDS:
            for m in MODES[pool_name]:
                L.append(f'| {cond} | {m} | ' + ' | '.join(_cell(D[f'{cond}|{m}|holdout|{r}']) for r in RULES) + ' |')
        L.append(f"| 基準 | 直前保持 / π0.5 | {_cell(D['base|keep_prev'], False)} | {_cell(D['base|pi05'], False)} | | |")
        L += ['', f'### {pool_name}：型選択の参考指標（各条件の中だけで意味がある）', '', '| 条件 | クエリ | 選択 | 差の回収率 | 厳しい top1 |', '|---|---|---|---:|---:|']
        for k, v in sorted(res['types'][pool_name].items()):
            cond, mode, which = k.split('|')
            L.append(f"| {cond} | {mode} | {'保留' if which == 'holdout' else 'Dev'} | {100 * v['gap_recovery'][0]:.0f}% ({100 * v['gap_recovery'][1]:.0f}〜{100 * v['gap_recovery'][2]:.0f}) | "
                     f"{100 * v['strict_top1'][0]:.0f}% ({100 * v['strict_top1'][1]:.0f}〜{100 * v['strict_top1'][2]:.0f}) |")
        L.append('')
    if (out / 'evaluation' / 'correlations.json').exists():
        C = json.loads((out / 'evaluation' / 'correlations.json').read_text())
        L += ['## スコアと型誤差の相関（Eval、クエリごとの順位相関の平均、全体 / 難しい行）', '',
              'スコア＝学習クエリの出力（保留タスクで選んだ状態、3 seed 平均）、π0.5 最近傍は −（π0.5 出力との型誤差）。'
              '相手は −（正解との型誤差）。候補は保留タスクの単位を除いた全部。', '',
              '| プール | 条件 | クエリ | Spearman | Pearson |', '|---|---|---|---:|---:|']
        for k, v in C.items():
            pn, cond, q = k.split('|')
            L.append(f"| {pn} | {cond} | {q} | {_f(v['spearman'][0])} / {_f(v['spearman_hard'][0])} | {_f(v['pearson'][0])} / {_f(v['pearson_hard'][0])} |")
        L.append('')
    if prep.get('note'): L += ['## この実行について', '', prep['note'], '']
    L += ['## 選択された設定（保留タスク）', '', '| 組 | 目標の鋭さ | エポック | 保留の top1 型誤差 | Dev 選択の鋭さ / エポック |', '|---|---:|---:|---:|---:|']
    for k, v in sel.items():
        L.append(f"| {k.replace('|', ' / ')} | 1/{round(1 / v['holdout']['frac'])} | {v['holdout']['holdout_epoch']} | {v['holdout']['holdout']:.5f} | 1/{round(1 / v['dev']['frac'])} / {v['dev']['dev_epoch']} |")
    L += ['', '## 確認', '',
          f"- W の R-so3 教師誤差と K3 の誤差の差（先頭200行の最大）：{prep['check_W_Rso3_vs_K3_maxabs']:.2e}",
          f"- U の R-so3 教師誤差と unit_retrieval の誤差の差（先頭200行の最大）：{prep['check_U_Rso3_vs_unit_retrieval_maxabs']:.2e}",
          '- 単位・区間は unit_retrieval と一致を確認（prepare の assert）。Eval の正解行動は evaluate の採点でのみ読む。']
    (out / 'report.md').write_text('\n'.join(L) + '\n')
    return res


# ======================================================================== entry
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('prepare', 'refresh', 'train', 'evaluate', 'correlations', 'report', 'all'))
    ap.add_argument('--src', default=None, help='refresh: finished run to copy the LIBERO-40 side from')
    ap.add_argument('--fix', default=None, help='refresh: orientation_fix.py output (pi05/, features/)')
    ap.add_argument('--modes', default=None, help='refresh/train: W query modes, e.g. P0,P5,P6 (refresh) or P0 (train only these)')
    ap.add_argument('--smoke', action='store_true'); ap.add_argument('--out', default=None)
    ap.add_argument('--device', default='cuda'); ap.add_argument('--threads', type=int, default=8)
    a = ap.parse_args()
    import torch; torch.set_num_threads(a.threads)
    if a.device.startswith('cuda'): torch.cuda.set_per_process_memory_fraction(.2)
    out = Path(a.out) if a.out else OUT / ('smoke' if a.smoke else 'production'); out.mkdir(parents=True, exist_ok=True)
    kw = dict(seeds=(SEEDS[0],), fracs=(1 / 10,), epochs=4) if a.smoke else {}
    seeds = kw.get('seeds', SEEDS)
    if a.stage == 'refresh':
        modes = {'W': tuple(a.modes.split(',')), 'U': MODES['U']} if a.modes else None
        refresh(out, Path(a.src), Path(a.fix), modes=modes); return
    if a.stage in ('prepare', 'all'): prepare(out, limit=3000 if a.smoke else None, device=a.device)
    if a.stage == 'train' and a.modes:
        train(out, device=a.device, configs=[('W', c, m) for m in a.modes.split(',') for c in CONDS], **kw)
    elif a.stage in ('train', 'all'): train(out, device=a.device, **kw)
    if a.stage in ('evaluate', 'all'): evaluate(out, seeds=seeds, device=a.device)
    if a.stage in ('correlations', 'all'): correlations(out, seeds=seeds, device=a.device)
    if a.stage in ('report', 'all'):
        report(out, seeds=seeds, title_suffix='（smoke：少数行・1 seed・4 エポック）' if a.smoke else '')
        log(out, stage='report', done=True)


if __name__ == '__main__':
    main()
