#!/usr/bin/env python3
"""LIBERO audit for separating gripper commands from the six motion axes.

This is an offline upper-bound diagnostic, not a retrieval policy: evaluation targets may
select an oracle candidate in section B, and are used for scoring only in section D.
Existing pipeline files are imported read-only.  Run with ``python gripper_split.py``.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.fft import dct, idct

HERE = Path(__file__).resolve().parent
ROOT = Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero')
RUN = Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/run-01')
PREP = ROOT / 'scale-only-02/production/prep'
OUT = ROOT / 'gripper-split-01'
sys.path.insert(0, str(HERE))


def action_parts_mse(pred, target):
    """Return motion and grip contributions; they sum to active7 MSE exactly."""
    p, t = np.asarray(pred)[..., :7], np.asarray(target)[..., :7]
    arm = np.square(p[..., :6] - t[..., :6]).sum(axis=(-1, -2)) / (p.shape[-2] * 7)
    grip = np.square(p[..., 6] - t[..., 6]).sum(axis=-1) / (p.shape[-2] * 7)
    return arm, grip, arm + grip


def dct_roundtrip(x, k=3):
    x = np.asarray(x, np.float64)
    c = dct(x, type=2, norm='ortho', axis=-2)
    c[..., min(k, x.shape[-2]):, :] = 0
    return idct(c, type=2, norm='ortho', axis=-2)


def dct_unit(x, k=3):
    x = np.asarray(x, np.float64)
    c = dct(x, type=2, norm='ortho', axis=0)
    c[min(k, len(x)):] = 0
    return idct(c, type=2, norm='ortho', axis=0)


def nearest_candidates(target, candidates, axes):
    """Per-query nearest candidate by normalized action MSE over selected axes."""
    y = np.asarray(target, np.float64)[..., axes].reshape(len(target), -1)
    x = np.asarray(candidates, np.float64)[..., axes].reshape(len(candidates), -1)
    # Squared-distance expansion avoids materializing [Q,K,T,D].
    d = (np.square(y).sum(1)[:, None] + np.square(x).sum(1)[None] - 2 * y @ x.T)
    return np.argmin(d, axis=1), np.maximum(d.min(1), 0) / y.shape[1]


def bootstrap_mean(values, episodes, reps=5000, seed=4101):
    values, episodes = np.asarray(values, float), np.asarray(episodes)
    unique = np.unique(episodes)
    by = {e: values[episodes == e] for e in unique}
    rng = np.random.default_rng(seed)
    means = np.empty(reps)
    for i in range(reps):
        draw = rng.choice(unique, len(unique), replace=True)
        means[i] = np.concatenate([by[e] for e in draw]).mean()
    return [float(values.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def paired_episode_bootstrap(values, episodes, reps=5000, seed=4102):
    values, episodes = np.asarray(values, float), np.asarray(episodes)
    unique = np.unique(episodes); by = {e: values[episodes == e] for e in unique}
    rng = np.random.default_rng(seed); means = np.empty(reps)
    for i in range(reps):
        draw = rng.choice(unique, len(unique), replace=True)
        means[i] = np.concatenate([by[e] for e in draw]).mean()
    return {'mean': float(values.mean()), 'ci95': [float(np.quantile(means, .025)), float(np.quantile(means, .975))]}


def episode_codes(task, episode):
    """Stable bootstrap cluster IDs from (task, episode), never episode alone."""
    pairs = list(zip(np.asarray(task).astype(np.int64).tolist(),
                     np.asarray(episode).astype(np.int64).tolist(), strict=True))
    lookup = {key: i for i, key in enumerate(dict.fromkeys(pairs))}
    return np.asarray([lookup[key] for key in pairs], np.int64)


def exclude_episode_rows(candidate_ids, task, episode, heldout_ids):
    """Remove every candidate row belonging to any held-out (task, episode)."""
    held = {(int(task[i]), int(episode[i])) for i in np.asarray(heldout_ids, np.int64)}
    return np.asarray([i for i in np.asarray(candidate_ids, np.int64)
                       if (int(task[i]), int(episode[i])) not in held], np.int64)


def memory_external_nonvalidation_rows(rows):
    """Return non-validation rows only from whole demos absent from memory."""
    task, episode = np.asarray(rows['task']), np.asarray(rows['episode'])
    memory, validation = np.asarray(rows['memory'], bool), np.asarray(rows['validation'], bool)
    memory_demos = {(int(task[i]), int(episode[i])) for i in np.flatnonzero(memory)}
    candidates = np.flatnonzero(~validation)
    return np.asarray([i for i in candidates
                       if (int(task[i]), int(episode[i])) not in memory_demos], np.int64)


def variable_dct3_parts(raw_unit):
    """Self-reconstruction MSE contributions for DCT3 arm + stored majority grip."""
    raw = np.asarray(raw_unit, np.float64)
    arm = dct_unit(raw[:, :6], 3)
    grip = np.full(len(raw), 1. if np.mean(raw[:, 6]) >= 0 else -1., np.float64)
    arm_mse = float(np.square(arm - raw[:, :6]).sum() / (len(raw) * 7))
    grip_mse = float(np.square(grip - raw[:, 6]).sum() / (len(raw) * 7))
    return arm_mse, grip_mse, arm_mse + grip_mse


def top_error_mask(errors, fraction=.2):
    """Mark the upper error fraction within one evaluation set."""
    e = np.asarray(errors, np.float64)
    if e.ndim != 1 or not len(e) or not 0 < fraction < 1:
        raise ValueError((e.shape, fraction))
    n = max(1, int(np.ceil(len(e) * fraction)))
    order = np.argsort(e, kind='stable')
    mask = np.zeros(len(e), bool)
    mask[order[-n:]] = True
    return mask


def majority_gripper_sign(values):
    """Map a unit's mean normalized gripper command to its stored majority sign.

    Ties use +1, matching ``_variable_pool``'s ``mean >= 0`` encoding.  Using
    ``np.sign`` here would emit an invalid zero command for exactly balanced units.
    """
    return np.where(np.asarray(values) >= 0, 1., -1.).astype(np.float32)


def tau_neighbor_matrix(coefficients, *, include_gripper, tau=.08):
    """CMP2-style coefficient-space neighborhood, normalized by active axes."""
    c = np.asarray(coefficients, np.float64)
    if c.ndim != 2 or c.shape[1] < 2:
        raise ValueError(f'expected [units, coefficients], got {c.shape}')
    m = c if include_gripper else c[:, :-1]
    axes = 7 if include_gripper else 6
    within = np.empty((len(m), len(m)), bool)
    for i in range(len(m)):
        delta = m - m[i]
        within[i] = np.sum(delta * delta, axis=1) / axes <= float(tau)
    return within


def _json_write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def _load_rows():
    rows_dir = RUN / 'rows'
    return {p.stem: np.load(p, mmap_mode='r', allow_pickle=False) for p in rows_dir.glob('*.npy')}


def _traces(rows, ids):
    result = {}
    for i in ids:
        key = (int(rows['task'][i]), int(rows['episode'][i]))
        result.setdefault(key, []).append(int(i))
    for key, ids0 in result.items():
        result[key] = sorted(ids0, key=lambda i: int(rows['frame'][i]))
    return result


def _variable_pool(rows, memory_ids):
    import motion_core_eval as mc
    pool, raw_pool, metadata = [], [], []
    for (task, ep), ids in _traces(rows, memory_ids).items():
        trace = np.concatenate([np.asarray(rows['actions'][i, :, :7], np.float32) for i in ids])
        # Anchor cutting is defined on the full stitched demonstration, including switch cuts.
        for s, e in mc.anchor_spans(trace, min_length=5):
            seg = trace[s:e]
            phys = _to_physical(seg[:, :6])
            # The variable representation retains the natural length and DCT-3 motion;
            # grip is discrete and stored as the segment majority sign.
            rec = np.column_stack([_idct_low(phys, 3), np.full(len(seg), 1. if np.mean(seg[:, 6]) >= 0 else -1.)])
            pool.append(rec.astype(np.float32))
            raw_pool.append(seg.astype(np.float32))
            metadata.append((task, ep, s, e - s, int(np.sign(np.mean(seg[:, 6])))))
    return pool, np.asarray(metadata, np.int64), raw_pool


def _to_physical(x):
    import variable_shape_memory as vs
    return vs.to_physical(np.asarray(x, np.float32))


def _to_norm(x):
    import variable_shape_memory as vs
    return vs.to_normalized(np.asarray(x, np.float32))


def _idct_low(x, k):
    c = dct(np.asarray(x, np.float64), type=2, norm='ortho', axis=0)
    c[min(k, len(x)):] = 0
    return idct(c, type=2, norm='ortho', axis=0)


def _resample10(x):
    x = np.asarray(x, np.float64)
    t = np.linspace(0, len(x) - 1, 10)
    return np.stack([np.interp(t, np.arange(len(x)), x[:, j]) for j in range(x.shape[1])], 1)


def _hard_eval():
    return json.loads((ROOT / 'oracle-01/hard_threshold.json').read_text())


def _legacy_anchor_dp_reference():
    """Re-run the archived variable-unit trajectory DP used for the .0344/.0685 benchmark."""
    import stored_units_eval as stored
    import motion_core_eval as mc
    stored.load_data(); S=mc.ev.Shared
    rows=_load_rows(); mem=np.flatnonzero(np.asarray(rows['memory'],bool))
    mc.Data.memory_rotations={k:np.repeat(np.eye(3)[None],len(t),axis=0) for k,t in S.memory.items()}
    mc.Data.query_rotations={k:np.repeat(np.eye(3)[None],len(t),axis=0) for k,t in S.traces.items()}
    values,lengths=mc.memory_values(lambda key,trace:mc.anchor_spans(trace,min_length=5),'world',1)
    if len(values)!=4516: raise AssertionError(('legacy anchor count',len(values)))
    metrics=mc.score(values,lengths,'world','joint',1,10,False)
    if abs(metrics['all']-.034447897620114326)>2e-6 or abs(metrics['hard']-.06845424480130637)>2e-6:
        raise AssertionError(f"variable DCT3 reference mismatch: {metrics}")
    return {'units':len(values),'tau08_units_joint':int(len(mc.choose_units(values,.08,'joint'))),
            'tau08_units_separate':int(len(mc.choose_units(values,.08,'separate'))),**metrics}


def _pool_metrics(target, cand, hard, episode, name):
    outs = {}
    for mode, axes in [('seven_axis_oracle', np.arange(7)), ('arm6_oracle', np.arange(6))]:
        idx, _ = nearest_candidates(target, cand, axes)
        arm, grip, total = action_parts_mse(cand[idx], target)
        outs[mode] = {}
        for subset, mask in [('all', np.ones(len(target), bool)), ('hard', hard)]:
            vals = arm[mask]
            outs[mode][subset] = {'n': int(mask.sum()), 'arm_mse': float(vals.mean()) if len(vals) else None,
                                  'arm_ci95_episode': bootstrap_mean(vals, episode[mask]) if len(vals) else None,
                                  'grip_mse_contribution': float(grip[mask].mean()),
                                  'active7_mse': float(total[mask].mean())}
        outs[mode]['chosen_index'] = idx
    return outs


def _switch_metrics(pred, target):
    ps = np.sign(np.asarray(pred)[..., 6]); ts = np.sign(np.asarray(target)[..., 6])
    p_evt = np.flatnonzero(ps[1:] != ps[:-1]) + 1
    t_evt = np.flatnonzero(ts[1:] != ts[:-1]) + 1
    hit_p, hit_t = set(), set()
    for i, p in enumerate(p_evt):
        close = [(abs(int(p - t)), j) for j, t in enumerate(t_evt) if abs(int(p - t)) <= 2]
        close = [x for x in close if x[1] not in hit_t]
        if close:
            _, j = min(close); hit_p.add(i); hit_t.add(j)
    return len(p_evt), len(t_evt), len(hit_p), len(hit_t)


def _classifier_inputs(rows, ids):
    x, y, keys = [], [], []
    row_map = {}
    for i in ids:
        row_map[(int(rows['task'][i]), int(rows['episode'][i]), int(rows['frame'][i]))] = int(i)
    for i in ids:
        task, ep, fr = (int(rows[k][i]) for k in ('task', 'episode', 'frame'))
        hist = np.zeros((10, 7), np.float32)
        prev = row_map.get((task, ep, fr - 10))
        if prev is not None:
            hist = np.asarray(rows['actions'][prev, :, :7], np.float32)
        x.append(np.r_[hist.reshape(-1), np.asarray(rows['state'][i, :8], np.float32)])
        y.append(np.asarray(rows['actions'][i, :, 6] > 0, np.float32))
        keys.append((task, ep, fr))
    return np.asarray(x, np.float32), np.asarray(y, np.float32), np.asarray(keys, np.int64)


def _classifier_network(input_dim, width):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(input_dim, width), nn.GELU(), nn.Linear(width, 10))


def train_gripper_probe(train_x, train_y, dev_x, dev_y, *, epochs=120):
    """Three-seed small MLP; width selected by mean seed BCE on Dev, not Eval."""
    import torch
    from torch import nn
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
    records, eval_models = [], []
    for width in (64, 128):
        best = []
        for seed in (4101, 4102, 4103):
            torch.manual_seed(seed)
            net = _classifier_network(train_x.shape[1], width)
            opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
            tx, ty = torch.tensor(train_x), torch.tensor(train_y)
            dx, dy = torch.tensor(dev_x), torch.tensor(dev_y)
            best_loss, best_state = float('inf'), None
            for _ in range(epochs):
                net.train(); opt.zero_grad(); loss = nn.functional.binary_cross_entropy_with_logits(net(tx), ty); loss.backward(); opt.step()
                net.eval()
                with torch.no_grad(): v = nn.functional.binary_cross_entropy_with_logits(net(dx), dy).item()
                if v < best_loss: best_loss, best_state = v, {k: t.detach().clone() for k, t in net.state_dict().items()}
            net.load_state_dict(best_state); net.eval()
            with torch.no_grad(): pred = torch.sigmoid(net(dx)).numpy()
            best.append((best_loss, net, pred))
            records.append({'width': width, 'seed': seed, 'dev_bce': best_loss})
        eval_models.append((width, best))
    width_scores = {w: float(np.mean([r['dev_bce'] for r in records if r['width'] == w]))
                    for w in (64, 128)}
    chosen_width = min(width_scores, key=width_scores.get)
    chosen = next(v for w, v in eval_models if w == chosen_width)
    for r in records:
        r['width_mean_dev_bce'] = width_scores[r['width']]
    return chosen_width, [m for _, m, _ in chosen], records


class GripperPolicyHook:
    """Wrap an existing memory policy and replace only its returned gripper dimension.

    ``unit_grippers`` must be in the same index order as the selector's pool.  Rules are
    ``unit``, ``previous``, ``g1``, ``pi05`` or ``classifier``.  The parent policy's arm
    actions and memory-query path are left intact.
    """
    def __init__(self, policy, rule, unit_grippers, classifier=None):
        if rule not in {'unit', 'previous', 'g1', 'pi05', 'classifier'}:
            raise ValueError(rule)
        self.policy, self.rule = policy, rule
        self.unit_grippers = np.asarray(unit_grippers, np.float32)
        self.classifier = classifier
        self.last_grip = {}

    def infer(self, observation):
        raw = {k: observation[k] for k in ['observation/image', 'observation/wrist_image', 'observation/state', 'prompt']}
        row = self.policy.transform(raw)
        ep = str(observation.get('s2_episode_id', 'episode'))
        query_id = int(observation.get('s2_query_id', 0))
        if query_id == 0:
            self.last_grip[ep] = 0.
        prior = self.last_grip.get(ep, 0.)
        prior_hist = list(getattr(self.policy, 'hist', {}).get(ep, []))
        if query_id == 0: prior_hist = []
        result = self.policy.infer(observation)
        dbg = result.get('s2_debug', {})
        if 'unit' not in dbg:
            return result
        if self.rule == 'unit':
            idx = int(dbg['unit']); norm = float(majority_gripper_sign(self.unit_grippers[idx]))
            seq = np.full(10, norm, np.float32)
        elif self.rule == 'previous':
            seq = np.full(10, prior, np.float32)
        elif self.rule == 'g1':
            if query_id == 0:
                norm = -1. if float(np.asarray(row['state'])[6]) > .3019486666 else 1.
            else:
                norm = prior
            seq = np.full(10, norm, np.float32)
        elif self.rule == 'pi05':
            # The wrapped policy already computed the same observation's π0.5 chunk.
            # Its returned grip is precisely that path's gripper when memory modifies arm only.
            actions = np.asarray(result['actions'])
            seq = np.sign(actions[:, 6]).astype(np.float32)
        else:
            import torch
            hist = np.asarray(prior_hist, np.float32)
            hist = hist[-10:] if len(hist) >= 10 else [np.zeros(7, np.float32)] * (10-len(hist)) + list(hist)
            x = np.r_[np.asarray(hist, np.float32).reshape(-1), np.asarray(row['state'], np.float32)[:8]]
            with torch.no_grad():
                pred = torch.sigmoid(self.classifier(torch.as_tensor(x, dtype=torch.float32)[None])).numpy()[0]
            seq = np.where(pred >= .5, 1., -1.).astype(np.float32)
        # Convert normalized grip through the policy's action normalizer without touching arm dims.
        normalized = np.zeros((10, 32), np.float32); normalized[:, 6] = seq
        physical = self.policy.unnormalize({'actions': normalized, 'state': row['state']})['actions'][:, 6]
        result['actions'][:, 6] = physical
        self.last_grip[ep] = float(seq[4])
        # Keep the base policy's executed-action history consistent with what was actually returned.
        executed = getattr(self.policy, 'hist', {}).get(ep)
        if executed is not None and len(executed) >= 5:
            for a, v in zip(executed[-5:], seq[:5]): a[6] = v
        dbg['gripper_rule'] = self.rule
        dbg['gripper_norm_first'] = float(seq[0])
        return result


def run_offline(out=OUT):
    out = Path(out)
    if out.exists() and any(out.iterdir()): raise FileExistsError(f'output exists; refusing overwrite: {out}')
    out.mkdir(parents=True, exist_ok=True)
    rows = _load_rows()
    memory = np.flatnonzero(np.asarray(rows['memory'], bool))
    val = np.asarray(rows['validation'], bool)
    unseen_dir = ROOT / 'unseen-rows-02'
    ua = np.load(unseen_dir / 'actions.npy', mmap_mode='r', allow_pickle=False)
    utask = np.load(unseen_dir / 'task.npy', mmap_mode='r', allow_pickle=False)
    uep = np.load(unseen_dir / 'episode.npy', mmap_mode='r', allow_pickle=False)
    uframe = np.load(unseen_dir / 'frame.npy', mmap_mode='r', allow_pickle=False)
    isdev = np.load(unseen_dir / 'is_dev.npy', mmap_mode='r', allow_pickle=False).astype(bool)
    eval_idx = np.flatnonzero(~isdev)
    y_eval = np.asarray(ua[eval_idx, :, :7], np.float32)
    hard_cut = _hard_eval()['threshold']
    raw_oracle = np.load(ROOT / 'oracle-01/drafts/raw_none_eval.npz', allow_pickle=False)
    hard = np.asarray(raw_oracle['pre_mse'] >= hard_cut, bool)
    if len(y_eval) != 619 or hard.sum() != 134: raise AssertionError((len(y_eval), hard.sum()))
    eval_ep = episode_codes(utask[eval_idx], uep[eval_idx])

    # Fixed window candidates from all 400 LIBERO-40 memory demonstrations.
    W = np.asarray(rows['actions'][memory, :, :7], np.float32)
    _, werr = nearest_candidates(y_eval, W, np.arange(7))
    base_fixed = {'units': len(W), 'mse_all': float(werr.mean()), 'mse_hard': float(werr[hard].mean()),
                  'expected_reference': {'all': .0340196118, 'hard': .0816144943}}
    if abs(base_fixed['mse_all']-.0340196118)>2e-6 or abs(base_fixed['mse_hard']-.0816144943)>2e-6:
        raise AssertionError(f"fixed raw reference mismatch: {base_fixed}")
    # Per-demo DCT-3 reconstruction, with arm and gripper measured separately.
    fixed_rec = dct_roundtrip(W, 3)
    f_arm, f_grip, f_total = action_parts_mse(fixed_rec, W)
    fixed_switch = np.any(np.sign(W[:, 1:, 6]) != np.sign(W[:, :-1, 6]), axis=1)

    # Variable units: same speed-minimum + grip-switch cutter as the reviewed unit bank.
    Ulist, Umeta, Uraw = _variable_pool(rows, memory)
    if len(Ulist) != 4516: raise AssertionError(f'anchor pool count changed: {len(Ulist)}')
    U10 = np.stack([_resample10(s) for s in Ulist]).astype(np.float32)
    U10[:, :, :6] = _to_norm(U10[:, :, :6])
    # Retrieval candidates use their DCT-3 reconstructed, natural-length action; query-time scoring
    # is 10-frame phase interpolation (a diagnostic alignment, distinct from closed-loop playback).
    _, uerr = nearest_candidates(y_eval, U10, np.arange(7))
    memory_traces = _traces(rows, memory)
    # Self reconstruction at native length (correctly unpadded) for DCT loss.
    var_arm, var_grip, var_total = [], [], []
    switch_units = []
    for raw_unit in Uraw:
        arm_part, grip_part, total_part = variable_dct3_parts(raw_unit)
        var_arm.append(arm_part); var_grip.append(grip_part); var_total.append(total_part)
    # Switch-ending flag on raw source units (cut endpoints at a switch).
    for task, ep, start, length, grip in Umeta:
        ids = memory_traces.get((int(task), int(ep)), [])
        trace = np.concatenate([np.asarray(rows['actions'][i, :, :7], np.float32) for i in ids]) if ids else np.zeros((0,7))
        end = int(start + length)
        switch_units.append(bool(end < len(trace) and np.sign(trace[end,6]) != np.sign(trace[end-1,6])))
    switch_units = np.asarray(switch_units, bool)

    # In-sample domain: ~memory & ~validation, sample 15 windows/task with fixed seed.
    outside = memory_external_nonvalidation_rows(rows)
    rng = np.random.default_rng(4101)
    seen_ids = []
    for t in np.unique(rows['task'][outside]):
        pool = outside[np.asarray(rows['task'][outside]) == t]
        seen_ids.extend(rng.choice(pool, min(15, len(pool)), replace=False).tolist())
    seen_ids = np.asarray(sorted(seen_ids), np.int64)
    memory_pairs={(int(rows['task'][i]),int(rows['episode'][i])) for i in memory}
    seen_pairs={(int(rows['task'][i]),int(rows['episode'][i])) for i in seen_ids}
    if memory_pairs & seen_pairs: raise AssertionError('learned evaluation contains a memory episode')
    y_seen = np.asarray(rows['actions'][seen_ids, :, :7], np.float32)
    # W oracle per-window, all memory candidates; U oracle via aligned DCT units.
    _, seen_raw_err = nearest_candidates(y_seen, W, np.arange(7))
    seen_hard = top_error_mask(seen_raw_err, .2)
    seen_ep_key = episode_codes(rows['task'][seen_ids], rows['episode'][seen_ids])
    eval_ep_key = eval_ep
    Wseen = _pool_metrics(y_seen, W, seen_hard, seen_ep_key, 'W')
    Weval = _pool_metrics(y_eval, W, hard, eval_ep_key, 'W')
    Useen = _pool_metrics(y_seen, U10, seen_hard, seen_ep_key, 'U')
    Ueval = _pool_metrics(y_eval, U10, hard, eval_ep_key, 'U')
    def arm_delta(pool, indices7, indices6, target, episode, mask):
        e7 = np.square(pool[indices7, ..., :6] - target[..., :6]).sum((1,2)) / (target.shape[1] * 7)
        e6 = np.square(pool[indices6, ..., :6] - target[..., :6]).sum((1,2)) / (target.shape[1] * 7)
        return paired_episode_bootstrap((e6-e7)[mask], np.asarray(episode)[mask])
    paired_selection = {
        'W_seen': arm_delta(W, Wseen['seven_axis_oracle']['chosen_index'], Wseen['arm6_oracle']['chosen_index'], y_seen, seen_ep_key, np.ones(len(y_seen), bool)),
        'W_eval': arm_delta(W, Weval['seven_axis_oracle']['chosen_index'], Weval['arm6_oracle']['chosen_index'], y_eval, eval_ep_key, np.ones(len(y_eval), bool)),
        'W_eval_hard': arm_delta(W, Weval['seven_axis_oracle']['chosen_index'], Weval['arm6_oracle']['chosen_index'], y_eval, eval_ep_key, hard),
        'U_seen': arm_delta(U10, Useen['seven_axis_oracle']['chosen_index'], Useen['arm6_oracle']['chosen_index'], y_seen, seen_ep_key, np.ones(len(y_seen), bool)),
        'U_eval': arm_delta(U10, Ueval['seven_axis_oracle']['chosen_index'], Ueval['arm6_oracle']['chosen_index'], y_eval, eval_ep_key, np.ones(len(y_eval), bool)),
        'U_eval_hard': arm_delta(U10, Ueval['seven_axis_oracle']['chosen_index'], Ueval['arm6_oracle']['chosen_index'], y_eval, eval_ep_key, hard)}

    # Section A: raw signal properties and DCT diagnostics.
    action_all = np.asarray(rows['actions'][memory, :, :7], np.float32)
    switch_per_demo = []
    reps = []
    for key, ids in _traces(rows, memory).items():
        trace = np.concatenate([np.asarray(rows['actions'][i, :, :7], np.float32) for i in ids])
        switch_per_demo.append(int(np.sum(np.sign(trace[1:,6]) != np.sign(trace[:-1,6]))))
        if len(reps) < 4: reps.append({'task': key[0], 'episode': key[1], 'action': trace.tolist(), 'reconstruction': np.column_stack([dct_unit(trace[:,:6],3), dct_unit(trace[:,6:7],3)]).tolist()})
    near_binary = float(np.mean(np.abs(action_all[...,6]) >= .9))
    grip_switch_window_fixed = float(fixed_switch.mean())
    # DCT3 for all fixed windows, split the arm/grip contributions and windows containing a switch.
    dct_fixed = {'arm_mean': float(f_arm.mean()), 'grip_contribution_mean': float(f_grip.mean()), 'active7_mean': float(f_total.mean()),
                 'switch_windows': int(fixed_switch.sum()), 'switch_window_arm_mean': float(f_arm[fixed_switch].mean()) if fixed_switch.any() else None,
                 'switch_window_grip_contribution_mean': float(f_grip[fixed_switch].mean()) if fixed_switch.any() else None}

    # Section C: tau=.08 coefficient-space greedy cover, consistent with DCT tolerance merge.
    coeff = []
    for x in Uraw:
        c_full = dct(np.asarray(x[:, :6], np.float64), type=2, norm='ortho', axis=0)
        c = np.zeros((3, 6), np.float64); c[:min(3, len(c_full))] = c_full[:3]
        g_full = dct(np.asarray(x[:, 6], np.float64), type=2, norm='ortho')
        gc = np.zeros(1, np.float64); gc[:min(1, len(g_full))] = g_full[:1]
        coeff.append(np.r_[c.reshape(-1), gc] / np.sqrt(len(x)))
    coeff = np.asarray(coeff, np.float64)
    from dct_tolerance_merge import greedy_cover
    def merge_count(include_grip):
        within = tau_neighbor_matrix(coeff, include_gripper=include_grip, tau=.08)
        return int(len(greedy_cover(within)[0]))
    c_counts = {'tau': .08, 'tau_metric': 'sum of normalized-action orthonormal DCT coefficient squared differences / 7 (motion_core_eval active7 ruler)',
                'with_gripper_units': merge_count(True), 'without_gripper_units': merge_count(False)}
    # Like CMP2-D, compare cluster separation on an equal-size deterministic sample.
    # Both representations use the same U units and phase samples; only the grip axis
    # is removed in the second representation.
    import compression_matched as cmp2
    phase_sample = np.stack([cmp2.phase_resample(Uraw[i], 32) for i in range(len(Uraw))])
    c_counts['cluster_separation_k95'] = {
        'definition': 'CMP2-D between-center / within-cluster squared-distance ratio on 32-phase samples',
        'equalized_units': int(len(phase_sample)),
        'same_unit_sample': True,
        'with_gripper': cmp2.between_within_ratio(phase_sample, 95),
        'without_gripper': cmp2.between_within_ratio(phase_sample[:, :, :6], 95),
    }

    # Section D: hold arm from six-axis oracle selection; only replace the grip trace.
    d_idx, _ = nearest_candidates(y_eval, U10, np.arange(6))
    base_arm = U10[d_idx].copy()
    W_eval_inputs = np.load(PREP / 'W_inputs_eval.npz', allow_pickle=False)
    pi05 = np.asarray(W_eval_inputs['pi05'], np.float32)
    if not np.array_equal(W_eval_inputs['row'], eval_idx): raise AssertionError('pi05 cache rows do not align to unseen Eval indices')
    # G1 episode start threshold is documented scale-only threshold .30195.
    previous = np.zeros_like(y_eval[:,:,6]); starts = np.asarray(uframe[eval_idx]) == 0
    for i in range(len(eval_idx)):
        if not starts[i]:
            same = (np.asarray(utask[eval_idx]) == utask[eval_idx[i]]) & (np.asarray(uep[eval_idx]) == uep[eval_idx[i]]) & (np.asarray(uframe[eval_idx]) == uframe[eval_idx[i]] - 10)
            j = np.flatnonzero(same)
            previous[i] = y_eval[j[0], -1, 6] if len(j) else 0
    # classifier: memory-external rows, held-out Dev task6/7 only for model selection.
    # Use the 113 W-input Dev rows, which are a strict independent task split.
    policy_train_ids = np.flatnonzero(~val)
    classifier_train_ids = exclude_episode_rows(policy_train_ids, rows['task'], rows['episode'], seen_ids)
    if not len(classifier_train_ids): raise AssertionError('classifier training set is empty after episode holdout')
    classifier_episode_pairs={(int(rows['task'][i]),int(rows['episode'][i])) for i in classifier_train_ids}
    if classifier_episode_pairs & seen_pairs: raise AssertionError('learned evaluation episode leaked into classifier training')
    tx, ty, _ = _classifier_inputs(rows, classifier_train_ids)
    # Build eval histories from previous true executed actions and state.
    un_states = np.load(unseen_dir/'state.npy', mmap_mode='r', allow_pickle=False)
    eval_state = np.asarray(un_states[eval_idx,:8], np.float32)
    eval_hist = np.zeros((len(eval_idx),10,7), np.float32)
    for i, (t,e,f) in enumerate(zip(utask[eval_idx], uep[eval_idx], uframe[eval_idx])):
        if f >= 10:
            j = np.flatnonzero((utask[eval_idx] == t) & (uep[eval_idx] == e) & (uframe[eval_idx] == f-10))
            if len(j): eval_hist[i] = y_eval[j[0]]
    import scale_only_query as soq
    g1_norm = soq.gripper_g1({'frame':np.asarray(uframe[eval_idx]),'hist':eval_hist,'last':eval_hist[:,-1],
                              'state':eval_state},.3019486666,'W')
    g1_values=np.broadcast_to(g1_norm[:,None],(len(eval_idx),10)).copy()
    ex = np.c_[eval_hist.reshape(len(eval_idx),-1),eval_state]
    # Dev split inputs are from LIBERO-90 6/7 and are for model selection only.
    dev_uids = np.flatnonzero(isdev)
    dev_cache=np.load(PREP/'W_inputs_dev.npz',allow_pickle=False)
    if not np.array_equal(dev_cache['row'],dev_uids): raise AssertionError('Dev cache rows misaligned')
    dev_hist = np.asarray(dev_cache['hist'],np.float32)
    dev_state = np.asarray(dev_cache['state'],np.float32)
    dy = np.asarray(ua[dev_uids,:,6] > 0,np.float32)
    dx = np.c_[dev_hist.reshape(len(dev_uids),-1),dev_state]
    width, models, clf_records = train_gripper_probe(tx,ty,dx,dy)
    import torch
    classifier_pred=[]
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','8')))
    with torch.no_grad():
        for model in models:
            classifier_pred.append(torch.sigmoid(model(torch.tensor(ex))).numpy())
    classifier_pred=(np.mean(classifier_pred,axis=0) >= .5).astype(np.float32)*2-1
    dev_y=np.asarray(ua[dev_uids,:,:7],np.float32)
    dev_arm_idx,_=nearest_candidates(dev_y,U10,np.arange(6));dev_arm=U10[dev_arm_idx].copy()
    dev_g1=soq.gripper_g1({'frame':np.asarray(uframe[dev_uids]),'hist':dev_hist,'last':dev_hist[:,-1],
                           'state':dev_state},.3019486666,'W')
    dev_g1=np.broadcast_to(dev_g1[:,None],(len(dev_uids),10)).copy()
    dev_prev=np.broadcast_to(dev_hist[:,-1,6,None],(len(dev_uids),10)).copy()
    dev_prev[~np.asarray(dev_cache['has_hist'],bool)]=0.
    with torch.no_grad():
        dev_classifier=(np.mean([torch.sigmoid(m(torch.tensor(dx))).numpy() for m in models],axis=0)>=.5).astype(np.float32)*2-1
    dev_grips={'G0_unit_gripper':U10[dev_arm_idx,:,6], 'G2_previous_hold':dev_prev,
               'G1_state_initialized_hold':dev_g1, 'G3_pi05_gripper':np.asarray(dev_cache['pi05'])[:,:,6],
               'G4_classifier':dev_classifier}
    D_dev={}
    for name,g in dev_grips.items():
        pr=dev_arm.copy();pr[:,:,6]=g;_,gc,total=action_parts_mse(pr,dev_y)
        D_dev[name]={'n':len(dev_y),'grip_mse_contribution':float(gc.mean()),
                     'grip_sign_accuracy':float(np.mean(np.sign(pr[:,:,6])==np.sign(dev_y[:,:,6]))),
                     'active7_mse':float(total.mean())}
    dev_best_rule=min(D_dev,key=lambda name:D_dev[name]['grip_mse_contribution'])
    # Learned-LIBERO comparison uses the exact rows in the cached P5 training actions.
    import hashlib
    train_policy=np.load(ROOT/'key-query-01/policy_features/train_actions.npy',mmap_mode='r',allow_pickle=False)
    manifest=json.loads((ROOT/'key-query-01/policy_features/manifest.json').read_text())
    if hashlib.sha256(np.asarray(policy_train_ids,dtype=np.int64).tobytes()).hexdigest()!=manifest['row_ids_sha256']['train']:
        raise AssertionError('cached pi05 train rows do not match LIBERO-40 non-validation split')
    pos={int(r):i for i,r in enumerate(policy_train_ids)}; seen_pos=np.asarray([pos[int(r)] for r in seen_ids])
    seen_pi=np.asarray(train_policy[seen_pos],np.float32)
    train_pos={int(r):i for i,r in enumerate(policy_train_ids)}
    policy_train_x,_,_=_classifier_inputs(rows,policy_train_ids)
    seen_hist=np.stack([policy_train_x[train_pos[int(r)],:70].reshape(10,7) for r in seen_ids])
    seen_state=np.asarray(rows['state'][seen_ids,:8],np.float32)
    seen_starts=np.asarray(rows['frame'][seen_ids])==0
    seen_g1=soq.gripper_g1({'frame':np.asarray(rows['frame'][seen_ids]),'hist':seen_hist,'last':seen_hist[:,-1],'state':seen_state},.3019486666,'W')
    seen_g1=np.broadcast_to(seen_g1[:,None],(len(seen_ids),10)).copy()
    seen_prev=np.broadcast_to(seen_hist[:,-1,6,None],(len(seen_ids),10)).copy()
    seen_prev[seen_starts]=0
    seen_arm_idx,_=nearest_candidates(y_seen,U10,np.arange(6));seen_arm=U10[seen_arm_idx].copy()
    seen_grip_idx=np.argmin(np.square(y_seen[:,None,:,6]-U10[None,:,:,6]).mean(axis=2),axis=1)
    seen_D={}
    seen_strategies={'G0_unit_gripper':U10[seen_arm_idx,:,6], 'G2_previous_hold':seen_prev,'G1_state_initialized_hold':seen_g1,
                     'G3_pi05_gripper':seen_pi[:,:,6], 'G4_classifier':(np.mean([torch.sigmoid(m(torch.tensor(np.c_[seen_hist.reshape(len(seen_ids),-1),seen_state],dtype=torch.float32))).detach().numpy() for m in models],axis=0)>=.5).astype(np.float32)*2-1,
                     'G_upper_oracle_grip':U10[seen_grip_idx,:,6]}
    for name,g in seen_strategies.items():
        pr=seen_arm.copy();pr[:,:,6]=g;_,gc,total=action_parts_mse(pr,y_seen)
        seen_D[name]={}
        for subset,mask in [('all',np.ones(len(seen_ids),bool)),('hard',seen_hard)]:
            counts=np.zeros(4,int)
            for pred_seq,true_seq in zip(pr[mask],y_seen[mask]): counts+=_switch_metrics(pred_seq,true_seq)
            pe,te,hp,ht=counts
            seen_D[name][subset]={'n':int(mask.sum()),'grip_mse_contribution':float(gc[mask].mean()),'grip_sign_accuracy':float(np.mean(np.sign(pr[mask,:,6])==np.sign(y_seen[mask,:,6]))),'active7_mse':float(total[mask].mean()),
                'switch_precision_within_2':float(hp/max(pe,1)),'switch_recall_within_2':float(ht/max(te,1))}
        seen_D[name]['by_half']={}
        for half,sl in [('first5',slice(0,5)),('second5',slice(5,10))]:
            seen_D[name]['by_half'][half]={'grip_mse_contribution':float(np.square(pr[:,:,6][:,sl]-y_seen[:,:,6][:,sl]).mean()/7),
                'sign_accuracy':float(np.mean(np.sign(pr[:,:,6][:,sl])==np.sign(y_seen[:,:,6][:,sl])))}
            seen_D[name].setdefault('by_half_hard',{})[half]={'grip_mse_contribution':float(np.square(pr[seen_hard,:,6][:,sl]-y_seen[seen_hard,:,6][:,sl]).mean()/7) if seen_hard.any() else None,
                'sign_accuracy':float(np.mean(np.sign(pr[seen_hard,:,6][:,sl])==np.sign(y_seen[seen_hard,:,6][:,sl]))) if seen_hard.any() else None}
    strategies = {'G0_unit_gripper': np.broadcast_to(U10[d_idx,:,6],(len(eval_idx),10)).copy(),
                  'G2_previous_hold': previous.copy(), 'G1_state_initialized_hold': g1_values.copy(), 'G3_pi05_gripper': pi05[:,:,6].copy(),
                  'G4_classifier': classifier_pred.copy()}
    # For “stored” per candidate, use the anchor unit's sign. For G1, first-frame initialization as above.
    grip_err = np.square(y_eval[:, None, :, 6] - U10[None, :, :, 6]).mean(axis=2)
    d_upper = np.argmin(grip_err, axis=1)
    strategies['G_upper_oracle_grip'] = U10[d_upper,:,6].copy()
    D = {}
    for name,g in strategies.items():
        pred = base_arm.copy(); pred[:,:,6]=g
        arm, gc, total=action_parts_mse(pred,y_eval)
        d={}
        for subset,mask in [('all',np.ones(len(y_eval),bool)),('hard',hard)]:
            d[subset]={'grip_mse_contribution':float(gc[mask].mean()),'grip_sign_accuracy':float(np.mean(np.sign(pred[mask,:,6])==np.sign(y_eval[mask,:,6]))),
                       'active7_mse':float(total[mask].mean()),'grip_switch_window_count':int(np.sum(np.any(np.sign(y_eval[mask,1:,6])!=np.sign(y_eval[mask,:-1,6]),axis=1)))}
            counts=np.zeros(4,int)
            for p,t in zip(pred[mask],y_eval[mask]): counts += _switch_metrics(p,t)
            p_evt,t_evt,hp,ht=counts
            d[subset]['switch_precision_within_2']=float(hp/max(p_evt,1));d[subset]['switch_recall_within_2']=float(ht/max(t_evt,1));d[subset]['predicted_switches']=int(p_evt);d[subset]['true_switches']=int(t_evt)
        D[name]=d
    D_task={}
    for task in np.unique(utask[eval_idx]):
        mask=utask[eval_idx]==task
        D_task[str(int(task))]={}
        for name,g in strategies.items():
            pred=base_arm[mask].copy();pred[:,:,6]=g[mask]
            _,gc,total=action_parts_mse(pred,y_eval[mask])
            D_task[str(int(task))][name]={'n':int(mask.sum()),'grip_mse_contribution':float(gc.mean()),
                'sign_accuracy':float(np.mean(np.sign(pred[:,:,6])==np.sign(y_eval[mask,:,6]))),'active7_mse':float(total.mean())}
    # MSE by first/second half and hard/all, for all D strategies.
    for name,g in strategies.items():
        pred=base_arm.copy();pred[:,:,6]=g
        D[name]['by_half']={}
        for half,sl in [('first5',slice(0,5)),('second5',slice(5,10))]:
            D[name]['by_half'][half]={'grip_mse_contribution':float(np.square(pred[:,:,6][:,sl]-y_eval[:,:,6][:,sl]).mean()/7),
                                      'sign_accuracy':float(np.mean(np.sign(pred[:,:,6][:,sl])==np.sign(y_eval[:,:,6][:,sl])))}
            D[name].setdefault('by_half_hard',{})[half]={'grip_mse_contribution':float(np.square(pred[hard,:,6][:,sl]-y_eval[hard,:,6][:,sl]).mean()/7) if hard.any() else None,
                                      'sign_accuracy':float(np.mean(np.sign(pred[hard,:,6][:,sl])==np.sign(y_eval[hard,:,6][:,sl]))) if hard.any() else None}

    # Figure data: four representative demos, actual and DCT-3 traces.
    figdata={'representative_demos':reps,'fixed_window_count':len(W),'variable_unit_count':len(Ulist)}
    legacy_anchor = _legacy_anchor_dp_reference()
    c_counts['with_gripper_units']=legacy_anchor['tau08_units_joint']
    c_counts['without_gripper_units']=legacy_anchor['tau08_units_separate']
    c_counts['tau_metric']='exact motion_core_eval greedy-cover distance: coefficient squared error / 7 with grip, / 6 without grip'
    result={'normalization':{'input_actions':'run-01 rows actions, normalized with pi05_libero Q01/Q99 to [-1,1]','active7_mse':'mean squared error over time and seven active axes',
                              'arm6_contribution':'sum of six-axis squared errors / (steps*7)','grip_contribution':'gripper squared error / (steps*7)','sum_equals_active7':True},
            'counts':{'memory_windows':int(len(memory)),'memory_tasks':int(len(np.unique(rows['task'][memory]))),'memory_episodes':len(_traces(rows,memory)),
             'seen_sample_rows':len(seen_ids),'seen_sample_tasks':int(len(np.unique(rows['task'][seen_ids]))),'seen_sample_demos':int(len(seen_pairs)),
             'seen_memory_episode_overlap':int(len(memory_pairs & seen_pairs)),'seen_classifier_train_episode_overlap':int(len(classifier_episode_pairs & seen_pairs)),
             'eval_rows':len(eval_idx),'eval_tasks':int(len(np.unique(utask[eval_idx]))),'eval_demos':int(len(np.unique(eval_ep_key))),'eval_hard_rows':int(hard.sum()),'dev_rows':int(len(dev_uids)),
             'fixed_units_by_task':{str(int(t)):int(np.sum(np.asarray(rows['task'][memory])==t)) for t in np.unique(rows['task'][memory])},
             'variable_units_by_task':{str(int(t)):int(np.sum(Umeta[:,0]==t)) for t in np.unique(Umeta[:,0])},
             'variable_length_frames':{'min':int(np.min(Umeta[:,3])),'p25':float(np.quantile(Umeta[:,3],.25)),'median':float(np.median(Umeta[:,3])),'p75':float(np.quantile(Umeta[:,3],.75)),'max':int(np.max(Umeta[:,3]))},
             'classifier_train_rows':int(len(classifier_train_ids))},
            'fixed_raw_reference':base_fixed,'variable_anchor_reference':{'units':len(Ulist),'legacy_full_trajectory_DP':legacy_anchor,'legacy_expected_all':.0344478976,'legacy_expected_hard':.0684542448,
              'per_window_10frame_phase_alignment_oracle_all':float(uerr.mean()),'per_window_10frame_phase_alignment_oracle_hard':float(uerr[hard].mean())},
            'A_signal':{'gripper_fraction_abs_ge_0_9':near_binary,'switches_per_demo':{'mean':float(np.mean(switch_per_demo)),'median':float(np.median(switch_per_demo)),'p25':float(np.quantile(switch_per_demo,.25)),'p75':float(np.quantile(switch_per_demo,.75))},
             'fixed_window_switch_fraction':grip_switch_window_fixed,'fixed_window_DCT3_self_reconstruction':dct_fixed,
             'variable_native_DCT3_self_reconstruction':{'arm_mse_mean':float(np.mean(var_arm)),'grip_mse_contribution_mean':float(np.mean(var_grip)),'active7_mse_mean':float(np.mean(var_total)),
                'switch_ending_units':int(switch_units.sum()),'units':len(switch_units)}},
            'B_oracle_selection':{'W_seen':Wseen,'W_eval':Weval,'U_seen':Useen,'U_eval':Ueval,'paired_arm6_minus_7axis':paired_selection},'C_tau08':c_counts,
            'D_gripper_decisions':{'classifier_width':width,'classifier_dev_records':clf_records,'dev':D_dev,'dev_selected_no_oracle_rule':dev_best_rule,
                'seen_sample':seen_D,'eval':D,'eval_by_task':D_task,
                'arm_selection':'U pool, six-axis oracle selection; identical base arm predictions across grip strategies'},
            'limits':['B selects candidates using the scored target and is an upper bound, not a deployable selector.',
                      'Variable-unit per-window B comparison linearly phase-resamples each natural-length DCT-3 unit to 10 steps; the legacy motion_core value is a full-trajectory dynamic program and is reported separately.',
                      'Classifier training excludes every episode in the learned evaluation sample; width and epoch are selected on LIBERO-90 Dev only. Eval labels do not enter training or model selection.']}
    # Keep large per-query candidate indices in npz only; summary JSON is compact.
    W_eval_idx = Weval['seven_axis_oracle']['chosen_index'].copy()
    U_eval_idx = Ueval['seven_axis_oracle']['chosen_index'].copy()
    U_eval_arm_idx = Ueval['arm6_oracle']['chosen_index'].copy()
    for pool_result in result['B_oracle_selection'].values():
        for mode_result in pool_result.values(): mode_result.pop('chosen_index', None)
    torch.save({'input_dim':int(tx.shape[1]),'width':int(width),
                'state_dicts':[m.state_dict() for m in models]},out/'classifier.pt')
    _json_write(out/'results.json',result)
    np.savez_compressed(out/'arrays.npz',eval_rows=eval_idx,eval_task=utask[eval_idx],eval_episode=uep[eval_idx],eval_frame=uframe[eval_idx],eval_hard=hard,
                        W_eval_oracle=W_eval_idx,U_eval_oracle=U_eval_idx,
                        U_eval_arm_oracle=U_eval_arm_idx,U_eval_arm_draft=base_arm,
                        eval_target=y_eval,eval_pi05=pi05,classifier_grip=classifier_pred)
    _json_write(out/'figure_data.json',figdata)
    return result


def write_report(out=OUT):
    p=Path(out)/'results.json'; r=json.loads(p.read_text())
    lines=['# GRIP1：グリッパーと腕を分ける根拠（オフライン）','',
      '物差し：LIBERO-40 の run-01 行動は pi05_libero の Q01/Q99 で [-1,1] に正規化済み。腕6軸寄与は二乗誤差和/(ステップ×7)、グリッパー寄与は同じ分母の1軸誤差で、和が active7 MSE。',
      f"データ：メモリ {r['counts']['memory_tasks']}タスク・{r['counts']['memory_episodes']}デモ・{r['counts']['memory_windows']}窓。学習済み側 {r['counts']['seen_sample_tasks']}タスク・{r['counts']['seen_sample_demos']}メモリ外デモ・{r['counts']['seen_sample_rows']}窓。未学習 Eval {r['counts']['eval_tasks']}タスク・{r['counts']['eval_demos']}デモ・{r['counts']['eval_rows']}窓（難 {r['counts']['eval_hard_rows']}）。Dev {r['counts']['dev_rows']}窓。難しい行は各評価集合の raw-memory nearest error上位20%（Eval は固定済み threshold={_hard_eval()['threshold']:.6f}）。",'','## 再現チェック','',
      '| プール | 単位数 | 全体 MSE | 難 MSE | 参照値 |','|---|---:|---:|---:|---:|',
      f"| 固定10 raw、全400デモを候補 | {r['fixed_raw_reference']['units']} | {r['fixed_raw_reference']['mse_all']:.6f} | {r['fixed_raw_reference']['mse_hard']:.6f} | .034020 / .081614 |",
      f"| 可変 anchor DCT3、1窓ずつ選択・長さ10へ位相補間 | {r['variable_anchor_reference']['units']} | {r['variable_anchor_reference']['per_window_10frame_phase_alignment_oracle_all']:.6f} | {r['variable_anchor_reference']['per_window_10frame_phase_alignment_oracle_hard']:.6f} | historical full-trajectory DP .034448 / .068454（別指標） |",
      '', '固定rawの参照値は全候補の生10窓から正解行動で最近傍を選ぶため一致対象。可変長の .034448/.068454 は逐次全軌跡DPの値であり、10フレーム窓の位相補間上限と同一指標ではない。', '', '## A. 信号の性質','',
      f"- グリッパー値が |a|≥.9：{r['A_signal']['gripper_fraction_abs_ge_0_9']:.3f}。デモあたり切替：平均 {r['A_signal']['switches_per_demo']['mean']:.2f}、中央値 {r['A_signal']['switches_per_demo']['median']:.0f}（IQR {r['A_signal']['switches_per_demo']['p25']:.0f}–{r['A_signal']['switches_per_demo']['p75']:.0f}）。",
      f"- 固定10窓で切替を含む割合：{r['A_signal']['fixed_window_switch_fraction']:.3f}。固定窓DCT3の自己復元腕誤差 {r['A_signal']['fixed_window_DCT3_self_reconstruction']['arm_mean']:.6f}、グリッパー寄与 {r['A_signal']['fixed_window_DCT3_self_reconstruction']['grip_contribution_mean']:.6f}。",
      f"- 可変単位は {r['A_signal']['variable_native_DCT3_self_reconstruction']['units']}。腕DCT3誤差 {r['A_signal']['variable_native_DCT3_self_reconstruction']['arm_mse_mean']:.6f}、離散グリッパー自己復元寄与 {r['A_signal']['variable_native_DCT3_self_reconstruction']['grip_mse_contribution_mean']:.6f}、切替で終わる単位 {r['A_signal']['variable_native_DCT3_self_reconstruction']['switch_ending_units']}。",'',
      f"- 可変単位長分布：min/p25/median/p75/max = {r['counts']['variable_length_frames']['min']}/{r['counts']['variable_length_frames']['p25']:.0f}/{r['counts']['variable_length_frames']['median']:.0f}/{r['counts']['variable_length_frames']['p75']:.0f}/{r['counts']['variable_length_frames']['max']} frames。タスク別の固定窓・可変単位数は results.json の counts に保存。",'',
      '## B. 候補選択への影響（正解を見た上限）','',
      f"評価窓と候補単位の {r['counts']['memory_windows']} LIBERO-40 memory窓（{r['counts']['memory_tasks']}タスク・{r['counts']['memory_episodes']}デモ）を使う。腕MSEは腕6軸二乗和/(10×7)。seen は {r['counts']['seen_sample_tasks']}タスク・{r['counts']['seen_sample_demos']}メモリ外デモ・{r['counts']['seen_sample_rows']}窓、Eval は {r['counts']['eval_tasks']}タスク・{r['counts']['eval_demos']}デモ・{r['counts']['eval_rows']}窓。区間はエピソード単位のbootstrap。", '', '| プール/選択 | 評価集合 | n | 腕MSE | エピソードbootstrap 95% CI |','|---|---|---:|---:|---:|']
    for pool in ('W','U'):
      for mode,label in [('seven_axis_oracle','7軸 oracle 選択'),('arm6_oracle','腕6軸 oracle 選択')]:
        for split,subset in [('seen sample','all'),('seen sample','hard'),('Eval','all'),('Eval','hard')]:
          a=r['B_oracle_selection'][f'{pool}_seen' if split=='seen sample' else f'{pool}_eval'][mode][subset]
          if a['arm_mse'] is None: continue
          ci=a['arm_ci95_episode'];lines.append(f"| {pool} / {label} | {split} {subset} | {a['n']} | {a['arm_mse']:.6f} | [{ci[1]:.6f}, {ci[2]:.6f}] |")
    lines += ['', '腕6軸選択−7軸選択の対応差（負ならグリッパーを選択距離に含めた方が腕を損なう）：']
    for k,v in r['B_oracle_selection']['paired_arm6_minus_7axis'].items(): lines.append(f"- {k}: {v['mean']:+.6f}, episode bootstrap 95% [{v['ci95'][0]:+.6f}, {v['ci95'][1]:+.6f}]")
    c95=r['C_tau08']['cluster_separation_k95']
    lines+=['','## C. τ=.08 のまとまり（係数距離）','',f"グリッパー係数を距離に含めると代表数 {r['C_tau08']['with_gripper_units']}、除くと {r['C_tau08']['without_gripper_units']}。motion_core_eval と同じく距離は係数二乗和を含む場合は7、除く場合は6で割る。",f"CMP2-D と同じ 32 位相・k=95 の between/within 比：両条件で同じ {c95['equalized_units']} 単位を使用。グリッパー込み {c95['with_gripper']['ratio']:.4f}（k_used={c95['with_gripper']['k_used']}）、除外 {c95['without_gripper']['ratio']:.4f}（k_used={c95['without_gripper']['k_used']}）。",'',
      '## D. グリッパーの決め方','',
      f"誤差は grip 二乗誤差/(10×7)、arm は6軸oracle選択で固定。学習済み評価は {r['counts']['seen_sample_tasks']}タスク・{r['counts']['seen_sample_demos']}デモ・{r['counts']['seen_sample_rows']}窓、未学習 Eval は {r['counts']['eval_tasks']}タスク・{r['counts']['eval_demos']}デモ・{r['counts']['eval_rows']}窓（hard {r['counts']['eval_hard_rows']}）。", '', '| 規則（腕はU 6軸oracle選択で固定） | 全 Eval Grip寄与 | hard Grip寄与 | 全符号一致 | 全切替適合率±2 | 全切替再現率±2 |','|---|---:|---:|---:|---:|---:|']
    for name,v in r['D_gripper_decisions']['eval'].items():
      a=v['all'];h=v['hard'];lines.append(f"| {name} | {a['grip_mse_contribution']:.6f} | {h['grip_mse_contribution']:.6f} | {a['grip_sign_accuracy']:.3f} | {a['switch_precision_within_2']:.3f} | {a['switch_recall_within_2']:.3f} |")
    lines += ['', '先頭5/続き5（Eval全体）のグリッパー寄与と符号一致：']
    for name,v in r['D_gripper_decisions']['eval'].items():
      for half,z in v['by_half'].items(): lines.append(f"- {name} {half}: grip寄与 {z['grip_mse_contribution']:.6f}, 符号一致 {z['sign_accuracy']:.3f}")
    lines += ['', 'hard Eval の先頭5/続き5：']
    for name,v in r['D_gripper_decisions']['eval'].items():
      for half,z in v['by_half_hard'].items(): lines.append(f"- {name} {half}: grip寄与 {z['grip_mse_contribution']:.6f}, 符号一致 {z['sign_accuracy']:.3f}")
    lines += ['',f"分類器は幅 {r['D_gripper_decisions']['classifier_width']}、3 seed。seen 評価の episode は分類器学習から除外し、学習は {r['counts']['classifier_train_rows']}行。Dev 113窓の BCE のseed平均で幅を選択。π0.5出力は向きを直したキャッシュを使用。タスク別は results.json の eval_by_task。", '', '閉ループ規則の選択（Dev の grip MSE、oracle を除く）：', '| 規則 | Grip寄与 | 符号一致 | active7 |', '|---|---:|---:|---:|']
    for name,v in r['D_gripper_decisions']['dev'].items():
      lines.append(f"| {name} | {v['grip_mse_contribution']:.6f} | {v['grip_sign_accuracy']:.3f} | {v['active7_mse']:.6f} |")
    lines += [f"Dev 最良規則：**{r['D_gripper_decisions']['dev_selected_no_oracle_rule']}**。分類器規則を選ぶ場合は保存した `classifier.pt` を serve の `--classifier` に渡す。", '', '学習済みLIBERO側（各タスク最大15窓の固定seedサンプル、同じ候補Uと oracle選択）:', '| 規則 | all Grip寄与 | hard Grip寄与 | all active7 |','|---|---:|---:|---:|']
    for name,v in r['D_gripper_decisions']['seen_sample'].items():
      a=v['all'];h=v['hard'];lines.append(f"| {name} | {a['grip_mse_contribution']:.6f} | {h['grip_mse_contribution']:.6f} | {a['active7_mse']:.6f} |")
    lines += ['', '学習済み評価の時間分割（grip寄与 / 符号一致）：']
    for name,v in r['D_gripper_decisions']['seen_sample'].items():
      for half,z in v['by_half'].items(): lines.append(f"- {name} {half}: {z['grip_mse_contribution']:.6f} / {z['sign_accuracy']:.3f}")
    lines += ['', '学習済み評価 hard の時間分割：']
    for name,v in r['D_gripper_decisions']['seen_sample'].items():
      for half,z in v['by_half_hard'].items(): lines.append(f"- {name} {half}: {z['grip_mse_contribution']:.6f} / {z['sign_accuracy']:.3f}")
    smoke_path=Path(out)/'smoke'/'smoke.json'
    if smoke_path.exists():
      sm=json.loads(smoke_path.read_text());tr=sm.get('trial',{})
      smoke_note=f"実環境1試行スモーク完走：task {tr.get('task_id')} init {tr.get('initial_state_index')}、{tr.get('env_steps')} steps、success={tr.get('success')}、queries={tr.get('num_policy_queries')}。"
    else: smoke_note='実環境1試行スモークは未実施。'
    lines += ['',
      '## E. 閉ループ用フック','', '新しいモジュールの `GripperPolicyHook(policy, rule, unit_grippers, classifier=None)` は既存 memory policy の infer を包み、腕を変えずに選択単位のグリッパー／直前保持／G1／π0.5／分類器の規則を差し替える。`memory_closed_loop.py` は編集していない。', '',
      smoke_note,'','## 解釈と制約','', 'B は正解で候補を選ぶ診断上限であり、閉ループ検索性能の主張ではない。U の窓比較では可変長単位を10点へ位相補間するため、自然長を順に実行する評価とは別。グリッパー分類器の入力履歴はデモの直前実行行動で、閉ループでは実際に出したコマンドに置き換わる。',
      '共通ゲート `/work/docs/memory_pilot_gates.md` は環境内に見つからなかった。オフライン解析はCPUで実施。']
    (Path(out)/'report.md').write_text('\n'.join(lines)+'\n')


def serve_hook(args):
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','8')))
    os.environ['MEM_POOL']=str(args.pool);os.environ['MEM_CKPT']=str(args.query)
    os.environ['MEM_T0']='0';os.environ.pop('MEM_EXPERT',None)
    sys.path.insert(0,str(HERE))
    import memory_closed_loop as mcl
    policy=mcl.make_policy(args.checkpoint)
    z=np.load(args.pool,allow_pickle=False)
    unit_grips=majority_gripper_sign(np.asarray(z['shape'])[:,:,6].mean(1))
    classifier=None
    if args.rule=='classifier':
        if args.classifier is None: raise ValueError('--classifier is required for classifier rule')
        ckpt=torch.load(args.classifier,map_location='cpu',weights_only=False)
        members=[]
        for state in ckpt['state_dicts']:
            model=_classifier_network(int(ckpt['input_dim']),int(ckpt['width']))
            model.load_state_dict(state);model.eval();members.append(model)
        class Ensemble:
            def __call__(self,x):
                probability=torch.stack([torch.sigmoid(m(x)) for m in members]).mean(0).clamp(1e-6,1-1e-6)
                return torch.logit(probability)
        classifier=Ensemble()
    wrapped=GripperPolicyHook(policy,args.rule,unit_grips,classifier)
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    WebsocketPolicyServer(wrapped,host='127.0.0.1',port=args.port,metadata={'policy':'memory_gripper_split','gripper_rule':args.rule}).serve_forever()


def wrap_gripper_policy(policy, rule, unit_grippers, classifier=None):
    """Construct the inference hook for reuse by a server module."""
    return GripperPolicyHook(policy, rule, unit_grippers, classifier)


def main():
    import argparse
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='cmd')
    p=sub.add_parser('offline');p.add_argument('--output',type=Path,default=OUT);p.add_argument('--report-only',action='store_true')
    s=sub.add_parser('serve');s.add_argument('--checkpoint',type=Path,required=True);s.add_argument('--pool',type=Path,required=True);s.add_argument('--query',type=Path,required=True);s.add_argument('--rule',choices=['unit','previous','g1','pi05','classifier'],default='pi05');s.add_argument('--classifier',type=Path);s.add_argument('--port',type=int,required=True)
    a=ap.parse_args()
    if a.cmd=='serve': serve_hook(a);return
    out=OUT if a.cmd is None else a.output
    report_only=False if a.cmd is None else a.report_only
    if not report_only: run_offline(out)
    write_report(out)
    print(out/'report.md')

if __name__=='__main__': main()
