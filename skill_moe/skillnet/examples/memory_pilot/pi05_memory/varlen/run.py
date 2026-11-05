#!/usr/bin/env python3
"""Offline variable-duration action-segment retrieval for pi05-memory40.

Segments are cut on frame-level gripper transitions, optionally augmented with
low-velocity minima. Segment keys use the frozen retrieval-4b wrist/state
contrastive embedding. No expert training is performed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PI05 = Path(__file__).resolve().parents[1]
RUN = Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/run-01')
GAP = Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/memory-gap-01')
RETRIEVAL = Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/retrieval-4b-01')
TOPK = Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/topk-long-01/production/predictions_and_errors.npz')
OUT = Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/varlen-01')
NOISE_SEED = 4101
T0S = (0.2, 0.4)
sys.path.insert(0, str(PI05))


def load_rows(run: Path = RUN) -> dict[str, np.ndarray]:
    return {p.stem: np.load(p, mmap_mode='r', allow_pickle=False)
            for p in (run / 'rows').glob('*.npy')}


def smooth(x: np.ndarray, width: int = 5) -> np.ndarray:
    x = np.asarray(x, np.float64)
    if width <= 1 or len(x) < width:
        return x.copy()
    pad = width // 2
    xp = np.pad(x, (pad, width - 1 - pad), mode='edge')
    return np.convolve(xp, np.ones(width) / width, mode='valid')


def _merge_boundaries(candidates: list[int], length: int, min_len: int = 10) -> np.ndarray:
    points = [0] + sorted(set(int(np.clip(x, 1, length - 1)) for x in candidates)) + [length]
    kept = [0]
    for x in points[1:-1]:
        if x - kept[-1] >= min_len and length - x >= min_len:
            kept.append(x)
    kept.append(length)
    return np.asarray(kept, np.int32)


def segment_boundaries(raw_actions: np.ndarray, mode: str, *, velocity_threshold: float | None = None,
                       eef_position: np.ndarray | None = None, min_len: int = 10) -> np.ndarray:
    """Return frame boundaries, including 0/end; raw actions are unnormalized."""
    a = np.asarray(raw_actions, np.float32)
    if a.ndim != 2 or a.shape[1] < 7 or len(a) < min_len:
        raise ValueError(f'expected frame actions [T,>=7], got {a.shape}')
    if mode not in ('grip', 'grip_vel', 'fixed10'):
        raise ValueError(mode)
    if mode == 'fixed10':
        vals = list(range(0, len(a), 10))
        if not vals or vals[-1] != len(a):
            vals.append(len(a))
        return np.asarray(vals, np.int32)
    grip = np.where(a[:, 6] >= 0, 1, -1)
    switches = (np.flatnonzero(grip[1:] != grip[:-1]) + 1).tolist()
    candidates = list(switches)
    if mode == 'grip_vel':
        if velocity_threshold is None or not np.isfinite(velocity_threshold) or eef_position is None:
            raise ValueError('grip_vel requires Dev threshold and measured EEF positions')
        pos = np.asarray(eef_position, np.float32)
        if pos.shape != (len(a), 3):
            raise ValueError(f'expected EEF positions [{len(a)},3], got {pos.shape}')
        speed = np.linalg.norm(np.diff(pos, axis=0, prepend=pos[:1]), axis=1) * 10.0
        speed = smooth(speed, width=5)
        minima = np.flatnonzero((speed[1:-1] <= speed[:-2]) & (speed[1:-1] < speed[2:]) &
                                (speed[1:-1] <= velocity_threshold)) + 1
        candidates.extend(int(x) for x in minima if min_len <= x <= len(a) - min_len)
    return _merge_boundaries(candidates, len(a), min_len=min_len)


def fit_velocity_threshold(dev_raw_actions: list[np.ndarray], dev_positions: list[np.ndarray], *,
                           min_len: int = 10) -> tuple[float, dict]:
    """Select a low-speed cutoff using non-Long Dev episodes only.

    The precommitted target is median segment duration 2 seconds at 10 Hz,
    with the central mass in the requested 1--4 second range.
    """
    if len(dev_raw_actions) != len(dev_positions):
        raise ValueError('Dev action/position episode counts differ')
    all_speed = np.concatenate([smooth(np.linalg.norm(np.diff(p, axis=0, prepend=p[:1]), axis=1) * 10., 5)
                                for p in dev_positions])
    candidates = []
    for q in (.05, .1, .15, .2, .25, .3, .4, .5):
        threshold = float(np.quantile(all_speed, q))
        lengths = np.concatenate([
            np.diff(segment_boundaries(x, 'grip_vel', velocity_threshold=threshold,
                                       eef_position=p, min_len=min_len))
            for x, p in zip(dev_raw_actions, dev_positions, strict=True)
        ])
        median = float(np.median(lengths))
        in_band = float(np.mean((lengths >= 10) & (lengths <= 40)))
        # Prefer the target median, then coverage of 1--4 s, then fewer cut points.
        score = abs(median - 20.) - 10. * in_band + .001 * len(lengths)
        candidates.append({'quantile': q, 'threshold': threshold, 'median_frames': median,
                           'fraction_1_to_4s': in_band, 'segments': int(len(lengths)), 'score': score})
    chosen = min(candidates, key=lambda x: (x['score'], x['quantile']))
    return float(chosen['threshold']), {'candidates': candidates, 'selected': chosen,
                                        'target_median_frames': 20, 'dev_episodes': len(dev_raw_actions)}


def extract_wrist(tokens: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Mean pooled wrist-camera prefix tokens, matching retrieval-4b input."""
    t, g = np.asarray(tokens), np.asarray(groups)
    if t.ndim != 3 or g.shape != t.shape[:2]:
        raise ValueError('token/group shapes differ')
    mask = g == 1  # prefix group IDs: base=0, wrist=1, language=2
    denom = mask.sum(1, keepdims=True)
    if np.any(denom == 0):
        raise ValueError('no wrist tokens in one or more observations')
    return ((t.astype(np.float32) * mask[..., None]).sum(1) / denom).astype(np.float32)


def draft_from_segment(segment_actions: np.ndarray, elapsed_frames: int, query_len: int,
                       alignment: str) -> np.ndarray:
    """Take one 10-frame draft; clamp exhausted segments to their final command."""
    x = np.asarray(segment_actions, np.float32)
    if x.ndim != 2 or x.shape[1] != 32 or len(x) < 1:
        raise ValueError(f'invalid variable segment action {x.shape}')
    if alignment == 'elapsed':
        start = int(elapsed_frames)
    elif alignment == 'phase':
        start = int(round(float(elapsed_frames) / max(int(query_len), 1) * len(x)))
    else:
        raise ValueError(alignment)
    ids = np.minimum(np.arange(start, start + 10, dtype=np.int64), len(x) - 1)
    return x[ids].copy()


def _load_raw_episode(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=['actions', 'state'])
    actions = np.asarray(t['actions'].to_pylist(), np.float32)
    state = np.asarray(t['state'].to_pylist(), np.float32)
    if state.ndim != 2 or state.shape[1] < 3:
        raise AssertionError(f'LIBERO state must contain EEF xyz, got {state.shape}')
    return actions, state[:, :3]


def _episode_action_sequence(rows: dict, row_ids: np.ndarray) -> np.ndarray:
    order = row_ids[np.argsort(np.asarray(rows['frame'])[row_ids])]
    if not len(order):
        raise ValueError('episode has no row windows')
    frames = np.asarray(rows['frame'])[order]
    if frames[0] != 0 or np.any(np.diff(frames) != 10):
        raise AssertionError('run-01 episode rows are not contiguous stride-10 windows')
    return np.asarray(rows['actions'][order], np.float32).reshape(-1, 32)


def key_observation_index(row_frames: np.ndarray, segment_start_frame: int) -> int:
    """First cached observation at/after the boundary, matching the query-time key."""
    frames = np.asarray(row_frames, np.int64)
    if frames.ndim != 1 or not len(frames):
        raise ValueError('row_frames must be a nonempty vector')
    return int(np.clip(np.searchsorted(frames, int(segment_start_frame), side='left'), 0, len(frames)-1))


def _segment_episode(rows: dict, episode_row_ids: np.ndarray, raw_actions: np.ndarray,
                     eef_position: np.ndarray,
                     mode: str, threshold: float | None) -> list[dict]:
    ordered = episode_row_ids[np.argsort(np.asarray(rows['frame'])[episode_row_ids])]
    norm_seq = _episode_action_sequence(rows, ordered)
    covered = min(len(norm_seq), len(raw_actions), len(eef_position))
    if covered < 10:
        return []
    norm_seq, raw, position = norm_seq[:covered], raw_actions[:covered], eef_position[:covered]
    boundaries = segment_boundaries(raw, mode, velocity_threshold=threshold,
                                    eef_position=position if mode == 'grip_vel' else None)
    task, episode = int(rows['task'][ordered[0]]), int(rows['episode'][ordered[0]])
    starts_at = np.asarray(rows['frame'])[ordered]
    out = []
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        if end - start < 10:
            continue
        key_ix = key_observation_index(starts_at, int(start))
        out.append({'task': task, 'episode': episode, 'start': int(start), 'end': int(end),
                    'start_row': int(ordered[key_ix]), 'actions': norm_seq[start:end].copy()})
    return out


def _bootstrap_ci(a: np.ndarray, b: np.ndarray, episodes: np.ndarray, *, seed: int = 4101,
                  reps: int = 5000) -> list[float]:
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    ep = np.asarray(episodes)
    groups = np.unique(ep)
    sums = np.asarray([d[ep == e].sum() for e in groups])
    counts = np.asarray([(ep == e).sum() for e in groups])
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, np.float64)
    for st in range(0, reps, 256):
        en = min(st + 256, reps)
        ix = rng.integers(0, len(groups), size=(en - st, len(groups)))
        draws[st:en] = sums[ix].sum(1) / counts[ix].sum(1)
    return np.quantile(draws, [.025, .975]).tolist()


def _save_json(path: Path, obj: dict):
    def serializable(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, dict):
            return {str(k): serializable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [serializable(v) for v in x]
        return x
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(serializable(obj), ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def prepare(out: Path, *, run: Path = RUN, gap: Path = GAP, retrieval: Path = RETRIEVAL):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'refusing to overwrite nonempty output: {out}')
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(run)
    gap_npz = np.load(gap / 'predictions_and_errors.npz', allow_pickle=False)
    query_rows = np.asarray(gap_npz['row_ids'], np.int64)
    if len(query_rows) != 1318:
        raise AssertionError(f'memory-gap Eval contract changed: {len(query_rows)}')
    long_rows = query_rows[np.asarray(rows['task'][query_rows]) < 10]
    if len(long_rows) != 548 or set(np.asarray(rows['task'][long_rows]).tolist()) != set(range(10)):
        raise AssertionError('expected 548 Long Eval windows from tasks 0..9')
    hard_source = np.load(TOPK, allow_pickle=False)
    if not np.array_equal(hard_source['query_rows'], long_rows):
        raise AssertionError('topk-long query ordering differs from memory-gap Long rows')
    hard = np.asarray(hard_source['hard_mask'], bool)
    if hard.sum() != 63:
        raise AssertionError(f'expected fixed 63 hard rows; found {hard.sum()}')

    # Eval episodes and noLong memory are formed by episode identity; strict leak assertions.
    eval_episode_keys = {(int(rows['task'][r]), int(rows['episode'][r])) for r in long_rows}
    memory_ids = np.flatnonzero(np.asarray(rows['memory'], bool) & (np.asarray(rows['task']) >= 10))
    memory_episode_keys = {(int(rows['task'][r]), int(rows['episode'][r])) for r in memory_ids}
    assert not (memory_episode_keys & eval_episode_keys), 'Long Eval episode leaked into memory'
    assert np.all(np.asarray(rows['task'])[memory_ids] >= 10), 'Long task rows leaked into memory'
    assert not np.isin(long_rows, memory_ids).any(), 'Long Eval rows leaked into memory'

    # Read run-01's exact source episode files for raw gripper/velocity boundary signals.
    manifest = json.loads((run / 'data_manifest.json').read_text())
    source_by_key = {(int(x['task']), int(x['episode'])): Path(x['path']) for x in manifest['episodes']}
    all_needed_keys = memory_episode_keys | eval_episode_keys
    dev_episode_ids = set()
    policy_mod = __import__('importlib.util').util
    import importlib.util
    spec = importlib.util.spec_from_file_location('varlen_pi05_run', PI05 / 'run.py')
    runmod = importlib.util.module_from_spec(spec); spec.loader.exec_module(runmod)
    dev_ids, _ = runmod.calibration_episodes(rows)
    dev_episode_ids = set(map(int, dev_ids))
    # Correctly select distinct validation Dev episodes for task>=10.
    dev_keys = set()
    for r in np.flatnonzero(np.asarray(rows['validation'], bool) & (np.asarray(rows['task']) >= 10) &
                            np.isin(rows['episode'], np.asarray(list(dev_episode_ids), np.int64))):
        dev_keys.add((int(rows['task'][r]), int(rows['episode'][r])))
    if not dev_keys:
        raise AssertionError('no non-Long Dev episodes available for velocity calibration')
    needed = all_needed_keys | dev_keys
    raw_by_key = {}
    for ix, key in enumerate(sorted(needed)):
        if key not in source_by_key:
            raise AssertionError(f'missing manifest parquet for episode {key}')
        raw_by_key[key] = _load_raw_episode(source_by_key[key])
        if ix % 50 == 0:
            print(json.dumps({'stage': 'read_source', 'episodes': ix + 1, 'total': len(needed)}), flush=True)
    dev_raw = [raw_by_key[k][0] for k in sorted(dev_keys)]
    dev_pos = [raw_by_key[k][1] for k in sorted(dev_keys)]
    velocity_threshold, threshold_meta = fit_velocity_threshold(dev_raw, dev_pos)
    threshold_meta['dev_episode_keys'] = [list(k) for k in sorted(dev_keys)]

    # Reuse the already-trained retrieval-4b wrist/state contrastive projection; no fitting here.
    embedding = np.load(retrieval / 'wrist_state_embedding.npy', mmap_mode='r', allow_pickle=False)
    if embedding.shape != (len(rows['task']), 64):
        raise AssertionError(f'contrastive embedding shape mismatch: {embedding.shape}')
    emb = np.array(embedding, dtype=np.float32, copy=True)
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
    wrist = np.load(retrieval / 'wrist_token_mean.npy', mmap_mode='r', allow_pickle=False)
    if len(wrist) != len(rows['task']):
        raise AssertionError('wrist token cache not aligned with run-01 rows')

    def episode_groups(row_ids):
        out = {}
        for rid in row_ids:
            key = (int(rows['task'][rid]), int(rows['episode'][rid]))
            out.setdefault(key, []).append(int(rid))
        return out

    mem_groups = episode_groups(memory_ids)
    eval_groups = episode_groups(long_rows)
    dev_groups = episode_groups(np.flatnonzero(np.asarray(rows['validation'], bool) &
                                               (np.asarray(rows['task']) >= 10) &
                                               np.isin(rows['episode'], np.asarray(list(dev_episode_ids), np.int64))))
    memory_segments, eval_segments = {}, {}
    for mode in ('grip', 'grip_vel'):
        memory_segments[mode] = []
        eval_segments[mode] = []
        for key, ids in sorted(mem_groups.items()):
            raw, pos = raw_by_key[key]
            memory_segments[mode].extend(_segment_episode(rows, np.asarray(ids), raw, pos, mode,
                                                          velocity_threshold if mode == 'grip_vel' else None))
        for key, ids in sorted(eval_groups.items()):
            if key in memory_episode_keys:
                raise AssertionError('Long Eval episode overlaps noLong memory')
            raw, pos = raw_by_key[key]
            eval_segments[mode].extend(_segment_episode(rows, np.asarray(ids), raw, pos, mode,
                                                        velocity_threshold if mode == 'grip_vel' else None))
    # Fixed 10-step rows are represented as the existing memory rows, retaining their query-time keys.
    fixed_memory = [{'task': int(rows['task'][r]), 'episode': int(rows['episode'][r]),
                     'start': int(rows['frame'][r]), 'end': int(rows['frame'][r]) + 10,
                     'start_row': int(r), 'actions': np.asarray(rows['actions'][r], np.float32)}
                    for r in memory_ids]

    # Eval segment memberships: only the 548 frozen Long query rows are scored.
    qrow_set = set(map(int, long_rows))
    eval_members = {}
    for mode, segments in eval_segments.items():
        members = [[] for _ in segments]
        for qi, rid in enumerate(long_rows):
            task, ep, frame = int(rows['task'][rid]), int(rows['episode'][rid]), int(rows['frame'][rid])
            matches = [j for j, s in enumerate(segments) if s['task'] == task and s['episode'] == ep and s['start'] <= frame < s['end']]
            if len(matches) != 1:
                raise AssertionError(f'query row {rid} maps to {len(matches)} {mode} segments')
            members[matches[0]].append(int(rid))
        eval_members[mode] = members

    # Candidate banks store flat variable-length actions and offsets, plus start-row retrieval keys.
    pools = {'grip': memory_segments['grip'], 'grip_vel': memory_segments['grip_vel'],
             'fixed10': fixed_memory}
    banks, bank_meta = {}, {}
    for mode, segs in pools.items():
        offsets = [0]; acts = []; start_rows = []
        for s in segs:
            acts.append(np.asarray(s['actions'], np.float32)); offsets.append(offsets[-1] + len(s['actions']))
            start_rows.append(int(s['start_row']))
        bank_act = np.concatenate(acts, axis=0) if acts else np.zeros((0, 32), np.float32)
        start_rows = np.asarray(start_rows, np.int64)
        if np.any(np.asarray(rows['task'])[start_rows] < 10):
            raise AssertionError('Long task/Eval row present in memory candidate bank')
        banks[mode] = {'actions': bank_act, 'offsets': np.asarray(offsets, np.int64),
                       'start_rows': start_rows, 'key': emb[start_rows]}
        bank_meta[mode] = {'segments': len(segs), 'rows': len(bank_act),
                           'cross_task_nearest_action_mse': _cross_task_similarity(segs)}

    # One retrieval decision at each query-segment start; then draft continuation by elapsed time/phase.
    drafts, decisions = {}, {}
    for mode in ('grip', 'grip_vel'):
        segs, members = eval_segments[mode], eval_members[mode]
        for alignment in ('elapsed', 'phase'):
            d = np.zeros((len(long_rows), 10, 32), np.float32)
            decision_rows = np.full(len(long_rows), -1, np.int64)
            selected_ids = np.full(len(long_rows), -1, np.int64)
            selected_scores = np.full(len(long_rows), np.nan, np.float32)
            oracle_ids = np.full(len(long_rows), -1, np.int64)
            for si, qrows_in_segment in enumerate(members):
                if not qrows_in_segment:
                    continue
                qseg = segs[si]
                qrows_arr = np.asarray(qrows_in_segment, np.int64)
                qix = np.asarray([int(np.flatnonzero(long_rows == q)[0]) for q in qrows_arr], np.int64)
                first_q = qrows_arr[0]
                qkey = emb[first_q]
                sims = banks[mode]['key'] @ qkey
                retrieved = int(np.argmax(sims))
                selected_ids[qix] = retrieved
                selected_scores[qix] = float(sims[retrieved])
                decision_rows[qix] = int(first_q)
                for loc, qi in zip(qrows_arr, qix, strict=True):
                    elapsed = int(rows['frame'][loc]) - int(qseg['start'])
                    d[qi] = draft_from_segment(_bank_segment(banks[mode], retrieved), elapsed,
                                               int(qseg['end'] - qseg['start']), alignment)
                # Privileged segment oracle: choose one memory segment minimizing aggregate action MSE
                # over all scored rows in this query segment; actions do not affect ordinary retrieval.
                truth = np.asarray(rows['actions'][qrows_arr], np.float32)
                best_id, best_err = None, np.inf
                for ci in range(len(segs_memory := memory_segments[mode])):
                    pred = []
                    candidate = segs_memory[ci]
                    for loc in qrows_arr:
                        elapsed = int(rows['frame'][loc]) - int(qseg['start'])
                        pred.append(draft_from_segment(candidate['actions'], elapsed,
                                                        int(qseg['end'] - qseg['start']), alignment))
                    err = float(np.mean((np.asarray(pred)[:, :, :7] - truth[:, :, :7]) ** 2))
                    if err < best_err:
                        best_err, best_id = err, ci
                oracle_ids[qix] = int(best_id)
            if np.any(selected_ids < 0) or np.any(oracle_ids < 0):
                raise AssertionError(f'{mode}/{alignment}: some eval rows have no segment selection')
            drafts[f'{mode}_{alignment}'] = d
            decisions[f'{mode}_{alignment}'] = {'query_rows': long_rows.tolist(),
                'selection_start_query_row': decision_rows, 'selected_segment_index': selected_ids,
                'selected_cosine': selected_scores, 'oracle_segment_index': oracle_ids}
    # Fixed 10-step Top-1 with the same frozen retrieval-4b embedding and cosine,
    # re-ranked over the required noLong memory pool (the published all40 list
    # can contain other Long tasks and therefore cannot be reused verbatim).
    fixed_mids = memory_ids
    cand = np.empty(len(long_rows), np.int64)
    row_ids = np.arange(len(rows['task']), dtype=np.int64)
    for qi, qrow in enumerate(long_rows):
        allowed = fixed_mids[np.asarray(rows['task'])[fixed_mids] != int(rows['task'][qrow])]
        score = emb[allowed] @ emb[qrow]
        cand[qi] = allowed[np.lexsort((allowed, -score))[0]]
    if not np.all(np.asarray(rows['memory'], bool)[cand]) or np.any(np.asarray(rows['task'])[cand] < 10):
        raise AssertionError('fixed retrieval top1 violates noLong memory pool')
    drafts['fixed10_reselect'] = np.asarray(rows['actions'][cand], np.float32)
    drafts['zero'] = np.zeros((len(long_rows), 10, 32), np.float32)
    actions = np.asarray(rows['actions'][long_rows], np.float32)
    pre_mse = {name: np.mean((value[:, :, :7] - actions[:, :, :7]) ** 2, axis=(1, 2))
               for name, value in drafts.items()}

    # Oracle drafts for variable segments (created above as indexes) are expanded now.
    for mode in ('grip', 'grip_vel'):
        for alignment in ('elapsed', 'phase'):
            name = f'{mode}_{alignment}'
            oracle_draft = np.zeros_like(drafts[name])
            idxs = decisions[name]['oracle_segment_index']
            for qi, segid in enumerate(idxs):
                qrow = int(long_rows[qi]); qseg_id = next(j for j, mem in enumerate(eval_members[mode]) if qrow in mem)
                qseg = eval_segments[mode][qseg_id]
                elapsed = int(rows['frame'][qrow]) - int(qseg['start'])
                oracle_draft[qi] = draft_from_segment(memory_segments[mode][int(segid)]['actions'], elapsed,
                                                       int(qseg['end']-qseg['start']), alignment)
            drafts[name + '_oracle'] = oracle_draft
            pre_mse[name + '_oracle'] = np.mean((oracle_draft[:, :, :7] - actions[:, :, :7]) ** 2, axis=(1, 2))

    # Segment-level statistics, per-mode length and cross-task action similarity.
    stats = {}
    for mode in ('grip', 'grip_vel', 'fixed10'):
        segs = pools[mode]
        lengths = np.asarray([s['end'] - s['start'] for s in segs], np.float32)
        stats[mode] = {'segments': len(segs), 'length_frames': {
            'p10': float(np.quantile(lengths, .1)), 'median': float(np.median(lengths)),
            'p90': float(np.quantile(lengths, .9)), 'mean': float(lengths.mean()),
            'min': int(lengths.min()), 'max': int(lengths.max())},
            'length_seconds_10hz': {'median': float(np.median(lengths) / 10),
                                    'p10': float(np.quantile(lengths, .1) / 10),
                                    'p90': float(np.quantile(lengths, .9) / 10)},
            **bank_meta[mode]}
    eval_stats = {}
    for mode in ('grip', 'grip_vel'):
        eval_lens = np.asarray([s['end'] - s['start'] for s in eval_segments[mode]], np.float32)
        eval_stats[mode] = {'segments': len(eval_lens), 'query_rows_covered': int(len(long_rows)),
            'length_seconds_10hz': {'p10': float(np.quantile(eval_lens, .1) / 10),
                'median': float(np.median(eval_lens) / 10), 'p90': float(np.quantile(eval_lens, .9) / 10),
                'mean': float(eval_lens.mean() / 10)}}

    # Save bank/predrafts/selection metadata without modifying source artifacts.
    segment_arrays = {f'{mode}_actions': bank['actions'] for mode, bank in banks.items()}
    segment_arrays.update({f'{mode}_offsets': bank['offsets'] for mode, bank in banks.items()})
    segment_arrays.update({f'{mode}_start_rows': bank['start_rows'] for mode, bank in banks.items()})
    segment_arrays.update({f'{mode}_keys': bank['key'] for mode, bank in banks.items()})
    for mode, pool in pools.items():
        segment_arrays[f'{mode}_task'] = np.asarray([s['task'] for s in pool], np.int32)
        segment_arrays[f'{mode}_episode'] = np.asarray([s['episode'] for s in pool], np.int32)
        segment_arrays[f'{mode}_start_frame'] = np.asarray([s['start'] for s in pool], np.int32)
        segment_arrays[f'{mode}_end_frame'] = np.asarray([s['end'] for s in pool], np.int32)
    for mode, pool in eval_segments.items():
        segment_arrays[f'long_eval_{mode}_task'] = np.asarray([s['task'] for s in pool], np.int32)
        segment_arrays[f'long_eval_{mode}_episode'] = np.asarray([s['episode'] for s in pool], np.int32)
        segment_arrays[f'long_eval_{mode}_start_frame'] = np.asarray([s['start'] for s in pool], np.int32)
        segment_arrays[f'long_eval_{mode}_end_frame'] = np.asarray([s['end'] for s in pool], np.int32)
    np.savez_compressed(out / 'segments.npz', **segment_arrays)
    np.savez_compressed(out / 'drafts_and_errors.npz', query_rows=long_rows, hard_mask=hard,
                        actions=actions, **{f'draft__{n}': x for n, x in drafts.items()},
                        **{f'pre_mse__{n}': x for n, x in pre_mse.items()})
    _save_json(out / 'selection.json', decisions)
    protocol = {'source_run': str(run), 'source_frames': manifest['source'],
        'query_rows': long_rows.tolist(), 'query_tasks': np.asarray(rows['task'][long_rows]).tolist(),
        'query_episodes': np.asarray(rows['episode'][long_rows]).tolist(),
        'long_eval_rows': len(long_rows), 'hard_rows': int(hard.sum()),
        'hard_threshold': 0.01549410422449, 'hard_mask_source': str(TOPK),
        'noLong_memory_rows': len(memory_ids), 'assertions': {'memory_task_ids_ge_10': True,
        'eval_episodes_excluded': True, 'all_memory_segments_from_memory_rows': True},
        'boundary_rules': {'grip': 'raw source parquet action[6] sign changes; minimum 10 frames',
        'grip_vel': 'gripper changes plus smoothed (5-frame) measured EEF xyz velocity norm local minima <= nonLong Dev-calibrated threshold; minimum 10 frames',
        'min_segment_frames': 10, 'frame_rate_hz': 10, 'velocity_calibration': threshold_meta},
        'keys': 'frozen retrieval-4b wrist_state_contrastive 64D embedding at segment start observation row; cosine; memory candidates are noLong memory episodes',
        'retrieval': 'one top-1 segment chosen per query segment and held for all query rows in it; no blending',
        'alignments': {'elapsed': 'source offset equals elapsed query frames', 'phase': 'source offset = round(query phase * source segment length)'},
        'candidate_exclusion': 'all memory candidates have task IDs 10..39; Long tasks 0..9 excluded',
        'fixed_control': 'retrieval-4b wrist_state_contrastive 64D cosine Top-1 per query row, re-ranked over noLong memory rows (published all40 candidate list can include other Long tasks)',
        'oracle': 'best one memory segment for all scored rows of a query segment, minimizing active7 draft MSE; privileged upper bound',
        'Expert': str(run / 'bridge/001500'), 'expert_steps': 1500,
        'bridge_t0': list(T0S), 'bridge_noise_seed': NOISE_SEED,
        'bridge_noise_rule': 'jax.random.normal(fold_in(key(4101), row_id), [10,32]); bridge key fold_in(key(4101), batch_start)',
        'GPU_generation': False}
    _save_json(out / 'protocol.json', protocol)
    summary = {'segment_stats': stats, 'long_eval_segment_stats': eval_stats,
        'velocity_calibration': threshold_meta,
        'pre_mse': {name: {'all': float(v.mean()), 'hard': float(v[hard].mean())}
                    for name, v in pre_mse.items()},
        'hard_rows': int(hard.sum()), 'long_rows': len(long_rows),
        'bootstrap': {'unit': 'episode', 'reps': 5000, 'seed': 4101}}
    _save_json(out / 'summary_pre.json', summary)
    return rows, long_rows, hard, drafts, actions, decisions, summary


def _bank_segment(bank: dict, index: int) -> np.ndarray:
    lo, hi = int(bank['offsets'][index]), int(bank['offsets'][index + 1])
    return bank['actions'][lo:hi]


def _cross_task_similarity(segments: list[dict]) -> dict:
    if len(segments) < 2:
        return {'median_nearest_cross_task_prefix_mse': None, 'fraction_nearest_le_0p01': None}
    prefix = np.asarray([s['actions'][:10, :7] for s in segments], np.float32).reshape(len(segments), -1)
    tasks = np.asarray([s['task'] for s in segments])
    values = []
    for i in range(len(segments)):
        allowed = tasks != tasks[i]
        if not allowed.any():
            continue
        d = np.mean((prefix[allowed] - prefix[i]) ** 2, axis=1)
        values.append(float(d.min()))
    if not values:
        return {'median_nearest_cross_task_prefix_mse': None, 'fraction_nearest_le_0p01': None}
    return {'median_nearest_cross_task_prefix_mse': float(np.median(values)),
            'fraction_nearest_le_0p01': float(np.mean(np.asarray(values) <= .01)),
            'definition': 'nearest different-task 10-frame prefix active7 MSE <= .01'}


def infer(out: Path, rows: dict, query_rows: np.ndarray, hard: np.ndarray, drafts: dict,
          actions: np.ndarray, *, batch_size: int = 4, mem_fraction: float = .20):
    import os
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', str(mem_fraction))
    if os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE', '').lower() != 'false':
        raise RuntimeError('XLA_PYTHON_CLIENT_PREALLOCATE must remain false')
    if float(os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION', mem_fraction)) > .20:
        raise RuntimeError('GPU memory fraction must be <= .20')
    import importlib.util
    spec = importlib.util.spec_from_file_location('pi05_varlen_topk', PI05 / 'topk_long/run.py')
    topk = importlib.util.module_from_spec(spec); spec.loader.exec_module(topk)
    jax, jnp, policy, base, baseline, bridge = topk._load_runtime()
    normal_ref = np.load(TOPK, allow_pickle=False)
    if not np.array_equal(normal_ref['query_rows'], query_rows):
        raise AssertionError('query order mismatch with topk-long controls')
    truth = actions[:, :, :7]
    conditions = ['zero', 'fixed10_reselect', 'grip_elapsed', 'grip_phase', 'grip_vel_elapsed', 'grip_vel_phase',
                  'grip_elapsed_oracle', 'grip_phase_oracle', 'grip_vel_elapsed_oracle', 'grip_vel_phase_oracle']
    conditions = [x for x in conditions if x in drafts]
    pred_by_t0 = {}
    started = time.monotonic()
    for t0 in T0S:
        cond_preds = {name: np.empty((len(query_rows), 10, 7), np.float32) for name in conditions}
        for begin in range(0, len(query_rows), batch_size):
            end = min(begin + batch_size, len(query_rows)); qids = query_rows[begin:end]
            obs = policy.obs_batch(rows, qids)
            dnames = conditions
            for cstart in range(0, len(dnames), 8):
                group = dnames[cstart:cstart + 8]
                db = np.stack([drafts[n][begin:end] for n in group], axis=1)
                noise_seeds = [NOISE_SEED] * len(group)
                pred = topk._sample_bridge(bridge, jax, jnp, policy, rows, qids, db,
                    t0, noise_seeds, begin, batch_size=batch_size, condition_batch=8)
                for ci, name in enumerate(group):
                    cond_preds[name][begin:end] = pred[:, ci]
            if end == len(query_rows) or begin % 100 == 0:
                print(json.dumps({'stage': 'bridge', 't0': t0, 'rows': end,
                                  'elapsed_s': time.monotonic() - started}), flush=True)
        pred_by_t0[t0] = cond_preds
    # Parity checks: frozen public single and zero draft controls are reused from topk-long.
    normal_err = np.asarray(normal_ref['normal_single_mse'], np.float32)
    zero_mse_ref = np.asarray(normal_ref['zero_mse'], np.float32)
    # Verify exact normal reference values against query truth.
    ref_truth = np.asarray(normal_ref['truth'], np.float32)
    assert np.array_equal(ref_truth, truth[:, :, :7])
    normal_pred = np.asarray(normal_ref['normal_single'], np.float32)
    normal_recomputed = np.mean((normal_pred - truth) ** 2, axis=(1, 2))
    if not np.allclose(normal_recomputed, normal_err, atol=1e-7, rtol=1e-5):
        raise AssertionError('normal single prediction and saved per-row error disagree')
    metrics = {'conditions': {}, 'paired_delta_vs_fixed': {}, 'normal_single_mse': float(normal_err.mean()),
               'normal_single_hard_mse': float(normal_err[hard].mean()),
               'zero_reference_mse': {str(t): float(zero_mse_ref[i, 0].mean())
                                      for i, t in enumerate(T0S)},
               'zero_parity_vs_topk_seed4101': {}}
    ep = np.asarray(rows['episode'][query_rows])
    allmask, hardmask = np.ones(len(query_rows), bool), np.asarray(hard, bool)
    predictions = {}
    zero_ref_predictions = np.asarray(normal_ref['zero_predictions'], np.float32)
    for ti, (t0, values) in enumerate(pred_by_t0.items()):
        # The existing topk-long control used the same first zero-draft seed (4101).
        our_zero_mse = np.mean((values['zero'] - truth) ** 2, axis=(1, 2))
        saved_zero_mse = np.mean((zero_ref_predictions[ti, 0] - truth) ** 2, axis=(1, 2))
        metrics['zero_parity_vs_topk_seed4101'][str(t0)] = {
            'max_abs_row_mse_difference': float(np.max(np.abs(our_zero_mse-saved_zero_mse))),
            'mean_abs_row_mse_difference': float(np.mean(np.abs(our_zero_mse-saved_zero_mse))),
            'parity_exact': bool(np.allclose(our_zero_mse, saved_zero_mse, atol=2e-6, rtol=1e-5))}
        for name, pred in values.items():
            err = np.mean((pred - truth) ** 2, axis=(1, 2))
            predictions[f'{name}_t0_{t0:.1f}'] = pred
            rec = metrics['conditions'].setdefault(name, {})
            rec[f't0_{t0:.1f}'] = {'all_mse': float(err.mean()), 'hard_mse': float(err[hardmask].mean()),
                'draft_all_mse': float(np.mean((drafts[name][:, :, :7] - truth) ** 2)),
                'draft_hard_mse': float(np.mean((drafts[name][hardmask, :, :7] - truth[hardmask]) ** 2))}
            if name != 'fixed10_reselect':
                fixed = np.mean((values['fixed10_reselect'] - truth) ** 2, axis=(1, 2))
                metrics['paired_delta_vs_fixed'][f'{name}_t0_{t0:.1f}'] = {
                    'all_mean_delta': float((err - fixed).mean()),
                    'all_ci95': _bootstrap_ci(err, fixed, ep),
                    'hard_mean_delta': float((err[hardmask] - fixed[hardmask]).mean()),
                    'hard_ci95': _bootstrap_ci(err[hardmask], fixed[hardmask], ep[hardmask])}
            metrics.setdefault('paired_delta_vs_normal', {})[f'{name}_t0_{t0:.1f}'] = {
                'all_mean_delta': float((err-normal_err).mean()),
                'all_ci95': _bootstrap_ci(err, normal_err, ep),
                'hard_mean_delta': float((err[hardmask]-normal_err[hardmask]).mean()),
                'hard_ci95': _bootstrap_ci(err[hardmask], normal_err[hardmask], ep[hardmask])}
    # Existing normal/zero bridge output reference consistency (MSE parity, tolerance for device kernels).
    # normal single is exactly saved; zero t0 values are summarized as references rather than re-run.
    np.savez_compressed(out / 'bridge_predictions.npz', query_rows=query_rows, hard_mask=hardmask,
        truth=truth, **{f'pred__{k}': v for k, v in predictions.items()},
        normal_single_mse=normal_err, topk_zero_mse=np.asarray(zero_mse_ref))
    _save_json(out / 'summary.json', metrics)
    with (out / 'progress.jsonl').open('a') as f:
        f.write(json.dumps({'stage': 'bridge_done', 'seconds': time.monotonic() - started,
                            'gpu_mem_fraction': mem_fraction}) + '\n')
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=OUT)
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--infer-prepared', action='store_true',
                    help='run only the frozen bridge on an existing prepared output directory')
    ap.add_argument('--skip-gpu', action='store_true')
    ap.add_argument('--mem-fraction', type=float, default=.20)
    a = ap.parse_args()
    if a.infer_prepared:
        rows = load_rows()
        z = np.load(a.output / 'drafts_and_errors.npz', allow_pickle=False)
        query_rows = np.asarray(z['query_rows'], np.int64)
        hard = np.asarray(z['hard_mask'], bool)
        actions = np.asarray(z['actions'], np.float32)
        drafts = {k[len('draft__'):]: np.asarray(z[k], np.float32)
                  for k in z.files if k.startswith('draft__')}
        infer(a.output, rows, query_rows, hard, drafts, actions, mem_fraction=a.mem_fraction)
        (a.output / 'DONE').write_text('done\n')
        print(json.dumps({'output': str(a.output), 'long_rows': len(query_rows),
                          'hard_rows': int(hard.sum()), 'gpu': True}), flush=True)
        return
    result = prepare(a.output)
    rows, qrows, hard, drafts, actions, decisions, pre = result
    if not a.prepare_only and not a.skip_gpu:
        infer(a.output, rows, qrows, hard, drafts, actions, mem_fraction=a.mem_fraction)
    if not a.prepare_only and not a.skip_gpu:
        (a.output / 'DONE').write_text('done\n')
    print(json.dumps({'output': str(a.output), 'long_rows': len(qrows), 'hard_rows': int(hard.sum()),
                      'gpu': not a.skip_gpu and not a.prepare_only}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
