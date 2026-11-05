#!/usr/bin/env python3
"""Privileged simulator-situation retrieval ceiling diagnostic (CPU only)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import time

import h5py
import numpy as np
import pyarrow.parquet as pq

from core import (align_subsequence, assert_nonlong_roles, choose_action_aligned_demo, combine_blocks,
                  normalize_block, topk_row_ids)

import os as _os
_REPO = Path(__file__).resolve().parents[6]
_STATE_DIR = Path(_os.environ.get('SKILLNET_STATE_DIR', _REPO / '.skillnet-state'))
_PI05_ROOT = Path(_os.environ.get('SKILLNET_PI05_ROOT', _STATE_DIR / 'outputs/memory-pilot/pi05-memory40-20261104'))
_LIBERO_HDF5 = Path(_os.environ.get('SKILLNET_LIBERO_HDF5', _REPO / 'data/LIBERO'))
ROOT = _PI05_ROOT
RUN = ROOT / 'run-01'
LOCAL = ROOT / 'local-key-02'
MEMORY = ROOT / 'skill-memories-01' / 'raw-rows-noLong'
DEFAULT_OUT = ROOT / 'situation-oracle-01'
H5_ROOT = _LIBERO_HDF5
NORM = _STATE_DIR / 'checkpoints/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json'
SUITES = ((0, 10, 'libero_10'), (10, 20, 'libero_goal'),
          (20, 30, 'libero_object'), (30, 40, 'libero_spatial'))
SEED = 20261105
BOOTSTRAP_REPS = 5000


def jwrite(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def load_rows():
    return {p.stem: np.load(p, mmap_mode='r', allow_pickle=False)
            for p in (RUN / 'rows').glob('*.npy')}


def calibration_dev_episodes(rows):
    dev, evaluation = [], []
    for task in np.unique(rows['task']):
        if int(task) < 10:
            continue
        episodes = np.unique(rows['episode'][(rows['task'] == task) & rows['validation']])
        order = np.random.default_rng(SEED + int(task)).permutation(episodes)
        dev.extend(order[:3].tolist())
        evaluation.extend(order[3:].tolist())
    if set(dev) & set(evaluation):
        raise AssertionError('calibration dev/eval episodes overlap')
    return np.asarray(dev, np.int64), np.asarray(evaluation, np.int64)


def suite_name(task_id):
    for lo, hi, name in SUITES:
        if lo <= int(task_id) < hi:
            return name
    raise ValueError(task_id)


def instruction_slug(text):
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


def load_task_hdf5_files(tasks):
    """Match HDF5 tasks by their language instruction, never by filename guess."""
    result = {}
    for task_id, instruction in tasks.items():
        suite = suite_name(int(task_id))
        folder = H5_ROOT / suite
        hits = []
        for path in folder.glob('*.hdf5'):
            with h5py.File(path, 'r') as f:
                info = json.loads(f['data'].attrs['problem_info'])
            if info['language_instruction'].strip().lower() == instruction.strip().lower():
                hits.append(path)
        if len(hits) != 1:
            raise ValueError(f'task {task_id}: expected one HDF5 by instruction, found {hits}')
        result[int(task_id)] = hits[0]
    return result


def load_hdf5_demo_actions(path):
    with h5py.File(path, 'r') as h:
        group = h['data']
        demos = {str(name): np.asarray(group[name]['actions'], np.float32)
                 for name in sorted(group, key=lambda s: int(s.split('_')[-1]))}
    return demos


def norm_actions(actions, q01, q99):
    x = np.asarray(actions, np.float32).astype(np.float64)
    # Match openpi.transforms.Normalize._normalize_quantile exactly; it does
    # not clip after quantile scaling.
    return (((x - q01) / (q99 - q01 + 1e-6)) * 2.0 - 1.0).astype(np.float32)


def find_episode_record(manifest):
    by_ep = {}
    for rec in manifest['episodes']:
        ep = int(rec['episode'])
        if ep in by_ep:
            raise AssertionError(f'duplicate LeRobot episode index {ep}')
        by_ep[ep] = rec
    return by_ep


def _action_match_for_episode(ep_idx, rec, task_id, rows, rowids, task_h5, normstats, demo_cache):
    table = pq.read_table(rec['path'], columns=['actions', 'source_episode_index', 'episode_index', 'task_index'])
    cols = table.to_pydict()
    if len(set(map(int, cols['source_episode_index']))) != 1:
        raise AssertionError(f'episode {ep_idx}: source_episode_index is not constant')
    if len(set(map(int, cols['episode_index']))) != 1 or int(cols['episode_index'][0]) != ep_idx:
        raise AssertionError(f'episode id mismatch {ep_idx} vs parquet')
    if len(set(map(int, cols['task_index']))) != 1 or int(cols['task_index'][0]) != task_id:
        raise AssertionError(f'task mismatch for episode {ep_idx}')
    src_actions = np.asarray(cols['actions'], np.float32)
    if task_id not in demo_cache:
        demo_cache[task_id] = load_hdf5_demo_actions(task_h5[task_id])
    demo_name, action_map = choose_action_aligned_demo(src_actions, demo_cache[task_id])
    q01, q99 = normstats
    wanted = set(map(int, rowids))
    row_frame_to_id = {int(rows['frame'][r]): int(r) for r in rowids}
    with h5py.File(task_h5[task_id], 'r') as h:
        demo = h['data'][demo_name]
        h5_actions = np.asarray(demo['actions'], np.float32)
        h5_states = demo['states']
        if len(action_map) != len(src_actions) or np.any(action_map < 0):
            raise AssertionError('incomplete source-to-HDF5 frame mapping')
        # Verify every selected 10-step active7 action window through the same
        # quantile normalization used by the cached policy rows.
        for start, rid in row_frame_to_id.items():
            if start < 0 or start + 10 > len(action_map):
                raise AssertionError(f'row {rid}: action window outside source episode')
            h5_ix = action_map[start:start + 10]
            source_window = np.asarray(cols['actions'][start:start + 10], np.float32)
            original_window = h5_actions[h5_ix]
            if not np.allclose(source_window, original_window, atol=1e-6, rtol=0):
                err = float(np.max(np.abs(source_window - original_window)))
                raise AssertionError(f'row {rid}: raw HDF5 action window mismatch max={err}')
            # Apply the same transform to the exact float32 source used when
            # preparing rows. Casting HDF5 float64 values to float32 can be
            # amplified by narrow quantile ranges; the raw HDF5/source check
            # above is the authoritative source-match assertion.
            target = norm_actions(source_window, q01, q99)
            cached = np.asarray(rows['actions'][rid, :, :7], np.float32)
            if not np.allclose(target, cached, atol=1e-5, rtol=0):
                err = float(np.max(np.abs(target - cached)))
                raise AssertionError(f'row {rid}: normalized 10-step HDF5 actions mismatch max={err}')
        states = np.asarray(h5_states[action_map[np.asarray(sorted(row_frame_to_id))]], np.float64)
    return int(cols['source_episode_index'][0]), demo_name, action_map, states


def _rotvec(mat):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(np.asarray(mat, np.float64).reshape(3, 3)).as_rotvec().astype(np.float32)


def _build_env(task_text):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    from pathlib import Path
    found = []
    bdict = benchmark.get_benchmark_dict()
    for _, _, suite_name_ in SUITES:
        suite = bdict[suite_name_]()
        for i in range(suite.n_tasks):
            task = suite.get_task(i)
            if task.language.strip().lower() == task_text.strip().lower():
                found.append(task)
    if len(found) != 1:
        raise ValueError(f'expected unique LIBERO task for instruction {task_text!r}; found {len(found)}')
    task = found[0]
    bddl = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
    env = ControlEnv(bddl_file_name=str(bddl), use_camera_obs=False,
                     has_renderer=False, has_offscreen_renderer=False,
                     control_freq=20, horizon=1000)
    env.reset()
    return env


def simulator_situation(env, flat_state):
    """Return E/R/G from exact simulator geometry without object identity."""
    inner = env.env
    sim = inner.sim
    sim.set_state_from_flattened(np.asarray(flat_state, np.float64))
    sim.forward()
    obs = inner._get_observations()
    eef_pos = np.asarray(obs['robot0_eef_pos'], np.float32)
    eef_body = 'gripper0_eef'
    eef_R = np.asarray(sim.data.get_body_xmat(eef_body), np.float64).reshape(3, 3)
    E = np.r_[eef_pos, _rotvec(eef_R)].astype(np.float32)

    # `inner.objects` is the task's object inventory. Names are used only to
    # retrieve geometry; they are removed before ranking or feature creation.
    objects = []
    for obj in inner.objects:
        body = str(obj.root_body)
        try:
            p = np.asarray(sim.data.get_body_xpos(body), np.float64).reshape(3)
            R = np.asarray(sim.data.get_body_xmat(body), np.float64).reshape(3, 3)
        except Exception:
            continue
        objects.append((float(np.linalg.norm(p - eef_pos)), p, R, body))
    objects.sort(key=lambda x: x[0])
    rel = []
    for _, p, R, _body in objects[:2]:
        rel.extend((eef_R.T @ (p - eef_pos)).astype(np.float32))
        rel.extend(_rotvec(eef_R.T @ R))
    while len(rel) < 12:
        rel.extend([0.0] * 6)
    Rfeat = np.asarray(rel[:12], np.float32)

    gripq = np.asarray(obs['robot0_gripper_qpos'], np.float32).reshape(-1)
    opening = float(np.mean(np.abs(gripq))) if gripq.size else 0.0
    model = sim.model
    finger_bodies = set()
    for bid, name in enumerate(model.body_names):
        if name and ('finger' in str(name).lower()):
            finger_bodies.add(int(bid))
    object_roots = set()
    for obj in inner.objects:
        try:
            object_roots.add(int(model.body_name2id(str(obj.root_body))))
        except Exception:
            pass
    # Descendants count as object geometry for contacts.
    object_bodies = set(object_roots)
    parents = np.asarray(model.body_parentid, np.int64)
    for bid in range(len(parents)):
        cur = bid
        while cur > 0:
            if cur in object_roots:
                object_bodies.add(bid)
                break
            cur = int(parents[cur])
    geom_body = np.asarray(model.geom_bodyid, np.int64)
    contact = False
    for ci in range(int(sim.data.ncon)):
        c = sim.data.contact[ci]
        b1, b2 = int(geom_body[int(c.geom1)]), int(geom_body[int(c.geom2)])
        if ((b1 in finger_bodies and b2 in object_bodies) or
                (b2 in finger_bodies and b1 in object_bodies)):
            contact = True
            break
    G = np.asarray([opening, float(contact)], np.float32)
    return E, Rfeat, G


def previous_action_features(rows, rowids):
    lookup = {(int(t), int(e), int(f)): int(i)
              for i, (t, e, f) in enumerate(zip(rows['task'], rows['episode'], rows['frame']))}
    actions = np.asarray(rows['actions'][:, :, :7], np.float32)
    out = np.zeros((len(rowids), 71), np.float32)
    for j, rid in enumerate(rowids):
        prev = lookup.get((int(rows['task'][rid]), int(rows['episode'][rid]), int(rows['frame'][rid]) - 10))
        if prev is not None:
            out[j, :70] = actions[prev].reshape(-1)
            out[j, 70] = 1.0
    return out


def bootstrap_mean(values, episodes, seed=SEED, reps=BOOTSTRAP_REPS):
    x = np.asarray(values, np.float64)
    ep = np.asarray(episodes, np.int64)
    unique = np.unique(ep)
    groups = {int(e): np.flatnonzero(ep == e) for e in unique}
    rng = np.random.default_rng(seed)
    samples = np.empty(reps, np.float64)
    for b in range(reps):
        chosen = rng.choice(unique, len(unique), replace=True)
        idx = np.concatenate([groups[int(e)] for e in chosen])
        samples[b] = x[idx].mean()
    return {"mean": float(x.mean()), "ci95": [float(np.quantile(samples, .025)),
            float(np.quantile(samples, .975))], "episodes": int(len(unique)),
            "rows": int(len(x)), "bootstrap_reps": int(reps)}


def _retrieval(score_qm, qids, pool_ids, rows, k):
    cand = np.empty((len(qids), min(k, len(pool_ids))), np.int64)
    pool_tasks = np.asarray(rows['task'][pool_ids], np.int64)
    q_tasks = np.asarray(rows['task'][qids], np.int64)
    for i in range(len(qids)):
        score = np.asarray(score_qm[i], np.float32).copy()
        score[pool_tasks == q_tasks[i]] = -np.inf
        cand[i] = topk_row_ids(score, pool_ids, k)
    return cand


def _weights_grid():
    grid = []
    for name in ('E', 'R', 'G', 'H'):
        grid.append((name, (name,), (1.0,)))
    for w in ((.75, .25), (.5, .5), (.25, .75)):
        grid.append((f'R+G:{w[0]:g}/{w[1]:g}', ('R', 'G'), w))
    for w in ((.5, .25, .25), (.25, .5, .25), (.25, .25, .5), (1/3, 1/3, 1/3)):
        grid.append((f'R+G+H:{w[0]:.3f}/{w[1]:.3f}/{w[2]:.3f}', ('R', 'G', 'H'), w))
    for w in ((.4, .2, .2, .2), (.2, .4, .2, .2), (.2, .2, .4, .2),
              (.2, .2, .2, .4), (.25, .25, .25, .25)):
        grid.append((f'E+R+G+H:{w[0]:.2f}/{w[1]:.2f}/{w[2]:.2f}/{w[3]:.2f}',
                     ('E', 'R', 'G', 'H'), w))
    return grid


def _action_metrics(qids, cand_ids, rows, pool_ids, episodes):
    truth = np.asarray(rows['actions'][qids, :, :7], np.float32)
    ca = np.asarray(rows['actions'][cand_ids, :, :7], np.float32)
    pool_actions = np.asarray(rows['actions'][pool_ids, :, :7], np.float32)
    top1 = np.mean((ca[:, 0] - truth) ** 2, axis=(1, 2))
    top5best = np.min(np.mean((ca[:, :5] - truth[:, None]) ** 2, axis=(2, 3)), axis=1)
    # Full-pool action oracle and Recall@8 use the active7 normalized action MSE.
    flat_truth = truth.reshape(len(qids), -1)
    flat_pool = pool_actions.reshape(len(pool_ids), -1)
    all_mse = np.maximum((np.sum(flat_truth**2, axis=1)[:, None] +
                          np.sum(flat_pool**2, axis=1)[None, :] - 2 * flat_truth @ flat_pool.T) /
                         flat_truth.shape[1], 0.0)
    oracle = all_mse.min(axis=1)
    rel = all_mse <= .01
    pool_map = {int(r): i for i, r in enumerate(pool_ids)}
    recall = []
    for i in range(len(qids)):
        denom = int(rel[i].sum())
        if denom == 0:
            recall.append(float('nan'))
        else:
            pos = [pool_map[int(r)] for r in cand_ids[i, :min(8, cand_ids.shape[1])]]
            recall.append(float(rel[i, pos].sum() / denom))
    qxyz = truth[:, :, :3].sum(axis=1)
    pxyz = ca[:, 0, :, :3].sum(axis=1)
    cos = np.sum(qxyz * pxyz, axis=1) / np.maximum(np.linalg.norm(qxyz, axis=1) * np.linalg.norm(pxyz, axis=1), 1e-12)
    grip = np.mean(np.sign(truth[:, :, 6]) == np.sign(ca[:, 0, :, 6]), axis=1)
    return {"top1_mse": top1, "top5_best_mse": top5best, "oracle_mse": oracle,
            "recall8": np.asarray(recall, np.float32), "xyz_cosine": cos,
            "gripper_sign_agreement": grip}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--smoke', action='store_true', help='map/extract a few rows only')
    args = ap.parse_args()
    out = args.output
    if out.exists():
        raise FileExistsError(f'refusing to overwrite {out}')
    out.mkdir(parents=True)
    (out / 'logs').mkdir()
    started = time.monotonic()
    rows = load_rows()
    pred = np.load(LOCAL / 'predictions.npz', allow_pickle=False)
    qids = np.asarray(pred['query_rows'], np.int64)
    hard_mask = np.asarray(pred['hard_mask'], bool)
    if len(qids) != 548 or int(hard_mask.sum()) != 63:
        raise AssertionError((len(qids), int(hard_mask.sum())))
    hard_ids = qids[hard_mask]
    pool_z = np.load(MEMORY / 'units.npz', allow_pickle=False)
    pool_ids = np.asarray(pool_z['pool_row_ids'], np.int64)
    if len(pool_ids) != 3857 or np.any(np.asarray(rows['task'][pool_ids]) < 10):
        raise AssertionError('raw-rows-noLong pool does not match the fixed nonLong pool')
    if len(np.unique(pool_ids)) != len(pool_ids):
        raise AssertionError('duplicate pool row ID')
    dev_eps, _ = calibration_dev_episodes(rows)
    dev_ids = np.flatnonzero(np.isin(rows['episode'], dev_eps) & np.asarray(rows['validation']) & (np.asarray(rows['task']) >= 10))
    if not len(dev_ids) or np.any(np.asarray(rows['task'][dev_ids]) < 10):
        raise AssertionError('invalid non-Long Dev selection rows')
    assert_nonlong_roles(rows['task'], {'memory_pool': pool_ids, 'dev_selection': dev_ids})
    long_memory = np.flatnonzero(np.asarray(rows['memory']) & (np.asarray(rows['task']) < 10))
    ref_ids = np.unique(np.r_[qids, long_memory])
    history_ids = np.unique(np.r_[pool_ids, dev_ids, qids, long_memory])
    if args.smoke:
        history_ids = np.unique(np.r_[pool_ids[:16], dev_ids[:4], qids[:8], long_memory[:8]])
    task_texts = {int(k): str(v) for k, v in json.load(open(RUN / 'data_manifest.json'))['tasks'].items()}
    h5paths = load_task_hdf5_files(task_texts)
    stats = json.load(open(NORM))['norm_stats']['actions']
    q01, q99 = np.asarray(stats['q01'], np.float64)[:7], np.asarray(stats['q99'], np.float64)[:7]
    manifest = json.load(open(RUN / 'data_manifest.json'))
    by_ep = find_episode_record(manifest)
    feat = {k: np.full((len(rows['task']), d), np.nan, np.float32)
            for k, d in [('E', 6), ('R', 12), ('G', 2)]}
    source_ids = np.full(len(rows['task']), -1, np.int64)
    h5_frame_ids = np.full(len(rows['task']), -1, np.int64)
    demo_names = {}
    demo_cache = {}
    from collections import defaultdict
    ids_by_ep = defaultdict(list)
    for rid in history_ids:
        ids_by_ep[int(rows['episode'][rid])].append(int(rid))
    # Action-history is based only on the already executed preceding chunk.
    feat['H'] = np.full((len(rows['task']), 71), np.nan, np.float32)
    hist = previous_action_features(rows, history_ids)
    feat['H'][history_ids] = hist
    ep_records = []
    envs = {}
    try:
        for n, (ep_idx, ep_rowids) in enumerate(sorted(ids_by_ep.items()), 1):
            rec = by_ep.get(ep_idx)
            if rec is None:
                raise KeyError(f'no manifest episode record for row episode {ep_idx}')
            task_id = int(rec['task'])
            if any(int(rows['task'][r]) != task_id for r in ep_rowids):
                raise AssertionError(f'episode {ep_idx} has inconsistent task')
            src_id, demo_name, action_map, state_rows = _action_match_for_episode(
                ep_idx, rec, task_id, rows, ep_rowids, h5paths, (q01, q99), demo_cache)
            demo_names[ep_idx] = demo_name
            text = task_texts[task_id]
            if task_id not in envs:
                envs[task_id] = _build_env(text)
            env = envs[task_id]
            with h5py.File(h5paths[task_id], 'r') as h:
                demo = h['data'][demo_name]
                all_states = demo['states']
                for rid in ep_rowids:
                    frame = int(rows['frame'][rid])
                    h5frame = int(action_map[frame])
                    state = np.asarray(all_states[h5frame], np.float64)
                    E, R, G = simulator_situation(env, state)
                    feat['E'][rid], feat['R'][rid], feat['G'][rid] = E, R, G
                    source_ids[rid], h5_frame_ids[rid] = src_id, h5frame
            ep_records.append({'episode': ep_idx, 'task': task_id, 'source_episode_index': src_id,
                               'hdf5_demo': demo_name, 'rows': len(ep_rowids)})
            if n % 25 == 0 or n == len(ids_by_ep):
                line = {'episodes_done': n, 'episodes_total': len(ids_by_ep), 'rows': int(sum(len(x) for x in ids_by_ep.values())),
                        'elapsed_s': round(time.monotonic() - started, 2)}
                with open(out / 'logs' / 'progress.jsonl', 'a') as f:
                    f.write(json.dumps(line) + '\n')
                print(json.dumps(line), flush=True)
    finally:
        for env in envs.values():
            env.close()
    if args.smoke:
        jwrite(out / 'smoke.json', {'mapped_rows': int(np.isfinite(feat['E']).all(axis=1).sum()),
            'expected_rows': int(len(history_ids)), 'actions_aligned': True,
            'row_id_sample': history_ids[:10].astype(int).tolist(), 'elapsed_s': time.monotonic() - started})
        return
    for name in feat:
        if not np.isfinite(feat[name][history_ids]).all():
            raise AssertionError(f'feature extraction incomplete: {name}')
    np.savez_compressed(out / 'simulator_features.npz', row_ids=history_ids,
        E=feat['E'][history_ids], R=feat['R'][history_ids], G=feat['G'][history_ids], H=feat['H'][history_ids],
        source_episode_index=source_ids[history_ids], hdf5_frame=h5_frame_ids[history_ids])
    jwrite(out / 'episode_mapping.json', ep_records)

    all_actions = np.asarray(rows['actions'][:, :, :7], np.float32)
    # Standardize using only the fixed nonLong memory rows.
    norm_tr, norm_q, block_stats = {}, {}, {}
    for b in ('E', 'R', 'G', 'H'):
        norm_tr[b], norm_all, block_stats[b] = normalize_block(feat[b][pool_ids], feat[b][history_ids])
        norm_q[b] = np.full_like(feat[b], np.nan)
        norm_q[b][history_ids] = norm_all
    grid = _weights_grid()
    dev_lookup = {int(r): i for i, r in enumerate(history_ids)}
    dev_pos = np.asarray([dev_lookup[int(r)] for r in dev_ids], np.int64)
    pool_pos = np.asarray([dev_lookup[int(r)] for r in pool_ids], np.int64)
    ydev = np.asarray(rows['actions'][dev_ids, :, :7], np.float32)
    dev_tasks, pool_tasks = np.asarray(rows['task'][dev_ids]), np.asarray(rows['task'][pool_ids])
    grid_scores = []
    # Build memory/query normalized features once per block, then select weights
    # on nonLong Dev only. Long features and labels are not touched in selection.
    qnorm = {}
    for b in ('E', 'R', 'G', 'H'):
        _tr, qall, _ = normalize_block(feat[b][pool_ids], feat[b][history_ids])
        qnorm[b] = qall
    best = None
    for name, names, weights in grid:
        pool_key = combine_blocks({b: norm_tr[b] for b in names}, names, weights)
        dev_key = combine_blocks({b: qnorm[b][dev_pos] for b in names}, names, weights)
        scores = dev_key @ pool_key.T
        cand = _retrieval(scores, dev_ids, pool_ids, rows, 1)
        mse = np.mean((all_actions[cand[:, 0]] - ydev) ** 2, axis=(1, 2))
        value = float(mse.mean())
        grid_scores.append({'name': name, 'blocks': list(names), 'weights': list(weights), 'dev_top1_mse': value})
        if best is None or value < best['dev_top1_mse']:
            best = grid_scores[-1]
    jwrite(out / 'weight_selection.json', {'fit_rows': 0, 'selection_rows': len(dev_ids),
        'selection_tasks_min': int(np.min(rows['task'][dev_ids])), 'candidate_rows': len(pool_ids),
        'selected': best, 'grid': grid_scores, 'stats_source': '3857 raw-rows-noLong rows only'})

    # Query/evaluation functions preserve row IDs throughout (compact features
    # and candidate pool positions are converted only at retrieval boundaries).
    qpos = np.asarray([dev_lookup[int(r)] for r in qids], np.int64)
    norm_pool = {b: norm_tr[b] for b in ('E', 'R', 'G', 'H')}
    norm_query = {b: qnorm[b][qpos] for b in ('E', 'R', 'G', 'H')}
    localz = np.load(LOCAL / 'ranks_wrist_lowerhalf_prev.npz', allow_pickle=False)
    local_pred = np.load(LOCAL / 'predictions.npz', allow_pickle=False)
    local_qids = np.asarray(local_pred['query_rows'], np.int64)
    local_cands = np.asarray(localz['cand_rows'], np.int64)
    local_lookup = {int(r): i for i, r in enumerate(local_qids)}
    local_top = np.stack([local_cands[local_lookup[int(r)]] for r in qids])
    qepisodes = np.asarray(rows['episode'][qids], np.int64)
    metrics = {}
    result_rows = {}
    key_specs = [(b, (b,), (1.0,)) for b in ('E', 'R', 'G', 'H')]
    key_specs += [('R+G', ('R', 'G'), (.5, .5)),
                  ('R+G+H', ('R', 'G', 'H'), (1/3, 1/3, 1/3)),
                  ('E+R+G+H', ('E', 'R', 'G', 'H'), (.25, .25, .25, .25)),
                  ('DEV_SELECTED', tuple(best['blocks']), tuple(best['weights']))]
    for name, blocks, weights in key_specs:
        pk = combine_blocks(norm_pool, blocks, weights)
        qk = combine_blocks(norm_query, blocks, weights)
        scores = qk @ pk.T
        cand = _retrieval(scores, qids, pool_ids, rows, 8)
        result_rows[name] = cand
        met = _action_metrics(qids, cand, rows, pool_ids, qepisodes)
        metrics[name] = met
    # Main nonLong baselines: cached local key, hold-last, random, action oracle.
    baseline = {}
    baseline['wrist_lowerhalf+prev'] = _action_metrics(qids, local_top[:, :5], rows, pool_ids, qepisodes)
    lookup = {(int(t), int(e), int(f)): i for i, (t, e, f) in enumerate(zip(rows['task'], rows['episode'], rows['frame']))}
    hold = np.zeros_like(all_actions[qids])
    for j, rid in enumerate(qids):
        prev = lookup.get((int(rows['task'][rid]), int(rows['episode'][rid]), int(rows['frame'][rid]) - 10))
        if prev is not None:
            hold[j] = all_actions[prev, -1][None]
    hold_mse = np.mean((hold - all_actions[qids]) ** 2, axis=(1, 2))
    rng = np.random.default_rng(SEED + 718)
    random_ids = np.empty(len(qids), np.int64)
    for i, rid in enumerate(qids):
        allowed = pool_ids[np.asarray(rows['task'][pool_ids]) != int(rows['task'][rid])]
        random_ids[i] = int(rng.choice(allowed))
    random_cand = random_ids[:, None]
    random_metrics = _action_metrics(qids, np.tile(random_cand, (1, 5)), rows, pool_ids, qepisodes)
    oracle_mse = np.asarray(pred['oracle_mse'], np.float32)
    baseline['hold_last'] = {'top1_mse': hold_mse, 'top5_best_mse': hold_mse, 'oracle_mse': oracle_mse,
        'recall8': np.full(len(qids), np.nan), 'xyz_cosine': np.full(len(qids), np.nan),
        'gripper_sign_agreement': np.mean(np.sign(all_actions[qids, :, 6]) == np.sign(hold[:, :, 6]), axis=1)}
    baseline['random_one'] = random_metrics
    baseline['action_oracle'] = {'top1_mse': oracle_mse, 'top5_best_mse': oracle_mse, 'oracle_mse': oracle_mse,
        'recall8': np.full(len(qids), np.nan), 'xyz_cosine': np.full(len(qids), np.nan),
        'gripper_sign_agreement': np.full(len(qids), np.nan)}
    # Same Long task, different episode pool is a reference only; no tuning.
    reference = {}
    selected_blocks, selected_weights = tuple(best['blocks']), tuple(best['weights'])
    selected_key_pool = combine_blocks(norm_pool, selected_blocks, selected_weights)
    selected_key_query = combine_blocks(norm_query, selected_blocks, selected_weights)
    ref_results = {}
    for name, blocks, weights in key_specs:
        qkey = combine_blocks(norm_query, blocks, weights)
        pool_key = combine_blocks({b: qnorm[b][np.asarray([dev_lookup[int(r)] for r in long_memory], np.int64)]
                                   for b in blocks}, blocks, weights)
        refs = np.full((len(qids), 5), -1, np.int64)
        for i, rid in enumerate(qids):
            allow_mask = ((np.asarray(rows['task'][long_memory]) == int(rows['task'][rid])) &
                          (np.asarray(rows['episode'][long_memory]) != int(rows['episode'][rid])))
            allow_pos = np.flatnonzero(allow_mask)
            if len(allow_pos) == 0:
                continue
            scores = qkey[i] @ pool_key[allow_pos].T
            order = np.lexsort((long_memory[allow_pos], -scores))[:5]
            refs[i, :len(order)] = long_memory[allow_pos[order]]
        ref_results[name] = refs
        valid = refs[:, 0] >= 0
        truth_valid = all_actions[qids[valid]]
        cands = all_actions[np.maximum(refs[valid], 0)]
        top1_ref = np.mean((cands[:, 0] - truth_valid) ** 2, axis=(1, 2))
        best5_ref = np.min(np.mean((cands - truth_valid[:, None]) ** 2, axis=(2, 3)), axis=1)
        reference[name] = {'candidate_count': int(len(long_memory)), 'rows_with_candidate': int(valid.sum()),
            'top1_mse': top1_ref, 'top5_best_mse': best5_ref,
            'query_rows': qids[valid], 'episode': np.asarray(rows['episode'][qids[valid]], np.int64)}
    # k-nearest situation action dispersion (within-neighbor pairwise MSE).
    selected_pk = selected_key_pool
    selected_qk = selected_key_query
    score_all = selected_qk @ selected_pk.T
    nn_order = np.argsort(-score_all, axis=1, kind='stable')
    dispersion = {}
    for k in (8, 32):
        per_query = np.empty(len(qids), np.float32)
        for i in range(len(qids)):
            pos = nn_order[i, :k]
            acts = all_actions[pool_ids[pos]].reshape(k, -1)
            gram = (np.sum(acts * acts, axis=1)[:, None] + np.sum(acts * acts, axis=1)[None] - 2 * acts @ acts.T) / acts.shape[1]
            per_query[i] = float(np.mean(gram[np.triu_indices(k, 1)]))
        dispersion[str(k)] = per_query

    # Store rowwise arrays to make all intervals and paired comparisons auditable.
    raw = {'query_rows': qids, 'episode': qepisodes, 'hard': hard_mask,
           'local_key_top5_rows': local_top, 'hold_last_mse': hold_mse, 'random_rows': random_ids,
           **{f'{n}_top5_rows': c for n, c in result_rows.items()},
           **{f'reference_{n}_top5_rows': c for n, c in ref_results.items()},
           **{f'{n}_{k}': np.asarray(v) for n, d in metrics.items() for k, v in d.items()},
           **{f'baseline_{n}_{k}': np.asarray(v) for n, d in baseline.items() for k, v in d.items()},
           **{f'dispersion_{k}': v for k, v in dispersion.items()}}
    np.savez_compressed(out / 'row_metrics.npz', **raw)

    # Summaries and bootstrap CIs for all/hard subsets, all on query episodes.
    summary = {'protocol': {'queries': len(qids), 'hard_queries': int(hard_mask.sum()),
        'pool_rows': len(pool_ids), 'pool_long_tasks': False, 'dev_rows': len(dev_ids),
        'dev_long_rows': 0, 'same_task_reference_rows': len(long_memory),
        'hdf5_action_windows_verified': int(len(history_ids)), 'episode_mappings': len(ep_records),
        'hard_threshold': .01549410422449, 'bootstrap_reps': BOOTSTRAP_REPS,
        'fit_or_weight_selection_used_long': False, 'gpu_used': False},
        'selected_key': best, 'metrics': {}, 'comparisons': {}, 'reference': {},
        'conditional_action_dispersion': {}, 'action_oracle_input': 'fixed run-01 oracle_mse; recomputed from raw rows for other metrics',
        'elapsed_seconds': round(time.monotonic() - started, 2)}
    method_data = {**metrics, **baseline}
    for name, d in reference.items():
        v = np.asarray(d['top1_mse'], np.float32)
        b5 = np.asarray(d['top5_best_mse'], np.float32)
        ep = np.asarray(d['episode'], np.int64)
        summary['reference'][name] = {'candidate_count': d['candidate_count'], 'rows_with_candidate': d['rows_with_candidate'],
            'top1_mse': bootstrap_mean(v, ep), 'top5_best_mse': bootstrap_mean(b5, ep)}
    for subset, mask in [('all_long', np.ones(len(qids), bool)), ('hard', hard_mask)]:
        summary['metrics'][subset] = {}
        for name, d in method_data.items():
            summary['metrics'][subset][name] = {}
            for key, val in d.items():
                arr = np.asarray(val)
                if arr.ndim != 1 or len(arr) != len(qids) or not np.isfinite(arr[mask]).all():
                    continue
                summary['metrics'][subset][name][key] = bootstrap_mean(arr[mask], qepisodes[mask])
        for k, val in dispersion.items():
            summary['conditional_action_dispersion'].setdefault(subset, {})[k] = bootstrap_mean(val[mask], qepisodes[mask])
        for name, d in method_data.items():
            if 'top1_mse' not in d or name == 'action_oracle':
                continue
            ref = baseline['wrist_lowerhalf+prev']['top1_mse']
            delta = np.asarray(d['top1_mse']) - np.asarray(ref)
            summary['comparisons'].setdefault(subset, {})[f'{name}_minus_wrist_lowerhalf_prev'] = bootstrap_mean(delta[mask], qepisodes[mask])
            delta_hold = np.asarray(d['top1_mse']) - hold_mse
            summary['comparisons'][subset][f'{name}_minus_hold_last'] = bootstrap_mean(delta_hold[mask], qepisodes[mask])
    jwrite(out / 'results.json', summary)
    print(json.dumps({'complete': True, 'output': str(out), 'elapsed_s': summary['elapsed_seconds']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

