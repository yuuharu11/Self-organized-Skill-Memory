#!/usr/bin/env python3
"""Dev v2: a wider selection-only Dev set for query/key experiments.

Why: the old Dev (217 rows) is two LIBERO-90 tasks (6, 7: open a drawer) with
no gripper switch, and its ranking disagreed with unseen Eval on MK1c/EKQ.

Rows are built exactly like the locked unseen Eval rows (KEY2-F):
  * source: LIBERO-90 HDF5 demos, every other frame (10 Hz), images rotated
    180 degrees (frame[::-1, ::-1]) before the pi0.5 transforms -- the same
    observe() as seg_context_extract.extract_unseen_hdf5_stride5;
  * query starts every 10 frames over the WHOLE demo (frame % 10 == 0);
  * actions = next 10 normalized actions [:7]; state = normalized state[:8];
  * history = last 10 executed normalized actions of the causal trace
    (== key_shared_latent._last10 of build_unseen_queries' history stream);
  * VLM = final pi0.5 prefix tokens packed to 532 valid slots (+ groups).

Task choice: LIBERO-90 tasks that are not in the closed-loop / offline unseen
Eval set (0,10,13,31,35,43,57,61,84,88), whose instruction is not a LIBERO-40
memory instruction (20, 44, 77 excluded), whose HDF5 exists (51 absent), and
not the old Dev tasks (6, 7).  Twelve tasks, stratified like Eval (2
articulated, 4 kitchen place, 3 living-room pick/place, 3 study), chosen by a
fixed seed, 8 demos each.  Eval labels are never read for the choice.

Use ``load_dev_v2()`` for a split dict compatible with key_final.load_split
(key, history, history_mask, state, actions, task, episode, hard, tokens,
groups) plus subset masks (first_unit, post_switch, dir_change).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[5]
STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', REPO_ROOT / '.skillnet-state')).expanduser()
sys.path.insert(0, str(HERE))

ROOT = (STATE_DIR / 'outputs/memory-pilot/unseen-libero')
OUT = ROOT / 'dev-v2-01'
H5ROOT = (STATE_DIR / 'data/libero/datasets/libero_90')
CKPT = (STATE_DIR / 'checkpoints/openpi-assets/checkpoints/pi05_libero')
EVAL_TASKS = (0, 10, 13, 31, 35, 43, 57, 61, 84, 88)
OLD_DEV = (6, 7)
L40_SAME_INSTRUCTION = (20, 44, 77)
MISSING = (51,)
SEED = 4101
DEMOS = 8
EP_OFFSET = 3_000_000
# Strata mirror the Eval mix (2 articulated, 4 kitchen place, 2-3 living room, 2-3 study).
STRATA = {
    'articulated': [11, 22, 23, 28, 33, 39],
    'kitchen_place': [1, 2, 3, 4, 5, 8, 9, 12, 14, 15, 16, 17, 18, 19, 21, 24, 25, 26, 27, 29, 30, 32, 34, 36,
                      37, 38, 40, 41, 42, 45],
    'living_room': [46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72],
    'study': [73, 74, 75, 76, 78, 79, 80, 81, 82, 83, 85, 86, 87, 89],
}
PER_STRATUM = {'articulated': 2, 'kitchen_place': 4, 'living_room': 3, 'study': 3}
PARITY_KEYS = ((6, 0, 0), (6, 0, 10), (7, 3, 20))  # old Dev rows: pipeline parity probe only


def log(event: str, out: Path = OUT, **kw: Any) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'progress.log').open('a') as f:
        f.write(json.dumps({'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'stage': event, **kw},
                           ensure_ascii=False) + '\n')


def write_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def choose_tasks(seed: int = SEED) -> list[int]:
    banned = set(EVAL_TASKS) | set(OLD_DEV) | set(L40_SAME_INSTRUCTION) | set(MISSING)
    rng = np.random.default_rng(seed)
    out = []
    for name, ids in STRATA.items():
        ids = [i for i in ids if i not in banned]
        out += sorted(rng.choice(ids, size=PER_STRATUM[name], replace=False).astype(int).tolist())
    return sorted(out)


def task_table() -> dict[int, dict[str, str]]:
    from libero.libero import benchmark
    s = benchmark.get_benchmark_dict()['libero_90']()
    return {i: {'name': s.get_task(i).name, 'instruction': s.get_task(i).language} for i in range(int(s.n_tasks))}


# ----------------------------------------------------------------------------
# subset definitions (ground truth used only to define scoring subsets)
# ----------------------------------------------------------------------------

def subset_masks(frame: np.ndarray, history: np.ndarray, history_mask: np.ndarray,
                 actions: np.ndarray) -> dict[str, np.ndarray]:
    """first_unit: frame 0 (no history).  post_switch: the gripper command sign
    changed within the last 5 executed steps or changes within the first 3
    future steps.  dir_change: the mean arm xyz direction of the last 5
    executed steps and the first 5 future steps has cosine < 0 (with both
    non-negligible).  Defined from executed/true actions for scoring only."""
    frame = np.asarray(frame)
    h = np.asarray(history, np.float32); m = np.asarray(history_mask, bool); a = np.asarray(actions, np.float32)
    first = frame == 0
    gs_h = np.sign(h[:, :, 6]); gs_a = np.sign(a[:, :, 6])
    sw = np.zeros(len(h), bool)
    for i in range(len(h)):
        seq = np.r_[gs_h[i][m[i]][-6:], gs_a[i][:4]]
        seq = seq[seq != 0]
        sw[i] = len(seq) > 1 and np.any(seq[1:] != seq[:-1])
    vh = (h[:, -5:, :3] * m[:, -5:, None]).sum(1); va = a[:, :5, :3].sum(1)
    nh = np.linalg.norm(vh, axis=1); na = np.linalg.norm(va, axis=1)
    cos = (vh * va).sum(1) / np.maximum(nh * na, 1e-8)
    ok = (m[:, -5:].sum(1) == 5) & (nh > .25) & (na > .25)
    dc = ok & (cos < 0)
    return {'first_unit': first, 'post_switch': sw, 'dir_change': dc, 'switch_or_dir': sw | dc}


# ----------------------------------------------------------------------------
# build
# ----------------------------------------------------------------------------

def build(out: Path = OUT, *, batch_size: int = 4, demos: int = DEMOS, tasks: list[int] | None = None) -> dict[str, Any]:
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '.20')
    import h5py
    from openpi.transforms import Normalize, ResizeImages, TokenizePrompt, PadStatesAndActions
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.policies.libero_policy import LiberoInputs
    from openpi.models.model import ModelType
    from openpi.shared.normalize import load
    from openpi.shared.nnx_utils import module_jit
    import seg_context_extract as sce

    out = Path(out); raw = out / 'raw'; raw.mkdir(parents=True, exist_ok=True)
    tasks = tasks or choose_tasks()
    table = task_table()
    records = []
    for t in tasks:
        p = H5ROOT / f"{table[t]['name']}_demo.hdf5"
        with h5py.File(p, 'r') as f:
            names = sorted((x for x in f['data'] if x.startswith('demo_')), key=lambda x: int(x.split('_')[1]))
            for e, nm in enumerate(names[:demos]):
                n = len(f['data'][nm]['actions'][::2])
                records.append({'task_id': t, 'task_name': table[t]['name'], 'instruction': table[t]['instruction'],
                                'episode': e, 'source_demo': nm, 'source_hdf5': str(p), 'source_steps': n})
    # parity probe records (old Dev task 6/7 demos, same as unseen-rows-02)
    src = json.loads((ROOT / 'unseen-rows-02/manifest.json').read_text())['source_records']
    pmap = {(int(r['task_id']), int(r['episode'])): r for r in src}
    keys: list[tuple[int, int, int]] = []
    recmap = {(r['task_id'], r['episode']): r for r in records}
    for r in records:
        keys += [(r['task_id'], r['episode'], f) for f in range(0, max(0, r['source_steps'] - 9), 10)]
    parity = [k for k in PARITY_KEYS]
    for k in parity:
        recmap[(k[0], k[1] + 10_000)] = pmap[(k[0], k[1])]
    allkeys = keys + [(k[0], k[1] + 10_000, k[2]) for k in parity]
    write_json(out / 'plan.json', {'tasks': tasks, 'names': {t: table[t]['name'] for t in tasks}, 'demos': demos,
                                    'rows': len(keys), 'parity_keys': parity, 'seed': SEED, 'strata': STRATA,
                                    'per_stratum': PER_STRATUM})
    log('build_plan', out, tasks=tasks, rows=len(keys))
    norm_files = list((CKPT / 'assets').rglob('norm_stats.json'))
    transforms = [LiberoInputs(ModelType.PI05), Normalize(load(norm_files[0].parent), use_quantiles=True),
                  ResizeImages(224, 224), TokenizePrompt(PaligemmaTokenizer(200), discrete_state_input=False),
                  PadStatesAndActions(32)]
    run = sce._load_module(HERE.parent / 'pi05_memory' / 'run.py', 'devv2_pi05_run')
    model, _ = run.load_policy(CKPT); encode = module_jit(model.prefix_features)
    h5 = {}

    def observe(key):
        t, e, f = map(int, key); rec = recmap[(t, e)]
        p = Path(rec['source_hdf5'])
        if p not in h5:
            if len(h5) >= 1:
                _, old = h5.popitem(); old.close()
            h5[p] = h5py.File(p, 'r')
        d = h5[p]['data'][rec['source_demo']]; obs = d['obs']; si = 2 * f
        state = np.concatenate([np.asarray(obs['ee_pos'][si], np.float32), np.asarray(obs['ee_ori'][si], np.float32),
                                np.asarray(obs['gripper_states'][si], np.float32)])
        base = np.asarray(obs['agentview_rgb'][si], np.uint8)[::-1, ::-1].copy()
        wrist = np.asarray(obs['eye_in_hand_rgb'][si], np.uint8)[::-1, ::-1].copy()
        acts = np.asarray(d['actions'][::2][f:f + 10], np.float32)
        hist = np.asarray(d['actions'][::2][max(0, f - 10):f], np.float32)
        item = {'observation/image': base, 'observation/wrist_image': wrist, 'observation/state': state,
                'actions': acts, 'prompt': rec['instruction']}
        for tr in transforms:
            item = tr(item)
        hn = np.zeros((10, 7), np.float32)
        if len(hist):
            hx = {'observation/image': base, 'observation/wrist_image': wrist, 'observation/state': state,
                  'actions': hist, 'prompt': rec['instruction']}
            for tr in transforms:
                hx = tr(hx)
            hn[-len(hist):] = np.asarray(hx['actions'])[:len(hist), :7]
        return {'image_base': np.asarray(item['image']['base_0_rgb']),
                'image_wrist': np.asarray(item['image']['left_wrist_0_rgb']),
                'state': np.asarray(item['state']), 'actions': np.asarray(item['actions']),
                'prompt': np.asarray(item['tokenized_prompt']), 'prompt_mask': np.asarray(item['tokenized_prompt_mask']),
                'hist': hn}

    def encode_batch(items):
        n = len(items)
        obs = {k: np.stack([x[k] for x in items]) for k in ('image_base', 'image_wrist', 'state', 'prompt', 'prompt_mask')}
        h, g = encode(run.obs_batch(obs, np.arange(n)))
        return np.asarray(h), np.asarray(g)

    # extract_prefix_memmap stores state/actions; we add hist + a thumbnail via a wrapper.
    aux = out / 'aux'; aux.mkdir(exist_ok=True)
    hist_path = aux / 'hist.npy'
    hist_mm = np.lib.format.open_memmap(hist_path, mode='r+' if hist_path.exists() else 'w+', dtype=np.float32,
                                        shape=(len(allkeys), 10, 7))
    thumb_path = aux / 'thumb_base.npy'
    thumb_mm = np.lib.format.open_memmap(thumb_path, mode='r+' if thumb_path.exists() else 'w+', dtype=np.uint8,
                                         shape=(len(allkeys), 56, 56, 3))
    slot = {tuple(k): i for i, k in enumerate(allkeys)}

    def observe_rec(key):
        x = observe(key); i = slot[tuple(key)]
        hist_mm[i] = x['hist']
        img = np.asarray(x['image_base'])
        if img.dtype != np.uint8:
            img = ((img + 1) * 127.5).clip(0, 255).astype(np.uint8)
        thumb_mm[i] = img[::4, ::4]
        return x

    t0 = time.time()
    try:
        meta = sce.extract_prefix_memmap(allkeys, observe_rec, encode_batch, raw, batch_size=batch_size)
    finally:
        for f in h5.values():
            f.close()
        hist_mm.flush(); thumb_mm.flush()
    log('extract_done', out, rows=len(allkeys), seconds=time.time() - t0)
    finalize(out)
    return meta


def finalize(out: Path = OUT) -> dict[str, Any]:
    """Pack raw extraction into the split arrays + subset masks + parity check."""
    raw = out / 'raw'
    man = json.loads((raw / 'manifest.json').read_text())
    keys = np.asarray(man['keys'], np.int64)
    tok = np.load(raw / 'tokens.npy', mmap_mode='r'); grp = np.load(raw / 'groups.npy', mmap_mode='r')
    st = np.load(raw / 'state.npy', mmap_mode='r'); ac = np.load(raw / 'actions.npy', mmap_mode='r')
    hist = np.load(out / 'aux' / 'hist.npy', mmap_mode='r')
    par = keys[:, 1] >= 10_000
    # ---- parity vs locked old-Dev cache (same pipeline must reproduce it) ----
    import key_final as kf
    old = kf.load_split('dev', kf.OUT, tokens=True)
    omap = {tuple(map(int, k)): i for i, k in enumerate(old['key'])}
    checks = []
    for i in np.flatnonzero(par):
        k = (int(keys[i, 0]), int(keys[i, 1]) - 10_000, int(keys[i, 2]))
        j = omap[k]
        v = np.asarray(grp[i]) >= 0; vo = np.asarray(old['groups'][j]) >= 0
        a, b = np.asarray(tok[i], np.float32)[v], np.asarray(old['tokens'][j], np.float32)[vo]
        same_shape = a.shape == b.shape
        checks.append({'key': k, 'valid_tokens': int(v.sum()), 'old_valid_tokens': int(vo.sum()),
                       'token_max_abs_diff': float(np.abs(a - b).max()) if same_shape else None,
                       'token_rel_err': float(np.linalg.norm(a - b) / np.linalg.norm(b)) if same_shape else None,
                       'state_max_abs_diff': float(np.abs(np.asarray(st[i])[:8] - old['state'][j]).max()),
                       'actions_max_abs_diff': float(np.abs(np.asarray(ac[i])[:10, :7] - old['actions'][j]).max()),
                       'history_max_abs_diff': float(np.abs(np.asarray(hist[i]) - old['history'][j]).max()),
                       'history_mask_equal': bool(np.array_equal((np.arange(10) >= 10 - min(10, k[2])), old['history_mask'][j]))})
    # ---- pack Dev rows ----
    ix = np.flatnonzero(~par)
    d = out / 'data'; d.mkdir(exist_ok=True)
    k = keys[ix]
    hm = np.zeros((len(ix), 10), bool)
    for r, f in enumerate(k[:, 2]):
        n = min(10, int(f))
        if n:
            hm[r, -n:] = True
    arrays = {'key': k.astype(np.int32), 'task': k[:, 0].astype(np.int64), 'episode': (k[:, 1] + EP_OFFSET).astype(np.int64),
              'frame': k[:, 2].astype(np.int32), 'history': np.asarray(hist)[ix].astype(np.float32), 'history_mask': hm,
              'state': np.asarray(st)[ix, :8].astype(np.float32), 'actions': np.asarray(ac)[ix, :10, :7].astype(np.float32),
              'hard': np.zeros(len(ix), bool)}
    arrays['history'][~hm] = 0
    sub = subset_masks(arrays['frame'], arrays['history'], arrays['history_mask'], arrays['actions'])
    for n_, m_ in sub.items():
        arrays['sub_' + n_] = m_
    for n_, v in arrays.items():
        np.save(d / f'{n_}.npy', v, allow_pickle=False)
    tk = np.lib.format.open_memmap(d / 'tokens.npy', mode='w+', dtype=np.float16, shape=(len(ix), 532, 2048))
    for s in range(0, len(ix), 256):
        tk[s:s + 256] = tok[ix[s:s + 256]]
    tk.flush()
    np.save(d / 'groups.npy', np.asarray(grp)[ix], allow_pickle=False)
    np.save(d / 'thumb_base.npy', np.load(out / 'aux' / 'thumb_base.npy')[ix], allow_pickle=False)
    summary = {'rows': int(len(ix)), 'tasks': sorted(set(k[:, 0].tolist())), 'episodes': int(len({(a, b) for a, b, _ in k})),
               'frame_max': int(k[:, 2].max()), 'subsets': {n_: int(m_.sum()) for n_, m_ in sub.items()},
               'rows_per_task': {int(t): int((k[:, 0] == t).sum()) for t in np.unique(k[:, 0])},
               'parity_vs_old_dev': checks}
    write_json(out / 'summary.json', summary)
    log('finalize', out, rows=int(len(ix)))
    return summary


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------

def load_dev_v2(out: Path = OUT, *, tokens: bool = False) -> dict[str, np.ndarray]:
    """Split dict compatible with key_final.load_split (+ frame and sub_* masks)."""
    d = Path(out) / 'data'
    names = ('key', 'task', 'episode', 'frame', 'history', 'history_mask', 'state', 'actions', 'hard',
             'sub_first_unit', 'sub_post_switch', 'sub_dir_change', 'sub_switch_or_dir')
    s = {n: np.load(d / f'{n}.npy', allow_pickle=False) for n in names}
    if tokens:
        s['tokens'] = np.load(d / 'tokens.npy', mmap_mode='r')
        s['groups'] = np.load(d / 'groups.npy', mmap_mode='r')
    return s


def eval_subsets(split: dict[str, np.ndarray], frame: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Same subset masks for the locked learned/unseen splits (frame from key[:,2])."""
    f = np.asarray(split['key'])[:, 2] if frame is None else frame
    # learned rows: frame is absolute in the episode; history_mask tells whether it is the start.
    first = ~np.asarray(split['history_mask'], bool).any(1)
    m = subset_masks(np.where(first, 0, np.maximum(f, 1)), split['history'], split['history_mask'], split['actions'])
    return m


# ----------------------------------------------------------------------------
# 20 Hz rebuild (memory time scale).  The LIBERO-40 memory units are LeRobot
# frames = consecutive HDF5 steps after OpenVLA no-op removal (20 Hz); the 10 Hz
# build above (actions[::2]) doubles the time span of each 10-step target.
# Same as vq_geometry.build_rate_matched (VQ-C): start step = first kept step
# >= 2*frame (same start observation, so the VLM tokens are reused), history =
# previous 10 kept steps, target = next 10 kept steps, state = ee_states +
# gripper at the start step; q01/q99 normalization from the pi0.5 norm stats.
# ----------------------------------------------------------------------------

def build_20hz(out: Path = OUT) -> dict[str, Any]:
    import h5py
    import vq_geometry as vg
    d = Path(out) / 'data'; d20 = Path(out) / 'data20'; d20.mkdir(exist_ok=True)
    plan = json.loads((Path(out) / 'plan.json').read_text())
    table = task_table()
    ns = json.loads(vg.NORM.read_text())['norm_stats']
    q = lambda x, n: 2 * (x - np.asarray(ns[n]['q01'][:x.shape[-1]])) / (
        np.asarray(ns[n]['q99'][:x.shape[-1]]) - np.asarray(ns[n]['q01'][:x.shape[-1]])) - 1
    keys = np.load(d / 'key.npy')
    H, M, S, A, keep, shift = [], [], [], [], [], []
    cache = {}
    for i, (t, e, f) in enumerate(keys):
        if (t, e) not in cache:
            p = H5ROOT / f"{table[int(t)]['name']}_demo.hdf5"
            with h5py.File(p, 'r') as fo:
                names = sorted((x for x in fo['data'] if x.startswith('demo_')), key=lambda x: int(x.split('_')[1]))
                g = fo['data'][names[int(e)]]
                act = g['actions'][()]; ees = g['obs/ee_states'][()]; grp = g['obs/gripper_states'][()]
            cache = {(t, e): (act, ees, grp, np.flatnonzero(vg.noop_keep(act)))}
        act, ees, grp, K = cache[(t, e)]
        p_ = int(np.searchsorted(K, 2 * int(f)))
        if p_ + 10 > len(K):
            continue
        hist = np.zeros((10, 7), np.float32); m = np.zeros(10, bool)
        hk = K[max(0, p_ - 10):p_]
        if len(hk):
            hist[10 - len(hk):] = q(act[hk], 'actions'); m[10 - len(hk):] = True
        H.append(hist); M.append(m)
        S.append(q(np.concatenate([ees[K[p_]], grp[K[p_]]]), 'state').astype(np.float32))
        A.append(q(act[K[p_:p_ + 10]], 'actions').astype(np.float32))
        keep.append(i); shift.append(int(K[p_] - 2 * int(f)))
    keep = np.asarray(keep)
    arrays = {'row': keep, 'key': keys[keep], 'task': np.load(d / 'task.npy')[keep], 'episode': np.load(d / 'episode.npy')[keep],
              'frame': np.load(d / 'frame.npy')[keep], 'history': np.stack(H), 'history_mask': np.stack(M),
              'state': np.stack(S), 'actions': np.stack(A), 'hard': np.zeros(len(keep), bool),
              'start_shift_steps': np.asarray(shift)}
    first = ~arrays['history_mask'].any(1)
    sub = subset_masks(np.where(first, 0, np.maximum(arrays['frame'], 1)), arrays['history'], arrays['history_mask'], arrays['actions'])
    for n_, m_ in sub.items():
        arrays['sub_' + n_] = m_
    for n_, v in arrays.items():
        np.save(d20 / f'{n_}.npy', v, allow_pickle=False)
    summ = {'rows': int(len(keep)), 'dropped': int(len(keys) - len(keep)), 'subsets': {k: int(v.sum()) for k, v in sub.items()},
            'start_shift_steps_nonzero': int((np.asarray(shift) != 0).sum()), 'start_shift_max': int(np.max(shift))}
    write_json(Path(out) / 'summary20.json', summ)
    log('build_20hz', out, **summ)
    return summ


def load_dev_v2_20(out: Path = OUT, *, tokens: bool = False) -> dict[str, np.ndarray]:
    """20 Hz Dev v2 (memory time scale).  VLM tokens are the same start observations (row-indexed)."""
    d20 = Path(out) / 'data20'
    names = ('key', 'task', 'episode', 'frame', 'history', 'history_mask', 'state', 'actions', 'hard', 'row',
             'sub_first_unit', 'sub_post_switch', 'sub_dir_change', 'sub_switch_or_dir')
    s = {n: np.load(d20 / f'{n}.npy', allow_pickle=False) for n in names}
    if tokens:
        s['tokens'] = _Rows(np.load(Path(out) / 'data' / 'tokens.npy', mmap_mode='r'), s['row'])
        s['groups'] = np.load(Path(out) / 'data' / 'groups.npy')[s['row']]
    return s


class _Rows:
    """Row-indexed view of a memmap (keeps tokens on disk)."""
    def __init__(self, arr, rows):
        self.arr, self.rows = arr, np.asarray(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, ix):
        r = self.rows[ix]
        if np.ndim(r) == 0:
            return self.arr[int(r)]
        o = np.argsort(r); out = np.asarray(self.arr[r[o]]); return out[np.argsort(o)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('build', 'finalize', 'tasks', 'build20'))
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--batch-size', type=int, default=4)
    a = ap.parse_args()
    if a.cmd == 'tasks':
        print(choose_tasks()); return
    if a.cmd == 'build20':
        print(build_20hz(a.out)); return
    if a.cmd == 'build':
        build(a.out, batch_size=a.batch_size)
    else:
        print(json.dumps(finalize(a.out), indent=1)[:3000])


if __name__ == '__main__':
    main()
