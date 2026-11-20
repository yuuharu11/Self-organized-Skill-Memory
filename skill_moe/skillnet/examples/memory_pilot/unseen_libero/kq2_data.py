#!/usr/bin/env python3
"""KQ2 data: per-step state+action histories at the memory rate (20 Hz) and VLM tokens.

Splits
  eh       LIBERO-40 stride-5 memory frames (12,766; C1b training rows), LeRobot parquet
  learned  LIBERO-40 locked Eval rows (621), LeRobot parquet
  unseen20 LIBERO-90 locked Eval rows at 20 Hz (619; vq_geometry.build_rate_matched)
  dev2     Dev v2 at 20 Hz (667; dev_v2.build_20hz), selection only

For each split we store the whole episode sequence (normalized actions [T,7] and states
[T,8], pi0.5 q01/q99) at the memory time scale and, per row, (episode slot, t) where t is
the start step.  The LeRobot frames are the HDF5 steps after OpenVLA no-op removal, so the
LIBERO-90 sequences are built from HDF5 obs/ee_states + obs/gripper_states on the kept steps.

Query tokens (built at load time, kq2_run.history_tokens):
  token for step tau (tau = t-9 .. t): [a_{tau-1} (7), s_tau (8), action-valid flag]
  i.e. the last 10 executed actions (the same history as C1b) and the states s_{t-9..t}
  (the last one is the current state).  Missing steps (tau < 0) are padding.

VLM tokens: pi0.5 VLM outputs at the start observation from vq-b-01/feats (the 20 Hz
re-extraction for the LIBERO-90 splits, so rows whose start moved because of leading no-ops
use the correct image): 32 image tokens (base+wrist, 4x4 pooled, per-token L2-normalized
VLM outputs) + prompt-mean + prompt-last token = 34 x 2048.
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

ROOT = STATE_DIR / 'outputs/memory-pilot/unseen-libero'
OUT = ROOT / 'kq2-01'
KF_DATA = ROOT / 'key-motion-01/final/data'
VQB = ROOT / 'vq-b-01/feats'
VQC = ROOT / 'vq-c-01'
DEV2 = ROOT / 'dev-v2-01'
RUN_ROOT = STATE_DIR / 'outputs/memory-pilot/pi05-memory40-20261104/run-01'
H5_90 = STATE_DIR / 'data/libero/datasets/libero_90'
NORM = STATE_DIR / 'checkpoints/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json'
SPLITS = ('eh', 'learned', 'unseen20', 'dev2')
VQB_NAME = {'eh': 'eh', 'learned': 'learned', 'unseen20': 'unseen20', 'dev2': 'dev2_20'}


def log(event: str, out: Path = OUT, **kw: Any) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'progress.log').open('a') as f:
        f.write(json.dumps({'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'stage': event, **kw},
                           ensure_ascii=False, default=str) + '\n')


def write_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=float) + '\n')
    tmp.replace(path)


_NS = None


def qnorm(x: np.ndarray, name: str) -> np.ndarray:
    global _NS
    if _NS is None:
        _NS = json.loads(NORM.read_text())['norm_stats']
    lo = np.asarray(_NS[name]['q01'][:x.shape[-1]]); hi = np.asarray(_NS[name]['q99'][:x.shape[-1]])
    return (2 * (np.asarray(x, np.float64) - lo) / (hi - lo) - 1).astype(np.float32)


def noop_keep(actions: np.ndarray) -> np.ndarray:
    import vq_geometry as vg
    return vg.noop_keep(actions)


# ----------------------------------------------------------------------------
# episode sequence readers
# ----------------------------------------------------------------------------

def lerobot_records() -> dict[tuple[int, int], dict]:
    dm = json.loads((RUN_ROOT / 'data_manifest.json').read_text())
    return {(int(x['task']), int(x['episode'])): x for x in dm['episodes']}


def lerobot_seq(rec: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import pyarrow.parquet as pq
    tab = pq.read_table(rec['path'], columns=['actions', 'state'])
    a = np.asarray(tab['actions'].to_pylist(), np.float32); s = np.asarray(tab['state'].to_pylist(), np.float32)
    return qnorm(a, 'actions'), qnorm(s, 'state'), a, s


def hdf5_seq(path: str, demo: str):
    import h5py
    with h5py.File(path, 'r') as fo:
        g = fo['data'][demo]
        act = g['actions'][()]; ees = g['obs/ee_states'][()]; grp = g['obs/gripper_states'][()]
    K = np.flatnonzero(noop_keep(act))
    raw_s = np.concatenate([ees[K], grp[K]], -1).astype(np.float32)
    return qnorm(act[K], 'actions'), qnorm(raw_s, 'state'), K


def libero90_record_fn():
    un = json.loads((ROOT / 'unseen-rows-02/manifest.json').read_text())
    r90 = {(int(x['task_id']), int(x['episode'])): (x['source_hdf5'], x['source_demo']) for x in un['source_records']}
    plan = json.loads((DEV2 / 'plan.json').read_text())
    cache: dict[int, list[str]] = {}

    def dev2(t: int, e: int):
        import h5py
        p = H5_90 / f"{plan['names'][str(t)]}_demo.hdf5"
        if t not in cache:
            with h5py.File(p, 'r') as fo:
                cache[t] = sorted((x for x in fo['data'] if x.startswith('demo_')), key=lambda x: int(x.split('_')[1]))
        return str(p), cache[t][e]
    return r90, dev2


# ----------------------------------------------------------------------------
# build
# ----------------------------------------------------------------------------

def reference_split(name: str) -> dict[str, np.ndarray]:
    if name in ('eh', 'learned'):
        return {k: np.load(KF_DATA / f'{name}__{k}.npy') for k in
                ('key', 'task', 'episode', 'history', 'history_mask', 'state', 'actions')}
    if name == 'unseen20':
        d = VQC / 'unseen20'
        return {k: np.load(d / f'{k}.npy') for k in ('key', 'task', 'episode', 'history', 'history_mask', 'state', 'actions', 'row')}
    import dev_v2
    return dev_v2.load_dev_v2_20()


def build_split(name: str, out: Path = OUT) -> dict[str, Any]:
    d = out / 'data' / name
    if (d / 'DONE').exists():
        return json.loads((d / 'check.json').read_text())
    d.mkdir(parents=True, exist_ok=True)
    ref = reference_split(name)
    keys = np.asarray(ref['key'], np.int64)
    eps = sorted({(int(t), int(e)) for t, e, _ in keys})
    slot = {k: i for i, k in enumerate(eps)}
    A, S, off, ln = [], [], [], []
    starts = np.zeros(len(keys), np.int64)
    r90 = dev2 = recs = None
    if name in ('eh', 'learned'):
        recs = lerobot_records()
    else:
        r90, dev2 = libero90_record_fn()
    ksteps = {}
    o = 0
    for (t, e) in eps:
        if recs is not None:
            a, s, _, _ = lerobot_seq(recs[(t, e)])
        else:
            src = r90[(t, e)] if name == 'unseen20' else dev2(t, e)
            a, s, K = hdf5_seq(*src)
            ksteps[(t, e)] = K
        A.append(a); S.append(s); off.append(o); ln.append(len(a)); o += len(a)
    for i, (t, e, f) in enumerate(keys):
        if recs is not None:
            starts[i] = f
        else:
            starts[i] = int(np.searchsorted(ksteps[(int(t), int(e))], 2 * int(f)))
    seq_a = np.concatenate(A); seq_s = np.concatenate(S)
    off = np.asarray(off, np.int64); ln = np.asarray(ln, np.int64)
    ep_slot = np.asarray([slot[(int(t), int(e))] for t, e, _ in keys], np.int64)
    # ---- consistency with the reference rows ----
    g = off[ep_slot] + starts
    tgt = np.stack([seq_a[g[i]:g[i] + 10] for i in range(len(keys))])
    hist = np.zeros((len(keys), 10, 7), np.float32); hm = np.zeros((len(keys), 10), bool)
    for i in range(len(keys)):
        n = int(min(10, starts[i]))
        if n:
            hist[i, 10 - n:] = seq_a[g[i] - n:g[i]]; hm[i, 10 - n:] = True
    cur = seq_s[g]
    sd = np.abs(cur - ref['state']).max(1)
    chk = {'rows': int(len(keys)), 'episodes': len(eps),
           'actions_max_abs_diff': float(np.abs(tgt - ref['actions']).max()),
           'history_max_abs_diff': float(np.abs(hist - ref['history']).max()),
           'history_mask_equal': bool(np.array_equal(hm, np.asarray(ref['history_mask'], bool))),
           'state_max_abs_diff': float(sd.max()), 'state_rows_diff_gt_1e-3': int((sd > 1e-3).sum())}
    if name == 'eh':
        f5 = keys[:, 2] % 10 == 5
        chk['state_rows_diff_gt_1e-3_frame_mod10_eq5'] = int((sd[f5] > 1e-3).sum())
        chk['state_rows_frame_mod10_eq5'] = int(f5.sum())
        chk['state_max_abs_diff_frame_mod10_eq0'] = float(sd[~f5].max())
        # the KQ-final/CT1 eh cache took the state 5 frames earlier for frame%10==5 rows
        g5 = g - 5
        chk['state_max_abs_diff_vs_frame_minus5_on_mod5'] = float(np.abs(seq_s[g5[f5]] - ref['state'][f5]).max())
    np.save(d / 'seq_actions.npy', seq_a); np.save(d / 'seq_states.npy', seq_s)
    np.save(d / 'ep_offset.npy', off); np.save(d / 'ep_length.npy', ln)
    np.save(d / 'ep_slot.npy', ep_slot); np.save(d / 'start.npy', starts)
    for k in ('key', 'task', 'episode'):
        np.save(d / f'{k}.npy', np.asarray(ref[k]))
    np.save(d / 'actions.npy', np.asarray(ref['actions'], np.float32))
    np.save(d / 'history.npy', np.asarray(ref['history'], np.float32))
    np.save(d / 'history_mask.npy', np.asarray(ref['history_mask'], bool))
    np.save(d / 'state_ref.npy', np.asarray(ref['state'], np.float32))
    np.save(d / 'state.npy', cur)
    write_json(d / 'check.json', chk)
    (d / 'DONE').write_text('ok\n')
    log('build_split', out, split=name, **chk)
    return chk


def build_vlm(name: str, out: Path = OUT) -> None:
    d = out / 'data' / name
    p = d / 'vlm.npy'
    if (d / 'VLM_DONE').exists():
        return
    src = VQB / VQB_NAME[name]
    key = np.load(src / 'key.npy')
    if not np.array_equal(key, np.load(d / 'key.npy')) or not np.load(src / 'done.npy').all():
        raise AssertionError(f'vq-b feature rows do not match {name}')
    img = np.load(src / 'vlm_img.npy', mmap_mode='r')
    tm = np.load(src / 'vlm_txt_mean.npy', mmap_mode='r'); tl = np.load(src / 'vlm_txt_last.npy', mmap_mode='r')
    mm = np.lib.format.open_memmap(p, mode='w+', dtype=np.float16, shape=(len(key), 34, 2048))
    for s in range(0, len(key), 512):
        e = min(s + 512, len(key))
        mm[s:e, :32] = img[s:e]
        mm[s:e, 32] = np.asarray(tm[s:e], np.float16)
        mm[s:e, 33] = np.asarray(tl[s:e], np.float16)
    mm.flush()
    (d / 'VLM_DONE').write_text('ok\n')
    log('build_vlm', out, split=name, rows=int(len(key)))


def state_definition_check(out: Path = OUT, tasks_n: int = 3, demos_per_task: int = 2) -> dict[str, Any]:
    """LeRobot state vs HDF5 obs/ee_states+gripper_states on the SAME LIBERO-40 episodes.

    LeRobot LIBERO-40 frames are HDF5 steps after no-op removal (vq-c-01/align40.json gives the
    exact demo and kept steps, max action error 0).  We range-read obs/ee_states and
    obs/gripper_states of those HDF5 demos and compare raw and normalized values per dim."""
    import fsspec
    import h5py
    import urllib.request
    import vq_geometry as vg
    p = out / 'state_definition_check.json'
    if p.exists():
        return json.loads(p.read_text())
    al = json.loads((VQC / 'align40.json').read_text())
    tasks = {int(k): v for k, v in json.loads((VQC / 'hdf5_states' / 'tasks.json').read_text()).items()}
    recs = lerobot_records()
    by_task: dict[int, list[str]] = {}
    for k in al:
        t = int(k.split('_')[0])
        by_task.setdefault(t, []).append(k)
    pick = [0, 13, 25][:tasks_n]  # one per suite family (libero_10 / goal / object-ish)
    rows = []
    raw_diffs, norm_diffs = [], []
    for t in pick:
        info = tasks[t]
        url = f"{vg.HF_BASE}/{info['suite']}/{info['name']}_demo.hdf5"

        class _NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *a, **k):
                return None
        try:
            urllib.request.build_opener(_NoRedirect).open(urllib.request.Request(url, method='HEAD'))
            cdn = url
        except urllib.error.HTTPError as e:
            cdn = e.headers['Location']
        fs = fsspec.filesystem('https')
        with fs.open(cdn, 'rb', block_size=2 ** 20, cache_type='blockcache') as f, h5py.File(f, 'r') as h:
            for k in by_task[t][:demos_per_task]:
                a = al[k]; e = int(k.split('_')[1])
                g = h['data'][a['demo']]
                ees = g['obs/ee_states'][()]; grp = g['obs/gripper_states'][()]
                steps = np.asarray(a['steps'])
                hs = np.concatenate([ees[steps], grp[steps]], -1).astype(np.float32)
                _, sn, _, sraw = lerobot_seq(recs[(t, e)])
                n = min(len(hs), len(sraw))
                rd = np.abs(hs[:n] - sraw[:n]); nd = np.abs(qnorm(hs[:n], 'state') - sn[:n])
                raw_diffs.append(rd); norm_diffs.append(nd)
                rows.append({'task': t, 'episode': e, 'demo': a['demo'], 'steps': int(n),
                             'raw_max_abs_per_dim': rd.max(0).tolist(), 'raw_mean_abs_per_dim': rd.mean(0).tolist(),
                             'norm_max_abs_per_dim': nd.max(0).tolist(), 'norm_mean_abs_per_dim': nd.mean(0).tolist(),
                             'lerobot_raw_first': sraw[0].tolist(), 'hdf5_raw_first': hs[0].tolist()})
    rd = np.concatenate(raw_diffs); nd = np.concatenate(norm_diffs)
    res = {'episodes': rows, 'raw_max_abs_per_dim': rd.max(0).tolist(), 'raw_mean_abs_per_dim': rd.mean(0).tolist(),
           'norm_max_abs_per_dim': nd.max(0).tolist(), 'norm_mean_abs_per_dim': nd.mean(0).tolist(),
           'norm_p99_abs_per_dim': np.percentile(nd, 99, axis=0).tolist(),
           'note': 'state = [ee_pos(3), ee_ori axis-angle(3), gripper qpos(2)]; LeRobot vs HDF5 obs on aligned steps'}
    write_json(p, res)
    log('state_definition_check', out, raw_max=res['raw_max_abs_per_dim'], norm_max=res['norm_max_abs_per_dim'])
    return res


def state_distribution_check(out: Path = OUT) -> dict[str, Any]:
    """Per-dim normalized-state statistics of every split (LeRobot vs HDF5 rebuilt)."""
    res = {}
    for n in SPLITS:
        s = np.load(out / 'data' / n / 'seq_states.npy')
        res[n] = {'mean': s.mean(0).tolist(), 'std': s.std(0).tolist(), 'p01': np.percentile(s, 1, 0).tolist(),
                  'p99': np.percentile(s, 99, 0).tolist()}
    write_json(out / 'state_distribution.json', res)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('build', 'state_check'))
    ap.add_argument('--splits', nargs='*', default=list(SPLITS))
    a = ap.parse_args()
    if a.cmd == 'build':
        for s in a.splits:
            print(s, build_split(s), flush=True)
            build_vlm(s)
        state_distribution_check()
    else:
        print(json.dumps(state_definition_check(), indent=1)[:4000])


if __name__ == '__main__':
    main()
