#!/usr/bin/env python3
"""VQ-C: is low-dimensional task geometry what the retrieval query is missing?

Step 1 (PRIVILEGED upper bound, diagnostic only): simulator geometry of the
instruction's goal entities relative to the end-effector (+ grasp / contact flags)
is injected into the C1b history+state MLP query by FiLM (identity init).
Key = X1 (MK1b) fixed, recipe = C1b (key_final.train_row, rows='eh', same_scene,
tau .01, lr 1e-4, 16 epochs, batch 8 + 256 random candidates, no augmentation).

Data sources
  LIBERO-40 (eh / learned rows): the LeRobot copy has no simulator state.  The
    official LIBERO HDF5 files (HF yifengzhu-hf/LIBERO-datasets) are read with HTTP
    range requests, only ``actions``, ``states`` and ``obs/ee_pos`` (no images).
    LeRobot episodes are the no-noop replays of those demos, so each LeRobot
    episode is matched to its demo by exact action equality after removing no-ops
    (OpenVLA rule: ||a[:6]|| < 1e-4 and gripper equal to the previous action),
    which also gives the frame -> HDF5 step map.
  LIBERO-90 (unseen / dev rows): local HDF5, frame f -> step 2f (as unseen-rows-02).

Target entities (rule): the goal atoms of the task BDDL, in order.  For
  On/In(a, b): a then b; Open/Close/Turnon/Turnoff(x): x.  Deduplicated, first 4
  kept.  An entity that is an object -> its body position; a region / site
  name -> that site's position (object-level fallback: the object's body).
Per-row geometry (all in the world frame, relative to the eef site):
  for each slot k<4: rel xyz (3), distance (1), valid (1), finger contact (1)
  plus: any-goal-object grasped (both fingers in contact), focus rel xyz + dist
  where focus = first goal atom not yet satisfied (env predicate check):
  its moved object a if a is not grasped, else its destination b.
No future, task id or ground-truth action enters these features; they are
privileged because they come from the simulator state.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[5]
STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', REPO_ROOT / '.skillnet-state')).expanduser()
sys.path.insert(0, str(HERE))

ROOT = (STATE_DIR / 'outputs/memory-pilot/unseen-libero')
OUT = ROOT / 'vq-c-01'
KF_DATA = ROOT / 'key-motion-01/final/data'
RUN_ROOT = (STATE_DIR / 'outputs/memory-pilot/pi05-memory40-20261104/run-01')
HF_BASE = 'https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/main'
DEV_V2 = ROOT / 'dev-v2-01'
SLOTS = 4
GEO_DIM = SLOTS * 6 + 1 + 4  # 29


def log(event: str, out: Path = OUT, **kw: Any) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'progress.log').open('a') as f:
        f.write(json.dumps({'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'stage': event, **kw},
                           ensure_ascii=False, default=str) + '\n')


def write_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=float) + '\n')
    tmp.replace(path)


# ----------------------------------------------------------------------------
# task tables
# ----------------------------------------------------------------------------

def libero40_tasks() -> dict[int, dict[str, str]]:
    """LeRobot task index -> {'suite','name','language'} via the instruction string."""
    from libero.libero import benchmark
    d = json.loads((RUN_ROOT / 'data_manifest.json').read_text())
    lang = {}
    bd = benchmark.get_benchmark_dict()
    for s in ('libero_10', 'libero_goal', 'libero_object', 'libero_spatial'):
        b = bd[s]()
        for i in range(b.n_tasks):
            t = b.get_task(i)
            lang[t.language] = {'suite': s, 'name': t.name, 'language': t.language}
    return {int(k): lang[v] for k, v in d['tasks'].items()}


def libero90_tasks() -> dict[int, dict[str, str]]:
    from libero.libero import benchmark
    b = benchmark.get_benchmark_dict()['libero_90']()
    return {i: {'suite': 'libero_90', 'name': b.get_task(i).name, 'language': b.get_task(i).language}
            for i in range(b.n_tasks)}


def bddl_path(suite: str, name: str) -> str:
    from libero.libero import get_libero_path
    return os.path.join(get_libero_path('bddl_files'), suite, name + '.bddl')


def goal_entities(bddl_file: str) -> tuple[list[str], list[tuple[str, ...]]]:
    """Ordered goal entities and goal atoms from the BDDL :goal block (rule in module doc)."""
    txt = ' '.join(Path(bddl_file).read_text().split())
    g = txt[txt.index('(:goal'):]
    atoms = [tuple(m.group(1).split()) for m in re.finditer(r'\((?!And)(\w+ [\w ]+?)\)', g)]
    ents: list[str] = []
    for a in atoms:
        for x in a[1:]:
            if x not in ents:
                ents.append(x)
    return ents[:SLOTS], atoms


# ----------------------------------------------------------------------------
# stage 1: remote partial reads of LIBERO-40 HDF5 (no images)
# ----------------------------------------------------------------------------

def _fetch_one(task: int, info: dict[str, str], dest: Path) -> str:
    p = dest / f'task{task:02d}.npz'
    if p.exists():
        return 'skip'
    import fsspec
    import h5py
    import urllib.request
    url = f"{HF_BASE}/{info['suite']}/{info['name']}_demo.hdf5"
    fs = fsspec.filesystem('https')
    arrays = {}
    for attempt in range(8):
        try:
            # resolve the HF redirect once (HF limits resolver calls); range-read the signed CDN URL
            class _NoRedirect(urllib.request.HTTPRedirectHandler):
                def redirect_request(self, *a, **k):
                    return None
            try:
                urllib.request.build_opener(_NoRedirect).open(urllib.request.Request(url, method='HEAD'))
                cdn = url
            except urllib.error.HTTPError as e:
                if e.code not in (301, 302, 303, 307, 308):
                    raise
                cdn = e.headers['Location']
            with fs.open(cdn, 'rb', block_size=2 ** 20, cache_type='blockcache') as f, h5py.File(f, 'r') as h:
                d = h['data']
                for k in sorted(d.keys(), key=lambda s: int(s.split('_')[1])):
                    if f'{k}__states' in arrays:
                        continue
                    g = d[k]
                    arrays[f'{k}__actions'] = g['actions'][()].astype(np.float32)
                    arrays[f'{k}__states'] = g['states'][()]
                    arrays[f'{k}__ee_pos'] = g['obs/ee_pos'][()].astype(np.float32)
            break
        except Exception as e:  # network hiccup: retry, keep what we have
            log('fetch_retry', task=task, attempt=attempt, error=repr(e))
            time.sleep(30 * (attempt + 1))
    else:
        raise RuntimeError(f'fetch failed {task}')
    tmp = dest / f'task{task:02d}.tmp.npz'
    np.savez(tmp, **arrays)
    tmp.replace(p)
    return url


def fetch40(out: Path = OUT, workers: int = 6) -> None:
    dest = out / 'hdf5_states'
    dest.mkdir(parents=True, exist_ok=True)
    tasks = libero40_tasks()
    write_json(dest / 'tasks.json', tasks)
    import multiprocessing as mp
    with ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn')) as ex:
        futs = {t: ex.submit(_fetch_one, t, info, dest) for t, info in tasks.items()}
        for t, fu in futs.items():
            log('fetch_done', task=t, source=fu.result())
    log('fetch_complete')


# ----------------------------------------------------------------------------
# stage 2: align LeRobot episodes to HDF5 demos (exact actions after no-op removal)
# ----------------------------------------------------------------------------

def noop_keep(actions: np.ndarray) -> np.ndarray:
    keep = np.ones(len(actions), bool)
    prev = None
    for i, a in enumerate(actions):  # OpenVLA is_noop; prev = last kept action
        small = np.linalg.norm(a[:-1]) < 1e-4
        if small and (prev is None or a[-1] == prev[-1]):
            keep[i] = False
            continue
        prev = a
    return keep


def align40(out: Path = OUT) -> None:
    import pyarrow.parquet as pq
    dm = json.loads((RUN_ROOT / 'data_manifest.json').read_text())
    need = set()
    for s in ('eh', 'learned'):
        k = np.load(KF_DATA / f'{s}__key.npy')
        need |= {(int(a), int(b)) for a, b, _ in k}
    recs = {(int(x['task']), int(x['episode'])): x for x in dm['episodes']}
    res, stats = {}, []
    cache: dict[int, dict] = {}
    for (t, e) in sorted(need):
        if t not in cache:
            z = np.load(out / 'hdf5_states' / f'task{t:02d}.npz')
            demos = sorted({k.split('__')[0] for k in z.files}, key=lambda s: int(s.split('_')[1]))
            cache = {t: {dname: (z[f'{dname}__actions'], noop_keep(z[f'{dname}__actions'])) for dname in demos}}
        tab = pq.read_table(recs[(t, e)]['path'], columns=['actions', 'state'])
        act = np.asarray(tab['actions'].to_pylist(), np.float32)
        best = None
        for dname, (a, keep) in cache[t].items():
            ak = a[keep]
            if len(ak) != len(act):
                continue
            err = float(np.abs(ak - act).max())
            if best is None or err < best[1]:
                best = (dname, err)
        if best is None:  # fall back: nearest by length-truncated error
            for dname, (a, keep) in cache[t].items():
                ak = a[keep]
                n = min(len(ak), len(act))
                err = float(np.abs(ak[:n] - act[:n]).max()) + abs(len(ak) - len(act))
                if best is None or err < best[1]:
                    best = (dname, err)
        steps = np.flatnonzero(cache[t][best[0]][1])
        res[f'{t}_{e}'] = {'demo': best[0], 'max_abs_action_err': best[1], 'steps': steps.tolist(),
                           'lerobot_len': int(len(act))}
        stats.append(best[1])
    write_json(out / 'align40.json', res)
    log('align_complete', episodes=len(res), max_err=float(np.max(stats)), n_bad=int(np.sum(np.asarray(stats) > 1e-3)))


# ----------------------------------------------------------------------------
# stage 3: simulator geometry
# ----------------------------------------------------------------------------

class SimGeometry:
    def __init__(self, suite: str, name: str):
        from libero.libero.envs import OffScreenRenderEnv
        self.env = OffScreenRenderEnv(bddl_file_name=bddl_path(suite, name), camera_heights=64, camera_widths=64,
                                      has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
        self.env.reset()
        self.ents, self.atoms = goal_entities(bddl_path(suite, name))
        inner = self.env.env
        self.inner = inner
        self.sim = inner.sim
        self.eef_site = inner.robots[0].eef_site_id
        self.grip = inner.robots[0].gripper
        self.finger_geoms = (set(self.grip.important_geoms['left_fingerpad']) | set(self.grip.important_geoms['left_finger']),
                             set(self.grip.important_geoms['right_fingerpad']) | set(self.grip.important_geoms['right_finger']))
        self.ent_kind = {e: self._resolve(e) for e in self.ents}

    def _resolve(self, e: str):
        inner = self.inner
        m = self.sim.model
        objs = getattr(inner, 'objects_dict', {}) | getattr(inner, 'fixtures_dict', {})
        geoms_of = lambda o: set(o.contact_geoms) if hasattr(o, 'contact_geoms') else set()
        if e in objs:
            o = objs[e]
            return ('body', m.body_name2id(o.root_body), {m.geom_name2id(g) for g in geoms_of(o) if g in m.geom_names})
        # region / site names (e.g. wooden_cabinet_1_top_region)
        for cand in (e, e.replace('_region', ''), ):
            if cand in m.site_names:
                return ('site', m.site_name2id(cand), set())
        for oname, o in objs.items():
            if e.startswith(oname + '_'):
                suff = e[len(oname) + 1:]
                sites = [s for s in m.site_names if s.startswith(oname) and suff in s]
                if sites:
                    return ('site', m.site_name2id(sites[0]), set())
                return ('body', m.body_name2id(o.root_body), {m.geom_name2id(g) for g in geoms_of(o) if g in m.geom_names})
        raise KeyError(e)

    def _contacts(self) -> tuple[set, set]:
        d = self.sim.data
        m = self.sim.model
        lf, rf = set(), set()
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = m.geom_id2name(c.geom1), m.geom_id2name(c.geom2)
            for a, b in ((g1, c.geom2), (g2, c.geom1)):
                if a in self.finger_geoms[0]:
                    lf.add(b)
                if a in self.finger_geoms[1]:
                    rf.add(b)
        return lf, rf

    def _atom_ok(self, atom) -> bool:
        try:
            return bool(self.inner._eval_predicate(list(atom)))
        except Exception:
            try:
                return bool(self.inner._eval_predicate([atom[0].lower(), *atom[1:]]))
            except Exception:
                return False

    def features(self, state: np.ndarray) -> tuple[np.ndarray, dict]:
        self.sim.set_state_from_flattened(state)
        self.sim.forward()
        d = self.sim.data
        eef = np.asarray(d.site_xpos[self.eef_site]).copy()
        lf, rf = self._contacts()
        f = np.zeros(GEO_DIM, np.float32)
        pos, grasped = {}, {}
        for k, e in enumerate(self.ents):
            kind, idx, geoms = self.ent_kind[e]
            p = np.asarray(d.site_xpos[idx] if kind == 'site' else d.xpos[idx])
            rel = p - eef
            pos[e] = rel
            touch_l, touch_r = bool(geoms & lf), bool(geoms & rf)
            grasped[e] = touch_l and touch_r
            f[k * 6:k * 6 + 3] = rel
            f[k * 6 + 3] = np.linalg.norm(rel)
            f[k * 6 + 4] = 1.
            f[k * 6 + 5] = float(touch_l or touch_r)
        f[SLOTS * 6] = float(any(grasped.values()))
        focus = None
        for atom in self.atoms:
            if self._atom_ok(atom):
                continue
            if len(atom) == 3:
                a, b = atom[1], atom[2]
                focus = b if grasped.get(a, False) else a
            else:
                focus = atom[1]
            break
        if focus is None:
            focus = self.ents[-1]
        if focus not in pos:  # entity beyond the 4 slots
            kind, idx, _ = self._resolve(focus)
            p = np.asarray(d.site_xpos[idx] if kind == 'site' else d.xpos[idx])
            pos[focus] = p - eef
        f[SLOTS * 6 + 1:SLOTS * 6 + 4] = pos[focus]
        f[SLOTS * 6 + 4] = np.linalg.norm(pos[focus])
        return f, {'focus': focus}

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


def row_states(split: str, out: Path = OUT):
    """Yield (task_name_info, rows idx, states) grouped by task for a split."""
    keys = split_keys(split)
    if split in ('eh', 'learned'):
        tasks = libero40_tasks()
        al = json.loads((out / 'align40.json').read_text())
        for t in sorted(set(keys[:, 0].tolist())):
            z = np.load(out / 'hdf5_states' / f'task{t:02d}.npz')
            ix = np.flatnonzero(keys[:, 0] == t)
            st, eefp = [], []
            for i in ix:
                _, e, fr = map(int, keys[i])
                a = al[f'{t}_{e}']
                step = a['steps'][min(fr, len(a['steps']) - 1)]
                st.append(z[f"{a['demo']}__states"][step])
                eefp.append(z[f"{a['demo']}__ee_pos"][step])
            yield tasks[t], ix, np.stack(st), np.stack(eefp)
    else:
        import h5py
        tasks = libero90_tasks()
        recs = _libero90_records()
        for t in sorted(set(keys[:, 0].tolist())):
            ix = np.flatnonzero(keys[:, 0] == t)
            st, eefp = [], []
            for i in ix:
                _, e, fr = map(int, keys[i])
                rec = recs(t, e)
                with h5py.File(rec['source_hdf5'], 'r') as fo:
                    g = fo['data'][rec['source_demo']]
                    n = len(g['states'])
                    idx = min(fr * 2, n - 1)
                    st.append(g['states'][idx])
                    eefp.append(g['obs/ee_pos'][idx])
            yield tasks[t], ix, np.stack(st), np.stack(eefp)


def _libero90_records():
    un = json.loads((ROOT / 'unseen-rows-02/manifest.json').read_text())
    r90 = {(int(x['task_id']), int(x['episode'])): x for x in un['source_records']}

    def get(t, e):
        if (t, e) in r90:
            return r90[(t, e)]
        import dev_v2
        return dev_v2.source_record(t, e)
    return get


def split_keys(name: str) -> np.ndarray:
    if name == 'dev2':
        import dev_v2
        return np.asarray(dev_v2.load_dev()['key'], np.int64)
    return np.load(KF_DATA / f'{name}__key.npy').astype(np.int64)


def geometry(split: str, out: Path = OUT) -> None:
    gd = out / 'geometry'
    gd.mkdir(parents=True, exist_ok=True)
    p = gd / f'{split}.npz'
    if p.exists():
        return
    keys = split_keys(split)
    feats = np.zeros((len(keys), GEO_DIM), np.float32)
    focus = np.empty(len(keys), object)
    eef_gap = np.zeros(len(keys), np.float32)
    ents = {}
    for info, ix, st, eefp in row_states(split, out):
        sg = SimGeometry(info['suite'], info['name'])
        ents[info['name']] = sg.ents
        for j, i in enumerate(ix):
            f, meta = sg.features(st[j])
            feats[i] = f
            focus[i] = meta['focus']
            eef_gap[i] = np.linalg.norm(np.asarray(sg.sim.data.site_xpos[sg.eef_site]) - eefp[j])
        sg.close()
        log('geometry_task', split=split, task=info['name'], rows=len(ix))
    np.savez(p, feats=feats, focus=focus.astype(str), eef_gap=eef_gap, key=keys)
    write_json(gd / f'{split}_entities.json', ents)
    log('geometry_complete', split=split, rows=len(keys), eef_gap_max=float(eef_gap.max()))


# ----------------------------------------------------------------------------
# stage 4: FiLM query on C1b + training via key_final.train_row
# ----------------------------------------------------------------------------

def _models():
    import torch
    from torch import nn
    from torch.nn import functional as F
    import key_final as kf

    class GeoFiLMQuery(nn.Module):
        """C1b tower 88->512-GELU->256 with FiLM on the hidden layer from geometry (zero-init => C1b)."""
        def __init__(self, geo_dim: int, mode: str = 'film'):
            super().__init__()
            self.mode = mode
            in_dim = 88 + (geo_dim if mode == 'concat' else 0)
            self.l1, self.l2 = nn.Linear(in_dim, 512), nn.Linear(512, 256)
            self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / .07), dtype=torch.float32))
            if mode == 'film':
                self.cond = nn.Sequential(nn.Linear(geo_dim, 128), nn.GELU(), nn.Linear(128, 1024))
                nn.init.zeros_(self.cond[-1].weight)
                nn.init.zeros_(self.cond[-1].bias)

        def forward(self, *, hs, geo=None, **_):
            if self.mode == 'concat':
                return F.normalize(self.l2(F.gelu(self.l1(torch.cat([hs.float(), geo.float()], -1)))), dim=-1)
            h = F.gelu(self.l1(hs.float()))
            if self.mode == 'film':
                g, b = self.cond(geo.float()).chunk(2, -1)
                h = (1 + g) * h + b
            return F.normalize(self.l2(h), dim=-1)

    return kf, GeoFiLMQuery


FEATURE_SETS = {
    # columns of the 29-d geometry vector
    'full': list(range(GEO_DIM)),
    'focus_grasp': [SLOTS * 6] + list(range(SLOTS * 6 + 1, SLOTS * 6 + 5)),   # 5 numbers
    'slots': list(range(SLOTS * 6 + 1)),
}


def run_rows(plan: dict[str, dict], out: Path = OUT, device: str = 'cuda:0', dev_split: str = 'dev') -> None:
    import torch
    kf, GeoFiLMQuery = _models()
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
    rows_dir = out / 'rows'
    rows_dir.mkdir(parents=True, exist_ok=True)
    _, ctx = _context(out, device, dev_split)

    orig_make, orig_pack = kf.make_query, kf.pack
    for name, cfg in plan.items():
        if (rows_dir / f'{name}.json').exists():
            continue
        cols = FEATURE_SETS[cfg.get('feat', 'full')]
        mode = cfg.get('mode', 'film')

        def make_query(c, _cols=cols, _mode=mode):
            if c.get('geo_mode', 'none') == 'none':
                return orig_make(c)
            return GeoFiLMQuery(len(_cols), _mode)

        def pack(split, ids, c, dev, *, rng=None, augment=False, mhist=None, _cols=cols):
            o = orig_pack(split, ids, c, dev, rng=rng, augment=augment, mhist=mhist)
            if c.get('geo_mode', 'none') != 'none':
                g = np.asarray(split['geo'][np.asarray(ids, np.int64)][:, _cols], np.float32)
                o['geo'] = torch.as_tensor(g, device=dev)
            return o
        kf.make_query, kf.pack = make_query, pack
        try:
            log('row_start', out, name=name, cfg=cfg)
            r = kf.train_row(name, {**cfg, 'query': 'hsmlp'}, ctx, save_dir=rows_dir, save_scores=False)
            ev = r['evaluation']
            log('row_complete', out, name=name, dev=r['dev_top1_arm6_mse'], epoch=r['selected_epoch'],
                learned=ev['learned']['all']['top1_arm6_mse'], unseen=ev['unseen']['all']['top1_arm6_mse'])
        finally:
            kf.make_query, kf.pack = orig_make, orig_pack
        torch.cuda.empty_cache()


def _context(out: Path, device: str, dev_split: str = 'dev'):
    import torch
    kf, _ = _models()
    ctx = kf.RowContext(KF_DATA.parent, torch.device(device), False)
    geo = {s: np.load(out / 'geometry' / f'{s}.npz', allow_pickle=False)['feats']
           for s in ('eh', 'learned', 'unseen', 'dev')}
    mu, sd = geo['eh'].mean(0), geo['eh'].std(0) + 1e-3
    for s, g in geo.items():
        ctx.splits[s]['geo'] = ((g - mu) / sd).astype(np.float32)
    if dev_split == 'dev20':  # 20 Hz Dev (rate matched to the memory) replaces the Dev used for epoch selection
        import key_shared_latent as k2
        sp = load_rate_matched('dev20', out)
        sp['task'] = sp['task'].astype(np.int64); sp['episode'] = sp['episode'].astype(np.int64)
        sp['hard'] = sp['hard'].astype(bool)
        sp['errors'] = k2._arm6_candidate_mse(sp['actions'], ctx.windows)
        sp['hs'] = kf.hs_features(sp)
        sp['geo'] = ((sp['geo_raw'] - mu) / sd).astype(np.float32)
        ctx.splits['dev'] = sp
    elif dev_split == 'devv2':  # VQ-A Dev v2 at 20 Hz (12 tasks, 667 rows)
        import dev_v2
        import key_shared_latent as k2
        sp = dev_v2.load_dev_v2_20()
        sp['task'] = sp['task'].astype(np.int64); sp['episode'] = sp['episode'].astype(np.int64)
        sp['hard'] = sp['hard'].astype(bool)
        sp['errors'] = k2._arm6_candidate_mse(sp['actions'], ctx.windows)
        sp['hs'] = kf.hs_features(sp)
        sp['geo'] = ((np.load(out / 'geometry' / 'devv2_20.npz')['feats'] - mu) / sd).astype(np.float32)
        ctx.splits['dev'] = sp
    elif dev_split != 'dev':
        raise ValueError(dev_split)
    return kf, ctx


def _eval_split20(sname: str, ctx, kf, out: Path = OUT) -> dict:
    import key_shared_latent as k2
    sp = load_rate_matched(sname, out)
    sp['task'] = sp['task'].astype(np.int64); sp['episode'] = sp['episode'].astype(np.int64)
    sp['hard'] = sp['hard'].astype(bool)
    sp['errors'] = k2._arm6_candidate_mse(sp['actions'], ctx.windows)
    sp['hs'] = kf.hs_features(sp)
    g = np.load(out / 'geometry' / 'eh.npz')['feats']
    sp['geo'] = ((sp['geo_raw'] - g.mean(0)) / (g.std(0) + 1e-3)).astype(np.float32)
    return sp


def ablate(names: list[str], out: Path = OUT, device: str = 'cuda:0', seed: int = 9001) -> dict:
    """Test-time input destruction on trained geometry rows (geometry shuffled / zeroed, history zeroed)."""
    import torch
    kf, GeoFiLMQuery = _models()
    _, ctx = _context(out, device)
    res = {}
    orig_make, orig_pack = kf.make_query, kf.pack
    for name in names:
        blob = torch.load(out / 'rows' / f'{name}.pt', map_location='cpu', weights_only=False)
        cfg = blob['cfg']
        cols = FEATURE_SETS[cfg.get('feat', 'full')]
        kf.make_query = lambda c, _c=cols, _m=cfg.get('mode', 'film'): GeoFiLMQuery(len(_c), _m)

        def pack(split, ids, c, dev, *, rng=None, augment=False, mhist=None, _cols=cols):
            o = orig_pack(split, ids, c, dev, rng=rng, augment=augment, mhist=mhist)
            o['geo'] = torch.as_tensor(np.asarray(split['geo'][np.asarray(ids, np.int64)][:, _cols], np.float32), device=dev)
            return o
        kf.pack = pack
        try:
            cfg, q, ks = kf.load_row_model(out / 'rows' / f'{name}.pt', ctx)
            out_n = {}
            for sname in ('learned', 'unseen', 'unseen20'):
                sp = ctx.splits[sname] if sname in ctx.splits else _eval_split20(sname, ctx, kf, out)
                perm = np.random.default_rng(seed).permutation(len(sp['key']))
                variants = {'intact': sp,
                            'geo_shuffled': {**sp, 'geo': sp['geo'][perm]},
                            'geo_mean': {**sp, 'geo': np.zeros_like(sp['geo'])},
                            'history_zeroed': {**sp, 'history': np.zeros_like(np.asarray(sp['history'])),
                                               'history_mask': np.zeros_like(np.asarray(sp['history_mask']))}}
                r = {}
                for vn, v in variants.items():
                    sc, _ = kf.score_split(q, ks, cfg, ctx, v)
                    r[vn] = float(sp['errors'][np.arange(len(sc)), sc.argmax(1)].mean())
                out_n[sname] = r
            res[name] = out_n
            log('ablate', out, name=name, result=out_n)
        finally:
            kf.make_query, kf.pack = orig_make, orig_pack
    write_json(out / 'ablation.json', res)
    return res


def _install_dev2(ctx, kf) -> None:
    raise NotImplementedError('dev2 not available')


# ----------------------------------------------------------------------------
# stage 5: subsets, paired bootstrap, report tables
# ----------------------------------------------------------------------------

def subsets(split: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Rows where a VLM / geometry should matter (definitions in the report).

    first   : no valid history (frame 0, the first unit of the episode)
    switch  : the gripper command changes inside the 10-step history window or between
              the last history step and the first target step (scoring only)
    turn    : cos(mean xyz of last 5 history actions, mean xyz of the next 10 target
              actions) < .5, both mean norms > .1 (normalized action units; scoring only)
    post    : switch | turn (rows with history only)
    """
    h = np.asarray(split['history'], np.float32)
    m = np.asarray(split['history_mask'], bool)
    a = np.asarray(split['actions'], np.float32)
    valid = m.sum(1)
    first = valid == 0
    hg = np.sign(h[..., 6])
    sw = np.zeros(len(h), bool)
    for i in range(len(h)):
        g = hg[i][m[i]]
        if len(g) == 0:
            continue
        sw[i] = (len(np.unique(g)) > 1) or (g[-1] != np.sign(a[i, 0, 6]))
    hv = np.where(m[:, -5:, None], h[:, -5:, :3], 0).sum(1) / np.maximum(m[:, -5:].sum(1, keepdims=True), 1)
    fv = a[:, :, :3].mean(1)
    nh, nf = np.linalg.norm(hv, axis=1), np.linalg.norm(fv, axis=1)
    cos = (hv * fv).sum(1) / np.maximum(nh * nf, 1e-8)
    turn = (valid > 0) & (nh > .1) & (nf > .1) & (cos < .5)
    sw = sw & (valid > 0)
    return {'all': np.ones(len(h), bool), 'first': first, 'switch': sw, 'turn': turn, 'post': sw | turn}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, episodes: np.ndarray, n: int = 2000, seed: int = 0):
    """Mean (a-b) and an episode-cluster bootstrap 95% CI."""
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    ue = np.unique(episodes)
    idx = [np.flatnonzero(episodes == e) for e in ue]
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        pick = rng.integers(len(ue), size=len(ue))
        rows = np.concatenate([idx[p] for p in pick])
        bs.append(d[rows].mean())
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


C1B_ROWS = ROOT / 'key-motion-01/final/rows'
NORM = (STATE_DIR / 'checkpoints/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json')


# ----------------------------------------------------------------------------
# side check: the LIBERO-90 Eval rows are sampled every other HDF5 step (10 Hz) while the
# memory / LIBERO-40 rows are every (non-noop) 20 Hz control step.  Rebuild the same Eval
# start times at the memory rate (no-op removal as OpenVLA) and re-score trained queries.
# ----------------------------------------------------------------------------

def build_rate_matched(split: str = 'unseen', out: Path = OUT) -> None:
    import h5py
    d = out / f'{split}20'
    if (d / 'DONE').exists():
        return
    d.mkdir(parents=True, exist_ok=True)
    ns = json.loads(NORM.read_text())['norm_stats']
    q = lambda x, n: 2 * (x - np.asarray(ns[n]['q01'][:x.shape[-1]])) / (
        np.asarray(ns[n]['q99'][:x.shape[-1]]) - np.asarray(ns[n]['q01'][:x.shape[-1]])) - 1
    keys = split_keys(split)
    recs = _libero90_records()
    tasks = libero90_tasks()
    H, M, S, A, ST, keep_rows = [], [], [], [], [], []
    for i, (t, e, f) in enumerate(keys):
        rec = recs(int(t), int(e))
        with h5py.File(rec['source_hdf5'], 'r') as fo:
            g = fo['data'][rec['source_demo']]
            act = g['actions'][()]
            ees = g['obs/ee_states'][()]
            grp = g['obs/gripper_states'][()]
            sts = g['states'][()]
        K = np.flatnonzero(noop_keep(act))
        p = int(np.searchsorted(K, 2 * int(f)))
        if p + 10 > len(K):
            continue
        hist = np.zeros((10, 7), np.float32)
        m = np.zeros(10, bool)
        hk = K[max(0, p - 10):p]
        if len(hk):
            hist[10 - len(hk):] = q(act[hk], 'actions')
            m[10 - len(hk):] = True
        H.append(hist); M.append(m)
        S.append(q(np.concatenate([ees[K[p]], grp[K[p]]]), 'state').astype(np.float32))
        A.append(q(act[K[p:p + 10]], 'actions').astype(np.float32))
        ST.append(sts[K[p]])
        keep_rows.append(i)
    keep_rows = np.asarray(keep_rows)
    if (KF_DATA / f'{split}__task.npy').exists():
        base = {k: np.load(KF_DATA / f'{split}__{k}.npy') for k in ('task', 'episode', 'hard')}
    else:  # e.g. VQ-A dev2: no cached arrays, take task/episode from the keys
        base = {'task': keys[:, 0], 'episode': keys[:, 1], 'hard': np.zeros(len(keys), bool)}
    np.save(d / 'key.npy', keys[keep_rows])
    np.save(d / 'row.npy', keep_rows)
    for k, v in base.items():
        np.save(d / f'{k}.npy', v[keep_rows])
    np.save(d / 'history.npy', np.stack(H)); np.save(d / 'history_mask.npy', np.stack(M))
    np.save(d / 'state.npy', np.stack(S)); np.save(d / 'actions.npy', np.stack(A))
    # geometry at the matched step
    feats = np.zeros((len(keep_rows), GEO_DIM), np.float32)
    kk = keys[keep_rows]
    for t in sorted(set(kk[:, 0].tolist())):
        sg = SimGeometry('libero_90', tasks[t]['name'])
        for j in np.flatnonzero(kk[:, 0] == t):
            feats[j] = sg.features(ST[j])[0]
        sg.close()
    np.save(d / 'geo_raw.npy', feats)
    (d / 'DONE').write_text('ok')
    log('rate_matched_built', out, split=split, rows=int(len(keep_rows)), dropped=int(len(keys) - len(keep_rows)))


def geometry_devv2(out: Path = OUT) -> None:
    """PRIVILEGED geometry for VQ-A's 20 Hz Dev v2 rows (dev_v2.load_dev_v2_20 order).

    Same start-step rule as dev_v2.build_20hz: first non-noop HDF5 step >= 2*frame of demo
    number ``episode`` (demos sorted by index)."""
    import h5py
    import dev_v2
    p = out / 'geometry' / 'devv2_20.npz'
    if p.exists():
        return
    s = dev_v2.load_dev_v2_20()
    tasks = libero90_tasks()
    feats = np.zeros((len(s['key']), GEO_DIM), np.float32)
    for t in sorted(set(s['key'][:, 0].tolist())):
        sg = SimGeometry('libero_90', tasks[int(t)]['name'])
        with h5py.File(dev_v2.H5ROOT / f"{tasks[int(t)]['name']}_demo.hdf5", 'r') as fo:
            names = sorted((x for x in fo['data'] if x.startswith('demo_')), key=lambda x: int(x.split('_')[1]))
            for j in np.flatnonzero(s['key'][:, 0] == t):
                _, e, f = map(int, s['key'][j])
                g = fo['data'][names[e]]
                K = np.flatnonzero(noop_keep(g['actions'][()]))
                st = g['states'][int(K[int(np.searchsorted(K, 2 * f))])]
                feats[j] = sg.features(st)[0]
        sg.close()
    np.savez(p, feats=feats, key=s['key'])
    log('geometry_complete', out, split='devv2_20', rows=int(len(feats)))


def load_rate_matched(split: str, out: Path = OUT) -> dict[str, np.ndarray]:
    """Rows for reuse by other experiments (VQ-A/B): 20 Hz rebuild of an Eval split.

    split in {'unseen20', 'dev20', 'learned'}.  Returned arrays are aligned with
    ``key`` (task, episode, frame) of the ORIGINAL row; ``row`` is the index into the
    original key-motion-01/final/data/<split>__*.npy arrays, so features extracted per
    original key (images/VLM at the same start time) can be reused by indexing with it.
    The start time is the first non-noop HDF5 step >= 2*frame (i.e. the same instant as
    the old 10 Hz row up to skipped no-ops); history = the 10 previous non-noop 20 Hz steps,
    target = the next 10 non-noop 20 Hz steps (pi05 quantile normalization, as before).
    ``learned`` is returned unchanged (it is already 20 Hz LeRobot frames).
    ``geo_raw`` is the PRIVILEGED simulator geometry (diagnostic only).
    """
    if split == 'learned':
        d = {k: np.load(KF_DATA / f'learned__{k}.npy') for k in
             ('key', 'task', 'episode', 'hard', 'history', 'history_mask', 'state', 'actions')}
        d['row'] = np.arange(len(d['key']))
        return d
    base = split[:-2] if split.endswith('20') else split
    d = out / f'{base}20'
    if not (d / 'DONE').exists():
        build_rate_matched(base, out)
    return {p.stem: np.load(p) for p in sorted(d.glob('*.npy'))}


def compare_rates(out: Path = OUT, device: str = 'cuda:0') -> dict:
    """Old 10 Hz Eval rows vs 20 Hz rebuild: C1b, C1a, KEYGRID H+S prototype, hold-last, oracle."""
    import torch
    import key_shared_latent as k2
    import key_grid as kg
    kf, _ = _models()
    _, ctx = _context(out, device)
    groups = {'C1b': [f'H_x1_mk1b_cteff_tau0.01_s{i}' for i in (1, 2, 3)],
              'C1a': ['Afix_x1_fixed_e_h_cteff', 'Afix_x1_fixed_e_h_cteff_s2', 'Afix_x1_fixed_e_h_cteff_s3']}
    sel = json.loads((KF_DATA.parent / 'keygrid_selected.json').read_text())
    kg_ck = [sel[k]['checkpoint'] for k in sorted(sel)]
    res = {}
    for sname in ('learned', 'dev', 'dev20', 'unseen', 'unseen20'):
        if sname.endswith('20'):
            sp = load_rate_matched(sname, out)
            sp['task'] = sp['task'].astype(np.int64); sp['episode'] = sp['episode'].astype(np.int64)
            sp['errors'] = k2._arm6_candidate_mse(sp['actions'], ctx.windows)
            sp['hs'] = kf.hs_features(sp)
        else:
            sp = ctx.splits[sname]
        r = {'rows': int(len(sp['key'])), 'oracle': float(np.asarray(sp['errors']).min(1).mean())}
        hold = k2._hold_last_actions(sp)
        r['hold_last'] = float(k2.action_errors(hold, np.asarray(sp['actions']))['arm6'].mean())
        for g, names in groups.items():
            vals = []
            for n in names:
                cfg, qm, ks = kf.load_row_model(C1B_ROWS / f'{n}.pt', ctx)
                sc, _ = kf.score_split(qm, ks, cfg, ctx, sp)
                vals.append(float(np.asarray(sp['errors'])[np.arange(len(sc)), sc.argmax(1)].mean()))
            r[g] = vals
        vals = []
        for ck in kg_ck:
            m = kg.PairMLPRanker(88, shared=False).to(ctx.device)
            m.load_state_dict(torch.load(ck, map_location=ctx.device, weights_only=True))
            sc = kg._score_vectors(m, sp['hs'], ctx.bank_hs, device=str(ctx.device))
            vals.append(float(np.asarray(sp['errors'])[np.arange(len(sc)), sc.argmax(1)].mean()))
        r['HS_MLP_prototype'] = vals
        res[sname] = r
        log('compare_rates', out, split=sname, result=r)
    write_json(out / 'rate_comparison.json', res)
    return res


def eval_rate_matched(names: list[str], split: str = 'unseen', out: Path = OUT, device: str = 'cuda:0') -> dict:
    """Score trained rows (C1b KQ-final checkpoints and VQ-C geometry rows) on the 20 Hz rebuilt rows."""
    import torch
    import key_shared_latent as k2
    kf, GeoFiLMQuery = _models()
    _, ctx = _context(out, device)
    d = out / f'{split}20'
    sp = {k: np.load(d / f'{k}.npy') for k in ('key', 'task', 'episode', 'hard', 'history', 'history_mask', 'state', 'actions')}
    sp['task'] = sp['task'].astype(np.int64); sp['episode'] = sp['episode'].astype(np.int64)
    g = np.load(out / 'geometry' / 'eh.npz')['feats']
    sp['geo'] = ((np.load(d / 'geo_raw.npy') - g.mean(0)) / (g.std(0) + 1e-3)).astype(np.float32)
    sp['errors'] = k2._arm6_candidate_mse(sp['actions'], ctx.windows)
    sp['hs'] = kf.hs_features(sp)
    rows = np.load(d / 'row.npy')
    res = {}
    orig_make, orig_pack = kf.make_query, kf.pack
    for name in names:
        path = (C1B_ROWS / f'{name}.pt') if (C1B_ROWS / f'{name}.pt').exists() else (out / 'rows' / f'{name}.pt')
        cfg = torch.load(path, map_location='cpu', weights_only=False)['cfg']
        if cfg.get('geo_mode', 'none') != 'none':
            cols = FEATURE_SETS[cfg.get('feat', 'full')]
            kf.make_query = lambda c, _c=cols, _m=cfg.get('mode', 'film'): GeoFiLMQuery(len(_c), _m)

            def pack(split_, ids, c, dev, *, rng=None, augment=False, mhist=None, _cols=cols):
                o = orig_pack(split_, ids, c, dev, rng=rng, augment=augment, mhist=mhist)
                o['geo'] = torch.as_tensor(np.asarray(split_['geo'][np.asarray(ids, np.int64)][:, _cols], np.float32), device=dev)
                return o
            kf.pack = pack
        try:
            cfg, qm, ks = kf.load_row_model(path, ctx)
            sc, _ = kf.score_split(qm, ks, cfg, ctx, sp)
            per = sp['errors'][np.arange(len(sc)), sc.argmax(1)]
            res[name] = {'per_query_top1': per.tolist(), 'all': float(per.mean()), 'rows': rows.tolist(),
                         'oracle': float(sp['errors'].min(1).mean())}
            log('rate_matched_eval', out, name=name, split=split, top1=float(per.mean()))
        finally:
            kf.make_query, kf.pack = orig_make, orig_pack
    write_json(out / f'{split}20_eval.json', res)
    return res


def summarize(out: Path = OUT) -> dict:
    """Seed means, subsets and paired episode bootstrap vs C1b (the KQ-final rows, seed-averaged)."""
    import glob
    rows = {}
    for f in sorted(glob.glob(str(out / 'rows' / '*.json'))):
        r = json.loads(Path(f).read_text())
        if 'evaluation' not in r:
            continue
        base = re.sub(r'_s\d+$', '', r['name'])
        rows.setdefault(base, []).append(r)
    rows['C1b (KQ-final)'] = [json.loads((C1B_ROWS / f'H_x1_mk1b_cteff_tau0.01_s{i}.json').read_text()) for i in (1, 2, 3)]
    sub, ep = {}, {}
    for s in ('dev', 'learned', 'unseen'):
        sp = {k: np.load(KF_DATA / f'{s}__{k}.npy') for k in ('history', 'history_mask', 'actions', 'task', 'episode', 'hard')}
        sub[s] = subsets(sp)
        sub[s]['hard'] = sp['hard'].astype(bool)
        ep[s] = sp['task'].astype(np.int64) * 10000 + sp['episode'].astype(np.int64)
    ref = {s: np.mean([np.asarray(r['evaluation'][s]['per_query_top1']) for r in rows['C1b (KQ-final)']], 0)
           for s in ('learned', 'unseen')}
    res = {}
    for base, rs in rows.items():
        e = {'seeds': len(rs), 'dev_selected': [r['dev_top1_arm6_mse'] for r in rs], 'epochs': [r.get('selected_epoch') for r in rs]}
        for s in ('dev', 'learned', 'unseen'):
            per = np.stack([np.asarray(r['evaluation'][s]['per_query_top1']) for r in rs])
            blk = {}
            for k, m in sub[s].items():
                if m.any():
                    v = per[:, m].mean(1)
                    blk[k] = {'mean': float(v.mean()), 'sd': float(v.std(ddof=1)) if len(v) > 1 else None, 'n': int(m.sum())}
            if s in ref:
                pm = per.mean(0)
                blk['diff_vs_c1b'] = {k: paired_bootstrap(pm[m], ref[s][m], ep[s][m]) for k, m in sub[s].items() if m.any()}
            e[s] = blk
        res[base] = e
    write_json(out / 'results.json', {'rows': res, 'note': 'PRIVILEGED simulator geometry (upper-bound diagnostic)'})
    return res


def final_tables(out: Path = OUT, ref: str = 'C1b_v2', device: str = 'cuda:0') -> dict:
    """Seed-mean tables on learned (20 Hz) and unseen20, subsets, paired episode bootstrap vs ``ref``."""
    import glob
    names = sorted(Path(f).stem for f in glob.glob(str(out / 'rows' / '*.json')))
    names = [n for n in names if 'evaluation' in json.loads((out / 'rows' / f'{n}.json').read_text())]
    ev = json.loads((out / 'unseen20_eval.json').read_text()) if (out / 'unseen20_eval.json').exists() else {}
    todo = [n for n in names if n not in ev]
    if todo:
        ev.update(eval_rate_matched(todo, 'unseen', out, device))
        write_json(out / 'unseen20_eval.json', ev)
    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(re.sub(r'_s\d+$', '', n), []).append(n)
    sp20 = load_rate_matched('unseen20', out)
    spl = load_rate_matched('learned', out)
    sub = {'unseen20': subsets(sp20), 'learned': subsets(spl)}
    epi = {'unseen20': sp20['task'].astype(np.int64) * 10000 + sp20['episode'],
           'learned': spl['task'].astype(np.int64) * 10000 + spl['episode']}
    per = {}
    for g, ns in groups.items():
        per[g] = {'unseen20': np.stack([np.asarray(ev[n]['per_query_top1']) for n in ns]),
                  'learned': np.stack([np.asarray(json.loads((out / 'rows' / f'{n}.json').read_text())
                                                  ['evaluation']['learned']['per_query_top1']) for n in ns])}
    res = {}
    for g in groups:
        e = {'seeds': len(groups[g]), 'dev_selected': [json.loads((out / 'rows' / f'{n}.json').read_text())['dev_top1_arm6_mse'] for n in groups[g]]}
        for s in ('unseen20', 'learned'):
            blk = {}
            for k, m in sub[s].items():
                v = per[g][s][:, m].mean(1)
                blk[k] = {'mean': float(v.mean()), 'sd': float(v.std(ddof=1)) if len(v) > 1 else None, 'n': int(m.sum()),
                          'seeds': v.tolist()}
                if ref in per and g != ref:
                    blk[k]['diff_vs_ref'] = paired_bootstrap(per[g][s].mean(0)[m], per[ref][s].mean(0)[m], epi[s][m])
            e[s] = blk
        res[g] = e
    write_json(out / 'results.json', {'reference': ref, 'rows': res,
                                      'note': 'G_* rows use PRIVILEGED simulator geometry (upper-bound diagnostic only)'})
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('fetch40', 'align40', 'geometry', 'rows', 'ablate', 'summary', 'rate20', 'eval20', 'compare', 'geo_devv2', 'final'))
    ap.add_argument('--split', nargs='*', default=['eh', 'learned', 'unseen', 'dev'])
    ap.add_argument('--plan', type=Path)
    ap.add_argument('--names', nargs='*')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--dev-split', default='dev')
    a = ap.parse_args()
    if a.cmd == 'fetch40':
        fetch40()
    elif a.cmd == 'align40':
        align40()
    elif a.cmd == 'geometry':
        for s in a.split:
            geometry(s)
    elif a.cmd == 'rate20':
        for sp in a.split:
            build_rate_matched(sp)
    elif a.cmd == 'eval20':
        for sp in a.split:
            eval_rate_matched(a.names, sp, device=a.device)
    elif a.cmd == 'final':
        final_tables(device=a.device)
    elif a.cmd == 'geo_devv2':
        geometry_devv2()
    elif a.cmd == 'compare':
        compare_rates(device=a.device)
    elif a.cmd == 'summary':
        summarize()
    elif a.cmd == 'ablate':
        ablate(a.names, device=a.device)
    else:
        plan = json.loads(a.plan.read_text())
        if a.names:
            plan = {k: plan[k] for k in a.names}
        run_rows(plan, device=a.device, dev_split=a.dev_split)


if __name__ == '__main__':
    main()
