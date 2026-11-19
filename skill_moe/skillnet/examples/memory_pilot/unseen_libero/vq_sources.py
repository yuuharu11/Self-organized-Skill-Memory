#!/usr/bin/env python3
"""VQ-B: take image and language separately from cleaner sources, inject into the
C1b history+state MLP query by FiLM.

Feature sources (one frozen π0.5 prefix pass per row; nothing trainable here):
  I1  pre-LLM image tokens (SigLIP + projection, i.e. ``PaliGemma.img``), base + wrist,
      16x16 patches average-pooled to a 4x4 grid per camera (32 tokens x 2048).
  I2  π0.5 VLM output at the image positions (the current source), per-token L2
      normalized as in ``core.prefix_features``, pooled the same way.
  L1a π0.5 VLM output at the prompt positions (mean / last valid token). NOTE: the
      π0.5 prefix is bidirectional, so these tokens also attend to the images.
  L1b input text embedding (``llm(..., method='embed')``) before the LLM: depends only on
      the prompt string (mean over valid tokens).
  L1c the same LLM run over the prompt alone (image masks off): language-only
      contextual embedding (mean / last valid token).
Images and prompts go through the exact π0.5 LIBERO transforms used by the LT1 / KEY2
extractors (LIBERO-90 HDF5 frames are flipped [::-1, ::-1] = corrected orientation).

Model: FiLM on the C1b query tower hidden layer (88 -> 512 -GELU-> 256):
    h = GELU(W1 x),  h' = (1 + γ) ⊙ h + β,  q = norm(W2 h')
with (γ, β) from a small head on the condition (L, I, or L+I), last layer zero-init so
the model starts exactly at C1b.  Training rows, teacher, loss, candidates, key and
schedule are those of C1b (key_final.train_row with rows='eh', same_scene=True, τ=.01).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[5]
STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', REPO_ROOT / '.skillnet-state')).expanduser()
sys.path.insert(0, str(HERE))

ROOT = (STATE_DIR / 'outputs/memory-pilot/unseen-libero')
DEV_V2 = ROOT / 'dev-v2-01'
OUT = ROOT / 'vq-b-01'
KF_DATA = ROOT / 'key-motion-01/final/data'
RUN_ROOT = (STATE_DIR / 'outputs/memory-pilot/pi05-memory40-20261104/run-01')
LIBERO90_DIR = (STATE_DIR / 'data/libero/datasets/libero_90')
POOL = 4  # 16x16 patches -> 4x4 grid
# 20 Hz rate-matched LIBERO-90 rows (VQ-C vq_geometry.build_rate_matched / VQ-A Dev v2 20 Hz).
# Start observation = HDF5 step K[p], K = no-op-filtered steps, p = searchsorted(K, 2*frame).
RATE20 = {'unseen20': ROOT / 'vq-c-01/unseen20', 'dev20': ROOT / 'vq-c-01/dev20', 'dev2_20': DEV_V2 / 'data20'}


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
# row sets
# ----------------------------------------------------------------------------

def split_keys(name: str) -> tuple[np.ndarray, str]:
    """(task, episode, frame) keys and domain for a row set."""
    if name in ('eh', 'learned', 'start'):
        return np.load(KF_DATA / f'{name}__key.npy').astype(np.int32), 'libero40'
    if name in ('unseen', 'dev'):
        return np.load(KF_DATA / f'{name}__key.npy').astype(np.int32), 'libero90'
    if name in RATE20:
        return np.load(RATE20[name] / 'key.npy').astype(np.int32), 'libero90_20'
    if name == 'dev2':
        import dev_v2  # built by VQ-A
        d = dev_v2.load_dev_v2()
        return np.asarray(d['key'], np.int32), 'libero90'
    raise ValueError(name)


# ----------------------------------------------------------------------------
# observation reading (no target actions read)
# ----------------------------------------------------------------------------

def start_step_20hz(actions: np.ndarray, frame: int) -> int:
    """HDF5 step of a 20 Hz row's start observation (same rule as vq_geometry.build_rate_matched)."""
    from vq_geometry import noop_keep
    K = np.flatnonzero(noop_keep(actions))
    return int(K[int(np.searchsorted(K, 2 * int(frame)))])


class ObsReader:
    def __init__(self, res128: bool = False):
        self.res128 = res128  # downscale LIBERO-40 (256px) frames to LIBERO-90's 128px before the transforms
        data = json.loads((RUN_ROOT / 'data_manifest.json').read_text())
        self.records40 = {(int(x['task']), int(x['episode'])): x for x in data['episodes']}
        self.tasks40 = {int(k): str(v) for k, v in data['tasks'].items()}
        un = json.loads((ROOT / 'unseen-rows-02/manifest.json').read_text())
        self.records90 = {(int(x['task_id']), int(x['episode'])): x for x in un['source_records']}
        self._tab = (None, None)
        self.extra90 = {}

    def libero90_record(self, t: int, e: int) -> dict:
        if (t, e) in self.records90:
            return self.records90[(t, e)]
        if (t, e) in self.extra90:
            return self.extra90[(t, e)]
        # Dev v2 rows (VQ-A): demo index e of the sorted demo_* list of the task's HDF5
        import h5py
        import dev_v2
        if not hasattr(self, '_table'):
            self._table = dev_v2.task_table()
        p = dev_v2.H5ROOT / f"{self._table[t]['name']}_demo.hdf5"
        with h5py.File(p, 'r') as fo:
            names = sorted((x for x in fo['data'] if x.startswith('demo_')), key=lambda x: int(x.split('_')[1]))
        rec = {'source_hdf5': str(p), 'source_demo': names[e], 'instruction': self._table[t]['instruction']}
        self.extra90[(t, e)] = rec
        return rec

    def read(self, key, domain: str) -> dict:
        import io
        from PIL import Image
        t, e, f = map(int, key)
        if domain == 'libero40':
            import pyarrow.parquet as pq
            path = self.records40[(t, e)]['path']
            if self._tab[0] != path:
                self._tab = (path, pq.read_table(path, columns=['image', 'wrist_image', 'state']))
            tab = self._tab[1]
            imgs = []
            for col in ('image', 'wrist_image'):
                with Image.open(io.BytesIO(tab[col][f].as_py()['bytes'])) as im:
                    im = im.convert('RGB')
                    if self.res128:
                        im = im.resize((128, 128), Image.BILINEAR, reducing_gap=None)
                    imgs.append(np.asarray(im))
            state = np.asarray(tab['state'][f].as_py(), np.float32)
            prompt = self.tasks40[t]
        else:
            import h5py
            rec = self.libero90_record(t, e)
            with h5py.File(rec['source_hdf5'], 'r') as fo:
                obs = fo['data'][rec['source_demo']]['obs']
                if domain == 'libero90_20':
                    idx = start_step_20hz(fo['data'][rec['source_demo']]['actions'][()], f)
                else:
                    idx = min(f * 2, len(obs['agentview_rgb']) - 1)
                imgs = [np.asarray(obs['agentview_rgb'][idx], np.uint8)[::-1, ::-1],
                        np.asarray(obs['eye_in_hand_rgb'][idx], np.uint8)[::-1, ::-1]]
                state = np.concatenate([np.asarray(obs['ee_states'][idx], np.float32),
                                        np.asarray(obs['gripper_states'][idx], np.float32)])
            prompt = str(rec.get('instruction', rec.get('task_name', '')))
        return {'observation/image': imgs[0], 'observation/wrist_image': imgs[1],
                'observation/state': state, 'prompt': prompt}


# ----------------------------------------------------------------------------
# frozen π0.5 feature pass
# ----------------------------------------------------------------------------

def _vq_features(self, observation):
    """Bound to the π0.5 module.  Returns pooled pre-LLM / VLM-output features."""
    import jax.numpy as jnp
    from openpi.models import model as model_api
    from openpi.models.pi0 import make_attn_mask
    obs = model_api.preprocess_observation(None, observation, train=False)
    tokens, valid, ar = self.embed_prefix(obs)
    (hidden, _), _ = self.PaliGemma.llm([tokens, None], mask=make_attn_mask(valid, ar),
                                       positions=jnp.cumsum(valid, axis=1) - 1)
    b = tokens.shape[0]
    hidden = hidden.astype(jnp.float32)
    tokens = tokens.astype(jnp.float32)
    hn = hidden / jnp.maximum(jnp.linalg.norm(hidden, axis=-1, keepdims=True), 1e-8)

    def grid(x):  # [b,256,D] -> [b,16,D] 4x4 mean pooling of the 16x16 patch grid
        x = x.reshape(b, 16 // POOL, POOL, 16 // POOL, POOL, x.shape[-1])
        return x.mean((2, 4)).reshape(b, (16 // POOL) ** 2, x.shape[-1])

    pre_img = jnp.concatenate([grid(tokens[:, :256]), grid(tokens[:, 256:512])], 1)
    vlm_img = jnp.concatenate([grid(hn[:, :256]), grid(hn[:, 256:512])], 1)
    tv = (observation.tokenized_prompt > 2) & valid[:, 768:]
    w = tv.astype(jnp.float32)[..., None]
    cnt = jnp.maximum(w.sum(1), 1.)
    last = jnp.maximum(jnp.sum(tv, 1) - 1, 0)
    txt_h, txt_e = hn[:, 768:], tokens[:, 768:]
    return {'pre_img': pre_img.astype(jnp.float16), 'vlm_img': vlm_img.astype(jnp.float16),
            'vlm_txt_mean': (txt_h * w).sum(1) / cnt,
            'vlm_txt_last': jnp.take_along_axis(txt_h, last[:, None, None], 1)[:, 0],
            'pre_txt_mean': (txt_e * w).sum(1) / cnt,
            # cached-token comparison (orientation / pipeline check): per-camera means of normalized tokens
            'vlm_img_mean': jnp.stack([hn[:, :256].mean(1), hn[:, 256:512].mean(1)], 1)}


class Pi05Features:
    def __init__(self):
        import importlib.util
        from openpi.transforms import Normalize, ResizeImages, TokenizePrompt, PadStatesAndActions
        from openpi.policies.libero_policy import LiberoInputs
        from openpi.models.model import ModelType
        from openpi.shared.normalize import load as load_norm
        from openpi.shared.nnx_utils import module_jit
        from openpi.models.tokenizer import PaligemmaTokenizer
        pi05_dir = HERE.parent / 'pi05_memory'
        sys.path.insert(0, str(pi05_dir))
        spec = importlib.util.spec_from_file_location('vq_pi05_run', pi05_dir / 'run.py')
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        self.mod = mod
        ckpt = Path(json.loads((RUN_ROOT / 'data_manifest.json').read_text())['checkpoint'])
        self.model, _ = mod.load_policy(ckpt)
        self.fn = module_jit(types.MethodType(_vq_features, self.model))
        norm = list((ckpt / 'assets').rglob('norm_stats.json'))
        assert len(norm) == 1
        self.transforms = [LiberoInputs(ModelType.PI05), Normalize(load_norm(norm[0].parent), use_quantiles=True),
                           ResizeImages(224, 224), TokenizePrompt(PaligemmaTokenizer(200), discrete_state_input=False),
                           PadStatesAndActions(32)]

    def __call__(self, raw_items: list[dict], *, text_only: bool = False) -> dict[str, np.ndarray]:
        import jax.numpy as jnp
        items = []
        for row in raw_items:
            for tr in self.transforms:
                row = tr(row)
            items.append(row)
        rows = {'image_base': np.stack([x['image']['base_0_rgb'] for x in items]),
                'image_wrist': np.stack([x['image']['left_wrist_0_rgb'] for x in items]),
                'state': np.stack([x['state'] for x in items]),
                'prompt': np.stack([x['tokenized_prompt'] for x in items]),
                'prompt_mask': np.stack([x['tokenized_prompt_mask'] for x in items])}
        obs = self.mod.obs_batch(rows, np.arange(len(items)))
        if text_only:
            import dataclasses
            obs = dataclasses.replace(obs, image_masks={k: jnp.zeros_like(v) for k, v in obs.image_masks.items()})
        out = self.fn(obs)
        return {k: np.asarray(v) for k, v in out.items()}


ROW_FEATS = {'pre_img': ((32, 2048), np.float16), 'vlm_img': ((32, 2048), np.float16),
             'vlm_txt_mean': ((2048,), np.float32), 'vlm_txt_last': ((2048,), np.float32),
             'pre_txt_mean': ((2048,), np.float32), 'vlm_img_mean': ((2, 2048), np.float32)}


def extract(splits: list[str], out: Path = OUT, batch: int = 32, limit: int | None = None,
            res128: bool = False) -> None:
    """Restartable per-split memmaps under out/feats/<split>/ (out/feats_r128/ with res128)."""
    fx = None
    reader = ObsReader(res128)
    for name in splits:
        keys, domain = split_keys(name)
        if limit:
            keys = keys[:limit]
        if res128 and domain != 'libero40':
            continue  # LIBERO-90 frames are already 128px
        d = out / ('feats_r128' if res128 else 'feats') / name
        d.mkdir(parents=True, exist_ok=True)
        if (d / 'DONE').exists():
            continue
        n = len(keys)
        mode = 'r+' if (d / 'done.npy').exists() else 'w+'
        if mode == 'r+' and not np.array_equal(np.load(d / 'key.npy'), keys):
            raise ValueError(f'{name}: cached keys differ')
        np.save(d / 'key.npy', keys)
        arr = {k: np.lib.format.open_memmap(d / f'{k}.npy', mode=mode, dtype=dt, shape=(n, *sh))
               for k, (sh, dt) in ROW_FEATS.items()}
        done = np.lib.format.open_memmap(d / 'done.npy', mode=mode, dtype=np.bool_, shape=(n,))
        if mode == 'w+':
            done[:] = False
        order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
        todo = order[~np.asarray(done)[order]]
        if fx is None and len(todo):
            fx = Pi05Features()
        prompts = {}
        t0 = time.time()
        for bi, s in enumerate(range(0, len(todo), batch)):
            ids = todo[s:s + batch]
            raws = [reader.read(keys[i], domain) for i in ids]
            for i, r in zip(ids, raws):
                prompts[(int(keys[i, 0]), domain)] = r['prompt']
            pad = batch - len(raws)
            res = fx(raws + [raws[-1]] * pad)
            for k in ROW_FEATS:
                arr[k][ids] = res[k][:len(ids)]
            done[ids] = True
            if bi % 20 == 0:
                for a in arr.values():
                    a.flush()
                done.flush()
                log('extract', out, split=name, completed=int(np.asarray(done).sum()), total=n,
                    seconds=round(time.time() - t0, 1))
        for a in arr.values():
            a.flush()
        done.flush()
        if not np.asarray(done).all():
            raise RuntimeError(f'{name} incomplete')
        # prompts per task (also needed for rows done in an earlier process)
        pmap = {}
        for t in np.unique(keys[:, 0]):
            if domain == 'libero40':
                pmap[int(t)] = reader.tasks40[int(t)]
            elif False:
                pass
            else:
                e = int(keys[keys[:, 0] == t][0, 1])
                rec = reader.libero90_record(int(t), e)
                pmap[int(t)] = str(rec.get('instruction', rec.get('task_name', '')))
        write_json(d / 'prompts.json', {'domain': domain, 'prompts': pmap})
        (d / 'DONE').write_text('ok\n')
        log('extract_done', out, split=name, rows=n)
    if res128:
        return
    # language-only features per unique prompt (L1b pre-LLM embedding, L1c LLM over prompt alone)
    allp = {}
    for name in splits:
        p = out / 'feats' / name / 'prompts.json'
        if p.exists():
            for t, s in json.loads(p.read_text())['prompts'].items():
                allp[s] = None
    lp = out / 'feats' / 'lang_prompts.json'
    have = json.loads(lp.read_text()) if lp.exists() else {}
    need = [s for s in allp if s not in have]
    if need:
        if fx is None:
            fx = Pi05Features()
        blank = np.zeros((224, 224, 3), np.uint8)
        for s in need:
            raw = {'observation/image': blank, 'observation/wrist_image': blank,
                   'observation/state': np.zeros(8, np.float32), 'prompt': s}
            r = fx([raw], text_only=True)
            have[s] = {'pre_txt_mean': r['pre_txt_mean'][0].tolist(), 'lang_only_mean': r['vlm_txt_mean'][0].tolist(),
                       'lang_only_last': r['vlm_txt_last'][0].tolist()}
        write_json(lp, have)
        log('lang_prompts_done', out, prompts=len(have))


# ----------------------------------------------------------------------------
# orientation / pipeline check against the cached KEY2 tokens
# ----------------------------------------------------------------------------

def check_against_cache(out: Path = OUT) -> dict:
    res = {}
    for name in ('learned', 'unseen', 'dev', 'dev2'):
        d = out / 'feats' / name
        if not (d / 'DONE').exists():
            continue
        src = (DEV_V2 / 'data' / 'tokens.npy', DEV_V2 / 'data' / 'groups.npy') if name == 'dev2' else \
            (KF_DATA / f'{name}__tokens.npy', KF_DATA / f'{name}__groups.npy')
        tok = np.load(src[0], mmap_mode='r')
        grp = np.load(src[1], mmap_mode='r')
        mine = np.load(d / 'vlm_img_mean.npy')
        txt = np.load(d / 'vlm_txt_mean.npy')
        n = min(len(mine), 64)
        cos_img, cos_txt, cos_flip = [], [], []
        for i in range(n):
            t = np.asarray(tok[i], np.float32); g = np.asarray(grp[i])
            for c in (0, 1):
                ref = t[g == c].mean(0)
                cos_img.append(float(ref @ mine[i, c] / np.linalg.norm(ref) / np.linalg.norm(mine[i, c])))
            ref = t[g == 2].mean(0)
            cos_txt.append(float(ref @ txt[i] / np.linalg.norm(ref) / np.linalg.norm(txt[i])))
        res[name] = {'rows_checked': n, 'cos_img_min': min(cos_img), 'cos_img_mean': float(np.mean(cos_img)),
                     'cos_txt_min': min(cos_txt), 'cos_txt_mean': float(np.mean(cos_txt))}
    write_json(out / 'cache_check.json', res)
    return res


# ----------------------------------------------------------------------------
# conditions, model, training (C1b recipe)
# ----------------------------------------------------------------------------

L_SOURCES = ('L1a', 'L1b', 'L1c')   # VLM-output prompt (image-contaminated) / pre-LLM embed / LLM over prompt alone
I_SOURCES = ('I1', 'I2')            # pre-LLM SigLIP tokens / VLM-output image tokens
C1B = dict(rows='eh', same_scene=True, tau=.01, epochs=16, batch=8, rand=256, lr=1e-4)


def load_conditions(name: str, keys: np.ndarray, out: Path = OUT) -> dict[str, np.ndarray]:
    """Row-aligned condition arrays for one split (keys must match the extracted ones)."""
    d = out / 'feats' / name
    if not (d / 'DONE').exists():
        raise FileNotFoundError(f'features for {name} not extracted')
    fk = np.load(d / 'key.npy')
    if not np.array_equal(fk, keys):
        raise ValueError(f'{name}: feature keys differ from split keys')
    prompts = json.loads((d / 'prompts.json').read_text())['prompts']
    lang = json.loads((out / 'feats' / 'lang_prompts.json').read_text())
    per_task = {int(t): lang[s] for t, s in prompts.items()}
    task = keys[:, 0]
    c = {'L1a': np.load(d / 'vlm_txt_mean.npy').astype(np.float32),
         'L1b': np.stack([np.asarray(per_task[int(t)]['pre_txt_mean'], np.float32) for t in task]),
         'L1c': np.stack([np.asarray(per_task[int(t)]['lang_only_mean'], np.float32) for t in task]),
         'I1': np.load(d / 'pre_img.npy', mmap_mode='r'),
         'I2': np.load(d / 'vlm_img.npy', mmap_mode='r')}
    # resolution-matched image sources: LIBERO-40 rows from 128px-downscaled frames (LIBERO-90 rows unchanged)
    dr = out / 'feats_r128' / name
    if (dr / 'DONE').exists():
        if not np.array_equal(np.load(dr / 'key.npy'), keys):
            raise ValueError(f'{name}: r128 keys differ')
        c['I1r'] = np.load(dr / 'pre_img.npy', mmap_mode='r')
        c['I2r'] = np.load(dr / 'vlm_img.npy', mmap_mode='r')
    elif name in ('unseen', 'dev', 'dev2') or name in RATE20 or name.startswith('dev2'):
        c['I1r'], c['I2r'] = c['I1'], c['I2']
    return c


def _torch():
    import torch
    from torch import nn
    from torch.nn import functional as F
    return torch, nn, F


def build_model(cond: dict[str, str | None], *, tok_dim: int = 32, img_queries: int = 2, lang_dim: int = 64,
                film_hidden: int = 128):
    """C1b tower + FiLM.  cond = {'L': source|None, 'I': source|None}."""
    torch, nn, F = _torch()
    import math

    class ImgEnc(nn.Module):
        """Per camera: LN -> Linear(2048->tok_dim) on the 16 grid tokens, learned-query attention pooling."""
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(2048)
            self.proj = nn.Linear(2048, tok_dim)
            self.cam = nn.Parameter(torch.zeros(2, 1, tok_dim))
            self.pos = nn.Parameter(torch.randn(1, 32, tok_dim) * .02)
            self.q = nn.Parameter(torch.randn(2, img_queries, tok_dim) * .02)
            self.out_dim = 2 * img_queries * tok_dim

        def forward(self, x):  # [b,32,2048]
            z = self.proj(self.ln(x.float())) + self.pos
            z = z.view(len(z), 2, 16, -1) + self.cam[None]
            att = torch.einsum('cqd,bctd->bcqt', self.q, z) / math.sqrt(z.shape[-1])
            pooled = torch.einsum('bcqt,bctd->bcqd', att.softmax(-1), z)
            return pooled.reshape(len(x), -1)

    class LangEnc(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, lang_dim), nn.GELU())
            self.out_dim = lang_dim

        def forward(self, x):
            return self.net(x.float())

    class FiLMQuery(nn.Module):
        def __init__(self):
            super().__init__()
            self.cond = dict(cond)
            self.l1 = nn.Linear(88, 512)
            self.l2 = nn.Linear(512, 256)
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / .07)))
            self.lenc = LangEnc() if cond.get('L') else None
            self.ienc = ImgEnc() if cond.get('I') else None
            cdim = (self.lenc.out_dim if self.lenc else 0) + (self.ienc.out_dim if self.ienc else 0)
            if cdim:
                self.film = nn.Sequential(nn.Linear(cdim, film_hidden), nn.GELU(), nn.Linear(film_hidden, 2 * 512))
                nn.init.zeros_(self.film[-1].weight); nn.init.zeros_(self.film[-1].bias)
            else:
                self.film = None

        def gamma_beta(self, lang=None, img=None):
            parts = []
            if self.lenc is not None:
                parts.append(self.lenc(lang))
            if self.ienc is not None:
                parts.append(self.ienc(img))
            gb = self.film(torch.cat(parts, -1))
            return gb[:, :512], gb[:, 512:]

        def forward(self, hs, lang=None, img=None, return_film=False):
            h = F.gelu(self.l1(hs.float()))
            g = b = None
            if self.film is not None:
                g, b = self.gamma_beta(lang, img)
                h = (1 + g) * h + b
            q = F.normalize(self.l2(h), dim=-1)
            return (q, g, b, h) if return_film else q

    return FiLMQuery()


def _hs(split):
    h = np.asarray(split['history'], np.float32).reshape(len(split['history']), -1)
    m = np.asarray(split['history_mask'], np.float32)
    return np.concatenate([h, m, np.asarray(split['state'], np.float32)], -1).astype(np.float32)


def subsets(split: dict) -> dict[str, np.ndarray]:
    """Rows where vision/language should matter (definitions fixed before any scoring).

    first_nohist : no history at all (frame 0 of the episode).
    early        : frame <= 20 (first ~2 s, typically still inside the first unit).
    post_switch  : a gripper command switch between the last 3 history steps and the first 3 target
                   steps, OR the xyz direction reverses (cos < 0 between the mean xyz of the last 3
                   history actions and of the first 3 target actions, both with norm > .05 in
                   normalized action units).  The target is used only to define the scoring subset.
    """
    k = np.asarray(split['key']); h = np.asarray(split['history'], np.float32)
    m = np.asarray(split['history_mask'], bool); a = np.asarray(split['actions'], np.float32)
    first = m.sum(1) == 0
    early = k[:, 2] <= 20
    g = np.concatenate([np.where(m[:, -3:], np.sign(h[:, -3:, 6]), 0), np.sign(a[:, :3, 6])], 1)
    gs = np.zeros(len(k), bool)
    for i in range(len(k)):
        v = g[i][g[i] != 0]
        gs[i] = len(v) > 1 and (np.diff(v) != 0).any()
    hv = np.where(m[:, -3:, None], h[:, -3:, :3], 0).sum(1) / np.maximum(m[:, -3:].sum(1, keepdims=True), 1)
    av = a[:, :3, :3].mean(1)
    nh, na = np.linalg.norm(hv, axis=1), np.linalg.norm(av, axis=1)
    cos = (hv * av).sum(1) / np.maximum(nh * na, 1e-8)
    rev = (nh > .05) & (na > .05) & (cos < 0) & m[:, -1]
    return {'first_nohist': first, 'early': early, 'post_switch': gs | rev, 'grip_switch': gs, 'dir_reverse': rev}


def load_rate20(d: Path) -> dict:
    s = {k: np.load(d / f'{k}.npy') for k in ('key', 'task', 'episode', 'hard', 'history', 'history_mask',
                                               'state', 'actions')}
    s['key'] = s['key'].astype(np.int32)
    for p in d.glob('sub_*.npy'):
        s[p.stem] = np.load(p)
    return s


def subsets_all(sp: dict, name: str) -> dict[str, np.ndarray]:
    """VQ-A's shared subset definitions (dev_v2.subset_masks) as the main ones, plus VQ-B's own."""
    import dev_v2
    if any(k.startswith('sub_') for k in sp):
        main = {k[4:]: np.asarray(sp[k], bool) for k in sp if k.startswith('sub_')}
    else:
        main = {k: np.asarray(v, bool) for k, v in dev_v2.eval_subsets(sp).items()}
    own = {'vqb_' + k: v for k, v in subsets(sp).items()}
    return {**main, **own}


class Ctx:
    """Splits from the KQ-final cache (C1b's exact rows) + VQ-B condition features."""
    def __init__(self, device, dev_name: str, out: Path = OUT):
        import torch
        import key_final as kf
        import key_shared_latent as k2
        self.kf, self.k2 = kf, k2
        self.device = device
        self.bank = kf.load_bank(kf.OUT)
        self.windows = np.asarray(self.bank['actions'], np.float32)
        self.bt = np.asarray(self.bank['task'], np.int64)
        self.be = np.asarray(self.bank['episode'], np.int64)
        self.keys = torch.as_tensor(np.load(KF_DATA / 'x1_keys.npy'), device=device, dtype=torch.float32)
        self.splits = {}
        names = ['eh', 'learned', 'unseen', 'dev']
        for n in names:
            self.splits[n] = kf.load_split(n, kf.OUT)
        if 'dev2' in dev_name.split(','):
            import dev_v2
            self.splits['dev2'] = dev_v2.load_dev_v2()
        for n20, d20 in RATE20.items():
            if (d20 / 'key.npy').exists() and (out / 'feats' / n20 / 'DONE').exists():
                self.splits[n20] = load_rate20(d20)
        self.dev_names = [d for d in dev_name.split(',')]
        for d in self.dev_names:
            if d not in self.splits:
                raise FileNotFoundError(d)
        for n, sp in self.splits.items():
            sp['task'] = np.asarray(sp['task'], np.int64); sp['episode'] = np.asarray(sp['episode'], np.int64)
            sp['hs_np'] = _hs(sp)
            sp['cond'] = load_conditions(n, np.asarray(sp['key'], np.int32), out)
            if n != 'eh':
                sp['errors'] = k2._arm6_candidate_mse(np.asarray(sp['actions']), self.windows)
                sp['subsets'] = subsets_all(sp, n)
                if 'hard' not in sp:
                    sp['hard'] = np.zeros(len(sp['key']), bool)
        self.splits['eh']['candidate_id'] = kf.candidate_ids_for(self.splits['eh'], self.bank)
        self._gpu = {}

    def tensors(self, n: str, cond: dict):
        """GPU copies of hs and the chosen condition arrays for split n (cached)."""
        import torch
        key = (n, cond.get('L'), cond.get('I'))
        if key not in self._gpu:
            sp = self.splits[n]
            t = {'hs': torch.as_tensor(sp['hs_np'], device=self.device)}
            if cond.get('L'):
                t['lang'] = torch.as_tensor(sp['cond'][cond['L']], device=self.device)
            if cond.get('I'):
                t['img'] = torch.as_tensor(np.asarray(sp['cond'][cond['I']]), device=self.device)
            self._gpu[key] = t
        return self._gpu[key]


def _query(model, t, ix):
    import torch
    i = torch.as_tensor(ix, device=t['hs'].device)
    return model(t['hs'][i], t['lang'][i] if 'lang' in t else None, t['img'][i] if 'img' in t else None)


def score_rows(model, ctx: Ctx, n: str, cond: dict, batch: int = 256, film: bool = False):
    import torch
    model.eval()
    sp = ctx.splits[n]
    t = ctx.tensors(n, cond)
    out, qs, gam, bet, hid = [], [], [], [], []
    with torch.no_grad():
        for s in range(0, len(sp['key']), batch):
            ix = np.arange(s, min(s + batch, len(sp['key'])))
            i = torch.as_tensor(ix, device=ctx.device)
            q, g, b, h = model(t['hs'][i], t['lang'][i] if 'lang' in t else None,
                               t['img'][i] if 'img' in t else None, return_film=True)
            out.append((q @ ctx.keys.T * model.logit_scale.exp().clamp(max=100.)).float().cpu().numpy())
            qs.append(q.cpu().numpy())
            if film and g is not None:
                gam.append(g.cpu().numpy()); bet.append(b.cpu().numpy()); hid.append(h.cpu().numpy())
    sc = np.concatenate(out)
    allowed = ~((sp['task'][:, None] == ctx.bt[None, :]) & (sp['episode'][:, None] == ctx.be[None, :]))
    sc[~allowed] = -1e9
    extra = {'gamma': np.concatenate(gam), 'beta': np.concatenate(bet)} if gam else {}
    return sc, np.concatenate(qs), extra


def eval_split(model, ctx: Ctx, n: str, cond: dict) -> dict:
    sc, _, _ = score_rows(model, ctx, n, cond)
    sp = ctx.splits[n]
    res = ctx.kf.metrics(sc, sp['errors'], sp, ctx.bt)
    per = np.asarray(res['per_query_top1'])
    res['subsets'] = {k: {'rows': int(v.sum()), 'top1_arm6_mse': float(per[v].mean()) if v.any() else None}
                      for k, v in sp['subsets'].items()}
    return res


def train(name: str, cond: dict, seed: int, ctx: Ctx, save_dir: Path, *, film_kw: dict | None = None,
          final_eval: bool = True) -> dict:
    """Exact C1b loop (key_final.train_row, rows='eh', same_scene, τ=.01) with the FiLM query."""
    torch, nn, F = _torch()
    k2 = ctx.k2
    save_dir.mkdir(parents=True, exist_ok=True)
    jp = save_dir / f'{name}.json'
    if jp.exists():
        return json.loads(jp.read_text())
    cfg = {**C1B, 'seed': seed, 'cond': cond, 'film_kw': film_kw or {}, 'dev': ctx.dev_names}
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(cond, **(film_kw or {})).to(ctx.device)
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=1e-4)
    tr = ctx.splits['eh']
    tt = ctx.tensors('eh', cond)
    qn = len(tr['key']); ids = np.arange(qn); nb = len(ctx.bt)
    dev_epochs = set(range(2, cfg['epochs'] + 1, 2))
    best = {d: float('inf') for d in ctx.dev_names}
    best_ep, best_state, hist = {}, {}, []
    t0 = time.time()
    for ep in range(1, cfg['epochs'] + 1):
        rng.shuffle(ids)
        model.train()
        losses = []
        for s in range(0, qn, cfg['batch']):
            qi = ids[s:s + cfg['batch']]
            rand = rng.choice(nb, size=min(cfg['rand'], nb), replace=False)
            own = tr['candidate_id'][qi]
            cids = np.unique(np.r_[own[own >= 0], rand])
            allowed = ~((tr['task'][qi, None] == ctx.bt[cids][None, :]) & (tr['episode'][qi, None] == ctx.be[cids][None, :]))
            q = _query(model, tt, qi)
            sc = (q @ ctx.keys[torch.as_tensor(cids, device=ctx.device)].T) * model.logit_scale.exp().clamp(max=100.)
            err = torch.as_tensor(k2._arm6_candidate_mse(np.asarray(tr['actions'][qi]), ctx.windows[cids]), device=ctx.device)
            am = torch.as_tensor(allowed, device=ctx.device)
            loss = k2.soft_listwise_loss(sc, err, am, float(cfg['tau']))
            same = (ctx.bt[cids][None, :] == tr['task'][qi, None]) & (ctx.be[cids][None, :] != tr['episode'][qi, None])
            sm = torch.as_tensor(same, device=ctx.device) & am
            negid = err.masked_fill(~sm, -torch.inf).argmax(-1)
            posid = err.masked_fill(~am, torch.inf).argmin(-1)
            has = sm.any(-1)
            if has.any():
                pos = sc.gather(1, posid[:, None]).squeeze(1); neg = sc.gather(1, negid[:, None]).squeeze(1)
                loss = loss + .1 * F.softplus(neg[has] - pos[has] + .1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.)
            opt.step()
            with torch.no_grad():
                model.logit_scale.clamp_(max=float(np.log(100.)))
            losses.append(float(loss.detach()))
        rec = {'epoch': ep, 'loss': float(np.mean(losses)), 'seconds': round(time.time() - t0, 1)}
        if ep in dev_epochs:
            for dn in ctx.dev_names:
                sc, _, _ = score_rows(model, ctx, dn, cond)
                dsp = ctx.splits[dn]
                v = float(dsp['errors'][np.arange(len(sc)), sc.argmax(1)].mean())
                rec[f'{dn}_top1_arm6_mse'] = v
                if v < best[dn]:
                    best[dn], best_ep[dn] = v, ep
                    best_state[dn] = {k: x.detach().cpu().clone() for k, x in model.state_dict().items()}
        hist.append(rec)
        log('epoch', save_dir.parent, name=name, **rec)
    torch.save({'cfg': cfg, 'state': best_state}, save_dir / f'{name}.pt')
    row = {'name': name, 'cfg': cfg, 'history': hist, 'by_dev': {}}
    # Selection rule fixed before scoring: every Dev in ctx.dev_names gets its own epoch choice and one
    # Eval scoring; the report uses dev2_20 when VQ-A's DEV_READY names the 20 Hz build, else dev20.
    done_states = {}
    for dn in ctx.dev_names:
        ep_ = best_ep[dn]
        if ep_ in done_states:
            row['by_dev'][dn] = {**done_states[ep_], 'selected_epoch': ep_, 'dev_top1_arm6_mse': best[dn]}
            continue
        model.load_state_dict(best_state[dn])
        ev = {}
        if final_eval:
            ev = {n: eval_split(model, ctx, n, cond) for n in ('learned', 'unseen20', 'unseen') if n in ctx.splits}
            ev['film'] = film_stats(model, ctx, cond)
        done_states[ep_] = {'evaluation': ev}
        row['by_dev'][dn] = {'evaluation': ev, 'selected_epoch': ep_, 'dev_top1_arm6_mse': best[dn]}
    write_json(jp, row)
    return row


def film_stats(model, ctx: Ctx, cond: dict) -> dict:
    """How far FiLM moves h away from identity, learned vs unseen."""
    if model.film is None:
        return {}
    out = {}
    for n in ('learned', 'unseen20'):
        _, _, ex = score_rows(model, ctx, n, cond, film=True)
        g, b = ex['gamma'], ex['beta']
        out[n] = {'gamma_abs_mean': float(np.abs(g).mean()), 'beta_abs_mean': float(np.abs(b).mean()),
                  'gamma_rowstd_mean': float(g.std(0).mean()), 'beta_rowstd_mean': float(b.std(0).mean())}
    return out


COND_NAMES = {
    'none': {}, 'L1a': {'L': 'L1a'}, 'L1b': {'L': 'L1b'}, 'L1c': {'L': 'L1c'},
    'I1': {'I': 'I1'}, 'I2': {'I': 'I2'},
    'L1b+I1': {'L': 'L1b', 'I': 'I1'}, 'L1c+I1': {'L': 'L1c', 'I': 'I1'}, 'L1a+I2': {'L': 'L1a', 'I': 'I2'},
    'L1c+I2': {'L': 'L1c', 'I': 'I2'},
    'I1r': {'I': 'I1r'}, 'I2r': {'I': 'I2r'}, 'L1c+I1r': {'L': 'L1c', 'I': 'I1r'}, 'L1b+I1r': {'L': 'L1b', 'I': 'I1r'},
}


def run_rows(rows: list[str], seeds: list[int], dev_name: str, out: Path, device: str, tag: str) -> None:
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
    ctx = Ctx(torch.device(device), dev_name, out)
    for r in rows:
        for sd in seeds:
            name = f'{r}_s{sd}'
            log('row_start', out, name=name, tag=tag, dev=dev_name)
            row = train(name, COND_NAMES[r], sd, ctx, out / tag)
            for dn, bd in row['by_dev'].items():
                ev = bd['evaluation']
                log('row_complete', out, name=name, tag=tag, sel_dev=dn, dev=bd['dev_top1_arm6_mse'],
                    epoch=bd['selected_epoch'],
                    learned=ev.get('learned', {}).get('all', {}).get('top1_arm6_mse'),
                    unseen20=ev.get('unseen20', {}).get('all', {}).get('top1_arm6_mse'),
                    unseen10_old=ev.get('unseen', {}).get('all', {}).get('top1_arm6_mse'))
            torch.cuda.empty_cache()


# ----------------------------------------------------------------------------
# why-analysis: what each source identifies
# ----------------------------------------------------------------------------

def source_vectors(n: str, out: Path = OUT) -> dict[str, np.ndarray]:
    """One vector per row and source (I*: mean of the 32 pooled tokens; L*: as stored)."""
    import key_final as kf
    sp = kf.load_split(n, kf.OUT)
    c = load_conditions(n, np.asarray(sp['key'], np.int32), out)
    v = {k: c[k] for k in L_SOURCES}
    for k in I_SOURCES:
        a = np.asarray(c[k], np.float32)
        v[k] = a.mean(1)
        v[k + '_base'] = a[:, :16].mean(1)
        v[k + '_wrist'] = a[:, 16:].mean(1)
    v['HS'] = _hs(sp)
    return v, sp


def _knn_task(x: np.ndarray, task: np.ndarray, episode: np.ndarray, k: int = 1) -> float:
    """Leave-one-episode-out k-NN (cosine, after per-feature standardization) task accuracy."""
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    s = x @ x.T
    same_ep = (task[:, None] == task[None]) & (episode[:, None] == episode[None])
    s[same_ep] = -np.inf
    nn_ = np.argsort(-s, 1)[:, :k]
    pred = np.array([np.bincount(task[r]).argmax() for r in nn_])
    return float((pred == task).mean())


def _domain_auc(xa: np.ndarray, xb: np.ndarray, seed: int = 0) -> float:
    """5-fold logistic-regression AUC separating learned (LIBERO-40) from unseen (LIBERO-90) rows."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.decomposition import PCA
    x = np.concatenate([xa, xb]); y = np.r_[np.zeros(len(xa)), np.ones(len(xb))]
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(x, y):
        pca = PCA(min(64, x.shape[1]), random_state=seed).fit(x[tr])
        m = LogisticRegression(C=.1, max_iter=2000).fit(pca.transform(x[tr]), y[tr])
        p[te] = m.predict_proba(pca.transform(x[te]))[:, 1]
    return float(roc_auc_score(y, p))


def analyze_sources(out: Path = OUT) -> dict:
    res = {'task_knn_1nn_leave_episode_out': {}, 'domain_auc_learned_vs_unseen': {}}
    vecs = {n: source_vectors(n, out) for n in ('learned', 'unseen', 'eh')}
    names = [k for k in vecs['learned'][0]]
    for n, (v, sp) in vecs.items():
        if n == 'eh':  # subsample for speed: every 3rd row
            ix = np.arange(0, len(sp['key']), 3)
        else:
            ix = np.arange(len(sp['key']))
        t = np.asarray(sp['task'], np.int64)[ix]; e = np.asarray(sp['episode'], np.int64)[ix]
        res['task_knn_1nn_leave_episode_out'][n] = {
            'chance': float(np.max(np.bincount(t)) / len(t)), 'tasks': int(len(np.unique(t))),
            **{k: _knn_task(np.asarray(v[k])[ix], t, e) for k in names}}
    for k in names:
        res['domain_auc_learned_vs_unseen'][k] = _domain_auc(np.asarray(vecs['learned'][0][k]),
                                                             np.asarray(vecs['unseen'][0][k]))
    write_json(out / 'source_analysis.json', res)
    return res


# ----------------------------------------------------------------------------
# aggregation
# ----------------------------------------------------------------------------

SUBS = ('first_unit', 'post_switch', 'dir_change', 'switch_or_dir')
PRIMARY_DEV = 'dev2_20'   # VQ-A Dev v2 at 20 Hz (DEV_READY rewritten); dev20 = old Dev at 20 Hz, secondary
EVAL = ('unseen20', 'learned', 'unseen')   # unseen = old 10 Hz rows (rate mismatch), kept as 旧 column


def _rows(out: Path, tag: str, dev: str = PRIMARY_DEV) -> dict[str, list[dict]]:
    """Per condition, the seeds' rows with the evaluation of the epoch chosen on `dev`."""
    by = {}
    for p in sorted((out / tag).glob('*.json')):
        r = json.loads(p.read_text())
        bd = r.get('by_dev', {}).get(dev)
        if not bd or not bd.get('evaluation'):
            continue
        cond = r['name'].rsplit('_s', 1)[0]
        by.setdefault(cond, []).append({'name': r['name'], **bd})
    return by


def _ms(v):
    v = np.asarray(v, float)
    return {'mean': float(np.nanmean(v)), 'sd': float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0., 'values': v.tolist()}


def _agg(rs: list[dict]) -> dict:
    a = {'seeds': len(rs), 'dev': _ms([r['dev_top1_arm6_mse'] for r in rs]), 'epochs': [r['selected_epoch'] for r in rs]}
    for sp in EVAL:
        if sp not in rs[0]['evaluation']:
            continue
        e0 = rs[0]['evaluation'][sp]
        a[sp] = {'all': _ms([r['evaluation'][sp]['all']['top1_arm6_mse'] for r in rs]),
                 'hard': _ms([r['evaluation'][sp]['hard']['top1_arm6_mse'] for r in rs]),
                 'other_task_top1': _ms([r['evaluation'][sp]['all']['other_task_top1_fraction'] for r in rs])}
        for k in e0['subsets']:
            a[sp][k] = _ms([r['evaluation'][sp]['subsets'][k]['top1_arm6_mse'] or np.nan for r in rs])
            a[sp][k]['rows'] = e0['subsets'][k]['rows']
    a['gap_unseen20_minus_learned'] = a['unseen20']['all']['mean'] - a['learned']['all']['mean']
    f = [r['evaluation'].get('film') for r in rs if r['evaluation'].get('film')]
    if f:
        a['film'] = {sp: {k: float(np.mean([x[sp][k] for x in f])) for k in f[0][sp]} for sp in f[0]}
    return a


def eval_refs() -> dict:
    """Per-row hold-last / oracle top-1 arm MSE and subset masks on each Eval split (20 Hz rows)."""
    import key_final as kf
    import key_shared_latent as k2
    import dev_v2
    bank = kf.load_bank(kf.OUT)
    win = np.asarray(bank['actions'], np.float32)
    bt, be = np.asarray(bank['task'], np.int64), np.asarray(bank['episode'], np.int64)
    out = {}
    for sp in EVAL:
        s = load_rate20(RATE20[sp]) if sp in RATE20 else kf.load_split(sp, kf.OUT)
        err = k2._arm6_candidate_mse(np.asarray(s['actions']), win)
        own = (np.asarray(s['task'])[:, None] == bt[None]) & (np.asarray(s['episode'])[:, None] == be[None])
        err = np.where(own, np.inf, err)
        hold = k2.action_errors(k2._hold_last_actions(s), np.asarray(s['actions']))['arm6']
        subs = {**{k: np.asarray(v, bool) for k, v in dev_v2.eval_subsets(s).items()},
                **{'vqb_' + k: v for k, v in subsets(s).items()}}
        out[sp] = {'task': np.asarray(s['task'], np.int64), 'episode': np.asarray(s['episode'], np.int64),
                   'hold': hold, 'oracle': err.min(1), 'subs': subs}
    return out


def _boot(diff, t, e):
    import key_shared_latent as k2
    return k2.episode_bootstrap_ci(diff, t, e) if len(diff) else None


def paired(rows: list[dict], refs: dict, base: list[dict] | None) -> dict:
    """Seed-averaged per-row difference vs `base` rows (or hold-last if base is None), episode bootstrap."""
    out = {}
    for sp in ('unseen20', 'learned'):
        a = np.mean([r['evaluation'][sp]['per_query_top1'] for r in rows], 0)
        b = refs[sp]['hold'] if base is None else np.mean([r['evaluation'][sp]['per_query_top1'] for r in base], 0)
        t, e = refs[sp]['task'], refs[sp]['episode']
        out[sp] = {'all': _boot(a - b, t, e)}
        for k in ('first_unit', 'post_switch', 'switch_or_dir'):
            m = refs[sp]['subs'][k]
            out[sp][k] = _boot((a - b)[m], t[m], e[m]) if m.any() else None
    return out


def report(out: Path = OUT) -> dict:
    refs = eval_refs()
    ref_tab = {sp: {'hold_last': {'all': float(r['hold'].mean()), **{k: float(r['hold'][m].mean()) if m.any() else None
                                                                      for k, m in r['subs'].items()}},
                    'oracle': {'all': float(r['oracle'].mean()), **{k: float(r['oracle'][m].mean()) if m.any() else None
                                                                    for k, m in r['subs'].items()}},
                    'rows': {'all': int(len(r['hold'])), **{k: int(m.sum()) for k, m in r['subs'].items()}}}
               for sp, r in refs.items()}
    res = {'primary_dev': PRIMARY_DEV, 'references': ref_tab}
    for tag in ('sweep20', 'final20'):
        rows = _rows(out, tag)
        if not rows:
            continue
        res[tag] = {c: _agg(rs) for c, rs in rows.items()}
        res[tag + '_secondary_dev20'] = {c: _agg(rs) for c, rs in _rows(out, tag, 'dev20').items()}
        if 'none' in rows:
            res[tag + '_paired_vs_none'] = {c: paired(rs, refs, rows['none']) for c, rs in rows.items() if c != 'none'}
        res[tag + '_paired_vs_hold'] = {c: paired(rs, refs, None) for c, rs in rows.items()}
    for f in ('source_analysis.json', 'cache_check.json', 'resolution_check.json'):
        if (out / f).exists():
            res[f[:-5]] = json.loads((out / f).read_text())
    write_json(out / 'results.json', res)
    (out / 'tables.md').write_text(render_tables(res))
    return res


def render_tables(res: dict) -> str:
    def f(x):
        return '—' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.4f}'

    def pm(d):
        return f(d['mean']) + (f' ± {d["sd"]:.4f}' if d.get('sd') else '')

    def ci(x):
        return '—' if x is None else f"{x['mean']:+.4f} [{x['ci95_low']:+.4f}, {x['ci95_high']:+.4f}]"
    R = res['references']
    L = [f"Dev 選択：{res['primary_dev']}（Dev v2 20Hz）。主表＝未学習 20Hz（unseen20）。旧＝10Hz ずれありの未学習行。\n"]
    L.append('|基準|未学習20Hz 全体|最初の単位|切替後|向き変化|切替or向き|学習済み 全体|学習済み 最初の単位|旧（10Hz ずれあり）|')
    L.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for nm in ('hold_last', 'oracle'):
        u, l, o = R['unseen20'][nm], R['learned'][nm], R['unseen'][nm]
        L.append(f"|{nm}|{f(u['all'])}|{f(u['first_unit'])}|{f(u['post_switch'])}|{f(u['dir_change'])}|"
                 f"{f(u['switch_or_dir'])}|{f(l['all'])}|{f(l['first_unit'])}|{f(o['all'])}|")
    ru, rl = R['unseen20']['rows'], R['learned']['rows']
    L.append(f"\n行数：未学習20Hz 全体 {ru['all']}・最初の単位 {ru['first_unit']}・切替後 {ru['post_switch']}・"
             f"向き変化 {ru['dir_change']}・切替or向き {ru['switch_or_dir']}；学習済み 全体 {rl['all']}・最初の単位 {rl['first_unit']}"
             f"・切替後 {rl['post_switch']}\n")
    for tag in ('final20', 'sweep20'):
        if not res.get(tag):
            continue
        L.append(f'### {tag}\n')
        L.append('|条件|seed|Dev v2 20Hz|未学習20Hz 全体 / 難|最初の単位|切替後|向き変化|学習済み 全体|学習済み 最初の単位|差(未20−学)|旧（10Hz ずれあり）未学習|他タスク top1 未20/学|epoch|')
        L.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
        order = sorted(res[tag], key=lambda c: res[tag][c]['dev']['mean'])
        for c in order:
            a = res[tag][c]; u, l = a['unseen20'], a['learned']
            old = a.get('unseen', {}).get('all')
            L.append(f"|{c}|{a['seeds']}|{f(a['dev']['mean'])}|{pm(u['all'])} / {f(u['hard']['mean'])}|"
                     f"{f(u['first_unit']['mean'])}|{f(u['post_switch']['mean'])}|{f(u['dir_change']['mean'])}|"
                     f"{pm(l['all'])}|{f(l['first_unit']['mean'])}|{f(a['gap_unseen20_minus_learned'])}|"
                     f"{f(old['mean']) if old else '—'}|{u['other_task_top1']['mean']:.2f} / {l['other_task_top1']['mean']:.2f}|"
                     f"{a['epochs']}|")
        L.append('')
        sec = res.get(tag + '_secondary_dev20', {})
        if sec:
            L.append('副（旧 Dev 20Hz で選んだ場合の未学習20Hz）：' + '、'.join(
                f"{c} {sec[c]['unseen20']['all']['mean']:.4f}" for c in order if c in sec) + '\n')
        film = [(c, res[tag][c]['film']) for c in order if 'film' in res[tag][c]]
        if film:
            L.append('|条件|学習済み mean abs γ / β|未学習20Hz mean abs γ / β|行間 sd γ 学/未|')
            L.append('|---|---:|---:|---:|')
            for c, fm in film:
                L.append(f"|{c}|{fm['learned']['gamma_abs_mean']:.3f} / {fm['learned']['beta_abs_mean']:.3f}|"
                         f"{fm['unseen20']['gamma_abs_mean']:.3f} / {fm['unseen20']['beta_abs_mean']:.3f}|"
                         f"{fm['learned']['gamma_rowstd_mean']:.3f} / {fm['unseen20']['gamma_rowstd_mean']:.3f}|")
            L.append('')
        for key, lab in ((tag + '_paired_vs_none', '行 − none（C1b 同条件）'), (tag + '_paired_vs_hold', '行 − 直前保持')):
            if res.get(key):
                L.append(f'#### {lab}（seed 平均の行ごとの差、episode bootstrap 95%）\n')
                L.append('|条件|未学習20Hz 全体|最初の単位|切替後|切替or向き|学習済み 全体|')
                L.append('|---|---:|---:|---:|---:|---:|')
                for c, d in res[key].items():
                    L.append(f"|{c}|{ci(d['unseen20']['all'])}|{ci(d['unseen20']['first_unit'])}|"
                             f"{ci(d['unseen20']['post_switch'])}|{ci(d['unseen20']['switch_or_dir'])}|{ci(d['learned']['all'])}|")
                L.append('')
    return '\n'.join(L) + '\n'


def ablate(names: list[str], tag: str = 'sweep20', dev: str = PRIMARY_DEV, out: Path = OUT,
           device: str = 'cuda:0', seed: int = 9001) -> dict:
    """Diagnostic: score the Dev-chosen checkpoint with its condition replaced at test time.

    intact / train_mean (FiLM input = mean over the eh training rows, i.e. a neutral condition) /
    shuffled (condition of another row of the same split) / lang_nearest_train (L only: the
    LIBERO-40 instruction embedding nearest by cosine).  Eval actions only score.
    """
    import torch
    ctx = Ctx(torch.device(device), dev, out)
    res = {}
    eh = ctx.splits['eh']['cond']
    for name in names:
        blob = torch.load(out / tag / f'{name}.pt', map_location='cpu', weights_only=False)
        cond = blob['cfg']['cond']
        model = build_model(cond, **blob['cfg'].get('film_kw', {})).to(ctx.device)
        model.load_state_dict(blob['state'][dev])
        r = {}
        for sp in (dev, 'unseen20', 'learned'):
            base = ctx.splits[sp]
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(base['key']))
            variants = {'intact': {}}
            tm, sh = {}, {}
            for k in ('L', 'I'):
                if cond.get(k):
                    src = cond[k]
                    full = np.asarray(eh[src], np.float32)
                    tm[src] = np.broadcast_to(full.mean(0), (len(base['key']),) + full.shape[1:]).copy()
                    sh[src] = np.asarray(base['cond'][src])[perm]
            variants['train_mean'] = tm
            variants['shuffled'] = sh
            if cond.get('L'):
                src = cond['L']
                tr = np.asarray(eh[src], np.float32)
                uniq = np.unique(tr, axis=0)
                x = np.asarray(base['cond'][src], np.float32)
                cs = (x / np.linalg.norm(x, axis=1, keepdims=True)) @ (uniq / np.linalg.norm(uniq, axis=1, keepdims=True)).T
                variants['lang_nearest_train'] = {src: uniq[cs.argmax(1)]}
                r[sp + '_lang_nearest_cos'] = float(cs.max(1).mean())
            for vn, rep in variants.items():
                saved = dict(base['cond'])
                base['cond'] = {**saved, **rep}
                ctx._gpu = {}
                ev = eval_split(model, ctx, sp, cond)
                base['cond'] = saved
                ctx._gpu = {}
                r[f'{sp}/{vn}'] = {'all': ev['all']['top1_arm6_mse'],
                                    **{k: v['top1_arm6_mse'] for k, v in ev['subsets'].items() if k in SUBS}}
        res[name] = r
        log('ablate', out, name=name)
    write_json(out / f'ablation_{tag}.json', res)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('extract', 'check', 'train', 'analyze', 'report', 'ablate'))
    ap.add_argument('--rows', nargs='*', default=list(COND_NAMES))
    ap.add_argument('--seeds', nargs='*', type=int, default=[4101])
    ap.add_argument('--dev', default='dev2_20,dev20')
    ap.add_argument('--tag', default='rows')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--splits', nargs='*', default=['learned', 'unseen', 'dev', 'eh'])
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--res128', action='store_true')
    a = ap.parse_args()
    if a.cmd == 'extract':
        extract(a.splits, a.out, a.batch, a.limit, a.res128)
        print(json.dumps(check_against_cache(a.out), indent=1))
    elif a.cmd == 'train':
        run_rows(a.rows, a.seeds, a.dev, a.out, a.device, a.tag)
    elif a.cmd == 'analyze':
        print(json.dumps(analyze_sources(a.out), indent=1))
    elif a.cmd == 'ablate':
        print(json.dumps(ablate(a.rows, a.tag, out=a.out, device=a.device), indent=1))
    elif a.cmd == 'report':
        report(a.out)
        print((a.out / 'tables.md').read_text())
    elif a.cmd == 'check':
        print(json.dumps(check_against_cache(a.out), indent=1))


if __name__ == '__main__':
    main()
