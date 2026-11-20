#!/usr/bin/env python3
"""KQ2-CL stage O: Expert observation rows + pi0.5 reference chunks for learned / unseen20 / dev2.

  learned   LIBERO-40 Eval rows (621) = pi05-memory40 run-01 rows (EX2 learned row_ids; same order as KQ2)
  unseen20  LIBERO-90 Eval rows at 20 Hz (619), start observation = HDF5 step K[searchsorted(K, 2f)]
  dev2      Dev v2 20 Hz (667), same rule (selection only)

For LIBERO-90 splits the raw (image, wrist, state, prompt) is read by vq_sources.ObsReader (images
flipped [::-1, ::-1], the corrected orientation) and passed through the pi0.5 LIBERO transforms (the
same chain as the closed-loop server). Saved: obs/<split>.npz {image_base, image_wrist, state(32),
prompt, prompt_mask}. pi0.5 reference: fixed-noise sample_actions (10 flow steps) -> pi05/<split>.npy.

Parity: the pi0.5 VLM tokens recomputed from these obs rows (vq_sources._vq_features) are compared
with the KQ2 tokens (vq-b-01/feats) on 48 rows per split -> obs/parity.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np

import kq2cl_common as C

NOISE_OFFSET = {'learned': 0, 'unseen20': 50_000_000, 'dev2': 60_000_000}
VQB = C.ROOT / 'vq-b-01/feats'
VQB_NAME = {'learned': 'learned', 'unseen20': 'unseen20', 'dev2': 'dev2_20'}


def transforms():
    from openpi import transforms as T
    from openpi.policies.libero_policy import LiberoInputs
    from openpi.models.model import ModelType
    from openpi.shared.normalize import load as load_norm
    from openpi.models.tokenizer import PaligemmaTokenizer
    norm = list((C.PI05 / 'assets').rglob('norm_stats.json'))
    assert len(norm) == 1
    return [LiberoInputs(ModelType.PI05), T.Normalize(load_norm(norm[0].parent), use_quantiles=True),
            T.ResizeImages(224, 224), T.TokenizePrompt(PaligemmaTokenizer(200), discrete_state_input=False),
            T.PadStatesAndActions(32)]


def apply_transforms(trs, raw: dict) -> dict:
    row = dict(raw)
    for tr in trs:
        row = tr(row)
    return row


def build_obs(split: str) -> Path:
    dest = C.OUT / 'obs' / f'{split}.npz'
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    key = np.load(C.KQ2 / 'data' / split / 'key.npy')
    if split == 'learned':
        with np.load(C.ROOT / 'expert-ex2-01/production-05/retrieval/learned_retrieval.npz') as z:
            if not np.array_equal(z['key'], key):
                raise AssertionError('learned order')
            ids = np.asarray(z['row_ids'], np.int64)
        rows = {p.stem: np.load(p, mmap_mode='r') for p in (C.RUN40 / 'rows').glob('*.npy')}
        chk = np.stack([rows['task'][ids], rows['episode'][ids], rows['frame'][ids]], 1)
        if not np.array_equal(chk, key):
            raise AssertionError('learned run40 rows')
        out = {k: np.asarray(rows[k][ids]) for k in ('image_base', 'image_wrist', 'state', 'prompt', 'prompt_mask')}
    else:
        import vq_sources as vs
        reader = vs.ObsReader(); trs = transforms()
        acc = {k: [] for k in ('image_base', 'image_wrist', 'state', 'prompt', 'prompt_mask')}
        raw_state = []
        for i, k in enumerate(key):
            raw = reader.read(k, 'libero90_20')
            raw_state.append(np.asarray(raw['observation/state'], np.float32))
            r = apply_transforms(trs, raw)
            acc['image_base'].append(r['image']['base_0_rgb']); acc['image_wrist'].append(r['image']['left_wrist_0_rgb'])
            acc['state'].append(r['state']); acc['prompt'].append(r['tokenized_prompt']); acc['prompt_mask'].append(r['tokenized_prompt_mask'])
        out = {k: np.stack(v) for k, v in acc.items()}
        out['raw_state'] = np.stack(raw_state)
        # the transformed state must equal the KQ2 normalized state of the row (same start step)
        st = np.load(C.KQ2 / 'data' / split / 'state.npy')
        out_check = float(np.abs(out['state'][:, :8] - st).max())
        C.log('obs_state_check', split=split, max_abs_diff_vs_kq2_state=out_check)
    out['key'] = key
    np.savez(dest, **out)
    C.log('obs_built', split=split, rows=int(len(key)))
    return dest


def load_obs(split: str) -> dict:
    with np.load(C.OUT / 'obs' / f'{split}.npz') as z:
        return {k: z[k] for k in z.files}


def pi05_and_parity(splits: list[str]) -> None:
    import jax
    import jax.numpy as jnp
    from openpi.shared.nnx_utils import module_jit
    import skill_expert_refine as exp3
    import vq_sources as vs
    base = exp3._module(C.HERE.parent / 'pi05_memory/run.py', 'kq2cl_obs_base')
    model, _ = base.load_policy(C.PI05)
    fn = module_jit(model.sample_actions, static_argnames=('num_steps',))
    feat = module_jit(types.MethodType(vs._vq_features, model))
    parity = C.read_json(C.OUT / 'obs/parity.json', {})
    for split in splits:
        rows = load_obs(split)
        dest = C.OUT / 'pi05' / f'{split}.npy'
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            n = len(rows['key']); ids = np.arange(n)
            noise = exp3._seed_noise(ids + NOISE_OFFSET[split])
            res = []
            for s in range(0, n, 16):
                b = ids[s:s + 16]
                res.append(np.asarray(fn(jax.random.key(0), base.obs_batch(rows, b), num_steps=10,
                                         noise=jnp.asarray(noise[b])), np.float32)[..., :7])
            np.save(dest, np.concatenate(res))
            tgt = np.load(C.KQ2 / 'data' / split / 'actions.npy')
            C.log('pi05_ref', split=split, arm6=float(C.arm6(np.load(dest), tgt).mean()))
        if split not in parity:
            src = VQB / VQB_NAME[split]
            if not np.array_equal(np.load(src / 'key.npy'), rows['key']):
                raise AssertionError('vq-b key order')
            ids = np.linspace(0, len(rows['key']) - 1, 48).astype(int)
            out = {k: np.asarray(v) for k, v in feat(base.obs_batch(rows, ids)).items()}
            ref_img = np.asarray(np.load(src / 'vlm_img.npy', mmap_mode='r')[ids], np.float32)
            ref_tm = np.asarray(np.load(src / 'vlm_txt_mean.npy', mmap_mode='r')[ids], np.float32)

            def cos(a, b):
                a = np.asarray(a, np.float32).reshape(len(ids), -1, a.shape[-1]); b = b.reshape(a.shape)
                return float(((a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-8)).min())
            parity[split] = {'rows': len(ids), 'vlm_img_min_cos': cos(out['vlm_img'], ref_img),
                             'vlm_img_max_abs': float(np.abs(np.asarray(out['vlm_img'], np.float32) - ref_img).max()),
                             'vlm_txt_mean_min_cos': cos(out['vlm_txt_mean'][:, None], ref_tm[:, None])}
            C.write_json(C.OUT / 'obs/parity.json', parity)
            C.log('obs_parity', split=split, **parity[split])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=('build', 'pi05'))
    ap.add_argument('--splits', default='learned,unseen20,dev2')
    a = ap.parse_args()
    sp = a.splits.split(',')
    C.log('start', stage='obs.' + a.cmd)
    if a.cmd == 'build':
        for s in sp:
            build_obs(s)
    else:
        pi05_and_parity(sp)
    C.log('end', stage='obs.' + a.cmd)


if __name__ == '__main__':
    main()
