#!/usr/bin/env python3
"""Compress an action window and decode it back, with no denoising and no retrieval.

Every window is reconstructed from its own compressed form, so this measures only
what the representation loses. Fixed 10-step windows on two sets: Long (the policy
learned it, the fits below did not, except M1) and unseen LIBERO-90 tasks.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.fft import dct, idct

HERE = Path(__file__).resolve().parent
PI05 = HERE.parent / 'pi05_memory'
REPO_ROOT = Path(__file__).resolve().parents[5]
STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', REPO_ROOT / '.skillnet-state')).expanduser()
ROOT = STATE_DIR / 'outputs/memory-pilot'
PI05_ROOT = Path(os.environ.get('SKILLNET_PI05_ROOT', ROOT / 'pi05-memory40-20261104')).expanduser()
RUN = PI05_ROOT / 'run-01'
UNSEEN = Path(os.environ.get('UNSEEN_LIBERO_OUT', ROOT / 'unseen-libero' / 'oracle-01')).expanduser()
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
from unseen_paths import UNSEEN_ROWS  # corrected rows (image orientation fix)
CODECS = PI05_ROOT / 'compress-vs-raw-01' / 'dimension-extension-01' / 'codecs'
TOPK = PI05_ROOT / 'topk-long-01' / 'production' / 'predictions_and_errors.npz'
PCA_K = (4, 8, 12, 16, 24, 32, 48)
DCT_K = (1, 2, 3, 4, 5)
KEY_K = (2, 3, 4, 5, 6)
GROUPS = {'xyz': (0, 3), 'rot': (3, 6), 'grip': (6, 7)}


def dct_roundtrip(x, k):
    """Keep the k lowest temporal frequencies of each action dimension."""
    c = dct(np.asarray(x, np.float64), type=2, norm='ortho', axis=1)
    c[:, k:] = 0.
    return idct(c, type=2, norm='ortho', axis=1)


def keyframe_roundtrip(x, k):
    """Keep k equally spaced frames, including both ends, and interpolate linearly."""
    x = np.asarray(x, np.float64)
    t = np.arange(x.shape[1])
    keep = np.round(np.linspace(0, x.shape[1] - 1, k)).astype(int)
    out = np.empty_like(x)
    for row in range(len(x)):
        for dim in range(x.shape[2]):
            out[row, :, dim] = np.interp(t, keep, x[row, keep, dim])
    return out


def pca_basis(fit):
    x = np.asarray(fit, np.float64).reshape(len(fit), -1)
    mean = x.mean(0)
    _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
    return mean, vt


def pca_roundtrip(x, mean, vt, k):
    y = np.asarray(x, np.float64).reshape(len(x), -1)
    return (((y - mean) @ vt[:k].T @ vt[:k]) + mean).reshape(x.shape)


def summarize(pred, target, hard):
    p, t = np.asarray(pred)[:, :, :7], np.asarray(target)[:, :, :7]
    e = np.mean((p - t) ** 2, axis=(1, 2))
    u, v = p[:, :, :3].sum(1), t[:, :, :3].sum(1)
    n = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
    cos = np.divide((u * v).sum(1), n, out=np.ones(len(u)), where=n > 1e-8)
    out = {'all': float(e.mean()), 'hard': float(e[hard].mean()), 'median': float(np.median(e)),
           'p95': float(np.quantile(e, .95)), 'within_01': float((e <= .01).mean()),
           'within_04': float((e <= .04).mean()), 'xyz_cos': float(cos.mean()),
           'hard_xyz_cos': float(cos[hard].mean()),
           'grip_sign': float(np.mean(np.sign(p[:, :, 6]) == np.sign(t[:, :, 6])))}
    for name, (lo, hi) in GROUPS.items():
        out[f'mse_{name}'] = float(np.mean((p[:, :, lo:hi] - t[:, :, lo:hi]) ** 2))
    return out


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT / 'unseen-libero' / 'draft-reconstruction-01')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = {p.stem: np.load(p, mmap_mode='r', allow_pickle=False) for p in (RUN / 'rows').glob('*.npy')}
    pooled = np.load(RUN / 'pooled.npy', mmap_mode='r')
    task, val = np.asarray(rows['task']), np.asarray(rows['validation'], bool)
    z = np.load(TOPK, allow_pickle=False)
    long_ids, long_hard = z['query_rows'], z['hard_mask'].astype(bool)
    unseen_is_dev = np.load(UNSEEN_ROWS / 'is_dev.npy')
    un = {k: np.load(UNSEEN_ROWS / f'{k}.npy', mmap_mode='r') for k in ('actions', 'state', 'pooled')}
    raw = np.load(UNSEEN / 'drafts' / 'raw_none_eval.npz', allow_pickle=False)
    ev = np.flatnonzero(~unseen_is_dev)
    # The draft cache numbers unseen rows after the 26,581 LIBERO-40 rows.
    if not np.array_equal(raw['query_ids'], ev + len(task)):
        raise AssertionError('unseen eval rows differ from the draft cache')
    threshold = json.loads((UNSEEN / 'hard_threshold.json').read_text())['threshold']
    sets = {'long': {'ids': long_ids, 'hard': long_hard, 'actions': np.asarray(rows['actions'][long_ids], np.float32),
                     'state': np.asarray(rows['state'][long_ids], np.float32), 'feature': np.asarray(pooled[long_ids], np.float32),
                     # Long is left out so the fit has not seen it; the unseen fit may use all 40 tasks.
                     'fit': (task >= 10) & ~val},
            'unseen': {'ids': ev, 'hard': raw['pre_mse'] >= threshold, 'actions': np.asarray(un['actions'][ev], np.float32),
                       'state': np.asarray(un['state'][ev], np.float32), 'feature': np.asarray(un['pooled'][ev], np.float32),
                       'fit': ~val}}
    dim_ext = load_module(PI05 / 'compress_vs_raw' / 'dimension_extension.py', 'draft_recon_dim')
    import torch
    torch.set_num_threads(8)
    codecs = {f'M1 b{d}': dim_ext.load_codec(torch.load(CODECS / f'{name}.pt', map_location='cpu', weights_only=False), d)
              for name, d in (('b12m', 12), ('b24', 24), ('b48', 48), ('b96', 96))}
    result = {}
    for name, s in sets.items():
        a, hard = s['actions'], s['hard']
        target = a[:, :, :7].astype(np.float64)
        mean, vt = pca_basis(np.asarray(rows['actions'][np.flatnonzero(s['fit'])])[:, :, :7])
        out = {}
        for k in PCA_K:
            out[f'PCA {k}'] = {'values': k, **summarize(pca_roundtrip(target, mean, vt, k), target, hard)}
        for k in DCT_K:
            out[f'DCT {k}'] = {'values': 7 * k, **summarize(dct_roundtrip(target, k), target, hard)}
        for k in KEY_K:
            out[f'keyframe {k}'] = {'values': 7 * k, **summarize(keyframe_roundtrip(target, k), target, hard)}
        for label, codec in codecs.items():
            pred = np.empty_like(a)
            for st in range(0, len(a), 128):
                A, S, F = a[st:st + 128], s['state'][st:st + 128], s['feature'][st:st + 128]
                b, _ = codec.encode(A, S)
                _, r = codec.prior(F, S)
                pred[st:st + 128] = np.asarray(codec.decode(np.asarray(b), np.asarray(r), F, S), np.float32)
            out[label] = {'values': int(label.split('b')[-1]), **summarize(pred, a, hard)}
        result[name] = {'rows': len(a), 'hard_rows': int(hard.sum()), 'representations': out}
    path = args.output / 'results.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(path)


if __name__ == '__main__':
    main()
