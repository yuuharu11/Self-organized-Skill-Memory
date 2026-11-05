"""Shared row-ID-safe helpers for the raw versus b12 comparison."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import os as _os
_REPO = Path(__file__).resolve().parents[6]
_STATE_DIR = Path(_os.environ.get('SKILLNET_STATE_DIR', _REPO / '.skillnet-state'))
_PI05_ROOT = Path(_os.environ.get('SKILLNET_PI05_ROOT', _STATE_DIR / 'outputs/memory-pilot/pi05-memory40-20261104'))
ROOT = _PI05_ROOT
RUN = ROOT / 'run-01'
FAIR = ROOT / 'fair-expert-01'
MEMORY = ROOT / 'skill-memories-01/raw-rows-noLong'
OUT = ROOT / 'compress-vs-raw-01'
LONG_TASKS = np.arange(10, dtype=np.int32)
T0S = (.2, .4, .6)


def require_new_artifacts(paths):
    """Prevent an accidental rerun from replacing previously saved results."""
    existing = [Path(path) for path in paths if Path(path).exists()]
    if existing:
        raise FileExistsError(f'refusing to overwrite existing artifacts: {existing}')


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def fair_module():
    return load_module('compress_vs_raw_fair', HERE.parent / 'fair_expert/run.py')


def base_module():
    return load_module('compress_vs_raw_base', HERE.parent / 'run.py')


def decoded_candidate_drafts(codec, bases, features, states, query_rows, candidate_rows, batch=256):
    """Decode candidate base vectors using each query's prior residual/context.

    candidate_rows and query_rows are dataset row IDs. In particular, bases is
    indexed directly by candidate row ID, not by an offset into the memory pool.
    """
    q = np.asarray(query_rows, np.int64)
    c = np.asarray(candidate_rows, np.int64)
    if c.ndim == 1:
        c = c[:, None]
    if c.shape[0] != len(q):
        raise ValueError('candidate rows must align with query rows')
    if c.size and (c.min() < 0 or c.max() >= len(bases)):
        raise IndexError('candidate dataset row ID out of bounds')
    out = np.empty((len(q), c.shape[1], 10, 32), np.float32)
    for lo in range(0, len(q), batch):
        sl = slice(lo, min(lo + batch, len(q)))
        qq = q[sl]
        _, r = codec.prior(np.asarray(features[qq], np.float32), np.asarray(states[qq], np.float32))
        bb = np.asarray(bases[c[sl]], np.float32).reshape(-1, bases.shape[-1])
        rr = np.repeat(np.asarray(r, np.float32), c.shape[1], axis=0)
        ff = np.repeat(np.asarray(features[qq], np.float32), c.shape[1], axis=0)
        ss = np.repeat(np.asarray(states[qq], np.float32), c.shape[1], axis=0)
        decoded = codec.decode(bb, rr, ff, ss)
        out[sl] = np.asarray(decoded, np.float32).reshape(len(qq), c.shape[1], 10, 32)
    return out


def episode_ids(rows, query_rows):
    q = np.asarray(query_rows, np.int64)
    return np.asarray(rows['task'][q], np.int64) * 100000 + np.asarray(rows['episode'][q], np.int64)


def classify_diff(ci):
    lo, hi = ci['ci95']
    if hi < 0:
        return 'raw_better'
    if lo > 0:
        return 'compressed_better'
    return 'no_detected_difference'
