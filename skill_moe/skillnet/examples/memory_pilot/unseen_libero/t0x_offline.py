#!/usr/bin/env python3
"""Measure Expert denoising displacement as a candidate confidence signal (T2)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import kq2cl_common as C
import kq2cl_expert as E
import expert_ex3 as EX
from t0x_common import OUT, progress, write_json

NOISE_OFFSETS = {'learned': 0, 'unseen20': 50_000_000, 'dev2': 60_000_000}


def evaluate(device: str = 'cuda:0', splits=('dev2', 'unseen20', 'learned'), batch: int = 1,
             queries=('iii', 'i')) -> dict:
    """Report row-wise m at t_probe=.15/.30 and compare with the existing ê head.

    m is the Frobenius norm over the 10x6 arm coordinates, averaged over two independent
    deterministic noise draws. Truth is the cached arm6 draft MSE; no training is performed.
    """
    sel = C.read_json(C.OUT / 'expert/select.json')['selected']
    ex = EX.LiveExpert(); ex.load(Path(sel['checkpoint']))
    result = {'checkpoint': sel['checkpoint'], 'selection': 'KQ2-CL Dev selection K1:500', 'splits': {}}
    calibration = {}
    for split in splits:
        rows, _, tgt, _, _, _, drafts = E.split_inputs(split)
        obs = ex.base.obs_batch(rows, np.arange(len(tgt)))
        srec = {}
        for query in queries:
            draft = np.asarray(drafts[query], np.float32)
            truth = C.arm6(draft, tgt).astype(np.float64)
            qpred = np.load(C.OUT / 'heads' / f'pred_{query}.npz')[f'{split}__ehat'].astype(np.float64)
            if split == 'dev2': calibration[f'dev2_{query}_ehat'] = qpred.astype(np.float32)
            qrec = {'n': len(draft), 'truth_arm6_median': float(np.median(truth)),
                    'ehat_spearman': float(spearmanr(qpred, truth).statistic)}
            for t_probe in (.15, .30):
                m_draws = []
                for draw in (0, 1):
                    ids = np.arange(len(draft), dtype=np.int64) + NOISE_OFFSETS[split] + draw * 13_579_111
                    noise = ex.exp3._seed_noise(ids)
                    out = ex.bridge(obs, draft, noise, np.full(len(draft), t_probe, np.float32), batch=batch)
                    m = np.linalg.norm(out[..., :6] - draft[..., :6], axis=(-1, -2))
                    m_draws.append(m)
                mmean = np.mean(m_draws, axis=0)
                if split == 'dev2': calibration[f'dev2_{query}_m_t{t_probe:.2f}'] = mmean.astype(np.float32)
                qrec[f't{t_probe:.2f}'] = {
                    'm_spearman': float(spearmanr(mmean, truth).statistic),
                    'm_draw_spearman': [float(spearmanr(md, truth).statistic) for md in m_draws],
                    'm_median': float(np.median(mmean)), 'm_quantiles': np.quantile(mmean, [.05,.25,.5,.75,.95]).tolist(),
                    'noise_pair_correlation': float(spearmanr(m_draws[0], m_draws[1]).statistic),
                }
            srec[query] = qrec
        result['splits'][split] = srec
        progress('t2', 'denoising displacement split complete', split=split, rows=len(tgt))
    result['rule'] = 'm = Frobenius norm of (Expert bridge output - draft) over [10,6], averaged over two deterministic noise draws; Spearman vs cached true draft arm6 MSE.'
    dest = OUT / 't2' / 'displacement.json'
    write_json(dest, result)
    np.savez_compressed(OUT / 't2' / 'displacement-calibration.npz', **calibration)
    progress('t2', 'offline displacement comparison complete', output=str(dest))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--splits', default='dev2,unseen20,learned')
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--queries', default='iii,i')
    a = p.parse_args()
    r = evaluate(a.device, tuple(a.splits.split(',')), batch=a.batch, queries=tuple(a.queries.split(',')))
    print(json.dumps(r, indent=2))


if __name__ == '__main__':
    main()

