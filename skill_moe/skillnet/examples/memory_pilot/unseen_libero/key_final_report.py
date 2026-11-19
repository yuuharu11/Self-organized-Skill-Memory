#!/usr/bin/env python3
"""KQ-final aggregation: tables, Dev/unseen rank agreement, paired bootstrap, diagnostics.

Reads key-motion-01/final/{rows,keygrid}/*.json written by key_final.py and the
locked MK1c / CT1 / KEYGRID / KEY2 result files.  Selection uses Dev only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import os
REPO_ROOT = Path(__file__).resolve().parents[5]
STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', REPO_ROOT / '.skillnet-state')).expanduser()
sys.path.insert(0, str(HERE))
import key_shared_latent as k2  # noqa: E402

ROOT = (STATE_DIR / 'outputs/memory-pilot/unseen-libero')
OUT = ROOT / 'key-motion-01/final'
MK1C = ROOT / 'key-motion-01/query/results.json'
CT1 = ROOT / 'key-contrastive-01/production/results.json'
KEY2_SEP = ROOT / 'key-shared-01/attempt-27-production/final/separate'


def load_rows() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted((OUT / 'rows').glob('*.json'))}


def load_keygrid() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted((OUT / 'keygrid').glob('*.json'))}


def ev(row, split, sub='all'):
    e = row['evaluation'][split]
    return e[sub]['top1_arm6_mse'] if e.get(sub) else float('nan')


def per_query(row, split):
    return np.asarray(row['evaluation'][split]['per_query_top1'], np.float64)


def keygrid_selected(kg: dict[str, dict], seed: int) -> dict:
    """KEYGRID rule: min Dev over lr x tau x epoch grid (ties broken as in key_grid)."""
    cands = []
    for n, r in kg.items():
        if r['seed'] != seed:
            continue
        for g in r['grid']:
            cands.append((g['dev_top1_arm6_mse'], r['lr'], g['epoch'], r['tau'], g['checkpoint']))
    return min(cands)


def key2_separate_per_query(split: str, keys: np.ndarray) -> np.ndarray:
    z = np.load(KEY2_SEP / f'separate_{split}_paired_predictions.npz')
    err = ((z['top1_actions'][..., :6] - z['target_actions'][..., :6]) ** 2).sum((1, 2)) / 70.
    m = {tuple(map(int, k)): i for i, k in enumerate(z['key'])}
    return err[[m[tuple(map(int, k))] for k in keys]]


def boot(diff, split):
    return k2.episode_bootstrap_ci(diff, split['task'], split['episode'], reps=5000, seed=7301)


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def fmt(x, d=4):
    return '—' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.{d}f}'


def keygrid_eval_all(ctx, kg):
    """Evaluate the Dev-selected KEYGRID checkpoint of each seed (cached)."""
    import key_final as kf
    path = OUT / 'keygrid_selected.json'
    cache = json.loads(path.read_text()) if path.exists() else {}
    for seed in (1, 2, 3):
        if str(seed) in cache:
            continue
        d, lr, ep, tau, ck = keygrid_selected(kg, seed)
        cache[str(seed)] = {'dev_selected': d, 'lr': lr, 'epoch': ep, 'tau': tau, 'checkpoint': ck,
                            'evaluation': kf.eval_keygrid_checkpoint(ck, ctx)}
    kf.write_json(path, cache)
    return cache


GROUPS = {
    # name: (label, [row names per seed])
    'final_ct_x1': ('K-x1(MK1b) 固定 × E-H 注意クエリ（CT有効要素版）', ['Afix_x1_fixed_e_h_cteff', 'Afix_x1_fixed_e_h_cteff_s2', 'Afix_x1_fixed_e_h_cteff_s3']),
    'hs_x1_cteff': ('K-x1(MK1b) 固定 × 履歴＋状態MLP（CT有効要素版）', [f'H_x1_mk1b_cteff_tau0.01_s{i}' for i in (1, 2, 3)]),
    'hs_x1_simple': ('K-x1(MK1b) 固定 × 履歴＋状態MLP（簡易）', [f'H_x1_mk1b_simple_tau0.01_s{i}' for i in (1, 2, 3)]),
    'hs_dct_cteff': ('K-dct × 履歴＋状態MLP（CT有効要素版）', [f'H_dct_cteff_tau0.005_s{i}' for i in (1, 2, 3)]),
    'hs_dct_simple': ('K-dct × 履歴＋状態MLP（簡易）', [f'H_dct_simple_tau0.002_s{i}' for i in (1, 2, 3)]),
    'hs_kg_cteff': ('KEYGRID鍵 × 履歴＋状態MLP（CT有効要素版、同一学習器）', [f'H_keygrid_cteff_tau0.01_s{i}' for i in (1, 2, 3)]),
    'hs_kg_simple': ('KEYGRID鍵 × 履歴＋状態MLP（簡易、同一学習器）', [f'H_keygrid_simple_tau0.01_s{i}' for i in (1, 2, 3)]),
    'ct1_simple': ('CT1 簡易 = K-x1(MK1b) × E-H 注意クエリ（再現）', ['A0_ct_repro', 'A0_seed4102', 'A0_seed4103']),
    'mqe_dct': ('K-dct × E-H 注意クエリ（簡易）', ['Afix_dct_e_h_s1', 'Afix_dct_e_h_s2', 'Afix_dct_e_h_s3']),
    'mk1c_orig': ('MK1c x1_fixed_e_h 再現（観測拡張あり）', ['B0_mk1c_repro', 'B0_seed702', 'B0_seed703']),
    'mk1c_fixed': ('MK1c x1_fixed_e_h 拡張だけ外す', ['B_aug=False', 'B_aug=False_seed702', 'B_aug=False_seed703']),
}


def main():
    import torch
    import key_final as kf
    ctx = kf.RowContext(OUT, torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), False)
    kg = keygrid_eval_all(ctx, load_keygrid())
    rows = load_rows()
    res = {'groups': {}, 'keygrid_original': kg}
    # --- per-group seed summaries
    def summarize(evals, devs):
        out = {'dev_mean': float(np.mean(devs)), 'dev_seeds': [float(x) for x in devs]}
        for sp in ('unseen', 'learned'):
            for sub in ('all', 'hard'):
                v = [e[sp][sub]['top1_arm6_mse'] for e in evals]
                out[f'{sp}_{sub}_mean'] = float(np.mean(v)); out[f'{sp}_{sub}_sd'] = float(np.std(v, ddof=1)); out[f'{sp}_{sub}_seeds'] = v
        u = evals
        out['diag'] = {sp: {k: float(np.mean([e[sp]['all'][k] for e in u])) for k in ('spearman', 'oracle_rank_median', 'other_task_top1_fraction')}
                       for sp in ('unseen', 'learned')}
        for sp in ('unseen', 'learned'):
            out['diag'][sp]['recall_at'] = {k: float(np.mean([e[sp]['all']['recall_at'][k] for e in u])) for k in ('1', '5', '20', '100')}
        out['gap_unseen_minus_learned'] = out['unseen_all_mean'] - out['learned_all_mean']
        return out
    perq = {}
    for g, (label, names) in GROUPS.items():
        have = [n for n in names if n in rows]
        if not have:
            continue
        evals = [rows[n]['evaluation'] for n in have]
        res['groups'][g] = {'label': label, 'rows': have, **summarize(evals, [rows[n]['dev_top1_arm6_mse'] for n in have])}
        perq[g] = {sp: np.mean([per_query(rows[n], sp) for n in have], 0) for sp in ('learned', 'unseen')}
    kgev = [kg[s]['evaluation'] for s in ('1', '2', '3')]
    res['groups']['keygrid_orig'] = {'label': '履歴＋状態MLP（KEYGRID 原型、Dev選択、3 seed）', 'rows': ['keygrid seed1-3'],
                                     **summarize(kgev, [kg[s]['dev_selected'] for s in ('1', '2', '3')])}
    perq['keygrid_orig'] = {sp: np.mean([per_query(e, sp) if False else np.asarray(e[sp]['per_query_top1']) for e in kgev], 0) for sp in ('learned', 'unseen')}
    # --- final choice: finalists = best seed-1 Dev in the motion-key pool and runner-up; final by 3-seed mean Dev
    finalists = ['final_ct_x1', 'hs_x1_cteff']
    best = min(finalists, key=lambda g: res['groups'][g]['dev_mean'])
    res['final_choice'] = {'finalists': finalists, 'rule': 'seed-1 Dev で上位2つを候補にし、3 seed 平均 Dev で決定（Eval 不使用）', 'best': best}
    # --- paired bootstrap on seed-averaged per-query errors
    splits = {sp: ctx.splits[sp] for sp in ('learned', 'unseen')}
    pairs = [(best, 'keygrid_orig', '最良行 − 履歴＋状態MLP(KEYGRID原型)'), (best, 'key2', '最良行 − KEY2 別キー'),
             ('hs_x1_simple', 'hs_dct_simple', 'K-x1 − K-dct（MLPクエリ簡易）'), ('hs_x1_cteff', 'hs_dct_cteff', 'K-x1 − K-dct（MLPクエリCT版）'),
             ('ct1_simple', 'mqe_dct', 'K-x1 − K-dct（注意クエリ簡易）'),
             ('hs_x1_simple', 'hs_kg_simple', 'K-x1 − KEYGRID鍵（MLPクエリ簡易）'), ('hs_x1_cteff', 'hs_kg_cteff', 'K-x1 − KEYGRID鍵（MLPクエリCT版）'),
             ('hs_kg_simple', 'keygrid_orig', '同一学習器KEYGRID鍵 − KEYGRID原型'),
             ('final_ct_x1', 'hs_x1_cteff', '注意クエリ − MLPクエリ（K-x1, CT版）'),
             ('mk1c_orig', 'mk1c_fixed', 'MK1c 拡張あり − 拡張なし')]
    res['bootstrap'] = []
    for a, b, label in pairs:
        for sp in ('learned', 'unseen'):
            pa = perq[a][sp]
            pb = key2_separate_per_query(sp, np.asarray(splits[sp]['key'])) if b == 'key2' else perq[b][sp]
            d = boot(pa - pb, splits[sp])
            res['bootstrap'].append({'a': a, 'b': b, 'label': label, 'split': sp, **d})
    res['key2_check'] = {sp: float(key2_separate_per_query(sp, np.asarray(splits[sp]['key'])).mean()) for sp in ('learned', 'unseen')}
    # --- Dev vs unseen rank agreement across every setting
    mk = json.loads(MK1C.read_text()); ct = json.loads(CT1.read_text())
    pts = [('MK1c:' + n, r['dev_top1_arm6_mse'], r['evaluation']['unseen']['top1_arm6_mse']) for n, r in mk['rows'].items()]
    pts += [('CT1:' + n, r['dev_top1_arm6_mse'], r['evaluation']['unseen']['top1_arm6_mse']) for n, r in ct['rows'].items()]
    pts += [('KQF:' + n, r['dev_top1_arm6_mse'], ev(r, 'unseen')) for n, r in rows.items()]
    def rho(sel):
        sel = list(sel); return {'n': len(sel), 'spearman': spearman([x[1] for x in sel], [x[2] for x in sel])}
    res['dev_unseen_rank'] = {'all': rho(pts), 'MK1c_original': rho(p for p in pts if p[0].startswith('MK1c')),
                              'CT1': rho(p for p in pts if p[0].startswith('CT1')),
                              'KQF_all': rho(p for p in pts if p[0].startswith('KQF')),
                              'KQF_no_aug': rho(p for p in pts if p[0].startswith('KQF') and not rows[p[0][4:]]['cfg']['aug']),
                              'points': pts}
    # Dev composition
    dsp = ctx.splits['dev']; usp = ctx.splits['unseen']
    res['dev_composition'] = {'tasks': {int(t): int(c) for t, c in zip(*np.unique(dsp['task'], return_counts=True))},
                              'episodes': int(len(set(zip(dsp['task'].tolist(), dsp['episode'].tolist())))),
                              'frame_quantiles': np.percentile(np.asarray(dsp['key'])[:, 2], [0, 25, 50, 75, 100]).tolist(),
                              'unseen_tasks': {int(t): int(c) for t, c in zip(*np.unique(usp['task'], return_counts=True))},
                              'unseen_episodes': int(len(set(zip(usp['task'].tolist(), usp['episode'].tolist()))))}
    res['diag'] = json.loads((OUT / 'diag.json').read_text())
    kf.write_json(OUT / 'aggregate.json', res)
    for g, v in res['groups'].items():
        print(f"{g:14s} dev {v['dev_mean']:.4f} U {v['unseen_all_mean']:.4f}±{v['unseen_all_sd']:.4f}/{v['unseen_hard_mean']:.4f} L {v['learned_all_mean']:.4f}/{v['learned_hard_mean']:.4f} n={len(v['unseen_all_seeds'])}")
    print('best', best)
    for b in res['bootstrap']:
        print(f"{b['label']:40s} {b['split']:8s} {b['mean']:+.4f} [{b['ci95_low']:+.4f},{b['ci95_high']:+.4f}]")
    print(res['key2_check']); print({k: v for k, v in res['dev_unseen_rank'].items() if k != 'points'})


if __name__ == '__main__':
    main()
