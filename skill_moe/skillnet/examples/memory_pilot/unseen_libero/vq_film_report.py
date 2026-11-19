#!/usr/bin/env python3
"""VQ-A report: 20 Hz tables (+ old 10 Hz unseen column), hold-last, subsets, task-ID k-NN probe on z,
FiLM deviation, learned-unseen gap vs bottleneck, paired episode bootstrap vs C1b and hold-last."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import os
REPO_ROOT = Path(__file__).resolve().parents[5]
STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', REPO_ROOT / '.skillnet-state')).expanduser()
sys.path.insert(0, str(HERE))
import key_shared_latent as k2  # noqa: E402
import dev_v2  # noqa: E402

OUT = (STATE_DIR / 'outputs/memory-pilot/unseen-libero/vq-a-01')
KQ = (STATE_DIR / 'outputs/memory-pilot/unseen-libero/key-motion-01/final')
VQC = (STATE_DIR / 'outputs/memory-pilot/unseen-libero/vq-c-01')
ROWS = OUT / 'rows20'
SPLITS = ('dev2', 'learned', 'unseen20', 'unseen')
KQ_ROWS = {'C1b': ['H_x1_mk1b_cteff_tau0.01_s1', 'H_x1_mk1b_cteff_tau0.01_s2', 'H_x1_mk1b_cteff_tau0.01_s3'],
           'C1a': ['Afix_x1_fixed_e_h_cteff', 'Afix_x1_fixed_e_h_cteff_s2', 'Afix_x1_fixed_e_h_cteff_s3'],
           '連結 E-VLM+H（固定キー）': ['Afix_x1_fixed_e_vlm_h'],
           '連結 E-VLM+H（キー微調整）': ['Bfix_x1_finetune_e_vlm_h']}


def load_split(name):
    if name == 'dev2':
        return dev_v2.load_dev_v2_20()
    if name == 'unseen20':
        d = VQC / 'unseen20'
        return {k: np.load(d / f'{k}.npy') for k in ('key', 'task', 'episode', 'hard', 'history', 'history_mask', 'actions')}
    return {k: np.load(KQ / f'data/{name}__{k}.npy') for k in ('key', 'task', 'episode', 'hard', 'history', 'history_mask', 'actions')}


SP = {s: load_split(s) for s in SPLITS}
SUB = {s: ({k[4:]: v for k, v in SP[s].items() if k.startswith('sub_')} if s == 'dev2' else dev_v2.eval_subsets(SP[s]))
       for s in SPLITS}
for s in SPLITS:
    SUB[s]['hard'] = np.asarray(SP[s]['hard'], bool)


def knn_task_acc(x, task, episode, k=5):
    """Leave-one-episode-out cosine k-NN task decoding accuracy."""
    x = np.asarray(x, np.float64); x = x - x.mean(0)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    s = x @ x.T
    s[(episode[:, None] == episode[None, :]) & (task[:, None] == task[None, :])] = -np.inf
    nn = np.argsort(-s, 1)[:, :k]
    return float(np.mean(np.asarray([np.bincount(task[r]).argmax() for r in nn]) == task))


def hold_last():
    out = {}
    for s in SPLITS:
        e = k2.action_errors(k2._hold_last_actions(SP[s]), np.asarray(SP[s]['actions']))['arm6']
        out[s] = {'per': e, 'all': float(e.mean()), 'sub': {k: float(e[m].mean()) for k, m in SUB[s].items()}}
    return out


def summarize(pers: list[np.ndarray], s: str):
    p = np.mean(pers, 0)
    seeds = [float(x.mean()) for x in pers]
    return {'all': float(p.mean()), 'seed_all': seeds, 'sd': float(np.std(seeds)) if len(seeds) > 1 else None,
            'sub': {k: float(np.mean([x[m].mean() for x in pers])) for k, m in SUB[s].items()}, 'per': p}


def boot(a, b, s):
    t = np.asarray(SP[s]['task']).astype(np.int64); e = np.asarray(SP[s]['episode']).astype(np.int64)
    r = k2.episode_bootstrap_ci(a - b, t, e)
    out = {'all': r}
    fu = SUB[s]['first_unit']
    if fu.any():
        out['first_unit'] = k2.episode_bootstrap_ci((a - b)[fu], t[fu], e[fu])
    return out


def main():
    hl = hold_last()
    rows = {p.stem: json.loads(p.read_text()) for p in sorted(ROWS.glob('*.json'))}
    kq = json.loads((OUT / 'rescore_kq.json').read_text())
    groups: dict[str, dict] = {}
    for n, r in rows.items():
        groups.setdefault(re.sub(r'_s\d+$', '', n), []).append(r)
    table = {}
    for g, rs in groups.items():
        rec = {'cfg': {k: rs[0]['cfg'].get(k) for k in ('bottleneck', 'k', 'd', 'vib', 'kl', 'film_at')},
               'seeds': len(rs), 'dev2_selected': [r['dev_top1_arm6_mse'] for r in rs], 'epochs': [r['selected_epoch'] for r in rs]}
        for s in SPLITS:
            sm = summarize([np.asarray(r['evaluation'][s]['per_query_top1']) for r in rs], s)
            rec[s] = sm
            rec[s + '_film'] = rs[0]['evaluation'][s].get('film')
        rec['gap20'] = rec['unseen20']['all'] - rec['learned']['all']
        pr = {}
        for s in ('dev2', 'unseen20', 'learned'):
            p = ROWS / f'{rs[0]["name"]}__{s}_z.npz'
            if p.exists():
                pr[s] = knn_task_acc(np.load(p)['z'].astype(np.float32), np.asarray(SP[s]['task']).astype(int),
                                     np.asarray(SP[s]['episode']).astype(np.int64))
        rec['task_probe_z'] = pr
        table[g] = rec
    for g, names in KQ_ROWS.items():
        rec = {'cfg': 'KQ-final', 'seeds': len(names)}
        for s in SPLITS:
            rec[s] = summarize([np.asarray(kq[n][s]['per_query_top1']) for n in names], s)
        rec['gap20'] = rec['unseen20']['all'] - rec['learned']['all']
        table[g] = rec
    # paired comparisons for every row vs C1b (KQ-final seed mean) and vs hold-last
    for g, rec in table.items():
        rec['vs_C1b'] = {s: boot(rec[s]['per'], table['C1b'][s]['per'], s) for s in ('learned', 'unseen20')}
        rec['vs_hold_last'] = {s: boot(rec[s]['per'], hl[s]['per'], s) for s in ('learned', 'unseen20')}
    # reference task probes on raw inputs
    refs = {}
    for s in ('dev2', 'unseen20'):
        t = np.asarray(SP[s]['task']).astype(int); e = np.asarray(SP[s]['episode']).astype(np.int64)
        if s == 'dev2':
            tok = np.load(dev_v2.OUT / 'data/tokens.npy', mmap_mode='r'); grp = np.load(dev_v2.OUT / 'data/groups.npy')
            rix = SP[s]['row']
        else:
            tok = np.load(KQ / 'data/unseen__tokens.npy', mmap_mode='r'); grp = np.load(KQ / 'data/unseen__groups.npy')
            rix = np.load(VQC / 'unseen20/row.npy')
        pooled = []
        for a in range(0, len(rix), 64):
            ix = rix[a:a + 64]; tk = np.asarray(tok[ix], np.float32); g_ = grp[ix] >= 0
            pooled.append((tk * g_[..., None]).sum(1) / g_.sum(1, keepdims=True))
        hs = np.concatenate([np.asarray(SP[s]['history'], np.float32).reshape(len(t), -1), np.asarray(SP[s]['history_mask'], np.float32)], 1)
        refs[s] = {'pooled_vlm_2048': knn_task_acc(np.concatenate(pooled), t, e), 'history_70': knn_task_acc(hs, t, e),
                   'chance': float(1 / len(np.unique(t)))}

    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if k != 'per'}
        return x
    res = {'hold_last': strip(hl), 'rows': strip(table), 'reference_task_probe': refs, 'shift_check': None,
           'note': 'dev2 = Dev v2 20 Hz (selection); unseen20 = locked unseen 619 rebuilt at 20 Hz (VQ-C); unseen = old 10 Hz (time-scale bug), continuity only'}
    # Dev v2 20 Hz: 6 frame-0 rows start 1-11 steps after the 10 Hz image (episode-start no-ops), so their
    # VLM tokens come from a slightly earlier image.  Check that dropping them does not change the selection.
    shift = np.load(dev_v2.OUT / 'data20/start_shift_steps.npy') != 0
    for g, rec in table.items():
        rec['dev2_without_shifted6'] = float(rec['dev2']['per'][~shift].mean())
    order_all = sorted(table, key=lambda g: table[g]['dev2']['all'])
    order_drop = sorted(table, key=lambda g: table[g]['dev2_without_shifted6'])
    from scipy.stats import spearmanr
    res_shift = {'rows_shifted': int(shift.sum()), 'best_all': order_all[:3], 'best_without': order_drop[:3],
                 'spearman_all_vs_without': float(spearmanr([table[g]['dev2']['all'] for g in table],
                                                            [table[g]['dev2_without_shifted6'] for g in table]).statistic)}
    print('shift check', res_shift)
    res['shift_check'] = res_shift
    for g in res['rows']:
        res['rows'][g]['dev2_without_shifted6'] = table[g]['dev2_without_shifted6']
    (OUT / 'results.json').write_text(json.dumps(res, indent=1, ensure_ascii=False) + '\n')
    print('hold_last', {s: round(hl[s]['all'], 4) for s in SPLITS}, {s: round(hl[s]['sub']['first_unit'], 4) for s in SPLITS})
    print('refs', refs)
    for g, rec in sorted(table.items(), key=lambda x: x[1]['dev2']['all']):
        f = rec.get('unseen20_film') or {}
        print(f"{g:34s} n{rec['seeds']} D {rec['dev2']['all']:.4f} L {rec['learned']['all']:.4f} U20 {rec['unseen20']['all']:.4f} "
              f"U10 {rec['unseen']['all']:.4f} gap {rec['gap20']:+.4f} U20fu {rec['unseen20']['sub']['first_unit']:.4f} "
              f"U20sw {rec['unseen20']['sub']['post_switch']:.4f} Dfu {rec['dev2']['sub']['first_unit']:.4f} "
              f"rel {f.get('rel_change_norm', 0):.3f} probe {rec.get('task_probe_z', {}).get('unseen20', float('nan')):.2f}")


if __name__ == '__main__':
    main()
