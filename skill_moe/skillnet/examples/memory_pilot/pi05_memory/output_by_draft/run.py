#!/usr/bin/env python3
"""How generated output changes with the draft: analyses over saved artifacts (CPU only).

Writes results.json next to --output. Reads saved predictions and never overwrites them.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[6]
STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', REPO_ROOT / '.skillnet-state')).expanduser()
ROOT = Path(os.environ.get('SKILL_MEMORY_ARTIFACTS', os.environ.get('SKILLNET_PI05_ROOT',
                           STATE_DIR / 'outputs/memory-pilot/pi05-memory40-20261104'))).expanduser()
DRAFT_BINS = (0, .002, .02, .04, .08, 10.)
T0S = (.2, .4, .6)
PCA_K = (2, 4, 6, 8, 12, 16, 24, 32, 48)
AXES = ('x', 'y', 'z', 'rx', 'ry', 'rz', 'grip')


def direction_cos(a, b):
    """Mean cosine between the summed xyz displacement of two [N,H,>=3] chunks."""
    u, v = np.asarray(a)[:, :, :3].sum(1), np.asarray(b)[:, :, :3].sum(1)
    n = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
    return float(np.divide((u * v).sum(1), n, out=np.zeros(len(u)), where=n > 1e-8).mean())


def row_mse(a, b):
    return np.mean((np.asarray(a)[:, :, :7] - np.asarray(b)[:, :, :7]) ** 2, axis=(1, 2))


def bootstrap(diff, episodes, *, reps=5000, seed=4101):
    d, ep = np.asarray(diff, np.float64), np.asarray(episodes)
    uniq = np.unique(ep)
    groups = {u: np.flatnonzero(ep == u) for u in uniq}
    rng = np.random.default_rng(seed)
    means = [d[np.concatenate([groups[u] for u in rng.choice(uniq, len(uniq))])].mean() for _ in range(reps)]
    return {'mean': float(d.mean()), 'ci95': [float(np.quantile(means, .025)), float(np.quantile(means, .975))]}


def bin_by_draft(draft, normal, generated):
    out = []
    for lo, hi in zip(DRAFT_BINS[:-1], DRAFT_BINS[1:]):
        m = (draft >= lo) & (draft < hi)
        if m.sum() < 20:
            continue
        out.append({'draft_low': lo, 'draft_high': hi, 'points': int(m.sum()), 'normal': float(normal[m].mean()),
                    **{f't0={t}': float(generated[t][m].mean()) for t in T0S},
                    **{f'below_normal_t0={t}': float((generated[t][m] < normal[m]).mean()) for t in T0S}})
    return out


def latest_evaluation(cell_dir):
    files = sorted(glob.glob(str(cell_dir / 'attempt-*' / 'evaluation_rows.npz')))
    if not files:
        raise FileNotFoundError(cell_dir)
    return np.load(files[-1], allow_pickle=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT / 'output-by-draft-01')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    experts = ROOT / 'compress-cluster-oracle-01' / 'experts'
    fair = np.load(ROOT / 'fair-expert-01' / 'evaluation-final' / 'offline_predictions.npz', allow_pickle=False)
    rows, hard = fair['row_ids'], fair['hard_mask'].astype(bool)
    episodes = fair['task_ids'].astype(np.int64) * 100000 + fair['episode_ids'].astype(np.int64)
    normal, m2 = fair['mse_official_normal'], fair['mse_E-M2_M2_top1_t0_0.6']
    result = {'long_rows': len(rows), 'hard_rows': int(hard.sum())}

    # 1. Draft distance -> generated error, pooled over expert conditions (raw x none excluded: distance 0).
    draft, base, generated = [], [], {t: [] for t in T0S}
    cells = sorted(p.name for p in experts.iterdir() if p.is_dir())
    for cell in cells:
        z = latest_evaluation(experts / cell)
        if not np.array_equal(z['row_ids'], rows):
            raise AssertionError(f'{cell}: row order differs')
        if cell == 'raw_none':
            continue
        draft.append(row_mse(z['primary_draft'], z['target'])); base.append(normal)
        for t in T0S:
            generated[t].append(z[f'mse_primary_t{t}'])
    result['recovery_by_draft_distance'] = {
        'conditions': len(draft), 'points': int(sum(map(len, draft))),
        'bins': bin_by_draft(np.concatenate(draft), np.concatenate(base),
                             {t: np.concatenate(v) for t, v in generated.items()})}

    # 2. Wrong core: pass another task's draft and see whether the output stays near it.
    wrong = {}
    for cell in ('raw_none', 'raw_mid', 'raw_m6default'):
        z = latest_evaluation(experts / cell)
        d, target = z['shuffle_draft'][:, :, :7], z['target']
        wrong[cell] = {'draft_to_target': float(row_mse(d, target).mean()), 'draft_target_cos': direction_cos(d, target)}
        for t in T0S:
            p = z[f'pred_shuffle_t{t}'][:, :, :7]
            wrong[cell][f't0={t}'] = {'output_to_core': float(row_mse(p, d).mean()),
                                      'output_to_target': float(row_mse(p, target).mean()),
                                      'cos_core': direction_cos(p, d), 'cos_target': direction_cos(p, target)}
    result['wrong_core'] = wrong

    # 3. Oracle clusters versus the real retrieval (E-M2 + M2, t0=.6): is the upper bound above it?
    summary = json.loads((ROOT / 'compress-cluster-oracle-01' / 'summary.json').read_text())['experts']
    versus = {}
    for cell in ('raw_fine', 'raw_mid', 'raw_m6default', 'raw_coarse', 'b12_mid', 'b12_m6default', 'b12_coarse'):
        t = summary[cell]['selected_t0']
        a = latest_evaluation(experts / cell)[f'mse_primary_t{t:.1f}']
        versus[cell] = {'selected_t0': t, 'all': bootstrap(a - m2, episodes),
                        'hard': bootstrap((a - m2)[hard], episodes[hard])}
    result['oracle_cluster_minus_m2'] = versus

    # 4. PCA fitted without Long or the Long Eval rows.
    run = ROOT / 'run-01' / 'rows'
    actions = np.load(run / 'actions.npy', mmap_mode='r')
    task, validation = np.load(run / 'task.npy'), np.load(run / 'validation.npy').astype(bool)
    fit = np.flatnonzero((task >= 10) & ~validation)
    if np.any(task[fit] < 10) or np.intersect1d(fit, rows).size:
        raise AssertionError('Long rows entered the PCA fit')
    x = np.asarray(actions[fit])[:, :, :7].reshape(len(fit), -1).astype(np.float64)
    y = np.asarray(actions[rows])[:, :, :7].reshape(len(rows), -1).astype(np.float64)
    mean = x.mean(0)
    _, s, vt = np.linalg.svd(x - mean, full_matrices=False)
    explained = s ** 2 / (s ** 2).sum()
    pca = {'fit_rows': len(fit), 'reconstruction': [], 'components': []}
    for k in PCA_K:
        e = np.mean((((y - mean) @ vt[:k].T @ vt[:k] + mean) - y) ** 2, axis=1)
        pca['reconstruction'].append({'k': k, 'explained': float(explained[:k].sum()), 'all': float(e.mean()),
                                      'hard': float(e[hard].mean()), 'median': float(np.median(e)),
                                      'within_01': float((e <= .01).mean())})
    for i in range(6):
        energy = (vt[i].reshape(10, 7) ** 2).sum(0)
        pca['components'].append({'component': i + 1, 'explained': float(explained[i]),
                                  'energy': {AXES[j]: float(energy[j]) for j in np.argsort(-energy)[:2]}})
    result['pca'] = pca
    path = args.output / 'results.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(path)


if __name__ == '__main__':
    main()
