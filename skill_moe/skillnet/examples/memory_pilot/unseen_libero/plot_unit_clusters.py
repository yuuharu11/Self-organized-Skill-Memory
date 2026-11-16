#!/usr/bin/env python3
"""Figures: do the stored motion units really form types, and are the types scene-independent?

Uses the exact CMP3 / UM2 unit memory (LIBERO-40 memory demos, 4,516 units cut at
speed minima + gripper switches, DCT3 at natural length, resampled to 32 phases)
and the same greedy cover (``compression_compare.cover_units``) that defines the
types.  No evaluation data is used.  CPU only.

Outputs (PNG + a small JSON of the numbers) go to ``--out``:
  unit_space.png     PCA of the 4,516 units, coloured by type and by LIBERO suite
  type_members.png   member trajectories of the largest types (arm xyz + gripper)
  type_stats.png     compactness and task / suite mixing vs a shuffled-label baseline
  unit_clusters.json the numbers behind the figures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'pi05_memory')]

SUITES = ('LIBERO-10', 'goal', 'object', 'spatial')  # task ids 0-9, 10-19, 20-29, 30-39
# Validated categorical slots (dataviz reference palette, light surface).
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
OTHER = '#c9c8c2'
INK, INK2, SURFACE = '#0b0b0b', '#52514e', '#fcfcfb'


def load_units():
    import compression_seen_cuts as c3
    import compression_compare as cc
    d = c3.c2.prepare()
    raws, tasks, demos = [], [], []
    for key, tr in d['memory_traces'].items():
        r, _ = c3.spans({key: tr}, {'mode': 'all', 'min_len': 5, 'width': 3})
        raws += r
        tasks += [int(key[0])] * len(r)
        demos += [tuple(map(int, key))] * len(r)
    coded = c3.dct3(raws)
    phase = np.stack([cc.phase_view(x) for x in coded]).astype(np.float32)  # [N, 32, 7]
    lengths = np.asarray([len(x) for x in raws])
    return phase, np.asarray(tasks), lengths, cc


def assign(phase, reps):
    """Each unit joins its nearest representative (same mean-squared distance as the cover)."""
    flat = phase.reshape(len(phase), -1).astype(np.float64)
    rep = flat[reps]
    d2 = (flat ** 2).sum(1)[:, None] - 2 * flat @ rep.T + (rep ** 2).sum(1)[None]
    d2 /= flat.shape[1]
    lab = d2.argmin(1)
    return lab, d2[np.arange(len(flat)), lab]


def entropy_counts(x):
    p = np.bincount(x) / len(x)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def type_stats(phase, lab, tasks, rng, n_shuffle=20, min_size=5):
    flat = phase.reshape(len(phase), -1).astype(np.float64)
    suite = tasks // 10

    def compact(labels):
        # mean distance to own type centroid, divided by mean distance to global centroid
        within = 0.
        for k in np.unique(labels):
            m = labels == k
            within += ((flat[m] - flat[m].mean(0)) ** 2).mean(1).sum()
        return within / len(flat) / ((flat - flat.mean(0)) ** 2).mean(1).mean()

    def mixing(labels):
        big = [k for k in np.unique(labels) if (labels == k).sum() >= min_size]
        n_task = np.asarray([len(np.unique(tasks[labels == k])) for k in big])
        n_suite = np.asarray([len(np.unique(suite[labels == k])) for k in big])
        h_task = np.asarray([entropy_counts(tasks[labels == k]) for k in big])
        return {'types': len(big), 'median_tasks_per_type': float(np.median(n_task)),
                'share_types_multi_suite': float((n_suite >= 2).mean()),
                'mean_task_entropy_bits': float(h_task.mean()),
                'n_task': n_task.tolist(), 'n_suite': n_suite.tolist()}

    real = {'compactness': float(compact(lab)), **mixing(lab)}
    shuf_c, shuf_m = [], []
    for _ in range(n_shuffle):
        p = rng.permutation(lab)
        shuf_c.append(compact(p))
        shuf_m.append(mixing(p))
    shuf = {'compactness_mean': float(np.mean(shuf_c)), 'compactness_sd': float(np.std(shuf_c)),
            'median_tasks_per_type': float(np.mean([m['median_tasks_per_type'] for m in shuf_m])),
            'share_types_multi_suite': float(np.mean([m['share_types_multi_suite'] for m in shuf_m])),
            'mean_task_entropy_bits': float(np.mean([m['mean_task_entropy_bits'] for m in shuf_m])),
            'n_task_example': shuf_m[0]['n_task']}
    # Fully task-specific reference: each type split by task (what "memorised per scene" would look like)
    per_task = np.asarray([f'{a}_{b}' for a, b in zip(lab, tasks)])
    _, per_task = np.unique(per_task, return_inverse=True)
    task_split = {'compactness': float(compact(per_task)), **{k: v for k, v in mixing(per_task).items()
                                                              if k not in ('n_task', 'n_suite')}}
    return real, shuf, task_split


def style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(OTHER)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_facecolor(SURFACE)


def plot_space(phase, lab, sizes_order, tasks, out, tau):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    # Arm 6 axes only: the gripper is constant (+-1) inside a unit and would
    # otherwise take PC1 and split the cloud into open / closed blobs.
    flat = phase[:, :, :6].reshape(len(phase), -1)
    grip_open = phase[:, :, 6].mean(1) < 0
    pca = PCA(2, random_state=0).fit(flat)
    xy = pca.transform(flat)
    ev = pca.explained_variance_ratio_
    top = sizes_order[:8]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), facecolor=SURFACE)
    ax = axes[0]
    rest = ~np.isin(lab, top)
    ax.scatter(xy[rest, 0], xy[rest, 1], s=5, c=OTHER, lw=0, label=f'other {len(np.unique(lab)) - 8} types')
    for i, k in enumerate(top):
        m = lab == k
        ax.scatter(xy[m, 0], xy[m, 1], s=9, c=SERIES[i], lw=0,
                   label=f'type {i + 1} (n={m.sum()}, {len(np.unique(tasks[m]))} tasks, '
                         f'{"open" if grip_open[m].mean() > .5 else "closed"})')
    ax.set_title(f'Coloured by type (greedy cover, tau={tau}): 8 largest types', fontsize=10, color=INK, loc='left')
    ax.legend(fontsize=7, frameon=False, loc='best', markerscale=1.6)
    ax = axes[1]
    order = np.random.default_rng(0).permutation(len(xy))  # avoid one suite painting over another
    suite = tasks // 10
    for i, name in enumerate(SUITES):
        ax.scatter([], [], s=9, c=SERIES[i], label=f'{name} (n={(suite == i).sum()})')
    ax.scatter(xy[order, 0], xy[order, 1], s=5, c=np.asarray(SERIES[:4])[suite[order]], lw=0)
    ax.set_title('Same units, coloured by the LIBERO suite they were cut from', fontsize=10, color=INK, loc='left')
    ax.legend(fontsize=7, frameon=False, loc='best', markerscale=1.6)
    for ax in axes:
        style(ax)
        ax.set_xlabel(f'PC1 ({ev[0]:.0%} var.)', fontsize=8, color=INK2)
        ax.set_ylabel(f'PC2 ({ev[1]:.0%} var.)', fontsize=8, color=INK2)
    fig.suptitle('4,516 stored motion units (LIBERO-40 memory, DCT3, 32 phases); PCA of the arm 6 axes '
                 '(gripper is a separate channel)',
                 fontsize=11, color=INK, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(out / 'unit_space.png', dpi=150)
    plt.close(fig)
    return ev.tolist()


def plot_members(phase, lab, sizes_order, tasks, out, rng):
    import matplotlib.pyplot as plt
    top = sizes_order[:8]
    t = np.linspace(0, 1, phase.shape[1])
    axis_names = ['x', 'y', 'z']
    axis_col = [SERIES[0], SERIES[1], SERIES[2]]
    fig, axes = plt.subplots(1, 8, figsize=(16, 3.6), facecolor=SURFACE, sharex=True)
    lim = np.quantile(np.abs(phase[np.isin(lab, top)][:, :, :3]), .995)
    for i, k in enumerate(top):
        idx = np.flatnonzero(lab == k)
        show = rng.choice(idx, min(60, len(idx)), replace=False)
        ax = axes[i]
        for a in range(3):
            for j in show:
                ax.plot(t, phase[j, :, a], color=axis_col[a], lw=.6, alpha=.18)
            ax.plot(t, np.median(phase[idx, :, a], 0), color=axis_col[a], lw=2,
                    label=axis_names[a] if i == 0 else None)
        grip = 'open' if phase[idx, :, 6].mean() < 0 else 'closed'
        ax.set_title(f'type {i + 1} (gripper {grip})\nn={len(idx)}, {len(np.unique(tasks[idx]))} tasks, '
                     f'{len(np.unique(tasks[idx] // 10))} suites', fontsize=8, color=INK)
        ax.set_ylim(-lim, lim)
        style(ax)
        ax.axhline(0, color=OTHER, lw=.6)
        if i:
            ax.set_yticklabels([])
        ax.set_xlabel('phase', fontsize=7, color=INK2)
    axes[0].set_ylabel('arm translation\n(normalised action)', fontsize=8, color=INK2)
    axes[0].legend(fontsize=7, frameon=False, loc='upper left')
    fig.suptitle('Largest 8 types: up to 60 member units each (thin) and the per-phase median (bold)',
                 fontsize=11, color=INK, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(out / 'type_members.png', dpi=150)
    plt.close(fig)


def plot_stats(stats, out, tau):
    import matplotlib.pyplot as plt
    real, shuf, task_split = stats
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor=SURFACE)
    ax = axes[0]
    vals = [real['compactness'], shuf['compactness_mean']]
    ax.barh([1, 0], vals, color=[SERIES[0], OTHER], height=.5)
    ax.set_yticks([1, 0], ['types', 'shuffled labels'])
    for y, v in zip([1, 0], vals):
        ax.text(v, y, f'  {v:.2f}', va='center', fontsize=9, color=INK)
    ax.set_xlim(0, 1.25)
    ax.set_title('Within-type spread / total spread\n(lower = tighter types)', fontsize=9, color=INK, loc='left')
    ax = axes[1]
    bins = np.arange(1, 42)
    ax.hist(real['n_task'], bins=bins, color=SERIES[0], label='types', alpha=.9)
    ax.hist(shuf['n_task_example'], bins=bins, histtype='step', color=INK2, lw=1.2, label='shuffled labels')
    ax.set_xlabel('distinct tasks among members', fontsize=8, color=INK2)
    ax.set_ylabel(f'types with >= 5 members (n={real["types"]})', fontsize=8, color=INK2)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title(f'Tasks per type (median {real["median_tasks_per_type"]:.0f}; shuffled '
                 f'{shuf["median_tasks_per_type"]:.0f})', fontsize=9, color=INK, loc='left')
    ax = axes[2]
    labels = ['types', 'shuffled labels', 'types split\nby task']
    vals = [real['share_types_multi_suite'], shuf['share_types_multi_suite'], task_split['share_types_multi_suite']]
    ax.barh([2, 1, 0], vals, color=[SERIES[0], OTHER, OTHER], height=.5)
    ax.set_yticks([2, 1, 0], labels)
    for y, v in zip([2, 1, 0], vals):
        ax.text(v, y, f'  {v:.0%}', va='center', fontsize=9, color=INK)
    ax.set_xlim(0, 1.2)
    ax.set_title('Share of types drawing members\nfrom >= 2 LIBERO suites', fontsize=9, color=INK, loc='left')
    for ax in axes:
        style(ax)
    fig.suptitle(f'Are the types tight in motion, and shared across scenes? (tau={tau})',
                 fontsize=11, color=INK, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(out / 'type_stats.png', dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tau', type=float, default=.08)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    phase, tasks, lengths, cc = load_units()
    if len(phase) != 4516:
        raise AssertionError(f'expected the 4,516-unit CMP3 memory, got {len(phase)}')
    reps = np.asarray(cc.cover_units(phase, a.tau), np.int64)
    lab, dist = assign(phase, reps)
    sizes = np.bincount(lab, minlength=len(reps))
    order = np.argsort(-sizes, kind='stable')
    stats = type_stats(phase, lab, tasks, rng)
    ev = plot_space(phase, lab, order, tasks, a.out, a.tau)
    plot_members(phase, lab, order, tasks, a.out, rng)
    plot_stats(stats, a.out, a.tau)
    real, shuf, task_split = stats
    summary = {
        'units': int(len(phase)), 'tau': a.tau, 'types': int(len(reps)),
        'type_size': {'max': int(sizes.max()), 'median': float(np.median(sizes)),
                      'singletons': int((sizes == 1).sum()),
                      'units_in_8_largest': int(sizes[order[:8]].sum())},
        'pca_explained_variance_2d': ev,
        'real': {k: v for k, v in real.items() if k not in ('n_task', 'n_suite')},
        'shuffled_labels': {k: v for k, v in shuf.items() if k != 'n_task_example'},
        'types_split_by_task': task_split,
        'note': 'Units and types exactly as in CMP3/UM2 (cover_units on 32-phase DCT3 values). '
                'Non-representatives join their nearest representative. No evaluation data used.',
    }
    (a.out / 'unit_clusters.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
