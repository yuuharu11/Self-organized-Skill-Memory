#!/usr/bin/env python3
"""Render t0 sweep figures from noise_curves.csv (CPU only)."""
from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[6]
STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', REPO_ROOT / '.skillnet-state')).expanduser()
PI05_ROOT = Path(os.environ.get('SKILLNET_PI05_ROOT', STATE_DIR / 'outputs/memory-pilot/pi05-memory40-20261104')).expanduser()
ROOT = Path(os.environ.get('COMPRESS_CLUSTER_OUT', PI05_ROOT / 'compress-cluster-oracle-01')).expanduser()
OUT = Path(os.environ.get('NOISE_SWEEP_OUT', ROOT / 'noise-sweep-03'))
FIG = REPO_ROOT / 'docs/research/07_pi05_memory40/compress_cluster_oracle/figures'
T0 = np.asarray([.05, .1, .2, .3, .4, .5, .6, .8, 1.0])
REPS = ('raw', 'b12', 'b24', 'b48', 'b96')
LEVELS = ('none', 'fine', 'mid', 'coarse', 'm6default')
REP_LABEL = {'raw': 'raw', 'b12': 'b12', 'b24': 'b24', 'b48': 'b48', 'b96': 'b96'}
LEVEL_LABEL = {'none': 'none', 'fine': 'fine', 'mid': 'mid', 'coarse': 'coarse', 'm6default': 'm6default'}
PALETTE = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4')
REP_COLOR = dict(zip(REPS, PALETTE))
LEVEL_COLOR = dict(zip(LEVELS, PALETTE))
REP_MARK = dict(zip(REPS, ('o', 's', '^', 'D', 'v')))
LEVEL_MARK = dict(zip(LEVELS, ('o', 's', '^', 'D', 'v')))
DRAFT_MARK = {'primary': 'o', 'shuffle': 's', 'zero': '^'}
BG = '#fcfcfb'
TEXT = '#0b0b0b'
MUTED = '#52514e'


def read_rows():
    with (OUT / 'noise_curves.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ('mse_mean', 'ci_low', 'ci_high', 'sensitivity'):
            r[k] = float(r[k])
        r['t0'] = float(r['t0']) if r['t0'] else np.nan
    return rows


def value(rows, expert, draft, subset, field='mse_mean', t0=None):
    for r in rows:
        if (r['expert'] == expert and r['draft'] == draft and r['subset'] == subset and
                (t0 is None or np.isclose(r['t0'], t0, atol=1e-8))):
            return r[field]
    raise KeyError((expert, draft, subset, field, t0))


def _setup():
    plt.rcParams.update({
        'figure.facecolor': BG, 'axes.facecolor': BG, 'savefig.facecolor': BG,
        'text.color': TEXT, 'axes.labelcolor': TEXT, 'xtick.color': MUTED,
        'ytick.color': MUTED, 'axes.edgecolor': MUTED, 'font.size': 9,
        'axes.titlesize': 10, 'legend.fontsize': 8, 'lines.linewidth': 2,
        'lines.markersize': 8, 'grid.color': '#deddda', 'grid.linewidth': .6,
    })


def _baseline(ax, rows, subset, *, field='mse_mean'):
    if field == 'sensitivity':
        ax.axhline(0.0, color='#9b9a96', lw=1, ls=':', label='normal / E-zero = 0')
        ax.text(.99, 0.0, 'normal / E-zero 0', transform=ax.get_yaxis_transform(),
                ha='right', va='bottom', color=MUTED, fontsize=7,
                bbox={'facecolor': BG, 'edgecolor': 'none', 'alpha': .85, 'pad': 1})
        return
    normal = value(rows, 'official_normal', 'none', subset)
    ez = [(value(rows, 'E-zero', 'zero', subset, t0=float(t)), float(t)) for t in T0]
    best, best_t = min(ez)
    ax.axhline(normal, color='#9b9a96', lw=1, ls=':', label=f'normal {normal:.4f}')
    ax.axhline(best, color='#52514e', lw=1, ls='--', label=f'E-zero best {best:.4f} @ {best_t:g}')
    # Explicit direct labels at the right edge; x is in axes coordinates.
    ax.text(.99, normal, f'Normal {normal:.4f}', transform=ax.get_yaxis_transform(),
            ha='right', va='bottom', color=MUTED, fontsize=7,
            bbox={'facecolor': BG, 'edgecolor': 'none', 'alpha': .85, 'pad': 1})
    ax.text(.99, best, f'E-zero best {best:.4f}', transform=ax.get_yaxis_transform(),
            ha='right', va='top', color=MUTED, fontsize=7,
            bbox={'facecolor': BG, 'edgecolor': 'none', 'alpha': .85, 'pad': 1})


def _common_axis(ax, *, ylabel=True, log=False):
    ax.axvspan(.2, .6, color='#c9c8c3', alpha=.17, zorder=0)
    ax.set_xlim(.035, 1.02)
    ax.set_xticks(T0)
    ax.set_xticklabels([f'{x:g}' for x in T0], rotation=35, ha='right')
    ax.set_xlabel('start noise t0')
    if ylabel:
        ax.set_ylabel('generated active7 MSE' + (' (log scale)' if log else ''))
    if log:
        ax.set_yscale('log')
    ax.grid(True, axis='y', alpha=.65)
    ax.grid(False, axis='x')


def _plot_metric(rows, field, filename, *, by='level'):
    fig, axs = plt.subplots(2, 5, figsize=(18, 7.3), sharex=True, sharey='row',
                            constrained_layout=True)
    for ri, subset in enumerate(('all', 'hard')):
        for ci, panel in enumerate(LEVELS if by == 'level' else REPS):
            ax = axs[ri, ci]
            _baseline(ax, rows, subset, field=field)
            if by == 'level':
                for rep in REPS:
                    expert = f'{rep}_{panel}'
                    if not any(r['expert'] == expert and r['draft'] == 'primary' and r['subset'] == subset for r in rows):
                        continue
                    ys = np.asarray([value(rows, expert, 'primary', subset, field, float(t)) for t in T0])
                    color = REP_COLOR[rep]
                    ax.plot(T0, ys, color=color, marker=REP_MARK[rep], label=REP_LABEL[rep])
                ax.set_title(f'{panel} clustering')
            else:
                for level in LEVELS:
                    expert = f'{panel}_{level}'
                    if not any(r['expert'] == expert and r['draft'] == 'primary' and r['subset'] == subset for r in rows):
                        continue
                    ys = np.asarray([value(rows, expert, 'primary', subset, field, float(t)) for t in T0])
                    color = LEVEL_COLOR[level]
                    ax.plot(T0, ys, color=color, marker=LEVEL_MARK[level], label=LEVEL_LABEL[level])
                ax.set_title(f'{panel} representation')
            # The E-M2+M2 curve is a useful frozen baseline, separate from the
            # two horizontal policy references shown in every panel.
            m2 = np.asarray([value(rows, 'E-M2', 'M2', subset, field, float(t)) for t in T0])
            ax.plot(T0, m2, color='#787772', marker='P', ls='-.', label='E-M2 + M2')
            # The hard subset spans more than an order of magnitude, so use
            # a log axis for MSE panels; sensitivity remains on a linear axis.
            _common_axis(ax, ylabel=False, log=(field == 'mse_mean'))
            if ci == 0:
                ax.set_ylabel('generated active7 MSE (log scale)' if field == 'mse_mean'
                              else 'draft-induced output change (RMS)')
            if ci == 0:
                ax.text(-.25, .5, 'all Long' if subset == 'all' else 'hard',
                        transform=ax.transAxes, rotation=90, va='center', ha='center',
                        color=TEXT, fontsize=11)
            if ri == 0 and ci == 0:
                ax.legend(loc='upper left', frameon=False, ncol=2)
            if ri == 1 and ci == 4:
                ax.text(.99, .01, 'shaded: trained t0 [.2,.6]; outside = extrapolation',
                        transform=ax.transAxes, ha='right', va='bottom', fontsize=7, color=MUTED)
    fig.suptitle('Noise strength vs generated action error' if field == 'mse_mean'
                 else 'Noise strength vs draft-induced output sensitivity', fontsize=13)
    _save(fig, filename)


def _plot_draft_content(rows):
    fig, axs = plt.subplots(2, 5, figsize=(18, 7.3), sharex=True, sharey='row',
                            constrained_layout=True)
    styles = {'primary': '-', 'shuffle': '--', 'zero': ':'}
    for ri, subset in enumerate(('all', 'hard')):
        for ci, level in enumerate(LEVELS):
            ax = axs[ri, ci]
            _baseline(ax, rows, subset)
            for rep in ('raw', 'b12'):
                expert = f'{rep}_{level}'
                if not any(r['expert'] == expert and r['subset'] == subset for r in rows):
                    continue
                for draft in ('primary', 'shuffle', 'zero'):
                    ys = np.asarray([value(rows, expert, draft, subset, 'mse_mean', float(t)) for t in T0])
                    ax.plot(T0, ys, color=REP_COLOR[rep], marker=DRAFT_MARK[draft],
                            ls=styles[draft], label=f'{rep} {draft}')
            _common_axis(ax, ylabel=(ci == 0), log=True)
            ax.set_title(f'{level} clustering')
            if ci == 0:
                ax.text(-.25, .5, 'all Long' if subset == 'all' else 'hard',
                        transform=ax.transAxes, rotation=90, va='center', ha='center',
                        color=TEXT, fontsize=11)
            if ri == 0 and ci == 0:
                ax.legend(loc='upper left', frameon=False, ncol=2)
            if ri == 1 and ci == 4:
                ax.text(.99, .01, 'shaded: trained t0 [.2,.6]; outside = extrapolation',
                        transform=ax.transAxes, ha='right', va='bottom', fontsize=7, color=MUTED)
    fig.suptitle('Draft content effect: primary vs shuffled vs zero (log MSE)', fontsize=13)
    _save(fig, 'draft_content.png')


def _save(fig, name):
    path = FIG / name
    if path.exists():
        raise FileExistsError(f'refusing to overwrite figure: {path}')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(path)


def main():
    if not (OUT / 'noise_curves.csv').exists():
        raise FileNotFoundError(OUT / 'noise_curves.csv')
    _setup()
    FIG.mkdir(parents=True, exist_ok=True)
    rows = read_rows()
    for name in ('noise_by_level.png', 'noise_by_representation.png', 'sensitivity.png', 'draft_content.png'):
        if (FIG / name).exists():
            raise FileExistsError(f'refusing to overwrite figure: {FIG/name}')
    _plot_metric(rows, 'mse_mean', 'noise_by_level.png', by='level')
    _plot_metric(rows, 'mse_mean', 'noise_by_representation.png', by='representation')
    _plot_metric(rows, 'sensitivity', 'sensitivity.png', by='level')
    _plot_draft_content(rows)


if __name__ == '__main__':
    main()
