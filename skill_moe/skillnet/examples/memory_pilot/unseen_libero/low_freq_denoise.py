#!/usr/bin/env python3
"""Action-space denoising whose noise lives only in the low temporal DCT frequencies.

Same drafts, trainer, Dev selection and Eval as the white-noise A conditions of
`compressed_denoise.py`; only the noise changes. For a 10-frame window, white
noise eps [10, D] is projected onto the lowest k orthonormal DCT-II basis vectors
along time and scaled by sqrt(10/k), so the mean per-element variance stays 1:

    eps_low = sqrt(10/k) * B_k B_k^T eps        (B_k: [10, k] DCT basis)
    start   = (1 - t0) * draft + t0 * eps_low

Training and generation both use eps_low; the high-frequency part of the start
point is the draft's own. `compressed_denoise.py` is imported unchanged and its
noise hooks are replaced in this process only, so the white-noise path is intact.

    python low_freq_denoise.py energy                 # draft-error energy by frequency (CPU)
    python low_freq_denoise.py train    --condition L-vm-none-k3
    python low_freq_denoise.py evaluate --condition L-vm-none-k3
    python low_freq_denoise.py report
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.fft import dct
STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', Path(__file__).resolve().parents[5] / '.skillnet-state')).expanduser()

HERE = Path(__file__).resolve().parent
ROOT = (STATE_DIR / 'outputs/memory-pilot/unseen-libero/compressed-denoise-01')
WHITE = ROOT / 'production'          # white-noise A results and the shared drafts (read only)
OUT = ROOT / 'production-lowfreq'
HORIZON = 10
T0_EVAL = (.05, .1, .2, .3, .4, .6, .8, 1.)

# name -> (white-noise condition whose drafts and comparison are reused, k, train t0 range)
CONDS = {
    'L-vm-none-k3': ('A-vm-none', 3, (.05, .4)),
    'L-vm-none-k3-wide': ('A-vm-none', 3, (.05, 1.)),
    'L-vm-none-k5': ('A-vm-none', 5, (.05, .4)),
    'L-vm-08-k3': ('A-vm-08', 3, (.05, .4)),
    'L-raw-k3': ('A-raw', 3, (.05, .4)),
}


def dct_basis(n=HORIZON):
    """Orthonormal DCT-II basis as columns: B[:, j] is frequency j over time."""
    return dct(np.eye(n), type=2, norm='ortho', axis=0).T


def low_projector(k, n=HORIZON):
    """[n, n] map from white noise to low-frequency noise with mean element variance 1."""
    b = dct_basis(n)[:, :k]
    return np.sqrt(n / k) * b @ b.T


def project(noise, k):
    """Low-frequency version of [..., n, D] noise along the time axis."""
    p = low_projector(k, noise.shape[-2]).astype(np.float32)
    return np.einsum('ts,...sd->...td', p, np.asarray(noise, np.float32))


def _load_cd():
    path = HERE / 'compressed_denoise.py'
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location('compressed_denoise_lowfreq', path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def _link_drafts(name):
    """Expose the white condition's draft files under this condition's name (symlinks, no copy)."""
    white = CONDS[name][0]
    dst = OUT / 'drafts'
    dst.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'dev', 'eval'):
        src = WHITE / 'drafts' / f'{white.replace("-", "_")}_{split}.npz'
        link = dst / f'{name.replace("-", "_")}_{split}.npz'
        if not link.exists():
            link.symlink_to(src)
        elif link.resolve() != src.resolve():
            raise FileExistsError(f'{link} points elsewhere')


def install(cd, name):
    """Point compressed_denoise at this condition: output, noise, t0 range, Eval grid."""
    white, k, t0_range = CONDS[name]
    proj = low_projector(k).astype(np.float32)
    cd.OUT = OUT
    cd.CONDS = tuple(cd.CONDS) + (name,)
    cd.T0_EVAL = T0_EVAL
    cd._t0_range = lambda condition: t0_range
    original_predict = cd._predict_action

    def predict(model, obs, drafts, noise, t0, *, batch=cd.BATCH):
        return original_predict(model, obs, drafts, np.einsum('ts,nsd->ntd', proj, np.asarray(noise, np.float32)),
                                t0, batch=batch)

    def make_step(filt, tx, rng_t0):
        import flax.nnx as nnx
        import jax
        import jax.numpy as jnp
        import optax
        lo, hi = rng_t0
        p = jnp.asarray(proj)

        @nnx.jit
        def step(model, opt, obs, actions, draft, rng):
            r0, r1, r2 = jax.random.split(rng, 3)
            t0 = jax.random.uniform(r0, (len(actions),), minval=lo, maxval=hi)
            u = jax.random.uniform(r1, (len(actions),), minval=.01, maxval=1.)
            eps = jnp.einsum('ts,nsd->ntd', p, jax.random.normal(r2, actions.shape))
            source = (1 - t0[:, None, None]) * draft + t0[:, None, None] * eps
            x = (1 - u[:, None, None]) * actions + u[:, None, None] * source
            target = (source - actions) / t0[:, None, None]

            def loss_fn(current):
                v = current.velocity(obs, x, t0 * u)
                return jnp.mean(jnp.square(v[:, :, :7] - target[:, :, :7]))
            loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, filt))(model)
            state = nnx.state(model, filt)
            updates, opt = tx.update(grads, opt, state)
            nnx.update(model, optax.apply_updates(state, updates))
            return loss, opt
        return step

    cd._predict_action = predict
    cd._make_action_step = make_step
    _link_drafts(name)
    return cd


def energy():
    """Share of draft-error energy in the lowest k DCT frequencies (train: own task excluded)."""
    cd = _load_cd()
    base = cd._module(cd.PI05 / 'run.py', 'lowfreq_energy_base')
    rows = base.load_rows(cd.RUN)
    unseen = {p.stem: np.load(p, mmap_mode='r', allow_pickle=False) for p in (cd.UNSEEN / 'unseen_rows').glob('*.npy')}
    basis = dct_basis()
    out = {}
    for white in ('A-raw', 'A-vm-none', 'A-vm-08'):
        out[white] = {}
        for split, data in (('train', rows), ('eval', unseen)):
            z = np.load(WHITE / 'drafts' / f'{white.replace("-", "_")}_{split}.npz', allow_pickle=False)
            ids = np.asarray(z['row_ids'], np.int64)
            err = np.asarray(z['draft'][:, :, :7], np.float64) - np.asarray(data['actions'][ids, :, :7], np.float64)
            coef = np.einsum('tj,ntd->njd', basis, err) ** 2       # [rows, freq, axis]
            total = coef.sum(1)
            rec = {'rows': int(len(ids))}
            for k in (1, 2, 3, 5):
                rec[f'k{k}'] = {'all7': float(coef[:, :k].sum() / coef.sum()),
                                'per_axis': (coef[:, :k].sum((0, 1)) / total.sum(0)).tolist(),
                                'white_noise': k / HORIZON}
            hard = np.asarray(z['hard'], bool)
            rec['hard_k3_all7'] = float(coef[hard, :3].sum() / coef[hard].sum()) if hard.any() else None
            out[white][split] = rec
    _write_json(OUT / 'energy.json', out)
    return out


def train(name, threads=8):
    cd = install(_load_cd(), name)
    meta = cd.train_condition(name, threads=threads)
    meta = {**meta, 'noise': {'kind': 'low-frequency DCT', 'k': CONDS[name][1], 'scale': f'sqrt(10/{CONDS[name][1]})',
                              'train_t0': list(CONDS[name][2]), 'drafts_from': CONDS[name][0]}}
    _write_json(OUT / 'experts' / name / 'trained.json', meta)
    return meta


def evaluate(name, threads=8):
    cd = install(_load_cd(), name)
    if not (OUT / 'predictions' / name / 'result.json').exists():
        cd.evaluate_condition(name, threads=threads)
    return compare(name)


def compare(name):
    """Eval at the Dev-selected checkpoint/t0, against the draft and the white-noise A on the same rows."""
    cd = _load_cd()
    white = CONDS[name][0]
    mine = np.load(OUT / 'predictions' / name / 'eval_rows.npz', allow_pickle=False)
    theirs = np.load(WHITE / 'predictions' / white / 'eval_rows.npz', allow_pickle=False)
    if not np.array_equal(mine['row_ids'], theirs['row_ids']):
        raise AssertionError('Eval rows differ from the white-noise condition')
    for field in ('target', 'draft', 'hard', 'task', 'episode'):
        if field not in mine.files or field not in theirs.files:
            raise AssertionError(f'missing paired comparison field: {field}')
        if not np.array_equal(mine[field], theirs[field]):
            raise AssertionError(f'{field} differs from the white-noise condition')
    t_mine = json.loads((OUT / 'experts' / name / 'trained.json').read_text())['best']['t0']
    t_white = json.loads((WHITE / 'experts' / white / 'trained.json').read_text())['best']['t0']
    key = cd._episode_keys(mine['task'], mine['episode'])
    hard = np.asarray(mine['hard'], bool)
    draft = np.mean((mine['draft'] - mine['target']) ** 2, axis=(1, 2))
    e_mine = mine[f'err_t{t_mine}']
    e_white = theirs[f'err_t{t_white}']
    res = {'condition': name, 'white': white, 'dev_t0': t_mine, 'white_dev_t0': t_white,
           'draft': [float(draft.mean()), float(draft[hard].mean())],
           'lowfreq': [float(e_mine.mean()), float(e_mine[hard].mean())],
           'white_noise': [float(e_white.mean()), float(e_white[hard].mean())],
           'lowfreq_minus_draft': cd._bootstrap_all_and_hard(e_mine, draft, key, hard),
           'lowfreq_minus_white': cd._bootstrap_all_and_hard(e_mine, e_white, key, hard),
           'lowfreq_components': {
               'all': {axis: float(np.mean(mine[f'err_{field}_t{t_mine}']))
                       for axis, field in (('xyz', 'xyz'), ('rotation', 'rotation'), ('gripper', 'grip'))},
               'hard': {axis: float(np.mean(mine[f'err_{field}_t{t_mine}'][hard]))
                        for axis, field in (('xyz', 'xyz'), ('rotation', 'rotation'), ('gripper', 'grip'))},
           },
           'all_t0': {str(t): [float(mine[f'err_t{t}'].mean()), float(mine[f'err_t{t}'][hard].mean())] for t in T0_EVAL}}
    if 'oracle01_pred_selected_t0' in mine.files:
        legacy = np.mean((mine['oracle01_pred_selected_t0'] - mine['target']) ** 2, axis=(1, 2))
        res['oracle01_expert_lowfreq_noise_reference'] = {
            'mse': float(legacy.mean()), 'hard_mse': float(legacy[hard].mean()),
            'note': 'legacy oracle-01 Expert evaluated with low-frequency DCT noise by this process; not its original white-noise result',
        }
    _write_json(OUT / 'predictions' / name / 'compare.json', res)
    return res


def report():
    lines = ['# 低周波ノイズでのデノイズ', '',
             '未学習 Eval 619 行（難しい行 134）。チェックポイントと t0 は開発用2タスクだけで選んだ値。',
             '区間はエピソードの bootstrap 95%。差は負が改善。', '',
             '| 条件 | 下書き | 白色ノイズ A | 低周波 L | t0（白色/低周波） | L−下書き 全 / 難 | L−白色 全 / 難 | L xyz / 回転 / gripper（全） |',
             '|---|---|---|---|---|---|---|---|']
    done = {}
    for name in CONDS:
        p = OUT / 'predictions' / name / 'compare.json'
        if not p.exists():
            lines.append(f'| {name} | 未完了 | | | | | | |')
            continue
        r = json.loads(p.read_text())
        done[name] = r
        f = lambda v: f'{v[0]:.4f} / {v[1]:.4f}'
        ci = lambda d: f"{d['mean']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]" if 'ci95' in d else json.dumps(d)
        low_draft = f"{ci(r['lowfreq_minus_draft']['all'])} / {ci(r['lowfreq_minus_draft']['hard'])}"
        low_white = f"{ci(r['lowfreq_minus_white']['all'])} / {ci(r['lowfreq_minus_white']['hard'])}"
        comp = r['lowfreq_components']['all']
        components = f"{comp['xyz']:.4f} / {comp['rotation']:.4f} / {comp['gripper']:.4f}"
        lines.append(f"| {name} | {f(r['draft'])} | {f(r['white_noise'])} | {f(r['lowfreq'])} | {r['white_dev_t0']} / {r['dev_t0']} | "
                     f"{low_draft} | {low_white} | {components} |")
    lines += ['', '旧 oracle-01 Expert の参考値は、この評価プロセスで低周波ノイズを与えた値であり、元の白色ノイズ評価値ではない。',
              '各条件の `compare.json` にこの参考値と xyz・回転・gripper の全行／難行成分を保存する。',
              '', '## 全 t0（Eval の選択には使わない）', '', '| 条件 | ' + ' | '.join(str(t) for t in T0_EVAL) + ' |',
              '|---|' + '---|' * len(T0_EVAL)]
    for name, r in done.items():
        lines.append(f'| {name} | ' + ' | '.join(f"{r['all_t0'][str(t)][0]:.4f}" for t in T0_EVAL) + ' |')
    e = OUT / 'energy.json'
    if e.exists():
        en = json.loads(e.read_text())
        lines += ['', '## 下書きの誤差のうち低い周波数に入るエネルギーの割合（白色ノイズなら k/10）', '',
                  '| 下書き | 分割 | k=1 | k=2 | k=3 | k=5 | 難しい行 k=3 |', '|---|---|---|---|---|---|---|']
        for white, d in en.items():
            for split, rec in d.items():
                hk = rec['hard_k3_all7']
                lines.append(f"| {white} | {split} | " + ' | '.join(f"{rec[f'k{k}']['all7']:.3f}" for k in (1, 2, 3, 5))
                             + (f" | {hk:.3f} |" if hk is not None else ' | — |'))
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n')
    _write_json(OUT / 'results.json', done)
    return done


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['energy', 'train', 'evaluate', 'report'])
    p.add_argument('--condition', choices=list(CONDS))
    p.add_argument('--threads', type=int, default=int(os.environ.get('OMP_NUM_THREADS', 8)))
    a = p.parse_args()
    if a.stage == 'energy':
        r = energy()
    elif a.stage == 'report':
        r = report()
    elif a.stage == 'train':
        r = train(a.condition, a.threads)
    else:
        r = evaluate(a.condition, a.threads)
    print(json.dumps(r, ensure_ascii=False)[:4000])


if __name__ == '__main__':
    main()
