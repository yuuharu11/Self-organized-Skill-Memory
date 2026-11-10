"""Step 2: refine memory drafts (chosen shape × θ, G1 gripper) with π0.5 Expert bridge denoising.

Drafts come from theta_flow.evaluate (eval_drafts.npz). Experts: the public pi05_libero
(no fine-tuning) and the compressed-denoise A-vm-none / A-raw experts (their Dev-selected t0
is the pre-declared main setting; the full t0 grid is reported as a diagnostic).
Main draft choice is fixed before looking at Eval: shape = P6_tok_hist, θ = θ-flow variant
with the best LIBERO-40 holdout score. Unseen targets are used only for scoring.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import os
STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', Path(__file__).resolve().parents[5] / '.skillnet-state')).expanduser()

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compressed_denoise as cd  # noqa: E402

ROOT = (STATE_DIR / 'outputs/memory-pilot/unseen-libero')
FLOW = ROOT / 'theta-flow-01'
OUT = ROOT / 'expert-refine-01'
T0_GRID = (.05, .1, .2, .3, .4, .6)
EXPERTS = ('base', 'A-vm-none', 'A-raw')


def chosen_drafts():
    best = min(('flow', 'bridge', 'regress'),
               key=lambda v: json.loads((FLOW / 'models' / v / 'trained.json').read_text())['best']['holdout_six_axis'])
    keys = {f'main:{best}': f'{best}|seed|P6_tok_hist',
            'lora': 'lora|-|P6_tok_hist', 'theta_pi05': 'theta_pi05|-|P6_tok_hist',
            'upper:theta_star_oracle_shape': 'theta_star|-|oracle_shape'}
    return best, keys


def main():
    import jax, jax.numpy as jnp, flax.nnx as nnx, orbax.checkpoint as ocp
    from openpi.shared.nnx_utils import PathRegex
    OUT.mkdir(parents=True, exist_ok=True)
    best, keys = chosen_drafts()
    d = np.load(FLOW / 'eval_drafts.npz')
    base = cd._module(cd.PI05 / 'run.py', 'refine_base'); rows = base.load_rows(cd.RUN)
    unseen = {p.stem: np.load(p, mmap_mode='r', allow_pickle=False) for p in (cd.UNSEEN / 'unseen_rows').glob('*.npy')}
    ids = np.asarray(d['row_ids'], np.int64)
    obs = base.obs_batch(unseen, ids)
    eps = np.stack([np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(cd.NOISE_SEED), int(i)), (10, 32)))
                    for i in len(rows['actions']) + ids])
    # scorer-only
    target = np.asarray(unseen['actions'][ids], np.float32)[:, :, :7]
    k1 = np.load(ROOT / 'key-query-01' / 'production' / 'evaluation' / 'rows.npz')
    assert np.array_equal(k1['row_ids'], ids)
    hard = k1['hard'].astype(bool); episodes = k1['task'].astype(np.int64) * 100000 + k1['episode']
    normal = ((k1['query_pi05'][:, :, :7] - target) ** 2).mean((1, 2))
    res = {'main_theta_variant': best, 'drafts': keys, 'normal_pi05': [float(normal.mean()), float(normal[hard].mean())], 'runs': {}}
    preds = {}
    for expert in EXPERTS:
        model, _ = base.load_policy(cd.CHECKPOINT)
        sel_t0 = None
        if expert != 'base':
            tj = json.loads((cd.ROOT / 'compressed-denoise-01' / 'production' / 'experts' / expert / 'trained.json').read_text())
            filt = nnx.All(nnx.Param, PathRegex(r'.*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*'))
            with ocp.PyTreeCheckpointer() as c:
                part = c.restore(Path(tj['best']['path']))
            st = nnx.state(model, filt); st.replace_by_pure_dict(jax.tree.map(jnp.asarray, part)); nnx.update(model, st)
            sel_t0 = float(tj['best']['t0'])
        for label, key in keys.items():
            draft7 = d[key]; draft = np.zeros((len(draft7), 10, 32), np.float32); draft[:, :, :7] = draft7
            pre = ((draft7 - target) ** 2).mean((1, 2))
            res['runs'][f'{expert}|{label}|draft'] = {'all': float(pre.mean()), 'hard': float(pre[hard].mean())}
            for t0 in T0_GRID:
                p = cd._predict_action(model, obs, draft, eps, t0)[:, :, :7]
                e = ((p - target) ** 2).mean((1, 2))
                tag = f'{expert}|{label}|t0={t0}'
                rec = {'all': float(e.mean()), 'hard': float(e[hard].mean()),
                       'minus_draft': bootstrap(e, pre, episodes), 'minus_normal_pi05': bootstrap(e, normal, episodes),
                       'predeclared': sel_t0 is not None and abs(t0 - sel_t0) < 1e-9}
                res['runs'][tag] = rec; preds[tag] = p
                print(json.dumps({tag: [rec['all'], rec['hard']]}), flush=True)
                (OUT / 'results.json').write_text(json.dumps(res, indent=1))
        del model
    np.savez(OUT / 'predictions.npz', **preds, row_ids=ids)
    write_report(res)


def bootstrap(a, b, episodes, reps=5000, seed=4101):
    keys = np.unique(episodes); g = [np.flatnonzero(episodes == k) for k in keys]; rng = np.random.default_rng(seed); out = []
    for _ in range(reps):
        ix = np.concatenate([g[i] for i in rng.integers(len(g), size=len(g))]); out.append((a[ix] - b[ix]).mean())
    return [float((a - b).mean()), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def write_report(res):
    L = ['# Step 2: Expert refinement of memory drafts (unseen LIBERO-90 eval)', '',
         f"Main θ variant (chosen on LIBERO-40 holdout): {res['main_theta_variant']}. Normal π0.5 {res['normal_pi05'][0]:.4f} / {res['normal_pi05'][1]:.4f}.",
         'Pre-declared t0 = expert’s Dev-selected t0 (A-*: .05). Other t0 values and the base model are diagnostics.', '',
         '| expert | draft | t0 | all | hard | − draft [95%] | − normal π0.5 [95%] |', '|---|---|---|---:|---:|---:|---:|']
    for k, v in res['runs'].items():
        e, lab, t = k.split('|')
        if t == 'draft':
            L.append(f"| {e} | {lab} | draft | {v['all']:.4f} | {v['hard']:.4f} | — | — |"); continue
        a = v['minus_draft']; b = v['minus_normal_pi05']; star = ' *' if v['predeclared'] else ''
        L.append(f"| {e} | {lab} | {t[3:]}{star} | {v['all']:.4f} | {v['hard']:.4f} | {a[0]:+.4f} [{a[1]:+.4f}, {a[2]:+.4f}] | {b[0]:+.4f} [{b[1]:+.4f}, {b[2]:+.4f}] |")
    (OUT / 'report.md').write_text('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
