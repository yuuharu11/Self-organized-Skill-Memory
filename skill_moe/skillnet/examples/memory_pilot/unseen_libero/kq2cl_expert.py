#!/usr/bin/env python3
"""KQ2-CL stage E: key-agnostic skill Expert retrained on the draft law of the KQ2 (iii) retriever.

Same as EX3 (expert_ex3.py, read-only import) except:
  * draft rank law = mixture of the measured (iii) laws (kq2cl_retrieve measure):
      LIBERO-40 eh rows, task-fold (iii) query, own task masked (oofT)   weight w_mem
      Dev v2 20 Hz, production (iii) query                               weight w_dev
    + near_fraction truncated-geometric over the top 50 (p=.16) + uniform_fraction uniform over all
    eligible units ("a little random"). Ranks are over own-task-masked units by arm6 action MSE.
  * loss on the arm 6 channels only (the gripper comes from the separate decoder).
  * training rows, optimizer, filter, t0 rule (TRAIN_T0MAP on the true draft error, x U(.9,1.1)^2),
    draft gripper (hold-last executed) are EX3's (expert-ex3-01/prepare, read-only).

  prepare   law.npz / prepare.json (law preview vs the measured laws)
  train     --name K1 (5000 updates, batch 8, checkpoint every 500)
  select    Dev v2 20 Hz: (iii)+(i) top-1 drafts, t0 = clip(c*sqrt(true err), .05, .6), c in SCALES;
            every checkpoint + EX3 G2:4000 -> select.json (Eval unused)
  curves    per-row Expert arm6 over the t0 grid on dev2 / learned / unseen20 for the (iii) and (i)
            drafts and the oracle draft; selected Expert and EX3. Also stores the raw Expert gripper.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

import kq2cl_common as C

EX3_OUT = C.ROOT / 'expert-ex3-01'
SCALES = (.5, .75, 1., 1.5, 2., 3.)
NOISE_OFFSET = {'learned': 0, 'unseen20': 50_000_000, 'dev2': 60_000_000}
QUERIES = ('iii', 'i')


def sample_ranks(rng, n, pools, weights, n_eligible, *, near_fraction=.15, uniform_fraction=.05,
                 near_p=.16, near_k=50):
    u = rng.random(n)
    near = u < near_fraction
    uni = (u >= near_fraction) & (u < near_fraction + uniform_fraction)
    w = np.asarray(weights, np.float64); w /= w.sum()
    which = rng.choice(len(pools), size=n, p=w)
    out = np.empty(n, np.int64)
    for j, p in enumerate(pools):
        m = which == j
        out[m] = p[rng.integers(0, len(p), size=int(m.sum()))]
    probs = near_p * (1 - near_p) ** np.arange(near_k); probs /= probs.sum()
    out[near] = 1 + rng.choice(near_k, size=int(near.sum()), p=probs)
    ne = np.asarray(n_eligible, np.int64)
    out[uni] = 1 + (rng.random(int(uni.sum())) * ne[uni]).astype(np.int64)
    return np.minimum(np.maximum(out, 1), ne), near, uni


def law_spec():
    return C.read_json(C.OUT / 'expert/law.json')


def prepare(w_mem: float, w_dev: float, near_fraction: float, uniform_fraction: float) -> dict:
    dest = C.OUT / 'expert'; dest.mkdir(parents=True, exist_ok=True)
    R = C.OUT / 'retrieval'
    mem = np.load(R / 'iii__eh_oofT.npz')['rank'].astype(np.int64)
    dev = np.load(R / 'iii__dev2.npz')['rank'].astype(np.int64)
    spec = {'pools': ['iii eh oofT (fold query, own task masked)', 'iii dev2 (production query)'],
            'weights': [w_mem, w_dev], 'near_fraction': near_fraction, 'uniform_fraction': uniform_fraction,
            'near_rule': 'truncated geometric p=.16 over top-50', 'uniform_rule': 'uniform over all eligible (own task masked) units'}
    np.savez(dest / 'law.npz', mem=mem, dev=dev)
    C.write_json(dest / 'law.json', spec)
    prep = dict(np.load(EX3_OUT / 'prepare/rows.npz')); order = np.load(EX3_OUT / 'prepare/order.npy', mmap_mode='r')
    units = np.load(C.ROOT / 'key-motion-01/final/data/bank_actions.npy').astype(np.float32)
    rng = np.random.default_rng(1); n = len(prep['row_ids'])
    r, near, uni = sample_ranks(rng, n, [mem, dev], [w_mem, w_dev], prep['n_eligible'],
                                near_fraction=near_fraction, uniform_fraction=uniform_fraction)
    picked = np.asarray(order[np.arange(n), r - 1], np.int64)
    pre = C.arm6(units[picked], prep['target'])
    from scipy.stats import ks_2samp
    dev_err = np.load(R / 'iii__dev2.npz')['err']; mem_err = np.load(R / 'iii__eh_oofT.npz')['err']
    res = {'law': spec, 'rows': int(n), 'preview_rank': C.quantiles(r), 'preview_arm6': C.quantiles(pre),
           'measured_rank_mem_oofT': C.quantiles(mem), 'measured_rank_dev2': C.quantiles(dev),
           'measured_arm6_dev2': C.quantiles(dev_err), 'measured_arm6_mem_oofT': C.quantiles(mem_err),
           'ks_preview_vs_dev2': {'log_rank': float(ks_2samp(np.log(r), np.log(dev)).statistic),
                                  'arm6': float(ks_2samp(pre, dev_err).statistic)},
           'preview_t0_train': C.quantiles(C.TRAIN_T0MAP(pre)),
           'preview_t0_at_cap': float((C.TRAIN_T0MAP(pre) >= .6).mean()),
           'near_share': float(near.mean()), 'uniform_share': float(uni.mean())}
    C.write_json(dest / 'prepare.json', res)
    C.log('expert_prepare', preview_rank_med=res['preview_rank']['0.5'], preview_arm6_med=res['preview_arm6']['0.5'],
          ks=res['ks_preview_vs_dev2'])
    return res


def train(name: str, *, updates: int = 5000, batch: int = 8, microbatch: int = 8, save_every: int = 500,
          seed: int = 20261119) -> None:
    import jax, jax.numpy as jnp, flax.nnx as nnx, optax, orbax.checkpoint as ocp
    import expert_ex3 as ex3
    exp3, base = ex3._base()
    tdir = C.OUT / 'expert' / f'train-{name}'; ck = tdir / 'checkpoints'; ck.mkdir(parents=True, exist_ok=True)
    spec = law_spec(); law = np.load(C.OUT / 'expert/law.npz'); pools = [law['mem'], law['dev']]
    prep = dict(np.load(EX3_OUT / 'prepare/rows.npz')); order = np.load(EX3_OUT / 'prepare/order.npy', mmap_mode='r')
    units = np.load(C.ROOT / 'key-motion-01/final/data/bank_actions.npy').astype(np.float32)
    rows40 = base.load_rows(C.RUN40)
    ids = prep['row_ids']; target = prep['target']; n_el = prep['n_eligible']
    hl = C.hold_last_gripper(prep['last_grip'], prep['last_valid'])
    model, _ = base.load_policy(C.PI05)
    filt = ex3._filter()
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(2.5e-5, b1=.9, b2=.95, weight_decay=1e-5))
    state = nnx.state(model, filt); opt_state = optimizer.init(state)
    meta_path = tdir / 'train.json'
    meta = C.read_json(meta_path) or {
        'status': 'running', 'name': name, 'updates': updates, 'batch': batch, 'rows': int(len(ids)), 'law': spec,
        'draft_gripper': 'hold-last executed gripper', 'loss': 'mean over arm6 velocity sq (gripper weight 0)',
        't0_rule_train': f"TRAIN_T0MAP {C.TRAIN_T0MAP.as_dict()} on true arm6 x U(.9,1.1)^2",
        'optimizer': 'AdamW 2.5e-5 b(.9,.95) wd1e-5 clip1 (EX2 A / EX3)', 'param_filter': ex3.PARAM_REGEX, 'seed': seed, 'history': []}
    start = 0
    done = sorted(int(p.name) for p in ck.iterdir() if p.name.isdigit() and (ck / f'{p.name}.opt').exists())
    if done:
        start = done[-1]
        with ocp.PyTreeCheckpointer() as c:
            restored = c.restore(ck / f'{start:06d}')
            st = nnx.state(model, filt); st.replace_by_pure_dict(jax.tree.map(jnp.asarray, restored)); nnx.update(model, st)
            opt_state = c.restore(ck / f'{start:06d}.opt', item=opt_state)
        C.log('expert_train_resume', name=name, step=start)
    wv = jnp.asarray([1.] * 6 + [0.], jnp.float32)

    @nnx.jit
    def grad_step(current, obs, action, draft, t0, u, eps):
        source = (1. - t0[:, None, None]) * draft + t0[:, None, None] * eps
        x = (1. - u[:, None, None]) * action + u[:, None, None] * source
        target_v = (source - action) / jnp.maximum(t0[:, None, None], 1e-3)

        def loss_fn(m):
            pred = m.velocity(obs, x, t0 * u)
            sq = jnp.square(pred[:, :, :7] - target_v[:, :, :7])
            return jnp.mean(jnp.sum(sq * wv, -1)) / jnp.sum(wv)
        return nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, filt))(current)

    np_rng = np.random.default_rng(seed + start)
    rng = jax.random.fold_in(jax.random.key(seed), start)
    losses = []; t_start = time.time()
    for step in range(start + 1, updates + 1):
        bi = np_rng.integers(0, len(ids), size=batch)
        r, _, _ = sample_ranks(np_rng, batch, pools, spec['weights'], n_el[bi], near_fraction=spec['near_fraction'],
                               uniform_fraction=spec['uniform_fraction'])
        picked = np.asarray(order[bi, r - 1], np.int64)
        d = np.zeros((batch, 10, 32), np.float32); d[..., :6] = units[picked, :, :6]; d[..., 6] = hl[bi]
        a = np.zeros((batch, 10, 32), np.float32); a[..., :7] = target[bi]
        pre = C.arm6(d, a)
        t0 = jnp.asarray(C.TRAIN_T0MAP(pre * np_rng.uniform(.9, 1.1, size=batch) ** 2))
        rng, uk, ek = jax.random.split(rng, 3)
        u = jax.random.uniform(uk, (batch,), minval=.01, maxval=1.); eps = jax.random.normal(ek, (batch, 10, 32))
        grads = None; lv = 0.
        for lo in range(0, batch, microbatch):
            mb = np.arange(lo, min(lo + microbatch, batch))
            obs = base.obs_batch(rows40, ids[bi[mb]])
            val, g = grad_step(model, obs, jnp.asarray(a[mb]), jnp.asarray(d[mb]), t0[mb], u[mb], eps[mb])
            w = len(mb) / batch; lv += float(val) * w
            g = jax.tree.map(lambda x: x * w, g); grads = g if grads is None else jax.tree.map(jnp.add, grads, g)
        st = nnx.state(model, filt)
        upd, opt_state = optimizer.update(grads, opt_state, st); nnx.update(model, optax.apply_updates(st, upd))
        losses.append(lv)
        if step % 100 == 0:
            C.log('expert_train_step', name=name, step=step, loss=float(np.mean(losses[-100:])), sec=round(time.time() - t_start))
        if step % save_every == 0 or step == updates:
            exp3._save_state(model, filt, ck / f'{step:06d}', ocp)
            with ocp.PyTreeCheckpointer() as c:
                c.save(ck / f'{step:06d}.opt', opt_state, force=True)
            meta['history'].append({'step': step, 'loss_last': float(np.mean(losses[-save_every:])), 'checkpoint': str(ck / f'{step:06d}')})
            C.write_json(meta_path, meta)
    meta['status'] = 'complete'; C.write_json(meta_path, meta)
    C.log('expert_train_done', name=name)


# ---------------------------------------------------------------------------
# evaluation helpers
# ---------------------------------------------------------------------------

def split_inputs(split: str):
    """obs rows, fixed noise, targets, hold-last gripper, (iii)/(i)/oracle drafts (gripper = hold-last)."""
    import kq2cl_obs as O
    import skill_expert_refine as exp3
    rows = O.load_obs(split)
    n = len(rows['key'])
    noise = exp3._seed_noise(np.arange(n) + NOISE_OFFSET[split])
    D = C.KQ2 / 'data' / split
    tgt = np.load(D / 'actions.npy'); hist = np.load(D / 'history.npy'); hm = np.load(D / 'history_mask.npy')
    last, valid = hist[:, 9, 6], hm.any(1)
    hl = C.hold_last_gripper(last, valid)
    units = np.load(C.ROOT / 'key-motion-01/final/data/bank_actions.npy').astype(np.float32)
    drafts = {}
    for q in QUERIES:
        z = np.load(C.OUT / 'retrieval' / f'{q}__{split}.npz')
        d = units[z['ids'][:, 0]].copy(); d[..., 6] = hl; drafts[q] = d
    # oracle draft: best stored unit (own episode masked, as KQ2)
    o = C.oracle_ids(split)
    d = units[o].copy(); d[..., 6] = hl; drafts['oracle'] = d
    return rows, noise, tgt, last, valid, hl, drafts


def select() -> dict:
    import expert_ex3 as ex3
    dest = C.OUT / 'expert/select.json'
    prev = C.read_json(dest, {'scores': {}})
    ex = ex3.LiveExpert()
    rows, noise, tgt, last, valid, hl, drafts = split_inputs('dev2')
    obs = ex.base.obs_batch(rows, np.arange(len(tgt)))
    cks = {'EX3:G2:4000': C.EX3_CK}
    for tdir in sorted((C.OUT / 'expert').glob('train-*')):
        for p in sorted((tdir / 'checkpoints').iterdir()):
            if p.name.isdigit():
                cks[f'{tdir.name[6:]}:{int(p.name)}'] = p
    for label, path in cks.items():
        if label in prev['scores']:
            continue
        ex.load(path); rec = {'checkpoint': str(path), 'dev_arm6_by_scale': {}, 'per_query': {}}
        for c in SCALES:
            vals = []
            for q in QUERIES:
                d = drafts[q]
                t0 = C.T0Map(c, .05, .6)(C.arm6(d, tgt)); o = ex.bridge(obs, d, noise, t0); v = C.arm6(o, tgt); vals.append(v)
                rec['per_query'].setdefault(q, {})[str(c)] = float(v.mean())
            rec['dev_arm6_by_scale'][str(c)] = float(np.concatenate(vals).mean())
        prev['scores'][label] = rec; C.write_json(dest, prev)
        C.log('expert_select_scored', label=label, best=min(rec['dev_arm6_by_scale'].values()))
    mine = {k: v for k, v in prev['scores'].items() if not k.startswith('EX3')}
    best = min(mine, key=lambda k: min(mine[k]['dev_arm6_by_scale'].values()))
    sc = mine[best]['dev_arm6_by_scale']; c = min(sc, key=sc.get)
    e3 = prev['scores']['EX3:G2:4000']['dev_arm6_by_scale']; c3 = min(e3, key=e3.get)
    prev['selected'] = {'label': best, 'checkpoint': mine[best]['checkpoint'], 'scale': float(c), 'dev_arm6': sc[c]}
    prev['ex3_reference'] = {'checkpoint': str(C.EX3_CK), 'scale': float(c3), 'dev_arm6': e3[c3]}
    prev['draft_dev_arm6'] = {q: float(C.arm6(drafts[q], tgt).mean()) for q in QUERIES}
    prev['rule'] = 'Dev v2 20 Hz (667) x {(iii),(i)} production top-1 drafts, hold-last gripper; t0=clip(c*sqrt(true arm6),.05,.6)'
    C.write_json(dest, prev); C.log('expert_select_done', selected=prev['selected'], ex3=prev['ex3_reference'])
    return prev


def curves(splits: list[str], tags: list[str]) -> None:
    """curves/<split>__<tag>.npz (one file per Expert so runs can overlap); C.load_curves merges."""
    import expert_ex3 as ex3
    sel = C.read_json(C.OUT / 'expert/select.json') if 'K' in tags else None
    ex = ex3.LiveExpert()
    for split in splits:
        rows, noise, tgt, last, valid, hl, drafts = split_inputs(split)
        obs = ex.base.obs_batch(rows, np.arange(len(tgt)))
        for tag in tags:
            ck = sel['selected']['checkpoint'] if tag == 'K' else str(C.EX3_CK)
            dest = C.OUT / 'curves' / f'{split}__{tag}.npz'; dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = C.OUT / 'curves' / f'{split}__{tag}.tmp.npz'
            res = dict(np.load(dest)) if dest.exists() else {}
            res['checkpoint'] = np.asarray(str(ck))
            for q, d in drafts.items():
                res[f'draft__{q}'] = C.arm6(d, tgt).astype(np.float32)
                if f'{tag}__{q}__curve' in res:
                    continue
                ex.load(Path(ck))
                cur = np.zeros((len(d), len(C.T0_GRID)), np.float32); grip = np.zeros((len(d), len(C.T0_GRID), 10), np.float16)
                for j, t in enumerate(C.T0_GRID):
                    o = ex.bridge(obs, d, noise, np.full(len(d), t, np.float32))
                    cur[:, j] = C.arm6(o, tgt); grip[:, j] = o[..., 6]
                res[f'{tag}__{q}__curve'] = cur; res[f'{tag}__{q}__grip'] = grip
                np.savez_compressed(tmp, **res); os.replace(tmp, dest)
                C.log('curve', split=split, tag=tag, q=q, by_t0=[round(float(x), 4) for x in cur.mean(0)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=('prepare', 'train', 'select', 'curves'))
    ap.add_argument('--name', default='K1'); ap.add_argument('--updates', type=int, default=5000)
    ap.add_argument('--w-mem', type=float, default=.5); ap.add_argument('--w-dev', type=float, default=.5)
    ap.add_argument('--near', type=float, default=.15); ap.add_argument('--uniform', type=float, default=.05)
    ap.add_argument('--splits', default='dev2,learned,unseen20'); ap.add_argument('--tags', default='K,EX3')
    a = ap.parse_args()
    C.log('start', stage='expert.' + a.cmd)
    if a.cmd == 'prepare':
        prepare(a.w_mem, a.w_dev, a.near, a.uniform)
    elif a.cmd == 'train':
        train(a.name, updates=a.updates)
    elif a.cmd == 'select':
        select()
    else:
        curves(a.splits.split(','), a.tags.split(','))
    C.log('end', stage='expert.' + a.cmd)


if __name__ == '__main__':
    main()
