"""W2: does a WAM-style "predict the future" auxiliary loss make the type query generalize to unseen LIBERO-90?

Motivation: Fast-WAM (arXiv 2603.16666) reports that co-training on future video is what makes its features work
(10% without it).  Cheap analogue on cached features: the S1/I1 query network (soft teacher over LIBERO-40 type
errors, frac 0.1 = the value S1/I1 selected for every seed) gets an extra head on its 256-d query embedding that
predicts, for the same demo, the change from now to +10 and +20 frames of
    img    PCA-32 of the prefix layer-16 image means (I1 'mid16', pi0.5 frozen)
    state  the 8-d robot state
λ = 0 is exactly the I1 / S1 recipe (sanity check against I4's numbers).  Future rows exist only in LIBERO-40
train (stride 10), so the auxiliary signal never touches Dev / Eval.

Selection: epoch by LIBERO-40 holdout top-1 type error (as S1/I1); λ and aux target chosen on Dev (113 rows).
Eval: top-1 type error (s*-fitted S-shared, I4's main metric), all / hard, under the I4 history variants.
Output: image-query-01/wam-aux/.  CPU only.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
import image_query as iq  # noqa: E402

OUT = iq.OUT / 'wam-aux'
SEEDS = (4101, 4102, 4103)
FRAC = 0.1
EPOCHS = 20
HORIZONS = (10, 20)
NPCA = 32


def log(**rec):
    rec = {'t': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **rec}
    with open(OUT / 'progress.log', 'a') as f: f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(json.dumps(rec, ensure_ascii=False), flush=True)


def aux_targets():
    """[N_train, 2*(32+8)] future-minus-now targets (standardized) and a [N_train, 2] mask.  Cached."""
    p = OUT / 'aux_targets.npz'
    if p.exists():
        d = np.load(p); return d['y'], d['m'], d['cols']
    tr = dict(np.load(iq.S1 / 'prep' / 'W_inputs_train.npz')); n = len(tr['task'])
    f = iq.feature('train', 'mid16'); f = (f - f.mean(0)) / (f.std(0) + 1e-6)
    _, _, vt = np.linalg.svd(f[np.random.default_rng(0).choice(n, 8000, replace=False)], full_matrices=False)
    z = f @ vt[:NPCA].T; st = np.asarray(tr['state'], np.float32)
    key = {(int(t), int(e), int(fr)): i for i, (t, e, fr) in enumerate(zip(tr['task'], tr['episode'], tr['frame']))}
    ys, ms = [], []
    for h in HORIZONS:
        src = np.array([key.get((int(t), int(e), int(fr) + h), -1) for t, e, fr in zip(tr['task'], tr['episode'], tr['frame'])])
        ok = src >= 0; s = np.where(ok, src, 0)
        ys.append(np.concatenate([z[s] - z, st[s] - st], 1) * ok[:, None]); ms.append(ok)
    y = np.concatenate(ys, 1).astype(np.float32); m = np.stack(ms, 1)
    sd = np.concatenate([y[m[:, j], j * (NPCA + 8):(j + 1) * (NPCA + 8)].std(0) for j in range(len(HORIZONS))]) + 1e-6
    y = y / sd
    cols = np.array(['img'] * NPCA + ['state'] * 8) .repeat(1)
    cols = np.concatenate([cols] * len(HORIZONS))
    np.savez(p, y=y, m=m, cols=cols); return y, m, cols


def make_net(mode, in_dim, key_dim, aux_dim, d=256):
    import torch, torch.nn as nn
    F = nn.functional

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            if mode == 'P5':                                     # S1 P5 network
                self.query = nn.Sequential(nn.Linear(in_dim, 512), nn.GELU(), nn.Linear(512, d))
            else:                                                # I1 image+H network
                hid = 768 if in_dim > 300 else 512
                self.query = nn.Sequential(nn.Dropout(0.1 if in_dim > 300 else 0.), nn.Linear(in_dim, hid), nn.GELU(), nn.Linear(hid, d))
            self.key = nn.Sequential(nn.Linear(key_dim, 512), nn.GELU(), nn.Linear(512, d))
            self.log_scale = nn.Parameter(torch.tensor(float(np.log(10.))))
            self.aux = nn.Sequential(nn.GELU(), nn.Linear(d, 256), nn.GELU(), nn.Linear(256, aux_dim))

        def embed(self, x): return self.query(x)

        def forward(self, x, kin, return_emb=False):
            q = self.query(x); s = F.normalize(q, dim=-1) @ F.normalize(self.key(kin), dim=-1).T * self.log_scale.exp()
            return (s, q) if return_emb else s
    return Net()


def load(mode):
    import scale_only_query as so
    P = iq.S1 / 'prep'; pool = dict(np.load(P / 'W_pool.npz'))
    raw = {s: dict(np.load(P / f'W_inputs_{s}.npz')) for s in ('train', 'dev', 'eval')}
    if mode == 'P5': X = {s: so.query_inputs(raw[s], 'P5') for s in raw}
    else: X = iq.inputs(mode)[0]
    KI = so.key_inputs('W', pool, iq.COND)
    E = np.load(P / f'W_err_{iq.COND}_train.npz')['err']; Edev = np.load(P / f'W_err_{iq.COND}_dev.npz')['err']
    A = so.allowed_matrix(raw['train']['task'], pool['task'], pool['keep'], True)
    Adev = so.allowed_matrix(raw['dev']['task'], pool['task'], pool['keep'], False)
    hold = np.isin(raw['train']['task'], so.HOLDOUT)
    return dict(X=X, raw=raw, pool=pool, KI=KI, E=E, A=A, Edev=Edev, Adev=Adev, fit=np.flatnonzero(~hold), va=np.flatnonzero(hold))


def train_one(mode, aux, lam, seed, D):
    import torch, scale_only_query as so
    torch.manual_seed(seed)
    y, m, cols = aux_targets()
    sel = np.ones(len(cols), bool) if aux == 'both' else (cols == aux)
    msk = np.repeat(m, NPCA + 8, axis=1)[:, sel]; y = y[:, sel]
    X, KI, E, A, fit, va = D['X']['train'], D['KI'], D['E'], D['A'], D['fit'], D['va']
    tau = so.teacher_tau(E[fit], A[fit], FRAC)
    Xt = torch.as_tensor(X); Kt = torch.as_tensor(KI); Yt = torch.as_tensor(y); Mt = torch.as_tensor(msk.astype(np.float32))
    Et = torch.as_tensor(np.where(A, E, 1e3).astype(np.float32)); At = torch.as_tensor(A)
    Xd = torch.as_tensor(D['X']['dev']); Ad = torch.as_tensor(D['Adev'])
    net = make_net(mode, X.shape[1], KI.shape[1], int(sel.sum()))
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4); nb = (len(fit) + 255) // 256
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=EPOCHS * nb, pct_start=.05)
    best = {'holdout': (np.inf, None, 0), 'dev': (np.inf, None, 0)}; curve = []

    def pick(xs, am):
        with torch.no_grad(): return net(xs, Kt).masked_fill(~am, -1e9).argmax(-1).numpy()
    for ep in range(EPOCHS):
        net.train(); perm = np.random.default_rng(seed + ep).permutation(fit); la = []
        for i in range(0, len(perm), 256):
            b = torch.as_tensor(perm[i:i + 256])
            Eb = Et[b]; teacher = torch.softmax(-(Eb - Eb.min(1, keepdim=True).values) / tau, -1) * At[b]
            teacher = teacher / teacher.sum(1, keepdim=True)
            logit, q = net(Xt[b], Kt, return_emb=True); logit = logit.masked_fill(~At[b], -1e9)
            loss = -(teacher * torch.log_softmax(logit, -1)).sum(-1).mean()
            if lam > 0:
                mb = Mt[b]; la_ = (((net.aux(q) - Yt[b]) ** 2) * mb).sum() / mb.sum().clamp_min(1)
                loss = loss + lam * la_; la.append(float(la_))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        net.eval()
        vb = torch.as_tensor(va); pv = pick(Xt[vb], At[vb]); v = float(E[va, pv].mean())
        pd = pick(Xd, Ad); dv = float(D['Edev'][np.arange(len(pd)), pd].mean())
        curve.append({'epoch': ep + 1, 'holdout': v, 'dev': dv, 'aux_mse': float(np.mean(la)) if la else None})
        state = None
        for k, val in (('holdout', v), ('dev', dv)):
            if val < best[k][0]:
                state = state or {kk: t.detach().clone() for kk, t in net.state_dict().items()}
                best[k] = (val, state, ep + 1)
    return {'mode': mode, 'aux': aux, 'lam': lam, 'seed': seed, 'tau': tau, 'curve': curve, 'aux_dim': int(sel.sum()),
            'in_dim': int(X.shape[1]), **{k: {'value': best[k][0], 'epoch': best[k][2], 'state': best[k][1]} for k in best}}


def name_of(mode, aux, lam, seed): return f"{mode.replace('+', 'p')}_{aux}_l{lam:g}_s{seed}"


def train(configs, threads):
    import torch
    torch.set_num_threads(threads); (OUT / 'models').mkdir(parents=True, exist_ok=True)
    cache = {}
    for mode, aux, lam in configs:
        if mode not in cache: cache[mode] = load(mode)
        for seed in SEEDS:
            p = OUT / 'models' / f'{name_of(mode, aux, lam, seed)}.pt'
            if p.exists(): continue
            t0 = time.time(); r = train_one(mode, aux, lam, seed, cache[mode])
            tmp = p.with_suffix('.tmp'); torch.save(r, tmp); tmp.replace(p)
            log(stage='train', mode=mode, aux=aux, lam=lam, seed=seed, holdout=round(r['holdout']['value'], 5),
                h_ep=r['holdout']['epoch'], dev=round(r['dev']['value'], 5), aux_last=r['curve'][-1]['aux_mse'], sec=round(time.time() - t0, 1))


# ------------------------------------------------------------------ evaluation (I4 metric, CPU)
def history_variants(ev):
    """Same constructions and seed as image_query_recheck.py."""
    n = len(ev['task']); rng = np.random.default_rng(4101)
    perm = np.array([rng.choice(np.flatnonzero(ev['task'] != ev['task'][i])) for i in range(n)])
    H = {'true': (ev['hist'], np.arange(n)), 'swap': (ev['hist'][perm], np.arange(n))}
    r = ev['hist'].copy(); r[:, :, :3] *= -1; H['reverse'] = (r, np.arange(n)); H['zero'] = (np.zeros_like(ev['hist']), np.arange(n))
    key = {(int(t), int(e), int(f)): i for i, (t, e, f) in enumerate(zip(ev['task'], ev['episode'], ev['frame']))}
    for dt in (-20, -10, 10, 20):
        src = np.array([key.get((int(t), int(e), int(f) + dt), -1) for t, e, f in zip(ev['task'], ev['episode'], ev['frame'])])
        ok = np.flatnonzero(src >= 0); hh = ev['hist'].copy(); hh[ok] = ev['hist'][src[ok]]; H[f'phase{dt:+d}'] = (hh, ok)
    return H


def eval_errors(pool, ev):
    p = OUT / 'eval_E.npy'
    if p.exists(): return np.load(p)
    import scale_only_query as so
    cd, vs, kq = so._mods(); ua = np.load(kq.UNSEEN / 'actions.npy', mmap_mode='r')
    y = np.asarray(ua[ev['row'], :, :6], np.float32)                 # scoring only
    E = so.type_errors(vs.to_physical(y).astype(np.float32), pool['phys'], iq.COND, 'cpu'); np.save(p, E); return E


def evaluate(threads, which='holdout'):
    import torch, scale_only_query as so
    torch.set_num_threads(threads)
    P = iq.S1 / 'prep'; pool = dict(np.load(P / 'W_pool.npz')); ev = dict(np.load(P / 'W_inputs_eval.npz'))
    E = eval_errors(pool, ev); A = so.allowed_matrix(ev['task'], pool['task'], pool['keep'], False); hard = ev['hard'].astype(bool)
    KI = torch.as_tensor(so.key_inputs('W', pool, iq.COND)); H = history_variants(ev); n = len(E)
    groups = {}
    for p in sorted((OUT / 'models').glob('*.pt')):
        r = torch.load(p, weights_only=False); groups.setdefault((r['mode'], r['aux'], r['lam']), []).append(r)
    Xcache = {}; res = {}
    for (mode, aux, lam), rs in sorted(groups.items()):
        if mode not in Xcache: Xcache[mode] = (load(mode)['X']['eval'] if mode != 'P5' else so.query_inputs(ev, 'P5'))
        X0 = Xcache[mode]; row = {'n_seeds': len(rs), 'dev': float(np.mean([r['dev']['value'] for r in rs])),
                                  'dev_of_holdout_epoch': None, 'holdout': float(np.mean([r['holdout']['value'] for r in rs]))}
        dv_at = [r['curve'][r['holdout']['epoch'] - 1]['dev'] for r in rs]; row['dev_of_holdout_epoch'] = float(np.mean(dv_at))
        for hn, (hh, rows) in H.items():
            Xv = X0.copy(); Xv[:, -78:-8] = hh.reshape(n, -1); t1, t1h = [], []
            for r in rs:
                net = make_net(mode, r['in_dim'], KI.shape[1], r['aux_dim']); net.load_state_dict(r[which]['state']); net.eval()
                with torch.no_grad(): sc = net(torch.as_tensor(Xv), KI).numpy()
                pick = np.where(A, sc, -np.inf).argmax(1); te = E[np.arange(n), pick][rows]
                t1.append(te.mean()); t1h.append(te[hard[rows]].mean())
            row[hn] = [float(np.mean(t1)), float(np.mean(t1h)), float(np.min(t1)), float(np.max(t1))]
        res[f'{mode}|{aux}|{lam:g}'] = row
        log(stage='evaluate', key=f'{mode}|{aux}|{lam:g}', dev=round(row['dev_of_holdout_epoch'], 4),
            **{h: round(row[h][0], 4) for h in ('true', 'swap', 'reverse', 'zero', 'phase-10', 'phase+10')}, true_hard=round(row['true'][1], 4))
    orc = np.where(A, E, np.inf).min(1)
    res['reference'] = {'oracle': float(orc.mean()), 'oracle_hard': float(orc[hard].mean())}
    (OUT / f'results_{which}.json').write_text(json.dumps(res, indent=1) + '\n')
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=('targets', 'train', 'evaluate'))
    ap.add_argument('--configs', default='', help='mode:aux:lam,...'); ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--which', default='holdout')
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == 'targets':
        y, m, cols = aux_targets(); log(stage='targets', shape=list(y.shape), coverage=m.mean(0).round(3).tolist())
    elif a.stage == 'train':
        cf = [(c.split(':')[0], c.split(':')[1], float(c.split(':')[2])) for c in a.configs.split(',') if c]
        train(cf, a.threads)
    else: evaluate(a.threads, a.which)


if __name__ == '__main__':
    main()
