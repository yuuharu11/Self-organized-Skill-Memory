"""R1: is the readout of the VLM features (mean pooling) what loses the motion information?

Only the readout changes; everything else is I1 'M<l>+H' (history 10x7 + state 8 concatenated, S1 soft teacher over
LIBERO-40 type errors, frac 0.1, 20 epochs, epoch by LIBERO-40 holdout top-1 type error, 3 seeds).
Input tokens: prefix layer <l> image V tokens, 2x2-pooled, 8x8 per camera x 2 cameras, 256-d (pi0.5 frozen).
pi0.5 puts position into Q/K only (RoPE), so the V tokens carry no position: mean pooling removes "where" entirely.
Readouts
  mean     per-camera mean (= I1 M<l>+H, re-trained here)
  grid     1x1 conv 256->32 per token, keep the 2x8x8 layout, flatten (where is kept by the layout)
  attn     4 learned queries attend over tokens + learned 2D position / camera embedding
  hattn    4 queries made from history+state attend over the same tokens (K1 P1 idea, same recipe)
Eval: I4 main metric (top-1 type error, s*-fitted S-shared) all / hard, under the I4 history variants.
Output: image-query-01/spatial-readout/.  CPU only.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
import image_query as iq  # noqa: E402
import wam_aux_query as wa  # noqa: E402

OUT = iq.OUT / 'spatial-readout'
SEEDS = (4101, 4102, 4103)
FRAC, EPOCHS = 0.1, 20


def log(**rec):
    rec = {'t': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **rec}
    with open(OUT / 'progress.log', 'a') as f: f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(json.dumps(rec, ensure_ascii=False), flush=True)


def tokens(layer):
    """{split: [N,128,256] float32}, standardized per channel with train statistics."""
    li = iq.MID_LAYERS.index(layer); T = {}
    for s in ('train', 'dev', 'eval'):
        v = np.load(iq._pf(s) / f'{s}_mid_v.npy', mmap_mode='r'); out = []
        for i in range(0, len(v), 2048): out.append(np.asarray(v[i:i + 2048, li, :128], np.float32))
        T[s] = np.concatenate(out)
    mu = T['train'].mean((0, 1)); sd = T['train'].std((0, 1)) + 1e-6
    return {s: ((x - mu) / sd).astype(np.float32) for s, x in T.items()}


def hist_inputs():
    import scale_only_query as so
    P = iq.S1 / 'prep'; raw = {s: dict(np.load(P / f'W_inputs_{s}.npz')) for s in ('train', 'dev', 'eval')}
    return {s: so.query_inputs(raw[s], 'P5') for s in raw}, raw


def make_net(readout, key_dim, d=256, nq=4):
    import torch, torch.nn as nn
    F = nn.functional

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            if readout == 'mean': img_dim = 512
            elif readout == 'grid':
                self.red = nn.Linear(256, 32); img_dim = 128 * 32
            else:
                self.pos = nn.Parameter(torch.zeros(128, 256)); nn.init.normal_(self.pos, std=.02)
                self.kv = nn.Linear(256, 256); self.ln = nn.LayerNorm(256)
                if readout == 'attn': self.q = nn.Parameter(torch.randn(nq, 256) * .02)
                else: self.qh = nn.Sequential(nn.Linear(78, 256), nn.GELU(), nn.Linear(256, nq * 256))
                img_dim = nq * 256
            self.query = nn.Sequential(nn.Dropout(0.1), nn.Linear(img_dim + 78, 768), nn.GELU(), nn.Linear(768, d))
            self.key = nn.Sequential(nn.Linear(key_dim, 512), nn.GELU(), nn.Linear(512, d))
            self.log_scale = nn.Parameter(torch.tensor(float(np.log(10.))))

        def image(self, t, h):
            if readout == 'mean': return torch.cat([t[:, :64].mean(1), t[:, 64:].mean(1)], 1)
            if readout == 'grid': return self.red(t).flatten(1)
            k = self.ln(self.kv(t + self.pos))
            q = self.q.expand(len(t), -1, -1) if readout == 'attn' else self.qh(h).view(len(t), -1, 256)
            a = torch.softmax(q @ k.transpose(1, 2) / 16., -1)
            return (a @ t).flatten(1)

        def forward(self, t, h, kin):
            q = self.query(torch.cat([self.image(t, h), h], 1))
            return F.normalize(q, dim=-1) @ F.normalize(self.key(kin), dim=-1).T * self.log_scale.exp()
    return Net()


def train_one(readout, seed, D):
    import torch, scale_only_query as so
    torch.manual_seed(seed)
    T, H, KI, E, A, fit, va = D['T']['train'], D['H']['train'], D['KI'], D['E'], D['A'], D['fit'], D['va']
    tau = so.teacher_tau(E[fit], A[fit], FRAC)
    Tt, Ht, Kt = torch.as_tensor(T), torch.as_tensor(H), torch.as_tensor(KI)
    Et = torch.as_tensor(np.where(A, E, 1e3).astype(np.float32)); At = torch.as_tensor(A)
    Td, Hd, Ad = torch.as_tensor(D['T']['dev']), torch.as_tensor(D['H']['dev']), torch.as_tensor(D['Adev'])
    net = make_net(readout, KI.shape[1])
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4); nb = (len(fit) + 255) // 256
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=EPOCHS * nb, pct_start=.05)
    best = {'holdout': (np.inf, None, 0), 'dev': (np.inf, None, 0)}; curve = []

    def pick(t, h, am):
        out = []
        with torch.no_grad():
            for i in range(0, len(t), 2048): out.append(net(t[i:i + 2048], h[i:i + 2048], Kt).masked_fill(~am[i:i + 2048], -1e9).argmax(-1).numpy())
        return np.concatenate(out)
    for ep in range(EPOCHS):
        net.train(); perm = np.random.default_rng(seed + ep).permutation(fit)
        for i in range(0, len(perm), 256):
            b = torch.as_tensor(perm[i:i + 256])
            Eb = Et[b]; teacher = torch.softmax(-(Eb - Eb.min(1, keepdim=True).values) / tau, -1) * At[b]
            teacher = teacher / teacher.sum(1, keepdim=True)
            logit = net(Tt[b], Ht[b], Kt).masked_fill(~At[b], -1e9)
            loss = -(teacher * torch.log_softmax(logit, -1)).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        net.eval(); vb = torch.as_tensor(va)
        v = float(E[va, pick(Tt[vb], Ht[vb], At[vb])].mean())
        pd = pick(Td, Hd, Ad); dv = float(D['Edev'][np.arange(len(pd)), pd].mean())
        curve.append({'epoch': ep + 1, 'holdout': v, 'dev': dv})
        state = None
        for k, val in (('holdout', v), ('dev', dv)):
            if val < best[k][0]:
                state = state or {kk: t.detach().clone() for kk, t in net.state_dict().items()}
                best[k] = (val, state, ep + 1)
    return {'readout': readout, 'seed': seed, 'tau': tau, 'curve': curve,
            **{k: {'value': best[k][0], 'epoch': best[k][2], 'state': best[k][1]} for k in best}}


def load(layer):
    import scale_only_query as so
    P = iq.S1 / 'prep'; pool = dict(np.load(P / 'W_pool.npz')); H, raw = hist_inputs()
    hold = np.isin(raw['train']['task'], so.HOLDOUT)
    return dict(T=tokens(layer), H=H, raw=raw, pool=pool, KI=so.key_inputs('W', pool, iq.COND),
                E=np.load(P / f'W_err_{iq.COND}_train.npz')['err'], Edev=np.load(P / f'W_err_{iq.COND}_dev.npz')['err'],
                A=so.allowed_matrix(raw['train']['task'], pool['task'], pool['keep'], True),
                Adev=so.allowed_matrix(raw['dev']['task'], pool['task'], pool['keep'], False),
                fit=np.flatnonzero(~hold), va=np.flatnonzero(hold))


def train(layer, readouts, threads):
    import torch
    torch.set_num_threads(threads); (OUT / 'models').mkdir(parents=True, exist_ok=True); D = load(layer)
    for r in readouts:
        for seed in SEEDS:
            p = OUT / 'models' / f'L{layer}_{r}_s{seed}.pt'
            if p.exists(): continue
            t0 = time.time(); res = train_one(r, seed, D); res['layer'] = layer
            tmp = p.with_suffix('.tmp'); torch.save(res, tmp); tmp.replace(p)
            log(stage='train', layer=layer, readout=r, seed=seed, holdout=round(res['holdout']['value'], 5),
                h_ep=res['holdout']['epoch'], dev=round(res['dev']['value'], 5), sec=round(time.time() - t0, 1))


def evaluate(layer, threads):
    import torch, scale_only_query as so
    torch.set_num_threads(threads); D = load(layer); pool, ev = D['pool'], D['raw']['eval']
    E = wa.eval_errors(pool, ev); A = so.allowed_matrix(ev['task'], pool['task'], pool['keep'], False); hard = ev['hard'].astype(bool)
    KI = torch.as_tensor(D['KI']); HV = wa.history_variants(ev); n = len(E); T = torch.as_tensor(D['T']['eval'])
    res = json.loads((OUT / 'results.json').read_text()) if (OUT / 'results.json').exists() else {}
    for r in ('mean', 'grid', 'attn', 'hattn'):
        rs = [torch.load(p, weights_only=False) for p in sorted((OUT / 'models').glob(f'L{layer}_{r}_s*.pt'))]
        if not rs: continue
        row = {'n_seeds': len(rs), 'holdout': float(np.mean([x['holdout']['value'] for x in rs])),
               'dev_at_holdout_epoch': float(np.mean([x['curve'][x['holdout']['epoch'] - 1]['dev'] for x in rs]))}
        for hn, (hh, rows) in HV.items():
            Hv = D['H']['eval'].copy(); Hv[:, :70] = hh.reshape(n, -1); Hv = torch.as_tensor(Hv); t1, t1h = [], []
            for x in rs:
                net = make_net(r, KI.shape[1]); net.load_state_dict(x['holdout']['state']); net.eval()
                with torch.no_grad(): sc = net(T, Hv, KI).numpy()
                te = E[np.arange(n), np.where(A, sc, -np.inf).argmax(1)][rows]; t1.append(te.mean()); t1h.append(te[hard[rows]].mean())
            row[hn] = [float(np.mean(t1)), float(np.mean(t1h)), float(np.min(t1)), float(np.max(t1))]
        res[f'L{layer}|{r}'] = row
        log(stage='evaluate', key=f'L{layer}|{r}', holdout=round(row['holdout'], 4), dev=round(row['dev_at_holdout_epoch'], 4),
            true=round(row['true'][0], 4), true_hard=round(row['true'][1], 4), seed_range=[round(v, 4) for v in row['true'][2:]],
            **{h: round(row[h][0], 4) for h in ('swap', 'reverse', 'zero', 'phase-10', 'phase+10')})
    (OUT / 'results.json').write_text(json.dumps(res, indent=1) + '\n')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=('train', 'evaluate'))
    ap.add_argument('--layer', type=int, default=16); ap.add_argument('--readouts', default='mean,grid,attn,hattn')
    ap.add_argument('--threads', type=int, default=8)
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == 'train': train(a.layer, a.readouts.split(','), a.threads)
    else: evaluate(a.layer, a.threads)


if __name__ == '__main__':
    main()
