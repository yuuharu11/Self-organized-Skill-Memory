"""I2b: fusion weight / gate fitted on Dev (unseen LIBERO-90 tasks 6, 7) instead of LIBERO-40, where pi0.5 is trained
(pi0.5-NN holdout Spearman .98).  P5 frozen (S1).  Gate = linear on 7 agreement features -> softplus (a, b);
trained on Dev with the same history corruption as I2 (none .5 / swap .2 / reverse .15 / zero .15), 300 full-batch
steps, weight decay 1e-2 (declared; no Eval-based choice).  Eval: true / swap / reverse / zero history."""
import json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import image_query as iq, image_query_gate as g, scale_only_query as so
P = iq.S1 / 'prep'; pool = dict(np.load(P / 'W_pool.npz')); KI = torch.as_tensor(so.key_inputs('W', pool, iq.COND))
cd, vs, kq = so._mods()

def p5net(seed):
    s = json.loads((iq.S1 / 'selection.json').read_text())['selected'][f'W|{iq.COND}|P5|{seed}']['holdout']
    r = torch.load(iq.S1 / 'models' / f"W_{iq.COND}_P5_s{seed}_f{s['frac']:.4f}.pt", weights_only=False)
    n = so.make_net('W', 'P5', 78, KI.shape[1]); n.load_state_dict(r['holdout']['state']); n.eval(); return n

def corrupt_sets(d, rng):
    n = len(d['task']); perm = np.array([rng.choice(np.flatnonzero(d['task'] != d['task'][i])) if (d['task'] != d['task'][i]).any() else rng.integers(n) for i in range(n)])
    h = d['hist']; r = h.copy(); r[:, :, :3] *= -1
    return {'true': h, 'swap': h[perm], 'reverse': r, 'zero': np.zeros_like(h)}

def feats(h, d, act):
    ht = torch.as_tensor(h); return torch.cat([ht.reshape(len(h), -1), torch.as_tensor(d['state'])], 1), g.agreement(ht, torch.as_tensor(act))

dv = dict(np.load(P / 'W_inputs_dev.npz')); Ed = torch.as_tensor(np.load(P / f'W_err_{iq.COND}_dev.npz')['err'])
Ad = torch.as_tensor(so.allowed_matrix(dv['task'], pool['task'], pool['keep'], False)); actd = np.load(iq._pf('dev') / 'dev_actions.npy').astype(np.float32)
ev = dict(np.load(P / 'W_inputs_eval.npz')); acte = np.load(iq._pf('eval') / 'eval_actions.npy').astype(np.float32)
ua = np.load(kq.UNSEEN / 'actions.npy', mmap_mode='r'); y = np.asarray(ua[ev['row'], :, :6], np.float32)       # scoring only
Ee = so.type_errors(vs.to_physical(y).astype(np.float32), pool['phys'], iq.COND, 'cuda')
Ae = so.allowed_matrix(ev['task'], pool['task'], pool['keep'], False); hard = ev['hard'].astype(bool)
orc = np.where(Ae, Ee, np.inf).min(1); rnd = np.where(Ae, Ee, 0).sum(1) / Ae.sum(1)
znd = g.zt(torch.as_tensor(iq.nn_scores('dev')), Ad); zne = g.zt(torch.as_tensor(iq.nn_scores('eval')), torch.as_tensor(Ae))
HD = corrupt_sets(dv, np.random.default_rng(5)); HE = corrupt_sets(ev, np.random.default_rng(4101))
mix = {'true': .5, 'swap': .2, 'reverse': .15, 'zero': .15}

def metr(sc):
    n = len(sc); sp = iq.spearman_rows(sc, Ee, Ae); pick = np.where(Ae, sc, -np.inf).argmax(1); te = Ee[np.arange(n), pick]
    return {'spearman': round(float(sp.mean()), 4), 'spearman_hard': round(float(sp[hard].mean()), 4),
            'gap_recovered': round(float(((rnd - te) / np.maximum(rnd - orc, 1e-9)).mean()), 4), 'strict_top1': round(float((te <= orc + .1 * (rnd - orc)).mean()), 4)}
res = {}
for seed in (4101, 4102, 4103):
    net = p5net(seed)
    with torch.no_grad():
        zpd = {k: g.zt(net(feats(h, dv, actd)[0], KI), Ad) for k, h in HD.items()}
        zpe = {k: g.zt(net(feats(h, ev, acte)[0], KI), torch.as_tensor(Ae)) for k, h in HE.items()}
    # (1) constant weight chosen on Dev (mixed corruption), grid 0..6
    best = None
    for w in np.linspace(0, 6, 61):
        v = sum(p * float(g.spearman_t(zpd[k] + w * znd, Ed, Ad).mean()) for k, p in mix.items())
        best = max(best or (-9, 0), (v, float(w)))
    w = best[1]; res.setdefault('const_dev', []).append({'w': w, **{k: metr((zpe[k] + w * zne).numpy()) for k in HE}})
    # (2) linear gate on agreement features, trained on Dev
    torch.manual_seed(seed); lin = torch.nn.Linear(7, 2); torch.nn.init.zeros_(lin.weight); torch.nn.init.constant_(lin.bias, .5)
    opt = torch.optim.Adam(lin.parameters(), 1e-2, weight_decay=1e-2)
    fd = {k: feats(h, dv, actd)[1] for k, h in HD.items()}; mu = fd['true'].mean(0); sd = fd['true'].std(0) + 1e-6
    tau = so.teacher_tau(Ed.numpy(), Ad.numpy(), .1)
    teacher = torch.softmax(-(Ed - Ed.masked_fill(~Ad, 1e3).min(1, keepdim=True).values) / tau, -1) * Ad; teacher /= teacher.sum(1, keepdim=True)
    for step in range(300):
        l = 0
        for k, p in mix.items():
            ab = torch.nn.functional.softplus(lin((fd[k] - mu) / sd)) * 3
            sc = (ab[:, :1] * zpd[k] + ab[:, 1:] * znd).masked_fill(~Ad, -1e9)
            l = l + p * (-(teacher * torch.log_softmax(sc, -1)).sum(-1).mean() + (1 - g.pearson_t(sc.masked_fill(~Ad, 0), -Ed, Ad)).mean())
        opt.zero_grad(); l.backward(); opt.step()
    out = {}
    with torch.no_grad():
        for k, h in HE.items():
            ab = torch.nn.functional.softplus(lin((feats(h, ev, acte)[1] - mu) / sd)) * 3
            out[k] = {**metr((ab[:, :1] * zpe[k] + ab[:, 1:] * zne).numpy()), 'image_share': round(float((ab[:, 1] / ab.sum(1)).mean()), 3)}
    res.setdefault('gate_dev', []).append(out)
    print(seed, 'w', w, json.dumps(res['const_dev'][-1]), '\n gate', json.dumps(out), flush=True)
(iq.OUT / 'gate' / 'dev_results.json').write_text(json.dumps(res, indent=1) + '\n')
for name in res:
    for k in HE:
        print(name, k, {m: round(np.mean([r[k][m] for r in res[name]]), 3) for m in res[name][0][k]})
