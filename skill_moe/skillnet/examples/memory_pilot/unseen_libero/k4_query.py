"""K4: motion-focused queries on the K1 window setup (unseen LIBERO-90 eval, 1595 candidate shapes).

Every input is past-only (executed actions before the window, current/previous observations).
Unseen targets: Eval only for scoring; Dev (LIBERO-90 tasks 6, 7) targets only for model selection.

Variants
  hist         last 10 actions + state -> MLP (baseline)
  hist_canon   last 10 actions in their own motion frame (rotate net translation to +x, divide by rms;
               same normalization as the memory shapes) + log scale + direction + state
  long         last 40 actions as a sequence, temporal self-attention + attention pooling (+ state)
  long_canon   long, rotated/scaled by the last-10 motion frame
  wrist        wrist-camera tokens (64) read by queries from history; history also concatenated
  agent        third-person tokens (64), same (control)
  dir_attn     queries built only from the recent motion direction/speed, reading all 128 image tokens
  img_motion   difference of current and previous-window image tokens (what moved), read by history queries
  long_canon_wrist  long_canon + wrist
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np
STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', Path(__file__).resolve().parents[5] / '.skillnet-state')).expanduser()

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = (STATE_DIR / 'outputs/memory-pilot/unseen-libero')
K1 = ROOT / 'key-query-01'
K3 = ROOT / 'key-query-03'
OUT = ROOT / 'key-query-04'
SEED = 4101
LONG = 40
KINDS = ('hist', 'hist_canon', 'long', 'long_canon', 'wrist', 'agent', 'dir_attn', 'img_motion', 'long_canon_wrist')


# ---------------------------------------------------------------- prep
def long_history(actions_src, ids, task, episode, frame):
    import compressed_denoise as cd
    tr_all, row_at = cd._stitch(actions_src, task, episode, frame, ids)
    out = np.zeros((len(ids), LONG, 7), np.float32)
    for j, r in enumerate(ids):
        ep, off = row_at[int(r)]; past = tr_all[ep][max(0, off - LONG):off]
        if len(past): out[j, LONG - len(past):] = past
    return out


def canon(h10, hlong=None):
    """Motion-frame normalization from the last-10 history (same rule as memory shapes)."""
    import variable_shape_memory as vs
    n = len(h10); shape = np.zeros((n, 10, 6), np.float32); extra = np.zeros((n, 7), np.float32)
    lc = None if hlong is None else np.zeros((len(hlong), LONG, 7), np.float32)
    for i in range(n):
        if not np.abs(h10[i]).sum():
            continue
        R, s = vs.pose_of(h10[i, :, :6], 'rms')
        shape[i] = vs.to_shape(h10[i, :, :6], 'rms')
        p = vs.to_physical(h10[i, :, :6]); tot = p[:, :3].sum(0); rot = p[:, 3:].sum(0)
        extra[i] = [np.log(max(s, 1e-6)), *(tot / (np.linalg.norm(tot) + 1e-9)), *(rot / (np.linalg.norm(rot) + 1e-9))]
        if lc is not None:
            valid = np.abs(hlong[i]).sum(1) > 0; pl = vs.to_physical(hlong[i, :, :6])
            lc[i, valid, :3] = (pl[valid, :3] @ R.T) / s; lc[i, valid, 3:6] = (pl[valid, 3:] @ R.T) / s
            lc[i, :, 6] = hlong[i, :, 6]
    return np.concatenate([shape.reshape(n, -1), h10[:, :, 6], extra], 1).astype(np.float32), lc


def prep(device='cuda'):
    import key_query_probe as kq, variable_shape_memory as vs
    OUT.mkdir(parents=True, exist_ok=True)
    P = dict(np.load(K3 / 'prep.npz'))
    base, rows, unseen, splits = kq._production_data_splits(include_eval_targets=False)
    U = kq.UNSEEN; lu = lambda k: np.load(U / f'{k}.npy', mmap_mode='r', allow_pickle=False)
    ua, ut, ue, uf = lu('actions'), np.asarray(lu('task')), np.asarray(lu('episode')), np.asarray(lu('frame'))
    ids = {'train': splits['train'][1], 'dev': splits['dev'][1], 'eval': splits['eval'][1]}
    actions = np.load(kq.RUN / 'rows' / 'actions.npy', mmap_mode='r', allow_pickle=False)
    D = {}
    D['train_long'] = long_history(actions, ids['train'], rows['task'], rows['episode'], rows['frame'])
    for s in ('dev', 'eval'): D[f'{s}_long'] = long_history(ua, ids[s], ut, ue, uf)
    # dev basics (targets for selection only)
    import k3_query as k3
    # history = last 10 executed actions (identical to K1's frame-10 window); verified on train and eval
    assert np.allclose(D['train_long'][:, -10:], P['train_hist'], atol=1e-5) and np.allclose(D['eval_long'][:, -10:], P['eval_hist'], atol=1e-5)
    D['dev_hist'] = D['dev_long'][:, -10:].copy(); D['dev_state'] = np.asarray(lu('state')[ids['dev'], :8], np.float32)
    D['dev_prog'], _ = k3.progress_features(ua, ids['dev'], ut, ue, uf)
    ydev = np.asarray(ua[ids['dev'], :, :7], np.float32)
    D['dev_err'] = kq.shape_error_matrix(kq._motion_shape_from_actions(ydev, vs), P['pool_shape'], vs, batch=512, device=device).astype(np.float32)
    for s in ('train', 'dev', 'eval'):
        h = P[f'{s}_hist'] if s != 'dev' else D['dev_hist']
        D[f'{s}_hcanon'], D[f'{s}_lcanon'] = canon(np.asarray(h, np.float32), D[f'{s}_long'])
    # previous-window image token positions
    trid = ids['train']; key = {(int(rows['task'][r]), int(rows['episode'][r]), int(rows['frame'][r])): i for i, r in enumerate(trid)}
    D['train_prev'] = np.array([key.get((int(rows['task'][r]), int(rows['episode'][r]), int(rows['frame'][r]) - 10), -1) for r in trid])
    ukey = {(int(ut[r]), int(ue[r]), int(uf[r])): r for r in range(len(ut))}
    for s in ('dev', 'eval'):
        D[f'{s}_prev'] = np.array([ukey.get((int(ut[r]), int(ue[r]), int(uf[r]) - 10), -1) for r in ids[s]])
        D[f'{s}_rows'] = ids[s]
    tok = np.load(U / 'tokens.npy', mmap_mode='r'); grp = np.load(U / 'groups.npy', mmap_mode='r')
    np.save(OUT / 'img_unseen_all.npy', np.stack([kq.downsample_camera_tokens(tok[r], grp[r], add_position=False)[:128] for r in range(len(ut))]).astype(np.float16))
    np.savez(OUT / 'prep.npz', **D); print('prep done', {k: v.shape for k, v in D.items()}, flush=True)


# ---------------------------------------------------------------- models
def make_net(kind, d=256):
    import torch, torch.nn as nn
    F = nn.functional

    def enc(i, o=d): return nn.Sequential(nn.LayerNorm(i), nn.Linear(i, o), nn.GELU(), nn.Linear(o, o))

    class Temporal(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(7, 128); self.pos = nn.Parameter(torch.randn(LONG, 128) * .02)
            self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(128, 4, 256, .1, batch_first=True, norm_first=True), 2)
            self.q = nn.Parameter(torch.randn(1, 2, 128) * .02); self.pool = nn.MultiheadAttention(128, 4, batch_first=True)
            self.out = nn.Linear(256, d)
        def forward(self, x):
            h = self.tf(self.inp(x) + self.pos); z, _ = self.pool(self.q.expand(len(x), -1, -1), h, h)
            return self.out(z.flatten(1))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.key = nn.Sequential(nn.Linear(70, 512), nn.GELU(), nn.Linear(512, d))
            self.log_scale = nn.Parameter(torch.tensor(np.log(10.)))
            if kind == 'hist': self.h = enc(78)
            elif kind == 'hist_canon': self.h = enc(60 + 10 + 7 + 8)
            else: self.h = enc(78)
            if kind.startswith('long'): self.t = Temporal(); self.st = enc(8, 64)
            if kind in ('wrist', 'agent', 'dir_attn', 'img_motion', 'long_canon_wrist'):
                self.kv = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, d))
                qin = 7 if kind == 'dir_attn' else d
                self.qgen = nn.Sequential(nn.Linear(qin, d), nn.GELU(), nn.Linear(d, 4 * d))
                self.attn = nn.MultiheadAttention(d, 8, batch_first=True)
            fin = {'hist': d, 'hist_canon': d, 'long': d + 64 + d, 'long_canon': d + 64 + d,
                   'wrist': 5 * d, 'agent': 5 * d, 'dir_attn': 5 * d, 'img_motion': 5 * d, 'long_canon_wrist': d + 64 + d + 4 * d}[kind]
            self.fuse = nn.Sequential(nn.Linear(fin, 512), nn.GELU(), nn.Linear(512, d))

        def query(self, b):
            hs = torch.cat([b['hist'].flatten(1), b['state']], -1)
            if kind == 'hist_canon': return self.fuse(self.h(torch.cat([b['hcanon'], b['state']], -1)))
            h = self.h(hs)
            if kind == 'hist': return self.fuse(h)
            parts = [h]
            if kind.startswith('long'):
                parts += [self.st(b['state']), self.t(b['lcanon'] if 'canon' in kind else b['long'])]
            if 'tokens' in b:
                qsrc = b['dir'] if kind == 'dir_attn' else h
                q = self.qgen(qsrc).view(len(h), 4, -1); kv = self.kv(b['tokens'])
                a, _ = self.attn(q, kv, kv); parts.append(a.flatten(1))
            return self.fuse(torch.cat(parts, -1))

        def forward(self, b, K):
            return F.normalize(self.query(b), dim=-1) @ F.normalize(K, dim=-1).T * self.log_scale.exp()
    return Net()


class Data:
    def __init__(self, device):
        import torch
        self.t = torch; self.dev = device
        P = dict(np.load(K3 / 'prep.npz')); Q = dict(np.load(OUT / 'prep.npz'))
        self.D = {**P, **Q}
        self.img = {'train': torch.as_tensor(np.load(K1 / 'production' / 'token_pool_train_image_means.npy', mmap_mode='r')[:], device=device),
                    'unseen': torch.as_tensor(np.load(OUT / 'img_unseen_all.npy'), device=device)}
        pos = torch.as_tensor(np.load(K3 / 'pos.npy'), device=device); self.pos = torch.cat([pos, pos], 0)

    def batch(self, kind, split, idx):
        t, D = self.t, self.D
        T = lambda a: t.as_tensor(np.asarray(a[idx]), device=self.dev, dtype=t.float32)
        b = {'hist': T(D[f'{split}_hist']), 'state': T(D[f'{split}_state'])}
        if kind == 'hist_canon': b['hcanon'] = T(D[f'{split}_hcanon'])
        if kind.startswith('long'): b['long'] = T(D[f'{split}_long']); b['lcanon'] = T(D[f'{split}_lcanon'])
        if kind in ('wrist', 'agent', 'dir_attn', 'img_motion', 'long_canon_wrist'):
            if split == 'train': src = self.img['train']; cur = t.as_tensor(idx, device=self.dev); prev = D['train_prev'][idx]
            else: src = self.img['unseen']; cur = t.as_tensor(D[f'{split}_rows'][idx], device=self.dev); prev = D[f'{split}_prev'][idx]
            tok = src[cur].float() + self.pos[None]
            if kind == 'wrist' or kind == 'long_canon_wrist': tok = tok[:, 64:]
            elif kind == 'agent': tok = tok[:, :64]
            elif kind == 'img_motion':
                pv = src[t.as_tensor(np.maximum(prev, 0), device=self.dev)].float() + self.pos[None]
                tok = (tok - pv) * t.as_tensor(prev >= 0, device=self.dev, dtype=t.float32)[:, None, None]
            b['tokens'] = tok
            if kind == 'dir_attn': b['dir'] = T(D[f'{split}_hcanon'][:, -7:])
        return b


def run(device='cuda', epochs=100, taus=(.01, .003)):
    import torch
    epochs = int(os.environ.get('K4_EPOCHS', epochs))
    DA = Data(device); D = DA.D
    E = D['train_err']; A = D['train_allowed']; hold = np.isin(D['train_task'], D['holdout'])
    fit = np.flatnonzero(~hold); va = np.flatnonzero(hold)
    K = torch.as_tensor(D['pool_shape'].reshape(len(D['pool_shape']), -1), device=device, dtype=torch.float32)
    Et = torch.as_tensor(np.where(A, E, 1e3), device=device); At = torch.as_tensor(A, device=device)
    def metrics(Ee, P):
        n = len(Ee); orc = Ee.min(1); rnd = Ee.mean(1); strict = Ee <= (orc + .1 * (rnd - orc))[:, None]
        rank = np.argsort(-P, 1); p1 = rank[:, 0]; ok = strict[np.arange(n), p1]; se = Ee[np.arange(n), p1]
        return ok, {'top1_strict': float(ok.mean()), 'top5_strict': float(np.mean([strict[i, rank[i, :5]].any() for i in range(n)])),
                    'gap': float((rnd - se).mean() / (rnd - orc).mean()), 'shape': float(se.mean())}
    def predict(net, kind, split, n):
        out = []
        with torch.no_grad():
            kk = net.key(K)
            for i in range(0, n, 512):
                ix = np.arange(i, min(n, i + 512)); out.append(torch.softmax(net(DA.batch(kind, split, ix), kk), -1).cpu().numpy())
        return np.concatenate(out)
    rpath = OUT / 'results.json'; results = json.loads(rpath.read_text()) if rpath.exists() else {}
    hard = D['eval_hard'].astype(bool)
    for kind in KINDS:
        for tau in taus:
            tag = f'{kind}|tau{tau}'
            if tag in results: continue
            t0 = time.time(); torch.manual_seed(SEED)
            teacher = torch.softmax(-(Et - Et.min(1, keepdim=True).values) / tau, -1) * At; teacher /= teacher.sum(1, keepdim=True)
            net = make_net(kind).to(device); lr = 3e-4 if kind not in ('hist', 'hist_canon') else 1e-3
            opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-4)
            steps = epochs * ((len(fit) + 255) // 256); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=.05)
            best_h = (np.inf, None, 0); best_d = (-1, np.inf, None, 0); curve = []
            for ep in range(epochs):
                net.train(); perm = np.random.default_rng(SEED + ep).permutation(fit)
                for i in range(0, len(perm), 256):
                    ix = perm[i:i + 256]; logit = net(DA.batch(kind, 'train', ix), net.key(K)).masked_fill(~At[ix], -1e9)
                    loss = -(teacher[ix] * torch.log_softmax(logit, -1)).sum(-1).mean()
                    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                if (ep + 1) % 5 and ep + 1 != epochs: continue
                net.eval()
                with torch.no_grad():
                    kk = net.key(K); pk = []
                    for i in range(0, len(va), 1024):
                        ix = va[i:i + 1024]; pk.append(net(DA.batch(kind, 'train', ix), kk).masked_fill(~At[ix], -1e9).argmax(-1).cpu().numpy())
                vh = float(E[va, np.concatenate(pk)].mean())
                _, md = metrics(D['dev_err'], predict(net, kind, 'dev', len(D['dev_err'])))
                curve.append([ep + 1, vh, md['top1_strict'], md['shape']])
                sd = {k: v.detach().clone() for k, v in net.state_dict().items()}
                if vh < best_h[0]: best_h = (vh, sd, ep + 1)
                if (md['top1_strict'], -md['shape']) > (best_d[0], -best_d[1]): best_d = (md['top1_strict'], md['shape'], sd, ep + 1)
            rec = {'kind': kind, 'tau': tau, 'epochs': epochs, 'seconds': round(time.time() - t0), 'curve': curve}
            for sel, sd, ep in (('holdout', best_h[1], best_h[2]), ('dev', best_d[2], best_d[3])):
                net.load_state_dict(sd); net.eval()
                P = predict(net, kind, 'eval', len(D['eval_err'])); ok, m = metrics(D['eval_err'], P)
                m['top1_strict_hard'] = float(ok[hard].mean()); m['epoch'] = ep
                m['p_top1_correct'] = float(np.median(P.max(1)[ok])) if ok.any() else None; m['p_top1_wrong'] = float(np.median(P.max(1)[~ok]))
                _, mdv = metrics(D['dev_err'], predict(net, kind, 'dev', len(D['dev_err']))); m['dev_top1_strict'] = mdv['top1_strict']
                rec[sel] = m
            results[tag] = rec; rpath.write_text(json.dumps(results, indent=1))
            print(json.dumps({'tag': tag, 'sec': rec['seconds'], 'holdout_sel': rec['holdout'], 'dev_sel': rec['dev']}), flush=True)
            del net, opt, teacher; torch.cuda.empty_cache()
    report(results)


def report(results):
    L = ['# K4：動きに注目したクエリ（未学習 LIBERO-90 eval 619窓、候補1595、厳しい top1）', '',
         '各方式・τ について、LIBERO-40 保留タスクで選んだ時点と、未学習 Dev（タスク6,7、113窓）の厳しい top1 で選んだ時点の両方を示す。', '',
         '| 方式 | τ | 保留で選択：top1 / 難しい行 / top5 / 回収率 | Dev で選択：top1 / 難しい行 / top5 / 回収率 | Dev top1 | top1 確率 正 / 誤（Dev選択） | 秒 |', '|---|---:|---|---|---:|---:|---:|']
    for tag, r in results.items():
        h, d = r['holdout'], r['dev']
        f = lambda m: f"{100*m['top1_strict']:.0f}% / {100*m['top1_strict_hard']:.0f}% / {100*m['top5_strict']:.0f}% / {100*m['gap']:.0f}%"
        L.append(f"| {r['kind']} | {r['tau']} | {f(h)} | {f(d)} | {100*d['dev_top1_strict']:.0f}% | {d['p_top1_correct'] or 0:.4f} / {d['p_top1_wrong']:.4f} | {r['seconds']} |")
    (OUT / 'report.md').write_text('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    st = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if st in ('prep', 'all') and not (OUT / 'prep.npz').exists(): prep()
    if st in ('run', 'all'): run()
    if st == 'report': report(json.loads((OUT / 'results.json').read_text()))
