"""K3: query improvements on the K1 window setup (unseen LIBERO-90 eval, 1595 candidate shapes).

Variants (all trained with the same recipe on LIBERO-40 train rows, own task excluded, selected on
the LIBERO-40 holdout tasks by top-1 shape error; unseen targets are used only for scoring):
  hist            history(10x7)+state -> MLP                                   (= K1 P5 recipe)
  concat          [VLM group means 3x2048, history, state] raw concat -> MLP     (≈ K1 P6)
  enc             per-branch LayerNorm+encoder, then fuse
  enc_prog        enc + progress (elapsed frames, past gripper switches, past unit count)
  xattn_prog      final-layer image tokens (128, not averaged) + 3 group-mean tokens read by
                  cross-attention with queries from history/state/progress; history and progress
                  are also concatenated directly to the attention output
  mid{10,14,16}_prog  enc_prog with the VLM branch = prefix-V mean (image, language) of that layer

Progress uses only the past executed actions (demo actions before the window in the offline setup).
Eval candidate set: all 1595 pool shapes (fixes the K1 numeric task-id collision for unseen tasks).
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import os
STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', Path(__file__).resolve().parents[5] / '.skillnet-state')).expanduser()

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = (STATE_DIR / 'outputs/memory-pilot/unseen-libero')
K1 = ROOT / 'key-query-01'
OUT = ROOT / 'key-query-03'
SEED = 4101
MID = {10: 1, 14: 2, 16: 4}


def _kq():
    import key_query_probe as kq
    return kq


# ---------------------------------------------------------------- prep
def progress_features(actions_src, ids, task, episode, frame):
    """[N,4]: frame/100, log1p(frame)/5, past gripper switches, past unit count (all past-only)."""
    import compressed_denoise as cd
    tr_all, row_at = cd._stitch(actions_src, task, episode, frame, ids)
    out = np.zeros((len(ids), 4), np.float32); sw = np.zeros(len(ids), np.int64)
    for j, r in enumerate(ids):
        ep, off = row_at[int(r)]; past = tr_all[ep][:off]
        g = np.sign(past[:, 6]) if len(past) else np.zeros(0)
        nsw = int((np.diff(g) != 0).sum()) if len(g) > 1 else 0
        nunit = len(cd._anchor_spans(past)) if len(past) >= 5 else 0
        out[j] = [off / 100, np.log1p(off) / 5, nsw, nunit / 10]; sw[j] = nsw
    return out, sw


def prep(device='cuda'):
    import torch
    import theta_regression as tr, variable_shape_memory as vs
    kq = _kq(); OUT.mkdir(parents=True, exist_ok=True)
    base, rows, unseen, splits = kq._production_data_splits(include_eval_targets=False)
    feats = {s: kq.assemble_split_features(s, include_tokens=False, output=K1) for s in ('train', 'eval')}
    p0 = tr.candidate_shape_pool(rep='C_vm_none')
    sel = json.loads((K1 / 'production' / 'selection_summary.json').read_text()); hold = tuple(sel['holdout_tasks'])
    keep = ~np.isin(p0['source_task'], hold); pool_shape = p0['shape'][keep]; pool_task = p0['source_task'][keep]
    # train errors / allowed
    ftr = feats['train']; ids = ftr['row_ids']
    shape = kq._motion_shape_from_actions(np.asarray(rows['actions'][ids, :, :7], np.float32), vs)
    E = kq.shape_error_matrix(shape, pool_shape, vs, batch=512, device=device).astype(np.float32)
    A = (pool_task[None, :] != ftr['task'][:, None])
    actions = np.load(kq.RUN / 'rows' / 'actions.npy', mmap_mode='r', allow_pickle=False)
    prog_tr, _ = progress_features(actions, ids, rows['task'], rows['episode'], rows['frame'])
    fev = feats['eval']; eids = fev['row_ids']
    lu = lambda k: np.load(kq.UNSEEN / f'{k}.npy', mmap_mode='r', allow_pickle=False)  # past actions only (progress)
    prog_ev, _ = progress_features(lu('actions'), eids, np.asarray(lu('task')), np.asarray(lu('episode')), np.asarray(lu('frame')))
    z = np.load(K1 / 'production' / 'evaluation' / 'rows.npz'); assert np.array_equal(z['row_ids'], eids)
    # recompute eval shape errors against all pool shapes (K1 stored inf for the task-id-collision rows)
    ytgt = np.load(kq.UNSEEN / 'actions.npy', mmap_mode='r')[eids, :, :7].astype(np.float32)  # scoring only
    Eev = kq.shape_error_matrix(kq._motion_shape_from_actions(ytgt, vs), pool_shape, vs, batch=512, device=device).astype(np.float32)
    assert np.isfinite(Eev).all()
    np.savez(OUT / 'prep.npz', train_err=E, train_allowed=A, train_task=ftr['task'], train_hist=ftr['history'], train_state=ftr['state'],
             train_mean=ftr['mean'], train_prog=prog_tr, eval_err=Eev, eval_task=fev['task'], eval_hist=fev['history'],
             eval_state=fev['state'], eval_mean=fev['mean'], eval_prog=prog_ev, eval_hard=z['hard'], pool_shape=pool_shape, holdout=np.asarray(hold))
    # mid-layer prefix-V means [image mean, language mean]
    for split in ('train', 'eval'):
        V = np.load(K1 / 'policy_features' / f'{split}_mid_v.npy', mmap_mode='r'); M = np.load(K1 / 'policy_features' / f'{split}_mid_mask.npy', mmap_mode='r')
        res = {f'l{l}': np.zeros((len(V), 512), np.float32) for l in MID}
        for i in range(0, len(V), 512):
            m = np.asarray(M[i:i + 512]); lm = m[:, 128:].astype(np.float32)
            for l, li in MID.items():
                v = np.asarray(V[i:i + 512, li], np.float32)
                res[f'l{l}'][i:i + 512] = np.concatenate([v[:, :128].mean(1), (v[:, 128:] * lm[:, :, None]).sum(1) / np.maximum(lm.sum(1, keepdims=True), 1)], 1)
        np.savez(OUT / f'mid_{split}.npz', **res); print('mid', split, flush=True)
    # final-layer image tokens (128) with the same 2D position encoding as K1
    pos = kq._position_2d(64, 2048).astype(np.float32)
    img_tr = np.load(K1 / 'production' / 'token_pool_train_image_means.npy', mmap_mode='r')
    np.save(OUT / 'img_train_path.npy', np.array([str(K1 / 'production' / 'token_pool_train_image_means.npy')]))
    tok = np.load(kq.UNSEEN / 'tokens.npy', mmap_mode='r'); grp = np.load(kq.UNSEEN / 'groups.npy', mmap_mode='r')
    img_ev = np.stack([kq.downsample_camera_tokens(tok[int(r)], grp[int(r)], add_position=False)[:128] for r in eids]).astype(np.float16)
    np.save(OUT / 'img_eval.npy', img_ev); np.save(OUT / 'pos.npy', pos)
    assert img_tr.shape[0] == len(ids)
    print('prep done', E.shape, flush=True)


# ---------------------------------------------------------------- models
def make_net(kind, d=256, mid_dim=512):
    import torch, torch.nn as nn
    F = nn.functional

    def enc(i, o=d): return nn.Sequential(nn.LayerNorm(i), nn.Linear(i, o), nn.GELU(), nn.Linear(o, o))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.key = nn.Sequential(nn.Linear(70, 512), nn.GELU(), nn.Linear(512, d))
            self.log_scale = nn.Parameter(torch.tensor(np.log(10.)))
            if kind == 'hist': self.q = nn.Sequential(nn.Linear(78, 512), nn.GELU(), nn.Linear(512, d))
            elif kind == 'concat': self.q = nn.Sequential(nn.Linear(6144 + 78, 768), nn.GELU(), nn.Linear(768, d))
            else:
                self.hs = enc(78)
                vdim = mid_dim if kind.startswith('mid') else 6144
                prog = kind.endswith('prog')
                if prog: self.pg = enc(4 + 6, 64)
                if kind == 'xattn_prog':
                    self.kv = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, d))
                    self.qgen = nn.Linear(d + 64, 4 * d); self.attn = nn.MultiheadAttention(d, 8, batch_first=True)
                    fuse_in = 4 * d + d + 64
                else:
                    self.vl = enc(vdim); fuse_in = d + d + (64 if prog else 0)
                self.fuse = nn.Sequential(nn.Linear(fuse_in, 512), nn.GELU(), nn.Linear(512, d))

        def query(self, b):
            hs = torch.cat([b['hist'].flatten(1), b['state']], -1)
            if kind == 'hist': return self.q(hs)
            if kind == 'concat': return self.q(torch.cat([b['vlm'], hs], -1))
            h = self.hs(hs); parts = [h]
            if kind.endswith('prog'):
                p = b['prog']; sw = F.one_hot(p[:, 2].long().clamp(0, 5), 6).float()
                pg = self.pg(torch.cat([p, sw], -1))
            if kind == 'xattn_prog':
                q = self.qgen(torch.cat([h, pg], -1)).view(len(h), 4, -1)
                kv = self.kv(b['tokens']); a, _ = self.attn(q, kv, kv)
                parts = [a.flatten(1), h, pg]
            else:
                parts.append(self.vl(b['vlm']))
                if kind.endswith('prog'): parts.append(pg)
            return self.fuse(torch.cat(parts, -1))

        def forward(self, b, K):
            return F.normalize(self.query(b), dim=-1) @ F.normalize(K, dim=-1).T * self.log_scale.exp()
    return Net()


KINDS = ('hist', 'concat', 'enc', 'enc_prog', 'xattn_prog', 'mid10_prog', 'mid14_prog', 'mid16_prog')


def batch_for(kind, split, D, idx, device):
    import torch
    T = lambda a: torch.as_tensor(np.asarray(a[idx]), device=device, dtype=torch.float32)
    b = {'hist': T(D[f'{split}_hist']), 'state': T(D[f'{split}_state']), 'prog': T(D[f'{split}_prog'])}
    if kind in ('concat', 'enc', 'enc_prog'): b['vlm'] = T(D[f'{split}_mean']).flatten(1)
    if kind.startswith('mid'): b['vlm'] = T(D[f'{split}_mid_l{kind[3:5]}'])
    if kind == 'xattn_prog':
        img = D[f'{split}_img']
        ix = torch.as_tensor(idx, device=device)
        tok = img[ix].float() + D['pos2'][None]
        g = T(D[f'{split}_mean']).view(len(idx), 3, 2048)
        b['tokens'] = torch.cat([tok, g], 1)
    return b


def run(device='cuda', epochs=None, taus=(.01, .003)):
    import os
    epochs = int(epochs or os.environ.get('K3_EPOCHS', 20)); sfx = '' if epochs == 20 else f'_e{epochs}'
    import torch
    D = dict(np.load(OUT / 'prep.npz'))
    for s in ('train', 'eval'):
        m = np.load(OUT / f'mid_{s}.npz')
        for l in MID: D[f'{s}_mid_l{l}'] = m[f'l{l}']
    t0 = time.time()
    D['train_img'] = torch.as_tensor(np.load(str(np.load(OUT / 'img_train_path.npy')[0]), mmap_mode='r')[:], device=device)
    D['eval_img'] = torch.as_tensor(np.load(OUT / 'img_eval.npy'), device=device)
    pos = torch.as_tensor(np.load(OUT / 'pos.npy'), device=device); D['pos2'] = torch.cat([pos, pos], 0)
    print('loaded tokens', round(time.time() - t0), 's', flush=True)
    E = D['train_err']; A = D['train_allowed']; hold = np.isin(D['train_task'], D['holdout'])
    fit = np.flatnonzero(~hold); va = np.flatnonzero(hold)
    K = torch.as_tensor(D['pool_shape'].reshape(len(D['pool_shape']), -1), device=device, dtype=torch.float32)
    Et = torch.as_tensor(np.where(A, E, 1e3), device=device); At = torch.as_tensor(A, device=device)
    Ee = D['eval_err']; n = len(Ee); orc = Ee.min(1); rnd = Ee.mean(1)
    strict = Ee <= (orc + .1 * (rnd - orc))[:, None]; loose = Ee <= orc[:, None] + .01; hard = D['eval_hard'].astype(bool)
    rpath = OUT / f'results{sfx}.json'
    results = json.loads(rpath.read_text()) if rpath.exists() else {}
    for kind in KINDS:
        for tau in taus:
            tag = f'{kind}|tau{tau}'
            if tag in results: continue
            torch.manual_seed(SEED)
            teacher = torch.softmax(-(Et - Et.min(1, keepdim=True).values) / tau, -1) * At; teacher /= teacher.sum(1, keepdim=True)
            net = make_net(kind).to(device); opt = torch.optim.AdamW(net.parameters(), 3e-4 if kind == 'xattn_prog' else 1e-3, weight_decay=1e-4)
            best = (np.inf, None, 0); steps = epochs * ((len(fit) + 255) // 256)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=opt.param_groups[0]['lr'], total_steps=steps, pct_start=.05)
            curve = []
            for ep in range(epochs):
                net.train(); perm = np.random.default_rng(SEED + ep).permutation(fit)
                for i in range(0, len(perm), 256):
                    ix = perm[i:i + 256]; b = batch_for(kind, 'train', D, ix, device)
                    logit = net(b, net.key(K)).masked_fill(~At[ix], -1e9)
                    loss = -(teacher[ix] * torch.log_softmax(logit, -1)).sum(-1).mean()
                    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                net.eval(); picks = []
                with torch.no_grad():
                    kk = net.key(K)
                    for i in range(0, len(va), 1024):
                        ix = va[i:i + 1024]; picks.append(net(batch_for(kind, 'train', D, ix, device), kk).masked_fill(~At[ix], -1e9).argmax(-1).cpu().numpy())
                v = float(E[va, np.concatenate(picks)].mean()); curve.append(v)
                if v < best[0]: best = (v, {k: t.detach().clone() for k, t in net.state_dict().items()}, ep + 1)
            net.load_state_dict(best[1]); net.eval()
            with torch.no_grad():
                P = torch.softmax(net(batch_for(kind, 'eval', D, np.arange(n), device), net.key(K)), -1).cpu().numpy()
            rank = np.argsort(-P, 1); p1 = rank[:, 0]; se = Ee[np.arange(n), p1]
            ok = strict[np.arange(n), p1]
            rec = {'kind': kind, 'tau': tau, 'val_shape': best[0], 'epoch': best[2], 'epochs': epochs, 'val_curve_every10': curve[9::10],
                   'top1_strict': float(ok.mean()), 'top1_strict_hard': float(ok[hard].mean()),
                   'top5_strict': float(np.mean([strict[i, rank[i, :5]].any() for i in range(n)])),
                   'top1_loose': float(loose[np.arange(n), p1].mean()),
                   'gap': float((rnd - se).mean() / (rnd - orc).mean()), 'gap_hard': float((rnd - se)[hard].mean() / (rnd - orc)[hard].mean()),
                   'p_top1_median_correct': float(np.median(P.max(1)[ok])) if ok.any() else None,
                   'p_top1_median_wrong': float(np.median(P.max(1)[~ok])),
                   'strict_mass_median_wrong': float(np.median((P * strict).sum(1)[~ok]))}
            results[tag] = rec; print(json.dumps(rec), flush=True)
            rpath.write_text(json.dumps(results, indent=1))
            torch.save(best[1], OUT / f'model_{kind}_tau{tau}{sfx}.pt')
            del net, opt, teacher; torch.cuda.empty_cache()
    report(results, sfx)


def report(results, sfx=''):
    sel = {}
    for r in results.values():
        if r['kind'] not in sel or r['val_shape'] < sel[r['kind']]['val_shape']: sel[r['kind']] = r
    assert all(np.isfinite(r['gap']) for r in results.values())
    L = ['# K3：クエリの改良（未学習 LIBERO-90、619窓、候補1595）', '',
         'τ は LIBERO-40 保留タスクの top1 型誤差で選択。厳しい基準＝oracle から random までの差の10%以内。', '',
         '| 方式 | τ | top1（厳しい） | 難しい行 | top5（厳しい） | top1（緩い .01） | 差の回収率 | 難しい行 | top1 確率 正しい / 間違い |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for k in KINDS:
        if k not in sel: continue
        r = sel[k]
        L.append(f"| {k} | {r['tau']} | {100*r['top1_strict']:.0f}% | {100*r['top1_strict_hard']:.0f}% | {100*r['top5_strict']:.0f}% | {100*r['top1_loose']:.0f}% | "
                 f"{100*r['gap']:.0f}% | {100*r['gap_hard']:.0f}% | {r['p_top1_median_correct'] or 0:.4f} / {r['p_top1_median_wrong']:.4f} |")
    (OUT / f'report{sfx}.md').write_text('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    st = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if st in ('prep', 'all') and not (OUT / 'prep.npz').exists(): prep()
    if st in ('run', 'all'): run()
    if st == 'report': report(json.loads((OUT / 'results.json').read_text()))
