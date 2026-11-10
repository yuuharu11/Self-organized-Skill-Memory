"""Retrieval over the real variable-length DCT-3 memory units (no 10-frame window fragments).

Memory: LIBERO-40 memory demos (6,583 rows / 400 demos) cut at speed minima + gripper switches
(min 5 frames), each unit stored as pose-removed ('rms') DCT-3 values + length; holdout-task units removed.
Queries: every segment of a query demo cut by the same rule. At a segment start the query sees only the
past (previous 10 executed actions, zeros if <10; last observed row state at or before the start).
Segment cuts of unseen demos come from their actions and are used only to define scoring targets.

Shape error: unit decoded at a common length (16) vs the target segment resampled to 16, after the
optimal shared rotation (translation and rotation triplets) and scale, in physical units, mean over 6 axes.

  python unit_retrieval.py all
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
import variable_shape_memory as vs  # noqa: E402

OUT = (STATE_DIR / 'outputs/memory-pilot/unseen-libero/unit-retrieval-01')
HOLDOUT = (2, 7, 12, 17, 22, 27, 32, 37)
L = 16
SEED = 4101


# ---------------------------------------------------------------- data
def resample(x, n=L):
    x = np.asarray(x, np.float64); t = np.linspace(0, len(x) - 1, n)
    return np.stack([np.interp(t, np.arange(len(x)), x[:, j]) for j in range(x.shape[1])], 1)


def unit_shape(values, n=L):
    from scipy.fft import idct
    c = np.zeros((n, 6)); c[:vs.K] = np.asarray(values[:vs.K * 6]).reshape(vs.K, 6) * np.sqrt(n)
    return idct(c, type=2, norm='ortho', axis=0)


def segments_from(actions, task, episode, frame, state, row_ids, tag):
    """All speed-minimum segments with past-only query inputs."""
    traces, row_at = cd._stitch(actions, task, episode, frame, row_ids)
    first = {}
    for r, (ep, off) in row_at.items(): first.setdefault(ep, {})[off] = r
    out = {k: [] for k in ('task', 'episode', 'start', 'length', 'target', 'hist', 'state', 'prev', 'has_prev', 'last')}
    for ep, tr in sorted(traces.items()):
        spans = cd._anchor_spans(tr); prev = None
        for s, e in spans:
            if e - s < 2: continue
            hist = tr[s - 10:s, :7] if s >= 10 else np.zeros((10, 7), np.float32)
            r = first[ep][(s // 10) * 10]
            out['task'].append(ep[0]); out['episode'].append(ep[1]); out['start'].append(s); out['length'].append(e - s)
            out['target'].append(resample(vs.to_physical(tr[s:e, :6])))
            out['hist'].append(hist); out['state'].append(np.asarray(state[r][:8], np.float32))
            out['prev'].append(resample(vs.to_physical(tr[prev[0]:prev[1], :6])) if prev else np.zeros((L, 6)))
            out['has_prev'].append(prev is not None)
            out['last'].append(vs.to_physical(tr[s - 1:s, :6])[0] if s >= 1 else np.zeros(6))
            prev = (s, e)
    res = {k: np.asarray(v) for k, v in out.items()}
    print(tag, 'segments', len(res['task']), 'median length', np.median(res['length']), flush=True)
    return res


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    R = cd.RUN / 'rows'; ld = lambda k: np.load(R / f'{k}.npy', mmap_mode='r', allow_pickle=False)
    task, ep, fr, val = (np.asarray(ld(k)) for k in ('task', 'episode', 'frame', 'validation'))
    actions, state = ld('actions'), ld('state')
    mem_ids = np.asarray(np.load(cd.RUN / 'memory_row_ids.npy'), np.int64)
    rows = {'task': task, 'episode': ep, 'frame': fr}
    bank = cd._bank_from_memory(rows, mem_ids, mode='variable', form='rms', tau=None)
    m = bank['memory']; keep = ~np.isin(bank['task'], HOLDOUT)
    order = np.lexsort((bank['start'], bank['episode'], bank['task']))
    nxt = np.full(len(order), -1)
    for a, b in zip(order[:-1], order[1:]):
        if bank['task'][a] == bank['task'][b] and bank['episode'][a] == bank['episode'][b]: nxt[a] = b
    np.savez(OUT / 'units.npz', values=np.asarray(m.values), lengths=np.asarray(m.lengths), task=bank['task'], episode=bank['episode'],
             start=bank['start'], next=nxt, keep=keep, shapes=np.stack([unit_shape(v) for v in np.asarray(m.values)]))
    print('units', len(keep), 'kept', keep.sum(), flush=True)
    train_ids = np.flatnonzero(~val)
    segs = segments_from(actions, task, ep, fr, state, train_ids, 'libero40')
    np.savez(OUT / 'segments_libero40.npz', **segs)
    U = cd.UNSEEN_ROWS; lu = lambda k: np.load(U / f'{k}.npy', mmap_mode='r', allow_pickle=False)
    ut, ue, uf, dev = (np.asarray(lu(k)) for k in ('task', 'episode', 'frame', 'is_dev'))
    segs = segments_from(lu('actions'), ut, ue, uf, lu('state'), np.flatnonzero(~dev), 'unseen')
    np.savez(OUT / 'segments_unseen.npz', **segs)


# ---------------------------------------------------------------- shape errors
def procrustes_err(cands, targets, device='cuda', chunk=128):
    """cands [M,L,6], targets [N,L,6] physical -> [N,M] min over rotation+scale of mean sq error."""
    import torch
    C = torch.as_tensor(cands, dtype=torch.float64, device=device); T = torch.as_tensor(targets, dtype=torch.float64, device=device)
    cn = (C ** 2).sum((1, 2)); out = []
    for i in range(0, len(T), chunk):
        t = T[i:i + chunk]
        H = torch.einsum('mli,nlj->nmij', C[:, :, :3], t[:, :, :3]) + torch.einsum('mli,nlj->nmij', C[:, :, 3:], t[:, :, 3:])
        s = torch.linalg.svdvals(H); d = torch.sign(torch.linalg.det(H))
        tr = s[..., 0] + s[..., 1] + d * s[..., 2]
        e = ((t ** 2).sum((1, 2))[:, None] - tr.clamp_min(0) ** 2 / cn.clamp_min(1e-12)[None]) / (L * 6)
        out.append(e.clamp_min(0).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def errors():
    u = np.load(OUT / 'units.npz')
    for name in ('libero40', 'unseen'):
        s = np.load(OUT / f'segments_{name}.npz')
        E = procrustes_err(u['shapes'], s['target'])
        P = procrustes_err(u['shapes'], s['prev'])  # unit nearest to the previous segment (past only)
        np.savez(OUT / f'err_{name}.npz', err=E, prev_err=P)
        print(name, E.shape, 'oracle median', float(np.median(E[:, u['keep']].min(1))), flush=True)


# ---------------------------------------------------------------- learned query
def allowed_mask(seg_task, u, cross_task):
    a = np.broadcast_to(u['keep'], (len(seg_task), len(u['keep']))).copy()
    if cross_task: a &= u['task'][None, :] != np.asarray(seg_task)[:, None]
    return a


def make_net(d=256):
    import torch.nn as nn, torch
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Sequential(nn.Linear(78, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, d))
            self.key = nn.Sequential(nn.Linear(19 + 1, 512), nn.GELU(), nn.Linear(512, d))
            self.log_scale = nn.Parameter(torch.tensor(np.log(20.)))
        def keys(self, uin): return nn.functional.normalize(self.key(uin), dim=-1)
        def forward(self, x, K): return nn.functional.normalize(self.query(x), dim=-1) @ K.T * self.log_scale.exp()
    return Net()


def unit_inputs(u):
    return np.concatenate([u['values'], np.log(u['lengths'])[:, None] / 3], 1).astype(np.float32)


def query_inputs(s):
    return np.concatenate([s['hist'].reshape(len(s['hist']), -1), s['state']], 1).astype(np.float32)


def train(device='cuda', epochs=40):
    import torch
    torch.manual_seed(SEED)
    u = np.load(OUT / 'units.npz'); s = np.load(OUT / 'segments_libero40.npz'); E = np.load(OUT / 'err_libero40.npz')['err']
    A = allowed_mask(s['task'], u, True)
    hold = np.isin(s['task'], HOLDOUT); fit = np.flatnonzero(~hold); va = np.flatnonzero(hold)
    Em = np.where(A, E, np.inf); orc = Em.min(1); rnd = np.array([E[i, A[i]].mean() for i in range(len(E))])
    gap = float(np.median(rnd - orc))
    X = torch.as_tensor(query_inputs(s), device=device); UI = torch.as_tensor(unit_inputs(u), device=device)
    Et = torch.as_tensor(np.where(A, E, 1e3), device=device); At = torch.as_tensor(A, device=device)
    results = {}
    for frac in (1 / 3, 1 / 10, 1 / 30):
        tau = gap * frac
        teacher = torch.softmax(-(Et - Et.min(1, keepdim=True).values) / tau, -1) * At; teacher = teacher / teacher.sum(1, keepdim=True)
        net = make_net().to(device); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
        best = (np.inf, None, -1)
        for epoch in range(epochs):
            net.train(); perm = torch.as_tensor(np.random.default_rng(SEED + epoch).permutation(fit), device=device)
            for i in range(0, len(perm), 256):
                b = perm[i:i + 256]; K = net.keys(UI)
                logit = net(X[b], K).masked_fill(~At[b], -1e9)
                loss = -(teacher[b] * torch.log_softmax(logit, -1)).sum(-1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                pick = net(X[va], net.keys(UI)).masked_fill(~At[va], -1e9).argmax(-1).cpu().numpy()
            v = float(E[va, pick].mean())
            if v < best[0]: best = (v, {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}, epoch + 1)
        results[f'{frac:.4f}'] = {'tau': tau, 'val_shape': best[0], 'epoch': best[2],
                                  'teacher_eff_median': float(np.median(torch.exp(-(teacher * torch.log(teacher + 1e-30)).sum(1)).cpu().numpy()))}
        torch.save({'state_dict': best[1], 'tau': tau}, OUT / f'query_tau{frac:.4f}.pt')
        print(json.dumps({'teacher_frac': frac, **results[f'{frac:.4f}']}), flush=True)
    sel = min(results, key=lambda k: results[k]['val_shape'])
    (OUT / 'train.json').write_text(json.dumps({'runs': results, 'selected': sel, 'selection': 'LIBERO-40 holdout tasks, top-1 shape error'}, indent=1))


# ---------------------------------------------------------------- evaluate
def evaluate(device='cuda'):
    import torch
    u = np.load(OUT / 'units.npz'); s = np.load(OUT / 'segments_unseen.npz'); z = np.load(OUT / 'err_unseen.npz')
    E, PE = z['err'], z['prev_err']; n = len(E)
    A = allowed_mask(s['task'], u, False)  # unseen tasks never share a task with LIBERO-40
    Em = np.where(A, E, np.inf); orc = Em.min(1); oi = Em.argmin(1)
    rnd = np.array([E[i, A[i]].mean() for i in range(n)])
    good = Em <= (orc + .1 * (rnd - orc))[:, None]   # within 10% of the random-oracle gap above oracle
    rng = np.random.default_rng(SEED)
    picks, conf = {}, {}
    picks['random'] = np.array([rng.choice(np.flatnonzero(A[i])) for i in range(n)])
    PEm = np.where(A, PE, np.inf); near_prev = PEm.argmin(1)
    picks['直前の区間に最も近い単位'] = near_prev
    nx = u['next'][near_prev]; picks['その単位の次の単位（流れ）'] = np.where((nx >= 0) & A[np.arange(n), np.maximum(nx, 0)], nx, near_prev)
    tj = json.loads((OUT / 'train.json').read_text())
    X = torch.as_tensor(query_inputs(s), device=device); UI = torch.as_tensor(unit_inputs(u), device=device); At = torch.as_tensor(A, device=device)
    for frac in tj['runs']:
        net = make_net().to(device); net.load_state_dict(torch.load(OUT / f'query_tau{frac}.pt', weights_only=False)['state_dict']); net.eval()
        with torch.no_grad():
            P = torch.softmax(net(X, net.keys(UI)).masked_fill(~At, -1e9), -1).cpu().numpy().astype(np.float64)
        lab = f'学習クエリ（目標の鋭さ 1/{round(1 / float(frac))}）' + ('★' if frac == tj['selected'] else '')
        picks[lab] = P.argmax(1); conf[lab] = P
    picks['oracle'] = oi
    # trajectory baselines (not memory units)
    tgt = s['target']
    def traj_err(traj):
        return np.array([procrustes_err(traj[i:i + 1], tgt[i:i + 1], device='cpu')[0, 0] for i in range(n)])
    hold = traj_err(np.repeat(s['last'][:, None, :], L, 1)); rep = traj_err(s['prev'])
    hasp = s['has_prev'].astype(bool); hist0 = np.abs(s['hist']).sum((1, 2)) == 0
    eff = lambda P: np.exp(-(P * np.log(P + 1e-300)).sum(1))
    rows = []
    def add(lab, e, pick=None):
        r = {'method': lab, 'shape': float(e.mean()), 'shape_hasprev': float(e[hasp].mean()),
             'gap': float((rnd - e).mean() / (rnd - orc).mean())}
        if pick is not None:
            ok = good[np.arange(n), pick]; pct = np.array([(Em[i] < Em[i, pick[i]]).mean() for i in range(n)])
            r.update(top1_good=float(ok.mean()), rank_pct_median=float(np.median(pct)),
                     len_ratio_median=float(np.median(np.abs(np.log(u['lengths'][pick] / s['length'])))))
            if lab in conf:
                P = conf[lab]; p1 = P.max(1); mass = (P * good).sum(1)
                best_good_rank = np.array([int((P[i] > P[i][good[i]].max()).sum()) + 1 for i in range(n)])
                r['confidence'] = {g: {'n': int(m.sum()), 'p_top1_median': float(np.median(p1[m])), 'p_top1_q90': float(np.quantile(p1[m], .9)),
                                       'good_mass_median': float(np.median(mass[m])), 'good_count_median': float(np.median(good.sum(1)[m])),
                                       'best_good_rank_median': float(np.median(best_good_rank[m])), 'eff_candidates_median': float(np.median(eff(P)[m]))}
                                   for g, m in (('correct', ok), ('wrong', ~ok))}
        rows.append(r)
    for lab, pk in picks.items(): add(lab, E[np.arange(n), pk], pk)
    add('基準：直前の行動を保つ（軌道）', hold); add('基準：直前の区間を繰り返す（軌道）', np.where(hasp, rep, hold))
    res = {'segments': n, 'tasks': sorted(map(int, np.unique(s['task']))), 'units_allowed': int(A[0].sum()),
           'segment_length_median': float(np.median(s['length'])), 'random_shape': float(rnd.mean()), 'oracle_shape': float(orc.mean()),
           'good_count_median': float(np.median(good.sum(1))), 'no_history_segments': int(hist0.sum()), 'rows': rows, 'train': tj}
    (OUT / 'results.json').write_text(json.dumps(res, indent=1, ensure_ascii=False))
    report(res)


def report(res):
    Lr = ['# 可変長 DCT-3 単位の検索（未学習 LIBERO-90 eval）', '',
          f"区間 {res['segments']}（長さ中央値 {res['segment_length_median']:.0f}）、候補単位 {res['units_allowed']}。型誤差は向きと大きさを最適に合わせた物理単位の残差。",
          f"random {res['random_shape']:.5f}、oracle {res['oracle_shape']:.5f}。正しい = oracle から random までの差の10%以内（候補数の中央値 {res['good_count_median']:.0f}）。", '',
          '| 方法 | 型誤差 | 前区間ありの行 | 差の回収率 | top1 が正しい | 選んだ単位の順位（中央値） | 長さの比 |log| |', '|---|---:|---:|---:|---:|---:|---:|']
    for r in res['rows']:
        Lr.append(f"| {r['method']} | {r['shape']:.5f} | {r['shape_hasprev']:.5f} | {100 * r['gap']:.0f}% | "
                  + (f"{100 * r['top1_good']:.0f}% | 上位{100 * r['rank_pct_median']:.1f}% | {r['len_ratio_median']:.2f} |" if 'top1_good' in r else '— | — | — |'))
    Lr += ['', '## 確信度（学習クエリ）', '', '| 方法 | 区分 | 行 | top1 の確率 中央値 (90%点) | 正しい候補への確率の合計 | 正しい候補の数 | 正しい候補の最上位の順位 | 実質候補数 |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in res['rows']:
        for g, c in r.get('confidence', {}).items():
            Lr.append(f"| {r['method']} | {'正しい' if g == 'correct' else '間違い'} | {c['n']} | {c['p_top1_median']:.4f} ({c['p_top1_q90']:.4f}) | {c['good_mass_median']:.3f} | "
                      f"{c['good_count_median']:.0f} | {c['best_good_rank_median']:.0f} | {c['eff_candidates_median']:.0f} |")
    (OUT / 'report.md').write_text('\n'.join(Lr) + '\n'); print('\n'.join(Lr))


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if stage in ('build', 'all'): build()
    if stage in ('errors', 'all'): errors()
    if stage in ('train', 'all'): train()
    if stage in ('evaluate', 'all'): evaluate()
    if stage == 'report': report(json.loads((OUT / 'results.json').read_text()))
