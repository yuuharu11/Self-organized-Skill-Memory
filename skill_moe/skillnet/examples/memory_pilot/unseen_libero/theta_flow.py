"""θ denoising: flow matching over the 4-d pose (axis-angle 3 + log scale 1) of a chosen motion shape.

The skill (shape) is chosen first from memory; this model only places it in the scene.
Training uses LIBERO-40 rows only (K1 holdout tasks are kept out of the fit and used
for epoch/variant selection).  Unseen eval targets are read only in ``evaluate``.

Variants (same network):
  flow    : noise -> θ (openpi time convention: t=1 noise, integrate to t=0)
  bridge  : θ_src + σ·noise -> θ
  regress : direct regression (same inputs), to isolate the effect of denoising
Stages: prepare, train, evaluate, report.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import os
STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', Path(__file__).resolve().parents[5] / '.skillnet-state')).expanduser()

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = (STATE_DIR / 'outputs/memory-pilot/unseen-libero')
OUT = ROOT / 'theta-flow-01'
K1 = ROOT / 'key-query-01'
LORA_SELECTED = Path(STATE_DIR / 'outputs/memory-pilot/unseen-libero/theta-flow-01/lora_theta_selected.npz')
TOPK = 3
SIGMA_BRIDGE = .5
STEPS = 10
METHODS = ('oracle_shape', 'P5_hist', 'P6_tok_hist', 'P1_tok', 'P2_mid_l14', 'P0_mean')
VARIANTS = ('flow', 'bridge', 'regress')


def _atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp'); tmp.write_text(json.dumps(obj, indent=1)); tmp.replace(path)


def _mods():
    import key_query_probe as kq, theta_regression as tr, variable_shape_memory as vs
    return kq, tr, vs


# ---------------------------------------------------------------- geometry
def batched_procrustes(cand, target):
    """Vectorized compressed_denoise._procrustes: cand/target [N,10,6] physical -> θ [N,4], aligned, err."""
    from scipy.spatial.transform import Rotation
    y = np.asarray(cand, np.float64); x = np.asarray(target, np.float64)
    cross = np.einsum('nti,ntj->nij', y[:, :, :3], x[:, :, :3]) + np.einsum('nti,ntj->nij', y[:, :, 3:6], x[:, :, 3:6])
    u, _, vt = np.linalg.svd(cross)
    d = np.linalg.det(u @ vt)
    corr = np.tile(np.eye(3), (len(y), 1, 1)); corr[:, 2, 2] = d
    r = u @ corr @ vt
    yrot = np.concatenate([y[:, :, :3] @ r, y[:, :, 3:6] @ r], axis=2)
    scale = np.maximum(np.sum(yrot * x, axis=(1, 2)) / np.maximum(np.sum(y * y, axis=(1, 2)), 1e-12), 1e-6)
    theta = np.concatenate([Rotation.from_matrix(r).as_rotvec(), np.log(scale)[:, None]], axis=1).astype(np.float32)
    aligned = yrot * scale[:, None, None]
    return theta, aligned, np.mean((aligned - x) ** 2, axis=(1, 2))


def decode_many(shapes, thetas, vs):
    _, tr, _ = _mods()
    return np.stack([tr.decode_theta(s, t, vs) for s, t in zip(shapes, thetas)]).astype(np.float32)


def six_axis_mse(pred, target):
    return ((np.asarray(pred)[:, :, :6] - np.asarray(target)[:, :, :6]) ** 2).mean((1, 2)) * 6 / 7


# ---------------------------------------------------------------- features
def pool_image_tokens(img128):
    """[N,128,D] (two 8x8 cameras) -> [N,32,D] (two 4x4 cameras)."""
    a = np.asarray(img128, np.float32); n, _, d = a.shape
    return a.reshape(n, 2, 4, 2, 4, 2, d).mean((3, 5)).reshape(n, 32, d)


def build_tokens(feats, split, out_path, kq):
    """35 tokens: 32 pooled image tokens + base/wrist/language grouped means. fp16 memmap, resumable."""
    n = len(feats['row_ids'])
    prog = out_path.with_suffix('.progress.json')
    if out_path.exists() and not prog.exists():
        return np.load(out_path, mmap_mode='r')
    arr = np.lib.format.open_memmap(out_path, mode='r+' if out_path.exists() else 'w+', dtype=np.float16, shape=(n, 35, 2048))
    done = json.loads(prog.read_text())['done'] if prog.exists() else 0
    store = feats['tokens']
    for st in range(done, n, 256):
        ix = np.arange(st, min(st + 256, n))
        if store.image_cache is not None:
            img = np.asarray(store.image_cache[store.cache_positions[ix]], np.float32)
        else:
            img = np.stack([kq.downsample_camera_tokens(store.tokens[int(store.row_ids[i])], store.groups[int(store.row_ids[i])],
                                                        add_position=False)[:128] for i in ix]).astype(np.float32)
        arr[ix, :32] = pool_image_tokens(img).astype(np.float16)
        arr[ix, 32:] = np.asarray(feats['mean'][ix], np.float32).reshape(len(ix), 3, 2048).astype(np.float16)
        arr.flush(); _atomic_json(prog, {'split': split, 'done': int(ix[-1] + 1), 'total': n})
    del arr; prog.unlink()
    return np.load(out_path, mmap_mode='r')


def prepare(*, device='cuda'):
    """Train examples: top-K cross-task pool shapes per LIBERO-40 row and θ* fitted to its target window."""
    kq, tr, vs = _mods()
    OUT.mkdir(parents=True, exist_ok=True)
    k1sum = json.loads((K1 / 'production' / 'evaluation' / 'summary.json').read_text())
    hold = tuple(int(t) for t in k1sum['holdout_tasks'])
    p0 = tr.candidate_shape_pool(rep='C_vm_none'); keep = ~np.isin(p0['source_task'], hold)
    pool = {k: np.asarray(v)[keep] for k, v in p0.items()}
    np.savez(OUT / 'pool.npz', **pool, holdout_tasks=np.asarray(hold))
    for split in ('train', 'eval'):
        feats = kq.assemble_split_features(split, include_tokens=True, output=K1)
        build_tokens(feats, split, OUT / f'{split}_tokens.npy', kq)
        np.savez(OUT / f'{split}_cond.npz', row_ids=feats['row_ids'], task=feats['task'], episode=feats['episode'],
                 frame=feats['frame'], history=feats['history'], state=feats['state'])
        if split != 'train':
            continue
        base, rows, _, _ = kq._production_data_splits()
        target = np.asarray(rows['actions'][feats['row_ids'], :, :7], np.float32)   # LIBERO-40 labels only
        qshape = kq._motion_shape_from_actions(target, vs)
        err = kq.shape_error_matrix(qshape, pool['shape'], vs, batch=64, device=device)
        allowed = kq.filter_cross_task(pool, feats['task'], heldout=())
        err = np.where(allowed, err, np.inf)
        top = np.argsort(err, axis=1)[:, :TOPK]
        rows_ix = np.repeat(np.arange(len(target)), TOPK); cand = top.reshape(-1)
        tphys = vs.to_physical(target[:, :, :6])
        theta, _, perr = batched_procrustes(pool['shape'][cand, :, :6], tphys[rows_ix])
        np.savez(OUT / 'train_examples.npz', row=rows_ix, cand=cand, rank=np.tile(np.arange(TOPK), len(target)),
                 theta=theta, shape_err=perr, target=target)
    _atomic_json(OUT / 'prepare.json', {'complete': True, 'holdout_tasks': list(hold), 'pool': int(keep.sum()), 'topk': TOPK,
                                       'eval_targets_used': False})


# ---------------------------------------------------------------- model
def make_net(width=256):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Sequential(nn.Linear(2048, width), nn.LayerNorm(width))
            self.tok_type = nn.Parameter(torch.zeros(35, width))
            self.ctx = nn.Sequential(nn.Linear(70 + 4 + 70 + 8, width), nn.GELU(), nn.Linear(width, width))
            self.inp = nn.Sequential(nn.Linear(4 + 64, width), nn.GELU(), nn.Linear(width, width))
            self.q = nn.Parameter(torch.randn(4, width) * .02)
            self.att1 = nn.MultiheadAttention(width, 8, batch_first=True)
            self.att2 = nn.MultiheadAttention(width, 8, batch_first=True)
            self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))
            self.out = nn.Sequential(nn.LayerNorm(4 * width), nn.Linear(4 * width, width), nn.GELU(), nn.Linear(width, 4))

        def forward(self, tokens, ctx, xt, t):
            import math
            half = 32
            freq = torch.exp(-math.log(1000.) * torch.arange(half, device=t.device) / half)
            temb = torch.cat([torch.sin(t[:, None] * freq * 100), torch.cos(t[:, None] * freq * 100)], 1)
            kv = self.tok(tokens) + self.tok_type
            h = self.q[None] + (self.ctx(ctx) + self.inp(torch.cat([xt, temb], 1)))[:, None]
            h = h + self.att1(h, kv, kv, need_weights=False)[0]
            h = h + self.att2(h, kv, kv, need_weights=False)[0]
            h = h + self.ff(h)
            return self.out(h.reshape(len(h), -1))
    return Net()


def _ctx(shape, theta_src_n, history, state):
    return np.concatenate([shape.reshape(len(shape), -1), theta_src_n, history.reshape(len(history), -1), state], 1).astype(np.float32)


def _row_noise(row_ids, salt=0):
    return np.stack([np.random.default_rng(4101 + 7919 * int(r) + salt).standard_normal(4) for r in row_ids]).astype(np.float32)


def sample_theta(net, variant, tokens, ctx, theta_src_n, noise, device):
    """Single θ sample per row (never averages samples). Returns normalized θ."""
    import torch
    net.eval(); out = []
    with torch.no_grad():
        for st in range(0, len(ctx), 512):
            sl = slice(st, st + 512)
            tk = torch.as_tensor(np.asarray(tokens[sl], np.float32), device=device)
            c = torch.as_tensor(ctx[sl], device=device); e = torch.as_tensor(noise[sl], device=device)
            if variant == 'regress':
                out.append(net(tk, c, torch.zeros_like(e), torch.zeros(len(e), device=device)).cpu().numpy()); continue
            x = e if variant == 'flow' else torch.as_tensor(theta_src_n[sl], device=device) + SIGMA_BRIDGE * e
            for i in range(STEPS):
                t = torch.full((len(x),), 1 - i / STEPS, device=device)
                x = x - net(tk, c, x, t) / STEPS
            out.append(x.cpu().numpy())
    return np.concatenate(out)


def train(*, variant, epochs=30, device='cuda', batch=256, lr=3e-4, threads=8):
    import torch
    torch.set_num_threads(threads); torch.manual_seed(20261109)
    kq, tr, vs = _mods()
    vdir = OUT / 'models' / variant; vdir.mkdir(parents=True, exist_ok=True)
    if (vdir / 'trained.json').exists():
        return json.loads((vdir / 'trained.json').read_text())
    pool = np.load(OUT / 'pool.npz'); ex = np.load(OUT / 'train_examples.npz'); cond = np.load(OUT / 'train_cond.npz')
    tokens = np.load(OUT / 'train_tokens.npy', mmap_mode='r')
    hold = pool['holdout_tasks']; task = cond['task'][ex['row']]
    fit = ~np.isin(task, hold); val = np.isin(task, hold) & (ex['rank'] == 0)
    mu = ex['theta'][fit].mean(0); sd = ex['theta'][fit].std(0) + 1e-6
    tsrc = pool['source_theta'][ex['cand']]; tsrc_n = (tsrc - mu) / sd
    ctx = _ctx(pool['shape'][ex['cand']], tsrc_n, cond['history'][ex['row']], cond['state'][ex['row']])
    y = ((ex['theta'] - mu) / sd).astype(np.float32)
    net = make_net().to(device); opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    fit_ix = np.flatnonzero(fit); val_ix = np.flatnonzero(val)
    val_tok = np.asarray(tokens[ex['row'][val_ix]], np.float32)
    val_noise = _row_noise(cond['row_ids'][ex['row'][val_ix]])
    val_target = ex['target'][ex['row'][val_ix]]; val_shape = pool['shape'][ex['cand'][val_ix]]
    rng = np.random.default_rng(20261109); hist = []; best = None
    tok_rows = ex['row']
    for ep in range(epochs):
        net.train(); rng.shuffle(fit_ix); t0 = time.time(); losses = []
        for st in range(0, len(fit_ix), batch):
            b = fit_ix[st:st + batch]
            order = np.argsort(tok_rows[b]); b = b[order]
            tk = torch.as_tensor(np.asarray(tokens[tok_rows[b]], np.float32), device=device)
            c = torch.as_tensor(ctx[b], device=device); yy = torch.as_tensor(y[b], device=device)
            e = torch.randn_like(yy)
            if variant == 'regress':
                pred = net(tk, c, torch.zeros_like(yy), torch.zeros(len(yy), device=device)); loss = ((pred - yy) ** 2).mean()
            else:
                t = torch.rand(len(yy), device=device)
                src = e if variant == 'flow' else torch.as_tensor(tsrc_n[b], device=device) + SIGMA_BRIDGE * e
                xt = (1 - t[:, None]) * yy + t[:, None] * src
                loss = ((net(tk, c, xt, t) - (src - yy)) ** 2).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.); opt.step()
            losses.append(float(loss))
        if (ep + 1) % 2 == 0 or ep + 1 == epochs:
            th = sample_theta(net, variant, val_tok, ctx[val_ix], tsrc_n[val_ix], val_noise, device) * sd + mu
            score = float(six_axis_mse(decode_many(val_shape, th, vs), val_target).mean())
            rec = {'epoch': ep + 1, 'loss': float(np.mean(losses)), 'holdout_six_axis': score, 'sec': time.time() - t0}
            hist.append(rec); print(json.dumps({'variant': variant, **rec}), flush=True)
            if best is None or score < best['holdout_six_axis']:
                best = rec; torch.save(net.state_dict(), vdir / 'best.pt')
    res = {'variant': variant, 'best': best, 'history': hist, 'mu': mu.tolist(), 'sd': sd.tolist(),
           'fit_examples': int(fit.sum()), 'holdout_rows': int(val.sum()), 'selection': 'K1 LIBERO-40 holdout tasks, top-1 candidate six-axis MSE'}
    _atomic_json(vdir / 'trained.json', res)
    return res


# ---------------------------------------------------------------- evaluation
def gripper_g1(cond, state_thr):
    """Hold last executed gripper command; at episode start use the observed state (threshold from LIBERO-40)."""
    h = cond['history']; st = cond['state']; first = cond['frame'] == 0
    cur = np.where(first, np.where(st[:, 6] > state_thr, -1., 1.), h[:, -1, 6])
    return cur


def with_gripper(actions, cur):
    """Replace the shape's gripper channel with the G1 command (draft used for later Expert refinement)."""
    a = np.asarray(actions, np.float32)[:, :, :7].copy(); a[:, :, 6] = np.asarray(cur, np.float32)[:, None]
    return a


def bootstrap_diff(a, b, episodes, reps=5000, seed=4101):
    a = np.asarray(a); b = np.asarray(b); keys = np.unique(episodes); rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(episodes == k) for k in keys]; out = []
    for _ in range(reps):
        ix = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))]); out.append((a[ix] - b[ix]).mean())
    return [float(np.mean(a - b)), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def evaluate(*, device='cuda'):
    import torch
    kq, tr, vs = _mods()
    pool = np.load(OUT / 'pool.npz'); cond = dict(np.load(OUT / 'eval_cond.npz')); tokens = np.load(OUT / 'eval_tokens.npy', mmap_mode='r')
    z = np.load(K1 / 'production' / 'evaluation' / 'rows.npz')
    if not np.array_equal(z['row_ids'], cond['row_ids']):
        raise ValueError('K1 eval rows and θ-flow eval rows are misaligned')
    # ---- scorer-only reads below
    y = z['target'][:, :, :7]; hard = z['hard'].astype(bool); episodes = z['task'].astype(np.int64) * 100000 + z['episode']
    err = np.where(z['allowed'], z['shape_error'], np.inf)
    picks = {'oracle_shape': err.argmin(1)}
    for k in METHODS[1:]:
        picks[k] = z['rank_' + k][:, 0]
    # gripper (no targets): threshold from LIBERO-40 training rows
    _, rows, _, _ = kq._production_data_splits()
    trc = np.load(OUT / 'train_cond.npz'); tra = np.asarray(rows['actions'][trc['row_ids'], 0, 6])
    thr = float((trc['state'][tra < 0, 6].mean() + trc['state'][tra > 0, 6].mean()) / 2)
    cur = gripper_g1(cond, thr)
    G1 = ((cur[:, None] - y[:, :, 6]) ** 2).mean(1) / 7
    tphys = vs.to_physical(y[:, :, :6]); pi = z['query_pi05'][:, :, :7]; piphys = vs.to_physical(pi[:, :, :6])
    normal = ((pi - y) ** 2).mean((1, 2))
    lora = np.load(LORA_SELECTED) if LORA_SELECTED.exists() else None
    res = {'gripper_threshold': thr, 'normal_pi05': {'all': float(normal.mean()), 'hard': float(normal[hard].mean())}, 'methods': {}}
    per_row = {}; drafts = {}
    for variant in VARIANTS:
        tj = json.loads((OUT / 'models' / variant / 'trained.json').read_text()); mu = np.asarray(tj['mu'], np.float32); sd = np.asarray(tj['sd'], np.float32)
        net = make_net().to(device); net.load_state_dict(torch.load(OUT / 'models' / variant / 'best.pt', map_location=device))
        for name, pick in picks.items():
            shape = pool['shape'][pick]; tsrc_n = (pool['source_theta'][pick] - mu) / sd
            ctx = _ctx(shape, tsrc_n, cond['history'], cond['state'])
            for nz, noise in (('seed', _row_noise(cond['row_ids'])), ('zero', np.zeros((len(pick), 4), np.float32))):
                if variant == 'regress' and nz == 'zero':
                    continue
                th = sample_theta(net, variant, tokens, ctx, tsrc_n, noise, device) * sd + mu
                act = decode_many(shape, th, vs)
                total = six_axis_mse(act, y) + G1
                per_row[(variant, nz, name)] = total; drafts[(variant, nz, name)] = with_gripper(act, cur)
    for name, pick in picks.items():
        shape = pool['shape'][pick]
        base = {'theta_star': batched_procrustes(shape[:, :, :6], tphys)[0], 'theta_pi05': batched_procrustes(shape[:, :, :6], piphys)[0],
                'theta_src': pool['source_theta'][pick]}
        for k, th in base.items():
            act = decode_many(shape, th, vs)
            per_row[(k, '-', name)] = six_axis_mse(act, y) + G1; drafts[(k, '-', name)] = with_gripper(act, cur)
        if lora is not None and name in lora.files:
            per_row[('lora', '-', name)] = six_axis_mse(lora[name], y) + G1; drafts[('lora', '-', name)] = with_gripper(lora[name], cur)
    for (v, nz, name), tot in per_row.items():
        rec = {'all': float(tot.mean()), 'hard': float(tot[hard].mean()), 'vs_normal_pi05': bootstrap_diff(tot, normal, episodes)}
        if ('lora', '-', name) in per_row and v != 'lora':
            rec['vs_lora'] = bootstrap_diff(tot, per_row[('lora', '-', name)], episodes)
        res['methods'][f'{v}|{nz}|{name}'] = rec
    res['G1_gripper'] = {'all': float(G1.mean()), 'hard': float(G1[hard].mean())}
    _atomic_json(OUT / 'results.json', res)
    np.savez(OUT / 'eval_drafts.npz', **{f'{v}|{nz}|{n}': d for (v, nz, n), d in drafts.items()}, row_ids=cond['row_ids'])
    np.savez(OUT / 'eval_rows.npz', **{f'{v}|{nz}|{n}': t for (v, nz, n), t in per_row.items()}, row_ids=cond['row_ids'])
    return res


def report():
    r = json.loads((OUT / 'results.json').read_text())
    lines = ['# θ denoising (flow matching over pose) — unseen LIBERO-90 eval', '',
             'Normalized active-7 MSE = six-axis MSE of the chosen shape decoded with θ + G1 gripper (hold last executed command).',
             f"Normal π0.5: {r['normal_pi05']['all']:.4f} / {r['normal_pi05']['hard']:.4f}. G1 gripper part: {r['G1_gripper']['all']:.4f} / {r['G1_gripper']['hard']:.4f}.",
             'θ_star is a scorer-side upper bound, not an inference method. Unseen targets are used only for scoring.', '',
             '| θ source | noise | shape choice | all | hard | − normal π0.5 [95%] | − LoRA θ [95%] |', '|---|---|---|---:|---:|---:|---:|']
    for k, v in sorted(r['methods'].items(), key=lambda kv: kv[1]['all']):
        a, b, c = v['vs_normal_pi05']; l = v.get('vs_lora')
        ls = f"{l[0]:+.4f} [{l[1]:+.4f}, {l[2]:+.4f}]" if l else '—'
        th, nz, nm = k.split('|')
        lines.append(f"| {th} | {nz} | {nm} | {v['all']:.4f} | {v['hard']:.4f} | {a:+.4f} [{b:+.4f}, {c:+.4f}] | {ls} |")
    for variant in VARIANTS:
        p = OUT / 'models' / variant / 'trained.json'
        if p.exists():
            t = json.loads(p.read_text()); lines.append(f"\n{variant}: selected epoch {t['best']['epoch']}, holdout six-axis {t['best']['holdout_six_axis']:.4f}")
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=('prepare', 'train', 'evaluate', 'report', 'all'), default='all')
    ap.add_argument('--variant', choices=VARIANTS)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    if a.stage in ('prepare', 'all') and not (OUT / 'prepare.json').exists():
        prepare(device=a.device)
    if a.stage in ('train', 'all'):
        for v in ([a.variant] if a.variant else VARIANTS):
            train(variant=v, epochs=a.epochs, device=a.device)
    if a.stage in ('evaluate', 'all'):
        evaluate(device=a.device)
    if a.stage in ('report', 'all'):
        report()


if __name__ == '__main__':
    main()
