"""Memory pools of whole DCT-3 units with/without τ medoid clustering, plus a K1-style history query per pool.

Pool: LIBERO-40 memory demos (400) cut at speed minima + gripper switches, pose-removed DCT-3 units
('rms'), optionally merged by the reviewed τ greedy cover (medoids), holdout tasks removed.
Each candidate = one whole unit decoded at its own length and resampled to 10 frames (7 channels).
Query: K1 P5_hist architecture (history 10x7 + state -> MLP; key MLP over the 70-d shape), trained on
LIBERO-40 train rows with the soft listwise teacher (own task excluded), selected on the holdout tasks.
Unseen eval targets are used only for the offline scoring table.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import os
STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', Path(__file__).resolve().parents[5] / '.skillnet-state')).expanduser()

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = (STATE_DIR / 'outputs/memory-pilot/unseen-libero')
OUT = ROOT / 'cluster-pools-01'
HOLD = (2, 7, 12, 17, 22, 27, 32, 37)
TAUS = {'unit_none': None, 'unit_04': .04, 'unit_08': .08, 'unit_16': .16}
SEED = 4101


def resample(x, n=10):
    x = np.asarray(x, np.float64); t = np.linspace(0, len(x) - 1, n)
    return np.stack([np.interp(t, np.arange(len(x)), x[:, j]) for j in range(x.shape[1])], 1)


def build_pool(name):
    import compressed_denoise as cd
    R = cd.RUN / 'rows'; ld = lambda k: np.asarray(np.load(R / f'{k}.npy', mmap_mode='r'))
    rows = {'task': ld('task'), 'episode': ld('episode'), 'frame': ld('frame')}
    mem_ids = np.asarray(np.load(cd.RUN / 'memory_row_ids.npy'), np.int64)
    bank = cd._bank_from_memory(rows, mem_ids, mode='variable', form='rms', tau=TAUS[name])
    dec = cd._bank_decode(bank); keep = ~np.isin(bank['task'], HOLD)
    shapes = np.stack([resample(d) for d in dec]).astype(np.float32)
    np.savez(OUT / f'pool_{name}.npz', shape=shapes[keep], source_task=np.asarray(bank['task'])[keep],
             length=np.asarray(bank['memory'].lengths)[keep], raw_units=bank['raw_unit_count'], units=bank['unit_count'])
    print(name, 'units', bank['unit_count'], 'kept', int(keep.sum()), flush=True)


def train_query(name, device='cuda', epochs=100, tau=.01):
    import torch, key_query_probe as kq, variable_shape_memory as vs
    p = np.load(OUT / f'pool_{name}.npz'); S = p['shape']; ptask = p['source_task']
    P = np.load(ROOT / 'key-query-03' / 'prep.npz')  # train hist/state/task (same rows as K1)
    base, rows, unseen, splits = kq._production_data_splits(include_eval_targets=False)
    ids = splits['train'][1]
    tgt = kq._motion_shape_from_actions(np.asarray(rows['actions'][ids, :, :7], np.float32), vs)
    E = kq.shape_error_matrix(tgt, S, vs, batch=512, device=device).astype(np.float32)
    A = ptask[None, :] != P['train_task'][:, None]
    hold = np.isin(P['train_task'], HOLD); fit = np.flatnonzero(~hold); va = np.flatnonzero(hold)
    X = torch.as_tensor(np.concatenate([P['train_hist'].reshape(len(ids), -1), P['train_state']], 1), device=device, dtype=torch.float32)
    K = torch.as_tensor(S.reshape(len(S), -1), device=device); At = torch.as_tensor(A, device=device)
    Et = torch.as_tensor(np.where(A, E, 1e3), device=device)
    teacher = torch.softmax(-(Et - Et.min(1, keepdim=True).values) / tau, -1) * At; teacher /= teacher.sum(1, keepdim=True)
    torch.manual_seed(SEED); net = kq.ProbeNetwork('P5_hist').module.to(device)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs * ((len(fit) + 255) // 256), pct_start=.05)
    best = (np.inf, None, 0)
    for ep in range(epochs):
        net.train(); perm = np.random.default_rng(SEED + ep).permutation(fit)
        for i in range(0, len(perm), 256):
            ix = perm[i:i + 256]
            q = torch.nn.functional.normalize(net.query(X[ix]), dim=-1); k = torch.nn.functional.normalize(net.key(K), dim=-1)
            logit = (q @ k.T / .1).masked_fill(~At[ix], -1e9)
            loss = -(teacher[ix] * torch.log_softmax(logit, -1)).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            q = torch.nn.functional.normalize(net.query(X[va]), dim=-1); k = torch.nn.functional.normalize(net.key(K), dim=-1)
            pick = (q @ k.T).masked_fill(~At[va], -1e9).argmax(-1).cpu().numpy()
        v = float(E[va, pick].mean())
        if v < best[0]: best = (v, {kk: t.detach().cpu().clone() for kk, t in net.state_dict().items()}, ep + 1)
    torch.save({'state_dict': best[1], 'val_shape': best[0], 'epoch': best[2], 'tau': tau}, OUT / f'query_{name}.pt')
    # offline unseen eval (scoring only)
    net.load_state_dict(best[1]); net.eval()
    y = np.load(kq.UNSEEN / 'actions.npy', mmap_mode='r')[splits['eval'][1], :, :7].astype(np.float32)
    Ee = kq.shape_error_matrix(kq._motion_shape_from_actions(y, vs), S, vs, batch=512, device=device)
    xe = torch.as_tensor(np.concatenate([P['eval_hist'].reshape(len(y), -1), P['eval_state']], 1), device=device, dtype=torch.float32)
    with torch.no_grad():
        pick = (torch.nn.functional.normalize(net.query(xe), dim=-1) @ torch.nn.functional.normalize(net.key(K), dim=-1).T).argmax(-1).cpu().numpy()
    n = len(Ee); orc = Ee.min(1); rnd = Ee.mean(1); se = Ee[np.arange(n), pick]
    strict = Ee <= (orc + .1 * (rnd - orc))[:, None]
    # absolute reference: same thresholds as the unclustered window pool are not comparable; report shape errors too
    rec = {'pool': name, 'candidates': len(S), 'units_total': int(p['units']), 'val_shape': best[0], 'epoch': best[2],
           'eval_shape_selected': float(se.mean()), 'eval_shape_oracle': float(orc.mean()), 'eval_shape_random': float(rnd.mean()),
           'gap': float((rnd - se).mean() / (rnd - orc).mean()), 'top1_strict': float(strict[np.arange(n), pick].mean())}
    print(json.dumps(rec), flush=True)
    return rec


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True); res = []
    for name in TAUS:
        if not (OUT / f'pool_{name}.npz').exists(): build_pool(name)
        res.append(train_query(name))
    (OUT / 'offline.json').write_text(json.dumps(res, indent=1))
