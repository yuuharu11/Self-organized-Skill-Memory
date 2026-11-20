#!/usr/bin/env python3
"""KQ2-CL stage H: t0 (confidence) head and gripper decoder from key/query quantities only,
the Dev-chosen t0 policy with pi0.5 fallback, and the offline check table.

Inputs of both heads (kq2cl_common.head_features): the query vector q, the retrieved top-1 key k1,
the top-50 logits (top-1, gaps to 2/5/10/50), normalized softmax entropy, top-1 prob, top-5 mass,
raw dot products and the spread of the top-5 keys.  'scalar' = the 12 score features only,
'full' = + q + k1.  The gripper decoder additionally gets the execution-time gripper state (last
executed gripper command, has-history flag, gripper qpos).  No pi0.5 output, no target, no task id.

  train     per query in {iii, i}: training rows = eh (LIBERO-40) retrieval variants
            insE / insT / oofE / oofT / oofE+oofT (kq2cl_retrieve) x {scalar, full};
            confidence target = log arm6 of the top-1 draft; gripper target = next 10 commands.
            Selection on Dev v2 20 Hz: confidence by Spearman(ê, true err); gripper by
            missed + false switch rows with the switch threshold th.
  policy    t0 = clip(c*sqrt(ê), lo, hi) for ê <= b, pi0.5 for ê > b. (c, lo, hi) per fallback
            rate on Dev v2 (Expert curves); b = the largest Dev fallback rate whose Dev arm6 is within
            TOL of the best rate (offline MSE alone always under-uses pi0.5: CL1/EX3, see report).
            Constant t0 rows: .15 / .2 (+ the same fallback) and the Dev-best constant.
  offline   learned / unseen20 / dev2 arm6 and gripper switch metrics.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

import kq2cl_common as C

VARIANTS = ('insE', 'insT', 'oofE', 'oofT', 'oofE+oofT')
KINDS = ('scalar', 'full')
THS = (.5, .6, .7, .8, .9)
TOL = .002
B_QUANTILE = .9
QUERIES = ('iii', 'i')


def feats_for(q: str, split: str, keys: np.ndarray) -> tuple[dict, np.ndarray]:
    z = np.load(C.OUT / 'retrieval' / f'{q}__{split}.npz')
    f = C.head_features(z['q'], z['ids'], z['logits'], z['entropy'], z['top1_prob'], z['top5_mass'], keys,
                        float(z['logit_scale']))
    return f, z


def grip_data(split: str):
    D = C.KQ2 / 'data' / split
    hist = np.load(D / 'history.npy'); hm = np.load(D / 'history_mask.npy'); st = np.load(D / 'state.npy')
    tgt = np.load(D / 'actions.npy')
    last, valid = hist[:, 9, 6], hm.any(1)
    return C.grip_extra(last, valid, st[:, 6:8]), (tgt[..., 6] > 0).astype(np.float32), last, valid, tgt


def _concat(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def fit(x, y, kind, out_dim, loss, *, seed=1, epochs=30, device='cuda:0'):
    import torch
    torch.manual_seed(seed)
    m = C.make_head(kind, out_dim, extra=x.shape[1] - len(C.SCALAR_NAMES) - (512 if kind == 'full' else 0)).to(device)
    xt = torch.as_tensor(x, device=device); yt = torch.as_tensor(y, device=device)
    m.mu.copy_(xt.mean(0)); m.sd.copy_(xt.std(0).clamp_min(1e-4))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed); n = len(x)
    for _ in range(epochs):
        m.train()
        for s in range(0, n, 256):
            ix = torch.as_tensor(rng.integers(0, n, 256), device=device)
            p = m(xt[ix])
            if loss == 'reg':
                l = torch.nn.functional.smooth_l1_loss(p[:, 0], yt[ix])
            else:
                l = torch.nn.functional.binary_cross_entropy_with_logits(p, yt[ix])
            opt.zero_grad(); l.backward(); opt.step()
    return m.eval()


def predict(m, x, device='cuda:0'):
    import torch
    with torch.no_grad():
        return m(torch.as_tensor(x, device=device)).float().cpu().numpy()


def train(device: str) -> dict:
    import torch
    from scipy.stats import spearmanr
    keys = np.load(C.KEYS).astype(np.float32)
    save = C.OUT / 'heads'; save.mkdir(parents=True, exist_ok=True)
    res = C.read_json(save / 'heads.json', {})
    gx_eh, gy_eh, _, _, _ = grip_data('eh')
    gx_dev, gy_dev, last_dev, valid_dev, tgt_dev = grip_data('dev2')
    for q in QUERIES:
        if q in res and 'selected' in res[q]:
            continue
        rec = {'confidence': {}, 'gripper': {}}
        ev = {s: feats_for(q, s, keys) for s in C.EVAL}
        tr = {v: feats_for(q, f'eh_{v}', keys) for v in ('insE', 'insT', 'oofE', 'oofT')}
        for v in VARIANTS:
            parts = v.split('+')
            F = _concat([tr[p][0] for p in parts]); y = np.concatenate([np.log(tr[p][1]['err'] + 1e-6) for p in parts])
            gx = np.concatenate([gx_eh] * len(parts)); gy = np.concatenate([gy_eh] * len(parts))
            for kind in KINDS:
                name = f'{v}_{kind}'
                m = fit(C.head_input(F, kind), y, kind, 1, 'reg', device=device)
                torch.save({'kind': kind, 'state_dict': m.state_dict(), 'train': v}, save / f'{q}_conf_{name}.pt')
                r = {}
                for s in C.EVAL:
                    eh_ = np.exp(predict(m, C.head_input(ev[s][0], kind), device)[:, 0])
                    t = ev[s][1]['err']
                    r[s] = {'spearman': float(spearmanr(eh_, t).correlation), 'pred_median': float(np.median(eh_)),
                            'true_median': float(np.median(t)), 'ratio_median': float(np.median(eh_) / np.median(t)),
                            'geo_ratio': float(np.exp(np.mean(np.log(eh_ + 1e-6) - np.log(t + 1e-6))))}
                rec['confidence'][name] = r
                g = fit(C.head_input(F, kind, gx), gy, kind, 10, 'bce', device=device)
                torch.save({'kind': kind, 'state_dict': g.state_dict(), 'train': v}, save / f'{q}_grip_{name}.pt')
                pd = 1 / (1 + np.exp(-predict(g, C.head_input(ev['dev2'][0], kind, gx_dev), device)))
                gr = {}
                for th in THS:
                    dec = C.decode_gripper(pd, np.where(valid_dev, np.sign(last_dev), -1.), th)
                    sm = C.switch_metrics(dec, tgt_dev[..., 6], last_dev, valid_dev)
                    gr[str(th)] = sm | {'errors': sm['missed_rows'] + sm['false_switch_rows']}
                rec['gripper'][name] = gr
                C.log('head_fit', query=q, name=name, dev_spearman=r['dev2']['spearman'],
                      dev_grip_best=min(x['errors'] for x in gr.values()))
        cs = max(rec['confidence'], key=lambda k: rec['confidence'][k]['dev2']['spearman'])
        gsel = min(((k, th) for k in rec['gripper'] for th in rec['gripper'][k]),
                   key=lambda kt: (rec['gripper'][kt[0]][kt[1]]['errors'], -float(kt[1])))
        rec['selected'] = {'confidence': cs, 'gripper': gsel[0], 'gripper_th': float(gsel[1]),
                           'dev_gt_switch_rows': rec['gripper'][gsel[0]][gsel[1]]['gt_switch_rows']}
        res[q] = rec; C.write_json(save / 'heads.json', res)
        C.log('heads_selected', query=q, **rec['selected'])
    predict_all(device)
    return res


def predict_all(device: str) -> None:
    """ê and gripper decoder outputs of the selected heads on every Eval split -> heads/pred_<q>.npz"""
    import torch
    keys = np.load(C.KEYS).astype(np.float32)
    res = C.read_json(C.OUT / 'heads/heads.json')
    for q in QUERIES:
        sel = res[q]['selected']
        cb = torch.load(C.OUT / 'heads' / f"{q}_conf_{sel['confidence']}.pt", map_location=device, weights_only=False)
        cm = C.make_head(cb['kind'], 1).to(device); cm.load_state_dict(cb['state_dict']); cm.eval()
        gb = torch.load(C.OUT / 'heads' / f"{q}_grip_{sel['gripper']}.pt", map_location=device, weights_only=False)
        gm = C.make_head(gb['kind'], 10, extra=4).to(device); gm.load_state_dict(gb['state_dict']); gm.eval()
        out = {}
        for s in C.EVAL + ('eh_oofT',):
            f, z = feats_for(q, s, keys)
            out[f'{s}__ehat'] = np.exp(predict(cm, C.head_input(f, cb['kind']), device)[:, 0])
            out[f'{s}__true'] = z['err']
            if s in C.EVAL:
                gx, _, last, valid, _ = grip_data(s)
                p = 1 / (1 + np.exp(-predict(gm, C.head_input(f, gb['kind'], gx), device)))
                out[f'{s}__pclose'] = p
                out[f'{s}__grip'] = C.decode_gripper(p, np.where(valid, np.sign(last), -1.), sel['gripper_th'])
        np.savez_compressed(C.OUT / 'heads' / f'pred_{q}.npz', **out)
        # bundle for the server
        torch.save({'confidence': cb, 'gripper': gb, 'gripper_th': sel['gripper_th'], 'selected': sel},
                   C.OUT / 'heads' / f'bundle_{q}.pt')
        C.log('heads_predicted', query=q)


# ---------------------------------------------------------------------------
# t0 policy on Dev v2
# ---------------------------------------------------------------------------

CS = (.25, .5, .75, 1., 1.25, 1.5, 2., 2.5, 3.)
LOS = (.02, .05, .1, .15)
HIS = (.2, .25, .3, .4, .6)
FALLBACK_RATES = (0., .05, .1, .2, .3, .4, .5)
CONSTS = (.1, .15, .2, .25, .3)


def policy_value(curve, pi05, ehat, c, lo, hi, b):
    t0 = np.clip(c * np.sqrt(ehat), lo, hi)
    v = C.interp_curve(curve, C.T0_GRID, t0)
    fb = ehat > b
    return np.where(fb, pi05, v), t0, fb


def policy() -> dict:
    dev_pi = C.arm6(np.load(C.OUT / 'pi05/dev2.npy'), np.load(C.KQ2 / 'data/dev2/actions.npy'))
    cur = C.load_curves('dev2')
    res = {'tol': TOL, 'dev_pi05_arm6': float(dev_pi.mean()), 'queries': {}}
    for q in QUERIES:
        P = np.load(C.OUT / 'heads' / f'pred_{q}.npz')
        eh = P['dev2__ehat']; curve = cur[f'K__{q}__curve']
        rows = []
        for f in FALLBACK_RATES:
            b = float(np.quantile(eh, 1 - f)) if f > 0 else float('inf')
            best = None
            for c in CS:
                for lo in LOS:
                    for hi in HIS:
                        if hi <= lo:
                            continue
                        v, _, fb = policy_value(curve, dev_pi, eh, c, lo, hi, b)
                        m = float(v.mean())
                        if best is None or m < best['dev_arm6'] - 1e-9:
                            best = {'fallback_rate_target': f, 'b': b, 'c': c, 'lo': lo, 'hi': hi, 'dev_arm6': m,
                                    'dev_fallback_rate': float(fb.mean())}
            consts = {}
            for t in CONSTS:
                v, _, fb = policy_value(curve, dev_pi, eh, 0., t, t + 1e-6, b)
                consts[str(t)] = float(v.mean())
            best['const'] = consts
            rows.append(best)
        mn = min(r['dev_arm6'] for r in rows)
        tol_pick = max([r for r in rows if r['dev_arm6'] <= mn + TOL], key=lambda r: r['fallback_rate_target'])
        # b rule (used): fallback when ê exceeds the 90th percentile of ê on LIBERO-40 rows retrieved by a
        # task-held-out query (eh_oofT) = the retriever is less confident than on 90% of in-distribution queries.
        # Offline MSE cannot set b: pi0.5's offline arm6 (Dev .123) is far worse than any draft although its
        # closed-loop success equals the memory runs (CL1 32 vs 34), so every MSE rule gives b = inf.
        b = float(np.quantile(P['eh_oofT__ehat'], B_QUANTILE))
        best = None
        for c in CS:
            for lo in LOS:
                for hi in HIS:
                    if hi <= lo:
                        continue
                    v, _, fb = policy_value(curve, dev_pi, eh, c, lo, hi, b)
                    m = float(v.mean())
                    if best is None or m < best['dev_arm6'] - 1e-9:
                        best = {'fallback_rate_target': None, 'b_rule': f'quantile {B_QUANTILE} of ê on eh_oofT', 'b': b, 'c': c, 'lo': lo,
                                'hi': hi, 'dev_arm6': m, 'dev_fallback_rate': float(fb.mean())}
        best['const'] = {str(t): float(policy_value(curve, dev_pi, eh, 0., t, t + 1e-6, b)[0].mean()) for t in CONSTS}
        best['const_no_fallback'] = {str(t): float(C.interp_curve(curve, C.T0_GRID, np.full(len(curve), t)).mean()) for t in CONSTS}
        pick = best
        cbest = min((.15, .2), key=lambda t: pick['const'][str(t)])
        res['queries'][q] = {'by_fallback_rate': rows, 'selected': pick, 'mse_tolerance_pick': tol_pick, 'mse_only_best': min(rows, key=lambda r: r['dev_arm6']),
                             'const_t0_selected': float(cbest),
                             'note': 'const_t0_selected = Dev-better of .15/.2 under the same fallback b'}
        C.log('policy_selected', query=q, sel={k: pick[k] for k in ('b', 'c', 'lo', 'hi', 'dev_arm6', 'dev_fallback_rate')},
              const=float(cbest))
    sel = C.read_json(C.OUT / 'expert/select.json')
    res['true_error_map'] = {'c': sel['selected']['scale'], 'lo': .05, 'hi': .6, 'source': 'expert/select.json (Dev v2)'}
    C.write_json(C.OUT / 'policy.json', res)
    return res


# ---------------------------------------------------------------------------
# offline check table
# ---------------------------------------------------------------------------

def offline() -> dict:
    pol = C.read_json(C.OUT / 'policy.json')
    sel_ex = C.read_json(C.OUT / 'expert/select.json')
    units = np.load(C.ROOT / 'key-motion-01/final/data/bank_actions.npy').astype(np.float32)
    res = {'splits': {}}
    for split in C.EVAL:
        D = C.KQ2 / 'data' / split
        tgt = np.load(D / 'actions.npy'); task = np.load(D / 'task.npy'); epi = np.load(D / 'episode.npy')
        grp = task.astype(np.int64) * 10_000_000 + epi
        pi = np.load(C.OUT / 'pi05' / f'{split}.npy'); pi_arm = C.arm6(pi, tgt)
        cur = C.load_curves(split)
        gx, _, last, valid, _ = grip_data(split)
        hl = C.hold_last_gripper(last, valid)
        rows = {'pi05': pi_arm}
        grips = {'hold_last': hl, 'pi05': pi[..., 6]}
        extra = {}
        for q in QUERIES:
            P = np.load(C.OUT / 'heads' / f'pred_{q}.npz'); pq = pol['queries'][q]['selected']
            eh, tr = P[f'{split}__ehat'], P[f'{split}__true']
            rows[f'{q}:draft'] = cur[f'draft__{q}']
            for tag in ('K', 'EX3'):
                curve = cur[f'{tag}__{q}__curve']
                v, t0, fb = policy_value(curve, pi_arm, eh, pq['c'], pq['lo'], pq['hi'], pq['b'])
                rows[f'{q}:{tag}:head_t0+fallback'] = v
                extra[f'{q}:{tag}:head_t0+fallback'] = {'t0': C.quantiles(t0[~fb]) if (~fb).any() else {}, 'fallback_rate': float(fb.mean())}
                v2, _, _ = policy_value(curve, pi_arm, eh, pq['c'], pq['lo'], pq['hi'], float('inf'))
                rows[f'{q}:{tag}:head_t0'] = v2
                for t in (.15, .2):
                    rows[f'{q}:{tag}:const{t}'] = C.interp_curve(curve, C.T0_GRID, np.full(len(curve), t))
                    vv, _, _ = policy_value(curve, pi_arm, eh, 0., t, t + 1e-6, pq['b'])
                    rows[f'{q}:{tag}:const{t}+fallback'] = vv
                c_true = sel_ex['selected']['scale'] if tag == 'K' else sel_ex['ex3_reference']['scale']
                rows[f'{q}:{tag}:true_err_t0'] = C.interp_curve(curve, C.T0_GRID, C.T0Map(c_true, .05, .6)(tr))
                rows[f'{q}:{tag}:best_t0_per_row'] = curve.min(1)
            # grippers
            grips[f'{q}:decoder'] = P[f'{split}__grip']
            z = np.load(C.OUT / 'retrieval' / f'{q}__{split}.npz')
            grips[f'{q}:unit_stored'] = units[z['ids'][:, 0], :, 6]
            j = C.T0_GRID.index(.2)
            grips[f'{q}:EX3_expert@t0.2'] = cur[f'EX3__{q}__grip'][:, j].astype(np.float32)
            from scipy.stats import spearmanr
            extra[f'{q}:confidence'] = {'spearman': float(spearmanr(eh, tr).correlation), 'pred_median': float(np.median(eh)),
                                        'true_median': float(np.median(tr)), 'ratio_median': float(np.median(eh) / np.median(tr)),
                                        'geo_ratio': float(np.exp(np.mean(np.log(eh + 1e-6) - np.log(tr + 1e-6))))}
        rows['oracle:draft'] = cur['draft__oracle']
        rows['oracle:K:true_err_t0'] = C.interp_curve(cur['K__oracle__curve'], C.T0_GRID,
                                                      C.T0Map(sel_ex['selected']['scale'], .05, .6)(cur['draft__oracle']))
        met = {k: {'arm6': float(v.mean()), 'ci95': C.bootstrap_ci(v, grp)} for k, v in rows.items()}
        paired = {}
        for q in QUERIES:
            for k in (f'{q}:K:head_t0+fallback', f'{q}:K:const0.15', f'{q}:K:const0.2', f'{q}:K:head_t0'):
                d = rows[k] - rows[f'{q}:draft']
                paired[f'{k} - {q}:draft'] = {'mean': float(d.mean()), 'ci95': C.bootstrap_ci(d, grp)}
                d = rows[k] - rows['pi05']
                paired[f'{k} - pi05'] = {'mean': float(d.mean()), 'ci95': C.bootstrap_ci(d, grp)}
            d = rows[f'{q}:K:const0.2'] - rows[f'{q}:EX3:const0.2']
            paired[f'{q}:K:const0.2 - {q}:EX3:const0.2'] = {'mean': float(d.mean()), 'ci95': C.bootstrap_ci(d, grp)}
        d = rows['iii:K:head_t0+fallback'] - rows['i:K:head_t0+fallback']
        paired['iii:K:head_t0+fallback - i:K:head_t0+fallback'] = {'mean': float(d.mean()), 'ci95': C.bootstrap_ci(d, grp)}
        gm = {k: C.switch_metrics(v, tgt[..., 6], last, valid) for k, v in grips.items()}
        res['splits'][split] = {'rows': int(len(tgt)), 'arm6': met, 'paired': paired, 'gripper': gm, 'extra': extra}
        C.log('offline_split', split=split, **{k: round(v['arm6'], 4) for k, v in met.items() if ':K:' in k or k == 'pi05'})
    C.write_json(C.OUT / 'offline.json', res)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=('train', 'predict', 'policy', 'offline'))
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
    C.log('start', stage='heads.' + a.cmd)
    {'train': lambda: train(a.device), 'predict': lambda: predict_all(a.device), 'policy': policy, 'offline': offline}[a.cmd]()
    C.log('end', stage='heads.' + a.cmd)


if __name__ == '__main__':
    main()
