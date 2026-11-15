"""RM-OOF: episode-wise cross-fitted drafts and source-continuation diagnostic.

Writes only to expert-04/prep.  All query/candidate identities remain dataset
row IDs; an outer held-out fold is removed from model fitting and retrieval.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import experiment as ex
import expert
import expert2_prep as prep
import magnitude_head
import gripper

ROOT = Path('/work/.skillnet-state/outputs/robocasa-memory')
SOURCE = ROOT / 'memory-01/policy-order-v2'
CTX = ROOT / 'ctx-01'
OUT = ROOT / 'expert-04/prep'
KEYLITE = ROOT / 'key-lite-01'
SEED = 20261113


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=ex._json_default) + '\n')
    tmp.replace(path)


def progress(stage, **kw):
    rec = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'stage': stage, **kw}
    with (OUT / 'progress.jsonl').open('a') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(json.dumps(rec, ensure_ascii=False), flush=True)


def load_inputs():
    import torch
    rk = prep._load_ranker()
    rk.DEVICE = 'cpu'
    # Do not write progress into key-lite-01: that directory is read-only here.
    rk.log = lambda stage, **kw: print(json.dumps({'stage': stage, **kw}), flush=True)
    zf = np.load(CTX / 'cache/rows.npz', allow_pickle=False)
    z = {k: np.asarray(zf[k]) for k in zf.files}
    poolf = np.load(CTX / 'cache/pool.npz', allow_pickle=False)
    pool = {k: np.asarray(poolf[k]) for k in poolf.files}
    feat = np.asarray(np.load(CTX / 'cache/prefix_mean_fp16.npy', mmap_mode='r'), np.float32)
    teacherf = np.load(CTX / 'cache/train_teacher_final.npz', allow_pickle=False)
    teacher = {k: np.asarray(teacherf[k]) for k in teacherf.files}
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    assert len(manifest['task_splits']) == 6
    assert all(len(x['memory_episodes']) == 80 for x in manifest['task_splits'])
    assert len(z['memory_task']) == 11314 and len(pool['task']) == 7062
    return rk, z, pool, teacher, feat, manifest


def make_folds(manifest):
    rng = np.random.default_rng(SEED)
    fold_of = {}
    fold_eps = [[] for _ in range(5)]
    for task in range(6):
        eps = np.asarray(manifest['task_splits'][task]['memory_episodes'], np.int64)
        assert len(eps) == 80
        eps = eps[rng.permutation(len(eps))]
        for j, ep in enumerate(eps):
            fi = j % 5
            ge = task * 1000 + int(ep)
            fold_of[ge] = fi
            fold_eps[fi].append([task, int(ep)])
    assert all(len(fold_eps[i]) == 96 for i in range(5))
    return fold_of, fold_eps


def subset_pool(pool, ids):
    ids = np.asarray(ids, np.int64)
    out = {k: np.asarray(v)[ids] for k, v in pool.items()
           if k not in ('actions_flat', 'action_offsets') and len(np.asarray(v)) == len(pool['task'])}
    rows = [np.asarray(pool['actions_flat'][pool['action_offsets'][i]:pool['action_offsets'][i + 1]], np.float32)
            for i in ids]
    offsets = np.zeros(len(rows) + 1, np.int64)
    offsets[1:] = np.cumsum([len(x) for x in rows])
    out['action_offsets'] = offsets
    out['actions_flat'] = np.concatenate(rows, axis=0).astype(np.float32)
    return out


def subset_z(z, qids, cids):
    qids = np.asarray(qids, np.int64); cids = np.asarray(cids, np.int64)
    out = {}
    for k, v in z.items():
        if k.startswith('memory_'):
            out[k] = np.asarray(v)[qids]
        elif k.startswith('candidate_'):
            out[k] = np.asarray(v)[cids]
    # run_key_lite.train_ranker computes an unused evaluation context even for
    # history-aware fitting. Supply an empty, label-free split so no heldout
    # query can enter fitting or model selection.
    out['heldout_context'] = np.zeros((0,), np.int32)
    out['heldout_state'] = np.zeros((0, 16), np.float32)
    return out


def make_history(rk, z):
    return rk.build_history_rows(z, 'memory')


def fit_ranker(rk, z, pool, teacher, feat, qids, cids, fold_dir, fold):
    import torch
    zz = subset_z(z, qids, cids)
    pp = subset_pool(pool, cids)
    err = np.asarray(teacher['error'][qids][:, cids], np.float32)
    allowed = np.asarray(teacher['allowed'][qids][:, cids], bool).copy()
    # Reassert strict outer-fold and query-episode exclusions (no val/test fold
    # row can act as a candidate for this fitted model).
    allowed &= np.asarray(z['memory_episode'])[qids, None] != np.asarray(pool['global_episode'])[cids][None, :]
    if not allowed.any(1).all():
        raise AssertionError('a fold query has no allowed training-fold candidate')
    tc = {'error': err, 'allowed': allowed}
    hist = make_history(rk, zz)
    # The established key has an episode-disjoint early-stop split internal to
    # the 4-fold fit set. The chosen checkpoint is restored explicitly.
    net, ck = rk.train_ranker(zz, feat, pp, tc, hist, input_history=True,
                              tag=f'oof_fold_{fold}', max_epochs=12)
    net.load_state_dict(ck['state']); net.eval()
    fold_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'state': ck['state'], 'norm': ck['norm'], 'hist_norm': ck['hist_norm'],
                'q_dim': ck['q_dim'], 'k_dim': ck['k_dim'], 'input_history': True,
                'tau': ck['tau'], 'best_epoch': ck['best_epoch'],
                'dev_top1_type_error': ck['dev_top1_type_error'], 'dev_episodes': ck['dev_episodes'],
                'curve': ck['curve'], 'fit_query_count': len(qids), 'candidate_count': len(cids),
                'excluded_outer_fold': int(fold)}, fold_dir / 'key.pt')
    return zz, pp, net, ck


def rank_context(rk, zpart, feat, ck):
    qctx, cctx = rk.assemble_context(zpart, feat, ck['norm'], 'memory')
    return np.asarray(qctx, np.float32), np.asarray(cctx, np.float32)


def get_scoring_tensors(net, cctx):
    import torch
    with torch.no_grad():
        ct = torch.as_tensor(cctx, dtype=torch.float32)
        kemb = torch.nn.functional.normalize(net.key(ct), dim=-1)
    return kemb


def score_rows(net, ck, qctx, hist, mask, q_episode, kemb, candidate_episode, *, topk=1):
    import torch
    hn = ((np.asarray(hist, np.float32) - np.asarray(ck['hist_norm']['mean'], np.float32)) /
          np.asarray(ck['hist_norm']['std'], np.float32)) * np.asarray(mask, bool)[..., None]
    q = np.concatenate([qctx, hn.reshape(len(hist), -1), np.asarray(mask, np.float32)], axis=1).astype(np.float32)
    with torch.no_grad():
        qt = torch.as_tensor(q, dtype=torch.float32)
        qe = torch.nn.functional.normalize(net.query(qt), dim=-1)
        scores = (qe @ kemb.T) * net.log_scale.exp()
    s = scores.numpy()
    s[np.asarray(q_episode)[:, None] == np.asarray(candidate_episode)[None, :]] = -np.inf
    if topk == 1:
        return s.argmax(axis=1), s
    return s, s


def unit_meta(pool):
    return magnitude_head._unit_metadata(pool)[0].astype(np.float32)


def fit_magnitude(rk, z, pool, feat, qids, cids, net, ck, qctx, cctx, fold_dir, fold):
    import torch
    from magnitude_head import _MLP, _fit, _unit_metadata
    qhist = make_history(rk, subset_z(z, qids, cids))
    picks, _ = score_rows(net, ck, qctx, qhist['hist'], qhist['mask'],
                          np.asarray(z['memory_episode'])[qids],
                          get_scoring_tensors(net, cctx), np.asarray(pool['global_episode'])[cids])
    pp = subset_pool(pool, cids)
    umeta, _ = _unit_metadata(pp)
    # Context-only input is the v2 contract: normalized prefix/state + selected
    # candidate DCT key and raw action RMS. No action history enters the head.
    x = np.concatenate([qctx, umeta[picks]], axis=1).astype(np.float32)
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    infos = {(int(d['task_id']), int(d['episode'])): d for d in manifest['demos'] if d['split'] == 'memory'}
    offset = pp['action_offsets']; flat = pp['actions_flat']
    actions_cache = {}
    labels = np.empty(len(qids), np.float32)
    task = np.asarray(z['memory_task'])[qids]; ep = np.asarray(z['memory_episode'])[qids] % 1000
    frame = np.asarray(z['memory_frame'])[qids]
    for i, (t, e, f, u) in enumerate(zip(task, ep, frame, picks, strict=True)):
        key = (int(t), int(e))
        if key not in actions_cache:
            actions_cache[key] = ex._load_cached_episode(SOURCE, *key)[0]
        truth_ep = actions_cache[key]
        raw = flat[offset[u]:offset[u + 1]]
        n = max(1, min(len(raw), 50, len(truth_ep) - int(f)))
        dn = float(np.linalg.norm(raw[:n, 5:8])); tn = float(np.linalg.norm(truth_ep[int(f):int(f)+n, 5:8]))
        labels[i] = np.clip(tn / dn if dn > 1e-12 else 1., .25, 4.)
    # Split by episode within the outer four-fold training set for the same
    # early-stop rule as the established v2 magnitude regressor.
    unique = np.unique(ep * 100 + task)
    perm = np.random.default_rng(SEED + 50 + fold).permutation(unique)
    val_set = set(map(int, perm[:max(1, len(perm)//5)]))
    val = np.asarray([int(e)*100+int(t) in val_set for t, e in zip(task, ep, strict=True)])
    fit_ids = np.flatnonzero(~val); val_ids = np.flatnonzero(val)
    model, mck = _fit(x, labels, fit_ids, val_ids, seed=SEED+100+fold, max_epochs=100, label=f'oof_fold_{fold}')
    payload = {'state': mck['state'], 'feature_mean': mck['feature_mean'], 'feature_std': mck['feature_std'],
               'best_epoch': mck['best_epoch'], 'val_logscale_mse': mck['val_logscale_mse'],
               'input_dim': x.shape[1], 'history_free': True,
               'fit_query_count': len(qids), 'outer_fold': int(fold), 'curve': mck['curve']}
    torch.save(payload, fold_dir / 'magnitude.pt')
    return model, payload, picks


def fit_gripper(rk, z, feat, qids, qctx, fold_dir, fold):
    import torch
    from torch import nn
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
    # This is the Context head contract (prefix+state, five next signs, one
    # switch-within-five logit), independently refit only on the outer 4 folds.
    x = np.asarray(qctx, np.float32)
    mean = x.mean(0); std = np.maximum(x.std(0), 1e-4)
    xn = ((x - mean) / std).astype(np.float32)
    acts = np.asarray(z['memory_target'])[qids]
    state = np.asarray(z['memory_storage_state'])[qids]
    ys = (acts[:, :5, 11] >= 0).astype(np.float32)
    prev = (state[:, 14] < .03)
    sw = np.any((acts[:, :5, 11] >= 0) != prev[:, None], axis=1).astype(np.float32)
    y = np.concatenate([ys, sw[:, None]], axis=1)
    model = nn.Sequential(nn.Linear(x.shape[1], 256), nn.GELU(), nn.Linear(256, 6))
    torch.manual_seed(SEED + 200 + fold)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    xt = torch.as_tensor(xn); yt = torch.as_tensor(y)
    rng = np.random.default_rng(SEED + 300 + fold)
    curve = []
    for epoch in range(1, 41):
        order = rng.permutation(len(qids)); loss_log=[]
        model.train()
        for start in range(0, len(order), 256):
            ids = torch.as_tensor(order[start:start+256], dtype=torch.long)
            logits = model(xt[ids]); loss = nn.functional.binary_cross_entropy_with_logits(logits, yt[ids])
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
            loss_log.append(float(loss.detach()))
        curve.append({'epoch': epoch, 'train_bce': float(np.mean(loss_log))})
    model.eval()
    torch.save({'state': model.state_dict(), 'feature_mean': mean, 'feature_std': std,
                'input_dim': x.shape[1], 'outer_fold': fold, 'training_rows': len(qids), 'curve': curve},
               fold_dir / 'gripper.pt')
    return model, {'feature_mean': mean, 'feature_std': std, 'input_dim': x.shape[1]}


def generate_drafts(rk, zpart, pool, feat, net, rck, mnet, mck, gripnet, gck,
                    qctx, cctx, *, seed, stochastic=False, batch_size=64):
    import torch
    n = len(qctx); result = np.zeros((n, 50, 12), np.float32)
    first = np.full(n, -1, np.int32); first_len = np.zeros(n, np.int32)
    rng = np.random.default_rng(seed)
    off = pool['action_offsets']; flat = pool['actions_flat']; candidates = np.asarray(pool['global_episode'])
    kemb = get_scoring_tensors(net, cctx)
    meta = unit_meta(pool)
    qhist = make_history(rk, zpart)
    hist0 = np.asarray(qhist['hist'], np.float32); masks0 = np.asarray(qhist['mask'], bool)
    mags = np.asarray(mck['feature_mean'], np.float32); magsd = np.asarray(mck['feature_std'], np.float32)
    for base in range(0, n, batch_size):
        ids = np.arange(base, min(base + batch_size, n)); h = hist0[ids].copy(); mask = masks0[ids].copy()
        cursor = np.zeros(len(ids), np.int32)
        # Context grip head is evaluated once per query and used only on the
        # first selected unit. Following units keep their stored gripper data.
        gx = (qctx[ids] - np.asarray(gck['feature_mean'], np.float32)) / np.asarray(gck['feature_std'], np.float32)
        with torch.no_grad():
            gpred = (gripnet(torch.as_tensor(gx, dtype=torch.float32)).numpy()[:, :5] >= 0).astype(np.float32)*2-1
        while np.any(cursor < 50):
            active = np.flatnonzero(cursor < 50)
            qep = np.asarray(zpart['memory_episode'])[ids[active]]
            picks_greedy, scores = score_rows(net, rck, qctx[ids[active]], h[active], mask[active], qep, kemb, candidates, topk=1)
            if stochastic:
                picks = np.empty(len(active), np.int64)
                for ii, row_scores in enumerate(scores):
                    top = np.argsort(-row_scores, kind='stable')[:5]
                    p = np.exp(row_scores[top] - np.max(row_scores[top])); p /= p.sum()
                    picks[ii] = top[int(rng.choice(len(top), p=p))]
            else:
                picks = picks_greedy.astype(np.int64)
            mx = np.concatenate([qctx[ids[active]], meta[picks]], axis=1)
            mx = torch.as_tensor((mx - mags) / magsd, dtype=torch.float32)
            with torch.no_grad(): scales = torch.exp(mnet(mx)).clamp(.25, 4.).numpy()
            if stochastic: scales = np.clip(scales * rng.uniform(.8, 1.2, len(scales)), .25, 4.)
            for j, (local, u) in enumerate(zip(active, picks, strict=True)):
                row = int(ids[local]); at = int(cursor[local]); u = int(u)
                raw = np.asarray(flat[int(off[u]):int(off[u+1])], np.float32).copy()
                raw[:, 5:8] *= float(scales[j])  # FIXGR: magnitude is position-only.
                if at == 0:
                    first[row] = u; first_len[row] = len(raw)
                    raw[:min(5, len(raw)), 11] = gpred[local, :min(5, len(raw))]
                emit = raw[:min(len(raw), 50-at)]
                if not len(emit): raise RuntimeError('empty unit')
                result[row, at:at+len(emit)] = emit
                h[local] = np.concatenate([h[local], ex.action_storage_to_policy(emit)], axis=0)[-10:]
                mask[local] = True
                cursor[local] += len(emit)
        if base % (batch_size*10) == 0:
            print(json.dumps({'stage': 'draft_batch', 'stochastic': stochastic, 'done': int(min(base+batch_size,n)), 'rows': n}), flush=True)
    return {'draft': result, 'first_unit': first, 'first_unit_length': first_len}


def run_crossfit():
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
    rk, z, pool, teacher, feat, manifest = load_inputs()
    fold_of, fold_eps = make_folds(manifest)
    qfold = np.asarray([fold_of[int(e)] for e in z['memory_episode']], np.int8)
    cfold = np.asarray([fold_of[int(e)] for e in pool['global_episode']], np.int8)
    saved = []
    for fi in range(5):
        fold_dir = OUT / f'fold-{fi}'
        oof_ids = np.flatnonzero(qfold == fi)
        train_ids = np.flatnonzero(qfold != fi)
        cand_ids = np.flatnonzero(cfold != fi)
        assert len(fold_eps[fi]) == 96 and len(oof_ids) > 0 and len(train_ids) > 0
        assert not np.isin(z['memory_episode'][oof_ids], pool['global_episode'][cand_ids]).any()
        fold_dir.mkdir(parents=True, exist_ok=True)
        progress('fold_start', fold=fi, held_episodes=len(fold_eps[fi]), train_query_rows=len(train_ids), candidate_units=len(cand_ids))
        rkz, rpool, net, rck = fit_ranker(rk,z,pool,teacher,feat,train_ids,cand_ids,fold_dir,fi)
        qctx,cctx = rank_context(rk,rkz,feat,rck)
        mnet,mck,train_pick = fit_magnitude(rk,z,pool,feat,train_ids,cand_ids,net,rck,qctx,cctx,fold_dir,fi)
        gnet,gck = fit_gripper(rk,z,feat,train_ids,qctx,fold_dir,fi)
        zheld = subset_z(z,oof_ids,cand_ids)
        qctx_h,cctx_h = rank_context(rk,zheld,feat,rck)
        primary = generate_drafts(rk,zheld,rpool,feat,net,rck,mnet,mck,gnet,gck,qctx_h,cctx_h,seed=SEED+1000+fi)
        stochastic = generate_drafts(rk,zheld,rpool,feat,net,rck,mnet,mck,gnet,gck,qctx_h,cctx_h,seed=SEED+2000+fi,stochastic=True)
        dest = fold_dir / 'drafts.npz'
        if dest.exists(): raise FileExistsError(dest)
        with dest.open('wb') as f:
            np.savez_compressed(f, query_row_id=oof_ids, task=z['memory_task'][oof_ids],
                episode=z['memory_episode'][oof_ids], frame=z['memory_frame'][oof_ids],
                draft=primary['draft'], stochastic_draft=stochastic['draft'],
                first_unit=primary['first_unit'], first_unit_length=primary['first_unit_length'],
                stochastic_first_unit=stochastic['first_unit'], stochastic_first_unit_length=stochastic['first_unit_length'])
        summary={'fold':fi,'held_episodes':fold_eps[fi],'held_rows':len(oof_ids),'fit_rows':len(train_ids),
                 'candidate_units':len(cand_ids),'ranker_best_epoch':rck['best_epoch'],'ranker_dev_type_error':rck['dev_top1_type_error'],
                 'magnitude_best_epoch':mck['best_epoch'],'magnitude_val_logscale_mse':mck['val_logscale_mse'],
                 'gripper_epochs':40,'draft_file':str(dest)}
        write_json(fold_dir/'summary.json',summary); saved.append(summary)
        progress('fold_complete', **summary)
        del net,mnet,gnet,rpool,rkz,zheld,primary,stochastic
        import gc;gc.collect()
    return saved


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['crossfit'],default='crossfit')
    p.add_argument('--out',type=Path,default=OUT)
    a=p.parse_args()
    if a.out != OUT: OUT=a.out
    OUT.mkdir(parents=True,exist_ok=True)
    run_crossfit()
