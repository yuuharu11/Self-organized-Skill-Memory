"""XF1 CPU-only episode-crossfit retrieval confidence diagnostic.

Each fold removes its episodes from both the saved-coefficient memory pool and
KC1 query fitting. The locked Dev-v2 and unseen-20 Hz rows are scored by the
existing KC1 production checkpoint. Reliability features are computed without
target coefficients or action labels; labels enter only in the separate score
and calibration helpers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr, rankdata
from sklearn.isotonic import IsotonicRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kc1_run as K
import kq2cl_common as C

STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', Path(__file__).resolve().parents[5] / '.skillnet-state')).expanduser()
OUT = STATE_DIR / 'outputs/memory-pilot/unseen-libero/xf1-01'
SEED = 4101
QUERY_SEED = 3  # seed of the KC1 checkpoint selected for production
FOLDS = 5
EPOCHS = 16
TAU = .03
BATCH = 256
QUERY_NAME = 'i'
BEST_KC1 = K.OUT / 'offline_corrected.json'


def log(event: str, **fields):
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'event': event, **fields}
    with (OUT / 'progress.log').open('a') as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=float) + '\n')
    print(json.dumps(rec, ensure_ascii=False, default=float), flush=True)


def assign_episode_folds(task: np.ndarray, episode: np.ndarray, *, seed: int = SEED,
                         n_folds: int = FOLDS) -> dict[tuple[int, int], int]:
    """Deterministic, task-stratified balanced episode split."""
    t, e = np.asarray(task, np.int64), np.asarray(episode, np.int64)
    pairs = sorted(set(zip(t.tolist(), e.tolist())))
    result = {}
    for task_id in sorted(set(t.tolist())):
        eps = np.asarray(sorted(int(ep) for tt, ep in pairs if tt == task_id), np.int64)
        rng = np.random.default_rng(seed + 1009 * int(task_id))
        shuffled = eps[rng.permutation(len(eps))]
        for j, ep in enumerate(shuffled):
            result[(int(task_id), int(ep))] = int(j % n_folds)
    return result


def row_fold_ids(task: np.ndarray, episode: np.ndarray, assignments: dict) -> np.ndarray:
    out = np.empty(len(task), np.int8)
    for i, pair in enumerate(zip(np.asarray(task).tolist(), np.asarray(episode).tolist())):
        if (int(pair[0]), int(pair[1])) not in assignments:
            raise KeyError(f'episode absent from fixed split: {pair}')
        out[i] = assignments[(int(pair[0]), int(pair[1]))]
    return out


def fold_partition(query_task, query_episode, memory_task, memory_episode, assignment, fold):
    """Return optimizer rows, heldout rows, and allowed bank IDs; assert episode isolation."""
    qf = row_fold_ids(query_task, query_episode, assignment)
    mf = row_fold_ids(memory_task, memory_episode, assignment)
    train_rows = np.flatnonzero(qf != fold)
    eval_rows = np.flatnonzero(qf == fold)
    memory_ids = np.flatnonzero(mf != fold)
    held = set(zip(np.asarray(query_task)[eval_rows].tolist(), np.asarray(query_episode)[eval_rows].tolist()))
    trained = set(zip(np.asarray(query_task)[train_rows].tolist(), np.asarray(query_episode)[train_rows].tolist()))
    bank_pairs = set(zip(np.asarray(memory_task)[memory_ids].tolist(), np.asarray(memory_episode)[memory_ids].tolist()))
    if held & trained or held & bank_pairs:
        raise AssertionError(f'fold {fold}: heldout episodes leaked into query fitting or memory')
    return train_rows, eval_rows, memory_ids, qf


def confidence_from_scores(scores: np.ndarray, keys: np.ndarray, actions: np.ndarray,
                           allowed: np.ndarray, *, top_k: int = 50) -> dict[str, np.ndarray]:
    """Build label-free confidence features from scores, saved keys and actions."""
    s = np.asarray(scores, np.float64)
    k = np.asarray(keys, np.float32)
    a = np.asarray(actions, np.float32)
    ok = np.asarray(allowed, bool)
    if s.ndim != 2 or ok.shape != s.shape or len(k) != s.shape[1] or len(a) != s.shape[1]:
        raise ValueError(f'invalid feature arrays: scores={s.shape}, keys={k.shape}, actions={a.shape}, allowed={ok.shape}')
    if np.any(~ok.any(1)):
        raise ValueError('query has no allowed memory candidate')
    masked = np.where(ok, s, -np.inf)
    order = np.argsort(-masked, axis=1, kind='stable')[:, :min(top_k, s.shape[1])]
    top_score = np.take_along_axis(masked, order, axis=1)
    top_key = k[order]
    top_action = a[order, :, :6]

    def softmax(x):
        z = x - np.max(x, axis=1, keepdims=True)
        p = np.exp(z)
        return p / np.maximum(p.sum(1, keepdims=True), 1e-300)

    p50 = softmax(top_score)
    entropy50 = -(p50 * np.log(np.maximum(p50, 1e-300))).sum(1) / np.log(max(p50.shape[1], 2))
    pglobal = softmax(masked)
    pglobal[~ok] = 0.
    pglobal /= np.maximum(pglobal.sum(1, keepdims=True), 1e-300)
    top1 = top_action[:, :1]
    disagreement = np.square(top_action - top1).mean(axis=(2, 3))

    def topk_d(k_count):
        kk = min(k_count, top_score.shape[1])
        pp = softmax(top_score[:, :kk])
        return (pp * disagreement[:, :kk]).sum(1)

    n = len(s)
    gap12 = top_score[:, 0] - top_score[:, min(1, top_score.shape[1] - 1)]
    gap15 = top_score[:, 0] - top_score[:, min(4, top_score.shape[1] - 1)]
    spread = top_key[:, :min(10, top_key.shape[1])].var(axis=1).mean(1)
    return {
        'top_ids': order.astype(np.int32), 'top_scores': top_score.astype(np.float32),
        'D10': topk_d(10).astype(np.float32), 'D5': topk_d(5).astype(np.float32),
        'margin12': gap12.astype(np.float32), 'margin15': gap15.astype(np.float32),
        'entropy50': entropy50.astype(np.float32),
        'top5_mass': np.take_along_axis(pglobal, order[:, :min(5, order.shape[1])], 1).sum(1).astype(np.float32),
        'top5_mass_in50': p50[:, :min(5, p50.shape[1])].sum(1).astype(np.float32),
        'spread10': spread.astype(np.float32),
    }


def score_truth(features: dict, raw_target_coeff: np.ndarray, raw_keys: np.ndarray,
                target_actions: np.ndarray, memory_actions: np.ndarray) -> dict[str, np.ndarray]:
    """Score labels separately from confidence feature generation."""
    ids = np.asarray(features['top_ids'], np.int64)
    top1 = ids[:, 0]
    e_arm = C.arm6(np.asarray(memory_actions)[top1], np.asarray(target_actions))
    target = np.asarray(raw_target_coeff, np.float32).reshape(len(top1), -1)
    keys = np.asarray(raw_keys, np.float32).reshape(len(raw_keys), -1)
    e_coef = np.square(target - keys[top1]).sum(-1)
    return {'e_arm': e_arm.astype(np.float32), 'e_coef': e_coef.astype(np.float32)}


def _score_model(model, gs, sp, keys_t, keys_np, bank, target_coeff, *, batch=128):
    n, m = gs.n, len(keys_np)
    score_all = np.empty((n, m), np.float32)
    model.eval()
    with torch.inference_mode():
        for lo in range(0, n, batch):
            ix = torch.arange(lo, min(n, lo + batch), device=next(model.parameters()).device)
            q = model(gs, ix)
            score_all[lo:lo + len(ix)] = model.score(q, keys_t).float().cpu().numpy()
    allowed = K.candidate_mask(sp['task'], sp['episode'], bank['task'], bank['episode'])
    feat = confidence_from_scores(score_all, keys_np, bank['value_actions10'], allowed)
    truth = score_truth(feat, target_coeff, bank['coefficients'], sp['actions'], bank['value_actions10'])
    return feat | truth


def _load_production_query(kr):
    if not BEST_KC1.exists():
        raise FileNotFoundError(f'KC1 production query results missing: {BEST_KC1}')
    result = json.loads(BEST_KC1.read_text())
    selected = result['selected']
    if selected['query'] != 'i':
        raise AssertionError(f'expected KC1 production (i) query, got {selected["query"]}')
    blob = torch.load(selected['checkpoint'], map_location='cpu', weights_only=False)
    cfg = blob['cfg']
    if not (cfg['standardize'] and abs(float(cfg['tau']) - TAU) < 1e-8 and cfg['query'] == 'i'):
        raise AssertionError(f'KC1 selected checkpoint settings differ from expected production setup: {cfg}')
    model = K._load_model(kr, 'i', torch.device('cpu'), int(cfg['seed']))
    model.load_state_dict(blob['query']); model.eval()
    return model, cfg, result


def _fold_train(kr, ctx, targets, bank, assignment, fold, *, model_dir: Path):
    """Fit the fixed KC1 i/std/tau=.03 model on the other four episode folds."""
    from kq2_run import QueryNet
    sp = ctx.splits['eh']
    train_rows, eval_rows, memory_ids, folds = fold_partition(
        sp['task'], sp['episode'], bank['task'], bank['episode'], assignment, fold)
    train_pairs = set(zip(sp['task'][train_rows].tolist(), sp['episode'][train_rows].tolist()))
    eval_pairs = set(zip(sp['task'][eval_rows].tolist(), sp['episode'][eval_rows].tolist()))
    mem_pairs = set(zip(bank['task'][memory_ids].tolist(), bank['episode'][memory_ids].tolist()))
    if train_pairs & eval_pairs or mem_pairs & eval_pairs:
        raise AssertionError(f'fold {fold}: episode leakage in query train or memory')
    if len(train_rows) == 0 or len(eval_rows) == 0 or len(memory_ids) == 0:
        raise AssertionError(f'fold {fold}: empty train/eval/memory partition')

    dest = model_dir / f'fold{fold}_epoch16.pt'
    meta_path = model_dir / f'fold{fold}.json'
    raw_coeff = np.asarray(bank['coefficients'][memory_ids], np.float32)
    keys_np, mean, std = K.coefficient_keys(raw_coeff, standardize=True)
    mem_task = np.asarray(bank['task'][memory_ids], np.int64)
    mem_ep = np.asarray(bank['episode'][memory_ids], np.int64)
    train_target_raw = np.asarray(targets['eh'][0][train_rows], np.float32)
    train_target = (train_target_raw - mean) / std
    nearest = K.nearest_allowed_coefficients(train_target, keys_np, sp['task'][train_rows],
                                               sp['episode'][train_rows], mem_task, mem_ep)
    gs = K._as_gpusplit(kr, sp, torch.device('cpu'), False)
    torch.manual_seed(QUERY_SEED)
    model = QueryNet('gru', 1, 0, False, history='w10', dct=True).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    start_epoch = 0
    existing = sorted(int(p.stem.split('_epoch')[-1]) for p in model_dir.glob(f'fold{fold}_epoch*.pt'))
    if existing:
        last = existing[-1]
        ckpt = torch.load(model_dir / f'fold{fold}_epoch{last}.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model']); optimizer.load_state_dict(ckpt['optimizer']); start_epoch = last
        if ckpt.get('fold') != fold or not np.array_equal(ckpt.get('memory_ids'), memory_ids):
            raise AssertionError('resume checkpoint fold/memory ids mismatch')
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
    log('fold_train_start', fold=fold, train_rows=len(train_rows), eval_rows=len(eval_rows),
        memory_units=len(memory_ids), resume_epoch=start_epoch, heldout_episodes=len(eval_pairs))
    for epoch in range(start_epoch + 1, EPOCHS + 1):
        model.train()
        rng = np.random.default_rng(QUERY_SEED + fold * 100_003 + epoch)
        permutation = rng.permutation(train_rows)
        losses = []
        for s in range(0, len(permutation), BATCH):
            bix = permutation[s:s + BATCH]
            qids = torch.as_tensor(bix, dtype=torch.long)
            q = model(gs, qids)
            local_train = np.searchsorted(train_rows, bix)
            # train_rows is ascending, preserving a stable row->nearest mapping.
            nearest_local = nearest[local_train]
            row_allowed = K.candidate_mask(sp['task'][bix], sp['episode'][bix], mem_task, mem_ep)
            random_ids = rng.choice(len(memory_ids), size=min(K.RANDOM_CANDIDATES, len(memory_ids)), replace=False)
            candidate = np.unique(np.r_[nearest_local, random_ids]).astype(np.int64)
            allowed = torch.as_tensor(row_allowed[:, candidate], dtype=torch.bool)
            if not bool(allowed.any(-1).all()):
                raise AssertionError('fold candidate sample has no episode-legal memory')
            target_t = torch.as_tensor((targets['eh'][0][bix] - mean) / std, dtype=torch.float32)
            key_t = torch.as_tensor(keys_np, dtype=torch.float32)
            cand_t = torch.as_tensor(candidate, dtype=torch.long)
            dtrue = torch.square(target_t[:, None] - key_t[cand_t][None]).sum(-1)
            logits = model.score(q, key_t[cand_t])
            td = torch.softmax((-dtrue / TAU).masked_fill(~allowed, -torch.inf), -1)
            td = torch.nan_to_num(td, nan=0.)
            lp = torch.log_softmax(logits.masked_fill(~allowed, -torch.inf), -1)
            loss = -(td * torch.nan_to_num(lp, nan=0., neginf=0.)).sum(-1).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            with torch.no_grad(): model.logit_scale.clamp_(max=float(np.log(100.)))
            losses.append(float(loss.detach()))
        checkpoint = model_dir / f'fold{fold}_epoch{epoch}.pt'
        temp = checkpoint.with_suffix('.tmp')
        torch.save({'fold': fold, 'epoch': epoch, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'memory_ids': memory_ids,
                    'key_mean': mean, 'key_std': std, 'query_train_rows': train_rows,
                    'setting': {'query': 'i', 'encoder': 'GRU-only', 'standardize': True,
                                'tau': TAU, 'epochs': EPOCHS, 'seed': QUERY_SEED,
                                'batch': BATCH, 'lr': 1e-4, 'weight_decay': 1e-4}}, temp)
        temp.replace(checkpoint)
        log('fold_train_epoch', fold=fold, epoch=epoch, loss=float(np.mean(losses)), checkpoint=str(checkpoint))
    # Re-load final checkpoint so returned model and search keys come from the saved artifact.
    ckpt = torch.load(dest, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model']); model.eval()
    if not meta_path.exists():
        meta = {'fold': fold, 'train_rows': int(len(train_rows)), 'eval_rows': int(len(eval_rows)),
                'memory_units': int(len(memory_ids)), 'train_episode_count': len(train_pairs),
                'eval_episode_count': len(eval_pairs), 'heldout_episodes_not_in_query_training': True,
                'heldout_episodes_not_in_memory': True, 'selected_epoch': EPOCHS,
                'selected_by': 'fixed KC1 Dev-v2-selected setup; no fold-heldout labels used for checkpoint selection',
                'checkpoint': str(dest)}
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + '\n')
    return model, memory_ids, mean, std, eval_rows, folds


def _episode_key(sp):
    return np.asarray(sp['task'], np.int64) * 10_000_000 + np.asarray(sp['episode'], np.int64)


def _split_summary(features, split_name, fold_ids=None):
    # Correlations are computed separately from feature extraction, and labels are consumed here only.
    names = ['D10', 'D5', 'margin12', 'margin15', 'entropy50', 'top5_mass', 'top5_mass_in50', 'spread10']
    uncertainty_sign = {'D10': 1., 'D5': 1., 'margin12': -1., 'margin15': -1.,
                        'entropy50': 1., 'top5_mass': -1., 'top5_mass_in50': -1., 'spread10': 1.}
    e = np.asarray(features['e_arm'], np.float64)
    high = e >= np.quantile(e, .8)
    out = {'rows': len(e), 'feature_metrics': {}}
    task = np.asarray(features['task']) if 'task' in features else np.zeros(len(e), np.int64)
    for name in names:
        x = np.asarray(features[name], np.float64)
        rho = float(spearmanr(x, e).statistic)
        if not np.isfinite(rho): rho = 0.
        ux = x * uncertainty_sign[name]
        urho = float(spearmanr(ux, e).statistic)
        if not np.isfinite(urho): urho = 0.
        out['feature_metrics'][name] = {'spearman_raw_feature_vs_e_arm': rho,
            'uncertainty_orientation': uncertainty_sign[name], 'spearman_uncertainty_vs_e_arm': urho,
            'auroc_high_e_arm_q80': _auc(ux, high)}
    if fold_ids is not None:
        out['by_fold'] = {}
        for f in sorted(set(np.asarray(fold_ids).tolist())):
            ix = np.asarray(fold_ids) == f
            out['by_fold'][str(int(f))] = {'rows': int(ix.sum()), 'metrics': _split_summary({k: np.asarray(v)[ix] for k,v in features.items()
                if isinstance(v, np.ndarray) and len(v) == len(ix)}, f'fold{f}')['feature_metrics']}
    tasks = {}
    for t in sorted(set(task.tolist())):
        ix = task == t
        if ix.sum() < 10: continue
        tasks[str(int(t))] = {}
        for name in names:
            r = spearmanr(np.asarray(features[name])[ix], e[ix]).statistic
            tasks[str(int(t))][name] = float(r) if np.isfinite(r) else 0.
    out['task_count_with_ge10_rows'] = len(tasks)
    out['task_mean_spearman_uncertainty_vs_e_arm'] = {
        n: float(np.mean([v[n] * uncertainty_sign[n] for v in tasks.values()])) if tasks else None for n in names}
    out['task_metrics_raw_spearman'] = tasks
    return out


def _auc(score, label):
    x=np.asarray(score,np.float64).reshape(-1); y=np.asarray(label,bool).reshape(-1)
    if y.all() or (~y).all(): return None
    ranks=rankdata(x, method='average'); n1=y.sum(); n0=len(y)-n1
    return float((ranks[y].sum()-n1*(n1+1)/2)/(n1*n0))


def _quantiles(x):
    a=np.asarray(x,np.float64)
    if not len(a): return {}
    return {'n':int(len(a)),'mean':float(a.mean()),'p05':float(np.quantile(a,.05)),
            'p25':float(np.quantile(a,.25)),'p50':float(np.quantile(a,.5)),
            'p75':float(np.quantile(a,.75)),'p95':float(np.quantile(a,.95))}


def _expert_style_reference(target_actions, query_task, bank, rng_seed):
    """Recreate KQ2-CL Expert's rank-law draft sampler for comparison only."""
    import kq2cl_expert as KE
    law = np.load(C.OUT / 'expert' / 'law.npz')
    spec = json.loads((C.OUT / 'expert' / 'law.json').read_text())
    n = len(target_actions)
    eligible = query_task[:, None] != np.asarray(bank['task'])[None]
    n_el = eligible.sum(1)
    rng = np.random.default_rng(rng_seed)
    ranks, _, _ = KE.sample_ranks(rng, n, [law['mem'], law['dev']], spec['weights'], n_el,
                                  near_fraction=spec['near_fraction'], uniform_fraction=spec['uniform_fraction'])
    unit_actions = np.asarray(bank['value_actions10'], np.float32)[..., :6]
    flat = unit_actions.reshape(len(unit_actions), -1).astype(np.float32)
    q = np.asarray(target_actions, np.float32)[..., :6].reshape(n, -1).astype(np.float32)
    ids = np.empty(n, np.int32)
    for i in range(n):
        d = np.square(flat - q[i]).mean(-1)
        d[~eligible[i]] = np.inf
        rank = min(max(int(ranks[i]), 1), int(n_el[i]))
        ids[i] = np.argpartition(d, rank - 1)[rank - 1]
    err = C.arm6(np.asarray(bank['value_actions10'])[ids], target_actions)
    return {'unit_id': ids, 'rank': ranks.astype(np.int32), 'e_arm': err.astype(np.float32),
            'sampling_rule': 'kq2cl_expert.sample_ranks: 30% measured iii eh oofT / 70% iii Dev-v2 rank pool, 15% top-50 truncated geometric, 5% uniform; ranks mapped to own-task-excluded arm6 action-MSE order'}


def _one_fold(fold, kr, ctx, targets, bank, assignment, model_dir):
    result_path = OUT / f'rows_fold{fold}.npz'
    if result_path.exists():
        with np.load(result_path, allow_pickle=False) as z: return {k:z[k] for k in z.files}
    model, memory_ids, mean, std, eval_rows, folds = _fold_train(kr, ctx, targets, bank, assignment, fold,
                                                                  model_dir=model_dir)
    sp = ctx.splits['eh']; gs = K._as_gpusplit(kr, sp, torch.device('cpu'), False)
    key_raw = np.asarray(bank['coefficients'][memory_ids], np.float32)
    keys_np, _, _ = K.coefficient_keys(key_raw, standardize=True, mean=mean, std=std)
    keys_t = torch.as_tensor(keys_np, dtype=torch.float32)
    selected_sp = {k: (np.asarray(v)[eval_rows] if isinstance(v, np.ndarray) and len(v)==len(sp['key']) else v)
                   for k,v in sp.items()}
    eval_gs = gs
    # The GPUSplit owns the full split; score only heldout row ids, with the fold's memory key subset.
    scores=[]; model.eval()
    with torch.inference_mode():
        for lo in range(0, len(eval_rows), 128):
            rows=eval_rows[lo:lo+128]; ix=torch.as_tensor(rows,dtype=torch.long)
            q=model(eval_gs,ix)
            scores.append(model.score(q,keys_t).cpu().numpy())
    score=np.concatenate(scores)
    allowed=K.candidate_mask(sp['task'][eval_rows],sp['episode'][eval_rows],bank['task'][memory_ids],bank['episode'][memory_ids])
    conf=confidence_from_scores(score,keys_np,bank['value_actions10'][memory_ids],allowed)
    truth=score_truth(conf,targets['eh'][0][eval_rows],bank['coefficients'][memory_ids],
                      sp['actions'][eval_rows],bank['value_actions10'][memory_ids])
    # Full production KC1 query on these same rows, with its full saved-coefficient bank.
    prod, cfg, _ = _load_production_query(kr)
    full_raw=np.asarray(bank['coefficients'],np.float32)
    full_keys, full_mean, full_std=K.coefficient_keys(full_raw,standardize=cfg['standardize'],
                                                       mean=cfg.get('mean'),std=cfg.get('std'))
    prod_score=[]
    with torch.inference_mode():
        for lo in range(0,len(eval_rows),128):
            rows=eval_rows[lo:lo+128]; q=prod(gs,torch.as_tensor(rows,dtype=torch.long))
            prod_score.append(prod.score(q,torch.as_tensor(full_keys,dtype=torch.float32)).cpu().numpy())
    pscore=np.concatenate(prod_score)
    pallow=K.candidate_mask(sp['task'][eval_rows],sp['episode'][eval_rows],bank['task'],bank['episode'])
    pconf=confidence_from_scores(pscore,full_keys,bank['value_actions10'],pallow)
    ptruth=score_truth(pconf,targets['eh'][0][eval_rows],bank['coefficients'],sp['actions'][eval_rows],bank['value_actions10'])
    expert=_expert_style_reference(sp['actions'][eval_rows],np.asarray(sp['task'][eval_rows]),bank,SEED+40000+fold)
    arrays={'query_row_id':eval_rows.astype(np.int32),'task':np.asarray(sp['task'][eval_rows]),
            'episode':np.asarray(sp['episode'][eval_rows]),'frame':np.asarray(sp['key'][eval_rows])[:,2],
            'fold':np.full(len(eval_rows),fold,np.int8),
            **{f'cv_{k}':v for k,v in conf.items()},**{f'cv_{k}':v for k,v in truth.items()},
            **{f'kc1full_{k}':v for k,v in pconf.items()},**{f'kc1full_{k}':v for k,v in ptruth.items()},
            'cv_memory_unit_ids':memory_ids[np.asarray(conf['top_ids'],np.int64)],
            'cv_draft_top1_arm6':np.asarray(bank['value_actions10'])[memory_ids[np.asarray(conf['top_ids'])[:,0]],:,:6],
            'kc1full_memory_unit_ids':np.asarray(pconf['top_ids'],np.int32),
            'kc1full_draft_top1_arm6':np.asarray(bank['value_actions10'])[np.asarray(pconf['top_ids'])[:,0],:,:6],
            'expert_style_unit_id':expert['unit_id'],'expert_style_rank':expert['rank'],'expert_style_e_arm':expert['e_arm']}
    np.savez_compressed(result_path,**arrays)
    log('fold_rows_complete',fold=fold,rows=len(eval_rows),cv_e_arm=float(truth['e_arm'].mean()),
        kc1full_e_arm=float(ptruth['e_arm'].mean()),expert_style_e_arm=float(expert['e_arm'].mean()))
    return arrays


def _eval_locked(split, kr, ctx, targets, bank, prod, cfg):
    dest=OUT/f'rows_{split}.npz'
    if dest.exists():
        with np.load(dest,allow_pickle=False) as z:return {k:z[k] for k in z.files}
    sp=ctx.splits[split]; gs=K._as_gpusplit(kr,sp,torch.device('cpu'),False)
    raw=np.asarray(bank['coefficients'],np.float32)
    keys_np,_,_=K.coefficient_keys(raw,standardize=cfg['standardize'],mean=cfg.get('mean'),std=cfg.get('std'))
    keys=torch.as_tensor(keys_np,dtype=torch.float32)
    out=[]; prod.eval()
    with torch.inference_mode():
        for lo in range(0,gs.n,128):
            ix=torch.arange(lo,min(lo+128,gs.n));q=prod(gs,ix)
            out.append(prod.score(q,keys).cpu().numpy())
    scores=np.concatenate(out)
    allowed=K.candidate_mask(sp['task'],sp['episode'],bank['task'],bank['episode'])
    conf=confidence_from_scores(scores,keys_np,bank['value_actions10'],allowed)
    truth=score_truth(conf,targets[split][0],bank['coefficients'],sp['actions'],bank['value_actions10'])
    arr={'task':np.asarray(sp['task']),'episode':np.asarray(sp['episode']),'frame':np.asarray(sp['key'])[:,2],
         **conf,**truth,'memory_unit_ids':np.asarray(conf['top_ids'],np.int32),
         'draft_top1_arm6':np.asarray(bank['value_actions10'])[np.asarray(conf['top_ids'])[:,0],:,:6]}
    np.savez_compressed(dest,**arr)
    log('locked_split_scored',split=split,rows=gs.n,e_arm=float(truth['e_arm'].mean()))
    return arr


def _candidate_stats(arrays, name_prefix):
    fvals={}
    for f in range(FOLDS):
        sub=arrays['fold']==f
        if sub.any():fvals[str(f)]=_split_summary({k:v[sub] for k,v in arrays.items() if isinstance(v,np.ndarray) and len(v)==len(sub)},f'fold{f}')
    return _split_summary(arrays,name_prefix,arrays['fold'] if 'fold' in arrays else None) | {'fold_summaries':fvals}


def _calibration(best_name, cv, dev):
    signs={'D10':1.,'D5':1.,'margin12':-1.,'margin15':-1.,'entropy50':1.,'top5_mass':-1.,'top5_mass_in50':-1.,'spread10':1.}
    x=np.asarray(cv[best_name],np.float64)*signs[best_name]; y=np.asarray(cv['e_arm'],np.float64)
    d=np.asarray(dev[best_name],np.float64)*signs[best_name]; dy=np.asarray(dev['e_arm'],np.float64)
    cv_fold=np.asarray(cv['fold'],np.int64); oof=np.empty(len(y),np.float64)
    for f in range(FOLDS):
        tr=cv_fold!=f;te=~tr
        ir=IsotonicRegression(out_of_bounds='clip').fit(x[tr],y[tr])
        oof[te]=ir.predict(x[te])
    model=IsotonicRegression(out_of_bounds='clip').fit(x,y)
    pred=model.predict(d)
    order=np.argsort(pred,kind='stable');bins=np.array_split(order,10)
    deciles=[{'n':int(len(ix)),'mean_e_arm':float(dy[ix].mean()),'mean_ehat':float(pred[ix].mean()),
              'absolute_gap':float(abs(dy[ix].mean()-pred[ix].mean()))} for ix in bins if len(ix)]
    groups=np.array_split(np.argsort(pred,kind='stable'),5)
    quintiles=[{'n':int(len(ix)),'mean_ehat':float(pred[ix].mean()),'mean_e_arm':float(dy[ix].mean())} for ix in groups if len(ix)]
    return {'candidate':best_name,'uncertainty_orientation':signs[best_name],
            'crossfit_cv_ehat_mae':float(np.mean(np.abs(oof-y))),
            'dev_isotonic_ehat_mae':float(np.mean(np.abs(pred-dy))),
            'dev_decile_calibration':deciles,'dev_ehat_quintiles':quintiles}


def run() -> dict:
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','8')))
    OUT.mkdir(parents=True,exist_ok=True)
    for p in (OUT/'manifest.json',OUT/'report.md',OUT/'DONE'):
        if p.exists() and p.name in ('report.md','DONE'):
            raise FileExistsError(f'refusing to overwrite completed XF1 artifact: {p}')
    kr,ctx,targets=K._prepare_context()
    bank=K.memory_bank()
    assignment=assign_episode_folds(ctx.splits['eh']['task'],ctx.splits['eh']['episode'])
    fold_sizes={str(f):{} for f in range(FOLDS)}
    for f in range(FOLDS):
        for t in sorted(set(ctx.splits['eh']['task'].tolist())):
            fold_sizes[str(f)][str(int(t))]=sum(ff==f for (tt,_),ff in assignment.items() if tt==int(t))
    if not (K.OUT/'models').is_dir(): raise FileNotFoundError('KC1 production query checkpoints missing')
    model_dir=OUT/'models';model_dir.mkdir(exist_ok=True)
    write_manifest={'status':'running','split_seed':SEED,'query_seed':QUERY_SEED,'folds':FOLDS,'episode_assignments':{f'{t}:{e}':f for (t,e),f in assignment.items()},
        'fold_episode_counts_by_task':fold_sizes,'kc1_setup':{'query':'i','encoder':'GRU only','standardize':True,'tau':TAU,'epochs':EPOCHS,'seed':QUERY_SEED},
        'memory':'KC1 saved coefficients restricted to other four folds; standardization fitted only on allowed fold memory',
        'query_training':'EH rows from other four folds only; fold-heldout episodes excluded from optimizer and memory',
        'evaluation_rows':'EH rows assigned to each heldout fold; locked Dev-v2 and unseen20 with KC1 full production query',
        'gpu':'not used; CPU only'}
    manifest_path=OUT/'manifest.json'
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        if old.get('split_seed') != SEED or old.get('query_seed') != QUERY_SEED or old.get('folds') != FOLDS:
            raise AssertionError('existing XF1 manifest conflicts with this run; refusing to reuse artifacts')
    manifest_path.write_text(json.dumps(write_manifest,indent=2,ensure_ascii=False)+'\n')
    current_model,cfg,kc1result=_load_production_query(kr)
    arrays=[]
    for f in range(FOLDS):
        arrays.append(_one_fold(f,kr,ctx,targets,bank,assignment,model_dir))
    cv={}
    keys=arrays[0].keys()
    for k in keys:
        if isinstance(arrays[0][k],np.ndarray): cv[k]=np.concatenate([a[k] for a in arrays],axis=0)
    cv['e_arm']=cv['cv_e_arm']; cv['e_coef']=cv['cv_e_coef']
    dev=_eval_locked('dev2',kr,ctx,targets,bank,current_model,cfg)
    unseen=_eval_locked('unseen20',kr,ctx,targets,bank,current_model,cfg)
    feature_names=['D10','D5','margin12','margin15','entropy50','top5_mass','top5_mass_in50','spread10']
    candidate_summary={}
    for prefix,name in [('cv_','crossfit'),('kc1full_','kc1_production_query')]:
        arr={k[len(prefix):]:v for k,v in cv.items() if k.startswith(prefix)}
        arr['task']=cv['task']; arr['episode']=cv['episode']; arr['fold']=cv['fold']
        candidate_summary[name]=_candidate_stats(arr,name)
    dev_summary=_split_summary(dev,'dev2')
    unseen_summary=_split_summary(unseen,'unseen20')
    best=max(feature_names,key=lambda n:candidate_summary['crossfit']['feature_metrics'][n]['spearman_uncertainty_vs_e_arm'])
    calibration_cv={k[3:]:v for k,v in cv.items() if k.startswith('cv_') and k[3:] in feature_names+['e_arm']}
    calibration_cv['fold']=cv['fold']
    calibration=_calibration(best,calibration_cv,dev)
    expert_err=np.asarray(cv['expert_style_e_arm'],np.float64)
    write_manifest['status']='complete'
    write_manifest['completed_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    result={'status':'complete','split_protocol':write_manifest,'kc1_production_query':{
        'checkpoint':cfg.get('checkpoint',kc1result['selected'].get('checkpoint')),'selected_name':kc1result['selected']['name'],
        'settings':{'query':cfg['query'],'standardize':cfg['standardize'],'tau':cfg['tau'],'selected_seed':cfg['seed']}},
        'row_counts':{'crossfit':int(len(cv['e_arm'])),'dev2':int(len(dev['e_arm'])),'unseen20':int(len(unseen['e_arm']))},
        'confidence_candidates':feature_names,'confidence_orientation':'positive score means uncertainty/higher predicted error; margins and top5 mass are sign-reversed for this interpretation',
        'best_candidate_selected_without_unseen20':best,'selection_rule':'maximum crossfit pooled Spearman of oriented uncertainty against e_arm; Dev-v2 is reported as transfer check, unseen20 is report-only',
        'crossfit_confidence':candidate_summary['crossfit'],'kc1_full_query_on_same_crossfit_rows':candidate_summary['kc1_production_query'],
        'dev2_confidence':dev_summary,'unseen20_confidence':unseen_summary,'isotonic':calibration,
        'draft_error_distributions':{'fold_search_crossfit':_quantiles(cv['e_arm']),
            'kc1_full_production_query_on_same_fold_rows':_quantiles(cv['kc1full_e_arm']),
            'kq2cl_expert_training_sampler_recreated_on_same_rows':_quantiles(expert_err),
            'dev2_kc1_full_query':_quantiles(dev['e_arm']),'unseen20_kc1_full_query':_quantiles(unseen['e_arm'])},
        'e_coef_distributions':{'fold_search':_quantiles(cv['e_coef']),'kc1_full':_quantiles(cv['kc1full_e_coef']),
                                'dev2':_quantiles(dev['e_coef']),'unseen20':_quantiles(unseen['e_coef'])},
        'expert_style_reference_rule':_expert_style_reference.__doc__}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,default=float)+'\n')
    with np.load(OUT/'rows_fold0.npz') as _z: pass
    np.savez_compressed(OUT/'all_rows.npz',**cv)
    status='進める: 信頼度で t0 を変える' if (candidate_summary['crossfit']['feature_metrics'][best]['spearman_uncertainty_vs_e_arm']>=.3 and
        dev_summary['feature_metrics'][best]['spearman_uncertainty_vs_e_arm']>=.2) else '推奨: t0 固定で頑健化'
    report=_render_report(result,status)
    report_path=OUT/'report.md'
    if report_path.exists(): raise FileExistsError(f'refusing to overwrite {report_path}')
    report_path.write_text(report)
    (OUT/'DONE').write_text('XF1 CPU-only diagnostic complete\n')
    log('xf1_complete',best_candidate=best,decision=status,crossfit_spearman=candidate_summary['crossfit']['feature_metrics'][best]['spearman_uncertainty_vs_e_arm'],
        dev_spearman=dev_summary['feature_metrics'][best]['spearman_uncertainty_vs_e_arm'],unseen_spearman=unseen_summary['feature_metrics'][best]['spearman_uncertainty_vs_e_arm'])
    return result


def _render_report(res, decision):
    rows=['# XF1：検索信頼度と下書き誤差','',f'**判定: {decision}**','',
      '交差分割はタスク内 episode ごとに固定 seed で5分割し、fold のメモリ単位とクエリ学習の両方からその fold を除外した。誤差は検索・信頼度特徴には使わず、採点と較正にだけ使った。Dev v2 と unseen20 は KC1 の本番クエリで採点。unseen20 は選択に使っていない。','',
      '## 主表：信頼度候補と実際の誤差の関係','',
      'Spearman は「不確実性向きに符号をそろえた特徴」と e_arm の相関（正なら大きな不確実性ほど誤差が大きい）。AUROC は e_arm 上位20%を高不確実性で識別。','',
      '|候補|交差分割 Spearman|Dev v2 Spearman|未学習20Hz Spearman|交差分割 AUROC|Dev AUROC|未学習 AUROC|fold平均 Spearman|task平均 Spearman|',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    cv=res['crossfit_confidence'];dev=res['dev2_confidence'];un=res['unseen20_confidence']
    for n in res['confidence_candidates']:
        a=cv['feature_metrics'][n];b=dev['feature_metrics'][n];c=un['feature_metrics'][n]
        fm=[v['feature_metrics'][n]['spearman_uncertainty_vs_e_arm'] for v in cv['fold_summaries'].values()]
        tm=cv['task_mean_spearman_uncertainty_vs_e_arm'][n]
        rows.append(f'|{n}|{a["spearman_uncertainty_vs_e_arm"]:.3f}|{b["spearman_uncertainty_vs_e_arm"]:.3f}|{c["spearman_uncertainty_vs_e_arm"]:.3f}|{a["auroc_high_e_arm_q80"]:.3f}|{b["auroc_high_e_arm_q80"]:.3f}|{c["auroc_high_e_arm_q80"]:.3f}|{np.mean(fm):.3f}|{tm:.3f}|')
    d=res['draft_error_distributions']
    rows += ['', '## 下書き誤差の分布', '',
      'MSE_arm6 は6腕軸の10ステップ窓に対する平均二乗誤差。Expert 参照下書きは `kq2cl_expert.sample_ranks` と同じ順位混合・own-task 除外の action-MSE 順で同じ評価行に再生成した分布。','',
      '|生成法・集合|n|平均|p05|p25|中央値|p75|p95|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,item in d.items():
        rows.append(f'|{name}|{item["n"]}|{item["mean"]:.4f}|{item["p05"]:.4f}|{item["p25"]:.4f}|{item["p50"]:.4f}|{item["p75"]:.4f}|{item["p95"]:.4f}|')
    rows += ['', '## 最良候補の較正', '',f'選択候補（交差分割 Spearman 最大、unseen20 は不使用）: **{res["best_candidate_selected_without_unseen20"]}**。',
      f'交差適合 CV ehat MAE={res["isotonic"]["crossfit_cv_ehat_mae"]:.4f}；Dev v2 isotonic ehat MAE={res["isotonic"]["dev_isotonic_ehat_mae"]:.4f}.','',
      '### Dev v2 10分位の平均誤差と予測誤差','', '|bin|n|実測 e_arm|予測 ehat|絶対差|','|---:|---:|---:|---:|---:|']
    for i,b in enumerate(res['isotonic']['dev_decile_calibration'],1):rows.append(f'|{i}|{b["n"]}|{b["mean_e_arm"]:.4f}|{b["mean_ehat"]:.4f}|{b["absolute_gap"]:.4f}|')
    rows += ['', '### Dev v2 ehat 5群', '', '|group|n|mean ehat|mean e_arm|','|---:|---:|---:|---:|']
    for i,b in enumerate(res['isotonic']['dev_ehat_quintiles'],1):rows.append(f'|{i}|{b["n"]}|{b["mean_ehat"]:.4f}|{b["mean_e_arm"]:.4f}|')
    rows += ['', '## 分割と再現性','',
      f'- episode-fold assignment seed: {res["split_protocol"]["split_seed"]}; query seed: {res["split_protocol"]["query_seed"]}。fold ごとの件数と `(task,episode)→fold` は `manifest.json`。',
      f'- 各 fold は固定 KC1 (i, GRU-only, 標準化、τ=.03、16 epochs、query seed={res["split_protocol"]["query_seed"]}) で学習。heldout episode は検索メモリにも query optimizer にも含めていない。',
      '- 保存物: `rows_fold0..4.npz` (top50 unit id/score、下書き、confidence、採点用誤差)、`rows_dev2.npz`、`rows_unseen20.npz`、`all_rows.npz`、`summary.json`。',
      '- 信頼度特徴の生成関数 `confidence_from_scores` は正解行動・target coefficient を引数に取らない。`score_truth` は別関数。',
      f'- 実行時刻: {time.strftime("%Y-%m-%d %H:%M:%S JST", time.gmtime(time.time()+9*3600))}。progress.log は UTC。',
      '- CPU only。GPU は使用していない。','']
    return '\n'.join(rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--resume',action='store_true',default=True)
    args=ap.parse_args()
    run()


if __name__=='__main__': main()
