#!/usr/bin/env python3
"""Probe whether observation differences predict the residual of a P5 draft.

The retrieval model and candidate bank are frozen KEY2-F artifacts. This
module trains only small residual and gripper-transition predictors.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[5]
STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', REPO_ROOT / '.skillnet-state')).expanduser()
sys.path.insert(0, str(HERE))
import key_shared_latent as key2  # noqa: E402

ROOT = (STATE_DIR / 'outputs/memory-pilot/unseen-libero')
KEY2_ROOT = ROOT / 'key-shared-01/attempt-27-production'
RUN_ROOT = (STATE_DIR / 'outputs/memory-pilot/pi05-memory40-20261104/run-01')
MISSING = ROOT / 'key-shared-01/attempt-02/prefix-missing'
SEEN = ROOT / 'seg-context-01/seg5-cache'
UNSEEN = ROOT / 'seg-context-01/unseen-stride5'
VJEPA = ROOT / 'key-shared-01/attempt-27-production/vjepa/libero90_eval'
OUTPUT = ROOT / 'delta-probe-01'
VARIANTS = ('N', 'delta_state', 'delta_vlm', 'delta_history', 'delta_all')


def draft_features(query: dict[str, np.ndarray], bank_input: dict[str, np.ndarray],
                   selected_ids: np.ndarray, candidate_actions: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Build query-only bases and query-minus-selected-memory deltas."""
    ids = np.asarray(selected_ids, np.int64)
    if ids.shape != (len(query['key']),) or np.any((ids < 0) | (ids >= len(candidate_actions))):
        raise ValueError('selected candidate IDs do not align with query rows')
    pooled_q = key2._pool_tokens(query['tokens'], query['groups']).astype(np.float32)
    pooled_m = key2._pool_tokens(bank_input['tokens'][ids], bank_input['groups'][ids]).astype(np.float32)
    hist_q = np.concatenate([query['history'].reshape(len(ids), -1),
                             query['history_mask'].astype(np.float32)], axis=1)
    hist_m = np.concatenate([bank_input['history'][ids].reshape(len(ids), -1),
                             bank_input['history_mask'][ids].astype(np.float32)], axis=1)
    state_q = np.asarray(query['state'], np.float32)
    state_m = np.asarray(bank_input['state'][ids], np.float32)
    draft_actions = np.asarray(candidate_actions[ids, :, :6], np.float32)
    draft = draft_actions.reshape(len(ids), -1)
    base = np.concatenate([state_q, pooled_q, hist_q, draft], axis=1).astype(np.float32)
    deltas = {
        'N': base,
        'delta_state': np.concatenate([base, state_q - state_m], axis=1),
        'delta_vlm': np.concatenate([base, pooled_q - pooled_m], axis=1),
        'delta_history': np.concatenate([base, hist_q - hist_m], axis=1),
        'delta_all': np.concatenate([base, state_q - state_m, pooled_q - pooled_m,
                                     hist_q - hist_m], axis=1),
    }
    target_grip = np.asarray(query['actions'], np.float32)[..., 6]
    if not np.isfinite(base).all() or not np.isfinite(target_grip).all():
        raise ValueError('non-finite probe features')
    return deltas, draft_actions


def action_residual(target: np.ndarray, draft: np.ndarray) -> np.ndarray:
    """Ten-step arm6 residual; gripper is deliberately excluded."""
    y = np.asarray(target, np.float32)[..., :6]
    d = np.asarray(draft, np.float32)
    if y.shape != d.shape or y.ndim != 3 or y.shape[1:] != (10, 6):
        raise ValueError(f'expected matching [N,10,6], got {y.shape}, {d.shape}')
    return y - d


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardized primal ridge with an intercept (dual solve when p>n)."""
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    mean = x.mean(axis=0); scale = x.std(axis=0); scale[scale < 1e-6] = 1.
    z = (x - mean) / scale; ym = y.mean(axis=0); yc = y - ym
    if z.shape[1] > z.shape[0]:
        w = z.T @ np.linalg.solve(z @ z.T + alpha * np.eye(len(z)), yc)
    else:
        w = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ yc)
    return mean.astype(np.float32), scale.astype(np.float32), w.astype(np.float32), ym.astype(np.float32)


def ridge_predict(model: tuple[np.ndarray, ...], x: np.ndarray) -> np.ndarray:
    mean, scale, weight, ymean = model
    return ((np.asarray(x, np.float32) - mean) / scale) @ weight + ymean


class ProbeMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 256), nn.GELU(), nn.Linear(256, 128),
                                 nn.GELU(), nn.Linear(128, output_dim))
        self.grip = nn.Sequential(nn.Linear(input_dim, 128), nn.GELU(), nn.Linear(128, 10))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.net(x), self.grip(x)


def _switch_labels(actions: np.ndarray) -> np.ndarray:
    grip = np.asarray(actions, np.float32)[..., 6]
    sign = np.where(grip >= 0, 1, -1)
    out = np.zeros_like(sign, dtype=np.float32)
    out[:, 1:] = (sign[:, 1:] != sign[:, :-1])
    return out


def train_mlp(x: np.ndarray, residual: np.ndarray, grip_switch: np.ndarray, *, seed: int,
              dev_x: np.ndarray, dev_residual: np.ndarray, epochs: int = 80) -> tuple[ProbeMLP, list[float], int]:
    torch.manual_seed(seed); np.random.seed(seed)
    model = ProbeMLP(x.shape[1], int(np.prod(residual.shape[1:])))
    xm = x.mean(axis=0); xs = x.std(axis=0); xs[xs < 1e-6] = 1.
    xx = torch.from_numpy(((x - xm) / xs).astype(np.float32))
    yy = torch.from_numpy(residual.reshape(len(x), -1).astype(np.float32))
    gg = torch.from_numpy(grip_switch.astype(np.float32))
    pos = float(gg.sum()); neg = float(gg.numel() - pos)
    pos_weight = torch.tensor([min(30., neg / max(pos, 1.))], dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    hist=[]; best=float('inf'); best_state=None; best_epoch=0
    dx=torch.from_numpy(((np.asarray(dev_x,np.float32)-xm)/xs).astype(np.float32))
    dy=np.asarray(dev_residual,np.float32).reshape(len(dev_residual),-1)
    for epoch in range(1,epochs+1):
        model.train(); opt.zero_grad(set_to_none=True)
        pr, pg = model(xx)
        loss = F.mse_loss(pr, yy) + .15 * F.binary_cross_entropy_with_logits(pg, gg, pos_weight=pos_weight)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
        hist.append(float(loss.detach()))
        if epoch in (10,30,80) or epoch==epochs:
            model.eval()
            with torch.no_grad():
                vp,_=model(dx)
                score=float(np.mean((vp.numpy()-dy)**2))
            if score < best:
                best=score;best_epoch=epoch;best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
    if best_state is not None:model.load_state_dict(best_state)
    model.eval()
    model.register_buffer('probe_mean', torch.from_numpy(xm.astype(np.float32)))
    model.register_buffer('probe_scale', torch.from_numpy(xs.astype(np.float32)))
    return model, hist, best_epoch


def mlp_predict(model: ProbeMLP, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        p, g = model((torch.from_numpy(np.asarray(x, np.float32)) - model.probe_mean) / model.probe_scale)
    return p.numpy().reshape(-1, 10, 6), torch.sigmoid(g).numpy()


def _auc_time(prob: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, bool); prob = np.asarray(prob, np.float64)
    flat_y=labels.ravel(); flat_p=prob.ravel()
    if flat_y.any() and (~flat_y).any():
        ranks=rankdata(flat_p,method='average')
        npos=int(flat_y.sum());nneg=len(flat_y)-npos
        auc=float((ranks[flat_y].sum()-npos*(npos+1)/2)/(npos*nneg))
    else:auc=None
    event=labels.any(1); predicted=prob >= .5
    pred_event=predicted.any(1)
    times=[]
    for i in np.flatnonzero(event & pred_event):
        ti=int(np.flatnonzero(labels[i])[0]); pi=int(np.flatnonzero(predicted[i])[0]); times.append(abs(pi-ti))
    return {'switch_auc':auc,'switch_windows':int(event.sum()),'miss_rate':float((event & ~pred_event).sum()/max(event.sum(),1)),
            'event_precision':float((event & pred_event).sum()/max(pred_event.sum(),1)),
            'time_error_steps_mean_when_detected':float(np.mean(times)) if times else None}


def _metrics(pred_resid: np.ndarray, draft: np.ndarray, target: np.ndarray) -> dict[str, float]:
    y=np.asarray(target,np.float32)[...,:6];d=np.asarray(draft,np.float32);p=d+pred_resid
    # Per-axis mean squared error; KEY2 retrieval reports the same numerator / 70.
    return {'residual_mse_per_arm_element':float(np.mean((pred_resid-(y-d))**2)),
            'corrected_arm_mse6_over70':float(np.sum((p-y)**2)/(len(y)*70)),
            'draft_arm_mse6_over70':float(np.sum((d-y)**2)/(len(y)*70)),
            'oracle_arm_mse6_over70':0.0}


def _rank_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if np.std(x)==0 or np.std(y)==0:return None
    v=spearmanr(x,y).statistic
    return float(v) if np.isfinite(v) else None


def _top1_ids(model: torch.nn.Module, bank_input: dict[str,np.ndarray], bank: dict[str,np.ndarray],
              split: dict[str,np.ndarray], device: str, *, mask_self: bool = False) -> np.ndarray:
    """Use frozen KEY2-F P5 scores; self-episode masking is training-only."""
    # KEY2-F P5 encodes memory starts with the *query* tower on both sides
    # (`_encode_candidates` passes key=(mode=='separate'), i.e. False for p5);
    # `VectorRanker.k` is never trained.  Reuse the exact KEY2 encoders so the
    # top-1 draft reproduces KEY2-F (learned .04072 / unseen .10545).
    mode='p5'; model.eval()
    kr,_,_=key2._encode_candidates(model,mode,bank_input,bank,device)
    q,_=key2._query_representations(model,mode,split,device)
    scores=np.einsum('qkd,mkd->qm',q,kr,optimize=True)/q.shape[1]
    scores*=float(key2._logit_scale_parameter(model).detach().exp().clamp(max=key2.LOGIT_SCALE_MAX).cpu())
    if mask_self:
        same=(np.asarray(split['task'])[:,None]==np.asarray(bank['task'])[None,:]) & (np.asarray(split['episode'])[:,None]==np.asarray(bank['episode'])[None,:])
        scores[same]=-np.inf
        if np.any(~np.isfinite(scores).any(1)):raise ValueError('self-episode exclusion removed all candidates')
    return np.argmax(scores,axis=1).astype(np.int64)


def _episode_ci(diff: np.ndarray, split: dict[str,np.ndarray], seed: int) -> dict[str,float]:
    return key2.episode_bootstrap_ci(np.asarray(diff,np.float32),split['task'],split['episode'],reps=5000,seed=seed)


def run(output: Path = OUTPUT, *, smoke: bool = False, device: str = 'cpu') -> dict[str, Any]:
    output=Path(output)
    if output.exists() and any(output.iterdir()): raise FileExistsError(f'refusing to overwrite {output}')
    output.mkdir(parents=True,exist_ok=True)
    data=key2._build_key2_queries(RUN_ROOT,MISSING,SEEN,UNSEEN,VJEPA)
    bank,bi,fit_bank,fit_bi,train,learned,unseen,dev=data
    if smoke:
        train={k:v[:128] if isinstance(v,np.ndarray) and len(v)==len(train['key']) else v for k,v in train.items()}
        learned={k:v[:16] if isinstance(v,np.ndarray) and len(v)==len(learned['key']) else v for k,v in learned.items()}
        unseen={k:v[:16] if isinstance(v,np.ndarray) and len(v)==len(unseen['key']) else v for k,v in unseen.items()}
        dev={k:v[:16] if isinstance(v,np.ndarray) and len(v)==len(dev['key']) else v for k,v in dev.items()}
    ckpt=KEY2_ROOT/'final/p5/p5_seed1.pt'
    if not ckpt.exists(): raise FileNotFoundError(ckpt)
    model=key2._new_model('p5').to(device)
    model.load_state_dict(torch.load(ckpt,map_location=device,weights_only=True))
    # Build training candidates restricted to the KEY2 teacher-fit bank, which
    # excludes the fixed learned-evaluation episodes.
    bmap={tuple(map(int,k)):i for i,k in enumerate(bi['key'])}
    fit_global=np.asarray([bmap[tuple(map(int,k))] for k in fit_bi['key']],np.int64)
    fit_input={k:(v[fit_global] if isinstance(v,np.ndarray) and len(v)==len(bank['length']) else v) for k,v in bi.items()}
    fit_actions=key2._candidate_window_actions(np.arange(len(fit_bank['length'])),fit_bank)
    fit_train_ids=_top1_ids(model,fit_input,fit_bank,train,device,mask_self=True)
    train_parts,train_draft=draft_features(train,fit_input,fit_train_ids,fit_actions)
    train_y=action_residual(train['actions'],train_draft)
    train_switch=_switch_labels(train['actions'])
    # For exact reproduction of P5, evaluate against the complete KEY2-F unit bank.
    split_data={}
    for name,split in [('dev',dev),('learned',learned),('unseen',unseen)]:
        ids=_top1_ids(model,bi,bank,split,device)
        parts,draft=draft_features(split,bi,ids,key2._candidate_window_actions(np.arange(len(bank['length'])),bank))
        split_data[name]={'split':split,'ids':ids,'parts':parts,'draft':draft,'residual':action_residual(split['actions'],draft)}
        allerr=key2._arm6_candidate_mse(split['actions'],key2._candidate_window_actions(np.arange(len(bank['length'])),bank))
        split_data[name]['oracle_top1_arm6_mse']=float(allerr.min(1).mean())
        split_data[name]['draft_row_mse']=allerr[np.arange(len(ids)),ids].astype(np.float64)
    labels=_switch_labels(train['actions'])
    results={'protocol':{'p5_checkpoint':str(ckpt),'p5_seed':1,'bank_units':int(len(bank['length'])),
        'train_queries':int(len(train['key'])),'learned_rows':int(len(learned['key'])),'unseen_rows':int(len(unseen['key'])),
        'dev_rows':int(len(dev['key'])),'self_episode_excluded_training':True,'target_shape':'10x6 arm residual',
        'feature_variants':list(VARIANTS),'smoke':bool(smoke)},'models':{},'splits':{}}
    # Ridge alpha and the MLP checkpoint epoch are selected only on Dev.
    mlps={}; ridge_models={}
    for variant in VARIANTS:
        xtr=train_parts[variant]
        ridge_candidates=[]
        for alpha in (1.,10.,100.):
            rm=ridge_fit(xtr,train_y.reshape(len(train_y),-1),alpha)
            dp=split_data['dev'];pr=ridge_predict(rm,dp['parts'][variant]).reshape(-1,10,6)
            metric=_metrics(pr,dp['draft'],dp['split']['actions'])['corrected_arm_mse6_over70']
            ridge_candidates.append((metric,alpha,rm))
        _,alpha,rm=min(ridge_candidates,key=lambda x:(x[0],x[1]))
        dv=split_data['dev']
        mlp,hist,best_epoch=train_mlp(xtr,train_y,labels,seed=4101,
            dev_x=dv['parts'][variant],dev_residual=dv['residual'],epochs=5 if smoke else 80)
        mlps[variant]=(mlp,None)
        ridge_models[variant]=rm
        devp=split_data['dev'];mr,mg=mlp_predict(mlp,devp['parts'][variant])
        results['models'][variant]={'ridge':{'selected_alpha':alpha,'dev_grid':[{'alpha':float(a),'corrected_arm_mse6_over70':float(v)} for v,a,_ in ridge_candidates]},
            'mlp':{'epochs_trained':len(hist),'selected_epoch':best_epoch,'train_loss_first_last':[hist[0],hist[-1]]}}
        for name,sd in split_data.items():
            split=sd['split'];x=sd['parts'][variant];draft=sd['draft'];res=sd['residual'];truth=split['actions']
            rp=ridge_predict(rm,x).reshape(-1,10,6);mp,gp=mlp_predict(mlp,x)
            held=_switch_labels(truth).astype(bool)
            grip=_auc_time(gp,held)
            # Hold-last is the causally available previous gripper sign: the
            # first future action is predicted to retain the latest history value.
            last=np.zeros(len(truth),np.float32)
            for i in range(len(truth)):
                valid=np.flatnonzero(split['history_mask'][i])
                if len(valid):last[i]=split['history'][i,valid[-1],6]
            hold=np.zeros((len(truth),10),np.float32)
            target_switch=held
            hold_report=_auc_time(np.zeros_like(hold),target_switch)
            results['splits'].setdefault(name,{})[variant]={
                'ridge':_metrics(rp,draft,truth),'mlp':_metrics(mp,draft,truth),
                'ridge_residual_mse_per_arm_element':float(np.mean((rp-res)**2)),
                'mlp_residual_mse_per_arm_element':float(np.mean((mp-res)**2)),
                'grip_switch_mlp':grip,'grip_switch_hold_last':hold_report,
                'draft_residual_norm_quantiles':np.quantile(np.linalg.norm(res.reshape(len(res),-1),axis=1),[0,.25,.5,.75,.9,.99,1]).tolist(),
                'candidate_row_ids':sd['ids'].tolist() if name in ('dev','learned','unseen') else []}
    # Add compact Δ-vs-N paired bootstrap CIs on corrected arm error.
    for name,sd in split_data.items():
        split=sd['split'];base=sd['parts']['N'];draft=sd['draft'];res=sd['residual']
        for method in ('ridge','mlp'):
            metrics=[]
            for variant in VARIANTS:
                if method=='ridge':
                    fit=ridge_models[variant]
                    pred=ridge_predict(fit,sd['parts'][variant]).reshape(-1,10,6)
                else:
                    pred,_=mlp_predict(mlps[variant][0],sd['parts'][variant])
                metrics.append(float(np.sum((draft+pred-np.asarray(split['actions'])[...,:6])**2)/(len(draft)*70)))
            results['splits'][name]['paired_corrected_mse_'+method]={
                'variant_order':list(VARIANTS),'means':metrics,
                'delta_vs_N_episode_bootstrap':{v:_episode_ci(
                    np.sum((draft+((ridge_predict(ridge_models[v],sd['parts'][v]).reshape(-1,10,6)) if method=='ridge' else mlp_predict(mlps[v][0],sd['parts'][v])[0])-np.asarray(split['actions'])[...,:6])**2,axis=(1,2))/70
                    - np.sum((draft+((ridge_predict(ridge_models['N'],sd['parts']['N']).reshape(-1,10,6)) if method=='ridge' else mlp_predict(mlps['N'][0],sd['parts']['N'])[0])-np.asarray(split['actions'])[...,:6])**2,axis=(1,2))/70,
                    split,seed=4101+VARIANTS.index(v)) for v in VARIANTS if v!='N'}}
    # Keep top-1 row IDs and source unit IDs separately for auditability.
    for name,sd in split_data.items():
        e=sd['draft_row_mse'];srt=np.sort(e)[::-1]
        results['splits'][name]['standard_metric']={
            'definition':'top-1 arm6 MSE = sum of squared arm6 errors over 10 steps / 70 (KEY2-F top1_arm6_mse)',
            'draft_top1_arm6_mse':float(e.mean()),'querywise_oracle_top1_arm6_mse':sd['oracle_top1_arm6_mse'],
            'draft_row_mse_quantiles':np.quantile(e,[.25,.5,.75,.9,.99]).tolist(),
            'share_of_error_from_worst_25pct_rows':float(srt[:max(1,len(e)//4)].sum()/max(e.sum(),1e-12))}
    results['splits']['selection_audit']={name:{'candidate_rows':v['ids'].tolist()} for name,v in split_data.items()}
    _write_results(output,results)
    return results


def _write_results(output: Path, results: dict[str, Any]) -> None:
    (output/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    lines=['# DP1：記録時との差分から補正量が読めるか','',
        'P5（KEY2-F）の固定 checkpoint で選んだ単位を下書きにし、LIBERO-40 メモリ行から腕6軸残差とグリッパー切替を予測した。入力に正解行動は含めず、正解は教師・採点だけに使った。',
        '',f"- P5 checkpoint: `{results['protocol']['p5_checkpoint']}`",f"- メモリ単位: {results['protocol']['bank_units']}",
        f"- 評価行: learned={results['protocol']['learned_rows']}, unseen={results['protocol']['unseen_rows']}, Dev={results['protocol']['dev_rows']}",
        '- residual は query horizon 10×腕6 の単位ごとの差として定義。補正・下書き列は腕6軸二乗和 / 70（KEY2-F の top-1 arm6 MSE と同じ標準指標）。残差 MSE 列は60要素平均なので ×6/7 で標準指標に換算できる。', '',
        '## 補正結果', '', '| 分割 | 入力 | ridge α | ridge 残差 MSE (60要素平均) | ridge 補正 (標準) | MLP 残差 MSE (60要素平均) | MLP 補正 (標準) | 下書き (標準) |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for split,rows in results['splits'].items():
        if split=='selection_audit':continue
        if not isinstance(rows,dict):continue
        for v in VARIANTS:
            if v not in rows:continue
            x=rows[v];a=results['models'][v]['ridge']['selected_alpha']
            lines.append(f"| {split} | {v} | {a:g} | {x['ridge_residual_mse_per_arm_element']:.5f} | {x['ridge']['corrected_arm_mse6_over70']:.5f} | {x['mlp_residual_mse_per_arm_element']:.5f} | {x['mlp']['corrected_arm_mse6_over70']:.5f} | {x['ridge']['draft_arm_mse6_over70']:.5f} |")
    lines += ['', '## 標準指標（top-1 arm6 MSE、KEY2-F と同じ）と残差の分布', '',
        '| 分割 | 下書き P5 top-1 | 単位 oracle top-1 | ridge N 補正 | ridge Δall 補正 | MLP N 補正 | MLP Δall 補正 | 行誤差 中央値 | 行誤差 90% | 上位25%行が占める誤差 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for split,rows in results['splits'].items():
        if split=='selection_audit' or 'standard_metric' not in rows:continue
        m=rows['standard_metric'];q=m['draft_row_mse_quantiles']
        lines.append(f"| {split} | {m['draft_top1_arm6_mse']:.5f} | {m['querywise_oracle_top1_arm6_mse']:.5f} | {rows['N']['ridge']['corrected_arm_mse6_over70']:.5f} | {rows['delta_all']['ridge']['corrected_arm_mse6_over70']:.5f} | {rows['N']['mlp']['corrected_arm_mse6_over70']:.5f} | {rows['delta_all']['mlp']['corrected_arm_mse6_over70']:.5f} | {q[1]:.5f} | {q[3]:.5f} | {m['share_of_error_from_worst_25pct_rows']:.3f} |")
    lines += ['', '## Δ − N（標準指標、エピソード単位 bootstrap 95% CI、負なら Δ が良い）', '',
        '| 分割 | 予測器 | Δ | 平均差 | 95% CI |', '|---|---|---|---:|---|']
    for split,rows in results['splits'].items():
        if split=='selection_audit':continue
        for method in ('ridge','mlp'):
            block=rows.get('paired_corrected_mse_'+method)
            if not block:continue
            for v,ci in block['delta_vs_N_episode_bootstrap'].items():
                lo=ci['ci95_low'];hi=ci['ci95_high']
                lines.append(f"| {split} | {method} | {v} | {ci.get('mean'):.5f} | [{lo:.5f}, {hi:.5f}] |")
    lines += ['', '## グリッパー切替', '', 'AUC は10ステップ内の各位置に対する切替確率で計算。時刻誤差は切替を検出した窓の最初の切替位置の絶対誤差。hold-last は切替なしを予測する基準。', '',
        '| 分割 | 入力 | MLP AUC | 見逃し率 | 検出時刻誤差 (step) | hold-last AUC |', '|---|---|---:|---:|---:|---:|']
    for split,rows in results['splits'].items():
        if split=='selection_audit' or not isinstance(rows,dict):continue
        for v in VARIANTS:
            if v in rows:
                g=rows[v]['grip_switch_mlp'];h=rows[v]['grip_switch_hold_last']
                lines.append(f"| {split} | {v} | {g['switch_auc']} | {g['miss_rate']:.3f} | {g['time_error_steps_mean_when_detected']} | {h['switch_auc']} |")
    lines += ['', '## 解釈上の注意', '', 'N は現在の状態・VLM 平均・履歴と下書きを含み、Δ 版は選択単位の開始時観測との差を追加入力する。Ridge α は Dev の補正 MSE で選択し、MLP は seed 4101 の固定学習設定。結果はオフラインの10ステップ評価であり、Expert への接続や閉ループ性能を示さない。', '']
    (output/'report.md').write_text('\n'.join(lines))


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--device',default='cpu')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    out=run(a.output,smoke=a.smoke,device=a.device)
    print(json.dumps({'output':str(a.output),'models':list(out['models'])}))


if __name__=='__main__':main()
