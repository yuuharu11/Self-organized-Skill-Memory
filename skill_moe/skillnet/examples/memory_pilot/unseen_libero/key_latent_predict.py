#!/usr/bin/env python3
"""JP1: next shared-latent prediction objectives for KEY2-F E encoders.

This module supplies the JEPA-style objective, EMA target, transition builders,
and a complete-data smoke path. Retrieval scoring and metrics deliberately
delegate to ``key_shared_latent`` so that JP1 uses the same pools and evaluator.
The smoke path extracts only frozen VLM prefix features from L90X observations;
it never reads L90X action labels.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import key_shared_latent as key2

ROOT = Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero')
RUN_ROOT = Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/run-01')
MISSING_ROOT = ROOT / 'key-shared-01/attempt-02/prefix-missing'
SEEN_STRIDE5 = ROOT / 'seg-context-01/seg5-cache'
UNSEEN_STRIDE5 = ROOT / 'seg-context-01/unseen-stride5'
VJEPA_EVAL = ROOT / 'key-shared-01/attempt-27-production/vjepa/libero90_eval'
L90X_ROWS = ROOT / 'expert-refine-exp3b-01/l90x_rows'
KEY2_RESULTS = ROOT / 'key-shared-01/attempt-27-production/results.json'
DEFAULT_OUT = ROOT / 'latent-pred-01'


def condition_recipes() -> list[dict[str, Any]]:
    """Declared JP1 ablations; all model scoring is delegated to KEY2-F."""
    recipes=[]
    for mode in ('e_vlm','e_vlm_h'):
        recipes.extend([
            {'name':f'a_{mode}_lret','mode':mode,'stage':'ret_only','predictor_condition':'none'},
            {'name':f'b_{mode}_pred_then_ret','mode':mode,'stage':'pred_then_ret','predictor_condition':'dct'},
            {'name':f'b_{mode}_pred_then_ret_unconditioned','mode':mode,'stage':'pred_then_ret','predictor_condition':'none'},
            {'name':f'c_{mode}_joint','mode':mode,'stage':'joint','lambda_pred_grid':[.1,.3,1.]},
            {'name':f'd_{mode}_pred_only','mode':mode,'stage':'pred_only','predictor_condition':'none'},
            {'name':f'e_{mode}_l90x_pred_then_ret','mode':mode,'stage':'l90x_pred_then_ret','l90x_labels':'forbidden'},
        ])
    return recipes


class NextLatentPredictor(nn.Module):
    """Predict all K target tokens from current E tokens and optional DCT."""
    def __init__(self, token_count: int = 5, dim: int = 256, condition_dim: int = 0,
                 hidden_dim: int = 1024):
        super().__init__()
        self.token_count = int(token_count)
        self.dim = int(dim)
        self.condition_dim = int(condition_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(token_count * dim + condition_dim),
            nn.Linear(token_count * dim + condition_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, token_count * dim),
        )

    def forward(self, z: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        if z.ndim != 3 or z.shape[1:] != (self.token_count, self.dim):
            raise ValueError(f'expected z [B,{self.token_count},{self.dim}], got {tuple(z.shape)}')
        x = z.flatten(1)
        if self.condition_dim:
            if condition is None or condition.shape != (len(z), self.condition_dim):
                raise ValueError('conditioned predictor requires aligned DCT conditioning')
            x = torch.cat([x, condition.float()], dim=-1)
        elif condition is not None:
            raise ValueError('unconditioned predictor cannot consume a condition')
        # Keep the predictor's raw scale for VICReg variance/covariance
        # regularization. The prediction-matching term normalizes internally.
        return self.net(x).reshape(-1, self.token_count, self.dim)


def vicreg_prediction_loss(pred: torch.Tensor, target: torch.Tensor, *,
                           variance_weight: float = 1.0,
                           covariance_weight: float = 0.04,
                           variance_floor: float = 1.0) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Token-aligned normalized prediction loss plus anti-collapse VICReg terms."""
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError('predicted and target latents must have identical [B,K,D] shapes')
    p = F.normalize(pred.float(), dim=-1)
    t = F.normalize(target.detach().float(), dim=-1)
    pred_loss = (p - t).square().mean()
    # VICReg must see unnormalized predictor outputs. Applying its variance
    # floor to unit-normalized tokens makes the requested std=1 unattainable
    # across hundreds of dimensions and turns anti-collapse into a constant
    # pressure to inflate a representation that was already normalized away.
    flat = pred.float().reshape(-1, pred.shape[-1])
    centered = flat - flat.mean(0, keepdim=True)
    std = torch.sqrt(centered.var(0, unbiased=False) + 1e-4)
    var_loss = F.relu(float(variance_floor) - std).mean()
    if len(flat) > 1:
        cov = centered.T @ centered / max(1, len(flat) - 1)
        cov = cov - torch.diag_embed(torch.diagonal(cov))
        cov_loss = cov.square().sum() / cov.shape[0]
    else:
        cov_loss = pred_loss * 0.
    loss = pred_loss + float(variance_weight) * var_loss + float(covariance_weight) * cov_loss
    return loss, {'prediction': pred_loss, 'variance': var_loss, 'covariance': cov_loss}


@torch.no_grad()
def update_ema(target: nn.Module, online: nn.Module, decay: float = 0.996) -> None:
    """EMA-copy online weights into a gradient-free target encoder."""
    if not 0.0 <= decay < 1.0:
        raise ValueError('EMA decay must be in [0,1)')
    src = dict(online.named_parameters())
    for name, p in target.named_parameters():
        if name not in src or p.shape != src[name].shape:
            raise ValueError(f'EMA parameter mismatch: {name}')
        p.mul_(decay).add_(src[name].detach(), alpha=1.0 - decay)
    src_buffers = dict(online.named_buffers())
    for name, b in target.named_buffers():
        if name in src_buffers:
            b.copy_(src_buffers[name])
    target.eval()
    for p in target.parameters():
        p.requires_grad_(False)


def make_ema_target(encoder: nn.Module) -> nn.Module:
    import copy
    target = copy.deepcopy(encoder).eval()
    for p in target.parameters():
        p.requires_grad_(False)
    return target


def next_boundary_pairs(source: np.ndarray, *, eval_pairs: set[tuple[int, int]] | None = None
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return unit-start rows and the next unit-start frame within each demo.

    ``source`` columns are KEY2-F ``(task, episode, start_abs, end_abs)``.
    End frames must exactly match the next unit start; final units without a
    following unit are omitted. Fixed learned-evaluation episodes are excluded.
    """
    src = np.asarray(source, np.int64)
    if src.ndim != 2 or src.shape[1] != 4:
        raise ValueError('source must be [N,4] (task,episode,start,end)')
    starts = {(int(t), int(e), int(s)): i for i, (t, e, s, _end) in enumerate(src)}
    if len(starts) != len(src):
        raise ValueError('duplicate memory unit start')
    q, target, following = [], [], []
    held = eval_pairs or set()
    for i, (t, e, s, end) in enumerate(src):
        if (int(t), int(e)) in held:
            continue
        j = starts.get((int(t), int(e), int(end)))
        if j is not None:
            q.append(i); target.append(int(end)); following.append(j)
    return np.asarray(q, np.int64), np.asarray(target, np.int64), np.asarray(following, np.int64)


def tplus10_keys(keys: np.ndarray, *, step: int = 10) -> np.ndarray:
    """Build exact (task,episode,frame+10) observation identities."""
    k = np.asarray(keys, np.int64)
    if k.ndim != 2 or k.shape[1] != 3 or step < 1:
        raise ValueError('keys must be [N,3] and step positive')
    out = k.copy(); out[:, 2] += int(step)
    return out


def match_tplus10_pairs(keys: np.ndarray, available_keys: np.ndarray, *, step: int = 10
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Keep only causal t+10 observation pairs present in the exact cache."""
    src = np.asarray(keys, np.int64); available = np.asarray(available_keys, np.int64)
    if src.ndim != 2 or src.shape[1] != 3 or available.ndim != 2 or available.shape[1] != 3:
        raise ValueError('source and available keys must be [N,3]')
    slots = {tuple(map(int, k)): i for i, k in enumerate(available)}
    q, dst = [], []
    for i, key in enumerate(tplus10_keys(src, step=step)):
        j = slots.get(tuple(map(int, key)))
        if j is not None:
            q.append(i); dst.append(j)
    return np.asarray(q, np.int64), np.asarray(dst, np.int64)


def l90x_transition_indices(rows: dict[str, np.ndarray], *, step: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Pair fit-only L90X observation rows with same-episode t+10 rows.

    No action array is accessed. Selection-task rows are also excluded so that
    the complete observation split is kept out of latent pretraining.
    """
    required = ('task', 'episode', 'frame', 'is_select')
    if any(k not in rows for k in required):
        raise ValueError('L90X rows lack identity or selection fields')
    task, epi, frame = (np.asarray(rows[k]) for k in ('task', 'episode', 'frame'))
    select = np.asarray(rows['is_select'], bool)
    identity = {(int(t), int(e), int(f)): i for i, (t, e, f) in enumerate(zip(task, epi, frame, strict=True))}
    src, dst = [], []
    for i, (t, e, f) in enumerate(zip(task, epi, frame, strict=True)):
        if select[i]:
            continue
        j = identity.get((int(t), int(e), int(f) + int(step)))
        if j is not None and not select[j]:
            src.append(i); dst.append(j)
    return np.asarray(src, np.int64), np.asarray(dst, np.int64)


def choose_top1_dev(configs: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose checkpoint/config only by Dev top-1 arm6 MSE, never a blend."""
    if not configs:
        raise ValueError('no configurations to select')
    valid = [x for x in configs if np.isfinite(float(x['dev_top1_arm6_mse']))]
    if not valid:
        raise ValueError('all Dev top-1 metrics are non-finite')
    return min(valid, key=lambda x: float(x['dev_top1_arm6_mse']))


def train_prediction_step(online: nn.Module, target: nn.Module, predictor: nn.Module,
                          optimizer: torch.optim.Optimizer, current: dict[str, np.ndarray],
                          future: dict[str, np.ndarray], *, device: str,
                          condition: torch.Tensor | None = None, ema_decay: float = .996,
                          variance_weight: float = 1., covariance_weight: float = .04
                          ) -> dict[str, float]:
    """One Lpred optimizer step; target receives stop-gradient EMA latents."""
    online.train(); predictor.train(); target.eval()
    to = lambda x, dtype=None: torch.as_tensor(x, device=device, dtype=dtype)
    def encode(part, module):
        return module(to(part['tokens'], torch.float32), to(part['groups'], torch.long),
                      to(part['history'], torch.float32), to(part['history_mask'], torch.bool),
                      to(part['state'], torch.float32))
    z = encode(current, online)
    with torch.no_grad():
        z_next = encode(future, target)
    p = predictor(z, condition)
    loss, terms = vicreg_prediction_loss(p, z_next, variance_weight=variance_weight,
                                        covariance_weight=covariance_weight)
    optimizer.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(list(online.parameters()) + list(predictor.parameters()), 1.)
    optimizer.step(); update_ema(target, online, ema_decay)
    return {k: float(v.detach().cpu()) for k, v in {'loss': loss, **terms}.items()}


def fit_lpred(online: nn.Module, target: nn.Module, predictor: nn.Module,
              optimizer: torch.optim.Optimizer, pairs: dict[str, Any], *,
              epochs: int, batch_size: int = 64, device: str,
              ema_decay: float = .996, seed: int = 4101,
              variance_weight: float = 1., covariance_weight: float = .04) -> list[dict[str, float]]:
    """Train Lpred from prepacked current/target exact observations.

    ``pairs`` contains aligned current/future dicts with tokens/groups/history/
    history_mask/state and optional ``condition``. L90X pair construction can
    provide the same schema with no action-label fields.
    """
    current, future = pairs['current'], pairs['future']
    n = len(current['tokens'])
    if n < 2 or len(future['tokens']) != n:
        raise ValueError('Lpred fit requires at least two aligned transition pairs')
    for key in ('tokens','groups','history','history_mask','state'):
        if key not in current or key not in future or len(current[key]) != n or len(future[key]) != n:
            raise ValueError(f'missing/misaligned Lpred field {key}')
    cond = pairs.get('condition')
    if cond is not None and len(cond) != n:
        raise ValueError('DCT condition rows do not align to transitions')
    rng=np.random.default_rng(seed); history=[]
    for epoch in range(int(epochs)):
        order=rng.permutation(n); logs=[]
        for start in range(0,n,int(batch_size)):
            ix=order[start:start+int(batch_size)]
            part=lambda d:{k:np.asarray(v)[ix] for k,v in d.items()}
            c=torch.as_tensor(np.asarray(cond)[ix],device=device,dtype=torch.float32) if cond is not None else None
            logs.append(train_prediction_step(online,target,predictor,optimizer,part(current),part(future),
                device=device,condition=c,ema_decay=ema_decay,
                variance_weight=variance_weight,covariance_weight=covariance_weight))
        history.append({'epoch':epoch+1,**{k:float(np.mean([x[k] for x in logs])) for k in logs[0]}})
    return history


def evaluate_top1_dev(model: nn.Module, mode: str, bank: dict[str, np.ndarray],
                      bank_input: dict[str, np.ndarray], dev: dict[str, np.ndarray],
                      *, device: str) -> dict[str, Any]:
    """Use KEY2-F score/error helpers and return top-1-only Dev selection metric."""
    scores, errors = key2._score_model(model, mode, bank, bank_input, dev, device)
    chosen=np.argmax(scores,axis=1)
    top1=np.asarray(errors)[np.arange(len(chosen)),chosen]
    return {'dev_top1_arm6_mse':float(top1.mean()),'scores':scores,'errors':errors,
            'top1_candidate':chosen}


def _fit_lret(model: nn.Module, mode: str, bank: dict[str, np.ndarray],
              bank_input: dict[str, np.ndarray], train: dict[str, np.ndarray], *,
              seed: int, learning_rate: float, epochs: int, device: str,
              teacher_temperature: float, random_candidates: int = 256,
              steps_per_epoch: int | None = None, max_train_queries: int | None = None
              ) -> tuple[nn.Module, list[dict[str, float]]]:
    """Lret fine-tune for b/c/e; a uses KEY2-F's original trainer verbatim."""
    torch.manual_seed(seed); rng=np.random.default_rng(seed+101)
    model=model.to(device); opt=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=1e-4)
    qids=np.arange(len(train['key']))
    if max_train_queries is not None:qids=qids[:int(max_train_queries)]
    steps=int(steps_per_epoch or max(1,int(np.ceil(len(qids)/8))))
    cand_actions=key2._candidate_window_actions(np.arange(len(bank['length'])),bank)
    hist=[]
    for epoch in range(int(epochs)):
        model.train();rng.shuffle(qids);losses=[]
        for step in range(steps):
            qi=qids[(step*8)%len(qids):min((step+1)*8,len(qids))]
            if len(qi)==0:continue
            random_ids=rng.choice(len(bank['length']),size=min(int(random_candidates),len(bank['length'])),replace=False)
            cids,allowed=key2.contrastive_candidate_set(qi,random_ids,train['task'][qi],train['episode'][qi],bank['task'],bank['episode'])
            qpart=key2._pack_query_observation(train,qi)
            qpart['history'],qpart['history_mask']=key2._history_corrupt(qpart['history'],qpart['history_mask'],rng)
            kpart=key2._pack_query_observation(bank_input,cids)
            qz=key2._model_batch_encode(model,mode,qpart,device)
            kz=key2._model_batch_encode(model,mode,kpart,device)
            score=key2._scale_logits(model,(qz[:,None]*kz[None]).sum(-1).mean(-1))
            errors=key2._arm6_candidate_mse(train['actions'][qi],cand_actions[cids])
            loss=key2.contrastive_teacher_loss(score,torch.as_tensor(errors,device=device),torch.as_tensor(allowed,device=device),
                objective='soft',teacher_temperature=teacher_temperature)
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            key2._clamp_logit_scale_(model);losses.append(float(loss.detach()))
        hist.append({'epoch':epoch+1,'loss':float(np.mean(losses)) if losses else float('nan'),'steps':len(losses)})
    return model,hist


def _joint_prediction_retrieval_step(model: nn.Module, target: nn.Module, predictor: nn.Module,
        optimizer: torch.optim.Optimizer, current: dict[str,np.ndarray], future: dict[str,np.ndarray],
        condition: torch.Tensor | None, bank: dict[str,np.ndarray], bank_input: dict[str,np.ndarray],
        train: dict[str,np.ndarray], query_ids: np.ndarray, candidate_ids: np.ndarray, *,
        mode: str, device: str, lambda_pred: float, teacher_temperature: float,
        ema_decay: float=.996) -> dict[str,float]:
    """One joint Lpred+Lret update on legal memory-only retrieval candidates."""
    qpart=key2._pack_query_observation(train,query_ids);kpart=key2._pack_query_observation(bank_input,candidate_ids)
    qz=key2._model_batch_encode(model,mode,qpart,device)
    kz=key2._model_batch_encode(model,mode,kpart,device)
    score=key2._scale_logits(model,(qz[:,None]*kz[None]).sum(-1).mean(-1))
    action=key2._candidate_window_actions(candidate_ids,bank)
    err=key2._arm6_candidate_mse(train['actions'][query_ids],action)
    allowed=~((train['task'][query_ids,None]==bank['task'][candidate_ids][None]) &
              (train['episode'][query_ids,None]==bank['episode'][candidate_ids][None]))
    lret=key2.contrastive_teacher_loss(score,torch.as_tensor(err,device=device),torch.as_tensor(allowed,device=device),
        objective='soft',teacher_temperature=teacher_temperature)
    z=model(torch.as_tensor(current['tokens'],device=device,dtype=torch.float32),
            torch.as_tensor(current['groups'],device=device,dtype=torch.long),
            torch.as_tensor(current['history'],device=device,dtype=torch.float32),
            torch.as_tensor(current['history_mask'],device=device,dtype=torch.bool),
            torch.as_tensor(current['state'],device=device,dtype=torch.float32))
    with torch.no_grad():
        zt=target(torch.as_tensor(future['tokens'],device=device,dtype=torch.float32),
                  torch.as_tensor(future['groups'],device=device,dtype=torch.long),
                  torch.as_tensor(future['history'],device=device,dtype=torch.float32),
                  torch.as_tensor(future['history_mask'],device=device,dtype=torch.bool),
                  torch.as_tensor(future['state'],device=device,dtype=torch.float32))
    pred=predictor(z,condition);lpred,parts=vicreg_prediction_loss(pred,zt)
    loss=lret+float(lambda_pred)*lpred
    optimizer.zero_grad(set_to_none=True);loss.backward()
    torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(predictor.parameters()),1.)
    optimizer.step();key2._clamp_logit_scale_(model);update_ema(target,model,ema_decay)
    return {'loss':float(loss.detach()),'lret':float(lret.detach()),'lpred':float(lpred.detach()),
            **{k:float(v.detach()) for k,v in parts.items()}}


def _slice_rows(part: dict[str,np.ndarray], n: int) -> dict[str,np.ndarray]:
    anchor=part.get('key',part.get('length'))
    if anchor is None:raise ValueError('cannot infer row count for slice')
    return {k:(v[:n] if isinstance(v,np.ndarray) and len(v)==len(anchor) else v)
            for k,v in part.items()}


def run_condition_smoke(output: Path, *, device: str='cuda:0', batch_size: int=8,
                        lambda_pred: float=.3) -> dict[str,Any]:
    """Train/Dev-select/Eval-smoke all a-e recipes on small real-data subsets.

    This is an execution-path check, not the production run: one optimizer
    epoch/update per recipe, 16 Dev rows and 8 Eval rows. The final evaluation
    calls are made only after each recipe's Dev top-1 metric has been recorded.
    """
    out=Path(output)
    if out.exists() and any(p.name!='run.log' for p in out.iterdir()):
        raise FileExistsError(f'refusing to overwrite {out}')
    out.mkdir(parents=True,exist_ok=True)
    bank,bi,fit_bank,fit_bi,train,learned,unseen,dev=key2._build_key2_queries(
        RUN_ROOT,MISSING_ROOT,SEEN_STRIDE5,UNSEEN_STRIDE5,VJEPA_EVAL)
    bank=_slice_rows(bank,96);bi=_slice_rows(bi,96);fit_bank=_slice_rows(fit_bank,96);fit_bi=_slice_rows(fit_bi,96)
    train=_slice_rows(train,32);dev=_slice_rows(dev,16);learned=_slice_rows(learned,8);unseen=_slice_rows(unseen,8)
    trans=np.load(DEFAULT_OUT/'memory_transitions.npz',allow_pickle=False)
    qfull=np.asarray(trans['unit_indices']);jfull=np.asarray(trans['next_unit_indices'])
    valid=np.flatnonzero((qfull<32)&(jfull<96))
    if len(valid)<8:raise AssertionError('insufficient real boundary pairs for condition smoke')
    ids=valid[:8];qi=qfull[ids];nj=jfull[ids]
    mem_pairs={'current':key2._pack_query_observation(bi,qi),'future':key2._pack_query_observation(bi,nj),
               'condition':np.asarray(trans['dct_condition'][ids],np.float32)}
    lroot=DEFAULT_OUT/'features/l90x';lt=np.load(lroot/'tokens.npy',mmap_mode='r',allow_pickle=False);lg=np.load(lroot/'groups.npy',mmap_mode='r',allow_pickle=False)
    lrows={k:np.load(L90X_ROWS/f'{k}.npy',mmap_mode='r',allow_pickle=False)
           for k in ('is_select','hist','has_hist','state')}
    fitrows=np.flatnonzero(~np.asarray(lrows['is_select'],bool))
    lp=np.load(DEFAULT_OUT/'l90x_transitions.npz',allow_pickle=False);ls=np.asarray(lp['source_indices'][:8]);ld=np.asarray(lp['target_indices'][:8])
    def lpart(ix):
        raw=fitrows[ix];hist=np.asarray(lrows['hist'][raw],np.float32)
        return {'tokens':np.asarray(lt[ix]),'groups':np.asarray(lg[ix]),'history':hist,
          'history_mask':np.repeat(np.asarray(lrows['has_hist'][raw],bool)[:,None],10,axis=1),
          'state':np.asarray(lrows['state'][raw,:8],np.float32)}
    l90_pairs={'current':lpart(ls),'future':lpart(ld)}
    rng=np.random.default_rng(4101);reports=[];locked_models=[]
    for mode in ('e_vlm','e_vlm_h'):
      for variant in ('a_ret','b_pred_ret_cond','b_pred_ret','c_joint','d_pred','e_l90x_pred_ret'):
        torch.manual_seed(4101);model=key2.SharedLatentEncoder(input_mode=mode).to(device);train_log=[]
        if variant=='a_ret':
            # This is the exact KEY2-F training implementation, enabling a
            # later full-config parity check without a second trainer.
            model,train_log=key2._train_ranker(mode,fit_bank,fit_bi,train,seed=1,lr=3e-4,epochs=1,
                device=device,steps_per_epoch=1,max_train_queries=8,teacher_temperature=.02)
        elif variant in ('b_pred_ret_cond','b_pred_ret','d_pred','e_l90x_pred_ret'):
            l90x_mode=variant=='e_l90x_pred_ret';conditioned=variant=='b_pred_ret_cond'
            if l90x_mode:
                # Add only observation pairs; no L90X action arrays are loaded.
                # The LIBERO-40 cache may additionally carry V-JEPA fields;
                # this JP1 recipe is E-VLM(/+history), so use the common
                # causal fields only and never synthesize missing modalities.
                cur_keys=sorted(set(mem_pairs['current']) & set(l90_pairs['current']))
                fut_keys=sorted(set(mem_pairs['future']) & set(l90_pairs['future']))
                cur={k:np.concatenate([mem_pairs['current'][k][:4],l90_pairs['current'][k]],axis=0) for k in cur_keys}
                fut={k:np.concatenate([mem_pairs['future'][k][:4],l90_pairs['future'][k]],axis=0) for k in fut_keys}
                pp={'current':cur,'future':fut}
            else: pp={'current':mem_pairs['current'],'future':mem_pairs['future']}
            cond_dim=mem_pairs['condition'].shape[1] if conditioned else 0
            pred=NextLatentPredictor(condition_dim=cond_dim).to(device);target=make_ema_target(model)
            opt=torch.optim.AdamW(list(model.parameters())+list(pred.parameters()),lr=1e-4)
            cond=torch.as_tensor(mem_pairs['condition'],device=device,dtype=torch.float32) if conditioned else None
            if variant!='d_pred':
                if conditioned:
                    pp['condition']=mem_pairs['condition']
                    train_log=fit_lpred(model,target,pred,opt,pp,epochs=1,batch_size=8,device=device,seed=4101)
                else: train_log=fit_lpred(model,target,pred,opt,pp,epochs=1,batch_size=8,device=device,seed=4101)
                model,ret_hist=_fit_lret(model,mode,fit_bank,fit_bi,train,seed=4101,learning_rate=1e-4,
                    epochs=1,device=device,teacher_temperature=.02,random_candidates=32,steps_per_epoch=1,max_train_queries=8)
                train_log += [{'ret_'+k:v for k,v in h.items()} for h in ret_hist]
            else:
                train_log=fit_lpred(model,target,pred,opt,pp,epochs=1,batch_size=8,device=device,seed=4101)
        else: # c: joint E, predictor, EMA, and KEY2-F soft retrieval loss.
            pred=NextLatentPredictor(condition_dim=0).to(device);target=make_ema_target(model)
            opt=torch.optim.AdamW(list(model.parameters())+list(pred.parameters()),lr=1e-4)
            random_ids=rng.choice(len(fit_bank['length']),size=min(32,len(fit_bank['length'])),replace=False)
            _,allowed_ids=key2.contrastive_candidate_set(np.arange(8),random_ids,train['task'][:8],train['episode'][:8],fit_bank['task'],fit_bank['episode'])
            # Candidate IDs are the returned ID vector from contrastive set; the
            # mask is recomputed in the joint updater by exact task/episode.
            cids=np.unique(np.r_[np.arange(8),random_ids])
            log=_joint_prediction_retrieval_step(model,target,pred,opt,mem_pairs['current'],mem_pairs['future'],None,
                fit_bank,fit_bi,train,np.arange(8),cids,mode=mode,device=device,lambda_pred=lambda_pred,
                teacher_temperature=.02)
            train_log=[log]
        # Dev selects only top-1 MSE. Blend choice is logged independently.
        dev_result=evaluate_top1_dev(model,mode,bank,bi,dev,device=device)
        blend=key2._select_dev_blends(dev_result['scores'],dev,bank)['selected']
        selected=choose_top1_dev([{'name':variant,'dev_top1_arm6_mse':dev_result['dev_top1_arm6_mse']}])
        # Save small CPU copies, complete all Dev selection first, then enter
        # the locked smoke-Eval stage. Eval values never influence recipe choice.
        locked_models.append((mode,variant,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}))
        reports.append({'mode':mode,'variant':variant,'train_steps':len(train_log),
            'dev_top1_arm6_mse':selected['dev_top1_arm6_mse'],'dev_blend_selected_separately':blend,
            'train_log':train_log})
    # One locked evaluation stage after all smoke configurations are frozen by
    # the Dev-only records above. Each recipe is scored once on the same small
    # diagnostic subset; this is not the final 619-row JP1 evaluation.
    for report,(mode,variant,state) in zip(reports,locked_models,strict=True):
        model=key2.SharedLatentEncoder(input_mode=mode).to(device);model.load_state_dict(state);model.eval()
        ev_scores,ev_errors=key2._score_model(model,mode,bank,bi,unseen,device)
        top=np.argmax(ev_scores,axis=1);top1=ev_errors[np.arange(len(top)),top]
        report.update({'eval_smoke_rows':len(unseen['key']),'eval_top1_arm6_mse_smoke':float(top1.mean()),
            'score_shape':list(ev_scores.shape),'finite':bool(np.isfinite(ev_scores).all())})
        del model
    result={'status':'all_conditions_train_dev_eval_smoke_complete','seed':4101,'smoke_train_rows':32,
       'dev_rows':16,'eval_smoke_rows':8,'lambda_pred':lambda_pred,'recipes':reports,
       'selection':'per-recipe min Dev top1 arm6 MSE; blend k/temperature separately selected by KEY2-F Dev helper',
       'condition_a_trainer':'key_shared_latent._train_ranker called verbatim; full 16-epoch KEY2-F parity remains a production-size check',
       'l90x_action_labels_loaded':False}
    (out/'training_smoke.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    return result


def run_key2_retrain_parity(output: Path, *, device: str='cuda:0') -> dict[str,Any]:
    """Retrain KEY2-F condition a at its locked config and compare saved weights.

    This is a parity audit, not a new Eval-driven selection: the mode-specific
    LR/teacher temperature/16 epochs are read from KEY2-F's existing Dev
    selection manifest. The locked Eval split is scored only after weight
    comparison and is emitted as a diagnostic.
    """
    out=Path(output)
    if out.exists() and any(p.name!='run.log' for p in out.iterdir()):
        raise FileExistsError(f'refusing to overwrite {out}')
    out.mkdir(parents=True,exist_ok=True)
    bank,bi,fit_bank,fit_bi,train,learned,unseen,dev=key2._build_key2_queries(
        RUN_ROOT,MISSING_ROOT,SEEN_STRIDE5,UNSEEN_STRIDE5,VJEPA_EVAL)
    selection_path=ROOT/'key-shared-01/attempt-27-production/selected_configs.json'
    cfgs=json.loads(selection_path.read_text())['selected']
    result={'status':'running','seed':1,'source_selection_manifest':str(selection_path),'modes':{}}
    cp=out/'parity_progress.json'
    for mode in ('e_vlm','e_vlm_h'):
        cfg=cfgs[mode]
        if int(cfg['epochs'])!=16:raise AssertionError('unexpected KEY2-F selected epoch count')
        if mode=='e_vlm' and (float(cfg['learning_rate'])!=3e-4 or float(cfg['teacher_temperature'])!=.02):
            raise AssertionError('E-VLM parity settings differ from locked KEY2-F config')
        torch.manual_seed(1);np.random.seed(1)
        model,history=key2._train_ranker(mode,fit_bank,fit_bi,train,seed=1,
            lr=float(cfg['learning_rate']),epochs=int(cfg['epochs']),device=device,
            teacher_temperature=float(cfg['teacher_temperature']))
        saved_path=Path(cfg['checkpoint'])
        reference=torch.load(saved_path,map_location='cpu',weights_only=False)
        fitted=model.state_dict();common=set(reference)&set(fitted)
        if len(common)!=len(fitted):raise AssertionError('KEY2-F checkpoint parameter key mismatch')
        max_abs=max(float((fitted[k].detach().cpu()-reference[k]).abs().max()) for k in common)
        mean_abs=float(np.mean([float((fitted[k].detach().cpu()-reference[k]).abs().mean()) for k in common]))
        dev_scores,dev_errors=key2._score_model(model,mode,bank,bi,dev,device)
        dev_top=dev_errors[np.arange(len(dev)),np.argmax(dev_scores,axis=1)]
        # Diagnostic scoring begins only after exact training config and
        # checkpoint-distance measurements are fixed.
        eval_scores,eval_errors=key2._score_model(model,mode,bank,bi,unseen,device)
        eval_top=eval_errors[np.arange(len(unseen['key'])),np.argmax(eval_scores,axis=1)]
        result['modes'][mode]={'config':{k:cfg[k] for k in ('learning_rate','epochs','teacher_temperature')},
            'reference_checkpoint':str(saved_path),'train_epochs':len(history),
            'max_abs_parameter_delta_vs_KEY2F_checkpoint':max_abs,
            'mean_parameter_abs_delta_vs_KEY2F_checkpoint':mean_abs,
            'dev_top1_arm6_mse':float(dev_top.mean()),'eval619_top1_arm6_mse_diagnostic':float(eval_top.mean()),
            'train_loss_first_last':[history[0]['loss'],history[-1]['loss']]}
        # Persist mode-by-mode to make a long parity audit restartable.
        result['status']=f'{mode}_complete'
        cp.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
        del model
    result['status']='complete'
    cp.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    return result


def retrieval_objective_step(model: nn.Module, bank: dict[str, np.ndarray], bank_input: dict[str, np.ndarray],
                             train: dict[str, np.ndarray], query_ids: np.ndarray,
                             candidate_ids: np.ndarray, *, device: str,
                             teacher_temperature: float = .02) -> torch.Tensor:
    """KEY2-F soft InfoNCE for E; metric/evaluator remains key2._score_model."""
    qids = np.asarray(query_ids, np.int64); cids = np.asarray(candidate_ids, np.int64)
    q = key2._pack_query_observation(train, qids)
    k = key2._pack_query_observation(bank_input, cids)
    qz = key2._model_batch_encode(model, 'e_vlm_h', q, device)
    kz = key2._model_batch_encode(model, 'e_vlm_h', k, device)
    score = (qz[:, None] * kz[None]).sum(-1).mean(-1)
    score = key2._scale_logits(model, score)
    actions = key2._candidate_window_actions(cids, bank)
    err = key2._arm6_candidate_mse(train['actions'][qids], actions)
    allowed = ~((train['task'][qids, None] == bank['task'][cids][None]) &
                (train['episode'][qids, None] == bank['episode'][cids][None]))
    return key2.contrastive_teacher_loss(score, torch.as_tensor(err,device=device), torch.as_tensor(allowed, device=device),
                                        objective='soft', teacher_temperature=teacher_temperature)


def extract_l90x_prefix_cache(row_root: Path, output: Path, *, batch_size: int = 4,
                              memory_fraction: float = .25) -> dict[str, Any]:
    """Extract exact VLM prefix tokens for L90X fit observations only.

    Cache contains only prefix tokens/groups; action labels are not opened.
    The file is resumable by row index and is created under this experiment's
    new output directory, never inside the shared L90X source cache.
    """
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(float(memory_fraction))
    import torch as _torch
    _torch.set_num_threads(8)
    from latent_action_units_la2 import assert_l90x_manifest
    row_root = Path(row_root); output = Path(output)
    manifest = json.loads((row_root / 'manifest.json').read_text())
    assert_l90x_manifest(manifest)
    n = int(manifest['rows'])
    task = np.load(row_root / 'task.npy', mmap_mode='r', allow_pickle=False)
    select = np.load(row_root / 'is_select.npy', mmap_mode='r', allow_pickle=False)
    wanted = np.flatnonzero(~np.asarray(select, bool))
    if not len(wanted):
        raise ValueError('no fit-only L90X observations')
    output.mkdir(parents=True, exist_ok=True)
    tok_path = output / 'tokens.npy'; grp_path = output / 'groups.npy'; prog_path = output / 'progress.json'
    shape = (len(wanted), 532, 2048); gshape = (len(wanted), 532)
    if tok_path.exists() or grp_path.exists():
        tok = np.load(tok_path, mmap_mode='r+', allow_pickle=False)
        grp = np.load(grp_path, mmap_mode='r+', allow_pickle=False)
        if tok.shape != shape or grp.shape != gshape:
            raise ValueError('existing L90X token cache shape mismatch')
    else:
        tok = np.lib.format.open_memmap(tok_path, mode='w+', dtype=np.float16, shape=shape)
        grp = np.lib.format.open_memmap(grp_path, mode='w+', dtype=np.int8, shape=gshape)
    done = int(json.loads(prog_path.read_text()).get('done', 0)) if prog_path.exists() else 0
    if done < 0 or done > len(wanted): raise ValueError('invalid L90X feature progress marker')

    import theta_regression as tr
    base = tr._module(tr.PI05 / 'run.py', 'jp1_pi05_run')
    model, _config = base.load_policy(tr.CHECKPOINT)
    from openpi.shared.nnx_utils import module_jit
    encode = module_jit(model.prefix_features)
    rows = {name: np.load(row_root / f'{name}.npy', mmap_mode='r', allow_pickle=False)
            for name in ('image_base', 'image_wrist', 'state', 'prompt', 'prompt_mask')}
    for start in range(done, len(wanted), int(batch_size)):
        local = np.arange(start, min(start + int(batch_size), len(wanted)), dtype=np.int64)
        rid = wanted[local]
        obs = {'image_base': np.asarray(rows['image_base'][rid]),
               'image_wrist': np.asarray(rows['image_wrist'][rid]),
               'state': np.asarray(rows['state'][rid]), 'prompt': np.asarray(rows['prompt'][rid]),
               'prompt_mask': np.asarray(rows['prompt_mask'][rid])}
        hidden, groups = encode(base.obs_batch(obs, np.arange(len(rid))))
        hidden = np.asarray(hidden); groups = np.asarray(groups)
        for j, g in enumerate(groups):
            valid = g >= 0
            count = int(valid.sum())
            if count > 532 or count < 1:
                raise AssertionError(f'L90X valid prefix count outside [1,532]: {count} at row {rid[j]}')
            tok[local[j]] = 0; grp[local[j]] = -1
            tok[local[j], :count] = hidden[j, valid].astype(np.float16)
            grp[local[j], :count] = g[valid].astype(np.int8)
        tok.flush(); grp.flush()
        done = int(local[-1] + 1)
        tmp = prog_path.with_suffix('.tmp')
        tmp.write_text(json.dumps({'done': done, 'total': len(wanted)}) + '\n'); tmp.replace(prog_path)
    del model
    result = {'complete': True, 'rows_total': n, 'rows_fit_encoded': len(wanted),
              'rows_selection_excluded': n-len(wanted), 'task_ids': sorted(set(map(int, task[wanted]))),
              'tokens_shape': list(shape), 'groups_shape': list(gshape), 'dtype': 'float16/int8',
              'actions_read': False, 'source_manifest': str(row_root / 'manifest.json')}
    (output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def _all_tokens_forward(model: nn.Module, part: dict[str, np.ndarray], *, device: str,
                        batch_size: int = 8) -> tuple[int, float]:
    """Forward each provided real row once and return count/min token std."""
    model.eval(); n = len(part['tokens']); reps = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            ids = np.arange(start, min(start + batch_size, n))
            packed = key2._pack_query_observation(part, ids)
            z = key2._model_batch_encode(model, 'e_vlm_h', packed, device)
            if not torch.isfinite(z).all(): raise FloatingPointError('non-finite E output')
            reps.append(z.detach().float().cpu())
    joined = torch.cat(reps, dim=0)
    std = float(joined.std(dim=0, unbiased=False).mean())
    if std < 1e-4: raise AssertionError(f'collapsed latent outputs: mean std={std}')
    return n, std


def run_full_smoke(output: Path, *, device: str = 'cuda:0', batch_size: int = 8,
                   l90x_root: Path = L90X_ROWS, extract_l90x: bool = True) -> dict[str, Any]:
    """Exercise exact real-data row/cache/transition/model paths once; no main training."""
    output = Path(output)
    if output.exists() and ((output/'results.json').exists() or (output/'report.md').exists()):
        raise FileExistsError(f'refusing to overwrite completed output {output}')
    output.mkdir(parents=True, exist_ok=True)
    paths = {'run_root': RUN_ROOT, 'missing_root': MISSING_ROOT, 'seen_stride5': SEEN_STRIDE5,
             'unseen_stride5': UNSEEN_STRIDE5, 'vjepa_eval_cache': VJEPA_EVAL}
    bank, bank_input, fit_bank, fit_input, train, learned, unseen, dev = key2._build_key2_queries(**paths)
    controls = json.loads((ROOT / 'compress-seen-cuts-01/production-control/evaluation-selection.json').read_text())
    held = {tuple(map(int, x)) for x in controls['seen_demo_pairs']}
    qi, target_frame, next_idx = next_boundary_pairs(bank['source'], eval_pairs=held)
    if len(qi) == 0: raise AssertionError('no memory next-unit boundaries')
    target_keys = np.column_stack([bank['task'][qi], bank['episode'][qi], target_frame]).astype(np.int32)
    if not np.array_equal(target_keys, bank['source'][next_idx][:, [0, 1, 2]]):
        raise AssertionError('boundary target is not the exact next-unit start')
    coeff = np.asarray(bank['coefficients'][qi], np.float32).reshape(len(qi), -1)
    source_keys = bank['source'][qi][:, [0, 1, 2]]
    # Cache identities are exact dataset keys. t+10 pairs are a separate
    # self-supervised target; they are never confused with the next-unit target.
    available = {(int(t),int(e),int(f)) for t,e,f in zip(bank_input['key'][:,0],bank_input['key'][:,1],bank_input['key'][:,2],strict=True)}
    for cache_root in (RUN_ROOT, MISSING_ROOT, SEEN_STRIDE5):
        cache_root=Path(cache_root)
        if (cache_root/'rows/task.npy').exists():
            rt=np.load(cache_root/'rows/task.npy',mmap_mode='r',allow_pickle=False)
            re=np.load(cache_root/'rows/episode.npy',mmap_mode='r',allow_pickle=False)
            rf=np.load(cache_root/'rows/frame.npy',mmap_mode='r',allow_pickle=False)
            available.update((int(t),int(e),int(f)) for t,e,f in zip(rt,re,rf,strict=True))
        elif (cache_root/'manifest.json').exists():
            available.update(tuple(map(int,k)) for k in json.loads((cache_root/'manifest.json').read_text()).get('keys',()))
    avkeys=np.asarray(sorted(available),np.int64)
    t10_q,t10_dst=match_tplus10_pairs(source_keys,avkeys)
    t10_keys=tplus10_keys(source_keys[t10_q])
    np.savez_compressed(output / 'memory_transitions.npz',
        current_keys=bank['source'][qi][:, [0, 1, 2]], target_keys=target_keys,
        unit_indices=qi, next_unit_indices=next_idx, dct_condition=coeff,
        tplus10_source_indices=t10_q,tplus10_current_keys=source_keys[t10_q],
        tplus10_target_keys=t10_keys)

    l90x_meta = None; l90x_root = Path(l90x_root)
    if extract_l90x:
        l90x_meta = extract_l90x_prefix_cache(l90x_root, output / 'features/l90x')
        ltok = np.load(output / 'features/l90x/tokens.npy', mmap_mode='r', allow_pickle=False)
        lgrp = np.load(output / 'features/l90x/groups.npy', mmap_mode='r', allow_pickle=False)
        lrows = {k: np.load(l90x_root / f'{k}.npy', mmap_mode='r', allow_pickle=False)
                 for k in ('task','episode','frame','is_select','hist','state','has_hist')}
        l90x_fit_rows=np.flatnonzero(~np.asarray(lrows['is_select'],bool))
        src, dst = l90x_transition_indices({k:np.asarray(v)[l90x_fit_rows]
            for k,v in lrows.items()})
        if not len(src): raise AssertionError('no L90X t+10 fit-only pairs')
        np.savez_compressed(output / 'l90x_transitions.npz', source_indices=src, target_indices=dst,
                            source_row_indices=l90x_fit_rows[src],target_row_indices=l90x_fit_rows[dst])
        l90x_t10 = int(len(src))
    else:
        l90x_t10 = 0

    torch.manual_seed(4101)
    encoders = {}
    counts = {}
    for mode in ('e_vlm', 'e_vlm_h'):
        model = key2.SharedLatentEncoder(input_mode=mode).to(device)
        # E-VLM ignores causal fields; E-VLM+H uses them. Both also process
        # the exact unit bank and each fixed query split from the evaluator.
        counts[mode] = {}
        for name, part in (('memory_bank', bank_input), ('fit_queries', train),
                           ('learned_eval', learned), ('unseen_eval', unseen), ('dev', dev)):
            counts[mode][name], _ = _all_tokens_forward(model, part, device=device, batch_size=batch_size)
        if extract_l90x:
            model.eval(); zstd=[]; seen=0
            with torch.no_grad():
                for start in range(0, len(ltok), batch_size):
                    end=min(start+batch_size,len(ltok))
                    z=model(vlm_tokens=torch.as_tensor(np.asarray(ltok[start:end]),device=device,dtype=torch.float32),
                            groups=torch.as_tensor(np.asarray(lgrp[start:end]),device=device,dtype=torch.long),
                            history=None if mode=='e_vlm' else torch.as_tensor(np.asarray(lrows['hist'][np.flatnonzero(~np.asarray(lrows['is_select'],bool))[start:end]]),device=device,dtype=torch.float32),
                            history_mask=None if mode=='e_vlm' else torch.as_tensor(np.repeat(np.asarray(lrows['has_hist'][np.flatnonzero(~np.asarray(lrows['is_select'],bool))[start:end]])[:,None],10,axis=1),device=device,dtype=torch.bool),
                            state=None if mode=='e_vlm' else torch.as_tensor(np.asarray(lrows['state'][np.flatnonzero(~np.asarray(lrows['is_select'],bool))[start:end],:8]),device=device,dtype=torch.float32))
                    if not torch.isfinite(z).all(): raise FloatingPointError('non-finite L90X latent')
                    zstd.append(z.float().cpu());seen+=len(z)
            s=float(torch.cat(zstd).std(0,unbiased=False).mean())
            if s<1e-4:raise AssertionError('L90X latent collapsed')
            counts[mode]['l90x_fit_observations']=seen
        encoders[mode]=model

    # A one-batch genuine Lpred optimizer step validates all differentiable
    # pieces without starting the requested production training.
    smoke_n=min(8,len(qi))
    current_part = key2._pack_query_observation(bank_input, qi[:smoke_n])
    future_part = key2._pack_query_observation(bank_input, next_idx[:smoke_n])
    cdim=coeff.shape[1]
    pred=NextLatentPredictor(condition_dim=cdim).to(device)
    target=make_ema_target(encoders['e_vlm_h'])
    optimizer=torch.optim.AdamW(list(encoders['e_vlm_h'].parameters())+list(pred.parameters()),lr=1e-4)
    condition=torch.as_tensor(coeff[:smoke_n],device=device,dtype=torch.float32)
    losses=train_prediction_step(encoders['e_vlm_h'],target,pred,optimizer,current_part,future_part,
                                 device=device,condition=condition)
    # Check the shared evaluator can score real fixed Eval arrays; labels are
    # not consulted by this shape/finite smoke and no scores are used to tune.
    scores, errors=key2._score_model(encoders['e_vlm'], 'e_vlm', bank, bank_input, unseen, device)
    if scores.shape != (len(unseen['key']),len(bank['length'])) or not np.isfinite(scores).all():
        raise AssertionError('delegated KEY2 evaluator shape/finite check failed')
    results={
      'status':'smoke_complete_main_training_not_run',
      'reference':'KEY2-F evaluation and scoring implementation reused; no JP1 production checkpoint or Eval-driven selection was run.',
      'seed':4101,'l90x_manifest_assertion':'passed; fit only, Dev/Eval tasks excluded',
      'l90x_prefix':l90x_meta,'l90x_tplus10_fit_pairs':l90x_t10,
      'memory':{'bank_units':int(len(bank['length'])),'fit_queries':int(len(train['key'])),
        'learned_eval_rows':int(len(learned['key'])),'unseen_eval_rows':int(len(unseen['key'])),
        'dev_rows':int(len(dev['key'])),'memory_boundary_pairs_excluding_learned_eval':int(len(qi)),
        'boundary_targets_are_next_unit_starts':True,'boundary_target_features_exact':True,
        'tplus10_exact_pairs_available_at_unit_starts':int(len(t10_q))},
      'full_row_forward_counts_by_model':counts,
      'lpred_one_update':losses,'ret_eval_shape':list(scores.shape),
      'ret_eval_scores_finite':True,'selection_rule':'min Dev top1_arm6_mse only; top20 blend selected separately on Dev',
      'production_training':'not run by instruction; review required first',
      'leakage':'L90X labels are not loaded for feature extraction/pairs; Eval action labels are not used for model updates or configuration selection. KEY2 scorer computes Eval errors only as a diagnostic return.'}
    (output/'results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    lines=['# JP1 共有潜在トークンの次潜在予測：実装 smoke','',
      '## 主表','',
      '|状態|内容|件数・値|','|---|---|---:|',
      f"|KEY2-F bank|DCT3 可変単位|{len(bank['length'])}|",
      f"|学習クエリ|固定 learned 評価デモを除外|{len(train['key'])}|",
      f"|学習済み評価|正解ラベルを選択に使わず smoke forward|{len(learned['key'])}|",
      f"|未学習 Eval|正解ラベルを選択に使わず smoke forward|{len(unseen['key'])}|",
      f"|LIBERO-90 Dev|選択候補データ、smoke forward のみ|{len(dev['key'])}|",
      f"|L90X prefix|Dev/Eval 除外、行動ラベル非読込|{l90x_meta['rows_fit_encoded'] if l90x_meta else 'skip'}|",
      f"|メモリ境界遷移|評価デモ除外、次単位開始の特徴あり|{len(qi)}|",
      f"|L90X t+10 遷移|fit task 内、行動非参照|{l90x_t10}|",
      f"|1-step Lpred smoke|loss / prediction / variance / covariance|{losses['loss']:.6g} / {losses['prediction']:.6g} / {losses['variance']:.6g} / {losses['covariance']:.6g}|",
      '', '## 結果と範囲', '',
      '実データの全対象行について特徴を一度通す smoke、EMA 付き予測器の1更新、KEY2-F の scorer 呼び出しまで完了。これは学習結果ではなく、条件 a〜e の本番学習・Dev 選択・Eval 一回評価は実行していない。',
      '本番では E-VLM と E-VLM+H をそれぞれ用い、a: Lret 再学習、b: Lpred 事前学習→Lret、c: 同時学習、d: Lpred のみ、e: L90X 観測を足した b を構成する。条件付き予測器には実行単位の DCT 係数を与える。',
      'KEY2-F と同じ検索評価関数 `_score_model`, `_summarize_split`, `candidate_rank_metrics` を利用する。主選択は Dev top-1 腕 MSE、top-20 blend の k/温度は別途 Dev で選ぶ。Eval は選択後に一度だけ使う。',
      '', '## 実行上の注意', '',
      'L90X の全 fit 観測 prefix は JP1 専用 feature cache に保存し、L90X source cache は読み取り専用。レジューム点は `features/l90x/progress.json`。本番学習コマンドとハイパーパラメータ探索はレビュー後に設定して実行する。',
      '']
    (output/'report.md').write_text('\n'.join(lines))
    return results


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--smoke',action='store_true',help='run complete-data feature/model smoke, no production training')
    ap.add_argument('--condition-smoke',action='store_true',help='run one-update a-e train/Dev/Eval path smoke')
    ap.add_argument('--key2-parity',action='store_true',help='retrain a at locked KEY2-F config and compare weights')
    ap.add_argument('--output',type=Path,default=DEFAULT_OUT)
    ap.add_argument('--condition-output',type=Path,default=DEFAULT_OUT/'review-smoke-01')
    ap.add_argument('--parity-output',type=Path,default=DEFAULT_OUT/'review-parity-01')
    ap.add_argument('--l90x-root',type=Path,default=L90X_ROWS)
    ap.add_argument('--batch-size',type=int,default=8)
    ap.add_argument('--memory-fraction',type=float,default=.25)
    ap.add_argument('--skip-l90x-extract',action='store_true')
    args=ap.parse_args()
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']=str(args.memory_fraction)
    torch.set_num_threads(8)
    if args.condition_smoke:
        if torch.cuda.is_available():torch.cuda.set_per_process_memory_fraction(float(args.memory_fraction))
        print(json.dumps(run_condition_smoke(args.condition_output,device='cuda:0' if torch.cuda.is_available() else 'cpu',
            batch_size=args.batch_size),indent=2,ensure_ascii=False))
        return
    if args.key2_parity:
        if torch.cuda.is_available():torch.cuda.set_per_process_memory_fraction(float(args.memory_fraction))
        print(json.dumps(run_key2_retrain_parity(args.parity_output,device='cuda:0' if torch.cuda.is_available() else 'cpu'),
            indent=2,ensure_ascii=False))
        return
    if not args.smoke:
        ap.error('Select --smoke or --condition-smoke. Full JP1 production fitting is intentionally not started here.')
    print(json.dumps(run_full_smoke(args.output,device='cuda:0',batch_size=args.batch_size,
        l90x_root=args.l90x_root,extract_l90x=not args.skip_l90x_extract),indent=2,ensure_ascii=False))


if __name__=='__main__': main()
