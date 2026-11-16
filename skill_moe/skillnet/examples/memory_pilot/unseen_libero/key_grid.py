#!/usr/bin/env python3
"""Matched input-by-fusion key comparison for KEY2-F retrieval.

Feature construction, soft listwise targets, candidate masks, and evaluation
metrics are delegated to key_shared_latent. This module adds only vector MLP
rankers whose query/key inputs are deliberately matched.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import key_shared_latent as key2

ROOT = Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero')
DEFAULT_OUT = ROOT / 'key-grid-01'
KEY2_ROOT = ROOT / 'key-shared-01/attempt-27-production'

INPUTS = {
    'vlm': ('vlm',),
    'history_state': ('history_state',),
    'vlm_history_state': ('vlm', 'history_state'),
    'vjepa_history_state': ('vjepa', 'history_state'),
    'vjepa_vlm_history_state': ('vjepa', 'vlm', 'history_state'),
}
NEW_INPUTS = ('history_state', 'vjepa_history_state', 'vjepa_vlm_history_state')
EXISTING_E = {
    'vlm': 'e_vlm',
    'history_state': 'e_h',
    'vlm_history_state': 'e_vlm_h',
    'vjepa_history_state': 'e_vj_h',
    'vjepa_vlm_history_state': 'e_vj_vlm_h',
}


class PairMLPRanker(nn.Module):
    """VectorRanker-shaped query/key MLP, optionally with tied towers."""
    def __init__(self, input_dim: int, *, shared: bool, context_dim: int = 256,
                 token_count: int = 5):
        super().__init__()
        self.shared = bool(shared)
        self.token_count = int(token_count)
        self.context_dim = int(context_dim)
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(key2.LOGIT_SCALE_INITIAL))))
        def tower(output_dim):
            return nn.Sequential(nn.Linear(input_dim, 512), nn.GELU(),
                                 nn.Linear(512, output_dim))
        # Match KEY2-F's VectorRanker in the untied condition: one query
        # vector is compared with the mean of the key's token similarities.
        # The tied condition has one shared token-producing tower on both sides.
        self.query_tower = tower(context_dim * token_count if self.shared else context_dim)
        self.key_tower = self.query_tower if self.shared else tower(context_dim * token_count)

    def query(self, x: torch.Tensor) -> torch.Tensor:
        y = self.query_tower(x)
        if self.shared:
            y = y.reshape(-1, self.token_count, self.context_dim)
        else:
            y = y[:, None, :].expand(-1, self.token_count, -1)
        return F.normalize(y, dim=-1)

    def key(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.key_tower(x).reshape(-1, self.token_count, self.context_dim), dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return torch.einsum('bkd,nkd->bn', self.query(q), self.key(k)).div(self.token_count)


def _vjepa_two_camera(tokens: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Return [N,2048] by concatenating base and wrist token means."""
    t=np.asarray(tokens);g=np.asarray(groups)
    if t.ndim!=3 or g.shape!=t.shape[:2] or t.shape[-1]!=1024:
        raise ValueError('expected V-JEPA [N,T,1024] and aligned groups')
    out=np.zeros((len(t),2048),np.float32)
    for camera in (0,1):
        mask=(g==camera)
        if np.any(mask.sum(1)==0):raise ValueError(f'missing V-JEPA camera group {camera}')
        for s in range(0,len(t),64):
            sl=slice(s,min(s+64,len(t)));m=mask[sl]
            out[sl,camera*1024:(camera+1)*1024]=(np.where(m[...,None],t[sl],0).sum(1,dtype=np.float32)/m.sum(1,keepdims=True))
    return out


def make_vector_features(part: dict[str,np.ndarray], input_name: str) -> np.ndarray:
    """Build exactly the declared concatenated input, for queries and keys."""
    if input_name not in INPUTS:raise ValueError(f'unknown input row {input_name}')
    blocks=[]
    for modality in INPUTS[input_name]:
        if modality=='vlm':
            if 'pooled' in part: pooled=np.asarray(part['pooled'],np.float32)
            else: pooled=key2._pool_tokens(part['tokens'],part['groups'])
            blocks.append(pooled)
        elif modality=='vjepa':
            if 'vjepa_vector' in part:blocks.append(np.asarray(part['vjepa_vector'],np.float32))
            else:blocks.append(_vjepa_two_camera(part['vjepa_tokens'],part['vjepa_groups']))
        elif modality=='history_state':
            h=np.asarray(part['history'],np.float32).reshape(len(part['history']),-1)
            hm=np.asarray(part['history_mask'],np.float32).reshape(len(h),-1)
            st=np.asarray(part['state'],np.float32)
            blocks.extend((h,hm,st))
    result=np.concatenate(blocks,axis=-1).astype(np.float32,copy=False)
    if result.ndim!=2 or not np.isfinite(result).all():raise ValueError('nonfinite vector input')
    return result


def exact_candidate_row_ids(query_keys: np.ndarray, candidate_keys: np.ndarray) -> np.ndarray:
    """Map query identities to candidate-bank rows without assuming order.

    Training queries are also positives in the action-teacher candidate bank.
    Joining by the immutable ``(task, episode, frame)`` key makes that
    relationship explicit and prevents a filtered/reordered bank from silently
    assigning a query's positive label to another unit.
    """
    q=np.asarray(query_keys)
    c=np.asarray(candidate_keys)
    if q.ndim!=2 or c.ndim!=2 or q.shape[1]!=c.shape[1]:
        raise ValueError(f"query/candidate keys must be aligned matrices, got {q.shape} and {c.shape}")
    mapping={}
    for i,row in enumerate(c):
        key=tuple(map(int,row))
        if key in mapping:raise ValueError(f"duplicate candidate identity: {key}")
        mapping[key]=i
    result=[]
    for row in q:
        key=tuple(map(int,row))
        if key not in mapping:raise KeyError(f"query identity absent from candidate bank: {key}")
        result.append(mapping[key])
    return np.asarray(result,np.int64)


def _prepare_vector_sources(part: dict[str,np.ndarray]) -> dict[str,np.ndarray]:
    """Compute expensive token reductions once per aligned split."""
    prepared=dict(part)
    if 'tokens' in part and 'groups' in part:
        prepared['pooled']=key2._pool_tokens(part['tokens'],part['groups'])
    if 'vjepa_tokens' in part and 'vjepa_groups' in part:
        prepared['vjepa_vector']=_vjepa_two_camera(part['vjepa_tokens'],part['vjepa_groups'])
    return prepared


def _score_vectors(model: PairMLPRanker, queries: np.ndarray, candidates: np.ndarray,
                   *, device: str, batch: int=128) -> np.ndarray:
    model.eval();keys=[]
    with torch.no_grad():
        for s in range(0,len(candidates),batch):
            x=torch.as_tensor(candidates[s:s+batch],device=device,dtype=torch.float32)
            keys.append(model.key(x).cpu().numpy())
        kt=np.concatenate(keys)
        scale=float(model.logit_scale.detach().exp().clamp(max=key2.LOGIT_SCALE_MAX).cpu())
        result=np.empty((len(queries),len(candidates)),np.float32)
        for s in range(0,len(queries),batch):
            q=torch.as_tensor(queries[s:s+batch],device=device,dtype=torch.float32)
            qr=model.query(q).cpu().numpy()
            result[s:s+len(q)]=np.einsum('qkd,nkd->qn',qr,kt,optimize=True)/qr.shape[1]*scale
    return result


def _train_one(input_name: str, shared: bool, fit_q: np.ndarray, fit_k: np.ndarray,
               train: dict[str,np.ndarray], candidate_actions: np.ndarray, *, seed: int,
               learning_rate: float, epochs: int, teacher_tau: float, device: str,
               query_candidate_ids: np.ndarray,
               batch_size: int=8, random_candidates: int=256) -> tuple[PairMLPRanker,list[dict[str,Any]]]:
    torch.manual_seed(seed);rng=np.random.default_rng(seed+101)
    model=PairMLPRanker(fit_q.shape[1],shared=shared).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=1e-4)
    n=len(train['key']);qids=np.arange(n);steps=max(1,int(np.ceil(n/batch_size)));hist=[]
    for ep in range(epochs):
        model.train();rng.shuffle(qids);losses=[]
        for step in range(steps):
            qi=qids[step*batch_size:min((step+1)*batch_size,n)]
            random_ids=rng.choice(len(fit_k),size=min(random_candidates,len(fit_k)),replace=False)
            cids,allowed=key2.contrastive_candidate_set(query_candidate_ids[qi],random_ids,train['task'][qi],train['episode'][qi],
                                                        train['_candidate_task'],train['_candidate_episode'])
            q=torch.as_tensor(fit_q[qi],device=device,dtype=torch.float32)
            k=torch.as_tensor(fit_k[cids],device=device,dtype=torch.float32)
            scores=model(q,k)*model.logit_scale.detach().exp().clamp(max=key2.LOGIT_SCALE_MAX)
            errors=key2._arm6_candidate_mse(train['actions'][qi],candidate_actions[cids])
            loss=key2.contrastive_teacher_loss(scores,torch.as_tensor(errors,device=device),
                torch.as_tensor(allowed,device=device),objective='soft',teacher_temperature=teacher_tau)
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            key2._clamp_logit_scale_(model);losses.append(float(loss.detach()))
        hist.append({'epoch':ep+1,'loss':float(np.mean(losses)),'steps':len(losses)})
    return model,hist


def _rank_metrics(scores: np.ndarray, errors: np.ndarray) -> dict[str,float]:
    return key2.candidate_rank_metrics(scores,errors,tau=.01,temperature=1.)


def _split_eval(scores: np.ndarray, errors: np.ndarray, split: dict[str,np.ndarray]) -> dict[str,Any]:
    top=np.argmax(scores,axis=1);toperr=errors[np.arange(len(top)),top]
    metrics=_rank_metrics(scores,errors)
    hard=np.asarray(split['hard'],bool)
    metrics['rows']=len(top);metrics['hard_rows']=int(hard.sum())
    metrics['top1_arm6_mse_hard']=float(toperr[hard].mean()) if hard.any() else None
    metrics['rank_hard']=_rank_metrics(scores[hard],errors[hard]) if hard.any() else None
    return {'metrics':metrics,'top1_per_query':toperr}


def _top1_boot(diff: np.ndarray, split: dict[str,np.ndarray], seed: int) -> dict[str,float]:
    return key2.episode_bootstrap_ci(diff,split['task'],split['episode'],reps=5000,seed=seed)


def _existing_dev_selections() -> dict[str,dict[str,Any]]:
    """Reselect saved KEY2-F tuning checkpoints on Dev top-1 MSE."""
    modes=('p0','p5','e_vlm','e_h','e_vlm_h','e_vj_h','e_vj_vlm_h')
    out={}
    for mode in modes:
        rows=[]
        for p in sorted((KEY2_ROOT/'tuning').glob('*/dev_results.json')):
            d=json.loads(p.read_text());m=d.get('models',{}).get(mode)
            if not m:continue
            metric=m.get('dev_selection',{}).get('rank',{}).get('top1_arm6_mse')
            ckpt=m.get('checkpoint')
            if metric is not None and ckpt and Path(ckpt).exists():
                rows.append({'dev_top1_arm6_mse':float(metric),'checkpoint':ckpt,'dir':str(p.parent),
                    'blend_selected':m.get('dev_selection',{}).get('blend_selection',{}).get('selected')})
        if rows:out[mode]=min(rows,key=lambda x:(x['dev_top1_arm6_mse'],x['dir']))
    return out


def run_keygrid(output: Path, *, device: str='cuda:0', seed: int=1,
                learning_rates=(1e-4,3e-4), epochs_grid=(4,8,16),
                tau_grid=(.002,.005,.01,.02), smoke: bool=False,
                only_new: bool=False) -> dict[str,Any]:
    output=Path(output)
    if output.exists() and any(output.iterdir()) and not (output/'progress.jsonl').exists():
        raise FileExistsError(f'refusing to overwrite non-KEYGRID output {output}')
    output.mkdir(parents=True,exist_ok=True)
    run_root=key2.Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/run-01')
    bank,bi,fit_bank,fit_bi,train,learned,unseen,dev=key2._build_key2_queries(
      run_root,ROOT/'key-shared-01/attempt-02/prefix-missing',ROOT/'seg-context-01/seg5-cache',
      ROOT/'seg-context-01/unseen-stride5',ROOT/'key-shared-01/attempt-27-production/vjepa/libero90_eval')
    if len(bank['length'])!=4516:raise AssertionError('KEY2-F fixed bank must contain 4516 units')
    # Pack all rows once; MLP models consume a single matched vector per row.
    prepared={name:_prepare_vector_sources(part) for name,part in
      (('bank',bi),('fit_bank',fit_bi),('train',train),('learned',learned),('unseen',unseen),('dev',dev))}
    bank_x={name:make_vector_features(prepared['bank'],name) for name in INPUTS}
    fit_x={name:make_vector_features(prepared['fit_bank'],name) for name in INPUTS}
    train_x={name:make_vector_features(prepared['train'],name) for name in INPUTS}
    learned_x={name:make_vector_features(prepared['learned'],name) for name in INPUTS}
    unseen_x={name:make_vector_features(prepared['unseen'],name) for name in INPUTS}
    dev_x={name:make_vector_features(prepared['dev'],name) for name in INPUTS}
    candidate_actions=key2._candidate_window_actions(np.arange(len(bank['length'])),bank)
    fit_candidate_actions=key2._candidate_window_actions(np.arange(len(fit_bank['length'])),fit_bank)
    # The fit candidate bank is a strict subset; map its exact row identity to full-bank errors/actions.
    fit_key_to_full={tuple(map(int,k)):i for i,k in enumerate(bi['key'])}
    fit_full_ids=np.asarray([fit_key_to_full[tuple(map(int,k))] for k in fit_bi['key']],np.int64)
    fit_candidate_actions=candidate_actions[fit_full_ids]
    train_for_fit={**train,'_candidate_task':fit_bank['task'],'_candidate_episode':fit_bank['episode']}
    # Both arrays are filtered from the same ordered unit bank, but do not
    # rely on that implementation detail when forming the positive candidate
    # index for each query.
    train_candidate_ids=exact_candidate_row_ids(train['key'],fit_bi['key'])
    # Limit only smoke fixtures; production always uses all rows and fixed grid.
    if smoke:
        maxtrain=32;maxcand=128;maxdev=24;maxeval=16
        fit_x={k:v[:maxcand] for k,v in fit_x.items()};fit_candidate_actions=fit_candidate_actions[:maxcand]
        train_x={k:v[:maxtrain] for k,v in train_x.items()}
        train_for_fit={k:(v[:maxtrain] if isinstance(v,np.ndarray) and len(v)==len(train['key']) else v)
                       for k,v in train_for_fit.items()}
        train_for_fit['_candidate_task']=fit_bank['task'][:maxcand]
        train_for_fit['_candidate_episode']=fit_bank['episode'][:maxcand]
        dev={k:(v[:maxdev] if isinstance(v,np.ndarray) and len(v)==len(dev['key']) else v) for k,v in dev.items()}
        dev_x={k:v[:maxdev] for k,v in dev_x.items()}
        learned={k:(v[:maxeval] if isinstance(v,np.ndarray) and len(v)==len(learned['key']) else v) for k,v in learned.items()}
        learned_x={k:v[:maxeval] for k,v in learned_x.items()}
        unseen={k:(v[:maxeval] if isinstance(v,np.ndarray) and len(v)==len(unseen['key']) else v) for k,v in unseen.items()}
        unseen_x={k:v[:maxeval] for k,v in unseen_x.items()}
        learning_rates=(3e-4,);epochs_grid=(1,);tau_grid=(.02,)
    results={'status':'running','seed':seed,'smoke':smoke,'grid':{'learning_rates':list(learning_rates),
        'epochs':list(epochs_grid),'teacher_tau':list(tau_grid)},'new_models':{},'existing_reselected':{},
        'feature_dims':{k:int(v.shape[-1]) for k,v in bank_x.items()}}
    # Run new six rankers. Each config is checkpointed and Dev-scored before selection.
    for name in NEW_INPUTS:
      cfgrows=[]
      for shared in (False,True):
       fusion='shared_mlp' if shared else 'two_tower_mlp'
       for lr in learning_rates:
        for ep in epochs_grid:
         for tau in tau_grid:
          tag=f'{name}-{fusion}-lr{lr:g}-ep{ep}-tau{tau:g}'
          sub=output/'tuning'/tag;sub.mkdir(parents=True,exist_ok=True)
          recpath=sub/'record.json';modelpath=sub/'model.pt'
          if recpath.exists() and modelpath.exists():
              saved=json.loads(recpath.read_text())
              if saved.get('input')!=name or saved.get('fusion')!=fusion or saved.get('learning_rate')!=float(lr) or saved.get('epochs')!=int(ep) or saved.get('teacher_temperature')!=float(tau):
                  raise ValueError(f'resume metadata mismatch in {sub}')
              cfgrows.append(saved)
              continue
          model,history=_train_one(name,shared,train_x[name],fit_x[name],train_for_fit,fit_candidate_actions,
              seed=seed,learning_rate=float(lr),epochs=int(ep),teacher_tau=float(tau),device=device,
              query_candidate_ids=train_candidate_ids,
              random_candidates=32 if smoke else 256)
          # Score all available Dev candidates. Production uses full bank; smoke uses its fixed prefix.
          dev_bank_x=bank_x[name][:maxcand] if smoke else bank_x[name]
          dev_scores=_score_vectors(model,dev_x[name],dev_bank_x,device=device)
          # Errors use the same exact candidate action windows as KEY2-F.
          dev_actions=candidate_actions[:len(dev_bank_x)]
          dev_err=key2._arm6_candidate_mse(dev['actions'],dev_actions)
          dev_metric=_rank_metrics(dev_scores,dev_err)
          torch.save(model.state_dict(),modelpath)
          row={'input':name,'fusion':fusion,'learning_rate':float(lr),'epochs':int(ep),
              'teacher_temperature':float(tau),'dev_top1_arm6_mse':dev_metric['top1_arm6_mse'],
              'dev_metrics':dev_metric,'train_history':history,'checkpoint':str(modelpath)}
          recpath.write_text(json.dumps(row,indent=2,allow_nan=False)+'\n');cfgrows.append(row)
          with (output/'progress.jsonl').open('a') as f:
              f.write(json.dumps({'complete':tag,'dev_top1':row['dev_top1_arm6_mse']})+'\n')
      variants={}
      for fusion in ('two_tower_mlp','shared_mlp'):
        frows=[x for x in cfgrows if x['fusion']==fusion]
        best=min(frows,key=lambda x:(x['dev_top1_arm6_mse'],x['learning_rate'],x['epochs'],x['teacher_temperature']))
        state=torch.load(best['checkpoint'],map_location=device,weights_only=True)
        model=PairMLPRanker(train_x[name].shape[1],shared=fusion=='shared_mlp').to(device);model.load_state_dict(state)
        full_bank_x=bank_x[name][:maxcand] if smoke else bank_x[name]
        ev_rows={}
        for split_name,split,x in (('learned',learned,learned_x[name]),('unseen',unseen,unseen_x[name])):
            scores=_score_vectors(model,x,full_bank_x,device=device)
            errors=key2._arm6_candidate_mse(split['actions'],candidate_actions[:len(full_bank_x)])
            ev_rows[split_name]=_split_eval(scores,errors,split)
        variants[fusion]={**best,'candidate_grid':len(frows),
            'evaluation':{k:{'metrics':v['metrics']} for k,v in ev_rows.items()},
            '_per_query':{k:v['top1_per_query'] for k,v in ev_rows.items()}}
        del model
      results['new_models'][name]={'variants':variants,'tuning':cfgrows}
    if not only_new:
      # Reselect existing checkpoints strictly on Dev top-1, then evaluate each once.
      selected=_existing_dev_selections()
      cfg=json.loads((KEY2_ROOT/'selected_configs.json').read_text())['selected']
      for mode,best in selected.items():
        model=key2._new_model(mode,token_count=5).to(device)
        model.load_state_dict(torch.load(best['checkpoint'],map_location=device,weights_only=True));model.eval()
        ev={}
        for split_name,split in (('learned',learned),('unseen',unseen)):
          scores,errors=key2._score_model(model,mode,bank,bi,split,device)
          ev[split_name]=_split_eval(scores,errors,split)
        # Preserve old blend-based choice for rank-order comparison.
        old=cfg.get(mode,{})
        original=json.loads((KEY2_ROOT/'results.json').read_text())['methods']
        results['existing_reselected'][mode]={'selected_by_dev_top1':best,
           'old_blend_selection':old.get('selected_blend'),
           'old_blend_eval_reference':{'learned':original['learned'][mode]['rank_all'],
              'learned_hard':original['learned'][mode]['rank_hard'],
              'unseen':original['unseen'][mode]['rank_all'],'unseen_hard':original['unseen'][mode]['rank_hard']},
           'evaluation':{k:{'metrics':v['metrics']} for k,v in ev.items()},
           '_per_query':{k:v['top1_per_query'] for k,v in ev.items()}}
        del model
    # Paired comparisons on fixed rows, cluster-resampled by episode.
    comparisons={}
    if not only_new:
      for name in NEW_INPUTS:
        e_mode=EXISTING_E[name];new=results['new_models'][name]['variants']
        comparisons[name]={}
        for split_name,split in (('learned',learned),('unseen',unseen)):
          ev=results['existing_reselected'][e_mode]['_per_query'][split_name]
          sep=new['two_tower_mlp']['_per_query'][split_name]
          tied=new['shared_mlp']['_per_query'][split_name]
          comparisons[name][split_name]={'E_minus_two_tower':_top1_boot(ev-sep,split,seed+11),
                                         'shared_minus_two_tower':_top1_boot(tied-sep,split,seed+12)}
    results['paired_episode_bootstrap_5000']=comparisons
    key2_results=json.loads((KEY2_ROOT/'results.json').read_text())['methods']
    results['key2_reference_rows']={}
    for split_name in ('learned','unseen'):
      fixed=key2_results[split_name]['fixed_baselines']
      results['key2_reference_rows'][split_name]={
        'separate':key2_results[split_name]['separate'],
        'random':{'all':fixed['random_rank'],'hard':fixed['random_rank_hard']},
        'oracle':{'all':fixed['oracle_rank'],'hard':fixed['oracle_rank_hard']},
        'hold_last':{'all':fixed['baselines']['hold_last'],'hard':fixed['baselines_hard']['hold_last']},
        'pi05':{'all':fixed['baselines']['pi05'],'hard':fixed['baselines_hard']['pi05']}}
    for group in ('new_models','existing_reselected'):
      for value in results[group].values():
        if group=='new_models':
          for variant in value.get('variants',{}).values():variant.pop('_per_query',None)
        else:value.pop('_per_query',None)
    results['status']='complete_smoke' if smoke else 'complete'
    (output/'results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    _write_report(output,results)
    return results


def _write_report(output: Path, r: dict[str,Any]) -> None:
    lines=['# KEYGRID: 入力 × 結合方法','',
      '## 未学習 LIBERO-90 Eval：top-1 腕 MSE（全体／難）','',
      '|入力|連結＋別MLP|連結＋共有MLP|潜在E|','|---|---:|---:|---:|']
    modes={'vlm':'e_vlm','history_state':'e_h','vlm_history_state':'e_vlm_h',
           'vjepa_history_state':'e_vj_h','vjepa_vlm_history_state':'e_vj_vlm_h'}
    names={'vlm':'VLM','history_state':'履歴＋状態','vlm_history_state':'VLM＋履歴＋状態',
           'vjepa_history_state':'V-JEPA＋履歴＋状態','vjepa_vlm_history_state':'V-JEPA＋VLM＋履歴＋状態'}
    for inp in INPUTS:
      row=r.get('new_models',{}).get(inp,{}).get('variants',{});e=r.get('existing_reselected',{}).get(modes[inp])
      if row:
        m=row['two_tower_mlp']['evaluation']['unseen']['metrics'];sep=f"{m['top1_arm6_mse']:.5f}/{m['top1_arm6_mse_hard']:.5f}"
        m=row['shared_mlp']['evaluation']['unseen']['metrics'];shared=f"{m['top1_arm6_mse']:.5f}/{m['top1_arm6_mse_hard']:.5f}"
      else:sep=shared='—'
      if e:
        m=e['evaluation']['unseen']['metrics'];ev=f"{m['top1_arm6_mse']:.5f}/{m['top1_arm6_mse_hard']:.5f}"
      else:ev='—'
      lines.append(f'|{names[inp]}|{sep}|{shared}|{ev}|')
    lines += ['', '## 未学習 Eval：順位指標','',
      '|入力|連結＋別MLP Spearman / R@1 / 回収率|連結＋共有MLP Spearman / R@1 / 回収率|潜在E Spearman / R@1 / 回収率|',
      '|---|---:|---:|---:|']
    for inp in INPUTS:
      row=r.get('new_models',{}).get(inp,{}).get('variants',{});e=r.get('existing_reselected',{}).get(modes[inp])
      def fmt(metrics):return f"{metrics['spearman_positive']:.4f} / {metrics['recall@1']:.4f} / {metrics['recovery']:.4f}"
      a=fmt(row['two_tower_mlp']['evaluation']['unseen']['metrics']) if row else '—'
      b=fmt(row['shared_mlp']['evaluation']['unseen']['metrics']) if row else '—'
      c=fmt(e['evaluation']['unseen']['metrics']) if e else '—'
      lines.append(f'|{names[inp]}|{a}|{b}|{c}|')
    lines += ['', '## 既存方式：Dev top-1再選択と従来blend選択','',
      '|方式|従来blend選択|top-1 Dev選択|Eval top-1 arm6 MSE (全体/難)|','|---|---|---:|---:|']
    for mode,x in r.get('existing_reselected',{}).items():
      old=x['old_blend_eval_reference']['unseen']['top1_arm6_mse']
      new=x['evaluation']['unseen']['metrics']['top1_arm6_mse']
      hard=x['evaluation']['unseen']['metrics']['top1_arm6_mse_hard']
      lines.append(f"|{mode}|{old:.5f}|{new:.5f}|{new:.5f}/{hard:.5f}|")
    lines += ['', '### 既存の別キー・固定対照','',
      '|split|方法|top-1 arm6 MSE (全体/難)|Spearman / R@1 / 回収率|','|---|---|---:|---:|']
    ref=r.get('key2_reference_rows',{})
    for split_name in ('learned','unseen'):
      d=ref.get(split_name,{})
      for method in ('separate','random','oracle','hold_last','pi05'):
        if method not in d:continue
        if method in ('separate',):
          allm=d[method]['rank_all'];hardm=d[method]['rank_hard'];a=allm['top1_arm6_mse'];h=hardm['top1_arm6_mse']
        elif method in ('random','oracle'):
          allm=d[method]['all'];hardm=d[method]['hard'];a=allm['top1_arm6_mse'];h=hardm['top1_arm6_mse']
        else:
          allm=d[method]['all'];hardm=d[method]['hard'];a=allm['arm6'];h=hardm['arm6']
        rank=(f"{allm.get('spearman_positive',0):.4f} / {allm.get('recall@1',0):.4f} / {allm.get('recovery',0):.4f}"
              if method in ('separate','random','oracle') else '—')
        lines.append(f'|{split_name}|{method}|{a:.5f}/{h:.5f}|{rank}|')
    lines += ['', '## 学習済み LIBERO-40 評価','',
      '|入力|連結＋別MLP|連結＋共有MLP|潜在E|','|---|---:|---:|---:|']
    for inp in INPUTS:
      row=r.get('new_models',{}).get(inp,{}).get('variants',{});e=r.get('existing_reselected',{}).get(modes[inp])
      if row:
        m=row['two_tower_mlp']['evaluation']['learned']['metrics'];sep=f"{m['top1_arm6_mse']:.5f}/{m['top1_arm6_mse_hard']:.5f}"
        m=row['shared_mlp']['evaluation']['learned']['metrics'];shared=f"{m['top1_arm6_mse']:.5f}/{m['top1_arm6_mse_hard']:.5f}"
      else:sep=shared='—'
      if e:
        m=e['evaluation']['learned']['metrics'];ev=f"{m['top1_arm6_mse']:.5f}/{m['top1_arm6_mse_hard']:.5f}"
      else:ev='—'
      lines.append(f'|{names[inp]}|{sep}|{shared}|{ev}|')
    lines += ['', '## 学習済み：順位指標','',
      '|入力|連結＋別MLP Spearman / R@1 / 回収率|連結＋共有MLP Spearman / R@1 / 回収率|潜在E Spearman / R@1 / 回収率|',
      '|---|---:|---:|---:|']
    for inp in INPUTS:
      row=r.get('new_models',{}).get(inp,{}).get('variants',{});e=r.get('existing_reselected',{}).get(modes[inp])
      def fmt(metrics):return f"{metrics['spearman_positive']:.4f} / {metrics['recall@1']:.4f} / {metrics['recovery']:.4f}"
      a=fmt(row['two_tower_mlp']['evaluation']['learned']['metrics']) if row else '—'
      b=fmt(row['shared_mlp']['evaluation']['learned']['metrics']) if row else '—'
      c=fmt(e['evaluation']['learned']['metrics']) if e else '—'
      lines.append(f'|{names[inp]}|{a}|{b}|{c}|')
    lines += ['', '## 対比較：episode bootstrap 95% CI','',
      '|入力|split|E − 別MLP (top1 arm6)|共有MLP − 別MLP (top1 arm6)|','|---|---|---:|---:|']
    for inp,by_split in r.get('paired_episode_bootstrap_5000',{}).items():
      for split_name,vals in by_split.items():
        def ci(x):return f"{x['mean']:.5f} [{x['ci95_low']:.5f}, {x['ci95_high']:.5f}]"
        lines.append(f"|{names[inp]}|{split_name}|{ci(vals['E_minus_two_tower'])}|{ci(vals['shared_minus_two_tower'])}|")
    lines += ['', '## 既存対照','',
      '既存 P0/P5/型のみ separate の Dev-top1 再選択値は `existing_reselected` に含まれる。random、直前保持、π0.5、oracle の再掲は KEY2-F 固定行の同一採点結果から取得する。', '', '## 指標定義','',
      '- Spearman：全4,516候補の予測順位と正解腕誤差順位の相関（符号は高得点＝小誤差にそろえる）。',
      '- Recall@1：スコア1位が全候補で誤差最小の単位と一致する割合。',
      '- 回収率：(random top-1 MSE − model top-1 MSE) / (random − oracle)。',
      '- top-1 arm MSE は6軸二乗和/(10×7)、難行は固定hard mask。', '',
      '設定は Dev top-1 arm MSE で選び、Eval は選択後に一度採点した。新規2塔／共有MLPは学習格子の各候補と履歴を `results.json` に記録。']
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,default=DEFAULT_OUT)
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--memory-fraction',type=float,default=.12)
    ap.add_argument('--only-new',action='store_true')
    a=ap.parse_args();os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false';torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','8')))
    if a.device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
    run_keygrid(a.output,device=a.device,smoke=a.smoke,only_new=a.only_new)

if __name__=='__main__':main()
