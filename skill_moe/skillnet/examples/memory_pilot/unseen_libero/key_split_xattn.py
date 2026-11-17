#!/usr/bin/env python3
"""KEY4: modality-split cross attention with an optional VLM residual gate.

The existing KEY2-F encoder is intentionally left untouched. This module uses
the same feature packing, contrastive target, and fixed rows as key_shared_latent.
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
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import key_shared_latent as key2

ROOT = Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero')
DEFAULT_OUT = ROOT / 'key-split-xattn-01'
VJEPA_CACHE = ROOT / 'key-shared-01/attempt-27-production/vjepa/libero90_eval'


class _SplitBlock(nn.Module):
    """Cross attention block matching KEY2-F's pre-norm, residual and FFN."""
    def __init__(self, dim: int, heads: int = 8, *, gated: bool = False):
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.gated = bool(gated)
        if gated:
            self.attn_gate = nn.Parameter(torch.zeros(()))
            self.ff_gate = nn.Parameter(torch.zeros(()))

    def forward(self, q: torch.Tensor, memory: torch.Tensor, padding_mask: torch.Tensor,
                *, return_weights: bool = False):
        kv = self.kv_norm(memory)
        a, w = self.attn(self.q_norm(q), kv, kv, key_padding_mask=padding_mask,
                         need_weights=return_weights, average_attn_weights=False)
        if self.gated:
            q = q + torch.tanh(self.attn_gate) * a
            q = q + torch.tanh(self.ff_gate) * self.ff(self.ff_norm(q))
        else:
            q = q + a
            q = q + self.ff(self.ff_norm(q))
        return q, w


class KeySplitXAttn(nn.Module):
    """Shared query/key encoder. A reads history/state then VLM separately."""
    def __init__(self, token_dim: int = 2048, context_dim: int = 256,
                 token_count: int = 5, history_steps: int = 10,
                 history_action_dim: int = 7, state_dim: int = 8,
                 group_count: int = 6, heads: int = 8, *, gated_vlm: bool = False):
        super().__init__()
        self.token_dim, self.context_dim, self.token_count = int(token_dim), int(context_dim), int(token_count)
        self.history_steps, self.history_action_dim, self.state_dim = int(history_steps), int(history_action_dim), int(state_dim)
        self.gated_vlm = bool(gated_vlm)
        self.vlm_projection = nn.Linear(token_dim, context_dim)
        self.group_embedding = nn.Embedding(group_count, context_dim)
        self.history_projection = nn.Linear(history_action_dim + 1, context_dim)
        self.history_position = nn.Parameter(torch.zeros(1, history_steps, context_dim))
        self.state_projection = nn.Sequential(nn.Linear(state_dim, context_dim), nn.GELU(), nn.Linear(context_dim, context_dim))
        self.state_type = nn.Parameter(torch.zeros(1, 1, context_dim))
        self.latent = nn.Parameter(torch.randn(1, token_count, context_dim) * .02)
        self.history_state_cross = _SplitBlock(context_dim, heads)
        self.vlm_cross = _SplitBlock(context_dim, heads, gated=gated_vlm)
        self.self_attention = nn.MultiheadAttention(context_dim, heads, batch_first=True)
        self.self_norm = nn.LayerNorm(context_dim)
        self.out_norm = nn.LayerNorm(context_dim)
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(key2.LOGIT_SCALE_INITIAL))))

    def forward(self, vlm_tokens: torch.Tensor, groups: torch.Tensor,
                history: torch.Tensor, history_mask: torch.Tensor, state: torch.Tensor,
                *, return_aux: bool = False):
        if vlm_tokens.ndim != 3 or vlm_tokens.shape[-1] != self.token_dim or groups.shape != vlm_tokens.shape[:2]:
            raise ValueError('VLM tokens/groups must be aligned [B,T,D]')
        b = vlm_tokens.shape[0]
        if history.shape != (b, self.history_steps, self.history_action_dim) or history_mask.shape != (b, self.history_steps):
            raise ValueError('history and validity mask have incompatible shapes')
        if state.shape != (b, self.state_dim):
            raise ValueError('state has incompatible shape')
        valid = groups >= 0
        safe = groups.clamp(0, self.group_embedding.num_embeddings - 1).long()
        vm = self.vlm_projection(vlm_tokens.float()) + self.group_embedding(safe)
        hi = torch.cat((history.float(), history_mask.float().unsqueeze(-1)), dim=-1)
        hm = self.history_projection(hi) + self.history_position
        sm = self.state_projection(state.float()).unsqueeze(1) + self.state_type
        hsm = torch.cat((hm, sm), dim=1)
        hmask = torch.cat((~history_mask.bool(), torch.zeros((b, 1), dtype=torch.bool, device=state.device)), dim=1)
        q = self.latent.expand(b, -1, -1)
        q, wh = self.history_state_cross(q, hsm, hmask, return_weights=return_aux)
        q, wv = self.vlm_cross(q, vm, ~valid, return_weights=return_aux)
        normalized = self.self_norm(q)
        q = q + self.self_attention(normalized, normalized, normalized, need_weights=False)[0]
        out = F.normalize(self.out_norm(q), dim=-1)
        if not return_aux:
            return out
        gates = None if not self.gated_vlm else {
            'attention': torch.tanh(self.vlm_cross.attn_gate),
            'ffn': torch.tanh(self.vlm_cross.ff_gate),
        }
        return out, {'history_state_attention': wh, 'vlm_attention': wv, 'gates': gates}


def _encode(model, part: dict[str, Any], device: str, *, batch: int = 16) -> np.ndarray:
    model.eval(); out=[]
    with torch.no_grad():
        for s in range(0, len(part['key']), batch):
            ix=slice(s, s+batch)
            def t(k, dtype): return torch.as_tensor(part[k][ix], device=device, dtype=dtype)
            out.append(model(t('tokens',torch.float32),t('groups',torch.long),t('history',torch.float32),
                t('history_mask',torch.bool),t('state',torch.float32)).cpu().numpy())
    return np.concatenate(out)


def _score(model, bank, bank_input, split, device):
    keys=_encode(model,bank_input,device)
    queries=_encode(model,split,device)
    scores=np.einsum('qkd,nkd->qnk',queries,keys,optimize=True).mean(-1)
    scale=float(model.logit_scale.detach().exp().clamp(max=key2.LOGIT_SCALE_MAX).cpu())
    scores*=scale
    actions=key2._candidate_window_actions(np.arange(len(bank['length'])),bank)
    errors=key2._arm6_candidate_mse(split['actions'],actions)
    return scores, errors


def _top1(scores, errors, split):
    ix=np.argmax(scores,axis=1);loss=errors[np.arange(len(ix)),ix];hard=np.asarray(split['hard'],bool)
    return {'rows':len(ix),'hard_rows':int(hard.sum()),'top1_arm6_mse':float(loss.mean()),
            'top1_arm6_mse_hard':float(loss[hard].mean()) if hard.any() else None,
            'ranking':key2.candidate_rank_metrics(scores,errors,tau=.01,temperature=1.)}


def _train_one(model_name, fit_bank, fit_input, bank, bank_input, train, dev, learned, unseen, *, seed, lr, tau, device, output):
    torch.manual_seed(seed); np.random.seed(seed)
    model=KeySplitXAttn(gated_vlm=(model_name=='B')).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    rng=np.random.default_rng(seed+101);qids=np.arange(len(train['key']))
    actions=key2._candidate_window_actions(np.arange(len(fit_bank['length'])),fit_bank)
    history=[];checkpoints={}
    for ep in range(16):
        model.train();rng.shuffle(qids);losses=[]
        for step_idx,st in enumerate(range(0,len(qids),8),1):
            qi=qids[st:st+8]
            if not len(qi):continue
            random_ids=rng.choice(len(fit_bank['length']),size=min(256,len(fit_bank['length'])),replace=False)
            cids,allowed=key2.contrastive_candidate_set(qi,random_ids,train['task'][qi],train['episode'][qi],fit_bank['task'],fit_bank['episode'])
            qpart=key2._pack_query_observation(train,qi)
            qh,qm=key2._history_corrupt(qpart['history'],qpart['history_mask'],rng)
            qpart['history'],qpart['history_mask']=qh,qm
            kpart=key2._pack_query_observation(fit_input,cids)
            def tensor(part,k,dtype):return torch.as_tensor(part[k],device=device,dtype=dtype)
            def run(part):return model(tensor(part,'tokens',torch.float32),tensor(part,'groups',torch.long),
                tensor(part,'history',torch.float32),tensor(part,'history_mask',torch.bool),tensor(part,'state',torch.float32))
            qr,kr=run(qpart),run(kpart)
            scores=(qr[:,None]*kr[None]).sum(-1).mean(-1)*model.logit_scale.detach().exp().clamp(max=key2.LOGIT_SCALE_MAX)
            err=torch.as_tensor(key2._arm6_candidate_mse(train['actions'][qi],actions[cids]),device=device)
            loss=key2.contrastive_teacher_loss(scores,err,torch.as_tensor(allowed,device=device),objective='soft',teacher_temperature=tau)
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            key2._clamp_logit_scale_(model);losses.append(float(loss.detach()))
            if step_idx % 100 == 0:
                with (output/'progress.log').open('a') as f:
                    f.write(json.dumps({'model':model_name,'epoch':ep+1,'step':step_idx,'steps_per_epoch':int(np.ceil(len(qids)/8)),'loss':float(np.mean(losses))})+'\n')
        history.append({'epoch':ep+1,'loss':float(np.mean(losses)),'steps':len(losses)})
        with (output/'progress.log').open('a') as f:
            f.write(json.dumps({'model':model_name,'epoch':ep+1,'epoch_complete':True,'train_loss':history[-1]['loss']})+'\n')
        if ep+1 in (4,8,16):
            ds,de=_score(model,bank,bank_input,dev,device);metric=_top1(ds,de,dev)
            path=output/f'{model_name.lower()}_epoch{ep+1}.pt';torch.save(model.state_dict(),path)
            checkpoints[ep+1]={'dev':metric,'checkpoint':str(path),'gate_values':_gates(model)}
            with (output/'progress.log').open('a') as f:
                f.write(json.dumps({'model':model_name,'epoch':ep+1,'train_loss':history[-1]['loss'],'dev_top1':metric['top1_arm6_mse']})+'\n')
    selected=min(checkpoints,key=lambda e:(checkpoints[e]['dev']['top1_arm6_mse'],e))
    model.load_state_dict(torch.load(checkpoints[selected]['checkpoint'],map_location=device,weights_only=True));model.eval()
    out={}
    for n,split in (('learned',learned),('unseen',unseen)):
        scores,errors=_score(model,bank,bank_input,split,device);out[n]=_top1(scores,errors,split)
    return {'selected_epoch':selected,'dev_checkpoints':checkpoints,'history':history,'eval':out,
            'gate_values':_gates(model),'checkpoint':checkpoints[selected]['checkpoint']}


def _gates(model):
    if not model.gated_vlm:return None
    return {'attention':float(torch.tanh(model.vlm_cross.attn_gate).detach().cpu()),
            'ffn':float(torch.tanh(model.vlm_cross.ff_gate).detach().cpu())}


def run(output: Path, *, device='cuda:0', seed=1, smoke=False):
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','8')))
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError(f'refusing to overwrite {output}')
    output.mkdir(parents=True,exist_ok=True)
    if device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get('TORCH_MEM_FRACTION','.20')))
    with (output/'progress.log').open('a') as f:
        f.write(json.dumps({'stage':'loading_fixed_KEY2_rows'})+'\n')
    bank,bank_input,fit_bank,fit_input,train,learned,unseen,dev=key2._build_key2_queries(
        Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/run-01'),
        ROOT/'key-shared-01/attempt-02/prefix-missing',ROOT/'seg-context-01/seg5-cache',
        ROOT/'seg-context-01/unseen-stride5',VJEPA_CACHE)
    if len(bank['length'])!=4516 or len(train['key'])==0 or len(learned['key'])!=621 or len(unseen['key'])!=619:
        raise AssertionError('KEY2-F fixed row counts do not match')
    if smoke:
        # Test-mode only; production never takes this branch.
        bank={k:(v[:96] if isinstance(v,np.ndarray) and len(v)==4516 else v) for k,v in bank.items()}
        bank_input={k:(v[:96] if isinstance(v,np.ndarray) and len(v)==4516 else v) for k,v in bank_input.items()}
        train={k:(v[:32] if isinstance(v,np.ndarray) and len(v)==len(train['key']) else v) for k,v in train.items()}
        fit_bank=bank;fit_input=bank_input;dev={k:(v[:16] if isinstance(v,np.ndarray) and len(v)==217 else v) for k,v in dev.items()}
        learned={k:(v[:8] if isinstance(v,np.ndarray) and len(v)==621 else v) for k,v in learned.items()}
        unseen={k:(v[:8] if isinstance(v,np.ndarray) and len(v)==619 else v) for k,v in unseen.items()}
    key2._LAST_SPLITS={'learned':learned,'unseen':unseen}
    lr,tau=.0001,.002
    rows={name:_train_one(name,fit_bank,fit_input,bank,bank_input,train,dev,learned,unseen,
        seed=seed,lr=lr,tau=tau,device=device,output=output) for name in ('A','B')}
    grid_path=ROOT/'key-grid-01/production-04/results.json'
    grid=json.loads(grid_path.read_text())
    kr=grid['key2_reference_rows']
    refs={}
    for key,name in (('e_vlm_h','E-VLM+H'),('e_h','E-H'),('p5','P5')):
        m=grid['existing_reselected'][key]['evaluation']
        unseen_m=m['unseen']['metrics'];learned_m=m['learned']['metrics']
        refs[name]={'unseen_all':float(unseen_m['top1_arm6_mse']),'unseen_hard':float(unseen_m['top1_arm6_mse_hard']),
                    'learned_all':float(learned_m['top1_arm6_mse']),'learned_hard':float(learned_m['top1_arm6_mse_hard'])}
    hs=grid['new_models']['history_state']['variants']['two_tower_mlp']['evaluation']
    refs['履歴＋状態 MLP']={'unseen_all':float(hs['unseen']['metrics']['top1_arm6_mse']),
        'unseen_hard':float(hs['unseen']['metrics']['top1_arm6_mse_hard']),
        'learned_all':float(hs['learned']['metrics']['top1_arm6_mse']),
        'learned_hard':float(hs['learned']['metrics']['top1_arm6_mse_hard'])}
    refs['oracle']={'unseen_all':kr['unseen']['oracle']['all']['top1_arm6_mse'],
        'unseen_hard':kr['unseen']['oracle']['hard']['top1_arm6_mse'],
        'learned_all':kr['learned']['oracle']['all']['top1_arm6_mse'],
        'learned_hard':kr['learned']['oracle']['hard']['top1_arm6_mse']}
    for key,label in (('hold_last','直前保持'),('pi05','π0.5')):
        refs[label]={'unseen_all':kr['unseen'][key]['all']['arm6'],'unseen_hard':kr['unseen'][key]['hard']['arm6'],
            'learned_all':kr['learned'][key]['all']['arm6'],'learned_hard':kr['learned'][key]['hard']['arm6']}
    results={'status':'complete','stage':'stage_1_interim','seed':seed,'smoke':smoke,
        'configuration':{'learning_rate':lr,'teacher_tau':tau,'epochs_tested':[4,8,16],
            'train_rows':len(train['key']),'retrieval_units':len(bank['length']),'learned_eval_rows':len(learned['key']),
            'unseen_eval_rows':len(unseen['key']),'dev_rows':len(dev['key'])},'models':rows,
        'references_from_keygrid_production04':refs,
        'scope_note':'Interim stage only. Additional seeds, paired bootstrap, attention breakdown, and VLM swap test are deferred.'}
    (output/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    _write_report(output/'report.md',results)
    return results


def _write_report(path, r):
    lines=['# KEY4：分離クロスアテンションと VLM ゲート（第1段・中間報告）','',
      '## 主表','', '| 行 | 未学習全体 top-1 arm MSE | 未学習難 top-1 arm MSE | 学習済み全体 top-1 arm MSE | 学習済み難 top-1 arm MSE | Dev 選択 epoch | tanh(g) attention / FFN |','|---|---:|---:|---:|---:|---:|---:|']
    for name,d in r['models'].items():
        u,l=d['eval']['unseen'],d['eval']['learned'];g=d['gate_values']
        gate_text='—' if g is None else f"{g['attention']:.4f} / {g['ffn']:.4f}"
        fmt=lambda x:'—' if x is None else f'{x:.5f}'
        lines.append(f"| {name} | {fmt(u['top1_arm6_mse'])} | {fmt(u['top1_arm6_mse_hard'])} | {fmt(l['top1_arm6_mse'])} | {fmt(l['top1_arm6_mse_hard'])} | {d['selected_epoch']} | {gate_text} |")
    for k,v in r['references_from_keygrid_production04'].items():
        lines.append(f"| {k} | {v['unseen_all']:.5f} | {v['unseen_hard']:.5f} | {v['learned_all']:.5f} | {v['learned_hard']:.5f} | KEYGRID 参照 | — |")
    lines += ['', '## 設定と進捗','',f"- KEYGRID E-VLM+H の選択設定を固定: lr={r['configuration']['learning_rate']}, teacher τ={r['configuration']['teacher_tau']}。Dev top-1 腕 MSE で epoch 4/8/16 を選択。",
      f"- 学習行={r['configuration']['train_rows']}、キー単位={r['configuration']['retrieval_units']}、Dev={r['configuration']['dev_rows']}、Eval: 学習済み={r['configuration']['learned_eval_rows']}／未学習={r['configuration']['unseen_eval_rows']}。学習は LIBERO-40 メモリ側のみ。",
      '- 目的関数は KEYGRID と同じ soft InfoNCE、batch 8 + 256 random candidates、same-episode exclusion、history corruption、learned logit scale (initial 1/0.07, max 100)。',
      '- 第1段の暫定結果。seed は1本。第2段（追加 seed、対比較 bootstrap、注意配分、VLM token swap）は未実施。',
      '- 比較参照 E-VLM+H .127、E-H .131、履歴＋状態 MLP .101、P5 .110 は KEYGRID production-04 の報告値を再掲。oracle／直前保持／π0.5 も同レポート参照値として別途残す。',
      '']
    path.write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DEFAULT_OUT);p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=1);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();run(a.output,device=a.device,seed=a.seed,smoke=a.smoke)

if __name__=='__main__':main()
