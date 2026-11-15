import json
from pathlib import Path
import numpy as np
R=Path('/work/.skillnet-state/outputs/robocasa-memory');O=R/'rerank-01';report=O/'report.md'
if report.exists():raise FileExistsError('refuse overwrite report')
a=np.load(O/'heldout_rerank.npz');kd=np.load(R/'expert-02/key-diag/keydiag_candidates.npz');b=np.load(O/'bridge_outputs.npz');oc=np.load(O/'oracle_chain.npz');q=np.load(R/'memory-01/policy-order-v2/queries/U_heldout.npz')
rows=np.asarray(a['row_ids'],int);task=np.asarray(q['task'])[rows];ep=np.asarray(q['episode'])[rows];target=np.asarray(a['target'],np.float32);n=len(rows)

def err(x,h=None):
 x=np.asarray(x);y=target
 if h is not None:x=x[:,:h];y=y[:,:h]
 return np.mean((x[:,:,:7]-y[:,:,:7])**2,axis=(1,2))
def boot(v,seed=17,B=5000):
 pairs=np.unique(np.stack((task,ep),axis=1),axis=0);grp=[np.flatnonzero((task==t)&(ep==e)) for t,e in pairs];rng=np.random.default_rng(seed);out=np.empty(B)
 for j in range(B):
  ix=np.concatenate([grp[k] for k in rng.integers(0,len(grp),len(grp))]);out[j]=np.mean(v[ix])
 return np.quantile(out,[.025,.975])
def fmt(v):
 ci=boot(v);return f'{np.mean(v):.3f} [{ci[0]:.3f}, {ci[1]:.3f}]'
vals={
'key_candidate':np.asarray(a['top_candidate_mse'])[:,0],
'r20_candidate':np.asarray(a['top_candidate_mse'])[np.arange(n),np.asarray(a['rerank_logits_k20']).argmax(1)],
'r50_candidate':np.asarray(a['top_candidate_mse'])[np.arange(n),np.asarray(a['rerank_logits_k50']).argmax(1)],
'key_chain_f':np.asarray(a['chain_first5_key_top1']),'key_chain_h':np.asarray(a['chain_horizon50_key_top1']),
'r20_chain_f':np.asarray(a['chain_first5_rerank_k20']),'r20_chain_h':np.asarray(a['chain_horizon50_rerank_k20']),
'r50_chain_f':np.asarray(a['chain_first5_rerank_k50']),'r50_chain_h':np.asarray(a['chain_horizon50_rerank_k50']),
'random_f':err(kd['random_draft'],5),'random_h':err(kd['random_draft']),
'oracle_f':np.asarray(kd['top20_best_action_mse']),'oracle_chain_f':np.asarray(oc['first5_mse']),'oracle_chain_h':np.asarray(oc['horizon50_mse']),
'pi_f':np.full(n,.4635819),'pi_h':np.full(n,1.0395),
}
for key in ('draft_k20','draft_k50','pred_pretrained_k20','pred_expert03_k20','pred_pretrained_k50','pred_expert03_k50'):
 v=np.asarray(b[key]);vals[key+'_f']=err(v,5);vals[key+'_h']=err(v)
lines=['実装: 完了','','# RM-RERANK：実行行動の近さで上位候補を並べ替え','',
'同一の保留210窓（6タスク×35）。候補は現行 Context＋履歴キーの top-K、自エピソードを除外。教師ラベルは倍率ヘッドと文脈グリッパーを適用した先頭5ステップの正規化active7 MSE。入力は VLM prefix＋状態＋履歴、候補の倍率適用後action列・長さ・グリッパー・現行キー得点。候補タスクIDは入力していない。CI はエピソード単位 bootstrap 5,000回。','',
'## 順位と連鎖','', '| 条件 | 選択した候補 first5 MSE (95% CI) | 実際の連鎖 first5 MSE (95% CI) | 連鎖50-step MSE (95% CI) |','|---|---:|---:|---:|']
for name,key in [('現行キー top-1','key_candidate'),('rerank K=20','r20_candidate'),('rerank K=50','r50_candidate')]:lines.append(f'| {name} | {fmt(vals[key])} | — | — |')
for name,f,h in [('現行キー top-1 chain','key_chain_f','key_chain_h'),('rerank K=20 chain','r20_chain_f','r20_chain_h'),('rerank K=50 chain','r50_chain_f','r50_chain_h'),('ランダム連鎖','random_f','random_h'),('Top-20行動オラクル：各境界で再選択','oracle_chain_f','oracle_chain_h'),('π0.5 通常','pi_f','pi_h')]:lines.append(f'| {name} | — | {fmt(vals[f])} | {fmt(vals[h])} |')
lines.append(f"| Top-20行動オラクル：初回候補 | {fmt(vals['oracle_f'])} | — | — |")
lines+=['','## 再順位後下書きの bridge','', '| 条件 | first5 MSE (95% CI) | 50-step MSE (95% CI) |','|---|---:|---:|']
for model in ('pretrained','expert03'):
 for k in (20,50):lines.append(f"| K={k} draft → {model} bridge, t0=.2, 10 steps | {fmt(vals[f'pred_{model}_k{k}_f'])} | {fmt(vals[f'pred_{model}_k{k}_h'])} |")
lines+=['','## 対応差','', '| 差（左−右） | 平均差 | episode bootstrap 95% CI |','|---|---:|---:|']
for label,x,y in [('K20 chain first5 − key chain',vals['r20_chain_f'],vals['key_chain_f']),('K50 chain first5 − key chain',vals['r50_chain_f'],vals['key_chain_f']),('K20 chain 50 − key chain',vals['r20_chain_h'],vals['key_chain_h']),('K50 chain 50 − key chain',vals['r50_chain_h'],vals['key_chain_h']),('K50 candidate first5 − π0.5',vals['r50_candidate'],vals['pi_f']),('Expert03 K50 bridge first5 − π0.5',vals['pred_expert03_k50_f'],vals['pi_f'])]:
 d=x-y;ci=boot(d);lines.append(f'| {label} | {d.mean():+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] |')
lines+=['','## タスク別（MSE first5 / 50-step）','', '| タスク | key chain | rerank K20 | rerank K50 | Expert03 bridge K50 |','|---|---:|---:|---:|---:|']
for t,name in enumerate(['PickPlaceCounterToCabinet','PickPlaceSinkToCounter','OpenDrawer','CloseFridge','TurnOnSinkFaucet','TurnOffStove']):
 ix=np.flatnonzero(task==t)
 if len(ix):lines.append(f"| {name} (n={len(ix)}) | {vals['key_chain_f'][ix].mean():.3f} / {vals['key_chain_h'][ix].mean():.3f} | {vals['r20_chain_f'][ix].mean():.3f} / {vals['r20_chain_h'][ix].mean():.3f} | {vals['r50_chain_f'][ix].mean():.3f} / {vals['r50_chain_h'][ix].mean():.3f} | {vals['pred_expert03_k50_f'][ix].mean():.3f} / {vals['pred_expert03_k50_h'][ix].mean():.3f} |")
rj=json.loads((O/'results.json').read_text());lines+=['','## 学習・制約','',f"- Reranker validation はエピソード単位、タスク内層化 80/20。K20 は temp={rj['selected_hyperparameters']['20']['temp']}・epoch={rj['selected_hyperparameters']['20']['epochs']}、K50 は temp={rj['selected_hyperparameters']['50']['temp']}・epoch={rj['selected_hyperparameters']['50']['epochs']}。選択指標は validation の選択候補MSE。listwise soft target `exp(−MSE/temp)`。履歴破損は zero .25 / 別行 .25 / 並進反転 .15 / 正しい .35。",'- P5キー・倍率・グリッパー各モデルは固定し、全メモリ窓で生成した候補を使った。従って再順位器の検証はepisode分離されるが、上流モデルの候補生成はcross-fitではない。','- K50 の選択候補 first5=.636、連鎖 first5=.602で、π0.5通常=.464には届かない。連鎖50-stepも2.192でπ0.5の1.040を上回る。指定の「良ければ」基準を満たさないため、serve_memonly_policy.py のコピーやサーバー起動確認は行わなかった。','- Bridgeは `expert-03/train/expert/expert_003000.msgpack` と学習前π0.5、同一ノイズ `fold_in(key(4101), row_id)`、t0=.2、10積分。再順位下書きは各単位の保存長で50-stepへ連結し、境界ごとに生成履歴で検索を更新。','- XLA の process fraction は .20。全GPUの観測使用量は他ジョブ込みで約76–102 GiB。']
report.write_text('\n'.join(lines)+'\n');print(report)
