"""Summarize paired RM-RERANK closed-loop runs and the fixed controls."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
ROOT=Path('/work/.skillnet-state/outputs/robocasa-memory');OUT=ROOT/'rerank-01/closedloop';REPORT=OUT/'report.md'
TASKS=['PickPlaceCounterToCabinet','PickPlaceSinkToCounter','TurnOnSinkFaucet']
CANON={'PickPlaceCounterToCabinet':0,'PickPlaceSinkToCounter':1,'TurnOnSinkFaucet':4}
def load(path):
 p=Path(path)
 if not p.exists():return None
 d=json.loads(p.read_text());records={}
 for r in d.get('records',[]):
  if r.get('task') in TASKS and int(r.get('trial',-1))<25:
   records[(r['task'],int(r['trial']))]=r
 return d,records
def wilson(k,n):
 if n<=0:return [float('nan'),float('nan')]
 z=1.959963984540054;p=k/n;den=1+z*z/n;c=(p+z*z/(2*n))/den;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return [c-h,c+h]
def outcome(d):
 return {key: bool(r.get('success',False)) for key,r in d.items() if r.get('status')=='complete'}
def pair(a,b):
 aa=outcome(a);bb=outcome(b);keys=sorted(aa.keys()&bb.keys());both=ao=bo=neither=0
 for k in keys:
  x,y=aa[k],bb[k]
  if x and y:both+=1
  elif x:ao+=1
  elif y:bo+=1
  else:neither+=1
 n=ao+bo
 p=1. if n==0 else min(1.,2.*sum(math.comb(n,j) for j in range(min(ao,bo)+1))/(2**n))
 return {'n':len(keys),'left_only':ao,'right_only':bo,'both_success':both,'both_failure':neither,'mcnemar_exact_p':p}
def success(d,task=None):
 rs=[r for (t,_),r in d.items() if (task is None or t==task) and r.get('status')=='complete'];k=sum(bool(r.get('success')) for r in rs)
 return k,len(rs),wilson(k,len(rs))
def format_rate(d,task=None):
 k,n,ci=success(d,task);return f'{k}/{n} ({k/n:.1%}; {ci[0]:.1%}–{ci[1]:.1%})' if n else 'n/a'
def main():
 if REPORT.exists():raise FileExistsError(f'refusing overwrite: {REPORT}')
 runs={}
 for cond in ('rerank','top1'):
  x=load(OUT/cond/'eval/results.json')
  if x:runs[cond]=x[1]
 controls={'v6-nosubtask':load(ROOT/'baseline-01/v6-nosubtask/eval/results.json'),
           'expert-02 eval300':load(ROOT/'expert-02/eval300/eval/results.json'),
           'expert-03 t0.4':load(ROOT/'expert-03/closedloop/e3_t04/eval/results.json')}
 controls={k:v[1] for k,v in controls.items() if v}
 lines=['実装: 完了','','# RM-RERANK-CL：閉ループ比較','','同じ canonical task index・trial seed（7 + task_index×50 + trial）、pretrain、replan=5。全ての主条件は seed/task/trial で対応付けた。Wilson 95%区間は完了試行を分母とする。McNemar は一致した完了ペアのみを用いる。','', '| 条件 | 合計成功率 (Wilson 95%) | CounterToCabinet | SinkToCounter | TurnOnSinkFaucet | error/incomplete |','|---|---:|---:|---:|---:|---:|']
 labels={'rerank':'K=50 reranker + Expert03 t0=.2','top1':'P5 top-1 + Expert03 t0=.2','v6-nosubtask':'v6-nosubtask π0.5','expert-02 eval300':'Expert02 eval300 (旧Expert t0=.2)','expert-03 t0.4':'Expert03 closedloop t0=.4'}
 for name,d in [*runs.items(),*controls.items()]:
  overall=format_rate(d);cells=[format_rate(d,t) for t in TASKS];expected=75 if name in ('rerank','top1') else 25*len(TASKS)
  errors=expected-len(outcome(d));lines.append(f'| {labels[name]} | {overall} | '+ ' | '.join(cells)+f' | {errors} |')
 pairs_to_show=[]
 if 'rerank' in runs and 'top1' in runs:pairs_to_show.append(('K50 rerank − current top-1',runs['rerank'],runs['top1']))
 for cond in ('rerank','top1'):
  if cond not in runs:continue
  for c in ('v6-nosubtask','expert-02 eval300','expert-03 t0.4'):
   if c in controls:pairs_to_show.append((f"{labels[cond]} − {labels[c]}",runs[cond],controls[c]))
 lines+=['','## 対の内訳と exact McNemar','', '| 比較（左−右） | 完了ペア | 左だけ成功 | 右だけ成功 | 両方成功 | 両方失敗 | exact p |','|---|---:|---:|---:|---:|---:|---:|']
 for name,a,b in pairs_to_show:
  p=pair(a,b);lines.append(f"| {name} | {p['n']} | {p['left_only']} | {p['right_only']} | {p['both_success']} | {p['both_failure']} | {p['mcnemar_exact_p']:.4f} |")
 lines+=['','## 実行条件・記録','',
 '- サーバーは FIXGR 完了後の `serve_memonly_policy.py` のコピー。`--draft-unit-gripper --grip-after-bridge --scale-position-only` を有効にした。初単位の先頭5だけ Context grip、後続単位は保存 grip、倍率は位置3軸のみ、bridge後も先頭5 grip をContext予測で上書き。',
 '- K50 reranker は各50-step連結境界で現行キー top-50 を再採点。bridge は expert-03 `expert_003000.msgpack`、t0=.2、10積分。top-1 条件は同じコピーサーバーで reranker checkpoint を省き、他設定を揃えた。',
 '- 試行ログは各 condition の `eval/results.json`、`queries.jsonl`、`server.log`、`client.log`。query log は選択単位、キー順位スコア、reranker rank/logit、FIXGR flags を記録。',
 '- GPU は起動時に空き確認後、`XLA_PYTHON_CLIENT_PREALLOCATE=false`、`XLA_PYTHON_CLIENT_MEM_FRACTION=.25`。',
 '- `chain_rmLITE.status` の FIXGR exit marker を確認してから開始。']
 if 'rerank' not in runs:lines.insert(2,'主条件の結果ファイルがなく、実行が完了していない。')
 if 'top1' not in runs:lines.insert(2,'top-1 対照は未完了または未実行（時間優先順位は reranker 主条件）。')
 REPORT.write_text('\n'.join(lines)+'\n')
 print(REPORT)
if __name__=='__main__':main()
