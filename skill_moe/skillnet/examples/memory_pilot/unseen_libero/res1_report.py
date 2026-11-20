"""Assemble RES1 offline and closed-loop production tables with paired tests."""
from __future__ import annotations
import json, math, os, time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
import kq2cl_common as C
OUT = Path(os.environ.get("RES1_RUN_ROOT", C.ROOT / "res1-01" / "production"))
KQ = C.OUT
RES1_ROOT = C.ROOT / "res1-01"
JST = timezone(timedelta(hours=9))


def read_trials(path):
    rows = {}
    if not path.exists(): return rows
    for line in path.read_text().splitlines():
        try:
            r=json.loads(line); key=(int(r['task_id']),int(r['trial_index']))
            rows[key]=r
        except (ValueError, KeyError, json.JSONDecodeError): pass
    return {k:v for k,v in rows.items() if v.get('status') == 'complete'}


def exact_mcnemar(baseline, res1):
    keys=sorted(set(baseline)&set(res1))
    if len(keys) != 100:
        raise RuntimeError(f'paired closed-loop comparison requires 100 matching trials, found {len(keys)}')
    b_only=sum(bool(res1[k]['success']) and not bool(baseline[k]['success']) for k in keys)
    a_only=sum(bool(baseline[k]['success']) and not bool(res1[k]['success']) for k in keys)
    n=b_only+a_only
    p=1. if not n else min(1.,2.*sum(math.comb(n,i) for i in range(min(a_only,b_only)+1))/2**n)
    return {'paired_n':len(keys),'res1_only':b_only,'baseline_only':a_only,'p_exact':p}


def baseline_rows(suite, name):
    p=KQ/'closed-loop'/suite/name/'trials.jsonl'
    if name=='pi05' and suite=='libero90':
        p=C.ROOT/'baseline-01'/'production'/'rollouts.jsonl'
    rows=read_trials(p)
    if name=='pi05' and suite=='libero10':
        old=read_trials(C.ROOT/'closed-loop-memory-01'/'libero10'/'normal'/'rollouts.jsonl')
        rows={**old,**rows}
    return rows


def compile_report():
    offline_path=OUT/'offline'/'results.json'
    offline=json.loads(offline_path.read_text()) if offline_path.exists() else None
    train={}
    for p in (OUT/'train').glob('*/train.json'):
        m=json.loads(p.read_text()); train[m['name']]=m
    selection=json.loads((OUT/'selection.json').read_text())
    conditions=('flow_iii','flow_iii_x','flow_i')
    summaries={}; per_task={}
    for suite in ('libero90','libero10'):
        summaries[suite]={}; per_task[suite]={}
        for cond in conditions:
            rr=read_trials(OUT/'closed-loop'/suite/cond/'trials.jsonl')
            if len(rr)!=100:
                raise RuntimeError(f'{suite}/{cond}: expected 100 complete trials, got {len(rr)}')
            s=sum(bool(r['success']) for r in rr.values())
            summaries[suite][cond]={'n':len(rr),'successes':s,'rate':s/len(rr)}
            per_task[suite][cond]={}
            for t in sorted({k[0] for k in rr}):
                rows=[r for (tt,_),r in rr.items() if tt==t]
                per_task[suite][cond][str(t)]={'successes':sum(bool(r['success']) for r in rows),'n':len(rows)}
        for base in ('A','B','Aconst','pi05'):
            br=baseline_rows(suite,base)
            summaries[suite][base]={'n':len(br),'successes':sum(bool(r['success']) for r in br.values()),
                                    'rate':float(np.mean([r['success'] for r in br.values()])) if br else None}
        paired={}
        for cond in conditions:
            rr=read_trials(OUT/'closed-loop'/suite/cond/'trials.jsonl')
            for base in ('A','B','Aconst','pi05'):
                br=baseline_rows(suite,base)
                paired[f'{cond}_vs_{base}']=exact_mcnemar(br,rr)
        summaries[suite]['paired_mcnemar']=paired
    started=datetime.fromtimestamp((OUT/'progress.log').stat().st_mtime,timezone.utc).astimezone(JST).isoformat()
    result={'selection':selection,'training_runs':train,'offline':offline,'closed_loop':summaries,
            'task_success':per_task,'report_generated_jst':datetime.now(JST).isoformat()}
    (RES1_ROOT/'results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,default=float)+'\n')
    lines=['# RES1：取り出したスキルへの残差補正','',
           '## 結論と主表','',
           '閉ループ結果は各条件・各スイートの全100試行を完了した値。対比較は task_id × trial_index を一致させた正確 McNemar 検定。RES1 のチェックポイントと設定は Dev v2 のみで選択した。','',
           '| スイート | 条件 | 成功 | A | B | A-const | π0.5 |','|---|---|---:|---:|---:|---:|---:|']
    for suite in ('libero90','libero10'):
        for cond in conditions:
            own=summaries[suite][cond]['successes']
            cells=[]
            for base in ('A','B','Aconst','pi05'):
                other=summaries[suite][base]['successes']
                m=summaries[suite]['paired_mcnemar'][f'{cond}_vs_{base}']
                cells.append(f'{other}/100; {m["res1_only"]}/{m["baseline_only"]}, p={m["p_exact"]:.3g}')
            lines.append(f'| {suite} | {cond} | {own}/100 | '+' | '.join(cells)+' |')
    lines += ['', 'McNemar 列の対内訳は RES1 だけ成功／比較条件だけ成功。', '',
              '## Dev v2 選択と学習', '',
              '| 条件 | 選択 run | 更新 | Dev v2 腕6 MSE | 残差比 |', '|---|---|---:|---:|---:|']
    for name in ('flow_iii','flow_iii_x','flow_i','reg_iii'):
        s=selection['selected'][name]; meta=train[s['run']]
        hist=meta.get('selected',{})
        ratio=hist.get('residual_ratio')
        lines.append(f'| {name} | {s["run"]} | {s["checkpoint_step"]} | {s["dev2_arm6_mse"]:.6f} | {ratio if ratio is not None else "see offline"} |')
    lines += ['', '## オフライン評価', '', '腕6軸 MSE は KQ2-CL と同じ正規化尺度（6軸二乗和を10×7で割る）。Decision subsets は KQ2 の `first_unit` と `post_switch` 定義。`residual_ratio` は各窓の ‖出力−下書き‖₂ / ‖下書き腕6軸‖₂。','',
              '| split | 行 | 条件 | 腕6 MSE [95% episode CI] | 下書き | 出力−下書き | 残差比 mean / median | first_unit n/MSE | post_switch n/MSE | delta gate |',
              '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    if offline:
        for split in ('learned','unseen20','dev2'):
            for name,m in offline['splits'][split].items():
                q=m['residual_ratio']; subs=m['subsets']
                f=subs['first_unit']; p=subs['post_switch']
                lines.append(f'| {split} | {m["rows"]} | {name} | {m["arm6_mse"]:.6f} [{m["arm6_ci95"][0]:.6f}, {m["arm6_ci95"][1]:.6f}] | {m["draft_arm6_mse"]:.6f} | {m["draft_to_output_arm6_mse"]:.6f} | {q["mean"]:.3f} / {q["0.5"]:.3f} | {f["rows"]}/{f["arm6_mse"]:.6f} | {p["rows"]}/{p["arm6_mse"]:.6f} | {m["delta_gate_tanh"]:.5f} |')
    lines += ['', '## 試行別成功数', '']
    for suite in ('libero90','libero10'):
        lines.append(f'### {suite}')
        lines.append('')
        lines.append('| 条件 | task_id | 成功/10 |')
        lines.append('|---|---:|---:|')
        for cond in conditions:
            for task, s in per_task[suite][cond].items():
                lines.append(f'| {cond} | {task} | {s["successes"]}/10 |')
        lines.append('')
    lines += ['## 運用記録','',
              '- 訓練・評価の時刻は `production/progress.log` に UTC で追記。報告生成時刻は results.json の JST 欄に記録。',
              '- OOM が発生した場合は再試行回数と設定を progress.log に残す。GPU 割当は XLA fraction ごとに実測を記録。',
              '- KQ2-CL のベースラインは既存の trial-level logs を使用し、再実行していない。',
              '- 評価対象: 学習済み 621、未学習20Hz 619、Dev v2 667。学習データ・候補選択から未学習 Eval を除外。','']
    (RES1_ROOT/'report.md').write_text('\n'.join(lines))
    return result


if __name__=='__main__':
    compile_report()
