#!/usr/bin/env python3
"""Assemble the revised EX1 offline and paired LIBERO suite reports."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--production-dir',type=Path,required=True)
    a=p.parse_args(); run=a.run_dir
    offline=json.loads((run/'offline/offline_results.json').read_text())
    loop90=json.loads((run/'closedloop-libero90/results.json').read_text())
    loop10=json.loads((run/'closedloop-libero10/results.json').read_text())
    train=json.loads((run/'train-aggregate/train.json').read_text())
    modes=('D0_P0_top1','D1_P5_top1','D3_E_VLM_H_top1')
    rows=[]
    for split_name,label in (('learned_eval','学習済み621'),('unseen_eval','未学習619')):
        split=offline[split_name]
        for subset in ('all','hard'):
            for mode in modes:
                for kind in ('draft','expert_A','expert_B'):
                    name=f'{mode}_{kind}' if kind=='draft' else f'{mode}_expert_{kind[-1]}'
                    value=split['metrics'].get(name,{}).get(subset)
                    if value:
                        rows.append(f"| {label} | {subset} | {mode} | {kind} | {value['arm6_mse']['mean']:.6f} | {value['active7_mse']['mean']:.6f} |")
            for name in ('pi05','hold_last','oracle_draft_expert_A','oracle_draft_expert_B'):
                value=split['metrics'].get(name,{}).get(subset)
                if value:rows.append(f"| {label} | {subset} | — | {name} | {value['arm6_mse']['mean']:.6f} | {value['active7_mse']['mean']:.6f} |")
    lines=['# EX1：P0・P5・潜在トークン検索を下書きにした Expert 評価','',
      '## オフライン主表','',
      '| データ | 範囲 | 検索 | 出力 | arm6 MSE | active7 MSE（π0.5 gripper） |','|---|---|---|---|---:|---:|',*rows,'',
      'D0 は P0 top-1、D1 は固定 KEY2-F P5 top-1、D3 は固定 KEY2-F E-VLM+H top-1。D2/D4 はオフライン参考として `offline_results.json` に残した。全出力のグリッパーは通常 π0.5 に復元。',
      '','## 閉ループ主表','',
      '| suite | 条件 | 成功/完了 | Wilson 95% | baseline | McNemar exact p (base-only / EX1-only) | 技術エラー |','|---|---|---:|---:|---:|---:|---:|']
    for suite,label in ((loop90,'LIBERO-90 Eval (10×10)'),(loop10,'LIBERO-10 learned (10×5)')):
        baseline_n=100 if suite is loop90 else 50
        b=suite['baseline_successes'];lines.append(f'| {label} | 通常 π0.5 baseline | {b}/{baseline_n} | — | — | — | 0 |')
        for cond,data in suite['conditions'].items():
            lo,hi=data['wilson95'];lines.append(f"| {label} | {cond} | {data['successes']}/{data['completed']} | {lo:.1%}–{hi:.1%} | {data['baseline_successes']} | {data['mcnemar']['p_exact']:.6g} ({data['mcnemar']['baseline_only']}/{data['mcnemar']['expert_only']}) | {data['errors']} |")
    lines += ['', '### タスク別', '']
    for suite,label in ((loop90,'LIBERO-90'),(loop10,'LIBERO-10')):
        lines += [f'#### {label}', '', '| 条件 | task id | task | 成功/完了 | baseline成功 | McNemar p (base-only / EX1-only) |', '|---|---:|---|---:|---:|---:|']
        for cond,data in suite['conditions'].items():
            for tid,task in data['tasks'].items():
                lines.append(f"| {cond} | {tid} | {task['name']} | {task['successes']}/{task['completed']} | {task['baseline_successes']} | {task['mcnemar']['p_exact']:.6g} ({task['mcnemar']['baseline_only']}/{task['mcnemar']['ex1_only']}) |")
        lines.append('')
    lines += ['## 選択と条件','',
      '学習済み621行・未学習619行、難行マスクは固定。Expert A/B は各ドラフトの Dev arm6 MSE が小さい方を閉ループに用いた。D0 に専用の Expert Dev 選択値がない場合は、学習時点で固定済みの D1 Dev 選択を代用し、Eval を使って選択しない。',
      '学習済み閉ループは baseline-01/closed-loop-memory-01 の同一10タスク・trial 0–4・同 seed、未学習は baseline-01/production の固定10タスク・trial 0–9・同 seed。両方 seed_base=7、environment seed=7+trial、replan=5、wait=10、上限520、`SKILLNET_SUBTASK_GENERATE=0`。',
      '学習選択・各モードの step/t0/Dev MSE は `train-aggregate/train.json` と `results.json` に保存。', '',
      '指標はオフライン行動誤差とこの二つの閉ループ試行に限る。失敗・未完了は成功として扱わず、差がないことを同等とは解釈しない。']
    (run/'report.md').write_text('\n'.join(lines)+'\n')
    combined={'status':'complete','offline':offline,'closedloop_libero90':loop90,'closedloop_libero10':loop10,
              'training':train,'report':str(run/'report.md')}
    (run/'results.json').write_text(json.dumps(combined,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    # Keep the requested production-level entry point additive: retain the old
    # interrupted report/results and link to this completed scope-change run.
    root_report=a.production_dir/'report.md'
    link='\n\n## EX1 revised scope completion\n\nThe revised P0/P5/E-VLM+H comparison and learned/unseen closed-loop results are in [scope-change-01 report](scope-change-01/report.md) and [results](scope-change-01/results.json).\n'
    if root_report.exists():
        text=root_report.read_text()
        if '## EX1 revised scope completion' not in text:root_report.write_text(text.rstrip()+link)
    else:root_report.write_text('# EX1 production-03\n'+link)
    doc=Path('/work/docs/research/09_unseen_libero/experiments.md')
    marker='### EX1 範囲変更：P0・P5・E-VLM+H'
    entry=f"\n\n{marker}\n\nP0/P5/E-VLM+H top-1 を下書きに用いた再評価。621/619行のオフライン結果、LIBERO-90 Eval と LIBERO-10 learned の閉ループ表を [production-03 scope-change-01](../../../.skillnet-state/outputs/memory-pilot/unseen-libero/expert-ex1-01/production-03/scope-change-01/report.md) に記録。\n"
    if doc.exists():
        text=doc.read_text()
        if marker not in text:doc.write_text(text.rstrip()+entry)

if __name__=='__main__':main()
