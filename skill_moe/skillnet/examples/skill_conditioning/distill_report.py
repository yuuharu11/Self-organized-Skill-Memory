"""Matched rescue/harm accounting, kept separate from conditional probe recovery."""
import json
import csv
from pathlib import Path
from datetime import datetime,timezone


def pair_counts(rows,left,right):
    pairs=[]
    for case in sorted({r['job']['case'] for r in rows}):
        match={r['job']['condition']:r for r in rows if r['job']['case']==case}
        if left in match and right in match:
            a,b=match[left],match[right]
            pairs.append(dict(case=case,task=a['job']['task'],left=bool(a['success']),right=bool(b['success'])))
    rescue=[p for p in pairs if p['right'] and not p['left']]
    harm=[p for p in pairs if p['left'] and not p['right']]
    tasks={p['task'] for p in rescue}
    return dict(left=left,right=right,paired=len(pairs),rescue=len(rescue),harm=len(harm),net=len(rescue)-len(harm),rescued_tasks=sorted(tasks),pairs=pairs)


def summarize(root):
    root=Path(root);analysis=root/'analysis';analysis.mkdir(exist_ok=True)
    rows=[]
    for p in sorted((root/'evaluation').glob('*/result.json')):
        r=json.loads(p.read_text())
        if r.get('status')=='complete':rows.append(r)
    probes=[]
    for p in sorted((root/'probes').glob('*/result.json')):
        r=json.loads(p.read_text())
        if r.get('status')=='complete':probes.append(r)
    probe_rows=[dict(r,job=dict(r['job'],case=r['job']['case']+'-seed'+str(r['job']['seed']))) for r in probes]
    probe_pairs=[pair_counts(probe_rows,a,b) for a,b in [('S0','M0'),('S0','M1'),('M0','M1')]]
    pairs=[pair_counts(rows,a,b) for a,b in [('S0','M0'),('S0','M1'),('M0','M1')]]
    complete=len(rows)==54 and all(p['paired']==18 for p in pairs)
    gate=complete and all(pairs[i]['net']>=2 for i in (1,2)) and len(pairs[1]['rescued_tasks'])>=2 and pairs[1]['harm']==0
    errors={p.name:json.loads(p.read_text()) for p in root.glob('*ERROR.json')}
    result=dict(updated_utc=datetime.now(timezone.utc).isoformat(),main_complete=complete,completed_main=len(rows),planned_main=54,completed_probe=len(probes),planned_probe=18,counts={c:dict(success=sum(r['success'] for r in rows if r['job']['condition']==c),completed=sum(r['job']['condition']==c for r in rows)) for c in ('S0','M0','M1')},comparisons=pairs,probe_comparisons=probe_pairs,task_counts={str(t):{c:dict(success=sum(r['success'] for r in rows if r['job']['task']==t and r['job']['condition']==c),completed=sum(r['job']['task']==t and r['job']['condition']==c for r in rows)) for c in ('S0','M0','M1')} for t in range(9)},progress_gate=gate,decision='M1の再現確認へ進む候補' if gate else ('候補を採用しない' if complete else '未完了：採否を判断しない'),errors=errors)
    if complete and not gate and pairs[0]['net']>0 and pairs[0]['harm']==0 and pairs[2]['net']<=0:result['decision']='回復データの追加価値は未確認。通常微調整M0を優先する'
    report=['# C成功行動の元指示への蒸留：小規模検証（2026-11-03）','',f"更新：{result['updated_utc']}。主評価 {len(rows)}/54、補助評価 {len(probes)}/18 完了。未完了・欠測を失敗に数えない。",'',f"**判断：{result['decision']}。**",'','| 方策 | 成功 | 完了 |','|---|---:|---:|']
    for c,v in result['counts'].items():report.append(f"| {c} | {v['success']} | {v['completed']} |")
    report+=['','| 対応比較（左→右） | 組数 | 救済 | 悪化 | 正味 |','|---|---:|---:|---:|---:|']
    for p in pairs:report.append(f"| {p['left']}→{p['right']} | {p['paired']} | {p['rescue']} | {p['harm']} | {p['net']} |")
    report+=['','| タスク | S0 成功/完了 | M0 成功/完了 | M1 成功/完了 |','|---|---:|---:|---:|']
    for t,counts in result['task_counts'].items():report.append('| task%02d | '%(int(t)+1)+' | '.join(str(counts[c]['success'])+'/'+str(counts[c]['completed']) for c in ('S0','M0','M1'))+' |')
    report+=['','学習は元の全体指示A・現在観測だけを使用する。M0は通常成功、M1は75%通常成功＋25%C成功継続区間。Cの成功4軌道は3親走行であり、独立した4初期状態ではない。C開始点は後段の物理停滞診断点と異なるため、「停滞からの回復教師」と呼ばない。','', '単一学習seedによる追加学習・適応の予備検証であり、ゼロショット構成汎化や論文上の有効性を主張しない。損失低下・保存点からの復帰だけで採用しない。']
    if (root/'training_budget.json').exists():
        budget=json.loads((root/'training_budget.json').read_text());report+=['',f"共通更新数：{budget['updates']}。microbatch {budget['microbatch']}、蓄積 {budget['accumulation']}。2更新の動作確認＋20更新の速度計測後に元重みへ戻した。"]
    index_path=root/'data/index.json'
    if index_path.exists():
        data=json.loads(index_path.read_text());shards=data['shards']
        report+=['',f"データ監査：通常成功 {sum(x['samples'] for x in shards if x['kind']=='normal')} 窓、C成功 {sum(x['samples'] for x in shards if x['kind']=='recovery')} 窓。除外親走行 {len(data['excluded_parents'])}。全教師を画像・自己状態・公式成功まで再生照合し、正規化の往復を確認した。",'', '| 教師 | seed | 開始→成功step | 学習窓 |','|---|---:|---:|---:|']
        for r in shards:
            if r['kind']=='recovery':report.append(f"| {r['case']} | {r['id'].rsplit('-',1)[-1]} | {r['start_step']}→{r['end_step']} | {r['samples']} |")
    report+=['','公開損失の既存実装には、バッチ化されたprefixマスクと1次元suffixマスクをaxis=0で連結する不整合があった。両方をバッチ化してaxis=1で連結する互換修正を両モデルに適用した。通常のフローマッチングのノイズ・時刻分布・目標・補助損失は維持し、検索アンカーは使用しない。','']
    training={}
    for c in ('M0','M1'):
        path=root/'models'/c/'result.json'
        if path.exists():
            training[c]=json.loads(path.read_text());report.append(f"{c}：{training[c]['updates']}更新完了。凍結重みSHA一致：{training[c]['frozen_before']==training[c]['frozen_after']}。")
        else:
            log=root/'models'/c/'training.jsonl'
            lines=log.read_text().splitlines() if log.exists() else []
            valid=[]
            for line in lines[-2:]:
                try:valid.append(json.loads(line))
                except ValueError:pass
            report.append(f"{c}：{valid[-1]['step'] if valid else 0}更新記録済み、最終チェックポイント未確定。")
    result['training']=training
    selection_path=root/'EVALUATION_SELECTION.json'
    missing=[]
    if selection_path.exists():
        selection=json.loads(selection_path.read_text())
        for t,inits in selection.items():
            for init in inits:
                for c in ('S0','M0','M1'):
                    name=f'task{int(t)+1:02d}-init{init:02d}-{c}';directory=root/'evaluation'/name
                    if not (directory/'result.json').exists():missing.append(dict(directory='evaluation/'+name,status='incomplete' if (directory/'job.json').exists() else 'not_started'))
    result['missing_main']=missing
    result['probe_missing']=[json.loads(p.read_text()) for p in sorted((root/'probe_missing').glob('*.json'))]
    if missing:report+=['',f'主評価の未完了・未着手：{len(missing)}件。内訳はanalysis/summary.jsonのmissing_mainに保存する。']
    if errors:report+=['','実行上の未解決事項：'+json.dumps(errors,ensure_ascii=False)]
    with (analysis/'rollouts.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['phase','case','task','init','model','seed','success','start_step','end_step','query_count','inference_s','wall_s']);writer.writeheader()
        for phase,items in [('main',rows),('probe',probes)]:
            for r in items:
                j=r['job'];writer.writerow(dict(phase=phase,case=j['case'],task=j['task']+1,init=j['init'],model=j['condition'],seed=j['seed'],**{k:r[k] for k in ['success','start_step','end_step','query_count','inference_s','wall_s']}))
    (analysis/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    (analysis/'report.md').write_text('\n'.join(report)+'\n')
    target=Path('/work/docs/research/reviews/recovery_distillation_results_2026-11-03.md')
    target.write_text('\n'.join(report)+f'\n\nローカル成果物：`{root}`。生データ・画像は外部公開しない。\n')
    return result
