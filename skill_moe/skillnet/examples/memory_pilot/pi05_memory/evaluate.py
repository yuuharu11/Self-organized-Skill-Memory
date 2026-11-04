"""Paired, budget-bounded LIBERO evaluation with an explicitly fixed order."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from run import HERE,write_json


def wilson(success,total):
    if total==0:
        return None
    z=1.959963984540054
    p=success/total;den=1+z*z/total
    center=(p+z*z/(2*total))/den
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [center-half,center+half]


def summarize(rows):
    latest={(r['task_suite'],r['task_id'],r['trial_index'],r['condition']):r for r in rows}
    result={}
    for suite in ['libero_10','libero_spatial','libero_object','libero_goal']:
        conditions={}
        for condition in ['normal','memory']:
            subset=[r for k,r in latest.items() if k[0]==suite and k[3]==condition and r['status']=='complete']
            n=len(subset);wins=sum(int(r['success']) for r in subset)
            task_rates={str(task):sum(int(r['success']) for r in subset if r['task_id']==task)/sum(r['task_id']==task for r in subset)
                        for task in sorted({r['task_id'] for r in subset})}
            conditions[condition]={'successes':wins,'complete':n,'success_rate':wins/n if n else None,
                'wilson95':wilson(wins,n),
                'task_success_rates':task_rates,
                'task_macro_success_rate':sum(task_rates.values())/len(task_rates) if task_rates else None,
                'tasks_with_completed_rollouts':len(task_rates),
                'errors':sum(1 for k,r in latest.items() if k[0]==suite and k[3]==condition and r['status']!='complete')}
        paired=[]
        for key,r in latest.items():
            if key[0]!=suite or key[3]!='normal' or r['status']!='complete':
                continue
            other=latest.get((*key[:3],'memory'))
            if other and other['status']=='complete':
                if r['initial_state_sha256']!=other['initial_state_sha256']:
                    raise ValueError('paired initial states differ')
                paired.append((int(r['success']),int(other['success'])))
        rescue=sum(a==0 and b==1 for a,b in paired)
        harm=sum(a==1 and b==0 for a,b in paired)
        discordant=rescue+harm
        p=min(1.,2*sum(math.comb(discordant,i) for i in range(min(rescue,harm)+1))/(2**discordant)) if discordant else 1.
        result[suite]={'conditions':conditions,'paired':len(paired),'rescues':rescue,'harms':harm,
            'paired_difference':(rescue-harm)/len(paired) if paired else None,'exact_mcnemar_p':p}
    return result


def write_report(output,rows,status):
    summary=summarize(rows)
    lines=['# π0.5 LIBERO-40 運動メモリ：閉ループ評価', '',
           f'実行状態：{status}。同じ公開π0.5の通常生成と、運動メモリ＋修正Expertを比較。', '',
           '| 評価集合 | 通常π0.5 成功/完了 | 提案 成功/完了 | 対応組 | 救済/悪化 | 対応差 |',
           '|---|---:|---:|---:|---:|---:|']
    for suite,result in summary.items():
        a,b=[result['conditions'][c] for c in ['normal','memory']]
        delta=result['paired_difference']
        label='未評価' if delta is None else f'{100*delta:+.2f}ポイント'
        lines.append(f"| {suite} | {a['successes']}/{a['complete']} | {b['successes']}/{b['complete']} | "
                     f"{result['paired']} | {result['rescues']}/{result['harms']} | {label} |")
    lines+=['', '公開参考値：OpenPI π0.5のLong 92.4%、SkillNet LIBERO-40のLong 98.4%。',
            'これらは他論文・公式配布の参考値であり、本実験との対応のある比較ではない。',
            'タスク・初期状態・ノイズseedは固定し、完了した失敗を再試行していない。環境/通信エラーの再試行は元記録も保持する。',
            '学習済み40タスク上の評価であり、未知タスクへの汎化を主張しない。途中の成功率や少数試行から98.4%超えを確定しない。', '',
            '## 信頼度と学習', '']
    for filename in ['trained.json','codec.json','calibration.json']:
        path=output/filename
        if path.exists():
            values=json.loads(path.read_text())
            if filename=='trained.json':values={k:v for k,v in values.items() if k!='logs'}
            lines.extend([f'### {filename}', '', '```json',json.dumps(values,ensure_ascii=False,indent=2),'```',''])
    lines+=['詳細：`closedloop_summary.json`（区間推定・タスク別・対応検定）、`rollouts.jsonl`（全試行の生ログ）。']
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    from concurrent.futures import TimeoutError as FuturesTimeout
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8154)
    parser.add_argument('--hours',type=float,default=24.)
    args=parser.parse_args()
    sys.path.insert(0,str(HERE.parent/'anchor_flow'))
    from run_case import _run_trial,suite_control_limit
    deadline=(args.output/'experiment.json').stat().st_mtime+args.hours*3600
    if time.time()>=deadline:
        write_json(args.output/'evaluation_status.json',{'status':'not_started_budget_exhausted'});return
    path=args.output/'rollouts.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    done={(r['task_suite'],r['task_id'],r['trial_index'],r['condition']) for r in rows if r['status']=='complete'}
    # A fixed schedule, never choose tasks from observed success/failure.
    suites=['libero_10','libero_spatial','libero_object','libero_goal']
    schedule=[(suite,task,initial) for initial in range(5) for suite in suites for task in range(10)]
    schedule += [('libero_10',task,initial) for initial in range(5,50) for task in range(10)]
    write_json(args.output/'evaluation_protocol.json',{'primary_suite':'libero_10',
        'screen':'all40 tasks x 5 public initial states x 2 conditions',
        'expansion':'Long only to 50 public initial states/task if time remains',
        'order':schedule,'deadline_unix':deadline,'nfe':10,'replan_steps':5,'seed_base':7,
        'published_reference_long':98.4,'reference_note':'cross-paper reference, not a matched significance test',
        'qualification':'public pretrained pi05 may have seen all task demonstrations; no unseen-task claim'})
    server_log=(args.output/'server.log').open('a')
    proc=subprocess.Popen([sys.executable,str(HERE/'serve.py'),'--output',str(args.output),
        '--checkpoint',str(args.checkpoint),'--port',str(args.port)],stdout=server_log,stderr=subprocess.STDOUT)
    try:
        import urllib.request
        ready=False
        for _ in range(180):
            if proc.poll() is not None:
                raise RuntimeError('policy server exited; see server.log')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/healthz',timeout=2) as response:
                    ready=response.status==200
            except Exception:
                pass
            if ready:break
            time.sleep(5)
        if not ready:raise TimeoutError('server did not become ready')
        for suite,task,initial in schedule:
            # Do not start a partial pair when only a few minutes remain.
            if time.time()>deadline-600:break
            order=['normal','memory'] if (task+initial)%2==0 else ['memory','normal']
            for condition in order:
                identity=(suite,task,initial,condition)
                if identity in done:continue
                job={'host':'127.0.0.1','port':args.port,'suite':suite,'task_id':task,
                    'trial_index':initial,'seed_base':7,'condition':condition,
                    'memory_snapshot':str(args.output),'query_text_source':'instruction_only',
                    'checkpoint':str(args.checkpoint),'wait_steps':10,'replan_steps':5,
                    'max_steps':suite_control_limit(suite)}
                # A fresh simulator process per rollout prevents accumulated
                # renderer resources; infer server is serialized for paired seeds.
                with ProcessPoolExecutor(max_workers=1) as pool:
                    row=pool.submit(_run_trial,job).result()
                rows.append(row)
                with path.open('a') as stream:
                    stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
                write_json(args.output/'closedloop_summary.json',summarize(rows))
                if len(rows)%10==0:write_report(args.output,rows,'評価中')
                write_json(args.output/'evaluation_status.json',{'status':'running','last':identity,
                    'recorded_rollouts':len(rows),'remaining_hours':max(0,deadline-time.time())/3600})
                if row['status']!='complete':
                    # Retry the identical state/seed once; keep both attempts
                    # and never retry a complete task failure.
                    with ProcessPoolExecutor(max_workers=1) as pool:
                        retry=pool.submit(_run_trial,job).result()
                    retry['retry_of_infrastructure_error']=True
                    rows.append(retry)
                    with path.open('a') as stream:
                        stream.write(json.dumps(retry)+'\n');stream.flush();os.fsync(stream.fileno())
                    write_json(args.output/'closedloop_summary.json',summarize(rows))
                    if retry['status']!='complete':
                        write_json(args.output/'evaluation_status.json',{'status':'infrastructure_error','last':identity,
                                   'error':retry.get('error'),'recorded_rollouts':len(rows)})
                        raise RuntimeError('repeated rollout infrastructure error; raw attempts retained')
                done.add(identity)
        write_json(args.output/'evaluation_status.json',{'status':'complete' if len(done)==2*len(schedule) else 'budget_stopped',
            'recorded_rollouts':len(rows),'summary':str(args.output/'closedloop_summary.json')})
        write_report(args.output,rows,'完了' if len(done)==2*len(schedule) else '時間予算までの完了組を集計')
    finally:
        proc.terminate()
        try:proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill();proc.wait()
        server_log.close()


if __name__=='__main__':
    main()
