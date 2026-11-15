#!/usr/bin/env python3
"""Run resumable EX1 D1/D2 LIBERO-90 production conditions in parallel."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,os,socket,subprocess,sys,time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
PI05=HERE.parent/'pi05_memory'
BASELINE=Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01')
PI05_CKPT=Path('/work/.skillnet-state/checkpoints/openpi-assets/checkpoints/pi05_libero')

def free_port():
    with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]

def wilson(k,n):
    if not n:return [None,None]
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;radius=z*((p*(1-p)/n+z*z/(4*n*n))**.5)/den
    return [max(0.,center-radius),min(1.,center+radius)]

def mcnemar(a,b):
    a=list(map(bool,a));b=list(map(bool,b));n10=sum(x and not y for x,y in zip(a,b));n01=sum((not x) and y for x,y in zip(a,b))
    n=n10+n01
    if not n:return {'baseline_only':n10,'ex1_only':n01,'discordant':0,'p_exact':1.}
    tail=sum(__import__('math').comb(n,k) for k in range(min(n10,n01)+1))/2**n
    return {'baseline_only':n10,'ex1_only':n01,'discordant':n,'p_exact':min(1.,2*tail)}

def write_report(out, *, offline_path=None):
    out=Path(out);final=json.loads((out/'results.json').read_text())
    lines=['# EX1：検索下書きから直す Expert の本番評価','']
    if offline_path and Path(offline_path).is_file():
        offline=json.loads(Path(offline_path).read_text())
        lines += ['## オフライン主表','',
            '| データ | subset | 手法 | arm6 MSE | active7 MSE |','|---|---|---|---:|---:|']
        for splitname,split in (('学習済み621',offline['learned_eval']),('未学習619',offline['unseen_eval'])):
            for subset in ('all','hard'):
                for name in ('D1_P5_top1_draft','D1_P5_top1_expert_A','D1_P5_top1_expert_B',
                             'D2_P5_top20_draft','D2_P5_top20_expert_A','D2_P5_top20_expert_B',
                             'D3_E_VLM_H_top1_draft','D3_E_VLM_H_top1_expert_A','D3_E_VLM_H_top1_expert_B',
                             'D4_E_VJ_H_top20_draft','D4_E_VJ_H_top20_expert_A','D4_E_VJ_H_top20_expert_B',
                             'pi05','hold_last','zero_draft','pi05_nearest_draft',
                             'oracle_draft_expert_A','oracle_draft_expert_B'):
                    v=split['metrics'].get(name,{}).get(subset)
                    if v:
                        lines.append(f"| {splitname} | {subset} | {name} | {v['arm6_mse']['mean']:.6f} | {v['active7_mse']['mean']:.6f} |")
        lines += ['', '対比較の episode bootstrap 95% CI は `offline_results.json` の各 split に保存。hard 行は固定マスク（学習済み125、未学習134）。',
                  '']
    lines += ['## 閉ループ主表','',
        '| 条件 | success / processed | Wilson 95% | baseline same trials | McNemar exact p | errors | median successful steps |',
        '|---|---:|---:|---:|---:|---:|---:|']
    # Reuse baseline row in the summary, rather than recomputing its estimator.
    bcount=next(iter(final['conditions'].values()))['baseline_successes'] if final['conditions'] else 0
    lines.append(f"| 通常 π0.5 baseline-01 | {bcount}/100 | — | — | — | 0 | — |")
    for cond,d in final['conditions'].items():
        lo,hi=d['wilson95'];lo=0 if lo is None else lo;hi=0 if hi is None else hi
        med=[x['successful_steps_median'] for x in d['tasks'].values() if x['successful_steps_median'] is not None]
        # Pooled successful-trial median is calculated from the preserved JSONL.
        rows=[]
        for line in (Path(out)/cond/'trials.jsonl').read_text().splitlines():rows.append(json.loads(line))
        steps=[int(x['env_steps']) for x in rows if x.get('status')=='complete' and x.get('success')]
        pooled=float(np.median(steps)) if steps else None
        lines.append(f"| {cond} | {d['successes']}/{d['completed']} | {lo:.1%}–{hi:.1%} | {d['baseline_successes']}/100 | {d['mcnemar']['p_exact']:.6g} ({d['mcnemar']['baseline_only']} vs {d['mcnemar']['ex1_only']}) | {d['errors']} | {pooled if pooled is not None else '—'} |")
    lines += ['', '### タスク別','',
        '| 条件 | task id | 成功 / 完了 | Wilson 95% | baseline | McNemar p (base-only / EX1-only) | median steps |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for cond,d in final['conditions'].items():
        for tid,t in d['tasks'].items():
            lo,hi=t['wilson95'];lo=0 if lo is None else lo;hi=0 if hi is None else hi
            lines.append(f"| {cond} | {tid} | {t['successes']}/{t['completed']} | {lo:.1%}–{hi:.1%} | {t['baseline_successes']}/10 | {t['mcnemar']['p_exact']:.6g} ({t['mcnemar']['baseline_only']}/{t['mcnemar']['ex1_only']}) | {t['successful_steps_median'] if t['successful_steps_median'] is not None else '—'} |")
    lines += ['', '## 条件と注意','',
        f"- D1 の Dev 選択 Expert: {final['expert_selection']['D1_P5_top1']['condition']} step {final['expert_selection']['D1_P5_top1']['step']} t0={final['expert_selection']['D1_P5_top1']['t0']:.3f}; D2: {final['expert_selection']['D2_P5_top20']['condition']} step {final['expert_selection']['D2_P5_top20']['step']} t0={final['expert_selection']['D2_P5_top20']['t0']:.3f}.",
        '- D1 は locked KEY2-F P5 top-1、D2 は KEY2-F Dev 選択 k=20/temperature の保存済み単位窓ブレンド。Expert は腕6軸を補正し、グリッパーは通常 π0.5。D2 draft-only は bridge を呼ばない。',
        '- seed_base=7、environment seed=7+trial、query seed=(7+trial)×100000+query index、初期状態=trial 0–9、replan=5、wait=10、上限520。`SKILLNET_SUBTASK_GENERATE=0`。',
        '- 通常 π0.5 は同一 baseline-01/production 100 trial を再利用。成功率が一致しても統計的同等性を意味しない。',
        '- L90X task 51 の入力 HDF5 は欠損のまま。残る77タスクの fit rowsを使った。',
        '- 本評価は offline action metric と LIBERO-90 の閉ループ試行。一般化や全 LIBERO の性能を断定しない。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    return str(out/'report.md')

def _baseline_result(task_id,trial):
    path=BASELINE/'production/attempts'/f'task{task_id:02d}-init{trial:02d}'/'attempt-1/result.json'
    if not path.is_file():raise FileNotFoundError(f'missing paired baseline result: {path}')
    row=json.loads(path.read_text())
    if row.get('status')!='complete' or int(row.get('task_id',-1))!=task_id or int(row.get('trial_index',-1))!=trial:
        raise ValueError(f'baseline row does not match task/trial: {path}')
    return row

def main():
    p=argparse.ArgumentParser();p.add_argument('--train-dir',type=Path,required=True)
    p.add_argument('--candidate-cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=12);p.add_argument('--fraction',type=float,default=.16)
    p.add_argument('--timeout',type=int,default=900);p.add_argument('--offline-results',type=Path)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.out.exists() and not a.resume:raise FileExistsError(a.out)
    train=json.loads((a.train_dir/'train.json').read_text())
    chosen={}
    for mode in ('D1_P5_top1','D2_P5_top20'):
        cond=min(('A','B'),key=lambda c:train['conditions'][c]['best_by_draft'][mode]['mse6'])
        chosen[mode]={'condition':cond,**train['conditions'][cond]['best_by_draft'][mode]}
    selected=json.loads((BASELINE/'plan.json').read_text())['selected_tasks']
    tasks=[{'suite':'libero_90','task_id':int(x['task_id']),'task_key':x['name'],'language':x['language']} for x in selected]
    if len(tasks)!=10:raise ValueError('expected the fixed baseline 10-task evaluation set')
    baseline_rows={(t['task_id'],j):_baseline_result(t['task_id'],j) for t in tasks for j in range(10)}
    if len(baseline_rows)!=100:raise AssertionError('baseline pairing incomplete')
    a.out.mkdir(parents=True,exist_ok=True)
    # Load locked KEY2-F P5 weights from its frozen selection manifest.
    sys.path.insert(0,str(HERE))
    import expert_ex1
    p5_ckpt=Path(expert_ex1.load_locked_key2_checkpoints()['p5']['checkpoint'])
    logs=a.out/'servers';logs.mkdir()
    ports={'D1':free_port(),'D2':free_port()}
    env0=os.environ.copy();env0.update(SKILLNET_SUBTASK_GENERATE='0',XLA_PYTHON_CLIENT_PREALLOCATE='false',
        XLA_PYTHON_CLIENT_MEM_FRACTION=str(a.fraction),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    specs=[('D1','D1_P5_top1',chosen['D1_P5_top1']),('D2','D2_P5_top20',chosen['D2_P5_top20'])]
    procs=[];log_handles=[]
    try:
        # D1 and D2 servers run concurrently; D2 also handles its no-Expert arm.
        for tag,mode,selected_ckpt in specs:
            port=ports[tag];log=(logs/f'{tag}.log').open('w');log_handles.append(log)
            ckpt=selected_ckpt['checkpoint']
            cmd=['taskset','-c','8-11' if tag=='D1' else '12-15','nice','-n','10',sys.executable,
                str(HERE/'serve_ex1_production.py'),'--mode',mode,'--candidate-cache',str(a.candidate_cache),
                '--p5-checkpoint',str(p5_ckpt),'--pi05-checkpoint',str(PI05_CKPT),'--expert-checkpoint',str(ckpt),
                '--t0',str(selected_ckpt['t0']),'--port',str(port)]
            proc=subprocess.Popen(cmd,env=env0,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);procs.append(proc)
        # Await both websocket metadata handshakes without taking a policy step.
        sys.path.insert(0,str(HERE.parent))
        from anchor_flow.run_case import S2WebsocketClientPolicy
        for port in ports.values():
            deadline=time.monotonic()+600
            while True:
                if time.monotonic()>deadline:raise TimeoutError(f'EX1 server failed readiness on {port}')
                if any(q.poll() is not None for q in procs):raise RuntimeError('EX1 server exited during startup')
                try:
                    client=S2WebsocketClientPolicy('127.0.0.1',port);client.close();break
                except (ConnectionError,OSError,TimeoutError):time.sleep(1)
        from anchor_flow import run_case
        jobs=[]
        for tag,mode,ck in specs:
            for task in tasks:
                for trial in range(10):
                    base=baseline_rows[(task['task_id'],trial)]
                    common={'host':'127.0.0.1','port':ports[tag],'suite':'libero_90','task_id':task['task_id'],
                        'task_key':task['task_key'],'trial_index':trial,'seed_base':7,'max_steps':520,'wait_steps':10,
                        'replan_steps':5,'condition':f'EX1_{mode}_expert','memory_snapshot':'KEY2-F locked P5 + DCT3 4516',
                        'query_text_source':'instruction_only','checkpoint':ck['checkpoint'],'timeout':a.timeout}
                    jobs.append(common)
                    if tag=='D2':
                        jobs.append({**common,'condition':'EX1_D2_draft_only','checkpoint':'none','draft_only':True})
        # Each condition owns a JSONL so a terminated run can be resumed by
        # re-running a fresh attempt directory from remaining trial indices.
        pending=[]
        for job in jobs:
            condition=job['condition'];trialout=a.out/condition
            trialout.mkdir(exist_ok=True)
            job['_result_file']=str(trialout/'trials.jsonl')
            if job.get('draft_only'):job['_port_tag']='D2'
            result_file=Path(job['_result_file'])
            latest={}
            if result_file.exists():
                for line in result_file.read_text().splitlines():
                    old=json.loads(line)
                    latest[(int(old['task_id']),int(old['trial_index']))]=old.get('status')
            if latest.get((int(job['task_id']),int(job['trial_index'])))!='complete':
                pending.append(job)
        start=time.monotonic();done=0;results=[]
        with ProcessPoolExecutor(max_workers=int(a.workers)) as pool:
            futures={pool.submit(run_case._run_trial,{k:v for k,v in job.items() if not k.startswith('_')}):job for job in pending}
            # Append incrementally from the parent; original request's trial order
            # is still deterministic, while worker completion may be out of order.
            for fut in as_completed(futures):
                job=futures[fut];row=fut.result();row['baseline_success']=bool(baseline_rows[(job['task_id'],job['trial_index'])]['success'])
                row['condition']=job['condition'];row['selected_expert_condition']=chosen['D1_P5_top1']['condition'] if job['condition'].startswith('EX1_D1') else chosen['D2_P5_top20']['condition']
                path=Path(job['_result_file'])
                with path.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
                results.append(row);done+=1
                if done%10==0:
                    (a.out/'progress.json').write_text(json.dumps({'completed':done,'total':len(pending),'elapsed_seconds':time.monotonic()-start,
                        'by_condition_task':_partial_counts(results)},indent=2)+'\n')
        (a.out/'progress.json').write_text(json.dumps({'completed':done,'total':len(pending),'elapsed_seconds':time.monotonic()-start,
            'by_condition_task':_partial_counts(results)},indent=2)+'\n')
        # Read the latest row per task/trial so --resume replaces prior errors.
        results=[]
        for condition in sorted({job['condition'] for job in jobs}):
            path=a.out/condition/'trials.jsonl'
            latest={}
            if path.exists():
                for line in path.read_text().splitlines():
                    row=json.loads(line)
                    latest[(int(row['task_id']),int(row['trial_index']))]=row
            results.extend(latest.values())
        # Pair tables against exactly the baseline task/trial results.
        summary={}
        for cond in sorted({r['condition'] for r in results}):
            rows=[r for r in results if r['condition']==cond];bytask={}
            for task in tasks:
                tr=[r for r in rows if int(r['task_id'])==task['task_id']]
                base=[baseline_rows[(task['task_id'],int(r['trial_index']))] for r in tr]
                good=[r for r in tr if r.get('status')=='complete']
                bytask[str(task['task_id'])]={'name':task['task_key'],'completed':len(good),'errors':len(tr)-len(good),
                    'successes':sum(bool(r.get('success')) for r in good),'wilson95':wilson(sum(bool(r.get('success')) for r in good),len(good)),
                    'baseline_successes':sum(bool(x['success']) for x in base),
                    'mcnemar':mcnemar([x['success'] for x in base],[r.get('success',False) for r in tr]),
                    'successful_steps_median':float(np.median([r['env_steps'] for r in good if r.get('success')])) if any(r.get('success') for r in good) else None}
            good=[r for r in rows if r.get('status')=='complete'];base=[baseline_rows[(int(r['task_id']),int(r['trial_index']))] for r in rows]
            summary[cond]={'completed':len(good),'errors':len(rows)-len(good),'successes':sum(bool(r.get('success')) for r in good),
                'wilson95':wilson(sum(bool(r.get('success')) for r in good),len(good)),
                'baseline_successes':sum(bool(x['success']) for x in base),
                'mcnemar':mcnemar([x['success'] for x in base],[r.get('success',False) for r in rows]),'tasks':bytask}
        final={'status':'complete','task_ids':[t['task_id'] for t in tasks],'trials_per_task':10,'seed_base':7,
            'protocol':{'suite':'libero_90','initial_states':'0..9','replan_steps':5,'wait_steps':10,'max_steps':520,
                'environment_seed':'7 + trial_index','query_seed':'(7 + trial_index) * 100000 + query_number',
                'subtask_generation':False,'pi05_baseline':'baseline-01/production paired runs reused'},
            'expert_selection':chosen,'baseline_source':str(BASELINE/'production'),'conditions':summary,
            'elapsed_seconds':time.monotonic()-start}
        (a.out/'results.json').write_text(json.dumps(final,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
        write_report(a.out,offline_path=a.offline_results)
        return final
    finally:
        for proc in procs:
            if proc.poll() is None:proc.terminate()
        for proc in procs:
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:proc.kill()
        for h in log_handles:h.close()

def _partial_counts(rows):
    out={}
    for r in rows:
        key=f"{r.get('condition')}|{r.get('task_id')}";v=out.setdefault(key,{'completed':0,'errors':0,'success':0})
        v['completed']+=int(r.get('status')=='complete');v['errors']+=int(r.get('status')!='complete');v['success']+=int(bool(r.get('success')))
    return out

if __name__=='__main__':
    main()
