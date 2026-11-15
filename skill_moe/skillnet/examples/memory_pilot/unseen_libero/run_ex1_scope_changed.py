#!/usr/bin/env python3
"""Resumable paired EX1 evaluation for the revised P0/P5/E-VLM+H comparison."""
from __future__ import annotations
import argparse, json, os, socket, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
BASE=Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero')
BASELINE90=BASE/'baseline-01/production'
BASELINE10=BASE/'closed-loop-memory-01/libero10/normal/attempts.jsonl'
PI05=Path('/work/.skillnet-state/checkpoints/openpi-assets/checkpoints/pi05_libero')
MODES={'D0_P0_top1':'D0_P0_top1','D1_P5_top1':'D1_P5_top1','D3_E_VLM_H_top1':'D3_E_VLM_H_top1'}

def free_port():
    with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]

def wilson(k,n):
    if not n:return [None,None]
    z=1.959963984540054;p=k/n;den=1+z*z/n
    c=(p+z*z/(2*n))/den;r=z*((p*(1-p)/n+z*z/(4*n*n))**.5)/den
    return [max(0.,c-r),min(1.,c+r)]

def mcnemar(base,exp):
    n10=sum(bool(a) and not bool(b) for a,b in zip(base,exp));n01=sum(not bool(a) and bool(b) for a,b in zip(base,exp));n=n10+n01
    if not n:return {'baseline_only':0,'expert_only':0,'discordant':0,'p_exact':1.0}
    import math
    tail=sum(math.comb(n,k) for k in range(min(n10,n01)+1))/(2**n)
    return {'baseline_only':n10,'expert_only':n01,'discordant':n,'p_exact':min(1.,2*tail)}

def load_baseline(suite):
    if suite=='libero90':
        plan=json.loads((BASE/'baseline-01/plan.json').read_text())['selected_tasks']
        tasks=[{'task_id':int(t['task_id']),'name':t['name'],'language':t['language']} for t in plan]
        rows={}
        for t in tasks:
            for trial in range(10):
                p=BASELINE90/'attempts'/f"task{t['task_id']:02d}-init{trial:02d}/attempt-1/result.json"
                r=json.loads(p.read_text())
                if r.get('status')!='complete' or int(r['trial_index'])!=trial:raise ValueError(f'incomplete baseline row {p}')
                rows[(t['task_id'],trial)]=r
    else:
        values=[json.loads(x) for x in BASELINE10.read_text().splitlines() if x.strip()]
        values=[r for r in values if r.get('condition')=='normal' and r.get('status')=='complete']
        if not values: values=[r for r in [json.loads(x) for x in BASELINE10.read_text().splitlines() if x.strip()] if r.get('status')=='complete']
        tasks=[]
        for r in sorted(values,key=lambda x:(int(x['task_id']),int(x['trial_index']))):
            if not any(t['task_id']==int(r['task_id']) for t in tasks):tasks.append({'task_id':int(r['task_id']),'name':r['task'],'language':r['task']})
        rows={(int(r['task_id']),int(r['trial_index'])):r for r in values}
        tasks=tasks[:10];rows={k:v for k,v in rows.items() if k[0] in {t['task_id'] for t in tasks} and k[1] in range(5)}
        for t in tasks:
            for trial in range(5):
                if (t['task_id'],trial) not in rows:raise FileNotFoundError(f'missing paired LIBERO-10 normal row {t["task_id"]}/{trial}')
    expected=10*(10 if suite=='libero90' else 5)
    if len(rows)!=expected or len(tasks)!=10:raise AssertionError(f'{suite}: baseline rows/tasks {len(rows)}/{len(tasks)} expected {expected}/10')
    return tasks,rows

def summarize(rows,baseline,tasks,trials):
    bycond={}
    for cond in sorted(set(r['condition'] for r in rows)):
        rr=[r for r in rows if r['condition']==cond]
        b=[baseline[(int(r['task_id']),int(r['trial_index']))] for r in rr]
        good=[r for r in rr if r.get('status')=='complete']
        task_summary={}
        for t in tasks:
            tr=[r for r in rr if int(r['task_id'])==t['task_id']]
            tb=[baseline[(t['task_id'],int(r['trial_index']))] for r in tr]
            g=[r for r in tr if r.get('status')=='complete']
            task_summary[str(t['task_id'])]={'name':t['name'],'completed':len(g),'successes':sum(bool(x.get('success')) for x in g),
                'wilson95':wilson(sum(bool(x.get('success')) for x in g),len(g)),
                'baseline_successes':sum(bool(x.get('success')) for x in tb),
                'mcnemar':mcnemar([x['success'] for x in tb],[x.get('success',False) for x in tr]),
                'median_steps':float(np.median([x['env_steps'] for x in g if x.get('success')])) if any(x.get('success') for x in g) else None}
        bycond[cond]={'completed':len(good),'successes':sum(bool(x.get('success')) for x in good),'errors':len(rr)-len(good),
            'wilson95':wilson(sum(bool(x.get('success')) for x in good),len(good)),
            'baseline_successes':sum(bool(x.get('success')) for x in b),
            'mcnemar':mcnemar([x['success'] for x in b],[x.get('success',False) for x in rr]),'tasks':task_summary}
    return bycond

def run_suite(suite,train_dir,cache,out,workers,fraction,timeout,resume=False):
    out=Path(out)
    if out.exists() and not resume:raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True);(out/'servers').mkdir(exist_ok=True)
    tasks,baseline=load_baseline(suite);trials=10 if suite=='libero90' else 5
    train=json.loads((Path(train_dir)/'train.json').read_text())
    chosen={}
    for mode in MODES:
        train_key=mode
        def best(c):
            d=train['conditions'][c]['best_by_draft']
            return d.get(train_key,d.get('D1_P5_top1'))
        c=min(('A','B'),key=lambda c:best(c)['mse6'])
        chosen[mode]={'condition':c,**best(c)}
    import expert_ex1
    p5=expert_ex1.load_locked_key2_checkpoints()['p5']['checkpoint']
    all_rows=[]
    for mode in MODES:
        cond=f'EX1_{mode}_expert';path=out/cond/'trials.jsonl';path.parent.mkdir(exist_ok=True)
        existing={}
        if path.exists():
            for line in path.read_text().splitlines():
                r=json.loads(line);existing[(int(r['task_id']),int(r['trial_index']))]=r
        pending=[(t,tr) for t in tasks for tr in range(trials) if existing.get((t['task_id'],tr),{}).get('status')!='complete']
        if pending:
            port=free_port();env=os.environ.copy();env.update(SKILLNET_SUBTASK_GENERATE='0',XLA_PYTHON_CLIENT_PREALLOCATE='false',
                XLA_PYTHON_CLIENT_MEM_FRACTION=str(fraction),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            log=(out/'servers'/f'{mode}.log').open('w')
            ck=chosen[mode];cmd=['taskset','-c','8-15','nice','-n','10',sys.executable,str(HERE/'serve_ex1_production.py'),
                '--mode',mode,'--candidate-cache',str(cache),'--p5-checkpoint',str(p5),'--pi05-checkpoint',str(PI05),
                '--expert-checkpoint',str(ck['checkpoint']),'--t0',str(ck['t0']),'--port',str(port)]
            proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                sys.path.insert(0,str(HERE.parent));from anchor_flow.run_case import S2WebsocketClientPolicy, _run_trial
                deadline=time.monotonic()+600
                while True:
                    if proc.poll() is not None:raise RuntimeError(f'{mode} server exited; inspect {out}/servers/{mode}.log')
                    if time.monotonic()>deadline:raise TimeoutError(f'{mode} server readiness timed out')
                    try:
                        c=S2WebsocketClientPolicy('127.0.0.1',port);c.close();break
                    except (ConnectionError,OSError,TimeoutError):time.sleep(1)
                jobs=[]
                suite_api='libero_90' if suite=='libero90' else 'libero_10'
                for t,tr in pending:
                    task=next(x for x in tasks if x['task_id']==t)
                    jobs.append({'host':'127.0.0.1','port':port,'suite':suite_api,'task_id':t,'task_key':task['name'],
                        'trial_index':tr,'seed_base':7,'max_steps':520,'wait_steps':10,'replan_steps':5,
                        'condition':cond,'memory_snapshot':'KEY2-F DCT3 4516','checkpoint':ck['checkpoint'],'timeout':timeout})
                with ProcessPoolExecutor(max_workers=int(workers)) as pool:
                    futures={pool.submit(_run_trial,j):j for j in jobs}
                    done=0
                    for future in as_completed(futures):
                        r=future.result();r['condition']=cond;r['baseline_success']=bool(baseline[(int(r['task_id']),int(r['trial_index']))]['success'])
                        r['expert_condition']=ck['condition'];r['selected_checkpoint']=ck['checkpoint']
                        with path.open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
                        existing[(int(r['task_id']),int(r['trial_index']))]=r;done+=1
                        (out/'progress.json').write_text(json.dumps({'suite':suite,'condition':cond,'completed_this_condition':done,'pending':len(pending),'updated_unix':time.time()},indent=2)+'\n')
            finally:
                if proc.poll() is None:proc.terminate()
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:proc.kill()
                log.close()
        all_rows.extend(existing.values())
    summary=summarize(all_rows,baseline,tasks,trials)
    result={'status':'complete','suite':suite,'task_count':len(tasks),'trials_per_task':trials,
       'seed_base':7,'environment_seed':'7 + trial_index','initial_states':f'0..{trials-1}',
       'wait_steps':10,'replan_steps':5,'max_steps':520,'subtask_generation':False,
       'baseline_source':str(BASELINE90 if suite=='libero90' else BASELINE10),
       'baseline_successes':sum(bool(x['success']) for x in baseline.values()),'experts_selected_by_dev':chosen,'conditions':summary}
    (out/'results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',choices=('libero90','libero10'),required=True)
    p.add_argument('--train-dir',type=Path,required=True);p.add_argument('--candidate-cache',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=24)
    p.add_argument('--fraction',type=float,default=.16);p.add_argument('--timeout',type=int,default=900)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    print(json.dumps(run_suite(a.suite,a.train_dir,a.candidate_cache,a.out,a.workers,a.fraction,a.timeout,a.resume),indent=2,ensure_ascii=False))
if __name__=='__main__':main()
