#!/usr/bin/env python3
"""Resumable EX2 LIBERO-90/10 closed-loop evaluation against EX1 baselines."""
from __future__ import annotations
import argparse,json,os,socket,subprocess,sys,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_ex1_scope_changed as ex1run

MODES=('D1_P5_top1','D3_E_VLM_H_top1','D5_HS_top1')
PI05=Path('/work/.skillnet-state/checkpoints/openpi-assets/checkpoints/pi05_libero')

def free_port():
    with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]

def run_suite(suite,train_dir,retrieval_dir,candidate_cache,confidence,schedule,out,*,workers=24,fraction=.16,timeout=900,resume=True):
    out=Path(out)
    if out.exists() and not resume:raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True);(out/'servers').mkdir(exist_ok=True)
    tasks,baseline=ex1run.load_baseline(suite);trials=10 if suite=='libero90' else 5
    train=json.loads((Path(train_dir)/'train.json').read_text());sel=json.loads(Path(schedule).read_text())['expert']
    checkpoint=Path(sel['checkpoint']);chosen={'condition':sel['condition'],**sel}
    with __import__('numpy').load(Path(candidate_cache),allow_pickle=False) as z:
        source=np.asarray(z['source'],np.int32);actions=np.asarray(z['unit_windows'],np.float32)
    # New retrieval cache carries source keys in the exact same order as the
    # EX1 runner's immutable candidate bank. Action value equality catches any
    # accidental key/value permutation before launching environments.
    metadata=np.load(Path(retrieval_dir)/'candidate_metadata.npz',allow_pickle=False)
    if not np.array_equal(source,metadata['source']):
        raise AssertionError('EX2 retrieval candidate order mismatch with EX1 closed-loop bank')
    candidate_max_abs_diff=float(np.max(np.abs(actions-metadata['unit_actions'])))
    if candidate_max_abs_diff>1e-3:
        raise AssertionError(f'EX2 retrieval/replay candidate values differ by {candidate_max_abs_diff:g}')
    import expert_ex1
    p5=expert_ex1.load_locked_key2_checkpoints()['p5']['checkpoint']
    sys.path.insert(0,str(HERE.parent));from anchor_flow.run_case import S2WebsocketClientPolicy,_run_trial
    all_rows=[]
    for mode in MODES:
        cond=f'EX2_{mode}_confidence';trialfile=out/cond/'trials.jsonl';trialfile.parent.mkdir(exist_ok=True)
        existing={}
        if trialfile.exists():
            for line in trialfile.read_text().splitlines():
                if line.strip():
                    r=json.loads(line);existing[(int(r['task_id']),int(r['trial_index']))]=r
        pending=[(t,int(i)) for t in tasks for i in range(trials) if existing.get((t['task_id'],i),{}).get('status')!='complete']
        if pending:
            port=free_port();env=os.environ.copy();env.update(SKILLNET_SUBTASK_GENERATE='0',XLA_PYTHON_CLIENT_PREALLOCATE='false',
                XLA_PYTHON_CLIENT_MEM_FRACTION=str(fraction),TORCH_MEM_FRACTION=str(fraction),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            log=(out/'servers'/f'{mode}.log').open('a')
            cmd=['taskset','-c','8-15','nice','-n','10',sys.executable,str(HERE/'serve_expert_ex2.py'),
                '--mode',mode,'--candidate-cache',str(candidate_cache),'--retrieval-cache',str(retrieval_dir),
                '--schedule',str(schedule),'--confidence',str(Path(confidence)/'confidence_head.pt'),
                '--pi05-checkpoint',str(PI05),'--expert-checkpoint',str(checkpoint),'--p5-checkpoint',str(p5),'--port',str(port)]
            proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                deadline=time.monotonic()+600
                while True:
                    if proc.poll() is not None:raise RuntimeError(f'{mode} server exited; see {out}/servers/{mode}.log')
                    if time.monotonic()>deadline:raise TimeoutError(f'{mode} server did not become ready')
                    try:c=S2WebsocketClientPolicy('127.0.0.1',port);c.close();break
                    except (ConnectionError,OSError,TimeoutError):time.sleep(1)
                suite_api='libero_90' if suite=='libero90' else 'libero_10';jobs=[]
                for task,trial in pending:
                    t=next(x for x in tasks if x['task_id']==task['task_id'])
                    jobs.append({'host':'127.0.0.1','port':port,'suite':suite_api,'task_id':int(task['task_id']),
                       'task_key':t['name'],'trial_index':trial,'seed_base':7,'max_steps':520,'wait_steps':10,
                       'replan_steps':5,'condition':cond,'checkpoint':str(checkpoint),'timeout':timeout,
                       'memory_snapshot':'EX2 rank-sampled stored DCT3 units; no key-dependent selection',
                       'query_text_source':'instruction_only'})
                done=0
                with ProcessPoolExecutor(max_workers=int(workers)) as pool:
                    futs={pool.submit(_run_trial,j):j for j in jobs}
                    for fut in as_completed(futs):
                        r=fut.result();r['condition']=cond;r['baseline_success']=bool(baseline[(int(r['task_id']),int(r['trial_index']))]['success'])
                        r['expert_condition']=chosen['condition'];r['selected_checkpoint']=str(checkpoint)
                        with trialfile.open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
                        existing[(int(r['task_id']),int(r['trial_index']))]=r;done+=1
                        (out/'progress.json').write_text(json.dumps({'suite':suite,'condition':cond,'new_completed':done,'pending':len(pending),'time':time.time()},indent=2)+'\n')
            finally:
                if proc.poll() is None:proc.terminate()
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:proc.kill()
                log.close()
        all_rows.extend(existing.values())
    # Shared summary helpers preserve the same trial-pair and Wilson conventions.
    summaries=ex1run.summarize(all_rows,baseline,tasks,trials)
    for mode in MODES:
        name=f'EX2_{mode}_confidence';condition=summaries.get(name)
        rr=[r for r in all_rows if r.get('condition')==name]
        if condition is None:continue
        traces=[q for r in rr for q in r.get('selected_units_and_t0',[]) if isinstance(q,dict)]
        condition['policy_diagnostics']={'query_count':len(traces),
            'pi05_fallback_queries':sum(bool(q.get('fallback_pi05')) for q in traces),
            'pi05_fallback_fraction':float(np.mean([bool(q.get('fallback_pi05')) for q in traces])) if traces else None,
            't0_median':float(np.median([q['t0'] for q in traces if 't0' in q])) if any('t0' in q for q in traces) else None,
            'predicted_error_median':float(np.median([q['confidence_error'] for q in traces if 'confidence_error' in q])) if any('confidence_error' in q for q in traces) else None}
    result={'status':'complete' if all(len([r for r in all_rows if r['condition']==f'EX2_{m}_confidence' and r.get('status')=='complete'])==len(tasks)*trials for m in MODES) else 'partial',
      'suite':suite,'task_count':len(tasks),'trials_per_task':trials,'seed_base':7,'initial_states':f'0..{trials-1}',
      'replan_steps':5,'wait_steps':10,'max_steps':520,'subtask_generation':False,'selected_expert':chosen,
      'candidate_bank':{'source_order_exact':True,'max_abs_action_value_diff':candidate_max_abs_diff,'tolerance':1e-3},
      'baseline_source':str(ex1run.BASELINE90 if suite=='libero90' else ex1run.BASELINE10),'conditions':summaries}
    (out/'results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n');return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite',choices=('libero90','libero10','both'),default='both')
    p.add_argument('--train-dir',type=Path,required=True);p.add_argument('--retrieval-dir',type=Path,required=True);p.add_argument('--candidate-cache',type=Path,required=True)
    p.add_argument('--confidence',type=Path,required=True);p.add_argument('--schedule',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=24);p.add_argument('--fraction',type=float,default=.16);p.add_argument('--timeout',type=int,default=900);p.add_argument('--resume',action='store_true');a=p.parse_args()
    suites=('libero90','libero10') if a.suite=='both' else (a.suite,)
    result={s:run_suite(s,a.train_dir,a.retrieval_dir,a.candidate_cache,a.confidence,a.schedule,a.out/s,workers=a.workers,fraction=a.fraction,timeout=a.timeout,resume=a.resume) for s in suites}
    print(json.dumps(result,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
