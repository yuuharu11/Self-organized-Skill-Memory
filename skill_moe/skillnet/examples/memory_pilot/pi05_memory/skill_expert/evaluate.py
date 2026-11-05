#!/usr/bin/env python3
"""Resumable closed-loop smoke/runner for conditions C, A, D (optional B)."""
from __future__ import annotations
import argparse, hashlib, json, os, signal, socket, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
HERE=Path(__file__).resolve().parent
PI05=HERE.parent
ROOT=Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104')
RUN=ROOT/'run-01'
CHECKPOINT=Path('/work/.skillnet-state/checkpoints/openpi-assets/checkpoints/pi05_libero')

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
def append(path,value):
    with Path(path).open('a') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def free_port():
    with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]
def wait_ready(proc,port,log):
    for _ in range(180):
        if proc.poll() is not None: raise RuntimeError(f'server exited; see {log}')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz',timeout=2) as r:
                if r.status==200:return
        except Exception:time.sleep(2)
    raise TimeoutError(f'policy server startup timed out: {log}')
def latest(path):
    result={}
    if path.exists():
        for line in path.read_text().splitlines():
            x=json.loads(line)
            if x.get('status')=='complete':result[(int(x['task_id']),int(x['trial_index']))]=x
    return result
def baseline_normal(path):
    result={}
    if path.exists():
        for line in path.read_text().splitlines():
            x=json.loads(line)
            if x.get('status')=='complete' and x.get('condition')=='normal' and x.get('task_suite')=='libero_10':
                result[(int(x['task_id']),int(x['trial_index']))]=x
    return result

def run_one(job,out,attempt,timeout):
    key=f"task{job['task_id']:02d}-init{job['trial_index']:02d}"
    folder=out/'attempts'/key/f'attempt-{attempt}';folder.mkdir(parents=True,exist_ok=False)
    jp,rp=folder/'job.json',folder/'result.json';write(jp,job)
    env=os.environ.copy();env['OMP_NUM_THREADS']=env['MKL_NUM_THREADS']=env['OPENBLAS_NUM_THREADS']='2';env['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    log=(folder/'worker.log').open('w');started=time.time()
    proc=subprocess.Popen(['taskset','-c','16-31','nice','-n','10',sys.executable,str(HERE/'trial_worker.py'),str(jp),str(rp)],
        stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    try:
        proc.wait(timeout=timeout)
        if proc.returncode or not rp.exists():return {'status':'error','error':f'worker exit {proc.returncode}','elapsed_seconds':time.time()-started,'attempt_dir':str(folder)}
        result=json.loads(rp.read_text());result['attempt']=attempt;result['attempt_dir']=str(folder);return result
    except subprocess.TimeoutExpired:
        try:os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            proc.wait()
        return {'status':'timeout','error':f'rollout exceeded {timeout}s','elapsed_seconds':time.time()-started,'attempt_dir':str(folder)}
    finally:log.close()

def run_condition(args,condition,tasks,trials):
    out=args.output/condition;out.mkdir(parents=True,exist_ok=True)
    port=args.port or free_port();status=out/'server';status.mkdir(exist_ok=True)
    env=os.environ.copy();env['OMP_NUM_THREADS']=env['MKL_NUM_THREADS']=env['OPENBLAS_NUM_THREADS']=str(args.server_threads)
    env['XLA_PYTHON_CLIENT_PREALLOCATE']='false';env['XLA_PYTHON_CLIENT_MEM_FRACTION']=str(args.mem_fraction)
    env['ADAPTIVE_TORCH_MEM_FRACTION']=str(args.torch_fraction)
    log=(out/'server.log').open('a')
    server=subprocess.Popen(['taskset','-c','8-15','nice','-n','10',sys.executable,str(HERE/'serve.py'),
       '--run',str(args.run),'--output',str(args.expert),'--checkpoint',str(args.checkpoint),
       '--condition',condition,'--port',str(port),'--status-dir',str(status)],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    try:
        wait_ready(server,port,out/'server.log')
        checks=json.loads((status/'server_checks.json').read_text())
        bcheck=checks.get('bprior_check',{})
        prior_delta=max(float(bcheck.get('cached_b_hat_max_abs_diff_vs_score_audit',1)),
                        float(bcheck.get('cached_r_hat_max_abs_diff_vs_score_audit',1)))
        if not checks.get('normal_fallback_exact') or prior_delta>1e-5:
            raise AssertionError('server did not pass fallback/prior checks')
        jobs=[]
        for task in tasks:
            for trial in trials:
                jobs.append({'host':'127.0.0.1','port':port,'suite':'libero_10','task_id':int(task),
                    'trial_index':int(trial),'seed_base':7,'condition':condition,'memory_snapshot':str(args.run),
                    'query_text_source':'instruction_only','checkpoint':str(args.checkpoint),
                    'wait_steps':10,'replan_steps':5,'max_steps':520})
        path=out/'rollouts.jsonl';done=latest(path);todo=[j for j in jobs if (j['task_id'],j['trial_index']) not in done]
        for job in todo:append(path,{'task_suite':'libero_10','task_id':job['task_id'],'trial_index':job['trial_index'],'condition':condition,'status':'in_progress'})
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(run_one,j,out,1,args.timeout):j for j in todo}
            for f in as_completed(futures):
                job=futures[f]
                try:r=f.result()
                except Exception as e:r={'status':'error','error':repr(e)}
                if r.get('status')!='complete':
                    r2=run_one(job,out,2,args.timeout)
                    if r2.get('status')=='complete':r=r2
                r.update(task_suite='libero_10',task_id=job['task_id'],trial_index=job['trial_index'],condition=condition)
                append(path,r)
                if r.get('status')=='complete':
                    if r.get('environment_seed')!=7+job['trial_index']:raise AssertionError('environment seed mismatch')
                    for qno,trace in enumerate(r.get('selected_units_and_t0',[])):
                        expected=(7+job['trial_index'])*100000+qno
                        if trace.get('query_seed')!=expected:raise AssertionError(f'query seed mismatch {trace.get("query_seed")} != {expected}')
                        if condition=='C' and trace.get('fallback') and not trace.get('normal_fallback_exact'):
                            raise AssertionError('fallback did not match official normal path')
                    # Compare the exact public initial-state object to run-01 normal.
                    baseline=baseline_normal(args.run/'rollouts.jsonl').get((job['task_id'],job['trial_index']))
                    if baseline is None:raise AssertionError('matching run-01 normal rollout not found')
                    if baseline.get('initial_state_sha256')!=r.get('initial_state_sha256'):
                        raise AssertionError('initial-state hash differs from run-01')
                    if baseline.get('trial_seed')!=r.get('trial_seed') or baseline.get('environment_seed')!=r.get('environment_seed'):
                        raise AssertionError('run-01 trial/environment seed mismatch')
                write(out/'progress.json',{'condition':condition,'complete':len(latest(path)),'scheduled':len(jobs),'updated':time.time()})
        records=latest(path);write(out/'summary.json',{'condition':condition,'completed':len(records),'scheduled':len(jobs),
            'successes':sum(bool(x.get('success')) for x in records.values()),'checks':checks,
            'task_init_hashes_match_run01':True,'seed_rules_verified':True})
    finally:
        if server.poll() is None:
            try:os.killpg(server.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:os.killpg(server.pid,signal.SIGKILL)
                except ProcessLookupError:pass
        log.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--expert',type=Path,required=True)
    p.add_argument('--run',type=Path,default=RUN);p.add_argument('--checkpoint',type=Path,default=CHECKPOINT)
    p.add_argument('--condition',choices=['C','A','D','B'],action='append');p.add_argument('--tasks',type=int,nargs='+')
    p.add_argument('--trials',type=int,nargs='+');p.add_argument('--smoke',action='store_true');p.add_argument('--port',type=int)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--timeout',type=int,default=600)
    p.add_argument('--server-threads',type=int,default=2)
    p.add_argument('--mem-fraction',type=float,default=.20);p.add_argument('--torch-fraction',type=float,default=.20)
    a=p.parse_args()
    if a.smoke:tasks,trials,conditions=[0],[0,1],a.condition or ['C','A','D']
    else:tasks,trials,conditions=a.tasks or list(range(10)),a.trials or list(range(10)),a.condition or ['C','A','D']
    if a.output.exists() and any(a.output.iterdir()):raise FileExistsError(f'refusing nonempty output: {a.output}')
    a.output.mkdir(parents=True,exist_ok=True)
    write(a.output/'protocol.json',{'conditions':conditions,'tasks':tasks,'trials':trials,'seed_base':7,
        'environment_seed':'7+init_state_id','query_seed':'(7+init_state_id)*100000+query_number',
        'wait_steps':10,'replan_steps':5,'max_steps':520,'environment_horizon':1000,
        'nfe':10,'workers':a.workers,'timeout_s':a.timeout,'smoke':a.smoke})
    for condition in conditions:run_condition(a,condition,tasks,trials)
    write(a.output/'DONE_SMOKE.json' if a.smoke else a.output/'DONE.json',{'conditions':conditions,'finished':time.time()})

if __name__=='__main__':main()
