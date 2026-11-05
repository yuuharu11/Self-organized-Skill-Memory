"""Resumable production controller; exp owns running it after the smoke."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import evaluate_smoke as common


def append_jsonl(path,row):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--trials-json',type=Path,required=True,
       help='JSON array of {task_id, trial_index, seed_base}; selected normal failures only')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--timeout',type=int,default=600)
    p.add_argument('--port',type=int)
    a=p.parse_args()
    if not 1<=a.workers<=4:raise ValueError('--workers must be in 1..4')
    common.OUT=a.output
    a.output.mkdir(parents=True,exist_ok=True)
    trials=json.loads(a.trials_json.read_text())
    if not isinstance(trials,list) or not trials:raise ValueError('trials-json must be a nonempty list')
    if len({(int(x['task_id']),int(x['trial_index'])) for x in trials})!=len(trials):
        raise ValueError('duplicate task/initial pair in trials-json')
    port=a.port or common.free_port()
    server_log=a.output/'server.log'
    with server_log.open('a') as log:
        server=subprocess.Popen(['taskset','-c','8-13',sys.executable,str(common.HERE/'serve.py'),
            '--run',str(common.RUN),'--checkpoint',str(common.CHECKPOINT),'--port',str(port)],
            stdout=log,stderr=subprocess.STDOUT,env=common.env_for(),start_new_session=True)
    all_results=[]
    try:
        common.wait_ready(server,port,server_log)
        for trial in trials:
            task_id,trial_index=int(trial['task_id']),int(trial['trial_index'])
            seed_base=int(trial.get('seed_base',7))
            replay=common.run_capture(port,task_id,trial_index,seed_base)
            expected=trial.get('expected_initial_state_sha256')
            if expected and replay['initial_state_sha256']!=expected:
                raise ValueError(f'initial state hash mismatch for task {task_id}, init {trial_index}')
            expected_seed=trial.get('expected_trial_seed')
            if expected_seed is not None and int(expected_seed)!=int(replay['trial_seed']):
                raise ValueError(f'trial seed mismatch for task {task_id}, init {trial_index}')
            if replay.get('success'):
                append_jsonl(a.output/'progress.jsonl',{'status':'skip_not_failure','task_id':task_id,
                    'trial_index':trial_index,'utc':time.time()})
                continue
            trial_dir=a.output/'trials'/f'task{task_id:02d}_init{trial_index:02d}'
            jobs=[]
            for cp in replay['checkpoints']:
                cp_path=Path(cp['path']); point=f'at_{int(float(cp["fraction"])*100):02d}'
                folder=trial_dir/'branches'/point
                jobs.extend({'name':'normal_alt_rng_'+str(i+1),'condition':'normal_alt',
                    'seed_offset':1009*(i+1),'checkpoint':cp_path,'folder':folder,
                    'replay_path':trial_dir/'normal_replay.json'} for i in range(8))
                jobs.extend([{'name':'zero_draft','condition':'zero','checkpoint':cp_path,'folder':folder},
                    {'name':'current_search','condition':'current','checkpoint':cp_path,'folder':folder}])
                jobs[-2]['replay_path']=jobs[-1]['replay_path']=trial_dir/'normal_replay.json'
                for rank,row in enumerate(cp['candidates'],1):
                    for t0 in (.4,.2):jobs.append({'name':f'candidate{rank:02d}_row{row}_t0_{t0:.1f}',
                        'condition':'memory_row','memory_row':int(row),'t0':t0,
                        'checkpoint':cp_path,'folder':folder,'replay_path':trial_dir/'normal_replay.json'})
            def submit(j,idx):
                job=dict(j,checkpoint=j['checkpoint'])
                return common.run_branch(job,port,j['folder'],idx,timeout=a.timeout)
            completed=0
            with ThreadPoolExecutor(max_workers=a.workers) as pool:
                futures={pool.submit(submit,j,i):j for i,j in enumerate(jobs)}
                for f in as_completed(futures):
                    result=f.result();result.update(task_id=task_id,trial_index=trial_index)
                    all_results.append(result);append_jsonl(a.output/'progress.jsonl',result)
                    completed+=1
            if completed!=len(jobs):raise RuntimeError('branch count mismatch')
        summary={'trials_requested':len(trials),'branch_results':len(all_results),
            'complete':sum(x['status']=='complete' for x in all_results),
            'errors':sum(x['status']=='error' for x in all_results),
            'timeouts':sum(x['status']=='timeout' for x in all_results),
            'parallel_workers':a.workers,'server_port':port,'server_log':str(server_log),
            'note':'Results are counterfactual LIBERO simulation rollouts, not physical robot evaluations.'}
        out_json=a.output/'summary.json'
        if out_json.exists():out_json=a.output/f'summary_{int(time.time())}.json'
        out_json.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps(summary,ensure_ascii=False))
    finally:
        try:os.killpg(server.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:os.killpg(server.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            server.wait()


if __name__=='__main__':main()
