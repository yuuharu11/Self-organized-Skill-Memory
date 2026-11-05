"""Resumable full controller for the 44-condition rescue branch protocol.

This is for exp to launch from a fresh snapshot after the single-failure smoke.
It deliberately defaults to no production output path.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

HERE=Path(__file__).resolve().parent
PI05=HERE.parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(PI05))
sys.path.insert(0,str(PI05.parent/'anchor_flow'))
from evaluate_expanded_smoke import env_for, free_port, wait_ready
from key_merge import KEY_ORDER
from rollout_extended import capture

ROOT=Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/rescue-branching-01')
RUN=ROOT.parent/'run-01'
CHECKPOINT=Path('/work/.skillnet-state/checkpoints/openpi-assets/checkpoints/pi05_libero')


def append_jsonl(path, record):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(record,ensure_ascii=False)+'\n')
        f.flush();os.fsync(f.fileno())


def find_valid_capture(trial_home):
    attempts=sorted(trial_home.glob('capture-attempt-*'))
    for attempt in reversed(attempts):
        replay_path=attempt/'trials'/trial_home.name/'normal_replay.json'
        if not replay_path.exists():
            continue
        try:
            replay=json.loads(replay_path.read_text())
            if replay.get('status')!='complete' or len(replay.get('checkpoints',[]))!=3:
                continue
            if all(Path(cp['path']).exists() and Path(cp['manifest_path']).exists()
                   for cp in replay['checkpoints']):
                return attempt,replay
        except Exception:
            continue
    return None,None


def next_capture_attempt(trial_home):
    trial_home.mkdir(parents=True,exist_ok=True)
    used=[]
    for path in trial_home.glob('capture-attempt-*'):
        try: used.append(int(path.name.rsplit('-',1)[1]))
        except ValueError: pass
    path=trial_home/f'capture-attempt-{max(used,default=0)+1:02d}'
    path.mkdir()
    return path


def candidate_jobs(cp, attempt, branch_dir):
    manifest=json.loads(Path(cp['manifest_path']).read_text())
    candidates=manifest['merged_candidates']
    if len(candidates)!=8 or len({int(x['memory_row']) for x in candidates})!=8:
        raise ValueError('branchpoint candidate manifest must contain eight distinct rows')
    jobs=[]
    for i in range(8):
        jobs.append({'name':f'normal_alt_rng_{i+1:02d}','condition':'normal_alt',
                     'seed_offset':1009*(i+1)})
    for t0 in (.6,.2):
        jobs.append({'name':f'zero_t0_{t0:.1f}','condition':'zero_fixed','t0':t0})
        jobs.append({'name':f'current_vlm_t0_{t0:.1f}','condition':'current_fixed','t0':t0})
    provenance_dir=branch_dir/'candidate_provenance'
    provenance_dir.mkdir(parents=True,exist_ok=True)
    for ix,candidate in enumerate(candidates,1):
        row=int(candidate['memory_row'])
        provenance=provenance_dir/f'row_{row}.json'
        value={'memory_row':row,'key_ranks':candidate['key_ranks']}
        if provenance.exists():
            if json.loads(provenance.read_text())!=value:
                raise ValueError(f'candidate provenance mismatch at {provenance}')
        else:
            provenance.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
        for fmt,condition in [('codec','candidate_codec'),('raw','candidate_raw')]:
            for t0 in (.4,.2):
                jobs.append({'name':f'candidate{ix:02d}_row{row}_{fmt}_t0_{t0:.1f}',
                    'condition':condition,'memory_row':row,'t0':t0,
                    'candidate_sources_json':provenance})
    if len(jobs)!=44:
        raise AssertionError(f'expected 44 branches per point, got {len(jobs)}')
    return jobs


def execute_job(job, *, port, checkpoint_path, replay_path, branch_dir, timeout, worker_index):
    base=branch_dir/f'{job["name"]}.json'
    if base.exists():
        prior=json.loads(base.read_text())
        if prior.get('status')=='complete':
            prior.update(name=job['name'],reused=True,result_path=str(base))
            return prior
        attempt=2
        while (branch_dir/f'{job["name"]}_attempt{attempt}.json').exists():
            attempt+=1
        out=branch_dir/f'{job["name"]}_attempt{attempt}.json'
    else:
        out=base
    log=out.with_suffix('.log')
    cmd=['taskset','-c',('12-16','17-21','22-26','27-31')[worker_index%4],
         'nice','-n','10',sys.executable,str(HERE/'rollout_extended.py'),
         '--port',str(port),'--checkpoint',str(checkpoint_path),'--replay',str(replay_path),
         '--condition',job['condition'],'--output',str(out)]
    if job.get('memory_row') is not None:
        cmd += ['--memory-row',str(job['memory_row']),'--t0',str(job['t0']),
                '--candidate-sources-json',str(job['candidate_sources_json'])]
    elif job.get('t0') is not None:
        cmd += ['--t0',str(job['t0'])]
    if job.get('seed_offset'):
        cmd += ['--seed-offset',str(job['seed_offset'])]
    start=time.monotonic()
    with log.open('w') as f:
        try:
            result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env_for(threads='5'),
                                  timeout=timeout,start_new_session=True)
        except subprocess.TimeoutExpired:
            return {'name':job['name'],'status':'timeout','elapsed_seconds':float(timeout),'log':str(log)}
    if result.returncode or not out.exists():
        return {'name':job['name'],'status':'error','returncode':result.returncode,
                'elapsed_seconds':time.monotonic()-start,'log':str(log)}
    value=json.loads(out.read_text())
    value.update(name=job['name'],status='complete',elapsed_seconds=time.monotonic()-start,
                 result_path=str(out),log=str(log))
    return value


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True,
        help='new output directory (do not use the existing preflight production directory)')
    p.add_argument('--trials-json',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--timeout',type=int,default=600)
    p.add_argument('--port',type=int)
    a=p.parse_args()
    if not 1<=a.workers<=4:raise ValueError('workers must be in 1..4')
    # Production output may be resumed, but a previous preflight or unrelated
    # output directory must not be silently adopted.
    a.output.mkdir(parents=True,exist_ok=True)
    lock=a.output/'controller.lock'
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError as e:
        raise RuntimeError(f'controller lock exists, inspect before resuming: {lock}') from e
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    trials=json.loads(a.trials_json.read_text())
    if not isinstance(trials,list) or not trials:raise ValueError('trials-json must be nonempty')
    if len(trials)>10:raise ValueError('the approved maximum is ten failures')
    if len({(int(x['task_id']),int(x['trial_index'])) for x in trials})!=len(trials):
        raise ValueError('duplicate task/initial pair')
    port=a.port or free_port()
    server_log=a.output/'server.log'
    with server_log.open('a') as f:
        server_env=env_for(threads='4')
        # Conservative per-process limits: the token index retains ~18 GiB and
        # the shared GPU may also host unrelated experiments.
        server_env['XLA_PYTHON_CLIENT_MEM_FRACTION']='0.11'
        server_env['BRANCH_TORCH_MEM_FRACTION']='0.15'
        server=subprocess.Popen(['taskset','-c','8-11',sys.executable,str(HERE/'serve_extended.py'),
            '--run',str(RUN),'--checkpoint',str(CHECKPOINT),'--port',str(port)],
            stdout=f,stderr=subprocess.STDOUT,env=server_env,start_new_session=True)
    records=[]
    try:
        wait_ready(server,port,server_log)
        for trial in trials:
            task,init=int(trial['task_id']),int(trial['trial_index'])
            seed_base=int(trial.get('seed_base',7))
            trial_home=a.output/'trials'/f'task{task:02d}_init{init:02d}'
            attempt,replay=find_valid_capture(trial_home)
            if replay is None:
                attempt=next_capture_attempt(trial_home)
                replay=capture(port,task,init,seed_base,attempt)
            if replay.get('success'):
                append_jsonl(a.output/'progress.jsonl',{'status':'skip_not_failure','task_id':task,
                    'trial_index':init,'capture':str(attempt),'utc':time.time()})
                continue
            if trial.get('expected_trial_seed') is not None and int(replay['trial_seed'])!=int(trial['expected_trial_seed']):
                raise ValueError(f'trial seed mismatch for task {task} init {init}')
            expected=trial.get('expected_initial_state_sha256')
            if expected and replay['initial_state_sha256']!=expected:
                raise ValueError(f'initial state hash mismatch for task {task} init {init}')
            trial_dir=attempt/'trials'/f'task{task:02d}_init{init:02d}'
            all_jobs=[]
            for cp in replay['checkpoints']:
                point=f'at_{int(float(cp["fraction"])*100):02d}'
                branch_dir=trial_dir/'branches'/point
                jobs=candidate_jobs(cp,attempt,branch_dir)
                all_jobs.extend([(job,Path(cp['path']),trial_dir/'normal_replay.json',branch_dir)
                                 for job in jobs])
            def submit(index,value):
                job,cp_path,replay_path,branch_dir=value
                return execute_job(job,port=port,checkpoint_path=cp_path,replay_path=replay_path,
                    branch_dir=branch_dir,timeout=a.timeout,worker_index=index)
            with ThreadPoolExecutor(max_workers=a.workers) as pool:
                futures={pool.submit(submit,i,j):j for i,j in enumerate(all_jobs)}
                results=[]
                for future in as_completed(futures):
                    item=future.result()
                    item.update(task_id=task,trial_index=init)
                    results.append(item);records.append(item)
                    append_jsonl(a.output/'progress.jsonl',item)
            if len(results)!=132:
                raise AssertionError(f'trial expected 3×44=132 branches, got {len(results)}')
            append_jsonl(a.output/'progress.jsonl',{'event':'trial_finished','task_id':task,
                'trial_index':init,'capture':str(attempt),'complete':sum(x.get('status')=='complete' for x in results),
                'errors':sum(x.get('status')=='error' for x in results),
                'timeouts':sum(x.get('status')=='timeout' for x in results),'utc':time.time()})
        errors=sum(x.get('status')=='error' for x in records)
        timeouts=sum(x.get('status')=='timeout' for x in records)
        summary={'status':'complete' if not errors and not timeouts else 'incomplete',
            'trials_requested':len(trials),'branch_results_this_run':len(records),
            'complete':sum(x.get('status')=='complete' for x in records),
            'errors':errors,'timeouts':timeouts,
            'condition_spec':{'normal_alt_rng':8,'zero_t0':[.6,.2],'current_vlm_t0':[.6,.2],
                'four_keys':list(KEY_ORDER),'candidates_per_branchpoint':8,
                'candidate_forms':['codec_decode','D-raw_rows_actions'],'candidate_t0':[.4,.2]},
            'parallel_workers':a.workers,'server_port':port,'server_log':str(server_log),
            'note':'Counterfactual LIBERO simulation branches, not physical robot trials.'}
        summary_path=a.output/'summary.json'
        if summary_path.exists():
            raise FileExistsError(f'refusing to overwrite {summary_path}; use summary attempt paths')
        summary_path.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps(summary,ensure_ascii=False))
    finally:
        try:os.killpg(server.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:os.killpg(server.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            server.wait()
        try:lock.unlink()
        except FileNotFoundError:pass


if __name__=='__main__':main()
