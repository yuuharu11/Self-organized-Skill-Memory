"""Bounded distillation evaluation; owns only PIDs it launches or records."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from distill_report import summarize
from distill_train import write


def terminate(process):
    if process.poll() is not None:return
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()


def run_job(root,manifest,job,processes):
    if time.time()>=manifest['launch_deadline']:return
    directory=root/job['directory']
    if (directory/'job.json').exists():return
    job=dict(job,port=manifest['ports'][job['condition']],hard_deadline=manifest['collect_deadline'])
    path=root/'jobs'/(directory.name+'.json');write(path,job)
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'worker.log').open('w') as log:
        process=subprocess.Popen(['bash',str(root/'run_cpu.sh'),str(root/'frozen/distill_worker.py'),'--root',str(root),'--job',str(path)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(process)
        while process.poll() is None:
            if time.time()>=manifest['collect_deadline'] or (root/'STOP_REQUESTED').exists():terminate(process);break
            time.sleep(1)
        print(json.dumps(dict(directory=job['directory'],returncode=process.returncode,complete=(directory/'result.json').exists())),flush=True)


def evaluate_model(root,manifest,condition,selection,processes):
    firsts={int(t):values[0] for t,values in selection.items()}
    # Probe-target parents first, but freeze the entire selection before results.
    order=sorted(firsts,key=lambda t:(t not in manifest['probe_tasks'],t))
    for task in order:
        for init in selection[str(task)]:
            case=f'task{task+1:02d}-init{init:02d}'
            run_job(root,manifest,dict(task=task,init=init,case=case,condition=condition,seed=manifest['evaluation_seed'],directory=f'evaluation/{case}-{condition}',save_probe=condition=='S0' and task in manifest['probe_tasks'] and init==firsts[task]),processes)
    # Each model waits for the same S0 snapshot; no own-policy snapshot selection.
    for task in manifest['probe_tasks']:
        init=firsts[task];case=f'task{task+1:02d}-init{init:02d}'
        snapshot=root/'probe_snapshots'/(case+'.npz')
        while not snapshot.exists() and time.time()<manifest['launch_deadline']:
            parent=root/'evaluation'/(case+'-S0')
            if (parent/'result.json').exists() or (parent/'incomplete.json').exists():break
            time.sleep(2)
        if not snapshot.exists():
            write(root/'probe_missing'/f'{case}-{condition}.json',dict(case=case,reason='S0 snapshot unavailable; not replaced',count_as_failure=False));continue
        for seed in manifest['probe_seeds']:
            run_job(root,manifest,dict(task=task,init=init,case=case,condition=condition,seed=seed,snapshot=str(snapshot.relative_to(root)),directory=f'probes/{case}-seed{seed}-{condition}'),processes)


def main(root):
    manifest=json.loads((root/'manifest.json').read_text());processes=[];servers=[]
    try:
        while not (root/'TRAIN_READY').exists():
            if (root/'TRAIN_ERROR.json').exists():raise RuntimeError('training failed; do not compare unequal budgets')
            if time.time()>=manifest['train_deadline']:
                pid=json.loads((root/'train_pid.json').read_text())['pid']
                try:os.killpg(pid,signal.SIGTERM)
                except ProcessLookupError:pass
                raise TimeoutError('three-hour training deadline')
            summarize(root)
            time.sleep(5)
        budget=json.loads((root/'training_budget.json').read_text())
        trained=[json.loads((root/'models'/c/'result.json').read_text()) for c in ('M0','M1')]
        if any(r['updates']!=budget['updates'] or r['frozen_before']!=r['frozen_after'] for r in trained):raise ValueError('unequal training budgets or frozen-weight mutation')
        while not (root/'TESTS_READY').exists():
            if time.time()>=manifest['launch_deadline']:raise TimeoutError('validation did not complete')
            time.sleep(2)
        from rescue_followup import used_initial_states
        if (root/'EVALUATION_SELECTION.json').exists():
            selection=json.loads((root/'EVALUATION_SELECTION.json').read_text())
        else:
            used=used_initial_states(root,list(range(9)))
            selection={str(t):[i for i in (30,31,*range(32,50)) if i not in used[t]][:2] for t in range(9)}
        if any(len(v)!=2 for v in selection.values()):raise ValueError('insufficient unused evaluation families')
        write(root/'EVALUATION_SELECTION.json',selection)
        (root/'jobs').mkdir(exist_ok=True)
        for condition,port in manifest['ports'].items():
            cp=manifest['base_checkpoint'] if condition=='S0' else str(root/'models'/condition/'params')
            audit=root/'training_budget.json' if condition=='S0' else root/'models'/condition/'result.json'
            log=(root/(condition+'_server.log')).open('w')
            proc=subprocess.Popen(['bash',str(root/'run_server.sh'),str(root/'frozen/effect_server.py'),'--port',str(port),'--checkpoint',cp,'--weight-audit-file',str(audit)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            servers.append(proc)
        while True:
            ready=[]
            for port in manifest['ports'].values():
                with socket.socket() as sock:ready.append(sock.connect_ex(('127.0.0.1',port))==0)
            if all(ready):break
            if any(p.poll() is not None for p in servers):raise RuntimeError('evaluation server failed')
            if time.time()>=manifest['launch_deadline']:raise TimeoutError('servers not ready before evaluation cutoff')
            time.sleep(3)
        write(root/'status.json',dict(phase='evaluation',selection=selection))
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(evaluate_model,root,manifest,c,selection,processes) for c in manifest['ports']]
            while not all(f.done() for f in futures):
                summarize(root);time.sleep(15)
            for f in futures:f.result()
        write(root/'PIPELINE_FINISHED.json',dict(epoch=time.time()))
    except Exception as exc:write(root/'PIPELINE_ERROR.json',dict(error=repr(exc),epoch=time.time()));raise
    finally:
        for p in processes+servers:terminate(p)
        summarize(root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args();main(a.root)
