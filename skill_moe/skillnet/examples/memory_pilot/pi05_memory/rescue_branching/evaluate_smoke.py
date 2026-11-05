"""Start one policy server, capture a failure, and smoke saved-state branches."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

import numpy as np

HERE=Path(__file__).resolve().parent
PI05=HERE.parent
import os as _os
_REPO = Path(__file__).resolve().parents[6]
_STATE_DIR = Path(_os.environ.get('SKILLNET_STATE_DIR', _REPO / '.skillnet-state'))
_PI05_ROOT = Path(_os.environ.get('SKILLNET_PI05_ROOT', _STATE_DIR / 'outputs/memory-pilot/pi05-memory40-20261104'))
RUN=_PI05_ROOT / 'run-01'
OUT=_PI05_ROOT / 'rescue-branching-01'
CHECKPOINT=_STATE_DIR / 'checkpoints/openpi-assets/checkpoints/pi05_libero'


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1',0));return s.getsockname()[1]


def wait_ready(proc,port,log):
    for _ in range(180):
        if proc.poll() is not None: raise RuntimeError(f'server exited; see {log}')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz',timeout=2) as r:
                if r.status==200:return
        except Exception:pass
        time.sleep(2)
    raise TimeoutError(f'server startup timeout; see {log}')


def env_for(threads='2'):
    e=os.environ.copy()
    e['OMP_NUM_THREADS']=e['MKL_NUM_THREADS']=e['OPENBLAS_NUM_THREADS']=threads
    e['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    e['XLA_PYTHON_CLIENT_MEM_FRACTION']='0.15'
    e['BRANCH_TORCH_MEM_FRACTION']='0.20'
    return e


def run_capture(port,task_id=6,trial_index=0,seed_base=7,fractions=(.2,.4,.6)):
    folder=OUT/'trials'/f'task{task_id:02d}_init{trial_index:02d}'
    folder.mkdir(parents=True,exist_ok=True)
    existing=folder/'normal_replay.json'
    if existing.exists():
        result=json.loads(existing.read_text())
        if result.get('status')!='complete':raise ValueError(f'invalid existing capture: {existing}')
        return result
    log=folder/'capture.log'
    with log.open('w') as f:
        p=subprocess.run(['taskset','-c','14-19','nice','-n','10',sys.executable,str(HERE/'rollout.py'),
            'capture','--port',str(port),'--task-id',str(task_id),'--trial-index',str(trial_index),
            '--seed-base',str(seed_base),'--output-dir',str(OUT),'--fractions',*[str(x) for x in fractions]],stdout=f,stderr=subprocess.STDOUT,env=env_for())
    if p.returncode:raise RuntimeError(f'normal replay failed; see {log}')
    result=json.loads((folder/'normal_replay.json').read_text())
    if result['success']:
        raise RuntimeError('chosen run-01 normal trial did not reproduce as a failure; no branch smoke was counted')
    if len(result['checkpoints'])!=3:raise RuntimeError('did not capture all 20/40/60% branchpoints')
    return result


def run_branch(job,port,folder,idx,timeout=600):
    folder.mkdir(parents=True,exist_ok=True)
    stem=f'branch_{idx:02d}_{job["name"]}'
    path=folder/f'{stem}.json'
    if path.exists():
        prior=json.loads(path.read_text())
        if prior.get('status')=='complete':
            prior.update(name=job['name'],reused=True,result_path=str(path))
            return prior
        attempt=2
        while (folder/f'{stem}_attempt{attempt}.json').exists():attempt+=1
        path=folder/f'{stem}_attempt{attempt}.json'
    log=path.with_suffix('.log')
    cmd=['taskset','-c',('14-19','20-25','26-31','32-37')[idx%4],
         'nice','-n','10',sys.executable,str(HERE/'rollout.py'),'branch',
         '--port',str(port),'--checkpoint-path',str(job['checkpoint']),
         '--condition',job['condition'],'--output',str(path)]
    if job.get('replay_path'):cmd += ['--replay-path',str(job['replay_path'])]
    if job.get('memory_row') is not None:cmd += ['--memory-row',str(job['memory_row']),'--t0',str(job['t0'])]
    if job.get('seed_offset'):cmd += ['--seed-offset',str(job['seed_offset'])]
    with log.open('w') as f:
        started=time.monotonic()
        try:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env_for(),timeout=timeout)
        except subprocess.TimeoutExpired:return {'name':job['name'],'status':'timeout','elapsed_seconds':timeout,'log':str(log)}
    if p.returncode or not path.exists():return {'name':job['name'],'status':'error','returncode':p.returncode,'elapsed_seconds':time.monotonic()-started,'log':str(log)}
    result=json.loads(path.read_text());result.update(name=job['name'],status='complete',log=str(log),result_path=str(path))
    result['elapsed_seconds']=time.monotonic()-started
    return result


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--branch-fraction',type=float,default=.6)
    parser.add_argument('--port',type=int)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--consistency-only',action='store_true')
    parser.add_argument('--capture-only',action='store_true')
    a=parser.parse_args()
    if not 1<=a.workers<=4:raise ValueError('smoke supports at most four simulator workers')
    OUT=a.output
    OUT.mkdir(parents=True,exist_ok=True)
    port=a.port or free_port()
    server_log=OUT/'server.log'
    with server_log.open('a') as log:
        proc=subprocess.Popen(['taskset','-c','8-13',sys.executable,str(HERE/'serve.py'),
          '--run',str(RUN),'--checkpoint',str(CHECKPOINT),'--port',str(port)],
          stdout=log,stderr=subprocess.STDOUT,env=env_for(),start_new_session=True)
    try:
        wait_ready(proc,port,server_log)
        replay=run_capture(port)
        failures=RUN.parent/'paired-breakdown-01'/'normal_failures.jsonl'
        paired=None
        if failures.exists():
            paired=next((json.loads(line) for line in failures.read_text().splitlines()
                if line.strip() and json.loads(line).get('task_suite')=='libero_10' and
                int(json.loads(line).get('task_id',-1))==int(replay['task_id']) and
                int(json.loads(line).get('initial_state_index',-1))==int(replay['trial_index'])),None)
            if paired is None or paired.get('normal_failure',{}).get('termination_reason')!='control_budget':
                raise ValueError('smoke source is not listed as a normal-policy control-budget failure')
            if paired.get('initial_state_sha256')!=replay['initial_state_sha256']:
                raise ValueError('replayed initial state does not match paired-breakdown source')
        if a.capture_only:
            print(json.dumps({'capture_only':True,'normal_steps':replay['env_steps'],
                'checkpoints':replay['checkpoints']},ensure_ascii=False));return
        cp=next(x for x in replay['checkpoints'] if abs(float(x['fraction'])-a.branch_fraction)<1e-8)
        branchpoint=Path(cp['path'])
        trial_key=f'task{int(replay["task_id"]):02d}_init{int(replay["trial_index"]):02d}'
        branch_folder=OUT/'trials'/trial_key/'branches'
        jobs=[{'name':'normal_same_rng','condition':'normal','replay_path':OUT/'trials'/trial_key/'normal_replay.json'}]
        if not a.consistency_only:
            jobs += [{'name':f'normal_alt_rng_{i+1}','condition':'normal_alt','seed_offset':1009*(i+1)} for i in range(8)]
            jobs += [{'name':'zero_draft','condition':'zero'}, {'name':'current_search','condition':'current'}]
            for row in cp['candidates']:
                for t0 in (.4,.2):jobs.append({'name':f'memory_{row}_t0_{t0:.1f}','condition':'memory_row','memory_row':int(row),'t0':t0})
        jobs=[dict(x,checkpoint=branchpoint,replay_path=OUT/'trials'/trial_key/'normal_replay.json') for x in jobs]
        results=[]
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures={pool.submit(run_branch,j,port,branch_folder,i):j for i,j in enumerate(jobs)}
            for f in as_completed(futures):results.append(f.result())
        results.sort(key=lambda x:x['name'])
        baseline_hashes={int(x['query_id']):x['sha256'] for x in replay['queries'] if int(x['query_id'])>=int(cp['query_id'])}
        same=next(x for x in results if x['name']=='normal_same_rng')
        branch_hashes={int(x['query_id']):x['sha256'] for x in same.get('query_hashes',[])}
        matches=[q for q,h in branch_hashes.items() if baseline_hashes.get(q)==h]
        obs_match=(same.get('branch_observation_sha256')==same.get('saved_observation_sha256'))
        consistency={'reference_branch_query_chunks':len(branch_hashes),'exact_action_chunk_matches':len(matches),
          'observation_exact_match':obs_match,
          'all_match':obs_match and bool(branch_hashes) and len(matches)==len(branch_hashes),
          'first_mismatch_query_id':next((q for q in sorted(branch_hashes) if baseline_hashes.get(q)!=branch_hashes[q]),None)}
        report={'trial':{'task_id':6,'trial_index':0,'seed':7,'normal_replay_success':replay['success'],
            'normal_replay_steps':replay['env_steps'],'selected_from':'run-01 normal failure',
            'paired_breakdown_match':paired is not None,
            'branchpoint':cp,'branchpoint_count':len(replay['checkpoints'])},
          'same_rng_resume':consistency,'branch_count':len(results),'branch_results':results,
          'parallel_workers':a.workers,'server_port':port,'server_log':str(server_log),
          'smoke_only':True}
        path=OUT/'smoke_results.json'
        if path.exists():path=OUT/f'smoke_results_{int(time.time())}.json'
        path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
        completed=[x for x in results if x.get('status')=='complete']
        report_md=['# Rescue branching smoke','',
          '閉ループシミュレーションの単一失敗試行での動作確認。物理ロボットの評価ではない。','',
          '| 確認 | 結果 |','|---|---|',
          f"| 元の通常方策 | task {replay['task_id']} / init {replay['trial_index']}, seed {replay['trial_seed']}, success={replay['success']}, {replay['env_steps']} steps |",
          f"| paired-breakdownとの初期状態一致 | {paired is not None}; sha256={replay['initial_state_sha256']} |",
          f"| 状態保存点 | {', '.join(str(x['env_steps'])+' steps' for x in replay['checkpoints'])} |",
          f"| 同乱数の再開 | {len(matches)}/{len(branch_hashes)} action chunks 完全一致 |",
          f"| 分岐条件 | {len(completed)}/{len(results)} complete; {sum(x.get('status')=='error' for x in results)} errors; {sum(x.get('status')=='timeout' for x in results)} timeouts |",
          f"| 1枝の時間 | mean {np.mean([x['elapsed_seconds'] for x in completed]):.1f}s; max {max([x['elapsed_seconds'] for x in completed],default=0):.1f}s |",
          '',f"詳細: `{path}` と各枝の JSON/ログ。GPU server log: `{server_log}`.",
          '本番の概算は各枝の平均実測時間 × 780 ÷ 4 に、10件の通常リプレイ時間を加えて算出する。']
        report_path=OUT/'report.md'
        if report_path.exists():report_path=OUT/f'report_{int(time.time())}.md'
        report_path.write_text('\n'.join(report_md)+'\n')
        print(json.dumps({'output':str(path),'normal_steps':replay['env_steps'],'branch_count':len(results),
                          'consistency':consistency},ensure_ascii=False))
    finally:
        try:os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            proc.wait()


if __name__=='__main__':main()
