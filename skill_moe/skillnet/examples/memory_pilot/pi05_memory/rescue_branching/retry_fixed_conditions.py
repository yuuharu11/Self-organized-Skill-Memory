"""Retry the four fixed-t0 smoke jobs after a request-field bug fix."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from evaluate_expanded_smoke import ROOT, RUN, CHECKPOINT, env_for, free_port, run_job, wait_ready


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=ROOT/'expanded-smoke-05'/'retry-fixed-t0')
    p.add_argument('--port',type=int)
    p.add_argument('--xla-fraction',type=float,default=.12)
    p.add_argument('--torch-fraction',type=float,default=.18)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    port=a.port or free_port()
    server_log=a.output/'server.log'
    with server_log.open('w') as f:
        server_env=env_for(threads='4')
        server_env['XLA_PYTHON_CLIENT_MEM_FRACTION']=str(a.xla_fraction)
        server_env['BRANCH_TORCH_MEM_FRACTION']=str(a.torch_fraction)
        server=subprocess.Popen(['taskset','-c','8-11',sys.executable,str(HERE/'serve_extended.py'),
          '--run',str(RUN),'--checkpoint',str(CHECKPOINT),'--port',str(port)],
          stdout=f,stderr=subprocess.STDOUT,env=server_env,start_new_session=True)
    cp=ROOT/'resume-check-08'/'trials'/'task06_init00'/'branchpoints'/'at_60.npz'
    replay=ROOT/'resume-check-08'/'trials'/'task06_init00'/'normal_replay.json'
    jobs=[]
    for t0 in (.6,.2):
        jobs.append({'name':f'zero_t0_{t0:.1f}','condition':'zero_fixed','t0':t0})
        jobs.append({'name':f'current_vlm_t0_{t0:.1f}','condition':'current_fixed','t0':t0})
    try:
        wait_ready(server,port,server_log)
        results=[]
        for i,job in enumerate(jobs):
            result=run_job(job,port=port,cp_path=cp,replay_path=replay,
                           out_dir=a.output,index=i)
            results.append(result)
        if len(results)!=4 or any(x.get('status')!='complete' for x in results):
            raise RuntimeError('fixed-t0 retry did not complete all four jobs')
        summary={'status':'complete','requested_conditions':4,'completed':len(results),
            'branch_results':results,'server_port':port,'server_log':str(server_log)}
        target=a.output/'retry_results.json'
        target.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps({'completed':len(results),'output':str(target)},ensure_ascii=False))
    finally:
        try:os.killpg(server.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:os.killpg(server.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            server.wait()


if __name__=='__main__':main()
