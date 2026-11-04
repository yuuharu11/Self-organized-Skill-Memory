"""Minute-by-minute read-only health checks, independent of the worker chain."""
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import subprocess
import time

from run import write_json


def read_json(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}


def tail_events(path):
    if not path.exists():return []
    with path.open('rb') as stream:
        stream.seek(max(0,path.stat().st_size-32768))
        lines=stream.read().decode(errors='replace').splitlines()
    result=[]
    for line in lines:
        try:
            value=json.loads(line)
            if isinstance(value,dict):result.append(value)
        except ValueError:pass
    return result


def inspect(output):
    status=read_json(output/'status.json')
    continuation=read_json(output/'continuation.json')
    evaluation=read_json(output/'evaluation_status.json')
    stage=continuation.get('stage') if continuation.get('status')=='running' else status.get('stage')
    stage=stage or ('evaluate' if evaluation else status.get('status','unknown'))
    log_name='train' if stage=='train' else stage
    log=output/(str(log_name)+'.log')
    events=tail_events(log)
    latest=events[-1] if events else {}
    now=time.time()
    heartbeat=max([p.stat().st_mtime for p in [log,output/'status.json',output/'continuation.json',
                 output/'evaluation_status.json',output/'training_progress.json'] if p.exists()],default=now)
    alerts=[]
    if status.get('status')=='failed' or continuation.get('status') in ['failed','upstream_failed']:
        alerts.append('worker_failed')
    if evaluation.get('status')=='infrastructure_error':alerts.append('evaluation_infrastructure_error')
    if now-heartbeat>900 and status.get('status')=='running':alerts.append('no_progress_for_15_minutes')
    if 'loss' in latest and not math.isfinite(latest['loss']):alerts.append('nonfinite_training_loss')
    if 'seconds_per_query' in latest and latest['seconds_per_query']>2:
        alerts.append('retrieval_slower_than_two_seconds_per_query')
    gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],
                       capture_output=True,text=True,timeout=15)
    if gpu.returncode:alerts.append('gpu_query_failed')
    return {'utc':datetime.now(timezone.utc).isoformat(),'stage':stage,'worker':status,
            'continuation':continuation,'evaluation':evaluation,'latest_event':latest,
            'heartbeat_age_seconds':max(0.,now-heartbeat),'gpu_memory_mib_utilization':gpu.stdout.strip(),
            'alerts':alerts,'policy':'observe only; never tune on evaluation outcomes or terminate unrelated jobs'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    while True:
        report=inspect(args.output)
        write_json(args.output/'health.json',report)
        with (args.output/'health.jsonl').open('a') as stream:stream.write(json.dumps(report)+'\n')
        if args.once:
            print(json.dumps(report));return
        if report['continuation'].get('status')=='finished':return
        time.sleep(60)


if __name__=='__main__':
    main()
