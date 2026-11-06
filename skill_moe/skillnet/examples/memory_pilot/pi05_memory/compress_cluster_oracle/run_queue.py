#!/usr/bin/env python3
"""GPU-aware train/evaluate queue; completed combinations are never rerun."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
EXPERIMENT=HERE/'experiment.py'
REPO_ROOT=Path(__file__).resolve().parents[6]
STATE_DIR=Path(os.environ.get('SKILLNET_STATE_DIR',REPO_ROOT/'.skillnet-state')).expanduser()
PI05_ROOT=Path(os.environ.get('SKILLNET_PI05_ROOT',STATE_DIR/'outputs/memory-pilot/pi05-memory40-20261104')).expanduser()
OUT=Path(os.environ.get('COMPRESS_CLUSTER_OUT',PI05_ROOT/'compress-cluster-oracle-01')).expanduser()
BASE=[('raw',x) for x in ('none','mid','coarse','m6default')]
BASE += [('b12',x) for x in ('none','mid','coarse','m6default')]
BASE += [('b48',x) for x in ('none','mid','coarse','m6default')]
BASE += [('b24',x) for x in ('none','coarse')]
BASE += [('b96',x) for x in ('none','coarse')]
EXTRA=[('raw','fine'),('b12','fine'),('b24','mid'),('b96','mid')]


def gpu_status():
    try:
        out=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=8)
        rows=[]
        for line in out.splitlines():
            a,b=[int(x.strip()) for x in line.split(',')[:2]]; rows.append((a,b))
        return rows[0] if rows else (0,100)
    except Exception:
        return 0,100


def write_event(event,**extra):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'queue_progress.jsonl').open('a') as f:
        f.write(json.dumps({'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'event':event,**extra})+'\n')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--include-extra',action='store_true')
    ap.add_argument('--poll-seconds',type=int,default=15)
    args=ap.parse_args()
    if not (OUT/'pre_generation.json').exists():
        raise FileNotFoundError('complete the 5x5 CPU pre-generation table before training')
    # Keep the requested cell list unique.  b48/mid is already part of BASE;
    # listing it again as an extra used to inflate the queue's requested and
    # completed counts even though the second occurrence was skipped.
    jobs=list(dict.fromkeys(BASE+(EXTRA if args.include_extra else [])))
    missing=[f'{kind}_{level}_{split}' for kind,level in jobs for split in ('train','dev')
             if not (OUT/f'drafts_{kind}_{level}_{split}.npz').exists()]
    if missing:
        raise FileNotFoundError(f'pre-generation metadata exists, but draft caches are still being written: {missing}')
    pending=[]; retries={}; resumable_attempt={}; failures=[]
    for kind,level in jobs:
        root=OUT/'experts'/f'{kind}_{level}'
        if list(root.glob('attempt-*/evaluation.json')):
            write_event('skip_completed',representation=kind,level=level)
        else:
            attempts=sorted(root.glob('attempt-*')) if root.exists() else []
            trained=[p for p in attempts if (p/'trained.json').exists()]
            if trained:
                n=max(int(p.name.split('-')[-1]) for p in trained)
                resumable_attempt[(kind,level)]=n
                retries[(kind,level)]=1 if n>=2 else 0
            elif attempts:
                n=max(int(p.name.split('-')[-1]) for p in attempts)
                if n>=4:
                    failures.append({'representation':kind,'level':level,'reason':f'incomplete attempt-{n:02d}; retry budget exhausted'})
                    continue
                retries[(kind,level)]=1
                resumable_attempt[(kind,level)]=n+1
            pending.append((kind,level))
    active={}; limit=1; throttled=False; throttle_retry_key=None; cohort_fraction='.35'
    while pending or active:
        # Reap workers before launching replacement work.
        for p,meta in list(active.items()):
            rc=p.poll()
            if rc is None: continue
            log=meta['log'].name
            meta['log'].close(); del active[p]
            if rc==0:
                write_event('complete',representation=meta['kind'],level=meta['level'],attempt=meta['attempt'],log=log)
                if meta['attempt']==2 and (meta['kind'],meta['level'])==throttle_retry_key:
                    throttled=False; throttle_retry_key=None
            else:
                key=(meta['kind'],meta['level']); count=retries.get(key,0)
                oom=False
                try: oom=any(s in meta['path'].read_text(errors='replace').lower() for s in ('resource_exhausted','out of memory','cuda error: out of memory'))
                except Exception: pass
                write_event('failed',representation=key[0],level=key[1],attempt=meta['attempt'],returncode=rc,oom=oom,log=log)
                if count<1:
                    retries[key]=count+1; pending.insert(0,key)
                    limit=max(1,limit-1); throttled=True; throttle_retry_key=key; time.sleep(30)
                else:
                    failures.append({'representation':key[0],'level':key[1],'rc':rc,'log':log})
        free,util=gpu_status()
        if free<30_000 or util>=97:
            write_event('gpu_wait',free_mib=free,utilization=util,active=len(active),queued=len(pending))
            time.sleep(max(args.poll_seconds,30)); continue
        if not active:
            # A single .35 process still hit the allocator cap during training;
            # use one .50 process with the 143GB device currently available.
            cohort_fraction='.50'
            limit=1
        elif throttled:
            limit=1
            cohort_fraction='.50'
        while pending and len(active)<limit:
            free,util=gpu_status()
            if free<30_000 or util>=97: break
            kind,level=pending.pop(0)
            attempt=resumable_attempt.get((kind,level),2 if (kind,level) in retries else 1)
            expdir=OUT/'experts'/f'{kind}_{level}'/f'attempt-{attempt:02d}'
            if (expdir/'evaluation.json').exists():
                write_event('skip_completed',representation=kind,level=level,attempt=attempt); continue
            fraction=cohort_fraction or '.50'
            used_slots={m['slot'] for m in active.values()}
            slot=next(i for i in range(5) if i not in used_slots)
            cpus=list(range(8+slot*8,8+(slot+1)*8))
            logfile=OUT/'logs'/f'{kind}_{level}_attempt{attempt}.log'; logfile.parent.mkdir(parents=True,exist_ok=True)
            lf=logfile.open('w')
            cmd=(f"cd /work && source env/activate.sh >/dev/null && "
                 f"export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 "
                 f"XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION={fraction} "
                 f"&& taskset -c {','.join(map(str,cpus))} nice -n 10 python {EXPERIMENT} train --representation {kind} --level {level} --threads 8 --attempt {attempt} "
                 f"&& taskset -c {','.join(map(str,cpus))} nice -n 10 python {EXPERIMENT} evaluate --representation {kind} --level {level} --threads 8 --batch 8 --attempt {attempt}")
            proc=subprocess.Popen(['bash','-lc',cmd],stdin=subprocess.DEVNULL,stdout=lf,stderr=subprocess.STDOUT,
                                  start_new_session=True)
            active[proc]={'kind':kind,'level':level,'attempt':attempt,'log':lf,'path':logfile,'slot':slot}
            write_event('start',representation=kind,level=level,attempt=attempt,worker_pid=proc.pid,
                        mem_fraction=fraction,cpus=cpus,gpu_free_mib=free,gpu_util=util,active=len(active))
        time.sleep(args.poll_seconds)
    if not failures:
        # Use one controlled CPU process for final aggregation and 5k episode bootstrap.
        cmd=(f"cd /work && source env/activate.sh >/dev/null && export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 "
             f"&& taskset -c 8-15 nice -n 10 python {EXPERIMENT} finalize --threads 8")
        subprocess.run(['bash','-lc',cmd],check=True)
    write_event('queue_finished',failures=failures,requested=len(jobs),completed=len(jobs)-len(failures))
    if failures: raise SystemExit(1)


if __name__=='__main__': main()
