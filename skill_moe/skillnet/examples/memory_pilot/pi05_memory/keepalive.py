"""tmux-side supervisor: keep current work intact, recover a lost client job."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run import HERE,write_json


def existing_jobs(output):
    jobs=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid():
            continue
        try:
            args=(entry/'cmdline').read_bytes().decode().split('\0')
        except (OSError,UnicodeError):
            continue
        if str(output) not in args:
            continue
        if any(arg.endswith(('pi05_memory/run.py','pi05_memory/finish.py')) for arg in args):
            jobs.append(int(entry.name))
    return jobs


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    args=parser.parse_args()
    restarts=0
    while True:
        jobs=existing_jobs(args.output)
        if jobs:
            write_json(args.output/'detached_supervisor.json',{'status':'watching_existing_jobs',
                       'pid':os.getpid(),'jobs':jobs,'unix_time':time.time()})
            time.sleep(10)
            continue
        continuation=args.output/'continuation.json'
        if continuation.exists() and json.loads(continuation.read_text()).get('status')=='finished':
            write_json(args.output/'detached_supervisor.json',{'status':'finished','pid':os.getpid()})
            return
        status=json.loads((args.output/'status.json').read_text())
        if status.get('status')=='failed' and status.get('returncode',1)>0:
            write_json(args.output/'detached_supervisor.json',{'status':'upstream_error_needs_inspection','upstream':status})
            return
        restarts+=1
        if restarts>3:
            write_json(args.output/'detached_supervisor.json',{'status':'restart_limit_reached'})
            return
        if not (args.output/'trained.json').exists():
            command=[sys.executable,str(HERE/'run.py'),'pipeline','--output',str(args.output),
                     '--checkpoint',str(args.checkpoint)]
        else:
            write_json(args.output/'status.json',{'status':'training_complete','recovered_by_tmux':True})
            command=[sys.executable,str(HERE/'finish.py'),'--output',str(args.output),
                     '--checkpoint',str(args.checkpoint)]
        write_json(args.output/'detached_supervisor.json',{'status':'recovering_client_disconnect',
                   'attempt':restarts,'command':command})
        with (args.output/'detached_recovery.log').open('a') as stream:
            completed=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
        if completed.returncode:
            write_json(args.output/'detached_supervisor.json',{'status':'recovery_failed','returncode':completed.returncode})
            return


if __name__=='__main__':
    main()
