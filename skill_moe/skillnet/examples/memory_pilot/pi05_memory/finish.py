"""Continue a running preparation/training job with calibration and evaluation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from run import HERE,write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    args=parser.parse_args()
    deadline=(args.output/'experiment.json').stat().st_mtime+24*3600
    while time.time()<deadline:
        status=json.loads((args.output/'status.json').read_text())
        if status['status']=='failed':
            write_json(args.output/'continuation.json',{'status':'upstream_failed','upstream':status});return
        if status['status']=='training_complete':break
        time.sleep(10)
    else:
        write_json(args.output/'continuation.json',{'status':'budget_exhausted_before_training_finished'});return
    if not (args.output/'gradient_smoke.json').exists():
        raise RuntimeError('gradient smoke has not passed')
    for stage,marker in [('calibrate','calibration.json'),('evaluate','evaluation_status.json')]:
        if stage=='calibrate' and (args.output/marker).exists():continue
        write_json(args.output/'continuation.json',{'status':'running','stage':stage})
        with (args.output/(stage+'.log')).open('a') as stream:
            result=subprocess.run([sys.executable,str(HERE/(stage+'.py')),'--output',str(args.output),
                                   '--checkpoint',str(args.checkpoint)],stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            write_json(args.output/'continuation.json',{'status':'failed','stage':stage,'returncode':result.returncode});return
    write_json(args.output/'continuation.json',{'status':'finished','evaluation':json.loads((args.output/'evaluation_status.json').read_text())})


if __name__=='__main__':
    main()
