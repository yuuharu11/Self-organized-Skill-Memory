#!/usr/bin/env python3
"""Run one independent Dev grid point."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from key_shared_latent import run_experiment


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--mode',required=True)
    ap.add_argument('--learning-rate',type=float,required=True)
    ap.add_argument('--epochs',type=int,required=True)
    ap.add_argument('--teacher-temperature',type=float,default=.005)
    ap.add_argument('--objective',choices=('soft','hard'),default='soft')
    ap.add_argument('--vjepa-eval-cache',type=Path,required=True)
    args=ap.parse_args()
    common={'seed':1,'steps_per_epoch':None,'dev_only':True,'vjepa_eval_cache':args.vjepa_eval_cache}
    result=run_experiment(args.output,mode=args.mode,learning_rate=args.learning_rate,epochs=args.epochs,
                        objective=args.objective,teacher_temperature=args.teacher_temperature,**common)
    print(json.dumps({'mode':args.mode,'objective':args.objective,'status':result['status'],
                      'teacher_temperature':args.teacher_temperature,'output':str(args.output)}))


if __name__=='__main__':main()
