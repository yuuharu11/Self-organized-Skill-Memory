#!/usr/bin/env python3
"""Summarize cross-task membership for selected Dev-frozen M6 memories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import run as shape
from data import HELDOUT_TASKS, LONG_TASKS
from representations import m6_fit_generic


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True)
    args=ap.parse_args();out=args.out
    episodes,rows,_=shape.load_source(args.run_dir,args.checkpoint)
    segsets,_,_=shape.make_segment_sets(episodes,rows,out/'share_audit_segments.jsonl')
    selected=json.loads((out/'selected_configs.json').read_text())['selected']
    # Cover the mandatory controls and selected grain sizes; duplicate configs
    # at other grain levels do not change task-sharing for the same M6 setup.
    chosen=[];seen=set()
    for x in selected:
        key=(x['config'],float(x['kappa']),int(x['min_support']),int(x['min_episodes']))
        if key not in seen:chosen.append(x);seen.add(key)
    result=[]
    for c in chosen:
        meta=json.loads((out/'representations'/c['config']/'manifest.json').read_text())
        sname=meta['segmentation']
        mem=[s for s in segsets[sname] if s['meta'].get('present',False)
             and s['meta'].get('memory',False) and s['task'] not in LONG_TASKS+HELDOUT_TASKS]
        assert mem and not any(s['task'] in LONG_TASKS+HELDOUT_TASKS for s in mem)
        _,codes,_,loaded,_=shape.load_eval_bundle(out,c['config'],segsets[sname],episodes)
        assert len(mem)==len(loaded)
        t=np.asarray([s['task'] for s in loaded],np.int32)
        e=np.asarray([s['episode'] for s in loaded],np.int32)
        m=m6_fit_generic(codes,t,e,kappa=float(c['kappa']),seed=20261104,
                         min_support=int(c['min_support']),min_episodes=int(c['min_episodes']))
        per_unit=np.asarray([len(set(t[ix].tolist())) for ix in m['members_local']],np.int32)
        result.append({'config':c['config'],'target_units':c['target_units'],
            'units':int(m['registered']),'kappa':float(c['kappa']),
            'support':[int(c['min_support']),int(c['min_episodes'])],
            'unit_task_count_quantiles':np.quantile(per_unit,[.1,.5,.9]).tolist(),
            'multitask_unit_fraction':float(np.mean(per_unit>1)),
            'member_fraction_in_multitask_units':float(
                sum(len(ix) for ix,nt in zip(m['members_local'],per_unit,strict=True) if nt>1)
                / max(1,sum(map(len,m['members_local'])))),
            'memory_segments':len(mem),'memory_tasks':sorted(set(t.tolist()))})
    path=out/'unit_sharing.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
