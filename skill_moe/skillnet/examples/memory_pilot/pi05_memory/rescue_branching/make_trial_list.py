"""Select up to ten run-01 normal failures for exp's branching run."""
from __future__ import annotations
import argparse,json
from pathlib import Path

import os as _os
_REPO = Path(__file__).resolve().parents[6]
_STATE_DIR = Path(_os.environ.get('SKILLNET_STATE_DIR', _REPO / '.skillnet-state'))
_PI05_ROOT = Path(_os.environ.get('SKILLNET_PI05_ROOT', _STATE_DIR / 'outputs/memory-pilot/pi05-memory40-20261104'))
ROOT=_PI05_ROOT

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=ROOT/'paired-breakdown-01'/'normal_failures.jsonl')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-trials',type=int,default=10)
    a=p.parse_args()
    rows=[json.loads(x) for x in a.source.read_text().splitlines() if x.strip()]
    candidates=[r for r in rows if r.get('task_suite')=='libero_10' and
        int(r['task_id']) in (6,8) and r.get('normal_failure',{}).get('status')=='complete' and
        r.get('normal_failure',{}).get('termination_reason')=='control_budget']
    groups={task:sorted([r for r in candidates if int(r['task_id'])==task],
        key=lambda r:int(r['initial_state_index'])) for task in (6,8)}
    balanced=[]
    for offset in range(max((len(v) for v in groups.values()),default=0)):
        for task in (6,8):
            if offset<len(groups[task]):balanced.append(groups[task][offset])
    out=[{'task_id':int(r['task_id']),'trial_index':int(r['initial_state_index']),'seed_base':7,
          'expected_trial_seed':int(r['trial_seed']),'expected_initial_state_sha256':r['initial_state_sha256']}
         for r in balanced[:a.max_trials]]
    if not out:raise SystemExit('no eligible run-01 normal failures')
    if a.output.exists():raise SystemExit(f'refusing to overwrite existing trial list: {a.output}')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'selected':len(out),'output':str(a.output),'trials':out},indent=2))

if __name__=='__main__':main()
