"""One killable rollout for skill_expert/evaluate.py."""
from __future__ import annotations
import json, os, sys
from pathlib import Path

def main():
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','2')))
    anchor=Path(__file__).resolve().parents[2]/'anchor_flow'
    sys.path.insert(0,str(anchor.resolve()))
    from run_case import _run_trial
    job=json.loads(Path(sys.argv[1]).read_text())
    result=_run_trial(job)
    Path(sys.argv[2]).write_text(json.dumps(result,ensure_ascii=False)+'\n')
    if result.get('status')!='complete': raise SystemExit(2)
if __name__=='__main__': main()
