"""Audit paired initial states, immutable prompts, tokens and real control budgets."""
import argparse
import json
from pathlib import Path

import numpy as np

from effect_worker import write
from rescue_protocol import noise_seed,read_rows


def audit(root):
    failures=[];checked=[];initials={}
    for phase in ("main","physical","confirmation"):
        for resultfile in sorted((root/phase).glob("*/result.json")):
            try:
                result=json.loads(resultfile.read_text());job=result["job"]
                start,end=result["start_step"],result["end_step"]
                assert 0<=start<=end<=800,"control budget"
                states=read_rows(resultfile.parent/"states.jsonl")
                queries=read_rows(resultfile.parent/"queries.jsonl") if (resultfile.parent/"queries.jsonl").exists() else []
                assert [s["step"] for s in states]==list(range(start+1,end+1)),"missing/extra executed control"
                assert sum(s["source"]=="teacher" for s in states)==result["teacher_steps"]<=20,"teacher budget"
                if phase!="confirmation":
                    replay=json.loads((resultfile.parent/"replay_audit.json").read_text())
                    assert replay["state_max_abs"]==0 and replay["images_equal"],"replay mismatch"
                    assert replay["step"]==start,"replay time"
                if job.get("prompts_file"):
                    frozen=json.loads((root/job["prompts_file"]).read_text())
                    assert frozen["matched_token_difference"]<=1,"unmatched prompt lengths"
                    assert all(not a["truncated"] and a["exact_encoding"] for a in frozen["audits"].values()),"initial tokens"
                    expected=frozen["prompts"][job["condition"]]
                else:
                    expected=result["full_prompt"]
                for q in queries:
                    assert q["seed"]==noise_seed(job["seed"],job["init"],q["step"]),"noise-time key"
                    assert not q["token_audit"]["truncated"] and q["token_audit"]["exact_encoding"],"actual tokens"
                    assert q["execution_limit"]<=job["execute_steps"] and q["step"]+q["execution_limit"]<=800,"queued execution budget"
                    assert np.isfinite(np.asarray(q["actions"])).all(),"nonfinite actions"
                    before_switch=phase=="confirmation" and (result["switch_step"] is None or q["step"]<result["switch_step"])
                    assert q["prompt"]==(result["full_prompt"] if before_switch else expected),"prompt changed after freezing"
                if phase=="main" and queries:
                    key=(job["case"],job["seed"])
                    current=(start,queries[0]["image_sha256"],queries[0]["state"],queries[0]["sim_state"])
                    if key in initials:
                        assert current==initials[key],"paired initial state differs"
                    else:
                        initials[key]=current
                checked.append(str(resultfile.parent.relative_to(root)))
            except Exception as exc:
                failures.append(dict(directory=str(resultfile.parent.relative_to(root)),error=str(exc)))
    payload=dict(passed=not failures,completed_episodes_checked=len(checked),paired_initial_keys=len(initials),failures=failures)
    (root/"analysis").mkdir(exist_ok=True)
    write(root/"analysis/data_audit.json",payload)
    return payload


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);a=p.parse_args()
    result=audit(a.root);print(json.dumps(result,indent=2));raise SystemExit(0 if result["passed"] else 1)
