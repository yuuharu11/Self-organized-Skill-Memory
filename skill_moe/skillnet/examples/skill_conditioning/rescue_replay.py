"""Replay existing original-policy actions; never requests policy inference."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from collect_interventions import append,image_hash,policy_input
from effect_worker import setup,write
from rescue_protocol import MAIN_CASES,case_ids,group_name,read_rows,window_phenomena
from rescue_runtime import Metrics,last_contact_reference,ordered_task,save_snapshot


def replay(root):
    manifest=json.loads((root/"manifest.json").read_text())
    source=Path(manifest["source_root"])
    c,stages=setup(root)
    cases={}
    for p in sorted((source/"evaluation").glob("*-original/result.json")):
        original=json.loads(p.read_text())
        name=p.parent.name.removesuffix("-original")
        oracle=json.loads((source/"evaluation"/(name+"-oracle")/"result.json").read_text())
        cases[name]=dict(original_success=original["success"],oracle_success=oracle["success"],end_step=original["end_step"],
                         group=group_name(original["success"],oracle["success"]))
    order=list(MAIN_CASES)+[name for name,meta in cases.items() if name not in MAIN_CASES and (not meta["original_success"] or meta["group"]=="harm")]
    for name in order:
        if time.time()>=manifest["started_epoch"]+3600:
            write(root/"REPLAY_TIME_LIMIT.json",dict(next_case=name));break
        out=root/"replay"/name;out.mkdir(parents=True,exist_ok=True)
        if (out/"result.json").exists():
            continue
        task,init=case_ids(name)
        recorded=read_rows(source/"evaluation"/(name+"-original")/"queries.jsonl")
        env,obs,full,annotation=c.setup(task,init)
        goals,residual=ordered_task(env,stages,task,annotation)
        metrics=Metrics(env.env,goals)
        rows=[];prefix=[];run=0;matched=True;checks=[];snapshot=None;physical={}
        try:
            for query in recorded:
                step=len(prefix)
                raw=policy_input(obs,full,annotation,init=init,query=step,evaluator=c.evaluator)
                ok=image_hash(raw)==query["image_sha256"] and np.array_equal(raw["observation/state"],query["state"])
                checks.append(dict(step=step,images_equal=image_hash(raw)==query["image_sha256"],state_equal=bool(np.array_equal(raw["observation/state"],query["state"]))))
                matched=matched and ok
                if snapshot is None and step>=50 and run>=5:
                    snapshot=str((out/"achievement.npz").relative_to(root))
                    save_snapshot(root/snapshot,env,raw,prefix,step)
                if step>=100 and len(rows)>=50:
                    labels,evidence=window_phenomena(rows)
                    reference=last_contact_reference(rows)
                    for label in ("contact_stall","contact_missing","oscillation_or_reversal"):
                        if label in labels and label not in physical:
                            path=out/(label+".npz")
                            save_snapshot(path,env,raw,prefix,step)
                            physical[label]=dict(snapshot=str(path.relative_to(root)),step=step,
                                                 reference=reference,evidence=evidence)
                for action in query["actions"][:cases[name]["end_step"]-len(prefix)]:
                    obs,_,_,_=env.step(action)
                    prefix.append(action)
                    row=metrics.observe(obs,len(prefix),action)
                    run=run+1 if row["truth"]["verified"][0] else 0
                    rows.append(row)
                    append(out/"trace.jsonl",row)
            labels=set();evidence=[]
            for end in range(50,len(rows)+1,5):
                observed,numbers=window_phenomena(rows[:end])
                if observed:
                    labels.update(observed)
                    evidence.append(dict(step=end,labels=observed,measurements=numbers))
            for goal in range(2):
                achieved=False
                for row in rows:
                    achieved=achieved or row["truth"]["raw_goals"][goal]
                    if achieved and not row["truth"]["raw_goals"][goal]:
                        labels.add("achieved_goal_regression")
            if rows and rows[-1]["normalized_goal_distance"] is not None and rows[-50]["normalized_goal_distance"] is not None:
                delta=rows[-50]["normalized_goal_distance"]-rows[-1]["normalized_goal_distance"]
                if rows[-50]["target"]==rows[-1]["target"] and delta>.01:
                    labels.add("terminal_progress")
                    evidence.append(dict(step=len(rows),terminal_50step_normalized_goal_distance_reduction=delta))
            if not labels:
                labels.add("unresolved")
            result=dict(case=name,**cases[name],status="complete",replay_matches=bool(matched),
                        checks=len(checks),steps=len(rows),labels=sorted(labels),
                        achievement_snapshot=snapshot if matched else None,
                        physical_candidates=physical if matched else {},full=full,remaining=residual,
                        final_truth=rows[-1]["truth"],evidence=evidence)
            write(out/"checks.json",checks);write(out/"result.json",result)
            print(json.dumps({key:result[key] for key in ("case","replay_matches","labels","achievement_snapshot")}),flush=True)
        finally:
            env.close()
    write(root/"replay_cases.json",cases)
    (root/"REPLAY_FINISHED").write_text("No policy inference used.\n")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    replay(p.parse_args().root)
