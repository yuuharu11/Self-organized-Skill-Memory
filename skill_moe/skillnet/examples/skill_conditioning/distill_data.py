"""Replay successful rollouts into audited, original-instruction BC samples."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from collect_interventions import image_hash,policy_input
from effect_worker import setup,truth,write
from rescue_protocol import action_prefix,case_ids,read_rows
from rescue_runtime import ordered_task


TEACHERS=(("task02-init21",89),("task03-init20",73),("task03-init20",89),("task07-init21",89))


def executed_actions(queries,start,end):
    actions=[]
    for query in queries:
        if query["step"]!=start+len(actions):
            raise ValueError("noncontiguous executed actions")
        actions.extend(query["actions"][:end-start-len(actions)])
    if len(actions)!=end-start:
        raise ValueError("executed action coverage differs from result end step")
    return np.asarray(actions,np.float64)


def normalized_transforms(checkpoint):
    from openpi import transforms
    from openpi.training import checkpoints,config_moe_skill
    from s2_policy import integer_object_ids
    from serve_s2_policy import _transforms_for_server
    config=config_moe_skill.get_config("pi05_libero_moe_skill_4_90")
    data=config.data.create(config.assets_dirs,config.model)
    stats=checkpoints.load_norm_stats(Path(checkpoint).parent/"assets",data.asset_id)
    input_transform=transforms.compose(_transforms_for_server(data,stats,inputs=True))
    reverse=transforms.Unnormalize(stats,use_quantiles=data.use_quantile_norm)
    return lambda raw:integer_object_ids(input_transform(raw)),reverse


def specs_for(root):
    manifest=json.loads((root/"manifest.json").read_text())
    source=Path(manifest["source_root"]);prior=Path(manifest["prior_root"])
    specs=[]
    for case,seed in TEACHERS:
        path=source/"main"/f"{case}-seed{seed}-C"
        row=json.loads((path/"result.json").read_text())
        base=json.loads((source/"main"/f"{case}-seed{seed}-A/result.json").read_text())
        if not row["success"] or base["success"]:
            raise ValueError("fixed teacher is not an A-fail/C-success pair")
        specs.append(dict(directory=str(path),kind="recovery",case=case,source_root=str(source),
                          id=f"recovery-{case}-{seed}"))
    for path in sorted((prior/"evaluation").glob("*-original/result.json")):
        row=json.loads(path.read_text())
        if row["success"]:
            case=path.parent.name.removesuffix("-original")
            specs.append(dict(directory=str(path.parent),kind="normal",case=case,source_root=str(prior),id="normal-full-"+case))
    for path in sorted((source/"main").glob("*-A/result.json")):
        row=json.loads(path.read_text())
        if row["success"]:
            case=row["job"]["case"]
            specs.append(dict(directory=str(path.parent),kind="normal",case=case,source_root=str(source),id="normal-branch-"+path.parent.name))
    return specs


def export(root):
    manifest=json.loads((root/"manifest.json").read_text())
    transform,reverse=normalized_transforms(manifest["base_checkpoint"])
    c,stages=setup(root)
    out=root/"data";out.mkdir(exist_ok=True)
    reports=[];bad_parents=set();seen=set()
    for spec in specs_for(root):
        if time.time()>=manifest["train_deadline"]:
            raise TimeoutError("data preparation consumed the training budget")
        task,init=case_ids(spec["case"])
        if init in (30,31):
            raise ValueError("reserved evaluation family in training")
        directory=Path(spec["directory"]);result=json.loads((directory/"result.json").read_text())
        queries=read_rows(directory/"queries.jsonl")
        start,end=result["start_step"],result["end_step"]
        actions=executed_actions(queries,start,end)
        env,obs,full,annotation=c.setup(task,init)
        goals,_=ordered_task(env,stages,task,annotation)
        sample={key:[] for key in ("image_base","image_left_wrist","state","actions","tokenized_prompt","tokenized_prompt_mask","skills","skill_mask","objects","step")}
        parent=spec["case"];checks=0;dropped=0;duplicates=0;roundtrip=0.
        try:
            if start:
                snapshot=Path(spec["source_root"])/result["job"]["snapshot"]
                prefix=np.load(snapshot,allow_pickle=False)["prefix"]
                if len(prefix)!=start:
                    raise ValueError("snapshot prefix length")
                for action in prefix:
                    obs,_,_,_=env.step(action.tolist());truth(env.env,goals);env.check_success()
                if not np.array_equal(env.get_sim_state(),np.load(snapshot,allow_pickle=False)["sim_state"]):
                    raise ValueError("teacher/ordinary branch prefix full-state mismatch")
            offset=0
            for query in queries:
                raw=policy_input(obs,full,annotation,init=init,query=query["step"],evaluator=c.evaluator)
                if image_hash(raw)!=query["image_sha256"] or not np.array_equal(raw["observation/state"],query["state"]):
                    raise ValueError(f"image/proprio mismatch at {query['step']}")
                checks+=1
                if len(actions)-offset>=10:
                    target=actions[offset:offset+10]
                    identity=hashlib.sha256(parent.encode()+str(query["step"]).encode()+raw["observation/image"].tobytes()+raw["observation/wrist_image"].tobytes()+np.asarray(raw["observation/state"]).tobytes()+target.tobytes()).hexdigest()
                    if spec["kind"]=="normal" and identity in seen:
                        duplicates+=1
                    else:
                        values=transform(dict(raw,actions=target))
                        restored=reverse(dict(state=values["state"],actions=np.asarray(values["actions"],np.float32)))
                        error=float(np.max(np.abs(restored["actions"][:,:7]-target)))
                        roundtrip=max(roundtrip,error)
                        if error>1e-5 or np.asarray(values["actions"]).shape!=(10,32):
                            raise ValueError("action normalization/padding roundtrip failed")
                        sample["image_base"].append(values["image"]["base_0_rgb"])
                        sample["image_left_wrist"].append(values["image"]["left_wrist_0_rgb"])
                        for key in ("state","actions","tokenized_prompt","tokenized_prompt_mask","skills","skill_mask","objects"):
                            sample[key].append(values[key])
                        sample["step"].append(query["step"])
                        if spec["kind"]=="normal":seen.add(identity)
                else:
                    dropped+=1
                count=min(len(query["actions"]),len(actions)-offset)
                for action in actions[offset:offset+count]:
                    obs,_,_,_=env.step(action.tolist());truth(env.env,goals);env.check_success()
                offset+=count
            if offset!=len(actions) or not env.check_success():
                raise ValueError("replay did not end in original whole-task success")
            arrays={key:np.asarray(value) for key,value in sample.items()}
            for key in ("state","actions"):arrays[key]=arrays[key].astype(np.float32)
            path=out/(spec["id"]+".npz")
            np.savez(path,**arrays)
            report=dict(**spec,parent=parent,task=task,init=init,start_step=start,end_step=end,
                samples=len(sample["step"]),checks=checks,dropped_terminal_windows=dropped,
                duplicate_normal_windows=duplicates,roundtrip_max_abs=roundtrip,path=str(path),status="verified",
                student_instruction=full,scope="successful_intervention_continuation" if spec["kind"]=="recovery" else "ordinary_success")
        except Exception as exc:
            bad_parents.add(parent)
            report=dict(**spec,parent=parent,status="excluded",error=repr(exc))
        finally:
            env.close()
        reports.append(report)
        write(out/"replay_audit.json",dict(trajectories=reports,excluded_parents=sorted(bad_parents)))
        print(json.dumps({k:report.get(k) for k in ("id","status","samples","start_step","error")}),flush=True)
    eligible=[row for row in reports if row["status"]=="verified" and row["parent"] not in bad_parents and row["samples"]>0]
    recovery=[row for row in eligible if row["kind"]=="recovery"]
    normal=[row for row in eligible if row["kind"]=="normal"]
    recovery_tasks={row["task"] for row in recovery}
    if len({row["parent"] for row in recovery})<2 or len(recovery_tasks)<2 or not recovery_tasks<={row["task"] for row in normal}:
        raise ValueError("insufficient verified teacher families or task-matched ordinary successes")
    comparison=[]
    for row in recovery:
        if row["task"] in (2,6):
            physical=json.loads((Path(manifest["source_root"])/"PHYSICAL_SELECTION.json").read_text())
            p=next(item for item in physical["states"] if item["case"]==row["case"])
            comparison.append(dict(case=row["case"],C_start=row["start_step"],physical_start=p["step"],
                same_state=bool(row["start_step"]==p["step"]),claim="successful continuation, not demonstrated recovery from the late contact-stall state"))
    index=dict(shards=eligible,excluded_parents=sorted(bad_parents),recovery_tasks=sorted(recovery_tasks),
               teacher_vs_physical=comparison,input_contract="original instruction, current public observations only; future actions are targets only",
               teacher_paths=len(recovery),teacher_parents=len({r['parent'] for r in recovery}))
    write(out/"index.json",index);(root/"DATA_READY").write_text(str(time.time()))
    return index


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);a=p.parse_args()
    try:export(a.root)
    except Exception as exc:
        write(a.root/"DATA_ERROR.json",dict(error=repr(exc)));raise
