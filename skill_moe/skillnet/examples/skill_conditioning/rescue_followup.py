"""Prepare exactly one outcome-dependent follow-up, without prompt search."""
import importlib
import json
from pathlib import Path

from collect_interventions import image_hash,policy_input
from effect_worker import setup,write
import rescue_protocol

# Follow-up preparation is imported after the primary phase. Load its finalized
# helper definitions even if the scheduler imported the module during setup.
importlib.reload(rescue_protocol)
from rescue_protocol import SEEDS,action_prefix,case_ids,read_rows,teacher_translation_feasible
from rescue_runtime import save_snapshot


def used_initial_states(root,tasks):
    used={task:set() for task in tasks};sources=[]
    base=Path(json.loads((root/"manifest.json").read_text())["source_root"]).parent
    def inspect(value,task=None,suite=None):
        found=[]
        if isinstance(value,dict):
            suite=value.get("suite",value.get("task_suite_name",value.get("task_suite",suite)))
            if isinstance(suite,str) and "libero" in suite and suite not in ("libero_skill","libero_skill_obj"):
                return []
            current=value.get("task",value.get("task_id",task))
            if not isinstance(current,int):
                current=task
            for key in ("init","init_state_id","initial_state_index","trial_index"):
                item=value.get(key)
                if isinstance(item,int) and 0<=item<50:
                    for target in ([current] if current in used else tasks):
                        used[target].add(item)
                        found.append((target,item))
            for key in ("trial_indices","init_indices","init_states"):
                items=value.get(key)
                if isinstance(items,list) and all(isinstance(i,int) for i in items):
                    for target in ([current] if current in used else tasks):
                        used[target].update(i for i in items if 0<=i<50)
            for item in value.values():
                if isinstance(item,(dict,list)):
                    found.extend(inspect(item,current,suite))
        elif isinstance(value,list):
            for item in value:
                found.extend(inspect(item,task,suite))
        return found
    for path in base.rglob("*.json"):
        if path.name not in ("job.json","manifest.json") or root in path.parents:
            continue
        try:
            items=inspect(json.loads(path.read_text()))
            if items:
                sources.append(str(path))
        except (ValueError,OSError):
            continue
    write(root/"used_initial_states_audit.json",dict(used={str(k):sorted(v) for k,v in used.items()},source_files=sources))
    return used


def confirmation(root,decision):
    tasks=sorted({case_ids(case)[0] for case in decision["rescued_original_failure_states"]})[:2]
    used=used_initial_states(root,tasks)
    groups=[];chosen={}
    for task in tasks:
        available=[i for i in (30,31,*range(32,50)) if i not in used[task]]
        if len(available)<2:
            write(root/"confirmation_unavailable.json",dict(task=task,reason="fewer than two unused public initial states"));return []
        chosen[task]=available[:2]
        template=next(path for path in sorted((root/"prompts").glob("task*.json")) if case_ids(path.stem)[0]==task)
        payload=json.loads(template.read_text())
        for init in available[:2]:
            case=f"task{task+1:02d}-init{init:02d}"
            promptfile=root/"prompts"/(case+".json")
            write(promptfile,{**payload,"case":case,"reused_frozen_template_from":template.name,"snapshot":None})
            groups.append([dict(mode="confirmation",case=case,task=task,init=init,seed=seed,condition=condition,
                prompts_file=str(promptfile.relative_to(root)),execute_steps=5,
                directory=f"confirmation/{case}-seed{seed}-{condition}") for seed in SEEDS for condition in ("A","C","D")])
    write(root/"CONFIRMATION_SELECTION.json",dict(tasks=tasks,initial_states=chosen))
    return groups


def physical(root,summary):
    manifest=json.loads((root/"manifest.json").read_text());source=Path(manifest["source_root"])
    selection=summary["classification"]["physical_selection"]
    frozen_target=root/"PHYSICAL_TARGET_FROZEN.json"
    if frozen_target.exists():
        expected=json.loads(frozen_target.read_text())["states"]
        if [(row["case"],row["step"]) for row in selection] != [(row["case"],row["step"]) for row in expected]:
            raise ValueError("physical target changed after original-trace selection was frozen")
    c,_=setup(root);groups=[];prepared=[]
    for info in selection:
        case=info["case"];task,init=case_ids(case);step=info["step"]
        env,obs,full,ann=c.setup(task,init)
        try:
            queries=read_rows(source/"evaluation"/(case+"-original")/"queries.jsonl")
            prefix=action_prefix(queries,step)
            for action in prefix:
                obs,_,_,_=env.step(action.tolist())
            raw=policy_input(obs,full,ann,init=init,query=step,evaluator=c.evaluator)
            old=next(q for q in queries if q["step"]==step)
            if image_hash(raw)!=old["image_sha256"]:
                raise RuntimeError("physical snapshot replay image mismatch")
            path=root/"physical_snapshots"/(case+".npz");path.parent.mkdir(exist_ok=True)
            save_snapshot(path,env,raw,prefix,step)
            teacher_ok=teacher_translation_feasible(info["teacher_reference"])
            prepared.append({**info,"teacher_available":teacher_ok,
                             "teacher_unavailable_reason":None if teacher_ok else "reference exceeds nominal ten-step approach displacement bound"})
            arms=("baseline5","replan2","teacher") if teacher_ok else ("baseline5","replan2")
            groups.append([dict(mode="physical",case=case,task=task,init=init,seed=seed,condition=condition,
                snapshot=str(path.relative_to(root)),execute_steps=2 if condition=="replan2" else 5,
                teacher_reference=info["teacher_reference"],
                directory=f"physical/{case}-seed{seed}-{condition}") for seed in SEEDS for condition in arms])
        finally:
            env.close()
    write(root/"PHYSICAL_SELECTION.json",dict(phenomenon=summary["classification"]["chosen_physical_phenomenon"],states=prepared))
    return groups


def prepare_followup(root,decision):
    import rescue_report
    summary=importlib.reload(rescue_report).summarize(root)
    if decision["passes"]:
        return "confirmation",confirmation(root,decision)
    return "physical",physical(root,summary)
