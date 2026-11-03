"""Frozen decisions for the bounded rescue diagnosis; no simulator imports."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


MAIN_CASES = ("task03-init21", "task08-init20", "task02-init21", "task04-init20", "task03-init20", "task07-init21")
CONDITIONS = ("A", "B", "C", "D")
SEEDS = (73, 89)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def case_ids(name):
    task, init = name.split("-init")
    return int(task.removeprefix("task"))-1, int(init)


def noise_seed(seed, init, step):
    # Use absolute control time, including teacher steps, not the number of RPCs.
    return seed*1000000 + init*1000 + step


def action_prefix(queries, end_step):
    actions = []
    for query in queries:
        if query["step"] != len(actions):
            raise ValueError("noncontiguous recorded action prefix")
        actions.extend(query["actions"][:max(0, end_step-len(actions))])
        if len(actions) == end_step:
            break
    if len(actions) != end_step:
        raise ValueError("recording does not cover requested control step")
    return np.asarray(actions, np.float64)


def group_name(original, oracle):
    if original:
        return "both_success" if oracle else "harm"
    return "rescue" if oracle else "both_fail"


def base_prompts(full, remaining):
    return dict(A=full, B=remaining,
                C=f"{full}. Current operation: {remaining}.",
                D=f"{full}. Current operation: follow the original instruction.")


def match_prompts(prompts, audit):
    """audit is a read-only tokenizer RPC; never uses rollout outcomes."""
    prompts = dict(prompts)
    audits = {key:audit(value) for key,value in prompts.items()}
    initial = {key:value["length"] for key,value in audits.items()}
    additions = dict(C=0, D=0)
    while abs(audits["C"]["length"]-audits["D"]["length"])>1:
        shorter = min(("C","D"),key=lambda key:audits[key]["length"])
        prompts[shorter] += " please"
        additions[shorter] += 1
        audits[shorter] = audit(prompts[shorter])
        if audits[shorter]["length"]-initial[shorter]>16 or additions[shorter]>32:
            raise ValueError("neutral token padding exceeded the 16-token limit")
    for value in audits.values():
        if value["truncated"] or not value["exact_encoding"]:
            raise ValueError("input truncation or unexpected tokenizer encoding")
    return dict(prompts=prompts, audits=audits, neutral_please_additions=additions,
                matched_token_difference=abs(audits["C"]["length"]-audits["D"]["length"]))


def pair_table(rows, first, second):
    pairs = []
    for key, arms in sorted(rows.items()):
        if first in arms and second in arms:
            a,b=bool(arms[first]["success"]),bool(arms[second]["success"])
            pairs.append(dict(case=key[0],seed=key[1],first=a,second=b,delta=int(a)-int(b)))
    by_parent=defaultdict(list)
    for row in pairs:
        by_parent[row["case"]].append(row["delta"])
    return dict(first=first,second=second,n_seed_pairs=len(pairs),n_parent_states=len(by_parent),
                rescue=sum(row["delta"]==1 for row in pairs),harm=sum(row["delta"]==-1 for row in pairs),
                net=sum(row["delta"] for row in pairs),
                parent_mean_deltas={key:float(np.mean(values)) for key,values in by_parent.items()},pairs=pairs)


def main_decision(rows, original_outcomes, eligible_cases):
    complete = {key:arms for key,arms in rows.items() if set(arms)==set(CONDITIONS)}
    ca=pair_table(complete,"C","A"); cd=pair_table(complete,"C","D")
    all_complete = set(eligible_cases)==set(MAIN_CASES) and len(complete)==12
    rescued=[]
    for case, delta in ca["parent_mean_deltas"].items():
        if not original_outcomes[case] and delta>0:
            rescued.append(case)
    preserved=all(not arms["A"]["success"] or arms["C"]["success"] for arms in complete.values())
    old_success_preserved=all(arms["C"]["success"] for (case,_),arms in complete.items() if original_outcomes[case])
    passes=all_complete and ca["net"]>0 and cd["net"]>0 and preserved and old_success_preserved and len(rescued)>=2 and len({case_ids(case)[0] for case in rescued})>=2
    return dict(complete=all_complete,passes=bool(passes),C_vs_A=ca,C_vs_D=cd,
                C_vs_B=pair_table(complete,"C","B"),D_vs_A=pair_table(complete,"D","A"),
                rescued_original_failure_states=rescued,original_success_preserved=bool(old_success_preserved),
                reason="prespecified preliminary gate passed" if passes else "prespecified gate not met; no prompt search")


def window_phenomena(rows):
    """Operational trace labels, never causal labels. Window is 50 controls."""
    if len(rows)<50:
        return [],{}
    window=rows[-50:]
    contacts=sum(bool(row["target_contact"]) for row in window)
    movement=np.asarray([row["progress_vector"] for row in window],float)
    span=float(np.max(np.ptp(movement,axis=0))) if movement.shape[1] else None
    command=float(np.mean([np.linalg.norm(row["command"][:3]) for row in window]))
    min_distance=min(row["target_distance"] for row in window)
    labels=[]
    same_target=len({row.get("target") for row in window})==1
    if same_target and span is not None and contacts>=5 and span<.01 and command>.02:
        labels.append("contact_stall")
    if same_target and contacts==0:
        labels.append("contact_missing")
        if min_distance>.15:
            labels.append("target_unreached")
    reversals=0
    if movement.shape[1]:
        d=np.diff(movement[:,0]);sign=np.sign(d[np.abs(d)>.001])
        reversals=int(np.sum(sign[1:]!=sign[:-1]))
        if same_target and reversals>=4 and span<.1:
            labels.append("oscillation_or_reversal")
    for k in range(2):
        values=[row["truth"]["raw_goals"][k] for row in window]
        if any(values) and not any(values[-5:]):
            labels.append("achieved_goal_regression")
    return sorted(set(labels)),dict(target_contact_steps=contacts,normalized_progress_span=span,
        mean_translation_command=command,minimum_target_distance_m=min_distance,reversal_count=reversals)


def teacher_action(current_pos, current_quat, reference, index, output_max, orientation_error):
    target=np.asarray(reference["eef"],float).copy()
    if index<10:
        target=np.asarray(reference["retreat_origin"],float)+np.asarray([0.,0.,.05])
    translation=np.clip(.5*(target-np.asarray(current_pos))/np.asarray(output_max[:3]),-.5,.5)
    rotation=np.clip(.5*np.asarray(orientation_error(reference["quat"],current_quat))/np.asarray(output_max[3:6]),-.5,.5)
    return np.concatenate([translation,rotation,[np.clip(reference["grip"],-1.,1.)]])


def teacher_translation_feasible(reference):
    retreat=np.asarray(reference["retreat_origin"],float)+np.asarray([0.,0.,.05])
    # Ten approach steps, each command at most .5 with OSC .05m scaling.
    # Necessary nominal bound, not an obstacle-free reachability guarantee.
    return bool(np.all(np.abs(np.asarray(reference["eef"])-retreat)<=.25))
