"""Follow-up: distinguish placement-goal truth from actual release completion.

The first cohort can trigger while the object is still grasped. Select the first
recorded baseline replan with stable placement, no grasp contact, and five open
commands, then branch only the instruction. These are exploratory follow-ups,
kept separate from the original preregistered cohort.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_interventions import read_jsonl
from collect_interventions import Collector, append, image_hash, policy_input, rpc


def find_released_snapshot(c, case, task, init):
    queries = {r["step"]: r for r in read_jsonl(case / "baseline/queries.jsonl")}
    actions = read_jsonl(case / "baseline/states.jsonl")
    env, obs, prompt, annotation = c.setup(task, init)
    stage = c.Stages(env.env, task, annotation["plan"])
    stage.observe(0, -1)
    prefix = []
    try:
        for row in actions:
            action = row["action"]
            obs, _, _, _ = env.step(action)
            prefix.append(action)
            step = row["step"]
            stage.observe(step, float(action[6]))
            if step not in queries or not stage.rows[1]["final"] or stage.rows[1]["first_stable3_step"] is None:
                continue
            if stage.rows[2]["final"] or not all(a[6] < 0 for a in prefix[-5:]):
                continue
            name = stage.rows[1]["goal"][1]
            obj = env.env.get_object(name)
            grasp = bool(env.env._check_grasp(gripper=env.env.robots[0].gripper, object_geoms=obj.contact_geoms))
            if grasp:
                continue
            raw = policy_input(obs, prompt, annotation, init=init, query=step // 5, evaluator=c.evaluator)
            delta = float(np.max(np.abs(env.get_sim_state() - np.asarray(queries[step]["sim_state"]))))
            assert delta <= c.manifest["restore_tolerance_sim_state"] and image_hash(raw) == queries[step]["image_sha256"]
            snapshot = {"step": step, "prefix": prefix, "sim_state": env.get_sim_state().copy(),
                        "image_sha256": image_hash(raw), "grasp_contact": grasp,
                        "last_five_gripper_commands": [a[6] for a in prefix[-5:]]}
            p = case / "released_snapshot"
            p.mkdir(exist_ok=False)
            np.savez_compressed(p / "observation.npz", image=raw["observation/image"], wrist=raw["observation/wrist_image"],
                                state=raw["observation/state"], sim_state=snapshot["sim_state"])
            (p / "selection.json").write_text(json.dumps({k: v for k, v in snapshot.items() if k not in ("prefix", "sim_state")}, indent=2) + "\n")
            return snapshot
    finally:
        env.close()


def run(root):
    followup = {"status": "exploratory measurement-correction follow-up", "cases": "same four development cases, no replacements",
                "selection": "first baseline replan with stable placement, absent grasp contact, five prior open commands, fixture goal false",
                "branches": ["released_original", "released_remaining", "released_original_repeat"], "control_budget_each": 200,
                "reason": "placement-goal truth does not imply the gripper has released the object",
                "behavior_source": "untouched public server; instrumented routes are audited auxiliary readouts"}
    (root / "released_followup_manifest.json").write_text(json.dumps(followup, indent=2) + "\n")
    c = Collector(root)
    c.reference_behavior = True
    try:
        for init in c.manifest["initial_states"]:
            for task in c.manifest["tasks_zero_based"]:
                c.check_budget()
                case = root / f"task{task+1:02d}-init{init:02d}"
                snap = find_released_snapshot(c, case, task, init)
                if snap is None:
                    append(root / "released_snapshot_missing.jsonl", {"task_id": task, "init": init, "reason": "no released qualifying baseline state"})
                    continue
                for condition in followup["branches"]:
                    env, obs, prompt, annotation = c.setup(task, init)
                    stage = c.Stages(env.env, task, annotation["plan"])
                    stage.observe(0, -1)
                    try:
                        for i, action in enumerate(snap["prefix"], 1):
                            obs, _, _, _ = env.step(action)
                            stage.observe(i, float(action[6]))
                        raw = policy_input(obs, prompt, annotation, init=init, query=snap["step"] // 5, evaluator=c.evaluator)
                        delta = float(np.max(np.abs(env.get_sim_state() - snap["sim_state"])))
                        same = image_hash(raw) == snap["image_sha256"]
                        append(root / "released_restore_checks.jsonl", {"case": case.name, "branch": condition,
                            "state_max_abs": delta, "image_identical": same})
                        if delta > c.manifest["restore_tolerance_sim_state"] or not same:
                            raise RuntimeError("released follow-up did not reproduce identical input")
                        branch_prompt = annotation["plan"][-1] if condition == "released_remaining" else prompt
                        c.segment(env, obs, branch_prompt, annotation, stage, case / condition, init=init,
                                  start_step=snap["step"], cap=200)
                    finally:
                        env.close()
        (root / "RELEASED_FOLLOWUP_DONE").write_text("complete\n")
    finally:
        c.client.close();c.reference.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    run(p.parse_args().root)
