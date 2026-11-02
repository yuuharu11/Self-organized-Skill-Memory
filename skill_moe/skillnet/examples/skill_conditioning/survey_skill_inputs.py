"""Initial-observation skill interventions across all nine LIBERO-Skill tasks.

No policy action is executed. This complements selected failure-state branches
with a fixed broad survey; it does not measure closed-loop usefulness.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from collect_interventions import Collector, counterfactual_inputs, policy_input, rpc, serial, image_hash


def survey(root):
    out = root / "initial_survey"
    out.mkdir(exist_ok=False)
    (out / "manifest.json").write_text(json.dumps({"tasks_zero_based": list(range(9)), "init": 0,
        "conditions": ["original", "reverse_skills", "wrong_skill_ids", "no_skills", "noise_only"],
        "scope": "initial stabilized observations only; no executed policy actions; one initial state per task",
        "selection": "all nine tasks, not selected by outcomes"}, indent=2) + "\n")
    c = Collector(root)
    try:
        for tid in range(9):
            case = out / f"task{tid+1:02d}"
            case.mkdir()
            env, obs, prompt, annotation = c.setup(tid, 0)
            try:
                raw = policy_input(obs, prompt, annotation, init=0, query=0, evaluator=c.evaluator)
                np.savez_compressed(case / "observation.npz", image=raw["observation/image"], wrist=raw["observation/wrist_image"],
                                    state=raw["observation/state"], sim_state=env.get_sim_state())
                original = rpc(c.client, raw)
                reference = rpc(c.reference, raw)
                parity = float(np.max(np.abs(original["actions"] - reference["actions"])))
                if parity > c.manifest["parity_tolerance_raw_action"]:
                    raise RuntimeError(f"sampler mismatch {parity}")
                records = []
                for name, v in counterfactual_inputs(raw, "unused").items():
                    if name == "remaining_instruction":
                        continue
                    r = rpc(c.client, v)
                    difference = np.asarray(r["actions"])[:5, :7] - np.asarray(original["actions"])[:5, :7]
                    routes = np.asarray(r["diagnostics"]["router_top1"])[:, :5]
                    base_routes = np.asarray(original["diagnostics"]["router_top1"])[:, :5]
                    records.append({"condition": name, "action_rms_by_family": [float(np.sqrt(np.mean(difference[:, s] ** 2)))
                                    for s in (slice(0, 3), slice(3, 6), slice(6, 7))],
                                    "route_change_fraction": float(np.mean(routes != base_routes)), "response": serial(r)})
                result = {"task_number": tid+1, "init": 0, "prompt": prompt, "raw_skills": raw["skills"],
                          "raw_objects": raw["objects"], "seed": raw["s2_rng_seed"], "image_sha256": image_hash(raw),
                          "sampler_parity_max_abs": parity, "original": serial(original), "interventions": records}
                (case / "result.json").write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps({"task": tid+1, "parity": parity, "status": "complete"}), flush=True)
            finally:
                env.close()
        (out / "DONE").write_text("complete\n")
    finally:
        c.client.close();c.reference.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    survey(p.parse_args().root)
