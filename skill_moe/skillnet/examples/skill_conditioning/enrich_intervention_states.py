"""Export model joint metadata and verified initial observations, without inference.

Joint positions are read from the recorded flattened MuJoCo state after checking
its layout against the simulator API. No stored policy outcome is changed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from collect_interventions import Collector, image_hash, policy_input
from analyze_interventions import read_jsonl


def enrich(root):
    from libero.libero import benchmark
    c = object.__new__(Collector)  # Environment setup only; no policy clients.
    c.suite = benchmark.get_benchmark_dict()["libero_skill_obj"]()
    c.annotations = json.loads((root / "frozen/annotations.json").read_text())
    spec = importlib.util.spec_from_file_location("enrichment_frozen_evaluator", root / "frozen/eval_prompts.py")
    c.evaluator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = c.evaluator
    spec.loader.exec_module(c.evaluator)
    result = []
    for case in sorted(root.glob("task*-init*")):
        task_id = int(case.name.split("-")[0].removeprefix("task")) - 1
        init = int(case.name.split("init")[1])
        queries = read_jsonl(case / "baseline/queries.jsonl")
        if not queries:
            continue
        env, obs, prompt, annotation = c.setup(task_id, init)
        try:
            model = env.env.sim.model
            state = env.env.sim.get_state()
            flat = state.flatten()
            np.testing.assert_array_equal(flat[1:1 + model.nq], state.qpos)
            raw = policy_input(obs, prompt, annotation, init=init, query=0, evaluator=c.evaluator)
            assert image_hash(raw) == queries[0]["image_sha256"]
            np.testing.assert_array_equal(raw["observation/state"], queries[0]["state"])
            np.savez_compressed(case / "initial_observation.npz", image=raw["observation/image"],
                                wrist=raw["observation/wrist_image"], state=raw["observation/state"], sim_state=flat)
            joints = {name: {"flat_index": 1 + int(model.jnt_qposadr[i]), "range": model.jnt_range[i].tolist()}
                      for i, name in enumerate(model.joint_names)
                      if not name.startswith("robot") and int(model.jnt_type[i]) in (2, 3)}
            (case / "joint_metadata.json").write_text(json.dumps(joints, indent=2) + "\n")
            for branch in ("baseline", "original", "remaining_instruction", "original_repeat", "released_original", "released_remaining", "released_original_repeat"):
                rows = []
                for q in read_jsonl(case / branch / "queries.jsonl"):
                    rows.append({"step": q["step"], "joints": {n: q["sim_state"][v["flat_index"]] for n, v in joints.items()}})
                if rows:
                    (case / branch / "joint_traces.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            result.append({"case": case.name, "initial_policy_input_identical": True, "joint_layout_checked": True})
        finally:
            env.close()
    (root / "enrichment_checks.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(enrich(parser.parse_args().root)))
