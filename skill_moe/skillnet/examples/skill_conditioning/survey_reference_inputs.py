"""Repeat the nine fixed observations using untouched-public-server actions.

Instrumented outputs are kept separately, with per-condition discrepancy. This
does not relax the original survey's parity gate or rewrite its stopped result.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from collect_interventions import Collector, counterfactual_inputs, image_hash, rpc, serial, _task_language


def survey(root):
    out = root / "reference_initial_survey"
    out.mkdir(exist_ok=False)
    (out / "manifest.json").write_text(json.dumps({"tasks": list(range(1, 10)), "init": 0,
        "behavior_source": "untouched existing public server", "no_policy_actions_executed": True,
        "instrumentation": "separate approximate route readout; record max action discrepancy for every condition",
        "reason": "original instrumented survey stopped on task09 parity discrepancy; preserve that cohort separately"}, indent=2) + "\n")
    c = Collector(root)
    summary = []
    try:
        for tid in range(9):
            task = c.suite.get_task(tid)
            prompt = _task_language("libero_skill_obj", task)
            ann = c.annotations[prompt]
            with np.load(root / "initial_survey" / f"task{tid+1:02d}/observation.npz") as z:
                raw = {"observation/image": z["image"], "observation/wrist_image": z["wrist"], "observation/state": z["state"],
                       "prompt": prompt, "skills": str(ann["all_classes"]), "objects": str(ann["objects"]),
                       "s2_rng_seed": 700000, "s2_condition": "anchor_disabled_t0_1", "s2_query_text_source": "instruction_only", "s2_query_id": 0}
            variants = {"original": raw, **{k: v for k, v in counterfactual_inputs(raw, "unused").items() if k != "remaining_instruction"}}
            variants["no_skills"]["skill_mask_override"] = np.zeros(4, dtype=bool)
            official = {}; instrumented = {}
            for name, variant in variants.items():
                official[name] = rpc(c.reference, variant)
                instrumented[name] = rpc(c.client, variant)
            rows = []
            for name in variants:
                d = official[name]["actions"][:5, :7] - official["original"]["actions"][:5, :7]
                a = np.asarray(instrumented[name]["diagnostics"]["router_top1"])[:, :5]
                b = np.asarray(instrumented["original"]["diagnostics"]["router_top1"])[:, :5]
                rows.append({"condition": name, "action_rms_by_family": [float(np.sqrt(np.mean(d[:, s] ** 2)))
                             for s in (slice(0, 3), slice(3, 6), slice(6, 7))],
                             "instrumented_route_change_fraction": float(np.mean(a != b)),
                             "capture_action_max_abs": float(np.max(np.abs(official[name]["actions"] - instrumented[name]["actions"])))})
            case = out / f"task{tid+1:02d}"
            case.mkdir()
            result = {"task_number": tid+1, "prompt": prompt, "raw_skills": raw["skills"],
                      "image_sha256": image_hash(raw), "conditions": rows,
                      "public_outputs": serial(official), "instrumented_outputs": serial(instrumented)}
            (case / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            summary.append({k: v for k, v in result.items() if not k.endswith("outputs")})
            (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps({"task": tid+1, "status": "complete"}), flush=True)
        (out / "DONE").write_text("complete\n")
    finally:
        c.client.close();c.reference.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    survey(p.parse_args().root)
