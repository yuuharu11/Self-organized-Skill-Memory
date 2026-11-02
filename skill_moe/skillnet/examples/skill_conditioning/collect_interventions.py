"""Exploratory paired SkillNet diagnosis; never trains a policy.

Replay a recorded action prefix to reproduce the reached state, verify the full
simulator state and policy images, then branch only the instruction. This also
captures routing and fixed-observation skill interventions. See manifest.json.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "memory_pilot/anchor_flow"))
from run_case import S2WebsocketClientPolicy, _task_language, episode_status
from skillnet_client import image_tools, msgpack_numpy


def append(path, row):
    with path.open("a") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    return value


def policy_input(obs, prompt, annotation, *, init, query, evaluator):
    images = [image_tools.convert_to_uint8(image_tools.resize_with_pad(
        np.ascontiguousarray(obs[k][::-1, ::-1]), 224, 224))
        for k in ("agentview_image", "robot0_eye_in_hand_image")]
    state = np.concatenate((obs["robot0_eef_pos"], evaluator._quat2axisangle(
        obs["robot0_eef_quat"].copy()), obs["robot0_gripper_qpos"]))
    return {"observation/image": images[0], "observation/wrist_image": images[1],
            "observation/state": state, "prompt": prompt,
            "skills": str(annotation["all_classes"]), "objects": str(annotation["objects"]),
            "s2_rng_seed": (7 + init) * 100000 + query,
            "s2_condition": "anchor_disabled_t0_1", "s2_query_text_source": "instruction_only",
            "s2_query_id": init * 1000 + query}


def image_hash(raw):
    return hashlib.sha256(raw["observation/image"].tobytes() + raw["observation/wrist_image"].tobytes()).hexdigest()


def rpc(client, raw):
    client._ws.send(client._packer.pack(raw))
    result = client._ws.recv(timeout=300)
    if isinstance(result, str):
        raise RuntimeError(result)
    result = msgpack_numpy.unpackb(result)
    actions = np.asarray(result["actions"])
    # The released output transform may retain only seven horizon rows. The
    # evaluator executes the first five, so validate that public contract.
    if actions.ndim != 2 or actions.shape[0] < 5 or actions.shape[1] < 7 or not np.isfinite(actions).all():
        raise ValueError(f"invalid action shape/values {actions.shape}")
    return result


def counterfactual_inputs(raw, remaining):
    import ast
    skills = ast.literal_eval(raw["skills"])
    mapping = {0: 3, 1: 4, 3: 0, 4: 1}
    variants = {}
    for name in ("reverse_skills", "wrong_skill_ids", "no_skills", "remaining_instruction", "noise_only"):
        v = copy.deepcopy(raw)
        if name == "reverse_skills":
            v["skills"] = str(skills[::-1])
        elif name == "wrong_skill_ids":
            v["skills"] = str([mapping[s] for s in skills])
        elif name == "no_skills":
            v["diagnostic_no_skills"] = True
        elif name == "remaining_instruction":
            v["prompt"] = remaining
        else:
            v["s2_rng_seed"] += 10000000
        variants[name] = v
    return variants


class Collector:
    def __init__(self, output, *, public_only=False):
        self.out = output
        self.manifest = json.loads((output / "manifest.json").read_text())
        self.started = time.monotonic()
        self.public_only = public_only
        self.client = None if public_only else S2WebsocketClientPolicy("127.0.0.1", self.manifest["port"])
        self.reference = S2WebsocketClientPolicy("127.0.0.1", self.manifest.get("reference_port", 38723))
        frozen = output / "frozen"
        sys.path.insert(0, str(frozen))
        import stages
        self.Stages = stages.Stages
        spec = importlib.util.spec_from_file_location("diagnosis_frozen_evaluator", frozen / "eval_prompts.py")
        evaluator = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = evaluator
        spec.loader.exec_module(evaluator)
        self.evaluator = evaluator
        self.annotations = json.loads((frozen / "annotations.json").read_text())
        from libero.libero import benchmark
        self.suite = benchmark.get_benchmark_dict()["libero_skill_obj"]()
        evaluator._validate_libero_skill_tasks(self.suite)

    def check_budget(self):
        if (self.out / "STOP_REQUESTED").exists():
            raise RuntimeError("user stop requested")
        if time.monotonic() - self.started > self.manifest["stop_after_seconds"]:
            raise TimeoutError("diagnostic wall-time cap reached")

    def setup(self, task_id, init):
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        task = self.suite.get_task(task_id)
        prompt = _task_language("libero_skill_obj", task)
        path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(path), camera_heights=256,
                                 camera_widths=256, horizon=811)
        env.seed(7)
        env.reset()
        obs = env.set_init_state(np.asarray(self.suite.get_task_init_states(task_id)[init]))
        for _ in range(10):
            obs, _, _, _ = env.step([0.] * 6 + [-1.])
        return env, obs, prompt, self.annotations[prompt]

    def fixed_comparisons(self, directory, raw, base, remaining):
        rows = []
        for name, variant in counterfactual_inputs(raw, remaining).items():
            self.check_budget()
            response = rpc(self.client, variant)
            d = np.asarray(response["actions"])[:5, :7] - np.asarray(base["actions"])[:5, :7]
            routes = np.asarray(response["diagnostics"]["router_top1"])[:, :5]
            original_routes = np.asarray(base["diagnostics"]["router_top1"])[:, :5]
            row = {"condition": name, "image_sha256": image_hash(variant),
                   "seed": variant["s2_rng_seed"], "prompt": variant["prompt"], "raw_skills": variant["skills"],
                   "action_rms_by_family": [float(np.sqrt(np.mean(d[:, s] ** 2))) for s in (slice(0, 3), slice(3, 6), slice(6, 7))],
                   "route_change_fraction": float(np.mean(routes != original_routes)),
                   "response": serial(response)}
            assert row["image_sha256"] == image_hash(raw)
            append(directory / "counterfactuals.jsonl", row)
            rows.append(row)
        return rows

    def segment(self, env, obs, prompt, annotation, stage, directory, *, init, start_step, cap, prefix=None, find_snapshot=False):
        directory.mkdir(parents=True, exist_ok=False)
        prefix = list(prefix or [])
        snapshot = None
        success = False
        q = start_step // 5
        step = start_step
        with imageio.get_writer(str(directory / "video.mp4"), fps=4, codec="libx264", ffmpeg_params=["-threads", "1"]) as video:
            while step < start_step + cap:
                self.check_budget()
                raw = policy_input(obs, prompt, annotation, init=init, query=q, evaluator=self.evaluator)
                tick = time.monotonic()
                if self.public_only:
                    response = rpc(self.reference, raw)
                    response["behavior_source"] = "untouched_public_server"
                else:
                    response = rpc(self.client, raw)
                if not self.public_only and getattr(self, "reference_behavior", False):
                    # Instrumentation can change XLA floating-point evaluation.
                    # Use the untouched public server for follow-up behavior;
                    # retain instrumented routes with an explicit parity audit.
                    reference = rpc(self.reference, raw)
                    response["diagnostics"]["capture_action_max_abs"] = float(np.max(np.abs(
                        response["actions"] - reference["actions"])))
                    response["instrumented_actions"] = response["actions"]
                    response["actions"] = reference["actions"]
                    response["behavior_source"] = "untouched_public_server"
                record = {"step": step, "query": q, "seed": raw["s2_rng_seed"], "prompt": prompt,
                          "raw_skills": raw["skills"], "image_sha256": image_hash(raw),
                          "state": raw["observation/state"].tolist(), "stage_truth": [r["final"] for r in stage.rows],
                          "sim_state": env.get_sim_state().tolist(), "inference_s": time.monotonic() - tick,
                          "response": serial(response)}
                append(directory / "queries.jsonl", record)
                if q % 10 == 0:
                    append(self.out / "progress.jsonl", {"directory": str(directory.relative_to(self.out)), "step": step, "query": q})
                if find_snapshot and snapshot is None and stage.rows[1]["final"] and stage.rows[1]["first_stable3_step"] is not None:
                    snapshot = {"step": step, "prefix": list(prefix), "sim_state": env.get_sim_state().copy(),
                                "image_sha256": image_hash(raw), "raw": raw, "base_response": response}
                    snapdir = directory / "snapshot"
                    snapdir.mkdir()
                    np.savez_compressed(snapdir / "observation.npz", image=raw["observation/image"],
                                        wrist=raw["observation/wrist_image"], state=raw["observation/state"], sim_state=snapshot["sim_state"])
                    self.fixed_comparisons(snapdir, raw, response, annotation["plan"][-1])
                if start_step == 0 and step == 0:
                    reference = rpc(self.reference, raw)
                    difference = float(np.max(np.abs(response["actions"] - reference["actions"])))
                    append(self.out / "sampler_parity.jsonl", {"case": str(directory.relative_to(self.out)), "max_abs": difference})
                    if difference > self.manifest["parity_tolerance_raw_action"]:
                        raise RuntimeError(f"diagnostic sampler differs from existing public sampler: {difference}")
                    self.fixed_comparisons(directory, raw, response, annotation["plan"][-1])
                video.append_data(np.concatenate([raw["observation/image"], raw["observation/wrist_image"]], axis=1))
                for action in np.asarray(response["actions"])[:5, :7]:
                    obs, _, returned, _ = env.step(action.tolist())
                    step += 1
                    prefix.append(action.tolist())
                    truth = stage.observe(step, float(action[6]))
                    success, done = episode_status(env, returned)
                    append(directory / "states.jsonl", {"step": step, "action": action.tolist(), "stage_truth": truth,
                                                         "eef": obs["robot0_eef_pos"].tolist()})
                    if done:
                        break
                q += 1
                if success or done:
                    break
        result = {"status": "complete", "success": bool(success), "start_step": start_step, "end_step": step,
                  "budget": cap, "prompt": prompt, **stage.result(success)}
        (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        append(self.out / "results.jsonl", {"case": str(directory.relative_to(self.out)), **result})
        return snapshot

    def run_case(self, task_id, init):
        directory = self.out / f"task{task_id+1:02d}-init{init:02d}"
        directory.mkdir(exist_ok=False)
        env, obs, prompt, annotation = self.setup(task_id, init)
        stage = self.Stages(env.env, task_id, annotation["plan"])
        stage.observe(0, -1)
        try:
            snap = self.segment(env, obs, prompt, annotation, stage, directory / "baseline", init=init,
                                start_step=0, cap=self.manifest["baseline_control_cap"], find_snapshot=True)
        finally:
            env.close()
        if snap is None:
            append(self.out / "snapshot-missing.jsonl", {"task_id": task_id, "init": init, "reason": "placement never stable at a replan boundary"})
            return
        for condition in self.manifest["branches"]:
            env, obs, _, _ = self.setup(task_id, init)
            stage = self.Stages(env.env, task_id, annotation["plan"])
            stage.observe(0, -1)
            try:
                for i, action in enumerate(snap["prefix"], 1):
                    obs, _, _, _ = env.step(action)
                    stage.observe(i, float(action[6]))
                raw = policy_input(obs, prompt, annotation, init=init, query=snap["step"] // 5, evaluator=self.evaluator)
                delta = float(np.max(np.abs(env.get_sim_state() - snap["sim_state"])))
                same_image = image_hash(raw) == snap["image_sha256"]
                append(self.out / "restore_checks.jsonl", {"task_id": task_id, "init": init, "branch": condition,
                                                          "state_max_abs": delta, "image_identical": same_image})
                if delta > self.manifest["restore_tolerance_sim_state"] or not same_image:
                    raise RuntimeError("replay does not reproduce the same state/images; no causal comparison allowed")
                branch_prompt = annotation["plan"][-1] if condition == "remaining_instruction" else prompt
                self.segment(env, obs, branch_prompt, annotation, stage, directory / condition, init=init,
                             start_step=snap["step"], cap=self.manifest["branch_control_cap"])
            finally:
                env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collector = Collector(args.output)
    try:
        for init in collector.manifest["initial_states"]:
            for task in collector.manifest["tasks_zero_based"]:
                collector.run_case(task, init)
        (args.output / "COLLECTION_DONE").write_text("complete\n")
    finally:
        collector.client.close()
        collector.reference.close()


if __name__ == "__main__":
    main()
