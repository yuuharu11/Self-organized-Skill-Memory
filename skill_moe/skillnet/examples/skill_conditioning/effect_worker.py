"""Finite from-start and paired-branch collection using the public sampler."""
import argparse
import hashlib
import importlib.util
import json
import os
import pickle
from pathlib import Path
import sys
import time

import numpy as np

from collect_interventions import Collector, S2WebsocketClientPolicy, append, image_hash, policy_input, rpc, serial, episode_status
from effect_controller import InstructionController, causal_vector, predict_effect, predict_probabilities


def write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(serial(value), ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def truth(base, ordered_goals):
    goal_truth, verified, grasped = [], [], []
    for goal in ordered_goals:
        yes = bool(base._eval_predicate(goal))
        grasp = False
        # Only placement goals require release; fixture open/close goals do not.
        placement = goal[0].lower() in ("in", "on")
        if placement and goal[1] in base.objects_dict:
            obj = base.get_object(goal[1])
            grasp = bool(base._check_grasp(gripper=base.robots[0].gripper, object_geoms=obj.contact_geoms))
        goal_truth.append(yes)
        verified.append(yes and not grasp)
        grasped.append(grasp)
    return dict(raw_goals=goal_truth, verified=verified, grasp=grasped)


def setup(root):
    # Reuse environment setup only, with no connection to instrumented samplers.
    c = object.__new__(Collector)
    frozen = root / "frozen"
    sys.path.insert(0, str(frozen))
    import stages
    spec = importlib.util.spec_from_file_location("effect_frozen_evaluator", frozen / "eval_prompts.py")
    evaluator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = evaluator
    spec.loader.exec_module(evaluator)
    from libero.libero import benchmark
    c.evaluator = evaluator
    c.annotations = json.loads((frozen / "annotations.json").read_text())
    c.suite = benchmark.get_benchmark_dict()["libero_skill_obj"]()
    evaluator._validate_libero_skill_tasks(c.suite)
    return c, stages


def run(root, job):
    directory = root / job["directory"]
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "result.json").exists():
        raise FileExistsError("completed outcomes must not be retried")
    write(directory / "job.json", job)
    c, stages = setup(root)
    env, obs, full, annotation = c.setup(job["task"], job["init"])
    from_order = [(index, pair[1]) for index, pair in enumerate(stages.BINDINGS[job["task"]]) if pair[0] == "goal"]
    assert len(from_order) == 2
    goals = [env.env.parsed_problem["goal_state"][index] for _, index in from_order]
    residual = ". ".join(annotation["plan"][from_order[0][0] + 1:])
    assert residual
    stage = stages.Stages(env.env, job["task"], annotation["plan"])
    prefix = []
    previous_state = None
    previous_action = np.zeros(7, np.float32)
    start_step = 0
    if job.get("snapshot"):
        snapshot = np.load(root / job["snapshot"], allow_pickle=False)
        prefix = snapshot["prefix"].tolist()
        for step, action in enumerate(prefix, 1):
            obs, _, _, _ = env.step(action)
            stage.observe(step, float(action[6]))
        start_step = len(prefix)
        raw = policy_input(obs, full, annotation, init=job["init"], query=start_step // 5, evaluator=c.evaluator)
        delta = float(np.max(np.abs(env.get_sim_state() - snapshot["sim_state"])))
        hash_ok = image_hash(raw) == str(snapshot["image_sha256"])
        write(directory / "replay_audit.json", dict(state_max_abs=delta, images_equal=hash_ok))
        if delta != 0 or not hash_ok:
            raise RuntimeError("paired state or image mismatch")
        previous_state = snapshot["previous_state"]
        previous_action = snapshot["previous_action"]
    client = S2WebsocketClientPolicy("127.0.0.1", job["port"])
    monitor = pickle.loads((root / "models/monitor.pkl").read_bytes()) if job["kind"] != "parent" else None
    effect = pickle.loads((root / "models/effect.pkl").read_bytes()) if job.get("condition") == "effect" else None
    condition = job.get("condition", "original")
    control = InstructionController(condition)
    if job["kind"] == "pair":
        control = InstructionController("original")
        control.switched = job["arm"] == "update"
        control.locked = True
    verified_run = [0, 0]
    snap_late = False
    vectors, labels, steps, raw_goals = [], [], [], []
    success = False
    done = False
    started = time.monotonic()
    step = start_step
    decision = dict(probabilities=[0., 0., 0.], beliefs=["unknown", "unknown"], effect=None)
    import imageio.v2 as imageio
    video = imageio.get_writer(str(directory / "video.mp4"), fps=2, codec="libx264", ffmpeg_params=["-threads", "1"])
    try:
        while step < 800:
            if time.time() >= job["hard_deadline"] or (root / "STOP_REQUESTED").exists():
                raise TimeoutError("experiment deadline or stop request")
            raw = policy_input(obs, residual if control.switched else full, annotation,
                               init=job["init"], query=step // 5, evaluator=c.evaluator)
            # Counterfactual arms share future seeds at every absolute replan time.
            raw["s2_rng_seed"] = job["seed"] * 1000000 + job["init"] * 1000 + step // 5
            monitored = step % 10 == 0
            before = truth(env.env, goals)
            if monitored:
                raw["monitor_prompt"] = full
                raw["monitor_only"] = True
                tick = time.monotonic()
                client._ws.send(client._packer.pack(raw))
                from skillnet_client import msgpack_numpy
                response = client._ws.recv(timeout=300)
                if isinstance(response, str):
                    raise RuntimeError(response)
                response = msgpack_numpy.unpackb(response)
                vector = causal_vector(response["monitor_feature"], raw["observation/state"],
                                       previous_state, previous_action, step)
                label = [*before["verified"], before["grasp"][0]]
                vectors.append(vector)
                labels.append(label)
                steps.append(step)
                raw_goals.append(before["raw_goals"])
                p = predict_probabilities(monitor, vector) if monitor else np.asarray(label, float)
                e = predict_effect(effect, vector, p) if effect else None
                if job["kind"] == "evaluation":
                    control.decide(label[:2] if condition == "oracle" else p[:2], step, e)
                elif job["kind"] == "pair":
                    # Fixed continuation: updates may roll back after confirmed regression.
                    states = [b.update(x) for b, x in zip(control.beliefs, p[:2])]
                    if control.switched and states[0] == "regressed":
                        control.switched = False
                        control.rollback_count += 1
                decision = dict(probabilities=p.tolist(), beliefs=[b.state for b in control.beliefs],
                                effect=e, monitor_s=time.monotonic() - tick)
                # Save pre-action states using a predeclared rule, never branch outcomes.
                if job["kind"] == "parent":
                    late = step >= 100 and verified_run[0] >= 3 or step >= 300
                    name = "early" if step == 50 else "late" if late and not snap_late else None
                    if name:
                        np.savez_compressed(directory / f"snapshot-{name}.npz", prefix=np.asarray(prefix),
                            sim_state=env.get_sim_state(), image_sha256=image_hash(raw), vector=vector,
                            previous_state=previous_state, previous_action=previous_action,
                            step=step, label=np.asarray(label), task=job["task"], init=job["init"])
                        if name == "late":
                            snap_late = True
                previous_state = raw["observation/state"].copy()
                video.append_data(np.concatenate([raw["observation/image"], raw["observation/wrist_image"]], axis=1))
            raw.pop("monitor_prompt", None)
            raw.pop("monitor_only", None)
            raw["prompt"] = residual if control.switched else full
            tick = time.monotonic()
            response = rpc(client, raw)
            append(directory / "queries.jsonl", dict(step=step, seed=raw["s2_rng_seed"], prompt=raw["prompt"],
                image_sha256=image_hash(raw), state=raw["observation/state"].tolist(), truth=before,
                decision=decision, switched=control.switched, inference_s=time.monotonic()-tick,
                actions=np.asarray(response["actions"])[:5, :7].tolist()))
            for action in np.asarray(response["actions"])[:5, :7]:
                obs, _, returned, _ = env.step(action.tolist())
                step += 1
                prefix.append(action.tolist())
                previous_action = action.copy()
                stage.observe(step, float(action[6]))
                now = truth(env.env, goals)
                verified_run = [run + 1 if value else 0 for run, value in zip(verified_run, now["verified"])]
                success, done = episode_status(env, returned)
                if done:
                    break
            if success or done:
                break
        np.savez_compressed(directory / "features.npz", vectors=np.asarray(vectors), labels=np.asarray(labels),
                            steps=np.asarray(steps), raw_goals=np.asarray(raw_goals))
        write(directory / "result.json", dict(status="complete", success=bool(success), end_step=step,
             start_step=start_step, wall_s=time.monotonic()-started, full_prompt=full, residual=residual,
             switches=control.switch_count, rollbacks=control.rollback_count, job=job,
             final_truth=truth(env.env, goals), **stage.result(success)))
    finally:
        video.close()
        client.close()
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, json.loads(args.job.read_text()))
