#!/usr/bin/env python3
"""Resumable LIBERO websocket case runner; episodes may run in parallel."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np


class S2WebsocketClientPolicy:
    """LIBERO client with no short ping timeout during first-call JAX compilation."""

    def __init__(self, host: str, port: int):
        from skillnet_client import msgpack_numpy
        import websockets.sync.client

        self._packer = msgpack_numpy.Packer()
        uri = f"ws://{host}:{port}"
        started = time.monotonic()
        while True:
            if time.monotonic() - started > 300:
                raise TimeoutError(f"S2 websocket server did not become available at {uri}")
            try:
                self._ws = websockets.sync.client.connect(
                    uri, compression=None, max_size=None, ping_interval=None,
                    open_timeout=300, close_timeout=300,
                )
                self._server_metadata = msgpack_numpy.unpackb(self._ws.recv())
                break
            except ConnectionRefusedError:
                time.sleep(1)

    def infer(self, observation):
        from skillnet_client import msgpack_numpy
        self._ws.send(self._packer.pack(observation))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in S2 inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def close(self):
        self._ws.close()


def _suite_name(name: str) -> str:
    return {"libero90": "libero_90", "libero_90": "libero_90",
            "libero10": "libero_10", "libero_10": "libero_10",
            "libero_skill_obj": "libero_skill_obj", "libero_skill": "libero_skill"}.get(name, name)


def _load_annotation(task_suite: str, task_language: str) -> dict:
    here = Path(__file__).resolve().parents[2] / "libero" / "annotations"
    files = ("instruct2plan_obj_90.json", "libero_skill_obj_annotations.json")
    if task_suite in ("libero_skill", "libero_skill_obj"):
        files = ("libero_skill_obj_annotations.json",)
    for filename in files:
        path = here / filename
        if not path.exists():
            continue
        annotations = json.loads(path.read_text())
        if task_language in annotations:
            return annotations[task_language]
    raise KeyError(f"No task-level SkillNet annotation for {task_language!r} in {files}")


def _quat2axisangle(quat):
    quat = np.asarray(quat, np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(max(1.0 - quat[3] * quat[3], 0.0))
    if den < 1e-8:
        return np.zeros(3, np.float32)
    return (quat[:3] * (2.0 * np.arccos(quat[3])) / den).astype(np.float32)


def _append_jsonl(path: Path, row: dict):
    with path.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def completed_trial_indices(rows_path: Path) -> set[int]:
    """Return the latest complete status per index for safe interrupted resumes."""
    latest = {}
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            row = json.loads(line)
            latest[int(row["trial_index"])] = row.get("status")
    return {index for index, status in latest.items() if status == "complete"}


def _run_trial(job: dict) -> dict:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from skillnet_client import image_tools

    suite_name = _suite_name(job["suite"])
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(job["task_id"])
    language = str(task.language).strip()
    annotation = _load_annotation(suite_name, language)
    initial_states = suite.get_task_init_states(job["task_id"])
    state_index = job["trial_index"] % len(initial_states)
    initial_state = np.asarray(initial_states[state_index])
    init_digest = hashlib.sha256(np.ascontiguousarray(initial_state).view(np.uint8)).hexdigest()
    trial_seed = job["seed_base"] + job["trial_index"]
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(trial_seed)
    client = None
    traces = []
    started = time.perf_counter()
    done = False
    steps = 0
    query_id = 0
    action_plan = []
    try:
        client = S2WebsocketClientPolicy(job["host"], job["port"])
        env.reset()
        obs = env.set_init_state(initial_state)
        dummy = [0.0] * 6 + [-1.0]
        for _ in range(job["wait_steps"]):
            obs, _, done, _ = env.step(dummy)
        done = False
        while steps < job["max_steps"] and not done:
            if not action_plan:
                image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                image = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224))
                wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, 224, 224))
                element = {
                    "observation/image": image,
                    "observation/wrist_image": wrist,
                    "observation/state": np.concatenate((obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
                    "prompt": language,
                    "skills": str(annotation["all_classes"]),
                    "objects": str(annotation["objects"]) if "objects" in annotation else "[]",
                    "s2_condition": job["condition"],
                    "s2_memory_snapshot": job["memory_snapshot"],
                    "s2_query_text_source": job["query_text_source"],
                    "s2_episode_id": f"{suite_name}:{job['task_id']}:{job['trial_index']}",
                    "s2_init_state_id": f"{suite_name}:{job['task_id']}:{state_index}:{init_digest[:16]}",
                    "s2_query_id": job["trial_index"] * 1000 + query_id,
                    "s2_rng_seed": trial_seed * 100000 + query_id,
                }
                response = client.infer(element)
                chunk = np.asarray(response["actions"], dtype=np.float32)
                if chunk.ndim != 2:
                    raise ValueError(f"unexpected action chunk shape {chunk.shape}")
                # The checkpoint emits 32 padded channels; LIBERO executes xyz,
                # axis-angle xyz, and gripper only. Some legacy output transforms
                # also slice the chunk length, so validate the executable prefix.
                chunk = chunk[:, :7]
                if chunk.shape[0] < job["replan_steps"]:
                    raise ValueError(f"action chunk too short for replan: {chunk.shape}")
                action_plan.extend(chunk[:job["replan_steps"]])
                traces.append(response.get("s2_debug", {"query_id": query_id, "missing_debug": True}))
                query_id += 1
            action = action_plan.pop(0)
            obs, _, done, _ = env.step(np.asarray(action).tolist())
            steps += 1
        return {
            "trial_index": job["trial_index"], "trial_seed": trial_seed,
            "initial_state_index": state_index, "initial_state_sha256": init_digest,
            "task_suite": suite_name, "task_id": job["task_id"], "task": language,
            "condition": job["condition"], "memory_snapshot": job["memory_snapshot"],
            "checkpoint": job["checkpoint"], "query_text_source": job["query_text_source"],
            "success": bool(done), "env_steps": steps, "num_policy_queries": query_id,
            "selected_units_and_t0": traces, "elapsed_seconds": time.perf_counter() - started,
            "status": "complete",
        }
    except Exception as exc:
        return {"trial_index": job["trial_index"], "trial_seed": trial_seed,
                "initial_state_index": state_index, "initial_state_sha256": init_digest,
                "task_suite": suite_name, "task_id": job["task_id"], "task": language,
                "condition": job["condition"], "memory_snapshot": job["memory_snapshot"],
                "checkpoint": job["checkpoint"], "query_text_source": job["query_text_source"],
                "success": False, "env_steps": steps, "num_policy_queries": query_id,
                "selected_units_and_t0": traces,
                "status": "error", "error": repr(exc), "traceback": traceback.format_exc(),
                "elapsed_seconds": time.perf_counter() - started}
    finally:
        if client is not None:
            client.close()
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("S2_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("S2_PORT", "0")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("S2_WORKERS", "2")))
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--max-env-steps", type=int, default=0, help="0 uses the existing LIBERO suite limit")
    parser.add_argument("--output", default=os.environ.get("S2_OUTPUT_DIR"))
    args = parser.parse_args()
    required_env = ("S2_TASK_SUITE", "S2_TASK_ID", "S2_TASK_KEY", "S2_CONDITION",
                    "S2_MEMORY_SNAPSHOT", "S2_NUM_TRIALS", "S2_TRIAL_SEED_BASE",
                    "S2_CHECKPOINT", "S2_QUERY_TEXT_SOURCE")
    missing = [name for name in required_env if not os.environ.get(name)]
    if missing:
        raise ValueError(f"missing case environment variables: {missing}")
    if not args.output or args.port < 1 or args.workers < 1:
        raise ValueError("output, a live websocket port, and at least one worker are required")
    suite = _suite_name(os.environ["S2_TASK_SUITE"])
    task_id = int(os.environ["S2_TASK_ID"])
    trials = int(os.environ["S2_NUM_TRIALS"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "trials.jsonl"
    metadata = {"suite": suite, "task_id": task_id, "task_key": os.environ["S2_TASK_KEY"],
                "condition": os.environ["S2_CONDITION"], "memory_snapshot": os.environ["S2_MEMORY_SNAPSHOT"],
                "checkpoint": os.environ["S2_CHECKPOINT"], "trials": trials,
                "seed_base": int(os.environ["S2_TRIAL_SEED_BASE"]),
                "query_text_source": os.environ["S2_QUERY_TEXT_SOURCE"], "workers": args.workers}
    metadata_path = output / "case.json"
    if metadata_path.exists():
        if json.loads(metadata_path.read_text()) != metadata:
            raise ValueError("existing case metadata differs; refusing to resume this output directory")
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    complete = completed_trial_indices(rows_path)
    suite_obj = __import__("libero.libero.benchmark", fromlist=["get_benchmark_dict"]).get_benchmark_dict()[suite]()
    task = suite_obj.get_task(task_id)
    limits = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
              "libero_10": 520, "libero_90": 400, "libero_skill_obj": 800, "libero_skill": 800}
    max_steps = args.max_env_steps or limits[suite]
    jobs = [{"host": args.host, "port": args.port, "suite": suite, "task_id": task_id,
             "trial_index": index, "seed_base": metadata["seed_base"], "condition": metadata["condition"],
             "memory_snapshot": metadata["memory_snapshot"], "query_text_source": metadata["query_text_source"],
             "checkpoint": metadata["checkpoint"], "wait_steps": args.wait_steps,
             "replan_steps": args.replan_steps, "max_steps": max_steps}
            for index in range(trials) if index not in complete]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_run_trial, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            _append_jsonl(rows_path, row)
            if row.get("status") == "error":
                print(json.dumps({"trial_index": row["trial_index"], "error": row.get("error")} ), flush=True)
    latest_by_trial = {}
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            row = json.loads(line)
            latest_by_trial[int(row["trial_index"])] = row
    remaining_errors = sum(row.get("status") != "complete" for row in latest_by_trial.values())
    print(json.dumps({"case": metadata, "completed_before": len(complete), "scheduled": len(jobs),
                      "trial_rows": str(rows_path), "elapsed_seconds": time.perf_counter() - started}, indent=2))
    if remaining_errors:
        raise SystemExit(f"{remaining_errors} trial(s) failed; rerun this command to resume failed trials")


if __name__ == "__main__":
    main()
