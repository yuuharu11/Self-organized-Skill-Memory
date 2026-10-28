"""Record LIBERO rollouts for offline memory-pilot experiments."""
import argparse
import dataclasses
import datetime as dt
import json
import pathlib
import uuid

import imageio
import jax
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from skillnet_client import image_tools

from openpi import transforms
from openpi.policies.policy_config import create_trained_policy
from openpi.training import config_moe_skill

ROOT = pathlib.Path(__file__).resolve().parents[4]
STATE = pathlib.Path(__import__("os").environ.get("SKILLNET_STATE_DIR", ROOT / ".skillnet-state"))
ANNOTATION_DIR = pathlib.Path(__file__).resolve().parents[1] / "libero" / "annotations"


def fresh_run_dir(tag):
    base = STATE / "outputs" / "memory-pilot"
    base.mkdir(parents=True, exist_ok=True)
    for _ in range(20):
        path = base / f"{tag}-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
        try:
            path.mkdir()
            (path / "episodes").mkdir()
            return path
        except FileExistsError:
            pass
    raise RuntimeError("could not allocate a unique output directory")


def _suite_skill_ids(suite, annotation):
    """Use all tasks in this suite, rather than the multi-suite annotation file."""
    return sorted({int(skill_id) for task_id in range(suite.n_tasks)
                   for skill_id in annotation[suite.get_task(task_id).language.strip()]["all_classes"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-ids", type=int, nargs="+", default=[0])
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=220)
    ap.add_argument("--replan-steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--run-dir", type=pathlib.Path)
    args = ap.parse_args()
    config = config_moe_skill.get_config("pi05_libero_moe_skill_4_40")
    checkpoint = pathlib.Path(__import__("os").environ.get("SKILLNET_MEMORY_PILOT_CHECKPOINT", STATE / "checkpoints" / config.name / "moe_add_balance_4_128_30000" / "29999"))
    config = dataclasses.replace(config, exp_name=checkpoint.parent.name)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    policy = create_trained_policy(config, checkpoint, repack_transforms=transforms.Group(), sample_kwargs={"num_steps": 10, "capture_auxiliary": True})
    annotation_file = ANNOTATION_DIR / ("libero_skill_obj_annotations.json" if "skill_obj" in args.suite or args.suite == "libero_skill" else "instruct2plan_40.json")
    annotation = json.loads(annotation_file.read_text())
    suite = benchmark.get_benchmark_dict()[args.suite]()
    suite_skill_ids = _suite_skill_ids(suite, annotation)
    run_dir = args.run_dir or fresh_run_dir("record")
    if args.run_dir:
        run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "episodes").mkdir(exist_ok=True)
    manifest = {
        "format_version": 1, "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checkpoint": str(checkpoint), "config": config.name, "suite": args.suite,
        "seed": args.seed, "replan_steps": args.replan_steps,
        # env_* contains only post-action observations: array index 0 is env step 1.
        "env_step_offset": 1,
        "segment_definition": "one segment per policy inference call",
        "hidden_features": {"enabled": True, "layers": [2, 9, 17], "timestep": "t_last: approximately 0.1, before the final Euler update (10 steps)", "aggregation": "mean over action suffix tokens; post-block residual; zero-based layer indices"},
        "suite_skill_ids": suite_skill_ids,
        "episodes": [],
    }
    for task_id in args.task_ids:
        task = suite.get_task(task_id)
        desc = task.language.strip()
        ann = annotation[desc]
        skill_ids = np.asarray(ann["all_classes"], np.int32)
        task_states = suite.get_task_init_states(task_id)
        env = OffScreenRenderEnv(bddl_file_name=pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file,
                                 camera_heights=256, camera_widths=256)
        env.seed(args.seed)
        for ep in range(args.trials):
            episode_id = f"{args.suite}_t{task_id:02d}_i{ep:03d}_{uuid.uuid4().hex[:6]}"
            obs = env.reset()
            obs = env.set_init_state(task_states[ep])
            image_rows, wrist_rows, states, prompts = [], [], [], []
            chunks, steps, n_exec, segments = [], [], [], []
            features, router_top1 = [], []
            env_pos, env_quat, env_grip = [], [], []
            call_pos, call_quat, call_grip = [], [], []
            t, done, queued = 0, False, []
            current_record = None
            while t < args.max_steps and not done:
                if t < 10:
                    obs, _, done, _ = env.step([0.0] * 6 + [-1.0]); t += 1
                    env_pos.append(obs["robot0_eef_pos"].copy()); env_quat.append(obs["robot0_eef_quat"].copy()); env_grip.append(obs["robot0_gripper_qpos"].copy())
                    continue
                if not queued:
                    base = image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), 224, 224))
                    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]), 224, 224))
                    state = np.concatenate((obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])).astype(np.float32)
                    raw = {"observation/image": base, "observation/wrist_image": wrist, "observation/state": state,
                           "prompt": desc, "skills": json.dumps(skill_ids.tolist())}
                    inference = policy.infer(raw)
                    chunk = np.asarray(inference["actions"], np.float32)
                    index = len(chunks)
                    # step is the observation's environment step number. Readers
                    # subtract env_step_offset once to obtain the env_* row.
                    chunks.append(chunk); steps.append(t); n_exec.append(0); segments.append(index)
                    layer_states = np.asarray(inference["memory_pilot_features"], np.float32)
                    router = np.asarray(inference["memory_pilot_router_top1"], np.int32)
                    features.append([layer_states[layer_id].mean(axis=0) for layer_id in (2, 9, 17)])
                    router_top1.append(router)
                    image_rows.append(base); wrist_rows.append(wrist); states.append(state); prompts.append(desc)
                    call_pos.append(obs["robot0_eef_pos"].copy()); call_quat.append(obs["robot0_eef_quat"].copy()); call_grip.append(obs["robot0_gripper_qpos"].copy())
                    current_record = index
                    queued = list(chunk[:args.replan_steps])
                act = queued.pop(0)
                obs, _, done, _ = env.step(act.tolist())
                n_exec[current_record] += 1
                t += 1
                env_pos.append(obs["robot0_eef_pos"].copy()); env_quat.append(obs["robot0_eef_quat"].copy()); env_grip.append(obs["robot0_gripper_qpos"].copy())
            success = bool(done)
            skill_tokens = np.zeros(max(4, len(skill_ids)), np.int32)
            skill_tokens[:len(skill_ids)] = skill_ids + 1
            skill_mask = np.arange(len(skill_tokens)) < len(skill_ids)
            arrays = dict(step=np.asarray(steps, np.int32), proprio=np.asarray(states, np.float32),
                          pred_chunk=np.asarray(chunks, np.float32), n_executed=np.asarray(n_exec, np.int32),
                          skills=np.broadcast_to(skill_tokens, (len(steps), len(skill_tokens))).copy(),
                          skill_mask=np.broadcast_to(skill_mask, (len(steps), len(skill_tokens))).copy(), segment_id=np.asarray(segments, np.int32),
                          eef_pos=np.asarray(call_pos, np.float32), eef_quat=np.asarray(call_quat, np.float32), gripper=np.asarray(call_grip, np.float32),
                          env_eef_pos=np.asarray(env_pos, np.float32), env_eef_quat=np.asarray(env_quat, np.float32),
                          env_gripper=np.asarray(env_grip, np.float32), img_base=np.asarray(image_rows, np.uint8),
                          img_wrist=np.asarray(wrist_rows, np.uint8), prompt=np.asarray(prompts, dtype="U256"))
            for feature_index, layer_id in enumerate((2, 9, 17)):
                arrays[f"feat__L{layer_id:02d}_mean_t_last"] = np.asarray([row[feature_index] for row in features], np.float32)
            arrays["router_top1"] = np.asarray(router_top1, np.int32)
            np.savez_compressed(run_dir / "episodes" / f"{episode_id}.npz", **arrays)
            manifest["episodes"].append({"episode_id": episode_id, "task": desc, "task_id": task_id, "suite": args.suite,
                                         "init_state_id": ep, "success": success, "steps": t, "inferences": len(steps)})
            print(f"{episode_id}: success={success}, env_steps={t}, chunks={len(steps)}")
        env.close()
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(run_dir)


def _quat2axisangle(q):
    q = np.array(q, dtype=np.float64, copy=True); q[3] = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(1.0 - q[3] * q[3])
    return np.zeros(3) if np.isclose(den, 0) else q[:3] * (2 * np.arccos(q[3])) / den

if __name__ == "__main__": main()
