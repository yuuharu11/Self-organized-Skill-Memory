"""RoboCasa websocket evaluator; simulator runs in the RoboCasa venv."""
import argparse
import datetime
import json
import os
import pathlib
import subprocess
import signal
import sys
import time

import numpy as np
import gymnasium as gym
from openpi_client import image_tools, websocket_client_policy
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_task_horizon
from robocasa.utils.env_utils import convert_action


def official_state(obs):
    """Exact state concatenation used by robocasa-benchmark/openpi main.py."""
    names = ("state.end_effector_position_relative", "state.end_effector_rotation_relative",
             "state.base_position", "state.base_rotation", "state.gripper_qpos")
    value = np.concatenate([np.asarray(obs[k]).reshape(-1) for k in names]).astype(np.float32)
    if value.shape != (16,):
        raise ValueError(f"expected RoboCasa state[16], got {value.shape}")
    return value


def policy_observation(obs, resize=224):
    def image(name):
        x = image_tools.resize_with_pad(np.asarray(obs[name]), resize, resize)
        return image_tools.convert_to_uint8(x)
    return {
        "observation/image": image("video.robot0_agentview_left"),
        "observation/right_image": image("video.robot0_agentview_right"),
        "observation/wrist_image": image("video.robot0_eye_in_hand"),
        "observation/state": official_state(obs),
        "prompt": obs["annotation.human.task_description"],
    }


def _run_worker(args, tasks):
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port, keepalive=False)
    shard = pathlib.Path(args.output) / "workers" / f"worker-{args.worker_index:02d}.json"
    if args.resume:
        from memory_server_utils import completed_trials_from_shard
        completed, records = completed_trials_from_shard(shard)
        # Completed trials are global identities, not shard positions. This
        # also allows resume after the launcher adjusts worker count to load.
        for old_shard in (pathlib.Path(args.output) / "workers").glob("worker-*.json"):
            old_completed, _ = completed_trials_from_shard(old_shard)
            completed.update(old_completed)
    else:
        completed, records = set(), []
    client_id = f"{os.uname().nodename}:{os.getpid()}:worker-{args.worker_index}"
    seed_task_indices = args.seed_task_indices or list(range(len(tasks)))
    if len(seed_task_indices) != len(tasks):
        raise ValueError("--seed-task-indices must have one entry per task")
    for task_position, task in enumerate(tasks):
        task_index = int(seed_task_indices[task_position])
        horizon = get_task_horizon(task)
        env = gym.make("robocasa/" + task, split=args.split, seed=args.seed)
        try:
            for trial in range(args.worker_index, args.num_trials, args.worker_count):
                seed = args.seed + task_index * int(args.seed_task_stride or args.num_trials) + trial
                if (task, trial) in completed:
                    continue
                obs, _ = env.reset(seed=seed)
                queue = []
                episode_id = f"{task}:{trial}:seed-{seed}"
                episode_start = True
                success = False
                steps = 0
                frames = []
                for steps in range(1, horizon + 1):
                    if not queue:
                        request = policy_observation(obs, args.resize)
                        request.update({"client_id": client_id, "episode_id": episode_id,
                                        "episode_start": episode_start})
                        result = client.infer(request)
                        episode_start = False
                        actions = np.asarray(result["actions"])
                        if len(actions) < args.replan_steps:
                            raise ValueError(f"policy returned {len(actions)} actions, need {args.replan_steps}")
                        queue = list(actions[:args.replan_steps])
                    obs, _, _, _, info = env.step(convert_action(np.asarray(queue.pop(0))))
                    success = bool(info.get("success", False))
                    if args.save_video and trial < args.video_trials:
                        if steps % 2 == 0 or success or steps == horizon:
                            frame = env.render()
                            if frame is not None:
                                frames.append(np.ascontiguousarray(frame))
                    if success:
                        break
                if frames:
                    import imageio.v2 as imageio
                    video_dir = pathlib.Path(args.output) / "videos"
                    video_dir.mkdir(parents=True, exist_ok=True)
                    imageio.mimsave(video_dir / f"{task}_trial{trial:03d}_seed{seed}.mp4", frames, fps=10)
                rec = {"task": task, "trial": trial, "seed": seed,
                       "success": success, "steps": steps, "status": "complete"}
                records.append(rec)
                shard.parent.mkdir(parents=True, exist_ok=True)
                shard_tmp = shard.with_name(f"{shard.name}.tmp-{os.getpid()}")
                shard_tmp.write_text(json.dumps(records, indent=2) + "\n")
                os.replace(shard_tmp, shard)
                completed.add((task, trial))
                print(json.dumps(rec), flush=True)
        finally:
            env.close()
    return records


def _wilson(successes, n, z=1.959963984540054):
    if not n:
        return [None, None]
    p = successes / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n))/denom
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    return [center-half, center+half]


def main():
    try:
        import torch
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    except ImportError:
        pass
    p = argparse.ArgumentParser()
    p.add_argument('--host', default='127.0.0.1'); p.add_argument('--port', type=int, default=8000)
    p.add_argument('--task-set', default='atomic_seen'); p.add_argument('--tasks', nargs='*')
    p.add_argument('--split', default='pretrain', choices=('pretrain','target'))
    p.add_argument('--num-trials', type=int, default=50); p.add_argument('--replan-steps', type=int, default=5)
    p.add_argument('--output', required=True); p.add_argument('--seed', type=int, default=7)
    p.add_argument('--seed-task-stride', type=int, help='seed offset per canonical task index; defaults to num-trials')
    p.add_argument('--seed-task-indices', type=int, nargs='*', help='canonical task positions used in seed formula')
    p.add_argument('--resize', type=int, default=224); p.add_argument('--workers', type=int, default=1)
    p.add_argument('--worker-index', type=int); p.add_argument('--worker-count', type=int, default=1)
    p.add_argument('--resume', action='store_true', help='skip complete task/trial records in worker shards')
    p.add_argument('--save-video', action='store_true'); p.add_argument('--video-trials', type=int, default=2)
    a = p.parse_args()
    tasks = a.tasks or TASK_SET_REGISTRY[a.task_set]
    out = pathlib.Path(a.output)
    if a.worker_index is not None:
        _run_worker(a, tasks)
        return
    # The orchestrator may create an empty output directory before invoking us.
    # Permit only that case; never silently reuse a directory with prior results.
    if out.exists():
        if not out.is_dir() or (any(out.iterdir()) and not a.resume):
            raise FileExistsError(f"refusing to overwrite non-empty output directory: {out}")
    else:
        out.mkdir(parents=True, exist_ok=False)
    if a.workers < 1:
        raise ValueError("workers must be >=1")
    if a.workers == 1:
        a.worker_index, a.worker_count = 0, 1
        _run_worker(a, tasks)
    else:
        procs = []
        script = str(pathlib.Path(__file__).resolve())
        base = [sys.executable, script, '--host', a.host, '--port', str(a.port), '--split', a.split,
                '--num-trials', str(a.num_trials), '--replan-steps', str(a.replan_steps),
                '--output', str(out), '--seed', str(a.seed), '--resize', str(a.resize),
                '--worker-count', str(a.workers)]
        if a.seed_task_stride is not None: base += ['--seed-task-stride', str(a.seed_task_stride)]
        if a.seed_task_indices is not None: base += ['--seed-task-indices', *map(str, a.seed_task_indices)]
        if a.save_video: base.append('--save-video')
        if a.resume: base.append('--resume')
        base += ['--video-trials', str(a.video_trials)]
        if a.tasks: base += ['--tasks', *a.tasks]
        else: base += ['--task-set', a.task_set]
        for i in range(a.workers):
            procs.append(subprocess.Popen(base + ['--worker-index', str(i)]))
        previous_sigterm = signal.getsignal(signal.SIGTERM)
        def stop_workers_on_sigterm(_signum, _frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, stop_workers_on_sigterm)
        try:
            codes = [proc.wait() for proc in procs]
        except BaseException:
            for proc in procs:
                if proc.poll() is None:
                    proc.terminate()
            for proc in procs:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            raise
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
        if any(codes):
            raise RuntimeError(f"worker failures: exit codes {codes}; partial shards are preserved in {out}/workers")
    from memory_server_utils import merge_worker_shards
    records = merge_worker_shards(out / 'workers')
    records.sort(key=lambda r: (tasks.index(r['task']), r['trial']))
    per_task = {}
    for task in tasks:
        rs = [r for r in records if r['task'] == task]
        s = sum(bool(r['success']) for r in rs)
        per_task[task] = {"successes": s, "trials": len(rs), "rate": s/len(rs) if rs else None,
                          "wilson95": _wilson(s, len(rs))}
    success_count = sum(bool(r['success']) for r in records)
    repo_root = pathlib.Path(__file__).resolve().parents[4]
    try:
        commit_id = subprocess.check_output(['git','-C',str(repo_root),'rev-parse','HEAD'],text=True).strip()
    except Exception:
        commit_id = 'unknown'
    result = {"split": a.split, "task_set": a.task_set, "tasks": tasks, "num_trials": a.num_trials,
              "seed_rule": "seed + canonical_task_index*seed_task_stride + trial",
              "seed_task_stride": a.seed_task_stride or a.num_trials,
              "seed_task_indices": a.seed_task_indices or list(range(len(tasks))), "seed": a.seed,
              "replan_steps": a.replan_steps, "policy": f"ws://{a.host}:{a.port}", "records": records,
              "success_rate": success_count/len(records) if records else None,
              "wilson95": _wilson(success_count, len(records)), "per_task": per_task,
              "checkpoint": os.environ.get("SKILLNET_ROBOCASA_CHECKPOINT"),
              "commit_id": commit_id, "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "config": vars(a)}
    (out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print('output', out)


if __name__ == '__main__':
    main()
