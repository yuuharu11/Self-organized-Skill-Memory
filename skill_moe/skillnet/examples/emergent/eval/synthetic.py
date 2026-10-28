"""Synthetic z recordings (SCHEMA.md) with known structure, for tests and smoke runs.

Motion primitives (dpos per step, gripper command) are shared across tasks; each task is a
sequence of primitives in its own scene. `z_post` / `z_prior` encode the primitive active
over the z interval (prior noisier, optionally uninformative), `z_untrained_*` are noise,
keys mix a task-specific scene vector with the current primitive.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PRIMITIVES = np.array([  # dx, dy, dz, gripper command (+1 close, -1 open)
    [0.01, 0.0, 0.0, -1.0],
    [0.0, 0.01, 0.0, -1.0],
    [0.0, 0.0, -0.01, -1.0],
    [0.0, 0.0, 0.01, 1.0],
])


def make_synthetic_z_run(run_dir: str | Path, *, n_tasks: int = 6, episodes_per_task: int = 4, horizon: int = 10,
                         replan: int = 5, segment_len: int = 20, z_dim: int = 6, prior_informative: bool = True,
                         prior_noise: float = 0.3, seed: int = 0, suites: tuple[str, ...] = ("suite_a", "suite_b"),
                         unexecuted_tail: bool = True) -> Path:
    rng = np.random.default_rng(seed)
    run_dir = Path(run_dir)
    (run_dir / "episodes").mkdir(parents=True, exist_ok=False)
    code = rng.normal(size=(len(PRIMITIVES), z_dim)) * 2
    scenes = rng.normal(size=(n_tasks, 16))
    task_plans = [rng.permutation(len(PRIMITIVES))[:3] for _ in range(n_tasks)]
    entries = []
    for t in range(n_tasks):
        suite = suites[t % len(suites)]
        for e in range(episodes_per_task):
            eid = f"{suite}_t{t:02d}_i{e:03d}"
            prims = np.repeat(task_plans[t], segment_len)
            steps_total = len(prims)
            vel = PRIMITIVES[prims]
            pos = np.concatenate([np.zeros((1, 3)), np.cumsum(vel[:, :3], axis=0)]) + rng.normal(0, 1e-4, (steps_total + 1, 3))
            closed = np.concatenate([[False], vel[:, 3] > 0])
            gap = np.where(closed, 0.0, 0.08)
            grip = np.stack([gap / 2, -gap / 2], axis=1)
            quat = np.tile([0., 0., 0., 1.], (steps_total + 1, 1))
            steps = np.arange(0, steps_total, replan)
            n = len(steps)
            n_exec = np.minimum(replan, steps_total - steps)
            if unexecuted_tail:  # one final inference whose chunk was never executed
                steps = np.append(steps, steps_total)
                n_exec = np.append(n_exec, 0)
                n += 1
            actions = np.concatenate([vel[:, :3] * 100, np.zeros((steps_total, 3)), vel[:, 3:4]], axis=1)
            chunk = np.zeros((n, horizon, 7), dtype=np.float32)
            for i, s in enumerate(steps):
                seg = actions[s:s + horizon]
                chunk[i, :len(seg)] = seg
            dominant = np.array([np.bincount(prims[s:s + horizon], minlength=len(PRIMITIVES)).argmax()
                                 if s < steps_total else 0 for s in steps])
            interval = np.clip(steps_total - steps, 0, horizon)
            executed = n_exec > 0
            full = executed & (interval == horizon)
            z_post = code[dominant] + rng.normal(0, 0.1, (n, z_dim))
            z_post[~full] = np.nan
            z_prior = (code[dominant] + rng.normal(0, prior_noise, (n, z_dim)) if prior_informative
                       else rng.normal(0, 1, (n, z_dim)) + scenes[t, :z_dim])
            proprio = np.concatenate([pos[steps], grip[steps], np.zeros((n, 3))], axis=1).astype(np.float32)
            onehot = np.eye(len(PRIMITIVES))[dominant]
            key_nolang = np.concatenate([np.tile(scenes[t], (n, 1)), 3 * onehot], axis=1) + rng.normal(0, .3, (n, 20))
            key_lang = np.concatenate([key_nolang, 3 * np.tile(np.eye(n_tasks)[t], (n, 1))], axis=1)
            np.savez(run_dir / "episodes" / f"{eid}.npz",
                     step=steps.astype(np.int32), proprio=proprio, pred_chunk=chunk,
                     n_executed=n_exec.astype(np.int32), skills=np.tile([1, 2, 0], (n, 1)).astype(np.int32),
                     skill_mask=np.tile([True, True, False], (n, 1)), segment_id=np.arange(n, dtype=np.int32),
                     env_eef_pos=pos.astype(np.float32), env_eef_quat=quat.astype(np.float32),
                     env_gripper=grip.astype(np.float32),
                     z_post_mean=z_post, z_post_logvar=np.where(full[:, None], -2.0, np.nan) * np.ones((n, z_dim)),
                     z_prior_mean=z_prior, z_prior_logvar=np.zeros((n, z_dim)),
                     z_untrained_post_mean=rng.normal(size=(n, z_dim)), z_untrained_prior_mean=rng.normal(size=(n, z_dim)),
                     z_interval_len=interval.astype(np.int32), z_executed=executed,
                     key__vlm_img_nolang=key_nolang, key__vlm_img_lang=key_lang)
            entries.append({"episode_id": eid, "task": f"task {t}", "task_id": t, "suite": suite,
                            "init_state_id": e, "success": True, "steps": int(steps_total), "env_step_offset": 0})
    manifest = {"format_version": 1, "suite": "synthetic", "replan_steps": replan, "env_step_offset": 0,
                "episodes": entries,
                "emergent": {"schema_version": 1, "z_horizon": horizon, "z_dim": z_dim, "model": "synthetic",
                             "checkpoint": "none", "posterior_input": "synthetic primitive code",
                             "prior_input": "synthetic", "key": {"vlm_img_nolang": {}, "vlm_img_lang": {}}}}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return run_dir
