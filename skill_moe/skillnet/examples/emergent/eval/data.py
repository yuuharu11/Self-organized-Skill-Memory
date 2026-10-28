"""Read z recordings (SCHEMA.md) into one flat, row-aligned table.

The reader reuses the G2 loader (`load_run`: namespaced initial-state IDs,
environment time origin, duplicate checks) and its motion target, so motion
labels and leakage rules are identical to `memory_pilot/g2_retrieval`.
NumPy only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from skill_moe.skillnet.examples.memory_pilot.g2_retrieval.evaluate import (
    Episode,
    _episode_entries,
    _motion_target,
    load_run,
    split_by_init_state,
)

Z_KEY = re.compile(r"^z_(?P<source>.+)_mean$")
KEY_PREFIX = "key__"
REQUIRED_Z = ("post", "prior")


@dataclass
class ZRun:
    """All inference rows of a run; `usable` marks rows that enter evaluation/memory."""

    run_dir: Path
    manifest: dict[str, Any]
    horizon: int
    episodes: list[Episode]
    episode_id: np.ndarray
    init_state_id: np.ndarray
    task: np.ndarray
    task_index: np.ndarray
    suite: np.ndarray
    step: np.ndarray
    row: np.ndarray            # row index inside the episode NPZ
    executed: np.ndarray       # chunk of this inference was executed
    interval_len: np.ndarray   # executed env steps in [step, step+H)
    has_future: np.ndarray     # env_* contains H observations after the row
    usable: np.ndarray
    z: dict[str, np.ndarray]
    z_logvar: dict[str, np.ndarray]
    keys: dict[str, np.ndarray]
    proprio: np.ndarray
    motion: np.ndarray         # (N, H, 7+Dg) G2 target: dpos(3), dquat xyzw(4), dgrip; NaN if not usable
    target_actions: np.ndarray | None
    target_mask: np.ndarray | None
    derived: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.step)

    def subset_mask(self, episode_ids: set[str]) -> np.ndarray:
        return np.isin(self.episode_id, sorted(episode_ids))


def _executed_action_window(arrays: dict[str, np.ndarray], row: int, horizon: int):
    """Concatenate executed `pred_chunk[:n_executed]` over [step, step+H)."""
    steps = np.asarray(arrays["step"]).astype(int)
    n_exec = np.asarray(arrays["n_executed"]).astype(int)
    chunk = np.asarray(arrays["pred_chunk"], dtype=np.float32)
    out = np.zeros((horizon, chunk.shape[2]), dtype=np.float32)
    mask = np.zeros(horizon, dtype=bool)
    start = steps[row]
    for j in range(row, len(steps)):
        offset = steps[j] - start
        if offset >= horizon:
            break
        n = min(n_exec[j], horizon - offset, chunk.shape[1])
        out[offset:offset + n] = chunk[j, :n]
        mask[offset:offset + n] = True
    return out, mask


def load_z_run(run_dir: str | Path, *, horizon: int | None = None,
               aliases: Mapping[str, str] | None = None,
               required_sources: tuple[str, ...] = REQUIRED_Z) -> ZRun:
    """Load a z recording. `aliases` maps schema keys to existing NPZ keys.

    Example for an existing record: ``{"z_post_mean": "feat__L09_mean_t_last",
    "key__vlm_img_nolang": "feat__L17_mean_t_last"}``. Missing `z_post_logvar` becomes
    zeros, missing `z_interval_len`/`z_executed` are derived from `n_executed`,
    episode `steps` and the env arrays; every substitution is listed in `derived`.
    """
    run_dir = Path(run_dir)
    manifest, episodes = load_run(run_dir)
    entries = {str(e.get("episode_id", e.get("id"))): e for e in _episode_entries(manifest)}
    meta = manifest.get("emergent", {}) or {}
    H = int(horizon if horizon is not None else meta.get("z_horizon", 0))
    if H < 1:
        raise ValueError("z horizon unknown: pass horizon= or set manifest emergent.z_horizon")
    aliases = dict(aliases or {})
    derived: dict[str, str] = {f"alias:{k}": v for k, v in aliases.items()}
    cols: dict[str, list] = {n: [] for n in ("episode_id", "init_state_id", "task", "task_index", "suite", "step", "row",
                                             "executed", "interval_len", "has_future")}
    z: dict[str, list] = {}
    z_logvar: dict[str, list] = {}
    keys: dict[str, list] = {}
    proprio, motion, actions, action_mask = [], [], [], []
    sources = None
    key_names = None
    for ep in episodes:
        arrays = ep.arrays
        for target, source in aliases.items():
            if source not in arrays:
                raise ValueError(f"{ep.episode_id}: alias source {source} missing")
            arrays[target] = arrays[source]
        ep_sources = sorted(m.group("source") for k in arrays if (m := Z_KEY.match(k)))
        ep_keys = sorted(k[len(KEY_PREFIX):] for k in arrays if k.startswith(KEY_PREFIX))
        if sources is None:
            sources, key_names = ep_sources, ep_keys
            missing = [s for s in required_sources if s not in sources]
            if missing:
                raise ValueError(f"missing z sources {missing}; found {sources}")
            if not key_names:
                derived["keys"] = "no key__* arrays; memory keys unavailable"
        if ep_sources != sources or ep_keys != key_names:
            raise ValueError(f"{ep.episode_id}: z/key arrays differ from other episodes")
        steps = np.asarray(arrays["step"])
        n = len(steps)
        entry = entries[ep.episode_id]
        n_exec = np.asarray(arrays.get("n_executed", np.ones(n, dtype=int)))
        if "z_executed" in arrays:
            executed = np.asarray(arrays["z_executed"]).astype(bool)
        else:
            executed = n_exec > 0
            derived["z_executed"] = "n_executed > 0"
        total_steps = entry.get("steps")
        if total_steps is None:
            total_steps = int(steps[-1] + n_exec[-1])
        if "z_interval_len" in arrays:
            interval = np.asarray(arrays["z_interval_len"]).astype(int)
        else:
            interval = np.clip(int(total_steps) - steps.astype(int), 0, H)
            derived["z_interval_len"] = "min(H, episode steps - step)"
        T = len(arrays["env_eef_pos"])
        env_index = steps.astype(int) - ep.env_step_offset
        has_future = (env_index >= 0) & (env_index + H < T)
        for name, arr in (("executed", executed), ("interval", interval)):
            if len(arr) != n:
                raise ValueError(f"{ep.episode_id}: {name} length differs from step")
        zrow = {}
        for s in sources:
            mean = np.asarray(arrays[f"z_{s}_mean"], dtype=np.float64)
            if mean.ndim != 2 or len(mean) != n:
                raise ValueError(f"{ep.episode_id}: z_{s}_mean must be (N, Dz)")
            lv_key = f"z_{s}_logvar"
            if lv_key in arrays:
                logvar = np.asarray(arrays[lv_key], dtype=np.float64)
            else:
                logvar = np.full_like(mean, np.nan)
                if s == "post":
                    logvar = np.zeros_like(mean)
                    derived["z_post_logvar"] = "absent; zeros (KL diagnostics meaningless)"
            if logvar.shape != mean.shape:
                raise ValueError(f"{ep.episode_id}: {lv_key} shape differs from mean")
            zrow[s] = (mean, logvar)
        krow = {}
        for kname in key_names:
            kv = np.asarray(arrays[KEY_PREFIX + kname], dtype=np.float64)
            if kv.ndim != 2 or len(kv) != n:
                raise ValueError(f"{ep.episode_id}: key__{kname} must be (N, D)")
            krow[kname] = kv
        prop = np.asarray(arrays["proprio"], dtype=np.float64)
        if "z_target_actions" in arrays:
            ta = np.asarray(arrays["z_target_actions"], dtype=np.float32)
            tm = (np.asarray(arrays["z_target_mask"]).astype(bool) if "z_target_mask" in arrays
                  else np.ones(ta.shape[:2], dtype=bool))
            if ta.ndim != 3 or ta.shape[:2] != (n, H) or tm.shape != (n, H):
                raise ValueError(f"{ep.episode_id}: z_target_actions must be (N, H, Da) with H={H}")
        elif "pred_chunk" in arrays and "n_executed" in arrays:
            pairs = [_executed_action_window(arrays, i, H) for i in range(n)]
            ta = np.stack([p[0] for p in pairs])
            tm = np.stack([p[1] for p in pairs])
            derived["z_target_actions"] = "executed pred_chunk[:n_executed] concatenated over [step, step+H)"
        else:
            ta = tm = None
        for i in range(n):
            finite = all(np.all(np.isfinite(zrow[s][0][i])) for s in sources)
            finite &= all(np.all(np.isfinite(krow[k][i])) for k in key_names)
            ok = bool(executed[i] and interval[i] == H and has_future[i] and finite)
            if executed[i] and interval[i] == H and has_future[i] and not finite:
                raise ValueError(f"{ep.episode_id} row {i}: executed full-interval row has nonfinite z/key")
            cols["episode_id"].append(ep.episode_id)
            cols["init_state_id"].append(ep.init_state_id)
            cols["task"].append(str(entry.get("task", "")).strip())
            task_index = entry.get("task_index", entry.get("task_id"))
            cols["task_index"].append(int(task_index) if task_index is not None else -1)
            if task_index is not None:
                # Historical feature caches had goal/spatial swapped in their suite fields.
                from skill_moe.skillnet.examples.emergent.light.extract import SUITE_BY_TASK
                suite = SUITE_BY_TASK.get(int(task_index), str(entry.get("suite", manifest.get("suite", ""))))
            else:
                suite = str(entry.get("suite", manifest.get("suite", "")))
            cols["suite"].append(suite)
            cols["step"].append(int(steps[i]))
            cols["row"].append(i)
            cols["executed"].append(bool(executed[i]))
            cols["interval_len"].append(int(interval[i]))
            cols["has_future"].append(bool(has_future[i]))
            for s in sources:
                z.setdefault(s, []).append(zrow[s][0][i])
                z_logvar.setdefault(s, []).append(zrow[s][1][i])
            for k in key_names:
                keys.setdefault(k, []).append(krow[k][i])
            proprio.append(prop[i])
            if has_future[i]:
                motion.append(_motion_target(arrays, int(env_index[i]), H))
            else:
                motion.append(None)
            if ta is not None:
                actions.append(ta[i])
                action_mask.append(tm[i])
            cols.setdefault("_ok", []).append(ok)
    if not cols["step"]:
        raise ValueError("run contains no inference rows")
    width = next(m.shape[1] for m in motion if m is not None) if any(m is not None for m in motion) else 8
    motion_arr = np.stack([m if m is not None else np.full((H, width), np.nan) for m in motion])
    return ZRun(
        run_dir=run_dir, manifest=manifest, horizon=H, episodes=episodes,
        episode_id=np.asarray(cols["episode_id"]), init_state_id=np.asarray(cols["init_state_id"]),
        task=np.asarray(cols["task"]), task_index=np.asarray(cols["task_index"], dtype=np.int32),
        suite=np.asarray(cols["suite"]),
        step=np.asarray(cols["step"]), row=np.asarray(cols["row"]),
        executed=np.asarray(cols["executed"]), interval_len=np.asarray(cols["interval_len"]),
        has_future=np.asarray(cols["has_future"]), usable=np.asarray(cols["_ok"]),
        z={s: np.stack(v) for s, v in z.items()}, z_logvar={s: np.stack(v) for s, v in z_logvar.items()},
        keys={k: np.stack(v) for k, v in keys.items()}, proprio=np.stack(proprio), motion=motion_arr,
        target_actions=np.stack(actions) if actions else None,
        target_mask=np.stack(action_mask) if action_mask else None, derived=derived,
    )


def split_rows(run: ZRun, *, query_fraction: float = 0.25, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """G2 split by whole (namespaced) initial-state groups -> (query_mask, store_mask) over rows."""
    query_ids, store_ids = split_by_init_state(run.episodes, query_fraction, seed)
    q, s = run.subset_mask(query_ids), run.subset_mask(store_ids)
    if set(run.init_state_id[q]) & set(run.init_state_id[s]):
        raise AssertionError("initial-state leakage between query and store rows")
    return q, s
