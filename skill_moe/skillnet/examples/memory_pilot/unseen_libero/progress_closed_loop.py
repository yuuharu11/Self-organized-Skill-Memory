"""Progress-tracked natural-time unit retrieval for LIBERO closed-loop runs.

This is a new snapshot entry point. It imports the existing MemorySelector and
websocket environment harness; the original policy and runner are unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "pi05_memory"))
import memory_closed_loop as legacy  # noqa: E402

STATE_DIR = Path(os.environ.get("SKILLNET_STATE_DIR", Path(__file__).resolve().parents[5] / ".skillnet-state")).expanduser()
ROOT = STATE_DIR / "outputs/memory-pilot/unseen-libero"
OUT = ROOT / "progress-01"
POOL = ROOT / "cluster-pools-01/pool_unit_08.npz"
POOL_CKPT = ROOT / "cluster-pools-01/query_unit_08.pt"
CHECKPOINT = STATE_DIR / "checkpoints/openpi-assets/checkpoints/pi05_libero"
BASELINE = ROOT / "baseline-01"
OLD90 = ROOT / "closed-loop-memory-01/libero90_unit_08/memory/rollouts.jsonl"
OLD10 = ROOT / "closed-loop-memory-01/libero10_unit_08/memory/rollouts.jsonl"
PI10 = ROOT / "closed-loop-memory-01/libero10/normal/rollouts.jsonl"
REPLAN = 5


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=True) + "\n")
    tmp.replace(path)


def append(path, row):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n"); f.flush(); os.fsync(f.fileno())


def terminal(path):
    import run as base
    return base.read_terminal(Path(path))


def calibrate_threshold(output: Path):
    """Set c from LIBERO-40 validation only: half the 95th percentile 10-step residual.

    Row states and actions are unnormalized with the official checkpoint quantile
    statistics. Consecutive stride-10 validation windows supply measured 10-step
    displacement and the matching sum of ten commanded xyz deltas. The half-step
    threshold is the fixed calibration rule used by the online 5-step trigger.
    """
    from openpi.shared.normalize import load
    from openpi.transforms import Unnormalize
    import importlib.util
    spec = importlib.util.spec_from_file_location("progress_pi05_data_run", HERE.parent / "pi05_memory" / "run.py")
    if spec is None or spec.loader is None: raise ImportError("cannot load pi05_memory/run.py")
    data_run = importlib.util.module_from_spec(spec); sys.modules[spec.name] = data_run; spec.loader.exec_module(data_run)
    load_rows = data_run.load_rows

    run_dir = Path(os.environ.get("SKILLNET_PI05_ROOT", STATE_DIR / "outputs/memory-pilot/pi05-memory40-20261104")) / "run-01"
    rows = load_rows(run_dir)
    ids = np.flatnonzero(np.asarray(rows["validation"], bool))
    by = {}
    for i in ids:
        by.setdefault((int(rows["task"][i]), int(rows["episode"][i])), []).append(int(i))
    normfile = next((CHECKPOINT / "assets").rglob("norm_stats.json"))
    stats = load(normfile.parent)
    unnorm = Unnormalize(stats, use_quantiles=True)
    commands, measured_values = [], []
    for inds in by.values():
        inds.sort(key=lambda i: int(rows["frame"][i]))
        for ia, ib in zip(inds, inds[1:]):
            if int(rows["frame"][ib]) - int(rows["frame"][ia]) != 10:
                continue
            s0 = np.asarray(rows["state"][ia], np.float32)
            s1 = np.asarray(rows["state"][ib], np.float32)
            an = np.asarray(rows["actions"][ia, :10, :7], np.float32)
            s0p = unnorm({"state": s0, "actions": np.zeros((1, 7), np.float32)})["state"]
            s1p = unnorm({"state": s1, "actions": np.zeros((1, 7), np.float32)})["state"]
            ap = unnorm({"state": s0, "actions": an})["actions"]
            # First three state and action coordinates are eef xyz and commanded delta xyz.
            actual = np.asarray(s1p[:3] - s0p[:3], np.float64)
            commanded = np.asarray(ap[:10, :3].sum(0), np.float64)
            commands.append(commanded); measured_values.append(actual)
    if len(commands) < 100:
        raise ValueError(f"too few validation displacement pairs for threshold: {len(commands)}")
    commands = np.asarray(commands, np.float64); measured_values = np.asarray(measured_values, np.float64)
    # Controller actions are not a literal sum of eef displacement. Fit the
    # command-to-observed map on LIBERO-40 validation, then threshold its residual.
    map_xyz = np.linalg.lstsq(commands, measured_values, rcond=None)[0]
    residuals = np.linalg.norm(measured_values - commands @ map_xyz, axis=1)
    q95 = float(np.quantile(residuals, .95))
    # Using the full 10-step p95 as a 5-step threshold is conservative, avoiding
    # frequent false track resets from simply halving a stride-10 residual.
    threshold = q95
    result = {"threshold_m": threshold, "source": "LIBERO-40 validation only", "pairs": len(residuals),
              "command_to_motion_linear_map": map_xyz.tolist(),
              "rule": "fit linear command xyz-sum -> measured xyz delta on validation; use the 10-step p95 residual norm as a conservative 5-step threshold",
              "p50_10step_residual_m": float(np.quantile(residuals, .50)), "p95_10step_residual_m": q95,
              "unseen_labels_used": False}
    write(output / "tracking_threshold.json", result)
    return result


def _interp_natural(shape, length):
    """Interpolate a 10-sample pooled unit to its stored natural frame count."""
    z = np.asarray(shape, np.float32)
    length = max(1, int(length))
    if length == 1:
        return z[[0]].copy()
    old = np.linspace(0.0, 1.0, len(z))
    new = np.linspace(0.0, 1.0, length)
    return np.stack([np.interp(new, old, z[:, j]) for j in range(z.shape[1])], axis=1).astype(np.float32)


def _segment(unit, tau, length):
    """Natural-time [tau,tau+10) segment; hold final sample beyond unit end."""
    z = _interp_natural(unit, length)
    ix = np.minimum(np.arange(int(tau), int(tau) + 10), len(z) - 1)
    return z[ix]


def _safe_grip_sign(x):
    return 1 if float(x) >= 0 else -1


def make_policy(checkpoint, threshold):
    pi05_dir = HERE.parent / "pi05_memory"
    sys.path.insert(0, str(pi05_dir))
    # `memory_closed_loop` deliberately puts its own sibling `run.py` first.
    # The π0.5 server imports a different run.py, so bind that module explicitly.
    import importlib.util
    spec = importlib.util.spec_from_file_location("run", pi05_dir / "run.py")
    if spec is None or spec.loader is None: raise ImportError("cannot load pi05_memory/run.py")
    pi05_run = importlib.util.module_from_spec(spec); sys.modules["run"] = pi05_run; spec.loader.exec_module(pi05_run)
    from serve import MemoryPolicy
    import theta_flow as tf
    import theta_regression as tr
    import variable_shape_memory as vs

    class ProgressPolicy(MemoryPolicy):
        def __init__(self, ckpt):
            super().__init__(ckpt, ckpt, normal_only=True)
            self.sel = legacy.MemorySelector()
            self.hist = {}
            self.tracks = {}
            self.threshold_info = json.loads((OUT / "tracking_threshold.json").read_text())
            self.threshold = float(threshold)
            self.track_map = np.asarray(self.threshold_info["command_to_motion_linear_map"], np.float32)
            self.pool = np.load(POOL, allow_pickle=False)
            self.shapes = np.asarray(self.pool["shape"], np.float32)
            self.lengths = np.asarray(self.pool["length"], np.int32)
            self.src_task = np.asarray(self.pool["source_task"], np.int32)
            self.tf, self.tr, self.vs = tf, tr, vs

        def _decode(self, shape, theta, natural_chunk):
            posed = self.tr.decode_theta(natural_chunk, theta, self.vs)
            return posed

        def infer(self, observation):
            from openpi.models.model import Observation
            started = time.monotonic()
            raw = {k: observation[k] for k in ["observation/image", "observation/wrist_image", "observation/state", "prompt"]}
            row = self.transform(raw)
            obs = Observation(
                images={k: self.jnp.asarray(v)[None].astype(self.jnp.float32) / 255 * 2 - 1 for k, v in row["image"].items()},
                image_masks={k: self.jnp.asarray([v]) for k, v in row["image_mask"].items()},
                state=self.jnp.asarray(row["state"])[None],
                tokenized_prompt=self.jnp.asarray(row["tokenized_prompt"])[None],
                tokenized_prompt_mask=self.jnp.asarray(row["tokenized_prompt_mask"])[None])
            seed = int(observation["s2_rng_seed"])
            normal = np.asarray(self.normal(self.jax.random.key(seed), obs, num_steps=10))[0].astype(np.float32)
            if observation["s2_condition"] == "normal":
                actions = self.unnormalize({"actions": normal, "state": row["state"]})["actions"][:, :7]
                return {"actions": actions.astype(np.float32), "s2_debug": {"condition": "normal", "seconds": time.monotonic() - started}}

            suite, task_id = observation["s2_episode_id"].split(":")[:2]
            ep = observation["s2_episode_id"]
            # Paired conditions can have identical suite/task/init keys while
            # being independent simulator trajectories served concurrently.
            track_key = ep + "|" + str(observation["s2_condition"])
            qid = int(observation["s2_query_id"]) % 1000
            if qid == 0:
                self.hist[track_key] = []
                self.tracks[track_key] = None
            executed = self.hist.setdefault(track_key, [])
            hist = np.stack(executed[-10:]) if len(executed) >= 10 else np.zeros((10, 7), np.float32)
            state_norm = np.asarray(row["state"], np.float32)[:8]
            exclude = legacy.LONG_TO_40[int(task_id)] if suite == "libero_10" else None
            track = self.tracks.get(track_key)
            reason = None
            residual = None
            if track is not None:
                if int(track["tau"]) >= int(track["length"]):
                    reason = "a"
                grip_sign = _safe_grip_sign(normal[0, 6])
                if reason is None and track.get("last_grip_sign") is not None and grip_sign != track["last_grip_sign"]:
                    reason = "b"
                raw_xyz = np.asarray(observation["observation/state"], np.float32)[:3]
                if reason is None and track.get("prev_raw_xyz") is not None and track.get("prev_command_xyz") is not None:
                    measured = raw_xyz - track["prev_raw_xyz"]
                    residual = float(np.linalg.norm(measured - track["prev_command_xyz"] @ self.track_map))
                    if residual > self.threshold:
                        reason = "c"
            searched = track is None or reason is not None
            if searched:
                idx = int(self.sel.pick(hist[None], state_norm[None], exclude)[0])
                length = max(1, int(self.lengths[idx]))
                theta_fix = self.tf.batched_procrustes(self.shapes[idx:idx+1, :, :6],
                    self.vs.to_physical(normal[None, :, :6]))[0][0]
                track = {"unit": idx, "length": length, "tau": 0, "theta_fix": theta_fix.astype(np.float32),
                         "last_grip_sign": None, "prev_raw_xyz": None, "prev_command_xyz": None,
                         "exit_reason": None}
                self.tracks[track_key] = track
            idx, length, tau = int(track["unit"]), int(track["length"]), int(track["tau"])
            src = _segment(self.shapes[idx], tau, length)
            if observation["s2_condition"] == "progress_chunk":
                theta = self.tf.batched_procrustes(src[None, :, :6], self.vs.to_physical(normal[None, :, :6]))[0][0]
            else:
                theta = np.asarray(track["theta_fix"], np.float32)
            mem = self._decode(self.shapes[idx], theta, src)
            out = normal.copy(); out[:, :6] = mem[:, :6]  # gripper remains the PI0.5 output
            actions = self.unnormalize({"actions": out, "state": row["state"]})["actions"][:, :7]
            if not np.isfinite(actions).all():
                raise FloatingPointError("nonfinite action")
            command_xyz = np.asarray(actions[:REPLAN, :3].sum(0), np.float32)
            raw_xyz = np.asarray(observation["observation/state"], np.float32)[:3].copy()
            track["tau"] = tau + REPLAN
            track["last_grip_sign"] = _safe_grip_sign(normal[0, 6])
            track["prev_raw_xyz"] = raw_xyz
            track["prev_command_xyz"] = command_xyz
            dbg = {"condition": observation["s2_condition"], "unit": idx, "length": length,
                   "query_id": qid, "tau": tau, "tau_after": tau + REPLAN, "searched": searched,
                   "switch_reason": ("initial" if qid == 0 and reason is None else reason),
                   "theta_scale": float(np.exp(np.clip(float(theta[3]), -5, 5))),
                   "tracking_residual_m": residual, "tracking_threshold_m": self.threshold,
                   "seconds": time.monotonic() - started}
            return {"actions": actions.astype(np.float32), "s2_debug": dbg}

    return ProgressPolicy(checkpoint)


def serve(args):
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    policy = make_policy(args.checkpoint, args.threshold)
    WebsocketPolicyServer(policy, host="127.0.0.1", port=args.port,
        metadata={"policy": "progress-tracked-unit", "theta_modes": ["progress_chunk", "progress_fix"]}).serve_forever()


def _free_port():
    import run as base
    return base.free_port()


def _run_trial(job, folder, timeout, cores):
    import run as base
    folder.mkdir(parents=True, exist_ok=False); write(folder / "job.json", job)
    env = os.environ.copy(); env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    with (folder / "worker.log").open("w") as log:
        p = subprocess.Popen(["taskset", "-c", cores, "nice", "-n", "10", sys.executable,
            str(HERE / "worker.py"), str(folder / "job.json"), str(folder / "result.json")],
            env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            base.terminate(p)
            return dict(job, status="timeout", success=False, error="trial wall timeout", attempt_dir=str(folder))
    if p.returncode or not (folder / "result.json").exists():
        return dict(job, status="error", success=False, error=f"worker exit {p.returncode}", attempt_dir=str(folder))
    r = json.loads((folder / "result.json").read_text()); r["attempt_dir"] = str(folder)
    return r


def run(args):
    import run as base
    output = args.output; output.mkdir(parents=True, exist_ok=True)
    plan = json.loads((BASELINE / "plan.json").read_text())
    tasks = [int(x["task_id"]) for x in plan["selected_tasks"]] if args.suite == "libero_90" else list(range(10))
    if args.task_ids:
        wanted = {int(x) for x in args.task_ids.split(",")}
        tasks = [t for t in tasks if t in wanted]
    conds = args.conditions.split(",")
    done = {c: terminal(output / c / "rollouts.jsonl") for c in conds}
    servers, logs = [], []
    for i in range(args.servers):
        port = _free_port(); folder = output / "servers" / str(i); folder.mkdir(parents=True, exist_ok=True)
        log = (folder / "server.log").open("a"); logs.append(log)
        cores = "8-11" if i == 0 else "12-15"
        env = os.environ.copy(); env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION=".35")
        p = subprocess.Popen(["taskset", "-c", cores, "nice", "-n", "10", sys.executable, str(HERE / "progress_closed_loop.py"),
            "serve", "--checkpoint", str(args.checkpoint), "--port", str(port), "--threshold", str(args.threshold)],
            env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        servers.append((p, port))
    try:
        for p, port in servers: base.ready(p, port)
        jobs = [(c, t, i) for c in conds for t in tasks for i in range(args.inits)
                if (t, i) not in terminal(output / c / "rollouts.jsonl")]
        total = len(tasks) * args.inits
        lane_cores = ["16-23", "24-31", "32-39", "40-47"]
        def execute(cond, task, init, slot):
            p, port = servers[slot % len(servers)]
            job = {"host": "127.0.0.1", "port": port, "suite": args.suite, "task_id": task, "trial_index": init,
                   "seed_base": 7, "condition": cond, "memory_snapshot": "none", "query_text_source": "instruction_only",
                   "checkpoint": str(args.checkpoint), "wait_steps": 10, "replan_steps": 5, "max_steps": 520, "record": init < 2}
            name = f"task{task:02d}-init{init:02d}"; root = output / cond / "attempts" / name
            exists = list(root.glob("attempt-*")) if root.exists() else []
            start = max([int(z.name.split("-")[-1]) for z in exists], default=0) + 1
            result = None
            for attempt in range(start, start + 2):
                result = _run_trial(job, root / f"attempt-{attempt}", args.timeout, lane_cores[slot % 4])
                append(output / cond / "attempts.jsonl", result)
                if result["status"] == "complete": break
            result.update(task_id=task, trial_index=init, task_suite=args.suite)
            append(output / cond / "rollouts.jsonl", result)
            data = terminal(output / cond / "rollouts.jsonl")
            write(output / cond / "progress.json", {"processed": len(data), "scheduled": total,
                "successes": sum(x.get("status") == "complete" and x.get("success", False) for x in data.values()), "updated_unix": time.time()})
        def lane(slot):
            for j, (cond, task, init) in enumerate(jobs[slot::4]):
                p, _ = servers[slot % len(servers)]
                if p.poll() is not None: raise RuntimeError(f"server {slot % len(servers)} died")
                execute(cond, task, init, slot)
        with ThreadPoolExecutor(max_workers=4) as ex:
            for f in as_completed([ex.submit(lane, i) for i in range(4)]): f.result()
    finally:
        for p, _ in servers: base.terminate(p)
        for l in logs: l.close()
    report(output, args.suite, args.inits)


def _mcnemar(a, b):
    from scipy.stats import binomtest
    keys = set(a) & set(b)
    success = lambda x: x.get("status") == "complete" and bool(x.get("success"))
    both = sum(success(a[k]) and success(b[k]) for k in keys)
    ao = sum(success(a[k]) and not success(b[k]) for k in keys)
    bo = sum(success(b[k]) and not success(a[k]) for k in keys)
    return {"n": len(keys), "both_success": both, "a_only": ao, "b_only": bo,
            "both_fail": len(keys)-both-ao-bo,
            "exact_mcnemar_p": float(binomtest(min(ao,bo), ao+bo, .5, alternative="two-sided").pvalue) if ao+bo else 1.0}


def _success(x): return x.get("status") == "complete" and bool(x.get("success"))


def _reselect(path):
    d = terminal(path); total = same = 0
    for r in d.values():
        traces = r.get("selected_units_and_t0", [])
        previous = None
        for q in traces:
            unit = q.get("unit")
            if unit is None: continue
            # Legacy unit_08 selected at every query; new policy marks searches.
            searched = bool(q.get("searched", True))
            if searched and previous is not None:
                total += 1; same += int(unit == previous)
            previous = unit
    return {"adjacent_query_pairs": total, "same_unit_pairs": same, "fraction": same/total if total else None}


def report(output, suite="libero_90", inits=10):
    conds = [p.name for p in output.iterdir() if p.is_dir() and (p / "rollouts.jsonl").exists()]
    result = {c: terminal(output / c / "rollouts.jsonl") for c in conds}
    reference = terminal(BASELINE / "production/rollouts.jsonl") if suite == "libero_90" else terminal(PI10)
    old_memory = terminal(OLD90 if suite == "libero_90" else OLD10)
    lines = ["# LIBERO 閉ループ：自然長のスキル進捗管理", "",
             "未学習 LIBERO-90 ではタスク固有の正解行動を選択・姿勢・追従閾値に使用していない。追従閾値は LIBERO-40 validation のみから固定した。", "",
             "| 条件 | 成功 | PI0.5のみ/条件のみ (p) | unit_08のみ/条件のみ (p) | 検索/試行 | 完了検索/新規単位 | 同一単位再選択 |", "|---|---:|---:|---:|---:|---:|---:|"]
    stats = {}
    for c in conds:
        rs = result[c]; s = sum(_success(x) for x in rs.values())
        pair = _mcnemar(rs, reference)
        old_pair = _mcnemar(rs, old_memory)
        queries = [q for r in rs.values() for q in r.get("selected_units_and_t0", [])]
        searches = [q for q in queries if q.get("searched")]
        reasons = {}
        for q in searches: reasons[q.get("switch_reason", "?")] = reasons.get(q.get("switch_reason", "?"), 0)+1
        # A unit is completed if subsequent retrieval was caused by the natural-end trigger.
        ends = sum(q.get("switch_reason") == "a" for q in searches)
        pair_reselect = _reselect(output / c / "rollouts.jsonl")
        lines.append(f"| {c} | {s}/{len(rs)} | {pair['b_only']}/{pair['a_only']} (p={pair['exact_mcnemar_p']:.3g}) | {old_pair['b_only']}/{old_pair['a_only']} (p={old_pair['exact_mcnemar_p']:.3g}) | {len(searches)/max(len(rs),1):.2f} | {ends}/{max(len(searches),1)} ({ends/max(len(searches),1):.1%}) | {pair_reselect['same_unit_pairs']}/{pair_reselect['adjacent_query_pairs']} ({pair_reselect['fraction']:.1%}) |")
        stats[c] = {"success": s, "against_pi05": pair, "against_unit08": old_pair, "queries": len(queries), "search_count": len(searches),
                    "searches_per_trial": len(searches)/max(len(rs),1), "search_reasons": reasons,
                    "unit_completion_proxy_fraction": ends/max(len(searches),1), "reselect": pair_reselect,
                    "task_success": {str(t): {"n": sum(k[0] == t for k in rs), "success": sum(_success(v) for k,v in rs.items() if k[0] == t)} for t in sorted({k[0] for k in rs})}}
    threshold = json.loads((OUT / "tracking_threshold.json").read_text())
    lines += ["", "## 追跡閾値", "", f"LIBERO-40 validation の連続stride-10窓 {threshold['pairs']:,}組で指令→実変位の線形写像を当て、残差 p95={threshold['p95_10step_residual_m']:.4f} m を保守的な5ステップ閾値 {threshold['threshold_m']:.4f} m とした。LIBERO-90ラベルは不使用。", "",
              "| 条件 | 切り替え理由の内訳 |", "|---|---|"]
    for c, st in stats.items(): lines.append(f"| {c} | `{json.dumps(st['search_reasons'], ensure_ascii=False, sort_keys=True)}` |")
    lines += ["",
              "## 現行 unit_08 の同一単位再選択", "",
              f"LIBERO-90: `{_reselect(OLD90)['same_unit_pairs']}/{_reselect(OLD90)['adjacent_query_pairs']}` = {_reselect(OLD90)['fraction']:.1%}; LIBERO-10: `{_reselect(OLD10)['same_unit_pairs']}/{_reselect(OLD10)['adjacent_query_pairs']}` = {_reselect(OLD10)['fraction']:.1%}.", "",
              "検索理由・試行ごとの単位完了詳細は各rolloutの `selected_units_and_t0` と summary.json に保存。単位完了率は自然終端で次回検索された割合を検索イベントから算出した。", ""]
    write(output / "summary.json", {"suite": suite, "conditions": stats, "baseline_source": str(BASELINE if suite == "libero_90" else PI10)})
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def snapshot(out, dirname="snapshot"):
    """Freeze relevant source folders for all later execution; refuse overwrite."""
    dst = out / dirname
    if dst.exists(): raise FileExistsError(dst)
    src = HERE.parents[0]  # examples/memory_pilot
    dst.mkdir(parents=True)
    for folder in ["unseen_libero", "pi05_memory", "anchor_flow"]:
        shutil.copytree(src / folder, dst / folder, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    snap_script = dst / "unseen_libero/progress_closed_loop.py"
    # this file is already in the copied tree; record immutable hashes.
    import hashlib
    files = sorted(p for p in dst.rglob("*.py"))
    write(out / "snapshot_manifest.json", {"source": str(HERE), "files": len(files),
        "sha256": {str(p.relative_to(dst)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}})
    return snap_script


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("calibrate"); c.add_argument("--output", type=Path, default=OUT)
    ss = sp.add_parser("serve"); ss.add_argument("--checkpoint", type=Path, required=True); ss.add_argument("--port", type=int, required=True); ss.add_argument("--threshold", type=float, required=True)
    r = sp.add_parser("run"); r.add_argument("--output", type=Path, required=True); r.add_argument("--checkpoint", type=Path, required=True); r.add_argument("--threshold", type=float, required=True); r.add_argument("--suite", default="libero_90"); r.add_argument("--conditions", default="progress_chunk,progress_fix"); r.add_argument("--servers", type=int, default=2); r.add_argument("--timeout", type=int, default=900); r.add_argument("--inits", type=int, default=10); r.add_argument("--task-ids", default="")
    rp = sp.add_parser("report"); rp.add_argument("--output", type=Path, required=True); rp.add_argument("--suite", default="libero_90"); rp.add_argument("--inits", type=int, default=10)
    sn = sp.add_parser("snapshot"); sn.add_argument("--output", type=Path, default=OUT); sn.add_argument("--dirname", default="snapshot")
    a = p.parse_args()
    if a.cmd == "calibrate": print(json.dumps(calibrate_threshold(a.output), indent=2))
    elif a.cmd == "serve": serve(a)
    elif a.cmd == "run": run(a)
    elif a.cmd == "report": report(a.output, a.suite, a.inits)
    elif a.cmd == "snapshot": print(snapshot(a.output, a.dirname))


if __name__ == "__main__": main()
