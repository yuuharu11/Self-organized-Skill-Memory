"""RES1 closed-loop runner using KQ2-CL's exact trial protocol (no trials run by default)."""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kq2cl_closed as KC
import kq2cl_common as C

OUT = Path(os.environ.get("RES1_RUN_ROOT", C.ROOT / "res1-01" / "production"))
TRIAL_OUT = Path(os.environ.get("RES1_TRIAL_ROOT", OUT))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _checkpoints(*, allow_smoke: bool = False) -> dict:
    spec = {
        "flow_iii": {"query": "iii", "use_delta": False},
        "flow_iii_x": {"query": "iii", "use_delta": True},
        "flow_i": {"query": "i", "use_delta": False},
    }
    selection_meta = C.read_json(OUT / 'selection.json', {})
    if selection_meta.get('smoke', False) and not allow_smoke:
        raise ValueError('smoke-only Dev selection cannot be used for production trials')
    chosen = selection_meta.get('selected', {})
    for name, row in spec.items():
        selected = chosen.get(name)
        if selected is None:
            raise RuntimeError(f"run Dev-v2 selection for {name} first")
        selection = C.read_json(OUT / "train" / selected["run"] / "train.json")
        if selection is None or not selection.get("selected"):
            raise RuntimeError(f"missing selected training checkpoint for {name}")
        step = int(selection["selected"]["checkpoint_step"])
        row["checkpoint"] = str(OUT / "train" / selected["run"] / "checkpoints" / f"{step:06d}")
    return spec


def _read_done(path: Path) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
                out[(int(r["task_id"]), int(r["trial_index"]))] = r
            except (KeyError, json.JSONDecodeError):
                continue
    return out


def run(conditions: list[str], *, suite: str, trials: int = 10, workers: int = 16,
        fraction: float = .40, smoke: bool = False) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    if not C.OUT.joinpath("DONE").exists():
        raise RuntimeError("KQ2-CL DONE is required before RES1 closed loop")
    if suite not in ("libero90", "libero10"):
        raise ValueError(suite)
    cp = _checkpoints(allow_smoke=smoke)
    root = TRIAL_OUT / ("closed-smoke" if smoke else "closed-loop") / suite
    root.mkdir(parents=True, exist_ok=True)
    task_list = KC.tasks(suite)
    for name in conditions:
        if name not in cp:
            raise ValueError(f"unknown condition {name}")
        cfg = cp[name]
        cdir = root / name; cdir.mkdir(parents=True, exist_ok=True)
        trial_file = cdir / "trials.jsonl"
        done = _read_done(trial_file)
        if smoke:
            task_list = KC.tasks(suite)[:1]
            trial_ids = [0]
        else:
            trial_ids = list(range(trials))
        todo = [(t, i) for t in task_list for i in trial_ids if done.get((int(t["task_id"]), i), {}).get("status") != "complete"]
        if not todo:
            continue
        port = free_port()
        env = os.environ.copy()
        env.update({"SKILLNET_SUBTASK_GENERATE": "0", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                    "XLA_PYTHON_CLIENT_MEM_FRACTION": str(fraction), "TORCH_MEM_FRACTION": ".08",
                    "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8"})
        logs = root / "servers"; logs.mkdir(exist_ok=True)
        server_log = (logs / f"{name}-{port}.log").open("a")
        cmd = ["taskset", "-c", "8-15", "nice", "-n", "10", sys.executable, str(HERE / "res1_serve.py"),
               "--query", cfg["query"], "--checkpoint", cfg["checkpoint"], "--condition", name, "--port", str(port)]
        if cfg["use_delta"]:
            cmd.append("--use-delta")
        proc = subprocess.Popen(cmd, env=env, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            sys.path.insert(0, str(HERE.parent))
            from anchor_flow.run_case import S2WebsocketClientPolicy
            deadline = time.monotonic() + 900
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"server exited; inspect {server_log.name}")
                try:
                    cl = S2WebsocketClientPolicy("127.0.0.1", port); cl.close(); break
                except (OSError, ConnectionError, TimeoutError):
                    if time.monotonic() > deadline: raise TimeoutError("RES1 server start timeout")
                    time.sleep(2)
            jobs = [{"host": "127.0.0.1", "port": port, "suite": KC.SUITE_API[suite],
                     "task_id": int(t["task_id"]), "trial_index": int(i), "seed_base": 7,
                     "max_steps": 520, "wait_steps": 10, "replan_steps": 5,
                     "condition": f"RES1_{name}"} for t, i in todo]
            # One server (8 CPU threads) plus at most three env workers (8 each)
            # keeps this run within the shared 32-core CPU budget.
            with ProcessPoolExecutor(max_workers=1 if smoke else min(workers, 3)) as pool:
                futs = {pool.submit(KC.run_trial, j): j for j in jobs}
                with trial_file.open("a") as f:
                    for fut in as_completed(futs):
                        job = futs[fut]
                        try:
                            result = fut.result()
                        except Exception as exc:
                            result = {"task_id": job["task_id"], "trial_index": job["trial_index"],
                                      "condition": name, "status": "error", "error": repr(exc)}
                        f.write(json.dumps(result, ensure_ascii=False, default=float) + "\n"); f.flush()
        finally:
            proc.terminate()
            try: proc.wait(timeout=20)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait()
            server_log.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("run", "summarize"))
    ap.add_argument("--conditions", default="flow_iii,flow_iii_x,flow_i")
    ap.add_argument("--suite", default="libero90", choices=("libero90", "libero10"))
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--fraction", type=float, default=.40)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.action == "run":
        run(a.conditions.split(","), suite=a.suite, trials=a.trials, workers=a.workers,
            fraction=a.fraction, smoke=a.smoke)
    else:
        root = TRIAL_OUT / "closed-loop" / a.suite
        summary = {}
        for p in root.glob("*/trials.jsonl"):
            rows = _read_done(p); valid = [x for x in rows.values() if x.get("status") == "complete"]
            summary[p.parent.name] = {"n": len(valid), "successes": int(sum(bool(x["success"]) for x in valid)),
                                      "rate": float(np.mean([x["success"] for x in valid])) if valid else None}
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
