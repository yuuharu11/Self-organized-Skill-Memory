"""Bounded three-server research run, with explicit phase and final deadlines."""
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from effect_worker import write


def epoch(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def execute_group(root, port, jobs, deadline):
    results = []
    for job in jobs:
        if time.time() >= deadline or (root / "STOP_REQUESTED").exists():
            break
        directory = root / job["directory"]
        if (directory / "result.json").exists():
            results.append(dict(directory=job["directory"], status="previously_complete"))
            continue
        # No automatic retry of interrupted or failed attempts.
        if (directory / "job.json").exists():
            results.append(dict(directory=job["directory"], status="previous_attempt_incomplete"))
            continue
        directory.mkdir(parents=True, exist_ok=True)
        job = dict(job, port=port, hard_deadline=deadline)
        jobfile = root / "jobs" / (directory.name + ".json")
        write(jobfile, job)
        log = (directory / "worker.log").open("a")
        cmd = ["bash", str(root / "run_cpu.sh"), str(root / "frozen/effect_worker.py"),
               "--root", str(root), "--job", str(jobfile)]
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while process.poll() is None:
            if time.time() > deadline + 15 or (root / "STOP_REQUESTED").exists():
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(2)
        code = process.wait()
        log.close()
        row = dict(directory=job["directory"], returncode=code,
                   status="complete" if (directory / "result.json").exists() else "incomplete")
        results.append(row)
        print(json.dumps(row), flush=True)
    return results


def phase(root, name, groups, ports, deadline):
    write(root / "status.json", dict(phase=name, started_utc=datetime.now(timezone.utc).isoformat(),
                                     planned_groups=len(groups), deadline=deadline))
    outcomes = []
    iterator = iter(groups)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(ports)) as pool:
        active = {}
        for port in ports:
            group = next(iterator, None)
            if group is not None:
                active[pool.submit(execute_group, root, port, group, deadline)] = port
        while active:
            done, _ = concurrent.futures.wait(active, timeout=20, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                port = active.pop(future)
                outcomes.extend(future.result())
                write(root / (name + "_outcomes.json"), outcomes)
                if time.time() < deadline and not (root / "STOP_REQUESTED").exists():
                    group = next(iterator, None)
                    if group is not None:
                        active[pool.submit(execute_group, root, port, group, deadline)] = port
    return outcomes


def fit(root, which):
    log = (root / ("fit-" + which + ".log")).open("a")
    code = subprocess.call(["bash", str(root / "run_cpu.sh"), str(root / "frozen/fit_effect_controller.py"),
                            which, "--root", str(root)], stdout=log, stderr=subprocess.STDOUT)
    log.close()
    if code:
        raise RuntimeError("model fitting failed: " + which)


def main(root):
    manifest = json.loads((root / "manifest.json").read_text())
    ports = manifest["ports"]
    (root / "jobs").mkdir(exist_ok=True)
    while not (root / "PARITY_PASS").exists():
        if time.time() > epoch("2026-11-02T20:00:00Z"):
            raise TimeoutError("public parity gate not passed")
        time.sleep(10)
    parent_groups = []
    # First one train initial state/task, then validation, then second train init.
    for init in (10, 12, 11):
        for task in manifest["train_tasks"]:
            parent_groups.append([dict(kind="parent", task=task, init=init, seed=7,
                split="validation" if init == 12 else "train",
                directory=f"parents/task{task+1:02d}-init{init:02d}")])
    phase(root, "parents", parent_groups, ports, epoch("2026-11-02T20:50:00Z"))
    fit(root, "monitor")
    pair_groups = []
    for parent_group in parent_groups:
        parent = parent_group[0]
        parent_dir = root / parent["directory"]
        if not (parent_dir / "result.json").exists():
            continue
        for snapshot in sorted(parent_dir.glob("snapshot-*.npz")):
            # A scheduling unit keeps both seeds and both arms together.
            group = []
            for seed in (17, 29):
                for arm in ("keep", "update"):
                    group.append(dict(kind="pair", task=parent["task"], init=parent["init"], seed=seed,
                        split=parent["split"], parent=parent["directory"], snapshot=str(snapshot.relative_to(root)),
                        arm=arm, directory=f"pairs/{parent_dir.name}-{snapshot.stem}-seed{seed}-{arm}"))
            pair_groups.append(group)
    phase(root, "pairs", pair_groups, ports, epoch("2026-11-02T23:15:00Z"))
    fit(root, "effect")
    evaluation = []
    for init in (20, 21):
        for task in manifest["test_tasks"]:
            # Each group is a full matched five-condition initial state.
            evaluation.append([dict(kind="evaluation", task=task, init=init, seed=43, condition=condition,
                split="test", directory=f"evaluation/task{task+1:02d}-init{init:02d}-{condition}")
                for condition in manifest["conditions"]])
    phase(root, "evaluation", evaluation, ports, epoch(manifest["stop_launch_utc"]))
    write(root / "COLLECTION_FINISHED.json", dict(time=datetime.now(timezone.utc).isoformat(),
        evaluation_complete=len(list((root/"evaluation").glob("*/result.json"))) == 90))


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    try:
        main(root)
    except Exception as exc:
        write(root / "PIPELINE_ERROR.json", dict(error=repr(exc), time=datetime.now(timezone.utc).isoformat()))
        raise
