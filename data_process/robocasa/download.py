#!/usr/bin/env python3
"""Thin, non-destructive wrapper around RoboCasa's official downloader."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
TASK_SETS_PATH = ROOT / "data_process/robocasa/task_sets.json"
ROBOCASA_COMMIT = "3443cdf3dd20f9ae8ed41adba442059b506891d0"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-set", choices=["atomic_seen", "composite_seen", "composite_unseen", "target50"])
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--split", choices=["pretrain", "target"], default="target")
    parser.add_argument("--source", choices=["human", "mimicgen"], default="human")
    parser.add_argument("--preset", choices=["smoke"])
    parser.add_argument("--dryrun", action="store_true")
    return parser


def _dataset_inventory(state: pathlib.Path, tasks: list[str]) -> list[dict]:
    registry = json.loads(TASK_SETS_PATH.read_text())
    inventory = []
    for task in tasks:
        rel = registry["tasks"][task]["dataset_path"]
        root = state / "data/robocasa" / rel
        info_path = root / "meta/info.json"
        info = json.loads(info_path.read_text()) if info_path.is_file() else {}
        inventory.append(
            {
                "task": task,
                "path": str(root),
                "present": info_path.is_file(),
                "episodes": info.get("total_episodes"),
                "frames": info.get("total_frames"),
            }
        )
    return inventory


def _append_manifest(path: pathlib.Path, run: dict) -> None:
    if path.exists():
        existing = json.loads(path.read_text())
        runs = existing.get("runs", [existing])
    else:
        runs = []
    runs.append(run)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"runs": runs}, indent=2) + "\n")
    tmp.replace(path)


def main() -> None:
    args = _parser().parse_args()
    if args.tasks and args.task_set:
        _parser().error("choose either --tasks or --task-set")
    registry = json.loads(TASK_SETS_PATH.read_text())
    tasks = args.tasks or (registry["task_sets"][args.task_set] if args.task_set else None)
    if args.preset == "smoke":
        tasks = ["OpenDrawer"]
    if not tasks:
        _parser().error("provide --tasks, --task-set, or --preset smoke")
    if args.preset == "smoke" and len(tasks) > 2:
        _parser().error("smoke is limited to two tasks")

    env = os.environ.copy()
    state = pathlib.Path(env.get("SKILLNET_STATE_DIR", ROOT / ".skillnet-state"))
    env["SKILLNET_STATE_DIR"] = str(state)
    command = [
        sys.executable,
        "-m",
        "robocasa.scripts.download_datasets",
        "--tasks",
        *tasks,
        "--split",
        args.split,
        "--source",
        args.source,
    ]
    if args.dryrun:
        command.append("--dryrun")
    # Do not pass --overwrite: the official downloader skips existing datasets.
    result = subprocess.run(command, input="y\n", text=True, env=env, check=True)
    run = {
        "tasks": tasks,
        "split": args.split,
        "source": args.source,
        "robocasa_commit": ROBOCASA_COMMIT,
        "lerobot_version": "0.3.3",
        "command": command,
        "dryrun": args.dryrun,
        "completed_at_unix": time.time(),
        "dataset_base": str(state / "data/robocasa"),
        "datasets": _dataset_inventory(state, tasks),
        "official_downloader_exit": result.returncode,
    }
    manifest = state / "data/robocasa/download_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    _append_manifest(manifest, run)
    print("manifest:", manifest)


if __name__ == "__main__":
    main()
