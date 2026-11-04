#!/usr/bin/env python3
"""Launch each trained S3 websocket server and issue exactly one query."""
from __future__ import annotations

import json
import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np

ROOT = Path("/work/.skillnet-state/outputs/memory-pilot/dmp-route-20261102/s3-train-v3c")
SERVE = Path("/work/skill_moe/skillnet/examples/memory_pilot/dmp_route/serve_s3.py")


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def query_once(condition: str, checkpoint: Path, run_id: str):
    port = free_port()
    log = ROOT / f"server_smoke_{condition}_{run_id}.log"
    if log.exists():
        raise FileExistsError(log)
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.30",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
    })
    cmd = [sys.executable, str(SERVE), "--checkpoint", str(checkpoint),
           "--condition", condition, "--host", "127.0.0.1", "--port", str(port),
           "--prompt-level", "L0", "--bridge-t0", "0.4", "--denoise-steps", "2"]
    with log.open("w") as f:
        proc = subprocess.Popen(cmd, cwd="/work/skill_moe/skillnet", env=env,
                                stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f"{condition} server exited early with {proc.returncode}; see {log}")
                if "serving S3 on" in log.read_text(errors="replace"):
                    break
                time.sleep(2)
            else:
                raise TimeoutError(f"{condition} server did not listen within 10 minutes")

            from skillnet_client.websocket_client_policy import WebsocketClientPolicy
            client = WebsocketClientPolicy("127.0.0.1", port)
            rng = np.random.default_rng(43)
            image = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
            query = {
                "observation/image": image,
                "observation/wrist_image": image.copy(),
                "observation/state": np.zeros(8, np.float32),
                "prompt": "pick up the object",
                "episode_id": f"s3-smoke-{condition}",
                "dmp_history_id": f"s3-smoke-{condition}",
                "dmp_reset_history": True,
            }
            result = client.infer(query)
            actions = np.asarray(result["actions"])
            route = result.get("dmp_route", {})
            # The LIBERO policy transform's released convention is action-first
            # and may retain the model's padded width; require a 7D LIBERO axis.
            if actions.ndim != 2 or 7 not in actions.shape or not np.isfinite(actions).all():
                raise AssertionError(f"invalid output action array {actions.shape}")
            if condition == "G1" and "prim_id" not in route:
                raise AssertionError("G1 response lacks predicted primitive route")
            return {"condition": condition, "status": "complete", "port": port,
                    "action_shape": list(actions.shape), "finite": True,
                    "prim_id": route.get("prim_id"), "t0": 0.4,
                    "max_abs_action": float(np.max(np.abs(actions)))}
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


def latest_checkpoint(condition: str) -> Path:
    attempts = ROOT / "attempts"
    candidates = sorted(attempts.glob(f"{condition}-attempt-*/step_1500/params"))
    for checkpoint in reversed(candidates):
        metrics = checkpoint.parents[1] / "metrics.jsonl"
        if metrics.exists() and any('"event": "complete"' in line for line in metrics.read_text().splitlines()):
            return checkpoint
    raise FileNotFoundError(f"no completed checkpoint for {condition} under {attempts}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    outputs = []
    for condition in ("G1-dmp", "G0"):
        checkpoint = latest_checkpoint(condition)
        outputs.append(query_once("G1" if condition == "G1-dmp" else "G0", checkpoint, args.run_id))
    (ROOT / f"server_smoke_{args.run_id}.json").write_text(json.dumps(outputs, indent=2) + "\n")
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
