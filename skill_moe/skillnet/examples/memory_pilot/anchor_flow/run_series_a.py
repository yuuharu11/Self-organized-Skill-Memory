#!/usr/bin/env python3
"""Start one S2 websocket server and run resumable paired LIBERO cases."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from memory import load_memory
from smoke import DEFAULT_CHECKPOINT

HERE = Path(__file__).resolve().parent


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_port(port: int, process: subprocess.Popen, log_path: Path, timeout: int = 180):
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        if process.poll() is not None:
            raise RuntimeError(f"S2 websocket server exited ({process.returncode}); see {log_path}")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(1)
    raise TimeoutError(f"S2 websocket server not ready on {port}; see {log_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--memory-dir", required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--generator-checkpoint", default="/work/.skillnet-state/checkpoints/openpi-assets/checkpoints/pi05_base/params")
    parser.add_argument("--query-text-source", choices=("instruction_only", "pi05_generated_subtask"),
                        default="instruction_only")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--workers", type=int, default=2,
                        help="parallel LIBERO environments sharing one serialized GPU policy server")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--minimal", action="store_true", help="92-case S2 gate matrix")
    parser.add_argument("--full-matrix", action="store_true", help="182-case extended comparison matrix")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--server-gpu", default=None, help="optional CUDA_VISIBLE_DEVICES for the single server")
    parser.add_argument("--absolute-similarity-low", type=float, default=0.2)
    parser.add_argument("--absolute-similarity-high", type=float, default=0.8)
    parser.add_argument("--fixed-t0", type=float, default=0.5)
    args = parser.parse_args()
    if args.minimal and args.full_matrix:
        raise ValueError("choose either --minimal or --full-matrix")
    if args.trials < 20:
        raise ValueError("series A requires at least 20 trials per task/condition")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    output, memory_dir = Path(args.output), Path(args.memory_dir)
    if output.exists() and not args.resume:
        raise FileExistsError(f"{output} exists; pass --resume to continue")
    output.mkdir(parents=True, exist_ok=True)
    initial_memory_path = memory_dir / "initial_libero90.pkl"
    initial_memory = load_memory(initial_memory_path)
    key_text_source = initial_memory.key_text_source
    if key_text_source != args.query_text_source:
        raise ValueError(f"memory keys use {key_text_source}, requested {args.query_text_source}")
    port = free_port()
    server_log = output / "policy_server.log"
    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if args.server_gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.server_gpu
    if str(HERE.parents[2] / "third_party" / "libero") not in env.get("PYTHONPATH", ""):
        env["PYTHONPATH"] = str(HERE.parents[2] / "third_party" / "libero") + os.pathsep + env.get("PYTHONPATH", "")
    env["S2_MEMORY_DIR"] = str(memory_dir.resolve())
    metadata = {"checkpoint": str(Path(args.checkpoint).resolve()), "memory_dir": str(memory_dir.resolve()),
                "normalization_sha256": initial_memory.normalization_sha256,
                "key_text_source": key_text_source, "query_text_source": args.query_text_source,
                "port": port, "workers": args.workers, "trials_per_case": args.trials,
                "paired_seed": args.seed, "minimal_gate_matrix": args.minimal,
                "full_matrix": args.full_matrix, "absolute_similarity_low": args.absolute_similarity_low,
                "absolute_similarity_high": args.absolute_similarity_high, "fixed_t0": args.fixed_t0}
    metadata_path = output / "runner.json"
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text())
        # A resumed run can choose a new free port, but its experimental inputs must match.
        for key in metadata:
            if key != "port" and old.get(key) != metadata[key]:
                raise ValueError(f"resume metadata mismatch at {key}: {old.get(key)!r} != {metadata[key]!r}")
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    server_command = [sys.executable, str(HERE / "serve_s2_policy.py"),
        "--checkpoint", args.checkpoint, "--memory-dir", str(memory_dir.resolve()),
        "--normalization-sha256", initial_memory.normalization_sha256,
        "--port", str(port), "--seed", str(args.seed),
        "--query-text-source", args.query_text_source,
        "--generator-checkpoint", args.generator_checkpoint,
        "--absolute-similarity-low", str(args.absolute_similarity_low),
        "--absolute-similarity-high", str(args.absolute_similarity_high), "--fixed-t0", str(args.fixed_t0)]
    server = None
    try:
        with server_log.open("a", encoding="utf-8") as log:
            server = subprocess.Popen(server_command, cwd=HERE.parents[2], env=env,
                                      stdout=log, stderr=subprocess.STDOUT)
        wait_port(port, server, server_log)
        runner = [sys.executable, str(HERE / "run_case.py"), "--port", str(port), "--workers", str(args.workers)]
        eval_command = [sys.executable, str(HERE / "eval_series_a.py"),
            "--output", str(output / "cases"), "--trials", str(args.trials),
            "--runner", " ".join(runner), "--checkpoint", args.checkpoint,
            "--seed", str(args.seed), "--query-text-source", args.query_text_source, "--execute"]
        if args.minimal or not args.full_matrix:
            eval_command.append("--minimal")
        if args.resume:
            eval_command.append("--resume")
        completed = subprocess.run(eval_command, cwd=HERE.parents[2], env=env, check=False)
        if completed.returncode:
            raise SystemExit(completed.returncode)
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    print(json.dumps({"output": str(output.resolve()), "server_log": str(server_log),
                      "port": port, "workers": args.workers,
                      "case_matrix": "full" if args.full_matrix else "minimal_92",
                      "query_text_source": args.query_text_source}, indent=2))


if __name__ == "__main__":
    main()
