"""Compare websocket inference against the newest pre-injection source snapshot."""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import pathlib
import socket
import subprocess
import sys
import time
import urllib.request
import uuid

import numpy as np

ROOT = pathlib.Path("/work")
STATE = ROOT / ".skillnet-state"
CHECKPOINT = STATE / "checkpoints/pi05_libero_moe_skill_4_40/moe_add_balance_4_128_30000/29999"
DEFAULT_SNAPSHOT = STATE / "snapshots/20261025T140928Z/skillnet"
DEFAULT_ROLLOUT = STATE / "outputs/memory-pilot/record-20261025T130411-ff650e"
SCRIPT = pathlib.Path(__file__).resolve()


def _new_run_dir():
    parent = STATE / "outputs" / "memory-pilot"
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / f"review-websocket-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    path.mkdir(exist_ok=False)
    return path


def _available_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _server_environment(snapshot: pathlib.Path | None):
    env = os.environ.copy()
    if snapshot is not None:
        paths = [snapshot / "src", snapshot / "packages" / "skillnet-client" / "src"]
        env["PYTHONPATH"] = os.pathsep.join([*(str(path) for path in paths), env.get("PYTHONPATH", "")])
    return env


def _wait_ready(proc, port, logfile, timeout=180):
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/healthz"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"policy server exited ({proc.returncode}); see {logfile}")
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError(f"policy server did not become ready; see {logfile}")


def _infer_server(snapshot, raw, port, run_dir, tag):
    from skillnet_client.websocket_client_policy import WebsocketClientPolicy

    logfile = run_dir / f"{tag}-server.log"
    with logfile.open("x", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [sys.executable, str(SCRIPT), "--server", "--port", str(port), "--checkpoint", str(CHECKPOINT)],
            cwd=ROOT / "skill_moe" / "skillnet",
            env=_server_environment(snapshot),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_ready(proc, port, logfile)
            client = WebsocketClientPolicy("127.0.0.1", port)
            try:
                result = client.infer(raw)
            finally:
                client._ws.close()
            return np.asarray(result["actions"]), proc
        finally:
            if proc.poll() is None:
                proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


def _serve(port, checkpoint):
    # This branch is imported under either the current source tree or the
    # snapshot's PYTHONPATH, so it exercises each tree in a fresh process.
    import jax.numpy as jnp
    from openpi import transforms
    from openpi.models import model as model_lib
    from openpi.policies.policy import Policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.training import checkpoints, config_moe_skill

    config = dataclasses.replace(
        config_moe_skill.get_config("pi05_libero_moe_skill_4_40"), exp_name=checkpoint.parent.name
    )
    model = config.model.load(model_lib.restore_params(checkpoint / "params", dtype=jnp.bfloat16))
    data_config = config.data.create(config.assets_dirs, config.model)
    norm_stats = checkpoints.load_norm_stats(checkpoint / "assets", data_config.asset_id)
    policy = Policy(
        model,
        transforms=[
            transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        # Leave sample_kwargs unset in both processes to use num_steps=10.
        metadata=config.policy_metadata,
    )
    WebsocketPolicyServer(policy, host="127.0.0.1", port=port, metadata=config.policy_metadata).serve_forever()


def main(snapshot=DEFAULT_SNAPSHOT, rollout_dir=DEFAULT_ROLLOUT):
    snapshot = snapshot.resolve()
    if not (snapshot / "src" / "openpi").exists():
        raise FileNotFoundError(f"snapshot source tree not found: {snapshot}")
    run_dir = _new_run_dir()
    manifest = json.loads((rollout_dir / "manifest.json").read_text())
    episode = manifest["episodes"][0]
    archive = np.load(rollout_dir / "episodes" / f"{episode['episode_id']}.npz")
    ids = (archive["skills"][0][archive["skill_mask"][0]] - 1).astype(np.int32).tolist()
    raw = {
        "observation/image": archive["img_base"][0],
        "observation/wrist_image": archive["img_wrist"][0],
        "observation/state": archive["proprio"][0],
        "prompt": str(archive["prompt"][0]),
        "skills": str(ids),
    }
    port = _available_port()
    old_actions, _ = _infer_server(snapshot, raw, port, run_dir, "snapshot")
    current_actions, _ = _infer_server(None, raw, port, run_dir, "current")
    delta = current_actions.astype(np.float64) - old_actions.astype(np.float64)
    exact = bool(np.array_equal(old_actions.view(np.uint8), current_actions.view(np.uint8)))
    result = {
        "snapshot": str(snapshot),
        "snapshot_git_sha": json.loads((snapshot.parent / "snapshot.json").read_text())["git_sha"],
        "checkpoint": str(CHECKPOINT),
        "episode_id": episode["episode_id"],
        "websocket_port": port,
        "public_default_num_steps": 10,
        "policy_rng_seed": "default JAX key 0 in both fresh server processes",
        "snapshot_output_shape": list(old_actions.shape),
        "current_output_shape": list(current_actions.shape),
        "snapshot_current_bitwise_equal": exact,
        "max_abs_difference": float(np.max(np.abs(delta))),
        "mean_abs_difference": float(np.mean(np.abs(delta))),
    }
    np.savez_compressed(run_dir / "actions.npz", snapshot=old_actions, current=current_actions)
    (run_dir / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(run_dir)
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--port", type=int)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=CHECKPOINT)
    parser.add_argument("--snapshot", type=pathlib.Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--rollout-dir", type=pathlib.Path, default=DEFAULT_ROLLOUT)
    args = parser.parse_args()
    if args.server:
        _serve(args.port, args.checkpoint)
    else:
        main(args.snapshot, args.rollout_dir)
