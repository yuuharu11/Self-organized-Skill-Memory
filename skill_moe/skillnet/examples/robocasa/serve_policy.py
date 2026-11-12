"""Serve the ordinary RoboCasa multitask pi0.5 checkpoint."""
import argparse
import pathlib
import os

from openpi.policies import policy_config
from openpi.policies.robocasa_policy import configure_robocasa_subtask_generation
from openpi.shared import normalize
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training import config


def configure_plain_pi05_inference():
    """Default to the released fork behavior while respecting explicit opt-in."""
    configure_robocasa_subtask_generation()


def main():
    configure_plain_pi05_inference()
    try:
        import torch
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    except ImportError:
        pass
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=pathlib.Path)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--config", default="pi05_robocasa_multitask")
    a = p.parse_args()
    cfg = config.get_config(a.config)
    stats = normalize.load(a.checkpoint / "assets")
    policy = policy_config.create_trained_policy(cfg, a.checkpoint, norm_stats=stats)
    print(f"serving {a.config} checkpoint={a.checkpoint} port={a.port} "
          f"subtask_generation={os.environ.get('SKILLNET_SUBTASK_GENERATE', os.environ.get('OPENPI_SUBTASK_GENERATE', '0'))}",
          flush=True)
    WebsocketPolicyServer(policy, host=a.host, port=a.port,
                          metadata={"config": a.config, "action_horizon": 50,
                                    "cameras": ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]},
                          # The synchronous Pi0.5 inference can take longer than
                          # the websocket default ping timeout during first JIT.
                          # Disable server pings rather than closing live rollouts.
                          ping_interval=None, ping_timeout=None,
                          infer_in_thread=True).serve_forever()


if __name__ == "__main__":
    main()
