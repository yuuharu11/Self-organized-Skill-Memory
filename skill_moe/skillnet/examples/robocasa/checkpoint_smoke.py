"""Load the released RoboCasa checkpoint and run policy + MemoryPi05 bridge."""
import argparse
import pathlib
import json
import imageio.v2 as imageio
import pyarrow.parquet as pq

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from openpi.models import model as model_api
from openpi.policies import policy_config
from openpi.policies.robocasa_policy import (
    configure_robocasa_subtask_generation,
    dataset_action_to_policy,
)
from openpi.shared import normalize, nnx_utils
from openpi.training import config


def main():
    # RoboCasa defaults to the official task-only prompt; explicit opt-in remains possible.
    configure_robocasa_subtask_generation()
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=pathlib.Path, required=True)
    a = p.parse_args()
    cfg = config.get_config("pi05_robocasa_multitask")
    assert cfg.model.action_horizon == 50 and cfg.model.action_dim == 32
    assert type(cfg.model).__module__.endswith("pi0_config")
    stats = normalize.load(a.checkpoint / "assets")
    assert all(stat.q01 is None and stat.q99 is None for stat in stats.values())
    policy = policy_config.create_trained_policy(cfg, a.checkpoint, norm_stats=stats)
    rng = np.random.default_rng(7)
    raw = {
        "observation/image": rng.integers(0,256,(224,224,3),dtype=np.uint8),
        "observation/right_image": rng.integers(0,256,(224,224,3),dtype=np.uint8),
        "observation/wrist_image": rng.integers(0,256,(224,224,3),dtype=np.uint8),
        "observation/state": np.zeros(16,np.float32),
        "prompt": "Open the drawer",
    }
    pred = policy.infer(raw)["actions"]
    assert pred.shape == (50,12) and np.isfinite(pred).all(), (pred.shape, np.isfinite(pred).all())
    print("policy action", pred.shape, "finite", True, flush=True)

    # Also exercise one real pretrain demonstration observation and record the
    # one-step action error against the matching saved action (not a success claim).
    data_root = pathlib.Path("/work/.skillnet-state/data/robocasa/v1.0/pretrain/atomic/OpenDrawer/20250819/lerobot")
    parquet = sorted((data_root / "data").glob("**/*.parquet"))[0]
    row = pq.read_table(parquet).slice(0, 1).to_pylist()[0]
    row_idx = int(row["frame_index"])
    video_keys = {
        "observation/image": "robot0_agentview_left",
        "observation/right_image": "robot0_agentview_right",
        "observation/wrist_image": "robot0_eye_in_hand",
    }
    demo = {}
    for output_key, camera in video_keys.items():
        video = data_root / "videos/chunk-000" / f"observation.images.{camera}" / "episode_000000.mp4"
        reader = imageio.get_reader(video)
        try:
            demo[output_key] = reader.get_data(row_idx)
        finally:
            reader.close()
    dataset_state = np.asarray(row["observation.state"], np.float32)
    # Convert the saved modality order to the official online state order.
    demo_state = np.concatenate([dataset_state[7:10], dataset_state[10:14],
                                 dataset_state[0:3], dataset_state[3:7], dataset_state[14:16]])
    task_rows = [json.loads(line) for line in (data_root / "meta/tasks.jsonl").read_text().splitlines() if line]
    task_index = int(row["annotation.human.task_description"])
    demo.update({"observation/state": demo_state, "prompt": task_rows[task_index]["task"]})
    demo_pred = policy.infer(demo)["actions"]
    demo_action = dataset_action_to_policy(np.asarray(row["action"], np.float32))
    print("demo one-step policy-order action MSE",
          float(np.mean((demo_pred[0] - demo_action) ** 2)),
          "prompt", demo["prompt"], flush=True)

    # Build the bridge-capable subclass from the same plain checkpoint params.
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "memory_pilot" / "pi05_memory"))
    import core
    bridge_model, bridge_cfg = core.make_model(
        action_horizon=50, discrete_state_input=True, max_token_len=200, robocasa=True
    )
    from openpi.models import model as _model
    loaded = cfg.model.load(_model.restore_params(a.checkpoint / "params", dtype=jnp.bfloat16))
    nnx.update(bridge_model, nnx.state(loaded))

    transformed = policy._input_transform(raw)
    observation = model_api.Observation(
        images={k:jnp.asarray(v,dtype=jnp.float32)[None] for k,v in transformed["image"].items()},
        image_masks={k:jnp.asarray([v]) for k,v in transformed["image_mask"].items()},
        state=jnp.asarray(transformed["state"])[None],
        tokenized_prompt=jnp.asarray(transformed["tokenized_prompt"])[None],
        tokenized_prompt_mask=jnp.asarray(transformed["tokenized_prompt_mask"])[None],
    )
    assert bridge_cfg.action_horizon==50 and bridge_cfg.discrete_state_input is True
    draft=jnp.zeros((1,50,32),jnp.float32)
    bridged=bridge_model.sample_bridge(jax.random.key(91),observation,draft,jnp.asarray([.4]),num_steps=2)
    bridged=np.asarray(bridged)
    assert bridged.shape == (1,50,32) and np.isfinite(bridged).all(), bridged.shape
    print("sample_bridge", bridged.shape, "finite", True, "t0=0.4 n=2", flush=True)


if __name__ == "__main__":
    main()
