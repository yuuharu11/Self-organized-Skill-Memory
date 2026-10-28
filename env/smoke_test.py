"""Check runtime imports, actual GPU computation, and optional LIBERO rendering.

No model weights or training datasets are downloaded by this check.
"""

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true", help="Allow CPU only (does not validate GPU readiness)")
    parser.add_argument("--render", action="store_true", help="Create one LIBERO env and render/reset/step")
    args = parser.parse_args()
    if args.cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"

    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch
    import tree

    versions = {}
    for name in ("jax", "jaxlib", "flax", "orbax-checkpoint", "torch", "lerobot", "transformers", "mujoco", "robosuite"):
        versions[name] = importlib.metadata.version(name)
    print(json.dumps({"python": sys.version, "versions": versions}, indent=2), flush=True)
    devices = jax.devices()
    print("JAX devices:", devices, flush=True)
    if not args.cpu and not any(d.platform == "gpu" for d in devices):
        raise RuntimeError("JAX has no GPU; check NVIDIA Container Toolkit/driver/CUDA wheels")
    x = jnp.ones((32, 32))
    np.testing.assert_allclose(np.asarray((x @ x).block_until_ready()), 32.0)
    if not args.cpu:
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot access CUDA")
        assert torch.ones(8, device="cuda").sum().item() == 8
    assert tree.map_structure(lambda x: x + 1, {"x": 1}) == {"x": 2}

    for name in (
        "skillnet_client.websocket_client_policy",
        "openpi.models.pi0_moe_skill",
        "openpi.policies.policy_config",
        "openpi.training.data_loader_skill",
    ):
        importlib.import_module(name)
        print("IMPORT OK:", name, flush=True)
    from openpi.training import config_moe_skill

    for name in ("pi05_libero_moe_skill_4_40", "pi05_libero_moe_skill_4_90"):
        cfg = config_moe_skill.get_config(name)
        assert cfg.model.action_expert_variant == "gemma_300m_moe_4"
        print("CONFIG OK:", cfg.name, flush=True)

    from libero.libero import benchmark, get_libero_path

    suite = benchmark.get_benchmark_dict()["libero_skill_obj"]()
    repo = Path(os.environ.get("SKILLNET_REPO_ROOT", str(Path(__file__).resolve().parents[1])))
    manifest = repo / "skill_moe/skillnet/third_party/libero/libero/libero/bddl_files/libero_skill_obj/public_task_manifest.json"
    expected = json.loads(manifest.read_text())["tasks"]
    observed = [Path(suite.get_task(i).bddl_file).stem for i in range(suite.n_tasks)]
    assert observed == expected, (observed, expected)
    for i in range(suite.n_tasks):
        task = suite.get_task(i)
        assert Path(suite.get_task_bddl_file_path(i)).is_file()
        assert (Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file).is_file()
    init_states = suite.get_task_init_states(0)
    assert len(init_states) > 0
    print("LIBERO-Skill manifest OK:", len(observed), "tasks", flush=True)

    if args.render:
        from libero.libero.envs import OffScreenRenderEnv

        env = OffScreenRenderEnv(
            bddl_file_name=suite.get_task_bddl_file_path(0),
            camera_heights=128,
            camera_widths=128,
        )
        try:
            env.seed(7)
            obs = env.reset()
            obs = env.set_init_state(init_states[0])
            for _ in range(2):
                obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
            assert obs["agentview_image"].shape == (128, 128, 3)
            print("LIBERO render/reset/step OK:", obs["agentview_image"].shape, flush=True)
        finally:
            env.close()
    print("SMOKE PASSED", flush=True)


if __name__ == "__main__":
    main()
