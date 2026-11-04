"""Integration checks for the explicit S1-v3c command-space bridge."""
from __future__ import annotations

import csv
import importlib.util
import inspect
from pathlib import Path
import sys

import numpy as np

ROOT = Path("/work/.skillnet-state/outputs/memory-pilot/dmp-route-20261102")
S1 = ROOT / "s1-v3c"
FRAMES = ROOT / "s1-v3" / "frames.npz"
PRIV = Path("/work/.skillnet-state/outputs/memory-pilot/privstate-20261101T003957Z")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from command_dmp import command_draft
from s3_policy import DmpRoutePolicy
from train_utils import load_s1_library, low_memory_adamw


def _load_dmp():
    spec = importlib.util.spec_from_file_location("s1_v3c_s3_test", S1 / "dmp.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_real_v3c_dictionary_draft_matches_raw_s1_rollout_and_uses_only_quantile_norm():
    library = load_s1_library(S1 / "library.npz")
    with np.load(FRAMES, allow_pickle=False) as z:
        frames = {k: np.asarray(z[k]) for k in z.files}
    # Pick a Train segment start, where S1 starts the DMP at the actual current
    # command and phase zero. The shared exact action mapping supplies raw y0.
    idx = next(i for i in range(len(frames["t"]))
               if frames["split"][i] == "train" and frames["t"][i] == frames["seg_start"][i]
               and frames["seg_end"][i] - frames["seg_start"][i] >= 10)
    ep = str(frames["episode_id"][idx])
    mapping = next(r for r in csv.DictReader(open(PRIV / "memory_episode_map.csv")) if r["episode_id"] == ep)
    action_path = PRIV / "episodes" / (Path(mapping["source_file"]).stem + "__" + mapping["demo_key"] + ".npz")
    with np.load(action_path, allow_pickle=False) as z:
        actions = np.asarray(z["action"], np.float32)
    st, en = int(frames["seg_start"][idx]), int(frames["seg_end"][idx])
    pid = int(frames["prim_id"][idx])
    y0 = actions[st, :6]
    g = frames["g"][idx]
    tau = float(frames["tau"][idx])
    phase = float(frames["phase_s"][idx])
    dmp = _load_dmp()
    raw_expected = dmp.rollout(pid, y0, g, tau, en - st, library=library,
                               phase_s=phase, dt=float(library["dt"]))
    # A synthetic but valid released normalization bundle. No pose scale or
    # finite-difference conversion is applied anywhere in this adapter.
    q01 = np.asarray([-1.2, -1.1, -1.3, -1.4, -1.2, -1.1, -1.0], np.float32)
    q99 = -q01
    route = type("Route", (), {"prim_id": np.asarray([pid]), "goal": g[None],
                               "tau": np.asarray([tau]), "phase_s": np.asarray([phase])})()
    normed = command_draft(dmp, library, route, y0[None], q01=q01, q99=q99,
                           horizon=en-st, dt=float(library["dt"]))
    raw_restored = (normed[..., :6] + 1.) * .5 * (q99[:6] - q01[:6] + 1e-6) + q01[:6]
    np.testing.assert_allclose(raw_restored[0], raw_expected, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(normed[0, :, 6],
                               ((float(library["grip"][pid]) - q01[6]) /
                                (q99[6] - q01[6] + 1e-6) * 2. - 1.), atol=1e-6)


def test_real_primitive_swap_changes_dmp_draft_and_inference_has_no_teacher_args():
    library = load_s1_library(S1 / "library.npz")
    dmp = _load_dmp()
    q01 = np.full(7, -1.0, np.float32)
    q99 = np.full(7, 1.0, np.float32)
    goal = np.asarray([[.2, -.1, .03, .1, .04, -.02]], np.float32)
    y0 = np.zeros((1, 6), np.float32)
    def make(pid):
        route = type("Route", (), {"prim_id": np.asarray([pid]), "goal": goal,
                                   "tau": np.asarray([.8]), "phase_s": np.asarray([.2])})()
        return command_draft(dmp, library, route, y0, q01=q01, q99=q99,
                             horizon=10, dt=float(library["dt"]))
    pair = next((a, b) for a in range(len(library["grip"]))
                for b in range(a+1, len(library["grip"]))
                if np.max(np.abs(library["coef"][a] - library["coef"][b])) > 1e-3)
    assert not np.allclose(make(pair[0]), make(pair[1]))
    assert set(inspect.signature(DmpRoutePolicy.infer).parameters) == {"self", "obs", "noise"}


def test_low_memory_adamw_keeps_updates_and_compacts_both_moments():
    import jax.numpy as jnp

    tx = low_memory_adamw(lambda _: jnp.asarray(1e-3), moment_dtype=jnp.bfloat16)
    params = {"w": jnp.asarray([1., -1.], jnp.float32)}
    grads = {"w": jnp.asarray([.25, -.5], jnp.float32)}
    state = tx.init(params)
    updates, next_state = tx.update(grads, state, params)
    assert next_state.mu["w"].dtype == jnp.bfloat16
    assert next_state.nu["w"].dtype == jnp.bfloat16
    assert updates["w"].dtype == jnp.float32
    assert float(updates["w"][0]) < 0 and float(updates["w"][1]) > 0
