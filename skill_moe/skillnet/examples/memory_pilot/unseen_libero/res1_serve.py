"""RES1 live policy: KQ2 retrieval + KQ2 gripper/fallback + residual Expert."""
from __future__ import annotations

import argparse
import json
import os
import sys
import types
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "pi05_memory")]
import kq2cl_common as C
import kq2cl_serve as KQSERVER
import res1_experiment as RE

CTX_DIR = C.ROOT / "res1-01" / "context-exact-01"


class RES1Policy(KQSERVER.KQ2CLPolicy):
    """KQ2-CL policy with `draft + residual` in place of KQ2's bridge output."""
    def __init__(self, *, query: str, checkpoint: Path, use_delta: bool, bundle: Path | None = None,
                 fallback_b: float | None = None):
        pol = C.read_json(C.OUT / "policy.json")
        if not pol:
            raise FileNotFoundError("KQ2-CL policy selection is missing")
        q = "iii" if query == "iii" else "i"
        selected = pol["queries"][q]["selected"]
        bundle = bundle or (C.OUT / "heads" / f"bundle_{q}.pt")
        super().__init__(mode="memory", query=q,
                         expert_checkpoint=Path(C.OUT / "expert" / "train-K1" / "checkpoints" / "000500"),
                         bundle=bundle, tpol={"const": .15, "b": selected["b"] if fallback_b is None else fallback_b})
        import flax.nnx as nnx
        import jax
        import jax.numpy as jnp
        import orbax.checkpoint as ocp
        import kq2cl_closed as KC
        if use_delta:
            RE.attach_delta_attention(self.model, rng_seed=4101)
        filt = RE.res1_trainable_filter()
        # A smoke may point at the KQ2 K1 initialization itself; the base server
        # already restored those weights. Production RES1 checkpoints include
        # the optional delta-attention leaves and are loaded below.
        if Path(checkpoint).resolve() != Path(C.OUT / "expert" / "train-K1" / "checkpoints" / "000500").resolve():
            with ocp.PyTreeCheckpointer() as cp:
                restored = cp.restore(Path(checkpoint))
            state = nnx.state(self.model, filt)
            state.replace_by_pure_dict(jax.tree.map(jnp.asarray, restored)); nnx.update(self.model, state)
        self.res1_filter = filt
        self.use_delta = bool(use_delta)
        self.saved_context = {}
        context_meta = C.read_json(CTX_DIR / f'bank_{q}.json', {})
        if not context_meta.get('unit_start_exact', False):
            raise ValueError('saved context is not certified exact-start')
        with np.load(CTX_DIR / f"bank_{q}.npz", allow_pickle=False) as z:
            self.saved_context = {k: z[k] for k in z.files}
        source = np.load(C.ROOT / 'key-motion-01/final/data/bank__source.npy', allow_pickle=False)
        if not np.array_equal(self.saved_context.get('bank_source'), source):
            raise ValueError('saved context bank source/order mismatch')
        if len(self.saved_context.get("bank_source", ())) != 4516:
            raise AssertionError("saved context is not aligned to the 4,516-unit bank")
        if q == "iii" and self.saved_context["vlm"].shape != (4516, 34, 256):
            raise AssertionError("exact-start iii VLM context shape mismatch")
        self.units = np.load(C.ROOT / "key-motion-01/final/data/bank_actions.npy").astype(np.float32)
        self.unit_task = np.load(C.ROOT / "key-motion-01/final/data/bank__task.npy").astype(np.int64)
        if self.use_delta:
            self._residual = self._build_residual_fn(jax, jnp, nnx, use_delta=True)
        else:
            self._residual = self._build_residual_fn(jax, jnp, nnx, use_delta=False)

    def _build_residual_fn(self, jax, jnp, nnx, *, use_delta: bool):
        model = self.model
        if use_delta:
            @nnx.jit
            def run(current, ob, dr, eps, saved, now, valid):
                def vf(o, draft, residual, u, saved_context=None, current_context=None, context_valid=None,
                       prefix_context=None):
                    return RE.res1_velocity(current, o, draft, residual, u, saved_context=saved_context,
                                            current_context=current_context, context_valid=context_valid,
                                            use_delta=True)
                return RE.integrate_k1_residual(vf, ob, dr, eps, steps=10,
                                               saved_context=saved, current_context=now, context_valid=valid)
        else:
            @nnx.jit
            def run(current, ob, dr, eps):
                def vf(o, draft, residual, u, saved_context=None, current_context=None, context_valid=None,
                       prefix_context=None):
                    return RE.res1_velocity(current, o, draft, residual, u)
                return RE.integrate_k1_residual(vf, ob, dr, eps, steps=10)
        return run

    def _current_ctx(self, split, state, kv):
        import torch
        qnet = self.qnet
        ix = torch.as_tensor([0], dtype=torch.long, device=self.device)
        h0 = qnet.encode_h(split, ix)
        if self.query == "iii":
            x = h0[:, None]
            for i in range(qnet.xattn_n):
                a, _ = qnet.attn[i](x, kv, kv, need_weights=False)
                x = qnet.post[i](x + a)
                if qnet.ffn is not None:
                    x = qnet.ffn[i](x)
            h = x[:, 0]
        else:
            x = qnet.post0(h0)
            if qnet.ffn0 is not None:
                x = qnet.ffn0(x)
            h = x
        out = {"h": h.detach(), "state": torch.as_tensor(state, device=self.device).reshape(1, 8).float()}
        if self.query == "iii":
            out["vlm"] = kv.detach()
            out["vlm_valid"] = torch.ones(kv.shape[:2], dtype=torch.bool, device=self.device)
        return out

    def infer(self, observation):
        import jax
        import jax.numpy as jnp
        import torch
        started = time.monotonic()
        row, obs = self._obs(observation)
        key = jax.random.key(int(observation["s2_rng_seed"]))
        (x, valid), sn, an, t = self.tokens(observation)
        vlm = None
        kv = None
        if self.query == "iii":
            feats = {k: np.asarray(v) for k, v in self.vlm(obs).items()}
            vlm = np.zeros((1, 34, 2048), np.float16)
            vlm[0, :32] = feats["vlm_img"][0]
            vlm[0, 32] = feats["vlm_txt_mean"][0].astype(np.float16)
            vlm[0, 33] = feats["vlm_txt_last"][0].astype(np.float16)
            kv = self.qnet.kv(torch.as_tensor(vlm, device=self.device).float())
        T = lambda a, dt=torch.float32: torch.as_tensor(a, device=self.device, dtype=dt)
        split = C.InlineSplit(T(x[None]), T(valid[None], torch.bool), None if vlm is None else T(vlm, torch.float16))
        with torch.no_grad():
            qvec = self.qnet(split, T([0], torch.long))
            scores = self.qnet.score(qvec, self.keys).double().cpu().numpy()
        summary = C.summarize_scores(scores, np.ones_like(scores, bool))
        q_np = qvec.float().cpu().numpy()
        feats = C.head_features(q_np, summary["ids"], summary["logits"], summary["entropy"],
                                summary["top1_prob"], summary["top5_mass"], self.keys_np, self.scale)
        has_hist = len(an) > 0
        last = float(an[-1, 6]) if has_hist else C.DEFAULT_GRIP
        gx = C.grip_extra(np.asarray([last]), np.asarray([has_hist]), sn[-1:, 6:8])
        with torch.no_grad():
            ehat = float(np.exp(self.conf(torch.as_tensor(C.head_input(feats, self.ckind))).numpy()[0, 0]))
            pclose = 1 / (1 + np.exp(-self.grip(torch.as_tensor(C.head_input(feats, self.gkind, gx))).numpy()))
        grip = C.decode_gripper(pclose, np.asarray([1. if has_hist and last > 0 else -1.]), self.th)[0]
        unit = int(summary["ids"][0, 0])
        fallback = self.tpol.get("b") is not None and ehat > float(self.tpol["b"])
        if fallback:
            action = np.asarray(self.normal(key, obs, num_steps=10), np.float32).copy()
        else:
            draft = np.zeros((1, 10, 32), np.float32)
            draft[0, :, :6] = self.units[unit, :, :6]
            draft[0, :, 6] = 1. if has_hist and last > 0 else -1.
            noise = jax.random.normal(jax.random.fold_in(key, 7), (1, 10, 6), dtype=jnp.float32)
            if self.use_delta:
                saved = {"h": self.saved_context["h"][unit:unit+1], "state": self.saved_context["state"][unit:unit+1],
                         "vlm": self.saved_context["vlm"][unit:unit+1]}
                current = self._current_ctx(split, sn[-1], kv)
                valid_ctx = current["vlm_valid"]
                saved = jax.tree.map(jnp.asarray, saved)
                current = {k: v.detach().cpu().numpy() for k, v in current.items() if k != 'vlm_valid'}
                current = jax.tree.map(jnp.asarray, current)
                arm = np.asarray(self._residual(self.model, obs, jnp.asarray(draft), noise, saved, current,
                                                jnp.asarray(valid_ctx.detach().cpu().numpy())), np.float32)
            else:
                arm = np.asarray(self._residual(self.model, obs, jnp.asarray(draft), noise), np.float32)
            action = np.asarray(draft, np.float32).copy()
            action[0, :, :6] = arm
            action[0, :, 6] = grip
        physical = self.unnormalize({"actions": action[0], "state": row["state"]})["actions"][:, :7]
        if not np.isfinite(physical).all():
            raise FloatingPointError("RES1 returned nonfinite actions")
        return {"actions": physical.astype(np.float32), "s2_debug": {
            "runtime": "RES1", "query": self.query, "unit_id": unit, "unit_task": int(self.unit_task[unit]),
            "residual_enabled": not fallback, "fallback_pi05": bool(fallback), "confidence_error": ehat,
            "fallback_b": self.tpol.get("b"), "residual_ratio": float(np.linalg.norm(action[0, :, :6] - draft[0, :, :6]) /
                                                                           max(np.linalg.norm(draft[0, :, :6]), 1e-8)) if not fallback else None,
            "gripper_chunk": grip.tolist(), "step_t": t, "seconds": time.monotonic() - started}}


def main():
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, choices=("i", "iii"))
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--condition", default="flow_iii")
    ap.add_argument("--use-delta", action="store_true")
    ap.add_argument("--port", type=int, default=9200)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    policy = RES1Policy(query=a.query, checkpoint=a.checkpoint, use_delta=a.use_delta)
    if a.smoke:
        import kq2cl_obs as O
        row = O.load_obs("dev2")
        elem = {"observation/image": row["image_base"][0], "observation/wrist_image": row["image_wrist"][0],
                "observation/state": row["raw_state"][0], "prompt": "place the object", "kq2/state_hist": np.asarray(row["raw_state"][0], np.float32).reshape(1, 8),
                "kq2/action_hist": np.zeros((0, 7), np.float32), "kq2/t": 0, "s2_rng_seed": 4101}
        out = policy.infer(elem)
        print(json.dumps({"shape": list(out["actions"].shape), "finite": bool(np.isfinite(out["actions"]).all()),
                          "debug": out["s2_debug"]}, indent=2))
        return
    WebsocketPolicyServer(policy, host="127.0.0.1", port=a.port,
                          metadata={"experiment": "RES1", "condition": a.condition, "horizon": 10}).serve_forever()


if __name__ == "__main__":
    main()
