#!/usr/bin/env python3
"""T5 KQ2-CL server: top-candidate disagreement confidence and Expert gripper output."""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

import kq2cl_serve as S
import kq2cl_common as C
from t0x_common import t0_add, t0_from_disagreement


class T5Policy:
    def __init__(self, *, mode, query, expert_checkpoint, bundle, tpol):
        if mode != "memory" or query != "iii":
            raise ValueError("T5 conditions require memory mode and K-x1 query (iii)")
        self.condition = str(tpol["condition"])
        self.gripper_mode = str(tpol["gripper_mode"])
        self.confidence_mode = str(tpol["confidence_mode"])
        calibration = tpol.get("calibration", {})
        if isinstance(calibration, str):
            with open(calibration, encoding="utf-8") as f:
                calibration = json.load(f)
        self.calibration = calibration
        # This model wrapper supplies the same checkpoint, retrieval, heads and normal pi0.5 path as T1.
        self.impl = S.KQ2CLPolicy(mode=mode, query=query, expert_checkpoint=expert_checkpoint,
                                  bundle=bundle, tpol={"c": float(tpol["c"]), "lo": .15,
                                                      "hi": .35, "b": float(tpol["b"])})
        self.b = float(tpol["b"])
        if self.confidence_mode == "disagreement":
            self.d_b = float(self.calibration["D_b"])
            if not np.isfinite(self.d_b) or self.d_b <= 0:
                raise ValueError(f"invalid T5 D_b={self.d_b}")
        else:
            self.d_b = None

    def infer(self, observation):
        p = self.impl
        torch, jax, jnp = p.torch, p.jax, p.jnp
        started = time.monotonic()
        row, obs = p._obs(observation)
        rng_key = jax.random.key(int(observation["s2_rng_seed"]))
        (x, valid), sn, an, step_t = p.tokens(observation)
        f = {k: np.asarray(v) for k, v in p.vlm(obs).items()}
        vlm = np.zeros((1, 34, 2048), np.float16)
        vlm[0, :32] = f["vlm_img"][0]
        vlm[0, 32] = f["vlm_txt_mean"][0].astype(np.float16)
        vlm[0, 33] = f["vlm_txt_last"][0].astype(np.float16)
        T = lambda a, dt=torch.float32: torch.as_tensor(a, device=p.device, dtype=dt)
        split = C.InlineSplit(T(x[None]), T(valid[None], torch.bool), T(vlm, torch.float16))
        with torch.no_grad():
            q = p.qnet(split, T([0], torch.long))
            logits = p.qnet.score(q, p.keys).double().cpu().numpy()
        qn = q.float().cpu().numpy()
        summ = C.summarize_scores(logits, np.ones_like(logits, bool))
        top_ids = np.asarray(summ["ids"][0, :10], np.int64)
        top_scores = np.asarray(summ["logits"][0, :10], np.float64)
        weights = np.exp(top_scores - np.max(top_scores))
        weights /= weights.sum()
        top_arm = np.asarray(p.units[top_ids, :, :6], np.float64)
        d = np.mean(np.square(top_arm - top_arm[:1]), axis=(1, 2))
        disagreement = float(np.sum(weights * d))

        features = C.head_features(qn, summ["ids"], summ["logits"], summ["entropy"],
                                   summ["top1_prob"], summ["top5_mass"], p.keys_np, p.scale)
        last_valid = len(an) > 0
        last_cmd = float(an[-1, 6]) if last_valid else C.DEFAULT_GRIP
        last_sign = 1. if (last_valid and last_cmd > 0) else -1.
        gx = C.grip_extra(np.asarray([last_cmd]), np.asarray([last_valid]), sn[-1:, 6:8])
        with torch.no_grad():
            ehat = float(np.exp(p.conf(torch.as_tensor(C.head_input(features, p.ckind))).numpy()[0, 0]))
            pclose = 1 / (1 + np.exp(-p.grip(torch.as_tensor(C.head_input(features, p.gkind, gx))).numpy()))
        decoded_grip = C.decode_gripper(pclose, np.asarray([last_sign]), p.th)[0]

        unit = int(top_ids[0])
        draft = np.zeros((1, 10, 32), np.float32)
        draft[0, :, :6] = p.units[unit, :, :6]
        if self.gripper_mode == "expert":
            draft[0, :, 6] = last_sign  # Same hold-last convention used when Expert K1:500 was trained.
        elif self.gripper_mode == "decoder":
            draft[0, :, 6] = decoded_grip
        else:
            raise ValueError(self.gripper_mode)

        if self.confidence_mode == "ehat":
            t0 = float(t0_add(ehat, self.b))
            fallback = bool(ehat > self.b)
        elif self.confidence_mode == "disagreement":
            t0 = float(t0_from_disagreement(disagreement, self.calibration))
            fallback = bool(disagreement > self.d_b)
        else:
            raise ValueError(self.confidence_mode)

        if fallback:
            output = np.asarray(p.normal(rng_key, obs, num_steps=10), np.float32).copy()
        else:
            noise = jax.random.normal(jax.random.fold_in(rng_key, 7), (1, 10, 32))
            output = np.asarray(p.bridge(jax.random.key(0), obs, jnp.asarray(draft),
                                         jnp.asarray([t0]), num_steps=10, noise=noise), np.float32).copy()
            # T5 deliberately keeps the Expert's generated gripper output. No decoder overwrite.
        physical = p.unnormalize({"actions": output[0], "state": row["state"]})["actions"][:, :7]
        if not np.isfinite(physical).all():
            raise FloatingPointError("nonfinite T5 action output")
        grip_values = output[0, :, 6].astype(float).tolist()
        grip_switches = int(np.sum(np.asarray(grip_values[1:]) != np.asarray(grip_values[:-1])))
        debug = {
            "runtime": "t0x_t5", "condition": self.condition,
            "query": "iii", "unit_id": unit, "unit_task": int(p.unit_task[unit]),
            "top10_unit_ids": top_ids.tolist(), "top10_probabilities": weights.tolist(),
            "top10_arm_mse_to_top1": d.tolist(), "disagreement_D": disagreement,
            "D_b": self.d_b, "confidence_error": ehat, "t0": t0,
            "fallback_pi05": fallback, "gripper_mode": self.gripper_mode,
            "draft_gripper": draft[0, :, 6].astype(float).tolist(),
            "expert_gripper": grip_values, "gripper_switches_chunk": grip_switches,
            "decoder_pclose": pclose[0].astype(float).tolist(),
            "step_t": step_t, "history_valid_steps": int(valid.sum()),
            "seconds": time.monotonic() - started,
        }
        return {"actions": physical.astype(np.float32), "s2_debug": debug}


def main():
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=("memory",))
    ap.add_argument("--query", required=True, choices=("iii",))
    ap.add_argument("--expert-checkpoint", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--tpol", required=True)
    ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args()
    policy = T5Policy(mode=args.mode, query=args.query, expert_checkpoint=args.expert_checkpoint,
                      bundle=args.bundle, tpol=json.loads(args.tpol))
    WebsocketPolicyServer(policy, host="127.0.0.1", port=args.port,
                          metadata={"experiment": "T0X-T5", "horizon": 10}).serve_forever()


if __name__ == "__main__":
    main()
