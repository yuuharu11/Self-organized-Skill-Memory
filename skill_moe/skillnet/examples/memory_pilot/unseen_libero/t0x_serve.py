#!/usr/bin/env python3
"""KQ2-CL serving wrapper with T0X maps and optional pi0.5 gripper diagnostic."""
from __future__ import annotations

import argparse
import json

import numpy as np
import kq2cl_serve as S
from t0x_common import quantile_map, t0_add, t0_from_m


class _T0NumpyProxy:
    """Delegates numpy except sqrt, which implements the T0X additive-radius map."""
    def __init__(self, module, coeff: float):
        self._np = module
        self._coeff = float(coeff)

    def __getattr__(self, name):
        if name == "sqrt":
            return lambda x: self._np.sqrt(.15**2 + self._coeff * self._np.maximum(x, 0.))
        return getattr(self._np, name)


class T0XPolicy:
    def __init__(self, *, mode, query, expert_checkpoint, bundle, tpol):
        self.impl = S.KQ2CLPolicy(mode=mode, query=query, expert_checkpoint=expert_checkpoint, bundle=bundle,
                                  tpol={k: v for k, v in tpol.items() if k not in ("formula", "gripper_mode")})
        self.tpol = dict(tpol)
        self._module = S
        if self.tpol.get("formula") == "additive_radius":
            b = float(self.tpol["b"])
            if not np.isfinite(b) or b <= 0:
                raise ValueError(f"additive-radius formula needs positive finite b, got {b}")
            coeff = (.35**2 - .15**2) / b
            self.impl.tpol = {"c": 1., "lo": .15, "hi": .35, "b": b}
            # The original KQ2-CL infer path computes c*sqrt(ê); this scoped module proxy makes
            # c=1 and sqrt(ê)=sqrt(.15² + coeff*ê), leaving all other numpy operations unchanged.
            self._module.np = _T0NumpyProxy(np, coeff)
        elif "const" in self.tpol:
            self.impl.tpol = {k: v for k, v in self.tpol.items() if k != "gripper_mode"}
        elif self.tpol.get("formula") == "expert_displacement":
            self.impl.tpol = {"const": float(self.tpol.get("probe", .15)), "b": None}
            self.calibration = np.load(self.tpol['calibration'])
            self.probe = float(self.tpol['probe'])
            self.b = float(self.tpol['b'])
        elif self.tpol.get("formula") in ("m_radius", "max_radius"):
            self.impl.tpol = {"const": .30, "b": None}
            cal = self.tpol.get("calibration")
            if not isinstance(cal, dict):
                with open(cal, encoding="utf-8") as f:
                    cal = json.load(f)
            self.calibration = cal
            self.b = float(cal["b"])
            if abs(self.b - float(self.tpol.get("b", self.b))) > 1e-10:
                raise AssertionError("T4 server b differs from its Dev calibration")
            self.probe = float(cal.get("probe_t0", .30))
        else:
            raise ValueError("t0x policy requires additive_radius, expert_displacement or const tpol")

    def _infer_t4(self, observation):
        """Single-pass KQ2-CL query plus probe bridge for T4 m-radius policies."""
        import time

        p = self.impl
        started = time.monotonic()
        torch, jax = p.torch, p.jax
        C = S.C
        row, obs = p._obs(observation)
        rng_key = jax.random.key(int(observation["s2_rng_seed"]))
        (x, valid), sn, an, t = p.tokens(observation)
        vlm = None
        if p.vlm is not None:
            f = {k: np.asarray(v) for k, v in p.vlm(obs).items()}
            vlm = np.zeros((1, 34, 2048), np.float16)
            vlm[0, :32] = f["vlm_img"][0]
            vlm[0, 32] = f["vlm_txt_mean"][0].astype(np.float16)
            vlm[0, 33] = f["vlm_txt_last"][0].astype(np.float16)
        T = lambda a, dt=torch.float32: torch.as_tensor(a, device=p.device, dtype=dt)
        split = C.InlineSplit(T(x[None]), T(valid[None], torch.bool), None if vlm is None else T(vlm, torch.float16))
        with torch.no_grad():
            q = p.qnet(split, T([0], torch.long))
            logits = p.qnet.score(q, p.keys).double().cpu().numpy()
        qn = q.float().cpu().numpy()
        summ = C.summarize_scores(logits, np.ones_like(logits, bool))
        feats = C.head_features(qn, summ["ids"], summ["logits"], summ["entropy"], summ["top1_prob"],
                                summ["top5_mass"], p.keys_np, p.scale)
        last_valid = len(an) > 0
        last_cmd = float(an[-1, 6]) if last_valid else C.DEFAULT_GRIP
        last_sign = 1. if (last_valid and last_cmd > 0) else -1.
        gx = C.grip_extra(np.asarray([last_cmd]), np.asarray([last_valid]), sn[-1:, 6:8])
        with torch.no_grad():
            ehat = float(np.exp(p.conf(torch.as_tensor(C.head_input(feats, p.ckind))).numpy()[0, 0]))
            pclose = 1 / (1 + np.exp(-p.grip(torch.as_tensor(C.head_input(feats, p.gkind, gx))).numpy()))
        grip = C.decode_gripper(pclose, np.asarray([last_sign]), p.th)[0]
        unit = int(summ["ids"][0, 0])

        draft = np.zeros((1, 10, 32), np.float32)
        draft[0, :, :6] = p.units[unit, :, :6]
        draft[0, :, 6] = last_sign
        # One deterministic probe draw per query: trial_seed*100000 + query_id, folded with 7.
        # This is the same single-noise convention used by the KQ2-CL bridge output.
        noise = jax.random.normal(jax.random.fold_in(rng_key, 7), (1, 10, 32))
        probe_out = np.asarray(p.bridge(jax.random.key(0), obs, p.jnp.asarray(draft), p.jnp.asarray([self.probe]),
                                        num_steps=10, noise=noise), np.float32)
        m = float(np.linalg.norm(probe_out[0, :, :6] - draft[0, :, :6]))
        m_prime = float(self.b * (m / float(self.calibration["m_b"])) ** 2)
        t0_m = float(t0_from_m(m, self.calibration))
        t0_e = float(t0_add(ehat, self.b))
        formula = self.tpol["formula"]
        t0 = t0_m if formula == "m_radius" else max(t0_e, t0_m)
        fallback = bool(ehat > self.b)
        if fallback:
            out = np.asarray(p.normal(rng_key, obs, num_steps=10), np.float32).copy()
        else:
            out = np.asarray(p.bridge(jax.random.key(0), obs, p.jnp.asarray(draft), p.jnp.asarray([t0]),
                                      num_steps=10, noise=noise), np.float32).copy()
            out[0, :, 6] = grip
        physical = p.unnormalize({"actions": out[0], "state": row["state"]})["actions"][:, :7]
        if not np.isfinite(physical).all():
            raise FloatingPointError("nonfinite T4 KQ2-CL action output")
        debug = {"runtime": "t0x_t4", "query": p.query, "unit_id": unit,
                 "unit_task": int(p.unit_task[unit]), "confidence_error": ehat,
                 "expert_displacement": m, "m_prime": m_prime, "m_b": float(self.calibration["m_b"]),
                 "p_b": float(self.calibration["p_b"]), "t0_m": t0_m, "t0_e": t0_e, "t0": float(t0),
                 "fallback_pi05": fallback, "step_t": t, "history_valid_steps": int(valid.sum()),
                 "draft_gripper": last_sign, "gripper_chunk": [float(v) for v in out[0, :, 6]],
                 "decoder_pclose": [float(v) for v in pclose[0]], "top1_logit": float(summ["logits"][0, 0]),
                 "gap2": float(summ["logits"][0, 0] - summ["logits"][0, 1]),
                 "entropy": float(summ["entropy"][0]), "t0x_signal": formula,
                 "probe_t0": self.probe, "probe_noise_fold": 7,
                 "seconds": time.monotonic() - started}
        return {"actions": physical.astype(np.float32), "s2_debug": debug}

    def infer(self, observation):
        if self.tpol.get("formula") in ("m_radius", "max_radius"):
            return self._infer_t4(observation)
        result = self.impl.infer(observation)
        if self.tpol.get("formula") == "expert_displacement":
            p = self.impl; dbg = result['s2_debug']
            row, obs = p._obs(observation)
            key = p.jax.random.key(int(observation['s2_rng_seed']))
            unit = int(dbg['unit_id']); draft = np.zeros((1, 10, 32), np.float32)
            draft[0, :, :6] = p.units[unit, :, :6]
            draft[0, :, 6] = float(dbg['draft_gripper'])
            noise = p.jax.random.normal(p.jax.random.fold_in(key, 7), (1, 10, 32))
            probe_out = np.asarray(p.bridge(p.jax.random.key(0), obs, p.jnp.asarray(draft),
                                             p.jnp.asarray([self.probe]), num_steps=10, noise=noise), np.float32)
            m = float(np.linalg.norm(probe_out[0, :, :6] - draft[0, :, :6]))
            mx = self.calibration[f'dev2_iii_m_t{self.probe:.2f}']
            ey = self.calibration['dev2_iii_ehat']
            e_equiv = float(quantile_map(m, mx, ey))
            fallback = e_equiv > self.b
            if fallback:
                raw = np.asarray(p.normal(key, obs, num_steps=10), np.float32)
                physical = p.unnormalize({'actions': raw[0], 'state': row['state']})['actions'][:, :7]
                result['actions'] = physical.astype(np.float32)
            else:
                t0 = float(t0_add(e_equiv, self.b))
                out = np.asarray(p.bridge(p.jax.random.key(0), obs, p.jnp.asarray(draft), p.jnp.asarray([t0]),
                                          num_steps=10, noise=noise), np.float32)
                physical = p.unnormalize({'actions': out[0], 'state': row['state']})['actions'][:, :7]
                result['actions'][:, :6] = physical[:, :6]
            dbg.update({'confidence_error_head': float(dbg.get('confidence_error', 0.)),
                        'confidence_error': e_equiv, 'expert_displacement': m,
                        't0': float(t0_add(e_equiv, self.b)), 'fallback_pi05': bool(fallback),
                        't0x_signal': 'expert_displacement_quantile'})
        if self.tpol.get("gripper_mode") == "pi05":
            p = self.impl
            row, obs = p._obs(observation)
            key = p.jax.random.key(int(observation["s2_rng_seed"]))
            raw = np.asarray(p.normal(key, obs, num_steps=10), np.float32)
            physical = p.unnormalize({"actions": raw[0], "state": row["state"]})["actions"][:, :7]
            result["actions"] = np.asarray(result["actions"], np.float32)
            result["actions"][:, 6] = physical[:, 6]
            result["s2_debug"]["t0x_gripper_source"] = "pi05_diagnostic"
        return result


def main():
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', required=True, choices=('memory', 'pi05'))
    p.add_argument('--query', choices=('iii', 'i')); p.add_argument('--expert-checkpoint', type=str)
    p.add_argument('--bundle', type=str); p.add_argument('--tpol', required=True)
    p.add_argument('--port', type=int, required=True)
    a = p.parse_args(); tpol = json.loads(a.tpol)
    policy = T0XPolicy(mode=a.mode, query=a.query, expert_checkpoint=a.expert_checkpoint,
                       bundle=a.bundle, tpol=tpol)
    WebsocketPolicyServer(policy, host='127.0.0.1', port=a.port,
                          metadata={'experiment': 'T0X', 'mode': a.mode, 'query': a.query, 'horizon': 10}).serve_forever()


if __name__ == '__main__':
    main()

