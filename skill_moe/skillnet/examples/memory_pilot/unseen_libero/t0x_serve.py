#!/usr/bin/env python3
"""KQ2-CL serving wrapper with T0X maps and optional pi0.5 gripper diagnostic."""
from __future__ import annotations

import argparse
import json

import numpy as np
import kq2cl_serve as S
from t0x_common import quantile_map, t0_add


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
        else:
            raise ValueError("t0x policy requires additive_radius, expert_displacement or const tpol")

    def infer(self, observation):
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

