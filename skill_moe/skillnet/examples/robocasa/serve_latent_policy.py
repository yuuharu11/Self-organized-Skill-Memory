"""Serve the docs/research/11 policy: latent-token key draft + key-agnostic Expert bridge.

Per query:
  1. π0.5 prefix tokens (3 cameras + prompt) -> shared latent encoder E (query side).
  2. Score every memory unit key (precomputed, same E), take top-1, replay it at its
     natural length, update the own history with it, repeat until 50 steps (as in
     latent_expert.chain_key, but with no same-episode exclusion: the bank is memory only).
  3. Bridge the normalized draft with the trained Expert from t0 (fixed, or chosen per
     query from the draft-vs-π0.5 distance with the Dev-fit selector).
  4. EEF6 comes from the Expert. Gripper, base and mode come from an ordinary π0.5
     sample that shares the observation and noise (the offline eval replaced the
     gripper the same way; base/mode are not in the memory draft).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np

os.environ.setdefault("SKILLNET_SUBTASK_GENERATE", "0")
ROOT = Path(__file__).resolve().parents[4]
for _p in (ROOT / "skill_moe/skillnet/src", ROOT / "skill_moe/skillnet/examples/memory_pilot/robocasa",
           ROOT / "skill_moe/skillnet/examples/robocasa"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import expert as expert_api  # noqa: E402
import latent_skill as ls  # noqa: E402

STATE = Path(os.environ.get("SKILLNET_STATE_DIR", "/work/.skillnet-state"))
OUT = STATE / "outputs/robocasa-memory/latent-skill-01"
HORIZON = 50


def bank_keys(out, mode, run, device, cache):
    """Key-side E embeddings of every memory unit start, cached on disk."""
    import torch
    import latent_key as lk
    if cache.exists():
        d = torch.load(cache, map_location="cpu", weights_only=False)
        if d["run"] == run:
            return d
    inp = lk.Inputs(out, mode, bank_device=None)
    model = lk.new_model(mode).to(device)
    model.load_state_dict(torch.load(Path(out) / "key/runs" / run / "model.pt", map_location=device))
    model.eval()
    n = len(inp.bank_in["task"]); keys = []
    with torch.no_grad():
        for s in range(0, n, 64):
            keys.append(lk.encode(model, inp.batch(inp.bank_in, np.arange(s, min(s + 64, n)), device), key=True).cpu())
    d = {"run": run, "keys": torch.cat(keys), "state_mean": inp.state_mean, "state_std": inp.state_std,
         "length": inp.bank["length"], "natural_actions": inp.bank["natural_actions"],
         "task": inp.bank["task"], "episode": inp.bank["episode"], "start": inp.bank["start"]}
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(d, cache)
    return d


class LatentPolicy:
    def __init__(self, a):
        import torch
        import jax
        import latent_key as lk
        from openpi.shared.nnx_utils import module_jit
        import context_key as ctx

        self.a = a; self.lock = threading.RLock(); self.histories = {}; self.log_lock = threading.Lock()
        out = Path(a.out)
        m = json.loads((out / "manifest.json").read_text())
        self.checkpoint = Path(m["checkpoint"])
        self.stats12 = ls.load_action_stats(self.checkpoint)
        d = json.loads((self.checkpoint / "assets/norm_stats.json").read_text())["norm_stats"]["actions"]
        self.mean32 = np.zeros(32, np.float32); self.std32 = np.ones(32, np.float32)
        mu, sd = np.asarray(d["mean"], np.float32), np.asarray(d["std"], np.float32)
        self.mean32[:len(mu)] = mu[:32]; self.std32[:len(sd)] = sd[:32]
        self.transform = ctx._make_transforms(self.checkpoint)

        sel = json.loads((out / "key/selected_configs.json").read_text())["selected"][a.mode]
        self.device = torch.device("cuda" if torch.cuda.is_available() and a.key_device == "cuda" else "cpu")
        self.bank = bank_keys(out, a.mode, sel["run"], self.device, Path(a.cache_dir) / f"bank_keys_{a.mode}.pt")
        self.keys = self.bank["keys"].to(self.device)
        self.key_model = lk.new_model(a.mode).to(self.device)
        self.key_model.load_state_dict(torch.load(out / "key/runs" / sel["run"] / "model.pt", map_location=self.device))
        self.key_model.eval()
        self.lk = lk; self.torch = torch
        self.store_seq = 3 * ls.IMAGE_TOKENS + 200

        # π0.5 (untrained action expert) for prefix tokens, context, and the ordinary sample.
        self.base, _ = expert_api._load_expert_model(self.checkpoint)
        self.tok_fn = ls._prefix_token_extractor(self.base)
        self.enc = module_jit(self.base.encode_prefix_context)
        self.normal = module_jit(self.base.sample_actions_with_context, static_argnames=("num_steps",))
        self.expert, _ = expert_api._load_expert_model(self.checkpoint)
        expert_api._overlay_expert(self.expert, Path(a.expert))
        self.bridge = module_jit(self.expert.sample_bridge, static_argnames=("num_steps",))
        self.selector = None
        if a.t0_policy == "dist":
            s = json.loads(Path(a.selector).read_text())
            self.selector = (np.asarray(s["t0_grid"], np.float32), np.asarray(s["W"], np.float32))
        self.calls = 0
        self.log_path = Path(a.query_log) if a.query_log else None
        print(json.dumps({"serving": "latent-key-expert", "mode": a.mode, "key_run": sel["run"], "expert": str(a.expert),
                          "t0_policy": a.t0_policy, "t0": a.t0, "bank_units": int(len(self.keys)),
                          "port": a.port}, default=str), flush=True)

    def _query_scores(self, tokens, valid, hist_norm, hmask, state):
        torch, lk = self.torch, self.lk
        seq = tokens.shape[0]
        if seq != self.store_seq:   # the bank used 3x256 image + 200 text tokens
            t = np.zeros((self.store_seq, tokens.shape[1]), np.float16); v = np.zeros(self.store_seq, bool)
            k = min(seq, self.store_seq); t[:k] = tokens[:k]; v[:k] = valid[:k]
            tokens, valid = t, v
        groups = np.where(valid, ls.group_ids(self.store_seq - 3 * ls.IMAGE_TOKENS), -1).astype(np.int64)
        st = (state - self.bank["state_mean"]) / self.bank["state_std"]
        b = {"tokens": torch.as_tensor(tokens[None], device=self.device),
             "groups": torch.as_tensor(groups[None], device=self.device),
             "history": torch.as_tensor(hist_norm[None], device=self.device),
             "history_mask": torch.as_tensor(hmask[None], device=self.device),
             "state": torch.as_tensor(st[None].astype(np.float32), device=self.device)}
        with torch.no_grad():
            q = lk.encode(self.key_model, b, key=False)
            return lk.token_scores(q, self.keys).float().cpu().numpy()[0]

    def infer(self, raw):
        import jax
        import jax.numpy as jnp
        from openpi.models import model as model_api
        from memory_server_utils import ClientHistory
        obs_in = dict(raw)
        client = str(obs_in.pop("client_id", "default")); episode = str(obs_in.pop("episode_id", "0"))
        start_flag = bool(obs_in.pop("episode_start", False))
        state = np.asarray(obs_in["observation/state"], np.float32)
        with self.lock:
            t_start = time.perf_counter()
            hobj = self.histories.setdefault(client, ClientHistory(10))
            hist_phys, hmask, req = hobj.begin_request(episode, start_flag)
            hist = np.zeros((10, 7), np.float32)
            hist[hmask] = ls.normalize_policy_actions(hist_phys[hmask], self.stats12)[:, :7]
            tr = self.transform(obs_in)
            obs = model_api.Observation.from_dict(jax.tree.map(lambda x: jnp.asarray(x)[None, ...], tr))
            tok, val = self.tok_fn(obs)
            tok = np.asarray(tok)[0]; val = np.asarray(val)[0].astype(bool)
            # Chain top-1 units to 50 steps, feeding the own draft back as history.
            parts, filled, units = [], 0, []
            lh, lm = hist.copy(), hmask.copy(); margin = None
            while filled < HORIZON:
                sc = self._query_scores(tok, val, lh, lm, state)
                j = int(sc.argmax())
                if margin is None:
                    top2 = np.sort(np.partition(sc, -2)[-2:]); margin = float(top2[1] - top2[0])
                L = int(self.bank["length"][j]); u = np.asarray(self.bank["natural_actions"][j, :L], np.float32)
                parts.append(u); filled += L
                units.append({"unit": j, "task": int(self.bank["task"][j]), "episode": int(self.bank["episode"][j]),
                              "start": int(self.bank["start"][j]), "length": L, "score": float(sc[j])})
                lh = np.concatenate([lh[lm] if lm.any() else np.zeros((0, 7), np.float32), u[:, :7]])[-10:]
                pad = 10 - len(lh); lm = np.r_[np.zeros(pad, bool), np.ones(len(lh), bool)]
                lh = np.r_[np.zeros((pad, 7), np.float32), lh]
            draft = np.concatenate(parts)[:HORIZON]
            ctx = self.enc(obs)
            key = jax.random.fold_in(jax.random.key(self.a.noise_seed), self.calls)
            noise = jax.random.normal(key, (1, HORIZON, 32))
            pi_norm = np.asarray(self.normal(jax.random.key(0), obs, ctx, num_steps=10, noise=noise), np.float32)[0]
            dist = float(ls.arm6_mse(draft[None, :10, :7], pi_norm[None, :10, :7])[0])
            if self.selector is not None:
                grid, W = self.selector
                t0 = float(grid[int((np.array([1.0, np.log(max(dist, 1e-6))], np.float32) @ W).argmin())])
            else:
                t0 = float(self.a.t0)
            dr = jnp.asarray(expert_api_draft(draft)[None])
            ref = np.asarray(self.bridge(jax.random.key(0), obs, dr, jnp.asarray([t0], jnp.float32), num_steps=10,
                                         noise=noise, prefix_context=ctx), np.float32)[0]
            phys = lambda x: x * (self.std32 + 1e-6) + self.mean32
            ref_p, pi_p = phys(ref)[:, :12], phys(pi_norm)[:, :12]
            out = pi_p.copy(); out[:, :6] = ref_p[:, :6]      # EEF6 from Expert; grip/base/mode from π0.5
            if not np.isfinite(out).all():
                raise AssertionError("non-finite action")
            hobj.set_pending(out[:int(self.a.replan_steps)])
            self.calls += 1
            if self.log_path is not None:
                rec = {"client_id": client, "episode_id": episode, "query_index": int(req), "t0": t0,
                       "dist_pi05": dist, "margin": margin, "units": units, "history_valid": int(hmask.sum()),
                       "seconds": time.perf_counter() - t_start}
                with self.log_lock, self.log_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
            return {"actions": out.astype(np.float32)}


def expert_api_draft(draft7):
    d = np.zeros((HORIZON, 32), np.float32); d[:, :7] = draft7
    return d


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--mode", default="e_vlm_h")
    p.add_argument("--expert", type=Path, default=OUT / "expert/train/key-agnostic-01/expert_005000.msgpack")
    p.add_argument("--t0-policy", choices=("fixed", "dist"), default="fixed")
    p.add_argument("--t0", type=float, default=0.1)
    p.add_argument("--selector", type=Path, default=OUT / "closed-loop/selector_e_vlm_h.json")
    p.add_argument("--cache-dir", type=Path, default=OUT / "closed-loop")
    p.add_argument("--key-device", default="cuda")
    p.add_argument("--noise-seed", type=int, default=4101)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=8000)
    p.add_argument("--query-log", type=Path)
    a = p.parse_args()
    if str(a.out.resolve()).startswith("/work/data"):
        p.error("refusing to use the read-only data mount as output")
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    pol = LatentPolicy(a)
    print(f"serving latent-key-expert on {a.host}:{a.port}", flush=True)
    WebsocketPolicyServer(pol, host=a.host, port=a.port, metadata={"policy": "latent-key-expert", "action_horizon": HORIZON},
                          ping_interval=None, ping_timeout=None, infer_in_thread=True).serve_forever()


if __name__ == "__main__":
    main()
