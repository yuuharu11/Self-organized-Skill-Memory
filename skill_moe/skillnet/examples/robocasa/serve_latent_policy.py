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
        self.unit_coeff = np.asarray(np.load(out / "units/memory_units.npz")["coeff"], np.float32)
        if len(self.unit_coeff) != len(self.keys):
            raise ValueError("unit coefficient and key bank rows differ")
        self.prev_selected = {}; self.prev_episode = {}
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
        self.conf_model = self.conf_policy = None
        self.gripper_classifier = None
        self.latent_heads = self.latent_error_model = self.latent_grip_model = None
        self.latent_error_config = self.latent_grip_config = None
        self.latent_fallback_threshold = None
        self.t0_map_quantiles = None
        self.cross_mode = self.cross_model = self.cross_keys = None
        if a.t0_policy == "conf":
            conf_all = json.loads(Path(a.conf_config).read_text())
            conf_mode = conf_all[a.mode]
            self.conf_model = conf_mode["feature_model"]
            self.conf_policy = conf_mode["a_b_policies"][str(a.conf_rate)]
            if not float(self.conf_policy["a"]) < float(self.conf_policy["b"]):
                raise ValueError("confidence policy requires a < b")
            train_cfg = Path(a.expert).parent / "config.json"
            if not train_cfg.exists():
                raise FileNotFoundError(f"missing Expert t0 map config: {train_cfg}")
            self.t0_map_quantiles = np.asarray(json.loads(train_cfg.read_text())["t0_map_quantiles"], np.float64)
            if self.t0_map_quantiles.shape != (101,):
                raise ValueError("Expert T0Map must have 101 training quantiles")
            self.cross_mode = "e_vlm_h" if a.mode == "p5" else "p5"
            cross_sel = json.loads((out / "key/selected_configs.json").read_text())["selected"][self.cross_mode]
            self.cross_model = lk.new_model(self.cross_mode).to(self.device)
            self.cross_model.load_state_dict(torch.load(out / "key/runs" / cross_sel["run"] / "model.pt", map_location=self.device))
            self.cross_model.eval()
            self.cross_bank = bank_keys(out, self.cross_mode, cross_sel["run"], self.device,
                                        Path(a.cache_dir) / f"bank_keys_{self.cross_mode}.pt")
            for field in ("task", "episode", "start", "length"):
                if not np.array_equal(np.asarray(self.bank[field]), np.asarray(self.cross_bank[field])):
                    raise ValueError(f"cross-key banks are not row-aligned: {field}")
            self.cross_keys = self.cross_bank["keys"].to(self.device)
            if a.gripper_checkpoint is not None:
                from confidence_gripper import GripperClassifier
                self.gripper_classifier = GripperClassifier(a.gripper_checkpoint)
        elif a.t0_policy == "latent":
            import latent_heads as lh
            self.latent_heads = torch.load(a.latent_heads, map_location="cpu", weights_only=False)
            ec, gc = self.latent_heads["error"], self.latent_heads["grip"]
            self.latent_error_model = lh.ErrorHead.make(int(ec["input_dim"]))
            self.latent_error_model.load_state_dict(ec["state"]); self.latent_error_model.eval()
            self.latent_grip_model = lh.GripHead.make(1280, int(gc["hidden"]))
            self.latent_grip_model.load_state_dict(gc["state"]); self.latent_grip_model.eval()
            self.latent_error_config, self.latent_grip_config = ec, gc
            sel_heads = json.loads((out / "latent-heads/selection.json").read_text())
            self.latent_fallback_threshold = float(sel_heads["fallback_thresholds_dev"][str(a.latent_fallback_rate)])
            self.t0_map_quantiles = np.asarray(self.latent_heads["t0_quantiles"], np.float64)
            if self.latent_error_config["selected"] == "latent_plus_7":
                self.cross_mode = "p5"
                cross_sel = json.loads((out / "key/selected_configs.json").read_text())["selected"][self.cross_mode]
                self.cross_model = lk.new_model(self.cross_mode).to(self.device)
                self.cross_model.load_state_dict(torch.load(out / "key/runs" / cross_sel["run"] / "model.pt", map_location=self.device))
                self.cross_model.eval()
                self.cross_bank = bank_keys(out, self.cross_mode, cross_sel["run"], self.device,
                                            Path(a.cache_dir) / f"bank_keys_{self.cross_mode}.pt")
                for field in ("task", "episode", "start", "length"):
                    if not np.array_equal(np.asarray(self.bank[field]), np.asarray(self.cross_bank[field])):
                        raise ValueError(f"cross-key banks are not row-aligned: {field}")
                self.cross_keys = self.cross_bank["keys"].to(self.device)
        self.calls = 0
        self.log_path = Path(a.query_log) if a.query_log else None
        print(json.dumps({"serving": "latent-key-expert", "mode": a.mode, "key_run": sel["run"], "expert": str(a.expert),
                          "t0_policy": a.t0_policy, "t0": a.t0, "bank_units": int(len(self.keys)),
                          "port": a.port}, default=str), flush=True)

    def _query_batch(self, tokens, valid, hist_norm, hmask, state):
        torch = self.torch
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
        # P0/P5 use the masked mean of valid prefix tokens, matching TokenStore.pooled.
        pooled = tokens[valid].astype(np.float32).mean(0)
        b["pooled"] = torch.as_tensor(pooled[None], device=self.device)
        return b

    def _query_scores(self, tokens, valid, hist_norm, hmask, state, mode=None, model=None, keys=None, return_embedding=False):
        torch = self.torch
        mode = mode or self.a.mode
        model = model or self.key_model
        keys = self.keys if keys is None else keys
        b = self._query_batch(tokens, valid, hist_norm, hmask, state)
        with torch.no_grad():
            q = self.lk.encode(model, b, key=False)
            scores = self.lk.token_scores(q, keys).float().cpu().numpy()[0]
            return (scores, q) if return_embedding else scores

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
            if start_flag or self.prev_episode.get(client) not in (None, episode):
                self.prev_selected.pop(client, None)
                self.prev_episode.pop(client, None)
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
            main_scores = None; selected_j = None; query_latent = None
            while filled < HORIZON:
                if filled == 0 and self.a.t0_policy == "latent":
                    sc, query_latent = self._query_scores(tok, val, lh, lm, state, return_embedding=True)
                else:
                    sc = self._query_scores(tok, val, lh, lm, state)
                j = int(sc.argmax())
                if filled == 0:
                    main_scores = sc.copy(); selected_j = j
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
            pi_norm = None; dist = None; predicted_error = None; fallback = False; conf_features = None
            if self.selector is not None:
                pi_norm = np.asarray(self.normal(jax.random.key(0), obs, ctx, num_steps=10, noise=noise), np.float32)[0]
                dist = float(ls.arm6_mse(draft[None, :10, :7], pi_norm[None, :10, :7])[0])
                grid, W = self.selector
                t0 = float(grid[int((np.array([1.0, np.log(max(dist, 1e-6))], np.float32) @ W).argmin())])
            elif self.a.t0_policy == "conf":
                cross_scores = self._query_scores(tok, val, hist, hmask, state, mode=self.cross_mode,
                                                  model=self.cross_model, keys=self.cross_keys)
                cross_top = int(cross_scores.argmax())
                top10 = np.argpartition(main_scores, -10)[-10:]
                top2 = np.partition(main_scores, -2)[-2:]
                logit_scale = float(self.key_model.logit_scale.exp().clamp(max=self.lk.LOGIT_SCALE_MAX).item())
                z = main_scores[top10].astype(np.float64) * logit_scale
                z -= z.max(); prob = np.exp(z); prob /= prob.sum()
                entropy = float(-(prob * np.log(np.maximum(prob, 1e-30))).sum())
                spread = float(np.sqrt(np.mean((self.unit_coeff[top10] - self.unit_coeff[selected_j]) ** 2)))
                valid_hist = np.flatnonzero(hmask)
                last_action = hist[valid_hist[-1], :6] if valid_hist.size else np.zeros(6, np.float32)
                jump = float(np.sqrt(np.mean((draft[0, :6] - last_action) ** 2)))
                same = float(self.prev_episode.get(client) == episode and self.prev_selected.get(client) == selected_j)
                cross_move = float(np.sqrt(np.mean((self.unit_coeff[selected_j] - self.unit_coeff[cross_top]) ** 2)))
                conf_features = {"k_top1": float(main_scores[selected_j]), "k_gap": float(top2.max() - top2.min()),
                                 "k_entropy10": entropy, "r_spread10": spread, "r_p5_vs_e_top1": cross_move,
                                 "c_draft_jump": jump, "c_same_unit_repeat": same}
                fm = self.conf_model
                xv = np.asarray([conf_features[k] for k in fm["features"]], np.float64)
                zv = np.nan_to_num((xv - np.asarray(fm["center"])) / np.asarray(fm["scale"]))
                predicted_error = float(np.exp(np.clip(fm["intercept"] + zv @ np.asarray(fm["coefficients"]), -20, 20)))
                a0, b0 = float(self.conf_policy["a"]), float(self.conf_policy["b"])
                if predicted_error >= b0:
                    fallback = True; t0 = None
                else:
                    # Match training: t0 = .1 + .8 * empirical_CDF_train(error).
                    # The old conf ramp collapsed most queries at t0=.1.
                    from confidence_gripper import t0_from_error
                    t0 = float(t0_from_error(predicted_error, self.t0_map_quantiles))
                if fallback:
                    pi_norm = np.asarray(self.normal(jax.random.key(0), obs, ctx, num_steps=10, noise=noise), np.float32)[0]
            elif self.a.t0_policy == "latent":
                import torch
                from confidence_gripper import t0_from_error
                top10 = np.argpartition(main_scores, -10)[-10:]
                top2 = np.partition(main_scores, -2)[-2:]
                scale = float(self.key_model.logit_scale.exp().clamp(max=self.lk.LOGIT_SCALE_MAX).item())
                z = main_scores[top10].astype(np.float64) * scale; z -= z.max()
                prob = np.exp(z); prob /= prob.sum()
                entropy = float(-(prob * np.log(np.maximum(prob, 1e-30))).sum())
                spread = float(np.sqrt(np.mean((self.unit_coeff[top10] - self.unit_coeff[selected_j]) ** 2)))
                cross_move = 0.0
                if self.latent_error_config["selected"] == "latent_plus_7":
                    cross_scores = self._query_scores(tok, val, hist, hmask, state, mode="p5", model=self.cross_model, keys=self.cross_keys)
                    cross_top = int(cross_scores.argmax())
                    cross_move = float(np.sqrt(np.mean((self.unit_coeff[selected_j] - self.unit_coeff[cross_top]) ** 2)))
                valid_hist = np.flatnonzero(hmask)
                last = hist[valid_hist[-1], :6] if len(valid_hist) else np.zeros(6, np.float32)
                jump = float(np.sqrt(np.mean((draft[0, :6] - last) ** 2)))
                repeated = float(self.prev_episode.get(client) == episode and self.prev_selected.get(client) == selected_j)
                latent_signals = np.asarray([float(main_scores[selected_j]), float(top2.max()-top2.min()), entropy,
                                             spread, cross_move, jump, repeated], np.float32)
                qn = query_latent.detach().cpu().numpy().reshape(1, 5, 256).astype(np.float32)
                kn = self.keys[selected_j:selected_j+1].detach().cpu().numpy().astype(np.float32)
                xerr = np.concatenate([qn.reshape(1,-1), kn.reshape(1,-1), (qn-kn).reshape(1,-1), (qn*kn).reshape(1,-1)], axis=1)
                if self.latent_error_config["selected"] == "latent_plus_7":
                    xerr = np.concatenate([xerr, latent_signals[None]], axis=1)
                ec = self.latent_error_config
                xerr = ((xerr - np.asarray(ec["center"], np.float32)) / np.asarray(ec["scale"], np.float32)).astype(np.float32)
                with torch.no_grad():
                    predicted_error = float(np.exp(np.clip(self.latent_error_model(torch.from_numpy(xerr)).item(), -20, 20)))
                fallback = predicted_error >= self.latent_fallback_threshold
                t0 = None if fallback else float(t0_from_error(predicted_error, self.t0_map_quantiles))
                latent_signal_log = {"k_top1":float(main_scores[selected_j]),"k_gap":float(top2.max()-top2.min()),
                                     "k_entropy10":entropy,"r_spread10":spread,"r_p5_vs_e_top1":cross_move,
                                     "c_draft_jump":jump,"c_same_unit_repeat":repeated}
                if fallback:
                    pi_norm = np.asarray(self.normal(jax.random.key(0), obs, ctx, num_steps=10, noise=noise), np.float32)[0]
            else:
                t0 = float(self.a.t0)
            phys = lambda x: x * (self.std32 + 1e-6) + self.mean32
            if self.a.t0_policy in ("conf", "latent"):
                if fallback:
                    out = phys(pi_norm)[:, :12]
                    gripper_source = "pi05_fallback"
                else:
                    dr = jnp.asarray(expert_api_draft(draft)[None])
                    ref = np.asarray(self.bridge(jax.random.key(0), obs, dr, jnp.asarray([t0], jnp.float32), num_steps=10,
                                                 noise=noise, prefix_context=ctx), np.float32)[0]
                    # In conf mode the ordinary π0.5 sample is only generated on fallback.
                    out = phys(ref)[:, :12]
                    if self.gripper_classifier is not None:
                        pooled = tok[val].astype(np.float32).mean(0)
                        grip = self.gripper_classifier.predict(pooled, hist, state)
                        out[:, 6] = grip[:len(out)]
                        out[:, 7:12] = self.gripper_classifier.fixed_tail[1:][None, :]
                        gripper_source = "classifier"
                    elif self.a.t0_policy == "latent":
                        gc = self.latent_grip_config
                        gx = ((query_latent.detach().cpu().numpy().reshape(1,-1).astype(np.float32) -
                               np.asarray(gc["center"], np.float32)) / np.asarray(gc["scale"], np.float32)).astype(np.float32)
                        with self.torch.no_grad():
                            grip_logits = self.latent_grip_model(self.torch.from_numpy(gx)).numpy()[0]
                        out[:, 6] = np.where(grip_logits[:len(out)] >= 0, 1., -1.)
                        out[:, 7:12] = np.asarray(self.latent_heads["tail_values"], np.float32)[1:][None, :]
                        gripper_source = "latent_token_head"
                    else:
                        gripper_source = "legacy_expert_output"
            else:
                if pi_norm is None:
                    pi_norm = np.asarray(self.normal(jax.random.key(0), obs, ctx, num_steps=10, noise=noise), np.float32)[0]
                dr = jnp.asarray(expert_api_draft(draft)[None])
                ref = np.asarray(self.bridge(jax.random.key(0), obs, dr, jnp.asarray([t0], jnp.float32), num_steps=10,
                                             noise=noise, prefix_context=ctx), np.float32)[0]
                ref_p, pi_p = phys(ref)[:, :12], phys(pi_norm)[:, :12]
                out = pi_p.copy(); out[:, :6] = ref_p[:, :6]      # EEF6 from Expert; grip/base/mode from π0.5
            if not np.isfinite(out).all():
                raise AssertionError("non-finite action")
            hobj.set_pending(out[:int(self.a.replan_steps)])
            if self.a.t0_policy in ("conf", "latent"):
                self.prev_selected[client] = int(selected_j)
                self.prev_episode[client] = episode
            self.calls += 1
            if self.log_path is not None:
                rec = {"client_id": client, "episode_id": episode, "query_index": int(req), "t0": t0,
                       "dist_pi05": dist, "margin": margin, "units": units, "history_valid": int(hmask.sum()),
                       "seconds": time.perf_counter() - t_start}
                if self.a.t0_policy == "conf":
                    rec.update({"predicted_error": predicted_error, "fallback": fallback, "confidence_features": conf_features,
                                "fallback_rate_candidate": int(self.a.conf_rate),
                                "gripper_source": gripper_source,
                                "gripper_open_fraction": float(np.mean(out[:, 6] > 0)),
                                "base_mode_source": "pi05_fallback" if fallback else ("train_modal_constant" if self.gripper_classifier is not None else "legacy_expert_output")})
                elif self.a.t0_policy == "latent":
                    rec.update({"predicted_error": predicted_error, "fallback": fallback,
                                "fallback_threshold": self.latent_fallback_threshold, "latent_signals": latent_signal_log,
                                "gripper_source": gripper_source, "gripper_open_fraction": float(np.mean(out[:, 6] > 0)),
                                "base_mode_source": "pi05_fallback" if fallback else "train_modal_constant"})
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
    p.add_argument("--t0-policy", choices=("fixed", "dist", "conf", "latent"), default="fixed")
    p.add_argument("--t0", type=float, default=0.1)
    p.add_argument("--selector", type=Path, default=OUT / "closed-loop/selector_e_vlm_h.json")
    p.add_argument("--conf-config", type=Path, default=OUT / "conf/closed/models.json")
    p.add_argument("--conf-rate", type=int, choices=(0, 20, 40), default=20)
    p.add_argument("--gripper-checkpoint", type=Path, default=None,
                   help="RCX5 classifier; when set, non-fallback grip uses the classifier and base/mode use train modal constants")
    p.add_argument("--latent-heads", type=Path, default=OUT / "latent-heads/heads.pt")
    p.add_argument("--latent-fallback-rate", type=int, choices=(0,10,20,30), default=20)
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
