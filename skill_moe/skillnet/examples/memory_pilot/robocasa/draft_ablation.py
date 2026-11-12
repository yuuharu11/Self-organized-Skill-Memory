"""RM-DRAFT-ABL server wrapper around the existing memory bridge policy.

The offline mode evaluates several drafts at an identical observation and
bridge-noise draw, then returns the canonical draft action. Closed-loop modes
return either a uniformly sampled memory unit or a static EEF draft.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import threading
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
ROBOCASA_EXAMPLES = HERE.parents[1] / "robocasa"
if str(ROBOCASA_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(ROBOCASA_EXAMPLES))

import serve_memonly_policy as base


TASKS = ["PickPlaceCounterToCabinet", "PickPlaceSinkToCounter", "OpenDrawer",
         "CloseFridge", "TurnOnSinkFaucet", "TurnOffStove"]


def _norm(v):
    return float(np.linalg.norm(np.asarray(v, np.float64).reshape(-1)))


class DraftAblationPolicy(base.MemoryOnlyPolicy):
    def __init__(self, args):
        super().__init__(args)
        self.ablation_mode = args.ablation_mode
        self.ablation_path = Path(args.ablation_log) if args.ablation_log else None
        self.ablation_limit = int(args.ablation_queries)
        self.ablation_counts = {t: 0 for t in TASKS}
        self.ablation_lock = threading.Lock()
        self._active_mode = "original"
        self._active_units = []
        self._active_log = None
        original_bridge = self.bridge_fn
        if original_bridge is not None:
            def bridge(key, observation, draft, t0, *, num_steps, noise):
                if self._active_mode == "static":
                    draft = np.asarray(draft).copy()
                    stats = self.stats["actions"]
                    mean = np.asarray(stats.mean, np.float32)[:6]
                    std = np.asarray(stats.std, np.float32)[:6]
                    draft[..., :6] = (0.0 - mean) / (std + 1e-6)
                result = original_bridge(key, observation, draft, t0,
                                          num_steps=num_steps, noise=noise)
                return result
            self.bridge_fn = bridge

    def _log(self, record):
        if self._active_log is not None:
            self._active_log.append(record)
            return
        return super()._log(record)

    def _choose(self, q_context, history, mask):
        mode = self._active_mode
        selected, score, rank_query = super()._choose(q_context, history, mask)
        if mode == "original" or mode == "static":
            return selected, score, rank_query
        if mode == "random":
            # Per-query deterministic RNG: unaffected by server request order.
            seed = int.from_bytes(hashlib.sha256(np.asarray(rank_query, np.float32).tobytes()).digest()[:8], "little")
            rng = np.random.default_rng(seed)
            return int(rng.integers(len(self.pool["length"]))), score, rank_query
        if mode == "other_task_top1":
            torch = self._extra[0]
            q = torch.as_tensor(np.asarray(rank_query, np.float32)[None], dtype=torch.float32)
            with torch.no_grad():
                qemb = torch.nn.functional.normalize(self.rank_net.query(q), dim=-1)
                scores = (qemb @ self.candidate_embeddings.T)[0]
            task_id = int(getattr(self, "_query_task_id", int(self.pool["task"][selected])))
            allowed = torch.as_tensor(np.asarray(self.pool["task"]) != task_id, device=scores.device)
            scores = scores.masked_fill(~allowed, -torch.inf)
            alt = int(torch.argmax(scores).item())
            return alt, float(scores[alt].item()), rank_query
        raise ValueError(f"unknown ablation mode {mode}")

    def _one_infer(self, raw_observation, mode, capture=False):
        with self.lock:
            self._active_mode = mode
            self._active_units = []
            logs = []
            before = base.materialize_unit
            def materialize(*args, **kwargs):
                unit = before(*args, **kwargs)
                if capture:
                    self._active_units.append(np.asarray(unit, np.float32).copy())
                return unit
            base.materialize_unit = materialize
            old_log = self._active_log
            if capture:
                self._active_log = logs
            try:
                result = super().infer(raw_observation)
            finally:
                base.materialize_unit = before
                self._active_log = old_log
            draft = np.concatenate(self._active_units, axis=0)[:50] if self._active_units else None
            return result, draft, logs

    def infer(self, raw_observation):
        episode_for_task = str(raw_observation.get("episode_id", ""))
        task_name_for_id = next((t for t in TASKS if episode_for_task.startswith(t + ":")), None)
        if task_name_for_id is not None:
            self._query_task_id = TASKS.index(task_name_for_id)
        if self.ablation_mode != "offline":
            return self._one_infer(raw_observation, self.ablation_mode)[0]

        episode = str(raw_observation.get("episode_id", ""))
        task_name = next((t for t in TASKS if episode.startswith(t + ":")), None)
        if task_name is None or self.ablation_counts[task_name] >= self.ablation_limit:
            return self._one_infer(raw_observation, "original")[0]

        # Candidate diagnostics are pure branches from the same pre-query
        # history. The request index (hence Gaussian bridge noise) is identical.
        initial_histories = copy.deepcopy(self.histories)
        initial_calls = self.action_expert_calls
        branches = {}
        modes = ("original", "random", "static", "other_task_top1")
        for mode in modes:
            self.histories = copy.deepcopy(initial_histories)
            self.action_expert_calls = initial_calls
            output, draft, logs = self._one_infer(raw_observation, mode, capture=True)
            branches[mode] = (np.asarray(output["actions"], np.float32), draft, logs)
        # Return the real policy's output and retain only its history update.
        canonical_histories = copy.deepcopy(self.histories)
        self.histories = copy.deepcopy(initial_histories)
        self.action_expert_calls = initial_calls
        canonical, canonical_draft, canonical_logs = self._one_infer(
            raw_observation, "original", capture=True)
        canonical_histories = copy.deepcopy(self.histories)
        self.histories = canonical_histories
        self.action_expert_calls = initial_calls + 1
        if not np.allclose(np.asarray(canonical["actions"]), branches["original"][0], rtol=1e-5, atol=1e-6):
            raise AssertionError("canonical output differs from its same-observation offline branch")

        base_log = canonical_logs[-1] if canonical_logs else {}
        first_units = base_log.get("selected_units", [])
        unit_horizon = max(1, min(50, int(first_units[0]["length"]))) if first_units else 50
        ref_draft = branches["original"][1]
        if ref_draft is None:
            ref_draft = canonical_draft
        record = {"task": task_name, "episode_id": episode,
                  "query_index": base_log.get("query_index"),
                  "unit_horizon": unit_horizon, "noise_shared": True,
                  "selected_units": first_units, "branches": {}}
        for mode, (actions, draft, _) in branches.items():
            if draft is None or ref_draft is None:
                continue
            if mode == "static":
                draft = draft.copy(); draft[:, :6] = 0.0
            branch_log = branches[mode][2][-1] if branches[mode][2] else {}
            record["branches"].setdefault(mode, {})["selected_units"] = branch_log.get("selected_units", [])
            for label, h in (("first5", min(5, len(actions))),
                             ("unit", min(unit_horizon, len(actions)))):
                d_out = _norm(actions[:h] - branches["original"][0][:h])
                d_draft = _norm(draft[:h] - ref_draft[:h])
                record["branches"].setdefault(mode, {})[label] = {
                    "delta_output_l2": d_out, "delta_draft_l2": d_draft,
                    "ratio": d_out / d_draft if d_draft > 1e-12 else None,
                }
        self.ablation_counts[task_name] += 1
        if self.ablation_path:
            self.ablation_path.parent.mkdir(parents=True, exist_ok=True)
            with self.ablation_lock, self.ablation_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return canonical


def main():
    import os
    import torch
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    p = argparse.ArgumentParser()
    state = Path(os.environ.get("SKILLNET_STATE_DIR", "/work/.skillnet-state"))
    root = state / "outputs/robocasa-memory"
    p.add_argument("--checkpoint", type=Path, default=Path(os.environ.get("SKILLNET_ROBOCASA_CHECKPOINT", state / "checkpoints/robocasa365/pi05_pretrain_human300/multitask_learning/75000")))
    p.add_argument("--memory-dir", type=Path, default=root / "memory-01/policy-order-v2")
    p.add_argument("--context-dir", type=Path, default=root / "ctx-01")
    p.add_argument("--key-checkpoint", type=Path, default=root / "key-lite-01/context_history.pt")
    p.add_argument("--magnitude-checkpoint", type=Path, default=root / "key-lite-01/magnitude/magnitude_head.pt")
    p.add_argument("--magnitude-multiplier", type=float, default=1.0)
    p.add_argument("--bridge-checkpoint", type=Path, default=None)
    p.add_argument("--bridge-use-base-expert", action="store_true")
    p.add_argument("--bridge-t0", type=float, default=.2); p.add_argument("--bridge-steps", type=int, default=10)
    p.add_argument("--bridge-noise-seed", type=int, default=4101)
    p.add_argument("--gripper-checkpoint", type=Path, default=root / "ctx-01/context_model.pt")
    p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=8000)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--query-log", type=Path); p.add_argument("--diagnostic-log", type=Path)
    p.add_argument("--diagnostic-reference-count", type=int, default=0)
    p.add_argument("--ablation-mode", choices=("original", "random", "static", "offline"), default="original")
    p.add_argument("--ablation-log", type=Path); p.add_argument("--ablation-queries", type=int, default=40)
    args = p.parse_args()
    if args.magnitude_multiplier <= 0 or args.ablation_queries < 0:
        p.error("magnitude multiplier must be positive and ablation query count nonnegative")
    os.environ["SKILLNET_SUBTASK_GENERATE"] = "0"
    policy = DraftAblationPolicy(args)
    label = "memory-bridge" if policy.bridge_fn is not None else "memory-only"
    print(f"serving rm-{label}-draft-ablation on {args.host}:{args.port}; mode={args.ablation_mode}", flush=True)
    WebsocketPolicyServer(policy, host=args.host, port=args.port,
        metadata={"policy": label, "action_horizon": 50}, ping_interval=None,
        ping_timeout=None, infer_in_thread=True).serve_forever()


if __name__ == "__main__":
    main()
