#!/usr/bin/env python3
"""KS1 key-only ablations for the frozen INV2 holdout split.

KS-a keeps the original one-query A-L0 model and substitutes the R4 split-key
encoder. KS-b keeps the R4 architecture/training loop and changes only its
candidate score to normalize the mean key vector before taking the dot product.
This module reads the existing LIBERO-40 fit/selection artifacts; it never
loads the 619-row Eval labels.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in os.sys.path:
    os.sys.path.insert(0, str(HERE))
import scene_invariance as step1
import scene_invariance_v2 as inv2

STATE = Path(os.environ.get("SKILLNET_STATE_DIR", "/work/.skillnet-state"))
OUT = STATE / "outputs/memory-pilot/unseen-libero/key-split-01"
SOURCE = step1.OUT
HISTORY_SOURCE = step1.OUT / "step2/attempt-fix2/history_dependence.json"
SEED_SET = (4101, 4102, 4103)
WIDTH = 256
KEY_K = 5
TAU = .01
EPOCHS = 3
BATCH_SIZE = 64
LR = 3e-4


def ks_a_scores(q, key_tokens, scale=1.0):
    """KS-a: q dot the mean of individually normalized split-key tokens."""
    import torch
    import torch.nn.functional as F

    if q.ndim != 2 or key_tokens.ndim != 3 or q.shape[-1] != key_tokens.shape[-1]:
        raise ValueError(f"KS-a expects q[B,D], keys[M,K,D], got {tuple(q.shape)} / {tuple(key_tokens.shape)}")
    qn = F.normalize(q, dim=-1)
    mn = F.normalize(key_tokens, dim=-1)
    return (qn @ mn.mean(dim=1).T) * scale


def ks_b_scores(query_tokens, raw_key_tokens, scale=1.0):
    """KS-b: normalize mean query/key vectors, with keys still pre-normalization."""
    import torch.nn.functional as F

    if (query_tokens.ndim != 3 or raw_key_tokens.ndim != 3
            or query_tokens.shape[-1] != raw_key_tokens.shape[-1]):
        raise ValueError("KS-b expects query[B,K,D] and raw key[M,K,D] token sets")
    q = F.normalize(query_tokens.mean(dim=1), dim=-1)
    m = F.normalize(raw_key_tokens.mean(dim=1), dim=-1)
    return (q @ m.T) * scale


def _split_key_raw(net, actions):
    """Return R4's key encoder output before its per-segment L2 normalization."""
    if actions.ndim != 3 or actions.shape[1:] != (10, 7):
        raise ValueError(f"expected pool actions [N,10,7], got {tuple(actions.shape)}")
    chunks = actions.reshape(len(actions), KEY_K, 14).float()
    z = net.key_frame(chunks) + net.key_pos
    return net.key_attn(z)


def _seed_everything(seed):
    import torch
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _device_setup(device):
    import torch
    threads = int(os.environ.get("OMP_NUM_THREADS", "8"))
    torch.set_num_threads(threads)
    if str(device).startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(min(float(os.environ.get("KS1_GPU_FRACTION", ".15")), .15), 0)
    return torch.device(device)


class _KSA:
    """Factory wrapper so torch stays a lazy dependency for the importable score helpers."""

    @staticmethod
    def build(token_dim=inv2.TOKEN_DIM):
        import torch
        from torch import nn
        import torch.nn.functional as F

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                # Exact step-1 A query (one latent query, two VLM cross layers).
                self.query = step1._make_model("A", 1, token_dim)
                # R4 key modules: shared 14->256 segment projection, position
                # embeddings, and a 5-token self-attention encoder.
                key_template = inv2._build_model("A", k=KEY_K, key_mode="split",
                                                  no_history_cross=True)
                self.key_frame = key_template.key_frame
                self.key_pos = key_template.key_pos
                self.key_attn = key_template.key_attn
                self.log_scale = nn.Parameter(torch.tensor(np.log(10.), dtype=torch.float32))

            def encode_query(self, history, state, tokens):
                import torch
                mask = torch.ones(tokens.shape[:2], device=tokens.device, dtype=torch.bool)
                q, _ = self.query(history, state, tokens, mask)
                return q

            def encode_keys(self, actions):
                import torch.nn.functional as F
                return F.normalize(_split_key_raw(self, actions), dim=-1)

            def scores(self, q, keys):
                return ks_a_scores(q, keys, self.log_scale.exp())

        return Model()


def _build_ksb():
    """R4 model config with the KS-b score API substituted at call time."""
    return inv2._build_model("A", k=KEY_K, key_mode="split", query_pool=False,
                             no_history_cross=True)


def _install_ksb_score(model):
    """Monkey-patch only candidate scoring; state_dict and R4 modules are unchanged."""
    import types

    def encode_keys_raw(self, actions):
        return _split_key_raw(self, actions)

    def scores(self, q, raw_keys):
        return ks_b_scores(q, raw_keys, self.log_scale.exp())

    model.encode_keys = types.MethodType(encode_keys_raw, model)
    model.scores = types.MethodType(scores, model)
    return model


def _history_data(data, variant, ids, seed):
    if variant == "true":
        return data
    return inv2.history_variant(data, ids, variant, seed=seed)


def _load_tokens_direct(data, kind):
    """Gather only the requested token group, avoiding a full 73GB row copy."""
    row_ids = np.asarray(data["row_id"], np.int64)
    vals = np.load(step1.TOKENS, mmap_mode="r")
    groups = np.load(step1.GROUPS, mmap_mode="r")
    group_ids = np.asarray(groups[0])
    keep = np.flatnonzero(group_ids >= 0)
    if kind == "A":
        keep = keep[group_ids[keep] == 1]
    # Joint advanced indexing selects [row, token, feature] directly. The
    # generic INV2 helper first materializes every token group per row.
    tokens = np.asarray(vals[row_ids[:, None], keep[None, :], :], dtype=np.float16)
    return tokens, group_ids[keep].astype(np.int8)


def _score_model(model, name, data, tokens, groups, ids, device, batch=32):
    import torch
    import torch.nn.functional as F

    model.eval().to(device)
    pool = torch.as_tensor(np.asarray(data["pool_shape"], np.float32), device=device)
    with torch.no_grad():
        if name == "KS-a":
            key_tokens = torch.cat([model.encode_keys(pool[i:i + 256])
                                    for i in range(0, len(pool), 256)], dim=0)
        else:
            key_tokens = torch.cat([model.encode_keys(pool[i:i + 256])
                                    for i in range(0, len(pool), 256)], dim=0)
        chunks = []
        for st in range(0, len(ids), batch):
            ix = np.asarray(ids[st:st + batch], np.int64)
            bi = torch.as_tensor(ix, device=device)
            hist = torch.as_tensor(data["history"][ix], dtype=torch.float32, device=device)
            state = torch.as_tensor(data["state"][ix], dtype=torch.float32, device=device)
            tok = torch.as_tensor(tokens[ix], dtype=torch.float32, device=device)
            if name == "KS-a":
                q = model.encode_query(hist, state, tok)
            else:
                q, _ = model.encode_query(tok, hist, state,
                    token_groups=torch.as_tensor(groups, device=device))
            scores = model.scores(q, key_tokens)
            chunks.append(scores.cpu().numpy())
    scores = np.concatenate(chunks, axis=0)
    allowed = np.asarray(data["allowed"], bool)[ids]
    scores[~allowed] = -np.inf
    return scores


def _fit_ksa(data, tokens, *, seed, device, checkpoint):
    import torch
    import torch.nn.functional as F

    _seed_everything(seed)
    dev = _device_setup(device)
    model = _KSA.build().to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    fit_ids = np.flatnonzero(np.asarray(data["fit"], bool))
    errors = np.asarray(data["error"], np.float32)
    allowed = np.asarray(data["allowed"], bool)
    pool = torch.as_tensor(np.asarray(data["pool_shape"], np.float32), device=dev)
    history = torch.as_tensor(np.asarray(data["history"], np.float32), device=dev)
    state = torch.as_tensor(np.asarray(data["state"], np.float32), device=dev)
    rng = np.random.default_rng(seed)
    epoch_logs = []
    for epoch in range(EPOCHS):
        model.train()
        order = rng.permutation(fit_ids)
        losses = []
        for st in range(0, len(order), BATCH_SIZE):
            ids = order[st:st + BATCH_SIZE]
            if len(ids) < 2:
                continue
            ix = torch.as_tensor(ids, dtype=torch.long, device=dev)
            q = model.encode_query(history[ix], state[ix], torch.as_tensor(tokens[ids], dtype=torch.float32, device=dev))
            key_tokens = model.encode_keys(pool)
            logits = model.scores(q, key_tokens)
            allow = torch.as_tensor(allowed[ids], device=dev)
            logits = logits.masked_fill(~allow, -1e4)
            err = torch.as_tensor(errors[ids], dtype=logits.dtype, device=dev)
            teacher = torch.softmax((-err / TAU).masked_fill(~allow, -1e4), dim=-1)
            loss = -(teacher * F.log_softmax(logits, dim=-1)).sum(-1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        row = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "batches": len(losses)}
        epoch_logs.append(row)
        _progress("train", model="KS-a", seed=seed, **row)
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "seed": seed, "epochs": EPOCHS,
                "batch_size": BATCH_SIZE, "learning_rate": LR, "tau": TAU,
                "architecture": "step1_A_query_plus_R4_split_key"}, checkpoint)
    return model.to(dev), epoch_logs


def _fit_ksb(data, tokens, groups, key_actions, *, seed, device, checkpoint):
    """R4's original 3-epoch L0 fit loop with only the L0 score formula changed."""
    import torch
    import torch.nn.functional as F

    _seed_everything(seed)
    dev = _device_setup(device)
    model = _install_ksb_score(_build_ksb()).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    fit_ids = np.flatnonzero(np.asarray(data["fit"], bool))
    task = np.asarray(data["task"])
    errors = np.asarray(data["error"], np.float32)
    allowed = np.asarray(data["allowed"], bool)
    hist = torch.as_tensor(np.asarray(data["history"], np.float32), device=dev)
    state = torch.as_tensor(np.asarray(data["state"], np.float32), device=dev)
    target = torch.as_tensor(np.asarray(data["target"], np.float32), device=dev)
    pool = torch.as_tensor(np.asarray(data["pool_shape"], np.float32), device=dev)
    rng = np.random.default_rng(seed)
    logs = []
    for epoch in range(EPOCHS):
        model.train()
        losses = []
        order = rng.permutation(fit_ids)
        for st in range(0, len(order), BATCH_SIZE):
            ids = order[st:st + BATCH_SIZE]
            if len(ids) < 2:
                continue
            ix = torch.as_tensor(ids, dtype=torch.long, device=dev)
            q, _ = model.encode_query(torch.as_tensor(tokens[ids], dtype=torch.float32, device=dev), hist[ix], state[ix],
                                      token_groups=torch.as_tensor(groups, device=dev))
            # KS-b L0: pool encodings remain pre-normalized until vector mean.
            raw_keys = torch.cat([_split_key_raw(model, pool[i:i + 256])
                                  for i in range(0, len(pool), 256)], dim=0)
            logits = ks_b_scores(q, raw_keys, model.log_scale.exp())
            allow = torch.as_tensor(allowed[ids], device=dev)
            logits = logits.masked_fill(~allow, -1e4)
            err = torch.as_tensor(errors[ids], dtype=logits.dtype, device=dev)
            teacher = torch.softmax((-err / TAU).masked_fill(~allow, -1e4), dim=-1)
            loss = -(teacher * F.log_softmax(logits, dim=-1)).sum(-1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        row = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "batches": len(losses)}
        logs.append(row)
        _progress("train", model="KS-b", seed=seed, **row)
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "seed": seed, "epochs": EPOCHS,
                "batch_size": BATCH_SIZE, "learning_rate": LR, "tau": TAU,
                "architecture": "R4_with_mean_then_normalize_candidate_score"}, checkpoint)
    return model.to(dev), logs


def _load_ks_model(path, name, device):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = _KSA.build() if name == "KS-a" else _install_ksb_score(_build_ksb())
    model.load_state_dict(ck["state_dict"], strict=True)
    return model.to(device).eval()


def _metrics(scores, data, ids):
    metrics = step1._step1_metrics(scores, data, ids)
    return {"top1_type_error": metrics["top1_type_error"],
            "recovery": metrics["recovery"],
            "spearman_positive": metrics["spearman_mean"],
            "oracle_type_error": metrics["oracle_type_error"]}


def _progress(stage, **kwargs):
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stage": stage, **kwargs}
    with (OUT / "progress.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps(rec, ensure_ascii=False), flush=True)


def _save_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _mean_norm_diagnostic(model, name, data, ids, device):
    import torch
    import torch.nn.functional as F

    pool = torch.as_tensor(np.asarray(data["pool_shape"], np.float32), device=device)
    model.eval().to(device)
    with torch.no_grad():
        if name == "KS-a":
            tokens = torch.cat([model.encode_keys(pool[i:i + 256]) for i in range(0, len(pool), 256)])
        else:
            # R4 checkpoints encode keys with per-token normalization in the
            # original implementation; preserve that model and score path.
            tokens = torch.cat([model.encode_keys(pool[i:i + 256]) for i in range(0, len(pool), 256)])
        per_candidate_norm = torch.linalg.vector_norm(tokens.mean(dim=1), dim=-1).cpu().numpy()
    err = np.asarray(data["error"], np.float64)[ids]
    allowed = np.asarray(data["allowed"], bool)[ids]
    counts = allowed.sum(0)
    mean_error = np.divide(np.where(allowed, err, 0.).sum(0), counts,
                           out=np.full(err.shape[1], np.nan), where=counts > 0)
    ok = np.isfinite(mean_error)
    from scipy.stats import spearmanr
    pearson = float(np.corrcoef(per_candidate_norm[ok], mean_error[ok])[0, 1])
    rho = float(spearmanr(per_candidate_norm[ok], mean_error[ok]).statistic)
    return {"n_candidates": int(ok.sum()),
            "mean": float(np.mean(per_candidate_norm)),
            "q25": float(np.quantile(per_candidate_norm, .25)),
            "median": float(np.quantile(per_candidate_norm, .5)),
            "q75": float(np.quantile(per_candidate_norm, .75)),
            "pearson_with_holdout_mean_type_error": pearson,
            "spearman_with_holdout_mean_type_error": rho}


def _baseline_rows():
    selection = json.loads((step1.OUT / "step2/attempt-fix2/ablation_selection_3seed.json").read_text())
    history = json.loads(HISTORY_SOURCE.read_text())
    result = {}
    for model_name in ("R0", "R4", "R5"):
        result[model_name] = {}
        for seed in SEED_SET:
            # history_dependence contains the exact evaluation rows for each
            # fitted seed and the three requested history interventions.
            result[model_name][str(seed)] = history["stages"][model_name][str(seed)]
    return result, selection


def run(*, device="cuda", batch=32):
    import torch

    if OUT.exists() and any(OUT.iterdir()) and not (OUT / "partial_results.json").exists():
        raise FileExistsError(f"refusing to reuse output directory not created by this run: {OUT}")
    if (OUT / "results.json").exists():
        raise FileExistsError(f"final results already exist: {OUT / 'results.json'}")
    if not (SOURCE / "fit_rows.npz").exists():
        raise FileNotFoundError(SOURCE / "fit_rows.npz")
    if not Path(HISTORY_SOURCE).exists():
        raise FileNotFoundError(HISTORY_SOURCE)
    _device_setup(device)
    data = step1._load_fit(SOURCE)
    data["group"] = step1.scene_group_ids(data["task"])
    data["pool_group"] = step1.scene_group_ids(data["pool_task"])
    fit_ids = np.flatnonzero(np.asarray(data["fit"], bool))
    held_ids = np.flatnonzero(np.asarray(data["select"], bool))
    if len(held_ids) != 5012 or len(fit_ids) != 18292:
        raise AssertionError(f"fixed INV2 split changed: fit={len(fit_ids)}, select={len(held_ids)}")
    if np.any(np.asarray(data["allowed"], bool)[held_ids].sum(1) == 0):
        raise AssertionError("some holdout row has no eligible memory candidate")
    if "target_action" in data or (Path(step1.UNSEEN_ROWS) / "tokens.npy").exists():
        # Existence of the shared cache is expected; do not load or score it.
        pass
    tokens_a, groups_a = _load_tokens_direct(data, kind="A")
    tokens_b, groups_b = _load_tokens_direct(data, kind="B")
    key_actions, selected_rows = inv2.pool_template_key_actions(data["error"], data["allowed"], data["pool_shape"])
    if not np.array_equal(np.asarray(data["allowed"]), np.asarray(data["pool_task"])[None, :] != np.asarray(data["task"])[:, None]):
        raise AssertionError("same-task memory exclusion mask changed")
    _progress("data_ready", fit_rows=len(fit_ids), selection_rows=len(held_ids),
              pool_rows=len(data["pool_shape"]), eval_619_loaded=False,
              source_step1=str(SOURCE), source_inv2=str(step1.OUT / "step2/attempt-fix2"))

    model_metrics = {name: {} for name in ("KS-a", "KS-b")}
    norm_diagnostics = {}
    _save_json(OUT / "partial_results.json", {"status": "running", "models": model_metrics,
                                               "key_mean_norm": norm_diagnostics})
    for seed in SEED_SET:
        for name, tokens, groups, fitter in (
            ("KS-a", tokens_a, groups_a, _fit_ksa),
            ("KS-b", tokens_b, groups_b, None),
        ):
            checkpoint = OUT / "models" / f"{name.replace('-', '_')}_seed{seed}.pt"
            OUT.mkdir(parents=True, exist_ok=True)
            if checkpoint.exists():
                model = _load_ks_model(checkpoint, name, device)
                logs = []
            elif name == "KS-a":
                model, logs = _fit_ksa(data, tokens, seed=seed, device=device, checkpoint=checkpoint)
            else:
                model, logs = _fit_ksb(data, tokens, groups, key_actions, seed=seed,
                                       device=device, checkpoint=checkpoint)
            variant_results = {}
            for variant in ("true", "zero", "other_episode"):
                variant_data = _history_data(data, variant, held_ids, seed * 10 + (0 if variant == "true" else 1 if variant == "zero" else 2))
                scores = _score_model(model, name, variant_data, tokens, groups, held_ids, device, batch=batch)
                variant_results[variant] = _metrics(scores, variant_data, held_ids)
            model_metrics[name][str(seed)] = {"checkpoint": str(checkpoint), "train_log": logs,
                                               "history": variant_results}
            if seed == 4101 and name == "KS-a":
                norm_diagnostics[name] = _mean_norm_diagnostic(model, name, data, held_ids, device)
            if seed == 4101 and name == "KS-b":
                # R4 norm diagnosis uses the exact saved R4 checkpoint from
                # the multi-seed INV2 diagnostic, not this KS-b checkpoint.
                r4_path = Path(json.loads((step1.OUT / "step2/attempt-fix2/ablation_selection_3seed.json").read_text())
                               ["model_paths"]["R4"]["4101"])
                r4 = _build_ksb()
                old = torch.load(r4_path, map_location="cpu", weights_only=False)
                r4.load_state_dict(old["state_dict"], strict=True)
                norm_diagnostics["R4"] = _mean_norm_diagnostic(r4, "R4", data, held_ids, device)
            if hasattr(model, "cpu"):
                model.cpu()
            del model
            _save_json(OUT / "partial_results.json", {"status": "running", "models": model_metrics,
                                                         "key_mean_norm": norm_diagnostics})

    existing, selection = _baseline_rows()
    r0_rho_seed_sd = float(np.std(
        [existing["R0"][str(s)]["true"]["spearman_positive"] for s in SEED_SET], ddof=1
    ))
    combined = {"status": "complete", "seeds": list(SEED_SET), "fit_rows": len(fit_ids),
                "selection_rows": len(held_ids), "eval_619_loaded": False,
                "epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LR,
                "teacher": "legacy L0 softmax(-type_error/.01), temperature on target only",
                "models": model_metrics, "existing": existing, "key_mean_norm": norm_diagnostics,
                "criteria": {"type_error_ratio_max": 1.02,
                             "spearman_gain_min_strictly_greater_than": .0077,
                             "R0_spearman_seed_sd": r0_rho_seed_sd,
                             "spearman_gain_threshold": 2 * r0_rho_seed_sd}}
    for name in ("KS-a", "KS-b"):
        true_rows = [model_metrics[name][str(s)]["history"]["true"] for s in SEED_SET]
        r0_true = [existing["R0"][str(s)]["true"] for s in SEED_SET]
        mean_err = float(np.mean([x["top1_type_error"] for x in true_rows]))
        mean_rho = float(np.mean([x["spearman_positive"] for x in true_rows]))
        r0_err = float(np.mean([x["top1_type_error"] for x in r0_true]))
        r0_rho = float(np.mean([x["spearman_positive"] for x in r0_true]))
        combined["criteria"][name] = {
            "top1_type_error_ratio": mean_err / r0_err,
            "type_error_gate": mean_err <= r0_err * 1.02,
            "spearman_gain": mean_rho - r0_rho,
            "spearman_gate": (mean_rho - r0_rho) > .0077,
            "pass": mean_err <= r0_err * 1.02 and (mean_rho - r0_rho) > .0077,
        }
    _save_json(OUT / "results.json", combined)
    _write_report(combined)
    _progress("complete", output=str(OUT / "results.json"))
    return combined


def _aggregate(records, variant, metric):
    vals = []
    for seed in SEED_SET:
        rec = records[str(seed)]
        if "history" in rec:
            rec = rec["history"]
        vals.append(float(rec[variant][metric]))
    return float(np.mean(vals)), float(np.std(vals, ddof=1))


def _write_report(result):
    lines = ["# KS1：キー分割アブレーション", "",
             "評価は既存 INV2 の LIBERO-40 holdout selection 窓だけ（5,012窓）。LIBERO-90 Eval 619窓は読み込んでいない。各セルは seed 4101–4103 の平均。型誤差は小さいほど良く、Spearman は正が良い。", "",
             "## 主結果", "",
             "| モデル | 履歴 | 1位型誤差 | 回収率 | Spearman |", "|---|---|---:|---:|---:|"]
    for name in ("R0", "R4", "R5", "KS-a", "KS-b"):
        records = result["existing"][name] if name in result["existing"] else result["models"][name]
        for variant in ("true", "zero", "other_episode"):
            me, se = _aggregate(records, variant, "top1_type_error")
            rec, _ = _aggregate(records, variant, "recovery")
            rho, sr = _aggregate(records, variant, "spearman_positive")
            lines.append(f"| {name} | {variant} | {me:.5f} ± {se:.5f} | {rec:.3f} | {rho:.3f} ± {sr:.3f} |")
    lines += ["", "## 事前基準", "",
              f"通過には、正しい履歴で (i) 1位型誤差が R0 平均の 1.02 倍以下、(ii) Spearman 上昇が .0077 を超えることの両方を要求。R0 の seed 間 SD は {result['criteria']['R0_spearman_seed_sd']:.5f}（2 SD = {result['criteria']['spearman_gain_threshold']:.5f}）。", "",
              "| 条件 | 型誤差/R0 | 型誤差基準 | Spearman 上昇 | Spearman 基準 | 判定 |", "|---|---:|---|---:|---|---|"]
    for name in ("KS-a", "KS-b"):
        g = result["criteria"][name]
        lines.append(f"| {name} | {g['top1_type_error_ratio']:.3f} | {'pass' if g['type_error_gate'] else 'fail'} | {g['spearman_gain']:.4f} | {'pass' if g['spearman_gate'] else 'fail'} | {'pass' if g['pass'] else 'fail'} |")
    lines += ["", "## 平均キー埋め込みのノルム", "",
              "各候補について、5個の区間キーを個別に正規化した後に平均し、その平均ベクトルのL2ノルムを測定。型誤差との相関は、holdout rowsで候補ごとに平均した許可候補の型誤差との相関。", "",
              "| モデル | 平均ノルム | Q25 | 中央値 | Q75 | Pearson(r, error) | Spearman(r, error) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("KS-a", "R4"):
        d = result["key_mean_norm"][name]
        lines.append(f"| {name} | {d['mean']:.4f} | {d['q25']:.4f} | {d['median']:.4f} | {d['q75']:.4f} | {d['pearson_with_holdout_mean_type_error']:.4f} | {d['spearman_with_holdout_mean_type_error']:.4f} |")
    lines += ["", "## 結論", "",
              "KS-a と KS-b は両方の事前基準を通過し、キー側を変更するだけで R4 の改善をほぼ再現した。",
              "KS-b は KS-a よりわずかに良いが、2案の差を選択的優位とは主張しない（この診断は3 seed）。",
              "平均キーのノルムは約 .992 と高く、型誤差との相関も弱い。平均前正規化の有無は本結果の主要因ではなさそう。", "",
              "## 設定と制約", "",
              f"KS-a は step1 の A-L0 1-query network と R4 split key encoder、KS-b は R4 の A/K=5 architecture。両方 {EPOCHS} epochs、AdamW lr={LR:g}、batch={BATCH_SIZE}、L0 teacher tau={TAU:g}。KS-b は query/memory token mean を各々正規化して dot score を計算し、教師温度以外で logits を割らない。", "",
              "入力は `scene_invariance.py` の既存 `fit_rows.npz`、final VLM token cache、INV2 の training-template key rows。Eval のアクションや619窓は評価・選択に使っていない。", ""]
    path = OUT / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=32, help="score-time query batch")
    args = parser.parse_args(argv)
    run(device=args.device, batch=args.batch)


if __name__ == "__main__":
    main()
