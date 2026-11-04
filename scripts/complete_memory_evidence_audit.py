"""Bounded CPU-only completion of archived memory-FP evidence.

Restores local models with PyTorch's restricted weights-only loader.  No model
training, GPU, simulator, or modification of historical outputs is performed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import analyze_memory_slide_evidence as audit
from skill_moe.skillnet.examples.emergent.light.methods.m1 import M1


def safe_checkpoint(path):
    allowed = [np.core.multiarray._reconstruct, np.ndarray, np.dtype]
    allowed += [type(np.dtype(t)) for t in ("float16", "float32", "float64", "int32", "int64", "bool")]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)


def load_rows(cache):
    data = {k: [] for k in ("episode", "task", "frame", "f", "s", "a")}
    manifest = json.loads((cache / "manifest.json").read_text())
    for ep in manifest["episodes"]:
        with np.load(ep["cache"], allow_pickle=False) as z:
            task = int(z["task_index"])
            if task not in range(10, 40):
                continue
            steps, actions, states = z["step"].astype(int), z["action"], z["env_state"]
            frames = z["env_step"] if "env_step" in z else np.arange(len(states))
            lookup = {int(s): i for i, s in enumerate(frames)}
            features, proprio = z["f"], z["proprio"]
            for i, step in enumerate(steps):
                start = lookup.get(int(step), -1)
                if start >= 0 and start + 25 < len(states) and start + 25 <= len(actions):
                    for key, value in (("episode", str(ep["episode_id"])), ("task", task),
                                       ("frame", int(step)), ("f", features[i]), ("s", proprio[i]),
                                       ("a", actions[start:start+25])):
                        data[key].append(value)
    return {k: np.asarray(v, dtype=np.float32 if k in ("f", "s", "a") else None)
            for k, v in data.items()}


def get_partitions(rows, split, seed, per_task):
    train = audit.subset(rows, np.isin(rows["task"], audit.TRAIN[split]))
    held = audit.heldout_metadata(rows, split, seed, per_task)
    train = audit.subset(train, ~np.isin(train["episode"], held["episode"]))
    episodes = np.unique(train["episode"])
    rng = np.random.default_rng(seed + 401)
    nvalid = min(len(episodes)-1, max(1, int(round(len(episodes)*.15))))
    valid_ep = rng.permutation(episodes)[:nvalid]
    fit = audit.subset(train, ~np.isin(train["episode"], valid_ep))
    valid = audit.subset(train, np.isin(train["episode"], valid_ep))
    test = audit.subset(rows, np.isin(rows["task"], audit.EVAL[split]))
    if set(fit["episode"]) & (set(valid["episode"]) | set(held["episode"]) | set(test["episode"])):
        raise ValueError("historical fit/evaluation episode overlap")
    return fit, valid, held, test


def select_episode_windows(rows, per_task, seed=8301):
    """One random valid window per episode; choose without actions or losses."""
    rng, indices = np.random.default_rng(seed), []
    for task in np.unique(rows["task"]):
        episodes = np.unique(rows["episode"][rows["task"] == task])
        chosen = rng.choice(episodes, min(per_task, len(episodes)), replace=False)
        for episode in sorted(chosen):
            eligible = np.flatnonzero(rows["episode"] == episode)
            indices.append(int(rng.choice(eligible)))
    return np.asarray(indices, dtype=int)


def restore_m1(state):
    m = M1(device="cpu")
    for k, v in state["M1_stats"].items():
        setattr(m, k, v)
    width = int(state["m1"]["0.weight"].shape[0])
    def mlp(din, dout):
        return nn.Sequential(nn.Linear(din, width), nn.GELU(), nn.Linear(width, width),
                             nn.GELU(), nn.Linear(width, dout)).eval()
    for name, key in (("encoder", "m1"), ("prior_net", "m1_prior"),
                      ("base_decoder", "m1_base_decoder"), ("residual_decoder", "m1_residual_decoder")):
        weights = state[key]
        module = mlp(int(weights["0.weight"].shape[1]), int(weights["4.weight"].shape[0]))
        module.load_state_dict(weights)
        setattr(m, name, module)
    return m


def functional_signatures(m1, bases, probe, block=64):
    state = (probe["s"] - m1.sm) / m1.ss
    _, residual = m1.prior(probe["f"], probe["s"])
    outputs = []
    with torch.no_grad():
        r = m1.residual_decoder(torch.from_numpy(residual.astype(np.float32)))
        for lo in range(0, len(bases), block):
            b = np.asarray(bases[lo:lo+block], dtype=np.float32)
            bb = np.repeat(b, len(state), axis=0)
            ss = np.tile(state, (len(b), 1)).astype(np.float32)
            y = m1.base_decoder(torch.from_numpy(np.concatenate([bb, ss], axis=1)))
            y = y.reshape(len(b), len(state), -1) + r[None]
            y = y.numpy() * m1.ass + m1.am
            outputs.append(y.reshape(len(b), -1) / np.sqrt(y.shape[1]*y.shape[2]))
    return np.concatenate(outputs).astype(np.float32)


def normalized(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def restore_bank(m1, fit, train_keys, centers, epsilon, seed):
    """Replay exact train-only unit assignment; never admit post-stream centers."""
    if len(train_keys) != len(fit["a"]):
        raise ValueError("training keys do not align with reconstructed fit rows")
    bases, _ = m1.encode(fit["a"], fit["s"])
    rng = np.random.default_rng(seed + 19 + 6046)
    probe_ix = rng.choice(len(bases), size=64, replace=len(bases)<64)
    probe = {k: fit[k][probe_ix] for k in ("f", "s")}
    center_sig = functional_signatures(m1, centers, probe)
    assignments = np.full(len(bases), -1, int)
    for lo in range(0, len(bases), 64):
        b = bases[lo:lo+64]
        nearest = np.argsort(np.linalg.norm(b[:, None]-centers[None], axis=2), axis=1,
                             kind="stable")[:, :min(10, len(centers))]
        sig = functional_signatures(m1, b, probe)
        diff = sig[:, None].astype(np.float64)-center_sig[nearest].astype(np.float64)
        distances = np.sum(diff*diff, axis=2)
        # Original M6 resolves functional ties by unit id, not by latent rank.
        for j in range(len(b)):
            ix = min(range(nearest.shape[1]), key=lambda k: (distances[j, k], int(nearest[j, k])))
            if distances[j, ix] <= epsilon:
                assignments[lo+j] = int(nearest[j, ix])
    ids = np.unique(assignments[assignments >= 0])
    keys = np.stack([train_keys[assignments == i].astype(np.float32).mean(0) for i in ids])
    return ids, keys, probe, center_sig[ids], assignments


def retrieval_metrics(scores, gold, functional_distances, epsilon):
    order = np.argsort(-scores, axis=1, kind="stable")
    rank = np.argmax(order == gold[:, None], axis=1)+1
    valid = functional_distances <= epsilon
    covered = valid.any(axis=1)
    out = {"queries": len(scores), "units": scores.shape[1], "median_gold_rank": float(np.median(rank)),
           "functional_coverage": float(covered.mean()), "hits": {}, "functional_recall_on_covered": {}}
    for k in (1, 5, 10, 20):
        kk = min(k, scores.shape[1])
        out["hits"][str(k)] = float(np.mean(rank <= kk))
        found = np.take_along_axis(valid, order[:, :kk], axis=1).any(axis=1)
        out["functional_recall_on_covered"][str(k)] = float(found[covered].mean()) if covered.any() else None
    return out, rank, order


def flow_module(weights):
    layers = nn.Sequential(nn.Linear(weights["net.0.weight"].shape[1], weights["net.0.weight"].shape[0]),
                           nn.GELU(), nn.Linear(weights["net.2.weight"].shape[1], weights["net.2.weight"].shape[0]),
                           nn.GELU(), nn.Linear(weights["net.4.weight"].shape[1], weights["net.4.weight"].shape[0])).eval()
    layers.load_state_dict({k.removeprefix("net."): v for k, v in weights.items()})
    return layers


def flow_predict(net, norm, rows, eps, anchor, t0, n, block=64):
    samples, count, dim = eps.shape
    result = np.empty((samples, count, dim), np.float32)
    ff = (rows["f"]-norm["f_mean"])/norm["f_scale"]
    ss = (rows["s"]-norm["s_mean"])/norm["s_scale"]
    src = None if anchor is None else (anchor.reshape(count, dim)-norm["a_mean"])/norm["a_scale"]
    t0 = np.broadcast_to(np.asarray(t0, np.float32), count)
    n = np.broadcast_to(np.asarray(n, int), count)
    with torch.no_grad():
        for lo in range(0, count, block):
            hi = min(lo+block, count); width = hi-lo
            tt = torch.from_numpy(np.tile(t0[lo:hi], samples))
            nn_ = np.tile(n[lo:hi], samples)
            noise = torch.from_numpy(eps[:, lo:hi].reshape(-1, dim))
            x = noise.clone() if src is None else tt[:, None]*noise + (1-tt[:, None])*torch.from_numpy(np.tile(src[lo:hi], (samples, 1)))
            f = torch.from_numpy(np.tile(ff[lo:hi], (samples, 1)))
            s = torch.from_numpy(np.tile(ss[lo:hi], (samples, 1)))
            dt = -tt / torch.from_numpy(nn_.astype(np.float32))
            for j in range(int(nn_.max())):
                active = np.flatnonzero(j < nn_)
                time_value = tt[active]*torch.from_numpy((1-j/nn_[active]).astype(np.float32))
                v = net(torch.cat([x[active], time_value[:, None], f[active], s[active]], dim=1))
                x[active] += dt[active, None]*v
            result[:, lo:hi] = x.numpy().reshape(samples, width, dim)
    result = result*norm["a_scale"]+norm["a_mean"]
    return result.reshape(samples, count, 25, 7)


def sample_errors(pred, target):
    return ((pred-target[None])**2).mean(axis=(2, 3))


def medoid_indices(keys, count, seed):
    x = normalized(keys)
    km = MiniBatchKMeans(n_clusters=count, n_init=1, batch_size=512, max_iter=30,
                        max_no_improvement=5, random_state=seed).fit(x)
    selected = []
    for center in km.cluster_centers_:
        order = np.argsort(np.sum((x-center)**2, axis=1), kind="stable")
        selected.append(next(int(i) for i in order if int(i) not in selected))
    return np.asarray(selected)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fair", type=Path, required=True)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episodes-per-task", type=int, default=20)
    args = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("set CUDA_VISIBLE_DEVICES='' for CPU-only execution")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    history = json.loads((args.fair/"results.json").read_text())
    review = json.loads((args.review/"results.json").read_text())
    old_errors = np.load(args.fair/"per_row.npz", allow_pickle=False)
    rows = load_rows(args.cache)
    results = {"protocol": {"cuda": False, "torch_threads": 1, "priority": 19,
                             "window_selection_seed": 8301, "episodes_per_task": args.episodes_per_task,
                             "samples": 8, "model_training": False,
                             "retrieval_note": "reconstructed with archived float16 query/train keys; replay audited"},
               "source_hashes": {}, "runs": [], "fixed_validation": [], "grid": [], "dimensions": []}
    saved = {}
    print("loaded", len(rows["a"]), "full action windows", flush=True)
    for source in [args.fair/"results.json", args.fair/"per_row.npz", args.review/"results.json"]:
        results["source_hashes"][str(source)] = audit.sha256(source)
    for split in ("A", "B"):
        for seed in (0, 1, 2):
            stem = f"{split}_s{seed}"
            hrun = next(r for r in history["runs"] if r["split"] == split and r["seed"] == seed)
            rrun = next(r for r in review["runs"] if r["split"] == split and r["seed"] == seed)
            fit, valid, held, test = get_partitions(rows, split, seed, history["holdout_episodes_per_task"])
            for name, data, ref in (("train_fit", fit, "train_fit"), ("validation", valid, "validation"),
                                    ("D1", held, "D1"), ("new", test, "new")):
                if len(data["a"]) != hrun["rows"][ref]:
                    raise ValueError(f"{stem} {name} row count mismatch")
            audit.check_alignment(test, old_errors[f"{stem}|new_task"])
            path = args.review/f"{stem}_models.pt"
            results["source_hashes"][str(path)] = audit.sha256(path)
            state = safe_checkpoint(path)
            m1 = restore_m1(state)
            count = int(hrun["M6_units_train"])
            centers = state["M6_centers"][:count].astype(np.float32)
            train_keys = state["train_keys"].astype(np.float32)
            print(stem, "restoring train-only bank", count, flush=True)
            ids, keys, probe, center_sig, assignments = restore_bank(
                m1, fit, train_keys, centers, float(state["M6_epsilon"]), seed)
            emb_path = args.review/f"{stem}_embeddings.npz"
            results["source_hashes"][str(emb_path)] = audit.sha256(emb_path)
            with np.load(emb_path, allow_pickle=False) as emb:
                queries = emb["D3_query"].astype(np.float32)
                recorded_b = emb["D3_unit_base"].astype(np.float32)
            expected_pos = np.argmin(np.sum((recorded_b[:, None]-centers[ids][None])**2, axis=2), axis=1)
            if np.max(np.abs(centers[ids][expected_pos]-recorded_b)) > .02:
                raise ValueError("recorded unit bases cannot be identified in train-only centers")
            scores = normalized(queries) @ normalized(keys).T
            replay = float(np.mean(np.argmax(scores, axis=1) == expected_pos))
            if replay < .99:
                raise ValueError(f"{stem} reconstructed retrieval replay agreement {replay:.4f} below .99")
            gold_b, _ = m1.encode(test["a"], test["s"])
            gold_sig = functional_signatures(m1, gold_b, probe)
            # Match the original full functional-nearest oracle, not latent L2.
            distances = np.maximum(np.sum(gold_sig**2, axis=1)[:, None]+
                                   np.sum(center_sig**2, axis=1)[None]-2*(gold_sig@center_sig.T), 0)
            gold = distances.argmin(axis=1)
            retrieval, ranks, order = retrieval_metrics(scores, gold, distances, float(state["M6_epsilon"]))
            retrieval["top1_replay_agreement"] = replay
            retrieval["historical_top1"] = hrun["sources"]["instruction"]["conditions"]["D3"]["populations"]["all"]["_top1_is_gold_unit"]
            retrieval["reconstructed_minus_historical_top1"] = retrieval["hits"]["1"]-retrieval["historical_top1"]
            if abs(retrieval["reconstructed_minus_historical_top1"]) > .01:
                raise ValueError("functional gold labels do not reproduce historical strict hit rate")
            selected = select_episode_windows(test, args.episodes_per_task)
            sample = audit.subset(test, np.isin(np.arange(len(test["a"])), selected))
            # np.isin restores source order, so use source-sorted selected ids throughout.
            selected = np.sort(selected)
            rng = np.random.default_rng(seed*1_000_003+31)
            rng.standard_normal((8, len(valid["a"]), 175))
            rng.standard_normal((8, len(held["a"]), 175))
            eps = rng.standard_normal((8, len(test["a"]), 175)).astype(np.float32)[:, selected]
            normal_net, memory_net = flow_module(state["F_normal"]), flow_module(state["F_anchor"])
            normal = flow_predict(normal_net, state["F_normal_norm"], sample, eps, None, 1., 10)
            normal_err = sample_errors(normal, sample["a"])
            source_root = f"{stem}|instruction_D3"
            t0 = old_errors[f"{source_root}|A:unit:cal_sim|t0"][selected]
            n = np.maximum(1, np.rint(10*t0)).astype(int)
            anchors = m1.decode(centers[ids][expected_pos[selected]], np.zeros((len(selected), 4), np.float32), sample["f"], sample["s"])
            memory = flow_predict(memory_net, state["F_anchor_norm"], sample, eps, anchors, t0, n)
            mem_err = sample_errors(memory, sample["a"])
            delta_normal = float(np.max(np.abs(normal_err.mean(0)-old_errors[f"{source_root}|N:noise:n10|mse"][selected])))
            delta_memory = float(np.max(np.abs(mem_err.mean(0)-old_errors[f"{source_root}|A:unit:cal_sim|mse"][selected])))
            if max(delta_normal, delta_memory) > 1e-4:
                raise ValueError(f"{stem} restored FP replay mismatch: normal {delta_normal}, memory {delta_memory}")
            print(stem, "FP replay matched; independent-noise/count controls", flush=True)
            # Match each memory path's n with a fresh-noise normal-policy path.
            second = flow_predict(normal_net, state["F_normal_norm"], sample, np.roll(eps, 1, axis=0), None, 1., n)
            second_err = sample_errors(second, sample["a"])
            # Equal-entry-count raw controls, same generator and n/t0 as unit path.
            unit_fixed = flow_predict(normal_net, state["F_normal_norm"], sample, eps, anchors, .4, 4)
            raw_indices = medoid_indices(train_keys, len(ids), seed+6101)
            raw_pos = (normalized(queries[selected]) @ normalized(train_keys[raw_indices]).T).argmax(axis=1)
            raw_anchor = fit["a"][raw_indices[raw_pos]]
            raw_fixed = flow_predict(normal_net, state["F_normal_norm"], sample, eps, raw_anchor, .4, 4)
            err = {"normal10": normal_err, "memory_adaptive": mem_err,
                   "independent_normal_matched_n": second_err,
                   "oracle_normal_memory": np.minimum(normal_err, mem_err),
                   "oracle_two_normal_matched_n": np.minimum(normal_err, second_err),
                   "unit_fixed4": sample_errors(unit_fixed, sample["a"]),
                   "raw_equal_entries_fixed4": sample_errors(raw_fixed, sample["a"])}
            for k,v in err.items():
                saved[f"{stem}|{k}|per_noise_mse"] = v
            for k in ("task", "episode", "frame"):
                saved[f"{stem}|sample_{k}"] = sample[k]
            saved[f"{stem}|gold_unit_rank"] = ranks
            saved[f"{stem}|top20_unit_ids"] = ids[order[:, :min(20, len(ids))]]
            saved[f"{stem}|unit_ids"] = ids
            saved[f"{stem}|query"] = queries.astype(np.float16)
            saved[f"{stem}|unit_keys"] = keys.astype(np.float16)
            run = {"split": split, "seed": seed, "retrieval": retrieval,
                   "sampled_windows": len(selected), "sampled_episodes": len(np.unique(sample["episode"])),
                   "fp_replay_max_abs_error_difference": {"normal": delta_normal, "memory": delta_memory},
                   "candidate_nfe_per_row": float(10+n.mean()), "raw_entries": len(raw_indices), "unit_entries": len(ids),
                   "mean_errors": {k: float(v.mean()) for k,v in err.items()},
                   "raw_selection": "MiniBatchKMeans key medoids; train-only; n_init1,max_iter30,seed+6101",
                   "payload_floats_per_item": {"unit": int(96+12), "raw": int(96+175)}}
            results["runs"].append(run)
            for fkey in ("A", "N"):
                src = hrun["sources"]["instruction"]
                val_err = src["calibration_detail"][fkey]["unit"]["fixed_t0_validation_mse"]
                best = min((.2,.4,.6), key=lambda t: val_err[str(t)])
                pop = src["conditions"]["D3"]["populations"]["all"]
                results["fixed_validation"].append({"split": split,"seed": seed,"generator": fkey,
                    "selected_t0": best,"fixed_mse":pop[f"{fkey}:unit:fix{best}"]["mse"],
                    "adaptive_mse":pop[f"{fkey}:unit:cal_sim"]["mse"]})
            for t in (.2,.4,.6):
                results["grid"].append({"split": split,"seed": seed,"t0":t,
                    "mse_by_n": {str(n): rrun["R6"]["grid"]["D3"][f"t0={t:g},n={n}"]["mse"] for n in (1,2,3,5,10)}})
            pop = hrun["sources"]["instruction"]["conditions"]["D3"]["populations"]["all"]
            results["dimensions"].append({"split":split,"seed":seed,
                "normal":{k:pop[audit.NORMAL][k] for k in ("mse","best8","mse_pos","mse_rot","mse_grip","grip_agree","pos_term_mm","spread")},
                "memory":{k:pop[audit.MEMORY][k] for k in ("mse","best8","mse_pos","mse_rot","mse_grip","grip_agree","pos_term_mm","spread")}})
            print(stem,"complete",run["mean_errors"],flush=True)
            del state, m1, normal_net, memory_net, gold_sig
    results["elapsed_seconds"] = time.monotonic()-start
    results["source_unchanged"] = all(audit.sha256(Path(p))==h for p,h in results["source_hashes"].items())
    np.savez_compressed(args.output/"diagnostic_arrays.npz", **saved)
    (args.output/"summary.json").write_text(json.dumps(results,indent=2,ensure_ascii=False,allow_nan=False))
    print("complete audit",args.output,"seconds",round(results["elapsed_seconds"],1),flush=True)


if __name__ == "__main__":
    main()
