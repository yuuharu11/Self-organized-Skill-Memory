"""Offline v2 combinations using the frozen review-tests checkpoints.

The R4 multi-positive dual encoder and canonical M1/F checkpoints are loaded
from review-tests output. S3 gripper-segmented skill groups are reconstructed
from those same frozen representations; no network weights are updated.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from ....research_paths import REPO_ROOT as RESEARCH_REPO_ROOT, STATE_DIR as RESEARCH_STATE_DIR


DEFAULT_REVIEW = Path(str(RESEARCH_STATE_DIR / 'outputs/memory-pilot/review-tests-20261028T180100Z'))
DEFAULT_S3 = Path(str(RESEARCH_STATE_DIR / 'outputs/memory-pilot/hier-skills-20261028T173818Z-071506'))


def _mlp(state: dict[str, torch.Tensor]) -> nn.Sequential:
    linear_ids = sorted({int(k.split(".")[0]) for k in state if k.endswith(".weight")})
    layers = []
    for pos, idx in enumerate(linear_ids):
        w = state[f"{idx}.weight"]
        layer = nn.Linear(w.shape[1], w.shape[0])
        layer.load_state_dict({"weight": state[f"{idx}.weight"], "bias": state[f"{idx}.bias"]})
        layers.append(layer)
        if pos != len(linear_ids) - 1:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


def _restore_m1(checkpoint: dict, *, device):
    from ..methods.m1 import M1
    from ..methods._common import AdditiveActionDecoder
    stats = checkpoint["M1_stats"]
    model = M1(independence_weight=.01, beta_r=1., width=128, device=str(device),
               base_dim=12, residual_dim=4, decoder_obs=False)
    for name in ("am", "ass", "sm", "ss", "fm", "fs"):
        setattr(model, name, np.asarray(stats[name], np.float32))
    model.action_shape = (25, 7); model.a_dim = 175; model.state_dim = 8; model.feature_dim = 2048
    model.encoder = _mlp(checkpoint["m1"]).to(device)
    model.prior_net = _mlp(checkpoint["m1_prior"]).to(device)
    model.base_decoder = _mlp(checkpoint["m1_base_decoder"]).to(device)
    model.residual_decoder = _mlp(checkpoint["m1_residual_decoder"]).to(device)
    model.decoder = AdditiveActionDecoder(model.base_decoder, model.residual_decoder,
        base_dim=12, residual_dim=4, state_dim=8, feature_dim=2048,
        decoder_obs=False, residual_uses_context=False).to(device)
    model.encoder.eval(); model.prior_net.eval(); model.base_decoder.eval(); model.residual_decoder.eval()
    return model


def _restore_encoder(state, *, device):
    from .clip_retrieval import DualEncoder
    image_in = state["image_tower.0.weight"].shape[1]
    state_in = state["state_tower.0.weight"].shape[1]
    base_in = state["skill.0.weight"].shape[1]
    # The saved R4 encoder used sentence embeddings, 96-D joint keys.
    model = DualEncoder(image_in - 4096, 4096, state_in, base_in, 1,
                        subtask_mode="sentence", subtask_input_dim=2048, embed_dim=96).to(device)
    model.load_state_dict(state); model.eval()
    return model


def _load_r4_query_context(rows, *, model, train_rows, all_rows, seed, device, cache_dir):
    from . import anchor_flow as af
    from . import fair_compare as fc
    from .experiment import _episode_train_validation_indices, _standardize, _standardize_fit
    # The query-side subtask text can come from held-out suites, as in fair_compare;
    # only the normalization statistics below are fitted on train_rows.
    texts, emb = fc.precompute_texts(all_rows, ("instruction",), cache_dir)
    tf = fc.attach_text(train_rows, texts, emb, "instruction")
    qtrain = af._normal_query_rows(tf)
    fit_ix, _ = _episode_train_validation_indices(qtrain, seed + 19)
    norm = {}
    for dim in ("f", "k", "s"):
        norm[f"{dim}_mean"], norm[f"{dim}_scale"] = _standardize_fit(qtrain[dim][fit_ix])
    fit = {"normalizers": norm, "subtask_mode": "sentence", "class_lookup": {}}
    train_q = af._query_embedding(model, fit, qtrain, device=device)
    query = {}
    for name, view in rows.items():
        tv = fc.attach_text(view, texts, emb, "instruction")
        query[name] = af._query_embedding(model, fit, af._normal_query_rows(tv), device=device)
    return train_q, query


def _fit_r4_assets(rowsets, *, train_rows, all_rows, bases, centers, m1, seed,
                   m6_epsilon, device, cache_dir):
    """Recreate only the missing R4 multi-positive E checkpoint, keeping M1 frozen."""
    from . import anchor_flow as af
    from . import fair_compare as fc
    from . import clip_retrieval as cr
    from .experiment import _episode_train_validation_indices, _standardize_fit
    from unittest.mock import patch
    texts, emb = fc.precompute_texts(all_rows, ("instruction",), cache_dir)
    tf = fc.attach_text(train_rows, texts, emb, "instruction")
    qrows = af._normal_query_rows(tf)
    probe_rng = np.random.default_rng(seed + 334)
    probe_ix = np.sort(probe_rng.choice(len(train_rows["a"]), min(32, len(train_rows["a"])), replace=False))
    probe = {k:(v[probe_ix] if isinstance(v,np.ndarray) and len(v)==len(train_rows["a"]) else v)
             for k,v in train_rows.items()}
    prior_r = m1.prior(probe["f"],probe["s"])[1]
    sig = np.empty((len(bases),len(probe_ix)*175),np.float16)
    for lo in range(0,len(bases),32):
        hi=min(len(bases),lo+32); count=hi-lo
        bb=np.repeat(bases[lo:hi],len(probe_ix),axis=0)
        rr=np.tile(prior_r,(count,1)); ff=np.tile(probe["f"],(count,1)); ss=np.tile(probe["s"],(count,1))
        out=m1.decode(bb,rr,ff,ss).reshape(count,-1)
        sig[lo:hi]=(out/np.sqrt(len(probe_ix)*175)).astype(np.float16)
    sig_tensor=torch.as_tensor(sig.astype(np.float32),device=device)
    lookup={np.ascontiguousarray(x,dtype=np.float32).tobytes():i for i,x in enumerate(bases)}
    original=cr._positive_mask
    def functional_positive(batch_b,batch_episode,_threshold,*,enabled):
        n=len(batch_b); ident=torch.eye(n,dtype=torch.bool,device=batch_b.device)
        if not enabled: return ident
        x=batch_b.detach().cpu().numpy().astype(np.float32,copy=False)
        ix=[lookup.get(np.ascontiguousarray(row).tobytes(),-1) for row in x]
        if min(ix,default=-1)<0: return original(batch_b,batch_episode,_threshold,enabled=True)
        vectors=sig_tensor[torch.as_tensor(ix,device=device)]
        dist=torch.cdist(vectors,vectors).square()
        cross=batch_episode[:,None] != batch_episode[None,:]
        return ident | ((dist <= float(m6_epsilon)) & cross)
    fit_ix,_=_episode_train_validation_indices(qrows,seed+720)
    norm={}
    for dim in ("f","k","s"):
        norm[f"{dim}_mean"],norm[f"{dim}_scale"]=_standardize_fit(qrows[dim][fit_ix])
    with patch.object(cr,"_positive_mask",functional_positive):
        model,fit=cr.fit_dual_encoder(qrows,bases,use_subtask=True,subtask_mode="sentence",
            nearby_positive=True,seed=seed,epochs=cr.CLIP_EPOCHS,patience=cr.CLIP_PATIENCE,
            device=device,allowed_task_indices=tuple(np.unique(train_rows["task_index"]).tolist()))
    fit["normalizers"]={**norm}; fit["class_lookup"]={}
    train_e=af._query_embedding(model,fit,qrows,device=device)
    query={}
    for name,view in rowsets.items():
        tv=fc.attach_text(view,texts,emb,"instruction")
        query[name]=af._query_embedding(model,fit,af._normal_query_rows(tv),device=device)
    with torch.no_grad():
        unit_e=model.encode_skill(torch.as_tensor(centers,dtype=torch.float32,device=device)).cpu().numpy()
    return model,fit,train_e,query,unit_e


def _build_skill_memory(rows, bases, keys, centers, m1, *, seed):
    from . import hier_skills as hs
    primitive_ids = np.argmin(((bases[:, None] - centers[None]) ** 2).mean(-1), axis=1)
    segments, _ = hs._make_episode_segments(rows, bases, keys, mode="gripper",
        stride_seconds=.25, functional_threshold=0., functional_m1=m1,
        primitive_ids=primitive_ids)
    rng = np.random.default_rng(seed + 334)
    probe_ix = np.sort(rng.choice(len(rows["a"]), min(32, len(rows["a"])), replace=False))
    probe = {k: (v[probe_ix] if isinstance(v, np.ndarray) and len(v) == len(rows["a"]) else v)
             for k, v in rows.items()}
    memory, _, _ = hs._segment_memory(segments, m1, probe, seed)
    return memory, len(segments)


def _softmax_retrieval(query, embedding, bases, *, scale, k=5):
    from .review_tests import topk_candidates
    q = np.asarray(query, np.float32); e = np.asarray(embedding, np.float32)
    scores = (q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)) @ (
        e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-8)).T
    scores *= float(scale)
    ids, weights = topk_candidates(scores, k)
    return np.sum(np.asarray(bases, np.float32)[ids] * weights[..., None], axis=1), scores.max(1)


def _skill_retrieval(query, memory, m1, *, scale, k=5):
    keys = np.stack([u.key for u in memory.units]).astype(np.float32)
    q = np.asarray(query, np.float32)
    qn = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)
    kn = keys / np.maximum(np.linalg.norm(keys, axis=1, keepdims=True), 1e-8)
    scores = float(scale) * (qn @ kn.T)
    from .review_tests import topk_candidates
    ids, weights = topk_candidates(scores, k)
    candidate_bases = np.empty((len(query), k, 12), np.float32)
    maxima = scores.max(1)
    # Cache each skill's member keys/bases once; rank all query rows with BLAS.
    members = []
    for unit in memory.units:
        member_keys = np.concatenate([np.asarray(m["segment"]["keys"], np.float32)
                                      for m in unit.members], axis=0)
        member_bases = np.concatenate([np.asarray(m["segment"]["bases"], np.float32)
                                       for m in unit.members], axis=0)
        member_keys /= np.maximum(np.linalg.norm(member_keys, axis=1, keepdims=True), 1e-8)
        members.append((member_keys, member_bases))
    for sid, (member_keys, member_bases) in enumerate(members):
        query_ix, slots = np.where(ids == sid)
        if not len(query_ix):
            continue
        member_sim = qn[query_ix] @ member_keys.T
        best = member_sim.argmax(axis=1)
        candidate_bases[query_ix, slots] = member_bases[best]
    anchors = np.sum(candidate_bases * weights[..., None], axis=1)
    return anchors, maxima


def _mse_rows(pred, target):
    return np.mean((np.asarray(pred, np.float32) - np.asarray(target, np.float32)[None]) ** 2,
                   axis=(0, 2, 3))


def _confidence_features(query, keys, anchor, rows, m1):
    from . import fair_compare as fc
    from .review_tests import _key_features
    return _key_features(fc, query, keys, m1.decode(anchor,
        np.zeros((len(anchor), m1.residual_dim), np.float32), rows["f"], rows["s"]), rows, m1)


def _roll_f_anchor(fc, model, norm, rows, anchor, eps, t0, *, device):
    n = fc.steps_for(np.asarray(t0, np.float32))
    return fc.rollout(model, norm, rows["f"], rows["s"], anchor, eps,
                      np.asarray(t0, np.float32), n, device=device)


def _restore_flow(state, *, device):
    from . import anchor_flow as af
    model = af.FlowActionHead(2048, 8, 175).to(device)
    model.load_state_dict(state); model.eval()
    return model


def _evaluate_combo(fc, m1, flows, rows, target, eps, anchor, t0, *, device,
                    bridge_sigma=None):
    anchor_action = m1.decode(anchor, np.zeros((len(anchor), m1.residual_dim), np.float32),
                              rows["f"], rows["s"])
    if bridge_sigma is None:
        pred = _roll_f_anchor(fc, flows["anchor"][0], flows["anchor"][1],
                              rows, anchor_action, eps, t0, device=device)
    else:
        from .review_tests import _bridge_rollout
        pred = _bridge_rollout(flows[f"bridge{bridge_sigma:g}"][0],
            flows[f"bridge{bridge_sigma:g}"][1], rows, anchor_action, eps,
            sigma=bridge_sigma, n=10, device=device)
    return pred, _mse_rows(pred, target)


def run(output: Path, *, review_dir=DEFAULT_REVIEW, s3_dir=DEFAULT_S3,
        splits=("A", "B"), seeds=(0, 1, 2), device_name="cuda", threads=8,
        fixed_t0=.2, max_queries=None, resume=False):
    from . import anchor_flow as af
    from . import fair_compare as fc
    from . import hier_skills as hs
    from .splits import SPLITS, subset_rows as _subset
    from .experiment import DEFAULT_CACHE, DEFAULT_DATASET, _episode_train_validation_indices, load_rows

    output = Path(output)
    if output.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True,exist_ok=resume); (output / "figures").mkdir(exist_ok=resume)
    torch.set_num_threads(int(threads))
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    device = torch.device(device_name)
    review_dir, s3_dir = Path(review_dir), Path(s3_dir)
    base_results = json.loads((review_dir / "results.json").read_text())
    s3_results = json.loads((s3_dir / "results.json").read_text())
    archive = np.load(str(RESEARCH_STATE_DIR / 'outputs/memory-pilot/fair-compare-20261028T101842Z/per_row.npz'))
    rows_all = load_rows(DEFAULT_CACHE, DEFAULT_DATASET, max_horizon=fc.HORIZON)
    partial_path=output/"results.partial.json"
    runs=[]
    if resume and partial_path.exists():
        prior=json.loads(partial_path.read_text())
        if prior.get("config",{}).get("splits")!=list(splits) or prior.get("config",{}).get("seeds")!=list(seeds):
            raise ValueError("resume configuration differs from partial run")
        runs=list(prior.get("runs",[]))
    completed={(r["split"],int(r["seed"])) for r in runs}
    start = time.perf_counter()
    for split in splits:
        train_tasks, eval_tasks, _ = SPLITS[split]
        task_ids = sorted(set(train_tasks) | set(eval_tasks))
        selected = _subset(rows_all, np.isin(rows_all["task_index"], task_ids) & rows_all["valid_h25"])
        train_all = _subset(selected, np.isin(selected["task_index"], train_tasks))
        for seed in seeds:
            if (split,int(seed)) in completed:
                continue
            # Rebuild only the deterministic row split; all learned weights are loaded.
            train, d1 = fc.holdout_split(train_all, split, seed, per_task=fc.HOLDOUT_PER_TASK)
            fit_ix, val_ix = _episode_train_validation_indices(train, seed + 401)
            fit = _subset(train, np.isin(np.arange(len(train["a"])), fit_ix))
            val = _subset(train, np.isin(np.arange(len(train["a"])), val_ix))
            test = _subset(selected, np.isin(selected["task_index"], eval_tasks))
            rowsets = {"D1": d1, "D2": test, "D3": test}
            primary = torch.load(review_dir / f"{split}_s{seed}_models.pt", map_location=device, weights_only=False)
            m1 = _restore_m1(primary, device=device)
            bases, _ = m1.encode(fit["a"][:, :fc.HORIZON], fit["s"])
            r4_path=review_dir/f"{split}_s{seed}_R1_R4_models.pt"
            s3_run = next(r for r in s3_results["runs"] if r["split"] == split and r["seed"] == seed)
            if r4_path.exists():
                r4ckpt = torch.load(r4_path, map_location=device, weights_only=False)
                enc = _restore_encoder(r4ckpt["R4_E_state"], device=device)
                train_e, q_r4 = _load_r4_query_context(rowsets, model=enc, train_rows=fit, all_rows=selected,
                    seed=seed + 701, device=device, cache_dir=DEFAULT_CACHE)
                centers = np.asarray(r4ckpt["M6_centers"], np.float32)
                unit_emb = np.asarray(r4ckpt["unit_skill_embeddings"], np.float32)
                query_diff = float(np.max(np.abs(q_r4["D3"].astype(np.float16).astype(np.float32) -
                                             r4ckpt["D3_query"].astype(np.float32))))
                if query_diff > 2e-3:
                    raise ValueError(f"R4 query reconstruction mismatch for {split}/s{seed}: max={query_diff:.6g}")
                r4_source="saved_R4_checkpoint"
            elif (output/f"{split}_s{seed}_R4E_retrained.pt").exists() and (output/f"{split}_s{seed}_R4_query_embeddings.npz").exists():
                own=torch.load(output/f"{split}_s{seed}_R4E_retrained.pt",map_location=device,weights_only=False)
                enc=_restore_encoder(own["E_state"],device=device)
                qfile=np.load(output/f"{split}_s{seed}_R4_query_embeddings.npz")
                train_e=qfile["train"].astype(np.float32)
                q_r4={k:qfile[k].astype(np.float32) for k in ("D1","D2","D3")}
                centers=np.asarray(own["M6_centers"],np.float32)
                unit_emb=np.asarray(own["unit_skill_embeddings"],np.float32)
                r4ckpt={"M6_centers":centers,"unit_skill_embeddings":unit_emb,
                        "R4_E_state":own["E_state"],"D3_query":q_r4["D3"].astype(np.float16)}
                r4_source="retrained_E_checkpoint_resumed"
            else:
                initial_count=int(s3_run["M6_units"])
                centers=np.asarray(primary["M6_centers"][:initial_count],np.float32)
                epsilon=float(primary["M6_epsilon"])
                enc,_,train_e,q_r4,unit_emb=_fit_r4_assets(rowsets,train_rows=fit,all_rows=selected,
                    bases=bases,centers=centers,m1=m1,seed=seed+701,m6_epsilon=epsilon,
                    device=device,cache_dir=DEFAULT_CACHE)
                r4ckpt={"M6_centers":centers,"unit_skill_embeddings":unit_emb,
                        "R4_E_state":enc.state_dict(),"D3_query":q_r4["D3"].astype(np.float16)}
                r4_source="retrained_E_only_missing_checkpoint"
            if not r4_path.exists():
                torch.save({"E_state":{k:v.detach().cpu() for k,v in enc.state_dict().items()},
                    "M6_centers":centers,"unit_skill_embeddings":unit_emb.astype(np.float16),
                    "positive_rule":"cross-episode M6 functional-distance positives",
                    "functional_epsilon":float(primary["M6_epsilon"])},
                    output/f"{split}_s{seed}_R4E_retrained.pt")
                np.savez_compressed(output/f"{split}_s{seed}_R4_query_embeddings.npz",
                                    train=train_e.astype(np.float16),**{k:v.astype(np.float16) for k,v in q_r4.items()})
            if len(centers) != len(unit_emb):
                raise ValueError(f"M6 center/key count mismatch: {len(centers)} vs {len(unit_emb)}")
            skill_mem, n_segments = _build_skill_memory(fit, bases, train_e, centers, m1, seed=seed)
            expected_s3 = s3_run
            if len(skill_mem.units) != expected_s3["segmentation"]["gripper"]["n_skills"]:
                raise ValueError(f"S3 skill reconstruction mismatch {split}/s{seed}: "
                    f"{len(skill_mem.units)} != {expected_s3['segmentation']['gripper']['n_skills']}")
            skill_count = len(skill_mem.units)
            skill_hist = {"segments": n_segments, "skills": skill_count,
                          "s3_reference_skills": expected_s3["segmentation"]["gripper"]["n_skills"]}
            if "D1" not in q_r4 or "D3" not in q_r4:
                raise AssertionError("query extraction missing eval partition")
            # Canonical h features retain the exact representation on which R5 was calibrated.
            emb_np = np.load(review_dir / f"{split}_s{seed}_embeddings.npz")
            r5 = next(r for r in base_results["runs"] if r["split"] == split and r["seed"] == seed)["R5"]
            base_run = next(r for r in base_results["runs"] if r["split"] == split and r["seed"] == seed)
            train_keys = np.asarray(primary["train_keys"], np.float32)
            stats = primary["M1_stats"]
            m1.device = device
            bridge_states = torch.load(review_dir / f"{split}_s{seed}_R7_bridge_models.pt",
                                       map_location=device, weights_only=False)
            flows = {"anchor": (_restore_flow(primary["F_anchor"], device=device), primary["F_anchor_norm"])}
            for sigma in (.03, .1):
                skey = f"sigma{sigma:g}"
                flows[f"bridge{sigma:g}"] = (_restore_flow(bridge_states[skey]["state"], device=device),
                                               bridge_states[skey]["norm"])
            # Restore F_normal for the row-wise D1 comparison.
            normal_model = _restore_flow(primary["F_normal"], device=device)
            normal_norm = primary["F_normal_norm"]
            # Exact fair_compare noise stream, including the preceding validation draw.
            rng = np.random.default_rng(seed * 1_000_003 + 31)
            dim = 175
            eps_val = rng.standard_normal((fc.SAMPLES, len(val["a"]), dim)).astype(np.float32)
            eps_d1 = rng.standard_normal((fc.SAMPLES, len(d1["a"]), dim)).astype(np.float32)
            eps_d3 = rng.standard_normal((fc.SAMPLES, len(test["a"]), dim)).astype(np.float32)
            eps_by = {"D1": eps_d1, "D2": eps_d3, "D3": eps_d3}
            targets = {"D1": d1["a"][:, :fc.HORIZON], "D2": test["a"][:, :fc.HORIZON],
                       "D3": test["a"][:, :fc.HORIZON]}
            rows_by = rowsets
            normal_by = {}
            for cond in ("D1", "D2", "D3"):
                rows = rows_by[cond]; eps = eps_by[cond]
                normal_pred = fc.rollout(normal_model, normal_norm, rows["f"], rows["s"], None,
                    eps, np.ones(len(rows["a"]), np.float32), np.full(len(rows["a"]), 10), device=device)
                normal_by[cond] = _mse_rows(normal_pred, targets[cond])
            archive_key = f"{split}_s{seed}|instruction_D1|N:noise:n10|mse"
            archived_normal = np.asarray(archive[archive_key], np.float32)
            normal_archive_max_diff = float(np.max(np.abs(normal_by["D1"] - archived_normal)))
            if normal_archive_max_diff > 2e-5:
                raise ValueError(f"restored D1 F_normal differs from fair_compare: "
                                 f"{split}/s{seed}, max={normal_archive_max_diff:.6g}")
            reference_by = {}
            for cond in ("D1", "D2", "D3"):
                archive_prefix = f"{split}_s{seed}|instruction_{cond}|"
                def archive_mean(suffix):
                    key = archive_prefix + suffix
                    return float(np.mean(archive[key])) if key in archive.files else None
                equal = base_run["R8"]["conditions"]
                reference_by[cond] = {
                    "F_normal": float(normal_by[cond].mean()),
                    "full_raw_interval": archive_mean("N:raw:cal_sim|mse"),
                    "equal_count_random": equal["random_ids"].get(cond, {}).get("mse"),
                    "equal_count_medoid": equal["kmedoids_ids"].get(cond, {}).get("mse"),
                    "S3_hier_skill_saved": (expected_s3["D2_hierarchical_prequential"]["gripper"]["hier_skill"]["mse"]
                        if cond == "D2" else expected_s3["D1_D3"]["gripper"][cond]["hier_skill"]["mse"]),
                    "equal_count_budget": base_run["R8"]["item_budget"]}
            # R5's learned h: query similarity, key density and anchor/Prior disagreement.
            learned_risk, learned_t0 = {}, {}
            canonical_queries = {"D1": emb_np["D1_query"].astype(np.float32),
                                 "D2": emb_np["D2_query"].astype(np.float32),
                                 "D3": emb_np["D3_query"].astype(np.float32)}
            for cond in ("D1", "D2", "D3"):
                rows = rows_by[cond]
                base = emb_np[f"{cond}_unit_base"].astype(np.float32)
                feat = _confidence_features(canonical_queries[cond], train_keys, base, rows, m1)
                from .review_tests import predict_confidence_model
                risk = predict_confidence_model(r5["risk_model"], feat)
                learned_risk[cond] = risk
                learned_t0[cond] = fc.apply_g(r5["t0_calibration"], -risk).astype(np.float32)
            records = []
            harm_detail = {}
            for cond in ("D1", "D2", "D3"):
                rows, target, eps = rows_by[cond], targets[cond], eps_by[cond]
                q = q_r4[cond]
                r4blend, _ = _softmax_retrieval(q, unit_emb, centers,
                    scale=float(enc.logit_scale.exp().detach().cpu()), k=5)
                skillblend, _ = _skill_retrieval(q, skill_mem, m1,
                    scale=float(enc.logit_scale.exp().detach().cpu()), k=5)
                for unit_name, base in (("M6", r4blend), ("S3_gripper", skillblend)):
                    pred_anchor_action = m1.decode(base, np.zeros((len(base),4),np.float32), rows["f"],rows["s"])
                    risk_feat = _confidence_features(canonical_queries[cond], train_keys, base, rows, m1)
                    from .review_tests import predict_confidence_model
                    risk = predict_confidence_model(r5["risk_model"], risk_feat)
                    t0_policies = {"h_learned": fc.apply_g(r5["t0_calibration"], -risk).astype(np.float32),
                                   "fixed_0.2": np.full(len(base), fixed_t0, np.float32)}
                    for flow_name, bridge_sigma in (("F_anchor", None), ("bridge_0.03", .03), ("bridge_0.1", .1)):
                        for policy, t0 in t0_policies.items():
                            if max_queries is not None and len(base) > max_queries:
                                # Deterministic query subsampling for smoke only.
                                ids = np.linspace(0, len(base)-1, max_queries).astype(np.int64)
                                rr = {k:(v[ids] if isinstance(v,np.ndarray) and len(v)==len(base) else v) for k,v in rows.items()}
                                bb, tt, ee, nnorm, rk = base[ids], target[ids], eps[:,ids], normal_by[cond][ids], risk[ids]
                                t0_eval = t0[ids]
                            else:
                                ids=np.arange(len(base)); rr=rows; bb=base;tt=target;ee=eps;nnorm=normal_by[cond];rk=risk;t0_eval=t0
                            pred, row_mse = _evaluate_combo(fc,m1,flows,rr,tt,ee,bb,t0_eval,device=device,bridge_sigma=bridge_sigma)
                            record={"condition":cond,"unit":unit_name,"flow":flow_name,"confidence":policy,
                                "mse":float(row_mse.mean()),"best8":float(_mse_rows(pred,tt).min()) if False else None,
                                "frac_worse_any_vs_normal":float(np.mean(row_mse>nnorm)),
                                "frac_worse_5pct_vs_normal":float(np.mean(row_mse>nnorm*1.05)),
                                "average_t0":float(np.mean(t0_eval)),"t0_applied_to_flow":bridge_sigma is None,
                                "baseline_normal_mse":float(nnorm.mean()),"n":int(len(ids))}
                            # best-of-8 is the mean across rows of each row's best noise draw.
                            best8=np.mean(np.min(np.mean((pred-tt[None])**2,axis=(2,3)),axis=0))
                            record["best8"]=float(best8)
                            records.append(record)
                            if cond=="D1" and policy=="h_learned":
                                edges=np.quantile(rk,np.linspace(0,1,7)); bins=np.digitize(rk,edges[1:-1],right=True)
                                bd=[]
                                for bidx in range(6):
                                    mask=bins==bidx
                                    bd.append({"bin":bidx,"n":int(mask.sum()),"risk_mean":float(rk[mask].mean()),
                                        "t0_mean":float(t0_eval[mask].mean()),
                                        "worse_any":float(np.mean(row_mse[mask]>nnorm[mask])) if mask.any() else None,
                                        "worse_5pct":float(np.mean(row_mse[mask]>nnorm[mask]*1.05)) if mask.any() else None})
                                harm_detail[f"{unit_name}/{flow_name}"]=bd
            # D1 t0=.4 safety sensitivity on M6 with the three correction paths.
            fixed04=[]
            if max_queries is None:
                q=q_r4["D1"]; base,_=_softmax_retrieval(q,unit_emb,centers,
                    scale=float(enc.logit_scale.exp().detach().cpu()),k=5)
                for flow_name,sigma in (("F_anchor",None),("bridge_0.03",.03),("bridge_0.1",.1)):
                    pred,row_mse=_evaluate_combo(fc,m1,flows,d1,d1["a"][:,:fc.HORIZON],eps_d1,base,
                        np.full(len(base),.4,np.float32),device=device,bridge_sigma=sigma)
                    fixed04.append({"flow":flow_name,"t0":.4,
                        "frac_worse_any_vs_normal":float(np.mean(row_mse>normal_by['D1'])),
                        "frac_worse_5pct_vs_normal":float(np.mean(row_mse>normal_by['D1']*1.05)),
                        "mse":float(row_mse.mean())})
            # Existing S3 full-run point estimates, retained as an independent reference.
            s3_ref={k:expected_s3["D1_D3"]["gripper"][k] for k in ("D1","D3")}
            runs.append({"split":split,"seed":seed,"skill_memory":skill_hist,
                         "combos":records,"D1_harm_by_confidence_bin":harm_detail,
                         "D1_fixed_t0_0.4_sensitivity":fixed04,
                         "references":reference_by,
                         "S3_saved_reference":s3_ref,
                         "R4_model_source":r4_source,
                         "D1_normal_archive_max_abs_difference":normal_archive_max_diff,
                         "checks":{"R4_query_half_precision_max_abs":float(np.max(np.abs(q_r4['D3'].astype(np.float16).astype(np.float32)-r4ckpt['D3_query'].astype(np.float32)))),
                                   "skill_count_matches_saved_S3":True}})
            del primary,r4ckpt,m1,enc,flows,skill_mem
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            partial_payload={"schema_version":"review-v2-combinations-v1",
                "config":{"review_source":str(review_dir),"s3_source":str(s3_dir),"splits":list(splits),
                  "seeds":list(seeds),"horizon":25,"device":device_name,"threads":threads,
                  "combinations":"E multipositive top-5 fixed; F_anchor/bridge sigma .03/.10 × learned h/fixed t0=.2 × M6/S3 gripper skills",
                  "fixed_t0_sensitivity":[.2,.4],"frozen_weights":True,"max_queries":max_queries},
                "runs":runs,"elapsed_seconds":time.perf_counter()-start}
            partial_path.write_text(json.dumps(partial_payload,indent=1,ensure_ascii=False)+"\n")
    payload={"schema_version":"review-v2-combinations-v1",
        "config":{"review_source":str(review_dir),"s3_source":str(s3_dir),"splits":list(splits),
                  "seeds":list(seeds),"horizon":25,"device":device_name,"threads":threads,
                  "combinations":"E multipositive top-5 fixed; F_anchor/bridge sigma .03/.10 × learned h/fixed t0=.2 × M6/S3 gripper skills",
                  "fixed_t0_sensitivity":[.2,.4],"frozen_weights":True,"max_queries":max_queries},
        "runs":runs,"elapsed_seconds":time.perf_counter()-start,
        "interpretation_limits":["h is reused from R5; on S3 anchors its third input changes, so it is a transfer test, not recalibrated confidence.",
            "D2 is scored with the stored sequential-query embeddings and frozen training memories; this does not append D2 episodes during v2.",
            "S3 groups are reconstructed with the saved M1 and R4 encoder; skill counts are checked against the saved S3 run.",
            "For seeds whose R4 model was not present in review-tests, only the multi-positive E was retrained on the frozen saved M1 bases; M1/F were not retrained."]}
    (output/"results.json").write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n")
    _write_summary(output/"summary.md",payload)
    _plot(output/"figures",payload)
    return output


def _write_summary(path, payload):
    lines=["# Review v2 combinations (offline, frozen checkpoint)","",
        f"- Conditions: H=25, Split {payload['config']['splits']}, seeds {payload['config']['seeds']}; saved M1/E/F/R5/R7 weights reused.",
        "- E is multi-positive InfoNCE with top-5 softmax; unit choices are M6 primitives or reconstructed S3 gripper groups.",
        "- D1 harm is per-row MSE above F_normal: report both any increase and increase >5%; h bins are D1 risk quantiles.",""]
    for split in payload['config']['splits']:
        rs=[r for r in payload['runs'] if r['split']==split]
        lines += [f"## Split {split}",""]
        grouped={}
        for r in rs:
            for x in r['combos']:
                if x['condition']=="D3":
                    key=(x['unit'],x['flow'],x['confidence'])
                    grouped.setdefault(key,[]).append(x['mse'])
        best=sorted(((float(np.mean(v)),k) for k,v in grouped.items()),key=lambda q:q[0])[:3]
        lines.append("Best D3 combinations averaged over available seeds (lower is better): " +
                     "; ".join(f"{k[0]}/{k[1]}/{k[2]}={v:.4f}" for v,k in best)+".")
        ref_rows=[]
        for cond in ("D1","D2","D3"):
            for method in ("F_normal","full_raw_interval","equal_count_random","equal_count_medoid","S3_hier_skill_saved"):
                vals=[r['references'][cond].get(method) for r in rs]
                vals=[float(v) for v in vals if v is not None]
                ref_rows.append((cond,method,float(np.mean(vals)) if vals else None))
        lines += ["| Condition | Baseline/reference | MSE (seed mean) |", "|---|---|---:|"]
        for cond,method,value in ref_rows:
            lines.append(f"| {cond} | {method} | {'NA' if value is None else f'{value:.4f}'} |")
        lines.append("")
        lines.append("| Seed | Condition | Unit | Flow | Confidence | MSE | best-of-8 | D1 worse / >5% | mean t0 |")
        lines.append("|---:|---|---|---|---|---:|---:|---:|---:|")
        for r in rs:
            for x in r['combos']:
                if x['condition']=="D3":
                    # The D1 fraction is attached by joining the corresponding D1 record.
                    d1=next((a for a in r['combos'] if a['condition']=="D1" and a['unit']==x['unit'] and a['flow']==x['flow'] and a['confidence']==x['confidence']),None)
                    harm="NA" if d1 is None else f"{d1['frac_worse_any_vs_normal']:.3f} / {d1['frac_worse_5pct_vs_normal']:.3f}"
                    lines.append(f"| {r['seed']} | D3 | {x['unit']} | {x['flow']} | {x['confidence']} | {x['mse']:.4f} | {x['best8']:.4f} | {harm} | {x['average_t0']:.3f} |")
        lines.append("")
        lines += ["| Seed | D1 unit/flow | confidence bin t0 / harmed any / harmed >5% (low→high risk) |",
                  "|---:|---|---|"]
        for r in rs:
            for name,bins in r['D1_harm_by_confidence_bin'].items():
                vals="; ".join(f"{b['t0_mean']:.2f}/{b['worse_any']:.2f}/{b['worse_5pct']:.2f}" for b in bins)
                lines.append(f"| {r['seed']} | {name} | {vals} |")
        lines.append("")
    lines += ["## Interpretation", "",
        "Bridge integrates its full t=1→0 path, so t0 does not alter bridge predictions; the learned-h/fixed-t0 labels are retained in the factorial table for bookkeeping and must be read as identical bridge variants. h/fixed comparisons are operational for F_anchor.",
        "D1 safety should be selected from training-suite held-out risk calibration: cap F_anchor t0 at 0.2 in low-confidence bins, and only relax it for high-confidence bins if the measured D1 harmed rate is acceptably low. Fixed t0=.4 is a sensitivity point, not a value to select from D1 labels.",
        "S3 point estimates are also retained from the saved hierarchical run; the v2 interaction grid reconstructs S3 gripper groups with the frozen R4 encoder. D2 uses the saved sequential query embeddings but keeps the v2 training memory frozen.",""]
    path.write_text("\n".join(lines))


def _plot(directory, payload):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory=Path(directory)
    for split in payload['config']['splits']:
        rs=[r for r in payload['runs'] if r['split']==split]
        combos=[x for r in rs for x in r['combos'] if x['condition']=="D3"]
        labels=[f"{x['unit']}\n{x['flow']}\n{x['confidence']}" for x in combos]
        vals=[x['mse'] for x in combos]
        fig,ax=plt.subplots(figsize=(max(12,len(vals)*.25),5))
        ax.bar(np.arange(len(vals)),vals);ax.set_xticks(np.arange(len(vals)),labels,rotation=75,ha='right',fontsize=6)
        ax.set_ylabel("D3 MSE");ax.set_title(f"Split {split}: v2 combinations");ax.grid(axis='y',alpha=.2)
        fig.tight_layout();fig.savefig(directory/f"{split}_D3_combinations.png",dpi=120);plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--review-dir",type=Path,default=DEFAULT_REVIEW)
    parser.add_argument("--s3-dir",type=Path,default=DEFAULT_S3)
    parser.add_argument("--splits",default="A,B");parser.add_argument("--seeds",default="0,1,2")
    parser.add_argument("--device",default="cuda");parser.add_argument("--threads",type=int,default=8)
    parser.add_argument("--fixed-t0",type=float,default=.2)
    parser.add_argument("--max-queries",type=int,default=None,help="smoke-only deterministic query cap")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args(argv)
    out=run(args.output,review_dir=args.review_dir,s3_dir=args.s3_dir,
        splits=tuple(args.splits.split(',')),seeds=tuple(map(int,args.seeds.split(','))),
        device_name=args.device,threads=args.threads,fixed_t0=args.fixed_t0,max_queries=args.max_queries,resume=args.resume)
    print(out)


if __name__ == '__main__': main()
