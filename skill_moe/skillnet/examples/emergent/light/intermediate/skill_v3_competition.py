"""V3 competitive experts and V4 scene-conditioned geometric skills.

This module is deliberately standalone: skill_v2 is imported as a read-only
evaluation/toolbox module. Task and language labels are consumed only by the
post-hoc reporting path.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import skill_v2 as v2
from ....research_paths import REPO_ROOT as RESEARCH_REPO_ROOT, STATE_DIR as RESEARCH_STATE_DIR


class ExpertDecoder(nn.Module):
    """Small per-skill decoder: scene, proprioception and interval phase -> action."""
    def __init__(self, scene_dim: int, state_dim: int, action_dim: int, width: int = 96):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(scene_dim + state_dim + 1, width), nn.GELU(),
                                  nn.Linear(width, width), nn.GELU(), nn.Linear(width, action_dim))

    def forward(self, scene, state, phase):
        if phase.ndim == 1:
            phase = phase[:, None]
        return self.net(torch.cat((scene, state, phase), dim=-1))


def _phase_rows(segment):
    return np.linspace(0., 1., len(segment["rows"]), dtype=np.float32)


def _segment_model_data(rows, segments, *, device="cpu"):
    """Flatten segmented frame rows while retaining per-segment spans."""
    scene, state, phase, target, owner = [], [], [], [], []
    for j, seg in enumerate(segments):
        ix = np.asarray(seg["rows"], np.int64)
        if not len(ix):
            continue
        scene.append(np.asarray(rows["k"][ix], np.float32))
        state.append(np.asarray(rows["s"][ix], np.float32))
        phase.append(_phase_rows(seg))
        target.append(np.asarray(rows["a"][ix, 0], np.float32))
        owner.extend([j] * len(ix))
    if not scene:
        raise ValueError("no non-empty segments")
    arrays = (np.concatenate(scene), np.concatenate(state), np.concatenate(phase),
              np.concatenate(target), np.asarray(owner, np.int64))
    return tuple(torch.as_tensor(x, device=device) for x in arrays)


def _decoder_losses(model, data):
    scene, state, phase, target, owner = data
    model.eval()
    with torch.no_grad():
        pred = model(scene, state, phase)
        frame_loss = (pred - target).square().mean(-1)
    out = np.full(int(owner.max()) + 1, np.inf, np.float32)
    own = owner.detach().cpu().numpy()
    loss = frame_loss.detach().cpu().numpy()
    for i in range(len(out)):
        mask = own == i
        if mask.any():
            out[i] = float(loss[mask].mean())
    return out


def _rotation_between(a, b):
    a=np.asarray(a,np.float32); b=np.asarray(b,np.float32)
    a=a/max(float(np.linalg.norm(a)),1e-8); b=b/max(float(np.linalg.norm(b)),1e-8)
    v=np.cross(a,b); c=float(np.dot(a,b)); sn=float(np.linalg.norm(v))
    if sn<1e-7:
        return np.eye(3,dtype=np.float32) if c>0 else np.diag([-1.,-1.,1.]).astype(np.float32)
    vx=np.asarray([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]],np.float32)
    return np.eye(3,dtype=np.float32)+vx+vx@vx*((1-c)/(sn*sn))


def _fit_decoder(model, data, frame_mask, *, seed, epochs=18, lr=8e-4, weights=None):
    scene, state, phase, target, _owner = data
    ix = torch.as_tensor(frame_mask, dtype=torch.bool, device=scene.device)
    w = None if weights is None else torch.as_tensor(weights,dtype=torch.float32,device=scene.device)
    if not bool(ix.any()):
        return {"epochs": 0, "loss": None}
    torch.manual_seed(int(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    best, best_state, stale = float("inf"), None, 0
    batch = min(2048, int(ix.sum()))
    for epoch in range(epochs):
        model.train()
        order = torch.where(ix)[0]
        order = order[torch.randperm(len(order), device=order.device)]
        losses = []
        for start in range(0, len(order), batch):
            b = order[start:start + batch]
            row_loss=(model(scene[b], state[b], phase[b]) - target[b]).square().mean(-1)
            if w is None: loss=row_loss.mean()
            else:
                wb=w[b];loss=(row_loss*wb).sum()/wb.sum().clamp_min(1e-6)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step(); losses.append(float(loss.detach().cpu()))
        value = float(np.mean(losses))
        if value < best - 1e-6:
            best, best_state, stale = value, {k:v.detach().clone() for k,v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= 3:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {"epochs": epoch + 1, "loss": best}


class CompetitiveExperts:
    """Hard-EM experts with error-quantile births and functional merging.

    Birth proposals are clusters of training segments above the current best-
    expert q95 error. M6's support/episode guard is reused for these proposals;
    it is not used to assign ordinary segments to an existing expert.
    """
    def __init__(self, scene_dim, state_dim, action_dim, *, q=0.95,
                 min_support=10, min_episodes=3, max_rounds=3, temperature=0.,device="cpu"):
        self.scene_dim, self.state_dim, self.action_dim = scene_dim, state_dim, action_dim
        self.q, self.min_support, self.min_episodes = q, min_support, min_episodes
        self.max_rounds, self.temperature, self.device = max_rounds, float(temperature), torch.device(device)
        self.experts = []
        self.assignments = np.zeros(0, np.int64)
        self.births, self.merges, self.thresholds = [], [], []

    def fit(self, rows, segments, *, seed=0, epochs=18):
        if len(segments) < 1:
            raise ValueError("V3 needs at least one training segment")
        data = _segment_model_data(rows, segments, device=self.device)
        scene, state, phase, target, owner = data
        frame_owner = owner.detach().cpu().numpy()
        first = ExpertDecoder(self.scene_dim, self.state_dim, self.action_dim).to(self.device)
        self.experts = [first]
        all_frames = np.ones(len(frame_owner), bool)
        self.fit_history = [_fit_decoder(first, data, all_frames, seed=seed, epochs=epochs)]
        seg_ep = np.asarray([str(s["episode_id"]) for s in segments])
        self.assignments = np.zeros(len(segments), np.int64)
        for round_ix in range(self.max_rounds):
            losses = np.stack([_decoder_losses(m, data) for m in self.experts], axis=1)
            if self.temperature>0:
                logits=-losses/max(self.temperature,1e-6)
                logits-=logits.max(axis=1,keepdims=True)
                resp=np.exp(logits);resp/=np.maximum(resp.sum(axis=1,keepdims=True),1e-8)
                self.responsibilities=resp.astype(np.float32)
                self.assignments=resp.argmax(axis=1)
            else:
                self.assignments = losses.argmin(axis=1)
                self.responsibilities=np.eye(len(self.experts),dtype=np.float32)[self.assignments]
            best = losses.min(axis=1)
            threshold = float(np.quantile(best[np.isfinite(best)], self.q))
            self.thresholds.append(threshold)
            hard = np.zeros(len(frame_owner), bool)
            for sid, expert_id in enumerate(self.assignments):
                hard[frame_owner == sid] = expert_id == self.assignments[sid]
            for expert_id, model in enumerate(self.experts):
                weights=self.responsibilities[frame_owner,expert_id]
                mask=weights>1e-6
                self.fit_history.append(_fit_decoder(model, data, mask, seed=seed + round_ix*19 + expert_id + 1,
                                                     epochs=epochs,weights=weights))
            outlier = np.flatnonzero(best > threshold)
            # A train-only candidate memory proposes a new decoder only when
            # the high-error set has repeatable support across episodes.
            if round_ix < self.max_rounds - 1 and len(outlier) >= self.min_support:
                from . import skill_v2 as v2
                proposal_segments = [segments[int(i)] for i in outlier]
                # Cluster the high-error proposal set at M6's training-derived
                # functional scale instead of collapsing every outlier into one.
                proposal = v2.TrajectorySkillMemory(1., kappa=.1,
                    min_support=self.min_support, min_episodes=self.min_episodes).fit(proposal_segments, seed=seed + 333 + round_ix)
                for unit in proposal.units:
                    ids = [int(m["segment_id"]) for m in unit.members]
                    if len({seg_ep[i] for i in ids}) < self.min_episodes:
                        continue
                    model = ExpertDecoder(self.scene_dim, self.state_dim, self.action_dim).to(self.device)
                    self.experts.append(model)
                    # Train only the new expert on its proposed outlier group.
                    mask = np.isin(frame_owner, ids)
                    fit = _fit_decoder(model, data, mask, seed=seed + 991 + len(self.experts), epochs=epochs)
                    self.fit_history.append(fit)
                    self.births.append({"round": round_ix, "expert_id": len(self.experts)-1,
                                        "support": len(ids), "episodes": len({seg_ep[i] for i in ids}),
                                        "threshold": threshold})
            # Merge experts whose predictions are functionally indistinguishable
            # on a fixed common training-frame probe.
            if len(self.experts) > 1:
                probe = np.linspace(0, len(frame_owner)-1, min(256, len(frame_owner))).astype(int)
                merged = set()
                for i in range(len(self.experts)):
                    if i in merged: continue
                    for j in range(i+1, len(self.experts)):
                        if j in merged: continue
                        with torch.no_grad():
                            pi = self.experts[i](scene[probe], state[probe], phase[probe])
                            pj = self.experts[j](scene[probe], state[probe], phase[probe])
                            dist = float((pi-pj).square().mean().cpu())
                        if dist < max(1e-5, threshold * .02):
                            merged.add(j)
                            self.merges.append({"removed": j, "into": i, "functional_mse": dist})
                if merged:
                    self.experts = [x for i,x in enumerate(self.experts) if i not in merged]
        losses = np.stack([_decoder_losses(m, data) for m in self.experts], axis=1)
        self.assignments = losses.argmin(axis=1)
        used=np.unique(self.assignments)
        if len(used)<len(self.experts):
            remap={int(old):new for new,old in enumerate(used.tolist())}
            self.experts=[self.experts[int(old)] for old in used]
            self.assignments=np.asarray([remap[int(x)] for x in self.assignments],np.int64)
            self.responsibilities=np.eye(len(self.experts),dtype=np.float32)[self.assignments]
        self.threshold = float(np.quantile(losses.min(axis=1), self.q))
        self.member_ids = self.assignments.copy()
        self.training_segment_errors = losses.min(axis=1)
        self._train_data = data
        self._frame_owner = frame_owner
        return self

    def predict(self, rows, phases, *, device=None):
        """Return all expert predictions, (N,K,A), and router scores, (N,K)."""
        device = torch.device(device or self.device)
        scene = torch.as_tensor(rows["k"], dtype=torch.float32, device=device)
        state = torch.as_tensor(rows["s"], dtype=torch.float32, device=device)
        phase = torch.as_tensor(phases, dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = torch.stack([m(scene, state, phase) for m in self.experts], dim=1)
        # A light scene/state router is fitted below from hard assignments.
        scores = self.router(torch.cat((scene,state,phase[:,None]),dim=-1)).detach().cpu().numpy()
        return pred.detach().cpu().numpy(), scores

    def fit_router(self, rows, segments, *, seed=0, epochs=40):
        device=self.device
        scene, state, phase, _target, owner = self._train_data
        labels = torch.as_tensor(self.assignments[owner.detach().cpu().numpy()], device=device)
        xdim = self.scene_dim + self.state_dim + 1
        self.router = nn.Sequential(nn.Linear(xdim,96),nn.GELU(),nn.Linear(96,len(self.experts))).to(device)
        torch.manual_seed(seed); opt=torch.optim.AdamW(self.router.parameters(),lr=1e-3,weight_decay=1e-5)
        for _ in range(epochs):
            logits=self.router(torch.cat((scene,state,phase[:,None]),-1))
            loss=nn.functional.cross_entropy(logits,labels)
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        self.router.eval()
        return self


def canonicalize_v4(curve):
    """Remove net translation direction and distance from a relative action curve."""
    x=np.asarray(curve,np.float32).copy()
    xyz=x[:,:3]-x[:1,:3]
    net=xyz[-1]
    dist=float(np.linalg.norm(net))
    direction=net/max(dist,1e-8)
    if dist < 1e-7: direction=np.asarray([1.,0.,0.],np.float32)
    # Rotate direction onto +x with a stable Rodrigues rotation.
    a=np.asarray([1.,0.,0.],np.float32); b=direction.astype(np.float32)
    rot=_rotation_between(b,a)
    x[:,:3]=(xyz@rot.T)/max(dist,1e-7)
    return x, direction, dist


def restore_v4_geometry(shape, direction, distance):
    """Restore scene-predicted direction and distance to a canonical motion."""
    x=np.asarray(shape,np.float32).copy(); d=np.asarray(direction,np.float32)
    d=d/max(float(np.linalg.norm(d)),1e-8)
    a=np.asarray([1.,0.,0.],np.float32); rot=_rotation_between(a,d)
    x[:,:3]=(x[:,:3]@rot.T)*max(float(distance),0.)
    return x


def fit_v4_geometry_regressor(segments, geometries, ridge=2.0):
    """Ridge map from scene query keys to motion direction and log distance."""
    x=np.stack([s["key"] for s in segments]).astype(np.float64)
    x=np.c_[np.ones(len(x)),x]
    y=np.c_[[g[0] for g in geometries],np.log(np.maximum([g[1] for g in geometries],1e-4))]
    penalty=np.eye(x.shape[1])*float(ridge); penalty[0,0]=0
    coef=np.linalg.solve(x.T@x+penalty,x.T@y)
    return {"coef":coef.astype(np.float32),"input_dim":int(x.shape[1]-1),"ridge":float(ridge)}


def predict_v4_geometry(regressor, query):
    q=np.asarray(query,np.float32); x=np.c_[np.ones(len(q)),q]
    y=x@np.asarray(regressor["coef"],np.float32)
    direction=y[:,:3]
    direction/=np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-8)
    distance=np.exp(np.clip(y[:,3],-8,3))
    return direction.astype(np.float32),distance.astype(np.float32)


def _verb_from_plan(text):
    """Post-hoc first-action verb heuristic; never called during fitting."""
    tokens="".join(ch.lower() if ch.isalnum() else " " for ch in str(text)).split()
    while tokens and tokens[0] in {"the","a","an","robot","arm","then"}:
        tokens.pop(0)
    return tokens[0] if tokens else "<empty>"


def _v4_schedule(rows, query, segments, memory, regressor, *, stride=.5):
    q=np.asarray(query,np.float32); qn=q/np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-8)
    keys=np.stack([u.key for u in memory.units]).astype(np.float32)
    kn=keys/np.maximum(np.linalg.norm(keys,axis=1,keepdims=True),1e-8)
    sim=qn@kn.T; ids=np.argmax(sim,axis=1)
    direction,distance=predict_v4_geometry(regressor,q)
    anchor=np.zeros((len(q),v2.HORIZON,rows["a"].shape[-1]),np.float32)
    phase=np.zeros(len(q),np.float32)
    for seg in segments:
        ix=np.asarray(seg["rows"],np.int64)
        if len(ix): phase[ix]=_phase_rows(seg)
    for row,uid in enumerate(ids):
        unit=memory.units[int(uid)]
        restored=restore_v4_geometry(unit.mean,direction[row],distance[row])
        anchor[row]=v2._trajectory_chunk(restored,phase[row],unit.median_length,stride)
    scores=sim
    conf=np.zeros(len(q),np.float32); margin=np.zeros(len(q),np.float32)
    for i in range(len(q)):
        conf[i],_,margin[i]=v2._confidence_for(scores[i],int(ids[i]))
    top_ids=np.argsort(-scores,axis=1)[:,:min(5,scores.shape[1])]
    return {"anchor":anchor,"skill_id":ids,"phi":phase,"confidence":conf,
            "similarity":scores[np.arange(len(q)),ids],"margin":margin,"scores":scores,"top_ids":top_ids}


def _v3_schedule(rows, segments, model, *, device="cpu"):
    phase=np.zeros(len(rows["a"]),np.float32)
    for seg in segments:
        ix=np.asarray(seg["rows"],np.int64)
        if len(ix): phase[ix]=_phase_rows(seg)
    scene=torch.as_tensor(rows["k"],dtype=torch.float32,device=device)
    state=torch.as_tensor(rows["s"],dtype=torch.float32,device=device)
    ph=torch.as_tensor(phase,dtype=torch.float32,device=device)[:,None]
    with torch.no_grad():
        scores=model.router(torch.cat((scene,state,ph),dim=-1)).cpu().numpy()
        ids=scores.argmax(1)
        pred=np.stack([m(scene,state,ph).cpu().numpy() for m in model.experts],axis=1)
    anchor=pred[np.arange(len(ids)),ids]
    # The flow head consumes a 25-step chunk. Repeat the row-action prediction
    # across that chunk as a local constant target for its denoising correction.
    anchor=np.repeat(anchor[:,None,:],v2.HORIZON,axis=1)
    conf=np.zeros(len(ids),np.float32);margin=np.zeros(len(ids),np.float32)
    for i in range(len(ids)): conf[i],_,margin[i]=v2._confidence_for(scores[i],int(ids[i]))
    top_ids=np.argsort(-scores,axis=1)[:,:min(5,scores.shape[1])]
    return {"anchor":anchor,"skill_id":ids,"phi":phase,"confidence":conf,
            "similarity":scores[np.arange(len(ids)),ids],"margin":margin,"scores":scores,"top_ids":top_ids}


def _prequential_d2(variant, rows, query, segments, memory, model, flow, norm,
                    calibration, eps, *, seed, device, stride=.5):
    """Per-episode D2: query first, then append that episode without gradients."""
    from .compare_splits import _subset
    n=len(rows["a"]); target=rows["a"][:,:v2.HORIZON]
    pred=np.zeros((eps.shape[0],n,*target.shape[1:]),np.float32)
    schedule={"anchor":np.zeros_like(target),"skill_id":np.full(n,-1,np.int64),
              "phi":np.zeros(n,np.float32),"confidence":np.zeros(n,np.float32),
              "similarity":np.zeros(n,np.float32),"margin":np.zeros(n,np.float32),
              "top_ids":np.full((n,5),-1,np.int64)}
    order=np.random.default_rng(seed+841).permutation(np.unique(rows["episode_id"].astype(str)))
    stream=memory.clone()
    for ep in order:
        ix=np.flatnonzero(rows["episode_id"].astype(str)==ep)
        ix=ix[np.argsort(rows["frame"][ix],kind="stable")]
        sub=_subset(rows,np.isin(np.arange(n),ix)); qsub=np.asarray(query)[ix]
        ss=[dict(s,rows=np.searchsorted(ix,s["rows"])) for s in segments if s["episode_id"]==ep]
        if not ss: continue
        if variant=="V3": sched=_v3_schedule(sub,ss,model,device=device)
        else: sched=_v4_schedule(sub,qsub,ss,stream,model,stride=stride)
        tau,steps=v2.apply_confidence_map(calibration,sched["confidence"])
        out=v2._predict(flow,norm,sub,sched["anchor"],eps[:,ix],tau,steps,device)
        pred[:,ix]=out
        for k in schedule:
            if k=="top_ids": schedule[k][ix,:sched[k].shape[1]]=sched[k]
            else: schedule[k][ix]=sched[k]
        # New intervals are appended only after their episode has been queried.
        if variant=="V4":
            canon=[]
            for s in segments:
                if s["episode_id"]==ep:
                    curve,_,_=canonicalize_v4(s["curve"])
                    canon.append(dict(s,curve=curve))
            stream.add_episode(canon)
        else:
            # Expert assignments are stored as metadata; weights and router are
            # frozen, as in the gradient-free memory-add condition.
            for seg in ss:
                row0=int(seg["rows"][0]); uid=int(sched["skill_id"][row0]); unit=stream.units[uid]
                unit.members.append(seg); unit.support+=1
                unit.tasks.add(int(seg["task_index"]));unit.episodes.add(str(seg["episode_id"]))
                unit.key=(unit.key*(unit.support-1)+np.asarray(seg["key"],np.float32))/unit.support
        # V4 updates its prototypes at existing-unit assignment time and flushes
        # its normal pending rule only after enough new intervals accumulated.
    if variant=="V4": stream.finish_stream()
    return pred,schedule,stream


def _summary(result, output):
    lines=["# Skill V3 competition / V4 geometry study", "",
           "Rows and fit labels are unlabeled; task and plan text are used only for post-hoc metrics.", "",
           "| Split/seed | Variant | Skills | Length median (s) | Tasks/skill median | Verb purity | D1 top1/top5 | D3 top1/top5 | D1/D2/D3 MSE | D1 worse vs F_normal | b-swap Δaction |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in result.get("runs",[]):
        for variant,x in r["variants"].items():
            lines.append(f"| {r['split']}/{r['seed']} | {variant} | {x['n_skills']} | {x['median_length_seconds']:.2f} | {x['median_tasks_per_skill']:.1f} | {x['verb_purity']:.3f} | {x['D1_top1']:.3f}/{x['D1_top5']:.3f} | {x['D3_top1']:.3f}/{x['D3_top5']:.3f} | {x['D1_MSE']:.4f}/{x['D2_MSE']:.4f}/{x['D3_MSE']:.4f} | {x['D1_worse_fraction']:.3f} | {x['b_swap_action_delta']:.4f} |")
        b=r.get("baselines_v21_D3",{})
        lines.append(f"Baseline {r['split']}/{r['seed']} D3 MSE: F_normal={b.get('normal',float('nan')):.4f}, raw={b.get('raw_interval',float('nan')):.4f}, P-a={b.get('primitive_Pa',float('nan')):.4f}, v2.1={b.get('skill_v2',float('nan')):.4f}.")
    lines += ["", "Baselines are referenced from saved v2.1 results under the same split/seed. D2 queries each episode before adding it without gradient updates; V3 expert/router weights remain frozen. See results.json for per-cell formation history.", ""]
    (output/"summary.md").write_text("\n".join(lines))


def run(*, output, splits=("A","B"), seeds=(0,1,2), device="cuda", threads=8,
        smoke=False, max_queries=None):
    """Run the V3/V4 comparison through v2's action-flow evaluation helpers."""
    from . import anchor_flow as af, fair_compare as fc
    from .compare_splits import SPLITS, _subset
    from .experiment import DEFAULT_CACHE, DEFAULT_DATASET, _episode_train_validation_indices, load_rows
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True); (output/"checkpoints").mkdir()
    torch.set_num_threads(int(threads))
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    dev=torch.device(device); start=time.perf_counter()
    rows=load_rows(DEFAULT_CACHE,DEFAULT_DATASET,max_horizon=fc.HORIZON)
    cache_dir=Path(DEFAULT_CACHE)
    all_outputs=[]
    for split in splits:
      train_tasks,eval_tasks,_=SPLITS[split]
      selected=_subset(rows,np.isin(rows["task_index"],sorted(set(train_tasks)|set(eval_tasks))) & rows["valid_h25"])
      train_all=_subset(selected,np.isin(selected["task_index"],train_tasks))
      for seed in seeds:
        tic=time.perf_counter()
        fit_rows,d1=fc.holdout_split(train_all,split,seed,per_task=2 if smoke else fc.HOLDOUT_PER_TASK)
        if smoke:
          fit_rows=af._select_episodes(fit_rows,train_tasks,12,seed+101)
          d1=af._select_episodes(d1,train_tasks,4,seed+311)
        fit_ix,val_ix=_episode_train_validation_indices(fit_rows,seed+401)
        fit=_subset(fit_rows,np.isin(np.arange(len(fit_rows["a"])),fit_ix))
        val=_subset(fit_rows,np.isin(np.arange(len(fit_rows["a"])),val_ix))
        test=_subset(selected,np.isin(selected["task_index"],eval_tasks))
        if smoke: test=af._select_episodes(test,eval_tasks,4,seed+211)
        # The learning/routing features are language-free image features k plus
        # proprioception. Task/plan labels are read only for post-hoc reporting.
        m6,fit_bases,actions=af._fit_m6_backbone_and_units(fit,seed=seed+19,device=dev,epochs=12 if smoke else 60)
        m1=m6.backbone
        def embed(part): return np.concatenate((part["k"],part["s"]),axis=-1).astype(np.float32)
        fit_q=embed(fit)
        q={"fit":fit_q,"val":embed(val),"D1":embed(d1),"D2":embed(test),"D3":embed(test)}
        limits=v2.derive_length_limits(fit)
        segments,row_segment=v2.make_segments(fit,fit_q,limits,.5)
        # v2's segmenter supplies the exact same gripper-boundary protocol;
        # its key field is replaced with the language-free scene/proprio key.
        for seg in segments: seg["key"]=fit_q[int(seg["rows"][0])].copy()
        eval_rows={"D1":d1,"D2":test,"D3":test}
        eval_segments={k:v2.make_segments(r,q[k],limits,.5)[0] for k,r in eval_rows.items()}
        for cond,segs in eval_segments.items():
          for seg in segs: seg["key"]=q[cond][int(seg["rows"][0])].copy()
        # Train the ordinary head, then initialize skill heads from it (v2.1 recipe).
        dim=int(np.prod(actions.shape[1:])); zeros=np.zeros((len(actions),dim),np.float32)
        normal_sources=(zeros,np.zeros(len(actions),np.float32),zeros.copy(),np.zeros(len(actions),np.float32))
        f_normal,norm_normal,_=af.fit_anchor_flow(fit,actions,normal_sources,source_mix=fc.NORMAL_MIX,
            seed=seed+827,epochs=12 if smoke else 80,patience=4 if smoke else 8,batch_size=512,device=dev)
        rng=np.random.default_rng(seed*1_000_003+31)
        eps={"val":rng.standard_normal((fc.SAMPLES,len(val["a"]),dim)).astype(np.float32),
             "D1":rng.standard_normal((fc.SAMPLES,len(d1["a"]),dim)).astype(np.float32),
             "D3":rng.standard_normal((fc.SAMPLES,len(test["a"]),dim)).astype(np.float32)}
        # V2.1 comparison action model is trained once for this cell; custom banks
        # use the identical initialization, source mixture and evaluation helpers.
        variant_results={}
        bank_models={}
        for variant in ("V3","V4"):
          if variant=="V3":
            model=CompetitiveExperts(fit["k"].shape[1],fit["s"].shape[1],actions.shape[-1],
                q=.95,min_support=10,min_episodes=3,max_rounds=3,device=dev).fit(fit,segments,seed=seed+901,epochs=8 if smoke else 18)
            model.fit_router(fit,segments,seed=seed+991,epochs=8 if smoke else 40)
            unit_count=len(model.experts)
            # Training router scores define the identity space; units carry mean
            # resampled training curves and scene-key prototypes for shared v2 tools.
            ids=model.assignments
            units=[]
            for uid in range(unit_count):
              members=[segments[i] for i in range(len(segments)) if ids[i]==uid]
              if not members: continue
              curve=np.mean([s["curve"] for s in members],axis=0)
              u=v2.SkillUnit(uid,curve,curve.copy(),np.var([s["curve"] for s in members],axis=0),
                    float(np.median([s["length"] for s in members])),np.mean([s["key"] for s in members],axis=0).astype(np.float32))
              u.members=members;u.support=len(members);u.tasks={s["task_index"] for s in members};u.episodes={s["episode_id"] for s in members};u.classes=[s.get("skill_class",-1) for s in members]
              u.mean_rep=v2._rep(curve,u.median_length);units.append(u)
            memory=v2.TrajectorySkillMemory(1.) ; memory.units=units
            memory.assignments={int(s["segment_id"]):int(ids[i]) for i,s in enumerate(segments) if int(ids[i])<len(units)}
            # Hard-EM router and expert action prediction are retained for routing
            # diagnostics; flow anchors are generated from the expert skill curves.
            model._memory=memory
          else:
            canonical=[]; geometry=[]
            for seg in segments:
              curve,direction,distance=canonicalize_v4(seg["curve"])
              ns=dict(seg,curve=curve); canonical.append(ns); geometry.append((direction,distance))
            memory=v2.TrajectorySkillMemory(1.,kappa=.1,min_support=10,min_episodes=3).fit(canonical,seed=seed+811)
            ids=np.asarray([memory.assignments.get(int(s["segment_id"]),-1) for s in canonical],np.int64)
            # Scene->direction/distance ridge regression is fitted only on train
            # segments. Geometric labels are never used as task/skill labels.
            model=fit_v4_geometry_regressor(segments,geometry,ridge=2.)
          variant_results[variant]={"n_skills":len(memory.units),"memory":memory,"ids":ids,
              "model":model,"segments":segments if variant=="V3" else canonical,
              "geometry":geometry if variant=="V4" else None}
        # Router-derived training anchors for the v2.1 flow head.
        for variant,cell in variant_results.items():
          memory=cell["memory"]; units=memory.units; ids=cell["ids"]
          if not units: raise RuntimeError(f"{split}/{seed}/{variant}: no registered skills")
          sources=np.zeros((5,len(fit["a"]),*actions.shape[1:]),np.float32); conf=np.full((5,len(fit["a"])),.5,np.float32)
          rng_src=np.random.default_rng(seed+1053)
          for si,seg in enumerate(cell["segments"]):
            uid=int(ids[si])
            if uid<0 or uid>=len(units): uid=0
            span=seg["rows"];unit=units[uid]
            if variant=="V4":
              dir_hat,dist_hat=predict_v4_geometry(cell["model"],fit_q[span])
            for local,row in enumerate(span):
              phi=local/max(len(span)-1,1)
              if variant=="V3":
                phase=torch.full((1,),phi,dtype=torch.float32,device=dev)
                scene=torch.as_tensor(fit["k"][row:row+1],dtype=torch.float32,device=dev)
                state=torch.as_tensor(fit["s"][row:row+1],dtype=torch.float32,device=dev)
                with torch.no_grad():
                  good=cell["model"].experts[uid](scene,state,phase).cpu().numpy()[0]
                  shifted=cell["model"].experts[uid](scene,state,torch.tensor([np.clip(phi+.2,0,1)],dtype=torch.float32,device=dev)).cpu().numpy()[0]
                sources[0,row]=np.repeat(good[None],v2.HORIZON,axis=0)
                sources[1,row]=np.repeat(shifted[None],v2.HORIZON,axis=0)
              else:
                canonical_shape=unit.mean
                restored=restore_v4_geometry(canonical_shape,dir_hat[local],dist_hat[local])
                sources[0,row]=v2._trajectory_chunk(restored,phi,unit.median_length,.5)
                sources[1,row]=v2._trajectory_chunk(restored,float(np.clip(phi+.2,0,1)),unit.median_length,.5)
              other=[u for u in units if u.skill_id!=uid]
              near=other[0] if other else unit; random_u=units[int(rng_src.integers(len(units)))]
              if variant=="V4":
                sources[2,row]=v2._trajectory_chunk(restore_v4_geometry(near.mean,dir_hat[local],dist_hat[local]),phi,near.median_length,.5)
                sources[3,row]=v2._trajectory_chunk(restore_v4_geometry(random_u.mean,dir_hat[local],dist_hat[local]),phi,random_u.median_length,.5)
              else:
                sources[2,row]=v2._trajectory_chunk(near.mean,phi,near.median_length,.5)
                sources[3,row]=v2._trajectory_chunk(random_u.mean,phi,random_u.median_length,.5)
          mix=v2._normalise_mix(v2.V21_SOURCE_MIX)
          flow,norm,fit_flow=v2._fit_skill_flow(fit,actions,sources,conf,mix,seed=seed+907,epochs=12 if smoke else 80,
              patience=4 if smoke else 8,batch_size=512,device=dev,initial_model=f_normal,initial_norm=norm_normal,
              teacher_model=f_normal,teacher_weight=2.)
          # Validation schedule uses training-only skill similarity for its tau map.
          val_segs,_=v2.make_segments(val,q["val"],limits,.5)
          for seg in val_segs: seg["key"]=q["val"][int(seg["rows"][0])].copy()
          if variant=="V3": val_sched=_v3_schedule(val,val_segs,cell["model"],device=dev)
          else: val_sched=_v4_schedule(val,q["val"],val_segs,memory,cell["model"],stride=.5)
          calibration=v2._calibrate_tau(flow,norm,val,val_sched,eps["val"],device=dev)
          cell["tau_calibration"]=calibration
          for cond in ("D1","D2","D3"):
            erows=eval_rows[cond]; ese=eval_segments[cond]; equery=q[cond]; enoise=eps["D1"] if cond=="D1" else eps["D3"]
            if cond=="D2":
              pred,sched,_stream=_prequential_d2(variant,erows,equery,ese,memory,cell["model"],flow,norm,
                  calibration,enoise,seed=seed,device=dev)
              tau,steps=v2.apply_confidence_map(calibration,sched["confidence"])
              mse=v2._metrics(pred,erows["a"][:,:v2.HORIZON])
              normal=v2._predict(f_normal,norm_normal,erows,np.zeros_like(sched["anchor"]),enoise,
                                  np.ones(len(erows["a"]),np.float32),np.full(len(erows["a"]),10),dev)
              mse["D1_worse_fraction"]=float(np.mean(mse["per_row_mse"]>v2._metrics(normal,erows["a"][:,:v2.HORIZON])["per_row_mse"]))
              cell[cond]={"mse":mse["mse"],"D1_worse_fraction":mse["D1_worse_fraction"],"selected_ids":sched["skill_id"],"confidence":sched["confidence"],"prediction":pred.mean(0),"normal":normal.mean(0),"schedule":sched}
              continue
            if variant=="V3":
              sched=_v3_schedule(erows,ese,cell["model"],device=dev)
            else:
              sched=_v4_schedule(erows,equery,ese,memory,cell["model"],stride=.5)
            tau,n=v2.apply_confidence_map(calibration,sched["confidence"])
            pred=v2._predict(flow,norm,erows,sched["anchor"],enoise,tau,n,dev)
            mse=v2._metrics(pred,erows["a"][:,:v2.HORIZON])
            normal=v2._predict(f_normal,norm_normal,erows,np.zeros_like(sched["anchor"]),enoise,
                                np.ones(len(erows["a"]),np.float32),np.full(len(erows["a"]),10),dev)
            mse["D1_worse_fraction"]=float(np.mean(mse["per_row_mse"]>v2._metrics(normal,erows["a"][:,:v2.HORIZON])["per_row_mse"]))
            cell[cond]={"mse":mse["mse"],"D1_worse_fraction":mse["D1_worse_fraction"],
                        "selected_ids":sched["skill_id"],"confidence":sched["confidence"],
                        "prediction":pred.mean(0),"normal":normal.mean(0),"schedule":sched}
          cell["fit"]={"best_epoch":fit_flow["best_epoch"],"initial_state_equal":fit_flow["initial_state_equal"]}
          cell["flow_model"]=flow; cell["flow_norm"]=norm
          bank_models[variant]=cell
        # Post-hoc statistics and b intervention.
        for variant,cell in bank_models.items():
          memory=cell["memory"]
          def acc(cond):
            erows=eval_rows[cond];ese=eval_segments[cond];qv=q[cond]
            if variant=="V3":
              # Truth assignment is the lowest expert reconstruction error on
              # that labelled evaluation trajectory; query selection itself is
              # scene/state-only through the router.
              data=_segment_model_data(erows,ese,device=dev)
              truth=np.stack([_decoder_losses(m,data) for m in cell["model"].experts],axis=1).argmin(1)
              scene=torch.as_tensor(erows["k"],device=dev);state=torch.as_tensor(erows["s"],device=dev)
              phase=np.zeros(len(erows["a"]),np.float32)
              for seg in ese: phase[seg["rows"]]=_phase_rows(seg)
              with torch.no_grad(): scores=cell["model"].router(torch.cat((scene,state,torch.as_tensor(phase,device=dev)[:,None]),dim=-1)).cpu().numpy()
              truth_rows=np.full(len(erows["a"]),-1,np.int64)
              for sid,seg in enumerate(ese): truth_rows[seg["rows"]]=truth[sid]
            else:
              normalized=[]
              for seg in ese: normalized.append(canonicalize_v4(seg["curve"])[0])
              truth=np.asarray([int(np.argmin([np.mean((x-u.mean)**2) for u in memory.units])) for x in normalized])
              qn=qv/np.maximum(np.linalg.norm(qv,axis=1,keepdims=True),1e-8)
              kn=np.stack([u.key for u in memory.units]);kn/=np.maximum(np.linalg.norm(kn,axis=1,keepdims=True),1e-8)
              scores=qn@kn.T
              truth_rows=np.full(len(erows["a"]),-1,np.int64)
              for sid,seg in enumerate(ese): truth_rows[seg["rows"]]=truth[sid]
            # Exact selection metrics count each segment once at its first row.
            predtop=np.argsort(-scores,axis=1)
            if cond=="D2" and "top_ids" in cell[cond]["schedule"]:
              predtop=cell[cond]["schedule"]["top_ids"]
            ids=np.asarray([int(seg["rows"][0]) for seg in ese if len(seg["rows"])])
            tr=np.asarray([truth_rows[i] for i in ids])
            return float(np.mean(predtop[ids,0]==tr)),float(np.mean([t in row[:5] for t,row in zip(tr,predtop[ids])]))
          d1acc=acc("D1");d3acc=acc("D3")
          # Skill count and post-hoc task/verb purity.
          taskcounts=[len(u.tasks) for u in memory.units]
          lengths=[u.median_length*.5 for u in memory.units]
          # Plan phrases and task labels are joined only after all fitting and
          # evaluation for this variant have finished.
          from . import clip_retrieval as cr
          fit_posthoc=cr.attach_subtask_texts(fit,cache_dir=DEFAULT_CACHE,
                                               annotations_path=cr.DEFAULT_ANNOTATIONS)
          verb_purity=[]; verb_histograms={}
          for u in memory.units:
            members=getattr(u,"members",[])
            verbs=[_verb_from_plan(fit_posthoc["subtask_text"][int(m["rows"][0])])
                   for m in members if len(m.get("rows",[]))]
            if verbs:
              values,counts=np.unique(verbs,return_counts=True)
              hist={str(k):int(v) for k,v in zip(values,counts)}
              verb_histograms[str(u.skill_id)]=hist
              verb_purity.append(float(counts.max()/counts.sum()))
          # Swap skill IDs while keeping scene/state/phase fixed; action change
          # is measured directly at anchor time, before flow correction.
          d3=eval_rows["D3"];sched3=cell["D3"]["schedule"]
          if variant=="V3":
            phase=np.zeros(len(d3["a"]),np.float32)
            for seg in eval_segments["D3"]: phase[seg["rows"]]=_phase_rows(seg)
            sc=torch.as_tensor(d3["k"],dtype=torch.float32,device=dev); st=torch.as_tensor(d3["s"],dtype=torch.float32,device=dev)
            ph=torch.as_tensor(phase,dtype=torch.float32,device=dev)[:,None]
            with torch.no_grad():
              epred=np.stack([m(sc,st,ph).cpu().numpy() for m in cell["model"].experts],axis=1)
            rank=np.argsort(-sched3["scores"],axis=1)
            alt=rank[:,1] if epred.shape[1]>1 else rank[:,0]
            base=rank[:,0]
            bdelta=float(np.mean(np.abs(epred[np.arange(len(base)),base]-epred[np.arange(len(base)),alt])))
          else:
            reg=cell["model"];dirs,dists=predict_v4_geometry(reg,q["D3"])
            scores=sched3["scores"]; rank=np.argsort(-scores,axis=1); base=rank[:,0]; alt=rank[:,1] if scores.shape[1]>1 else base
            deltas=[]
            for i,(buid,auid) in enumerate(zip(base,alt)):
              bcurve=restore_v4_geometry(memory.units[int(buid)].mean,dirs[i],dists[i])
              acurve=restore_v4_geometry(memory.units[int(auid)].mean,dirs[i],dists[i])
              deltas.append(np.mean(np.abs(v2._trajectory_chunk(bcurve,0,memory.units[int(buid)].median_length,.5)-v2._trajectory_chunk(acurve,0,memory.units[int(auid)].median_length,.5))))
            bdelta=float(np.mean(deltas))
          cell["summary"]={"n_skills":len(memory.units),"median_length_seconds":float(np.median(lengths)) if lengths else 0.,
            "median_tasks_per_skill":float(np.median(taskcounts)) if taskcounts else 0.,"verb_purity":float(np.mean(verb_purity)) if verb_purity else 0.,
            "D1_top1":d1acc[0],"D1_top5":d1acc[1],"D3_top1":d3acc[0],"D3_top5":d3acc[1],
            "D1_MSE":cell["D1"]["mse"],"D2_MSE":cell["D2"]["mse"],"D3_MSE":cell["D3"]["mse"],"D1_worse_fraction":cell["D1"]["D1_worse_fraction"],
            "b_swap_action_delta":bdelta,"verb_counts_by_unit":verb_histograms,
            "formation":( {"n_skills":len(memory.units),"births":cell["model"].births,
                            "merges":cell["model"].merges,"error_q95_by_round":cell["model"].thresholds,
                            "support":[u.support for u in memory.units]} if variant=="V3" else memory.describe())}
        # Keep row-level arrays out of JSON; optional checkpoint stores trained experts and flow models.
        checkpoint={}
        for variant,cell in bank_models.items():
          summary=cell["summary"]
          summary["flow_fit"]=cell["fit"]
          variant_results[variant].update(summary)
          # Remove runtime-only fields from serializable records below.
          if variant=="V3":
            checkpoint[variant]={"experts":[{k:v.cpu() for k,v in x.state_dict().items()} for x in cell["model"].experts],
                                 "router":{k:v.cpu() for k,v in cell["model"].router.state_dict().items()},
                                 "flow":{k:v.detach().cpu() for k,v in cell["flow_model"].state_dict().items()},
                                 "flow_norm":cell["flow_norm"],"tau_calibration":cell["tau_calibration"],
                                 "unit_means":np.stack([u.mean for u in cell["memory"].units]),
                                 "unit_keys":np.stack([u.key for u in cell["memory"].units]),
                                 "unit_lengths":np.asarray([u.median_length for u in cell["memory"].units])}
          else:
            checkpoint[variant]={"flow":{k:v.detach().cpu() for k,v in cell["flow_model"].state_dict().items()},
                                 "flow_norm":cell["flow_norm"],"geometry_regressor":cell["model"],
                                 "unit_means":np.stack([u.mean for u in cell["memory"].units]),
                                 "unit_keys":np.stack([u.key for u in cell["memory"].units]),
                                 "unit_lengths":np.asarray([u.median_length for u in cell["memory"].units]),
                                 "tau_calibration":cell["tau_calibration"]}
        torch.save(checkpoint,output/"checkpoints"/f"{split}_s{seed}.pt")
        baseline_path=Path(str(RESEARCH_STATE_DIR / 'outputs/memory-pilot/skill-v21-20261029T060213Z-7493c4/results.json'))
        base=json.loads(baseline_path.read_text())
        base_run=next(x for x in base["runs"] if x["split"]==split and x["seed"]==seed)
        baselines={k:base_run["conditions"]["D3"]["mean"][k]["mse"] for k in ("normal","raw_interval","primitive_Pa","skill_v2")}
        serial={}
        for v,x in bank_models.items(): serial[v]=x["summary"]
        all_outputs.append({"split":split,"seed":seed,"variants":serial,"baselines_v21_D3":baselines,
                            "segment_count":len(segments),"fit_rows":len(fit["a"]),"D1_rows":len(d1["a"]),"D3_rows":len(test["a"]),
                            "D2_evaluation_mode":"prequential_episode_append_no_gradient","duration_seconds":time.perf_counter()-tic})
        payload={"schema_version":"skill-v3-competition-v1","config":{"splits":list(splits),"seeds":list(seeds),"smoke":smoke,"device":device,"threads":threads,"v2_flow_source_mix":list(v2.V21_SOURCE_MIX)},"runs":all_outputs,"elapsed_seconds":time.perf_counter()-start}
        (output/"results.partial.json").write_text(json.dumps(payload,indent=2,ensure_ascii=False,default=str)+"\n")
    payload={"schema_version":"skill-v3-competition-v1","config":{"splits":list(splits),"seeds":list(seeds),"smoke":smoke,"device":device,"threads":threads,"v2_flow_source_mix":list(v2.V21_SOURCE_MIX)},"runs":all_outputs,"elapsed_seconds":time.perf_counter()-start}
    (output/"results.json").write_text(json.dumps(payload,indent=2,ensure_ascii=False,default=str)+"\n")
    _summary(payload,output)
    return output


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--splits",default="A,B");p.add_argument("--seeds",default="0,1,2")
    p.add_argument("--device",default="cuda");p.add_argument("--threads",type=int,default=8)
    p.add_argument("--smoke",action="store_true");p.add_argument("--max-queries",type=int)
    a=p.parse_args(argv);print(run(output=a.output,splits=tuple(a.splits.split(',')),seeds=tuple(map(int,a.seeds.split(','))),device=a.device,threads=a.threads,smoke=a.smoke,max_queries=a.max_queries))


if __name__=="__main__": main()
