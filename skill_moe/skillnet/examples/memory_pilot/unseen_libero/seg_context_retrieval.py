#!/usr/bin/env python3
"""Context-conditioned retrieval of variable-length LIBERO motion segments.

This module is implementation-only infrastructure. It does not launch training,
feature extraction, or evaluation on import. Prefix tokens are supplied by the
caller (final pi05 prefix, all valid base/wrist/instruction tokens); a single
shared encoder is used for memory keys and live queries.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HELDOUT_TASKS=tuple(range(2,38,5))


def prefix_token_mask(groups):
    """Use cached modality ids: base/wrist/instruction >=0; padding/special -1."""
    return np.asarray(groups)>=0


def training_row_mask(task_ids, memory_mask, *, heldout_tasks=HELDOUT_TASKS):
    """LIBERO-40 train/memory eligibility; held-out tasks never leak into either."""
    task=np.asarray(task_ids);memory=np.asarray(memory_mask,bool)
    if task.shape!=memory.shape:raise ValueError("task/memory mask shape mismatch")
    return (~memory)&(~np.isin(task,np.asarray(heldout_tasks)))


def memory_row_mask(task_ids,memory_mask,*,heldout_tasks=HELDOUT_TASKS):
    """Memory contains designated memory rows, excluding the fixed holdout tasks."""
    task=np.asarray(task_ids);memory=np.asarray(memory_mask,bool)
    if task.shape!=memory.shape:raise ValueError("task/memory mask shape mismatch")
    return memory&(~np.isin(task,np.asarray(heldout_tasks)))


def sample_query_row_ids(task,episode,frame,eligible,*,stride=5,include_unit_starts=()):
    """Select both segment starts and stride-five query ticks from eligible rows."""
    task=np.asarray(task);episode=np.asarray(episode);frame=np.asarray(frame);eligible=np.asarray(eligible,bool)
    if not(len(task)==len(episode)==len(frame)==len(eligible)):raise ValueError("row metadata length mismatch")
    wanted=set(tuple(map(int,x)) for x in include_unit_starts)
    wanted.update((int(t),int(e),int(f)) for t,e,f in zip(task[eligible],episode[eligible],frame[eligible])
                  if int(f)%int(stride)==0)
    # Return only source-cache row IDs; missing intermediate frames are emitted
    # by seg_context_extract rather than silently mapped to a neighbouring row.
    row_map={(int(t),int(e),int(f)):i for i,(t,e,f) in enumerate(zip(task,episode,frame))}
    return np.asarray([row_map[k] for k in sorted(wanted) if k in row_map and eligible[row_map[k]]],np.int64)


@dataclass(frozen=True)
class SegmentConfig:
    token_dim: int = 2048
    width: int = 256
    heads: int = 4
    cross_layers: int = 1
    max_history: int = 40
    dct_coefficients: int = 3
    exclude_same_task: bool = True
    continuity_weight: float | str = 0.0
    boundary_blend: bool = False
    loop_exclude_previous_unit: bool = False
    use_history: bool = True
    use_pose_scale_head: bool = True
    value_head_mode: str = "dct"  # dct or split (scene_invariance_v2 action-key encoder)

    def __post_init__(self):
        if self.value_head_mode not in ('dct','split'):raise ValueError(self.value_head_mode)


def _physical6(values):
    import sys
    from pathlib import Path
    here=Path(__file__).resolve().parent
    if str(here) not in sys.path:sys.path.insert(0,str(here))
    import variable_shape_memory as vs
    return np.asarray(vs.to_physical(np.asarray(values)[...,:6]),np.float64)


def _normalized6(values):
    import sys
    from pathlib import Path
    here=Path(__file__).resolve().parent
    if str(here) not in sys.path:sys.path.insert(0,str(here))
    import variable_shape_memory as vs
    return np.asarray(vs.to_normalized(np.asarray(values)[...,:6]),np.float32)


def _motion_speed(actions: np.ndarray) -> np.ndarray:
    a = np.asarray(actions, np.float32)
    if a.ndim != 2 or a.shape[-1] < 6:
        raise ValueError("actions must be [T,>=6]")
    return np.linalg.norm(np.diff(a[:, :3], axis=0, prepend=a[:1, :3]), axis=-1)


def causal_completed_segments(actions: np.ndarray, states: np.ndarray | None = None, *,
                              min_length: int = 10, max_history: int = 40) -> list[tuple[int, int]]:
    """Return completed spans detectable from this prefix only; end is exclusive.

    Boundaries are confirmed by a gripper transition or a velocity minimum whose
    right neighbour has already arrived. The unfinished suffix is intentionally
    omitted, so a query cannot use future actions to construct its history.
    """
    a = np.asarray(actions, np.float32)
    if a.ndim != 2 or a.shape[1] < 7:
        raise ValueError("actions must have shape [T,7]")
    if states is not None and len(states) != len(a):
        raise ValueError("actions/states length mismatch")
    n = len(a)
    if n < min_length + 1:return []
    # Reuse the existing speed-minimum/gripper-switch cut implementation on
    # the observed prefix only. Its final returned interval is the unfinished
    # suffix and is therefore never admitted as history.
    import sys
    from pathlib import Path
    here=Path(__file__).resolve().parent
    if str(here) not in sys.path:sys.path.insert(0,str(here))
    import motion_core_eval as motion
    spans=motion.anchor_spans(a,min_length=min_length)
    if spans:spans=spans[:-1]
    # Keep only recent history, retaining true segment starts where possible.
    if spans:
        cutoff = max(0, n - max_history)
        spans = [(max(s, cutoff), min(e, n)) for s, e in spans if e > cutoff]
        spans = [(s, e) for s, e in spans if e - s >= min_length]
    return spans


def causal_history_at(actions,states,query_step,*,min_length=10,max_history=40):
    """Build the history available at ``query_step`` from prefix actions only."""
    a=np.asarray(actions,np.float32);s=np.asarray(states,np.float32)
    q=int(query_step)
    if not 0<=q<=len(a) or len(a)!=len(s):raise ValueError("invalid causal query index")
    return latest_completed_history(a[:q],s[:q],min_length=min_length,max_history=max_history)


def latest_completed_history(actions: np.ndarray, states: np.ndarray, *,
                             min_length: int = 10, max_history: int = 40):
    """Return the most recent complete segment's paired action/state history."""
    a, s = np.asarray(actions, np.float32), np.asarray(states, np.float32)
    spans = causal_completed_segments(a, s, min_length=min_length, max_history=max_history)
    if not spans:
        return None
    start, end = spans[-1]
    return a[start:end][-max_history:], s[start:end][-max_history:], (start, end)


def pack_histories(histories, *, max_length=40):
    """Pad variable histories with an explicit mask; padding never enters GRU state."""
    if not histories:raise ValueError("at least one history is required")
    actions=np.zeros((len(histories),max_length,7),np.float32)
    states=np.zeros((len(histories),max_length,8),np.float32)
    mask=np.zeros((len(histories),max_length),bool)
    for i,(a,s) in enumerate(histories):
        a=np.asarray(a,np.float32)[-max_length:];s=np.asarray(s,np.float32)[-max_length:]
        if a.ndim!=2 or a.shape[1]<7 or s.ndim!=2 or s.shape[1]<8 or len(a)!=len(s):
            raise ValueError("each history must align actions [T,>=7] with states [T,>=8]")
        n=len(a);actions[i,:n]=a[:,:7];states[i,:n]=s[:,:8];mask[i,:n]=True
    return actions,states,mask


def stride_query_frames(frame_count: int, *, stride=5, include_zero=True):
    """Frame offsets used for both unit starts and periodic retrieval queries."""
    if frame_count < 0 or stride < 1: raise ValueError("invalid frame count/stride")
    start=0 if include_zero else stride
    return np.arange(start,frame_count,stride,dtype=np.int32)


def episode_exclusion_mask(query_task, query_episode, memory_task, memory_episode, *,
                           same_task=True, previous_unit=None, memory_unit=None,
                           exclude_previous_unit=False):
    """NumPy counterpart of the retrieval mask for data preparation and tests."""
    qt,qe,mt,me=map(np.asarray,(query_task,query_episode,memory_task,memory_episode))
    mask=(qe[:,None]!=me[None,:])
    if same_task:mask &= qt[:,None]!=mt[None,:]
    if exclude_previous_unit:
        if previous_unit is None or memory_unit is None:raise ValueError("unit IDs required")
        mask &= np.asarray(previous_unit)[:,None]!=np.asarray(memory_unit)[None,:]
    return mask


def corrupt_history(actions: torch.Tensor, states: torch.Tensor, *, generator=None,
                    probabilities=(.25, .25, .15, .35)):
    """Batch history corruption: zero, other, translation-reversed, correct."""
    if actions.ndim != 3 or states.ndim != 3 or len(actions) != len(states):
        raise ValueError("history tensors must be [B,T,D]")
    p = torch.as_tensor(probabilities, device=actions.device, dtype=torch.float32)
    if len(probabilities) != 4 or not torch.isclose(p.sum(), p.new_tensor(1.0)):
        raise ValueError("corruption probabilities must sum to one")
    modes = torch.multinomial(p.expand(len(actions), -1), 1, generator=generator).squeeze(1)
    out_a, out_s = actions.clone(), states.clone()
    zero = modes == 0
    out_a[zero] = 0
    out_s[zero] = 0
    other = modes == 1
    if other.any() and len(actions) > 1:
        shift = torch.randint(1, len(actions), (int(other.sum()),), device=actions.device,
                              generator=generator)
        src = (torch.arange(len(actions), device=actions.device)[other] + shift) % len(actions)
        out_a[other], out_s[other] = actions[src], states[src]
    reverse = modes == 2
    if reverse.any():
        out_a[reverse, :, :3] *= -1
        out_s[reverse, :, :3] *= -1
    return out_a, out_s, modes


class SharedContextEncoder(nn.Module):
    """Shared key/query encoder: prefix tokens cross-attended by state/history."""
    def __init__(self, config: SegmentConfig):
        super().__init__()
        self.config = config
        d = config.width
        self.token_proj = nn.Linear(config.token_dim, d)
        self.state_proj = nn.Linear(8, d)
        self.action_proj = nn.Linear(7, d)
        self.history_gru = nn.GRU(d * 2, d, batch_first=True)
        self.context_proj = nn.Sequential(nn.Linear(d + d, d), nn.GELU(), nn.LayerNorm(d))
        self.cross = nn.ModuleList([
            nn.MultiheadAttention(d, config.heads, batch_first=True) for _ in range(config.cross_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in self.cross])
        self.ffns = nn.ModuleList([nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))
                                   for _ in self.cross])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(d) for _ in self.cross])
        self.out = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d))
        self.logit_scale = nn.Parameter(torch.tensor(np.log(10.), dtype=torch.float32))
        self.learned_continuity_scale = nn.Parameter(torch.tensor(0., dtype=torch.float32))

    def context(self, history_actions, history_states, state, history_mask=None):
        if not self.config.use_history:
            h = torch.zeros((len(state), self.config.width), device=state.device, dtype=state.dtype)
        else:
            x = torch.cat([self.action_proj(history_actions), self.state_proj(history_states[..., :8])], -1)
            if history_mask is not None:
                lengths=history_mask.bool().sum(1)
                packed=nn.utils.rnn.pack_padded_sequence(x,lengths.clamp_min(1).cpu(),batch_first=True,enforce_sorted=False)
                _,hidden=self.history_gru(packed);h=hidden[-1]
                h=h*(lengths>0).to(h.dtype)[:,None]
            else:
                _, hidden = self.history_gru(x);h = hidden[-1]
        return self.context_proj(torch.cat([h, self.state_proj(state[..., :8])], -1))

    def forward(self, tokens, token_mask, history_actions, history_states, state, history_mask=None):
        if tokens.ndim != 3 or token_mask.shape != tokens.shape[:2]:
            raise ValueError("tokens/mask must be [B,T,D]/[B,T]")
        memory = self.token_proj(tokens)
        q = self.context(history_actions, history_states, state,history_mask)[:, None, :]
        for attn, norm, ffn, ffn_norm in zip(self.cross, self.norms, self.ffns, self.ffn_norms):
            read, _ = attn(q, memory, memory, key_padding_mask=~token_mask.bool(), need_weights=False)
            q = norm(q + read)
            q = ffn_norm(q + ffn(q))
        return F.normalize(self.out(q[:, 0]), dim=-1)


@torch.no_grad()
def encode_context_batch(encoder, tokens, token_mask, history_actions, history_states, state,
                         *, batch_size=64, device=None):
    """Encode row/key/query arrays with the same module and preserve source order."""
    n=len(tokens);device=device or next(encoder.parameters()).device
    chunks=[];encoder.eval()
    for start in range(0,n,int(batch_size)):
        end=min(n,start+int(batch_size))
        args=[torch.as_tensor(x[start:end],device=device) for x in
              (tokens,token_mask,history_actions,history_states,state)]
        chunks.append(encoder(*args).cpu().numpy())
    return np.concatenate(chunks,axis=0) if chunks else np.empty((0,encoder.config.width),np.float32)


class PoseScaleHead(nn.Module):
    """Predict an axis-angle correction initialized to identity and log magnitude."""
    def __init__(self, width=256, coeff_dim=21):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width * 2 + coeff_dim, width), nn.GELU(),
                                 nn.Linear(width, width // 2), nn.GELU(), nn.Linear(width // 2, 4))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, query_key, memory_key, coefficients):
        return self.net(torch.cat([query_key, memory_key, coefficients.flatten(1)], -1))


class SegmentContextModel(nn.Module):
    """Trainable container for the shared encoder and pose/size prediction head."""
    def __init__(self, config: SegmentConfig):
        super().__init__()
        self.encoder=SharedContextEncoder(config)
        value_dim=config.dct_coefficients*6
        self.action_value_encoder=None
        if config.value_head_mode=='split':
            import sys
            from pathlib import Path
            here=Path(__file__).resolve().parent
            if str(here) not in sys.path:sys.path.insert(0,str(here))
            import scene_invariance_v2 as inv2
            self.action_value_encoder=inv2._build_model('A',k=5,key_mode='split')
            value_dim=5*256
        self.pose_head=PoseScaleHead(config.width,value_dim)

    def encode_value(self,actions10,coefficients):
        if self.action_value_encoder is None:return coefficients.flatten(1)
        return self.action_value_encoder.encode_keys(actions10).flatten(1)


def fit_pose_scale(shape: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Least-squares SO(3) + shared scale target [axis-angle xyz, log-scale]."""
    from scipy.spatial.transform import Rotation
    x, y = _physical6(np.asarray(target,np.float64)), _physical6(np.asarray(shape,np.float64))
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] < 6:
        raise ValueError("shape and target must both be [T,>=6]")
    cross = y[:, :3].T @ x[:, :3] + y[:, 3:6].T @ x[:, 3:6]
    u, _, vt = np.linalg.svd(cross, full_matrices=False)
    corr = np.eye(3); corr[-1, -1] = np.linalg.det(u @ vt)
    rot = u @ corr @ vt
    aligned = np.concatenate([y[:, :3] @ rot, y[:, 3:6] @ rot], axis=-1)
    scale = max(float((aligned * x[:, :6]).sum() / max((y[:, :6] ** 2).sum(), 1e-12)), 1e-6)
    # Return the column-vector convention consumed by offline_draft, which applies
    # row actions as ``action @ R.T``.
    return np.r_[Rotation.from_matrix(rot.T).as_rotvec(), np.log(scale)].astype(np.float32)


def pose_scale_loss(prediction: torch.Tensor, target: torch.Tensor):
    """Geodesic rotation loss plus squared log-scale error."""
    def matrix(v):
        theta=torch.linalg.vector_norm(v,dim=-1,keepdim=True).clamp_min(1e-8)
        axis=v/theta
        x,y,z=axis.unbind(-1);zero=torch.zeros_like(x);one=torch.ones_like(x)
        k=torch.stack([zero,-z,y,z,zero,-x,-y,x,zero],-1).reshape(*v.shape[:-1],3,3)
        eye=torch.eye(3,device=v.device,dtype=v.dtype).expand(*v.shape[:-1],3,3)
        return eye+torch.sin(theta)[...,None]*k+(1-torch.cos(theta))[...,None]*(k@k)
    rel=matrix(prediction[...,:3])@matrix(target[...,:3]).transpose(-1,-2)
    cosine=((rel.diagonal(dim1=-2,dim2=-1).sum(-1)-1)*.5).clamp(-1+1e-7,1-1e-7)
    rot=torch.acos(cosine).square()
    scale = (prediction[..., 3] - target[..., 3]).square()
    return (rot + scale).mean()


def training_objective(encoder: SharedContextEncoder, pose_head: PoseScaleHead, batch: dict,
                       *, tau=.01, tau_action=.03, same_task_exclusion=True, pose_weight=1., generator=None):
    """One shared-weight contrastive/pose training step; labels are training-only."""
    ha,hs,modes=corrupt_history(batch['history_actions'],batch['history_states'],generator=generator)
    q=encoder(batch['query_tokens'],batch['query_token_mask'],ha,hs,batch['query_state'],batch.get('history_mask'))
    k=encoder(batch['memory_tokens'],batch['memory_token_mask'],
              batch['memory_history_actions'],batch['memory_history_states'],batch['memory_state'],
              batch.get('memory_history_mask'))
    scores=q@k.T*encoder.logit_scale.exp().clamp(max=100.)
    if 'continuity_features' in batch:
        if encoder.config.continuity_weight == 'learned':
            scale=F.softplus(encoder.learned_continuity_scale).clamp(max=10.)
        else:
            scale=float(encoder.config.continuity_weight)
        scores=scores+scale*batch['continuity_features']
    candidate=(batch['query_episode'][:,None]!=batch['memory_episode'][None,:])
    if same_task_exclusion:
        candidate &= batch['query_task'][:,None]!=batch['memory_task'][None,:]
    target=soft_action_targets(batch['type_error'],batch['action_distance'],candidate,
                               tau_candidates=(.01,.03),tau_action=tau_action,tau_type=tau)
    contrastive=soft_infonce(scores,target,candidate)
    value=batch.get('positive_value_features',batch['positive_coefficients'].flatten(1))
    if 'positive_memory_index' in batch:
        pkey=k[batch['positive_memory_index'].long()]
    else:
        pkey=batch['positive_memory_key']
    pred=pose_head(q,pkey,value)
    pose=pose_scale_loss(pred,batch['pose_target'])
    return {'loss':contrastive+float(pose_weight)*pose,'contrastive':contrastive,
            'pose':pose,'history_corruption':modes,'query_keys':q,'memory_keys':k}


def segment_value_encode(actions: np.ndarray, *, coefficients=3):
    """DCT-encode variable-length xyz+rotation after magnitude-only scaling.

    Returns coefficients [coefficients,6], magnitude scalar and the source
    duration. Direction/rotation remain in the normalized shape; gripper is
    intentionally omitted and is supplied by pi05 at generation time.
    """
    from scipy.fft import dct
    a = np.asarray(actions, np.float32)
    if a.ndim != 2 or a.shape[1] < 6 or len(a) < 1:
        raise ValueError("segment actions must be [T,>=6]")
    motion = _physical6(a[:, :6])
    magnitude = float(np.sqrt(np.mean(motion.astype(np.float64) ** 2)))
    magnitude = max(magnitude, 1e-6)
    shape = motion / magnitude
    coeff = dct(shape, type=2, norm="ortho", axis=0)[:coefficients]
    padded = np.zeros((coefficients, 6), np.float32)
    padded[:len(coeff)] = coeff
    return padded, magnitude, int(len(a))


def segment_value_decode(coefficients: np.ndarray, magnitude: float, length: int, *, out_length=10):
    """Decode DCT shape to 10 steps and restore its stored magnitude."""
    from scipy.fft import idct
    c = np.asarray(coefficients, np.float32)
    if c.ndim != 2 or c.shape[1] != 6 or length < 1 or out_length < 1:
        raise ValueError("invalid segment code")
    full = np.zeros((length, 6), np.float32)
    full[:min(length, len(c))] = c[:min(length, len(c))]
    seq = idct(full, type=2, norm="ortho", axis=0) * float(magnitude)
    src_t = np.linspace(0., 1., len(seq), dtype=np.float32)
    dst_t = np.linspace(0., 1., out_length, dtype=np.float32)
    return np.stack([np.interp(dst_t, src_t, seq[:, j]) for j in range(6)], -1).astype(np.float32)


def s_shared_type_error(query_segments, memory_segments, *, normalized_to_physical=None):
    """Pairwise S-shared type distance used by the common LIBERO scorer.

    Inputs are resampled [N,10,6] normalized actions. The shared scale is fit
    independently per pair, with no rotation, exactly as S-shared specifies.
    """
    q=np.asarray(query_segments,np.float32);m=np.asarray(memory_segments,np.float32)
    if q.ndim!=3 or m.ndim!=3 or q.shape[1:]!=m.shape[1:] or q.shape[-1]!=6:
        raise ValueError("expected query/memory arrays [N,10,6]")
    if normalized_to_physical is None:
        from pathlib import Path
        import sys
        here=Path(__file__).resolve().parent
        sys.path.insert(0,str(here))
        import variable_shape_memory as vs
        normalized_to_physical=vs.to_physical
    qp=np.asarray(normalized_to_physical(q),np.float64)
    mp=np.asarray(normalized_to_physical(m),np.float64)
    # S-shared removes one nonnegative scale from all six axes; retain the
    # per-frame normalized distance, matching scale_only_query.scale_only_error.
    dot=np.einsum('ntc,mtc->nm',qp,mp)
    norm=np.einsum('mtc,mtc->m',mp,mp)
    qnorm=np.einsum('ntc,ntc->n',qp,qp)
    residual=qnorm[:,None]-np.maximum(dot,0.)**2/np.maximum(norm[None],1e-12)
    return np.maximum(residual,0.)/(q.shape[1]*q.shape[2])


def future_action_mse(query_actions, memory_actions):
    """Pairwise raw active7 future-action MSE for soft positive weights."""
    q=np.asarray(query_actions,np.float32);m=np.asarray(memory_actions,np.float32)
    if q.ndim!=3 or m.ndim!=3 or q.shape[1:]!=m.shape[1:] or q.shape[-1]<7:
        raise ValueError("expected query/memory arrays [N,T,>=7]")
    return ((q[:,None,:,:7]-m[None,:,:,:7])**2).mean(axis=(2,3))


def encode_segment_bank(actions_by_episode: dict, *, min_length=10, max_history=40, coefficients=3):
    """Form a variable-size, non-fixed-K memory from causal boundary segments.

    ``actions_by_episode`` maps ``(task, episode)`` to full demonstration action
    traces. Output unit count is data-determined; no cluster count is specified.
    """
    out={k:[] for k in ('coefficients','magnitude','length','task','episode','start','source')}
    for (task,episode),trace in sorted(actions_by_episode.items()):
        a=np.asarray(trace,np.float32)
        spans=causal_completed_segments(a,min_length=min_length,max_history=max(len(a),max_history))
        # Full demos are available here; close the remaining suffix at episode end.
        consumed=spans[-1][1] if spans else 0
        tail=max(consumed,len(a)-max_history)
        if len(a)-tail>=min_length:spans.append((tail,len(a)))
        for start,end in spans:
            c,m,l=segment_value_encode(a[start:end],coefficients=coefficients)
            out['coefficients'].append(c);out['magnitude'].append(m);out['length'].append(l)
            out['task'].append(int(task));out['episode'].append(int(episode));out['start'].append(int(start))
            out['source'].append((int(task),int(episode),int(start),int(end)))
            source=np.asarray(a[start:end],np.float32)
            src_t=np.linspace(0.,1.,len(source),dtype=np.float32);dst_t=np.linspace(0.,1.,10,dtype=np.float32)
            out.setdefault('value_actions10',[]).append(np.stack([np.interp(dst_t,src_t,source[:,j])
                for j in range(7)],axis=-1).astype(np.float32))
    if out['coefficients']:
        out['coefficients']=np.asarray(out['coefficients'],np.float32)
        out['magnitude']=np.asarray(out['magnitude'],np.float32)
    else:
        out['coefficients']=np.empty((0,coefficients,6),np.float32);out['magnitude']=np.empty((0,),np.float32)
    out['length']=np.asarray(out['length'],np.int32);out['task']=np.asarray(out['task'],np.int32)
    out['episode']=np.asarray(out['episode'],np.int32);out['start']=np.asarray(out['start'],np.int32)
    out['source']=np.asarray(out['source'],np.int32).reshape(-1,4)
    out['value_actions10']=np.asarray(out.get('value_actions10',[]),np.float32).reshape(-1,10,7)
    return out


def build_memory_segments(rows, memory_row_ids, *, holdout_tasks=HELDOUT_TASKS,
                          min_length=10, coefficients=3):
    """Build raw-row memory units, excluding held-out tasks before segmentation.

    Episode action traces are stitched from stride-10 chunks exactly as the
    existing compressed-denoise memory bank does. ``source`` is task/episode/
    absolute start/end frame so the extraction manifest can request exact keys.
    """
    ids=np.asarray(memory_row_ids,np.int64)
    task=np.asarray(rows['task'],np.int32);episode=np.asarray(rows['episode'],np.int32)
    frame=np.asarray(rows['frame'],np.int32);actions=rows['actions']
    ids=ids[~np.isin(task[ids],np.asarray(holdout_tasks,np.int32))]
    grouped={}
    for rid in ids:grouped.setdefault((int(task[rid]),int(episode[rid])),[]).append(int(rid))
    banks=[]
    for key,rids in sorted(grouped.items()):
        rids=np.asarray(sorted(rids,key=lambda r:int(frame[r])),np.int64)
        fs=frame[rids]
        if len(fs)>1 and not np.array_equal(fs,fs[0]+10*np.arange(len(fs))):
            raise ValueError(f"memory episode has gaps/overlaps: {key}")
        trace=np.asarray(actions[rids,:,:7],np.float32).reshape(-1,7)
        b=encode_segment_bank({key:trace},min_length=min_length,coefficients=coefficients)
        if len(b['length']):
            b['source'][:,2]+=int(fs[0]);b['source'][:,3]+=int(fs[0])
            b['row_id']=np.asarray([rids[min(len(rids)-1,int(s)//10)] for s in b['start']],np.int64)
            b['start_velocity']=np.stack([_physical6(trace[int(s):int(s)+1,:6])[0,:3]
                                          for s in b['start']]).astype(np.float32)
            banks.append(b)
    if not banks:
        return {'coefficients':np.empty((0,coefficients,6),np.float32),'magnitude':np.empty(0,np.float32),
            'length':np.empty(0,np.int32),'task':np.empty(0,np.int32),'episode':np.empty(0,np.int32),
            'start':np.empty(0,np.int32),'source':np.empty((0,4),np.int32),'row_id':np.empty(0,np.int64),
            'start_velocity':np.empty((0,3),np.float32),'value_actions10':np.empty((0,10,7),np.float32)}
    keys=('coefficients','magnitude','length','task','episode','start','source','row_id','start_velocity','value_actions10')
    return {name:np.concatenate([b[name] for b in banks],axis=0) for name in keys}


def soft_action_targets(type_error: torch.Tensor, action_distance: torch.Tensor, candidate_mask: torch.Tensor,
                        *, tau_candidates=(.01, .03), tau_action=.03,tau_type=None):
    """Soft target combining S-shared type error and future action proximity."""
    if type_error.shape != action_distance.shape or candidate_mask.shape != type_error.shape:
        raise ValueError("pairwise target shapes differ")
    tau = float(tau_type if tau_type is not None else tau_candidates[0])
    logits = -type_error / max(tau, 1e-6) - action_distance / max(float(tau_action),1e-6)
    logits = logits.masked_fill(~candidate_mask.bool(), -torch.inf)
    valid = candidate_mask.any(-1, keepdim=True)
    target = torch.softmax(torch.where(valid, logits, torch.zeros_like(logits)), -1)
    return torch.where(valid, target, torch.zeros_like(target))


def soft_infonce(scores, target_distribution, candidate_mask):
    logits = scores.masked_fill(~candidate_mask.bool(), -torch.inf)
    valid=candidate_mask.bool().any(-1,keepdim=True)
    logp = torch.log_softmax(torch.where(valid,logits,torch.zeros_like(logits)), -1)
    safe = torch.where(candidate_mask.bool(), logp, torch.zeros_like(logp))
    return -(target_distribution * safe).sum(-1).mean()


def train_steps(model, optimizer, batches: Iterable[dict], *, steps, tau=.01,tau_action=.03,
                same_task_exclusion=True, pose_weight=1., generator=None):
    """Bounded training loop; caller supplies train-only minibatches."""
    history=[]
    model.train()
    iterator=iter(batches)
    for _ in range(int(steps)):
        batch=next(iterator)
        optimizer.zero_grad(set_to_none=True)
        result=training_objective(model.encoder,model.pose_head,batch,tau=tau,tau_action=tau_action,
            same_task_exclusion=same_task_exclusion,pose_weight=pose_weight,generator=generator)
        result['loss'].backward();optimizer.step()
        history.append({k:float(result[k].detach()) for k in ('loss','contrastive','pose')})
    return history


def select_dev_configuration(candidates, dev_evaluator: Callable, *, split_name):
    """Pick a setting using Dev rows only; intentionally rejects Eval input."""
    if str(split_name).lower() not in ('dev','development','validation'):
        raise ValueError("configuration selection is restricted to LIBERO-90 Dev")
    if not candidates:raise ValueError("no candidates")
    scored=[(dict(dev_evaluator(candidate)),candidate) for candidate in candidates]
    if any(not np.isfinite(float(x.get('type_error',np.nan))) for x,_ in scored):
        raise ValueError("Dev evaluator must return finite type_error")
    return min(scored,key=lambda item:(float(item[0]['type_error']),-float(item[0].get('spearman',-np.inf))))[1]


def retrieval_scores(query_keys, memory_keys, *, query_episode, memory_episode,
                     query_task=None, memory_task=None, previous_unit=None,
                     memory_unit=None, exclude_same_task=True, loop_exclude=False,
                     continuity=None, continuity_weight=0., score_scale=1.):
    """Cosine scores and eligibility mask; exclusions are explicit and auditable."""
    # Match training/offline evaluation: contrastive cosine logits are scaled
    # before the additive continuity term is applied.
    scores = (F.normalize(query_keys, dim=-1) @ F.normalize(memory_keys, dim=-1).T) * float(score_scale)
    eligible = torch.ones_like(scores, dtype=torch.bool)
    eligible &= torch.as_tensor(query_episode, device=scores.device)[:, None] != torch.as_tensor(memory_episode, device=scores.device)[None]
    if exclude_same_task:
        if query_task is None or memory_task is None:
            raise ValueError("task IDs required when same-task exclusion is active")
        eligible &= torch.as_tensor(query_task, device=scores.device)[:, None] != torch.as_tensor(memory_task, device=scores.device)[None]
    if loop_exclude and previous_unit is not None:
        if memory_unit is None:
            raise ValueError("memory_unit required for loop exclusion")
        eligible &= torch.as_tensor(previous_unit, device=scores.device)[:, None] != torch.as_tensor(memory_unit, device=scores.device)[None]
    if continuity is not None:
        scores = scores + float(continuity_weight) * continuity
    return scores.masked_fill(~eligible, -torch.inf), eligible


def continuity_bonus(candidate_start_velocity, previous_executed_velocity):
    """Negative distance term (higher means smoother continuation)."""
    return -torch.linalg.vector_norm(candidate_start_velocity - previous_executed_velocity, dim=-1)


def blend_boundary(actions, previous_action, *, steps=2):
    """Linearly bridge the first `steps` actions from the last executed action."""
    out = np.asarray(actions, np.float32).copy()
    prev = np.asarray(previous_action, np.float32)
    n = min(int(steps), len(out))
    if n < 0:
        raise ValueError("steps must be nonnegative")
    for i in range(n):
        alpha = (i + 1) / (n + 1)
        out[i] = (1 - alpha) * prev + alpha * out[i]
    return out


def select_memory(scores: torch.Tensor, eligible: torch.Tensor):
    """Return one memory index per query, or -1 if no eligible item exists."""
    if scores.shape != eligible.shape:
        raise ValueError("scores/eligibility shape mismatch")
    any_valid = eligible.any(-1)
    result = scores.argmax(-1)
    return torch.where(any_valid, result, torch.full_like(result, -1))


def offline_draft(coefficients, magnitudes, lengths, query_pose_scale, *, grip_actions,
                  out_length=10, boundary_blend_enabled=False, previous_action=None):
    """Pure NumPy offline draft builder shared with the online adapter."""
    coeff=np.asarray(coefficients);mag=np.atleast_1d(np.asarray(magnitudes));lens=np.atleast_1d(np.asarray(lengths))
    pose=np.asarray(query_pose_scale)
    if coeff.ndim==2: coeff=coeff[None]
    if pose.ndim==1:pose=pose[None]
    grips=np.asarray(grip_actions)
    if grips.ndim==1:grips=grips[None]
    result=[]
    for i in range(len(coeff)):
        base=segment_value_decode(coeff[i], float(mag[i]), int(lens[i]), out_length=out_length)
        rvec=pose[i,:3]
        angle=float(np.linalg.norm(rvec))
        if angle>1e-8:
            from scipy.spatial.transform import Rotation
            rot=Rotation.from_rotvec(rvec).as_matrix().T
            base=np.concatenate([base[:,:3]@rot,base[:,3:6]@rot],axis=-1)
        base*=float(np.exp(np.clip(pose[i,3],-5,5)))
        base=_normalized6(base)
        grip=np.asarray(grips[i],np.float32).reshape(out_length,-1)[:,:1]
        action=np.concatenate([base,grip],-1)
        if boundary_blend_enabled and previous_action is not None:
            action=blend_boundary(action, np.asarray(previous_action)[i])
        result.append(action)
    return np.asarray(result,np.float32)


def draft_one_unit(coefficients, magnitude, length, pose_scale, grip_action, *,
                   previous_action=None, boundary_blend_enabled=False):
    """Canonical one-unit decoder used by offline reports and online inference."""
    return offline_draft(np.asarray(coefficients)[None],np.asarray([magnitude]),np.asarray([length]),
        np.asarray(pose_scale)[None],grip_actions=np.asarray(grip_action)[None],
        boundary_blend_enabled=boundary_blend_enabled,
        previous_action=None if previous_action is None else np.asarray(previous_action)[None])[0]


class SegmentContextPolicy:
    """Small online adapter; caller owns the frozen pi05 prefix/action functions."""
    def __init__(self, encoder: SharedContextEncoder, bank: dict, *, prefix_fn: Callable,
                 pi05_fn: Callable, config: SegmentConfig, decode_fn: Callable | None=None,
                 pose_head: PoseScaleHead | None=None,value_encoder: Callable | None=None):
        self.encoder, self.bank = encoder, bank
        self.prefix_fn, self.pi05_fn, self.decode_fn = prefix_fn, pi05_fn, decode_fn
        self.pose_head=pose_head
        self.value_encoder=value_encoder
        if decode_fn is None and config.use_pose_scale_head and pose_head is None:
            raise ValueError("pose_head is required for the learned pose/scale path")
        self.config = config
        self.previous_actions, self.previous_states = [], []
        self.previous_unit = None
        self.previous_action = None
        self.previous_velocity = None

    def reset(self):
        self.previous_actions.clear(); self.previous_states.clear()
        self.previous_unit = None; self.previous_action = None; self.previous_velocity=None

    @torch.no_grad()
    def act(self, observation, state, *, task_id, episode_id):
        history=latest_completed_history(np.asarray(self.previous_actions),np.asarray(self.previous_states),
                max_history=self.config.max_history) if self.previous_actions else None
        if history is None:
            action = self.pi05_fn(observation)
            return action, {"fallback": True, "unit_id": None}
        tokens, mask = self.prefix_fn(observation)
        # Match the offline cache exactly: left-aligned fixed-length history
        # tensors plus an explicit mask. This also keeps the GRU path stable
        # between server (single query) and offline scoring.
        history_a,history_s,_=history
        packed_a,packed_s,packed_mask=pack_histories([(history_a,history_s)],max_length=self.config.max_history)
        hist_a=torch.as_tensor(packed_a,device=next(self.encoder.parameters()).device)
        hist_s=torch.as_tensor(packed_s,device=hist_a.device)
        hist_mask=torch.as_tensor(packed_mask,device=hist_a.device)
        tok=torch.as_tensor(tokens[None] if np.asarray(tokens).ndim==2 else tokens,device=hist_a.device,dtype=torch.float32)
        m=torch.as_tensor(mask[None] if np.asarray(mask).ndim==1 else mask,device=hist_a.device,dtype=torch.bool)
        st=torch.as_tensor(np.asarray(state)[None],device=hist_a.device,dtype=torch.float32)
        q=self.encoder(tok,m,hist_a,hist_s,st,hist_mask)
        continuity=None
        if self.config.continuity_weight and self.previous_velocity is not None and 'start_velocity' in self.bank:
            cv=continuity_bonus(torch.as_tensor(self.bank['start_velocity'],device=hist_a.device),
                                torch.as_tensor(self.previous_velocity,device=hist_a.device))
            continuity=cv[None]
        memory_keys=torch.as_tensor(self.bank['keys'],device=hist_a.device,dtype=q.dtype)
        scores,eligible=retrieval_scores(q,memory_keys,query_episode=[episode_id],memory_episode=self.bank["episode"],
            query_task=[task_id],memory_task=self.bank["task"],previous_unit=[self.previous_unit] if self.previous_unit is not None else None,
            memory_unit=self.bank["unit"],exclude_same_task=self.config.exclude_same_task,
            loop_exclude=self.config.loop_exclude_previous_unit,continuity=continuity,
            continuity_weight=self.config.continuity_weight,
            score_scale=float(self.encoder.logit_scale.detach().exp().clamp(max=100.).cpu()))
        ix=int(select_memory(scores,eligible)[0].item())
        theta=None
        if ix<0:
            action=self.pi05_fn(observation);unit=None
        else:
            if self.decode_fn is not None:
                action=self.decode_fn(q,ix,self.bank)
            else:
                baseline=np.asarray(self.pi05_fn(observation),np.float32)
                if baseline.ndim==3:baseline=baseline[0]
                baseline=baseline[:10]
                coeff=np.asarray(self.bank['coefficients'][ix]);mag=float(self.bank['magnitude'][ix])
                length=int(self.bank['length'][ix])
                if self.config.value_head_mode=='split':
                    if self.value_encoder is None or 'value_actions10' not in self.bank:
                        raise ValueError('split value head requires scene_invariance_v2 encode_keys adapter')
                    value=self.value_encoder(self.bank['value_actions10'][ix:ix+1])
                    if not torch.is_tensor(value):value=torch.as_tensor(value,device=q.device,dtype=q.dtype)
                    value=value.reshape(1,-1)
                else:
                    value=torch.as_tensor(coeff.reshape(1,-1),device=q.device,dtype=q.dtype)
                if self.config.use_pose_scale_head:
                    with torch.no_grad():
                        theta=self.pose_head(q,self.bank['keys'][ix:ix+1],
                            value).cpu().numpy()[0]
                else:
                    source=segment_value_decode(coeff,mag,length,out_length=10)
                    theta=fit_pose_scale(_normalized6(source),baseline[:,:6])
                action=draft_one_unit(coeff,mag,length,theta,baseline[:,6])
            unit=int(self.bank["unit"][ix])
            if self.config.boundary_blend and self.previous_action is not None:
                action=blend_boundary(action,self.previous_action)
        if unit is not None:self.previous_unit=unit
        return action,{"fallback":ix<0,"unit_id":unit,"score":float(scores[0,ix]) if ix>=0 else None,
            "pose_scale":None if theta is None else np.asarray(theta).tolist()}

    def observe_executed(self, actions, states):
        """Append only actions/states actually executed by the environment."""
        a=np.asarray(actions,np.float32)
        s=np.asarray(states,np.float32)
        if a.ndim==1:a=a[None]
        if s.ndim==1:s=np.broadcast_to(s,(len(a),len(s)))
        if a.ndim!=2 or a.shape[1]<7 or s.ndim!=2 or len(a)!=len(s) or s.shape[1]<8:
            raise ValueError("executed actions/states must align [T,>=7]/[T,>=8]")
        self.previous_actions.extend(row[:7].copy() for row in a)
        self.previous_states.extend(row[:8].copy() for row in s)
        self.previous_action=a[-1,:7].copy()
        self.previous_velocity=_physical6(a[-1:,:6])[0,:3]


def evaluate_retrieval(scores, type_error, action_error, *, row_episode, hard=None, type_tolerance=0.01):
    """Report type recall/rank and selected-action MSE split into first/next five."""
    from scipy.stats import spearmanr
    score=np.asarray(scores);te=np.asarray(type_error);ae=np.asarray(action_error)
    ix=np.argmax(score,axis=1);rows=np.arange(len(ix));chosen_type=te[rows,ix]
    positive=te <= (te.min(axis=1,keepdims=True)+float(type_tolerance))
    ranks=np.argsort(np.argsort(-score,axis=1),axis=1)+1
    def pack(mask):
        if not mask.any():return {"rows":0}
        rho=[spearmanr(score[i],te[i]).statistic for i in np.flatnonzero(mask)]
        chosen=ae[rows,ix]
        return {"rows":int(mask.sum()),"top1_type_error":float(chosen_type[mask].mean()),
            "recall_at_1":float(np.mean(positive[rows,ix][mask])),
            "recall_at_5":float(np.mean((positive & (ranks<=5)).any(1)[mask])),
            "spearman_mean":float(np.nanmean(rho)),
            "mse_first5_x6_7":float(chosen[mask,:5].mean()*6/7),
            "mse_continuation5_x6_7":float(chosen[mask,5:10].mean()*6/7)}
    mask=np.ones(len(ix),bool)
    out={"all":pack(mask)}
    if hard is not None:out["hard"]=pack(np.asarray(hard,bool))
    return out


def baseline_action_metrics(target, *, normal, previous_hold, unit08_pose_draft, selected):
    """Common active7 6-axis MSE report for requested SEG1 reference policies."""
    y=np.asarray(target,np.float32)[...,:7]
    sources={'normal_pi05':normal,'hold_last':previous_hold,
             'query_unit08_plus_pi05_procrustes':unit08_pose_draft,'selected_segment':selected}
    out={}
    for name,value in sources.items():
        p=np.asarray(value,np.float32)[...,:7]
        if p.shape!=y.shape:raise ValueError(f"{name}: shape mismatch {p.shape} != {y.shape}")
        err=(p-y)**2
        out[name]={'all_x6_7':float(err.mean()*6/7),
                   'first5_x6_7':float(err[:,:5].mean()*6/7),
                   'continuation5_x6_7':float(err[:,5:10].mean()*6/7)}
    return out


def paired_episode_bootstrap(a,b,episodes,*,reps=5000,seed=4101):
    """Paired mean difference CI with episode-level resampling."""
    a=np.asarray(a,np.float64);b=np.asarray(b,np.float64);ep=np.asarray(episodes)
    if not(len(a)==len(b)==len(ep)):raise ValueError("paired bootstrap inputs differ in length")
    d=a-b;uniq,inv=np.unique(ep,return_inverse=True)
    sums=np.bincount(inv,weights=d);counts=np.bincount(inv)
    rng=np.random.default_rng(seed);ix=rng.integers(0,len(uniq),size=(int(reps),len(uniq)))
    draws=sums[ix].sum(1)/counts[ix].sum(1)
    return {'mean':float(d.mean()),'ci95':[float(np.quantile(draws,.025)),float(np.quantile(draws,.975))],
            'episodes':int(len(uniq)),'reps':int(reps)}


def write_extraction_manifest(cache_root, output_root, *, query_stride=5, segment_starts=()):
    """Inspect existing row-token cache and declare missing stride query frames."""
    from pathlib import Path
    root=Path(cache_root); out=Path(output_root)
    rows=root/"rows"
    frame=np.load(rows/"frame.npy",mmap_mode="r",allow_pickle=False)
    task=np.load(rows/"task.npy",mmap_mode="r",allow_pickle=False)
    episode=np.load(rows/"episode.npy",mmap_mode="r",allow_pickle=False)
    token=np.load(root/"tokens.npy",mmap_mode="r",allow_pickle=False)
    groups=np.load(root/"groups.npy",mmap_mode="r",allow_pickle=False)
    existing={(int(t),int(e),int(f)) for t,e,f in zip(task,episode,frame)}
    required=set(segment_starts)
    for t,e in sorted(set((int(t),int(e)) for t,e in zip(task,episode))):
        fs=frame[(task==t)&(episode==e)]
        if len(fs):
            lo,hi=int(fs.min()),int(fs.max())
            first=((lo+int(query_stride)-1)//int(query_stride))*int(query_stride)
            required.update((t,e,f) for f in range(first,hi+1,int(query_stride)))
    missing=sorted(required-existing)
    differences=np.diff(np.unique(frame))
    cached_stride=int(np.gcd.reduce(differences)) if len(differences) else int(query_stride)
    return {"token_shape":list(token.shape),"group_shape":list(groups.shape),"cached_stride":cached_stride,
            "query_stride":int(query_stride),"cached_rows":len(frame),"required_query_count":len(required),
            "missing_count_estimate":len(missing),
            "missing_task_episode_frame":missing,"output_root":str(out),"cache_reused_read_only":True}
