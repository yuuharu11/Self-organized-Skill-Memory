#!/usr/bin/env python3
"""Frozen AX/PT token sources and KEY2-compatible adapters (LT1).

AX is the action-expert suffix stream. Extraction is callback based so the
policy integration loop remains the single owner of its exact model call.
PT is causal CoTracker3 input: the supplied clip must end at the query frame.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

AX_TIMES = (1.0, 0.5, 0.0)
AX_LAYER_INDICES = (2, 8, 16)  # 0-based layers of the 18-layer action expert
AX_TAPS = tuple(f"action_expert_layer_{i + 1:02d}_v" for i in AX_LAYER_INDICES)
PT_HISTORY = 16
LT1_CONDITIONS = {
    "E-AX": ("ax", False), "E-AX+H": ("ax", True),
    "E-PT": ("pt", False), "E-PT+H": ("pt", True),
    "E-AX+PT+H": ("ax_pt", True),
}


def fixed_noise(row_key: Sequence[int], *, seed: int = 4101, shape=(10, 7)) -> np.ndarray:
    """Stable per-row Gaussian noise, independent of iteration order."""
    key = tuple(map(int, row_key))
    digest = hashlib.blake2b(np.asarray((seed, *key), np.int64).tobytes(), digest_size=8).digest()
    rng = np.random.default_rng(int.from_bytes(digest, "little"))
    return rng.standard_normal(shape).astype(np.float32)


def collect_ax_path(noise: np.ndarray, step_fn: Callable[[np.ndarray, float], tuple[np.ndarray, Mapping[str, np.ndarray]]],
                    *, steps: int = 10, capture_times: Sequence[float] = AX_TIMES):
    """Integrate the OpenPI t=1→0 flow and collect requested hidden token taps.

    ``step_fn(x,t)`` must return ``(velocity, {layer_name: [H,D]})`` from the
    frozen action expert at the same x,t. It is intentionally explicit: a
    caller cannot silently substitute the final action or a policy output.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    x = np.asarray(noise, np.float32).copy()
    if x.ndim != 2:
        raise ValueError("noise must have shape [horizon, action_dim]")
    wanted = {float(t) for t in capture_times}
    if any(not 0 <= t <= 1 for t in wanted):
        raise ValueError("capture times must be in [0,1]")
    out: dict[float, dict[str, np.ndarray]] = {}
    dt = 1.0 / steps
    for i in range(steps):
        t = 1.0 - i / steps
        velocity, hidden = step_fn(x, t)
        if any(abs(t - q) < 1e-8 for q in wanted):
            out[float(t)] = {str(k): np.asarray(v, np.float16) for k, v in hidden.items()}
        v = np.asarray(velocity, np.float32)
        if v.shape != x.shape:
            raise ValueError("expert velocity shape differs from action state")
        x -= dt * v
    # t=0 hidden is evaluated at the integrated endpoint, as in sample_actions.
    if 0.0 in wanted:
        _, hidden = step_fn(x, 0.0)
        out[0.0] = {str(k): np.asarray(v, np.float16) for k, v in hidden.items()}
    missing = wanted - set(out)
    if missing:
        raise ValueError(f"requested flow times not visited by steps={steps}: {sorted(missing)}")
    return x, out


def validate_ax_feature_grid(hidden_by_time: Mapping[float, Mapping[str, np.ndarray]], *,
                             times: Sequence[float] = AX_TIMES,
                             layers: Sequence[str] = AX_TAPS):
    """Fail closed unless all requested real AX time/layer taps are present."""
    missing_time = [float(t) for t in times if float(t) not in hidden_by_time]
    if missing_time:
        raise ValueError(f"AX extraction missing requested flow times: {missing_time}")
    result=[]; tap_shape=None
    for t in times:
        block=hidden_by_time[float(t)]
        absent=[name for name in layers if name not in block]
        if absent:
            raise ValueError(f"AX extraction missing expert layer taps at t={t}: {absent}")
        row=[]
        for layer in layers:
            x=np.asarray(block[layer])
            if x.ndim not in (2,3):
                raise ValueError(f"AX {layer} at t={t} must be [H,D] or [B,H,D]")
            if tap_shape is None: tap_shape=x.shape
            if x.shape!=tap_shape:
                raise ValueError("AX taps must have equal batch, action horizon and width")
            row.append(x.astype(np.float16,copy=False))
        result.append(row)
    return np.stack(result,axis=0)


def _ax_velocity_and_v_taps(model, obs, prefix_mask, prefix_cache, x, t, layer_indices):
    """One exact OpenPI action-expert step plus per-layer cached V tokens.

    OpenPI's scanned Gemma block does not expose residual-stream activations.
    Its returned KV cache does expose each layer's action-suffix V token rows;
    LT1 uses these explicitly labeled internal V representations (not residual
    hidden states) rather than pretending the final projection input is three
    different layers.
    """
    import jax.numpy as jnp
    from openpi.models.pi0 import make_attn_mask
    suffix, smask, sar, cond = model.embed_suffix(obs, x, jnp.broadcast_to(jnp.asarray(t,jnp.float32),(len(x),)))
    attn=jnp.concatenate([jnp.broadcast_to(prefix_mask[:,None,:],(len(x),suffix.shape[1],prefix_mask.shape[1])),
                          make_attn_mask(smask,sar)],axis=-1)
    pos=jnp.sum(prefix_mask,axis=1)[:,None]+jnp.cumsum(smask,axis=-1)-1
    (_,out), suffix_cache=model.PaliGemma.llm([None,suffix],mask=attn,positions=pos,
        kv_cache=prefix_cache,adarms_cond=[None,cond])
    values=suffix_cache[1]
    if values.ndim!=5 or values.shape[0]<=max(layer_indices):
        raise ValueError(f"unexpected Gemma V cache shape {values.shape}")
    horizon=int(model.action_horizon)
    taps={AX_TAPS[j]:values[int(layer),:,-horizon:].reshape((len(x),horizon,-1))
          for j,layer in enumerate(layer_indices)}
    return model.action_out_proj(out[:,-horizon:]), taps


def extract_ax_model_path(model, observation, row_keys, *, seed: int = 4101, steps: int = 10,
                          layer_indices: Sequence[int] = AX_LAYER_INDICES,
                          capture_times: Sequence[float] = AX_TIMES):
    """Extract real action-expert V-token taps at fixed-noise flow points.

    The same prefix KV cache is reused at every integration step, matching
    ``Pi0.sample_actions``. Output actions are diagnostic only and must not be
    supplied as retrieval features.
    """
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_api
    from openpi.models.pi0 import make_attn_mask
    obs=model_api.preprocess_observation(None,observation,train=False)
    prefix,pmask,par=model.embed_prefix(obs)
    _,prefix_cache=model.PaliGemma.llm([prefix,None],mask=make_attn_mask(pmask,par),
                                      positions=jnp.cumsum(pmask,axis=1)-1)
    row_keys=np.asarray(row_keys,np.int64)
    if row_keys.shape!=(len(prefix),3):raise ValueError("AX row_keys must be [batch,3] exact identities")
    # Fold a stable packed identity into a base key; no target action enters.
    row_hash=np.asarray([int.from_bytes(hashlib.blake2b(k.tobytes(),digest_size=4).digest(),'little') for k in row_keys],np.uint32)
    key=jax.vmap(lambda r:jax.random.fold_in(jax.random.key(seed),r))(jnp.asarray(row_hash))
    x=jax.vmap(lambda k:jax.random.normal(k,(model.action_horizon,model.action_dim)))(key)
    wanted={float(t) for t in capture_times}
    if not wanted.issubset(set(AX_TIMES)):
        raise ValueError(f"model extractor supports fixed flow points {AX_TIMES}")
    layer_names=[AX_TAPS[AX_LAYER_INDICES.index(int(i))] for i in layer_indices]
    action_config=model.PaliGemma.llm.module.configs[1]
    v_width=int(action_config.num_kv_heads*action_config.head_dim)
    tap_shape=(len(prefix),len(layer_names),int(model.action_horizon),v_width)
    states=jnp.zeros((int(steps),)+tap_shape,dtype=jnp.float16)
    dt=jnp.asarray(-1.0/float(steps),jnp.float32)
    initial=(x,jnp.asarray(1.0,jnp.float32),jnp.asarray(0,jnp.int32),states)
    def cond(carry): return carry[1]>=-dt/2
    def body(carry):
        value,time,index,all_taps=carry
        vel,taps=_ax_velocity_and_v_taps(model,obs,pmask,prefix_cache,value,time,tuple(map(int,layer_indices)))
        layer_values=jnp.stack([taps[name] for name in layer_names],axis=1).astype(jnp.float16)
        all_taps=all_taps.at[index].set(layer_values)
        return value+dt*vel,time+dt,index+1,all_taps
    x,_,count,states=jax.lax.while_loop(cond,body,initial)
    if int(np.asarray(count))!=int(steps):
        raise AssertionError(f"OpenPI integration step count changed: {int(np.asarray(count))}")
    _,final_taps=_ax_velocity_and_v_taps(model,obs,pmask,prefix_cache,x,0.0,tuple(map(int,layer_indices)))
    captured={}
    # State layout is [step, batch, layer, action_horizon, width].
    def unpack(layer_rows):
        return {name:np.asarray(layer_rows[:,j]) for j,name in enumerate(layer_names)}
    if 1.0 in wanted:captured[1.0]=unpack(states[0])
    if 0.5 in wanted:captured[0.5]=unpack(states[int(steps)//2])
    if 0.0 in wanted:captured[0.0]={k:np.asarray(v) for k,v in final_taps.items()}
    if wanted-set(captured):raise ValueError(f"AX integration did not visit {sorted(wanted-set(captured))}")
    grid=validate_ax_feature_grid(captured,times=capture_times,layers=[AX_TAPS[AX_LAYER_INDICES.index(int(i))] for i in layer_indices])
    return np.asarray(x[...,:7]),grid


def align_feature_cache(wanted_keys, cache_keys, tokens, groups):
    """Join exact (task,episode,frame) rows, rejecting duplicates and misses."""
    wanted=np.asarray(wanted_keys,np.int64); available=np.asarray(cache_keys,np.int64)
    tok=np.asarray(tokens); grp=np.asarray(groups)
    if wanted.ndim!=2 or wanted.shape[1]!=3 or available.ndim!=2 or available.shape[1]!=3:
        raise ValueError("feature keys must have shape [N,3]")
    if len(set(map(tuple,available.tolist())))!=len(available):
        raise ValueError("duplicate feature-cache row key")
    # Token caches may carry additional source axes (LT1 AX has [time,layer]
    # before token,width).  Preserve them while aligning the leading row axis.
    if tok.ndim<3 or grp.shape!=tok.shape[:-1] or len(tok)!=len(available):
        raise ValueError("feature cache token/group shape mismatch")
    index={tuple(map(int,k)):i for i,k in enumerate(available)}
    missing=[tuple(map(int,k)) for k in wanted if tuple(map(int,k)) not in index]
    if missing:
        raise KeyError(f"feature cache misses {len(missing)} exact rows; examples={missing[:3]}")
    selected=np.asarray([index[tuple(map(int,k))] for k in wanted],np.int64)
    return tok[selected],grp[selected]


def key2_condition_config(name: str):
    """Map an LT1 label to KEY2's shared-E input mode and source flags."""
    if name not in LT1_CONDITIONS:
        raise ValueError(f"unknown LT1 condition: {name}")
    source,history=LT1_CONDITIONS[name]
    if source=="ax_pt":mode="e_vj_vlm_h";token_branch="ax+pt"
    else:mode="e_vlm_h" if history else "e_vlm";token_branch=source
    return {"encoder_input_mode":mode,"source":source,"token_branch":token_branch,
            "use_history":history,"dev_metric":"top1_arm_mse"}


def inject_condition_features(data: Mapping[str, Mapping], *, condition: str,
                              ax_cache=None, pt_cache=None):
    """Inject AX/PT rows into KEY2's existing split structures by exact key."""
    cfg=key2_condition_config(condition);source=cfg["source"]
    if source in ("ax","ax_pt") and ax_cache is None:raise ValueError(f"{condition} requires AX cache")
    if source in ("pt","ax_pt") and pt_cache is None:raise ValueError(f"{condition} requires PT cache")
    out={}
    for name,values in data.items():
        if not isinstance(values,Mapping) or "key" not in values:
            out[name]=values;continue
        item=dict(values);keys=item["key"]
        if source in ("ax","ax_pt"):
            tok,grp=align_feature_cache(keys,*ax_cache[:3])
            item["tokens"]=tok;item["groups"]=grp
        if source=="pt":
            tok,grp=align_feature_cache(keys,*pt_cache[:3])
            item["tokens"]=tok;item["groups"]=grp
        elif source=="ax_pt":
            tok,grp=align_feature_cache(keys,*pt_cache[:3])
            item["vjepa_tokens"]=tok;item["vjepa_groups"]=grp
        # Eliminate stale inputs from the standard KEY2 cache, guaranteeing
        # that no VLM/V-JEPA fallback silently enters an AX/PT condition.
        if source!="ax_pt":item.pop("vjepa_tokens",None);item.pop("vjepa_groups",None)
        out[name]=item
    return out


def inject_source_cache(key2_mapping: Mapping[str, Mapping], *, cache_keys, tokens, groups,
                       names: Sequence[str] | None = None):
    """Replace only observation tokens in KEY2 split dictionaries by exact join.

    Action labels, candidate pool, allowed masks and split identity remain
    untouched. The caller supplies only named maps (e.g. bank_input/train/dev/
    learned/unseen), making train/Eval leakage checks straightforward.
    """
    out={}
    selected_names=set(names) if names is not None else set(key2_mapping)
    unknown=selected_names-set(key2_mapping)
    if unknown:
        raise KeyError(f"unknown KEY2 feature maps: {sorted(unknown)}")
    for name,values in key2_mapping.items():
        if name not in selected_names:
            out[name]=dict(values); continue
        if "key" not in values:
            raise KeyError(f"KEY2 map {name} has no exact (task,episode,frame) keys")
        tok,grp=align_feature_cache(values["key"],cache_keys,tokens,groups)
        item=dict(values)
        # Erase other visual sources so the named LT1 condition cannot
        # accidentally fall back to VLM or V-JEPA tokens.
        item.update(tokens=tok,groups=grp)
        item.pop("vjepa_tokens",None);item.pop("vjepa_groups",None)
        out[name]=item
    return out


def validate_causal_clip(frames: np.ndarray, *, current_frame_index: int, frame_indices: Sequence[int],
                         history: int = PT_HISTORY) -> np.ndarray:
    """Return exactly the past/current frames; reject any future or gaps."""
    f = np.asarray(frames)
    ix = np.asarray(frame_indices, np.int64)
    if f.ndim != 4 or len(f) != len(ix) or len(ix) == 0:
        raise ValueError("frames must be [T,H,W,C] and align with frame_indices")
    if np.any(np.diff(ix) != 1):
        raise ValueError("frame indices must be contiguous and strictly increasing")
    if np.any(ix > int(current_frame_index)):
        raise ValueError("future frame supplied to causal tracker")
    if ix[-1] != int(current_frame_index):
        raise ValueError("clip must end at the query frame")
    # The protocol requests history up to 16 frames; short episode prefixes
    # remain valid and are left padded by the caller's tracked-visibility mask.
    return f[-int(history):]


def load_hdf5_history(source_hdf5: Path, demo: str, camera: str, *, current_frame_10hz: int,
                      history: int = PT_HISTORY, subsample: int = 2, rotate180: bool = True):
    """Read past/current LIBERO RGB frames at 10Hz using the unseen-row convention.

    ``agentview_rgb`` is base and ``eye_in_hand_rgb`` is wrist. The 10Hz frame
    index maps to source HDF5 frame ``index * subsample``. Images are rotated
    180° before they enter the feature pipeline, matching unseen-rows-02.
    Short episode prefixes are left padded by repeating the earliest available
    observation; the returned validity mask distinguishes those copies.
    """
    import h5py
    names = {"base": "agentview_rgb", "wrist": "eye_in_hand_rgb"}
    if camera not in names or history < 1 or subsample < 1 or current_frame_10hz < 0:
        raise ValueError("invalid camera/history/frame settings")
    with h5py.File(source_hdf5, "r") as f:
        ds = f["data"][str(demo)]["obs"][names[camera]]
        n = int(ds.shape[0]); end = min(n - 1, int(current_frame_10hz) * int(subsample))
        if end < 0:
            raise ValueError("empty HDF5 episode")
        wanted10 = np.arange(int(current_frame_10hz) - int(history) + 1,
                             int(current_frame_10hz) + 1, dtype=np.int64)
        valid = wanted10 >= 0
        src = np.clip(wanted10, 0, None) * int(subsample)
        src = np.minimum(src, end)
        images = np.asarray(ds[src], dtype=np.uint8)
    if rotate180:
        images = images[:, ::-1, ::-1, :]
    return np.ascontiguousarray(images), valid


def grid_queries(height: int, width: int, grid: int = 10, *, device="cpu") -> torch.Tensor:
    """CoTracker query points [1,N,3], with t=0 and pixel x,y coordinates."""
    if min(height, width, grid) < 1:
        raise ValueError("image and grid sizes must be positive")
    ys = torch.linspace(0, height - 1, grid, device=device)
    xs = torch.linspace(0, width - 1, grid, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((torch.zeros_like(xx), xx, yy), dim=-1).reshape(1, grid * grid, 3)


def cotracker_video_tensor(frames: np.ndarray, *, device="cpu") -> torch.Tensor:
    """Public CoTracker input layout and pixel scale (RGB float in [0,255])."""
    x = np.asarray(frames)
    if x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError("expected RGB frames [T,H,W,3]")
    return torch.from_numpy(x.copy()).to(device).permute(0, 3, 1, 2)[None].float()


def tracks_to_tokens(tracks: np.ndarray, visibility: np.ndarray, *, group: int,
                     image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Encode tracked point displacement histories as per-point tokens.

    Each point token contains normalized dx/dy from first visible position and
    a visibility bit at every observed step. Group 0=base, 1=wrist.
    """
    tr = np.asarray(tracks, np.float32)
    vis = np.asarray(visibility, bool)
    if tr.ndim != 3 or tr.shape[-1] != 2 or vis.shape != tr.shape[:2]:
        raise ValueError("tracks must be [T,N,2], visibility [T,N]")
    h, w = map(int, image_size)
    if h < 1 or w < 1 or group not in (0, 1):
        raise ValueError("invalid image_size or camera group")
    t, n, _ = tr.shape
    first = np.zeros((n, 2), np.float32)
    for j in range(n):
        ids = np.flatnonzero(vis[:, j])
        if len(ids):
            first[j] = tr[ids[0], j]
    delta = tr - first[None]
    # CoTracker may return arbitrary coordinates while a point is invisible.
    # Do not let those values enter the displacement channels; visibility is
    # represented separately below.
    delta = np.where(vis[..., None], delta, 0.0)
    delta[..., 0] /= max(w, 1)
    delta[..., 1] /= max(h, 1)
    token = np.concatenate([delta.transpose(1, 0, 2), vis.T[..., None].astype(np.float32)], axis=-1)
    return token.reshape(n, t * 3).astype(np.float16), np.full(n, int(group), np.int8)


def track_cotracker3(frames: np.ndarray, *, grid: int = 10, model=None, device="cuda"):
    """Run public CoTracker3 online weights on one causal clip.

    Input frames are RGB uint8 [T,H,W,3], ordered oldest to current. The
    official online tracker sees no frame after the query by construction.
    """
    video = np.asarray(frames)
    if video.ndim != 4 or video.shape[-1] != 3 or len(video) < 1:
        raise ValueError("expected RGB clip [T,H,W,3]")
    if model is None:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online", pretrained=True)
    model = model.to(device).eval()
    x = cotracker_video_tensor(video, device=device)
    q = grid_queries(video.shape[1], video.shape[2], grid, device=device)
    with torch.inference_mode():
        # The public online predictor has a stateful init call; a plain
        # predictor(video, queries=...) is the offline API and leaks future
        # context in its default bidirectional mode.
        model(x, is_first_step=True, queries=q, grid_size=0)
        tracks, vis = model(x, is_first_step=False)
    return tracks[0].detach().cpu().numpy(), vis[0].detach().cpu().numpy().astype(bool)


class LT1ConditionEncoder(nn.Module):
    """KEY2 SharedLatentEncoder plus a trainable adapter for 48-D PT tokens.

    The wrapper keeps one shared encoder call for both memory and query. In the
    combined AX+PT condition, PT tracks are projected into the encoder's
    expected 1024-D V-JEPA slot; AX tokens remain in the VLM slot. Both
    projection and SharedLatentEncoder parameters are trainable together.
    """
    def __init__(self, encoder: nn.Module, *, pt_projection: nn.Module | None = None):
        super().__init__()
        self.encoder = encoder
        self.pt_projection = pt_projection

    @property
    def logit_scale(self):
        return self.encoder.logit_scale

    def forward(self, *, tokens, groups, history=None, history_mask=None, state=None,
                vjepa_tokens=None, vjepa_groups=None):
        if vjepa_tokens is not None:
            if self.pt_projection is None:
                raise ValueError("this LT1 condition has no PT projection")
            vjepa_tokens = self.pt_projection(vjepa_tokens)
        return self.encoder(vlm_tokens=tokens, groups=groups, history=history,
                            history_mask=history_mask, state=state,
                            vjepa_tokens=vjepa_tokens, vjepa_groups=vjepa_groups)


def make_condition_encoder(condition: str, *, token_count: int = 5):
    """Build the exact KEY2 encoder input mode for an LT1 condition.

    Returns a single module so an optimizer includes the shared E and, for
    AX+PT, the learned 48→1024 point-token adapter.
    """
    from key_shared_latent import SharedLatentEncoder
    cfg=key2_condition_config(condition)
    source=cfg["source"]
    token_dim=48 if source=="pt" else 256
    encoder=SharedLatentEncoder(token_dim=token_dim, context_dim=256,token_count=token_count,
                                input_mode=cfg["encoder_input_mode"],vjepa_dim=1024)
    pt_projection=PointTokenProjection(output_dim=1024) if source=="ax_pt" else None
    return LT1ConditionEncoder(encoder,pt_projection=pt_projection)


def make_shared_encoder(*, token_dim: int, token_count: int = 5, with_history: bool = False):
    """Compatibility factory for one AX/PT token source."""
    from key_shared_latent import SharedLatentEncoder
    return SharedLatentEncoder(token_dim=token_dim, context_dim=256, token_count=token_count,
                               input_mode="e_vlm_h" if with_history else "e_vlm")


class PointTokenProjection(nn.Module):
    """Trainable PT token projection to KEY2's 1024-dimensional token space."""
    def __init__(self, input_dim: int = PT_HISTORY * 3, output_dim: int = 1024):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, output_dim), nn.GELU(),
                                  nn.Linear(output_dim, output_dim))

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != PT_HISTORY * 3:
            raise ValueError(f"PT tokens must be [B,N,{PT_HISTORY * 3}]")
        return self.proj(x.float())


class MeanPoolMLP(nn.Module):
    """P0-style reference: masked mean source tokens then an MLP embedding."""
    def __init__(self, input_dim: int, output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 512), nn.GELU(), nn.Linear(512, output_dim))

    def forward(self, tokens, groups):
        valid = groups >= 0
        denom = valid.sum(1, keepdim=True).clamp_min(1).to(tokens.dtype)
        pooled = (tokens.float() * valid[..., None]).sum(1) / denom
        return torch.nn.functional.normalize(self.net(pooled), dim=-1)


def pack_source_tokens(*, ax_tokens=None, ax_groups=None, pt_tokens=None, pt_groups=None,
                       pt_projection: PointTokenProjection | None = None):
    """Pack AX/PT in one token dimension and disjoint base/wrist groups.

    AX's action-horizon tokens are already width 1024 in π0.5. PT's 16-frame
    point histories (48 values) pass through a learned projection. In a
    combined condition, both token sets enter the same KEY2 cross-attention.
    """
    toks=[]; groups=[]
    if ax_tokens is not None:
        a=torch.as_tensor(ax_tokens)
        g=torch.as_tensor(ax_groups, device=a.device, dtype=torch.long)
        if a.ndim != 3 or a.shape[-1] not in (256,1024) or g.shape != a.shape[:2]:
            raise ValueError("AX must be [B,T,256] (layer V) or [B,T,1024] (suffix hidden)")
        toks.append(a.float()); groups.append(g)
    if pt_tokens is not None:
        p=torch.as_tensor(pt_tokens)
        g=torch.as_tensor(pt_groups, device=p.device, dtype=torch.long)
        if p.ndim != 3 or p.shape[-1] != PT_HISTORY * 3 or g.shape != p.shape[:2]:
            raise ValueError("PT must be [B,N,48] with matching group IDs")
        if pt_projection is None:
            raise ValueError("PT tokens require a trainable PointTokenProjection")
        projected=pt_projection(p)
        if toks and projected.shape[-1]!=toks[0].shape[-1]:
            raise ValueError("AX and PT token widths must match after source projection")
        toks.append(projected); groups.append(g)
    if not toks:
        raise ValueError("at least one feature source is required")
    if len({int(x.shape[0]) for x in toks}) != 1:
        raise ValueError("AX/PT batch sizes differ")
    return torch.cat(toks, dim=1), torch.cat(groups, dim=1)


def point_tokens_for_two_cameras(base_frames: np.ndarray, wrist_frames: np.ndarray, *,
                                 base_indices, wrist_indices, current_frame: int,
                                 model=None, device="cuda", grid: int = 10):
    """Extract causal point tokens for both cameras, always ending at current."""
    base = validate_causal_clip(base_frames, current_frame_index=current_frame, frame_indices=base_indices)
    wrist = validate_causal_clip(wrist_frames, current_frame_index=current_frame, frame_indices=wrist_indices)
    if not np.array_equal(np.asarray(base_indices)[-len(base):], np.asarray(wrist_indices)[-len(wrist):]):
        raise ValueError("base/wrist history frames must align exactly")
    tb, vb = track_cotracker3(base, grid=grid, model=model, device=device)
    tw, vw = track_cotracker3(wrist, grid=grid, model=model, device=device)
    b, bg = tracks_to_tokens(tb, vb, group=0, image_size=base.shape[1:3])
    w, wg = tracks_to_tokens(tw, vw, group=1, image_size=wrist.shape[1:3])
    return np.concatenate([b, w]), np.concatenate([bg, wg])


def encode_key_query(shared_encoder, key_inputs: dict, query_inputs: dict):
    """Use one and the same KEY2 encoder for memory keys and query tokens."""
    key = shared_encoder(**key_inputs)
    query = shared_encoder(**query_inputs)
    return key, query


def choose_dev_setting(rows: Sequence[Mapping], *, metric: str = "top1_arm_mse") -> Mapping:
    """Pick feature/time/layer configuration by Dev top-1 arm MSE only."""
    if metric != "top1_arm_mse":
        raise ValueError("LT1 selection is locked to Dev top-1 arm MSE")
    if not rows:
        raise ValueError("no Dev settings to select")
    return min(rows, key=lambda x: (float(x[metric]), str(x.get("setting", ""))))


def write_feature_cache(path: Path, *, row_keys, tokens, groups, source: str, config: Mapping,
                        overwrite: bool = False):
    """Atomic cache writer with explicit source/config provenance."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    keys = np.asarray(row_keys, np.int64); tok = np.asarray(tokens, np.float16); grp = np.asarray(groups, np.int8)
    if keys.ndim != 2 or keys.shape[1] != 3 or tok.ndim != 3 or grp.shape != tok.shape[:2] or len(tok) != len(keys):
        raise ValueError("cache requires row_keys[N,3], tokens[N,T,D], groups[N,T]")
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite feature cache: {path}")
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, row_keys=keys, tokens=tok, groups=grp,
                            source=np.asarray(source), config_json=np.asarray(json.dumps(dict(config), sort_keys=True)))
    tmp.replace(path)


def load_feature_cache(path: Path):
    z = np.load(path, allow_pickle=False)
    keys = z["row_keys"]; tok = z["tokens"]; groups = z["groups"]
    if tok.shape[:2] != groups.shape or len(keys) != len(tok):
        raise ValueError("corrupt LT1 cache shapes")
    return keys, tok, groups, str(z["source"]), json.loads(str(z["config_json"]))


def lt1_fixed_row_keys(run_root: Path = Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/run-01')):
    """Return the immutable LT1 row sets: memory starts, learned, unseen, Dev."""
    import sys
    here=Path(__file__).resolve().parent
    sys.path[:0]=[str(here),str(here.parent/'pi05_memory')]
    import gripper_split as grip
    import motion_core_eval as motion
    from key_shared_latent import _load_key2_rows
    rows=_load_key2_rows(run_root)
    memory_ids=np.flatnonzero(np.asarray(rows['memory'],bool))
    starts=[]
    for (task,episode),row_ids in grip._traces(rows,memory_ids).items():
        trace=np.concatenate([np.asarray(rows['actions'][i,:,:7],np.float32) for i in row_ids])
        frame0=int(rows['frame'][row_ids[0]])
        starts.extend((int(task),int(episode),frame0+int(s)) for s,_ in motion.anchor_spans(trace,min_length=5))
    memory=np.asarray(starts,np.int32)
    if memory.shape!=(4516,3) or len(np.unique(memory,axis=0))!=4516:
        raise AssertionError(f'LT1 memory unit starts changed: {memory.shape}')
    root=Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero')
    ref=np.load(root/'pi05-reference-01/selection.npz',allow_pickle=False)
    selection=json.loads((root/'compress-seen-cuts-01/production-control/evaluation-selection.json').read_text())
    learned=np.column_stack([ref['learned_task'],ref['learned_episode'],ref['learned_frame']]).astype(np.int32)
    unseen=np.column_stack([ref['unseen_task'],ref['unseen_episode'],ref['unseen_frame']]).astype(np.int32)
    if learned.shape!=(621,3) or unseen.shape!=(619,3):raise AssertionError('fixed evaluation rows changed')
    sroot=root/'seg-context-01/unseen-stride5'
    task=np.load(sroot/'task.npy',mmap_mode='r');episode=np.load(sroot/'episode.npy',mmap_mode='r')
    frame=np.load(sroot/'frame.npy',mmap_mode='r');is_dev=np.load(sroot/'is_dev.npy',mmap_mode='r')
    dev=np.column_stack([task[is_dev],episode[is_dev],frame[is_dev]]).astype(np.int32)
    if len(dev)!=217:raise AssertionError(f'fixed Dev rows changed: {len(dev)}')
    if not np.array_equal(np.sort(np.asarray(selection['seen_row_ids'])),np.sort(ref['learned_row'])):
        raise AssertionError('learned rows disagree with fixed selection')
    return {'memory40':memory,'learned40':learned,'unseen90':unseen,'dev90':dev}


def _lt1_source_records(run_root: Path):
    data=json.loads((Path(run_root)/'data_manifest.json').read_text())
    records={(int(x['task']),int(x['episode'])):x for x in data['episodes']}
    task_names={int(k):str(v) for k,v in data['tasks'].items()}
    unseen=json.loads(Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/unseen-rows-02/manifest.json').read_text())
    hdf5={(int(x['task_id']),int(x['episode'])):x for x in unseen['source_records']}
    return records,task_names,hdf5


def _lt1_read_observation(key, *, domain, run_root, records, task_names, hdf5_records, parquet_cache):
    """Read one exact image/state/prompt row without consulting target actions."""
    import io
    import pyarrow.parquet as pq
    from PIL import Image
    t,e,f=map(int,key)
    if domain=='libero40':
        rec=records[(t,e)];path=Path(rec['path'])
        if path not in parquet_cache:
            table=pq.read_table(path,columns=['image','wrist_image','state'])
            parquet_cache[path]=table
        table=parquet_cache[path]
        imgs=[]
        for col in ('image','wrist_image'):
            obj=table[col][f].as_py()
            with Image.open(io.BytesIO(obj['bytes'])) as im:imgs.append(np.asarray(im.convert('RGB')))
        state=np.asarray(table['state'][f].as_py(),np.float32)
        prompt=task_names[t]
    else:
        rec=hdf5_records.get((t,e))
        if rec is None:raise KeyError(f'no corrected-orientation HDF5 record for query key {key}')
        import h5py
        with h5py.File(rec['source_hdf5'],'r') as fobj:
            obs=fobj['data'][rec['source_demo']]['obs'];idx=min(f*2,len(obs['agentview_rgb'])-1)
            imgs=[np.asarray(obs['agentview_rgb'][idx],np.uint8)[::-1,::-1],
                  np.asarray(obs['eye_in_hand_rgb'][idx],np.uint8)[::-1,::-1]]
            # The LIBERO conversion's 8-D state is ee_states(6)+gripper_states(2),
            # not position+quaternion (9-D); preserve the public converter layout.
            state=np.concatenate([np.asarray(obs['ee_states'][idx],np.float32),
                                  np.asarray(obs['gripper_states'][idx],np.float32)])
        prompt=str(rec.get('instruction',rec.get('task_name','')))
    return {'observation/image':imgs[0],'observation/wrist_image':imgs[1],
            'observation/state':state,'prompt':prompt}


def extract_lt1_ax_cache(*, output: Path, run_root: Path, checkpoint: Path | None=None,
                         batch_size: int=4, row_limit: int | None=None):
    """Build restartable complete AX tap cache for the four fixed row sets."""
    import io, os, sys, importlib.util
    import pyarrow.parquet as pq
    from PIL import Image
    from openpi.transforms import Normalize,ResizeImages,TokenizePrompt,PadStatesAndActions
    from openpi.policies.libero_policy import LiberoInputs
    from openpi.models.model import ModelType
    from openpi.shared.normalize import load as load_norm
    from openpi.shared.nnx_utils import module_jit
    from openpi.models.tokenizer import PaligemmaTokenizer
    run_root=Path(run_root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    sets=lt1_fixed_row_keys(run_root)
    records,task_names,hdf5_records=_lt1_source_records(run_root)
    groups={k:('libero40' if k.endswith('40') else 'libero90') for k in sets}
    cache_rows={};cache_domains={}
    for name,keys in sets.items():
        domain=groups[name]
        for key in keys:
            cache_rows[(domain,*map(int,key))]=None
        cache_domains[name]=keys
    all_keys=np.asarray([k[1:] for k in cache_rows],np.int32)
    all_domains=[k[0] for k in cache_rows]
    if row_limit is not None:
        all_keys=all_keys[:int(row_limit)];all_domains=all_domains[:int(row_limit)]
    manifest=output/'manifest.json';tok_path=output/'tokens.npy';done_path=output/'completed.npy'
    shape=(len(all_keys),3,3,10,256)
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old.get('keys_sha256')!=hashlib.sha256(all_keys.tobytes()+('|'.join(all_domains)).encode()).hexdigest():
            raise ValueError('AX cache row keys differ from requested fixed set')
        tokens=np.lib.format.open_memmap(tok_path,mode='r+',dtype=np.float16,shape=tuple(old['shape']))
        done=np.load(done_path,mmap_mode='r+')
    else:
        if any(output.iterdir()):raise FileExistsError(f'unrecognized partial AX cache directory: {output}')
        tokens=np.lib.format.open_memmap(tok_path,mode='w+',dtype=np.float16,shape=shape);tokens[:]=0
        done=np.lib.format.open_memmap(done_path,mode='w+',dtype=np.bool_,shape=(len(all_keys),));done[:]=False
        np.save(output/'row_keys.npy',all_keys,allow_pickle=False);np.save(output/'domain.npy',np.asarray(all_domains,dtype='U8'),allow_pickle=False)
        np.save(output/'groups.npy',np.full((len(all_keys),3,3,10),2,np.int8),allow_pickle=False)
        manifest.write_text(json.dumps({'contract':'lt1-ax-taps-v1','shape':list(shape),'dtype':'float16',
            'keys_sha256':hashlib.sha256(all_keys.tobytes()+('|'.join(all_domains)).encode()).hexdigest(),
            'sets':{k:len(v) for k,v in sets.items()},'row_limit':row_limit},indent=2)+'\n')
    if row_limit is None and len(all_keys)!=5973:raise AssertionError(f'expected 5,973 unique source rows, got {len(all_keys)}')
    pi05_dir=Path(__file__).resolve().parent.parent/'pi05_memory'
    sys.path.insert(0,str(pi05_dir))
    run_py=pi05_dir/'run.py';spec=importlib.util.spec_from_file_location('lt1_pi05_run',run_py)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    ckpt=Path(checkpoint or json.loads((run_root/'data_manifest.json').read_text())['checkpoint'])
    model,_=mod.load_policy(ckpt)
    encode=module_jit(model.prefix_features)
    norm_files=list((ckpt/'assets').rglob('norm_stats.json'))
    if len(norm_files)!=1:raise FileNotFoundError('expected unique π0.5 norm_stats.json')
    transforms=[LiberoInputs(ModelType.PI05),Normalize(load_norm(norm_files[0].parent),use_quantiles=True),
        ResizeImages(224,224),TokenizePrompt(PaligemmaTokenizer(200),discrete_state_input=False),PadStatesAndActions(32)]
    from collections import OrderedDict
    parquet_cache=OrderedDict()
    todo=np.flatnonzero(~np.asarray(done,bool))
    for start in range(0,len(todo),batch_size):
        ids=todo[start:start+batch_size];items=[]
        for ix in ids:
            row=_lt1_read_observation(all_keys[ix],domain=all_domains[ix],run_root=run_root,
                records=records,task_names=task_names,hdf5_records=hdf5_records,parquet_cache=parquet_cache)
            for tr in transforms:row=tr(row)
            items.append(row)
        packed={k:np.stack([x[k] for x in items]) for k in items[0]}
        obs=mod.obs_batch({'image_base':np.stack([x['image']['base_0_rgb'] for x in items]),
            'image_wrist':np.stack([x['image']['left_wrist_0_rgb'] for x in items]),
            'state':packed['state'],'prompt':packed['tokenized_prompt'],'prompt_mask':packed['tokenized_prompt_mask']},np.arange(len(items)))
        # The standard extractor owns a single prefix pass; use the exact observation
        # path in extract_ax_model_path and keep the result in a row-keyed memmap.
        _,grid=extract_ax_model_path(model,obs,all_keys[ids],seed=4101)
        if grid.shape!=(3,3,len(ids),10,256):raise ValueError(f'unexpected AX batch grid {grid.shape}')
        tokens[ids]=np.transpose(grid,(2,0,1,3,4))
        done[ids]=True
        if (start//batch_size)%25==0:tokens.flush();done.flush()
    tokens.flush();done.flush()
    if not np.asarray(done).all():raise RuntimeError(f'AX extraction incomplete: {int((~np.asarray(done)).sum())} rows')
    return {'rows':len(all_keys),'shape':list(shape),'complete':True,'cache':str(output)}


def extract_lt1_pt_cache(*, output: Path, run_root: Path, device='cuda', grid: int=10,
                         row_limit: int | None=None):
    """Build causal two-camera CoTracker3 cache, resumable per fixed source row."""
    import io, os
    import pyarrow.parquet as pq
    import h5py
    from PIL import Image
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    sets=lt1_fixed_row_keys(Path(run_root));records,task_names,hdf5_records=_lt1_source_records(Path(run_root))
    rowmap={}
    for name,keys in sets.items():
        domain='libero40' if name.endswith('40') else 'libero90'
        for k in keys:rowmap[(domain,*map(int,k))]=None
    all_keys=np.asarray([k[1:] for k in rowmap],np.int32);domains=[k[0] for k in rowmap]
    if row_limit is not None:all_keys=all_keys[:int(row_limit)];domains=domains[:int(row_limit)]
    digest=hashlib.sha256(all_keys.tobytes()+('|'.join(domains)).encode()).hexdigest()
    manifest=output/'manifest.json';tok_path=output/'tokens.npy';done_path=output/'completed.npy'
    shape=(len(all_keys),2*grid*grid,PT_HISTORY*3)
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old.get('keys_sha256')!=digest:raise ValueError('PT cache keys differ from requested fixed row set')
        tokens=np.lib.format.open_memmap(tok_path,mode='r+',dtype=np.float16,shape=tuple(old['shape']))
        done=np.load(done_path,mmap_mode='r+')
    else:
        if any(output.iterdir()):raise FileExistsError(f'unrecognized PT cache directory {output}')
        tokens=np.lib.format.open_memmap(tok_path,mode='w+',dtype=np.float16,shape=shape);tokens[:]=0
        done=np.lib.format.open_memmap(done_path,mode='w+',dtype=np.bool_,shape=(len(all_keys),));done[:]=False
        np.save(output/'row_keys.npy',all_keys,allow_pickle=False)
        np.save(output/'domain.npy',np.asarray(domains,dtype='U8'),allow_pickle=False)
        np.save(output/'groups.npy',np.tile(np.r_[np.zeros(grid*grid,np.int8),np.ones(grid*grid,np.int8)],(len(all_keys),1)),allow_pickle=False)
        manifest.write_text(json.dumps({'contract':'lt1-pt-cotracker-online-v1','shape':list(shape),
            'keys_sha256':digest,'sets':{k:len(v) for k,v in sets.items()},'row_limit':row_limit,
            'history_frames':PT_HISTORY,'grid_per_camera':grid,'image_orientation':'corrected 180 degree rotation for LIBERO-90; stored orientation for LIBERO-40',
            'causal':'all clips end at the query frame; no future frames'},indent=2)+'\n')
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','8')))
    if str(device).startswith('cuda'):
        dev=torch.device(device);torch.cuda.set_per_process_memory_fraction(float(os.environ.get('TORCH_MEM_FRACTION','.10')),dev.index or 0)
    model=torch.hub.load('facebookresearch/co-tracker','cotracker3_online',pretrained=True).to(device).eval()
    parquet_cache={}
    def read_clips(key,domain):
        t,e,f=map(int,key);idx=np.arange(f-PT_HISTORY+1,f+1,dtype=np.int64)
        actual=np.clip(idx,0,None)
        if domain=='libero40':
            path=Path(records[(t,e)]['path'])
            if path not in parquet_cache:parquet_cache[path]=pq.read_table(path,columns=['image','wrist_image'])
            table=parquet_cache[path];base=[];wrist=[]
            for j in actual:
                for col,out in (('image',base),('wrist_image',wrist)):
                    obj=table[col][int(j)].as_py()
                    with Image.open(io.BytesIO(obj['bytes'])) as im:out.append(np.asarray(im.convert('RGB')))
        else:
            rec=hdf5_records[(t,e)];base=[];wrist=[]
            with h5py.File(rec['source_hdf5'],'r') as fobj:
                obs=fobj['data'][rec['source_demo']]['obs'];n=len(obs['agentview_rgb'])
                for j in actual:
                    hidx=min(int(j)*2,n-1)
                    base.append(np.asarray(obs['agentview_rgb'][hidx],np.uint8)[::-1,::-1])
                    wrist.append(np.asarray(obs['eye_in_hand_rgb'][hidx],np.uint8)[::-1,::-1])
        def resize(xs):
            return np.stack([np.asarray(Image.fromarray(x).resize((128,128),Image.Resampling.BILINEAR),np.uint8) for x in xs])
        return resize(base),resize(wrist),idx
    todo=np.flatnonzero(~np.asarray(done,bool))
    for n,ix in enumerate(todo,1):
        base,wrist,frameix=read_clips(all_keys[ix],domains[ix])
        bt,bv=track_cotracker3(base,grid=grid,model=model,device=device)
        wt,wv=track_cotracker3(wrist,grid=grid,model=model,device=device)
        b,bg=tracks_to_tokens(bt,bv,group=0,image_size=(128,128));w,wg=tracks_to_tokens(wt,wv,group=1,image_size=(128,128))
        tokens[ix]=np.concatenate([b,w]);done[ix]=True
        if n%32==0:tokens.flush();done.flush()
    tokens.flush();done.flush()
    if not np.asarray(done).all():raise RuntimeError(f'PT extraction incomplete: {int((~np.asarray(done)).sum())}')
    return {'rows':len(all_keys),'shape':list(shape),'complete':True,'cache':str(output)}


def _source_rows(cache_dir: Path, domain: str):
    p=Path(cache_dir)
    keys=np.load(p/'row_keys.npy',mmap_mode='r',allow_pickle=False)
    domains=np.load(p/'domain.npy',mmap_mode='r',allow_pickle=False)
    tok=np.load(p/'tokens.npy',mmap_mode='r',allow_pickle=False)
    grp=np.load(p/'groups.npy',mmap_mode='r',allow_pickle=False)
    done=np.load(p/'completed.npy',mmap_mode='r',allow_pickle=False)
    if not np.all(done):raise RuntimeError(f'incomplete feature cache: {p}')
    ix=np.flatnonzero(domains==domain)
    return np.asarray(keys[ix]),tok[ix],grp[ix]


class LT1P0(nn.Module):
    """Shared mean-pool control for AX or PT source tokens."""
    def __init__(self,input_dim):
        super().__init__();self.encoder=MeanPoolMLP(input_dim);self.logit_scale=nn.Parameter(torch.tensor(float(np.log(1/.07))))
    def forward(self,tokens,groups):return self.encoder(tokens,groups)


def run_key2_source_condition(*, output: Path, condition: str, source: str,
                              ax_cache: Path | None, pt_cache: Path | None,
                              time_index: int=0, layer_index: int=0, p0=False,
                              vjepa_eval_cache: Path,
                              run_root=Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/run-01'),
                              missing_root=Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/key-shared-01/attempt-02/prefix-missing'),
                              seen_stride5=Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/seg-context-01/seg5-cache'),
                              unseen_stride5=Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/seg-context-01/unseen-stride5'),
                              seed=1,learning_rate=3e-4,epochs=16,teacher_temperature=.02,
                              dev_only=False,checkpoint=None,smoke=False):
    """Run the frozen KEY2 train/eval machinery with AX/PT rows injected.

    Data, targets, bank masks, metrics and objective are delegated to
    key_shared_latent.run_experiment. Only the frozen token source and model
    constructor are replaced. AX uses a fixed time/layer view selected on Dev.
    """
    import sys
    here=Path(__file__).resolve().parent
    sys.path[:0]=[str(here),str(here.parent/'pi05_memory')]
    import key_shared_latent as ks
    if source not in ('ax','pt','ax_pt'):raise ValueError(source)
    if source in ('ax','ax_pt') and ax_cache is None:raise ValueError('AX cache required')
    if source in ('pt','ax_pt') and pt_cache is None:raise ValueError('PT cache required')
    def load(cache,domain,is_ax):
        keys,tok,grp=_source_rows(cache,domain)
        if is_ax:
            if tok.ndim!=5 or tok.shape[1:3]!=(3,3):raise ValueError('AX cache must be [N,3,3,10,256]')
            tok=tok[:,int(time_index),int(layer_index)]
            grp=np.full(tok.shape[:2],2,np.int8)
        return keys,tok,grp
    ax40=load(ax_cache,'libero40',True) if source in ('ax','ax_pt') else None
    ax90=load(ax_cache,'libero90',True) if source in ('ax','ax_pt') else None
    pt40=load(pt_cache,'libero40',False) if source in ('pt','ax_pt') else None
    pt90=load(pt_cache,'libero90',False) if source in ('pt','ax_pt') else None
    original_builder=ks._build_key2_queries
    original_new=ks._new_model;original_encode=ks._model_batch_encode
    prepared={}
    def inject(part,domain):
        keys=np.asarray(part['key'])
        def exact(cache):
            ck,ct,cg=cache
            return align_feature_cache(keys,ck,ct,cg)
        if source in ('ax','ax_pt'):
            at,ag=exact(ax40 if domain=='libero40' else ax90)
            part['tokens']=at;part['groups']=ag
        if source=='ax':
            # key_shared_latent's base builder always attaches its V-JEPA
            # feature block. AX-only conditions must not accidentally feed
            # that block to the encoder (which has no PT adapter).
            part.pop('vjepa_tokens',None);part.pop('vjepa_groups',None)
        if source=='pt':
            pt,pg=exact(pt40 if domain=='libero40' else pt90)
            part['tokens']=pt;part['groups']=pg
            part.pop('vjepa_tokens',None);part.pop('vjepa_groups',None)
        elif source=='ax_pt':
            pt,pg=exact(pt40 if domain=='libero40' else pt90)
            part['vjepa_tokens']=pt;part['vjepa_groups']=pg
        return part
    def builder(*args,**kwargs):
        if 'data' not in prepared:
            vals=original_builder(*args,**kwargs)
            bank,bank_input,fit_bank,fit_bank_input,train,learned,unseen,dev=vals
            inject(bank_input,'libero40');inject(fit_bank_input,'libero40');inject(train,'libero40')
            inject(learned,'libero40');inject(unseen,'libero90');inject(dev,'libero90')
            prepared['data']=vals
        return prepared['data']
    mode='p0' if p0 else condition
    if p0:
        if source not in ('ax','pt'):raise ValueError('P0 is separately defined for AX and PT')
        cache40=ax40 if source=='ax' else pt40
        model_dim=int(cache40[1].shape[-1])
    def make_model(which,token_count=5):
        if which=='p0':return LT1P0(model_dim)
        return make_condition_encoder(condition,token_count=token_count)
    def encode(model,which,part,device,*,key=False):
        if which=='p0':
            tok=torch.as_tensor(part['tokens'],device=device,dtype=torch.float32)
            grp=torch.as_tensor(part['groups'],device=device,dtype=torch.long)
            return model(tok,grp)
        def tensor(k,dtype):
            v=part.get(k);return None if v is None else torch.as_tensor(v,device=device,dtype=dtype)
        return model(tokens=tensor('tokens',torch.float32),groups=tensor('groups',torch.long),
            history=tensor('history',torch.float32),history_mask=tensor('history_mask',torch.bool),
            state=tensor('state',torch.float32),vjepa_tokens=tensor('vjepa_tokens',torch.float32),
            vjepa_groups=tensor('vjepa_groups',torch.long))
    ks._build_key2_queries=builder;ks._new_model=make_model;ks._model_batch_encode=encode
    try:
        result=ks.run_experiment(Path(output),mode=mode,seed=seed,learning_rate=learning_rate,epochs=epochs,
            objective='soft',teacher_temperature=teacher_temperature,dev_only=dev_only,
            smoke=smoke,steps_per_epoch=2 if smoke else None,
            eval_only_checkpoint=checkpoint,vjepa_eval_cache=Path(vjepa_eval_cache),
            run_root=Path(run_root),missing_root=Path(missing_root),seen_stride5=Path(seen_stride5),
            unseen_stride5=Path(unseen_stride5))
        if p0:input_fields=[f'mean-pooled {source.upper()} source tokens']
        elif source=='ax_pt':input_fields=['AX action-expert V-cache tokens and groups','PT CoTracker tokens and camera groups',
            'strictly prior executed-action history','history-valid mask','current state']
        else:
            input_fields=[f'{source.upper()} source tokens and groups']
            if condition.endswith('+H'):input_fields+=['strictly prior executed-action history','history-valid mask','current state']
        result['input_fields']=input_fields
        audit=result.get('leakage_audit',{})
        audit['query_model_fields']=input_fields
        audit['memory_key_fields']=['unit-start '+source.upper()+' tokens/group IDs']
        if not p0 and (condition.endswith('+H') or source=='ax_pt'):
            audit['memory_key_fields']+=['prior executed-action history','unit-start state']
        audit['excluded_from_query_model']=['future actions','target continuation','task ID','episode ID','subtask label',
            'ground-truth boundary','π0.5 predicted action output']
        result['leakage_audit']=audit
        # KEY2's shared runner has no native AX/PT mode labels, so repair its
        # generic VLM audit fields in these newly created LT1-owned outputs.
        ks._atomic_json(Path(output)/('dev_results.json' if dev_only else 'results.json'),ks._without_private(result))
        if dev_only:
            rows=[]
            for name, summary in result['models'].items():
                rank=summary['dev_selection']['rank']
                rows.append(f"- {name}: Dev top-1 arm6 MSE={rank['top1_arm6_mse']:.6f}; "
                            f"Recall@1={rank.get('recall_at_1', float('nan')):.4f}; "
                            f"Spearman={rank.get('spearman_score_vs_negative_arm6_mse', float('nan')):.4f}")
            (Path(output)/'report.md').write_text(
                f"# LT1 Dev-only selection candidate\n\n"
                f"Condition: {condition}; source: {source}; time_index: {time_index}; "
                f"layer_index: {layer_index}; seed: {seed}.\n\n"+"\n".join(rows)+"\n")
        else:
            (Path(output)/'report.md').write_text(ks._render_report(result)+'\n')
        return result
    finally:
        ks._build_key2_queries=original_builder;ks._new_model=original_new;ks._model_batch_encode=original_encode


def _completed_source_cache(path: Path):
    path=Path(path)
    manifest=json.loads((path/'manifest.json').read_text())
    keys=np.load(path/'row_keys.npy',mmap_mode='r',allow_pickle=False)
    domains=np.load(path/'domain.npy',mmap_mode='r',allow_pickle=False)
    done=np.load(path/'completed.npy',mmap_mode='r',allow_pickle=False)
    if len(keys)!=5973 or len(domains)!=5973 or not np.asarray(done).all():
        raise RuntimeError(f'LT1 cache is incomplete: {path} ({int(np.asarray(done).sum())}/{len(done)})')
    expected=lt1_fixed_row_keys()
    index={(str(d),*map(int,k)) for d,k in zip(domains,keys,strict=True)}
    for label,wanted in expected.items():
        domain='libero40' if label.endswith('40') else 'libero90'
        missing=[(domain,*map(int,k)) for k in wanted if (domain,*map(int,k)) not in index]
        if missing:raise AssertionError(f'{path} lacks exact {label} row keys: {missing[:3]}')
    required=(5973,3,3,10,256) if manifest.get('contract')=='lt1-ax-taps-v1' else (5973,200,48)
    if tuple(manifest.get('shape',()))!=required:
        raise ValueError(f'LT1 cache has unexpected source/shape: {manifest.get("contract")} {manifest.get("shape")}')
    return manifest


def _unique_run_dir(path: Path):
    """Pick an unused attempt directory, preserving any interrupted output."""
    path=Path(path)
    if not path.exists() or not any(path.iterdir()):return path
    # Completed stage artifacts are immutable and can be reused on restart.
    # The caller checks the corresponding JSON before deciding to execute.
    if (path/'dev_results.json').is_file() or (path/'results.json').is_file():return path
    n=1
    while True:
        candidate=path.with_name(f'{path.name}-retry{n:02d}')
        if not candidate.exists() or not any(candidate.iterdir()):return candidate
        if (candidate/'dev_results.json').is_file() or (candidate/'results.json').is_file():return candidate
        n+=1


def run_lt1_production(*, output: Path, run_root: Path, vjepa_eval_cache: Path,
                       device='cuda', ax_batch_size=4, pt_device='cuda:0',
                       epochs=16, learning_rate=3e-4, teacher_temperature=.02,
                       extract=True):
    """Restartable LT1 production: exact caches, Dev-only selection, one Eval.

    AX time/layer settings are selected separately for each AX architecture by
    frozen Dev top-1 arm MSE. PT uses its single causal representation. P0
    controls use the E-AX-selected tap or the single PT feature set.
    """
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    existing_results=output/'results.json';existing_report=output/'report.md'
    if existing_results.exists():
        cached=json.loads(existing_results.read_text())
        if not existing_report.exists():existing_report.write_text(render_lt1_report(cached))
        return cached
    protocol_path=output/'protocol.json'
    protocol={'experiment':'LT1','seed':1,'ax_noise_seed':4101,'training':{'learning_rate':learning_rate,'epochs':epochs,
        'teacher_temperature':teacher_temperature,'objective':'soft-label InfoNCE'},
        'dev_selection':'top-1 arm6 MSE only; AX time x layer grid (3x3) per AX architecture',
        'eval':'fixed KEY2-F learned/unseen rows; run once after Dev selection; never used to select settings',
        'sources':{'AX':'fixed-noise pi05 action-expert suffix V-cache values at t=(1,.5,0), 0-based layers (2,8,16), horizon 10, width 256',
                   'PT':'CoTracker3 online two-camera 10x10 point grid over past 16 frames; displacement plus visibility',
                   'AX+PT':'shared KEY2 E, AX group 2, PT adapter from 48 to 1024 dimensions'},
        'row_sets':{'memory_units':4516,'learned_windows':621,'unseen_eval_windows':619,'dev_windows':217},
        'closed_loop':False}
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())!=protocol:
            raise RuntimeError('existing LT1 protocol differs from this run configuration')
    else:
        protocol_path.write_text(json.dumps(protocol,indent=2,sort_keys=True)+'\n')
    ax_cache=output/'features/ax_full';pt_cache=output/'features/pt_full'
    if extract:
        extract_lt1_ax_cache(output=ax_cache,run_root=run_root,batch_size=ax_batch_size)
        extract_lt1_pt_cache(output=pt_cache,run_root=run_root,device=pt_device)
    ax_manifest=_completed_source_cache(ax_cache);pt_manifest=_completed_source_cache(pt_cache)
    # Smoke-only fixture caches are structurally valid but must never become
    # production inputs.
    for path in (ax_cache,pt_cache):
        mf=json.loads((path/'manifest.json').read_text())
        if mf.get('row_limit') is not None or mf.get('smoke_fixture'):
            raise ValueError(f'refusing non-production LT1 cache: {path}')
    if Path(vjepa_eval_cache).resolve()!= (output/'features/vjepa_eval').resolve():
        # Keep one exact, explicitly named 619-row V-JEPA cache in this
        # production tree; no nearest-neighbour fill is allowed.
        src=Path(vjepa_eval_cache);dst=output/'features/vjepa_eval'
        if not dst.exists():
            dst.mkdir(parents=True)
            for name in ('row_keys.npy','tokens.npy','groups.npy','manifest.json'):
                import shutil
                shutil.copy2(src/name,dst/name)
        vjepa_eval_cache=dst
    vj=json.loads((Path(vjepa_eval_cache)/'manifest.json').read_text())
    if int(vj.get('rows',-1))!=619 or vj.get('complete') is not True:
        raise ValueError('the exact full Eval V-JEPA cache (619 rows) is required by shared KEY2 evaluator')
    selection_dir=output/'selection';selection_dir.mkdir(exist_ok=True)
    progress=output/'progress.jsonl'
    candidates=[]
    for condition,source in (('E-AX','ax'),('E-AX+H','ax'),('E-AX+PT+H','ax_pt')):
        for ti in range(3):
            for li in range(3):
                candidates.append({'setting':f'{condition}__t{ti}_l{li}','condition':condition,'source':source,
                                   'time_index':ti,'layer_index':li,'p0':False})
    for condition in ('E-PT','E-PT+H'):
        candidates.append({'setting':condition,'condition':condition,'source':'pt','time_index':0,'layer_index':0,'p0':False})
    # Candidate model/settings are selected once and only by Dev top-1 MSE.
    dev_rows=[]
    for spec in candidates:
        tag=spec['setting'].replace('+','_').replace('-','_')
        candidate_dir=_unique_run_dir(selection_dir/tag)
        dev_path=candidate_dir/'dev_results.json'
        if dev_path.exists():
            result=json.loads(dev_path.read_text())
        else:
            result=run_key2_source_condition(output=candidate_dir,condition=spec['condition'],source=spec['source'],
                ax_cache=ax_cache if spec['source'] in ('ax','ax_pt') else None,
                pt_cache=pt_cache if spec['source'] in ('pt','ax_pt') else None,
                time_index=spec['time_index'],layer_index=spec['layer_index'],
                vjepa_eval_cache=Path(vjepa_eval_cache),run_root=run_root,seed=1,
                learning_rate=learning_rate,epochs=epochs,teacher_temperature=teacher_temperature,dev_only=True)
        model_summary=result['models'][spec['condition']]
        metric=float(model_summary['dev_selection']['rank']['top1_arm6_mse'])
        dev_rows.append({**spec,'flow_time':AX_TIMES[spec['time_index']],
                         'action_expert_layer':AX_LAYER_INDICES[spec['layer_index']]+1,
                         'top1_arm_mse':metric,'checkpoint':model_summary['checkpoint'],
                         'run_dir':str(candidate_dir)})
        with progress.open('a') as f:
            f.write(json.dumps({'stage':'dev_candidate_complete','setting':spec['setting'],
                'top1_arm_mse':metric,'run_dir':str(candidate_dir)},sort_keys=True)+'\n')
    selected={}
    for condition in ('E-AX','E-AX+H','E-AX+PT+H','E-PT','E-PT+H'):
        pool=[x for x in dev_rows if x['condition']==condition]
        chosen=choose_dev_setting(pool)
        selected[condition]=chosen
    # Mean-pool controls are not allowed to alter the selected AX taps.
    p0_specs=[{'setting':'P0-AX','condition':'E-AX','source':'ax','time_index':selected['E-AX']['time_index'],
               'layer_index':selected['E-AX']['layer_index'],'p0':True},
              {'setting':'P0-PT','condition':'E-PT','source':'pt','time_index':0,'layer_index':0,'p0':True}]
    for spec in p0_specs:
        tag=spec['setting'];candidate_dir=_unique_run_dir(selection_dir/tag);dev_path=candidate_dir/'dev_results.json'
        if dev_path.exists():result=json.loads(dev_path.read_text())
        else:
            result=run_key2_source_condition(output=candidate_dir,condition=spec['condition'],source=spec['source'],
                ax_cache=ax_cache if spec['source']=='ax' else None,pt_cache=pt_cache if spec['source']=='pt' else None,
                time_index=spec['time_index'],layer_index=spec['layer_index'],p0=True,
                vjepa_eval_cache=Path(vjepa_eval_cache),run_root=run_root,seed=1,learning_rate=learning_rate,
                epochs=epochs,teacher_temperature=teacher_temperature,dev_only=True)
        metric=float(result['models']['p0']['dev_selection']['rank']['top1_arm6_mse'])
        dev_rows.append({**spec,'flow_time':AX_TIMES[spec['time_index']] if spec['source']=='ax' else None,
                         'action_expert_layer':AX_LAYER_INDICES[spec['layer_index']]+1 if spec['source']=='ax' else None,
                         'top1_arm_mse':metric,'checkpoint':result['models']['p0']['checkpoint'],
                         'run_dir':str(candidate_dir)})
        with progress.open('a') as f:f.write(json.dumps({'stage':'dev_candidate_complete','setting':tag,
            'top1_arm_mse':metric,'run_dir':str(candidate_dir)},sort_keys=True)+'\n')
    selected['P0-AX']=choose_dev_setting([x for x in dev_rows if x['setting']=='P0-AX'])
    selected['P0-PT']=choose_dev_setting([x for x in dev_rows if x['setting']=='P0-PT'])
    # Freeze selection before any fixed evaluation result is opened.
    selection_doc={'selection_metric':'Dev top-1 arm MSE','dev_rows':dev_rows,'selected':selected,
        'feature_caches':{'AX':ax_manifest,'PT':pt_manifest,'VJEPA_eval':vj},'seed':1,
        'training':{'epochs':epochs,'learning_rate':learning_rate,'teacher_temperature':teacher_temperature}}
    selection_path=output/'dev_selection.json'
    if selection_path.exists():
        previous=json.loads(selection_path.read_text())
        if previous.get('selected')!=selection_doc['selected']:
            raise RuntimeError('locked LT1 Dev selection differs from the existing production selection')
    else:
        tmp=selection_path.with_suffix('.tmp');tmp.write_text(json.dumps(selection_doc,indent=2,sort_keys=True)+'\n');tmp.replace(selection_path)
    final_dir=output/'final';final_dir.mkdir(exist_ok=True);final={}
    for label,spec in selected.items():
        tag=label.replace('+','_').replace('-','_')
        eval_dir=_unique_run_dir(final_dir/tag);result_path=eval_dir/'results.json'
        if result_path.exists():result=json.loads(result_path.read_text())
        else:
            is_p0=bool(spec.get('p0',False))
            result=run_key2_source_condition(output=eval_dir,condition=spec['condition'],source=spec['source'],
                ax_cache=ax_cache if spec['source'] in ('ax','ax_pt') else None,
                pt_cache=pt_cache if spec['source'] in ('pt','ax_pt') else None,
                time_index=spec['time_index'],layer_index=spec['layer_index'],p0=is_p0,
                vjepa_eval_cache=Path(vjepa_eval_cache),run_root=run_root,seed=1,
                learning_rate=learning_rate,epochs=epochs,teacher_temperature=teacher_temperature,
                checkpoint=Path(spec['checkpoint']),dev_only=False)
        final[label]={'setting':spec,'results':result,'output':str(eval_dir)}
        with progress.open('a') as f:f.write(json.dumps({'stage':'eval_complete','condition':label,
            'output':str(eval_dir)},sort_keys=True)+'\n')
    aggregate={'status':'complete','experiment':'LT1','selection':selection_doc,'final':final,
        'reference_results':_lt1_reference_metrics(),
        'paired_reference_comparisons_5000':_lt1_paired_reference_comparisons(final),
        'latent_quality_probes':_lt1_latent_quality_probes(run_root,ax_cache,pt_cache,
            selected['E-AX+H'],selected['E-AX+PT+H']),
        'limitations':{'AX_representation':'per-layer action-expert V-cache values; residual hidden states are not exposed by OpenPI scanned Gemma block API',
            'PT':'CoTracker3 online, 16-frame causal clips, 10x10 points per camera','closed_loop':False}}
    result_path=output/'results.json'
    if result_path.exists():raise FileExistsError(f'refusing to overwrite existing production results: {result_path}')
    tmp=result_path.with_suffix('.tmp');tmp.write_text(json.dumps(_jsonable_lt1(aggregate),indent=2,sort_keys=True)+'\n');tmp.replace(result_path)
    report=render_lt1_report(aggregate)
    report_path=output/'report.md'
    if report_path.exists():raise FileExistsError(f'refusing to overwrite existing production report: {report_path}')
    report_path.write_text(report)
    return aggregate


def _jsonable_lt1(value):
    if isinstance(value,dict):return {str(k):_jsonable_lt1(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_jsonable_lt1(x) for x in value]
    if isinstance(value,(np.integer,np.floating,np.bool_)):return value.item()
    if isinstance(value,Path):return str(value)
    return value


def _lt1_reference_metrics():
    path=Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/key-shared-01/attempt-27-production/results.json')
    data=json.loads(path.read_text())
    fixed=json.loads((path.parent/'fixed-baselines/baselines.json').read_text())
    refs={}
    for split in ('learned','unseen'):
        for name in ('e_vlm_h','p5'):
            block=data['methods'][split][name]
            refs.setdefault(name,{})[split]={'top1_arm6_mse':block['rank_all']['top1_arm6_mse'],
                'top20_arm6_mse':block['rank_all']['top20_arm6_mse'],
                'blend20_arm6_mse':block['actions_all']['arm6'],
                'spearman_positive':block['rank_all']['spearman_positive'],
                'cross_entropy':block['rank_all']['cross_entropy'],'kl_pq':block['rank_all']['kl_pq'],
                'recall@1':block['rank_all']['recall@1'],'recall@5':block['rank_all']['recall@5'],'recall@20':block['rank_all']['recall@20'],
                'ndcg@20':block['rank_all']['ndcg@20'],'mrr':block['rank_all']['mrr'],'recovery':block['rank_all']['recovery'],
                'hard_top1_arm6_mse':block['rank_hard']['top1_arm6_mse'],
                'hard_top20_arm6_mse':block['rank_hard']['top20_arm6_mse'],
                'hard_blend20_arm6_mse':block['actions_hard']['arm6'],
                'hard_spearman_positive':block['rank_hard']['spearman_positive'],
                'hard_cross_entropy':block['rank_hard']['cross_entropy'],'hard_kl_pq':block['rank_hard']['kl_pq'],
                'hard_recall@1':block['rank_hard']['recall@1'],'hard_recall@5':block['rank_hard']['recall@5'],
                'hard_recall@20':block['rank_hard']['recall@20'],'hard_ndcg@20':block['rank_hard']['ndcg@20'],
                'hard_mrr':block['rank_hard']['mrr'],'hard_recovery':block['rank_hard']['recovery']}
    for split in ('learned','unseen'):
        b=fixed['metrics'][split]
        refs.setdefault('random',{})[split]={'top1_arm6_mse':b['random_rank']['top1_arm6_mse'],
            'top20_arm6_mse':b['random_rank']['top20_arm6_mse'],'blend20_arm6_mse':b['baselines']['random20_blend']['arm6'],
            'spearman_positive':b['random_rank']['spearman_positive'],'cross_entropy':b['random_rank']['cross_entropy'],
            'kl_pq':b['random_rank']['kl_pq'],'recall@1':b['random_rank']['recall@1'],
            'recall@5':b['random_rank']['recall@5'],'recall@20':b['random_rank']['recall@20'],
            'ndcg@20':b['random_rank']['ndcg@20'],'mrr':b['random_rank']['mrr'],'recovery':b['random_rank']['recovery'],
            'hard_top1_arm6_mse':b['random_rank_hard']['top1_arm6_mse'],
            'hard_top20_arm6_mse':b['random_rank_hard']['top20_arm6_mse'],
            'hard_blend20_arm6_mse':b['baselines_hard']['random20_blend']['arm6'],
            'hard_spearman_positive':b['random_rank_hard']['spearman_positive'],
            'hard_cross_entropy':b['random_rank_hard']['cross_entropy'],'hard_kl_pq':b['random_rank_hard']['kl_pq'],
            'hard_recall@1':b['random_rank_hard']['recall@1'],'hard_recall@5':b['random_rank_hard']['recall@5'],
            'hard_recall@20':b['random_rank_hard']['recall@20'],'hard_ndcg@20':b['random_rank_hard']['ndcg@20'],
            'hard_mrr':b['random_rank_hard']['mrr'],'hard_recovery':b['random_rank_hard']['recovery']}
        refs.setdefault('oracle',{})[split]={'top1_arm6_mse':b['oracle_rank']['top1_arm6_mse'],
            'top20_arm6_mse':b['oracle_rank']['top20_arm6_mse'],'blend20_arm6_mse':None,
            'spearman_positive':b['oracle_rank']['spearman_positive'],'cross_entropy':b['oracle_rank']['cross_entropy'],
            'kl_pq':b['oracle_rank']['kl_pq'],'recall@1':b['oracle_rank']['recall@1'],
            'recall@5':b['oracle_rank']['recall@5'],'recall@20':b['oracle_rank']['recall@20'],
            'ndcg@20':b['oracle_rank']['ndcg@20'],'mrr':b['oracle_rank']['mrr'],'recovery':b['oracle_rank']['recovery'],
            'hard_top1_arm6_mse':b['oracle_rank_hard']['top1_arm6_mse'],'hard_blend20_arm6_mse':None,
            'hard_top20_arm6_mse':b['oracle_rank_hard']['top20_arm6_mse'],
            'hard_spearman_positive':b['oracle_rank_hard']['spearman_positive'],
            'hard_cross_entropy':b['oracle_rank_hard']['cross_entropy'],'hard_kl_pq':b['oracle_rank_hard']['kl_pq'],
            'hard_recall@1':b['oracle_rank_hard']['recall@1'],'hard_recall@5':b['oracle_rank_hard']['recall@5'],
            'hard_recall@20':b['oracle_rank_hard']['recall@20'],'hard_ndcg@20':b['oracle_rank_hard']['ndcg@20'],
            'hard_mrr':b['oracle_rank_hard']['mrr'],'hard_recovery':b['oracle_rank_hard']['recovery']}
        for name in ('hold_last','pi05'):
            refs.setdefault(name,{})[split]={'top1_arm6_mse':b['baselines'][name]['arm6'],
                'top20_arm6_mse':None,'blend20_arm6_mse':None,'spearman_positive':None,'recall@1':None,'recall@20':None,
                'hard_top1_arm6_mse':b['baselines_hard'][name]['arm6'],'hard_blend20_arm6_mse':None,
                'hard_top20_arm6_mse':None,
                'hard_spearman_positive':None,'hard_recall@1':None,'hard_recall@5':None,'hard_recall@20':None,
                'cross_entropy':None,'kl_pq':None,'recall@5':None,'ndcg@20':None,'mrr':None,'recovery':None,
                'hard_cross_entropy':None,'hard_kl_pq':None,'hard_ndcg@20':None,'hard_mrr':None,'hard_recovery':None}
    return refs


def _lt1_latent_quality_probes(run_root,ax_cache,pt_cache,ax_spec,combined_spec):
    """Five-fold LIBERO-40-only source-feature probes (diagnostic, not selection)."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression, Ridge
    import sys
    here=Path(__file__).resolve().parent
    sys.path[:0]=[str(here),str(here.parent/'pi05_memory')]
    from key_shared_latent import _load_key2_rows
    import gripper_split as grip
    import motion_core_eval as motion
    fixed=lt1_fixed_row_keys(run_root)['memory40']
    ak,at,ag=_source_rows(ax_cache,'libero40');pk,pt,pg=_source_rows(pt_cache,'libero40')
    ax,_=align_feature_cache(fixed,ak,at,ag)
    pt,_=align_feature_cache(fixed,pk,pt,pg)
    ax=np.asarray(ax[:,int(ax_spec['time_index']),int(ax_spec['layer_index'])],np.float32)
    ax_combo=np.asarray(np.asarray(align_feature_cache(fixed,ak,at,ag)[0])[:,
        int(combined_spec['time_index']),int(combined_spec['layer_index'])],np.float32)
    pt=np.asarray(pt,np.float32)
    # Mean/std summaries are fixed, label-free reductions of each source's
    # token set; they are diagnostic representations, not retrieval keys.
    axf=np.concatenate([ax.mean(1),ax.std(1)],axis=1)
    axcf=np.concatenate([ax_combo.mean(1),ax_combo.std(1)],axis=1)
    ptf=np.concatenate([pt.mean(1),pt.std(1)],axis=1)
    rows=_load_key2_rows(run_root);mids=np.flatnonzero(np.asarray(rows['memory'],bool))
    traces=grip._traces(rows,mids);actions_by_key={};tasks_by_key={}
    for (task,episode),row_ids in traces.items():
        trace=np.concatenate([np.asarray(rows['actions'][i,:,:7],np.float32) for i in row_ids])
        frame0=int(rows['frame'][row_ids[0]])
        for start,end in motion.anchor_spans(trace,min_length=5):
            key=(int(task),int(episode),frame0+int(start))
            window=trace[int(start):min(int(start)+10,len(trace)),:6]
            if len(window):
                disp=window.sum(0);norm=float(np.linalg.norm(disp))
                if norm>1e-7:actions_by_key[key]=disp/norm
                tasks_by_key[key]=int(task)
    ordered=[tuple(map(int,k)) for k in fixed]
    if any(k not in tasks_by_key for k in ordered):
        raise AssertionError('missing exact memory-unit target for LT1 linear probes')
    y=np.asarray([tasks_by_key[k] for k in ordered],np.int32)
    direction=np.asarray([actions_by_key.get(k,np.zeros(6,np.float32)) for k in ordered],np.float32)
    valid=np.linalg.norm(direction,axis=1)>1e-7
    classes,counts=np.unique(y,return_counts=True)
    folds=min(5,int(counts.min()))
    if folds<2:raise RuntimeError('not enough examples per task for task-ID probe')
    cv=StratifiedKFold(n_splits=folds,shuffle=True,random_state=4101)
    reports={}
    for name,feat in [('AX',axf),('PT',ptf),('AX+PT',np.concatenate([axcf,ptf],axis=1))]:
        classifier=make_pipeline(StandardScaler(),LogisticRegression(max_iter=500,C=1.0))
        pred=cross_val_predict(classifier,feat,y,cv=cv,n_jobs=1)
        direction_pred=np.zeros_like(direction)
        reg_cv=StratifiedKFold(n_splits=folds,shuffle=True,random_state=4101)
        for tr,te in reg_cv.split(feat,y):
            model=make_pipeline(StandardScaler(),Ridge(alpha=10.0))
            model.fit(feat[tr],direction[tr]);direction_pred[te]=model.predict(feat[te])
        denom=np.linalg.norm(direction_pred[valid],axis=1)*np.linalg.norm(direction[valid],axis=1)
        cos=np.sum(direction_pred[valid]*direction[valid],axis=1)/np.maximum(denom,1e-9)
        tden=np.linalg.norm(direction_pred[valid,:3],axis=1)*np.linalg.norm(direction[valid,:3],axis=1)
        tcos=np.sum(direction_pred[valid,:3]*direction[valid,:3],axis=1)/np.maximum(tden,1e-9)
        reports[name]={'task_id_5fold_accuracy':float(np.mean(pred==y)),
            'future_arm6_direction_5fold_cosine':float(np.mean(cos)),
            'future_translation_direction_5fold_cosine':float(np.mean(tcos)),
            'rows':int(len(y)),'tasks':int(len(classes)),'direction_rows':int(valid.sum()),
            'cross_validation':'stratified 5-fold on LIBERO-40 memory unit starts; diagnostic only'}
    return reports


def _lt1_paired_reference_comparisons(final):
    import sys
    here=Path(__file__).resolve().parent
    sys.path.insert(0,str(here))
    import key_shared_latent as ks
    root=Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/key-shared-01/attempt-27-production/final')
    out={}
    for label,item in final.items():
        mode='p0' if 'p0' in item['results']['models'] else item['setting']['condition']
        prefix='p0' if mode=='p0' else mode
        eval_dir=Path(item['output'])
        out[label]={}
        for split in ('learned','unseen'):
            own=np.load(eval_dir/f'{prefix}_{split}_paired_predictions.npz',allow_pickle=False)
            own_action=np.asarray(own['top1_actions'])
            target=np.asarray(own['target_actions']);task=np.asarray(own['task']);episode=np.asarray(own['episode'])
            hard=np.asarray(own['hard'],bool);key=np.asarray(own['key'])
            out[label][split]={}
            own_err=ks.action_errors(own_action,target)
            for ref in ('p5','e_vlm_h'):
                other=np.load(root/ref/f'{ref}_{split}_paired_predictions.npz',allow_pickle=False)
                for field in ('target_actions','task','episode','hard','key'):
                    if not np.array_equal(own[field],other[field]):
                        raise AssertionError(f'paired-row mismatch {label}/{split}/{ref}/{field}')
                ref_err=ks.action_errors(other['top1_actions'],target)
                val={}
                for metric in ('arm6','active7'):
                    d=np.asarray(own_err[metric])-np.asarray(ref_err[metric])
                    val[f'{metric}_mean']=float(d.mean())
                    val[metric]=ks._paired_bootstrap(d,task,episode,seed=4101)
                    if hard.any():
                        dh=d[hard]
                        val[f'{metric}_hard_mean']=float(dh.mean())
                        val[f'{metric}_hard']=ks._paired_bootstrap(dh,task[hard],episode[hard],seed=4102)
                out[label][split][ref]=val
    return out


def render_lt1_report(result):
    def fval(v):return '—' if v is None else f'{float(v):.5f}'
    rows=[]
    for label,item in result['final'].items():
        for split in ('learned','unseen'):
            m=item['results']['models']
            key='p0' if 'p0' in m else item['setting']['condition']
            block=m[key]['splits'][split]
            rank=block['rank_all'];hard=block['rank_hard']
            rows.append((label,split,rank['top1_arm6_mse'],hard['top1_arm6_mse'],
                         block['actions_all']['arm6'],block['actions_hard']['arm6'],
                         rank['recall@1'],hard['recall@1'],rank['recall@20'],hard['recall@20'],
                         rank['spearman_positive'],hard['spearman_positive']))
    lines=['# LT1: 潜在トークン検索（全件）','',
        '## 主表：固定 Eval を含む最終評価','',
        '値はグリッパー正解を使った腕6軸 MSE。top-1 は Dev の top-1 腕 MSE で選択した単一候補。top-20 は Dev で別途選んだブレンド。',
        'KEY2-F の P5 と E-VLM+H を参照行として併記する。未学習は LIBERO-90 Eval 619 窓。','',
        '|方式|split|top-1全体|top-1難|top-20ブレンド全体|top-20ブレンド難|Recall@1 全体/難|Recall@20 全体/難|Spearman(+) 全体/難|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    lines += [f'|{a}|{b}|{c:.5f}|{d:.5f}|{e:.5f}|{f:.5f}|{g:.4f}/{h:.4f}|{i:.4f}/{j:.4f}|{k:.4f}/{l:.4f}|' for a,b,c,d,e,f,g,h,i,j,k,l in rows]
    lines += ['', '### 候補順位・分布指標（全体／難）','',
        '|方式|split|対象|Spearman(+)|CE H(p,q)|KL(p‖q)|Recall@1/5/20|NDCG@20|MRR|top1/top20 候補MSE|回収率|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label,item in result['final'].items():
        model_key='p0' if 'p0' in item['results']['models'] else item['setting']['condition']
        for split in ('learned','unseen'):
            block=item['results']['models'][model_key]['splits'][split]
            for group,rk in (('all',block['rank_all']),('hard',block['rank_hard'])):
                if rk is None:continue
                recalls='/'.join(f"{rk[k]:.4f}" for k in ('recall@1','recall@5','recall@20'))
                lines.append(f"|{label}|{split}|{group}|{rk['spearman_positive']:.4f}|{rk['cross_entropy']:.4f}|{rk['kl_pq']:.4f}|{recalls}|{rk['ndcg@20']:.4f}|{rk['mrr']:.4f}|{rk['top1_arm6_mse']:.5f}/{rk['top20_arm6_mse']:.5f}|{rk['recovery']:.4f}|")
    for name,blocks in result['reference_results'].items():
        if name in ('hold_last','pi05'):continue
        for split,b in blocks.items():
            for group,prefix in (('all',''),('hard','hard_')):
                recalls='/'.join(f"{b.get(prefix+k) if prefix else b.get(k):.4f}" for k in ('recall@1','recall@5','recall@20'))
                lines.append(f"|{name}|{split}|{group}|{b.get(prefix+'spearman_positive'):.4f}|{b.get(prefix+'cross_entropy'):.4f}|{b.get(prefix+'kl_pq'):.4f}|{recalls}|{b.get(prefix+'ndcg@20'):.4f}|{b.get(prefix+'mrr'):.4f}|{b.get(prefix+'top1_arm6_mse'):.5f}/{b.get(prefix+'top20_arm6_mse'):.5f}|{b.get(prefix+'recovery'):.4f}|")
    lines += ['', '## Dev による時間・層選択','', '|方式|flow t|action expert 層|Dev top-1 腕 MSE|', '|---|---:|---:|---:|']
    for k,v in result['selection']['selected'].items():
        t='—' if v.get('flow_time') is None else f"{v['flow_time']:.1f}"
        layer='—' if v.get('action_expert_layer') is None else str(v['action_expert_layer'])
        lines.append(f"|{k}|{t}|{layer}|{v['top1_arm_mse']:.6f}|")
    lines += ['', '## 潜在の質・参照条件','',
        '- AX は残差ストリームではなく action expert 各層の suffix V-cache 値を使用した。OpenPI の scanned Gemma block は residual hidden states を外部へ返さず、KV キャッシュの V は同じ前向き計算から得られる内部表現である。',
        '- PT は CoTracker3 online 版の過去16フレームのみを使用し、base/wrist 各100点の変位・可視性トークン。',
        '- 参照 KEY2-F の P5 と E-VLM+H は attempt-27-production の固定結果をそのまま参照。','',
        '|入力要約|タスクID 5-fold 正解率|未来腕6方向 cos|未来並進方向 cos|行数|','|---|---:|---:|---:|---:|']
    for name,b in result['latent_quality_probes'].items():
        lines.append(f"|{name}|{b['task_id_5fold_accuracy']:.4f}|{b['future_arm6_direction_5fold_cosine']:.4f}|{b['future_translation_direction_5fold_cosine']:.4f}|{b['rows']}|")
    lines += ['',
        '|参照|split|top-1 全体/難|top-20ブレンド 全体/難|Recall@1 全体/難|Recall@20 全体/難|Spearman(+) 全体/難|',
        '|---|---|---:|---:|---:|---:|---:|']
    for name,blocks in result['reference_results'].items():
        for split,b in blocks.items():
            lines.append(f"|{name}|{split}|{fval(b['top1_arm6_mse'])}/{fval(b['hard_top1_arm6_mse'])}|{fval(b['blend20_arm6_mse'])}/{fval(b['hard_blend20_arm6_mse'])}|{fval(b['recall@1'])}/{fval(b['hard_recall@1'])}|{fval(b['recall@20'])}/{fval(b['hard_recall@20'])}|{fval(b['spearman_positive'])}/{fval(b['hard_spearman_positive'])}|")
    lines += ['', '## 参考条件との差（対応付き episode bootstrap 95% CI）','',
        '差は LT1 top-1 − 参照 top-1。負値が LT1 側の誤差が小さいことを表す。','',
        '|方式|split|参照|arm6 平均差 [95% CI]|hard arm6 平均差 [95% CI]|','|---|---|---|---:|---:|']
    for label,splits in result['paired_reference_comparisons_5000'].items():
        for split,refs in splits.items():
            for ref,v in refs.items():
                ci=v['arm6'];hci=v['arm6_hard']
                lines.append(f"|{label}|{split}|{ref}|{v['arm6_mean']:.5f} [{ci['ci95_low']:.5f}, {ci['ci95_high']:.5f}]|{v['arm6_hard_mean']:.5f} [{hci['ci95_low']:.5f}, {hci['ci95_high']:.5f}]|")
    lines += ['', '## 手順・監査','',
        '- データは KEY2-F と同一の学習済み621窓、未学習619窓、Dev217窓、メモリ単位4,516個。task/episode は候補除外と分割監査だけに使用。',
        '- 各条件の time/layer は Dev top-1 腕 MSE の最小値で選び、選択を書き出してから Eval checkpoint を一度だけ評価した。Eval は設定選択に使用しない。',
        '- クエリ入力は現在の観測由来の潜在、必要な条件では過去実行履歴と現在状態。π0.5 の予測行動、未来行動、タスクIDは入力しない。',
        '- 1 seed（1）。閉ループ評価ではない。','']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check-cache", type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--list-fixed-rows',action='store_true')
    p.add_argument('--extract-ax',action='store_true')
    p.add_argument('--extract-pt',action='store_true')
    p.add_argument('--production',action='store_true',help='run full restartable extraction, Dev selection and final Eval')
    p.add_argument('--skip-extract',action='store_true',help='production only: require completed AX/PT caches')
    p.add_argument('--run-root',type=Path,default=Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/run-01'))
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--device',default='cuda')
    p.add_argument('--row-limit',type=int,default=None,help='smoke only; never report as production')
    p.add_argument('--vjepa-eval-cache',type=Path,default=Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/latent-sources-01/production/features/vjepa_eval'))
    p.add_argument('--epochs',type=int,default=16)
    p.add_argument('--learning-rate',type=float,default=3e-4)
    p.add_argument('--teacher-temperature',type=float,default=.02)
    a = p.parse_args()
    if a.list_fixed_rows:
        rows=lt1_fixed_row_keys(a.run_root)
        print(json.dumps({k:{'rows':len(v),'sha256':hashlib.sha256(np.asarray(v,np.int32).tobytes()).hexdigest()}
                          for k,v in rows.items()},indent=2))
    elif a.extract_ax:
        if a.output is None:p.error('--extract-ax requires --output')
        print(json.dumps(extract_lt1_ax_cache(output=a.output,run_root=a.run_root,batch_size=a.batch_size,row_limit=a.row_limit),indent=2))
    elif a.extract_pt:
        if a.output is None:p.error('--extract-pt requires --output')
        print(json.dumps(extract_lt1_pt_cache(output=a.output,run_root=a.run_root,device=a.device,row_limit=a.row_limit),indent=2))
    elif a.check_cache:
        k, t, g, s, c = load_feature_cache(a.check_cache)
        print(json.dumps({"rows": len(k), "token_shape": list(t.shape), "source": s, "config": c}))
    elif a.production:
        if a.output is None:p.error('--production requires --output')
        if a.row_limit is not None:p.error('--row-limit is smoke-only and cannot be combined with --production')
        result=run_lt1_production(output=a.output,run_root=a.run_root,vjepa_eval_cache=a.vjepa_eval_cache,
            device=a.device,ax_batch_size=a.batch_size,pt_device=a.device,epochs=a.epochs,
            learning_rate=a.learning_rate,teacher_temperature=a.teacher_temperature,extract=not a.skip_extract)
        print(json.dumps({'status':result['status'],'selected':result['selection']['selected'],
                          'results':str(a.output/'results.json')},indent=2))
    else:
        p.error("select --list-fixed-rows, --extract-ax, --extract-pt, --check-cache or --production")


if __name__ == "__main__":
    main()
