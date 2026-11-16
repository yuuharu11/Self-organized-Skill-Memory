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
    if tok.ndim!=3 or grp.shape!=tok.shape[:2] or len(tok)!=len(available):
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check-cache", type=Path)
    a = p.parse_args()
    if a.check_cache:
        k, t, g, s, c = load_feature_cache(a.check_cache)
        print(json.dumps({"rows": len(k), "token_shape": list(t.shape), "source": s, "config": c}))
    else:
        p.error("full feature extraction is driven with an explicit frozen-model step_fn; use --check-cache for validation")


if __name__ == "__main__":
    main()
