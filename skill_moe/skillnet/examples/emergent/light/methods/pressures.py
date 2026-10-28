"""Composable training-time pressures shared across skill methods."""
from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations
from typing import Any

import numpy as np
import torch


def _to_tensor(x):
    return x if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float32)


def _batch_size(base) -> int:
    if isinstance(base, Mapping):
        for value in base.values():
            if value is not None:
                return _batch_size(value)
        raise ValueError("empty base mapping")
    if isinstance(base, (tuple, list)):
        return _batch_size(base[0])
    return len(base)


def _index(tree, indices: list[int]):
    if isinstance(tree, Mapping):
        return {k: None if v is None else _index(v, indices) for k, v in tree.items()}
    if isinstance(tree, tuple):
        return tuple(_index(v, indices) for v in tree)
    if isinstance(tree, list):
        return [_index(v, indices) for v in tree]
    ix = torch.as_tensor(indices, dtype=torch.long, device=tree.device) if torch.is_tensor(tree) else np.asarray(indices)
    return tree[ix]


def _discrete_columns(base: Any):
    """Return discrete code arrays when the method exposes code IDs."""
    if isinstance(base, Mapping):
        columns = []
        for key in ("code_id", "code", "codes", "coarse_id", "fine_id", "base_id", "indices"):
            value = base.get(key)
            if value is not None and (key.endswith("_id") or key in {"code", "codes", "indices"}):
                columns.append(np.asarray(value.detach().cpu() if torch.is_tensor(value) else value).reshape(len(value), -1))
        return columns or None
    if isinstance(base, (tuple, list)):
        cols = []
        for value in base:
            found = _discrete_columns(value)
            if found:
                cols.extend(found)
        return cols or None
    if torch.is_tensor(base):
        arr = base.detach().cpu().numpy()
    else:
        arr = np.asarray(base)
    if np.issubdtype(arr.dtype, np.integer):
        return [arr.reshape(len(arr), -1)]
    return None


def _continuous_array(base):
    """Flatten a structured continuous base while preserving its batch axis."""
    if isinstance(base, Mapping):
        parts = [_continuous_array(value) for _, value in sorted(base.items()) if value is not None]
        if not parts:
            raise ValueError("empty continuous base mapping")
        return np.concatenate(parts, axis=-1)
    if isinstance(base, (tuple, list)):
        return np.concatenate([_continuous_array(value) for value in base], axis=-1)
    if torch.is_tensor(base):
        return base.detach().cpu().float().reshape(len(base), -1).numpy()
    arr = np.asarray(base)
    return arr.reshape(len(arr), -1).astype(np.float32)


def _matching_pairs(base, *, max_base_distance: float | None) -> list[tuple[int, int]]:
    n = _batch_size(base)
    discrete = _discrete_columns(base)
    pairs = []
    if discrete:
        rows = [np.concatenate([col[i].reshape(-1) for col in discrete]) for i in range(n)]
        groups: dict[tuple, list[int]] = {}
        for i, row in enumerate(rows):
            groups.setdefault(tuple(row.tolist()), []).append(i)
        for group in groups.values():
            if len(group) > 1:
                # Ring neighbors bound construction to O(batch), even when a
                # code dominates the minibatch. The caller adds both directions.
                edges = {tuple(sorted((group[i], group[(i + 1) % len(group)]))) for i in range(len(group))}
                pairs.extend(sorted(edges))
        return pairs

    values = _continuous_array(base).reshape(n, -1)
    dist = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=-1)
    np.fill_diagonal(dist, np.inf)
    if max_base_distance is None:
        # Each row contributes only its closest neighbor; pairs are made unique.
        nearest = np.argmin(dist, axis=1)
        return sorted({tuple(sorted((i, int(j)))) for i, j in enumerate(nearest) if np.isfinite(dist[i, j])})
    for i, j in combinations(range(n), 2):
        if dist[i, j] <= max_base_distance:
            pairs.append((i, j))
    return pairs


def _m2_encode_with_grad(method, actions, state):
    """Use M2's fit-time modules directly; its public encode intentionally detaches to NumPy."""
    if not all(hasattr(method, name) for name in ("encoder", "codebook", "am", "sm", "embedding_dim", "residual_dim")):
        return None
    device = method.device
    a = actions.to(device).reshape(len(actions), -1)
    s = state.to(device)
    an = (a - torch.as_tensor(method.am, device=device)) / torch.as_tensor(method.ass, device=device)
    sn = (s - torch.as_tensor(method.sm, device=device)) / torch.as_tensor(method.ss, device=device)
    enc = method.encoder(torch.cat([an, sn], dim=-1))
    if callable(getattr(method, "_residual_stats", None)):
        z = enc[:, :method.embedding_dim]
        residual, _ = method._residual_stats(enc[:, method.embedding_dim:])
    else:
        z, residual = enc.split([method.embedding_dim, method.residual_dim], dim=-1)
    ids = torch.cdist(z, method.codebook).argmin(dim=-1)
    return ids, residual


def _m1_encode_with_grad(method, actions, state):
    if not all(hasattr(method, name) for name in ("encoder", "am", "ass", "sm", "ss", "a_dim")):
        return None
    device = method.device
    a = actions.to(device).reshape(len(actions), -1)
    s = state.to(device)
    an = (a - torch.as_tensor(method.am, device=device)) / torch.as_tensor(method.ass, device=device)
    sn = (s - torch.as_tensor(method.sm, device=device)) / torch.as_tensor(method.ss, device=device)
    latent = method.encoder(torch.cat([an, sn], dim=-1))
    base = latent[:, :method.base_dim]
    if callable(getattr(method, "_residual_stats", None)):
        residual, _ = method._residual_stats(latent[:, method.base_dim:])
    else:
        residual = latent[:, method.base_dim:]
    return base, residual


def _m45_encode_with_grad(method, actions, state):
    """Differentiable posterior for M4 prototypes and M5 sparse atoms."""
    if not all(hasattr(method, name) for name in ("net", "normalizers", "action_shape", "_normalise")):
        return None
    device = method.device
    a = actions.to(device).reshape(len(actions), -1)
    s = state.to(device)
    an = torch.as_tensor(method._normalise("a", a.detach().cpu().numpy()), device=device)
    sn = torch.as_tensor(method._normalise("s", s.detach().cpu().numpy()), device=device)
    if hasattr(method, "prototypes"):
        raw_base, residual = method.net.encode(an, sn)
        prototypes = torch.as_tensor(method.prototypes, dtype=raw_base.dtype, device=device)
        codes = torch.cdist(raw_base, prototypes).argmin(dim=-1)
        return {"codes": codes, "embedding": prototypes[codes]}, residual
    encoded = method.net.encode(an, sn, method.top_k)
    alpha, residual = encoded[0], encoded[1]
    return {"alpha": alpha}, residual


def _method_encode_with_grad(method, actions, state):
    """Use an opt-in differentiable encoder when a method exposes one."""
    if callable(getattr(method, "encode_trainable", None)):
        return method.encode_trainable(actions, state)
    for encoder in (_m2_encode_with_grad, _m1_encode_with_grad, _m45_encode_with_grad):
        encoded = encoder(method, actions, state)
        if encoded is not None:
            return encoded
    return None


def _m2_decode_with_grad(method, base, residual, feature, state):
    """Differentiable M2 decoder path used only for an in-training pressure loss."""
    if not all(hasattr(method, name) for name in ("decoder", "codebook", "am", "fm", "sm", "action_shape")):
        return None
    device = method.device
    ids = torch.as_tensor(base, dtype=torch.long, device=device).reshape(-1)
    r = torch.as_tensor(residual, dtype=torch.float32, device=device)
    f = torch.as_tensor(feature, dtype=torch.float32, device=device)
    s = torch.as_tensor(state, dtype=torch.float32, device=device)
    sn = (s - torch.as_tensor(method.sm, device=device)) / torch.as_tensor(method.ss, device=device)
    parts = [method.codebook[ids], r]
    if getattr(method, "decoder_obs", True):
        fn = (f - torch.as_tensor(method.fm, device=device)) / torch.as_tensor(method.fs, device=device)
        parts.append(fn)
    parts.append(sn)
    pred = method.decoder(torch.cat(parts, dim=-1))
    pred = pred * torch.as_tensor(method.ass, device=device) + torch.as_tensor(method.am, device=device)
    return pred.reshape(len(pred), *method.action_shape)


def _method_decode_with_grad(method, base, residual, feature, state):
    """Use an opt-in differentiable decoder when a method exposes one."""
    if callable(getattr(method, "decode_trainable", None)):
        return method.decode_trainable(base, residual, feature, state)
    decoded = _m2_decode_with_grad(method, base, residual, feature, state)
    if decoded is not None:
        return decoded
    # CanonicalizedMethod owns a structured geometry residual and delegates
    # its differentiable inner path through _decode_inner.
    if callable(getattr(method, "_decode_inner", None)):
        return method.decode(base, residual, feature, state)
    device = getattr(method, "device", torch.device("cpu"))
    b, r = base, _to_tensor(residual).to(device)
    f, s = _to_tensor(feature).to(device), _to_tensor(state).to(device)
    if all(hasattr(method, name) for name in ("decoder", "am", "ass", "fm", "fs", "sm", "ss", "base_dim")):
        b = _to_tensor(b).to(device)
        sn = (s - torch.as_tensor(method.sm, device=device)) / torch.as_tensor(method.ss, device=device)
        parts = [b, r]
        if getattr(method, "decoder_obs", True):
            fn = (f - torch.as_tensor(method.fm, device=device)) / torch.as_tensor(method.fs, device=device)
            parts.append(fn)
        parts.append(sn)
        pred = method.decoder(torch.cat(parts, dim=-1))
        pred = pred * torch.as_tensor(method.ass, device=device) + torch.as_tensor(method.am, device=device)
        return pred.reshape(len(pred), *method.action_shape)
    if hasattr(method, "net") and hasattr(method, "normalizers"):
        if isinstance(b, Mapping) and "codes" in b:
            ids = torch.as_tensor(b["codes"], dtype=torch.long, device=device).reshape(-1)
            base_embedding = torch.as_tensor(method.prototypes, dtype=torch.float32, device=device)[ids]
        elif isinstance(b, Mapping) and "alpha" in b:
            alpha = _to_tensor(b["alpha"]).to(device)
            dictionary = torch.nn.functional.normalize(method.net.dictionary, dim=-1, eps=1e-8)
            base_embedding = alpha @ dictionary
        else:
            return None
        sn = torch.as_tensor(method._normalise("s", s.detach().cpu().numpy()), device=device)
        fn = (torch.as_tensor(method._normalise("f", f.detach().cpu().numpy()), device=device)
              if getattr(method, "decoder_obs", True) else torch.zeros((len(s), 0), device=device))
        pred = method.net.decode(base_embedding, r, fn, sn)
        am, ass = method.normalizers["a"]
        pred = pred * torch.as_tensor(ass, device=device) + torch.as_tensor(am, device=device)
        return pred.reshape(len(pred), *method.action_shape)
    return None


def swap_reconstruction_loss(
    method,
    actions,
    state,
    feature,
    *,
    base=None,
    residual=None,
    max_base_distance: float | None = None,
    return_pair_count: bool = False,
):
    """Reconstruct each action with a nearby peer's swapped base/residual and own context.

    For discrete bases, only windows with identical code IDs are paired (all
    supplied levels are compared). For continuous bases, ``None`` selects each
    row's nearest peer in the minibatch; a numeric ``max_base_distance`` selects
    pairs within that radius. The default avoids assuming a scale for each method.
    Pair selection is label-free and nondifferentiable; decode and the loss
    retain gradients into the method's decoder/residual path.
    """
    actions, state, feature = _to_tensor(actions), _to_tensor(state), _to_tensor(feature)
    if base is None or residual is None:
        trainable = _method_encode_with_grad(method, actions, state)
        base, residual = trainable if trainable is not None else method.encode(actions, state)
    pairs = _matching_pairs(base, max_base_distance=max_base_distance)
    if not pairs:
        zero = actions.sum() * 0.0
        return (zero, 0) if return_pair_count else zero
    target_idx = [i for i, _ in pairs] + [j for _, j in pairs]
    donor_idx = [j for _, j in pairs] + [i for i, _ in pairs]
    b = _index(base, donor_idx)
    r = _index(residual, donor_idx)
    f = feature[target_idx]
    s = state[target_idx]
    target = actions[target_idx]
    prediction = _method_decode_with_grad(method, b, r, f, s)
    if prediction is None:
        prediction = method.decode(b, r, f, s)
    prediction = _to_tensor(prediction).to(device=target.device, dtype=target.dtype)
    loss = (prediction - target).square().mean()
    return (loss, len(target_idx)) if return_pair_count else loss


def add_swap_reconstruction_pressure(
    reconstruction_loss,
    method,
    actions,
    state,
    feature,
    *,
    weight: float = 0.1,
    max_base_distance: float | None = None,
    return_pair_count: bool = False,
):
    """Add S to an existing objective and optionally return the number of directed swaps."""
    swap, pair_count = swap_reconstruction_loss(
        method, actions, state, feature, max_base_distance=max_base_distance, return_pair_count=True
    )
    result = reconstruction_loss + float(weight) * swap
    return (result, swap, pair_count) if return_pair_count else result

