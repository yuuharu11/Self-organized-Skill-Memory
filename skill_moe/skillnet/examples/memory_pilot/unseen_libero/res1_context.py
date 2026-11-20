"""Saved/current KQ2 context extraction for RES1 (read-only imports)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from res1_core import SavedContext

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('SKILLNET_STATE_DIR', str(HERE.parents[4] / '.skillnet-state'))) / 'outputs/memory-pilot/unseen-libero'
OUT = ROOT / 'res1-01'
CONTEXT_VERSION = 2


class StateContextProjection(nn.Module):
    """Map raw 8-D state to the context width used by the cross-attention."""
    def __init__(self, in_dim: int = 8, dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, state):
        return self.net(state.float()).unsqueeze(1)


def kq_context_tokens(saved: dict[str, torch.Tensor], current: dict[str, torch.Tensor],
                      state_projector: nn.Module, *, visual: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Create aligned [saved,current,delta] context memory and validity mask.

    Context token order is h, state, then (for iii) 34 projected VLM tokens.
    No task label, target action, or future observation is consumed.
    """
    for name in ('h', 'state'):
        if saved[name].shape[0] != current[name].shape[0]:
            raise ValueError(f"batch mismatch in {name}")
    sh, ch = saved['h'].reshape(-1, 1, saved['h'].shape[-1]), current['h'].reshape(-1, 1, current['h'].shape[-1])
    ss, cs = state_projector(saved['state']), state_projector(current['state'])
    saved_parts, now_parts = [sh, ss], [ch, cs]
    masks = [torch.ones((len(sh), 2), dtype=torch.bool, device=sh.device)]
    if visual:
        sv, cv = saved.get('vlm'), current.get('vlm')
        if sv is None or cv is None or sv.shape != cv.shape:
            raise ValueError("iii needs aligned saved/current projected VLM tokens")
        saved_parts.append(sv); now_parts.append(cv)
        valid = current.get('vlm_valid', torch.ones(sv.shape[:2], dtype=torch.bool, device=sv.device)).bool()
        masks.append(valid)
    s = torch.cat(saved_parts, dim=1); c = torch.cat(now_parts, dim=1)
    valid = torch.cat(masks, dim=1)
    return torch.cat((s, c, c - s), dim=1), torch.cat((valid, valid, valid), dim=1)


def query_context(model, gsplit, row_ids, *, state: torch.Tensor, visual: bool) -> dict[str, torch.Tensor]:
    """Expose KQ2 query's final pre-L2 h and causal context tokens."""
    ix = torch.as_tensor(row_ids, dtype=torch.long, device=gsplit.start.device)
    h0 = model.encode_h(gsplit, ix)
    if model.xattn_n:
        if not visual or gsplit.vlm is None:
            raise ValueError('cross-attention query requires its causal VLM token cache')
        kv = model.kv(gsplit.vlm[ix].float())
        x = h0[:, None]
        for layer in range(model.xattn_n):
            a, _ = model.attn[layer](x, kv, kv, need_weights=False)
            x = model.post[layer](x + a)
            if model.ffn is not None:
                x = model.ffn[layer](x)
        h = x[:, 0]
    else:
        x = model.post0(h0)
        if model.ffn0 is not None:
            x = model.ffn0(x)
        h = x
    out = {'h': h.detach(), 'state': state.detach().float()}
    if visual:
        if gsplit.vlm is None:
            raise ValueError("visual context requested without VLM cache")
        out['vlm'] = model.kv(gsplit.vlm[ix].float()).detach()
        out['vlm_valid'] = torch.ones(out['vlm'].shape[:2], dtype=torch.bool, device=out['vlm'].device)
    return out


def extract_bank_contexts(which: str, output: Path | None = None, *, batch_size: int = 128) -> Path:
    """Cache unit-start contexts using exact KQ2 row-to-bank candidate ids.

    Only CPU inference is used here. This is a small frozen-query pass, not
    training or an Eval-label read.
    """
    if which not in ('i', 'iii'):
        raise ValueError("which must be i or iii")
    import kq2_run as kr
    import kq2cl_common as C
    out = Path(output or OUT / 'context'); out.mkdir(parents=True, exist_ok=True)
    dest = out / f'bank_{which}.npz'
    if dest.exists():
        meta_path = out / f'bank_{which}.json'
        if not meta_path.exists() or json.loads(meta_path.read_text()).get('context_version') != CONTEXT_VERSION:
            raise ValueError(f'legacy saved context has invalid time/row alignment; rebuild in a new output directory: {dest}')
        return dest
    ctx = kr.Ctx('cpu', splits=('eh',), vlm=(which == 'iii'), out=C.KQ2)
    sp = ctx.splits['eh']; gs = kr.GPUSplit(sp, torch.device('cpu'), which == 'iii')
    model, cfg = C.load_query(which, torch.device('cpu'))
    model.eval()
    nbank = len(ctx.bank['task'])
    source = np.asarray(ctx.bank['source'], np.int64)
    nbank = len(source)
    # The EH observation cache is stride-5. Reconstruct history/state at the
    # exact unit start from its episode sequence. For iii the VLM observation
    # must also be at that start; an older cache row is rejected below.
    row_for = np.full(nbank, -1, np.int64); frame_gap = np.zeros(nbank, np.int32)
    by_ep: dict[tuple[int, int], list[int]] = {}
    for row, key in enumerate(np.asarray(sp['key'])):
        by_ep.setdefault((int(key[0]), int(key[1])), []).append(row)
    slot_by_ep = {}
    for row, key in enumerate(np.asarray(sp['key'])):
        slot_by_ep.setdefault((int(key[0]), int(key[1])), int(gs.slot[row]))
    unit_slots = np.empty(nbank, np.int64)
    for unit, (task, episode, start, _end) in enumerate(source):
        ep=(int(task),int(episode)); rows=by_ep.get(ep)
        if rows is None:
            raise AssertionError(f'unit episode absent from KQ2 sequences: {ep}')
        frames=np.asarray(sp['key'])[rows,2]
        j=int(np.searchsorted(frames,start,side='right')-1)
        if j < 0: j=0
        row_for[unit]=rows[j];frame_gap[unit]=int(start-frames[j]);unit_slots[unit]=slot_by_ep[ep]
        if start < 0 or start >= int(gs.seq_s.shape[0] - gs.off[unit_slots[unit]]):
            raise AssertionError('unit start lies outside cached episode sequence')
        if which == 'iii' and frame_gap[unit] > 9:
            raise AssertionError('unit-start VLM cache alignment gap exceeds the causal history window')
    if which == 'iii' and np.any(frame_gap != 0):
        raise ValueError(f'exact unit-start VLM features required; {(frame_gap != 0).sum()} units have earlier cached observations')
    class UnitSplit:
        def __init__(self):
            self.seq_a,self.seq_s,self.off=gs.seq_a,gs.seq_s,gs.off
            self.slot=torch.as_tensor(unit_slots,dtype=torch.long)
            self.start=torch.as_tensor(source[:,2],dtype=torch.long)
            self.vlm=gs.vlm[torch.as_tensor(row_for,dtype=torch.long)] if which=='iii' else None
        def window(self,ix,L=10):
            t=self.start[ix][:,None];g=self.off[self.slot[ix]][:,None]
            rel=torch.arange(L)[None]-(L-1);tau=t+rel;valid=tau>=0;gsx=(g+t+rel).clamp(min=0)
            ss=self.seq_s[gsx]*valid[...,None];av=tau>=1
            aa=self.seq_a[(g+t+rel-1).clamp(min=0)]*av[...,None]
            return torch.cat([aa,ss,av[...,None].float()],-1),valid
    units=UnitSplit()
    hs, states, vlms = [], [], []
    with torch.inference_mode():
        for s in range(0, nbank, batch_size):
            ids = torch.arange(s,min(s+batch_size,nbank),dtype=torch.long)
            state=gs.seq_s[gs.off[units.slot[ids]]+units.start[ids]]
            qctx = query_context(model, units, ids, state=state, visual=(which == 'iii'))
            hs.append(qctx['h'].cpu().numpy().astype(np.float32))
            states.append(qctx['state'].cpu().numpy().astype(np.float32))
            if which == 'iii':
                vlms.append(qctx['vlm'].cpu().numpy().astype(np.float16))
    arrays = {'h': np.concatenate(hs), 'state': np.concatenate(states),
              'bank_task': ctx.bt.astype(np.int32), 'bank_episode': ctx.be.astype(np.int32),
              'bank_source': ctx.bank['source'].astype(np.int64)}
    if which == 'iii':
        arrays['vlm'] = np.concatenate(vlms)
        arrays['vlm_valid'] = np.ones(arrays['vlm'].shape[:2], bool)
    if arrays['h'].shape != (nbank, 256) or len(arrays['state']) != nbank:
        raise AssertionError('saved unit-context dimensions/order mismatch')
    tmp = dest.with_suffix('.npz.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(dest)
    meta = {'context_version': CONTEXT_VERSION, 'which': which, 'rows': nbank, 'checkpoint': str(C.QUERY_CK[which]),
            'query_row_stride': 5, 'unit_source_map': 'bank source (task,episode,start,end) -> exact KQ2 episode sequence and unit-start VLM observation for iii',
            'h_definition': 'KQ2 QueryNet final output immediately before L2 normalization (GRU plus iii cross-attention when applicable)',
            'state': 'normalized state at unit-start observation',
            'vlm': 'KQ2 QueryNet.kv projected 34-token cache; exact unit-start observation' if which == 'iii' else None,
            'unit_start_cached_exactly': int(nbank if which=='i' else (frame_gap==0).sum()),
            'vlm_start_cached_exactly': int((frame_gap==0).sum()) if which=='iii' else None,
            'unit_start_cached_nearest':int(nbank),
            'contains_target_or_task_input': False}
    (out / f'bank_{which}.json').write_text(json.dumps(meta, indent=2) + '\n')
    return dest


def load_saved(path: Path, device='cpu') -> SavedContext:
    path = Path(path)
    metadata = path.with_suffix('.json')
    if not metadata.exists() or json.loads(metadata.read_text()).get('context_version') != CONTEXT_VERSION:
        raise ValueError(f'legacy saved context has invalid time/row alignment; rebuild in a new output directory: {path}')
    with np.load(path, allow_pickle=False) as z:
        h = torch.as_tensor(z['h'], device=device).float()[:, None]
        # raw state is kept, then passed through the trainable state projection
        st = torch.as_tensor(z['state'], device=device).float()
        vlm = torch.as_tensor(z['vlm'], device=device).float() if 'vlm' in z else None
        mask = torch.as_tensor(z['vlm_valid'], device=device).bool() if 'vlm_valid' in z else None
    result = SavedContext(h, st, vlm, mask); result.validate(); return result

