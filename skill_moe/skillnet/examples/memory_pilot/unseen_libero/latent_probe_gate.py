#!/usr/bin/env python3
"""CPU/offline gate for scale and gripper heads on LIBERO-90 unseen rows.

The command is deliberately separate from model training and closed-loop code.
It selects a scale-only W type (never a rotated type), applies one of several
amplitude/gripper rules, and reports the pre-registered Dev gate.  Ground truth
is read only by the oracle selectors, scoring, and the Dev-only S-const rule.

Example production invocation (after review):
  taskset -c 8-47 nice -n 10 env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  OPENBLAS_NUM_THREADS=8 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python latent_probe_gate.py --out /work/.skillnet-state/outputs/memory-pilot/unseen-libero/latent-probe-01/gate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import scale_only_query as sq
import variable_shape_memory as vs

ROOT = Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero')
SCALE = ROOT / 'scale-only-02/production'
ROWS = ROOT / 'unseen-rows-02'
LEGACY_ROWS = ROOT / 'oracle-01/unseen_rows'
PRED = ROOT / 'fast-query-01/production/predictions'
EVAL_ROWS = ROOT / 'lib-closed-01/eval_rows.npz'
INV2 = ROOT / 'scene-invariance-01/step2/attempt-fix2'
P5_CHECKPOINT = INV2 / 'noise_history_models/P5_seed4101.pt'
R5_CHECKPOINT = INV2 / 'ablation_tuning/R5_from_R4_k5_w0.1.pt'
GRIPPER_THRESHOLD = 0.3019487
BOOT_REPS = 5000
REPRO_TOL = 1e-5


def mse6_over7(pred, target, start=0, stop=None):
    """Six motion axes' squared-error sum divided by active7 count (T*7)."""
    p = np.asarray(pred, np.float64)
    y = np.asarray(target, np.float64)
    if p.shape != y.shape or p.ndim < 2 or p.shape[-1] < 6:
        raise ValueError(f'expected matching [...,T,>=6] arrays, got {p.shape}, {y.shape}')
    stop = p.shape[-2] if stop is None else int(stop)
    d = p[..., int(start):stop, :6] - y[..., int(start):stop, :6]
    return np.sum(d * d, axis=(-2, -1)) / (d.shape[-2] * 7.)


def active7_mse(pred, target, start=0, stop=None):
    p = np.asarray(pred, np.float64); y = np.asarray(target, np.float64)
    if p.shape != y.shape or p.shape[-1] < 7:
        raise ValueError(f'expected matching [...,T,>=7] arrays, got {p.shape}, {y.shape}')
    stop = p.shape[-2] if stop is None else int(stop)
    d = p[..., int(start):stop, :7] - y[..., int(start):stop, :7]
    return np.mean(d * d, axis=(-2, -1))


def fit_nonnegative_scale(template, reference):
    """One nonnegative scalar over six axes; deliberately no rotation."""
    x = np.asarray(template, np.float64)[..., :6]
    y = np.asarray(reference, np.float64)[..., :6]
    if x.shape != y.shape:
        raise ValueError(f'scale fit shape mismatch: {x.shape} vs {y.shape}')
    return np.maximum(np.sum(x * y, axis=(-2, -1)), 0.) / np.maximum(np.sum(x * x, axis=(-2, -1)), 1e-12)


def scale_without_rotation(canonical, scale):
    """Apply a scalar to the six motion axes without changing their direction."""
    x = np.asarray(canonical, np.float64).copy()
    s = np.asarray(scale, np.float64)
    x[..., :6] *= s[..., None, None]
    return x


def hold_last_actions(inp):
    """Causal held action baseline; episode starts use state gripper + zero motion."""
    h = np.asarray(inp['hist'], np.float32)
    s = np.asarray(inp['state'], np.float32)
    frame = np.asarray(inp['frame'])
    has = np.asarray(inp.get('has_hist', frame > 0), bool)
    out = np.zeros((len(h), 10, 7), np.float32)
    out[:] = h[:, -1:, :7]
    first = ~has | (frame == 0)
    out[first, :, 6] = np.where(s[first, 6] > GRIPPER_THRESHOLD, -1., 1.)[:, None]
    return out


def switch_events(actions):
    """Return per-window gripper sign-change indices (1..T-1)."""
    g = np.asarray(actions)[..., 6]
    sign = np.where(g >= 0, 1, -1)
    return [np.flatnonzero(sign[i, 1:] != sign[i, :-1]) + 1 for i in range(len(sign))]


def gripper_metrics(pred, target):
    """Sign agreement and event precision/recall/time error on true-switch rows."""
    p = np.asarray(pred)[..., 6]; y = np.asarray(target)[..., 6]
    ps = np.where(p >= 0, 1, -1); ys = np.where(y >= 0, 1, -1)
    pe, ye = switch_events(pred), switch_events(target)
    prec, rec, terr = [], [], []
    switch_rows = np.asarray([len(x) > 0 for x in ye])
    for pp, yy in zip(pe, ye):
        remaining = list(map(int, pp)); matched = []
        for t in yy:
            choices = [u for u in remaining if abs(u - int(t)) <= 1]
            if choices:
                u = min(choices, key=lambda v: (abs(v-int(t)), v)); remaining.remove(u); matched.append((int(t), u))
        tp = len(matched)
        prec.append(tp / max(len(pp), 1)); rec.append(tp / max(len(yy), 1))
        terr.extend(abs(a-b) for a,b in matched)
    return {
        'sign_agreement': float(np.mean(ps == ys)),
        'switch_rows': int(switch_rows.sum()),
        'switch_precision_on_true_switch_rows': float(np.mean(np.asarray(prec)[switch_rows])) if switch_rows.any() else None,
        'switch_recall_on_true_switch_rows': float(np.mean(np.asarray(rec)[switch_rows])) if switch_rows.any() else None,
        'matched_switch_time_abs_error_steps': float(np.mean(terr)) if terr else None,
    }


def episode_bootstrap_diff(a, b, episode, *, reps=BOOT_REPS, seed=4101):
    """Paired mean difference and episode-cluster bootstrap percentile interval."""
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    ep = np.asarray(episode)
    if d.shape != ep.shape or not len(d): raise ValueError('paired values and episode ids must match')
    unique, inv = np.unique(ep, return_inverse=True)
    sums = np.bincount(inv, weights=d); counts = np.bincount(inv)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(int(reps), len(unique)))
    values = sums[draws].sum(1) / counts[draws].sum(1)
    return {'mean': float(d.mean()), 'ci95': [float(np.quantile(values,.025)), float(np.quantile(values,.975))],
            'episodes': int(len(unique)), 'rows': int(len(d)), 'reps': int(reps)}


def _assert_same_rows(a, b, name):
    ra, rb = np.asarray(a['row']), np.asarray(b['row'])
    if len(np.unique(ra)) != len(ra) or len(np.unique(rb)) != len(rb) or not np.array_equal(ra, rb):
        raise AssertionError(f'{name} row IDs/order differ')


def _same_task_mask(inp, pool):
    """Compatibility shim: use the LIB1/K1 candidate-mask definition, never raw-ID mapping."""
    return candidate_mask('dev', inp, pool)


def candidate_mask(split, inp, pool, *, eval_record=None):
    """Shared LIB1/K1 mask. Eval uses saved allowed verbatim; Dev uses scale-only's unseen rule."""
    keep=np.asarray(pool.get('keep',np.ones(len(pool['shape']),bool)),bool)
    pool_task=np.asarray(pool.get('task',pool.get('source_task',np.full(len(keep),-1))),np.int64)
    if split=='eval':
        record=np.load(EVAL_ROWS) if eval_record is None else eval_record
        _assert_same_rows({'row':inp['row']},{'row':record['row_id']},'Eval candidate mask row order')
        allowed=np.asarray(record['allowed'],bool)
        if allowed.shape!=(len(inp['row']),len(keep)):
            raise ValueError(f"saved Eval allowed shape {allowed.shape} != {(len(inp['row']),len(keep))}")
        if np.any(allowed[:,~keep]): raise AssertionError('saved Eval mask allows a pool item outside keep')
        return allowed.copy()
    if split!='dev': raise ValueError(split)
    # scale_only_query.load_train_data defines unseen Dev with cross_task=False;
    # unseen LIBERO-90 task IDs are a different namespace from LIBERO-40 pool IDs.
    return sq.allowed_matrix(inp['task'],pool_task,keep,cross_task=False)


def _load_k1_pool():
    """Use exactly the 1,595-shape K1/LibClosed C_vm_none candidate order."""
    import theta_regression as tr
    hold=(2,7,12,17,22,27,32,37)
    base=tr.candidate_shape_pool(rep='C_vm_none')
    keep=~np.isin(np.asarray(base['source_task'],np.int64),hold)
    pool={k:(np.asarray(v)[keep] if isinstance(v,np.ndarray) and v.ndim and len(v)==len(keep) else v)
          for k,v in base.items()}
    pool['keep']=np.ones(len(pool['shape']),bool)
    if len(pool['shape'])!=1595: raise AssertionError(f"K1 pool count changed: {len(pool['shape'])}")
    return pool


def _load_action_rows_and_assert_same():
    """Use corrected unseen-rows-02 action data and verify legacy actions by metadata."""
    new={k:np.load(ROWS/f'{k}.npy',mmap_mode='r') for k in ('task','episode','frame','actions')}
    old={k:np.load(LEGACY_ROWS/f'{k}.npy',mmap_mode='r') for k in ('task','episode','frame','actions')}
    old_index=_row_index(old)
    keys=[(int(t),int(e),int(f)) for t,e,f in zip(new['task'],new['episode'],new['frame'])]
    if len(keys)!=len(old_index) or any(k not in old_index for k in keys):
        raise AssertionError('corrected and legacy unseen-row metadata do not align')
    oi=np.asarray([old_index[k] for k in keys],np.int64)
    newer=np.asarray(new['actions'],np.float32);older=np.asarray(old['actions'][oi],np.float32)
    if not np.array_equal(newer,older):
        delta=float(np.max(np.abs(newer-older)))
        raise AssertionError(f'action rows changed across orientation refresh (max abs={delta})')
    return new,{'rows':len(keys),'actions_equal':True,'max_abs_action_delta':0.0,
                'token_sources':{'corrected':str(ROWS/'tokens.npy'),'legacy':str(LEGACY_ROWS/'tokens.npy')}}


def _wrist_tokens_for_input(inp, *, corrected=True):
    """Read group-1 wrist tokens from the orientation-corrected row cache by metadata key."""
    source=ROWS if corrected else LEGACY_ROWS
    meta={k:np.load(source/f'{k}.npy',mmap_mode='r') for k in ('task','episode','frame','tokens','groups')}
    ix=_align_unseen_rows(inp,meta)
    result=[]
    for start in range(0,len(ix),16):
        ids=ix[start:start+16]
        ts=np.asarray(meta['tokens'][ids],np.float32);gs=np.asarray(meta['groups'][ids],np.int64)
        batches=[ts[j,gs[j]==1] for j in range(len(ids))]
        if any(len(x)!=256 for x in batches): raise ValueError('corrected cache must have 256 wrist tokens per row')
        result.extend(batches)
    return np.asarray(result,np.float32)


def load_corrected_references():
    """Load row-aligned rotated-correct pi05 chunks and assert as-is is the wrong orientation."""
    dev = np.load(PRED / 'pi05_dev_rot180.npz')
    eva = np.load(PRED / 'pi05_eval_rot180.npz')
    asis = np.load(PRED / 'pi05_eval_asis.npz')
    for z, name in ((dev,'corrected Dev'), (eva,'corrected Eval'), (asis,'as-is Eval')):
        if 'row' not in z or not any(k in z for k in ('actions','predictions','pi05')):
            raise ValueError(f'{name} file lacks row/actions arrays: {z.files}')
    return dev, eva, asis


def _pred_actions(z):
    for k in ('actions','predictions','pi05'):
        if k in z: return np.asarray(z[k], np.float32)
    raise KeyError(f'no action array in {z.files}')


def _row_index(meta):
    return {(int(t),int(e),int(f)): i for i,(t,e,f) in enumerate(zip(meta['task'],meta['episode'],meta['frame']))}


def _align_unseen_rows(inp, rows_meta):
    idx = _row_index(rows_meta)
    keys = [(int(t),int(e),int(f)) for t,e,f in zip(inp['task'],inp['episode'],inp['frame'])]
    missing = [k for k in keys if k not in idx]
    if missing: raise ValueError(f'unseen metadata does not cover query rows: {missing[:3]}')
    return np.asarray([idx[k] for k in keys], np.int64)


def audit_orientation_and_hold_last(*, assert_reference=True):
    """Validate corrected-vs-rotated pi05 and the documented hold-last values."""
    d, e, a = load_corrected_references()
    meta,action_audit=_load_action_rows_and_assert_same()
    inp = dict(np.load(SCALE/'prep/W_inputs_eval.npz'))
    ix = _align_unseen_rows(inp, meta)
    target = np.asarray(meta['actions'][ix,:,:7], np.float32)
    _assert_same_rows({'row':inp['row']}, {'row':e['row']}, 'Eval W-input / corrected pi05')
    _assert_same_rows({'row':inp['row']}, {'row':a['row']}, 'Eval W-input / as-is pi05')
    correct_mse = float(np.mean(mse6_over7(_pred_actions(e), target)))
    wrong_mse = float(np.mean(mse6_over7(_pred_actions(a), target)))
    hold = hold_last_actions(inp)
    hold_mse = float(np.mean(mse6_over7(hold, target)))
    if assert_reference:
        if not .105 <= correct_mse <= .125: raise AssertionError(f'rot180 pi05 MSE6/7 expected about .114, got {correct_mse:.6f}')
        if not .175 <= wrong_mse <= .210: raise AssertionError(f'as-is pi05 MSE6/7 expected about .192, got {wrong_mse:.6f}')
        if not .100 <= hold_mse <= .135: raise AssertionError(f'hold-last all-row MSE6/7 expected about .114, got {hold_mse:.6f}')
        hard = np.asarray(inp['hard'], bool)
        hm = float(np.mean(mse6_over7(hold[hard], target[hard])))
        if not .175 <= hm <= .230: raise AssertionError(f'hold-last hard MSE6/7 expected about .201, got {hm:.6f}')
    return {'pi05_rot180_mse6_over7':correct_mse, 'pi05_asis_mse6_over7':wrong_mse,
            'hold_last_eval_mse6_over7':hold_mse,
            'hold_last_hard_mse6_over7':float(np.mean(mse6_over7(hold[np.asarray(inp['hard'],bool)], target[np.asarray(inp['hard'],bool)]))),
            'corrected_action_cache_audit':action_audit}


def select_oracle_type(target_phys, canonical, allowed):
    """T-oracle under the same explicit candidate mask used by learned selectors."""
    err = sq.scale_only_error(target_phys[..., :6], np.asarray(canonical)[..., :6], 'S-shared')
    allowed = np.asarray(allowed, bool)
    if allowed.shape != err.shape:
        raise ValueError(f'oracle candidate mask {allowed.shape} does not match errors {err.shape}')
    if not allowed.any(1).all(): raise ValueError('same-task exclusion leaves no oracle type')
    masked = np.where(allowed, err, np.inf)
    pick = np.argmin(masked, axis=1)
    return pick.astype(np.int64), masked[np.arange(len(pick)),pick]


def w_scale_only_types(pool):
    """Canonical direction-preserving W type: remove only one shared scale."""
    return np.asarray(sq.type_of(np.asarray(pool['phys'],np.float64),'S-shared'),np.float32)


def amplitude_predictions(canonical, source, pi05_phys, hist_norm, has_hist, *, const_scale=None):
    """Build no-rotation source/pi05/prev/const/oracle/procrustes shapes for selected types."""
    can = np.asarray(canonical, np.float64); src = np.asarray(source, np.float64)
    pi = np.asarray(pi05_phys, np.float64); hist = vs.to_physical(np.asarray(hist_norm, np.float64)[...,:6])
    has = np.asarray(has_hist, bool)
    if can.shape != src.shape or can.shape != pi.shape or can.shape != (len(can),10,6):
        raise ValueError(f'expected aligned [N,10,6] inputs, got {can.shape}, {src.shape}, {pi.shape}')
    # Fit against normalized action-space error, the metric used by this gate.
    can_norm=vs.to_normalized(can);pi_norm=vs.to_normalized(pi)
    denom = np.maximum(np.sum(can_norm*can_norm,axis=(1,2)),1e-12)
    s_pi = np.maximum(np.sum(can_norm*pi_norm,axis=(1,2)),0.) / denom
    can_first = np.sqrt(np.mean(np.sum(can[:,:5]**2,axis=-1),axis=1))
    hist_last = np.sqrt(np.mean(np.sum(hist[:,-5:]**2,axis=-1),axis=1))
    s_prev = np.divide(hist_last, np.maximum(can_first,1e-12))
    s_source = np.sqrt(np.mean(np.sum(src**2,axis=-1),axis=1) / np.maximum(np.mean(np.sum(can**2,axis=-1),axis=1),1e-12))
    s_prev = np.where(has, s_prev, s_source)
    s_const = np.full(len(can), np.nan if const_scale is None else float(const_scale))
    s_oracle = np.zeros(len(can),np.float64)  # replaced using target labels in the caller
    # Procrustes is the explicitly labelled reference, not one of the no-rotation rules.
    rscale=[]; rotated=[]
    for x, y in zip(can,pi):
        r, s = sq.procrustes_rs(x,y); rscale.append(s); rotated.append(sq.rotate_scale(x,r,s))
    return {'S-source':src.copy(), 'S-pi05':scale_without_rotation(can,s_pi),
            'S-prev':scale_without_rotation(can,s_prev), 'S-const':scale_without_rotation(can,s_const),
            'S-oracle':None, 'S-procrustes':np.asarray(rotated,np.float64),
            '_scales':{'source':s_source,'pi05':s_pi,'prev':s_prev,'const':s_const,'procrustes':np.asarray(rscale)}}


def fit_oracle_scale(canonical, target_normalized):
    """Oracle scalar optimized in normalized evaluation space; no rotation."""
    return fit_nonnegative_scale(vs.to_normalized(canonical), target_normalized)


def predict_gripper(inp, pi05, target=None):
    """Return G-pi05 and causal G-hold chunks; G-oracle is caller-supplied scoring data."""
    pi = np.asarray(pi05,np.float32).copy()
    hold = hold_last_actions(inp)
    return {'G-pi05':pi[...,6], 'G-hold':hold[...,6], 'G-oracle':None}


def _metric_record(pred, target, inp, *, hard=None):
    p=np.asarray(pred,np.float32);y=np.asarray(target,np.float32)
    select=np.ones(len(y),bool) if hard is None else np.asarray(hard,bool)
    out={}
    for name,lo,hi in [('first5',0,5),('continue5',5,10),('all10',0,10)]:
        vals=mse6_over7(p,y,lo,hi)
        out[name]={'window_mean':float(vals[select].mean()) if select.any() else None,'n':int(select.sum())}
        if select.any():
            tasks=np.asarray(inp['task'])[select]
            per=[float(vals[select][tasks==t].mean()) for t in np.unique(tasks)]
            out[name]['task_macro_mean']=float(np.mean(per))
        else: out[name]['task_macro_mean']=None
    return out


def _load_r5_picks(split, pool, inp, allowed):
    """Run the exact K1 R5-seed-4101 encoder over its 1,595 C_vm_none pool."""
    import torch
    import torch.nn.functional as F
    import scene_invariance_v2 as v2
    torch.set_num_threads(8)
    # The corrected row cache contains 732 entries; map using task/episode/frame, not row order.
    meta={k:np.load(ROWS/f'{k}.npy',mmap_mode='r') for k in ('task','episode','frame','tokens','groups')}
    ridx=_align_unseen_rows(inp,meta)
    if meta['tokens'].shape[1:] != (712,2048): raise ValueError(f'expected corrected tokens [N,712,2048], got {meta["tokens"].shape}')
    shapes=np.asarray(pool['shape'],np.float32)
    if len(shapes)!=1595 or np.asarray(allowed).shape!=(len(inp['row']),len(shapes)):
        raise ValueError('R5 candidate pool/mask must be aligned to K1 C_vm_none pool')
    ck=torch.load(R5_CHECKPOINT,map_location='cpu',weights_only=False)
    net=v2._build_model('A',k=5,pooled=False,key_mode='split',query_pool=False,no_history_cross=True)
    net.load_state_dict(ck['state_dict']);net.eval()
    with torch.no_grad():
        keys=F.normalize(net.encode_keys(torch.as_tensor(shapes)),dim=-1)
        picks=[]
        for st in range(0,len(ridx),16):
            sl=slice(st,min(st+16,len(ridx)))
            # Convert only the current batch from the read-only fp16 memmap.
            qt=torch.as_tensor(np.asarray(meta['tokens'][ridx[sl]],np.float32))
            gg=torch.as_tensor(np.asarray(meta['groups'][ridx[sl]],np.int64))
            wrist=[qt[j,gg[j]==1] for j in range(len(qt))]
            if any(len(x)!=256 for x in wrist):
                raise ValueError(f'expected 256 wrist tokens per query; got {[len(x) for x in wrist]}')
            wrist=torch.stack(wrist)
            hh=torch.as_tensor(inp['hist'][sl],dtype=torch.float32);ss=torch.as_tensor(inp['state'][sl],dtype=torch.float32)
            q,_=net.encode_query(wrist,hh,ss)
            scores=(q[:,None]*keys[None]).sum(-1).mean(-1)*net.log_scale.exp()
            row_allowed=torch.as_tensor(np.asarray(allowed[sl],bool))
            scores=scores.masked_fill(~row_allowed,-1e9)
            picks.extend(scores.argmax(-1).cpu().numpy().tolist())
    return np.asarray(picks,np.int64)


def _load_p5_picks(split, pool, inp, allowed):
    """Run exact step-1 P5 scorer/checkpoint with the shared K1 candidate mask."""
    import torch
    import scene_invariance as s1
    torch.set_num_threads(8)
    if any(k in inp for k in ('target','actions','target_action')): raise ValueError('P5 query input unexpectedly contains labels')
    if np.asarray(pool['shape']).shape!=(1595,10,7):raise ValueError('P5 K1 shape pool changed')
    ck=torch.load(P5_CHECKPOINT,map_location='cpu',weights_only=False)
    net=s1._make_model('P5',1,2048);net.load_state_dict(ck['state_dict']);net.eval()
    data={'task':np.asarray(inp['task']),'history':np.asarray(inp['hist'],np.float32),
          'state':np.asarray(inp['state'],np.float32),'pool_shape':np.asarray(pool['shape'],np.float32),
          'allowed':np.asarray(allowed,bool)}
    scores,rank,_=s1._score_model(net,'P5-L0',data,np.arange(len(inp['row'])),layer='final',device='cpu',batch=16)
    return np.asarray(rank[:,0],np.int64),np.asarray(scores,np.float32)


def _load_combined_pool():
    """Join K1's exact C_vm_none pool ordering with the W physical values."""
    import key_query_probe as kq
    k1 = _load_k1_pool()
    w = dict(np.load(SCALE/'prep/W_pool.npz'))
    if len(k1['shape']) != len(w['phys']) or not np.array_equal(k1['source_task'], w['task']):
        raise AssertionError('K1 and scale-only W pool candidate ordering/task ids differ')
    decoded = kq.decode_pool_actions(k1, np.arange(len(k1['shape']), dtype=np.int64)[:, None], vs,
                                     theta_source='src')[:, 0, :, :6]
    decoded_phys = np.asarray(vs.to_physical(decoded), np.float32)
    max_delta = float(np.max(np.abs(decoded_phys - np.asarray(w['phys'], np.float32))))
    if max_delta > 1e-5:
        raise AssertionError(f'K1 candidate row/value alignment failed, max abs={max_delta:.3g}')
    return {'shape': np.asarray(k1['shape'], np.float32),
            'phys': np.asarray(w['phys'], np.float32),
            'canon': np.asarray(w['canon'], np.float32),
            'task': np.asarray(k1['source_task'], np.int64),
            'source_task': np.asarray(k1['source_task'], np.int64),
            'keep': np.asarray(w['keep'], bool),
            'source_row': np.asarray(k1['source_row'], np.int64),
            'source_unit': np.asarray(k1['source_unit'], np.int32),
            'source_theta': np.asarray(k1['source_theta'], np.float32),
            'alignment_max_abs_phys': max_delta}


def _load_split_rows(split):
    """Load unlabeled query inputs and attach action labels only for later scoring."""
    inp = dict(np.load(SCALE/f'prep/W_inputs_{split}.npz'))
    rows, action_audit = _load_action_rows_and_assert_same()
    idx = _align_unseen_rows(inp, rows)
    target = np.asarray(rows['actions'][idx, :, :7], np.float32)
    if split == 'dev':
        if len(inp['row']) != 113 or set(map(int, np.unique(inp['task']))) != {6, 7}:
            raise AssertionError('Dev split no longer matches the fixed LIBERO-90 tasks 6,7 / 113 rows')
    elif split == 'eval':
        record = np.load(EVAL_ROWS)
        _assert_same_rows({'row': inp['row']}, {'row': record['row_id']}, 'Eval W-input/lib-closed rows')
    return inp, target, action_audit


def eval_reproduction_check(pool, inp, target, allowed, picks, *, record=None):
    """Compare seed-4101 P5/R5 selections with the recorded LIB1 rows; Eval-only helper."""
    rec = np.load(EVAL_ROWS) if record is None else record
    _assert_same_rows({'row': inp['row']}, {'row': rec['row_id']}, 'Eval reproduction row order')
    if not np.array_equal(np.asarray(allowed, bool), np.asarray(rec['allowed'], bool)):
        raise AssertionError('Eval selectors did not receive lib-closed-01 allowed mask verbatim')
    result = {}
    type_pool = w_scale_only_types(pool)
    target_phys = vs.to_physical(np.asarray(target, np.float32)[:, :, :6])
    for name, key in (('T-P5', 'P5'), ('T-R5', 'R5')):
        if name not in picks: continue
        ix = np.asarray(picks[name], np.int64)
        reproduced_type = np.diagonal(sq.scale_only_error(target_phys, type_pool[ix], 'S-shared'))
        stored_type = np.asarray(rec[f'{key}_row_type_error'][0], np.float64)
        stored_candidate_type = np.asarray(rec['errors'][np.arange(len(ix)), ix], np.float64)
        stored_mse = np.asarray(rec[f'{key}_row_mse6_over7'][0], np.float64)
        src = np.asarray(pool['phys'][ix], np.float32)
        pred = np.zeros_like(target, np.float32)
        pred[:, :, :6] = vs.to_normalized(src)
        reproduced_mse = mse6_over7(pred, target)
        type_delta = max(float(np.max(np.abs(stored_type-reproduced_type))),
                         float(np.max(np.abs(stored_candidate_type-reproduced_type))))
        mse_delta = float(np.max(np.abs(reproduced_mse-stored_mse)))
        # Picks are verified through the pick-only MSE; LIB1 stored a differently
        # normalized type error, so the type delta is recorded but not gated.
        if mse_delta > REPRO_TOL:
            raise AssertionError(f'{name} Eval replay mismatch: type max={type_delta:.3g}, mse max={mse_delta:.3g}')
        result[name] = {'seed': 4101, 'mean_type_error': float(reproduced_type.mean()),
                        'mean_saved_scale_mse6_over7': float(reproduced_mse.mean()),
                        'stored_type_mean': float(stored_type.mean()), 'stored_mse_mean': float(stored_mse.mean()),
                        'max_abs_type_delta': type_delta, 'max_abs_mse_delta': mse_delta}
    result['T-oracle'] = {'mean_type_error': float(np.min(np.where(allowed, rec['errors'], np.inf), axis=1).mean()),
                          'reference': 'minimum S-shared error in lib-closed-01 eval_rows errors under saved allowed'}
    return result


def run_dev_check(out):
    """CPU-only actual-cache Dev diagnostic; intentionally makes no gate decision."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    audit = audit_orientation_and_hold_last()
    pool = _load_combined_pool()
    inp, target, action_audit = _load_split_rows('dev')
    allowed = candidate_mask('dev', inp, pool)
    if not np.all(allowed[:, pool['keep']]):
        raise AssertionError('Dev pool keep mask unexpectedly excluded candidate rows')
    # Labels are kept out of both learned selector inputs; they enter only this oracle and score path.
    target_phys = vs.to_physical(target[:, :, :6])
    type_pool = w_scale_only_types(pool)
    oracle, oracle_err = select_oracle_type(target_phys, type_pool, allowed)
    p5, p5_scores = _load_p5_picks('dev', pool, inp, allowed)
    r5 = _load_r5_picks('dev', pool, inp, allowed)
    picks = {'T-oracle': oracle, 'T-P5': p5, 'T-R5': r5}
    pi_z = np.load(PRED/'pi05_dev_rot180.npz')
    _assert_same_rows({'row': inp['row']}, {'row': pi_z['row']}, 'corrected pi05/Dev')
    pi = _pred_actions(pi_z)
    rows_out = {}
    for name, ix in picks.items():
        if np.any(~allowed[np.arange(len(ix)), ix]):
            raise AssertionError(f'{name} selected a disallowed candidate')
        selected_err = np.diagonal(sq.scale_only_error(target_phys, type_pool[ix], 'S-shared'))
        src = np.asarray(pool['phys'][ix], np.float32)
        pred = np.zeros_like(target, np.float32)
        pred[:, :, :6] = vs.to_normalized(src)
        mse = mse6_over7(pred, target)
        rows_out[name] = {'mean_type_error': float(selected_err.mean()),
                          'mean_saved_scale_mse6_over7': float(mse.mean()),
                          'median_saved_scale_mse6_over7': float(np.median(mse)),
                          'candidate_count': int(len(pool['shape'])),
                          'all_selected_allowed': True}
    payload = {
        'mode': 'actual-cache Dev CPU diagnostic only; no gate/decision and no Eval evaluation',
        'orientation_and_hold_audit': audit,
        'corrected_action_cache_audit': action_audit,
        'pool_alignment_max_abs_physical_action': pool['alignment_max_abs_phys'],
        'mask': {'source': 'scale_only_query.allowed_matrix(..., cross_task=False)',
                 'meaning': 'LIBERO-90 unseen Dev tasks have a separate task namespace; all kept LIBERO-40 pool entries are eligible',
                 'shape': list(allowed.shape), 'allowed_per_query_min': int(allowed.sum(1).min()),
                 'allowed_per_query_max': int(allowed.sum(1).max())},
        'rows': rows_out,
        'p5_score_mean': float(np.mean(np.max(p5_scores, axis=1))),
        'picks': {k: np.asarray(v, np.int64).tolist() for k, v in picks.items()},
        'protocol': {'query_action_source': str(ROWS/'actions.npy'),
                     'query_token_source_for_R5': str(ROWS/'tokens.npy'),
                     'query_join_key': ['task', 'episode', 'frame'],
                     'all_selectors_share_identical_allowed_mask': True,
                     'task_id_namespace_rule': 'no raw LIBERO-90 vs LIBERO-40 task-id comparison'}
    }
    (out/'dev_diagnostic.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    report = ['# LPG 修正後 Dev 診断', '',
              'CPU の実キャッシュ Dev のみを実行。これは数値確認であり、関門判定ではない。Eval の選択再現チェックは実装したが、この実行では呼び出していない。',
              '', f"回転確認: rot180 MSE6/7={audit['pi05_rot180_mse6_over7']:.6f}; as-is={audit['pi05_asis_mse6_over7']:.6f}.",
              f"補正キャッシュ行動は旧キャッシュと全行一致。K1 pool decode の物理値最大差={pool['alignment_max_abs_phys']:.3g}。", '',
              '## Dev 数値', '', '| 型選択 | 平均 S-shared 型誤差 | 保存スケール MSE6/7 平均 | 中央値 |', '|---|---:|---:|---:|']
    for name in ('T-oracle', 'T-P5', 'T-R5'):
        x = rows_out[name]
        report.append(f"| {name} | {x['mean_type_error']:.6f} | {x['mean_saved_scale_mse6_over7']:.6f} | {x['median_saved_scale_mse6_over7']:.6f} |")
    report += ['', '候補マスクは全方式共通。Dev は `scale_only_query.load_train_data` の unseen Dev 規則 `allowed_matrix(..., cross_task=False)` に従い、別名前空間のタスク ID を数値比較しない。詳細は `dev_diagnostic.json`。', '']
    (out/'report.md').write_text('\n'.join(report))
    return payload


def _paired_episode_ids(inp):
    return np.asarray(inp['task'],np.int64)*1_000_000+np.asarray(inp['episode'],np.int64)


def run_gate(out):
    """Full CPU gate; provided for a separately reviewed production invocation."""
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    # Audit must happen before reading any selector result; corrected pi05 only thereafter.
    audit=audit_orientation_and_hold_last()
    pool=_load_combined_pool()
    targets={k:np.load(ROWS/f'{k}.npy',mmap_mode='r') for k in ('task','episode','frame','actions')}
    all_results={};all_picks={};bootstrap={};eval_reproduction=None
    for split in ('dev','eval'):
        inp=dict(np.load(SCALE/f'prep/W_inputs_{split}.npz'))
        if split=='dev' and len(inp['row'])!=113:raise AssertionError(f'expected 113 Dev rows, got {len(inp["row"])}')
        if split=='eval' and len(inp['row'])!=619:raise AssertionError(f'expected 619 Eval rows, got {len(inp["row"])}')
        if split=='dev' and set(map(int,np.unique(inp['task'])))!={6,7}:
            raise AssertionError(f'Dev task split changed: {np.unique(inp["task"]).tolist()}')
        if split=='eval':
            eval_ref=np.load(EVAL_ROWS)
            _assert_same_rows({'row':inp['row']},{'row':eval_ref['row_id']},'Eval W-input/lib-closed rows')
            if not np.array_equal(np.asarray(inp['hard'],bool),np.asarray(eval_ref['hard'],bool)):
                raise AssertionError('Eval hard-row mask differs from lib-closed-01/eval_rows.npz')
        ridx=_align_unseen_rows(inp,targets)
        target_norm=np.asarray(targets['actions'][ridx,:,:7],np.float32)
        target_phys=vs.to_physical(target_norm[:,:,:6])
        pi_file=PRED/f'pi05_{split}_rot180.npz'; pi_z=np.load(pi_file)
        _assert_same_rows({'row':inp['row']},{'row':pi_z['row']},f'corrected pi05/{split}')
        pi_norm=_pred_actions(pi_z)
        pi_phys=vs.to_physical(pi_norm[:,:,:6])
        # Every selector and oracle receives the same explicit candidate mask.
        allowed=candidate_mask(split,inp,pool,eval_record=eval_ref if split=='eval' else None)
        p5,_=_load_p5_picks(split,pool,inp,allowed)
        type_pool=w_scale_only_types(pool)
        oracle,_=select_oracle_type(target_phys,type_pool,allowed)
        picks={'T-P5':p5,'T-oracle':oracle}
        if split=='eval':
            # The frozen LIB1 R5 (seed 4101) Eval choices are recovered exactly from
            # its saved source-scale predictions; the gate's own R5 replay did not
            # reproduce LIB1, so no re-implemented R5 is used.  Dev has no LIB1 R5 run.
            import variable_shape_memory as _vs
            rp=_vs.to_physical(np.asarray(eval_ref['R5_prediction'][0][...,:6],np.float64))
            ph=np.asarray(pool['phys'],np.float64)
            d=((rp[:,None]-ph[None])**2).sum((2,3));r5=d.argmin(1)
            if d.min(1).max()>1e-8: raise AssertionError('LIB1 R5 Eval picks not exactly recoverable')
            picks['T-R5']=r5.astype(np.int64)
        if split=='eval':
            eval_reproduction=eval_reproduction_check(pool,inp,target_norm,allowed,picks,record=eval_ref)
        all_picks[split]={k:v.tolist() for k,v in picks.items()}
        hard=np.asarray(inp['hard'],bool) if 'hard' in inp else None
        task_ids=np.asarray(inp['task']); episode=_paired_episode_ids(inp)
        split_rows={}
        for typ,pick in picks.items():
            can=np.asarray(type_pool[pick],np.float64);src=np.asarray(pool['phys'][pick],np.float64)
            base=amplitude_predictions(can,src,pi_phys,inp['hist'],inp['has_hist'])
            s_oracle=fit_oracle_scale(can,target_norm[:,:,:6]);base['S-oracle']=scale_without_rotation(can,s_oracle)
            # A single scale is calibrated only on Dev oracle fits, then frozen for Eval.
            if split=='dev':
                # T-const is the median scale fitted to the selected type on Dev.
                const=float(np.median(s_oracle))
                base['S-const']=scale_without_rotation(can,np.full(len(can),const))
            else:
                devpick=all_picks.get('dev',{}).get(typ, all_picks.get('dev',{}).get('T-P5'))  # T-R5 has no Dev run: use the P5 Dev constant
                # Dev has been processed first; recover the same typed Dev samples and scales.
                if devpick is None: raise AssertionError('Dev scales must precede Eval constant rule')
                dev_inp=dict(np.load(SCALE/'prep/W_inputs_dev.npz')); dev_idx=_align_unseen_rows(dev_inp,targets)
                dev_can=np.asarray(type_pool[np.asarray(devpick)],np.float64)
                dev_y_norm=np.asarray(targets['actions'][dev_idx,:,:6],np.float32)
                const=float(np.median(fit_oracle_scale(dev_can,dev_y_norm)))
                base['S-const']=scale_without_rotation(can,np.full(len(can),const))
            rule_results={}
            for size,pred_phys in base.items():
                if size.startswith('_'):continue
                pred=np.zeros_like(target_norm)
                pred[:,:,:6]=vs.to_normalized(pred_phys)
                pred[:,:,6]=pi_norm[:,:,6]
                metrics={'all':_metric_record(pred,target_norm,inp)}
                if hard is not None: metrics['hard']=_metric_record(pred,target_norm,inp,hard=hard)
                metrics['gripper']=gripper_metrics(pred,target_norm)
                # Paired bootstrap to causal hold-last and corrected pi05 for each time segment.
                metrics['paired_vs_hold']={}
                metrics['paired_vs_pi05']={}
                hold=hold_last_actions(inp)
                masks={'all':np.ones(len(inp['row']),bool)}
                if hard is not None: masks['hard']=hard
                phases={'first5':(0,5),'continue5':(5,10),'all10':(0,10)}
                for phase,(lo,hi) in phases.items():
                    metrics['paired_vs_hold'][phase]={};metrics['paired_vs_pi05'][phase]={}
                    pred_err=mse6_over7(pred,target_norm,lo,hi)
                    hold_err=mse6_over7(hold,target_norm,lo,hi)
                    pi_err=mse6_over7(pi_norm,target_norm,lo,hi)
                    for part,mask in masks.items():
                        metrics['paired_vs_hold'][phase][part]=episode_bootstrap_diff(pred_err[mask],hold_err[mask],episode[mask])
                        metrics['paired_vs_pi05'][phase][part]=episode_bootstrap_diff(pred_err[mask],pi_err[mask],episode[mask])
                rule_results[size]=metrics
            err_selected=np.diagonal(sq.scale_only_error(target_phys,type_pool[pick],'S-shared'))
            rule_results['_selected_type_error']={'all':float(np.mean(err_selected)),
                'hard':float(np.mean(err_selected[hard])) if hard is not None and hard.any() else None}
            if split=='dev':
                # Paired Dev gate only: Eval labels do not enter a gate decision.
                pred_oracle=np.zeros_like(target_norm);pred_oracle[:,:,:6]=vs.to_normalized(base['S-oracle']);pred_oracle[:,:,6]=pi_norm[:,:,6]
                pred_pi=np.zeros_like(target_norm);pred_pi[:,:,:6]=vs.to_normalized(base['S-pi05']);pred_pi[:,:,6]=pi_norm[:,:,6]
                pred_prev=np.zeros_like(target_norm);pred_prev[:,:,:6]=vs.to_normalized(base['S-prev']);pred_prev[:,:,6]=pi_norm[:,:,6]
                hold=hold_last_actions(inp)
                oerr=mse6_over7(pred_oracle,target_norm,0,5)
                err_pi=mse6_over7(pred_pi,target_norm,0,5);err_prev=mse6_over7(pred_prev,target_norm,0,5)
                best_err=np.minimum(err_pi,err_prev)
                rule_results['_size_gate']={
                    'oracle_minus_best_inference_size_first5':episode_bootstrap_diff(oerr,best_err,episode),
                    'oracle_minus_hold_last_first5':episode_bootstrap_diff(oerr,mse6_over7(hold,target_norm,0,5),episode),
                    'oracle_below_both_point_estimates':bool(oerr.mean()<err_pi.mean() and oerr.mean()<err_prev.mean() and oerr.mean()<mse6_over7(hold,target_norm,0,5).mean()),
                }
            split_rows[typ]=rule_results
        # Gripper table uses same selected types only to keep rows aligned; type is immaterial to G rules.
        gripper_table={}
        hold=hold_last_actions(inp)
        for name,g in {'G-pi05':pi_norm[:,:,6], 'G-hold':hold[:,:,6], 'G-oracle':target_norm[:,:,6]}.items():
            pred=hold.copy();pred[:,:,6]=g
            gm=gripper_metrics(pred,target_norm)
            gm['active7_mse_all']=float(active7_mse(pred,target_norm).mean())
            gm['active7_mse_true_switch_rows']=float(active7_mse(pred,target_norm)[np.asarray([len(x)>0 for x in switch_events(target_norm)])].mean()) if any(len(x)>0 for x in switch_events(target_norm)) else None
            gripper_table[name]=gm
        all_results[split]={'metrics':split_rows,'gripper':gripper_table}
    # Add reproducibility evidence and auditable row-ID choices.
    # The pre-registered go/no-go is based on Dev rows only.  An interval must
    # be wholly below zero for oracle-vs-best-inference amplitude and hold-last.
    size_decisions={}
    dev_metrics=all_results['dev']['metrics']
    for typ in ('T-P5','T-R5'):
        if typ not in dev_metrics: continue
        gate=dev_metrics[typ]['_size_gate']
        a=gate['oracle_minus_best_inference_size_first5']['ci95']
        b=gate['oracle_minus_hold_last_first5']['ci95']
        size_decisions[typ]=bool(gate['oracle_below_both_point_estimates'] and a[1]<0 and b[1]<0)
    dev_gt=np.asarray(targets['actions'][_align_unseen_rows(dict(np.load(SCALE/'prep/W_inputs_dev.npz')),targets),:,:7],np.float32)
    dev_switch=np.asarray([len(z)>0 for z in switch_events(dev_gt)],bool)
    dev_inp=dict(np.load(SCALE/'prep/W_inputs_dev.npz'))
    dev_pi=_pred_actions(np.load(PRED/'pi05_dev_rot180.npz'))
    gpi=np.zeros_like(dev_gt);gpi[:,:,6]=dev_pi[:,:,6]
    gor=np.zeros_like(dev_gt);gor[:,:,6]=dev_gt[:,:,6]
    grip_improve=float(np.mean(active7_mse(gpi,dev_gt)[dev_switch]-active7_mse(gor,dev_gt)[dev_switch])) if dev_switch.any() else None
    # Use P5 selected type and oracle-vs-Pi05 scalar size as the comparator size gain.
    dev_pick=np.asarray(all_picks['dev']['T-P5'],np.int64);dev_can=np.asarray(w_scale_only_types(pool)[dev_pick],np.float64)
    dev_pi_phys=vs.to_physical(dev_pi[:,:,:6]);dev_y_phys=vs.to_physical(dev_gt[:,:,:6])
    dev_oracle_scale=fit_oracle_scale(dev_can,dev_gt[:,:,:6])
    dev_o=np.zeros_like(dev_gt);dev_o[:,:,:6]=vs.to_normalized(scale_without_rotation(dev_can,dev_oracle_scale));dev_o[:,:,6]=dev_pi[:,:,6]
    dev_p=np.zeros_like(dev_gt);dev_p[:,:,:6]=vs.to_normalized(amplitude_predictions(dev_can,np.asarray(pool['phys'][dev_pick]),dev_pi_phys,dev_inp['hist'],dev_inp['has_hist'])['S-pi05']);dev_p[:,:,6]=dev_pi[:,:,6]
    size_gain=float(np.mean((mse6_over7(dev_p,dev_gt,0,10)-mse6_over7(dev_o,dev_gt,0,10))[dev_switch])) if dev_switch.any() else None
    gripper_useful=bool(grip_improve is not None and size_gain is not None and grip_improve>=size_gain)
    decisions={'size_head_useful':size_decisions,'size_head_useful_any':bool(any(size_decisions.values())), 'gripper_head_useful':gripper_useful,
               'gripper_vs_size_improvement':{'gripper_active7_mse_gain_true_switch_rows':grip_improve,'size_all10_mse6_over7_gain_T_P5':size_gain}}
    payload={'orientation_and_hold_audit':audit,'results':all_results,'picks':all_picks,'decisions':decisions,
             'eval_reproduction_check':eval_reproduction,
             'protocol':{'type_pool':'K1 C_vm_none pool matched to scale-only-02 W, 1,595 candidates',
                         'candidate_mask':'shared exact lib-closed Eval allowed; scale_only_query Dev cross_task=False',
                         'bootstrap_reps':BOOT_REPS,
                         'mse':'sum six-axis squared error / (T*7)','no_rotation_size_rules':['S-source','S-pi05','S-prev','S-const','S-oracle'],
                         'oracle_only_for_diagnostic_and_dev_constant':True}}
    (out/'summary.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    (out/'report.md').write_text(render_report(payload)+'\n')
    return payload


def render_report(payload):
    audit=payload['orientation_and_hold_audit']
    lines=['# LPG: 大きさとグリッパーの上限', '',
           '## データ向きの確認', '',
           f"rot180 π0.5 MSE6/7={audit['pi05_rot180_mse6_over7']:.4f}、asis={audit['pi05_asis_mse6_over7']:.4f}。以後は rot180 だけを使う。",
           f"hold-last={audit['hold_last_eval_mse6_over7']:.4f} (hard {audit['hold_last_hard_mse6_over7']:.4f})。", '',
           '## Dev の関門判定', '',
           f"大きさヘッド: T-P5={payload['decisions']['size_head_useful']['T-P5']}、T-R5={payload['decisions']['size_head_useful'].get('T-R5','Dev未実行')}。",
           f"グリッパーヘッド: {payload['decisions']['gripper_head_useful']} (切替行でのG改善={payload['decisions']['gripper_vs_size_improvement']['gripper_active7_mse_gain_true_switch_rows']}、サイズ改善={payload['decisions']['gripper_vs_size_improvement']['size_all10_mse6_over7_gain_T_P5']})。", '',
           '## 選択型の S-shared 誤差', '', '| split | 型 | all | hard |', '|---|---|---:|---:|']
    for split in ('dev','eval'):
        for typ in ('T-oracle','T-P5','T-R5'):
            m=payload['results'][split]['metrics'][typ]['_selected_type_error']
            hard='—' if m['hard'] is None else f"{m['hard']:.4f}"
            lines.append(f"| {split} | {typ} | {m['all']:.4f} | {hard} |")
    lines += ['', '## MSE6/7 の窓平均', '',
              '| split | 型 | size | first5 all | first5 hard | cont5 all | cont5 hard | all10 all | all10 hard |',
              '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for split in ('dev','eval'):
        for typ in ('T-oracle','T-P5','T-R5'):
            rows=payload['results'][split]['metrics'][typ]
            for size in ('S-source','S-pi05','S-prev','S-const','S-oracle','S-procrustes'):
                cells=[]
                for phase in ('first5','continue5','all10'):
                    cells.append(f"{rows[size]['all'][phase]['window_mean']:.4f}")
                    hv=rows[size].get('hard',{}).get(phase,{}).get('window_mean')
                    cells.append('—' if hv is None else f'{hv:.4f}')
                lines.append('| '+split+' | '+typ+' | '+size+' | '+' | '.join(cells)+' |')
    lines += ['', '## 対応差（方式 − 基準、episode bootstrap 95% CI）', '',
              '| split | 型 | size | 区間 | vs hold-last all | vs π0.5 all | vs hold-last hard | vs π0.5 hard |',
              '|---|---|---|---|---:|---:|---:|---:|']
    for split in ('dev','eval'):
        for typ in ('T-oracle','T-P5','T-R5'):
            rows=payload['results'][split]['metrics'][typ]
            for size in ('S-source','S-pi05','S-prev','S-const','S-oracle','S-procrustes'):
                for phase in ('first5','continue5','all10'):
                    def fmt_ci(which,part):
                        d=rows[size][f'paired_vs_{which}'][phase].get(part)
                        return '—' if d is None else f"{d['mean']:.4f} [{d['ci95'][0]:.4f},{d['ci95'][1]:.4f}]"
                    lines.append(f"| {split} | {typ} | {size} | {phase} | {fmt_ci('hold','all')} | {fmt_ci('pi05','all')} | {fmt_ci('hold','hard')} | {fmt_ci('pi05','hard')} |")
    lines += ['', '## グリッパー', '',
              '| split | 方式 | 符号一致 | 切替行 precision | recall | 時刻誤差(step) | active7 MSE (all) | active7 MSE (switch) |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for split in ('dev','eval'):
        for name,m in payload['results'][split]['gripper'].items():
            def fmt(v): return '—' if v is None else f'{v:.4f}'
            lines.append(f"| {split} | {name} | {fmt(m['sign_agreement'])} | {fmt(m['switch_precision_on_true_switch_rows'])} | {fmt(m['switch_recall_on_true_switch_rows'])} | {fmt(m['matched_switch_time_abs_error_steps'])} | {fmt(m['active7_mse_all'])} | {fmt(m['active7_mse_true_switch_rows'])} |")
    lines += ['', 'S-oracle は上限診断、T-oracle は型選択の上限。従来の「正解型をπ0.5へ合わせる」参照は T-oracle/S-pi05（回転なし）と T-oracle/S-procrustes（回転あり）で再計算する。π0.5予測は rot180 固定。MSE6/7 は6軸二乗誤差和を T×7 で割る。task-macro 平均、候補ID、全行・hard 指標は summary.json に保存。切替イベントは符号反転、予測イベントは真値±1ステップ内で一対一対応させた。', '']
    return '\n'.join(lines)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=ROOT/'latent-probe-01/gate')
    ap.add_argument('--audit-only',action='store_true',help='orientation and hold-last assertions only; no selection tables')
    ap.add_argument('--dev-only',action='store_true',help='run actual-cache Dev diagnostics only; never evaluates Eval')
    ap.add_argument('--full',action='store_true',help='reviewed full gate: Dev decision, Eval report-only')
    args=ap.parse_args(argv)
    if args.full:
        run_gate(args.out);print(json.dumps({'out':str(args.out),'mode':'full'}));return 0
    if args.audit_only:
        print(json.dumps(audit_orientation_and_hold_last(),indent=2));return 0
    if not args.dev_only:
        ap.error('this reviewed invocation is Dev-only; pass --dev-only explicitly. Eval is not run here.')
    result=run_dev_check(args.out)
    print(json.dumps({'out':str(args.out),'mode':result['mode'],'rows':result['rows']},indent=2))
    return 0


if __name__=='__main__': raise SystemExit(main())
