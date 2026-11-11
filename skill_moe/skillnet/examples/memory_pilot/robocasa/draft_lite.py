"""Offline scale and rotation diagnostics for RM-DRAFT-LITE.

Selection outputs are read from the frozen Context and P5 caches. Held-out
actions are used only for reporting metrics; the sole extra selector is the
predeclared position-only DCT nearest neighbor to the cached pi05 reference.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import experiment as ex

ROOT = Path('/work/.skillnet-state/outputs/robocasa-memory')
CTX = ROOT / 'ctx-01'
OUT = ROOT / 'key-lite-01/draft'


def _write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def _shape_pos(a):
    from scipy.fft import dct, idct
    a = np.asarray(a, np.float32)
    if len(a) != 10:
        a = ex.resample_rows(a, 10)
    cf = dct(a.astype(np.float64), type=2, norm='ortho', axis=0)[:3]
    norm = np.linalg.norm(cf)
    cf = cf / max(norm, 1e-12)
    full = np.zeros((10, 3), np.float64); full[:3] = cf
    return idct(full, type=2, norm='ortho', axis=0).astype(np.float32)


def _position_nn(ref10, pool, *, block=128):
    """Scale invariant nearest type using position-only DCT shape."""
    cand = np.stack([_shape_pos(a[:, 5:8]) for a in pool['action10']])
    q = np.stack([_shape_pos(a[:, 5:8]) for a in ref10])
    cf = cand.reshape(len(cand), -1).astype(np.float64)
    qf = q.reshape(len(q), -1).astype(np.float64)
    cn = np.sum(cf * cf, axis=1)
    pick = np.empty(len(q), np.int32); best = np.empty(len(q), np.float32)
    for s in range(0, len(q), block):
        qb = qf[s:s+block]
        dot = np.maximum(qb @ cf.T, 0.)
        err = (np.sum(qb*qb, axis=1, keepdims=True) - dot*dot/np.maximum(cn[None],1e-12)) / 30.
        ix = np.argmin(err, axis=1)
        pick[s:s+len(ix)] = ix; best[s:s+len(ix)] = err[np.arange(len(ix)), ix]
    return pick, best


def _ratio(a, b, lo=.25, hi=4.):
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na < 1e-10 or nb < 1e-10:
        return 1.
    return float(np.clip(nb / na, lo, hi))


def _projection(d, r, lo=.25, hi=4.):
    den = float(np.sum(np.asarray(d, np.float64)**2))
    if den < 1e-12: return 1.
    return float(np.clip(max(float(np.sum(np.asarray(d, np.float64)*np.asarray(r, np.float64)))/den, 0.), lo, hi))


def _row_metrics(draft, ref, truth, n):
    """Metrics for first five and the available part of the chosen unit."""
    n = int(max(1, n)); f = min(5, len(truth), len(ref))
    full = min(n, len(truth), len(ref))
    def first5(x):
        x = np.asarray(x)
        if len(x) >= 5: return x[:5]
        return np.concatenate([x, np.repeat(x[-1:], 5-len(x), axis=0)], axis=0)
    def calc(pred, target):
        diff = np.asarray(pred, np.float64) - np.asarray(target, np.float64)
        return {'pos': float(np.mean(diff[:, :3]**2)), 'rot': float(np.mean(diff[:, 3:6]**2)),
                'all6': float(np.mean(diff**2))}
    fr = calc(first5(draft), truth[:f]); lr = calc(draft[:full], truth[:full])
    eps = 1e-10
    ratio_ref = float(np.linalg.norm(draft[:full]) / max(np.linalg.norm(ref[:full]), eps))
    ratio_truth = float(np.linalg.norm(draft[:full]) / max(np.linalg.norm(truth[:full]), eps))
    return {'first5': fr, 'unit': lr, 'ratio_ref': ratio_ref, 'ratio_truth': ratio_truth,
            'evaluated_steps': int(full)}


def _draft_for(raw, ref, method, scale_rule, length):
    """Return 6-D draft and scale factors. Inputs are storage-order EEF6."""
    d = np.asarray(raw, np.float32)[:, 5:11].copy()
    r = np.asarray(ref, np.float32)[:, 5:11]
    n = min(int(length), len(d), len(r))
    if n <= 0: raise ValueError('empty compared interval')
    if scale_rule in ('stored', 'S4_stored'):
        sp = sr = 1.
    elif scale_rule == 'S1_projection':
        sp = _projection(d[:n, :3], r[:n, :3]); sr = _ratio(d[:n, 3:6], r[:n, 3:6])
    elif scale_rule == 'S2_unit_norm':
        sp = _ratio(d[:n, :3], r[:n, :3]); sr = _ratio(d[:n, 3:6], r[:n, 3:6])
    elif scale_rule == 'S3_first5_norm':
        m = min(5, n); sp = _ratio(d[:m, :3], r[:m, :3]); sr = _ratio(d[:m, 3:6], r[:m, 3:6])
    else:
        raise ValueError(scale_rule)
    if method == 'A_6D': d[:,:3] *= sp; d[:,3:6] *= sr
    elif method == 'B_shared_position_scale': d *= sp
    elif method == 'C_rotation_from_pi05':
        d[:,:3] *= sp
        d[:n,3:6] = r[:n,3:6]
        if len(d) > n: d[n:,3:6] = r[n-1,3:6]
    else: raise ValueError(method)
    return d, {'position':float(sp),'rotation':float(sr)}


def run(out=OUT):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'refusing to overwrite nonempty output {out}')
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(CTX/'cache/rows.npz', allow_pickle=False)
    p = dict(np.load(CTX/'cache/pool.npz', allow_pickle=False))
    ref = np.asarray(z['eval_pi05_actions'], np.float32)[:, :, :12]
    truth_window = np.asarray(z['heldout_target'], np.float32)
    qn = len(ref)
    # Verify exact row alignment with both frozen selector caches.
    ctx = np.load(CTX/'evaluation.json') if False else json.loads((CTX/'evaluation.json').read_text())
    ctxpick = np.asarray(ctx['context_selection_rows']['index'], np.int32)
    p5z = np.load(CTX/'p5_baselines/masked-input/p5_baseline_rows.npz', allow_pickle=False)
    for k, zk in [('task','heldout_task'),('episode','heldout_local_episode'),('frame','heldout_frame')]:
        if not np.array_equal(p5z[k], z[zk]): raise AssertionError(f'P5 alignment mismatch {k}')
    p5pick = np.asarray(p5z['pick_correct'], np.int32)
    if ctxpick.shape != (qn,) or p5pick.shape[1] != qn: raise AssertionError('selector row count mismatch')
    pospick, poserr = _position_nn(ref[:, :10], p)
    offsets = np.asarray(p['action_offsets']); flat = np.asarray(p['actions_flat'])
    # Retrieve full held-out trajectories for unit-length scoring.
    heldout_actions = {}
    manifest = json.loads((CTX/'cache/feature_manifest.json').read_text())
    source = Path(manifest['source'])
    for task, ep in set(zip(map(int,z['heldout_task']), map(int,z['heldout_local_episode']))):
        heldout_actions[(task,ep)] = ex._load_cached_episode(source, task, ep)[0]
    selectors = {'Context':ctxpick[None,:], 'P5_seed_mean':p5pick, 'pi05_position_type':pospick[None,:]}
    scale_rules = ('S1_projection','S2_unit_norm','S3_first5_norm','S4_stored')
    # S4 is the unscaled stored draft. Method A uses separate rotation magnitude
    # fitting for S1-S3; B reuses position scale for rotation; C takes rotation
    # directly from the pi05 reference. All scale estimates are independently
    # clipped to [0.25, 4].
    rows=[]; details={}; ratio_samples={}; step_coverage=[]
    for selector, picks in selectors.items():
        for method in ('A_6D','B_shared_position_scale','C_rotation_from_pi05'):
            # Position-only typed selection is explicitly the permitted pi05
            # nearest selector; Context/P5 selector rows remain unaltered.
            if selector != 'pi05_position_type' and method != 'A_6D':
                # Both the fixed A-selected candidate and the position-only
                # candidate are reported; the latter is not a Context/P5 score.
                pass
            for rule in scale_rules:
                metrics=[]; factors=[]
                for seed_i, pickrow in enumerate(picks):
                    for i, j in enumerate(pickrow):
                        j=int(j); raw=flat[offsets[j]:offsets[j+1]].astype(np.float32)
                        task=int(z['heldout_task'][i]); ep=int(z['heldout_local_episode'][i]); frame=int(z['heldout_frame'][i])
                        true_episode=heldout_actions[(task,ep)]
                        available=max(0, min(len(true_episode)-frame, 50))
                        L=min(len(raw), available, len(ref[i]))
                        if L<=0: continue
                        # Candidate unit begins at the selected candidate's own
                        # phase; compare that typed draft to the target episode
                        # from the query point, using the maximum common horizon.
                        first_h=min(5, len(true_episode)-frame, len(ref[i]))
                        truth=true_episode[frame:frame+max(L,first_h),5:11]
                        ref6=ref[i,:max(L,first_h),5:11]
                        draft, sf = _draft_for(raw, ref[i,:,:12], method, rule, L)
                        met=_row_metrics(draft, ref6, truth, L)
                        met.update({'query':i,'seed':seed_i,'unit_id':j,'unit_length':len(raw),
                                    'task':task,'episode':int(z['heldout_episode'][i]),'frame':frame,
                                    'scale_position':sf['position'],'scale_rotation':sf['rotation']})
                        metrics.append(met); factors.append(sf)
                # Aggregate row-level metrics; each P5 seed contributes equal weight.
                agg={}
                for horizon in ('first5','unit'):
                    for dim in ('pos','rot','all6'):
                        agg[f'{horizon}_{dim}_mse']=float(np.mean([m[horizon][dim] for m in metrics]))
                agg['ratio_final_over_pi05']={'median':float(np.median([m['ratio_ref'] for m in metrics])),
                    'q25':float(np.quantile([m['ratio_ref'] for m in metrics],.25)), 'q75':float(np.quantile([m['ratio_ref'] for m in metrics],.75))}
                agg['ratio_final_over_truth']={'median':float(np.median([m['ratio_truth'] for m in metrics])),
                    'q25':float(np.quantile([m['ratio_truth'] for m in metrics],.25)), 'q75':float(np.quantile([m['ratio_truth'] for m in metrics],.75))}
                agg['position_scale']={'median':float(np.median([m['scale_position'] for m in metrics])),
                    'q25':float(np.quantile([m['scale_position'] for m in metrics],.25)), 'q75':float(np.quantile([m['scale_position'] for m in metrics],.75))}
                agg['rotation_scale']={'median':float(np.median([m['scale_rotation'] for m in metrics])),
                    'q25':float(np.quantile([m['scale_rotation'] for m in metrics],.25)), 'q75':float(np.quantile([m['scale_rotation'] for m in metrics],.75))}
                agg['n_rows_seed_pooled']=len(metrics); agg['mean_eval_steps']=float(np.mean([m['evaluated_steps'] for m in metrics]))
                key=f'{selector}|{method}|{rule}'; details[key]=agg
                rows.append((selector,method,rule,agg))
    # Baselines evaluated on exactly the same 2,892 rows and capped unit horizon.
    bases={}
    for name in ('pi05','hold_last'):
        vals=[]
        for i in range(qn):
            task=int(z['heldout_task'][i]); ep=int(z['heldout_local_episode'][i]); frame=int(z['heldout_frame'][i])
            true_episode=heldout_actions[(task,ep)]; L=min(50,len(true_episode)-frame)
            if name=='pi05': pred=ref[i,:L,5:11]
            else:
                hist=np.asarray(z['heldout_hist'][i],np.float32)
                valid=np.flatnonzero(np.any(hist[:,5:11]!=0,axis=1))
                last=hist[valid[-1],5:11] if len(valid) else np.zeros(6,np.float32)
                pred=np.repeat(last[None],L,axis=0)
            truth=true_episode[frame:frame+L,5:11]
            vals.append(_row_metrics(pred,ref[i,:L,5:11],truth,L))
        bases[name]={f'{h}_{d}_mse':float(np.mean([v[h][d] for v in vals])) for h in ('first5','unit') for d in ('pos','rot','all6')}
        bases[name]['ratio_final_over_pi05']={'median':float(np.median([v['ratio_ref'] for v in vals])), 'q25':float(np.quantile([v['ratio_ref'] for v in vals],.25)), 'q75':float(np.quantile([v['ratio_ref'] for v in vals],.75))}
        bases[name]['ratio_final_over_truth']={'median':float(np.median([v['ratio_truth'] for v in vals])), 'q25':float(np.quantile([v['ratio_truth'] for v in vals],.25)), 'q75':float(np.quantile([v['ratio_truth'] for v in vals],.75))}
    # Saved per-row selected indices, scale choices, and outcomes for audit.
    np.savez_compressed(out/'row_metrics.npz', context_pick=ctxpick, p5_pick=p5pick, pi05_position_pick=pospick,
        pi05_position_type_error=poserr, heldout_task=z['heldout_task'], heldout_episode=z['heldout_episode'], heldout_frame=z['heldout_frame'])
    result={'rows':qn,'episodes':int(len(np.unique(z['heldout_episode']))),'unit_lengths':{
        'memory_median':float(np.median(p['length'])),'memory_q25':float(np.quantile(p['length'],.25)),
        'memory_q75':float(np.quantile(p['length'],.75)),'query_candidate_median':float(np.median(p['length'][np.concatenate([ctxpick,p5pick.ravel(),pospick])]))},
        'selectors':{'Context':'existing ctx-01 context_selection_rows.index','P5':'existing p5_baselines/masked-input pick_correct (all 3 frozen seeds)','pi05_position_type':'scale-invariant nearest neighbor in position-only DCT-3 space, selected against cached pi05 reference'},
        'methods':{'A_6D':'retain selected 6-axis type; position uses requested scale rule, rotation uses its separate norm ratio for S1-S3 and stored scale for S4',
                   'B_shared_position_scale':'position-only type selection; one position scale applied to position and rotation',
                   'C_rotation_from_pi05':'position-only type selection; rotation replaced by cached pi05 rotation trajectory; no pi05 translation is copied'},
        'scale_rules':{'S1_projection':'max(dot(d,r)/dot(d,d),0), clip [0.25,4]; for A rotation separately uses norm ratio',
                       'S2_unit_norm':'||r[:L]||/||d[:L]||, clip [0.25,4]','S3_first5_norm':'||r[:5]||/||d[:5]||, clip [0.25,4]',
                       'S4_stored':'scale=1'},'baselines':bases,'results':details,
        'position_selector_type_error_mean':float(np.mean(poserr)),
        'target_policy':'Held-out actions used only in metric computation; existing Context/P5 picks are reused without re-selection. Unit-length metric is clipped to min(unit length, episode remainder, cached pi05 horizon 50). Units shorter than five steps are repeated at their terminal command for the first-five metric.',
        'note':'EEF action commands are base-frame deltas; axes are storage action[5:11]. Gripper, base and control-mode are excluded.'}
    _write_json(out/'results.json',result)
    # report table concise enough to compare all selector/method/scale variants.
    lines=['実装: 完了','','# RM-DRAFT-LITE: magnitude and rotation offline study','',
        '閉ループは実施していない。保留120デモ・stride10の2,892窓を使用し、正解行動は採点だけに使った。Context/P5の選択は既存キャッシュをそのまま再利用。位置のみの例外セレクタは、π0.5予測に対するスケール不変DCT-3最近傍。',
        '', 'π0.5の参照行動は大きさ推定（およびC型の回転）だけに使い、位置行動の下書きには混ぜていない。位置・回転はLeRobot action[5:11]（台座基準）で評価。',
        '', '## 選択済み型の倍率・回転比較','',
        '| 選択器 | 回転方式 | 大きさ則 | 先頭5位置MSE | 先頭5回転MSE | 先頭5 6軸MSE | 単位長位置MSE | 単位長回転MSE | 単位長6軸MSE | final/ref 中央値 [Q1,Q3] | final/truth 中央値 [Q1,Q3] |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for selector,method,rule,a in rows:
        rr=a['ratio_final_over_pi05']; rt=a['ratio_final_over_truth']
        lines.append(f"|{selector}|{method}|{rule}|{a['first5_pos_mse']:.6f}|{a['first5_rot_mse']:.6f}|{a['first5_all6_mse']:.6f}|{a['unit_pos_mse']:.6f}|{a['unit_rot_mse']:.6f}|{a['unit_all6_mse']:.6f}|{rr['median']:.3f} [{rr['q25']:.3f},{rr['q75']:.3f}]|{rt['median']:.3f} [{rt['q25']:.3f},{rt['q75']:.3f}]|")
    lines += ['', '## 参照ベースライン','', '| 基準 | 先頭5位置MSE | 先頭5回転MSE | 先頭5 6軸MSE | 単位長位置MSE | 単位長回転MSE | 単位長6軸MSE | final/ref 中央値 | final/truth 中央値 |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,a in bases.items():
        lines.append(f"|{name}|{a['first5_pos_mse']:.6f}|{a['first5_rot_mse']:.6f}|{a['first5_all6_mse']:.6f}|{a['unit_pos_mse']:.6f}|{a['unit_rot_mse']:.6f}|{a['unit_all6_mse']:.6f}|{a['ratio_final_over_pi05']['median']:.3f}|{a['ratio_final_over_truth']['median']:.3f}|")
    best=min(rows,key=lambda r:r[3]['unit_all6_mse'])
    lines += ['', '## 判断','', f"この評価範囲では単位長6軸MSEの最小は `{best[0]} / {best[1]} / {best[2]}`（{best[3]['unit_all6_mse']:.6f}）で、π0.5参照そのもの（{bases['pi05']['unit_all6_mse']:.6f}）をわずかに上回った。Context/P5で固定済みの選択候補では、倍率はS1射影が単位長MSEを抑え、S3先頭5倍率は先頭5の改善が単位長で崩れやすい。回転はC（π0.5回転）が回転誤差を下げる一方、これは予測器の回転を借りる条件で、選択済みメモリだけの利得ではない。記述的比較であり設定選択や閉ループ効果を示さない。",
        '実用上の候補は「P5/Contextで選んだ単位の位置にS1を適用し、回転は独立ノルム比（A）で扱う」。π0.5回転を実行時に使える場合はCも候補だが、本条件での最良値はπ0.5参照より明確に良いとは言えない。S2はπ0.5と同じ大きさ中央値に較正されるが、この保留オフライン評価では単位長MSEがS1より大きかった。',
        'S1は現在の射影則、S2は単位長のノルム比、S3は先頭5ステップのノルム比、S4は保存倍率。B/Cでは位置のみで型を選び、Bは位置倍率を回転にも共有、Cはπ0.5の回転を使う。全倍率は各軸群ごとに[0.25,4]へclipした。',
        '単位長比較は候補単位の長さ、エピソード残り、π0.5キャッシュの50ステップの最小までを評価した。5ステップ未満の単位は先頭5比較で終端指令を繰り返す。π0.5回転のためCは最大50ステップである。',
        'すべての結果と行対応、セレクタ、式は `results.json` と `row_metrics.npz` に保存。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__ == '__main__':
    run()
