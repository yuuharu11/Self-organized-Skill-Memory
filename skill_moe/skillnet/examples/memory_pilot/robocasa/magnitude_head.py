"""Estimate draft magnitude from RoboCasa Context+history and selected unit.

Offline RM-MAG implementation. π0.5 actions are used only for the S2
comparison; they are not an input or target for the magnitude regressor.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
import experiment as ex

ROOT = Path('/work/.skillnet-state/outputs/robocasa-memory')
CTX = ROOT / 'ctx-01'
KEYLITE = ROOT / 'key-lite-01'
SOURCE = ROOT / 'memory-01/policy-order-v2'
OUT = KEYLITE / 'magnitude'


def _write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def _load_keylite():
    script = KEYLITE / 'run_key_lite.py'
    spec = importlib.util.spec_from_file_location('rm_key_lite_for_magnitude', script)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.DEVICE = 'cpu'
    return mod


def _masked_predictor_inputs(rk, ck, z, feat, pool, split, histdata):
    """Build exactly the query vector used by Context+history ranker."""
    qctx, keys = rk.assemble_context(z, feat, ck['norm'], split)
    hist = histdata['hist']; mask = histdata['mask']
    hn = ck['hist_norm']
    h = ((hist - hn['mean']) / hn['std']).astype(np.float32) * mask[..., None]
    x = np.concatenate([qctx, h.reshape(len(h), -1), mask.astype(np.float32)], axis=1).astype(np.float32)
    return x, keys


def _motion_scale(draft, target, *, minimum=.25, maximum=4.):
    d = np.asarray(draft, np.float64); t = np.asarray(target, np.float64)
    den = float(np.sum(d*d))
    if den < 1e-12: return 1.
    return float(np.clip(max(float(np.sum(d*t))/den, 0.), minimum, maximum))


def _norm_scale(draft, target, *, minimum=.25, maximum=4.):
    dn = float(np.linalg.norm(draft)); tn = float(np.linalg.norm(target))
    if dn < 1e-12 or tn < 1e-12: return 1.
    return float(np.clip(tn/dn, minimum, maximum))


def _selected_label(pool, q_episode, q_frame, unit_id, episode_actions):
    off = pool['action_offsets']; flat = pool['actions_flat']
    raw = np.asarray(flat[int(off[unit_id]):int(off[unit_id+1])], np.float32)
    truth_ep = episode_actions[(int(q_episode)//1000, int(q_episode)%1000)]
    rem = max(0, len(truth_ep)-int(q_frame)); n = min(len(raw), rem, 50)
    if n <= 0: return 1., raw, 0
    target = truth_ep[int(q_frame):int(q_frame)+n, 5:11]
    # One scalar magnitude is the supervised target, matching S-shared scale.
    s = _motion_scale(raw[:n, 5:11], target)
    return s, raw, n


def _unit_metadata(pool):
    """Per-unit key: normalized DCT type+log length, plus stored motion magnitude."""
    motion = np.asarray(pool['action10'], np.float32)[:, :, 5:11]
    magnitude = np.sqrt(np.mean(np.sum(motion.astype(np.float64)**2, axis=2), axis=1)).astype(np.float32)
    return np.concatenate([np.asarray(pool['key'], np.float32), np.log(np.maximum(magnitude, 1e-6))[:, None]], axis=1), magnitude


def _make_rows(rk, ck, z, pool, feat, split, episode_actions):
    histdata = rk.build_history_rows(z, split)
    xctx, _ = _masked_predictor_inputs(rk, ck, z, feat, pool, split, histdata)
    scores = rk.predict_ranker(ck, z, feat, pool, split=split, hist_data=histdata,
                               history_condition='correct', exclude_episode=True)
    pick = np.argmax(scores, axis=1).astype(np.int32)
    if np.any(np.asarray(z[f'{split}_episode']) == np.asarray(pool['global_episode'])[pick]):
        # The inequality should hold for every selected candidate after the
        # ranker applies the per-query same-episode mask.
        raise AssertionError(f'{split}: selected candidate came from the query episode')
    uk, magnitudes = _unit_metadata(pool)
    # Each feature row includes its selected unit key and stored magnitude.
    x = np.concatenate([xctx, uk[pick]], axis=1).astype(np.float32)
    targets = np.asarray(z[f'{split}_target'], np.float32)
    labels = np.empty(len(pick), np.float32); unit_lengths = np.empty(len(pick), np.int32)
    for i, j in enumerate(pick):
        episode = int(z[f'{split}_episode'][i]); frame = int(z[f'{split}_frame'][i])
        s, _raw, n = _selected_label(pool, episode, frame, int(j), episode_actions)
        labels[i] = s; unit_lengths[i] = n
    return {'x': x, 'pick': pick, 's_star': labels, 'unit_length_eval': unit_lengths,
            'unit_metadata': uk[pick], 'history':histdata}


class _MLP:
    @staticmethod
    def create(input_dim):
        import torch
        from torch import nn
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.net=nn.Sequential(nn.Linear(input_dim,256),nn.GELU(),nn.Linear(256,128),nn.GELU(),nn.Linear(128,1))
            def forward(self,x):return self.net(x).squeeze(-1)
        return Net()


def _fit(x, y, fit_ids, val_ids, *, seed=20261111, max_epochs=150, label='full'):
    import torch
    torch.set_num_threads(8); torch.manual_seed(int(seed))
    # Normalize features using training episodes only.
    mean=np.asarray(x[fit_ids].mean(0),np.float32)
    std=np.maximum(np.asarray(x[fit_ids].std(0),np.float32),1e-3)
    xn=((x-mean)/std).astype(np.float32)
    # Train on log clipped s*. The target is the positive scale, not a π0.5 action.
    yt=np.log(np.clip(np.asarray(y,np.float32),.25,4.)).astype(np.float32)
    model=_MLP.create(x.shape[1]); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    tx=torch.as_tensor(xn,dtype=torch.float32); ty=torch.as_tensor(yt,dtype=torch.float32)
    rng=np.random.default_rng(seed); best=(float('inf'),None,0); patience=15; stale=0; curve=[]
    batch=512
    for epoch in range(1,int(max_epochs)+1):
        model.train(); order=rng.permutation(fit_ids); losses=[]
        for start in range(0,len(order),batch):
            ids=order[start:start+batch]; pred=model(tx[ids]); loss=torch.mean((pred-ty[ids])**2)
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            pred=torch.cat([model(tx[val_ids[s:s+batch]]) for s in range(0,len(val_ids),batch)]).numpy()
        val=float(np.mean((pred-yt[val_ids])**2))
        curve.append({'epoch':epoch,'train_logscale_mse':float(np.mean(losses)),'val_logscale_mse':val})
        if val < best[0]-1e-7:
            best=(val,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},epoch);stale=0
        else:
            stale+=1
            if stale>=patience:break
    model.load_state_dict(best[1]);model.eval()
    return model, {'label':label,'best_epoch':best[2],'val_logscale_mse':best[0],
                   'fit_rows':len(fit_ids),'val_rows':len(val_ids),'feature_mean':mean,
                   'feature_std':std,'curve':curve,'state':best[1]}


def _predict(model, ck, x):
    import torch
    xx=(np.asarray(x,np.float32)-ck['feature_mean'])/ck['feature_std']
    with torch.no_grad(): pred=model(torch.as_tensor(xx,dtype=torch.float32)).numpy()
    return np.clip(np.exp(pred),.25,4.).astype(np.float32)


def _metric_rows(scales, selected, pool, z, split, episode_actions, refs=None):
    """Compute M0/M1/M2/S2/s* using only requested actions for scoring."""
    offsets=pool['action_offsets']; flat=pool['actions_flat']; hist=np.asarray(z[f'{split}_hist'],np.float32)
    targets=np.asarray(z[f'{split}_target'],np.float32)
    records={}
    for name, scale_rows in scales.items():
        records[name]=[]
        for i,j in enumerate(selected):
            raw=np.asarray(flat[offsets[int(j)]:offsets[int(j)+1]],np.float32)
            task=int(z[f'{split}_task'][i]); ep=int(z[f'{split}_local_episode'][i]) if f'{split}_local_episode' in z else int(z[f'{split}_episode'][i])%1000
            frame=int(z[f'{split}_frame'][i]); true_episode=episode_actions[(task,ep)]
            n=max(1,min(len(raw),50,max(1,len(true_episode)-frame)))
            truth=true_episode[frame:frame+n,5:11]; draft=raw[:n,5:11]
            # M2: norm ratio from available previous actions, over the candidate's
            # position commands for the same number of preceding samples.
            valid=np.flatnonzero(np.any(np.abs(hist[i,:,5:8])>1e-8,axis=1))
            prev=hist[i,valid,5:8] if len(valid) else np.zeros((0,3),np.float32)
            m=min(len(prev),n);s_hist=_norm_scale(draft[:m,:3],prev[-m:]) if m else 1.
            # S2 is a reference only: use the exact cached π0.5 norm ratio.
            s_pi=None
            if refs is not None:s_pi=_norm_scale(draft[:n,:3],np.asarray(refs[i,:n,5:8],np.float32))
            s_star=_motion_scale(draft[:,:3],truth[:,:3])
            values={'M0_stored':1.,'M2_history_norm':s_hist,'s_star':s_star}
            if s_pi is not None:values['S2_pi05_norm']=s_pi
            if name in ('M1_regression','M1_LOTO'):values[name]=float(scale_rows[i])
            for method,s in values.items():
                d=draft.copy()*float(s)
                f=min(5,len(true_episode)-frame,50)
                first=d[:min(5,len(d))]
                if len(first)<5:first=np.concatenate([first,np.repeat(first[-1:],5-len(first),axis=0)],axis=0)
                first_truth=true_episode[frame:frame+f,5:11]
                if len(first_truth)<5:first_truth=np.concatenate([first_truth,np.repeat(first_truth[-1:],5-len(first_truth),axis=0)],axis=0)
                def mse(x,y):return {'pos':float(np.mean((x[:,:3]-y[:,:3])**2)),
                    'rot':float(np.mean((x[:,3:6]-y[:,3:6])**2)),'all6':float(np.mean((x-y)**2))}
                rat_truth=float(np.linalg.norm(d)/max(np.linalg.norm(truth),1e-12))
                rat_pi=float(np.linalg.norm(d)/max(np.linalg.norm(refs[i,:n,5:11]),1e-12)) if refs is not None else np.nan
                records.setdefault(method,[]).append({'first5':mse(first,first_truth),'unit':mse(d,truth),
                    'ratio_truth':rat_truth,'ratio_pi05':rat_pi,'scale':float(s),'n':n,'task':task,'episode':int(z[f'{split}_episode'][i])})
    return records


def _summarize(records):
    result={}
    for name,rows in records.items():
        d={}
        for horizon in ('first5','unit'):
            for dim in ('pos','rot','all6'):
                d[f'{horizon}_{dim}_mse']=float(np.mean([r[horizon][dim] for r in rows]))
        for field,key in (('ratio_truth','ratio_final_over_truth'),('ratio_pi05','ratio_final_over_pi05'),('scale','scale')):
            vals=np.asarray([r[field] for r in rows],np.float64);vals=vals[np.isfinite(vals)]
            d[key]={'median':float(np.median(vals)),'q25':float(np.quantile(vals,.25)),'q75':float(np.quantile(vals,.75))}
        d['n_rows']=len(rows);d['mean_unit_steps']=float(np.mean([r['n'] for r in rows]))
        d['task_unit_all6_mse']={}
        for t in sorted(set(int(r['task']) for r in rows)):
            rr=[r for r in rows if int(r['task'])==t]
            d['task_unit_all6_mse'][str(t)]={'n':len(rr),'mse':float(np.mean([r['unit']['all6'] for r in rr]))}
        result[name]=d
    return result


def run(out=OUT, *, max_epochs=150):
    import torch
    torch.set_num_threads(8)
    out=Path(out)
    if out.exists():
        forbidden=[out/'results.json',out/'report.md',out/'selected_rows.npz']
        if any(p.exists() for p in forbidden): raise FileExistsError(f'refusing to overwrite completed output under {out}')
        allowed={f'magnitude_head.pt'}|{f'magnitude_head_loto_task{i}.pt' for i in range(len(ex.TASKS))}
        unknown=[p.name for p in out.iterdir() if p.name not in allowed]
        if unknown: raise FileExistsError(f'refusing unknown existing output files under {out}: {unknown}')
    out.mkdir(parents=True,exist_ok=True)
    rk=_load_keylite();z=np.load(CTX/'cache/rows.npz',allow_pickle=False);pool=dict(np.load(CTX/'cache/pool.npz',allow_pickle=False))
    feat=np.asarray(np.load(CTX/'cache/prefix_mean_fp16.npy',mmap_mode='r'),np.float32)
    ck=torch.load(KEYLITE/'context_history.pt',map_location='cpu',weights_only=False)
    if not ck.get('input_history',False):raise AssertionError('context_history checkpoint is not history aware')
    manifest=json.loads((SOURCE/'manifest.json').read_text())
    episode_actions={}
    for d in manifest['demos']:
        episode_actions[(int(d['task_id']),int(d['episode']))]=ex._load_cached_episode(SOURCE,int(d['task_id']),int(d['episode']))[0]
    mem=_make_rows(rk,ck,z,pool,feat,'memory',episode_actions)
    held=_make_rows(rk,ck,z,pool,feat,'heldout',episode_actions)
    if np.intersect1d(np.asarray(z['memory_episode']),np.asarray(z['heldout_episode'])).size:
        raise AssertionError('heldout episodes leaked into memory-side fit rows')
    # Verify that re-running the exact saved Context+history ranker reproduces
    # the reported held-out top-1 type error before fitting the new regressor.
    errmat=np.load(KEYLITE/'heldout_type_errors.npy',mmap_mode='r')
    replay_error=float(np.mean(errmat[np.arange(len(held['pick'])),held['pick']]))
    expected=0.0054465471766889095
    if abs(replay_error-expected)>2e-5:
        raise AssertionError(f'Context+history selection replay mismatch: {replay_error} vs {expected}')
    mem_eps=np.asarray(z['memory_episode'],np.int64);val_eps=set(map(int,ck['dev_episodes']))
    val=np.asarray([int(e) in val_eps for e in mem_eps]);fit_ids=np.flatnonzero(~val);val_ids=np.flatnonzero(val)
    if not len(fit_ids) or not len(val_ids):raise AssertionError('empty memory fit/validation split')
    full_path=out/'magnitude_head.pt'
    if full_path.exists():
        saved=torch.load(full_path,map_location='cpu',weights_only=False)
        model=_MLP.create(mem['x'].shape[1]);model.load_state_dict(saved['state']);model.eval()
        model_ck={'state':saved['state'],'feature_mean':saved['feature_mean'],'feature_std':saved['feature_std'],
            'best_epoch':saved['best_epoch'],'val_logscale_mse':saved['val_logscale_mse'],'label':'all_tasks'}
    else:
        model,model_ck=_fit(mem['x'],mem['s_star'],fit_ids,val_ids,seed=20261111,max_epochs=max_epochs,label='all_tasks')
        torch.save({'state':model_ck['state'],'feature_mean':model_ck['feature_mean'],'feature_std':model_ck['feature_std'],
            'best_epoch':model_ck['best_epoch'],'val_logscale_mse':model_ck['val_logscale_mse'],'fit_episodes':sorted(set(mem_eps[fit_ids].tolist())),
            'validation_episodes':sorted(val_eps),'input_dim':int(mem['x'].shape[1]),'target':'log clipped position s*'},full_path)
    m1=_predict(model,model_ck,held['x'])
    loto_pred=np.empty(len(held['x']),np.float32);loto_info={};mem_task=np.asarray(z['memory_task'],np.int64);held_task=np.asarray(z['heldout_task'],np.int64)
    for task in range(len(ex.TASKS)):
        fit_task=np.flatnonzero(mem_task!=task); eps=np.unique(mem_eps[fit_task])
        val_loto=set(map(int,np.random.default_rng(20261111+task).permutation(eps)[:max(1,len(eps)//5)]))
        isval=np.asarray([int(mem_eps[i]) in val_loto for i in fit_task]); fit_l=fit_task[~isval]; val_l=fit_task[isval]
        loto_path=out/f'magnitude_head_loto_task{task}.pt'
        if loto_path.exists():
            saved=torch.load(loto_path,map_location='cpu',weights_only=False)
            lm=_MLP.create(mem['x'].shape[1]);lm.load_state_dict(saved['state']);lm.eval()
            lck={'state':saved['state'],'feature_mean':saved['feature_mean'],'feature_std':saved['feature_std'],
                'best_epoch':saved['best_epoch'],'label':f'leave_task_{task}'}
            with torch.no_grad():
                vx=(mem['x'][val_l]-lck['feature_mean'])/lck['feature_std']
                vy=np.log(np.clip(mem['s_star'][val_l],.25,4.)).astype(np.float32)
                lck['val_logscale_mse']=float(np.mean((lm(torch.as_tensor(vx,dtype=torch.float32)).numpy()-vy)**2))
        else:
            lm,lck=_fit(mem['x'],mem['s_star'],fit_l,val_l,seed=20261200+task,max_epochs=max_epochs,label=f'leave_task_{task}')
            torch.save({'state':lck['state'],'feature_mean':lck['feature_mean'],'feature_std':lck['feature_std'],
                'best_epoch':lck['best_epoch'],'heldout_task':task,'validation_episodes':sorted(val_loto),'input_dim':int(mem['x'].shape[1])},loto_path)
        ids=np.flatnonzero(held_task==task); loto_pred[ids]=_predict(lm,lck,held['x'][ids])
        loto_info[str(task)]={'heldout_rows':len(ids),'best_epoch':lck['best_epoch'],'val_logscale_mse':lck['val_logscale_mse'],'fit_rows':len(fit_l),'validation_rows':len(val_l)}
        del lm,lck
    refs=np.asarray(z['eval_pi05_actions'],np.float32)
    full=_metric_rows({'M1_regression':m1},held['pick'],pool,z,'heldout',episode_actions,refs=refs)
    loto=_metric_rows({'M1_LOTO':loto_pred},held['pick'],pool,z,'heldout',episode_actions,refs=refs)
    summarized=_summarize(full);summarized.update(_summarize(loto))
    m1_val=_predict(model,model_ck,mem['x'][val_ids]);mem_s=np.asarray(mem['s_star'])[val_ids]
    val_summary={'rows':len(val_ids),'episodes':len(val_eps),'sstar_mae':float(np.mean(np.abs(m1_val-mem_s))),
        'sstar_log_mse':float(np.mean((np.log(np.clip(m1_val,.25,4))-np.log(np.clip(mem_s,.25,4)))**2)),
        'pred_scale_median_q25_q75':[float(np.quantile(m1_val,q)) for q in (.5,.25,.75)],
        'target_scale_median_q25_q75':[float(np.quantile(mem_s,q)) for q in (.5,.25,.75)]}
    np.savez_compressed(out/'selected_rows.npz',memory_pick=mem['pick'],heldout_pick=held['pick'],
        heldout_task=z['heldout_task'],heldout_episode=z['heldout_episode'],heldout_frame=z['heldout_frame'],
        m1_scale=m1,m1_loto_scale=loto_pred,heldout_sstar=held['s_star'],heldout_unit_length=held['unit_length_eval'],
        memory_sstar=mem['s_star'],memory_episode=z['memory_episode'],memory_frame=z['memory_frame'])
    result={'rows':len(held['pick']),'heldout_episodes':int(len(np.unique(z['heldout_episode']))),'memory_rows':len(mem['pick']),
        'selection':'Frozen key-lite-01/context_history.pt top-1; candidate episode masked for every memory and heldout query.',
        'selection_replay_top1_type_error':replay_error,
        'feature_definition':'Context query (normalized final-prefix mean 2048 + policy-order state16), exact Context+history history block (normalized previous actions120 + validity mask10), selected pool key (normalized DCT type18 + log length1), log stored motion RMS1. No heldout actions enter M1 features or checkpoint selection.',
        'training_target':'Memory-side only: log position s* = clipped max(dot(candidate_position, true position)/||candidate_position||²,0), [0.25,4], fitted on at most 50 overlapping frames of the selected unit; same scalar applied to position and rotation.',
        'selection_checkpoint':{'best_epoch':model_ck['best_epoch'],'memory_val_logscale_mse':model_ck['val_logscale_mse'],'fit_rows':len(fit_ids),'val_rows':len(val_ids),'val_episodes':sorted(val_eps),'metrics':val_summary},
        'loto_models':loto_info,'evaluation':summarized,
        'notes':'S2 uses cached pi05 predicted norm ratio only as an offline reference and never as M1 input. Heldout action truth is used only to compute s* reference and reported MSE. Unit horizon is min(selected unit length, episode remainder, cached reference horizon 50); first-five pads a shorter unit with its terminal command. All scales clipped [0.25,4].'}
    _write_json(out/'results.json',result)
    names=('M0_stored','M1_regression','M1_LOTO','M2_history_norm','S2_pi05_norm','s_star')
    lines=['実装: 完了','','# RM-MAG: π0.5を使わない倍率推定','',
        f"保留120デモ・{len(held['pick'])}窓を評価。単位選択は既存key-lite-01のContext+history 1位を固定して再利用した。学習はmemory行{len(mem['pick'])}のみ。heldout正解はMSEとs*の採点だけに用いた。",'',
        f"Context+history選択の再現誤差は{replay_error:.8f}（key-liteの記録値0.00544655と一致）。同一エピソードの単位はmemory・heldoutの両方で選択対象から除外した。",'',
        f"M1入力はprefix平均2048＋state16＋Context+historyの過去行動120・mask10＋選択単位の正規化DCT型/長さ/保存RMS。目的変数は学習行の選択単位に対する位置s*。全体モデルのbest epoch={model_ck['best_epoch']}、memory validation log-scale MSE={model_ck['val_logscale_mse']:.6g}。",'',
        '## 保留評価（先頭5ステップ／選択単位長）','',
        '|方法|5-step位置MSE|5-step 6軸MSE|単位長位置MSE|単位長6軸MSE|final/truth 中央値 [Q1,Q3]|final/π0.5 中央値 [Q1,Q3]|平均比較ステップ|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in names:
        a=summarized[name];rt=a['ratio_final_over_truth'];rp=a['ratio_final_over_pi05']
        lines.append(f"|{name}|{a['first5_pos_mse']:.6f}|{a['first5_all6_mse']:.6f}|{a['unit_pos_mse']:.6f}|{a['unit_all6_mse']:.6f}|{rt['median']:.3f} [{rt['q25']:.3f},{rt['q75']:.3f}]|{rp['median']:.3f} [{rp['q25']:.3f},{rp['q75']:.3f}]|{a['mean_unit_steps']:.1f}|")
    lines += ['', 'M0=保存倍率、M1=全タスクの回帰ヘッド、M1_LOTO=各評価タスクを除いて回帰ヘッドを学習、M2=直前行動の位置ノルム比、S2=π0.5参照の位置ノルム比、s*=正解上限。各条件で同じ倍率を位置・回転に適用した。', '',
        '## タスク別 M1','', '|task id|M1 全体モデル unit 6軸MSE|M1 LOTO unit 6軸MSE|', '|---:|---:|---:|']
    for task in range(len(ex.TASKS)):
        aa=np.asarray([r['unit']['all6'] for r in full['M1_regression'] if r['task']==task]);bb=np.asarray([r['unit']['all6'] for r in loto['M1_LOTO'] if r['task']==task])
        lines.append(f'|{task} ({ex.TASKS[task]})|{aa.mean():.6f}|{bb.mean():.6f}|')
    gap=summarized['M1_regression']['unit_all6_mse']-summarized['S2_pi05_norm']['unit_all6_mse']
    lines += ['', '## 判断','',
        f"π0.5不使用のM1は単位長6軸MSE {summarized['M1_regression']['unit_all6_mse']:.6f}（LOTO {summarized['M1_LOTO']['unit_all6_mse']:.6f}）。S2との差は全タスク学習で{gap:+.6f}（S2比{100*gap/summarized['S2_pi05_norm']['unit_all6_mse']:+.1f}%）、LOTOでは{summarized['M1_LOTO']['unit_all6_mse']-summarized['S2_pi05_norm']['unit_all6_mse']:+.6f}。全タスクM1はS2に近いが低い誤差には届かず、LOTOでは隔たりが大きい。倍率の最終/正解比が1より小さい側は遅め、大きい側は速めの傾向。", '',
        '制限：候補単位の選択は固定済みContext+history top-1。保留行は学習・早期停止に使わず、閉ループの成否も測っていない。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':run()
