#!/usr/bin/env python3
"""SEG-V CPU preparation: speed-only LIBERO-40 memory segmentation and oracle audit."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.fft import dct, idct

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent/'pi05_memory')]
import key_shared_latent as k2
import kq2_data as kd
import motion_core_eval as motion
import gripper_split as grip
import dev_v2
import kq2cl_common as C

STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', Path(__file__).resolve().parents[5] / '.skillnet-state')).expanduser()
ROOT=STATE_DIR / 'outputs/memory-pilot/unseen-libero'
RUN=STATE_DIR / 'outputs/memory-pilot/pi05-memory40-20261104/run-01'
OUT=STATE_DIR / 'outputs/memory-pilot/unseen-libero/segv-01'
KQDATA=ROOT/'kq2-01/data'
OLD_BANK=ROOT/'key-motion-01/final/data'
MIN_LENGTH=5


def log(stage:str,**kw):
    OUT.mkdir(parents=True,exist_ok=True)
    row={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'stage':stage,**kw}
    with (OUT/'progress.log').open('a') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')


def speed_only_spans(trace:np.ndarray,min_length:int=MIN_LENGTH)->list[tuple[int,int]]:
    """motion_core_eval.anchor_spans with only the gripper-switch cuts removed."""
    a=np.asarray(trace,np.float32)
    if a.ndim!=2 or a.shape[1]<7:raise ValueError('trace must be [T,>=7]')
    n=len(a)
    if n==0:return []
    speed=np.linalg.norm(grip._to_physical(a[:,:6])[:,:3],axis=1)
    smooth=np.convolve(speed,np.ones(3,dtype=np.float64)/3,mode='same')
    minima=[t for t in range(1,n-1) if smooth[t]<=smooth[t-1] and smooth[t]<smooth[t+1]]
    bounds=[0]
    for t in sorted(set(minima)):
        if t-bounds[-1]>=min_length:bounds.append(t)
    if n-bounds[-1]<min_length and len(bounds)>1:bounds.pop()
    return list(zip(bounds,bounds[1:]+[n],strict=True))


def _trace_index(rows,memory_ids):
    all_groups={}
    for i,(t,e) in enumerate(zip(rows['task'],rows['episode'],strict=True)):
        all_groups.setdefault((int(t),int(e)),[]).append(i)
    memory_set=set(map(int,memory_ids));traces={};row_ids={}
    for pair,ix0 in all_groups.items():
        ix=np.asarray(sorted(ix0,key=lambda j:int(rows['frame'][j])),np.int64)
        if not any(int(j) in memory_set for j in ix):continue
        # Memory manifest defines complete demonstrations; assert no mixed episode membership.
        is_mem=np.asarray([int(j) in memory_set for j in ix])
        if not is_mem.all():continue
        trace=np.asarray(rows['actions'][ix,:,:7],np.float32).reshape(-1,7)
        traces[pair]=(ix,trace,int(rows['frame'][ix[0]]));row_ids[pair]=ix
    return traces,row_ids


def _make_bank(rows,memory_ids,*,speed_only:bool):
    import variable_shape_memory as vs
    traces,row_ids=_trace_index(rows,memory_ids)
    records=[]
    # Match grip._variable_pool's deterministic demonstration order, then span order.
    for pair,(ix,trace,base) in traces.items():
        spans=speed_only_spans(trace) if speed_only else motion.anchor_spans(trace)
        for start,end in spans:
            if end<=start:continue
            raw=trace[start:end]
            physical=grip._to_physical(raw[:,:6])
            # Stored values are the same natural-length arm DCT3 reconstruction as CMP3.
            coeff=dct(physical,type=2,norm='ortho',axis=0)
            coeff[3:]=0
            recon_phys=idct(coeff,type=2,norm='ortho',axis=0)
            recon_norm=grip._to_norm(recon_phys).astype(np.float32)
            sign=1. if np.mean(raw[:,6])>=0 else -1.
            natural=np.column_stack([recon_norm,np.full(len(raw),sign,np.float32)]).astype(np.float32)
            coeff3,mag,length=__import__('seg_context_retrieval').segment_value_encode(natural,coefficients=3)
            base_frame=base+start;rowid=int(ix[min(len(ix)-1,start//10)])
            # Store nearest row/cache indexes by exact immutable key, not by order assumptions.
            records.append({'task':pair[0],'episode':pair[1],'start':start,'end':end,'base':base_frame,
                'source':(pair[0],pair[1],base_frame,base_frame+(end-start)),
                'coefficients':coeff3,'magnitude':mag,'length':length,'natural_actions':natural,
                'value_actions10':grip._resample10(natural).astype(np.float32),
                'raw_actions':raw.astype(np.float32),'row_id':rowid,
                'start_velocity':grip._to_physical(raw[:1,:6])[0,:3].astype(np.float32),
                'task':pair[0],'episode':pair[1]})
    return records,traces,row_ids


def _pad_bank(records):
    n=len(records);width=max(len(r['natural_actions']) for r in records)
    natural=np.zeros((n,width,7),np.float32)
    for i,r in enumerate(records):natural[i,:len(r['natural_actions'])]=r['natural_actions']
    coeff=np.stack([r['coefficients'] for r in records]).astype(np.float32)
    mag=np.asarray([r['magnitude'] for r in records],np.float32)
    length=np.asarray([r['length'] for r in records],np.int32)
    bank={
        'coefficients':coeff,'magnitude':mag,'length':length,
        'task':np.asarray([r['task'] for r in records],np.int32),
        'episode':np.asarray([r['episode'] for r in records],np.int32),
        'start':np.asarray([r['start'] for r in records],np.int32),
        'source':np.asarray([r['source'] for r in records],np.int32),
        'row_id':np.asarray([r['row_id'] for r in records],np.int64),
        'start_velocity':np.stack([r['start_velocity'] for r in records]).astype(np.float32),
        'value_actions10':np.stack([r['value_actions10'] for r in records]).astype(np.float32),
        'natural_actions':natural}
    bank['bank_actions']=k2._candidate_window_actions(np.arange(n),bank).astype(np.float32)
    return bank


def _episode_context(bank,traces):
    hist=[];mask=[];states=[];vlmkeys=[]
    for t,e,frame,_end in bank['source']:
        ix,trace,base=traces[(int(t),int(e))];f=int(frame)-base
        h=np.zeros((10,7),np.float32);m=np.zeros(10,bool);lo=max(0,f-10);n=f-lo
        if n:h[-n:]=trace[lo:f];m[-n:]=True
        state=np.asarray(kd._state_for_row(rows=None),np.float32) if False else None
        # The matching state row is the 10 Hz frame containing this start.
        rid=int(ix[min(len(ix)-1,max(0,f//10))]);states.append(rid)
        hist.append(h);mask.append(m);vlmkeys.append((int(t),int(e),int(frame)))
    return np.asarray(hist),np.asarray(mask),np.asarray(states,np.int64),np.asarray(vlmkeys,np.int32)


def _feature_indices(keys:np.ndarray)->dict[str,np.ndarray]:
    """Exact cache row references for per-unit VLM observations (never embeddings)."""
    sources=[];indices=[];usedkeys=[]
    caches=[]
    for name,path in (('run',RUN),('missing',ROOT/'key-shared-01/attempt-02/prefix-missing'),
                      ('seen5',ROOT/'seg-context-01/seg5-cache')):
        try:
            if name=='run':
                rr=k2._load_key2_rows(path);ck=np.column_stack([rr['task'],rr['episode'],rr['frame']])
            else:ck=np.asarray(json.loads((path/'manifest.json').read_text())['keys'],np.int32)
            mp={tuple(map(int,k)):i for i,k in enumerate(ck)}
            by_episode={}
            for (t,e,f),idx in mp.items():by_episode.setdefault((t,e),[]).append((f,idx))
            for k in by_episode:by_episode[k].sort()
            caches.append((name,mp,by_episode))
        except (FileNotFoundError,KeyError):continue
    for k in keys:
        key=tuple(map(int,k));found=None;used=None
        # Search every cache for an exact observation before accepting an older
        # row from an earlier cache. Cache order is only a tie breaker.
        for nm,mp,by_episode in caches:
            if key in mp:found=(nm,mp[key]);used=key;break
        if found is None:
            for nm,mp,by_episode in caches:
                ep=by_episode.get(key[:2],[]);prior=[(f,i) for f,i in ep if f<=key[2]]
                if prior:
                    f,i=prior[-1]
                    if used is None or f>used[2]:
                        found=(nm,i);used=(key[0],key[1],f)
        if found is None:sources.append('missing');indices.append(-1);usedkeys.append(key)
        else:sources.append(found[0]);indices.append(found[1]);usedkeys.append(used)
    return {'vlm_source':np.asarray(sources,dtype='U8'),'vlm_index':np.asarray(indices,np.int64),
            'vlm_key':np.asarray(usedkeys,np.int32)}


def _switch_internal(records,traces):
    distances=[];n_switch=0
    # Count proximity on full demonstrations so anchor-boundary switches remain included.
    for _pair,(_ix,trace,_base) in traces.items():
        sw=np.flatnonzero(np.sign(trace[1:,6])!=np.sign(trace[:-1,6]))+1
        n_switch+=len(sw)
        mins=[s for s,e in speed_only_spans(trace) if s>0]
        for s in sw:
            ds=[abs(int(s)-int(x)) for x in mins]
            distances.append(min(ds) if ds else None)
    internal_units=sum(bool(np.any(np.sign(r['raw_actions'][1:,6])!=np.sign(r['raw_actions'][:-1,6]))) for r in records)
    total=int(sum(len(trace) for _,trace,_ in traces.values()))
    return {'switch_count':n_switch,'units_with_internal_switch':int(internal_units),
        'units_with_internal_switch_fraction':float(np.mean([np.any(np.sign(r['raw_actions'][1:,6])!=np.sign(r['raw_actions'][:-1,6])) for r in records])),
        'switches_within_minimum_distance':{str(k):float(np.mean([d is not None and d<=k for d in distances])) if distances else None for k in (1,2,5,10)},
        'switch_minimum_distance_median':float(np.median([d for d in distances if d is not None])) if any(d is not None for d in distances) else None,
        'trace_steps':total}


def _reconstruction(records,traces):
    by_ep={}
    for r in records:by_ep.setdefault((r['task'],r['episode']),[]).append(r)
    all_sq=[];near=[];switchcount=0
    for pair,(_ix,trace,_base) in traces.items():
        recon=np.zeros_like(trace,dtype=np.float32)
        for r in by_ep.get(pair,[]):recon[r['start']:r['end']]=r['natural_actions']
        err=np.square(trace[:,:6]-recon[:,:6]).mean(-1);all_sq.extend(err.tolist())
        sw=np.flatnonzero(np.sign(trace[1:,6])!=np.sign(trace[:-1,6]))+1
        for s in sw:
            switchcount+=1;lo=max(0,int(s)-5);hi=min(len(trace),int(s)+5)
            if lo<hi:near.extend(err[lo:hi].tolist())
    return {'arm6_mse_all_frames':float(np.mean(all_sq)),'arm6_mse_switch_pm5':float(np.mean(near)) if near else None,
            'switches':switchcount,'frames':len(all_sq),'switch_band_frames':len(near)}


def _stats(records,traces):
    lengths=np.asarray([len(r['raw_actions']) for r in records],int)
    return {'units':len(records),'length':{'min':int(lengths.min()),'p10':float(np.quantile(lengths,.1)),
        'median':float(np.median(lengths)),'mean':float(lengths.mean()),'p90':float(np.quantile(lengths,.9)),'max':int(lengths.max())},
        **_switch_internal(records,traces),**{'reconstruction':_reconstruction(records,traces)}}


def _load_split(name):
    d=KQDATA/name
    files=['key','task','episode','actions','history','history_mask']
    if name=='dev2':
        sp=dev_v2.load_dev_v2_20()
    elif name=='unseen20':
        import vq_geometry
        sp=vq_geometry.load_rate_matched('unseen20')
    else:
        return {k:np.load(d/f'{k}.npy',allow_pickle=False) for k in files}
    if name in ('dev2','unseen20'):
        local={k:np.load(d/f'{k}.npy',allow_pickle=False) for k in files}
        for k in ('key','task','episode','actions','history','history_mask'):
            if not np.array_equal(sp[k],local[k]):raise AssertionError(f'{name} 20Hz field mismatch {k}')
        return sp


def _subsets(name,split):
    if name=='dev2':
        sub={k[4:]:np.asarray(v,bool) for k,v in split.items() if k.startswith('sub_')}
    else:
        sub=dev_v2.eval_subsets(split)
    if 'middle' not in sub:
        excluded=np.zeros(len(split['key']),bool)
        for name0 in ('first_unit','post_switch','dir_change'):
            if name0 in sub:excluded|=sub[name0]
        sub['middle']=~excluded
    return sub


def _oracle(bank,split,split_name):
    windows=bank['bank_actions'];errors=k2._arm6_candidate_mse(split['actions'],windows)
    same=((np.asarray(split['task'])[:,None]==bank['task'][None,:]) &
          (np.asarray(split['episode'])[:,None]==bank['episode'][None,:]))
    errors=np.where(same,np.inf,errors)
    top=errors.argmin(1);per=errors[np.arange(len(top)),top]
    ids=np.flatnonzero(split['history_mask'].any(1))
    switch=C.switch_metrics(windows[top,:,6],split['actions'][:,:,6],
        np.asarray([h[np.flatnonzero(m)[-1],6] if m.any() else 0 for h,m in zip(split['history'],split['history_mask'],strict=True)]),
        split['history_mask'].any(1))
    sub=_subsets(split_name,split)
    groups={'all':np.ones(len(per),bool),**sub}
    return {'rows':len(per),'oracle_arm6_mse':float(per.mean()),'subsets':{k:{'rows':int(v.sum()),'oracle_arm6_mse':float(per[v].mean()) if v.any() else None} for k,v in groups.items()},
        'gripper_from_oracle_unit_as_saved':switch,'per_query_error':per.astype(np.float32)}


def build(*,out:Path=OUT,smoke:bool=False):
    if out.exists() and any(out.iterdir()):raise FileExistsError(f'refuse overwrite: {out}')
    out.mkdir(parents=True,exist_ok=True);log('build_rows_start')
    rows=k2._load_key2_rows(RUN)
    memory_ids=np.flatnonzero(np.asarray(rows['memory'],bool))
    import variable_shape_memory as vs
    # Original anchor bank: same helper/order and fixed 4,516-unit cardinality as KQ2.
    old_pool,old_meta,old_raw=grip._variable_pool(rows,memory_ids)
    if len(old_meta)!=4516:raise AssertionError(f'expected original 4516 units, got {len(old_meta)}')
    log('original_bank_rebuilt',units=len(old_meta),memory_rows=len(memory_ids))
    old_records,old_traces,_=_make_bank(rows,memory_ids,speed_only=False)
    new_records,new_traces,row_ids=_make_bank(rows,memory_ids,speed_only=True)
    old_source=np.asarray([r['source'] for r in old_records],np.int32)
    canonical_source=np.load(OLD_BANK/'bank__source.npy',mmap_mode='r',allow_pickle=False)
    if not smoke and (old_source.shape!=canonical_source.shape or not np.array_equal(old_source,canonical_source)):
        mismatch=np.flatnonzero(np.any(old_source!=canonical_source,axis=1))[:5] if old_source.shape==canonical_source.shape else []
        raise AssertionError(f'rebuilt anchor order/source differs from canonical KQ2 bank: {old_source.shape} vs {canonical_source.shape}, {mismatch}')
    if smoke:
        old_records=old_records[:256];new_records=new_records[:256]
    old=_pad_bank(old_records);new=_pad_bank(new_records)
    if not smoke:
        # Sources alone cannot detect normalization or playback-format drift.
        for field,filename in (('source','bank__source.npy'),
                               ('coefficients','bank__coefficients.npy'),
                               ('natural_actions','bank__natural_actions.npy'),
                               ('bank_actions','bank_actions.npy')):
            expected=np.load(OLD_BANK/filename,mmap_mode='r',allow_pickle=False)
            if not np.array_equal(old[field],expected):
                raise AssertionError(f'rebuilt anchor {field} differs from canonical bank')
    traces=new_traces
    # Start context and exact VLM cache references.
    hist=[];hm=[];state_ids=[];unitkeys=[]
    for rec in new_records:
        ix,trace,base=traces[(rec['task'],rec['episode'])];f=rec['start'];h=np.zeros((10,7),np.float32);m=np.zeros(10,bool)
        lo=max(0,f-10);n=f-lo
        if n:h[-n:]=trace[lo:f];m[-n:]=True
        rid=int(ix[min(len(ix)-1,max(0,f//10))]);hist.append(h);hm.append(m);state_ids.append(rid)
        unitkeys.append((rec['task'],rec['episode'],rec['base']))
    # State values and row ids are aligned to original KQ2 key-final bank format.
    rows['state']=np.asarray(rows['state'])
    new['history']=np.asarray(hist,np.float32);new['history_mask']=np.asarray(hm,bool)
    new['state']=np.asarray(k2._exact_start_states(np.asarray(unitkeys,np.int32),rows,RUN,ROOT/'key-shared-01/attempt-02/prefix-missing'),np.float32)
    new['state_row_id']=np.asarray(state_ids,np.int64)
    new['unit_key']=np.asarray(unitkeys,np.int32);new.update(_feature_indices(new['unit_key']))
    outdata=out/'bank';outdata.mkdir(parents=True,exist_ok=True)
    for prefix,bank in (('anchor',old),('speed',new)):
        for k,v in bank.items():np.save(outdata/f'{prefix}__{k}.npy',v,allow_pickle=False)
    # Raw immutable boundary manifest allows exact audit/rebuild.
    oldsrc=old['source'];newsrc=new['source']
    _save_json(outdata/'manifest.json',{'complete':True,'speed_only':True,'source_rows':str(RUN/'rows'),
        'memory_rows':len(memory_ids),'original_units':len(old_records),'speed_only_units':len(new_records),
        'speed_source_order':'same sorted episode order as gripper_split._variable_pool; spans chronological',
        'anchor_source_order':'same demonstration set/order; anchor_spans rule','speed_only_spans':'motion_core_eval smoothing/minima/min length/tail, switch cuts removed',
        'state_field':'exact normalized state at unit start (row cache or source parquet)','vlm_index':'latest cached VLM row at or before unit start, with its exact key saved as vlm_key; -1 means absent'})
    log('banks_saved',anchor_units=len(old_records),speed_units=len(new_records),bank_dir=str(outdata))
    stats={'anchor':_stats(old_records,old_traces),'speed_only':_stats(new_records,new_traces)}
    # Oracle: full candidate bank, locked learned, true 20Hz unseen and Dev v2.
    oracles={}
    for name in ('learned','unseen20','dev2'):
        sp=_load_split(name)
        # Keep full source labels even in smoke; smoke oracle subsets only first 32 rows, explicitly tagged.
        if smoke:sp={k:(v[:32] if isinstance(v,np.ndarray) and len(v)==len(sp['key']) else v) for k,v in sp.items()}
        oracles[name]={}
        for bname,bank in (('anchor',old),('speed_only',new)):
            r=_oracle(bank,sp,name);r.pop('per_query_error',None);oracles[name][bname]=r
        log('oracle_split_complete',split=name,rows=len(sp['key']),anchor=oracles[name]['anchor']['oracle_arm6_mse'],speed_only=oracles[name]['speed_only']['oracle_arm6_mse'])
    result={'status':'complete_smoke' if smoke else 'cpu_complete','stats':stats,'oracle':oracles,
        'source':{'rows':str(RUN/'rows'),'memory_demo_rows':len(memory_ids),'anchor_bank_units':len(old_records),'speed_only_units':len(new_records)},
        'gpu_stage':'pending KQ2-CL DONE; no GPU process launched'}
    _save_json(out/'cpu_results.json',result)
    _write_report(out,result)
    log('cpu_stage_complete',status=result['status'])
    return result


def _save_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)


def _write_report(out,result):
    import datetime
    now=datetime.datetime.now(datetime.timezone.utc)
    utc=now.strftime('%Y-%m-%d %H:%M:%S UTC');jst=(now+datetime.timedelta(hours=9)).strftime('%Y-%m-%d %H:%M:%S JST')
    lines=['# SEG-V CPU 段レポート','',f'生成時刻: {jst} / {utc}','','## 単位数・長さ・グリッパー切替','','|バンク|単位数|長さ min / p10 / median / mean / p90 / max|単位内切替率|切替点から極小点までの距離|DCT3腕MSE 全体 / 切替±5|',
        '|---|---:|---|---:|---|---:|']
    for key,label in (('anchor','現行 anchor_spans'),('speed_only','speed_only_spans')):
        s=result['stats'][key];l=s['length'];r=s['reconstruction'];d=s['switches_within_minimum_distance']
        dist=' / '.join(f'{k}:{d[k]:.3f}' if d[k] is not None else f'{k}:n/a' for k in d)
        lines.append(f"|{label}|{s['units']}|{l['min']} / {l['p10']:.1f} / {l['median']:.1f} / {l['mean']:.1f} / {l['p90']:.1f} / {l['max']}|{s['units_with_internal_switch_fraction']:.3f}|{dist}|{r['arm6_mse_all_frames']:.6f} / {r['arm6_mse_switch_pm5'] if r['arm6_mse_switch_pm5'] is not None else '—'}|")
    lines += ['','','## Oracle 上限：腕6軸 MSE','','|分割|バンク|全体|最初の単位|切替直後|向き変化|途中|','|---|---|---:|---:|---:|---:|---:|']
    for split,rows in result['oracle'].items():
        for bank,r in rows.items():
            ss=r['subsets'];lines.append(f"|{split}|{bank}|{r['oracle_arm6_mse']:.6f}|{ss.get('first_unit',{}).get('oracle_arm6_mse')}|{ss.get('post_switch',{}).get('oracle_arm6_mse')}|{ss.get('dir_change',{}).get('oracle_arm6_mse')}|{ss.get('middle',{}).get('oracle_arm6_mse')}|")
    lines += ['','','## 保存値グリッパーの切替指標','', '各クエリは腕6軸 oracle で選んだ単位を使い、その保存済みグリッパー列を評価した。','', '|分割|バンク|見逃し率|誤切替率|切替時刻誤差（step）|','|---|---|---:|---:|---:|']
    for split,rows in result['oracle'].items():
        for bank,r in rows.items():
            g=r['gripper_from_oracle_unit_as_saved']
            lines.append(f"|{split}|{bank}|{g['missed_switch_rate']}|{g['false_switch_rate']:.4f}|{g['timing_error_mean_steps']}|")
    lines += ['','','## 実装と本番段の準備',"", "speed_only は `motion_core_eval.anchor_spans` と同じ平滑化（3点移動平均）、局所極小点、最小長5、末尾短区間の処理を使い、グリッパー切替境界だけを除いた。学習対象は元と同じ `rows['memory']` の LIBERO-40 デモ。state/history とVLM行参照も各 unit start key で整列保存した。",
        '', 'KQ2-CL の DONE marker がなく GPU 段は未実行（GPU 使用率も高いため待機）。`segv_gpu.py` は誤起動防止の preflight と手順 manifest のみで、再学習・Expert・閉ループの実行アダプターは未実装。KQ2-CL DONE 後にこの段を実行するには、そのアダプター実装が別途必要。', '', 'ログ時刻はUTC。報告時刻はJST（UTC+9）を併記。', '']
    (out/'stage0_report.md').write_text('\n'.join(lines))


def main():
    global OUT
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=OUT);ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args();OUT=a.out
    build(out=OUT,smoke=a.smoke)


if __name__=='__main__':main()
