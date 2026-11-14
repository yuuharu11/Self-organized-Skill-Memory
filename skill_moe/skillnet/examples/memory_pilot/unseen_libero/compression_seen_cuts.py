#!/usr/bin/env python3
"""CMP3 learned-task/unseen-task compression and speed-cut diagnostics (CPU only)."""
from __future__ import annotations
import argparse, json, math, sys, warnings
from pathlib import Path
import numpy as np
from scipy.signal import peak_prominences
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'pi05_memory')]
import compression_matched as c2
import dct_skill_draft_eval as ev
import motion_core_eval as motion
from draft_reconstruction import ROOT
from variable_length_error import stitch

SEED = 20261115
G1_EXPECT = {'units': 4516, 'all_mse': .034447897620114326, 'hard_mse': .06845424480130637}

def speed(trace, width=3):
    physical = motion.vs.to_physical(np.asarray(trace)[:, :6])
    v = np.linalg.norm(physical[:, :3], axis=1)
    return np.convolve(v, np.ones(width) / width, mode='same')

def extrema(s):
    a=np.asarray(s,np.float64)
    ix=np.flatnonzero((a[1:-1] <= a[:-2]) & (a[1:-1] < a[2:])) + 1
    if len(ix):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            prom=peak_prominences(-a,ix)[0]
    else: prom=np.empty(0,float)
    return ix.astype(int), np.asarray(prom,float)

def cut(trace, mode='all', min_len=5, width=3, prom=None, speed_max=None):
    a, n = np.asarray(trace), len(trace)
    if n == 0: return []
    switches = (np.flatnonzero(np.sign(a[1:, 6]) != np.sign(a[:-1, 6])) + 1).tolist()
    s = speed(a, width); minima, prominence = extrema(s)
    if mode == 'grip': candidates = []
    elif mode == 'all': candidates = minima
    elif mode == 'prom':
        if prom is None: raise ValueError('prominence threshold missing')
        candidates = minima[prominence >= prom]
    elif mode == 'slow':
        if speed_max is None: raise ValueError('speed threshold missing')
        candidates = minima[s[minima] <= speed_max]
    else: raise ValueError(mode)
    switchset, bounds = set(switches), [0]
    for t in sorted(set(map(int, candidates)) | switchset):
        if 0 < t < n and (t in switchset or t - bounds[-1] >= min_len): bounds.append(t)
    if n - bounds[-1] < min_len and len(bounds) > 1 and bounds[-1] not in switchset: bounds.pop()
    return list(zip(bounds, bounds[1:] + [n]))

def make_configs(memory_traces):
    all_prom, all_speed = [], []
    for tr in memory_traces.values():
        s = speed(tr, 3); ix, p = extrema(s)
        all_prom.extend(p.tolist()); all_speed.extend(s.tolist())
    pq = np.quantile(all_prom, [.25, .5, .75]); sq = np.quantile(all_speed, [.1, .2])
    cfg = {'G0_grip': dict(mode='grip'), 'G1_current': dict(mode='all', min_len=5, width=3)}
    cfg.update({f'G2_prom_q{q}': dict(mode='prom', min_len=5, width=3, prom=float(x)) for q,x in zip((25,50,75),pq)})
    cfg.update({f'G3_slow_q{q}': dict(mode='slow', min_len=5, width=3, speed_max=float(x)) for q,x in zip((10,20),sq)})
    cfg.update({'G4_smooth5':dict(mode='all',min_len=5,width=5), 'G4_smooth9':dict(mode='all',min_len=5,width=9),
                'G5_minlen10':dict(mode='all',min_len=10,width=3)})
    return cfg, {'prominence_q25_q50_q75':pq.tolist(),'speed_q10_q20':sq.tolist(),
                 'memory_minima_count':len(all_prom),'memory_frames':len(all_speed)}

def spans(traces, config):
    segs, lens = [], []
    for tr in traces.values():
        # Match CMP2's fixed10 memory pool: discard a trailing incomplete window.
        for s,e in (cut(tr, **config) if config else [(i,i+10) for i in range(0,len(tr)-9,10)]):
            segs.append(np.asarray(tr[s:e,:7],np.float32)); lens.append(e-s)
    return segs, np.asarray(lens,np.int64)

def select_seen(rows, seed=SEED):
    task,ep,frame=np.asarray(rows['task']),np.asarray(rows['episode']),np.asarray(rows['frame'])
    mem,val=np.asarray(rows['memory'],bool),np.asarray(rows['validation'],bool)
    rng=np.random.default_rng(seed); selected=[]
    for lo in range(0,40,10):
        suite=np.arange(lo,lo+10); selected.extend(rng.choice(suite,5,replace=False).tolist())
    demos=[]
    for t in sorted(selected):
        opts=[]
        for e in np.unique(ep[task==t]):
            ix=(task==t)&(ep==e)
            if val[ix].all() and not mem[ix].any(): opts.append(int(e))
        if len(opts)<2: raise AssertionError(f'need two validation/non-memory episodes for task {t}, found {len(opts)}')
        demos.extend((t,int(e)) for e in rng.choice(opts,2,replace=False))
    ids=np.flatnonzero(np.asarray([(int(t),int(e)) in set(demos) for t,e in zip(task,ep)]))
    if mem[ids].any() or not val[ids].all(): raise AssertionError('seen Eval leaks memory or train data')
    keys=np.stack([task[ids],ep[ids]],1)
    traces,_=stitch(np.asarray(rows['actions'][ids,:,:7]),keys,frame[ids])
    rows_of={k:np.flatnonzero(np.all(keys==k,axis=1))[np.argsort(frame[ids][np.all(keys==k,axis=1)])] for k in traces}
    return {'targets':np.asarray(rows['actions'][ids,:,:7],np.float32),'traces':traces,'rows_of':rows_of,
            'task_ids':sorted(selected),'demos':demos,'ids':ids,'task':task[ids],'episode':ep[ids], 'frame':frame[ids]}

def prepare():
    d=c2.prepare(); seen=select_seen(d['rows'])
    unseen={'targets':d['eval'],'traces':d['eval_traces'],'rows_of':d['rows_of'],
            'task_ids':sorted({int(k[0]) for k in d['eval_traces']}),'demos':sorted(d['eval_traces']),
            'hard':d['hard']}
    # Seen hard mask: upper quintile of nearest raw memory-window error.
    _,se=c2.nearest_templates(seen['targets'],d['memory']); seen['hard']=se>=np.quantile(se,.8)
    return d,{'learned':seen,'unseen':unseen}

def unit_meta(d, evals):
    memory_demos=len(d['memory_traces']); memory_tasks=len({int(k[0]) for k in d['memory_traces']})
    return {'memory':{'tasks':memory_tasks,'demos':memory_demos,'windows':len(d['memory'])},
      **{k:{'tasks':len(v['task_ids']),'demos':len(v['demos']),'windows':len(v['targets']),'hard':int(v['hard'].sum())} for k,v in evals.items()}}

def eval_variable(items,lengths,data):
    old=ev.Shared.target
    try:
        ev.Shared.target=data['targets']
        pred,info=c2.score_variable_memory(items,lengths,np.arange(len(items)),data['traces'],data['rows_of'])
    finally: ev.Shared.target=old
    met=c2.error_summary(pred,data['targets'],data['hard'])
    return met,pred,info

def subset_eval(data,n_demos=2):
    keys=sorted(data['traces'])[:n_demos]
    oldids=np.sort(np.concatenate([data['rows_of'][k] for k in keys]))
    new_of={int(old):new for new,old in enumerate(oldids)}
    traces={k:data['traces'][k] for k in keys}
    rows_of={k:np.asarray([new_of[int(i)] for i in data['rows_of'][k]],np.int64) for k in keys}
    return {**data,'targets':data['targets'][oldids],'hard':data['hard'][oldids],
            'traces':traces,'rows_of':rows_of,'demos':keys,
            'task_ids':sorted({int(k[0]) for k in keys})}

def verify_g1(d,unseen):
    cfg={'mode':'all','min_len':5,'width':3}; items=[]; lens=[]
    for _,tr in d['memory_traces'].items():
        for s,e in cut(tr,**cfg):
            vals=motion.encode(tr,None,s,e,'world',1)
            items.append(motion.decode(vals[None],e-s,1)[0].astype(np.float32)); lens.append(e-s)
    if len(items)!=G1_EXPECT['units']:
        raise RuntimeError(f"G1 unit count mismatch {len(items)} != {G1_EXPECT['units']}")
    met,_,_=eval_variable(items,np.asarray(lens),unseen)
    if abs(met['all_mse']-G1_EXPECT['all_mse'])>2e-6 or abs(met['hard_mse']-G1_EXPECT['hard_mse'])>2e-6:
        raise RuntimeError(f'G1 baseline mismatch: {met}; expected {G1_EXPECT}')
    return met

def subset_eval(data,n_demos=2):
    keys=sorted(data['traces'])[:n_demos]
    oldids=np.sort(np.concatenate([data['rows_of'][k] for k in keys]))
    new={int(old):i for i,old in enumerate(oldids)}
    rows_of={k:np.asarray([new[int(i)] for i in data['rows_of'][k]],np.int64) for k in keys}
    return {**data,'targets':data['targets'][oldids],'hard':data['hard'][oldids],
            'traces':{k:data['traces'][k] for k in keys},'rows_of':rows_of,
            'demos':keys,'task_ids':sorted({int(k[0]) for k in keys})}

def fixed_methods(d,evals,smoke):
    mem,fit=d['memory'],d['train']; rows=[]
    for name,v in [('raw',70)]+[(m,v) for v in (14,21,28) for m in ('dct','pca','keyframe')]:
        mean,basis=(c2.pca_fit(fit,v) if name=='pca' else (None,None))
        if name=='raw': rec=mem
        elif name=='dct': rec=c2.dct_decode(c2.dct_encode(mem,v//7),10).astype(np.float32)
        elif name=='keyframe': rec=c2.keyframe_decode(c2.keyframe_encode(mem,v//7),10).astype(np.float32)
        else: rec=c2.pca_restore(c2.pca_project(mem,mean,basis),mean,basis).astype(np.float32)
        out={'method':name,'V':v,'units':len(rec),'values_per_unit':v,'stored_values':len(rec)*v,'eval':{}}
        for split,e in evals.items():
            # In smoke mode, evals already contain the two-demo subset. Keep
            # all of it so the hard subset is represented in both splits.
            q=e['targets']; hard=e['hard']
            ix,err=c2.nearest_templates(q,rec)
            out['eval'][split]={'all_mse':float(err.mean()),'hard_mse':float(err[hard].mean()) if hard.any() else None,
                                'hard_count':int(hard.sum()),'windows':len(q)}
        rows.append(out)
    cfg={'mode':'all','min_len':5,'width':3}
    mem_items,mem_lengths=spans(d['memory_traces'],cfg)
    train_items,_=spans(d['train_traces'],cfg)
    mean,basis=c2.pca_fit(np.stack([c2.phase_resample(x,32) for x in train_items]),21)
    for kind,dim in [('dct',3),('pca',21),('raw32',224)]:
        items=c2.compressed_variable(mem_items,kind,dim,mean if kind=='pca' else None,basis if kind=='pca' else None)
        per={'dct':22,'pca':22,'raw32':225}[kind]
        row={'method':f'anchor_variable_{kind}','V':per-1,'values_per_unit':per,
             'units':len(items),'stored_values':len(items)*per,'eval':{}}
        for split,e in evals.items(): row['eval'][split]=eval_variable(items,mem_lengths,e)[0]
        rows.append(row)
    return rows

def segment_stats(items):
    return {'mean':float(np.mean([len(x) for x in items])),'median':float(np.median([len(x) for x in items])),
            'q25':float(np.quantile([len(x) for x in items],.25)),'q75':float(np.quantile([len(x) for x in items],.75))}

def dct3(items): return [c2.dct_decode(c2.dct_encode(x,min(3,len(x))),len(x)).astype(np.float32) for x in items]

def run_e2(d,evals,cfgs,thresholds,g1):
    all_mem={k:spans(d['memory_traces'],v) for k,v in cfgs.items()}
    n_equal=min(len(v[0]) for v in all_mem.values()); rng=np.random.default_rng(SEED)
    rows=[]
    for name,(units,lengths) in all_mem.items():
        coded=dct3(units); err={}
        for split,e in evals.items(): err[split]=eval_variable(coded,lengths,e)[0]
        sample=np.sort(rng.choice(len(units),n_equal,replace=False)) if len(units)>n_equal else np.arange(len(units))
        phases=np.stack([c2.phase_resample(units[i],32) for i in sample])
        variance={'equalized_units':len(sample),'low_frequency':c2.low_frequency_concentration(units),
                  'between_within_k95':c2.between_within_ratio(phases,95)}
        rows.append({'condition':name,'config':cfgs[name],'unit_count':len(units),'lengths':segment_stats(units),
                     'stored_values_per_unit':22,'stored_values_total':len(units)*22,'upper_error':err,'memory_variance':variance})
    # Fixed10 and condition-specific mean-length references.
    refs=[]
    ref_conditions=[('fixed10',10)]+[(f'fixedmean_{name}',max(1,int(np.floor(segment_stats(v[0])['mean']+.5))))
                                     for name,v in all_mem.items()]
    for label,length in ref_conditions:
        units,ll=spans(d['memory_traces'],None) if length==10 else spans(d['memory_traces'],{'mode':'all','min_len':length,'width':1})
        if length!=10:
            # Exact non-overlapping fixed windows at the requested length.
            units=[]; ll=[]
            for tr in d['memory_traces'].values():
                for s in range(0,len(tr)-length+1,length): units.append(tr[s:s+length,:7]); ll.append(length)
            ll=np.asarray(ll,np.int64)
        coded=dct3(units); metrics={s:eval_variable(coded,ll,e)[0] for s,e in evals.items()}
        refs.append({'condition':label,'window_length':length,'unit_count':len(units),'upper_error':metrics})
    return {'thresholds_from_memory':thresholds,'conditions':rows,'fixed_references':refs,
            'variance_input_count_equalized':n_equal,'G1_reproduction':g1}

def render(result,output):
    sets=result['evaluation_sets']
    full_output = output.parent / 'attempt-06' if result['mode']=='smoke' else output
    lines=['# CMP3：学習済み／未学習タスクと可変長カット','',
      '誤差は active7 正規化空間、10フレーム窓で正解行動との差を測る。正解は最近傍単位列の選択・採点のみ。圧縮の基底や速度しきい値は LIBERO-40 の学習・メモリ側だけで決めた。','',
      f"メモリ：{sets['memory']['tasks']}タスク・{sets['memory']['demos']}デモ・{sets['memory']['windows']}窓。",
      f"この実行の学習済み評価：{sets['learned']['tasks']}タスク・{sets['learned']['demos']}デモ・{sets['learned']['windows']}窓（hard {sets['learned']['hard']}）。未学習：{sets['unseen']['tasks']}タスク・{sets['unseen']['demos']}デモ・{sets['unseen']['windows']}窓（hard {sets['unseen']['hard']}）。",
      f"全評価予定件数：学習済み {result['full_evaluation_counts']['learned']['tasks']}タスク・{result['full_evaluation_counts']['learned']['demos']}デモ・{result['full_evaluation_counts']['learned']['windows']}窓（hard {result['full_evaluation_counts']['learned']['hard']}）、未学習 {result['full_evaluation_counts']['unseen']['tasks']}タスク・{result['full_evaluation_counts']['unseen']['demos']}デモ・{result['full_evaluation_counts']['unseen']['windows']}窓（hard {result['full_evaluation_counts']['unseen']['hard']}）。",
      ('これはスモーク実行で、各評価集合から先頭2デモだけを使用。以下の数値は本番評価ではない。' if result['mode']=='smoke' else '本番評価。'), '',
      '## E1 固定10窓・G1 可変長：学習済み 対 未学習','',
      f"誤差：active7 正規化空間での10フレーム窓と正解との差。メモリは {sets['memory']['tasks']}タスク・{sets['memory']['demos']}デモ・{sets['memory']['windows']}窓。学習済み評価は {sets['learned']['tasks']}タスク・{sets['learned']['demos']}デモ・{sets['learned']['windows']}窓、未学習評価は {sets['unseen']['tasks']}タスク・{sets['unseen']['demos']}デモ・{sets['unseen']['windows']}窓。", '',
      '| method | V | values/unit | learned all/hard | unseen all/hard |','|---|---:|---:|---:|---:|']
    for r in result['E1']:
        a,b=r['eval']['learned'],r['eval']['unseen']; lines.append(f"| {r['method']} | {r['V']} | {r['values_per_unit']} | {a['all_mse']:.5f}/{a['hard_mse']} | {b['all_mse']:.5f}/{b['hard_mse']} |")
    lines += ['', 'M1 b12/b24 は再利用 codec ファイルがこの環境に存在しなかったため計算対象外。既存 † は視覚特徴の向きに関する注記で、行動のみの圧縮復元には直接関係しない。']
    lines += ['', '## E2 可変長の切り方','',
      f"誤差：各可変長単位を DCT3 復元し、10フレームに1回の選択予算で再構成した active7 正規化MSE。メモリは {sets['memory']['tasks']}タスク・{sets['memory']['demos']}デモ・{sets['memory']['windows']}窓。学習済み評価は {sets['learned']['tasks']}タスク・{sets['learned']['demos']}デモ・{sets['learned']['windows']}窓、未学習評価は {sets['unseen']['tasks']}タスク・{sets['unseen']['demos']}デモ・{sets['unseen']['windows']}窓。速度・prominence のしきい値は LIBERO-40 メモリだけから算出。分散比は各条件で同数 {result['E2']['variance_input_count_equalized']}単位を使用。", '',
      '| Condition | units | length mean/median [q25,q75] | learned all/hard | unseen all/hard | lowfreq mean/median | between/within k95 |','|---|---:|---|---:|---:|---:|---:|']
    for r in result['E2']['conditions']:
        l=r['lengths']; a,b=r['upper_error']['learned'],r['upper_error']['unseen']; v=r['memory_variance']['low_frequency']
        lines.append(f"| {r['condition']} | {r['unit_count']} | {l['mean']:.1f}/{l['median']:.1f} [{l['q25']:.1f},{l['q75']:.1f}] | {a['all_mse']:.5f}/{a['hard_mse']} | {b['all_mse']:.5f}/{b['hard_mse']} | {v['mean']:.3f}/{v['median']:.3f} | {r['memory_variance']['between_within_k95']['ratio']:.3f} |")
    lines += ['', f"固定窓参照：完全な非重複窓のみ（末尾の10未満は除外）。DCT3復元・同じ選択予算・active7 正規化MSE。メモリは {sets['memory']['tasks']}タスク・{sets['memory']['demos']}デモ・{sets['memory']['windows']}窓、学習済み評価は {sets['learned']['tasks']}タスク・{sets['learned']['demos']}デモ・{sets['learned']['windows']}窓、未学習評価は {sets['unseen']['tasks']}タスク・{sets['unseen']['demos']}デモ・{sets['unseen']['windows']}窓。", '', '固定10および可変単位の平均長に合わせた固定窓参照（same oracle scoring）：','', '| window length | memory units | learned all/hard | unseen all/hard |','|---:|---:|---:|---:|']
    for r in result['E2']['fixed_references']:
        a,b=r['upper_error']['learned'],r['upper_error']['unseen']; lines.append(f"| {r['condition']} ({r['window_length']}) | {r['unit_count']} | {a['all_mse']:.5f}/{a['hard_mse']} | {b['all_mse']:.5f}/{b['hard_mse']} |")
    lines += ['', '再現コマンド：','', '```bash','cd /work && source env/activate.sh',
      f"taskset -c 8-47 nice -n 10 env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 python skill_moe/skillnet/examples/memory_pilot/unseen_libero/compression_seen_cuts.py --full --output {full_output}",'```','']
    return '\n'.join(lines)

def run(smoke,output):
    d,evals=prepare(); cfgs,thresholds=make_configs(d['memory_traces'])
    full_counts={k:{'tasks':len(v['task_ids']),'demos':len(v['demos']),'windows':len(v['targets']),
                    'hard':int(v['hard'].sum())} for k,v in evals.items()}
    g1=verify_g1(d,evals['unseen'])
    if smoke:
        evals={k:subset_eval(v,2) for k,v in evals.items()}
    e1=fixed_methods(d,evals,smoke)
    e2=run_e2(d,evals,cfgs,thresholds,g1)
    result={'mode':'smoke' if smoke else 'full','seed':SEED,'evaluation_sets':unit_meta(d,evals),
      'full_evaluation_counts':full_counts,
      'protocol':{'learned_eval':'20 balanced LIBERO-40 tasks, one validation non-memory demo/task, seed 20261115',
       'unseen_eval':'CMP2 LIBERO-90 Eval, 10 tasks/100 demos/619 windows',
       'hard':'evaluation-specific top quintile of nearest raw-memory fixed10 MSE',
       'cuts_and_thresholds_fit_from':'LIBERO-40 memory traces only','G1_expected':G1_EXPECT,
       'G1_reproduction':g1},'E1':e1,'E2':e2}
    output.mkdir(parents=True,exist_ok=False)
    (output/'results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    (output/'report.md').write_text(render(result,output))
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__); g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--smoke',action='store_true'); g.add_argument('--full',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'unseen-libero'/'compress-seen-cuts-01'/'attempt-01'); a=p.parse_args()
    r=run(a.smoke,a.output); print(json.dumps({'output':str(a.output),'mode':r['mode'],'G1':r['protocol']['G1_reproduction'],'sets':r['evaluation_sets']},indent=2))

if __name__=='__main__': main()
