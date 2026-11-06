#!/usr/bin/env python3
"""Select Dev M6 grains and make one frozen final Long pass for shape-skills A."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

import run as shape
from data import HELDOUT_TASKS, LONG_TASKS
from representations import m6_fit_generic

GRAIN_TARGETS = (30, 300, 3000)
LOW_SUPPORT_KAPPAS = (1e-5, 1e-4, 1e-3, 1e-2, .05, .2, .8, 3.2)


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def records_from_bundle(out, cfg):
    d = out / 'representations' / cfg
    z = np.load(d / 'segments.npz', allow_pickle=False)
    c = np.load(d / 'codes.npy', mmap_mode='r', allow_pickle=False)
    return z, c


def summary_point(cfg, target, model, codes, poses, mem, m6, qsegments,
                  qids, rows, episodes, lookup, pooled, q01, q99, *, support, episodes_min):
    m6['codes'] = codes
    res = shape.eval_config(model, codes, poses, mem, m6, qsegments, qids, rows,
                            episodes, lookup, pooled, q01, q99, seed=8401)
    if res is None or len(res['row_ids']) != len(qids):
        raise AssertionError(f'{cfg} target {target}: Dev coverage mismatch')
    mse = np.asarray(res['mse_aligned'], np.float64)
    random_mse = np.asarray(res['mse_random_aligned'], np.float64)
    row = {'config': cfg, 'target_units': int(target), 'units': int(m6['registered']),
           'unit_distance_from_target': int(abs(int(m6['registered']) - int(target))),
           'kappa': float(m6['grain_kappa']), 'min_support': int(support),
           'min_episodes': int(episodes_min), 'dev_n': int(len(mse)),
           'dev_aligned_mse': float(mse.mean()),
           'dev_random_aligned_mse': float(random_mse.mean()),
           'dev_aligned_minus_random': float(np.mean(mse - random_mse)),
           'dev_aligned_le_0p04': float(np.mean(mse <= .04)),
           'dev_episode_count': int(len(np.unique(rows['episode'][qids])))}
    return row, res


def make_figures(out, scores, grain_records, segment_stats, long_records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir=out/'figures';figdir.mkdir(parents=True,exist_ok=True)
    colors={'pca_raw':'#1f77b4','pca_raw_white':'#17becf','pca_canonical':'#ff7f0e',
            'pca_gripsep':'#bcbd22','procrustes_pca':'#9467bd','dmp':'#2ca02c',
            'spline':'#d62728','m1':'#8c564b','vq':'#e377c2','ae':'#7f7f7f'}
    marker_map={'S0_fixed10':'s','S1_greedy':'o','S1_dp':'^','S2_grip_velocity':'v',
                'S2_grip':'D','S3_type_change':'P'}
    def marker_for(name):
        for prefix,marker in marker_map.items():
            if str(name).startswith(prefix):return marker
        return 'o'
    fig,ax=plt.subplots(figsize=(10,6))
    allrows=list(scores)
    # Relaxed-support grain points extend the count range where standard M6
    # could not reach; both are shown without forcing unit counts.
    allrows += grain_records
    for family,color in colors.items():
        pts=[r for r in allrows if r.get('family')==family]
        for seg in sorted(set(str(r.get('segmentation','')) for r in pts)):
            sub=[r for r in pts if r.get('segmentation')==seg and int(r.get('units',0))>0]
            if not sub:continue
            ax.scatter([r['units'] for r in sub],[r.get('dev_aligned_mse',np.nan) for r in sub],
                       c=color,marker=marker_for(seg),s=17,alpha=.65,
                       label=f'{family} / {seg}')
    ax.set_xscale('log');ax.set_xlabel('M6 unit count (threshold determined)')
    ax.set_ylabel('Heldout Dev oracle-aligned 10-step active7 MSE')
    ax.grid(True,alpha=.25);ax.legend(fontsize=5,ncol=3,loc='best');fig.tight_layout()
    fig.savefig(figdir/'unit_count_dev_distance.png',dpi=160);plt.close(fig)

    fig,ax=plt.subplots(figsize=(11,5))
    seg_colors={'S0':'#1f77b4','S1':'#ff7f0e','S2':'#2ca02c','S3':'#9467bd'}
    for i,r in enumerate(segment_stats):
        q=np.asarray(r['length_q10_q25_q50_q75_q90'],float)
        ax.plot([i]*len(q),q,marker=marker_for(r['name']),
                color=next((v for k,v in seg_colors.items() if r['name'].startswith(k)),'#555555'),alpha=.8)
        ax.plot([i,i],[q[0],q[-1]],color=next((v for k,v in seg_colors.items() if r['name'].startswith(k)),'#555555'),alpha=.35)
    ax.set_xticks(range(len(segment_stats)),[x['name'] for x in segment_stats],rotation=60,ha='right')
    ax.set_ylabel('Segment length (10 Hz frames)');ax.grid(True,axis='y',alpha=.25);fig.tight_layout()
    fig.savefig(figdir/'segment_lengths.png',dpi=160);plt.close(fig)

    if long_records:
        fig,ax=plt.subplots(figsize=(11,5));x=np.arange(len(long_records));w=.36
        ax.bar(x-w/2,[r['metrics']['all']['mse_aligned']['mean'] for r in long_records],w,
               color='#1f77b4',label='selected type + oracle pose')
        ax.bar(x+w/2,[r['metrics']['all']['mse_random_aligned']['mean'] for r in long_records],w,
               color='#ff7f0e',label='random type + same oracle pose')
        ax.set_xticks(x,[r['selection'] for r in long_records],rotation=75,ha='right',fontsize=5)
        ax.set_ylabel('Long 10-step active7 MSE');ax.legend();ax.grid(True,axis='y',alpha=.25);fig.tight_layout()
        fig.savefig(figdir/'aligned_vs_random.png',dpi=160);plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--long-pool-upper', action='store_true')
    ap.add_argument('--config-shard', type=int, default=0)
    ap.add_argument('--config-shards', type=int, default=1)
    ap.add_argument('--finalize-only', action='store_true')
    args = ap.parse_args()
    if args.config_shards < 1 or not 0 <= args.config_shard < args.config_shards:
        raise ValueError('invalid config shard')
    out = args.out
    protocol_file = out / 'protocol.json'
    protocol = json.loads(protocol_file.read_text())
    if int(protocol['dev_rows']) != 105:
        raise AssertionError(f"expected 105 fixed heldout-Dev rows, got {protocol['dev_rows']}")
    episodes, rows, data_meta = shape.load_source(args.run_dir, args.checkpoint)
    q01, q99 = np.asarray(data_meta['q01'], np.float32), np.asarray(data_meta['q99'], np.float32)
    pooled = np.load(args.run_dir / 'pooled.npy', mmap_mode='r', allow_pickle=False)
    lookup = shape.make_row_lookup(rows)
    seglog=(out/f'grain-segments-progress-{args.config_shard:02}.jsonl'
            if args.config_shards>1 else out/'grain-segments-progress-final.jsonl')
    segment_sets, segment_stats, thresholds = shape.make_segment_sets(episodes, rows, seglog)
    tasks = np.asarray(rows['task'])
    devq = shape.select_dev_rows(rows)
    if len(devq) != 105 or np.any(~np.isin(tasks[devq], HELDOUT_TASKS)):
        raise AssertionError('Dev query count/task split mismatch')
    # Keep query segments partitioned by the same segmentation as the memory
    # pool. Combining segmentations makes overlapping intervals win the frame
    # lookup, e.g. giving a 20-step interval to a fixed-10 PCA model.
    qep_by_seg = {}
    dev_query_fallback_by_seg = {}
    for sname, segs in segment_sets.items():
        qep = {}
        for seg in segs:
            if seg['task'] in HELDOUT_TASKS:
                qep.setdefault((seg['task'], seg['episode']), []).append(seg)
        for k in qep:
            qep[k].sort(key=lambda x: x['start'])
        qep, fallback = shape.ensure_query_segment_coverage(
            qep, devq, rows, episodes, window=10)
        qep_by_seg[sname] = qep
        dev_query_fallback_by_seg[sname] = fallback

    dev_curve_by_cfg = {}
    scores = json.loads((out / 'dev_scores.json').read_text())
    for r in scores:
        dev_curve_by_cfg.setdefault(r['config'], []).append(r)
    configs = sorted(cfg for cfg in dev_curve_by_cfg
                     if (out / 'representations' / cfg / 'manifest.json').exists())
    if args.config_shards>1:
        width=(len(configs)+args.config_shards-1)//args.config_shards
        configs=configs[args.config_shard*width:(args.config_shard+1)*width]
    if not configs:
        raise RuntimeError('no completed representation bundles')

    shard_partial=out/f'grain_points_shard-{args.config_shard:02}.partial.json'
    if args.finalize_only:
        grain_records=json.loads((out/'grain_points.json').read_text())
    else:
        grain_records=json.loads(shard_partial.read_text()) if shard_partial.exists() else []
    done_configs={r['config'] for r in grain_records}
    t0 = time.monotonic()
    config_counter=0
    for cfg in configs:
        if cfg in done_configs:continue
        i=config_counter+1
        config_counter+=1
        bundle = out / 'representations' / cfg
        meta = json.loads((bundle / 'manifest.json').read_text())
        sname = str(meta['segmentation'])
        qep = qep_by_seg[sname]
        mem = [s for s in segment_sets[sname]
               if s['meta'].get('present', False) and s['meta'].get('memory', False)
               and s['task'] not in LONG_TASKS + HELDOUT_TASKS]
        if not mem or any(s['task'] in LONG_TASKS + HELDOUT_TASKS for s in mem):
            raise AssertionError(f'{cfg}: invalid memory split')
        codes = np.load(bundle / 'codes.npy', mmap_mode='r', allow_pickle=False)
        mtask = np.asarray([s['task'] for s in mem], np.int32)
        mepisode = np.asarray([s['episode'] for s in mem], np.int32)
        base_points = []
        for d in dev_curve_by_cfg[cfg]:
            base_points.append({'kappa': float(d['kappa']), 'units': int(d['units']),
                                'min_support': 10, 'min_episodes': 3})
        # Use one-support M6 only when the standard support rule cannot offer a
        # point near a requested 300/3000 grain. The count remains data driven.
        support_points = list(base_points)
        counts = [p['units'] for p in support_points]
        for target in GRAIN_TARGETS:
            if counts and min(abs(x-target) for x in counts) <= .25 * target:
                continue
            for kap in LOW_SUPPORT_KAPPAS:
                m6 = m6_fit_generic(codes, mtask, mepisode, kappa=kap,
                                    seed=20261104, min_support=1, min_episodes=1)
                support_points.append({'kappa': float(kap), 'units': int(m6['registered']),
                                       'min_support': 1, 'min_episodes': 1})
        # Candidate grain points are chosen by unit-count proximity only; Dev
        # action error ranks configurations, not the numerical unit count.
        for target in GRAIN_TARGETS:
            p = min(support_points, key=lambda x: (abs(x['units']-target), x['min_support'] != 1,
                                                   x['kappa']))
            m6 = m6_fit_generic(codes, mtask, mepisode, kappa=p['kappa'], seed=20261104,
                                min_support=p['min_support'], min_episodes=p['min_episodes'])
            m6['grain_kappa']=float(p['kappa'])
            m6['codes'] = codes
            if int(m6['registered']) != p['units']:
                raise AssertionError(f'{cfg}: nondeterministic M6 unit count')
            model, _, poses, loaded_mem, _ = shape.load_eval_bundle(out, cfg, segment_sets[sname], episodes)
            row, res = summary_point(cfg, target, model, codes, poses, loaded_mem, m6, qep,
                                     devq, rows, episodes, lookup, pooled, q01, q99,
                                     support=p['min_support'], episodes_min=p['min_episodes'])
            row['family']=str(meta['kind']);row['segmentation']=sname
            row['target_reached_within_25pct'] = bool(abs(row['units']-target) <= .25*target)
            grain_records.append(row)
        done_configs.add(cfg)
        if i % 10 == 0:
            save_json(shard_partial, grain_records)
            with (out / f'grain_grid_progress-{args.config_shard:02}.jsonl').open('a') as f:
                f.write(json.dumps({'event': 'grain_config_done', 'index': i,
                                    'total': len(configs), 'config': cfg,
                                    'elapsed_s': time.monotonic()-t0})+'\n')
    if args.config_shards>1 and not args.finalize_only:
        save_json(out/f'grain_points_shard-{args.config_shard:02}.json',grain_records)
        return
    if not args.finalize_only:
        save_json(out / 'grain_points.json', grain_records)

    # Pick no more than two configurations at each count tier using Dev ranks.
    grain_selected = []
    tier_selected = {}
    for target in GRAIN_TARGETS:
        tier = [r for r in grain_records if r['target_units'] == target]
        mse_rank = {r['config']: ix for ix, r in enumerate(sorted(tier,
                    key=lambda x: (x['dev_aligned_mse'], x['config'])))}
        cov_rank = {r['config']: ix for ix, r in enumerate(sorted(tier,
                    key=lambda x: (-x['dev_aligned_le_0p04'], x['config'])))}
        for r in tier:
            r['selection_rank_sum'] = mse_rank[r['config']] + cov_rank[r['config']]
        picked=[]; fams=set()
        for r in sorted(tier, key=lambda x: (x['selection_rank_sum'], x['dev_aligned_mse'], x['config'])):
            fam = json.loads((out/'representations'/r['config']/'manifest.json').read_text())['kind']
            if fam in fams: continue
            picked.append(r); fams.add(fam)
            if len(picked) == 2: break
        tier_selected[str(target)] = picked
        for r in picked:
            grain_selected.append({**r, 'selection_family': json.loads(
                (out/'representations'/r['config']/'manifest.json').read_text())['kind']})

    # Required Dev-only anchors (five entries, independent of the six grain picks).
    required = []
    all_cfg_best = {}
    for r in grain_records:
        old=all_cfg_best.get(r['config'])
        key=(r['dev_aligned_mse'], -r['dev_aligned_le_0p04'], abs(r['units']-r['target_units']))
        if old is None or key < old[0]: all_cfg_best[r['config']] = (key,r)
    candidates=[pair[1] for pair in all_cfg_best.values()]
    def pick_exact(config):
        xs=[r for r in candidates if r['config']==config]
        return min(xs,key=lambda r:(r['dev_aligned_mse'],-r['dev_aligned_le_0p04'])) if xs else None
    for cfg in ('S0_fixed10__pca_raw__d24','S0_fixed10__m1__d12'):
        r=pick_exact(cfg)
        if r: required.append({**r,'required_reason':'fixed control'})
    def best_kind(kind, mode=None):
        xs=[]
        for r in candidates:
            m=json.loads((out/'representations'/r['config']/'manifest.json').read_text())
            if m['kind'] != kind: continue
            if mode is not None and m.get('model',{}).get('input_mode') != mode: continue
            xs.append(r)
        return min(xs,key=lambda r:(r['dev_aligned_mse'],-r['dev_aligned_le_0p04'],r['units'])) if xs else None
    for kind,mode,reason in (('pca_canonical',None,'best canonical PCA'),
                             ('vq','raw','best VQ raw input'),
                             ('vq','procrustes','best VQ standard-form input')):
        r=best_kind(kind,mode)
        if r: required.append({**r,'required_reason':reason})
    if not any(r['config']=='S0_fixed10__pca_raw__d24' for r in required):
        raise AssertionError('required native raw PCA24 control missing')
    if not any(r['config']=='S0_fixed10__m1__d12' for r in required):
        raise AssertionError('required M1 b12 control missing')
    if not any(r.get('required_reason')=='best canonical PCA' for r in required):
        raise AssertionError('no canonical PCA family in full grid')
    selected = []
    seen=set()
    for r in required+grain_selected:
        key=(r['config'],r['target_units'],r['kappa'],r['min_support'])
        if key in seen: continue
        seen.add(key);selected.append(r)
    protocol_out={'selection_rule':'top 2 per each target grain by within-tier rank sum of Dev aligned MSE and <=.04 coverage, max 6 grain picks; required controls listed separately/additionally; no Long used in selection',
        'targets':list(GRAIN_TARGETS),'standard_support':[10,3],
        'relaxed_support':[1,1],'dev_rows':int(len(devq)),
        'dev_query_only_fallback_rows_by_segmentation':{
            k:int(len(v)) for k,v in dev_query_fallback_by_seg.items()},
        'grain_pick_count':len(grain_selected),'required_pick_count':len(required),
        'selected':[{'config':r['config'],'target_units':r['target_units'],'units':r['units'],
                     'kappa':r['kappa'],'min_support':r['min_support'],
                     'min_episodes':r['min_episodes'],'reason':r.get('required_reason','unit-grain Dev rank')}
                    for r in selected],
        'long_was_used_for_selection':False}
    save_json(out/'grain_selection.json',{'protocol':protocol_out,'grain_points':grain_records,
                                          'tier_selected':{k:[r['config'] for r in v] for k,v in tier_selected.items()},
                                          'required':[r['config'] for r in required]})
    # Preserve the interim runner's six-entry selection, then replace the
    # public selection artifact with the preregistered grain/required set.
    old=out/'selected_configs.json'
    if old.exists():
        backup=out/'selected_configs_interim.json'
        if backup.exists(): raise FileExistsError(backup)
        old.replace(backup)
    save_json(out/'selected_configs.json',protocol_out)

    # Fixed Long mask is used only after all Dev selections are frozen.
    hardz=np.load(shape.ROOT/'topk-long-01'/'production'/'predictions_and_errors.npz',allow_pickle=False)
    longq=np.asarray(hardz['query_rows'],np.int64); hard=np.asarray(hardz['hard_mask'],bool)
    if len(longq)!=548 or hard.sum()!=63: raise AssertionError('fixed Long query mask mismatch')
    long_records=[]
    long_selected=[]
    (out/'long_predictions').mkdir(parents=True,exist_ok=True)
    for r in selected:
        cfg=r['config']; bundle=out/'representations'/cfg
        if not (bundle/'manifest.json').exists(): continue
        meta=json.loads((bundle/'manifest.json').read_text()); sname=meta['segmentation']
        model,codes,poses,mem,m6old=shape.load_eval_bundle(out,cfg,segment_sets[sname],episodes)
        memcodes=np.asarray(codes,np.float32)
        mt=np.asarray([s['task'] for s in mem],np.int32); me=np.asarray([s['episode'] for s in mem],np.int32)
        m6=m6_fit_generic(memcodes,mt,me,kappa=r['kappa'],seed=20261104,
                          min_support=r['min_support'],min_episodes=r['min_episodes'])
        m6['codes']=memcodes
        qsegs={}
        for seg in segment_sets[sname]:
            if seg['task'] in LONG_TASKS:
                qsegs.setdefault((seg['task'],seg['episode']),[]).append(seg)
        for k in qsegs: qsegs[k].sort(key=lambda x:x['start'])
        qsegs,fallback=shape.ensure_query_segment_coverage(qsegs,longq,rows,episodes,window=10)
        res=shape.eval_config(model,memcodes,poses,mem,m6,qsegs,longq,rows,episodes,lookup,pooled,q01,q99,seed=4101)
        if res is None or len(res['row_ids'])!=548: raise AssertionError(f'{cfg}: Long query coverage mismatch')
        order={int(x):i for i,x in enumerate(res['row_ids'])}
        if set(order)!=set(longq.tolist()): raise AssertionError(f'{cfg}: Long row IDs differ')
        ix=np.asarray([order[int(x)] for x in longq],np.int64)
        res={k:(v[ix] if isinstance(v,np.ndarray) and len(v)==548 else v) for k,v in res.items()}
        summary=shape.metrics_summary(res,hard)
        tag=f"{cfg}__g{r['target_units']}_k{r['kappa']}_s{r['min_support']}"
        np.savez_compressed(out/'long_predictions'/f'{tag}.npz',**res,hard_mask=hard)
        long_records.append({'selection':tag,'config':cfg,'target_units':r['target_units'],
            'units':m6['registered'],'kappa':r['kappa'],'min_support':r['min_support'],
            'min_episodes':r['min_episodes'],'query_only_fallback_rows':len(fallback),
            'metrics':summary,'row_metrics':{k:v.tolist() for k,v in res.items()}})
        long_selected.append({'record':long_records[-1],'model':model,'codes':memcodes,'poses':poses,'mem':mem})
        with (out/'progress.jsonl').open('a') as f:
            f.write(json.dumps({'event':'long_selected_done','selection':tag,
                 'all_mse':summary['all']['mse_aligned']['mean'],
                 'hard_mse':summary['hard']['mse_aligned']['mean']})+'\n')
    save_json(out/'long_metrics.json',long_records)

    # One explicitly labelled Long-containing upper bound; it cannot affect selection.
    if args.long_pool_upper:
        target=pick_exact('S0_fixed10__pca_raw__d24')
        if target is None: raise AssertionError('missing PCA24 for Long-pool upper bound')
        cfg=target['config']; sname=json.loads((out/'representations'/cfg/'manifest.json').read_text())['segmentation']
        model,codes,poses,mem,_=shape.load_eval_bundle(out,cfg,segment_sets[sname],episodes)
        longmem=[s for s in segment_sets[sname] if s['meta'].get('present',False)
                 and s['meta'].get('memory',False) and s['task'] in LONG_TASKS]
        if not longmem: raise AssertionError('no Long memory episodes for upper-bound pool')
        extra_codes=[];extra_poses=[]
        for seg in longmem:
            c,p=shape.encode_segment(model,seg,lookup,rows,pooled)
            extra_codes.append(np.asarray(c,np.float32));extra_poses.append(p)
        allmem=mem+longmem;allcodes=np.concatenate([np.asarray(codes,np.float32),np.asarray(extra_codes,np.float32)])
        allposes=poses+extra_poses
        m6=m6_fit_generic(allcodes,np.asarray([s['task'] for s in allmem]),
                          np.asarray([s['episode'] for s in allmem]),kappa=target['kappa'],
                          seed=20261104,min_support=target['min_support'],min_episodes=target['min_episodes'])
        m6['codes']=allcodes
        long_ep={}
        for seg in segment_sets[sname]:
            if seg['task'] in LONG_TASKS: long_ep.setdefault((seg['task'],seg['episode']),[]).append(seg)
        for k in long_ep: long_ep[k].sort(key=lambda x:x['start'])
        qsegs,fallback=shape.ensure_query_segment_coverage(long_ep,longq,rows,episodes,window=10)
        res=shape.eval_config(model,allcodes,allposes,allmem,m6,qsegs,longq,rows,episodes,lookup,pooled,q01,q99,seed=4101)
        if res is None or len(res['row_ids'])!=548: raise AssertionError('Long-pool upper query coverage mismatch')
        order={int(x):i for i,x in enumerate(res['row_ids'])}; ix=np.asarray([order[int(x)] for x in longq])
        res={k:(v[ix] if isinstance(v,np.ndarray) and len(v)==548 else v) for k,v in res.items()}
        upper={'config':cfg,'pool':'nonLong memory plus Long memory; upper bound only',
               'long_memory_segments_added':len(longmem),'units':int(m6['registered']),
               'kappa':target['kappa'],'min_support':target['min_support'],
               'metrics':shape.metrics_summary(res,hard),'row_metrics':{k:v.tolist() for k,v in res.items()}}
        save_json(out/'long_pool_upper.json',upper)
        np.savez_compressed(out/'long_pool_upper.npz',**res,hard_mask=hard)

    make_figures(out, scores, grain_records, segment_stats, long_records)
    save_json(out/'final_summary.json',{'complete':True,'dev_rows':len(devq),
       'long_rows':len(longq),'hard_rows':int(hard.sum()),'representation_configs':len(configs),
       'grain_points':len(grain_records),'selected':protocol_out['selected'],
       'long_records':len(long_records),'selection_used_long':False,
       'segment_statistics':segment_stats,'thresholds':thresholds,
       'elapsed_s':time.monotonic()-t0})
    (out/'DONE.json').write_text(json.dumps({'complete':True,'grain_grid':True})+'\n')


if __name__ == '__main__': main()
