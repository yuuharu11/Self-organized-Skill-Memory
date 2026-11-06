#!/usr/bin/env python3
"""Final Long pass at the Dev-selected M6 grain for every representation."""
from __future__ import annotations
import json
import argparse
from pathlib import Path
import time
import numpy as np

from run import (DEFAULT_RUN, CHECKPOINT, ROOT, append_progress, load_source,
                 make_row_lookup, make_segment_sets, metrics_summary,
                 load_eval_bundle, eval_config, ensure_query_segment_coverage, LONG_TASKS)


def write_json(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(path)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=ROOT/'shape-skills-01-corrected')
    out=ap.parse_args().out
    out.mkdir(parents=True,exist_ok=True)
    extra=out/'long-grid';extra.mkdir(exist_ok=True)
    prog=extra/'progress.jsonl'
    episodes,rows,data_meta=load_source(DEFAULT_RUN,CHECKPOINT)
    pooled=np.load(DEFAULT_RUN/'pooled.npy',mmap_mode='r',allow_pickle=False)
    lookup=make_row_lookup(rows)
    segsets,segstats,thresholds=make_segment_sets(episodes,rows,prog)
    hard_path=ROOT/'topk-long-01'/'production'/'long_hard_rows.npz'
    if not hard_path.exists():hard_path=ROOT/'fair-expert-01'/'evaluation-final'/'offline_predictions.npz'
    hardz=np.load(hard_path,allow_pickle=False)
    long_rows=np.asarray(hardz['row_ids'],np.int64)
    long_rows=long_rows[np.asarray(rows['task'])[long_rows]<10]
    assert len(long_rows)==548
    if 'hard_mask' in hardz:
        mask=np.asarray(hardz['hard_mask'],bool)[np.asarray(hardz['task_ids'])<10]
    else:
        h=np.load(ROOT/'fair-expert-01'/'evaluation-final'/'offline_predictions.npz',allow_pickle=False)
        mask=np.asarray(h['hard_mask'],bool)
    assert mask.sum()==63
    existing=json.load(open(out/'long_metrics.json'))
    records={r['config']:r for r in existing}
    partial=extra/'metrics.json'
    if partial.exists():
        for r in json.load(open(partial)):
            records[r['config']]=r
    scores=json.load(open(out/'dev_scores.json'))
    best={}
    for r in scores:
        c=r['config']
        if c not in best or (r['dev_aligned_mse'],-r['dev_aligned_le_0p04'],r['units'],r['kappa']) < (best[c]['dev_aligned_mse'],-best[c]['dev_aligned_le_0p04'],best[c]['units'],best[c]['kappa']):
            best[c]=r
    append_progress(prog,event='start',config_count=len(best),already_evaluated=len(records),long_rows=len(long_rows),hard_rows=int(mask.sum()))
    for ix,(cfg,dev) in enumerate(sorted(best.items()),1):
        if cfg in records:continue
        segname=dev['segmentation']
        repdir=out/'representations'/cfg
        model,codes,poses,mem,m6=load_eval_bundle(out,cfg,segsets[segname],episodes)
        qsegs={}
        for s in segsets[segname]:
            if s['task'] in LONG_TASKS:qsegs.setdefault((s['task'],s['episode']),[]).append(s)
        for key in qsegs:qsegs[key].sort(key=lambda s:s['start'])
        qsegs,fallback_rows=ensure_query_segment_coverage(qsegs,long_rows,rows,episodes,window=10)
        res=eval_config(model,codes,poses,mem,m6,qsegs,long_rows,rows,episodes,lookup,pooled,
                        np.asarray(data_meta['q01'],np.float32),np.asarray(data_meta['q99'],np.float32),seed=4101)
        # Fixed Eval IDs may not be in the same sequence as episode traversal.
        # Require exact set coverage, then restore the canonical input order so
        # the frozen hard mask remains aligned.
        got=np.asarray(res['row_ids'],np.int64)
        assert len(got)==548 and len(np.unique(got))==548 and set(got.tolist())==set(long_rows.tolist()), (
            f"query coverage mismatch: got={len(got)} unique={len(np.unique(got))} "
            f"missing={sorted(set(long_rows.tolist())-set(got.tolist()))[:12]} "
            f"extra={sorted(set(got.tolist())-set(long_rows.tolist()))[:12]}")
        pos={int(r):i for i,r in enumerate(got)}
        order=np.asarray([pos[int(r)] for r in long_rows],np.int64)
        for key,value in list(res.items()):
            if isinstance(value,np.ndarray) and value.ndim and value.shape[0]==len(got):
                res[key]=value[order]
        hard=np.asarray([int(r) in set(long_rows[mask].tolist()) for r in res['row_ids']],bool)
        summary=metrics_summary(res,hard)
        unit_tasks=np.asarray([len({int(mem[i]['task']) for i in m}) for m in m6['members_local']],np.int32)
        manifest=json.load(open(repdir/'manifest.json'))
        records[cfg]={
            'config':cfg,'segmentation':segname,'family':model['kind'],'code_dim':int(codes.shape[1]),
            'kappa':float(manifest['m6_kappa']),'units':int(m6['registered']),
            'shared_units_fraction':float(np.mean(unit_tasks>=2)) if len(unit_tasks) else 0.,
            'unit_task_count_q10_q50_q90':np.quantile(unit_tasks,[.1,.5,.9]).round(2).tolist() if len(unit_tasks) else [],
            'units_shared_across_2plus_tasks':int(np.sum(unit_tasks>=2)),
            'code_values_per_step_mean':manifest['code_values_per_step_mean'],
            'pose_numeric_values_per_segment_mean':manifest['pose_numeric_values_per_segment_mean'],
            'query_only_fallback_rows':int(len(fallback_rows)),
            'metrics':summary,
        }
        append_progress(prog,event='long_eval_done',index=ix,total=len(best),config=cfg,
                        units=int(m6['registered']),aligned=summary['all']['mse_aligned']['mean'],
                        query_only_fallback_rows=int(len(fallback_rows)))
        if ix%10==0:
            write_json(extra/'metrics.json',list(records.values()))
    write_json(extra/'metrics.json',list(records.values()))
    write_json(extra/'summary.json',{
        'long_eval_rows':548,'hard_rows':63,
        'selection_grain':'per representation, selected using heldout Dev only by min aligned MSE, then max coverage <=.04, then fewer units, then lower kappa',
        'all_configs_at_dev_selected_grain':len(records),
        'long_used_for_config_or_grain_selection':False,
        'metric':'executed 10-step normalized active7 action-window MSE',
    })
    append_progress(prog,event='complete',configs=len(records))
    write_json(extra/'DONE.json',{'complete':True,'configs':len(records)})

if __name__=='__main__':main()
