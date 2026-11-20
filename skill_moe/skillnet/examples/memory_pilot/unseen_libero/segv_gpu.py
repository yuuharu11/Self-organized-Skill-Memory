#!/usr/bin/env python3
"""SEG-V downstream preflight and CPU smoke.

This stage is deliberately non-training: checker approval is required before
the later GPU run.  It verifies the immutable speed bank and exact-time VLM
features, then exercises the KQ2 key/query model interfaces on tiny tensors.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', Path(__file__).resolve().parents[5] / '.skillnet-state')).expanduser()
ROOT = STATE_DIR / 'outputs/memory-pilot/unseen-libero'
OUT = ROOT / 'segv-02'
KQ2_DONE = ROOT / 'kq2-cl-01' / 'DONE'
RUN_TAG = 'segv_run02'


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def log(event: str, **fields):
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {'utc': utc(), 'event': event, **fields}
    with (OUT / 'progress.log').open('a') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(json.dumps(rec, ensure_ascii=False), flush=True)


def verify_inputs():
    bank = OUT / 'bank'
    required = ('speed__natural_actions.npy', 'speed__length.npy', 'speed__task.npy',
                'speed__episode.npy', 'speed__value_actions10.npy', 'speed__bank_actions.npy',
                'speed__unit_key.npy', 'speed__vlm34.npy', 'speed__vlm34_key.npy',
                'speed__vlm34_source.npy')
    missing = [n for n in required if not (bank / n).is_file()]
    if missing:
        raise FileNotFoundError(f'missing SEG-V bank inputs: {missing}')
    keys = np.load(bank / 'speed__unit_key.npy', mmap_mode='r', allow_pickle=False)
    fkeys = np.load(bank / 'speed__vlm34_key.npy', mmap_mode='r', allow_pickle=False)
    feats = np.load(bank / 'speed__vlm34.npy', mmap_mode='r', allow_pickle=False)
    sources = np.load(bank / 'speed__vlm34_source.npy', mmap_mode='r', allow_pickle=False)
    if not np.array_equal(keys, fkeys):
        raise AssertionError('VLM feature keys are not exactly aligned with speed unit starts')
    if feats.shape != (len(keys), 34, 2048):
        raise AssertionError(f'unexpected exact-time feature shape: {feats.shape}')
    source_counts = {str(x): int((sources == x).sum()) for x in np.unique(sources)}
    meta = json.loads((OUT / 'vlm_rebuild.json').read_text())
    if meta['fresh_exact_rows'] != 256 or source_counts.get('fresh-vq-b') != 256:
        raise AssertionError('the 256 missing exact-time starts were not freshly extracted')
    if meta['parity']['direct_token_cos_min'] < .999:
        raise AssertionError('direct VQ-B overlap parity below .999 cosine')
    if not KQ2_DONE.exists():
        raise RuntimeError(f'KQ2-CL must be DONE before GPU stage: {KQ2_DONE}')
    return {'units': len(keys), 'vlm_shape': list(feats.shape), 'source_counts': source_counts,
            'vlm_parity': meta['parity'], 'kq2cl_done': True}


def smoke():
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
    sys.path.insert(0, str(HERE))
    import key_motion as km
    import kq2_run as kr

    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Exercise the exact key architecture and both KQ2 query forms with small
    # batches, without touching evaluation data or training checkpoints.
    key = km.ArchMotionEncoder('x1', 'absolute').to(dev)
    acts = torch.randn(2, 16, 6, device=dev)
    valid = torch.ones(2, 16, dtype=torch.bool, device=dev)
    kout = key(acts, valid)
    if kout.shape != (2, 1, 256) or not torch.isfinite(kout).all():
        raise AssertionError(f'K-x1 smoke failed: {tuple(kout.shape)}')
    query_checks = {}
    for name, xattn, ffn, use_vlm in (('iii', 1, True, True), ('i', 0, False, False)):
        model = kr.QueryNet('gru', 1, xattn, ffn).to(dev)
        gs = type('TinySplit', (), {})()
        gs.seq_a = torch.randn(40, 7, device=dev)
        gs.seq_s = torch.randn(40, 8, device=dev)
        gs.off = torch.tensor([0, 20], device=dev)
        gs.slot = torch.tensor([0, 1], device=dev)
        gs.start = torch.tensor([9, 12], device=dev)
        gs.n = 2
        gs.vlm = torch.randn(2, 34, 2048, device=dev, dtype=torch.float16) if use_vlm else None
        gs.errors = torch.zeros(2, 3, device=dev)
        gs.allowed = torch.ones(2, 3, device=dev, dtype=torch.bool)
        gs.task = torch.tensor([0, 1], device=dev)
        gs.episode = torch.tensor([0, 1], device=dev)
        def window(ix, L=10):
            t = gs.start[ix][:, None]
            rel = torch.arange(L, device=dev)[None] - (L - 1)
            tau = t + rel
            valid = tau >= 0
            g = gs.off[gs.slot[ix]][:, None]
            gsidx = (g + rel).clamp(min=0)
            s = gs.seq_s[gsidx] * valid[..., None]
            av = tau >= 1
            aa = gs.seq_a[(g + rel - 1).clamp(min=0)] * av[..., None]
            return torch.cat((aa, s, av[..., None].float()), -1), valid
        gs.window = window
        q = model(gs, torch.arange(2, device=dev))
        if q.shape != (2, 256) or not torch.isfinite(q).all():
            raise AssertionError(f'KQ2 query {name} smoke failed: {tuple(q.shape)}')
        query_checks[name] = {'query_shape': list(q.shape), 'vlm_input': use_vlm}
    return {'device': str(dev), 'key_shape': list(kout.shape), 'query_checks': query_checks,
            'training_started': False, 'closed_loop_started': False}


def train_key_and_queries(device='cuda:0'):
    """Run the bank-specific KQ2 key/query stages; downstream CL stages remain separate."""
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
    if not torch.cuda.is_available() and str(device).startswith('cuda'):
        raise RuntimeError('CUDA requested but unavailable')
    sys.path.insert(0, str(HERE))
    import kq2_data as kd
    import kq2_run as kr

    bank = OUT / 'bank'
    compat = OUT / 'compat' / 'key-motion-final-data'
    compat.mkdir(parents=True, exist_ok=True)
    aliases = {'bank__task.npy':'speed__task.npy', 'bank__episode.npy':'speed__episode.npy',
               'bank__natural_actions.npy':'speed__natural_actions.npy', 'bank__length.npy':'speed__length.npy',
               'bank__source.npy':'speed__source.npy', 'bank__value_actions10.npy':'speed__value_actions10.npy',
               'bank_actions.npy':'speed__bank_actions.npy'}
    for dst, src in aliases.items():
        p = compat / dst
        if not p.exists():
            p.symlink_to(bank / src)
    # KQ2 data split arrays are the source-of-truth offline observation rows;
    # KQ2-CL's obs/npz files are only closed-loop snapshots, not train splits.
    source_splits = ROOT / 'kq2-01' / 'data'
    for split in ('eh', 'dev2', 'learned', 'unseen20'):
        src = source_splits / split
        dst = OUT / 'data' / split
        if not src.is_dir():
            raise FileNotFoundError(f'KQ2-CL observation split missing: {src}')
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.symlink_to(src, target_is_directory=True)

    kd.KF_DATA = compat
    kd.OUT = OUT
    kr.KF_DATA = compat
    kr.ROOT = ROOT
    kr.OUT = OUT
    key_dir = OUT / 'stage1' / 'keys'
    key_dir.mkdir(parents=True, exist_ok=True)
    # The key checkpoint grid is selected on Dev2 only. Eval splits are not
    # materialized in the key-training context.
    kr.EVAL = ('dev2',)
    key_ctx = kr.Ctx(device, splits=('dev2',), out=OUT)
    key_row = kr.train_key(f'{RUN_TAG}_x1_dist_tau001', {
        'arch':'x1', 'tau':.01, 'seed':1, 'epochs':200,
        'ckpt_epochs':[4,10,25,50,100,150,200], 'lr':1e-4, 'batch':128,
    }, key_ctx, key_dir)
    keys_path = Path(key_row['keys'])

    # Refit both selected KQ2 query forms with three deterministic seeds. Dev2
    # selects each epoch; learned and unseen20 are evaluated once per model.
    kr.EVAL = ('dev2', 'learned', 'unseen20')
    qctx = kr.Ctx(device, splits=('eh',) + kr.EVAL, vlm=True, out=OUT)
    gs = {n: kr.GPUSplit(qctx.splits[n], qctx.device, n in ('eh','dev2','learned','unseen20'))
          for n in ('eh',) + kr.EVAL}
    query_dir = OUT / 'stage2' / 'rows'
    query_dir.mkdir(parents=True, exist_ok=True)
    trained = []
    specs = (('iii', 1, True, True), ('i', 0, False, False))
    for qname, xattn, ffn, use_vlm in specs:
        for seed in (1, 2, 3):
            name = f'{RUN_TAG}_{qname}_s{seed}'
            cfg = {'keys':str(keys_path), 'enc':'gru', 'gru_layers':1, 'xattn':xattn,
                   'ffn':ffn, 'history':'w10', 'objective':'soft', 'tau':.02,
                   'margin_w':.1, 'seed':seed, 'epochs':40, 'batch':256,
                   'lr':3e-4, 'wd':1e-4}
            gq = {n:kr.GPUSplit(qctx.splits[n], qctx.device, use_vlm) for n in ('eh',)+kr.EVAL}
            row = kr.train_query(name, cfg, qctx, gq, query_dir)
            trained.append({'name':name,'selected_epoch':row['selected_epoch'],
                            'dev2':row['dev_selected'], 'checkpoint':str(query_dir/f'{name}.pt')})
            torch.cuda.empty_cache()
    result = {'utc':utc(), 'status':'key_query_complete', 'key':key_row,
              'query_models':trained, 'key_file':str(keys_path),
              'training_data':'KQ2-CL eh LIBERO-40 memory rows',
              'selection':'Dev v2 only; learned and unseen20 evaluated once per query model'}
    path = OUT / 'model_stage.json'
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=float)+'\n')
    return result


def build_retrieval(device='cuda:0'):
    """Task-fold retrieval diagnostics for the SEG-V query checkpoints."""
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
    sys.path.insert(0, str(HERE))
    import kq2_data as kd
    import kq2_run as kr
    import kq2cl_common as C
    import kq2cl_retrieve as rr
    bank = OUT / 'bank'
    compat = OUT / 'compat' / 'key-motion-final-data'
    if not (compat / 'bank_actions.npy').exists():
        raise FileNotFoundError('run --train-key-query first to make compatibility bank links')
    kd.KF_DATA = kr.KF_DATA = compat
    kd.OUT = kr.OUT = OUT
    kr.ROOT = ROOT
    stage_meta = OUT / 'model_stage.json'
    if not stage_meta.exists():
        raise FileNotFoundError('run --train-key-query first')
    trained_meta = json.loads(stage_meta.read_text())
    key_path = Path(trained_meta['key_file'])
    key_meta = OUT / 'stage1' / 'keys' / f'{RUN_TAG}_x1_dist_tau001.json'
    if not key_meta.exists():
        raise FileNotFoundError('run --train-key-query first')
    meta = json.loads(key_meta.read_text())
    key_path = Path(meta['keys'])
    C.OUT = OUT
    C.KEYS = key_path
    C.EVAL = ('dev2',)
    C.QUERY_CK = {'iii':OUT/'stage2/rows'/f'{RUN_TAG}_iii_s1.pt',
                  'i':OUT/'stage2/rows'/f'{RUN_TAG}_i_s1.pt'}
    # The task-fold models train and select only on Dev2. Eval splits are loaded
    # only after all folds are frozen, for the rank-law cache.
    kr.EVAL = ('dev2',)
    original_train_query = kr.train_query
    def train_dev_only(name, cfg, ctx, gsplits, save):
        row = original_train_query(name, cfg, ctx, gsplits, save)
        # kq2cl_retrieve.folds emits a progress line that historically included
        # the held-out Eval score. The fold trainer must not inspect Eval, so
        # supply a null solely to satisfy that logger's schema.
        row['evaluation']['unseen20'] = {'all': None, 'status':'not-evaluated'}
        return row
    kr.train_query = train_dev_only
    rr.folds(device, ['iii','i'])
    kr.train_query = original_train_query
    C.EVAL = ('dev2','learned','unseen20')
    kr.EVAL = C.EVAL
    rr.retrieve(device, ['iii','i'])
    metrics = rr.measure(['iii','i'])
    return {'utc':utc(), 'status':'retrieval_complete', 'metrics':metrics,
            'folds':str(OUT/'folds'), 'retrieval':str(OUT/'retrieval')}


def configure_cl_paths():
    """Point read-only KQ2-CL components at the new SEG-V outputs and bank."""
    sys.path.insert(0, str(HERE))
    import kq2cl_common as C
    meta_path = OUT / 'model_stage.json'
    if not meta_path.exists():
        raise FileNotFoundError('run --train-key-query first')
    meta = json.loads(meta_path.read_text())
    C.OUT = OUT
    C.KQ2 = ROOT / 'kq2-01'
    C.ROOT = OUT / 'compat' / 'state'
    C.KEYS = Path(meta['key_file'])
    C.QUERY_CK = {'iii':OUT/'stage2/rows'/f'{RUN_TAG}_iii_s1.pt',
                  'i':OUT/'stage2/rows'/f'{RUN_TAG}_i_s1.pt'}
    data = C.ROOT / 'key-motion-01' / 'final' / 'data'
    data.mkdir(parents=True, exist_ok=True)
    for dst, src in (('bank_actions.npy','speed__bank_actions.npy'),('bank_task.npy','speed__task.npy'),
                     ('bank__task.npy','speed__task.npy'),('bank__episode.npy','speed__episode.npy'),
                     ('bank__natural_actions.npy','speed__natural_actions.npy'),('bank__length.npy','speed__length.npy'),
                     ('bank__source.npy','speed__source.npy'),('bank__value_actions10.npy','speed__value_actions10.npy')):
        p = data / dst
        if not p.exists(): p.symlink_to(OUT/'bank'/src)
    pi = OUT / 'pi05'
    if not pi.exists(): pi.symlink_to(ROOT/'kq2-cl-01'/'pi05', target_is_directory=True)
    obs = OUT / 'obs'
    if not obs.exists(): obs.symlink_to(ROOT/'kq2-cl-01'/'obs', target_is_directory=True)
    for name in ('closed-loop-kq-01','expert-kq-01','expert-ex3-01'):
        link=C.ROOT/name
        if not link.exists(): link.symlink_to(ROOT/name,target_is_directory=True)
    return C


def rebuild_expert_rank_rows(device='cuda:0'):
    """Rebuild K1's draft-rank order for the 3,981 speed units, never reuse 4,516 ids."""
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','8')))
    C=configure_cl_paths()
    src=ROOT/'expert-ex3-01'/'prepare'/'rows.npz'
    if not src.exists(): raise FileNotFoundError(src)
    prep={k:np.asarray(v) for k,v in np.load(src).items()}
    units=np.load(C.ROOT/'key-motion-01/final/data/bank_actions.npy',mmap_mode='r',allow_pickle=False)
    btask=np.load(C.ROOT/'key-motion-01/final/data/bank_task.npy',allow_pickle=False).astype(np.int64)
    target=torch.as_tensor(prep['target'][...,:6],device=device,dtype=torch.float32)
    windows=torch.as_tensor(np.asarray(units[...,:6]),device=device,dtype=torch.float32)
    tasks=np.asarray(prep['task'],np.int64); order_path=OUT/'compat'/'expert-speed'/'prepare'/'order.npy'
    dest=order_path.parent; dest.mkdir(parents=True,exist_ok=True)
    rows_path=dest/'rows.npz'
    if order_path.exists() or rows_path.exists():
        if not (order_path.exists() and rows_path.exists()):
            raise FileExistsError('partial SEG-V Expert rank cache; refusing to overwrite')
        old=np.load(rows_path,allow_pickle=False)
        idx=np.load(order_path,mmap_mode='r',allow_pickle=False)
        if idx.shape!=(len(tasks),len(btask)) or not np.array_equal(old['n_eligible'],
                np.asarray([np.count_nonzero(btask!=t) for t in tasks],np.int32)):
            raise AssertionError('cached SEG-V Expert ranks do not match bank/task exclusion')
        return {'rows':len(tasks),'bank_units':len(btask),'order_shape':list(idx.shape),
                'eligible_min':int(old['n_eligible'].min()),'eligible_max':int(old['n_eligible'].max()),
                'same_task_excluded':True,'reused':True,'paths':{'rows':str(rows_path),'order':str(order_path)}}
    order=np.lib.format.open_memmap(order_path,mode='w+',dtype=np.int16,shape=(len(tasks),len(btask)))
    n_eligible=np.zeros(len(tasks),np.int32)
    for start in range(0,len(tasks),64):
        end=min(start+64,len(tasks)); tb=torch.as_tensor(tasks[start:end],device=device)
        allow=torch.as_tensor(btask[None,:]!=tasks[start:end,None],device=device)
        n_eligible[start:end]=allow.sum(1).cpu().numpy().astype(np.int32)
        err=((target[start:end,None]-windows[None])**2).sum((-1,-2))/70.
        err=err.masked_fill(~allow,float('inf'))
        idx=torch.argsort(err,dim=1,stable=True).to(torch.int16).cpu().numpy()
        order[start:end]=idx
    order.flush()
    prep['n_eligible']=n_eligible
    p=rows_path
    np.savez_compressed(p,**prep)
    return {'rows':len(tasks),'bank_units':len(btask),'order_shape':[len(tasks),len(btask)],
            'eligible_min':int(n_eligible.min()),'eligible_max':int(n_eligible.max()),
            'same_task_excluded':True,'paths':{'rows':str(p),'order':str(order_path)}}


def train_expert_and_heads(device='cuda:0', expert_updates=5000):
    """KQ2-CL K1 Expert and confidence/gripper-head recipe on SEG-V retrievals."""
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','8')))
    C = configure_cl_paths()
    import kq2cl_expert as ex
    import kq2cl_heads as hd
    rank_rows=rebuild_expert_rank_rows(device)
    ex.EX3_OUT = OUT/'compat'/'expert-speed'
    # Prepare sampling rows with the frozen KQ2-CL EX3 training-row schedule,
    # but draw drafts from ranks measured for this speed-only bank.
    prep_path = ex.EX3_OUT / 'prepare' / 'rows.npz'
    order_path = ex.EX3_OUT / 'prepare' / 'order.npy'
    if not prep_path.exists() or not order_path.exists():
        raise FileNotFoundError('EX3 shared LIBERO-40 training-row cache missing')
    prep = ex.prepare(.5,.5,.15,.05)
    ex.train('K_SEG', updates=expert_updates, batch=8, microbatch=8, save_every=max(1,expert_updates//10))
    selected = ex.select()
    ex.curves(['dev2','learned','unseen20'], ['K','EX3'])
    hd.train(device)
    hd.policy()
    offline = hd.offline()
    result = {'utc':utc(),'status':'expert_heads_offline_complete','prepare':prep,'rank_rows':rank_rows,'expert_selected':selected.get('selected'),
              'offline_splits':{k:v['rows'] for k,v in offline['splits'].items()}}
    path=OUT/'expert_heads_stage.json'
    if path.exists(): raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False,default=float)+'\n')
    return result


def smoke_train_expert_heads(device='cuda:0'):
    """Tiny K1 update + real head fits, isolated from the future full-run outputs."""
    torch_threads=int(os.environ.get('OMP_NUM_THREADS','8'))
    import torch
    torch.set_num_threads(torch_threads)
    C=configure_cl_paths()
    rank_rows=rebuild_expert_rank_rows(device)
    smoke_out=OUT/'adapter-smoke-04'
    if smoke_out.exists(): raise FileExistsError(f'smoke outputs exist; preserve and choose a new run id: {smoke_out}')
    smoke_out.mkdir(parents=True)
    C.OUT=smoke_out
    for d in ('data','retrieval','pi05','obs'):
        src=(ROOT/'kq2-cl-01'/'obs') if d=='obs' else OUT/d
        dst=smoke_out/d
        if not src.exists(): raise FileNotFoundError(src)
        dst.symlink_to(src,target_is_directory=True)
    import kq2cl_expert as ex
    import kq2cl_heads as hd
    ex.EX3_OUT=OUT/'compat'/'expert-speed'
    ex.prepare(.5,.5,.15,.05)
    ex.train('K_SEG_SMOKE',updates=2,batch=2,microbatch=2,save_every=1,seed=20261120)
    heads=hd.train(device)
    result={'utc':utc(),'status':'adapter_smoke_models_complete','expert_updates':2,'rank_rows':rank_rows,
            'expert_checkpoint':str(smoke_out/'expert/train-K_SEG_SMOKE/checkpoints/000002'),
            'head_selected':{q:heads[q]['selected'] for q in ('iii','i')},
            't0_policy':'fixed .15 for closed-loop smoke; confidence and gripper heads fit with KQ2-CL recipe',
            'isolated_output':str(smoke_out)}
    (smoke_out/'smoke_stage.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,default=float)+'\n')
    return result


def finish_expert_heads_smoke(device='cuda:0'):
    """Resume selection and heads from the saved tiny K1 smoke checkpoint."""
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','8')))
    C=configure_cl_paths(); smoke_out=OUT/'adapter-smoke-04'
    if not (smoke_out/'expert/train-K_SEG_SMOKE/checkpoints').is_dir():
        raise FileNotFoundError('tiny K1 smoke checkpoint is missing')
    C.OUT=smoke_out
    import kq2cl_heads as hd
    heads=hd.train(device)
    result={'utc':utc(),'status':'adapter_smoke_models_complete','expert_updates':2,
            'expert_checkpoint':str(smoke_out/'expert/train-K_SEG_SMOKE/checkpoints/000002'),
            'head_selected':{q:heads[q]['selected'] for q in ('iii','i')},
            't0_policy':'fixed .15 for smoke; production Dev policy uses head outputs and Dev selection',
            'isolated_output':str(smoke_out)}
    path=smoke_out/'smoke_stage.json'
    if path.exists(): raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False,default=float)+'\n')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--train-key-query', action='store_true')
    p.add_argument('--retrieve', action='store_true')
    p.add_argument('--train-expert-heads', action='store_true')
    p.add_argument('--expert-updates', type=int, default=5000)
    p.add_argument('--smoke-expert-heads', action='store_true')
    p.add_argument('--finish-expert-heads-smoke', action='store_true')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--smoke-command', action='store_true', help='print the reproducible CPU smoke command')
    a = p.parse_args()
    if not (a.preflight or a.smoke or a.smoke_command or a.train_key_query or a.retrieve or a.train_expert_heads or a.smoke_expert_heads or a.finish_expert_heads_smoke):
        p.error('choose --preflight, --smoke, --smoke-command, --train-key-query, --retrieve, --train-expert-heads, --smoke-expert-heads, or --finish-expert-heads-smoke')
    result = {'utc': utc(), 'status': 'preflight_only', 'inputs': verify_inputs(),
              'kq2_recipe': {'key': 'ArchMotionEncoder x1 absolute, distance distillation tau=.01',
                             'queries': ['(iii) GRU x VLM', '(i) GRU no VLM'],
                             'query_seeds': [1, 2, 3], 'expert': 'KQ2-CL K1',
                             't0': 'constant .15', 'closed_loop': {'A-prime-const': 'iii', 'B-prime-const': 'i',
                             'suites': ['LIBERO-90 10x10', 'LIBERO-10 10x10'],
                             'paired_baselines': ['old A-const', 'old B-const'],
                             'old_B_const_L10': 'must run'}}}
    if a.smoke:
        result['smoke'] = smoke()
    if a.train_key_query:
        result['model_stage'] = train_key_and_queries(a.device)
    if a.retrieve:
        result['retrieval_stage'] = build_retrieval(a.device)
    if a.train_expert_heads:
        result['expert_heads_stage'] = train_expert_and_heads(a.device,a.expert_updates)
    if a.smoke_expert_heads:
        result['expert_heads_smoke'] = smoke_train_expert_heads(a.device)
    if a.finish_expert_heads_smoke:
        result['expert_heads_smoke'] = finish_expert_heads_smoke(a.device)
    result['smoke_command'] = (
        'cd /work && source env/activate.sh && '
        'OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 '
        'CUDA_VISIBLE_DEVICES= taskset -c 8-47 nice -n 10 python '
        f'{HERE / "segv_gpu.py"} --smoke'
    )
    if a.smoke_command:
        print(result['smoke_command'])
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    log('gpu_stage_preflight', status=result['status'], smoke=a.smoke,
        input_units=result['inputs']['units'], source_counts=result['inputs']['source_counts'])


if __name__ == '__main__':
    main()
