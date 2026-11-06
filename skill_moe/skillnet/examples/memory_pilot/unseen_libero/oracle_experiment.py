"""Search-free representation upper bound with fixed LIBERO-40 memory.

Stages are resumable and write only beneath unseen-libero/oracle-01.  The
candidate is an actual memory row (or its declared representation decode); a
single target-action oracle chooses the lowest active-7 MSE candidate.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA

HERE=Path(__file__).resolve().parent
PI05=HERE.parent/'pi05_memory'
RUN=Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/run-01')
OUT=Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/oracle-01')
CHECKPOINT=Path('/work/.skillnet-state/checkpoints/openpi-assets/checkpoints/pi05_libero')
REPS=('raw','b12','pca24','pca12','dct14','dct21','keyframe21','keyframe35')
LEVELS=('none','m6default')
T0S=(.05,.1,.2,.3,.4,.5,.6,.8,1.)
NTRAIN=4000; BATCH=8; SEED=20261104; NOISE_SEED=4101; BOOT=5000

def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def save_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=True)+'\n');t.replace(p)
def load_unseen():return {p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in (OUT/'unseen_rows').glob('*.npy')}
def full_action(a):
    a=np.asarray(a,np.float32)
    if a.shape[-2:]!=(10,32):raise ValueError(a.shape)
    return a
def pad_bridge_action(actions):
    """Map active7 or model-width action chunks into the bridge's 32-D space."""
    a=np.asarray(actions,np.float32)
    if a.ndim!=3 or a.shape[1]!=10 or a.shape[2] not in (7,32):
        raise ValueError(f'expected [N,10,7|32], got {a.shape}')
    if a.shape[2]==32:return a
    out=np.zeros((len(a),10,32),np.float32);out[:,:,:7]=a
    return out
def zero_bridge_draft(reference):
    """Return zero in the model's padded action space, not active7 target space."""
    return np.zeros_like(pad_bridge_action(reference))
def active_err(pred,target):return np.mean((np.asarray(pred,np.float32)[...,:7]-np.asarray(target,np.float32)[...,:7])**2,axis=(-2,-1))
def rows_episode_key(task,episode):return np.asarray(task,np.int64)*100000+np.asarray(episode,np.int64)

def _raw_query(rows,actions,features,ids):
    return np.asarray(actions[ids],np.float32),np.asarray(rows['state'][ids],np.float32),np.asarray(features[ids],np.float32),np.asarray(rows['task'][ids]),np.asarray(rows['episode'][ids])
def _fixed_values(actions,ids,name,pca):
    from oracle_repr import encode_fixed
    return encode_fixed(np.asarray(actions[ids,:,:7],np.float32),name,pca=pca)

def _m6_values(values,mem_rows,rows,seed=SEED):
    from oracle_repr import RepresentationMetric,m6_default_medoids
    z=m6_default_medoids(values,np.asarray(rows['task'][mem_rows]),np.asarray(rows['episode'][mem_rows]),mem_rows,seed=seed)
    return z

def _pairwise_choice(target,cand_actions,block=256):
    t=np.asarray(target,np.float32).reshape(len(target),70);c=np.asarray(cand_actions,np.float32).reshape(len(cand_actions),70)
    cn=np.einsum('ij,ij->i',c,c);best=np.full(len(t),np.inf,np.float32);bi=np.zeros(len(t),np.int32)
    for st in range(0,len(t),block):
        sl=slice(st,min(st+block,len(t)));x=t[sl];xn=np.einsum('ij,ij->i',x,x)
        d=np.maximum(xn[:,None]+cn[None,:]-2*x@c.T,0.)/70.
        ix=d.argmin(1);best[sl]=d[np.arange(len(ix)),ix];bi[sl]=ix
    return bi,best

def _m1_oracle(codec,bases,candidate_rows,q_actions,q_state,q_feature,*,q_batch=64,c_chunk=2048):
    """Exact query-conditioned decode scan; no retrieval score or shortlist."""
    q_actions=np.asarray(q_actions,np.float32);q_state=np.asarray(q_state,np.float32);q_feature=np.asarray(q_feature,np.float32)
    cand=np.asarray(candidate_rows,np.int64);n=len(q_actions)
    picked=np.empty(n,np.int64);draft=np.empty((n,10,32),np.float32);err=np.empty(n,np.float32)
    target=q_actions[:,:,:7]
    for qs in range(0,n,q_batch):
        if qs % (q_batch*4)==0:
            print(json.dumps({'stage':'m1-oracle-scan','query_rows_done':qs,'query_rows_total':n,
                              'candidate_rows':len(cand)}),flush=True)
        sl=slice(qs,min(qs+q_batch,n));qn=sl.stop-sl.start
        r=codec.prior(q_feature[sl],q_state[sl])[1]
        best=np.full(qn,np.inf,np.float32);bestrow=np.full(qn,-1,np.int64);bestpred=np.zeros((qn,10,32),np.float32)
        for cs in range(0,len(cand),c_chunk):
            cr=cand[cs:cs+c_chunk];k=len(cr);bb=np.broadcast_to(np.asarray(bases[cr],np.float32)[None],(qn,k,bases.shape[1])).reshape(qn*k,-1)
            rr=np.broadcast_to(r[:,None,:],(qn,k,r.shape[-1])).reshape(qn*k,-1)
            ff=np.broadcast_to(q_feature[sl,None,:],(qn,k,q_feature.shape[1])).reshape(qn*k,-1)
            ss=np.broadcast_to(q_state[sl,None,:],(qn,k,q_state.shape[1])).reshape(qn*k,-1)
            yp=np.asarray(codec.decode(bb,rr,ff,ss),np.float32).reshape(qn,k,10,32)
            de=np.mean((yp[:,:,:,:7]-target[sl,None])**2,axis=(2,3));ix=de.argmin(1);val=de[np.arange(qn),ix]
            update=val<best
            if update.any():best[update]=val[update];bestrow[update]=cr[ix[update]];bestpred[update]=yp[np.arange(qn),ix][update]
        picked[sl]=bestrow;draft[sl]=bestpred;err[sl]=best
    return picked,draft,err

def _write_draft_cache(rep,level,split,query_ids,candidate_rows,draft,pre_mse,task,episode):
    p=OUT/'drafts'/f'{rep}_{level}_{split}.npz';p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        old=np.load(p,allow_pickle=False)
        if not np.array_equal(old['query_ids'],query_ids):raise AssertionError(f'cache id mismatch {p}')
        return p
    np.savez_compressed(p,query_ids=np.asarray(query_ids,np.int64),candidate_rows=np.asarray(candidate_rows,np.int64),
        draft=full_action(draft),pre_mse=np.asarray(pre_mse,np.float32),task=np.asarray(task,np.int32),episode=np.asarray(episode,np.int32))
    return p

def build_drafts():
    if not (OUT/'unseen_rows/features.json').exists():raise FileNotFoundError('VLM extraction must finish first')
    sys.path.insert(0,str(PI05));from oracle_repr import encode_fixed,decode_fixed,RepresentationMetric,m6_default_medoids
    base=load_module(PI05/'run.py','unseen_oracle_base');fair=load_module(PI05/'fair_expert'/'run.py','unseen_oracle_fair')
    import torch
    torch.set_num_threads(8)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(.20)
    rows=base.load_rows(RUN);actions=np.load(RUN/'rows/actions.npy',mmap_mode='r',allow_pickle=False);features=np.load(RUN/'pooled.npy',mmap_mode='r',allow_pickle=False)
    train=np.flatnonzero(~np.asarray(rows['validation'],bool)).astype(np.int64)
    memory=np.asarray(np.load(RUN/'memory_row_ids.npy',allow_pickle=False),np.int64)
    if len(memory)!=6583:raise AssertionError(f'memory rows changed: {len(memory)}')
    unseen=load_unseen();dev=np.flatnonzero(unseen['is_dev']).astype(np.int64);ev=np.flatnonzero(~unseen['is_dev']).astype(np.int64)
    if len(np.unique(unseen['task'][dev]))!=2 or len(np.unique(unseen['task'][ev]))!=10:raise AssertionError('fixed task split changed')
    # Stable training minibatches shared by every expert.
    batch_ids=fair.make_batch_indices(train,steps=NTRAIN,batch=BATCH,seed=SEED)
    np.save(OUT/'shared_batch_row_ids.npy',batch_ids)
    train_targets=np.asarray(actions[train,:,:7],np.float32)
    pca24=PCA(n_components=24,svd_solver='full').fit(train_targets.reshape(len(train),70))
    pca12=PCA(n_components=12,svd_solver='full').fit(train_targets.reshape(len(train),70))
    np.savez_compressed(OUT/'pca_fit.npz',train_rows=train,mean24=pca24.mean_,components24=pca24.components_,
                        mean12=pca12.mean_,components12=pca12.components_)
    # A fixed source row id maps to the external query matrix at row_count+local.
    qsets={'train':(train,False),'dev':(dev,True),'eval':(ev,True)}
    # Load M1 codec only for b12; raw memory data remain indexed by global run row IDs.
    codec=base.restore_codec(RUN);bases=np.load(RUN/'bases.npy',mmap_mode='r',allow_pickle=False)
    if torch.cuda.is_available():
        codec.device=torch.device('cuda')
        for name in ('encoder','prior_net','base_decoder','residual_decoder'):
            getattr(codec,name).to('cuda')
    if bases.shape!=(len(actions),12):raise AssertionError(bases.shape)
    for rep in REPS+('zero',):
      for level in (('none',) if rep=='zero' else LEVELS):
        if rep=='zero' and level=='m6default':continue  # zero has no partition; keep the single declared control.
        if rep=='zero':candidate_rows=np.asarray([-1],np.int64);candidate_actions=None;unit_meta={'unit_count':1,'unassigned_rate':0.,'epsilon':0.}
        elif rep=='b12':
            memvals=np.asarray(bases[memory],np.float32)
            if level=='m6default':
                # The contract here is the fixed representation-space distance
                # (M1 base Euclidean), not the decoder-probe metric used by a
                # separate M6 study.
                fit=m6_default_medoids(memvals,np.asarray(rows['task'][memory]),np.asarray(rows['episode'][memory]),memory,seed=SEED)
                candidate_rows=np.asarray(fit['medoid_rows'],np.int64);unit_meta={k:fit[k] for k in ('unit_count','unassigned_rate','epsilon','functional_median')}
            else:candidate_rows=memory.copy();unit_meta={'unit_count':len(memory),'unassigned_rate':0.,'epsilon':0.}
            candidate_actions=None
        else:
            pca=pca24 if rep=='pca24' else (pca12 if rep=='pca12' else None)
            memvals=encode_fixed(np.asarray(actions[memory,:,:7],np.float32),rep,pca=pca)
            if level=='m6default':
                fit=m6_default_medoids(memvals,np.asarray(rows['task'][memory]),np.asarray(rows['episode'][memory]),memory,seed=SEED)
                candidate_rows=np.asarray(fit['medoid_rows'],np.int64);loc={int(r):i for i,r in enumerate(memory)}
                ci=np.asarray([loc[int(r)] for r in candidate_rows],np.int64);candidate_values=memvals[ci]
                unit_meta={k:fit[k] for k in ('unit_count','unassigned_rate','epsilon','functional_median')}
            else:candidate_rows=memory.copy();candidate_values=memvals;unit_meta={'unit_count':len(memory),'unassigned_rate':0.,'epsilon':0.}
            candidate_actions=decode_fixed(candidate_values,rep,pca=pca)
            cp=OUT/'candidate_banks';cp.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(cp/f'{rep}_{level}.npz',row_ids=candidate_rows,actions=candidate_actions)
        if rep=='zero':
            unit_meta['memory_reconstruction_mse']=float(np.mean(np.asarray(actions[memory,:,:7],np.float32)**2))
        elif rep=='b12':
            own=_m1_candidate_drafts(codec,bases,candidate_rows,features,rows['state'])
            unit_meta['memory_reconstruction_mse']=float(np.mean((own[:,:,:7]-np.asarray(actions[candidate_rows,:,:7],np.float32))**2))
        else:
            unit_meta['memory_reconstruction_mse']=float(np.mean((candidate_actions[:,:,:7]-np.asarray(actions[candidate_rows,:,:7],np.float32))**2))
        for split,(qids,external) in qsets.items():
            if external:
                qact=np.asarray(unseen['actions'][qids],np.float32);qstate=np.asarray(unseen['state'][qids],np.float32);qfeature=np.asarray(unseen['pooled'][qids],np.float32)
                qtask=np.asarray(unseen['task'][qids],np.int32);qep=np.asarray(unseen['episode'][qids],np.int32)
                query_ids=len(actions)+qids
            else:
                qact=np.asarray(actions[qids],np.float32);qstate=np.asarray(rows['state'][qids],np.float32);qfeature=np.asarray(features[qids],np.float32)
                qtask=np.asarray(rows['task'][qids],np.int32);qep=np.asarray(rows['episode'][qids],np.int32);query_ids=qids
            existing=OUT/'drafts'/f'{rep}_{level}_{split}.npz'
            if existing.exists():
                old=np.load(existing,allow_pickle=False)
                if not np.array_equal(old['query_ids'],query_ids):raise AssertionError(f'cached query IDs differ: {existing}')
                continue
            if rep=='zero':
                selected=np.full(len(qids),-1,np.int64);draft=np.zeros((len(qids),10,32),np.float32);pre=np.mean(qact[:,:,:7]**2,axis=(1,2))
            elif rep=='b12':
                selected,draft,pre=_m1_oracle(codec,bases,candidate_rows,qact,qstate,qfeature)
            else:
                picked,pre=_pairwise_choice(qact[:,:,:7],candidate_actions)
                selected=candidate_rows[picked];draft=np.zeros((len(qids),10,32),np.float32);draft[:,:,:7]=candidate_actions[picked]
            _write_draft_cache(rep,level,split,query_ids,selected,draft,pre,qtask,qep)
            print(json.dumps({'stage':'drafts','representation':rep,'level':level,'split':split,'rows':len(qids),
                              'candidates':len(candidate_rows),'unit_count':unit_meta['unit_count'],'pre_mse':float(np.mean(pre))}),flush=True)
        save_json(OUT/f'fit-{rep}-{level}.json',{'representation':rep,'level':level,**unit_meta,'candidate_rows':int(len(candidate_rows))})
    # Dev q80 raw-oracle threshold: only held-out Dev tasks, no Eval actions.
    z=np.load(OUT/'drafts/raw_none_dev.npz',allow_pickle=False);threshold=float(np.quantile(z['pre_mse'],.8))
    save_json(OUT/'hard_threshold.json',{'source':'Dev raw oracle candidate active7 MSE q80','threshold':threshold,'dev_rows':len(z['pre_mse']),
        'hard_eval_rows':int(np.sum(np.load(OUT/'drafts/raw_none_eval.npz',allow_pickle=False)['pre_mse']>=threshold)),
        'eval_rows':len(np.load(OUT/'drafts/raw_none_eval.npz',allow_pickle=False)['pre_mse'])})
    save_json(OUT/'drafts_ready.json',{'status':'complete','train_rows':len(train),'memory_rows':len(memory),'dev_rows':len(dev),'eval_rows':len(ev),
        'representations':list(REPS)+['zero'],'levels':list(LEVELS),'training_batch_sha256':hashlib.sha256(batch_ids.tobytes()).hexdigest(),
        'hard_q80':threshold,'oracle_rule':'exact minimum active7 MSE among decoded memory medoids; one row, no average'})

def _cell_key(rep,level):return f'{rep}_{level}'

def train_cell(rep,level,*,threads=8):
    """Fair Expert training, with a fixed per-cell oracle draft."""
    import torch;torch.set_num_threads(threads)
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import optax
    import orbax.checkpoint as ocp
    from openpi.shared.nnx_utils import PathRegex,module_jit
    base=load_module(PI05/'run.py','unseen_train_base')
    fair=load_module(PI05/'fair_expert'/'run.py','unseen_train_fair')
    rows=base.load_rows(RUN);train=np.flatnonzero(~np.asarray(rows['validation'],bool)).astype(np.int64)
    unseen=load_unseen();dev=np.flatnonzero(unseen['is_dev']).astype(np.int64)
    key=_cell_key(rep,level);draft_key='zero_none' if rep=='zero' else key
    root=OUT/'experts'/key
    if (root/'trained.json').exists():return json.loads((root/'trained.json').read_text())
    attempt=int(os.environ.get('UNSEEN_TRAIN_ATTEMPT','1'))
    expdir=root if attempt==1 else root/f'attempt-{attempt:02d}'
    if expdir.exists() and any(expdir.iterdir()):raise FileExistsError(f'incomplete expert attempt requires explicit review: {expdir}')
    trz=np.load(OUT/'drafts'/f'{draft_key}_train.npz',allow_pickle=False)
    dvz=np.load(OUT/'drafts'/f'{draft_key}_dev.npz',allow_pickle=False)
    if not np.array_equal(trz['query_ids'],train):raise AssertionError('training row order changed')
    ext_ids=len(rows['task'])+dev
    if not np.array_equal(dvz['query_ids'],ext_ids):raise AssertionError('Dev row ids changed')
    batches=np.load(OUT/'shared_batch_row_ids.npy',allow_pickle=False)
    if batches.shape!=(NTRAIN,BATCH) or not np.all(np.isin(batches,train)):raise AssertionError('shared batch list changed')
    tmap={int(r):i for i,r in enumerate(train)}
    train_drafts=np.asarray(trz['draft'],np.float32)
    model,_=base.load_policy(CHECKPOINT)
    filt=nnx.All(nnx.Param,PathRegex(r'.*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*'))
    selected=nnx.state(model,filt)
    if sum(v.value.size for v in selected.flat_state().values())<100_000_000:raise AssertionError('unexpected trainable filter')
    tx=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(2.5e-5,b1=.9,b2=.95,weight_decay=1e-5));opt=tx.init(selected)
    step=base.make_bridge_step(filt,tx);rng=jax.random.key(SEED);expdir.mkdir(parents=True,exist_ok=True);(expdir/'bridge').mkdir()
    dev_obs=base.obs_batch(unseen,dev);dev_actions=jnp.asarray(unseen['actions'][dev],jnp.float32)
    dev_draft=jnp.asarray(dvz['draft'],jnp.float32)
    dev_global=ext_ids
    noise=np.stack([np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(NOISE_SEED),int(r)),(10,32))) for r in dev_global])
    best=float('inf');best_info=None;logs=[];tstart=time.monotonic()
    for it in range(NTRAIN):
        ids=batches[it];loc=np.asarray([tmap[int(r)] for r in ids],np.int64)
        rng,sub=jax.random.split(rng)
        loss,opt=step(model,opt,base.obs_batch(rows,ids),jnp.asarray(rows['actions'][ids]),jnp.asarray(train_drafts[loc]),sub)
        loss=float(loss)
        if not np.isfinite(loss):raise FloatingPointError(f'{key} loss nonfinite at {it+1}')
        rec={'step':it+1,'train_velocity_mse':loss,'elapsed_s':time.monotonic()-tstart}
        if (it+1)%500==0:
            ckpt=expdir/'bridge'/f'{it+1:06d}';ocp.PyTreeCheckpointer().save(ckpt,nnx.state(model,filt).to_pure_dict())
            bridge=module_jit(model.sample_bridge,static_argnames=('num_steps',))
            row={}
            for t0 in (.2,.4,.6):
                pred=[]
                for st in range(0,len(dev),BATCH):
                    sl=slice(st,min(st+BATCH,len(dev)))
                    y=bridge(jax.random.key(0),jax.tree.map(lambda x:x[sl],dev_obs),dev_draft[sl],
                             jnp.full((sl.stop-sl.start,),t0),num_steps=10,noise=jnp.asarray(noise[sl]))
                    pred.extend(active_err(np.asarray(y),np.asarray(dev_actions[sl])).tolist())
                row[f't0_{t0:.1f}']=float(np.mean(pred))
            sel_t=min((.2,.4,.6),key=lambda x:row[f't0_{x:.1f}']);sel=row[f't0_{sel_t:.1f}'];rec.update({f'dev_mse_{k}':v for k,v in row.items()});rec['dev_selected_t0']=sel_t;rec['dev_selected_mse']=sel
            if sel<best:best=sel;best_info={'path':str(ckpt),'steps':it+1,'t0':sel_t,'dev_active7_mse':sel,'selection':'two held-out unseen tasks; oracle candidate draft; no Eval windows'}
            logs.append(rec);save_json(expdir/'training_progress.json',{'status':'running','steps':it+1,'best':best_info,'logs':logs})
            print(json.dumps({'stage':'train','cell':key,**rec}),flush=True)
        elif (it+1)%20==0:logs.append(rec)
    save_json(expdir/'best_bridge.json',best_info)
    trained={'cell':key,'representation':rep,'level':level,'updates':NTRAIN,'seed':SEED,'batch_size':BATCH,
        'train_rows':len(train),'dev_rows':len(dev),'batch_row_ids_sha256':hashlib.sha256(batches.tobytes()).hexdigest(),
        'best':best_info,'optimizer':'AdamW lr=2.5e-5 b1=.9 b2=.95 wd=1e-5 clip=1','training_seconds':time.monotonic()-tstart,
        'loss':'shared fair expert bridge velocity objective; active7','checkpoint_interval':500}
    trained['attempt']=attempt
    save_json(root/'trained.json',trained);return trained

def training_queue(max_parallel=2, mem_fraction='.25'):
    if not (OUT/'drafts_ready.json').exists():raise FileNotFoundError('draft stage incomplete')
    specs=[(r,l) for r in REPS for l in LEVELS]+[('zero','none')]
    slots=['8-15'];running={};pending=list(specs);records=[]
    # Resume safely after interrupted/OOM runs: every existing attempt directory
    # is preserved, and a fresh attempt gets a new numbered directory.
    attempts={}
    for rep,lev in specs:
        key=_cell_key(rep,lev);root=OUT/'experts'/key
        prior=[1] if root.exists() and any(root.iterdir()) else []
        prior += [int(p.name.split('-')[-1]) for p in root.glob('attempt-*') if p.is_dir() and p.name.split('-')[-1].isdigit()]
        attempts[key]=max(prior,default=0)
    attempt_limits={key:value+2 for key,value in attempts.items()}  # one new try after the next fresh attempt
    logs=OUT/'logs';logs.mkdir(exist_ok=True)
    def launch(spec,slot):
        rep,lev=spec;key=_cell_key(rep,lev);log=(logs/f'train-{key}.log').open('ab')
        env=os.environ.copy();env.update(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_MEM_FRACTION=str(mem_fraction),UNSEEN_TRAIN_ATTEMPT=str(attempts[key]+1))
        cmd=['taskset','-c',slots[slot],'nice','-n','10',sys.executable,str(HERE/'oracle_experiment.py'),'train-cell','--representation',rep,'--level',lev,'--threads','8']
        p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True);attempts[key]+=1;return p,log
    while pending or running:
        while pending and len(running)<max_parallel:
            # Keep the assigned share within the requested cap and do not start
            # another wave if the visible free GPU memory is unexpectedly low.
            free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
            if free<130000:break
            spec=pending.pop(0);slot=next(i for i in range(max_parallel) if i not in [x[0] for x in running.values()])
            p,log=launch(spec,slot);running[p.pid]=(slot,spec,p,log)
            save_json(OUT/'training_queue.json',{'running':[{'pid':v[2].pid,'cell':_cell_key(*v[1]),'cpu':slots[v[0]]} for v in running.values()],
                'pending':[_cell_key(*x) for x in pending],'completed':records,'gpu_free_mib_at_last_launch':free})
        time.sleep(5)
        for pid,(slot,spec,p,log) in list(running.items()):
            rc=p.poll()
            if rc is None:continue
            log.close();key=_cell_key(*spec)
            record={'cell':key,'exit_code':rc,'trained':(OUT/'experts'/key/'trained.json').exists(),'attempt':attempts[key]}
            records.append(record);del running[pid]
            if rc!=0 and not record['trained']:
                # One delayed retry only; a persistent failure is recorded and
                # does not discard already-completed cells.
                if attempts[key] < attempt_limits[key]:
                    time.sleep(60)
                    p2,l2=launch(spec,slot);running[p2.pid]=(slot,spec,p2,l2)
                    record['retry_pid']=p2.pid
        save_json(OUT/'training_queue.json',{'running':[{'pid':v[2].pid,'cell':_cell_key(*v[1]),'cpu':slots[v[0]]} for v in running.values()],
            'pending':[_cell_key(*x) for x in pending],'completed':records})
        if not running and pending:
            if int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])<130000:time.sleep(60)
    latest={}
    for r in records:latest[r['cell']]=r
    failed=[r for r in latest.values() if not r['trained']]
    save_json(OUT/'training_queue.json',{'status':'complete' if not failed else 'incomplete','records':records,'latest':latest,'failed':failed})
    if failed:raise RuntimeError(f'Expert queue incomplete: {failed}')


def _pca_from_saved():
    z=np.load(OUT/'pca_fit.npz',allow_pickle=False)
    from sklearn.decomposition import PCA
    p24=PCA(n_components=24);p24.mean_=z['mean24'];p24.components_=z['components24'];p24.n_features_in_=70
    p12=PCA(n_components=12);p12.mean_=z['mean12'];p12.components_=z['components12'];p12.n_features_in_=70
    return p24,p12


def _m1_candidate_drafts(codec,bases,row_ids,features,states,batch=256):
    out=np.empty((len(row_ids),10,32),np.float32)
    for st in range(0,len(row_ids),batch):
        ix=np.asarray(row_ids[st:st+batch],np.int64)
        r=codec.prior(np.asarray(features[ix],np.float32),np.asarray(states[ix],np.float32))[1]
        out[st:st+len(ix)]=codec.decode(np.asarray(bases[ix],np.float32),r,
                np.asarray(features[ix],np.float32),np.asarray(states[ix],np.float32))
    return out


def _cosine_action(pred,target):
    a=np.asarray(pred,np.float32)[...,:3].reshape(len(pred),-1);b=np.asarray(target,np.float32)[...,:3].reshape(len(target),-1)
    den=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
    return np.divide(np.einsum('ij,ij->i',a,b),den,out=np.zeros(len(a),np.float32),where=den>1e-8)


def _metrics(pred,target,mask):
    p=np.asarray(pred,np.float32)[mask];y=np.asarray(target,np.float32)[mask]
    if not len(p):return {'n':0}
    return {'n':int(len(p)),'mse':float(np.mean((p[:,:,:7]-y[:,:,:7])**2)),
        'median_mse':float(np.median(np.mean((p[:,:,:7]-y[:,:,:7])**2,axis=(1,2)))),
        'xyz_mse':float(np.mean((p[:,:,:3]-y[:,:,:3])**2)),
        'rotation_mse':float(np.mean((p[:,:,3:6]-y[:,:,3:6])**2)),
        'gripper_mse':float(np.mean((p[:,:,6]-y[:,:,6])**2)),
        'gripper_sign_accuracy':float(np.mean(np.sign(p[:,:,6])==np.sign(y[:,:,6]))),
        'direction_cosine':float(np.mean(_cosine_action(p,y)))}


def evaluate_cell(rep,level,*,batch=8,threads=8):
    import torch;torch.set_num_threads(threads)
    import jax
    import jax.numpy as jnp
    from openpi.shared.nnx_utils import module_jit
    if torch.cuda.is_available():torch.cuda.set_per_process_memory_fraction(.20)
    base=load_module(PI05/'run.py','unseen_eval_base')
    skill=load_module(PI05/'skill_expert'/'run.py','unseen_eval_skill')
    rows=base.load_rows(RUN);actions=np.load(RUN/'rows/actions.npy',mmap_mode='r',allow_pickle=False)
    features=np.load(RUN/'pooled.npy',mmap_mode='r',allow_pickle=False);states=rows['state']
    unseen=load_unseen();ids=np.flatnonzero(~np.asarray(unseen['is_dev'],bool)).astype(np.int64)
    row_ids=(len(actions)+ids).astype(np.int64);target=np.asarray(unseen['actions'][ids],np.float32)
    task=np.asarray(unseen['task'][ids],np.int32);episode=np.asarray(unseen['episode'][ids],np.int32)
    episode_key=task.astype(np.int64)*100000+episode
    hard_cut=float(json.loads((OUT/'hard_threshold.json').read_text())['threshold'])
    raw=np.load(OUT/'drafts/raw_none_eval.npz',allow_pickle=False)
    if not np.array_equal(raw['query_ids'],row_ids):raise AssertionError('Eval row order changed')
    hard=np.asarray(raw['pre_mse'])>=hard_cut
    key=_cell_key(rep,level);draft_key='zero_none' if rep=='zero' else key
    cache=np.load(OUT/'drafts'/f'{draft_key}_eval.npz',allow_pickle=False)
    if not np.array_equal(cache['query_ids'],row_ids):raise AssertionError('draft row order changed')
    primary=np.asarray(cache['draft'],np.float32)
    # Cross-task permutation of the same selected-candidate multiset.
    if rep=='zero':shuffle=np.zeros_like(primary);shuffle_rows=np.full(len(ids),-1,np.int64)
    else:
        from oracle_repr import deterministic_cross_task_shuffle,deterministic_cross_task_from_pool
        try:
            shuffle_rows=deterministic_cross_task_shuffle(cache['candidate_rows'],task,np.asarray(rows['task']),seed=NOISE_SEED+77)
            shuffle_mode='bijection_of_selected_candidates'
        except ValueError:
            # If the oracle-selected source multiset is too imbalanced to form
            # a cross-task permutation, draw unique counterfactuals from the
            # same representation's candidate pool instead.
            bank_path=OUT/'candidate_banks'/f'{rep}_{level}.npz'
            if bank_path.exists():pool_rows=np.load(bank_path,allow_pickle=False)['row_ids']
            elif rep=='b12':
                memory=np.asarray(np.load(RUN/'memory_row_ids.npy',allow_pickle=False),np.int64)
                if level=='m6default':
                    from oracle_repr import m6_default_medoids
                    values=np.asarray(np.load(RUN/'bases.npy',mmap_mode='r',allow_pickle=False)[memory],np.float32)
                    fit=m6_default_medoids(values,np.asarray(rows['task'][memory]),np.asarray(rows['episode'][memory]),memory,seed=SEED)
                    pool_rows=np.asarray(fit['medoid_rows'],np.int64)
                else:pool_rows=memory
            else:pool_rows=cache['candidate_rows']
            shuffle_rows=deterministic_cross_task_from_pool(pool_rows,task,np.asarray(rows['task']),seed=NOISE_SEED+77)
            shuffle_mode='unique_cross_task_from_same_candidate_pool'
        if rep=='b12':
            codec=base.restore_codec(RUN)
            if torch.cuda.is_available():
                codec.device=torch.device('cuda')
                for name in ('encoder','prior_net','base_decoder','residual_decoder'):getattr(codec,name).to('cuda')
            bases=np.load(RUN/'bases.npy',mmap_mode='r',allow_pickle=False)
            shuffle=_m1_candidate_drafts(codec,bases,shuffle_rows,features,states)
            del codec
        else:
            bank=np.load(OUT/'candidate_banks'/f'{rep}_{level}.npz',allow_pickle=False)
            loc={int(r):i for i,r in enumerate(bank['row_ids'])}
            shuffle=np.asarray(bank['actions'][np.asarray([loc[int(r)] for r in shuffle_rows])],np.float32)
    # The bridge operates on the model's padded 32-D action space.  Evaluation
    # targets are intentionally stored as active7, so do not derive the zero
    # draft or output buffer from `target` (which is [N,10,7]).
    zero=zero_bridge_draft(primary)
    expert_dir=OUT/'experts'/key
    trained=json.loads((expert_dir/'trained.json').read_text())
    checkpoint=Path(trained['best']['path'])
    model,_=base.load_policy(CHECKPOINT);skill.load_expert(model,checkpoint)
    bridge=module_jit(model.sample_bridge,static_argnames=('num_steps',))
    # Fixed per-row noise, keyed by a stable global row ID.
    noise=np.stack([np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(NOISE_SEED),int(r)),(10,32))) for r in row_ids])
    obs=base.obs_batch(unseen,ids)
    pred={}
    def run(drafts,t0):
        drafts=pad_bridge_action(drafts)
        out=np.empty((len(ids),10,32),np.float32)
        for st in range(0,len(ids),batch):
            sl=slice(st,min(st+batch,len(ids)))
            o=jax.tree.map(lambda x:x[sl],obs)
            out[sl]=np.asarray(bridge(jax.random.key(0),o,jnp.asarray(drafts[sl]),
                     jnp.full((sl.stop-sl.start,),float(t0)),num_steps=10,noise=jnp.asarray(noise[sl])))
        return out
    # Shared Expert-specific t0=1 reference; start draft is mathematically irrelevant.
    pred['t1']=run(zero,1.)
    for name,draft in [('primary',primary),('shuffle',shuffle),('zero',zero)]:
        pred[name]={}
        for t0 in T0S:pred[name][f'{t0:.2f}']=run(draft,t0)
    outdir=OUT/'predictions'/key;outdir.mkdir(parents=True,exist_ok=True)
    payload={'row_ids':row_ids,'task':task,'episode':episode,'episode_key':episode_key,'target':target[:,:,:7],
             'hard':hard,'candidate_rows':np.asarray(cache['candidate_rows']),'shuffle_rows':shuffle_rows,
             'primary_draft':primary[:,:,:7],'shuffle_draft':shuffle[:,:,:7],'primary_pre_mse':np.asarray(cache['pre_mse'])}
    payload['pred_t1']=pred['t1'][:,:,:7]
    for name in ('primary','shuffle','zero'):
        for t0,value in pred[name].items():payload[f'pred_{name}_t{t0}']=value[:,:,:7]
    np.savez_compressed(outdir/'eval_rows.npz',**payload)
    # Dev-selected cell checkpoint/t0; Eval is reported only after selection is frozen.
    selected_t0=float(trained['best']['t0'])
    episode_groups=episode_key
    curves=[]
    for subset,mask in [('all',np.ones(len(ids),bool)),('hard',hard)]:
      for draft_name in ('primary','shuffle','zero'):
        for t0 in T0S:
            yhat=pred[draft_name][f'{t0:.2f}'];m=_metrics(yhat,target,mask)
            curves.append({'representation':rep,'level':level,'draft':draft_name,'t0':t0,'subset':subset,**m,
                 'pre_generation_mse':float(np.mean(np.asarray(cache['pre_mse'])[mask])) if draft_name=='primary' else
                       (float(np.mean(active_err(shuffle[mask],target[mask]))) if draft_name=='shuffle' else float(np.mean(active_err(zero[mask],target[mask])))),
                 'selected_on_dev':bool(draft_name=='primary' and abs(t0-selected_t0)<1e-8)})
      for name,yhat in [('same_expert_t1',pred['t1'])]:
        curves.append({'representation':rep,'level':level,'draft':name,'t0':1.,'subset':subset,**_metrics(yhat,target,mask),
                       'pre_generation_mse':None,'selected_on_dev':False})
    # Per-row bootstrap contrasts, with rows clustered by task/episode.
    from oracle_repr import episode_bootstrap_diff
    contrasts=[]
    selected_key=f'{selected_t0:.2f}'
    selected_pred=pred['primary'][selected_key]
    for subset,mask in [('all',np.ones(len(ids),bool)),('hard',hard)]:
        if not np.any(mask):continue
        ep=episode_groups[mask];own=active_err(selected_pred[mask],target[mask])
        same_t1=active_err(pred['t1'][mask],target[mask]);shuf=active_err(pred['shuffle'][selected_key][mask],target[mask])
        contrasts.append({'contrast':'cell_primary_minus_same_expert_t1','subset':subset,'t0':selected_t0,
                          **episode_bootstrap_diff(own,same_t1,ep,reps=BOOT,seed=NOISE_SEED)})
        contrasts.append({'contrast':'cell_primary_minus_shuffle','subset':subset,'t0':selected_t0,
                          **episode_bootstrap_diff(own,shuf,ep,reps=BOOT,seed=NOISE_SEED+1)})
    result={'cell':key,'representation':rep,'level':level,'eval_rows':len(ids),'hard_rows':int(hard.sum()),
            'dev_selected':trained['best'],'shuffle_mode':('zero_control' if rep=='zero' else shuffle_mode),
            'shuffle_cross_task_verified':bool(rep=='zero' or np.all(np.asarray(rows['task'])[shuffle_rows]!=task)),
            'curves':curves,'contrasts':contrasts,'prediction_file':str(outdir/'eval_rows.npz')}
    save_json(outdir/'result.json',result)
    print(json.dumps({'stage':'eval-cell','cell':key,'selected_t0':selected_t0,'rows':len(ids),'hard':int(hard.sum())}),flush=True)
    del model,bridge
    return result


def evaluate_all():
    import torch;torch.set_num_threads(8)
    import jax
    import jax.numpy as jnp
    from openpi.shared.nnx_utils import module_jit
    base=load_module(PI05/'run.py','unseen_eval_normal_base')
    unseen=load_unseen();ids=np.flatnonzero(~np.asarray(unseen['is_dev'],bool)).astype(np.int64);actions=np.load(RUN/'rows/actions.npy',mmap_mode='r',allow_pickle=False)
    row_ids=len(actions)+ids;target=np.asarray(unseen['actions'][ids],np.float32)
    obs=base.obs_batch(unseen,ids)
    noise=np.stack([np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(NOISE_SEED),int(r)),(10,32))) for r in row_ids])
    model,_=base.load_policy(CHECKPOINT);fn=module_jit(model.sample_actions,static_argnames=('num_steps',));pred=np.empty_like(target)
    for st in range(0,len(ids),8):
        sl=slice(st,min(st+8,len(ids)));pred[sl]=np.asarray(fn(jax.random.key(0),jax.tree.map(lambda x:x[sl],obs),num_steps=10,noise=jnp.asarray(noise[sl])))
    np.savez_compressed(OUT/'official_normal_eval.npz',row_ids=row_ids,target=target[:,:,:7],prediction=pred[:,:,:7])
    save_json(OUT/'official_normal_metrics.json',{'all':_metrics(pred,target,np.ones(len(ids),bool)),'rows':len(ids)})
    print(json.dumps({'stage':'normal-eval','rows':len(ids),'metrics':_metrics(pred,target,np.ones(len(ids),bool))}),flush=True)
    del model,fn
    jax.clear_caches()
    cells=[(r,l) for r in REPS for l in LEVELS]+[('zero','none')]
    for rep,level in cells:
        result_path=OUT/'predictions'/_cell_key(rep,level)/'result.json'
        if result_path.exists():continue
        evaluate_cell(rep,level)


def finalize():
    import csv
    import matplotlib.pyplot as plt
    import jax
    from oracle_repr import episode_bootstrap_diff
    # Aggregate all cells and compute the requested paired contrasts on fixed Eval episodes.
    rows=load_unseen();ev=np.flatnonzero(~np.asarray(rows['is_dev'],bool));tasks=np.asarray(rows['task'][ev]);eps=np.asarray(rows['episode'][ev]);epkey=tasks.astype(np.int64)*100000+eps
    target=np.asarray(rows['actions'][ev],np.float32);hard_cut=float(json.loads((OUT/'hard_threshold.json').read_text())['threshold'])
    hard=np.asarray(np.load(OUT/'drafts/raw_none_eval.npz',allow_pickle=False)['pre_mse'])>=hard_cut
    keys=[_cell_key(r,l) for r in REPS for l in LEVELS]+['zero_none']
    result={k:json.loads((OUT/'predictions'/k/'result.json').read_text()) for k in keys}
    curves=[]
    for k,v in result.items():curves.extend(v['curves'])
    # Add official reference curve (a horizontal reference, shown at every grid point).
    norm=np.load(OUT/'official_normal_eval.npz',allow_pickle=False);normal=np.asarray(norm['prediction'],np.float32)
    for subset,mask in [('all',np.ones(len(ev),bool)),('hard',hard)]:
        m=_metrics(normal,target,mask)
        for t0 in T0S:curves.append({'representation':'official_pi05','level':'reference','draft':'normal','t0':t0,'subset':subset,**m,'pre_generation_mse':None,'selected_on_dev':False})
    reference_summary={'official_pi05':{},'mean_same_expert_t1_across_cells':{}}
    for subset,mask in [('all',np.ones(len(ev),bool)),('hard',hard)]:
        reference_summary['official_pi05'][subset]=_metrics(normal,target,mask)
        ref=[]
        for k in keys:
            rr=np.load(OUT/'predictions'/k/'eval_rows.npz',allow_pickle=False)
            ref.append(_metrics(rr['pred_t1'],target,mask)['mse'])
        reference_summary['mean_same_expert_t1_across_cells'][subset]=float(np.mean(ref))
    # Cache expected raw-oracle draft arrays for paired references and prepare a compact table.
    rows_csv=[];contrasts=[]
    for rep in REPS:
      for lev in LEVELS:
        k=_cell_key(rep,lev);v=result[k];predfile=np.load(OUT/'predictions'/k/'eval_rows.npz',allow_pickle=False)
        selected_t0=float(v['dev_selected']['t0']);t1=np.asarray(predfile['pred_t1'],np.float32)
        rawlev=result[_cell_key('raw',lev)];raw_t0=float(rawlev['dev_selected']['t0'])
        rawfile=np.load(OUT/'predictions'/_cell_key('raw',lev)/'eval_rows.npz',allow_pickle=False)
        fit_meta=json.loads((OUT/f'fit-{rep}-{lev}.json').read_text())
        for subset,mask in [('all',np.ones(len(ev),bool)),('hard',hard)]:
            for t0 in T0S:
                p=np.asarray(predfile[f'pred_primary_t{t0:.2f}'],np.float32)
                sh=np.asarray(predfile[f'pred_shuffle_t{t0:.2f}'],np.float32)
                rawp=np.asarray(rawfile[f'pred_primary_t{t0:.2f}'],np.float32)
                for contrast,b in [('minus_raw_same_level',rawp[mask]),('minus_same_expert_t1',t1[mask]),('minus_shuffle',sh[mask])]:
                    aerr=active_err(p[mask],target[mask]);berr=active_err(b,target[mask])
                    contrasts.append({'representation':rep,'level':lev,'contrast':contrast,'subset':subset,
                         't0':t0,'selected_on_dev':bool(abs(t0-selected_t0)<1e-8),
                         **episode_bootstrap_diff(aerr,berr,epkey[mask],reps=BOOT,seed=NOISE_SEED+len(contrasts))})
            m=_metrics(p,target,mask)
            rows_csv.append({'representation':rep,'level':lev,'subset':subset,'selected_t0':selected_t0,**m,
                'candidate_pre_mse':float(np.mean(predfile['primary_pre_mse'][mask])),
                'near_rate_mse_le_0p04':float(np.mean(predfile['primary_pre_mse'][mask]<=.04)),
                'reconstruction_mse_memory':fit_meta.get('memory_reconstruction_mse'),
                'unit_count':fit_meta.get('unit_count'),'unassigned_rate':fit_meta.get('unassigned_rate'),'epsilon':fit_meta.get('epsilon')})
    zkey='zero_none';zres=result[zkey];zfile=np.load(OUT/'predictions'/zkey/'eval_rows.npz',allow_pickle=False)
    zt0=float(zres['dev_selected']['t0'])
    for subset,mask in [('all',np.ones(len(ev),bool)),('hard',hard)]:
        zp=np.asarray(zfile[f'pred_primary_t{zt0:.2f}'],np.float32)
        rows_csv.append({'representation':'zero','level':'none','subset':subset,'selected_t0':zt0,**_metrics(zp,target,mask),
            'candidate_pre_mse':float(np.mean(active_err(np.zeros_like(target[mask]),target[mask]))),
            'near_rate_mse_le_0p04':None,'reconstruction_mse_memory':None})
        for t0 in T0S:
            zp=np.asarray(zfile[f'pred_primary_t{t0:.2f}'],np.float32);zt=np.asarray(zfile['pred_t1'],np.float32)
            contrasts.append({'representation':'zero','level':'none','contrast':'minus_same_expert_t1','subset':subset,'t0':t0,
               'selected_on_dev':bool(abs(t0-zt0)<1e-8),**episode_bootstrap_diff(active_err(zp[mask],target[mask]),active_err(zt[mask],target[mask]),epkey[mask],reps=BOOT,seed=NOISE_SEED+len(contrasts))})
            contrasts.append({'representation':'zero','level':'none','contrast':'minus_zero_shuffle_exact_sanity','subset':subset,'t0':t0,
               'selected_on_dev':bool(abs(t0-zt0)<1e-8),**episode_bootstrap_diff(active_err(zp[mask],target[mask]),active_err(zp[mask],target[mask]),epkey[mask],reps=BOOT,seed=NOISE_SEED+len(contrasts))})
    for subset,mask in [('all',np.ones(len(ev),bool)),('hard',hard)]:
        z=_metrics(normal,target,mask);rows_csv.append({'representation':'official_pi05','level':'reference','subset':subset,'selected_t0':None,**z})
    save_json(OUT/'summary.json',{'status':'complete','eval_rows':len(ev),'hard_rows':int(hard.sum()),'hard_threshold':hard_cut,
        'curves':curves,'selected_cell_metrics':rows_csv,'paired_contrasts':contrasts,'reference_summary':reference_summary,'episode_bootstrap_reps':BOOT,
        'limitations':['oracle sees the target action to select the nearest candidate','offline, not closed-loop','single training seed','unseen split applies to these demonstrations; public pretraining history is not audited']})
    with (OUT/'metrics.csv').open('w',newline='') as f:
        fields=sorted({k for r in curves for k in r});w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(curves)
    with (OUT/'paired_contrasts.csv').open('w',newline='') as f:
        fields=sorted({k for r in contrasts for k in r});w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(contrasts)
    # Persist row-wise all t0 predictions, drafts, and metrics already contained in cell NPZs.
    # Summary figure: one row per subset, columns per action representation; shared y scale.
    rep_labels=list(REPS)+['zero','official_pi05'];colors=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#7868a8','#4f9d9a','#a36f42','#52514e','#9b9b9b']
    markers=['o','s','^','D','v','P','X','*','h','+'];color=dict(zip(rep_labels,colors));marker=dict(zip(rep_labels,markers))
    plt.rcParams.update({'figure.facecolor':'#fcfcfb','axes.facecolor':'#fcfcfb','text.color':'#0b0b0b','axes.labelcolor':'#0b0b0b','xtick.color':'#52514e','ytick.color':'#52514e','font.size':9})
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True,sharey=True)
    for ax,subset in zip(axes,['all','hard']):
      subset_mask=np.ones(len(ev),bool) if subset=='all' else hard
      for rep in rep_labels:
        level_lines=[('none','-')] if rep in ('zero','official_pi05') else [('none','-'),('m6default','--')]
        for lev,ls in level_lines:
            sel=[x for x in curves if x['subset']==subset and x['representation']==rep and x['draft'] in ('primary','normal') and (x['level']==lev or rep in ('zero','official_pi05'))]
            if not sel:continue
            sel=sorted(sel,key=lambda x:x['t0']);ax.plot([x['t0'] for x in sel],[x['mse'] for x in sel],color=color[rep],marker=marker[rep],markersize=8,lw=2,ls=ls,label=f'{rep}/{lev}' if rep not in ('zero','official_pi05') else rep)
      expert_t1=[]
      for name,v in result.items():
        rr=np.load(OUT/'predictions'/name/'eval_rows.npz',allow_pickle=False)
        expert_t1.append(float(np.mean(active_err(rr['pred_t1'],target)[subset_mask])))
      ax.axhline(float(np.mean(expert_t1)),color='#777777',lw=1,ls=':',label='mean trained Expert t0=1')
      ax.axvspan(.2,.6,color='#e5e3de',alpha=.42,label='training range [0.2,0.6]')
      ax.set_ylabel('Active-7 MSE');ax.set_title('All Eval rows' if subset=='all' else 'Hard rows (Dev raw-oracle q80)');ax.grid(True,color='#d8d7d2',linewidth=.5)
    axes[-1].set_xlabel('Start noise t0');axes[0].legend(ncol=4,fontsize=7,loc='upper right')
    fig.tight_layout();figdir=Path('/work/docs/research/09_unseen_libero/figures');figdir.mkdir(parents=True,exist_ok=True);fig.savefig(figdir/'oracle_noise_curves.png',dpi=200,bbox_inches='tight')
    outfig=OUT/'figures';outfig.mkdir(parents=True,exist_ok=True);fig.savefig(outfig/'oracle_noise_curves.png',dpi=200,bbox_inches='tight');plt.close(fig)
    # Markdown report with short selected-t0 table and all paired intervals.
    def fmt(x):return '—' if x is None else (f'{x:.4f}' if isinstance(x,(int,float,np.floating)) else str(x))
    raw_ref=next(x for x in rows_csv if x['representation']=='raw' and x['level']=='none' and x['subset']=='all')
    comp_notes=[]
    for rr in REPS:
        if rr=='raw':continue
        for lev in LEVELS:
            c=next(x for x in contrasts if x['representation']==rr and x['level']==lev and x['contrast']=='minus_raw_same_level' and x['subset']=='all' and x.get('selected_on_dev'))
            comp_notes.append(f"{rr}/{lev} Δ{c['mean']:+.4f} [{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}]")
    time_notes=[]
    for rr in ('dct14','dct21','keyframe21','keyframe35'):
        a=next(x for x in rows_csv if x['representation']==rr and x['level']=='none' and x['subset']=='all')
        h=next(x for x in rows_csv if x['representation']==rr and x['level']=='none' and x['subset']=='hard')
        c=next(x for x in contrasts if x['representation']==rr and x['level']=='none' and x['contrast']=='minus_raw_same_level' and x['subset']=='all' and x.get('selected_on_dev'))
        time_notes.append(f"{rr} 全/難={a['mse']:.4f}/{h['mse']:.4f}, Δraw={c['mean']:+.4f}[{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}]")
    lines=['# 未学習LIBERO：表現上限の結果','',
      f"- 未学習タスクで圧縮は役立つか：raw/none MSE={raw_ref['mse']:.4f}。圧縮表現−同じクラスタ水準のraw（Dev選択t0、全Eval、95%CI）は {'; '.join(comp_notes)}。差なしを同等とは扱わない。Eval {len(ev)}窓、難しい行 {int(hard.sum())}窓。",
      f"- 時間方向の圧縮：{'；'.join(time_notes)}。DCT/キーフレームの次元はDevで選んでいない。",
      '- これは正解行動を見て選ぶ検索なし上限診断で、通常の検索や閉ループ性能ではない。訓練seedは1つで、公開の事前学習履歴全体での未学習を保証しない。', '',
      '表記：dct14/dct21はDCT k=2/3（14/21係数）、keyframe21/keyframe35は等間隔k=3/5点（21/35値）。',
      '',
      '## Devで選んだt0でのEval', '',
      '| 表現 | クラスタ | 単位数 | 未割当率 | t0 | 全行 MSE | 難しい行 MSE | 候補前 MSE | ≤0.04 | 復元 MSE（memory代表） | 軌道方向cos |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows_csv:
        if r['representation']=='official_pi05':continue
        a=next(x for x in rows_csv if x['representation']==r['representation'] and x['level']==r['level'] and x['subset']=='all')
        h=next(x for x in rows_csv if x['representation']==r['representation'] and x['level']==r['level'] and x['subset']=='hard')
        if r['subset']!='all':continue
        lines.append(f"| {r['representation']} | {r['level']} | {fmt(a.get('unit_count'))} | {fmt(a.get('unassigned_rate'))} | {fmt(a['selected_t0'])} | {fmt(a['mse'])} | {fmt(h['mse'])} | {fmt(a['candidate_pre_mse'])} | {fmt(a['near_rate_mse_le_0p04'])} | {fmt(a.get('reconstruction_mse_memory'))} | {fmt(a['direction_cosine'])} |")
    lines+=['', '## ペア差（生成後active-7 MSE、episode bootstrap 95% CI）','',
      '| 表現 | クラスタ | 比較 | 対象 | 差 | 95% CI |','|---|---|---|---|---:|---:|']
    for c in contrasts:
        if c.get('selected_on_dev',False):
            lines.append(f"| {c['representation']} | {c['level']} | {c['contrast']} (t0={c.get('t0',c.get('candidate_t0'))}) | {c['subset']} | {c['mean']:.4f} | [{c['ci95'][0]:.4f}, {c['ci95'][1]:.4f}] |")
    lines+=['',f"基準の通常π0.5 MSE（全行/難しい行）={reference_summary['official_pi05']['all']['mse']:.4f}/{reference_summary['official_pi05']['hard']['mse']:.4f}。同一Expert群のt0=1平均={reference_summary['mean_same_expert_t1_across_cells']['all']:.4f}/{reference_summary['mean_same_expert_t1_across_cells']['hard']:.4f}。", '', '## t0曲線','', '![t0ごとの生成後誤差](figures/oracle_noise_curves.png)', '',
      't0=.05,.1,.8,1.0はExpertの学習範囲[.2,.6]外の外挿である。通常π0.5はt0を持たない水平基準、同一Expertのt0=1は各Expert別に評価した。全t0の3種下書き・対比較の全値はmetrics.csv / paired_contrasts.csv。',
      '生成前の候補誤差とmemory代表の復元誤差は表に記録した。主候補は実在窓または実在medoidを選び、混合していない。シャッフルは同じ候補multisetを別タスクへ一対一置換し、成立しない場合は同じcandidate poolの別タスクから一意割当または重複許容抽出を行い、方式をresults JSONに記録した。',
      '対象は未学習タスク実演を意図的に含む候補選択上限であり、汎化や閉ループでの実効性を主張しない。']
    report='\n'.join(lines)+'\n';(OUT/'oracle_results.md').write_text(report);Path('/work/docs/research/09_unseen_libero/oracle_results.md').write_text(report)
    save_json(OUT/'finalized.json',{'status':'complete','rows':len(ev),'hard_rows':int(hard.sum()),'report':str(OUT/'oracle_results.md'),'figure':str(figdir/'oracle_noise_curves.png')})


def main():
    p=argparse.ArgumentParser();sp=p.add_subparsers(dest='stage',required=True)
    sp.add_parser('drafts')
    tc=sp.add_parser('train-cell');tc.add_argument('--representation',required=True);tc.add_argument('--level',required=True);tc.add_argument('--threads',type=int,default=8)
    tq=sp.add_parser('train-queue');tq.add_argument('--max-parallel',type=int,default=2);tq.add_argument('--mem-fraction',default='.25')
    ec=sp.add_parser('eval-cell');ec.add_argument('--representation',required=True);ec.add_argument('--level',required=True);ec.add_argument('--threads',type=int,default=8);ec.add_argument('--batch',type=int,default=8)
    sp.add_parser('evaluate-all');sp.add_parser('finalize')
    a=p.parse_args()
    if a.stage=='drafts':build_drafts()
    elif a.stage=='train-cell':train_cell(a.representation,a.level,threads=a.threads)
    elif a.stage=='train-queue':training_queue(a.max_parallel,a.mem_fraction)
    elif a.stage=='eval-cell':evaluate_cell(a.representation,a.level,batch=a.batch,threads=a.threads)
    elif a.stage=='evaluate-all':evaluate_all()
    elif a.stage=='finalize':finalize()


if __name__=='__main__':main()
