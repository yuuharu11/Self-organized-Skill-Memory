"""X1: image/history cross-attention motion-type retrieval on LIBERO-90 X tasks.

Stages are resumable: extract-lib40, cache-teachers, train, evaluate, report.
L90X rows are produced by l90x_rows.py.  The Eval action array is opened only
inside ``evaluate``; no training, fusion-weight selection, or checkpoint
selection receives Eval action data.
"""
from __future__ import annotations
import argparse, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
import image_query as iq
import image_query_gate as ig
import key_query_probe as kq
import scale_only_query as so
from unseen_paths import UNSEEN_ROWS

ROOT=Path(os.environ.get('SKILLNET_STATE_DIR',Path(__file__).resolve().parents[5]/'.skillnet-state'))/'outputs/memory-pilot/unseen-libero'
OUT=ROOT/'x-attn-01'; S1=ROOT/'scale-only-02'/'production'; K1=ROOT/'key-query-01'
PI=ROOT/'image-query-01'/'policy_features'; UNSEEN=Path(UNSEEN_ROWS)
POOL_DIR=S1/'prep'; SEED=4101; SEEDS=(4101,4102,4103)
# Training mixture required by X1 addendum. Model selection is separately
# balanced 50/50 between true and corrupted histories.
CORRUPT={'true':.4,'phase':.2,'swap':.15,'reverse':.1,'zero':.15}
SELECTION_CORRUPT={'true':.5,'phase':1/6,'swap':.125,'reverse':1/12,'zero':.125}
DETECT_CORRUPT={'swap':.25,'reverse':.25,'zero':.15,'phase':.35}
BASE_ARCHES=('P5','XH','XC','XA')
# The context variants use an actual within-episode image-feature difference,
# its magnitude, executed-motion/image-change ratio, and a predecessor flag.
ARCHES=BASE_ARCHES+tuple(f'{a}-imgdelta' for a in BASE_ARCHES)
LAYERS=('final','mid16','mid17','top3')
TEMPERATURES=(0.5,1.0,2.0)
def _plain_arch(arch):return arch[:-9] if str(arch).endswith('-imgdelta') else arch

def write_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=_json)+'\n');tmp.replace(p)
def _json(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if hasattr(x,'detach'):return x.detach().cpu().tolist()
    raise TypeError(type(x).__name__)
def log(out,**rec):
    rec={'time':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**rec};print(json.dumps(rec,ensure_ascii=False),flush=True)
    p=Path(out)/'progress.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:f.write(json.dumps(rec,ensure_ascii=False)+'\n')

def corrupt_history(hist,mode,*,seed=4101,task=None):
    """K1-compatible corruption, leaving the image/state untouched."""
    h=np.asarray(hist,np.float32).copy();rng=np.random.default_rng(seed)
    if mode=='true':return h
    if mode=='zero':h[:]=0;return h
    if mode=='reverse':h[:,:,:3]*=-1;return h
    if mode=='swap':
        t=np.zeros(len(h),np.int64) if task is None else np.asarray(task)
        for i in range(len(h)):
            choices=np.flatnonzero(t!=t[i])
            h[i]=hist[int(rng.choice(choices if len(choices) else np.arange(len(h))))]
        return h
    raise ValueError(mode)

def corruption_codes(n,*,seed,selection=False):
    # Fixed-seed iid assignment preserves the requested proportions in expectation
    # across the many training examples (rather than distorting small minibatches).
    distribution=SELECTION_CORRUPT if selection else CORRUPT
    names=np.array(tuple(distribution));p=np.array([distribution[x] for x in names])
    return np.random.default_rng(seed).choice(names,size=int(n),p=p)

def predecessor_indices(task,episode,frame):
    """Index each row's exact previous 10-frame window within its episode."""
    task=np.asarray(task);episode=np.asarray(episode);frame=np.asarray(frame)
    lookup={(int(t),int(e),int(f)):i for i,(t,e,f) in enumerate(zip(task,episode,frame))}
    return np.asarray([lookup.get((int(t),int(e),int(f)-10),-1)
                       for t,e,f in zip(task,episode,frame)],np.int64)

def _attach_predecessors(ds):
    ds['prev_index']=predecessor_indices(ds['task'],ds['episode'],ds['frame'])
    return ds

def phase_shift_history(ds,shift):
    """Return same-episode history at frame+shift and rows with a valid match."""
    hist=np.asarray(ds['hist'],np.float32);task=np.asarray(ds['task']);episode=np.asarray(ds['episode']);frame=np.asarray(ds['frame'])
    index={(int(t),int(e),int(f)):i for i,(t,e,f) in enumerate(zip(task,episode,frame))}
    src=np.asarray([index.get((int(t),int(e),int(f)+int(shift)),-1) for t,e,f in zip(task,episode,frame)],np.int64)
    valid=np.flatnonzero(src>=0);out=hist.copy();out[valid]=hist[src[valid]]
    return out,valid

def split_fit_select(is_select):
    s=np.asarray(is_select,bool);fit=np.flatnonzero(~s);sel=np.flatnonzero(s)
    if len(fit)==0 or len(sel)==0 or np.intersect1d(fit,sel).size:raise ValueError('empty/overlapping fit/select split')
    return fit,sel

def smoke_positions(task,episode,frame,n):
    """Choose short contiguous blocks from several episodes for phase-shift smoke."""
    task=np.asarray(task);episode=np.asarray(episode);frame=np.asarray(frame)
    groups=[]
    for key in np.unique(np.stack([task,episode],axis=1),axis=0):
        ix=np.flatnonzero((task==key[0])&(episode==key[1]));ix=ix[np.argsort(frame[ix])]
        if len(ix):groups.append(ix)
    if not groups:return np.empty(0,np.int64)
    ng=min(len(groups),max(1,int(n)//4));chosen=np.linspace(0,len(groups)-1,ng,dtype=np.int64)
    per=max(1,int(n)//ng);out=[]
    for gi in chosen:
        ix=groups[int(gi)];take=min(per,len(ix));start=max(0,(len(ix)-take)//2);out.extend(ix[start:start+take].tolist())
    return np.asarray(out[:int(n)],np.int64)

def assert_eval_boundary(dataset):
    """Training/select datasets must be explicitly non-Eval; target data is not accepted here."""
    if 'eval_actions' in dataset or 'target_eval' in dataset:raise ValueError('Eval target leakage into fit/select')
    if dataset.get('split')=='eval':raise ValueError('Eval rows cannot enter fit/select')

def _pool():
    global _POOL_CACHE
    if '_POOL_CACHE' in globals():return _POOL_CACHE
    p=dict(np.load(POOL_DIR/'W_pool.npz'));p['key']=so.key_inputs('W',p,'S-shared').astype(np.float32)
    p['phys']=np.asarray(p['phys'],np.float32)
    # Candidate phase uses source episode metadata, aligned to the exact W-pool order.
    import theta_regression as tr
    cp=tr.candidate_shape_pool(rep='C_vm_none');keep=~np.isin(cp['source_task'],so.HOLDOUT)
    if not np.array_equal(cp['source_task'][keep],p['task']):raise ValueError('W pool/source-row task alignment mismatch')
    _,rows,_,_=kq._production_data_splits(include_eval_targets=False);sr=cp['source_row'][keep]
    if np.any(rows['task'][sr]!=p['task']):raise ValueError('W pool source-row task mismatch')
    _POOL_CACHE=p;return p

def _load_row_dir(d):
    return {p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in Path(d).glob('*.npy')}

def _pool_image_final(tokens,groups,out_path,*,chunk=128):
    """Pool final-layer 16x16 grids to 8x8 for each of the two cameras."""
    tok=np.load(tokens,mmap_mode='r');grp=np.load(groups,mmap_mode='r');n=len(tok);dim=tok.shape[-1]
    out=np.lib.format.open_memmap(out_path,mode='r+' if Path(out_path).exists() else 'w+',dtype=np.float16,shape=(n,128,dim))
    for st in range(0,n,chunk):
        for j in range(st,min(st+chunk,n)):
            blocks=[]
            for label in (0,1):
                x=np.asarray(tok[j,np.asarray(grp[j])==label],np.float32)
                if len(x)!=256:raise ValueError(f'row {j} camera group {label} has {len(x)} tokens')
                x=x.reshape(16,16,dim).reshape(8,2,8,2,dim).mean((1,3)).reshape(64,dim)
                blocks.append(x)
            out[j]=np.concatenate(blocks).astype(np.float16)
        out.flush()
    return out

def extract_l90x(rows_dir,*,threads=8,batch=4):
    """Delegates row preparation to l90x_rows and stores compact image-token tensors."""
    import l90x_rows as lr
    return lr.extract(rows_dir,batch=batch,threads=threads)

def extract_lib40_final(out,*,batch=4,threads=8):
    """Extract final VLM tokens for LIBERO-40 fit rows only; reuse K1 mid/Expert/action caches."""
    import torch;torch.set_num_threads(int(threads));os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION','.25')
    d=Path(out)/'lib40';d.mkdir(parents=True,exist_ok=True)
    if (d/'complete.json').exists():return json.loads((d/'complete.json').read_text())
    base,rows,_,splits=kq._production_data_splits(include_eval_targets=False);ids=np.asarray(splits['train'][1],np.int64)
    # K1 train rows already exclude eval actions; validation rows are omitted here.
    import theta_regression as tr
    run=tr._module(tr.PI05/'run.py','x1_lib40_run');model,_=run.load_policy(tr.CHECKPOINT)
    from openpi.shared.nnx_utils import module_jit
    prefix=module_jit(model.prefix_features);n=len(ids)
    tok=np.lib.format.open_memmap(d/'tokens.npy',mode='r+' if (d/'tokens.npy').exists() else 'w+',dtype=np.float16,shape=(n,712,2048))
    grp=np.lib.format.open_memmap(d/'groups.npy',mode='r+' if (d/'groups.npy').exists() else 'w+',dtype=np.int8,shape=(n,712))
    prog=d/'progress.json';done=int(json.loads(prog.read_text())['done']) if prog.exists() else 0
    for st in range(done,n,int(batch)):
        ix=ids[st:min(n,st+int(batch))];obs=base.obs_batch(rows,ix);h,g=prefix(obs)
        tok[st:st+len(ix)]=np.asarray(h,np.float16);grp[st:st+len(ix)]=np.asarray(g,np.int8);tok.flush();grp.flush()
        done=st+len(ix);write_json(prog,{'done':done,'total':n,'complete':done==n})
    _make_final_image_cache(d,d)
    write_json(d/'complete.json',{'rows':n,'source':'LIBERO-40 non-validation rows only','target_actions_used_for_features':False})
    return json.loads((d/'complete.json').read_text())

def _make_final_image_cache(rows_dir,feature_dir):
    fd=Path(feature_dir);fd.mkdir(parents=True,exist_ok=True)
    p=fd/'image_final.npy'
    if p.exists():return np.load(p,mmap_mode='r')
    a=_pool_image_final(Path(rows_dir)/'tokens.npy',Path(rows_dir)/'groups.npy',p)
    return a

def _mid_cache_from_l90x(rows_dir,layer):
    d=Path(rows_dir)/'features';x=np.load(d/'mid_v.npy',mmap_mode='r')
    ix={'mid16':0,'mid17':1}[layer]
    return x[:,ix,:128]

def _mid_cache_lib40(layer):
    p=K1/'policy_features'/'train_mid_v.npy';x=np.load(p,mmap_mode='r');ix={'mid16':4,'mid17':5}[layer]
    return x[:,ix,:128]

def prepare_l90x_features(rows_dir):
    """Create compact final image tokens; raw VLM tokens are retained for audit."""
    return _make_final_image_cache(Path(rows_dir)/'features',Path(rows_dir)/'features')

def _error_matrix(actions,pool,*,chunk=256):
    _,vs,_=so._mods();a=np.asarray(actions,np.float32)[:,:,:6];phys=vs.to_physical(a).astype(np.float32)
    out=np.empty((len(a),len(pool['phys'])),np.float32)
    for st in range(0,len(a),chunk):out[st:st+chunk]=so.scale_only_error(phys[st:st+chunk],pool['phys'],'S-shared')
    return out

def _teacher(err,allowed):
    tau=so.teacher_tau(err,allowed,1/10);logits=np.where(allowed,-np.asarray(err,np.float64)/max(tau,1e-8),-np.inf)
    logits-=np.max(logits,axis=1,keepdims=True);w=np.exp(logits);w[~allowed]=0;w/=np.maximum(w.sum(1,keepdims=True),1e-30)
    return w.astype(np.float32),float(tau)

def masked_training_loss(logits,teacher,errors,allowed):
    """Listwise + ordinal loss normalized over each query's allowed pool only."""
    import torch
    allowed=allowed.bool()
    if logits.shape!=teacher.shape or logits.shape!=errors.shape or logits.shape!=allowed.shape:
        raise ValueError('logits, teacher, errors, and allowed masks must have equal shapes')
    if torch.any(allowed.sum(1)==0):raise ValueError('each training query needs an allowed candidate')
    if torch.any(teacher.masked_select(~allowed)!=0):raise ValueError('teacher assigns mass to a disallowed candidate')
    masked=logits.masked_fill(~allowed,-1e9)
    ce=-(teacher*torch.log_softmax(masked,dim=-1)).sum(-1).mean()
    corr=ig.pearson_t(logits,-errors,allowed).mean()
    return ce+.25*(1-corr)

def _ensure_teacher_cache(out):
    out=Path(out);pdir=out/'teachers';pdir.mkdir(parents=True,exist_ok=True);pool=_pool()
    # L90X split; selection labels are allowed for checkpoint selection, while Eval rows are not in this cache.
    rd=out/'l90x_rows';feat=rd/'features';r=_load_row_dir(rd);sel=np.asarray(r['is_select'],bool)
    for split,ids in (('fit',np.flatnonzero(~sel)),('select',np.flatnonzero(sel))):
        p=pdir/f'l90x_{split}.npz'
        if p.exists():continue
        err=_error_matrix(np.asarray(r['actions'][ids]),pool);task=np.asarray(r['task'][ids],np.int64)
        allow=np.ones_like(err,bool);teach,tau=_teacher(err,allow)
        np.savez_compressed(p,ids=ids,err=err,allowed=allow,teacher=teach,tau=np.float32(tau))
    # LIBERO-40 existing K1 training rows and teacher target; explicitly cross-task mask.
    p=pdir/'lib40_fit.npz'
    if not p.exists():
        err=np.asarray(np.load(POOL_DIR/'W_err_S-shared_train.npz')['err'],np.float32)
        inp=dict(np.load(POOL_DIR/'W_inputs_train.npz'));task=np.asarray(inp['task'],np.int64)
        allow=so.allowed_matrix(task,pool['task'],pool['keep'],True);teach,tau=_teacher(err,allow)
        np.savez_compressed(p,ids=np.arange(len(err),dtype=np.int64),err=err,allowed=allow,teacher=teach,tau=np.float32(tau),task=task)
    return pool

def _lib40_data(out,*,need_image=True):
    pdir=Path(out)/'lib40';
    if need_image and not (pdir/'image_final.npy').exists():raise FileNotFoundError('run extract-lib40 first')
    inp=dict(np.load(POOL_DIR/'W_inputs_train.npz'));N=len(inp['task'])
    base={'hist':np.asarray(inp['hist'],np.float32),'state':np.asarray(inp['state'][:,:8],np.float32),
        'task':np.asarray(inp['task'],np.int64),'episode':np.asarray(inp['episode'],np.int64),
        'frame':np.asarray(inp['frame'],np.int64),'actions':np.load(K1/'policy_features/train_actions.npy',mmap_mode='r'),
        'image':np.load(pdir/'image_final.npy',mmap_mode='r'),
        'pi05':np.load(K1/'policy_features/train_actions.npy',mmap_mode='r'),
        'expert':np.load(K1/'policy_features/train_hidden_t1.npy',mmap_mode='r')}
    for layer in ('mid16','mid17'):base[layer]=_mid_cache_lib40(layer)
    if len(base['hist'])!=N:raise ValueError('LIBERO-40 row alignment mismatch')
    return _attach_predecessors(base)

def _l90x_data(out):
    rd=Path(out)/'l90x_rows';r=_load_row_dir(rd);f=rd/'features'
    if not (f/'complete.json').exists():raise FileNotFoundError('run l90x_rows.py extract first')
    image=prepare_l90x_features(rd)
    task=np.asarray(r['task'],np.int64);episode=np.asarray(r['episode'],np.int64);frame=np.asarray(r['frame'],np.int64)
    return _attach_predecessors({'hist':np.asarray(r['hist'],np.float32),'state':np.asarray(r['state'][:,:8],np.float32),
        'task':task,'episode':episode,'frame':frame,
        'is_select':np.asarray(r['is_select'],bool),'actions':np.load(rd/'actions.npy',mmap_mode='r'),
        'image':image,'pi05':np.load(f/'pi05.npy',mmap_mode='r'),'expert':np.load(f/'expert_t1.npy',mmap_mode='r'),
        'mid16':_mid_cache_from_l90x(rd,'mid16'),'mid17':_mid_cache_from_l90x(rd,'mid17')})

def _feature(ds,idx,layer,history=None,arch='XH',device='cpu'):
    import torch
    h=ds['hist'][idx] if history is None else history
    base_arch=_plain_arch(arch)
    def reduce_views(v):
        v=np.asarray(v,np.float32)[:,:128]
        # Two 8x8 camera grids -> two 4x4 grids.
        if v.shape[1]==128:
            d=v.shape[-1];v=v.reshape(len(v),2,8,8,d).reshape(len(v),2,4,2,4,2,d).mean((3,5)).reshape(len(v),32,d)
        return v
    if base_arch=='P5' and not arch.endswith('-imgdelta'):
        vis=np.zeros((len(idx),32,1),np.float32);delta_vis=np.zeros_like(vis)
    elif layer=='top3':
        # Concatenate the three preregistered image readouts (final, V16, V17)
        # after giving each the same 4x4-per-camera spatial support.
        sources=[('image','image'),('mid16','mid16'),('mid17','mid17')]
        currents=[];deltas=[]
        for name,key in sources:
            cur=reduce_views(ds[key][idx]);currents.append(cur)
            if 'prev_index' in ds:
                pi=np.asarray(ds['prev_index'][idx]);valid=pi>=0;prev=np.zeros_like(np.asarray(ds[key][idx],np.float32))
                if valid.any():prev[valid]=np.asarray(ds[key][pi[valid]],np.float32)
                prev=reduce_views(prev);delta=cur-prev;delta[~valid]=0
            else:delta=np.zeros_like(cur)
            deltas.append(delta)
        vis=np.concatenate(currents,axis=-1);delta_vis=np.concatenate(deltas,axis=-1)
    else:
        vis=reduce_views(ds[layer if layer!='final' else 'image'][idx])
        key=layer if layer!='final' else 'image'
        if 'prev_index' in ds:
            pi=np.asarray(ds['prev_index'][idx]);valid=pi>=0;prev=np.zeros_like(np.asarray(ds[key][idx],np.float32))
            if valid.any():prev[valid]=np.asarray(ds[key][pi[valid]],np.float32)
            delta_vis=vis-reduce_views(prev);delta_vis[~valid]=0
        else:delta_vis=np.zeros_like(vis)
    pi=np.asarray(ds.get('prev_index',np.full(len(ds['hist']),-1))[idx]);has_prev=(pi>=0).astype(np.float32)
    hist_motion=np.linalg.norm(np.asarray(h,np.float32)[...,:6],axis=(1,2))
    image_delta_norm=np.linalg.norm(delta_vis,axis=(1,2))
    # Relative visual change per unit of recent executed motion. Keep no-image
    # history exactly zero and distinguish it with the explicit predecessor bit.
    ratio=image_delta_norm/np.maximum(hist_motion,1e-6)
    context=np.stack([np.log1p(hist_motion),np.log1p(image_delta_norm),np.log1p(ratio),has_prev],axis=-1).astype(np.float32)
    result={'hist':torch.as_tensor(np.array(h,copy=True),dtype=torch.float32,device=device),
            'state':torch.as_tensor(np.array(ds['state'][idx],copy=True),dtype=torch.float32,device=device),
            'visual':torch.as_tensor(np.array(vis,copy=True),dtype=torch.float32,device=device),
            'visual_delta':torch.as_tensor(np.array(delta_vis,copy=True),dtype=torch.float32,device=device),
            'context':torch.as_tensor(context,dtype=torch.float32,device=device)}
    if base_arch=='XA':
        result['pi05']=torch.as_tensor(np.asarray(ds['pi05'][idx],np.float32),device=device)
        result['expert']=torch.as_tensor(np.asarray(ds['expert'][idx],np.float32),device=device)
    return result

def _token_dim(ds,layer):
    if layer=='final':return int(ds['image'].shape[-1])
    if layer=='top3':return int(ds['image'].shape[-1]+ds['mid16'].shape[-1]+ds['mid17'].shape[-1])
    return int(ds[layer].shape[-1])

def make_model(arch,in_dim,key_dim,token_dim,expert_dim=1024,d=256):
    import torch,torch.nn as nn
    F=nn.functional
    base_arch=_plain_arch(arch);use_imgdelta=arch.endswith('-imgdelta')
    class Cross(nn.Module):
        def __init__(self):
            super().__init__();self.attn=nn.MultiheadAttention(d,8,batch_first=True,dropout=.05);self.n1=nn.LayerNorm(d)
            self.ff=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d));self.n2=nn.LayerNorm(d)
        def forward(self,q,kv):
            a,_=self.attn(q,kv,kv,need_weights=False);q=self.n1(q+a);return self.n2(q+self.ff(q))
    class Net(nn.Module):
        def __init__(self):
            super().__init__();self.arch=arch;self.use_imgdelta=use_imgdelta
            self.key=nn.Sequential(nn.Linear(key_dim,512),nn.GELU(),nn.Linear(512,d))
            self.hist_step=nn.Linear(7,d);self.state=nn.Sequential(nn.Linear(8,128),nn.GELU(),nn.Linear(128,d))
            self.hist_global=nn.Sequential(nn.Linear(70+8,512),nn.GELU(),nn.Linear(512,d))
            self.vis=nn.Sequential(nn.Linear(token_dim,d),nn.LayerNorm(d))
            self.delta_vis=nn.Sequential(nn.Linear(token_dim,d),nn.LayerNorm(d))
            self.act=nn.Linear(7,d);self.expert=nn.Sequential(nn.Linear(expert_dim,d),nn.LayerNorm(d))
            self.cross1=Cross();self.cross2=Cross();self.qgen=nn.Sequential(nn.Linear(2*d,d*2),nn.GELU(),nn.Linear(d*2,4*d))
            self.out=nn.Sequential(nn.Linear(3*d,512),nn.GELU(),nn.Linear(512,1));self.log_scale=nn.Parameter(torch.tensor(np.log(10.),dtype=torch.float32))
            self.p5=nn.Sequential(nn.Linear(78+(4 if use_imgdelta else 0),512),nn.GELU(),nn.Linear(512,d))
        def forward(self,feat,candidates):
            keys=F.normalize(self.key(candidates),dim=-1);hg=self.hist_global(torch.cat([feat['hist'].flatten(1),feat['state']],-1))
            if base_arch=='P5':
                pieces=[feat['hist'].flatten(1),feat['state']]
                if use_imgdelta:pieces.append(feat['context'])
                return F.normalize(self.p5(torch.cat(pieces,-1)),dim=-1)@keys.T*self.log_scale.exp()
            if base_arch=='XH':
                q=self.qgen(torch.cat([hg, self.state(feat['state'])],-1)).reshape(len(hg),4,-1)
                visual=self.vis(feat['visual'])
                if use_imgdelta:visual=torch.cat([visual,self.delta_vis(feat['visual_delta'])],dim=1)
                q=self.cross2(self.cross1(q,visual),visual).mean(1)
                return F.normalize(q+hg,dim=-1)@keys.T*self.log_scale.exp()
            kv=[self.vis(feat['visual']),self.hist_step(feat['hist']),self.state(feat['state'])[:,None]]
            if use_imgdelta:kv.insert(1,self.delta_vis(feat['visual_delta']))
            if base_arch=='XA':kv += [self.act(feat['pi05']),self.expert(feat['expert'])]
            vals=torch.cat(kv,dim=1);cand=keys[None].expand(len(hg),-1,-1)
            ctx=self.cross1(cand,vals);ctx=self.cross2(ctx,vals);h=hg[:,None].expand(-1,len(keys),-1)
            return self.out(torch.cat([ctx,cand,h],-1)).squeeze(-1)*self.log_scale.exp()
    return Net()

def _load_state_compat(net,state):
    """Load current checkpoints and ignore the obsolete scalar phase bias."""
    result=net.load_state_dict(state,strict=False)
    missing=set(result.missing_keys);unexpected=set(result.unexpected_keys)
    if unexpected-{'phase_scale'} or missing:
        raise RuntimeError(f'checkpoint mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}')

def _score_net(net,ds,ids,layer,arch,pool_key,*,batch=8,history_override=None,device='cuda'):
    import torch
    nets=[];net.eval();kt=torch.as_tensor(pool_key,dtype=torch.float32,device=device)
    with torch.no_grad():
        for st in range(0,len(ids),batch):
            ii=np.asarray(ids[st:st+batch],np.int64);hh=None if history_override is None else history_override[st:st+batch]
            feat=_feature(ds,ii,layer,hh,arch,device);nets.append(net(feat,kt).cpu().numpy())
    return np.concatenate(nets)

def _zscore_rows(scores):
    x=np.asarray(scores,np.float32);return (x-x.mean(1,keepdims=True))/(x.std(1,keepdims=True)+1e-6)

def _rank_summary(scores):
    if hasattr(scores,'detach'):scores=scores.detach().cpu().numpy()
    x=_zscore_rows(scores);order=np.sort(x,axis=1)[:,::-1]
    p=np.exp(x-x.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
    entropy=-(p*np.log(np.maximum(p,1e-9))).sum(1)
    return np.stack([order[:,0],order[:,0]-order[:,1],x.std(1),entropy],axis=1)

def _fusion_gate_features(p5,nn,feat):
    import torch
    p=_rank_summary(p5);n=_rank_summary(nn)
    ctx=feat.get('context')
    if ctx is None:ctx=torch.zeros((len(p),4),device=feat['hist'].device)
    hmotion=torch.sqrt(torch.mean(feat['hist'][...,:6]**2,dim=(1,2))+1e-12)[:,None]
    snorm=torch.sqrt(torch.mean(feat['state']**2,dim=1)+1e-12)[:,None]
    base=torch.cat([torch.as_tensor(p,device=feat['hist'].device),torch.as_tensor(n,device=feat['hist'].device),ctx,torch.log1p(hmotion),torch.log1p(snorm)],dim=1)
    # Return the raw temporal token differences separately; the gate learns an
    # attention pooling over them instead of receiving only scalar summaries.
    return base,feat.get('visual_delta')

def make_fusion_gate(in_dim=14,delta_dim=2048):
    import torch.nn as nn
    class Gate(nn.Module):
        def __init__(self):
            super().__init__();self.delta=nn.Sequential(nn.Linear(delta_dim,64),nn.GELU())
            self.attn=nn.Linear(64,1);self.out=nn.Sequential(nn.Linear(in_dim+64,32),nn.GELU(),nn.Linear(32,1))
        def forward(self,x,delta=None):
            if delta is None:delta=x.new_zeros((len(x),1,delta_dim))
            z=self.delta(delta);w=torch.softmax(self.attn(z),dim=1);pooled=(z*w).sum(1)
            return self.out(torch.cat([x,pooled],dim=1))
    import torch
    return Gate()

def combine_with_gate(gate,p5,nn,feat):
    import torch
    x,delta=_fusion_gate_features(p5,nn,feat);alpha=torch.sigmoid(gate(x,delta))
    return alpha*_zscore_rows_torch(p5)+(1-alpha)*_zscore_rows_torch(nn),alpha.squeeze(-1)

def _zscore_rows_torch(x):
    return (x-x.mean(1,keepdim=True))/(x.std(1,keepdim=True,unbiased=False)+1e-6)

def train_fusion_gate(out,source,*,p5_temperature=1.0,seeds=SEEDS,seed=SEED,epochs=20,batch=64,device='cuda'):
    """Fit a query-conditioned P5/π0.5-NN gate on L90X fit only."""
    import torch
    torch.set_num_threads(8);torch.manual_seed(int(seed));out=Path(out);pool=_pool();data=_l90x_data(out)
    fit=np.flatnonzero(~data['is_select']);sel=np.flatnonzero(data['is_select'])
    te=np.load(out/'teachers'/'l90x_fit.npz');ve=np.load(out/'teachers'/'l90x_select.npz')
    fit_hist,modes=_mixed_history(data,fit,seed+100)
    p5=[];temp_tag=f'{float(p5_temperature):g}'.replace('.','p')
    for s in seeds:
        ck=torch.load(out/'models'/f'{source}_P5_final_t{temp_tag}_s{s}.pt',weights_only=False)
        net=make_model('P5',78,pool['key'].shape[1],2048).to(device);_load_state_compat(net,ck['state']);net.eval()
        p5.append(_score_net(net,data,fit,'final','P5',pool['key'],batch=batch,history_override=fit_hist,device=device))
    p5fit=np.mean(p5,axis=0);p5s=[]
    for s in seeds:
        ck=torch.load(out/'models'/f'{source}_P5_final_t{temp_tag}_s{s}.pt',weights_only=False)
        net=make_model('P5',78,pool['key'].shape[1],2048).to(device);_load_state_compat(net,ck['state']);net.eval()
        p5s.append(_score_net(net,data,sel,'final','P5',pool['key'],batch=batch,history_override=_mixed_history(data,sel,SEED+9000,selection=True)[0],device=device))
    p5sel=np.mean(p5s,axis=0);nnfit=-_teacher_nn(np.asarray(data['pi05'][fit]),pool);nnsel=-_teacher_nn(np.asarray(data['pi05'][sel]),pool)
    sel_hist,_=_mixed_history(data,sel,SEED+9000,selection=True)
    # Score-side features reflect the same corrupted history seen by P5.
    ffit=_feature(data,fit,'final',fit_hist,'P5-imgdelta',device)
    fsel=_feature(data,sel,'final',sel_hist,'P5-imgdelta',device)
    xf,df=_fusion_gate_features(p5fit,nnfit,ffit);xs,ds=_fusion_gate_features(p5sel,nnsel,fsel)
    gate=make_fusion_gate(xf.shape[1],df.shape[-1]).to(device);opt=torch.optim.AdamW(gate.parameters(),lr=3e-3,weight_decay=1e-3)
    target=torch.as_tensor(te['teacher'],device=device);allowed=torch.as_tensor(te['allowed'],dtype=torch.bool,device=device)
    err=torch.as_tensor(te['err'],device=device);xft=torch.as_tensor(xf,dtype=torch.float32,device=device)
    vst=torch.as_tensor(ve['teacher'],device=device);va=torch.as_tensor(ve['allowed'],dtype=torch.bool,device=device);verr=torch.as_tensor(ve['err'],device=device)
    best=(np.inf,None,0);curve=[];rng=np.random.default_rng(seed)
    for ep in range(int(epochs)):
        gate.train();order=rng.permutation(len(fit))
        for st in range(0,len(order),batch):
            ii=order[st:st+batch];alpha=torch.sigmoid(gate(xft[ii],torch.as_tensor(df[ii],device=device)));scores=alpha*_zscore_rows_torch(torch.as_tensor(p5fit[ii],device=device))+(1-alpha)*_zscore_rows_torch(torch.as_tensor(nnfit[ii],device=device))
            loss=masked_training_loss(scores,target[ii],err[ii],allowed[ii])
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        gate.eval()
        with torch.no_grad():
            alpha=torch.sigmoid(gate(torch.as_tensor(xs,dtype=torch.float32,device=device),torch.as_tensor(ds,device=device)))
            score=alpha*_zscore_rows_torch(torch.as_tensor(p5sel,device=device))+(1-alpha)*_zscore_rows_torch(torch.as_tensor(nnsel,device=device))
            pred=score.argmax(1).cpu().numpy();metric=float(np.mean(ve['err'][np.arange(len(sel)),pred]))
        curve.append({'epoch':ep+1,'selection_top1_type_error':metric,'mean_alpha':float(alpha.mean().cpu())})
        if metric<best[0]:best=(metric,{k:v.detach().cpu().clone() for k,v in gate.state_dict().items()},ep+1)
    gate_path=out/'models'/f'{source}_fusion_gate.pt';gate_path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'state':best[1],'in_dim':int(xf.shape[1]),'delta_dim':int(df.shape[-1]),'best_epoch':best[2],'selection_top1_type_error':best[0],
        'curve':curve,'fit_tasks':sorted(np.unique(data['task'][fit]).tolist()),'selection_tasks':sorted(np.unique(data['task'][sel]).tolist()),
        'eval_actions_used':False,'alpha_dynamic_per_query':True},gate_path)
    return {'best_epoch':best[2],'best_selection_top1_type_error':best[0],'curve':curve}

def _spearman_rows(sc,err):
    from scipy.stats import rankdata
    return np.asarray([np.corrcoef(rankdata(sc[i]),rankdata(-err[i]))[0,1] for i in range(len(sc))],np.float64)

def _conditional_spearman_rows(sc,err,allowed=None,frac=.1):
    """Spearman restricted to each row's best frac of eligible motion types."""
    from scipy.stats import rankdata
    sc=np.asarray(sc);err=np.asarray(err);allow=np.ones_like(err,bool) if allowed is None else np.asarray(allowed,bool);out=[]
    for s,e,a in zip(sc,err,allow):
        ids=np.flatnonzero(a)
        if len(ids)<2:out.append(0.);continue
        count=max(1,int(np.ceil(len(ids)*frac)));best=ids[np.argsort(e[ids],kind='stable')[:count]]
        if np.ptp(s[best])<1e-12 or np.ptp(e[best])<1e-12:out.append(0.);continue
        v=np.corrcoef(rankdata(s[best]),rankdata(-e[best]))[0,1];out.append(float(v) if np.isfinite(v) else 0.)
    return np.asarray(out,np.float64)

def _pearson_rows(scores,errors,allowed=None,frac=None):
    """Pearson(score, -log(error+eps)); constant rows score 0 and are counted."""
    s=np.asarray(scores,np.float64);e=np.asarray(errors,np.float64)
    allow=np.ones_like(e,bool) if allowed is None else np.asarray(allowed,bool);out=[];valid=[]
    for sr,er,ar in zip(s,e,allow):
        ids=np.flatnonzero(ar)
        if frac is not None and len(ids):ids=ids[np.argsort(er[ids],kind='stable')[:max(1,int(np.ceil(frac*len(ids))))]]
        if len(ids)<2:out.append(0.);valid.append(False);continue
        x=sr[ids];y=-np.log(np.maximum(er[ids],0)+1e-6)
        if np.ptp(x)<1e-12 or np.ptp(y)<1e-12:out.append(0.);valid.append(False);continue
        v=np.corrcoef(x,y)[0,1];ok=bool(np.isfinite(v));out.append(float(v) if ok else 0.);valid.append(ok)
    return np.asarray(out,np.float64),np.asarray(valid,bool)

def _retrieval_row_metrics(scores,errors,allowed=None):
    """Top-focused ranking metrics following image_query_metrics.py."""
    s=np.asarray(scores);e=np.asarray(errors);a=np.ones_like(e,bool) if allowed is None else np.asarray(allowed,bool)
    result={k:[] for k in ('conditional_spearman','conditional_spearman_valid','pearson_top10','pearson_all','pearson_top10_valid','pearson_all_valid',
                           'recall@1','recall@5','recall@20','ndcg@10','gap@1','gap@5')}
    for sr,er,ar in zip(s,e,a):
        ids=np.flatnonzero(ar)
        if not len(ids):continue
        ss=sr[ids];ee=er[ids];order=np.argsort(-ss,kind='stable');oracle=float(ee.min());random=float(ee.mean());gap=max(random-oracle,1e-9)
        nbest=max(1,int(np.ceil(.1*len(ids))));best=ids[np.argsort(ee,kind='stable')[:nbest]]
        from scipy.stats import rankdata
        valid_c=len(best)>=2 and np.ptp(sr[best])>=1e-12 and np.ptp(er[best])>=1e-12
        c=0. if not valid_c else np.corrcoef(rankdata(sr[best]),rankdata(-er[best]))[0,1]
        result['conditional_spearman'].append(float(c) if np.isfinite(c) else 0.)
        result['conditional_spearman_valid'].append(float(valid_c and np.isfinite(c)))
        p10,v10=_pearson_rows(sr[None,:],ee[None,:],ar[None,:],.1);pa,va=_pearson_rows(sr[None,:],ee[None,:],ar[None,:],None)
        result['pearson_top10'].append(float(p10[0]));result['pearson_all'].append(float(pa[0]))
        result['pearson_top10_valid'].append(float(v10[0]));result['pearson_all_valid'].append(float(va[0]))
        tau=max(.1*gap,1e-9);rel=np.exp(-(ee-oracle)/tau);ideal=np.sort(rel)[::-1][:10];disc=1/np.log2(np.arange(2,len(ideal)+2));den=float(np.sum(ideal*disc))
        good=ee<=oracle+.1*gap
        for k in (1,5,20):result[f'recall@{k}'].append(float(good[order[:k]].any()))
        result['ndcg@10'].append(float(np.sum(rel[order[:10]]*disc)/max(den,1e-12)))
        for k in (1,5):result[f'gap@{k}'].append(float((random-ee[order[:k]].min())/gap))
    return {k:np.asarray(v,np.float64) for k,v in result.items()}

def _episode_bootstrap_diff(a,b,episode,*,seed=4101,reps=2000):
    a=np.asarray(a,np.float64);b=np.asarray(b,np.float64);ep=np.asarray(episode)
    groups=np.unique(ep);rng=np.random.default_rng(seed);diff=a-b;vals=np.empty(reps,np.float64)
    members=[np.flatnonzero(ep==g) for g in groups]
    for i in range(reps):
        chosen=rng.integers(len(groups),size=len(groups));ix=np.concatenate([members[j] for j in chosen]);vals[i]=diff[ix].mean()
    return {'mean':float(diff.mean()),'ci95':[float(x) for x in np.quantile(vals,[.025,.975])],
        'episodes':int(len(groups)),'resamples':int(reps)}

def _mixed_history(ds,ids,seed,*,selection=False):
    modes=corruption_codes(len(ids),seed=seed,selection=selection);base=np.asarray(ds['hist'][ids],np.float32);out=base.copy();task=np.asarray(ds['task'][ids])
    for m in ('phase','swap','reverse','zero'):
        ix=np.flatnonzero(modes==m)
        if len(ix):
            if m=='phase':
                full_hist=np.asarray(ds['hist'],np.float32);full_task=np.asarray(ds['task']);full_episode=np.asarray(ds['episode']);full_frame=np.asarray(ds['frame'])
                lookup={(int(t),int(e),int(f)):j for j,(t,e,f) in enumerate(zip(full_task,full_episode,full_frame))}
                rng=np.random.default_rng(seed+31)
                for local_i in ix:
                    row=int(ids[local_i]);t=int(full_task[row]);e=int(full_episode[row]);f=int(full_frame[row])
                    candidates=[lookup[(t,e,f+shift)] for shift in (-20,-10,10,20) if (t,e,f+shift) in lookup]
                    if candidates:out[local_i]=full_hist[int(rng.choice(candidates))]
                    else:
                        # Preserve the true/corrupted balance. A missing phase
                        # neighbor becomes an explicit zero corruption instead
                        # of being silently relabeled as a true history.
                        out[local_i]=0;modes[local_i]='zero'
            elif m=='swap':out[ix]=corrupt_history(base,'swap',seed=seed+17,task=task)[ix]
            else:out[ix]=corrupt_history(base[ix],m,seed=seed+17)
    return out,modes

def _corrupt_histories_by_code(ds,ids,codes,seed,*,swap_pool_ids=None):
    """Create corruptions without crossing the caller's task split.

    A requested phase shift with no same-episode neighbor is recorded as zero,
    rather than silently labeling a zero history as a phase-shift example.
    """
    ids=np.asarray(ids,np.int64);codes=np.asarray(codes);out=np.asarray(ds['hist'][ids],np.float32).copy()
    task=np.asarray(ds['task']);episode=np.asarray(ds['episode']);frame=np.asarray(ds['frame'])
    pool_ids=ids if swap_pool_ids is None else np.asarray(swap_pool_ids,np.int64)
    lookup={(int(task[j]),int(episode[j]),int(frame[j])):int(j) for j in range(len(task))};rng=np.random.default_rng(seed)
    applied=codes.astype('U16',copy=True)
    for j,(row,code) in enumerate(zip(ids,codes)):
        if code=='true':continue
        if code=='swap':
            options=pool_ids[task[pool_ids]!=task[row]]
            if len(options):out[j]=ds['hist'][int(rng.choice(options))]
            else:out[j]=0;applied[j]='zero'
        elif code=='reverse':out[j,:,:3]*=-1
        elif code=='zero':out[j]=0
        elif code=='phase' or code.startswith('phase'):
            shifts=(-20,-10,10,20) if code=='phase' else (int(code.replace('phase','')) ,)
            options=[lookup[(int(task[row]),int(episode[row]),int(frame[row])+shift)]
                     for shift in shifts if (int(task[row]),int(episode[row]),int(frame[row])+shift) in lookup]
            if options:out[j]=ds['hist'][int(rng.choice(options))]
            else:out[j]=0;applied[j]='zero'
        else:raise ValueError(f'unknown history code {code}')
    return out,applied

def _consistency_features(ds,ids,hist,kind):
    """Query-side evidence for deciding whether execution history is inconsistent."""
    ids=np.asarray(ids,np.int64);hist=np.asarray(hist,np.float32)
    if kind=='vlm_mean':visual=np.asarray(ds['image'][ids],np.float32).mean(1)
    elif kind=='mid17':visual=np.asarray(ds['mid17'][ids],np.float32).mean(1)
    elif kind=='pi05':visual=np.asarray(ds['pi05'][ids],np.float32).reshape(len(ids),-1)
    else:raise ValueError(kind)
    state=np.asarray(ds['state'][ids],np.float32);action=np.asarray(ds['pi05'][ids],np.float32)
    agree=np.asarray(ig.agreement(hist,action),np.float32)
    return np.concatenate([visual,state,hist.reshape(len(ids),-1),agree],axis=1).astype(np.float32)

def _consistency_examples(ds,ids,kind,seed,*,selection=False):
    ids=np.asarray(ids,np.int64);rng=np.random.default_rng(seed)
    hist_true=np.asarray(ds['hist'][ids],np.float32)
    if selection:
        hist_bad,modes=_mixed_history(ds,ids,seed,selection=True);bad=np.asarray(modes)!='true'
        hist_bad=hist_bad
        actual_modes=np.asarray(modes,dtype='U16').copy()
    else:
        modes=rng.choice(tuple(DETECT_CORRUPT),size=len(ids),p=list(DETECT_CORRUPT.values()))
        ph=np.flatnonzero(modes=='phase')
        if len(ph):modes[ph]=rng.choice(['phase-10','phase+10','phase-20','phase+20'],size=len(ph))
        hist_bad,actual_modes=_corrupt_histories_by_code(ds,ids,modes,seed+71,swap_pool_ids=ids)
        bad=np.ones(len(ids),bool)
    xt=_consistency_features(ds,ids,hist_true,kind)
    xb=_consistency_features(ds,ids[bad],hist_bad[bad],kind)
    # Balance intact and corrupted examples; the validation mixture is also
    # defined as 50% intact / 50% corrupted.
    if len(xb)<len(xt):
        take=np.random.default_rng(seed+901).choice(len(xt),size=len(xb),replace=False);xt=xt[take]
    x=np.concatenate([xt,xb],axis=0);y=np.concatenate([np.zeros(len(xt),np.float32),np.ones(len(xb),np.float32)])
    modes_out=np.concatenate([np.full(len(xt),'true',dtype='U16'),np.asarray(actual_modes[bad],dtype='U16')])
    return x,y,modes_out

def _auc(y,p):
    from sklearn.metrics import roc_auc_score
    y=np.asarray(y);p=np.asarray(p)
    return float(roc_auc_score(y,p)) if len(np.unique(y))==2 else float('nan')

def make_consistency_model(in_dim):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(in_dim,512),nn.GELU(),nn.LayerNorm(512),nn.Linear(512,128),nn.GELU(),nn.Linear(128,1))

def train_consistency_model(out,source,kind,*,epochs=20,batch=128,seed=SEED,device='cuda'):
    """Train corruption detector; only L90X selection tasks choose its epoch."""
    import torch
    torch.set_num_threads(8);torch.manual_seed(int(seed));out=Path(out);l90=_l90x_data(out)
    fit90=np.flatnonzero(~l90['is_select']);sel=np.flatnonzero(l90['is_select'])
    if source=='l90x':train_sets=[(l90,fit90)]
    elif source=='lib40':train_sets=[(_lib40_data(out),np.arange(len(_lib40_data(out)['task'])))]
    elif source=='both':
        d40=_lib40_data(out);train_sets=[(l90,fit90),(d40,np.arange(len(d40['task'])))]
    else:raise ValueError(source)
    xs=[]
    for i,(ds,ids) in enumerate(train_sets):xs.append(_consistency_examples(ds,ids,kind,seed+i*101))
    x=np.concatenate([a[0] for a in xs]);y=np.concatenate([a[1] for a in xs])
    corruption_counts={str(name):int(sum(np.sum(a[2]==name) for a in xs)) for name in ('true','swap','reverse','zero','phase-10','phase+10','phase-20','phase+20','phase')}
    mu=x.mean(0);sd=x.std(0)+1e-5;x=((x-mu)/sd).astype(np.float32)
    xv,yv,mv=_consistency_examples(l90,sel,kind,SEED+9000,selection=True)
    xv=((xv-mu)/sd).astype(np.float32)
    net=make_consistency_model(x.shape[1]).to(device);opt=torch.optim.AdamW(net.parameters(),lr=5e-4,weight_decay=1e-4)
    xt=torch.as_tensor(x,device=device);yt=torch.as_tensor(y[:,None],device=device);vxt=torch.as_tensor(xv,device=device)
    rng=np.random.default_rng(seed);best=(-np.inf,None,0);curve=[]
    for ep in range(int(epochs)):
        net.train();order=rng.permutation(len(x))
        for st in range(0,len(order),batch):
            ii=order[st:st+batch];logit=net(xt[ii]);loss=torch.nn.functional.binary_cross_entropy_with_logits(logit,yt[ii])
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
        net.eval()
        with torch.no_grad():pv=torch.sigmoid(net(vxt)).squeeze(-1).cpu().numpy()
        auc=_auc(yv,pv);curve.append({'epoch':ep+1,'selection_auc':auc})
        if auc>best[0]:best=(auc,{k:v.detach().cpu().clone() for k,v in net.state_dict().items()},ep+1)
    p=out/'models'/f'{source}_consistency_{kind}.pt';p.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'state':best[1],'mu':mu.astype(np.float32),'sd':sd.astype(np.float32),'in_dim':int(x.shape[1]),
        'best_epoch':best[2],'selection_auc':best[0],'curve':curve,'train_source':source,'feature_kind':kind,
        'train_corruption_counts':corruption_counts,
        'fit_tasks':sorted(np.unique(l90['task'][fit90]).tolist()),'selection_tasks':sorted(np.unique(l90['task'][sel]).tolist()),
        'eval_labels_used_for_fit_or_selection':False},p)
    return {'best_epoch':best[2],'selection_auc':best[0],'curve':curve}

def train_one(out,source,arch,layer,seed,*,temperature=1.0,epochs=20,batch=16,device='cuda',lr=3e-4):
    import torch
    torch.set_num_threads(8);torch.manual_seed(int(seed));out=Path(out);pool=_pool();teachers=_ensure_teacher_cache(out)
    l90=_l90x_data(out); select=np.flatnonzero(l90['is_select']);fit90=np.flatnonzero(~l90['is_select'])
    # Cache references are keyed by the exact task-level split; unseen Eval never appears here.
    t90=np.load(out/'teachers'/'l90x_fit.npz');v90=np.load(out/'teachers'/'l90x_select.npz')
    sources=[]
    if source in ('l90x','both'):
        sources.append(('l90x',l90,fit90,np.asarray(t90['teacher']),np.asarray(t90['err']),np.asarray(t90['allowed'],bool)))
    if source in ('lib40','both'):
        d40=_lib40_data(out);t40=np.load(out/'teachers'/'lib40_fit.npz');sources.append(('lib40',d40,np.arange(len(d40['task'])),np.asarray(t40['teacher']),np.asarray(t40['err']),np.asarray(t40['allowed'],bool)))
    if not sources:raise ValueError(source)
    nval=len(select);valid_ds=l90;valid_hist,modes=_mixed_history(l90,select,SEED+9000,selection=True)
    valid_err=np.asarray(v90['err']);pool_key=pool['key']
    net=make_model(arch,78,pool_key.shape[1],_token_dim(l90,layer),
        expert_dim=int(l90['expert'].shape[-1])).to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=float(lr),weight_decay=1e-4)
    rng=np.random.default_rng(seed);best=(np.inf,None,0);curve=[]
    # Each source contributes equal draws in the mixed condition.
    for ep in range(int(epochs)):
        net.train();losses=[]
        steps=max(1,(sum(len(x[2]) for x in sources)+batch-1)//batch)
        for st in range(steps):
            chosen=[]
            for nm,ds,ids,teach,err,allowed in sources:
                cnt=batch//len(sources);rids=rng.choice(ids,size=cnt,replace=len(ids)<cnt);chosen.append((nm,ds,ids,rids,teach,err,allowed))
            # For an odd batch, add the remainder from a seeded source.
            if batch%len(sources):
                nm,ds,ids,teach,err,allowed=sources[0];rids=rng.choice(ids,size=batch%len(sources),replace=len(ids)<batch%len(sources));chosen.append((nm,ds,ids,rids,teach,err,allowed))
            feats=[];targets=[];errs=[];allows=[]
            for nm,ds,source_ids,ix,teach,err,allowed in chosen:
                hh,_=_mixed_history(ds,ix,seed+ep*100003+st*101+len(feats))
                # Cached labels are compact per split, while ix are row IDs in the full source.
                local=np.searchsorted(source_ids,ix)
                feats.append(_feature(ds,ix,layer,hh,arch,device));targets.append(teach[local]);errs.append(err[local]);allows.append(allowed[local])
            feat={k:torch.cat([f[k] for f in feats],0) for k in feats[0]};ids_t=np.concatenate([x[2] for x in chosen])
            target=torch.as_tensor(np.concatenate(targets),device=device);errt=torch.as_tensor(np.concatenate(errs),device=device)
            allowed_t=torch.as_tensor(np.concatenate(allows),device=device,dtype=torch.bool)
            logits=net(feat,torch.as_tensor(pool_key,device=device))
            # Both objectives use exactly the per-query candidate mask.
            loss=masked_training_loss(logits/float(temperature),target,errt,allowed_t)
            if not torch.isfinite(loss):raise FloatingPointError('non-finite objective')
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();losses.append(float(loss.detach()))
        net.eval();score=_score_net(net,valid_ds,select,layer,arch,pool_key,batch=batch,history_override=valid_hist,device=device)
        top=np.argmin(score,axis=1);toperr=float(np.mean(valid_err[np.arange(len(select)),top]));sp=_conditional_spearman_rows(score,valid_err);metric=float(np.nanmean(sp));curve.append({'epoch':ep+1,'loss':float(np.mean(losses)),
            'temperature':float(temperature),'selection_mixed_top1_type_error':toperr,'selection_mixed_conditional_spearman_diagnostic':metric,
            'selection_mixed_full_spearman':float(np.nanmean(_spearman_rows(score,valid_err))),
            'selection_true_top1_type_error':float(np.mean(valid_err[np.arange(len(select)),np.argmin(_score_net(net,valid_ds,select,layer,arch,pool_key,batch=batch,device=device),axis=1)]))})
        if toperr<best[0]:best=(toperr,{k:v.detach().cpu().clone() for k,v in net.state_dict().items()},ep+1)
    result={'source':source,'arch':arch,'layer':layer,'seed':seed,'temperature':float(temperature),'epochs':epochs,'batch':batch,'lr':lr,
        'best_selection_top1_type_error':best[0],'best_epoch':best[2],'state':best[1],'curve':curve,
        'selection_tasks':sorted(np.unique(l90['task'][select]).tolist()),'eval_actions_used_for_selection':False}
    md=out/'models';md.mkdir(parents=True,exist_ok=True);tag=f'{float(temperature):g}'.replace('.','p');p=md/f'{source}_{arch}_{layer}_t{tag}_s{seed}.pt';tmp=p.with_suffix('.tmp');torch.save(result,tmp);tmp.replace(p)
    return result

def _teacher_nn(pred,pool):
    _,vs,_=so._mods();q=vs.to_physical(np.asarray(pred,np.float32)[:,:,:6]);return so.scale_only_error(q,pool['phys'],'S-shared')

def _score_config(out,source,config,seeds,ds,ids,hist,pool,*,batch=32,device='cuda'):
    import torch
    arch=config['arch'];layer=config['layer'];temp=config['temperature'];tag=f'{float(temp):g}'.replace('.','p');parts=[]
    for seed in seeds:
        ck=torch.load(Path(out)/'models'/f'{source}_{arch}_{layer}_t{tag}_s{seed}.pt',weights_only=False)
        net=make_model(arch,78,pool['key'].shape[1],_token_dim(ds,layer),expert_dim=int(ds['expert'].shape[-1])).to(device)
        _load_state_compat(net,ck['state']);net.eval()
        parts.append(_score_net(net,ds,ids,layer,arch,pool['key'],batch=batch,history_override=hist,device=device))
    return np.mean(parts,axis=0)

def _predict_consistency(out,source,kind,ds,ids,hist,device='cuda'):
    import torch
    ck=torch.load(Path(out)/'models'/f'{source}_consistency_{kind}.pt',weights_only=False)
    net=make_consistency_model(int(ck['in_dim'])).to(device);net.load_state_dict(ck['state']);net.eval()
    x=_consistency_features(ds,ids,hist,kind);x=(x-ck['mu'])/ck['sd']
    with torch.no_grad():return torch.sigmoid(net(torch.as_tensor(x,dtype=torch.float32,device=device))).squeeze(-1).cpu().numpy()

def _evaluate_consistency_routes(out,source,ds,err,targets,hard,histories,pool,selection,split,*,device='cuda'):
    """Evaluate detector AUC and thresholded P5/image branch switching."""
    import torch
    n=len(targets);ids=np.arange(n);selected=selection['selected_models'];seeds=selection['seeds']
    def model_scores(arch,hh,qids):
        cfg=selected[arch];tag=f"{float(cfg['temperature']):g}".replace('.','p');parts=[]
        for seed in seeds:
            ck=torch.load(Path(out)/'models'/f"{source}_{arch}_{cfg['layer']}_t{tag}_s{seed}.pt",weights_only=False)
            net=make_model(arch,78,pool['key'].shape[1],_token_dim(ds,cfg['layer']),expert_dim=int(ds['expert'].shape[-1])).to(device)
            _load_state_compat(net,ck['state']);net.eval();parts.append(_score_net(net,ds,qids,cfg['layer'],arch,pool['key'],history_override=hh[qids],device=device))
        return np.mean(parts,axis=0)
    route=selection['route_thresholds'];p5_arch='P5';image_cfg=selection['best_image_model'];image_arch=image_cfg['arch'];branch=selection['selection_errors']['selected_image_branch']
    score_cache={}
    for name,(hh,qids) in histories.items():
        if not len(qids):continue
        p5=model_scores(p5_arch,hh,qids)
        img=(-_teacher_nn(np.asarray(ds['pi05'][qids]),pool)) if branch=='pi05_nn' else model_scores(image_arch,hh,qids)
        score_cache[name]=(qids,p5,img)
    out_by={}
    for kind in ('vlm_mean','mid17','pi05'):
        threshold=float(route[kind]['threshold']);probabilities={}
        for name,(hh,qids) in histories.items():
            if len(qids):probabilities[name]=(qids,_predict_consistency(out,source,kind,ds,qids,hh[qids],device))
        truep=probabilities.get('true',(np.empty(0,np.int64),np.empty(0)))[1]
        aucs={};corrupt_names=['swap','reverse','zero','phase-10','phase+10','phase-20','phase+20']
        for cname in corrupt_names:
            hist_key='none' if cname=='zero' else cname
            if 'phase' in cname:hist_key=cname
            if hist_key not in probabilities:continue
            qidsp,pc=probabilities[hist_key]
            # AUC pairs only the true/corrupt rows with identical query ids.
            qt,pt=probabilities['true'];common=np.intersect1d(qt,qidsp,assume_unique=True)
            if len(common)==0:aucs[cname]={'auc':None,'rows':0};continue
            ti=np.searchsorted(qt,common);ci=np.searchsorted(qidsp,common)
            aucs[cname]={'auc':_auc(np.r_[np.zeros(len(common)),np.ones(len(common))],np.r_[pt[ti],pc[ci]]),'rows':int(len(common))}
        fpr=float(np.mean(truep>=threshold)) if len(truep) else None
        route_rows={}
        for name,(qids,p5,img) in score_cache.items():
            qidsp,pb=probabilities[name];chosen=np.where(pb[:,None]>=threshold,img,p5);pick=chosen.argmax(1)
            te=err[qidsp,pick];route_rows[name]={'rows':int(len(qidsp)),'top1_type_error':float(te.mean()),
                'top1_type_error_hard':float(te[hard[qidsp]].mean()) if np.any(hard[qidsp]) else None,
                'routed_to_image_fraction':float(np.mean(pb>=threshold)),'mean_p_broken':float(np.mean(pb))}
        out_by[kind]={'threshold':threshold,'image_branch':branch,'image_model':image_cfg,'false_positive_rate_true_history':fpr,
            'auc_corruption_vs_true':aucs,'routed_type_error':route_rows}
    return out_by

def train_all(out,*,source='l90x',architectures=ARCHES,layers=LAYERS,seeds=SEEDS,temperatures=TEMPERATURES,epochs=20,batch=16,device='cuda'):
    out=Path(out);_ensure_teacher_cache(out);data=_l90x_data(out);fit,sel=split_fit_select(data['is_select']);pool=_pool()
    jobs=[]
    for arch in architectures:
        for layer in (('final',) if _plain_arch(arch)=='P5' else layers):
            for temp in temperatures:
                for seed in seeds:jobs.append((arch,layer,float(temp),seed))
    for arch,layer,temp,seed in jobs:
        tag=f'{temp:g}'.replace('.','p');p=out/'models'/f'{source}_{arch}_{layer}_t{tag}_s{seed}.pt'
        if not p.exists():
            res=train_one(out,source,arch,layer,seed,temperature=temp,epochs=epochs,batch=batch,device=device)
            log(out,stage='train',source=source,arch=arch,layer=layer,temperature=temp,seed=seed,epoch=res['best_epoch'],selection_top1_type_error=round(res['best_selection_top1_type_error'],5))
    select_hist,_=_mixed_history(data,sel,SEED+9000,selection=True);select_err=np.load(out/'teachers'/'l90x_select.npz')['err']
    selected={};selection_rows=[]
    for arch in architectures:
        best=(np.inf,None)
        for layer in (('final',) if _plain_arch(arch)=='P5' else layers):
            for temp in temperatures:
                cfg={'arch':arch,'layer':layer,'temperature':float(temp)}
                scores=_score_config(out,source,cfg,seeds,data,sel,select_hist,pool,batch=batch,device=device)
                pred=scores.argmax(1);top=float(np.mean(select_err[np.arange(len(sel)),pred]))
                rec={**cfg,'selection_mixed_top1_type_error':top};selection_rows.append(rec)
                if top<best[0]:best=(top,cfg)
        selected[arch]={**best[1],'selection_mixed_top1_type_error':float(best[0])}
    image_candidates=[(v['selection_mixed_top1_type_error'],a,v) for a,v in selected.items() if _plain_arch(a) in ('XH','XC','XA')]
    if not image_candidates:raise ValueError('at least one XH/XC/XA image model is required for consistency routing')
    _,best_image_arch,best_image_cfg=min(image_candidates,key=lambda x:x[0])
    best_image_cfg={'arch':best_image_arch,**best_image_cfg}
    # Compare π0.5-nearest and the best learned image model on selection only.
    pi_sel=-_teacher_nn(np.asarray(data['pi05'][sel]),pool)
    image_sel=_score_config(out,source,best_image_cfg,seeds,data,sel,select_hist,pool,batch=batch,device=device)
    pi_err=float(np.mean(select_err[np.arange(len(sel)),pi_sel.argmax(1)]));learned_img_err=float(np.mean(select_err[np.arange(len(sel)),image_sel.argmax(1)]))
    image_branch='pi05_nn' if pi_err<=learned_img_err else 'learned_image'
    image_score=pi_sel if image_branch=='pi05_nn' else image_sel
    # Selection-tuned fixed scalar remains a comparator, now selected by the primary top-1 error.
    p5cfg=selected['P5'];p5cfg={'arch':'P5',**p5cfg}
    p5_sel=_score_config(out,source,p5cfg,seeds,data,sel,select_hist,pool,batch=batch,device=device)
    p5_temp=p5cfg['temperature'];fw_best=(np.inf,0.)
    for w in np.linspace(0,3,31):
        sc=soz(p5_sel)+float(w)*soz(image_score);pred=sc.argmax(1);m=float(np.mean(select_err[np.arange(len(sel)),pred]))
        if m<fw_best[0]:fw_best=(m,float(w))
    gate=train_fusion_gate(out,source,p5_temperature=p5_temp,seeds=seeds,epochs=min(epochs,20),batch=max(batch,32),device=device)
    # Train all three detector inputs; the held-out L90X tasks alone choose epoch and route threshold.
    detectors={};route_thresholds={};mixed_hist,mixed_codes=_mixed_history(data,sel,SEED+9000,selection=True)
    p5_mix=p5_sel;img_mix=image_score
    for kind in ('vlm_mean','mid17','pi05'):
        det=train_consistency_model(out,source,kind,epochs=epochs,batch=max(batch,32),device=device)
        prob=_predict_consistency(out,source,kind,data,sel,mixed_hist,device)
        best=(np.inf,None)
        for threshold in np.linspace(0,1,101):
            sc=np.where(prob[:,None]>=threshold,img_mix,p5_mix);pred=sc.argmax(1)
            val=float(np.mean(select_err[np.arange(len(sel)),pred]))
            if val<best[0]:best=(val,float(threshold))
        route_thresholds[kind]={'threshold':best[1],'selection_mixed_top1_type_error':best[0],
            'selection_detector_auc':det['selection_auc'],'image_branch':image_branch,'image_model':best_image_cfg}
        detectors[kind]=det
    pi_sel_err=_teacher_nn(np.asarray(data['pi05'][sel]),pool)
    spath=out/'selection.json'
    write_json(spath,{'selection':'primary metric: mean top-1 S-shared type error on L90X task-held-out selection rows with true/corrupted histories mixed 50/50; Eval never used',
        'selected_tasks':sorted(np.unique(data['task'][sel]).tolist()),'temperature_grid':list(temperatures),
        'temperature_definition':'loss logits divided by this scalar during training; each value trains a separate representation; argmax error selects epoch and temperature',
        'selection_candidates':selection_rows,'selected_models':selected,'best_image_model':best_image_cfg,
        'selection_errors':{'pi05_nn':pi_err,'best_learned_image':learned_img_err,'selected_image_branch':image_branch},
        'fixed_fusion':{'weight':fw_best[1],'selection_top1_type_error':fw_best[0]},
        'fusion_gate':gate,'consistency_detectors':detectors,'route_thresholds':route_thresholds,
        'seeds':list(seeds),'architecture_runs':[list(x) for x in jobs]})

def _scale_shape_to_ref(shape_phys,ref_norm):
    _,vs,_=so._mods();q=vs.to_physical(np.asarray(ref_norm,np.float32)[:,:,:6]).astype(np.float32)
    x=np.asarray(shape_phys,np.float32);den=np.sum(x*x,axis=(1,2));num=np.sum(x*q,axis=(1,2))
    scale=np.maximum(num/np.maximum(den,1e-12),0.)
    return vs.to_normalized(x*scale[:,None,None]).astype(np.float32)

def _metric_pack(scores,err,targets,hard,pool,history_ref,pi_ref,episode):
    pick=np.argmax(scores,axis=1);rows=np.arange(len(pick));typeerr=err[rows,pick]
    rank=_retrieval_row_metrics(scores,err);sp=_spearman_rows(scores,err);phys=np.asarray(pool['phys'][pick],np.float32)
    pred_stored=so._mods()[1].to_normalized(phys);pred_pi=_scale_shape_to_ref(phys,pi_ref);pred_hist=_scale_shape_to_ref(phys,history_ref);pred_oracle=_scale_shape_to_ref(phys,targets)
    mse={'stored':so.six_axis(pred_stored,targets),'pi05_scale':so.six_axis(pred_pi,targets),
         'history_scale':so.six_axis(pred_hist,targets),'oracle_scale':so.six_axis(pred_oracle,targets)}
    metrics={'rows':len(pick),'hard_rows':int(np.sum(hard)),'top1_type_error':float(np.mean(typeerr)),
        'top1_type_error_hard':float(np.mean(typeerr[hard])) if np.any(hard) else None,
        'spearman_all':float(np.mean(sp)),'spearman_top10':float(np.mean(rank['conditional_spearman'])),
        'spearman_top10_hard':float(np.mean(rank['conditional_spearman'][hard])) if np.any(hard) else None,
        'pearson_top10':float(np.mean(rank['pearson_top10'])),'pearson_top10_hard':float(np.mean(rank['pearson_top10'][hard])) if np.any(hard) else None,
        'pearson_all':float(np.mean(rank['pearson_all'])),'pearson_all_hard':float(np.mean(rank['pearson_all'][hard])) if np.any(hard) else None,
        'spearman_top10_valid_windows':int(np.sum(rank['conditional_spearman_valid'])),'spearman_top10_valid_hard_windows':int(np.sum(rank['conditional_spearman_valid'][hard])) if np.any(hard) else 0,
        'pearson_top10_valid_windows':int(np.sum(rank['pearson_top10_valid'])),'pearson_all_valid_windows':int(np.sum(rank['pearson_all_valid'])),
        'pearson_top10_valid_hard_windows':int(np.sum(rank['pearson_top10_valid'][hard])) if np.any(hard) else 0,
        'pearson_all_valid_hard_windows':int(np.sum(rank['pearson_all_valid'][hard])) if np.any(hard) else 0,
        'recall@1':float(np.mean(rank['recall@1'])),'recall@5':float(np.mean(rank['recall@5'])),'recall@20':float(np.mean(rank['recall@20'])),
        'ndcg@10':float(np.mean(rank['ndcg@10'])),'gap@1':float(np.mean(rank['gap@1'])),'gap@5':float(np.mean(rank['gap@5'])),
        **{f'mse6_{k}':float(np.mean(v)) for k,v in mse.items()},
        **{f'mse6_{k}_hard':float(np.mean(v[hard])) if np.any(hard) else None for k,v in mse.items()}}
    return metrics,{'top1_type_error_rows':typeerr,'spearman_all_rows':sp,'episode':np.asarray(episode),**{f'mse6_{k}_rows':v for k,v in mse.items()}}

def eval_sources(out,*,source='l90x',seeds=SEEDS,layers=LAYERS,device='cuda',max_rows=None):
    """Evaluate chosen temperature/model only after all L90X selection is fixed."""
    import torch
    out=Path(out);pool=_pool();selection=json.loads((out/'selection.json').read_text());r=_load_row_dir(UNSEEN)
    chosen=selection['selected_models']
    for arch,cfg in chosen.items():
        tag=f"{float(cfg['temperature']):g}".replace('.','p')
        for seed in seeds:
            path=out/'models'/f"{source}_{arch}_{cfg['layer']}_t{tag}_s{seed}.pt"
            if not path.exists():raise FileNotFoundError(f'missing selected model checkpoint {path}')
    gate_ck=torch.load(out/'models'/f'{source}_fusion_gate.pt',weights_only=False)
    gate=make_fusion_gate(int(gate_ck['in_dim']),int(gate_ck['delta_dim'])).to(device);gate.load_state_dict(gate_ck['state']);gate.eval()
    final=_make_final_image_cache(UNSEEN,out/'external_features');res={};base_metrics={};split_rows={};route_metrics={}
    for split in ('dev','eval'):
        prep=dict(np.load(POOL_DIR/f'W_inputs_{split}.npz'));allrow=np.asarray(prep['row'],np.int64);nfull=len(allrow);pos=np.arange(nfull,dtype=np.int64)
        if max_rows is not None and nfull>int(max_rows):pos=smoke_positions(prep['task'],prep['episode'],prep['frame'],max_rows)
        rowids=allrow[pos];n=len(pos);mids=np.load(PI/f'{split}_mid_v.npy',mmap_mode='r');hid=np.load(PI/f'{split}_hidden_t1.npy',mmap_mode='r');pact=np.load(PI/f'{split}_actions.npy',mmap_mode='r')
        d=_attach_predecessors({'hist':np.asarray(prep['hist'][pos],np.float32),'state':np.asarray(prep['state'][pos,:8],np.float32),
            'task':np.asarray(prep['task'][pos],np.int64),'episode':np.asarray(prep['episode'][pos],np.int64),'frame':np.asarray(prep['frame'][pos],np.int64),
            'has_hist':np.asarray(prep['has_hist'][pos],bool),'pi05':np.asarray(pact[pos],np.float32),'expert':np.asarray(hid[pos],np.float32),
            'image':np.asarray(final[rowids]),'mid16':np.asarray(mids[pos,4,:128]),'mid17':np.asarray(mids[pos,5,:128])})
        hard=np.asarray(prep.get('hard',np.zeros(nfull,bool))[pos],bool);targets=np.asarray(r['actions'][rowids,:,:6],np.float32);err=_error_matrix(targets,pool)
        pierr=_teacher_nn(d['pi05'],pool);pnn=-pierr;pi_pick=np.argmin(pierr,axis=1);pi_shape=pool['phys'][pi_pick]
        hist_ref=np.asarray(d['hist'][:,-10:,:6],np.float32);hist_ref[~d['has_hist']]=0;pi_ref=np.asarray(d['pi05'][:,:,:6],np.float32)
        previous=np.broadcast_to(hist_ref[:,-1:,:],targets.shape).copy();base_metrics[split]={'previous_action_mse6':float(so.six_axis(previous,targets).mean()),
            'pi05_mse6':float(so.six_axis(pi_ref,targets).mean()),'pi05_nn_top1_type_error':float(np.mean(err[np.arange(n),pi_pick]))}
        split_rows[split]={'rows':int(n),'available_rows':int(nfull),'hard':int(hard.sum()),'smoke_subset':max_rows is not None}
        allrows=np.arange(n);histories={'true':(d['hist'],allrows),'swap':(corrupt_history(d['hist'],'swap',seed=4101,task=d['task']),allrows),
            'reverse':(corrupt_history(d['hist'],'reverse'),allrows),'none':(corrupt_history(d['hist'],'zero'),allrows)}
        for shift in (-20,-10,10,20):histories[f'phase{shift:+d}']=phase_shift_history(d,shift)
        split_res={}
        for arch,cfg in chosen.items():
            scores=[];tag=f"{float(cfg['temperature']):g}".replace('.','p')
            for seed in seeds:
                ck=torch.load(out/'models'/f"{source}_{arch}_{cfg['layer']}_t{tag}_s{seed}.pt",weights_only=False)
                net=make_model(arch,78,pool['key'].shape[1],_token_dim(d,cfg['layer']),expert_dim=int(d['expert'].shape[-1])).to(device)
                _load_state_compat(net,ck['state']);net.eval()
                scores.append(net)
            for hn,(hh,qids) in histories.items():
                if not len(qids):continue
                per=[]
                for net in scores:per.append(_score_net(net,d,qids,cfg['layer'],arch,pool['key'],history_override=hh[qids],device=device))
                sc=np.mean(per,axis=0);model_name=f"{arch}@T{float(cfg['temperature']):g}"
                if arch=='P5':
                    fixed=soz(sc)+float(selection['fixed_fusion']['weight'])*soz(pnn[qids])
                    gatefeat=_feature(d,qids,'final',hh[qids],'P5-imgdelta',device)
                    with torch.no_grad():gated,_=combine_with_gate(gate,torch.as_tensor(sc,device=device),torch.as_tensor(pnn[qids],device=device),gatefeat)
                    variants=[(model_name,sc),('P5+pi05NN-fixed',fixed),('P5+pi05NN-gated',gated.cpu().numpy())]
                else:variants=[(model_name,sc)]
                for name,score in variants:
                    met,raw=_metric_pack(score,err[qids],targets[qids],hard[qids],pool,hh[qids][:,-10:,:6],pi_ref[qids],d['task'][qids]*1000+d['episode'][qids])
                    key=f'{source}|{name}|{split}|{hn}';split_res[key]={'metrics':met,**{k:v.tolist() for k,v in raw.items()}}
        # Selection-only chosen route architecture and classifier thresholds are now fixed.
        route_metrics[split]=_evaluate_consistency_routes(out,source,d,err,targets,hard,histories,pool,selection,split,device=device)
        res.update(split_res)
    # Keep raw per-row metrics for bootstrap or downstream inspection.
    paired={}
    for key,v in res.items():
        bits=key.split('|')
        if len(bits)!=4 or bits[1].startswith('P5@') or bits[1].startswith('P5+'):continue
        src,name,split,hn=bits
        p5key=next((k for k in res if k.startswith(f'{src}|P5@T') and k.endswith(f'|{split}|{hn}')),None)
        if p5key:
            a=v['top1_type_error_rows'];b=res[p5key]['top1_type_error_rows']
            paired[key]=_episode_bootstrap_diff(a,b,v['episode'])
    write_json(out/'evaluation.json',{'metrics':{k:v['metrics'] for k,v in res.items()},'row_metrics':res,
        'paired_top1_type_error_vs_P5':paired,'baselines':base_metrics,'consistency_routing':route_metrics,'split_rows':split_rows,
        'model_selection_source':'L90X task-held-out selection rows, minimized top-1 S-shared type error','eval_actions_used_for':'metrics only after model/temperature/detector/threshold selection fixed','seeds':list(seeds)})
    return res

def soz(x):
    a=np.asarray(x,np.float32);return (a-a.mean(1,keepdims=True))/(a.std(1,keepdims=True)+1e-8)

def report(out):
    out=Path(out);ev=json.loads((out/'evaluation.json').read_text());sel=json.loads((out/'selection.json').read_text());lines=['# X1 results','',
      'Primary selection criterion: mean top-1 S-shared type error on L90X held-out selection tasks, with true/corrupted histories mixed 50/50. Temperature and checkpoint are selected by this criterion; Eval targets are used only for scoring.','',
      f"Selection tasks: {sel['selected_tasks']}; temperature candidates: {sel['temperature_grid']}; selected models: {sel['selected_models']}; P5/nearest fixed-weight comparator: {sel['fixed_fusion']}; best image-side branch: {sel['selection_errors']}.",'',
      '| model / split / history | n all/hard | top1 type error all/hard | Spearman top10 all/hard | Pearson top10 all/hard | Pearson all all/hard | valid Spearman / Pearson-top10 / Pearson-all windows (all; hard) |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for key,entry in ev['row_metrics'].items():
        _,model,split,hist=key.split('|');hist='zero' if hist=='none' else hist;m=entry['metrics'];n=m['rows'];nh=m['hard_rows']
        lines.append(f"| {model} / {split} / {hist} | {n}/{nh} | {m['top1_type_error']:.5f}/{m['top1_type_error_hard'] if m['top1_type_error_hard'] is not None else float('nan'):.5f} | {m['spearman_top10']:.3f}/{m['spearman_top10_hard'] if m['spearman_top10_hard'] is not None else float('nan'):.3f} | {m['pearson_top10']:.3f}/{m['pearson_top10_hard'] if m['pearson_top10_hard'] is not None else float('nan'):.3f} | {m['pearson_all']:.3f}/{m['pearson_all_hard'] if m['pearson_all_hard'] is not None else float('nan'):.3f} | S {m['spearman_top10_valid_windows']}/{n} ({m['spearman_top10_valid_hard_windows']}/{nh}); P10 {m['pearson_top10_valid_windows']}/{n} ({m['pearson_top10_valid_hard_windows']}/{nh}); Pall {m['pearson_all_valid_windows']}/{n} ({m['pearson_all_valid_hard_windows']}/{nh}) |")
    lines+=['','## Secondary: 6-axis MSE, stored / π0.5 magnitude / history speed / oracle s*','',
      '| model / split / history | stored all/hard | π0.5 scale all/hard | history scale all/hard | s* all/hard | R@1 / R@5 / R@20 | NDCG@10 | gap@1 / gap@5 |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for key,entry in ev['row_metrics'].items():
        _,model,split,hist=key.split('|');hist='zero' if hist=='none' else hist;m=entry['metrics']
        lines.append(f"| {model} / {split} / {hist} | {m['mse6_stored']:.4f}/{m['mse6_stored_hard'] if m['mse6_stored_hard'] is not None else float('nan'):.4f} | {m['mse6_pi05_scale']:.4f}/{m['mse6_pi05_scale_hard'] if m['mse6_pi05_scale_hard'] is not None else float('nan'):.4f} | {m['mse6_history_scale']:.4f}/{m['mse6_history_scale_hard'] if m['mse6_history_scale_hard'] is not None else float('nan'):.4f} | {m['mse6_oracle_scale']:.4f}/{m['mse6_oracle_scale_hard'] if m['mse6_oracle_scale_hard'] is not None else float('nan'):.4f} | {m['recall@1']:.3f}/{m['recall@5']:.3f}/{m['recall@20']:.3f} | {m['ndcg@10']:.3f} | {m['gap@1']:.3f}/{m['gap@5']:.3f} |")
    lines+=['','## Paired episode bootstrap: top-1 type error difference vs selected P5','']
    for key,v in ev.get('paired_top1_type_error_vs_P5',{}).items():
        if '|eval|' in key:lines.append(f"- `{key}`: Δ {v['mean']:+.5f} [{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}].")
    lines+=['','## History-consistency detector and switching','',
        '| split / detector input | threshold | true-history false positive | corruption AUC | routed top1 type error (true and each corrupted history) |','|---|---:|---:|---|---|']
    for split,by_kind in ev.get('consistency_routing',{}).items():
        for kind,report in by_kind.items():
            auc='; '.join(f"{c}:{v['auc'] if v['auc'] is not None else float('nan'):.3f} (n={v['rows']})" for c,v in report['auc_corruption_vs_true'].items())
            errs='; '.join(f"{h}:{v['top1_type_error']:.5f} / route={v['routed_to_image_fraction']:.2f}" for h,v in report['routed_type_error'].items())
            lines.append(f"| {split} / {kind} | {report['threshold']:.2f} | {report['false_positive_rate_true_history'] if report['false_positive_rate_true_history'] is not None else float('nan'):.3f} | {auc} | {errs} |")
    lines+=['','## Baselines','', '| split | previous-action MSE6 | π0.5 MSE6 | π0.5-NN top1 type error |','|---|---:|---:|---:|']
    for split,m in ev.get('baselines',{}).items():
        lines.append(f"| {split} | {m['previous_action_mse6']:.4f} | {m['pi05_mse6']:.4f} | {m['pi05_nn_top1_type_error']:.5f} |")
    gatefile=ROOT/'image-query-01'/'gate'/'dev_results.json'
    if gatefile.exists():
        gate=json.loads(gatefile.read_text()).get('gate_dev',[])
        if gate:
            summary={h:float(np.mean([r[h]['spearman'] for r in gate])) for h in ('true','swap','reverse','zero')}
            lines += ['',f"Existing Dev-trained gate reference (I2b; not refit here), Eval Spearman: {summary}."]
    lines+=['','Temporal variants use the visual feature difference from the exact previous 10-frame window. The learned P5+π0.5-NN gate attends to raw Δ-image tokens as well as the difference norm, executed-history/image-difference ratio, history motion norm, and predecessor flag. The fixed-weight blend is retained as a comparator.','',
      'The consistency detector is trained on an explicitly balanced true/corrupted binary set using swap, reverse, zero, and same-episode phase shifts (±10/±20). Its feature inputs are final VLM mean, layer-17 mean, or π0.5 action output, each with state, history, and image_query_gate.agreement. Epoch uses L90X selection AUC; route threshold minimizes L90X selection top-1 type error.','',
      f"The smoke trained P5 and XH-imgdelta (final-layer input), with temperatures {sel['temperature_grid']}, on 4 L90X tasks × 2 demos, one seed, and two epochs. For metric-path verification only, it evaluated a deterministic contiguous subset ({ev.get('split_rows',{}).get('dev',{}).get('rows',0)} Dev / {ev.get('split_rows',{}).get('eval',{}).get('rows',0)} Eval rows, from 113 / 619 available); it does not establish full-scale performance.",'']
    (out/'report.md').write_text('\n'.join(lines))

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['extract-lib40','cache-teachers','train','evaluate','report','smoke']);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--rows-dir',type=Path);p.add_argument('--source',choices=['l90x','lib40','both'],default='l90x')
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch',type=int,default=16);p.add_argument('--threads',type=int,default=8);p.add_argument('--device',default='cuda')
    p.add_argument('--temperatures',default=','.join(map(str,TEMPERATURES)),help='comma-separated retrieval temperature candidates')
    p.add_argument('--arch',default='all');p.add_argument('--layer',default='all');a=p.parse_args();out=a.out
    if a.stage=='extract-lib40':print(json.dumps(extract_lib40_final(out,batch=a.batch,threads=a.threads),ensure_ascii=False))
    elif a.stage=='cache-teachers':_ensure_teacher_cache(out)
    elif a.stage=='train':
        arch=ARCHES if a.arch=='all' else tuple(a.arch.split(','));layers=LAYERS if a.layer=='all' else tuple(a.layer.split(','))
        train_all(out,source=a.source,architectures=arch,layers=layers,temperatures=tuple(float(x) for x in a.temperatures.split(',')),epochs=a.epochs,batch=a.batch,device=a.device)
    elif a.stage=='evaluate':eval_sources(out,source=a.source,layers=LAYERS if a.layer=='all' else tuple(a.layer.split(',')),device=a.device)
    elif a.stage=='report':report(out)
    elif a.stage=='smoke':
        if a.rows_dir is None:raise ValueError('--rows-dir required for smoke')
        extract_l90x(a.rows_dir);prepare_l90x_features(a.rows_dir);_ensure_teacher_cache(out)
        train_all(out,source='l90x',architectures=ARCHES,layers=LAYERS,seeds=(4101,),temperatures=tuple(float(x) for x in a.temperatures.split(',')),epochs=2,batch=8,device=a.device)
        eval_sources(out,source='l90x',seeds=(4101,),layers=LAYERS,device=a.device,max_rows=16);report(out)
if __name__=='__main__':main()
