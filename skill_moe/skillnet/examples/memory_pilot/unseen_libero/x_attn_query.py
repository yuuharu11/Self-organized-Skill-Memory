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
BASE_ARCHES=('P5','XH','XC','XA')
# The context variants use an actual within-episode image-feature difference,
# its magnitude, executed-motion/image-change ratio, and a predecessor flag.
ARCHES=BASE_ARCHES+tuple(f'{a}-imgdelta' for a in BASE_ARCHES)
LAYERS=('final','mid16','mid17','top3')
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
    return torch.cat([torch.as_tensor(p,device=feat['hist'].device),torch.as_tensor(n,device=feat['hist'].device),ctx,torch.log1p(hmotion),torch.log1p(snorm)],dim=1)

def make_fusion_gate(in_dim=14):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(in_dim,32),nn.GELU(),nn.Linear(32,1))

def combine_with_gate(gate,p5,nn,feat):
    import torch
    x=_fusion_gate_features(p5,nn,feat);alpha=torch.sigmoid(gate(x))
    return alpha*_zscore_rows_torch(p5)+(1-alpha)*_zscore_rows_torch(nn),alpha.squeeze(-1)

def _zscore_rows_torch(x):
    return (x-x.mean(1,keepdim=True))/(x.std(1,keepdim=True,unbiased=False)+1e-6)

def train_fusion_gate(out,source,*,seed=SEED,epochs=20,batch=64,device='cuda'):
    """Fit a query-conditioned P5/π0.5-NN gate on L90X fit only."""
    import torch
    torch.set_num_threads(8);torch.manual_seed(int(seed));out=Path(out);pool=_pool();data=_l90x_data(out)
    fit=np.flatnonzero(~data['is_select']);sel=np.flatnonzero(data['is_select'])
    te=np.load(out/'teachers'/'l90x_fit.npz');ve=np.load(out/'teachers'/'l90x_select.npz')
    fit_hist,modes=_mixed_history(data,fit,seed+100)
    p5=[]
    for s in SEEDS[:1]:
        ck=torch.load(out/'models'/f'{source}_P5_final_s{s}.pt',weights_only=False)
        net=make_model('P5',78,pool['key'].shape[1],2048).to(device);_load_state_compat(net,ck['state']);net.eval()
        p5.append(_score_net(net,data,fit,'final','P5',pool['key'],batch=batch,history_override=fit_hist,device=device))
    p5fit=np.mean(p5,axis=0);p5s=[]
    for s in SEEDS[:1]:
        ck=torch.load(out/'models'/f'{source}_P5_final_s{s}.pt',weights_only=False)
        net=make_model('P5',78,pool['key'].shape[1],2048).to(device);_load_state_compat(net,ck['state']);net.eval()
        p5s.append(_score_net(net,data,sel,'final','P5',pool['key'],batch=batch,history_override=_mixed_history(data,sel,SEED+9000,selection=True)[0],device=device))
    p5sel=np.mean(p5s,axis=0);nnfit=-_teacher_nn(np.asarray(data['pi05'][fit]),pool);nnsel=-_teacher_nn(np.asarray(data['pi05'][sel]),pool)
    sel_hist,_=_mixed_history(data,sel,SEED+9000,selection=True)
    # Score-side features reflect the same corrupted history seen by P5.
    ffit=_feature(data,fit,'final',fit_hist,'P5-imgdelta',device)
    fsel=_feature(data,sel,'final',sel_hist,'P5-imgdelta',device)
    xf=_fusion_gate_features(p5fit,nnfit,ffit);xs=_fusion_gate_features(p5sel,nnsel,fsel)
    gate=make_fusion_gate(xf.shape[1]).to(device);opt=torch.optim.AdamW(gate.parameters(),lr=3e-3,weight_decay=1e-3)
    target=torch.as_tensor(te['teacher'],device=device);allowed=torch.as_tensor(te['allowed'],dtype=torch.bool,device=device)
    err=torch.as_tensor(te['err'],device=device);xft=torch.as_tensor(xf,dtype=torch.float32,device=device)
    vst=torch.as_tensor(ve['teacher'],device=device);va=torch.as_tensor(ve['allowed'],dtype=torch.bool,device=device);verr=torch.as_tensor(ve['err'],device=device)
    best=(-np.inf,None,0);curve=[];rng=np.random.default_rng(seed)
    for ep in range(int(epochs)):
        gate.train();order=rng.permutation(len(fit))
        for st in range(0,len(order),batch):
            ii=order[st:st+batch];alpha=torch.sigmoid(gate(xft[ii]));scores=alpha*_zscore_rows_torch(torch.as_tensor(p5fit[ii],device=device))+(1-alpha)*_zscore_rows_torch(torch.as_tensor(nnfit[ii],device=device))
            loss=masked_training_loss(scores,target[ii],err[ii],allowed[ii])
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        gate.eval()
        with torch.no_grad():
            alpha=torch.sigmoid(gate(torch.as_tensor(xs,dtype=torch.float32,device=device)))
            score=alpha*_zscore_rows_torch(torch.as_tensor(p5sel,device=device))+(1-alpha)*_zscore_rows_torch(torch.as_tensor(nnsel,device=device))
            metric=float(np.nanmean(_conditional_spearman_rows(score.cpu().numpy(),ve['err'])))
        curve.append({'epoch':ep+1,'selection_conditional_spearman':metric,'mean_alpha':float(alpha.mean().cpu())})
        if metric>best[0]:best=(metric,{k:v.detach().cpu().clone() for k,v in gate.state_dict().items()},ep+1)
    gate_path=out/'models'/f'{source}_fusion_gate.pt';gate_path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'state':best[1],'in_dim':int(xf.shape[1]),'best_epoch':best[2],'selection_conditional_spearman':best[0],
        'curve':curve,'fit_tasks':sorted(np.unique(data['task'][fit]).tolist()),'selection_tasks':sorted(np.unique(data['task'][sel]).tolist()),
        'eval_actions_used':False,'alpha_dynamic_per_query':True},gate_path)
    return {'best_epoch':best[2],'best_selection_conditional_spearman':best[0],'curve':curve}

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
        count=max(2,int(np.ceil(len(ids)*frac)));best=ids[np.argsort(e[ids],kind='stable')[:count]]
        if np.ptp(s[best])<1e-12 or np.ptp(e[best])<1e-12:out.append(0.);continue
        v=np.corrcoef(rankdata(s[best]),rankdata(-e[best]))[0,1];out.append(float(v) if np.isfinite(v) else 0.)
    return np.asarray(out,np.float64)

def _retrieval_row_metrics(scores,errors,allowed=None):
    """Top-focused ranking metrics following image_query_metrics.py."""
    s=np.asarray(scores);e=np.asarray(errors);a=np.ones_like(e,bool) if allowed is None else np.asarray(allowed,bool)
    result={k:[] for k in ('conditional_spearman','recall@1','recall@5','recall@20','ndcg@10','gap@1','gap@5')}
    for sr,er,ar in zip(s,e,a):
        ids=np.flatnonzero(ar)
        if not len(ids):continue
        ss=sr[ids];ee=er[ids];order=np.argsort(-ss,kind='stable');oracle=float(ee.min());random=float(ee.mean());gap=max(random-oracle,1e-9)
        nbest=max(2,int(np.ceil(.1*len(ids))));best=ids[np.argsort(ee,kind='stable')[:nbest]]
        from scipy.stats import rankdata
        c=0. if len(best)<2 or np.ptp(sr[best])<1e-12 or np.ptp(er[best])<1e-12 else np.corrcoef(rankdata(sr[best]),rankdata(-er[best]))[0,1]
        result['conditional_spearman'].append(float(c) if np.isfinite(c) else 0.)
        tau=max(.1*gap,1e-9);rel=np.exp(-(ee-oracle)/tau);ideal=np.sort(rel)[::-1][:10];disc=1/np.log2(np.arange(2,12));den=float(np.sum(ideal*disc))
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
                    else:modes[local_i]='true'
            elif m=='swap':out[ix]=corrupt_history(base,'swap',seed=seed+17,task=task)[ix]
            else:out[ix]=corrupt_history(base[ix],m,seed=seed+17)
    return out,modes

def train_one(out,source,arch,layer,seed,*,epochs=20,batch=16,device='cuda',lr=3e-4):
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
    rng=np.random.default_rng(seed);best=(-np.inf,None,0);curve=[]
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
            loss=masked_training_loss(logits,target,errt,allowed_t)
            if not torch.isfinite(loss):raise FloatingPointError('non-finite objective')
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();losses.append(float(loss.detach()))
        net.eval();score=_score_net(net,valid_ds,select,layer,arch,pool_key,batch=batch,history_override=valid_hist,device=device)
        sp=_conditional_spearman_rows(score,valid_err);metric=float(np.nanmean(sp));curve.append({'epoch':ep+1,'loss':float(np.mean(losses)),'selection_mixed_conditional_spearman':metric,
            'selection_mixed_full_spearman':float(np.nanmean(_spearman_rows(score,valid_err))),
            'selection_true_conditional_spearman':float(np.nanmean(_conditional_spearman_rows(_score_net(net,valid_ds,select,layer,arch,pool_key,batch=batch,device=device),valid_err)))})
        if metric>best[0]:best=(metric,{k:v.detach().cpu().clone() for k,v in net.state_dict().items()},ep+1)
    result={'source':source,'arch':arch,'layer':layer,'seed':seed,'epochs':epochs,'batch':batch,'lr':lr,
        'best_selection_conditional_spearman':best[0],'best_epoch':best[2],'state':best[1],'curve':curve,
        'selection_tasks':sorted(np.unique(l90['task'][select]).tolist()),'eval_actions_used_for_selection':False}
    md=out/'models';md.mkdir(parents=True,exist_ok=True);p=md/f'{source}_{arch}_{layer}_s{seed}.pt';tmp=p.with_suffix('.tmp');torch.save(result,tmp);tmp.replace(p)
    return result

def _teacher_nn(pred,pool):
    _,vs,_=so._mods();q=vs.to_physical(np.asarray(pred,np.float32)[:,:,:6]);return so.scale_only_error(q,pool['phys'],'S-shared')

def train_all(out,*,source='l90x',architectures=ARCHES,layers=LAYERS,seeds=SEEDS,epochs=20,batch=16,device='cuda'):
    out=Path(out);_ensure_teacher_cache(out)
    jobs=[]
    for a in architectures:
        for l in (('final',) if _plain_arch(a)=='P5' else layers):
            for s in seeds:jobs.append((a,l,s))
    for a,l,s in jobs:
        p=out/'models'/f'{source}_{a}_{l}_s{s}.pt'
        if not p.exists():
            res=train_one(out,source,a,l,s,epochs=epochs,batch=batch,device=device)
            log(out,stage='train',source=source,arch=a,layer=l,seed=s,epoch=res['best_epoch'],selection_conditional_spearman=round(res['best_selection_conditional_spearman'],4))
    # Keep the scalar blend as a baseline, then fit a query-conditioned gate
    # on L90X fit tasks and select its epoch on the disjoint L90X tasks.
    data=_l90x_data(out);sel=np.flatnonzero(data['is_select']);pool=_pool();teacher=np.load(out/'teachers'/'l90x_select.npz')['err']
    select_hist,_=_mixed_history(data,sel,SEED+9000,selection=True)
    pierr=_teacher_nn(np.asarray(data['pi05'][sel]),pool);nn=-pierr
    p5=[]
    for s in seeds:
        r=__import__('torch').load(out/'models'/f'{source}_P5_final_s{s}.pt',weights_only=False)
        net=make_model('P5',78,pool['key'].shape[1],2048).to(device);_load_state_compat(net,r['state']);net.eval()
        p5.append(_score_net(net,data,sel,'final','P5',pool['key'],batch=batch,history_override=select_hist,device=device))
    z=lambda x:(x-x.mean(1,keepdims=True))/(x.std(1,keepdims=True)+1e-8)
    scores=np.mean(p5,axis=0);best=(float('-inf'),0.)
    pi05_nn_selection_spearman=float(np.nanmean(_spearman_rows(-pierr,teacher)))
    for w in np.linspace(0,3,31):
        val=float(np.nanmean(_conditional_spearman_rows(z(scores)+w*z(nn),teacher)))
        if val>best[0]:best=(val,float(w))
    spath=out/'selection.json';old=json.loads(spath.read_text()) if spath.exists() else {}
    fw=dict(old.get('fusion_weights',{}));fw[source]={'weight':best[1],'selection_spearman':best[0],
        'role':'fixed scalar reference only'}
    dynamic=train_fusion_gate(out,source,epochs=min(epochs,20),batch=max(batch,32),device=device)
    runs=[];seen=set()
    for job in old.get('architecture_runs',[])+jobs:
        normalized=tuple(job)
        if normalized not in seen:seen.add(normalized);runs.append(list(normalized))
    write_json(spath,{'selection':'conditional Spearman on fixed L90X selection tasks; true/corrupted histories mixed 50/50; Eval never used',
        'selected_tasks':sorted(np.unique(data['task'][sel]).tolist()),
        'pi05_nn_selection_spearman':pi05_nn_selection_spearman,'fusion_weights':fw,
        'fusion_gate':dynamic,'fusion_gate_training':'L90X fit tasks only; epoch selected by conditional Spearman on separate L90X selection tasks',
        'seeds':list(seeds),'architecture_runs':runs})

def eval_sources(out,*,source='l90x',seeds=SEEDS,layers=LAYERS,device='cuda',max_rows=None):
    """Score Dev/Eval with labels loaded only after trained state/model config is fixed."""
    import torch
    out=Path(out);pool=_pool();selection=json.loads((out/'selection.json').read_text());r=_load_row_dir(UNSEEN)
    # Refuse to inspect unseen targets before the selection record and all selected weights are fixed.
    for arch in ARCHES:
        for layer in (('final',) if _plain_arch(arch)=='P5' else tuple(layers)):
            for seed in seeds:
                if not (out/'models'/f'{source}_{arch}_{layer}_s{seed}.pt').exists():raise FileNotFoundError(f'missing selected model {source}/{arch}/{layer}/{seed}')
    gate_ck=torch.load(out/'models'/f'{source}_fusion_gate.pt',weights_only=False)
    gate=make_fusion_gate(int(gate_ck['in_dim'])).to(device);gate.load_state_dict(gate_ck['state']);gate.eval()
    final=_make_final_image_cache(UNSEEN,out/'external_features')
    res={};base_metrics={};split_rows={}
    for split in ('dev','eval'):
        prep=dict(np.load(POOL_DIR/f'W_inputs_{split}.npz'));all_rowids=np.asarray(prep['row'],np.int64);nfull=len(all_rowids)
        pos=np.arange(nfull,dtype=np.int64)
        if max_rows is not None and nfull>int(max_rows):
            # Contiguous short blocks keep the phase-shift history check meaningful.
            pos=smoke_positions(prep['task'],prep['episode'],prep['frame'],max_rows)
        rowids=all_rowids[pos];n=len(rowids)
        mids=np.load(PI/f'{split}_mid_v.npy',mmap_mode='r');hid=np.load(PI/f'{split}_hidden_t1.npy',mmap_mode='r');pact=np.load(PI/f'{split}_actions.npy',mmap_mode='r')
        d={'hist':np.asarray(prep['hist'][pos],np.float32),'state':np.asarray(prep['state'][pos,:8],np.float32),'task':np.asarray(prep['task'][pos],np.int64),
           'episode':np.asarray(prep['episode'][pos],np.int64),'frame':np.asarray(prep['frame'][pos],np.int64),'has_hist':np.asarray(prep['has_hist'][pos],bool),
           'pi05':np.asarray(pact[pos],np.float32),'expert':np.asarray(hid[pos],np.float32),
           'image':np.asarray(final[rowids]),'mid16':np.asarray(mids[pos,4,:128]),'mid17':np.asarray(mids[pos,5,:128])}
        _attach_predecessors(d)
        hard=np.asarray(prep.get('hard',np.zeros(nfull,bool))[pos],bool);hard_s=hard
        allrows=np.arange(n,dtype=np.int64)
        histories={'true':(d['hist'],allrows),'swap':(corrupt_history(d['hist'],'swap',seed=4101,task=d['task']),allrows),
            'reverse':(corrupt_history(d['hist'],'reverse'),allrows),'none':(corrupt_history(d['hist'],'zero'),allrows)}
        for shift in (-20,-10,10,20):histories[f'phase{shift:+d}']=phase_shift_history(d,shift)
        # Eval labels are dereferenced only below, after all model paths and selection have been validated.
        targets=np.asarray(r['actions'][rowids,:,:6],np.float32);err=_error_matrix(targets,pool)
        oracle=err.min(1);random=err.mean(1);pierr=_teacher_nn(d['pi05'],pool);pnn=-pierr
        prev=np.broadcast_to(d['hist'][:,-1:,:6],(n,10,6)).copy();prev[~d['has_hist']]=0;pi_pred=d['pi05'][:,:,:6]
        base_metrics[split]={'previous_action_mse6':float(so.six_axis(prev,targets).mean()),'pi05_mse6':float(so.six_axis(pi_pred,targets).mean()),
            'previous_action_hard_mse6':float(so.six_axis(prev[hard],targets[hard]).mean()) if hard.any() else None,
            'pi05_hard_mse6':float(so.six_axis(pi_pred[hard],targets[hard]).mean()) if hard.any() else None}
        split_rows[split]={'rows':n,'available_rows':nfull,'hard':int(hard.sum()),'smoke_subset':max_rows is not None};local={}
        for arch in ARCHES:
            for layer in (('final',) if _plain_arch(arch)=='P5' else tuple(layers)):
                for seed in seeds:
                    path=out/'models'/f'{source}_{arch}_{layer}_s{seed}.pt';rr=torch.load(path,weights_only=False)
                    net=make_model(arch,78,pool['key'].shape[1],_token_dim(d,layer),
                        expert_dim=int(d['expert'].shape[-1])).to(device);_load_state_compat(net,rr['state']);net.eval()
                    for hn,(hh,qids) in histories.items():
                        if len(qids)==0:continue
                        raw=_score_net(net,d,qids,layer,arch,pool['key'],history_override=hh[qids],device=device)
                        variants=[(arch,raw)]
                        if arch=='P5':
                            variants.append(('P5+pi05NN-fixed',soz(raw)+float(selection['fusion_weights'][source]['weight'])*soz(pnn[qids])))
                            gatefeat=_feature(d,qids,'final',hh[qids],'P5-imgdelta',device)
                            with torch.no_grad():gated,_=combine_with_gate(gate,torch.as_tensor(raw,device=device),torch.as_tensor(pnn[qids],device=device),gatefeat)
                            variants.append(('P5+pi05NN-gated',gated.cpu().numpy()))
                        for model_name,sc in variants:
                            er=err[qids];y=targets[qids];hq=hard[qids];pick=np.argmax(sc,axis=1);te=er[np.arange(len(qids)),pick]
                            pred=so._mods()[1].to_normalized(pool['phys'][pick]);mse6=so.six_axis(pred,y);sp=_spearman_rows(sc,er)
                            rank=_retrieval_row_metrics(sc,er);metrics={'spearman':float(np.nanmean(sp)),
                                'spearman_hard':float(np.nanmean(sp[hq])) if hq.any() else None,
                                'conditional_spearman_hard':float(np.nanmean(rank['conditional_spearman'][hq])) if hq.any() else None,
                                **{k:float(np.nanmean(v)) for k,v in rank.items()},
                                'strict_top1':float(np.mean(rank['recall@1'])),
                                'gap_recovered':float(np.mean(rank['gap@1'])),'mse6_stored':float(np.mean(mse6)),
                                'rows':int(len(qids))}
                            key=f'{source}|{model_name}|{layer}|{split}|{hn}'
                            local.setdefault(key,[]).append({'seed':int(seed),'history':hn,'metrics':metrics,
                                'mse6_rows':mse6.tolist(),'spearman_rows':sp.tolist(),'episode':(d['task'][qids]*1000+d['episode'][qids]).tolist()})
                if _plain_arch(arch)=='P5':
                    pick=np.argmin(pierr,axis=1);pred=so._mods()[1].to_normalized(pool['phys'][pick]);mse6=so.six_axis(pred,targets)
                    sp=_spearman_rows(pnn,err);rank=_retrieval_row_metrics(pnn,err)
                    metrics={'spearman':float(np.nanmean(sp)),'spearman_hard':float(np.nanmean(sp[hard])) if hard.any() else None,
                        'conditional_spearman_hard':float(np.nanmean(rank['conditional_spearman'][hard])) if hard.any() else None,
                        **{k:float(np.nanmean(v)) for k,v in rank.items()},'strict_top1':float(np.mean(rank['recall@1'])),
                        'gap_recovered':float(np.mean(rank['gap@1'])),'mse6_stored':float(np.mean(mse6)),'rows':int(n)}
                    local[f'{source}|pi05_nn|none|{split}|none']=[{'seed':int(seeds[0]),'history':'none','metrics':metrics,
                        'mse6_rows':mse6.tolist(),'spearman_rows':sp.tolist(),'episode':(d['task']*1000+d['episode']).tolist()}]
        res.update(local)
    # Paired episode bootstrap against the same-source P5 under the same history corruption.
    diffs={}
    for key,runs in res.items():
        bits=key.split('|')
        if len(bits)!=5 or bits[1] in ('P5','P5-imgdelta','P5+pi05NN-fixed','P5+pi05NN-gated','pi05_nn'):continue
        src,arch,layer,split,hn=bits;ref_arch=_plain_arch(arch);refkey=f'{src}|{ref_arch}|final|{split}|{hn}';ref=res.get(refkey,[])
        for rr in runs:
            p5=next((q for q in ref if q['seed']==rr['seed']),None)
            if p5 is None:continue
            diffs[f'{key}|seed={rr["seed"]}']={
                'spearman_delta':_episode_bootstrap_diff(rr['spearman_rows'],p5['spearman_rows'],rr['episode']),
                'mse6_delta':_episode_bootstrap_diff(rr['mse6_rows'],p5['mse6_rows'],rr['episode'])}
    ep=out/'evaluation.json';prev=json.loads(ep.read_text()) if ep.exists() else {}
    metrics_all=dict(prev.get('metrics',{}));metrics_all.update(res)
    base_all=dict(prev.get('baselines',{}));base_all.update(base_metrics)
    diff_all=dict(prev.get('p5_paired_episode_bootstrap',{}));diff_all.update(diffs)
    rows_all=dict(prev.get('split_rows',{}));rows_all.update(split_rows)
    write_json(ep,{'metrics':metrics_all,'baselines':base_all,'p5_paired_episode_bootstrap':diff_all,'target_actions_used_for':'ranking metrics only after models and selection fixed',
        'split_rows':rows_all,
        'model_selection_source':'L90X 16 tasks only','seeds':list(seeds)})
    return res

def soz(x):
    a=np.asarray(x,np.float32);return (a-a.mean(1,keepdims=True))/(a.std(1,keepdims=True)+1e-8)

def report(out):
    out=Path(out);ev=json.loads((out/'evaluation.json').read_text());sel=json.loads((out/'selection.json').read_text());lines=['# X1 smoke/production results','',
      'Eval labels were used only after model and fusion selection. Selection used the task-disjoint L90X selection split.','',
      f"L90X selection tasks: {sel['selected_tasks']}; pi05-NN selection Spearman={sel.get('pi05_nn_selection_spearman',float('nan')):.3f}; fixed P5+NN reference weights={sel.get('fusion_weights',{})}; dynamic gate={sel.get('fusion_gate',{})}.",'',
      '| config / split / history | rows | Spearman all | conditional Spearman all / hard | Spearman hard | R@1 / R@5 / R@20 | NDCG@10 | gap@1 / gap@5 | stored 6-axis MSE |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def mean_range(values):
        vv=np.asarray(values,np.float64);return f"{vv.mean():.3f} [{vv.min():.3f}, {vv.max():.3f}]"
    for key,runs in ev['metrics'].items():
        histories=sorted({x['history'] for x in runs},key=lambda h:(0 if h=='true' else 1 if h=='swap' else 2 if h=='reverse' else 3 if h=='none' else 4,h))
        for h in histories:
            rr=[x['metrics'] for x in runs if x['history']==h]
            if not rr:continue
            conf='/'.join(key.split('|')[1:-1])
            vals=[]
            for metric in ('conditional_spearman_hard','spearman','conditional_spearman','spearman_hard',
                           'recall@1','recall@5','recall@20','ndcg@10','gap@1','gap@5','mse6_stored'):
                a=[x[metric] for x in rr if x[metric] is not None]
                vals.append('—' if not a else mean_range(a))
            nrows=mean_range([x['rows'] for x in rr])
            lines.append(f"| {conf} / {h} | {nrows} | {vals[1]} | {vals[2]} / {vals[0]} | {vals[3]} | {vals[4]} / {vals[5]} / {vals[6]} | {vals[7]} | {vals[8]} / {vals[9]} | {vals[10]} |")
    lines+=['','## Paired episode-bootstrap deltas versus P5','',
        'Each cell reports seed mean [min, max] in the table above. Here Δ = configuration − P5 under the same history and seed; CI resamples task×episode groups.']
    for key,v in ev.get('p5_paired_episode_bootstrap',{}).items():
        if '|eval|' not in key:continue
        s=v['spearman_delta'];m=v['mse6_delta']
        lines.append(f"- `{key}`: Spearman Δ {s['mean']:+.3f} [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}], MSE6 Δ {m['mean']:+.3f} [{m['ci95'][0]:+.3f}, {m['ci95'][1]:+.3f}].")
    lines+=['','## Baselines','', '| split | prev action MSE6 (all / hard) | pi0.5 MSE6 (all / hard) |','|---|---:|---:|']
    for split,m in ev.get('baselines',{}).items():
        hard_prev='—' if m.get('previous_action_hard_mse6') is None else f"{m['previous_action_hard_mse6']:.4f}"
        hard_pi='—' if m.get('pi05_hard_mse6') is None else f"{m['pi05_hard_mse6']:.4f}"
        lines.append(f"| {split} | {m['previous_action_mse6']:.4f} / {hard_prev} | {m['pi05_mse6']:.4f} / {hard_pi} |")
    gatefile=ROOT/'image-query-01'/'gate'/'dev_results.json'
    if gatefile.exists():
        gate=json.loads(gatefile.read_text()).get('gate_dev',[])
        if gate:
            summary={h:float(np.mean([r[h]['spearman'] for r in gate])) for h in ('true','swap','reverse','zero')}
            lines += ['',f"Existing Dev-trained gate reference (I2b; not refit here), Eval Spearman: {summary}."]
    lines+=['','Temporal-context variants use the visual feature difference from the exact previous 10-frame window within the same episode. They also receive log image-difference norm, log executed-history/image-difference norm ratio, log history motion norm, and a predecessor-available flag.','',
      'P5+π0.5-NN is reported both with the selection-tuned fixed scalar and a query-conditioned gate trained on L90X fit tasks and epoch-selected on disjoint L90X selection tasks.','',
      f"The smoke trained on 4 L90X tasks × 2 demos, one seed, and two epochs. For metric-path verification only, it evaluated a deterministic contiguous subset ({ev.get('split_rows',{}).get('dev',{}).get('rows',0)} Dev / {ev.get('split_rows',{}).get('eval',{}).get('rows',0)} Eval rows, from 113 / 619 available); it does not establish full-scale performance.",'']
    (out/'report.md').write_text('\n'.join(lines))

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['extract-lib40','cache-teachers','train','evaluate','report','smoke']);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--rows-dir',type=Path);p.add_argument('--source',choices=['l90x','lib40','both'],default='l90x')
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch',type=int,default=16);p.add_argument('--threads',type=int,default=8);p.add_argument('--device',default='cuda')
    p.add_argument('--arch',default='all');p.add_argument('--layer',default='all');a=p.parse_args();out=a.out
    if a.stage=='extract-lib40':print(json.dumps(extract_lib40_final(out,batch=a.batch,threads=a.threads),ensure_ascii=False))
    elif a.stage=='cache-teachers':_ensure_teacher_cache(out)
    elif a.stage=='train':
        arch=ARCHES if a.arch=='all' else tuple(a.arch.split(','));layers=LAYERS if a.layer=='all' else tuple(a.layer.split(','))
        train_all(out,source=a.source,architectures=arch,layers=layers,epochs=a.epochs,batch=a.batch,device=a.device)
    elif a.stage=='evaluate':eval_sources(out,source=a.source,layers=LAYERS if a.layer=='all' else tuple(a.layer.split(',')),device=a.device)
    elif a.stage=='report':report(out)
    elif a.stage=='smoke':
        if a.rows_dir is None:raise ValueError('--rows-dir required for smoke')
        extract_l90x(a.rows_dir);prepare_l90x_features(a.rows_dir);_ensure_teacher_cache(out)
        train_all(out,source='l90x',architectures=ARCHES,layers=LAYERS,seeds=(4101,),epochs=2,batch=8,device=a.device)
        eval_sources(out,source='l90x',seeds=(4101,),layers=LAYERS,device=a.device,max_rows=16);report(out)
if __name__=='__main__':main()
