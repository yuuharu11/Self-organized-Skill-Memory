"""Unsupervised RoboCasa motion-unit extraction adapter."""
from __future__ import annotations
import argparse, json, pathlib
from functools import lru_cache
import numpy as np

def cut(actions:np.ndarray, *, gripper=True, base_motion_boundaries=False, min_length=10):
    """Return half-open segments using gripper switches and smoothed EEF-speed minima.

    `actions` is [time,12] in the published RoboCasa order. Base/torso channels are
    retained separately and are not used as boundaries unless explicitly requested.
    No task/subtask labels or success signals are read.
    """
    x=np.asarray(actions,dtype=np.float32)
    if x.ndim!=2 or x.shape[1]<12: raise ValueError('LeRobot actions must have shape [T,12]')
    n=len(x)
    if n==0:return []
    switches=set((np.flatnonzero(np.sign(x[1:,11])!=np.sign(x[:-1,11]))+1).tolist()) if gripper else set()
    signals=[np.linalg.norm(x[:,5:8],axis=1)]
    if base_motion_boundaries: signals.append(np.linalg.norm(x[:,:4],axis=1))
    minima=set()
    if n>=3:
        # Match the anchor-span rule used by LIBERO: 3-frame smoothing,
        # local minima, and a minimum interval unless a gripper switch forces it.
        for speed in signals:
            smooth=np.convolve(speed,np.ones(3,dtype=np.float32)/3,mode='same')
            minima.update((np.flatnonzero((smooth[1:-1]<=smooth[:-2])&
                (smooth[1:-1]<smooth[2:]))+1).tolist())
    kept=[0]
    for t in sorted(minima|switches):
        if t in switches or t-kept[-1]>=min_length:kept.append(int(t))
    # A short terminal tail is folded into the preceding unit unless a
    # gripper transition deliberately established the last boundary.
    if n-kept[-1]<min_length and len(kept)>1 and kept[-1] not in switches:kept.pop()
    seg=list(zip(kept,kept[1:]+[n]))
    rows=[]
    for s,e in seg:
        # DCT-II coefficients summarize type only; raw action channels are preserved.
        arm=np.concatenate([x[s:e,5:11],x[s:e,11:12]],axis=-1); res=np.zeros((10,7),np.float32)
        phase=np.linspace(0,len(arm)-1,10).round().astype(int); res[:]=arm[phase]
        from scipy.fft import dct as dct_ii
        coeff=dct_ii(res,type=2,norm='ortho',axis=0)[:6].reshape(-1)
        rows.append({'start':s,'end':e,'length':e-s,'type_dct':coeff.tolist(),
                     'theta':{'eef_translation':x[s:e,5:8].sum(0).tolist(),'eef_rotation':x[s:e,8:11].sum(0).tolist()},
                     'arm_actions':arm.tolist(),'other_actions':x[s:e,:5].tolist()})
    return rows

def load_episode(path, episode):
    """Read one episode action stream from parquet without requiring RoboCasa imports."""
    import pyarrow.compute as pc, pyarrow.parquet as pq
    all_rows=_load_action_table(str(pathlib.Path(path).resolve()))
    table=all_rows.filter(pc.equal(all_rows['episode_index'],int(episode)))
    return np.stack(table['action'].to_pylist())

@lru_cache(maxsize=2)
def _load_action_table(root):
    import pyarrow.parquet as pq
    tables=[pq.read_table(p,columns=['episode_index','action']) for p in sorted((pathlib.Path(root)/'data').glob('**/*.parquet'))]
    return __import__('pyarrow').concat_tables(tables)

def main():
    p=argparse.ArgumentParser(); p.add_argument('dataset'); p.add_argument('--base-motion-boundaries',action='store_true'); a=p.parse_args()
    root=pathlib.Path(a.dataset); units=[]
    for task_dir in sorted((root/'extras').glob('episode_*')):
        meta=json.loads((task_dir/'ep_meta.json').read_text()); actions=load_episode(root,int(task_dir.name.split('_')[-1]))
        for unit in cut(actions,base_motion_boundaries=a.base_motion_boundaries): units.append({'episode':task_dir.name,'task':meta.get('task_name',meta.get('env_name','unknown')),'unit':unit})
    lengths=[x['unit']['length'] for x in units]
    print(f'units={len(units)} length_median={float(np.median(lengths)) if lengths else 0:.1f} p10={float(np.quantile(lengths,.1)) if lengths else 0:.1f} p90={float(np.quantile(lengths,.9)) if lengths else 0:.1f}')
if __name__=='__main__': main()
