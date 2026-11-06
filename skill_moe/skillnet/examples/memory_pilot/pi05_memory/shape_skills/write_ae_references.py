#!/usr/bin/env python3
"""Add Train-only Procrustes references to rebuilt AE bundles."""
import os
from pathlib import Path
import numpy as np
import ae

def main():
    root=Path(os.environ.get('SKILLNET_PI05_ROOT',ae.PI05_ROOT)).expanduser()/'shape-skills-02/representations'
    episodes,rows,_,segsets,_=ae._load_splits_and_segments()
    out=[]
    for segname in ae.SEGMENTATIONS:
        segs=[s for s in segsets[segname] if s['task'] not in ae.LONG|ae.HELDOUT
              and (s['task'],s['episode']) in episodes and s['end']>s['start']]
        train=[s for s in segs if ae._is_train(s,episodes)]
        if not train or any(s['task'] in ae.LONG|ae.HELDOUT for s in train):
            raise AssertionError(f'invalid AE Train split: {segname}')
        ref=ae._make_procrustes_reference(train,episodes)
        for variant in ('procrustes_shape','raw_invariant'):
            for dim in ae.LATENT_DIMS:
                path=root/f'ae_{segname}_{variant}_d{dim}'/'procrustes_reference.npy'
                if not path.parent.joinpath('manifest.json').exists():
                    raise FileNotFoundError(path.parent)
                if path.exists():
                    old=np.load(path,allow_pickle=False)
                    if not np.array_equal(old,ref):raise AssertionError(f'reference mismatch {path}')
                else:np.save(path,ref,allow_pickle=False)
                out.append(str(path))
    print({'reference_sidecars':len(out),'shape':ref.shape,'train_tasks_excluded':sorted(ae.LONG|ae.HELDOUT)})
if __name__=='__main__':main()
