"""Spawn-safe CPU-only row scoring; ordered map preserves query IDs and seeds."""
STATE=None

def initialize(bank,decoded,traces,row_at,qtask,form,exclude_task,random_pick,seedbase):
    global STATE
    STATE=(bank,decoded,traces,row_at,qtask,form,exclude_task,random_pick,seedbase)

def select_row(item):
    import compressed_denoise as m
    bank,decoded,traces,row_at,qtask,form,exclude_task,random_pick,seedbase=STATE
    j,rid=item;key,offset=row_at[int(rid)]
    return m._select_sequence(bank,decoded,traces[key],offset,int(qtask[j]),form=form,
                             exclude_task=exclude_task,random_pick=random_pick,seed=seedbase+int(key[0])*100000+int(key[1]))
