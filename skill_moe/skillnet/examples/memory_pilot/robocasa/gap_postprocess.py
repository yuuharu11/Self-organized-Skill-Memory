import json, numpy as np
from pathlib import Path
from openpi.shared import normalize
ROOT=Path('/work/.skillnet-state/outputs/robocasa-memory')
OUT=ROOT/'expert-02/gap'; q=np.load(ROOT/'memory-01/policy-order-v2/queries/U_heldout.npz',allow_pickle=False)
d=np.load(OUT/'offline_predictions.npz',allow_pickle=False)
rows=json.loads((OUT/'offline_comparison.json').read_text())
stats=normalize.load(Path('/work/.skillnet-state/checkpoints/robocasa365/pi05_pretrain_human300/multitask_learning/75000/assets'))['actions']
mean=np.asarray(stats.mean,np.float32);std=np.asarray(stats.std,np.float32)
idx=np.asarray(d['row_ids'],np.int64); tasks=np.asarray(q['task'])[idx]
length=np.asarray(d['length'],np.int32)
target=np.asarray(d['target'],np.float32)
series={'target':target,'retrieved_draft':np.asarray(d['retrieved'],np.float32),'oracle_draft':np.asarray(d['oracle'],np.float32),
        'retrieved_expert':np.asarray(d['pred_retrieved'],np.float32),'oracle_expert':np.asarray(d['pred_oracle'],np.float32),'pi05_normal':np.asarray(d['pred_pi05'],np.float32)}

def denorm(x): return x*(std[None,None,:32]+1e-6)+mean[None,None,:32]
# Axis-wise EEF6 MSE and grip timing for heldout SinkFaucet and two pick-place tasks.
axis={}; grip={}
for ti in (0,1,4):
 sel=np.flatnonzero(tasks==ti);axis[('task',ti)]={}
 for name,x in series.items():
  xx=denorm(x); e=(np.asarray(x,np.float32)[sel,:,:6]-target[sel,:,:6])**2
  L=np.minimum(length[sel],50)
  first=e[:,:5].mean(axis=(0,1))
  unit=np.asarray([e[j,:int(L[j])].mean(axis=0) for j in range(len(sel))]).mean(axis=0)
  axis[('task',ti)][name]={'first5_axis_mse6':first.tolist(),'unit_axis_mse6':unit.tolist(), 'first5_pos_mean':float(first[:3].mean()),'first5_rot_mean':float(first[3:].mean()),'unit_pos_mean':float(unit[:3].mean()),'unit_rot_mean':float(unit[3:].mean())}
 # Discrete sign transition proxy; only for windows with at least one target transition in 50.
 truth=denorm(target)[sel,:,6]
 tgt=(truth>=0).astype(int)
 def firstflip(z):
  return next((int(j) for j in range(1,len(z)) if z[j]!=z[0]),None)
 valid=[]
 for j in range(len(sel)):
  f=firstflip(tgt[j]);
  if f is not None: valid.append((j,f))
 grip[('task',ti)]={'rows':len(sel),'target_switch_rows':len(valid),'timing_by_condition':{}}
 for name,x in series.items():
  z=(denorm(x)[sel,:,6]>=0).astype(int)
  switches=[firstflip(z[j]) for j,_ in valid]
  diffs=[abs(s-f) for (_,f),s in zip(valid,switches) if s is not None]
  grip[('task',ti)]['timing_by_condition'][name]={'target_switch_rows':len(valid),'pred_switch_found':sum(s is not None for s in switches),
      'mean_abs_first_switch_steps_on_found':float(np.mean(diffs)) if diffs else None,'missed_switches':sum(s is None for s in switches),
      'false_switch_rate_target_no_switch':float(np.mean([firstflip(z[j]) is not None for j in range(len(sel)) if firstflip(tgt[j]) is None])) if any(firstflip(tgt[j]) is None for j in range(len(sel))) else None}
tasks=['PickPlaceCounterToCabinet','PickPlaceSinkToCounter','OpenDrawer','CloseFridge','TurnOnSinkFaucet','TurnOffStove']
axis_out={f'{k[1]}:{tasks[k[1]]}':v for k,v in axis.items()}
grip_out={f'{k[1]}:{tasks[k[1]]}':v for k,v in grip.items()}
rows['axis_mse_detail']=axis_out;rows['gripper_switch_timing_proxy']=grip_out
(OUT/'offline_comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps({'axis':axis_out,'gripper':grip_out},indent=2))
