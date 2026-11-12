import json, collections, pathlib, statistics
ROOT=pathlib.Path('/work/.skillnet-state/outputs/robocasa-memory')
EXP=ROOT/'expert-02/eval300'
BASE=ROOT/'baseline-01/v6-nosubtask/eval/results.json'
EVAL=EXP/'eval/results.json'; OLD=ROOT/'expert-02/closedloop-v2/run-01/eval/results.json'
TASKS=['PickPlaceCounterToCabinet','PickPlaceSinkToCounter','OpenDrawer','CloseFridge','TurnOnSinkFaucet','TurnOffStove']

def records(p): return json.load(open(p))['records']
base={(r['task'],int(r['trial'])):r for r in records(BASE)}
new={(r['task'],int(r['trial'])):r for r in records(EVAL)}
old={(r['task'],int(r['trial'])):r for r in records(OLD)}
rep={}
for t in TASKS[:1]+TASKS[2:4]:
 shared=[i for i in range(12) if (t,i) in old and (t,i) in new]
 rep[t]={'n':len(shared),'discordant':sum(bool(old[t,i]['success'])!=bool(new[t,i]['success']) for i in shared), 'success_old':sum(bool(old[t,i]['success']) for i in shared),'success_eval300':sum(bool(new[t,i]['success']) for i in shared)}
print('repeatability',rep)
# Flat runtime selected unit events with trial result grouping.
events=collections.defaultdict(list); firsts=collections.defaultdict(list)
with (EXP/'queries.jsonl').open() as f:
 for line in f:
  try:r=json.loads(line)
  except Exception:continue
  if r.get('record_type')!='memory_bridge_query':continue
  task,trial,seed=r['episode_id'].rsplit(':',2); trial=int(trial)
  cls='both_success' if base[(task,trial)]['success'] and new[(task,trial)]['success'] else ('baseline_only_success' if base[(task,trial)]['success'] and not new[(task,trial)]['success'] else ('expert_only_success' if new[(task,trial)]['success'] and not base[(task,trial)]['success'] else 'both_fail'))
  units=r.get('selected_units',[])
  for u in units: events[(task,cls)].append(int(u['task_id']))
  if units:firsts[(task,cls,trial)].append((int(r.get('query_index',0)),int(units[0]['unit_id']),int(units[0]['task_id'])))
summary={}
for t in TASKS:
 summary[t]={}
 for cls in ('baseline_only_success','both_success','expert_only_success','both_fail'):
  ev=events.get((t,cls),[])
  wrong=sum(u!=TASKS.index(t) for u in ev)
  seqs=[]
  for (tt,cc,tr),xs in firsts.items():
   if tt==t and cc==cls:
    xs=sorted(xs)
    seqs.append([x[1] for x in xs])
  pairs=[(a,b) for s in seqs for a,b in zip(s,s[1:])]
  summary[t][cls]={'selected_unit_events':len(ev),'wrong_task':wrong,'wrong_fraction':wrong/len(ev) if ev else None,
                   'consecutive_query_pairs':len(pairs),'same_first_unit_pairs':sum(a==b for a,b in pairs),
                   'same_first_unit_fraction':sum(a==b for a,b in pairs)/len(pairs) if pairs else None}
first_summary={}
for t in TASKS:
 first_summary[t]={}
 for cls in ('baseline_only_success','both_success','expert_only_success','both_fail'):
  arr=[x for (tt,cc,tr),xs in firsts.items() if tt==t and cc==cls for x in xs]
  wrong=sum(x[2]!=TASKS.index(t) for x in arr)
  first_summary[t][cls]={'queries':len(arr),'wrong_task':wrong,'wrong_fraction':wrong/len(arr) if arr else None,
                         'same_unit_consecutive':summary[t][cls]['same_first_unit_pairs'],
                         'consecutive_query_pairs':summary[t][cls]['consecutive_query_pairs']}
initial={}
for t in TASKS:
 vals=[]
 with (EXP/'queries.jsonl').open() as f:
  for line in f:
   try:r=json.loads(line)
   except:continue
   if r.get('record_type')=='memory_bridge_query' and r.get('query_index')==0 and r['episode_id'].startswith(t+':') and r.get('selected_units'):
    vals.append(int(r['selected_units'][0]['task_id']))
 initial[t]={'n':len(vals),'first_selected_task_ids':dict(collections.Counter(vals)), 'most_common':collections.Counter(vals).most_common(5)}
print('query selection stats; wrong_fraction_all_planned_units and first_unit_wrong_fraction separated')
print(json.dumps({'planned_unit_events':summary,'first_selected_unit_by_outcome':first_summary,'episode_start_first_selected_unit_counts':initial},indent=2))
# Task map in source manifest (task_id maps exactly to list order).
manifest=json.load(open(ROOT/'memory-01/policy-order-v2/manifest.json'))
print('manifest tasks',manifest['tasks'])
(ROOT/'expert-02/gap/log_summary.json').write_text(json.dumps({'repeatability':rep,'query_selection':summary,'first_selected_by_outcome':first_summary,'episode_start_counts':initial,'task_id_map':{i:t for i,t in enumerate(manifest['tasks'])}},indent=2)+'\n')
