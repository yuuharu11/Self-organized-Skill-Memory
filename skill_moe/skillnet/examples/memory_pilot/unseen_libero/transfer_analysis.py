"""TR1: read-only provenance analysis of saved retrieval/rollout logs, CPU only.

No policy inference, training, simulator, or image feature extraction is run.
Labels describe source task instructions, not learned labels or unit semantics.
"""
from __future__ import annotations
import argparse, ast, collections, csv, hashlib, json, math, os, re, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[5]
STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', REPO_ROOT / '.skillnet-state')).expanduser()
ROOT = (STATE_DIR / 'outputs/memory-pilot/unseen-libero')
RC = (STATE_DIR / 'outputs/robocasa-memory/latent-skill-01')
RUN = (STATE_DIR / 'outputs/memory-pilot/pi05-memory40-20261104/run-01')
EX1 = ROOT/'expert-ex1-01/production-03/scope-change-01'
CATEGORIES = ['same_task', 'same_verb_other_scene', 'other_verb',
              'same_verb_same_scene_other_task', 'same_verb_scene_unknown', 'unknown']
LABEL = {'same_task':'同タスク', 'same_verb_other_scene':'同動詞・別場面',
         'other_verb':'別動詞', 'same_verb_same_scene_other_task':'同動詞・同場面の別タスク',
         'same_verb_scene_unknown':'同動詞・場面不明', 'unknown':'不明'}
INPUTS = {}
def record(path):
    path = Path(path); s = path.stat()
    INPUTS[str(path)] = {'size':s.st_size, 'mtime_ns':s.st_mtime_ns}
    return path
def read(path): return json.loads(record(path).read_text())
def lines(path):
    result=[]
    for i, s in enumerate(record(path).read_text().splitlines()):
        try: result.append(json.loads(s))
        except json.JSONDecodeError:
            # An active writer may leave one final incomplete line. Never hide an interior error.
            if i != len(path.read_text().splitlines())-1: raise
    return result
def array(path): return np.load(record(path), allow_pickle=False)
def dump(path, x):
    path.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
def norm(x): return re.sub(r'[^a-z0-9]+', ' ', x.lower()).strip()
def verbs(x):
    x=norm(x); found=set()
    patterns={'turn_on':r'turn on|switch on', 'turn_off':r'turn off|switch off',
      'pick':r'pick up|pickplace|pick place', 'place':r'\bput\b|\bplace\b|pickplace|pick place',
      'open':r'\bopen\b', 'close':r'\bclose\b|\bshut\b', 'push':r'\bpush\b',
      'pull':r'\bpull\b', 'stack':r'\bstack\b'}
    for k,p in patterns.items():
        if re.search(p,x):found.add(k)
    return sorted(found)
def instruction(name):
    return re.sub(r'^(?:KITCHEN|LIVING_ROOM|STUDY)_SCENE\d+_', '', name).replace('_',' ').lower()
def scene(name, bddl=None):
    m=re.search(r'(KITCHEN|LIVING_ROOM|STUDY)_SCENE\d+',name)
    if m:return m[0]
    if bddl and bddl.exists():
        # An exact static layout signature, not a learned scene similarity label.
        t=record(bddl).read_text();t=re.sub(r'\(:language[^)]*\)', '',t)
        t=t.split('(:goal')[0]
        return 'BDDL-layout:'+hashlib.sha256(re.sub(r'\s+','',t).encode()).hexdigest()[:12]
    return None
def catalog():
    sys.path.insert(0,str(HERE))
    import compressed_denoise as cd
    meta=read(RUN/'data_manifest.json')
    rows=lines(Path(meta['source'])/'meta/tasks.jsonl')
    mappath=(REPO_ROOT / 'skill_moe/skillnet/third_party/libero/libero/libero/benchmark/libero_suite_task_map.py')
    mod=ast.parse(record(mappath).read_text())
    suites=ast.literal_eval(next(n.value for n in mod.body if isinstance(n,ast.Assign)))
    config=(STATE_DIR / 'cache/libero/config.yaml').read_text()
    bddlroot=Path(re.search(r'^bddl_files: (.*)$',config,re.M)[1])
    forty={}
    for r in rows:
        text=r['task']; matches=[(suite,n) for suite in ['libero_10','libero_goal','libero_object','libero_spatial']
             for n in suites[suite] if norm(instruction(n))==norm(text)]
        assert len(matches)==1,(text,matches)
        suite,name=matches[0]
        forty[int(r['task_index'])]={'name':name,'instruction':text,'scene':scene(name,bddlroot/suite/(name+'.bddl')),
                                  'verbs':verbs(text),'suite':suite}
    ninety={r['task_id']:{'name':r['name'],'instruction':r['instruction'],
             'scene':scene(r['name']), 'verbs':verbs(r['instruction'])}
             for r in read(ROOT/'baseline-01/decision.json')['tasks']}
    rcm=read(RC/'manifest.json');rc={}
    for i,name in enumerate(rcm['tasks']):
        split=re.sub(r'([a-z])([A-Z])',r'\1 \2',name)
        rc[i]={'name':name,'instruction':split,'scene':None,'verbs':verbs(split)}
    return forty,ninety,rc
def category(query, source):
    if not source:return 'unknown'
    if norm(query['instruction'])==norm(source['instruction']):return 'same_task'
    if not set(query['verbs']) & set(source['verbs']):return 'other_verb'
    if not query.get('scene') or not source.get('scene'):return 'same_verb_scene_unknown'
    return 'same_verb_same_scene_other_task' if query['scene']==source['scene'] else 'same_verb_other_scene'
def numeric_mean(xs):
    a=[float(x) for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(a)) if a else None
def fmt(x,d=3):return '—' if x is None else f'{x:.{d}f}'
def csvwrite(path,rows):
    if not rows:return
    cols=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})

def libero_logs(forty,ninety,pool,bank):
    methods={}; missing=[]; audit={}
    sources=[('L90 unit_08',ROOT/'closed-loop-memory-01/libero90_unit_08/memory/rollouts.jsonl','cluster95')]
    for prefix,folder in [('EX1',EX1/'closedloop-libero90'),
        ('EX2修正前',ROOT/'expert-ex2-01/production-04/closed-loop/libero90')]:
        for path in sorted(folder.glob('*/trials.jsonl')):sources.append((prefix+' '+path.parent.name,path,'bank4516'))
    for method,path,kind in sources:
        raw=lines(path);last={(int(r['task_id']),int(r['trial_index'])):r for r in raw}
        trials=[];events=[];invalid_source=0
        for (task,initial),r in sorted(last.items()):
            trial={'method':method,'domain':'LIBERO90','task':task,'initial':initial,
                   'success':bool(r.get('success',False)),'steps':r.get('env_steps'),
                   'technical_error':r.get('status')=='error','baseline_success':r.get('baseline_success'),
                   'source_log':str(path),'expected_queries':r.get('num_policy_queries')}
            if trial['baseline_success'] is None:trial['baseline_success']=BASELINE.get((task,initial))
            for qi,q in enumerate(r.get('selected_units_and_t0',[])):
                unit=q.get('unit',q.get('unit_id'))
                if unit is None:continue
                unit=int(unit);src=int(pool['source_task'][unit]) if kind=='cluster95' else int(bank['task'][unit])
                if q.get('unit_task',src)!=src:invalid_source+=1
                if kind=='bank4516' and 'unit_source' in q:
                    assert list(map(int,q['unit_source']))==bank['source'][unit].astype(int).tolist()
                ev={'method':method,'domain':'LIBERO90','task':task,'initial':initial,'q':qi,
                    'unit':unit,'pool':kind,'source_task':src,'category':category(ninety[task],forty[src]),
                    'success':trial['success'],'used':not q.get('fallback_pi05',False),
                    'score':q.get('cos_top1', (q.get('top_scores') or [None])[0]),
                    'confidence':q.get('p_top1'), 'predicted_error':q.get('confidence_error'),
                    'margin':q.get('margin'), 't0':q.get('t0'), 'source_log':str(path)}
                events.append(ev)
            trial['events']=[e for e in events if e['task']==task and e['initial']==initial]
            trials.append(trial)
        assert not invalid_source
        methods[method]={'events':events,'trials':trials,'domain':'LIBERO90','kind':'closed_loop'}
        audit[method]={'raw_trials':len(raw),'final_trials':len(last),'earlier_attempt_rows_not_counted':len(raw)-len(last),
          'final_technical_errors':sum(t['technical_error'] for t in trials),'source_id_mismatches':invalid_source}
    return methods,missing,audit

def rc_logs(rc):
    methods={};audit={}
    paths=[(f'RCX2 {mode}',RC/f'closed/{mode}') for mode in ['p0','p5','e_vlm_h']]
    paths += [('RCX6 latent',RC/'latent-heads/closed/six')]
    for method,folder in paths:
        result=read(folder/'eval/results.json');records=[r for r in result['records'] if int(r['trial'])<30]
        grouped=collections.defaultdict(dict);qraw=lines(folder/'queries.jsonl')
        for q in qraw:
            eid=q['episode_id'];task,trial,_=eid.split(':',2)
            if int(trial)>=30:continue
            key=(task,int(trial));client=q.get('client_id','unknown')
            grouped[key].setdefault(client,{})[int(q['query_index'])]=q
        events=[];trials=[];ambiguous=0;seed_mismatch=0
        for r in records:
            task_id=list(rc).index(next(i for i,x in rc.items() if x['name']==r['task']))
            key=(r['task'],int(r['trial']));sessions=grouped.get(key,{})
            trial={'method':method,'domain':'RoboCasa','task':task_id,'initial':int(r['trial']),
                   'success':bool(r.get('success',False)),'steps':r.get('steps'),'technical_error':r.get('status') not in ['complete',None],
                   'source_log':str(folder/'queries.jsonl'),'events':[]}
            # Result files do not identify a server-client session. Ambiguous retries are not assigned to outcomes.
            if len(sessions)!=1:
                ambiguous+=1;trials.append(trial);continue
            seq=next(iter(sessions.values()))
            for qi,q in sorted(seq.items()):
                if not q.get('units'):continue
                seed=int(q['episode_id'].split('seed-')[-1])
                if seed!=int(r['seed']):seed_mismatch+=1;continue
                u=q['units'][0];uid=int(u['unit']);src=int(u['task'])
                ev={'method':method,'domain':'RoboCasa','task':task_id,'initial':trial['initial'],'q':qi,
                    'unit':uid,'pool':'rc9897','source_task':src,'category':category(rc[task_id],rc[src]),
                    'success':trial['success'],'used':not q.get('fallback',False),'score':u.get('score'),
                    'confidence':None,'predicted_error':q.get('predicted_error'), 'margin':q.get('margin'),
                    't0':q.get('t0'),'source_log':trial['source_log']}
                events.append(ev);trial['events'].append(ev)
            trials.append(trial)
        assert seed_mismatch==0
        methods[method]={'events':events,'trials':trials,'domain':'RoboCasa','kind':'closed_loop'}
        audit[method]={'trials_0_29':len(records),'query_log_rows_snapshot':len(qraw),
           'trials_with_multiple_or_missing_client_sessions_not_joined':ambiguous,'seed_mismatches':seed_mismatch}
    return methods,audit

def offline(forty,ninety,bank):
    from scipy.spatial import cKDTree
    # Recover IDs only by matching saved top-1 actions to the exact saved serving bank.
    # Ambiguous candidate identities are retained as sets, never called uniquely identified.
    windows=bank['unit_windows'];tree=cKDTree(windows[:,:,:6].reshape(len(windows),-1))
    methods={};audit={}
    root=ROOT/'key-shared-01/attempt-27-production/final'
    for path in sorted(root.glob('*/*unseen_paired_predictions.npz')):
        z=array(path);method='KEY2-F '+path.parent.name
        pred=z['top1_actions'];target=z['target_actions'];events=[];ties=0;missing=0
        dist,idx=tree.query(pred[:,:,:6].reshape(len(pred),-1),k=1)
        for i,(distance,first) in enumerate(zip(dist,idx)):
            if distance>2e-5:missing+=1;ids=[]
            else:
                near=tree.query_ball_point(pred[i,:,:6].ravel(),r=max(2e-6,float(distance)+1e-7))
                ids=[j for j in near if np.max(np.abs(windows[j,:,:6]-pred[i,:,:6]))<2e-6]
            if not ids:cat='unknown';uid=-1;src=-1
            else:
                ids=sorted(ids);uid=int(ids[0]);src=int(bank['task'][uid]);ties+=len(ids)>1
                cats={category(ninety[int(z['task'][i])],forty[int(bank['task'][j])]) for j in ids}
                cat=next(iter(cats)) if len(cats)==1 else 'unknown'
            err=float(np.square(pred[i,:,:6].astype(np.float64)-target[i,:,:6]).sum()/70)
            events.append({'method':method,'domain':'LIBERO90','task':int(z['task'][i]),'initial':int(z['episode'][i]),
               'q':i,'unit':uid,'pool':'bank4516','source_task':src,'category':cat,'success':None,'used':True,
               'score':None,'confidence':None,'predicted_error':None,'margin':None,'t0':None,
               'arm_mse':err,'hard':bool(z['hard'][i]),'matching_units':ids,'source_log':str(path)})
        methods[method]={'events':events,'trials':[],'domain':'LIBERO90','kind':'offline'}
        audit[method]={'rows':len(pred),'unique_identity_rows':len(pred)-ties-missing,
           'multiple_candidate_matches':ties,'unmatched':missing,'max_nearest_L2_arm_difference':float(dist.max()),
           'score_or_confidence_saved':False,'recovery':'exact arm-action match to saved 4516-unit serving bank; no model rerun'}
    return methods,audit

def proportions(events):
    counts=collections.Counter(e['category'] for e in events);n=len(events)
    return {'queries':n,'counts':{k:counts[k] for k in CATEGORIES},
            'fraction':{k:counts[k]/n if n else None for k in CATEGORIES}}
def summarize(methods):
    out={}
    for name,m in methods.items():
        ev=m['events'];entry={'domain':m['domain'],'kind':m['kind'],**proportions(ev),
            'applied':proportions([e for e in ev if e['used']]),'trials':len(m['trials']),
            'trials_with_queries':sum(bool(t['events']) for t in m['trials']),
            'successes':sum(t['success'] for t in m['trials'])}
        for t in m['trials']:
            seq=t['events'];pairs=[(a,b) for a,b in zip(seq,seq[1:]) if b['q']==a['q']+1]
            ids=[e['unit'] for e in seq]
            longest=current=0;prev=None
            for uid in ids:current=current+1 if uid==prev else 1;longest=max(longest,current);prev=uid
            t.update(query_count=len(seq),switches=sum(a['unit']!=b['unit'] for a,b in pairs),
                     repeat_fraction=numeric_mean([a['unit']==b['unit'] for a,b in pairs]),
                     unique_units=len(set(ids)),longest_repeat=longest,
                     other_verb_fraction=numeric_mean([e['category']=='other_verb' for e in seq]),
                     same_task_fraction=numeric_mean([e['category']=='same_task' for e in seq]))
        if m['kind']=='closed_loop':
            entry['by_outcome']={}
            for success in [True,False]:
                ts=[t for t in m['trials'] if t['success']==success and t['events']]
                ee=[e for t in ts for e in t['events']]
                entry['by_outcome'][str(success)]={**proportions(ee),'trials':len(ts),
                  'trial_mean_other_verb_fraction':numeric_mean([t['other_verb_fraction'] for t in ts]),
                  'switches_mean':numeric_mean([t['switches'] for t in ts]),
                  'repeat_fraction_mean':numeric_mean([t['repeat_fraction'] for t in ts]),
                  'longest_repeat_mean':numeric_mean([t['longest_repeat'] for t in ts]),
                  **{k:numeric_mean([e[k] for e in ee]) for k in ['score','confidence','predicted_error','margin','t0']}}
            entry['task_success_fail_contrast']=[]
            for task in sorted({t['task'] for t in m['trials']}):
                ts=[t for t in m['trials'] if t['task']==task and t['events']]
                a=[t['other_verb_fraction'] for t in ts if t['success']];b=[t['other_verb_fraction'] for t in ts if not t['success']]
                if a and b:entry['task_success_fail_contrast'].append({'task':task,'success_minus_fail_other_verb':float(np.mean(a)-np.mean(b))})
        else:
            entry['MSE_by_origin']={}
            taskmean={task:np.mean([e['arm_mse'] for e in ev if e['task']==task]) for task in {e['task'] for e in ev}}
            for cat in CATEGORIES:
                ee=[e for e in ev if e['category']==cat]
                entry['MSE_by_origin'][cat]={'rows':len(ee),'arm_mse':numeric_mean([e['arm_mse'] for e in ee]),
                 'arm_mse_hard':numeric_mean([e['arm_mse'] for e in ee if e['hard']]),
                 'task_centered_arm_mse':numeric_mean([e['arm_mse']-taskmean[e['task']] for e in ee])}
        out[name]=entry
    return out

def plot_examples(out,methods,forty,ninety,rc,bank,rcbank):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import variable_shape_memory as vs
    # Predefined evidence search: successful, baseline-failed where known, source verbs disjoint;
    # representative unit ranked by use frequency. This is illustrative, not a statistical selection.
    candidates=[]
    for name,m in methods.items():
        if m['kind']!='closed_loop':continue
        for t in m['trials']:
            ee=[e for e in t['events'] if e['category']=='other_verb' and e['used']]
            if not t['success'] or not ee:continue
            freq=collections.Counter(e['unit'] for e in ee);unit,count=freq.most_common(1)[0]
            e=next(e for e in ee if e['unit']==unit)
            if e['pool']=='cluster95':continue
            candidates.append((t.get('baseline_success') is False,count/max(1,len(t['events'])),name,t,e))
    candidates.sort(key=lambda x:(not x[0],-x[1],x[2],x[3]['task'],x[3]['initial']))
    # Show distinct inference paths; do not let a known-invalid pre-fix EX2 path
    # dominate the illustrative evidence from the valid recorded paths.
    chosen=[]
    for prefix in ['EX1 ', 'RCX2 ', 'RCX6 ', 'EX2修正前 ']:
        match=next((c for c in candidates if c[2].startswith(prefix)),None)
        if match:chosen.append(match)
    examples=[];used=set()
    for _,frac,name,t,e in chosen:
        k=(e['domain'],e['task'])
        if k in used:continue
        used.add(k)
        if e['pool']=='rc9897':
            length=int(rcbank['length'][e['unit']]);a=rcbank['natural_actions'][e['unit'],:length,:6]
            src=rc[e['source_task']];query=rc[e['task']];unitlabel='normalized delta commands'
        else:
            a=vs.to_physical(bank['unit_windows'][e['unit'],:,:6]);length=int(bank['source'][e['unit'],3]-bank['source'][e['unit'],2])
            src=forty[e['source_task']];query=ninety[e['task']];unitlabel='unnormalized robot commands (not meters)'
        fig=plt.figure(figsize=(8,3.4));ax=fig.add_subplot(121,projection='3d')
        p=np.vstack([np.zeros(3),np.cumsum(a[:,:3],axis=0)])
        ax.plot(p[:,0],p[:,1],p[:,2],marker='o',ms=2);ax.scatter(*p[0],c='g');ax.scatter(*p[-1],c='r')
        ax.set(xlabel='dx cumulative',ylabel='dy cumulative',zlabel='dz cumulative',title=unitlabel)
        bx=fig.add_subplot(122);bx.plot(a[:,:3]);bx.legend(['x','y','z']);bx.set(xlabel='saved playback step',ylabel='translation command',title=f'unit {e["unit"]}, natural length {length}')
        title=name.replace('EX2修正前','EX2 pre-fix')
        fig.suptitle(f'{title}\nsource: {src["instruction"]}',fontsize=9);fig.tight_layout()
        file=f'example-{len(examples)+1}.png';fig.savefig(out/file,dpi=150);plt.close(fig)
        examples.append({'method':name,'task':e['task'],'initial':e['initial'],'source_task':e['source_task'],
          'unit':e['unit'],'query':query,'source':src,'natural_length':length,'query_indices':[x['q'] for x in t['events'] if x['unit']==e['unit']],
          'used_other_verb_query_fraction':frac,'success':True,'baseline_success':t.get('baseline_success'),
          'figure':file,'motion_sum_translation':a[:,:3].sum(0).tolist(),
          'interpretation':'cumulative commanded translation, not measured end-effector trajectory; not a causal intervention'})
        if len(examples)==4:break
    return examples

def top_units(methods,forty,ninety,rc,bank,rcbank,pool):
    import variable_shape_memory as vs
    result=[]
    for name,m in methods.items():
        for task in sorted({e['task'] for e in m['events']}):
            events=[e for e in m['events'] if e['task']==task and e['unit']>=0]
            for uid,count in collections.Counter(e['unit'] for e in events).most_common(3):
                e=next(e for e in events if e['unit']==uid);src=(rc if m['domain']=='RoboCasa' else forty)[e['source_task']]
                query=(rc if m['domain']=='RoboCasa' else ninety)[task]
                physical=False
                if e['pool']=='cluster95':a=pool['shape'][uid,:,:6];length=int(pool['length'][uid]);note='pose removed; original direction/magnitude unavailable in saved pool'
                elif e['pool']=='rc9897':length=int(rcbank['length'][uid]);a=rcbank['natural_actions'][uid,:length,:6];note='normalized policy delta, source frame';physical=True
                else:a=vs.to_physical(bank['unit_windows'][uid,:,:6]);length=int(bank['source'][uid,3]-bank['source'][uid,2]);note='unnormalized robot command, not meters; first10 saved playback incl. padding';physical=True
                net=a[:,:3].sum(0);axis=int(np.argmax(np.abs(net)))
                result.append({'method':name,'query_task':task,'query_instruction':query['instruction'],'unit':uid,
                    'queries':count,'query_fraction':count/max(1,len(events)),'source_task':e['source_task'],
                    'source_name':src['name'],'source_instruction':src['instruction'],'source_verbs':src['verbs'],
                    'category':e['category'],'natural_length':length,'translation_command_sum':net.tolist() if physical else None,
                    'translation_command_rms':float(np.sqrt(np.mean(a[:,:3]**2))) if physical else None,
                    'direction':('xyz'[axis]+('+' if net[axis]>=0 else '-')) if physical else None,
                    'source_units_available':physical,'note':note})
    return result

def candidate_baseline(methods, summary, forty, ninety, rc, pool, bank, rcbank):
    """Account for candidate composition; this is not a new retrieval experiment."""
    for name,m in methods.items():
        e=m['events'];qcounts=collections.Counter(x['task'] for x in e)
        source_ids=pool['source_task'] if name=='L90 unit_08' else (rcbank['task'] if m['domain']=='RoboCasa' else bank['task'])
        source_catalog=rc if m['domain']=='RoboCasa' else forty
        query_catalog=rc if m['domain']=='RoboCasa' else ninety
        by_task={}
        for task,n in qcounts.items():
            cats=collections.Counter(category(query_catalog[task],source_catalog[int(t)]) for t in source_ids)
            by_task[task]={'candidate_count':len(source_ids),'counts':dict(cats),
                'fractions':{k:cats[k]/len(source_ids) for k in CATEGORIES}}
        total=sum(qcounts.values())
        weighted={k:sum(qcounts[t]*by_task[t]['fractions'][k] for t in qcounts)/total if total else None for k in CATEGORIES}
        summary[name]['uniform_candidate_composition']={'query_weighted_fraction':weighted,'by_task':by_task,
            'note':'available candidates with equal unit weight, not a sampled policy; LIBERO90 serving has no task exclusion'}
        summary[name]['compound_source_query_fraction']=numeric_mean([
            len(source_catalog[x['source_task']]['verbs'])>1 for x in e if x['source_task'] in source_catalog])
        summary[name]['successful_trials_with_other_verb_applied']=sum(
            t['success'] and any(x['used'] and x['category']=='other_verb' for x in t['events']) for t in m['trials'])

def render(out,r):
    mem=r['summary']['L90 unit_08']
    ex1=[v['by_outcome']['True']['fraction']['other_verb'] for k,v in r['summary'].items() if k.startswith('EX1 ')]
    memory_sentence=(f"43/100の単位メモリでは別動詞由来の問い合わせが全体{100*mem['fraction']['other_verb']:.1f}%、"
      f"成功試行{100*mem['by_outcome']['True']['fraction']['other_verb']:.1f}%、失敗試行{100*mem['by_outcome']['False']['fraction']['other_verb']:.1f}%。"
      f"EX1の成功試行では同動詞由来が約{100*(1-max(ex1)):.0f}–{100*(1-min(ex1)):.0f}%を占める。"
      '別動詞由来を使った成功例は存在する一方、成功が別動詞転用に強く依存するという結果ではない。これらはタスク・試行長の交絡を含む記述値。')
    L=['# TR1：atomicタスクでの単位の転用', '', '実行: 完了（既存ログのみ、CPU。学習・方策推論・閉ループの追加実行なし）', '',
       '## 集計規則と限界', '',
       '出どころは元タスクの指示文で判定する。同タスクは正規化した指示文の完全一致。同動詞は語の集合が交わる場合（put/placeは統合、pickは別、turn_on/offは別）。複合指示は全動詞を使うため、単位そのものの意味・実行段階の正解ラベルではない。',
       '場面は明示されたSCENE名、その他はBDDLの言語・goalを除いた静的配置の署名で判定する。物体・配置が似ている程度の類似性は測っていない。同動詞・同場面の別タスク、場面不明は強制的に別動詞へ含めない。',
       '割合は問い合わせを分母とする。成功失敗比較には試行を等重みとした別動詞率も併記する。成功試行は短く、タスク難度も異なるため、全問い合わせの割合差を因果効果と解釈しない。',
       'RoboCasa atomic 6種はメモリ作成・学習に含まれる既知タスクの新エピソードであり、未学習タスクへの汎化の証拠ではない。RCX2/6は各タスクtrial 0–29に固定。units配列は時間順の再生計画であり上位k候補ではない。先頭の1単位だけを問い合わせの出どころとして数え、未実行の将来単位を重複計上しない。',
       'EX2は依頼どおりproduction-04（修正前）のログに限定した。動作経路の既知の修正・再学習後の性能とは区別し、出どころの記述にのみ使用する。', '',
       '## 1. 出どころ（問い合わせ比率）', '',
       '|方式|試行/記録あり|問い合わせ|実際に適用|同タスク|同動詞・別場面|別動詞|同動詞・同場面別task|場面不明/不明|',
       '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,s in r['summary'].items():
        f=s['fraction'];p=lambda k:'—' if f[k] is None else f'{100*f[k]:.1f}%'
        unknown=sum(f[k] or 0 for k in ['same_verb_scene_unknown','unknown'])
        L.append(f"|{name}|{s['trials']}/{s['trials_with_queries']}|{s['queries']}|{s['applied']['queries']}|{p('same_task')}|{p('same_verb_other_scene')}|{p('other_verb')}|{p('same_verb_same_scene_other_task')}|{100*unknown:.1f}%|")
    L+=['', 'fallback時にも検索IDが記録されるため、表は検索された出どころ、適用列はfallbackを除く件数。適用した問い合わせだけの出どころもresults.jsonのappliedに保存した。', '',
        '### 候補プール構成との比較', '',
        '|方式|別動詞の検索率|候補を一様に数えた別動詞率|複合動詞の元タスク由来|成功試行中に別動詞を適用した試行|',
        '|---|---:|---:|---:|---:|']
    for name,s in r['summary'].items():
        f=s['uniform_candidate_composition']['query_weighted_fraction']
        L.append(f"|{name}|{fmt(s['fraction']['other_verb'])}|{fmt(f['other_verb'])}|{fmt(s['compound_source_query_fraction'])}|{s['successful_trials_with_other_verb_applied']}/{s['successes']}|")
    L+=['', '候補比率は、その方式で実際に生じた問い合わせのタスク分布で重み付けした一様候補の構成。未実行のランダム方策の成功率ではない。LIBERO-90は候補に同タスク・同じSCENE名がほぼ存在しないようにタスクが選ばれており、別場面からの参照という事実だけで場面不変性は示せない。BDDL署名が異なっても視覚・配置の類似は残りうる。', '',
        '## 2. 成功試行と失敗試行', '',
        '|方式|成否|試行|別動詞率（問い合わせ/試行等重み）|同タスク率|score|p_top1|予測誤差|切替回数/試行|直前単位の反復率|最長反復/試行|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,s in r['summary'].items():
        for outcome,b in s.get('by_outcome',{}).items():
            p=b['fraction'];L.append(f"|{name}|{'成功' if outcome=='True' else '失敗'}|{b['trials']}|{fmt(p['other_verb'])}/{fmt(b['trial_mean_other_verb_fraction'])}|{fmt(p['same_task'])}|{fmt(b['score'])}|{fmt(b['confidence'])}|{fmt(b['predicted_error'])}|{fmt(b['switches_mean'],1)}|{fmt(b['repeat_fraction_mean'])}|{fmt(b['longest_repeat_mean'],1)}|")
    L+=['', 'スコアは方式ごとにcosine・学習済みlogitなどが異なり、方式間で直接比較しない。切替は連続する問い合わせ番号だけを比較する。繰り返しは再利用・保持にも必要であり、それだけで停滞とは判定しない。タスク内の成功−失敗別動詞率もresults.jsonに保存した。', '',
        '## 3. タスクごとによく使う単位', '',
        '方式×タスクの上位3単位を [top-units.csv](top-units.csv) に全件保存。以下は閉ループの各タスクの1位。方向は保存されたソースの並進行動コマンドの主軸、長さは自然長。95型プールは向き・大きさを除いて保存されているため元方向は不明とする。', '',
        '|方式|タスク|単位|出どころのタスク|割合|自然長|主方向|並進RMS|', '|---|---:|---:|---|---:|---:|---|---:|']
    seen=set()
    for x in r['top_units']:
        key=(x['method'],x['query_task'])
        if key in seen or x['method'].startswith('KEY2'):continue
        seen.add(key);L.append(f"|{x['method']}|{x['query_task']}|{x['unit']}|{x['source_name']}|{100*x['query_fraction']:.1f}%|{x['natural_length']}|{x['direction'] or '不明'}|{fmt(x['translation_command_rms'],4)}|")
    for i,x in enumerate(r['examples'],1):
        L += ['', f"### 例{i}：{x['method']}、task {x['task']}、initial {x['initial']}", '',
              f"成功したタスク「{x['query']['instruction']}」で、動詞が重ならない「{x['source']['instruction']}」由来の単位{x['unit']}を使用。問い合わせ番号 {x['query_indices']}。同初期状態の通常π0.5は {x['baseline_success']}（Noneは照合値なし）。これは成功中に使われた例であり、その単位が成功を引き起こしたことは未確認。", '',
              f"![ソース単位の累積並進コマンド]({x['figure']})", '', '図は行動コマンドを積算した軌跡であり、実測した手先位置ではない。LIBEROの逆正規化値もロボット制御入力であり、メートルへ変換した変位ではない。π0.5への姿勢合わせやExpert後に実際に実行された軌跡でもない。']
    L += ['', '## 4. 未学習オフラインの腕MSEと出どころ', '',
          '腕6軸二乗和÷70。全体/難の値。task中心化は同じ方式・同じタスクの平均誤差を引いたもので、タスク構成の交絡を見る参考。閉ループで実際に経験した状態の誤差ではない。', '',
          '|方式|出どころ|行数|腕MSE|難の腕MSE|task中心化差|', '|---|---|---:|---:|---:|---:|']
    for name,s in r['summary'].items():
        for cat,v in s.get('MSE_by_origin',{}).items():
            if v['rows']:L.append(f"|{name}|{LABEL[cat]}|{v['rows']}|{fmt(v['arm_mse'],5)}|{fmt(v['arm_mse_hard'],5)}|{fmt(v['task_centered_arm_mse'],5)}|")
    L += ['', '## 記録の不足・照合', '', *['- '+s for s in r['limitations']], '',
          '## 解釈', '',
          memory_sentence,
          'KEY2-Fでは別動詞群の生の平均MSEが同動詞群より高いが、タスク内平均を引くと符号が反転する方式もある（P5など）。評価タスクの構成に影響されており、別動詞由来そのものが誤差を増やすという因果的解釈はできない。同タスク由来の行が0なので、同タスク対別タスクの直接比較はできない。',
          '別動詞の元タスク由来の単位を使いながら成功した具体例は上記のとおり。ただし元の複合タスクの動詞と、短い単位が表す運動は同じ概念ではない。共有の到達・保持・把持前後の動きが、異なるタスク名から引かれることはありうる。',
          '元タスクが異なること、別動詞由来の問い合わせが成功試行にも存在することはログで確認できる。一方「その単位がなければ失敗した」「類似場面ではなく運動の意味を捉えた」は、この観察分析からは言えない。π0.5の姿勢・グリッパー・Expert・fallbackの寄与も分離していない。',
          '研究の主張としては、ラベルを選択器へ与えずに切り出した単位がタスク境界を越えて参照・適用された事実まで。別タスク転用が成功の原因であることや、新規の運動構成への汎化を示すには、単位/場面を統制した対応介入が必要。今回は追加実行しない。', '',
          '## 再現', '', f"`OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES='' taskset -c 16-23 nice -n 10 python {Path(__file__).resolve()} --output 新しい出力先`", '',
          '[results.json](results.json)・[queries.csv](queries.csv)・[trials.csv](trials.csv)・[入力台帳](inputs.json)', '']
    (out/'report.md').write_text('\n'.join(L))

BASELINE={}
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'transfer-analysis-01');a=ap.parse_args();out=a.output
    out.mkdir(exist_ok=True,parents=True)
    if (out/'results.json').exists() or (out/'report.md').exists():raise FileExistsError('refusing to overwrite completed outputs')
    import torch;torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'CPU-only: set CUDA_VISIBLE_DEVICES empty'
    forty,ninety,rc=catalog()
    for r in lines(ROOT/'baseline-01/production/rollouts.jsonl'):BASELINE[(int(r['task_id']),int(r['trial_index']))]=bool(r['success'])
    pool=array(ROOT/'cluster-pools-01/pool_unit_08.npz')
    bank=array(EX1/'eval-drafts/closedloop_p5_bank.npz');rcbank=array(RC/'units/memory_units.npz')
    print('catalog and banks loaded',flush=True)
    methods,missing,audit=libero_logs(forty,ninety,pool,bank)
    rc_methods,rcaudit=rc_logs(rc);methods.update(rc_methods);audit.update(rcaudit)
    print('closed-loop logs joined',flush=True)
    off,offaudit=offline(forty,ninety,bank);methods.update(off);audit.update(offaudit)
    summary=summarize(methods)
    candidate_baseline(methods,summary,forty,ninety,rc,pool,bank,rcbank)
    examples=plot_examples(out,methods,forty,ninety,rc,bank,rcbank)
    top=top_units(methods,forty,ninety,rc,bank,rcbank,pool)
    limitations=[
      'KEYGRID production-04のresults.jsonは全体指標だけを保存し、_per_queryを明示的に削除している。単位ID・予測・スコアの保存がないため出どころ/MSEの対応は不明。再推論して埋めてはいない。',
      'KEY2-Fは単位IDを保存せずtop1_actionsだけ保存。保存済み4516単位の腕行動と一致する候補集合を復元した。同一行動の複数候補は候補集合を保持し、出どころ分類が異なるときは不明とする。top-unitsの代表IDは最小IDの表示例であり一意復元の主張ではない。',
      'KEY2-Fの問い合わせ別スコア・信頼度は保存されていない。KEYGRID/KEY2の学習や設定選択は一切行っていない。',
      '95型プールに保存された出どころはsource_taskと自然長のみ。元デモID・開始位置・ソースの姿勢と大きさがなく、元向きの軌跡は描けない。EX1/EX2とRoboCasaの保存単位で具体例を示した。',
      'LIBEROのtrials.jsonlは同じ(task,initial)の最後の記録だけを使い、過去の技術エラーや再試行を二重計上しない。監査に元の行数を保存。最終のエラーは試行分母に含めるが検索記録のない行に出どころを補わない。',
      'RoboCasaのresultはclient_idを持たない。複数/欠損client sessionがある試行は成否と問い合わせを確実に結べないので出どころ比較から除外し、数を監査に残した。問い合わせログは実行中に追記されうるため読み込み時点のスナップショットを使った。',
      'グリッパー切替、対象への接近、段階達成などのシミュレータ状態はこの分析で取得しない。源タスク名に含まれる動詞の一致は、局面に適した運動であることの証明ではない。',
    ]
    result={'status':'complete','created_utc':datetime.now(timezone.utc).isoformat(),'catalog':{'libero40':forty,'libero90':ninety,'robocasa':rc},
      'summary':summary,'top_units':top,'examples':examples,'join_audit':audit,'limitations':limitations,
      'execution':{'threads':torch.get_num_threads(),'affinity':sorted(os.sched_getaffinity(0)),'nice':os.getpriority(os.PRIO_PROCESS,0),
        'GPU':False,'training':False,'policy_inference':False,'new_rollouts':False,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    dump(out/'results.json',result);dump(out/'inputs.json',INPUTS)
    csvwrite(out/'queries.csv',[e for m in methods.values() for e in m['events']])
    csvwrite(out/'trials.csv',[{k:v for k,v in t.items() if k!='events'} for m in methods.values() for t in m['trials']])
    csvwrite(out/'top-units.csv',top)
    render(out,result)
    (out/'DONE').write_text('complete\n')
    print(json.dumps({'status':'complete','methods':len(methods),'queries':sum(len(m['events']) for m in methods.values()),'examples':len(examples)}),flush=True)
if __name__=='__main__':main()
