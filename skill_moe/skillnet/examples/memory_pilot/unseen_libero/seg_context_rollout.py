"""Paired LIBERO-90 rollouts for the frozen SEG2 model."""
from __future__ import annotations
import argparse, importlib.util, json, os, signal, socket, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/seg-context-01')
BASE=Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01')
CHECKPOINT=Path('/work/.skillnet-state/checkpoints/openpi-assets/checkpoints/pi05_libero')

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def port():
    with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]
def terminate(p):
    if p.poll() is not None:return
    try:os.killpg(p.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:p.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:os.killpg(p.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        p.wait()
def ready(p,pt):
    for _ in range(180):
        if p.poll() is not None:raise RuntimeError('SEG2 server exited; inspect server.log')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{pt}/healthz',timeout=2) as r:
                if r.status==200:return
        except Exception:time.sleep(2)
    raise TimeoutError('SEG2 server health timeout')
def append(path,row):
    with Path(path).open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())

def run(out=OUT,*,variants=('selected','no_pose_head'),timeout=900,servers=2):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if servers!=len(variants) or servers>3:raise ValueError('one server per active condition, at most three')
    harness=module(HERE/'run.py','seg2_rollout_harness')
    plan=json.loads((BASE/'plan.json').read_text());tasks=[int(x['task_id']) for x in plan['selected_tasks']]
    procs={};logs=[]
    try:
        for variant in variants:
            pt=port();folder=out/'servers'/variant;folder.mkdir(parents=True,exist_ok=True);log=(folder/'server.log').open('a');logs.append(log)
            env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_MEM_FRACTION='.10',TORCH_MEM_FRACTION='.05')
            proc=subprocess.Popen(['taskset','-c','8-15','nice','-n','10',sys.executable,str(HERE/'seg_context_server.py'),
                '--output',str(out),'--checkpoint',str(CHECKPOINT),'--variant',variant,'--port',str(pt)],
                env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            procs[variant]=(proc,pt);ready(proc,pt)
        def lane(variant):
            proc,pt=procs[variant];dest=out/'rollouts'/f'{variant}.jsonl';dest.parent.mkdir(parents=True,exist_ok=True)
            done=harness.read_terminal(dest)
            for task in tasks:
                for init in range(10):
                    if (task,init) in done and done[(task,init)].get('status')=='complete':continue
                    job={'host':'127.0.0.1','port':pt,'suite':'libero_90','task_id':task,'trial_index':init,
                        'seed_base':7,'condition':'memory','memory_snapshot':'none','query_text_source':'instruction_only',
                        'checkpoint':str(CHECKPOINT),'wait_steps':10,'replan_steps':5,'max_steps':520,'record':False}
                    root=out/'rollouts'/variant/'attempts'/f'task{task:02d}-init{init:02d}';root.mkdir(parents=True,exist_ok=True)
                    ats=sorted(root.glob('attempt-*'));start=max([int(x.name.split('-')[-1]) for x in ats],default=0)+1
                    result=None
                    for att in range(start,start+2):
                        result=harness.trial(job,root/f'attempt-{att}',timeout,att)
                        append(out/'rollouts'/variant/'attempts.jsonl',result)
                        if result.get('status')=='complete':break
                    append(dest,result);done=harness.read_terminal(dest)
                    if len(done)%5==0:
                        successes=sum(x.get('status')=='complete' and x.get('success',False) for x in done.values())
                        harness.write(out/'rollouts'/variant/'progress.json',{'completed':len(done),'scheduled':100,'successes':successes,'updated_unix':time.time()})
        with ThreadPoolExecutor(max_workers=len(variants)) as ex:list(ex.map(lane,variants))
    finally:
        for proc,_ in procs.values():terminate(proc)
        for f in logs:f.close()
    return summarize(out,variants)

def summarize(out=OUT,variants=('selected','no_pose_head')):
    from scipy.stats import binomtest
    harness=module(HERE/'run.py','seg2_rollout_harness_summary')
    baseline=harness.read_terminal(BASE/'production/rollouts.jsonl')
    previous=harness.read_terminal(Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/closed-loop-memory-01/libero90_unit_08/memory/rollouts.jsonl'))
    result={}
    for variant in variants:
        cur=harness.read_terminal(Path(out)/'rollouts'/f'{variant}.jsonl');common=sorted(k for k in cur if k in baseline and k[1]<10)
        success=lambda r:r.get('status')=='complete' and bool(r.get('success'))
        if len(common)!=100:result[variant]={'terminal_trials':len(common),'expected':100,'status':'incomplete'};continue
        pair=[(success(cur[k]),success(baseline[k])) for k in common]
        both=sum(a and b for a,b in pair);win=sum(a and not b for a,b in pair);loss=sum(b and not a for a,b in pair)
        p=float(binomtest(min(win,loss),win+loss,.5).pvalue) if win+loss else 1.
        units=[];fallback=[];reselect=[];angles=[];scales=[]
        for k in common:
            steps=cur[k].get('selected_units_and_t0',[]);seq=[x.get('unit_id',x.get('unit')) for x in steps if x.get('unit_id',x.get('unit')) is not None]
            units.append(len(set(seq)));fallback.extend(bool(x.get('fallback',False)) for x in steps)
            if len(seq)>1:reselect.extend(np.asarray(seq[1:])==np.asarray(seq[:-1]))
            for x in steps:
                theta=x.get('pose_scale')
                if theta is not None:angles.append(float(np.linalg.norm(theta[:3])));scales.append(float(np.exp(theta[3])))
        old=[(success(cur[k]),success(previous[k])) for k in common if k in previous]
        pw=sum(a and not b for a,b in old);pl=sum(b and not a for a,b in old)
        result[variant]={'successes':sum(success(cur[k]) for k in common),'baseline_successes':sum(success(baseline[k]) for k in common),
            'both':both,'condition_only':win,'baseline_only':loss,'mcnemar_exact_p':p,
            'unit08_paired':{'both':sum(a and b for a,b in old),'condition_only':pw,'unit08_only':pl,
                'mcnemar_exact_p':float(binomtest(min(pw,pl),pw+pl,.5).pvalue) if pw+pl else 1.},
            'mean_distinct_units':float(np.mean(units)),'fallback_query_fraction':float(np.mean(fallback)) if fallback else None,
            'same_unit_reselect_fraction':float(np.mean(reselect)) if reselect else None,
            'rotation_angle_rad_median':float(np.median(angles)) if angles else None,
            'scale_median':float(np.median(scales)) if scales else None,'status':'complete'}
    out_json={'variants':result,'seed_base':7,'inits':list(range(10)),'eval_touched_for_selection':False}
    p=Path(out)/'closed_loop_summary.json';p.write_text(json.dumps(out_json,indent=2)+'\n');return out_json


def compare_reference_conditions(out=OUT,variants=('selected','no_pose_head')):
    """Pair completed SEG2 trials with all locked LIB1 reference rollouts."""
    from scipy.stats import binomtest
    harness=module(HERE/'run.py','seg2_reference_pairing')
    refs={
        'baseline':BASE/'production/rollouts.jsonl',
        'unit08':Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/closed-loop-memory-01/libero90_unit_08/memory/rollouts.jsonl'),
        'R5':Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/lib-closed-01/R5/rollouts.jsonl'),
        'state_only':Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/lib-closed-01/state_only/rollouts.jsonl'),
    }
    def ok(row):return row.get('status')=='complete' and bool(row.get('success'))
    report={}
    for variant in variants:
        cur=harness.read_terminal(Path(out)/'rollouts'/f'{variant}.jsonl');pairs={}
        for ref_name,ref_path in refs.items():
            ref=harness.read_terminal(ref_path);common=sorted(set(cur)&set(ref))
            wins=sum(ok(cur[k]) and not ok(ref[k]) for k in common)
            losses=sum(ok(ref[k]) and not ok(cur[k]) for k in common)
            both=sum(ok(cur[k]) and ok(ref[k]) for k in common)
            p=float(binomtest(min(wins,losses),wins+losses,.5).pvalue) if wins+losses else 1.
            per_task={}
            for task in sorted({k[0] for k in common}):
                ks=[k for k in common if k[0]==task]
                per_task[str(task)]={'n':len(ks),'seg2':sum(ok(cur[k]) for k in ks),'reference':sum(ok(ref[k]) for k in ks)}
            pairs[ref_name]={'n':len(common),'both':both,'seg2_only':wins,'reference_only':losses,
                'mcnemar_exact_two_sided_p':p,'per_task':per_task,
                'seg2_terminal_errors':sum(cur[k].get('status')!='complete' for k in common),
                'reference_terminal_errors':sum(ref[k].get('status')!='complete' for k in common)}
        report[variant]=pairs
    result={'comparisons':report,'trial_keys':'(task_id, trial_index), expected 100 per comparator',
        'success_definition':'status==complete and success is true; technical errors count as failures and are separately counted'}
    p=Path(out)/'closed_loop_references.json'
    if p.exists():raise FileExistsError(f'refusing to overwrite {p}')
    p.write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);p.add_argument('--timeout',type=int,default=900);p.add_argument('--variants',nargs='+',default=['selected','no_pose_head']);p.add_argument('--references-only',action='store_true');a=p.parse_args()
    result=compare_reference_conditions(a.output,tuple(a.variants)) if a.references_only else run(a.output,variants=tuple(a.variants),timeout=a.timeout,servers=len(a.variants))
    print(json.dumps(result,indent=2))
