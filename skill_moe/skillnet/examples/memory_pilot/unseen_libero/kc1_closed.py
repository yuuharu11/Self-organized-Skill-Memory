#!/usr/bin/env python3
"""KC1 closed-loop runner, hard-gated on completion of all T5 conditions."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', Path(__file__).resolve().parents[5] / '.skillnet-state')).expanduser()
ROOT=STATE_DIR / 'outputs/memory-pilot/unseen-libero'
OUT=ROOT/'kc1-01'
T5=ROOT/'t0x-01/t5'
CONDITION='KCtopkGdec'


def log(event,**kw):
    OUT.mkdir(parents=True,exist_ok=True)
    row={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'event':event,**kw}
    with (OUT/'progress.log').open('a') as f:f.write(json.dumps(row,ensure_ascii=False,default=float)+'\n')
    print(json.dumps(row,ensure_ascii=False,default=float),flush=True)


def t5_gate():
    done=T5/'DONE';progress=T5/'progress.log'
    if not progress.is_file():raise RuntimeError('T5 progress log is missing; KC1 GPU phase is blocked')
    seen={}
    for line in progress.read_text().splitlines():
        try:r=json.loads(line)
        except Exception:continue
        if r.get('message')=='condition complete' and not r.get('smoke',True):
            seen[(r.get('suite'),r.get('condition'))]=int(r.get('completed',0))
    expected={(s,c) for s in ('libero90','libero10') for c in ('AaddGexp','AtopkGexp','AtopkGdec')}
    missing={k:seen.get(k,0) for k in sorted(expected) if seen.get(k,0)!=100}
    if missing:raise RuntimeError(f'T5 full conditions not complete: {missing}')
    active=subprocess.run(['pgrep','-af','[t]0x_t5(_serve)?\\.py'],capture_output=True,text=True).stdout.strip()
    if active:raise RuntimeError(f'T5 process still active; KC1 GPU phase is blocked: {active}')
    artifacts_present=all((T5/name).is_file() for name in ('results.json','report.md'))
    return {'completed':{f'{s}/{c}':seen[(s,c)] for s,c in sorted(expected)},
            'runner_exited':True,'done_file_present':done.is_file(),
            't5_report_artifacts_present':artifacts_present}


def gpu_memory():
    r=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],
                     capture_output=True,text=True,check=True)
    a,b=map(int,r.stdout.strip().split(',')[:2]);return {'used_mib':a,'total_mib':b}


def spec():
    meta=json.loads((OUT/'offline_corrected.json').read_text())
    query=meta['selected']['query']
    expert=ROOT/'kq2-cl-01/expert/train-K1/checkpoints/000500'
    bundle=OUT/'heads/bundle.pt'
    calibration=OUT/'calibration.json'
    for p in (expert,bundle,calibration):
        if not p.exists():raise FileNotFoundError(p)
    return {CONDITION:{'mode':'memory','query':query,'expert':str(expert),'bundle':str(bundle),
                       'tpol':{'confidence_mode':'disagreement','gripper_mode':'decoder',
                               'calibration':str(calibration),'condition':CONDITION}}}


def _popen_kc1(args,*pos,**kw):
    if isinstance(args,(list,tuple)):
        args=list(args)
        for i,v in enumerate(args):
            if str(v).endswith('/kq2cl_serve.py'):
                args[i]=str(HERE/'kc1_serve.py');break
    return REAL_POPEN(args,*pos,**kw)


def run_suite(suite,*,smoke,workers,fraction,cpus):
    import kq2cl_common as C
    import kq2cl_closed as K
    C.OUT=OUT
    sp=spec();stage=OUT
    specpath=stage/('closed-spec-smoke.json' if smoke else 'closed-spec-production.json')
    if specpath.exists() and json.loads(specpath.read_text())!=sp:
        raise FileExistsError(f'refusing to replace KC1 spec {specpath}')
    if not specpath.exists():C.write_json(specpath,sp)
    K.PLAN[suite]=dict(K.PLAN[suite]);K.PLAN[suite][CONDITION]=range(10)
    old=K.subprocess.Popen;K.subprocess.Popen=_popen_kc1
    envold={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}
    os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    try:
        before=gpu_memory();log('kc1_condition_start',suite=suite,smoke=smoke,gpu=before,fraction=fraction)
        K.run([CONDITION],workers=1 if smoke else workers,fraction=fraction,smoke=smoke,
              cpus=[cpus],spec_path=str(specpath),suite=suite)
        root=OUT/('closed-loop-smoke'+('' if suite=='libero90' else '-libero10') if smoke else f'closed-loop/{suite}')
        rows=K.read_trials(root/CONDITION);expected=2 if smoke else 100
        complete=sum(r.get('status')=='complete' for r in rows.values())
        if complete!=expected:raise RuntimeError(f'KC1 {suite}: {complete}/{expected} trials complete')
        successes=sum(bool(r.get('success')) for r in rows.values() if r.get('status')=='complete')
        log('kc1_condition_complete',suite=suite,smoke=smoke,completed=complete,successes=successes,
            gpu_after=gpu_memory())
    finally:
        K.subprocess.Popen=old
        for k,v in envold.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v
        C.OUT=OUT


def _by_trial(path):
    rows={}
    for line in Path(path).read_text().splitlines():
        if not line.strip():continue
        r=json.loads(line)
        if r.get('status')=='complete':rows[(int(r['task_id']),int(r['trial_index']))]=r
    return rows


def _paired(a,b):
    common=sorted(set(a)&set(b));left=np.asarray([a[k]['success'] for k in common],bool);right=np.asarray([b[k]['success'] for k in common],bool)
    aonly=int((left&~right).sum());bonly=int((right&~left).sum());n=aonly+bonly
    import math
    p=1. if not n else min(1.,2.*sum(math.comb(n,i) for i in range(min(aonly,bonly)+1))/2**n)
    return {'pairs':len(common),'kc1_only':bonly,'reference_only':aonly,'mcnemar_p_exact':p}


def summarize():
    from t0x_report import baseline_for_suite
    result={'status':'complete','condition':CONDITION,'selected_query':json.loads((OUT/'offline_corrected.json').read_text())['selected']['query'],
        't0_calibration':json.loads((OUT/'calibration.json').read_text()),'suites':{},
        'paired_protocol':'same suite/task/trial index, seed_base=7, wait=10, replan=5, max 520'}
    for suite in ('libero90','libero10'):
        kc=_by_trial(OUT/'closed-loop'/suite/CONDITION/'trials.jsonl')
        t5=_by_trial(T5/'closed-loop/closed-loop'/suite/'AtopkGdec'/'trials.jsonl')
        t1=_by_trial(ROOT/'t0x-01/t1/closed-loop'/suite/'Aadd'/'trials.jsonl')
        _,pi=baseline_for_suite(suite);pi={(int(k[0]),int(k[1])):v for k,v in pi.items() if v.get('status')=='complete'}
        result['suites'][suite]={'kc1':{'completed':len(kc),'successes':sum(bool(r['success']) for r in kc.values())},
            'references':{'T5_AtopkGdec':{'completed':len(t5),'successes':sum(bool(r['success']) for r in t5.values())},
                          'T1_Aadd':{'completed':len(t1),'successes':sum(bool(r['success']) for r in t1.values())},
                          'pi05':{'completed':len(pi),'successes':sum(bool(r['success']) for r in pi.values())}},
            'paired':{'T5_AtopkGdec':_paired(kc,t5),'T1_Aadd':_paired(kc,t1),'pi05':_paired(kc,pi)},
            'tasks':{str(t):sum(bool(r['success']) for (tt,_),r in kc.items() if tt==t)
                     for t in sorted({k[0] for k in kc})}}
    if any(result['suites'][s]['kc1']['completed']!=100 for s in result['suites']):
        raise RuntimeError('cannot mark KC1 DONE: incomplete production suite')
    path=OUT/'closed_loop_results.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False,default=float)+'\n')
    (OUT/'DONE').write_text('DONE '+time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())+' UTC\n')
    log('kc1_closed_loop_done',successes={s:x['kc1']['successes'] for s,x in result['suites'].items()})
    return result


def run_all(*,workers=6,fraction=.2,cpus='24-39'):
    gate=t5_gate();mem=gpu_memory()
    if mem['used_mib']>2048:raise RuntimeError(f'GPU not free after T5 gate: {mem}')
    log('t5_gate_passed',gate=gate,gpu=mem)
    run_suite('libero90',smoke=True,workers=workers,fraction=fraction,cpus='40-47')
    for suite in ('libero90','libero10'):
        run_suite(suite,smoke=False,workers=workers,fraction=fraction,cpus=cpus)
    return summarize()


REAL_POPEN=subprocess.Popen
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('cmd',choices=('gate','smoke','run','all','summarize'))
    ap.add_argument('--suite',choices=('libero90','libero10'),default='libero90');ap.add_argument('--workers',type=int,default=6)
    ap.add_argument('--fraction',type=float,default=.2);ap.add_argument('--cpus',default='24-39');a=ap.parse_args()
    if a.cmd=='gate': print(json.dumps({'t5':t5_gate(),'gpu':gpu_memory()},indent=2))
    elif a.cmd=='summarize':print(json.dumps(summarize(),indent=2,ensure_ascii=False))
    elif a.cmd=='all':print(json.dumps(run_all(workers=a.workers,fraction=a.fraction,cpus=a.cpus),indent=2,ensure_ascii=False))
    else:
        gate=t5_gate();mem=gpu_memory()
        if mem['used_mib']>2048:raise RuntimeError(f'GPU not free after T5 gate: {mem}')
        if not a.cmd=='smoke':run_suite(a.suite,smoke=False,workers=a.workers,fraction=a.fraction,cpus=a.cpus)
        else:run_suite(a.suite,smoke=True,workers=a.workers,fraction=a.fraction,cpus='40-47')
        log('t5_gate_passed',gate=gate,gpu=mem)

if __name__=='__main__':main()
