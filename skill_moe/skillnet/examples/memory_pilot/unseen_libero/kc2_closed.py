#!/usr/bin/env python3
"""KC2 KCtopkGdec-compatible closed-loop runner with T5/KC1 pairing."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import kc2_run as K
ROOT=K.ROOT;OUT=K.OUT;T5=ROOT/'t0x-01/t5';KC1=ROOT/'kc1-01'
CONDITION='KC2-dot-Gdec'


def log(event,**kw):
    K.log(event,**kw)


def t5_gate():
    p=T5/'progress.log'
    if not (T5/'DONE').is_file() or not (T5/'results.json').is_file() or not p.is_file():
        raise RuntimeError('T5 full run artifacts/DONE are missing')
    complete={}
    for line in p.read_text().splitlines():
        try:r=json.loads(line)
        except Exception:continue
        if r.get('message')=='condition complete' and not r.get('smoke',True):
            complete[(r.get('suite'),r.get('condition'))]=int(r.get('completed',0))
    expected={(s,c) for s in ('libero90','libero10') for c in ('AaddGexp','AtopkGexp','AtopkGdec')}
    missing={k:complete.get(k,0) for k in expected if complete.get(k)!=100}
    if missing:raise RuntimeError(f'T5 not fully complete: {missing}')
    active=subprocess.run(['pgrep','-af','[t]0x_t5(_serve)?\\.py'],capture_output=True,text=True).stdout.strip()
    if active:raise RuntimeError(f'T5 processes still active: {active}')
    return {'conditions':{f'{s}/{c}':complete[(s,c)] for s,c in sorted(expected)},'done':True}


def gpu_memory():
    r=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    a,b=map(int,r.stdout.strip().split(',')[:2]);return {'used_mib':a,'total_mib':b}


def spec():
    offline=json.loads((OUT/'offline_results.json').read_text());selected=offline['selected']
    expert=ROOT/'kq2-cl-01/expert/train-K1/checkpoints/000500'
    calibration=OUT/'calibration.json';bundle=Path(selected['checkpoint'])
    for p in (expert,calibration,bundle):
        if not p.exists():raise FileNotFoundError(p)
    return {CONDITION:{'mode':'memory','query':selected['q_name'],'expert':str(expert),
        'bundle':str(bundle),'calibration':str(calibration),'score_mode':selected['score_mode'],
        'lambda':selected['lambda'],'tpol':{'confidence_mode':'topk_disagreement','gripper_mode':'decoder',
        'calibration':str(calibration),'condition':CONDITION}}}


REAL_POPEN=subprocess.Popen
def _popen_kc2(args,*pos,**kw):
    if isinstance(args,(list,tuple)):
        args=list(args)
        for i,v in enumerate(args):
            if str(v).endswith('/kq2cl_serve.py'):
                args[i]=str(HERE/'kc2_serve.py');break
    return REAL_POPEN(args,*pos,**kw)


def run_suite(suite,*,smoke,workers,fraction,cpus):
    import kq2cl_common as C
    import kq2cl_closed as CL
    C.OUT=OUT
    sp=spec();spfile=OUT/('closed-spec-smoke.json' if smoke else 'closed-spec-production.json')
    if spfile.exists() and json.loads(spfile.read_text())!=sp:raise FileExistsError(spfile)
    if not spfile.exists():C.write_json(spfile,sp)
    CL.PLAN[suite]=dict(CL.PLAN[suite]);CL.PLAN[suite][CONDITION]=range(10)
    orig=CL.subprocess.Popen;CL.subprocess.Popen=_popen_kc2
    oldenv={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}
    os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    try:
        before=gpu_memory();log('kc2_condition_start',suite=suite,smoke=smoke,gpu=before,fraction=fraction)
        expected=2 if smoke else 100
        for attempt in range(1,4):
            CL.run([CONDITION],workers=1 if smoke else workers,fraction=fraction,smoke=smoke,
                   cpus=[cpus],spec_path=str(spfile),suite=suite)
            root=OUT/('closed-loop-smoke' if smoke else f'closed-loop/{suite}')
            rows=CL.read_trials(root/CONDITION)
            complete=sum(x.get('status')=='complete' for x in rows.values())
            if complete==expected:break
            bad=[r for r in rows.values() if r.get('status')!='complete']
            oom=sum('RESOURCE_EXHAUSTED' in str(r.get('error','')) or 'out of memory' in str(r.get('error','')).lower() for r in bad)
            log('kc2_retry_incomplete_trials',suite=suite,attempt=attempt,complete=complete,expected=expected,oom_rows=oom)
        else:raise RuntimeError(f'KC2 {suite}: incomplete after 3 attempts')
        rows=CL.read_trials(root/CONDITION);success=sum(bool(x.get('success')) for x in rows.values() if x.get('status')=='complete')
        log('kc2_condition_complete',suite=suite,smoke=smoke,completed=complete,successes=success,
            gpu_after=gpu_memory(),oom_retries=sum(1 for r in rows.values() if 'oom' in str(r.get('error','')).lower()))
    finally:
        CL.subprocess.Popen=orig
        for k,v in oldenv.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v
        C.OUT=OUT


def _by_trial(path):
    out={}
    for line in Path(path).read_text().splitlines():
        if not line.strip():continue
        r=json.loads(line)
        if r.get('status')=='complete':out[(int(r['task_id']),int(r['trial_index']))]=bool(r['success'])
    return out


def summarize():
    from t0x_report import baseline_for_suite
    final={'status':'complete','condition':CONDITION,'selected':json.loads((OUT/'offline_results.json').read_text())['selected'],
        'calibration':json.loads((OUT/'calibration.json').read_text()),'paired_protocol':'same suite/task/trial index; seed_base=7; wait=10; replan=5; max 520',
        'suites':{}}
    for suite in ('libero90','libero10'):
        kc=_by_trial(OUT/'closed-loop'/suite/CONDITION/'trials.jsonl')
        kc1=_by_trial(KC1/'closed-loop'/suite/'KCtopkGdec'/'trials.jsonl')
        t5=_by_trial(T5/'closed-loop/closed-loop'/suite/'AtopkGdec'/'trials.jsonl')
        _,pirows=baseline_for_suite(suite);pi={(int(k[0]),int(k[1])):bool(v['success']) for k,v in pirows.items() if v.get('status')=='complete'}
        refs={'KC1':kc1,'T5_AtopkGdec':t5,'pi05':pi}
        final['suites'][suite]={'kc2':{'completed':len(kc),'successes':sum(kc.values())},
          'references':{n:{'completed':len(x),'successes':sum(x.values())} for n,x in refs.items()},
          'paired':{n:K.paired_counts(kc,x) for n,x in refs.items()}}
    if any(x['kc2']['completed']!=100 for x in final['suites'].values()):raise RuntimeError('KC2 incomplete suite')
    path=OUT/'closed_loop_results.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(final,indent=2,ensure_ascii=False,default=float)+'\n')
    (OUT/'DONE').write_text('DONE '+time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())+' UTC\n')
    log('kc2_closed_loop_done',success={s:v['kc2']['successes'] for s,v in final['suites'].items()})
    return final


def run_all(workers=6,fraction=.2,cpus='24-39'):
    gate=t5_gate();mem=gpu_memory()
    if mem['used_mib']>2048:raise RuntimeError(f'GPU is not available: {mem}')
    log('t5_gate_passed',gate=gate,gpu=mem)
    run_suite('libero90',smoke=True,workers=1,fraction=fraction,cpus='40-47')
    for s in ('libero90','libero10'):run_suite(s,smoke=False,workers=workers,fraction=fraction,cpus=cpus)
    return summarize()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('cmd',choices=('gate','smoke','run','summarize','all'))
    ap.add_argument('--suite',choices=('libero90','libero10'),default='libero90');ap.add_argument('--workers',type=int,default=6)
    ap.add_argument('--fraction',type=float,default=.2);ap.add_argument('--cpus',default='24-39');a=ap.parse_args()
    if a.cmd=='gate':print(json.dumps({'t5':t5_gate(),'gpu':gpu_memory()},indent=2))
    elif a.cmd=='summarize':print(json.dumps(summarize(),indent=2,ensure_ascii=False))
    elif a.cmd=='all':print(json.dumps(run_all(a.workers,a.fraction,a.cpus),indent=2,ensure_ascii=False))
    else:
        gate=t5_gate();mem=gpu_memory()
        if mem['used_mib']>2048:raise RuntimeError(f'GPU is not available: {mem}')
        run_suite(a.suite,smoke=a.cmd=='smoke',workers=a.workers,fraction=a.fraction,cpus='40-47' if a.cmd=='smoke' else a.cpus)
        log('t5_gate_passed',gate=gate,gpu=mem)

if __name__=='__main__':main()
