"""Snapshot-only resumable normal-pi05 evaluation, with Long preflight."""
from pathlib import Path
import argparse,json,os,sys,time,socket,subprocess,signal,urllib.request,threading
from concurrent.futures import ThreadPoolExecutor,as_completed
HERE=Path(__file__).resolve().parent
LOCK=threading.Lock()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)
def append(path,row):
    with LOCK:
        with path.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def free_port():
    with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]
def terminate(proc):
    if proc.poll() is not None:return
    try:os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:os.killpg(proc.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        proc.wait()
def ready(proc,port):
    for _ in range(180):
        if proc.poll() is not None:raise RuntimeError('policy server exited during startup')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz',timeout=2) as r:
                if r.status==200:return
        except Exception:time.sleep(2)
    raise TimeoutError('policy server startup timeout')
def read_terminal(path):
    data={}
    if path.exists():
        for l in path.read_text().splitlines():
            try:x=json.loads(l)
            except json.JSONDecodeError:continue
            if x.get('status') in ['complete','error','timeout']:data[x['task_id'],x['trial_index']]=x
    return data

def trial(job,folder,timeout,attempt):
    folder.mkdir(parents=True,exist_ok=False);write(folder/'job.json',job)
    env=os.environ.copy();env['OMP_NUM_THREADS']=env['MKL_NUM_THREADS']=env['OPENBLAS_NUM_THREADS']='1'
    with (folder/'worker.log').open('w') as log:
        proc=subprocess.Popen(['taskset','-c','16-31','nice','-n','10',sys.executable,str(HERE/'worker.py'),str(folder/'job.json'),str(folder/'result.json')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:terminate(proc);return dict(job,status='timeout',success=False,error='wall clock timeout',attempt=attempt,attempt_dir=str(folder))
    if proc.returncode or not (folder/'result.json').exists():return dict(job,status='error',success=False,error=f'worker exit {proc.returncode}',attempt=attempt,attempt_dir=str(folder))
    result=json.loads((folder/'result.json').read_text());result.update(attempt=attempt,attempt_dir=str(folder));return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--servers',type=int,default=4);p.add_argument('--timeout',type=int,default=600)
    a=p.parse_args();out=a.output;plan=json.loads((out/'plan.json').read_text());servers=[];logs=[]
    write(out/'runtime.json',{'pid':os.getpid(),'started_unix':time.time(),'snapshot_code':str(HERE),'servers':a.servers})
    def start_server(index):
        port=free_port();folder=out/'servers'/str(index);folder.mkdir(parents=True,exist_ok=True);log=(folder/'server.log').open('a');logs.append(log)
        env=os.environ.copy();env['OMP_NUM_THREADS']=env['MKL_NUM_THREADS']=env['OPENBLAS_NUM_THREADS']='2';env['XLA_PYTHON_CLIENT_PREALLOCATE']='false';env['XLA_PYTHON_CLIENT_MEM_FRACTION']='.20'
        proc=subprocess.Popen(['taskset','-c','8-15','nice','-n','10',sys.executable,str(HERE/'serve.py'),'--checkpoint',str(a.checkpoint),'--port',str(port)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        servers.append((proc,port));write(folder/'process.json',{'pid':proc.pid,'port':port});return proc,port
    def makejob(suite,task,init,port,record):
        return {'host':'127.0.0.1','port':port,'suite':suite,'task_id':task,'trial_index':init,'seed_base':7,'condition':'normal','memory_snapshot':'none','query_text_source':'instruction_only','checkpoint':str(a.checkpoint),'wait_steps':10,'replan_steps':5,'max_steps':520,'record':record}
    def execute(job,phase):
        name=f"task{job['task_id']:02d}-init{job['trial_index']:02d}";attempt_root=out/phase/'attempts'/name
        existing=list(attempt_root.glob('attempt-*')) if attempt_root.exists() else []
        first=max([int(x.name.split('-')[-1]) for x in existing],default=0)+1
        results=[]
        for attempt in range(first,min(first+2,3)):
            r=trial(job,attempt_root/f'attempt-{attempt}',a.timeout,attempt);results.append(r)
            append(out/phase/'attempts.jsonl',r)
            if r['status']=='complete':break
        if not results:raise RuntimeError(f'previous two technical attempts exhausted: {name}')
        r=results[-1];r.update(task_id=job['task_id'],trial_index=job['trial_index'],task_suite=job['suite'])
        append(out/phase/'rollouts.jsonl',r)
        with LOCK:
            done=read_terminal(out/phase/'rollouts.jsonl');write(out/phase/'progress.json',{'processed':len(done),'successes':sum(x.get('status')=='complete' and x.get('success',False) for x in done.values()),'scheduled':4 if phase=='smoke' else 100,'updated_unix':time.time()})
        return r
    try:
        proc,port=start_server(0);ready(proc,port)
        smoke=[]
        if not (out/'smoke/comparison.json').exists():
            done=read_terminal(out/'smoke/rollouts.jsonl')
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures={pool.submit(execute,makejob('libero_10',ref['task_id'],ref['trial_index'],port,True),'smoke'):ref for ref in plan['smoke_reference'] if (ref['task_id'],ref['trial_index']) not in done}
                for f in as_completed(futures):f.result()
            done=read_terminal(out/'smoke/rollouts.jsonl')
            for ref in plan['smoke_reference']:
                actual=done[ref['task_id'],ref['trial_index']]
                matched=actual['status']=='complete' and actual['success']==ref['success'] and actual['initial_state_sha256']==ref['initial_state_sha256'] and actual['environment_seed']==ref['environment_seed'] and actual['trial_seed']==ref['trial_seed']
                smoke.append({'task_id':ref['task_id'],'initial_state_id':ref['trial_index'],'expected_success':ref['success'],'actual_success':actual.get('success'),'expected_steps':ref['env_steps'],'actual_steps':actual.get('env_steps'),'matched':matched})
            write(out/'smoke/comparison.json',{'all_match':all(x['matched'] for x in smoke),'cases':smoke})
        assert json.loads((out/'smoke/comparison.json').read_text())['all_match'],'Long preflight mismatch; production not started'
        for index in range(1,a.servers):start_server(index)
        for proc,port in servers:ready(proc,port)
        done=read_terminal(out/'production/rollouts.jsonl');jobs=[(t['task_id'],i) for t in plan['selected_tasks'] for i in range(10) if (t['task_id'],i) not in done]
        # One sequential lane per worker: exactly two environments per server.
        def lane(slot):
            proc,port=servers[slot%a.servers]
            for task,init in jobs[slot::a.servers*2]:
                if proc.poll() is not None:raise RuntimeError(f'policy server {slot%a.servers} died')
                execute(makejob('libero_90',task,init,port,init<3),'production')
        with ThreadPoolExecutor(max_workers=a.servers*2) as pool:
            for f in as_completed([pool.submit(lane,k) for k in range(a.servers*2)]):f.result()
        assert len(read_terminal(out/'production/rollouts.jsonl'))==100
        write(out/'EVALUATION_COMPLETE.json',{'trials':100,'finished_unix':time.time()})
    finally:
        for proc,_ in servers:terminate(proc)
        for log in logs:log.close()
if __name__=='__main__':main()
