"""Original-instruction evaluation, with optional S0-only causal probe snapshots."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from collect_interventions import S2WebsocketClientPolicy,append,episode_status,image_hash,policy_input,rpc
from effect_worker import setup,truth,write
from rescue_protocol import noise_seed
from rescue_runtime import ordered_task,restore_snapshot,save_snapshot


def run(root,job):
    directory=root/job['directory']
    if (directory/'job.json').exists():raise FileExistsError('attempts are never automatically retried')
    directory.mkdir(parents=True,exist_ok=True);write(directory/'job.json',job)
    c,stages=setup(root)
    if job.get('snapshot'):
        env,obs,full,annotation,stage,audit=restore_snapshot(c,stages,job['task'],job['init'],root/job['snapshot'])
        step=audit['step'];write(directory/'replay_audit.json',audit)
        expected=np.load(root/job['snapshot'],allow_pickle=False)['state']
        raw=policy_input(obs,full,annotation,init=job['init'],query=step,evaluator=c.evaluator)
        if not np.array_equal(expected,raw['observation/state']):raise ValueError('snapshot proprio mismatch')
    else:
        env,obs,full,annotation=c.setup(job['task'],job['init'])
        stage=stages.Stages(env.env,job['task'],annotation['plan']);step=0
    goals,_=ordered_task(env,stages,job['task'],annotation)
    start=step;prefix=[];stable=0;probe_saved=False
    client=S2WebsocketClientPolicy('127.0.0.1',job['port'])
    started=time.monotonic();success=False;done=False;queries=0;inference=0.
    ever=[False]*len(goals);regressions=[0]*len(goals)
    try:
        while step<800 and not done:
            if time.time()>=job['hard_deadline'] or (root/'STOP_REQUESTED').exists():raise TimeoutError('collection deadline; not a policy failure')
            raw=policy_input(obs,full,annotation,init=job['init'],query=step,evaluator=c.evaluator)
            if job.get('save_probe') and not probe_saved and ((step>=50 and stable>=5) or step==300):
                path=root/'probe_snapshots'/(job['case']+'.npz');path.parent.mkdir(exist_ok=True)
                save_snapshot(path,env,raw,prefix,step)
                write(path.with_suffix('.json'),dict(step=step,reason='stable_goal' if stable>=5 else 'fallback300',parent=job['directory'],snapshot=str(path.relative_to(root))))
                probe_saved=True
            raw['s2_rng_seed']=noise_seed(job['seed'],job['init'],step)
            raw['diagnostic_tokens']=True
            tick=time.monotonic();response=rpc(client,raw);latency=time.monotonic()-tick
            tokens=response['token_audit']
            if tokens['truncated'] or not tokens['exact_encoding']:raise ValueError('original prompt token preservation failed')
            count=min(5,800-step)
            actions=np.asarray(response['actions'])[:count,:7]
            append(directory/'queries.jsonl',dict(step=step,seed=raw['s2_rng_seed'],prompt=full,image_sha256=image_hash(raw),state=np.asarray(raw['observation/state']).tolist(),sim_state=env.get_sim_state().tolist(),actions=actions.tolist(),token_audit=tokens,inference_s=latency,truth=truth(env.env,goals)))
            queries+=1;inference+=latency
            for action in actions:
                obs,_,returned,_=env.step(action.tolist());step+=1
                stage.observe(step,float(action[6]));status=truth(env.env,goals)
                stable=stable+1 if status['verified'][0] else 0
                for i,value in enumerate(status['raw_goals']):
                    if ever[i] and not value:regressions[i]+=1
                    ever[i]=ever[i] or value
                prefix.append(action.tolist())
                append(directory/'states.jsonl',dict(step=step,command=action.tolist(),truth=status))
                success,done=episode_status(env,returned)
                if done:break
        write(directory/'result.json',dict(status='complete',success=bool(success),start_step=start,end_step=step,query_count=queries,inference_s=inference,wall_s=time.monotonic()-started,goal_regressed_steps=regressions,job=job,full_prompt=full,final_truth=truth(env.env,goals)))
    finally:client.close();env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--job',type=Path,required=True);a=p.parse_args()
    try:run(a.root,json.loads(a.job.read_text()))
    except Exception as exc:
        job=json.loads(a.job.read_text());write(a.root/job['directory']/'incomplete.json',dict(error=repr(exc),count_as_failure=False));raise
