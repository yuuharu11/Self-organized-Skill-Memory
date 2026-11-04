"""Fixed-prompt branches, shorter execution, or a bounded physical teacher."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from collect_interventions import S2WebsocketClientPolicy,append,episode_status,image_hash,policy_input,rpc
from effect_worker import setup,truth,write
from rescue_protocol import base_prompts,noise_seed,teacher_action
from rescue_runtime import Metrics,ordered_task,restore_snapshot,token_rpc


def run(root,job):
    directory=root/job["directory"]
    if (directory/"job.json").exists():
        raise FileExistsError("no automatic retry of completed or interrupted attempts")
    directory.mkdir(parents=True,exist_ok=True)
    write(directory/"job.json",job)
    c,stages=setup(root)
    if job.get("snapshot"):
        env,obs,full,annotation,stage,audit=restore_snapshot(c,stages,job["task"],job["init"],root/job["snapshot"])
        step=audit["step"]
        write(directory/"replay_audit.json",audit)
    else:
        env,obs,full,annotation=c.setup(job["task"],job["init"])
        stage=stages.Stages(env.env,job["task"],annotation["plan"])
        step=0
    start_step=step
    goals,remaining=ordered_task(env,stages,job["task"],annotation)
    metrics=Metrics(env.env,goals)
    client=S2WebsocketClientPolicy("127.0.0.1",job["port"])
    prompts=json.loads((root/job["prompts_file"]).read_text())["prompts"] if job.get("prompts_file") else base_prompts(full,remaining)
    condition=job["condition"]
    prompt=prompts.get(condition,full) if job["mode"]!="confirmation" else full
    stable=0
    switched=False
    switch_step=start_step if job["mode"]=="main" else None
    control_step=job.get("execute_steps",5)
    success=False;done=False;query_count=0;teacher_steps=0;inference_s=0.
    started=time.monotonic()
    ever=[False,False];regressions=[0,0]
    import imageio.v2 as imageio
    video=imageio.get_writer(str(directory/"video.mp4"),fps=4,codec="libx264",ffmpeg_params=["-threads","1"])

    def check_deadline():
        if time.time()>=job["hard_deadline"] or (root/"STOP_REQUESTED").exists():
            raise TimeoutError("collection wall-time limit; not a policy failure")

    def execute(action,source):
        nonlocal obs,step,success,done,stable
        obs,_,returned,_=env.step(np.asarray(action).tolist())
        step+=1
        stage.observe(step,float(action[6]))
        record=metrics.observe(obs,step,action)
        for i,value in enumerate(record["truth"]["raw_goals"]):
            if ever[i] and not value:
                regressions[i]+=1
            ever[i]=ever[i] or value
        stable=stable+1 if record["truth"]["verified"][0] else 0
        append(directory/"states.jsonl",dict(source=source,**record))
        success,done=episode_status(env,returned)

    try:
        if condition=="teacher":
            from robosuite.utils import transform_utils as transforms
            controller=env.env.robots[0].controller
            if not np.allclose(controller.output_max,[.05,.05,.05,.5,.5,.5]):
                raise ValueError("teacher OSC scale differs from audited configuration")
            reference=job["teacher_reference"]
            def orientation_error(target,current):
                return transforms.quat2axisangle(transforms.quat_multiply(np.asarray(target),transforms.quat_conjugate(np.asarray(current))))
            for index in range(min(20,800-step)):
                check_deadline()
                action=teacher_action(obs["robot0_eef_pos"],obs["robot0_eef_quat"],reference,index,
                                      controller.output_max,orientation_error)
                execute(action,"teacher")
                teacher_steps+=1
                if done:
                    break
        while step<800 and not done:
            check_deadline()
            if job["mode"]=="confirmation" and not switched and step>=50 and stable>=5:
                prompt=prompts[condition]
                switched=True;switch_step=step
            raw=policy_input(obs,prompt,annotation,init=job["init"],query=step,evaluator=c.evaluator)
            raw["s2_rng_seed"]=noise_seed(job["seed"],job["init"],step)
            raw["diagnostic_tokens"]=True
            tick=time.monotonic();response=rpc(client,raw);latency=time.monotonic()-tick
            tokens=response["token_audit"]
            if tokens["truncated"] or not tokens["exact_encoding"]:
                write(directory/"invalid_input.json",tokens)
                raise ValueError("actual prompt truncated or token encoding failed")
            count=min(control_step,800-step)
            actions=np.asarray(response["actions"])[:count,:7]
            append(directory/"queries.jsonl",dict(step=step,seed=raw["s2_rng_seed"],prompt=prompt,
                image_sha256=image_hash(raw),state=np.asarray(raw["observation/state"]).tolist(),
                sim_state=env.get_sim_state().tolist(),token_audit=tokens,actions=actions.tolist(),
                execution_limit=count,inference_s=latency,truth=truth(env.env,goals)))
            query_count+=1;inference_s+=latency
            video.append_data(np.concatenate([raw["observation/image"],raw["observation/wrist_image"]],axis=1))
            for action in actions:
                execute(action,"frozen_policy")
                if done:
                    break
        write(directory/"result.json",dict(status="complete",success=bool(success),start_step=start_step,
            end_step=step,remaining_budget=800-start_step,teacher_steps=teacher_steps,query_count=query_count,
            inference_s=inference_s,wall_s=time.monotonic()-started,switch_step=switch_step,
            full_prompt=full,final_prompt=prompt,goal_regressed_steps=regressions,job=job,
            final_truth=truth(env.env,goals),**stage.result(success)))
    finally:
        video.close();client.close();env.close()


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--job",type=Path,required=True)
    a=p.parse_args();run(a.root,json.loads(a.job.read_text()))
