"""Simulator and RPC helpers shared by replay and rescue branches."""
import time

import numpy as np

from collect_interventions import image_hash, policy_input
from effect_worker import truth


def ordered_task(env, stages, task, annotation):
    order=[(index,pair[1]) for index,pair in enumerate(stages.BINDINGS[task]) if pair[0]=="goal"]
    return [env.env.parsed_problem["goal_state"][idx] for _,idx in order], ". ".join(annotation["plan"][order[0][0]+1:])


def token_rpc(client, raw):
    from skillnet_client import msgpack_numpy
    client._ws.send(client._packer.pack(dict(raw,diagnostic_tokens_only=True)))
    result=client._ws.recv(timeout=300)
    if isinstance(result,str):
        raise RuntimeError(result)
    return msgpack_numpy.unpackb(result)["token_audit"]


def save_snapshot(path, env, raw, prefix, step, reference=None):
    np.savez_compressed(path,prefix=np.asarray(prefix,np.float64),sim_state=env.get_sim_state(),
        image_sha256=image_hash(raw),step=step,image=raw["observation/image"],
        wrist=raw["observation/wrist_image"],state=raw["observation/state"])


def restore_snapshot(c, stages, task, init, path):
    env,obs,full,annotation=c.setup(task,init)
    snap=np.load(path,allow_pickle=False)
    prefix=np.asarray(snap["prefix"],float)
    stage=stages.Stages(env.env,task,annotation["plan"])
    for step,action in enumerate(prefix,1):
        obs,_,_,_=env.step(action.tolist())
        stage.observe(step,float(action[6]))
    raw=policy_input(obs,full,annotation,init=init,query=len(prefix),evaluator=c.evaluator)
    delta=float(np.max(np.abs(env.get_sim_state()-snap["sim_state"])))
    audit=dict(state_max_abs=delta,images_equal=image_hash(raw)==str(snap["image_sha256"]),step=len(prefix))
    if delta!=0 or not audit["images_equal"]:
        env.close()
        raise RuntimeError(f"snapshot mismatch: {audit}")
    return env,obs,full,annotation,stage,audit


class Metrics:
    def __init__(self,base,goals):
        self.base=base
        self.goals=goals
        model=base.sim.model
        self.gripper=set(base.robots[0].gripper.contact_geoms)
        names=list(base.objects_dict)+list(base.fixtures_dict)
        self.targets=[]
        for goal in goals:
            candidates=[name for name in names if str(goal[1]).startswith(name)]
            name=max(candidates,key=len) if candidates else str(goal[1])
            obj=base.get_object(name)
            geoms=set(obj.contact_geoms)
            ids=[model.geom_name2id(g) for g in geoms]
            joints=[j for j in obj.joints if int(model.jnt_type[model.joint_name2id(j)]) in (2,3)]
            region=[word for word in ("top","bottom","middle") if word in str(goal[1])]
            if region:
                selected=[j for j in joints if any(word in j for word in region)]
                if selected:
                    joints=selected
            metadata=[]
            for j in joints:
                ji=model.joint_name2id(j)
                lo,hi=model.jnt_range[ji]
                metadata.append(dict(name=j,adr=int(model.jnt_qposadr[ji]),kind=int(model.jnt_type[ji]),
                                     scale=max(float(hi-lo),1e-6),range=[float(lo),float(hi)]))
            self.targets.append(dict(name=name,geoms=geoms,ids=ids,joints=metadata,
                                     body=model.body_name2id(obj.root_body)))

    def observe(self,obs,step,command):
        base=self.base;model=base.sim.model;data=base.sim.data
        status=truth(base,self.goals)
        active=next((i for i,v in enumerate(status["raw_goals"]) if not v),1)
        target=self.targets[active]
        contacts=[]
        target_contact=False
        for index in range(data.ncon):
            ct=data.contact[index]
            a,b=model.geom_id2name(int(ct.geom1)),model.geom_id2name(int(ct.geom2))
            if a in self.gripper or b in self.gripper:
                contacts.append([a,b])
                target_contact=target_contact or (a in self.gripper and b in target["geoms"]) or (b in self.gripper and a in target["geoms"])
        eef=np.asarray(obs["robot0_eef_pos"],float)
        progress=np.zeros(3)
        joints={}
        goal_distance=None
        if target["joints"]:
            distances=[]
            for k,meta in enumerate(target["joints"][:3]):
                value=float(data.qpos[meta["adr"]]);joints[meta["name"]]=value
                progress[k]=value/meta["scale"]
                end=0. if self.goals[active][0].lower() in ("close","turnoff") else max(meta["range"],key=abs)
                distances.append(abs(value-end)/meta["scale"])
            goal_distance=float(np.mean(distances))
        else:
            progress=np.asarray(data.body_xpos[target["body"]],float)/.05
        distance=min(float(np.linalg.norm(eef-data.geom_xpos[index])) for index in target["ids"])
        return dict(step=step,truth=status,active_goal=active,target=target["name"],
            target_contact=bool(target_contact),gripper_contacts=contacts,target_distance=distance,
            target_relative_position=(data.body_xpos[target["body"]]-eef).tolist(),
            eef=eef.tolist(),quat=np.asarray(obs["robot0_eef_quat"],float).tolist(),
            progress_vector=progress.tolist(),joints=joints,normalized_goal_distance=goal_distance,
            command=np.asarray(command,float).tolist())


def last_contact_reference(rows):
    if not rows:
        return None
    target=rows[-1]["target"]
    row=next((row for row in reversed(rows) if row["target"]==target and row["target_contact"]),None)
    if row is None:
        return None
    return dict(eef=row["eef"],quat=row["quat"],grip=float(rows[-1]["command"][6]),
                contact_step=row["step"],target=target,retreat_origin=rows[-1]["eef"])
