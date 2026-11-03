"""Audit unchanged public actions and freeze prompts before branch outcomes."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from collect_interventions import S2WebsocketClientPolicy,image_hash,policy_input,rpc
from effect_worker import setup,write
from rescue_protocol import MAIN_CASES,base_prompts,case_ids,match_prompts,read_rows
from rescue_runtime import ordered_task,restore_snapshot,token_rpc


def parity(root):
    manifest=json.loads((root/"manifest.json").read_text());source=Path(manifest["source_root"])
    c,_=setup(root);rows=[]
    clients={port:S2WebsocketClientPolicy("127.0.0.1",port) for port in manifest["ports"]}
    try:
        for task in (0,2,4):
            case=f"task{task+1:02d}-init20"
            old=read_rows(source/"evaluation"/(case+"-original")/"queries.jsonl")[0]
            env,obs,full,ann=c.setup(task,20)
            try:
                raw=policy_input(obs,full,ann,init=20,query=0,evaluator=c.evaluator)
                assert image_hash(raw)==old["image_sha256"]
                raw["s2_rng_seed"]=old["seed"]
                for port,client in clients.items():
                    original=rpc(client,raw)
                    audited=rpc(client,dict(raw,diagnostic_tokens=True))
                    row=dict(task=task,port=port,stored_public_max_abs=float(np.max(np.abs(
                        np.asarray(original["actions"])[:5,:7]-old["actions"]))),
                        token_audit_action_max_abs=float(np.max(np.abs(np.asarray(original["actions"])-audited["actions"]))),
                        tokens=audited["token_audit"])
                    rows.append(row);write(root/"public_parity.json",rows)
                    print(json.dumps({k:v for k,v in row.items() if k!="tokens"}),flush=True)
                    if row["stored_public_max_abs"]>1e-6 or row["token_audit_action_max_abs"]>1e-6 or not row["tokens"]["exact_encoding"]:
                        raise RuntimeError("public action/token audit gate failed")
            finally:
                env.close()
    finally:
        for client in clients.values():
            client.close()
    (root/"PARITY_PASS").write_text("Stored public first five actions and separate token auditing agree.\n")


def prompts(root,cases=MAIN_CASES):
    manifest=json.loads((root/"manifest.json").read_text());c,stages=setup(root)
    client=S2WebsocketClientPolicy("127.0.0.1",manifest["ports"][0])
    missing=[]
    try:
        for case in cases:
            result=root/"replay"/case/"result.json"
            if not result.exists():
                missing.append(dict(case=case,reason="replay unavailable"));continue
            info=json.loads(result.read_text())
            if not info["achievement_snapshot"]:
                missing.append(dict(case=case,reason="no reproducible eligible achievement state"));continue
            task,init=case_ids(case)
            env,obs,full,ann,_,audit=restore_snapshot(c,stages,task,init,root/info["achievement_snapshot"])
            try:
                _,remaining=ordered_task(env,stages,task,ann)
                raw=policy_input(obs,full,ann,init=init,query=audit["step"],evaluator=c.evaluator)
                try:
                    payload=match_prompts(base_prompts(full,remaining),lambda prompt:token_rpc(client,dict(raw,prompt=prompt)))
                except ValueError as exc:
                    missing.append(dict(case=case,reason=str(exc)));continue
                payload.update(case=case,snapshot=info["achievement_snapshot"],replay_audit=audit,
                               original_success=info["original_success"],group=info["group"])
                (root/"prompts").mkdir(exist_ok=True)
                write(root/"prompts"/(case+".json"),payload)
                print(json.dumps(dict(case=case,lengths={k:v["length"] for k,v in payload["audits"].items()},padding=payload["neutral_please_additions"])),flush=True)
            finally:
                env.close()
    finally:
        client.close()
    write(root/"prompt_missing.json",missing)
    (root/"PROMPTS_FROZEN").write_text(str(time.time()))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("phase",choices=["parity","prompts"]);p.add_argument("--root",type=Path,required=True)
    a=p.parse_args();(parity if a.phase=="parity" else prompts)(a.root)
