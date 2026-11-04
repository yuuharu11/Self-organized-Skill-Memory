"""Bounded three-replica scheduler with one preregistered follow-up branch."""
import argparse
import concurrent.futures
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from effect_worker import write
from rescue_protocol import MAIN_CASES,SEEDS,case_ids
from rescue_report import summarize


def run_group(root,port,jobs,deadline,launch_deadline):
    outcomes=[]
    for job in jobs:
        if time.time()>=launch_deadline or (root/"STOP_REQUESTED").exists():
            break
        directory=root/job["directory"]
        if (directory/"job.json").exists():
            outcomes.append(dict(directory=job["directory"],status="previous_attempt_not_retried"));continue
        directory.mkdir(parents=True,exist_ok=True)
        job=dict(job,port=port,hard_deadline=deadline)
        jobfile=root/"jobs"/(directory.name+".json")
        write(jobfile,job)
        log=(directory/"worker.log").open("a")
        process=subprocess.Popen(["bash",str(root/"run_cpu.sh"),str(root/"frozen/rescue_worker.py"),
            "--root",str(root),"--job",str(jobfile)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while process.poll() is None:
            if time.time()>=deadline or (root/"STOP_REQUESTED").exists():
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL)
                break
            time.sleep(1)
        code=process.wait();log.close()
        row=dict(directory=job["directory"],returncode=code,
                 status="complete" if (directory/"result.json").exists() else "incomplete")
        outcomes.append(row);print(json.dumps(row),flush=True)
    return outcomes


def phase(root,name,groups,manifest):
    write(root/"status.json",dict(phase=name,started_utc=datetime.now(timezone.utc).isoformat(),groups=len(groups)))
    outcomes=[];iterator=iter(groups)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(manifest["ports"])) as pool:
        active={}
        for port in manifest["ports"]:
            group=next(iterator,None)
            if group:
                active[pool.submit(run_group,root,port,group,manifest["collect_deadline"],manifest["launch_deadline"])]=port
        while active:
            done,_=concurrent.futures.wait(active,timeout=30,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                port=active.pop(future);outcomes.extend(future.result())
                write(root/(name+"_outcomes.json"),outcomes)
                if time.time()<manifest["launch_deadline"]:
                    group=next(iterator,None)
                    if group:
                        active[pool.submit(run_group,root,port,group,manifest["collect_deadline"],manifest["launch_deadline"])]=port
            summarize(root)
    return outcomes


def main(root):
    manifest=json.loads((root/"manifest.json").read_text())
    (root/"jobs").mkdir(exist_ok=True)
    while not (root/"PROMPTS_FROZEN").exists() or not (root/"REPLAY_FINISHED").exists():
        if time.time()>manifest["started_epoch"]+3600:
            raise TimeoutError("preparation exceeded the first-hour allocation")
        time.sleep(2)
    if not (root/"PARITY_PASS").exists():
        raise RuntimeError("public sampler parity gate not passed")
    groups=[]
    for case in MAIN_CASES:
        promptfile=root/"prompts"/(case+".json")
        if not promptfile.exists():
            continue
        info=json.loads(promptfile.read_text());task,init=case_ids(case)
        groups.append([dict(case=case,task=task,init=init,seed=seed,condition=condition,mode="main",
            snapshot=info["snapshot"],prompts_file=str(promptfile.relative_to(root)),execute_steps=5,
            directory=f"main/{case}-seed{seed}-{condition}") for seed in SEEDS for condition in ("A","B","C","D")])
    phase(root,"main",groups,manifest)
    decision=summarize(root)["main_decision"]
    write(root/"MAIN_DECISION.json",decision)
    if time.time()<manifest["launch_deadline"]:
        # Imported only when needed, so the follow-up implementation is checked
        # separately before any follow-up job is submitted.
        from rescue_followup import prepare_followup
        name,groups=prepare_followup(root,decision)
        if groups:
            phase(root,name,groups,manifest)
    write(root/"PIPELINE_FINISHED.json",dict(time=datetime.now(timezone.utc).isoformat()))
    report=summarize(root)
    target=(Path(__file__).resolve().parents[4]
            /"docs/research/05_skillnet_diagnosis/rescue/results.md")
    target.write_text((root/"analysis/report.md").read_text()+"\nローカル詳細：`"+str(root)+"`。\n")
    print(json.dumps(dict(finished=True,counts=report["counts"],adoption=report["adoption"]),ensure_ascii=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);a=p.parse_args()
    try:
        main(a.root)
    except Exception as exc:
        write(a.root/"PIPELINE_ERROR.json",dict(error=repr(exc),time=datetime.now(timezone.utc).isoformat()))
        summarize(a.root)
        raise
