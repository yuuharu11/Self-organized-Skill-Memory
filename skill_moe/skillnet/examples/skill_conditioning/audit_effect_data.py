"""Check branch pairing, runtime feature boundaries, and test partitioning."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def audit(root):
    rows=[]
    groups=defaultdict(dict)
    for result in sorted(root.glob("*/*/result.json")):
        data=json.loads(result.read_text())
        job=data["job"]
        assert data["status"]=="complete"
        assert 0 <= data["start_step"] <= data["end_step"] <= 800
        if job["kind"]=="parent":
            assert job["init"] in ({10,11} if job["split"]=="train" else {12})
            assert job["task"] in (0,1,2,3,4,6,7)
        if job["kind"]=="evaluation":
            assert job["init"] in (20,21) and job["seed"]==43
        if job["kind"]=="pair":
            replay=json.loads((result.parent/"replay_audit.json").read_text())
            assert replay["state_max_abs"]==0 and replay["images_equal"]
            assert job["init"] in (10,11,12)
            queries=[json.loads(line) for line in (result.parent/"queries.jsonl").read_text().splitlines()]
            for query in queries:
                assert query["seed"]==job["seed"]*1000000+job["init"]*1000+query["step"]//5
            key=(job["parent"],job["snapshot"],job["seed"])
            groups[key][job["arm"]]=dict(start=data["start_step"],queries=queries)
        rows.append(str(result.relative_to(root)))
    pairs=0
    for arms in groups.values():
        if set(arms)!={"keep","update"}:continue
        a,b=arms["keep"],arms["update"]
        assert a["start"]==b["start"]
        assert a["queries"][0]["image_sha256"]==b["queries"][0]["image_sha256"]
        assert a["queries"][0]["state"]==b["queries"][0]["state"]
        common=min(len(a["queries"]),len(b["queries"]))
        assert [q["seed"] for q in a["queries"][:common]]==[q["seed"] for q in b["queries"][:common]]
        pairs+=1
    report=dict(complete_results_checked=len(rows),complete_seed_pairs_checked=pairs,passed=True)
    (root/"analysis").mkdir(exist_ok=True)
    (root/"analysis/data_audit.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True)
    args=parser.parse_args();print(json.dumps(audit(args.root)))
