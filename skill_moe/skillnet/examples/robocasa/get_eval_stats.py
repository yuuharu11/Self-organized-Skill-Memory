"""Print success rates and Wilson 95% confidence intervals."""
import argparse
import json
import pathlib
import math


def wilson(successes, n, z=1.959963984540054):
    if n == 0:
        return None
    p = successes / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return [c-h, c+h]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    args = parser.parse_args()
    payload = json.loads(pathlib.Path(args.results).read_text())
    rows = payload.get("records", payload if isinstance(payload, list) else [])
    grouped = {}
    for item in rows:
        grouped.setdefault(item["task"], []).append(bool(item["success"]))
    total_n = total_s = 0
    for task, vals in sorted(grouped.items()):
        s, n = sum(vals), len(vals)
        total_s += s; total_n += n
        ci = wilson(s,n)
        print(f"{task}: {s}/{n}={s/n:.3f}, Wilson95=[{ci[0]:.3f}, {ci[1]:.3f}]")
    if total_n:
        ci = wilson(total_s,total_n)
        print(f"overall: {total_s}/{total_n}={total_s/total_n:.3f}, Wilson95=[{ci[0]:.3f}, {ci[1]:.3f}]")


if __name__ == "__main__":
    main()
