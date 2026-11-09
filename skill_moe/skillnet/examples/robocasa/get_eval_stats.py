#!/usr/bin/env python3
import argparse,json,pathlib
p=argparse.ArgumentParser(); p.add_argument('results'); a=p.parse_args(); d=json.loads(pathlib.Path(a.results).read_text()); by={}
for r in d['records']: by.setdefault(r['task'],[]).append(r['success'])
for task,x in by.items(): print(f'{task}: {sum(x)}/{len(x)} ({sum(x)/len(x):.3f})')
print('overall:',sum(map(sum,by.values())),'/',sum(map(len,by.values())))
