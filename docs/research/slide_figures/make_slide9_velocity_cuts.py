import json
from pathlib import Path
HERE = Path(__file__).parent
# data: motion_core_eval.anchor_spans on libero_90 STUDY_SCENE3 red mug, demo_0, stride 2
d=json.load(open(HERE/'slide9_velocity_cuts_data.json'))
INK="#1B2433"; BODY="#3F4A5A"; ACC="#2F6FB5"; ORG="#D9682E"; GRID="#D5D9DE"; FAINT="#9DB8D9"
F='font-family="Noto Sans JP, sans-serif"'
W,H=900,500; L,R,T,B=90,20,70,150
n=len(d['smooth']); sm=d['smooth']; ymax=max(d['speed'])*1.05
sx=lambda t: L+(W-L-R)*t/(n-1)
sy=lambda v: T+(H-T-B)*(1-v/ymax)
o=[f'<svg aria-label="移動速度の曲線と切れ目。速度の極小とグリッパーの切替で切る" viewBox="0 0 {W} {H}" width="{W}" height="{H}">']
# legend
o.append(f'<line x1="{L}" x2="{L+40}" y1="22" y2="22" stroke="{ACC}" stroke-width="4"/><text x="{L+50}" y="30" font-size="24" fill="{INK}" {F}>速度の極小で切る</text>')
o.append(f'<line x1="{L+290}" x2="{L+330}" y1="22" y2="22" stroke="{ORG}" stroke-width="4"/><text x="{L+340}" y="30" font-size="24" fill="{INK}" {F}>グリッパー切替で切る</text>')
# axes
o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(0)}" y2="{sy(0)}" stroke="{GRID}" stroke-width="2"/>')
o.append(f'<text x="{L-10}" y="{T+10}" font-size="24" fill="{BODY}" text-anchor="end" {F}>速</text><text x="{L-10}" y="{sy(0)}" font-size="24" fill="{BODY}" text-anchor="end" {F}>0</text>')
# skipped minima
cuts=[s for s,_ in d['spans']][1:]
for m in d['minima']:
    if m not in cuts:
        o.append(f'<line x1="{sx(m)}" x2="{sx(m)}" y1="{T}" y2="{sy(0)}" stroke="{BODY}" stroke-width="2" stroke-dasharray="6 6"/>')
        o.append(f'<text x="{sx(m)+6}" y="{T+28}" font-size="24" fill="{BODY}" {F}>短すぎるので切らない</text>')
for c in cuts:
    col=ORG if c in d['switch'] else ACC
    o.append(f'<line x1="{sx(c)}" x2="{sx(c)}" y1="{T}" y2="{sy(0)+60}" stroke="{col}" stroke-width="4"/>')
# raw and smooth speed
o.append('<polyline fill="none" stroke="%s" stroke-width="2" points="%s"/>'%(FAINT,' '.join(f'{sx(t):.1f},{sy(v):.1f}' for t,v in enumerate(d['speed']))))
o.append('<polyline fill="none" stroke="%s" stroke-width="4" points="%s"/>'%(INK,' '.join(f'{sx(t):.1f},{sy(v):.1f}' for t,v in enumerate(sm))))
for m in d['minima']:
    if m in cuts: o.append(f'<circle cx="{sx(m)}" cy="{sy(sm[m])}" r="8" fill="{ACC}"/>')
# gripper band
gy=sy(0)+18
o.append(f'<text x="{L-10}" y="{gy+26}" font-size="24" fill="{BODY}" text-anchor="end" {F}>手</text>')
t0=0
g=d['grip']
for t in range(1,n+1):
    if t==n or (g[t]>0)!=(g[t0]>0):
        closed=g[t0]>0
        o.append(f'<rect x="{sx(t0)}" y="{gy}" width="{max(sx(min(t,n-1))-sx(t0),3):.1f}" height="36" fill="{"#5B6576" if closed else "#E3E6EA"}"/>')
        t0=t
o.append(f'<text x="{sx(4)}" y="{gy+28}" font-size="24" fill="{INK}" {F}>開</text><text x="{sx(40)}" y="{gy+28}" font-size="24" fill="#F6F5F1" {F}>閉</text>')
# segment numbers
for i,(s,e) in enumerate(d['spans']):
    if e-s>=3: o.append(f'<text x="{(sx(s)+sx(e))/2}" y="{gy+80}" font-size="24" fill="{INK}" text-anchor="middle" {F}>{i+1}</text>')
o.append(f'<text x="{W-R}" y="{H-8}" font-size="24" fill="{BODY}" text-anchor="end" {F}>ステップ（10 Hz、6秒）→</text>')
o.append('</svg>')
s=''.join(o); open(HERE/'slide9_velocity_cuts.svg','w').write(s.replace('<svg ','<svg xmlns="http://www.w3.org/2000/svg" ',1)); print(len(s))
