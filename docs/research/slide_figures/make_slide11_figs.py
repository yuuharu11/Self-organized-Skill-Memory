"""Slide 11 figures as self-contained SVG (Japanese text rendered by the viewer's fonts)."""
import json
import sys
from pathlib import Path

MAG3 = Path("/work/.skillnet-state/outputs/memory-pilot/unseen-libero/magnitude-03/offline_report.json")
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

STYLE = """<style>
  .bg{fill:#fcfcfb} .ink{fill:#0b0b0b} .ink2{fill:#52514e} .grid{stroke:#e4e3df}
  .bar{fill:#2a78d6} .bar2{fill:#86b6ef} .ref{stroke:#52514e} .refl{fill:#52514e}
  text{font-family:"Hiragino Sans","Noto Sans CJK JP","Yu Gothic",sans-serif}
  @media (prefers-color-scheme: dark){
    .bg{fill:#1a1a19} .ink{fill:#f0efec} .ink2{fill:#c3c2b7} .grid{stroke:#383835}
    .bar{fill:#3987e5} .bar2{fill:#184f95} .ref{stroke:#c3c2b7} .refl{fill:#c3c2b7}
  }
</style>"""


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def panel(x0, y0, w, title, rows, xmax, refs, label_w=170):
    """rows: [(label, sublabel, value, highlight)] ; refs: [(value, text)]"""
    out = []
    plot_x, plot_w = x0 + label_w, w - label_w - 50
    bar_h, gap = 22, 18
    out.append(f'<text class="ink" x="{x0}" y="{y0}" font-size="15" font-weight="600">{esc(title)}</text>')
    top = y0 + 18
    h = len(rows) * (bar_h + gap)
    sx = lambda v: plot_x + plot_w * v / xmax
    for t in [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        if t > xmax + 1e-9:
            break
        out.append(f'<line class="grid" x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{top}" y2="{top + h}" stroke-width="1"/>')
        out.append(f'<text class="ink2" x="{sx(t):.1f}" y="{top + h + 16}" font-size="11" text-anchor="middle">{t:.2f}</text>')
    for i, (lab, sub, v, hi) in enumerate(rows):
        y = top + i * (bar_h + gap) + gap / 2
        out.append(f'<text class="ink" x="{plot_x - 8}" y="{y + (10 if sub else 16)}" font-size="13" text-anchor="end">{esc(lab)}</text>')
        if sub:
            out.append(f'<text class="ink2" x="{plot_x - 8}" y="{y + 24}" font-size="10.5" text-anchor="end">{esc(sub)}</text>')
        cls = "bar" if hi else "bar2"
        out.append(f'<rect class="{cls}" x="{plot_x}" y="{y}" width="{max(sx(v) - plot_x, 1):.1f}" height="{bar_h}" rx="4"/>')
        out.append(f'<text class="ink" x="{sx(v) + 5:.1f}" y="{y + 15}" font-size="12">{v:.3f}</text>')
    for v, t in refs:
        out.append(f'<line class="ref" x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{top - 4}" y2="{top + h}" stroke-width="1.5" stroke-dasharray="4 3"/>')
        out.append(f'<text class="refl" x="{sx(v) + 4:.1f}" y="{top + 6}" font-size="11">{esc(t)}</text>')
    return out, top + h + 30


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img">'
            f"<title>{esc(title)}</title>{STYLE}<rect class=\"bg\" width=\"{w}\" height=\"{h}\"/>" + "".join(body) + "</svg>\n")


# ---------- Figure 11-A: orientation-only vs magnitude-only (MAG3, Eval, rule picked on Dev)
d = json.loads(MAG3.read_text())
summ = {}
for c in d["combinations"]:
    summ.setdefault((c["pool"], c["orientation"], c["amplitude"]), {})[(c["split"], c["window"])] = c["mean"]


def mean(pool, o, a, split):
    v = summ[(pool, o, a)]
    return (v[(split, "first5")] + v[(split, "continuation5_9")]) / 2


def cat(o, a):
    if o == "O4_oracle":
        return "both" if a == "A1_oracle" else "ori"
    return "mag" if a == "A1_oracle" else "none"


ONAME = {"O1_source": "出どころ", "O2_pi05": "π0.5", "O3_history": "直前の動き", "O4_oracle": "正解"}
ANAME = {"A1_oracle": "正解", "A2_history": "直前の速さ", "A3_pi05": "π0.5", "A4_source": "保存値"}
pi05 = (d["baseline"]["eval"]["pi05_first5"] + d["baseline"]["eval"]["pi05_continuation5_9"]) / 2
picked = {}
body, y = [], 34
body.append('<text class="ink" x="20" y="22" font-size="13">6軸 正規化MSE（Eval、先頭5と続き5〜9の平均、小さいほど良い）</text>')
body.append('<text class="ink2" x="20" y="40" font-size="12">型は正解で選択。推定側の規則は Dev で最良のもの（下段の小さい字）</text>')
y = 52
for pool, title in [("pool_unit_08", "95型（向き・大きさを外した型、43/100 の構成）"),
                    ("scale_only_W", "1,595型（向きを型に残す scale-only W）")]:
    best = {}
    for (p, o, a) in summ:
        if p != pool or o in ("pi05", "previous_hold"):
            continue
        k = cat(o, a)
        dv = mean(pool, o, a, "dev")
        if k not in best or dv < best[k][0]:
            best[k] = (dv, o, a)
    picked[pool] = {k: (o, a, mean(pool, o, a, "eval")) for k, (_, o, a) in best.items()}
    rows = []
    for k, lab in [("both", "両方正解"), ("ori", "向きだけ正解"), ("mag", "大きさだけ正解"), ("none", "どちらも推定")]:
        _, o, a = best[k]
        rows.append((lab, f"向き={ONAME[o]}・大きさ={ANAME[a]}", mean(pool, o, a, "eval"), k in ("both", "ori", "mag")))
    part, y = panel(20, y + 22, 720, title, rows, 0.25, [(pi05, f"通常 π0.5 {pi05:.3f}")], label_w=240)
    body += part
(OUT / "slide11_orientation_vs_magnitude.svg").write_text(
    svg(760, y, body, "向きだけ正解と大きさだけ正解の行動誤差（MAG3）"))
print(json.dumps({p: {k: [o, a, round(v, 4)] for k, (o, a, v) in r.items()} for p, r in picked.items()}, ensure_ascii=False, indent=1))
print("pi05 eval mean", round(pi05, 4))

# ---------- Figure 11-B: table 11-1 (oracle type, overall)
rows = [("θ*（正解に合わせる）", "上限", .088, True),
        ("LoRA θ†", "", .186, False),
        ("flow†", "", .205, False),
        ("bridge†", "", .206, False),
        ("回帰†", "", .221, False),
        ("π0.5 の出力に合わせる†", "", .244, False),
        ("型の出どころの θ", "", .261, False)]
body = ['<text class="ink" x="20" y="22" font-size="13">θ の決め方別の誤差（active7、oracle の型、全体。† は逆さ画像の時点の値）</text>']
part, y = panel(20, 56, 640, "推定した θ はどれも上限から遠い", rows, 0.30, [(.088, "θ* 上限")], label_w=200)
body += part
(OUT / "slide11_theta_methods.svg").write_text(svg(680, y, body, "θ の決め方別の誤差（表11-1）"))
