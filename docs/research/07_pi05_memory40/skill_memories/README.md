# 複数の運動メモリ：raw区間とM6単位

## 結論

Long除外の生行動メモリは、Long hard 63行で正解との最良距離が .0399。M6 medoid単位では最良でも .1320（κ=.1）、b12-M6は .1310だった。代表へまとめることで、Long場面に近い実区間へのアクセスが悪化している。一方、既存のwrist+state cosineでkey-ranked Top-5を取るとraw行メモリもhard MSE .2089で、近い実区間をキーだけで選ぶ問題は残る。

次にExpertを学習するなら、**raw-rows-noLongを主メモリ**にし、raw-m6-noLong-k0p1を容量削減の対照として並べる。raw区間は行動の情報を保持するが、現在のkey Top-5では弱い。M6は少数単位にまとめやすい一方、oracle候補距離の悪化が大きい。routerは未学習なので、この比較だけではどちらがExpert学習後に有効かは決められない。

## 作り方と制約

- 入力は `run-01/rows` の固定10-step/stride10行、`memory` 分割。raw actionは `[10,32]`、機能距離はnormalized active7の10×7 MSE。
- M6登録は既存 `emergent/light/methods/m6.py` をCPUで模した。固定Kやk-meansは使わず、ε以内への割当、pending区間のgreedy εまとまり、min_support=10かつmin_episodes=3で登録、近い既存単位への統合。κはM6既定の `epsilon=κ×median(random-pair functional distance)`。
- Long除外はtask id 0–9。noLongのmemory poolは3,857行。全40 poolは6,583行。Long Eval 548行・hard63行はmemory-gap固定行で、topk-longとquery順・hard maskが一致することをassertした。noLong poolやDevにはLongまたはLong Eval行が入らないこともassertした。
- κ候補をLong以外のDev episodeだけで選択した。Dev登録数はκ=.0125/.025/.05/.1/.2/.4/.8で1/2/12/14/12/3/3。事前ルールで低κ側を300目標に最も近い.05、高κ側を30目標に最も近い.2、中央を.1とした。ただしDevで数十〜数百の登録数に届かず、実memoryでも3値は16/25/32単位に留まった。値はLong Evalを見て変更していない。
- `b12-m6-noLong` は保存済み`run-01/bases.npy`とM6のdecoder functional distanceを使用。64個のprobeはnoLong memoryから固定seedで選択し、M6のbase-L2 top10 prefilterを適用した。既存codecは全40 taskの非validationデータでfit済みなので、b表現の重みはLong学習データを含む。noLongの値・所属・medoid・EvalにはLong行を使っていない。
- router用キーは各representativeのwrist token mean＋`state[:8]`を保存した。補助のTop-5診断だけ、既存retrieval-4bの train-standardized `wrist_state_cos` でkey順位を付けた。router学習は行っていない。
- 通常π0.5のLong single出力は all .00684 / hard .04557、zero draft生成後はt0=.2で .01061 / .04016、t0=.4で .00623 / .03412。これらは生成後MSEであり、下表の生成前代表距離と直接の同一指標比較ではない。

## 各メモリとLong診断

「最良代表」は正解行動に対する全代表内のoracle最近傍。「key Top-5」はretrieval-4bのwrist+state cosine上位5候補のうち正解に最も近い距離。どちらも生成前normalized active7 MSEで、生成後性能・実行時のoracleではない。unassignedはM6 poolのうちsupported unitへ登録されずpendingに残った割合。

| ID | pool | κ | units | unassigned | hard最近傍oracle MSE | hard key Top-5 best MSE |
|---|---|---:|---:|---:|---:|---:|
| raw-rows-noLong | Long以外30 task、memory行 | — | 3,857 | 0% | .03993 | .20894 |
| raw-rows-all40 | 全40 task、memory行 | — | 6,583 | 0% | .02881 | .15689 |
| raw-m6-noLong-k0p05 | Long以外30 task | .05 | 25 | 59.1% | .18249 | .17492 |
| raw-m6-noLong-k0p1 | Long以外30 task | .1 | 32 | 22.9% | .13197 | .13248 |
| raw-m6-noLong-k0p2 | Long以外30 task | .2 | 16 | 5.5% | .18864 | .15770 |
| raw-m6-all40 | 全40 task | .1 | 42 | 23.5% | .10309 | .11078 |
| b12-m6-noLong | Long以外30 task | .1 | 31 | 24.9% | .13097 | .15124 |

### 単位の共有・長さ

| ID | unit長 median / p90 / max | unitが複数taskに跨る割合 |
|---|---:|---:|
| raw-m6-noLong-k0p05 | 51 / 102 / 275 | 100% (25/25) |
| raw-m6-noLong-k0p1 | 47 / 225 / 401 | 96.9% (31/32) |
| raw-m6-noLong-k0p2 | 117 / 695 / 735 | 87.5% (14/16) |
| raw-m6-all40 | 61 / 304 / 618 | 97.6% (41/42) |
| b12-m6-noLong | 60 / 197 / 358 | 100% (31/31) |

各`<ID>/diagnostics.json`には単位長・task/episode数の分布と代表例を保存している。代表例はnormalized Δxyzの平均方向とgrip channelの開始・終了sign（−をclose、＋をopenとして要約）。例えばκ=.1 noLongの先頭単位は319区間/30 task、代表はrow15416、平均方向−z、grip close→close。b12-M6の先頭単位は314区間/30 task、同じ代表row15416。代表行動や所属は`units.npz`、router用入力は`keys.npz`。

## 再現

- Builder：[build.py](../../../../skill_moe/skillnet/examples/memory_pilot/pi05_memory/skill_memories/build.py)
- CPU test：[test_pi05_skill_memories.py](../../../../tests/test_pi05_skill_memories.py)
- 出力：[skill-memories-01](../../../../.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/skill-memories-01/)
- 実行例：`cd /work && source env/activate.sh && OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 taskset -c 8-15 nice -n 10 python -u skill_moe/skillnet/examples/memory_pilot/pi05_memory/skill_memories/build.py`

Expert学習、GPU推論、シミュレーションでの閉ループ評価は行っていない。
