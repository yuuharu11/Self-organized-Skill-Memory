# アルゴリズム仕様

ここでは生の行動区間メモリ、M2検索、下書きから修正するAction Expertを定義する。タスク除外ID、データsplit、学習更新数、checkpoint選択はデータセット側が指定する。

## 1. 区間メモリ

各エピソードの正規化行動列を `A_e∈R^{T×32}` とし、horizon `H=10`、stride `S=10` で開始位置 `u=0,S,2S,...` の窓 `A_e[u:u+H]` を作る。末尾が不完全な窓は既定で破棄する。各区間には全データセットで一意な `row_id`、`task_id`、`episode_id` を付与する。保存値は生の `[10,32]` 行動。検索候補のIDは必ず `row_id` で、候補取得は `dataset_actions[row_id]` とする。メモリ-local offsetは検索用配列へアクセスする内部対応表に限る。

メモリ候補を選ぶ前に、目的に応じて同一task、同一episode、または評価suiteを除く。Coreは特定suite番号を固定しない。

## 2. 文脈とM2 key

観測側 prefix の512画像tokenを `X∈R^{512×2048}`、state/history contextを `c∈R^{79}` とする。文脈は8状態値＋70直前行動値＋flagが入る。既存重みの再現では `build_context(...,layout="as_trained")` を使う。既知slice不具合のため、意図された連結に直す `fixed` は既存重みと別モデルである。

各tokenの隠れ埋め込み:

```text
h_i = normalize(W_tok x_i + p_i)                     W_tok: 2048→96
g   = W_c2 GELU(W_c1 c)                               W_c1: 79→96, W_c2: 96→96
l_i = w_s2 tanh(w_s1 tanh(h_i + g))                   w_s1: 96→48, w_s2: 48→1
α   = softmax(l masked by valid tokens); invalid α_i=0
z   = normalize(W_o Σ_i α_i h_i)                      W_o: 96→96
```

学習ではaction距離をteacherとしたmulti-positive in-batch InfoNCEを使う。二つの行動windowの距離はactive7の70数値の平均二乗誤差。positiveは `distance≤τ` かつtaskが異なるもの。queryとpositiveの同task・同一index pairはpositiveから外す。`sim_ij = z_i·z_j / 0.12`; 各anchorでpositive logitsをlogsumexpし、分母は全batch（self除外）のlogsumexpとする。M2学習のlossはpooled `z` のみ。token salienceはこのlossから間接的にのみ学習される。

保存時、`h` と `α` をそれぞれfp16へ丸める。検索ではfp16保存値をfp32へ戻し、再正規化せず内積する。

## 3. 二段検索

1. pooled query/memory keysをL2正規化し、cosine `Q Mᵀ` を計算する。同一taskのmemory行を `−∞` にする。候補は `lexsort((dataset_row_id, -cosine))` 上位K。既定K=64。
2. 各queryと各候補について保存済みtoken `q_i,m_j` の内積行列 `S_ij=q_i·m_j` を作る。token maskによりsalience weightがゼロのtokenは加点しない。
3. 双方向MaxSim:

```text
a = Σ_i α^q_i max_j S_ij
b = Σ_j α^m_j max_i S_ij
score = (a+b)/2
```

4. score降順で安定ソートする。等値はcoarse候補順を保持し、coarse順はscore同値ならdataset row ID順。

一つの実演を選ぶ場合、Top-1 `row_id`から元の生行動を読む。候補チャンクを平均して新しい動作へしない。

## 4. 学習下書きとシャッフル

候補ID列 `r_1…r_K` から一つだけ選ぶ。標準学習分布は `P(rank=1)=0.70`、`P(rank=j)=0.075` for `j=2…5`。乱数抽選はqueryごとに独立。下書きは `D=dataset_actions[r_rank]`。評価行のcross-task shuffle対照はこの分布で候補を一つ抽選した後、query間で並べ替え、各queryのtaskと異なることを検査する。

## 5. Expert bridge学習

正解行動 `A∈R^{B×10×32}`、下書き `D`、noise `ε~N(0,I)` に対し独立に `t0~Uniform(.2,.6)`、`u~Uniform(.01,1)` を引く。

```text
B0 = (1−t0)D + t0 ε
X_u = (1−u)A + u B0
v* = (B0−A)/t0
t = t0 u
L = mean((vθ(X_u,t)[:,:,0:7]−v*[:,:,0:7])²)
```

ここで `B0` はbridge source（開始点）を表し、通常の行動`A`と混同しない。更新対象はAction Expert側の `llm.*_1|action_in_proj|time_mlp|action_out_proj` に合うパラメータ。AdamW lr `2.5e-5`、β1 `.9`、β2 `.95`、weight decay `1e-5`、global gradient norm clip `1`。

## 6. 推論

下書き `D` を選び `X_0=(1−t0)D+t0 ε` とする。`i=0…n−1` で `t_i=t0(1−i/n)`、

```text
X_{i+1} = X_i − (t0/n) vθ(X_i,t_i)
```

を計算し、`t0→0` の出力を得る。prefix KVを一度作成し、suffix velocityを各stepで計算する。通常推論は別の明示的な呼び出し先であり、本Coreに成功条件依存のfallback gateは含めない。

## 7. M6風の可変個数Raw単位

単位形成だけが必要な場合、各70-D active actionを `x/√70` に写す。機能距離は二乗L2、すなわち全70成分の平均二乗誤差。

1. `seed+6046` の固定乱数で64組の行を引き、機能距離中央値 `m` を得る。しきい値 `ε=kappa·m`（override可能）。
2. 同seedのランダム行順を1回走査。最近傍centerがε以内なら既存単位へオンライン平均centerを更新。そうでなければ行をpendingへ。
3. pendingが200到達するたび（最後は強制）、pending値を順番にε連結距離で仮cluster化。supportが10区間以上かつ(task,episode)が3以上のclusterを登録。既存centerからε以内ならmerge、さもなくば新規単位。
4. 未登録pendingは再試行用に保持し、ageが1000に到達するとexpireする。`row_units=-1` が未割当行。単位数はデータから決まり、K-meansや固定個数を指定しない。
5. 代表窓は所属実行区間からmedoid（単位内平均距離最小）として選ぶ。候補再利用の際に代表に置き換えると距離が変わるため、全member IDを残す。
