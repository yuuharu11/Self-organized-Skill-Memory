# Fair Expert: 検索動作の内容を公平な学習条件で比較

評価前に事前固定。目的は、M2 が返す生行動の下書きが、同じ初期値・更新予算で学習したゼロ専用Expertより生成後誤差を改善するかをオフラインで判定すること。

## 固定条件

- データ/方策/codec は `pi05-memory40-20261104/run-01` を読み取り専用で使用。M2 は `key-mechanism-01/production/models.pt` の確定済み M2SalientTokens を使い、候補メモリは `skill-memories-01/raw-rows-noLong` の 3,857 行。
- 学習行は `~validation` かつ task ID が 10 以上、かつ保留 task `{17,19,28,29,35,38}` 以外。したがって Long task 0–9 は使わない。選択用 Dev は保留6 task の validation 行だけ。Eval は固定 calibration Eval から Long 548 行、hard 63 行。
- M2 検索は本番 key-mechanism と同じ coarse top-64 と salient token MaxSim rerank。候補は別 task のみ。保存される候補は元データの row ID。下書きは必ず `rows['actions'][candidate_row_id]` の raw normalized action chunk を使い、codec復元も混合も行わない。
- E-M2 と E-zero は通常Expertの同じ初期値から独立学習。seed 20261104、同一の行IDバッチ列、batch 8、AdamW 2.5e-5、make_bridge_step と同じ flow bridge 損失、t0~U[.2,.6]、最大4,000更新、500更新保存。E-M2だけ下書きが M2 top-1 (p=.7) / rank 2–5 一様の1候補 (p=.3)。E-zero は常にゼロ。
- 各Expertの checkpoint と t0 は、保留6 task validation 行で、当該Expert専用下書きに対する同一ノイズ生成後 active7 MSE を最小化して選択。候補 t0 は {.2,.4,.6}、Long Eval は選択に使わない。
- 評価は固定 Long 548 / hard 63 行、10積分、noise seed 4101。条件は E-M2+M2 top1/shuffle/zero、E-zero+zero、旧 001500 Expert+M2/zero、公式 pi05、key-mechanism 直接回帰参照。各t0 {.2,.4,.6} を報告。差のCIは episode cluster bootstrap 5,000回。

## 事前判定

選択された t0 で一度だけ判定する。F1 は `(E-M2+M2) - (E-zero+zero)`、F2 は `(E-M2+M2) - (E-M2+shuffle)` の episode bootstrap 95% CI 上端が、全Longまたは hard のいずれかで 0 未満なら成立。統計的有意差がないことを同等とは解釈しない。F3 の公式pi05比較は参考で、通常方策とExpertの初期値・学習条件が異なるため因果判定に使わない。分析はオフライン行動誤差であり閉ループ成功率ではない。

## 期待される検査

- assert で Long 行がExpert学習に入らないこと、Eval episode/taskの漏洩がないこと、全 M2候補が別taskであること、候補 row ID と `actions[row_id]` の一致、シャッフル候補の別task条件を検査する。
- E-M2/E-zero の更新回数と batch row ID 列の SHA256 が一致することを記録する。
- M2学習に使った候補メモリ由来の行と、保留6 task上の下書き誤差を分けて記録し、差を報告する。
