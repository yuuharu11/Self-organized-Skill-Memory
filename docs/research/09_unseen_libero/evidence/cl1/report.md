# CL1：運動キー＋Expert の閉ループ（LIBERO-90 Eval 10タスク×10試行）

状態：**complete**（完了試行数 C1a 100、C1b 100、C2 100、C3 100、pi05 100）

## 結論
- 成功数：C1a 34、C1b 29、C3 28、π0.5 32（再掲）、C2 3。どの対比較も有意ではない（C1a−C3 10勝4敗 p=.18、C1b−C3 5勝4敗 p=1.0、C1a−π0.5 5勝3敗 p=.73）。C2（下書きのまま）だけが π0.5 より明確に悪い（3勝32敗）。
- オフライン未学習 MSE の順位（下書き：C1b .084 < C1a .094 < C3 .101）は閉ループの順位（C1a 34 > C1b 29 ≈ C3 28）と一致しない。差は主に task 57（C1a 10、C1b 6、C3 4、π0.5 10）の1タスクで生じている。
- Expert が下書きを大きく直している。t0 の中央値は .20 前後で3条件ほぼ同じ。C2 の成功3回はすべて task 0（引き出しを閉める。π0.5 は 0/10）。
- 単位の出どころ：同じ動詞の別タスクが約 76%、別の動詞が約 24%。成功と失敗の差（同動詞 .89 対 .73）はタスクの違いによる。open/close の task 0・35 ではほぼ全部が別動詞の単位になる。同じタスク内で比べると差はほぼない。

## 主表：成功数

|条件|内容|成功/完了|
|---|---|---:|
|C1a|K-x1 × E-H 注意クエリ（最終行）＋Expert|34/100|
|C1b|K-x1 × 履歴＋状態MLP（CT版）＋Expert|29/100|
|C2|C1a の下書きのまま（Expert なし）|3/100|
|C3|D5 履歴＋状態MLP＋Expert|28/100|
|pi05|通常 π0.5（baseline-01 再掲）|32/100|

## タスク別成功数（/10）

|task|名前|C1a|C1b|C2|C3|pi05|
|---|---|---:|---:|---:|---:|---:|
|0|KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet|1|1|3|1|0|
|10|KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet|9|10|0|10|10|
|13|KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate|0|0|0|0|0|
|31|KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet|1|0|0|0|0|
|35|KITCHEN_SCENE7_open_the_microwave|2|1|0|2|1|
|43|KITCHEN_SCENE9_put_the_white_bowl_on_top_of_the_cabinet|1|2|0|1|1|
|57|LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_th|10|6|0|4|10|
|61|LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_|10|9|0|10|10|
|84|STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_o|0|0|0|0|0|
|88|STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_on_t|0|0|0|0|0|

## McNemar（同じ試行の対、正確二項検定）

|比較 B − A|対|B だけ成功|A だけ成功|p|
|---|---:|---:|---:|---:|
|C1a - C3|100|10|4|0.180|
|C1b - C3|100|5|4|1.000|
|C1a - pi05|100|5|3|0.727|
|C1b - pi05|100|4|7|0.549|
|C3 - pi05|100|4|8|0.388|
|C2 - pi05|100|3|32|0.000|
|C1a - C2|100|34|3|0.000|
|C1b - C1a|100|3|8|0.227|

## 信頼度・t0・単位の出どころ

t0 = clip(√ê, .05, .6)（ê＝信頼度器の予測腕誤差）。C2 は Expert を適用しないが、同じ ê と t0 を記録した。出どころは TR1 の分類（未学習タスクの指示と LIBERO-40 の元タスクの指示で動詞が重なるか）。

|条件|問い合わせ数|t0 中央値 [5%, 95%]|t0=下限 .05 の割合|t0=上限 .6 の割合|ê 中央値|同動詞の単位|別動詞の単位|
|---|---:|---|---:|---:|---:|---:|---:|
|C1a|8390|0.196 [0.102, 0.390]|0.000|0.003|0.0382|0.751|0.249|
|C1b|8646|0.197 [0.095, 0.394]|0.002|0.002|0.0388|0.762|0.238|
|C2|10282|0.345 [0.168, 0.584]|0.000|0.039|0.1190|0.808|0.192|
|C3|8576|0.206 [0.096, 0.402]|0.001|0.002|0.0424|0.765|0.235|

成功試行と失敗試行の比較（試行平均）

|条件|群|試行|平均 t0|平均 ê|切り替え回数|切り替え/問い合わせ|異なる単位数|同動詞率|別動詞率|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|C1a|成功|34|0.181|0.0384|36.4|0.850|31.6|0.885|0.115|
|C1a|失敗|66|0.221|0.0577|77.2|0.749|60.6|0.731|0.269|
|C1b|成功|29|0.182|0.0382|35.8|0.847|31.0|0.918|0.082|
|C1b|失敗|71|0.219|0.0575|80.4|0.780|60.2|0.746|0.254|
|C2|成功|3|0.354|0.1467|22.0|0.466|23.0|0.004|0.996|
|C2|失敗|97|0.352|0.1398|26.6|0.258|25.7|0.823|0.177|
|C3|成功|28|0.210|0.0515|31.1|0.832|28.5|0.900|0.100|
|C3|失敗|72|0.223|0.0592|74.3|0.721|50.7|0.755|0.245|

## オフライン未学習 MSE との順位

- C1a/C2 draft (K-x1 × E-H, seed1): 0.0936
- C1a/C2 draft 3 seed mean: 0.0945
- C1b draft (K-x1 × HS-MLP CT, seed1): 0.0836
- C1b draft 3 seed mean: 0.0863
- C3 draft (D5, KEYGRID 原型): 0.1011
- C3 D5+Expert(production-04): 0.0806
- pi05: 0.115
- note: KQ-final report.md / EX2 report.md; production-05 Expert offline values for the new drafts were not computed

## 設定

- LIBERO-90 Eval [0, 10, 13, 31, 35, 43, 57, 61, 84, 88]、各10試行、seed_base 7、公開初期状態 0..9、replan 5、wait 10、最大 520 step（lib-closed-01 / EX1 / EX2 と同じ）。π0.5 は baseline-01 の 32/100 を再掲（同じ runner・同じ seed・同じ初期状態）。
- Expert：EX2 修正版 production-05 の A、checkpoint step 2500（/work/.skillnet-state/outputs/memory-pilot/unseen-libero/expert-ex2-01/production-05/train-A/checkpoints/002500）。学習時の Dev 曲線は module_jit で重みが固定されたまま計算されており（全 step 同値）、EX2 の修正済み再採点（repair_dev_checkpoint_selection、Dev 217 行・D1 下書き、Eval 不使用）で選び直した。再学習はしていない。
- t0 規則：t0 = clip(sqrt(predicted arm6 error), .05, .6); no π0.5 fallback。グリッパーは π0.5 の通常出力。
- 下書き：C1a/C2 は best/Afix_x1_fixed_e_h_cteff.pt（seed 1）、C1b は best/H_x1_mk1b_cteff_tau0.01_s1.pt、キーは final/data/x1_keys.npy。C3 は KEYGRID production-04 の履歴＋状態 MLP（EX2 の D5）。候補は LIBERO-40 の 4,516 単位（EX2 と同じ並び）。
- パリティ：オフライン（parity_offline.json）でサーバーと同じ経路のスコアが KQ-final export の top-1 と未学習 619 行・Dev 217 行で 100% 一致。閉ループ smoke（parity_inloop.json）で、ループ内に記録した入力から再計算した top-1 が全問い合わせで一致。
- コード：unseen_libero/run_cl1_closed_loop.py（実行・集計）、serve_expert_ex2.py（KQ_* モードと --no-expert を追加）、report_cl1.py。

## 気になる点
- production-05 の train.json にある「選択 step 500」は無効。学習中の Dev 採点が module_jit で最初の重みに固定され、全 step で同じ値になっていたため。EX2 の修正済み再採点で step 2500 を選び直した（Dev .0858）。B は step 4500（.1013）で A より悪い。
- 信頼度器の予測誤差 ê（中央値 .038〜.042）は、未学習でのオフラインの真の誤差（.08〜.10）の半分以下。t0 は小さめに出ている。また、新しい検索器のスコアの尺度は、信頼度器を学習したときの D0〜D5 と違う（マージン・エントロピー特徴）。
- C3（D5＋production-05 Expert）は 28。EX2 修正前（production-04、step 5000）の D5 36 から下がった。Expert の checkpoint・学習行の修正・初期化が混ざっているので、原因は切り分けていない。
- 切り替え率：成功試行でも約 85% の問い合わせで単位が替わる。単位を区切りまで実行していない（5 step ごとに検索し直す）。
- 4 サーバー同時実行（fraction .16）は読み込み時に OOM。.22 でも 4 本だと推論中に CUBLAS の確保に失敗し、全試行が error になった。3 本＋C2 単独で実行し直した。失敗分は failed-oom-run-01/ と smoke/failed-oom-01/ に残してあり、集計には入れていない。
