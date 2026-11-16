# EX1：P0・P5・潜在トークン検索を下書きにした Expert 評価

## オフライン主表

| データ | 範囲 | 検索 | 出力 | arm6 MSE | active7 MSE（π0.5 gripper） |
|---|---|---|---|---:|---:|
| 学習済み621 | all | D0_P0_top1 | draft | 0.036119 | 0.040348 |
| 学習済み621 | all | D0_P0_top1 | expert_A | 0.001855 | 0.006084 |
| 学習済み621 | all | D0_P0_top1 | expert_B | 0.002588 | 0.006817 |
| 学習済み621 | all | D1_P5_top1 | draft | 0.040721 | 0.044950 |
| 学習済み621 | all | D1_P5_top1 | expert_A | 0.001911 | 0.006140 |
| 学習済み621 | all | D1_P5_top1 | expert_B | 0.002578 | 0.006807 |
| 学習済み621 | all | D3_E_VLM_H_top1 | draft | 0.034979 | 0.039208 |
| 学習済み621 | all | D3_E_VLM_H_top1 | expert_A | 0.001903 | 0.006132 |
| 学習済み621 | all | D3_E_VLM_H_top1 | expert_B | 0.002672 | 0.006901 |
| 学習済み621 | all | — | pi05 | 0.002217 | 0.006446 |
| 学習済み621 | all | — | hold_last | 0.056397 | 0.103433 |
| 学習済み621 | all | — | oracle_draft_expert_A | 0.001722 | 0.005951 |
| 学習済み621 | all | — | oracle_draft_expert_B | 0.002304 | 0.006533 |
| 学習済み621 | hard | D0_P0_top1 | draft | 0.068245 | 0.073731 |
| 学習済み621 | hard | D0_P0_top1 | expert_A | 0.003146 | 0.008631 |
| 学習済み621 | hard | D0_P0_top1 | expert_B | 0.004218 | 0.009703 |
| 学習済み621 | hard | D1_P5_top1 | draft | 0.075421 | 0.080907 |
| 学習済み621 | hard | D1_P5_top1 | expert_A | 0.003409 | 0.008895 |
| 学習済み621 | hard | D1_P5_top1 | expert_B | 0.004656 | 0.010142 |
| 学習済み621 | hard | D3_E_VLM_H_top1 | draft | 0.066372 | 0.071858 |
| 学習済み621 | hard | D3_E_VLM_H_top1 | expert_A | 0.003316 | 0.008802 |
| 学習済み621 | hard | D3_E_VLM_H_top1 | expert_B | 0.004917 | 0.010403 |
| 学習済み621 | hard | — | pi05 | 0.003734 | 0.009219 |
| 学習済み621 | hard | — | hold_last | 0.097244 | 0.142518 |
| 学習済み621 | hard | — | oracle_draft_expert_A | 0.002896 | 0.008382 |
| 学習済み621 | hard | — | oracle_draft_expert_B | 0.003819 | 0.009305 |
| 未学習619 | all | D0_P0_top1 | draft | 0.132226 | 0.158586 |
| 未学習619 | all | D0_P0_top1 | expert_A | 0.109740 | 0.136101 |
| 未学習619 | all | D0_P0_top1 | expert_B | 0.117022 | 0.143383 |
| 未学習619 | all | D1_P5_top1 | draft | 0.105446 | 0.131807 |
| 未学習619 | all | D1_P5_top1 | expert_A | 0.106753 | 0.133114 |
| 未学習619 | all | D1_P5_top1 | expert_B | 0.091607 | 0.117967 |
| 未学習619 | all | D3_E_VLM_H_top1 | draft | 0.124963 | 0.151323 |
| 未学習619 | all | D3_E_VLM_H_top1 | expert_A | 0.109117 | 0.135478 |
| 未学習619 | all | D3_E_VLM_H_top1 | expert_B | 0.110812 | 0.137173 |
| 未学習619 | all | — | pi05 | 0.114979 | 0.141339 |
| 未学習619 | all | — | hold_last | 0.113995 | 0.188606 |
| 未学習619 | all | — | oracle_draft_expert_A | 0.100567 | 0.126927 |
| 未学習619 | all | — | oracle_draft_expert_B | 0.047298 | 0.073659 |
| 未学習619 | hard | D0_P0_top1 | draft | 0.275040 | 0.309103 |
| 未学習619 | hard | D0_P0_top1 | expert_A | 0.245235 | 0.279299 |
| 未学習619 | hard | D0_P0_top1 | expert_B | 0.258472 | 0.292535 |
| 未学習619 | hard | D1_P5_top1 | draft | 0.184163 | 0.218226 |
| 未学習619 | hard | D1_P5_top1 | expert_A | 0.237649 | 0.271713 |
| 未学習619 | hard | D1_P5_top1 | expert_B | 0.169248 | 0.203312 |
| 未学習619 | hard | D3_E_VLM_H_top1 | draft | 0.256061 | 0.290125 |
| 未学習619 | hard | D3_E_VLM_H_top1 | expert_A | 0.244032 | 0.278096 |
| 未学習619 | hard | D3_E_VLM_H_top1 | expert_B | 0.239663 | 0.273727 |
| 未学習619 | hard | — | pi05 | 0.252887 | 0.286950 |
| 未学習619 | hard | — | hold_last | 0.200980 | 0.267315 |
| 未学習619 | hard | — | oracle_draft_expert_A | 0.229030 | 0.263094 |
| 未学習619 | hard | — | oracle_draft_expert_B | 0.083319 | 0.117383 |

D0 は P0 top-1、D1 は固定 KEY2-F P5 top-1、D3 は固定 KEY2-F E-VLM+H top-1。D2/D4 はオフライン参考として `offline_results.json` に残した。全出力のグリッパーは通常 π0.5 に復元。

## 閉ループ主表

| suite | 条件 | 成功/完了 | Wilson 95% | baseline | McNemar exact p (base-only / EX1-only) | 技術エラー |
|---|---|---:|---:|---:|---:|---:|
| LIBERO-90 Eval (10×10) | 通常 π0.5 baseline | 32/100 | — | — | — | 0 |
| LIBERO-90 Eval (10×10) | EX1_D0_P0_top1_expert | 32/100 | 23.7%–41.7% | 32 | 1 (3/3) | 0 |
| LIBERO-90 Eval (10×10) | EX1_D1_P5_top1_expert | 33/100 | 24.6%–42.7% | 32 | 1 (2/3) | 0 |
| LIBERO-90 Eval (10×10) | EX1_D3_E_VLM_H_top1_expert | 33/100 | 24.6%–42.7% | 32 | 1 (2/3) | 0 |
| LIBERO-10 learned (10×5) | 通常 π0.5 baseline | 46/50 | — | — | — | 0 |
| LIBERO-10 learned (10×5) | EX1_D0_P0_top1_expert | 45/50 | 78.6%–95.7% | 46 | 1 (2/1) | 0 |
| LIBERO-10 learned (10×5) | EX1_D1_P5_top1_expert | 46/50 | 81.2%–96.8% | 46 | 1 (2/2) | 0 |
| LIBERO-10 learned (10×5) | EX1_D3_E_VLM_H_top1_expert | 42/50 | 71.5%–91.7% | 46 | 0.289062 (6/2) | 0 |

### タスク別

#### LIBERO-90

| 条件 | task id | task | 成功/完了 | baseline成功 | McNemar p (base-only / EX1-only) |
|---|---:|---|---:|---:|---:|
| EX1_D0_P0_top1_expert | 0 | KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet | 0/10 | 0 | 1 (0/0) |
| EX1_D0_P0_top1_expert | 10 | KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet | 9/10 | 10 | 1 (1/0) |
| EX1_D0_P0_top1_expert | 13 | KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate | 0/10 | 0 | 1 (0/0) |
| EX1_D0_P0_top1_expert | 31 | KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet | 2/10 | 0 | 0.5 (0/2) |
| EX1_D0_P0_top1_expert | 35 | KITCHEN_SCENE7_open_the_microwave | 1/10 | 1 | 1 (1/1) |
| EX1_D0_P0_top1_expert | 43 | KITCHEN_SCENE9_put_the_white_bowl_on_top_of_the_cabinet | 0/10 | 1 | 1 (1/0) |
| EX1_D0_P0_top1_expert | 57 | LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray | 10/10 | 10 | 1 (0/0) |
| EX1_D0_P0_top1_expert | 61 | LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray | 10/10 | 10 | 1 (0/0) |
| EX1_D0_P0_top1_expert | 84 | STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy | 0/10 | 0 | 1 (0/0) |
| EX1_D0_P0_top1_expert | 88 | STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_on_the_cabinet_shelf | 0/10 | 0 | 1 (0/0) |
| EX1_D1_P5_top1_expert | 0 | KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet | 1/10 | 0 | 1 (0/1) |
| EX1_D1_P5_top1_expert | 10 | KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet | 10/10 | 10 | 1 (0/0) |
| EX1_D1_P5_top1_expert | 13 | KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate | 1/10 | 0 | 1 (0/1) |
| EX1_D1_P5_top1_expert | 31 | KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet | 1/10 | 0 | 1 (0/1) |
| EX1_D1_P5_top1_expert | 35 | KITCHEN_SCENE7_open_the_microwave | 0/10 | 1 | 1 (1/0) |
| EX1_D1_P5_top1_expert | 43 | KITCHEN_SCENE9_put_the_white_bowl_on_top_of_the_cabinet | 0/10 | 1 | 1 (1/0) |
| EX1_D1_P5_top1_expert | 57 | LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray | 10/10 | 10 | 1 (0/0) |
| EX1_D1_P5_top1_expert | 61 | LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray | 10/10 | 10 | 1 (0/0) |
| EX1_D1_P5_top1_expert | 84 | STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy | 0/10 | 0 | 1 (0/0) |
| EX1_D1_P5_top1_expert | 88 | STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_on_the_cabinet_shelf | 0/10 | 0 | 1 (0/0) |
| EX1_D3_E_VLM_H_top1_expert | 0 | KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet | 1/10 | 0 | 1 (0/1) |
| EX1_D3_E_VLM_H_top1_expert | 10 | KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet | 10/10 | 10 | 1 (0/0) |
| EX1_D3_E_VLM_H_top1_expert | 13 | KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate | 0/10 | 0 | 1 (0/0) |
| EX1_D3_E_VLM_H_top1_expert | 31 | KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet | 2/10 | 0 | 0.5 (0/2) |
| EX1_D3_E_VLM_H_top1_expert | 35 | KITCHEN_SCENE7_open_the_microwave | 0/10 | 1 | 1 (1/0) |
| EX1_D3_E_VLM_H_top1_expert | 43 | KITCHEN_SCENE9_put_the_white_bowl_on_top_of_the_cabinet | 0/10 | 1 | 1 (1/0) |
| EX1_D3_E_VLM_H_top1_expert | 57 | LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray | 10/10 | 10 | 1 (0/0) |
| EX1_D3_E_VLM_H_top1_expert | 61 | LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray | 10/10 | 10 | 1 (0/0) |
| EX1_D3_E_VLM_H_top1_expert | 84 | STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy | 0/10 | 0 | 1 (0/0) |
| EX1_D3_E_VLM_H_top1_expert | 88 | STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_on_the_cabinet_shelf | 0/10 | 0 | 1 (0/0) |

#### LIBERO-10

| 条件 | task id | task | 成功/完了 | baseline成功 | McNemar p (base-only / EX1-only) |
|---|---:|---|---:|---:|---:|
| EX1_D0_P0_top1_expert | 0 | put both the alphabet soup and the tomato sauce in the basket | 5/5 | 5 | 1 (0/0) |
| EX1_D0_P0_top1_expert | 1 | put both the cream cheese box and the butter in the basket | 5/5 | 5 | 1 (0/0) |
| EX1_D0_P0_top1_expert | 2 | turn on the stove and put the moka pot on it | 5/5 | 5 | 1 (0/0) |
| EX1_D0_P0_top1_expert | 3 | put the black bowl in the bottom drawer of the cabinet and close it | 5/5 | 5 | 1 (0/0) |
| EX1_D0_P0_top1_expert | 4 | put the white mug on the left plate and put the yellow and white mug on the right plate | 4/5 | 5 | 1 (1/0) |
| EX1_D0_P0_top1_expert | 5 | pick up the book and place it in the back compartment of the caddy | 5/5 | 5 | 1 (0/0) |
| EX1_D0_P0_top1_expert | 6 | put the white mug on the plate and put the chocolate pudding to the right of the plate | 5/5 | 4 | 1 (0/1) |
| EX1_D0_P0_top1_expert | 7 | put both the alphabet soup and the cream cheese box in the basket | 5/5 | 5 | 1 (0/0) |
| EX1_D0_P0_top1_expert | 8 | put both moka pots on the stove | 1/5 | 2 | 1 (1/0) |
| EX1_D0_P0_top1_expert | 9 | put the yellow and white mug in the microwave and close it | 5/5 | 5 | 1 (0/0) |
| EX1_D1_P5_top1_expert | 0 | put both the alphabet soup and the tomato sauce in the basket | 5/5 | 5 | 1 (0/0) |
| EX1_D1_P5_top1_expert | 1 | put both the cream cheese box and the butter in the basket | 5/5 | 5 | 1 (0/0) |
| EX1_D1_P5_top1_expert | 2 | turn on the stove and put the moka pot on it | 5/5 | 5 | 1 (0/0) |
| EX1_D1_P5_top1_expert | 3 | put the black bowl in the bottom drawer of the cabinet and close it | 5/5 | 5 | 1 (0/0) |
| EX1_D1_P5_top1_expert | 4 | put the white mug on the left plate and put the yellow and white mug on the right plate | 5/5 | 5 | 1 (0/0) |
| EX1_D1_P5_top1_expert | 5 | pick up the book and place it in the back compartment of the caddy | 4/5 | 5 | 1 (1/0) |
| EX1_D1_P5_top1_expert | 6 | put the white mug on the plate and put the chocolate pudding to the right of the plate | 5/5 | 4 | 1 (0/1) |
| EX1_D1_P5_top1_expert | 7 | put both the alphabet soup and the cream cheese box in the basket | 5/5 | 5 | 1 (0/0) |
| EX1_D1_P5_top1_expert | 8 | put both moka pots on the stove | 2/5 | 2 | 1 (1/1) |
| EX1_D1_P5_top1_expert | 9 | put the yellow and white mug in the microwave and close it | 5/5 | 5 | 1 (0/0) |
| EX1_D3_E_VLM_H_top1_expert | 0 | put both the alphabet soup and the tomato sauce in the basket | 5/5 | 5 | 1 (0/0) |
| EX1_D3_E_VLM_H_top1_expert | 1 | put both the cream cheese box and the butter in the basket | 5/5 | 5 | 1 (0/0) |
| EX1_D3_E_VLM_H_top1_expert | 2 | turn on the stove and put the moka pot on it | 5/5 | 5 | 1 (0/0) |
| EX1_D3_E_VLM_H_top1_expert | 3 | put the black bowl in the bottom drawer of the cabinet and close it | 5/5 | 5 | 1 (0/0) |
| EX1_D3_E_VLM_H_top1_expert | 4 | put the white mug on the left plate and put the yellow and white mug on the right plate | 5/5 | 5 | 1 (0/0) |
| EX1_D3_E_VLM_H_top1_expert | 5 | pick up the book and place it in the back compartment of the caddy | 3/5 | 5 | 0.5 (2/0) |
| EX1_D3_E_VLM_H_top1_expert | 6 | put the white mug on the plate and put the chocolate pudding to the right of the plate | 5/5 | 4 | 1 (0/1) |
| EX1_D3_E_VLM_H_top1_expert | 7 | put both the alphabet soup and the cream cheese box in the basket | 5/5 | 5 | 1 (0/0) |
| EX1_D3_E_VLM_H_top1_expert | 8 | put both moka pots on the stove | 1/5 | 2 | 1 (2/1) |
| EX1_D3_E_VLM_H_top1_expert | 9 | put the yellow and white mug in the microwave and close it | 3/5 | 5 | 0.5 (2/0) |

## Dev 選択とオフライン差の区間

Dev は task 6,7 の113行。各モデルの live checkpoint step と t0 を arm6 MSE 最小で選択し、Eval は選択に使っていない。

| 条件 | Expert | Dev arm6 MSE | step | t0 |
|---|---|---:|---:|---:|
| D1_P5_top1 | A | 0.084287 | 2500 | 0.30 |
| D1_P5_top1 | B | 0.092067 | 4000 | 0.30 |
| D3_E_VLM_H_top1 | A | 0.088492 | 2500 | 0.30 |
| D3_E_VLM_H_top1 | B | 0.099031 | 4000 | 0.30 |

D0 は専用の D0 学習分岐がなかったため、学習前に定めた代替として D1 の checkpoint 選択を使う。閉ループは A を使用（D1: step 2500, t0=.30。D3 も A が Dev で最良）。D0/D1/D3 は同じ A checkpoint を使い、検索下書きだけが異なる。

選択モデル A の対応するオフライン arm6 MSE 差（episode bootstrap、5,000回）：

| データ | 条件 | Expert−draft mean [95% CI] | Expert−π0.5 mean [95% CI] |
|---|---|---:|---:|
| 学習済み621 | D0_P0_top1 | -0.034264 [-0.038111, -0.030372] | -0.000362 [-0.000552, -0.000164] |
| 学習済み621 | D1_P5_top1 | -0.038809 [-0.042919, -0.034642] | -0.000306 [-0.000514, -0.000092] |
| 学習済み621 | D3_E_VLM_H_top1 | -0.033076 [-0.037266, -0.028994] | -0.000314 [-0.000512, -0.000112] |
| 未学習619 | D0_P0_top1 | -0.022486 [-0.026977, -0.018305] | -0.005238 [-0.006855, -0.003759] |
| 未学習619 | D1_P5_top1 | 0.001307 [-0.006558, 0.010291] | -0.008225 [-0.009945, -0.006750] |
| 未学習619 | D3_E_VLM_H_top1 | -0.015846 [-0.019046, -0.012746] | -0.005861 [-0.007338, -0.004519] |

差の区間は行ごとの arm6 MSE を episode 単位で再標本化した95%区間。0を含む区間を「同等」とは扱わない。

## 選択と条件

学習済み621行・未学習619行、難行マスクは固定。D0 は専用の Expert Dev 選択値がないため、上記の通り D1 checkpoint を代用し、Eval を使って選択しない。
学習済み閉ループは baseline-01/closed-loop-memory-01 の同一10タスク・trial 0–4・同 seed、未学習は baseline-01/production の固定10タスク・trial 0–9・同 seed。両方 seed_base=7、environment seed=7+trial、replan=5、wait=10、上限520、`SKILLNET_SUBTASK_GENERATE=0`。
学習選択・各モードの step/t0/Dev MSE は `train-aggregate/train.json` と `results.json` に保存。

指標はオフライン行動誤差とこの二つの閉ループ試行に限る。失敗・未完了は成功として扱わず、差がないことを同等とは解釈しない。

Detailed machine-readable result: [scope-change-01/results.json](scope-change-01/results.json).
