## T5：上位候補の運動不一致による信頼度と Expert グリッパー

Dev v2 のみで校正（667窓）。ê のフォールバック割合 p_b=0.204 に合わせた D の上側しきい値は D_b=0.0374474（超過時に π0.5）。D と正解の top-1 腕誤差の Dev Spearman=0.419、ê の Spearman=0.482。T1 と同じ K-x1/(iii)/K1:500 を使い、学習はしていない。G-exp は hold-last を下書きに入れ、G-dec はデコーダ出力を下書きに入れ、両方とも bridge 後の Expert グリッパー出力をそのまま返す。

| Suite | 条件 | 成功/完了 | Wilson 95% | π0.5 p | A-const p | A-add p | A-const-pg p | fallback | D P05/50/95 | t0 P05/50/95 | 切替/試行 成功 | 切替/試行 失敗 |
|---|---|---:|---|---:|---:|---:|---:|---:|---|---|---:|---:|
| libero90 | AaddGexp | 30/100 | [0.219, 0.396] | 0.688 | 0.125 | 0.0625 | — | 18.8% | 0.006/0.016/0.066 | 0.218/0.284/0.350 | 1.000/3.000/9.650 | 0.000/1.000/13.100 |
| libero90 | AtopkGexp | 14/100 | [0.085, 0.221] | 4.01e-05 | 5.72e-06 | 5.72e-06 | — | 6.9% | 0.005/0.012/0.044 | 0.190/0.231/0.350 | 0.650/2.000/7.000 | 0.000/0.000/4.000 |
| libero90 | AtopkGdec | 34/100 | [0.255, 0.437] | 0.688 | 1 | 1 | — | 12.4% | 0.006/0.018/0.058 | 0.199/0.264/0.350 | 1.000/2.000/3.350 | 0.000/2.000/6.000 |
| libero10 | AaddGexp | 1/100 | [0.002, 0.054] | 1.01e-28 | 1.36e-20 | 1.06e-22 | 6.46e-27 | 1.3% | 0.004/0.011/0.034 | 0.188/0.228/0.309 | 2.000/2.000/2.000 | 0.000/0.000/4.100 |
| libero10 | AtopkGexp | 4/100 | [0.016, 0.098] | 8.08e-28 | 1.08e-19 | 1.57e-20 | 5.17e-26 | 3.9% | 0.005/0.011/0.032 | 0.186/0.229/0.330 | 1.000/1.000/2.700 | 0.000/0.000/5.000 |
| libero10 | AtopkGdec | 81/100 | [0.722, 0.875] | 0.00131 | 0.00443 | 0.18 | 0.0768 | 12.7% | 0.006/0.016/0.054 | 0.199/0.257/0.350 | 1.000/3.000/4.000 | 0.000/1.000/4.200 |

T5 条件は試行番号・初期状態・seed_base=7、wait=10、replan=5、最大520制御stepを T1 と共有する。McNemar は同一 task/trial の対応試行で計算。グリッパー切替数は実行した各チャンクの先頭5指令だけを連結して数えた。

**タスク別成功数**

| Suite | 条件 | Task | 成功/完了 |
|---|---|---|---:|
| libero90 | AaddGexp | KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet (`0`) | 1/10 |
| libero90 | AaddGexp | KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet (`10`) | 10/10 |
| libero90 | AaddGexp | KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate (`13`) | 0/10 |
| libero90 | AaddGexp | KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet (`31`) | 0/10 |
| libero90 | AaddGexp | KITCHEN_SCENE7_open_the_microwave (`35`) | 0/10 |
| libero90 | AaddGexp | KITCHEN_SCENE9_put_the_white_bowl_on_top_of_the_cabinet (`43`) | 1/10 |
| libero90 | AaddGexp | LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray (`57`) | 8/10 |
| libero90 | AaddGexp | LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray (`61`) | 10/10 |
| libero90 | AaddGexp | STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy (`84`) | 0/10 |
| libero90 | AaddGexp | STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_on_the_cabinet_shelf (`88`) | 0/10 |
| libero90 | AtopkGexp | KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet (`0`) | 1/10 |
| libero90 | AtopkGexp | KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet (`10`) | 2/10 |
| libero90 | AtopkGexp | KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate (`13`) | 0/10 |
| libero90 | AtopkGexp | KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet (`31`) | 0/10 |
| libero90 | AtopkGexp | KITCHEN_SCENE7_open_the_microwave (`35`) | 0/10 |
| libero90 | AtopkGexp | KITCHEN_SCENE9_put_the_white_bowl_on_top_of_the_cabinet (`43`) | 0/10 |
| libero90 | AtopkGexp | LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray (`57`) | 2/10 |
| libero90 | AtopkGexp | LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray (`61`) | 9/10 |
| libero90 | AtopkGexp | STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy (`84`) | 0/10 |
| libero90 | AtopkGexp | STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_on_the_cabinet_shelf (`88`) | 0/10 |
| libero90 | AtopkGdec | KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet (`0`) | 1/10 |
| libero90 | AtopkGdec | KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet (`10`) | 10/10 |
| libero90 | AtopkGdec | KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate (`13`) | 0/10 |
| libero90 | AtopkGdec | KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet (`31`) | 2/10 |
| libero90 | AtopkGdec | KITCHEN_SCENE7_open_the_microwave (`35`) | 0/10 |
| libero90 | AtopkGdec | KITCHEN_SCENE9_put_the_white_bowl_on_top_of_the_cabinet (`43`) | 1/10 |
| libero90 | AtopkGdec | LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray (`57`) | 10/10 |
| libero90 | AtopkGdec | LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray (`61`) | 10/10 |
| libero90 | AtopkGdec | STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy (`84`) | 0/10 |
| libero90 | AtopkGdec | STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_on_the_cabinet_shelf (`88`) | 0/10 |
| libero10 | AaddGexp | put both the alphabet soup and the tomato sauce in the basket (`0`) | 0/10 |
| libero10 | AaddGexp | put both the cream cheese box and the butter in the basket (`1`) | 0/10 |
| libero10 | AaddGexp | turn on the stove and put the moka pot on it (`2`) | 0/10 |
| libero10 | AaddGexp | put the black bowl in the bottom drawer of the cabinet and close it (`3`) | 0/10 |
| libero10 | AaddGexp | put the white mug on the left plate and put the yellow and white mug on the right plate (`4`) | 0/10 |
| libero10 | AaddGexp | pick up the book and place it in the back compartment of the caddy (`5`) | 1/10 |
| libero10 | AaddGexp | put the white mug on the plate and put the chocolate pudding to the right of the plate (`6`) | 0/10 |
| libero10 | AaddGexp | put both the alphabet soup and the cream cheese box in the basket (`7`) | 0/10 |
| libero10 | AaddGexp | put both moka pots on the stove (`8`) | 0/10 |
| libero10 | AaddGexp | put the yellow and white mug in the microwave and close it (`9`) | 0/10 |
| libero10 | AtopkGexp | put both the alphabet soup and the tomato sauce in the basket (`0`) | 0/10 |
| libero10 | AtopkGexp | put both the cream cheese box and the butter in the basket (`1`) | 0/10 |
| libero10 | AtopkGexp | turn on the stove and put the moka pot on it (`2`) | 1/10 |
| libero10 | AtopkGexp | put the black bowl in the bottom drawer of the cabinet and close it (`3`) | 0/10 |
| libero10 | AtopkGexp | put the white mug on the left plate and put the yellow and white mug on the right plate (`4`) | 0/10 |
| libero10 | AtopkGexp | pick up the book and place it in the back compartment of the caddy (`5`) | 3/10 |
| libero10 | AtopkGexp | put the white mug on the plate and put the chocolate pudding to the right of the plate (`6`) | 0/10 |
| libero10 | AtopkGexp | put both the alphabet soup and the cream cheese box in the basket (`7`) | 0/10 |
| libero10 | AtopkGexp | put both moka pots on the stove (`8`) | 0/10 |
| libero10 | AtopkGexp | put the yellow and white mug in the microwave and close it (`9`) | 0/10 |
| libero10 | AtopkGdec | put both the alphabet soup and the tomato sauce in the basket (`0`) | 10/10 |
| libero10 | AtopkGdec | put both the cream cheese box and the butter in the basket (`1`) | 10/10 |
| libero10 | AtopkGdec | turn on the stove and put the moka pot on it (`2`) | 8/10 |
| libero10 | AtopkGdec | put the black bowl in the bottom drawer of the cabinet and close it (`3`) | 10/10 |
| libero10 | AtopkGdec | put the white mug on the left plate and put the yellow and white mug on the right plate (`4`) | 10/10 |
| libero10 | AtopkGdec | pick up the book and place it in the back compartment of the caddy (`5`) | 10/10 |
| libero10 | AtopkGdec | put the white mug on the plate and put the chocolate pudding to the right of the plate (`6`) | 9/10 |
| libero10 | AtopkGdec | put both the alphabet soup and the cream cheese box in the basket (`7`) | 9/10 |
| libero10 | AtopkGdec | put both moka pots on the stove (`8`) | 2/10 |
| libero10 | AtopkGdec | put the yellow and white mug in the microwave and close it (`9`) | 3/10 |

Dev 校正: D P05/P25/P50/P75/P95 = 0.009077/0.01531/0.02305/0.03389/0.08436; D_b=0.0374474; p_b=0.204。D は上位10候補内の softmax(検索logit)重み付き top-1 との arm6 MSE。

スモーク: AaddGexp 2/2, AtopkGexp 0/2, AtopkGdec 2/2. OOM ログ検出数: {"libero90": 0, "libero10": 0}.

JST の進捗・実行ログは `t5/progress.log` と `t5/t5-run.log`。
