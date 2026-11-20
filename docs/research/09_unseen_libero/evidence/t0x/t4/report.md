## T4：Expert 修正量による閉ループ t0（A-exp / A-max）

Dev v2 だけで校正し、Eval の正解は使っていない。probe は t0=.30、固定ノイズ seed 1本。 p_b=Pr(ê>b)、m_b=Q_(1-p_b)(m) とし、m'=b·(m/m_b)^2、t0_m=clip(√(.15²+((.35²−.15²)/b)m'), .15, .35)。A-exp は t0_m、A-max は max(t0_ê,t0_m)。ê>b の場合は両条件とも π0.5 fallback。

| Suite | Condition | 成功/完了 | Wilson 95% | π0.5のみ / 条件のみ (p) | A-constのみ / 条件のみ (p) | A-addのみ / 条件のみ (p) | fallback | m P05/50/95 | t0 P05/50/95 | ê P05/50/95 | 問合せ秒 P05/50/95 | T1 A-add 比の中央値差 (秒) |
|---|---|---:|---|---|---|---|---:|---|---|---|---|---:|

Dev 校正：667 行、b=0.06849, p_b=0.204, m_b=1.84366, m 分位点=0.796。

| libero90 | Aexp | 34/100 | [0.255, 0.437] | 1 / 3 (p=0.625) | 3 / 2 (p=1) | 2 / 1 (p=1) | 27.4% | 0.712/1.226/1.889 | 0.193/0.258/0.350 | 0.021/0.050/0.128 | 0.226/0.253/0.304 | +0.0871 |
| libero90 | Amax | 37/100 | [0.282, 0.468] | 1 / 6 (p=0.125) | 3 / 5 (p=0.727) | 1 / 3 (p=0.625) | 27.1% | 0.700/1.217/1.895 | 0.237/0.317/0.350 | 0.021/0.051/0.121 | 0.230/0.254/0.307 | +0.0880 |
| libero10 | Aexp | 68/100 | [0.583, 0.763] | 27 / 0 (p=1.49e-08) | 10 / 10 (p=1) | 13 / 6 (p=0.167) | 3.7% | 0.574/1.069/1.936 | 0.179/0.237/0.350 | 0.010/0.022/0.062 | 0.155/0.165/0.198 | -0.0064 |
| libero10 | Amax | 72/100 | [0.625, 0.799] | 24 / 1 (p=1.55e-06) | 10 / 14 (p=0.541) | 9 / 6 (p=0.607) | 4.3% | 0.587/1.090/1.926 | 0.196/0.253/0.350 | 0.010/0.023/0.065 | 0.156/0.166/0.199 | -0.0053 |

**T4 タスク別成功数**

| Suite | Condition | Task | 成功/完了 |
|---|---|---|---:|
| libero90 | Aexp | KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet (`0`) | 2/10 |
| libero90 | Aexp | KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet (`10`) | 10/10 |
| libero90 | Aexp | KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate (`13`) | 0/10 |
| libero90 | Aexp | KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet (`31`) | 0/10 |
| libero90 | Aexp | KITCHEN_SCENE7_open_the_microwave (`35`) | 1/10 |
| libero90 | Aexp | KITCHEN_SCENE9_put_the_white_bowl_on_top_of_the_cabinet (`43`) | 1/10 |
| libero90 | Aexp | LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray (`57`) | 10/10 |
| libero90 | Aexp | LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray (`61`) | 10/10 |
| libero90 | Aexp | STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy (`84`) | 0/10 |
| libero90 | Aexp | STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_on_the_cabinet_shelf (`88`) | 0/10 |
| libero90 | Amax | KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet (`0`) | 2/10 |
| libero90 | Amax | KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet (`10`) | 10/10 |
| libero90 | Amax | KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate (`13`) | 1/10 |
| libero90 | Amax | KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet (`31`) | 2/10 |
| libero90 | Amax | KITCHEN_SCENE7_open_the_microwave (`35`) | 1/10 |
| libero90 | Amax | KITCHEN_SCENE9_put_the_white_bowl_on_top_of_the_cabinet (`43`) | 1/10 |
| libero90 | Amax | LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray (`57`) | 10/10 |
| libero90 | Amax | LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray (`61`) | 10/10 |
| libero90 | Amax | STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy (`84`) | 0/10 |
| libero90 | Amax | STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_on_the_cabinet_shelf (`88`) | 0/10 |
| libero10 | Aexp | put both the alphabet soup and the tomato sauce in the basket (`0`) | 9/10 |
| libero10 | Aexp | put both the cream cheese box and the butter in the basket (`1`) | 10/10 |
| libero10 | Aexp | turn on the stove and put the moka pot on it (`2`) | 4/10 |
| libero10 | Aexp | put the black bowl in the bottom drawer of the cabinet and close it (`3`) | 10/10 |
| libero10 | Aexp | put the white mug on the left plate and put the yellow and white mug on the right plate (`4`) | 9/10 |
| libero10 | Aexp | pick up the book and place it in the back compartment of the caddy (`5`) | 10/10 |
| libero10 | Aexp | put the white mug on the plate and put the chocolate pudding to the right of the plate (`6`) | 5/10 |
| libero10 | Aexp | put both the alphabet soup and the cream cheese box in the basket (`7`) | 10/10 |
| libero10 | Aexp | put both moka pots on the stove (`8`) | 0/10 |
| libero10 | Aexp | put the yellow and white mug in the microwave and close it (`9`) | 1/10 |
| libero10 | Amax | put both the alphabet soup and the tomato sauce in the basket (`0`) | 9/10 |
| libero10 | Amax | put both the cream cheese box and the butter in the basket (`1`) | 10/10 |
| libero10 | Amax | turn on the stove and put the moka pot on it (`2`) | 5/10 |
| libero10 | Amax | put the black bowl in the bottom drawer of the cabinet and close it (`3`) | 9/10 |
| libero10 | Amax | put the white mug on the left plate and put the yellow and white mug on the right plate (`4`) | 10/10 |
| libero10 | Amax | pick up the book and place it in the back compartment of the caddy (`5`) | 10/10 |
| libero10 | Amax | put the white mug on the plate and put the chocolate pudding to the right of the plate (`6`) | 6/10 |
| libero10 | Amax | put both the alphabet soup and the cream cheese box in the basket (`7`) | 9/10 |
| libero10 | Amax | put both moka pots on the stove (`8`) | 1/10 |
| libero10 | Amax | put the yellow and white mug in the microwave and close it (`9`) | 3/10 |

OOM 文字列検出数（再試行ログを含む）: `{'libero90': 0, 'libero10': 0}`。各 condition の試行は最大3回まで再開・再試行し、完了記録のみ集計した。


**タスク別成功数**

| Suite | Condition | Task | 成功/試行 |
|---|---|---|---:|
| libero90 | Aadd | KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet (`0`) | 2/10 |
| libero90 | Aadd | KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet (`10`) | 10/10 |
| libero90 | Aadd | KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate (`13`) | 0/10 |
| libero90 | Aadd | KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet (`31`) | 1/10 |
| libero90 | Aadd | KITCHEN_SCENE7_open_the_microwave (`35`) | 1/10 |
| libero90 | Aadd | KITCHEN_SCENE9_put_the_white_bowl_on_top_of_the_cabinet (`43`) | 1/10 |
| libero90 | Aadd | LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray (`57`) | 10/10 |
| libero90 | Aadd | LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray (`61`) | 10/10 |
| libero90 | Aadd | STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy (`84`) | 0/10 |
| libero90 | Aadd | STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_on_the_cabinet_shelf (`88`) | 0/10 |
| libero90 | Badd | KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet (`0`) | 0/10 |
| libero90 | Badd | KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet (`10`) | 10/10 |
| libero90 | Badd | KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate (`13`) | 0/10 |
| libero90 | Badd | KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet (`31`) | 2/10 |
| libero90 | Badd | KITCHEN_SCENE7_open_the_microwave (`35`) | 2/10 |
| libero90 | Badd | KITCHEN_SCENE9_put_the_white_bowl_on_top_of_the_cabinet (`43`) | 1/10 |
| libero90 | Badd | LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray (`57`) | 10/10 |
| libero90 | Badd | LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray (`61`) | 10/10 |
| libero90 | Badd | STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy (`84`) | 0/10 |
| libero90 | Badd | STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_on_the_cabinet_shelf (`88`) | 0/10 |
| libero10 | Aadd | put both the alphabet soup and the tomato sauce in the basket (`0`) | 10/10 |
| libero10 | Aadd | put both the cream cheese box and the butter in the basket (`1`) | 10/10 |
| libero10 | Aadd | turn on the stove and put the moka pot on it (`2`) | 6/10 |
| libero10 | Aadd | put the black bowl in the bottom drawer of the cabinet and close it (`3`) | 9/10 |
| libero10 | Aadd | put the white mug on the left plate and put the yellow and white mug on the right plate (`4`) | 9/10 |
| libero10 | Aadd | pick up the book and place it in the back compartment of the caddy (`5`) | 10/10 |
| libero10 | Aadd | put the white mug on the plate and put the chocolate pudding to the right of the plate (`6`) | 8/10 |
| libero10 | Aadd | put both the alphabet soup and the cream cheese box in the basket (`7`) | 9/10 |
| libero10 | Aadd | put both moka pots on the stove (`8`) | 2/10 |
| libero10 | Aadd | put the yellow and white mug in the microwave and close it (`9`) | 2/10 |
| libero10 | Badd | put both the alphabet soup and the tomato sauce in the basket (`0`) | 9/10 |
| libero10 | Badd | put both the cream cheese box and the butter in the basket (`1`) | 3/10 |
| libero10 | Badd | turn on the stove and put the moka pot on it (`2`) | 6/10 |
| libero10 | Badd | put the black bowl in the bottom drawer of the cabinet and close it (`3`) | 7/10 |
| libero10 | Badd | put the white mug on the left plate and put the yellow and white mug on the right plate (`4`) | 3/10 |
| libero10 | Badd | pick up the book and place it in the back compartment of the caddy (`5`) | 9/10 |
| libero10 | Badd | put the white mug on the plate and put the chocolate pudding to the right of the plate (`6`) | 2/10 |
| libero10 | Badd | put both the alphabet soup and the cream cheese box in the basket (`7`) | 9/10 |
| libero10 | Badd | put both moka pots on the stove (`8`) | 2/10 |
| libero10 | Badd | put the yellow and white mug in the microwave and close it (`9`) | 7/10 |

条件の対比較は KQ2-CL と同じ初期状態・seed（trial i: env seed 7+i）、wait=10、replan=5、最大520制御step。A-add は query (iii)、B-add は query (i)。各100試行。T1の状態は `results.json` に保存する。

## T2：Expert 修正量による信頼度

m は probe t0=.15/.30 で Expert が下書きを直した腕6軸の差分ノルム `‖x̂0−draft‖₂`（[10,6] 全体）で、独立ノイズ2 seed の平均。真の下書き誤差との Spearman を比較した。dev2 は667窓、unseen20 は619窓、learned は621窓。

| split | query | ê Spearman | m@.15 Spearman (gain) | m@.30 Spearman (gain) |
|---|---|---:|---:|---:|
| dev2 | iii | 0.482 | 0.143 (-0.338) | 0.399 (-0.082) |
| dev2 | i | 0.579 | 0.092 (-0.487) | 0.331 (-0.248) |
| unseen20 | iii | 0.467 | 0.082 (-0.384) | 0.306 (-0.161) |
| unseen20 | i | 0.602 | 0.124 (-0.477) | 0.313 (-0.289) |
| learned | iii | 0.718 | 0.836 (+0.118) | 0.976 (+0.258) |
| learned | i | 0.608 | 0.855 (+0.247) | 0.969 (+0.361) |

Dev gate A-exp: 不通過（基準 +0.10）。Probe: 該当なし。

## T3：L10 診断条件と既存 A-const 失敗

| Condition | 成功 | 完了 | Wilson 95% | π0.5のみ / 条件のみ (p; n paired) | A-constのみ / 条件のみ (p; n paired) | fallback | t0 P05/50/95 |
|---|---:|---:|---|---|---|---:|---|
| AconstPG | 89 | 100 | [0.814, 0.937] | 8 / 2 (p=0.109; n=100) | 3 / 24 (p=4.92e-05; n=100) | 5.6% | 0.150/0.150/0.150 |
| AconstT25 | 73 | 100 | [0.636, 0.807] | 23 / 1 (p=2.98e-06; n=100) | 6 / 11 (p=0.332; n=100) | 4.8% | 0.250/0.250/0.250 |

**T3 タスク別成功数**

| Condition | Task | 成功/試行 |
|---|---|---:|
| AconstPG | put both the alphabet soup and the tomato sauce in the basket (`0`) | 8/10 |
| AconstPG | put both the cream cheese box and the butter in the basket (`1`) | 10/10 |
| AconstPG | turn on the stove and put the moka pot on it (`2`) | 9/10 |
| AconstPG | put the black bowl in the bottom drawer of the cabinet and close it (`3`) | 10/10 |
| AconstPG | put the white mug on the left plate and put the yellow and white mug on the right plate (`4`) | 10/10 |
| AconstPG | pick up the book and place it in the back compartment of the caddy (`5`) | 10/10 |
| AconstPG | put the white mug on the plate and put the chocolate pudding to the right of the plate (`6`) | 10/10 |
| AconstPG | put both the alphabet soup and the cream cheese box in the basket (`7`) | 10/10 |
| AconstPG | put both moka pots on the stove (`8`) | 4/10 |
| AconstPG | put the yellow and white mug in the microwave and close it (`9`) | 8/10 |
| AconstT25 | put both the alphabet soup and the tomato sauce in the basket (`0`) | 10/10 |
| AconstT25 | put both the cream cheese box and the butter in the basket (`1`) | 10/10 |
| AconstT25 | turn on the stove and put the moka pot on it (`2`) | 7/10 |
| AconstT25 | put the black bowl in the bottom drawer of the cabinet and close it (`3`) | 9/10 |
| AconstT25 | put the white mug on the left plate and put the yellow and white mug on the right plate (`4`) | 9/10 |
| AconstT25 | pick up the book and place it in the back compartment of the caddy (`5`) | 10/10 |
| AconstT25 | put the white mug on the plate and put the chocolate pudding to the right of the plate (`6`) | 9/10 |
| AconstT25 | put both the alphabet soup and the cream cheese box in the basket (`7`) | 8/10 |
| AconstT25 | put both moka pots on the stove (`8`) | 0/10 |
| AconstT25 | put the yellow and white mug in the microwave and close it (`9`) | 1/10 |

A-const-pg はグリッパーだけを π0.5 出力に置換した診断条件であり、提案方式ではない。A-const-t25 は固定 t0=.25。
A-const の L10 失敗: 32試行、終了理由 `{'control_budget': 32}`。env step中央値 520.0、query数中央値 104.0、fallback率 2.1%、実行グリッパー切替中央値 1.0。
選択単位が指示文と一致した割合: 13.0%（比較可能な選択 3328、対応付け不明 0）。閉ループログからは意味的な作業段階を記録できないため、停止段階の手動分類はできていない。

## smoke・実行記録・注意点

A-add smoke (LIBERO-90 task 10, trial 0-1): 2/2。A-const-pg smoke (LIBERO-10 task 0, trial 0-1): 2/2。query trace は `results.json` の smoke 節。

KQ2-CL の既存値：L90 A=28/100、A-const=.15=35/100、B=7/100、B-const=32/100、π0.5=32/100。L10 A=25/100、A-const=68/100、B=3/100、π0.5=95/100。

GPU OOM: T1 後の空き確認で、T2 は JAX 割合 .15・batch 8/2/1 の順に OOM。GPU が空いていることを再確認し、.40・batch 1 で完了した。稼働中ジョブを止めず、CPU/GPU の thread・fraction を記録した。T1 の技術エラーは `progress.log` と試行 JSONL に記録。

## 再現コマンド

```bash
cd /work && source env/activate.sh
export SKILLNET_STATE_DIR=/work/.skillnet-state XLA_PYTHON_CLIENT_PREALLOCATE=false
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 taskset -c 16-23 python skill_moe/skillnet/examples/memory_pilot/unseen_libero/t0x_run.py t1 --workers 6 --fraction .20 --cpus 24-31,32-39
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 taskset -c 16-23 python skill_moe/skillnet/examples/memory_pilot/unseen_libero/t0x_run.py t3 --workers 6 --fraction .20 --cpus 24-31,32-39
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 XLA_PYTHON_CLIENT_MEM_FRACTION=.40 nice -n 10 taskset -c 8-15 python skill_moe/skillnet/examples/memory_pilot/unseen_libero/t0x_run.py t2 --batch 1 --fraction .40
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 XLA_PYTHON_CLIENT_MEM_FRACTION=.20 nice -n 10 taskset -c 8-23 python skill_moe/skillnet/examples/memory_pilot/unseen_libero/t0x_t4.py all --workers 6 --fraction .20 --cpus 24-31,32-39 --batch 1
```
