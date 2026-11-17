| Split | Row | Grip MSE all / hard | Active7 P5 arm | Active7 E-VLM+H arm |
|---|---|---:|---:|---:|
| learned | History+state | 0.037827 / 0.040691 | 0.078548 | 0.072806 |
| learned | G-a | 0.041601 / 0.045264 | 0.082321 | 0.076580 |
| learned | G-e | 0.030740 / 0.033376 | 0.071461 | 0.065719 |
| learned | G-b | 0.088916 / 0.075904 | 0.129636 | 0.123895 |
| learned | G-f | 0.045926 / 0.045264 | 0.086647 | 0.080905 |
| learned | G-c | 0.070042 / 0.063554 | 0.110763 | 0.105021 |
| learned | History+state trained MLP | 0.037827 / 0.040691 | 0.078548 | 0.072806 |
| unseen | History+state | 0.051525 / 0.060993 | 0.156972 | 0.176488 |
| unseen | G-a | 0.053188 / 0.061847 | 0.158634 | 0.178150 |
| unseen | G-e | 0.042936 / 0.051181 | 0.148383 | 0.167899 |
| unseen | G-b | 0.100103 / 0.118161 | 0.205549 | 0.225066 |
| unseen | G-f | 0.055958 / 0.066113 | 0.161405 | 0.180921 |
| unseen | G-c | 0.072951 / 0.090856 | 0.178397 | 0.197913 |
| unseen | History+state trained MLP | 0.051525 / 0.060993 | 0.156972 | 0.176488 |
| unseen reference | π0.5 | 0.026360 / 0.037033 | — | — |
| unseen reference | classifier | 0.049217 / 0.058008 | — | — |
| unseen reference | G1 hold | 0.051532 / 0.061005 | — | — |
| unseen reference | G2 hold | 0.074611 / 0.066336 | — | — |
| unseen reference | saved value | 0.193532 / 0.299019 | — | — |
| unseen reference | π0.5 grip + P5 arm | 0.026360 / 0.037033 | 0.131807 | — |
| unseen reference | π0.5 grip + E-VLM+H arm | 0.026360 / 0.037033 | — | 0.151323 |

### production-03/04 から何を変えたか

- 全行の全ヘッドに直前10実行actionと有効フラグを追加。履歴＋状態の学習MLPはDevでG1規則を上回らなかったため規則をゲート選択した。学習MLP自体もEvalで符号MSE .051525となり、G1符号基準 .051525 と一致した。
- Dev task 6/7を選択に使わず、LIBERO-40 heldout 80 episode（1,317窓）とL90X heldout 15 task（1,984行）で選択。fit側320 episode・5,266窓、learned Eval episodeとの重複0。
- BCE正例重みを新Devで選択。production-03/04の旧潜在ヘッド数値は1 epoch 1更新等の欠陥があるため無効。
- L90X 11,030行をLA2経由でGPU抽出し、fit 9,046／Dev 1,984に分割。V-JEPAはexact cacheのfit 647／Dev 154行だけを利用。

unseen references: π0.5 0.026360 (GRIP1 .026362, Δ -1.13e-06); classifier 0.049217; G1 continuous 0.051532 / sign 0.051525; G2 0.074611; stored 0.193532.

G-c retrieval arm6 top-1 MSE (joint / frozen E-VLM+H / Δ):
- learned: 0.034434 / 0.034979 / -0.000545
- unseen: 0.126899 / 0.124963 / +0.001936

Paired episode bootstrap, 5,000 reps; mean difference and 95% CI:
- learned:
  - G-a: +0.037372 [+0.031519, +0.043117]
  - G-b: +0.084687 [+0.074267, +0.094529]
  - G-c: +0.065813 [+0.056221, +0.075632]
  - G-e: +0.026511 [+0.020609, +0.032490]
  - G-f: +0.041697 [+0.035295, +0.048386]
  - History+state: +0.033598 [+0.027674, +0.039357]
  - G-e_minus_G-a: -0.010861 [-0.016558, -0.005403]
  - G-c_minus_G-a: +0.028441 [+0.020895, +0.036513]
  - History+state trained MLP: +0.033598 [+0.027674, +0.039357]
- unseen:
  - G-a: +0.026827 [+0.020773, +0.033092]
  - G-b: +0.073743 [+0.065235, +0.081814]
  - G-c: +0.046590 [+0.038265, +0.055211]
  - G-e: +0.016576 [+0.011645, +0.021706]
  - G-f: +0.029598 [+0.023487, +0.035999]
  - History+state: +0.025165 [+0.019188, +0.031569]
  - G-e_minus_G-a: -0.010251 [-0.014821, -0.006261]
  - G-c_minus_G-a: +0.019763 [+0.012400, +0.027755]
  - History+state trained MLP: +0.025165 [+0.019188, +0.031569]

Selected Dev settings: G-a/G-b/G-e hidden=64, lr=3e-4, epoch=8, positive weight=1.0; G-f selects epoch 7. G-c selects λ=0.3, lr=1e-4, positive weight=1.0, max epoch 2. The history+state control selects the exact G1 rule over the trained state/history MLP by Dev MSE.

