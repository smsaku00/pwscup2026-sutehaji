# 第5回MIA 最終独立検証

結果は **PASS**。大会へ未提出。生成器・方式driver・希少生成器をimportせず、公開本戦入力・保存原本・スコア配列から検証した。

提出用CSV SHA256: `3e1f260ed7ab2f8beed57e6eb5b6e30d0c96388ac15c1e50ef0cfd1fe0f96c7a`

| 検証 | 確認数・結果 |
|---|---:|
| 行・対象・値 | 1049行 × 22対象、全23078値が有限で[0,1] |
| 原本列順・record_id順 | 完全一致 |
| 5対象外の通常 | 14875セル差分0 |
| Haruna 本戦9 | 全1049セルの数値・文字列差分0 |
| 通常非共有150人 | 3300セル全0 |
| 希少24人 | 528セルが保存固定5bit第3問と一致 |
| 選択通常5列 | 4375セルが方式出力と一致 |
| 通常値域 | 0.6003428571428571 ～ 0.8999999999999999 |
| 希少最大 | 0.4 |
| ZIP | F_mia.csvのみ、CSV byte完全一致・破損なし |
| 公式MIA logical validator | PASS |

既知非共有163人のうち希少13人は保存仕様どおり希少あり6対象で0.1／0.4を使う。常時0は非希少150人であり、希少13人を一律0に変更していない。

本戦の公開Bと自己課題の7属性から通常875・通常非共有150・希少24の排他的区分を再導出した。保存5bit表の第1／第2／第3bitから、第3回・第4回・今回第5回の希少528セルをそれぞれ照合した。

方式入力50ファイルのhash記録を確認し、公開Q23ファイルと本戦B、原source8ファイル・driver・input spec・保存原本・生成物の実ファイルhashを照合した。背景はstarter→prelim→main、各5143行の全15429行を保持。GMMは性別分割diag4／reg0.01、seedはA=20261008、C=20261008+本戦対象IDである。

保存raw値と公開Qから全4 q7方式 × 22対象の較正を別実装で再算出した。対象自身を除く22公開課題7172行を使い、1049行Bの平均順位による同点項、通常875人の平均順位、他21対象の中央値 × 0.1、通常875人のmin順位による0.6＋0.3×rank/875まで、全5方式 × 22対象が誤差0で一致した。同点はrecord_idで破っていない。選択後の混合列には追加補正を加えていない。

| 比較原本 | 通常セル差分 | 希少セル差分 | 通常非共有セル差分 | 全セル差分 |
|---|---:|---:|---:|---:|
| 第3回 | 4366 | 36 | 0 | 4402 |
| 第4回 | 13938 | 36 | 0 | 13974 |

第4回は別の通常方式を使用した原本である。第4回比の差分を今回の5対象変更だけの効果と解釈しない。統合器差分表44行も独立算出と完全一致した。

数値証跡は[JSON](final_round5_independent_audit.json)、対象別差分は[CSV](final_round5_independent_differences.csv)。[検証スクリプト](audit_final_round5.py)は各入力を監査前後にhash照合する。方式のsource互換監査と再生成一致は[方式資料](methods/README.md)を参照。

再実行には方式driverで内部NPZを再生成してから次を実行する。内部NPZは約7.2 MBの検証cacheでありGitHub掲載対象から除外できる。

```sh
python -B 攻撃フェーズ/Tashiro/MIA全体/round5_generation_20261010/audit_final_round5.py \
  --candidate 攻撃フェーズ/MIA/5回目/F_mia.csv \
  --component-zip 攻撃フェーズ/MIA/5回目/MIA_round5_components.zip \
  --selected-ordinary 攻撃フェーズ/Tashiro/MIA全体/round5_generation_20261010/methods/generated/selected_confidence.csv \
  --method-arrays 攻撃フェーズ/Tashiro/MIA全体/round5_generation_20261010/methods/generated/all_method_arrays.npz \
  --method-manifest 攻撃フェーズ/Tashiro/MIA全体/round5_generation_20261010/methods/generated/generation_manifest.json
```

本監査は整合性・再現処理の検証であり、未知の本戦所属正解による精度評価ではない。希少人数は保存公式結果と共有条件に基づく条件付き導出である。MIA単体ZIPは配布用で、公式攻撃ZIPにはF_aia.csvとtoken.txtの統合が必要。大会提出・採点・個人所属確定は行っていない。
