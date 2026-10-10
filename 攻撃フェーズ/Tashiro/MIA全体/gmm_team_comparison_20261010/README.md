# 予備戦20チーム：固定GMM7背景と公式加工成績

[チーム名付きCSV](予備戦20チーム_固定GMM7背景と公式有用性_MIA被害_匿名性.csv)は、同じ20チームの固定GMMを7背景で評価したTPRと、予備戦公式のU・MIA被害成分・Anonを横並びにした資料です。

- 固定方式: `q13_robust_sex0_diag_reg0.01_gmm4__raw__cross0`。
- 7背景: starter、prelim、main、starter+prelim、starter+main、prelim+main、全3種。背景にmainを使った列も、標的と正解は予備戦です。
- GMMは今回の実験のTPR@FPR=0.01。公式MIA被害成分は20攻撃者の全体/希少TPRを集約し偶然補正・正規化した別指標です。
- U・Anonは高いほど良く、MIA/AIA被害成分は低いほど良い。単一GMMのTPRを公式匿名性へ置き換えていません。
- 希少メンバーがいない場合の公式希少TPRは空欄で、数値0とは区別しています。
- これは予備戦の元ルールによる公式点との比較です。[本戦ルールへの仮想変更](../prelim_under_main_rules_20261010/README.md)とは別表です。

## 検証・出典

`official_join_validation.json`に、20チーム×7背景=140点の結合、公式U/匿名性/総合点の算術一致、出典ハッシュを記録しました。CSVには各セル・各背景の元ファイルと出典行番号を保持しています。

公式結果: [公開用結果表_20260909.xlsx](https://drive.google.com/file/d/1BLiwflR7FJ9tp5lCeSkP-kJga4ByHRnB/view)。SHA-256は`8d6380feafa50a850b28aa50eb4b04cd0ffe132ca15904dc0c84b85fd8ff6508`。

## 再現

`python reproduce_official_comparison.py --workbook /path/公開用結果表_20260909.xlsx --gmm-source /path/fixed_methods_all_teams_seven_backgrounds.csv --utility-ledger ../prelim_under_main_rules_20261010/utility/prelim_official_utility_all22.csv --out-dir /path/new_output`

標準ライブラリのみ。生データや個人別攻撃確信度を同梱していません。GMM元集計のハッシュは`official_join_validation.json`、元集計の相対パスはCSVに記載しています。
