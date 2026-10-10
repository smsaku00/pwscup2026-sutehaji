# 第5回 MIA 指定5方式の再現

第3回提出原本の通常875人へ移植する5列を生成する支援資料。**大会へは未提出**。希少質問更新・原本17対象の保存・提出ZIP化は上位の統合スクリプトが担当する。

基準は `smsaku00/pwscup2026-sutehaji` の `1a5d47bba85711322994098b0de16df894acdf95`。`generate_selected.py` はこの基準の `攻撃フェーズ/Tashiro/source_recovery/20261010_MIA元コード回収/originals/本戦/攻撃フェーズ/Tashiro/` にある原コードを参照する。原本がこの作業用コピーにある場合は同ディレクトリの別名ファイルを使う。8原本はすべて取得したGit blob SHAで確認し、関数本文を変更せずASTで指定pure関数だけ抽出する。原モジュールのトップレベル・main・evaluate・所属正解読込関数は実行・ロードしない。

| 本戦対象 | 方式ID | 定義上の確認 |
|---|---|---|
| 15 さいふぉん-3 | `q7_hist5_s100_pairwise__county500__cross0.1` | raw数値5属性のA5分位binとsex/prefecture、21ペアの対数頻度比平均。A add-one prior、C shrink100。 |
| 13 あかぎ団 | `q7_pref_sex_bins20_s1_zscore__county500__cross0.1` | pref×sex群、5数値のA20分位bin、A群を全体prior50人へ縮約、C shrink1の頻度差z合計。 |
| 7 Trailblazer | `q7_robust_lofC_k100__age100__cross0.1` | TG/ALT log1p、A全体median/IQRでrobust変換後の数値5属性のみ。Cにnovelty LOF k100をfit、score_samples。 |
| 5 神神公社 | `q13_robust_sex1_diag_reg0.01_gmm4__raw__cross0.1` | 数値8属性をA全体robust変換してから性別別4成分diag GMM reg.01。C-A log密度差に5カテゴリ周辺比alpha20を加算。性別非分割の安定案とは別。 |
| 14 全部生成 | `q7_median_iqr_sex1_kde_ratio_h1__age100__cross0.1` | C数値を性別別median/IQRでAへ整列、TG/ALTはlog空間で整列。KDEはhard性別分割せず、原混合カテゴリ核とGaussian数値核によるC/A log密度差。 |

背景はstarter→prelim→mainの各5,143行を同じヘッダで連結した15,429行。bare record_idの重複は保持する。連結CSVのSHA256は `01772d964672cd742c0d44ac69e058d2db91283c330edac78671a0ec2d7fb650` で既存all3 subsetに一致。全CSVを `float_precision='round_trip'` で読む。

較正は対象自身を除く本戦公開AIA Q全体（自チーム28を含む）を使う。county500はprefecture×sex、age100はsex×全A年齢5群。midpoint CDFのlocal/global混合にB1049人内average順位0.001を加え、通常875人内average順位へ変換する。同一方式の全22対象を作成して、対象自身を除く他21対象の行中央値を0.1引く。最後に通常875人内min順位で `0.6+0.3*rank/875` とし、指定列だけ移植する。列を混合した後の追加cross・割当・アンサンブルは行わない。

GMMはA seed20261008、C seed20261008+本戦対象ID、max_iter200、n_init2。他4方式に乱数はない。KDEはquery rowsのみ256行ずつ分割して原mixed_logsを呼ぶ。全queryのsex/prefecture水準がAに含まれることを確認し、各対象12行を原geometryで直接再計算して同値を検証した。

## 再実行

リポジトリのルート `.venv/bin/python` を使用し、出力には新規ディレクトリを指定する。`--data-root` は公式データを含むリポジトリ、または既存ユーザー側checkout。`--source-root` は上記回収原コードを含むリポジトリ。starter背景は確認済みblob `34aa7c6b9cf14772502c1e12e1d9088340afa9eb` のA_bg.csvを指定する。

```sh
.venv/bin/python -B 攻撃フェーズ/Tashiro/MIA全体/round5_generation_20261010/methods/generate_selected.py \
  --data-root /path/to/repository \
  --source-root /path/to/repository \
  --starter-bg /path/to/starter/participant_data/A_bg.csv \
  --output /path/to/new-output-directory
```

`input_spec.json` は固定baseの入力blobと原コードpath/hashを保存する。生入力は再掲載しない。実行環境はPython3.12.15 / numpy2.4.6 / pandas2.3.3 / scipy1.17.1 / scikit-learn1.9.0、数値ライブラリthread limit1。原alignの整数列から小数列への代入はこのpandasで保持するため、将来版で挙動が変わった場合は原方式の値を再確認する。

## 成果物と検証

- `generated/selected_confidence.csv`: `record_id,5,7,13,14,15`、通常875行。統合入力。
- `generated/row_partitions.csv`: 原B順1049行、排他的区分はordinary875 / nonrare_known_negative150 / rare24。
- `generated/generation_manifest.json`: 入力50ファイル・原8コードのSHA256とGit blob、環境、seed、変換、出力hash、再実行同値証明。
- `generated/gmm_fit_diagnostics.csv`: A2群+C44群の適合情報。46群すべて収束。
- `generated/all_method_arrays.npz`: ローカル検証用。5×875×22のvariant/cross、5×1049×22のconfidence/rawB、4×8547×22のrawQI7 queryを保存。GitHub・配布ZIPには掲載せず、独立監査の再実行時に生成する。
- `repeated_generation_validation.json`: 最終driverで再実行し、selected CSV・区分CSV・全NPZ配列はbyte一致。GMM数値適合値も一致し、物理CPU照会に関する初回の環境warningだけ表示差がある。

通常confidenceは0.6003428571428571〜0.8999999999999999。この再現器の希少質問更新前の中間出力では、既知非共有163人の全方式・全対象confidenceは0。最終統合では原固定5bit仕様に従い、希少と重なる13人へ質問値0.1/0.4を適用し、非希少150人を0のまま保持する。KDE原geometry対照は22対象×12行すべて誤差0。独立した静的監査でも原8コードblob、5方式の入力・変換・cross・min順位化が一致した。

選定には過去の公開予備戦結果を使っている。今回のfitや生成では予備戦所属正解も本戦所属正解も読み込まない。本戦TPR・優位性・実際の会員識別は生成・監査だけでは判定できない。
