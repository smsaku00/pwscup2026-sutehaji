# 第5回MIA：指定5対象の通常部分＋固定5bit第3問

**作成済み。大会には未提出。** 第3回の保存回答を基準に、指定5対象の通常875人部分だけを更新します。希少24人部分は固定5bit第3問です。以前の未提出第5回候補とは別の新しい生成物で、[manifest.json](manifest.json)に来歴とハッシュを記録します。

- 提出用MIA：[F_mia.csv](F_mia.csv)
- **完成提出ZIP名：`MIA_AIA_round5_submission.zip`**。ルート直下にF_mia.csv・F_aia.csv・token.txtの3点のみ。ローカルとユーザーのLibraryに納品済み。認証情報の再掲載禁止に従いGitHubには置きません。
- MIA＋AIA配布ZIP：[MIA_AIA_round5_no_token.zip](MIA_AIA_round5_no_token.zip)。tokenなしのため、そのまま大会には提出できません。
- MIAだけの配布ZIP：[MIA_round5_components.zip](MIA_round5_components.zip)。中身はF_mia.csvのみ。
- [AIA・ZIP独立検証](../../Tashiro/MIA全体/round5_generation_20261010/aia_zip_independent_audit.md) / [提出ZIP検証記録](submission_zip_validation.json)。
- [生成・検証資料](../../Tashiro/MIA全体/round5_generation_20261010/README.md)
- [第3回・第4回との対象別差分](differences.csv)
- [追加の判断表・反復数比較資料（固定コミット）](https://github.com/smsaku00/pwscup2026-sutehaji/blob/928db490b092e9569d1917a0a07f3eca32be5127/攻撃フェーズ/Tashiro/MIA全体/continuity_attack_review_20261010/README.md)
- [5手法の説明（固定コミット）](https://github.com/smsaku00/pwscup2026-sutehaji/blob/5196467a82ce69dee1a5a84a1e68797f59b9c5a5/攻撃フェーズ/Tashiro/MIA全体/continuity_attack_review_20261010/手法の説明.md)

| 本戦ID | チーム | 通常部分の方式ID |
|---:|---|---|
| 15 | さいふぉん-3 | `q7_hist5_s100_pairwise__county500__cross0.1` |
| 13 | あかぎ団 | `q7_pref_sex_bins20_s1_zscore__county500__cross0.1` |
| 7 | Trailblazer | `q7_robust_lofC_k100__age100__cross0.1` |
| 5 | 神神公社 | `q13_robust_sex1_diag_reg0.01_gmm4__raw__cross0.1` |
| 14 | 全部生成 | `q7_median_iqr_sex1_kde_ratio_h1__age100__cross0.1` |

背景はstarter→prelim→mainの全3背景、15429行。各方式を本戦全22対象で計算し、方式・較正ごとの通常順位から、自対象を除く他21対象の中央値を0.1引きます。補正後に通常875人内のmin-rankで `.6 + .3 * rank / 875` へ変換し、上記5列だけ移植します。混合後の列へ追加の競合補正やアンサンブルを加えません。予備戦正解を本戦fitに使いません。

他17対象の通常部分は第3回原本を保持。k等分（本戦6）は通常変更から除外。HarunaDan@GU（本戦9）は全1049行を第3回の保存P01のまま固定します。神神の今回のsex1案は、過去監査のsex0安定案とは別設定です。

希少は第3回＝第1問、第4回＝第2問、今回第5回＝第3問（候補47・90・264・462・495）。希少あり対象は本戦6/11/17が各1人、10/16/25が各2人です。保存仕様の人数調整paddingも保持します。

排他的区分は通常875人、非希少の既知非共有150人、希少24人。既知非共有163人のうち希少13人は重複しており、原方式では希少あり6対象の質問値0.1/0.4を持ちます。したがって**常時0にするのは150人**です。163人全員を0にすると固定質問仕様が壊れます。通常は>0.6、希少は≤0.4の分離を維持します。

第3回比は「5対象の通常変更＋希少第3問への変更」で、通常4366セル・希少36セルが変わりました。第4回比は通常13938セル・希少36セルです。第4回は通常部分に別のチーム別6方式を使っているため、第4回比は5対象だけの効果ではありません。差分表を両基準で分けています。本戦正解が非公開のため本戦TPR・得点向上は未確認です。

[独立最終監査](../../Tashiro/MIA全体/round5_generation_20261010/final_round5_independent_audit.md)はPASS。1049行・22対象、有限[0,1]、列とID順、通常/希少分離、非希少非共有150人全0、Haruna全列固定、5対象外通常差分0、希少第3問528セル一致を確認しました。全5方式×22対象の較正・交差補正・順位変換を独立再計算して誤差0、公式現行MIA検証器もPASSです。統計生成と統合組立をそれぞれ再実行して同値を確認しています。

再現は生成資料のREADMEを参照。`assemble_round5.py` は検証した通常5列・区分表・希少期待値を第3回原本へ移植し、既存出力の上書きを拒否します。`package_submission.py` は指定したAIAと配布tokenを組み合わせ、公式検証器でローカル提出ZIPを検査します。tokenを含むZIPをGitHubへ掲載しないでください。

追加指定されたAIAは、2026-10-10の最新main `b023ca627fb664ffb0f4b26f34ca6a4448b26418` の [AIA/5_2/結果/F_aia.csv](https://github.com/smsaku00/pwscup2026-sutehaji/blob/b023ca627fb664ffb0f4b26f34ca6a4448b26418/攻撃フェーズ/AIA/5_2/結果/F_aia.csv) を変更せず使用。SHA256 `6f625e41a324ffb32ff2958c312a30ee2e06d3a3fac6504015443cb8f867d85c`、7172行、列順 `target_k,challenge_row_id,time_hat`、MIAと同じ22対象×各326行です。公開課題のID集合・重複・有限値と公式AIA検証を確認しました。完成ZIPのCRCと再展開CSVの原本SHA256一致も確認。完成ZIPはLibraryの `file_00000000117c8206b42e115237f90fa8`、`libfile_d417053d7a708191aef3cc7395ec8781` として保存しています。大会には未提出です。

公式仕様：[本戦ルールブック§4.1・§5.2](https://github.com/pwscup/pwscup2026-kit/blob/main/rulebook/PWS_Cup2026_ルールブック_main-process-20260912.pdf)、[公式攻撃提出検証器](https://github.com/pwscup/pwscup2026-kit/blob/main/starter/src/pwscup2026/kit/validate.py)。大会の実際の対象数は配布mia_columns.jsonを優先し、今回は自チーム28以外の22対象です。
