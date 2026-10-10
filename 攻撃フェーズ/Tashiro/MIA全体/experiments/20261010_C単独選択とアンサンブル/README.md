# C単独の方式選択・アンサンブル

> **固定P01との比較を追加（2026-10-10 04:25 UTC改訂）**：主18対象では固定P01（予備戦BG）が平均TPR 0.029775、8家族代表の順位平均は0.021801（3勝10分5敗）。今回の結果は、既存P01を置き換える根拠を示していません。[固定P01との比較表](fixed_P01_baseline_comparison.csv)／[条件・行対応の監査](fixed_P01_baseline_audit.json)。従来のLOTO訓練選択基準との比較値は、条件を区別して本文に残しています。

[検証結果](Cだけの方式選択とアンサンブル_検証結果.md)／[C特徴対応シャッフルと独立監査の追補](追補_C特徴対応シャッフルと独立監査.md)

- [方式別集計](summary.csv)
- [保留対象別の結果](heldout_by_target.csv)
- [主18対象の比較](primary18_team_comparison.csv)
- [独立検算](independent_validation.json)
- [C特徴対応シャッフル](C_feature_permutation_control.csv)
- [方式候補カタログ（圧縮CSV）](candidate_catalog.csv.gz)

候補カタログは個人行ではなく、方式・背景・対象別の集計です。元配布物のmanifestでカタログを照合する場合は、gzip解凍後のCSVのSHA256を使います。個人別予測配列、所属真値、元C行データは含みません。
