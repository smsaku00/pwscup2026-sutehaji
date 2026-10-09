# 保存した出力の見方

このフォルダは、**公式予備戦21対象の探索的実験**の結果です。`summary.csv`、`auc_summary.csv`、`augment_summary.csv`、`paired_statistics.json`から読むことを推奨します。

対象別の詳細はGitHub容量と閲覧性のためgzip圧縮（`per_team_metrics.csv.gz`、`augment_target_metrics.csv.gz`、`background_split_detail.csv.gz`）。`gzip -dc <ファイル名>.csv.gz > <ファイル名>.csv` で展開してください。

行ごとの高次元スコア`density_public_scores.csv`（6,846行）と統合モデルの行別予測`augment_public_predictions.csv`はGitHubへ追加していません。コード実行で生成できます。GitHubに置いたコードと集計は原データの個票や秘密トークンを含みません。

`manifest.json`は公式予備戦入力のSHA256を記録しています。コホート別`phi`は計算せず、得点は\(R_{all}\)、公式希少補正付き\(R_{rare}\)、\(R_{max}\)までです。
