# 加工変更比較の再現

Python 3、numpy、pandas、scipyを使用します。元の個人単位レコードは成果物に複製していません。

## 入力

- `MIA_INPUT_MANIFEST`: 固定配布C、phase別A_bg、元B_selfを指す既存の `manifest.json`。元Bは自チーム分だけです。
- `MIA_UTILITY_DIR`: `phase_cohort_crosswalk.csv`、`utility_C_binding_audit.csv` のあるディレクトリ。
- `MIA_CFEATURE_DIR`: 既存の予備戦20対象 `C_only_features.csv` のあるディレクトリ。
- `MIA_MAIN_UTILITY_CSV`: 本戦の選定提出ID・ユーザー名を含む `main_utility_scores.csv`。

入力manifestの各pathは実行環境内の実ファイルを指す必要があります。スクリプトの既定値は解析実施環境のものです。別環境では環境変数とmanifest内pathをその環境に合わせて設定してください。過去特徴量は流用し、攻撃モデルは再実行しません。

## 実行

```bash
python compare_processing.py
python check_robustness.py
python write_report.py
```

全出力はスクリプトと同じディレクトリに保存されます。`compare_processing.py` は両phaseの公開Cと背景・自チーム元Bを記述統計化し、`check_robustness.py` は丸め許容の背景値再利用と自チーム元B対Cを計算します。`write_report.py` の判定文は測定結果に基づく定性的な解釈であり、統計的に学習した分類器ではありません。

## 公開範囲

`PUBLIC_FILES.json` に記載されたファイルのみが公開対象です。元B、加工後C、参加者単位の攻撃予測や正解ラベル、内部の入力パスを含む解析manifestは対象外です。公式の既公開C・公開背景への出典リンクとファイルhashは、再現のため `public_provenance.json` に記載しています。

`チーム別_加工変更の可能性.csv` を主一覧とし、詳細な集計値を `team_processing_evidence_metrics.csv`、数値分布差・相関差・丸め耐性を各CSVで確認できます。元B対Cは自チームの集計値だけを保存します。
