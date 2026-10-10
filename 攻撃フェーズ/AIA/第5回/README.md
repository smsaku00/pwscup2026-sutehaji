# 第5回の提出案：LR10＋推定希少群の弱いペア補正

Team28 / zhang / 2026-10-10 作成。

**第5回として用意した提出候補。競技サイトへの提出・本戦の公式採点はまだ行っていない。**

**GitHubの共有ZIPは `token.txt` を除いたもの。そのまま提出できない。以下のコードでトークンをローカルでコピーした提出ZIPを再生成する。**

| 内容 | ファイル |
|---|---|
| トークンなしの共有ZIP | [zhang_T28_test05_LR10_rare_soft_no_token.zip](結果/zhang_T28_test05_LR10_rare_soft_no_token.zip) |
| AIA回答 7,172行 | [F_aia.csv](結果/F_aia.csv) |
| 詳しい仕組み・数式・評価・限界 | [第5回_AIA_詳細説明書.md](第5回_AIA_詳細説明書.md) |
| 第5回だけを再生成するコード | [reproduce.py](reproduce.py) |
| 所属推定と希少群補正 | [round5_engine.py](round5_engine.py) |
| 公開入力からの特徴量計算 | [feature_engine.py](feature_engine.py) |
| 公式形式チェック | [verify_submission.py](verify_submission.py) |
| 検証・生成記録 | [generation_audit.json](結果/generation_audit.json) |

所属らしさの上位75%に9.5年、その他に11年を答える。推定希少候補の上位26人だけで、同じ都道府県・性別・5歳帯の偶数人数群を作り、弱い補正を加える。本戦ではこの補正により62行が9.5年から11年に変わった。

ローカルの提出ZIPではMIAは既存の第3回MIAを同一バイトでコピーし、トークンもTeam28のものをコピーした。提出ZIP直下は `F_mia.csv`・`F_aia.csv`・`token.txt` の3ファイル。GitHubの共有ZIP直下は `F_mia.csv`・`F_aia.csv` の2ファイルのみ。トークンはGitHubのこのフォルダ、コード、説明書には収録していない。

## 再生成

Python 3.12 の環境で `pip install -r requirements.txt`。ZIPと展開済みディレクトリのどちらも入力にできる。WSLからも実行可能。

```bash
python reproduce.py \
  --prelim-common /path/to/PWSCup2026_prelim_common.zip \
  --prelim-attack /path/to/PWSCup2026_prelim_attack.zip \
  --prelim-truth /path/to/prelim_truth.zip \
  --main-attack /path/to/PWSCup2026_final_attack.zip \
  --main-team /path/to/team28.zip \
  --base-submission /path/to/round3.zip \
  --work-dir /path/to/round5_work \
  --out /path/to/round5_reproduced
```

`--base-submission` はMIAとトークンのコピー元。第3回と同じMIAを用いるZIPを指定する。代わりに `--mia-csv` と `--token-file` を指定してもよい。

再現時のローカル出力は第5回のZIP、`F_aia.csv`、`generation_audit.json`、`row_diagnostics.csv`。再生成したZIPはコード内で公式の形式検証を実行する。個人ごとの属性・所属推定スコアを含む診断CSVはGitHubには収録せず、標的別の変更件数を生成記録に収録した。

```bash
python verify_submission.py \
  --submission /path/to/round5_reproduced/zhang_T28_test05_LR10_rare_soft.zip \
  --attack-dir /path/to/round5_work/tmp/main_attack \
  --base-submission /path/to/round3.zip
```

再現用の入力展開先・出力先は、このフォルダの外を指定する。正解データは公開済みの予備選のものだけを学習に使い、本戦のB・B_self・所属正解・time正解は読み込まない。

## 予備選の確認値

自コホートを学習から外した評価。`max(R_all, R_rare)`、難易度差し引き0。

| 方法 | 加工済み18標的 | 全20標的 |
|---|---:|---:|
| LR10、希少群補正なし | 0.162984 | 0.206215 |
| **第5回：LR10＋希少群補正** | **0.175711** | **0.218155** |

18標的はコピー標的1・2と予備選自チーム5を除く。20標的は自チーム5だけを除く。これは本戦得点の予測値ではない。

[本戦AIAの入口](../README.md)
