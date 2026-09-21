# TTのAIA攻撃プログラム3案 — v1.0.0

2026-09-20作成。指定ルールブック `main-process-20260912` に基づくAIA専用の初版。
**3本はそれぞれ単独で動作する。攻撃性能は未測定。配布データ・回答CSVは同梱しない。**

| 案 | プログラム | 説明書 |
|---|---|---|
| 01 | [部分属性照合](TT-01-部分属性照合/TT-01-部分属性照合.py) | [TT-01-部分属性照合.md](TT-01-部分属性照合/TT-01-部分属性照合.md) |
| 02 | [全体対応推定](TT-02-全体対応推定/TT-02-全体対応推定.py) | [TT-02-全体対応推定.md](TT-02-全体対応推定/TT-02-全体対応推定.md) |
| 03 | [条件分布推定](TT-03-条件分布推定/TT-03-条件分布推定.py) | [TT-03-条件分布推定.md](TT-03-条件分布推定/TT-03-条件分布推定.md) |

01は部分属性の一致、02は課題全体の対応、03はCで学習したtime分布を使う。3案ともBの内容を推定に使わず、B_self・正解・採点結果は読まない。MIAは未対応。

## 実行

Python 3.11以上を想定。確認環境は3.12.7（macOS arm64）。新しく環境を作る場合:

```sh
cd "TT-01-部分属性照合"  # 02・03を使う場合は対応するフォルダへ移動
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

各案は個別フォルダに置く。使う案のフォルダへ移動し、依存ライブラリを入れてから実行する。以下の各コマンドは対応する案のフォルダで実行する:

```sh
python "TT-01-部分属性照合.py" --b "B.csv" --a-bg "A_bg.csv" --targets "targets.json"
python "TT-02-全体対応推定.py" --b "B.csv" --a-bg "A_bg.csv" --targets "targets.json"
python "TT-03-条件分布推定.py" --b "B.csv" --a-bg "A_bg.csv" --targets "targets.json"
```

`targets.例.json` を参考に対象を指定する。Cの名前は自由。同じtarget_kの複数加工候補は別々の実行にする。JSON内の相対パスはJSONの親フォルダ基準。課題ファイルはAIA用に必須。

```json
[
  {
    "target_k": 2,
    "c_path": "C_名前-01-手法名.csv",
    "aia_challenge_path": "aia_challenge_2.csv"
  }
]
```

公式の対象一覧を使う実行では、上記に `--target-set official --mia-columns "mia_columns.json" --self-team-id "$ATTACKER_TEAM_ID"` を追加する。`ATTACKER_TEAM_ID` は利用者が自チームIDに設定する変数。全公式対象から自チームを除いた集合と照合し、不足・余分があれば終了する。既定はlocalであり、実行記録にも対象集合の種類を残す。

## このフォルダの練習例

[公式キット](https://github.com/pwscup/pwscup2026-kit/releases/tag/main-process-20260912)の指定版を別途入手し、ルートを `starter-kit/` に置く。使う案のフォルダを作業フォルダにする。練習用データと本戦データは混ぜない。

```sh
python "TT-01-部分属性照合.py" --b "starter-kit/participant_data/B_practice.csv" --a-bg "starter-kit/participant_data/A_bg.csv" --targets "targets.練習.json"
python "TT-02-全体対応推定.py" --b "starter-kit/participant_data/B_practice.csv" --a-bg "starter-kit/participant_data/A_bg.csv" --targets "targets.練習.json"
python "TT-03-条件分布推定.py" --b "starter-kit/participant_data/B_practice.csv" --a-bg "starter-kit/participant_data/A_bg.csv" --targets "targets.練習.json"
```

## 出力と履歴

CSVはスクリプトと同じフォルダへ `F_aia_TT-番号-手法名.csv` の名前で保存する。列は `target_k,challenge_row_id,time_hat`。指定された全行を出し、target_k・challenge_row_idの数値順、UTF-8 BOMなし、LF、小数点以下10桁に固定する。

同名の実行記録JSONに設定・入力／コード／出力ハッシュ、時間・メモリ、local/officialの区分を残す。以前と異なる出力になる場合は `履歴/` に旧ファイルを内容ハッシュ付きで保存する。入力は変更しない。別案や加工案のフォルダを探索しない。

生成されるAIA回答だけでは大会用の完全な提出物ではない。統合・ZIP作成・競技への提出は別の作業として行う。

## 確認済みのこと

- 練習対象1・2、計652行を各案で2回実行し、CSVバイト列が一致。
- 指定版公式AIA形式検査と、識別子の一意性・過不足・値域の独自検査を通過。
- 対象順・C行順・課題行順・背景行順を変更しても回答一致。
- 別対象3への差し替え、任意のC名、相対パス、旧出力保存を確認。
- Bの中身を非CSVに変えても同じ回答。Bの内容を利用しないことを確認。
- 重複対象・ID、欠損、無限大、値域違反、正解列混入、公式対象の不足・自己対象などを拒否。

ローカル検査結果の共有用抜粋は `確認結果.json`。3案共通の再確認は親フォルダの本READMEと同じ場所から、公式キットの所在を指定して行う:

```sh
python "動作確認.py" --kit "TT-01-部分属性照合/starter-kit"
```

参照元の版・SHA-256は `参照情報.json`。形式検査は練習2対象の参照集合に対する検査で、公式本戦の全対象を揃えた提出ZIPの検査ではない。

## まだ測っていないこと

攻撃得点、標的／対照・希少群の正解数、基準法からの改善、複数加工・複数試行のばらつきは未測定。φ_iを0と仮定した本戦得点は出していない。評価担当が正解を分離して、同じ加工元に対応するCと混合課題を組み合わせて比較する。

実測の所要時間・ピークメモリと、各手法の前提・固定パラメータ・弱点は各説明書の⑥～⑫に記載した。プログラムは作成済みv1.0.0と同一。共有用説明書の参照先を2026-09-21に調整した。

## 収録範囲

プログラム3本、各説明書、依存ライブラリ一覧、対象一覧の書式例、動作確認コードと確認結果の抜粋、参照情報を収録する。配布CSV、B_self、utility_ref、トークン、正解、回答CSV、実行記録、加工候補、公式キット本体は含めない。`.gitignore` は共有対象ファイルだけを許可する構成で、実行してできたCSVやデータを通常の追加操作で追跡しない。

## フォルダ構成（2026-09-21更新）

各案のフォルダにプログラム、説明書、requirements.txt、対象例2件を格納する。共通の動作確認コード、参照情報、確認結果はこの親フォルダに配置する。Python本体はv1.0.0と同一。
