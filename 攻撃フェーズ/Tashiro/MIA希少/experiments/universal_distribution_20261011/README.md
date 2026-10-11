# 全配布パターン検証の再現コード

人数が判明した後の個人特定を、候補 `0..10` の抽象問題として再現します。競技 CSV、公開属性の照合、公式キットは使いません。保存証明は [analysis/universal_distribution_20261011](../../analysis/universal_distribution_20261011/README.md)、結果の説明は [全配布検証の説明](../../analysis/universal_distribution_20261011/説明.md) を参照してください。

Python 3.12 と NumPy 2.4.6 を使用します。基本監査は Z3 や旧ソルバーを import しません。旧ソルバー原本のハッシュと、旧 a3b3 木の保存証明は検査するため、既存の [fast_solver_20261007](../fast_solver_20261007/) と [full_depth4_20261007](../../analysis/full_depth4_20261007/) も同じ位置に必要です。旧 manifest の `candidate_ids` は計算に使いません。

リポジトリ直下の Python 環境で実行します。

```bash
.venv/bin/python -m pip install -r 攻撃フェーズ/Tashiro/MIA希少/experiments/universal_distribution_20261011/requirements.txt
.venv/bin/python -B 攻撃フェーズ/Tashiro/MIA希少/experiments/universal_distribution_20261011/audit.py
```

既定の監査は読み取り専用です。最後に `"status": "PASS"` と表示されます。全状態再生を含み、検証時は約 16 秒でした。実行時間は環境によって変わります。

## 基本監査が再計算する内容

- 固定 5 問：42 人数型、正人数 41 型の全 13,103 回答、zero 型の 1 回答、各平均の完全分離と最小間隔。実採用の 5 問について単独 11 通り、ペア 55 通りの符号を確認します。
- 固定 4 問：対象共通の 4bit 集合 65,536 通りと 11 語集合 4,368 通りの下界、共通固定の 14 SAT / 28 UNSAT、対象別固定 SAT の追加 4 証明を確認します。追加証明は 461,340 ペア状態と 473,220 残余集合キーを全件照合します。
- 適応 4 問の下界：単一ペアの候補 10・11 人では組合せ下界、9 人では全質問分割の独立 AND/OR 再計算、8 人では保存 SAT 木の 28 状態を再生します。9 人の下界を `b>0, a+2b<=4` の 4 型へ移します。補足 `anchor_b2.json.gz` の専用 56 根探索は、この基本監査では再実行しません。同じ `(0,2)` の結論は 9 人の下界から検証します。
- 適応 4 問の保存 SAT：`(a,b)=(1,5),(2,4),(0,5),(4,3)` の全 **9,979,200 状態**を再生します。4 回ずつの **39,916,800 平均応答**から全対象の人数応答を逆算し、補集合による `n-h` 復元、候補選択、既知非会員 13 行の padding、完全な回答履歴と一意な所属を確認します。旧 a3b3 の 53 証明を使う a4b3 lift も全 4,989,600 状態を確認します。
- 公開証明の原本 45 ファイルと旧ソルバー原本の SHA256、42 型の最終分類を確認します。全ての入力が実行中に変化しなかったことも検査します。

平均点は `Fraction` で導出した共通分母の整数で扱います。各対象の式は `(24-n)h / [100*n*(M-h)]`、平均は正人数対象だけを分母に含めます。13 行の既知非会員を追加して `M` を実現します。小数 4 桁の画面表示から復号できるという主張は含みません。

全人数ベクトルへの固定 5 問保証は、型別の平均分離と対象別の局所符号を合成した証明です。82,995,960 個の人数ベクトルのそれぞれについて全個人所属を直接総当たりする方法は使っていません。適応 4 問の新 4 型は全個人所属を直接再生しています。

## 監査結果を保存する

`--output-dir` を明示した場合だけ、新しいフォルダと `audit.json` を作成します。指定先が既に存在する場合は、監査開始前に拒否します。

```bash
.venv/bin/python -B 攻撃フェーズ/Tashiro/MIA希少/experiments/universal_distribution_20261011/audit.py \
  --output-dir /tmp/mia-universal-audit-new
```

## 82,995,960 件の保存列挙を監査する

125 MiB の列挙 gzip は Git 履歴に入れず、Release 添付として扱います。[release_attachment.json](../../analysis/universal_distribution_20261011/release_attachment.json) のサイズ・ハッシュ・42 gzip member の形式に対応します。取得したファイルを明示すると、基本監査に加えて全 82,995,960 件を読みます。

```bash
.venv/bin/python -B 攻撃フェーズ/Tashiro/MIA希少/experiments/universal_distribution_20261011/audit.py \
  --enumeration-file /path/to/all-count-patterns.uint64.gz \
  --output-dir /tmp/mia-universal-full-audit-new
```

各レコードの 22bit マスクの範囲、単独・ペアラベルの非重複、人数、型内の厳密昇順、型ごとの組合せ件数、全 raw / gzip ハッシュを検査します。各ラベルへ ID 昇順で M と同じ 5 回の padding を割り当てます。全員がいずれかの対象に属する `=11` の 6 型 **49,402,850 件**も部分集合として確認します。

歴史 gzip の bytes が一致する場合は記録された 42 member の圧縮範囲も検査します。圧縮実装やヘッダーの違いで gzip の bytes が異なる場合は、連結展開 stream を型ごとの正確な raw 件数で区切り、全レコード・各型の raw SHA・全体の raw SHA を検査します。raw が一致しなければ失敗します。圧縮差は `historical_gzip_byte_match: false` と、実際の gzip SHA・サイズに記録されます。

列挙ファイルを手元で再生成する場合は `enumerate.py` を使います。こちらも `--output-dir` が必須で、既存出力を拒否します。

```bash
.venv/bin/python -B 攻撃フェーズ/Tashiro/MIA希少/experiments/universal_distribution_20261011/enumerate.py \
  --output-dir /tmp/mia-universal-enumeration-new
```

## ファイルと探索の扱い

| ファイル | 内容 |
|---|---|
| `audit.py` | 基本監査・任意の全列挙監査の入口 |
| `adaptive_three.py` | 新 3 型の独立生成・代表木・対称移送・平均復号 |
| `adaptive_lift.py` | 旧 a3b3 木から a4b3 への全状態 lift |
| `fixed_certificates.py` | 対象別固定 4 問の OR/AND 残余集合監査 |
| `binary_bounds.py` | 固定 4bit 下界、共通固定の完全分類 |
| `scalar_bounds.py` | 適応単一ペアの独立下界と保存木再生 |
| `enumeration.py` | 大きな保存列挙を chunk ごとに監査 |
| `enumerate.py` | 抽象人数ベクトルの明示的な新規生成 |
| `provenance.json` | 原本コードの SHA256 と公開先への移動対応 |

この監査は保存証明を再生するものです。大規模な SAT 木の探索をやり直しません。探索の元実装は [旧ソルバー](../fast_solver_20261007/) に維持し、探索用の追加依存は `requirements-search.txt` に固定しています。固定 SAT の 18 型の適応探索は行いません。期限付き探索の UNKNOWN を UNSAT に変換しません。

対象別固定 4 問は 18 SAT / 20 UNSAT / 4 UNKNOWN です。適応 4 問は固定 SAT を 18 型で省略し、追加 5 SAT / 4 UNSAT / 15 未判定です。単一ペア 8 人の条件付き SAT を全体の人数型へ直接移すことはしていません。任意の配布への保証は、この二値質問モデルでは 5 問です。最初の人数特定 2 回、C や公開属性による前処理、任意の多段階 confidence への下界は対象外です。
