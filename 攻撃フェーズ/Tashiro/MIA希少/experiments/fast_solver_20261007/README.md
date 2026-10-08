# MIA希少の探索実装・独立検証・質問参照

> **【2026-10-08の注意】** ここは適応4回・5回方式の探索実験です。第3回の実提出は別の非適応・固定5bitです。`inspect_strategy.py`の適応4回質問を、**実提出の続きとしてそのまま使用しないでください**。[共通MIAの実提出](../../../../MIA/20261008_第3回MIA_固定5bitと非会員較正/README.md)／[現在地](../../README.md)。

- 更新日：2026年10月7日
- 共通入口：[MIA希少のREADME](../../README.md)
- 方法の説明：[5回案](../../docs/17_追加5回で個人特定できる理由.md)、[4回案](../../docs/18_追加4回の進め方と完遂保証.md)

## 1 適応4回案としての完成結果

- **4回案：全2,494,800配置・53代表・209 q3枝の具体的な質問木の構成済み**
- 独立replay：衝突0、q4後最大bucket1,338、q5後最大cell44、最大追加depth4
- 公式平均点：全216回答の一意復号、最小間隔 `1/54,000`
- 証明資料：[全状態証明](../../docs/16_追加4回完遂の全状態証明.md)
- 完成証明書・最終集計：`../../analysis/full_depth4_20261007/`
- 本番の個人所属：未特定、今後の実際の平均点の取得が必要

## 2 現在の実装の役割

- `solver.py`、`continuation.py`：AND/OR探索、memoization、failure cache、query生成への制約フィードバック、beam・LDS・minimax ordering
- `complete_witness.py`、`solve_all_orbits.py`：q4全枝のq5・q6構成、自己同型による移植、checkpoint
- `solve_two_cegis.py`：exact depth-2 domain内の候補生成、最終1回の失敗coreを制約へ反映
- `verify.py`、`verify_failures.py`：局所証明書、移植、固定query範囲の不可能証明の独立検査
- `verify_all_orbits.py`：全合法状態の独立生成、全209枝のreplay、深さ、状態集合の移植一致、平均点一意性の検査
- `verify_score_encoding.py`：有理数ROC補間、公式採点・集約関数との照合
- `inspect_strategy.py`：正規化済みcode・生の平均点から次の人選・paddingを参照する読み取り専用ツール
- 独立checkerと参照ツール：標準ライブラリのみ、NumPy・Z3不要
- SAT・UNSAT・UNKNOWN：有界探索の未発見はUNKNOWN、最終SATの根拠は完成証明書と独立audit

## 3 比較実験の起動手順

- Python：プロジェクトルートの `.venv/bin/python`、Homebrew Python 3.12
- 実行場所：この `experiments/fast_solver_20261007/` フォルダ
- 下の例：ルート `.venv` のパスを指定し、読み取り専用の参照と独立検証を実行
- 証明書・コード・入力：再検証前にハッシュと実行環境を確認

```bash
PWS_PYTHON=/Users/taisei1105/pws2026-local/.venv/bin/python
"$PWS_PYTHON" inspect_strategy.py ../../analysis/full_depth4_20261007
"$PWS_PYTHON" inspect_strategy.py ../../analysis/full_depth4_20261007 --codes 30 253 246
"$PWS_PYTHON" verify_all_orbits.py ../../analysis/full_depth4_20261007
"$PWS_PYTHON" verify_score_encoding.py ../../analysis/full_depth4_20261007/universal_score_encoding.json
```

- 平均点を使う参照：`--means <q3点数> <q4点数> ...`
- 4回答後：1配置への到達時に所属IDを表示
- 新しい探索の再開：別の出力先と実行記録を使用
- 依存の導入：ルート `requirements.txt` とプロジェクト指定のPython環境を使用

## 4 初期試作から完成までの履歴

- 以下の記録：初期ベンチマーク、3枝の完遂、全軌道への拡張を時系列で保存
- 「未実装」「未解決」「次の改善」：各段階の状態
- Python 3.13.5・Linuxのベンチマーク：当時の実行環境の記録
- 下の旧コマンド・旧依存指定：当時の記録、現在の起動は上の手順
- 初期ベンチマークの未測定事項：全軌道の質問木を完成する所要時間の短縮率
- 現在の状態と最終集計：上の完成結果と全状態証明を参照

### 初期ベンチマークの結論

- 主要な検査処理では高速化を実測した
- ただし、**全軌道の完全決定木を求める時間の短縮率は未測定**であり、「全体が100倍速い」とは主張しない
- 既存探索や提出CSVを変更せず、新規実験フォルダ内で実装・試験した

#### 同一入力・同一環境での比較

- 比較対象は、会話で使っていた「mask列挙・固定他Teamのfiber検査・状態対被覆」の手順を再現した単純な参照実装 `reference.py`
- 過去ログの実行時間や既存リポジトリ全体との直接比較ではない

|対象|参照実装|高速化版|比率|
|---|---:|---:|---:|
|662状態の残り2回必要条件検査・コールド開始|0.185877秒|0.009360秒|約19.9倍|
|40,428状態に対する異なる第4回9候補・前処理込み初回|3.958694秒|1.536287秒|約2.6倍|
|同じ9候補・部品キャッシュ構築済み|3.182890秒|0.023589秒|約134.9倍|
|最終1回の20セル判定・ソルバキャッシュ空|0.463674秒|0.250496秒|約1.9倍|

- コールドの2行目は1回のbatch計測
- それ以外は同じ条件の3回の中央値
- 生成済み状態を入力にし、importと状態生成は計測外
- 速度比較は単一プロセスで実施した
- ウォームでは両実装ともcacheを残した
- 高速化版は回答結果を丸ごとlookupしただけではなく、各候補の全状態responseを毎回再計算している

- 環境：Python 3.13.5 / NumPy 2.3.5 / Linux x86_64
- 詳細な生計測は `benchmark_results.json`

### 実装したもの

- `Solver.solve(states, depth)`：残り深さに対するAND/OR探索
- 正確に同一の状態集合をメモ化する
- `SAT / UNSAT / UNKNOWN` の3値
- beamや時間上限で止まった場合は **UNKNOWN**
- 不可能としてcacheしない
- 最終1回：Teamごとに同じpartitionを作るmaskをまとめ、状態対被覆を厳密DFS
- 返ったqueryは別の整数演算で再確認する
- 残り2回：他5Team固定fiberごとに、許容される次の1024 maskをbitsetで保持する
- 共通集合が空なら必要条件違反として棄却する
- 第4回候補比較：親状態のfiberを一度だけ作り、(Team, mask)ごとの制約を再利用する
- これが今回の大きな速度差の主要因
- 有界beamによる質問候補生成
- 小さいquery直積は全列挙
- 固定q3の自己同型を実際の入射関係から導出・検査する補助関数
- candidate側の同一profile入替も含め144変換、Team側は6変換

- **まだ含めないもの**：再帰内部で全candidate relabelingを使った同型cache、LDS、SAT連携、全q3軌道の探索、公式mean-TPRでのpadding・丸め・提出ファイル検証
- 文献のDL8.5そのものを実行したわけではなく、今回の識別モデル向けの試作

### 正しさの確認

- `python -m unittest -v test_solver` の7テストグループが成功

- q3 code30の40,428状態、既知第4回の213枝・最大1,077・問題枝662を再生成して一致を確認
- 4頂点の全63個の非空edge集合について、高速版と別ループ実装の許容mask集合が一致
- 最終1回の24小ケースを別実装と照合
- SAT witnessは純Python整数のcheckerで再検算
- 時間・ノード上限はUNKNOWNになり、負のcacheに入らないことを確認
- response合成時は掛け算前にint32へ昇格
- 全候補を高側にしたresponse code637などでuint8 overflowを検出する回帰テスト
- 何も新しく分離しないTeamもDFS上で通過させる
- 無情報Teamを理由に後続Teamの探索を誤って止めない
- complementによるpartition一致と、正しいq3自己同型を検査

- 探索コードとは別の `verify.py` は、各reachable outcome、葉のsingleton性、最終queryの一意性、深さをチェックする
- 小さい6枝について証拠JSONを保存済み

### 並行探索の結果

- 2プロセスで、code30の第4回についてTeam10/16を変える2つの有限gridを検査済み

- 候補評価回数：1,024回（2 gridにbaseline重複を含むため、すべてが異なる候補とは主張しない）
- wall time：6.416秒（プロセス起動等を含む）
- 残り2回の必要条件を全枝が通過した候補：0
- この有限探索の全体結果：**UNKNOWN**
- code30全体の不可能性は証明していない

- 統一solverの動作確認として、現在の第4回の小さい6枝（4/6/6/6/8/9状態）を追加で解き、すべてwitnessの独立replayが成功した
- これは40,428状態全体の解決ではなく、既知の662状態の障害も残っている
- 完成済み軌道数を増やす結果ではない

### 再実行

```bash
python -m pip install 'numpy==2.3.5'
python -m unittest -v test_solver
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python probe.py --workers 2
python verify.py certificate_q3_30_q4_621.json
```

- `--workers 1` は逐次実行
- プロセスは終了を待ち、常駐・定期実行・自動提出はしない
- 入力は条件から再生成した組合せ状態であり、実レコードの属性値や秘密ラベルは使わない

### 研究上の位置づけ

- DL8.5はcacheとbranch-and-boundによる高速化を報告しているが、対象は分類木の最適化であり、今回の6 Team・多値回答の識別問題に同じ倍率が出る保証はない

- CA-DL8.5は厳密DFSが初期に偏る問題を扱い、探索の制限を徐々に緩めるcomplete anytime searchを提案する
- 今回の有限beamは未緩和のため、未発見をUNSATにしない

- 2-defective QGTの文献は1 Teamのcount oracleに対応するが、6 Teamの結合と平均スコアの復号まで自動で扱うものではない

- Aglin et al., AAAI 2020, https://ojs.aaai.org/index.php/AAAI/article/view/5711
- Kiossou et al., 2025, https://arxiv.org/abs/2508.06064
- Karpelevitch, 2026-09-29 preprint, https://arxiv.org/abs/2609.38489

- 次の改善は、今回高速化したscreenを使った**不可能部分状態の記録と再利用**、およびquery生成への制約フィードバック
- 必要条件だけを通過した質問を採用せず、必ず完全なwitnessを独立検証する

### 続行結果：code30/246/270を完遂（2026-10-07）

- 前半は初期試作・ベンチマーク時点の記録
- 続行でfailure subproblem cache、query生成へのUNSAT制約フィードバック、minimax/beam/LDS、q5 partition代表の重複除去、独立状態生成・移植checkerを追加した

- code30の第4回 `[432,962,939,562,392,951]` で全208枝を解決し、246/270へ移植した
- 各40,428状態を独立replayし、q4最大bucket899、q5最大cell20、q5セル16,321個、衝突0・最大depth3を確認済み
- 詳細と具体的な3枝のmaskは [完遂報告](../../docs/15_q3軌道30_246_270_6回完遂.md)
- 証拠は `../../analysis/code30_continuation_20261007/`

- 追加コード:

- `continuation.py`: 2-Team gridと第3Teamのactive mask代表、失敗coreの禁止直積、minimax beam / rank-sum LDS
- 固定範囲の完全除外のみUNSAT
- 有界・未解決の場合はUNKNOWN
- `complete_witness.py`: 全q4枝のq5/q6構成、checkpoint再開、30→246/270移植、独立replay
- singletonにもq6の具体的なゼロmaskを保存
- `verify.py`: solver/modelから独立した状態再生成・fingerprint・全state replay
- `--source` で全状態集合と全query partitionの移植一致も確認
- `verify_failures.py`: scalar全maskループによるUNSAT core検算、禁止直積の全grid被覆検算
- `constrained_search.py`: optional Z3の有界CEGIS試験
- SATは質問生成の制約充足だけを意味し、決定木SATとは区別する
- 今回の有界実行はUNKNOWN

- Python 3.12 / NumPy 2.3.5で続行した
- Z3はoptionalで、利用時のみ `requirements_constraints.txt` を追加する
- 完成witnessの検証にNumPy/Z3は不要

```bash
python continuation.py --seconds 120 --beam 16 --output two_team.json.gz
python continuation.py --skip-baseline --third-team 4 --third-all --load-cores two_team.json.gz --seconds 120 --beam 16 --output three_team17.json.gz
python complete_witness.py --query 432 962 939 562 392 951 --seconds-per-branch 60 --sample-queries 4096 --query-budget 256
python verify.py ../../analysis/code30_continuation_20261007/q3_30_certificate.json
python verify.py ../../analysis/code30_continuation_20261007/q3_246_certificate.json --source ../../analysis/code30_continuation_20261007/q3_30_certificate.json
python verify.py ../../analysis/code30_continuation_20261007/q3_270_certificate.json --source ../../analysis/code30_continuation_20261007/q3_30_certificate.json
```

- 既存のSAT checkpointは最終独立replayを再実行する
- 再探索したい場合は別の `--output-dir` を指定する
- 11テストと保存した3枝の全state replayが成功した

### 全軌道と公式平均点の完遂

- 追加の構成と検証:

- `solve_all_orbits.py`: 全53代表のq4/q5/q6構成とcheckpoint
- 提案q5の失敗をq4枝全体のUNSATにしない
- `solve_two_cegis.py`: exact depth-2 domain内の質問生成と、最終1回で証明した失敗coreの制約フィードバック
- 有界終了はUNKNOWN
- `score_encoding.py`: 補集合で各queryを5bit以下に正規化したときの、全質問共通の平均点符号化
- 高側行数 `[6,7,7,11,10,12]`、最小間隔 `1/54,000`
- `verify_all_orbits.py`: 標準ライブラリだけの全状態生成・自己同型状態集合・全209枝replay・平均点一意性のchecker
- `verify_score_encoding.py`: 独立した有理数ROC補間と、任意で公式採点・平均関数との照合
- `inspect_strategy.py`: 正規化済み回答codeまたは生の平均点から、次のqueryとpaddingを参照する読み取り専用ツール
- 4回答後は識別した所属IDを表示
- `test_depth4.py`: 失敗した提案q5の扱い、depth-2 SAT/UNSAT/UNKNOWN、補集合の回答ラベル、216平均点と小数5桁からの復号の回帰検査

```bash
python -m unittest -v test_solver test_depth4
python verify_all_orbits.py ../../analysis/full_depth4_20261007
python verify_score_encoding.py ../../analysis/full_depth4_20261007/universal_score_encoding.json
python inspect_strategy.py ../../analysis/full_depth4_20261007 --codes 30 253 246
```

- 15テスト成功
- 全状態auditはSAT、2,494,800状態、衝突0、q4〜q6最大depth3、q3〜q6最大depth4
- 前節までのideal count-vectorだけの結果を、公式mean-TPRで復号可能な全状態の存在証明へ拡張した
- 提出用CSV・`rounds/`は変更なし
