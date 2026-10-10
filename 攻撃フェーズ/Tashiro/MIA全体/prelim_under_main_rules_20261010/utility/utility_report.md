# 予備戦加工データを本戦有用性ルールで評価する

## 結論

予備戦の各チームの加工データ C を固定し、本戦の有用性採点コード・設定へ変更する反実仮想を検証した。自チーム5は元の B と参照値が揃っているため、公式コードと公式固定依存版を使って全項目をローカル再計算できた。**U は 0.9825875274 から 0.7861794165 に下がる。新設された列間相関の忠実度が制約となる。**

他20チームの元 B とチーム別参照値は確認した資料にないため、完全な再採点値を作ることはできない。ただし、C だけで確定する新規減点から、次の3チームは有用性低下を確定できる。

| 予備戦番号 | チーム | 予備戦 U | 本戦式 U の C 単独上限 | 低下幅の下限 | 原因 |
|---|---|---:|---:|---:|---|
| 17 | あれ / are | 0.9418734761 | 0.7500000000 | 0.1918734751 | 非発症かつ非死亡の行が0件。新 cens は0 |
| 19 | Asakatsu2026 | 0.9711889682 | 0.9463775024 | 0.0248114648 | 完全重複の余分な行45件。no_dup は0.7855100095 |
| 29 | 神ぼ大/jinbodai | 0.9865575382 | 0.8605815062 | 0.1259760310 | 完全重複の余分な行117件。no_dup は0.4423260248 |

この表は最大限有利に評価した場合の上限であり、点予測ではない。未計算の相関・裾・新規性でさらに低くなる可能性がある。予備戦の掲載値に対して保守的な 1e-9 の許容幅を取った。匿名性および総合得点は本ファイルの対象外であり、U の低下がそのまま総合得点低下になるとは限らない。

## 何を固定して何を変更したか

- 固定: 予備戦の C の元ファイル、対応する予備戦 B、予備戦の Mahalanobis・KM 参照値、予備戦 A_bg
- 変更: 採点コードと採点設定を main-process-20260912 に変更。本戦設定に含まれる既定の相関目盛り、tail の固定較正表、novel の deadzone を採用
- CSV の読み方も各期の公式実装に従う。予備戦は pandas 既定、本戦は read_scoring_csv の round_trip
- 本戦の実際の B や C への取り替え、予備戦母集団での本戦 tail 較正表の作り直し、加工方法の変更は行わない
- 本戦順位表の予測や、事務局が公表した本戦採点結果ではない
- 自チーム5を含む C がある21チームを対象とし、C のない21番は欠測行を残す

## 自チーム5の全項目再計算

| 項目 | 予備戦式の再計算 | 本戦式の再計算 | 差 |
|---|---:|---:|---:|
| U | 0.9825875274 | 0.7861794165 | -0.1964081108 |
| U_gen | 0.9873212583 | 0.7861794165 | -0.2011418418 |
| U_spec | 0.9825875274 | 0.9825875274 | 数値丸め差のみ |
| U_rare | 0.9976011726 | 0.9976011726 | 0 |
| U_valid | 1.0000000000 | 0.9995233556 | -0.0004766444 |

予備戦式の U と全4ファセットは、公式掲載値との最大絶対差 4.13e-11 で再現した。公式 requirements-kit.txt の全30依存パッケージの版を一致させて再実行し、初回の別依存版での結果とも U は一致した。同じ Docker イメージそのものではなく、公式コード・固定依存版によるローカル再計算である。

本戦 U_gen の5要素は、周辺分布 0.9873212583、pMSE 0.9952725475、発症曲線 1.0、相関 0.7861794165、裾 0.9897985459。最小の相関が U_gen と全体 U を決める。

相関は9変数の36組の Spearman 相関の差を使う。平均差0.0694978700からの点は0.8262553249、差の90%点0.1710564668からの点は0.7861794165となり、後者が制約となった。差が大きい組は sex–HDL、sex–ALT、sex–BMI など。周辺分布を維持していても変数同士の関係の変化を新ルールが検出することを示す。これだけから加工アルゴリズムの名称や意図は断定しない。

U_valid は cens 0.9980934223、no_fab 1、no_dup 1、novel 1 の平均となる。

## 全チームの算定方法と限界

公式関数の実行可能 AST を比較し、U_spec、U_rare、その補助関数、既存の周辺分布・pMSE・時間・no_fab と最終 min 集約が同一であることを確認した。利用する既存設定も同一である。

本戦 U_gen は、予備戦 U_gen に相関と裾の条件を min で追加する。本戦 U_valid は、旧 mean(cens, no_fab) から mean(cens, no_fab, no_dup, novel) に変わり、cens も強化される。

### B 不要の厳密な上限

main cens は C 単独で求まる cens_rowwise 以下で、no_fab と novel はいずれも0以上1以下。従って、

U_main <= U_valid_main <= (cens_rowwise + no_dup + 2) / 4

となる。この上限は予備戦ファセット値を移し替えず、未知の B やその参照値にも依存しない。CSV の main_U_C_only_upper 列がこの上限である。17、19、29番の結論はこの上限から得た。

### 掲載された予備戦ファセットを条件とした上限

予備戦 U_valid と C 単独の旧 cens から no_fab = 2 * U_valid_pre - cens_pre を逆算できる。これと本戦 no_dup、新 cens の範囲、novel の範囲 [0,1] から U_valid_main の上下限を計算する。さらに、既存ファセットが同じ値を保つ条件で、

U_main <= min(U_gen_pre, U_spec_pre, U_rare_pre, U_valid_main_upper)

とできる。この条件付き上限が CSV の main_U_upper 列で、自チーム5だけは再計算した点に更新した。19番では0.9463724236まで絞れる。他チームの U_gen 追加指標は未知なので、U 全体に正の下限は設定できず0としている。下限0は悲観的な点予測ではない。

条件付き上限は、CSV 読み取りによる極微小な丸め差が既存ファセットへ与える変化を除いた、同じ数値データに対する数式上の比較である。予備戦のファイルを本戦パーサーで読むことで、列間相関以外にも浮動小数点の差が生じ得る。自チームでは U_spec に2.22e-16の差だけが生じ、他の共通ファセットは同値だった。他チームのこの影響まで厳密に再実行するには元 B が必要である。

## 数値読み取りで見つかった注意点

本戦公式 read_scoring_csv は float_precision="round_trip" を指定している。29番の余分な完全重複はこの正しい読み方で117件だが、pandas 既定読取では122件になる。全21 C についてこの差を調べ、重複数が変わるのは29番のみだった。主要結果は公式本戦パーサーに合わせた117件を使った。打切り判定は全21チームで一致した。

## 新規性に関する補助診断

予備戦 A_bg と13特徴が完全一致する C 行は18番で429件、29番で255件見つかった。novel が見る12特徴でも同じ件数だった。これは背景データと近すぎる行を減点する新指標と関係する観察だが、novel は C と B の最近傍距離分布を比較するため、この一致件数だけで novel や U の点数を代用しない。

## 検証済み事項

- 全21 C の SHA256 は公式予備戦結果表の掲載対象 C ハッシュと一致
- 本戦の公開スキーマ検証を全21 C が通過。1049行、13列、値域等のハード制約を確認
- 自チームの予備戦 U と全ファセットを最大誤差4.13e-11で再現
- 公式固定依存版30個が完全一致。採点依存コードも両公式タグの Git blob と一致
- U_gen の min、U_valid の mean、全体 U の min を別途照合
- C の行順を並べ替えても公式の正準化結果は同じ
- 原 C、B、A_bg、参照ファイルに変更なし。欠測チームの B を推定・復元して埋めない
- 入手可能な予備戦 truth は pool_id と希少ラベル、AIA 真値であり、他チーム全行の元 B を代替しない

## 成果物

- all22_utility_final.csv: 自チーム5の点再計算、他20チームの範囲、欠測21番を含む主要表
- all21_utility_final.csv: C がある21チームのみ
- all22_utility_bounds.csv / all21_utility_bounds.csv: 自チームを含め全チームを範囲として扱った表
- self5_paired_scores.csv: 自チームの4ファセットと U の差
- self5_prelim_breakdown.json / self5_main_breakdown.json: 個人行を含まない採点内訳
- self5_correlation_pair_differences.csv: 自チームの36変数ペアの相関変化
- prelim_background_overlap_diagnostic.csv: 背景との完全一致行数の診断
- csv_parser_audit.csv: default と公式本戦 round_trip の比較
- validation.json / main_schema_validation_all21.json: 検証結果
- all21_C_ledger_binding_audit.csv: 公式掲載 C とのバイト一致確認

## 再実行

この作業フォルダの元入力がある環境では compute_bounds.py、次に UTILITY_LOCKED=1 を設定した score_self5.py、finalize_results.py、verify_results.py の順に実行する。score_self5.py は sources/main-process-20260912_requirements-kit.txt の固定依存を deps_locked から読む。元 B と team5_utility_ref.json は入力であり、公開成果物には含めない。公開用の可搬 CLI は統合側で入力パスを指定する版を用意する。

## 公式出典

- [予備戦有用性コード](https://github.com/pwscup/pwscup2026-kit/blob/prelim-attack-20260825/starter/src/pwscup2026/scoring/utility.py)
- [本戦有用性コード](https://github.com/pwscup/pwscup2026-kit/blob/main-process-20260912/starter/src/pwscup2026/scoring/utility.py)
- [本戦採点設定](https://github.com/pwscup/pwscup2026-kit/blob/main-process-20260912/participant_data/scoring_config.yaml)
- [本戦 CSV パーサーとスキーマ](https://github.com/pwscup/pwscup2026-kit/blob/main-process-20260912/starter/src/pwscup2026/common/submission_schema.py)
- [本戦依存版の固定一覧](https://github.com/pwscup/pwscup2026-kit/blob/main-process-20260912/starter/requirements-kit.txt)
- [予備戦公式掲載結果](https://drive.google.com/file/d/1BLiwflR7FJ9tp5lCeSkP-kJga4ByHRnB/view)

作成日 2026-10-10 UTC
