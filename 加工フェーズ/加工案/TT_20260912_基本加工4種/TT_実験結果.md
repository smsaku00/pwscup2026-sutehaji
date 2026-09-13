# TT 基本加工4種類の提出実験

2026-09-12、チーム28の本戦B（1049行）を使用。指定版はmain-process-20260912。**4件ともCodaBenchの本戦加工フェーズで検証・採点が完了。**

## 公式結果

| 提出ZIP | ID | U | U_gen | U_spec | U_rare | U_valid | protection |
|---|---:|---:|---:|---:|---:|---:|---:|
| [C_equals_B.zip](https://www.codabench.org/competitions/17698/detailed_results/923855/) | 923855 | 0.9899 | 0.9994 | 1.0000 | 0.9899 | 1.0000 | 0.0000 |
| [TT-01-B_row_shuffle.zip](https://www.codabench.org/competitions/17698/detailed_results/923880/) | 923880 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| [TT-02-main_Abg_1NN.zip](https://www.codabench.org/competitions/17698/detailed_results/923882/) | 923882 | 0.5153 | 0.9366 | 0.5442 | 0.7200 | 0.5153 | 1.0000 |
| [TT-03-prelim_Abg_1NN.zip](https://www.codabench.org/competitions/17698/detailed_results/923886/) | 923886 | 0.7159 | 0.9401 | 0.7159 | 0.8566 | 0.8074 | 1.0000 |
| [TT-04-starterkit_Abg_1NN.zip](https://www.codabench.org/competitions/17698/detailed_results/923887/) | 923887 | 0.6188 | 0.9260 | 0.6188 | 0.7908 | 0.7654 | 0.9615 |

根拠は各提出のCodaBench詳細画面。今回の追加提出4件はすべてTTで開始する。元のC_equals_Bは既存提出であり再提出していない。掲載選択は元のC_equals_Bのまま変更していない。

## 結果の読み方

- **実測**：1-NNの中では予備戦A_bgがU=0.7159で最高。本戦A_bgは0.5153、キットA_bgは0.6188。
- **公式コード＋ローカル実測**：本戦A_bgの再利用はnovel=0。重複分193行によりno_dup=0.0801で、U_valid=0.5153が有用性を制限した。予備戦A_bg・キットA_bgの制約はU_spec（Coxの結論一致）。
- **実測**：行シャッフルはU=1.0000、protection=0。元の個票の集合をそのまま保っており、匿名化の改善ではない。
- **不確実性**：既存C_equals_BのU=0.9899との差は、元提出のCSVと入力Bをバイト比較できていないため原因未特定。ローカルでBそのままと行シャッフルは同点で、公式コードも行の内容による正準化を行う。したがって差を「行をシャッフルすれば上がる」と解釈しない。
- **不確実性**：protection=1でも安全性の証明ではない。この画面の内訳は固定参照MIAのみ。AIA、φ_iによる補正、全チーム攻撃によるAnon、最終総合得点は未算出。
- **評価の分離**：生成は指定BとA_bgのみ。本戦B_self/utility_refは生成後の自己採点だけに使用した。本戦A_bgは全案共通の採点基準。予備戦・キットA_bgを採点側に差し替えてはいない。

## 共通1-NN定義

連続8列（age/BMI/SBP/TG/HDL/ALT/FPG/time）の差を指定A_bgの標準偏差で割った二乗和と、カテゴリ5列（sex/prefecture/smoking/onset/death）の不一致数を加算。最小の1行を丸ごと採用。同点はA_bgの先頭、重複使用可、数値の丸めなし。3種類で同じ処理・設定を用いた。

## 成果物と確認

各番号の `.py`、`.md`、`C_TT-*.csv`、TTから始まる提出ZIPを同じフォルダに保存。ZIPはC.csvとtoken.txtのみ。4本は各々単独で実行可能。使用ライブラリ・所要時間・ピークメモリ・データの出典・全11項目は各説明書参照。

- `TT_実行記録.json`：入力・コード・出力・公式コードのSHA-256、2回再現性、予備戦Bへの差し替え、実測時間・メモリ。
- `TT_CodaBench公式結果.json` / `TT_比較結果.csv`：画面から転記した公式結果。空欄/nullは未算出。
- `TT_ローカル自己採点.json`：生成と分離した自己採点の全内訳。サーバとのU_rare差を区別して保存。
- `TT_自己採点.py`：入力先を明示して公式キットによる自己採点を再実行する補助コード。
- 各 `_形式検査.txt`：公式のZIP形式検査ログ。Bと同一行数のチェックは公式関数へ行数を明示して補完。

今回の結果は開発用の探索結果。最終比較用の独立データや複数攻撃での安全性評価は未実施。既存資料・配布データは変更していない。

自己採点の再実行例（各パスは利用環境の配置に合わせて指定）：

```sh
python "TT_自己採点.py" --kit "pwscup2026-kit-main_revised0912" --dist "PWSCup2026_final_team28_配布フォルダ" --config "pwscup2026-kit-main_revised0912/participant_data/scoring_config.yaml" --c "C_TT-02-本戦Abg_1NN.csv" --out "再自己採点.json"
```

Python依存は指定キットの `starter/requirements-kit.txt` で導入する。生成はNumPyのみだが、自己採点は同キットの採点依存が必要。
