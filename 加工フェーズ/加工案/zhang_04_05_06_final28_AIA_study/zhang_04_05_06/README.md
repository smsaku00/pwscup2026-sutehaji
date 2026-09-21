# 本番 team28 — AIA 重視の3方式（zhang_04〜06）

本番チーム28の匿名化結果です。提出用ZIPは `submissions/` の3個です。一式のZIPや説明PDFは提出形式ではありません。いずれも本番tokenを含むためチーム内で扱ってください。CodaBenchへのアップロードは実施していません。

| 提出ZIP | 数値と転帰の仕組み | 本番公式ローカル有用性 U |
| --- | --- | ---: |
| `zhang_04_copula_bgtime.zip` | 背景モデルのガウスコピュラ、背景由来の転帰時刻 | 0.921999 |
| `zhang_05_conditional_bgtime.zip` | 背景の近傍から列を順に条件付き生成、背景由来の転帰時刻 | 0.907491 |
| `zhang_06_mixture_btime.zip` | 背景の3成分混合モデル、Bの集計時刻プール | 0.906265 |

3案はパラメータ値だけの変更ではありません。通常層の合成方法をそれぞれ異なるモデルへ変更し、転帰時刻の元も変更しています。3案共通の部分は、公開の外れ値判定で層を分け、外れ値層は本番Bの順位分布を使うこと、発症とCox係数の集計値を調整することです。リスクは `zhang_06` が B の集計時刻プールを使う分、他2案と異なります。個人別のBの元の列の組合せや転帰をそのまま移してはいません。

## 予備選で分かったこと

9月9日公開の**予備選team05**は有用性0.982588、MIA成分0、AIA成分0.461538でした。攻撃20件を照合すると、個人のCに一度もアクセスせず全候補に10年と答える攻撃だけでも、生成元163人中143人、対照163人中135人に正解します。成功率差は4.91ポイント、ルールの比率 R は0.285714です。20件の希少群のAIA成分は0で、主な課題は全体のAIAです。

この予備選の本人・対照構成による差は、**どんなCを作っても0にできません**。予備選の正解・攻撃提出物は `code/analyze_prelim.py` で集計するためだけに使い、AIA対策の方向を決めるために分析しました。各モデルの特徴や数値の学習、パラメータの採点には使っていません。予備選のスコアを本番の保護点として扱いません。

## 本番で測ったこと

`results/utility_summary.csv` と各 `_utility.json` は、本番1049行の各提出用Cを固定された本番版公式ローカル採点器で測った値です。3案とも `U>=0.9` で、3個のZIPは公式の提出形式検証に合格しました。

`results/holdout950x/` は生成器を固定した後、Bを734/315人に分けた4回の参照MIA/AIA診断です。3案のMIA平均AUCは0.5288、0.5192、0.5153。対応対照で比べたAIA平均成功率差は−1.03、+1.03、−0.41ポイントです。ただし05案は9501分割で+5.49ポイント、04案の9501分割はMIA AUC0.5465です。**「AIAもMIAも痕跡ゼロ」は立証できていません。** 未補正・補正後の検定値は各分割のCSVに記録しました。非有意は同等や安全性の証明ではありません。

同じ9501〜9504分割と対応対照で旧方式をAIA評価すると平均差は+0.86ポイントです。予備選と本番に共通する「全員10年」攻撃の正解率差も別列に併記しました。本番の隠れたチャレンジ、非共有ブロック、正式な補正値φがないため、サーバ上の保護点は分かりません。

## 再現

本番team28の配布ZIPが手元に必要です。予備選team05のZIPはプログラムが拒否します。Python 3.12を想定します。依存版は同梱の本番キットと同一です。

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python code/reproduce.py --team-zip "/path/to/PWSCup2026_final_team28.zip" --out reproduced
```

Windowsの場合は `.venv/bin/python` を `.venv\Scripts\python.exe` に置き換えます。出力先 `reproduced/private_input/` に原データが展開されます。これを提出・共有しないでください。提出対象は `reproduced/zhang_04_...zip` など個別の3個です。

```sh
.venv/bin/python code/check_attacks.py --dist reproduced/private_input --out diagnostics
```

予備選の公開結果の集計を再現する場合は、追加で `requirements_prelim_analysis.txt` をインストールし、`code/analyze_prelim.py --help` の3入力を指定します。生成器が予備選のファイルを参照することはありません。

## 中身と注意

- `explanation.pdf`：具体例を含む説明、予備選の分析、本番の3方式・評価・限界。
- `code/generate_three.py`：3方式を実装。`code/shared_methods.py`：共通の公開層分け・Cox係数較正。
- `code/reproduce.py`：本番ZIPの識別、生成、公式採点、提出ZIP検証。
- `code/check_attacks.py` と `code/analyze_prelim.py`：評価用コード。生成には使いません。
- `official_scoring/`：固定版 `main-process-20260912` の採点ソース、設定、ライセンス。
- `results/`：有用性詳細、参照攻撃の分割ごとの値、予備選の集計、形式検証と再現ハッシュ。

`C.csv` は1049行13列で `record_id` を含みません。`submissions/` の各ZIP直下には `C.csv` と本番 `token.txt` のみです。行が元データと完全一致しないことは安全性を意味しません。差分プライバシーの保証もありません。

出典：本番team28の配布データ・schema、予備選公開結果表と公開された攻撃提出物・正解データ、本番版公式キット `https://github.com/pwscup/pwscup2026-kit/releases/tag/main-process-20260912`。
