# 有用性Uの本戦ルール再評価

結論・計算式・全チームの限界は **[utility_report.md](utility_report.md)** を参照。

- 自チームの完全再計算: [5指標の比較](self5_paired_scores.csv) / [本戦式内訳](self5_main_breakdown.json)
- 21提出＋1欠測: [全22行の統合表](all22_utility_final.csv)
- 他チームの2種類の上限: `main_U_C_only_upper`はCだけから成立する上限、`main_U_upper`は旧ファセット転用条件付き。自チームだけ後者は実計算値。
- [入力CSVの読み取り監査](csv_parser_audit.csv): 本戦公式round_tripで神ぼ大の余分な重複は117行。既定パーサーの122行は本戦評価に使わない。
- [検証結果](validation.json): 公式依存30版一致、旧公式点の再現、全21Cの公式ハッシュ一致を含む。

## 公開用CLI

元の個人単位B/C/A_bg、チーム別採点参照JSONは同梱しない。自チームの再計算には保有する原本を指定する。

### 依存版

Python 3.12で、隔離環境へ公式固定版をインストールする。

```bash
python -m venv /path/venv
/path/venv/bin/pip install -r sources/main-process-20260912_requirements-kit.txt
```

### 全チームのC単独項目・条件付き境界

```bash
/path/venv/bin/python reproduce_utility_bounds.py --c-dir /path/PWSCup2026_prelim_attack --out-dir /path/new_output
```

`C_input_manifest.json`のSHA-256で入力Cを確認し、同梱の公式ソース・設定・旧U台帳を使う。Cがないbot035は欠測のまま。

### 自チームの両ルール再計算

```bash
/path/venv/bin/python reproduce_self5.py --b /path/B_self.csv --c /path/C_5.csv --background /path/A_bg.csv --reference /path/team5_utility_ref.json --out-dir /path/new_output
```

公式コピー`official_modules`を使い、予備戦は既定CSV読込、本戦は公式read_scoring_csvを使う。依存版を別ディレクトリへインストール済みなら`--dependency-dir /path/deps`を指定できる。新しい出力へ結果を保存し、元入力を変更しない。

### 集計の再生成

`finalize_results.py`は同じフォルダの境界表と自チーム内訳を結合する。再計算した新出力先へ同スクリプトをコピーして`python finalize_results.py`を実行すれば`all22_utility_final.csv`と`all21_utility_final.csv`が得られる。

公開用CLIを元入力と公式固定依存版で再実行し、同梱境界CSV・自チーム5指標・両内訳JSONのバイト一致を検証した。同じ公式Dockerイメージそのものを使ったという主張ではなく、公式採点の受領結果でもない。
