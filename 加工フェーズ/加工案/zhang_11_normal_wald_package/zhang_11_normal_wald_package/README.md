# Zhang 11：Zhang 04 の属性と希少転帰を保持し、通常層の転帰だけ Wald 較正

対象は **PWS Cup 2026 本戦・コホート28** の `B.csv`、`A_bg.csv`、`schema.json`。提出用ファイルは `zhang_11_normal_wald.zip`（内部に `C.csv` と `token.txt`）です。予備戦チーム05用ではありません。

## なぜ Zhang 10 の protection が落ちたか

本戦版採点コードの加工フェーズ `protection` は、参照 MIA の `1 − max(TPR_all, TPR_rare)`。MIA の入力は `age, BMI, SBP, TG, HDL, ALT, FPG` の7列と `A_bg`、非公開の参照 B・共有真値です。`time`, `onset`, `death` はこの値に直接入りません。

したがって、Zhang 10 で採用した seo02 の転帰の移植や Wald 較正だけを直接の原因とは言えません。Zhang 10 は「Zhang 04 の通常層」と説明していましたが、実際の提出 C を比較すると、Zhang 04 と一致する7列のレコードは1,049行中 **0行**でした。通常層も乱数消費や実装が異なり、7列が変わりました。公開外れ値の判定数は `B=51, Zhang04=45, Zhang10=34`。Zhang 10 の希少行への seo02 由来属性の挿入と通常層を含む再生成により KDE の密度比が変わり、非公開参照コホートの特に希少群で MIA が大きく当たった可能性があります。**どの行が protection=0 を生んだかは、非公開 B_ref / pool 対応がなく断定できません。** 希少行数の減少だけで protection=0 が決まるわけでもありません。

## 今回の合成方法

1. Zhang 04 の Gaussian copula による全属性生成をコード・seed ごと同じにし、公開外れ値も通常層も7列を **行順まで完全一致**させる。
2. Zhang 04 と同じ、`A_bg` から独立にサンプルした発症・死亡時刻と独立な発症時計を出発点にする。元 B の患者別 `time/onset/death` 三点セットを C に移植しない。
3. 通常層に限り、seo02 に由来する Cox の **Wald 値（係数÷標準誤差）**も転帰較正の選択基準に含める。希少層の転帰は Zhang 04 のままにする。係数誤差70%、Wald誤差30%で選んだ較正反復を使う。

この方法は Zhang 04 と連続7列が全行で同一なので、**同じ公式採点データ・設定の固定 MIA なら protection は Zhang 04 と同じ値**になります。ユーザー報告の Zhang 04 の公式値が1なら、本案も同条件では1になる計算です。これは公式サーバーでの再提出結果ではなく、参照攻撃コードからの推論です。ほかの攻撃や最終 Anon の同一性は意味しません。

## 実測比較

| 指標 | Zhang 04 | Zhang 10 | 今回 Zhang 11 |
|---|---:|---:|---:|
| 公式ローカル有用性 U | 0.9220 | 0.9398 | **0.9220** |
| U_gen / U_spec / U_rare | .9365 / .9631 / .9220 | .9466 / .9447 / .9398 | **.9365 / .9599 / .9220** |
| 公開外れ値 C の人数 | 45 | 34 | **45** |
| 4分割 MIA TPR 全体平均 | .0170 | 未測定 | **.0170** |
| 4分割 MIA TPR 希少平均 | .0882 | 未測定 | **.0882** |
| 4分割 MIA の最悪 split | .1176 | 未測定 | **.1176** |
| 模擬 AIA main 10パネル R_all 最大 | .0612 | 未測定 | **.0612** |
| 模擬 AIA main 10パネル R_all 中央値 | 0 | 未測定 | **0** |

Zhang 04 と本案の MIA は連続7列が一致するため、4分割の値も完全一致しました。模擬 AIA main は、各パネルで対照を外した `A_bg` から C を**作り直し**、10パネルの R_all が Zhang 04 と全パネルで同じでした。希少対象の参考差も同じ（最大0）。模擬 AIA の希少対照に真の希少ラベルはなく、公式の AIA rare は再現できません。

Wald を全行に適用した別案は模擬 AIA main の最大が .0816、希少層だけに適用した別案も .0816 で、どちらも Zhang 04 の .0612 より悪化したため採用していません。本案は有用性を Zhang 04 から上げる結果ではありません。プライバシーの速報値を維持しながら、通常層の転帰に Wald 較正を取り入れる目的の案です。

## ファイルと再現

- `zhang_11_normal_wald.zip`：提出形式。`C.csv` と元チーム配布の `token.txt` を収録。
- `code/generate_normal_wald.py`：今回の入口。`generate.py`, `generate_three.py`, `shared_methods.py` が同じフォルダに必要。
- `code/eval_wrapper.py`：同梱の再利用評価器に渡すラッパ。`B_self.csv` の `is_rare` は**評価用の分割にだけ**使い、生成器には渡さない。
- `results/comparison.csv`, `results/mia_holdout.csv`, `results/aia_main.csv`：比較と評価の数字。原 B と A_bg、模擬 challenge の個票は収録しない。

再生成には `numpy`, `pandas`, `scipy`, `scikit-learn`, `lifelines` を用意し、展開したフォルダから実行：

```bash
python code/generate_normal_wald.py \
  --b /path/to/final_team28/B.csv \
  --a-bg /path/to/final_team28/A_bg.csv \
  --schema /path/to/final_team28/schema.json \
  --out reproduced/C.csv
```

`C.csv` の SHA-256：`fd8a01d256d3c05f6a7dfd7689ebf565e7613219098684106429f690db9229a9`。同じ本戦入力とライブラリ環境で再生成した結果と一致しました。提出 ZIP は本戦版形式検証に合格しています。公式 `protection` をローカルで直接再現するには非公開の B_ref と pool 真値が必要です。
