# PWS Cup 2026 本戦・初回攻撃提出

このフォルダには、再現コード `attack_aia_gate.py`、この説明書、およびそのまま提出できる `attack_submission.zip` がある。提出するのは **ZIPだけ**。ZIPの直下には `F_aia.csv`、`F_mia.csv`、`token.txt` の3ファイルが入っている。自チームは28番。

## AIAで答える値

各チームの `aia_challenge_<k>.csv` の326行について、次の計算を行う。

1. 相手の `C_<k>.csv` と、共通の `A_bg.csv` から固定抽出した1,049行を用意する。乱数の種は `20261004`。背景の行数をCと同じにそろえる。
2. それぞれの表から、課題行と性別・都道府県が一致する行を探す。該当行がなければ性別が一致する行、それもなければ全行を使う。
3. `age`、`SBP`、`TG`、`HDL`、`ALT` の差を各列の背景データの標準偏差で割り、5列の差の二乗平均の平方根で距離を出す。最も近い行の `time` がC側の候補値。同距離の行が複数あれば、それらの`time`の平均。
4. **C側の距離 < 背景側の距離 × 0.25** のときだけC側の候補値を回答する。それ以外は **9.5年** と答える。

`0.25` は「Cが背景より十分近い」を数値にした初回用の保守的な設定。データから公式得点の最大値を探索して決めた値ではない。距離が小さいこと自体は、真の`time`や標的の所属を保証しない。特に9.5年は真値が9.0～10.0年のとき、公式の±0.5年判定で正解となるが、対照も同じように正解するとAIA得点は増えない。

## MIAと得点について

攻撃ZIPにはMIAの予測 `F_mia.csv` も必須なので、配布キットの参照方法と同じ7列（`age`、`BMI`、`SBP`、`TG`、`HDL`、`ALT`、`FPG`）のKDE密度比を用いて作成した。これは今回のAIA条件とは独立した、初回提出用のMIAである。以前作ったMIA案を使う場合は `F_mia.csv` を差し替えてZIPを作り直す必要がある。

自チーム28の課題では、標的163人の真の`time`は手元の `B_self.csv` から確認できるが、対照163人の真の値は分からない。他チームの正解も公開されていない。したがって、このZIPの公式AIA得点は手元では計算できない。模擬課題で距離比0.25と0.5を比べても、どちらが常に優れるとは確認できなかった。公式のφも、この予測を採点する際に差し引かれるため、背景との距離比較だけでは得点を予言できない。

## 再現方法

Python 3.10以降と `numpy`、`pandas`、`scipy` が必要。配布された2つのZIPを手元に置き、次を実行する。

```bash
python -m pip install numpy pandas scipy
python attack_aia_gate.py \
  --attack-zip PWSCup2026_final_attack.zip \
  --team-zip 'PWSCup2026_final_team28_ステテコは恥だが役に立つ.zip' \
  --out-dir generated
```

`generated/attack_submission.zip` が提出用。実行時に `generated/F_aia.csv`、`generated/F_mia.csv`、`generated/token.txt` も作る。背景の抽出条件を変えるには `--seed`、距離比を変えるには `--ratio` を指定する。**同梱の提出ZIPは既定値のまま生成**したもの。

トークンはチームの認証に使うため、提出ZIPと出力された `token.txt` を公開リポジトリへ置かない。再現コードとこの説明書にはトークンを埋め込んでいない。

## 使用した資料

- 本戦配布物 `PWSCup2026_final_attack.zip`（23チームのCとAIA課題）
- チーム28配布物 `PWSCup2026_final_team28_ステテコは恥だが役に立つ.zip`（背景、B、トークン）
- 配布キット `pwscup2026-kit-main(3).zip` の `starter/src/pwscup2026/attack/aia.py`、`mia.py`、`starter/src/pwscup2026/kit/validate.py`
- `PWS Cup 2026 ルールブック main-process-20260912` のAIA・提出形式・採点の節
