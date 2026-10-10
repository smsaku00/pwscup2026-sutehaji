"""Append verified fixed-P01 comparison to the report while retaining prior results."""
from pathlib import Path
import pandas as pd
R=Path(__file__).resolve().parent
MARKER='## 改訂記録：固定P01比較の追加'
LABELS={
'global_train_best':'LOTO訓練選択・単一方式（実験内対照）','C_gate_fixed_k3_weight05':'Cのみ・3近傍/局所50%固定','C_gate_nested':'Cのみ・内側LOTO調整','oracle_infeasible':'対象正解で後出し最良（実行不能oracle）',
'ensemble_train_champion_mean':'訓練側8家族代表・順位平均','ensemble_train_champion_median':'訓練側8家族代表・順位中央値','ensemble_train_champion_max':'訓練側8家族代表・順位最大値','ensemble_train_top3_families_mean':'訓練側上位3家族代表・順位平均','ensemble_train_top5_families_mean':'訓練側上位5家族代表・順位平均','ensemble_train_champion_weighted':'訓練成績重み付き8家族代表',
'fixed_uniform_all_candidates':'固定・全候補順位平均','fixed_equal_family_background_mean':'固定・家族/BG等重み順位平均','fixed_family_consensus_median':'固定・8家族平均の中央値','fixed_family_consensus_max':'固定・8家族平均の最大値','fixed_P01_prelimBG_all20inference':'既存の固定P01・予備戦BG（採用判断の基準）'}
def corrected(text):
    if MARKER in text:return text
    revision='''## 改訂記録：固定P01比較の追加

2026-10-10 04:25 UTC改訂。初版では「各LOTO訓練foldで選び直す単一方式」を基準に改善を述べたが、既存の固定P01との比較が欠けていた。今回、固定P01を独立に再採点して追加し、**既存P01を置き換える根拠はない**と採用判断を明確にした。以前の予測、LOTO選択、平均値、対応差は変更していない。従来の `summary.csv` の差分列は引き続き訓練選択基準に対する値であり、固定P01との差は新しい `fixed_P01_baseline_comparison.csv` に分離した。

初版の集約ZIPは不変のまま保存し、本改訂の判断はこの修正版レポートと固定P01比較ファイルを優先する。

'''
    text=text.replace('## 結論（Team 1・2を除いた18対象を重視）',revision+'## 結論（Team 1・2を除いた18対象を重視）',1)
    lead='''- **既存の固定P01を置き換えない。** 固定P01（予備戦BG、全20対象で基礎予測、1/2除外の18対象で評価）は平均TPR **0.029775**、0.10超 **2/18**。今回の実行可能な全Cゲート/アンサンブルは、この平均を下回った。
- 8家族代表の順位平均は0.021801で、固定P01との差は**−0.007974、3勝10分5敗**、0.10超1/18。Team 13の0.12という局所改善はあるが、全体性能の改善とは言えない。
'''
    text=text.replace('## 結論（Team 1・2を除いた18対象を重視）\n\n','## 結論（Team 1・2を除いた18対象を重視）\n\n'+lead,1)
    text=text.replace('単一方式基準0.004444に対し','LOTO訓練選択の単一方式（実験内対照）0.004444に対し',1)
    text=text.replace('- アンサンブルは限定的に改善した。','- LOTO訓練選択の単一方式との比較に限れば、アンサンブルは改善した。',1)
    text=text.replace('基準差+0.017357、7勝10分1敗。','訓練選択基準との差+0.017357、7勝10分1敗。既存の固定P01との比較は上記のとおり悪化した。',1)
    text=text.replace('訓練対象で選ぶ単一方式（基準）','LOTO訓練選択・単一方式（実験内対照）')
    text=text.replace('| 基準との差 |','| 訓練選択基準との差 |')
    text=text.replace('- 推論時の基礎予測列は全20対象を保ったまま固定。','- 固定P01も今回の8家族も、基礎予測は全20対象の公開Cを使う同じ推論範囲。18対象LOTOは新しいセレクター/アンサンブルの学習と評価に適用し、基礎予測を18対象だけで作り直す意味ではない。固定P01にはこの新しいセレクター学習はない。\n- 推論時の基礎予測列は全20対象を保ったまま固定。',1)
    text=text.replace('学習済みmembershipモデル、policy、過去の対象別採用履歴を使う合成は除外した。','学習済みmembershipモデル、policy、過去の対象別採用履歴を使う合成は除外した。公開データ分類器272設定×7BGはこの8家族バンクの凍結後に完了したため、本版には含めない。',1)
    d=pd.read_csv(R/'fixed_P01_baseline_comparison.csv',float_precision='round_trip');d=d[d.view=='primary18_train_and_test_without1_2']
    lines=['### 4.1 採用判断の基準：既存の固定P01との比較','','P01と予備戦BGは、評価対象のTPRを見て選び直した組合せではなく、もともとの固定対照。今回バンクの `geometry/prelim/gen7_reference__county500__cross0.1`（candidate_index=14109）と**875×20の予測配列そのものが完全一致**する。異なる母集団や別尺度の対照ではない。','','| 方式 | 平均TPR | >0.10 | 固定P01との差 | 勝/分/敗 |','|---|---:|---:|---:|---:|']
    for _,v in d.iterrows():lines.append(f'| {LABELS[v.strategy]} | {v.mean_tpr:.6f} | {v.TPR_gt010}/18 | {v.delta_fixed_P01:+.6f} | {v.wins}/{v.ties}/{v.losses} |')
    lines += ['','8家族代表平均は固定P01比でTeam 3/6/13が改善、Team 9/10/12/17/24が低下、他10対象は同じ。特にTeam 10は0.115385→0.038462、Team 24は0.148148→0。Team 13の0→0.12と区別して判断する。','','### 4.2 以前のLOTO訓練選択基準との比較（数値は維持）','']
    text=text.replace('## 4. 主結果：1/2を訓練・評価の両方から除外\n\n','## 4. 主結果：1/2を訓練・評価の両方から除外\n\n'+'\n'.join(lines)+'\n',1)
    text=text.replace('8家族代表平均は、基準比で','8家族代表平均は、LOTO訓練選択基準比で',1)
    text=text.replace('3. 8家族代表平均は単一方式選択の不安定さを和らげた可能性がある。','3. 既存の固定P01は平均0.029775であり、今回の実行可能な全方式より高い。P01からの置き換えは推奨しない。8家族代表平均はLOTO単一方式選択の不安定さを和らげた可能性がある。',1)
    text=text.replace('主要ファイル：','固定P01の追加監査では、全20対象の公式採点誤差0、既存バンク内の同一設定との予測配列誤差0を確認した。新規の基礎モデル計算は行っていない。\n\n主要ファイル：',1)
    text=text.replace('- `summary.csv`：','- `fixed_P01_baseline_comparison.csv` / `fixed_P01_baseline_by_target.csv`：既存固定P01との差、全戦略・全対象。\n- `fixed_P01_baseline_audit.json` / `fixed_P01_bank_binding.json` / `audit_fixed_P01_baseline.py`：公式再採点、同一候補照合、再現コード。\n- `summary.csv`：',1)
    return text
if __name__=='__main__':
    p=R/'Cだけの方式選択とアンサンブル_検証結果.md';p.write_text(corrected(p.read_text()))
