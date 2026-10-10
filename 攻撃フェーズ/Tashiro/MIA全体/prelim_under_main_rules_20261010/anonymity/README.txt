予備戦の加工データに本戦ルールを適用する場合：匿名性の検証
確認日：2026-10-10

結論
予備戦の攻撃提出、真値、希少フラグ、攻撃参加者をすべて固定するなら、MIA成分は予備戦の公表値と同じです。本戦ではAIA成分からコホートごとの背景由来の水準phi_iを引きます。この反実仮想に必要なのは「予備戦の世界・コホート・標的/対照構成」に対応するphi_iであり、実際の本戦の別コホートのphi_iではありません。この値は今回取得した公式予備戦結果表に含まれていないため、根拠のない0や他コホートの値を代入していません。

確認した変更点
・MIAのFPR=0.01、同値をまとめるROCの線形補間、全体/希少の大きい方、攻撃者間平均、20,000回置換の95パーセンタイルによる偶然補正、K=12は同じ。
・AIAの正解判定±0.5年、希少側SEデッドゾーンz=1.64、全体/希少の大きい方、攻撃者間最大も同じ。
・本戦の追加は、AIAの攻撃者間最大を取った後に一度だけclip(AIA_max−phi_i,0,1)。
・phi_iは、Cの代わりにそのコホートの個人を含まない背景データから1,049行を取り、参照AIAを当てたRの1,000回平均。Cには依存しません。Rが[0,1]にclipされるため0≤phi_i≤1です。
・本戦攻撃フェーズの公式リリースには、ルールブックはmain-process-20260912から変更なしと明記されています。

計算式
M_i = 公表された予備戦MIA成分
A_i = 公表された予備戦AIA成分
A_main_i(phi_i) = clip(A_i−phi_i,0,1)
Anon_main_i(phi_i) = 1−max(M_i,A_main_i(phi_i))
したがって予備戦Anon_i ≤ Anon_main_i ≤ 1−M_i。

この上下限は、未取得のphi_iに対する論理的な範囲です。信頼区間や実際のphi_iの推定値ではなく、上下端が実際に達成されるという主張でもありません。
9チーム（1,2,3,4,8,10,13,17,24）はM_i≥A_iなので、phi_iが分からなくても匿名性は一点に確定します。
12チーム（5,6,7,9,12,18,19,20,22,23,26,29）はphi_iに依存します。
公式表のチーム21は得点欄が空欄なので、0とせず欠損のまま残しています。

有用性と結合する式
本戦有用性U_main_iが正確に分かる場合：
Score_i(phi_i) = min(U_main_i,Anon_main_i(phi_i))
Score_i ∈ [min(U_main_i,Anon_pre_i), min(U_main_i,1−M_i)]
U_main_i自体にL_i≤U_main_i≤H_iの範囲しかない場合：
Score_i ∈ [min(L_i,Anon_pre_i), min(H_i,1−M_i)]
これは当該ラウンドの得点です。大会の予備戦0.1・本戦0.9の合算を計算したものではありません。
チーム1と2はMIA=1のため、U_mainやphi_iにかかわらず、この凍結攻撃条件下の総合得点は0です。

凍結条件の重要性
今回のGMMなど新しい攻撃出力は、この再採点に追加していません。本戦の参加者、候補、攻撃法、提出確信度に置き換えると、平均TPRも偶然補正の基準も変わりうるため、公表予備戦MIAをそのまま使えません。実際の本戦の匿名性や総合得点を予測したものではありません。

検証内容
・公式予備戦XLSXは22行、うち21チームに得点あり。
・420件のMIAの攻撃者別t_j、21チームのbeta_raw・偶然補正・MIA・Anon・総合得点の関係を再検証。
・AIAは公式行列の非欠損値の最大と公表成分が一致。標的23/攻撃者4は公表行列が空欄で、標的23のみ19非欠損値（その他20）です。空欄を0に置換していません。
・予備戦攻撃版、本戦加工版、本戦攻撃版のattack_score.pyは、Git blob SHA d7fc73a993c66f2ce4b831053529c2eb4b46eac8で完全一致。
・公式の20,000回置換によるbeta_chanceを再利用。提出確信度から20,000回の再実行はしていません。
・範囲端点、単調性、上限到達に必要なphi_i=max(0,A_i−M_i)を確認。
・元の公式XLSXを変更していないことをSHA-256で検証。

ファイル
main_rule_anonymity_bounds_all22.csv：公式22行（有効21、空欄1）
main_rule_anonymity_bounds_previous20.csv：直前のGMM比較と同じ20チーム
main_rule_anonymity_bounds_all22.json：同内容の機械可読データ
reproduce_anonymity_bounds.py：標準ライブラリだけで再実行できる抽出・検証・計算スクリプト
validation.json：検証結果と計算式
official_source_snapshot.json：確認した公式コードと現行設定の本文・Git blob・URL

主要出典
公式予備戦結果表：
https://drive.google.com/file/d/1BLiwflR7FJ9tp5lCeSkP-kJga4ByHRnB/view
総括!H:J = beta_raw, beta_chance, beta、K=MIA、L=AIA、M=Anon、N=当該ラウンド総合。
SHA-256：8d6380feafa50a850b28aa50eb4b04cd0ffe132ca15904dc0c84b85fd8ff6508

本戦正本ルールブック（§6、pp.13・19〜21）：
https://github.com/pwscup/pwscup2026-kit/blob/3dd71bd3436f2abfdbb64da97e860f22729ea054/rulebook/PWS_Cup2026_ルールブック_main-process-20260912.pdf
予備戦攻撃版（§6.2、pp.17〜19）：
https://github.com/pwscup/pwscup2026-kit/blob/4477277e18a8e3979ec6015838fb672744fb64b6/rulebook/PWS_Cup2026_ルールブック_prelim-attack-20260825.pdf
本戦攻撃版でルール変更なしの公式リリース：
https://github.com/pwscup/pwscup2026-kit/releases/tag/main-attack-20260928
現在の公開設定（K=12、FPR=.01、AIA±.5、希少z=1.64）：
https://github.com/pwscup/pwscup2026-kit/blob/main-attack-20260928/participant_data/scoring_config.yaml
各攻撃の採点コード：
https://github.com/pwscup/pwscup2026-kit/blob/main-attack-20260928/starter/src/pwscup2026/scoring/attack_score.py

公開用CLIでの再現
python reproduce_anonymity_bounds.py --official-workbook /path/公開用結果表_20260909.xlsx --out-dir /path/new_output
公式xlsxは上記出典から別途取得します。official_source_snapshot.jsonはスクリプトと同じフォルダから読み込みます。標準ライブラリのみで実行可能です。公開用CLIを再実行し、同梱CSVとJSONの一致を確認しました。
