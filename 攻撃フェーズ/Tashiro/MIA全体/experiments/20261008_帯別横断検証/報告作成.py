"""今回の集計成果物から日本語報告を作成する。元実験は変更しない。"""
from pathlib import Path
import csv
import json
from collections import defaultdict
import argparse

def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def run(d):
    record=json.loads((d/'集計記録.json').read_text())
    configs=read(d/'方式条件一覧.csv'); allrows=read(d/'全方式評価.csv'); best=read(d/'チーム別最高結果.csv')
    lookup={r['condition_id']:r for r in configs}
    bymethod=defaultdict(dict)
    for r in allrows:
        cfg=lookup[r['condition_id']]
        if cfg['evaluation_class']=='C':bymethod[cfg['method'],cfg['condition']][int(r['team'])]=float(r['tpr'])
    p01=bymethod['Q_prefecture_sex_shrink500_cross01','full']
    labels=[('P01（比較基準）','Q_prefecture_sex_shrink500_cross01'),('① 県性別200を90%＋性別年齢100を10%','mix_county200_age100_w0.1'),
            ('② 県性別100','Q_prefecture_sex_shrink100_cross01'),('③ 最近傍距離比k=1','global_nn_k1')]
    table=[]
    band=lambda x:1+sum(x>a for a in (.25,.5,.75))
    for label,m in labels:
        vals=bymethod[m,'full']
        table.append(f"| {label} | {sum(v>.25 for v in vals.values())} | {sum(band(vals[t])>band(p01[t]) for t in vals)} | {sum(vals.values())/len(vals):.9f} | {sum(vals[t]>p01[t]+1e-12 for t in vals)}/{sum(vals[t]<p01[t]-1e-12 for t in vals)} |")
    tabletext='\n'.join(table)
    attained='・'.join(str(t) for t in record['band2_teams']) or 'なし'
    new='・'.join(str(t) for t in record['new_band2_teams']) or 'なし'
    besttable='\n'.join(f"| {r['team']} | {float(r['tpr']):.9f} | {r['band']} | {r['method']} | {r['condition_id']} |" for r in best)
    perteam=[]
    for t in record['band2_teams']:
        rows=[r for r in allrows if int(r['team'])==t]
        good=[r for r in rows if float(r['tpr'])>.25]
        perteam.append(f"Team {t}は、評価した{len(rows):,}方式条件中{len(good):,}条件で帯2以上。最小TPRは{min(float(r['tpr']) for r in rows):.9f}、最高は{max(float(r['tpr']) for r in rows):.9f}であり、すべての方式で達成するわけではない。")
    notes=[]
    for x in record['notes']:
        if isinstance(x,str):notes.append('- '+x)
        else:notes.append('- '+' / '.join(str(v) for v in x.values()))
    audit=json.loads((d/'公式採点監査.json').read_text())
    summary=f'''# MIA全体：帯2以上の横断検証（2026-10-08）

1. **帯2以上の達成チームはTeam {attained}。** Team5は攻撃対象外、Team12は最終報告から除外した。
2. **Team1・2以外の新しい達成例は{new}。** 閾値・対象集合を変更して成功例を作っていない。
3. P01、Gen-LRA7/9、属性・距離の変更、近傍距離比、較正・参照・学習の多数の既存設定でTeam1・2の達成を確認した。全部の設定を[帯2以上_全方式.csv](帯2以上_全方式.csv)へ保存した。P01は0.56/0.52で帯3、最近傍距離比k=1は0.80/0.92で帯4。旧方式も含めた最高値は下表を参照。
4. **本戦への性能保証はない。** 公開属性だけから実行できる方式Cと、評価対象ラベルを保留した学習Bを分けた。予備戦正解を反復使用した開発・事後選択の影響は残る。P01の本戦第3回実使用は既存公式履歴、今回の表は予備戦公開正解による評価である。
5. **第4回の全体MIAは、変更するなら①較正CDFの90%/10%混合を小さな検証候補とする。** 新しい帯2対象の増加は0。採用済みP01を上回る保証はなく、今ある成功方式のチーム別寄せ集めは勧めない。MIA希少の固定5bitを継続し、AIAの改良検証も進める既定方針を維持する。

## 検証範囲

パラメータを含む方式名 **{record['parameterized_methods']:,}**、方式×背景・分割等の条件 **{record['method_conditions']:,}**、Team12除外後 **{record['target_rows']:,}対象行**を整理した。これは独立なアルゴリズム{record['parameterized_methods']:,}種という意味ではない。seed・view・学習器・係数などを含む実装済み設定の数である。全19対象の欠測を0で補完しない。各条件の実評価チームは[方式条件一覧.csv](方式条件一覧.csv)に記載した。

[全方式評価.csv](全方式評価.csv)の`condition_id`を方式条件一覧へ、`source_ids`を[結果保存元.csv](結果保存元.csv)へ結合すると、属性・係数・背景・分割・保存元・検算状態を追える。[集計記録.json](集計記録.json)に件数と除外理由を保存した。

## 第4回の比較候補（全背景・19対象、FPR=1%）

| 方式 | 帯2以上対象数 | P01から帯上昇 | 平均TPR | P01より改善/悪化対象 |
|---|---:|---:|---:|---:|
{tabletext}

- **①混合：** P01の県×性別500を、県×性別200のCDF90%＋性別×年齢100のCDF10%へ変更。Gen-LRA7とcross0.1を維持。全背景の上積みはTeam7の0.04→0.08（1人）だけ。背景抽出6条件を19対象で見ると3改善・2悪化・1同点。0.5%FPRで悪化する既存結果もあり、尾の安定改善とは言えない。第3回に対する小さな変更仮説を試す価値がある、という限定した推奨。
- **②100：** 県×性別500→100だけを変更。全背景は①と同じだが、抽出6条件では2改善・3悪化・1同点（19対象平均）。局所群への依存が増す。①より優先する根拠は弱い。
- **③k=1：** Gen-LRA＋較正を、公開AとCへの最近傍距離の比へ変更。Team1・2は帯3→4だが、新しい帯2チームは増えず、Team10・17・18・24・7・9はP01より悪化。Team24は4/27→1/27。全背景以外で同じ全処理の安定性は未検証。平均と既存高得点対象に偏るため、今回の目的では採用優先度を下げる。

本戦で見えるのは主に平均TPRと相手別の帯。予備戦の番号に基づく方式割当てや、帯から正確なTPRを復元した選択は行えない。第4回を過去最良の統合に置き換えず、第8回の統合予定と区別する。提出CSV・ZIPは作成せず、CodaBenchへ提出していない。

## 達成と未達成

{' '.join(perteam)}

P01とk=1を元コード・元入力から全20対象で再現し、保存値との誤差は全40行で0。Team1・2の達成はFPR0.5%/2%でも維持し、FPR1%のROC補間の両端で会員検出数が同じだった。[代表方式再現監査](代表方式再現監査.json)。P01は14/25人・13/25人、k=1は20/25人・23/25人の検出に相当する。

未達成17チームは全員、最高TPRが0.10超〜0.20以下。0.20超の未達成チームと、全方式0.10以下のチームはない。

以下は**公開正解を見て選んだ事後oracleの最高値**。本戦で事前に選べる方式や未知対象の性能を示さない。完全な条件・同率候補は[チーム別最高結果.csv](チーム別最高結果.csv)にある。

| Team | 過去・今回再評価の最高TPR | 帯 | 方式 | 条件ID |
|---|---:|---:|---|---|
{besttable}

## 採点・再現・未確認

公式の改訂予備戦/本戦採点実装で、同点をまとめたROCとFPR1%の線形補間を確認。境界は`0.25 < TPR`で帯2、0.25は帯1。157順位キャッシュ・12,236方式条件・244,720対象行を独立再採点し、保存TPRとの差は全件0。[公式採点監査](公式採点監査.json)。旧初版の同点処理を参照しない。

未確認・除外範囲、旧方式の今回再評価、学習・トランスダクティブ処理の留保は[検証詳細.md](検証詳細.md)へ集約した。学習最高値や公開非会員CDF診断だけを本戦の検出力保証に使わない。GitHubだけでは個人入力と過去依存コードがそろわず、完全再実行にはローカル作業ルートが必要。
'''
    details=f'''# 検証詳細

## 前提と評価集合

対象は予備戦の自Team5のB1049候補と、Team5以外の20対象。Team12を結果表から除き19対象として報告する。Team12の学習ラベル/公開Cを機械的に除去しない。対象一覧: {', '.join(map(str,record['report_teams']))}。

所属真値は`予備戦/配布データ/結果/正解データ/pool_ids_5.csv`のrecord_id順と各対象pool_idの集合で突合し、メモリ内だけで使用した。会員数25〜27、非会員1022〜1024。FPR1%は10.22〜10.24人の偽陽性相当なのでROC補間を含み、`tp_equivalent`は整数の検出数に限らない。

帯は生値で`<=.25→1, <=.5→2, <=.75→3, それより上→4`。隣接FPRは安定性診断であり、成功例の判定は常に.01。旧本戦22対象の公開帯を、予備戦の正確TPRへ変換していない。

## 調査した保存先とファイル

- GitHub `smsaku00/pwscup2026-sutehaji`、接続アカウントTaisei-Tashiro、探索スナップショット1617fc1e5a6de90c1637d5db1040d29786863344。MIA全体のexperiments/20261007、20261008、proposal、docs、analysisを確認。[GitHub探索記録.json](GitHub探索記録.json)にblob識別子・サイズを列挙。
- ローカル`本戦/攻撃フェーズ/Tashiro/20261007_MIA全体改善の検討/`。対象別CSVを基準に、参照拡張、追加最終版、順位校正版を優先。GitHubの10/7掲載対象別CSVはローカル原本とGit blob SHA一致。README/説明は編集差があるが結果CSVの差はない。
- `本戦/攻撃フェーズ/Tashiro/20261008_MIA全体_較正の広範比較/`。最初の65と追加31（重複7を含む）から89独立設定、7背景条件。4種FPRのうち.01のみ採用。
- `本戦/攻撃フェーズ/Tashiro/20261008_MIA全体_朝9時までの方式探索/`。204対象別TPR CSV、保存NPZ、実行記録、方式一覧、固定条件、失敗記録、技術引継ぎ複製を確認。引継ぎZIPの複製表は重複採用しない。
- `本戦/攻撃フェーズ/Tashiro/20261005_MIA初回_GenLRA比較/`、`予備戦/結果分析/集計/`、`予備戦/攻撃フェーズ/提出記録/`、分析21/22、旧アーカイブの移動前work/過去コード、GitHub作業コピー。過去の相対参照・入力の移動先を正規配布ルートと照合した。
- iCloud側のPWS作業ルートも調査し、MIA全体の追加原本は見つからなかった（現在残るのは加工候補の比較Excel）。他担当領域の変更は行わない。

集計した全保存ファイルは[結果保存元.csv](結果保存元.csv)。同じ実験のローカル・GitHub作業コピー・移管コピーは重複を除く。保存元がコード#設定の場合、その設定を今回元コードから再評価した結果であり、過去TPRが保存されていたという意味ではない。

## 方式・条件と重複整理

全方式名は[方式条件一覧.csv](方式条件一覧.csv)に列挙。Gen-LRA7/9/10、性別分割、1/3/10近傍、属性除去/部分集合、尺度・相関/whitening、他対象補正（自対象を中央値に含む型と除く型）、公開Q/A較正、群別・階層・可変・混合CDF、背景bootstrap/抽出、参照拡充と重み、点KDE/距離比/局所密度/直接密度比、混合分布、Gaussian、学習2/5/24特徴・47背景view・中心10view、対象4分割/LOO等を含む。旧ライブラリと旧cross-C/density/redteamも、利用可能な既定設定を再評価した。

方式名には係数・view・学習器が含まれるので、{record['parameterized_methods']:,}方式名はアルゴリズムの種数ではない。{record['method_conditions']:,}条件は方式名×背景・分割・評価区分等。方式ごとのseedや分割を結果を見て統合しない。異なるアルゴリズムの同じTPRは別方式のまま残す。原理とコードが同じ別名だけを統合する。学習rank1は単調対照で出力順位がGen7に一致しても、学習Bの実行履歴をCと混ぜない。

取り込む前の対象行は{record['raw_target_rows']:,}、Team12除外直後{record['after_team12_exclusion_before_dedup']:,}、重複整理後{record['target_rows']:,}。評価区分別の対象行は{record['evaluation_class_counts']}。複数のseed/viewにまたがる最高値と、事前に固定して実行できる1方式の値を区別する。

## 再計算と公式実装との照合

1. `oct07_results.py`で保存対象別TPRの読取りと別名整理。旧撤回学習・AIA補助proxyを除外。`oct08_results.py`で204CSVと個別TPR欠落のprelim順位を再採点。`historical_results.py`で旧保存個別スコアを現公式へ再採点し、元純関数の既定設定を必要範囲で再現。
2. `official_audit.py`は改訂予備戦/本戦attack_score.pyを独立名前空間で読む。2版はSHA256同一。初版は同点塊をまとめない差があり不採用。247合成入力を独立ROC・sklearnと照合し、帯境界とnextafterも確認。
3. 157キャッシュ×12,236方式条件×20対象=244,720対象行について、元record_id順と875通常候補マスクを確認後、同じ信頼度変換で公式再採点。保存表との差は全件0。Team12を含む検算と、除外する報告を分ける。
4. P01と距離比k1は保存順位を使わず元純関数・入力から全20対象で再現し、40対象行の最大誤差0。元コード/入力SHA一致、旧実行Pythonは記録なしのため未特定。現在はルート.venvだけを使用。
5. 後半の離散依存木は個別スコア保存なし。公開予測だけを元計算からメモリ再現し、保存上位12の人数・入替・Spearman60行を誤差0で照合して公式採点した。後半公開Gaussian等も順位キャッシュから個別TPRを補充。個人配列を成果物に出していない。

詳細は[公式採点監査.json](公式採点監査.json)、[代表方式再現監査.json](代表方式再現監査.json)、旧方式集計cacheのsource_hashesを参照。既存CSV/JSON・入力を上書きせず、生成先は今回の新フォルダのみ。ルートREADMEにある予備戦A=10286行の説明は現入力と異なり、今回確認した正規A_bgは5143行である。過去記録を一括書換えせず現ファイルhash・実人数で評価した。

## 達成チームと未達成チーム

{' '.join(perteam)}

帯2以上の全設定と属性/係数/背景/P01との差/保存元は[帯2以上_全方式.csv](帯2以上_全方式.csv)。P01が帯1だったチームの新規達成一覧は{new}。未達成17チームは全員最高TPRが0.10超〜0.20以下で、0.20超の未達成チームと全方式0.10以下のチームはない。未達成の各区分（>0.20、0.10〜0.20、<=0.10）は[チーム別最高結果.csv](チーム別最高結果.csv)のproximity列。最高値が同じ全条件IDも保存した。

P01は7背景条件すべてでTeam1/2が帯2以上だが、TPRや帯3/2は動く。1%での少人数検出は1人で約0.037〜0.04変わる。背景抽出やFPR隣接点での安定性は、未知加工チームの性能保証ではない。

## 学習・事後選択の留保

- A欄には評価対象の正解を利用した参考結果を分類した。今回の57行は、公開真値を見た後の旧Gen-LRA7/9/10の対象別固定差替3条件×19対象であり、学習器の学習内fitを再評価した結果ではない。selection列でoracle組込参考として区別し、本戦推奨の根拠に用いない。
- Bは評価対象ラベルを保留した人物5分割×対象4分割またはLOO。all20_reproductionは人物/対象を保留したBであり、all20が学習内評価を意味するわけではない。source5/6の滑らかな補正も元OOF学習のBを受け継ぐ。
- Cは真の所属ラベルなしで予測計算する方式。C対Aの公開データ分類器やGMM内BICは真の会員ラベル学習とは異なる。Cでもパラメータ開発に公開正解を反復使用した適応的選択の偏りは残る。
- 対象公開C、他対象C/Q、B候補属性と全候補順位を参照するトランスダクティブ処理を明記。ラベル保留だけでは対象C未観測の外部検証にならない。背景47viewは同じ人物を複数表示し、独立標本47集団ではない。
- チーム別最高表は全方式から公開真値を見た事後oracle。学習内値、CV値、ラベル不使用値をまとめて「本戦で達成できる方式」へ変換しない。方式選択を含む独立外側評価はない。

## 未確認・除外・探索メモ

{chr(10).join(notes)}

本戦真値は未知なので本戦TPRの新規検証はできない。元inputs/旧依存がGitHubに全部掲載されていないため、この成果物だけで独立再実行できるとは主張しない。元コードから再計算できた範囲と、保存値だけ確認した範囲はverification列で区別した。平均だけの未確認実験から個別TPRを推測していない。

## 再現方法

ローカルPWSルートの.venvで次を実行する。既存成果物を上書きしないため、出力先には新しいフォルダを指定する。

```sh
<PWSルート>/.venv/bin/python -B 集計.py --root <PWSルート> --output-dir <新しい保存先>
<PWSルート>/.venv/bin/python -B official_audit.py --root <PWSルート> --output-dir <別の新しい監査保存先>
```

旧mainや旧Python固定パスは実行せず、子プロセスはsys.executableを使用する。依存関係はroot requirements.txtを変更していない。historical_cacheは対象別集計だけで、各元source hash一致時のみ読取り、変化時は既存定義を再計算する。今回の成果物に個人別正解・非公開入力・大会回答CSVは複製していない。
'''
    for name,text in [('README.md',summary),('検証詳細.md',details)]:
        if (d/name).exists():raise FileExistsError(name)
        (d/name).write_text(text,encoding='utf-8')
    print(json.dumps({'README_bytes':(d/'README.md').stat().st_size,'detail_bytes':(d/'検証詳細.md').stat().st_size},ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);run(parser.parse_args().directory)
