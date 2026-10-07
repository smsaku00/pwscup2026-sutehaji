"""保存した集計から説明を作る。新たな推定回答や提出物は作らない。"""
from __future__ import annotations
import 分析 as core
import datetime, hashlib, json
from pathlib import Path
import pandas as pd


def run():
    out=core.HERE/'集計'
    official=json.loads((core.HERE/'公式第1回第2回の確認.json').read_text())[-1]
    splits=pd.read_csv(out/'参照C_反復分割_揺れの集計.csv')
    fixed=splits[splits.policy.eq('fixed')].set_index('metric')
    adaptive=splits[splits.policy.eq('rare_guard_all')].set_index('metric')
    paired=pd.read_csv(out/'参照C_対応層を保った内側再選択置換.csv').set_index('policy')
    diagnoses=pd.read_csv(out/'提案規則_本戦の変更件数.csv')
    assert len(diagnoses)==22 and diagnoses.outside_candidates_changed.sum()==0
    assert int(diagnoses[diagnoses.team.eq(9)].changed.iloc[0])==0
    null=pd.read_csv(out/'他C参照_探索時のラベル置換.csv').set_index('family')
    single=pd.read_csv(out/'参照C_選択基準比較_平均.csv')
    guarded=single[(single.family.eq('published8keep'))&(single.search.eq('small7'))&(single.objective.eq('rare_guard_all'))].iloc[0]
    rare=json.loads((core.HERE/'希少候補_実行記録.json').read_text())
    today=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(timespec='seconds')
    info=dict(generated_at_jst=today,main_official_rare=float(official['r_time_rare']),user_target_rare=.0216,
              official_target_score_achieved=False,main_new_score_evaluated=False,codabench_submitted=False,new_submission_files_created=False,
              fixed_rule_split_mean_rare=float(fixed.loc['r_rare','split_mean']),fixed_rule_split_mean_all=float(fixed.loc['r_all','split_mean']),
              adaptive_split_mean_rare=float(adaptive.loc['r_rare','split_mean']),unique_public_evaluation_targets=18,split_repeats=50,
              proposed_main_changed_rows=int(diagnoses.changed.sum()),target9_changed=0,outside_estimated_candidates_changed=0)
    core.save_json(core.HERE/'主要な結論.json',info)
    methods=[('希少候補の別照合','二重チーム分離_平均.csv','family','published8keep'),
             ('個人単位の時間学習','個人単位の希少時間学習_二重CV_平均.csv','family','published8keep'),
             ('一対一照合・全体非低下条件','一対一照合_二重CV_平均.csv','objective','rare_guard_all'),
             ('他C参照・253設定から希少で選択','他チームC参照_二重CV_平均.csv','family','published8keep')]
    comparison=[]
    for label,name,col,value in methods:
        d=pd.read_csv(out/name);r=d[d[col].eq(value)].iloc[0]
        comparison.append(f'| {label} | {r.r_rare:.6f} | {r.r_all:.6f} |')
    comparison.append(f'| 他C参照・7設定＋全体非低下条件・既定1分割 | {guarded.r_rare:.6f} | {guarded.r_all:.6f} |')
    comparison.append(f"| 同条件・50分割の平均 | {adaptive.loc['r_rare','split_mean']:.6f} | {adaptive.loc['r_all','split_mean']:.6f} |")
    comparison.append(f"| 他C参照・上位52件／距離差0を固定・50分割 | {fixed.loc['r_rare','split_mean']:.6f} | {fixed.loc['r_all','split_mean']:.6f} |")
    comparison.append('| 比較元の予備戦第8回・同じ18対象 | 0.000000 | 0.086860 |')
    text=f'''# AIA希少を0.0216へ近づける方法の分析

更新：{today}。**本戦で0.0216に達したという結果ではない。新しい提出用CSV・ZIPは作成せず、CodaBenchには提出していない。**

9:00までの掲載は完了できなかった。本資料には、13:18以降の継続指示を受けて追加した分割安定性・対応置換の確認も含む。

## 1. 前提

目標0.0216はユーザーが提示した値【ユーザー指定】。自チームの本戦第2回の公式AIA希少平均は **{official['r_time_rare']:.12f}**、全体平均は {official['r_time_all']:.12f}【公式スコア】。差は {(.0216-official['r_time_rare']):.12f}、22対象の寄与合計では {(.0216-official['r_time_rare'])*22:.12f} の改善が必要になる。他チームの成績の再検証は本分析の対象外。

各AIA課題は326人で、標的・対照各163人、そのうち希少各13人・希少発症各4人。対照は県・性別・年齢帯・発症有無を標的とそろえてある。標的は非共有ブロックPから選ばれるため、MIAの共有希少24人の復元を、そのままAIA標的の復元として使うことはできない【公式ルール・配布データ】。

希少13対13では、標的正解数m・対照正解数cに対し、p_m=m/13、p_c=c/13 として

`R_rare = clip((p_m-p_c-1.64*sqrt((p_m*(1-p_m)+p_c*(1-p_c))/13))/(1-p_c), 0, 1)`

を計算し、22対象で平均する。独立実装と公式関数を全196組で照合した【公式コード・再集計】。例えば、現在寄与0の1対象を(m,c)=(6,0)にでき、他の対象が変わらなければ平均は約0.024091になる。(5,0)では約0.020843にとどまる。これは必要規模を示す仮定計算であり、実際の対象の正解数を特定したものではない。

各対象の採点には`max(R_all,R_rare)`から床を引く処理がある。表示の希少平均だけを上げても、総合得点にそのまま加算されるとは限らない【公式コード】。また、公式検証器にはtime_hatの上限10年の検査がなく、11年の値検証も確認した。真のtime≤10年・許容差0.5年なので11年は必ず外れる。0年は真値が0.5年以下なら当たり、同じ役割ではない。ここでは分析用の棄権値として11年を比較した。

## 2. 仮説

**第一候補は、推定希少候補だけについて「対象Cに固有の近接」を測り、根拠が弱い候補では11年を使う固定規則。**

希少判定は公開済み予備戦のQI（age・SBP・TG・HDL・ALT・sex）から学習する。未学習チームに対する平均AUCは約0.942、上位26件の希少再現率は約0.673。本戦自チームB28での外部評価AUCは {rare['external_B28']['auc']:.6f} だったが、Bは希少2.3%、AIAは約8%と構成が異なる。希少判定が良いことは、標的と対照を分けられることやtimeが当たることを意味しない【公開予備戦正解による検証・自チーム外部評価】。

対象Cへの最近傍距離d_Cと、他の各公開Cへの最近傍距離の中央値d_refを比べる。AIA標的は非共有なので、他Cは本人の答えを探す場所ではなく、偶然の近接を測る参照として使う。ただし他Cは加工方式が異なり、参照分布としての偏りは残る【公式ルールに基づく仮説】。

MIA文献の[DOMIAS](https://proceedings.mlr.press/v206/breugel23a.html)は公開合成データと参照分布の密度比を使う。本案はその考え方をAIAの照合に応用したもので、論文の性能が本競技で保証されるという主張ではない。本案の距離比も、標的の確率や厳密な尤度比ではない。

比較した別仮説は、背景由来C行の除去、希少候補だけへの別距離・時間窓、標的／対照への時間適中を別々に学習する方法、個人単位の多出力学習、同じC行の重複利用を防ぐ相互最近傍・一対一割当。これらを強い採用根拠としては扱わない。

## 3. 分析方法と結果

予備戦の公開正解21対象を利用し、自チーム5と加工が弱い2対象（1・2）を除く18対象を主評価とした。原データのほぼ直接照合だけで得られる成績が、加工された本戦Cに対する期待を押し上げないようにするためである。他C参照には公開Cをラベルなしで使う。外側7分割、内側6分割で、外側チームの正解を希少判定の学習と設定選択の両方から外した。真のis_rareを推定時の選別には渡していない【公開予備戦正解による検証】。

比較元は実際の予備戦第8回回答。下表のRはこの18対象の値で、本戦22対象の公式値とは直接比較できない。探索で候補群を作った後の追加検証であり、最後まで未使用の独立データはない。

| 方法／設定選択 | 希少平均 | 全体平均 |
| --- | ---: | ---: |
{chr(10).join(comparison)}

多くの設定を探索すると、希少平均0.02〜0.03を超えるものは見つかる。しかし別照合・時間学習・一対一照合では、設定選択から評価チームを外すと改善が縮んだ。他C参照も253設定から希少平均だけで選ぶ方式は安定しなかった。

**0.024335が出た7設定の全体非低下条件は、50分割では平均{adaptive.loc['r_rare','split_mean']:.6f}、範囲{adaptive.loc['r_rare','split_min']:.6f}〜{adaptive.loc['r_rare','split_max']:.6f}になった。** 内側で全体を守る条件を置いても、外側の全体非低下は保証されない。

**上位52件・距離差0を固定した規則**は、50分割で希少平均{fixed.loc['r_rare','split_mean']:.6f}、範囲{fixed.loc['r_rare','split_min']:.6f}〜{fixed.loc['r_rare','split_max']:.6f}、全体平均{fixed.loc['r_all','split_mean']:.6f}だった。50回は同じ18対象の分割を変えた数で、900個の独立対象を評価したわけではない。分割範囲は性能の信頼区間ではない。固定規則自体も先行探索の結果から選んだため、この値には選択の影響が残る。

偶然性は二段階で確認した。253設定の最大値を、対象・希少発症層を保った2万回のラベル置換と比較すると、予備戦第8回保持方式の探索最大0.022826に対する簡易の多設定補正pは{null.loc['published8keep','exploratory_family_adjusted_p']:.6f}だった。単一設定だけのp≈0.008を、採用の証明に使うことはできない。

追加で公式の対応条件（対象・希少・県・性別・10歳年齢帯・発症有無）を保ち、希少ラベルだけを置換し、内側の設定選択も毎回再計算した。固定規則の条件付きpは{paired.loc['fixed','conditional_p']:.6f}、比較した3方針の最大値による補正pは{paired.loc['fixed','three_policy_max_adjusted_p']:.6f}。**初めの候補群の探索全体は補正しておらず、非希少のラベルを固定した条件付き診断である。本戦の有意性検定や、0.0216達成の証明ではない。**

本戦の診断では、推定希少上位52件だけへ固定規則を適用し、チーム9を現回答のまま保持すると、変更候補は合計{int(diagnoses.changed.sum())}行だった。候補外の変更は0行、チーム9の変更も0行と確認した。正解は未知なので新しい本戦スコアは計算していない。変更件数は正解数ではない【本戦公開入力と既存回答の診断】。

## 4. 結論と提案

**本戦0.0216を達成できると確認した方法は、まだない。次の比較候補は、設定を増やして最高値を探す方式より、他C参照を使う上位52件・距離差0の固定規則を優先する。**

今回変える案は、(1)公開予備戦だけで学習したQI分類器で各課題の上位52件を推定希少候補とする、(2)log変換したTG・ALTを含む5数値項目を背景IQRで標準化し、性別一致で最近傍を取る、(3)d_C≤他C距離の中央値ならCのtimeを使い、そうでなければ11年、(4)候補外と本戦チーム9は第2回回答を保持、というもの。地域は距離の制約から外す。

比較するのは現回答と、この固定規則による候補内変更である。まずローカルで候補内外の保持、公開予備戦の標的正解数／対照正解数の変化、全体Rの低下を確認する。この比較により、単に希少らしい人の一般的なtimeを当てているのか、対象Cに固有の情報を拾えているのかを判断する。本戦の他チームの真値はないため、ローカルだけで公式希少Rや0.0216を確定することはできない。

11年を増やすと対照だけでなく標的の正解も失う。希少候補の誤選択、参照Cの加工の差、本戦と予備戦の分布差、18対象という小さい評価規模も残る。現在の本戦希少平均に予備戦の改善値を加えた「予測スコア」は作らない。

これは方法の提案と分析の記録である。新しいF_aia.csvや提出ZIPの作成は、採用・実装の指示を受けてから行う。**CodaBenchへの提出は行わない。**

### 再現とファイル

ルート`.venv/bin/python`と`requirements.txt`の環境を使用。集計は`集計/`、計算の条件・ハッシュは各`*_実行記録.json`、主要値は`主要な結論.json`にある。非公開B・utility_ref・token、学習済みモデル、作業用配列、提出CSV／ZIPは掲載対象外。B28の利用は外部評価と標的のみの採点で、他チームへの推定器の学習には使っていない。

GitHubのコードを再実行するときは、コードだけを新しい作業フォルダへコピーし、`--root`にPWS配布データと既存回答を置いた作業ルートを指定する。集計の旧記録を上書きしない。全工程を再現する入口は`分析.py --all --root <PWS作業ルート>`。個別確認なら以下を使う。

```sh
.venv/bin/python -B "本戦/攻撃フェーズ/Tashiro/20261007_AIA希少_改善方法の分析/分析.py"
.venv/bin/python -B "本戦/攻撃フェーズ/Tashiro/20261007_AIA希少_改善方法の分析/参照Cの分割安定性と対応置換.py" --repeats 50 --permutations 20000
```

時間学習の比較特徴量は`比較特徴量キャッシュ.py`と`時間候補の特徴量.py`で再作成できる。既存の凍結キャッシュを読み取る場合は元コードと入力ハッシュを確認する。新たに公開入力だけから作った配列は、既存キャッシュの全配列と完全一致を確認した。新しい配列にも本戦の未知真値は含まない。

公式根拠：[AIA採点コード](https://github.com/pwscup/pwscup2026-kit/blob/main-attack-20260928/starter/src/pwscup2026/scoring/attack_score.py)、[値検証](https://github.com/pwscup/pwscup2026-kit/blob/main-attack-20260928/starter/src/pwscup2026/kit/validate.py)、[公式ルールブックの所在](https://github.com/pwscup/pwscup2026-kit/tree/main-process-20260912/rulebook)。本戦ルールブックの§3.1・§4.2・§6.2.2と、公開済み予備戦の正解を参照した。
'''
    (core.HERE/'説明.md').write_text(text,encoding='utf-8')
    (core.HERE/'README.md').write_text('# AIA希少の方法検討・2026-10-07\n\nまず[説明.md](説明.md)を読む。本戦0.0216達成の確認ではない。\n\n次の比較候補は、推定希少上位52件に対する他チームC参照の固定規則。分析コードと集計のみを掲載し、CodaBenchには提出していない。\n',encoding='utf-8')
    core.save_json(core.HERE/'レポート_再集計記録.json',dict(generated_at_jst=today,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_csv_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.csv')},submission_created=False,codabench_submitted=False))
    print(json.dumps(info,ensure_ascii=False),flush=True)


if __name__=='__main__':run()
