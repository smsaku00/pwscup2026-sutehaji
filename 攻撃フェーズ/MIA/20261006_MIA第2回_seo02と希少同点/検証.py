"""生成処理を呼ばず、保存CSVと固定資料から独立に第2回MIAを検査する。

既定は読取専用。--writeは新規作成中の一時フォルダだけに検査記録を作る。
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import html
import importlib.util
import itertools
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_curve

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PARENT = HERE.parent
FIRST = PARENT/'20261005_MIA希少_第1回同点データ/結果'
BASE = PARENT/'20261005_MIA初回_GenLRA比較'
PROOF = PARENT/'20261006_MIA希少_本戦102配置検証/結果'
INPUT = PARENT/'20260928_配布データ基本分析/入力'
TEAMS = list(map(str, [*range(18), 22, 25, 26, 27]))
ROUND2 = dict(zip(TEAMS, [12,13,23,22,11,17,23,14,20,15,13,18,16,16,23,20,14,19,24,21,23,13]))
SEO_SHA = 'f5ebe3d5fad771d7504c8aafce5d29861764cd090161a26fe3aa3f7c25d55511'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jread(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.reader(f))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def official_functions():
    kit, = (ROOT/'本戦').glob('配布*/pwscup2026-kit-main_revised0912')
    sys.path.insert(0, str(kit/'starter/src'))
    validate = module('pwscup2026.kit.validate_round2', BASE/'参照コード/validate_20260928.py')
    score = module('pwscup2026.scoring.attack_score_round2', BASE/'参照コード/attack_score_20260928.py')
    return validate.validate_mia_submission, score


def main():
    if not __debug__:
        raise RuntimeError('検査を省略する実行オプションは禁止です')
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=HERE/'結果')
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    out = args.directory
    if args.write:
        assert out.parent.name.startswith('.検証中_') and out.parent.parent == HERE
        assert not (out/'検証結果.json').exists()
    if (out/'出力SHA256.json').exists():
        for name, digest in jread(out/'出力SHA256.json').items():
            assert sha(out/name) == digest, f'出力変更: {name}'
    for name, digest in jread(out/'入力SHA256.json').items():
        assert sha(Path(name)) == digest, f'入力変更: {name}'
    assert sha(PROOF/'初期条件.json') == '67285b9d6aeebf1a75ec43f692db1f15869e1a3db6f099340155378a8cae4ccb'
    assert sha(PROOF/'人数102通り.json') == '51be9f8d08ac1c785c01e4719bce39f44d907af828803de4e886a9d812897201'
    initial = jread(PROOF/'初期条件.json')
    public = initial['public']
    for name, digest in initial['source_sha256'].items():
        assert sha(Path(name)) == digest
    for name, digest in jread(FIRST/'出力SHA256.json').items():
        assert sha(FIRST/name) == digest
    plan = jread(out/'設定.json')
    assert plan['round'] == 2 and plan['own_team'] == 28 and plan['teams'] == TEAMS
    assert public['teams'] == TEAMS and public['round2_sizes'] == list(ROUND2.values())
    first_plan = jread(FIRST/'設定.json')
    for t, n in zip(first_plan['teams'], public['round1_sizes']):
        assert n == t['round1_tied']
        assert ROUND2[t['target_k']] == (21 if t['target_k'] == '25' else t['round2_tied'])
    assert (plan['rare_high'], plan['rare_low']) == ('0.4', '0.1')
    bpath, = (ROOT/'本戦').glob('配布*/PWSCup2026_final_team28_*/B_self.csv')
    b = pd.read_csv(bpath, float_precision='round_trip')
    q = pd.read_csv(INPUT/'aia_challenge_28.csv', float_precision='round_trip')
    qi = ['age','SBP','TG','HDL','ALT','sex','prefecture']
    assert not b.duplicated(qi).any() and not q.duplicated(qi).any()
    private = set(b.merge(q[qi], on=qi, validate='one_to_one').record_id)
    rare = sorted(b.loc[b.is_rare == 1, 'record_id'])
    unknown = sorted(set(rare)-private)
    negative_rare = sorted(set(rare)&private)
    regular = sorted(set(b.record_id)-private-set(rare))
    negative_regular = sorted(private-set(rare))
    assert (len(rare),len(unknown),len(negative_rare),len(regular),len(negative_regular)) == (24,11,13,875,150)
    assert rare == public['rare_ids'] == plan['rare_ids']
    assert unknown == public['unknown_ids'] == plan['unknown_rare_ids']
    assert negative_rare == public['known_nonmember_ids'] == plan['known_nonmember_rare_ids']
    assert regular == plan['overall_candidate_ids'] and negative_regular == plan['nonrare_known_nonmember_ids']
    expected_highs = {t: sorted(unknown+negative_rare[:ROUND2[t]-11]) for t in TEAMS}
    assert plan['rare_settings'] == [{'target_k':t,'tied_count':ROUND2[t], 'high_record_ids':expected_highs[t]} for t in TEAMS]
    raw = pd.read_csv(out/'全体手法の出力.csv', float_precision='round_trip')
    assert raw.columns.tolist() == ['record_id',*TEAMS] and raw.record_id.tolist() == list(range(1049))
    assert np.isfinite(raw.to_numpy()).all()
    # 生成処理とは別に、順位を分数で正確に変換して期待値を作る。
    ranks = raw.loc[regular,TEAMS].rank(method='average')
    expected_regular = ranks.map(lambda r: float(Fraction(11,20) + Fraction(2,5)*Fraction(str(r))/875))
    data = rows(out/'F_mia.csv')
    declared = jread(INPUT/'mia_columns.json')
    assert [t for t in declared['target_columns'] if t != '28'] == TEAMS
    validator, official = official_functions()

    def structure(csv_rows):
        assert len(csv_rows) == 1050 and csv_rows[0] == ['record_id',*TEAMS], '行列数または列順'
        assert all(len(r) == 23 for r in csv_rows), '列欠落'
        assert [r[0] for r in csv_rows[1:]] == list(map(str,range(1049))), 'IDまたは行順'
        numeric = np.array([r[1:] for r in csv_rows[1:]], dtype=float)
        assert np.isfinite(numeric).all() and ((numeric>=0)&(numeric<=1)).all(), '値域または欠損'
        for j,t in enumerate(TEAMS):
            for rid in rare:
                assert csv_rows[rid+1][j+1] == ('0.4' if rid in expected_highs[t] else '0.1'), f'希少セル team{t} ID{rid}'
        assert np.array_equal(numeric[regular], expected_regular.to_numpy()), '全体順位と保存値'
        assert np.all(numeric[negative_regular] == 0), '非希少の確定非会員'
        assert np.all(numeric[regular] > .55) and np.all(numeric[regular] <= .95), '全体と希少の値域'
        frame = pd.DataFrame(numeric, columns=TEAMS)
        frame.insert(0, 'record_id', range(1049))
        valid = validator(frame, {}, {'mia_columns':declared,'team_id':28,'attacker_record_ids':set(range(1049))})
        assert valid.ok, valid.errors
        return frame

    frame = structure(data)
    print('希少528セル・全体19250セル・非会員3300セル・公式形式を確認', flush=True)
    # チーム単位の全66通りを、保存済みCSVから公式コードと別ライブラリの両方で採点する。
    checks = 0
    max_error = 0.
    subset_rates = {}
    for t in TEAMS:
        confidence = frame.loc[rare,t].to_numpy()
        subset_rates[t] = {}
        for m in (1,2):
            expected = float(Fraction(24-m,100*(ROUND2[t]-m)))
            for ids in itertools.combinations(unknown,m):
                truth = np.isin(rare,ids).astype(int)
                value = official._tpr_at_fpr(confidence,truth,.01)[0]
                fp,tp,_ = roc_curve(truth,confidence,drop_intermediate=False)
                separate = float(np.interp(.01,fp,tp))
                assert abs(value-expected)<1e-14 and abs(separate-expected)<1e-14
                max_error = max(max_error,abs(value-expected))
                checks += 1
            subset_rates[t][m] = value
    assert checks == 1452
    # 1回目の実際の点数から、0/1/2人・合計11人以下の候補全てを再探索する。
    count = module('round2_count_reference', PARENT/'20261005_MIA希少_2回200通り検証/分析.py')
    report_path = Path('/Users/taisei1105/Downloads/scoring_result (2)/detailed_results.html')
    result1 = json.loads(html.unescape(re.search(r'<pre[^>]*>(.*?)</pre>',report_path.read_text(),re.S).group(1)))
    first_score = result1['mia']['tpr_rare']
    assert first_score == public['first_score'] == plan['first_rare_score']
    candidates = count.Decoder([(n,n) for n in public['round1_sizes']]).decode([first_score,first_score])
    saved_cases = jread(PROOF/'人数102通り.json')['cases']
    assert candidates == [tuple(case['counts'][t] for t in TEAMS) for case in saved_cases]
    assert len(candidates) == 102
    old = pd.read_csv(FIRST/'F_mia.csv',float_precision='round_trip')
    score_table = []
    rare_mask = b.is_rare.to_numpy(bool)
    full_checks = 0
    for case, counts in zip(saved_cases,candidates):
        active = sum(m>0 for m in counts)
        exact = sum((Fraction(24-m,100*(ROUND2[t]-m)) for t,m in zip(TEAMS,counts) if m),Fraction())/active
        assert exact == Fraction(case['revised_round2'])
        first_values, second_values = [], []
        position = 0
        for t,m in zip(TEAMS,counts):
            ids = unknown[position:position+m]
            position += m
            truth = np.isin(b.record_id,ids).astype(int)
            # 模擬ラベル: 非希少20人を加えても希少採点が同一になることを確認する。
            truth[regular[:20]] = 1
            detail = official.score_mia(frame[t].to_numpy(),truth,{'scoring':{'low_fpr':.01}},rare_mask=rare_mask).breakdown
            changed = frame[t].to_numpy().copy()
            changed[~rare_mask] = 1.-changed[~rare_mask]
            changed_detail = official.score_mia(changed,truth,{'scoring':{'low_fpr':.01}},rare_mask=rare_mask).breakdown
            if m:
                assert abs(detail['tpr_rare']-subset_rates[t][m]) < 1e-14
                assert detail['tpr_rare'] == changed_detail['tpr_rare']
                first_values.append(official._tpr_at_fpr(old.loc[rare,t].to_numpy(),np.isin(rare,ids).astype(int),.01)[0])
                second_values.append(detail['tpr_rare'])
            else:
                assert np.isnan(detail['tpr_rare']) and np.isnan(changed_detail['tpr_rare'])
            full_checks += 1
        assert abs(float(np.mean(first_values))-first_score) < 1e-14
        observed = float(np.mean(second_values))
        assert abs(observed-float(exact))<1e-14
        score_table.append({'case_id':case['case_id'],'round2_score_exact':str(exact),
                            'round2_score':observed,'counts':dict(zip(TEAMS,counts)),
                            'active_teams':active,'total_members':sum(counts)})
    exact_scores = sorted(Fraction(v['round2_score_exact']) for v in score_table)
    assert len(set(exact_scores)) == 102
    gap = min(b-a for a,b in zip(exact_scores,exact_scores[1:]))
    assert gap == Fraction(1,936000)
    for item in score_table:
        hits = [v for v in score_table if abs(v['round2_score']-item['round2_score']) <= 1e-13]
        assert len(hits)==1 and hits[0]['case_id']==item['case_id']
    print('全1452通りの希少採点と、102通りの人数の区別を確認', flush=True)
    # 誤りが検査を通過しないことも確認。ディスク上の候補は変更しない。
    failures = []
    def reject(label, mutate):
        bad = copy.deepcopy(data)
        mutate(bad)
        try:
            structure(bad)
        except (AssertionError,ValueError,TypeError):
            failures.append(label)
        else:
            raise AssertionError(f'誤りを見逃した: {label}')
    def cell(rid,t,value):
        return lambda d: d[rid+1].__setitem__(TEAMS.index(t)+1,value)
    reject('希少候補1人を0.1に変更',cell(unknown[0],'0','0.1'))
    reject('希少既知非会員を一括0化する誤り',cell(negative_rare[0],'0','0'))
    reject('希少値への微小な加算',cell(unknown[0],'0','0.4000000000001'))
    reject('希少値を全体用へ変換',cell(unknown[0],'0','0.95'))
    def old_team25(d):
        high=set(unknown+negative_rare[:3])
        for rid in rare:
            d[rid+1][TEAMS.index('25')+1]='0.4' if rid in high else '0.1'
    reject('チーム25を旧設定14人へ戻す',old_team25)
    reject('チーム列名の入替',lambda d:d[0].__setitem__(slice(1,3),d[0][1:3][::-1]))
    reject('チーム列の値だけ入替',lambda d:[r.__setitem__(slice(1,3),r[1:3][::-1]) for r in d[1:]])
    reject('1回目CSVの再利用',lambda d:d.__setitem__(slice(None),rows(FIRST/'F_mia.csv')))
    reject('行欠落',lambda d:d.pop())
    reject('ID重複',lambda d:d[2].__setitem__(0,d[1][0]))
    reject('行順の変更',lambda d:d.__setitem__(slice(1,3),d[1:3][::-1]))
    reject('自チーム列の混入',lambda d:d[0].__setitem__(1,'28'))
    reject('列欠落',lambda d:[r.pop() for r in d])
    reject('非希少既知非会員を高くする',cell(negative_regular[0],'0','0.9'))
    reject('全体順位の破損',cell(regular[0],'0','0.56'))
    for invalid in ['NaN','inf','1.1','-0.1','']:
        reject('不正値 '+invalid,cell(regular[0],'0',invalid))
    # 原版のMIA処理を再実行。生成処理の中間値と保存後の全22列を照合する。
    assert sha(out/'参照seo02.py') == SEO_SHA
    seo = module('independent_seo02',out/'参照seo02.py')
    bb=seo.read_table(bpath.with_name('B.csv'),'B')
    a=seo.read_table(INPUT/'A_bg.csv','A')
    scaler=seo.Scaler(a)
    for t in TEAMS:
        actual=seo.mia_scores(bb,a,seo.read_table(INPUT/f'C_{t}.csv','C'),scaler)
        assert np.array_equal(actual,raw[t].to_numpy()), f'seo-02再実行 team{t}'
    comparisons=[]
    for t in TEAMS:
        before=sorted(regular,key=lambda rid:(-old.loc[rid,t],rid))[:20]
        after=sorted(regular,key=lambda rid:(-frame.loc[rid,t],rid))[:20]
        comparisons.append({'team':t,'top20_common':len(set(before)&set(after)),
                            'spearman':float(spearmanr(old.loc[regular,t],frame.loc[regular,t]).statistic)})
    report={'status':'全検査成功','rows':1049,'opponent_columns':22,'rare_cells_checked':528,
            'nonrare_rank_cells_checked':875*22,'nonrare_zero_cells_checked':150*22,
            'official_format_validation':True,'official_and_sklearn_subset_cases':checks,
            'maximum_rare_score_error':max_error,'count_candidates_regenerated':102,
            'uniquely_distinguishable_count_cases':102,'minimum_round2_gap_exact':str(gap),
            'minimum_round2_gap':float(gap),'score_match_tolerance':1e-13,
            'full_1049_row_rare_isolation_checks':full_checks,
            'intentional_errors_rejected':failures,'seo02_recomputed_columns':22,
            'team25_tied_count':21,'rare_unknown_11_all_high':True,
            'first_csv_sha256':sha(FIRST/'F_mia.csv'),'second_csv_sha256':sha(out/'F_mia.csv'),
            'real_main_accuracy':'未採点。本戦の実際の所属は使用していない。',
            'submitted':False}
    outputs={'検証結果.json':report,'2回目の点数と人数102通り.json':score_table,'初回との順位比較.json':comparisons}
    if args.write:
        for name,value in outputs.items():
            with (out/name).open('x',encoding='utf-8') as f:
                json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
                f.write('\n')
        with (out/'確認用_希少24人.csv').open('x',encoding='utf-8',newline='') as f:
            csv.writer(f,lineterminator='\n').writerows([data[0]]+[data[rid+1] for rid in rare])
        with (out/'確認用_チーム別同点人数.csv').open('x',encoding='utf-8',newline='') as f:
            csv.writer(f,lineterminator='\n').writerows([['本戦チーム番号','0.4の人数','未確定候補','確定非会員の追加','0.1の人数']]+[[t,ROUND2[t],11,ROUND2[t]-11,24-ROUND2[t]] for t in TEAMS])
    else:
        for name,value in outputs.items():
            assert jread(out/name)==value, f'保存検査記録との不一致: {name}'
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
