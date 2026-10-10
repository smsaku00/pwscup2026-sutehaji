"""Read-only round-5 decoding from frozen originals and actual Library HTML.

Run with the PWS project Python: python -B decode_actual_round5.py --inputs-dir LOCAL_FROZEN_INPUTS
No submission, original edits or GitHub mutations. All outputs stay beside this script.
The published prior audit is the default code/report source. Its hash-pinned
private inputs must be supplied locally and must not be uploaded to GitHub.
"""
from __future__ import annotations
import argparse
import csv
from collections import Counter, defaultdict
from fractions import Fraction as F
import hashlib
import html
import importlib.util
from itertools import combinations, product
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1] / 'MIA全体/round5_generation_20261010/希少残り2問_独立監査'
INPUT_OVERRIDE = None
ACTUAL = HERE / 'inputs/detailed_results.html'
LIBRARY_ID = 'libfile_b942113fd8c8819194a7296d6bb49a41'
EXPECTED_HTML_SHA = '2921cdf8a7af4958b78ac64bae04e6297d46eabf1e7b664d48e48f07a8a980dd'

def sha(b):
    return hashlib.sha256(b).hexdigest()

def dump(name, obj):
    (HERE / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def write_csv(name, rows):
    with (HERE / name).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def run():
    spec = importlib.util.spec_from_file_location('independent_base_audit', BASE / 'audit.py')
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    if INPUT_OVERRIDE is not None:
        audit.INPUT = INPUT_OVERRIDE
    # Imported helpers read originals; disable their output paths to preserve prior audit.
    audit.dump = lambda *args, **kwargs: None
    audit.table = lambda *args, **kwargs: None
    before = {p.name: sha(p.read_bytes()) for p in BASE.iterdir() if p.is_file()}
    manifest = audit.verify_inputs()
    actual_bytes = ACTUAL.read_bytes()
    assert len(actual_bytes) == 3884 and sha(actual_bytes) == EXPECTED_HTML_SHA
    text = actual_bytes.decode('utf-8')
    plain = html.unescape(re.sub('<[^>]+>', '\n', text))
    assert 'チームID 28' in plain and '検証OK・採点完了' in plain
    blocks = [json.loads(html.unescape(s)) for s in re.findall(r'<pre[^>]*>(.*?)</pre>', text, re.S)]
    actual_report, = [x for x in blocks if 'mia' in x]
    observed5 = actual_report['mia']['tpr_rare']
    assert observed5 == 0.011166666666666667
    assert actual_report['constants']['low_fpr'] == .01
    # mia_per_target contains tier numbers, not per-target rare TPR.
    assert set(actual_report['mia_per_target']) == set(map(str, audit.TEAMS))

    rare, negative, candidates, private = audit.original_partition()
    matrices = {r: audit.matrix(r) for r in range(1, 6)}
    previous = [audit.observed(r) for r in range(1, 5)]
    scorer, nanmean = audit.load_official()
    high_early = {r: tuple(map(int, (matrices[r][rare, :] == .4).sum(0))) for r in (1, 2)}
    counts, count_check = audit.rederive_counts(high_early[1], high_early[2], previous[:2])
    active = tuple(t for t in audit.TEAMS if counts[t])
    ms = tuple(counts[t] for t in active)
    assert (active, ms) == ((6,10,11,16,17,25), (1,2,1,2,1,2))
    high = tuple(int((matrices[3][rare, audit.TEAMS.index(t)] == .4).sum()) for t in active)
    questions = {r: set(q) for r,q in audit.FIXED_QUESTIONS.items()}
    saved_plan = json.loads((audit.INPUT / 'fixed_plan.json').read_text())
    assert {int(r): set(q) for r,q in saved_plan['fixed_queries'].items()} == questions
    padding = {t: tuple(negative[:n-5]) for t,n in zip(active,high)}
    rare_matrices = {r: matrices[r][rare, :] for r in range(1,6)}
    for r in (3,4,5,6,7):
        v = np.zeros((24,len(audit.TEAMS)))
        for t in active:
            selected = questions[r] | set(padding[t])
            v[:,audit.TEAMS.index(t)] = [.4 if i in selected else .1 for i in rare]
        if r <= 5:
            assert np.array_equal(v, rare_matrices[r])
        else:
            rare_matrices[r] = v
    grid = {ks: sum((audit.question_score(n,m,k) for n,m,k in zip(high,ms,ks)),F(0))/6
            for ks in product(*(range(m+1) for m in ms))}
    assert len(grid) == len(set(grid.values())) == 216
    patterns = {}
    for r,obs in ((3,previous[2]),(4,previous[3]),(5,observed5)):
        matched = [ks for ks,s in grid.items() if abs(float(s)-obs)<1e-15]
        assert len(matched) == 1
        patterns[r] = matched[0]
    groups3 = [list(g for g in combinations(candidates,m) if len(set(g)&questions[3])==k)
               for m,k in zip(ms,patterns[3])]
    after3 = sum(1 for _ in audit.legal_product(groups3))
    groups4 = [list(g for g in gs if len(set(g)&questions[4])==k)
               for gs,k in zip(groups3,patterns[4])]
    placements = list(audit.legal_product(groups4))
    assert after3 == 43200 and len(placements) == 1296
    cfg = {'scoring': {'low_fpr': .01}}
    cache = {}; calls = 0
    for r in range(1,8):
        for t in audit.TEAMS:
            for g in combinations(candidates,counts[t]) if counts[t] else [()]:
                truth = np.array([int(i in g) for i in rare])
                val = scorer.score_mia(rare_matrices[r][:,audit.TEAMS.index(t)],truth,cfg,
                                       rare_mask=np.ones(24,dtype=bool)).breakdown['tpr_rare']
                calls += 1
                if counts[t]:
                    expected = audit.count_score(high_early[r][audit.TEAMS.index(t)],counts[t]) if r<=2 else audit.question_score(high[active.index(t)],counts[t],len(set(g)&questions[r]))
                    assert abs(val-float(expected))<1e-15
                else:
                    assert np.isnan(val)
                cache[r,t,g] = val
    computed = []; all_branches = defaultdict(list)
    for pid,placement in enumerate(placements,1):
        exact = {r: sum((audit.question_score(n,m,len(set(g)&questions[r])) for n,m,g in zip(high,ms,placement)),F(0))/6 for r in (5,6,7)}
        floats = {r: nanmean([cache[r,t,placement[active.index(t)] if t in active else ()] for t in audit.TEAMS]) for r in range(1,8)}
        assert all(floats[r] == previous[r-1] for r in range(1,5))
        assert all(abs(floats[r]-float(exact[r]))<1e-15 for r in (5,6,7))
        row = {'placement_id':pid, **{f'Team{t}':'+'.join(map(str,g)) for t,g in zip(active,placement)},
               'outside_22_targets':'+'.join(map(str,sorted(set(candidates)-{i for g in placement for i in g})))}
        row.update({f'round{r}_exact':str(exact[r]) for r in (5,6,7)})
        row.update({f'round{r}_official_float':repr(floats[r]) for r in (5,6,7)})
        computed.append((row,placement,exact,floats))
        all_branches[exact[5]].append(pid)
    # Independent recomputation must match the previously published 1296-row forecast.
    with (BASE / 'all_1296_placements_and_observations.csv').open(newline='') as f:
        predicted = list(csv.DictReader(f))
    assert [{k:str(v) for k,v in row.items()} for row,_,_,_ in computed] == predicted
    surviving = [x for x in computed if x[3][5] == observed5]
    assert len(surviving) == 16
    observed_exact = {x[2][5] for x in surviving}
    assert observed_exact == {F(67,6000)}
    branch_id = sorted(all_branches).index(F(67,6000))+1
    assert len(all_branches)==91 and branch_id==21
    # Full 1049-row real CSV scoring for all 16 witnesses; ordinary rows cannot
    # affect this metric because official code selects rare_mask before ROC.
    rare_mask = np.zeros(1049,dtype=bool); rare_mask[list(rare)] = True
    for row,placement,exact,floats in surviving:
        all_scores=[]
        for t in audit.TEAMS:
            truth=np.zeros(1049,dtype=int)
            if t in active:
                truth[list(placement[active.index(t)])]=1
            all_scores.append(scorer.score_mia(matrices[5][:,audit.TEAMS.index(t)],truth,cfg,rare_mask=rare_mask).breakdown['tpr_rare'])
            calls+=1
        assert nanmean(all_scores)==observed5
    target_candidates={}
    for ti,t in enumerate(active):
        subsets=sorted({placement[ti] for _,placement,_,_ in surviving})
        union=sorted(set().union(*map(set,subsets)))
        intersection=sorted(set.intersection(*map(set,subsets)))
        target_candidates[str(t)]={'number_members':counts[t],'possible_member_subsets':subsets,
                                  'possible_member_ids':union,'certain_member_ids':intersection,
                                  'certain_nonmember_ids_within_11_candidates':sorted(set(candidates)-set(union))}
    outside=sorted({tuple(map(int,row['outside_22_targets'].split('+'))) for row,_,_,_ in surviving})
    pairs=defaultdict(list); sixth=defaultdict(list)
    for row,placement,exact,floats in surviving:
        sixth[exact[6]].append(row)
        pairs[exact[6],exact[7]].append(row)
    collisions=[{'round6_exact':str(s6),'round7_exact':str(s7),'placement_ids':[r['placement_id'] for r in rows]}
                for (s6,s7),rows in pairs.items() if len(rows)>1]
    assert len(sixth)==6 and len(pairs)==16 and not collisions
    future=[]
    for s6,rows in sorted(sixth.items()):
        s7s=sorted({F(r['round7_exact']) for r in rows})
        future.append({'round6_exact':str(s6),'round6_official_float':rows[0]['round6_official_float'],
                       'remaining_after_round6':len(rows),'round7_response_branches':len(s7s),
                       'worst_remaining_after_round7':max(len(pairs[s6,s7]) for s7 in s7s),
                       'possible_round7_official_floats':' | '.join(next(r['round7_official_float'] for r in rows if F(r['round7_exact'])==s7) for s7 in s7s),
                       'placement_ids':'+'.join(str(r['placement_id']) for r in rows)})
    rounding=[]
    for digits in range(3,7):
        buckets=Counter(tuple(format(float(row[f'round{r}_official_float']),f'.{digits}f') for r in (6,7)) for row,_,_,_ in surviving)
        rounding.append({'decimal_places':digits,'unique_future_response_pairs':len(buckets),
                         'collision_groups':sum(n>1 for n in buckets.values()),'worst_remaining':max(buckets.values())})
    independent=json.loads((HERE / 'independent_result_check.json').read_text())
    assert independent['configurations_after_actual_round5']==len(surviving)
    independent_placements={tuple(tuple(row['team_members'][str(t)]) for t in active) for row in independent['all_16_configurations']}
    assert independent_placements == {placement for _,placement,_,_ in surviving}
    for t in active:
        assert independent['target_candidates_after_actual_round5'][str(t)]['possible_member_subsets']==[list(g) for g in target_candidates[str(t)]['possible_member_subsets']]
    write_csv('remaining_16_placements_and_predictions.csv',[row for row,_,_,_ in surviving])
    write_csv('round6_prediction_branches.csv',future)
    dump('target_candidates.json',target_candidates)
    dump('collision_check_actual_branch.json',{'tested_placements':16,'unique_round6_7_pairs':len(pairs),
          'collision_groups':0,'worst_after_round6':max(map(len,sixth.values())),
          'worst_after_round7':max(map(len,pairs.values())),'collisions':collisions})
    dump('rounding_precision_actual_branch.json',rounding)
    hashes={name:{'sha256':info['sha256'],'bytes':info['bytes'],'source_path':str(audit.INPUT/name)} for name,info in manifest.items()}
    hashes['actual_round5_detailed_results.html']={'sha256':sha(actual_bytes),'bytes':len(actual_bytes),
          'source_path':str(ACTUAL),'library_file_id':LIBRARY_ID,'team_id':28,'validation_ok':True}
    for p in (BASE/'audit.py',BASE/'all_1296_placements_and_observations.csv',HERE/'independent_result_check.json',HERE/'independent_result_check.py'):
        hashes[p.name]={'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size,'source_path':str(p)}
    dump('input_hashes_actual_round5.json',hashes)
    result={'status':'PASS','conclusion':'条件付きで、変更しない第6・7回の固定質問（5bit第4・5問）により全16配置を必ず一意に特定できる。',
       'actual_round5_observed':True,'round5_official_float':repr(observed5),'round5_exact':'67/6000',
       'prior_audit_publication_commit':'42deb52906836eafc73edec941b385aeabaaa70c',
       'round5_submission_source_commit':audit.COMMIT5,'round5_csv_sha256':manifest['round5_F_mia.csv']['sha256'],
       'raw_html_sha256':sha(actual_bytes),'raw_html_bytes':len(actual_bytes),'library_file_id':LIBRARY_ID,
       'team_id':28,'validation_ok':True,'count_rederivation':count_check,
       'prior_official_observations_exactly_reproduced':True,'prior_forecast_all_1296_rows_exactly_reproduced':True,
       'remaining_after_round3':after3,'remaining_after_round4':len(placements),
       'matching_round5_branch_id':branch_id,'all_round5_branches':len(all_branches),'remaining_after_actual_round5':16,
       'round5_positive_high_counts_by_target':dict(zip(map(str,active),patterns[5])),
       'target_candidates':target_candidates,'zero_positive_targets':[t for t in audit.TEAMS if not counts[t]],
       'possible_outside_22_target_pairs':outside,'certain_outside_22_target_ids':sorted(set.intersection(*map(set,outside))),
       'future_questions':{str(r):sorted(questions[r]) for r in (6,7)},'future_padding_by_target':{str(t):list(g) for t,g in padding.items()},
       'future_round6_branches':future,'round6_worst_remaining':4,'round7_unique_response_pairs':16,
       'round7_worst_remaining':1,'collision_groups':0,'adaptive_question_change_required':False,
       'official_scorer_calls':calls,'full_1049_row_round5_witnesses_checked':16,
       'independent_enumeration_placements_match':True,'minimum_future_decimal_places_for_actual_16_given_raw_past':4,
       'assumptions':['第5回保存CSVと実際のサーバー入力の同一性は依頼者の回番号対応に依存し、サーバー入力ZIPのhashは未確認',
          '元独立監査の候補・人数・共有排他・固定所属・公式希少マスク・nan除外平均の前提が維持',
          '第6・7回は示した固定質問と対象別paddingを使う（将来CSVは未生成・未提出）',
          '生のmia.tpr_rareを用いる。今回の実16分岐は未来2問4桁でも区別するが、第5回まで4桁だけに丸める保証ではない'],
       'prior_audit_unchanged':True,'github_updated_by_this_script':False,'submission_csv_changed':False,'tournament_submitted':False}
    assert before=={p.name:sha(p.read_bytes()) for p in BASE.iterdir() if p.is_file()}
    dump('actual_round5_results.json',result)
    dump('output_hashes_actual_round5.json',{p.name:sha(p.read_bytes()) for p in sorted(HERE.iterdir()) if p.is_file() and p.name!='output_hashes_actual_round5.json'})
    print(json.dumps({k:result[k] for k in ('status','matching_round5_branch_id','remaining_after_actual_round5','round6_worst_remaining','round7_worst_remaining','collision_groups','official_scorer_calls','independent_enumeration_placements_match')},ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-audit-dir', type=Path, default=BASE,
                        help='Published prior audit directory (audit.py, input_manifest.json and 1296-row CSV).')
    parser.add_argument('--inputs-dir', type=Path,
                        help='Local frozen private inputs whose hashes match the prior input_manifest.json.')
    parser.add_argument('--actual-html', type=Path, default=ACTUAL,
                        help='Original round-5 scoring HTML with the pinned SHA256.')
    args = parser.parse_args()
    BASE = args.base_audit_dir.resolve()
    INPUT_OVERRIDE = args.inputs_dir.resolve() if args.inputs_dir else None
    ACTUAL = args.actual_html.resolve()
    for name in ('audit.py', 'input_manifest.json', 'all_1296_placements_and_observations.csv'):
        if not (BASE / name).is_file():
            parser.error(f'Missing prior published audit file: {BASE / name}; specify --base-audit-dir.')
    if not (INPUT_OVERRIDE if INPUT_OVERRIDE else BASE / 'inputs').is_dir():
        parser.error('Supply the hash-pinned private input snapshot with --inputs-dir. Do not upload it.')
    run()
