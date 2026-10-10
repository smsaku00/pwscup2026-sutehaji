"""Independent audit: reads frozen originals; no submission, no original edits.

Run with /Users/taisei1105/pws2026-local/.venv/bin/python -B audit.py
Initial snapshot: add --snapshot-repo REPO --round5 CSV
All new files are confined to this script's directory.
"""
from __future__ import annotations

import argparse
import ast
import csv
from collections import Counter, defaultdict
from decimal import Decimal
from fractions import Fraction as F
import hashlib
import html
import importlib.util
from itertools import combinations, product
import json
from math import factorial, lcm
from pathlib import Path
import platform
import re
import sys
import types
import unicodedata

import numpy as np

HERE = Path(__file__).resolve().parent
INPUT = HERE / 'inputs'
TEAMS = tuple(range(18)) + (22, 25, 26, 27)
QI = ('age', 'SBP', 'TG', 'HDL', 'ALT', 'sex', 'prefecture')
COMMIT5 = '27d40de61aeb025858038230802bd822381a773c'
BLOB5 = '33b4c23ee276fda47b04dae6d54a241bd9ad3ef3'
OFFICIAL_KIT = Path('/Users/taisei1105/pws2026-local/本戦/配布データ/pwscup2026-kit-main_revised0912')
# The saved prospective plan is read independently of the generator.
FIXED_QUESTIONS = {
    3: (115, 169, 264, 383, 462), 4: (47, 115, 304, 462, 675),
    5: (47, 90, 264, 462, 495), 6: (47, 76, 264, 304, 383),
    7: (47, 76, 90, 115, 169),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git_blob(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def dump(name, obj):
    (HERE / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def table(name, fields, rows):
    with (HERE / name).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def normalized_dir(root, name):
    found = [p for p in root.iterdir() if p.is_dir() and unicodedata.normalize('NFC', p.name) == name]
    assert len(found) == 1, (root, name)
    return found[0]


def snapshot(repo, round5):
    mia = repo / '攻撃フェーズ/MIA'
    r1 = normalized_dir(mia, '20261005_MIA希少_第1回同点データ')
    r2 = mia / '20261006_MIA第2回_seo02と希少同点'
    r3 = mia / '20261008_第3回MIA_固定5bitと非会員較正'
    r4 = mia / '20261010_第4回MIA_提出原本と固定5bit照合'
    original = normalized_dir(repo / '公式データ', 'PWSCup2026_final_team28_ステテコは恥だが役に立つ')
    files = {
        'B_self.csv': original / 'B_self.csv',
        'aia_challenge_28.csv': repo / '公式データ/PWSCup2026_final_attack/aia_challenge_28.csv',
        'round1_F_mia.csv': r1 / '結果/F_mia.csv',
        'round2_F_mia.csv': r2 / '結果/F_mia.csv',
        'round3_F_mia.csv': r3 / '結果/F_mia.csv',
        'round4_F_mia.csv': r4 / 'F_mia.csv',
        'round5_F_mia.csv': round5,
        'round1_detailed_results.html': r2 / '採点結果/第1回_detailed_results.html',
        'round2_detailed_results.html': r2 / '採点結果/第2回_detailed_results.html',
        'round3_detailed_results.html': repo / '攻撃フェーズ/AIA/第3回/detailed_results.html',
        'round4_detailed_results.html': repo / '攻撃フェーズ/AIA/第4回/detailed_results.html',
        'fixed_plan.json': r3 / '結果/設定.json',
        'fixed_code.csv': r3 / '結果/固定5bit符号.csv',
        'official_attack_score.py': OFFICIAL_KIT / 'starter/src/pwscup2026/scoring/attack_score.py',
        'official_scoring_io.py': OFFICIAL_KIT / 'starter/src/pwscup2026/kit/codabench/scoring_io.py',
        'official_result.py': OFFICIAL_KIT / 'starter/src/pwscup2026/scoring/result.py',
        'round4_provenance.json': r4 / '出典と照合.json',
    }
    INPUT.mkdir(exist_ok=True)
    manifest = {}
    for name, source in files.items():
        data = source.read_bytes()
        if name == 'round5_F_mia.csv':
            assert git_blob(data) == BLOB5, 'Local round5 differs from GitHub fixed commit blob'
        (INPUT / name).write_bytes(data)
        manifest[name] = {'source_path': str(source), 'sha256': sha(data), 'git_blob_sha1': git_blob(data), 'bytes': len(data)}
    manifest['official_scoring_io.py'].update(github_blob='6cf93250763b7ba51fc6e29050a275a0222d8e42',
        github_ref='main-attack-20260928', github_url='https://github.com/pwscup/pwscup2026-kit/blob/main-attack-20260928/starter/src/pwscup2026/kit/codabench/scoring_io.py')
    assert manifest['official_scoring_io.py']['git_blob_sha1'] == manifest['official_scoring_io.py']['github_blob']
    manifest['round5_F_mia.csv'].update(github_commit=COMMIT5, github_blob=BLOB5,
        github_url=f'https://github.com/smsaku00/pwscup2026-sutehaji/blob/{COMMIT5}/攻撃フェーズ/MIA/5回目/F_mia.csv')
    dump('input_manifest.json', manifest)


def verify_inputs():
    manifest = json.loads((HERE / 'input_manifest.json').read_text())
    for name, info in manifest.items():
        data = (INPUT / name).read_bytes()
        assert (len(data), sha(data), git_blob(data)) == (info['bytes'], info['sha256'], info['git_blob_sha1'])
    assert manifest['round5_F_mia.csv']['git_blob_sha1'] == BLOB5
    assert manifest['official_attack_score.py']['sha256'] == '1e62a591bf5f564eef30bd64edb4e8146563ab497a249ac15230ef61178186c6'
    assert manifest['official_scoring_io.py']['sha256'] == '32df3a36762ca876449f6c635f4ed8d5fc3ecb5f322697ff167d6c17949d5a9b'
    return manifest


def read_csv(name):
    with (INPUT / name).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(set(reader.fieldnames)) == len(reader.fieldnames)
        assert all(None not in r and None not in r.values() for r in rows)
        return reader.fieldnames, rows


def observed(r):
    source = (INPUT / f'round{r}_detailed_results.html').read_text()
    blocks = [json.loads(html.unescape(x)) for x in re.findall(r'<pre[^>]*>(.*?)</pre>', source, re.S)]
    reports = [o for o in blocks if 'mia' in o]
    assert len(reports) == 1
    assert reports[0]['constants']['low_fpr'] == .01
    return reports[0]['mia']['tpr_rare']


def original_partition():
    _, b = read_csv('B_self.csv')
    _, a = read_csv('aia_challenge_28.csv')
    assert len(b) == 1049 and [int(r['record_id']) for r in b] == list(range(1049))
    def key(r):
        return tuple(r[k] if k == 'prefecture' else Decimal(r[k]) for k in QI)
    kb, ka = [key(r) for r in b], [key(r) for r in a]
    assert len(set(kb)) == len(b) and len(set(ka)) == len(a) == 326
    private = {i for i, k in enumerate(kb) if k in set(ka)}
    rare = tuple(i for i, row in enumerate(b) if row['is_rare'] == '1')
    negative = tuple(sorted(set(rare) & private))
    candidates = tuple(sorted(set(rare) - private))
    assert (len(private), len(rare), len(negative), len(candidates)) == (163, 24, 13, 11)
    dump('candidate_partition.json', {'private': sorted(private), 'rare': rare, 'known_negative_rare': negative,
        'candidate_ids': candidates, 'match_method': 'seven QIs, exact Decimal equality, unique keys'})
    return rare, negative, candidates, private


def matrix(r):
    header, raw = read_csv(f'round{r}_F_mia.csv')
    assert header == ['record_id'] + list(map(str, TEAMS))
    assert len(raw) == 1049 and [int(x['record_id']) for x in raw] == list(range(1049))
    values = np.array([[float(x[str(t)]) for t in TEAMS] for x in raw])
    assert np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()
    return values


def load_official():
    pkg = types.ModuleType('independent_official')
    pkg.__path__ = [str(INPUT)]
    sys.modules[pkg.__name__] = pkg
    for stem, name in [('result', 'official_result.py'), ('attack_score', 'official_attack_score.py')]:
        spec = importlib.util.spec_from_file_location('independent_official.' + stem, INPUT / name)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    tree = ast.parse((INPUT / 'official_scoring_io.py').read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('_finite', '_nanmean')]
    assert len(selected) == 2
    env = {'np': np}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(INPUT / 'official_scoring_io.py'), 'exec'), env)
    return sys.modules['independent_official.attack_score'], env['_nanmean']


def count_score(n, m):
    return F(24 - m, 100 * (n - m)) if m else F(0)


def rederive_counts(n1, n2, obs):
    """All 0/1/2 states, <=11 positives, unknown number of active targets.

    Uses an exact integer lattice and two independent 3**11 half enumerations.
    The second score filters freshly generated first-score matches.
    """
    scale = lcm(*(count_score(n, m).denominator for n in n1 for m in (1, 2)))
    halves = []
    for begin in (0, 11):
        state = np.array(list(product(range(3), repeat=11)), dtype=np.int8)
        total, active = state.sum(1), (state != 0).sum(1)
        keep = total <= 11
        state, total, active = state[keep], total[keep], active[keep]
        weights = np.array([[int(count_score(n1[begin+j], m) * scale) for m in range(3)] for j in range(11)])
        score = sum(weights[j, state[:, j]] for j in range(11))
        halves.append((state, total, active, score))
    rs, rt, ra, rv = halves[1]
    lookup = defaultdict(list)
    for j in range(len(rs)):
        lookup[int(ra[j]), int(rv[j])].append(j)
    first = []
    ls, lt, la, lv = halves[0]
    for active in range(1, 12):
        target = round(obs[0] * active * scale)
        # At each possible denominator the only matching lattice point is this one.
        if abs(F(target, active * scale) - F(str(obs[0]))) > F(1, 10**15):
            continue
        for i in range(len(ls)):
            if int(la[i]) > active:
                continue
            for j in lookup.get((active - int(la[i]), target - int(lv[i])), ()):
                if int(lt[i]) + int(rt[j]) <= 11:
                    first.append(tuple(map(int, np.r_[ls[i], rs[j]])))
    first = sorted(set(first))
    assert all(abs(float(sum((count_score(n, m) for n, m in zip(n1, row)), F(0)) / sum(m > 0 for m in row)) - obs[0]) < 1e-15 for row in first)
    both = []
    records = []
    for row in first:
        active = sum(m > 0 for m in row)
        s2 = sum((count_score(n, m) for n, m in zip(n2, row)), F(0)) / active
        hit = abs(float(s2) - obs[1]) < 1e-15
        if hit:
            both.append(row)
        records.append(dict(zip(map(str, TEAMS), row), round2_exact=str(s2), round2_float=repr(float(s2)), matches_round2=hit))
    table('round1_count_candidates.csv', list(map(str, TEAMS)) + ['round2_exact', 'round2_float', 'matches_round2'], records)
    assert len(first) == 102 and len(both) == 1
    spacing = min(b-a for a, b in zip(sorted({F(r['round2_exact']) for r in records}), sorted({F(r['round2_exact']) for r in records})[1:]))
    return dict(zip(TEAMS, both[0])), {'first_candidates': len(first), 'both_candidates': len(both),
        'round1_integer_scale': scale, 'lattice_minimum_step': str(F(1, 11*scale)),
        'round2_candidate_minimum_gap': str(spacing), 'score_comparison_tolerance': '1e-15'}


def question_score(n, m, k):
    return F((24-m)*k, 100*m*(n-k))


def legal_product(groups):
    """Recursive, disjoint subset enumeration; no decoder/old results imported."""
    def step(j, used, current):
        if j == len(groups):
            yield tuple(current)
            return
        for group in groups[j]:
            if used.isdisjoint(group):
                yield from step(j+1, used | set(group), current + [group])
    return step(0, set(), [])


def min_gap(values):
    s = sorted(set(values))
    return str(min(b-a for a, b in zip(s, s[1:]))) if len(s) > 1 else None


def run():
    manifest = verify_inputs()
    rare, negative, candidates, private = original_partition()
    matrices = {r: matrix(r) for r in range(1, 6)}
    obs = [observed(r) for r in range(1, 5)]
    scorer, nanmean = load_official()
    n = {}
    for r in (1, 2):
        assert set(matrices[r][rare, :].ravel()) <= {.1, .4}
        assert (matrices[r][candidates, :] == .4).all()
        n[r] = list(map(int, (matrices[r][rare, :] == .4).sum(0)))
    counts, count_report = rederive_counts(n[1], n[2], obs[:2])
    active = tuple(t for t in TEAMS if counts[t] > 0)
    ms = tuple(counts[t] for t in active)
    total_members = sum(ms)
    high_counts = tuple(int((matrices[3][rare, TEAMS.index(t)] == .4).sum()) for t in active)
    assert total_members == 9 and len(active) == 6
    assert (active, ms, high_counts) == ((6,10,11,16,17,25), (1,2,1,2,1,2), (6,7,7,11,10,12))
    plan = json.loads((INPUT / 'fixed_plan.json').read_text())
    assert {int(r): tuple(q) for r, q in plan['fixed_queries'].items()} == FIXED_QUESTIONS
    queries = {r: set(q) for r, q in FIXED_QUESTIONS.items()}
    # All 528 rare cells for each submitted/planned CSV; exactly, without tolerance.
    checked = []
    for r in (3,4,5):
        for t in TEAMS:
            ci = TEAMS.index(t)
            high = queries[r] | set(negative[:high_counts[active.index(t)]-5]) if t in active else set()
            expected = np.array([.4 if i in high else .1 if t in active else 0 for i in rare])
            assert np.array_equal(matrices[r][rare, ci], expected), (r,t)
        checked.append({'round': r, 'checked': 24*len(TEAMS), 'mismatches': 0})
    assert (matrices[5][sorted(private-set(rare)), :] == 0).all()
    # Create prospective rare columns for rounds 6/7 in memory only.
    rare_matrices = {r: matrices[r][rare, :] for r in range(1,6)}
    for r in (6,7):
        v = np.zeros((24, len(TEAMS)))
        for t, high in zip(active, high_counts):
            h = queries[r] | set(negative[:high-5])
            v[:, TEAMS.index(t)] = [.4 if i in h else .1 for i in rare]
        rare_matrices[r] = v
    # Exhaustive response table (216 k vectors per query), independently of members.
    possibilities = list(product(*(range(m+1) for m in ms)))
    response = {ks: sum((question_score(h,m,k) for h,m,k in zip(high_counts,ms,ks)),F(0))/6 for ks in possibilities}
    assert len(response) == 216 and len(set(response.values())) == 216
    response_grid = {ks: int(s*54000) for ks,s in response.items()}
    assert all(F(response_grid[ks],54000)==s for ks,s in response.items())
    patterns = {}
    for r in (3,4):
        matches = [ks for ks,s in response.items() if abs(float(s)-obs[r-1]) < 1e-15]
        assert len(matches) == 1
        patterns[r] = matches[0]
    groups_after3 = [list(c for c in combinations(candidates,m) if len(set(c)&queries[3]) == k) for m,k in zip(ms,patterns[3])]
    after3 = sum(1 for _ in legal_product(groups_after3))
    groups = [list(c for c in g if len(set(c)&queries[4]) == k) for g,k in zip(groups_after3,patterns[4])]
    placements = list(legal_product(groups))
    assert len(placements) == 1296
    # Dependency check: with already-known counts, final decoding does not need
    # cross-target exclusivity. Enumerate even impossible overlapping placements.
    no_exclusion = Counter()
    for placement in product(*groups):
        signature = tuple(response_grid[tuple(len(set(g)&queries[r]) for g in placement)] for r in (5,6,7))
        no_exclusion[signature] += 1
    assert sum(no_exclusion.values()) == 60750 and max(no_exclusion.values()) == 1
    dump('exclusivity_dependency.json',{'counts_known':True,'overlapping_placements_tested':60750,
        'unique_final_observation_sequences':len(no_exclusion),'collision_groups':0,
        'note':'Exclusivity is used for the initial <=11 count model and the 1296 current legal set; fixed-code uniqueness with known counts survives removing it.'})
    cfg = {'scoring': {'low_fpr': .01}}
    mask_all = np.ones(24, dtype=bool)
    official_calls = 0
    # Every possible 1/2-person subset is scored with official code for every round.
    score_cache = {}
    for r in range(1,8):
        for t in TEAMS:
            for members in combinations(candidates, counts[t]) if counts[t] else [()]:
                truth = np.array([int(i in members) for i in rare])
                result = scorer.score_mia(rare_matrices[r][:, TEAMS.index(t)], truth, cfg, rare_mask=mask_all)
                s = result.breakdown['tpr_rare']
                official_calls += 1
                if counts[t]:
                    expected = count_score(n[r][TEAMS.index(t)],counts[t]) if r <= 2 else question_score(high_counts[active.index(t)],counts[t],len(set(members)&queries[r]))
                    assert abs(s-float(expected)) < 1e-15
                else:
                    assert np.isnan(s)
                score_cache[r,t,members] = s
    signatures = []
    branch = defaultdict(list)
    outputs = []
    max_float_error = 0.
    # FULL 1049-row official scoring of actual round5, for each legal possibility.
    # These rare-only truth vectors are synthetic witnesses, not true memberships.
    rare_mask = np.zeros(1049,dtype=bool)
    rare_mask[list(rare)] = True
    for pid, placement in enumerate(placements,1):
        exact, floats = {}, {}
        for r in range(1,8):
            score_list = [score_cache[r,t,placement[active.index(t)] if t in active else ()] for t in TEAMS]
            floats[r] = nanmean(score_list)
            exact[r] = sum((count_score(n[r][TEAMS.index(t)],m) if r<=2 else question_score(h,m,len(set(g)&queries[r])) for t,m,h,g in zip(active,ms,high_counts,placement)),F(0))/6
            max_float_error = max(max_float_error, abs(floats[r]-float(exact[r])))
            assert abs(floats[r]-float(exact[r])) < 1e-15
            if r<=4:
                assert floats[r] == obs[r-1], (r, repr(floats[r]), repr(obs[r-1]))
        full_scores = []
        for t in TEAMS:
            truth = np.zeros(1049,dtype=int)
            if t in active:
                truth[list(placement[active.index(t)])] = 1
            result = scorer.score_mia(matrices[5][:,TEAMS.index(t)],truth,cfg,rare_mask=rare_mask)
            full_scores.append(result.breakdown['tpr_rare'])
            official_calls += 1
        assert nanmean(full_scores) == floats[5]
        sig = tuple(exact[r] for r in (5,6,7))
        signatures.append(sig)
        branch[sig[0]].append((pid,sig[1],sig[2]))
        row = {'placement_id':pid, **{f'Team{t}':'+'.join(map(str,g)) for t,g in zip(active,placement)},
            'outside_22_targets': '+'.join(map(str,sorted(set(candidates)-set(i for g in placement for i in g))))}
        row.update({f'round{r}_exact':str(exact[r]) for r in (5,6,7)})
        row.update({f'round{r}_official_float':repr(floats[r]) for r in (5,6,7)})
        outputs.append(row)
    table('all_1296_placements_and_observations.csv',list(outputs[0]),outputs)
    final = defaultdict(list)
    for pid,sig in enumerate(signatures,1):
        final[sig].append(pid)
    collisions = [{'observations_exact':list(map(str,sig)), 'placement_ids':ids} for sig,ids in final.items() if len(ids)>1]
    dump('collision_check.json', {'tested_placements':len(placements),'unique_signatures':len(final),
        'collision_groups':len(collisions),'maximum_final_remaining':max(map(len,final.values())), 'collisions':collisions})
    branches = []
    for bid,(score, cases) in enumerate(sorted(branch.items()),1):
        sixth = Counter(c[1] for c in cases)
        pairs = Counter((c[1],c[2]) for c in cases)
        official_values = {outputs[p-1]['round5_official_float'] for p,_,_ in cases}
        assert len(official_values) == 1
        branches.append({'branch_id':bid,'round5_exact':str(score),'round5_official_float':next(iter(official_values)),
            'round5_exact_fraction_as_float':repr(float(score)),
            'remaining_after_round5':len(cases),'round6_branches':len(sixth),'worst_remaining_after_round6':max(sixth.values()),
            'round6_7_response_pairs':len(pairs),'worst_remaining_after_round7':max(pairs.values()),
            'collision_pairs':sum(v>1 for v in pairs.values())})
    table('round5_all_branches.csv',list(branches[0]),branches)
    # Exhaustive worst branch witnesses and exact 6/7 response pairs.
    worst5=max(len(x) for x in branch.values())
    worst_details=[]
    for score,cases in sorted(branch.items()):
        if len(cases)==worst5:
            worst_details.append({'round5_exact':str(score),'round5_official_float':outputs[cases[0][0]-1]['round5_official_float'],'remaining':len(cases),
                'responses':[{'placement_id':p,'round6_exact':str(s6),'round7_exact':str(s7)} for p,s6,s7 in cases]})
    dump('worst_round5_branches.json',worst_details)
    # Rounding robustness: decoder receives only displayed decimals for all five questions.
    precision=[]
    for digits in range(3,11):
        buckets=defaultdict(list)
        for pid,row in enumerate(outputs,1):
            key=tuple(format(float(row[f'round{r}_official_float']),f'.{digits}f') for r in (5,6,7))
            buckets[key].append(pid)
        unresolved=[(key,ids) for key,ids in buckets.items() if len(ids)>1]
        worst=max(map(len,buckets.values()))
        witness=None
        if unresolved:
            key,ids=max(unresolved,key=lambda x:len(x[1]))
            witness={'displayed_round5_6_7':key,'placement_ids':ids[:2], 'placements':[outputs[i-1] for i in ids[:2]]}
        precision.append({'decimal_places':digits,'unique_observation_sequences':len(buckets),
            'collision_groups':len(unresolved),'worst_final_remaining':worst,'example':witness})
    dump('rounding_precision.json',precision)
    # The entire fixed wordbook, not just this observed branch, separates all singles/pairs.
    words={i:tuple(int(i in queries[r]) for r in range(3,8)) for i in candidates}
    pairwords={g:tuple(sum(words[i][j] for i in g) for j in range(5)) for g in combinations(candidates,2)}
    assert len(set(words.values()))==11 and len(set(pairwords.values()))==55
    code=read_csv('fixed_code.csv')[1]
    assert {int(x['record_id']):x['bits'] for x in code}=={i:''.join(map(str,v)) for i,v in words.items()}
    response_by_prefix=[len(set(tuple(s[:k]) for s in signatures)) for k in (1,2,3)]
    report={'status':'PASS','conclusion':'条件付きで必ず特定できる。予定第6・7回の事前固定質問で十分。',
        'round5_submitted':False,'counts':{str(t):m for t,m in counts.items()},'count_rederivation':count_report,
        'official_observations_round1_to_4':obs,'round3_and_4_positive_high_counts':{str(r):patterns[r] for r in (3,4)},
        'past_4_official_float_means_exactly_reproduced': True,
        'rare_cell_checks':checked,'round5_padding_cells':len(negative)*len(active),
        'nonrare_known_negative_cells_zero':150*len(TEAMS),'all_legal_placements_before_round3':factorial(11)//(2**4),
        'remaining_after_round3':after3,'remaining_after_round4':len(placements),
        'round5_observation_branches':len(branch),'worst_remaining_after_round5':worst5,
        'round5_branch_size_distribution': dict(sorted(Counter(len(cases) for cases in branch.values()).items())),
        'worst_remaining_after_round6':max(x['worst_remaining_after_round6'] for x in branches),
        'worst_remaining_after_round7':max(x['worst_remaining_after_round7'] for x in branches),
        'observation_prefix_counts_round5_to_7':response_by_prefix,'collision_groups':len(collisions),
        'query_average_minimum_gap_all_216':min_gap(response.values()),'round5_legal_response_minimum_gap':min_gap(branch),
        'single_word_count':11,'unique_pair_word_count':55,'official_scorer_calls':official_calls,
        'max_exact_vs_float_error':max_float_error,'minimum_sufficient_decimal_places_for_these_1296':next(x['decimal_places'] for x in precision if x['worst_final_remaining']==1),
        'known_counts_without_exclusivity_final_unique_placements': len(no_exclusion),
        'assumptions':['保存CSVと公式結果の回番号対応が正しい（サーバー入力ZIPのhash未確認）',
            'is_rare列が公式希少マスクに対応する','自AIA課題のB一致163人は非共有で、希少13人も全相手で非会員',
            '希少1人の共有先は最大1対象、各対象0/1/2人、合計11人以下',
            '第1・2回の人数推定、22対象、真の所属が第1〜7回で固定',
            'FPR=0.01、同点一括ROC補間、陽性0はnan除外という公式採点が維持',
            '第6・7回も保存5bitの希少528セルをpadding込みで使う',
            '観測にdetailed_results.htmlの生のmia.tpr_rareを使う（4桁だけでは保証しない）'],
        'python':sys.version,'python_executable':sys.executable,'numpy':np.__version__,'platform':platform.platform(),
        'input_sha256':{k:v['sha256'] for k,v in manifest.items()}}
    assert not collisions and report['worst_remaining_after_round7']==1
    dump('audit_results.json',report)
    dump('output_manifest.json',{p.name:sha(p.read_bytes()) for p in sorted(HERE.iterdir()) if p.is_file() and p.name!='output_manifest.json'})
    print(json.dumps({k:report[k] for k in ['status','conclusion','remaining_after_round3','remaining_after_round4',
        'round5_observation_branches','worst_remaining_after_round5','worst_remaining_after_round6','worst_remaining_after_round7',
        'collision_groups','official_scorer_calls','minimum_sufficient_decimal_places_for_these_1296']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--snapshot-repo',type=Path)
    parser.add_argument('--round5',type=Path)
    args=parser.parse_args()
    if args.snapshot_repo:
        assert args.round5
        snapshot(args.snapshot_repo,args.round5)
    run()
