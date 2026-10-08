"""旧r2の数値背景感度と、転帰時間exact/nearの固定方式検証。

公開入力だけで全7背景×20対象の予測を作り終え、その後にだけ公式採点する。
個人別予測・ラベル・確定行集合は保存しない。旧mainも実行しない。
"""
from __future__ import annotations

import ast
import importlib.util
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
R2 = 'legacy_saved_work_20260901以前__prelim_attack__r2__F_mia'
EXACT = 'legacy_redteam_event_time_exact'
NEAR = 'legacy_redteam_event_time_near'
LAB = ['BMI', 'SBP', 'TG', 'HDL', 'ALT', 'FPG']
KEY = ['sex', 'prefecture', 'smoking', 'onset', 'death']


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def legacy_read(bc, path):
    """元r2と同じCSV parser。入力ハッシュも共通記録へ追加する。"""
    path = Path(path).resolve()
    bc.u.base.INPUT_HASHES[str(path)] = bc.sha(path)
    return pd.read_csv(path)


def r9(value):
    return np.round(np.asarray(value, float), 9)


def distance_matrix(b, cs, background, teams):
    mu = background[LAB].mean().to_numpy(float)
    sd = background[LAB].std(ddof=0).to_numpy(float)
    sd = np.where(sd > 0, sd, 1.)
    bz = (b[LAB].to_numpy(float) - mu) / sd
    values = []
    for team in teams:
        cz = (cs[team][LAB].to_numpy(float) - mu) / sd
        values.append(NearestNeighbors(n_neighbors=1).fit(cz).kneighbors(bz)[0][:, 0])
    return np.column_stack(values)


def fixed_public_gates(b, a1, a2, public, cs, teams):
    """Fisher/copy/帰無0確定集合はfull公開入力から一度だけ固定。"""
    background = pd.concat([a1, a2], ignore_index=True)
    continuous = LAB + ['time']
    pub = {c: set().union(*[set(r9(d[c])) for d in public]) for c in continuous}
    bset = {c: set(r9(b[c])) for c in continuous}
    unique, control_values = {}, {}
    for col in continuous:
        row_groups = {}
        for i, value in enumerate(r9(b[col])):
            row_groups.setdefault(value, []).append(i)
        unique[col] = {v: rows[0] for v, rows in row_groups.items()
                       if len(rows) == 1 and v not in pub[col]}
        counts = {}
        for frame in public:
            for value in r9(frame[col]):
                counts[value] = counts.get(value, 0) + 1
        control_values[col] = {v for v, n in counts.items() if n == 1 and v not in bset[col]}

    public_row_keys = set(map(tuple, r9(background[LAB].to_numpy(float))))
    index = {}
    for j, (key, age) in enumerate(zip(map(tuple, background[KEY].to_numpy(object)),
                                      background.age.to_numpy())):
        index.setdefault(key, []).append((age, j))
    rng = np.random.default_rng(0)
    controls = []
    for _ in range(6):
        chosen = []
        for key, age in zip(map(tuple, b[KEY].to_numpy(object)), b.age.to_numpy()):
            candidates = [j for a, j in index.get(key, []) if abs(a - age) <= 1]
            if candidates:
                chosen.append(rng.choice(candidates))
        controls.append(np.array(chosen))

    full_distances = distance_matrix(b, cs, background, teams)
    mu = background[LAB].mean().to_numpy(float)
    sd = background[LAB].std(ddof=0).to_numpy(float)
    sd = np.where(sd > 0, sd, 1.)
    confirmed, diagnostics = {}, []
    alpha = .01 / (len(continuous) * len(teams))
    for ti, team in enumerate(teams):
        c = cs[team]
        copy_rate = float(np.mean([key in public_row_keys for key in map(tuple, r9(c[LAB].to_numpy(float)))]))
        selected = set()
        fingerprint_count = 0
        if copy_rate < .95:
            for col in continuous:
                cv = set(r9(c[col]))
                hits = {unique[col][v] for v in cv & set(unique[col])}
                if not hits:
                    continue
                nh, nc = len(hits), len(control_values[col] & cv)
                pv = (fisher_exact([[nh, len(unique[col]) - nh],
                                    [nc, len(control_values[col]) - nc]], alternative='greater')[1]
                      if len(control_values[col]) >= 20 and len(unique[col]) >= 20 else 1.)
                if pv < alpha:
                    selected |= hits
            fingerprint_count = len(selected)
            nn = NearestNeighbors(n_neighbors=1).fit((c[LAB].to_numpy(float) - mu) / sd)
            null_distances = np.concatenate([
                nn.kneighbors((background.iloc[chosen][LAB].to_numpy(float) - mu) / sd)[0][:, 0]
                for chosen in controls
            ])
            for threshold in (.15, .05):
                if int((null_distances < threshold).sum()) == 0:
                    selected |= set(np.flatnonzero(full_distances[:, ti] < threshold).tolist())
                    break
        confirmed[team] = selected
        diagnostics.append(dict(team=team, public_copy_rate=copy_rate,
                                full_fingerprint_count=fingerprint_count,
                                full_confirmed_count=len(selected),
                                fixed_top_slots=max(4, 10 - len(selected))))
    return confirmed, diagnostics


def r2_confidence(b, rare, distances, confirmed, numeric_teams):
    """元round=1・県事前・nb・希少順を維持する。"""
    tokyo = (b.prefecture == '東京都').to_numpy()
    rare_nt = np.flatnonzero(rare & ~tokyo)
    rare_all = np.flatnonzero(rare)
    log_distance = -np.log(np.clip(distances, 1e-12, None))
    residual = (log_distance - log_distance.mean(1, keepdims=True)
                - log_distance.mean(0, keepdims=True) + log_distance.mean())
    scores = []
    for ti, team in enumerate(numeric_teams):
        value = residual[:, ti]
        rank = pd.Series(value).rank(pct=True).to_numpy()
        confidence = np.zeros(len(b))
        confidence[tokyo] = .02 + .08 * rank[tokyo]
        confidence[~tokyo] = .12 + .38 * rank[~tokyo]
        candidates = np.flatnonzero(~tokyo & ~rare)
        n_top = max(4, 10 - len(confirmed[team]))
        top = candidates[np.argsort(-value[candidates])[:n_top]]
        if len(top):
            confidence[top] = .70 + .10 * np.linspace(1, 0, len(top))
        nominee = rare_nt[(ti + 1) % len(rare_nt)]
        order = ([nominee] + [x for x in rare_nt if x != nominee]
                 + [x for x in rare_all if tokyo[x]])
        for position, ix in enumerate(order):
            confidence[ix] = .58 - .06 * position / max(len(order) - 1, 1)
        for ix in confirmed[team]:
            confidence[ix] = 1.
        scores.append(np.clip(confidence, 0., 1.))
    return np.column_stack(scores)


def outcome_functions(path):
    names = {'_event_code', 'event_time_exact', 'event_time_near'}
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {n.name for n in nodes} == names
    environment = dict(np=np, pd=pd, NearestNeighbors=NearestNeighbors,
                       D=SimpleNamespace(HORIZON=10.0))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), environment)
    return SimpleNamespace(**{name: environment[name] for name in names})


def main():
    bc = module('_legacy_background_common', HERE / 'background_common.py')
    started = time.perf_counter()
    data = bc.setup()  # 所属真値を読み込まない。
    source_r2 = ROOT / 'アーカイブ/予備戦_攻撃フェーズ/予備選分析_スクリプト/build_attack2.py'
    source_outcome = ROOT / 'アーカイブ/予備戦_攻撃フェーズ/work_20260901以前/outcome_mia_redteam.py'
    saved_path = ROOT / 'アーカイブ/予備戦_攻撃フェーズ/work_20260901以前/prelim_attack/r2/F_mia.csv'
    source_hashes = {str(p.relative_to(ROOT)): bc.sha(p)
                     for p in [Path(__file__), source_r2, source_outcome]}
    share = ROOT / '予備戦/加工フェーズ/分析/資料/加工コード_share_team05/data'
    b = legacy_read(bc, share / 'B.csv')
    rare = legacy_read(bc, share / 'B_self.csv').is_rare.to_numpy(bool)
    a1 = legacy_read(bc, share / 'A_bg.csv')
    a2 = legacy_read(bc, ROOT / 'スターターキット/初版/participant_data/A_bg.csv')
    public = [a1, a2,
              legacy_read(bc, ROOT / 'スターターキット/初版/participant_data/B_practice.csv'),
              legacy_read(bc, ROOT / 'スターターキット/改訂版_20260824/participant_data/B_practice.csv')]
    numeric_teams = sorted(data['teams'])  # 希少round配置は元の数値順を維持。
    cs = {team: legacy_read(bc, data['targets'] / f'C_{team}.csv') for team in numeric_teams}
    assert np.array_equal(b.record_id, data['b'].record_id)
    assert np.array_equal(rare, data['rare'])
    assert len(a1) == len(data['a']) == 5143 and len(a2) == 5143
    saved = legacy_read(bc, saved_path)
    assert np.array_equal(saved.record_id, b.record_id)
    outcome = outcome_functions(source_outcome)
    predictions = {}
    condition_details = []
    with threadpool_limits(limits=1):
        confirmed, gates = fixed_public_gates(b, a1, a2, public, cs, numeric_teams)
        exact = np.column_stack([outcome.event_time_exact(data['cs'][t], data['a'], data['b'])
                                 for t in data['teams']])
        saved_error = None
        for label, changed in bc.conditions(data):
            _, fraction, seed = next(spec for spec in bc.CONDITIONS if spec[0] == label)
            selected_a1 = (a1.copy() if fraction == 1.
                           else bc.u.cal.stratified_subset(a1, fraction, seed))
            assert len(selected_a1) == len(changed['a'])
            background = pd.concat([selected_a1, a2], ignore_index=True)
            distances = distance_matrix(b, cs, background, numeric_teams)
            raw_r2 = r2_confidence(b, rare, distances, confirmed, numeric_teams)
            if label == 'full':
                saved_error = float(np.max(np.abs(raw_r2 - saved[list(map(str, numeric_teams))].to_numpy())))
                assert saved_error < 1e-12, saved_error
            order = [numeric_teams.index(t) for t in data['teams']]
            predictions[label, R2] = raw_r2[:, order]
            predictions[label, EXACT] = exact.copy()
            predictions[label, NEAR] = np.column_stack([
                outcome.event_time_near(data['cs'][t], changed['a'], data['b']) for t in data['teams']
            ])
            condition_details.append(dict(condition=label, a1_rows=len(selected_a1),
                                          r2_a2_rows=len(a2), r2_scale_rows=len(background),
                                          event_near_background_rows=len(changed['a'])))
            bc.log(stage='legacy_predictions_complete', condition=label,
                   r2_scale_rows=len(background), seconds=time.perf_counter() - started)
    assert len(predictions) == 7 * 3
    assert all(np.array_equal(predictions['full', EXACT], predictions[label, EXACT])
               for label, _, _ in bc.CONDITIONS)
    for path, original in source_hashes.items():
        assert bc.sha(ROOT / path) == original
    specs = [
        dict(method=R2, family='旧r2固定公開gate・数値尺度背景感度',
             attributes=LAB, fixed_public_gates=True, fixed_round=1,
             full_public_pool='A1/A2/初版B_practice/改訂版B_practice',
             varying_reference='A1_condition+A2_full', full_gates_rows=10286,
             gate='Fisher Bonferroni .01/(7*20);copy_rate>=.95で確定行停止;帰無0近接集合full固定',
             top_slots='max(4,10-full_confirmed_count)', cross='元20C double-centering'),
        dict(method=EXACT, family='event時間完全一致', attributes=['time', 'onset', 'death'],
             background_independent=True, rtol=1e-9, atol=1e-10, administrative_horizon=10.,
             administrative_atol=.01, score_transform='元raw1049・jitterなし'),
        dict(method=NEAR, family='event時間近傍距離比', attributes=['time', 'onset', 'death'],
             varying_reference='A1_condition', c_neighbors=1, a_neighbors='min(2,n_A_event)',
             epsilon=1/36525., administrative_horizon=10., administrative_atol=.01,
             score_transform='元raw1049・jitterなし'),
    ]
    # 全21行列が確定し、ここまで所属真値を一度も読んでいない。
    bc.score_and_save(data, predictions, 'legacy_results.csv', 'legacy_run.json', specs, started,
                      score_kind='full_scores', extra=dict(
                          source_code_sha256=source_hashes,
                          r2_saved_full_confidence_max_error=saved_error,
                          r2_condition_reference_rows=condition_details,
                          r2_fixed_public_gate_diagnostics=gates,
                          r2_public_copy_judgment_fixed=True,
                          r2_a2_fixed_at_full_rows=5143,
                          r2_sensitivity_limitation='A2全体を残すため尺度変化が縮小。数値背景感度であり全gate依存を変える実験ではない。',
                          known_nonmember_or_normal_rank_recalibration=False,
                          exact_equal_across_all_seven_conditions=True,
                          labels_first_read='score_and_save after all 21 prediction matrices'))


if __name__ == '__main__':
    main()
