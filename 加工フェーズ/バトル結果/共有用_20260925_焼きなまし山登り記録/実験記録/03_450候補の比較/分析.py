"""450 independent one-swap candidates, full utility and 12 AIA attacks.

Run with --run. Fixed inputs, deterministic candidates and resumable checkpoints.
No model API, no time limit, no cumulative swaps, no new panels.
"""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse, ast, fcntl, hashlib, importlib.util, inspect, json, multiprocessing
import shutil, subprocess, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
HILL = HERE.parent / '山登り_time_3ベース'
SA1 = HERE.parent / 'SA1_3候補_追加AIA比較'
PREV = ROOT / '本戦/バトル/0923'
PYTHON = ROOT / 'スターターキット/初版/starter/.venv/bin/python'
BASES = ['Zhang11', 'Zhang04', 'Seo03']
ATTRS = ['time', 'age', 'SBP', 'TG', 'HDL', 'ALT']
SETTINGS = dict(workers=16, candidates_per_attribute=25, attributes=ATTRS,
    seed=2026092402, total=450, total_swap_count=1, starting_point='original C, not SA1',
    cumulative=False, time_limit=None, utility='full official self_score for every candidate',
    utility_floor='own original U', objective='max weighted AIA over all 12 attacks',
    constant_in_objective=False, tolerance=1e-12, panel='old01: 387 targets + 387 controls',
    time_pair_rule='same onset/death; exclude (0,0); different values',
    other_pair_rule='any two rows with different values; uniform unordered pairs without replacement',
    phi_applied=False, independent_validation=False, base_order=BASES,
    previous_pilot_results_reused=False, optimization_version='shared-v1')
S = None

def load(path): return json.loads(Path(path).read_text())
def read(path): return pd.read_csv(path, float_precision='round_trip')
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
        default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + '\n')
    temp.replace(path)
def out(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    pd.DataFrame(rows).to_csv(temp, index=False, encoding='utf-8-sig', float_format='%.17g')
    temp.replace(path)
def say(message): print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)
def copy(src, dst):
    dst = Path(dst); dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists(): assert sha(src) == sha(dst), str(dst)
    else: shutil.copyfile(src, dst)
    return str(dst)
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec); sys.modules[name] = obj
    spec.loader.exec_module(obj); return obj
def bar(done, total, width=24):
    filled = int(width * done / total)
    return '[' + '#' * filled + '-' * (width - filled) + f'] {done}/{total} ({100*done/total:.1f}%)'
def assert_hashes(p):
    assert p['engine_sha256'] == sha(__file__), 'Engine changed after plan freeze'
    for f, h in p['hashes'].items(): assert sha(f) == h, f

def prepare():
    if (HERE / '計画.json').exists():
        p = load(HERE / '計画.json'); assert p['settings'] == SETTINGS
        assert_hashes(p); return p
    hp = load(HILL / '計画.json'); sp = load(SA1 / '計画.json')
    assert load(SA1 / '検証結果.json')['passed']
    for f, h in hp['hashes'].items(): assert sha(f) == h, f
    for f, h in sp['hashes'].items(): assert sha(f) == h, f
    hashes = {}; variants = []; attacks = []; baseline_predictions = {}
    def freeze(src, dst):
        path = copy(src, dst); hashes[str(src)] = sha(src); hashes[path] = sha(path); return path
    for v in hp['variants']:
        cp = freeze(v['path'], HERE / 'bases' / v['label'] / 'C.csv')
        variants.append(dict(v, path=cp))
    panel = dict(hp['panel'])
    for k, sub in [('challenge', 'public'), ('truth', 'evaluation_private')]:
        panel[k] = freeze(panel[k], HERE / sub / Path(panel[k]).name)
    bg = freeze(hp['a_bg'], HERE / 'public/A_bg.csv')
    ab = freeze(hp['attack_b'], HERE / 'public/B_attacker.csv')
    b = freeze(hp['b'], HERE / 'utility_inputs/B.csv')
    for filename in ['B_self.csv', 'utility_ref.json']:
        freeze(Path(hp['b']).with_name(filename), HERE / 'utility_inputs' / filename)
    for a in sp['attacks']:
        src = PREV / a['path']; assert sha(src) == a['sha256']
        cp = freeze(src, HERE / 'programs' / a['id'] / src.name)
        for asset in a['assets']:
            f = src.parent / asset['name']; assert sha(f) == asset['sha256']
            freeze(f, Path(cp).parent / asset['name'])
        attacks.append(dict(a, local_path=cp))
    provenance = load(SA1 / '予測の来歴.json')
    for base in BASES:
        baseline_predictions[base] = {}
        for a in attacks:
            rec = next(r for r in provenance if r['candidate'] == base + '-SA1'
                and r['kind'] == 'original' and r['attack'] == a['id'])
            assert sha(rec['path']) == rec['sha256']
            baseline_predictions[base][a['id']] = freeze(rec['path'],
                HERE / 'baseline_cli' / base / a['id'] / 'F_aia.csv')
    for f in list((Path(hp['kit']) / 'starter/src/pwscup2026').rglob('*.py')) + [
        Path(hp['kit']) / 'participant_data/scoring_config.yaml', HILL / '計画.json',
        SA1 / '計画.json', SA1 / '予測の来歴.json', SA1 / '検証結果.json', SA1 / '集計.json']:
        hashes[str(f)] = sha(f)
    p = dict(created=time.strftime('%Y-%m-%dT%H:%M:%S%z'), settings=SETTINGS,
        engine_sha256=sha(__file__), variants=variants, attacks=attacks, panel=panel,
        a_bg=bg, attack_b=ab, b=b, kit=hp['kit'], hashes=hashes,
        baseline_predictions=baseline_predictions, sa1_summary=load(SA1 / '集計.json'),
        versions=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__))
    save(HERE / '計画.json', p); return p

def candidates(p):
    rows = []; seen_hashes = set()
    for v in p['variants']:
        original = read(v['path']); assert len(original) == 1049 and not original.isna().any().any()
        for ai, attr in enumerate(ATTRS):
            seed = SETTINGS['seed'] + ai * 100; rng = np.random.default_rng(seed); seen = set()
            selected = []
            while len(selected) < 25:
                i, j = sorted(map(int, rng.choice(len(original), 2, replace=False)))
                if (i, j) in seen or original.at[i, attr] == original.at[j, attr]: continue
                if attr == 'time':
                    ki = tuple(original.loc[i, ['onset', 'death']]); kj = tuple(original.loc[j, ['onset', 'death']])
                    if ki != kj or ki == (0, 0): continue
                seen.add((i, j)); selected.append((i, j))
            for index, (i, j) in enumerate(selected, 1):
                cid = f'{v["label"]}-{attr}-{index:03d}'
                c = original.copy(); c.loc[[i, j], attr] = original.loc[[j, i], attr].to_numpy()
                path = HERE / 'candidates' / cid / 'C.csv'
                if not path.exists(): out(path, c)
                saved = read(path); pd.testing.assert_frame_equal(c, saved)
                assert (original != saved).sum().sum() == 2
                pd.testing.assert_frame_equal(original.drop(columns=attr), saved.drop(columns=attr))
                np.testing.assert_array_equal(np.sort(original[attr]), np.sort(saved[attr]))
                h = sha(path); assert h not in seen_hashes; seen_hashes.add(h)
                rows.append(dict(id=cid, base=v['label'], attribute=attr, index=index, seed=seed,
                    row1=i + 1, row2=j + 1, old1=float(original.at[i, attr]), old2=float(original.at[j, attr]),
                    total_swaps=1, path=str(path), C_sha256=h))
    assert len(rows) == 450
    # Interleave bases and attributes so each group appears early in progress.
    rows.sort(key=lambda r: (r['index'], ATTRS.index(r['attribute']), BASES.index(r['base'])))
    target = HERE / '候補計画.json'
    if target.exists(): assert load(target) == rows
    else: save(target, rows)
    out(HERE / '候補計画.csv', rows); return rows

class Scorer:
    def __init__(self, p):
        self.p = p; self.truth = read(p['panel']['truth']); self.q = read(p['panel']['challenge'])
        np.testing.assert_array_equal(self.q.challenge_row_id, self.truth.challenge_row_id)
        self.member = self.truth.label.eq('member').to_numpy()
        self.weights = self.truth.weight_163.to_numpy(float); self.times = self.truth.time.to_numpy(float)
        assert self.member.sum() == 387 and (~self.member).sum() == 387
        np.testing.assert_allclose([self.weights[self.member].sum(), self.weights[~self.member].sum()], [163, 163], atol=1e-10, rtol=0)
        self.mods = {}
        for a in p['attacks']:
            src = Path(a['local_path'])
            self.mods[a['id']] = module('pilot_' + a['id'], src.parent / 'official_aia.py' if a['id'] == 'A12' else src)
        sys.path.insert(0, str(Path(p['kit']) / 'starter/src'))
        from pwscup2026.kit.selfscore import self_score
        from pwscup2026.scoring.utility import UtilityReference
        self.full_score = self_score; self.b = read(p['b'])
        rr = load(Path(p['b']).with_name('utility_ref.json')); bs = read(Path(p['b']).with_name('B_self.csv'))
        self.ref = UtilityReference(mu=np.array(rr['mu']), sigma=np.array(rr['sigma']), band=rr['band'],
            b_is_rare=bs.is_rare.to_numpy(bool), km_signal_D=rr['km_signal_D'], km_floor=rr['km_floor'], a_bg=read(p['a_bg']))
        self.cfg = yaml.safe_load((Path(p['kit']) / 'participant_data/scoring_config.yaml').read_text())
        self.initialize_shared()
    def initialize_shared(self):
        # Share parsers only when function code and every relevant schema agree.
        for group in [('A01', 'A02', 'A03'), ('A08', 'A09', 'A10', 'A11')]:
            first = self.mods[group[0]]
            for aid in group[1:]:
                other = self.mods[aid]
                assert ast.dump(ast.parse(inspect.getsource(first.read_table))) == ast.dump(ast.parse(inspect.getsource(other.read_table)))
                keys = ['C_COLS', 'Q_COLS', 'BG_COLS', 'RANGES', 'PREFECTURES', 'HORIZON'] if group[0] == 'A01' else ['COLS', 'QI', 'RANGES', 'HORIZON']
                for k in keys: assert getattr(first, k) == getattr(other, k), (aid, k)
        self.fixed = {}; self.models = {}; self.shared_c = {}; self.cost = None
        for aid in ['A01', 'A04', 'A05', 'A06', 'A08']:
            m = self.mods[aid]; kind = 'background' if aid in ['A01', 'A08'] else 'A'
            q = m.read_table(Path(self.p['panel']['challenge']), 'challenge')
            bg = m.read_table(Path(self.p['a_bg']), kind)
            if aid == 'A01': q, bg = q[0], bg[0]
            self.fixed[aid] = (q, bg)
        self.ttscale = self.mods['A01'].scale_from_background(self.fixed['A01'][1])
        self.scaler = self.mods['A05'].Scaler(self.fixed['A05'][1])
        self.transform = self.mods['A06'].BackgroundTransform(self.fixed['A06'][1])
        for aid in ['A08', 'A09', 'A10', 'A11']:
            self.models[aid] = self.mods[aid].Attack(self.fixed['A08'][1])
        self.raw_assignment = self.mods['A06'].assignment_cost
        self.mods['A06'].assignment_cost = self.shared_assignment
        self.mods['A07'].CORE.assignment_cost = self.shared_assignment
        self.a5_weights = self.mods['A05'].local_weights(*self.fixed['A05'], self.scaler)
        self.a5_windows = {}
    def shared_assignment(self, q, c, transform):
        if self.cost is None: self.cost = self.raw_assignment(q, c, transform)
        # Callers receive their own array; no cross-attack mutation can leak.
        return self.cost.copy()
    def parsed_c(self, aid, cp):
        group = 'A01' if aid in ['A01', 'A02', 'A03'] else 'A06' if aid == 'A07' else 'A08' if aid in ['A08', 'A09', 'A10', 'A11'] else aid
        if group not in self.shared_c:
            d = self.mods[group].read_table(cp, 'c' if group == 'A01' else 'C')
            self.shared_c[group] = d[0] if group == 'A01' else d
        return self.shared_c[group]
    def fast_a5(self, c):
        # Same window candidates, summation order and lexicographic tie-break as CLI.
        # Sum each row contiguously, preserving numpy's original 1-D reduction.
        m = self.mods['A05']; q, a = self.fixed['A05']; ct = c.time.to_numpy(float); at = a.time.to_numpy(float)
        key = np.sort(ct).tobytes()
        if key not in self.a5_windows:
            times = np.r_[ct, at]
            centers = np.unique(np.clip(np.r_[times-m.TAU_TIME, times+m.TAU_TIME, m.HORIZON], 0., m.HORIZON))
            masses = np.column_stack([np.ascontiguousarray(self.a5_weights[:, np.abs(at-x) <= m.TAU_TIME+m.EPS]).sum(axis=1) for x in centers])
            self.a5_windows[key] = (centers, masses)
        centers, mass_a = self.a5_windows[key]
        wc = m.local_weights(q, c, self.scaler, m.c_novelty(c, a, self.scaler))
        best_s = np.full(len(q), -np.inf); best_m = best_s.copy(); answers = np.zeros(len(q))
        for j, center in enumerate(centers):
            mass = np.ascontiguousarray(wc[:, np.abs(ct-center) <= m.TAU_TIME+m.EPS]).sum(axis=1)
            score = mass - m.BACKGROUND_PENALTY * mass_a[:, j]
            # Centers lie in [0,HORIZON]; the third original tie-break prefers larger centers.
            better = (score > best_s) | ((score == best_s) & ((mass > best_m) | ((mass == best_m) & (center > answers))))
            best_s[better] = score[better]; best_m[better] = mass[better]; answers[better] = center
        return answers
    def metrics(self, pred):
        pred = np.asarray(pred, float)
        assert pred.shape == (774,) and np.isfinite(pred).all() and (pred >= 0).all() and (pred <= 10).all()
        hit = np.abs(pred - self.times) <= .5; m = self.member
        pm = float(np.average(hit[m], weights=self.weights[m])); pc = float(np.average(hit[~m], weights=self.weights[~m]))
        return dict(AIA=float(np.clip((pm - pc) / max(1 - pc, 1e-9), 0, 1)),
            target_rate=pm, control_rate=pc, target_hits=int(hit[m].sum()), control_hits=int(hit[~m].sum()))
    def predict(self, aid, cp):
        m = self.mods[aid]; qp = Path(self.p['panel']['challenge']); bp = Path(self.p['a_bg']); cp = Path(cp)
        if aid in ['A01', 'A02', 'A03']:
            c = self.parsed_c(aid, cp); q, bg = self.fixed['A01']
            pred = m.predict(c, q, bg, self.ttscale)[0]
            pred = np.array([float(format(x, m.FLOAT_FORMAT)) for x in pred])
        elif aid in ['A04', 'A05', 'A06', 'A07']:
            core = m.CORE if aid == 'A07' else m
            c = self.parsed_c(aid, cp); q, bg = self.fixed['A06' if aid == 'A07' else aid]
            if aid == 'A04': pred = m.aia_times(q, bg, c, m.Features(bg, c))
            elif aid == 'A05': pred = self.fast_a5(c)
            elif aid == 'A06': pred = m.aia_times(q, c, self.transform)[0]
            else: pred = m.relaxed_times(q, c, self.transform, 2)[0]
        elif aid in ['A08', 'A09', 'A10', 'A11']:
            c = self.parsed_c(aid, cp); q, bg = self.fixed['A08']
            pred = self.models[aid].predict(c, q)[0]
            pred = m.exact_override(q, c, np.asarray(pred, float))[0]
        else:
            cfg = {'aia': {'qi_continuous': ['age', 'SBP', 'TG', 'HDL', 'ALT'], 'block_keys': ['sex', 'prefecture'], 'k_nn_onset': 1}}
            _, pred = m.aia_predict(read(cp), read(qp), read(bp), cfg)
        return np.asarray(pred, float)
    def run(self, item):
        assert sha(item['path']) == item['C_sha256']
        start = time.monotonic(); cp = Path(item['path']); dst = HERE / 'results' / item['id']; dst.mkdir(parents=True, exist_ok=True)
        pred = {}; scores = {}; timings = {}; self.shared_c = {}; self.cost = None
        for a in self.p['attacks']:
            t = time.monotonic(); aid = a['id']; pred[aid] = self.predict(aid, cp)
            scores[aid] = self.metrics(pred[aid]); timings[aid] = time.monotonic() - t
        t = time.monotonic(); u = self.full_score(read(cp), self.b, self.ref, self.cfg)
        timings['full_U'] = time.monotonic() - t
        assert np.isfinite(u['utility']) and np.isfinite(list(u['facets'].values())).all()
        optimization_check = None
        if item.get('index') == 1 or item['id'].startswith('baseline-'):
            # Cover all 3 bases x 6 attributes, including reuse after other candidates.
            for aid in ['A05', 'A08', 'A09', 'A10', 'A11']:
                m = self.mods[aid]; kind = 'A' if aid == 'A05' else 'background'
                bg = m.read_table(Path(self.p['a_bg']), kind); q = m.read_table(Path(self.p['panel']['challenge']), 'challenge')
                c = m.read_table(cp, 'C')
                if aid == 'A05': expected = m.aia_times(q, bg, c, m.Scaler(bg))
                else:
                    expected = m.Attack(bg).predict(c, q)[0]
                    expected = m.exact_override(q, c, np.asarray(expected, float))[0]
                compare(pred[aid], np.asarray(expected), self.times)
            optimization_check = dict(passed=True, fresh_reference_attacks=['A05','A08','A09','A10','A11'])
        np.savez_compressed(dst / '予測.npz', challenge_row_id=self.q.challenge_row_id.to_numpy(), **pred)
        r = dict(id=item['id'], base=item['base'], C_sha256=item['C_sha256'], engine_sha256=self.p['engine_sha256'],
            utility=u, attacks=scores, max_AIA=max(s['AIA'] for s in scores.values()),
            seconds=time.monotonic() - start, timings=timings, optimization_check=optimization_check, predictions=str(dst / '予測.npz'),
            predictions_sha256=sha(dst / '予測.npz'))
        save(dst / '結果.json', r); return r

def initialize(p):
    global S
    S = Scorer(p)
def evaluate(item): return S.run(item)
def checkpoint(item, p):
    path = HERE / 'results' / item['id'] / '結果.json'
    if not path.exists(): return None
    r = load(path)
    assert r['C_sha256'] == item['C_sha256'] == sha(item['path']) and r['engine_sha256'] == p['engine_sha256']
    assert sha(r['predictions']) == r['predictions_sha256'] and len(r['attacks']) == 12
    return r
def read_prediction(path, p):
    d = read(path); q = read(p['panel']['challenge'])
    assert set(d.columns) == {'target_k', 'challenge_row_id', 'time_hat'}
    assert len(d) == 774 and d.target_k.eq(p['panel']['target_id']).all() and d.challenge_row_id.is_unique
    assert set(d.challenge_row_id) == set(q.challenge_row_id)
    return d.set_index('challenge_row_id').loc[q.challenge_row_id, 'time_hat'].to_numpy(float)
def compare(actual, expected, times):
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
    np.testing.assert_array_equal(np.abs(actual - times) <= .5, np.abs(expected - times) <= .5)

def validate_baselines(p, results):
    times = read(p['panel']['truth']).time.to_numpy(float)
    for v in p['variants']:
        if v['label'] not in results: continue
        r = results[v['label']]; z = np.load(r['predictions'])
        for a in p['attacks']: compare(z[a['id']], read_prediction(p['baseline_predictions'][v['label']][a['id']], p), times)
        np.testing.assert_allclose(r['utility']['utility'], v['utility']['U'], atol=1e-12, rtol=0)
        say(f'事前照合OK {v["label"]}: 12攻撃の予測・的中判定一致 / U={r["utility"]["utility"]:.6f} / 最大AIA={r["max_AIA"]:.6f}')
    save(HERE / '事前検証.json', dict(passed=True, bases=list(results), prediction_comparisons=12*len(results), exact_hit_masks=True, original_U_matches=True))

def cli_check(p, item, a):
    work = HERE / 'final_cli' / item['id'] / a['id']; work.mkdir(parents=True, exist_ok=True)
    script = Path(copy(a['local_path'], work / Path(a['local_path']).name))
    for asset in a['assets']: copy(Path(a['local_path']).parent / asset['name'], work / asset['name'])
    save(work / 'targets.json', [dict(target_k=p['panel']['target_id'], c_path=item['path'], aia_challenge_path=p['panel']['challenge'])])
    command = [str(PYTHON), '-B', str(script), '--b', p['attack_b'], '--a-bg', p['a_bg'], '--targets', str(work / 'targets.json')]
    with (work / '実行.log').open('w') as f:
        run = subprocess.run(command, cwd=work, stdout=f, stderr=subprocess.STDOUT)
    path = work / ('F_aia_' + script.stem + '.csv')
    assert run.returncode == 0 and path.exists(), str(work / '実行.log')
    expected = np.load(HERE / 'results' / item['id'] / '予測.npz')[a['id']]
    compare(read_prediction(path, p), expected, read(p['panel']['truth']).time.to_numpy(float))
    r = dict(candidate=item['id'], attack=a['id'], passed=True, command=command,
        prediction=str(path), prediction_sha256=sha(path), returncode=run.returncode)
    save(work / '検証.json', r); return r

def aggregate(p, items, results, baselines):
    rows = []; attacks = []
    for item in items:
        r = results[item['id']]; b = baselines[item['base']]; u = r['utility']; feasible = u['utility'] >= b['utility']['utility'] - 1e-12
        improved = r['max_AIA'] < b['max_AIA'] - 1e-12
        row = dict(item, U=u['utility'], original_U=b['utility']['utility'], U_feasible=feasible,
            max_AIA=r['max_AIA'], original_max_AIA=b['max_AIA'], AIA_change=r['max_AIA'] - b['max_AIA'],
            feasible_improvement=feasible and improved, limiting_U=min(u['facets'], key=u['facets'].get),
            seconds=r['seconds'], **u['facets'])
        rows.append(row)
        for a in p['attacks']:
            attacks.append(dict(candidate=item['id'], base=item['base'], attribute=item['attribute'],
                attack_id=a['id'], attack_name=a['name'], attack_group=a['group'], **r['attacks'][a['id']],
                original_AIA=b['attacks'][a['id']]['AIA'], delta=r['attacks'][a['id']]['AIA'] - b['attacks'][a['id']]['AIA']))
    out(HERE / '450候補一覧.csv', rows); out(HERE / '全攻撃結果.csv', attacks)
    df = pd.DataFrame(rows); groups = []
    for base in BASES:
        for attr in ATTRS:
            sub = df[df.base.eq(base) & df.attribute.eq(attr)]; ok = sub[sub.U_feasible]
            groups.append(dict(base=base, attribute=attr, candidates=len(sub), U_feasible=int(sub.U_feasible.sum()),
                feasible_improved=int(sub.feasible_improvement.sum()),
                max_AIA_min=float(sub.max_AIA.min()), max_AIA_median=float(sub.max_AIA.median()), max_AIA_max=float(sub.max_AIA.max()),
                best_feasible_AIA=None if ok.empty else float(ok.max_AIA.min()),
                U_min=float(sub.U.min()), U_max=float(sub.U.max())))
    out(HERE / '属性別集計.csv', groups)
    best = []
    for base in BASES:
        sub = df[df.base.eq(base) & df.feasible_improvement]
        if len(sub):
            row = sub.sort_values(['max_AIA', 'U', 'id'], ascending=[True, False, True]).iloc[0]
            best.append(next(i for i in items if i['id'] == row.id))
    return rows, groups, best

def report(p, rows, groups, best, baselines, elapsed, checks):
    s = Scorer(p); constant = s.metrics(np.full(774, 10.))['AIA']; save(HERE / '定数10年.json', dict(AIA=constant))
    lines = ['# 初回テスト：450候補・16並列', '',
        '- 【ローカル模擬実験】元のZhang11・Zhang04・Seo03から、1回だけ交換した候補を450個評価しました。',
        '- 各ベースについて、time・age・SBP・TG・HDL・ALTの6属性 × 25候補＝150候補です。',
        '- Zhang11 → Zhang04 → Seo03の順に、各ベース内で16並列処理しました。先行実行の32候補の点数は使わず、全て計算し直しました。',
        '- 全候補に12攻撃と公式コードによる有用性Uの自己採点を実施しました。',
        '- 標的387人＋対照387人の既存1パネルを使用。Zhangさん作成パネルではありません。',
        '- AIAは小さいほど良い指標です。Uが元案以上で、12攻撃の最大AIAが下がった候補だけを改善候補としました。',
        '- 450候補は独立です。良い交換を順に積み重ねる探索は、今回は行っていません。',
        f'- 実測の所要時間：{elapsed / 60:.1f}分（準備・照合・最終検証を含む）。', '',
        '## 元案・1回交換・既存SA1の比較', '',
        '| ベース | 元の最大AIA | 今回の最良1回交換 | 最大AIA | U | 改善候補数／150 | SA1の最大AIA（交換回数） |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: |']
    summaries = []
    for base in BASES:
        b = baselines[base]; chosen = next((i for i in best if i['base'] == base), None)
        row = next((r for r in rows if chosen and r['id'] == chosen['id']), None)
        n = sum(r['feasible_improvement'] for r in rows if r['base'] == base)
        oldsa = next(x for x in p['sa1_summary'] if x['name'] == base + '-SA1')
        count = dict(Zhang11=6, Zhang04=8, Seo03=2)[base]
        value = row['max_AIA'] if row else b['max_AIA']; u = row['U'] if row else b['utility']['utility']
        label = f'[{chosen["id"]}](最良候補/{base}-1swap.csv)' if chosen else '改善なし・元案を維持'
        lines.append(f'| {base} | {b["max_AIA"]:.6f} | {label} | {value:.6f} | {u:.6f} | {n} | {oldsa["all12_after"]["max"]:.6f}（{count}回） |')
        summaries.append(dict(base=base, original_max_AIA=b['max_AIA'], best_candidate=chosen['id'] if chosen else None,
            selected_max_AIA=value, selected_U=u, selected_total_swaps=1 if chosen else 0, improved_candidates=n,
            constant10_AIA=constant, max_AIA_minus_constant=value-constant, with_constant_max=max(value, constant),
            SA1_max_AIA=oldsa['all12_after']['max'], SA1_swaps=count))
        if chosen: copy(chosen['path'], HERE / '最良候補' / f'{base}-1swap.csv')
    out(HERE / 'ベース別まとめ.csv', summaries)
    lines += ['', '## どの属性が効いたか', '', '| ベース | 交換する属性 | Uを維持／25 | 改善／25 | Uを維持した中で最小の最大AIA |', '| --- | --- | ---: | ---: | ---: |']
    for g in groups:
        value = '—' if g['best_feasible_AIA'] is None else f'{g["best_feasible_AIA"]:.6f}'
        lines.append(f'| {g["base"]} | {g["attribute"]} | {g["U_feasible"]} | {g["feasible_improved"]} | {value} |')
    lines += ['', '## 最良候補への各攻撃の結果', '', '| 攻撃名 | Zhang11：元→選択 | Zhang04：元→選択 | Seo03：元→選択 |', '| --- | ---: | ---: | ---: |']
    for a in p['attacks']:
        cells = []
        for base in BASES:
            b = baselines[base]; chosen = next((i for i in best if i['base'] == base), None)
            r = load(HERE / 'results' / chosen['id'] / '結果.json') if chosen else b
            cells.append(f'{b["attacks"][a["id"]]["AIA"]:.6f} → {r["attacks"][a["id"]]["AIA"]:.6f}')
        lines.append('| ' + a['name'] + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## 読み取りの注意', '',
        f'- 常に10年と答える基準は **{constant:.6f}**。Cを交換しても変わりません。基準との差と、基準も含めた最大値は「ベース別まとめ.csv」に保存しています。',
        '- 各25候補で改善が見つからなくても、その属性で改善不可能とは言えません。試した交換に限った結果です。',
        '- 同じパネルで候補を選んだ結果です。本番の匿名性得点の予測値ではなく、別パネルでの確認も未実施です。φ補正とMIAは含みません。',
        '- 属性交換は各列の値の分布を保ちますが、属性間の関係が変わり、有用性Uが下がることがあります。',
        '- AIAは±0.5年を的中とし、標的・対照の率を138：12：9：4に重み付け。（標的率−対照率）÷（1−対照率）を0〜1に収めます。',
        '- 対照の真の希少区分は不明のため、対応する標的の重みを引き継ぐ既存方式です。', '',
        '## 保存物', '', '- [450候補の結果](450候補一覧.csv)／[各候補×12攻撃の5,400結果](全攻撃結果.csv)',
        '- [属性別の最小・中央値・最大](属性別集計.csv)／[ベース別まとめ](ベース別まとめ.csv)',
        '- [交換する行と再現用の乱数](候補計画.csv)／[前提と分析方法](説明.md)',
        '- [検証結果](検証結果.json)／[実行ログ](実行.log)', '']
    (HERE / 'README.md').write_text('\n'.join(lines))
    for item in best:
        r = load(HERE / 'results' / item['id'] / '結果.json')
        check = s.full_score(read(HERE / '最良候補' / f'{item["base"]}-1swap.csv'), s.b, s.ref, s.cfg)
        np.testing.assert_allclose(check['utility'], r['utility']['utility'], atol=1e-12, rtol=0)
    assert_hashes(p)
    save(HERE / '検証結果.json', dict(passed=True, candidates=450, attack_scores=5400, full_utility_scores=450,
        candidate_structure_validated=True, inputs_unchanged=True, baseline_predictions_compared=36,
        selected_candidate_cli_comparisons=len(checks), checks=checks, elapsed_seconds=elapsed,
        independent_panel_validation=False, previous_pilot_results_reused=False,
        optimization_parity_checks=21, sequential_base_order=BASES))
    return summaries

def run():
    start = time.monotonic(); lock = (HERE / '.実行.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    p = prepare(); state = HERE / '状態.json'
    if state.exists() and load(state)['status'] == 'complete':
        assert load(HERE / '検証結果.json')['passed']; say('450候補は完了済みです。README.mdを確認してください。'); return
    started_epoch = time.time()
    save(state, dict(status='preparing', pid=os.getpid(), completed=0, total=450, workers=16, started_epoch=started_epoch))
    say('新規実行。Zhang11 → Zhang04 → Seo03、各150候補を16並列。以前の途中結果は再利用しません。')
    items = candidates(p); baselines = {}; results = {}; errors = []; checks = []
    for item in items:
        r = checkpoint(item, p)
        if r: results[item['id']] = r
    stage_start = time.monotonic(); initial_done = len(results)
    def progress(stage, base, latest=None):
        counts = {name: sum(i['id'] in results for i in items if i['base'] == name) for name in BASES}
        n = len(results); fresh = n-initial_done
        eta = None if fresh < 16 else (time.monotonic()-stage_start)/fresh*(450-n)/60
        value = dict(status=stage, pid=os.getpid(), completed=n, total=450, workers=16, base=base,
            per_base=counts, base_total=150, started_epoch=started_epoch, updated_epoch=time.time(),
            errors=errors, latest=latest, elapsed_seconds=time.monotonic()-start, eta_minutes=eta)
        save(state, value)
        (HERE / '進捗.txt').write_text(f'全体 {bar(n,450)}\n' + '\n'.join(f'{name:7s} {bar(counts[name],150)}' for name in BASES) + '\n')
        return counts[base], eta
    with ProcessPoolExecutor(max_workers=16, mp_context=multiprocessing.get_context('spawn'), initializer=initialize, initargs=(p,)) as pool:
        for base in BASES:
            v = next(v for v in p['variants'] if v['label'] == base)
            progress('baseline', base); say(f'{base}: 元案の12攻撃・Uを新しく計算し、元CLIと照合します。')
            baseline = dict(id='baseline-' + base, base=base, path=v['path'], C_sha256=sha(v['path']))
            baselines[base] = checkpoint(baseline, p) or pool.submit(evaluate, baseline).result()
            validate_baselines(p, baselines)
            todo = [i for i in items if i['base'] == base and i['id'] not in results]
            progress('running', base); say(f'{base}: 150候補の採点開始（16並列）。全体 {bar(len(results),450)}')
            futures = {pool.submit(evaluate, item): item for item in todo}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    r = future.result(); results[item['id']] = r
                    b = baselines[base]; feasible = r['utility']['utility'] >= b['utility']['utility'] - 1e-12
                    improve = feasible and r['max_AIA'] < b['max_AIA'] - 1e-12
                    label = '改善候補' if improve else 'U不足' if not feasible else 'AIA同点/悪化'
                    count, eta = progress('running', base, item['id'])
                    say(f'全体 {bar(len(results),450)} | {base} {count}/150 | {item["attribute"]}-{item["index"]:03d} | U={r["utility"]["utility"]:.6f} | 最大AIA={r["max_AIA"]:.6f} | {label}')
                except Exception:
                    error = dict(candidate=item['id'], traceback=traceback.format_exc()); errors.append(error)
                    save(HERE / 'errors' / (item['id'] + '.json'), error); progress('running', base, item['id'])
                    say(f'採点エラー {item["id"]}（このベースの他の候補は続行）')
            assert not errors, f'{base}: failed candidates={len(errors)}'
            base_rows = []
            for item in items:
                if item['base'] != base: continue
                r = results[item['id']]; b = baselines[base]
                base_rows.append(dict(candidate=item['id'], attribute=item['attribute'], U=r['utility']['utility'],
                    max_AIA=r['max_AIA'], improvement=r['utility']['utility'] >= b['utility']['utility']-1e-12 and r['max_AIA'] < b['max_AIA']-1e-12))
            out(HERE / 'ベース別途中結果' / (base+'.csv'), base_rows)
            improved = sorted([r for r in base_rows if r['improvement']], key=lambda r: (r['max_AIA'], -r['U'], r['candidate']))
            if improved:
                chosen = next(i for i in items if i['id'] == improved[0]['candidate'])
                progress('validating_base', base, chosen['id']); say(f'{base}: 150/150採点完了。最良候補{chosen["id"]}を12個の元CLIで照合します。')
                with ThreadPoolExecutor(max_workers=16) as threads:
                    jobs = [threads.submit(cli_check, p, chosen, a) for a in p['attacks']]
                    for f in as_completed(jobs): checks.append(f.result())
            say(f'{base}: 完了。Uを維持して改善した候補={len(improved)}/150。')
    assert not errors and len(results) == 450, f'Failed candidates: {len(errors)}; completed={len(results)}'
    rows, groups, best = aggregate(p, items, results, baselines)
    progress('reporting', BASES[-1]); say('450/450採点と各ベースの元CLI照合が完了。結果をまとめています。')
    elapsed = time.monotonic() - start; summaries = report(p, rows, groups, best, baselines, elapsed, checks)
    save(state, dict(status='complete', completed=450, total=450, elapsed_seconds=elapsed, started_epoch=started_epoch,
        per_base={base:150 for base in BASES}, base=BASES[-1], base_total=150,
        selected=[i['id'] for i in best], completed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), summaries=summaries))
    for s in summaries: say(f'{s["base"]}: 元={s["original_max_AIA"]:.6f} → 選択={s["selected_max_AIA"]:.6f} / 改善候補{s["improved_candidates"]}個 / {s["best_candidate"] or "元案維持"}')
    say(f'完了。450候補・5,400攻撃評価・有用性450採点。所要{elapsed/60:.1f}分。結果: {HERE / "README.md"}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run', action='store_true'); args = parser.parse_args()
    if not args.run: print('実行: 分析.py --run（保存済みの結果を再利用して450候補まで継続）')
    else:
        try: run()
        except BlockingIOError:
            say('すでに別のプロセスで実行中です。新しい計算は開始しません。'); sys.exit(2)
        except Exception:
            traceback.print_exc()
            previous = load(HERE / '状態.json') if (HERE / '状態.json').exists() else {}
            save(HERE / '状態.json', dict(previous, status='failed', error=traceback.format_exc()))
            sys.exit(1)
