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
class Scorer:
    def __init__(self, p):
        self.p = p; self.truth = read(p['panel']['truth']); self.q = read(p['panel']['challenge'])
        np.testing.assert_array_equal(self.q.challenge_row_id, self.truth.challenge_row_id)
        self.member = self.truth.label.eq('member').to_numpy()
        self.weights = self.truth.weight_163.to_numpy(float); self.times = self.truth.time.to_numpy(float)
        assert self.member.sum() == p['panel']['n'] and (~self.member).sum() == p['panel']['n']
        np.testing.assert_allclose([self.weights[self.member].sum(), self.weights[~self.member].sum()], [163, 163], atol=1e-10, rtol=0)
        self.mods = {}
        for a in p['attacks']:
            src = Path(a['local_path'])
            self.mods[a['id']] = module('search_' + p['panel']['id'] + '_' + a['id'], src.parent / 'official_aia.py' if a['id'] == 'A12' else src)
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
        assert pred.shape == (len(self.q),) and np.isfinite(pred).all() and (pred >= 0).all() and (pred <= 10).all()
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
