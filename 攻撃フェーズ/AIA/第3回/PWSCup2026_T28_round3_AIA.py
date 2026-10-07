"""PWSCup2026 / Team 28 / Round 3 AIA.

Public inputs: C_*.csv, aia_challenge_*.csv, A_bg.csv and the actual round-1/2
answer CSVs. Prior answers are explicit baselines, not opponent truth.
P01 is adapted from Tashiro's stable_matching/sex_quota/long75.
Python 3.10+, numpy, pandas, scipy. No network or submission.
See 第3回_AIA_説明書.md for the frozen team routes and input arguments.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse, csv, hashlib, io, json, platform, re, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import scipy
from scipy.spatial.distance import cdist

SEED=20260922
TAU=.5
HORIZON=10.
CENTERS=np.arange(.5,10.001,.5)
N3_CENTERS=np.arange(0,10.00001,.25)
NUM=['age','SBP','TG','HDL','ALT']
CAT=['sex','prefecture']
QI=NUM+CAT
COLS=['age','sex','prefecture','BMI','SBP','TG','HDL','ALT','smoking','FPG','time','onset','death']
RANGES={'age':(40,74),'BMI':(10,55),'SBP':(40,250),'TG':(1,3000),'HDL':(10,160),'ALT':(.1,700),'FPG':(40,400)}
EPS=1e-12
SOURCE_LIMIT=1200
BATCH=96
SHADOW_KINDS=('copy','noise','round','shuffle','pool','mixture')
SHADOW_REPEATS=2
SHADOW_ROWS=256
SHADOW_QUERY_PER_GROUP=48
N_TREES=160
ITERATIONS=64
BACKGROUND_WEIGHT=.90
DESKTOP=Path.home()/'Desktop'
DEFAULT_BG=DESKTOP/'pwscup2026-kit-main-process-20260912/participant_data/A_bg.csv'
DEFAULT_DATA=DESKTOP/'PWSCup2026_final_attack'


def natural_id(x, name):
    if isinstance(x, bool) or not re.fullmatch(r"[0-9]+", str(x)):
        raise ValueError(f"{name}: nonnegative integer required")
    value = int(x)
    if value > np.iinfo(np.int64).max:
        raise ValueError(f"{name}: int64 overflow")
    return value

def read_table(path, kind):
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f), [])
    required = ["challenge_row_id"] + QI if kind == "challenge" else COLS
    allowed = [set(required)]
    if kind == "background":
        allowed.append(set(COLS + ["record_id"]))
    if len(header) != len(set(header)) or set(header) not in allowed:
        raise ValueError(f"{path.name}: missing/duplicate/extra columns; truth columns are forbidden")
    d = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    if d.empty:
        raise ValueError(f"{path.name}: empty data")
    if "record_id" in d:
        # The public background's ID is not a model input.
        ids = [natural_id(x, "record_id") for x in d.record_id]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate background record_id")
    if "challenge_row_id" in d:
        d["challenge_row_id"] = [natural_id(x, "challenge_row_id") for x in d.challenge_row_id]
        if d.challenge_row_id.duplicated().any():
            raise ValueError("duplicate challenge_row_id")
    for col in required:
        if col == "challenge_row_id":
            continue
        if col == "prefecture":
            if not d[col].str.strip().str.len().gt(0).all():
                raise ValueError("empty prefecture")
            continue
        d[col] = pd.to_numeric(d[col], errors="raise")
        v = d[col].to_numpy(float)
        if not np.isfinite(v).all():
            raise ValueError(f"nonfinite {col}")
        if col in ("age","sex","smoking","onset","death") and not np.equal(v, np.rint(v)).all():
            raise ValueError(f"noninteger {col}")
        if col in ("sex","smoking","onset","death") and not np.isin(v, [0,1]).all():
            raise ValueError(f"invalid binary {col}")
        if col in RANGES and ((v < RANGES[col][0]) | (v > RANGES[col][1])).any():
            raise ValueError(f"out-of-range {col}")
        if col in ("age","sex","smoking","onset","death"):
            d[col] = v.astype(np.int64)
    if kind != "challenge":
        if not ((d.time > 0) & (d.time <= HORIZON)).all():
            raise ValueError("input time must be in (0,10]")
        if ((d.onset == 1) & (d.death == 1)).any():
            raise ValueError("onset and death cannot both be 1")
    order = ["challenge_row_id"] if kind == "challenge" else COLS
    return d[required].sort_values(order, kind="stable").reset_index(drop=True)

def subset(d, n, seed=SEED):
    if len(d) <= n:
        return d.reset_index(drop=True)
    ix = np.sort(np.random.default_rng(seed).choice(len(d), n, replace=False))
    return d.iloc[ix].reset_index(drop=True)

class Metric:
    def __init__(self, a):
        raw = self.raw(a)
        self.center = np.median(raw, axis=0)
        self.scale = np.quantile(raw,.75,axis=0)-np.quantile(raw,.25,axis=0)
        self.scale = np.where(self.scale > 1e-8, self.scale, np.maximum(raw.std(0),1.))
        self.levels = {col: sorted(a[col].astype(str).unique()) for col in CAT}

    @staticmethod
    def raw(d):
        x = d[NUM].to_numpy(float).copy()
        x[:,2] = np.log1p(x[:,2])
        x[:,4] = np.log1p(x[:,4])
        return x

    def numeric(self,d):
        return (self.raw(d)-self.center)/self.scale

    def features(self,d):
        features = [np.clip(self.numeric(d),-12,12)]
        for col in CAT:
            values = d[col].astype(str).to_numpy()
            features.append(np.column_stack([values == v for v in self.levels[col]]).astype(float))
            features.append((~np.isin(values,self.levels[col])).astype(float)[:,None])
        return np.column_stack(features)

    def distance(self,q,r):
        dist = cdist(self.numeric(q),self.numeric(r),"sqeuclidean")/len(NUM)
        for col in CAT:
            dist += (q[col].astype(str).to_numpy()[:,None] !=
                     r[col].astype(str).to_numpy()[None,:]).astype(float)
        return dist

class ResearchAttack:
    def __init__(self,a,s4):
            self.a=a;self.s4=s4;self.metric=s4.Metric(a)
            self.ref=s4.subset(a,min(len(a),1049),s4.SEED+1)
            self.probe=s4.subset(a,min(len(a),768),261006)
            self.bgtime=a.time.to_numpy(float)

    def raw(self,f):return self.metric.raw(f)

    def numeric(self,q,c,omit=None):
            use=[i for i,v in enumerate(NUM) if v!=omit]
            return cdist((self.raw(q)[:,use]-self.metric.center[use])/self.metric.scale[use],
                         (self.raw(c)[:,use]-self.metric.center[use])/self.metric.scale[use],'sqeuclidean')/len(use)

    def cost(self,q,c,omit=None,pref=True):
            d=self.numeric(q,c,omit)
            for col in (CAT if pref else ['sex']):d+=(q[col].to_numpy()[:,None]!=c[col].to_numpy()[None,:])
            return d

    def partial(self,c,q):
            probe=self.probe.copy()
            # Actual challenge is single-prefecture. Relabel proxy probes only if the region
            # is too sparse; preserve sex, age decade and numeric/time values.
            region=q.prefecture.mode().iloc[0]
            same=self.a[self.a.prefecture.eq(region)]
            imputed=len(same)<192
            if not imputed:probe=self.s4.subset(same,min(768,len(same)),261006)
            else:probe['prefecture']=region
            qviews=[];nviews=[]
            for omit in [None,*NUM]:
                qviews.append(self.cost(q,c,omit));nviews.append(self.cost(probe,c,omit).min(1))
            mins=np.stack([d.min(1) for d in qviews],axis=1);null=np.stack(nviews,axis=1)
            p=np.empty_like(mins);joint=np.empty(len(q));view=np.empty(len(q),int)
            # Empirical null is conditioned on sex + age decade where sample supports it.
            pairp=np.empty((len(q),len(c)));groups={}
            for i,row in enumerate(q.itertuples()):
                key=(int(row.sex),int(row.age)//10)
                if key not in groups:
                    mask=(probe.sex.to_numpy()==key[0])&(probe.age.to_numpy()//10==key[1])
                    if mask.sum()<24:mask=probe.sex.to_numpy()==key[0]
                    values=null[mask];sorts=[np.sort(values[:,v]) for v in range(6)];n=len(values)
                    nullp=np.column_stack([(np.searchsorted(sorts[v],values[:,v],'right')+.5)/(n+1) for v in range(6)])
                    merged=np.sort(nullp.min(1));groups[key]=(sorts,n,merged)
                sorts,n,merged=groups[key]
                p[i]=[(np.searchsorted(sorts[v],mins[i,v],'right')+.5)/(n+1) for v in range(6)]
                v=int(np.argmin(p[i]));view[i]=v
                joint[i]=(np.searchsorted(merged,p[i,v],'right')+.5)/(n+1)
                rawp=(np.searchsorted(sorts[v],qviews[v][i],'right')+.5)/(n+1)
                pairp[i]=(np.searchsorted(merged,rawp,'right')+.5)/(n+1)
            donor=np.array([qviews[v][i].argmin() for i,v in enumerate(view)])
            t=c.time.to_numpy()[donor]
            return t,joint,pairp,{'proxy_prefecture_imputed':imputed,'probe_rows':len(probe),
                'partial_p_lt010':int((joint<.1).sum()),'partial_p_lt030':int((joint<.3).sum()),
                'partial_p_median':float(np.median(joint)),
                'chosen_view_counts':dict(zip(['full',*NUM],map(int,np.bincount(view,minlength=6))))}

def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def csv_bytes(rows):
    stream=io.StringIO(newline='')
    writer=csv.writer(stream,lineterminator='\n')
    writer.writerow(['target_k','challenge_row_id','time_hat'])
    for k,r,t in rows:
        writer.writerow([str(k),str(r),format(t,'.17g')])
    return stream.getvalue().encode('utf-8')


from scipy.optimize import linear_sum_assignment

P01_TEAMS = {1, 2, 5, 6, 7, 8, 10, 12, 13, 16, 17, 22, 25, 26, 27}
ROUTE = {k: 'P01_partial2_sex_quota75_long95' for k in P01_TEAMS}
ROUTE.update({0: 'R2_restore_R1_nonzero', 15: 'R2_restore_R1_nonzero',
              3: 'R2_partial30_override', 4: 'R2_partial30_override',
              11: 'R2_S3_partial30_override', 14: 'R2_partial30_override',
              9: 'R2_N9_add_N4_positive14'})
EXPECTED_N9_ADDITIONS = [10, 30, 50, 58, 118, 157, 166, 170, 221, 224, 239, 247, 289, 293]


def p01_distance(q, c, scale):
    x, y = Metric.raw(q) / scale, Metric.raw(c) / scale
    distance = np.sort((x[:, None] - y[None]) ** 2, axis=2)[:, :, :2].mean(2)
    distance[q.sex.to_numpy()[:, None] != c.sex.to_numpy()[None]] = np.inf
    return distance


def p01_matrices(q, c, bg):
    raw = Metric.raw(bg)
    scale = np.maximum(np.quantile(raw, .75, axis=0) - np.quantile(raw, .25, axis=0), 1e-5)
    dc, db = p01_distance(q, c, scale), p01_distance(q, bg, scale)
    pair = np.empty_like(dc)
    for i, (row, ref) in enumerate(zip(dc, np.sort(db, axis=1))):
        finite = ref[np.isfinite(ref)]
        if not len(finite):
            raise ValueError('P01: missing same-sex public background')
        interpolated = np.interp(row, np.r_[0., finite],
                                 np.r_[0., np.arange(1, len(finite) + 1)], right=len(finite))
        pair[i] = -np.log((interpolated + .25) / (len(bg) + 1))
    pair[~np.isfinite(dc)] = -1e4
    full = cdist(Metric.raw(q) / scale, Metric.raw(c) / scale, 'sqeuclidean') / 5
    pair -= 1e-10 * np.minimum(full, 1000.)
    return pair, dc


def p01_assignment(pair, count):
    nq, nc = pair.shape
    if nc < count:
        raise ValueError('P01: insufficient same-sex C rows')
    dummy = nq - count
    size = nc + dummy
    cost = np.zeros((size, size), dtype=float)
    cost[:nq, :nc] = -pair
    cost[nq:, nc:] = 1e6
    ri, ci = linear_sum_assignment(cost)
    if not np.array_equal(ri, np.arange(size)):
        raise ValueError('P01: assignment row order changed')
    assigned = ci[:nq]
    chosen = assigned < nc
    if chosen.sum() != count or np.unique(assigned[chosen]).size != count:
        raise ValueError('P01: quota or donor uniqueness failed')
    return chosen


def p01_select(pair, dc, qs, cs):
    # Include every row tied with the 12th closest donor.
    limit = np.partition(dc, 11, axis=1)[:, 11]
    limited = np.where(np.isfinite(dc) & (dc <= limit[:, None]), pair, -1e4)
    selected = np.zeros(len(qs), dtype=bool)
    for sex in np.unique(qs):
        a, b = qs == sex, cs == sex
        selected[a] = p01_assignment(limited[a][:, b], int(np.ceil(a.sum() * .75)))
    return selected


def read_baseline(path):
    table = pd.read_csv(path, encoding='utf-8-sig', float_precision='round_trip')
    cols = ['target_k', 'challenge_row_id', 'time_hat']
    if list(table.columns) != cols or table.duplicated(cols[:2]).any():
        raise ValueError(f'{Path(path).name}: invalid baseline header or duplicate keys')
    for col in cols[:2]:
        values = table[col].to_numpy(float)
        if not np.isfinite(values).all() or not np.equal(values, np.rint(values)).all():
            raise ValueError('Noninteger baseline key')
        table[col] = values.astype(np.int64)
    keys = set(map(tuple, table[cols[:2]].to_numpy()))
    expected = {(k, r) for k in ROUTE for r in range(326)}
    if keys != expected or len(table) != 7172:
        raise ValueError('Baseline must cover exactly the 22 opponents and row IDs 0..325')
    values = table.time_hat.to_numpy(float)
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 10)).all():
        raise ValueError('Invalid baseline time')
    return table.set_index(cols[:2]).sort_index()


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--a-bg', type=Path, default=DEFAULT_BG)
    parser.add_argument('--round1-csv', type=Path, default=here.parent / '一回目/F_aia.csv')
    parser.add_argument('--round2-csv', type=Path, default=here.parent / '第2回/F_aia.csv')
    parser.add_argument('--output-dir', type=Path, default=here / 'recomputed')
    parser.add_argument('--rows-json', action='store_true', help='Write JSON rows for artifact authoring instead of CSV')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    destinations = [output / ('prediction_rows.json' if args.rows_json else 'F_aia.csv'),
                    output / 'execution_record.json']
    metadata_path = args.data_dir / 'mia_columns.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8-sig'))
    allowed = {natural_id(value, 'target_columns') for value in metadata['target_columns']} - {28}
    if allowed != set(ROUTE):
        raise ValueError('Opponent metadata differs from the round-3 frozen routes')
    inputs = [args.a_bg, metadata_path, args.round1_csv, args.round2_csv]
    inputs += [args.data_dir / f'{stem}_{k}.csv' for k in sorted(ROUTE) for stem in ('C', 'aia_challenge')]
    if any(dest.resolve() in {p.resolve() for p in inputs} for dest in destinations):
        raise ValueError('Output would replace an input')
    if not args.overwrite and any(p.exists() for p in destinations):
        raise FileExistsError('Output exists. Select a new --output-dir or use --overwrite')
    hashes = {str(p.resolve()): file_hash(p) for p in inputs}
    baseline1, baseline2 = read_baseline(args.round1_csv), read_baseline(args.round2_csv)
    bg = read_table(args.a_bg, 'background')
    if len(bg) != 5143:
        raise ValueError('Expected 5143 public main background rows')
    attack = ResearchAttack(bg, SimpleNamespace(Metric=Metric, subset=subset, SEED=SEED))
    rows, diagnostics = [], []
    start = time.perf_counter()
    for k in sorted(ROUTE):
        c = read_table(args.data_dir / f'C_{k}.csv', 'C')
        q = read_table(args.data_dir / f'aia_challenge_{k}.csv', 'challenge')
        if len(c) != 1049 or len(q) != 326 or not np.array_equal(q.challenge_row_id, np.arange(326)):
            raise ValueError(f'Team {k}: expected C1049 / challenge326 with IDs 0..325')
        ids = q.challenge_row_id.to_numpy(int)
        old = baseline2.loc[k].loc[ids, 'time_hat'].to_numpy(float)
        first = baseline1.loc[k].loc[ids, 'time_hat'].to_numpy(float)
        details = {}
        if k in P01_TEAMS:
            pair, dc = p01_matrices(q, c, bg)
            chosen = p01_select(pair, dc, q.sex.to_numpy(), c.sex.to_numpy())
            values = np.where(chosen, 9.5, 0.)
            details['selected'] = int(chosen.sum())
            # Validate the delivery against the published main-phase quota count.
            if chosen.sum() != 245:
                raise ValueError(f'Team {k}: P01 quota differs from this frozen release')
        elif k in (0, 15):
            mask = (old == 0) & (first > 0)
            values = np.where(mask, first, old)
            expected = 161 if k == 0 else 153
            if mask.sum() != expected:
                raise ValueError(f'Team {k}: unexpected baseline restore count')
            details['restored'] = int(mask.sum())
        else:
            pt, joint, pairp, pdiag = attack.partial(c, q)
            details.update(pdiag)
            if k == 9:
                costs = np.repeat(pairp, 2, axis=1)
                rr, cc = linear_sum_assignment(np.column_stack([costs, np.full((len(q), len(q)), .3)]))
                assigned = np.empty(len(q), dtype=int)
                assigned[rr] = cc
                real = assigned < len(c) * 2
                n4 = np.zeros(len(q))
                n4[real] = c.time.to_numpy(float)[assigned[real] // 2]
                mask = (old == 0) & (n4 > 0)
                if ids[mask].tolist() != EXPECTED_N9_ADDITIONS:
                    raise ValueError('Team 9: expected exactly the reviewed 14 additional row IDs')
                values = np.where(mask, n4, old)
                details['additional_row_ids'] = ids[mask].tolist()
            else:
                mask = joint < .3
                values = np.where(mask, pt, old)
                details['partial_overrides'] = int(mask.sum())
        if values.shape != (326,) or not np.isfinite(values).all() or not ((values >= 0) & (values <= 10)).all():
            raise ValueError(f'Team {k}: invalid predictions')
        change = values != old
        disjoint = np.abs(values - old) > 1.0
        details.update(target_k=k, method=ROUTE[k], rows=326,
                       changed_vs_round2=int(change.sum()), zero=int((values == 0).sum()),
                       nonzero=int((values > 0).sum()), late=int((values >= 9).sum()),
                       disjoint_hit_windows_vs_round2=int(disjoint.sum()))
        diagnostics.append(details)
        rows.extend((k, int(r), float(t)) for r, t in zip(ids, values))
        print(json.dumps(details, ensure_ascii=False), flush=True)
    if len(rows) != 7172 or len({(k, r) for k, r, _ in rows}) != 7172:
        raise ValueError('Incomplete or duplicated output keys')
    if any(file_hash(p) != hashes[str(p.resolve())] for p in inputs):
        raise ValueError('Input changed during computation')
    data = csv_bytes(rows)
    output.mkdir(parents=True, exist_ok=True)
    if args.rows_json:
        destinations[0].write_text(json.dumps(rows, allow_nan=False), encoding='utf8')
    else:
        destinations[0].write_bytes(data)
    record = dict(round=3, team_id=28, opponents=22, rows=7172, route=ROUTE,
                  input_hashes=hashes, script_sha256=file_hash(__file__),
                  python_csv_sha256=hashlib.sha256(data).hexdigest(),
                  seconds=round(time.perf_counter()-start, 3), diagnostics=diagnostics,
                  dependencies=dict(python=platform.python_version(), numpy=np.__version__,
                                    pandas=pd.__version__, scipy=scipy.__version__),
                  official_round3_score_available=False, submitted_to_competition=False)
    destinations[1].write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({'rows':7172, 'output':str(destinations[0])}, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        raise SystemExit(1)
