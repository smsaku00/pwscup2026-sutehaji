"""PWSCup2026 Team 28, round 4: standalone public-input AIA generation.

Inputs: public A_bg.csv, C_<k>.csv, aia_challenge_<k>.csv, mia_columns.json.
No previous answers, cached predictions, private truth, network or scoring calls.
Python 3.10+; numpy, pandas, scipy, scikit-learn. See 第4回_AIA_説明書.md.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse, csv, hashlib, io, json, platform, re, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.special import ndtri
from sklearn.covariance import LedoitWolf

SEED = 20260922
HORIZON = 10.
CENTERS = np.arange(0, 10.00001, .25)
NUM = ['age','SBP','TG','HDL','ALT']
CAT = ['sex','prefecture']
QI = NUM + CAT
COLS = ['age','sex','prefecture','BMI','SBP','TG','HDL','ALT','smoking','FPG','time','onset','death']
RANGES = {'age':(40,74),'BMI':(10,55),'SBP':(40,250),'TG':(1,3000),'HDL':(10,160),'ALT':(.1,700),'FPG':(40,400)}
DESKTOP = Path.home() / 'Desktop'
DEFAULT_BG = DESKTOP / 'pwscup2026-kit-main-process-20260912/participant_data/A_bg.csv'
DEFAULT_DATA = DESKTOP / 'PWSCup2026_final_attack'
ROUTE = {0: 'N9_95_Ctime', 1: 'P01_65_95', 2: 'copula_window25', 3: 'P03_85_95', 4: 'P01_85_95', 5: 'P05_65_95', 6: 'CA_cap2_prior50_const95', 7: 'P03_75_95', 8: 'P01_50_95', 9: 'bin_95_Ctime', 10: 'CA_cap2_prior50', 11: 'CA_cap2_prior50', 12: 'N9_100_late95', 13: 'P01_50_95', 14: 'P03_85_95', 15: 'P05_50_95', 16: 'P04_85_95', 17: 'P03_75_95', 22: 'P04_85_95', 25: 'S4_85_Ctime', 26: 'CA_cap1_prior50', 27: 'P03_85_95'}
TEAMS = sorted(ROUTE)

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

def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def csv_bytes(rows):
    stream=io.StringIO(newline='')
    writer=csv.writer(stream,lineterminator='\n')
    writer.writerow(['target_k','challenge_row_id','time_hat'])
    for k,r,t in rows:
        writer.writerow([str(k),str(r),format(t,'.17g')])
    return stream.getvalue().encode('utf-8')

def gate(scores,fraction=.75):
    keep=np.zeros(len(scores),bool);keep[np.argsort(-scores,kind='stable')[:int(np.ceil(fraction*len(scores)))]]=True;return keep

def stable_window(prob):
    # Tie preference: later centers, deterministic. Not a default 10-year answer.
    return CENTERS[len(CENTERS)-1-np.argmax(prob[:,::-1],axis=1)]

def time_mass(d,t,k):
    order=np.argsort(d,axis=1,kind='stable')[:,:k]
    ds=np.take_along_axis(d,order,axis=1)
    # Keep separated nearest donors informative; soften only uncertain distances.
    h=np.maximum(ds[:,-1],1e-6)
    w=np.exp(-ds/np.maximum(.35*h[:,None],1e-6));w/=w.sum(1,keepdims=True)
    return np.einsum('nk,nkl->nl',w,(np.abs(t[order,None]-CENTERS[None,None,:])<=.5).astype(float))

class ResearchAttack:

    def __init__(self, a, s4):
        self.a = a
        self.s4 = s4
        self.metric = s4.Metric(a)
        self.ref = s4.subset(a, min(len(a), 1049), s4.SEED + 1)
        self.probe = s4.subset(a, min(len(a), 768), 261006)
        self.bgtime = a.time.to_numpy(float)

    def raw(self, f):
        return self.metric.raw(f)

    def numeric(self, q, c, omit=None):
        use = [i for i, v in enumerate(NUM) if v != omit]
        return cdist((self.raw(q)[:, use] - self.metric.center[use]) / self.metric.scale[use], (self.raw(c)[:, use] - self.metric.center[use]) / self.metric.scale[use], 'sqeuclidean') / len(use)

    def cost(self, q, c, omit=None, pref=True):
        d = self.numeric(q, c, omit)
        for col in CAT if pref else ['sex']:
            d += q[col].to_numpy()[:, None] != c[col].to_numpy()[None, :]
        return d

    def copula(self, q, c):
        raw = self.raw(self.a)

        def transform(f, reference):
            out = np.empty((len(f), len(NUM)))
            v = self.raw(f)
            for j in range(len(NUM)):
                r = np.sort(reference[:, j])
                rank = (np.searchsorted(r, v[:, j], 'left') + np.searchsorted(r, v[:, j], 'right')) / 2
                out[:, j] = ndtri(np.clip((rank + 0.5) / (len(r) + 1), 0.005, 0.995))
            return out
        b = transform(self.a, raw)
        x = transform(q, raw)
        y = transform(c, self.raw(c))
        inv = LedoitWolf().fit(b).precision_
        eig, U = np.linalg.eigh(inv)
        root = U @ np.diag(np.sqrt(np.maximum(eig, 1e-08))) @ U.T
        dc = cdist(x @ root, y @ root, 'sqeuclidean') / len(NUM)
        db = cdist(x @ root, b @ root, 'sqeuclidean') / len(NUM)
        dc += q.sex.to_numpy()[:, None] != c.sex.to_numpy()[None, :]
        db += q.sex.to_numpy()[:, None] != self.a.sex.to_numpy()[None, :]
        return (dc, db)

    def bin_frames(self, c, q):
        frames = [c.copy(), q.copy(), self.ref.copy()]
        changed = []
        for col in NUM:
            levels = np.sort(c[col].unique())
            limit = 20 if col == 'age' else 48
            if len(levels) < 3 or len(levels) > limit:
                continue
            if len(levels) >= self.a[col].nunique() * 0.8:
                continue
            edges = (levels[1:] + levels[:-1]) / 2
            for f in frames:
                f[col] = levels[np.searchsorted(edges, f[col].to_numpy(), side='left')]
            changed.append(col)
        return (frames, changed)

def pairs(q,c,bg,dimension):
    raw=Metric.raw(bg);scale=np.maximum(np.quantile(raw,.75,axis=0)-np.quantile(raw,.25,axis=0),1e-5)
    def distance(ref):
        x,y=Metric.raw(q)/scale,Metric.raw(ref)/scale
        d=np.sort((x[:,None]-y[None])**2,axis=2)[:,:,:dimension].mean(2)
        d[q.sex.to_numpy()[:,None]!=ref.sex.to_numpy()[None]]=np.inf
        return d
    dc,db=distance(c),distance(bg);pair=np.empty_like(dc)
    for i,row in enumerate(dc):
        ref=np.sort(db[i]);ref=ref[np.isfinite(ref)]
        interpolation=np.interp(row,np.r_[0.,ref],np.r_[0.,np.arange(1,len(ref)+1)],right=len(ref))
        pair[i]=-np.log((interpolation+.25)/(len(bg)+1))
    pair[~np.isfinite(dc)]=-1e4
    full=cdist(Metric.raw(q)/scale,Metric.raw(c)/scale,'sqeuclidean')/5
    pair-=1e-10*np.minimum(full,1000.)
    return pair,dc

def assignment(pair,dc,q,c,fraction):
    limit=np.partition(dc,11,axis=1)[:,11]
    limited=np.where(np.isfinite(dc)&(dc<=limit[:,None]),pair,-1e4)
    chosen=np.zeros(len(q),bool);donor=np.argmin(dc,axis=1)
    for sex in np.unique(q.sex):
        ai=np.flatnonzero(q.sex.to_numpy()==sex);bi=np.flatnonzero(c.sex.to_numpy()==sex)
        score=limited[ai][:,bi];nq,nc=score.shape;count=int(np.ceil(nq*fraction));dummy=nq-count;size=nc+dummy
        cost=np.zeros((size,size));cost[:nq,:nc]=-score;cost[nq:,nc:]=1e6
        rr,cc=linear_sum_assignment(cost);assert np.array_equal(rr,np.arange(size))
        assigned=cc[:nq];real=assigned<nc;assert real.sum()==count
        chosen[ai[real]]=True;donor[ai[real]]=bi[assigned[real]]
        assert np.unique(donor[ai[real]]).size==count
    return chosen,donor

def partial_distance(q,c,bg,dim=3):
    scale=np.maximum(np.quantile(Metric.raw(bg),.75,axis=0)-np.quantile(Metric.raw(bg),.25,axis=0),1e-5)
    x,y=Metric.raw(q)/scale,Metric.raw(c)/scale
    d=np.sort((x[:,None]-y[None])**2,axis=2)[:,:,:dim].mean(2)
    d[q.sex.to_numpy()[:,None]!=c.sex.to_numpy()[None,:]]=np.inf
    return d

def evidence(q,c,bg,dim=3):
    ref=subset(bg,min(1049,len(bg)),SEED+1)
    values=[]
    for start in range(0,len(q),192):
        part=q.iloc[start:start+192]
        dc=partial_distance(part,c,bg,dim);db=partial_distance(part,ref,bg,dim)
        values.extend(np.log((db.min(1)+1e-8)/(dc.min(1)+1e-8)))
    return np.array(values)

def calibrated_pairs(q,c,bg,neg,dimension=3):
    dc=partial_distance(q,c,bg,dimension);ref=subset(bg,min(1049,len(bg)),SEED+1)
    db=partial_distance(q,ref,bg,dimension)
    with np.errstate(divide='ignore'):
        pair_e=np.log((db.min(1)[:,None]+1e-8)/(dc+1e-8))
    ne=evidence(neg,c,bg,dimension)
    pair=np.empty_like(dc);pv=np.empty(len(q));groups={}
    for i,row in enumerate(q.itertuples()):
        key=(int(row.sex),int(row.age)//10)
        if key not in groups:
            mask=(neg.sex.to_numpy()==key[0])&(neg.age.to_numpy()//10==key[1])
            if mask.sum()<48:mask=neg.sex.to_numpy()==key[0]
            vals=np.sort(ne[mask]);groups[key]=vals
        vals=groups[key];n=len(vals)
        pp=(n-np.searchsorted(vals,pair_e[i],side='left')+.5)/(n+1)
        pair[i]=-np.log(np.clip(pp,1e-8,1));pv[i]=pp.min()
    pair[~np.isfinite(dc)]=-1e4
    return pair, dc

def competitive_assignment(bg, c, q, capacity):
    """Full-QI C/background competition; at most half of each sex can abstain.

    Each C donor is reused at most `capacity` times. This is a matching prior,
    not an estimate of an individual row's membership probability.
    """
    b = Metric.raw(bg)
    scale = np.maximum(np.quantile(b,.75,axis=0)-np.quantile(b,.25,axis=0),1e-5)
    rng = np.random.default_rng(608408)
    ref = bg.iloc[np.sort(rng.choice(len(bg),min(1049,len(bg)),False))].reset_index(drop=True)
    def distance(frame):
        d = cdist(Metric.raw(q)/scale, Metric.raw(frame)/scale, 'sqeuclidean')/5
        d[q.sex.to_numpy()[:,None] != frame.sex.to_numpy()[None,:]] = np.inf
        d += .5*(q.prefecture.astype(str).to_numpy()[:,None] != frame.prefecture.astype(str).to_numpy()[None,:])
        return d
    dc, db = distance(c), distance(ref)
    with np.errstate(divide='ignore'):
        evidence_matrix = np.log((db.min(1)[:,None]+1e-7)/(dc+1e-7))
    value = np.zeros(len(q)); donor = np.full(len(q),-1,int)
    for sex in np.unique(q.sex):
        ai = np.flatnonzero(q.sex.to_numpy()==sex)
        bi = np.flatnonzero(c.sex.to_numpy()==sex)
        cost = -np.repeat(evidence_matrix[ai][:,bi],capacity,axis=1)
        abstentions = len(ai)-int(np.ceil(.5*len(ai)))
        rr, cc = linear_sum_assignment(np.column_stack([cost,np.zeros((len(ai),abstentions))]))
        keep = cc < len(bi)*capacity
        selected, donors = ai[rr[keep]], bi[cc[keep]//capacity]
        value[selected] = c.time.to_numpy()[donors]
        donor[selected] = donors
    used = donor[donor>=0]
    if len(used):
        assert np.unique(used,return_counts=True)[1].max() <= capacity
    return value, donor


def predict(method, bg, c, q, comparison_qi):
    """Generate the frozen team's answers from public data, in query-ID order."""
    if method.startswith(('P01_', 'P03_', 'P04_', 'P05_')):
        family, fraction, _ = method.split('_')
        dim = 2 if family in ('P01','P05') else 3
        if family in ('P04','P05'):
            pair, dc = calibrated_pairs(q,c,bg,comparison_qi,dim)
        else:
            pair, dc = pairs(q,c,bg,dim)
        keep, _ = assignment(pair,dc,q,c,int(fraction)/100.)
        return np.where(keep,9.5,0.)
    if method.startswith('CA_'):
        capacity = 1 if method.startswith('CA_cap1_') else 2
        value, donor = competitive_assignment(bg,c,q,capacity)
        if method.endswith('_const95'):
            return np.where(donor>=0,9.5,0.)
        return value
    ra = ResearchAttack(bg,SimpleNamespace(Metric=Metric,subset=subset,SEED=SEED))
    if method == 'copula_window25':
        dc, db = ra.copula(q,c)
        c_mass = time_mass(dc,c.time.to_numpy(),20)
        bg_mass = time_mass(db,bg.time.to_numpy(),40)
        return stable_window(c_mass-.25*bg_mass)
    if method == 'bin_95_Ctime':
        (bc,bq,br), _ = ra.bin_frames(c,q)
        dc = ra.cost(bq,bc); db = ra.cost(bq,br)
        evidence_score = np.log((db.min(1)+1e-8)/(dc.min(1)+1e-8))
        return np.where(gate(evidence_score,.95),bc.time.to_numpy()[dc.argmin(1)],0.)
    dc = ra.cost(q,c)
    nearest = c.time.to_numpy()[dc.argmin(1)]
    if method.startswith('N9_'):
        draws = np.stack([ra.cost(q,subset(bg,min(len(bg),1049),seed)).min(1)
                          for seed in range(261006,261022)])
        evidence_score = np.median(np.log((draws+1e-8)/(dc.min(1)[None]+1e-8)),axis=0)
        fraction = int(method.split('_')[1])/100.
        if method.endswith('_late95'):
            nearest = np.where(nearest>=9,9.5,nearest)
    elif method == 'S4_85_Ctime':
        evidence_score = np.log((ra.cost(q,ra.ref).min(1)+1e-8)/(dc.min(1)+1e-8))
        fraction = .85
    else:
        raise ValueError(f'Unknown frozen method: {method}')
    return np.where(gate(evidence_score,fraction),nearest,0.)


def main():
    parser = argparse.ArgumentParser(description='Generate round-4 Team-28 F_aia.csv from public inputs.')
    parser.add_argument('--data-dir',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--a-bg',type=Path,default=DEFAULT_BG)
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'recomputed')
    parser.add_argument('--overwrite',action='store_true',help='Replace generated files in output-dir only.')
    args = parser.parse_args()
    start = time.time()
    paths = [args.a_bg,args.data_dir/'mia_columns.json']
    paths += [args.data_dir/f'C_{k}.csv' for k in TEAMS]
    paths += [args.data_dir/f'aia_challenge_{k}.csv' for k in TEAMS+[28]]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    before = {str(path.resolve()):file_hash(path) for path in paths}
    metadata = json.loads((args.data_dir/'mia_columns.json').read_text(encoding='utf-8-sig'))
    targets = {natural_id(x,'target_columns') for x in metadata['target_columns']}
    if targets != set(TEAMS+[28]):
        raise ValueError('Target metadata differs from the frozen 22-opponent, Team-28 protocol.')
    bg = read_table(args.a_bg,'background')
    if len(bg) != 5143:
        raise ValueError('Expected the 5143-row public main background.')
    qs = {k:read_table(args.data_dir/f'aia_challenge_{k}.csv','challenge') for k in TEAMS+[28]}
    for q in qs.values():
        if q.challenge_row_id.tolist() != list(range(326)):
            raise ValueError('Expected challenge IDs 0..325 for every team.')
    destination = args.output_dir.resolve()
    if destination == args.data_dir.resolve():
        raise ValueError('Output directory must differ from the public input directory.')
    outputs = [destination/'F_aia.csv',destination/'execution_record.json']
    if any(x.resolve() in {p.resolve() for p in paths} for x in outputs):
        raise ValueError('Refusing to replace an input.')
    if not args.overwrite and any(x.exists() for x in outputs):
        raise FileExistsError('Output exists; use a new output directory or --overwrite.')
    rows = []; team_record = []
    for k in TEAMS:
        c = read_table(args.data_dir/f'C_{k}.csv','processed')
        if len(c) != 1049:
            raise ValueError(f'C_{k}.csv: expected 1049 rows.')
        comparison = pd.concat([qs[j][QI] for j in TEAMS+[28] if j!=k],ignore_index=True)
        value = predict(ROUTE[k],bg,c,qs[k],comparison)
        if value.shape != (326,) or not np.isfinite(value).all() or not ((value>=0)&(value<=10)).all():
            raise ValueError(f'Invalid answers for team {k}.')
        rows.extend((k,int(r),float(v)) for r,v in zip(qs[k].challenge_row_id,value))
        entry = {'target_k':k,'method':ROUTE[k],'positive_answers':int((value>0).sum()),'zero_answers':int((value==0).sum())}
        team_record.append(entry)
        print(json.dumps(entry),flush=True)
    if len(rows)!=7172 or len({(k,r) for k,r,_ in rows})!=7172 or any(k==28 for k,_,_ in rows):
        raise ValueError('Invalid submission key set.')
    after = {str(path.resolve()):file_hash(path) for path in paths}
    if before != after:
        raise RuntimeError('Public inputs changed during generation; no output written.')
    payload = csv_bytes(rows)
    destination.mkdir(parents=True,exist_ok=True)
    outputs[0].write_bytes(payload)
    record = {'rows':len(rows),'teams':team_record,'input_sha256':before,'output_sha256':hashlib.sha256(payload).hexdigest(),
              'script_sha256':file_hash(Path(__file__)),'seconds':round(time.time()-start,3),
              'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                          'scipy':scipy.__version__,'scikit_learn':sklearn.__version__},
              'no_network_or_scoring_calls':True,'public_inputs_only':True}
    outputs[1].write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(outputs[0]),'rows':len(rows),'sha256':record['output_sha256']},ensure_ascii=True),flush=True)


if __name__ == '__main__':
    main()
