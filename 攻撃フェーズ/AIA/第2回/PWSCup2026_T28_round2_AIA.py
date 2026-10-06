"""PWSCup2026 main AIA, team 28, round 2, 2026-10-06.

Recomputes predictions from current public C/challenge/A_bg data.
Single file; Python 3.10+, numpy, pandas, scipy, scikit-learn.
No B/B_self, opponent truth, cached answers, tokens, network or upload.
Fixed route: S3 / S2 / N2 alpha .3 zero / N3 k5 lambda .25 / N9 16 BG samples.
Official utility scores have not been used in this frozen route.
See 제2회_AIA_설명서_한국어.md for assumptions and submission packaging.
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
import scipy, sklearn
from scipy.spatial.distance import cdist
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesRegressor

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


ROUTE={0: 'N2_partial_a30_0', 1: 'S3', 2: 'S3', 3: 'N3_window_sex_k5_w25', 4: 'S3', 5: 'N3_window_full_k5_w25', 6: 'N3_window_sex_k5_w25', 7: 'S3', 8: 'S2', 9: 'N9_ensemble75_0', 10: 'N3_window_full_k5_w25', 11: 'S3', 12: 'N3_window_full_k5_w25', 13: 'N2_partial_a30_0', 14: 'N3_window_sex_k5_w25', 15: 'N2_partial_a30_0', 16: 'S3', 17: 'N3_window_full_k5_w25', 22: 'N3_window_full_k5_w25', 25: 'N3_window_full_k5_w25', 26: 'N3_window_full_k5_w25', 27: 'S3'}

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

def candidate_times(q,c,metric):
    distance = metric.distance(q,c)
    k = min(3,len(c))
    near = np.argsort(distance,axis=1,kind="stable")[:,:k]
    return np.column_stack([np.tile(CENTERS,(len(q),1)), c.time.to_numpy(float)[near]]), distance

def kernel_weights(distance, k=12):
    k = min(k,distance.shape[1])
    bandwidth = np.partition(distance,k-1,axis=1)[:,k-1]
    bandwidth = np.maximum(bandwidth,1e-5)
    w = np.exp(-np.minimum(distance / bandwidth[:,None],700.))
    return w / np.maximum(w.sum(1,keepdims=True),EPS)

def masses(times, weights, centers):
    result = np.zeros(centers.shape,float)
    for j in range(centers.shape[1]):
        hit = np.abs(times[None,:]-centers[:,j,None]) <= TAU
        result[:,j] = (weights*hit).sum(1)
    return result

def select_best(centers,score):
    # Fixed column order resolves exact ties.
    return centers[np.arange(len(centers)),np.argmax(score,axis=1)]

def exact_override(q,c,prediction):
    groups = {}
    for i,row in enumerate(c[QI].itertuples(index=False,name=None)):
        groups.setdefault(row,[]).append(float(c.time.iloc[i]))
    result = prediction.copy()
    matches = 0
    for i,row in enumerate(q[QI].itertuples(index=False,name=None)):
        times = groups.get(row)
        if times is None:
            continue
        t = np.asarray(times)
        candidates = np.unique(np.r_[t, np.clip(t-TAU,0,10), np.clip(t+TAU,0,10)])
        count = (np.abs(t[:,None]-candidates[None,:]) <= TAU).sum(0)
        result[i] = candidates[np.argmax(count)]
        matches += 1
    return result,matches

def shadow_process(raw,reference,kind,rng):
    c=raw[COLS].copy().reset_index(drop=True)
    if kind in ("noise","round","mixture"):
        cols=NUM+["BMI","FPG"]
        if kind=="mixture":
            perm=rng.permutation(len(c))
            c[cols]=.85*c[cols].to_numpy()+.15*c[cols].to_numpy()[perm]
        elif kind=="noise":
            for col in cols:
                spread=max(float(np.subtract(*np.quantile(c[col],[.75,.25]))),1e-3)
                c[col]=c[col]+rng.normal(0,.10*spread,len(c))
        else:
            for col,step in {"age":5,"BMI":2,"SBP":10,"TG":20,"HDL":5,"ALT":5,"FPG":10}.items():
                c[col]=np.rint(c[col]/step)*step
        for col,(lo,hi) in RANGES.items():
            c[col]=np.clip(c[col],lo,hi)
        c["age"]=np.rint(c.age)
    if kind in ("noise","round","mixture"):
        event=(c.onset.to_numpy()+c.death.to_numpy())>0
        c.loc[event,"time"]=np.clip(c.loc[event,"time"]+rng.normal(0,.12,event.sum()),.001,10.)
    if kind=="shuffle":
        # Preserve coarse QI-risk groups while breaking exact time links.
        order=np.argsort(c.SBP.to_numpy()+.3*c.age.to_numpy(),kind="stable")
        for group in np.array_split(order,8):
            c.loc[group,["time","onset","death"]]=c.loc[rng.permutation(group),["time","onset","death"]].to_numpy()
    if kind=="pool":
        metric=Metric(reference)
        pairs=metric.distance(raw,reference).argmin(1)
        c=reference.iloc[pairs][COLS].reset_index(drop=True).copy()
    return c

def proposal_features(q,c,a,metric):
    centers,dc=candidate_times(q,c,metric)
    reference=subset(a,min(len(c),SOURCE_LIMIT),SEED+4)
    da=metric.distance(q,reference)
    wc,wa=kernel_weights(dc),kernel_weights(da)
    pc=masses(c.time.to_numpy(float),wc,centers)
    pa=masses(reference.time.to_numpy(float),wa,centers)
    near=np.sort(dc,axis=1)[:,:min(3,len(c))]
    d1=near[:,0]; d2=near[:,min(1,near.shape[1]-1)]
    a1=da.min(1)
    common=np.column_stack([np.log1p(d1),np.log1p(a1),
                np.log((a1+1e-8)/(d1+1e-8)),(d2-d1)/(d2+1e-8),
                np.log1p(np.median(dc,axis=1)),
                np.log1p(np.median(da,axis=1))])
    local=[]
    for j in range(centers.shape[1]):
        hit=np.abs(c.time.to_numpy(float)[None,:]-centers[:,j,None])<=TAU
        dh=np.where(hit,dc,np.inf).min(1)
        dh=np.where(np.isfinite(dh),dh,1e3)
        local.append(np.column_stack([common,centers[:,j]/HORIZON,pc[:,j],pa[:,j],
                       pc[:,j]-pa[:,j],np.log((pc[:,j]+.005)/(pa[:,j]+.005)),
                       np.log1p(dh),np.log((a1+1e-8)/(dh+1e-8))]))
    # Rows grouped by candidate column then query.
    return np.vstack(local),centers,pa

class ShadowAttack:
    def __init__(self,a):
        if len(a)<128:
            raise ValueError("Shadow training needs A_bg with at least 128 rows")
        self.a=a
        self.metric=Metric(a)
        rng=np.random.default_rng(SEED+31)
        idx=rng.permutation(len(a))
        split=max(32,len(a)//3)
        ref=a.iloc[idx[:split]].reset_index(drop=True)
        pool=a.iloc[idx[split:]].reset_index(drop=True)
        size=min(SHADOW_ROWS,len(pool)//2)
        qn=min(SHADOW_QUERY_PER_GROUP,size)
        features=[]; labels=[]
        for repeat in range(SHADOW_REPEATS):
            pick=rng.permutation(len(pool))
            raw=pool.iloc[pick[:size]].reset_index(drop=True)
            control=pool.iloc[pick[size:2*size]].reset_index(drop=True)
            # Matched-control approximation: match sex, prefecture, age decade, onset.
            # These labels belong only to public A_bg shadow examples, never real challenge.
            chosen=[]; available=list(range(len(control)))
            member=raw.iloc[:qn].reset_index(drop=True)
            for row in member.itertuples():
                loss=np.array([4*(control.sex.iloc[j]!=row.sex)+
                               4*(control.prefecture.iloc[j]!=row.prefecture)+
                               4*(control.onset.iloc[j]!=row.onset)+
                               abs(control.age.iloc[j]//10-row.age//10) for j in available])
                at=int(np.argmin(loss)); chosen.append(available.pop(at))
            query_full=pd.concat([member,control.iloc[chosen]],ignore_index=True)
            query=query_full[QI].copy()
            truth=query_full.time.to_numpy(float)
            membership=np.r_[np.ones(qn),-np.ones(qn)]
            for kind in SHADOW_KINDS:
                release=shadow_process(raw,ref,kind,rng)
                x,centers,_=proposal_features(query,release,ref,self.metric)
                hit=(np.abs(truth[:,None]-centers)<=TAU).astype(float)
                label=(hit*membership[:,None]).T.reshape(-1)
                features.append(x); labels.append(label)
        self.model=ExtraTreesRegressor(n_estimators=N_TREES,min_samples_leaf=16,
                       max_depth=18,max_features=1.,random_state=SEED,n_jobs=1)
        self.model.fit(np.vstack(features),np.concatenate(labels))
        self.training_rows=sum(len(y) for y in labels)

    def predict(self,c,q):
        result=[]; confidence=[]
        for start in range(0,len(q),BATCH):
            part=q.iloc[start:start+BATCH]
            x,centers,_=proposal_features(part,c,self.a,self.metric)
            scores=self.model.predict(x).reshape(centers.shape[1],len(part)).T
            result.extend(select_best(centers,scores))
            confidence.extend(scores.max(1))
        return np.asarray(result),{"shadow_training_rows":int(self.training_rows),
                                  "mean_predicted_gain":float(np.mean(confidence))}

def fit_windows(x,t):
    models=[]
    for center in CENTERS:
        y=(np.abs(t-center)<=TAU).astype(int)
        if len(np.unique(y))==1:
            models.append(float(y[0])); continue
        model=HistGradientBoostingClassifier(max_iter=ITERATIONS,learning_rate=.08,
                 max_leaf_nodes=15,min_samples_leaf=12,l2_regularization=3.,
                 categorical_features=None,early_stopping=False,random_state=SEED)
        models.append(model.fit(x,y))
    return models

def window_probabilities(models,x):
    return np.column_stack([np.full(len(x),model) if isinstance(model,float)
                            else model.predict_proba(x)[:,1] for model in models])

class WindowBoostAttack:
    def __init__(self,a):
        self.metric=Metric(a)
        self.reference=subset(a,min(len(a),SOURCE_LIMIT),SEED+3)
        self.background=fit_windows(self.metric.features(self.reference),
                                    self.reference.time.to_numpy(float))

    def predict(self,c,q):
        target=fit_windows(self.metric.features(c),c.time.to_numpy(float))
        x=self.metric.features(q)
        pc=window_probabilities(target,x)
        pa=window_probabilities(self.background,x)
        score=pc-BACKGROUND_WEIGHT*pa
        centers=np.tile(CENTERS,(len(q),1))
        return select_best(centers,score),{"window_models":len(CENTERS),
                         "mean_best_contrast":float(np.max(score,axis=1).mean())}

def gate(scores,fraction=.75):
    keep=np.zeros(len(scores),bool);keep[np.argsort(-scores,kind='stable')[:int(np.ceil(fraction*len(scores)))]]=True;return keep

def stable_window(prob):
    # Tie preference: later centers, deterministic. Not a default 10-year answer.
    return N3_CENTERS[len(N3_CENTERS)-1-np.argmax(prob[:,::-1],axis=1)]

def time_mass(d,t,k):
    order=np.argsort(d,axis=1,kind='stable')[:,:k]
    ds=np.take_along_axis(d,order,axis=1)
    # Keep separated nearest donors informative; soften only uncertain distances.
    h=np.maximum(ds[:,-1],1e-6)
    w=np.exp(-ds/np.maximum(.35*h[:,None],1e-6));w/=w.sum(1,keepdims=True)
    return np.einsum('nk,nkl->nl',w,(np.abs(t[order,None]-N3_CENTERS[None,None,:])<=.5).astype(float))

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

def predict_research(attack, c, q, method):
    if method == 'N2_partial_a30_0':
        times, evidence, _, details = attack.partial(c,q)
        return np.where(evidence < .3, times, 0.), details
    if method.startswith('N3_window_'):
        pref = method == 'N3_window_full_k5_w25'
        dc = attack.cost(q,c,pref=pref)
        da = attack.cost(q,attack.a,pref=pref)
        pc = time_mass(dc,c.time.to_numpy(float),5)
        pa = time_mass(da,attack.bgtime,40)
        return stable_window(pc-.25*pa), {'C_neighbors':5,'BG_neighbors':40,'background_weight':.25,'prefecture_penalty':pref}
    if method == 'N9_ensemble75_0':
        dc = attack.cost(q,c)
        donor = dc.argmin(1)
        scores=[]
        for i in range(16):
            ref=subset(attack.a,min(1049,len(attack.a)),261006+i)
            distance=attack.cost(q,ref).min(1)
            scores.append(np.log((distance+1e-8)/(dc.min(1)+1e-8)))
        keep=gate(np.median(scores,axis=0),.75)
        return np.where(keep,c.time.to_numpy(float)[donor],0.), {'background_samples':16,'accepted':int(keep.sum()),'fallback':0.}
    raise ValueError('Unknown method: '+method)

def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def csv_bytes(rows):
    stream=io.StringIO(newline='')
    writer=csv.writer(stream,lineterminator='\n')
    writer.writerow(['target_k','challenge_row_id','time_hat'])
    for k,r,t in rows:
        writer.writerow([str(k),str(r),format(t,'.17g')])
    return stream.getvalue().encode('utf-8')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a-bg',type=Path,default=DEFAULT_BG)
    parser.add_argument('--data-dir',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'recomputed')
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    output=args.output_dir.resolve()
    metadata_path=args.data_dir/'mia_columns.json'
    metadata=json.loads(metadata_path.read_text(encoding='utf-8-sig'))
    allowed={natural_id(value,'target_columns') for value in metadata['target_columns']}-{28}
    if allowed!=set(ROUTE): raise ValueError('Released opponent metadata differs from the frozen round-2 route')
    files=[args.a_bg,metadata_path]+[args.data_dir/f'{stem}_{k}.csv' for k in sorted(ROUTE) for stem in ('C','aia_challenge')]
    csv_dest=output/'F_aia.csv'
    record_dest=output/'execution_record.json'
    if any(csv_dest.resolve()==p.resolve() or record_dest.resolve()==p.resolve() for p in files):
        raise ValueError('Output cannot replace an input')
    if not args.overwrite and (csv_dest.exists() or record_dest.exists()):
        raise FileExistsError('Output exists; choose a new --output-dir or use --overwrite')
    bg=read_table(args.a_bg,'background')
    if len(bg)!=5143: raise ValueError('Main background must have 5143 rows')
    tables=[]
    for k in sorted(ROUTE):
        c=read_table(args.data_dir/f'C_{k}.csv','C')
        q=read_table(args.data_dir/f'aia_challenge_{k}.csv','challenge')
        if len(c)!=1049 or len(q)!=326: raise ValueError(f'Team {k}: expected C1049 / challenge326')
        tables.append((k,c,q))
    hashes={str(p.resolve()):file_hash(p) for p in files}
    start=time.perf_counter()
    shadow=ShadowAttack(bg)
    boosting=WindowBoostAttack(bg)
    attack=ResearchAttack(bg,SimpleNamespace(Metric=Metric,subset=subset,SEED=SEED))
    rows=[]; details=[]
    for k,c,q in tables:
        method=ROUTE[k]
        if method in ('S3','S2'):
            model=shadow if method=='S3' else boosting
            values,diagnostic=model.predict(c,q)
            values,exact=exact_override(q,c,np.asarray(values,float))
            diagnostic['exact_QI_rows']=exact
        else:
            values,diagnostic=predict_research(attack,c,q,method)
        if values.shape!=(326,) or not np.isfinite(values).all() or not ((values>=0)&(values<=10)).all():
            raise ValueError(f'Team {k}: invalid predictions')
        ids=q.challenge_row_id.astype(int).tolist()
        rows.extend((k,r,float(t)) for r,t in zip(ids,values))
        diagnostic.update(target_k=k,method=method,rows=326,zero=int((values==0).sum()),late=int((values>=9).sum()))
        details.append(diagnostic)
        print(f'Team {k}: {method}, 326 rows, zero={diagnostic["zero"]}',flush=True)
    if len(rows)!=7172 or len({(k,r) for k,r,_ in rows})!=7172: raise ValueError('Output keys incomplete or duplicated')
    if any(file_hash(p)!=hashes[str(p.resolve())] for p in files): raise ValueError('Inputs changed during computation')
    data=csv_bytes(rows)
    output.mkdir(parents=True,exist_ok=True)
    csv_dest.write_bytes(data)
    record={'team_id':28,'round':2,'rows':7172,'opponents':22,'route':ROUTE,'input_hashes':hashes,
        'csv_sha256':hashlib.sha256(data).hexdigest(),'script_sha256':file_hash(__file__),
        'seconds':round(time.perf_counter()-start,3),'dependencies':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},
        'diagnostics':details,'utility_scores_used':False,'official_score_available':False,'uploaded':False}
    record_dest.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(csv_dest),'rows':7172,'sha256':record['csv_sha256'],'uploaded':False},ensure_ascii=False),flush=True)
    return 0

if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}',file=sys.stderr)
        raise SystemExit(1)

