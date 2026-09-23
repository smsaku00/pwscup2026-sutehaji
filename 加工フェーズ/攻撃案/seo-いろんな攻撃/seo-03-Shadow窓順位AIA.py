"""seo AIA additional proposal / v1.0 / 2026-09-22.
Standalone: python SCRIPT.py --b B.csv --a-bg A_bg.csv --targets targets.json
AIA ONLY. B argument is accepted but B contents are never read.
Rule reference: main-process-20260912, pp.8-10, 20-22.
No target truth, other team's B, historical predictions or team-specific exceptions.
"""
from __future__ import annotations
import os
for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_env] = "1"
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import re
import sys
import tempfile
import time
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesRegressor
import scipy
import sklearn

SEED = 20260922
TAU = 0.5
HORIZON = 10.
CENTERS = np.arange(0.5, 10.001, 0.5)
NUM = ["age", "SBP", "TG", "HDL", "ALT"]
CAT = ["sex", "prefecture"]
QI = NUM + CAT
COLS = ["age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL", "ALT",
        "smoking", "FPG", "time", "onset", "death"]
RANGES = {"age": (40,74), "BMI": (10,55), "SBP": (40,250), "TG": (1,3000),
          "HDL": (10,160), "ALT": (.1,700), "FPG": (40,400)}
EPS = 1e-12
SOURCE_LIMIT = 1200
BATCH = 96
PROPOSAL = "03-shadow-window-ranking"
SHADOW_KINDS = ("copy","noise","round","shuffle","pool","mixture")
SHADOW_REPEATS = 2
SHADOW_ROWS = 256
SHADOW_QUERY_PER_GROUP = 48
N_TREES = 160


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


def load_targets(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list) or not data:
        raise ValueError("targets.json must be a nonempty array")
    result = []
    for item in data:
        if not isinstance(item, dict) or set(item) != {"target_k","c_path","aia_challenge_path"}:
            raise ValueError("each target requires target_k, c_path, aia_challenge_path only")
        k = natural_id(item["target_k"], "target_k")
        paths = []
        for key in ("c_path", "aia_challenge_path"):
            value = item[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"invalid {key}")
            p = Path(value)
            p = (path.parent / p).resolve() if not p.is_absolute() else p.resolve()
            if not p.is_file():
                raise ValueError(f"missing {key}: {p}")
            paths.append(p)
        result.append((k, *paths))
    if len({k for k,_,_ in result}) != len(result):
        raise ValueError("duplicate target_k")
    return sorted(result)


def check_scope(targets, args):
    if args.scope == "local":
        if args.mia_columns or args.self_team_id is not None:
            raise ValueError("official metadata requires --scope official")
        return
    if args.mia_columns is None or args.self_team_id is None:
        raise ValueError("official mode needs --mia-columns and --self-team-id")
    meta = json.loads(args.mia_columns.read_text(encoding="utf-8-sig"))
    values = meta.get("target_columns") if isinstance(meta,dict) else None
    if not isinstance(values,list) or not values:
        raise ValueError("target_columns must be a nonempty list")
    ids = [natural_id(v, "target_columns") for v in values]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate official target")
    own = natural_id(args.self_team_id, "self-team-id")
    got = {k for k,_,_ in targets}
    if got - set(ids) or set(ids) - {own} - got:
        raise ValueError("all official opponents and no unknown target must be supplied")


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


def csv_bytes(frame):
    s = io.StringIO(newline="")
    w = csv.writer(s,lineterminator="\n")
    w.writerow(["target_k","challenge_row_id","time_hat"])
    for k,r,t in frame.itertuples(index=False,name=None):
        w.writerow([str(int(k)),str(int(r)),format(float(t),".17g")])
    return s.getvalue().encode("utf-8")


def save_csv(frame,input_paths,overwrite=False):
    dest = Path(__file__).resolve().with_name("F_aia_"+Path(__file__).stem+".csv")
    blob = csv_bytes(frame)
    if dest.resolve() in input_paths:
        raise ValueError("output collides with an input")
    if dest.exists() and dest.read_bytes() != blob and not overwrite:
        raise ValueError("different output already exists; use --overwrite")
    if not dest.exists() or dest.read_bytes()!=blob:
        with tempfile.NamedTemporaryFile(dir=dest.parent,delete=False,suffix=".tmp") as f:
            tmp=Path(f.name); f.write(blob)
        try:
            os.replace(tmp,dest)
        finally:
            tmp.unlink(missing_ok=True)
    return dest,hashlib.sha256(blob).hexdigest()


def peak_memory_mib():
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes
            class Memory(ctypes.Structure):
                _fields_=[("cb",wintypes.DWORD),("PageFaultCount",wintypes.DWORD),
                          ("PeakWorkingSetSize",ctypes.c_size_t),("WorkingSetSize",ctypes.c_size_t),
                          ("QuotaPeakPagedPoolUsage",ctypes.c_size_t),("QuotaPagedPoolUsage",ctypes.c_size_t),
                          ("QuotaPeakNonPagedPoolUsage",ctypes.c_size_t),("QuotaNonPagedPoolUsage",ctypes.c_size_t),
                          ("PagefileUsage",ctypes.c_size_t),("PeakPagefileUsage",ctypes.c_size_t)]
            info=Memory(); info.cb=ctypes.sizeof(info)
            kernel=ctypes.WinDLL("kernel32"); kernel.GetCurrentProcess.restype=wintypes.HANDLE
            psapi=ctypes.WinDLL("psapi")
            psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Memory),wintypes.DWORD]
            if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(info),info.cb):
                return info.PeakWorkingSetSize/1024**2
        except Exception:
            return None
    return None


def run_cli():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b",type=Path,required=True,help="interface only; not read")
    parser.add_argument("--a-bg",type=Path,required=True)
    parser.add_argument("--targets",type=Path,required=True)
    parser.add_argument("--scope",choices=["local","official"],default="local")
    parser.add_argument("--mia-columns",type=Path)
    parser.add_argument("--self-team-id",type=int)
    parser.add_argument("--overwrite",action="store_true")
    args=parser.parse_args()
    start=time.perf_counter()
    try:
        targets=load_targets(args.targets)
        check_scope(targets,args)
        a=read_table(args.a_bg,"background")
        if len(a)<128:
            raise ValueError("A_bg needs at least 128 rows")
        # Validate every target before training or producing output.
        tables=[(k,read_table(cp,"C"),read_table(qp,"challenge")) for k,cp,qp in targets]
        if any(len(c) < 8 for _,c,_ in tables):
            raise ValueError("C needs at least 8 rows")
        attack=Attack(a)
        frames=[]; diagnostics=[]
        for k,c,q in tables:
            prediction,details=attack.predict(c,q)
            prediction,exact=exact_override(q,c,np.asarray(prediction,float))
            if prediction.shape!=(len(q),) or not np.isfinite(prediction).all():
                raise ValueError("invalid prediction shape/value")
            if ((prediction<0)|(prediction>10)).any():
                raise ValueError("prediction outside [0,10]")
            frames.append(pd.DataFrame({"target_k":k,"challenge_row_id":q.challenge_row_id,
                                        "time_hat":prediction}))
            diagnostics.append({"target_k":k,"exact_QI_rows":exact,**details})
        out=pd.concat(frames,ignore_index=True).sort_values(["target_k","challenge_row_id"],kind="stable")
        if out.duplicated(["target_k","challenge_row_id"]).any():
            raise ValueError("duplicate output key")
        inputs={p.resolve() for p in [args.b,args.a_bg,args.targets]}
        inputs.update(p for _,cp,qp in targets for p in (cp,qp))
        if args.mia_columns:
            inputs.add(args.mia_columns.resolve())
        dest,digest=save_csv(out,inputs,args.overwrite)
        print(json.dumps({"proposal":PROPOSAL,"version":"1.0","scope":args.scope,
                          "rows":len(out),"targets":len(tables),"B_contents_used":False,
                          "output":dest.name,"sha256":digest,"diagnostics":diagnostics,
                          "seconds":round(time.perf_counter()-start,3),
                          "peak_memory_mib":peak_memory_mib(),
                          "dependencies":{"python":platform.python_version(),"numpy":np.__version__,
                                          "pandas":pd.__version__,"scipy":scipy.__version__,
                                          "sklearn":sklearn.__version__},
                          "official_score":None,"uploaded":False},ensure_ascii=True,allow_nan=False))
        return 0
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}",file=sys.stderr)
        return 1



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

class Attack:
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

if __name__ == "__main__":
    raise SystemExit(run_cli())
