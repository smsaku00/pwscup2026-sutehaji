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
PROPOSAL = "04-selective-nearest"
KEEP_FRACTION = 0.75


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



class Attack:
    def __init__(self,a):
        self.a=a
        self.metric=Metric(a)

    def predict(self,c,q):
        cost=self.metric.distance(q,c)
        donor=np.argmin(cost,axis=1)
        dc=cost[np.arange(len(q)),donor]
        reference=subset(self.a,min(len(c),SOURCE_LIMIT),SEED+1)
        da=self.metric.distance(q,reference).min(1)
        evidence=np.log((da+1e-8)/(dc+1e-8))
        # Select using observed QI only. No membership labels are read.
        order=np.argsort(-evidence,kind="stable")
        retained=order[:int(np.ceil(KEEP_FRACTION*len(q)))]
        prediction=np.zeros(len(q))
        prediction[retained]=c.time.to_numpy(float)[donor[retained]]
        return prediction,{"retained_rows":int(len(retained)),
                           "keep_fraction":KEEP_FRACTION,
                           "low_evidence_time":0.0}

if __name__ == "__main__":
    raise SystemExit(run_cli())
