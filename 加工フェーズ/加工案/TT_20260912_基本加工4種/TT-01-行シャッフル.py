#!/usr/bin/env python3
"""PWS Cup 2026 TT baseline, v1.0, 2026-09-12.
Rules/schema: main-process-20260912, §§3,5.1,6.1.4.
Only --b and --a-bg are individual-level inputs. No scoring input is read.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
import numpy as np

VERSION = "1.0"
METHOD = "row_shuffle"
SEED = 20260912
CONTINUOUS = ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG", "time"]
CATEGORICAL = ["sex", "prefecture", "smoking", "onset", "death"]
CATEGORY_MISMATCH_WEIGHT = 1.0
# Official constants: schema.json + scoring_config.yaml, main-process-20260912.
COLUMNS = ["age","sex","prefecture","BMI","SBP","TG","HDL","ALT","smoking","FPG","time","onset","death"]
RANGES = {"age":(40,74),"BMI":(10,55),"SBP":(40,250),"TG":(1,3000),"HDL":(10,160),"ALT":(0.1,700),"FPG":(40,400)}
HORIZON = 10.0
# Official utility._u_valid uses atol=0.01 (with numpy default rtol=1e-5).
CENSOR_ATOL = 0.01

def read_input(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        expected = ["record_id"] + COLUMNS
        if reader.fieldnames is None or len(reader.fieldnames) != len(expected) or set(reader.fieldnames) != set(expected):
            raise ValueError(f"{Path(path).name}: 14列（record_id＋指定13列）が必要です")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{Path(path).name}: データ行がありません")
    seen = set()
    for i, row in enumerate(rows, 2):
        prefix=f"{Path(path).name} CSV行{i}"
        if None in row or any(row.get(c) is None or not row[c].strip() for c in expected):
            raise ValueError(f"{prefix}: 欠測または列数不一致")
        vals={}
        for c in expected:
            if c == "prefecture":
                continue
            try:
                vals[c]=float(row[c])
            except ValueError:
                raise ValueError(f"{prefix}: {c}が単一の数値ではありません") from None
            if not math.isfinite(vals[c]):
                raise ValueError(f"{prefix}: {c}が非有限値")
        for c in ["record_id","age"]:
            if not vals[c].is_integer():
                raise ValueError(f"{prefix}: {c}は整数が必要です")
        if vals['record_id'] in seen:
            raise ValueError(f"{prefix}: record_id重複")
        seen.add(vals['record_id'])
        for c,(lo,hi) in RANGES.items():
            if not lo <= vals[c] <= hi:
                raise ValueError(f"{prefix}: {c}は[{lo},{hi}]の範囲外")
        for c in ["sex","smoking","onset","death"]:
            if vals[c] not in (0,1):
                raise ValueError(f"{prefix}: {c}は0/1が必要です")
        if not 0 < vals['time'] <= HORIZON or vals['onset']+vals['death'] > 1:
            raise ValueError(f"{prefix}: 転帰の値域・排他規則違反")
        if vals['onset']==0 and vals['death']==0 and abs(vals['time']-HORIZON)>CENSOR_ATOL+1e-5*HORIZON:
            raise ValueError(f"{prefix}: 非発症・非死亡の追跡年数が10年と整合しません")
    return rows

def generate(b, a):
    if METHOD == "row_shuffle":
        order=list(range(len(b)))
        random.Random(SEED).shuffle(order)
        return [b[i] for i in order]
    # Mixed squared Euclidean distance. Continuous scale uses supplied A_bg,
    # population standard deviation ddof=0; a constant column uses scale 1.
    matrix=np.array([[float(r[c]) for c in CONTINUOUS] for r in a],dtype=np.float64)
    scale=matrix.std(axis=0,ddof=0)
    scale=np.where(scale>0,scale,1.0)
    cats=np.array([[r[c] if c=='prefecture' else str(int(float(r[c]))) for c in CATEGORICAL] for r in a])
    result=[]
    for row in b:
        x=np.array([float(row[c]) for c in CONTINUOUS],dtype=np.float64)
        differences=(matrix-x)/scale
        distances=np.sum(differences*differences,axis=1)
        values=np.array([row[c] if c=='prefecture' else str(int(float(row[c]))) for c in CATEGORICAL])
        distances+=CATEGORY_MISMATCH_WEIGHT*np.sum(cats!=values,axis=1)
        # Exact ties resolve to the first row in the supplied A_bg. Reuse allowed.
        result.append(a[int(np.argmin(distances))])
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--b',required=True,type=Path)
    ap.add_argument('--a-bg',required=True,type=Path)
    args=ap.parse_args()
    start=time.perf_counter()
    destination=Path(__file__).resolve().with_name('C_'+Path(__file__).stem+'.csv')
    if destination in [args.b.resolve(),args.a_bg.resolve()]:
        ap.error('出力先が入力ファイルと同じです')
    try:
        b=read_input(args.b)
        # Row shuffle accepts the same argument but intentionally does not read A_bg.
        a=None if METHOD=='row_shuffle' else read_input(args.a_bg)
        c=generate(b,a)
        if len(c)!=len(b):
            raise ValueError('出力行数不一致')
        # Preserve every source cell's original decimal text: no numeric rounding.
        with destination.open('w',encoding='utf-8',newline='') as f:
            writer=csv.writer(f,lineterminator='\n')
            writer.writerow(COLUMNS)
            writer.writerows([[row[col] for col in COLUMNS] for row in c])
    except (ValueError,OSError,csv.Error) as e:
        ap.error(str(e))
    print(json.dumps({'version':VERSION,'method':METHOD,'rows':len(c),'output':destination.name,
        'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'elapsed_seconds':time.perf_counter()-start},ensure_ascii=False))
    return 0

if __name__=='__main__':
    sys.exit(main())
