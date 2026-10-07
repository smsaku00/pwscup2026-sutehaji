"""AIAの比較分析。提出CSV・ZIP・token・外部書込みは作成／実行しない。

予備戦の公開正解を評価に使用する。本戦は公開入力の診断と自チーム標的のみ。
ルート .venv/bin/python -B 分析.py --root <PWS作業ルート> で実行する。
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import hashlib
import io
import json
import platform
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
NUM = ["age", "SBP", "TG", "HDL", "ALT"]
CAT = ["sex", "prefecture"]
QI = NUM + CAT
METRICS = ["S4_log_iqr_soft", "raw_sd_both", "log_iqr_both", "log_sd_both",
           "log_iqr_sex", "log_sd_sex", "log_sex_sd_both", "log_sex_mahal_both",
           "partial4_log_iqr_both"]
SEED = 20261007
HASHES = {}


def read(path: Path) -> pd.DataFrame:
    data = path.read_bytes()
    HASHES[str(path)] = hashlib.sha256(data).hexdigest()
    return pd.read_csv(io.BytesIO(data), float_precision="round_trip")


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def scores(pred, truth):
    pred = np.asarray(pred, float)
    assert len(pred) == len(truth) and np.isfinite(pred).all()
    assert ((pred >= 0) & (pred <= 10)).all()
    member = truth.label.eq("member").to_numpy()
    control = truth.label.eq("control").to_numpy()
    assert (member | control).all() and member.sum() == control.sum() == 163
    hit = np.abs(pred - truth.time.to_numpy(float)) <= .5
    rare = truth.is_rare.to_numpy(bool)
    pm, pc = float(hit[member].mean()), float(hit[control].mean())
    r = float(np.clip((pm-pc) / max(1-pc, 1e-9), 0, 1))
    mm, cc = member & rare, control & rare
    assert mm.sum() == cc.sum() == 13
    pmr, pcr = float(hit[mm].mean()), float(hit[cc].mean())
    se = np.sqrt(pmr*(1-pmr)/mm.sum() + pcr*(1-pcr)/cc.sum())
    rr = float(np.clip((pmr-pcr-1.64*se) / max(1-pcr, 1e-9), 0, 1))
    return dict(r_all=r, r_rare=rr, r_max=max(r,rr), p_m=pm, p_c=pc,
                hit_m=int(hit[member].sum()), hit_c=int(hit[control].sum()),
                hit_m_rare=int(hit[mm].sum()), hit_c_rare=int(hit[cc].sum()))


def raw(df, log=True):
    x = df[NUM].to_numpy(float).copy()
    if log:
        x[:, [2,4]] = np.log1p(x[:, [2,4]])
    return x


def cost(q, c, bg, metric):
    log = not metric.startswith("raw_")
    b, x, y = raw(bg, log), raw(q, log), raw(c, log)
    if "sex_sd" in metric or "sex_mahal" in metric:
        d = np.empty((len(q),len(c)))
        for sex in [0,1]:
            mask = q.sex.eq(sex).to_numpy()
            if not mask.any():
                continue
            ref = b[bg.sex.eq(sex).to_numpy()]
            scale = np.maximum(ref.std(0,ddof=1), 1e-5)
            if "mahal" in metric:
                precision = LedoitWolf().fit(ref / scale).precision_
                d[mask] = cdist(x[mask]/scale, y/scale, "mahalanobis", VI=precision)**2 / len(NUM)
            else:
                d[mask] = cdist(x[mask]/scale, y/scale, "sqeuclidean") / len(NUM)
    else:
        scale = (np.quantile(b,.75,axis=0)-np.quantile(b,.25,axis=0)) if "iqr" in metric else b.std(0,ddof=1)
        scale = np.maximum(scale,1e-5)
        x, y = x/scale, y/scale
        if metric.startswith("partial4"):
            parts = []
            for omit in range(len(NUM)):
                use = [j for j in range(len(NUM)) if j != omit]
                parts.append(cdist(x[:,use], y[:,use], "sqeuclidean") / len(use))
            d = np.min(parts,axis=0)
        else:
            d = cdist(x, y, "sqeuclidean") / len(NUM)
    mismatch_sex = q.sex.to_numpy()[:,None] != c.sex.to_numpy()[None,:]
    mismatch_pref = q.prefecture.to_numpy()[:,None] != c.prefecture.to_numpy()[None,:]
    if metric == "S4_log_iqr_soft":
        d += mismatch_sex.astype(float) + mismatch_pref.astype(float)
    else:
        d[mismatch_sex] = np.inf
        if metric.endswith("both"):
            d[mismatch_pref] = np.inf
        empty = ~np.isfinite(d).any(1)
        if empty.any():
            # 県条件で候補ゼロの場合だけ性別一致へ緩和。その件数を呼出側で記録する。
            fallback = cdist(x[empty]/scale if "sex_sd" in metric or "sex_mahal" in metric else x[empty],
                             y/scale if "sex_sd" in metric or "sex_mahal" in metric else y,
                             "sqeuclidean") / len(NUM)
            fallback[mismatch_sex[empty]] = np.inf
            assert np.isfinite(fallback).any(1).all()
            d[empty] = fallback
    return d


def keep(scores_, fraction):
    out = np.zeros(len(scores_),bool)
    out[np.argsort(-scores_,kind="stable")[:int(np.ceil(len(out)*fraction))]] = True
    return out


def window_masses(d, time_, k, centers):
    order = np.argsort(d,axis=1,kind="stable")[:,:k]
    dist = np.take_along_axis(d,order,axis=1)
    h = np.maximum(np.where(np.isfinite(dist[:,-1]),dist[:,-1],dist[:,0]),1e-8)
    w = np.exp(-np.minimum(dist / (.35*h[:,None]),700))
    w[~np.isfinite(dist)] = 0
    w /= w.sum(1,keepdims=True)
    times = np.asarray(time_,float)[order]
    return np.einsum("nk,nkl->nl",w,(np.abs(times[:,:,None]-centers[None,None,:])<=.5).astype(float))


def later_best(centers, mass):
    return centers[len(centers)-1-np.argmax(mass[:,::-1],axis=1)]


def predict_methods(q, c, bg):
    """推定側にはラベル・正解time・Bを渡さない。メモリ上の分析予測のみ。"""
    predictions = {f"constant_{t:g}":np.full(len(q),t) for t in [0,9.5,9.75,10]}
    features = []
    bg_sample = bg.iloc[np.sort(np.random.default_rng(SEED).choice(len(bg),min(len(c),len(bg)),False))].reset_index(drop=True)
    centers = np.arange(0,10.001,.25)
    for metric in METRICS:
        dc = cost(q,c,bg,metric)
        db = cost(q,bg_sample,bg,metric)
        nn = dc.argmin(1)
        value = c.time.to_numpy(float)[nn]
        d1 = dc[np.arange(len(q)),nn]
        d2 = np.partition(dc,1,axis=1)[:,1]
        b1 = db.min(1)
        ratio = np.log((b1+1e-8)/(d1+1e-8))
        gap = np.zeros_like(d1)
        finite_second = np.isfinite(d2)
        gap[finite_second] = (d2[finite_second]-d1[finite_second])/np.maximum(d2[finite_second],1e-8)
        predictions[f"{metric}/nn_all"] = value
        predictions[f"{metric}/nn_long95"] = np.where(value>=9.5,9.5,value)
        for fraction in [.25,.5,.75,.9]:
            selected = keep(ratio,fraction)
            predictions[f"{metric}/nn_gate{fraction:g}_0"] = np.where(selected,value,0)
            predictions[f"{metric}/long_gate{fraction:g}_0"] = np.where(selected,9.5,0)
            if fraction in [.5,.75]:
                predictions[f"{metric}/nn_gate{fraction:g}_95"] = np.where(selected,value,9.5)
        pc = window_masses(dc,c.time.to_numpy(),5,centers)
        pb = window_masses(cost(q,bg,bg,metric),bg.time.to_numpy(),40,centers)
        for lam in [0,.25,.5,.9]:
            predictions[f"{metric}/window5_lam{lam:g}"] = later_best(centers,pc-lam*pb)
        same = (c.sex.to_numpy()[None,:] == q.sex.to_numpy()[:,None]) & (c.prefecture.to_numpy()[None,:] == q.prefecture.to_numpy()[:,None])
        features.extend([np.log1p(d1), np.log1p(b1), ratio, gap,
                         same.sum(1)/len(c), value/10, pc[:,-3], pb[:,-3]])
    base = raw(bg)
    scale = np.maximum(np.quantile(base,.75,axis=0)-np.quantile(base,.25,axis=0),1e-5)
    features.extend(((raw(q)-np.median(base,axis=0))/scale).T)
    features.append(q.sex.to_numpy(float))
    return predictions, np.column_stack(features)


def validate_scores(truths):
    """公式関数と独立計算を、全公開課題・乱数回答で照合。"""
    import sys
    code = HERE / "参照" / "attack_score_20260928.py"
    if not code.exists():
        return {"official_function_checked":False}
    import types
    import importlib.util
    pkg = types.ModuleType("aia_audit_reference"); pkg.__path__ = [str(code.parent)]
    sys.modules[pkg.__name__] = pkg
    spec = importlib.util.spec_from_file_location(pkg.__name__+".attack_score",code)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    rng = np.random.default_rng(SEED)
    n = 0
    for t in truths.values():
        m = t.label.eq("member").to_numpy(); r = t.is_rare.to_numpy(bool)
        for prediction in [np.full(len(t),9.5),np.zeros(len(t)),rng.uniform(0,10,len(t))]:
            a = scores(prediction,t)
            b = mod.score_aia(None,prediction[m],t.onset.to_numpy()[m],t.time.to_numpy()[m],
                              None,prediction[~m],t.onset.to_numpy()[~m],t.time.to_numpy()[~m],
                              {"scoring":{"low_fpr":.01,"aia_rare_se_z":1.64},"aia":{"tau_time":.5}},
                              member_rare_mask=r[m],control_rare_mask=r[~m])
            assert abs(a["r_all"]-b.breakdown["r_time_all"])<1e-12
            assert abs(a["r_rare"]-b.breakdown["r_time_rare"])<1e-12
            n += 1
    return {"official_function_checked":True,"cases":n}


def main(args):
    start = time.time()
    root = args.root.resolve()
    out = HERE / "集計"
    out.mkdir(parents=True,exist_ok=True)
    pdist = root / "予備戦/配布データ/PWSCup2026_prelim_attack"
    pbg = read(root / "予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv")
    td = next((root / "予備戦/配布データ/結果").glob("正解*"))
    truths = {int(p.stem.rsplit("_",1)[1]):read(p).sort_values("challenge_row_id").reset_index(drop=True) for p in td.glob("aia_truth_*.csv")}
    rows, caches = [], []
    for k in sorted(truths):
        q = read(pdist/f"aia_challenge_{k}.csv").sort_values("challenge_row_id").reset_index(drop=True)
        c = read(pdist/f"C_{k}.csv")
        assert q.challenge_row_id.tolist() == truths[k].challenge_row_id.tolist()
        pred, x = predict_methods(q,c,pbg)
        for name,v in pred.items():
            rows.append(dict(team=k,method=name,**scores(v,truths[k])))
        caches.append(dict(k=k,q=q,x=x,pred=pred,truth=truths[k]))
        print(json.dumps({"prelim_team":k,"methods":len(pred),"seconds":round(time.time()-start,1)}),flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out/"予備戦_全設定と対象別.csv",index=False)
    summary = table[table.team.ne(5)].groupby("method").agg(r_all=("r_all","mean"),r_rare=("r_rare","mean"),r_max=("r_max","mean"),p_m=("p_m","mean"),p_c=("p_c","mean"),targets=("team","count")).sort_values("r_all",ascending=False)
    summary.to_csv(out/"予備戦_全設定の平均.csv")
    diag = []
    mdist = root / "本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力"
    mbg = read(mdist/"A_bg.csv")
    b = read(next((root/"本戦/配布データ").glob("PWSCup2026_final_team28_*/B_self.csv")))
    main_cache = []
    for p in sorted(mdist.glob("aia_challenge_*.csv"),key=lambda p:int(p.stem.rsplit("_",1)[1])):
        k = int(p.stem.rsplit("_",1)[1]); q = read(p).sort_values("challenge_row_id").reset_index(drop=True); c = read(mdist/f"C_{k}.csv")
        pred,x = predict_methods(q,c,mbg)
        for name,v in pred.items():
            diag.append(dict(team=k,method=name,n=len(v),zero=int((v==0).sum()),long=int((v>=9).sum()),short=int(((v>0)&(v<9)).sum()),median=float(np.median(v))))
        main_cache.append(dict(k=k,q=q,x=x,pred=pred))
        if k==28:
            matched = q.merge(b[QI+["time","is_rare"]],on=QI,how="left",validate="one_to_one")
            member = matched.time.notna().to_numpy(); assert member.sum()==163
            t=matched.time.to_numpy(float); rare=matched.is_rare.fillna(False).to_numpy(bool)
            sr=[]
            for name,v in pred.items():
                h=np.abs(v-t)<=.5
                sr.append(dict(method=name,n_member=163,hit_member=int(h[member].sum()),p_m=float(h[member].mean()),n_rare=int(rare.sum()),hit_rare=int(h[rare].sum()),control_truth_known=False))
            pd.DataFrame(sr).sort_values("p_m",ascending=False).to_csv(out/"本戦自チーム28_標的のみ.csv",index=False)
            label_diag=[]
            for i,metric in enumerate(METRICS):
                score_=x[:,i*8+2]; label_diag.append(dict(metric=metric,selection_auc=float(roc_auc_score(member,score_))))
            pd.DataFrame(label_diag).to_csv(out/"本戦自チーム28_標的選別.csv",index=False)
        print(json.dumps({"main_team":k,"seconds":round(time.time()-start,1)}),flush=True)
    pd.DataFrame(diag).to_csv(out/"本戦_予測動作の診断.csv",index=False)
    # 本戦の正解はキャッシュに含めない。自チームも公開QI・分析予測だけ。
    cache=HERE/"作業用";cache.mkdir(exist_ok=True)
    names=list(caches[0]["pred"])
    np.savez_compressed(cache/"比較特徴量.npz",prelim_x=np.vstack([v["x"] for v in caches]),prelim_team=np.concatenate([np.full(len(v["q"]),v["k"]) for v in caches]),prelim_pred=np.vstack([np.column_stack([v["pred"][name] for name in names]) for v in caches]),prelim_member=np.concatenate([v["truth"].label.eq("member").to_numpy() for v in caches]),prelim_time=np.concatenate([v["truth"].time.to_numpy() for v in caches]),prelim_rare=np.concatenate([v["truth"].is_rare.to_numpy() for v in caches]),main_x=np.vstack([v["x"] for v in main_cache]),main_team=np.concatenate([np.full(len(v["q"]),v["k"]) for v in main_cache]),main_pred=np.vstack([np.column_stack([v["pred"][name] for name in names]) for v in main_cache]),names=np.array(names))
    dump(HERE/"検証結果.json",validate_scores(truths))
    dump(HERE/"実行記録.json",dict(seed=SEED,python=platform.python_version(),executable=str(__import__("sys").executable),seconds=time.time()-start,metrics=METRICS,methods=len(names),prelim_teams=len(caches),main_teams=len(main_cache),input_sha256=HASHES,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),submission_created=False,codabench_submitted=False))
    print(summary.head(15).to_string(),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",type=Path,required=True)
    main(ap.parse_args())
