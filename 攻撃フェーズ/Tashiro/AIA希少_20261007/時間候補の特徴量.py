"""既存の公開QI/C/BG比較から取り出した、時間候補と特徴量の計算。

正解やBを推定関数へ渡さない。元のスコア関数・提出機能は含まない。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.covariance import LedoitWolf
NUM = ["age", "SBP", "TG", "HDL", "ALT"]
CAT = ["sex", "prefecture"]
QI = NUM + CAT
SEED = 20261007
METRICS = ['S4_log_iqr_soft', 'raw_sd_both', 'log_iqr_both', 'log_sd_both', 'log_iqr_sex', 'log_sd_sex', 'log_sex_sd_both', 'log_sex_mahal_both', 'partial4_log_iqr_both']


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
