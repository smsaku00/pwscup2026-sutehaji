"""本戦MIA用Gen-LRA。入力真値を受け取らない計算部。"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

NUM7 = ('age', 'SBP', 'TG', 'HDL', 'ALT')
NUM9 = ('age', 'BMI', 'SBP', 'TG', 'HDL', 'ALT', 'FPG')
CATS = ('sex', 'prefecture')
QI7 = NUM7 + CATS
ATTR9 = NUM9 + CATS
COV10 = ('age', 'sex', 'prefecture', 'BMI', 'SBP', 'TG', 'HDL', 'ALT', 'smoking', 'FPG')
ALL13 = COV10 + ('time', 'onset', 'death')
K = 10
RHO = .98


def silverman(n, d):
    return (4.0 / (d + 2) / n) ** (1.0 / (d + 4))


def numeric(frame, cols):
    x = frame[list(cols)].to_numpy(float).copy()
    for col in ('TG', 'ALT'):
        if col in cols:
            x[:, list(cols).index(col)] = np.log1p(x[:, list(cols).index(col)])
    return x


def mixed_scores(reference, candidates, synthetic, nums):
    """予備戦分析21の混合KDEと近傍定義を移植。カテゴリ密度はCをqueryとする。"""
    r = numeric(reference, nums)
    scale = (np.quantile(r, .75, axis=0) - np.quantile(r, .25, axis=0)) / 1.349
    if not np.all(scale > 0):
        raise ValueError('背景IQRに0以下の列があります')
    r, b, c = r / scale, numeric(candidates, nums) / scale, numeric(synthetic, nums) / scale
    h = silverman(len(r), len(nums))
    # 予備戦コードと同じ加算順序・正規化定数を維持する。
    def distance2(left, right):
        out = np.zeros((len(left), len(right)))
        for j in range(left.shape[1]):
            out += (left[:, j, None]-right[None, :, j])**2
        return out
    cr = distance2(c, r)
    cb = distance2(c, b)
    d2 = cb.T.copy()
    cat_cr = np.zeros_like(cr)
    cat_cb = np.zeros_like(cb)
    for col in CATS:
        freq = (reference[col].value_counts() + 1) / (len(reference) + reference[col].nunique())
        if not (set(synthetic[col]) | set(candidates[col])) <= set(freq.index):
            raise ValueError(f'背景にないカテゴリ: {col}')
        cf = synthetic[col].map(freq).to_numpy()[:, None]
        eq_cr = synthetic[col].to_numpy()[:, None] == reference[col].to_numpy()[None, :]
        eq_cb = synthetic[col].to_numpy()[:, None] == candidates[col].to_numpy()[None, :]
        cat_cr += np.log(RHO * eq_cr + (1-RHO) * cf)
        cat_cb += np.log(RHO * eq_cb + (1-RHO) * cf)
        d2 += ~eq_cb.T
    d = len(nums)
    log_cr = -.5*cr/(h*h)-d*math.log(h)-d/2*math.log(2*math.pi)+cat_cr
    log_cb = -.5*cb/(h*h)-d*math.log(h)-d/2*math.log(2*math.pi)+cat_cb
    maximum = np.max(log_cr, axis=1, keepdims=True)
    denom = np.squeeze(maximum+np.log(np.exp(log_cr-maximum).sum(axis=1,keepdims=True)),axis=1)-math.log(len(r))
    delta = (np.logaddexp(0., log_cb-denom[:, None]-math.log(len(r)))-math.log1p(1/len(r))).T
    near = np.argsort(d2, axis=1, kind='stable')[:, :K]
    result = np.take_along_axis(delta, near, axis=1).sum(axis=1)
    if not np.isfinite(result).all():
        raise ArithmeticError('Gen-LRAに非有限値')
    return result


def ordinal_matrices(reference, candidates, synthetic, columns):
    """Yamazaki run_gen_lra.py a465f10の_feature_matricesと同じ変換。"""
    frames = (reference, candidates, synthetic)
    arrays = [[], [], []]
    for column in columns:
        if any(not pd.api.types.is_numeric_dtype(f[column]) for f in frames):
            categories = sorted(set().union(*(f[column].astype(str).unique() for f in frames)))
            mapping = {category: i for i, category in enumerate(categories)}
            values = [f[column].astype(str).map(mapping).to_numpy(dtype=np.float64) for f in frames]
        else:
            values = [f[column].to_numpy(dtype=np.float64) for f in frames]
        for dest, value in zip(arrays, values):
            if not np.isfinite(value).all():
                raise ValueError(f'{column}: 非有限値')
            dest.append(value)
    ref, cand, syn = (np.column_stack(parts) for parts in arrays)
    mean, scale = ref.mean(axis=0), ref.std(axis=0)
    scale[scale == 0] = 1.0
    return (ref-mean)/scale, (cand-mean)/scale, (syn-mean)/scale


def ordinal_squared(left, right):
    # メンバー実装の数値演算順序も維持する。
    return np.maximum(np.sum(left*left, axis=1)[:, None]
                      + np.sum(right*right, axis=1)[None, :] - 2.0*left@right.T, 0.)


def ordinal_core(reference, candidates, synthetic, h, k=K, batch_size=128):
    """メンバー案の固定帯域更新式。全属性に同じGaussian KDE。"""
    bases = []
    for start in range(0, len(synthetic), batch_size):
        sq = ordinal_squared(synthetic[start:start+batch_size], reference)
        bases.append(logsumexp(-sq/(2*h*h), axis=1))
    base = np.concatenate(bases)
    result = np.empty(len(candidates))
    baseline = np.log(len(reference)/(len(reference)+1.))
    for start in range(0, len(candidates), batch_size):
        sq = ordinal_squared(candidates[start:start+batch_size], synthetic)
        near = np.argpartition(sq, k-1, axis=1)[:, :k]
        logk = -np.take_along_axis(sq, near, axis=1)/(2*h*h)
        logden = base[near]
        result[start:start+len(sq)] = (baseline+np.logaddexp(logden, logk)-logden).sum(axis=1)
    if not np.isfinite(result).all():
        raise ArithmeticError('Gen-LRAに非有限値')
    return result


def ordinal_scores(reference, candidates, synthetic, columns, h_dimension):
    r, b, c = ordinal_matrices(reference, candidates, synthetic, columns)
    return ordinal_core(r, b, c, silverman(len(r), h_dimension))


def split_confidence(normal_raw, rare_raw, rare, private):
    """順位保存、同点維持。非希少>.5、希少<.5、確定非会員=0。"""
    out = np.zeros(len(rare))
    normal = ~rare & ~private
    possible_rare = rare & ~private
    if normal.any():
        a = rankdata(normal_raw[normal], method='min')
        out[normal] = .5+.5*(.1+.8*a/normal.sum())
    if possible_rare.any():
        descending = rankdata(-rare_raw[possible_rare], method='min')
        out[possible_rare] = .5*(.9-.8*(descending-1)/possible_rare.sum())
    return out


def global_confidence(raw, private=None):
    result = (rankdata(raw, method='average')-.5)/len(raw)
    if private is not None:
        result[private] = 0.
    return result


VARIANTS = {
    'A_mixed7_9': ('予備戦型7/9属性・初回用', '基準'),
    'B_ordinal7_9_fixedh': ('順序符号化7/9属性・帯域数値固定', 'Aと同属性・同帯域数値で前処理とカーネルを変更'),
    'C_ordinal7_9_autoh': ('順序符号化7/9属性・自動帯域', 'Bから帯域の次元だけ変更'),
    'D_ordinal10': ('順序符号化10属性', 'Cから全員10属性・帯域も変更する構成比較'),
    'E_ordinal13_fixedh10': ('順序符号化13属性・10次元帯域固定', 'Dから転帰3属性だけ追加（近傍も変わる）'),
    'F_ordinal13_autoh': ('順序符号化13属性・自動帯域', 'Eから13次元自動帯域へ変更'),
    'G_member_global_repaired': ('メンバー既定・全体順位・非会員修正', 'Fから希少と非希少の値域分離だけ除去'),
    'H_member_original': ('メンバー既定・除外なし参考', 'Gから専用163人の0化を除去。提出候補ではない'),
}


def calculate(reference, candidates, synthetic, rare, private):
    """推定に渡すのは公開許可データ、自己希少ラベル、確定非会員だけ。"""
    raw = {
        'mixed7': mixed_scores(reference, candidates, synthetic, NUM7),
        'mixed9': mixed_scores(reference, candidates, synthetic, NUM9),
        'ordinal7_h5': ordinal_scores(reference, candidates, synthetic, QI7, 5),
        'ordinal9_h7': ordinal_scores(reference, candidates, synthetic, ATTR9, 7),
        'ordinal7_h7': ordinal_scores(reference, candidates, synthetic, QI7, 7),
        'ordinal9_h9': ordinal_scores(reference, candidates, synthetic, ATTR9, 9),
        'ordinal10': ordinal_scores(reference, candidates, synthetic, COV10, 10),
        'ordinal13_h10': ordinal_scores(reference, candidates, synthetic, ALL13, 10),
        'ordinal13_h13': ordinal_scores(reference, candidates, synthetic, ALL13, 13),
    }
    pairs = [('mixed7','mixed9'), ('ordinal7_h5','ordinal9_h7'),
             ('ordinal7_h7','ordinal9_h9'), ('ordinal10','ordinal10'),
             ('ordinal13_h10','ordinal13_h10'), ('ordinal13_h13','ordinal13_h13')]
    answers = {name: split_confidence(raw[n], raw[r], rare, private)
               for name, (n,r) in zip(list(VARIANTS)[:6], pairs)}
    answers['G_member_global_repaired'] = global_confidence(raw['ordinal13_h13'], private)
    answers['H_member_original'] = global_confidence(raw['ordinal13_h13'])
    return raw, answers
