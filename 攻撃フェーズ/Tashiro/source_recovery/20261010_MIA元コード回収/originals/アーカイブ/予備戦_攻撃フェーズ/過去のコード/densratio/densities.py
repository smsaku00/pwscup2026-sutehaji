"""密度比 MIA/AIA 用の密度推定器ズー。

共通インタフェース
------------------
DensityModel : fit(X) -> self ; logpdf(Y) -> (m,)
RatioModel   : fit(X_num, X_den) -> self ; log_ratio(Y) -> (m,)

すべて「A_bg で z 標準化済みの連続列行列」を受け取る前提（公式参照実装 attack/mia.py と同じ空間）。
"""
from __future__ import annotations
import numpy as np
from scipy.stats import gaussian_kde, multivariate_normal, norm
from scipy.spatial import cKDTree
from scipy.special import gammaln, logsumexp
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

LOGEPS = -700.0


# ---------------------------------------------------------------- 密度モデル
class KDEModel:
    """D1: ガウシアン KDE。公式参照実装（scipy.gaussian_kde, Scott）と同一。h で帯域を倍する。"""
    def __init__(self, h=1.0, rule="scott"):
        self.h, self.rule = float(h), rule
        self.name = f"KDE(h={h:g})" if h != 1.0 else "KDE(Scott)"

    def fit(self, X):
        n, d = X.shape
        base = n ** (-1.0 / (d + 4)) if self.rule == "scott" else (n * (d + 2) / 4.0) ** (-1.0 / (d + 4))
        try:
            self.kde_ = gaussian_kde(X.T, bw_method=base * self.h)
        except np.linalg.LinAlgError:
            self.kde_ = None
        return self

    def logpdf(self, Y):
        if self.kde_ is None:
            return np.zeros(len(Y))
        out = np.empty(len(Y))
        for s in range(0, len(Y), 4000):          # メモリ節約
            out[s:s + 4000] = self.kde_.logpdf(Y[s:s + 4000].T)
        return out


class GaussModel:
    """D2: 多変量正規（平均・共分散のみ）。2次モーメントだけを見る密度。"""
    name = "Gauss(full-cov)"

    def fit(self, X):
        self.mu_ = X.mean(0)
        cov = np.cov(X.T) + 1e-6 * np.eye(X.shape[1])
        self.rv_ = multivariate_normal(self.mu_, cov, allow_singular=True)
        return self

    def logpdf(self, Y):
        return self.rv_.logpdf(Y)


class GMMModel:
    """D3: ガウス混合。K は BIC 選択（局所構造を拾う）。"""
    def __init__(self, ks=(1, 2, 3, 5, 8, 12, 16, 20), seed=0):
        self.ks, self.seed = ks, seed
        self.name = "GMM(BIC)"

    def fit(self, X):
        best, bestbic = None, np.inf
        for k in self.ks:
            if k * X.shape[1] > len(X) / 3:
                continue
            g = GaussianMixture(k, covariance_type="full", reg_covar=1e-4,
                                random_state=self.seed, n_init=2, max_iter=400).fit(X)
            b = g.bic(X)
            if b < bestbic:
                best, bestbic = g, b
        self.gm_ = best
        self.name = f"GMM(K={best.n_components})"
        return self

    def logpdf(self, Y):
        return self.gm_.score_samples(Y)


class KNNModel:
    """D4: k 近傍密度（Loftsgaarden-Quesenberry）。log p = log k - log n - log V_d - d log R_k。"""
    def __init__(self, k=25):
        self.k = int(k)
        self.name = f"kNN(k={k})"

    def fit(self, X):
        self.tree_ = cKDTree(X)
        self.n_, self.d_ = X.shape
        self.logV_ = self.d_ / 2 * np.log(np.pi) - gammaln(self.d_ / 2 + 1)
        return self

    def logpdf(self, Y):
        r, _ = self.tree_.query(Y, k=self.k, workers=-1)
        rk = np.maximum(r[:, -1], 1e-10)
        return np.log(self.k) - np.log(self.n_) - self.logV_ - self.d_ * np.log(rk)


class CopulaModel:
    """D7: ガウシアン・コピュラ密度 = 周辺（1次元KDE）× 依存構造（ガウスコピュラ）。

    log p = Σ_j log f_j(y_j) + log c(u)  を分解して保持するので、
    漏洩が「周辺分布」由来か「依存構造」由来かを切り分けられる。
    """
    name = "Copula(marg×gauss)"

    def fit(self, X):
        self.n_, self.d_ = X.shape
        self.marg_ = [gaussian_kde(X[:, j]) for j in range(self.d_)]
        self.xs_ = [np.sort(X[:, j]) for j in range(self.d_)]
        z = np.column_stack([norm.ppf(self._ecdf(j, X[:, j])) for j in range(self.d_)])
        R = np.corrcoef(z.T) + 1e-6 * np.eye(self.d_)
        self.Rinv_ = np.linalg.inv(R)
        self.logdetR_ = np.linalg.slogdet(R)[1]
        return self

    def _ecdf(self, j, v):
        r = np.searchsorted(self.xs_[j], v, side="right")
        return np.clip((r + 0.5) / (self.n_ + 1.0), 1e-6, 1 - 1e-6)

    def parts(self, Y):
        lm = np.zeros(len(Y))
        z = np.empty_like(Y)
        for j in range(self.d_):
            lm += np.log(np.maximum(self.marg_[j](Y[:, j]), 1e-300))
            z[:, j] = norm.ppf(self._ecdf(j, Y[:, j]))
        q = np.einsum("ij,jk,ik->i", z, self.Rinv_ - np.eye(self.d_), z)
        lc = -0.5 * self.logdetR_ - 0.5 * q
        return lm, lc

    def logpdf(self, Y):
        lm, lc = self.parts(Y)
        return lm + lc


# ---------------------------------------------------------------- 比モデル
class TwoDensityRatio:
    """密度を2本推定して比を取る（DOMIAS 型）。"""
    def __init__(self, maker):
        self.maker = maker
        self.name = maker().name

    def fit(self, Xn, Xd):
        self.pn_ = self.maker().fit(Xn)
        self.pd_ = self.maker().fit(Xd)
        self.name = self.pn_.name
        return self

    def log_ratio(self, Y):
        return np.clip(self.pn_.logpdf(Y) - self.pd_.logpdf(Y), -200, 200)


class ClassifierRatio:
    """D5: 確率的分類による直接密度比推定。

    log r(y) = logit P(C|y) + log(n_den/n_num)。C 対 参照の識別器を K-fold で
    クロスフィットし、評価点でのロジットを平均する。
    """
    def __init__(self, kind="logit2", folds=5, seed=0):
        self.kind, self.folds, self.seed = kind, folds, seed
        self.name = {"logit2": "Clf-DRE(logistic 2次)", "hgb": "Clf-DRE(勾配ブースト)"}[kind]

    def _mk(self):
        if self.kind == "logit2":
            return make_pipeline(StandardScaler(),
                                 PolynomialFeatures(2, include_bias=False),
                                 StandardScaler(),
                                 LogisticRegression(C=0.5, max_iter=4000))
        return HistGradientBoostingClassifier(max_depth=3, max_iter=250, learning_rate=0.08,
                                              l2_regularization=1.0, random_state=self.seed)

    def fit(self, Xn, Xd):
        X = np.vstack([Xn, Xd])
        y = np.r_[np.ones(len(Xn)), np.zeros(len(Xd))]
        self.models_ = []
        skf = StratifiedKFold(self.folds, shuffle=True, random_state=self.seed)
        for tr, _ in skf.split(X, y):
            self.models_.append(self._mk().fit(X[tr], y[tr]))
        self.offset_ = np.log(len(Xd) / len(Xn))
        return self

    def log_ratio(self, Y):
        L = []
        for m in self.models_:
            p = np.clip(m.predict_proba(Y)[:, 1], 1e-7, 1 - 1e-7)
            L.append(np.log(p / (1 - p)))
        return np.mean(L, 0) + self.offset_


class RuLSIF:
    """D6: uLSIF / RuLSIF（Sugiyama の直接密度比推定）。

    r_a(y) = p_num(y) / (a p_num(y) + (1-a) p_den(y))、ガウス核基底 b 個の線形和で
    二乗損失を最小化する解析解。a=0 が素の密度比（uLSIF）。
    """
    def __init__(self, alpha=0.0, b=200, seed=0):
        self.alpha, self.b, self.seed = float(alpha), int(b), seed
        self.name = f"{'uLSIF' if alpha == 0 else f'RuLSIF(a={alpha:g})'}"

    @staticmethod
    def _K(A, C, s):
        d2 = ((A[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        return np.exp(-d2 / (2 * s * s))

    def fit(self, Xn, Xd):
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(Xn), min(self.b, len(Xn)), replace=False)
        self.C_ = Xn[idx]
        sub = Xn[rng.choice(len(Xn), min(400, len(Xn)), replace=False)]
        med = np.median(np.sqrt(((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)))
        best, bestJ = None, np.inf
        for smul in (0.5, 1.0, 2.0):
            s = max(med * smul, 1e-3)
            Pn, Pd = self._K(Xn, self.C_, s), self._K(Xd, self.C_, s)
            H = self.alpha * (Pn.T @ Pn) / len(Xn) + (1 - self.alpha) * (Pd.T @ Pd) / len(Xd)
            h = Pn.mean(0)
            for lam in (1e-3, 1e-2, 1e-1, 1.0):
                a = np.linalg.solve(H + lam * np.eye(len(self.C_)), h)
                J = 0.5 * a @ H @ a - h @ a          # uLSIF の目的関数（小さいほど良い）
                if J < bestJ:
                    best, bestJ = (s, a), J
        self.s_, self.a_ = best
        return self

    def log_ratio(self, Y):
        r = self._K(Y, self.C_, self.s_) @ self.a_
        return np.log(np.maximum(r, 1e-8))


# ---------------------------------------------------------------- ズー定義
def ratio_zoo(fast=False):
    """(表示名, 生成関数) のリスト。"""
    z = [
        ("D1 KDE(Scott)",      lambda: TwoDensityRatio(lambda: KDEModel(1.0))),
        ("D1b KDE(h=0.5)",     lambda: TwoDensityRatio(lambda: KDEModel(0.5))),
        ("D1c KDE(h=0.25)",    lambda: TwoDensityRatio(lambda: KDEModel(0.25))),
        ("D1d KDE(h=2.0)",     lambda: TwoDensityRatio(lambda: KDEModel(2.0))),
        ("D2 Gauss",           lambda: TwoDensityRatio(GaussModel)),
        ("D3 GMM(BIC)",        lambda: TwoDensityRatio(GMMModel)),
        ("D4 kNN(k=25)",       lambda: TwoDensityRatio(lambda: KNNModel(25))),
        ("D4b kNN(k=5)",       lambda: TwoDensityRatio(lambda: KNNModel(5))),
        ("D5 Clf-DRE(logit2)", lambda: ClassifierRatio("logit2")),
        ("D5b Clf-DRE(HGB)",   lambda: ClassifierRatio("hgb")),
        ("D6 uLSIF",           lambda: RuLSIF(0.0)),
        ("D6b RuLSIF(a=0.5)",  lambda: RuLSIF(0.5)),
        ("D7 Copula",          lambda: TwoDensityRatio(CopulaModel)),
    ]
    if fast:
        z = [e for e in z if e[0].split()[0] in ("D1", "D2", "D3", "D4", "D5", "D6", "D7")]
    return z
