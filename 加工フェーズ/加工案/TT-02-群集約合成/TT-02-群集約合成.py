#!/usr/bin/env python3
"""TT 加工案 v1.0.0。個票入力は --b / --a-bg の2つだけ。

指定版: main-process-20260912。列・値域・Hは公式schema/config由来。
学習と生成は指定入力から毎回実施。課題・正解・自己採点資料は読み込まない。
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import resource
import sys
import time
import unicodedata
import warnings

# スレッド数も固定し、同一環境での数値演算の順序を安定させる。
for _key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_key] = "1"
import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri
from scipy.stats import rankdata
from lifelines import CoxPHFitter

# 以下はチーム内の実験設定。大会が指定したパラメータではない。
VERSION = "1.0.0"
SEED = 20260920
B_WEIGHT = 0.85                 # Bの集約統計の重み。残りをA_bgへ縮約する。
MIN_STAT_ROWS = 16             # 小さな標本はA_bgへ戻し、個人残差を復元しない。
CORR_RIDGE = 0.08              # 共分散・相関の安定化
MIN_GROUP = 32                 # 案02: 分割後の最小群人数（匿名性の保証ではない）
NEIGHBORS = 128                # 案02/03: 背景の条件付き分布を作る候補数
DONOR_SMOOTH = 0.18            # 案03: 背景標準偏差に対する連続値の平滑化幅
COX_RIDGE = 0.002              # 学習の数値安定化。公式の無罰則採点器とは別物。
COX_STEPS = 3                  # 案01/02: 合成標本の係数を集約目標へ近づける固定回数
GRID = np.array([0, .005, .01, .025, .05, .1, .2, .35, .5, .65, .8, .9, .95, .975, .99, .995, 1.])

# 公式定数の出典は末尾のSOURCEに版とSHA-256を記録する。
COLS = ["age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL", "ALT", "smoking", "FPG", "time", "onset", "death"]
CONT = ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG"]
FEATURES = CONT + ["sex", "smoking"]
RANGES = {"age": (40, 74), "BMI": (10, 55), "SBP": (40, 250), "TG": (1, 3000), "HDL": (10, 160), "ALT": (.1, 700), "FPG": (40, 400)}
PREFECTURES = "北海道 青森県 岩手県 宮城県 秋田県 山形県 福島県 茨城県 栃木県 群馬県 埼玉県 千葉県 東京都 神奈川県 新潟県 富山県 石川県 福井県 山梨県 長野県 岐阜県 静岡県 愛知県 三重県 滋賀県 京都府 大阪府 兵庫県 奈良県 和歌山県 鳥取県 島根県 岡山県 広島県 山口県 徳島県 香川県 愛媛県 高知県 福岡県 佐賀県 長崎県 熊本県 大分県 宮崎県 鹿児島県 沖縄県".split()
HORIZON = 10.0


METHOD = '群集約合成'
SOURCE = {'version': 'main-process-20260912', 'release_commit': '3dd71bd', 'schema_sha256': '1b79ba5b45cfdb80ebc67c120b4066bc2d89ac232579397fc3422d2ca21373ae', 'config_sha256': 'b5e92c09d9a00b6399aac450c58adabb65e459d2e3820164292b2cf632c3db22', 'url': 'https://github.com/pwscup/pwscup2026-kit/releases/tag/main-process-20260912'}
RARE_MU = [56.49594732888937, 23.163723280417887, 123.59351652486671, 100.60218604186069, 47.24784926016865, 18.921471743692987, 91.94078130951313]
RARE_SIGMA = [[103.08287129374209, -1.649013576927273, 54.65870323613054, 32.65914263768858, 0.08233031645032023, -17.95059613281468, 27.96694733006783], [-1.649013576927273, 15.257206396845461, 28.23331213265221, 100.49262296098792, -0.17122193051188014, 23.838155099371633, 14.042337820328894], [54.65870323613054, 28.23331213265221, 354.4306576745582, 326.8011870016662, 9.671647005807031, 56.0782542771646, 77.87298727450076], [32.65914263768858, 100.49262296098792, 326.8011870016662, 6141.354379874466, -21.545280794659792, 439.5369651138725, 254.5280468208063], [0.08233031645032023, -0.17122193051188014, 9.671647005807031, -21.545280794659792, 69.19024424568744, 22.424017514620967, 18.604446682180594], [-17.95059613281468, 23.838155099371633, 56.0782542771646, 439.5369651138725, 22.424017514620967, 302.0540090035288, 41.834695631413425], [27.96694733006783, 14.042337820328894, 77.87298727450076, 254.5280468208063, 18.604446682180594, 41.834695631413425, 222.04524025853107]]
RARE_LO = 4.0


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(df, label, with_id=True):
    """列・単一値・型・値域・論理整合を生成前後に検査する。"""
    expected = (["record_id"] if with_id else []) + COLS
    if len(df) == 0 or len(df.columns) != len(expected) or set(df.columns) != set(expected):
        raise ValueError(f"{label}: 指定列と一致しないか、行がありません")
    if df.isna().any().any():
        raise ValueError(f"{label}: 欠測があります")
    df = df.loc[:, expected].copy()
    df["prefecture"] = df.prefecture.map(lambda v: unicodedata.normalize("NFC", str(v)))
    if not df.prefecture.isin(PREFECTURES).all():
        raise ValueError(f"{label}: 都道府県名が不正です")
    for col in [v for v in expected if v != "prefecture"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(df[col].to_numpy(float)).all():
            raise ValueError(f"{label}: {col}に非有限値があります")
    for col in (["record_id"] if with_id else []) + ["age", "sex", "smoking", "onset", "death"]:
        if (df[col] != np.rint(df[col])).any():
            raise ValueError(f"{label}: {col}は整数が必要です")
        if col == "record_id" and (df[col].duplicated().any() or (df[col].abs() > 2**53-1).any()):
            raise ValueError(f"{label}: record_idは安全な範囲の一意な整数が必要です")
        df[col] = df[col].astype(np.int64)
    for col, (lo, hi) in RANGES.items():
        if not df[col].between(lo, hi).all():
            raise ValueError(f"{label}: {col}が値域[{lo},{hi}]外です")
    for col in ["sex", "smoking", "onset", "death"]:
        if not df[col].isin([0, 1]).all():
            raise ValueError(f"{label}: {col}は0/1が必要です")
    if not ((df.time > 0) & (df.time <= HORIZON)).all() or ((df.onset + df.death) > 1).any():
        raise ValueError(f"{label}: timeまたは発症・死亡の排他条件が不正です")
    mask = (df.onset == 0) & (df.death == 0)
    if not np.isclose(df.loc[mask, "time"], HORIZON, atol=.01, rtol=1e-5).all():
        raise ValueError(f"{label}: 非発症・非死亡のtimeが打ち切りと不整合です")
    return df


def read_input(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f), [])
    if len(header) != len(set(header)):
        raise ValueError(f"{path}: 列名が重複しています")
    df = validate(pd.read_csv(path, encoding="utf-8-sig", float_precision="round_trip"), Path(path).name)
    # 内容で正準化する。record_id・元の行順は学習も対応付けも支配しない。
    return df.sort_values(COLS, kind="stable").reset_index(drop=True)


def matrix(df):
    x = df[FEATURES].to_numpy(float).copy()
    x[:, CONT.index("TG")] = np.log(x[:, CONT.index("TG")])
    x[:, CONT.index("ALT")] = np.log(x[:, CONT.index("ALT")])
    return x


def frame(x):
    x = x.copy()
    for col in ["TG", "ALT"]:
        j = CONT.index(col)
        x[:, j] = np.exp(np.clip(x[:, j], -10, 10))
    out = pd.DataFrame(x, columns=FEATURES)
    for col, bounds in RANGES.items():
        out[col] = np.clip(out[col], *bounds)
    out.age = np.rint(out.age).astype(int)
    for col in ["sex", "smoking"]:
        out[col] = (out[col] >= .5).astype(int)
    return out


def rare(df):
    """公開検出器による層化。真のis_rareではない点を区別する。"""
    d = df[CONT].to_numpy(float) - np.asarray(RARE_MU)
    return np.einsum("ij,jk,ik->i", d, np.linalg.pinv(np.asarray(RARE_SIGMA)), d) >= RARE_LO**2


def blend_stats(b, bg):
    # 16行未満の群からは私的な分位点・相関を学習しない。
    if len(b) < MIN_STAT_ROWS:
        return bg, bg, 0.
    return b, bg, B_WEIGHT


def stable_cov(x):
    cov = np.cov(x, rowvar=False) if len(x) > 1 else np.zeros((x.shape[1], x.shape[1]))
    cov = np.nan_to_num(cov)
    return (1-CORR_RIDGE)*cov + CORR_RIDGE*np.diag(np.maximum(np.diag(cov), 1e-5)) + 1e-7*np.eye(x.shape[1])


def draw_normal(n, cov, rng):
    # 固定した固有値分解とPCG64。負の丸め誤差だけを0へ補正する。
    vals, vecs = np.linalg.eigh(cov)
    return rng.standard_normal((n, len(vals))) @ (vecs*np.sqrt(np.maximum(vals, 0))).T


def assign_pref(c, b, bg, rng):
    """年齢帯・性別ごとの県分布を縮約して抽出。B個人の県は対応させない。"""
    values = np.array(PREFECTURES, dtype=object)
    global_p = bg.prefecture.value_counts().reindex(values, fill_value=0).to_numpy(float) + .5
    global_p /= global_p.sum()
    result = np.empty(len(c), dtype=object)
    for sex in [0, 1]:
        for low in [40, 50, 60, 70]:
            idx = np.flatnonzero((c.sex == sex) & c.age.between(low, min(low+9, 74)))
            sub = b[(b.sex == sex) & b.age.between(low, min(low+9, 74))]
            counts = sub.prefecture.value_counts().reindex(values, fill_value=0).to_numpy(float)
            # 32人分の背景事前分布を加え、少数の県の一点指定を避ける。
            prob = (counts + MIN_GROUP*global_p)/(len(sub)+MIN_GROUP)
            result[idx] = rng.choice(values, len(idx), p=prob)
    c["prefecture"] = result
    return c


def cox_fit(df, center, scale, event, log):
    """2原因を別々のCoxで学習。失敗時は明記し、定数ハザードへ戻す。"""
    z = (matrix(df)-center)/scale
    active = np.std(z, axis=0) > 1e-7
    grid = np.linspace(0, HORIZON, 21)
    beta = np.zeros(len(FEATURES))
    if int(df[event].sum()) >= 10 and active.any():
        data = pd.DataFrame(z[:, active], columns=np.flatnonzero(active).astype(str))
        data["time"], data["event"] = df.time.to_numpy(), df[event].to_numpy()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                fit = CoxPHFitter(penalizer=COX_RIDGE).fit(data, "time", "event", fit_options={"max_steps": 100})
            beta[active] = fit.params_.to_numpy()
            # lifelinesの平均中心化基準を、共通のA_bg中心へ変換する。
            bh = fit.baseline_cumulative_hazard_
            hazard = np.interp(grid, np.r_[0., bh.index.to_numpy()], np.r_[0., bh.iloc[:, 0].to_numpy()])
            hazard *= np.exp(np.clip(-np.dot(z.mean(axis=0), beta), -10, 10))
            if np.isfinite(beta).all() and np.isfinite(hazard).all():
                return beta, hazard
        except Exception as exc:
            log.append(f"Cox {event}: {type(exc).__name__}。定数ハザードへ切替")
    else:
        log.append(f"Cox {event}: イベント10件未満または定数列のみ。定数ハザードへ切替")
    hazard = grid*(float(df[event].sum())/max(float(df.time.sum()), 1e-9))
    return beta*0, hazard


def synth_outcomes(c, b, bg, rng, log, offsets=None):
    """Bの係数・累積ハザードを背景へ縮約し、新しい競合イベントを発生させる。"""
    xbg = matrix(bg)
    center, scale = xbg.mean(axis=0), np.maximum(xbg.std(axis=0), .05)
    z = (matrix(c)-center)/scale
    models = []
    for event in ["onset", "death"]:
        bb, hb = cox_fit(b, center, scale, event, log)
        ba, ha = cox_fit(bg, center, scale, event, log)
        w = B_WEIGHT if len(b) >= MIN_STAT_ROWS else 0.
        models.append([w*bb+(1-w)*ba, w*hb+(1-w)*ha])
    target_beta = models[0][0].copy()
    # 層化一様乱数で標本揺れを抑える。個人ごとの元timeは使用しない。
    expo = np.column_stack([-np.log((rng.permutation(len(c))+.5)/len(c)) for _ in range(2)])
    grid = np.linspace(0, HORIZON, 21)
    for step in range(COX_STEPS+1):
        times = np.full((len(c), 2), HORIZON+1.)
        for j, (beta, hazard) in enumerate(models):
            rate = np.exp(np.clip(z @ beta, -6, 6))
            if offsets is not None:
                rate *= offsets[:, j]
            threshold = expo[:, j]/rate
            times[:, j] = np.interp(threshold, hazard, grid, left=1e-6, right=HORIZON+1.)
        # 早い原因だけが発症し、両原因が期間外ならtime=10で打ち切る。
        cause = np.argmin(times, axis=1)
        t = times[np.arange(len(c)), cause]
        c["time"] = np.minimum(t, HORIZON)
        c["onset"] = ((cause == 0) & (t <= HORIZON)).astype(int)
        c["death"] = ((cause == 1) & (t <= HORIZON)).astype(int)
        if step < COX_STEPS:
            fitted, _ = cox_fit(c, center, scale, "onset", log)
            # 回数・学習率は固定。課題や採点点数による候補選択はしない。
            models[0][0] = np.clip(models[0][0]+.5*(target_beta-fitted), -3, 3)
    return c


def save_preserving(path, content):
    """再生成後に差があれば、以前の出力を内容ハッシュ名で保存する。"""
    if path.exists() and path.read_bytes() != content:
        old = path.parent / "旧出力"
        old.mkdir(exist_ok=True)
        dest = old / f"{path.stem}_{sha(path)[:16]}{path.suffix}"
        if not dest.exists():
            dest.write_bytes(path.read_bytes())
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--b", required=True, type=Path)
    p.add_argument("--a-bg", required=True, type=Path)
    p.add_argument("--seed", type=int, default=SEED, help="安定性確認用。既定値を採用候補の固定seedとする")
    a = p.parse_args()
    if a.seed < 0:
        raise ValueError("seedは非負整数が必要です")
    out = Path(__file__).resolve().with_name("C_"+Path(__file__).stem+".csv")
    if out in [a.b.resolve(), a.a_bg.resolve()]:
        raise ValueError("出力先が入力と同じです")
    start = time.perf_counter()
    hashes = {str(q.resolve()): sha(q) for q in [a.b, a.a_bg]}
    b, bg = read_input(a.b), read_input(a.a_bg)
    if len(bg) < MIN_STAT_ROWS:
        raise ValueError(f"本手法はA_bgが{MIN_STAT_ROWS}行以上必要です")
    log = []
    c = generate(b, bg, np.random.Generator(np.random.PCG64(a.seed)), log)
    c = validate(c.loc[:, COLS], "生成結果", with_id=False)
    if len(c) != len(b):
        raise ValueError("生成行数がBと一致しません")
    # 各数値の表記を固定。record_idを出さず、行の内容で正準化する。
    c = c.sort_values(COLS, kind="stable").reset_index(drop=True)
    stream = io.StringIO(newline="")
    c.to_csv(stream, index=False, lineterminator="\n", float_format="%.10f")
    content = stream.getvalue().encode("utf-8")
    validate(pd.read_csv(io.BytesIO(content), float_precision="round_trip"), "保存前検査", False)
    if any(sha(q) != h for q, h in hashes.items()):
        raise ValueError("実行中に入力ファイルが変更されました")
    save_preserving(out, content)
    memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    memory_mb = memory/(1024**2 if sys.platform == "darwin" else 1024)
    record = dict(version=VERSION, method=METHOD, seed=a.seed, rows=len(c), input_sha256=hashes,
                  code_sha256=sha(__file__), output_sha256=sha(out), seconds=time.perf_counter()-start,
                  peak_rss_mb=memory_mb, warnings=log, source=SOURCE,
                  versions={v: importlib.metadata.version(v) for v in ["numpy", "pandas", "scipy", "lifelines"]},
                  python=sys.version, settings=dict(B_WEIGHT=B_WEIGHT, MIN_STAT_ROWS=MIN_STAT_ROWS,
                  CORR_RIDGE=CORR_RIDGE, MIN_GROUP=MIN_GROUP, NEIGHBORS=NEIGHBORS,
                  DONOR_SMOOTH=DONOR_SMOOTH, COX_RIDGE=COX_RIDGE, COX_STEPS=COX_STEPS, GRID=GRID.tolist()))
    save_preserving(out.with_suffix(".実行記録.json"), (json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode())
    print(json.dumps(dict(output=out.name, rows=len(c), seconds=record["seconds"], warnings=log), ensure_ascii=False))

def generate(b, bg, rng, log):
    """Bを内容で群に分け、群平均と背景の局所共分散から新しい人を生成する。"""
    xb, xa = matrix(b), matrix(bg)
    center, scale = xa.mean(axis=0), np.maximum(xa.std(axis=0), .05)
    zb, za = (xb-center)/scale, (xa-center)/scale
    groups = []

    def split(idx):
        if len(idx) < 2*MIN_GROUP:
            groups.append(idx)
            return
        # 最も散らばった連続属性で中央値分割。分割軸・同点順は固定。
        j = int(np.argmax(np.var(zb[idx, :len(CONT)], axis=0)))
        order = idx[np.argsort(zb[idx, j], kind="stable")]
        mid = len(order)//2
        split(order[:mid])
        split(order[mid:])

    # 外れ値を通常群に吸収して消さない。ただし小群は私的中心を使わない。
    r = rare(b)
    for mask in [~r, r]:
        idx = np.flatnonzero(mask)
        if len(idx):
            split(idx)
    blocks, offsets = [], []
    global_rates = (b[["onset", "death"]].sum().to_numpy()+1)/(len(b)+2)
    for idx in groups:
        n = len(idx)
        if n < MIN_STAT_ROWS:
            pool = bg[rare(bg) == bool(r[idx[0]])]
            if len(pool) < MIN_STAT_ROWS:
                pool = bg
            local = matrix(pool)
            mu, covariance = local.mean(axis=0), stable_cov(local)
            offset = np.ones(2)
            log.append(f"{n}人群: 背景分布へ戻し、Bの群平均・転帰率は未使用")
        else:
            private_mean = xb[idx].mean(axis=0)
            dist = np.square(za-(private_mean-center)/scale).sum(axis=1)
            neighbors = np.argsort(dist, kind="stable")[:min(NEIGHBORS, len(bg))]
            local = xa[neighbors]
            mu = B_WEIGHT*private_mean+(1-B_WEIGHT)*local.mean(axis=0)
            # 個人残差の再利用ではなく、群と背景の共分散だけを混ぜる。
            covariance = .5*stable_cov(xb[idx])+.5*stable_cov(local)
            rate = (b.iloc[idx][["onset", "death"]].sum().to_numpy()+MIN_GROUP*global_rates)/(n+MIN_GROUP)
            offset = np.clip(rate/global_rates, .5, 2.)
        x = mu+draw_normal(n, covariance, rng)
        for j in [7, 8]:
            # 潜在値の順位を用い、群内の2値属性の件数を集約平均へ合わせる。
            order = np.argsort(x[:, j], kind="stable")
            count = int(np.rint(n*np.clip(mu[j], 0, 1)))
            x[:, j] = 0
            if count:
                x[order[-count:], j] = 1
        blocks.append(frame(x))
        offsets.append(np.tile(offset, (n, 1)))
    c = assign_pref(pd.concat(blocks, ignore_index=True), b, bg, rng)
    return synth_outcomes(c, b, bg, rng, log, np.vstack(offsets))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, ImportError) as exc:
        print(f"入力・環境エラー: {exc}", file=sys.stderr)
        sys.exit(2)
