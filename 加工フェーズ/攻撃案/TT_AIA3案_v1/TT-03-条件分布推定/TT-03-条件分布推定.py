#!/usr/bin/env python3
"""TT-03-条件分布推定: 単独実行可能なAIA。詳細は同名Markdown参照。"""
from __future__ import annotations

import os

# 同じ環境での演算順を固定する。ライブラリの import より先に設定する。
for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_var] = "1"

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import re
import resource
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import scipy
from scipy.special import logsumexp

VERSION = "1.0.0"
RULEBOOK_VERSION = "main-process-20260912"
SEED = 20260920
HORIZON = 10.0
TAU = 0.5
FLOAT_FORMAT = ".10f"
CONT = ["age", "SBP", "TG", "HDL", "ALT"]
QI = CONT + ["sex", "prefecture"]
C_COLS = ["age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL", "ALT",
          "smoking", "FPG", "time", "onset", "death"]
BG_COLS = ["record_id"] + C_COLS
Q_COLS = ["challenge_row_id"] + QI
RANGES = {"age": (40, 74), "BMI": (10, 55), "SBP": (40, 250),
          "TG": (1, 3000), "HDL": (10, 160), "ALT": (0.1, 700), "FPG": (40, 400)}
PREFECTURES = ("北海道 青森県 岩手県 宮城県 秋田県 山形県 福島県 茨城県 栃木県 群馬県 "
               "埼玉県 千葉県 東京都 神奈川県 新潟県 富山県 石川県 福井県 山梨県 長野県 "
               "岐阜県 静岡県 愛知県 三重県 滋賀県 京都府 大阪府 兵庫県 奈良県 和歌山県 "
               "鳥取県 島根県 岡山県 広島県 山口県 徳島県 香川県 愛媛県 高知県 福岡県 "
               "佐賀県 長崎県 熊本県 大分県 宮崎県 鹿児島県 沖縄県").split()


import sklearn
from sklearn.ensemble import ExtraTreesRegressor

NAME = "TT-03-条件分布推定"
SETTINGS = {"trees": 96, "min_samples_leaf": 5, "max_depth": 16, "max_features": 1.0,
            "bootstrap": False, "n_jobs": 1, "criterion": "squared_error", "tau": TAU}



class InputError(ValueError):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strict_object(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise InputError(f"JSONのキーが重複しています: {k}")
        out[k] = v
    return out


def read_json(path):
    data = path.read_bytes()
    try:
        obj = json.loads(data.decode("utf-8-sig"), object_pairs_hook=strict_object,
                         parse_constant=lambda x: (_ for _ in ()).throw(InputError(f"非有限JSON値: {x}")))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise InputError(f"{path.name}: UTF-8 JSONを読み取れません: {exc}") from exc
    return obj, sha256(data)


def identifier(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise InputError(f"{label}: 非負の整数が必要です")
    if not re.fullmatch(r"[0-9]+", str(value)):
        raise InputError(f"{label}: 非負の整数が必要です")
    n = int(value)
    if n > np.iinfo(np.int64).max:
        raise InputError(f"{label}: 64bit整数の範囲を超えています")
    return n


def read_table(path: Path, kind: str):
    """許可した列だけを読み、ラベルなどの追加列を黙って使用しない。"""
    expected = {"background": BG_COLS, "c": C_COLS, "challenge": Q_COLS}[kind]
    raw = path.read_bytes()
    try:
        rows = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline=""), strict=True))
    except (UnicodeError, csv.Error) as exc:
        raise InputError(f"{path.name}: UTF-8 CSVを読み取れません: {exc}") from exc
    if len(rows) < 2:
        raise InputError(f"{path.name}: ヘッダーと1行以上のデータが必要です")
    head = rows[0]
    if len(head) != len(set(head)):
        raise InputError(f"{path.name}: 列名が重複しています")
    if set(head) != set(expected):
        raise InputError(f"{path.name}: 列が不一致。欠落={sorted(set(expected)-set(head))}, "
                         f"余分={sorted(set(head)-set(expected))}")
    for line, row in enumerate(rows[1:], 2):
        if len(row) != len(head) or any(not x.strip() for x in row):
            raise InputError(f"{path.name}: {line}行目に欠損または列数不一致があります")
    df = pd.DataFrame(rows[1:], columns=head)[expected]
    for col in expected:
        if col == "prefecture":
            if not df[col].isin(PREFECTURES).all():
                raise InputError(f"{path.name}: prefectureは指定版の47都道府県名が必要です")
        elif col in ("record_id", "challenge_row_id"):
            df[col] = [identifier(v, f"{path.name}:{col}") for v in df[col]]
            if df[col].duplicated().any():
                raise InputError(f"{path.name}: {col}が重複しています")
        else:
            try:
                v = pd.to_numeric(df[col], errors="raise").to_numpy(dtype=float)
            except (ValueError, TypeError) as exc:
                raise InputError(f"{path.name}: {col}を数値として読めません") from exc
            if not np.isfinite(v).all():
                raise InputError(f"{path.name}: {col}にNaNまたは無限大があります")
            if col in RANGES:
                lo, hi = RANGES[col]
                if ((v < lo) | (v > hi)).any():
                    raise InputError(f"{path.name}: {col}は[{lo},{hi}]の範囲が必要です")
            if col in ("age", "sex", "onset", "death", "smoking"):
                if (v != np.floor(v)).any():
                    raise InputError(f"{path.name}: {col}は整数が必要です")
            if col in ("sex", "onset", "death", "smoking") and not np.isin(v, [0, 1]).all():
                raise InputError(f"{path.name}: {col}は0または1が必要です")
            if col == "time" and ((v <= 0) | (v > HORIZON)).any():
                raise InputError(f"{path.name}: timeは0より大きく10以下が必要です")
            df[col] = v
    if kind != "challenge" and ((df.onset == 1) & (df.death == 1)).any():
        raise InputError(f"{path.name}: onset=1かつdeath=1の行があります")
    # Cの完全重複行は値域違反ではないため削除・拒否しない。ID重複とは区別する。
    order = ["challenge_row_id"] if kind == "challenge" else C_COLS + (["record_id"] if kind == "background" else [])
    return df.sort_values(order, kind="mergesort").reset_index(drop=True), sha256(raw)


def scale_from_background(bg):
    s = bg[CONT].to_numpy(dtype=float).std(axis=0, ddof=1) if len(bg) > 1 else np.ones(len(CONT))
    return np.where(np.isfinite(s) & (s > 0), s, 1.0)


def block_distances(query, reference, scale):
    """公式参照法の性別・県ブロック、空なら性別、さらに空なら全体。"""
    q = query[CONT].to_numpy(dtype=float) / scale
    r = reference[CONT].to_numpy(dtype=float) / scale
    d = np.sqrt(np.mean((q[:, None, :] - r[None, :, :]) ** 2, axis=2))
    for i in range(len(query)):
        sex = reference.sex.to_numpy() == query.sex.iloc[i]
        mask = sex & (reference.prefecture.to_numpy() == query.prefecture.iloc[i])
        if not mask.any():
            mask = sex
        if not mask.any():
            mask = np.ones(len(reference), dtype=bool)
        d[i, ~mask] = np.inf
    return d


def interval_mode(times, weights):
    """幅2τの窓の最大質量。等点なら窓の端の中点が大きい方。"""
    t = np.asarray(times, dtype=float)
    w = np.asarray(weights, dtype=float)
    if not len(t) or not np.isfinite(w).all() or (w < 0).any() or not (w.sum() > 0):
        raise InputError("追跡年数の分布を作れません")
    order = np.argsort(t, kind="stable")
    t, w = t[order], w[order]
    w = w / w.sum()
    cumulative = np.r_[0.0, np.cumsum(w)]
    right = np.searchsorted(t, t + 2 * TAU, side="right")
    mass = cumulative[right] - cumulative[np.arange(len(t))]
    centers = (t + t[right - 1]) / 2
    candidates = np.flatnonzero(mass >= mass.max() - 1e-12)
    return float(np.clip(centers[candidates].max(), np.nextafter(0.0, 1.0), HORIZON))


def background_prediction(bg, query, scale, k=25):
    result = []
    for start in range(0, len(query), 32):
        d = block_distances(query.iloc[start:start + 32], bg, scale)
        for row in d:
            finite = row[np.isfinite(row)]
            cut = np.partition(finite, min(k, len(finite)) - 1)[min(k, len(finite)) - 1]
            mask = row <= cut
            weights = np.exp(-0.5 * (row[mask] / max(cut, 1e-12)) ** 2)
            result.append(interval_mode(bg.time.to_numpy()[mask], weights))
    return np.asarray(result)


def atomic_bytes(path, data):
    """異なる旧出力を内容ハッシュで保存し、完成したデータだけ置き換える。"""
    if path.exists():
        old = path.read_bytes()
        if old == data:
            return
        archive = path.parent / "履歴" / path.name / (sha256(old) + path.suffix)
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists() and archive.read_bytes() != old:
            raise InputError("履歴ファイルの内容がハッシュと一致しません")
        if not archive.exists():
            archive.write_bytes(old)
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".tt-aia-", delete=False) as f:
            name = f.name
            f.write(data)
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def parse_targets(path):
    obj, digest = read_json(path)
    if not isinstance(obj, list) or not obj:
        raise InputError("targets.jsonは空でないJSON配列にしてください")
    records, seen = [], set()
    for i, item in enumerate(obj):
        if not isinstance(item, dict) or set(item) != {"target_k", "c_path", "aia_challenge_path"}:
            raise InputError(f"targets[{i}]: target_k, c_path, aia_challenge_pathの3項目だけが必要です")
        k = identifier(item["target_k"], f"targets[{i}].target_k")
        if k in seen:
            raise InputError(f"targets.json: target_k={k}が重複しています")
        seen.add(k)
        files = {}
        for field in ("c_path", "aia_challenge_path"):
            value = item[field]
            if not isinstance(value, str) or not value.strip():
                raise InputError(f"targets[{i}].{field}: パス文字列が必要です")
            p = Path(value).expanduser()
            files[field] = (p if p.is_absolute() else path.parent / p).resolve()
            if not files[field].is_file():
                raise InputError(f"targets[{i}].{field}: ファイルがありません: {files[field]}")
        records.append({"target_k": k, **files})
    return sorted(records, key=lambda x: x["target_k"]), digest


def check_official(args, targets):
    if args.target_set != "official":
        if args.mia_columns or args.self_team_id is not None:
            raise InputError("公式対象の照合には --target-set official を指定してください")
        return None
    if not args.mia_columns or args.self_team_id is None:
        raise InputError("公式実行には --mia-columns と --self-team-id が必要です")
    obj, digest = read_json(args.mia_columns.resolve())
    if not isinstance(obj, dict) or not isinstance(obj.get("target_columns"), list):
        raise InputError("mia_columns.json: target_columns配列が必要です")
    ids = [identifier(x, "mia_columns.target_columns") for x in obj["target_columns"]]
    if len(ids) != len(set(ids)):
        raise InputError("mia_columns.jsonの対象IDが重複しています")
    self_id = identifier(args.self_team_id, "self_team_id")
    expected = set(ids) - {self_id}
    got = {x["target_k"] for x in targets}
    # 公式版は自チームを出力しない。ローカルの自己検証はlocalで別実行する。
    if got != expected:
        raise InputError(f"公式対象集合が不一致。欠落={sorted(expected-got)}, 余分={sorted(got-expected)}")
    return {"sha256": digest, "self_team_id": self_id, "target_ids": sorted(expected)}


def main():
    parser = argparse.ArgumentParser(description=f"{NAME} v{VERSION}: AIA専用・正解ファイル不要")
    parser.add_argument("--b", required=True, type=Path, help="共通引数。存在だけ確認し、内容は読みません")
    parser.add_argument("--a-bg", required=True, type=Path, help="許可された背景データ")
    parser.add_argument("--targets", required=True, type=Path, help="対象一覧JSON。相対パスの基準はこのファイルの親")
    parser.add_argument("--target-set", choices=["local", "official"], default="local")
    parser.add_argument("--mia-columns", type=Path, help="officialで必須。公式の対象一覧")
    parser.add_argument("--self-team-id", type=int, help="officialで必須。攻撃者のチームID")
    args = parser.parse_args()
    started = time.perf_counter()
    script = Path(__file__).resolve()
    if not args.b.is_file():
        raise InputError("--bのファイルがありません（内容は使いません）")
    targets, target_hash = parse_targets(args.targets.resolve())
    official = check_official(args, targets)
    output = script.parent / f"F_aia_{NAME}.csv"
    record = script.parent / f"実行記録_{NAME}.json"
    inputs = [args.b, args.a_bg, args.targets] + ([args.mia_columns] if args.mia_columns else [])
    inputs += [t[p] for t in targets for p in ("c_path", "aia_challenge_path")]
    if any(p.resolve() in (output.resolve(), record.resolve()) for p in inputs):
        raise InputError("出力先と入力ファイルが同じです。入力は変更できません")
    bg, bg_hash = read_table(args.a_bg.resolve(), "background")
    scale = scale_from_background(bg)
    all_answers, reports = [], []
    for t in targets:
        c, c_hash = read_table(t["c_path"], "c")
        q, q_hash = read_table(t["aia_challenge_path"], "challenge")
        pred, detail = predict(c, q, bg, scale)
        pred = np.asarray(pred, dtype=float)
        if pred.shape != (len(q),) or not np.isfinite(pred).all() or ((pred <= 0) | (pred > HORIZON)).any():
            raise InputError(f"target_k={t['target_k']}: 予測の長さまたは値域が不正です")
        for rid, value in zip(q.challenge_row_id, pred):
            all_answers.append((t["target_k"], int(rid), float(value)))
        reports.append({"target_k": t["target_k"], "c_path": os.path.relpath(t["c_path"], script.parent),
                        "challenge_path": os.path.relpath(t["aia_challenge_path"], script.parent),
                        "c_sha256": c_hash, "challenge_sha256": q_hash,
                        "c_rows": len(c), "challenge_rows": len(q), "diagnostics": detail})
        print(f"target_k={t['target_k']}: {len(q)}行を推定", flush=True)
    keys = [(k, rid) for k, rid, _ in all_answers]
    if len(keys) != len(set(keys)) or keys != sorted(keys):
        raise InputError("出力識別子が重複、または順序が不正です")
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["target_k", "challenge_row_id", "time_hat"])
    writer.writerows((k, rid, format(v, FLOAT_FORMAT)) for k, rid, v in all_answers)
    output_bytes = buf.getvalue().encode("utf-8")
    # timeは元入力が極端に0へ近くても、10桁への丸めで0にしない。
    emitted = [float(row[2]) for row in list(csv.reader(io.StringIO(output_bytes.decode())))[1:]]
    if any(not (0 < v <= HORIZON) for v in emitted):
        raise InputError("数値表記への丸めでtime_hatの値域を外れました")
    atomic_bytes(output, output_bytes)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mib = peak / (1024**2 if sys.platform == "darwin" else 1024)
    manifest = {"name": NAME, "version": VERSION, "rulebook": RULEBOOK_VERSION,
                "code_sha256": sha256(script.read_bytes()), "seed": SEED, "settings": SETTINGS,
                "target_set": args.target_set, "official_check": official,
                "b_content_read": False, "a_bg_sha256": bg_hash, "targets_sha256": target_hash,
                "a_bg_path": os.path.relpath(args.a_bg.resolve(), script.parent),
                "targets_path": os.path.relpath(args.targets.resolve(), script.parent),
                "output_sha256": sha256(output_bytes), "output_rows": len(all_answers),
                "targets": reports, "scoring_performed": False,
                "environment": {"python": platform.python_version(), "platform": platform.platform(),
                                "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                                **extra_versions()},
                "elapsed_seconds": time.perf_counter() - started, "peak_rss_mib": peak_mib,
                "note": "時間・メモリは実測診断。再現性の比較対象はCSVバイト列。正解・採点結果は未使用。"}
    atomic_bytes(record, (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())
    print(f"保存: {output.name} ({len(all_answers)}行, sha256={sha256(output_bytes)})")


def extra_versions():
    return {}

def extra_versions():
    return {"scikit-learn": sklearn.__version__}


def features(df, scale):
    continuous = df[CONT].to_numpy(dtype=float) / scale
    pref = df.prefecture.to_numpy()
    encoded = np.column_stack([(pref == p).astype(float) for p in PREFECTURES])
    return np.column_stack([continuous, df.sex.to_numpy(dtype=float), encoded])


def predict(c, q, bg, scale):
    x, query = features(c, scale), features(q, scale)
    target = c.time.to_numpy(dtype=float)
    model = ExtraTreesRegressor(n_estimators=SETTINGS["trees"],
                               min_samples_leaf=SETTINGS["min_samples_leaf"],
                               max_depth=SETTINGS["max_depth"], max_features=SETTINGS["max_features"],
                               bootstrap=SETTINGS["bootstrap"], criterion=SETTINGS["criterion"],
                               random_state=SEED, n_jobs=SETTINGS["n_jobs"])
    model.fit(x, target)
    train_leaves, query_leaves = model.apply(x), model.apply(query)
    weights = np.zeros((len(q), len(c)), dtype=float)
    for tree in range(SETTINGS["trees"]):
        for leaf in np.unique(query_leaves[:, tree]):
            members = np.flatnonzero(train_leaves[:, tree] == leaf)
            queries = np.flatnonzero(query_leaves[:, tree] == leaf)
            weights[np.ix_(queries, members)] += 1.0 / (SETTINGS["trees"] * len(members))
    pred = np.asarray([interval_mode(target, row) for row in weights])
    return pred, {"trained_on": "C only", "features": int(x.shape[1]),
                  "trees": SETTINGS["trees"], "calibrated_probability": False}


if __name__ == "__main__":
    try:
        main()
    except (InputError, OSError, ValueError) as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        sys.exit(2)
