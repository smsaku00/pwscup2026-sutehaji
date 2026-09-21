"""seo-03a: 複数の共分散縮小率を合成する実験用MIA。

同じフォルダの seo-03-正則化生成比と全体割当.py を共通の入出力・検査基盤として使う。
正解、過去回答、対象別例外は使用しない。出力値は校正確率ではなく順位スコア。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd

BASE_NAME = "seo-03-正則化生成比と全体割当.py"
SHRINKAGES = (0.15, 0.35, 0.65, 0.90)


def load_core():
    path = Path(__file__).with_name(BASE_NAME)
    if not path.is_file():
        raise FileNotFoundError(f"required base script not found: {path}")
    spec = importlib.util.spec_from_file_location("seo03_core", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CORE = load_core()


def gaussian_logpdf(train, query, shrinkage):
    train, query = np.asarray(train, float), np.asarray(query, float)
    mean = train.mean(axis=0)
    empirical = np.atleast_2d(np.cov(train, rowvar=False, ddof=1))
    diagonal = np.diag(np.maximum(np.diag(empirical), 1e-6))
    covariance = (1.0 - shrinkage) * empirical + shrinkage * diagonal
    covariance += np.eye(covariance.shape[0]) * 1e-6
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ValueError("regularized covariance is not positive definite")
    precision = np.linalg.inv(covariance)
    delta = query - mean
    quadratic = np.einsum("ij,jk,ik->i", delta, precision, delta)
    return -0.5 * (quadratic + logdet + query.shape[1] * np.log(2.0 * np.pi))


def ensemble_scores(b, a, c, transform):
    groups = [
        CORE.CONT,
        ["age", "BMI", "SBP", "TG", "HDL", "FPG"],
        ["age", "SBP", "TG", "HDL", "ALT", "FPG", "time"],
        CORE.CONT + ["time"],
    ]
    components = []
    for columns in groups:
        xa = transform.normal_scores(a, columns)
        xc = transform.normal_scores(c, columns)
        xb = transform.normal_scores(b, columns)
        for shrinkage in SHRINKAGES:
            ratio = (gaussian_logpdf(xc, xb, shrinkage) -
                     gaussian_logpdf(xa, xb, shrinkage))
            components.append(CORE.percentile(ratio))
    category = CORE.categorical_log_ratio(b, c, a, CORE.CAT + CORE.EVENT_CAT)
    components.append(CORE.percentile(category))
    # 17モデルの中央値。単一の縮小率・単一の属性群への依存を抑える。
    return CORE.percentile(np.median(np.column_stack(components), axis=1))


def build_mia(b, a, target_tables):
    if len(a) < 16 or len(b) < 2:
        raise ValueError("A_bg needs at least 16 rows and B needs at least 2 rows")
    transform = CORE.BackgroundTransform(a)
    result = pd.DataFrame({"record_id": b.record_id.to_numpy(np.int64)})
    for target, c in target_tables:
        if len(c) < 16:
            raise ValueError(f"C for target {target} needs at least 16 rows")
        result[str(target)] = ensemble_scores(b, a, c, transform)
    return result


def save_output(data, input_paths, overwrite=False):
    destination = Path(__file__).with_name("F_mia_" + Path(__file__).stem + ".csv")
    blob = CORE.csv_bytes(data)
    if destination.resolve() in input_paths:
        raise ValueError("output path collides with an input")
    if destination.exists() and destination.read_bytes() != blob and not overwrite:
        raise ValueError(f"{destination.name} exists with different content; use --overwrite")
    if not destination.exists() or destination.read_bytes() != blob:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(blob)
        try:
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination, hashlib.sha256(blob).hexdigest()


def self_test():
    rng = np.random.default_rng(53031)

    def fake(n):
        frame = pd.DataFrame({col: rng.uniform(lo + .2*(hi-lo), lo + .7*(hi-lo), n)
                              for col, (lo, hi) in CORE.RANGES.items()})
        frame["age"] = rng.integers(40, 75, n)
        frame["sex"] = rng.integers(0, 2, n)
        frame["smoking"] = rng.integers(0, 2, n)
        frame["prefecture"] = rng.choice(["架空A", "架空B", "架空C"], n)
        frame["onset"] = rng.integers(0, 2, n)
        frame["death"] = 0
        frame["time"] = np.where(frame.onset == 0, 10.0, rng.uniform(.1, 9.7, n))
        return frame[CORE.COLS]

    a, c, b = fake(80), fake(64), fake(42)
    b.insert(0, "record_id", np.arange(42))
    first = build_mia(b, a, [(7, c)])
    second = build_mia(b, a, [(7, c)])
    assert CORE.csv_bytes(first) == CORE.csv_bytes(second)
    values = first["7"].to_numpy(float)
    assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
    print(json.dumps({"self_test": "passed", "real_data_used": False,
                      "models": 17, "reproducible": True,
                      "performance_evaluated": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b", type=Path)
    parser.add_argument("--a-bg", type=Path)
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--scope", choices=["local", "official"], default="local")
    parser.add_argument("--mia-columns", type=Path)
    parser.add_argument("--self-team-id", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    try:
        if args.self_test:
            self_test()
            return 0
        if any(value is None for value in [args.b, args.a_bg, args.targets]):
            raise ValueError("--b, --a-bg and --targets are required")
        targets = CORE.load_targets(args.targets.resolve())
        CORE.check_scope(targets, args.scope, args.mia_columns, args.self_team_id)
        b, a = CORE.read_table(args.b, "B"), CORE.read_table(args.a_bg, "A")
        tables = [(target, CORE.read_table(c_path, "C")) for target, c_path, _ in targets]
        output = build_mia(b, a, tables)
        inputs = {path.resolve() for path in [args.b, args.a_bg, args.targets]}
        inputs.update(c_path.resolve() for _, c_path, _ in targets)
        if args.mia_columns is not None:
            inputs.add(args.mia_columns.resolve())
        destination, digest = save_output(output, inputs, args.overwrite)
        print(json.dumps({"name": "seo", "variant": "03a", "rows": len(output),
                          "targets": [target for target, _, _ in targets],
                          "score_kind": "ensemble rank score, not probability",
                          "output": destination.name, "sha256": digest,
                          "seconds": round(time.perf_counter() - started, 3),
                          "official_scores": "not calculated"}, ensure_ascii=True))
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=__import__("sys").stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
