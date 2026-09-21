"""seo-03b: C行の再利用を少数だけ許す容量付き全体割当AIA。

厳密な一対一対応が不適切な加工（複製、クラスタ代表、再標本化）を想定した
実験用変種。既定capacity=2。正解、過去回答、対象別例外は使用しない。
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
SLOT_PENALTY = 0.18


def load_core():
    path = Path(__file__).with_name(BASE_NAME)
    if not path.is_file():
        raise FileNotFoundError(f"required base script not found: {path}")
    spec = importlib.util.spec_from_file_location("seo03_core_aia", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CORE = load_core()


def capacity_assign(cost, capacity):
    if capacity < 1:
        raise ValueError("capacity must be at least 1")
    rows, columns = cost.shape
    if rows > capacity * columns:
        raise ValueError("challenge rows exceed C rows * capacity; increase --capacity explicitly")
    layers = capacity
    slots = np.hstack([cost + SLOT_PENALTY * (layer ** 2)
                       for layer in range(layers)])
    assigned_slot = CORE.global_assign(slots)
    return assigned_slot % columns


def relaxed_times(challenge, c, transform, capacity):
    cost = CORE.assignment_cost(challenge, c, transform)
    assigned = capacity_assign(cost, capacity)
    c_times = c.time.to_numpy(float)
    answers = np.empty(len(challenge), float)
    support = min(CORE.LOCAL_SUPPORT, len(c))
    for row, donor in enumerate(assigned):
        order = np.argsort(cost[row], kind="stable")[:support]
        if donor not in order:
            order[-1] = donor
        local_cost = cost[row, order]
        positive = local_cost[local_cost > local_cost.min() + CORE.EPS]
        bandwidth = float(np.median(positive - local_cost.min())) if len(positive) else 1.0
        bandwidth = max(bandwidth, 1e-6)
        weights = np.exp(-(local_cost - local_cost.min()) / bandwidth)
        donor_position = int(np.flatnonzero(order == donor)[0])
        weights[donor_position] *= CORE.ASSIGNED_BOOST
        weights /= weights.sum()
        answers[row] = CORE.best_time_window(c_times[order], weights, c_times[donor])
    return answers, assigned


def build_aia(a, target_tables, capacity):
    transform = CORE.BackgroundTransform(a)
    answers, diagnostics = [], []
    for target, c, challenge in target_tables:
        times, assigned = relaxed_times(challenge, c, transform, capacity)
        counts = np.bincount(assigned, minlength=len(c))
        answers.append(pd.DataFrame({"target_k": target,
                                     "challenge_row_id": challenge.challenge_row_id.to_numpy(np.int64),
                                     "time_hat": times}))
        diagnostics.append({"target_k": target, "challenge_rows": len(challenge),
                            "used_c_rows": int((counts > 0).sum()),
                            "maximum_reuse": int(counts.max())})
    output = pd.concat(answers, ignore_index=True).sort_values(
        ["target_k", "challenge_row_id"], kind="stable").reset_index(drop=True)
    if output.duplicated(["target_k", "challenge_row_id"]).any():
        raise ValueError("duplicate AIA output key")
    return output, diagnostics


def save_output(data, input_paths, overwrite=False):
    destination = Path(__file__).with_name("F_aia_" + Path(__file__).stem + ".csv")
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
    rng = np.random.default_rng(53032)

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

    a, c = fake(80), fake(30)
    challenge = fake(50)[CORE.QI]
    challenge.insert(0, "challenge_row_id", np.arange(50))
    transform = CORE.BackgroundTransform(a)
    first, assigned1 = relaxed_times(challenge, c, transform, 2)
    second, assigned2 = relaxed_times(challenge, c, transform, 2)
    assert np.array_equal(first, second) and np.array_equal(assigned1, assigned2)
    assert np.bincount(assigned1, minlength=len(c)).max() <= 2
    assert np.isfinite(first).all() and ((first >= 0) & (first <= CORE.HORIZON)).all()
    print(json.dumps({"self_test": "passed", "real_data_used": False,
                      "capacity": 2, "maximum_reuse": int(np.bincount(assigned1).max()),
                      "reproducible": True, "performance_evaluated": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b", type=Path)
    parser.add_argument("--a-bg", type=Path)
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--capacity", type=int, default=2)
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
        CORE.read_table(args.b, "B")  # 標準CLIの入力検査のみ。AIA値には使わない。
        a = CORE.read_table(args.a_bg, "A")
        tables = [(target, CORE.read_table(c_path, "C"),
                   CORE.read_table(q_path, "challenge"))
                  for target, c_path, q_path in targets]
        output, diagnostics = build_aia(a, tables, args.capacity)
        inputs = {path.resolve() for path in [args.b, args.a_bg, args.targets]}
        inputs.update(path.resolve() for _, c_path, q_path in targets for path in [c_path, q_path])
        if args.mia_columns is not None:
            inputs.add(args.mia_columns.resolve())
        destination, digest = save_output(output, inputs, args.overwrite)
        print(json.dumps({"name": "seo", "variant": "03b", "rows": len(output),
                          "capacity": args.capacity, "output": destination.name,
                          "sha256": digest, "diagnostics": diagnostics,
                          "seconds": round(time.perf_counter() - started, 3),
                          "official_scores": "not calculated"}, ensure_ascii=True))
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=__import__("sys").stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
