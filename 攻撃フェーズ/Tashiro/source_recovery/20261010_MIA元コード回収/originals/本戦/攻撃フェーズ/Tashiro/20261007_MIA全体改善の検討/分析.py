"""MIA全体の改善案を検討するための再集計。攻撃回答は作成しない。

ルート .venv/bin/python -B で実行する。既定では標準出力のみ。
--output-dir を明示した場合だけ、新しいディレクトリへ集計を保存する。
本戦の所属正解は使用しない。予備戦の公開正解は予測確定後の採点だけに読む。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from scipy.stats import rankdata
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_curve

NUM7 = ("age", "SBP", "TG", "HDL", "ALT")
NUM9 = ("age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG")
QI7 = NUM7 + ("sex", "prefecture")
RHO = .98
INPUT_HASHES: dict[str, str] = {}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def child(parent: Path, name: str) -> Path:
    """macOS の結合文字と GitHub の合成済み文字の双方で同じ入力を探す。"""
    found = [p for p in parent.iterdir()
             if unicodedata.normalize("NFC", p.name) == unicodedata.normalize("NFC", name)]
    if len(found) != 1:
        raise FileNotFoundError(f"入力が一意に見つかりません: {parent / name}")
    return found[0]


def read(path: Path) -> pd.DataFrame:
    INPUT_HASHES[str(path.resolve())] = sha(path)
    return pd.read_csv(path, float_precision="round_trip")


def read_json(path: Path):
    INPUT_HASHES[str(path.resolve())] = sha(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def phase_inputs(root: Path, phase: str):
    if phase == "main":
        if (root / "本戦").exists():
            main = root / "本戦"
            distributed = child(main, "配布データ")
            own = next(distributed.glob("PWSCup2026_final_team28_*"))
            targets = main / "攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力"
        else:
            distributed = root / "公式データ"
            own = next(distributed.glob("PWSCup2026_final_team28_*"))
            targets = distributed / "PWSCup2026_final_attack"
        background = own / "A_bg.csv"
        self_id = 28
    else:
        distributed = child(root / "予備戦", "配布データ")
        own = next(distributed.glob("PWSCup2026_prelim_team05_*"))
        targets = distributed / "PWSCup2026_prelim_attack"
        background = distributed / "PWSCup2026_prelim_common/A_bg.csv"
        self_id = 5
    b = read(own / "B_self.csv")
    a = read(background)
    q = read(targets / f"aia_challenge_{self_id}.csv")
    assert b.record_id.is_unique and not b.duplicated(list(QI7)).any()
    assert q.challenge_row_id.is_unique and not q.duplicated(list(QI7)).any()
    private = pd.MultiIndex.from_frame(b[list(QI7)]).isin(pd.MultiIndex.from_frame(q[list(QI7)]))
    assert int(private.sum()) == 163
    rare = b.is_rare.to_numpy(bool)
    assert int(rare.sum()) == 24 and int((private & rare).sum()) == 13
    cs = {int(p.stem.split("_")[-1]): read(p) for p in sorted(targets.glob("C_*.csv"))}
    cs = {k: v for k, v in cs.items() if k != self_id}
    assert len(cs) == (22 if phase == "main" else 20)
    return b, a, q, private, rare, cs, targets, self_id


def numbers(frame: pd.DataFrame, columns) -> np.ndarray:
    result = frame[list(columns)].to_numpy(float).copy()
    for col in ("TG", "ALT"):
        if col in columns:
            result[:, list(columns).index(col)] = np.log1p(result[:, list(columns).index(col)])
    assert np.isfinite(result).all()
    return result


def scales(reference: pd.DataFrame, columns) -> np.ndarray:
    x = numbers(reference, columns)
    scale = (np.quantile(x, .75, axis=0) - np.quantile(x, .25, axis=0)) / 1.349
    assert (scale > 0).all()
    return scale


def core(reference, candidates, synthetic, columns, ks=(1, 3, 10), whiten=False):
    """既存の混合カーネル Gen-LRA と同じ追加前後の式。返すのは生スコア。"""
    scale = scales(reference, columns)
    r, b, c = [numbers(f, columns) / scale for f in (reference, candidates, synthetic)]
    if whiten:
        covariance = LedoitWolf().fit(r).covariance_
        transform = np.linalg.inv(np.linalg.cholesky(covariance)).T
        r, b, c = [x @ transform for x in (r, b, c)]
    h = (4. / (len(columns) + 2) / len(r)) ** (1. / (len(columns) + 4))
    cr = cdist(c, r, metric="sqeuclidean")
    cb = cdist(c, b, metric="sqeuclidean")
    distance = cb.T.copy()
    log_cr = -.5 * cr / h**2
    log_cb = -.5 * cb / h**2
    for col in ("sex", "prefecture"):
        categories = sorted(set(reference[col]) | set(candidates[col]) | set(synthetic[col]))
        freq = (reference[col].value_counts().reindex(categories, fill_value=0) + 1) / (len(reference) + len(categories))
        cf = synthetic[col].map(freq).to_numpy()[:, None]
        eq_cr = synthetic[col].to_numpy()[:, None] == reference[col].to_numpy()[None, :]
        eq_cb = synthetic[col].to_numpy()[:, None] == candidates[col].to_numpy()[None, :]
        log_cr += np.log(RHO * eq_cr + (1 - RHO) * cf)
        log_cb += np.log(RHO * eq_cb + (1 - RHO) * cf)
        distance += ~eq_cb.T
    # カーネルの共通正規化定数は分子・分母で相殺される。
    base = logsumexp(log_cr, axis=1)
    delta = (np.logaddexp(0., log_cb - base[:, None]) - np.log1p(1 / len(r))).T
    near = np.argsort(distance, axis=1, kind="stable")[:, :min(max(ks), len(c))]
    ordered = np.take_along_axis(delta, near, axis=1)
    output = {k: ordered[:, :min(k, len(c))].sum(axis=1) for k in ks}
    assert all(np.isfinite(x).all() for x in output.values())
    return output


def geometry_scores(a, b, c):
    result = {"global7": core(a, b, c, NUM7)[10]}
    has9 = all(set(NUM9).issubset(frame.columns) for frame in (a, b, c))
    if has9:
        result["global9"] = core(a, b, c, NUM9)[10]
    for label, cols, whitening, ks in [
        ("sex7", NUM7, False, (1, 3, 10)),
        ("sex9", NUM9, False, (10,)),
        ("sex_whiten7", NUM7, True, (10,)),
    ]:
        if label == "sex9" and not has9:
            continue
        values = {k: np.empty(len(b)) for k in ks}
        for sex in sorted(b.sex.unique()):
            mask = b.sex.to_numpy() == sex
            aa, bb, cc = a[a.sex == sex], b.loc[mask], c[c.sex == sex]
            assert len(aa) > 10 and len(cc) > 10
            group = core(aa, bb, cc, cols, ks, whitening)
            for k in ks:
                values[k][mask] = group[k]
        for k, value in values.items():
            result[label if k == 10 else f"{label}_k{k}"] = value
    return result


def roc(confidence, truth):
    truth = np.asarray(truth, dtype=int)
    n_pos = int(truth.sum())
    n_neg = len(truth) - n_pos
    assert n_pos > 0 and n_neg > 0
    order = np.argsort(-confidence, kind="stable")
    confidence, y = np.asarray(confidence)[order], truth[order]
    ends = np.r_[np.flatnonzero(np.diff(confidence) != 0), len(y) - 1]
    tp = np.r_[0., np.cumsum(y)[ends] / n_pos]
    fp = np.r_[0., np.cumsum(1 - y)[ends] / n_neg]
    fpr, tpr, _ = roc_curve(truth, np.asarray(confidence)[np.argsort(order)], drop_intermediate=False)
    answer = float(np.interp(.01, fp, tp))
    assert abs(answer - float(np.interp(.01, fpr, tpr))) < 1e-12
    return answer, float(np.trapezoid(tp, fp))


def confidence(values, normal, rare, private):
    result = np.zeros(len(normal))
    result[normal] = .6 + .3 * rankdata(values, method="min") / normal.sum()
    result[rare & ~private] = .4
    return result


def evaluate_prelim(root, b, private, rare, matrices, teams):
    """呼び出し前に全方式・全対象の予測を確定する。正解はこの関数だけで読む。"""
    truthdir = child(child(root / "予備戦", "配布データ") / "結果", "正解データ")
    own = read(truthdir / "pool_ids_5.csv").set_index("record_id").loc[b.record_id]
    normal = ~private & ~rare
    rows = []
    for j, team in enumerate(teams):
        pool = read(truthdir / f"pool_ids_{team}.csv")
        truth = own.pool_id.isin(pool.pool_id).to_numpy(int)
        assert int(truth[private].sum()) == 0
        for name, matrix in matrices.items():
            tpr, auc = roc(confidence(matrix[:, j], normal, rare, private), truth)
            rows.append({"team": team, "method": name, "members": int(truth.sum()), "mia_all": tpr, "auc": auc})
    table = pd.DataFrame(rows)
    baseline = table[table.method == "global7"].set_index("team").mia_all
    summaries = []
    for name, group in table.groupby("method", sort=True):
        delta = group.set_index("team").mia_all - baseline
        summaries.append({"method": name, "targets": len(group), "mia_all": float(group.mia_all.mean()),
                          "other18_mean": float(group[~group.team.isin([1, 2])].mia_all.mean()),
                          "auc": float(group.auc.mean()), "wins": int((delta > 1e-12).sum()),
                          "losses": int((delta < -1e-12).sum())})
    return rows, summaries


def exact_diagnostics(b, a, private, cs):
    rows = []
    for team, c in sorted(cs.items()):
        count = np.zeros((len(b), len(c)), dtype=np.uint8)
        for col in ("BMI", "SBP", "TG", "HDL", "ALT", "FPG"):
            count += np.isclose(b[col].to_numpy()[:, None], c[col].to_numpy()[None, :], rtol=0, atol=1e-8)
        maximum = count.max(axis=1)
        record = {"team": team, "same_row_two_clinical": int((maximum >= 2).sum()),
                  "same_row_three_clinical": int((maximum >= 3).sum())}
        for label, x in (("B", b), ("A", a)):
            eq = np.isclose(x.time.to_numpy()[:, None], c.time.to_numpy()[None, :], rtol=0, atol=1e-8)
            eq &= x.time.to_numpy()[:, None] < 9.49
            record[label + "_early_time_match"] = int(eq.any(axis=1).sum())
            for col in ("onset", "death", "sex", "age", "prefecture"):
                eq &= x[col].to_numpy()[:, None] == c[col].to_numpy()[None, :]
            record[label + "_time_outcome_sex_age_pref_match"] = int(eq.any(axis=1).sum())
        rows.append(record)
    return rows


def candidate_contrast(raw, normal):
    ranks = np.column_stack([rankdata(col[normal], method="average") / normal.sum() for col in raw.T])
    med = np.median(ranks, axis=1)
    result = {"baseline": ranks}
    for alpha in (.25, .5, 1.):
        result[f"cross_{alpha}"] = ranks - alpha * med[:, None]
    result["cross_product"] = ranks * (1 - med[:, None])
    result["cross_second"] = ranks - .5 * np.partition(ranks, -2, axis=1)[:, -2, None]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--include-prelim", action="store_true", help="予備戦の公開正解による探索的比較も再計算する")
    parser.add_argument("--output-dir", type=Path, help="既存でない集計保存先。省略時はファイルを作らない")
    args = parser.parse_args()
    if args.output_dir is not None and args.output_dir.exists():
        raise FileExistsError("既存の結果を上書きしません。新しい --output-dir を指定してください。")
    started = time.perf_counter()
    started_jst = datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()
    b, a, q, private, rare, cs, targets, own = phase_inputs(args.root, "main")
    normal = ~private & ~rare
    summary = {"scope": "方法の検討と集計。本戦の未知の所属正解は使わず、F_mia・F_aia・ZIPを作成しない。",
               "goal_mia_all": .0647, "goal_source": "2026-10-07 ユーザー共有値。CodaBenchへアクセスしない。",
               "main_counts": {"B": len(b), "A": len(a), "targets": len(cs), "private": int(private.sum()),
                               "rare": int(rare.sum()), "possible_nonrare": int(normal.sum())}}
    summary["main_scales"] = [{"sex": sex, "n": len(frame),
                               "iqr_over_1_349": dict(zip(NUM7, scales(frame, NUM7).tolist()))}
                              for sex, frame in [("all", a)] + [(int(s), a[a.sex == s]) for s in sorted(a.sex.unique())]]
    summary["main_exact_diagnostics"] = exact_diagnostics(b, a, private, cs)
    raw = np.column_stack([core(a, b, c, NUM7, (10,))[10] for c in cs.values()])
    contrast = candidate_contrast(raw, normal)
    ranked = contrast["baseline"]
    repeated = np.sum(ranked >= (normal.sum() - 19) / normal.sum(), axis=1)
    summary["main_repeated_top20"] = {"unique_candidates": int((repeated > 0).sum()),
                                      "in_at_least_3_targets": int((repeated >= 3).sum()),
                                      "largest_target_count": int(repeated.max())}
    summary["main_rank_changes"] = [{"team": team,
                                    "top20_common_weak_cross": len(set(np.argsort(-ranked[:, j], kind="stable")[:20]) &
                                                                   set(np.argsort(-contrast["cross_0.25"][:, j], kind="stable")[:20]))}
                                   for j, team in enumerate(cs)]
    # 自チーム公開課題での補助診断。MIAの共有ブロックとは別の標本。
    cself = read(targets / f"C_{own}.csv")
    scores = geometry_scores(a, q, cself)
    member = pd.MultiIndex.from_frame(q[list(QI7)]).isin(pd.MultiIndex.from_frame(b[list(QI7)])).astype(int)
    assert int(member.sum()) == 163
    summary["main_self_challenge_proxy"] = [{"method": name, "tpr_1pct": roc(value, member)[0],
                                              "auc": roc(value, member)[1]} for name, value in scores.items()]
    if args.include_prelim:
        pb, pa, pq, pp, pr, pcs, _, _ = phase_inputs(args.root, "prelim")
        pn = ~pp & ~pr
        computed = {t: geometry_scores(pa, pb, c) for t, c in pcs.items()}
        matrices = {name: np.column_stack([rankdata(computed[t][name][pn], method="average") / pn.sum()
                                          for t in pcs]) for name in next(iter(computed.values()))}
        for name in ("global7", "global9", "sex7", "sex9", "sex_whiten7"):
            mat = matrices[name]
            matrices[name + "_cross025"] = mat - .25 * np.median(mat, axis=1)[:, None]
        for name, mat in candidate_contrast(np.column_stack([computed[t]["global7"] for t in pcs]), pn).items():
            matrices["sensitivity_" + name] = mat
        rows, aggregates = evaluate_prelim(args.root, pb, pp, pr, matrices, list(pcs))
        summary["prelim_by_target"] = rows
        summary["prelim_aggregate"] = aggregates
        if (args.root / "本戦").exists():
            saved = read(args.root / "本戦/攻撃フェーズ/Tashiro/20261005_MIA初回_GenLRA比較/結果/予備戦_対象別評価.csv")
            reference = saved[saved.variant == "A_mixed7_9"].set_index("team").mia_all
            actual = pd.DataFrame(rows).query("method == 'global7'").set_index("team").mia_all
            error = float(np.max(np.abs(reference - actual)))
            assert error < 1e-12
            summary["prelim_baseline_saved_tpr_max_error"] = error
    summary["record"] = {"started_jst": started_jst,
                         "seconds": time.perf_counter() - started, "python": platform.python_version(),
                         "executable": sys.executable, "code_sha256": sha(Path(__file__)),
                         "numpy": np.__version__, "pandas": pd.__version__, "input_sha256": INPUT_HASHES,
                         "writes_submission_files": False, "uploads_or_submits": False}
    for path, expected in INPUT_HASHES.items():
        assert sha(Path(path)) == expected, f"入力が計算中に変わりました: {path}"
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True)
        (args.output_dir / "再集計.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        if args.include_prelim:
            pd.DataFrame(summary["prelim_by_target"]).to_csv(args.output_dir / "予備戦_対象別比較.csv", index=False, float_format="%.17g")
            pd.DataFrame(summary["prelim_aggregate"]).to_csv(args.output_dir / "予備戦_集約比較.csv", index=False, float_format="%.17g")
        print(json.dumps({"output_dir": str(args.output_dir), "seconds": summary["record"]["seconds"],
                          "prelim_aggregate": summary.get("prelim_aggregate"),
                          "main_self_challenge_proxy": summary["main_self_challenge_proxy"]}, ensure_ascii=False))
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
