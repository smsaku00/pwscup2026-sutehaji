"""公開予備戦正解による公式採点の監査。人物行・正解ラベルは保存しない。"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import sys
import types
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_curve


def child(parent: Path, name: str) -> Path:
    matches = [p for p in parent.iterdir()
               if unicodedata.normalize("NFC", p.name) == unicodedata.normalize("NFC", name)]
    if len(matches) != 1:
        raise FileNotFoundError(f"入力が一意ではありません: {parent / name}")
    return matches[0]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _official_paths(root: Path) -> tuple[Path, Path, Path]:
    prelim = root / "スターターキット/改訂版_20260824/starter/src/pwscup2026"
    main = child(child(root / "本戦", "配布データ"), "pwscup2026-kit-main_revised0912") / "starter/src/pwscup2026"
    return prelim / "scoring/attack_score.py", main / "scoring/attack_score.py", prelim / "kit/codabench/scoring_io.py"


def load_official(root: str | Path):
    """公式モジュールを独立名前空間へ読み込む。戻り値の _tpr_at_fpr を使用。"""
    root = Path(root).resolve()
    prelim, main, scoring_io = _official_paths(root)
    key = "_mia_band_audit_" + sha(prelim)[:12]
    if key + ".attack_score" in sys.modules:
        return sys.modules[key + ".attack_score"]
    package = types.ModuleType(key)
    package.__path__ = [str(prelim.parent)]
    sys.modules[key] = package
    for name, path in [("result", prelim.parent / "result.py"), ("attack_score", prelim)]:
        spec = importlib.util.spec_from_file_location(key + "." + name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    official = sys.modules[key + ".attack_score"]
    # 帯判定そのものを公式ソースから読み、I/Oモジュール全体の副作用を避ける。
    tree = ast.parse(scoring_io.read_text(encoding="utf-8"))
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "MIA_BAND_EDGES" for t in node.targets):
            nodes.append(node)
        if isinstance(node, ast.FunctionDef) and node.name == "_mia_band":
            nodes.append(node)
    if len(nodes) != 2:
        raise AssertionError("公式の帯判定が見つかりません")
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(scoring_io), "exec"), namespace)
    official._mia_band = namespace["_mia_band"]
    official._source_paths = [prelim, main, scoring_io]
    official._source_sha256 = {str(p.relative_to(root)): sha(p) for p in official._source_paths}
    return official


def load_inputs(root: str | Path):
    """内部検算用。B行順、既知非会員/希少マスクをメモリにのみ返す。"""
    root = Path(root).resolve()
    distributed = child(root / "予備戦", "配布データ")
    own = next(distributed.glob("PWSCup2026_prelim_team05_*"))
    targets = distributed / "PWSCup2026_prelim_attack"
    b = pd.read_csv(own / "B_self.csv", float_precision="round_trip")
    q = pd.read_csv(targets / "aia_challenge_5.csv", float_precision="round_trip")
    qi = ["age", "SBP", "TG", "HDL", "ALT", "sex", "prefecture"]
    private = pd.MultiIndex.from_frame(b[qi]).isin(pd.MultiIndex.from_frame(q[qi]))
    rare = b.is_rare.to_numpy(bool)
    normal = ~private & ~rare
    assert len(b) == 1049 and int(private.sum()) == 163 and int(rare.sum()) == 24 and int(normal.sum()) == 875
    return b, private, rare, normal, targets


def load_truth(root: str | Path) -> dict[int, np.ndarray]:
    """Team5のB_self.record_id順の1049行。Team5を除く20対象、Team12を含む。"""
    root = Path(root).resolve()
    b, private, _, _, targets = load_inputs(root)
    distributed = child(root / "予備戦", "配布データ")
    truthdir = child(distributed / "結果", "正解データ")
    own = pd.read_csv(truthdir / "pool_ids_5.csv").set_index("record_id").loc[b.record_id]
    teams = sorted(int(p.stem.split("_")[-1]) for p in targets.glob("C_*.csv") if p.stem != "C_5")
    result = {}
    for team in teams:
        pool = pd.read_csv(truthdir / f"pool_ids_{team}.csv")
        truth = own.pool_id.isin(pool.pool_id).to_numpy(int)
        assert not truth[private].any()
        result[team] = truth
    assert len(result) == 20
    return result


def band(tpr: float) -> int:
    """内部生値で判定。公式比較関数と境界テストで照合する。"""
    if not np.isfinite(tpr) or not 0 <= float(tpr) <= 1:
        raise ValueError(f"TPRが範囲外です: {tpr}")
    return 1 + sum(float(tpr) > edge for edge in (.25, .5, .75))


def _scorer(module_name: str, path: Path):
    """旧実験のROC実装だけをソースから読み込む。スクリプトは実行しない。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "roc"]
    if len(nodes) != 1:
        raise AssertionError(f"ROC関数が見つかりません: {path}")
    namespace = {"np": np, "roc_curve": roc_curve}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["roc"]


def _synthetic_checks(root: Path, official) -> dict:
    values = [0., .25, np.nextafter(.25, 1.), .5, np.nextafter(.5, 1.), .75, np.nextafter(.75, 1.), 1.]
    bounds = []
    for value in values:
        actual = band(value)
        expected = official._mia_band(value)
        assert actual == expected
        bounds.append({"tpr": float(value), "band": actual})
    constant = official._tpr_at_fpr(np.ones(1049), np.r_[np.ones(25), np.zeros(1024)], .01)
    assert constant == (.01, .5)
    old_path = root / "本戦/攻撃フェーズ/Tashiro/20261007_MIA全体改善の検討/分析.py"
    old_roc = _scorer("old_roc", old_path)
    rng = np.random.default_rng(20261008)
    cases = 0
    max_sklearn = 0.
    max_old = 0.
    # 同点・同点なし・陽性の垂直ROC区間の有る入力を比較。
    for n in (2, 20, 1049):
        for rep in range(100):
            truth = rng.integers(0, 2, n)
            scores = rng.integers(0, 9, n) / 8 if rep % 2 == 0 else rng.random(n)
            if truth.min() == truth.max():
                continue
            fp, tp, _ = roc_curve(truth, scores, drop_intermediate=False)
            expected, _ = official._tpr_at_fpr(scores, truth, .01)
            max_sklearn = max(max_sklearn, abs(expected - float(np.interp(.01, fp, tp))))
            max_old = max(max_old, abs(expected - old_roc(scores, truth)[0]))
            cases += 1
    assert max_sklearn < 1e-12 and max_old < 1e-12
    # FPR=1%が二つのROC動作点の間にある例。10.24件目の非会員位置を線形補間。
    truth = np.r_[np.ones(5, dtype=int), np.zeros(10, dtype=int), [1, 0], np.ones(19, dtype=int), np.zeros(1013, dtype=int)]
    scores = np.arange(len(truth), 0, -1, dtype=float)
    interpolated = official._tpr_at_fpr(scores, truth, .01)[0]
    assert abs(interpolated - 6 / 25) < 1e-12
    # 複合同点の中で追加陽性/陰性を同時に補間し、整数でない検出相当数を確認。
    truth2 = np.r_[np.ones(5, dtype=int), np.zeros(10, dtype=int), [1, 0], np.ones(19, dtype=int), np.zeros(1013, dtype=int)]
    scores2 = scores.copy()
    scores2[15:17] = scores2[15]
    fractional = official._tpr_at_fpr(scores2, truth2, .01)[0]
    assert abs(fractional - (5 + .24) / 25) < 1e-12
    old_initial = root / "スターターキット/初版/starter/src/pwscup2026/scoring/attack_score.py"
    return dict(boundary_checks=bounds, constant_confidence_tpr=constant[0], constant_confidence_auc=constant[1],
                synthetic_cases=cases, sklearn_max_error=max_sklearn, oct07_roc_max_error=max_old,
                oct07_roc_source_sha256=sha(old_path), fractional_tpr=fractional, fractional_detected_equivalent=5.24,
                initial_kit_source_sha256=sha(old_initial), initial_kit_discrepancy="初版は同値を束ねず行順に依存するため採点参照に使用しない。",
                roc_semantics="確信度降順。完全に同じ浮動小数値を一つの動作点に束ねる。原点を追加しnp.interpでFPR=0.01に線形補間。陽性または陰性が0ならTPR=0,AUC=0.5。")


def _cache_checks(root: Path, official, truth: dict[int, np.ndarray]) -> dict:
    b, private, rare, normal, _ = load_inputs(root)
    normal_ids = b.loc[normal, "record_id"].to_numpy()
    here = root / "本戦/攻撃フェーズ/Tashiro/20261008_MIA全体_朝9時までの方式探索"
    files = []
    total = 0
    missing = 0
    max_error = 0.
    failures = []
    for path in sorted(here.rglob("内部順位キャッシュ.npz")):
        csv_path = path.parent / "対象別TPR.csv"
        if not csv_path.exists():
            continue
        saved = pd.read_csv(csv_path, float_precision="round_trip")
        saved = saved[np.isclose(saved.fpr, .01, rtol=0., atol=1e-15)]
        if saved.duplicated(["method", "team"]).any():
            raise AssertionError(f"同一条件の重複結果: {csv_path}")
        expected = {(str(r.method), int(r.team)): float(r.mia_all) for r in saved.itertuples()}
        checked = 0
        file_error = 0.
        with np.load(path, allow_pickle=False) as cache:
            names, teams, scores = cache["names"], cache["teams"], cache["scores"]
            assert scores.shape == (len(names), 875, len(teams)), path
            assert np.array_equal(cache["record_ids"], normal_ids), path
            conf = np.zeros(1049)
            conf[rare & ~private] = .4
            for model, name in enumerate(names):
                for col, team in enumerate(teams):
                    key = str(name), int(team)
                    if key not in expected:
                        missing += 1
                        continue
                    conf[normal] = .6 + .3 * rankdata(scores[model, :, col], method="min") / 875
                    actual = official._tpr_at_fpr(conf, truth[int(team)], .01)[0]
                    error = abs(actual - expected[key])
                    file_error = max(file_error, error)
                    checked += 1
                    if error >= 1e-12:
                        failures.append({"source": str(path.relative_to(root)), "method": str(name), "team": int(team), "saved_tpr": expected[key], "official_tpr": actual})
        files.append({"source": str(path.relative_to(root)), "source_sha256": sha(path),
                      "saved_source": str(csv_path.relative_to(root)), "saved_sha256": sha(csv_path),
                      "method_conditions": len(names), "compared_records": checked, "max_abs_error": file_error})
        max_error = max(max_error, file_error)
        total += checked
    return dict(cache_files=len(files), compared_records=total, missing_saved_records=missing,
                max_abs_error=max_error, mismatch_count=len(failures), mismatches=failures, files=files,
                interpretation="個別TPR CSVが保存されたキャッシュを独立再採点。CSV未保存の後半方式はoct08_results.pyが補充する。")


def audit(root: str | Path) -> dict:
    root = Path(root).resolve()
    official = load_official(root)
    prelim, main, scoring_io = _official_paths(root)
    assert prelim.read_bytes() == main.read_bytes()
    main_io = main.parent.parent / "kit/codabench/scoring_io.py"
    main_tree = ast.parse(main_io.read_text(encoding="utf-8"))
    main_nodes = [n for n in main_tree.body if (isinstance(n, ast.FunctionDef) and n.name == "_mia_band") or
                  (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "MIA_BAND_EDGES" for t in n.targets))]
    ns = {}
    exec(compile(ast.Module(body=main_nodes, type_ignores=[]), str(main_io), "exec"), ns)
    for v in (0., .25, np.nextafter(.25, 1.), .5, np.nextafter(.5, 1.), .75, np.nextafter(.75, 1.), 1.):
        assert ns["_mia_band"](v) == official._mia_band(v)
    truth = load_truth(root)
    b, private, rare, normal, _ = load_inputs(root)
    members = {str(k): int(y.sum()) for k, y in truth.items()}
    report = {"official_sources": official._source_sha256, "main_band_source": str(main_io.relative_to(root)),
              "main_band_source_sha256": sha(main_io), "prelim_main_score_sources_identical": True,
              "report_tpr_channel": "tpr_at_low_fpr（MIA全体）。max(全体,希少)のscoreでは帯を判定しない。",
              "fpr": .01, "evaluation_teams": sorted(truth), "final_report_teams": sorted(set(truth) - {12}),
              "candidate_count": len(b), "normal_candidates": int(normal.sum()), "rare_candidates": int(rare.sum()),
              "known_private_candidates": int(private.sum()), "member_count_by_team": members,
              "nonmember_count_by_team": {str(k): len(y) - int(y.sum()) for k, y in truth.items()},
              "fpr_01_false_positive_equivalent_range": [10.22, 10.24],
              "band2_detected_equivalent": "会員25人なら6.25人超、26人なら6.5人超、27人なら6.75人超。ROC補間値なので整数人数と同一視しない。",
              "corrected_threshold": "TPR生値>0.25のみ帯2以上。保存小数の丸め値による境界判定はしない。",
              "synthetic": _synthetic_checks(root, official), "cache_recalculation": _cache_checks(root, official, truth)}
    report["passed"] = report["cache_recalculation"]["mismatch_count"] == 0
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output_dir / "公式採点監査.json"
    if destination.exists():
        raise FileExistsError(destination)
    result = audit(args.root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"passed": result["passed"], "cache_files": result["cache_recalculation"]["cache_files"],
                      "compared_records": result["cache_recalculation"]["compared_records"],
                      "max_abs_error": result["cache_recalculation"]["max_abs_error"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
