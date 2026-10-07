"""予備戦AIA課題の公開member/controlを使う、所属識別の補助検証。

MIA全体の公式課題は共有ブロック、本分析の標的は非共有ブロックであり別の評価。
個人別予測・回答CSVは保存しない。公開正解は予測確定後だけに読む。
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import 分析 as common


def values(a, q, c):
    result = {f"global7_k{k}": v for k, v in common.core(a, q, c, common.NUM7).items()}
    by_sex = {k: np.empty(len(q)) for k in (1, 3, 10)}
    for sex in sorted(q.sex.unique()):
        mask = q.sex.to_numpy() == sex
        group = common.core(a[a.sex == sex], q.loc[mask], c[c.sex == sex], common.NUM7)
        for k, value in group.items():
            by_sex[k][mask] = value
    result.update({f"sex7_k{k}": v for k, v in by_sex.items()})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir is not None and args.output_dir.exists():
        raise FileExistsError("既存の結果を上書きしません。")
    started = time.perf_counter()
    started_jst = datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()
    _, a, _, _, _, cs, targets, _ = common.phase_inputs(args.root, "prelim")
    teams = list(cs)
    predictions, queries = {}, {}
    for j, team in enumerate(teams):
        q = common.read(targets / f"aia_challenge_{team}.csv")
        queries[team] = q
        computed = [values(a, q, c) for c in cs.values()]
        methods = {}
        for name in computed[0]:
            rank = np.column_stack([rankdata(v[name], method="average") / len(q) for v in computed])
            methods[name] = rank[:, j]
            if name.endswith("k10"):
                others = np.median(np.delete(rank, j, axis=1), axis=1)
                for alpha in (.1, .25):
                    methods[name + f"_cross_{alpha}"] = rank[:, j] - alpha * others
        predictions[team] = methods
        print(json.dumps({"predictions_completed": len(predictions), "targets": len(teams)}, ensure_ascii=False), flush=True)
    # この時点で全対象・全方式の予測が確定済み。
    truthdir = common.child(common.child(args.root / "予備戦", "配布データ") / "結果", "正解データ")
    rows = []
    for team in teams:
        q = queries[team]
        labels = common.read(truthdir / f"aia_truth_{team}.csv").set_index("challenge_row_id").loc[q.challenge_row_id]
        member = (labels.label == "member").to_numpy(int)
        rare = labels.is_rare.to_numpy(bool)
        assert int(member.sum()) == 163 and int((1 - member).sum()) == 163
        assert int(rare.sum()) == 26 and int(rare[member == 1].sum()) == 13
        left = q.loc[member == 1].groupby(["sex", "prefecture"]).size()
        right = q.loc[member == 0].groupby(["sex", "prefecture"]).size()
        assert left.equals(right), "地域・性別を揃えた対照という条件を確認"
        for method, score in predictions[team].items():
            for subset, keep in (("all326", np.ones(len(q), dtype=bool)), ("nonrare300", ~rare)):
                tpr, auc = common.roc(score[keep], member[keep])
                rows.append({"team": team, "method": method, "subset": subset,
                             "members": int(member[keep].sum()), "controls": int((1 - member[keep]).sum()),
                             "proxy_tpr_1pct": tpr, "proxy_auc": auc})
    table = pd.DataFrame(rows)
    aggregate = []
    for (subset, method), frame in table.groupby(["subset", "method"], sort=True):
        baseline = table[(table.subset == subset) & (table.method == "global7_k10")].set_index("team").proxy_tpr_1pct
        delta = frame.set_index("team").proxy_tpr_1pct - baseline
        aggregate.append({"subset": subset, "method": method, "targets": len(frame),
                          "proxy_tpr_1pct": float(frame.proxy_tpr_1pct.mean()),
                          "other18_mean": float(frame[~frame.team.isin([1, 2])].proxy_tpr_1pct.mean()),
                          "proxy_auc": float(frame.proxy_auc.mean()), "wins": int((delta > 1e-12).sum()),
                          "losses": int((delta < -1e-12).sum())})
    record = {"started_jst": started_jst, "seconds": time.perf_counter() - started,
              "code_sha256": common.sha(Path(__file__)), "common_code_sha256": common.sha(Path(common.__file__)),
              "input_sha256": common.INPUT_HASHES,
              "scope_note": "AIA標的の非共有者と対照での補助評価。共有者を当てるMIA全体の公式得点とは別。",
              "writes_submission_files": False}
    for path, expected in common.INPUT_HASHES.items():
        assert common.sha(Path(path)) == expected
    result = {"aggregate": aggregate, "by_target": rows, "record": record}
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True)
        (args.output_dir / "補助検証.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        table.to_csv(args.output_dir / "対照一致_対象別比較.csv", index=False, float_format="%.17g")
        pd.DataFrame(aggregate).to_csv(args.output_dir / "対照一致_集約比較.csv", index=False, float_format="%.17g")
    print(json.dumps({"aggregate": aggregate, "seconds": record["seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
