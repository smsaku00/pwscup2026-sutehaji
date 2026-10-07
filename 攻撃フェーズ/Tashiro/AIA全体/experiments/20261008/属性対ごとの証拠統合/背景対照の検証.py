"""固定背景対照を再生成し、候補外の強制対応と保存的中人数を確認する。

方式を追加・調整せず、既存20反復×18対象×4方式の範囲を検証する。
"""
from __future__ import annotations
import hashlib, importlib.util, io, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("fixed_pair_evidence", HERE/"分析.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
ROOT = base.ROOT
HASHES = {}


def read(path):
    content = Path(path).read_bytes()
    HASHES[str(path)] = hashlib.sha256(content).hexdigest()
    return pd.read_csv(io.BytesIO(content), float_precision="round_trip")


def run():
    start = time.time()
    record = json.loads((HERE/"実行記録.json").read_text())
    assert hashlib.sha256((HERE/"分析.py").read_bytes()).hexdigest()==record["code_sha256"]
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest for path,digest in record["input_sha256"].items())
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest for path,digest in record["imported_code_sha256"].items())
    saved = read(HERE/"集計/独立背景_実Cと全反復.csv")
    saved = saved[saved.source.eq("BG_C")].set_index(["team","repeat","method"])
    folder = ROOT/"予備戦/配布データ/PWSCup2026_prelim_attack"
    bg = read(ROOT/"予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv")
    index = np.random.default_rng(base.SEED).permutation(len(bg))
    pool,ref = (bg.iloc[i].reset_index(drop=True) for i in [index[:len(bg)//2],index[len(bg)//2:]])
    assert len(pool)==2571 and len(ref)==2572
    assert set(index[:len(bg)//2]).isdisjoint(index[len(bg)//2:])
    samples = [pool.iloc[np.sort(np.random.default_rng(base.SEED+400+r).choice(len(pool),1049,False))].reset_index(drop=True) for r in range(base.REPEATS)]
    truth_folder = next((ROOT/"予備戦/配布データ/結果").glob("正解*"))
    output = []
    for team in sorted(saved.index.get_level_values("team").unique()):
        q = read(folder/f"aia_challenge_{team}.csv").sort_values("challenge_row_id").reset_index(drop=True)
        bginfo = base.background(q,ref)
        pending = []
        for repeat,sample in enumerate(samples):
            prediction,diagnostics = base.predict(q,sample,bginfo)
            pending.append((repeat,prediction,diagnostics))
        truth = read(truth_folder/f"aia_truth_{team}.csv").sort_values("challenge_row_id").reset_index(drop=True)
        for repeat,prediction,diagnostics in pending:
            for rec in diagnostics:
                method=rec["method"]
                scores=base.core.scores(prediction[method],truth)
                expected=saved.loc[(team,repeat,method)]
                errors=[abs(scores[key]-float(expected[key])) for key in ["r_all","r_rare","r_max","p_m","p_c"]]
                assert max(errors)<1e-12
                assert scores["hit_m"]==expected.hit_m and scores["hit_c"]==expected.hit_c
                output.append(dict(team=int(team),repeat=repeat,method=method,score_max_difference=float(max(errors)),
                                   hit_counts_match=True,**{key:rec[key] for key in ["selected_bad_pairs","selected","candidate_count_median","candidate_count_max"]}))
        print(json.dumps(dict(team=int(team),seconds=round(time.time()-start,1))),flush=True)
    table=pd.DataFrame(output)
    table.to_csv(HERE/"集計/背景対照_再生成と候補制限監査.csv",index=False)
    assert len(table)==1440
    result=dict(code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                analysis_code_sha256=record["code_sha256"],input_sha256=HASHES,
                pool_n=len(pool),reference_n=len(ref),disjoint_index_sets=True,
                reconstructed_score_cases=len(table),max_score_difference=float(table.score_max_difference.max()),
                max_selected_bad_pairs=int(table.selected_bad_pairs.max()),
                python=sys.executable,seconds=time.time()-start,method_rules_unchanged=True,submission_created=False)
    (HERE/"背景対照_検証記録.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print({key:value for key,value in result.items() if key!="input_sha256"},flush=True)


if __name__=="__main__":
    run()
