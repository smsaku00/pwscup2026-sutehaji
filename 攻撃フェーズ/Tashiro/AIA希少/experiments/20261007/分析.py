"""AIA希少の研究用集計。提出用CSV・ZIP・tokenを作らない。

実行は作業ルートの .venv/bin/python -B 分析.py --root <作業ルート>。
推定側は公開QI・C・背景のみ。正解は独立した採点側で扱う。
本戦の未知の正解を作ったり推測値を真値として扱ったりしない。
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
import hashlib
import html
import importlib
import itertools
import json
import platform
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
NUM = ["age", "SBP", "TG", "HDL", "ALT"]
FULL = ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG"]
QI = NUM + ["sex", "prefecture"]
SEED = 20261007
CENTERS = np.arange(0., 10.00001, .25)
HASHES: dict[str, str] = {}


def read(path):
    path = Path(path)
    HASHES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path, float_precision="round_trip")


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def raw(df, columns=NUM):
    a = df[columns].to_numpy(float).copy()
    for col in ["TG", "ALT"]:
        if col in columns:
            j = columns.index(col)
            a[:, j] = np.log1p(a[:, j])
    return a


def cost(q, source, bg, n=5, category="sex", metric="iqr"):
    b, x, y = raw(bg), raw(q), raw(source)
    scale = np.maximum((np.quantile(b, .75, axis=0)-np.quantile(b, .25, axis=0)) if metric=="iqr" else b.std(axis=0, ddof=1), 1e-5)
    parts = ((x[:, None, :]-y[None, :, :])/scale)**2
    d = np.sort(parts, axis=2)[:, :, :n].mean(axis=2)
    ms = q.sex.to_numpy()[:, None] != source.sex.to_numpy()[None, :]
    mp = q.prefecture.to_numpy()[:, None] != source.prefecture.to_numpy()[None, :]
    if category in ["sex", "both"]:
        d[ms] = np.inf
        if category=="both":
            d[mp] = np.inf
    elif category=="soft":
        d += ms.astype(float)+mp.astype(float)
    else:
        raise ValueError(category)
    # 条件付き候補がない行だけ、性別条件に戻す。
    empty = ~np.isfinite(d).any(axis=1)
    if empty.any():
        replacement = np.sort(parts[empty], axis=2)[:, :, :n].mean(axis=2)
        replacement[ms[empty]] = np.inf
        if not np.isfinite(replacement).any(axis=1).all():
            raise ValueError("性別一致の候補もない")
        d[empty] = replacement
    return d


def score(pred, truth):
    pred = np.asarray(pred, float)
    if pred.shape != (len(truth),) or not np.isfinite(pred).all() or not (0<=pred).all():
        raise ValueError("不正な分析予測")
    m = truth.label.eq("member").to_numpy()
    c = truth.label.eq("control").to_numpy()
    rare = truth.is_rare.to_numpy(bool)
    assert (m|c).all() and m.sum()==c.sum()==163
    assert (m&rare).sum()==(c&rare).sum()==13
    hit = np.abs(pred-truth.time.to_numpy(float)) <= .5
    hm, hc = int(hit[m].sum()), int(hit[c].sum())
    hrm, hrc = int(hit[m&rare].sum()), int(hit[c&rare].sum())
    pm, pc, pmr, pcr = hm/163, hc/163, hrm/13, hrc/13
    r = float(np.clip((pm-pc)/max(1-pc, 1e-9), 0, 1))
    rr = rare_score(hrm, hrc)
    return dict(r_all=r, r_rare=rr, r_max=max(r,rr), p_m=pm, p_c=pc,
                hit_m=hm, hit_c=hc, hit_m_rare=hrm, hit_c_rare=hrc)


def rare_score(hm, hc):
    pm, pc = hm/13, hc/13
    se = np.sqrt((pm*(1-pm)+pc*(1-pc))/13)
    return float(np.clip((pm-pc-1.64*se)/max(1-pc,1e-9),0,1))


def datasets(root, phase="prelim"):
    if phase=="prelim":
        folder = root/"予備戦/配布データ/PWSCup2026_prelim_attack"
        bg = read(root/"予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv")
        td = next((root/"予備戦/配布データ/結果").glob("正解*"))
        files = sorted(td.glob("aia_truth_*.csv"),key=lambda p:int(p.stem.rsplit("_",1)[1]))
    elif phase=="main":
        folder = root/"本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力"
        bg = read(folder/"A_bg.csv")
        files = sorted(folder.glob("aia_challenge_*.csv"),key=lambda p:int(p.stem.rsplit("_",1)[1]))
    else:
        raise ValueError(phase)
    out=[]
    for f in files:
        k=int(f.stem.rsplit("_",1)[1])
        q=read(folder/f"aia_challenge_{k}.csv").sort_values("challenge_row_id").reset_index(drop=True)
        c=read(folder/f"C_{k}.csv")
        assert len(q)==326 and len(c)==1049 and q.challenge_row_id.is_unique
        t=read(f).sort_values("challenge_row_id").reset_index(drop=True) if phase=="prelim" else None
        if t is not None:
            assert np.array_equal(q.challenge_row_id,t.challenge_row_id)
        out.append(dict(team=k,q=q,c=c,truth=t))
    return out,bg


def self_truth(root, q):
    b=read(next((root/"本戦/配布データ").glob("PWSCup2026_final_team28_*/B_self.csv")))
    t=q.merge(b[QI+["time","is_rare"]], on=QI, how="left",validate="one_to_one")
    assert t.time.notna().sum()==163 and t.is_rare.eq(1).sum()==13
    return t


def self_score(pred, truth):
    m=truth.time.notna().to_numpy()
    r=truth.is_rare.eq(1).to_numpy()
    h=np.abs(np.asarray(pred)-truth.time.to_numpy(float))<=.5
    return dict(n_member=int(m.sum()),hit_member=int(h[m].sum()),hit_member_rare=int(h[r].sum()),
                control_truth_known=False,official_r_rare_computable=False)


def summaries(table, name):
    out=HERE/"集計";out.mkdir(parents=True,exist_ok=True)
    table.to_csv(out/f"{name}_対象別.csv", index=False)
    parts=[]
    metrics=[c for c in ["r_all","r_rare","r_max","p_m","p_c","hit_m_rare","hit_c_rare"] if c in table]
    for group,excluded in [("予備戦20対象",[5]),("弱加工2対象を除く18対象",[1,2,5])]:
        d=table[~table.team.isin(excluded)]
        t=d.groupby("method")[metrics].mean().reset_index()
        pos=d.groupby("method").r_rare.apply(lambda x:int((x>0).sum())).rename("rare_positive_targets").reset_index()
        t=t.merge(pos,on="method");t["group"]=group;t["targets"]=d.team.nunique()
        parts.append(t)
    means=pd.concat(parts,ignore_index=True).sort_values(["group","r_rare"],ascending=[True,False])
    means.to_csv(out/f"{name}_平均.csv",index=False)
    # チームを丸ごと外してから他のチームの希少平均で設定を選ぶ。
    d=table[~table.team.isin([1,2,5])]
    choices=[]
    for k in sorted(d.team.unique()):
        training=d[d.team.ne(k)].groupby("method").r_rare.mean().sort_values(ascending=False,kind="stable")
        chosen=training.index[0]
        row=d[(d.team==k)&(d.method==chosen)].iloc[0]
        choices.append(dict(team=int(k),selected_method=chosen,training_rare=float(training.iloc[0]),
                            r_rare=float(row.r_rare),r_all=float(row.r_all)))
    pd.DataFrame(choices).to_csv(out/f"{name}_対象を外して選択.csv",index=False)
    return means


def record(name, extra):
    save_json(HERE/f"{name}_実行記録.json",dict(python=platform.python_version(), executable=sys.executable,
        seed=SEED,input_sha256=HASHES,submission_created=False,codabench_submitted=False,**extra))


def main(root):
    out=HERE/"集計";out.mkdir(parents=True,exist_ok=True)
    # 14×14すべての希少正解数の組合せを、公式関数と照合する。
    kit=root/"本戦/配布データ/pwscup2026-kit-main_revised0912/starter/src"
    sys.path.insert(0,str(kit))
    official=importlib.import_module("pwscup2026.scoring.attack_score")
    cells=[]
    for m in range(14):
        for c in range(14):
            v=rare_score(m,c)
            ref=official.R_from_rate_deadzone(np.r_[np.ones(m),np.zeros(13-m)],np.r_[np.ones(c),np.zeros(13-c)],1.64)
            assert abs(v-ref)<1e-12
            cells.append(dict(hit_m=m,hit_c=c,r_rare=v,contribution_to_22_target_mean=v/22))
    pd.DataFrame(cells).to_csv(out/"希少13対13_全196組.csv",index=False)
    refdir=root/"本戦/攻撃フェーズ/Tashiro/20261007_AIA全体向上の分析/参照"
    observed=[]
    for rnd in [1,2]:
        f=refdir/f"第{rnd}回_detailed_results.html"
        text=html.unescape(f.read_text())
        HASHES[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
        data=json.loads(re.search(r"<pre>(.*?)</pre>",text,re.S).group(1))
        aia=data["aia"]
        observed.append(dict(round=rnd,**aia,total_member_rare_hits=int(round(aia["p_m_rare"]*22*13)),
                             total_control_rare_hits=int(round(aia["p_c_rare"]*22*13)),
                             sum_target_rare=aia["r_time_rare"]*22))
    save_json(HERE/"公式第1回第2回の確認.json",observed)
    target=.0216
    examples=[]
    current=observed[-1]["r_time_rare"]
    for m,c in [(4,0),(5,0),(6,0),(7,0),(8,3),(9,4),(10,4),(10,3),(10,2),(11,4)]:
        v=rare_score(m,c)
        examples.append(dict(hit_m=m,hit_c=c,r_rare=v,one_new_target_gain=v/22,
                             hypothetical_current_plus_one=current+v/22))
    pd.DataFrame(examples).to_csv(out/"0.0216に届く正解数の例.csv",index=False)
    # 正の寄与が最少の場合だけの数学的な分解。対象ID・人数の確定とはしない。
    positive=[(v["hit_m"],v["hit_c"],v["r_rare"]) for v in cells if v["r_rare"]>0]
    decompositions=[]
    for obs in observed:
        total=obs["sum_target_rare"]
        for n in [1,2,3]:
            combinations=[x for x in itertools.combinations_with_replacement(positive,n) if abs(sum(y[2] for y in x)-total)<1e-10]
            if combinations:
                decompositions.append(dict(round=obs["round"],minimum_positive_targets=n,
                    minimum_only_solutions=combinations,target_ids_identified=False,more_positive_targets_excluded=False))
                break
    save_json(HERE/"平均値の最小寄与分解.json",decompositions)
    validator=importlib.import_module("pwscup2026.kit.validate")
    in_memory=pd.DataFrame(dict(target_k=[9],challenge_row_id=[0],time_hat=[11.]))
    check=validator.validate_aia_submission(in_memory,{})
    assert check.ok
    record("採点確認_11年許容",dict(official_cases_checked=196,current_rare=current,target_rare=target,
        gap=target-current,needed_sum_of_target_improvements=(target-current)*22,
        time_hat_11_passes_value_validation=True,true_time_upper=10,tau_time=.5,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(json.dumps(dict(current_rare=current,target=target,gap=target-current,official_checks=196),ensure_ascii=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--all",action="store_true",help="コードだけを新しい作業フォルダへコピーして全分析を再現する")
    args=parser.parse_args()
    if args.all and (HERE/"集計").exists() and any((HERE/"集計").glob("*.csv")):
        raise SystemExit("旧集計を保存するため、コードだけを新しい作業フォルダへコピーして実行してください。")
    main(args.root.resolve())
    if args.all:
        import subprocess
        stages=[("希少候補の学習.py",[]),("背景除去と希少照合.py",[]),("二重のチーム分離検証.py",[]),
                ("希少に絞った時間窓学習.py",[]),("個人単位の希少時間学習.py",["--nested"]),
                ("他チームCを参照する照合.py",[]),("参照Cの選択基準と偶然性.py",[]),
                ("参照Cの分割安定性と対応置換.py",[]),("希少候補の一対一照合.py",[]),
                ("提案規則の本戦診断.py",[])]
        for name,extra in stages:
            subprocess.run([sys.executable,"-B",str(HERE/name),"--root",str(args.root.resolve()),*extra],check=True)
        subprocess.run([sys.executable,"-B",str(HERE/"レポートの再集計.py")],check=True)
